#ifdef USE_GPU

#include "manifold.h"
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Signed raw theta difference on the real line.
// The solver passes theta_eff values as unwrapped angles, so the kernel
// must preserve the sign of the theta offset when computing segment
// projections and closest-approach estimates.
__device__ static double d_thetaDelta(double a, double b) {
    return a - b;
}

// Backward dynamics: dz/dt = -forwardDynamics(z, alpha)
__device__ static void d_bwd(double th, double ph, double l1, double l2, double alpha,
                               double* dth, double* dph, double* dl1, double* dl2) {
    double cos_th = cos(th);
    double sin_th = sin(th);
    // forward:
    //   dz[0] =  ph
    //   dz[1] =  sin(th) - alpha*ph - l2*cos²(th)
    //   dz[2] = -(sin(th) + l2*cos(th) + l2²*sin(th)*cos(th))
    //   dz[3] = -(ph + l1 - alpha*l2)
    // backward = negate:
    *dth = -(ph);
    *dph = -(sin_th - alpha * ph - l2 * cos_th * cos_th);
    *dl1 = -( -(sin_th + l2 * cos_th + l2 * l2 * sin_th * cos_th) );
    *dl2 = -( -(ph + l1 - alpha * l2) );
}

__device__ static void d_rk4(double* th, double* ph, double* l1, double* l2,
                               double h, double alpha) {
    double k1t, k1p, k1a, k1b;
    d_bwd(*th, *ph, *l1, *l2, alpha, &k1t, &k1p, &k1a, &k1b);

    double m0 = *th + 0.5*h*k1t, m1 = *ph + 0.5*h*k1p,
           m2 = *l1 + 0.5*h*k1a, m3 = *l2 + 0.5*h*k1b;
    double k2t, k2p, k2a, k2b;
    d_bwd(m0, m1, m2, m3, alpha, &k2t, &k2p, &k2a, &k2b);

    m0 = *th + 0.5*h*k2t; m1 = *ph + 0.5*h*k2p;
    m2 = *l1 + 0.5*h*k2a; m3 = *l2 + 0.5*h*k2b;
    double k3t, k3p, k3a, k3b;
    d_bwd(m0, m1, m2, m3, alpha, &k3t, &k3p, &k3a, &k3b);

    m0 = *th + h*k3t; m1 = *ph + h*k3p;
    m2 = *l1 + h*k3a; m3 = *l2 + h*k3b;
    double k4t, k4p, k4a, k4b;
    d_bwd(m0, m1, m2, m3, alpha, &k4t, &k4p, &k4a, &k4b);

    double c = h / 6.0;
    *th += c * (k1t + 2.0*k2t + 2.0*k3t + k4t);
    *ph += c * (k1p + 2.0*k2p + 2.0*k3p + k4p);
    *l1 += c * (k1a + 2.0*k2a + 2.0*k3a + k4a);
    *l2 += c * (k1b + 2.0*k2b + 2.0*k3b + k4b);
}

// ---------------------------------------------------------------------------
// Kernel: one thread per seed
// ---------------------------------------------------------------------------

__global__ void shootKernel(
        // seeds (SoA)
        const double* __restrict__ s_th,
        const double* __restrict__ s_ph,
        const double* __restrict__ s_l1,
        const double* __restrict__ s_l2,
        int n_seeds,
        // target
        double target_th, double target_ph,
        double alpha, int n_steps, double h, double epsilon,
        // outputs (SoA)
        double* __restrict__ o_min_dist,
        int*    __restrict__ o_flagged,
        double* __restrict__ o_th,
        double* __restrict__ o_ph,
        double* __restrict__ o_l1,
        double* __restrict__ o_l2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_seeds) return;

    double th = s_th[idx], ph = s_ph[idx],
           l1 = s_l1[idx], l2 = s_l2[idx];

    double min_dist = 1e18;
    double best_th = th, best_ph = ph, best_l1 = l1, best_l2 = l2;

    for (int s = 0; s < n_steps; ++s) {
        double prev_th = th, prev_ph = ph, prev_l1 = l1, prev_l2 = l2;

        d_rk4(&th, &ph, &l1, &l2, h, alpha);

        double dtheta = d_thetaDelta(th, target_th);
        double dphi   = ph - target_ph;
        double dist   = sqrt(dtheta * dtheta + dphi * dphi);

        // Sub-step linear interpolation for closest approach
        double a  = d_thetaDelta(prev_th, target_th);
        double b  = prev_ph - target_ph;
        double da = dtheta - a;
        double db = dphi   - b;
        double denom = da*da + db*db;

        double best_dist = dist;
        double bth = th, bph = ph, bl1 = l1, bl2 = l2;

        if (denom > 0.0) {
            double t = -(a*da + b*db) / denom;
            if (t > 0.0 && t < 1.0) {
                double ith = prev_th + t*(th - prev_th);
                double iph = prev_ph + t*(ph - prev_ph);
                double il1 = prev_l1 + t*(l1 - prev_l1);
                double il2 = prev_l2 + t*(l2 - prev_l2);
                double di  = ith - target_th;
                double dj  = iph - target_ph;
                double di2 = sqrt(di*di + dj*dj);
                if (di2 < best_dist) {
                    best_dist = di2;
                    bth = ith; bph = iph; bl1 = il1; bl2 = il2;
                }
            }
        }

        if (best_dist < min_dist) {
            min_dist = best_dist;
            best_th = bth; best_ph = bph; best_l1 = bl1; best_l2 = bl2;
        }

        if (min_dist < epsilon && dist > min_dist * 2.0) break;
    }

    o_min_dist[idx] = min_dist;
    o_flagged[idx]  = (min_dist < epsilon) ? 1 : 0;
    o_th[idx] = best_th; o_ph[idx] = best_ph;
    o_l1[idx] = best_l1; o_l2[idx] = best_l2;
}

// ---------------------------------------------------------------------------
// Host wrapper — same signature as CPU shootAndFlag
// ---------------------------------------------------------------------------

std::vector<FlagResult> shootAndFlagGPU(const std::vector<State>& seeds,
                                         double target_theta, double target_phi,
                                         double alpha, double T_max, double h,
                                         double epsilon) {
    int n = static_cast<int>(seeds.size());
    if (n == 0) return {};

    int n_steps = static_cast<int>(std::round(T_max / h));

    // Flatten seeds to SoA
    std::vector<double> h_th(n), h_ph(n), h_l1(n), h_l2(n);
    for (int i = 0; i < n; ++i) {
        h_th[i] = seeds[i][0]; h_ph[i] = seeds[i][1];
        h_l1[i] = seeds[i][2]; h_l2[i] = seeds[i][3];
    }

    // Allocate device memory
    double *d_th, *d_ph, *d_l1, *d_l2;
    double *d_oth, *d_oph, *d_ol1, *d_ol2, *d_min_dist;
    int    *d_flagged;

    size_t sz = n * sizeof(double);
    int alloc_err = 0;
    alloc_err |= (int)cudaMalloc(&d_th,  sz); alloc_err |= (int)cudaMalloc(&d_ph,  sz);
    alloc_err |= (int)cudaMalloc(&d_l1,  sz); alloc_err |= (int)cudaMalloc(&d_l2,  sz);
    alloc_err |= (int)cudaMalloc(&d_oth, sz); alloc_err |= (int)cudaMalloc(&d_oph, sz);
    alloc_err |= (int)cudaMalloc(&d_ol1, sz); alloc_err |= (int)cudaMalloc(&d_ol2, sz);
    alloc_err |= (int)cudaMalloc(&d_min_dist, sz);
    alloc_err |= (int)cudaMalloc(&d_flagged, n * sizeof(int));
    if (alloc_err != 0) {
        std::fprintf(stderr, "[GPU ERROR] cudaMalloc failed for %d seeds\n", n);
        return {};
    }

    cudaMemcpy(d_th, h_th.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ph, h_ph.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l1, h_l1.data(), sz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l2, h_l2.data(), sz, cudaMemcpyHostToDevice);

    int block = 256;
    int grid  = (n + block - 1) / block;

    shootKernel<<<grid, block>>>(
        d_th, d_ph, d_l1, d_l2, n,
        target_theta, target_phi,
        alpha, n_steps, h, epsilon,
        d_min_dist, d_flagged,
        d_oth, d_oph, d_ol1, d_ol2);

    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[GPU ERROR] kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_th);  cudaFree(d_ph);  cudaFree(d_l1);  cudaFree(d_l2);
        cudaFree(d_oth); cudaFree(d_oph); cudaFree(d_ol1); cudaFree(d_ol2);
        cudaFree(d_min_dist); cudaFree(d_flagged);
        return {};
    }

    // Copy results back
    std::vector<double> h_min(n), h_oth(n), h_oph(n), h_ol1(n), h_ol2(n);
    std::vector<int>    h_flag(n);
    cudaMemcpy(h_min.data(),  d_min_dist, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flag.data(), d_flagged,  n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oth.data(),  d_oth, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_oph.data(),  d_oph, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ol1.data(),  d_ol1, sz, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ol2.data(),  d_ol2, sz, cudaMemcpyDeviceToHost);

    cudaFree(d_th);  cudaFree(d_ph);  cudaFree(d_l1);  cudaFree(d_l2);
    cudaFree(d_oth); cudaFree(d_oph); cudaFree(d_ol1); cudaFree(d_ol2);
    cudaFree(d_min_dist); cudaFree(d_flagged);

    // Build FlagResult vector
    std::vector<FlagResult> results(n);
    for (int i = 0; i < n; ++i) {
        results[i].flagged        = (h_flag[i] != 0);
        results[i].min_dist       = h_min[i];
        results[i].state_at_flag  = {h_oth[i], h_oph[i], h_ol1[i], h_ol2[i]};
        results[i].time_index     = 0;
    }
    return results;
}

#endif // USE_GPU
