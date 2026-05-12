#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE


#include "solver.hpp"

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>


// PMP state container for (θ, φ, λ₁, λ₂, cost)
struct PMPState {
    double theta;
    double phi;
    double lambda1;
    double lambda2;
    double cost;
};

// Stable manifold basis storage used to seed patch points.
struct StableBasis {
    double entries[8];
};

// Candidate trajectory record produced by the GPU sampling stage.
struct PatchCandidate {
    double coeff_a;
    double coeff_b;
    double theta;
    double phi;
    double lambda1;
    double lambda2;
    double cost;
    double residual2;
    int ok;
};

static void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::exit(2);
    }
}

#define GPU_CHECK(x) checkCudaError((x), __FILE__, __LINE__)


// Hamiltonian right-hand side for a PMP state; available on host and device.
__host__ __device__
static PMPState compute_rhs(const PMPState& state, double alpha) {
    const double st = sin(state.theta);
    const double ct = cos(state.theta);
    const double ct2 = ct * ct;
    const double l22 = state.lambda2 * state.lambda2;

    PMPState rhs;
    rhs.theta   = state.phi;
    rhs.phi     = st - alpha * state.phi - state.lambda2 * ct2;
    rhs.lambda1 = -st - state.lambda2 * ct - l22 * ct * st;
    rhs.lambda2 = -state.phi - state.lambda1 + alpha * state.lambda2;
    rhs.cost    = 1.0 - ct + 0.5 * state.phi * state.phi + 0.5 * l22 * ct2;

    return rhs;
}

//cpu/gpu helper, add a scaled version of k to y, used in the RK4 integrator below
__host__ __device__
static PMPState addScaledState(const PMPState& base, const PMPState& increment, double h) {
    PMPState output;
    output.theta   = base.theta   + h * increment.theta;
    output.phi     = base.phi     + h * increment.phi;
    output.lambda1 = base.lambda1 + h * increment.lambda1;
    output.lambda2 = base.lambda2 + h * increment.lambda2;
    output.cost    = base.cost + h * increment.cost;
    return output;
}

// Single RK4 step for the PMP dynamical system.
__host__ __device__
static PMPState rk4StepState(const PMPState& state, double alpha, double step_size) {
    const PMPState k1 = compute_rhs(state, alpha);
    const PMPState k2 = compute_rhs(addScaledState(state, k1, 0.5 * step_size), alpha);
    const PMPState k3 = compute_rhs(addScaledState(state, k2, 0.5 * step_size), alpha);
    const PMPState k4 = compute_rhs(addScaledState(state, k3, step_size), alpha);

    PMPState output;
    const double h6 = step_size / 6.0;

    output.theta   = state.theta   + h6 * (k1.theta   + 2.0 * k2.theta   + 2.0 * k3.theta   + k4.theta);
    output.phi     = state.phi     + h6 * (k1.phi     + 2.0 * k2.phi     + 2.0 * k3.phi     + k4.phi);
    output.lambda1 = state.lambda1 + h6 * (k1.lambda1 + 2.0 * k2.lambda1 + 2.0 * k3.lambda1 + k4.lambda1);
    output.lambda2 = state.lambda2 + h6 * (k1.lambda2 + 2.0 * k2.lambda2 + 2.0 * k3.lambda2 + k4.lambda2);
    output.cost    = state.cost    + h6 * (k1.cost    + 2.0 * k2.cost    + 2.0 * k3.cost    + k4.cost);

    return output;
}

__host__ __device__
static double square(double x) {
    return x * x;
}

// GPU kernel evaluating candidate patch points in parallel.
__global__
static void launchPatchKernel(PatchCandidate* out,
                              int grid_n,
                              double radius,
                              int steps,
                              double dt,
                              double alpha,
                              StableBasis basis,
                              double target_th,
                              double target_ph) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //this line means that each thread will have a unique idx, and the kernel will be launched with enough threads to cover all points in the grid. 
    //So each thread will be responsible for simulating one trajectory starting from a point in the patch.
    const int total = grid_n * grid_n;
    //total number of candidate in the grid, we will have it 49.

    if (idx >= total) {
        return;
    }
    //extra gpu threads will do nothing

    //convert idx to  to grid coordinate (i,j)
    const int i = idx / grid_n;
    const int j = idx - i * grid_n;

    //map grid coordinate to patch coordinate (xi, xj) in [-1, 1] x [-1, 1]
    const double xi = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(grid_n - 1);
    const double xj = -1.0 + 2.0 * static_cast<double>(j) / static_cast<double>(grid_n - 1);

    //scale patch coordinate by radius to get initial condition on the stable manifold
    const double a = radius * xi;
    const double b = radius * xj;

    // Build the trajectory seed in local patch coordinates.
    PMPState y;
    y.theta   = basis.entries[0] * a + basis.entries[4] * b;
    y.phi     = basis.entries[1] * a + basis.entries[5] * b;
    y.lambda1 = basis.entries[2] * a + basis.entries[6] * b;
    y.lambda2 = basis.entries[3] * a + basis.entries[7] * b;
    y.cost    = 0.0;

    int ok = 1;

    for (int k = 0; k < steps; ++k) {
        y = rk4StepState(y, alpha, dt);

        if (!isfinite(y.theta) || !isfinite(y.phi) ||
            !isfinite(y.lambda1) || !isfinite(y.lambda2) ||
            fabs(y.theta) > 1.0e8 || fabs(y.phi) > 1.0e8 ||
            fabs(y.lambda1) > 1.0e8 || fabs(y.lambda2) > 1.0e8) {
            ok = 0;
            break;
        }
    }
    //store the result in the output array, we will copy this back to the CPU and sort by distance to target to get good initial conditions for the Newton refinement step.
    PatchCandidate c;
    c.coeff_a    = a;
    c.coeff_b    = b;
    c.theta      = y.theta;
    c.phi        = y.phi;
    c.lambda1    = y.lambda1;
    c.lambda2    = y.lambda2;
    c.cost       = -y.cost;
    c.ok         = ok;
    if (ok) {
        c.residual2 = square(y.theta - target_th) + square(y.phi - target_ph);
    } else {
        c.residual2 = 1.0e300;
    }

    out[idx] = c;
}

// Build the two most stable eigenvectors of the linearized PMP system.
static Eigen::Matrix<double, 4, 2> computeStableSubspace(double alpha) {
    Eigen::Matrix4d A;

    A << 0.0,    1.0,    0.0,    0.0,
         1.0,   -alpha,  0.0,   -1.0,
        -1.0,    0.0,    0.0,   -1.0,
         0.0,   -1.0,   -1.0,    alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(A);

    std::vector<std::pair<double, int>> idx;

    for (int i = 0; i < 4; ++i) {
        idx.emplace_back(es.eigenvalues()(i).real(), i);
    }

    std::sort(idx.begin(), idx.end());

    Eigen::Matrix<std::complex<double>, 4, 2> Vc;
    Vc.col(0) = es.eigenvectors().col(idx[0].second);
    Vc.col(1) = es.eigenvectors().col(idx[1].second);

    Eigen::Matrix<double, 4, 2> Vs = Vc.real();

    for (int j = 0; j < 2; ++j) {
        const double n = Vs.col(j).norm();
        if (n > 0.0) {
            Vs.col(j) /= n;
        }
    }
//so we got our stable subsapce basis 
    return Vs;
}

//CPU function, do the same kind of integration as the GPU kernel, but for one candidate.
//Very important to have this on the CPU so we can do the Newton refinement step that uses sequential integrations.
static Eigen::Matrix2d computeStableGain(double alpha) {
    const Eigen::Matrix<double, 4, 2> Vs = computeStableSubspace(alpha);

    Eigen::Matrix2d top = Vs.topRows<2>();
    Eigen::Matrix2d bot = Vs.bottomRows<2>();

    return bot * top.inverse();
}

static StableBasis assembleStableBasis(double alpha) {
    const Eigen::Matrix<double, 4, 2> Vs = computeStableSubspace(alpha);

    StableBasis basis;
    for (int r = 0; r < 4; ++r) {
        basis.entries[r]     = Vs(r, 0);
        basis.entries[4 + r] = Vs(r, 1);
    }
    return basis;
}

static PMPState propagatePatch(const StableBasis& basis,
                                 double a,
                                 double b,
                                 double alpha,
                                 double T,
                                 int steps) {
    PMPState y;
    y.theta   = basis.entries[0] * a + basis.entries[4] * b;
    y.phi     = basis.entries[1] * a + basis.entries[5] * b;
    y.lambda1 = basis.entries[2] * a + basis.entries[6] * b;
    y.lambda2 = basis.entries[3] * a + basis.entries[7] * b;
    y.cost    = 0.0;

    const double dt = -T / static_cast<double>(steps);

    for (int k = 0; k < steps; ++k) {
        y = rk4StepState(y, alpha, dt);

        if (!std::isfinite(y.theta) || !std::isfinite(y.phi) ||
            !std::isfinite(y.lambda1) || !std::isfinite(y.lambda2) ||
            std::fabs(y.theta) > 1.0e9 || std::fabs(y.phi) > 1.0e9 ||
            std::fabs(y.lambda1) > 1.0e9 || std::fabs(y.lambda2) > 1.0e9) {
            y.theta = 1.0e100;
            y.phi = 1.0e100;
            y.lambda1 = 1.0e100;
            y.lambda2 = 1.0e100;
            y.cost = -1.0e100;
            return y;
        }
    }

    return y;
}

//newton's refinement 

static PatchCandidate refinePatchNewton(const StableBasis& basis,
                                          double a0,
                                          double b0,
                                          double target_th,
                                          double target_ph,
                                          double alpha) {
    const double T = 20.0;
    const int steps = 2600;
    const int max_iter = 11;

    double a = a0;
    double b = b0;
//start from a gpu seed
    PMPState y = propagatePatch(basis, a, b, alpha, T, steps);
    double best_dist2 = square(y.theta - target_th) + square(y.phi - target_ph);

    for (int it = 0; it < max_iter; ++it) {
        if (!std::isfinite(best_dist2) || best_dist2 < 1.0e-16) {
            break;
        }

        const double ea = std::max(1.0e-12, 1.0e-5 * std::max(std::fabs(a), 1.0e-8));
        const double eb = std::max(1.0e-12, 1.0e-5 * std::max(std::fabs(b), 1.0e-8));
//cpu integration to compute the Jacobian, we will do two additional integrations with small perturbations in a and b to compute finite difference approximations of the Jacobian of the shooting function. 
//This is the main reason we need to do this on the CPU, since we need to do these sequential integrations and they are not very parallelizable.
//AI translated this from my python code, so it might look a bit weird since I was using a lot of numpy broadcasting and stuff in the python code, but here we have to write everything out explicitly.
        const PMPState ya = propagatePatch(basis, a + ea, b, alpha, T, steps);
        const PMPState yb = propagatePatch(basis, a, b + eb, alpha, T, steps);

        if (!std::isfinite(ya.theta) || !std::isfinite(yb.theta)) {
            break;
        }

        const double r1 = y.theta - target_th;
        const double r2 = y.phi - target_ph;

        const double J11 = (ya.theta - y.theta) / ea;
        const double J21 = (ya.phi - y.phi) / ea;
        const double J12 = (yb.theta - y.theta) / eb;
        const double J22 = (yb.phi - y.phi) / eb;

        const double det = J11 * J22 - J12 * J21;

        if (!std::isfinite(det) || std::fabs(det) < 1.0e-14) {
            break;
        }

        const double da = (-r1 * J22 + J12 * r2) / det;
        const double db = ( J21 * r1 - J11 * r2) / det;

        bool accepted = false;
        double scale = 1.0;

        for (int ls = 0; ls < 10; ++ls) {
            const double na = a + scale * da;
            const double nb = b + scale * db;

const PMPState trial = propagatePatch(basis, na, nb, alpha, T, steps);
        const double d2 = square(trial.theta - target_th) + square(trial.phi - target_ph);

            if (std::isfinite(d2) && d2 < best_dist2) {
                a = na;
                b = nb;
                y = trial;
                best_dist2 = d2;
                accepted = true;
                break;
            }

            scale *= 0.5;
        }

        if (!accepted) {
            break;
        }
    }

    PatchCandidate c;
    c.coeff_a    = a;
    c.coeff_b    = b;
    c.theta      = y.theta;
    c.phi        = y.phi;
    c.lambda1    = y.lambda1;
    c.lambda2    = y.lambda2;
    c.cost       = -y.cost;
    c.residual2  = best_dist2;
    c.ok         = std::isfinite(best_dist2) ? 1 : 0;

    return c;
}

//most imp part
//this function itself runs on CPU. It manages GPU work.
static std::vector<PatchCandidate> runPatchSearchGpu(double theta,
                                                         double phi,
                                                         double alpha,
                                                         const StableBasis& basis) {
    const int grid_n = 64; 
    //we will search a 49x49 grid of initial conditions on the stable manifold patch, this number is somewhat arbitrary but it seems to give good coverage of the patch without being too slow.
    //also set backwards integration time and step size for coarse GPU search 
    const int n = grid_n * grid_n;
    const double T = 20.0;
    const int steps = 1000;
    const double dt = -T / static_cast<double>(steps);

    //search many patch radii
    const std::vector<double> radii = {
        1.0e-10, 3.0e-10,
        1.0e-9,  3.0e-9,
        1.0e-8,  3.0e-8,
        1.0e-7,  3.0e-7,
        1.0e-6,  3.0e-6,
        1.0e-5,  3.0e-5,
        1.0e-4,  3.0e-4,
        1.0e-3
    };

    //allocate output memory on GPU

    PatchCandidate* d_out = nullptr;
    GPU_CHECK(cudaMalloc(&d_out, n * sizeof(PatchCandidate)));

    std::vector<PatchCandidate> h(n);
    std::vector<PatchCandidate> all;
    all.reserve(n * radii.size());

    //cpu vectors to store results, we will copy the results from the GPU to these vectors and then sort them by distance to target to get good initial conditions for the Newton refinement step.
    const int block = 256;
    const int blocks = (n + block - 1) / block;
    
//This launches the GPU kernel. This is where the GPU actually starts doing the coarse candidate integrations.
    for (double radius : radii) {
        launchPatchKernel<<<blocks, block>>>(d_out,
                                               grid_n,
                                               radius,
                                               steps,
                                               dt,
                                               alpha,
                                               basis,
                                               theta,
                                               phi);
        
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());

        GPU_CHECK(cudaMemcpy(h.data(),
                             d_out,
                             n * sizeof(PatchCandidate),
                             cudaMemcpyDeviceToHost));
//Copy candidate results from GPU memory back to CPU memory.
        for (const PatchCandidate& c : h) {
            if (c.ok && std::isfinite(c.residual2)) {
                all.push_back(c);
            }
        }
    }

    GPU_CHECK(cudaFree(d_out));
//CPU sorts candidates by distance.So the GPU finds many crude candidates; the CPU decides which ones are good.
    std::sort(all.begin(), all.end(),
              [](const PatchCandidate& x, const PatchCandidate& y) {
                  return x.residual2 < y.residual2;
              });

    return all;
}

//kinda driver function.

static PatchCandidate searchWellCandidates(const StableBasis& basis,
                                              double theta_eff,
                                              double phi,
                                              double alpha,
                                              int max_trials) {
    std::vector<PatchCandidate> seeds = runPatchSearchGpu(theta_eff, phi, alpha, basis);

    PatchCandidate best;
    best.coeff_a   = 0.0;
    best.coeff_b   = 0.0;
    best.theta     = 0.0;
    best.phi       = 0.0;
    best.lambda1   = 0.0;
    best.lambda2   = 0.0;
    best.cost      = 1.0e300;
    best.residual2 = 1.0e300;
    best.ok        = 0;

    const int trials = std::min<int>(max_trials, static_cast<int>(seeds.size()));
    //CPU refines the best few GPU seeds.
    for (int i = 0; i < trials; ++i) {
        PatchCandidate c = refinePatchNewton(basis,
                                            seeds[i].coeff_a,
                                            seeds[i].coeff_b,
                                            theta_eff,
                                            phi,
                                            alpha);

        if (c.ok && std::isfinite(c.residual2) && c.residual2 < best.residual2) {
            best = c;
        }
    }

    return best;
}

Result solve(double theta, double phi, double alpha) {
    if (std::fabs(theta) < 1.0e-14 && std::fabs(phi) < 1.0e-14) {
        return {0.0, 0.0, 0.0};
    }

    const double TWO_PI = 2.0 * M_PI;
    const int k_round = static_cast<int>(std::lround(theta / TWO_PI));
    int well_indices[] = {k_round, 0, k_round - 1, k_round + 1, k_round - 2, k_round + 2};
    
    std::vector<int> unique_wells;
    for (int w : well_indices) {
        bool found = false;
        for (int existing : unique_wells) {
            if (existing == w) { found = true; break; }
        }
        if (!found) unique_wells.push_back(w);
    }

    const StableBasis basis = assembleStableBasis(alpha);
    PatchCandidate best_global;
    best_global.cost      = 1.0e300;
    best_global.residual2 = 1.0e300;
    best_global.ok        = 0;
    const double DIST2_OK = 1.0e-9;

    for (int w : unique_wells) {
        const double theta_shifted = theta - TWO_PI * static_cast<double>(w);
        PatchCandidate candidate = searchWellCandidates(basis, theta_shifted, phi, alpha, 6);

        if (candidate.ok && std::isfinite(candidate.residual2) && 
            candidate.residual2 < DIST2_OK && std::isfinite(candidate.cost)) {
            if (candidate.cost < best_global.cost) {
                best_global = candidate;
            }
        }
    }

    if (best_global.ok) {
        return {best_global.lambda1, best_global.lambda2, best_global.cost};
    }

    const Eigen::Matrix2d K = computeStableGain(alpha);
    Eigen::Vector2d state;
    state << theta, phi;
    Eigen::Vector2d costate = K * state;
    const double value = 0.5 * (theta * costate(0) + phi * costate(1));

    return {costate(0), costate(1), value};
}

std::vector<Result> solve_many(
    const std::vector<double>& theta,
    const std::vector<double>& phi,
    const std::vector<double>& alpha
) {
    int N = static_cast<int>(theta.size());
    std::vector<Result> results(N);

    for (int i = 0; i < N; ++i) {
        results[i] = solve(theta[i], phi[i], alpha[i]);
    }

    return results;
}