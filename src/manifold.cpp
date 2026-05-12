#include "manifold.h"
#include "dynamics.h"
#include "rk4.h"
#include "utils.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

EigenpairResult computeStableEigenpairs(const Matrix4& jacobian) {
    Eigen::Matrix4d J;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            J(i, j) = jacobian[i][j];

    Eigen::EigenSolver<Eigen::Matrix4d> es(J);

    // Collect (Re(eigenvalue), column index) for stable roots
    std::vector<std::pair<double, int>> stable;
    for (int i = 0; i < 4; ++i) {
        double re = es.eigenvalues()[i].real();
        if (re < -1e-10)
            stable.push_back({re, i});
    }

    if (static_cast<int>(stable.size()) < 2)
        throw std::runtime_error(
            "computeStableEigenpairs: fewer than 2 stable eigenvalues found");

    // Sort so index 0 = most negative eigenvalue
    std::sort(stable.begin(), stable.end());

    EigenpairResult result;
    for (int k = 0; k < 2; ++k) {
        int idx = stable[k].second;
        result.eigenvalues[k] = es.eigenvalues()[idx].real();
        auto col = es.eigenvectors().col(idx);
        for (int i = 0; i < 4; ++i)
            result.eigenvectors[k][i] = col[i].real();
    }
    return result;
}

std::vector<State> generateSeedPoints(const EigenpairResult& eigenpairs,
                                       int N_psi, double r, double T_max) {
    double mu1 = eigenpairs.eigenvalues[0];
    double mu2 = eigenpairs.eigenvalues[1];
    const State& v1 = eigenpairs.eigenvectors[0];
    const State& v2 = eigenpairs.eigenvectors[1];

    // Seed formula: z_seed(psi) = Vs * exp(T_max * diag(mu)) * r * [cos(psi); sin(psi)]
    // Pre-scale each eigenvector column by r * exp(T_max * mu_k)
    double scale1 = r * std::exp(10.0 * mu1);
    double scale2 = r * std::exp(10.0 * mu2);

    std::vector<State> seeds(N_psi);
    for (int i = 0; i < N_psi; ++i) {
        double psi = 2.0 * M_PI * i / N_psi;
        double cp  = std::cos(psi);
        double sp  = std::sin(psi);
        for (int j = 0; j < 4; ++j)
            seeds[i][j] = v1[j] * scale1 * cp + v2[j] * scale2 * sp;
    }
    return seeds;
}

FlagResult shootAndCheckOne(const State& seed,
                             double target_theta, double target_phi,
                             double alpha, double T_max, double h,
                             double epsilon) {
    FlagResult result = {false, seed, 1e18, 0};

    auto f = [alpha](const State& z) -> State {
        return backwardDynamics(z, alpha);
    };

    State z = seed;
    int n_steps = static_cast<int>(std::round(T_max / h));

    for (int step = 0; step < n_steps; ++step) {
        State z_prev = z;
        z = rk4Step(z, h, f);

        double dtheta = wrappedThetaDistance(z[0], target_theta);
        double dphi   = z[1] - target_phi;
        double dist   = std::sqrt(dtheta * dtheta + dphi * dphi);

        // Sub-step interpolation: find closest point on segment z_prev -> z
        double a  = wrappedThetaDistance(z_prev[0], target_theta);
        double b  = z_prev[1] - target_phi;
        double da = dtheta - a;
        double db = dphi   - b;
        double denom = da * da + db * db;
        State  best_z    = z;
        double best_dist = dist;
        if (denom > 0.0) {
            double t = -(a * da + b * db) / denom;
            if (t > 0.0 && t < 1.0) {
                State z_interp;
                for (int j = 0; j < 4; ++j)
                    z_interp[j] = z_prev[j] + t * (z[j] - z_prev[j]);
                double di = wrappedThetaDistance(z_interp[0], target_theta);
                double dj = z_interp[1] - target_phi;
                double dist_interp = std::sqrt(di * di + dj * dj);
                if (dist_interp < best_dist) {
                    best_dist = dist_interp;
                    best_z    = z_interp;
                }
            }
        }

        if (best_dist < result.min_dist) {
            result.min_dist      = best_dist;
            result.state_at_flag = best_z;
            result.time_index    = step;
        }

        // Early exit: stop once we've passed the closest approach and are diverging.
        // Requires both (a) a genuine close approach and (b) clear post-minimum divergence.
        // Prevents spurious late-time crossings (n_flagged=250 collapse) with large T_max
        // without prematurely ejecting seeds that are still converging toward the target.
        if (result.min_dist < epsilon && dist > result.min_dist * 2.0) break;
    }

    result.flagged = (result.min_dist < epsilon);
    return result;
}

std::vector<State> generateSeedPointsArc(const EigenpairResult& eigenpairs,
                                          int N_psi, double r,
                                          double psi_center, double arc,
                                          const State& center) {
    double scale1 = r * std::exp(10.0 * eigenpairs.eigenvalues[0]);
    double scale2 = r * std::exp(10.0 * eigenpairs.eigenvalues[1]);
    const State& v1 = eigenpairs.eigenvectors[0];
    const State& v2 = eigenpairs.eigenvectors[1];

    std::vector<State> seeds(N_psi);
    for (int i = 0; i < N_psi; ++i) {
        double psi = psi_center + arc * (i - N_psi / 2.0) / N_psi;
        double cp  = std::cos(psi);
        double sp  = std::sin(psi);
        for (int j = 0; j < 4; ++j)
            seeds[i][j] = center[j] + v1[j] * scale1 * cp + v2[j] * scale2 * sp;
    }
    return seeds;
}

std::vector<State> generateSeedPointsGrid(const EigenpairResult& eigenpairs,
                                           int grid_n, double radius,
                                           const State& center) {
    const State& v1 = eigenpairs.eigenvectors[0];
    const State& v2 = eigenpairs.eigenvectors[1];

    std::vector<State> seeds;
    seeds.reserve(grid_n * grid_n);

    for (int i = 0; i < grid_n; ++i) {
        double xi = (grid_n > 1) ? -1.0 + 2.0 * i / (grid_n - 1) : 0.0;
        double a  = radius * xi;
        for (int j = 0; j < grid_n; ++j) {
            double xj = (grid_n > 1) ? -1.0 + 2.0 * j / (grid_n - 1) : 0.0;
            double b  = radius * xj;
            State s;
            for (int k = 0; k < 4; ++k)
                s[k] = center[k] + v1[k] * a + v2[k] * b;
            seeds.push_back(s);
        }
    }
    return seeds;
}

std::vector<FlagResult> shootAndFlag(const std::vector<State>& seeds,
                                      double target_theta, double target_phi,
                                      double alpha, double T_max, double h,
                                      double epsilon) {
    std::vector<FlagResult> results;
    results.reserve(seeds.size());
    for (const auto& seed : seeds)
        results.push_back(shootAndCheckOne(seed, target_theta, target_phi,
                                           alpha, T_max, h, epsilon));
    return results;
}

double computeStiffnessRatio(const Matrix4& J) {
    Eigen::Matrix4d M;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            M(i, j) = J[i][j];
    Eigen::EigenSolver<Eigen::Matrix4d> es(M, false);
    double max_re = 0.0, min_re = 1e18;
    for (int i = 0; i < 4; ++i) {
        double re = std::abs(es.eigenvalues()[i].real());
        if (re > 1e-10) {
            if (re > max_re) max_re = re;
            if (re < min_re) min_re = re;
        }
    }
    return (min_re < 1e17) ? max_re / min_re : 1.0;
}

std::vector<std::array<double, 2>> extractCostatesCandidates(
    const std::vector<FlagResult>& flags,
    const std::vector<State>&      /*seeds*/) {
    std::vector<std::array<double, 2>> candidates;
    for (const auto& fr : flags) {
        if (fr.flagged)
            candidates.push_back({fr.state_at_flag[2], fr.state_at_flag[3]});
    }
    return candidates;
}
