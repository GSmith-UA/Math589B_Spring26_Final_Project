// manifold.h — Stable manifold seeding, shooting, and flagging
// Read DESIGN.md §3 and §4 before implementing.
//
// PARALLELISM NOTE (future CUDA):
//   The shootAndFlag function's inner loop over psi samples is the
//   primary parallelism target. Each trajectory is independent.
//   Data layout uses SoA (structure of arrays) for coalesced GPU access.
//   See DESIGN.md §7 "Data Layout".

#pragma once
#include "dynamics.h"
#include <vector>
#include <complex>

struct EigenpairResult {
    std::array<double, 2> eigenvalues;          // stable (negative) eigenvalues
    std::array<State, 2>  eigenvectors;          // corresponding eigenvectors (real parts)
};

// Compute the two stable eigenpairs of a 4x4 Jacobian matrix.
// Stable means Re(eigenvalue) < 0.
// Throws if fewer than 2 stable eigenvalues found.
EigenpairResult computeStableEigenpairs(const Matrix4& jacobian);

// Generate seed points on the stable eigenplane circle.
// z_seed(psi) = Vs * exp(T_max * diag(mu)) * r * [cos(psi), sin(psi)]
// Returns N_psi seed states.
std::vector<State> generateSeedPoints(const EigenpairResult& eigenpairs,
                                       int N_psi, double r, double T_max);

// Generate seeds centered at psi_center spanning the given arc.
// psi_i = psi_center + arc * (i - N_psi/2) / N_psi
// Used for arc-refinement passes in the continuation solver.
std::vector<State> generateSeedPointsArc(const EigenpairResult& eigenpairs,
                                          int N_psi, double r,
                                          double psi_center, double arc,
                                          const State& center);

// Generate seeds on a 2D grid_n×grid_n grid in the stable eigenplane.
// a ∈ [-radius, radius], b ∈ [-radius, radius].
// z_seed(i,j) = center + v1*a + v2*b  (no exp pre-scaling; caller sweeps radii).
std::vector<State> generateSeedPointsGrid(const EigenpairResult& eigenpairs,
                                           int grid_n, double radius,
                                           const State& center);

struct FlagResult {
    bool   flagged;
    State  state_at_flag;   // z value when proximity was triggered
    double min_dist;        // closest approach distance to target
    int    time_index;      // index in trajectory where flag was set
};

// Shoot one trajectory backward from seed and check proximity to target.
// target_theta, target_phi: the (θ,φ) we are trying to reach
// epsilon: proximity threshold (wrapped distance)
// Returns flag result for this trajectory.
FlagResult shootAndCheckOne(const State& seed,
                             double target_theta, double target_phi,
                             double alpha, double T_max, double h,
                             double epsilon);

// Shoot all N_psi trajectories and return flags.
// PARALLELISM NOTE: This loop → one CUDA thread per trajectory in GPU version.
std::vector<FlagResult> shootAndFlag(const std::vector<State>& seeds,
                                      double target_theta, double target_phi,
                                      double alpha, double T_max, double h,
                                      double epsilon);

// Extract costate estimates from flagged trajectories.
// Returns (lambda1, lambda2) candidates for refinement.
std::vector<std::array<double, 2>> extractCostatesCandidates(
    const std::vector<FlagResult>& flags,
    const std::vector<State>& seeds);

// Stiffness ratio of a Jacobian: max|Re(μᵢ)| / min|Re(μᵢ)|
// over all eigenvalues with nonzero real part.
double computeStiffnessRatio(const Matrix4& J);
