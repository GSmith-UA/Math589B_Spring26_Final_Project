// continuation.h — Walking/continuation algorithm
// Read DESIGN.md §4 before implementing.

#pragma once
#include "dynamics.h"
#include "manifold.h"
#include <optional>
#include <vector>
#include <climits>

struct ContinuationParams {
    double delta_step   = 0.01;    // max arc length per continuation step (theta units)
    int    N_psi        = 1000;    // seed circle samples per step
    double r            = 1e-3;    // seed circle radius
    double T_max        = 20.0;    // backward integration horizon
    double h            = 1e-2;    // RK4 step size (forward residual + default for shooting)
    double h_shoot      = 0.0;     // backward-shooting step size; 0 means use h
    double epsilon_init = 0.05;    // initial flag tolerance
    double epsilon_fwd  = 1e-4;    // forward integration acceptance criterion
    int    refine_iters = 3;       // number of refinement passes
    int    max_passes   = 10;      // max arc-refinement passes per continuation step
    int    max_steps    = INT_MAX; // hard cap on continuation steps (0 = no cap)
    double epsilon_shrink = 0.5;   // tolerance shrink factor per refinement
    double alpha        = 0.1;     // damping
    int    N_psi_refine = 0;       // seeds per well refinement pass (0 = disabled)
};

struct CostateEstimate {
    double lambda1;
    double lambda2;
    double forward_residual;  // min |z(t)| after forward integration
    bool   accepted;
    double best_psi = 0.0;   // best ψ found; carried as psi_hint to next step
};

// Solve for costate at a single (theta, phi) given a warm-start costate
// from the previous continuation step.
// warm_start:  accepted (lambda1, lambda2) from step n-1 (pass {0,0} for step 1)
// cold_start:  true  → full 2π sweep; psi_center is reset internally
//              false → π/2 arc centred on psi_center
// psi_center:  best ψ from previous step; updated in-place each call.
CostateEstimate solveAtPoint(double theta, double phi,
                              std::array<double, 2> warm_start,
                              const ContinuationParams& params,
                              bool cold_start,
                              double& psi_center);

// Per-step stiffness record: Jacobian eigenvalue ratio at each accepted knot.
struct StiffnessRecord {
    double theta_q, phi_q;   // target of this walk
    int    step_n;           // continuation step index (1..N)
    double theta_n, phi_n;   // knot position
    double lambda1, lambda2; // accepted costate at knot
    double stiffness_ratio;  // max|Re(μ)| / min|Re(μ)| of Jacobian at knot
};

// Full continuation walk from (0,0) to (theta_q, phi_q).
// Returns the costate estimate at the target, or nullopt if failed.
// Uses shortest arc on S¹ for theta path (DESIGN.md §2).
// If stiffness_out is non-null, appends one record per accepted step.
std::optional<CostateEstimate> continuationWalk(
    double theta_q, double phi_q,
    const ContinuationParams& params,
    std::vector<StiffnessRecord>* stiffness_out = nullptr);
