// query.h — Top-level query interface
// Read DESIGN.md §5 before implementing.
//
// This is the public API of the stable manifold solver.
// Handles theta-shift copies and best-result selection.
//
// PARALLELISM NOTE (future CUDA):
//   The theta-shift copies (theta - 2pi, theta, theta + 2pi) run
//   sequentially here. In the GPU version they become parallel streams.
//   Keep them as independent function calls with no shared state
//   to make this transition easy.

#pragma once
#include "continuation.h"

struct QueryResult {
    double lambda1;
    double lambda2;
    double forward_residual;
    double theta_shift_used;   // which shift gave the best result
    double cost;               // optimal cost-to-go (integrated to closest approach)
    bool   success;
};

// Query the stable manifold for optimal costate at (theta_q, phi_q).
// Runs theta-shift copies {theta-2pi, theta, theta+2pi} sequentially.
// Returns the best result (lowest forward_residual that is accepted).
//
// If isSmallAngle(theta_q, phi_q), seeds from LQR estimate.
QueryResult queryManifold(double theta_q, double phi_q,
                           const ContinuationParams& params);

// Validate a costate estimate by forward integration.
// Returns |z(T_max)| — the forward residual.
double validateCostate(double theta, double phi,
                        double lambda1, double lambda2,
                        const ContinuationParams& params);
