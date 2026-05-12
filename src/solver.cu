#include "solver.hpp"
#include "query.h"
#include "continuation.h"
#include <cmath>

Result solve(double theta, double phi, double alpha) {
    // wrap theta to (-π, π] — dynamics and cost are 2π-periodic
    theta = std::atan2(std::sin(theta), std::cos(theta));

    ContinuationParams p;
    p.alpha      = alpha;
    p.h          = 1e-4;
    p.h_shoot    = 2.5e-4;
    p.delta_step = 0.01;
    p.T_max      = 15.0;
    p.r          = 1e-3;
    p.epsilon_init = 1e-2;
    p.epsilon_fwd  = 1e-3;
#ifdef USE_GPU
    p.N_psi      = 850000;   // 500K seeds per pass; pass-0: full 2π, pass-1: π/2 refinement
    p.max_passes = 2;
    p.max_steps  = 1;        // direct cold start at target
#else
    p.N_psi      = 500;
    p.max_passes = 10;
#endif

    QueryResult qr = queryManifold(theta, phi, p);

    Result r;
    if (qr.success) {
        r.l1   = qr.lambda1;
        r.l2   = qr.lambda2;
        r.cost = qr.cost;
    } else {
        r.l1   = 0.0;
        r.l2   = 0.0;
        r.cost = 1e100;
    }
    return r;
}
