#include "solver.hpp"
#include "query.h"
#include "continuation.h"

Result solve(double theta, double phi, double alpha) {
    ContinuationParams p;
    p.alpha      = alpha;
    p.h          = 1e-4;
    p.h_shoot    = 1e-4;
    p.delta_step = 0.01;
    p.T_max      = 30.0;
    p.r          = 1e-3;
    p.epsilon_init = 1e-2;
    p.epsilon_fwd  = 1e-3;
#ifdef USE_GPU
    p.N_psi      = 50000;    // 50K seeds; doubles to 100K on pass 1
    p.max_passes = 3;        // dense seed grid needs fewer passes
    p.max_steps  = 5;        // hard cap: never more than 5 continuation steps
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
