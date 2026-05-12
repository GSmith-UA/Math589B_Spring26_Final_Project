#include "solver.hpp"
#include "query.h"
#include "continuation.h"
#include <cmath>

Result solve(double theta, double phi, double alpha) {
    if (std::fabs(theta) < 1e-14 && std::fabs(phi) < 1e-14)
        return {0.0, 0.0, 0.0};

    ContinuationParams p;
    p.alpha        = alpha;
    p.h            = 1e-4;
    p.h_shoot      = 2.0e-4;
    p.delta_step   = 0.01;
    p.T_max        = 25.0;
    p.r            = 1e-3;
    p.epsilon_init = 1e-2;
    p.epsilon_fwd  = 1e-3;
#ifdef USE_GPU
    p.N_psi        = 0;        // unused: grid sweep replaces arc sweep
    p.N_psi_refine = 1;        // nonzero → triggers GPU cold-start grid path
    p.max_passes   = 1;
    p.max_steps    = 1;        // direct cold start at target
#else
    p.N_psi      = 500;
    p.max_passes = 10;
#endif

    // Try theta shifted by 2π*k for several k values; take minimum-cost
    // converged solution.  k=k_round is the wrapped candidate; k=0 is the
    // raw (un-shifted) candidate for large theta; k=k_round±1 cover neighbours.
    const double TWO_PI = 2.0 * M_PI;
    const int k_round = static_cast<int>(std::lround(theta / TWO_PI));

    const int k_arr[] = {k_round, 0, k_round - 1, k_round + 1};
    std::vector<int> k_cands;
    for (int k : k_arr) {
        bool seen = false;
        for (int kk : k_cands) if (kk == k) { seen = true; break; }
        if (!seen) k_cands.push_back(k);
    }

    Result best_r = {0.0, 0.0, 1e300};
    for (int k : k_cands) {
        double theta_eff = theta - TWO_PI * static_cast<double>(k);
        QueryResult qr = queryManifold(theta_eff, phi, p);
        if (qr.success && std::isfinite(qr.cost) && qr.cost < best_r.cost)
            best_r = {qr.lambda1, qr.lambda2, qr.cost};
    }

    if (best_r.cost < 1e299) return best_r;
    return {0.0, 0.0, 1e100};
}
