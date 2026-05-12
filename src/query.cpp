#include "query.h"
#include "continuation.h"
#include "dynamics.h"
#include "rk4.h"
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Cost (trapezoidal rule)
// ---------------------------------------------------------------------------

static double computeCost(double theta, double phi,
                           double lambda1, double lambda2,
                           const ContinuationParams& params) {
    State z = {theta, phi, lambda1, lambda2};
    auto f = [alpha = params.alpha](const State& zz) -> State {
        return forwardDynamics(zz, alpha);
    };
    int n_steps = static_cast<int>(std::round(params.T_max / params.h));

    double min_mag      = 1e18;
    double running_cost = 0.0;
    double cost_at_min  = 0.0;

    auto evalL = [](const State& s) {
        double th = s[0], ph = s[1], l2 = s[3];
        double u = -l2 * std::cos(th);
        return (1.0 - std::cos(th)) + 0.5*ph*ph + 0.5*u*u;
    };
    double L_prev = evalL(z);

    for (int s = 0; s < n_steps; ++s) {
        z = rk4Step(z, params.h, f);

        double L = evalL(z);
        running_cost += 0.5 * (L_prev + L) * params.h;
        L_prev = L;

        double mag = 0.0;
        for (double v : z) mag += v*v;
        mag = std::sqrt(mag);

        if (std::isnan(mag) || std::isinf(mag)) break;

        if (mag < min_mag) {
            min_mag     = mag;
            cost_at_min = running_cost;
        }
        if (mag > 2.0*min_mag) break;
    }
    return cost_at_min;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

QueryResult queryManifold(double theta_q, double phi_q,
                           const ContinuationParams& params) {
    QueryResult best = {0.0, 0.0, 1e18, theta_q, 1e18, false};

    auto res = continuationWalk(theta_q, phi_q, params);
    if (res.has_value()) {
        double cost = computeCost(theta_q, phi_q,
                                  res->lambda1, res->lambda2, params);
        best = {res->lambda1, res->lambda2, res->forward_residual, theta_q, cost, true};
    }
    return best;
}

double validateCostate(double theta, double phi,
                        double lambda1, double lambda2,
                        const ContinuationParams& params) {
    State z = {theta, phi, lambda1, lambda2};
    auto f = [alpha = params.alpha](const State& zz) -> State {
        return forwardDynamics(zz, alpha);
    };
    int    n_steps = static_cast<int>(std::round(params.T_max / params.h));
    double min_mag = 1e18;
    for (int s = 0; s < n_steps; ++s) {
        z = rk4Step(z, params.h, f);
        double mag = 0.0;
        for (double v : z) mag += v*v;
        mag = std::sqrt(mag);
        if (std::isnan(mag) || std::isinf(mag)) break;
        if (mag < min_mag) min_mag = mag;
    }
    return min_mag;
}
