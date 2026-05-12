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
    const double alpha   = params.alpha;
    const double h       = params.h;
    const int    n_steps = static_cast<int>(std::round(params.T_max / h));

    double th = theta, ph = phi, l1 = lambda1, l2 = lambda2;

    auto evalL = [](double th_, double ph_, double l2_) {
        double u = -l2_ * std::cos(th_);
        return (1.0 - std::cos(th_)) + 0.5*ph_*ph_ + 0.5*u*u;
    };

    double min_mag      = 1e18;
    double running_cost = 0.0;
    double cost_at_min  = 0.0;
    double L_prev       = evalL(th, ph, l2);

    for (int s = 0; s < n_steps; ++s) {
        State z  = {th, ph, l1, l2};
        State k1 = forwardDynamics(z, alpha);
        State zm = {th+0.5*h*k1[0], ph+0.5*h*k1[1], l1+0.5*h*k1[2], l2+0.5*h*k1[3]};
        State k2 = forwardDynamics(zm, alpha);
        zm = {th+0.5*h*k2[0], ph+0.5*h*k2[1], l1+0.5*h*k2[2], l2+0.5*h*k2[3]};
        State k3 = forwardDynamics(zm, alpha);
        zm = {th+h*k3[0], ph+h*k3[1], l1+h*k3[2], l2+h*k3[3]};
        State k4 = forwardDynamics(zm, alpha);
        double c6 = h / 6.0;
        th += c6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]);
        ph += c6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]);
        l1 += c6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]);
        l2 += c6*(k1[3]+2*k2[3]+2*k3[3]+k4[3]);

        double L = evalL(th, ph, l2);
        running_cost += 0.5*(L_prev + L)*h;
        L_prev = L;

        double mag = std::sqrt(th*th + ph*ph + l1*l1 + l2*l2);
        if (std::isnan(mag) || std::isinf(mag)) break;
        if (mag < min_mag) { min_mag = mag; cost_at_min = running_cost; }
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
