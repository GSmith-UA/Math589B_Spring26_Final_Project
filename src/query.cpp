#include "query.h"
#include "continuation.h"
#include "dynamics.h"
#include "rk4.h"
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Newton refinement via variational equations
// ---------------------------------------------------------------------------

// Compute Df(z, alpha) * p where Df is the 4x4 Jacobian of forwardDynamics.
// Df rows:
//   [0,                          1,   0,                0        ]
//   [cos(th)+l2*sin(2th),    -alpha,  0,           -cos^2(th)   ]
//   [-(cos(th)-l2*sin(th)+l2^2*cos(2th)), 0, 0, -(cos(th)+l2*sin(2th))]
//   [0,                         -1,  -1,             alpha       ]
static void dfTimesP(const State& z, double alpha, const double p[4], double out[4]) {
    double th  = z[0], l2 = z[3];
    double sth = std::sin(th), cth = std::cos(th);
    double s2th = 2.0*sth*cth;
    double c2th = cth*cth - sth*sth;
    double cth2 = cth*cth;
    out[0] =  p[1];
    out[1] =  (cth + l2*s2th)*p[0] - alpha*p[1] - cth2*p[3];
    out[2] = -(cth - l2*sth + l2*l2*c2th)*p[0] - (cth + l2*s2th)*p[3];
    out[3] = -p[1] - p[2] + alpha*p[3];
}

// One RK4 step for the coupled (state z, sensitivity p1, p2) system.
// p1 = dz/d(lambda1_0), p2 = dz/d(lambda2_0)
static void varRK4Step(State& z, double p1[4], double p2[4], double h, double alpha) {
    auto fwd = [alpha](const State& s){ return forwardDynamics(s, alpha); };

    // Stage 1
    State kz1 = fwd(z);
    double kp1_1[4], kp2_1[4];
    dfTimesP(z, alpha, p1, kp1_1);
    dfTimesP(z, alpha, p2, kp2_1);

    // Stage 2 (midpoint)
    State z2; double q1[4], q2[4];
    for (int i = 0; i < 4; ++i) {
        z2[i] = z[i]  + 0.5*h*kz1[i];
        q1[i] = p1[i] + 0.5*h*kp1_1[i];
        q2[i] = p2[i] + 0.5*h*kp2_1[i];
    }
    State kz2 = fwd(z2);
    double kp1_2[4], kp2_2[4];
    dfTimesP(z2, alpha, q1, kp1_2);
    dfTimesP(z2, alpha, q2, kp2_2);

    // Stage 3 (midpoint)
    State z3;
    for (int i = 0; i < 4; ++i) {
        z3[i] = z[i]  + 0.5*h*kz2[i];
        q1[i] = p1[i] + 0.5*h*kp1_2[i];
        q2[i] = p2[i] + 0.5*h*kp2_2[i];
    }
    State kz3 = fwd(z3);
    double kp1_3[4], kp2_3[4];
    dfTimesP(z3, alpha, q1, kp1_3);
    dfTimesP(z3, alpha, q2, kp2_3);

    // Stage 4 (endpoint)
    State z4;
    for (int i = 0; i < 4; ++i) {
        z4[i] = z[i]  + h*kz3[i];
        q1[i] = p1[i] + h*kp1_3[i];
        q2[i] = p2[i] + h*kp2_3[i];
    }
    State kz4 = fwd(z4);
    double kp1_4[4], kp2_4[4];
    dfTimesP(z4, alpha, q1, kp1_4);
    dfTimesP(z4, alpha, q2, kp2_4);

    // Combine
    double c = h / 6.0;
    for (int i = 0; i < 4; ++i) {
        z[i]  += c*(kz1[i]  + 2*kz2[i]  + 2*kz3[i]  + kz4[i]);
        p1[i] += c*(kp1_1[i] + 2*kp1_2[i] + 2*kp1_3[i] + kp1_4[i]);
        p2[i] += c*(kp2_1[i] + 2*kp2_2[i] + 2*kp2_3[i] + kp2_4[i]);
    }
}

// Newton refinement: starting from initial (lambda1, lambda2), integrate the
// state + variational equations, then solve the 2x2 Newton system to drive
// (theta(T*), phi(T*)) → 0. Converges quadratically near the solution.
static std::pair<double,double> newtonRefine(
        double theta, double phi, double lambda1, double lambda2,
        const ContinuationParams& params, int max_iter = 5) {
    double l1 = lambda1, l2 = lambda2;
    double alpha  = params.alpha;
    double h      = params.h;
    int    n_steps = static_cast<int>(std::round(params.T_max / h));

    for (int iter = 0; iter < max_iter; ++iter) {
        State  z  = {theta, phi, l1, l2};
        double p1[4] = {0.0, 0.0, 1.0, 0.0};  // dz/d(lambda1_0)
        double p2[4] = {0.0, 0.0, 0.0, 1.0};  // dz/d(lambda2_0)

        double min_mag = 1e18;
        double F0 = theta, F1 = phi;
        double J00 = 0, J01 = 0, J10 = 0, J11 = 0;

        for (int s = 0; s < n_steps; ++s) {
            varRK4Step(z, p1, p2, h, alpha);

            double mag = 0.0;
            for (double v : z) mag += v*v;
            mag = std::sqrt(mag);
            if (std::isnan(mag) || std::isinf(mag)) break;

            if (mag < min_mag) {
                min_mag = mag;
                F0 = z[0]; F1 = z[1];           // residual: (theta, phi) at T*
                J00 = p1[0]; J01 = p2[0];       // d(theta)/d(l1_0, l2_0)
                J10 = p1[1]; J11 = p2[1];       // d(phi)  /d(l1_0, l2_0)
            }
            if (mag > 2.0*min_mag) break;
        }

        double res = std::sqrt(F0*F0 + F1*F1);
        std::fprintf(stderr, "[NEWTON] iter=%d  res=%.3e  lambda=(%.6f,%.6f)\n",
                     iter, res, l1, l2);

        if (res < 1e-8) break;

        double det = J00*J11 - J01*J10;
        if (std::abs(det) < 1e-12) break;

        double d1 = -(J11*F0 - J01*F1) / det;
        double d2 = -(J00*F1 - J10*F0) / det;

        // Clamp step to avoid divergence
        double step = std::sqrt(d1*d1 + d2*d2);
        if (step > 5.0) { d1 *= 5.0/step; d2 *= 5.0/step; }

        l1 += d1;
        l2 += d2;
    }
    return {l1, l2};
}

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
        auto [rl1, rl2] = newtonRefine(theta_q, phi_q,
                                        res->lambda1, res->lambda2, params);
        double fwd  = validateCostate(theta_q, phi_q, rl1, rl2, params);
        double cost = computeCost(theta_q, phi_q, rl1, rl2, params);
        best = {rl1, rl2, fwd, theta_q, cost, true};
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
