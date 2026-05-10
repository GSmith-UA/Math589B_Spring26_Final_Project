#include "continuation.h"
#include "dynamics.h"
#include "manifold.h"
#include "rk4.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <Eigen/Dense>

static double forwardResidual(double theta, double phi,
                               double lambda1, double lambda2,
                               const ContinuationParams& params) {
    State z = {theta, phi, lambda1, lambda2};
    auto f  = [alpha = params.alpha](const State& zz) -> State {
        return forwardDynamics(zz, alpha);
    };
    int    n_steps = static_cast<int>(std::round(params.T_max / params.h));
    double min_mag = 1e18;
    for (int s = 0; s < n_steps; ++s) {
        z = rk4Step(z, params.h, f);
        double mag = 0.0;
        for (double v : z) mag += v * v;
        mag = std::sqrt(mag);
        if (std::isnan(mag) || std::isinf(mag)) break;
        if (mag < min_mag) min_mag = mag;
    }
    return min_mag;
}

// Greedy h* point selection using quadratic features.
// drop_phi=false: features = [θ², θ, φ, 1]  (4D)
// drop_phi=true:  features = [θ², θ, 1]     (3D)
static std::vector<FlagResult> hstarGreedy(
        const std::vector<FlagResult>& flags,
        double theta, double phi,
        int k,
        bool drop_phi,
        double lambda_reg = 1e-6,
        double eps_info   = 1e-10)
{
    int dim = drop_phi ? 3 : 4;

    Eigen::VectorXd x_star(dim);
    if (drop_phi) x_star << theta * theta, theta, 1.0;
    else          x_star << theta * theta, theta, phi, 1.0;

    Eigen::MatrixXd XtX_inv = (1.0 / lambda_reg) * Eigen::MatrixXd::Identity(dim, dim);

    std::vector<bool>       used(flags.size(), false);
    std::vector<FlagResult> selected;

    for (int iter = 0; iter < k; ++iter) {
        double best_delta = 0.0;
        int    best_idx   = -1;

        for (int i = 0; i < (int)flags.size(); ++i) {
            if (used[i]) continue;
            double tf = flags[i].state_at_flag[0];
            double pf = flags[i].state_at_flag[1];
            Eigen::VectorXd xf(dim);
            if (drop_phi) xf << tf * tf, tf, 1.0;
            else          xf << tf * tf, tf, pf, 1.0;
            Eigen::VectorXd v = XtX_inv * xf;
            double denom = 1.0 + xf.dot(v);
            if (denom <= 0.0) continue;
            double delta = (x_star.dot(v) * x_star.dot(v)) / denom;
            if (delta > best_delta) { best_delta = delta; best_idx = i; }
        }

        if (best_idx < 0 || best_delta < eps_info) break;

        double tf = flags[best_idx].state_at_flag[0];
        double pf = flags[best_idx].state_at_flag[1];
        Eigen::VectorXd xf(dim);
        if (drop_phi) xf << tf * tf, tf, 1.0;
        else          xf << tf * tf, tf, pf, 1.0;
        Eigen::VectorXd v = XtX_inv * xf;
        XtX_inv -= v * v.transpose() / (1.0 + xf.dot(v));

        used[best_idx] = true;
        selected.push_back(flags[best_idx]);
    }
    return selected;
}

CostateEstimate solveAtPoint(double theta, double phi,
                              std::array<double, 2> warm_start,
                              const ContinuationParams& params,
                              bool cold_start,
                              double& psi_center) {
    double alpha    = params.alpha;
    double flag_tol = params.epsilon_init;
    bool   drop_phi = std::abs(phi) < 1e-5;

    CostateEstimate best = {warm_start[0], warm_start[1], 1e18, false, 0.0};

    std::fprintf(stderr, "[DBG] solveAtPoint(%.4f,%.4f) %s  drop_phi=%d\n",
                 theta, phi, cold_start ? "COLD" : "WARM", (int)drop_phi);

    State   origin = {0.0, 0.0, 0.0, 0.0};
    Matrix4 J      = computeJacobian(origin, alpha);
    EigenpairResult eigs = computeStableEigenpairs(J);

    double arc = cold_start ? 2.0 * M_PI : M_PI / 2.0;
    if (cold_start) psi_center = 0.0;

    double h_s = (params.h_shoot > 0.0) ? params.h_shoot : params.h;

    std::vector<FlagResult> all_flags;

    for (int pass = 0; pass < params.max_passes; ++pass) {
        int n_seeds = params.N_psi * std::min(1 << pass, 2);

        auto seeds = generateSeedPointsArc(eigs, n_seeds, params.r,
                                           psi_center, arc, origin);
        auto flags = shootAndFlag(seeds, theta, phi, alpha,
                                  params.T_max, h_s, flag_tol);

        double     pass_best_dist = 1e18;
        double     pass_best_psi  = psi_center;
        FlagResult pass_best_fr;
        bool       pass_has_best  = false;
        int        pass_n_flagged = 0;
        int        pass_n_bad     = 0;

        for (int i = 0; i < (int)flags.size(); ++i) {
            if (!flags[i].flagged) continue;
            ++pass_n_flagged;

            double psi_i = psi_center + arc * (i - n_seeds / 2.0) / n_seeds;

            double l1 = flags[i].state_at_flag[2];
            double l2 = flags[i].state_at_flag[3];
            if (std::abs(l1) > 100.0 || std::abs(l2) > 100.0) {
                ++pass_n_bad;
                std::fprintf(stderr, "[BADFLAG] pass=%d λ=(%.3e,%.3e)\n", pass, l1, l2);
                continue;
            }

            all_flags.push_back(flags[i]);

            if (flags[i].min_dist < pass_best_dist) {
                pass_best_dist = flags[i].min_dist;
                pass_best_psi  = psi_i;
                pass_best_fr   = flags[i];
                pass_has_best  = true;
            }
        }

        if (pass_has_best) {
            const State& st = pass_best_fr.state_at_flag;
            double resid = forwardResidual(theta, phi, st[2], st[3], params);
            if (resid < best.forward_residual)
                best = {st[2], st[3], resid, resid < params.epsilon_fwd, pass_best_psi};
        }

        if (pass_has_best) {
            psi_center = pass_best_psi;
            arc        = M_PI / 2.0;
        } else {
            arc = std::min(arc * 2.0, 2.0 * M_PI);
        }

        std::fprintf(stderr,
            "[DBG]   pass=%d  n_seeds=%d  arc=%.4f  n_flagged=%d  n_bad=%d"
            "  best_dist=%.3e  fwd_resid=%.3e  psi_c=%.3f\n",
            pass, n_seeds, arc, pass_n_flagged, pass_n_bad, pass_best_dist,
            best.forward_residual, psi_center);

        if (best.accepted) break;

        // Build candidate pool: φ-filtered when drop_phi to avoid manifold contamination.
        std::vector<FlagResult> ls_candidates;
        if (drop_phi) {
            for (const auto& fr : all_flags)
                if (std::abs(fr.state_at_flag[1] - phi) < 0.02)
                    ls_candidates.push_back(fr);
        } else {
            ls_candidates = all_flags;
        }

        auto ls_pts  = hstarGreedy(ls_candidates, theta, phi, 12, drop_phi);
        int  min_pts = drop_phi ? 3 : 4;

        std::fprintf(stderr, "[DBG]   h* selected %d/%d candidates\n",
                     (int)ls_pts.size(), (int)ls_candidates.size());

        if ((int)ls_pts.size() >= min_pts) {
            int n    = (int)ls_pts.size();
            int ncol = drop_phi ? 3 : 4;

            Eigen::MatrixXd X(n, ncol), Y(n, 2);
            for (int i = 0; i < n; ++i) {
                double tf = ls_pts[i].state_at_flag[0];
                double pf = ls_pts[i].state_at_flag[1];
                X(i, 0) = tf * tf;
                X(i, 1) = tf;
                if (!drop_phi) { X(i, 2) = pf; X(i, 3) = 1.0; }
                else           { X(i, 2) = 1.0; }
                Y(i, 0) = ls_pts[i].state_at_flag[2];
                Y(i, 1) = ls_pts[i].state_at_flag[3];
            }

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(X);
            Eigen::VectorXd sv = svd.singularValues();
            double cond = (sv(sv.size()-1) > 0.0) ? sv(0)/sv(sv.size()-1) : 1e18;
            std::fprintf(stderr, "[LS_COND] cond=%.3e\n", cond);

            auto qr = X.colPivHouseholderQr();
            if (qr.rank() >= ncol) {
                Eigen::MatrixXd A = qr.solve(Y);
                Eigen::VectorXd xt(ncol);
                if (drop_phi) xt << theta * theta, theta, 1.0;
                else          xt << theta * theta, theta, phi, 1.0;
                Eigen::Vector2d lam = A.transpose() * xt;
                double l1 = lam(0), l2 = lam(1);

                if (std::abs(l1) < 1e6 && std::abs(l2) < 1e6) {
                    double resid  = forwardResidual(theta, phi, l1, l2, params);
                    bool   better = resid < best.forward_residual;
                    std::fprintf(stderr,
                        "[LIN] n=%d  lambda=(%.6f,%.6f)  fwd_resid=%.3e%s\n",
                        n, l1, l2, resid, better ? "  [BETTER]" : "");
                    if (better)
                        best = {l1, l2, resid, resid < params.epsilon_fwd, best.best_psi};
                } else {
                    std::fprintf(stderr, "[LIN] lambda exploded\n");
                }
            } else {
                std::fprintf(stderr, "[LIN] rank deficient\n");
            }
        }

        if (best.accepted) break;
    }

    std::fprintf(stderr, "[DBG] final lambda=(%.6f,%.6f) fwd_resid=%.3e\n",
                 best.lambda1, best.lambda2, best.forward_residual);
    return best;
}

std::optional<CostateEstimate> continuationWalk(double theta_q, double phi_q,
                                                 const ContinuationParams& params,
                                                 std::vector<StiffnessRecord>* stiffness_out) {
    double delta_theta = shortestArc(0.0, theta_q);
    double arc         = std::max(std::abs(delta_theta), std::abs(phi_q));
    int    N           = std::max(1, static_cast<int>(
                             std::ceil(arc / params.delta_step)));

    std::array<double, 2> warm_start = {0.0, 0.0};
    CostateEstimate current = {0.0, 0.0, 0.0, true, 0.0};
    double psi_center = 0.0;

    std::fprintf(stderr, "\n=== continuationWalk(%.4f,%.4f) N=%d ===\n",
                 theta_q, phi_q, N);

    for (int n = 1; n <= N; ++n) {
        double frac    = static_cast<double>(n) / N;
        double theta_n = frac * delta_theta;
        double phi_n   = frac * phi_q;

        ContinuationParams step_params = params;
        step_params.epsilon_fwd        = 1e-3;

        bool cold_start = (n == 1);

        std::fprintf(stderr, "\n--- step %d/%d  theta=%.4f phi=%.4f ---\n",
                     n, N, theta_n, phi_n);

        CostateEstimate est = solveAtPoint(theta_n, phi_n, warm_start,
                                           step_params, cold_start, psi_center);

        if (est.accepted) {
            warm_start = {est.lambda1, est.lambda2};
            current    = est;
            std::fprintf(stderr, "[DBG] step %d ACCEPTED  resid=%.3e\n",
                         n, est.forward_residual);
        } else {
            std::fprintf(stderr, "[DBG] step %d REJECTED  resid=%.3e\n",
                         n, est.forward_residual);
            return std::nullopt;
        }
    }
    return current;
}
