#include "continuation.h"
#include "dynamics.h"
#include "manifold.h"
#include "rk4.h"
#include "utils.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <Eigen/Dense>

#ifdef USE_GPU
extern std::vector<FlagResult> shootAndFlagGPU(const std::vector<State>&,
                                                double, double, double,
                                                double, double, double);
#define shootAndFlag shootAndFlagGPU
#endif

static inline State addScaled(const State& a, double s, const State& b) {
    return { a[0]+s*b[0], a[1]+s*b[1], a[2]+s*b[2], a[3]+s*b[3] };
}


static double forwardResidual(double theta, double phi,
                               double lambda1, double lambda2,
                               const ContinuationParams& params) {
    const double alpha   = params.alpha;
    const double h       = params.h;
    const int    n_steps = static_cast<int>(std::round(params.T_max / h));

    double th = theta, ph = phi, l1 = lambda1, l2 = lambda2;
    double min_mag = 1e18;

    for (int s = 0; s < n_steps; ++s) {
        State z   = {th, ph, l1, l2};
        State k1  = forwardDynamics(z, alpha);
        State zm  = addScaled(z, 0.5*h, k1);
        State k2  = forwardDynamics(zm, alpha);
        zm        = addScaled(z, 0.5*h, k2);
        State k3  = forwardDynamics(zm, alpha);
        zm        = addScaled(z, h, k3);
        State k4  = forwardDynamics(zm, alpha);
        double c6 = h / 6.0;
        th += c6 * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]);
        ph += c6 * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]);
        l1 += c6 * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]);
        l2 += c6 * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]);

        double mag = std::sqrt(th*th + ph*ph + l1*l1 + l2*l2);
        if (std::isnan(mag) || std::isinf(mag)) break;
        if (mag < min_mag) min_mag = mag;
        if (min_mag < params.epsilon_fwd) break;
    }
    return min_mag;
}

// ---------------------------------------------------------------------------
// Newton refinement via variational equations (2×2 shooting system)
// ---------------------------------------------------------------------------

static CostateEstimate newtonRefine(double theta, double phi,
                                     double l1_init, double l2_init,
                                     const ContinuationParams& params) {
    const double alpha   = params.alpha;
    const double h_n     = params.h;
    const int    n_steps = static_cast<int>(std::round(params.T_max / h_n));

    double l1 = l1_init, l2 = l2_init;
    double best_resid = forwardResidual(theta, phi, l1, l2, params);
    double best_l1 = l1, best_l2 = l2;

    for (int iter = 0; iter < 5 && best_resid > params.epsilon_fwd; ++iter) {
        double th = theta, ph = phi, lam1 = l1, lam2 = l2;
        // Φ columns: pa = dz/dl1(0), pb = dz/dl2(0)
        double pa[4] = {0, 0, 1, 0};
        double pb[4] = {0, 0, 0, 1};

        double min_mag = 1e18;
        double r_th = 0, r_ph = 0;
        double J00 = 0, J01 = 0, J10 = 0, J11 = 0;

        bool bad = false;
        for (int s = 0; s < n_steps; ++s) {
            State z1 = {th, ph, lam1, lam2};
            State f1 = forwardDynamics(z1, alpha);
            Matrix4 A1 = computeJacobian(z1, alpha);
            double k1a[4], k1b[4];
            for (int i=0;i<4;i++){
                k1a[i]=A1[i][0]*pa[0]+A1[i][1]*pa[1]+A1[i][2]*pa[2]+A1[i][3]*pa[3];
                k1b[i]=A1[i][0]*pb[0]+A1[i][1]*pb[1]+A1[i][2]*pb[2]+A1[i][3]*pb[3];
            }
            double h2=0.5*h_n;
            State z2={th+h2*f1[0],ph+h2*f1[1],lam1+h2*f1[2],lam2+h2*f1[3]};
            double pa2[4],pb2[4];
            for(int i=0;i<4;i++){pa2[i]=pa[i]+h2*k1a[i];pb2[i]=pb[i]+h2*k1b[i];}
            State f2=forwardDynamics(z2,alpha);
            Matrix4 A2=computeJacobian(z2,alpha);
            double k2a[4],k2b[4];
            for(int i=0;i<4;i++){
                k2a[i]=A2[i][0]*pa2[0]+A2[i][1]*pa2[1]+A2[i][2]*pa2[2]+A2[i][3]*pa2[3];
                k2b[i]=A2[i][0]*pb2[0]+A2[i][1]*pb2[1]+A2[i][2]*pb2[2]+A2[i][3]*pb2[3];
            }
            State z3={th+h2*f2[0],ph+h2*f2[1],lam1+h2*f2[2],lam2+h2*f2[3]};
            double pa3[4],pb3[4];
            for(int i=0;i<4;i++){pa3[i]=pa[i]+h2*k2a[i];pb3[i]=pb[i]+h2*k2b[i];}
            State f3=forwardDynamics(z3,alpha);
            Matrix4 A3=computeJacobian(z3,alpha);
            double k3a[4],k3b[4];
            for(int i=0;i<4;i++){
                k3a[i]=A3[i][0]*pa3[0]+A3[i][1]*pa3[1]+A3[i][2]*pa3[2]+A3[i][3]*pa3[3];
                k3b[i]=A3[i][0]*pb3[0]+A3[i][1]*pb3[1]+A3[i][2]*pb3[2]+A3[i][3]*pb3[3];
            }
            State z4={th+h_n*f3[0],ph+h_n*f3[1],lam1+h_n*f3[2],lam2+h_n*f3[3]};
            double pa4[4],pb4[4];
            for(int i=0;i<4;i++){pa4[i]=pa[i]+h_n*k3a[i];pb4[i]=pb[i]+h_n*k3b[i];}
            State f4=forwardDynamics(z4,alpha);
            Matrix4 A4=computeJacobian(z4,alpha);
            double k4a[4],k4b[4];
            for(int i=0;i<4;i++){
                k4a[i]=A4[i][0]*pa4[0]+A4[i][1]*pa4[1]+A4[i][2]*pa4[2]+A4[i][3]*pa4[3];
                k4b[i]=A4[i][0]*pb4[0]+A4[i][1]*pb4[1]+A4[i][2]*pb4[2]+A4[i][3]*pb4[3];
            }
            double c6=h_n/6.0;
            th   +=c6*(f1[0]+2*f2[0]+2*f3[0]+f4[0]);
            ph   +=c6*(f1[1]+2*f2[1]+2*f3[1]+f4[1]);
            lam1 +=c6*(f1[2]+2*f2[2]+2*f3[2]+f4[2]);
            lam2 +=c6*(f1[3]+2*f2[3]+2*f3[3]+f4[3]);
            for(int i=0;i<4;i++){
                pa[i]+=c6*(k1a[i]+2*k2a[i]+2*k3a[i]+k4a[i]);
                pb[i]+=c6*(k1b[i]+2*k2b[i]+2*k3b[i]+k4b[i]);
            }
            double mag=std::sqrt(th*th+ph*ph+lam1*lam1+lam2*lam2);
            if (!std::isfinite(mag)) { bad=true; break; }
            if (mag < min_mag) {
                min_mag=mag;
                r_th=th; r_ph=ph;
                J00=pa[0]; J01=pb[0];
                J10=pa[1]; J11=pb[1];
            }
            if (min_mag < params.epsilon_fwd) break;
            if (min_mag < 1.0 && mag > 2.0*min_mag) break;  // past min, diverging
        }
        if (bad) break;
        if (!std::isfinite(J00)||!std::isfinite(J01)||
            !std::isfinite(J10)||!std::isfinite(J11)) break;

        // 2×2 Newton: J·Δλ = -r  where r = (θ(T*), φ(T*))
        double det = J00*J11 - J01*J10;
        if (std::abs(det) < 1e-20) break;
        double dl1 = (-r_th*J11 + r_ph*J01) / det;
        double dl2 = ( r_th*J10 - r_ph*J00) / det;

        double dnorm = std::sqrt(dl1*dl1 + dl2*dl2);
        if (dnorm < 1e-12) break;
        if (dnorm > 5.0) { dl1 *= 5.0/dnorm; dl2 *= 5.0/dnorm; }

        double step = 1.0;
        bool improved = false;
        for (int bt=0; bt<8; ++bt, step*=0.5) {
            double nl1=l1+step*dl1, nl2=l2+step*dl2;
            if (std::abs(nl1)>50||std::abs(nl2)>50) continue;
            double nr=forwardResidual(theta,phi,nl1,nl2,params);
            if (nr < best_resid) {
                best_resid=nr; best_l1=nl1; best_l2=nl2;
                l1=nl1; l2=nl2; improved=true; break;
            }
        }
        if (!improved) break;
    }
    return {best_l1, best_l2, best_resid, best_resid < params.epsilon_fwd, 0.0};
}

// Greedy h* point selection using quadratic features.
// drop_phi=false: features = [θ², θ, θφ, φ², φ, 1]  (6D)
// drop_phi=true:  features = [θ², θ, 1]               (3D)
static std::vector<FlagResult> hstarGreedy(
        const std::vector<FlagResult>& flags,
        double theta, double phi,
        int k,
        bool drop_phi,
        double lambda_reg = 1e-6,
        double eps_info   = 1e-10)
{
    int dim = drop_phi ? 3 : 6;

    Eigen::VectorXd x_star(dim);
    if (drop_phi) x_star << theta * theta, theta, 1.0;
    else          x_star << theta * theta, theta, theta * phi, phi * phi, phi, 1.0;

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
            else          xf << tf * tf, tf, tf * pf, pf * pf, pf, 1.0;
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
        else          xf << tf * tf, tf, tf * pf, pf * pf, pf, 1.0;
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

    //std::fprintf(stderr, "[DBG] solveAtPoint(%.4f,%.4f) %s  drop_phi=%d\n",
    //             theta, phi, cold_start ? "COLD" : "WARM", (int)drop_phi);

    State   origin = {0.0, 0.0, 0.0, 0.0};
    Matrix4 J      = computeJacobian(origin, alpha);
    EigenpairResult eigs = computeStableEigenpairs(J);

    double arc = cold_start ? 2.0 * M_PI : M_PI / 2.0;
    if (cold_start) psi_center = 0.0;

    double h_s = (params.h_shoot > 0.0) ? params.h_shoot : params.h;

    std::vector<FlagResult> all_flags;
    std::vector<double>     all_psi;

    for (int pass = 0; pass < params.max_passes; ++pass) {
        int n_seeds = params.N_psi;

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
            if (std::abs(l1) > 25.0 || std::abs(l2) > 25.0) {
                ++pass_n_bad;
                //std::fprintf(stderr, "[BADFLAG] pass=%d λ=(%.3e,%.3e)\n", pass, l1, l2);
                continue;
            }

            all_flags.push_back(flags[i]);
            all_psi.push_back(psi_i);

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

        //std::fprintf(stderr,
        //    "[DBG]   pass=%d  n_seeds=%d  arc=%.4f  n_flagged=%d  n_bad=%d"
        //    "  best_dist=%.3e  fwd_resid=%.3e  psi_c=%.3f\n",
        //    pass, n_seeds, arc, pass_n_flagged, pass_n_bad, pass_best_dist,
        //    best.forward_residual, psi_center);

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
        int  min_pts = drop_phi ? 3 : 6;

        //std::fprintf(stderr, "[DBG]   h* selected %d/%d candidates\n",
        //             (int)ls_pts.size(), (int)ls_candidates.size());

        if ((int)ls_pts.size() >= min_pts) {
            int n    = (int)ls_pts.size();
            int ncol = drop_phi ? 3 : 6;

            Eigen::MatrixXd X(n, ncol), Y(n, 2);
            for (int i = 0; i < n; ++i) {
                double tf = ls_pts[i].state_at_flag[0];
                double pf = ls_pts[i].state_at_flag[1];
                X(i, 0) = tf * tf;
                X(i, 1) = tf;
                if (!drop_phi) { X(i, 2) = tf * pf; X(i, 3) = pf * pf; X(i, 4) = pf; X(i, 5) = 1.0; }
                else           { X(i, 2) = 1.0; }
                Y(i, 0) = ls_pts[i].state_at_flag[2];
                Y(i, 1) = ls_pts[i].state_at_flag[3];
            }

            //Eigen::JacobiSVD<Eigen::MatrixXd> svd(X);
            //Eigen::VectorXd sv = svd.singularValues();
            //double cond = (sv(sv.size()-1) > 0.0) ? sv(0)/sv(sv.size()-1) : 1e18;
            //std::fprintf(stderr, "[LS_COND] cond=%.3e\n", cond);

            auto qr = X.colPivHouseholderQr();
            if (qr.rank() >= ncol) {
                Eigen::MatrixXd A = qr.solve(Y);
                Eigen::VectorXd xt(ncol);
                if (drop_phi) xt << theta * theta, theta, 1.0;
                else          xt << theta * theta, theta, theta * phi, phi * phi, phi, 1.0;
                Eigen::Vector2d lam = A.transpose() * xt;
                double l1 = lam(0), l2 = lam(1);

                if (std::abs(l1) < 1e6 && std::abs(l2) < 1e6) {
                    double resid = forwardResidual(theta, phi, l1, l2, params);
                    if (resid < best.forward_residual)
                        best = {l1, l2, resid, resid < params.epsilon_fwd, best.best_psi};
                    //std::fprintf(stderr,
                    //    "[LIN] n=%d  lambda=(%.6f,%.6f)  fwd_resid=%.3e\n",
                    //    n, l1, l2, resid);
                }
                //else std::fprintf(stderr, "[LIN] lambda exploded\n");
            }
            //else std::fprintf(stderr, "[LIN] rank deficient\n");
        }

        if (best.accepted) break;
    }

    //std::fprintf(stderr, "[DBG] final lambda=(%.6f,%.6f) fwd_resid=%.3e\n",
    //             best.lambda1, best.lambda2, best.forward_residual);

    // --- Newton refinement from best estimate + diverse ψ-wells ---
    {
        CostateEstimate nr = newtonRefine(theta, phi,
                                          best.lambda1, best.lambda2, params);
        if (nr.forward_residual < best.forward_residual)
            best = {nr.lambda1, nr.lambda2, nr.forward_residual, nr.accepted, best.best_psi};

        if (!best.accepted && !all_flags.empty()) {
            const int    K_wells  = 2;
            const double psi_gap  = 0.3;
            std::vector<bool> excluded(all_flags.size(), false);

            // Exclude ψ-neighborhood of current best so wells are diverse
            for (int i = 0; i < (int)all_flags.size(); ++i) {
                double dp = std::abs(all_psi[i] - best.best_psi);
                if (dp > M_PI) dp = 2*M_PI - dp;
                if (dp < psi_gap) excluded[i] = true;
            }

            for (int k = 0; k < K_wells && !best.accepted; ++k) {
                int    widx  = -1;
                double wdist = 1e18;
                for (int i = 0; i < (int)all_flags.size(); ++i) {
                    if (!excluded[i] && all_flags[i].min_dist < wdist) {
                        wdist = all_flags[i].min_dist; widx = i;
                    }
                }
                if (widx < 0) break;

                double psi_w = all_psi[widx];
                for (int i = 0; i < (int)all_flags.size(); ++i) {
                    double dp = std::abs(all_psi[i] - psi_w);
                    if (dp > M_PI) dp = 2*M_PI - dp;
                    if (dp < psi_gap) excluded[i] = true;
                }

                const State& st = all_flags[widx].state_at_flag;
                CostateEstimate wr = newtonRefine(theta, phi, st[2], st[3], params);
                if (wr.forward_residual < best.forward_residual)
                    best = {wr.lambda1, wr.lambda2, wr.forward_residual,
                            wr.accepted, psi_w};
            }
        }
    }

    return best;
}

std::optional<CostateEstimate> continuationWalk(double theta_q, double phi_q,
                                                 const ContinuationParams& params,
                                                 std::vector<StiffnessRecord>* stiffness_out) {
    double delta_theta = shortestArc(0.0, theta_q);
    double arc         = std::max(std::abs(delta_theta), std::abs(phi_q));
    int    N           = std::max(1, static_cast<int>(
                             std::ceil(arc / params.delta_step)));
    if (params.max_steps > 0) N = std::min(N, params.max_steps);

    std::array<double, 2> warm_start = {0.0, 0.0};
    CostateEstimate current = {0.0, 0.0, 1e18, false, 0.0};
    double psi_center = 0.0;

    //std::fprintf(stderr, "\n=== continuationWalk(%.4f,%.4f) N=%d ===\n",
    //             theta_q, phi_q, N);

    for (int n = 1; n <= N; ++n) {
        double frac    = static_cast<double>(n) / N;
        double theta_n = frac * delta_theta;
        double phi_n   = frac * phi_q;

        ContinuationParams step_params = params;
        step_params.epsilon_fwd        = 1e-3;

        bool cold_start = (n == 1);

        //std::fprintf(stderr, "\n--- step %d/%d  theta=%.4f phi=%.4f ---\n",
        //             n, N, theta_n, phi_n);

        CostateEstimate est = solveAtPoint(theta_n, phi_n, warm_start,
                                           step_params, cold_start, psi_center);

        if (est.accepted) {
            warm_start = {est.lambda1, est.lambda2};
            current    = est;
            //std::fprintf(stderr, "[DBG] step %d ACCEPTED  resid=%.3e\n",
            //             n, est.forward_residual);
        } else {
            //std::fprintf(stderr, "[DBG] step %d REJECTED  resid=%.3e\n",
            //             n, est.forward_residual);
            if (est.forward_residual < current.forward_residual)
                current = est;
            break;
        }
    }
    return current;
}
