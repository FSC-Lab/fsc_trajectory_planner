// MIT License
// Copyright (c) 2026 FSC Lab
//
// Straight-line Picard transition planner — port of transition_planner.py's
// plan_transition() plus the polynomial helpers of compatible_trajectory.py
// (_fit_poly_cols / _eval_poly_cols on the single-segment path).
#include "fsc_trajectory_planner/transition_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fsc_trajectory_planner
{

namespace
{

const Vec3 kE3{0.0, 0.0, 1.0};
const Mat3 kSZ = (Mat3() << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
  .finished();
const Mat3 kSX = (Mat3() << 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
  .finished();

// Min-snap phase peak factors (computed once, numerically exact — the Python
// evaluates the same 4001-point grid).
struct PhasePeaks
{
  double ds{0.0}, d2s{0.0};
  PhasePeaks()
  {
    for (int i = 0; i <= 4000; ++i) {
      double s, d1, d2;
      minsnap3(i / 4000.0, &s, &d1, &d2);
      ds = std::max(ds, d1);
      d2s = std::max(d2s, std::abs(d2));
    }
  }
};
const PhasePeaks & phasePeaks()
{
  static const PhasePeaks p;
  return p;
}

// ---- polynomial in the scaled variable tau = (t - tc)/ts -------------------
// Coefficients ASCENDING in tau (3 rows = xyz).
struct Poly
{
  double tc{0.0}, ts{1.0};
  Eigen::Matrix<double, 3, Eigen::Dynamic> c;
  // value and 4 derivatives wrt TIME at t: out(d, :) = p^(d)(t).
  void eval(double t, Eigen::Matrix<double, 5, 3> * out) const
  {
    const double tau = (t - tc) / ts;
    const int deg = static_cast<int>(c.cols()) - 1;
    for (int i = 0; i < 3; ++i) {
      double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;
      for (int k = deg; k >= 0; --k) {
        p4 = p4 * tau + 4.0 * p3;
        p3 = p3 * tau + 3.0 * p2;
        p2 = p2 * tau + 2.0 * p1;
        p1 = p1 * tau + p0;
        p0 = p0 * tau + c(i, k);
      }
      (*out)(0, i) = p0;
      (*out)(1, i) = p1 / ts;
      (*out)(2, i) = p2 / (ts * ts);
      (*out)(3, i) = p3 / (ts * ts * ts);
      (*out)(4, i) = p4 / (ts * ts * ts * ts);
    }
  }
};

// Least-squares fit of Y (3 x n) over t (n) on [t0, t1], degree deg — the
// unconstrained single-segment fit (np.polyfit: Vandermonde columns scaled to
// unit norm, rank-revealing least squares).
Poly fitPoly(
  const Eigen::VectorXd & t, const Eigen::Matrix<double, 3, Eigen::Dynamic> & Y,
  double t0, double t1, int deg)
{
  Poly p;
  p.tc = 0.5 * (t0 + t1);
  p.ts = 0.5 * (t1 - t0);
  const int n = static_cast<int>(t.size());
  Eigen::MatrixXd A(n, deg + 1);
  for (int i = 0; i < n; ++i) {
    const double tau = (t(i) - p.tc) / p.ts;
    double v = 1.0;
    for (int k = 0; k <= deg; ++k) {
      A(i, k) = v;
      v *= tau;
    }
  }
  const Eigen::VectorXd scale = A.colwise().norm();
  Eigen::MatrixXd As = A;
  for (int k = 0; k <= deg; ++k) {As.col(k) /= scale(k);}
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(As, Eigen::ComputeThinU | Eigen::ComputeThinV);
  svd.setThreshold(n * std::numeric_limits<double>::epsilon());
  p.c.resize(3, deg + 1);
  for (int i = 0; i < 3; ++i) {
    const Eigen::VectorXd y = Y.row(i).transpose();
    const Eigen::VectorXd cs = svd.solve(y);
    for (int k = 0; k <= deg; ++k) {p.c(i, k) = cs(k) / scale(k);}
  }
  return p;
}

// ---- the prescribed task ----------------------------------------------------
struct TaskSample
{
  Vec3 r_ed, r_ed_dot, r_ed_ddot;
  Mat3 R_e;
  Vec3 b1_de, b1_de_dot, b1_de_ddot;
  Vec3 b1_d, b1_d_dot, b1_d_ddot;
  double split{0.5};
};

struct RestAngles
{
  double al, be, ga, pp, split;
};

RestAngles restAngles(double phi, const VecN & q)
{
  const double be = q(1) + q(2);
  const double split = std::abs(be) > 1e-9 ? q(1) / be : 0.5;
  return {phi + q(0), be, q(3), phi, split};
}

class StraightLineTrajectory : public Trajectory
{
public:
  double duration() const override {return T_;}
  RestSpec goalRest() const override {return rest1_;}
  const PlanDiag & diag() const override {return diag_;}

  TaskSample task(double tq) const
  {
    double s, ds, d2s;
    minsnap3(std::min(1.0, std::max(0.0, tq / T_)), &s, &ds, &d2s);
    const double sd = ds / T_, sdd = d2s / (T_ * T_);
    TaskSample tk;
    tk.r_ed = p_e0_ + dp_ * s;
    tk.r_ed_dot = dp_ * sd;
    tk.r_ed_ddot = dp_ * sdd;

    const double al = a0_.al + (a1_.al - a0_.al) * s, al1 = a1_.al - a0_.al;
    const double be = a0_.be + (a1_.be - a0_.be) * s, be1 = a1_.be - a0_.be;
    const double ga = a0_.ga + (a1_.ga - a0_.ga) * s, ga1 = a1_.ga - a0_.ga;
    const Mat3 A = Rz(al), B = Rx(be), C = Rz(ga);
    const Mat3 dA = kSZ * A * al1;
    const Mat3 d2A = kSZ * kSZ * A * al1 * al1;  // al2 = 0
    const Mat3 dB = kSX * B * be1;
    const Mat3 d2B = kSX * kSX * B * be1 * be1;
    const Mat3 dC = kSZ * C * ga1;
    const Mat3 d2C = kSZ * kSZ * C * ga1 * ga1;
    const Mat3 re = A * B * C;
    const Mat3 dre = dA * B * C + A * dB * C + A * B * dC;
    const Mat3 d2re = d2A * B * C + A * d2B * C + A * B * d2C +
      2.0 * (dA * dB * C + dA * B * dC + A * dB * dC);
    tk.R_e = re;
    tk.b1_de = re.col(0);
    tk.b1_de_dot = dre.col(0) * sd;
    tk.b1_de_ddot = d2re.col(0) * sd * sd + dre.col(0) * sdd;

    const double pp = a0_.pp + (a1_.pp - a0_.pp) * s, pp1 = a1_.pp - a0_.pp;
    const double cp = std::cos(pp), sp = std::sin(pp);
    const Vec3 b1d{cp, sp, 0.0};
    const Vec3 b1d_s1 = pp1 * Vec3{-sp, cp, 0.0};
    const Vec3 b1d_s2 = pp1 * pp1 * Vec3{-cp, -sp, 0.0};  // pp2 = 0
    tk.b1_d = b1d;
    tk.b1_d_dot = b1d_s1 * sd;
    tk.b1_d_ddot = b1d_s2 * sd * sd + b1d_s1 * sdd;

    tk.split = a0_.split + (a1_.split - a0_.split) * s;
    return tk;
  }

  // Thrust dir -> R0 -> z-x-z of R0' R_e -> q, with the time-varying split.
  void recoverQ(const Vec3 & pcdd, const TaskSample & tk, VecN * q, Mat3 * r0) const
  {
    const Vec3 ac = pcdd + params_.g * kE3;
    const Mat3 R0 = buildR0(ac / ac.norm(), tk.b1_d);
    double a, b, c;
    zxzAngles(R0.transpose() * tk.R_e, &a, &b, &c);
    (*q) << a, tk.split * b, (1.0 - tk.split) * b, c;
    if (r0 != nullptr) {*r0 = R0;}
  }

  WbReference eval(double t) const override
  {
    const double tq = std::min(T_, std::max(0.0, t));
    Eigen::Matrix<double, 5, 3> pv;
    pc_.eval(tq, &pv);
    const TaskSample tk = task(tq);
    VecN q;
    recoverQ(pv.row(2).transpose(), tk, &q, nullptr);
    // qdot by central FD of the recovered q (endpoints one-sided)
    const double h = 1e-3;
    const double ta = std::max(0.0, tq - h), tb = std::min(T_, tq + h);
    Eigen::Matrix<double, 5, 3> pa, pb;
    pc_.eval(ta, &pa);
    pc_.eval(tb, &pb);
    VecN qa, qb;
    recoverQ(pa.row(2).transpose(), task(ta), &qa, nullptr);
    recoverQ(pb.row(2).transpose(), task(tb), &qb, nullptr);
    WbReference out;
    out.x_cd = pv.row(0).transpose();
    out.x_cd_dot = pv.row(1).transpose();
    out.x_cd_ddot = pv.row(2).transpose();
    out.x_cd_d3 = pv.row(3).transpose();
    out.x_cd_d4 = pv.row(4).transpose();
    out.q_d = q;
    out.qdot_d = (qb - qa) / (tb - ta);
    out.b1_d = tk.b1_d;
    out.b1_d_dot = tk.b1_d_dot;
    out.b1_d_ddot = tk.b1_d_ddot;
    out.r_ed = tk.r_ed;
    out.r_ed_dot = tk.r_ed_dot;
    out.r_ed_ddot = tk.r_ed_ddot;
    out.b1_de = tk.b1_de;
    out.b1_de_dot = tk.b1_de_dot;
    out.b1_de_ddot = tk.b1_de_ddot;
    return out;
  }

  // filled by the planner
  WholeBodyParams params_;
  RestSpec rest1_;
  double T_{0.0};
  Vec3 p_e0_{Vec3::Zero()}, dp_{Vec3::Zero()};
  RestAngles a0_{}, a1_{};
  Poly pc_;
  PlanDiag diag_;
};

}  // namespace

std::unique_ptr<Trajectory> StraightLineTransitionPlanner::plan(
  const VehicleModel & vehicle, const PlanRequest & req,
  const PlanOptions & o) const
{
  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();
  if (!req.rest1.has_value()) {
    throw std::runtime_error("straight_line needs a goal rest (rest1)");
  }
  const WholeBodyParams & params = vehicle.params;
  const RestSpec & rest0 = req.rest0;
  const RestSpec & rest1 = *req.rest1;
  const VecN q0 = rest0.q, q1 = rest1.q;

  const RestSpec * ends[2] = {&rest0, &rest1};
  const char * names[2] = {"start", "goal"};
  for (int e = 0; e < 2; ++e) {
    const VecN & q = ends[e]->q;
    const double be = (q(1) + q(2)) * 180.0 / M_PI;
    if (be < vehicle.beta_min_deg) {
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(1);
      m << names[e] << " fold beta = " << be << " deg < ";
      m.precision(0);
      m << vehicle.beta_min_deg
        << " deg (wrist singularity) -- z-x-z recovery is ill-posed";
      throw std::runtime_error(m.str());
    }
    for (int j = 0; j < kNumJoints; ++j) {
      if (q(j) < vehicle.q_min(j) - 1e-9 || q(j) > vehicle.q_max(j) + 1e-9) {
        throw std::runtime_error(
                std::string(names[e]) + " joints outside the working range");
      }
    }
  }

  auto traj = std::make_unique<StraightLineTrajectory>();
  traj->params_ = params;
  traj->rest1_ = rest1;
  RestAngles a0 = restAngles(rest0.phi, q0);
  RestAngles a1 = restAngles(rest1.phi, q1);
  // continuity: goal angles on the branch nearest the start
  a1.al = unwrapNear(a1.al, a0.al);
  a1.ga = unwrapNear(a1.ga, a0.ga);
  a1.pp = unwrapNear(a1.pp, a0.pp);
  traj->a0_ = a0;
  traj->a1_ = a1;

  const WbReference rr0 = restReference(params, rest0);
  const WbReference rr1 = restReference(params, rest1);
  const Vec3 p_e0 = rr0.r_ed, p_e1 = rr1.r_ed;
  const double d_pos = (p_e1 - p_e0).norm();
  const double d_base = (rest1.x_b - rest0.x_b).norm();
  const double d_move = std::max(d_pos, d_base);
  const double d_ang = std::max(
    std::max(std::abs(a1.al - a0.al), std::abs(a1.be - a0.be)),
    std::max(std::abs(a1.ga - a0.ga), std::abs(a1.pp - a0.pp)));

  const PhasePeaks & pk = phasePeaks();
  const double t_v = d_move > 0.0 ? pk.ds * d_move / o.v_max : 0.0;
  const double t_a = d_move > 0.0 ? std::sqrt(pk.d2s * d_move / o.a_max) : 0.0;
  const double t_w = d_ang > 0.0 ? pk.ds * d_ang / o.w_max : 0.0;
  const double T = std::min(o.T_max, std::max(o.T_min, std::max(t_v, std::max(t_a, t_w))));
  traj->T_ = T;
  traj->p_e0_ = p_e0;
  traj->dp_ = p_e1 - p_e0;

  // ---- Picard fixed point on a single-segment polynomial p_c --------------
  const int n = o.N;
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(n, 0.0, T);
  Eigen::Matrix<double, 3, Eigen::Dynamic> pe(3, n);
  std::vector<TaskSample> tks(n);
  for (int k = 0; k < n; ++k) {
    tks[k] = traj->task(t(k));
    pe.col(k) = tks[k].r_ed;
  }
  Poly pc = fitPoly(t, pe, 0.0, T, o.deg);
  int iterations = 0;
  Eigen::Matrix<double, 3, Eigen::Dynamic> pc_new(3, n), pc_cur(3, n);
  for (int it = 0; it < o.maxit; ++it) {
    ++iterations;
    for (int k = 0; k < n; ++k) {
      if (o.yield_hook && k % 32 == 0) {o.yield_hook();}
      Eigen::Matrix<double, 5, 3> pv;
      pc.eval(t(k), &pv);
      pc_cur.col(k) = pv.row(0).transpose();
      VecN q;
      Mat3 r0;
      traj->recoverQ(pv.row(2).transpose(), tks[k], &q, &r0);
      Vec3 r0c, r0e;
      armKinematics(q, params, &r0c, &r0e, nullptr);
      pc_new.col(k) = pe.col(k) + r0 * (r0c - r0e);
    }
    const Eigen::Matrix<double, 3, Eigen::Dynamic> target =
      (1.0 - o.relax) * pc_cur + o.relax * pc_new;
    Poly pc_next = fitPoly(t, target, 0.0, T, o.deg);
    double step = 0.0;
    for (int k = 0; k < n; ++k) {
      Eigen::Matrix<double, 5, 3> pv;
      pc_next.eval(t(k), &pv);
      step = std::max(step, (pv.row(0).transpose() - pc_cur.col(k)).norm());
    }
    pc = pc_next;
    if (step < o.tol) {break;}
  }
  traj->pc_ = pc;

  // ---- diagnostics on a fine grid ------------------------------------------
  const int nf = o.Nfine;
  double e_dyn_max = 0.0, sig_min = 1e300, peak_v = 0.0;
  VecN qmin = VecN::Constant(1e300), qmax = VecN::Constant(-1e300);
  for (int k = 0; k < nf; ++k) {
    if (o.yield_hook && k % 32 == 0) {o.yield_hook();}
    const double tf = T * k / (nf - 1);
    Eigen::Matrix<double, 5, 3> pv;
    pc.eval(tf, &pv);
    const TaskSample tk = traj->task(tf);
    VecN q;
    Mat3 r0;
    traj->recoverQ(pv.row(2).transpose(), tk, &q, &r0);
    Vec3 r0c, r0e;
    armKinematics(q, params, &r0c, &r0e, nullptr);
    e_dyn_max = std::max(
      e_dyn_max, (pv.row(0).transpose() - (tk.r_ed + r0 * (r0c - r0e))).norm());
    sig_min = std::min(sig_min, sigmaNd(q, params));
    peak_v = std::max(peak_v, pv.row(1).norm());
    qmin = qmin.cwiseMin(q);
    qmax = qmax.cwiseMax(q);
  }
  Eigen::Matrix<double, 5, 3> pv0, pv1;
  pc.eval(0.0, &pv0);
  pc.eval(T, &pv1);
  const double end_err = std::max(
    (pv0.row(0).transpose() - rr0.x_cd).norm(),
    (pv1.row(0).transpose() - rr1.x_cd).norm());

  PlanDiag & d = traj->diag_;
  d.T = T;
  d.iterations = iterations;
  d.max_dyn_defect = e_dyn_max;
  d.min_sigma_nd = sig_min;
  d.q_min_deg = qmin * 180.0 / M_PI;
  d.q_max_deg = qmax * 180.0 / M_PI;
  d.peak_com_speed = peak_v;
  d.peak_ee_speed = T > 0.0 ? pk.ds * d_pos / T : 0.0;
  d.endpoint_mismatch = end_err;
  d.solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t_start).count();

  if (d.min_sigma_nd < vehicle.sigma_nd_margin) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(3);
    m << "transition leaves the certified-safe set: min sigma_nd = "
      << d.min_sigma_nd << " < ";
    m.precision(2);
    m << vehicle.sigma_nd_margin;
    throw std::runtime_error(m.str());
  }
  for (int j = 0; j < kNumJoints; ++j) {
    if (qmin(j) < vehicle.q_min(j) - 1e-6 * M_PI / 180.0 ||
      qmax(j) > vehicle.q_max(j) + 1e-6 * M_PI / 180.0)
    {
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(1);
      m << "recovered joint path exceeds the working range: min [";
      for (int i = 0; i < kNumJoints; ++i) {
        m << d.q_min_deg(i) << (i + 1 < kNumJoints ? ", " : "");
      }
      m << "] max [";
      for (int i = 0; i < kNumJoints; ++i) {
        m << d.q_max_deg(i) << (i + 1 < kNumJoints ? ", " : "");
      }
      m << "] deg";
      throw std::runtime_error(m.str());
    }
  }
  if (d.endpoint_mismatch > 2e-3) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "CoM fit endpoint mismatch " << d.endpoint_mismatch * 1e3
      << " mm -- refusing a stepped hold handover";
    throw std::runtime_error(m.str());
  }
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "T = " << T << " s, " << iterations << " Picard iters, defect "
      << std::scientific << std::setprecision(1) << e_dyn_max << " m, sigma_nd "
      << std::fixed << std::setprecision(3) << sig_min << ", peak CoM speed "
      << std::setprecision(2) << peak_v << " m/s, solve "
      << std::setprecision(1) << d.solve_ms << " ms";
    d.summary = m.str();
  }
  return traj;
}

// ---------------------------------------------------------------------------
// flat B-spline backend, wrapped
// ---------------------------------------------------------------------------
namespace
{
class FlatTrajectory : public Trajectory
{
public:
  FlatTrajectory(std::shared_ptr<const WholeBodyParams> params, FlatPlan plan, RestSpec rest1)
  : params_(std::move(params)), plan_(std::move(plan)), rest1_(std::move(rest1))
  {
    const FlatPlanDiag & fd = plan_.diag();
    diag_.T = fd.T;
    diag_.solve_ms = fd.solve_ms;
    diag_.iterations = fd.n_dilations;
    diag_.max_dyn_defect = fd.max_dyn_defect;
    diag_.min_sigma_nd = fd.min_sigma_nd;
    diag_.peak_com_speed = fd.peak_v;
    diag_.peak_ee_speed = fd.peak_v;
    diag_.endpoint_mismatch = std::max(fd.endpoint_xc_err, fd.rest_err);
    diag_.q_min_deg = fd.q_min_deg;
    diag_.q_max_deg = fd.q_max_deg;
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "T = " << fd.T << " s (" << fd.T_binding << ", " << fd.n_dilations
      << " dilations), defect " << std::scientific << std::setprecision(1)
      << fd.max_dyn_defect << " m, sigma_nd " << std::fixed
      << std::setprecision(3) << fd.min_sigma_nd << ", peak |v| "
      << std::setprecision(2) << fd.peak_v << " m/s, tau_j "
      << fd.peak_tau_joint << " N.m, rotor " << std::setprecision(1)
      << fd.rotor_lo << "/" << fd.rotor_hi << ", solve " << fd.solve_ms << " ms";
    diag_.summary = m.str();
  }
  double duration() const override {return plan_.T();}
  WbReference eval(double t) const override {return plan_.eval(t);}
  RestSpec goalRest() const override {return rest1_;}
  const PlanDiag & diag() const override {return diag_;}

private:
  // FlatPlan keeps a pointer to the params it was planned with: own a copy.
  std::shared_ptr<const WholeBodyParams> params_;
  FlatPlan plan_;
  RestSpec rest1_;
  PlanDiag diag_;
};
}  // namespace

std::unique_ptr<Trajectory> FlatBSplineTransitionPlanner::plan(
  const VehicleModel & vehicle, const PlanRequest & req,
  const PlanOptions & o) const
{
  if (!req.rest1.has_value()) {
    throw std::runtime_error("bspline needs a goal rest (rest1)");
  }
  auto params = std::make_shared<const WholeBodyParams>(vehicle.params);
  FlatPlanOptions fo;
  fo.v_max = o.v_max;
  fo.a_max = o.a_max;
  fo.w_max = o.w_max;
  fo.dw_max = o.dw_max;
  fo.T_min = o.T_min;
  fo.T_max = o.T_max;
  fo.tau_joint_max = o.tau_joint_max;
  fo.rotor_bounds = o.rotor_bounds;
  fo.n_check = o.n_check;
  fo.sigma_nd_min = vehicle.sigma_nd_margin;
  FlatPlan plan = planFlatTransition(*params, req.rest0, *req.rest1, fo);
  return std::make_unique<FlatTrajectory>(params, std::move(plan), *req.rest1);
}

}  // namespace fsc_trajectory_planner
