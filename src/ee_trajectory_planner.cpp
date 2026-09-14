// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/ee_trajectory_planner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include "fsc_trajectory_planner/bspline_fit.hpp"
#include "fsc_trajectory_planner/kinematics.hpp"

namespace fsc_trajectory_planner
{

namespace
{

const Vec3 kE3{0.0, 0.0, 1.0};

// Minimal rotation taking e3 to the unit thrust direction t (R2_from_t.m).
Mat3 tiltRotation(const Vec3 & t)
{
  const double t1 = t(0), t2 = t(1), t3 = t(2);
  const double d = 1.0 + t3;
  Mat3 R;
  R << 1.0 - t1 * t1 / d, -t1 * t2 / d, t1,
    -t1 * t2 / d, 1.0 - t2 * t2 / d, t2,
    -t1, -t2, t3;
  return R;
}

// integral of the min-snap phase: S(u) = int_0^u sigma
double phaseIntegral(double u)
{
  u = std::min(1.0, std::max(0.0, u));
  const double u5 = std::pow(u, 5);
  return 7.0 * u5 - 14.0 * u5 * u + 10.0 * u5 * u * u - 2.5 * u5 * u * u * u;
}

// --------------------------------------------------------------- the shape
struct ShapeFrame
{
  Vec3 p0{Vec3::Zero()};        // EE position at tau = 0 (world)
  double theta{0.0};            // rotation of the local shape into the world
};

// local shape: tau -> (p, dp/dtau); the tau = 0 tangent is rotated onto +x
void localShape(const EeShape & s, double tau, Vec3 * p, Vec3 * dp)
{
  const double w = 2.0 * M_PI / s.lap_time;
  if (s.type == "circle") {
    const double sg = s.ccw ? 1.0 : -1.0;
    *p = Vec3{s.radius * std::sin(w * tau), sg * s.radius * (1.0 - std::cos(w * tau)), 0.0};
    *dp = Vec3{s.radius * w * std::cos(w * tau), sg * s.radius * w * std::sin(w * tau), 0.0};
  } else if (s.type == "figure8") {
    // p = [A sin(w tau), B sin(2 w tau)], rotated so the tau = 0 tangent
    // (A w, 2 B w) lies along +x
    const double th = -std::atan2(2.0 * s.fig8_b, s.fig8_a);
    const Mat3 R = Rz(th);
    *p = R * Vec3{s.fig8_a * std::sin(w * tau), s.fig8_b * std::sin(2.0 * w * tau), 0.0};
    *dp = R * Vec3{s.fig8_a * w * std::cos(w * tau),
      2.0 * s.fig8_b * w * std::cos(2.0 * w * tau), 0.0};
  } else {
    throw std::runtime_error("unknown EE shape '" + s.type + "' (circle | figure8)");
  }
}

// Where the run starts and which way it leaves: the local shape's tau = 0
// point goes to f.p0 and its +x tangent to the ACTUAL azimuth f.theta.
//   figure-8, or a circle with center_origin = false: the current EE point,
//     tangent along the nose (the circle then bends left/right of it);
//   circle with center_origin = true: the circle is centred on the WORLD
//     origin at the current EE height, the start is the point of that circle
//     on the current EE's bearing, the tangent is the ccw/cw tangent there.
ShapeFrame anchorFrame(const EeShape & s, const VehicleModel & v, const RestSpec & hold)
{
  Vec3 r0e;
  armKinematics(hold.q, v.params, nullptr, &r0e, nullptr);
  const Vec3 p_e0 = hold.x_b + Rz(hold.phi) * r0e;
  ShapeFrame f;
  if (s.type == "circle" && s.center_origin) {
    double th0 = std::atan2(p_e0(1), p_e0(0));
    if (std::hypot(p_e0(0), p_e0(1)) < 1e-6) {th0 = hold.phi + 0.5 * M_PI - 0.5 * M_PI;}  // on the axis: bearing = nose
    f.p0 = Vec3{s.radius * std::cos(th0), s.radius * std::sin(th0), p_e0(2)};
    f.theta = th0 + (s.ccw ? 0.5 * M_PI : -0.5 * M_PI);
  } else {
    f.p0 = p_e0;
    f.theta = hold.phi + 0.5 * M_PI;   // actual nose azimuth
  }
  return f;
}

// world EE position, tangent heading (ACTUAL azimuth) and R_e (MODEL frame)
struct EeSample
{
  Vec3 p_e;
  Vec3 dp_e;
  double psi_tan;   // actual world azimuth of the tangent
  Mat3 R_e;         // Rz(psi_tan - pi/2) Rx(beta_e)
};

EeSample eeAt(const EeShape & s, const ShapeFrame & f, double beta_e, double tau)
{
  Vec3 pl, dpl;
  localShape(s, tau, &pl, &dpl);
  const Mat3 R = Rz(f.theta);
  EeSample e;
  e.p_e = f.p0 + R * pl;
  e.dp_e = R * dpl;
  e.psi_tan = std::atan2(e.dp_e(1), e.dp_e(0));
  e.R_e = Rz(e.psi_tan - 0.5 * M_PI) * Rx(beta_e);
  return e;
}

// ------------------------------------------------------- the time profile
struct TimeProfile
{
  double s{1.0}, Tr{0.0}, Tl{0.0}, T_total{0.0}, tau_end{0.0};
  // tau, dtau/dt, d2tau/dt2
  void at(double t, double * tau, double * d1, double * d2) const
  {
    t = std::min(std::max(t, 0.0), T_total);
    double sv, ds, d2s;
    if (t < Tr) {
      const double u = t / Tr;
      minsnap3(u, &sv, &ds, &d2s);
      *tau = s * Tr * phaseIntegral(u);
      *d1 = s * sv;
      *d2 = s * ds / Tr;
    } else if (t <= Tr + Tl) {
      *tau = 0.5 * s * Tr + s * (t - Tr);
      *d1 = s;
      *d2 = 0.0;
    } else {
      const double u = (T_total - t) / Tr;
      minsnap3(u, &sv, &ds, &d2s);
      *tau = tau_end - s * Tr * phaseIntegral(u);
      *d1 = s * sv;
      *d2 = -s * ds / Tr;
    }
  }
};

TimeProfile makeProfile(const EeShape & shape, const EeTrajectoryOptions & o)
{
  TimeProfile tp;
  tp.s = o.time_scale;
  tp.Tr = o.ramp_time;
  tp.tau_end = shape.laps * shape.lap_time;
  const double lap_phase = tp.tau_end - tp.s * tp.Tr;   // phase left for the laps
  if (tp.s <= 0.0) {throw std::runtime_error("time scale must be positive");}
  if (lap_phase < 0.05 * tp.tau_end) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "time scale " << tp.s << " too large for " << shape.laps << " lap(s) of "
      << shape.lap_time << " s with " << tp.Tr << " s ramps (the ramps alone use "
      << tp.s * tp.Tr << " s of the " << tp.tau_end << " s of phase)";
    throw std::runtime_error(m.str());
  }
  tp.Tl = lap_phase / tp.s;
  tp.T_total = 2.0 * tp.Tr + tp.Tl;
  return tp;
}

double q2At(const EeTrajectoryOptions & o, const EeShape & shape, double tau)
{
  const double P = o.q2_period_s > 0.0 ? o.q2_period_s : shape.lap_time;
  return (o.q2_center_deg + o.q2_amp_deg * std::sin(2.0 * M_PI * tau / P)) * M_PI / 180.0;
}

// ------------------------------------------------------------ the result
class EeTrajectory : public Trajectory
{
public:
  double duration() const override {return tp_.T_total;}
  RestSpec goalRest() const override {return rest0_;}
  const PlanDiag & diag() const override {return pdiag_;}

  WbReference eval(double t) const override
  {
    WbReference ref;
    FlatAux aux;
    evalWithAux(t, &ref, &aux, nullptr);
    return ref;
  }

  // reference + the flat-map intermediates + the fitted q derivatives
  void evalWithAux(
    double t, WbReference * ref, FlatAux * aux, Eigen::Matrix<double, 3, kNumJoints> * qmat) const
  {
    t = std::min(std::max(t, 0.0), tp_.T_total);
    Eigen::MatrixXd xc, ps, q;
    xc_.evalAll(t, 4, &xc);
    psi_.evalAll(t, 2, &ps);
    q_.evalAll(t, 2, &q);
    Eigen::Matrix<double, 5, 3> xcm = xc;
    const Vec3 psv = ps.col(0);
    Eigen::Matrix<double, 3, kNumJoints> qm = q;
    flatState(params_, xcm, psv, qm, ref, aux);
    if (qmat) {*qmat = qm;}
  }

  EeSample prescribed(double t) const
  {
    double tau, d1, d2;
    tp_.at(t, &tau, &d1, &d2);
    return eeAt(shape_, frame_, beta_e_, tau);
  }

  WholeBodyParams params_;
  EeShape shape_;
  ShapeFrame frame_;
  double beta_e_{0.0};
  TimeProfile tp_;
  RestSpec rest0_;
  BSplineFit xc_, psi_, q_;
  PlanDiag pdiag_;
  EeTrajectoryDiag ediag_;
};

}  // namespace

// =========================================================================
RestSpec EeTrajectoryPlanner::startRest(
  const VehicleModel & v, const RestSpec & hold, const EeShape & shape,
  const EeTrajectoryOptions & o)
{
  const ShapeFrame f = anchorFrame(shape, v, hold);
  const double beta_e = o.ee_fold_deg * M_PI / 180.0;
  const double q2 = q2At(o, shape, 0.0);
  RestSpec r;
  r.phi = f.theta - 0.5 * M_PI;           // nose along the tau = 0 tangent
  r.q << 0.0, q2, beta_e - q2, 0.0;
  Vec3 r0e0;
  armKinematics(r.q, v.params, nullptr, &r0e0, nullptr);
  r.x_b = f.p0 - Rz(r.phi) * r0e0;        // EE on the run's start point, at its fold
  return r;
}

std::unique_ptr<Trajectory> EeTrajectoryPlanner::plan(
  const VehicleModel & v, const RestSpec & hold, const EeShape & shape,
  const EeTrajectoryOptions & o, EeTrajectoryDiag * dg)
{
  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();
  EeTrajectoryDiag local;
  EeTrajectoryDiag & d = dg ? *dg : local;
  d = EeTrajectoryDiag{};
  const WholeBodyParams & P = v.params;

  auto traj = std::make_unique<EeTrajectory>();
  traj->params_ = P;
  traj->shape_ = shape;
  traj->beta_e_ = o.ee_fold_deg * M_PI / 180.0;
  traj->tp_ = makeProfile(shape, o);
  const TimeProfile & tp = traj->tp_;
  d.T_total = tp.T_total;
  d.T_lap = shape.lap_time / tp.s;
  d.ramp_time = tp.Tr;
  d.laps = shape.laps;
  d.s = tp.s;

  traj->frame_ = anchorFrame(shape, v, hold);
  traj->rest0_ = startRest(v, hold, shape, o);
  const WbReference rest_ref = restReference(P, traj->rest0_);

  // ---- grid and prescribed data --------------------------------------------
  const int N = static_cast<int>(std::floor(tp.T_total * o.sample_rate)) + 1;
  Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(N, 0.0, tp.T_total);
  std::vector<EeSample> ee(N);
  std::vector<double> q2(N);
  for (int k = 0; k < N; ++k) {
    double tau, d1, d2;
    tp.at(t(k), &tau, &d1, &d2);
    ee[k] = eeAt(shape, traj->frame_, traj->beta_e_, tau);
    q2[k] = q2At(o, shape, tau);
  }
  const int spans = std::max(8, static_cast<int>(std::ceil(tp.T_total * o.spans_per_s)));
  traj->xc_ = BSplineFit(7, spans, tp.T_total, 3);
  traj->psi_ = BSplineFit(5, spans, tp.T_total, 1);
  traj->q_ = BSplineFit(5, spans, tp.T_total, kNumJoints);
  // pinned rest ends: 5 coincident points zero x_c's derivatives 1..4, 3
  // zero the yaw's and the joints' velocity and acceleration -- the run
  // starts and ends at rest BY CONSTRUCTION (the flat planner's pin_xc /
  // pin_ang), and the Picard loop cannot ring at the clamped ends.
  traj->xc_.prepareFit(t, 5, 5);
  traj->psi_.prepareFit(t, 3, 3);
  traj->q_.prepareFit(t, 3, 3);

  // ---- Picard on the thrust direction ------------------------------------
  // r_k <- normalize(xdd_c(t_k) + g e3), x_c from the chain at the attitude
  // r_k implies: the feasibility residual of feas_cost_redundant driven to 0.
  std::vector<Vec3> r(N, kE3);
  Eigen::MatrixXd Xc(N, 3), Psi(N, 1), Q(N, kNumJoints);
  double psi_prev = traj->rest0_.phi, gam_prev = 0.0;
  double step = 0.0;
  int it = 0;
  for (it = 0; it < o.maxit; ++it) {
    psi_prev = traj->rest0_.phi;
    gam_prev = 0.0;
    for (int k = 0; k < N; ++k) {
      // first the MATLAB decomposition (R2 = minimal tilt, zxz of R2^T R_e)
      // for the drone yaw, then refine it onto the LAW's attitude construction
      // build(b3, [cos psi, sin psi, 0]) so that the reference the fitted
      // flat outputs produce through flatState is exactly this attitude:
      // with q1 = 0 the drone's x-axis must be orthogonal to the EE z-axis.
      const Mat3 R2 = tiltRotation(r[k]);
      double psi, beta, gam;
      zxzAngles(R2.transpose() * ee[k].R_e, &psi, &beta, &gam);
      if (std::sin(beta) < 1e-9) {psi = psi_prev;}
      psi = unwrapNear(psi, psi_prev);
      const Vec3 ez_e = ee[k].R_e.col(2);
      auto g = [&](double ps) {
          return buildR0(r[k], Vec3{std::cos(ps), std::sin(ps), 0.0}).col(0).dot(ez_e);
        };
      for (int nit = 0; nit < 4; ++nit) {
        const double h = 1e-6;
        const double g0 = g(psi), dg = (g(psi + h) - g(psi - h)) / (2.0 * h);
        if (std::abs(dg) < 1e-12) {break;}
        const double dpsi = -g0 / dg;
        psi += std::max(-0.5, std::min(0.5, dpsi));
        if (std::abs(dpsi) < 1e-12) {break;}
      }
      const Mat3 R0 = buildR0(r[k], Vec3{std::cos(psi), std::sin(psi), 0.0});
      double q1_res;
      zxzAngles(R0.transpose() * ee[k].R_e, &q1_res, &beta, &gam);
      gam = unwrapNear(gam, gam_prev);
      psi_prev = psi;
      gam_prev = gam;
      VecN q;
      q << 0.0, q2[k], beta - q2[k], gam;
      Vec3 r0c, r0e;
      armKinematics(q, P, &r0c, &r0e, nullptr);
      Xc.row(k) = (ee[k].p_e + R0 * (r0c - r0e)).transpose();
      Psi(k, 0) = psi;
      Q.row(k) = q.transpose();
    }
    traj->xc_.fit(Xc);
    step = 0.0;
    Eigen::MatrixXd der;
    for (int k = 0; k < N; ++k) {
      traj->xc_.evalAll(t(k), 2, &der);
      const Vec3 ac = der.row(2).transpose() + P.g * kE3;
      const Vec3 rn = ac / ac.norm();
      const double dk = (rn - r[k]).norm();
      step = std::max(step, dk);
      Vec3 rr = r[k] + o.relax * (rn - r[k]);
      r[k] = rr / rr.norm();
    }
    if (step < o.tol) {++it; break;}
  }
  d.iterations = it;
  // final fits of the other flat outputs on the converged attitude
  traj->psi_.fit(Psi);
  traj->q_.fit(Q);

  // ---- checks on a dense grid ----------------------------------------------
  const int nc = o.n_check > 0 ? o.n_check : N;
  double sig_min = 1e300, beta_min = 1e300, vmax = 0.0, amax = 0.0, wmax = 0.0,
    qdmax = 0.0, tjmax = 0.0, flo = 1e300, fhi = -1e300, pe_err = 0.0, re_err = 0.0,
    defect = 0.0;
  VecN qmin = VecN::Constant(1e300), qmax = VecN::Constant(-1e300);
  const RotorModel * rotor = o.rotor_bounds ? &v.rotor : nullptr;
  for (int k = 0; k < nc; ++k) {
    const double tk = tp.T_total * k / (nc - 1);
    WbReference ref;
    FlatAux aux;
    Eigen::Matrix<double, 3, kNumJoints> qm;
    traj->evalWithAux(tk, &ref, &aux, &qm);
    const VecN q = qm.row(0).transpose(), qd = qm.row(1).transpose(),
      qdd = qm.row(2).transpose();
    qmin = qmin.cwiseMin(q);
    qmax = qmax.cwiseMax(q);
    beta_min = std::min(beta_min, q(1) + q(2));
    sig_min = std::min(sig_min, sigmaNd(q, P));
    vmax = std::max(vmax, ref.x_cd_dot.norm());
    amax = std::max(amax, ref.x_cd_ddot.norm());
    Eigen::MatrixXd ps;
    traj->psi_.evalAll(tk, 1, &ps);
    wmax = std::max(wmax, std::abs(ps(1, 0)));
    qdmax = std::max(qdmax, qd.cwiseAbs().maxCoeff());
    const FlatInputs in = inverseInputs(P, aux, q, qd, qdd, rotor);
    tjmax = std::max(tjmax, in.tau_joint.cwiseAbs().maxCoeff());
    if (rotor) {
      flo = std::min(flo, in.rotor_force.minCoeff());
      fhi = std::max(fhi, in.rotor_force.maxCoeff());
    }
    // FK round trip against the prescribed EE pose (position + attitude)
    const EeSample pre = traj->prescribed(tk);
    pe_err = std::max(pe_err, (ref.r_ed - pre.p_e).norm());
    Mat3 Re;
    armKinematics(q, P, nullptr, nullptr, &Re);
    const Mat3 R_fk = aux.R0 * Re;
    const double c = std::min(1.0, std::max(-1.0, ((pre.R_e.transpose() * R_fk).trace() - 1.0) / 2.0));
    re_err = std::max(re_err, std::acos(c));
    // compatibility residual of the fitted CoM against its own thrust axis
    Vec3 r0c, r0e;
    armKinematics(q, P, &r0c, &r0e, nullptr);
    defect = std::max(defect, (ref.x_cd - (pre.p_e + aux.R0 * (r0c - r0e))).norm());
  }
  {
    const WbReference r0 = traj->eval(0.0), r1 = traj->eval(tp.T_total);
    d.rest_err = std::max(
      std::max((r0.x_cd - rest_ref.x_cd).norm(), (r1.x_cd - rest_ref.x_cd).norm()),
      std::max(r0.x_cd_dot.norm(), r1.x_cd_dot.norm()));
  }
  d.max_dyn_defect = defect;
  d.ee_pos_err_max = pe_err;
  d.ee_rot_err_max_deg = re_err * 180.0 / M_PI;
  d.min_sigma_nd = sig_min;
  d.min_beta_deg = beta_min * 180.0 / M_PI;
  d.peak_v = vmax;
  d.peak_a = amax;
  d.peak_w = wmax;
  d.peak_qdot = qdmax;
  d.peak_tau_joint = tjmax;
  d.rotor_lo = rotor ? flo : 0.0;
  d.rotor_hi = rotor ? fhi : 0.0;
  d.q_min_deg = qmin * 180.0 / M_PI;
  d.q_max_deg = qmax * 180.0 / M_PI;
  d.solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t_start).count();

  std::ostringstream viol;
  viol.setf(std::ios::fixed);
  viol.precision(3);
  if (d.min_beta_deg < o.beta_min_deg) {
    viol << "fold beta = " << d.min_beta_deg << " deg < " << o.beta_min_deg
         << " (wrist singularity: joints 1 and 4 align; a level end-effector "
         << "is this arm's singular pose)";
  }
  if (viol.str().empty() && sig_min < o.sigma_nd_min) {
    viol << "singularity margin sigma_nd = " << sig_min << " < " << o.sigma_nd_min;
  }
  for (int j = 0; j < kNumJoints && viol.str().empty(); ++j) {
    if (qmin(j) < v.q_min(j) - 1e-6 || qmax(j) > v.q_max(j) + 1e-6) {
      viol << "joint q" << (j + 1) << " leaves its range: [" << d.q_min_deg(j) << ", "
           << d.q_max_deg(j) << "] deg vs [" << v.q_min(j) * 180.0 / M_PI << ", "
           << v.q_max(j) * 180.0 / M_PI << "]";
    }
  }
  if (viol.str().empty() && vmax > o.v_max) {viol << "CoM speed " << vmax << " > " << o.v_max << " m/s";}
  if (viol.str().empty() && amax > o.a_max) {viol << "CoM acceleration " << amax << " > " << o.a_max << " m/s^2";}
  if (viol.str().empty() && wmax > o.w_max) {viol << "yaw rate " << wmax << " > " << o.w_max << " rad/s";}
  if (viol.str().empty() && qdmax > o.qdot_max) {viol << "joint rate " << qdmax << " > " << o.qdot_max << " rad/s";}
  if (viol.str().empty() && o.tau_joint_max > 0.0 && tjmax > o.tau_joint_max) {
    viol << "joint torque " << tjmax << " > " << o.tau_joint_max << " N.m";
  }
  if (viol.str().empty() && rotor && (flo < v.rotor.f_min || fhi > v.rotor.f_max)) {
    viol << "rotor force [" << flo << ", " << fhi << "] N outside [" << v.rotor.f_min << ", "
         << v.rotor.f_max << "]";
  }
  if (viol.str().empty() && d.rest_err > 5e-3) {
    viol << "run does not start/end at rest (" << d.rest_err * 1e3 << " mm)";
  }
  if (viol.str().empty() && pe_err > 0.02) {
    viol << "EE FK round trip " << pe_err * 1e3 << " mm off the prescribed curve";
  }
  d.violation = viol.str();
  d.feasible = d.violation.empty();
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << shape.type << " x" << shape.laps << ", s = " << tp.s << ", T = " << tp.T_total
      << " s (lap " << d.T_lap << " s), " << it << " Picard iters, defect "
      << std::scientific << std::setprecision(1) << defect << " m, EE err " << std::fixed
      << std::setprecision(1) << pe_err * 1e3 << " mm / " << d.ee_rot_err_max_deg
      << " deg, sigma_nd " << std::setprecision(3) << sig_min << ", peaks |v| "
      << std::setprecision(2) << vmax << " |a| " << amax << " |qdot| " << qdmax
      << " tau_j " << tjmax << ", rotor " << std::setprecision(0) << flo << "/" << fhi
      << ", solve " << std::setprecision(1) << d.solve_ms << " ms";
    if (!d.feasible) {m << " -- INFEASIBLE: " << d.violation;}
    d.summary = m.str();
  }
  traj->ediag_ = d;
  traj->pdiag_.T = tp.T_total;
  traj->pdiag_.solve_ms = d.solve_ms;
  traj->pdiag_.iterations = it;
  traj->pdiag_.max_dyn_defect = defect;
  traj->pdiag_.min_sigma_nd = sig_min;
  traj->pdiag_.peak_com_speed = vmax;
  traj->pdiag_.peak_ee_speed = vmax;
  traj->pdiag_.endpoint_mismatch = d.rest_err;
  traj->pdiag_.q_min_deg = d.q_min_deg;
  traj->pdiag_.q_max_deg = d.q_max_deg;
  traj->pdiag_.summary = d.summary;
  if (!d.feasible) {throw std::runtime_error(d.violation);}
  return traj;
}

double EeTrajectoryPlanner::maxTimeScale(
  const VehicleModel & v, const RestSpec & hold, const EeShape & shape,
  const EeTrajectoryOptions & o, double s_hi, double rel_tol)
{
  // the ramps must leave phase for the laps (see makeProfile)
  const double s_cap = 0.9 * shape.laps * shape.lap_time / std::max(o.ramp_time, 1e-3);
  s_hi = std::min(s_hi, s_cap);
  auto feasible = [&](double s) {
      EeTrajectoryOptions oo = o;
      oo.time_scale = s;
      oo.n_check = 0;
      EeTrajectoryDiag dd;
      try {
        plan(v, hold, shape, oo, &dd);
        return true;
      } catch (const std::exception &) {
        return false;
      }
    };
  double lo = 0.0, hi = s_hi;
  // find a feasible lower bracket, halving from s_hi
  double probe = std::min(1.0, s_hi);
  while (probe > 0.02 && !feasible(probe)) {probe *= 0.5;}
  if (probe <= 0.02) {return 0.0;}
  lo = probe;
  // grow until infeasible or the cap
  while (lo < s_hi) {
    const double nxt = std::min(s_hi, lo * 1.6);
    if (feasible(nxt)) {
      lo = nxt;
      if (nxt >= s_hi) {return s_hi;}
    } else {
      hi = nxt;
      break;
    }
  }
  while ((hi - lo) / hi > rel_tol) {
    const double mid = 0.5 * (lo + hi);
    if (feasible(mid)) {lo = mid;} else {hi = mid;}
  }
  return lo;
}

}  // namespace fsc_trajectory_planner
