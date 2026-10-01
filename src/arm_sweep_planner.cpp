// MIT License
// Copyright (c) 2026 FSC Lab
//
// The arm-sweep leg (see arm_sweep_planner.hpp): every flat output in closed
// form, mapped through wb_law's flatState(), duration searched on the full
// check.
#include "fsc_trajectory_planner/arm_sweep_planner.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fsc_trajectory_planner
{

void nonicPhase(double u, double out[5])
{
  u = std::min(1.0, std::max(0.0, u));
  // s = 126u^5 - 420u^6 + 540u^7 - 315u^8 + 70u^9, and with p = u(1-u):
  // s' = 630 p^4, s'' = 2520 p^3 (1-2u), s''' = 2520 (3p^2 - 14p^3),
  // s'''' = 15120 p (1-7p) (1-2u).
  const double u2 = u * u, u5 = u2 * u2 * u;
  const double p = u * (1.0 - u), dp = 1.0 - 2.0 * u;
  const double p2 = p * p, p3 = p2 * p;
  out[0] = u5 * (126.0 + u * (-420.0 + u * (540.0 + u * (-315.0 + u * 70.0))));
  out[1] = 630.0 * p2 * p2;
  out[2] = 2520.0 * p3 * dp;
  out[3] = 2520.0 * (3.0 * p2 - 14.0 * p3);
  out[4] = 15120.0 * p * (1.0 - 7.0 * p) * dp;
}

namespace
{

constexpr double kDeg = 180.0 / M_PI;
using XcJet = Eigen::Matrix<double, 5, 3>;
using QJet = Eigen::Matrix<double, 3, kNumJoints>;

class ArmSweepTrajectory : public Trajectory
{
public:
  ArmSweepTrajectory(
    std::shared_ptr<const WholeBodyParams> params, const RestSpec & r0,
    const RestSpec & r1, const ArmSweepOptions & o, double T)
  : params_(std::move(params)), rest1_(r1), o_(o), T_(T)
  {
    phi0_ = r0.phi;
    phi1_ = r1.phi + 2.0 * M_PI * std::round((phi0_ - r1.phi) / (2.0 * M_PI));
    Vec3 c0, c1;
    armKinematics(r0.q, *params_, &c0, nullptr, nullptr);
    armKinematics(r1.q, *params_, &c1, nullptr, nullptr);
    xc0_ = r0.x_b + Rz(r0.phi) * c0;
    xc1_ = r1.x_b + Rz(r1.phi) * c1;
    q0_ = r0.q;
    q1_ = r1.q;
    ramp_ = o.ramp_frac * T;
    omega_ = 2.0 * M_PI * o.cycles / (T - 2.0 * ramp_);
  }

  double duration() const override {return T_;}
  RestSpec goalRest() const override {return rest1_;}
  const PlanDiag & diag() const override {return diag_;}
  void setDiag(const PlanDiag & d) {diag_ = d;}

  WbReference eval(double t) const override
  {
    XcJet xc;
    Vec3 psi;
    QJet q;
    flat(t, &xc, &psi, &q);
    WbReference ref;
    flatState(*params_, xc, psi, q, &ref, nullptr);
    return ref;
  }

  // The flat outputs and the derivatives flatState() reads at elapsed t.
  void flat(double t, XcJet * xc, Vec3 * psi, QJet * q) const
  {
    const double tt = std::min(std::max(t, 0.0), T_);
    double s[5];
    nonicPhase(tt / T_, s);
    double scale = 1.0;
    const Vec3 dx = xc1_ - xc0_;
    for (int k = 0; k < 5; ++k) {
      xc->row(k) = (k == 0 ? Vec3(xc0_ + dx * s[0]) : Vec3(dx * (s[k] / scale))).transpose();
      scale *= T_;
    }
    const double dphi = phi1_ - phi0_;
    *psi = Vec3(phi0_ + dphi * s[0], dphi * s[1] / T_, dphi * s[2] / (T_ * T_));

    // plateau window
    double w = 1.0, w1 = 0.0, w2 = 0.0;
    double sw[5];
    if (tt < ramp_) {
      nonicPhase(tt / ramp_, sw);
      w = sw[0];
      w1 = sw[1] / ramp_;
      w2 = sw[2] / (ramp_ * ramp_);
    } else if (tt > T_ - ramp_) {
      nonicPhase((T_ - tt) / ramp_, sw);
      w = sw[0];
      w1 = -sw[1] / ramp_;
      w2 = sw[2] / (ramp_ * ramp_);
    }
    for (int j = 0; j < kNumJoints; ++j) {
      const double dq = q1_(j) - q0_(j);
      const double ql = q0_(j) + dq * s[0], ql1 = dq * s[1] / T_, ql2 = dq * s[2] / (T_ * T_);
      if (o_.hi(j) - o_.lo(j) <= 1e-12) {
        (*q)(0, j) = ql;
        (*q)(1, j) = ql1;
        (*q)(2, j) = ql2;
        continue;
      }
      const double c = 0.5 * (o_.lo(j) + o_.hi(j)), a = 0.5 * (o_.hi(j) - o_.lo(j));
      const double arg = omega_ * (tt - ramp_) + o_.phase(j);
      const double sj = c + a * std::sin(arg);
      const double sj1 = a * omega_ * std::cos(arg);
      const double sj2 = -a * omega_ * omega_ * std::sin(arg);
      (*q)(0, j) = (1.0 - w) * ql + w * sj;
      (*q)(1, j) = (1.0 - w) * ql1 + w * sj1 + w1 * (sj - ql);
      (*q)(2, j) = (1.0 - w) * ql2 + w * sj2 + 2.0 * w1 * (sj1 - ql1) + w2 * (sj - ql);
    }
  }

private:
  std::shared_ptr<const WholeBodyParams> params_;
  RestSpec rest1_;
  ArmSweepOptions o_;
  double T_{1.0};
  double phi0_{0.0}, phi1_{0.0};
  Vec3 xc0_{Vec3::Zero()}, xc1_{Vec3::Zero()};
  VecN q0_{VecN::Zero()}, q1_{VecN::Zero()};
  double ramp_{0.0}, omega_{0.0};
  PlanDiag diag_;
};

// The options themselves, before any duration is tried: a band outside the
// joint box can never pass, whatever T.
void validate(const VehicleModel & v, const ArmSweepOptions & o)
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(1);
  for (int j = 0; j < kNumJoints; ++j) {
    if (o.lo(j) > o.hi(j)) {
      m << "sweep band of q" << j + 1 << " is inverted: [" << o.lo(j) * kDeg << ", "
        << o.hi(j) * kDeg << "] deg";
      throw std::runtime_error(m.str());
    }
    if (o.hi(j) - o.lo(j) > 1e-12 &&
      (o.lo(j) < v.q_min(j) - 1e-9 || o.hi(j) > v.q_max(j) + 1e-9))
    {
      m << "sweep band of q" << j + 1 << " [" << o.lo(j) * kDeg << ", " << o.hi(j) * kDeg
        << "] deg leaves the joint range [" << v.q_min(j) * kDeg << ", "
        << v.q_max(j) * kDeg << "] deg";
      throw std::runtime_error(m.str());
    }
  }
  if (!(o.ramp_frac > 0.0 && o.ramp_frac < 0.5)) {
    throw std::runtime_error("sweep ramp_frac must lie in (0, 0.5)");
  }
  if (o.cycles < 1) {throw std::runtime_error("sweep cycles must be >= 1");}
  if (!(o.T_min > 0.0 && o.T_max >= o.T_min)) {
    throw std::runtime_error("sweep needs 0 < T_min <= T_max");
  }
}

// Every check on a dense grid; returns the first violated bound ("" = pass).
std::string check(
  const VehicleModel & v, const ArmSweepTrajectory & tr, const ArmSweepOptions & o,
  ArmSweepDiag * d)
{
  const WholeBodyParams & P = v.params;
  const RotorModel * rotor = o.rotor_bounds ? &v.rotor : nullptr;
  const int nc = std::max(21, o.n_check);
  double sig = 1e300, vmax = 0.0, amax = 0.0, wmax = 0.0, qdmax = 0.0, tj = 0.0,
    flo = 1e300, fhi = -1e300, vee = 0.0;
  VecN qmin = VecN::Constant(1e300), qmax = VecN::Constant(-1e300);
  for (int k = 0; k < nc; ++k) {
    const double t = tr.duration() * k / (nc - 1);
    XcJet xc;
    Vec3 psi;
    QJet qm;
    tr.flat(t, &xc, &psi, &qm);
    WbReference ref;
    FlatAux aux;
    flatState(P, xc, psi, qm, &ref, &aux);
    const VecN q = qm.row(0).transpose(), qd = qm.row(1).transpose(),
      qdd = qm.row(2).transpose();
    qmin = qmin.cwiseMin(q);
    qmax = qmax.cwiseMax(q);
    sig = std::min(sig, sigmaNd(q, P));
    vmax = std::max(vmax, ref.x_cd_dot.norm());
    amax = std::max(amax, ref.x_cd_ddot.norm());
    wmax = std::max(wmax, std::abs(psi(1)));
    qdmax = std::max(qdmax, qd.cwiseAbs().maxCoeff());
    vee = std::max(vee, ref.r_ed_dot.norm());
    const FlatInputs in = inverseInputs(P, aux, q, qd, qdd, rotor);
    tj = std::max(tj, in.tau_joint.cwiseAbs().maxCoeff());
    if (rotor) {
      flo = std::min(flo, in.rotor_force.minCoeff());
      fhi = std::max(fhi, in.rotor_force.maxCoeff());
    }
  }
  d->T = tr.duration();
  d->min_sigma_nd = sig;
  d->peak_v = vmax;
  d->peak_a = amax;
  d->peak_w = wmax;
  d->peak_qdot = qdmax;
  d->peak_tau_joint = tj;
  d->rotor_lo = rotor ? flo : 0.0;
  d->rotor_hi = rotor ? fhi : 0.0;
  d->peak_ee_speed = vee;
  d->q_min_deg = qmin * kDeg;
  d->q_max_deg = qmax * kDeg;

  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(3);
  for (int j = 0; j < kNumJoints; ++j) {
    if (qmin(j) < v.q_min(j) - 1e-6 || qmax(j) > v.q_max(j) + 1e-6) {
      m << "joint q" << j + 1 << " leaves its range: [" << d->q_min_deg(j) << ", "
        << d->q_max_deg(j) << "] deg vs [" << v.q_min(j) * kDeg << ", "
        << v.q_max(j) * kDeg << "]";
      return m.str();
    }
  }
  if (sig < o.sigma_nd_min) {
    m << "singularity margin sigma_nd = " << sig << " < " << o.sigma_nd_min;
  } else if (o.v_max > 0.0 && vmax > o.v_max) {
    m << "CoM speed " << vmax << " > " << o.v_max << " m/s";
  } else if (o.a_max > 0.0 && amax > o.a_max) {
    m << "CoM acceleration " << amax << " > " << o.a_max << " m/s^2";
  } else if (o.w_max > 0.0 && wmax > o.w_max) {
    m << "yaw rate " << wmax << " > " << o.w_max << " rad/s";
  } else if (o.qdot_max > 0.0 && qdmax > o.qdot_max) {
    m << "joint rate " << qdmax << " > " << o.qdot_max << " rad/s";
  } else if (o.tau_joint_max > 0.0 && tj > o.tau_joint_max) {
    m << "joint torque " << tj << " > " << o.tau_joint_max << " N.m";
  } else if (rotor && (flo < v.rotor.f_min || fhi > v.rotor.f_max)) {
    m << "rotor force [" << flo << ", " << fhi << "] N outside [" << v.rotor.f_min << ", "
      << v.rotor.f_max << "]";
  }
  return m.str();
}

void summarize(const ArmSweepOptions & o, ArmSweepDiag * d)
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(1);
  m << "arm sweep T = " << d->T << " s (" << d->probes << " probes), " << o.cycles << " cycles";
  for (int j = 0; j < kNumJoints; ++j) {
    if (o.hi(j) - o.lo(j) > 1e-12) {
      m << ", q" << j + 1 << " [" << o.lo(j) * kDeg << ", " << o.hi(j) * kDeg << "] deg";
    }
  }
  m << std::setprecision(2) << ", peak |v| " << d->peak_v << " m/s, |qdot| " << d->peak_qdot
    << " rad/s, |v_ee| " << d->peak_ee_speed << " m/s, tau_j " << d->peak_tau_joint
    << " N.m, rotor " << std::setprecision(1) << d->rotor_lo << "/" << d->rotor_hi
    << " N, sigma_nd " << std::setprecision(3) << d->min_sigma_nd << ", solve "
    << std::setprecision(1) << d->solve_ms << " ms";
  d->summary = m.str();
}

PlanDiag planDiagOf(const ArmSweepDiag & d)
{
  PlanDiag p;
  p.T = d.T;
  p.solve_ms = d.solve_ms;
  p.iterations = d.probes;
  p.min_sigma_nd = d.min_sigma_nd;
  p.peak_com_speed = d.peak_v;
  p.peak_ee_speed = d.peak_ee_speed;
  p.q_min_deg = d.q_min_deg;
  p.q_max_deg = d.q_max_deg;
  p.summary = d.summary;
  return p;
}

}  // namespace

std::unique_ptr<Trajectory> ArmSweepPlanner::planAt(
  const VehicleModel & v, const RestSpec & rest0, const RestSpec & rest1,
  const ArmSweepOptions & o, double T, ArmSweepDiag * diag)
{
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  ArmSweepDiag local;
  ArmSweepDiag & d = diag ? *diag : local;
  d = ArmSweepDiag{};
  validate(v, o);
  auto params = std::make_shared<const WholeBodyParams>(v.params);
  auto tr = std::make_unique<ArmSweepTrajectory>(params, rest0, rest1, o, T);
  d.violation = check(v, *tr, o, &d);
  d.probes = 1;
  d.feasible = d.violation.empty();
  d.solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  summarize(o, &d);
  if (!d.feasible) {throw std::runtime_error(d.violation);}
  tr->setDiag(planDiagOf(d));
  return tr;
}

std::unique_ptr<Trajectory> ArmSweepPlanner::plan(
  const VehicleModel & v, const RestSpec & rest0, const RestSpec & rest1,
  const ArmSweepOptions & o, ArmSweepDiag * diag)
{
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  ArmSweepDiag local;
  ArmSweepDiag & d = diag ? *diag : local;
  d = ArmSweepDiag{};
  validate(v, o);
  auto params = std::make_shared<const WholeBodyParams>(v.params);
  int probes = 0;
  auto probe = [&](double T, ArmSweepDiag * pd) {
      ++probes;
      ArmSweepTrajectory tr(params, rest0, rest1, o, T);
      return check(v, tr, o, pd);
    };

  // bracket: the shortest allowed duration, then doubling up to T_max
  double T_good = -1.0, T_bad = o.T_min;
  ArmSweepDiag probe_d;
  std::string why = probe(o.T_min, &probe_d);
  if (why.empty()) {
    T_good = o.T_min;
  } else {
    double T = o.T_min;
    while (T < o.T_max) {
      T = std::min(2.0 * T, o.T_max);
      why = probe(T, &probe_d);
      if (why.empty()) {T_good = T; break;}
      T_bad = T;
    }
  }
  if (T_good < 0.0) {
    d = probe_d;
    d.probes = probes;
    d.feasible = false;
    d.violation = why;
    d.solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
    summarize(o, &d);
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(0);
    m << why << " (even at T = " << o.T_max << " s)";
    throw std::runtime_error(m.str());
  }
  // bisect down to rel_tol on the shortest feasible duration
  if (T_good > o.T_min) {
    while (T_good - T_bad > o.rel_tol * T_good) {
      const double mid = 0.5 * (T_good + T_bad);
      if (probe(mid, &probe_d).empty()) {T_good = mid;} else {T_bad = mid;}
    }
  }
  auto tr = std::make_unique<ArmSweepTrajectory>(params, rest0, rest1, o, T_good);
  d.violation = check(v, *tr, o, &d);
  d.probes = probes + 1;
  d.feasible = d.violation.empty();
  d.solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  summarize(o, &d);
  if (!d.feasible) {throw std::runtime_error(d.violation);}   // cannot happen: T_good passed
  tr->setDiag(planDiagOf(d));
  return tr;
}

}  // namespace fsc_trajectory_planner
