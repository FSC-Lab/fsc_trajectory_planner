// MIT License
// Copyright (c) 2026 FSC Lab
//
// Push-and-pull goals and steps (see push_pull.hpp).
#include "fsc_trajectory_planner/push_pull.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "fsc_trajectory_planner/arm_sweep_planner.hpp"   // nonicPhase

namespace fsc_trajectory_planner
{

namespace
{

constexpr double kDeg = 180.0 / M_PI;

double wrapAngle(double a) {return std::atan2(std::sin(a), std::cos(a));}

void checkPose(const VehicleModel & v, const VecN & q, const char * what, int step)
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(1);
  for (int j = 0; j < kNumJoints; ++j) {
    if (q(j) < v.q_min(j) - 1e-9 || q(j) > v.q_max(j) + 1e-9) {
      m << pushPullStepName(step) << ": " << what << " q" << j + 1 << " = " << q(j) * kDeg
        << " deg is outside [" << v.q_min(j) * kDeg << ", " << v.q_max(j) * kDeg << "]";
      throw std::runtime_error(m.str());
    }
  }
  const double s = sigmaNd(q, v.params);
  if (s < v.sigma_nd_margin) {
    m.precision(3);
    m << pushPullStepName(step) << ": " << what << " sigma_nd = " << s << " < "
      << v.sigma_nd_margin << " -- too close to a singularity";
    throw std::runtime_error(m.str());
  }
}

std::unique_ptr<Trajectory> transitionLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const RestSpec & a, const RestSpec & b)
{
  PlanRequest req;
  req.rest0 = a;
  req.rest1 = b;
  return transition.plan(v, req, opts);
}

// The options for a rest-to-rest move of `dist` metres that must take at
// least T seconds: T_min = T and the translational bounds opened just enough
// for it (a min-snap move of d in T peaks at 2.19 d / T and 7.51 d / T^2; 30 %
// margin for the B-spline's own profile). The rotor and joint-torque bounds
// still apply. T <= 0: the options unchanged.
PlanOptions timedOptions(const PlanOptions & opts, double dist, double T)
{
  PlanOptions o = opts;
  if (T > 0.0) {
    o.T_min = T;
    o.v_max = std::max(o.v_max, 1.3 * 2.19 * std::abs(dist) / T);
    o.a_max = std::max(o.a_max, 1.3 * 7.51 * std::abs(dist) / (T * T));
  }
  return o;
}

// A TIMED static hold (HoldTrajectory has no duration): the settle before the
// push moves.
class PauseTrajectory : public Trajectory
{
public:
  PauseTrajectory(const WholeBodyParams & p, const RestSpec & rest, double T)
  : rest_(rest), ref_(restReference(p, rest)), T_(T)
  {
    diag_.T = T;
    diag_.min_sigma_nd = sigmaNd(rest.q, p);
    diag_.q_min_deg = rest.q * kDeg;
    diag_.q_max_deg = diag_.q_min_deg;
    std::ostringstream m;
    m << "settle " << std::fixed << std::setprecision(1) << T << " s";
    diag_.summary = m.str();
  }
  double duration() const override {return T_;}
  WbReference eval(double) const override {return ref_;}
  RestSpec goalRest() const override {return rest_;}
  const PlanDiag & diag() const override {return diag_;}
  bool timeScalable() const override {return true;}   // static

private:
  RestSpec rest_;
  WbReference ref_;
  double T_;
  PlanDiag diag_;
};

// The nose (actual body +x = model +y) of a model heading, world, horizontal.
Vec3 noseOf(double phi) {return Vec3{-std::sin(phi), std::cos(phi), 0.0};}

// THE TURN (2026-10-15): the rest `from` rotated about the vertical axis
// through `axis` by s(t) = angle * nonic(t / T) -- a rigid rotation of the
// whole vehicle, so the CoM, the base and the claw all move on circles about
// the axis and the heading turns with them; the arm is held. The flat outputs
// (x_c, psi, q) and the derivatives flatState() reads are exact: with f(t) =
// e^{i s(t)}, the complex-plane circle c(t) = c0 f(t) and its first four
// derivatives follow from s', s'', s''', s'''' (Faa di Bruno on the
// exponential). Zero velocity through snap at both ends (nonic phase).
class TurnTrajectory : public Trajectory
{
public:
  TurnTrajectory(const WholeBodyParams & p, const RestSpec & from, const Vec3 & axis, double angle, double T)
  : p_(p), from_(from), axis_(axis), angle_(angle), T_(T)
  {
    Vec3 r0c;
    armKinematics(from.q, p, &r0c, nullptr, nullptr);
    const Vec3 xc0 = from.x_b + Rz(from.phi) * r0c;
    c0_ = std::complex<double>(xc0(0) - axis(0), xc0(1) - axis(1));
    z_ = xc0(2);
    goal_ = from;
    goal_.phi = from.phi + angle;
    goal_.x_b = axis + Rz(angle) * (from.x_b - axis);
    goal_.x_b(2) = from.x_b(2);
    diag_.T = T;
    diag_.min_sigma_nd = sigmaNd(from.q, p);
    diag_.q_min_deg = from.q * kDeg;
    diag_.q_max_deg = diag_.q_min_deg;
    // peak rates: a nonic phase peaks at 2.4609 x the mean rate and 5.6 x the
    // mean-rate^2 / span in acceleration
    const double w_peak = 2.4609 * std::abs(angle) / T;
    diag_.peak_com_speed = w_peak * std::abs(c0_);
    Vec3 r0e;
    armKinematics(from.q, p, nullptr, &r0e, nullptr);
    const Vec3 claw0 = from.x_b + Rz(from.phi) * r0e;
    diag_.peak_ee_speed = w_peak * std::hypot(claw0(0) - axis(0), claw0(1) - axis(1));
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "turn " << std::setprecision(0) << angle * kDeg << " deg about the axis in "
      << std::setprecision(1) << T << " s (claw radius " << std::setprecision(3)
      << std::hypot(claw0(0) - axis(0), claw0(1) - axis(1)) << " m, CoM radius " << std::abs(c0_)
      << " m, peak yaw rate " << std::setprecision(1) << w_peak * kDeg << " deg/s, peak CoM speed "
      << std::setprecision(3) << diag_.peak_com_speed << " m/s)";
    diag_.summary = m.str();
  }
  double duration() const override {return T_;}
  RestSpec goalRest() const override {return goal_;}
  const PlanDiag & diag() const override {return diag_;}

  WbReference eval(double t) const override
  {
    double s[5];
    phase(t, s);
    return fromAngle(s);
  }

  // PATH FOLLOWING (2026-10-18): the same rotation with its schedule run at a
  // governed progress tau(t). With S(tau) the scheduled angle and r = {tau',
  // tau'', tau''', tau''''} the wall-time rates of the progress, Faa di Bruno
  // gives the wall-time derivatives of s(t) = S(tau(t)) exactly:
  //   s'    = S1 r1
  //   s''   = S2 r1^2 + S1 r2
  //   s'''  = S3 r1^3 + 3 S2 r1 r2 + S1 r3
  //   s'''' = S4 r1^4 + 6 S3 r1^2 r2 + S2 (3 r2^2 + 4 r1 r3) + S1 r4
  // so the reference stays the rigid rotation, compatible through snap,
  // whatever the progress does. r = {1, 0, 0, 0} is eval(tau) bit for bit.
  bool timeScalable() const override {return true;}
  WbReference evalScaled(double tau, const std::array<double, 4> & r) const override
  {
    double S[5];
    phase(tau, S);
    if (tau <= 0.0 || tau >= T_) {
      for (int k = 1; k < 5; ++k) {S[k] = 0.0;}   // clamped: at rest
    }
    const double r1 = r[0], r2 = r[1], r3 = r[2], r4 = r[3];
    double s[5];
    s[0] = S[0];
    s[1] = S[1] * r1;
    s[2] = S[2] * r1 * r1 + S[1] * r2;
    s[3] = S[3] * r1 * r1 * r1 + 3.0 * S[2] * r1 * r2 + S[1] * r3;
    s[4] = S[4] * r1 * r1 * r1 * r1 + 6.0 * S[3] * r1 * r1 * r2 +
      S[2] * (3.0 * r2 * r2 + 4.0 * r1 * r3) + S[1] * r4;
    return fromAngle(s);
  }

  // The scheduled rotation at time t: s[0] [rad] and its first four time
  // derivatives (nonic rest-to-rest over T).
  void phase(double t, double s[5]) const
  {
    const double tt = std::min(std::max(t, 0.0), T_);
    double ph[5];
    nonicPhase(tt / T_, ph);
    double scale = 1.0;
    for (int k = 0; k < 5; ++k) {
      s[k] = angle_ * ph[k] / scale;
      scale *= T_;
    }
  }

private:
  // The rigid rotation by s[0] about the axis, with its derivatives s[1..4].
  WbReference fromAngle(const double s[5]) const
  {
    using C = std::complex<double>;
    const C i(0.0, 1.0);
    const C f = std::exp(i * s[0]);
    const double s1 = s[1], s2 = s[2], s3 = s[3], s4 = s[4];
    const C d[5] = {
      f,
      i * s1 * f,
      (i * s2 - s1 * s1) * f,
      (i * s3 - 3.0 * s1 * s2 - i * s1 * s1 * s1) * f,
      (i * s4 - 4.0 * s1 * s3 - 3.0 * s2 * s2 - i * 6.0 * s1 * s1 * s2 + s1 * s1 * s1 * s1) * f};
    Eigen::Matrix<double, 5, 3> xc;
    for (int k = 0; k < 5; ++k) {
      const C ck = c0_ * d[k];
      xc(k, 0) = ck.real() + (k == 0 ? axis_(0) : 0.0);
      xc(k, 1) = ck.imag() + (k == 0 ? axis_(1) : 0.0);
      xc(k, 2) = k == 0 ? z_ : 0.0;
    }
    const Vec3 psi(from_.phi + s[0], s1, s2);
    Eigen::Matrix<double, 3, kNumJoints> q = Eigen::Matrix<double, 3, kNumJoints>::Zero();
    q.row(0) = from_.q.transpose();
    WbReference ref;
    flatState(p_, xc, psi, q, &ref, nullptr);
    return ref;
  }

  WholeBodyParams p_;
  RestSpec from_, goal_;
  Vec3 axis_;
  double angle_, T_;
  std::complex<double> c0_;
  double z_{0.0};
  PlanDiag diag_;
};

// THE CRUISE (2026-10-17, PushPullConfig::push_ramp): the whole vehicle
// translates by s(t) along `dir` with the arm and the heading held, x_c(t) =
// x_c0 + s(t) dir. The SPEED rises on a septic smoothstep S(u) = 35u^4 -
// 84u^5 + 70u^6 - 20u^7 (u = t / T_r), holds V, and falls on its mirror image;
// S', S'' and S''' vanish at both ends of a ramp, so s'..s'''' (everything
// flatState() reads) are continuous at every join and zero at both ends of the
// leg, the smoothness of the nonic rest-to-rest. Each ramp covers V T_r / 2,
// so V = d / (T - T_r).
class CruiseTrajectory : public Trajectory
{
public:
  CruiseTrajectory(
    const WholeBodyParams & p, const RestSpec & from, const Vec3 & dir, double dist, double T,
    double Tr)
  : p_(p), from_(from), dir_(dir), d_(dist), T_(T), Tr_(Tr), V_(dist / (T - Tr))
  {
    Vec3 r0c;
    armKinematics(from.q, p, &r0c, nullptr, nullptr);
    xc0_ = from.x_b + Rz(from.phi) * r0c;
    goal_ = from;
    goal_.x_b += dist * dir;
    diag_.T = T;
    diag_.min_sigma_nd = sigmaNd(from.q, p);
    diag_.q_min_deg = from.q * kDeg;
    diag_.q_max_deg = diag_.q_min_deg;
    diag_.peak_com_speed = std::abs(V_);
    diag_.peak_ee_speed = std::abs(V_);
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m << "cruise " << std::setprecision(3) << dist << " m in " << std::setprecision(1) << T
      << " s: " << Tr << " s ramps, " << std::setprecision(1) << std::abs(V_) * 1e3 << " mm/s for "
      << T - 2.0 * Tr << " s (peak acceleration " << 2.1875 * std::abs(V_) / Tr * 1e3
      << " mm/s^2)";
    diag_.summary = m.str();
  }
  double duration() const override {return T_;}
  RestSpec goalRest() const override {return goal_;}
  const PlanDiag & diag() const override {return diag_;}

  WbReference eval(double t) const override
  {
    const double tt = std::min(std::max(t, 0.0), T_);
    double s[5];
    if (tt <= Tr_) {
      rampUp(tt, s);
    } else if (tt >= T_ - Tr_) {
      // s(t) = d - g(T - t), g the ramp-up: the odd derivatives keep their
      // sign, the even ones flip
      rampUp(T_ - tt, s);
      s[0] = d_ - s[0];
      s[2] = -s[2];
      s[4] = -s[4];
    } else {
      s[0] = V_ * (0.5 * Tr_ + (tt - Tr_));
      s[1] = V_;
      s[2] = s[3] = s[4] = 0.0;
    }
    Eigen::Matrix<double, 5, 3> xc;
    for (int k = 0; k < 5; ++k) {
      xc.row(k) = (k == 0 ? Vec3(xc0_ + dir_ * s[0]) : Vec3(dir_ * s[k])).transpose();
    }
    const Vec3 psi(from_.phi, 0.0, 0.0);
    Eigen::Matrix<double, 3, kNumJoints> q = Eigen::Matrix<double, 3, kNumJoints>::Zero();
    q.row(0) = from_.q.transpose();
    WbReference ref;
    flatState(p_, xc, psi, q, &ref, nullptr);
    return ref;
  }

private:
  // the ramp-up at time tau in [0, T_r]: s, s', s'', s''', s''''
  void rampUp(double tau, double s[5]) const
  {
    const double u = std::min(std::max(tau / Tr_, 0.0), 1.0);
    const double u2 = u * u, u3 = u2 * u, w = 1.0 - u;
    s[0] = V_ * Tr_ * u2 * u3 * (7.0 + u * (-14.0 + u * (10.0 - 2.5 * u)));   // V T_r int S
    s[1] = V_ * u2 * u2 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u)));     // V S
    s[2] = V_ / Tr_ * 140.0 * u3 * w * w * w;                               // V S' / T_r
    s[3] = V_ / (Tr_ * Tr_) * 420.0 * u2 * w * w * (1.0 - 2.0 * u);
    s[4] = V_ / (Tr_ * Tr_ * Tr_) * 840.0 * u * w * (1.0 + u * (-5.0 + 5.0 * u));
  }

  WholeBodyParams p_;
  RestSpec from_, goal_;
  Vec3 dir_, xc0_;
  double d_, T_, Tr_, V_;
  PlanDiag diag_;
};

// A rest's checks on its own: fence, and the table at that one pose.
void checkRest(
  const VehicleModel & v, const PushPullConfig & c, const RestSpec & r, const char * what)
{
  const std::string why = c.fence.violation(r.x_b);
  if (!why.empty()) {throw std::runtime_error(std::string(what) + ": " + why);}
  HoldTrajectory hold(v.params, r);
  checkPathClearsTable(v, hold, c, what, false);
}

// Finish a step: wrap the planner's own error with the step's name, then the
// fence and table checks of the whole path.
template<typename Fn>
std::unique_ptr<Trajectory> step(
  const VehicleModel & v, const PushPullConfig & c, const std::string & what, Fn plan,
  bool table = true)
{
  std::unique_ptr<Trajectory> traj;
  try {
    traj = plan();
  } catch (const std::exception & e) {
    throw std::runtime_error(what + ": " + e.what());
  }
  checkPathInFence(v, *traj, c.fence, what);
  if (table) {checkPathClearsTable(v, *traj, c, what);}
  return traj;
}

}  // namespace

const char * pushPullStepName(int step)
{
  static const char * const names[kNumPushPullSteps] = {
    "go_to_start", "ready", "push", "exit", "go_to_land_start", "land"};
  return step >= 0 && step < kNumPushPullSteps ? names[step] : "?";
}

double TableKeepOut::distanceXY(const Vec3 & p) const
{
  const double dx = std::max({x_min - p(0), 0.0, p(0) - x_max});
  const double dy = std::max({y_min - p(1), 0.0, p(1) - y_max});
  return std::hypot(dx, dy);
}

namespace
{
// One keep-out's check (the table, or the obstacle): see checkPathClearsTable.
void checkPathClearsKeepOut(
  const VehicleModel & v, const Trajectory & traj, const PushPullConfig & c,
  const TableKeepOut & ko, const char * ko_name, const char * ko_param,
  const std::string & what, bool allow_low_start)
{
  if (!ko.valid()) {return;}
  // the parts checked: the landing gear (lowest point of the airframe) and the
  // arm chain's joints 2..4 and grasp point, level-attitude approximation
  // (the planned tilts are a degree or two)
  static const char * const kPart[5] = {"the landing gear", "arm joint 2", "arm joint 3",
    "arm joint 4", "the claw"};
  struct Sample {std::array<double, 5> low; std::array<double, 5> h;};
  const auto sample = [&](double t) {
      const WbReference ref = traj.eval(t);
      const double psi = std::atan2(ref.b1_d(1), ref.b1_d(0));
      Vec3 r0c;
      armKinematics(ref.q_d, v.params, &r0c, nullptr, nullptr);
      const Vec3 body = ref.x_cd - Rz(psi) * r0c;
      std::array<Vec3, 5> pts;
      armChainPoints(ref.q_d, v.params, &pts);
      Sample s;
      // gear: over the table when its footprint reaches the top's footprint
      const bool gear_over = ko.distanceXY(body) < c.gear_radius + c.table_margin_xy;
      s.h[0] = body(2) - c.gear_depth - ko.z_top;
      s.low[0] = gear_over ? std::max(0.0, c.table_clearance - s.h[0]) : 0.0;
      for (int i = 1; i < 5; ++i) {
        const Vec3 w = body + Rz(psi) * pts[i];
        s.h[i] = w(2) - ko.z_top;
        const bool over = ko.distanceXY(w) < c.table_margin_xy;
        s.low[i] = over ? std::max(0.0, c.table_clearance - s.h[i]) : 0.0;
      }
      return s;
    };
  // a step may START low (a re-fly from where an abort left it): then no
  // sample may be lower than the start (1 mm slack); clear, it must stay clear
  Sample s0 = sample(0.0);
  if (!allow_low_start) {s0.low.fill(0.0);}
  const int n = traj.duration() > 0.0 ? 200 : 0;
  for (int k = 0; k <= n; ++k) {
    const double t = n > 0 ? traj.duration() * k / n : 0.0;
    const Sample s = sample(t);
    for (int i = 0; i < 5; ++i) {
      if (s.low[i] <= (s0.low[i] > 0.0 ? s0.low[i] + 1e-3 : 0.0)) {continue;}
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(3);
      m << what << ": " << kPart[i] << " would be " << s.h[i] << " m above the " << ko_name << " top ("
        << ko.z_top << " m) at t = " << std::setprecision(1) << t << " s, over the " << ko_name
        << " -- it must keep " << std::setprecision(3) << c.table_clearance << " m"
        << (s0.low[i] > 0.0 ? " (and never go lower than where the step starts)" : "")
        << " -- check the " << ko_name << "'s pose (" << ko_param << ") and the box capture / EE offset";
      throw std::runtime_error(m.str());
    }
  }
}
}  // namespace

void checkPathClearsTable(
  const VehicleModel & v, const Trajectory & traj, const PushPullConfig & c,
  const std::string & what, bool allow_low_start)
{
  checkPathClearsKeepOut(v, traj, c, c.table, "table", "push_pull_table", what, allow_low_start);
  checkPathClearsKeepOut(
    v, traj, c, c.obstacle, "obstacle", "push_pull_obstacle", what, allow_low_start);
}

PushPullWaypoints pushPullWaypoints(
  const VehicleModel & v, const PushPullConfig & c, const PushPullTargets & t)
{
  checkPose(v, c.home, "home pose", kPlExit);
  checkPose(v, c.push_pose, "push pose", kPlReady);
  if (!(c.approach_dz > 0.0) && !(c.approach_back > 0.0)) {
    throw std::runtime_error(
      "ready: the safety margin must be > 0 (approach_dz above or approach_back behind the grasp)");
  }
  const double exit_dz = c.exit_dz > 0.0 ? c.exit_dz : c.approach_dz;
  if (!(exit_dz > 0.0)) {
    throw std::runtime_error("exit: the climb must be > 0 (exit_dz, or approach_dz when exit_dz is 0)");
  }
  PushPullWaypoints wp;
  // wrapped: Go To Land's "turn as typed" reads the typed land yaw against
  // this (2026-10-15: a beam at 180 deg + 90 gave 270 and a 630 deg turn)
  wp.drone_yaw = wrapAngle(t.box_yaw + c.yaw_from_box);
  const double phi = wp.drone_yaw - 0.5 * M_PI;   // actual yaw -> model heading
  wp.push_dir = noseOf(phi);
  Vec3 r0e;
  armKinematics(c.push_pose, v.params, nullptr, &r0e, nullptr);
  // the grasp point: the box CG + the ONE offset, turned with the box
  wp.grasp_ee = t.box + Rz(t.box_yaw) * c.ee_offset;
  wp.grasp.phi = phi;
  wp.grasp.q = c.push_pose;
  wp.grasp.x_b = wp.grasp_ee - Rz(phi) * r0e;
  wp.approach = wp.grasp;
  wp.approach.x_b(2) += c.approach_dz;
  wp.approach.x_b -= c.approach_back * wp.push_dir;
  wp.approach_ee = clawOf(v, wp.approach);
  wp.axis = t.box;
  wp.pushed = wp.grasp;
  if (c.turn_angle != 0.0) {
    // THE TURN: the grasp rest rotated about the captured body's vertical
    // axis; the nose must be along the turn's tangent at the handle, or the
    // jaws would drag the handle sideways instead of pushing it round
    const Vec3 radial = wp.grasp_ee - t.box;
    const double r = std::hypot(radial(0), radial(1));
    if (r < 0.02) {
      throw std::runtime_error(
        "push: the grasp point is on the turn's axis (push_pull_ee_offset must put it at the "
        "handle's radius from the captured body)");
    }
    const Vec3 tangent = (c.turn_angle > 0.0 ? 1.0 : -1.0) * Vec3(-radial(1), radial(0), 0.0) / r;
    const double along = wp.push_dir.dot(tangent);
    if (along < std::cos(15.0 / kDeg)) {
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(0);
      m << "push: the nose (push_pull_yaw_from_box_deg " << c.yaw_from_box * kDeg << ") is "
        << std::acos(std::clamp(along, -1.0, 1.0)) * kDeg << " deg off the turn's tangent at the "
        << "handle -- for a " << (c.turn_angle > 0.0 ? "counter-clockwise (+)" : "clockwise (-)")
        << " turn it must be " << (c.turn_angle > 0.0 ? "+90" : "-90")
        << " deg from the handle's bearing from the axis";
      throw std::runtime_error(m.str());
    }
    wp.pushed.phi = wp.grasp.phi + c.turn_angle;
    wp.pushed.x_b = t.box + Rz(c.turn_angle) * (wp.grasp.x_b - t.box);
    wp.pushed.x_b(2) = wp.grasp.x_b(2);
  } else {
    wp.pushed.x_b += c.push_distance * wp.push_dir;
    if (c.box_half_length > 0.0 && c.table.valid()) {
      // the pushed box's two ends along the push, inside the table less the margin
      const Vec3 end = t.box + c.push_distance * wp.push_dir;
      const double m = c.box_edge_margin;
      for (const double s : {-1.0, 1.0}) {
        const Vec3 p = end + s * c.box_half_length * wp.push_dir;
        if (p(0) < c.table.x_min + m || p(0) > c.table.x_max - m ||
          p(1) < c.table.y_min + m || p(1) > c.table.y_max - m)
        {
          std::ostringstream e;
          e.setf(std::ios::fixed);
          e.precision(3);
          e << "push: the box would end at (" << end(0) << ", " << end(1) << ") -- its "
            << (s > 0.0 ? "leading" : "trailing") << " end at (" << p(0) << ", " << p(1)
            << ") is off the table or within " << m << " m of its edge (push_pull_table x "
            << c.table.x_min << ".." << c.table.x_max << ", y " << c.table.y_min << ".."
            << c.table.y_max << "). Is the box's yaw right (the push follows the nose, box yaw "
            << std::setprecision(0) << t.box_yaw * kDeg << " deg + push_pull_yaw_from_box_deg "
            << c.yaw_from_box * kDeg << " deg)? Or shorten push_pull_push_distance.";
          throw std::runtime_error(e.str());
        }
      }
    }
  }
  wp.pushed_ee = clawOf(v, wp.pushed);
  wp.lifted = wp.pushed;
  wp.lifted.x_b(2) += exit_dz;
  wp.exited = wp.lifted;
  wp.exited.q = c.home;
  wp.land.x_b = t.land.p;
  wp.land.phi = t.land.yaw - 0.5 * M_PI;
  wp.land.q = c.home;
  wp.land_start = wp.land;
  if (t.land_start_set) {
    wp.land_start.x_b = t.land_start.p;
    wp.land_start.phi = t.land_start.yaw - 0.5 * M_PI;
  }
  checkRest(v, c, wp.approach, "go_to_start goal");
  checkRest(v, c, wp.grasp, "ready goal");
  checkRest(v, c, wp.pushed, "push goal");
  checkRest(v, c, wp.lifted, "exit climb");
  checkRest(v, c, wp.exited, "exit goal");
  checkRest(v, c, wp.land_start, "go_to_land_start goal");
  checkRest(v, c, wp.land, "land goal");
  return wp;
}

std::unique_ptr<Trajectory> planPushPullGoToStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  // the short way round to the push heading
  return step(v, c, "go_to_start", [&]() {
      return transitionLeg(v, transition, opts, from, wp.approach);
    });
}

std::unique_ptr<Trajectory> planPushPullReady(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from,
  const Vec3 & trim)
{
  RestSpec goal = wp.grasp;
  goal.x_b += trim;
  return step(v, c, "ready", [&]() {
      return transitionLeg(v, transition, opts, from, goal);
    });
}

std::unique_ptr<Trajectory> planPushPullPush(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, const Vec3 & axis)
{
  RestSpec goal = from;
  goal.x_b += c.push_distance * noseOf(from.phi);
  const bool turn = c.turn_angle != 0.0;
  if (turn && !(c.push_time > 0.0)) {
    throw std::runtime_error("push: a turn needs push_pull_push_time_s > 0 (its duration)");
  }
  const bool cruise = !turn && c.push_ramp > 0.0;
  if (cruise && !(c.push_time > 2.0 * c.push_ramp)) {
    std::ostringstream m;
    m << "push: push_pull_push_ramp_s (" << c.push_ramp
      << " s) must be less than half push_pull_push_time_s (" << c.push_time << " s)";
    throw std::runtime_error(m.str());
  }
  return step(v, c, "push", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (c.push_settle > 0.0) {
        segs.push_back(std::make_shared<PauseTrajectory>(v.params, from, c.push_settle));
      }
      if (turn) {
        segs.push_back(std::make_shared<TurnTrajectory>(v.params, from, axis, c.turn_angle, c.push_time));
        goal = segs.back()->goalRest();
      } else if (cruise) {
        segs.push_back(std::make_shared<CruiseTrajectory>(
          v.params, from, noseOf(from.phi), c.push_distance, c.push_time, c.push_ramp));
      } else {
        segs.push_back(transitionLeg(
          v, transition, timedOptions(opts, c.push_distance, c.push_time), from, goal));
      }
      // the unload measures the claw's task error over the end of this hold
      if (c.unload_time > 0.0 && c.unload_wait > 0.0) {
        segs.push_back(std::make_shared<PauseTrajectory>(v.params, goal, c.unload_wait));
      }
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    });
}

std::unique_ptr<Trajectory> planPushPullUnload(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, const Vec3 & shift)
{
  RestSpec goal = from;
  const bool move = shift.norm() >= 5e-4 && c.unload_time > 0.0;
  if (move) {goal.x_b += shift;}
  return step(v, c, "unload", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (move) {
        segs.push_back(transitionLeg(
          v, transition, timedOptions(opts, shift.norm(), c.unload_time), from, goal));
      }
      segs.push_back(std::make_shared<PauseTrajectory>(
        v.params, goal, std::max(c.unload_wait, 1e-3)));
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    });
}

std::unique_ptr<Trajectory> planPushPullExit(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from)
{
  const double exit_dz = c.exit_dz > 0.0 ? c.exit_dz : c.approach_dz;
  RestSpec up = from;
  up.x_b(2) += exit_dz;
  RestSpec home = up;
  home.q = c.home;
  return step(v, c, "exit", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs{
        transitionLeg(v, transition, timedOptions(opts, exit_dz, c.exit_time), from, up)};
      if ((up.q - c.home).cwiseAbs().maxCoeff() > 1e-6) {
        segs.push_back(transitionLeg(v, transition, opts, up, home));
      }
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    });
}

std::unique_ptr<Trajectory> planPushPullGoToLandStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  // THE TURN AS TYPED (user, 2026-10-03: -90 -> -360 deg is 270 deg
  // CLOCKWISE, not the 90 deg the short way would take): from the push
  // heading the turn is the typed difference Land Start yaw - drone yaw,
  // unwrapped; from any other heading (a re-fly, a step after Abort elsewhere)
  // the short way.
  RestSpec goal = wp.land_start;
  const double from_yaw = from.phi + 0.5 * M_PI;
  const bool at_push_heading = std::abs(wrapAngle(from_yaw - wp.drone_yaw)) < 30.0 / kDeg;
  const double turn = at_push_heading ? goal.phi + 0.5 * M_PI - wp.drone_yaw
                                      : wrapAngle(goal.phi - from.phi);
  goal.phi = from.phi + turn;
  goal.q = from.q;
  PlanOptions o = opts;
  o.heading_as_given = true;
  // ONE transition (2026-10-16, user: "simple but a safe transition", like
  // Go To Start): from the exit hold straight to Land Start, the arm held
  // home; step() checks the whole path against the table (gear + arm points),
  // the same guarantee every other leg has
  return step(v, c, "go_to_land_start", [&]() -> std::unique_ptr<Trajectory> {
      if ((goal.x_b - from.x_b).norm() < 1e-3 && std::abs(goal.phi - from.phi) < 1e-3) {
        return std::make_unique<PauseTrajectory>(v.params, from, 0.1);
      }
      return transitionLeg(v, transition, o, from, goal);
    });
}

std::unique_ptr<Trajectory> planPushPullLand(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  // straight to the Land hover, the heading and the arm kept (Land Start's
  // yaw was flown by go_to_land_start); a moment's pause when already there
  RestSpec goal = wp.land;
  goal.phi = from.phi;
  goal.q = from.q;
  return step(v, c, "land", [&]() -> std::unique_ptr<Trajectory> {
      if ((goal.x_b - from.x_b).norm() < 1e-3) {
        return std::make_unique<PauseTrajectory>(v.params, from, 0.1);
      }
      return transitionLeg(v, transition, opts, from, goal);
    });
}

std::unique_ptr<Trajectory> planPushPullAbort(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, double climb, double release_wait,
  const Vec3 & retreat)
{
  RestSpec back = from;
  back.x_b += Vec3(retreat(0), retreat(1), 0.0);
  RestSpec up = back;
  up.x_b(2) += std::max(0.0, climb);
  RestSpec home = up;
  home.q = c.home;
  return step(v, c, "abort", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (release_wait > 0.0) {
        segs.push_back(std::make_shared<PauseTrajectory>(v.params, from, release_wait));
      }
      const bool retreating = (back.x_b - from.x_b).norm() > 1e-3;
      if (retreating) {segs.push_back(transitionLeg(v, transition, opts, from, back));}
      if (climb > 1e-3) {segs.push_back(transitionLeg(v, transition, opts, back, up));}
      if ((up.q - c.home).cwiseAbs().maxCoeff() > 1e-6 ||
        (segs.size() == (release_wait > 0.0 ? 1u : 0u) + (retreating ? 1u : 0u)))
      {
        segs.push_back(transitionLeg(v, transition, opts, up, home));
      }
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    }, false);
}

std::vector<std::shared_ptr<const Trajectory>> planPushPullMission(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  std::vector<std::shared_ptr<const Trajectory>> steps;
  steps.push_back(planPushPullGoToStart(v, transition, opts, c, wp, from));
  steps.push_back(planPushPullReady(v, transition, opts, c, wp, wp.approach));
  steps.push_back(planPushPullPush(v, transition, opts, c, wp.grasp, wp.axis));
  steps.push_back(planPushPullExit(v, transition, opts, c, steps.back()->goalRest()));
  steps.push_back(planPushPullGoToLandStart(v, transition, opts, c, wp, steps.back()->goalRest()));
  steps.push_back(planPushPullLand(v, transition, opts, c, wp, steps.back()->goalRest()));
  return steps;
}

void ProgressGovernor::setup(double bandwidth, double recover_s)
{
  f_.setup(4, bandwidth, 1);
  recover_ = std::max(1e-3, recover_s);
  reset();
}

void ProgressGovernor::reset()
{
  f_.reset(Eigen::VectorXd::Constant(1, 1.0));
  tau_ = 0.0;
  gate_ = 1.0;
  rates_ = {{1.0, 0.0, 0.0, 0.0}};
}

void ProgressGovernor::step(double target, double dt)
{
  if (!(dt > 0.0)) {return;}
  target = std::isfinite(target) ? std::min(std::max(target, 0.0), 1.0) : 0.0;
  // falls at once, rises at most dt / recover per tick
  gate_ = std::min(target, gate_ + dt / recover_);
  const double r0 = rates_[0];
  f_.step(Eigen::VectorXd::Constant(1, gate_), dt);
  for (int k = 0; k < 4; ++k) {rates_[static_cast<size_t>(k)] = f_.deriv(k)(0);}
  rates_[0] = std::min(std::max(rates_[0], 0.0), 1.0);   // round-off only
  tau_ += 0.5 * (r0 + rates_[0]) * dt;
}

double followFactor(double e, double soft, double hard)
{
  if (!(hard > soft) || !(hard > 0.0)) {return 1.0;}
  if (!std::isfinite(e)) {return 0.0;}
  if (e <= soft) {return 1.0;}
  if (e >= hard) {return 0.0;}
  return (hard - e) / (hard - soft);
}

}  // namespace fsc_trajectory_planner
