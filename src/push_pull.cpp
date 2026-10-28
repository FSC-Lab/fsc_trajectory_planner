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
#include <limits>
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

// THE WAY ROUND THE TABLE (2026-10-22). A Bezier curve of any degree with its
// first four derivatives in its parameter (hodographs, de Casteljau).
class Bezier
{
public:
  explicit Bezier(std::vector<Vec3> ctrl)
  {
    d_[0] = std::move(ctrl);
    for (int k = 1; k < 5; ++k) {
      const std::vector<Vec3> & p = d_[static_cast<size_t>(k - 1)];
      std::vector<Vec3> q;
      const double n = static_cast<double>(p.size()) - 1.0;
      for (size_t i = 0; i + 1 < p.size(); ++i) {q.push_back(n * (p[i + 1] - p[i]));}
      d_[static_cast<size_t>(k)] = q;
    }
  }
  // the k-th derivative (k = 0..4) at s in [0, 1]
  Vec3 eval(int k, double s) const
  {
    std::vector<Vec3> b = d_[static_cast<size_t>(k)];
    if (b.empty()) {return Vec3::Zero();}
    for (size_t r = b.size() - 1; r > 0; --r) {
      for (size_t i = 0; i < r; ++i) {b[i] = (1.0 - s) * b[i] + s * b[i + 1];}
    }
    return b[0];
  }

private:
  std::array<std::vector<Vec3>, 5> d_;
};

// The smooth trapezoid phase s(t) from 0 to 1 over T: its rate rises on a septic
// smoothstep over Tr, holds 1 / (T - Tr) and falls on the mirror image -- the
// CruiseTrajectory's profile with d = 1, so s' .. s'''' are continuous and zero
// at both ends. s[0..4] = s and its first four time derivatives.
void trapezoidPhase(double t, double T, double Tr, double s[5])
{
  const double V = 1.0 / (T - Tr);
  const auto up = [&](double tau, double o[5]) {
      const double u = std::min(std::max(tau / Tr, 0.0), 1.0);
      const double u2 = u * u, u3 = u2 * u, w = 1.0 - u;
      o[0] = V * Tr * u2 * u3 * (7.0 + u * (-14.0 + u * (10.0 - 2.5 * u)));
      o[1] = V * u2 * u2 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u)));
      o[2] = V / Tr * 140.0 * u3 * w * w * w;
      o[3] = V / (Tr * Tr) * 420.0 * u2 * w * w * (1.0 - 2.0 * u);
      o[4] = V / (Tr * Tr * Tr) * 840.0 * u * w * (1.0 + u * (-5.0 + 5.0 * u));
    };
  const double tt = std::min(std::max(t, 0.0), T);
  if (tt <= Tr) {
    up(tt, s);
  } else if (tt >= T - Tr) {
    up(T - tt, s);
    s[0] = 1.0 - s[0];
    s[2] = -s[2];
    s[4] = -s[4];
  } else {
    s[0] = V * (0.5 * Tr + (tt - Tr));
    s[1] = V;
    s[2] = s[3] = s[4] = 0.0;
  }
}

// The CoM along `path` (world) at the phase s(t), the heading turning by dphi on
// the same phase, the arm held at from.q: the flat outputs and the derivatives
// flatState() reads follow from the curve's by Faa di Bruno, so the reference is
// compatible through snap and at rest at both ends.
class BypassTrajectory : public Trajectory
{
public:
  BypassTrajectory(
    const WholeBodyParams & p, const RestSpec & from, const RestSpec & goal, Bezier path,
    double dphi, double T, double Tr)
  : p_(p), from_(from), goal_(goal), path_(std::move(path)), dphi_(dphi), T_(T), Tr_(Tr)
  {
    diag_.T = T;
    diag_.min_sigma_nd = sigmaNd(from.q, p);
    diag_.q_min_deg = from.q * kDeg;
    diag_.q_max_deg = diag_.q_min_deg;
  }
  double duration() const override {return T_;}
  RestSpec goalRest() const override {return goal_;}
  const PlanDiag & diag() const override {return diag_;}
  PlanDiag & mutableDiag() {return diag_;}

  WbReference eval(double t) const override
  {
    double s[5];
    trapezoidPhase(t, T_, Tr_, s);
    const Vec3 P1 = path_.eval(1, s[0]), P2 = path_.eval(2, s[0]), P3 = path_.eval(3, s[0]),
      P4 = path_.eval(4, s[0]);
    const double s1 = s[1], s2 = s[2], s3 = s[3], s4 = s[4];
    Eigen::Matrix<double, 5, 3> xc;
    xc.row(0) = path_.eval(0, s[0]).transpose();
    xc.row(1) = (P1 * s1).transpose();
    xc.row(2) = (P2 * s1 * s1 + P1 * s2).transpose();
    xc.row(3) = (P3 * s1 * s1 * s1 + 3.0 * P2 * s1 * s2 + P1 * s3).transpose();
    xc.row(4) = (P4 * s1 * s1 * s1 * s1 + 6.0 * P3 * s1 * s1 * s2 +
      P2 * (3.0 * s2 * s2 + 4.0 * s1 * s3) + P1 * s4).transpose();
    const Vec3 psi(from_.phi + dphi_ * s[0], dphi_ * s1, dphi_ * s2);
    Eigen::Matrix<double, 3, kNumJoints> q = Eigen::Matrix<double, 3, kNumJoints>::Zero();
    q.row(0) = from_.q.transpose();
    WbReference ref;
    flatState(p_, xc, psi, q, &ref, nullptr);
    return ref;
  }

private:
  WholeBodyParams p_;
  RestSpec from_, goal_;
  Bezier path_;
  double dphi_, T_, Tr_;
  PlanDiag diag_;
};

// A plan with a note in front of its summary (the go_to_start that could not go
// round the table says why).
class NotedTrajectory : public Trajectory
{
public:
  NotedTrajectory(std::unique_ptr<Trajectory> t, const std::string & note)
  : t_(std::move(t)), diag_(t_->diag())
  {
    diag_.summary = note + diag_.summary;
  }
  double duration() const override {return t_->duration();}
  WbReference eval(double t) const override {return t_->eval(t);}
  RestSpec goalRest() const override {return t_->goalRest();}
  const PlanDiag & diag() const override {return diag_;}

private:
  std::unique_ptr<Trajectory> t_;
  PlanDiag diag_;
};

// THE WAY ROUND THE TABLE (2026-10-22): the push frame -- origin the box, x
// along the push, y to its left -- and the table's extent in it (exact for a
// table placed from the box): its box end at xe, its far end at xf, its sides
// at ylo / yhi.
struct RoundFrame
{
  Vec3 o, ux, uy;
  double xe{1e9}, xf{-1e9}, ylo{1e9}, yhi{-1e9};
  RoundFrame(const PushPullWaypoints & wp, const TableKeepOut & tb)
  : o(wp.axis), ux(wp.push_dir), uy(-wp.push_dir(1), wp.push_dir(0), 0.0)
  {
    for (const double cx : {tb.x_min, tb.x_max}) {
      for (const double cy : {tb.y_min, tb.y_max}) {
        const Vec3 p = toP(tb.toWorld(Vec3(cx, cy, 0.0)));
        xe = std::min(xe, p(0));
        xf = std::max(xf, p(0));
        ylo = std::min(ylo, p(1));
        yhi = std::max(yhi, p(1));
      }
    }
  }
  Vec3 toP(const Vec3 & w) const
  {
    const Vec3 d = w - o;
    return Vec3(d.dot(ux), d.dot(uy), w(2));
  }
  Vec3 fromP(const Vec3 & p) const
  {
    return Vec3(o(0) + p(0) * ux(0) + p(1) * uy(0), o(1) + p(0) * ux(1) + p(1) * uy(1), p(2));
  }
};

// The curve from rest A to rest B (the same arm pose; B's heading A's + dphi)
// round one END of the table, seen from above: the body bypass_clearance (R) or
// more from the table. Each end is either BESIDE the table (R or more off its
// side) or PAST one of its ends (R or more beyond it); between a beside end and a
// past end the curve hugs an L round that table end (nine control points, degree
// 8: 2026-10-22 prototype, <= 3 mm inside the L at the tightest start; a cubic
// cut it by 40-160 mm) whose corner sits at the past end's x and -- corner_on_side
// false -- the beside end's distance from the table (go_to_start and the way
// home: along the table's side, and across) or -- true -- R off the table's side
// (round the table's corner, then diagonally to the beside end); two ends on the
// same side, or past the same end, are joined straight. nullptr when A and B are
// the same rest. Throws a reason (no step prefix) otherwise.
std::shared_ptr<BypassTrajectory> roundTableCurve(
  const VehicleModel & v, const PlanOptions & opts, const PushPullConfig & c,
  const PushPullWaypoints & wp, const RestSpec & A, const RestSpec & B_in, double dphi,
  bool corner_on_side = false)
{
  const TableKeepOut & tb = c.table;
  const double R = c.bypass_clearance;
  const RoundFrame f(wp, tb);
  RestSpec B = B_in;
  B.q = A.q;
  B.phi = A.phi + dphi;
  Vec3 r0c;
  armKinematics(A.q, v.params, &r0c, nullptr, nullptr);
  const Vec3 p0 = f.toP(A.x_b + Rz(A.phi) * r0c);
  const Vec3 p1 = f.toP(B.x_b + Rz(B.phi) * r0c);
  if ((p1 - p0).norm() < 1e-3 && std::abs(dphi) < 1e-3) {return nullptr;}
  const Vec3 sa = f.toP(A.x_b), sb = f.toP(B.x_b);
  // past the box end (-1), past the far end (+1), neither (0); beside (+-1) or not
  const auto past = [&](const Vec3 & s) {
      return s(0) <= f.xe - R + 1e-6 ? -1 : s(0) >= f.xf + R - 1e-6 ? 1 : 0;
    };
  const auto side = [&](const Vec3 & s) {
      return s(1) >= f.yhi + R - 1e-6 ? 1 : s(1) <= f.ylo - R + 1e-6 ? -1 : 0;
    };
  constexpr int kN = 8;
  std::vector<Vec3> ctrl;
  if ((past(sa) != 0 && past(sa) == past(sb)) || (side(sa) != 0 && side(sa) == side(sb))) {
    for (int i = 0; i <= kN; ++i) {ctrl.push_back(p0 + (p1 - p0) * i / kN);}
  } else if ((side(sa) != 0 && past(sb) != 0) || (past(sa) != 0 && side(sb) != 0)) {
    // the corner: at the past end's x; beside at the beside end's y, or R off the side
    const Vec3 & pp = past(sb) != 0 ? p1 : p0;     // the past end
    const Vec3 & pb = past(sb) != 0 ? p0 : p1;     // the beside end
    const int sd = past(sb) != 0 ? side(sa) : side(sb);
    const double y = corner_on_side ? (sd > 0 ? f.yhi + R : f.ylo - R) : pb(1);
    const Vec3 corner(pp(0), y, 0.0);
    for (int j = 0; j < kN / 2; ++j) {ctrl.push_back(p0 + (corner - p0) * j / (kN / 2));}
    for (int j = 0; j <= kN / 2; ++j) {ctrl.push_back(corner + (p1 - corner) * j / (kN / 2));}
  } else {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    const Vec3 & bad = side(sa) == 0 && past(sa) == 0 ? sa : sb;
    m << "no way round the table from (" << (&bad == &sa ? A.x_b(0) : B.x_b(0)) << ", "
      << (&bad == &sa ? A.x_b(1) : B.x_b(1)) << "): it is neither " << R << " m beside the table "
      "nor past one of its ends (too close to it, or the two ends on opposite sides) -- start "
      "beside the table";
    throw std::runtime_error(m.str());
  }
  for (int i = 0; i <= kN; ++i) {
    ctrl[static_cast<size_t>(i)](2) = p0(2) + (p1(2) - p0(2)) * i / kN;   // the height, linear
    ctrl[static_cast<size_t>(i)] = f.fromP(ctrl[static_cast<size_t>(i)]);
  }
  Bezier path(ctrl);
  // THE TIMING: for a ramp fraction rho every peak scales exactly with T (the
  // speeds as 1 / T, the accelerations as 1 / T^2), so T is the largest the
  // bounds ask at T = 1; the fraction that needs the shortest T is taken.
  Vec3 r0e;
  armKinematics(A.q, v.params, nullptr, &r0e, nullptr);
  const double arm = (r0e - r0c).norm();   // the claw's lever about the CoM
  double T = std::numeric_limits<double>::infinity(), Tr = 0.0, v_claw = 0.0;
  for (double rho = 0.10; rho < 0.475; rho += 0.04) {
    double vp = 0.0, ap = 0.0, wp1 = 0.0, dwp = 0.0;
    for (int k = 0; k <= 400; ++k) {
      double ph[5];
      trapezoidPhase(k / 400.0, 1.0, rho, ph);
      const Vec3 d1 = path.eval(1, ph[0]), d2 = path.eval(2, ph[0]);
      const double w = std::abs(dphi) * ph[1];
      vp = std::max(vp, d1.norm() * ph[1] + w * arm);   // the claw, the faster
      ap = std::max(ap, (d2 * ph[1] * ph[1] + d1 * ph[2]).norm());
      wp1 = std::max(wp1, w);
      dwp = std::max(dwp, std::abs(dphi) * std::abs(ph[2]));
    }
    const double t = 1.02 * std::max({opts.T_min, vp / opts.v_max, std::sqrt(ap / opts.a_max),
          wp1 / opts.w_max, std::sqrt(dwp / opts.dw_max)});
    if (t < T) {
      T = t;
      Tr = rho * t;
      v_claw = vp / t;
    }
  }
  if (T > opts.T_max) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << "the way round the table takes " << T << " s, longer than the planner's t_max ("
      << opts.T_max << " s)";
    throw std::runtime_error(m.str());
  }
  auto curve = std::make_shared<BypassTrajectory>(v.params, A, B, path, dphi, T, Tr);
  // THE CLEARANCE, checked on the flown reference: the body origin's distance
  // from the table's footprint, seen from above
  double closest = std::min(tb.distanceXY(A.x_b), tb.distanceXY(B.x_b));
  double length = 0.0, v_com = 0.0;
  Vec3 prev = A.x_b;
  Vec3 prev_c = A.x_b + Rz(A.phi) * r0c;
  for (int k = 0; k <= 300; ++k) {
    const WbReference ref = curve->eval(T * k / 300.0);
    const double psi = std::atan2(ref.b1_d(1), ref.b1_d(0));
    const Vec3 body = ref.x_cd - Rz(psi) * r0c;
    closest = std::min(closest, tb.distanceXY(body));
    length += (body - prev).norm();
    if (k > 0) {v_com = std::max(v_com, (ref.x_cd - prev_c).norm() / (T / 300.0));}
    prev = body;
    prev_c = ref.x_cd;
  }
  if (closest < R - 5e-3) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "the way round the table would pass " << closest << " m from it (it keeps " << R
      << " m, push_pull_bypass_clearance) -- the vehicle starts " << tb.distanceXY(A.x_b)
      << " m and ends " << tb.distanceXY(B.x_b) << " m from the table: keep the start at least "
      << R + kBypassStartMargin << " m from its side";
    throw std::runtime_error(m.str());
  }
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m << "round the table: " << std::setprecision(2) << length << " m in "
    << std::setprecision(1) << T << " s, " << std::setprecision(2) << closest
    << " m from the table at the closest (keeps " << R << "), heading "
    << std::showpos << std::setprecision(0) << dphi * kDeg << std::noshowpos << " deg";
  curve->mutableDiag().summary = m.str();
  curve->mutableDiag().peak_com_speed = v_com;
  curve->mutableDiag().peak_ee_speed = v_claw;
  return curve;
}

// go_to_start round the table's BOX end (PushPullConfig::bypass): the curve to
// the entry -- at rest on the push's line R past the table's end, the push
// heading flown, the arm still as it was --, then straight in along the push.
// nullptr when `from` is already within bypass_clearance of the table (the
// caller flies the one transition).
std::unique_ptr<Trajectory> bypassGoToStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  const double R = c.bypass_clearance;
  if (c.table.distanceXY(from.x_b) < R) {return nullptr;}
  const RoundFrame f(wp, c.table);
  const RestSpec & G = wp.approach;
  const Vec3 g = f.toP(G.x_b);
  const double dphi = wrapAngle(G.phi - from.phi);
  RestSpec E = G;
  E.q = from.q;
  E.phi = from.phi + dphi;
  E.x_b = f.fromP(Vec3(std::min(g(0), f.xe - R), g(1), G.x_b(2)));
  std::vector<std::shared_ptr<const Trajectory>> segs;
  if (auto curve = roundTableCurve(v, opts, c, wp, from, E, dphi)) {segs.push_back(curve);}
  if ((E.x_b - G.x_b).norm() > 1e-4 || (E.q - G.q).cwiseAbs().maxCoeff() > 1e-6) {
    segs.push_back(transitionLeg(v, transition, opts, E, G));
  }
  if (segs.empty()) {return nullptr;}
  return std::make_unique<SequenceTrajectory>(std::move(segs));
}

// One keep-out's check (the table, the obstacle, the pushed box): defined below.
void checkPathClearsKeepOut(
  const VehicleModel & v, const Trajectory & traj, const PushPullConfig & c,
  const TableKeepOut & ko, const char * ko_name, const char * ko_param,
  const std::string & what, bool allow_low_start);

// go_to_land_start round the table's FAR end (2026-10-22, user: "go round the
// right side of the table, then land"): from a hold near the table (the exit, an
// abort's climb) straight on along the push's line, heading and arm kept, to R
// past the table's far end -- over the pushed box, checked against it (its sweep
// along the push, up to the handle's top) --, then the curve round the table's
// far end and back to Land Start, the mirror image of go_to_start's (across at
// R past the end, then along the table's side at the start's distance), the
// heading turning by `dphi` (as typed) on the way. From a hold already clear of
// the table only the curve.
std::unique_ptr<Trajectory> bypassGoToLandStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from,
  const RestSpec & goal, double dphi)
{
  const double R = c.bypass_clearance;
  const RoundFrame f(wp, c.table);
  std::vector<std::shared_ptr<const Trajectory>> segs;
  RestSpec out = from;
  if (c.table.distanceXY(from.x_b) < R) {
    const Vec3 s = f.toP(from.x_b);
    out.x_b = f.fromP(Vec3(std::max(s(0), f.xf + R), s(1), s(2)));
    if ((out.x_b - from.x_b).norm() > 1e-4) {
      auto on = transitionLeg(v, transition, opts, from, out);
      // the leg passes over the box wherever the push left it: the box swept along
      // the push (+ its half length), across the table's depth + 1 cm, up to the
      // handle's top -- the gear and the arm keep table_clearance above it
      if (c.box_half_length > 0.0) {
        TableKeepOut box;
        box.ox = wp.axis(0);
        box.oy = wp.axis(1);
        box.yaw = std::atan2(wp.push_dir(1), wp.push_dir(0));
        box.x_min = std::min(0.0, c.push_distance) - c.box_half_length;
        box.x_max = std::max(0.0, c.push_distance) + c.box_half_length;
        box.y_min = -0.5 * c.table_depth - 0.01;
        box.y_max = 0.5 * c.table_depth + 0.01;
        box.z_top = wp.grasp_ee(2) + c.handle_above_grasp;
        // strict (the start too): the exit climbs the gear ~0.11 m over the fin's top
        checkPathClearsKeepOut(v, *on, c, box, "pushed box", "push_pull_handle_top_above_grasp",
          "go_to_land_start", false);
      }
      segs.push_back(std::move(on));
    }
  }
  // (user, 2026-10-22: "symmetric to the entry part") the corner at the start's
  // distance from the table, as go_to_start's: across, then back along the table
  if (auto curve = roundTableCurve(v, opts, c, wp, out, goal, dphi, false)) {segs.push_back(curve);}
  if (segs.empty()) {return std::make_unique<PauseTrajectory>(v.params, from, 0.1);}
  return std::make_unique<SequenceTrajectory>(std::move(segs));
}

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

Vec3 TableKeepOut::toLocal(const Vec3 & p) const
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  const double dx = p(0) - ox, dy = p(1) - oy;
  return Vec3(c * dx + s * dy, -s * dx + c * dy, p(2));
}

Vec3 TableKeepOut::toWorld(const Vec3 & l) const
{
  const double c = std::cos(yaw), s = std::sin(yaw);
  return Vec3(ox + c * l(0) - s * l(1), oy + s * l(0) + c * l(1), l(2));
}

double TableKeepOut::distanceXY(const Vec3 & p) const
{
  const Vec3 l = toLocal(p);
  const double dx = std::max({x_min - l(0), 0.0, l(0) - x_max});
  const double dy = std::max({y_min - l(1), 0.0, l(1) - y_max});
  return std::hypot(dx, dy);
}

TableKeepOut tableFromBox(const PushPullConfig & c, const Vec3 & box, double box_yaw)
{
  // the box end of the table: the box's back face (box_half_length behind its
  // centre, against the push) plus the gap
  const double back = c.box_half_length + c.table_box_gap;
  if (!(c.box_half_length > 0.0) || !(c.table_box_gap >= 0.0) || !(c.table_depth > 0.0) ||
    !(c.table_length > 2.0 * c.box_half_length + c.table_box_gap))
  {
    std::ostringstream m;
    m << "the table placed from the box needs push_pull_box_half_length > 0, "
      "push_pull_table_box_gap >= 0 and push_pull_table_size [length, depth] with the length "
      "longer than the box plus the gap (have half length " << c.box_half_length << ", gap "
      << c.table_box_gap << ", size [" << c.table_length << ", " << c.table_depth << "])";
    throw std::runtime_error(m.str());
  }
  TableKeepOut t;
  t.ox = box(0);
  t.oy = box(1);
  t.yaw = wrapAngle(box_yaw + c.yaw_from_box);   // the push's direction (the nose at the box)
  t.x_min = -back;
  t.x_max = c.table_length - back;
  t.y_min = -0.5 * c.table_depth;
  t.y_max = 0.5 * c.table_depth;
  t.z_top = box(2) - c.box_half_height;
  return t;
}

TableKeepOut tableFromValve(const PushPullConfig & c, const Vec3 & axis, const Vec3 & start)
{
  if (!(c.table_length > 0.0) || !(c.table_depth > 0.0)) {
    std::ostringstream m;
    m << "the table placed from the valve needs push_pull_table_size [length, depth] > 0 (have ["
      << c.table_length << ", " << c.table_depth << "])";
    throw std::runtime_error(m.str());
  }
  TableKeepOut t;
  t.ox = axis(0);
  t.oy = axis(1);
  // x = the table's DEPTH, along the line from the start to the axis
  t.yaw = std::atan2(axis(1) - start(1), axis(0) - start(0));
  t.x_min = -0.5 * c.table_depth;
  t.x_max = 0.5 * c.table_depth;
  t.y_min = -0.5 * c.table_length;
  t.y_max = 0.5 * c.table_length;
  t.z_top = axis(2);
  return t;
}

TableKeepOut obstacleFromValve(const PushPullConfig & c, const Vec3 & axis)
{
  if (!(c.valve_reach > 0.0) || !(c.valve_top_above > 0.0)) {
    std::ostringstream m;
    m << "the valve's keep-out needs push_pull_valve_reach > 0 and push_pull_valve_top_above > 0 "
      "(have " << c.valve_reach << ", " << c.valve_top_above << ")";
    throw std::runtime_error(m.str());
  }
  TableKeepOut o;
  o.ox = axis(0);
  o.oy = axis(1);
  o.x_min = -c.valve_reach;
  o.x_max = c.valve_reach;
  o.y_min = -c.valve_reach;
  o.y_max = c.valve_reach;
  o.z_top = axis(2) + c.valve_top_above;
  return o;
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
  const VehicleModel & v, const PushPullConfig & c_in, const PushPullTargets & t)
{
  // THE TABLE PLACED FROM THE BOX (2026-10-22): every check below, and every
  // step after (wp.table), against the table the box says is there
  PushPullConfig c = c_in;
  if (c.table_from_box) {
    if (c.turn_angle != 0.0) {
      throw std::runtime_error(
        "push: push_pull_table_from_box is for a straight push (set push_pull_turn_deg 0, or "
        "type push_pull_table)");
    }
    c.table = tableFromBox(c, t.box, t.box_yaw);
  }
  // THE TURN'S AXIS (2026-10-28): obj_0 + Rz(yaw) * valve_axis_offset -- the
  // bearing axis on the plate-bottom plane when obj_0 is the markers' centroid
  // (Motive's default pivot); obj_0 itself with a zero offset (unchanged). A
  // straight push IGNORES the offset (it is the valve's: a planner switched
  // from the Valve Turning tab to the Box Pushing tab keeps it set).
  if (c.turn_angle != 0.0 && !c.valve_axis_offset.allFinite()) {
    throw std::runtime_error("push: push_pull_valve_axis_offset is not finite");
  }
  const Vec3 axis = c.turn_angle != 0.0 ? Vec3(t.box + Rz(t.box_yaw) * c.valve_axis_offset) : t.box;
  // THE KEEP-OUTS PLACED FROM THE VALVE (2026-10-27), and the layout they assume:
  // the vehicle starts in front of the valve, the beam (obj_0 +x) toward it
  if (c.keepouts_from_valve) {
    if (c.turn_angle == 0.0) {
      throw std::runtime_error(
        "push: push_pull_keepouts_from_valve is for a turn (set push_pull_turn_deg, or "
        "type push_pull_table / push_pull_obstacle)");
    }
    if (!t.start_set) {
      throw std::runtime_error("layout: the keep-outs placed from the valve need the start mark");
    }
    const Vec3 to_start(t.start.p(0) - axis(0), t.start.p(1) - axis(1), 0.0);
    const double d = to_start.norm();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    if (d < c.valve_start_min || d > c.valve_start_max) {
      m << "layout: the start (" << t.start.p(0) << ", " << t.start.p(1) << ") is " << d
        << " m from the valve's axis (" << axis(0) << ", " << axis(1) << ") -- start the vehicle "
        << c.valve_start_min << ".." << c.valve_start_max << " m in front of the valve "
        "(push_pull_valve_start_range) and press Adjust on the ground; or is obj_0 the valve "
        "(and push_pull_valve_axis_offset its pivot -> the axis)?";
      throw std::runtime_error(m.str());
    }
    const Vec3 beam(std::cos(t.box_yaw), std::sin(t.box_yaw), 0.0);
    const double off = std::acos(std::clamp(beam.dot(to_start) / d, -1.0, 1.0));
    if (c.valve_beam_tol > 0.0 && off > c.valve_beam_tol) {
      m << "layout: the beam (obj_0 +x, yaw " << std::setprecision(0) << t.box_yaw * kDeg
        << " deg) points " << off * kDeg << " deg away from the start (push_pull_valve_beam_tol_deg "
        << c.valve_beam_tol * kDeg << ") -- turn the beam so the fin faces the vehicle; "
        "or is obj_0's frame defined turned (its +x must run along the beam toward the fin)?";
      throw std::runtime_error(m.str());
    }
    c.table = tableFromValve(c, axis, t.start.p);
    c.obstacle = obstacleFromValve(c, axis);
  }
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
  // the grasp point: the box CG + the ONE offset, turned with the box -- in a
  // turn the axis point + the offset (2026-10-28: the same as obj_0 + offset
  // while the axis offset is zero)
  wp.grasp_ee = (c.turn_angle != 0.0 ? axis : t.box) + Rz(t.box_yaw) * c.ee_offset;
  wp.grasp.phi = phi;
  wp.grasp.q = c.push_pose;
  wp.grasp.x_b = wp.grasp_ee - Rz(phi) * r0e;
  wp.approach = wp.grasp;
  wp.approach.x_b(2) += c.approach_dz;
  wp.approach.x_b -= c.approach_back * wp.push_dir;
  wp.approach_ee = clawOf(v, wp.approach);
  wp.axis = axis;
  wp.pushed = wp.grasp;
  if (c.turn_angle != 0.0) {
    // THE TURN: the grasp rest rotated about the captured body's vertical
    // axis; the nose must be along the turn's tangent at the handle, or the
    // jaws would drag the handle sideways instead of pushing it round
    const Vec3 radial = wp.grasp_ee - axis;
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
    wp.pushed.x_b = axis + Rz(c.turn_angle) * (wp.grasp.x_b - axis);
    wp.pushed.x_b(2) = wp.grasp.x_b(2);
  } else {
    wp.pushed.x_b += c.push_distance * wp.push_dir;
    if (c.box_half_length > 0.0 && c.table.valid()) {
      // the pushed box's two ends along the push, inside the table less the margin
      // (in the table's own frame: a table placed from the box is turned with it)
      const Vec3 end = t.box + c.push_distance * wp.push_dir;
      const double m = c.box_edge_margin;
      for (const double s : {-1.0, 1.0}) {
        const Vec3 p = end + s * c.box_half_length * wp.push_dir;
        const Vec3 l = c.table.toLocal(p);
        if (l(0) < c.table.x_min + m || l(0) > c.table.x_max - m ||
          l(1) < c.table.y_min + m || l(1) > c.table.y_max - m)
        {
          std::ostringstream e;
          e.setf(std::ios::fixed);
          e.precision(3);
          e << "push: the box would end at (" << end(0) << ", " << end(1) << ") -- its "
            << (s > 0.0 ? "leading" : "trailing") << " end at (" << p(0) << ", " << p(1)
            << ") is off the table or within " << m << " m of its edge (";
          if (c.table_from_box) {
            e << "the table placed from the box: " << c.table.x_max - c.table.x_min
              << " m long, the box's back face " << c.table_box_gap << " m in from its end, so "
              << "at most " << c.table.x_max - m - c.box_half_length << " m of push";
          } else {
            e << "push_pull_table x " << c.table.x_min << ".." << c.table.x_max << ", y "
              << c.table.y_min << ".." << c.table.y_max;
          }
          e << "). Is the box's yaw right (the push follows the nose, box yaw "
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
  // the landing headings: as typed -- or, with the table placed from the box,
  // turned with the box (2026-10-22): the typed yaw keeps its offset from the
  // push heading (-360 from -90 = 270 deg clockwise, whichever way the table
  // stands), so the vehicle lands facing the table as it took off
  const double yaw_shift = c.table_from_box ? wp.drone_yaw - c.yaw_from_box : 0.0;
  wp.land.x_b = t.land.p;
  wp.land.phi = t.land.yaw + yaw_shift - 0.5 * M_PI;
  wp.land.q = c.home;
  wp.land_start = wp.land;
  if (t.land_start_set) {
    wp.land_start.x_b = t.land_start.p;
    wp.land_start.phi = t.land_start.yaw + yaw_shift - 0.5 * M_PI;
  }
  wp.table = c.table;
  wp.obstacle = c.obstacle;
  // THE LAYOUT (2026-10-22, the table placed from the box): the vehicle starts
  // on the table's middle line, in front of it (the user's rule) -- a start off
  // that line means the box sits elsewhere than at the table's end, or its yaw
  // (and with it the table) is turned (a box or mocap body read 180 deg off puts
  // the table's middle 0.56 m along from the real one)
  if (c.table_from_box && t.start_set) {
    const Vec3 s = c.table.toLocal(t.start.p);
    const double mid = 0.5 * (c.table.x_min + c.table.x_max);
    const double off = s(0) - mid;
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    if (c.start_middle_tol > 0.0 && std::abs(off) > c.start_middle_tol) {
      m << "layout: the start (" << t.start.p(0) << ", " << t.start.p(1) << ") is " << std::abs(off)
        << " m " << (off > 0.0 ? "along the push from" : "back from") << " the table's middle line "
        << "(push_pull_start_middle_tol " << c.start_middle_tol << " m). The table is placed from "
        << "the box -- its back face " << c.table_box_gap << " m in from the table's end, the push "
        << "along yaw " << std::setprecision(0) << wp.drone_yaw * kDeg << std::setprecision(2)
        << " deg: put the vehicle on the table's middle line and press Adjust on the ground; "
        << "or is the box at the table's other end, or its yaw wrong?";
      throw std::runtime_error(m.str());
    }
    const double side = std::max(c.table.y_min - s(1), s(1) - c.table.y_max);
    if (c.bypass && side < c.bypass_clearance + kBypassStartMargin) {
      m << "layout: the start (" << t.start.p(0) << ", " << t.start.p(1) << ") is " << side
        << " m from the table's side -- Go To Start goes round the table keeping "
        << c.bypass_clearance << " m (push_pull_bypass_clearance): start the vehicle at least "
        << c.bypass_clearance + kBypassStartMargin << " m in front of the table";
      throw std::runtime_error(m.str());
    }
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
  // the short way round to the push heading -- and (2026-10-22, c.bypass) round
  // the table's box end, seen from above, unless the hold is already near it
  return step(v, c, "go_to_start", [&]() -> std::unique_ptr<Trajectory> {
      if (c.bypass && c.table.valid() && c.turn_angle == 0.0) {
        if (auto round = bypassGoToStart(v, transition, opts, c, wp, from)) {return round;}
        std::ostringstream m;
        m.setf(std::ios::fixed);
        m.precision(2);
        m << "(straight: the hold is " << c.table.distanceXY(from.x_b)
          << " m from the table, within push_pull_bypass_clearance " << c.bypass_clearance << ") ";
        return std::make_unique<NotedTrajectory>(
          transitionLeg(v, transition, opts, from, wp.approach), m.str());
      }
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

double pushPullPushTime(const PushPullConfig & c)
{
  if (!(std::isfinite(c.push_speed) && c.push_speed > 0.0)) {
    std::ostringstream m;
    m << "push: push_pull_push_mean_speed is not set (" << c.push_speed
      << " m/s) -- set it in the yaml: a straight push's duration is distance / speed";
    throw std::runtime_error(m.str());
  }
  const double floor = std::isfinite(c.push_min_time) ? std::max(c.push_min_time, 0.0) : 0.0;
  return std::max(std::abs(c.push_distance) / c.push_speed, floor);
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
  // THE PUSH SPEED (2026-10-21): a straight push's duration is the speed's
  // (push_time is a turn's only)
  const double T = turn ? c.push_time : pushPullPushTime(c);
  const bool cruise = !turn && c.push_ramp > 0.0;
  if (cruise && !(T > 2.0 * c.push_ramp)) {
    std::ostringstream m;
    m << "push: push_pull_push_ramp_s (" << c.push_ramp
      << " s) must be less than half the push time (" << T << " s)";
    throw std::runtime_error(m.str());
  }
  // (2026-10-21 review) the flat planner clamps a leg's duration to T_max without
  // saying so: a slower push would fly faster than the time Plan and the GS show
  if (!turn && !cruise && T > opts.T_max) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m << std::setprecision(2) << "push: " << std::abs(c.push_distance) << " m at "
      << std::setprecision(1) << c.push_speed * 1e3 << " mm/s takes " << T
      << " s, longer than the planner's t_max (" << opts.T_max
      << " s) -- raise push_pull_push_mean_speed or shorten the Push Distance";
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
          v.params, from, noseOf(from.phi), c.push_distance, T, c.push_ramp));
      } else {
        segs.push_back(transitionLeg(
          v, transition, timedOptions(opts, c.push_distance, T), from, goal));
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
      // (2026-10-22, user) round the table like Go To Start, backwards
      if (c.bypass && c.table.valid() && c.turn_angle == 0.0) {
        return bypassGoToLandStart(v, transition, opts, c, wp, from, goal, turn);
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
  const Vec3 & retreat, const std::optional<VecN> & unbuckle)
{
  RestSpec back = from;
  back.x_b += Vec3(retreat(0), retreat(1), 0.0);
  RestSpec up = back;
  up.x_b(2) += std::max(0.0, climb);
  const bool retreating = (back.x_b - from.x_b).norm() > 1e-3;
  // THE UNBUCKLE (2026-10-21): the first motion leg brings the arm to it -- the
  // retreat, unless that would lower the claw against the base by more than
  // 10 mm (the open fingers would slide DOWN the handle at its height; the
  // retreat is level, so the base's z is the rests' z); the climb outruns any
  // such drop. Without such a leg the fold home does it alone.
  if (unbuckle.has_value()) {
    Vec3 e0, e1;
    armKinematics(from.q, v.params, nullptr, &e0, nullptr);
    armKinematics(*unbuckle, v.params, nullptr, &e1, nullptr);
    const bool drops = e1(2) < e0(2) - 0.010;
    if (retreating && !drops) {
      back.q = *unbuckle;
      up.q = *unbuckle;
    } else if (climb > 1e-3) {
      up.q = *unbuckle;   // the climb carries it (back.q is still from.q)
    }
  }
  RestSpec home = up;
  home.q = c.home;
  return step(v, c, "abort", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (release_wait > 0.0) {
        segs.push_back(std::make_shared<PauseTrajectory>(v.params, from, release_wait));
      }
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
  const PushPullConfig & c_in, const PushPullWaypoints & wp, const RestSpec & from)
{
  PushPullConfig c = c_in;
  c.table = wp.table;   // the plan's table (2026-10-22: placed from the box, or typed)
  c.obstacle = wp.obstacle;   // and obstacle (2026-10-27: placed from the valve, or typed)
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

bool boxSampleWithinGate(double d_m, double a_deg, double dt, double jump_m, double jump_deg)
{
  const bool pos_on = jump_m > 0.0, ang_on = jump_deg > 0.0;
  if (!pos_on && !ang_on) {return true;}               // the gate off: every sample
  dt = std::max(dt, 0.0);
  if (pos_on && !(d_m <= jump_m + kBoxGateV * dt + 0.5 * kBoxGateA * dt * dt)) {return false;}
  if (ang_on && !(a_deg <= jump_deg + kBoxGateW * dt)) {return false;}   // NaN is never good
  return true;
}

}  // namespace fsc_trajectory_planner
