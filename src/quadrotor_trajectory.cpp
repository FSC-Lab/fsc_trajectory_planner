// MIT License
// Copyright (c) 2026 FSC Lab

#include "fsc_trajectory_planner/quadrotor_trajectory.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fsc_trajectory_planner
{

namespace
{

constexpr double kTwoPi = 2.0 * M_PI;

// Septic smoothstep h(x) = 35x^4 - 84x^5 + 70x^6 - 20x^7 on [0, 1]:
// h(0) = 0, h(1) = 1, h', h'', h''' vanish at both ends; H = int_0^x h.
struct Septic
{
  double H, h, h1, h2, h3;
};

Septic septic(double x)
{
  x = std::clamp(x, 0.0, 1.0);
  const double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x, x6 = x5 * x, x7 = x6 * x, x8 = x7 * x;
  Septic s;
  s.H = 7.0 * x5 - 14.0 * x6 + 10.0 * x7 - 2.5 * x8;
  s.h = 35.0 * x4 - 84.0 * x5 + 70.0 * x6 - 20.0 * x7;
  s.h1 = 140.0 * x3 - 420.0 * x4 + 420.0 * x5 - 140.0 * x6;
  s.h2 = 420.0 * x2 - 1680.0 * x3 + 2100.0 * x4 - 840.0 * x5;
  s.h3 = 840.0 * x - 5040.0 * x2 + 8400.0 * x3 - 4200.0 * x4;
  return s;
}

}  // namespace

QuadShape parseQuadShape(const std::string & name)
{
  if (name == "hover") {return QuadShape::kHover;}
  if (name == "circle") {return QuadShape::kCircle;}
  if (name == "figure8") {return QuadShape::kFigure8;}
  if (name == "circle_tangent") {return QuadShape::kCircleTangent;}
  if (name == "figure8_tangent") {return QuadShape::kFigure8Tangent;}
  throw std::runtime_error(
          "unknown quadrotor trajectory '" + name +
          "' (hover | circle | figure8 | circle_tangent | figure8_tangent)");
}

std::string quadShapeName(QuadShape shape)
{
  switch (shape) {
    case QuadShape::kCircle: return "circle";
    case QuadShape::kFigure8: return "figure8";
    case QuadShape::kCircleTangent: return "circle_tangent";
    case QuadShape::kFigure8Tangent: return "figure8_tangent";
    default: return "hover";
  }
}

QuadShape quadShapeGeometry(QuadShape shape)
{
  if (shape == QuadShape::kCircleTangent) {return QuadShape::kCircle;}
  if (shape == QuadShape::kFigure8Tangent) {return QuadShape::kFigure8;}
  return shape;
}

bool quadShapeYawTangent(QuadShape shape)
{
  return shape == QuadShape::kCircleTangent || shape == QuadShape::kFigure8Tangent;
}

double wrapAngle(double a)
{
  a = std::fmod(a + M_PI, kTwoPi);
  if (a <= 0.0) {a += kTwoPi;}
  return a - M_PI;
}

Eigen::Vector3d quadShapeStartOffset(QuadShape shape, const QuadTrajectoryOptions & opts)
{
  // shapeDerivs() puts the circle's centre at anchor - R e_x and the figure-8's
  // crossing point on the anchor.
  if (quadShapeGeometry(shape) == QuadShape::kCircle) {return {opts.circle_radius, 0.0, 0.0};}
  return Eigen::Vector3d::Zero();
}

QuadTrajectory::QuadTrajectory(
  QuadShape shape, const QuadTrajectoryOptions & opts, const Eigen::Vector3d & anchor)
: shape_(shape), opts_(opts), anchor_(anchor)
{
  if (!(opts_.lap_time > 0.0) || !(opts_.ramp_time > 0.0) || opts_.laps < 1 || !(opts_.time_scale > 0.0)) {
    throw std::runtime_error("QuadTrajectory: lap_time, ramp_time, laps and time_scale must be positive");
  }
  const double lap = opts_.lap_time / opts_.time_scale;
  omega_ = kTwoPi / lap;
  // Ramps each cover half their length at cruise rate: total angle
  //   Omega * (ramp/2 + cruise + ramp/2) = 2 pi laps  ->  cruise = laps*lap - ramp.
  const double cruise = opts_.laps * lap - opts_.ramp_time;
  if (cruise < 0.0) {
    throw std::runtime_error("QuadTrajectory: ramp_time longer than laps * lap_time");
  }
  t_down_ = opts_.ramp_time + cruise;
}

void QuadTrajectory::requestStop(double t)
{
  if (t >= t_down_) {return;}  // ramp-out already running (or over)
  t_down_ = std::max(t, opts_.ramp_time);
}

Eigen::Matrix<double, 5, 1> QuadTrajectory::phase(double t) const
{
  Eigen::Matrix<double, 5, 1> th = Eigen::Matrix<double, 5, 1>::Zero();
  const double Tr = opts_.ramp_time, W = omega_;
  if (t <= 0.0) {
    return th;
  }
  if (t < Tr) {
    const Septic s = septic(t / Tr);
    th << W * Tr * s.H, W * s.h, W * s.h1 / Tr, W * s.h2 / (Tr * Tr), W * s.h3 / (Tr * Tr * Tr);
  } else if (t < t_down_) {
    th << W * (0.5 * Tr + (t - Tr)), W, 0.0, 0.0, 0.0;
  } else if (t < t_down_ + Tr) {
    const double x = (t - t_down_) / Tr;
    const Septic s = septic(x);
    th << W * (0.5 * Tr + (t_down_ - Tr)) + W * Tr * (x - s.H), W * (1.0 - s.h), -W * s.h1 / Tr,
      -W * s.h2 / (Tr * Tr), -W * s.h3 / (Tr * Tr * Tr);
  } else {
    th << W * t_down_, 0.0, 0.0, 0.0, 0.0;
  }
  return th;
}

void QuadTrajectory::shapeDerivs(double theta, Eigen::Vector3d d[5]) const
{
  const double sy = opts_.ccw ? 1.0 : -1.0;
  const double c = std::cos(theta), s = std::sin(theta);
  switch (quadShapeGeometry(shape_)) {
    case QuadShape::kCircle: {
      // s(theta) = R [cos(theta) - 1, sy sin(theta), 0]: starts at the anchor
      // heading +y (ccw), centre at anchor - R e_x.
      const double R = opts_.circle_radius;
      d[0] << R * (c - 1.0), sy * R * s, 0.0;
      d[1] << -R * s, sy * R * c, 0.0;
      d[2] << -R * c, -sy * R * s, 0.0;
      d[3] << R * s, -sy * R * c, 0.0;
      d[4] << R * c, sy * R * s, 0.0;
      break;
    }
    case QuadShape::kFigure8: {
      // s(theta) = [a sin(theta), sy (a/2) sin(2 theta), 0]: the crossing
      // point is the anchor.
      const double a = opts_.fig8_a;
      const double c2 = std::cos(2.0 * theta), s2 = std::sin(2.0 * theta);
      d[0] << a * s, sy * 0.5 * a * s2, 0.0;
      d[1] << a * c, sy * a * c2, 0.0;
      d[2] << -a * s, -sy * 2.0 * a * s2, 0.0;
      d[3] << -a * c, -sy * 4.0 * a * c2, 0.0;
      d[4] << a * s, sy * 8.0 * a * s2, 0.0;
      break;
    }
    default:
      for (int k = 0; k < 5; ++k) {d[k].setZero();}
  }
}

PositionJets QuadTrajectory::jets(double t) const
{
  const auto th = phase(t);
  Eigen::Vector3d d[5];
  shapeDerivs(th[0], d);
  const double t1 = th[1], t2 = th[2], t3 = th[3], t4 = th[4];
  // Faa di Bruno for p(theta(t)).
  PositionJets j;
  j.p = anchor_ + d[0];
  j.v = d[1] * t1;
  j.a = d[2] * t1 * t1 + d[1] * t2;
  j.j = d[3] * t1 * t1 * t1 + 3.0 * d[2] * t1 * t2 + d[1] * t3;
  j.s = d[4] * t1 * t1 * t1 * t1 + 6.0 * d[3] * t1 * t1 * t2 + d[2] * (3.0 * t2 * t2 + 4.0 * t1 * t3) +
    d[1] * t4;
  return j;
}

YawJets QuadTrajectory::yawJets(double t, double yaw_hold) const
{
  YawJets y;
  y.psi = yaw_hold;
  if (!quadShapeYawTangent(shape_)) {return y;}
  const auto th = phase(t);
  Eigen::Vector3d d[5];
  shapeDerivs(th[0], d);
  // Planar tangent (x', y') and its theta-derivatives; |(x', y')| > 0 on both paths.
  const double x1 = d[1].x(), y1 = d[1].y(), x2 = d[2].x(), y2 = d[2].y(), x3 = d[3].x(), y3 = d[3].y();
  const double D = x1 * x1 + y1 * y1;
  const double N = x1 * y2 - y1 * x2;
  const double dN = x1 * y3 - y1 * x3;           // the x'' y'' terms cancel
  const double dD = 2.0 * (x1 * x2 + y1 * y2);
  const double p1 = N / D;                       // dpsi/dtheta
  const double p2 = (dN * D - N * dD) / (D * D);  // d2psi/dtheta2
  y.psi = std::atan2(y1, x1);
  y.dpsi = p1 * th[1];
  y.ddpsi = p2 * th[1] * th[1] + p1 * th[2];
  return y;
}

double QuadTrajectory::yawRateMax() const
{
  if (!quadShapeYawTangent(shape_)) {return 0.0;}
  double r = 0.0;
  constexpr int kN = 4000;
  for (int k = 0; k <= kN; ++k) {
    r = std::max(r, std::abs(yawJets(duration() * k / kN, 0.0).dpsi));
  }
  return r;
}

double QuadTrajectory::yawAccelMax() const
{
  if (!quadShapeYawTangent(shape_)) {return 0.0;}
  double r = 0.0;
  constexpr int kN = 4000;
  for (int k = 0; k <= kN; ++k) {
    r = std::max(r, std::abs(yawJets(duration() * k / kN, 0.0).ddpsi));
  }
  return r;
}

double QuadTrajectory::cruiseSpeedMax() const
{
  double v = 0.0;
  for (int k = 0; k < 360; ++k) {
    Eigen::Vector3d d[5];
    shapeDerivs(kTwoPi * k / 360.0, d);
    v = std::max(v, d[1].norm() * omega_);
  }
  return v;
}

double QuadTrajectory::cruiseAccelMax() const
{
  double a = 0.0;
  for (int k = 0; k < 360; ++k) {
    Eigen::Vector3d d[5];
    shapeDerivs(kTwoPi * k / 360.0, d);
    a = std::max(a, d[2].norm() * omega_ * omega_);
  }
  return a;
}

QuadTransition::QuadTransition(
  const Eigen::Vector3d & p0, const Eigen::Vector3d & p1, double v_peak, double t_min,
  double yaw0, double yaw1, double yaw_rate_peak)
: p0_(p0), p1_(p1), yaw0_(yaw0), dyaw_(wrapAngle(yaw1 - yaw0))
{
  if (!(v_peak > 0.0) || !(t_min > 0.0) || !(yaw_rate_peak > 0.0)) {
    throw std::runtime_error("QuadTransition: v_peak, t_min and yaw_rate_peak must be positive");
  }
  // Peak of the nonic's rate is 630 / 256 = 2.4609 at mid-move.
  T_ = std::max({t_min, 2.4609375 * (p1 - p0).norm() / v_peak, 2.4609375 * std::abs(dyaw_) / yaw_rate_peak});
}

YawJets QuadTransition::yawJets(double t) const
{
  const double x = std::clamp(t / T_, 0.0, 1.0);
  const double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x, x6 = x5 * x, x7 = x6 * x, x8 = x7 * x;
  const double h = 126 * x5 - 420 * x6 + 540 * x7 - 315 * x8 + 70 * x8 * x;
  const double h1 = 630 * x4 - 2520 * x5 + 3780 * x6 - 2520 * x7 + 630 * x8;
  const double h2 = 2520 * x3 - 12600 * x4 + 22680 * x5 - 17640 * x6 + 5040 * x7;
  const bool moving = t > 0.0 && t < T_;
  YawJets y;
  y.psi = wrapAngle(yaw0_ + dyaw_ * h);
  y.dpsi = moving ? dyaw_ * h1 / T_ : 0.0;
  y.ddpsi = moving ? dyaw_ * h2 / (T_ * T_) : 0.0;
  return y;
}

PositionJets QuadTransition::jets(double t) const
{
  const double x = std::clamp(t / T_, 0.0, 1.0);
  const double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x, x6 = x5 * x, x7 = x6 * x, x8 = x7 * x;
  const double x9 = x8 * x;
  // Nonic smoothstep h = 126x^5 - 420x^6 + 540x^7 - 315x^8 + 70x^9 and derivatives.
  const double h = 126 * x5 - 420 * x6 + 540 * x7 - 315 * x8 + 70 * x9;
  const double h1 = 630 * x4 - 2520 * x5 + 3780 * x6 - 2520 * x7 + 630 * x8;
  const double h2 = 2520 * x3 - 12600 * x4 + 22680 * x5 - 17640 * x6 + 5040 * x7;
  const double h3 = 7560 * x2 - 50400 * x3 + 113400 * x4 - 105840 * x5 + 35280 * x6;
  const double h4 = 15120 * x - 151200 * x2 + 453600 * x3 - 529200 * x4 + 211680 * x5;
  const Eigen::Vector3d d = p1_ - p0_;
  const bool moving = t > 0.0 && t < T_;
  PositionJets j;
  j.p = p0_ + d * h;
  j.v = moving ? Eigen::Vector3d(d * h1 / T_) : Eigen::Vector3d::Zero();
  j.a = moving ? Eigen::Vector3d(d * h2 / (T_ * T_)) : Eigen::Vector3d::Zero();
  j.j = moving ? Eigen::Vector3d(d * h3 / (T_ * T_ * T_)) : Eigen::Vector3d::Zero();
  j.s = moving ? Eigen::Vector3d(d * h4 / (T_ * T_ * T_ * T_)) : Eigen::Vector3d::Zero();
  return j;
}

CcmReferenceSample sampleCcmReference(
  const QuadrotorModel & model, const QuadTransition & move, double t,
  const Eigen::Vector3d & disturbance, double h, const Eigen::Vector3d & angular_disturbance)
{
  const auto at = [&](double tt) {
      return flatReference(model, move.jets(tt), move.yawJets(tt), disturbance, angular_disturbance);
    };
  CcmReferenceSample out;
  out.ref = at(t);
  out.thrust_command = out.ref.thrust;
  out.torque_command = out.ref.torque;
  if (model.rotor_time_constant > 0.0) {
    const QuadrotorReference rp = at(t + h), rm = at(t - h);
    out.thrust_command += model.rotor_time_constant * (rp.thrust - rm.thrust) / (2.0 * h);
    out.torque_command += model.rotor_time_constant * (rp.torque - rm.torque) / (2.0 * h);
  }
  return out;
}

CcmReferenceSample sampleCcmReference(
  const QuadrotorModel & model, const QuadTrajectory * traj, const Eigen::Vector3d & hold_point,
  double yaw, double t, const Eigen::Vector3d & disturbance, double h,
  const Eigen::Vector3d & angular_disturbance)
{
  const auto at = [&](double tt) {
      PositionJets pj;
      YawJets yj;
      yj.psi = yaw;
      if (traj != nullptr) {
        pj = traj->jets(tt);
        yj = traj->yawJets(tt, yaw);
      } else {
        pj.p = hold_point;
      }
      return flatReference(model, pj, yj, disturbance, angular_disturbance);
    };
  CcmReferenceSample out;
  out.ref = at(t);
  out.thrust_command = out.ref.thrust;
  out.torque_command = out.ref.torque;
  if (model.rotor_time_constant > 0.0 && traj != nullptr) {
    const QuadrotorReference rp = at(t + h), rm = at(t - h);
    out.thrust_command += model.rotor_time_constant * (rp.thrust - rm.thrust) / (2.0 * h);
    out.torque_command += model.rotor_time_constant * (rp.torque - rm.torque) / (2.0 * h);
  }
  return out;
}

}  // namespace fsc_trajectory_planner
