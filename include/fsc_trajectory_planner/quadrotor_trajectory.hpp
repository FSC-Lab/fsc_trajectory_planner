// MIT License
// Copyright (c) 2026 FSC Lab
//
// Periodic QUADROTOR trajectories (hover, circle, figure-8) for the torque-mode
// (R)CCM pipeline, as analytic position jets up to snap.
//
// Heading: "circle" / "figure8" keep the heading the vehicle holds when the run
// starts (constant yaw). "circle_tangent" / "figure8_tangent" fly the same paths
// with the yaw ALONG THE PATH: psi = atan2 of the path tangent ds/dtheta, which is
// defined even at rest (theta' = 0), with analytic yaw rate and acceleration
//   psi' = (x' y'' - y' x'') / (x'^2 + y'^2)  (d/dtheta),
//   dpsi/dt = psi' theta',  d2psi/dt2 = psi'' theta'^2 + psi' theta''.
// The run starts at the tangent heading of s(0) (circle 90 deg ccw, figure-8
// 45 deg ccw), so the fly-to-start move rotates the vehicle there first.
//
// A shape is a closed planar curve s(theta), theta in [0, 2 pi) per lap, with
// s(0) = 0, ANCHORED at its start point: the run begins exactly there. The node
// places the shape by its CENTRE (quadShapeStartOffset gives start - centre) and
// flies the vehicle to the start first with a QuadTransition. The phase is
//
//   theta'(t) = Omega * rho(t),   Omega = 2 pi / lap_time
//
// with rho a septic (min-snap) ramp 0 -> 1 over ramp_time, 1 while cruising and
// 1 -> 0 over ramp_time at the end. The septic ramp has zero first..third
// derivatives at both ends, so velocity, acceleration, jerk AND snap -- hence
// the reference attitude, body rates, thrust and torques -- start and end at
// their hover values. That is the fsc_trajectory_planner rule that a task is
// never prescribed without min-snap phasing. The cruise length is chosen so the
// run covers exactly `laps` laps and ends back at the anchor.

#ifndef FSC_TRAJECTORY_PLANNER_QUADROTOR_TRAJECTORY_HPP_
#define FSC_TRAJECTORY_PLANNER_QUADROTOR_TRAJECTORY_HPP_

#include <Eigen/Dense>

#include <string>

#include "fsc_trajectory_planner/quadrotor_model.hpp"

namespace fsc_trajectory_planner
{

enum class QuadShape { kHover, kCircle, kFigure8, kCircleTangent, kFigure8Tangent };

// "hover" | "circle" | "figure8" | "circle_tangent" | "figure8_tangent";
// throws std::runtime_error otherwise.
QuadShape parseQuadShape(const std::string & name);
std::string quadShapeName(QuadShape shape);
// The path of a shape (the tangent variants fly their base shape's path).
QuadShape quadShapeGeometry(QuadShape shape);
// True for the shapes whose yaw follows the path tangent.
bool quadShapeYawTangent(QuadShape shape);
// Smallest signed angle a - b, in (-pi, pi].
double wrapAngle(double a);

struct QuadTrajectoryOptions
{
  double circle_radius{1.0};  // [m]
  double fig8_a{1.0};         // [m] x amplitude; y amplitude is a / 2
  double lap_time{8.0};       // [s] per lap at cruise
  int laps{2};
  double ramp_time{3.0};      // [s] septic ramp in and out
  bool ccw{true};             // counter-clockwise seen from above (+z)
  // Speed factor of the time-parameterised shapes: the cruise lap takes
  // lap_time / time_scale (ramps keep ramp_time).
  double time_scale{1.0};
};

// Start point minus shape centre: circle (R, 0, 0), figure-8 (0, 0, 0).
Eigen::Vector3d quadShapeStartOffset(QuadShape shape, const QuadTrajectoryOptions & opts);

class QuadTrajectory
{
public:
  // `anchor`: world ENU point where the run starts and (after whole laps) ends.
  // Throws std::runtime_error on invalid options (non-positive lap_time /
  // ramp_time / laps, or a ramp longer than the whole run).
  QuadTrajectory(QuadShape shape, const QuadTrajectoryOptions & opts, const Eigen::Vector3d & anchor);

  // Position jets at time t since the start. t <= 0 gives the anchor at rest;
  // t >= duration() gives the final point at rest.
  PositionJets jets(double t) const;

  // Phase theta(t) and its derivatives [theta, theta', ..., theta''''].
  Eigen::Matrix<double, 5, 1> phase(double t) const;

  // Yaw jets at t: the path tangent for the *_tangent shapes (psi wrapped to
  // (-pi, pi]), otherwise the constant `yaw_hold`.
  YawJets yawJets(double t, double yaw_hold) const;
  double startYaw(double yaw_hold) const {return yawJets(0.0, yaw_hold).psi;}
  // Peak |dpsi/dt| [rad/s] and |d2psi/dt2| [rad/s^2] over the whole run
  // (ramps included; 0 for constant-yaw shapes).
  double yawRateMax() const;
  double yawAccelMax() const;

  double duration() const {return t_down_ + opts_.ramp_time;}
  bool finished(double t) const {return t >= duration();}
  // Begin the ramp-out at t (or at the end of the ramp-in if t is earlier).
  // No effect once the ramp-out has started. The run then ends mid-lap.
  void requestStop(double t);

  QuadShape shape() const {return shape_;}
  const Eigen::Vector3d & anchor() const {return anchor_;}
  // Peak speed and acceleration magnitude over the cruise [m/s], [m/s^2].
  double cruiseSpeedMax() const;
  double cruiseAccelMax() const;

private:
  // Shape offset from the anchor and its first four theta-derivatives.
  void shapeDerivs(double theta, Eigen::Vector3d d[5]) const;

  QuadShape shape_;
  QuadTrajectoryOptions opts_;
  Eigen::Vector3d anchor_;
  double omega_{0.0};   // cruise phase rate [rad/s]
  double t_down_{0.0};  // start of the ramp-out [s]
};

// Rest-to-rest straight-line move p0 -> p1, yaw yaw0 -> yaw1 (the shorter way
// round), with NONIC (9th-order smoothstep) phasing: velocity, acceleration, jerk
// and snap -- and yaw rate and acceleration -- all vanish at both ends, so the
// reference attitude, body rates, thrust and torques are continuous when the move
// starts and ends. Duration = max(t_min, 2.4609 |p1 - p0| / v_peak,
// 2.4609 |yaw1 - yaw0| / yaw_rate_peak), so neither peak rate is exceeded.
class QuadTransition
{
public:
  QuadTransition(
    const Eigen::Vector3d & p0, const Eigen::Vector3d & p1, double v_peak, double t_min,
    double yaw0 = 0.0, double yaw1 = 0.0, double yaw_rate_peak = 0.5);
  PositionJets jets(double t) const;
  YawJets yawJets(double t) const;
  double duration() const {return T_;}
  const Eigen::Vector3d & goal() const {return p1_;}
  double goalYaw() const {return wrapAngle(yaw0_ + dyaw_);}

private:
  Eigen::Vector3d p0_, p1_;
  double yaw0_{0.0}, dyaw_{0.0};
  double T_{1.0};
};

// A complete CCM reference sample: x*, u* and the actuator-lag command
//   u*_c = u* + tau * du*/dt   (tau = model.rotor_time_constant),
// the derivative by a central difference of step h on the analytic reference.
struct CcmReferenceSample
{
  QuadrotorReference ref;
  double thrust_command{0.0};
  Eigen::Vector3d torque_command{Eigen::Vector3d::Zero()};
};

// traj == nullptr: hover at `hold_point` with heading `yaw`. With a trajectory,
// `yaw` is the heading of the constant-yaw shapes (the tangent shapes set their own).
CcmReferenceSample sampleCcmReference(
  const QuadrotorModel & model, const QuadTrajectory * traj, const Eigen::Vector3d & hold_point,
  double yaw, double t, const Eigen::Vector3d & disturbance = Eigen::Vector3d::Zero(), double h = 1e-3,
  const Eigen::Vector3d & angular_disturbance = Eigen::Vector3d::Zero());
// Same for a transition (its yaw comes from the move).
CcmReferenceSample sampleCcmReference(
  const QuadrotorModel & model, const QuadTransition & move, double t,
  const Eigen::Vector3d & disturbance = Eigen::Vector3d::Zero(), double h = 1e-3,
  const Eigen::Vector3d & angular_disturbance = Eigen::Vector3d::Zero());

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_QUADROTOR_TRAJECTORY_HPP_
