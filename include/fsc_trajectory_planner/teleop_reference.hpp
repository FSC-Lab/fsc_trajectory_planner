// MIT License
// Copyright (c) 2026 FSC Lab
//
// PS4 TELEOPERATION of the whole-body DIRECT controller (2026-09-27, user
// design): the pad commands VELOCITIES, they are integrated into position
// setpoints, and every setpoint is a full, compatible WholeBodyReference --
// streamed sample by sample, no trajectory optimisation online.
//
// THE FLAT OUTPUTS ARE DEFINED BY THE PAD, ALL EIGHT OF THEM:
//     system CoM x_c (3) + platform heading psi (1)   <- D-pad + action buttons
//     arm joints q (4)                                <- the two sticks
// The sticks move the end-effector RELATIVE TO THE AIRFRAME, in the body
// frame: the grasp point's position s = r_0e(q) (3, through a position IK on
// q1..q3 -- q4 is coaxial with the grasp point and cannot move it) plus the
// WRIST ROLL q4 directly. That is exactly joint space, so the (x_c, psi, q)
// triple -- the flat outputs of the bspline planner -- is fully specified and
// flatState() turns it into the law's reference with no fixed point: the
// thrust axis follows xdd_c alone, and the end-effector reference is the
// forward kinematics of the whole chain. Compatible by construction.
//
// SMOOTHING. Each raw target is filtered by a chain of n identical first-
// order lags (critically damped, bandwidth w). Discretised EXACTLY (zero-order
// hold), so the stages are always a CONVEX COMBINATION of past raw targets:
// the output cannot overshoot a wall the raw target stopped at, and a joint
// output cannot leave the joint box its raw target stayed inside. The
// derivatives come from the stage states in closed form,
//     y_n^(k) = w^k sum_j C(k, j) (-1)^j y_(n-k+j),   y_0 = the raw target,
// and every derivative flatState consumes comes from the STAGES alone -- the
// CoM (n = 5) through snap, the heading and joints (n = 3) through their
// second derivative -- so all of them are continuous. (The n-th derivative
// uses the raw target itself and steps at every tick, since that is held
// constant between ticks: with n = 4 the CoM snap jumped by w^4 v dt per
// tick, 1.25 m/s^4 at 0.2 m/s -- jitter straight into the attitude
// feedforward. Hence one stage more than the snap needs.)
//
// TIME. The generator runs in PHYSICAL time; reference(s) returns derivatives
// with respect to the caller's clock with dtau = s dt (s = 1 on hardware; the
// sim's real-time factor in Isaac, where the planner runs on the wall clock).

#ifndef FSC_TRAJECTORY_PLANNER_TELEOP_REFERENCE_HPP_
#define FSC_TRAJECTORY_PLANNER_TELEOP_REFERENCE_HPP_

#include <memory>
#include <optional>
#include <string>

#include <Eigen/Dense>

#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"
#include "fsc_trajectory_planner/wb_law.hpp"

namespace fsc_trajectory_planner
{

// n identical first-order stages, y_i' = w (y_(i-1) - y_i), y_0 = u.
class ChainFilter
{
public:
  void setup(int order, double bandwidth, int dim);
  void reset(const Eigen::VectorXd & x);
  // Exact ZOH step with the input held at u over dt.
  void step(const Eigen::VectorXd & u, double dt);
  // k-th time derivative of the output, k = 0..order (k = order uses u).
  Eigen::VectorXd deriv(int k) const;
  // The output has settled on its input: every stage within tol of u.
  bool settled(double tol) const;
  int order() const {return n_;}
  double bandwidth() const {return w_;}

private:
  int n_{0};
  int dim_{0};
  double w_{1.0};
  Eigen::MatrixXd y_;        // n x dim, row i-1 = stage i
  Eigen::VectorXd u_;        // the input held over the last step
};

struct TeleopOptions
{
  // Rates at FULL deflection (physical units).
  double com_speed_xy{0.20};     // m/s   D-pad
  double com_speed_z{0.15};      // m/s   triangle / cross
  double yaw_rate{0.35};         // rad/s square / circle
  double ee_speed{0.04};         // m/s   sticks, grasp point vs airframe
  double roll_rate{0.35};        // rad/s right stick X, wrist roll q4
  double home_rate{0.17};        // rad/s per joint while the arm homes
  // Smoother bandwidths [rad/s].
  double com_bandwidth{6.0};
  double yaw_bandwidth{4.0};
  double arm_bandwidth{6.0};
  // Walls on the RAW targets (the smoother then cannot cross them).
  double box_half_xy{1.5};       // m about the engage point; <= 0: off
  double com_z_min{0.70};        // m world
  double com_z_max{2.20};        // m world
  double ee_z_min{0.25};         // m world, grasp point at rest
  double leash{0.50};            // m |raw CoM - measured CoM|; <= 0: off
  // The pad's joint box: each joint's range SHRUNK about its centre to this
  // fraction (0.8 = 20 % kept in reserve, 10 % at each end -- user,
  // 2026-09-27), so the operator never drives the arm onto a hardware stop.
  // A target already outside it (the folded home has q2 = 40 deg against the
  // box's 37) may only move back toward it, never further out.
  double joint_range_frac{0.8};
  // Where PS folds the arm: the PAD'S home, INSIDE that box with room in
  // every direction (fold 60 deg: ~9 cm left/right/down, 4 up, 1.5 fwd/back).
  // The stowed flight home [0, 40, 40, 0] deg is NOT inside the 0.8 box
  // (q2 40 > 37), and from it the pad can only move the grasp point down.
  VecN home{(VecN() << 0.0, 30.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, 0.0).finished()};
  double q3_min{0.0};            // rad: the positive-fold branch guard
};

// Operator input, each channel in [-1, 1].
struct TeleopInput
{
  Vec3 com{Vec3::Zero()};        // (forward, left, up), vehicle HEADING frame
  double yaw{0.0};               // + = counter-clockwise from above
  Vec3 ee{Vec3::Zero()};         // (forward, left, up), airframe BODY frame
  double roll{0.0};              // + = +q4
  bool home{false};              // start folding the arm to the home pose
};

struct TeleopStatus
{
  unsigned com_blocked{0};       // bits 0/1/2: world x/y/z step refused
  unsigned arm_blocked{0};       // bits 0/1/2: body fwd/left/up, bit 3: roll
  bool leash_active{false};
  bool homing{false};
  double sigma_nd{0.0};          // singularity margin at the raw joint target
  std::string wall;              // last refusal, operator-readable ("" = none)
};

class TeleopReference
{
public:
  TeleopReference(std::shared_ptr<const VehicleModel> vehicle, const TeleopOptions & opts);

  // Seed at a REST (the hold the planner is streaming): CoM, MODEL heading,
  // joints. Every stage at rest there, so the first sample equals the hold.
  void seed(const Vec3 & x_c, double phi, const VecN & q);

  // Advance by dt PHYSICAL seconds. x_c_meas (measured system CoM, fresh
  // only) drives the leash; nullopt switches it off for this step.
  void step(double dt, const TeleopInput & in, const std::optional<Vec3> & x_c_meas);

  // The law's reference, derivatives on the caller's clock (dtau = s dt).
  WbReference reference(double time_scale) const;

  // Filters settled on the raw targets (the output is a rest).
  bool settled(double pos_tol = 1e-4, double ang_tol = 1e-4) const;
  // The rest the raw targets describe (base position, heading, joints).
  RestSpec targetRest() const;

  // Raw targets, for display.
  const Vec3 & comTarget() const {return com_raw_;}
  double phiTarget() const {return phi_raw_;}
  const VecN & qTarget() const {return q_raw_;}
  const Vec3 & sTarget() const {return s_raw_;}      // MODEL base frame
  const TeleopStatus & status() const {return status_;}
  const TeleopOptions & options() const {return opts_;}
  // The pad's joint box (see joint_range_frac), joint j.
  void jointBox(int j, double * lo, double * hi) const;
  void setOptions(const TeleopOptions & o);

  // Grasp point of the raw targets at rest (world), with R0 and R_e.
  void targetEe(Vec3 * p, Mat3 * r0, Mat3 * re) const;

  // 3-DOF position IK on q1..q3 (q4 held), damped Newton from `seed`.
  static bool ikPosition(
    const WholeBodyParams & p, const Vec3 & s, const VecN & seed, VecN * q);

private:
  // q_prev: the target being moved from -- the box refuses only a step that
  // takes a joint FURTHER outside it.
  bool jointsValid(
    const VecN & q, const VecN & q_prev, const Vec3 & com, std::string * why) const;
  double eeRestZ(const Vec3 & com, const VecN & q) const;

  std::shared_ptr<const VehicleModel> v_;
  TeleopOptions opts_;
  // raw targets
  Vec3 com_raw_{Vec3::Zero()};
  Vec3 box_centre_{Vec3::Zero()};
  double phi_raw_{0.0};
  VecN q_raw_{VecN::Zero()};
  Vec3 s_raw_{Vec3::Zero()};
  bool homing_{false};
  // smoothers
  ChainFilter com_f_, yaw_f_, q_f_;
  TeleopStatus status_;
};

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_TELEOP_REFERENCE_HPP_
