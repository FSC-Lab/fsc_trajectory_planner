// MIT License
// Copyright (c) 2026 FSC Lab
//
// Rest-point algebra, arm kinematics and inverse kinematics for the
// whole-body aerial manipulator — C++ port of the model-side helpers of
// fsc_PegasusSimulator's utils_planner/transition_planner.py and
// compatible_trajectory.py (arm_fk_model, rest_ref, ik_position_azimuth,
// ik_world, _build_R0, _zxz_angles, _J3y, _sigma_nd, _minsnap3).
//
// Everything is in the MODEL frame (AM_realign, arm on body +y). The ROS node
// owns every actual<->model conversion at its boundary, exactly as the Python
// planner and the C++ frame_adapter do.

#ifndef FSC_TRAJECTORY_PLANNER_KINEMATICS_HPP_
#define FSC_TRAJECTORY_PLANNER_KINEMATICS_HPP_

#include <string>

#include "fsc_trajectory_planner/wb_law.hpp"

namespace fsc_trajectory_planner
{

// Joint limits: kArmQMin/kArmQMax (wb_law.hpp) -- the flight node's own.

// Non-dimensional singularity margin of the certified-safe set
// (transition_planner.SIGMA_ND_MARGIN).
inline constexpr double kSigmaNdMargin = 0.10;

// RestSpec (a static hold: base position, MODEL heading, joints) is the
// flight node's, via wb_law.hpp.

// ---- elementary rotations ------------------------------------------------
Mat3 Rz(double a);
Mat3 Rx(double a);
// Rest-to-rest min-snap (septic) phase: s, ds/du, d2s/du2 at u in [0, 1]
// (clamped). Zero velocity, acceleration AND jerk at both ends.
void minsnap3(double u, double * s, double * ds, double * d2s);
// Platform attitude from a thrust direction and a heading (controller R0c).
Mat3 buildR0(const Vec3 & t_dir, const Vec3 & b1_d);
// M = Rz(a) Rx(b) Rz(c) with b in [0, pi].
void zxzAngles(const Mat3 & M, double * a, double * b, double * c);
// a + 2*pi*k nearest to ref.
double unwrapNear(double a, double ref);

// ---- arm chain -----------------------------------------------------------
// (r_0c^0, r_0e^0, R_e^0): base->system-CoM and base->EE offsets in the base
// frame plus the EE orientation, on the controller's exact chain (exact link
// CoM offsets, params.base_com honoured).
void armKinematics(
  const VecN & q, const WholeBodyParams & p, Vec3 * r0c, Vec3 * r0e,
  Mat3 * Re);
// Arm-only EE task Jacobian J_3y^0 (3 position rows + 1 EE-yaw row).
Mat4 armTaskJacobian(const VecN & q, const WholeBodyParams & p);
// sigma_min of J_3y^0 with the translational rows scaled by 1/Lchar.
double sigmaNd(const VecN & q, const WholeBodyParams & p);

// Full whole-body reference for a STATIC HOLD: all derivatives zero,
// compatible by construction (thrust vertical at rest => R0 = Rz(phi)).
WbReference restReference(const WholeBodyParams & p, const RestSpec & rest);

// ---- inverse kinematics --------------------------------------------------
struct IkResult
{
  VecN q{VecN::Zero()};
  bool ok{false};
  double residual{0.0};
  double sigma_nd{0.0};
  bool limit_ok{false};
  std::string reason;
};

// Solve r_0e^0(q) = r_e_rel and azimuth(b1_e^0(q)) = azim_rel (both in the
// BASE frame, model axes). Damped Newton with an FD Jacobian; on a failed
// primary seed retries from the folded home and from zero. q is returned
// even when !ok (an out-of-range solution is what the GS joint-limit row
// displays).
IkResult ikPositionAzimuth(
  const WholeBodyParams & p, const Vec3 & r_e_rel, double azim_rel,
  const VecN & q_seed, double tol = 1e-10, int maxit = 80,
  double step_max = 0.5);

// World-frame convenience: base pose (x_b, phi) + inertial EE target
// (position, model heading azimuth) -> joint vector.
IkResult ikWorld(
  const WholeBodyParams & p, const Vec3 & x_b, double phi,
  const Vec3 & p_e_world, double azim_world, const VecN & q_seed);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_KINEMATICS_HPP_
