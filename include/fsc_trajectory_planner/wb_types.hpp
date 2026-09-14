// MIT License

// Copyright (c) 2026 FSC Lab

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef FSC_TRAJECTORY_PLANNER_WB_TYPES_HPP_
#define FSC_TRAJECTORY_PLANNER_WB_TYPES_HPP_

// Fixed-size types for the whole-body aerial-manipulator law.
//
// EVERYTHING here lives in the MODEL frame — AM_realign's y-forward body
// frame, the frame make_params()/dynamics() in fsc_PegasusSimulator's
// utils_controller/controller.py describe (the port's source of truth). The
// flying asset (AM_xfwd) is x-forward; frame_adapter.hpp owns the boundary.
//
// The arm is the 4-DOF OM-X chain, so every size is compile-time fixed:
// n = 4 joints, 6 + n = 10 generalized coordinates.

#include "Eigen/Dense"

namespace fsc_trajectory_planner {

inline constexpr int kNumJoints = 4;            // n
inline constexpr int kNumGen = 6 + kNumJoints;  // 6 + n = 10

using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using VecN = Eigen::Matrix<double, kNumJoints, 1>;
using Vec10 = Eigen::Matrix<double, kNumGen, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat4 = Eigen::Matrix4d;
using MatN = Eigen::Matrix<double, kNumJoints, kNumJoints>;
using Mat10 = Eigen::Matrix<double, kNumGen, kNumGen>;
using Mat3xN = Eigen::Matrix<double, 3, kNumJoints>;
using MatNx3 = Eigen::Matrix<double, kNumJoints, 3>;
using Mat4x10 = Eigen::Matrix<double, 4, kNumGen>;
using Mat10x4 = Eigen::Matrix<double, kNumGen, 4>;
using Mat4x3 = Eigen::Matrix<double, 4, 3>;

// State vector, MATLAB layout X = [r0(3); vec(R0)(9, col-major); q(n);
// v0(3); omega0(3); qdot(n)] — v0/omega0 are BODY-frame twists.
struct WbState {
  Vec3 r0{Vec3::Zero()};
  Mat3 R0{Mat3::Identity()};
  VecN q{VecN::Zero()};
  Vec3 v0{Vec3::Zero()};      // body frame
  Vec3 omega0{Vec3::Zero()};  // body frame
  VecN qdot{VecN::Zero()};
};

// The task reference the law consumes: system-CoM position through SNAP, base
// heading through 2 derivatives, EE position/heading through 2 derivatives,
// plus the smoothed joint reference used by the posture objective.
// Cartesian quantities are world-frame, MODEL axes.
//
// q_d/qdot_d are NOT optional in practice. Deleting the posture term that
// reads them was tried on 2026-09-05 and DIRECT became unreliable (1 of 3
// matched-kf soaks survived; every mismatched run diverged). Command.md
// 7.14.6.
struct WbReference {
  Vec3 x_cd{Vec3::Zero()};
  Vec3 x_cd_dot{Vec3::Zero()};
  Vec3 x_cd_ddot{Vec3::Zero()};
  Vec3 x_cd_d3{Vec3::Zero()};
  Vec3 x_cd_d4{Vec3::Zero()};
  Vec3 b1_d{Vec3::UnitX()};
  Vec3 b1_d_dot{Vec3::Zero()};
  Vec3 b1_d_ddot{Vec3::Zero()};
  Vec3 r_ed{Vec3::Zero()};
  Vec3 r_ed_dot{Vec3::Zero()};
  Vec3 r_ed_ddot{Vec3::Zero()};
  Vec3 b1_de{Vec3::UnitX()};
  Vec3 b1_de_dot{Vec3::Zero()};
  Vec3 b1_de_ddot{Vec3::Zero()};
  VecN q_d{VecN::Zero()};
  VecN qdot_d{VecN::Zero()};
};

// The law's physical outputs (model frame): collective thrust u1 [N] along
// R0*e3, base moment tau_body [N*m], joint torques tau_joint [N*m].
struct WbCommand {
  double u1{0.0};
  Vec3 u2{Vec3::Zero()};         // transformed base moment
  VecN u3{VecN::Zero()};         // transformed arm generalized force
  Vec3 tau_body{Vec3::Zero()};   // physical base moment (model frame)
  VecN tau_joint{VecN::Zero()};  // physical joint torques (what servos apply)
  Vec3 tau_body_arm{Vec3::Zero()};  // N1^T u3 — arm-reaction share of tau_body
  Mat3 R0c{Mat3::Identity()};
  Vec3 e_R{Vec3::Zero()};
  Vec4 e_y{Vec4::Zero()};
  // arm-torque source breakdown (diagnostic; mirrors controller.py's return)
  VecN t_thrust{VecN::Zero()};
  VecN t_imp{VecN::Zero()};
  VecN t_dist{VecN::Zero()};
  VecN t_cpl{VecN::Zero()};
  // The joint-posture PID's ACTUAL contribution to u3 this tick, as applied
  // (zero while `hold` owns the arm, since u3 is discarded there). Reported
  // so a run can be checked rather than assumed: the manuscript's law has no
  // joint-space term, and with the gains absent from the yaml this must be
  // exactly 0. Printed by the client; see also the LAW CHECK banner.
  VecN t_posture{VecN::Zero()};
  // The task-space disturbance force u3 actually consumed this step. With the
  // GMO it is the LUMPED (J_y^#)^T d_e_hat; with the L1 observer and
  // attribution on it is Lambda_y S_e Lambda_e^-1 w_hat_e. Reported because
  // in FREE FLIGHT the true interaction wrench is exactly zero, so whatever
  // is here is the phantom contact force the impedance law is rendering
  // compliance against -- the one number that compares the two designs on
  // the note's own terms, and it has to come from the same slot on both or
  // the comparison is not one.
  Vec4 F_hat_y{Vec4::Zero()};
  // GMO estimates (transformed frame)
  Vec3 d_t_hat{Vec3::Zero()};
  Vec3 d_r_hat{Vec3::Zero()};
  VecN d_rho_hat{VecN::Zero()};
  Vec10 d_e_hat{Vec10::Zero()};
  bool gmo_active{false};
  int n_sat{0};  // joints clamped at tau_max this step
};

// The transformed wrench u = [u1*R0*e3; u2; u3] that the plant ACHIEVED last
// tick, for the GMO's momentum integration (anti-windup: precedent is the L1
// predictor fed the allocator's achieved wrench). When absent the observer
// integrates the commanded u — the Python law's exact behaviour, and what the
// parity fixtures encode.
struct WbAchievedWrench {
  double u1{0.0};
  Vec3 u2{Vec3::Zero()};
  VecN u3{VecN::Zero()};
  bool valid{false};
};

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_WB_TYPES_HPP_
