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

#ifndef FSC_TRAJECTORY_PLANNER_WB_MODEL_HPP_
#define FSC_TRAJECTORY_PLANNER_WB_MODEL_HPP_

// Whole-body coupled airframe+arm model — C++ port of
// fsc_PegasusSimulator .../robotic_arm/utils_controller/controller.py's
// make_params() + dynamics() (which is itself the faithful port of the
// validated MATLAB dynamics.m). Pure Eigen, no ROS.
//
// FRAME: everything is in the MODEL frame (AM_realign, arm on body +y). The
// flying AM_xfwd asset is that geometry rotated by Rz(-90) with the frame
// kept — frame_adapter.hpp owns the boundary; nothing here may know about it.
//
// PARITY: verified against controller.py over the wb_truth_t650.json fixture
// (tests/wb_parity_test.cpp) — any edit here must keep that test at <=1e-8.
// The section markers A..K below match dynamics()'s comments one-to-one so
// review stays mechanical.

#include <array>

#include "fsc_trajectory_planner/wb_types.hpp"

namespace fsc_trajectory_planner {

// ---------------------------------------------------------------------------
// Model parameters (make_params() port).
//
// The joint-axis TOPOLOGY [+z, +x, +x, +z] and the joint re-registration are
// mechanism, not data, and live in code. The NUMBERS (masses, inertias, link
// geometry) are data: defaults below are the AM_realign extraction with the
// T650 body override applied CORRECTLY in the model frame (the values
// generate_wb_truth.py prints — computed as
//   I0_model(T650) = I0_model(AM_realign) + R_MODEL^T dI_xfwd R_MODEL,
// NOT the un-rotated delta 05 uses, which is harmless there because only
// gravity is consumed but wrong for anything that touches I0). The client
// overwrites them from the YAML at startup so a flight's numbers always trace
// to one file.
// ---------------------------------------------------------------------------
struct WholeBodyParams {
  // m_i[0] = T650 body (2.95) + 4 AM rotors (0.15955) = 3.1095500 kg;
  // m_i[1..4] = arm link masses (link 4 = the lumped gripper assembly).
  std::array<double, kNumJoints + 1> m_i{3.1095500, 0.1048326, 0.1423463,
                                         0.13992727, 0.2495176};
  // I_i_i[0] = T650 body + rotor parallel-axis, MODEL frame (xx/yy swapped and
  // Ixy negated vs the x-forward t650_params tensor — see generate_wb_truth).
  // I_i_i[1..4] = link inertias + J_arm * h h^T (reflected rotor armature).
  std::array<Mat3, kNumJoints + 1> I_i_i{};
  std::array<Vec3, kNumJoints> l_i{};      // O_{i-1} -> O_i, link-i frame
  std::array<Vec3, kNumJoints> com_i{};    // O_{i-1} -> CoM_i, link-i frame
  std::array<Vec3, kNumJoints> h_i_im1{};  // joint axes at q = 0
  // Base-link CoM relative to the body origin, MODEL frame [m]. The body
  // origin is the geometric rotor centre (the asset's four rotors are
  // symmetric about it to 1e-11 m), which is also the point the wrench
  // allocator references, so this vector is exactly "how far the bare
  // airframe's CoM sits from the thrust axis".
  //
  // ZERO is the asset's own answer and the value every simulation config
  // uses: in Isaac the plant IS this model. It is NOT the real vehicle's
  // answer -- see wb_base_com_* in the hardware yaml for the flight
  // measurement. Only the HORIZONTAL components produce a moment; the
  // system CoM enters the body torque solely as u1 * (r_0c_0 x e3), which
  // annihilates z.
  Vec3 base_com{Vec3::Zero()};
  double g{9.81};

  // Fills every array with the AM_realign extraction + T650 body override —
  // the same numbers make_params_t650() in generate_wb_truth.py produces.
  static WholeBodyParams t650Defaults();

  double totalMass() const {
    double m = 0.0;
    for (double v : m_i) m += v;
    return m;
  }
};

// ---------------------------------------------------------------------------
// Everything dynamics() returns that the law (or a caller) consumes.
// Names match the Python dict keys.
// ---------------------------------------------------------------------------
struct WholeBodyDynamics {
  Mat3 M_r, C_r;
  Mat3xN C_rp;
  MatN C_p;
  Mat10 M_tilde, C_tilde;
  Vec10 g_tilde;
  Mat10 T;
  Mat10 M, C;
  Vec10 g;
  Mat3xN A;
  MatNx3 N1;
  Vec3 r_0c_0, r_0e_0;
  Mat3 R_e_0;
  Mat4x10 J_y, J_y_dot;
  Mat4 Lambda_y;
  Mat4x3 J_1y, J_2y;
  Mat4 J_3y;
  Mat3xN J_q_omega_e, J_q_dot_omega_e;
  Vec3 omega_0e_0;
  // Arm-only end-effector Jacobian in the BASE frame: rdot_0e^0 = A_e qdot.
  // Additive (2026-09-06) for the L1 augmentation's contact decomposition,
  // which needs the FULL end-effector Jacobian J_e and this is its arm block.
  // Mirrors controller.py's dynamics()["A_e"]. Nothing else reads it, and the
  // parity fixture compares an explicit field list, so it is a pure addition.
  Mat3xN A_e;
};

// hat / vee / Rodrigues — byte-matched to controller.py's helpers.
Mat3 hat(const Vec3& v);
Vec3 vee(const Mat3& S);
Mat3 jointRotation(const Vec3& h, double q);

// dynamics(X, params) — sections A..K of controller.py:288.
WholeBodyDynamics computeDynamics(const WbState& x, const WholeBodyParams& p);

// Arm FK in the model frame: EE position/orientation relative to the BASE
// (r_0e^0 and R_e^0), for the reference builder. A strict subset of
// computeDynamics — extracted so the 250 Hz reference path stays cheap.
void armForwardKinematics(const VecN& q, const WholeBodyParams& p, Vec3* r_0e_0,
                          Mat3* R_e_0, Mat3xN* A_e = nullptr);

// ||M - T^T Mtilde T|| / ||M|| — the transform identity that validates
// N1/T/M_tilde (expect ~1e-15; controller.py's own self-check).
double massIdentityResidual(const WholeBodyDynamics& dyn);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_WB_MODEL_HPP_
