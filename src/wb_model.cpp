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

#include "fsc_trajectory_planner/wb_model.hpp"

namespace fsc_trajectory_planner {

namespace {
constexpr int N = kNumJoints;
const Vec3 kE3{0.0, 0.0, 1.0};
}  // namespace

Mat3 hat(const Vec3& v) {
  Mat3 s;
  // clang-format off
  s <<    0.0, -v(2),  v(1),
        v(2),    0.0, -v(0),
       -v(1),  v(0),    0.0;
  // clang-format on
  return s;
}

Vec3 vee(const Mat3& S) { return Vec3{S(2, 1), S(0, 2), S(1, 0)}; }

Mat3 jointRotation(const Vec3& h, double q) {
  // Rodrigues, matching controller.py: a = h / (||h|| + 1e-15).
  const Vec3 a = h / (h.norm() + 1e-15);
  const Mat3 K = hat(a);
  return Mat3::Identity() + std::sin(q) * K + (1.0 - std::cos(q)) * (K * K);
}

WholeBodyParams WholeBodyParams::t650Defaults() {
  // ---- make_params() port: AM_realign extraction + T650 body override -----
  WholeBodyParams p;

  // T650 body 2.95 kg + the 4 AM rotors (4 x 0.039887, make_params' 0.15955).
  p.m_i = {2.95 + 0.15955, 0.1048326, 0.1423463, 0.13992727, 0.2495176};

  // Base inertia, MODEL frame. Derived exactly as generate_wb_truth.py does:
  //   I0_model(T650) = I0_model(AM_realign)
  //                    + R_MODEL^T (T650_tensor - diag(AM_xfwd authored)) R_MODEL
  // where R_MODEL = Rz(-90) swaps xx<->yy and negates Ixy. Constants below are
  // t650_params.INERTIA_TENSOR and AM_xfwd.usda's authored /body inertia; the
  // arithmetic is done here (not pre-rounded) so parity with the Python truth
  // generator holds to machine precision.
  const Mat3 I0_realign =
      Vec3{0.07368128, 0.07250296, 0.11707737}.asDiagonal();
  Mat3 dI_actual;  // T650 body tensor minus AM_xfwd authored diag (x-fwd frame)
  // clang-format off
  dI_actual << 0.077      - 0.06334175, 0.0089,                   0.0,
               0.0089,                  0.06408172 - 0.06301228,  0.0,
               0.0,                     0.0,          0.065004565 - 0.09868092;
  // clang-format on
  Mat3 r_model;  // Rz(-90), columns [[0,1,0],[-1,0,0],[0,0,1]] — boundary use
  // clang-format off
  r_model <<  0.0, 1.0, 0.0,
             -1.0, 0.0, 0.0,
              0.0, 0.0, 1.0;
  // clang-format on
  const Mat3 I0 = I0_realign + r_model.transpose() * dI_actual * r_model;

  // Link inertia diagonals (link frame = body frame at q = 0), as authored.
  const std::array<Vec3, N> link_inertias = {
      Vec3{3.720000e-05, 3.749000e-05, 2.161000e-05},
      Vec3{3.593700e-04, 3.485900e-04, 5.506000e-05},
      Vec3{3.130200e-04, 3.253000e-05, 3.184700e-04},
      Vec3{1.908700e-04, 2.637700e-04, 1.951000e-04}};

  // Joint axes (body frame at q = 0) — mechanism, mirrored in armFK below.
  p.h_i_im1 = {Vec3{0.0, 0.0, 1.0}, Vec3{1.0, 0.0, 0.0}, Vec3{1.0, 0.0, 0.0},
               Vec3{0.0, 0.0, 1.0}};

  // RAW chain extraction (identical numbers to make_params' raw_l/raw_com).
  const std::array<Vec3, N> raw_l = {
      Vec3{+0.000500, +0.002000, -0.042000},   // base         -> manip_joint1
      Vec3{-0.019000, +0.000070, -0.040500},   // manip_joint1 -> manip_joint2
      Vec3{+0.000089, +0.024000, -0.128000},   // manip_joint2 -> manip_joint3
      Vec3{+0.019976, +0.128927, -0.021750}};  // manip_joint3 -> manip_joint4
  const std::array<Vec3, N> raw_com = {
      Vec3{+0.000960, +0.001520, -0.073027},
      Vec3{+0.000320, +0.010163, -0.137795},
      Vec3{+0.019480, +0.107164, -0.128866},
      Vec3{+0.018278, +0.123460, -0.061369}};

  // Step 1: accumulate to absolute positions (base at 0).
  std::array<Vec3, N + 1> O_raw;
  O_raw[0] = Vec3::Zero();
  for (int k = 0; k < N; ++k) O_raw[k + 1] = O_raw[k] + raw_l[k];
  // joint_pos[k] = O_raw[k+1]; link_com[k] = O_raw[k] + raw_com[k]
  std::array<Vec3, N> link_com;
  for (int k = 0; k < N; ++k) link_com[k] = O_raw[k] + raw_com[k];

  // Step 2: re-register so joint k pivots about O[k-1] = manip_joint_k (the
  // joint-3 gravity fix; see make_params' comment block).
  //
  // EE = the GRIPPER pad midpoint (2026-08-23), not the wrist: the task
  // variable y = [r_e; b_1e] is the gripper. make_params() itself still ends
  // at the wrist and the legacy demos add this offset by hand, so the shift
  // lives in the T650 whole-body variant on BOTH sides — mirrored exactly by
  // transition_planner.GRIPPER_OFF_WRIST, and locked by the parity fixture.
  // Purely along the joint-4 axis, so q4 stays a roll that cannot move r_e:
  // position remains a (q1,q2,q3) problem and q4 buys the heading, which is
  // what the z-x-z recovery assumes. Only l_i[3] changes; com_i[3] is measured
  // from O_chain[3] = manip_joint4, so no mass property moves.
  // 0.108 since 2026-08-31 (user decision): the URDF end_effector_link --
  // the grasp point between the claw fingertip pads -- replacing the 0.0494
  // pad-midpoint measurement. Mirrored in transition_planner.GRIPPER_OFF_WRIST
  // and the arm GS's l4; wb_truth_t650.json regenerated with it.
  const Vec3 kGripperOffWrist{0.0, 0.0, -0.108};
  const Vec3 ee_pos = O_raw[4] + kGripperOffWrist;
  const std::array<Vec3, N + 1> O_chain = {Vec3::Zero(), O_raw[2], O_raw[3],
                                           O_raw[4], ee_pos};
  for (int k = 1; k <= N; ++k) p.l_i[k - 1] = O_chain[k] - O_chain[k - 1];
  for (int i = 1; i <= N; ++i) p.com_i[i - 1] = link_com[i - 1] - O_chain[i - 1];

  // Reflected rotor (armature) inertia along each joint axis, rank-1.
  const double J_arm = 353.5 * 353.5 * 1.6e-7;
  p.I_i_i[0] = I0;
  for (int i = 0; i < N; ++i) {
    p.I_i_i[i + 1] = Mat3(link_inertias[i].asDiagonal()) +
                     J_arm * (p.h_i_im1[i] * p.h_i_im1[i].transpose());
  }
  p.g = 9.81;
  return p;
}

void armForwardKinematics(const VecN& q, const WholeBodyParams& p, Vec3* r_0e_0,
                          Mat3* R_e_0, Mat3xN* A_e_out) {
  // Sections A/B of computeDynamics, kinematics only.
  std::array<Mat3, N + 1> R_i_0;
  R_i_0[0] = Mat3::Identity();
  for (int i = 0; i < N; ++i) {
    R_i_0[i + 1] = R_i_0[i] * jointRotation(p.h_i_im1[i], q(i));
  }
  std::array<Vec3, N + 1> h_i_0;
  h_i_0[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) h_i_0[i] = R_i_0[i - 1] * p.h_i_im1[i - 1];
  std::array<Vec3, N + 1> O;
  O[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) O[i] = O[i - 1] + R_i_0[i] * p.l_i[i - 1];
  if (r_0e_0 != nullptr) *r_0e_0 = O[N];
  if (R_e_0 != nullptr) *R_e_0 = R_i_0[N];
  if (A_e_out != nullptr) {
    for (int k = 1; k <= N; ++k) {
      A_e_out->col(k - 1) = h_i_0[k].cross(O[N] - O[k - 1]);
    }
  }
}

WholeBodyDynamics computeDynamics(const WbState& x, const WholeBodyParams& p) {
  WholeBodyDynamics dyn;
  const auto& mi = p.m_i;
  double m_total = p.totalMass();
  const Mat3 R_0 = x.R0;
  const VecN& q = x.q;
  const Vec3& omega_0 = x.omega0;
  const VecN& qdot = x.qdot;
  const double g_const = p.g;

  // ---- A. rotation kinematics ----
  std::array<Mat3, N + 1> R_i_0;
  R_i_0[0] = Mat3::Identity();
  for (int i = 0; i < N; ++i) {
    R_i_0[i + 1] = R_i_0[i] * jointRotation(p.h_i_im1[i], q(i));
  }
  const Mat3 R_e_0 = R_i_0[N];
  std::array<Mat3, N + 1> I_i_0;
  for (int i = 0; i <= N; ++i) {
    I_i_0[i] = R_i_0[i] * p.I_i_i[i] * R_i_0[i].transpose();
  }
  std::array<Vec3, N + 1> h_i_0;
  h_i_0[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) h_i_0[i] = R_i_0[i - 1] * p.h_i_im1[i - 1];

  // ---- B. translation kinematics ----
  std::array<Vec3, N + 1> O;
  O[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) O[i] = O[i - 1] + R_i_0[i] * p.l_i[i - 1];
  const Vec3 r_0e_0 = O[N];
  std::array<Vec3, N + 1> r0i;
  // Base-link CoM. Zero (the asset's answer) unless a hardware config sets
  // wb_base_com_*; it propagates from here into r_0c_0, d[0], L[0], M_r and
  // the GMO consistently, which a correction bolted onto r_0c_0 would not.
  r0i[0] = p.base_com;
  for (int i = 1; i <= N; ++i) r0i[i] = O[i - 1] + R_i_0[i] * p.com_i[i - 1];
  Vec3 r_0c_0 = Vec3::Zero();
  for (int i = 0; i <= N; ++i) r_0c_0 += mi[i] * r0i[i];
  r_0c_0 /= m_total;

  std::array<Mat3xN, N + 1> A_i;
  for (auto& a : A_i) a.setZero();
  for (int i = 1; i <= N; ++i) {
    for (int k = 1; k <= i; ++k) {
      A_i[i].col(k - 1) = h_i_0[k].cross(r0i[i] - O[k - 1]);
    }
  }
  Mat3xN A_e = Mat3xN::Zero();
  for (int k = 1; k <= N; ++k) {
    A_e.col(k - 1) = h_i_0[k].cross(r_0e_0 - O[k - 1]);
  }
  Mat3xN A = Mat3xN::Zero();
  for (int i = 0; i <= N; ++i) A += mi[i] * A_i[i];
  A /= m_total;
  std::array<Mat3xN, N + 1> A_til;
  for (int i = 0; i <= N; ++i) A_til[i] = A_i[i] - A;
  const Mat3xN A_til_e = A_e - A;
  std::array<Vec3, N + 1> d;
  for (int i = 0; i <= N; ++i) d[i] = r0i[i] - r_0c_0;
  const Vec3 d_e = r_0e_0 - r_0c_0;

  // ---- C. rotation velocity-level ----
  std::array<Vec3, N + 1> h_dot;
  for (auto& h : h_dot) h.setZero();
  for (int i = 2; i <= N; ++i) {
    for (int j = 1; j < i; ++j) {
      h_dot[i] += h_i_0[j].cross(h_i_0[i]) * qdot(j - 1);
    }
  }
  std::array<Mat3xN, N + 1> Jq;
  for (auto& j : Jq) j.setZero();
  for (int i = 1; i <= N; ++i) {
    for (int k = 1; k <= i; ++k) Jq[i].col(k - 1) = h_i_0[k];
  }
  const Mat3xN Jq_e = Jq[N];
  std::array<Mat3xN, N + 1> Jqd;
  for (auto& j : Jqd) j.setZero();
  for (int i = 1; i <= N; ++i) {
    for (int k = 1; k <= i; ++k) Jqd[i].col(k - 1) = h_dot[k];
  }
  const Mat3xN Jqd_e = Jqd[N];
  std::array<Vec3, N + 1> w0i;
  w0i[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) w0i[i] = Jq[i] * qdot;
  const Vec3 w0e = w0i[N];
  std::array<Vec3, N + 1> wi;
  wi[0] = omega_0;
  for (int i = 1; i <= N; ++i) wi[i] = omega_0 + w0i[i];
  std::array<Vec3, N + 1> rdotO;
  for (auto& r : rdotO) r.setZero();
  for (int i = 1; i <= N; ++i) {
    for (int j = 1; j <= i; ++j) {
      rdotO[i] += w0i[j].cross(R_i_0[j] * p.l_i[j - 1]);
    }
  }

  // ---- D. simplified-notation derivatives ----
  std::array<Vec3, N + 1> rdot0i;
  rdot0i[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) rdot0i[i] = A_i[i] * qdot;
  std::array<Mat3xN, N + 1> A_dot_i;
  for (auto& a : A_dot_i) a.setZero();
  for (int i = 1; i <= N; ++i) {
    for (int k = 1; k <= i; ++k) {
      A_dot_i[i].col(k - 1) = h_dot[k].cross(r0i[i] - O[k - 1]) +
                              h_i_0[k].cross(rdot0i[i] - rdotO[k - 1]);
    }
  }
  const Vec3 rdot0e = A_e * qdot;
  Mat3xN A_dot_e = Mat3xN::Zero();
  for (int k = 1; k <= N; ++k) {
    A_dot_e.col(k - 1) = h_dot[k].cross(r_0e_0 - O[k - 1]) +
                         h_i_0[k].cross(rdot0e - rdotO[k - 1]);
  }
  Mat3xN A_dot = Mat3xN::Zero();
  for (int i = 0; i <= N; ++i) A_dot += mi[i] * A_dot_i[i];
  A_dot /= m_total;
  std::array<Mat3xN, N + 1> A_til_dot;
  for (int i = 0; i <= N; ++i) A_til_dot[i] = A_dot_i[i] - A_dot;
  const Mat3xN A_til_dot_e = A_dot_e - A_dot;
  std::array<Vec3, N + 1> d_dot;
  for (int i = 0; i <= N; ++i) d_dot[i] = A_til[i] * qdot;
  const Vec3 d_dot_e = A_til_e * qdot;
  const Mat3 W0 = hat(omega_0);
  std::array<Mat3, N + 1> Xi, I_dot_0;
  for (int i = 0; i <= N; ++i) {
    Xi[i] = hat(wi[i]) * I_i_0[i] + I_i_0[i] * W0;
    I_dot_0[i] = Xi[i] + Xi[i].transpose();
  }

  // ---- E. transformed preliminaries: M_rho, B, N1, L, W (+ derivatives) ----
  MatN M_rho = MatN::Zero();
  MatNx3 B = MatNx3::Zero();
  MatN M_rho_dot = MatN::Zero();
  MatNx3 B_dot = MatNx3::Zero();
  // B/B_dot sign forced by the decoupling identity M == T^T Mtilde T.
  for (int i = 0; i <= N; ++i) {
    M_rho += mi[i] * A_til[i].transpose() * A_til[i] +
             Jq[i].transpose() * I_i_0[i] * Jq[i];
    B += -mi[i] * A_i[i].transpose() * hat(d[i]) + Jq[i].transpose() * I_i_0[i];
    M_rho_dot += mi[i] * (A_til_dot[i].transpose() * A_til[i] +
                          A_til[i].transpose() * A_til_dot[i]) +
                 Jqd[i].transpose() * I_i_0[i] * Jq[i] +
                 Jq[i].transpose() * I_dot_0[i] * Jq[i] +
                 Jq[i].transpose() * I_i_0[i] * Jqd[i];
    B_dot += -mi[i] * (A_til_dot[i].transpose() * hat(d[i]) +
                       A_til[i].transpose() * hat(d_dot[i])) +
             Jqd[i].transpose() * I_i_0[i] + Jq[i].transpose() * I_dot_0[i];
  }
  const Eigen::PartialPivLU<MatN> M_rho_lu(M_rho);
  const MatNx3 N1 = M_rho_lu.solve(B);
  const MatNx3 N1_dot = M_rho_lu.solve(B_dot - M_rho_dot * N1);

  std::array<Mat3, N + 1> L;
  for (int i = 0; i <= N; ++i) L[i] = -hat(d[i]) - A_til[i] * N1;
  const Mat3 L_e = -hat(d_e) - A_til_e * N1;
  std::array<Mat3, N + 1> L_dot;
  for (int i = 0; i <= N; ++i) {
    L_dot[i] = -hat(d_dot[i]) - A_til_dot[i] * N1 - A_til[i] * N1_dot;
  }
  const Mat3 L_dot_e = -hat(d_dot_e) - A_til_dot_e * N1 - A_til_e * N1_dot;
  std::array<Mat3, N + 1> W;
  for (int i = 0; i <= N; ++i) W[i] = Mat3::Identity() - Jq[i] * N1;
  const Mat3 W_e = Mat3::Identity() - Jq_e * N1;
  std::array<Mat3, N + 1> W_dot;
  for (int i = 0; i <= N; ++i) W_dot[i] = -Jqd[i] * N1 - Jq[i] * N1_dot;
  const Mat3 W_dot_e = -Jqd_e * N1 - Jq_e * N1_dot;

  // ---- F. transformed whole-body: Mtilde blocks, gtilde, Ctilde blocks ----
  const Mat3 M_t = m_total * Mat3::Identity();
  Mat3 M_r = Mat3::Zero(), C_r = Mat3::Zero();
  Mat3xN C_rp = Mat3xN::Zero();
  MatN C_p = MatN::Zero();
  for (int i = 0; i <= N; ++i) {
    M_r += mi[i] * L[i].transpose() * L[i] + W[i].transpose() * I_i_0[i] * W[i];
    C_r += mi[i] * L[i].transpose() * (W0 * L[i] + L_dot[i]) +
           W[i].transpose() * I_i_0[i] * W_dot[i] +
           W[i].transpose() * Xi[i] * W[i];
    C_rp += mi[i] * L[i].transpose() * (W0 * A_til[i] + A_til_dot[i]) +
            W[i].transpose() * I_i_0[i] * Jqd[i] +
            W[i].transpose() * Xi[i] * Jq[i];
    C_p += mi[i] * A_til[i].transpose() * (W0 * A_til[i] + A_til_dot[i]) +
           Jq[i].transpose() * I_i_0[i] * Jqd[i] +
           Jq[i].transpose() * Xi[i] * Jq[i];
  }
  Vec10 g_tilde = Vec10::Zero();
  g_tilde.head<3>() = m_total * g_const * kE3;
  // Full transformed Coriolis Ctilde: translation row/col zero, arm<->rotation
  // coupling antisymmetric (-C_rp^T). The GMO needs Ctilde^T * xi.
  Mat10 C_tilde = Mat10::Zero();
  C_tilde.block<3, 3>(3, 3) = C_r;
  C_tilde.block<3, N>(3, 6) = C_rp;
  C_tilde.block<N, 3>(6, 3) = -C_rp.transpose();
  C_tilde.block<N, N>(6, 6) = C_p;

  // ---- G. end-effector task Jacobian J_y, J_y_dot, Lambda_y, partition ----
  const Mat3 R_0e = R_e_0.transpose();
  const Eigen::Matrix<double, 1, 3> e3R0e = R_0e.row(2);  // E3^T * R_0e
  Mat4x10 J_y = Mat4x10::Zero();
  J_y.block<3, 3>(0, 0) = Mat3::Identity();
  J_y.block<3, 3>(0, 3) = R_0 * L_e;
  J_y.block<3, N>(0, 6) = R_0 * A_til_e;
  J_y.block<1, 3>(3, 3) = e3R0e * W_e;
  J_y.block<1, N>(3, 6) = e3R0e * Jq_e;
  Mat4x10 J_y_dot = Mat4x10::Zero();
  J_y_dot.block<3, 3>(0, 3) = R_0 * (W0 * L_e + L_dot_e);
  J_y_dot.block<3, N>(0, 6) = R_0 * (W0 * A_til_e + A_til_dot_e);
  const Mat3 W0e = hat(w0e);
  J_y_dot.block<1, 3>(3, 3) = e3R0e * (W_dot_e - W0e * W_e);
  J_y_dot.block<1, N>(3, 6) = e3R0e * (Jqd_e - W0e * Jq_e);

  Mat10 M_tilde = Mat10::Zero();
  M_tilde.block<3, 3>(0, 0) = M_t;
  M_tilde.block<3, 3>(3, 3) = M_r;
  M_tilde.block<N, N>(6, 6) = M_rho;
  const Eigen::PartialPivLU<Mat10> M_tilde_lu(M_tilde);
  const Mat10x4 Minv_JyT = M_tilde_lu.solve(J_y.transpose());
  const Mat4 Lambda_y = Mat4(J_y * Minv_JyT).inverse();
  const Mat10x4 J_y_pinv = Minv_JyT * Lambda_y;
  const Mat4x10 J_y_pinv_T = J_y_pinv.transpose();
  const Mat4x3 J_1y = J_y_pinv_T.block<4, 3>(0, 0);
  const Mat4x3 J_2y = J_y_pinv_T.block<4, 3>(0, 3);
  const Mat4 J_3y = J_y_pinv_T.block<4, N>(0, 6);

  // ---- H. transformed-state map T ----
  Mat10 T = Mat10::Zero();
  T.block<3, 3>(0, 0) = R_0;
  T.block<3, 3>(0, 3) = -R_0 * hat(r_0c_0);
  T.block<3, N>(0, 6) = R_0 * A;
  T.block<3, 3>(3, 3) = Mat3::Identity();
  T.block<N, 3>(6, 3) = N1;
  T.block<N, N>(6, 6) = MatN::Identity();

  // ---- I. original (untransformed) dynamics M, C, g (the plant) ----
  Mat10 M = Mat10::Zero(), C = Mat10::Zero();
  Vec10 g = Vec10::Zero();
  for (int i = 0; i <= N; ++i) {
    Eigen::Matrix<double, 3, kNumGen> Jvi_p;
    Jvi_p.setZero();
    Jvi_p.block<3, 3>(0, 0) = Mat3::Identity();
    Jvi_p.block<3, 3>(0, 3) = -hat(r0i[i]);
    Jvi_p.block<3, N>(0, 6) = A_i[i];
    const Eigen::Matrix<double, 3, kNumGen> Jvi = R_0 * Jvi_p;
    Eigen::Matrix<double, 3, kNumGen> Jvi_p_dot;
    Jvi_p_dot.setZero();
    Jvi_p_dot.block<3, 3>(0, 0) = W0;
    Jvi_p_dot.block<3, 3>(0, 3) = -(W0 * hat(r0i[i]) + hat(rdot0i[i]));
    Jvi_p_dot.block<3, N>(0, 6) = W0 * A_i[i] + A_dot_i[i];
    Eigen::Matrix<double, 3, kNumGen> Jwi;
    Jwi.setZero();
    Jwi.block<3, 3>(0, 3) = Mat3::Identity();
    Jwi.block<3, N>(0, 6) = Jq[i];
    Eigen::Matrix<double, 3, kNumGen> Jwi_dot;
    Jwi_dot.setZero();
    Jwi_dot.block<3, N>(0, 6) = Jqd[i];
    M += mi[i] * (Jvi_p.transpose() * Jvi_p) +
         Jwi.transpose() * I_i_0[i] * Jwi;
    C += mi[i] * (Jvi_p.transpose() * Jvi_p_dot) +
         Jwi.transpose() * I_i_0[i] * Jwi_dot + Jwi.transpose() * Xi[i] * Jwi;
    g += Jvi.transpose() * (mi[i] * g_const * kE3);
  }

  dyn.M_r = M_r;
  dyn.C_r = C_r;
  dyn.C_rp = C_rp;
  dyn.C_p = C_p;
  dyn.M_tilde = M_tilde;
  dyn.C_tilde = C_tilde;
  dyn.g_tilde = g_tilde;
  dyn.T = T;
  dyn.M = M;
  dyn.C = C;
  dyn.g = g;
  dyn.A = A;
  dyn.N1 = N1;
  dyn.r_0c_0 = r_0c_0;
  dyn.r_0e_0 = r_0e_0;
  dyn.R_e_0 = R_e_0;
  dyn.J_y = J_y;
  dyn.J_y_dot = J_y_dot;
  dyn.Lambda_y = Lambda_y;
  dyn.J_1y = J_1y;
  dyn.J_2y = J_2y;
  dyn.J_3y = J_3y;
  dyn.J_q_omega_e = Jq_e;
  dyn.J_q_dot_omega_e = Jqd_e;
  dyn.omega_0e_0 = w0e;
  dyn.A_e = A_e;
  return dyn;
}

double massIdentityResidual(const WholeBodyDynamics& dyn) {
  return (dyn.M - dyn.T.transpose() * dyn.M_tilde * dyn.T).norm() /
         dyn.M.norm();
}

}  // namespace fsc_trajectory_planner
