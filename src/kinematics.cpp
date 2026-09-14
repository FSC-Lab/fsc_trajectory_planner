// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>

namespace fsc_trajectory_planner
{

namespace
{
constexpr int N = kNumJoints;
const Vec3 kE3{0.0, 0.0, 1.0};

double wrapPi(double a) {return std::atan2(std::sin(a), std::cos(a));}

// Wrapped angle from b1e's horizontal projection to az_target. Returns false
// when the projection is degenerate (b1e near vertical).
bool azimuthResidual(const Vec3 & b1e, double az_target, double * out)
{
  const double h = std::hypot(b1e(0), b1e(1));
  if (h < 1e-6) {return false;}
  const double az = std::atan2(b1e(1), b1e(0));
  *out = wrapPi(az_target - az);
  return true;
}
}  // namespace

Mat3 Rz(double a)
{
  const double c = std::cos(a), s = std::sin(a);
  Mat3 R;
  R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
  return R;
}

Mat3 Rx(double a)
{
  const double c = std::cos(a), s = std::sin(a);
  Mat3 R;
  R << 1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c;
  return R;
}

void minsnap3(double u, double * s, double * ds, double * d2s)
{
  u = std::min(1.0, std::max(0.0, u));
  const double u2 = u * u, u3 = u2 * u, u4 = u3 * u, u5 = u4 * u, u6 = u5 * u,
    u7 = u6 * u;
  *s = 35 * u4 - 84 * u5 + 70 * u6 - 20 * u7;
  *ds = 140 * u3 - 420 * u4 + 420 * u5 - 140 * u6;
  *d2s = 420 * u2 - 1680 * u3 + 2100 * u4 - 840 * u5;
}

Mat3 buildR0(const Vec3 & t_dir, const Vec3 & b1_d)
{
  const Vec3 b3c = t_dir / t_dir.norm();
  const Mat3 P1 = Mat3::Identity() - b3c * b3c.transpose();
  Vec3 s = P1 * b1_d;
  double ns = s.norm();
  if (ns < 1e-9) {
    Vec3 tmp{1.0, 0.0, 0.0};
    if (std::abs(b3c(0)) > 0.9) {tmp = Vec3{0.0, 1.0, 0.0};}
    s = P1 * tmp;
    ns = s.norm();
  }
  const Vec3 b1c = s / ns;
  const Vec3 b2c = b3c.cross(b1c);
  Mat3 R;
  R.col(0) = b1c;
  R.col(1) = b2c;
  R.col(2) = b3c;
  return R;
}

void zxzAngles(const Mat3 & M, double * a, double * b, double * c)
{
  const double sb = std::hypot(M(0, 2), M(1, 2));
  *b = std::atan2(sb, M(2, 2));
  if (sb > 1e-9) {
    *a = std::atan2(M(0, 2), -M(1, 2));
    *c = std::atan2(M(2, 0), M(2, 1));
  } else {
    *a = std::atan2(-M(0, 1), M(0, 0));
    *c = 0.0;
  }
}

double unwrapNear(double a, double ref)
{
  return a + 2.0 * M_PI * std::round((ref - a) / (2.0 * M_PI));
}

void armKinematics(
  const VecN & q, const WholeBodyParams & p, Vec3 * r0c, Vec3 * r0e,
  Mat3 * Re)
{
  std::array<Mat3, N + 1> R;
  R[0] = Mat3::Identity();
  for (int i = 0; i < N; ++i) {
    R[i + 1] = R[i] * jointRotation(p.h_i_im1[i], q(i));
  }
  std::array<Vec3, N + 1> O;
  O[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) {
    O[i] = O[i - 1] + R[i] * p.l_i[i - 1];
  }
  if (r0e != nullptr) {*r0e = O[N];}
  if (Re != nullptr) {*Re = R[N];}
  if (r0c != nullptr) {
    Vec3 acc = p.m_i[0] * p.base_com;
    for (int i = 1; i <= N; ++i) {
      acc += p.m_i[i] * (O[i - 1] + R[i] * p.com_i[i - 1]);
    }
    *r0c = acc / p.totalMass();
  }
}

Mat4 armTaskJacobian(const VecN & q, const WholeBodyParams & p)
{
  std::array<Mat3, N + 1> R;
  R[0] = Mat3::Identity();
  for (int i = 0; i < N; ++i) {
    R[i + 1] = R[i] * jointRotation(p.h_i_im1[i], q(i));
  }
  std::array<Vec3, N> h0;
  for (int k = 1; k <= N; ++k) {h0[k - 1] = R[k - 1] * p.h_i_im1[k - 1];}
  std::array<Vec3, N + 1> O;
  O[0] = Vec3::Zero();
  for (int i = 1; i <= N; ++i) {O[i] = O[i - 1] + R[i] * p.l_i[i - 1];}
  const Vec3 r0e = O[N];
  const Vec3 b3e = R[N] * kE3;
  Mat4 J = Mat4::Zero();
  for (int k = 1; k <= N; ++k) {
    J.block<3, 1>(0, k - 1) = h0[k - 1].cross(r0e - O[k - 1]);
    J(3, k - 1) = b3e.dot(h0[k - 1]);
  }
  return J;
}

double sigmaNd(const VecN & q, const WholeBodyParams & p)
{
  double lchar = 0.0;
  for (const Vec3 & l : p.l_i) {lchar += l.norm();}
  lchar *= 0.5;
  Mat4 J = armTaskJacobian(q, p);
  J.topRows(3) /= lchar;
  return J.jacobiSvd().singularValues()(3);
}

WbReference restReference(const WholeBodyParams & p, const RestSpec & rest)
{
  const Mat3 r0 = Rz(rest.phi);
  Vec3 r0c, r0e;
  Mat3 re;
  armKinematics(rest.q, p, &r0c, &r0e, &re);
  WbReference r;  // every derivative defaults to zero
  r.x_cd = rest.x_b + r0 * r0c;
  r.b1_d = Vec3{std::cos(rest.phi), std::sin(rest.phi), 0.0};
  r.r_ed = rest.x_b + r0 * r0e;
  r.b1_de = r0 * re.col(0);
  r.q_d = rest.q;
  r.qdot_d = VecN::Zero();
  return r;
}

IkResult ikPositionAzimuth(
  const WholeBodyParams & p, const Vec3 & r_e_rel, double azim_rel,
  const VecN & q_seed, double tol, int maxit, double step_max)
{
  const VecN home = (VecN() << 0.0, 40.0 * M_PI / 180.0, 40.0 * M_PI / 180.0,
    0.0).finished();
  const std::array<VecN, 3> seeds = {q_seed, home, VecN::Zero()};

  // residual(q) -> false when the EE heading is vertical (azimuth undefined)
  auto residual = [&](const VecN & q, Eigen::Vector4d * f) {
      Vec3 r0e;
      Mat3 re;
      armKinematics(q, p, nullptr, &r0e, &re);
      double az = 0.0;
      if (!azimuthResidual(re.col(0), azim_rel, &az)) {return false;}
      f->head<3>() = r0e - r_e_rel;
      (*f)(3) = az;
      return true;
    };

  bool have_best = false;
  double best_res = 0.0;
  VecN best_q = q_seed;
  for (const VecN & seed : seeds) {
    VecN q = seed;
    bool degenerate = false;
    Eigen::Vector4d f;
    for (int it = 0; it < maxit; ++it) {
      if (!residual(q, &f)) {degenerate = true; break;}
      if (f.norm() < tol) {break;}
      Mat4 jac;
      const double eps = 1e-6;
      for (int j = 0; j < N; ++j) {
        VecN qp = q;
        qp(j) += eps;
        Eigen::Vector4d fp;
        if (!residual(qp, &fp)) {degenerate = true; break;}
        jac.col(j) = (fp - f) / eps;
      }
      if (degenerate) {break;}
      Eigen::Vector4d dq =
        jac.completeOrthogonalDecomposition().solve(-f);
      const double n = dq.norm();
      if (n > step_max) {dq *= step_max / n;}
      q += dq;
    }
    if (degenerate) {continue;}
    if (!residual(q, &f)) {continue;}
    const double res = f.norm();
    if (!have_best || res < best_res) {
      have_best = true;
      best_res = res;
      best_q = q;
    }
    if (res < 1e-8) {break;}
  }

  IkResult out;
  if (!have_best) {
    out.q = q_seed;
    out.ok = false;
    out.residual = std::numeric_limits<double>::infinity();
    out.sigma_nd = 0.0;
    out.limit_ok = false;
    out.reason = "degenerate EE heading";
    return out;
  }
  VecN q = best_q;
  q(0) = wrapPi(q(0));
  q(3) = wrapPi(q(3));
  bool limit_ok = true;
  for (int j = 0; j < N; ++j) {
    if (q(j) < kArmQMin[j] - 1e-9 || q(j) > kArmQMax[j] + 1e-9) {
      limit_ok = false;
    }
  }
  const double sig = sigmaNd(q, p);
  out.q = q;
  out.residual = best_res;
  out.sigma_nd = sig;
  out.limit_ok = limit_ok;
  out.ok = best_res < 1e-8 && limit_ok && sig >= kSigmaNdMargin;
  if (best_res >= 1e-8) {
    out.reason = "IK did not converge (target outside the reachable set?)";
  } else if (!limit_ok) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << "joint limits: q = [";
    for (int j = 0; j < N; ++j) {
      m << q(j) * 180.0 / M_PI << (j + 1 < N ? ", " : "");
    }
    m << "] deg vs [";
    m.precision(0);
    m << "[";
    for (int j = 0; j < N; ++j) {
      m << kArmQMin[j] * 180.0 / M_PI << (j + 1 < N ? ", " : "");
    }
    m << "], [";
    for (int j = 0; j < N; ++j) {
      m << kArmQMax[j] * 180.0 / M_PI << (j + 1 < N ? ", " : "");
    }
    m << "]]";
    out.reason = m.str();
  } else if (sig < kSigmaNdMargin) {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(3);
    m << "singularity margin: sigma_nd = " << sig << " < ";
    m.precision(2);
    m << kSigmaNdMargin;
    out.reason = m.str();
  }
  return out;
}

IkResult ikWorld(
  const WholeBodyParams & p, const Vec3 & x_b, double phi,
  const Vec3 & p_e_world, double azim_world, const VecN & q_seed)
{
  const Mat3 r0 = Rz(phi);
  const Vec3 r_e_rel = r0.transpose() * (p_e_world - x_b);
  return ikPositionAzimuth(p, r_e_rel, azim_world - phi, q_seed);
}

}  // namespace fsc_trajectory_planner
