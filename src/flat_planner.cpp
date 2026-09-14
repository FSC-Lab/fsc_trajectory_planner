#include "fsc_trajectory_planner/flat_planner.hpp"

#include "fsc_trajectory_planner/kinematics.hpp"

#include <algorithm>
#include <functional>
#include <limits>
#include <chrono>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace fsc_trajectory_planner
{
namespace
{

constexpr int kP1 = kMaxDegree + 1;   // 8: scratch width for degree <= 7

// --- knot helpers ---------------------------------------------------------
Eigen::VectorXd clampedKnots(int p, int spans, double T)
{
  Eigen::VectorXd u(spans + 2 * p + 1);
  for (int i = 0; i <= p; ++i) {
    u(i) = 0.0;
  }
  for (int j = 1; j < spans; ++j) {
    u(p + j) = T * j / spans;
  }
  for (int i = 0; i <= p; ++i) {
    u(spans + p + i) = T;
  }
  return u;
}

int findSpan(int n_ctrl, int p, double t, const Eigen::VectorXd & u)
{
  if (t >= u(n_ctrl)) {return n_ctrl - 1;}
  if (t <= u(p)) {return p;}
  int lo = p, hi = n_ctrl, mid = (lo + hi) / 2;
  while (t < u(mid) || t >= u(mid + 1)) {
    if (t < u(mid)) {hi = mid;} else {lo = mid;}
    mid = (lo + hi) / 2;
  }
  return mid;
}

// NURBS book A2.3: basis functions AND their derivatives up to `n`, one pass.
// ders[k][j] is the k-th derivative of N_{span-p+j, p}. This is the whole
// reason the C++ evaluation is cheap: the Python builds and walks a separate
// derivative spline per order.
void dersBasisFuns(
  int span, double t, int p, int n, const Eigen::VectorXd & u,
  double ders[][kP1])
{
  double ndu[kP1][kP1], a[2][kP1], left[kP1], right[kP1];
  ndu[0][0] = 1.0;
  for (int j = 1; j <= p; ++j) {
    left[j] = t - u(span + 1 - j);
    right[j] = u(span + j) - t;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      ndu[j][r] = right[r + 1] + left[j - r];
      const double tmp = ndu[r][j - 1] / ndu[j][r];
      ndu[r][j] = saved + right[r + 1] * tmp;
      saved = left[j - r] * tmp;
    }
    ndu[j][j] = saved;
  }
  for (int j = 0; j <= p; ++j) {
    ders[0][j] = ndu[j][p];
  }
  for (int r = 0; r <= p; ++r) {
    int s1 = 0, s2 = 1;
    a[0][0] = 1.0;
    for (int k = 1; k <= n; ++k) {
      double d = 0.0;
      const int rk = r - k, pk = p - k;
      if (r >= k) {
        a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
        d = a[s2][0] * ndu[rk][pk];
      }
      const int j1 = (rk >= -1) ? 1 : -rk;
      const int j2 = (r - 1 <= pk) ? k - 1 : p - r;
      for (int j = j1; j <= j2; ++j) {
        a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
        d += a[s2][j] * ndu[rk + j][pk];
      }
      if (r <= pk) {
        a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
        d += a[s2][k] * ndu[r][pk];
      }
      ders[k][r] = d;
      std::swap(s1, s2);
    }
  }
  double f = p;
  for (int k = 1; k <= n; ++k) {
    for (int j = 0; j <= p; ++j) {
      ders[k][j] *= f;
    }
    f *= (p - k);
  }
}

// --- derivative / Gram / cost matrices, all on the UNIT interval -----------
Eigen::MatrixXd derivMatrix(const Eigen::VectorXd & u, int p, int n)
{
  Eigen::MatrixXd D = Eigen::MatrixXd::Zero(n - 1, n);
  for (int i = 0; i < n - 1; ++i) {
    const double den = u(i + p + 1) - u(i + 1);
    const double f = (den <= 0.0) ? 0.0 : p / den;
    D(i, i) = -f;
    D(i, i + 1) = f;
  }
  return D;
}

void gaussLegendre(int n, std::vector<double> * x, std::vector<double> * w)
{
  x->assign(n, 0.0);
  w->assign(n, 0.0);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), pp = 0.0;
    for (int it = 0; it < 100; ++it) {
      double p0 = 1.0, p1 = 0.0;
      for (int j = 0; j < n; ++j) {
        const double p2 = p1;
        p1 = p0;
        p0 = ((2.0 * j + 1.0) * z * p1 - j * p2) / (j + 1);
      }
      pp = n * (z * p0 - p1) / (z * z - 1.0);
      const double dz = p0 / pp;
      z -= dz;
      if (std::abs(dz) < 1e-15) {break;}
    }
    (*x)[i] = -z;
    (*w)[i] = 2.0 / ((1.0 - z * z) * pp * pp);
  }
}

Eigen::MatrixXd gramMatrix(const Eigen::VectorXd & u, int p, int n)
{
  std::vector<double> brk;
  for (int i = 0; i < u.size(); ++i) {
    if (brk.empty() || u(i) > brk.back() + 1e-15) {brk.push_back(u(i));}}
  std::vector<double> xg, wg;
  gaussLegendre(p + 1, &xg, &wg);
  Eigen::MatrixXd G = Eigen::MatrixXd::Zero(n, n);
  Eigen::VectorXd nv(n);
  double ders[1][kP1];
  for (size_t s = 0; s + 1 < brk.size(); ++s) {
    const double a = brk[s], b = brk[s + 1];
    if (b <= a) {continue;}
    const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
    for (size_t k = 0; k < xg.size(); ++k) {
      const double t = mid + half * xg[k];
      const int span = findSpan(n, p, t, u);
      dersBasisFuns(span, t, p, 0, u, ders);
      nv.setZero();
      for (int j = 0; j <= p; ++j) {
        nv(span - p + j) = ders[0][j];
      }
      G.selfadjointView<Eigen::Lower>().rankUpdate(nv, wg[k] * half);
    }
  }
  return G.selfadjointView<Eigen::Lower>();
}

// c' Q c = integral ||d^order/dt^order spline||^2 on [0, 1]. Cached: the
// answer depends only on the basis, and by Q_r(T) = T^(1-2r) Q_r(1) the scale
// cancels in the minimizer, so T never enters it.
const Eigen::MatrixXd & costMatrix(int degree, int spans, int order)
{
  static std::map<std::tuple<int, int, int>, Eigen::MatrixXd> cache;
  const auto key = std::make_tuple(degree, spans, order);
  auto it = cache.find(key);
  if (it != cache.end()) {return it->second;}
  Eigen::VectorXd u = clampedKnots(degree, spans, 1.0);
  int p = degree, m = spans + degree;
  Eigen::MatrixXd D = Eigen::MatrixXd::Identity(m, m);
  for (int r = 0; r < order; ++r) {
    D = derivMatrix(u, p, m) * D;
    u = u.segment(1, u.size() - 2).eval();
    --p;
    --m;
  }
  Eigen::MatrixXd Q = D.transpose() * gramMatrix(u, p, m) * D;
  return cache.emplace(key, std::move(Q)).first->second;
}

// Min-||d^order||^2 control points for one channel, all axes at once, with the
// first/last n_pin rows pinned -- which is what makes the first n_pin-1
// derivatives vanish at each end. The box is checked, not imposed: the
// unconstrained optimum for a joint channel is the global min-jerk quintic,
// which is monotone, so both endpoints inside the box put the path inside it.
Eigen::MatrixXd solveChannel(
  int degree, int spans, int order,
  const Eigen::VectorXd & y0,
  const Eigen::VectorXd & y1, int n_pin)
{
  const int n = spans + degree, d = static_cast<int>(y0.size());
  if (2 * n_pin > n) {
    throw std::runtime_error("too few control points to pin both ends");
  }
  const Eigen::MatrixXd & Q = costMatrix(degree, spans, order);
  Eigen::MatrixXd c(n, d);
  for (int i = 0; i < n_pin; ++i) {
    c.row(i) = y0.transpose();
    c.row(n - 1 - i) = y1.transpose();
  }
  const int nf = n - 2 * n_pin;
  if (nf > 0) {
    const int f0 = n_pin;
    Eigen::MatrixXd A = Q.block(f0, f0, nf, nf);
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nf, d);
    B.noalias() -= Q.block(f0, 0, nf, n_pin) * c.topRows(n_pin);
    B.noalias() -= Q.block(f0, n - n_pin, nf, n_pin) * c.bottomRows(n_pin);
    c.block(f0, 0, nf, d) = A.ldlt().solve(B);
  }
  return c;
}

Mat3 hatOf(const Vec3 & v)
{
  Mat3 S;
  S << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
  return S;
}

// f = v/||v|| with two exact time derivatives.
void normalizeD2(
  const Vec3 & v, const Vec3 & vd, const Vec3 & vdd, Vec3 * f,
  Vec3 * fd, Vec3 * fdd)
{
  const double nrm = v.norm();
  if (nrm < 1e-12) {throw std::runtime_error("cannot normalize a zero vector");}
  *f = v / nrm;
  const double nd = f->dot(vd);
  *fd = (vd - *f * nd) / nrm;
  const double ndd = fd->dot(vd) + f->dot(vdd);
  *fdd = (vdd - 2.0 * (*fd) * nd - *f * ndd) / nrm;
}

double scanMax(
  const std::function<double(double)> & g, int n_scan = 65,
  int n_ref = 25)
{
  double best = -1e300, sbest = 0.0;
  std::vector<double> v(n_scan);
  for (int i = 0; i < n_scan; ++i) {
    const double s = static_cast<double>(i) / (n_scan - 1);
    v[i] = g(s);
    if (v[i] > best) {best = v[i]; sbest = s;}
  }
  const double h = 1.0 / (n_scan - 1);
  double a = std::max(0.0, sbest - h), b = std::min(1.0, sbest + h);
  const double gr = 0.5 * (std::sqrt(5.0) - 1.0);
  double c = b - gr * (b - a), d = a + gr * (b - a), fc = g(c), fd = g(d);
  for (int i = 0; i < n_ref; ++i) {
    if (fc > fd) {b = d; d = c; fd = fc; c = b - gr * (b - a); fc = g(c);} else {
      a = c; c = d; fc = fd; d = a + gr * (b - a); fd = g(d);
    }
  }
  return std::max({best, fc, fd});
}

}  // namespace

// ---------------------------------------------------------------------------
void ClampedBSpline::setup(int degree, int spans, double duration, int dim)
{
  if (degree < 1 || degree > kMaxDegree) {
    throw std::runtime_error("B-spline degree out of range");
  }
  p_ = degree;
  spans_ = spans;
  T_ = duration;
  u_ = clampedKnots(p_, spans_, T_);
  c_ = Eigen::MatrixXd::Zero(spans_ + p_, dim);
}

void ClampedBSpline::setDuration(double duration)
{
  T_ = duration;
  u_ = clampedKnots(p_, spans_, T_);
}

void ClampedBSpline::evalAll(
  double t, int order,
  Eigen::Ref<Eigen::MatrixXd> out) const
{
  const int n = numCtrl();
  const double tt = std::min(std::max(t, 0.0), T_);
  const int span = findSpan(n, p_, tt, u_);
  double ders[kP1][kP1];
  dersBasisFuns(span, tt, p_, std::min(order, p_), u_, ders);
  out.setZero();
  for (int k = 0; k <= order; ++k) {
    if (k > p_) {break;}
    for (int j = 0; j <= p_; ++j) {
      out.row(k).noalias() += ders[k][j] * c_.row(span - p_ + j);
    }
  }
}

RotorModel RotorModel::t650()
{
  // Geometry from the controller's own mixer (model frame); coefficients from
  // t650_params. The two assets label the four channels differently but
  // describe the same four positions, and only the per-rotor magnitude is
  // bounded, so the labelling does not enter.
  RotorModel r;
  const double pos[4][2] = {{0.229907, 0.229907},
    {-0.229907, -0.229907},
    {0.229907, -0.229907},
    {-0.229907, 0.229907}};
  const double dir[4] = {-1.0, -1.0, 1.0, 1.0};
  const double k_torque = 2.4741519e-06;
  Eigen::Matrix<double, 4, 4> B;
  for (int i = 0; i < 4; ++i) {
    B(0, i) = 1.0;
    B(1, i) = pos[i][1];
    B(2, i) = -pos[i][0];
    B(3, i) = dir[i] * k_torque / r.k_thrust;
  }
  r.b_inv = B.inverse();
  r.f_min = r.k_thrust * 64.0603 * 64.0603;
  r.f_max = r.k_thrust * 730.0507 * 730.0507;
  return r;
}

void collectiveThrustLimits(
  const WholeBodyParams & p, const RotorModel & rotor,
  double * upper, double * lower)
{
  const double m = p.totalMass();
  if (upper) {*upper = 4.0 * rotor.f_max / m - p.g;}
  if (lower) {*lower = p.g - 4.0 * rotor.f_min / m;}
}

void flatState(
  const WholeBodyParams & p, const Eigen::Matrix<double, 5, 3> & xc,
  const Vec3 & psi, const Eigen::Matrix<double, 3, kNumJoints> & q,
  WbReference * ref, FlatAux * aux)
{
  const double cp = std::cos(psi(0)), sp = std::sin(psi(0));
  const Vec3 b1(cp, sp, 0.0);
  const Vec3 b1d = psi(1) * Vec3(-sp, cp, 0.0);
  const Vec3 b1dd = psi(2) * Vec3(-sp, cp, 0.0) - psi(1) * psi(1) * Vec3(cp, sp, 0.0);

  const Vec3 acc = xc.row(2).transpose() + p.g * Vec3::UnitZ();
  Vec3 b3, b3d, b3dd;
  normalizeD2(acc, xc.row(3).transpose(), xc.row(4).transpose(), &b3, &b3d, &b3dd);
  const double s = b3.dot(b1);
  const double sd = b3d.dot(b1) + b3.dot(b1d);
  const double sdd = b3dd.dot(b1) + 2.0 * b3d.dot(b1d) + b3.dot(b1dd);
  const Vec3 w = b1 - s * b3;
  if (w.norm() < 1e-9) {
    throw std::runtime_error("heading is parallel to the thrust axis");
  }
  Vec3 b1c, b1cd, b1cdd;
  normalizeD2(
    w, b1d - sd * b3 - s * b3d,
    b1dd - sdd * b3 - 2.0 * sd * b3d - s * b3dd, &b1c, &b1cd, &b1cdd);
  const Vec3 b2c = b3.cross(b1c);
  const Vec3 b2cd = b3d.cross(b1c) + b3.cross(b1cd);
  const Vec3 b2cdd = b3dd.cross(b1c) + 2.0 * b3d.cross(b1cd) + b3.cross(b1cdd);
  Mat3 R0, R0d, R0dd;
  R0 << b1c, b2c, b3;
  R0d << b1cd, b2cd, b3d;
  R0dd << b1cdd, b2cdd, b3dd;

  // Arm chain with a forward velocity/acceleration recursion -- the kinematic
  // half of RNEA. O(n) and Hessian-free: propagating omega_i and omegadot_i is
  // what replaces forming d2 r_0e / dq2.
  const VecN qq = q.row(0).transpose(), qd = q.row(1).transpose(),
    qdd = q.row(2).transpose();
  std::array<Mat3, kNumJoints + 1> R;
  std::array<Vec3, kNumJoints + 1> wv, av, O, Od, Odd;
  R[0].setIdentity();
  wv[0].setZero(); av[0].setZero();
  O[0].setZero(); Od[0].setZero(); Odd[0].setZero();
  for (int i = 0; i < kNumJoints; ++i) {
    const Vec3 axis = R[i] * p.h_i_im1[i];
    R[i + 1] = R[i] * jointRotation(p.h_i_im1[i], qq(i));
    wv[i + 1] = wv[i] + qd(i) * axis;
    av[i + 1] = av[i] + qdd(i) * axis + qd(i) * wv[i].cross(axis);
  }
  for (int i = 1; i <= kNumJoints; ++i) {
    const Vec3 v = R[i] * p.l_i[i - 1];
    O[i] = O[i - 1] + v;
    Od[i] = Od[i - 1] + wv[i].cross(v);
    Odd[i] = Odd[i - 1] + av[i].cross(v) + wv[i].cross(wv[i].cross(v));
  }
  double m_total = 0.0;
  for (double v : p.m_i) {
    m_total += v;
  }
  Vec3 c = p.m_i[0] * p.base_com, cd = Vec3::Zero(), cdd = Vec3::Zero();
  for (int i = 1; i <= kNumJoints; ++i) {
    const Vec3 v = R[i] * p.com_i[i - 1];
    c += p.m_i[i] * (O[i - 1] + v);
    cd += p.m_i[i] * (Od[i - 1] + wv[i].cross(v));
    cdd += p.m_i[i] * (Odd[i - 1] + av[i].cross(v) + wv[i].cross(wv[i].cross(v)));
  }
  const Vec3 r0c = c / m_total, r0c_d = cd / m_total, r0c_dd = cdd / m_total;
  const Mat3 Sw = hatOf(wv[kNumJoints]), Sa = hatOf(av[kNumJoints]);
  const Mat3 Re = R[kNumJoints], Re_d = Sw * Re, Re_dd = (Sa + Sw * Sw) * Re;

  // r_e = x_c - R0 (r_0c - r_0e); product rule twice
  const Vec3 d0 = r0c - O[kNumJoints], d1 = r0c_d - Od[kNumJoints],
    d2 = r0c_dd - Odd[kNumJoints];
  const Vec3 u0 = R0 * d0, u1 = R0d * d0 + R0 * d1,
    u2 = R0dd * d0 + 2.0 * (R0d * d1) + R0 * d2;
  const Mat3 W = R0 * Re, Wd = R0d * Re + R0 * Re_d,
    Wdd = R0dd * Re + 2.0 * (R0d * Re_d) + R0 * Re_dd;

  if (ref) {
    ref->x_cd = xc.row(0); ref->x_cd_dot = xc.row(1); ref->x_cd_ddot = xc.row(2);
    ref->x_cd_d3 = xc.row(3); ref->x_cd_d4 = xc.row(4);
    ref->b1_d = b1; ref->b1_d_dot = b1d; ref->b1_d_ddot = b1dd;
    ref->r_ed = xc.row(0).transpose() - u0;
    ref->r_ed_dot = xc.row(1).transpose() - u1;
    ref->r_ed_ddot = xc.row(2).transpose() - u2;
    ref->b1_de = W.col(0); ref->b1_de_dot = Wd.col(0); ref->b1_de_ddot = Wdd.col(0);
    ref->q_d = qq; ref->qdot_d = qd;
  }
  if (aux) {
    aux->a_c = acc;
    aux->R0 = R0; aux->R0_d = R0d; aux->R0_dd = R0dd;
    const Mat3 what = R0.transpose() * R0d;
    aux->omega = vee(what);
    aux->omega_dot = vee(R0.transpose() * R0dd - what * what);
    const Vec3 cw = R0 * r0c, cwd = R0d * r0c + R0 * r0c_d,
      cwdd = R0dd * r0c + 2.0 * (R0d * r0c_d) + R0 * r0c_dd;
    aux->r_0 = xc.row(0).transpose() - cw;
    aux->r_0_d = xc.row(1).transpose() - cwd;
    aux->r_0_dd = xc.row(2).transpose() - cwdd;
  }
}

FlatInputs inverseInputs(
  const WholeBodyParams & p, const FlatAux & aux,
  const VecN & q, const VecN & qd, const VecN & qdd,
  const RotorModel * rotor)
{
  // The generalized velocity is (v_body, omega_body, qdot) -- BODY frame in
  // both of the first two blocks. So the first acceleration block is
  // d/dt(R0^T rdot_0) = R0^T rddot_0 - omega x v_body; the -omega x v term
  // cannot be in C, because computeDynamics never reads the linear-velocity
  // slot. Diagnosed from g[0:3] = m g R0^T e3, which rotates with the vehicle.
  WbState x;
  x.r0 = aux.r_0;
  x.R0 = aux.R0;
  x.q = q;
  x.v0 = aux.R0.transpose() * aux.r_0_d;
  x.omega0 = aux.omega;
  x.qdot = qd;
  const WholeBodyDynamics dyn = computeDynamics(x, p);
  Vec10 nu, nud;
  nu << x.v0, x.omega0, qd;
  nud << (aux.R0.transpose() * aux.r_0_dd - aux.omega.cross(x.v0)),
    aux.omega_dot, qdd;
  const Vec10 tau = dyn.M * nud + dyn.C * nu + dyn.g;

  FlatInputs out;
  out.thrust = p.totalMass() * aux.a_c.norm();   // Newton, exact
  out.tau_body = tau.segment<3>(3);
  out.tau_joint = tau.tail<kNumJoints>();
  out.thrust_residual = std::abs(out.thrust - tau.head<3>().dot(Vec3::UnitZ()));
  if (rotor) {out.rotor_force = rotor->forces(out.thrust, out.tau_body);}
  return out;
}

WbReference FlatPlan::eval(double t) const
{
  const double tt = std::min(std::max(t, 0.0), T_);
  if (scratch_xc_.size() == 0) {
    scratch_xc_.resize(5, 3);
    scratch_psi_.resize(3, 1);
    scratch_q_.resize(3, kNumJoints);
  }
  xc_.evalAll(tt, 4, scratch_xc_);
  psi_.evalAll(tt, 2, scratch_psi_);
  q_.evalAll(tt, 2, scratch_q_);
  Eigen::Matrix<double, 5, 3> XC = scratch_xc_;
  Eigen::Matrix<double, 3, kNumJoints> QQ = scratch_q_;
  WbReference ref;
  flatState(
    *params_, XC, Vec3(
      scratch_psi_(0, 0), scratch_psi_(1, 0),
      scratch_psi_(2, 0)),
    QQ, &ref, nullptr);
  return ref;
}

}  // namespace fsc_trajectory_planner

namespace fsc_trajectory_planner
{
namespace
{

struct ChannelPeaks { double d1{0.0}, d2{0.0}; };

// Peaks of ||Z'|| and ||Z''|| on the UNIT interval. The control points are
// T-independent, so these are the shape's own numbers and the duration follows
// in closed form: z^(k) scales exactly as T^-k.
ChannelPeaks unitPeaks(
  const Eigen::MatrixXd & c, int degree, int spans,
  bool use_norm)
{
  ClampedBSpline sp;
  sp.setup(degree, spans, 1.0, static_cast<int>(c.cols()));
  sp.coeffs() = c;
  Eigen::MatrixXd out(3, c.cols());
  auto at = [&](double s, int k) {
      sp.evalAll(s, 2, out);
      return use_norm ? out.row(k).norm() : out.row(k).cwiseAbs().maxCoeff();
    };
  ChannelPeaks pk;
  pk.d1 = scanMax([&](double s) {return at(s, 1);});
  pk.d2 = scanMax([&](double s) {return at(s, 2);});
  return pk;
}

struct InputPeaks
{
  double tau_joint{0.0};
  double f_min{1e300}, f_max{-1e300};
  double residual{0.0};
};

InputPeaks inputPeaks(
  const WholeBodyParams & p, const ClampedBSpline & xs,
  const ClampedBSpline & ps, const ClampedBSpline & qs,
  double T, int N, const RotorModel * rotor,
  bool quasi_static)
{
  InputPeaks pk;
  Eigen::MatrixXd bx(5, 3), bp(3, 1), bq(3, kNumJoints);
  for (int i = 0; i < N; ++i) {
    const double t = (N == 1) ? 0.0 : T * i / (N - 1);
    xs.evalAll(t, 4, bx);
    ps.evalAll(t, 2, bp);
    qs.evalAll(t, 2, bq);
    if (quasi_static) {
      bx.bottomRows(4).setZero();
      bp(1, 0) = bp(2, 0) = 0.0;
      bq.bottomRows(2).setZero();
    }
    Eigen::Matrix<double, 5, 3> XC = bx;
    Eigen::Matrix<double, 3, kNumJoints> QQ = bq;
    FlatAux aux;
    flatState(p, XC, Vec3(bp(0, 0), bp(1, 0), bp(2, 0)), QQ, nullptr, &aux);
    const FlatInputs u = inverseInputs(
      p, aux, QQ.row(0).transpose(),
      QQ.row(1).transpose(),
      QQ.row(2).transpose(), rotor);
    pk.tau_joint = std::max(pk.tau_joint, u.tau_joint.cwiseAbs().maxCoeff());
    pk.residual = std::max(pk.residual, u.thrust_residual);
    if (rotor) {
      pk.f_min = std::min(pk.f_min, u.rotor_force.minCoeff());
      pk.f_max = std::max(pk.f_max, u.rotor_force.maxCoeff());
    }
  }
  return pk;
}

// T multiplier that brings a bound-violating peak back inside. Every dynamic
// term is quadratic in 1/T (M a linear in accelerations, C v quadratic in
// velocities), so peak(T) ~ static + K/T^2 and one measurement sets the step.
// An ESTIMATE -- R0 itself moves with T -- so the caller iterates.
double dilate(double x, double x_static, double bound)
{
  const double num = x - x_static, den = bound - x_static;
  if (num <= 0.0) {return 1.0;}
  if (den <= 0.0) {return std::numeric_limits<double>::infinity();}
  return std::sqrt(num / den);
}

double sigmaNdDyn(const VecN & q, const WholeBodyParams & p)
{
  double lchar = 0.0;
  for (const Vec3 & l : p.l_i) {
    lchar += l.norm();
  }
  lchar *= 0.5;
  WbState x;
  x.q = q;
  const WholeBodyDynamics dyn = computeDynamics(x, p);
  Mat4 J = dyn.J_3y;
  J.topRows(3) /= lchar;
  return J.jacobiSvd().singularValues()(3);
}

// Base-origin -> system-CoM offset in the BASE frame. Straight from the model
// rather than inferred from a flatState call, which only happens to give it
// when R0 is the identity.
Vec3 armCoM(const VecN & q, const WholeBodyParams & p)
{
  WbState x;
  x.q = q;
  return computeDynamics(x, p).r_0c_0;
}

Mat3 rzFlat(double a)
{
  Mat3 R;
  R << std::cos(a), -std::sin(a), 0.0, std::sin(a), std::cos(a), 0.0, 0.0, 0.0, 1.0;
  return R;
}

}  // namespace

FlatPlan planFlatTransition(
  const WholeBodyParams & p, const RestSpec & rest0,
  const RestSpec & rest1, const FlatPlanOptions & o)
{
  using clock = std::chrono::steady_clock;
  const auto t_start = clock::now();

  // ---- (C12) joint angles: the ENDPOINTS are a hard refusal ---------------
  const RestSpec * ends[2] = {&rest0, &rest1};
  const char * names[2] = {"start", "goal"};
  for (int e = 0; e < 2; ++e) {
    for (int j = 0; j < kNumJoints; ++j) {
      const double v = ends[e]->q(j);
      if (v < kArmQMin[j] - 1e-9 ||
        v > kArmQMax[j] + 1e-9)
      {
        std::ostringstream m;
        m << names[e] << " joints outside the working range: q" << (j + 1)
          << " = " << v * 180.0 / M_PI << " not in ["
          << kArmQMin[j] * 180.0 / M_PI << ", "
          << kArmQMax[j] * 180.0 / M_PI << "] deg";
        throw std::runtime_error(m.str());
      }
    }
    if (o.sigma_nd_min > 0.0) {
      const double s = sigmaNdDyn(ends[e]->q, p);
      if (s < o.sigma_nd_min) {
        std::ostringstream m;
        m << names[e] << " pose sigma_nd = " << s << " < " << o.sigma_nd_min
          << " -- too close to a singularity";
        throw std::runtime_error(m.str());
      }
    }
  }

  // ---- the shape: control points, independent of T ------------------------
  double phi0 = rest0.phi;
  double phi1 = rest1.phi;
  phi1 += 2.0 * M_PI * std::round((phi0 - phi1) / (2.0 * M_PI));
  const Vec3 xc0 = rest0.x_b + rzFlat(phi0) * armCoM(rest0.q, p);
  const Vec3 xc1 = rest1.x_b + rzFlat(phi1) * armCoM(rest1.q, p);

  Eigen::VectorXd y0, y1;
  y0 = xc0; y1 = xc1;
  const Eigen::MatrixXd Cx =
    solveChannel(o.deg_xc, o.n_spans, o.order_xc, y0, y1, o.pin_xc);
  y0 = Eigen::VectorXd::Constant(1, phi0);
  y1 = Eigen::VectorXd::Constant(1, phi1);
  const Eigen::MatrixXd Cp =
    solveChannel(o.deg_psi, o.n_spans, o.order_psi, y0, y1, o.pin_ang);
  y0 = rest0.q; y1 = rest1.q;
  const Eigen::MatrixXd Cq =
    solveChannel(o.deg_q, o.n_spans, o.order_q, y0, y1, o.pin_ang);

  // ---- (C13) velocity / acceleration: CLOSED FORM, no search --------------
  const ChannelPeaks px = unitPeaks(Cx, o.deg_xc, o.n_spans, true);
  const ChannelPeaks pp = unitPeaks(Cp, o.deg_psi, o.n_spans, false);
  const ChannelPeaks pq = unitPeaks(Cq, o.deg_q, o.n_spans, false);
  const double p1a = std::max(pp.d1, pq.d1), p2a = std::max(pp.d2, pq.d2);
  struct Cand { const char * name; double T; };
  std::vector<Cand> cands{{"T_min", o.T_min}};
  if (o.v_max > 0.0) {cands.push_back({"v_max", px.d1 / o.v_max});}
  if (o.a_max > 0.0) {cands.push_back({"a_max", std::sqrt(px.d2 / o.a_max)});}
  if (o.w_max > 0.0) {cands.push_back({"w_max", p1a / o.w_max});}
  if (o.dw_max > 0.0) {cands.push_back({"dw_max", std::sqrt(p2a / o.dw_max)});}
  const bool forced = o.T_forced > 0.0;
  double T = forced ? o.T_forced : 0.0;
  std::string binding = forced ? "forced" : "";
  if (!forced) {
    for (const Cand & c : cands) {
      if (c.T > T) {T = c.T; binding = c.name;}}
  }

  FlatPlan plan;
  plan.params_ = &p;
  plan.xc_.setup(o.deg_xc, o.n_spans, T, 3);
  plan.xc_.coeffs() = Cx;
  plan.psi_.setup(o.deg_psi, o.n_spans, T, 1);
  plan.psi_.coeffs() = Cp;
  plan.q_.setup(o.deg_q, o.n_spans, T, kNumJoints);
  plan.q_.coeffs() = Cq;
  auto retime = [&](double TT) {
      plan.xc_.setDuration(TT);
      plan.psi_.setDuration(TT);
      plan.q_.setDuration(TT);
    };

  // ---- (C14) control inputs: dilate T, 1/T^2 law --------------------------
  const RotorModel rm = RotorModel::t650();
  const RotorModel * rotor = o.rotor_bounds ? &rm : nullptr;
  const bool check_inputs = (o.tau_joint_max > 0.0) || (rotor != nullptr);
  InputPeaks pk{}, stat{};
  bool have_stat = false, have_pk = false, feasible = !check_inputs || forced;
  int n_dilations = 0;
  const int n_rep = (o.n_check > 0) ? std::min(o.n_check, 61) : o.n_bound;
  int grid = o.n_bound;

  auto violations = [&](const InputPeaks & v) {
      std::vector<std::tuple<double, double, const char *>> bad;
      if (o.tau_joint_max > 0.0 && v.tau_joint > o.tau_joint_max + 1e-9) {
        bad.emplace_back(v.tau_joint, o.tau_joint_max, "tau_joint");
      }
      if (rotor) {
        if (v.f_max > rotor->f_max + 1e-9) {
          bad.emplace_back(v.f_max, rotor->f_max, "rotor upper");
        }
        if (-v.f_min > -rotor->f_min + 1e-9) {
          bad.emplace_back(-v.f_min, -rotor->f_min, "rotor lower");
        }
      }
      return bad;
    };

  if (check_inputs && !forced) {
    for (int it = 0; it <= o.max_dilations; ++it) {
      retime(T);
      pk = inputPeaks(p, plan.xc_, plan.psi_, plan.q_, T, grid, rotor, false);
      have_pk = true;
      auto bad = violations(pk);
      if (bad.empty()) {
        if (grid >= n_rep) {feasible = true; break;}
        grid = n_rep;
        pk = inputPeaks(p, plan.xc_, plan.psi_, plan.q_, T, grid, rotor, false);
        bad = violations(pk);
        if (bad.empty()) {feasible = true; break;}
      }
      if (T >= o.T_max - 1e-9) {break;}
      // The quasi-static sweep is the DIAGNOSIS, not the verdict: the peak
      // does not approach it monotonically (measured, it dips ~0.1% below and
      // comes back), so "static > bound" does not prove infeasibility. The
      // verdict is taken at T_max.
      if (!have_stat) {
        stat = inputPeaks(p, plan.xc_, plan.psi_, plan.q_, T, grid, rotor, true);
        have_stat = true;
      }
      double k = 1.0;
      for (const auto & [x, b, nm] : violations(pk)) {
        const double sv = (std::string(nm) == "tau_joint") ? stat.tau_joint :
          (std::string(nm) == "rotor upper") ? stat.f_max :
          -stat.f_min;
        k = std::max(k, dilate(x, sv, b));
      }
      if (!std::isfinite(k) || k <= 1.0 + 1e-9) {k = 1.5;}
      ++n_dilations;
      T = std::min(T * std::min(1.02 * k, 3.0), o.T_max);
    }
    if (!feasible) {
      std::ostringstream m;
      m << "control-input bounds unmet at T = " << T << " s (";
      for (const auto & [x, b, nm] : violations(pk)) {
        m << nm << " " << x << " vs " << b << "; ";
      }
      if (!have_stat) {
        stat = inputPeaks(p, plan.xc_, plan.psi_, plan.q_, T, grid, rotor, true);
        have_stat = true;
      }
      m << ") -- " << (violations(stat).empty() ?
      "too demanding for the actuators at any admissible duration" :
      "the quasi-static load alone exceeds the bound: a POSE problem");
      throw std::runtime_error(m.str());
    }
  }
  if (!forced) {T = std::min(std::max(T, o.T_min), o.T_max);}
  retime(T);
  plan.T_ = T;
  plan.q1_ = rest1.q;

  // ---- report -------------------------------------------------------------
  FlatPlanDiag & d = plan.diag_;
  d.T = T;
  d.T_binding = binding;
  d.n_dilations = n_dilations;
  d.peak_v = px.d1 / T;
  d.peak_a = px.d2 / (T * T);
  d.peak_w = p1a / T;
  d.peak_dw = p2a / (T * T);
  for (double te : {0.0, T}) {
    const WbReference r = plan.eval(te);
    double e = 0.0;
    for (const Vec3 * v : {&r.x_cd_dot, &r.x_cd_ddot, &r.x_cd_d3, &r.x_cd_d4,
        &r.b1_d_dot, &r.b1_d_ddot, &r.r_ed_dot, &r.r_ed_ddot,
        &r.b1_de_dot, &r.b1_de_ddot})
    {
      e = std::max(e, v->cwiseAbs().maxCoeff());
    }
    e = std::max(e, r.qdot_d.cwiseAbs().maxCoeff());
    d.rest_err = std::max(d.rest_err, e);
  }
  const WbReference r_end = plan.eval(T);
  d.endpoint_xc_err = (r_end.x_cd - xc1).norm();
  d.endpoint_q_err = (r_end.q_d - rest1.q).cwiseAbs().maxCoeff();

  if (o.n_check > 0) {
    if (!have_pk || grid != n_rep) {
      pk = inputPeaks(p, plan.xc_, plan.psi_, plan.q_, T, n_rep, rotor, false);
    }
    d.peak_tau_joint = pk.tau_joint;
    d.thrust_residual = pk.residual;
    if (rotor) {
      d.rotor_hi = rm.omegaOf(pk.f_max);
      d.rotor_lo = rm.omegaOf(pk.f_min);
    }
    d.min_sigma_nd = 1e300;
    d.q_min_deg.setConstant(1e300);
    d.q_max_deg.setConstant(-1e300);
    Eigen::MatrixXd bq(3, kNumJoints);
    for (int i = 0; i < o.n_check; ++i) {
      const double t = T * i / (o.n_check - 1);
      plan.q_.evalAll(t, 2, bq);
      const VecN qq = bq.row(0).transpose();
      d.q_min_deg = d.q_min_deg.cwiseMin(qq);
      d.q_max_deg = d.q_max_deg.cwiseMax(qq);
      d.min_sigma_nd = std::min(d.min_sigma_nd, sigmaNdDyn(qq, p));
    }
    d.q_min_deg *= 180.0 / M_PI;
    d.q_max_deg *= 180.0 / M_PI;
    // (A6) residual: x_c == r_e + R0 (r_0c - r_0e). An IDENTITY here, not a
    // solved fixed point, so it comes out at machine precision or the flat map
    // is wrong. R0 is taken from the same flatState the reference came from --
    // rebuilding it from a zeroed state was the bug this check first caught in
    // itself (it returned R0*r_0c where r_0c was wanted, and read 9.1e-03).
    {
      Eigen::MatrixXd bx(5, 3), bp(3, 1), bq(3, kNumJoints);
      for (int i = 0; i < 41; ++i) {
        const double t = T * i / 40.0;
        plan.xc_.evalAll(t, 4, bx);
        plan.psi_.evalAll(t, 2, bp);
        plan.q_.evalAll(t, 2, bq);
        Eigen::Matrix<double, 5, 3> XC = bx;
        Eigen::Matrix<double, 3, kNumJoints> QQ = bq;
        WbReference r;
        FlatAux a2;
        flatState(p, XC, Vec3(bp(0, 0), bp(1, 0), bp(2, 0)), QQ, &r, &a2);
        WbState xs;
        xs.q = r.q_d;
        const WholeBodyDynamics dq = computeDynamics(xs, p);
        d.max_dyn_defect = std::max(
          d.max_dyn_defect,
          (r.x_cd - (r.r_ed + a2.R0 * (dq.r_0c_0 - dq.r_0e_0))).norm());
      }
    }
    d.bounds_ok =
      (o.v_max <= 0 || d.peak_v <= o.v_max + 1e-9) &&
      (o.a_max <= 0 || d.peak_a <= o.a_max + 1e-9) &&
      (o.w_max <= 0 || d.peak_w <= o.w_max + 1e-9) &&
      (o.dw_max <= 0 || d.peak_dw <= o.dw_max + 1e-9) &&
      (o.tau_joint_max <= 0 || d.peak_tau_joint <= o.tau_joint_max + 1e-9) &&
      (!rotor || (d.rotor_hi <= std::sqrt(rm.f_max / rm.k_thrust) + 1e-6 &&
      d.rotor_lo >= std::sqrt(rm.f_min / rm.k_thrust) - 1e-6));
  }
  d.solve_ms = std::chrono::duration<double, std::milli>(clock::now() - t_start).count();
  return plan;
}

}  // namespace fsc_trajectory_planner
