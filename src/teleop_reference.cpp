// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/teleop_reference.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

namespace fsc_trajectory_planner
{

namespace
{
constexpr int N = kNumJoints;

double binom(int n, int k)
{
  double r = 1.0;
  for (int i = 1; i <= k; ++i) {r = r * (n - k + i) / i;}
  return r;
}

std::string deg(double rad, int prec = 1)
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(prec);
  m << rad * 180.0 / M_PI;
  return m.str();
}
}  // namespace

// ------------------------------------------------------------ ChainFilter
void ChainFilter::setup(int order, double bandwidth, int dim)
{
  n_ = std::max(1, order);
  w_ = std::max(1e-3, bandwidth);
  dim_ = dim;
  y_ = Eigen::MatrixXd::Zero(n_, dim_);
  u_ = Eigen::VectorXd::Zero(dim_);
}

void ChainFilter::reset(const Eigen::VectorXd & x)
{
  for (int i = 0; i < n_; ++i) {y_.row(i) = x.transpose();}
  u_ = x;
}

void ChainFilter::step(const Eigen::VectorXd & u, double dt)
{
  u_ = u;
  if (!(dt > 0.0)) {return;}
  // Phi = e^{-a} sum_k (a N)^k / k!, N the sub-diagonal shift: lower-
  // triangular Toeplitz with Phi(i, j) = e^{-a} a^(i-j) / (i-j)!. The input
  // enters through (I - Phi) 1 u, the constant-input steady state being every
  // stage equal to u. Both weights are >= 0 and sum to 1 per row.
  const double a = w_ * dt;
  const double ea = std::exp(-a);
  std::vector<double> c(static_cast<size_t>(n_));
  double term = 1.0;
  for (int k = 0; k < n_; ++k) {
    c[static_cast<size_t>(k)] = ea * term;
    term *= a / (k + 1);
  }
  Eigen::MatrixXd next(n_, dim_);
  for (int i = 0; i < n_; ++i) {
    Eigen::RowVectorXd acc = Eigen::RowVectorXd::Zero(dim_);
    double wsum = 0.0;
    for (int j = 0; j <= i; ++j) {
      const double w = c[static_cast<size_t>(i - j)];
      acc += w * y_.row(j);
      wsum += w;
    }
    next.row(i) = acc + (1.0 - wsum) * u_.transpose();
  }
  y_ = next;
}

Eigen::VectorXd ChainFilter::deriv(int k) const
{
  k = std::max(0, std::min(k, n_));
  Eigen::VectorXd out = Eigen::VectorXd::Zero(dim_);
  // y_n^(k) = w^k sum_j C(k, j) (-1)^j y_(n-k+j); stage index 0 is the input
  for (int j = 0; j <= k; ++j) {
    const int stage = n_ - k + j;                 // 1-based; 0 = input
    const Eigen::VectorXd ys = stage == 0 ? u_ : Eigen::VectorXd(y_.row(stage - 1).transpose());
    out += ((j % 2) ? -1.0 : 1.0) * binom(k, j) * ys;
  }
  return std::pow(w_, k) * out;
}

bool ChainFilter::settled(double tol) const
{
  for (int i = 0; i < n_; ++i) {
    if ((y_.row(i).transpose() - u_).cwiseAbs().maxCoeff() > tol) {return false;}
  }
  return true;
}

// ------------------------------------------------------- TeleopReference
TeleopReference::TeleopReference(
  std::shared_ptr<const VehicleModel> vehicle, const TeleopOptions & opts)
: v_(std::move(vehicle)), opts_(opts)
{
  com_f_.setup(5, opts_.com_bandwidth, 3);
  yaw_f_.setup(3, opts_.yaw_bandwidth, 1);
  q_f_.setup(3, opts_.arm_bandwidth, N);
}

void TeleopReference::setOptions(const TeleopOptions & o)
{
  // Rates and walls apply at once; the bandwidths only at the next seed(),
  // since the running chains keep theirs (re-sizing one mid-run would step
  // its derivatives).
  opts_ = o;
}

void TeleopReference::seed(const Vec3 & x_c, double phi, const VecN & q)
{
  com_f_.setup(5, opts_.com_bandwidth, 3);
  yaw_f_.setup(3, opts_.yaw_bandwidth, 1);
  q_f_.setup(3, opts_.arm_bandwidth, N);
  com_raw_ = x_c;
  box_centre_ = x_c;
  phi_raw_ = phi;
  q_raw_ = q;
  armKinematics(q_raw_, v_->params, nullptr, &s_raw_, nullptr);
  homing_ = false;
  com_f_.reset(com_raw_);
  Eigen::VectorXd y(1);
  y(0) = phi_raw_;
  yaw_f_.reset(y);
  q_f_.reset(q_raw_);
  status_ = TeleopStatus{};
  status_.sigma_nd = sigmaNd(q_raw_, v_->params);
}

bool TeleopReference::ikPosition(
  const WholeBodyParams & p, const Vec3 & s, const VecN & seed, VecN * q_out)
{
  VecN q = seed;
  constexpr double kLam2 = 1e-8;         // damping, m^2: invisible off-singular
  for (int it = 0; it < 30; ++it) {
    Vec3 r0e;
    armKinematics(q, p, nullptr, &r0e, nullptr);
    const Vec3 f = r0e - s;
    if (f.norm() < 1e-10) {break;}
    const Mat3 J = armTaskJacobian(q, p).block<3, 3>(0, 0);
    const Mat3 A = J * J.transpose() + kLam2 * Mat3::Identity();
    Vec3 dq = J.transpose() * A.ldlt().solve(-f);
    const double n = dq.norm();
    if (n > 0.2) {dq *= 0.2 / n;}
    q.head<3>() += dq;
  }
  Vec3 r0e;
  armKinematics(q, p, nullptr, &r0e, nullptr);
  *q_out = q;
  return (r0e - s).norm() < 1e-7 && q.allFinite();
}

double TeleopReference::eeRestZ(const Vec3 & com, const VecN & q) const
{
  // At rest R0 = Rz(phi), which leaves z alone: EE z = x_c z + (r_0e - r_0c) z.
  Vec3 r0c, r0e;
  armKinematics(q, v_->params, &r0c, &r0e, nullptr);
  return com(2) + r0e(2) - r0c(2);
}

void TeleopReference::jointBox(int j, double * lo, double * hi) const
{
  const double f = std::min(1.0, std::max(0.0, opts_.joint_range_frac));
  const double c = 0.5 * (kArmQMin[j] + kArmQMax[j]);
  const double h = 0.5 * (kArmQMax[j] - kArmQMin[j]) * f;
  *lo = c - h;
  *hi = c + h;
}

bool TeleopReference::jointsValid(
  const VecN & q, const VecN & q_prev, const Vec3 & com, std::string * why) const
{
  for (int j = 0; j < N; ++j) {
    double lo, hi;
    jointBox(j, &lo, &hi);
    const double out = std::max(0.0, std::max(q(j) - hi, lo - q(j)));
    const double out_prev = std::max(0.0, std::max(q_prev(j) - hi, lo - q_prev(j)));
    if (out > 1e-12 && out > out_prev - 1e-12) {
      *why = "joint " + std::to_string(j + 1) + " at the pad's inner bound [" + deg(lo, 0) +
        ", " + deg(hi, 0) + "] deg";
      return false;
    }
  }
  if (q(2) < opts_.q3_min) {
    *why = "q3 below " + deg(opts_.q3_min, 0) + " deg (elbow-branch guard)";
    return false;
  }
  if (q(1) + q(2) < v_->beta_min_deg * M_PI / 180.0) {
    *why = "fold q2+q3 below " + deg(v_->beta_min_deg * M_PI / 180.0, 0) + " deg (wrist singularity)";
    return false;
  }
  const double sig = sigmaNd(q, v_->params);
  if (sig < v_->sigma_nd_margin) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(3);
    o << "singularity margin sigma_nd " << sig;
    *why = o.str();
    return false;
  }
  if (eeRestZ(com, q) < opts_.ee_z_min) {
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(2);
    o << "end-effector floor (z >= " << opts_.ee_z_min << " m)";
    *why = o.str();
    return false;
  }
  return true;
}

void TeleopReference::step(
  double dt, const TeleopInput & in, const std::optional<Vec3> & x_c_meas)
{
  status_.com_blocked = 0;
  status_.arm_blocked = 0;
  status_.leash_active = false;
  std::string why;
  auto refuse = [&](const std::string & w) {why = w;};

  // ---- platform: CoM in the vehicle HEADING frame, and the heading --------
  // MODEL frame: the heading phi is model body-x; the vehicle's FORWARD (its
  // arm side) is model body +y and its LEFT is model body -x.
  const double cp = std::cos(phi_raw_), sp = std::sin(phi_raw_);
  const Vec3 fwd{-sp, cp, 0.0}, left{-cp, -sp, 0.0};
  const Vec3 v_com = in.com(0) * opts_.com_speed_xy * fwd +
    in.com(1) * opts_.com_speed_xy * left +
    Vec3{0.0, 0.0, in.com(2) * opts_.com_speed_z};
  for (int i = 0; i < 3; ++i) {
    const double d = v_com(i) * dt;
    if (d == 0.0) {continue;}
    Vec3 c = com_raw_;
    c(i) += d;
    bool ok = true;
    if (i < 2 && opts_.box_half_xy > 0.0 &&
      std::abs(c(i) - box_centre_(i)) > opts_.box_half_xy &&
      std::abs(c(i) - box_centre_(i)) > std::abs(com_raw_(i) - box_centre_(i)))
    {
      ok = false;
      refuse("geofence (+-" + std::to_string(opts_.box_half_xy).substr(0, 4) + " m about the engage point)");
    }
    if (ok && i == 2 && ((c(2) < opts_.com_z_min && d < 0.0) || (c(2) > opts_.com_z_max && d > 0.0))) {
      ok = false;
      refuse("altitude limit");
    }
    if (ok && i == 2 && d < 0.0 && eeRestZ(c, q_raw_) < opts_.ee_z_min) {
      ok = false;
      refuse("end-effector floor");
    }
    if (ok && opts_.leash > 0.0 && x_c_meas.has_value()) {
      const double before = (com_raw_ - *x_c_meas).norm();
      const double after = (c - *x_c_meas).norm();
      if (after > opts_.leash && after > before) {
        ok = false;
        status_.leash_active = true;
        refuse("leash: the vehicle is not keeping up");
      }
    }
    if (ok) {com_raw_ = c;} else {status_.com_blocked |= 1u << i;}
  }
  phi_raw_ += in.yaw * opts_.yaw_rate * dt;

  // ---- arm: grasp point relative to the airframe + the wrist roll ---------
  const bool arm_input = in.ee.cwiseAbs().maxCoeff() > 0.0 || in.roll != 0.0;
  if (in.home) {homing_ = true;}
  if (arm_input) {homing_ = false;}
  if (homing_) {
    // Fold home in joint space at home_rate per joint, the platform held.
    const VecN d = opts_.home - q_raw_;
    const double step_max = opts_.home_rate * dt;
    VecN q = q_raw_;
    bool done = true;
    for (int j = 0; j < N; ++j) {
      if (std::abs(d(j)) > step_max) {
        q(j) += std::copysign(step_max, d(j));
        done = false;
      } else {
        q(j) = opts_.home(j);
      }
    }
    std::string w;
    if (!jointsValid(q, q_raw_, com_raw_, &w) && eeRestZ(com_raw_, q) < opts_.ee_z_min) {
      // Only the floor can refuse a fold toward the (valid) home pose.
      refuse(w);
      homing_ = false;
    } else {
      q_raw_ = q;
      armKinematics(q_raw_, v_->params, nullptr, &s_raw_, nullptr);
      if (done) {homing_ = false;}
    }
  } else {
    // Body-frame stick axes in MODEL coordinates: forward = +y, left = -x.
    const Vec3 ds = Vec3{-in.ee(1), in.ee(0), in.ee(2)} * (opts_.ee_speed * dt);
    // bit per OPERATOR axis: 0 fwd (model y), 1 left (model x), 2 up (z)
    const int bit_of[3] = {1, 0, 2};
    for (int i = 0; i < 3; ++i) {
      if (ds(i) == 0.0) {continue;}
      Vec3 s = s_raw_;
      s(i) += ds(i);
      VecN q;
      std::string w;
      if (ikPosition(v_->params, s, q_raw_, &q) && jointsValid(q, q_raw_, com_raw_, &w)) {
        s_raw_ = s;
        q_raw_ = q;
      } else {
        status_.arm_blocked |= 1u << bit_of[i];
        refuse(w.empty() ? std::string("arm reach (IK does not converge)") : w);
      }
    }
    if (in.roll != 0.0) {
      VecN q = q_raw_;
      q(3) += in.roll * opts_.roll_rate * dt;
      std::string w;
      if (jointsValid(q, q_raw_, com_raw_, &w)) {
        q_raw_ = q;
      } else {
        status_.arm_blocked |= 1u << 3;
        refuse(w);
      }
    }
  }
  status_.homing = homing_;
  status_.sigma_nd = sigmaNd(q_raw_, v_->params);
  status_.wall = why;

  // ---- smoothers -------------------------------------------------------------
  com_f_.step(com_raw_, dt);
  Eigen::VectorXd y(1);
  y(0) = phi_raw_;
  yaw_f_.step(y, dt);
  q_f_.step(q_raw_, dt);
}

WbReference TeleopReference::reference(double time_scale) const
{
  const double s = time_scale;
  Eigen::Matrix<double, 5, 3> xc;
  double sk = 1.0;
  for (int k = 0; k <= 4; ++k) {
    xc.row(k) = sk * com_f_.deriv(k).transpose();
    sk *= s;
  }
  const Vec3 psi{yaw_f_.deriv(0)(0), s * yaw_f_.deriv(1)(0), s * s * yaw_f_.deriv(2)(0)};
  Eigen::Matrix<double, 3, N> q;
  q.row(0) = q_f_.deriv(0).transpose();
  q.row(1) = s * q_f_.deriv(1).transpose();
  q.row(2) = s * s * q_f_.deriv(2).transpose();
  WbReference ref;
  nodelib::wb::flatState(v_->params, xc, psi, q, &ref, nullptr);
  return ref;
}

bool TeleopReference::settled(double pos_tol, double ang_tol) const
{
  return com_f_.settled(pos_tol) && yaw_f_.settled(ang_tol) && q_f_.settled(ang_tol) && !homing_;
}

RestSpec TeleopReference::targetRest() const
{
  Vec3 r0c;
  armKinematics(q_raw_, v_->params, &r0c, nullptr, nullptr);
  RestSpec r;
  r.phi = phi_raw_;
  r.q = q_raw_;
  r.x_b = com_raw_ - Rz(phi_raw_) * r0c;
  return r;
}

void TeleopReference::targetEe(Vec3 * p, Mat3 * r0, Mat3 * re) const
{
  const RestSpec r = targetRest();
  Vec3 r0e;
  Mat3 R;
  armKinematics(r.q, v_->params, nullptr, &r0e, &R);
  const Mat3 R0 = Rz(r.phi);
  if (p) {*p = r.x_b + R0 * r0e;}
  if (r0) {*r0 = R0;}
  if (re) {*re = R;}
}

}  // namespace fsc_trajectory_planner
