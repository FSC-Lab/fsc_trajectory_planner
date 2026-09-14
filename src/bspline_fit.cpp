// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/bspline_fit.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fsc_trajectory_planner
{

BSplineFit::BSplineFit(int degree, int n_spans, double T, int dim)
: p_(degree), spans_(n_spans), n_ctrl_(n_spans + degree), dim_(dim), T_(T)
{
  if (degree < 1 || n_spans < 1 || T <= 0.0 || dim < 1) {
    throw std::invalid_argument("BSplineFit: bad setup");
  }
  // clamped uniform knots: p+1 copies of 0 and T, spans-1 interior knots
  u_.assign(n_ctrl_ + p_ + 1, 0.0);
  for (int i = 0; i <= p_; ++i) {u_[i] = 0.0;}
  for (int i = 1; i < spans_; ++i) {u_[p_ + i] = T * i / spans_;}
  for (int i = n_ctrl_; i < n_ctrl_ + p_ + 1; ++i) {u_[i] = T;}
  c_ = Eigen::MatrixXd::Zero(n_ctrl_, dim_);
}

int BSplineFit::findSpan(double t) const
{
  if (t >= T_) {return n_ctrl_ - 1;}
  if (t <= 0.0) {return p_;}
  int s = p_ + static_cast<int>(std::floor(t / T_ * spans_));
  s = std::min(std::max(s, p_), n_ctrl_ - 1);
  while (t < u_[s]) {--s;}
  while (t >= u_[s + 1]) {++s;}
  return s;
}

void BSplineFit::basis(double t, int order, int * first, Eigen::MatrixXd * out) const
{
  t = std::min(std::max(t, 0.0), T_);
  const int i = findSpan(t);
  *first = i - p_;
  // N[d](j) for degree d = 0..p, function index i-d+j, j = 0..d
  std::vector<std::vector<double>> N(p_ + 1);
  N[0] = {1.0};
  for (int d = 1; d <= p_; ++d) {
    N[d].assign(d + 1, 0.0);
    for (int j = 0; j <= d; ++j) {
      const int idx = i - d + j;  // function index at degree d
      double left = 0.0, right = 0.0;
      if (j > 0) {
        const double den = u_[idx + d] - u_[idx];
        if (den > 0.0) {left = (t - u_[idx]) / den * N[d - 1][j - 1];}
      }
      if (j < d) {
        const double den = u_[idx + d + 1] - u_[idx + 1];
        if (den > 0.0) {right = (u_[idx + d + 1] - t) / den * N[d - 1][j];}
      }
      N[d][j] = left + right;
    }
  }
  // derivatives by the recurrence D^k N_{j,d} = d (D^{k-1}N_{j,d-1}/(u_{j+d}-u_j)
  //   - D^{k-1}N_{j+1,d-1}/(u_{j+d+1}-u_{j+1})), built degree by degree.
  // D[k][d] holds derivative k of the degree-d functions (d+1 entries).
  std::vector<std::vector<std::vector<double>>> D(order + 1);
  D[0] = N;
  for (int k = 1; k <= order; ++k) {
    D[k].assign(p_ + 1, {});
    for (int d = 0; d <= p_; ++d) {
      D[k][d].assign(d + 1, 0.0);
      if (d < k) {continue;}
      for (int j = 0; j <= d; ++j) {
        const int idx = i - d + j;
        double a = 0.0, b = 0.0;
        if (j > 0) {
          const double den = u_[idx + d] - u_[idx];
          if (den > 0.0) {a = D[k - 1][d - 1][j - 1] / den;}
        }
        if (j < d) {
          const double den = u_[idx + d + 1] - u_[idx + 1];
          if (den > 0.0) {b = D[k - 1][d - 1][j] / den;}
        }
        D[k][d][j] = d * (a - b);
      }
    }
  }
  out->resize(order + 1, p_ + 1);
  for (int k = 0; k <= order; ++k) {
    for (int j = 0; j <= p_; ++j) {(*out)(k, j) = D[k][p_][j];}
  }
}

void BSplineFit::evalAll(double t, int order, Eigen::MatrixXd * out) const
{
  int first = 0;
  Eigen::MatrixXd B;
  basis(t, order, &first, &B);
  *out = B * c_.middleRows(first, p_ + 1);
}

void BSplineFit::prepareFit(const Eigen::VectorXd & t, int n_pin_lo, int n_pin_hi, double ridge)
{
  const int n = static_cast<int>(t.size());
  pin_lo_ = std::max(0, n_pin_lo);
  pin_hi_ = std::max(0, n_pin_hi);
  if (pin_lo_ + pin_hi_ >= n_ctrl_) {
    throw std::invalid_argument("BSplineFit: too many pinned control points");
  }
  std::vector<Eigen::Triplet<double>> trip;
  trip.reserve(static_cast<size_t>(n) * (p_ + 1));
  Eigen::MatrixXd B;
  for (int k = 0; k < n; ++k) {
    int first = 0;
    basis(t(k), 0, &first, &B);
    for (int j = 0; j <= p_; ++j) {
      if (B(0, j) != 0.0) {trip.emplace_back(k, first + j, B(0, j));}
    }
  }
  A_ = Eigen::SparseMatrix<double>(n, n_ctrl_);
  A_.setFromTriplets(trip.begin(), trip.end());
  const int n_free = n_ctrl_ - pin_lo_ - pin_hi_;
  A_free_ = A_.middleCols(pin_lo_, n_free);
  Eigen::SparseMatrix<double> AtA = A_free_.transpose() * A_free_;
  Eigen::SparseMatrix<double> I(n_free, n_free);
  I.setIdentity();
  AtA += ridge * I;
  ldlt_ = std::make_shared<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>>();
  ldlt_->compute(AtA);
  if (ldlt_->info() != Eigen::Success) {
    throw std::runtime_error("BSplineFit: normal equations are singular");
  }
}

double BSplineFit::fit(const Eigen::MatrixXd & Y)
{
  if (!ldlt_ || Y.rows() != A_.rows() || Y.cols() != dim_) {
    throw std::invalid_argument("BSplineFit::fit: prepareFit grid / dim mismatch");
  }
  const int n_free = n_ctrl_ - pin_lo_ - pin_hi_;
  c_ = Eigen::MatrixXd::Zero(n_ctrl_, dim_);
  // pinned ends: coincident control points at the first / last sample value
  for (int i = 0; i < pin_lo_; ++i) {c_.row(i) = Y.row(0);}
  for (int i = 0; i < pin_hi_; ++i) {c_.row(n_ctrl_ - 1 - i) = Y.row(Y.rows() - 1);}
  Eigen::MatrixXd R = Y;
  if (pin_lo_ > 0) {R -= A_.leftCols(pin_lo_) * c_.topRows(pin_lo_);}
  if (pin_hi_ > 0) {R -= A_.rightCols(pin_hi_) * c_.bottomRows(pin_hi_);}
  const Eigen::MatrixXd rhs = A_free_.transpose() * R;
  c_.middleRows(pin_lo_, n_free) = ldlt_->solve(rhs);
  const Eigen::MatrixXd res = A_ * c_ - Y;
  return std::sqrt(res.squaredNorm() / static_cast<double>(res.size()));
}

}  // namespace fsc_trajectory_planner
