// MIT License
// Copyright (c) 2026 FSC Lab
//
// Clamped uniform B-spline with least-squares fitting -- the curve family
// the end-effector trajectory planner fits its flat outputs with. Small and
// self-contained (Cox-de Boor + the derivative recurrence, sparse normal
// equations) so a periodic run of any length can be represented with a
// smooth, C^(p-1) curve whose derivatives to 4th order are analytic.

#ifndef FSC_TRAJECTORY_PLANNER_BSPLINE_FIT_HPP_
#define FSC_TRAJECTORY_PLANNER_BSPLINE_FIT_HPP_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <memory>
#include <vector>

namespace fsc_trajectory_planner
{

class BSplineFit
{
public:
  BSplineFit() = default;
  // degree p, n_spans uniform spans on [0, T], dim channels.
  BSplineFit(int degree, int n_spans, double T, int dim);

  int degree() const {return p_;}
  int numCtrl() const {return n_ctrl_;}
  int dim() const {return dim_;}
  double duration() const {return T_;}

  // Basis values and derivatives 0..order at t: rows = order, cols = the p+1
  // nonzero functions starting at index `first`.
  void basis(double t, int order, int * first, Eigen::MatrixXd * out) const;

  // value + derivatives 0..order of the curve at t: out is (order+1) x dim.
  void evalAll(double t, int order, Eigen::MatrixXd * out) const;

  // Prepare the least-squares operator for samples at times t (N): builds
  // the sparse collocation matrix and factors the normal equations ONCE, so
  // repeated fits on the same grid (every Picard iteration, every channel)
  // cost one back-substitution each.
  // n_pin_lo / n_pin_hi end control points are held COINCIDENT (equal to
  // the first / last sample): k coincident end points make derivatives 1..k-1
  // vanish there -- a clamped-rest start/end that a free fit cannot promise.
  void prepareFit(
    const Eigen::VectorXd & t, int n_pin_lo = 0, int n_pin_hi = 0, double ridge = 1e-10);
  // Fit the N x dim samples Y (must match prepareFit's grid). Returns the
  // RMS residual.
  double fit(const Eigen::MatrixXd & Y);

  Eigen::MatrixXd & coeffs() {return c_;}
  const Eigen::MatrixXd & coeffs() const {return c_;}

private:
  int p_{0}, spans_{0}, n_ctrl_{0}, dim_{0};
  double T_{1.0};
  std::vector<double> u_;   // knots
  Eigen::MatrixXd c_;       // n_ctrl x dim control points
  Eigen::SparseMatrix<double> A_;       // full collocation matrix
  Eigen::SparseMatrix<double> A_free_;  // its free (unpinned) columns
  int pin_lo_{0}, pin_hi_{0};
  std::shared_ptr<Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>>> ldlt_;
  int findSpan(double t) const;
};

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_BSPLINE_FIT_HPP_
