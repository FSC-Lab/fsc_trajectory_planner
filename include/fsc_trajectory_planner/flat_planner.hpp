// Setpoint-to-setpoint whole-body transitions, parameterized as B-SPLINES IN
// THE FLAT OUTPUTS. C++/Eigen port of fsc_PegasusSimulator's
// utils_planner/flat_bspline_planner.py, held to it by flat_planner_selftest.
//
// WHY IT IS HERE AND NOT IN PYTHON
// The Python planner spends ~135 ms of a 192 ms plan inside dynamics(), which
// this package already has as computeDynamics(): 6.3 us against 1003 us for
// the identical computation, measured on one machine. The model is hundreds of
// operations on 3x3 and 10x10 matrices, so numpy's per-call overhead dwarfs the
// arithmetic. Porting the planner beside the model removes that gap and, more
// usefully, makes the (C14) discretization dense enough to mean something: a
// 401-point verification costs ~2.5 ms here against ~650 ms there.
//
// THE PROBLEM (see the Compatible Trajectory Planner note, Part C)
//   flat outputs   z(t) = [x_c ; psi ; q] in R^8 -- system CoM, platform
//                  heading, joint angles. 8 outputs against 8 inputs, and
//                  m xdd_c = f R0 e3 - m g e3 makes the thrust direction a
//                  function of xdd_c alone, so nothing downstream needs solving.
//   parameterized  one clamped uniform B-spline per channel on shared spans;
//                  the control points are the decision variables.
//   cost           min-snap on x_c, min-acceleration on psi, min-jerk on q.
//   constraints    rest at both ends (structural: coincident control points),
//                  joint box, speed/acceleration, rotor force and joint torque,
//                  T in [T_min, T_max].
//   solved as      an inner convex QP for the SHAPE -- which is independent of
//                  T -- plus a one-dimensional search for T. No NLP solver.
#ifndef FSC_TRAJECTORY_PLANNER_FLAT_PLANNER_HPP_
#define FSC_TRAJECTORY_PLANNER_FLAT_PLANNER_HPP_

#include <Eigen/Dense>
#include <array>
#include <functional>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/wb_model.hpp"
#include "fsc_trajectory_planner/wb_types.hpp"

namespace fsc_trajectory_planner
{

// Degree ceiling, so every basis evaluation runs on stack scratch.
inline constexpr int kMaxDegree = 7;

// ---------------------------------------------------------------------------
// Clamped uniform B-spline. Holds ONE channel (dim <= 4 columns).
// ---------------------------------------------------------------------------
class ClampedBSpline
{
public:
  void setup(int degree, int spans, double duration, int dim);

  // Value and derivatives 1..order at t, one row each: out is (order+1) x dim.
  // ONE basis pass for every order (A2.3 of the NURBS book), where the Python
  // evaluates a separately-built derivative spline per order -- five de Boor
  // passes for x_c against one here.
  void evalAll(double t, int order, Eigen::Ref<Eigen::MatrixXd> out) const;

  Eigen::MatrixXd & coeffs() {return c_;}
  const Eigen::MatrixXd & coeffs() const {return c_;}
  int numCtrl() const {return static_cast<int>(c_.rows());}
  int degree() const {return p_;}
  int spans() const {return spans_;}
  double duration() const {return T_;}
  // Rescale the knots to a new duration WITHOUT touching the control points:
  // the shape is T-independent, which is what lets T be searched separately.
  void setDuration(double duration);

private:
  int p_{0}, spans_{0};
  double T_{1.0};
  Eigen::VectorXd u_;
  Eigen::MatrixXd c_;
};

// ---------------------------------------------------------------------------
// Rotor speed limits as a per-rotor FORCE interval (f = k_f w^2 is a fixed
// map, so it converts once; the constraint then stays smooth and signed,
// where sqrt(F/k_f) is nondifferentiable at 0 and undefined for the negative
// allocation that IS the infeasibility).
// ---------------------------------------------------------------------------
struct RotorModel
{
  Eigen::Matrix<double, 4, 4> b_inv{Eigen::Matrix<double, 4, 4>::Zero()};
  double k_thrust{4.0412832584460006e-05};
  double f_min{0.0}, f_max{0.0};

  static RotorModel t650();
  Eigen::Vector4d forces(double thrust, const Vec3 & tau_body) const
  {
    Eigen::Vector4d rhs;
    rhs << thrust, tau_body;
    return b_inv * rhs;
  }
  double omegaOf(double force) const
  {
    return std::sqrt(std::max(force, 0.0) / k_thrust);
  }
};

struct FlatPlanOptions
{
  // basis
  int deg_xc{7};   // snap C^2 -> every field of WbReference is at least C^2
  int deg_psi{5};
  int deg_q{5};
  int n_spans{12};
  int order_xc{4};   // snap
  int order_psi{2};
  int order_q{3};    // jerk -> C^1 joint torque
  int pin_xc{5};     // value + 4 derivatives zero at each end => zero torque
  int pin_ang{3};
  // constraints (a non-positive bound disables that row)
  double v_max{0.30}, a_max{0.15}, w_max{0.30}, dw_max{0.60};
  double tau_joint_max{3.0};
  bool rotor_bounds{true};
  double sigma_nd_min{-1.0};  // <= 0 disables the endpoint conditioning test
  double T_min{3.0}, T_max{40.0};
  double T_forced{-1.0};  // > 0 pins T and only REPORTS the bounds
  int n_bound{21};        // grid the input bounds are SIZED on
  int n_check{101};       // diagnostic grid; 0 disables
  int max_dilations{8};
};

struct FlatPlanDiag
{
  double T{0.0}, solve_ms{0.0};
  std::string T_binding;
  int n_dilations{0};
  double rest_err{0.0}, endpoint_xc_err{0.0}, endpoint_q_err{0.0};
  double peak_v{0.0}, peak_a{0.0}, peak_w{0.0}, peak_dw{0.0};
  double peak_tau_joint{0.0}, rotor_hi{0.0}, rotor_lo{0.0};
  double min_sigma_nd{0.0}, max_dyn_defect{0.0}, thrust_residual{0.0};
  VecN q_min_deg{VecN::Zero()}, q_max_deg{VecN::Zero()};
  bool bounds_ok{true};
};

// Intermediates the reference does not carry but the inverse dynamics needs.
struct FlatAux
{
  Vec3 a_c{Vec3::Zero()};
  Mat3 R0{Mat3::Identity()}, R0_d{Mat3::Zero()}, R0_dd{Mat3::Zero()};
  Vec3 omega{Vec3::Zero()}, omega_dot{Vec3::Zero()};
  Vec3 r_0{Vec3::Zero()}, r_0_d{Vec3::Zero()}, r_0_dd{Vec3::Zero()};
};

struct FlatInputs
{
  double thrust{0.0};
  Vec3 tau_body{Vec3::Zero()};
  VecN tau_joint{VecN::Zero()};
  Eigen::Vector4d rotor_force{Eigen::Vector4d::Zero()};
  double thrust_residual{0.0};
};

class FlatPlan
{
public:
  double T() const {return T_;}
  const FlatPlanDiag & diag() const {return diag_;}
  VecN goalJoints() const {return q1_;}
  // The reference at elapsed time t, clamped to [0, T].
  WbReference eval(double t) const;

private:
  friend FlatPlan planFlatTransition(
    const WholeBodyParams &, const RestSpec &,
    const RestSpec &, const FlatPlanOptions &);
  double T_{0.0};
  const WholeBodyParams * params_{nullptr};
  ClampedBSpline xc_, psi_, q_;
  VecN q1_{VecN::Zero()};
  FlatPlanDiag diag_;
  mutable Eigen::MatrixXd scratch_xc_, scratch_psi_, scratch_q_;
};

// The flat map: (x_c and 4 derivatives, psi and 2, q and 2) -> the reference
// the law consumes, plus the intermediates the inverse dynamics needs.
void flatState(
  const WholeBodyParams & p, const Eigen::Matrix<double, 5, 3> & xc,
  const Vec3 & psi, const Eigen::Matrix<double, 3, kNumJoints> & q,
  WbReference * ref, FlatAux * aux);

// Thrust, body torque, joint torques and the per-rotor allocation along a
// reference. COLLECTIVE THRUST COMES FROM NEWTON, f = m||xdd_c + g e3||, not
// from M a + C v + g: routing it through the plant blocks agrees at the
// endpoints and drifts 1.4e-2 N / 0.76 deg mid-transition.
FlatInputs inverseInputs(
  const WholeBodyParams & p, const FlatAux & aux,
  const VecN & q, const VecN & qd, const VecN & qdd,
  const RotorModel * rotor);

// Throws std::runtime_error with an operator-readable reason when infeasible.
FlatPlan planFlatTransition(
  const WholeBodyParams & p, const RestSpec & rest0,
  const RestSpec & rest1,
  const FlatPlanOptions & opts = {});

// The COLLECTIVE half of the rotor bound, reduced to a limit on ||xdd_c||:
// convex in the control points and, on the T650, 88x slacker than a_max --
// so it is discharged by the kinematic bound and never costs a dynamics call.
void collectiveThrustLimits(
  const WholeBodyParams & p, const RotorModel & rotor,
  double * upper, double * lower);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_FLAT_PLANNER_HPP_
