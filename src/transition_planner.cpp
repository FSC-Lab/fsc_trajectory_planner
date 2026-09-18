// MIT License
// Copyright (c) 2026 FSC Lab
//
// The flat B-spline SETPOINT-TO-SETPOINT transition backend: wb_law's
// planFlatTransition() behind the TrajectoryPlanner interface, so the node
// plans every rest-to-rest move through the registry.
//
//   flat outputs   z(t) = [x_c ; psi ; q], one clamped uniform B-spline per
//                  channel on shared spans; the control points are the
//                  decision variables.
//   cost           min-snap on x_c, min-acceleration on psi, min-jerk on q.
//   constraints    rest at both ends (structural: coincident control points),
//                  joint box, speed/acceleration, rotor force and joint
//                  torque, T in [T_min, T_max].
//
// THE STRAIGHT-LINE PICARD BACKEND WAS REMOVED 2026-09-17 (user request). It
// was the C++ port of fsc_PegasusSimulator's utils_planner/transition_planner
// .py: a straight EE line with the CoM on a degree-16 polynomial solved by a
// Picard fixed point, and the joints recovered per sample. Every shipped
// config had already moved to "bspline", which enforces what that backend
// only verified and additionally bounds rotor force and joint torque. To
// recover it, see this file before that commit.
#include "fsc_trajectory_planner/transition_planner.hpp"

#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace fsc_trajectory_planner
{

namespace
{
class FlatTrajectory : public Trajectory
{
public:
  FlatTrajectory(std::shared_ptr<const WholeBodyParams> params, FlatPlan plan, RestSpec rest1)
  : params_(std::move(params)), plan_(std::move(plan)), rest1_(std::move(rest1))
  {
    const FlatPlanDiag & fd = plan_.diag();
    diag_.T = fd.T;
    diag_.solve_ms = fd.solve_ms;
    diag_.iterations = fd.n_dilations;
    diag_.max_dyn_defect = fd.max_dyn_defect;
    diag_.min_sigma_nd = fd.min_sigma_nd;
    diag_.peak_com_speed = fd.peak_v;
    diag_.peak_ee_speed = fd.peak_v;
    diag_.endpoint_mismatch = std::max(fd.endpoint_xc_err, fd.rest_err);
    diag_.q_min_deg = fd.q_min_deg;
    diag_.q_max_deg = fd.q_max_deg;
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "T = " << fd.T << " s (" << fd.T_binding << ", " << fd.n_dilations
      << " dilations), defect " << std::scientific << std::setprecision(1)
      << fd.max_dyn_defect << " m, sigma_nd " << std::fixed
      << std::setprecision(3) << fd.min_sigma_nd << ", peak |v| "
      << std::setprecision(2) << fd.peak_v << " m/s, tau_j "
      << fd.peak_tau_joint << " N.m, rotor " << std::setprecision(1)
      << fd.rotor_lo << "/" << fd.rotor_hi << ", solve " << fd.solve_ms << " ms";
    diag_.summary = m.str();
  }
  double duration() const override {return plan_.T();}
  WbReference eval(double t) const override {return plan_.eval(t);}
  RestSpec goalRest() const override {return rest1_;}
  const PlanDiag & diag() const override {return diag_;}

private:
  // FlatPlan keeps a pointer to the params it was planned with: own a copy.
  std::shared_ptr<const WholeBodyParams> params_;
  FlatPlan plan_;
  RestSpec rest1_;
  PlanDiag diag_;
};
}  // namespace

std::unique_ptr<Trajectory> FlatBSplineTransitionPlanner::plan(
  const VehicleModel & vehicle, const PlanRequest & req,
  const PlanOptions & o) const
{
  if (!req.rest1.has_value()) {
    throw std::runtime_error("bspline needs a goal rest (rest1)");
  }
  auto params = std::make_shared<const WholeBodyParams>(vehicle.params);
  FlatPlanOptions fo;
  fo.v_max = o.v_max;
  fo.a_max = o.a_max;
  fo.w_max = o.w_max;
  fo.dw_max = o.dw_max;
  fo.T_min = o.T_min;
  fo.T_max = o.T_max;
  fo.tau_joint_max = o.tau_joint_max;
  fo.rotor_bounds = o.rotor_bounds;
  fo.n_check = o.n_check;
  fo.sigma_nd_min = vehicle.sigma_nd_margin;
  FlatPlan plan = planFlatTransition(*params, req.rest0, *req.rest1, fo);
  return std::make_unique<FlatTrajectory>(params, std::move(plan), *req.rest1);
}

}  // namespace fsc_trajectory_planner
