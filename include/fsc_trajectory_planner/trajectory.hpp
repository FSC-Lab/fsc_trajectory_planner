// MIT License
// Copyright (c) 2026 FSC Lab
//
// The planner-agnostic trajectory interface and the planner registry.
//
//   Trajectory          a finished plan: duration + the full whole-body
//                       reference (WbReference) at any elapsed time, clamped
//                       to [0, T]. HoldTrajectory is the T = 0 rest case.
//   TrajectoryPlanner   turns a request into a Trajectory, or throws
//                       std::runtime_error with an operator-readable reason.
//   registry            name -> planner. The node's `planner` parameter picks
//                       one; adding a shape (circle, figure-8, ...) is a new
//                       TrajectoryPlanner registered in plannerFactories().
//
// A PlanRequest carries the start rest, an optional goal rest (transitions)
// and a free-form numeric parameter map for shapes that need more than two
// endpoints (radius, period, laps, ...), so the node never has to change when
// a shape is added.

#ifndef FSC_TRAJECTORY_PLANNER_TRAJECTORY_HPP_
#define FSC_TRAJECTORY_PLANNER_TRAJECTORY_HPP_

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"
#include "fsc_trajectory_planner/wb_law.hpp"

namespace fsc_trajectory_planner
{

struct PlanDiag
{
  double T{0.0};
  double solve_ms{0.0};
  int iterations{0};
  double max_dyn_defect{0.0};
  double min_sigma_nd{0.0};
  double peak_com_speed{0.0};
  double peak_ee_speed{0.0};
  double endpoint_mismatch{0.0};
  VecN q_min_deg{VecN::Zero()};
  VecN q_max_deg{VecN::Zero()};
  // One line for the operator log ("T = .. s, defect .., sigma_nd ..").
  std::string summary;
};

class Trajectory
{
public:
  virtual ~Trajectory() = default;
  virtual double duration() const = 0;
  // The reference at elapsed time t, clamped to [0, T].
  virtual WbReference eval(double t) const = 0;
  // The rest the trajectory ends on (the next hold).
  virtual RestSpec goalRest() const = 0;
  virtual const PlanDiag & diag() const = 0;
};

// A static hold at one rest spec.
class HoldTrajectory : public Trajectory
{
public:
  HoldTrajectory(const WholeBodyParams & p, const RestSpec & rest);
  double duration() const override {return 0.0;}
  WbReference eval(double) const override {return ref_;}
  RestSpec goalRest() const override {return rest_;}
  const PlanDiag & diag() const override {return diag_;}

private:
  RestSpec rest_;
  WbReference ref_;
  PlanDiag diag_;
};

struct PlanOptions
{
  // kinematic bounds, honoured by every backend
  double v_max{0.30};    // peak EE / CoM translational speed [m/s]
  double a_max{0.15};    // peak translational acceleration [m/s^2]
  double w_max{0.30};    // peak angular rate, any angle channel [rad/s]
  double dw_max{0.60};   // peak angular acceleration [rad/s^2] (bspline)
  double T_min{3.0};     // never faster than this [s]
  double T_max{40.0};    // refuse rather than crawl [s]
  // input bounds (bspline): joint torque [N.m], <= 0 disables; rotor speed
  double tau_joint_max{3.0};
  bool rotor_bounds{true};
  int n_check{101};      // diagnostic grid, 0 disables
  // straight_line (Picard) knobs -- the classic planner's defaults
  int deg{16};
  int N{201};
  int maxit{60};
  double tol{1e-10};
  double relax{1.0};
  int Nfine{801};
  // Called periodically inside a solve so a caller can yield (unused by the
  // C++ node -- solves run on a worker thread -- kept for parity).
  std::function<void()> yield_hook;
};

struct PlanRequest
{
  RestSpec rest0;
  std::optional<RestSpec> rest1;
  std::map<std::string, double> shape;  // shape-specific parameters
};

class TrajectoryPlanner
{
public:
  virtual ~TrajectoryPlanner() = default;
  virtual std::string name() const = 0;
  // One line describing the backend for the startup log.
  virtual std::string description() const = 0;
  // Throws std::runtime_error with the reason when infeasible.
  virtual std::unique_ptr<Trajectory> plan(
    const VehicleModel & vehicle, const PlanRequest & req,
    const PlanOptions & opts) const = 0;
};

using PlannerFactory = std::function<std::unique_ptr<TrajectoryPlanner>()>;

// name -> factory. The ONE place a new backend or shape is added.
const std::map<std::string, PlannerFactory> & plannerFactories();
std::vector<std::string> plannerNames();
// Throws std::runtime_error naming the known planners on an unknown name.
std::unique_ptr<TrajectoryPlanner> makePlanner(const std::string & name);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_TRAJECTORY_HPP_
