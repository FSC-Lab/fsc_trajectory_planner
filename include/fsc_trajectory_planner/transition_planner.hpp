// MIT License
// Copyright (c) 2026 FSC Lab
//
// Compatible SETPOINT-TO-SETPOINT transitions behind the TrajectoryPlanner
// interface. One backend since 2026-09-17: the flat B-spline planner
// (flat_planner.hpp, wb_law). The straight-line Picard backend was removed
// that day -- see transition_planner.cpp's header for what it was and why.

#ifndef FSC_TRAJECTORY_PLANNER_TRANSITION_PLANNER_HPP_
#define FSC_TRAJECTORY_PLANNER_TRANSITION_PLANNER_HPP_

#include <memory>
#include <string>

#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

// The flat B-spline backend (flat_planner.hpp) behind the planner interface.
class FlatBSplineTransitionPlanner : public TrajectoryPlanner
{
public:
  std::string name() const override {return "bspline";}
  std::string description() const override
  {
    return "flat B-spline outputs, constraints ENFORCED (rest, joint box, "
           "speed/accel, rotor + joint-torque bounds)";
  }
  std::unique_ptr<Trajectory> plan(
    const VehicleModel & vehicle, const PlanRequest & req,
    const PlanOptions & opts) const override;
};

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_TRANSITION_PLANNER_HPP_
