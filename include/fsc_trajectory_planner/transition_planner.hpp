// MIT License
// Copyright (c) 2026 FSC Lab
//
// Straight-line compatible SETPOINT-TO-SETPOINT transition — C++ port of
// fsc_PegasusSimulator's utils_planner/transition_planner.py (the
// flight-validated backend, "straight_line").
//
//   prescribed task  every scalar channel A->B on ONE shared min-snap phase:
//                    EE position on a straight line p_e0 -> p_e1, z-x-z EE
//                    orientation angles (al, be, ga), platform heading pp,
//                    and the beta split q2/(q2+q3).
//   rest identity    at a rest point R0 = Rz(phi) exactly, so the task angles
//                    are (phi + q1, q2 + q3, q4) and both endpoints are
//                    algebraic in (phi, q): the plan starts and ends EXACTLY
//                    on the holds.
//   CoM solve        single-segment Picard fixed point on a degree-`deg`
//                    polynomial p_c (thrust-direction <-> arm-kinematics
//                    coupling), q recovered by the z-x-z decomposition with
//                    the time-varying split.

#ifndef FSC_TRAJECTORY_PLANNER_TRANSITION_PLANNER_HPP_
#define FSC_TRAJECTORY_PLANNER_TRANSITION_PLANNER_HPP_

#include <memory>
#include <string>

#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

class StraightLineTransitionPlanner : public TrajectoryPlanner
{
public:
  std::string name() const override {return "straight_line";}
  std::string description() const override
  {
    return "straight-line EE task, Picard CoM fixed point (flight-validated)";
  }
  std::unique_ptr<Trajectory> plan(
    const VehicleModel & vehicle, const PlanRequest & req,
    const PlanOptions & opts) const override;
};

// The flat B-spline backend (flat_planner.hpp) behind the same interface.
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
