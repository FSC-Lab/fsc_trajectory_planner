// MIT License
// Copyright (c) 2026 FSC Lab
//
// The whole-body MODEL and the flat B-spline planner come from the flight
// stack's exported library `fsc_autopilot_ros2::wb_law` -- the same
// wb_model.cpp / flat_planner.cpp the whole-body node flies and its parity
// tests lock. Nothing is copied: a change there is a change here on the next
// build. This header is the ONE place the planner names that dependency; it
// pulls the types it uses into this package's namespace so the rest of the
// code reads as its own.

#ifndef FSC_TRAJECTORY_PLANNER_WB_LAW_HPP_
#define FSC_TRAJECTORY_PLANNER_WB_LAW_HPP_

#include "single_aerial_manipulator_whole_body_direct_actuation/flat_planner.hpp"
#include "single_aerial_manipulator_whole_body_direct_actuation/wb_model.hpp"
#include "single_aerial_manipulator_whole_body_direct_actuation/wb_reference_builder.hpp"
#include "single_aerial_manipulator_whole_body_direct_actuation/wb_types.hpp"

namespace fsc_trajectory_planner
{

// fixed-size types (wb_types.hpp)
using nodelib::wb::kNumJoints;
using nodelib::wb::Vec3;
using nodelib::wb::Vec4;
using nodelib::wb::VecN;
using nodelib::wb::Mat3;
using nodelib::wb::Mat4;
using nodelib::wb::Mat3xN;
using nodelib::wb::WbState;
using nodelib::wb::WbReference;

// the model (wb_model.hpp)
using nodelib::wb::WholeBodyParams;
using nodelib::wb::WholeBodyDynamics;
using nodelib::wb::computeDynamics;
using nodelib::wb::armForwardKinematics;
using nodelib::wb::jointRotation;
using nodelib::wb::hat;
using nodelib::wb::vee;

// the flat B-spline planner (flat_planner.hpp)
using nodelib::wb::RestSpec;
using nodelib::wb::RotorModel;
using nodelib::wb::FlatPlan;
using nodelib::wb::FlatPlanDiag;
using nodelib::wb::FlatPlanOptions;
using nodelib::wb::planFlatTransition;
using nodelib::wb::FlatAux;
using nodelib::wb::FlatInputs;
using nodelib::wb::flatState;
using nodelib::wb::inverseInputs;

// the validated OM-X working range, owned by the flight node's reference
// builder (wb_reference_builder.hpp) -- one source for both processes.
using nodelib::wb::WbReferenceBuilder;
inline constexpr auto & kArmQMin = WbReferenceBuilder::kQMin;
inline constexpr auto & kArmQMax = WbReferenceBuilder::kQMax;

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_WB_LAW_HPP_
