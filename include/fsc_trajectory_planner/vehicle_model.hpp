// MIT License
// Copyright (c) 2026 FSC Lab
//
// Vehicle registry: everything a planner needs to know about ONE airframe +
// arm combination, looked up by name. Adding a vehicle = one factory function
// in vehicle_model.cpp registered in vehicleFactories(); nothing in the
// planners or the node names a vehicle.

#ifndef FSC_TRAJECTORY_PLANNER_VEHICLE_MODEL_HPP_
#define FSC_TRAJECTORY_PLANNER_VEHICLE_MODEL_HPP_

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/flat_planner.hpp"
#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/wb_model.hpp"

namespace fsc_trajectory_planner
{

struct VehicleModel
{
  std::string name;
  // Whole-body dynamic model (masses, inertias, chain geometry), MODEL frame.
  WholeBodyParams params;
  // Rotor speed limits as a force interval, for the input-bounded planners.
  RotorModel rotor;
  // Arm working range, singularity margin and the folded home pose.
  VecN q_min{VecN::Zero()};
  VecN q_max{VecN::Zero()};
  VecN home{VecN::Zero()};
  double sigma_nd_margin{kSigmaNdMargin};
  // Wrist-singularity guard on the fold angle beta = q2 + q3 [deg].
  double beta_min_deg{5.0};
  // Model <- actual body-frame rotation (columns). The flying asset is the
  // model geometry yawed; the node uses this at its ROS boundary only.
  Mat3 r_model{Mat3::Identity()};
};

struct VehicleOptions
{
  // Bare-airframe CoM relative to the body origin, MODEL frame [m]. Zero is
  // the asset's own answer (simulation); on hardware a flight measurement
  // that MUST equal the flight node's wb_base_com_*.
  Vec3 base_com{Vec3::Zero()};
};

using VehicleFactory = std::function<VehicleModel(const VehicleOptions &)>;

// name -> factory. The ONE place a new airframe is added.
const std::map<std::string, VehicleFactory> & vehicleFactories();
std::vector<std::string> vehicleNames();
// Throws std::runtime_error naming the known vehicles on an unknown name.
std::shared_ptr<const VehicleModel> makeVehicleModel(
  const std::string & name, const VehicleOptions & opts = {});

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_VEHICLE_MODEL_HPP_
