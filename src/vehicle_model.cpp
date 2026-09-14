// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/vehicle_model.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fsc_trajectory_planner
{

namespace
{

// T650 airframe + OM-X 4-DOF arm (the AM-T650 flown by the whole-body node).
// Numbers: WholeBodyParams::t650Defaults() (the make_params_t650 port the
// flight law runs on), the T650 rotor model, the validated OM-X working
// range and the [0, 40, 40, 0] deg folded home.
VehicleModel makeT650AerialManipulator(const VehicleOptions & o)
{
  VehicleModel v;
  v.name = "t650_aerial_manipulator";
  v.params = WholeBodyParams::t650Defaults();
  v.params.base_com = o.base_com;
  v.rotor = RotorModel::t650();
  for (int j = 0; j < kNumJoints; ++j) {
    v.q_min(j) = kArmQMin[j];
    v.q_max(j) = kArmQMax[j];
  }
  v.home << 0.0, 40.0 * M_PI / 180.0, 40.0 * M_PI / 180.0, 0.0;
  v.sigma_nd_margin = kSigmaNdMargin;
  v.beta_min_deg = 5.0;
  // AM_realign (y-forward model) <- AM_xfwd (x-forward flying asset): Rz(-90).
  v.r_model << 0.0, 1.0, 0.0,
    -1.0, 0.0, 0.0,
    0.0, 0.0, 1.0;
  return v;
}

}  // namespace

const std::map<std::string, VehicleFactory> & vehicleFactories()
{
  static const std::map<std::string, VehicleFactory> table = {
    {"t650_aerial_manipulator", makeT650AerialManipulator},
  };
  return table;
}

std::vector<std::string> vehicleNames()
{
  std::vector<std::string> out;
  for (const auto & kv : vehicleFactories()) {out.push_back(kv.first);}
  return out;
}

std::shared_ptr<const VehicleModel> makeVehicleModel(
  const std::string & name, const VehicleOptions & opts)
{
  const auto & table = vehicleFactories();
  const auto it = table.find(name);
  if (it == table.end()) {
    std::ostringstream m;
    m << "unknown vehicle '" << name << "' -- known:";
    for (const auto & kv : table) {m << " " << kv.first;}
    throw std::runtime_error(m.str());
  }
  return std::make_shared<const VehicleModel>(it->second(opts));
}

}  // namespace fsc_trajectory_planner
