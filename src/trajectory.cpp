// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/trajectory.hpp"

#include <sstream>
#include <stdexcept>

#include "fsc_trajectory_planner/transition_planner.hpp"

namespace fsc_trajectory_planner
{

HoldTrajectory::HoldTrajectory(const WholeBodyParams & p, const RestSpec & rest)
: rest_(rest), ref_(restReference(p, rest))
{
  diag_.summary = "static hold";
}

const std::map<std::string, PlannerFactory> & plannerFactories()
{
  static const std::map<std::string, PlannerFactory> table = {
    {"bspline",
      []() {return std::unique_ptr<TrajectoryPlanner>(new FlatBSplineTransitionPlanner());}},
    // Future shapes (circle, figure8, ...) register here: implement
    // TrajectoryPlanner::plan() reading req.shape["radius"], ["period"], ...
  };
  return table;
}

std::vector<std::string> plannerNames()
{
  std::vector<std::string> out;
  for (const auto & kv : plannerFactories()) {out.push_back(kv.first);}
  return out;
}

std::unique_ptr<TrajectoryPlanner> makePlanner(const std::string & name)
{
  const auto & table = plannerFactories();
  const auto it = table.find(name);
  if (it == table.end()) {
    std::ostringstream m;
    m << "unknown planner '" << name << "' -- known:";
    for (const auto & kv : table) {m << " " << kv.first;}
    throw std::runtime_error(m.str());
  }
  return it->second();
}

}  // namespace fsc_trajectory_planner
