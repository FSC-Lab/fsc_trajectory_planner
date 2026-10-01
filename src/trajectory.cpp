// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/trajectory.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "fsc_trajectory_planner/transition_planner.hpp"

namespace fsc_trajectory_planner
{

HoldTrajectory::HoldTrajectory(const WholeBodyParams & p, const RestSpec & rest)
: rest_(rest), ref_(restReference(p, rest))
{
  diag_.summary = "static hold";
}

SequenceTrajectory::SequenceTrajectory(std::vector<std::shared_ptr<const Trajectory>> segments)
: segments_(std::move(segments))
{
  if (segments_.empty()) {
    throw std::runtime_error("SequenceTrajectory needs at least one segment");
  }
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(2);
  for (size_t i = 0; i < segments_.size(); ++i) {
    const Trajectory & s = *segments_[i];
    if (i > 0) {
      // the previous segment's end must be this one's start
      const WbReference a = segments_[i - 1]->eval(segments_[i - 1]->duration());
      const WbReference b = s.eval(0.0);
      const double gap = std::max((a.x_cd - b.x_cd).norm(), (a.q_d - b.q_d).cwiseAbs().maxCoeff());
      if (gap > 1e-6) {
        std::ostringstream e;
        e << "SequenceTrajectory: segment " << i << " does not start where segment "
          << i - 1 << " ends (gap " << gap << ")";
        throw std::runtime_error(e.str());
      }
      m << " | ";
    }
    t0_.push_back(T_);
    T_ += s.duration();
    const PlanDiag & d = s.diag();
    diag_.solve_ms += d.solve_ms;
    diag_.iterations += d.iterations;
    diag_.max_dyn_defect = std::max(diag_.max_dyn_defect, d.max_dyn_defect);
    diag_.min_sigma_nd = i == 0 ? d.min_sigma_nd : std::min(diag_.min_sigma_nd, d.min_sigma_nd);
    diag_.peak_com_speed = std::max(diag_.peak_com_speed, d.peak_com_speed);
    diag_.peak_ee_speed = std::max(diag_.peak_ee_speed, d.peak_ee_speed);
    diag_.endpoint_mismatch = std::max(diag_.endpoint_mismatch, d.endpoint_mismatch);
    diag_.q_min_deg = i == 0 ? d.q_min_deg : VecN(diag_.q_min_deg.cwiseMin(d.q_min_deg));
    diag_.q_max_deg = i == 0 ? d.q_max_deg : VecN(diag_.q_max_deg.cwiseMax(d.q_max_deg));
    m << "[" << i + 1 << "] " << d.summary;
  }
  diag_.T = T_;
  diag_.summary = m.str();
}

WbReference SequenceTrajectory::eval(double t) const
{
  size_t i = segments_.size() - 1;
  while (i > 0 && t < t0_[i]) {--i;}
  return segments_[i]->eval(t - t0_[i]);
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
