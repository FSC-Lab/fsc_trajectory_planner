// MIT License
// Copyright (c) 2026 FSC Lab
//
// The USABLE end-effector workspace as an (r, z) occupancy grid about the
// body origin: r the horizontal radius, z the height, both of the grasp point
// in the (level) body frame. (r, z) depends on (q2, q3) alone -- q1 swings the
// point about the vertical, q4 rolls about the claw axis -- so the grid is
// swept at q1 = q4 = 0 and is a NECESSARY reachability bound, not a
// sufficient one. The node publishes it on whole_body_planner/workspace_rz;
// the arm GS's EE Whole-Body and PS4 Remote tabs draw it and the PS4 tab
// refuses a stick increment that would leave it.
//
// Which poses count: every (q2, q3) in the joint box whose fold beta = q2 + q3
// is at least `fold_min`. Below the vehicle's own beta_min_deg (the
// EE-trajectory mode's wrist-singularity guard) a pose must also keep
// sigma_nd >= the vehicle's margin -- that is how the pick-and-place mode's
// claw-down pose (beta = 0, sigma_nd 0.169) is admitted without admitting the
// whole folded-under region.

#ifndef FSC_TRAJECTORY_PLANNER_WORKSPACE_HPP_
#define FSC_TRAJECTORY_PLANNER_WORKSPACE_HPP_

#include <cstdint>
#include <vector>

#include "fsc_trajectory_planner/vehicle_model.hpp"

namespace fsc_trajectory_planner
{

struct WorkspaceGrid
{
  // grid extent [m] and size; cells row-major, rows = z from z_max DOWN
  double r_min{0.0}, r_max{0.0}, z_min{0.0}, z_max{0.0};
  int nr{0}, nz{0};
  std::vector<uint8_t> cells;
  // extent of the samples themselves (for the log)
  double rs_min{0.0}, rs_max{0.0}, zs_min{0.0}, zs_max{0.0};
  // The cell lookup the arm GS's PS4 tab performs (Ps4Panel::reachable).
  bool contains(double r, double z) const;
  double filledFraction() const;
};

// Empty grid (nr = nz = 0) when no pose qualifies.
WorkspaceGrid usableWorkspace(const VehicleModel & v, double fold_min);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_WORKSPACE_HPP_
