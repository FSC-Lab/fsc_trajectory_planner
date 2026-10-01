// MIT License
// Copyright (c) 2026 FSC Lab
//
// PICK-AND-PLACE MODE (2026-10-01): six operator-triggered legs, each a
// compatible whole-body move that ends AT REST, flown one button at a time:
//
//   0 go_to_start        -> the START base pose, arm at home
//   1 execute_pick       -> the claw on the PICK point (mocap body obj_0), arm
//                           in the pick pose, nose turned to face the object
//   2 go_to_place_start  -> retreat (up + back off the table), then the PLACE
//                           START base pose with the arm in the carry pose
//   3 execute_place      -> the claw on the PLACE point (mocap body drop_0),
//                           the arm SWEEPING between its limits on the way
//                           (arm_sweep_planner.hpp): the payload moves in 3-D
//   4 go_to_land_start   -> retreat, then the LAND START base pose, arm home
//   5 execute_land       -> the LAND base pose (a hover over the spot), arm home. The
//                           planner never lands, arms or changes PX4 mode: the
//                           touchdown is the operator's SAFETY -> land.
//
// Base poses are given in the world frame with the ACTUAL yaw (the drone
// ground station's convention); RestSpec carries the MODEL heading phi =
// yaw - pi/2. A claw target places the grasp point (the EE of the model) on
// the measured point plus an offset; the heading is the bearing from the
// previous leg's base to that point, so the vehicle turns to face the object
// over the whole leg (the flat planner's heading channel is smooth and
// rate-bounded), and the arm sits on that bearing.
//
// THE PICK POSE [0, 0, 0, 0] (the default): upper arm hanging straight down,
// forearm out along the nose, claw pointing straight DOWN -- the arm's two
// pitch joints at a right angle each. The grasp point is then 0.155 m ahead
// of and 0.340 m below the body origin, sigma_nd 0.169 (> 0.10). It is the
// bottom of the arm's reach: from there the arm alone cannot lower the claw
// (the base must), only raise it, pull it in or swing it sideways.

#ifndef FSC_TRAJECTORY_PLANNER_PICK_PLACE_HPP_
#define FSC_TRAJECTORY_PLANNER_PICK_PLACE_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/arm_sweep_planner.hpp"
#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

enum PickPlaceLeg : int
{
  kGoToStart = 0,
  kExecutePick,
  kGoToPlaceStart,
  kExecutePlace,
  kGoToLandStart,
  kExecuteLand,
  kNumPickPlaceLegs
};

// "go_to_start", "execute_pick", ... (the node's service names); "?" otherwise.
const char * pickPlaceLegName(int leg);
// Legs whose goal is a claw target (execute_pick, execute_place).
bool pickPlaceLegIsClaw(int leg);
// Legs that retreat off the table first (go_to_place_start, go_to_land_start).
bool pickPlaceLegRetreats(int leg);

struct BasePose
{
  Vec3 p{Vec3::Zero()};   // world [m]
  double yaw{0.0};        // ACTUAL yaw [rad]
};

struct PickPlaceConfig
{
  // arm poses [rad, MODEL convention]
  VecN home{(VecN() << 0.0, 0.698132, 0.698132, 0.0).finished()};
  VecN carry{(VecN() << 0.0, 0.698132, 0.698132, 0.0).finished()};
  VecN pick_pose{VecN::Zero()};
  VecN place_pose{VecN::Zero()};
  // claw target = measured point + offset (world) [m]
  Vec3 pick_ee_offset{Vec3::Zero()};
  Vec3 place_ee_offset{Vec3::Zero()};
  // turn the nose to face the claw target (else keep the previous heading)
  bool face_target{true};
  // retreat before go_to_place_start / go_to_land_start: climb dz, then back
  // off along -nose by `back`, arm unchanged. Both zero disables it.
  double retreat_dz{0.15};
  double retreat_back{0.20};
  // the execute_place leg
  ArmSweepOptions sweep;
};

struct PickPlaceTargets
{
  // base poses, already shifted by the operator's calibration
  BasePose start, place_start, land_start, land;
  // measured claw points (mocap obj_0 / drop_0), world [m]
  Vec3 pick{Vec3::Zero()}, place{Vec3::Zero()};
};

struct PickPlaceWaypoints
{
  std::array<RestSpec, kNumPickPlaceLegs> goal;   // the rest each leg ends on
  std::array<Vec3, kNumPickPlaceLegs> ee;         // the claw at that rest, world
};

// The six goal rests. Throws std::runtime_error "<leg>: <reason>" when a goal
// is impossible on its own (an arm pose outside the joint box or closer to a
// singularity than the vehicle's sigma_nd margin).
PickPlaceWaypoints pickPlaceWaypoints(
  const VehicleModel & v, const PickPlaceConfig & c, const PickPlaceTargets & t);

// The base pose `from` climbs and backs off to before leaving a table.
RestSpec retreatRest(const RestSpec & from, const PickPlaceConfig & c);

// World claw (grasp point) of a rest spec.
Vec3 clawOf(const VehicleModel & v, const RestSpec & r);

// One leg from the rest `from` (the CURRENT hold when flown). Throws
// std::runtime_error "<leg>: <reason>" when it is infeasible.
std::unique_ptr<Trajectory> planPickPlaceLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from);

// The mission dry run behind the Plan button: leg 0 from `from`, every later
// leg from the goal of the one before. Throws "<leg>: <reason>" at the first
// infeasible leg.
std::vector<std::shared_ptr<const Trajectory>> planPickPlaceMission(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, const RestSpec & from);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_PICK_PLACE_HPP_
