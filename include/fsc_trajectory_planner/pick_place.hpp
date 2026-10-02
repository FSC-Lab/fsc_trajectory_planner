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
// Legs that, in the nominal order, start at a table (go_to_place_start,
// go_to_land_start): the mission dry run retreats before them. When FLOWN, a
// leg retreats whenever the vehicle holds at a table (see planPickPlaceLeg).
bool pickPlaceLegRetreats(int leg);

// The volume every leg must stay in (2026-10-02 audit, H1): body-origin
// height and a horizontal box, WORLD frame [m]. The body stands at 0.305 m on
// its gear and the whole-body law oscillates ~+-0.1 m, hence the 0.6 m floor.
struct Geofence
{
  double min_z{0.6}, max_z{1.8};
  double x_min{-2.5}, x_max{2.5}, y_min{-2.5}, y_max{2.5};
  // how far outside each bound [m], 0 inside: floor, ceiling, x_min, x_max, y_min, y_max
  std::array<double, 6> excess(const Vec3 & body) const;
  // bound k (the excess() index) violated by `body`, in words
  std::string describe(const Vec3 & body, int bound) const;
  // "" inside, else the first bound violated
  std::string violation(const Vec3 & body) const;
};

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
  // retreat before any leg flown from a table: climb dz, then back off along
  // -nose by `back`, arm unchanged. Both zero disables it.
  double retreat_dz{0.15};
  double retreat_back{0.20};
  // execute_pick / execute_place first fly to the target + approach_dz (the
  // claw above it), then -- once the claw is within the arrival tolerance --
  // descend vertically onto it (2026-10-02 audit, M2). 0 disables.
  double approach_dz{0.10};
  Geofence fence;
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
  // claw legs: the rest approach_dz above the goal (same heading and arm) and
  // its claw; every other leg: the goal itself
  std::array<RestSpec, kNumPickPlaceLegs> approach;
  std::array<Vec3, kNumPickPlaceLegs> approach_ee;
};

// The six goal rests and the approach rests. Throws std::runtime_error
// "<leg>: <reason>" when a goal is impossible on its own: an arm pose outside
// the joint box or closer to a singularity than the vehicle's sigma_nd
// margin, or a goal / approach body outside the geofence.
PickPlaceWaypoints pickPlaceWaypoints(
  const VehicleModel & v, const PickPlaceConfig & c, const PickPlaceTargets & t);

// The base pose `from` climbs and backs off to before leaving a table.
RestSpec retreatRest(const RestSpec & from, const PickPlaceConfig & c);

// World claw (grasp point) of a rest spec.
Vec3 clawOf(const VehicleModel & v, const RestSpec & r);

// One leg from the rest `from` (the CURRENT hold when flown): to the goal, or
// for a claw leg to its APPROACH rest. `from_table`: the vehicle holds at a
// table (after a claw leg), so the leg retreats first. The whole body path is
// checked against the geofence. Throws std::runtime_error "<leg>: <reason>".
std::unique_ptr<Trajectory> planPickPlaceLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from, bool from_table);

// A claw leg's DESCENT: from `from` (the approach hold) vertically onto the
// goal. Throws std::runtime_error "<leg> descent: <reason>".
std::unique_ptr<Trajectory> planPickPlaceDescent(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from);

// Throws "<what>: the path leaves the geofence ..." when a sample of the body
// origin along `traj` is outside `fence` -- or, for a bound the start itself
// violates (a leg starting outside), further outside than the start.
void checkPathInFence(
  const VehicleModel & v, const Trajectory & traj, const Geofence & fence, const std::string & what);

// The mission dry run behind the Plan button: leg 0 from `from`, every later
// leg from the goal of the one before (a claw leg = its approach + descent,
// the leg after it retreating). Throws "<leg>: <reason>" at the first
// infeasible leg.
std::vector<std::shared_ptr<const Trajectory>> planPickPlaceMission(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, const RestSpec & from);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_PICK_PLACE_HPP_
