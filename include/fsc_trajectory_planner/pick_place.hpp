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
// the measured / typed point plus ONE shared EE offset (PickPlaceConfig::
// ee_offset); the heading is the bearing from the
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
  // THE EE offset [m], one for both claw legs (2026-10-03, user decision: the
  // same payload on the same pillars, so the claw goes to the same place
  // relative to the box at both): from the box CG to the grasp point, in the
  // box's frame. Each claw target is its 4-D pose plus the offset in that
  // pose's frame: pick = obj_0 + Rz(pick_yaw) * ee_offset, place = the place
  // pose + Rz(place_yaw) * ee_offset.
  Vec3 ee_offset{Vec3::Zero()};
  // THE CLAW LEGS' DRONE YAW (2026-10-03, user design): true = the target
  // pose's yaw -- the captured object's at the pick (its handle aligned with
  // the jaws), the typed place pose's at the place -- so the arm meets the
  // payload the same way at both. false: face_target below.
  bool align_yaw{true};
  // align_yaw false: turn the nose to face the claw target (else keep the
  // previous heading)
  bool face_target{true};
  // retreat before any leg flown from a table: climb dz, then back off along
  // -nose by `back`, arm unchanged. Both zero disables it.
  double retreat_dz{0.15};
  double retreat_back{0.20};
  // execute_pick / execute_place first fly to the target + approach_dz (the
  // claw above it), then -- once the claw is within the arrival tolerance --
  // descend vertically onto it (2026-10-02 audit, M2). 0 disables.
  double approach_dz{0.10};
  // HOOK GRASP (2026-10-07, the box payload: a hanger whose arch rests on the
  // two closed fingers -- the fingers must slide in UNDER the arch, so neither
  // claw leg may come down onto the payload from above). Both 0 = the vertical
  // legs above, unchanged.
  // pick_approach_back > 0: execute_pick's approach rest is this far BEHIND the
  // goal (along -nose) at the goal's own height, and the pick "descent" is a
  // horizontal move in -- no trim-first split (the trimmed goal is flown
  // directly). The exit is still straight up: it lifts the payload.
  double pick_approach_back{0.0};
  // place_exit_back > 0: Exit To Place first backs off this far along -nose at
  // the hold's height (the fingers slide out from under the arch), then climbs
  // approach_dz.
  double place_exit_back{0.0};
  // EXIT TO PICK / PLACE duration [s] (2026-10-03); 0 = the leg bounds (a_max
  // 0.15 m/s^2: 3.2 s for a 0.20 m climb). Clamped to a payload that still
  // rests on its pillar the vehicle must lift it OFF fast: a min-snap climb
  // barely moves at first (8 mm in its first 0.7 s at 3.2 s), so the box stayed
  // on the cap ~0.9 s into the exit, and the decoupled rig, pinned to it,
  // tipped over (geo_16 / 19). When > 0 the exit is planned with T_min = exit_t
  // and its translational bounds opened just enough for it; the rotor and
  // joint-torque bounds still apply.
  double exit_t{0.0};
  // TRIM FIRST (2026-10-03): with a descent trim larger than kTrimFirstMin, the
  // descent becomes three segments -- the claw moves SIDEWAYS by the trim at
  // the hover height, holds there trim_settle_s, then descends STRAIGHT down.
  // 0 = the old single segment, which moves the claw sideways WHILE it
  // descends: the decoupled law (a ~5 cm trim) lagged that sideways part and
  // overshot ~10 mm when it stopped -- at the bottom, with the open fingertips
  // only 8.75 mm clear of the handle (geo_22: a finger on the handle, the claw
  // then swung +-0.1 m around it and knocked the payload off its pillar).
  double trim_settle_s{0.0};
  Geofence fence;
  // the execute_place leg
  ArmSweepOptions sweep;
};

struct PickPlaceTargets
{
  // base poses, already shifted by the operator's calibration
  BasePose start, place_start, land_start, land;
  // the box CG at the pick (mocap obj_0) and where it is to be released (the
  // typed place pose, or drop_0), world [m] -- the SAME reference, so one
  // ee_offset serves both
  Vec3 pick{Vec3::Zero()}, place{Vec3::Zero()};
  // the captured pick OBJECT's yaw [rad] (mocap obj_0 = the payload's bottom
  // box, its CG and heading; 2026-10-03): ee_offset is expressed in this
  // object frame, so the claw target turns with the object. 0 = world axes.
  double pick_yaw{0.0};
  // the place pose's yaw [rad] (typed, 2026-10-03; or drop_0's): the frame
  // ee_offset is applied in at the place. 0 = world axes.
  double place_yaw{0.0};
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

// A claw leg's EXIT (2026-10-03, user design: Ready -> gripper -> Exit): from
// `from` (the hold on the target, gripper just closed / opened) straight UP by
// approach_dz -- the safety margin -- heading and arm unchanged, back to the
// clearance point. Throws "<leg> exit: <reason>".
std::unique_ptr<Trajectory> planPickPlaceExit(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c, int leg, const RestSpec & from);

// Throws "<what>: the path leaves the geofence ..." when a sample of the body
// origin along `traj` is outside `fence` -- or, for a bound the start itself
// violates (a leg starting outside), further outside than the start.
void checkPathInFence(
  const VehicleModel & v, const Trajectory & traj, const Geofence & fence, const std::string & what);

// The mission dry run behind the Plan button: leg 0 from `from`, every later
// leg from the goal of the one before (a claw leg = its approach + descent,
// the leg after it retreating). With `exits`, each claw leg is followed by
// its EXIT climb (stored there, indexed by the claw leg) and the next leg
// starts from the clearance point instead, with no retreat. Throws
// "<leg>: <reason>" at the first infeasible leg.
std::vector<std::shared_ptr<const Trajectory>> planPickPlaceMission(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, const RestSpec & from,
  std::array<std::shared_ptr<const Trajectory>, kNumPickPlaceLegs> * exits = nullptr);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_PICK_PLACE_HPP_
