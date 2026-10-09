// MIT License
// Copyright (c) 2026 FSC Lab
//
// PUSH-AND-PULL MODE (2026-10-03): a box on a table is grasped by a handle on
// its SIDE and slid along the table by the vehicle, the arm held still. Five
// operator-triggered steps, each a compatible whole-body move that ends AT
// REST (the arm ground station's "Push & Pull" tab, one button each):
//
//   0 go_to_start  -> the claw approach_dz straight ABOVE the handle's grasp
//                     point, nose along the push (the box's yaw + yaw_from_box),
//                     the arm unfolding from home into the push pose on the way
//   1 ready        -> straight down onto the grasp point (the arm GS opens the
//                     gripper first; the node holds the claw in the WORLD
//                     from the hover above through the grasp)
//   2 push         -> (the arm GS has closed the gripper on the handle) the
//                     base moves push_distance along the nose -- the arm, the
//                     heading and the height held FIXED -- after a settle in
//                     place while the node switches the law to contact
//   3 exit         -> (the arm GS has opened the gripper) straight up by
//                     approach_dz with the arm still in the push pose (the
//                     open jaws leave the handle vertically), then the arm
//                     folds home in place
//   4 go_to_land   -> a hover over the landing spot, arm home, the heading
//                     turning the way the typed land yaw says (the scene's
//                     -90 -> -360 deg is 270 deg CLOCKWISE). The planner never
//                     lands: the touchdown is the operator's SAFETY -> land.
//
// The box is MEASURED (mocap obj_0: its CG and yaw); the grasp point is the
// box pose plus ONE EE offset in the box's frame. The drone faces along the
// push: its ACTUAL yaw is the box's yaw + yaw_from_box (-90 deg: the handle is
// on the box's +y side, the push goes along the box's -y). Every step is
// checked against the geofence AND against the TABLE (an axis-aligned keep-
// out box): wherever the landing gear's footprint or a point of the arm is
// over the table top, it must stay table_clearance above it.

#ifndef FSC_TRAJECTORY_PLANNER_PUSH_PULL_HPP_
#define FSC_TRAJECTORY_PLANNER_PUSH_PULL_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/pick_place.hpp"   // Geofence, BasePose, clawOf
#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

enum PushPullStep : int
{
  kPlGoToStart = 0,
  kPlReady,
  kPlPush,
  kPlExit,
  kPlGoToLand,
  kNumPushPullSteps
};

// "go_to_start", "ready", "push", "exit", "go_to_land" (the node's service
// names); "?" otherwise.
const char * pushPullStepName(int step);

// The table top as an axis-aligned keep-out box, WORLD frame [m]: its
// footprint and the height of its top surface.
struct TableKeepOut
{
  double x_min{0.0}, x_max{0.0}, y_min{0.0}, y_max{0.0}, z_top{0.0};
  bool valid() const {return x_max > x_min && y_max > y_min;}
  // horizontal distance from p to the footprint, 0 over it
  double distanceXY(const Vec3 & p) const;
};

struct PushPullConfig
{
  // arm poses [rad, MODEL convention]
  VecN home{(VecN() << 0.0, 0.698132, 0.698132, 0.0).finished()};
  VecN push_pose{VecN::Zero()};
  // box CG -> grasp point, in the BOX's frame [m] (z up)
  Vec3 ee_offset{Vec3::Zero()};
  // the drone's ACTUAL yaw = the box's yaw + this [rad]; the nose (the push
  // direction) then points along the box's -y for -pi/2
  double yaw_from_box{-0.5 * M_PI};
  // the safety margin: go_to_start holds the claw this far above the grasp
  // point, ready descends it, exit climbs it [m]
  double approach_dz{0.20};
  // (2026-10-06) the approach ALSO starts this far BEHIND the grasp point along
  // the nose [m]: with a claw pitched forward (push pose beta ~60 deg) the jaws
  // slide onto a vertical post handle instead of dropping onto a fin. 0 = the
  // straight-down approach (unchanged). approach_dz may then be 0.
  double approach_back{0.0};
  // the exit climb [m]; 0 = approach_dz (unchanged)
  double exit_dz{0.0};
  // how far the push moves the base along the nose [m]; < 0 PULLS back
  double push_distance{0.50};
  // the push's shortest duration [s] (its speed: a min-snap move of d in T
  // peaks at ~2.19 d / T)
  double push_time{12.0};
  // held in place at the grasp before the push moves [s] (the law has been in
  // contact since the end of ready; the node keeps the claw world-held)
  double push_settle{1.0};
  // the exit climb's duration [s]; 0 = the leg bounds
  double exit_time{3.0};
  // THE UNLOAD (2026-10-07): when the slide ends the box is held by static
  // friction while the claw still sits short of its reference, so the arm's
  // impedance stores K_y e (~2 N at the end of a pull) that is released into
  // the arm when CONTACT ends or the jaws open. unload_time > 0: the push holds
  // unload_wait at its end, the node then re-seats the hold on the claw's
  // measured position (planPushPullUnload, the law's own task error, gripping
  // and in CONTACT) over unload_time and holds unload_wait again before the
  // push completes. 0 = off (unchanged).
  double unload_time{0.0};
  double unload_wait{1.0};
  Geofence fence;
  TableKeepOut table;
  // the gear bottom and every arm point must stay this far above the table
  // top wherever they are over it (+ table_margin_xy) [m]
  double table_clearance{0.05};
  double table_margin_xy{0.05};
  // the landing gear: its bottom below the body origin and the radius of its
  // footprint about the body origin [m] (AM_xfwd: skids 0.313 m down, within
  // +-0.17 m of the origin)
  double gear_depth{0.313};
  double gear_radius{0.25};
};

struct PushPullTargets
{
  // the box CG (mocap obj_0) and its yaw [rad], world
  Vec3 box{Vec3::Zero()};
  double box_yaw{0.0};
  // the hover over the landing spot (ACTUAL yaw, as typed: its 2*pi branch is
  // the turn direction from the push heading)
  BasePose land;
};

struct PushPullWaypoints
{
  RestSpec approach;   // go_to_start: the claw approach_dz above the grasp point
  RestSpec grasp;      // ready: the claw ON the grasp point
  RestSpec pushed;     // push: the base push_distance along the nose
  RestSpec lifted;     // exit, first segment: approach_dz up, push pose kept
  RestSpec exited;     // exit: the arm home, in place
  RestSpec land;       // go_to_land
  Vec3 approach_ee{Vec3::Zero()}, grasp_ee{Vec3::Zero()}, pushed_ee{Vec3::Zero()};
  double drone_yaw{0.0};          // ACTUAL yaw at the box [rad]
  Vec3 push_dir{Vec3::Zero()};    // unit, horizontal, world: the nose at the box
};

// The waypoints. Throws std::runtime_error "<step>: <reason>" when one is
// impossible on its own: an arm pose outside the joint box or too close to a
// singularity, a rest outside the geofence, or one over the table too low.
PushPullWaypoints pushPullWaypoints(
  const VehicleModel & v, const PushPullConfig & c, const PushPullTargets & t);

// Each step from the rest `from` (the CURRENT hold when flown). Throw
// std::runtime_error "<step>: <reason>"; every path is checked against the
// geofence and the table.
std::unique_ptr<Trajectory> planPushPullGoToStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from);
// straight down onto the grasp point; `trim` shifts the goal (the hover
// error measured above the target, as the pick's descent trim)
std::unique_ptr<Trajectory> planPushPullReady(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from,
  const Vec3 & trim = Vec3::Zero());
// from the hold on the handle: settle push_settle s, then the base along the
// hold's nose by push_distance -- arm, heading and height unchanged
std::unique_ptr<Trajectory> planPushPullPush(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from);
// straight up approach_dz (arm unchanged), then the arm home in place
// The unload (after the push, still gripping, in CONTACT): the whole rest
// translated by `shift` (the claw's measured task error, world [m]) over
// c.unload_time, then held c.unload_wait. A shift under 0.5 mm is only the hold.
std::unique_ptr<Trajectory> planPushPullUnload(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, const Vec3 & shift);
std::unique_ptr<Trajectory> planPushPullExit(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from);
std::unique_ptr<Trajectory> planPushPullGoToLand(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from);
// ABORT: from `from` (a hold, or the reference a run was cut at) held still
// `release_wait` s (the claw on the handle: the gripper is opening -- a climb
// with the jaws still closed would lift the box), then straight up `climb`
// with the arm unchanged -- the jaws leave the handle vertically --, then the
// arm home in place. Checked against the fence only (it must never be refused
// for the table: it climbs away from it).
std::unique_ptr<Trajectory> planPushPullAbort(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, double climb, double release_wait = 0.0);

// Throws "<what>: <part> is <h> m above the table top at t = .. s ..." when a
// sample of `traj` puts the landing gear (within gear_radius of the body
// origin) or a point of the arm chain over the table top (+ table_margin_xy)
// lower than table_clearance above it -- or, for a part the start itself has
// there, lower than at the start (a step starting low must climb away).
// allow_low_start false: the start must be clear too (a rest on its own).
void checkPathClearsTable(
  const VehicleModel & v, const Trajectory & traj, const PushPullConfig & c,
  const std::string & what, bool allow_low_start = true);

// The dry run behind the Plan button: go_to_start from `from`, every later
// step from the goal of the one before. Throws at the first infeasible step.
std::vector<std::shared_ptr<const Trajectory>> planPushPullMission(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_PUSH_PULL_HPP_
