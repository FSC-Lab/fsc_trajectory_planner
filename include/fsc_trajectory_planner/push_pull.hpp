// MIT License
// Copyright (c) 2026 FSC Lab
//
// PUSH-AND-PULL MODE (2026-10-03): a box on a table is grasped by a handle on
// its SIDE and slid along the table by the vehicle, the arm held still. Six
// operator-triggered steps (five until 2026-10-16), each a compatible
// whole-body move that ends AT REST (the arm ground station's "Box Pushing" /
// "Valve Turning" tabs, one button each):
//
//   0 go_to_start  -> the claw approach_dz straight ABOVE the handle's grasp
//                     point, nose along the push (the box's yaw + yaw_from_box),
//                     the arm unfolding from home into the push pose on the way
//                     (2026-10-22, c.bypass: first ROUND the table's box end,
//                     then straight in along the push)
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
//   4 go_to_land_start -> (2026-10-16, user design) ONE transition from the
//                     exit hold to the typed Land Start (high, clear of the
//                     table), the heading turning the way the typed yaw says
//                     (-90 -> -360 deg is 270 deg CLOCKWISE), the arm home;
//                     checked against the table like every other leg.
//                     (2026-10-22, c.bypass: on along the push's line, over the
//                     pushed box, past the table's FAR end, then round that
//                     end back to Land Start -- go_to_start's mirror image)
//   5 land         -> straight from Land Start to the Land hover, heading kept.
//                     The planner never touches down: the touchdown is the
//                     operator's SAFETY -> land.
//
// The box is MEASURED (mocap obj_0: its CG and yaw); the grasp point is the
// box pose plus ONE EE offset in the box's frame. The drone faces along the
// push: its ACTUAL yaw is the box's yaw + yaw_from_box (-90 deg: the handle is
// on the box's +y side, the push goes along the box's -y). Every step is
// checked against the geofence AND against the TABLE (a keep-out box, typed or
// -- 2026-10-22 -- placed from the box capture): wherever the landing gear's
// footprint or a point of the arm is over the table top, it must stay
// table_clearance above it.

#ifndef FSC_TRAJECTORY_PLANNER_PUSH_PULL_HPP_
#define FSC_TRAJECTORY_PLANNER_PUSH_PULL_HPP_

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/pick_place.hpp"   // Geofence, BasePose, clawOf
#include "fsc_trajectory_planner/teleop_reference.hpp"   // ChainFilter
#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

enum PushPullStep : int
{
  kPlGoToStart = 0,
  kPlReady,
  kPlPush,
  kPlExit,
  kPlGoToLandStart,
  kPlLand,
  kNumPushPullSteps
};

// "go_to_start", "ready", "push", "exit", "go_to_land_start", "land" (the
// node's service names); "?" otherwise.
const char * pushPullStepName(int step);

// The table top as a keep-out box [m]: its footprint and the height of its top
// surface. The footprint's bounds are in the frame turned by `yaw` about
// (ox, oy): world axes when both are zero (the typed push_pull_table), the
// push's axes about the box for a table placed from the box (2026-10-22,
// tableFromBox: x along the push, the box end at x_min).
struct TableKeepOut
{
  double x_min{0.0}, x_max{0.0}, y_min{0.0}, y_max{0.0}, z_top{0.0};
  double ox{0.0}, oy{0.0}, yaw{0.0};
  bool valid() const {return x_max > x_min && y_max > y_min;}
  // p in the table's frame (x, y turned; z unchanged)
  Vec3 toLocal(const Vec3 & p) const;
  // the inverse: a point of the table's frame in the world
  Vec3 toWorld(const Vec3 & l) const;
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
  // a TURN's duration [s] (push_pull_push_time_s). Since 2026-10-21 a straight
  // push takes its duration from push_speed instead (pushPullPushTime).
  double push_time{12.0};
  // THE PUSH SPEED (2026-10-21, user): a straight push's MEAN speed [m/s]
  // (push_pull_push_mean_speed) -- the one source of its duration, which is
  // max(|push_distance| / push_speed, push_min_time). 0 = not set: a straight
  // push is refused. A min-snap move of d in T peaks at ~2.19 d / T.
  double push_speed{0.0};
  // a straight push's shortest duration [s] (push_pull_push_min_time_s): a
  // short push at the mean speed would accelerate at 7.51 v^2 / d. 0 = none.
  double push_min_time{0.0};
  // held in place at the grasp before the push moves [s] (the law has been in
  // contact since the end of ready; the node keeps the claw world-held)
  double push_settle{1.0};
  // THE CRUISE (2026-10-17): > 0 = the straight push speeds up over this many
  // seconds, holds a CONSTANT speed push_distance / (T - push_ramp), and slows
  // down over the same time, all inside T = pushPullPushTime(). Its speed ramps
  // are septic, so the CoM stays smooth through snap at every join. 0 = the
  // min-snap transition (unchanged: no constant-speed part, the speed peaks at
  // 2.19 x its mean). A box held by friction slides without stopping only
  // while the claw outruns the vehicle's ~0.7 Hz speed ripple (~25 mm/s in
  // the 10-16 Isaac pushes), and the min-snap spends over half its time below
  // that. Ignored by a turn. Must be below T / 2.
  double push_ramp{0.0};
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
  // THE TURN (2026-10-15, the valve-turning task): non-zero = the push step
  // ROTATES instead of sliding -- the claw, the base and the heading turn
  // together about the captured body's vertical axis (the box pose = the
  // valve's bearing axis + the beam's yaw; the grasp point = that axis +
  // ee_offset turned with the beam) by this angle [rad, + = counter-clockwise
  // seen from above], the arm held still, over push_time. The nose must lie
  // along the turn's tangent at the handle (yaw_from_box = +90 deg of the
  // handle's bearing for a + turn, -90 for a - turn). push_distance is then
  // ignored. 0 = the straight push, unchanged.
  double turn_angle{0.0};
  Geofence fence;
  TableKeepOut table;
  // A SECOND keep-out with the same rule (2026-10-15): the valve's swept
  // circle, whose BEAM top is higher than the table it stands on. Invalid
  // (x_max <= x_min) = none.
  TableKeepOut obstacle;
  // the gear bottom and every arm point must stay this far above the table
  // top wherever they are over it (+ table_margin_xy) [m]
  double table_clearance{0.05};
  double table_margin_xy{0.05};
  // the landing gear: its bottom below the body origin and the radius of its
  // footprint about the body origin [m] (AM_xfwd: skids 0.313 m down, within
  // +-0.17 m of the origin)
  double gear_depth{0.313};
  double gear_radius{0.25};
  // THE BOX STAYS ON THE TABLE (2026-10-16, the box-pushing campaign's worst
  // cases): > 0 = the box's half length along the push [m]; the plan is refused
  // unless both of the PUSHED box's ends along the push lie inside the table's
  // footprint, box_edge_margin in from its edges. Catches a box (or its mocap
  // body) turned 180 deg -- the nose, and with it the push, then points the
  // other way -- and a typed distance longer than the table. 0 = off (unchanged).
  // Straight pushes only (a turn keeps the body on its axis).
  double box_half_length{0.0};
  double box_edge_margin{0.03};
  // THE TABLE PLACED FROM THE BOX (2026-10-22, user: the lab table goes wherever
  // the floor is flat, so a room-fixed keep-out refused the box): true = `table`
  // is replaced by the table's footprint placed by the box capture -- the table
  // table_length along the push x table_depth across it, the box centred across
  // it with its back face (box_half_length behind its centre) table_box_gap in
  // from the table's end, the top box_half_height under the box centre. The
  // landing headings then turn with the box (as typed from the push heading),
  // and pushPullWaypoints refuses a start more than start_middle_tol off the
  // table's middle line (the layout rule: the vehicle starts on it, in front --
  // and the catch for a box or mocap body turned 180 deg). Straight pushes only.
  bool table_from_box{false};
  double table_length{1.0};
  double table_depth{0.15};
  double table_box_gap{0.10};
  double box_half_height{0.05};
  double start_middle_tol{0.30};
  // GO TO START GOES ROUND THE TABLE (2026-10-22, user): true = go_to_start
  // flies a smooth curve round the table's BOX end, the body bypass_clearance
  // or more from the table's footprint (seen from above), climbing to the
  // approach height and turning to the push heading on the way; it stops on the
  // push's axis bypass_clearance past the table's end and flies straight in
  // along the push while the arm unfolds. The start must be beside the table,
  // bypass_clearance + kBypassStartMargin or more from it. false = one
  // transition straight to the approach (unchanged).
  bool bypass{false};
  double bypass_clearance{0.70};
  // the handle's top above the grasp point [m] (the CAD fin: the claw closes 25 mm
  // under its top): with bypass the way home flies over the pushed box, which is
  // then a keep-out up to there (2026-10-22)
  double handle_above_grasp{0.025};
};

// The start's extra room for the curve round the table [m]: the curve leaves the
// start parallel to the table and bends towards it only past the table's end, by
// up to this much (measured on the degree-8 curve, push_pull.cpp).
constexpr double kBypassStartMargin = 0.05;

// The table's footprint placed from the box (PushPullConfig::table_from_box).
// Throws std::runtime_error when the sizes cannot describe a table under the box.
TableKeepOut tableFromBox(const PushPullConfig & c, const Vec3 & box, double box_yaw);

struct PushPullTargets
{
  // the box CG (mocap obj_0) and its yaw [rad], world
  Vec3 box{Vec3::Zero()};
  double box_yaw{0.0};
  // the hover over the landing spot (ACTUAL yaw, as typed: its 2*pi branch is
  // the turn direction from the push heading)
  BasePose land;
  // Land Start (2026-10-16): where go_to_land_start ends, ACTUAL yaw as typed
  // (the turn direction, as land's was); unset = the same pose as land
  BasePose land_start;
  bool land_start_set{false};
  // the start mark (Adjust: where the vehicle took off) -- the layout check of
  // a table placed from the box (2026-10-22); unset = no check
  BasePose start;
  bool start_set{false};
};

struct PushPullWaypoints
{
  RestSpec approach;   // go_to_start: the claw approach_dz above the grasp point
  RestSpec grasp;      // ready: the claw ON the grasp point
  RestSpec pushed;     // push: the base push_distance along the nose (or turned by turn_angle)
  RestSpec lifted;     // exit, first segment: approach_dz up, push pose kept
  RestSpec exited;     // exit: the arm home, in place
  RestSpec land_start; // go_to_land_start
  RestSpec land;       // land
  Vec3 approach_ee{Vec3::Zero()}, grasp_ee{Vec3::Zero()}, pushed_ee{Vec3::Zero()};
  double drone_yaw{0.0};          // ACTUAL yaw at the box [rad]
  Vec3 push_dir{Vec3::Zero()};    // unit, horizontal, world: the nose at the box
  Vec3 axis{Vec3::Zero()};        // the turn's axis (the captured body), world [m]
  // the table the plan was made against (2026-10-22): the typed one, or the one
  // placed from the box -- every later step is checked against THIS, so it
  // stays when the capture is dropped (an abort that touched the box)
  TableKeepOut table;
};

// The waypoints. Throws std::runtime_error "<step>: <reason>" when one is
// impossible on its own: an arm pose outside the joint box or too close to a
// singularity, a rest outside the geofence, or one over the table too low.
PushPullWaypoints pushPullWaypoints(
  const VehicleModel & v, const PushPullConfig & c, const PushPullTargets & t);

// Each step from the rest `from` (the CURRENT hold when flown). Throw
// std::runtime_error "<step>: <reason>"; every path is checked against the
// geofence and the table (c.table: the caller passes the plan's, wp.table).
// go_to_start with c.bypass: the curve round the table's box end, then straight
// in (see PushPullConfig::bypass); from a hold already within bypass_clearance
// of the table (a re-fly near it) the one transition as before.
std::unique_ptr<Trajectory> planPushPullGoToStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from);
// straight down onto the grasp point; `trim` shifts the goal (the hover
// error measured above the target, as the pick's descent trim)
std::unique_ptr<Trajectory> planPushPullReady(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from,
  const Vec3 & trim = Vec3::Zero());
// THE PUSH SPEED (2026-10-21): a straight push's duration [s] (the moving part,
// without push_settle) -- max(|push_distance| / push_speed, push_min_time).
// Throws when push_speed is not set (<= 0 or not finite).
double pushPullPushTime(const PushPullConfig & c);

// from the hold on the handle: settle push_settle s, then the base along the
// hold's nose by push_distance -- arm, heading and height unchanged. With
// c.turn_angle != 0: instead the whole rest (claw, base, heading) rotates
// about the vertical axis through `axis` by turn_angle over push_time, the
// arm unchanged (an analytic circle on a nonic phase, no solve).
std::unique_ptr<Trajectory> planPushPullPush(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, const Vec3 & axis = Vec3::Zero());
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
// one transition to Land Start, the heading turned as typed (a pause when
// already there); with c.bypass (2026-10-22) on along the push's line past the
// table's far end (over the pushed box, checked against it), then round the
// table's far corner to it
std::unique_ptr<Trajectory> planPushPullGoToLandStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from);
// straight to the Land hover, heading kept (a pause when already there)
std::unique_ptr<Trajectory> planPushPullLand(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from);
// ABORT: from `from` (a hold, or the reference a run was cut at) held still
// `release_wait` s (the claw on the handle: the gripper is opening -- a climb
// with the jaws still closed would lift the box), then straight up `climb`
// with the arm unchanged -- the jaws leave the handle vertically --, then the
// arm home in place. Checked against the fence only (it must never be refused
// for the table: it climbs away from it). `retreat` (2026-10-16, world [m], zero =
// none): after the release wait the base first moves by it -- back against the
// push, so the opened fingers leave the fin horizontally before the climb (Isaac:
// a jammed / tipping box was LIFTED by the open claw climbing straight up, and the
// vehicle flipped when it dropped off). `unbuckle` (2026-10-21, empty = none): the
// arm pose the FIRST motion leg after the release wait brings the arm to -- the
// retreat when there is one, else the climb; the climb and the fold home then
// start from it. Meant for an abort held on the MEASURED pose (the 10-21 flight:
// a q3-guard trip left q2 on its limit and q3 at ~2 deg, the elbow-singular
// boundary, and the abort held that for 9 s until the fold). When the claw would
// end more than 10 mm lower against the base than it starts, the climb carries the
// arm instead of the retreat, so the open claw never drops along the handle.
std::unique_ptr<Trajectory> planPushPullAbort(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, double climb, double release_wait = 0.0,
  const Vec3 & retreat = Vec3::Zero(), const std::optional<VecN> & unbuckle = std::nullopt);

// Throws "<what>: <part> is <h> m above the table top at t = .. s ..." when a
// sample of `traj` puts the landing gear (within gear_radius of the body
// origin) or a point of the arm chain over the table top (+ table_margin_xy)
// lower than table_clearance above it -- or, for a part the start itself has
// there, lower than at the start (a step starting low must climb away).
// allow_low_start false: the start must be clear too (a rest on its own).
// The same rule is applied to c.obstacle when it is valid.
void checkPathClearsTable(
  const VehicleModel & v, const Trajectory & traj, const PushPullConfig & c,
  const std::string & what, bool allow_low_start = true);

// The dry run behind the Plan button: go_to_start from `from`, every later
// step from the goal of the one before. Throws at the first infeasible step.
std::vector<std::shared_ptr<const Trajectory>> planPushPullMission(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from);

// PATH FOLLOWING FOR THE TURN (2026-10-18, push_pull_follow): the turn is
// flown at a governed PROGRESS tau instead of the wall clock. Each stream tick
// the caller hands step() a target rate in [0, 1] -- the smallest of its
// channel factors (followFactor: the measured valve's lag behind the plan,
// the vehicle's tilt, the wrist roll's excursion) -- and evaluates the plan at
// tau() with rates() (Trajectory::evalScaled). The target is gated
// ASYMMETRICALLY: it may fall at once but rises at most 1 / recover_s per
// second (slow down quickly, resume slowly, so a channel oscillating with the
// airframe's sway does not modulate the turn at that frequency), then passes a
// 4-stage chain filter of `bandwidth` [rad/s], whose output and three
// derivatives are the progress rate and its derivatives -- continuous through
// the third, so the reference stays continuous through snap. The filter's
// weights are non-negative and sum to one, so the rate never leaves [0, 1]:
// the turn never runs ahead of its schedule and never reverses. Target 1
// throughout = the scheduled turn exactly (tau = t).
class ProgressGovernor
{
public:
  void setup(double bandwidth, double recover_s);
  void reset();                       // tau = 0, rate 1
  void step(double target, double dt);
  double tau() const {return tau_;}
  double rate() const {return rates_[0];}
  double gate() const {return gate_;}
  const std::array<double, 4> & rates() const {return rates_;}

private:
  ChainFilter f_;
  double recover_{5.0};
  double tau_{0.0};
  double gate_{1.0};
  std::array<double, 4> rates_{{1.0, 0.0, 0.0, 0.0}};
};

// A channel's rate factor: 1 at or below `soft`, 0 at or above `hard`, linear
// between; 1 when the channel is off (hard <= soft or hard <= 0).
double followFactor(double e, double soft, double hard);

// THE BOX SAMPLE GATE (2026-10-21): the drone mocap guard's jump test
// (fsc_drone_state_estimator_ros2 MocapGuard), with its constants. A box sample
// d_m [m] and a_deg [deg] from the LAST GOOD one, dt [s] after it, is good when
//   d_m <= jump_m + kBoxGateV dt + 1/2 kBoxGateA dt^2  and  a_deg <= jump_deg + kBoxGateW dt.
// The allowance grows with the gap, so a pose that really moved is let back in.
// A non-positive limit disables its test (both: every sample is good).
constexpr double kBoxGateV = 1.0;    // [m/s]
constexpr double kBoxGateA = 2.0;    // [m/s^2]
constexpr double kBoxGateW = 90.0;   // [deg/s]
bool boxSampleWithinGate(double d_m, double a_deg, double dt, double jump_m, double jump_deg);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_PUSH_PULL_HPP_
