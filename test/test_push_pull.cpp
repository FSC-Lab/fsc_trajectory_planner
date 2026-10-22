// The push-and-pull mode on the T650 aerial manipulator (push_pull.hpp): the
// waypoints (the claw on the box's handle, the nose along the push, the gear
// clear of the table), each of the six steps (go_to_start, the vertical
// ready descent, the push with the arm and heading held, the exit climbing
// before it folds, go_to_land_start -- off the table, then the arc with the
// land turn flown as typed --, land) and the table check.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <array>

#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/push_pull.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"

using namespace fsc_trajectory_planner;

namespace
{
constexpr double kDeg = M_PI / 180.0;

// The Isaac scene (07_px4_t650_aerial_manipulator_box_pushing.py): a
// 1.0 x 0.15 m console table 0.80 m high, its length along y, in front of the
// start; a 200 (y) x 100 x 60 mm box near its +y end, the fin handle on the
// box's +y face, the grasp point 0.155 m out along +y and 0.065 m up from its
// centre (the generated push-and-pull yaml's push_pull_ee_offset).
PushPullTargets scene()
{
  PushPullTargets t;
  t.box << 1.20, 0.25, 0.83;
  t.box_yaw = 0.0;
  t.land.p << -0.40, -0.07, 0.80;
  t.land.yaw = -360.0 * kDeg;
  return t;
}

PushPullConfig config(const VehicleModel & v)
{
  PushPullConfig c;
  c.home = v.home;
  c.push_pose << 0.0, -20.0 * kDeg, 30.0 * kDeg, 0.0;
  c.ee_offset << 0.0, 0.155, 0.065;
  c.table.x_min = 1.125;
  c.table.x_max = 1.275;
  c.table.y_min = -0.50;
  c.table.y_max = 0.50;
  c.table.z_top = 0.80;
  c.fence.max_z = 2.2;
  c.push_speed = c.push_distance / 12.0;   // 0.50 m in 12 s (2026-10-21: the speed sets the time)
  return c;
}

RestSpec hover()
{
  RestSpec r;
  r.x_b << 0.10, -0.07, 1.0;
  r.phi = -0.5 * M_PI;   // actual yaw 0
  r.q << 0.0, 40.0 * kDeg, 40.0 * kDeg, 0.0;
  return r;
}

double headingOf(const WbReference & ref) {return std::atan2(ref.b1_d(1), ref.b1_d(0));}

Vec3 bodyOf(const VehicleModel & v, const WbReference & ref)
{
  Vec3 r0c;
  armKinematics(ref.q_d, v.params, &r0c, nullptr, nullptr);
  return ref.x_cd - Rz(headingOf(ref)) * r0c;
}
}  // namespace

TEST(PushPull, WaypointsPutTheClawOnTheHandleFacingThePush)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PushPullTargets t = scene();
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, t);

  // the grasp point = the box + the offset (box yaw 0: the offset as given)
  EXPECT_LT((wp.grasp_ee - (t.box + c.ee_offset)).norm(), 1e-12);
  EXPECT_LT((clawOf(*v, wp.grasp) - wp.grasp_ee).norm(), 1e-12);
  // the drone's yaw = the box's - 90 deg; the nose (the push) along -y
  EXPECT_NEAR(wp.drone_yaw, -0.5 * M_PI, 1e-12);
  EXPECT_LT((wp.push_dir - Vec3(0.0, -1.0, 0.0)).norm(), 1e-12);
  EXPECT_NEAR(wp.grasp.phi, -M_PI, 1e-12);
  EXPECT_LT((wp.grasp.q - c.push_pose).norm(), 1e-12);
  // the approach: the safety margin straight above
  EXPECT_LT((wp.approach_ee - (wp.grasp_ee + Vec3(0.0, 0.0, c.approach_dz))).norm(), 1e-12);
  // pushed: the claw push_distance along -y, same height
  EXPECT_LT((wp.pushed_ee - (wp.grasp_ee + Vec3(0.0, -c.push_distance, 0.0))).norm(), 1e-12);
  // the push pose: claw 0.316 m under the body origin, so the gear (0.313 m)
  // clears the 0.80 m table top by ~10 cm with the grasp 0.095 m above it
  EXPECT_NEAR(wp.grasp.x_b(2) - wp.grasp_ee(2), 0.316, 2e-3);
  EXPECT_GT(wp.grasp.x_b(2) - c.gear_depth - c.table.z_top, 0.09);
  // the land hover: as typed, arm home
  EXPECT_LT((wp.land.x_b - t.land.p).norm(), 1e-12);
  EXPECT_LT((wp.land.q - v->home).norm(), 1e-12);
}

TEST(PushPull, PushHoldsTheArmAndHeadingAndMovesAlongTheNose)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp);

  EXPECT_GE(traj->duration(), c.push_settle + pushPullPushTime(c) - 1e-9);
  const WbReference r0 = traj->eval(0.0);
  const Vec3 claw0 = r0.r_ed;
  double max_dq = 0.0, max_dpsi = 0.0, max_dz = 0.0, max_lat = 0.0, max_speed = 0.0;
  for (int k = 0; k <= 400; ++k) {
    const double t = traj->duration() * k / 400.0;
    const WbReference r = traj->eval(t);
    max_dq = std::max(max_dq, (r.q_d - c.push_pose).cwiseAbs().maxCoeff());
    const double dpsi = headingOf(r) - headingOf(r0);
    max_dpsi = std::max(max_dpsi, std::abs(std::atan2(std::sin(dpsi), std::cos(dpsi))));
    max_dz = std::max(max_dz, std::abs(r.r_ed(2) - claw0(2)));
    max_lat = std::max(max_lat, std::abs(r.r_ed(0) - claw0(0)));
    max_speed = std::max(max_speed, r.x_cd_dot.norm());
    if (t <= c.push_settle) {
      EXPECT_LT((r.r_ed - claw0).norm(), 1e-9) << "moved during the settle at t = " << t;
    }
  }
  EXPECT_LT(max_dq, 1e-6) << "the arm moved during the push";
  EXPECT_LT(max_dpsi, 1e-6) << "the heading moved during the push";
  // the claw's height: only what the ~0.2 deg acceleration tilt swings it (a
  // compatible reference tilts the body to push it)
  EXPECT_LT(max_dz, 1e-3) << "the claw changed height during the push";
  EXPECT_LT(max_lat, 1e-6) << "the claw left the push line";
  // ~2.19 d / T for a min-snap move of d = 0.5 m in 12 s
  EXPECT_LT(max_speed, 0.12);
  const RestSpec end = traj->goalRest();
  EXPECT_LT((end.x_b - wp.pushed.x_b).norm(), 1e-9);
  EXPECT_LT((traj->eval(traj->duration()).r_ed - wp.pushed_ee).norm(), 1e-6);
}

// THE CRUISE (2026-10-17): push_ramp > 0 = ramp up, a CONSTANT speed d / (T -
// T_r), ramp down, inside push_time; the arm and heading held, the CoM on the
// push line, its chain exact and continuous through snap at every join, at
// rest at both ends; a ramp of half the push time or more is refused.
TEST(PushPull, PushCanCruiseAtConstantSpeed)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = config(*v);
  c.push_speed = c.push_distance / 20.0;   // 20 s
  c.push_ramp = 3.0;
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp);
  const double T = pushPullPushTime(c);
  const double V = c.push_distance / (T - c.push_ramp);
  const double t0 = c.push_settle, t1 = t0 + T;
  EXPECT_NEAR(traj->duration(), t1, 1e-9);
  const Vec3 nose = wp.push_dir;
  const WbReference r0 = traj->eval(0.0);
  double max_dq = 0.0, max_dpsi = 0.0, max_lat = 0.0, max_speed = 0.0, max_cruise = 0.0;
  double max_fd = 0.0;
  const double h = 1e-4;
  for (int k = 0; k <= 2000; ++k) {
    const double t = traj->duration() * k / 2000.0;
    const WbReference r = traj->eval(t);
    max_dq = std::max(max_dq, (r.q_d - c.push_pose).cwiseAbs().maxCoeff());
    const double dpsi = headingOf(r) - headingOf(r0);
    max_dpsi = std::max(max_dpsi, std::abs(std::atan2(std::sin(dpsi), std::cos(dpsi))));
    const Vec3 dx = r.x_cd - r0.x_cd;
    max_lat = std::max(max_lat, (dx - dx.dot(nose) * nose).norm());
    max_speed = std::max(max_speed, r.x_cd_dot.norm());
    if (t > t0 + c.push_ramp + 1e-6 && t < t1 - c.push_ramp - 1e-6) {
      max_cruise = std::max(
        {max_cruise, std::abs(r.x_cd_dot.dot(nose) - V), r.x_cd_ddot.norm(), r.x_cd_d4.norm()});
    }
    if (t > t0 + h && t < t1 - h) {
      const WbReference a = traj->eval(t - h), b = traj->eval(t + h);
      max_fd = std::max(
        {max_fd, ((b.x_cd - a.x_cd) / (2.0 * h) - r.x_cd_dot).norm(),
          ((b.x_cd_dot - a.x_cd_dot) / (2.0 * h) - r.x_cd_ddot).norm(),
          ((b.x_cd_ddot - a.x_cd_ddot) / (2.0 * h) - r.x_cd_d3).norm(),
          ((b.x_cd_d3 - a.x_cd_d3) / (2.0 * h) - r.x_cd_d4).norm()});
    }
  }
  EXPECT_LT(max_dq, 1e-9) << "the arm moved during the push";
  EXPECT_LT(max_dpsi, 1e-9) << "the heading moved during the push";
  EXPECT_LT(max_lat, 1e-9) << "the CoM left the push line";
  EXPECT_LT(max_cruise, 1e-9) << "the cruise is not at the constant speed " << V;
  EXPECT_LT(max_speed, V + 1e-9) << "faster than the cruise somewhere";
  EXPECT_LT(max_fd, 1e-6) << "the CoM chain is not its own derivative";
  // every join (settle | ramp | cruise | ramp | end) continuous through snap
  for (const double tj : {t0, t0 + c.push_ramp, t1 - c.push_ramp, t1 - 1e-12}) {
    const WbReference a = traj->eval(tj - 1e-9), b = traj->eval(tj + 1e-9);
    EXPECT_LT(
      std::max(
        {(a.x_cd_dot - b.x_cd_dot).norm(), (a.x_cd_ddot - b.x_cd_ddot).norm(),
          (a.x_cd_d3 - b.x_cd_d3).norm(), (a.x_cd_d4 - b.x_cd_d4).norm()}),
      1e-6) << "a step at t = " << tj;
  }
  const WbReference end = traj->eval(traj->duration());
  EXPECT_LT(
    std::max({end.x_cd_dot.norm(), end.x_cd_ddot.norm(), end.x_cd_d3.norm(), end.x_cd_d4.norm()}),
    1e-9) << "not at rest at the end";
  EXPECT_LT((traj->goalRest().x_b - wp.pushed.x_b).norm(), 1e-9);
  EXPECT_LT((end.r_ed - wp.pushed_ee).norm(), 1e-6);
  // the ramps must fit
  c.push_ramp = 10.0;
  EXPECT_THROW(planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp), std::runtime_error);
}

// THE UNLOAD (2026-10-07): off by default (the push unchanged); on, the push
// ends with a hold of unload_wait, and the unload leg translates the whole rest
// by the measured task error -- the claw by exactly that, arm and heading held
// -- then holds; a sub-millimetre error is the hold alone.
TEST(PushPull, UnloadReSeatsTheHoldOnTheMeasuredClaw)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const double T_off = planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp)->duration();
  c.unload_time = 1.5;
  c.unload_wait = 1.0;
  const auto push = planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp);
  EXPECT_NEAR(push->duration(), T_off + c.unload_wait, 1e-9);
  EXPECT_LT((push->goalRest().x_b - wp.pushed.x_b).norm(), 1e-9);

  const RestSpec from = push->goalRest();
  const Vec3 claw0 = clawOf(*v, from);
  const Vec3 shift(0.001, -0.007, 0.006);   // the law's e_y at the end of a pull
  const auto un = planPushPullUnload(*v, *planner, PlanOptions{}, c, from, shift);
  EXPECT_GE(un->duration(), c.unload_time + c.unload_wait - 1e-9);
  EXPECT_LT((un->eval(0.0).r_ed - claw0).norm(), 1e-6) << "the unload must start on the push's end";
  EXPECT_LT((un->eval(un->duration()).r_ed - (claw0 + shift)).norm(), 1e-6);
  EXPECT_LT((un->goalRest().q - from.q).norm(), 1e-12);
  EXPECT_NEAR(un->goalRest().phi, from.phi, 1e-12);
  double max_dq = 0.0, max_off = 0.0;
  for (int k = 0; k <= 200; ++k) {
    const WbReference r = un->eval(un->duration() * k / 200.0);
    max_dq = std::max(max_dq, (r.q_d - from.q).cwiseAbs().maxCoeff());
    // the claw moves along the shift only (no overshoot past either end)
    const Vec3 d = r.r_ed - claw0;
    const double s = d.dot(shift) / shift.squaredNorm();
    max_off = std::max(max_off, (d - s * shift).norm());
    EXPECT_GT(s, -0.05);
    EXPECT_LT(s, 1.05);
  }
  EXPECT_LT(max_dq, 1e-6) << "the arm moved in the unload";
  // off the line only by what the acceleration tilt swings the claw (as the push)
  EXPECT_LT(max_off, 1e-3) << "the claw left the shift's line";

  const auto still = planPushPullUnload(*v, *planner, PlanOptions{}, c, from, Vec3(2e-4, 0.0, 0.0));
  EXPECT_NEAR(still->duration(), c.unload_wait, 1e-9);
  EXPECT_LT((still->goalRest().x_b - from.x_b).norm(), 1e-12);
}

TEST(PushPull, ReadyDescendsStraightOntoTheGraspPoint)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullReady(*v, *planner, PlanOptions{}, c, wp, wp.approach);
  double max_side = 0.0;
  for (int k = 0; k <= 200; ++k) {
    const WbReference r = traj->eval(traj->duration() * k / 200.0);
    max_side = std::max(max_side, (r.r_ed - wp.grasp_ee).head<2>().norm());
  }
  EXPECT_LT(max_side, 1e-6) << "the claw moved sideways on the way down";
  EXPECT_LT((traj->eval(traj->duration()).r_ed - wp.grasp_ee).norm(), 1e-6);
  // a trim shifts the goal
  const Vec3 trim{0.004, -0.003, 0.0};
  const auto trimmed = planPushPullReady(*v, *planner, PlanOptions{}, c, wp, wp.approach, trim);
  EXPECT_LT((trimmed->goalRest().x_b - (wp.grasp.x_b + trim)).norm(), 1e-9);
}

TEST(PushPull, ExitClimbsOffTheHandleBeforeTheArmFolds)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullExit(*v, *planner, PlanOptions{}, c, wp.pushed);
  const auto * seq = dynamic_cast<const SequenceTrajectory *>(traj.get());
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(seq->segmentStarts().size(), 2u);
  const double t_climb = seq->segmentStarts()[1];
  EXPECT_GE(t_climb, c.exit_time - 1e-9);
  // the climb: arm unchanged, claw straight up
  for (int k = 0; k <= 100; ++k) {
    const WbReference r = traj->eval(t_climb * k / 100.0);
    EXPECT_LT((r.q_d - c.push_pose).cwiseAbs().maxCoeff(), 1e-6);
    EXPECT_LT((r.r_ed - wp.pushed_ee).head<2>().norm(), 1e-6);
  }
  EXPECT_NEAR(traj->eval(t_climb).r_ed(2), wp.pushed_ee(2) + c.approach_dz, 1e-6);
  // then home, in place
  const RestSpec end = traj->goalRest();
  EXPECT_LT((end.q - v->home).norm(), 1e-9);
  EXPECT_LT((end.x_b - wp.exited.x_b).norm(), 1e-9);
}

TEST(PushPull, GoToLandStartTurnsTheTypedWayRound)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullGoToLandStart(*v, *planner, PlanOptions{}, c, wp, wp.exited);
  // -90 -> -360 deg typed: 270 deg CLOCKWISE (the short way is 90 deg the other way)
  double unwrapped = headingOf(traj->eval(0.0)), prev = unwrapped, min_rate = 0.0;
  for (int k = 1; k <= 600; ++k) {
    const double h = headingOf(traj->eval(traj->duration() * k / 600.0));
    double d = h - prev;
    d = std::atan2(std::sin(d), std::cos(d));
    unwrapped += d;
    min_rate = std::min(min_rate, d);
    EXPECT_LE(d, 1e-9) << "the heading turned counter-clockwise";
    prev = h;
  }
  EXPECT_NEAR(unwrapped - headingOf(traj->eval(0.0)), -270.0 * kDeg, 1e-3);
  EXPECT_LT((traj->goalRest().x_b - wp.land.x_b).norm(), 1e-9);

  // from any heading but the push's (a re-fly from the land hover): the short way
  RestSpec at_land = wp.land;
  at_land.phi = std::atan2(std::sin(at_land.phi), std::cos(at_land.phi));
  const auto again = planPushPullGoToLandStart(*v, *planner, PlanOptions{}, c, wp, at_land);
  EXPECT_NEAR(headingOf(again->eval(again->duration())), headingOf(again->eval(0.0)), 1e-6);
}

TEST(PushPull, TheTableIsChecked)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PushPullTargets t = scene();
  // a grasp point too low for the gear: 6 cm lower leaves the skids 3.8 cm over
  // the top (< the 5 cm clearance; the scene's own grasp leaves 9.8 cm)
  PushPullConfig c = config(*v);
  c.ee_offset(2) -= 0.06;
  try {
    pushPullWaypoints(*v, c, t);
    FAIL() << "a grasp that puts the landing gear onto the table was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("landing gear"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("table"), std::string::npos) << e.what();
  }
  // the table off: accepted
  c.table = TableKeepOut{};
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, t));
}

// THE BOX STAYS ON THE TABLE (2026-10-16): with box_half_length set, a push
// whose box would end off the table -- the box (or its mocap body) turned
// 180 deg, so the nose and the push point the other way, or too long a
// distance -- is refused; the scene's own 0.5 m push from y = 0.25 is not.
TEST(PushPull, ThePushedBoxMustStayOnTheTable)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  PushPullTargets t = scene();
  PushPullConfig c = config(*v);
  c.box_half_length = 0.10;          // the 200 mm box along the push
  c.box_edge_margin = 0.03;
  // the scene's push: y 0.25 -> -0.25, its trailing end at -0.35 > -0.47
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, t));
  // too far: 0.70 m ends at -0.45, its leading end at -0.55, off the table
  c.push_distance = 0.70;
  try {
    pushPullWaypoints(*v, c, t);
    FAIL() << "a push that slides the box off the table's end was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("off the table"), std::string::npos) << e.what();
  }
  // the box turned 180 deg: the push goes +y, from 0.25 to 0.75
  c.push_distance = 0.50;
  t.box_yaw = M_PI;
  t.box << 1.20, -0.25, 0.83;        // mirrored, so only the direction is wrong
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, t));   // -0.25 -> +0.25: still on it
  t.box << 1.20, 0.25, 0.83;
  try {
    pushPullWaypoints(*v, c, t);
    FAIL() << "a push toward the near end (box yaw 180 deg) was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("box's yaw"), std::string::npos) << e.what();
  }
  // off (the default): the same push is accepted -- unchanged behaviour
  c.box_half_length = 0.0;
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, t));
}

TEST(PushPull, TheMissionDryRunsAndEveryStepEndsAtRest)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto steps = planPushPullMission(*v, *planner, PlanOptions{}, c, wp, hover());
  ASSERT_EQ(steps.size(), static_cast<size_t>(kNumPushPullSteps));
  double total = 0.0;
  for (int k = 0; k < kNumPushPullSteps; ++k) {
    const auto & s = steps[k];
    total += s->duration();
    for (double t : {0.0, s->duration()}) {
      const WbReference r = s->eval(t);
      EXPECT_LT(r.x_cd_dot.norm(), 1e-6) << pushPullStepName(k) << " not at rest at t = " << t;
      EXPECT_LT(r.qdot_d.norm(), 1e-6) << pushPullStepName(k);
    }
    std::cout << pushPullStepName(k) << ": " << s->duration() << " s -- " << s->diag().summary << "\n";
  }
  EXPECT_LT((steps[kPlGoToStart]->goalRest().x_b - wp.approach.x_b).norm(), 1e-9);
  EXPECT_LT((steps[kPlReady]->goalRest().x_b - wp.grasp.x_b).norm(), 1e-9);
  EXPECT_LT((steps[kPlPush]->goalRest().x_b - wp.pushed.x_b).norm(), 1e-9);
  EXPECT_LT((steps[kPlGoToLandStart]->goalRest().x_b - wp.land_start.x_b).norm(), 1e-9);
  EXPECT_LT((steps[kPlLand]->goalRest().x_b - wp.land.x_b).norm(), 1e-9);
  std::cout << "mission: " << total << " s\n";
}

TEST(PushPull, AbortClimbsThenFoldsHome)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullAbort(*v, *planner, PlanOptions{}, c, wp.grasp, 0.30);
  const auto * seq = dynamic_cast<const SequenceTrajectory *>(traj.get());
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(seq->segmentStarts().size(), 2u);
  const double t1 = seq->segmentStarts()[1];
  for (int k = 0; k <= 50; ++k) {
    const WbReference r = traj->eval(t1 * k / 50.0);
    EXPECT_LT((r.q_d - c.push_pose).cwiseAbs().maxCoeff(), 1e-6) << "folded before it climbed";
  }
  EXPECT_NEAR(bodyOf(*v, traj->eval(traj->duration()))(2), wp.grasp.x_b(2) + 0.30, 1e-6);
  EXPECT_LT((traj->goalRest().q - v->home).norm(), 1e-9);
}

// THE RETREAT (2026-10-16): with a retreat the abort holds for the release, then
// moves the base horizontally by it -- level, the arm still in its push pose --
// and only then climbs and folds; it ends above the retreated point.
TEST(PushPull, AbortRetreatsHorizontallyBeforeTheClimb)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const Vec3 back = -0.03 * wp.push_dir;
  const auto traj = planPushPullAbort(*v, *planner, PlanOptions{}, c, wp.grasp, 0.30, 1.0, back);
  const auto * seq = dynamic_cast<const SequenceTrajectory *>(traj.get());
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(seq->segmentStarts().size(), 4u);   // release, retreat, climb, fold
  const double t_ret = seq->segmentStarts()[1], t_climb = seq->segmentStarts()[2];
  // the release: still on the grasp
  EXPECT_LT((bodyOf(*v, traj->eval(0.5 * t_ret)) - wp.grasp.x_b).norm(), 1e-6);
  // the retreat: level, the arm held, ending `back` away
  for (int k = 0; k <= 50; ++k) {
    const WbReference r = traj->eval(t_ret + (t_climb - t_ret) * k / 50.0);
    EXPECT_NEAR(bodyOf(*v, r)(2), wp.grasp.x_b(2), 1e-6) << "climbed before it backed off";
    EXPECT_LT((r.q_d - c.push_pose).cwiseAbs().maxCoeff(), 1e-6);
  }
  EXPECT_LT((bodyOf(*v, traj->eval(t_climb)) - (wp.grasp.x_b + back)).norm(), 1e-6);
  // the end: above the retreated point, the arm home
  const Vec3 end = bodyOf(*v, traj->eval(traj->duration()));
  EXPECT_LT((end.head<2>() - (wp.grasp.x_b + back).head<2>()).norm(), 1e-6);
  EXPECT_NEAR(end(2), wp.grasp.x_b(2) + 0.30, 1e-6);
  EXPECT_LT((traj->goalRest().q - v->home).norm(), 1e-9);
}

// THE UNBUCKLE (2026-10-21): an abort held on a buckled measured arm (the 10-21
// q3-guard trip: q2 on its +45 limit, q3 at 5 deg) keeps the arm still through the
// release wait (the jaws open), then brings q2 / q3 to the push pose's in the
// retreat with q1 / q4 untouched; the open claw never drops against its start and
// never moves toward the box; the climb and the fold start from the unbuckled pose.
// From an over-folded arm (unbuckling would LOWER the claw 84 mm against the base)
// the climb carries it instead, and the claw still only rises.
TEST(PushPull, AbortUnbucklesTheArmOnceTheJawsAreOpen)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const Vec3 back = -0.03 * wp.push_dir;
  auto clawOf = [&](const WbReference & r) {
      Vec3 r0e;
      armKinematics(r.q_d, v->params, nullptr, &r0e, nullptr);
      return Vec3(bodyOf(*v, r) + Rz(headingOf(r)) * r0e);
    };
  VecN buckled, folded;
  buckled << 1.9 * kDeg, 44.9 * kDeg, 5.0 * kDeg, -2.0 * kDeg;   // 10-21, model signs
  folded << 1.0 * kDeg, 40.0 * kDeg, 45.0 * kDeg, 3.0 * kDeg;
  const char * const names[2] = {"buckled", "over-folded"};
  for (int n = 0; n < 2; ++n) {
    SCOPED_TRACE(names[n]);
    const bool retreat_carries = n == 0;
    RestSpec from = wp.grasp;
    from.q = n == 0 ? buckled : folded;
    VecN unb = from.q;
    unb(1) = 32.0 * kDeg;
    unb(2) = 28.0 * kDeg;
    const auto traj = planPushPullAbort(*v, *planner, PlanOptions{}, c, from, 0.30, 1.5, back, unb);
    const auto * seq = dynamic_cast<const SequenceTrajectory *>(traj.get());
    ASSERT_NE(seq, nullptr);
    ASSERT_EQ(seq->segmentStarts().size(), 4u);   // release, retreat, climb, fold
    const double t_ret = seq->segmentStarts()[1], t_climb = seq->segmentStarts()[2];
    const double t_fold = seq->segmentStarts()[3];
    EXPECT_NEAR(t_ret, 1.5, 1e-9);
    // the release: the measured arm held while the jaws open
    for (int i = 0; i <= 20; ++i) {
      EXPECT_LT((traj->eval(t_ret * i / 20.0).q_d - from.q).cwiseAbs().maxCoeff(), 1e-9);
    }
    const Vec3 claw0 = clawOf(traj->eval(0.0));
    // the retreat and the climb: q1 / q4 untouched, the claw never below its start,
    // never toward the box along the push
    for (int i = 0; i <= 100; ++i) {
      const WbReference r = traj->eval(t_ret + (t_fold - t_ret) * i / 100.0);
      EXPECT_NEAR(r.q_d(0), from.q(0), 1e-9);
      EXPECT_NEAR(r.q_d(3), from.q(3), 1e-9);
      const Vec3 claw = clawOf(r);
      EXPECT_GT(claw(2), claw0(2) - 1e-3) << "the open claw dropped along the handle";
      if (retreat_carries) {   // the buckled arm folds back toward the body
        EXPECT_LT((claw - claw0).dot(wp.push_dir), 1e-3) << "the open claw moved toward the box";
      }
    }
    // which leg carried it
    const VecN q_at_climb = traj->eval(t_climb).q_d;
    EXPECT_LT((q_at_climb - (retreat_carries ? unb : from.q)).cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT((traj->eval(t_fold).q_d - unb).cwiseAbs().maxCoeff(), 1e-9);
    // the fold ends home, above the retreated point
    const Vec3 end = bodyOf(*v, traj->eval(traj->duration()));
    EXPECT_LT((end.head<2>() - (from.x_b + back).head<2>()).norm(), 1e-6);
    EXPECT_NEAR(end(2), from.x_b(2) + 0.30, 1e-6);
    EXPECT_LT((traj->goalRest().q - v->home).norm(), 1e-9);
  }
  // no retreat (the mocap-guard abort): the climb carries the unbuckle
  RestSpec from = wp.grasp;
  from.q = buckled;
  VecN unb = from.q;
  unb(1) = 32.0 * kDeg;
  unb(2) = 28.0 * kDeg;
  const auto traj = planPushPullAbort(*v, *planner, PlanOptions{}, c, from, 0.30, 1.5, Vec3::Zero(), unb);
  const auto * seq = dynamic_cast<const SequenceTrajectory *>(traj.get());
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(seq->segmentStarts().size(), 3u);   // release, climb, fold
  EXPECT_LT((traj->eval(seq->segmentStarts()[2]).q_d - unb).cwiseAbs().maxCoeff(), 1e-9);
}

// THE TURN (2026-10-15, the valve-turning task): with turn_angle set, the push
// step rotates the whole rest about the captured body's vertical axis -- the
// claw on a circle of its radius from the axis, the heading turning with it,
// the arm held -- and ends back on the grasp (a full turn) with the heading
// advanced by 2 pi; the reference's derivative chain is consistent with its
// own position to finite-difference accuracy; a nose off the tangent is refused.
TEST(PushPull, TurnRotatesTheRestAboutTheAxisWithTheArmHeld)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = config(*v);
  // the valve: the captured body IS the axis, the fin 0.160 m along the beam
  // (yaw 180 deg -> toward the start), the nose +90 deg from the handle's
  // bearing for a counter-clockwise turn
  c.ee_offset << 0.160, 0.0006, 0.1795;
  c.push_pose << 0.0, -5.0 * kDeg, 40.0 * kDeg, 0.0;
  c.yaw_from_box = 0.5 * M_PI;
  c.turn_angle = 2.0 * M_PI;
  c.push_time = 36.0;
  c.push_settle = 1.5;
  c.gear_depth = 0.2425;   // the T650 gear the valve scene flies
  PushPullTargets t = scene();
  t.box << 1.20, 0.0, 0.80;
  t.box_yaw = M_PI;
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, t);
  // the grasp point: the axis + the offset turned by the beam's yaw
  EXPECT_LT((wp.grasp_ee - (t.box + Rz(t.box_yaw) * c.ee_offset)).norm(), 1e-12);
  // the nose: -y (the tangent at the fin, which sits on the axis' -x side)
  EXPECT_LT((wp.push_dir - Vec3(0.0, -1.0, 0.0)).norm(), 1e-9);
  // a full turn ends on the grasp, heading + 2 pi
  EXPECT_LT((wp.pushed.x_b - wp.grasp.x_b).norm(), 1e-9);
  EXPECT_NEAR(wp.pushed.phi, wp.grasp.phi + 2.0 * M_PI, 1e-12);
  EXPECT_LT((wp.pushed_ee - wp.grasp_ee).norm(), 1e-9);

  const auto traj = planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp, wp.axis);
  EXPECT_NEAR(traj->duration(), c.push_settle + c.push_time, 1e-9);
  const double r_claw = std::hypot(wp.grasp_ee(0) - t.box(0), wp.grasp_ee(1) - t.box(1));
  EXPECT_NEAR(r_claw, 0.160, 1e-3);
  const double r_com = std::hypot(traj->eval(0.0).x_cd(0) - t.box(0), traj->eval(0.0).x_cd(1) - t.box(1));
  double max_dq = 0.0, max_dr = 0.0, max_drc = 0.0, max_dz = 0.0, max_fd = 0.0, max_tan = 0.0;
  double unwrapped = headingOf(traj->eval(0.0)), prev = unwrapped;
  const double h = 1e-3;
  for (int k = 0; k <= 720; ++k) {
    const double tt = traj->duration() * k / 720.0;
    const WbReference r = traj->eval(tt);
    max_dq = std::max(max_dq, (r.q_d - c.push_pose).cwiseAbs().maxCoeff());
    // the claw and the CoM stay on their circles about the axis, at height
    max_dr = std::max(max_dr, std::abs(std::hypot(r.r_ed(0) - t.box(0), r.r_ed(1) - t.box(1)) - r_claw));
    max_drc = std::max(max_drc, std::abs(std::hypot(r.x_cd(0) - t.box(0), r.x_cd(1) - t.box(1)) - r_com));
    max_dz = std::max(max_dz, std::abs(r.r_ed(2) - wp.grasp_ee(2)));
    // the nose stays tangent: the CoM velocity is along the heading's tangent
    if (r.x_cd_dot.norm() > 1e-3) {
      const Vec3 radial = r.x_cd - t.box;
      max_tan = std::max(max_tan, std::abs(r.x_cd_dot.dot(radial) / (r.x_cd_dot.norm() * radial.norm())));
    }
    // derivative chain: velocity and acceleration vs central differences
    if (tt > h && tt < traj->duration() - h) {
      const WbReference a = traj->eval(tt - h), b = traj->eval(tt + h);
      const Vec3 v_fd = (b.x_cd - a.x_cd) / (2.0 * h);
      const Vec3 a_fd = (b.x_cd_dot - a.x_cd_dot) / (2.0 * h);
      const Vec3 j_fd = (b.x_cd_ddot - a.x_cd_ddot) / (2.0 * h);
      max_fd = std::max({max_fd, (v_fd - r.x_cd_dot).norm(), (a_fd - r.x_cd_ddot).norm(),
          (j_fd - r.x_cd_d3).norm() * 0.1});
    }
    double d = headingOf(r) - prev;
    d = std::atan2(std::sin(d), std::cos(d));
    unwrapped += d;
    EXPECT_GE(d, -1e-9) << "the heading turned clockwise at t = " << tt;
    prev = headingOf(r);
    if (tt <= c.push_settle) {
      EXPECT_LT((r.r_ed - wp.grasp_ee).norm(), 1e-9) << "moved during the settle at t = " << tt;
    }
  }
  EXPECT_LT(max_dq, 1e-9) << "the arm moved during the turn";
  // the claw: on its circle to what the acceleration tilt swings it (a
  // compatible reference tilts the body to accelerate; the CoM is exact)
  EXPECT_LT(max_dr, 2e-3) << "the claw left its circle";
  EXPECT_LT(max_drc, 1e-9) << "the CoM left its circle";
  EXPECT_LT(max_dz, 1e-3) << "the claw changed height during the turn";
  EXPECT_LT(max_tan, 1e-6) << "the CoM velocity is not tangent";
  EXPECT_LT(max_fd, 1e-4) << "the derivative chain disagrees with finite differences";
  EXPECT_NEAR(unwrapped - headingOf(traj->eval(0.0)), 2.0 * M_PI, 1e-6);
  const WbReference end = traj->eval(traj->duration());
  EXPECT_LT((end.r_ed - wp.grasp_ee).norm(), 1e-6);
  EXPECT_LT(end.x_cd_dot.norm(), 1e-9);
  EXPECT_LT((traj->goalRest().x_b - wp.pushed.x_b).norm(), 1e-9);
  EXPECT_NEAR(traj->goalRest().phi, wp.pushed.phi, 1e-9);
  std::cout << "turn: " << traj->diag().summary << "\n";

  // the nose across the tangent (the box's -90 deg) is refused
  c.yaw_from_box = -0.5 * M_PI;
  try {
    pushPullWaypoints(*v, c, t);
    FAIL() << "a nose 180 deg off the turn's tangent was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("tangent"), std::string::npos) << e.what();
  }
  // a clockwise turn wants the other side
  c.turn_angle = -2.0 * M_PI;
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, t));
  // after a clockwise full turn Go To Land still turns the typed way: the
  // nose at the fin is now +90 (the other tangent), so the typed -360 is 450
  // deg clockwise, from the unwrapped heading
  const PushPullWaypoints wpc = pushPullWaypoints(*v, c, t);
  const auto tc = planPushPullPush(*v, *planner, PlanOptions{}, c, wpc.grasp, wpc.axis);
  const auto ex = planPushPullExit(*v, *planner, PlanOptions{}, c, tc->goalRest());
  const auto land = planPushPullGoToLandStart(*v, *planner, PlanOptions{}, c, wpc, ex->goalRest());
  double unw = headingOf(land->eval(0.0)), pv = unw;
  for (int k = 1; k <= 600; ++k) {
    const double hh = headingOf(land->eval(land->duration() * k / 600.0));
    double d = hh - pv;
    d = std::atan2(std::sin(d), std::cos(d));
    unw += d;
    pv = hh;
  }
  EXPECT_NEAR(wpc.drone_yaw, 0.5 * M_PI, 1e-9);
  EXPECT_NEAR(unw - headingOf(land->eval(0.0)), -450.0 * kDeg, 1e-3);
}

TEST(PushPull, TheObstacleIsCheckedLikeTheTable)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  PushPullConfig c = config(*v);
  c.ee_offset << 0.160, 0.0006, 0.1795;
  c.push_pose << 0.0, -5.0 * kDeg, 40.0 * kDeg, 0.0;
  c.yaw_from_box = 0.5 * M_PI;
  c.turn_angle = 2.0 * M_PI;
  c.gear_depth = 0.2425;   // the T650 gear
  PushPullTargets t = scene();
  t.box << 1.20, 0.0, 0.80;
  t.box_yaw = M_PI;
  // the valve's swept circle, its beam top 0.0495 m above the table: at the
  // [0, -5, 40, 0] pose the skids pass 0.13 m over it -- accepted
  c.obstacle.x_min = 1.20 - 0.20;
  c.obstacle.x_max = 1.20 + 0.20;
  c.obstacle.y_min = -0.20;
  c.obstacle.y_max = 0.20;
  c.obstacle.z_top = 0.80 + 0.0495;
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, t));
  // the 60 deg push pose puts the skids 2.7 cm over the beam (7.7 over the table): refused, naming the obstacle
  c.push_pose << 0.0, 26.0 * kDeg, 34.0 * kDeg, 0.0;
  try {
    pushPullWaypoints(*v, c, t);
    FAIL() << "a pose that puts the landing gear onto the beam was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("landing gear"), std::string::npos) << e.what();
    EXPECT_NE(std::string(e.what()).find("obstacle"), std::string::npos) << e.what();
  }
  // ... and accepted with the obstacle off (the table alone is 7.7 cm away)
  c.obstacle = TableKeepOut{};
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, t));
}

// THE WAY TO LAND START (2026-10-16, user: "simple but a safe transition"):
// ONE transition from the exit hold to the typed Land Start (high), the
// heading turning the typed 270 deg clockwise without ever turning back, the
// gear above the table top by the clearance wherever it is over the table,
// then land straight down.
TEST(PushPull, GoToLandStartIsOneTransitionClearOfTheTable)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  PushPullTargets t = scene();
  t.land_start.p << -0.40, -0.07, 1.30;
  t.land_start.yaw = -360.0 * kDeg;
  t.land_start_set = true;
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, t);
  const auto traj = planPushPullGoToLandStart(*v, *planner, PlanOptions{}, c, wp, wp.exited);
  const double T = traj->duration();
  EXPECT_LT((traj->goalRest().x_b - wp.land_start.x_b).norm(), 1e-9);
  for (const double te : {0.0, T}) {
    const WbReference r = traj->eval(te);
    EXPECT_LT(r.x_cd_dot.norm() + r.x_cd_ddot.norm(), 1e-9) << te;
  }
  const double h0 = headingOf(traj->eval(0.0));
  double unwrapped = 0.0, prev = h0, min_gear_gap = 1e9;
  const int n = 2000;
  for (int k = 1; k <= n; ++k) {
    const WbReference r = traj->eval(T * k / n);
    const double h = headingOf(r);
    const double d = std::atan2(std::sin(h - prev), std::cos(h - prev));
    EXPECT_LE(d, 1e-9) << "the heading turned counter-clockwise";
    unwrapped += d;
    prev = h;
    const Vec3 body = bodyOf(*v, r);
    if (c.table.distanceXY(body) < 0.25) {   // the gear's footprint over the table
      min_gear_gap = std::min(min_gear_gap, body(2) - c.gear_depth - c.table.z_top);
    }
  }
  EXPECT_NEAR(unwrapped, -270.0 * kDeg, 1e-3);
  EXPECT_GT(min_gear_gap, c.table_clearance);
  std::cout << "go_to_land_start: " << T << " s, gear >= " << min_gear_gap << " m over the table -- "
            << traj->diag().summary << "\n";

  // land: straight down from Land Start, heading and arm kept
  const auto land = planPushPullLand(*v, *planner, PlanOptions{}, c, wp, traj->goalRest());
  EXPECT_LT((land->goalRest().x_b - wp.land.x_b).norm(), 1e-9);
  for (int k = 0; k <= 50; ++k) {
    const WbReference r = land->eval(land->duration() * k / 50.0);
    const Vec3 body = bodyOf(*v, r);
    EXPECT_NEAR(body(0), wp.land.x_b(0), 1e-3);
    EXPECT_NEAR(body(1), wp.land.x_b(1), 1e-3);
    EXPECT_NEAR(std::atan2(std::sin(headingOf(r) - h0 - unwrapped), std::cos(headingOf(r) - h0 - unwrapped)),
      0.0, 1e-6);
  }
}

// PATH FOLLOWING (2026-10-18): the turn flown at a governed progress. With the
// progress rate held at 1 the scaled evaluation IS the scheduled one; with the
// governor slowing, holding and resuming the turn, the reference stays the
// rigid rotation (arm held, CoM on its circle) and its derivative chain stays
// consistent with its own position through snap; the rate never leaves [0, 1]
// and recovers no faster than recover_s.
TEST(PushPull, TurnFollowsAGovernedProgressExactly)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = config(*v);
  c.ee_offset << 0.160, 0.0006, 0.1795;
  c.push_pose << 0.0, 10.0 * kDeg, 20.0 * kDeg, 0.0;
  c.yaw_from_box = 0.5 * M_PI;
  c.turn_angle = 2.0 * M_PI;
  c.push_time = 36.0;
  c.push_settle = 1.5;
  c.gear_depth = 0.2425;
  PushPullTargets t = scene();
  t.box << 1.20, 0.0, 0.80;
  t.box_yaw = M_PI;
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, t);
  const auto traj = planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp, wp.axis);
  ASSERT_TRUE(traj->timeScalable());

  // rate 1: bit for bit the scheduled turn
  const std::array<double, 4> one{{1.0, 0.0, 0.0, 0.0}};
  double max_same = 0.0;
  for (int k = 0; k <= 400; ++k) {
    const double tau = traj->duration() * k / 400.0;
    const WbReference a = traj->eval(tau), b = traj->evalScaled(tau, one);
    max_same = std::max({max_same, (a.x_cd - b.x_cd).norm(), (a.x_cd_dot - b.x_cd_dot).norm(),
        (a.x_cd_ddot - b.x_cd_ddot).norm(), (a.x_cd_d3 - b.x_cd_d3).norm(),
        (a.x_cd_d4 - b.x_cd_d4).norm(), (a.b1_d - b.b1_d).norm(), (a.b1_d_dot - b.b1_d_dot).norm(),
        (a.b1_d_ddot - b.b1_d_ddot).norm(), (a.r_ed - b.r_ed).norm(), (a.q_d - b.q_d).norm()});
  }
  EXPECT_EQ(max_same, 0.0) << "rate 1 is not the scheduled turn";

  // an all-ones target: tau = t
  {
    ProgressGovernor g;
    g.setup(6.0, 5.0);
    const double dt = 0.01;
    for (int k = 0; k < 3000; ++k) {g.step(1.0, dt);}
    EXPECT_NEAR(g.tau(), 30.0, 1e-9);
    EXPECT_NEAR(g.rate(), 1.0, 1e-12);
  }

  // slow, hold, resume: 1 until 10 s, 0 (hold) to 16 s, 0.4 to 22 s, then 1
  ProgressGovernor g;
  g.setup(6.0, 5.0);
  const double dt = 0.004;
  const double r_com = std::hypot(traj->eval(0.0).x_cd(0) - t.box(0), traj->eval(0.0).x_cd(1) - t.box(1));
  std::vector<WbReference> refs;
  std::vector<double> taus, rates;
  double t_hold_rate = 1.0, t_recover = -1.0;
  for (int k = 0; k < 30000 && g.tau() < traj->duration(); ++k) {
    const double tt = k * dt;
    const double target = tt < 10.0 ? 1.0 : (tt < 16.0 ? 0.0 : (tt < 22.0 ? 0.4 : 1.0));
    g.step(target, dt);
    if (tt > 15.5 && tt < 16.0) {t_hold_rate = std::min(t_hold_rate, g.rate());}
    if (t_recover < 0.0 && tt > 22.0 && g.gate() >= 1.0) {t_recover = tt - 22.0;}
    refs.push_back(traj->evalScaled(g.tau(), g.rates()));
    taus.push_back(g.tau());
    rates.push_back(g.rate());
  }
  ASSERT_GT(refs.size(), 100u);
  EXPECT_GE(g.tau(), traj->duration()) << "the governed turn never finished";
  EXPECT_LT(t_hold_rate, 1e-3) << "the hold did not stop the progress";
  // 0.4 -> 1 at 1 / recover_s per second: 3 s
  EXPECT_NEAR(t_recover, 0.6 * 5.0, 0.01);
  double max_fd[4] = {0.0, 0.0, 0.0, 0.0}, scale[4] = {0.0, 0.0, 0.0, 0.0};
  double max_dq = 0.0, max_drc = 0.0;
  for (size_t k = 0; k < refs.size(); ++k) {
    EXPECT_GE(rates[k], 0.0);
    EXPECT_LE(rates[k], 1.0);
    if (k > 0) {EXPECT_GE(taus[k], taus[k - 1]) << "the progress reversed";}
    max_dq = std::max(max_dq, (refs[k].q_d - c.push_pose).cwiseAbs().maxCoeff());
    max_drc = std::max(max_drc,
        std::abs(std::hypot(refs[k].x_cd(0) - t.box(0), refs[k].x_cd(1) - t.box(1)) - r_com));
    if (k == 0 || k + 1 >= refs.size()) {continue;}
    const WbReference & a = refs[k - 1], & r = refs[k], & b = refs[k + 1];
    const Vec3 fd[4] = {(b.x_cd - a.x_cd) / (2.0 * dt), (b.x_cd_dot - a.x_cd_dot) / (2.0 * dt),
      (b.x_cd_ddot - a.x_cd_ddot) / (2.0 * dt), (b.x_cd_d3 - a.x_cd_d3) / (2.0 * dt)};
    const Vec3 an[4] = {r.x_cd_dot, r.x_cd_ddot, r.x_cd_d3, r.x_cd_d4};
    for (int i = 0; i < 4; ++i) {
      max_fd[i] = std::max(max_fd[i], (fd[i] - an[i]).norm());
      scale[i] = std::max(scale[i], an[i].norm());
    }
  }
  EXPECT_LT(max_dq, 1e-9) << "the arm moved";
  EXPECT_LT(max_drc, 1e-9) << "the CoM left its circle";
  // snap is continuous, but at the test's 1 -> 0 target STEP the fifth
  // derivative steps, so a central difference of jerk is off there by about
  // dt x that step: a looser bound for the fourth (a real channel ramps)
  const double tol[4] = {2e-3, 2e-3, 2e-3, 5e-2};
  for (int i = 0; i < 4; ++i) {
    EXPECT_LT(max_fd[i], tol[i] * std::max(scale[i], 1e-3))
      << "derivative " << i + 1 << " disagrees with finite differences (" << max_fd[i]
      << " vs peak " << scale[i] << ")";
  }
  std::cout << "governed turn: " << refs.size() * dt << " s for a " << traj->duration()
            << " s schedule; FD error / peak " << max_fd[0] / scale[0] << " " << max_fd[1] / scale[1]
            << " " << max_fd[2] / scale[2] << " " << max_fd[3] / scale[3] << "\n";
}

TEST(PushPull, FollowFactorRampsBetweenSoftAndHard)
{
  EXPECT_EQ(followFactor(2.0, 4.0, 8.0), 1.0);
  EXPECT_EQ(followFactor(4.0, 4.0, 8.0), 1.0);
  EXPECT_NEAR(followFactor(6.0, 4.0, 8.0), 0.5, 1e-12);
  EXPECT_EQ(followFactor(8.0, 4.0, 8.0), 0.0);
  EXPECT_EQ(followFactor(99.0, 4.0, 8.0), 0.0);
  EXPECT_EQ(followFactor(99.0, 0.0, 0.0), 1.0);   // channel off
  EXPECT_EQ(followFactor(std::nan(""), 4.0, 8.0), 0.0);   // unknown = hold
}

// THE PUSH SPEED (2026-10-21, user): a straight push lasts max(|d| / v, floor) --
// the mean speed is its one source (push_time is a turn's only); no speed = the
// push refused; the floor holds a short push to a gentle move; a pull takes |d|.
TEST(PushPull, PushTimeComesFromTheMeanSpeed)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = config(*v);
  c.push_distance = 0.10;
  c.push_speed = 0.017;
  c.push_time = 99.0;   // a turn's: must not matter
  EXPECT_NEAR(pushPullPushTime(c), 0.10 / 0.017, 1e-12);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp);
  EXPECT_NEAR(traj->duration(), c.push_settle + 0.10 / 0.017, 1e-6);
  EXPECT_LT((traj->goalRest().x_b - wp.pushed.x_b).norm(), 1e-9);
  // the floor
  c.push_min_time = 8.0;
  EXPECT_NEAR(pushPullPushTime(c), 8.0, 1e-12);
  EXPECT_NEAR(planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp)->duration(),
    c.push_settle + 8.0, 1e-6);
  c.push_min_time = 0.0;
  // a pull: the same time for |d|
  c.push_distance = -0.10;
  EXPECT_NEAR(pushPullPushTime(c), 0.10 / 0.017, 1e-12);
  // not set: refused, with the key named
  c.push_distance = 0.10;
  for (const double bad : {0.0, -0.01, std::nan("")}) {
    c.push_speed = bad;
    try {
      planPushPullPush(*v, *planner, PlanOptions{}, c, wp.grasp);
      FAIL() << "a push with speed " << bad << " was planned";
    } catch (const std::runtime_error & e) {
      EXPECT_NE(std::string(e.what()).find("push_pull_push_mean_speed"), std::string::npos) << e.what();
    }
  }
  // (2026-10-21 review) slower than T_max: refused, never silently flown at T_max
  PlanOptions o;
  o.T_max = 40.0;
  c.push_distance = 0.40;
  c.push_speed = 0.008;   // 50 s
  try {
    planPushPullPush(*v, *planner, o, c, wp.grasp);
    FAIL() << "a 50 s push was planned against T_max 40 s";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("t_max"), std::string::npos) << e.what();
  }
  c.push_speed = 0.0105;  // 38.1 s: inside, flown at its own time
  EXPECT_NEAR(planPushPullPush(*v, *planner, o, c, wp.grasp)->duration(),
    c.push_settle + 0.40 / 0.0105, 1e-6);
}

TEST(PushPull, BoxSampleGateDropsTheMisSolvesOnly)
{
  // (2026-10-21) the drone mocap guard's jump test on the box, with the 10-21 hardware numbers:
  // 20 mm / 10 deg from the LAST GOOD sample, growing +1 m/s, +2 m/s^2, +90 deg/s over the gap
  const double jm = 0.020, jd = 10.0, dt = 1.0 / 120.0;
  // the mis-solve (a 94-96 deg roll, the origin 61 mm off) and the 180 deg flip: dropped
  EXPECT_FALSE(boxSampleWithinGate(0.061, 95.0, dt, jm, jd));
  EXPECT_FALSE(boxSampleWithinGate(0.179, 178.8, dt, jm, jd));
  // ... and still dropped after the longest run seen, 25 samples (199 ms): the angle holds it
  EXPECT_FALSE(boxSampleWithinGate(0.061, 94.8, 0.199, jm, jd));
  // the largest real steps: the push (6.7 mm / 4.1 deg), a hand carry at 1.9 m/s (17.3 mm)
  EXPECT_TRUE(boxSampleWithinGate(0.0067, 4.06, dt, jm, jd));
  EXPECT_TRUE(boxSampleWithinGate(0.0173, 3.66, 0.0091, jm, jd));
  // a pose that really moved is let back in as the allowance grows: 61 mm after 41 ms, a
  // 95 deg turn after (95 - 10) / 90 = 0.94 s
  EXPECT_FALSE(boxSampleWithinGate(0.061, 0.0, 0.035, jm, jd));
  EXPECT_TRUE(boxSampleWithinGate(0.061, 0.0, 0.041, jm, jd));
  EXPECT_FALSE(boxSampleWithinGate(0.0, 95.0, 0.93, jm, jd));
  EXPECT_TRUE(boxSampleWithinGate(0.0, 95.0, 0.95, jm, jd));
  // NaN is never good; each limit can be off alone; both off = every sample, NaN included
  EXPECT_FALSE(boxSampleWithinGate(std::nan(""), 0.0, dt, jm, jd));
  EXPECT_FALSE(boxSampleWithinGate(0.0, std::nan(""), dt, jm, jd));
  EXPECT_TRUE(boxSampleWithinGate(0.061, 0.0, dt, 0.0, jd));
  EXPECT_FALSE(boxSampleWithinGate(0.061, 95.0, dt, 0.0, jd));
  EXPECT_TRUE(boxSampleWithinGate(0.0, 95.0, dt, jm, 0.0));
  EXPECT_TRUE(boxSampleWithinGate(1.0, 180.0, dt, 0.0, 0.0));
  EXPECT_TRUE(boxSampleWithinGate(std::nan(""), std::nan(""), dt, 0.0, 0.0));
}

// THE TABLE PLACED FROM THE BOX and GO TO START ROUND IT (2026-10-22, user: the lab
// table goes where the floor is flat; the box always sits at its left end, its back
// face 0.10 m in; the vehicle starts on the table's middle line, in front of it). The
// box-pushing yaml of 2026-10-22: the HOOBRO table (1.000 x 0.150, top 0.80), the
// CAD box 240 x 160 x 100 mm centred at (0.675, 0.28), the fin's grasp 0.183 m over
// its centre, push pose [0, 32, 28, 0], margins 0.15 up / 0.12 back, T650 gear.
namespace
{
PushPullConfig layoutConfig(const VehicleModel & v)
{
  PushPullConfig c;
  c.home = v.home;
  c.push_pose << 0.0, 32.0 * kDeg, 28.0 * kDeg, 0.0;
  c.ee_offset << 0.0, 0.0, 0.183;
  c.approach_dz = 0.15;
  c.approach_back = 0.12;
  c.exit_dz = 0.25;
  c.push_distance = 0.10;
  c.push_speed = 0.021;
  c.push_min_time = 3.0;
  c.push_settle = 1.5;
  c.gear_depth = 0.2425;
  c.fence.min_z = 0.25;
  c.fence.max_z = 2.2;
  c.box_half_length = 0.12;
  c.table_from_box = true;
  c.table_length = 1.0;
  c.table_depth = 0.15;
  c.table_box_gap = 0.10;
  c.box_half_height = 0.05;
  c.start_middle_tol = 0.30;
  c.bypass = true;
  c.bypass_clearance = 0.70;
  return c;
}

PushPullTargets layoutScene(double box_yaw = 0.0)
{
  PushPullTargets t;
  t.box << 0.675, 0.28, 0.85;
  t.box_yaw = box_yaw;
  t.start.p << -0.50, -0.07, 1.0;
  t.start_set = true;
  t.land.p << -0.50, -0.07, 0.30;
  t.land.yaw = -360.0 * kDeg;
  t.land_start.p << -0.50, -0.07, 1.30;
  t.land_start.yaw = -360.0 * kDeg;
  t.land_start_set = true;
  return t;
}

// the takeoff hover over the start, facing the table (actual yaw 0), arm home
RestSpec takeoffHover(const VehicleModel & v)
{
  RestSpec r;
  r.x_b << -0.50, -0.07, 1.0;
  r.phi = -0.5 * M_PI;
  r.q = v.home;
  return r;
}
}  // namespace

TEST(PushPullLayout, TheTableIsPlacedUnderTheBox)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PushPullConfig c = layoutConfig(*v);
  // box yaw 0, the push along -y: the Isaac table exactly (x 0.600..0.750, y -0.5..0.5)
  const TableKeepOut tb = tableFromBox(c, layoutScene().box, 0.0);
  const std::array<Vec3, 4> corners = {
    tb.toWorld(Vec3(tb.x_min, tb.y_min, 0.0)), tb.toWorld(Vec3(tb.x_min, tb.y_max, 0.0)),
    tb.toWorld(Vec3(tb.x_max, tb.y_min, 0.0)), tb.toWorld(Vec3(tb.x_max, tb.y_max, 0.0))};
  double x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  for (const Vec3 & p : corners) {
    x0 = std::min(x0, p(0));
    x1 = std::max(x1, p(0));
    y0 = std::min(y0, p(1));
    y1 = std::max(y1, p(1));
  }
  EXPECT_NEAR(x0, 0.600, 1e-12);
  EXPECT_NEAR(x1, 0.750, 1e-12);
  EXPECT_NEAR(y0, -0.50, 1e-12);
  EXPECT_NEAR(y1, 0.50, 1e-12);
  EXPECT_NEAR(tb.z_top, 0.80, 1e-12);
  // the box end is the +y end, 0.22 m behind the box centre along the push
  EXPECT_NEAR(tb.distanceXY(Vec3(0.675, 0.70, 0.0)), 0.20, 1e-12);
  EXPECT_NEAR(tb.distanceXY(Vec3(0.675, -0.70, 0.0)), 0.20, 1e-12);
  EXPECT_NEAR(tb.distanceXY(Vec3(-0.50, 0.0, 0.0)), 1.10, 1e-12);
  EXPECT_EQ(tb.distanceXY(Vec3(0.70, 0.0, 0.0)), 0.0);
  // round trip, and a table turned with a box at 30 deg
  const Vec3 p(0.3, -0.2, 1.1);
  EXPECT_LT((tb.toWorld(tb.toLocal(p)) - p).norm(), 1e-12);
  const TableKeepOut t30 = tableFromBox(c, Vec3(1.0, 2.0, 0.85), 30.0 * kDeg);
  const Vec3 end = Vec3(1.0, 2.0, 0.0) - 0.22 * Vec3(std::cos(-60.0 * kDeg), std::sin(-60.0 * kDeg), 0.0);
  EXPECT_NEAR(t30.distanceXY(end), 0.0, 1e-12);
  EXPECT_NEAR(t30.distanceXY(end - 0.10 * Vec3(std::cos(-60.0 * kDeg), std::sin(-60.0 * kDeg), 0.0)),
    0.10, 1e-12);
  // the waypoints carry it, and the box may slide at most 1.0 - 0.10 - 0.24 - 0.03 = 0.63 m
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, layoutScene());
  EXPECT_NEAR(wp.table.x_min, tb.x_min, 1e-12);
  EXPECT_NEAR(wp.table.yaw, tb.yaw, 1e-12);
  PushPullConfig longer = c;
  longer.push_distance = 0.62;
  EXPECT_NO_THROW(pushPullWaypoints(*v, longer, layoutScene()));
  longer.push_distance = 0.64;
  EXPECT_THROW(pushPullWaypoints(*v, longer, layoutScene()), std::runtime_error);
}

TEST(PushPullLayout, PlanRefusesAStartThatDoesNotFitTheTable)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PushPullConfig c = layoutConfig(*v);
  EXPECT_NO_THROW(pushPullWaypoints(*v, c, layoutScene()));
  const auto refused = [&](const PushPullTargets & t, const std::string & what) {
      try {
        pushPullWaypoints(*v, c, t);
      } catch (const std::exception & e) {
        EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
        std::cout << "refused: " << e.what() << "\n";
        return;
      }
      ADD_FAILURE() << "accepted: " << what;
    };
  // 0.35 m off the table's middle line (y 0 with the box at the +y end)
  PushPullTargets off = layoutScene();
  off.start.p(1) = 0.35;
  refused(off, "middle line");
  // the box (or its mocap body) read 180 deg off: the table would lie the other way
  refused(layoutScene(M_PI), "middle line");
  // 0.55 m from the table's front: the way round keeps 0.70 + 0.05
  PushPullTargets close = layoutScene();
  close.start.p(0) = 0.600 - 0.55;
  refused(close, "from the table's side");
}

TEST(PushPullLayout, LandingHeadingsTurnWithTheBox)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = layoutConfig(*v);
  for (const double yaw_deg : {0.0, 30.0, -100.0, 170.0}) {
    const PushPullTargets t = layoutScene(yaw_deg * kDeg);
    // the start on the middle line of the table this box places, 1.2 m in front
    const TableKeepOut tb = tableFromBox(c, t.box, t.box_yaw);
    PushPullTargets t2 = t;
    t2.start.p = tb.toWorld(Vec3(0.5 * (tb.x_min + tb.x_max), -1.275, 1.0));
    t2.land.p = Vec3(t2.start.p(0), t2.start.p(1), 0.30);
    t2.land_start.p = Vec3(t2.start.p(0), t2.start.p(1), 1.30);
    const PushPullWaypoints wp = pushPullWaypoints(*v, c, t2);
    // land facing the table as typed from the push heading: -360 - (-90) = 270 deg clockwise
    const double turn = wp.land_start.phi + 0.5 * M_PI - wp.drone_yaw;
    EXPECT_NEAR(turn, -270.0 * kDeg, 1e-9) << yaw_deg;
    const double facing = std::atan2(std::sin(wp.land.phi + 0.5 * M_PI - t.box_yaw),
        std::cos(wp.land.phi + 0.5 * M_PI - t.box_yaw));
    EXPECT_NEAR(facing, 0.0, 1e-9) << yaw_deg;
    const auto home = planPushPullGoToLandStart(*v, *planner, PlanOptions{}, c, wp, wp.exited);
    double unwrapped = 0.0, prev = headingOf(home->eval(0.0));
    for (int k = 1; k <= 400; ++k) {
      const double h = headingOf(home->eval(home->duration() * k / 400.0));
      unwrapped += std::atan2(std::sin(h - prev), std::cos(h - prev));
      prev = h;
    }
    EXPECT_NEAR(unwrapped, -270.0 * kDeg, 1e-3) << yaw_deg;
  }
}

TEST(PushPullLayout, GoToStartGoesRoundTheTablesBoxEnd)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = layoutConfig(*v);
  for (const double front : {1.10, 0.76, 2.0}) {
    for (const double along : {0.0, 0.29, -0.29}) {
      PushPullTargets t = layoutScene();
      const TableKeepOut tb = tableFromBox(c, t.box, t.box_yaw);
      // the start on (or off by `along`) the middle line, `front` m from the table's side
      t.start.p = tb.toWorld(Vec3(0.5 * (tb.x_min + tb.x_max) + along, tb.y_min - front, 1.0));
      const PushPullWaypoints wp = pushPullWaypoints(*v, c, t);
      c.table = wp.table;
      RestSpec from = takeoffHover(*v);
      from.x_b = t.start.p;
      const auto traj = planPushPullGoToStart(*v, *planner, PlanOptions{}, c, wp, from);
      const double T = traj->duration();
      const auto * seq = dynamic_cast<const SequenceTrajectory *>(traj.get());
      ASSERT_NE(seq, nullptr) << traj->diag().summary;
      ASSERT_EQ(seq->segmentStarts().size(), 2u) << traj->diag().summary;
      const double t_in = seq->segmentStarts()[1];
      EXPECT_LT((traj->goalRest().x_b - wp.approach.x_b).norm(), 1e-9);
      for (const double te : {0.0, t_in, T}) {
        const WbReference r = traj->eval(te);
        EXPECT_LT(r.x_cd_dot.norm() + r.x_cd_ddot.norm() + r.x_cd_d3.norm(), 1e-6) << te;
      }
      // the curve: the body keeps 0.70 m from the table seen from above, the arm home
      double closest = 1e9;
      for (int k = 0; k <= 2000; ++k) {
        const WbReference r = traj->eval(t_in * k / 2000.0);
        closest = std::min(closest, wp.table.distanceXY(bodyOf(*v, r)));
        EXPECT_LT((r.q_d - v->home).cwiseAbs().maxCoeff(), 1e-12);
      }
      EXPECT_GT(closest, c.bypass_clearance - 5e-3) << front << " " << along;
      // in: along the push line, the heading already the push's
      const WbReference e = traj->eval(t_in);
      const Vec3 entry = bodyOf(*v, e);
      EXPECT_NEAR(wp.table.distanceXY(entry), c.bypass_clearance, 1e-6);
      EXPECT_NEAR(std::cos(headingOf(e) - wp.approach.phi), 1.0, 1e-9);
      const Vec3 lat = (entry - wp.approach.x_b) - (entry - wp.approach.x_b).dot(wp.push_dir) * wp.push_dir;
      EXPECT_LT(Vec3(lat(0), lat(1), 0.0).norm(), 1e-9);
      // the derivative chain of the curve is consistent with its own position,
      // through snap and the heading's rate (FD error relative to each peak)
      double err[5] = {0, 0, 0, 0, 0}, peak[5] = {0, 0, 0, 0, 0};
      const double h = 1e-4;
      for (int k = 1; k < 400; ++k) {
        const double tt = t_in * k / 400.0;
        const WbReference a = traj->eval(tt - h), b = traj->eval(tt), d = traj->eval(tt + h);
        const Vec3 fd[5] = {(d.x_cd - a.x_cd) / (2 * h), (d.x_cd_dot - a.x_cd_dot) / (2 * h),
          (d.x_cd_ddot - a.x_cd_ddot) / (2 * h), (d.x_cd_d3 - a.x_cd_d3) / (2 * h),
          (d.b1_d - a.b1_d) / (2 * h)};
        const Vec3 an[5] = {b.x_cd_dot, b.x_cd_ddot, b.x_cd_d3, b.x_cd_d4, b.b1_d_dot};
        for (int i = 0; i < 5; ++i) {
          err[i] = std::max(err[i], (fd[i] - an[i]).norm());
          peak[i] = std::max(peak[i], an[i].norm());
        }
      }
      for (int i = 0; i < 5; ++i) {
        EXPECT_LT(err[i], 1e-3 * peak[i]) << "derivative " << i + 1 << ": FD " << err[i]
                                          << " vs peak " << peak[i];
      }
      EXPECT_LT(T, 30.0);
      std::cout << "go_to_start, " << front << " m in front, " << along << " m along: " << T
                << " s, the curve " << t_in << " s, " << closest << " m from the table -- "
                << traj->diag().summary << "\n";
    }
  }
}

TEST(PushPullLayout, GoToStartNearTheTableFliesStraightAndPastItsFarEndRefuses)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = layoutConfig(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, layoutScene());
  c.table = wp.table;
  // a re-fly from just above the handle (an abort's climb): the one transition
  RestSpec near = wp.approach;
  near.x_b(2) += 0.30;
  near.q = v->home;
  const auto straight = planPushPullGoToStart(*v, *planner, PlanOptions{}, c, wp, near);
  EXPECT_EQ(straight->diag().summary.rfind("(straight:", 0), 0u) << straight->diag().summary;
  // past the table's FAR end, on its line: no way round its box end
  RestSpec far = takeoffHover(*v);
  far.x_b = wp.table.toWorld(Vec3(wp.table.x_max + 0.8, 0.0, 1.0));
  EXPECT_THROW(planPushPullGoToStart(*v, *planner, PlanOptions{}, c, wp, far), std::runtime_error);
  // past the BOX end already: straight to the entry, then in
  RestSpec past = takeoffHover(*v);
  past.x_b = wp.table.toWorld(Vec3(wp.table.x_min - 1.2, 0.4, 1.0));
  const auto in = planPushPullGoToStart(*v, *planner, PlanOptions{}, c, wp, past);
  EXPECT_LT((in->goalRest().x_b - wp.approach.x_b).norm(), 1e-9);
}

TEST(PushPullLayout, TheMissionDryRunsRoundTheTable)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = layoutConfig(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, layoutScene());
  const auto steps = planPushPullMission(*v, *planner, PlanOptions{}, c, wp, takeoffHover(*v));
  ASSERT_EQ(steps.size(), static_cast<size_t>(kNumPushPullSteps));
  for (size_t k = 0; k < steps.size(); ++k) {
    std::cout << pushPullStepName(static_cast<int>(k)) << ": " << steps[k]->duration() << " s -- "
              << steps[k]->diag().summary << "\n";
  }
  // without the bypass the same mission flies straight to the approach and home;
  // the steps at the box are the same
  PushPullConfig direct = c;
  direct.bypass = false;
  const auto d = planPushPullMission(*v, *planner, PlanOptions{}, direct, wp, takeoffHover(*v));
  EXPECT_GT(steps[0]->duration(), d[0]->duration());
  for (const int k : {kPlReady, kPlPush, kPlExit, kPlLand}) {
    EXPECT_NEAR(steps[k]->duration(), d[k]->duration(), 1e-9) << k;
  }
  EXPECT_NE(steps[kPlGoToLandStart]->diag().summary.find("round the table"), std::string::npos);
}

// THE WAY HOME ROUND THE TABLE (2026-10-22, user): Go To Land Start flies on
// along the push's line past the table's FAR end (at the exit height, over the
// pushed box), then curves round that corner of the table to Land Start; landing
// at yaw -180 is a 90 deg clockwise turn from the push heading -90 (no full turn).
TEST(PushPullLayout, GoToLandStartGoesRoundTheTableToo)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PushPullConfig c = layoutConfig(*v);
  PushPullTargets t = layoutScene();
  t.land.yaw = -180.0 * kDeg;
  t.land_start.yaw = -180.0 * kDeg;
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, t);
  c.table = wp.table;
  const auto steps = planPushPullMission(*v, *planner, PlanOptions{}, c, wp, takeoffHover(*v));
  const auto & home = steps[kPlGoToLandStart];
  const double T = home->duration();
  EXPECT_LT((home->goalRest().x_b - wp.land_start.x_b).norm(), 1e-9);
  const RestSpec exited = steps[kPlExit]->goalRest();
  double unwrapped = 0.0, prev = headingOf(home->eval(0.0)), low_near = 1e9, closest_low = 1e9;
  for (int k = 0; k <= 2000; ++k) {
    const WbReference r = home->eval(T * k / 2000.0);
    const double h = headingOf(r);
    const double d = std::atan2(std::sin(h - prev), std::cos(h - prev));
    EXPECT_LE(d, 1e-9) << "turned counter-clockwise";
    unwrapped += d;
    prev = h;
    const Vec3 body = bodyOf(*v, r);
    const double dist = wp.table.distanceXY(body);
    // near the table only at the exit height (backing out over its end)
    if (dist < c.bypass_clearance - 5e-3) {low_near = std::min(low_near, body(2) - exited.x_b(2));}
    if (body(2) < exited.x_b(2) - 1e-3) {closest_low = std::min(closest_low, dist);}
  }
  EXPECT_NEAR(unwrapped, -90.0 * kDeg, 1e-3);
  EXPECT_GT(low_near, -1e-3);
  // round the FAR end: the farthest it goes along the push is R past the table's far end
  double reach = -1e9;
  for (int k = 0; k <= 400; ++k) {
    const Vec3 l = wp.table.toLocal(bodyOf(*v, home->eval(T * k / 400.0)));
    reach = std::max(reach, l(0));
  }
  EXPECT_NEAR(reach, wp.table.x_max + c.bypass_clearance, 2e-3);
  // too low a handle clearance is refused (the pushed box as a keep-out): a fin
  // 0.40 m above the grasp would stand above the gear's path
  PushPullConfig tall = c;
  tall.handle_above_grasp = 0.40;
  EXPECT_THROW(planPushPullGoToLandStart(*v, *planner, PlanOptions{}, tall, wp, exited), std::runtime_error);
  EXPECT_GT(closest_low, c.bypass_clearance - 5e-3);
  const double land_yaw = wp.land.phi + 0.5 * M_PI;
  EXPECT_NEAR(std::cos(land_yaw - M_PI), 1.0, 1e-12);
  std::cout << "go_to_land_start: " << T << " s -- " << home->diag().summary << "\n";
  // land: straight down, heading kept
  EXPECT_NEAR(std::cos(headingOf(steps[kPlLand]->eval(steps[kPlLand]->duration())) - wp.land.phi), 1.0, 1e-9);
}
