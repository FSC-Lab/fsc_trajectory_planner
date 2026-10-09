// The push-and-pull mode on the T650 aerial manipulator (push_pull.hpp): the
// waypoints (the claw on the box's handle, the nose along the push, the gear
// clear of the table), each of the five steps (go_to_start, the vertical
// ready descent, the push with the arm and heading held, the exit climbing
// before it folds, the land turn flown as typed) and the table check.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/push_pull.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"

using namespace fsc_trajectory_planner;

namespace
{
constexpr double kDeg = M_PI / 180.0;

// The Isaac scene (07_px4_direct_t650_aerial_manipulator_push_and_pull.py): a
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

  EXPECT_GE(traj->duration(), c.push_settle + c.push_time - 1e-9);
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

TEST(PushPull, GoToLandTurnsTheTypedWayRound)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PushPullConfig c = config(*v);
  const PushPullWaypoints wp = pushPullWaypoints(*v, c, scene());
  const auto traj = planPushPullGoToLand(*v, *planner, PlanOptions{}, c, wp, wp.exited);
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
  const auto again = planPushPullGoToLand(*v, *planner, PlanOptions{}, c, wp, at_land);
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
  EXPECT_LT((steps[kPlGoToLand]->goalRest().x_b - wp.land.x_b).norm(), 1e-9);
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
