// The pick-and-place mode on the T650 aerial manipulator: the six goal
// rests (claw on the measured point, nose facing it, the calibrated base
// poses), the retreat, the approach above a claw target and the vertical
// descent onto it, the geofence, the arm-sweep leg (compatible, at rest at
// both ends, sweeping its band, every bound honoured, the shortest feasible
// duration) and the whole mission dry run the Plan button performs.
#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "fsc_trajectory_planner/arm_sweep_planner.hpp"
#include "fsc_trajectory_planner/pick_place.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"
#include "fsc_trajectory_planner/workspace.hpp"

using namespace fsc_trajectory_planner;

namespace
{
constexpr double kDeg = M_PI / 180.0;

// A lab-sized scene: two tables 0.7 m high, start / place-start / land-start
// / land on the other side of the room; the land hover at 0.8 m (above the
// 0.6 m fence floor).
PickPlaceTargets scene()
{
  PickPlaceTargets t;
  t.start.p << 0.0, 0.0, 1.0;
  t.start.yaw = 0.0;
  t.pick << 1.0, 0.6, 0.70;
  t.place_start.p << 1.0, -0.6, 1.1;
  t.place_start.yaw = 0.0;
  t.place << 1.6, -1.0, 0.70;
  t.land_start.p << 0.0, -1.2, 1.0;
  t.land_start.yaw = 0.0;
  t.land.p << 0.0, -1.2, 0.8;
  t.land.yaw = 0.0;
  return t;
}

PickPlaceConfig config(const VehicleModel & v)
{
  PickPlaceConfig c;
  c.home = v.home;
  c.carry = v.home;
  return c;
}

double nearestAngle(double a) {return std::atan2(std::sin(a), std::cos(a));}
}  // namespace

TEST(PickPlace, WaypointsPutTheClawOnTheTargetFacingIt)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PickPlaceTargets t = scene();
  PickPlaceConfig c = config(*v);
  c.pick_ee_offset << 0.0, 0.0, 0.03;
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, t);

  // base legs: exactly the given pose, ACTUAL yaw -> model heading
  EXPECT_LT((wp.goal[kGoToStart].x_b - t.start.p).norm(), 1e-12);
  EXPECT_NEAR(wp.goal[kGoToStart].phi, t.start.yaw - 0.5 * M_PI, 1e-12);
  EXPECT_LT((wp.goal[kGoToStart].q - v->home).norm(), 1e-12);
  EXPECT_LT((wp.goal[kExecuteLand].x_b - t.land.p).norm(), 1e-12);
  EXPECT_LT((wp.goal[kGoToPlaceStart].q - c.carry).norm(), 1e-12);

  // claw legs: the grasp point ON the measured point + offset, arm in its pose
  EXPECT_LT((wp.ee[kExecutePick] - (t.pick + c.pick_ee_offset)).norm(), 1e-9);
  EXPECT_LT((wp.ee[kExecutePlace] - t.place).norm(), 1e-9);
  EXPECT_LT((wp.goal[kExecutePick].q - c.pick_pose).norm(), 1e-12);
  // the pick pose: claw straight down, 0.34 m under the body origin
  EXPECT_NEAR(wp.goal[kExecutePick].x_b(2) - wp.ee[kExecutePick](2), 0.340, 2e-3);
  // facing it: the nose (model +y) points along the bearing from the
  // previous leg's base to the claw target, to within the arm's 2 mm
  // lateral offset
  for (int leg : {kExecutePick, kExecutePlace}) {
    const Vec3 from = wp.goal[leg - 1].x_b;
    const Vec3 d = wp.ee[leg] - from;
    const double bearing = std::atan2(d(1), d(0));
    const double nose = wp.goal[leg].phi + 0.5 * M_PI;
    EXPECT_LT(std::abs(nearestAngle(nose - bearing)), 1.0 * kDeg) << pickPlaceLegName(leg);
    // and the base sits between the previous base and the target
    const Vec3 b = wp.goal[leg].x_b - from;
    EXPECT_GT(b.head<2>().dot(d.head<2>()), 0.0);
    EXPECT_LT(b.head<2>().norm(), d.head<2>().norm());
  }
}

TEST(PickPlace, AnImpossibleArmPoseIsRefusedNamingTheLeg)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  PickPlaceConfig c = config(*v);
  c.pick_pose << 0.0, 0.0, 60.0 * kDeg, 0.0;   // q3 stops at +50 deg
  try {
    pickPlaceWaypoints(*v, c, scene());
    FAIL() << "a pick pose outside the joint box was accepted";
  } catch (const std::runtime_error & e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("execute_pick"), std::string::npos) << m;
    EXPECT_NE(m.find("q3"), std::string::npos) << m;
  }
}

TEST(PickPlace, RetreatClimbsAndBacksOffAlongTheNose)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  PickPlaceConfig c = config(*v);
  RestSpec from;
  from.x_b << 1.0, 2.0, 1.0;
  from.phi = 0.3 - 0.5 * M_PI;   // nose at actual yaw 0.3
  from.q = c.pick_pose;
  const RestSpec r = retreatRest(from, c);
  const Vec3 nose{std::cos(0.3), std::sin(0.3), 0.0};
  const Vec3 d = r.x_b - from.x_b;
  EXPECT_NEAR(d(2), c.retreat_dz, 1e-12);
  EXPECT_NEAR(d.dot(nose), -c.retreat_back, 1e-12);
  EXPECT_NEAR(d.head<2>().norm(), c.retreat_back, 1e-12);
  EXPECT_EQ(r.phi, from.phi);
  EXPECT_LT((r.q - from.q).norm(), 1e-15);
}

TEST(ArmSweep, LegIsCompatibleAtRestAndSweepsItsBand)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PickPlaceConfig c = config(*v);
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, scene());
  const RestSpec & r0 = wp.goal[kGoToPlaceStart];
  const RestSpec & r1 = wp.goal[kExecutePlace];
  ArmSweepDiag d;
  const auto traj = ArmSweepPlanner::plan(*v, r0, r1, c.sweep, &d);
  std::cout << "  " << d.summary << "\n";
  ASSERT_TRUE(d.feasible) << d.violation;
  const double T = traj->duration();

  // at rest on both rests
  const WbReference want0 = restReference(v->params, r0), want1 = restReference(v->params, r1);
  const WbReference a = traj->eval(0.0), b = traj->eval(T);
  EXPECT_LT((a.x_cd - want0.x_cd).norm(), 1e-9);
  EXPECT_LT((b.x_cd - want1.x_cd).norm(), 1e-9);
  EXPECT_LT((a.r_ed - want0.r_ed).norm(), 1e-9);
  EXPECT_LT((b.r_ed - want1.r_ed).norm(), 1e-9);
  EXPECT_LT((a.q_d - r0.q).cwiseAbs().maxCoeff(), 1e-12);
  EXPECT_LT((b.q_d - r1.q).cwiseAbs().maxCoeff(), 1e-12);
  for (const WbReference * e : {&a, &b}) {
    EXPECT_LT(e->x_cd_dot.norm() + e->x_cd_ddot.norm() + e->x_cd_d3.norm() + e->x_cd_d4.norm(), 1e-9);
    EXPECT_LT(e->qdot_d.cwiseAbs().maxCoeff(), 1e-9);
    EXPECT_LT(e->r_ed_dot.norm(), 1e-9);
  }

  // the analytic derivatives are the derivatives (central differences)
  const double h = 1e-4;
  double e_v = 0.0, e_a = 0.0, e_j = 0.0, e_s = 0.0, e_q = 0.0, e_b = 0.0, e_re = 0.0;
  for (int k = 1; k < 50; ++k) {
    const double t = T * k / 50.0;
    const WbReference m = traj->eval(t - h), z = traj->eval(t), p = traj->eval(t + h);
    e_v = std::max(e_v, ((p.x_cd - m.x_cd) / (2 * h) - z.x_cd_dot).norm());
    e_a = std::max(e_a, ((p.x_cd_dot - m.x_cd_dot) / (2 * h) - z.x_cd_ddot).norm());
    e_j = std::max(e_j, ((p.x_cd_ddot - m.x_cd_ddot) / (2 * h) - z.x_cd_d3).norm());
    e_s = std::max(e_s, ((p.x_cd_d3 - m.x_cd_d3) / (2 * h) - z.x_cd_d4).norm());
    e_q = std::max(e_q, ((p.q_d - m.q_d) / (2 * h) - z.qdot_d).cwiseAbs().maxCoeff());
    e_b = std::max(e_b, ((p.b1_d - m.b1_d) / (2 * h) - z.b1_d_dot).norm());
    e_re = std::max(e_re, ((p.r_ed - m.r_ed) / (2 * h) - z.r_ed_dot).norm());
  }
  EXPECT_LT(e_v, 1e-6);
  EXPECT_LT(e_a, 1e-6);
  EXPECT_LT(e_j, 1e-6);
  EXPECT_LT(e_s, 1e-5);
  EXPECT_LT(e_q, 1e-6);
  EXPECT_LT(e_b, 1e-6);
  EXPECT_LT(e_re, 1e-6);

  // the swept joints visit both ends of their bands on the plateau (the
  // window's ramps blend in and out of the start / goal angles)
  VecN qlo = VecN::Constant(1e9), qhi = VecN::Constant(-1e9);
  const double r = c.sweep.ramp_frac * T;
  for (int k = 0; k <= 2000; ++k) {
    const VecN q = traj->eval(r + (T - 2.0 * r) * k / 2000.0).q_d;
    qlo = qlo.cwiseMin(q);
    qhi = qhi.cwiseMax(q);
  }
  for (int j : {0, 1}) {
    EXPECT_NEAR(qlo(j), c.sweep.lo(j), 1.0 * kDeg) << "q" << j + 1;
    EXPECT_NEAR(qhi(j), c.sweep.hi(j), 1.0 * kDeg) << "q" << j + 1;
  }
  // every bound honoured
  EXPECT_LE(d.peak_v, c.sweep.v_max + 1e-9);
  EXPECT_LE(d.peak_a, c.sweep.a_max + 1e-9);
  EXPECT_LE(d.peak_w, c.sweep.w_max + 1e-9);
  EXPECT_LE(d.peak_qdot, c.sweep.qdot_max + 1e-9);
  EXPECT_LE(d.peak_tau_joint, c.sweep.tau_joint_max + 1e-9);
  EXPECT_GE(d.min_sigma_nd, c.sweep.sigma_nd_min);
  EXPECT_GE(d.rotor_lo, v->rotor.f_min);
  EXPECT_LE(d.rotor_hi, v->rotor.f_max);
  // and the duration is (to rel_tol) the shortest that passes
  if (T > c.sweep.T_min * (1.0 + 1e-9)) {
    EXPECT_THROW(ArmSweepPlanner::planAt(*v, r0, r1, c.sweep, T / (1.0 + 2.0 * c.sweep.rel_tol)),
      std::runtime_error);
  }
}

TEST(ArmSweep, DefaultBandIsTenDegreesEachSide)
{
  const ArmSweepOptions o;
  EXPECT_NEAR(o.lo(0), -10.0 * kDeg, 1e-12);
  EXPECT_NEAR(o.hi(0), 10.0 * kDeg, 1e-12);
  EXPECT_NEAR(o.hi(1) - o.lo(1), 20.0 * kDeg, 1e-12);
  EXPECT_EQ(o.lo(2), o.hi(2));
  EXPECT_EQ(o.lo(3), o.hi(3));
  EXPECT_EQ(o.cycles, 2);
}

TEST(ArmSweep, BandOutsideTheJointBoxIsRefused)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PickPlaceConfig c0 = config(*v);
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c0, scene());
  ArmSweepOptions o;
  o.hi(1) = 60.0 * kDeg;   // q2 stops at +50 deg
  try {
    ArmSweepPlanner::plan(*v, wp.goal[kGoToPlaceStart], wp.goal[kExecutePlace], o);
    FAIL() << "a sweep band outside the joint box was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("q2"), std::string::npos) << e.what();
  }
}

TEST(PickPlace, MissionDryRunPlansAllSixLegsChained)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = config(*v);
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, scene());
  RestSpec hold;   // wherever DIRECT was engaged
  hold.x_b << 0.3, 0.2, 1.0;
  hold.phi = 0.4;
  hold.q = v->home;
  PlanOptions opts;
  const auto legs = planPickPlaceMission(*v, *planner, opts, c, wp, hold);
  ASSERT_EQ(static_cast<int>(legs.size()), static_cast<int>(kNumPickPlaceLegs));
  double total = 0.0;
  RestSpec from = hold;
  for (int k = 0; k < kNumPickPlaceLegs; ++k) {
    const Trajectory & L = *legs[k];
    std::cout << "  " << pickPlaceLegName(k) << ": T " << L.duration() << " s -- "
              << L.diag().summary << "\n";
    total += L.duration();
    const WbReference s = L.eval(0.0), e = L.eval(L.duration());
    EXPECT_LT((s.x_cd - restReference(v->params, from).x_cd).norm(), 1e-6) << k;
    EXPECT_LT((e.x_cd - restReference(v->params, wp.goal[k]).x_cd).norm(), 1e-6) << k;
    EXPECT_LT((e.q_d - wp.goal[k].q).cwiseAbs().maxCoeff(), 1e-6) << k;
    EXPECT_LT((e.r_ed - wp.ee[k]).norm(), 1e-6) << k;
    if (pickPlaceLegIsClaw(k)) {
      // approach, at rest approach_dz above the goal, then the descent
      const auto * seq = dynamic_cast<const SequenceTrajectory *>(&L);
      ASSERT_NE(seq, nullptr) << pickPlaceLegName(k);
      ASSERT_EQ(seq->segmentStarts().size(), 2u);
      const WbReference via = L.eval(seq->segmentStarts()[1]);
      EXPECT_LT((via.x_cd - restReference(v->params, wp.approach[k]).x_cd).norm(), 1e-6);
      EXPECT_LT((via.r_ed - wp.approach_ee[k]).norm(), 1e-6);
      EXPECT_LT(via.x_cd_dot.norm(), 1e-6);
    }
    if (pickPlaceLegRetreats(k)) {
      // the leg passes through the retreat point, at rest, before transiting
      const auto * seq = dynamic_cast<const SequenceTrajectory *>(&L);
      ASSERT_NE(seq, nullptr) << pickPlaceLegName(k);
      ASSERT_EQ(seq->segmentStarts().size(), 2u);
      const WbReference via = L.eval(seq->segmentStarts()[1]);
      EXPECT_LT((via.x_cd - restReference(v->params, retreatRest(from, c)).x_cd).norm(), 1e-6);
      EXPECT_LT(via.x_cd_dot.norm(), 1e-6);
    }
    from = wp.goal[k];
  }
  std::cout << "  mission total " << total << " s\n";
}

TEST(PickPlace, ClawLegsStopAboveTheTargetThenDescendVertically)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = config(*v);
  ASSERT_GT(c.approach_dz, 0.0);
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, scene());
  for (int k = 0; k < kNumPickPlaceLegs; ++k) {
    const Vec3 up = wp.approach[k].x_b - wp.goal[k].x_b;
    const Vec3 up_ee = wp.approach_ee[k] - wp.ee[k];
    if (pickPlaceLegIsClaw(k)) {
      // straight above: same heading and arm, the claw approach_dz higher
      EXPECT_LT((up - Vec3{0.0, 0.0, c.approach_dz}).norm(), 1e-12) << pickPlaceLegName(k);
      EXPECT_LT((up_ee - Vec3{0.0, 0.0, c.approach_dz}).norm(), 1e-9) << pickPlaceLegName(k);
      EXPECT_EQ(wp.approach[k].phi, wp.goal[k].phi);
      EXPECT_LT((wp.approach[k].q - wp.goal[k].q).norm(), 1e-15);
    } else {
      EXPECT_LT(up.norm(), 1e-15) << pickPlaceLegName(k);
    }
  }
  PlanOptions opts;
  for (int k : {kExecutePick, kExecutePlace}) {
    // the leg (flown from the previous goal, not from a table) ends ABOVE
    const auto leg = planPickPlaceLeg(*v, *planner, opts, c, wp, k, wp.goal[k - 1], false);
    const WbReference e = leg->eval(leg->duration());
    EXPECT_LT((e.r_ed - wp.approach_ee[k]).norm(), 1e-6) << pickPlaceLegName(k);
    EXPECT_LT(e.x_cd_dot.norm(), 1e-9);
    // the descent goes down onto the target, the claw on the vertical
    const auto down = planPickPlaceDescent(*v, *planner, opts, c, wp, k, wp.approach[k]);
    const WbReference a = down->eval(0.0), b = down->eval(down->duration());
    EXPECT_LT((a.r_ed - wp.approach_ee[k]).norm(), 1e-6);
    EXPECT_LT((b.r_ed - wp.ee[k]).norm(), 1e-6);
    double off_axis = 0.0, z_lo = 1e9, z_hi = -1e9;
    for (int i = 0; i <= 200; ++i) {
      const Vec3 p = down->eval(down->duration() * i / 200.0).r_ed;
      off_axis = std::max(off_axis, (p - wp.ee[k]).head<2>().norm());
      z_lo = std::min(z_lo, p(2));
      z_hi = std::max(z_hi, p(2));
    }
    std::cout << "  " << pickPlaceLegName(k) << " descent: T " << down->duration()
              << " s, claw off the vertical <= " << off_axis * 1e3 << " mm\n";
    EXPECT_LT(off_axis, 0.005) << pickPlaceLegName(k);
    // and never below the target (the claw would push into the table)
    EXPECT_GT(z_lo, wp.ee[k](2) - 0.005) << pickPlaceLegName(k);
    EXPECT_LT(z_hi, wp.approach_ee[k](2) + 0.005) << pickPlaceLegName(k);
    // the descent is only for claw legs
    EXPECT_THROW(planPickPlaceDescent(*v, *planner, opts, c, wp, k + 1, wp.goal[k + 1]),
      std::runtime_error);
  }
}

TEST(PickPlace, AnyLegFlownFromATableRetreatsFirst)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = config(*v);
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, scene());
  PlanOptions opts;
  // execute_pick flown AGAIN with the claw on the object: retreat, then approach
  const RestSpec at_table = wp.goal[kExecutePick];
  const auto again = planPickPlaceLeg(*v, *planner, opts, c, wp, kExecutePick, at_table, true);
  const auto * seq = dynamic_cast<const SequenceTrajectory *>(again.get());
  ASSERT_NE(seq, nullptr);
  ASSERT_EQ(seq->segmentStarts().size(), 2u);
  EXPECT_LT((again->eval(seq->segmentStarts()[1]).x_cd -
    restReference(v->params, retreatRest(at_table, c)).x_cd).norm(), 1e-6);
  // not at a table: straight there
  const auto direct = planPickPlaceLeg(*v, *planner, opts, c, wp, kGoToPlaceStart, wp.goal[0], false);
  const auto * one = dynamic_cast<const SequenceTrajectory *>(direct.get());
  ASSERT_NE(one, nullptr);
  EXPECT_EQ(one->segmentStarts().size(), 1u);
}

TEST(PickPlace, ReflyingAClawLegFromItsApproachHoldIsAShortRest)
{
  // after a wait timeout the vehicle holds ABOVE the target; flying the leg
  // again is then a zero-length move followed by the wait
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = config(*v);
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, scene());
  PlanOptions opts;
  const auto again =
    planPickPlaceLeg(*v, *planner, opts, c, wp, kExecutePick, wp.approach[kExecutePick], false);
  std::cout << "  zero-length leg: T " << again->duration() << " s\n";
  double moved = 0.0;
  const Vec3 x0 = restReference(v->params, wp.approach[kExecutePick]).x_cd;
  for (int i = 0; i <= 50; ++i) {
    moved = std::max(moved, (again->eval(again->duration() * i / 50.0).x_cd - x0).norm());
  }
  EXPECT_LT(moved, 1e-9);
}

TEST(Geofence, GoalsApproachesAndPathsOutsideAreRefused)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  Geofence f;
  EXPECT_EQ(f.violation(Vec3{0.0, 0.0, 1.0}), "");
  EXPECT_NE(f.violation(Vec3{0.0, 0.0, 0.5}).find("floor"), std::string::npos);
  EXPECT_NE(f.violation(Vec3{0.0, 0.0, 1.9}).find("ceiling"), std::string::npos);
  EXPECT_NE(f.violation(Vec3{2.6, 0.0, 1.0}).find("box"), std::string::npos);

  // a land hover at 0.4 m: refused at Plan, naming the leg and the floor
  PickPlaceConfig c = config(*v);
  PickPlaceTargets t = scene();
  t.land.p(2) = 0.4;
  try {
    pickPlaceWaypoints(*v, c, t);
    FAIL() << "a land hover under the fence floor was accepted";
  } catch (const std::runtime_error & e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("execute_land goal"), std::string::npos) << m;
    EXPECT_NE(m.find("floor"), std::string::npos) << m;
  }
  // a pick so high that the point above it is over the ceiling
  t = scene();
  c.fence.max_z = 1.10;   // the pick body is at 1.04 m, its approach at 1.14 m
  try {
    pickPlaceWaypoints(*v, c, t);
    FAIL() << "an approach over the fence ceiling was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("execute_pick approach"), std::string::npos) << e.what();
  }
  // every rest inside, but the retreat off the pick table climbs to 1.19 m
  c.fence.max_z = 1.16;
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, t);
  PlanOptions opts;
  try {
    planPickPlaceLeg(*v, *planner, opts, c, wp, kGoToPlaceStart, wp.goal[kExecutePick], true);
    FAIL() << "a retreat over the fence ceiling was accepted";
  } catch (const std::runtime_error & e) {
    const std::string m = e.what();
    EXPECT_NE(m.find("go_to_place_start: the path leaves the geofence"), std::string::npos) << m;
    EXPECT_NE(m.find("ceiling"), std::string::npos) << m;
  }
  // the same leg without the retreat stays inside
  EXPECT_NO_THROW(
    planPickPlaceLeg(*v, *planner, opts, c, wp, kGoToPlaceStart, wp.goal[kExecutePick], false));
  // a leg may START outside (DIRECT engaged at 0.4 m) and climb back in ...
  RestSpec low = wp.goal[kGoToStart];
  low.x_b(2) = 0.4;
  EXPECT_NO_THROW(planPickPlaceLeg(*v, *planner, opts, c, wp, kGoToStart, low, false));
  // ... but not go further out on the way: a retreat climbing from over the ceiling
  RestSpec high = wp.goal[kExecutePick];
  high.x_b(2) = 1.20;   // over the 1.16 m ceiling; the retreat climbs 0.15 m more
  try {
    planPickPlaceLeg(*v, *planner, opts, c, wp, kGoToPlaceStart, high, true);
    FAIL() << "a leg climbing further over the ceiling was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("further out than where the leg starts"), std::string::npos)
      << e.what();
  }
}

TEST(SequenceTrajectory, RefusesSegmentsThatDoNotMeet)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  RestSpec a, b;
  a.q = b.q = v->home;
  a.x_b << 0.0, 0.0, 1.0;
  b.x_b << 0.5, 0.0, 1.0;
  std::vector<std::shared_ptr<const Trajectory>> segs = {
    std::make_shared<HoldTrajectory>(v->params, a), std::make_shared<HoldTrajectory>(v->params, b)};
  EXPECT_THROW(SequenceTrajectory{segs}, std::runtime_error);
}

namespace
{
// (r, z) of the grasp point about the body origin, as the PS4 tab measures it
void clawRz(const VehicleModel & v, const VecN & q, double * r, double * z)
{
  Vec3 r0e;
  armKinematics(q, v.params, nullptr, &r0e, nullptr);
  const Vec3 b = v.r_model * r0e;
  *r = std::hypot(b(0), b(1));
  *z = b(2);
}
}  // namespace

TEST(Workspace, ClawDownPoseIsInsideTheEnvelopeTheNodePublishes)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const double guard = v->beta_min_deg * kDeg;
  const WorkspaceGrid legacy = usableWorkspace(*v, guard);       // before 2026-10-01
  const WorkspaceGrid ext = usableWorkspace(*v, -1.0 * kDeg);    // pick pose fold 0 - 1 deg
  ASSERT_FALSE(legacy.cells.empty());
  ASSERT_FALSE(ext.cells.empty());
  double r = 0.0, z = 0.0;
  clawRz(*v, v->home, &r, &z);
  EXPECT_TRUE(legacy.contains(r, z));
  EXPECT_TRUE(ext.contains(r, z));
  const VecN down = VecN::Zero();
  clawRz(*v, down, &r, &z);
  // THE BLOCKER: the claw-down grasp point is outside the beta >= 5 deg grid,
  // so the PS4 tab refused every stick increment there
  EXPECT_FALSE(legacy.contains(r, z)) << "r " << r << " z " << z;
  EXPECT_TRUE(ext.contains(r, z)) << "r " << r << " z " << z;
  // from there the fine correction can go up, in and a little sideways ...
  EXPECT_TRUE(ext.contains(r, z + 0.01));
  EXPECT_TRUE(ext.contains(r - 0.01, z));
  EXPECT_TRUE(ext.contains(std::hypot(r, 0.01), z));
  // ... but not down or out: the bottom of the arm's reach
  EXPECT_FALSE(ext.contains(r, z - 0.02));
  EXPECT_FALSE(ext.contains(r + 0.05, z));
  // the extension only adds cells (same grid geometry? then cell by cell)
  int lost = 0, checked = 0;
  for (int zi = 0; zi < legacy.nz; ++zi) {
    for (int ri = 0; ri < legacy.nr; ++ri) {
      if (!legacy.cells[static_cast<size_t>(zi) * legacy.nr + ri]) {continue;}
      const double rr = legacy.r_min + (legacy.r_max - legacy.r_min) * ri / (legacy.nr - 1);
      const double zz = legacy.z_max - (legacy.z_max - legacy.z_min) * zi / (legacy.nz - 1);
      ++checked;
      // one cell of slack: the two grids are scaled to their own extents
      bool near = false;
      for (double dr : {-0.002, 0.0, 0.002}) {
        for (double dz : {-0.002, 0.0, 0.002}) {near = near || ext.contains(rr + dr, zz + dz);}
      }
      lost += near ? 0 : 1;
    }
  }
  EXPECT_GT(checked, 1000);
  EXPECT_EQ(lost, 0);
  std::cout << "  legacy " << 100 * legacy.filledFraction() << "% filled, z >= " << legacy.zs_min
            << "; extended " << 100 * ext.filledFraction() << "%, z >= " << ext.zs_min
            << "; claw-down at r " << r << " z " << z << "\n";
}

TEST(ClawHeading, IkWithThePublishedHeadingLeavesTheArmWhereItIs)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  std::vector<VecN> poses;
  VecN q;
  q << 0.0, 0.0, 0.0, 0.0;           // claw down (the pick pose)
  poses.push_back(q);
  poses.push_back(v->home);
  q << 20.0 * kDeg, 10.0 * kDeg, 15.0 * kDeg, -30.0 * kDeg;
  poses.push_back(q);
  q << -15.0 * kDeg, -5.0 * kDeg, 10.0 * kDeg, 45.0 * kDeg;   // near claw-down, yawed
  poses.push_back(q);
  for (const VecN & qq : poses) {
    RestSpec r;
    r.x_b << 0.4, -0.3, 1.1;
    r.phi = 0.7;
    r.q = qq;
    bool ok = false;
    const double az = clawAzimuth(v->params, Rz(r.phi), r.q, &ok);
    ASSERT_TRUE(ok);
    // the same angle the hold's reference carries (the node's hold anchor)
    const WbReference ref = restReference(v->params, r);
    EXPECT_NEAR(std::atan2(ref.b1_de(1), ref.b1_de(0)), az, 1e-12);
    // a target at the CURRENT claw with this heading is the current arm pose,
    // from a seed a few degrees off (the encoders, a settling arm)
    const IkResult ik = ikWorld(
      v->params, r.x_b, r.phi, clawOf(*v, r), az, VecN(r.q + VecN::Constant(2.0 * kDeg)));
    ASSERT_TRUE(ik.ok) << ik.reason;
    EXPECT_LT((ik.q - r.q).cwiseAbs().maxCoeff(), 1e-6) << "q " << (r.q / kDeg).transpose();
  }
}
