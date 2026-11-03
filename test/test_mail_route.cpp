// The mail route (mail_route.hpp): three letters from three hats in a line in
// front of two mailboxes, posted alternately -- one letter is exactly the
// single mail mission, the boxes alternate, every letter starts where the
// previous one's place exit ended, the land legs come after the last letter
// only, and the far-to-near order keeps every pick over empty hats.
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/mail_route.hpp"
#include "fsc_trajectory_planner/pick_place.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"

using namespace fsc_trajectory_planner;

namespace
{
constexpr double kDeg = M_PI / 180.0;

// the sim mail yaml's mission block (the insert, the 25 deg diagonal)
PickPlaceConfig mailConfig(const VehicleModel & v)
{
  PickPlaceConfig c;
  c.home = v.home;
  c.pick_pose = (VecN() << 0.0, 36.0 * kDeg, 44.0 * kDeg, 90.0 * kDeg).finished();
  c.carry = c.pick_pose;
  c.place_pose = (VecN() << 0.0, 20.0 * kDeg, 45.0 * kDeg, 90.0 * kDeg).finished();
  c.sweep.lo = c.sweep.hi = c.carry;
  c.ee_offset << -0.06, 0.0, 0.0;
  c.approach_dz = 0.20;
  c.pick_approach_back = 0.10;
  c.place_exit_back = 0.10;
  c.place_insert = true;
  c.place_approach_back = 0.15;
  c.place_insert_pitch_deg = 25.0;
  c.fence.min_z = 0.30;
  c.fence.max_z = 2.2;
  return c;
}

// the 07 scene's mailboxes (fronts at x = 1.70, RED y = +0.25, BLUE y = -0.25)
// and the user's line of hats (2026-11-03): on y = 0, extending away from the
// boxes, the nearest 1.0 m out (beyond Place Start, 0.709 m), 0.6 m apart,
// each letter TURNED 90 deg (yaw +90: the pick comes in from -y), NEAREST
// first. The letter is the localization frame on the hat + the sim yaml's pick
// locator, as Get computes it.
const std::vector<double> kLocator{-0.0914, -0.0064, -0.039, 0.0};

MailRouteTargets scene(int n_letters = 3, int n_boxes = 2, double spacing = 0.6, double yaw_deg = 90.0)
{
  MailRouteTargets t;
  t.start.p << 0.70, -0.70, 1.10;    // beside the nearest hat, off the line
  t.start.yaw = 0.5 * M_PI;
  t.land_start.p << -0.5, -1.0, 1.0;
  t.land.p << -0.5, -1.0, 0.35;
  for (int i = 0; i < n_letters; ++i) {
    const double yaw = yaw_deg * kDeg;
    const Vec3 hat(0.70 - spacing * i, 0.0, 0.80 + 0.039);   // the frame's markers, on the hat
    t.letters.push_back(hat + Rz(yaw) * Vec3(kLocator[0], kLocator[1], kLocator[2]));
    t.letter_yaw.push_back(yaw);
  }
  const double box_y[2] = {0.25, -0.25};
  for (int b = 0; b < n_boxes; ++b) {
    t.boxes.push_back(Vec3(1.709, box_y[b], 1.024));
    t.box_yaw.push_back(0.0);
  }
  t.place_start_offset = {-0.709, 0.0, 0.196, 0.0};
  t.pick_start_offset = {-0.50, 0.0, 0.35, 0.0};
  return t;
}

RestSpec hold(const VehicleModel & v)
{
  RestSpec r;   // DIRECT engaged over the take-off spot, off the line of hats
  r.x_b << -0.5, -1.0, 1.0;
  r.phi = -0.5 * M_PI;
  r.q = v.home;
  return r;
}

double maxDiff(const Trajectory & a, const Trajectory & b)
{
  double d = std::abs(a.duration() - b.duration());
  for (int i = 0; i <= 50; ++i) {
    const double t = a.duration() * i / 50.0;
    const WbReference x = a.eval(t), y = b.eval(t);
    d = std::max({d, (x.x_cd - y.x_cd).norm(), (x.r_ed - y.r_ed).norm(), (x.q_d - y.q_d).norm()});
  }
  return d;
}
}  // namespace

TEST(MailRoute, ARouteOfOneLetterIsTheSingleMailMissionExactly)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = mailConfig(*v);
  MailRouteTargets t = scene(1, 1);
  t.pick_start_offset.clear();          // no Pick Start: go_to_start flies to Start, as today
  const auto wps = mailRouteWaypoints(*v, c, t);
  ASSERT_EQ(wps.size(), 1u);
  const MailRoutePlan plan = planMailRoute(*v, *planner, PlanOptions{}, c, wps, hold(*v));

  // the same targets through the six-leg mission
  PickPlaceTargets s;
  s.start = t.start;
  s.pick = t.letters[0];
  s.pick_yaw = t.letter_yaw[0];
  s.place = t.boxes[0];
  s.place_start = mailRouteOffsetPose(t.boxes[0], 0.0, t.place_start_offset);
  s.land_start = t.land_start;
  s.land = t.land;
  const PickPlaceWaypoints wp = pickPlaceWaypoints(*v, c, s);
  std::array<std::shared_ptr<const Trajectory>, kNumPickPlaceLegs> exits;
  const auto legs = planPickPlaceMission(*v, *planner, PlanOptions{}, c, wp, hold(*v), &exits);
  for (int k = 0; k < kNumPickPlaceLegs; ++k) {
    ASSERT_TRUE(plan.legs[0][k]) << pickPlaceLegName(k);
    EXPECT_LT(maxDiff(*plan.legs[0][k], *legs[k]), 1e-12) << pickPlaceLegName(k);
    EXPECT_EQ(static_cast<bool>(plan.exits[0][k]), static_cast<bool>(exits[k])) << pickPlaceLegName(k);
    if (exits[k]) {EXPECT_LT(maxDiff(*plan.exits[0][k], *exits[k]), 1e-12) << pickPlaceLegName(k);}
  }
}

TEST(MailRoute, ThreeLettersArePostedIntoTheTwoBoxesInAlternation)
{
  EXPECT_EQ(mailRouteBox(0, 2), 0);
  EXPECT_EQ(mailRouteBox(1, 2), 1);
  EXPECT_EQ(mailRouteBox(2, 2), 0);
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PickPlaceConfig c = mailConfig(*v);
  const MailRouteTargets t = scene();
  const auto wps = mailRouteWaypoints(*v, c, t);
  ASSERT_EQ(wps.size(), 3u);
  const int box_of[3] = {0, 1, 0};   // RED, BLUE, RED
  for (int i = 0; i < 3; ++i) {
    // each letter's claw on its own hat, and on its box's mouth target
    EXPECT_LT((wps[i].ee[kExecutePick] - (t.letters[i] + Rz(t.letter_yaw[i]) * c.ee_offset)).norm(), 1e-9) << i;
    EXPECT_LT((wps[i].ee[kExecutePlace] - (t.boxes[box_of[i]] + c.ee_offset)).norm(), 1e-9) << i;
    // the box's Place Start follows the box
    const Vec3 ps = t.boxes[box_of[i]] + Vec3(-0.709, 0.0, 0.196);
    EXPECT_LT((wps[i].goal[kGoToPlaceStart].x_b - ps).norm(), 1e-12) << i;
  }
}

TEST(MailRoute, EachLetterStartsWhereThePreviousOnesPlaceExitEnded)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = mailConfig(*v);
  const MailRouteTargets t = scene();
  const auto wps = mailRouteWaypoints(*v, c, t);
  const MailRoutePlan plan = planMailRoute(*v, *planner, PlanOptions{}, c, wps, hold(*v));
  ASSERT_EQ(plan.legs.size(), 3u);
  double total = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    const bool last = i == 2;
    for (int k = 0; k < kNumPickPlaceLegs; ++k) {
      // the land legs after the last letter only
      ASSERT_EQ(static_cast<bool>(plan.legs[i][k]), k < kGoToLandStart || last) << i << " " << k;
      if (!plan.legs[i][k]) {continue;}
      const Trajectory & L = *plan.legs[i][k];
      total += L.duration() + (plan.exits[i][k] ? plan.exits[i][k]->duration() : 0.0);
      std::cout << "  letter " << i + 1 << " " << pickPlaceLegName(k) << ": T " << L.duration() << " s"
                << (plan.exits[i][k] ? " + exit " + std::to_string(plan.exits[i][k]->duration()) + " s" : "")
                << "\n";
      EXPECT_LT((L.eval(L.duration()).x_cd - restReference(v->params, wps[i].goal[k]).x_cd).norm(), 1e-6);
    }
    // go_to_start: every letter's own PICK START (the operator picks the order);
    // after the first, flown straight from the previous place exit's clearance point
    const Vec3 pick_start = mailRouteOffsetPose(t.letters[i], t.letter_yaw[i], t.pick_start_offset).p;
    EXPECT_LT((wps[i].goal[kGoToStart].x_b - pick_start).norm(), 1e-12) << i;
    if (i > 0) {
      const Trajectory & X = *plan.exits[i - 1][kExecutePlace];
      const Trajectory & G = *plan.legs[i][kGoToStart];
      EXPECT_LT((G.eval(0.0).x_cd - X.eval(X.duration()).x_cd).norm(), 1e-9) << i;
      EXPECT_LT((G.eval(0.0).q_d - X.eval(X.duration()).q_d).norm(), 1e-9) << i;
      const auto * seq = dynamic_cast<const SequenceTrajectory *>(&G);
      ASSERT_NE(seq, nullptr);
      EXPECT_EQ(seq->segmentStarts().size(), 1u) << "letter " << i + 1 << " retreated after an exit";
    }
  }
  // the whole route is one chain, every leg and exit in order
  EXPECT_NEAR(plan.whole->duration(), total, 1e-9);
  const Trajectory & land = *plan.legs[2][kExecuteLand];
  EXPECT_LT((plan.whole->eval(plan.whole->duration()).x_cd - land.eval(land.duration()).x_cd).norm(), 1e-9);
  std::cout << "  route total " << total << " s\n";
}

TEST(MailRoute, TheHatsAreWhereTheFrameStoodOnThem)
{
  // the letter = the frame + Rz(yaw) locator, so the hat = the letter minus it
  const MailRouteTargets t = scene();
  const auto hats = mailRouteHats(t, kLocator);
  ASSERT_EQ(hats.size(), 3u);
  for (int i = 0; i < 3; ++i) {
    EXPECT_LT((hats[i].head<2>() - Vec3(0.70 - 0.6 * i, 0.0, 0.0).head<2>()).norm(), 1e-12) << i;
    EXPECT_NEAR(hats[i](2), t.letters[i](2), 1e-12);   // the top: the letter's height
  }
  EXPECT_LT((mailRouteHats(t, {})[1] - t.letters[1]).norm(), 1e-12);   // no locator: at the letters
  // obj_0 is the markers' CENTROID, c from the hat's centre O in the frame's axes
  // (2026-11-03 audit): the hat is O = obj_0 - Rz(yaw) c; the letters here are turned
  // 90 deg, so Rz c = (-c_y, c_x)
  const std::vector<double> c{-0.00358, 0.00638};
  const auto at_o = mailRouteHats(t, kLocator, c);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(at_o[i](0), 0.70 - 0.6 * i + c[1], 1e-12) << i;
    EXPECT_NEAR(at_o[i](1), -c[0], 1e-12) << i;
    EXPECT_NEAR(at_o[i](2), t.letters[i](2), 1e-12) << i;
  }
}

TEST(MailRoute, ALowVerticalMarginStillLiftsTheLetterOffItsOwnHat)
{
  // 2026-11-03 audit: the go_to_place_start after a pick starts over the hat the letter
  // was just lifted from. Its claw was checked against that hat, so with the pick's exit
  // under ~0.13 m above the letter (Vertical Margin 0.10) the leg was refused at t = 0
  // with the letter in the claw. That hat is exempt for the claw on this leg too (the
  // leg climbs to the cruise floor first); the airframe is still checked against it.
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  PickPlaceConfig c = mailConfig(*v);
  c.approach_dz = 0.10;
  const MailRouteTargets t = scene();
  const auto hats = mailRouteHats(t, kLocator);
  std::string why;
  try {
    planMailRoute(*v, *planner, PlanOptions{}, c, mailRouteWaypoints(*v, c, t), hold(*v), &hats);
  } catch (const std::runtime_error & e) {
    why = e.what();
  }
  EXPECT_TRUE(why.empty()) << why;
}

TEST(MailRoute, AMidRoutePlanIsCheckedFromLetterOnesPickStart)
{
  // 2026-11-03 audit: a route re-planned at a pick hold (after a margin edit there) was
  // dry-run from that hold with letter 1's go_to_start -- the claw starts on the hat it
  // is picking from, which is not that leg's own, so the whole Plan was refused. The
  // node now dry-runs a route under way from letter 1's Pick Start; the first leg is
  // then a rest-to-rest of zero length, which the planner flies as a short hold.
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = mailConfig(*v);
  const MailRouteTargets t = scene();
  const auto wps = mailRouteWaypoints(*v, c, t);
  const auto hats = mailRouteHats(t, kLocator);
  std::string old_why;
  try {
    planMailRoute(*v, *planner, PlanOptions{}, c, wps, wps[1].goal[kExecutePick], &hats);
  } catch (const std::runtime_error & e) {
    old_why = e.what();
  }
  std::cout << "  from letter 2's pick hold: " << old_why << "\n";
  EXPECT_NE(old_why.find("letter 1: go_to_start"), std::string::npos) << old_why;
  EXPECT_NE(old_why.find("hat 2"), std::string::npos) << old_why;
  std::string why;
  try {
    const MailRoutePlan p = planMailRoute(*v, *planner, PlanOptions{}, c, wps, wps[0].goal[kGoToStart], &hats);
    EXPECT_TRUE(p.whole);
  } catch (const std::runtime_error & e) {
    why = e.what();
  }
  EXPECT_TRUE(why.empty()) << why;
}

TEST(MailRoute, EveryLegKeepsTheAirframeAndClawClearOfTheHats)
{
  // The pick holds the claw ~0.07 m under the body origin, so the gear (0.243 m
  // under it) hangs below the hat tops there: a hat behind the one picked from
  // is struck. The whole route, every leg and exit, against every hat.
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = mailConfig(*v);
  const auto fly = [&](const MailRouteTargets & t) {
      const auto hats = mailRouteHats(t, kLocator);
      try {
        planMailRoute(*v, *planner, PlanOptions{}, c, mailRouteWaypoints(*v, c, t), hold(*v), &hats);
      } catch (const std::runtime_error & e) {
        return std::string(e.what());
      }
      return std::string();
    };
  // the chosen layout: 0.6 m apart, letters turned 90 deg -- clear end to end
  std::string why = fly(scene());
  EXPECT_TRUE(why.empty()) << why;
  // 0.5 m apart, turned (the user's first spacing): clear as well -- the nearest
  // prop disc stays off the next hat (and 0.13 m above its top)
  why = fly(scene(3, 2, 0.5));
  EXPECT_TRUE(why.empty()) << why;
  // the letters facing the boxes, 0.5 m apart, picked ALONG the line: the pick
  // stands the gear on the next hat back, whatever the order and the Start
  MailRouteTargets along = scene(3, 2, 0.5, 0.0);
  along.start.p << 0.30, -0.80, 1.10;   // beside the line, so letter 1's leg in crosses no hat
  along.start.yaw = 0.0;
  why = fly(along);
  std::cout << "  0.5 m, facing the boxes: " << why << "\n";
  EXPECT_NE(why.find("letter 1: execute_pick"), std::string::npos) << why;
  EXPECT_NE(why.find("gear"), std::string::npos) << why;
  EXPECT_NE(why.find("hat 2"), std::string::npos) << why;
  // the claw may reach its own hat, never another's: letter 2 sent to hat 1's letter
  MailRouteTargets swap = scene();
  swap.letters[1] = swap.letters[0];
  const auto hats = mailRouteHats(scene(), kLocator);
  try {
    planMailRoute(*v, *planner, PlanOptions{}, c, mailRouteWaypoints(*v, c, swap), hold(*v), &hats);
    ADD_FAILURE() << "a claw driven onto another letter's hat was accepted";
  } catch (const std::runtime_error & e) {
    EXPECT_NE(std::string(e.what()).find("letter 2"), std::string::npos) << e.what();
  }
}

TEST(MailRoute, APicksOwnHatIsClearOfTheFrameWithRoomForATrim)
{
  // route3 (Isaac, 2026-11-03): letter 2's pick descent, re-planned from the hold with
  // its trim, was refused -- a SQUARE frame model (+-0.253 m) put the airframe 29 mm
  // from its own hat's rim. Straight ahead the frame ends at 0.081 m (the measured
  // outline): the mail pick's own hat, the descent nudged 20 mm toward it, is clear.
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const auto planner = makePlanner("bspline");
  const PickPlaceConfig c = mailConfig(*v);
  const MailRouteTargets t = scene();
  const auto wps = mailRouteWaypoints(*v, c, t);
  const auto hats = mailRouteHats(t, kLocator);
  for (int i = 0; i < 3; ++i) {
    PickPlaceWaypoints wp = wps[i];
    const double phi = wp.goal[kExecutePick].phi;
    wp.goal[kExecutePick].x_b += 0.020 * Vec3(-std::sin(phi), std::cos(phi), 0.0);   // toward the hat
    const auto D = planPickPlaceDescent(*v, *planner, PlanOptions{}, c, wp, kExecutePick, wp.approach[kExecutePick]);
    const std::string why = hatConflict(*v, *D, hats, i, HatClearance{}, "descent");
    EXPECT_TRUE(why.empty()) << "letter " << i + 1 << ": " << why;
  }
}

TEST(MailRoute, RefusesWhatTheRouteCannotFly)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  const PickPlaceConfig c = mailConfig(*v);
  const auto expectRefused = [&](const PickPlaceConfig & cc, const MailRouteTargets & tt, const std::string & what) {
      try {
        mailRouteWaypoints(*v, cc, tt);
        ADD_FAILURE() << "accepted; expected a refusal naming '" << what << "'";
      } catch (const std::runtime_error & e) {
        EXPECT_NE(std::string(e.what()).find(what), std::string::npos) << e.what();
      }
    };
  PickPlaceConfig not_insert = c;
  not_insert.place_insert = false;
  expectRefused(not_insert, scene(), "INSERT");
  MailRouteTargets t = scene();
  t.place_start_offset.clear();
  expectRefused(c, t, "pick_place_place_start_offset");
  expectRefused(c, scene(0, 2), "no letter");
  expectRefused(c, scene(3, 0), "no mailbox");
  t = scene();
  t.letters[1](0) = -3.0;   // the second hat outside the geofence
  expectRefused(c, t, "letter 2");
}
