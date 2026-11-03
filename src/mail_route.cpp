// MIT License
// Copyright (c) 2026 FSC Lab
#include "fsc_trajectory_planner/mail_route.hpp"

#include "fsc_trajectory_planner/kinematics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fsc_trajectory_planner
{

namespace
{
std::string letterName(size_t i) {return "letter " + std::to_string(i + 1);}
}  // namespace

int mailRouteBox(int letter, int n_boxes)
{
  return n_boxes > 0 ? letter % n_boxes : 0;
}

BasePose mailRouteOffsetPose(const Vec3 & p, double yaw, const std::vector<double> & d)
{
  BasePose b;
  const double c = std::cos(yaw), s = std::sin(yaw);
  b.p = p + Vec3{c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]};
  b.yaw = yaw + d[3] * M_PI / 180.0;
  return b;
}

std::vector<PickPlaceTargets> mailRouteLetterTargets(const MailRouteTargets & t)
{
  if (t.letters.empty()) {throw std::runtime_error("mail route: no letter");}
  if (t.boxes.empty()) {throw std::runtime_error("mail route: no mailbox");}
  if (t.letter_yaw.size() != t.letters.size() || t.box_yaw.size() != t.boxes.size()) {
    throw std::runtime_error("mail route: one yaw per letter and per mailbox");
  }
  if (t.place_start_offset.size() != 4) {
    throw std::runtime_error(
      "mail route: each mailbox's Place Start follows the box -- pick_place_place_start_offset "
      "needs [dx, dy, dz, dyaw_deg]");
  }
  if (!t.pick_start_offset.empty() && t.pick_start_offset.size() != 4) {
    throw std::runtime_error("mail route: pick_place_route_pick_start_offset needs [] or [dx, dy, dz, dyaw_deg]");
  }
  std::vector<PickPlaceTargets> out;
  for (size_t i = 0; i < t.letters.size(); ++i) {
    PickPlaceTargets g;
    // every letter from its own Pick Start (the operator chooses the order)
    g.start = t.pick_start_offset.empty()
      ? t.start : mailRouteOffsetPose(t.letters[i], t.letter_yaw[i], t.pick_start_offset);
    g.pick = t.letters[i];
    g.pick_yaw = t.letter_yaw[i];
    const int b = mailRouteBox(static_cast<int>(i), static_cast<int>(t.boxes.size()));
    g.place = t.boxes[b];
    g.place_yaw = t.box_yaw[b];
    g.place_start = mailRouteOffsetPose(g.place, g.place_yaw, t.place_start_offset);
    g.land_start = t.land_start;
    g.land = t.land;
    out.push_back(g);
  }
  return out;
}

std::vector<PickPlaceWaypoints> mailRouteWaypoints(
  const VehicleModel & v, const PickPlaceConfig & c, const MailRouteTargets & t)
{
  if (!c.place_insert || c.place_drop) {
    throw std::runtime_error(
      "mail route: the letters are posted with the INSERT place (pick_place_place_insert true, "
      "pick_place_place_drop false)");
  }
  const auto targets = mailRouteLetterTargets(t);
  std::vector<PickPlaceWaypoints> wps;
  for (size_t i = 0; i < targets.size(); ++i) {
    try {
      wps.push_back(pickPlaceWaypoints(v, c, targets[i]));
    } catch (const std::exception & e) {
      throw std::runtime_error(letterName(i) + ": " + e.what());
    }
  }
  return wps;
}

std::vector<Vec3> mailRouteHats(const MailRouteTargets & t, const std::vector<double> & pick_locator)
{
  std::vector<Vec3> hats;
  const bool loc = pick_locator.size() == 3 || pick_locator.size() == 4;
  for (size_t i = 0; i < t.letters.size(); ++i) {
    Vec3 h = t.letters[i];
    if (loc) {
      // the letter = the frame + Rz(frame yaw) [dx, dy, dz], frame yaw = letter yaw - dyaw
      const double fy = t.letter_yaw[i] - (pick_locator.size() == 4 ? pick_locator[3] * M_PI / 180.0 : 0.0);
      h.head<2>() -= (Rz(fy) * Vec3(pick_locator[0], pick_locator[1], 0.0)).head<2>();
    }
    hats.push_back(h);
  }
  return hats;
}

std::string hatConflict(
  const VehicleModel & v, const Trajectory & traj, const std::vector<Vec3> & hats, int own,
  const HatClearance & h, const std::string & what)
{
  const int n = std::max(200, static_cast<int>(std::ceil(traj.duration() / 0.02)));
  const double rr = h.hat_radius + h.margin;
  for (int k = 0; k <= n; ++k) {
    const double t = traj.duration() * k / n;
    const WbReference ref = traj.eval(t);
    // the body origin, level at the reference heading (checkPathInFence's)
    Vec3 r0c;
    armKinematics(ref.q_d, v.params, &r0c, nullptr, nullptr);
    const double phi = std::atan2(ref.b1_d(1), ref.b1_d(0));
    const Vec3 body = ref.x_cd - Rz(phi) * r0c;
    const Vec3 nose{-std::sin(phi), std::cos(phi), 0.0}, side{std::cos(phi), std::sin(phi), 0.0};
    for (size_t j = 0; j < hats.size(); ++j) {
      const double top = hats[j](2) + h.hat_cap + h.margin;   // what a part must stay above
      Vec3 rel = hats[j] - body;
      rel(2) = 0.0;
      const char * part = nullptr;
      double d = 0.0, low = 0.0;
      if (body(2) + h.gear_bottom < top) {
        const double dn = std::max(std::abs(rel.dot(nose)) - h.gear_half_nose, 0.0);
        const double ds = std::max(std::abs(rel.dot(side)) - h.gear_half_side, 0.0);
        if (std::hypot(dn, ds) < rr) {part = "gear"; d = std::hypot(dn, ds); low = body(2) + h.gear_bottom;}
      }
      // the hat in the body's axes (the airframe is symmetric under a half turn)
      const double an = std::abs(rel.dot(nose)), as = std::abs(rel.dot(side));
      if (part == nullptr && body(2) + h.frame_bottom < top) {
        // the frame's outline over the sectors the hat (+ margin) spans, seen from the body
        const double D = std::hypot(an, as);
        const double half = D > rr ? std::asin(rr / D) : M_PI;
        const double th = std::atan2(as, an);          // from the nose (the outline is symmetric)
        double reach = 0.0;
        for (int b = 0; b < 36; ++b) {
          const double mid = (b + 0.5) * M_PI / 18.0;
          const double diff = std::abs(std::remainder(mid - th, 2.0 * M_PI));
          if (diff <= half + M_PI / 36.0) {reach = std::max(reach, h.frame_outline[b]);}
        }
        if (D - reach < rr) {part = "frame"; d = std::max(D - reach, 0.0); low = body(2) + h.frame_bottom;}
      }
      if (part == nullptr && body(2) + h.prop_bottom < top) {
        // the nearest of the four discs: the motor in the hat's quadrant
        const double dp = std::hypot(an - h.rotor_offset, as - h.rotor_offset) - h.prop_radius;
        if (dp < rr) {part = "props"; d = std::max(dp, 0.0); low = body(2) + h.prop_bottom;}
      }
      if (part == nullptr && static_cast<int>(j) != own && ref.r_ed(2) - h.claw_drop < top) {
        const double dc = (hats[j] - ref.r_ed).head<2>().norm() - h.claw_radius;
        if (dc < rr) {part = "claw"; d = std::max(dc, 0.0); low = ref.r_ed(2) - h.claw_drop;}
      }
      if (part == nullptr) {continue;}
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(2);
      const double edge = d - h.hat_radius;   // < 0: over the hat's disc
      m << what << ": at t = " << std::setprecision(1) << t << " s the " << part << " is " << std::setprecision(3)
        << std::abs(edge) << " m " << (edge < 0.0 ? "inside" : "outside") << " hat " << j + 1 << "'s edge, "
        << low - hats[j](2) << " m above its top (needs " << h.margin << " m clear of the edge, or "
        << h.hat_cap + h.margin << " m above the top; body at [" << body(0) << ", " << body(1) << ", "
        << body(2) << "], hat at [" << hats[j](0) << ", " << hats[j](1) << "], top " << hats[j](2) << ")";
      return m.str();
    }
  }
  return "";
}

double mailRouteCruiseFloor(const std::vector<Vec3> & hats, const HatClearance & h)
{
  double floor = -std::numeric_limits<double>::infinity();
  for (const Vec3 & hat : hats) {floor = std::max(floor, hat(2) + h.hat_cap + h.margin - h.gear_bottom);}
  return floor;
}

std::unique_ptr<Trajectory> planMailRouteLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from, bool from_table, double floor)
{
  const bool transit = leg == kGoToStart || leg == kGoToPlaceStart || leg == kGoToLandStart;
  if (!transit || !(from.x_b(2) < floor - 1e-3)) {
    return planPickPlaceLeg(v, transition, opts, c, wp, leg, from, from_table);
  }
  RestSpec up = from;
  up.x_b(2) = floor;
  std::vector<std::shared_ptr<const Trajectory>> segs;
  try {
    const std::string why = c.fence.violation(up.x_b);
    if (!why.empty()) {throw std::runtime_error("the climb to the cruise floor: " + why);}
    PlanRequest req;
    req.rest0 = from;
    req.rest1 = up;
    segs.push_back(transition.plan(v, req, opts));
  } catch (const std::exception & e) {
    throw std::runtime_error(std::string(pickPlaceLegName(leg)) + ": " + e.what());
  }
  segs.push_back(planPickPlaceLeg(v, transition, opts, c, wp, leg, up, from_table));
  return std::make_unique<SequenceTrajectory>(std::move(segs));
}

MailRoutePlan planMailRoute(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const std::vector<PickPlaceWaypoints> & wps, const RestSpec & from,
  const std::vector<Vec3> * hats, const HatClearance & hc)
{
  if (wps.empty()) {throw std::runtime_error("mail route: no letter");}
  MailRoutePlan plan;
  plan.legs.resize(wps.size());
  plan.exits.resize(wps.size());
  std::vector<std::shared_ptr<const Trajectory>> all;   // drawn: each exit after its leg
  const double floor = hats != nullptr ? mailRouteCruiseFloor(*hats, hc)
                                       : -std::numeric_limits<double>::infinity();
  RestSpec r = from;
  bool exited = false;   // the leg before was a claw leg followed by its exit
  for (size_t i = 0; i < wps.size(); ++i) {
    const PickPlaceWaypoints & wp = wps[i];
    const bool last = i + 1 == wps.size();
    for (int k = 0; k < (last ? kNumPickPlaceLegs : kGoToLandStart); ++k) {
      try {
        // planPickPlaceMission's chaining: approach + descent on a claw leg,
        // the exit after it, no retreat after an exit
        std::shared_ptr<const Trajectory> leg =
          planMailRouteLeg(v, transition, opts, c, wp, k, r, pickPlaceLegRetreats(k) && !exited, floor);
        exited = false;
        if (pickPlaceLegIsClaw(k)) {
          std::vector<std::shared_ptr<const Trajectory>> segs{
            leg, std::shared_ptr<const Trajectory>(
              planPickPlaceDescent(v, transition, opts, c, wp, k, wp.approach[k]))};
          leg = std::make_shared<SequenceTrajectory>(std::move(segs));
        }
        // the claw is MEANT to reach its own letter's hat on the pick and its exit
        const int own = k == kExecutePick ? static_cast<int>(i) : -1;
        const auto check = [&](const Trajectory & tr, const std::string & what) {
            if (hats == nullptr) {return;}
            const std::string why = hatConflict(v, tr, *hats, own, hc, what);
            if (!why.empty()) {throw std::runtime_error(why);}
          };
        check(*leg, pickPlaceLegName(k));
        plan.legs[i][k] = leg;
        all.push_back(leg);
        r = wp.goal[k];
        if (pickPlaceLegIsClaw(k)) {
          std::shared_ptr<const Trajectory> ex = planPickPlaceExit(v, transition, opts, c, k, r, &wp);
          check(*ex, std::string(pickPlaceLegName(k)) + " exit");
          r = ex->goalRest();
          plan.exits[i][k] = ex;
          all.push_back(std::move(ex));
          exited = true;
        }
      } catch (const std::exception & e) {
        throw std::runtime_error(letterName(i) + ": " + e.what());
      }
    }
  }
  plan.whole = std::make_shared<SequenceTrajectory>(std::move(all));
  return plan;
}

}  // namespace fsc_trajectory_planner
