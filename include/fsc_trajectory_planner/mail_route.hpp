// MIT License
// Copyright (c) 2026 FSC Lab
//
// THE MAIL ROUTE (2026-11-03, user design): the mail delivery extended from one
// envelope to THREE, picked from three hats in a line in front of the
// mailboxes, extending away from them (the nearest ~1 m out, 0.6 m apart), and
// posted into TWO mailboxes in ALTERNATION -- letter 1 the first box (RED),
// letter 2 the second (BLUE), letter 3 the first again. The letters lie
// TURNED 90 deg on their hats, so each pick comes in from the side of the line
// and the vehicle turns to the boxes on the way there (along the line, 0.5 m
// apart, a pick stands the gear on the next hat back: see HatClearance).
//
// It is the pick-and-place insert mission flown once per letter, NOT a second
// mission: every letter's legs are pick_place.hpp's (planPickPlaceLeg /
// Descent / Exit), on per-letter waypoints from pickPlaceWaypoints, so the
// insert -- approach, gate, slide, release window, exits, abort -- is the one
// the single envelope flies. What this file adds is the sequence:
//
//   go_to_start           the letter's PICK START, the letter + pick_start_offset
//                         (in the letter's frame); no offset = Start
//   execute_pick          the letter's hook pick (its exit climbs off the hat)
//   go_to_place_start     its box's Place Start (the box + place_start_offset)
//   execute_place         the insert into its box (its exit backs out and climbs)
//   ... the next letter from that exit's clearance point, with no retreat ...
//   go_to_land_start, execute_land   after the LAST letter only
//
// In FLIGHT the operator chooses the letter (the arm GS's Mail Route tab, one
// button per letter, 2026-11-03, user design); planMailRoute's dry run flies them
// 1, 2, 3. Nearest hat first is the safe order: every transit to a box and back
// then passes only emptied hats (a waiting 60 g letter under the rotors can blow
// off).

#ifndef FSC_TRAJECTORY_PLANNER_MAIL_ROUTE_HPP_
#define FSC_TRAJECTORY_PLANNER_MAIL_ROUTE_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/pick_place.hpp"
#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

struct MailRouteTargets
{
  // the mission's base poses, already shifted by Adjust (pick_place_start,
  // _land_start, _land)
  BasePose start, land_start, land;
  // each letter's pick point [m] (the claw target before the EE offset, like
  // PickPlaceTargets::pick) and its yaw [rad], in flying order
  std::vector<Vec3> letters;
  std::vector<double> letter_yaw;
  // each mailbox's place point [m] and yaw [rad] (PickPlaceTargets::place)
  std::vector<Vec3> boxes;
  std::vector<double> box_yaw;
  // a box's Place Start = the box + this [dx, dy, dz, dyaw_deg], dx and dy
  // turned by the box's yaw (pick_place_place_start_offset). Required: two
  // boxes cannot share one typed Place Start.
  std::vector<double> place_start_offset;
  // a letter's PICK START = the letter + this [dx, dy, dz, dyaw_deg], turned by
  // its yaw; every letter's go_to_start flies there. [] = to Start.
  std::vector<double> pick_start_offset;
};

// The box letter `letter` (0-based) is posted into: alternating over the
// boxes, letter 0 into box 0.
int mailRouteBox(int letter, int n_boxes);

// A pose + [dx, dy, dz, dyaw_deg], dx and dy turned by the pose's yaw (the
// Place Start / Pick Start rule).
BasePose mailRouteOffsetPose(const Vec3 & p, double yaw, const std::vector<double> & d);

// Each letter's six-leg targets. Throws std::runtime_error naming what is
// missing or malformed (no letter, no box, an offset of the wrong size).
std::vector<PickPlaceTargets> mailRouteLetterTargets(const MailRouteTargets & t);

// Each letter's waypoints (pickPlaceWaypoints). Throws "letter N: <leg>:
// <reason>" -- and refuses a config that is not the insert place.
std::vector<PickPlaceWaypoints> mailRouteWaypoints(
  const VehicleModel & v, const PickPlaceConfig & c, const MailRouteTargets & t);

// THE HATS ARE OBSTACLES (2026-11-03). The pick pose holds the claw nearly
// level with the body origin (~0.07 m below it), so at a pick the gear hangs
// ~0.18 m below the hat tops, 0.3-0.4 m behind the claw: a hat 0.5 m behind
// the one picked from is struck, an emptied one included. Every leg of the
// route is therefore checked against every hat, the airframe level at the
// reference's heading, as four bodies around the body origin (measured off
// AM_T650.usda's /body mesh and rotor joints, 2026-11-03); a hat is a column
// from the floor up to its top, so a part only has to stay clear sideways or
// above it. Not modelled: the arm's links between the body and the claw.
struct HatClearance
{
  // the skids and struts: a box, half-extents along the nose / sideways
  double gear_half_nose{0.163}, gear_half_side{0.168}, gear_bottom{-0.243};
  // the frame plates, arms and motors from the plates up: their OUTLINE, the
  // largest radius [m] in each 10 deg sector from the nose (measured: the plates
  // ~0.08 m all round, the four diagonal arms out to the motors at 0.35 m; a
  // square box was 3x too wide straight ahead and refused the mail pick's own hat)
  std::array<double, 36> frame_outline{{
      0.081, 0.084, 0.091, 0.154, 0.350, 0.154, 0.092, 0.085, 0.080,
      0.081, 0.084, 0.092, 0.154, 0.350, 0.154, 0.091, 0.084, 0.081,
      0.081, 0.085, 0.091, 0.154, 0.350, 0.154, 0.092, 0.084, 0.081,
      0.080, 0.085, 0.092, 0.154, 0.350, 0.154, 0.091, 0.085, 0.081}};
  double frame_bottom{-0.05};
  // the four 15" prop discs, centred on the motors (+-0.2299, +-0.2299) m, their
  // plane 0.068 m up: where the bottom of the blades sweeps
  double rotor_offset{0.2299}, prop_radius{0.19}, prop_bottom{0.05};
  // the jaws, and a letter hanging nose-down from them
  double claw_radius{0.10}, claw_drop{0.08};
  // a hat: its disc's radius, and what lies on it above its top
  double hat_radius{0.1003}, hat_cap{0.02};
  double margin{0.03};
};

// Where each hat stands -- its centre (x, y) and its TOP (z) -- from the
// letters and the pick LOCATOR (pick_place_pick_locator: the letter = the
// localization frame on the hat + Rz(yaw) [dx, dy, dz], so the frame, i.e. the
// hat, is the letter - Rz(yaw) [dx, dy, 0]; the hat top is the letter's
// height). An empty locator puts each hat at its letter.
std::vector<Vec3> mailRouteHats(const MailRouteTargets & t, const std::vector<double> & pick_locator);

// "" when no sample of `traj` brings the gear, the rotors or the claw within
// the margin of a hat; else the first conflict, in words, starting with
// `what`. `own` = the hat the claw is MEANT to reach on this motion (the
// letter's own pick and its exit), exempt for the claw only; -1 = none.
std::string hatConflict(
  const VehicleModel & v, const Trajectory & traj, const std::vector<Vec3> & hats, int own,
  const HatClearance & h, const std::string & what);

// THE CRUISE FLOOR: the body height at which the gear clears every hat top
// (plus what lies on it and the margin). -inf without hats.
double mailRouteCruiseFloor(const std::vector<Vec3> & hats, const HatClearance & h);

// One leg of the route -- what the node flies and planMailRoute chains:
// planPickPlaceLeg, except that a TRANSIT leg (go_to_start, go_to_place_start,
// go_to_land_start) starting below `floor` first climbs straight up to it,
// heading and arm unchanged (after a pick exit the vehicle turns 90 deg toward
// the boxes beside the hat it just emptied, 0.2 m above it: the gear's corner
// swept 4 cm over its top). Throws "<leg>: <reason>".
std::unique_ptr<Trajectory> planMailRouteLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from, bool from_table, double floor);

struct MailRoutePlan
{
  // per letter, indexed by leg: legs 0..3 for every letter, 4..5 for the last
  // (null elsewhere); exits on the two claw legs
  std::vector<std::array<std::shared_ptr<const Trajectory>, kNumPickPlaceLegs>> legs, exits;
  // the whole route back to back (each exit after its leg): what Plan draws
  std::shared_ptr<const Trajectory> whole;
};

// The Plan button's dry run of the whole route, the letters in order: letter
// 1's go_to_start from `from`, every later leg from the one before, each claw leg followed by its
// exit, each letter after the first from the clearance point the previous
// letter's place exit ended on. With `hats` (mailRouteHats) every leg and exit
// is checked against them (hatConflict). Throws "letter N: <leg>: <reason>" at
// the first infeasible leg.
MailRoutePlan planMailRoute(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const std::vector<PickPlaceWaypoints> & wps, const RestSpec & from,
  const std::vector<Vec3> * hats = nullptr, const HatClearance & hc = HatClearance{});

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_MAIL_ROUTE_HPP_
