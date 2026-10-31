// MIT License
// Copyright (c) 2026 FSC Lab
//
// Pick-and-place goals and legs (see pick_place.hpp).
#include "fsc_trajectory_planner/pick_place.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fsc_trajectory_planner
{

namespace
{

constexpr double kDeg = 180.0 / M_PI;

RestSpec baseRest(const BasePose & b, const VecN & q)
{
  RestSpec r;
  r.x_b = b.p;
  r.phi = b.yaw - 0.5 * M_PI;   // actual yaw -> model heading
  r.q = q;
  return r;
}

// The rest that puts the claw on `ee` with the arm in pose q, the arm's
// horizontal reach lying on the bearing from `from` to `ee` (else at the
// fallback heading).
RestSpec clawRest(
  const VehicleModel & v, const VecN & q, const Vec3 & ee, const Vec3 & from,
  bool face, double fallback_phi)
{
  Vec3 r0e;
  armKinematics(q, v.params, nullptr, &r0e, nullptr);
  RestSpec r;
  r.q = q;
  r.phi = fallback_phi;
  const double dx = ee(0) - from(0), dy = ee(1) - from(1);
  if (face && std::hypot(dx, dy) > 1e-3) {
    const double bearing = std::atan2(dy, dx);
    // the reach's azimuth in the base frame; an arm folded straight down has
    // none, and then the NOSE (model +y) faces the target
    const double reach = std::hypot(r0e(0), r0e(1));
    const double alpha = reach > 1e-3 ? std::atan2(r0e(1), r0e(0)) : 0.5 * M_PI;
    r.phi = bearing - alpha;
  }
  r.x_b = ee - Rz(r.phi) * r0e;
  return r;
}

void checkPose(const VehicleModel & v, const VecN & q, const char * what, int leg)
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(1);
  for (int j = 0; j < kNumJoints; ++j) {
    if (q(j) < v.q_min(j) - 1e-9 || q(j) > v.q_max(j) + 1e-9) {
      m << pickPlaceLegName(leg) << ": " << what << " q" << j + 1 << " = " << q(j) * kDeg
        << " deg is outside [" << v.q_min(j) * kDeg << ", " << v.q_max(j) * kDeg << "]";
      throw std::runtime_error(m.str());
    }
  }
  const double s = sigmaNd(q, v.params);
  if (s < v.sigma_nd_margin) {
    m.precision(3);
    m << pickPlaceLegName(leg) << ": " << what << " sigma_nd = " << s << " < "
      << v.sigma_nd_margin << " -- too close to a singularity";
    throw std::runtime_error(m.str());
  }
}

// How far past a half turn a typed turn is still flown as given [rad].
constexpr double kTurnKeep = M_PI + 30.0 * M_PI / 180.0;

double wrapAngle(double a) {return std::atan2(std::sin(a), std::cos(a));}

// The heading leg `leg` ends on, on the 2*pi branch that makes its turn the
// one the WAYPOINTS ask for (2026-10-03, user: typed 0 -> 180 -> 360 deg turns
// the same way round twice, never twisting back): the change from the previous
// leg's goal heading, kept as given up to kTurnKeep (so a measured pick yaw a
// hair past the half turn does not flip the direction), the short way beyond
// it and for the first leg; added to where the vehicle actually is (a re-fly
// or a leg after ABORT starts elsewhere).
double legHeading(const PickPlaceWaypoints & wp, int leg, const RestSpec & from)
{
  const double goal = wp.goal[leg].phi;
  double d = leg == 0 ? wrapAngle(goal - from.phi) : goal - wp.goal[leg - 1].phi;
  if (std::abs(d) > kTurnKeep) {d = wrapAngle(d);}
  return goal + 2.0 * M_PI * std::round((from.phi + d - goal) / (2.0 * M_PI));
}

std::unique_ptr<Trajectory> transitionLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const RestSpec & a, const RestSpec & b)
{
  PlanRequest req;
  req.rest0 = a;
  req.rest1 = b;
  return transition.plan(v, req, opts);
}

}  // namespace

const char * pickPlaceLegName(int leg)
{
  static const char * const names[kNumPickPlaceLegs] = {
    "go_to_start", "execute_pick", "go_to_place_start", "execute_place",
    "go_to_land_start", "execute_land"};
  return leg >= 0 && leg < kNumPickPlaceLegs ? names[leg] : "?";
}

bool pickPlaceLegIsClaw(int leg) {return leg == kExecutePick || leg == kExecutePlace;}

bool pickPlaceLegRetreats(int leg) {return leg == kGoToPlaceStart || leg == kGoToLandStart;}

Vec3 clawOf(const VehicleModel & v, const RestSpec & r)
{
  Vec3 r0e;
  armKinematics(r.q, v.params, nullptr, &r0e, nullptr);
  return r.x_b + Rz(r.phi) * r0e;
}

std::array<double, 6> Geofence::excess(const Vec3 & body) const
{
  return {std::max(0.0, min_z - body(2)), std::max(0.0, body(2) - max_z),
    std::max(0.0, x_min - body(0)), std::max(0.0, body(0) - x_max),
    std::max(0.0, y_min - body(1)), std::max(0.0, body(1) - y_max)};
}

std::string Geofence::describe(const Vec3 & body, int bound) const
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(2);
  switch (bound) {
    case 0: m << "body z " << body(2) << " m below the fence floor " << min_z << " m"; break;
    case 1: m << "body z " << body(2) << " m above the fence ceiling " << max_z << " m"; break;
    default:
      m << "body (" << body(0) << ", " << body(1) << ") m outside the fence box x [" << x_min
        << ", " << x_max << "], y [" << y_min << ", " << y_max << "]";
  }
  return m.str();
}

std::string Geofence::violation(const Vec3 & body) const
{
  const auto e = excess(body);
  for (int k = 0; k < 6; ++k) {
    if (e[k] > 0.0) {return describe(body, k);}
  }
  return "";
}

PickPlaceWaypoints pickPlaceWaypoints(
  const VehicleModel & v, const PickPlaceConfig & c, const PickPlaceTargets & t)
{
  checkPose(v, c.home, "home pose", kGoToStart);
  checkPose(v, c.pick_pose, "pick pose", kExecutePick);
  checkPose(v, c.carry, "carry pose", kGoToPlaceStart);
  checkPose(v, c.place_pose, "place pose", kExecutePlace);
  if (c.place_insert && c.place_drop) {
    throw std::runtime_error(std::string(pickPlaceLegName(kExecutePlace)) +
      ": the insert place and the drop place exclude each other (pick_place_place_insert, "
      "pick_place_place_drop)");
  }
  if (c.place_insert && !(c.place_insert_pitch_deg >= 0.0 && c.place_insert_pitch_deg <= 60.0)) {
    throw std::runtime_error(std::string(pickPlaceLegName(kExecutePlace)) +
      ": the insertion line's pitch must be 0..60 deg down (pick_place_place_insert_pitch_deg)");
  }
  if (c.place_insert && !(c.place_approach_back > 0.0)) {
    throw std::runtime_error(std::string(pickPlaceLegName(kExecutePlace)) +
      ": the insert place needs pick_place_place_approach_back > 0 (the arm GS's Insert Margin)");
  }
  PickPlaceWaypoints wp;
  wp.goal[kGoToStart] = baseRest(t.start, c.home);
  // each claw target = its 4-D pose + the ONE ee_offset in that pose's frame;
  // the drone's yaw there = that pose's yaw (align_yaw), else facing it
  const bool face = c.face_target && !c.align_yaw;
  wp.goal[kExecutePick] = clawRest(
    v, c.pick_pose, t.pick + Rz(t.pick_yaw) * c.ee_offset, wp.goal[kGoToStart].x_b, face,
    c.align_yaw ? t.pick_yaw - 0.5 * M_PI : wp.goal[kGoToStart].phi);
  wp.goal[kGoToPlaceStart] = baseRest(t.place_start, c.carry);
  // the place: the same offset, in the typed (or captured) place pose's frame;
  // a DROP place ends drop_height above that (0 = on it), where the gripper opens
  if (c.place_drop && !(c.drop_height >= 0.0)) {
    throw std::runtime_error(std::string(pickPlaceLegName(kExecutePlace)) +
      ": the drop margin is negative (pick_place_drop_height, the arm GS's Drop Margin)");
  }
  const Vec3 drop_up{0.0, 0.0, c.place_drop ? c.drop_height : 0.0};
  wp.goal[kExecutePlace] = clawRest(
    v, c.place_pose, t.place + Rz(t.place_yaw) * c.ee_offset + drop_up, wp.goal[kGoToPlaceStart].x_b,
    face, c.align_yaw ? t.place_yaw - 0.5 * M_PI : wp.goal[kGoToPlaceStart].phi);
  if (c.place_drop && c.drop_altitude > 0.0) {
    // the drop phase at ONE altitude: the drop rest's base is raised to
    // drop_altitude; the release point (place point + EE offset + drop height)
    // is only its floor -- a release point above it is refused
    const double dz = c.drop_altitude - wp.goal[kExecutePlace].x_b(2);
    if (dz < -1e-9) {
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(3);
      m << pickPlaceLegName(kExecutePlace) << ": the drop altitude " << c.drop_altitude
        << " m puts the drop " << -dz << " m below the release point (place point + EE offset + drop "
        << "height) -- raise pick_place_drop_altitude";
      throw std::runtime_error(m.str());
    }
    wp.goal[kExecutePlace].x_b(2) += dz;
  }
  wp.goal[kGoToLandStart] = baseRest(t.land_start, c.home);
  wp.goal[kExecuteLand] = baseRest(t.land, c.home);
  for (int k = 0; k < kNumPickPlaceLegs; ++k) {
    wp.ee[k] = clawOf(v, wp.goal[k]);
    wp.approach[k] = wp.goal[k];
    if (k == kExecutePick && c.pick_approach_back > 0.0) {
      // the hook grasp: behind the target at its own height (model +y = the nose)
      const double phi = wp.goal[k].phi;
      wp.approach[k].x_b -= c.pick_approach_back * Vec3{-std::sin(phi), std::cos(phi), 0.0};
    } else if (k == kExecutePlace && c.place_insert) {
      // the insert: behind the mouth along the insertion line, slid in along it --
      // level at pitch 0, else back AND up (the diagonal insert, 2026-10-30)
      const double phi = wp.goal[k].phi;
      const double th = c.place_insert_pitch_deg * M_PI / 180.0;
      wp.approach[k].x_b -= c.place_approach_back *
        Vec3{-std::sin(phi) * std::cos(th), std::cos(phi) * std::cos(th), -std::sin(th)};
    } else if (k == kExecutePlace && c.place_drop) {
      // the drop: flown straight to the drop point (approach == goal), so the
      // leg completes there with no wait and no descent
    } else if (pickPlaceLegIsClaw(k)) {
      wp.approach[k].x_b(2) += c.approach_dz;
    }
    wp.approach_ee[k] = clawOf(v, wp.approach[k]);
    for (const RestSpec * r : {&wp.goal[k], &wp.approach[k]}) {
      const std::string why = c.fence.violation(r->x_b);
      if (!why.empty()) {
        throw std::runtime_error(
          std::string(pickPlaceLegName(k)) + (r == &wp.goal[k] ? " goal: " : " approach: ") + why);
      }
    }
  }
  return wp;
}

RestSpec retreatRest(const RestSpec & from, const PickPlaceConfig & c)
{
  RestSpec r = from;
  const Vec3 nose{-std::sin(from.phi), std::cos(from.phi), 0.0};   // model +y, world
  r.x_b += Vec3{0.0, 0.0, c.retreat_dz} - c.retreat_back * nose;
  return r;
}

void checkPathInFence(
  const VehicleModel & v, const Trajectory & traj, const Geofence & fence, const std::string & what)
{
  // the body origin along the reference, level-attitude approximation (the
  // planned tilts are a degree or two): x_b = x_cd - Rz(psi) r_0c(q_d)
  const auto body_at = [&](double t) {
      const WbReference ref = traj.eval(t);
      Vec3 r0c;
      armKinematics(ref.q_d, v.params, &r0c, nullptr, nullptr);
      return Vec3(ref.x_cd - Rz(std::atan2(ref.b1_d(1), ref.b1_d(0))) * r0c);
    };
  // A leg may START outside (DIRECT engaged low, an abort hold over the
  // ceiling): it must then come back in, so per bound no sample may lie
  // further outside than the start (1 mm slack); inside, it must stay inside.
  const auto allowed = fence.excess(body_at(0.0));
  const int n = 200;
  for (int k = 0; k <= n; ++k) {
    const double t = traj.duration() * k / n;
    const Vec3 body = body_at(t);
    const auto e = fence.excess(body);
    for (int b = 0; b < 6; ++b) {
      if (e[b] <= (allowed[b] > 0.0 ? allowed[b] + 1e-3 : 0.0)) {continue;}
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(1);
      m << what << ": the path leaves the geofence at t = " << t << " s (" << fence.describe(body, b)
        << (allowed[b] > 0.0 ? ", further out than where the leg starts" : "") << ")";
      throw std::runtime_error(m.str());
    }
  }
}

std::unique_ptr<Trajectory> planPickPlaceLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from, bool from_table)
{
  if (leg < 0 || leg >= kNumPickPlaceLegs) {
    throw std::runtime_error("pick-and-place: no leg " + std::to_string(leg));
  }
  // a claw leg ends ABOVE its target; the descent is its own plan. Its
  // heading is on the branch the waypoints' turn asks for, flown as given.
  RestSpec target = wp.approach[leg];
  target.phi = legHeading(wp, leg, from);
  PlanOptions turn = opts;
  turn.heading_as_given = true;
  std::unique_ptr<Trajectory> traj;
  try {
    std::vector<std::shared_ptr<const Trajectory>> segs;
    RestSpec start = from;
    if (from_table && (c.retreat_dz != 0.0 || c.retreat_back != 0.0)) {
      const RestSpec via = retreatRest(from, c);
      segs.push_back(transitionLeg(v, transition, turn, from, via));
      start = via;
    }
    if (leg == kExecutePlace && c.place_drop && std::abs(target.x_b(2) - start.x_b(2)) > 0.01) {
      // the DROP from a Place Start at another altitude: ONE smooth curve whose
      // altitude change is done in the first drop_climb_lead of the leg --
      // whatever sets the drop's height (the drop margin, 2026-10-15, or a
      // legacy drop_altitude), so the payload never crosses the container's
      // wall on a climbing diagonal
      ArmSweepOptions o = c.sweep;
      o.z_lead = std::min(1.0, std::max(0.2, c.drop_climb_lead));
      segs.push_back(ArmSweepPlanner::plan(v, start, target, o));
    } else if (leg == kExecutePlace) {
      segs.push_back(ArmSweepPlanner::plan(v, start, target, c.sweep));
    } else {
      segs.push_back(transitionLeg(v, transition, turn, start, target));
    }
    traj = std::make_unique<SequenceTrajectory>(std::move(segs));
  } catch (const std::exception & e) {
    throw std::runtime_error(std::string(pickPlaceLegName(leg)) + ": " + e.what());
  }
  checkPathInFence(v, *traj, c.fence, pickPlaceLegName(leg));
  return traj;
}

namespace
{
// A TIMED static hold (HoldTrajectory has no duration): the claw settling
// between the trim-first sideways move and the vertical descent.
class PauseTrajectory : public Trajectory
{
public:
  PauseTrajectory(const WholeBodyParams & p, const RestSpec & rest, double T)
  : rest_(rest), ref_(restReference(p, rest)), T_(T)
  {
    diag_.T = T;
    diag_.min_sigma_nd = sigmaNd(rest.q, p);
    diag_.q_min_deg = rest.q * (180.0 / M_PI);
    diag_.q_max_deg = diag_.q_min_deg;
    std::ostringstream m;
    m << "settle " << std::fixed << std::setprecision(1) << T << " s";
    diag_.summary = m.str();
  }
  double duration() const override {return T_;}
  WbReference eval(double) const override {return ref_;}
  RestSpec goalRest() const override {return rest_;}
  const PlanDiag & diag() const override {return diag_;}

private:
  RestSpec rest_;
  WbReference ref_;
  double T_;
  PlanDiag diag_;
};

}  // namespace

std::unique_ptr<Trajectory> planPickPlaceDescent(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from)
{
  const std::string what = std::string(pickPlaceLegName(leg)) + " descent";
  if (!pickPlaceLegIsClaw(leg)) {throw std::runtime_error(what + ": not a claw leg");}
  if (leg == kExecutePlace && c.place_drop) {
    throw std::runtime_error(what + ": a drop place has no descent (the gripper opens on the drop point)");
  }
  // the goal carries the descent trim; from is the hold above the target
  const RestSpec & goal = wp.goal[leg];
  const double side = (goal.x_b - from.x_b).head<2>().norm();
  std::unique_ptr<Trajectory> traj;
  try {
    if ((leg == kExecutePick && c.pick_approach_back > 0.0) || (leg == kExecutePlace && c.place_insert)) {
      // the hook grasp's / the insert's approach is already on the line into the
      // goal (level, or the insert's pitched line): fly straight in
      traj = transitionLeg(v, transition, opts, from, goal);
    } else if (c.trim_settle_s > 0.0 && side > c.trim_first_min) {
      // (a trim at or below trim_first_min is flown in the descent itself --
      // the whole-body law's, 1-7 mm, with the default 5 mm)
      // TRIM FIRST: sideways at the hover height, settle, then straight down
      RestSpec above = from;
      above.x_b.head<2>() = goal.x_b.head<2>();
      std::vector<std::shared_ptr<const Trajectory>> segs{
        transitionLeg(v, transition, opts, from, above),
        std::make_shared<PauseTrajectory>(v.params, above, c.trim_settle_s),
        transitionLeg(v, transition, opts, above, goal)};
      traj = std::make_unique<SequenceTrajectory>(std::move(segs));
    } else {
      traj = transitionLeg(v, transition, opts, from, goal);
    }
  } catch (const std::exception & e) {
    throw std::runtime_error(what + ": " + e.what());
  }
  checkPathInFence(v, *traj, c.fence, what);
  return traj;
}

std::unique_ptr<Trajectory> planPickPlaceAbort(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const RestSpec & from, const RestSpec & goal, const VecN * release_q, double hold_s,
  const RestSpec * backout)
{
  std::vector<std::shared_ptr<const Trajectory>> segs;
  const RestSpec * start = &from;
  if (backout != nullptr) {
    try {
      segs.push_back(transitionLeg(v, transition, opts, from, *backout));
    } catch (const std::exception & e) {
      throw std::runtime_error(std::string("backing out: ") + e.what());
    }
    start = backout;
  }
  if (release_q == nullptr) {
    segs.push_back(transitionLeg(v, transition, opts, *start, goal));
    return std::make_unique<SequenceTrajectory>(std::move(segs));
  }
  // the climbed point, arm in the release pose
  RestSpec release = goal;
  release.q = *release_q;
  try {
    segs.push_back(transitionLeg(v, transition, opts, *start, release));
  } catch (const std::exception & e) {
    throw std::runtime_error(std::string("to the release pose: ") + e.what());
  }
  if (hold_s > 0.0) {
    segs.push_back(std::make_shared<PauseTrajectory>(v.params, release, hold_s));
  }
  try {
    segs.push_back(transitionLeg(v, transition, opts, release, goal));
  } catch (const std::exception & e) {
    throw std::runtime_error(std::string("release pose to home: ") + e.what());
  }
  return std::make_unique<SequenceTrajectory>(std::move(segs));
}

double pickPlaceExitBack(const PickPlaceConfig & c)
{
  return c.place_drop && c.drop_exit_back >= 0.0 ? c.drop_exit_back : c.place_exit_back;
}

bool pickPlaceExitEndsAtLandStart(const PickPlaceConfig & c, int leg)
{
  return leg == kExecutePlace && c.place_drop && c.drop_exit_to_land_start;
}

std::unique_ptr<Trajectory> planPickPlaceExit(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c, int leg, const RestSpec & from,
  const PickPlaceWaypoints * wp)
{
  const std::string what = std::string(pickPlaceLegName(leg)) + " exit";
  if (!pickPlaceLegIsClaw(leg)) {throw std::runtime_error(what + ": not a claw leg");}
  if (pickPlaceExitEndsAtLandStart(c, leg)) {
    // the DROP's exit STRAIGHT ON TO LAND START (2026-10-15, user design): a
    // little straight up off the drop point (the take-off), then on to Land
    // Start -- Go To Land Start's own goal and heading, the arm folding home on
    // the way. No hover at a drop exit point.
    if (wp == nullptr) {throw std::runtime_error(what + " (to Land Start): no waypoints");}
    RestSpec up = from;
    up.x_b(2) += std::max(0.0, c.drop_exit_climb);
    RestSpec to = wp->goal[kGoToLandStart];
    const bool climbs = up.x_b(2) - from.x_b(2) > 1e-3;
    to.phi = legHeading(*wp, kGoToLandStart, climbs ? up : from);
    for (const RestSpec * r : {&up, &to}) {
      const std::string why_r = c.fence.violation(r->x_b);
      if (!why_r.empty()) {throw std::runtime_error(what + " (to Land Start): " + why_r);}
    }
    PlanOptions turn = opts;
    turn.heading_as_given = true;
    std::unique_ptr<Trajectory> traj;
    try {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (climbs) {segs.push_back(transitionLeg(v, transition, opts, from, up));}
      segs.push_back(transitionLeg(v, transition, turn, climbs ? up : from, to));
      traj = std::make_unique<SequenceTrajectory>(std::move(segs));
    } catch (const std::exception & e) {
      throw std::runtime_error(what + " (to Land Start): " + e.what());
    }
    checkPathInFence(v, *traj, c.fence, what);
    return traj;
  }
  if (leg == kExecutePlace && c.place_drop && c.drop_exit_set) {
    // the DROP EXIT: a little straight up from the drop point, then LEVEL out
    // of the place region to the exit pose; the arm keeps its place pose
    RestSpec up = from;
    up.x_b(2) += std::max(0.0, c.drop_exit_climb);
    RestSpec to = baseRest(c.drop_exit, from.q);
    to.x_b(2) = up.x_b(2);
    for (const RestSpec * r : {&up, &to}) {
      const std::string why_r = c.fence.violation(r->x_b);
      if (!why_r.empty()) {throw std::runtime_error(what + " (drop exit): " + why_r);}
    }
    PlanOptions turn = opts;
    turn.heading_as_given = true;
    std::unique_ptr<Trajectory> traj;
    try {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (up.x_b(2) - from.x_b(2) > 1e-3) {segs.push_back(transitionLeg(v, transition, opts, from, up));}
      segs.push_back(transitionLeg(v, transition, turn, up.x_b(2) - from.x_b(2) > 1e-3 ? up : from, to));
      traj = std::make_unique<SequenceTrajectory>(std::move(segs));
    } catch (const std::exception & e) {
      throw std::runtime_error(what + " (drop exit): " + e.what());
    }
    checkPathInFence(v, *traj, c.fence, what);
    return traj;
  }
  if (!(c.approach_dz > 0.0)) {throw std::runtime_error(what + ": the vertical margin is 0");}
  RestSpec up = from;
  up.x_b(2) += c.approach_dz;
  const std::string why = c.fence.violation(up.x_b);
  if (!why.empty()) {throw std::runtime_error(what + ": " + why);}
  PlanOptions o = opts;
  if (c.exit_t > 0.0) {
    // a rest-to-rest min-snap climb of dz in T peaks at 2.19 dz / T [m/s] and
    // 7.51 dz / T^2 [m/s^2]; open the bounds to that (30 % margin for the
    // B-spline's own profile) and let T_min set the duration
    o.T_min = c.exit_t;
    o.v_max = std::max(o.v_max, 1.3 * 2.19 * c.approach_dz / c.exit_t);
    o.a_max = std::max(o.a_max, 1.3 * 7.51 * c.approach_dz / (c.exit_t * c.exit_t));
  }
  std::unique_ptr<Trajectory> traj;
  try {
    const double back_d = pickPlaceExitBack(c);
    if (leg == kExecutePlace && back_d > 0.0) {
      // the hook grasp: slide the fingers out from under the arch, then climb;
      // a drop: back out the Back Margin, out of the container's downwash
      RestSpec back = from;
      back.x_b -= back_d * Vec3{-std::sin(from.phi), std::cos(from.phi), 0.0};
      up.x_b.head<2>() = back.x_b.head<2>();
      const std::string why_back = c.fence.violation(up.x_b);
      if (!why_back.empty()) {throw std::runtime_error(why_back);}
      std::vector<std::shared_ptr<const Trajectory>> segs{
        transitionLeg(v, transition, opts, from, back),
        transitionLeg(v, transition, o, back, up)};
      traj = std::make_unique<SequenceTrajectory>(std::move(segs));
    } else {
      traj = transitionLeg(v, transition, o, from, up);
    }
  } catch (const std::exception & e) {
    throw std::runtime_error(what + ": " + e.what());
  }
  checkPathInFence(v, *traj, c.fence, what);
  return traj;
}

std::vector<std::shared_ptr<const Trajectory>> planPickPlaceMission(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, const RestSpec & from,
  std::array<std::shared_ptr<const Trajectory>, kNumPickPlaceLegs> * exits)
{
  std::vector<std::shared_ptr<const Trajectory>> legs;
  RestSpec r = from;
  bool exited = false;   // the leg before was a claw leg followed by its exit
  for (int k = 0; k < kNumPickPlaceLegs; ++k) {
    std::shared_ptr<const Trajectory> leg =
      planPickPlaceLeg(v, transition, opts, c, wp, k, r, pickPlaceLegRetreats(k) && !exited);
    exited = false;
    if (pickPlaceLegIsClaw(k) && !(k == kExecutePlace && c.place_drop)) {
      // approach, then the descent from the approach rest (a drop place has none)
      std::vector<std::shared_ptr<const Trajectory>> segs{
        leg, std::shared_ptr<const Trajectory>(
          planPickPlaceDescent(v, transition, opts, c, wp, k, wp.approach[k]))};
      leg = std::make_shared<SequenceTrajectory>(std::move(segs));
    }
    legs.push_back(leg);
    r = wp.goal[k];
    if (exits != nullptr && pickPlaceLegIsClaw(k)) {
      std::shared_ptr<const Trajectory> ex = planPickPlaceExit(v, transition, opts, c, k, r, &wp);
      r = ex->goalRest();
      (*exits)[k] = std::move(ex);
      exited = true;
      if (pickPlaceExitEndsAtLandStart(c, k)) {
        // the exit already flew Go To Land Start: that leg is a 0 s hold there
        legs.push_back(std::make_shared<PauseTrajectory>(v.params, r, 0.0));
        ++k;
        exited = false;
      }
    }
  }
  return legs;
}

}  // namespace fsc_trajectory_planner
