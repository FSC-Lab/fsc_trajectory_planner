// MIT License
// Copyright (c) 2026 FSC Lab
//
// Pick-and-place goals and legs (see pick_place.hpp).
#include "fsc_trajectory_planner/pick_place.hpp"

#include <cmath>
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

PickPlaceWaypoints pickPlaceWaypoints(
  const VehicleModel & v, const PickPlaceConfig & c, const PickPlaceTargets & t)
{
  checkPose(v, c.home, "home pose", kGoToStart);
  checkPose(v, c.pick_pose, "pick pose", kExecutePick);
  checkPose(v, c.carry, "carry pose", kGoToPlaceStart);
  checkPose(v, c.place_pose, "place pose", kExecutePlace);
  PickPlaceWaypoints wp;
  wp.goal[kGoToStart] = baseRest(t.start, c.home);
  wp.goal[kExecutePick] = clawRest(
    v, c.pick_pose, t.pick + c.pick_ee_offset, wp.goal[kGoToStart].x_b, c.face_target,
    wp.goal[kGoToStart].phi);
  wp.goal[kGoToPlaceStart] = baseRest(t.place_start, c.carry);
  wp.goal[kExecutePlace] = clawRest(
    v, c.place_pose, t.place + c.place_ee_offset, wp.goal[kGoToPlaceStart].x_b, c.face_target,
    wp.goal[kGoToPlaceStart].phi);
  wp.goal[kGoToLandStart] = baseRest(t.land_start, c.home);
  wp.goal[kExecuteLand] = baseRest(t.land, c.home);
  for (int k = 0; k < kNumPickPlaceLegs; ++k) {wp.ee[k] = clawOf(v, wp.goal[k]);}
  return wp;
}

RestSpec retreatRest(const RestSpec & from, const PickPlaceConfig & c)
{
  RestSpec r = from;
  const Vec3 nose{-std::sin(from.phi), std::cos(from.phi), 0.0};   // model +y, world
  r.x_b += Vec3{0.0, 0.0, c.retreat_dz} - c.retreat_back * nose;
  return r;
}

std::unique_ptr<Trajectory> planPickPlaceLeg(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, int leg, const RestSpec & from)
{
  if (leg < 0 || leg >= kNumPickPlaceLegs) {
    throw std::runtime_error("pick-and-place: no leg " + std::to_string(leg));
  }
  const RestSpec & goal = wp.goal[leg];
  try {
    if (leg == kExecutePlace) {
      return ArmSweepPlanner::plan(v, from, goal, c.sweep);
    }
    if (pickPlaceLegRetreats(leg) && (c.retreat_dz != 0.0 || c.retreat_back != 0.0)) {
      const RestSpec via = retreatRest(from, c);
      std::vector<std::shared_ptr<const Trajectory>> segs;
      segs.push_back(transitionLeg(v, transition, opts, from, via));
      segs.push_back(transitionLeg(v, transition, opts, via, goal));
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    }
    return transitionLeg(v, transition, opts, from, goal);
  } catch (const std::exception & e) {
    throw std::runtime_error(std::string(pickPlaceLegName(leg)) + ": " + e.what());
  }
}

std::vector<std::shared_ptr<const Trajectory>> planPickPlaceMission(
  const VehicleModel & v, const TrajectoryPlanner & transition,
  const PlanOptions & opts, const PickPlaceConfig & c,
  const PickPlaceWaypoints & wp, const RestSpec & from)
{
  std::vector<std::shared_ptr<const Trajectory>> legs;
  RestSpec r = from;
  for (int k = 0; k < kNumPickPlaceLegs; ++k) {
    legs.push_back(planPickPlaceLeg(v, transition, opts, c, wp, k, r));
    r = wp.goal[k];
  }
  return legs;
}

}  // namespace fsc_trajectory_planner
