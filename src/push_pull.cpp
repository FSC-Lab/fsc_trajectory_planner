// MIT License
// Copyright (c) 2026 FSC Lab
//
// Push-and-pull goals and steps (see push_pull.hpp).
#include "fsc_trajectory_planner/push_pull.hpp"

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

double wrapAngle(double a) {return std::atan2(std::sin(a), std::cos(a));}

void checkPose(const VehicleModel & v, const VecN & q, const char * what, int step)
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(1);
  for (int j = 0; j < kNumJoints; ++j) {
    if (q(j) < v.q_min(j) - 1e-9 || q(j) > v.q_max(j) + 1e-9) {
      m << pushPullStepName(step) << ": " << what << " q" << j + 1 << " = " << q(j) * kDeg
        << " deg is outside [" << v.q_min(j) * kDeg << ", " << v.q_max(j) * kDeg << "]";
      throw std::runtime_error(m.str());
    }
  }
  const double s = sigmaNd(q, v.params);
  if (s < v.sigma_nd_margin) {
    m.precision(3);
    m << pushPullStepName(step) << ": " << what << " sigma_nd = " << s << " < "
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

// The options for a rest-to-rest move of `dist` metres that must take at
// least T seconds: T_min = T and the translational bounds opened just enough
// for it (a min-snap move of d in T peaks at 2.19 d / T and 7.51 d / T^2; 30 %
// margin for the B-spline's own profile). The rotor and joint-torque bounds
// still apply. T <= 0: the options unchanged.
PlanOptions timedOptions(const PlanOptions & opts, double dist, double T)
{
  PlanOptions o = opts;
  if (T > 0.0) {
    o.T_min = T;
    o.v_max = std::max(o.v_max, 1.3 * 2.19 * std::abs(dist) / T);
    o.a_max = std::max(o.a_max, 1.3 * 7.51 * std::abs(dist) / (T * T));
  }
  return o;
}

// A TIMED static hold (HoldTrajectory has no duration): the settle before the
// push moves.
class PauseTrajectory : public Trajectory
{
public:
  PauseTrajectory(const WholeBodyParams & p, const RestSpec & rest, double T)
  : rest_(rest), ref_(restReference(p, rest)), T_(T)
  {
    diag_.T = T;
    diag_.min_sigma_nd = sigmaNd(rest.q, p);
    diag_.q_min_deg = rest.q * kDeg;
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

// The nose (actual body +x = model +y) of a model heading, world, horizontal.
Vec3 noseOf(double phi) {return Vec3{-std::sin(phi), std::cos(phi), 0.0};}

// A rest's checks on its own: fence, and the table at that one pose.
void checkRest(
  const VehicleModel & v, const PushPullConfig & c, const RestSpec & r, const char * what)
{
  const std::string why = c.fence.violation(r.x_b);
  if (!why.empty()) {throw std::runtime_error(std::string(what) + ": " + why);}
  HoldTrajectory hold(v.params, r);
  checkPathClearsTable(v, hold, c, what, false);
}

// Finish a step: wrap the planner's own error with the step's name, then the
// fence and table checks of the whole path.
template<typename Fn>
std::unique_ptr<Trajectory> step(
  const VehicleModel & v, const PushPullConfig & c, const std::string & what, Fn plan,
  bool table = true)
{
  std::unique_ptr<Trajectory> traj;
  try {
    traj = plan();
  } catch (const std::exception & e) {
    throw std::runtime_error(what + ": " + e.what());
  }
  checkPathInFence(v, *traj, c.fence, what);
  if (table) {checkPathClearsTable(v, *traj, c, what);}
  return traj;
}

}  // namespace

const char * pushPullStepName(int step)
{
  static const char * const names[kNumPushPullSteps] = {
    "go_to_start", "ready", "push", "exit", "go_to_land"};
  return step >= 0 && step < kNumPushPullSteps ? names[step] : "?";
}

double TableKeepOut::distanceXY(const Vec3 & p) const
{
  const double dx = std::max({x_min - p(0), 0.0, p(0) - x_max});
  const double dy = std::max({y_min - p(1), 0.0, p(1) - y_max});
  return std::hypot(dx, dy);
}

void checkPathClearsTable(
  const VehicleModel & v, const Trajectory & traj, const PushPullConfig & c,
  const std::string & what, bool allow_low_start)
{
  if (!c.table.valid()) {return;}
  // the parts checked: the landing gear (lowest point of the airframe) and the
  // arm chain's joints 2..4 and grasp point, level-attitude approximation
  // (the planned tilts are a degree or two)
  static const char * const kPart[5] = {"the landing gear", "arm joint 2", "arm joint 3",
    "arm joint 4", "the claw"};
  struct Sample {std::array<double, 5> low; std::array<double, 5> h;};
  const auto sample = [&](double t) {
      const WbReference ref = traj.eval(t);
      const double psi = std::atan2(ref.b1_d(1), ref.b1_d(0));
      Vec3 r0c;
      armKinematics(ref.q_d, v.params, &r0c, nullptr, nullptr);
      const Vec3 body = ref.x_cd - Rz(psi) * r0c;
      std::array<Vec3, 5> pts;
      armChainPoints(ref.q_d, v.params, &pts);
      Sample s;
      // gear: over the table when its footprint reaches the top's footprint
      const bool gear_over = c.table.distanceXY(body) < c.gear_radius + c.table_margin_xy;
      s.h[0] = body(2) - c.gear_depth - c.table.z_top;
      s.low[0] = gear_over ? std::max(0.0, c.table_clearance - s.h[0]) : 0.0;
      for (int i = 1; i < 5; ++i) {
        const Vec3 w = body + Rz(psi) * pts[i];
        s.h[i] = w(2) - c.table.z_top;
        const bool over = c.table.distanceXY(w) < c.table_margin_xy;
        s.low[i] = over ? std::max(0.0, c.table_clearance - s.h[i]) : 0.0;
      }
      return s;
    };
  // a step may START low (a re-fly from where an abort left it): then no
  // sample may be lower than the start (1 mm slack); clear, it must stay clear
  Sample s0 = sample(0.0);
  if (!allow_low_start) {s0.low.fill(0.0);}
  const int n = traj.duration() > 0.0 ? 200 : 0;
  for (int k = 0; k <= n; ++k) {
    const double t = n > 0 ? traj.duration() * k / n : 0.0;
    const Sample s = sample(t);
    for (int i = 0; i < 5; ++i) {
      if (s.low[i] <= (s0.low[i] > 0.0 ? s0.low[i] + 1e-3 : 0.0)) {continue;}
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(3);
      m << what << ": " << kPart[i] << " would be " << s.h[i] << " m above the table top ("
        << c.table.z_top << " m) at t = " << std::setprecision(1) << t << " s, over the table "
        << "-- it must keep " << std::setprecision(3) << c.table_clearance << " m"
        << (s0.low[i] > 0.0 ? " (and never go lower than where the step starts)" : "")
        << " -- check the table's pose (push_pull_table) and the box capture / EE offset";
      throw std::runtime_error(m.str());
    }
  }
}

PushPullWaypoints pushPullWaypoints(
  const VehicleModel & v, const PushPullConfig & c, const PushPullTargets & t)
{
  checkPose(v, c.home, "home pose", kPlExit);
  checkPose(v, c.push_pose, "push pose", kPlReady);
  if (!(c.approach_dz > 0.0)) {
    throw std::runtime_error("ready: the safety margin (approach_dz) must be > 0");
  }
  PushPullWaypoints wp;
  wp.drone_yaw = t.box_yaw + c.yaw_from_box;
  const double phi = wp.drone_yaw - 0.5 * M_PI;   // actual yaw -> model heading
  wp.push_dir = noseOf(phi);
  Vec3 r0e;
  armKinematics(c.push_pose, v.params, nullptr, &r0e, nullptr);
  // the grasp point: the box CG + the ONE offset, turned with the box
  wp.grasp_ee = t.box + Rz(t.box_yaw) * c.ee_offset;
  wp.grasp.phi = phi;
  wp.grasp.q = c.push_pose;
  wp.grasp.x_b = wp.grasp_ee - Rz(phi) * r0e;
  wp.approach = wp.grasp;
  wp.approach.x_b(2) += c.approach_dz;
  wp.approach_ee = clawOf(v, wp.approach);
  wp.pushed = wp.grasp;
  wp.pushed.x_b += c.push_distance * wp.push_dir;
  wp.pushed_ee = clawOf(v, wp.pushed);
  wp.lifted = wp.pushed;
  wp.lifted.x_b(2) += c.approach_dz;
  wp.exited = wp.lifted;
  wp.exited.q = c.home;
  wp.land.x_b = t.land.p;
  wp.land.phi = t.land.yaw - 0.5 * M_PI;
  wp.land.q = c.home;
  checkRest(v, c, wp.approach, "go_to_start goal");
  checkRest(v, c, wp.grasp, "ready goal");
  checkRest(v, c, wp.pushed, "push goal");
  checkRest(v, c, wp.lifted, "exit climb");
  checkRest(v, c, wp.exited, "exit goal");
  checkRest(v, c, wp.land, "go_to_land goal");
  return wp;
}

std::unique_ptr<Trajectory> planPushPullGoToStart(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  // the short way round to the push heading
  return step(v, c, "go_to_start", [&]() {
      return transitionLeg(v, transition, opts, from, wp.approach);
    });
}

std::unique_ptr<Trajectory> planPushPullReady(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from,
  const Vec3 & trim)
{
  RestSpec goal = wp.grasp;
  goal.x_b += trim;
  return step(v, c, "ready", [&]() {
      return transitionLeg(v, transition, opts, from, goal);
    });
}

std::unique_ptr<Trajectory> planPushPullPush(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from)
{
  RestSpec goal = from;
  goal.x_b += c.push_distance * noseOf(from.phi);
  return step(v, c, "push", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (c.push_settle > 0.0) {
        segs.push_back(std::make_shared<PauseTrajectory>(v.params, from, c.push_settle));
      }
      segs.push_back(transitionLeg(
        v, transition, timedOptions(opts, c.push_distance, c.push_time), from, goal));
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    });
}

std::unique_ptr<Trajectory> planPushPullExit(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from)
{
  RestSpec up = from;
  up.x_b(2) += c.approach_dz;
  RestSpec home = up;
  home.q = c.home;
  return step(v, c, "exit", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs{
        transitionLeg(v, transition, timedOptions(opts, c.approach_dz, c.exit_time), from, up)};
      if ((up.q - c.home).cwiseAbs().maxCoeff() > 1e-6) {
        segs.push_back(transitionLeg(v, transition, opts, up, home));
      }
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    });
}

std::unique_ptr<Trajectory> planPushPullGoToLand(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  // THE TURN AS TYPED (user, 2026-10-03: -90 -> -360 deg is 270 deg
  // CLOCKWISE, not the 90 deg the short way would take): from the push
  // heading the turn is the typed difference land yaw - drone yaw, unwrapped;
  // from any other heading (a re-fly, a leg after Abort elsewhere) the short way.
  RestSpec goal = wp.land;
  const double from_yaw = from.phi + 0.5 * M_PI;
  const bool at_push_heading = std::abs(wrapAngle(from_yaw - wp.drone_yaw)) < 30.0 / kDeg;
  const double turn = at_push_heading ? goal.phi + 0.5 * M_PI - wp.drone_yaw
                                      : wrapAngle(goal.phi - from.phi);
  goal.phi = from.phi + turn;
  PlanOptions o = opts;
  o.heading_as_given = true;
  return step(v, c, "go_to_land", [&]() {
      return transitionLeg(v, transition, o, from, goal);
    });
}

std::unique_ptr<Trajectory> planPushPullAbort(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const RestSpec & from, double climb, double release_wait)
{
  RestSpec up = from;
  up.x_b(2) += std::max(0.0, climb);
  RestSpec home = up;
  home.q = c.home;
  return step(v, c, "abort", [&]() -> std::unique_ptr<Trajectory> {
      std::vector<std::shared_ptr<const Trajectory>> segs;
      if (release_wait > 0.0) {
        segs.push_back(std::make_shared<PauseTrajectory>(v.params, from, release_wait));
      }
      if (climb > 1e-3) {segs.push_back(transitionLeg(v, transition, opts, from, up));}
      if ((up.q - c.home).cwiseAbs().maxCoeff() > 1e-6 ||
        (segs.size() == (release_wait > 0.0 ? 1u : 0u)))
      {
        segs.push_back(transitionLeg(v, transition, opts, up, home));
      }
      return std::make_unique<SequenceTrajectory>(std::move(segs));
    }, false);
}

std::vector<std::shared_ptr<const Trajectory>> planPushPullMission(
  const VehicleModel & v, const TrajectoryPlanner & transition, const PlanOptions & opts,
  const PushPullConfig & c, const PushPullWaypoints & wp, const RestSpec & from)
{
  std::vector<std::shared_ptr<const Trajectory>> steps;
  steps.push_back(planPushPullGoToStart(v, transition, opts, c, wp, from));
  steps.push_back(planPushPullReady(v, transition, opts, c, wp, wp.approach));
  steps.push_back(planPushPullPush(v, transition, opts, c, wp.grasp));
  steps.push_back(planPushPullExit(v, transition, opts, c, steps.back()->goalRest()));
  steps.push_back(planPushPullGoToLand(v, transition, opts, c, wp, steps.back()->goalRest()));
  return steps;
}

}  // namespace fsc_trajectory_planner
