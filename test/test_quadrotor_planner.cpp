// MIT License
// Copyright (c) 2026 FSC Lab
//
// quadrotor_model / quadrotor_trajectory: parity with the Python planner the
// CCM was trained and simulated against, dynamic consistency of the
// reference, and the ramp / stop / command bookkeeping.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/quadrotor_model.hpp"
#include "fsc_trajectory_planner/quadrotor_trajectory.hpp"

using fsc_trajectory_planner::CcmReferenceSample;
using fsc_trajectory_planner::makeQuadrotorModel;
using fsc_trajectory_planner::parseQuadShape;
using fsc_trajectory_planner::QuadShape;
using fsc_trajectory_planner::QuadTrajectory;
using fsc_trajectory_planner::QuadTrajectoryOptions;
using fsc_trajectory_planner::sampleCcmReference;

namespace
{
const Eigen::Vector3d kAnchor(0.3, -0.2, 1.1);
constexpr double kYaw = 0.4;

Eigen::Matrix3d hat(const Eigen::Vector3d & w)
{
  Eigen::Matrix3d W;
  W << 0, -w.z(), w.y(), w.z(), 0, -w.x(), -w.y(), w.x(), 0;
  return W;
}
}  // namespace

TEST(QuadrotorPlanner, RegistryKnowsT650AndRejectsUnknown)
{
  const auto m = makeQuadrotorModel("t650");
  EXPECT_NEAR(m.mass, 3.033921, 1e-12);
  EXPECT_NEAR(m.rotor_time_constant, 0.0997, 1e-12);
  EXPECT_THROW(makeQuadrotorModel("x999"), std::runtime_error);
}

TEST(QuadrotorPlanner, RegistryKnowsX650)
{
  const auto m = makeQuadrotorModel("x650");
  EXPECT_NEAR(m.mass, 3.25, 1e-12);
  EXPECT_NEAR(m.inertia(0, 0), 0.05777498, 1e-12);
  EXPECT_NEAR(m.inertia(0, 1), 0.0, 1e-12);
  EXPECT_NEAR(m.rotor_time_constant, 0.115, 1e-12);
}

// Python (CCM_quad planners/planner_quad_torque.py) produced these samples;
// the C++ reference must match them to numerical precision.
TEST(QuadrotorPlanner, MatchesPythonPlanner)
{
  std::ifstream f("data/python_quadrotor_ccm_reference.txt");
  ASSERT_TRUE(f.good()) << "fixture missing (run from the test/ directory)";
  const auto model = makeQuadrotorModel("t650");
  QuadTrajectoryOptions opts;  // defaults = the fixture's
  const QuadTrajectory circle(QuadShape::kCircle, opts, kAnchor);
  const QuadTrajectory fig8(QuadShape::kFigure8, opts, kAnchor);
  const QuadTrajectory circle_t(QuadShape::kCircleTangent, opts, kAnchor);
  const QuadTrajectory fig8_t(QuadShape::kFigure8Tangent, opts, kAnchor);
  const QuadTrajectory * shapes[4] = {&circle, &fig8, &circle_t, &fig8_t};
  std::string line;
  int rows = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') {continue;}
    std::istringstream is(line);
    int shape;
    double t;
    std::vector<double> v(26);
    is >> shape >> t;
    for (double & x : v) {is >> x;}
    const auto s = sampleCcmReference(model, shapes[shape], kAnchor, kYaw, t);
    const auto & r = s.ref;
    Eigen::Matrix<double, 26, 1> got;
    got << r.position, r.velocity, r.R.row(0).transpose(), r.R.row(1).transpose(), r.R.row(2).transpose(),
      r.omega, r.thrust, r.torque, s.thrust_command, s.torque_command;
    for (int i = 0; i < 26; ++i) {
      // The command carries a central difference of step 1e-3 on both sides.
      const double tol = i >= 22 ? 1e-6 : 1e-9;
      EXPECT_NEAR(got[i], v[i], tol * std::max(1.0, std::abs(v[i]))) << "shape " << shape << " t " << t
                                                                         << " column " << i;
    }
    ++rows;
  }
  EXPECT_EQ(rows, 48);
}

// The reference is a solution of the rigid-body dynamics it claims:
// p' = v, m v' = -m g e3 + T R e3, R' = R omega^, J omega' = tau - omega x J omega.
TEST(QuadrotorPlanner, ReferenceSatisfiesDynamics)
{
  const auto model = makeQuadrotorModel("t650");
  const Eigen::Vector3d e3(0, 0, 1);
  for (const auto shape : {QuadShape::kCircle, QuadShape::kFigure8, QuadShape::kCircleTangent,
      QuadShape::kFigure8Tangent})
  {
    const QuadTrajectory tr(shape, QuadTrajectoryOptions{}, kAnchor);
    for (double t : {0.5, 1.7, 2.95, 5.0, 9.3, 16.4, 18.2}) {
      const double h = 1e-4;
      const auto r0 = sampleCcmReference(model, &tr, kAnchor, kYaw, t).ref;
      const auto rp = sampleCcmReference(model, &tr, kAnchor, kYaw, t + h).ref;
      const auto rm = sampleCcmReference(model, &tr, kAnchor, kYaw, t - h).ref;
      const Eigen::Vector3d pdot = (rp.position - rm.position) / (2 * h);
      const Eigen::Vector3d vdot = (rp.velocity - rm.velocity) / (2 * h);
      const Eigen::Matrix3d Rdot = (rp.R - rm.R) / (2 * h);
      const Eigen::Vector3d wdot = (rp.omega - rm.omega) / (2 * h);
      EXPECT_LT((pdot - r0.velocity).norm(), 1e-6);
      EXPECT_LT((vdot - (-model.gravity * e3 + r0.thrust / model.mass * r0.R * e3)).norm(), 1e-5);
      EXPECT_LT((Rdot - r0.R * hat(r0.omega)).norm(), 1e-5);
      const Eigen::Vector3d wdot_model =
        model.inertia.inverse() * (r0.torque - r0.omega.cross(model.inertia * r0.omega));
      EXPECT_LT((wdot - wdot_model).norm(), 1e-4) << "t " << t;
      EXPECT_NEAR((r0.R.transpose() * r0.R - Eigen::Matrix3d::Identity()).norm(), 0.0, 1e-12);
    }
  }
}

TEST(QuadrotorPlanner, StartsAndEndsAtRestOnTheAnchor)
{
  const auto model = makeQuadrotorModel("t650");
  for (const auto shape : {QuadShape::kCircle, QuadShape::kFigure8}) {
    const QuadTrajectory tr(shape, QuadTrajectoryOptions{}, kAnchor);
    EXPECT_NEAR(tr.duration(), 19.0, 1e-12);
    for (double t : {0.0, tr.duration()}) {
      const auto j = tr.jets(t);
      EXPECT_LT((j.p - kAnchor).norm(), 1e-9);
      EXPECT_LT(j.v.norm() + j.a.norm() + j.j.norm() + j.s.norm(), 1e-9);
      // At rest the reference is a level hover at the held heading.
      const auto s = sampleCcmReference(model, &tr, kAnchor, kYaw, t);
      EXPECT_NEAR(s.ref.thrust, model.mass * model.gravity, 1e-9);
      EXPECT_LT(s.ref.torque.norm() + s.ref.omega.norm(), 1e-9);
      EXPECT_NEAR(std::atan2(s.ref.R(1, 0), s.ref.R(0, 0)), kYaw, 1e-12);
    }
    // Phase and its derivatives are continuous across both ramp boundaries.
    for (double tb : {3.0, 16.0}) {
      const auto a = tr.phase(tb - 1e-9), b = tr.phase(tb + 1e-9);
      EXPECT_LT((a - b).norm(), 1e-6) << "boundary " << tb;
    }
  }
}

TEST(QuadrotorPlanner, StopRampsOutEarly)
{
  QuadTrajectory tr(QuadShape::kCircle, QuadTrajectoryOptions{}, kAnchor);
  tr.requestStop(6.0);
  EXPECT_NEAR(tr.duration(), 9.0, 1e-12);
  const auto j = tr.jets(9.0);
  EXPECT_LT(j.v.norm() + j.a.norm(), 1e-9);
  tr.requestStop(7.0);  // already ramping out: no effect
  EXPECT_NEAR(tr.duration(), 9.0, 1e-12);
}

TEST(QuadrotorPlanner, CommandLeadsByTheRotorLag)
{
  const auto model = makeQuadrotorModel("t650");
  auto no_lag = model;
  no_lag.rotor_time_constant = 0.0;
  const QuadTrajectory tr(QuadShape::kFigure8, QuadTrajectoryOptions{}, kAnchor);
  for (double t : {1.0, 2.0, 7.0}) {
    const auto s = sampleCcmReference(model, &tr, kAnchor, kYaw, t);
    const auto s0 = sampleCcmReference(no_lag, &tr, kAnchor, kYaw, t);
    EXPECT_NEAR(s0.thrust_command, s0.ref.thrust, 1e-12);
    const double h = 1e-3;
    const double dT = (sampleCcmReference(model, &tr, kAnchor, kYaw, t + h).ref.thrust -
      sampleCcmReference(model, &tr, kAnchor, kYaw, t - h).ref.thrust) / (2 * h);
    EXPECT_NEAR(s.thrust_command - s.ref.thrust, model.rotor_time_constant * dT, 1e-9);
  }
  // Hover: the command is the reference wrench.
  const auto hover = sampleCcmReference(model, nullptr, kAnchor, kYaw, 0.0);
  EXPECT_NEAR(hover.thrust_command, model.mass * model.gravity, 1e-9);
}

TEST(QuadrotorPlanner, TransitionIsRestToRestAndDynamicallyConsistent)
{
  using fsc_trajectory_planner::QuadTransition;
  const auto model = makeQuadrotorModel("t650");
  const Eigen::Vector3d p0(0.1, -0.2, 1.0), p1(1.3, 0.4, 1.4);
  const QuadTransition mv(p0, p1, 0.4, 3.0, kYaw, kYaw);
  EXPECT_NEAR(mv.duration(), 2.4609375 * (p1 - p0).norm() / 0.4, 1e-12);
  for (double t : {0.0, mv.duration()}) {
    const auto j = mv.jets(t);
    EXPECT_LT((j.p - (t > 0 ? p1 : p0)).norm(), 1e-12);
    EXPECT_LT(j.v.norm() + j.a.norm() + j.j.norm() + j.s.norm(), 1e-12);
  }
  // Snap is continuous at both ends (nonic phasing), so the torque reference is too.
  EXPECT_LT(mv.jets(1e-4).s.norm(), 1e-3);
  EXPECT_LT(mv.jets(mv.duration() - 1e-4).s.norm(), 1e-3);
  // Peak speed is the requested one, at mid-move.
  EXPECT_NEAR(mv.jets(0.5 * mv.duration()).v.norm(), 0.4, 1e-9);
  // Derivatives match finite differences of the jets they come from.
  for (double t : {0.7, 2.2, 3.9}) {
    const double h = 1e-5;
    const auto jp = mv.jets(t + h), jm = mv.jets(t - h), j0 = mv.jets(t);
    EXPECT_LT(((jp.p - jm.p) / (2 * h) - j0.v).norm(), 1e-7);
    EXPECT_LT(((jp.a - jm.a) / (2 * h) - j0.j).norm(), 1e-5);
    EXPECT_LT(((jp.j - jm.j) / (2 * h) - j0.s).norm(), 1e-4);
  }
  const auto s = sampleCcmReference(model, mv, 0.0);
  EXPECT_NEAR(s.ref.thrust, model.mass * model.gravity, 1e-9);
}

// The *_tangent shapes: same path as the base shape, heading = direction of travel
// whenever moving, yaw jets consistent with finite differences, constant-yaw shapes
// unchanged, and the start heading of s(0).
TEST(QuadrotorPlanner, TangentYawFollowsThePath)
{
  using fsc_trajectory_planner::wrapAngle;
  const QuadTrajectoryOptions opts;
  const QuadTrajectory circle(QuadShape::kCircle, opts, kAnchor), circle_t(QuadShape::kCircleTangent, opts, kAnchor);
  const QuadTrajectory fig8(QuadShape::kFigure8, opts, kAnchor), fig8_t(QuadShape::kFigure8Tangent, opts, kAnchor);
  EXPECT_NEAR(circle_t.startYaw(kYaw), M_PI / 2, 1e-12);   // ccw circle heads +y
  EXPECT_NEAR(fig8_t.startYaw(kYaw), M_PI / 4, 1e-12);     // figure-8 crossing at 45 deg
  EXPECT_NEAR(circle.startYaw(kYaw), kYaw, 1e-15);
  for (const auto * pair : {&circle, &fig8}) {
    const QuadTrajectory & tan = pair == &circle ? circle_t : fig8_t;
    for (double t = 0.2; t < tan.duration(); t += 0.37) {
      EXPECT_LT((tan.jets(t).p - pair->jets(t).p).norm(), 1e-12);
      EXPECT_EQ(pair->yawJets(t, kYaw).psi, kYaw);
      const auto y = tan.yawJets(t, kYaw);
      const Eigen::Vector3d v = tan.jets(t).v;
      if (v.head<2>().norm() > 0.05) {
        EXPECT_NEAR(wrapAngle(std::atan2(v.y(), v.x()) - y.psi), 0.0, 1e-9) << "t " << t;
      }
      const double h = 1e-5;
      const auto yp = tan.yawJets(t + h, kYaw), ym = tan.yawJets(t - h, kYaw);
      EXPECT_NEAR(wrapAngle(yp.psi - ym.psi) / (2 * h), y.dpsi, 1e-6);
      EXPECT_NEAR((yp.dpsi - ym.dpsi) / (2 * h), y.ddpsi, 1e-5);
    }
    // At rest (ends of the run) the heading is fixed.
    EXPECT_LT(std::abs(tan.yawJets(0.0, kYaw).dpsi) + std::abs(tan.yawJets(tan.duration(), kYaw).dpsi), 1e-12);
  }
  // Peaks the node's start check uses: circle 2 pi / lap, figure-8 well above the 1.5 rad/s default limit.
  EXPECT_NEAR(circle_t.yawRateMax(), 2 * M_PI / opts.lap_time, 1e-3);
  EXPECT_EQ(circle.yawRateMax(), 0.0);
  EXPECT_GT(fig8_t.yawRateMax(), 1.5);
  QuadTrajectoryOptions slow = opts;
  slow.time_scale = 0.55;
  EXPECT_LT(QuadTrajectory(QuadShape::kFigure8Tangent, slow, kAnchor).yawRateMax(), 1.5);
  EXPECT_THROW(parseQuadShape("figure8_sideways"), std::runtime_error);
  EXPECT_EQ(fsc_trajectory_planner::quadShapeName(parseQuadShape("circle_tangent")), "circle_tangent");
}

// A transition turns the heading the shorter way, rest to rest, within its peak yaw rate.
TEST(QuadrotorPlanner, TransitionTurnsTheShorterWay)
{
  using fsc_trajectory_planner::QuadTransition;
  using fsc_trajectory_planner::wrapAngle;
  const Eigen::Vector3d p(0.0, 0.0, 1.0);
  const double yaw0 = 3.0, yaw1 = -3.0;   // 0.283 rad apart across +-pi
  const QuadTransition turn(p, p, 0.4, 1.0, yaw0, yaw1, 0.5);
  EXPECT_NEAR(wrapAngle(turn.goalYaw() - yaw1), 0.0, 1e-12);
  EXPECT_NEAR(turn.duration(), 2.4609375 * (2 * M_PI - 6.0) / 0.5, 1e-12);   // 0.28 rad, not 6.0 rad
  const QuadTransition big(p, p, 0.4, 1.0, 0.0, M_PI / 2, 0.5);
  EXPECT_NEAR(big.duration(), 2.4609375 * (M_PI / 2) / 0.5, 1e-12);
  EXPECT_NEAR(big.yawJets(0.5 * big.duration()).dpsi, 0.5, 1e-9);   // peak yaw rate mid-turn
  for (double t : {0.0, big.duration()}) {
    EXPECT_LT(std::abs(big.yawJets(t).dpsi) + std::abs(big.yawJets(t).ddpsi), 1e-12);
  }
  double prev = turn.yawJets(0.0).psi, travelled = 0.0;
  for (double t = 0.01; t <= turn.duration(); t += 0.01) {
    const double psi = turn.yawJets(t).psi;
    travelled += std::abs(wrapAngle(psi - prev));
    prev = psi;
  }
  EXPECT_NEAR(travelled, 2 * M_PI - 6.0, 1e-6);
}

TEST(QuadrotorPlanner, TimeScaleShortensTheLapsAndPlacesByCentre)
{
  QuadTrajectoryOptions opts;
  opts.time_scale = 1.25;
  const QuadTrajectory tr(QuadShape::kCircle, opts, kAnchor);
  EXPECT_NEAR(tr.duration(), opts.laps * opts.lap_time / 1.25 + opts.ramp_time, 1e-12);
  EXPECT_NEAR(tr.cruiseSpeedMax(), 2 * M_PI * opts.circle_radius / (opts.lap_time / 1.25), 1e-3);
  // The circle's centre is start - offset.
  const Eigen::Vector3d centre = kAnchor - fsc_trajectory_planner::quadShapeStartOffset(QuadShape::kCircle, opts);
  for (double t : {4.0, 7.3, 10.1}) {
    const auto p = tr.jets(t).p;
    EXPECT_NEAR((p - centre).head<2>().norm(), opts.circle_radius, 1e-9);
  }
  EXPECT_LT(fsc_trajectory_planner::quadShapeStartOffset(QuadShape::kFigure8, opts).norm(), 1e-15);
}

// With disturbances folded in, the reference solves the DISTURBED dynamics it is meant for:
//   m v' = -m g e3 + T R e3 + m d,   J omega' = tau - omega x J omega + J d_w.
TEST(QuadrotorPlanner, ReferenceSatisfiesDisturbedDynamics)
{
  const auto model = makeQuadrotorModel("t650");
  const Eigen::Vector3d e3(0, 0, 1), d(0.3, -0.2, 0.15), dw(0.4, -0.6, 0.8);
  for (const auto shape : {QuadShape::kCircle, QuadShape::kFigure8Tangent}) {
    const QuadTrajectory tr(shape, QuadTrajectoryOptions{}, kAnchor);
    for (double t : {1.7, 5.0, 9.3}) {
      const double h = 1e-4;
      const auto at = [&](double tt) {return sampleCcmReference(model, &tr, kAnchor, kYaw, tt, d, 1e-3, dw).ref;};
      const auto r0 = at(t), rp = at(t + h), rm = at(t - h);
      const Eigen::Vector3d vdot = (rp.velocity - rm.velocity) / (2 * h);
      const Eigen::Vector3d wdot = (rp.omega - rm.omega) / (2 * h);
      EXPECT_LT((vdot - (-model.gravity * e3 + r0.thrust / model.mass * r0.R * e3 + d)).norm(), 1e-5);
      const Eigen::Vector3d wdot_model =
        model.inertia.inverse() * (r0.torque - r0.omega.cross(model.inertia * r0.omega)) + dw;
      EXPECT_LT((wdot - wdot_model).norm(), 1e-4) << "t " << t;
      // The angular term only shifts the torque, by -J d_w.
      const auto clean = sampleCcmReference(model, &tr, kAnchor, kYaw, t, d).ref;
      EXPECT_LT((clean.torque - r0.torque - model.inertia * dw).norm(), 1e-9);
      EXPECT_NEAR(clean.thrust, r0.thrust, 1e-12);
    }
  }
}
