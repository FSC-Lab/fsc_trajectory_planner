// The end-effector trajectory mode: circle and figure-8 runs on the T650
// aerial manipulator -- rest at both ends, FK round trip against the
// prescribed EE pose, every bound honoured, and a positive maximum time
// scale found by bisection.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "fsc_trajectory_planner/ee_trajectory_planner.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"

using namespace fsc_trajectory_planner;

namespace
{
RestSpec homeHold(const VehicleModel & v)
{
  RestSpec h;
  h.x_b << 0.0, 0.0, 1.2;
  h.phi = -0.5 * M_PI;   // actual yaw 0
  h.q = v.home;
  return h;
}
}  // namespace

class EeTrajectoryTest : public ::testing::TestWithParam<const char *> {};

TEST_P(EeTrajectoryTest, RunIsCompatibleAndBounded)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  EeShape shape;
  shape.type = GetParam();
  shape.laps = 1;
  EeTrajectoryOptions o;
  // what the ground station does: plan inside the feasible time-scale range
  const double s_max = EeTrajectoryPlanner::maxTimeScale(*v, homeHold(*v), shape, o);
  ASSERT_GT(s_max, 0.1);
  o.time_scale = 0.9 * s_max;
  EeTrajectoryDiag d;
  const auto traj = EeTrajectoryPlanner::plan(*v, homeHold(*v), shape, o, &d);
  std::cout << "  s_max " << s_max << ": " << d.summary << "\n";
  EXPECT_TRUE(d.feasible) << d.violation;
  EXPECT_LT(d.max_dyn_defect, 1e-3);
  EXPECT_LT(d.ee_pos_err_max, 5e-3);
  EXPECT_LT(d.ee_rot_err_max_deg, 1.0);
  EXPECT_LT(d.rest_err, 2e-3);
  EXPECT_GE(d.min_sigma_nd, 0.10);
  EXPECT_GE(d.min_beta_deg, 5.0);
  // the run starts and ends on the start rest
  const RestSpec r0 = EeTrajectoryPlanner::startRest(*v, homeHold(*v), shape, o);
  const WbReference want = restReference(v->params, r0);
  const WbReference a = traj->eval(0.0), b = traj->eval(traj->duration());
  EXPECT_LT((a.x_cd - want.x_cd).norm(), 2e-3);
  EXPECT_LT((b.x_cd - want.x_cd).norm(), 2e-3);
  EXPECT_LT((a.q_d - r0.q).cwiseAbs().maxCoeff(), 2e-3);
  EXPECT_LT((b.q_d - r0.q).cwiseAbs().maxCoeff(), 2e-3);
  EXPECT_LT((traj->goalRest().x_b - r0.x_b).norm(), 1e-12);
  // q1 is the fixed joint, q2 the assigned sinusoid
  EXPECT_LT(std::abs(a.q_d(0)), 1e-9);
  // the EE heading follows the tangent: at t = 0 the model heading is the hold's
  EXPECT_NEAR(std::atan2(a.b1_de(1), a.b1_de(0)), homeHold(*v).phi, 1e-3);
  // smooth: no gap in the streamed reference at 100 Hz
  double worst = 0.0;
  WbReference prev = traj->eval(0.0);
  for (double t = 0.01; t <= traj->duration(); t += 0.01) {
    const WbReference cur = traj->eval(t);
    worst = std::max(worst, (cur.x_cd - prev.x_cd).norm());
    prev = cur;
  }
  EXPECT_LT(worst, 0.01);
}

INSTANTIATE_TEST_SUITE_P(Shapes, EeTrajectoryTest, ::testing::Values("circle", "figure8"));

TEST(EeTrajectory, MaxTimeScaleIsPositiveAndBinding)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  EeShape shape;
  shape.type = "circle";
  shape.laps = 1;
  EeTrajectoryOptions o;
  const auto t0 = std::chrono::steady_clock::now();
  const double s_max = EeTrajectoryPlanner::maxTimeScale(*v, homeHold(*v), shape, o);
  const double ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - t0).count();
  std::cout << "  s_max = " << s_max << " (" << ms << " ms)\n";
  EXPECT_GT(s_max, 0.2);
  // at s_max the plan passes; at 1.1 s_max some bound binds
  EeTrajectoryOptions ok = o;
  ok.time_scale = s_max;
  EeTrajectoryDiag d;
  EXPECT_NO_THROW(EeTrajectoryPlanner::plan(*v, homeHold(*v), shape, ok, &d));
  std::cout << "  at s_max: " << d.summary << "\n";
  EeTrajectoryOptions over = o;
  over.time_scale = s_max * 1.1;
  EXPECT_THROW(EeTrajectoryPlanner::plan(*v, homeHold(*v), shape, over, &d), std::runtime_error);
  std::cout << "  over: " << d.violation << "\n";
}

TEST(EeTrajectory, LevelEndEffectorIsRefusedAsSingular)
{
  const auto v = makeVehicleModel("t650_aerial_manipulator");
  EeShape shape;
  EeTrajectoryOptions o;
  o.ee_fold_deg = 0.0;          // literal zero roll/pitch: the wrist singularity
  o.q2_center_deg = 0.0;
  o.q2_amp_deg = 0.0;
  EeTrajectoryDiag d;
  EXPECT_THROW(EeTrajectoryPlanner::plan(*v, homeHold(*v), shape, o, &d), std::runtime_error);
  std::cout << "  level EE: " << d.violation << "\n";
}
