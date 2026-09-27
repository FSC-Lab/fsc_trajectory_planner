// MIT License
// Copyright (c) 2026 FSC Lab
//
// PS4 teleoperation reference generator (teleop_reference.hpp): the smoother's
// closed-form derivatives, a bumpless seed, EXACT compatibility of every
// streamed sample, the walls, and the settle onto a rest.
#include <gtest/gtest.h>

#include <cmath>

#include "fsc_trajectory_planner/teleop_reference.hpp"

using namespace fsc_trajectory_planner;

namespace
{
std::shared_ptr<const VehicleModel> t650()
{
  // Teleop is kinematic (the flat map is FK); the armature model does not
  // enter it, so the plain vehicle is enough.
  VehicleOptions vo;
  vo.base_com = Vec3{0.0, -0.017854, 0.0};
  return makeVehicleModel("t650_aerial_manipulator", vo);
}

VecN home()
{
  return (VecN() << 0.0, 40.0 * M_PI / 180.0, 40.0 * M_PI / 180.0, 0.0).finished();
}

// The sample's own FK: base from x_cd and the thrust-direction attitude,
// then the chain -- r_ed and b1_de must be exactly that.
void expectCompatible(const VehicleModel & v, const WbReference & r, double tol)
{
  const Mat3 R0 = buildR0(r.x_cd_ddot + v.params.g * Vec3::UnitZ(), r.b1_d);
  Vec3 r0c, r0e;
  Mat3 Re;
  armKinematics(r.q_d, v.params, &r0c, &r0e, &Re);
  const Vec3 x_b = r.x_cd - R0 * r0c;
  EXPECT_LT((x_b + R0 * r0e - r.r_ed).norm(), tol);
  EXPECT_LT(((R0 * Re).col(0) - r.b1_de).norm(), tol);
}
}  // namespace

TEST(ChainFilter, DerivativesMatchFiniteDifferences)
{
  ChainFilter f;
  f.setup(5, 6.0, 1);
  Eigen::VectorXd x(1);
  x(0) = 0.0;
  f.reset(x);
  const double dt = 1e-4;
  double u = 0.0;
  std::vector<Eigen::VectorXd> d0, d1;
  for (int k = 0; k < 20000; ++k) {
    u += 0.2 * dt;                         // a ramp: the stick held
    x(0) = u;
    f.step(x, dt);
    if (k == 9000 || k == 9001) {
      Eigen::VectorXd row(5);
      for (int j = 0; j <= 4; ++j) {row(j) = f.deriv(j)(0);}   // all from stages
      (k == 9000 ? d0 : d1).push_back(row);
    }
  }
  for (int j = 0; j < 4; ++j) {
    const double fd = (d1[0](j) - d0[0](j)) / dt;
    EXPECT_NEAR(fd, d0[0](j + 1), 1e-3 * (1.0 + std::abs(d0[0](j + 1))))
      << "derivative " << j + 1;
  }
}

TEST(ChainFilter, NeverOvershootsTheInput)
{
  ChainFilter f;
  f.setup(4, 6.0, 1);
  Eigen::VectorXd x(1);
  x(0) = 0.0;
  f.reset(x);
  double u = 0.0, ymax = 0.0;
  for (int k = 0; k < 400; ++k) {
    if (k < 100) {u += 0.002;}             // ramp to 0.2, then stop dead
    x(0) = u;
    f.step(x, 0.01);
    ymax = std::max(ymax, f.deriv(0)(0));
  }
  EXPECT_LE(ymax, 0.2 + 1e-12);
  EXPECT_NEAR(f.deriv(0)(0), 0.2, 1e-6);
}

TEST(TeleopReference, SeedIsTheHold)
{
  auto v = t650();
  RestSpec rest;
  rest.x_b = Vec3{0.3, -0.2, 1.2};
  rest.phi = 0.4;
  rest.q = home();
  const WbReference h = restReference(v->params, rest);
  TeleopReference t(v, TeleopOptions{});
  t.seed(h.x_cd, rest.phi, rest.q);
  const WbReference r = t.reference(1.0);
  EXPECT_LT((r.x_cd - h.x_cd).norm(), 1e-12);
  EXPECT_LT((r.r_ed - h.r_ed).norm(), 1e-12);
  EXPECT_LT((r.b1_de - h.b1_de).norm(), 1e-12);
  EXPECT_LT((r.q_d - h.q_d).norm(), 1e-12);
  EXPECT_LT(r.x_cd_dot.norm() + r.x_cd_ddot.norm() + r.r_ed_dot.norm(), 1e-12);
}

TEST(TeleopReference, EverySampleIsCompatibleAndSmooth)
{
  auto v = t650();
  RestSpec rest;
  rest.x_b = Vec3{0.0, 0.0, 1.2};
  rest.phi = 0.0;
  rest.q = home();
  const WbReference h = restReference(v->params, rest);
  TeleopReference t(v, TeleopOptions{});
  t.seed(h.x_cd, rest.phi, rest.q);
  const double dt = 0.01, s = 0.5;
  WbReference prev = t.reference(s);
  double max_jump_acc = 0.0;
  for (int k = 0; k < 800; ++k) {
    TeleopInput in;
    if (k < 300) {in.com = Vec3{1.0, 0.5, 0.0}; in.yaw = 1.0;}      // fly and turn
    if (k >= 100 && k < 400) {in.ee = Vec3{0.0, 0.6, -1.0}; in.roll = 1.0;}   // arm
    t.step(dt * s, in, std::nullopt);
    const WbReference r = t.reference(s);
    expectCompatible(*v, r, 1e-9);
    // derivatives on the CALLER's clock: finite differences over dt agree
    EXPECT_LT(((r.x_cd - prev.x_cd) / dt - 0.5 * (r.x_cd_dot + prev.x_cd_dot)).norm(), 2e-4);
    EXPECT_LT(((r.q_d - prev.q_d) / dt - 0.5 * (r.qdot_d + prev.qdot_d)).norm(), 2e-3);
    max_jump_acc = std::max(max_jump_acc, (r.x_cd_ddot - prev.x_cd_ddot).norm());
    for (int j = 0; j < kNumJoints; ++j) {
      EXPECT_GE(r.q_d(j), kArmQMin[j]);
      EXPECT_LE(r.q_d(j), kArmQMax[j]);
    }
    prev = r;
  }
  EXPECT_LT(max_jump_acc, 0.01);           // no acceleration steps
  // moved: forward (model +y) and left (model -x) at heading 0, turned CCW
  EXPECT_GT(t.comTarget()(1) - h.x_cd(1), 0.2);
  EXPECT_LT(t.comTarget()(0) - h.x_cd(0), -0.1);
  EXPECT_GT(t.phiTarget(), 0.5);
}

TEST(TeleopReference, WallsHoldAndSettleOntoARest)
{
  auto v = t650();
  RestSpec rest;
  rest.x_b = Vec3{0.0, 0.0, 1.0};
  rest.phi = 0.0;
  rest.q = home();
  const WbReference h = restReference(v->params, rest);
  TeleopOptions o;
  TeleopReference t(v, o);
  t.seed(h.x_cd, rest.phi, rest.q);
  // CoM down into the altitude limit, arm forward into its reach.
  for (int k = 0; k < 2000; ++k) {
    TeleopInput in;
    in.com = Vec3{0.0, 0.0, -1.0};
    in.ee = Vec3{1.0, 0.0, 0.0};
    t.step(0.01, in, std::nullopt);
  }
  EXPECT_GE(t.comTarget()(2), o.com_z_min - 1e-9);
  EXPECT_TRUE(t.status().com_blocked & 4u);
  EXPECT_TRUE(t.status().arm_blocked & 1u);
  EXPECT_FALSE(t.status().wall.empty());
  // released: settles on the targets, and the rest reference is the sample
  for (int k = 0; k < 2000 && !t.settled(); ++k) {t.step(0.01, TeleopInput{}, std::nullopt);}
  ASSERT_TRUE(t.settled());
  const WbReference r = t.reference(1.0);
  const WbReference rr = restReference(v->params, t.targetRest());
  EXPECT_LT((r.x_cd - rr.x_cd).norm(), 2e-4);
  EXPECT_LT((r.r_ed - rr.r_ed).norm(), 2e-4);
  EXPECT_LT((r.q_d - rr.q_d).cwiseAbs().maxCoeff(), 2e-4);
}

TEST(TeleopReference, HomeFoldsTheArmBack)
{
  auto v = t650();
  RestSpec rest;
  rest.x_b = Vec3{0.0, 0.0, 1.2};
  rest.phi = 0.0;
  rest.q = home();
  const WbReference h = restReference(v->params, rest);
  TeleopReference t(v, TeleopOptions{});
  t.seed(h.x_cd, rest.phi, rest.q);
  for (int k = 0; k < 150; ++k) {
    TeleopInput in;
    in.ee = Vec3{0.0, 1.0, -1.0};
    in.roll = -1.0;
    t.step(0.01, in, std::nullopt);
  }
  EXPECT_GT((t.qTarget() - TeleopOptions{}.home).norm(), 0.1);
  TeleopInput go;
  go.home = true;
  t.step(0.01, go, std::nullopt);
  for (int k = 0; k < 3000 && t.status().homing; ++k) {t.step(0.01, TeleopInput{}, std::nullopt);}
  EXPECT_LT((t.qTarget() - TeleopOptions{}.home).norm(), 1e-12);   // the PAD's home
  EXPECT_LT((t.comTarget() - h.x_cd).norm(), 1e-12);    // the platform held
}

TEST(TeleopReference, PositionIkRoundTrip)
{
  auto v = t650();
  VecN q;
  q << 0.2, 0.5, 0.6, 0.3;
  Vec3 s;
  armKinematics(q, v->params, nullptr, &s, nullptr);
  VecN seed = home();
  seed(3) = q(3);
  VecN out;
  ASSERT_TRUE(TeleopReference::ikPosition(v->params, s, seed, &out));
  EXPECT_LT((out - q).norm(), 1e-6);
}

TEST(TeleopReference, JointBoxKeepsTheArmOffItsStops)
{
  auto v = t650();
  RestSpec rest;
  rest.x_b = Vec3{0.0, 0.0, 1.5};
  rest.phi = 0.0;
  rest.q = home();
  const WbReference h = restReference(v->params, rest);
  const std::vector<std::pair<Vec3, double>> pushes = {
    {Vec3{1, 0, 0}, 0.0}, {Vec3{-1, 0, 0}, 0.0}, {Vec3{0, 1, 0}, 0.0},
    {Vec3{0, -1, 0}, 0.0}, {Vec3{0, 0, 1}, 0.0}, {Vec3{0, 0, -1}, 0.0},
    {Vec3{0, 0, 0}, 1.0}, {Vec3{0, 0, 0}, -1.0}, {Vec3{0.7, -0.7, -0.7}, 1.0}};
  for (const auto & push : pushes) {
    TeleopReference t(v, TeleopOptions{});
    t.seed(h.x_cd, rest.phi, rest.q);
    for (int k = 0; k < 3000; ++k) {
      TeleopInput in;
      in.ee = push.first;
      in.roll = push.second;
      t.step(0.01, in, std::nullopt);
      for (int j = 0; j < kNumJoints; ++j) {
        double lo, hi;
        t.jointBox(j, &lo, &hi);
        // never further outside the box than the pose it started from
        EXPECT_LE(t.qTarget()(j), std::max(hi, home()(j)) + 1e-9) << "joint " << j + 1;
        EXPECT_GE(t.qTarget()(j), std::min(lo, home()(j)) - 1e-9) << "joint " << j + 1;
        // and 20 % of every range in reserve against the HARDWARE stop
        const double reserve = 0.1 * (kArmQMax[j] - kArmQMin[j]) - 1e-9;
        EXPECT_LE(t.qTarget()(j), kArmQMax[j] - std::min(reserve, kArmQMax[j] - home()(j)));
        EXPECT_GE(t.qTarget()(j), kArmQMin[j] + reserve);
      }
    }
  }
  // Once inside, the box holds: down first (q2 drops below its bound), then
  // fold up for a long time -- q2 stops at the bound, not back at home's 40.
  TeleopReference t(v, TeleopOptions{});
  t.seed(h.x_cd, rest.phi, rest.q);
  for (int k = 0; k < 400; ++k) {TeleopInput in; in.ee = Vec3{0, 0, -1}; t.step(0.01, in, std::nullopt);}
  double lo2, hi2;
  t.jointBox(1, &lo2, &hi2);
  ASSERT_LT(t.qTarget()(1), hi2);
  for (int k = 0; k < 4000; ++k) {TeleopInput in; in.ee = Vec3{0, 0, 1}; t.step(0.01, in, std::nullopt);}
  EXPECT_LE(t.qTarget()(1), hi2 + 1e-9);
}

// Not a check: prints how far the pad can move the grasp point from the
// folded home, per direction, and which wall stops it (Command.md 7.21).
TEST(TeleopReference, ReachFromHomeReport)
{
  auto v = t650();
  RestSpec rest;
  rest.x_b = Vec3{0.0, 0.0, 1.5};
  rest.phi = 0.0;
  const char * name[6] = {"forward", "back", "left", "right", "up", "down"};
  const Vec3 dir[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
  for (const VecN & q0 : {home(), VecN((VecN() << 0.0, 30.0 * M_PI / 180.0, 30.0 * M_PI / 180.0, 0.0).finished())}) {
  rest.q = q0;
  const WbReference h = restReference(v->params, rest);
  std::printf("from q = [%.0f %.0f %.0f %.0f] deg:\n", q0(0) * 180 / M_PI, q0(1) * 180 / M_PI,
    q0(2) * 180 / M_PI, q0(3) * 180 / M_PI);
  for (int i = 0; i < 6; ++i) {
    TeleopReference t(v, TeleopOptions{});
    t.seed(h.x_cd, rest.phi, rest.q);
    const Vec3 s0 = t.sTarget();
    std::string wall;
    for (int k = 0; k < 3000; ++k) {
      TeleopInput in;
      in.ee = dir[i];
      t.step(0.01, in, std::nullopt);
      if (!t.status().wall.empty()) {wall = t.status().wall;}
    }
    const Vec3 d = t.sTarget() - s0;     // model frame: fwd = +y, left = -x
    const Vec3 op{d(1), -d(0), d(2)};
    std::printf("  %-8s %5.1f cm  (fwd %+5.1f left %+5.1f up %+5.1f)  stopped by: %s\n", name[i],
      100.0 * op.dot(dir[i]), 100.0 * op(0), 100.0 * op(1), 100.0 * op(2), wall.c_str());
  }
  }
}
