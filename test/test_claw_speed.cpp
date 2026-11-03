// MIT License
// Copyright (c) 2026 FSC Lab
//
// The claw speed of the autonomous insert / drop gates (claw_speed.hpp): no NaN
// from 10 Hz tick jitter once the buffer is full (the 10-31 hardware mail flights),
// a still claw reads slow, a moving one reads its speed, a gap starts afresh.
#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <deque>
#include <random>
#include <vector>

#include "fsc_trajectory_planner/claw_speed.hpp"

using namespace fsc_trajectory_planner;

namespace
{
using TimePoint = std::chrono::steady_clock::time_point;

TimePoint at(double t)
{
  return TimePoint{} + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
    std::chrono::duration<double>(10.0 + t));
}
}  // namespace

// THE 10-31 REGRESSION. Four short ticks then a long one (0.099 x 4 + 0.105): the
// sample four periods back is 0.396 s old while the one five back (0.501 s) has
// just left the 0.5 s window. With the old 0.4 s minimum span that tick read NaN,
// failed the speed gate and restarted the release dwell.
TEST(ClawSpeed, TickJitterNeverDropsTheSpeedOnceFull)
{
  const std::vector<double> dts{0.099, 0.099, 0.099, 0.099, 0.105};
  std::deque<ClawSample> buf;
  const Eigen::Vector3d claw{1.6, 0.25, 1.02};
  double t = 0.0;
  int short_span_ticks = 0;
  for (int k = 0; k < 60; ++k) {
    const double v = clawSpeed(&buf, claw, at(t));
    if (t > kClawSpeedWindow) {
      EXPECT_TRUE(std::isfinite(v)) << "tick " << k << " at t = " << t;
      EXPECT_LT(v, 1e-9);
      const double span = std::chrono::duration<double>(at(t) - buf.front().first).count();
      if (span < 0.4) {++short_span_ticks;}
    }
    t += dts[k % dts.size()];
  }
  // the sequence does reproduce the old failure: spans under 0.4 s occurred
  EXPECT_GT(short_span_ticks, 0);
}

TEST(ClawSpeed, AStillClawWithMillimetreJitterReadsSlow)
{
  std::mt19937 rng(7);
  std::normal_distribution<double> n(0.0, 0.002);       // 2 mm per axis
  std::uniform_real_distribution<double> j(-0.008, 0.008);  // +-8 ms tick jitter
  std::deque<ClawSample> buf;
  double t = 0.0, vmax = 0.0;
  for (int k = 0; k < 200; ++k) {
    const double v = clawSpeed(&buf, Eigen::Vector3d{1.6 + n(rng), 0.25 + n(rng), 1.02 + n(rng)}, at(t));
    if (t > kClawSpeedWindow) {
      ASSERT_TRUE(std::isfinite(v)) << "tick " << k;
      vmax = std::max(vmax, v);
    }
    t += 0.1 + j(rng);
  }
  // the insert release gate is 0.05 m/s, the drop's and the auto-insert start's 0.03
  EXPECT_LT(vmax, 0.05);
}

TEST(ClawSpeed, AMovingClawReadsItsSpeed)
{
  std::deque<ClawSample> buf;
  double v = 0.0;
  for (int k = 0; k <= 20; ++k) {
    const double t = 0.1 * k;
    v = clawSpeed(&buf, Eigen::Vector3d{0.1 * t, 0.0, 0.0}, at(t));
  }
  // on a straight line at constant speed both terms ARE the speed: the fit, and
  // 2 rmax / span (the mean sits mid-span, the ends half a span's travel away)
  EXPECT_NEAR(v, 0.1, 1e-6);
}

TEST(ClawSpeed, NaNUntilFullAndAfterAGap)
{
  std::deque<ClawSample> buf;
  const Eigen::Vector3d claw{1.6, 0.25, 1.02};
  constexpr double dt = 0.105;
  // fewer than 4 samples: no speed; the 4th (span 0.315 s >= kClawSpeedMinSpan) gives one
  for (int k = 0; k < 3; ++k) {
    EXPECT_TRUE(std::isnan(clawSpeed(&buf, claw, at(dt * k)))) << "tick " << k;
  }
  EXPECT_TRUE(std::isfinite(clawSpeed(&buf, claw, at(dt * 3))));
  // a 0.3 s gap (> kClawSpeedMaxGap) clears the samples: NaN again until refilled
  const double t1 = dt * 3 + 0.3;
  EXPECT_TRUE(std::isnan(clawSpeed(&buf, claw, at(t1))));
  EXPECT_EQ(buf.size(), 1u);
  for (int k = 1; k < 3; ++k) {
    EXPECT_TRUE(std::isnan(clawSpeed(&buf, claw, at(t1 + dt * k)))) << "refill tick " << k;
  }
  EXPECT_TRUE(std::isfinite(clawSpeed(&buf, claw, at(t1 + dt * 3))));
}
