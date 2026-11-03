// MIT License
// Copyright (c) 2026 FSC Lab
//
// The claw's speed for the autonomous insert / drop gates (2026-10-30), over the
// samples of the last kClawSpeedWindow s: the LARGER of its least-squares velocity
// and its spread as a speed (2 x the largest deviation from the samples' mean / the
// span). A two-sample speed read the 10 Hz samples' few-mm jitter as 0.05-0.08 m/s
// on a claw holding within 15 mm (sim, layout_auto1), so the 0.03 m/s gate never
// held; a displacement over 0.5 s alone was blind to a ~1 Hz swing
// (drop_auto_check). The fit rejects jitter, the spread catches a swing.
//
// NaN until 4 samples span kClawSpeedMinSpan s; a gap > 0.25 s starts afresh.
// THE MINIMUM SPAN IS 0.3 s, NOT 0.4 (2026-10-31, the 10-31 hardware mail flights):
// at 10 Hz a 0.5 s window keeps the samples 0.0 .. ~0.5 s old, and the one ~0.4 s
// old is 4 periods back -- a few ms of tick jitter put it at 0.39x s while the
// ~0.5 s one had just been dropped, so ONE tick read NaN, failed the speed gate
// and restarted the release dwell. 4cm held inside the release window for 1.7 s
// and never released; 5cm released on its third dwell. A full buffer still spans
// 0.4-0.5 s, so the estimate itself is unchanged.

#ifndef FSC_TRAJECTORY_PLANNER_CLAW_SPEED_HPP_
#define FSC_TRAJECTORY_PLANNER_CLAW_SPEED_HPP_

#include <algorithm>
#include <chrono>
#include <deque>
#include <limits>
#include <utility>

#include <Eigen/Dense>

namespace fsc_trajectory_planner
{

constexpr double kClawSpeedWindow = 0.5;    // [s] the samples kept
constexpr double kClawSpeedMinSpan = 0.3;   // [s] the oldest kept sample's age before a speed is given
constexpr double kClawSpeedMaxGap = 0.25;   // [s] a longer gap between samples starts afresh

using ClawSample = std::pair<std::chrono::steady_clock::time_point, Eigen::Vector3d>;

// Adds `claw` at `now` to `buf` and returns the speed [m/s] (NaN while too few).
inline double clawSpeed(std::deque<ClawSample> * buf, const Eigen::Vector3d & claw,
  std::chrono::steady_clock::time_point now)
{
  const auto age = [&](const std::chrono::steady_clock::time_point & t) {
      return std::chrono::duration<double>(now - t).count();
    };
  if (!buf->empty() && age(buf->back().first) > kClawSpeedMaxGap) {buf->clear();}
  buf->emplace_back(now, claw);
  while (!buf->empty() && age(buf->front().first) > kClawSpeedWindow + 1e-6) {buf->pop_front();}
  const double span = age(buf->front().first);
  if (buf->size() < 4 || span < kClawSpeedMinSpan) {return std::numeric_limits<double>::quiet_NaN();}
  double tm = 0.0;
  Eigen::Vector3d xm = Eigen::Vector3d::Zero();
  for (const auto & s : *buf) {
    tm += -age(s.first);
    xm += s.second;
  }
  tm /= static_cast<double>(buf->size());
  xm /= static_cast<double>(buf->size());
  double stt = 0.0, rmax = 0.0;
  Eigen::Vector3d stx = Eigen::Vector3d::Zero();
  for (const auto & s : *buf) {
    const double dtm = -age(s.first) - tm;
    stt += dtm * dtm;
    stx += dtm * (s.second - xm);
    rmax = std::max(rmax, (s.second - xm).norm());
  }
  const double fit = stt > 1e-9 ? (stx / stt).norm() : 0.0;
  return std::max(fit, 2.0 * rmax / span);
}

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_CLAW_SPEED_HPP_
