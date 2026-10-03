// MIT License
// Copyright (c) 2026 FSC Lab
//
// ARM-SWEEP LEG: a rest-to-rest whole-body move during which chosen joints
// sweep back and forth between two limits. It is the pick-and-place mode's
// "execute to place" leg (pick_place.hpp), flown to show the whole-body law
// carrying a payload that moves in 3-D while the vehicle travels.
//
// Every channel is a FLAT OUTPUT prescribed in closed form against time:
//
//   x_c(t)   = x_c0 + (x_c1 - x_c0) s(t/T)          s: nonic, rest through SNAP
//   psi(t)   = phi0 + (phi1 - phi0) s(t/T)             at both ends
//   q_lin(t) = q0 + (q1 - q0) s(t/T)
//   q_j(t)   = (1 - w) q_lin_j + w (c_j + a_j sin(omega (t - r) + phase_j))
//              for a swept joint (band [lo_j, hi_j], c = mid, a = half-width),
//              q_lin_j for the others
//   w(t)     plateau window: nonic ramp-in over r = ramp_frac T, 1, ramp-out
//   omega    = 2 pi cycles / (T - 2 r): `cycles` whole sweeps on the plateau
//
// so wb_law's flatState() maps them straight to a compatible reference: the
// CoM is prescribed, not solved for, and nothing needs a fixed point (unlike
// the EE-trajectory mode, nothing is prescribed in task space). The CoM goes
// in a straight line while the arm swings; the airframe moves to carry the
// arm's CoM excursion, and that is the demonstration.
//
// Nonic rather than septic phasing on x_c: flatState() reads x_c up to its 4th
// derivative, and a septic phase has a SNAP step at both ends -- a step in
// the body-torque reference at the hold joins. The nonic is zero through snap.
//
// The duration is the SHORTEST T (bracket, then bisection to rel_tol) at
// which every check passes on a dense grid: joint box, sigma_nd, CoM speed and
// acceleration, yaw rate, joint rate, joint torque and rotor force through
// inverseInputs(). Rates fall as T grows, so a sweep that fails at T_max is
// refused with the bound that binds there (geometry or static torque).

#ifndef FSC_TRAJECTORY_PLANNER_ARM_SWEEP_PLANNER_HPP_
#define FSC_TRAJECTORY_PLANNER_ARM_SWEEP_PLANNER_HPP_

#include <cmath>
#include <memory>
#include <string>

#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

struct ArmSweepOptions
{
  // Sweep band per joint [rad, MODEL convention]. A joint whose lo == hi is
  // not swept: it moves straight from its start angle to its goal angle.
  // Default (2026-10-02, user decision after the Isaac tilt-watchdog trips at
  // +-25 deg): +-10 deg on each swept joint -- the arm yaw q1 [-10, 10] (the
  // payload sideways) and the shoulder q2 [17.5, 37.5] (the old band's centre
  // +-10: out and up), a quarter cycle apart.
  VecN lo{(VecN() << -10.0, 17.5, 0.0, 0.0).finished() * (M_PI / 180.0)};
  VecN hi{(VecN() << 10.0, 37.5, 0.0, 0.0).finished() * (M_PI / 180.0)};
  VecN phase{(VecN() << 0.0, 90.0, 0.0, 0.0).finished() * (M_PI / 180.0)};
  int cycles{2};
  double ramp_frac{0.2};       // window ramp-in and ramp-out, each, as a fraction of T
  // bounds (the node fills these from its own transition bounds)
  double v_max{0.30}, a_max{0.15}, w_max{0.30};
  double qdot_max{0.5};        // [rad/s]
  double tau_joint_max{3.0};   // [N.m]; <= 0 disables
  bool rotor_bounds{true};
  double sigma_nd_min{0.10};
  double T_min{8.0}, T_max{120.0};
  int n_check{401};
  double rel_tol{0.02};
};

struct ArmSweepDiag
{
  double T{0.0};
  int probes{0};
  double solve_ms{0.0};
  double peak_v{0.0}, peak_a{0.0}, peak_w{0.0}, peak_qdot{0.0};
  double peak_tau_joint{0.0}, rotor_lo{0.0}, rotor_hi{0.0};
  double min_sigma_nd{0.0};
  double peak_ee_speed{0.0};
  VecN q_min_deg{VecN::Zero()}, q_max_deg{VecN::Zero()};
  bool feasible{false};
  std::string violation;       // first bound that failed, or empty
  std::string summary;
};

class ArmSweepPlanner
{
public:
  // Shortest feasible duration. Throws std::runtime_error naming the bound
  // when no T in [T_min, T_max] passes; *diag is filled either way.
  static std::unique_ptr<Trajectory> plan(
    const VehicleModel & v, const RestSpec & rest0, const RestSpec & rest1,
    const ArmSweepOptions & o, ArmSweepDiag * diag = nullptr);

  // The same leg at a FIXED duration; throws when it is infeasible there.
  static std::unique_ptr<Trajectory> planAt(
    const VehicleModel & v, const RestSpec & rest0, const RestSpec & rest1,
    const ArmSweepOptions & o, double T, ArmSweepDiag * diag = nullptr);
};

// Nonic rest-to-rest phase at u in [0, 1] (clamped): out[0] = s, out[k] =
// d^k s / du^k for k = 1..4. Zero velocity through snap at both ends.
void nonicPhase(double u, double out[5]);

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_ARM_SWEEP_PLANNER_HPP_
