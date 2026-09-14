// MIT License
// Copyright (c) 2026 FSC Lab
//
// END-EFFECTOR TRAJECTORY MODE: a periodic end-effector pose trajectory
// (circle / figure-8, yaw along the tangent) turned into a compatible
// whole-body reference for the z-x-x-z aerial manipulator.
//
// Pipeline (the MATLAB task-space planner's main_redundant_zyxx, adapted):
//   1. EE pose curve   p_e(tau), R_e(tau) = Rz(psi_tan(tau) - pi/2) Rx(beta_e)
//                      in the MODEL frame: heading along the curve tangent,
//                      fold angle beta_e about it (see ee_fold_deg).
//   2. time scaling    tau(t): min-snap ramp-in, constant rate s, ramp-out;
//                      the run starts and ends AT REST on the same pose and
//                      covers `laps` whole periods.
//   3. flat outputs    thrust direction r -> R2(r) (minimal tilt), A = R2^T R_e,
//                      (psi, beta, gamma) = zxz(A): the first z angle is the
//                      DRONE yaw (q1 is fixed at 0), beta = q2 + q3 with q2
//                      the ASSIGNED slow sinusoid, gamma = q4; x_c from the
//                      chain (A6). The thrust direction is solved by Picard
//                      iteration on the feasibility residual the MATLAB
//                      script minimises with fmincon (r = normalize(xdd_c + g e3)),
//                      with x_c represented by a clamped B-spline.
//   4. fitting         x_c (deg 7), psi and q (deg 5) as B-splines over the run.
//   5. reference       wb_law's flatState() on the fitted flat outputs -> the
//                      16-field WbReference; inverseInputs() for the bounds.
//   6. checks          joint box, beta >= beta_min, sigma_nd >= margin,
//                      |v|,|a|,|psi_dot|,|q_dot|, joint torque, rotor force,
//                      FK round trip of the EE pose vs the prescribed curve.
//   7. time scale      maxTimeScale(): bisection on the largest s passing 6.

#ifndef FSC_TRAJECTORY_PLANNER_EE_TRAJECTORY_PLANNER_HPP_
#define FSC_TRAJECTORY_PLANNER_EE_TRAJECTORY_PLANNER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "fsc_trajectory_planner/trajectory.hpp"

namespace fsc_trajectory_planner
{

struct EeShape
{
  std::string type{"circle"};   // "circle" | "figure8"
  double radius{0.5};           // circle radius [m]
  double fig8_a{0.5};           // figure-8 half-length along the heading [m]
  double fig8_b{0.25};          // figure-8 half-width across it [m]
  double lap_time{24.0};        // period of one lap at time scale 1 [s]
  int laps{2};
  bool ccw{true};               // circle turns left of the start heading
};

struct EeTrajectoryOptions
{
  double time_scale{1.0};       // s: tau runs at s times real time on the laps
  double ramp_time{4.0};        // min-snap ramp-in / ramp-out [s]
  // EE attitude about its heading: beta_e = q2 + q3 at rest. 80 deg is the
  // flown home fold; 0 would be a level EE = the wrist singularity of this arm.
  double ee_fold_deg{80.0};
  // the assigned redundancy DoF: q2(tau) = c + a sin(2 pi tau / P)
  double q2_center_deg{40.0};
  double q2_amp_deg{8.0};
  double q2_period_s{0.0};      // 0 -> one lap
  // bounds
  double v_max{0.30}, a_max{0.15}, w_max{0.30}, qdot_max{0.5};
  double tau_joint_max{3.0};
  bool rotor_bounds{true};
  double sigma_nd_min{0.10};
  double beta_min_deg{5.0};
  // numerics
  double sample_rate{20.0};     // Picard / fit grid [Hz]
  // B-spline spans per second. Deliberately COARSE: the fixed-point map
  // amplifies a thrust-direction perturbation at frequency w by L w^2 / g
  // (L ~ 0.2 m arm lever), i.e. above ~1 Hz it is not a contraction, so the
  // basis must not be able to represent such content -- 0.5 s spans of
  // degree 7 carry the 0.05-0.3 Hz laps and the 4 s ramps, and with the
  // relaxation below every mode they can represent still contracts.
  double spans_per_s{2.0};
  // under-relaxation of the thrust update, r <- r + relax (r_new - r): with
  // relax < 2 / (1 + L w^2 / g) every representable mode contracts.
  double relax{0.5};
  int maxit{60};
  double tol{1e-8};
  int n_check{0};               // 0 -> the fit grid
};

struct EeTrajectoryDiag
{
  double T_total{0.0}, T_lap{0.0}, ramp_time{0.0};
  int laps{0};
  double s{0.0};
  int iterations{0};
  double solve_ms{0.0};
  double max_dyn_defect{0.0};     // (A6) residual of the fitted x_c
  double ee_pos_err_max{0.0};     // FK(fitted) vs prescribed EE position [m]
  double ee_rot_err_max_deg{0.0}; // FK(fitted) vs prescribed EE attitude [deg]
  double rest_err{0.0};           // start/end vs the rest configuration [m]
  double min_sigma_nd{0.0}, min_beta_deg{0.0};
  double peak_v{0.0}, peak_a{0.0}, peak_w{0.0}, peak_qdot{0.0};
  double peak_tau_joint{0.0}, rotor_lo{0.0}, rotor_hi{0.0};
  VecN q_min_deg{VecN::Zero()}, q_max_deg{VecN::Zero()};
  bool feasible{false};
  std::string violation;          // first bound that failed, or empty
  std::string summary;
};

class EeTrajectoryPlanner
{
public:
  // The rest the run starts and ends on, anchored on the hold: EE at the
  // hold's EE position, heading = the hold's nose, fold beta_e, q1 = 0,
  // q2 = q2(0), q4 = 0.
  static RestSpec startRest(
    const VehicleModel & v, const RestSpec & hold, const EeShape & shape,
    const EeTrajectoryOptions & o);

  // Plan the whole run at o.time_scale. Throws std::runtime_error with the
  // violated bound when infeasible; *diag is filled either way.
  static std::unique_ptr<Trajectory> plan(
    const VehicleModel & v, const RestSpec & hold, const EeShape & shape,
    const EeTrajectoryOptions & o, EeTrajectoryDiag * diag);

  // Largest time scale (to `rel_tol`) at which plan() passes every check.
  static double maxTimeScale(
    const VehicleModel & v, const RestSpec & hold, const EeShape & shape,
    const EeTrajectoryOptions & o, double s_hi = 6.0, double rel_tol = 0.02);
};

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_EE_TRAJECTORY_PLANNER_HPP_
