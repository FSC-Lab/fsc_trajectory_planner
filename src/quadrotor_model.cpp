// MIT License
// Copyright (c) 2026 FSC Lab

#include "fsc_trajectory_planner/quadrotor_model.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fsc_trajectory_planner
{

namespace
{

// FSC Tarot 650 as simulated by fsc_PegasusSimulator (t650_params.py,
// 2026-08-06 mass model; inertia flight-identified 2026-08-13) and flown by
// fsc_autopilot_ros2's T650 direct-actuation configs. MUST match the model the
// RCCM was trained for (CCM_quad configs/config_quad_torque.py VEHICLE 't650').
QuadrotorModel makeT650()
{
  QuadrotorModel m;
  m.name = "t650";
  m.mass = 3.033921;
  m.inertia << 0.077, 0.0089, 0.0,
    0.0089, 0.06408172, 0.0,
    0.0, 0.0, 0.065004565;
  m.gravity = 9.81;
  // MN4010 rotor lag, lambda = 10.0265 1/s (t650_params.py ROTOR_LAMBDA).
  m.rotor_time_constant = 0.0997;
  return m;
}

// FSC X650 (MN4014 + 15x5), flown from 2026-10 on: total mass 3.25 kg with battery;
// inertia = the X650 sim's (fsc_PegasusSimulator x650_params.py INERTIA_DIAG),
// a placeholder until identified from flight data. MUST match the model the
// RCCM was trained for (CCM_quad configs/config_quad_torque.py VEHICLE 'x650').
QuadrotorModel makeX650()
{
  QuadrotorModel m;
  m.name = "x650";
  m.mass = 3.25;
  m.inertia = Eigen::Vector3d(0.05777498, 0.06408172, 0.065004565).asDiagonal();
  m.gravity = 9.81;
  // MN4014 rotor lag, lambda = 10.51 1/s (MN_4014_15x5 bench report, x650_params.py ROTOR_LAMBDA).
  m.rotor_time_constant = 0.0951;
  return m;
}

// The 1 kg reference quad of CCM_quad's 'generic' vehicle.
QuadrotorModel makeGeneric()
{
  QuadrotorModel m;
  m.name = "generic";
  m.mass = 1.0;
  m.inertia = Eigen::Vector3d(0.01, 0.01, 0.03).asDiagonal();
  m.gravity = 9.81;
  m.rotor_time_constant = 0.0;
  return m;
}

struct UnitJets
{
  Eigen::Vector3d u, du, ddu;
};

// z = a / |a| and its first two time derivatives from a, a', a''.
UnitJets unitJets(const Eigen::Vector3d & a, const Eigen::Vector3d & ad, const Eigen::Vector3d & add)
{
  const double n = a.norm();
  if (!(n > 1e-9)) {
    throw std::domain_error("flatReference: vanishing vector (free fall or heading parallel to thrust)");
  }
  UnitJets out;
  out.u = a / n;
  const double nd = out.u.dot(ad);
  out.du = (ad - nd * out.u) / n;
  const double ndd = out.du.dot(ad) + out.u.dot(add);
  out.ddu = (add - ndd * out.u - 2.0 * nd * out.du) / n;
  return out;
}

Eigen::Vector3d veeSkew(const Eigen::Matrix3d & M)
{
  const Eigen::Matrix3d S = 0.5 * (M - M.transpose());
  return {S(2, 1), S(0, 2), S(1, 0)};
}

}  // namespace

const std::map<std::string, QuadrotorFactory> & quadrotorModelFactories()
{
  static const std::map<std::string, QuadrotorFactory> factories{
    {"t650", makeT650},
    {"x650", makeX650},
    {"generic", makeGeneric},
  };
  return factories;
}

std::vector<std::string> quadrotorModelNames()
{
  std::vector<std::string> names;
  for (const auto & kv : quadrotorModelFactories()) {names.push_back(kv.first);}
  return names;
}

QuadrotorModel makeQuadrotorModel(const std::string & name)
{
  const auto & f = quadrotorModelFactories();
  const auto it = f.find(name);
  if (it == f.end()) {
    std::ostringstream oss;
    oss << "unknown quadrotor model '" << name << "'; known:";
    for (const auto & n : quadrotorModelNames()) {oss << " " << n;}
    throw std::runtime_error(oss.str());
  }
  return it->second();
}

QuadrotorReference flatReference(
  const QuadrotorModel & model, const PositionJets & jets, const YawJets & yaw,
  const Eigen::Vector3d & disturbance, const Eigen::Vector3d & angular_disturbance)
{
  const Eigen::Vector3d a = jets.a + Eigen::Vector3d(0.0, 0.0, model.gravity) - disturbance;
  const UnitJets z = unitJets(a, jets.j, jets.s);

  const double cs = std::cos(yaw.psi), sn = std::sin(yaw.psi);
  const Eigen::Vector3d xc(cs, sn, 0.0);
  const Eigen::Vector3d xc_perp(-sn, cs, 0.0);
  const Eigen::Vector3d xcd = yaw.dpsi * xc_perp;
  const Eigen::Vector3d xcdd = yaw.ddpsi * xc_perp - yaw.dpsi * yaw.dpsi * xc;

  const Eigen::Vector3d c = z.u.cross(xc);
  const Eigen::Vector3d cd = z.du.cross(xc) + z.u.cross(xcd);
  const Eigen::Vector3d cdd = z.ddu.cross(xc) + 2.0 * z.du.cross(xcd) + z.u.cross(xcdd);
  const UnitJets y = unitJets(c, cd, cdd);

  const Eigen::Vector3d x = y.u.cross(z.u);
  const Eigen::Vector3d xd = y.du.cross(z.u) + y.u.cross(z.du);
  const Eigen::Vector3d xdd = y.ddu.cross(z.u) + 2.0 * y.du.cross(z.du) + y.u.cross(z.ddu);

  Eigen::Matrix3d R, Rd, Rdd;
  R << x, y.u, z.u;
  Rd << xd, y.du, z.du;
  Rdd << xdd, y.ddu, z.ddu;

  QuadrotorReference ref;
  ref.position = jets.p;
  ref.velocity = jets.v;
  ref.acceleration = jets.a;
  ref.R = R;
  ref.omega = veeSkew(R.transpose() * Rd);                 // omega^ = R^T R'
  const Eigen::Vector3d omega_dot = veeSkew(R.transpose() * Rdd);  // skew part of R^T R''
  ref.thrust = model.mass * a.norm();
  ref.torque = model.inertia * (omega_dot - angular_disturbance) + ref.omega.cross(model.inertia * ref.omega);
  ref.yaw = yaw.psi;
  return ref;
}

}  // namespace fsc_trajectory_planner
