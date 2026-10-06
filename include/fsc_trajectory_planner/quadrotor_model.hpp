// MIT License
// Copyright (c) 2026 FSC Lab
//
// QUADROTOR models for the torque-mode (R)CCM pipeline, and their
// differential-flatness map. Separate from vehicle_model.hpp on purpose: a
// VehicleModel there is an airframe + 4-joint ARM (WholeBodyParams, joint box,
// arm home pose) and every consumer of it assumes the arm. A bare quadrotor
// needs only mass, inertia and the rotor time constant, so it has its own
// small registry. Adding a quadrotor = one entry in quadrotorModelFactories().
//
// The flatness map is a line-for-line port of CCM_quad's
// planners/planner_quad_torque.flat_reference (closed form, validated there
// against nested forward-mode AD to 2e-15), so the reference the ROS stack
// flies is the one the controller was trained and simulated against.

#ifndef FSC_TRAJECTORY_PLANNER_QUADROTOR_MODEL_HPP_
#define FSC_TRAJECTORY_PLANNER_QUADROTOR_MODEL_HPP_

#include <Eigen/Dense>

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace fsc_trajectory_planner
{

struct QuadrotorModel
{
  std::string name;
  double mass{1.0};                                   // [kg]
  Eigen::Matrix3d inertia{Eigen::Matrix3d::Identity()};  // [kg m^2], body FLU
  double gravity{9.81};                               // [m/s^2]
  // First-order rotor (wrench) time constant [s] used for the reference
  // COMMAND u*_c = u* + tau * du*/dt. 0 = instantaneous actuators.
  double rotor_time_constant{0.0};
};

using QuadrotorFactory = std::function<QuadrotorModel()>;
const std::map<std::string, QuadrotorFactory> & quadrotorModelFactories();
std::vector<std::string> quadrotorModelNames();
// Throws std::runtime_error naming the known models on an unknown name.
QuadrotorModel makeQuadrotorModel(const std::string & name);

// Flat outputs: position and its first four derivatives, yaw and two.
struct PositionJets
{
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};  // [m]
  Eigen::Vector3d v{Eigen::Vector3d::Zero()};  // [m/s]
  Eigen::Vector3d a{Eigen::Vector3d::Zero()};  // [m/s^2]
  Eigen::Vector3d j{Eigen::Vector3d::Zero()};  // [m/s^3]
  Eigen::Vector3d s{Eigen::Vector3d::Zero()};  // [m/s^4]
};

struct YawJets
{
  double psi{0.0};
  double dpsi{0.0};
  double ddpsi{0.0};
};

// One sample of the rigid-body reference (x*, u*). World ENU, body FLU.
struct QuadrotorReference
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acceleration{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};  // body FLU -> world ENU
  Eigen::Vector3d omega{Eigen::Vector3d::Zero()};  // [rad/s], body FLU
  double thrust{0.0};                              // [N], along body +z
  Eigen::Vector3d torque{Eigen::Vector3d::Zero()};  // [N m], body FLU
  double yaw{0.0};
};

// Differential flatness of  m p'' = -m g e3 + T R e3 + m d,
// R' = R omega^,  J omega' = tau - omega x J omega.
//   a_T = p'' + g e3 - d,  T = m |a_T|,  z_b = a_T / |a_T|,
//   y_b = z_b x x_c / |.| (x_c = heading),  x_b = y_b x z_b,
//   omega^ = R^T R',  tau = J omega' + omega x J omega.
// `d` is a disturbance ACCELERATION treated as constant (e.g. UDE / mass), world ENU, and
// `d_w` an angular-acceleration disturbance, body FLU:  J omega' = tau - omega x J omega + J d_w,
// so the reference torque is tau = J (omega' - d_w) + omega x J omega.
// Throws std::domain_error if the thrust vector vanishes (free fall) or is
// parallel to the heading.
QuadrotorReference flatReference(
  const QuadrotorModel & model, const PositionJets & jets, const YawJets & yaw,
  const Eigen::Vector3d & disturbance = Eigen::Vector3d::Zero(),
  const Eigen::Vector3d & angular_disturbance = Eigen::Vector3d::Zero());

}  // namespace fsc_trajectory_planner

#endif  // FSC_TRAJECTORY_PLANNER_QUADROTOR_MODEL_HPP_
