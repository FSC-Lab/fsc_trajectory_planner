// MIT License
// Copyright (c) 2026 FSC Lab
//
// whole_body_trajectory_planner — the setpoint/trajectory processor between
// the two ground stations and the whole-body direct-actuation node. C++ port
// of fsc_autopilot_ros2's planner/whole_body_planner.py (2026-09-14), with the
// planning maths in this package's own library instead of the Pegasus
// extension.
//
//   SAFETY   silent. The flight node consumes the drone-GS reference directly
//            and its internal builder covers DIRECT if this node is absent.
//   DIRECT   streams WholeBodyReference continuously, so the flight node's
//            stream-freshness gate makes it the law's reference:
//     HOLD        stream the rest reference captured at DIRECT entry
//     PENDING     a drone-GS send is captured, NOT executed; its base target
//                 is republished for the arm GS's inertial EE tab
//     CALCULATING/PLANNED/INFEASIBLE
//                 any target change replans on a worker thread: goal = the
//                 pending base (else the hold base) + the inertial EE target
//                 (else ride-along: keep the current arm pose); backend
//                 chosen by the `planner` parameter through the registry
//     EXECUTING   after the operator's explicit Send (std_srvs/Trigger) the
//                 plan streams sample-by-sample; on completion the goal
//                 becomes the new HOLD
//     EE TRAJECTORY MODE (2026-09-15): a periodic end-effector trajectory
//                 (circle / figure-8) selected from the arm ground station
//                 is planned as a whole run (ramp-in, laps, ramp-out) by
//                 ee_trajectory_planner; `go_to_start` plans and executes the
//                 compatible transition to its start rest, `start` streams it
//                 -- refused unless the vehicle and arm are at that rest.
//   Mode leaves DIRECT at ANY point -> streaming stops instantly.
//
// FRAMES: the ROS boundary is the ACTUAL world/FLU convention (odometry, GS
// yaw, EE targets); the planner and the streamed message are MODEL frame.
//     R0_model = R0_actual @ R_MODEL          phi_model = psi_actual - pi/2
//
// Every topic and service is RELATIVE, so `--ros-args -r __ns:=/uav_N` (the
// launch file's uav_prefix) namespaces the whole interface per vehicle.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <px4_msgs/msg/vehicle_attitude.hpp>

#include <fsc_autopilot_ros2_msgs/msg/position_controller_reference.hpp>
#include <fsc_autopilot_ros2_msgs/msg/whole_body_reference.hpp>

#include "fsc_trajectory_planner/ee_trajectory_planner.hpp"
#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/trajectory.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"

namespace fsc_trajectory_planner
{

namespace
{

using geometry_msgs::msg::PoseStamped;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::JointState;
using std_msgs::msg::Float64;
using std_msgs::msg::Float64MultiArray;
using std_msgs::msg::String;
using std_srvs::srv::Trigger;
using trajectory_msgs::msg::JointTrajectory;
using trajectory_msgs::msg::JointTrajectoryPoint;
using px4_msgs::msg::VehicleAttitude;
using fsc_autopilot_ros2_msgs::msg::PositionControllerReference;
using fsc_autopilot_ros2_msgs::msg::WholeBodyReference;

using Clock = std::chrono::steady_clock;

const char * const kArmJointNames[kNumJoints] = {"joint1", "joint2", "joint3",
  "joint4"};

// Odometry older than this must not be composed into the world-frame
// current_ee: a frozen mocap feed masquerading as a live drone pose is the
// failure shape of the 2026-08-03 feedback-loss incident.
constexpr double kOdomFreshS = 0.5;
// The capture is a REST spec; warn if the arm is still moving at capture.
constexpr double kArmRestQdot = 0.05;

// NED/FRD -> ENU/FLU, byte-for-byte the C++ getNEDqFromENUq (ros2_support).
const Mat3 kRIE = (Mat3() << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
  .finished();
const Mat3 kRBV = (Mat3() << 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0)
  .finished();

double wrapPi(double a) {return std::atan2(std::sin(a), std::cos(a));}

double yawFromQuat(double qx, double qy, double qz, double qw)
{
  return std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
}

double secondsSince(const std::optional<Clock::time_point> & t)
{
  if (!t.has_value()) {return 1e300;}
  return std::chrono::duration<double>(Clock::now() - *t).count();
}

std::string degList(const VecN & q, int prec = 1)
{
  std::ostringstream m;
  m.setf(std::ios::fixed);
  m.precision(prec);
  m << "[";
  for (int j = 0; j < kNumJoints; ++j) {
    m << q(j) * 180.0 / M_PI << (j + 1 < kNumJoints ? ", " : "");
  }
  m << "]";
  return m.str();
}

}  // namespace

class WholeBodyTrajectoryPlanner : public rclcpp::Node
{
public:
  WholeBodyTrajectoryPlanner()
  : rclcpp::Node("whole_body_trajectory_planner")
  {
    // ---- parameters -------------------------------------------------------
    // The airframe + arm model, from the registry (vehicle_model.cpp).
    vehicle_name_ = declare_parameter<std::string>("vehicle", "t650_aerial_manipulator");
    // "straight_line" (Picard, flight-validated) or "bspline" (flat outputs,
    // constraints enforced). Default bspline: what the sim configs fly.
    planner_name_ = declare_parameter<std::string>("planner", "bspline");
    stream_rate_ = declare_parameter<double>("stream_rate", 100.0);
    declare_parameter<double>("v_max", 0.30);
    declare_parameter<double>("a_max", 0.15);
    declare_parameter<double>("w_max", 0.30);
    declare_parameter<double>("t_min", 3.0);
    declare_parameter<double>("dw_max", 0.60);
    declare_parameter<double>("t_max", 40.0);
    declare_parameter<double>("tau_joint_max", 3.0);
    declare_parameter<bool>("rotor_bounds", true);
    declare_parameter<int>("n_check", 101);
    // The folded HOME pose the arm returns to (Go Home). Same numbers as the
    // arm controller's home_position.
    const auto home = declare_parameter<std::vector<double>>(
      "home_pose", std::vector<double>{0.0, 0.698132, 0.698132, 0.0});
    // Bare-airframe CoM relative to the body origin, MODEL frame [m]. MUST
    // equal the flight node's wb_base_com_x/y/z.
    const auto base_com = declare_parameter<std::vector<double>>(
      "base_com", std::vector<double>{0.0, 0.0, 0.0});
    // World-EE-anchored HOLD: re-solve the arm IK against the CURRENT base
    // pose so the EE holds its inertial position. OFF by default (2026-08-31):
    // the re-solve moves x_cd with base drift and leaves the position loop
    // effectively open. Keep false until the reference split is fixed.
    hold_ee_world_ = declare_parameter<bool>("hold_ee_world", false);
    // Sign map ARM convention <-> MODEL convention on joint_states in and the
    // arm reference out. Identity = Isaac; URDF/Dynamixel hardware rigs pass
    // [-1, 1, 1, -1] (joint1/joint4 axes are opposite).
    const auto sign = declare_parameter<std::vector<double>>(
      "arm_joint_sign", std::vector<double>{1.0, 1.0, 1.0, 1.0});
    // Topic layout. The planner-owned topics live under `topic_prefix` so the
    // arm ground station, the Isaac visualiser and the campaign drivers keep
    // resolving them; the inputs default to the flight stack's names.
    const auto prefix = declare_parameter<std::string>("topic_prefix", "whole_body_planner");
    const auto mode_topic = declare_parameter<std::string>(
      "mode_topic", "fsc_autopilot_ros2/whole_body_direct_actuation/mode");
    const auto ref_topic = declare_parameter<std::string>(
      "reference_topic", "fsc_autopilot_ros2/whole_body_direct_actuation/reference");
    const auto gs_topic = declare_parameter<std::string>(
      "gs_reference_topic", "fsc_autopilot_ros2/position_controller/reference");
    const auto odom_topic = declare_parameter<std::string>(
      "odom_topic", "state_estimator/local_position/odom");
    const auto att_topic = declare_parameter<std::string>(
      "attitude_topic", "fmu/out/vehicle_attitude");
    const auto js_topic = declare_parameter<std::string>(
      "joint_states_topic", "fsc_open_manipulator/joint_states");
    const auto arm_ref_topic = declare_parameter<std::string>(
      "arm_reference_topic",
      "fsc_open_manipulator/external_torque_controller/reference_joint_trajectory");

    // --- end-effector trajectory mode -------------------------------------
    declare_parameter<double>("ee_traj_circle_radius", 0.5);
    declare_parameter<double>("ee_traj_fig8_a", 0.5);
    declare_parameter<double>("ee_traj_fig8_b", 0.25);
    declare_parameter<double>("ee_traj_lap_time", 24.0);
    declare_parameter<int>("ee_traj_laps", 2);
    declare_parameter<bool>("ee_traj_ccw", true);
    declare_parameter<double>("ee_traj_ramp_time", 4.0);
    // EE attitude about its heading (= the fold q2+q3 at rest). 80 deg is the
    // flown home fold; 0 (a level EE) is this arm's wrist singularity and is
    // refused by the planner's beta/sigma_nd guards.
    declare_parameter<double>("ee_traj_fold_deg", 80.0);
    declare_parameter<double>("ee_traj_q2_center_deg", 40.0);
    declare_parameter<double>("ee_traj_q2_amp_deg", 8.0);
    declare_parameter<double>("ee_traj_q2_period_s", 0.0);
    declare_parameter<double>("ee_traj_qdot_max", 0.5);
    declare_parameter<double>("ee_traj_time_scale", 1.0);
    declare_parameter<double>("ee_traj_start_pos_tol", 0.05);
    declare_parameter<double>("ee_traj_start_yaw_tol_deg", 5.0);
    declare_parameter<double>("ee_traj_start_joint_tol_deg", 3.0);
    ee_time_scale_req_ = get_parameter("ee_traj_time_scale").as_double();

    if (home.size() != kNumJoints || base_com.size() != 3 || sign.size() != kNumJoints) {
      throw std::runtime_error("home_pose/arm_joint_sign need 4 values, base_com 3");
    }
    for (int j = 0; j < kNumJoints; ++j) {
      home_pose_(j) = home[j];
      joint_sign_(j) = sign[j];
      if (std::abs(sign[j]) != 1.0) {
        throw std::runtime_error("arm_joint_sign must be four values of +-1");
      }
    }
    VehicleOptions vo;
    vo.base_com = Vec3{base_com[0], base_com[1], base_com[2]};
    vehicle_ = makeVehicleModel(vehicle_name_, vo);
    planner_ = makePlanner(planner_name_);
    RCLCPP_INFO(
      get_logger(), "vehicle: %s (whole-body model total %.6f kg, base_com "
      "[%+.5f %+.5f %+.5f] m model frame -- must match the node's wb_base_com_*)",
      vehicle_->name.c_str(), vehicle_->params.totalMass(), vo.base_com(0),
      vo.base_com(1), vo.base_com(2));
    RCLCPP_INFO(
      get_logger(), "transition planner: %s (%s)", planner_->name().c_str(),
      planner_->description().c_str());

    // ---- ROS interfaces ---------------------------------------------------
    const auto latched = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    const auto px4_qos = rclcpp::QoS(rclcpp::KeepLast(5)).best_effort().durability_volatile();

    ref_pub_ = create_publisher<WholeBodyReference>(ref_topic, 10);
    status_pub_ = create_publisher<String>(prefix + "/status", latched);
    base_pub_ = create_publisher<PoseStamped>(prefix + "/pending_base", latched);
    // Solved GOAL joints for the arm GS's joint-limit row: published on every
    // plan attempt INCLUDING infeasible ones.
    joints_pub_ = create_publisher<Float64MultiArray>(prefix + "/target_joints", latched);
    // The WHOLE-BODY usable workspace as an (r, z) occupancy grid.
    ws_pub_ = create_publisher<Float64MultiArray>(prefix + "/workspace_rz", latched);
    // Isaac visualisation: planned path (latched) and current sample (~20 Hz),
    // 12 world-frame doubles per sample: x_cd, nose, r_ed, claw.
    viz_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/viz_path", latched);
    viz_pose_pub_ = create_publisher<Float64MultiArray>(prefix + "/viz_pose", 10);
    // The arm's CURRENT EE (grasp point) on the MEASURED joints, from THIS
    // node's model: inertial (gated on fresh odometry) and drone-body.
    cur_ee_pub_ = create_publisher<PoseStamped>(prefix + "/current_ee", 10);
    cur_ee_body_pub_ = create_publisher<PoseStamped>(prefix + "/current_ee_body", 10);
    // THE ARM REFERENCE IN DIRECT: this node is the arm's only reference
    // source while it streams (the arm planner owns it in SAFETY).
    arm_ref_pub_ = create_publisher<JointTrajectory>(arm_ref_topic, 10);

    // --- end-effector trajectory mode (arm GS "EE trajectory" tab) ---------
    ee_status_pub_ = create_publisher<String>(prefix + "/ee_trajectory/status", latched);
    ee_info_pub_ = create_publisher<Float64MultiArray>(prefix + "/ee_trajectory/info", latched);
    ee_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/ee_trajectory/path", latched);
    ee_start_err_pub_ = create_publisher<Float64MultiArray>(prefix + "/ee_trajectory/start_error", 10);
    ee_ref_pose_pub_ = create_publisher<PoseStamped>(prefix + "/ee_trajectory/reference_pose", 10);
    ee_select_sub_ = create_subscription<String>(
      prefix + "/ee_trajectory/select", 10, [this](const String & m) {onEeSelect(m);});
    ee_scale_sub_ = create_subscription<Float64>(
      prefix + "/ee_trajectory/time_scale", 10, [this](const Float64 & m) {onEeTimeScale(m);});
    ee_go_srv_ = create_service<Trigger>(
      prefix + "/ee_trajectory/go_to_start",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onEeGoToStart(r);});
    ee_start_srv_ = create_service<Trigger>(
      prefix + "/ee_trajectory/start",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onEeStart(r);});
    ee_err_timer_ = create_wall_timer(
      std::chrono::milliseconds(100), [this]() {publishEeStartError();});
    publishEeStatus("NONE");
    publishEePath(nullptr);

    mode_sub_ = create_subscription<String>(
      mode_topic, latched, [this](const String & m) {onMode(m);});
    gs_sub_ = create_subscription<PositionControllerReference>(
      gs_topic, 10, [this](const PositionControllerReference & m) {onGsReference(m);});
    // PX4 EKF2 attitude: the LAW's attitude source, so the two agree.
    att_sub_ = create_subscription<VehicleAttitude>(
      att_topic, px4_qos, [this](const VehicleAttitude & m) {onAttitude(m);});
    odom_sub_ = create_subscription<Odometry>(
      odom_topic, 10, [this](const Odometry & m) {onOdom(m);});
    js_sub_ = create_subscription<JointState>(
      js_topic, 10, [this](const JointState & m) {onJointStates(m);});
    ee_sub_ = create_subscription<PoseStamped>(
      prefix + "/ee_target", 10, [this](const PoseStamped & m) {onEeTarget(m);});

    send_srv_ = create_service<Trigger>(
      prefix + "/send",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onSend(r);});
    clear_srv_ = create_service<Trigger>(
      prefix + "/clear",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onClear(r);});
    home_srv_ = create_service<Trigger>(
      prefix + "/go_home",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onGoHome(r);});

    stream_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / stream_rate_), [this]() {streamTick();});
    ee_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / 15.0), [this]() {currentEeTick();});
    viz_decim_period_ = std::max(1, static_cast<int>(std::lround(stream_rate_ / 20.0)));

    publishStatus();
    publishWorkspace();
    // Seed the latched curve as EMPTY so a visualiser started before the
    // first plan learns there is nothing to draw.
    publishVizPath(nullptr);
    RCLCPP_INFO(
      get_logger(), "whole-body trajectory planner up: silent in SAFETY, streams the "
      "compatible reference in DIRECT (HOLD -> plan -> Send -> EXECUTING).");
  }

  ~WholeBodyTrajectoryPlanner() override
  {
    if (worker_.joinable()) {worker_.join();}
    if (ee_worker_.joinable()) {ee_worker_.join();}
  }

private:
  // ---------------------------------------------------------------- frames
  // (x_b_actual, R0_actual): position from odometry, attitude from PX4.
  bool odomPair(Vec3 * x_b, Mat3 * r0) const
  {
    if (!odom_p_.has_value() || !att_R_.has_value()) {return false;}
    *x_b = *odom_p_;
    *r0 = *att_R_;
    return true;
  }
  // Age of the OLDER half: either going stale withholds the pair.
  double odomAge() const
  {
    return std::max(secondsSince(odom_p_time_), secondsSince(att_time_));
  }

  // Rest spec (x_b, phi, q) from MEASUREMENT throughout: base position from
  // odometry, base heading from PX4, arm pose from the encoders.
  std::optional<RestSpec> restFromMeasurements()
  {
    Vec3 x_b;
    Mat3 r0_actual;
    if (!odomPair(&x_b, &r0_actual)) {
      if (!att_warned_ && odom_p_.has_value()) {
        att_warned_ = true;
        RCLCPP_ERROR(
          get_logger(), "No PX4 attitude yet on the attitude topic -- the hold "
          "CANNOT be captured. Odometry is arriving, so this is the attitude "
          "topic alone (agent down, or px4_msgs QoS mismatch). The planner "
          "stays silent and the node falls back to its internal builder.");
      }
      return std::nullopt;
    }
    if (!q_meas_.has_value()) {return std::nullopt;}
    const double qd_max = qdot_meas_.cwiseAbs().maxCoeff();
    if (qd_max > kArmRestQdot) {
      RCLCPP_WARN(
        get_logger(), "arm is NOT at rest at capture: max |qdot| = %.3f rad/s "
        "(> %.2f rad/s). Settle the arm before engaging DIRECT.", qd_max,
        kArmRestQdot);
    }
    const Mat3 r0_model = r0_actual * vehicle_->r_model;
    RestSpec r;
    r.x_b = x_b;
    r.phi = std::atan2(r0_model(1, 0), r0_model(0, 0));
    r.q = *q_meas_;
    return r;
  }

  // ------------------------------------------------------------- callbacks
  void onMode(const String & msg)
  {
    std::string s = msg.data;
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    const bool direct = s == "DIRECT";
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (direct == mode_direct_) {return;}
      mode_direct_ = direct;
      if (direct) {
        const auto rest = restFromMeasurements();
        if (!rest.has_value()) {
          hold_.reset();
          state_ = "HOLD";
          RCLCPP_WARN(
            get_logger(), "DIRECT engaged before odometry/arm samples -- hold "
            "captures on the first streamed tick.");
        } else {
          setHold(*rest);
          state_ = "HOLD";
          logHold("DIRECT engaged");
        }
      } else {
        state_ = "IDLE";
        hold_.reset();
        hold_ref_.reset();
        pending_base_.reset();
        ee_target_.reset();
        home_goal_ = false;
        plan_.reset();
        exec_t0_.reset();
        ++plan_gen_;
        goal_override_.reset();
        auto_send_ = false;
        ee_traj_.reset();
        ee_shape_type_.clear();
        ++ee_gen_;
        RCLCPP_INFO(get_logger(), "SAFETY -- planner silent, targets dropped.");
      }
    }
    publishStatus();
    publishBaseAnchor();
    if (!direct) {
      publishVizPath(nullptr);
      publishVizPose(nullptr);
      publishEeStatus("NONE");
      publishEePath(nullptr);
    }
  }

  void setHold(const RestSpec & rest, bool keep_anchor = false)
  {
    hold_ = rest;
    hold_ref_ = restReference(vehicle_->params, rest);
    if (!keep_anchor) {
      hold_anchor_ = std::make_pair(
        hold_ref_->r_ed, std::atan2(hold_ref_->b1_de(1), hold_ref_->b1_de(0)));
    }
    publishTargetJoints(rest.q);
  }

  void logHold(const char * why)
  {
    const WbReference & r = *hold_ref_;
    RCLCPP_INFO(
      get_logger(), "%s: hold captured -- base [%.3f %.3f %.3f] m, q %s deg, "
      "EE (inertial) [%.3f %.3f %.3f] m az %.1f deg", why, hold_->x_b(0),
      hold_->x_b(1), hold_->x_b(2), degList(hold_->q).c_str(), r.r_ed(0),
      r.r_ed(1), r.r_ed(2),
      std::atan2(r.b1_de(1), r.b1_de(0)) * 180.0 / M_PI);
  }

  // Drone-GS target. SAFETY: ignored (the flight node consumes it). DIRECT:
  // captured as the pending base target -- never executed directly.
  void onGsReference(const PositionControllerReference & msg)
  {
    bool replan = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_) {return;}
      double yaw = msg.yaw;
      if (msg.yaw_unit == PositionControllerReference::DEGREES) {yaw *= M_PI / 180.0;}
      const double phi = yaw - 0.5 * M_PI;  // actual yaw -> model heading
      const Vec3 x_b{msg.position.x, msg.position.y, msg.position.z};
      // Ignore an UNCHANGED target: ground stations re-send the same
      // setpoint, and each copy would restart the solve.
      if (pending_base_.has_value()) {
        const bool same = (x_b - pending_base_->first).norm() < 1e-3 &&
          std::abs(wrapPi(phi - pending_base_->second)) < 2e-3;
        if (same && (state_ == "PENDING" || state_ == "CALCULATING" ||
          state_ == "PLANNED" || state_ == "INFEASIBLE" || state_ == "EXECUTING"))
        {
          return;
        }
      }
      pending_base_ = std::make_pair(x_b, phi);
      if (state_ == "EXECUTING") {
        RCLCPP_INFO(get_logger(), "drone-GS target queued (executing) -- replans on completion.");
      } else {
        state_ = "PENDING";
        replan = true;
      }
    }
    publishBaseAnchor();
    publishStatus();
    if (replan) {startPlanning();}
  }

  // Inertial EE target from the arm GS tab 2: position + heading azimuth.
  void onEeTarget(const PoseStamped & msg)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_) {
        RCLCPP_WARN(get_logger(), "EE target ignored -- not in whole-body DIRECT mode.");
        return;
      }
      const Vec3 p{msg.pose.position.x, msg.pose.position.y, msg.pose.position.z};
      // ACTUAL yaw on the wire -> MODEL heading azimuth (the same -pi/2 the
      // drone-GS yaw gets).
      double az = yawFromQuat(
        msg.pose.orientation.x, msg.pose.orientation.y,
        msg.pose.orientation.z, msg.pose.orientation.w) - 0.5 * M_PI;
      az = wrapPi(az);
      if (ee_target_.has_value() && !home_goal_) {
        const double d_az = wrapPi(az - ee_target_->second);
        if ((p - ee_target_->first).norm() < 1e-3 && std::abs(d_az) < 2e-3 &&
          (state_ == "CALCULATING" || state_ == "PLANNED" ||
          state_ == "INFEASIBLE" || state_ == "EXECUTING"))
        {
          return;
        }
      }
      ee_target_ = std::make_pair(p, az);
      home_goal_ = false;
      if (state_ == "EXECUTING") {
        RCLCPP_INFO(get_logger(), "EE target queued (executing) -- replans on completion.");
        return;
      }
    }
    startPlanning();
  }

  // POSITION only: the orientation field is the mocap attitude, which the
  // law does not fly on.
  void onOdom(const Odometry & msg)
  {
    std::lock_guard<std::recursive_mutex> lk(lock_);
    odom_p_ = Vec3{msg.pose.pose.position.x, msg.pose.pose.position.y,
      msg.pose.pose.position.z};
    odom_p_time_ = Clock::now();
  }

  // ATTITUDE from the PX4 EKF2 topic the law regulates against; msg.q is
  // (w, x, y, z) in NED/FRD.
  void onAttitude(const VehicleAttitude & msg)
  {
    double w = msg.q[0], x = msg.q[1], y = msg.q[2], z = msg.q[3];
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (!std::isfinite(n) || n < 1e-9) {return;}
    w /= n; x /= n; y /= n; z /= n;
    Mat3 r_ned;
    r_ned << 1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y);
    std::lock_guard<std::recursive_mutex> lk(lock_);
    att_R_ = kRIE * r_ned * kRBV;
    att_time_ = Clock::now();
  }

  // Arm joints by name, ARM convention -> MODEL convention via arm_joint_sign.
  void onJointStates(const JointState & msg)
  {
    VecN q, qd;
    for (int j = 0; j < kNumJoints; ++j) {
      const auto it = std::find(msg.name.begin(), msg.name.end(), kArmJointNames[j]);
      if (it == msg.name.end()) {return;}
      const size_t k = static_cast<size_t>(it - msg.name.begin());
      if (k >= msg.position.size()) {return;}
      q(j) = msg.position[k];
      qd(j) = k < msg.velocity.size() ? msg.velocity[k] : 0.0;
    }
    if (!q.allFinite() || !qd.allFinite()) {return;}
    std::lock_guard<std::recursive_mutex> lk(lock_);
    q_meas_ = joint_sign_.cwiseProduct(q);
    qdot_meas_ = joint_sign_.cwiseProduct(qd);
  }

  // Grasp-point EE at the MEASURED joints, 15 Hz (a display feed).
  void currentEeTick()
  {
    std::optional<VecN> q;
    Vec3 x_b;
    Mat3 r0;
    bool have_odom = false;
    double age = 1e300;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      q = q_meas_;
      have_odom = odomPair(&x_b, &r0);
      age = odomAge();
    }
    if (!q.has_value()) {return;}
    Vec3 r0e;
    armKinematics(*q, vehicle_->params, nullptr, &r0e, nullptr);
    const Vec3 v = vehicle_->r_model * r0e;
    const auto stamp = now();
    PoseStamped body;
    body.header.stamp = stamp;
    body.header.frame_id = "drone_body";
    body.pose.position.x = v(0);
    body.pose.position.y = v(1);
    body.pose.position.z = v(2);
    body.pose.orientation.w = 1.0;
    cur_ee_body_pub_->publish(body);
    if (!have_odom || age > kOdomFreshS) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "current_ee (world) withheld -- odometry absent or stale; current_ee_body still streams.");
      return;
    }
    const Vec3 p = x_b + r0 * v;
    PoseStamped world;
    world.header.stamp = stamp;
    world.header.frame_id = "world";
    world.pose.position.x = p(0);
    world.pose.position.y = p(1);
    world.pose.position.z = p(2);
    world.pose.orientation.w = 1.0;
    cur_ee_pub_->publish(world);
  }

  // -------------------------------------------------------------- planning
  // Goal rest from pending base (else hold base) + EE target (else
  // ride-along). Called under the lock.
  std::optional<RestSpec> goalRest(std::string * err, VecN * q_goal)
  {
    RestSpec g;
    if (goal_override_.has_value()) {
      g = *goal_override_;
      *q_goal = g.q;
      return g;
    }
    if (pending_base_.has_value()) {
      g.x_b = pending_base_->first;
      g.phi = pending_base_->second;
    } else {
      g.x_b = hold_->x_b;
      g.phi = hold_->phi;
    }
    if (home_goal_) {
      g.q = home_pose_;
      *q_goal = g.q;
      return g;
    }
    if (!ee_target_.has_value()) {
      g.q = hold_->q;
      *q_goal = g.q;
      return g;
    }
    const IkResult ik = ikWorld(
      vehicle_->params, g.x_b, g.phi, ee_target_->first, ee_target_->second,
      hold_->q);
    *q_goal = ik.q;  // returned either way: the GS joint-limit row shows it
    if (!ik.ok) {
      *err = ik.reason;
      return std::nullopt;
    }
    g.q = ik.q;
    return g;
  }

  PlanOptions planOptions()
  {
    PlanOptions o;
    o.v_max = get_parameter("v_max").as_double();
    o.a_max = get_parameter("a_max").as_double();
    o.w_max = get_parameter("w_max").as_double();
    o.T_min = get_parameter("t_min").as_double();
    o.T_max = get_parameter("t_max").as_double();
    o.dw_max = get_parameter("dw_max").as_double();
    const double tjm = get_parameter("tau_joint_max").as_double();
    o.tau_joint_max = tjm > 0.0 ? tjm : -1.0;
    o.rotor_bounds = get_parameter("rotor_bounds").as_bool();
    o.n_check = get_parameter("n_check").as_int();
    return o;
  }

  void startPlanning()
  {
    PlanRequest req;
    PlanOptions opts;
    unsigned gen = 0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {return;}
      if (state_ == "EXECUTING") {return;}
      req.rest0 = *hold_;
      std::string err;
      VecN q_goal;
      const auto rest1 = goalRest(&err, &q_goal);
      plan_.reset();
      publishTargetJoints(q_goal);
      if (!rest1.has_value()) {
        state_ = "INFEASIBLE";
        infeasible_reason_ = err;
        RCLCPP_WARN(get_logger(), "goal infeasible: %s", err.c_str());
        publishStatus();
        return;
      }
      req.rest1 = *rest1;
      state_ = "CALCULATING";
      gen = ++plan_gen_;
      opts = planOptions();
    }
    publishStatus();

    // The solve runs on a worker thread so the reference stream keeps its
    // rate. In C++ a plan takes a few ms, but a stalled solver must still
    // never stall the stream.
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, req, opts, gen]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          traj = planner_->plan(*vehicle_, req, opts);
        } catch (const std::exception & e) {
          err = e.what();
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != plan_gen_ || state_ != "CALCULATING") {return;}  // superseded
          if (!traj) {
            state_ = "INFEASIBLE";
            infeasible_reason_ = err;
            RCLCPP_WARN(get_logger(), "planning failed: %s", err.c_str());
          } else {
            plan_ = traj;
            state_ = "PLANNED";
            if (auto_send_) {
              auto_send_ = false;
              goal_override_.reset();
              exec_t0_ = Clock::now();
              state_ = "EXECUTING";
              RCLCPP_INFO(
                get_logger(), "planned: %s -- executing now (go to start).",
                traj->diag().summary.c_str());
            } else {
              RCLCPP_INFO(get_logger(), "planned: %s -- awaiting Send.", traj->diag().summary.c_str());
            }
          }
          if (!traj) {auto_send_ = false; goal_override_.reset();}
        }
        publishStatus();
        publishVizPath(traj.get());
      });
  }

  void onSend(Trigger::Response::SharedPtr resp)
  {
    double T = 0.0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (state_ != "PLANNED" || !plan_) {
        resp->success = false;
        resp->message = "nothing to send (state " + state_ + "); assign a target and wait for PLANNED";
        return;
      }
      exec_t0_ = Clock::now();
      state_ = "EXECUTING";
      T = plan_->duration();
    }
    publishStatus();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "executing compatible transition, T = " << T << " s";
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
  }

  void onGoHome(Trigger::Response::SharedPtr resp)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {
        resp->success = false;
        resp->message = "not in whole-body DIRECT";
        return;
      }
      if (state_ == "EXECUTING") {
        resp->success = false;
        resp->message = "executing -- wait for the transition to finish";
        return;
      }
      home_goal_ = true;
      ee_target_.reset();
    }
    RCLCPP_INFO(
      get_logger(), "Go Home: planning a compatible transition to the folded home pose %s deg.",
      degList(home_pose_).c_str());
    startPlanning();
    resp->success = true;
    resp->message = "planning the transition home -- press Send Compatible Trajectory once it reads PLANNED";
  }

  void onClear(Trigger::Response::SharedPtr resp)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (state_ == "EXECUTING") {
        resp->success = false;
        resp->message = "executing -- cannot clear mid-transition (revert to SAFETY to abort)";
        return;
      }
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      plan_.reset();
      ++plan_gen_;
      if (mode_direct_) {state_ = "HOLD";}
    }
    publishStatus();
    publishBaseAnchor();
    publishVizPath(nullptr);
    resp->success = true;
    resp->message = "targets cleared, holding";
  }

  // ------------------------------------------------------------- streaming
  void streamTick()
  {
    WbReference ref;
    bool finished = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_) {return;}
      if (!hold_.has_value()) {
        const auto rest = restFromMeasurements();
        if (!rest.has_value()) {return;}
        setHold(*rest);
        logHold("late hold capture");
      }
      bool have_ref = false;
      if (state_ == "EXECUTING" && plan_) {
        const double t = std::chrono::duration<double>(Clock::now() - *exec_t0_).count();
        if (t >= plan_->duration()) {
          finished = true;
        } else {
          ref = plan_->eval(t);
          have_ref = true;
        }
      }
      if (finished) {
        setHold(plan_->goalRest());
        plan_.reset();
        pending_base_.reset();
        ee_target_.reset();
        home_goal_ = false;
        state_ = "HOLD";
        logHold("transition complete");
      }
      if (!have_ref) {
        if (hold_ee_world_ && state_ == "HOLD" && hold_anchor_.has_value()) {
          trackBaseMotion();
        }
        ref = *hold_ref_;
      }
    }
    if (finished) {
      publishStatus();
      publishBaseAnchor();
      publishVizPath(nullptr);
    }
    publishRef(ref);
    // The arm reference, from the SAME sample the law gets, at the same rate.
    publishArmSync(ref.q_d, ref.qdot_d);
    viz_decim_ = (viz_decim_ + 1) % viz_decim_period_;
    if (viz_decim_ == 0) {
      publishVizPose(&ref);
      publishEeRefPose(ref);
    }
    if (finished) {refreshEeAfterHold();}
  }

  // World-EE-anchored HOLD: re-solve the arm against the CURRENT base pose.
  // Called from the stream tick, under the lock, HOLD state only.
  void trackBaseMotion()
  {
    Vec3 x_b;
    Mat3 r0_actual;
    if (!odomPair(&x_b, &r0_actual) || odomAge() > kOdomFreshS) {return;}
    if (secondsSince(last_base_resolve_) < 0.05) {return;}  // cap the IK at 20 Hz
    const Mat3 r0_model = r0_actual * vehicle_->r_model;
    const double phi = std::atan2(r0_model(1, 0), r0_model(0, 0));
    const double dp = (x_b - hold_->x_b).norm();
    const double dphi = std::abs(wrapPi(phi - hold_->phi));
    if (dp < 0.002 && dphi < 0.005) {return;}
    last_base_resolve_ = Clock::now();
    const IkResult ik = ikWorld(
      vehicle_->params, x_b, phi, hold_anchor_->first, hold_anchor_->second, hold_->q);
    if (!ik.ok) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "EE world-hold: base moved %.1f cm but the anchored EE is unreachable from "
        "there (%s) -- holding the last feasible reference.", dp * 100.0, ik.reason.c_str());
      return;
    }
    RestSpec r;
    r.x_b = x_b;
    r.phi = phi;
    r.q = ik.q;
    setHold(r, true);
  }

  static void setV(geometry_msgs::msg::Vector3 & v, const Vec3 & x)
  {
    v.x = x(0);
    v.y = x(1);
    v.z = x(2);
  }

  void publishRef(const WbReference & r)
  {
    WholeBodyReference m;
    m.header.stamp = now();
    m.header.frame_id = "model";
    setV(m.x_cd, r.x_cd);
    setV(m.x_cd_dot, r.x_cd_dot);
    setV(m.x_cd_ddot, r.x_cd_ddot);
    setV(m.x_cd_d3, r.x_cd_d3);
    setV(m.x_cd_d4, r.x_cd_d4);
    setV(m.b1_d, r.b1_d);
    setV(m.b1_d_dot, r.b1_d_dot);
    setV(m.b1_d_ddot, r.b1_d_ddot);
    setV(m.r_ed, r.r_ed);
    setV(m.r_ed_dot, r.r_ed_dot);
    setV(m.r_ed_ddot, r.r_ed_ddot);
    setV(m.b1_de, r.b1_de);
    setV(m.b1_de_dot, r.b1_de_dot);
    setV(m.b1_de_ddot, r.b1_de_ddot);
    for (int j = 0; j < kNumJoints; ++j) {
      m.q_d[j] = r.q_d(j);
      m.qdot_d[j] = r.qdot_d(j);
    }
    ref_pub_->publish(m);
  }

  // ------------------------------------------------------ usable workspace
  // Rasterise the reachable (r, z) set of the WHOLE-BODY chain, keeping only
  // poses the planner can use (fold beta >= beta_min_deg), as a filled grid:
  // [r_min, r_max, z_min, z_max, nr, nz, cells(nr*nz row-major, rows = z
  // from z_max down)]. (r, z) depends on (q2, q3) alone.
  void publishWorkspace()
  {
    const WholeBodyParams & P = vehicle_->params;
    const double beta_min = vehicle_->beta_min_deg * M_PI / 180.0;
    std::vector<double> rs, zs;
    const int n2 = 241, n3 = 181;
    for (int i = 0; i < n2; ++i) {
      const double q2 = vehicle_->q_min(1) + (vehicle_->q_max(1) - vehicle_->q_min(1)) * i / (n2 - 1);
      for (int k = 0; k < n3; ++k) {
        const double q3 = vehicle_->q_min(2) + (vehicle_->q_max(2) - vehicle_->q_min(2)) * k / (n3 - 1);
        if (q2 + q3 < beta_min) {continue;}
        VecN q;
        q << 0.0, q2, q3, 0.0;
        Vec3 r0e;
        armKinematics(q, P, nullptr, &r0e, nullptr);
        const Vec3 v = vehicle_->r_model * r0e;
        rs.push_back(std::hypot(v(0), v(1)));
        zs.push_back(v(2));
      }
    }
    if (rs.empty()) {
      RCLCPP_ERROR(get_logger(), "usable workspace is EMPTY -- check the joint limits.");
      return;
    }
    const double pad = 0.012;
    const double r_min = 0.0;
    const double r_max = *std::max_element(rs.begin(), rs.end()) + pad;
    const double z_min = *std::min_element(zs.begin(), zs.end()) - pad;
    const double z_max = std::max(0.0, *std::max_element(zs.begin(), zs.end())) + pad;
    const int nr = 192, nz = 192;
    std::vector<uint8_t> grid(static_cast<size_t>(nr) * nz, 0);
    for (size_t i = 0; i < rs.size(); ++i) {
      const int ci = static_cast<int>((rs[i] - r_min) / (r_max - r_min) * (nr - 1) + 0.5);
      const int ri = static_cast<int>((z_max - zs[i]) / (z_max - z_min) * (nz - 1) + 0.5);
      if (ci < 0 || ci >= nr || ri < 0 || ri >= nz) {continue;}
      // 3x3 stamp seals the sampling holes so the region draws as an area
      for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
          const int rr = std::min(nz - 1, std::max(0, ri + dr));
          const int cc = std::min(nr - 1, std::max(0, ci + dc));
          grid[static_cast<size_t>(rr) * nr + cc] = 1;
        }
      }
    }
    Float64MultiArray m;
    m.data = {r_min, r_max, z_min, z_max, static_cast<double>(nr), static_cast<double>(nz)};
    m.data.reserve(6 + grid.size());
    double filled = 0.0;
    for (uint8_t c : grid) {
      m.data.push_back(c);
      filled += c;
    }
    ws_pub_->publish(m);
    RCLCPP_INFO(
      get_logger(), "usable workspace published: %dx%d grid, r [%.3f, %.3f] m, z "
      "[%.3f, %.3f] m, %.0f%% filled (fold beta >= %.0f deg)", nr, nz,
      *std::min_element(rs.begin(), rs.end()), *std::max_element(rs.begin(), rs.end()),
      *std::min_element(zs.begin(), zs.end()), *std::max_element(zs.begin(), zs.end()),
      100.0 * filled / grid.size(), vehicle_->beta_min_deg);
  }

  void publishTargetJoints(const VecN & q)
  {
    Float64MultiArray m;
    for (int j = 0; j < kNumJoints; ++j) {m.data.push_back(q(j));}
    joints_pub_->publish(m);
  }

  // Stream the arm reference the torque controller tracks (MODEL -> ARM
  // convention through the sign map, its own inverse).
  void publishArmSync(const VecN & q, const VecN & qdot)
  {
    JointTrajectory m;
    m.header.stamp = now();
    for (int j = 0; j < kNumJoints; ++j) {m.joint_names.push_back(kArmJointNames[j]);}
    JointTrajectoryPoint pt;
    for (int j = 0; j < kNumJoints; ++j) {
      pt.positions.push_back(joint_sign_(j) * q(j));
      pt.velocities.push_back(joint_sign_(j) * qdot(j));
      pt.accelerations.push_back(0.0);
    }
    m.points.push_back(pt);
    arm_ref_pub_->publish(m);
  }

  // ---------------------------------------------- trajectory visualisation
  static constexpr int kVizPathSamples = 120;

  // One visualisation sample in directions a human can read: x_cd, the
  // vehicle's NOSE (actual body +x = Rz(+90) b1_d), r_ed, and the CLAW axis
  // (-R_e e3, where the grasp point lies).
  void vizRow(const WbReference & ref, std::vector<double> * out) const
  {
    const Vec3 & b1 = ref.b1_d;
    Vec3 nose{-b1(1), b1(0), 0.0};
    const double n = nose.norm();
    nose = n > 1e-9 ? Vec3(nose / n) : Vec3{1.0, 0.0, 0.0};
    Vec3 claw = ref.b1_de;
    const Vec3 ac = ref.x_cd_ddot + vehicle_->params.g * Vec3{0.0, 0.0, 1.0};
    if (ac.norm() > 1e-9) {
      const Mat3 r0 = buildR0(ac / ac.norm(), b1);
      Mat3 re;
      armKinematics(ref.q_d, vehicle_->params, nullptr, nullptr, &re);
      claw = -(r0 * re).col(2);
    }
    for (int i = 0; i < 3; ++i) {out->push_back(ref.x_cd(i));}
    for (int i = 0; i < 3; ++i) {out->push_back(nose(i));}
    for (int i = 0; i < 3; ++i) {out->push_back(ref.r_ed(i));}
    for (int i = 0; i < 3; ++i) {out->push_back(claw(i));}
  }

  // Sample the planned transition for the Isaac drawing; nullptr publishes
  // an empty array, which is how the drawing is cleared.
  void publishVizPath(const Trajectory * traj)
  {
    Float64MultiArray m;
    if (traj != nullptr) {
      for (int k = 0; k < kVizPathSamples; ++k) {
        vizRow(traj->eval(traj->duration() * k / (kVizPathSamples - 1)), &m.data);
      }
    }
    viz_path_pub_->publish(m);
  }

  void publishVizPose(const WbReference * ref)
  {
    Float64MultiArray m;
    if (ref != nullptr) {vizRow(*ref, &m.data);}
    viz_pose_pub_->publish(m);
  }


  // ================================================== EE trajectory mode
  EeShape eeShape() const
  {
    EeShape sh;
    sh.type = ee_shape_type_;
    sh.radius = get_parameter("ee_traj_circle_radius").as_double();
    sh.fig8_a = get_parameter("ee_traj_fig8_a").as_double();
    sh.fig8_b = get_parameter("ee_traj_fig8_b").as_double();
    sh.lap_time = get_parameter("ee_traj_lap_time").as_double();
    sh.laps = static_cast<int>(get_parameter("ee_traj_laps").as_int());
    sh.ccw = get_parameter("ee_traj_ccw").as_bool();
    return sh;
  }

  EeTrajectoryOptions eeOptions() const
  {
    EeTrajectoryOptions o;
    o.ramp_time = get_parameter("ee_traj_ramp_time").as_double();
    o.ee_fold_deg = get_parameter("ee_traj_fold_deg").as_double();
    o.q2_center_deg = get_parameter("ee_traj_q2_center_deg").as_double();
    o.q2_amp_deg = get_parameter("ee_traj_q2_amp_deg").as_double();
    o.q2_period_s = get_parameter("ee_traj_q2_period_s").as_double();
    o.qdot_max = get_parameter("ee_traj_qdot_max").as_double();
    o.v_max = get_parameter("v_max").as_double();
    o.a_max = get_parameter("a_max").as_double();
    o.w_max = get_parameter("w_max").as_double();
    const double tjm = get_parameter("tau_joint_max").as_double();
    o.tau_joint_max = tjm > 0.0 ? tjm : -1.0;
    o.rotor_bounds = get_parameter("rotor_bounds").as_bool();
    o.sigma_nd_min = vehicle_->sigma_nd_margin;
    o.beta_min_deg = vehicle_->beta_min_deg;
    return o;
  }

  void onEeSelect(const String & msg)
  {
    std::string type = msg.data;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (type == "none" || type.empty()) {
        ee_shape_type_.clear();
        ee_traj_.reset();
        ee_s_max_.reset();
        ++ee_gen_;
      } else {
        if (type != "circle" && type != "figure8") {
          publishEeStatus("INFEASIBLE: unknown trajectory '" + type + "' (circle | figure8)");
          return;
        }
        const bool changed = type != ee_shape_type_;
        ee_shape_type_ = type;
        if (changed) {ee_s_max_.reset();}
      }
    }
    if (type == "none" || type.empty()) {
      publishEeStatus("NONE");
      publishEePath(nullptr);
      return;
    }
    startEePlanning();
  }

  void onEeTimeScale(const Float64 & msg)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      ee_time_scale_req_ = msg.data;
      if (ee_shape_type_.empty()) {return;}
    }
    startEePlanning();
  }

  // Re-anchor the selected trajectory on a NEW hold (a transition other than
  // go-to-start finished): the run must start where the vehicle now is.
  void refreshEeAfterHold()
  {
    bool replan = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (ee_shape_type_.empty() || !hold_.has_value()) {return;}
      if (ee_traj_) {
        const RestSpec r0 = ee_traj_->goalRest();
        const bool same = (r0.x_b - hold_->x_b).norm() < 1e-3 &&
          std::abs(wrapPi(r0.phi - hold_->phi)) < 2e-3 &&
          (r0.q - hold_->q).cwiseAbs().maxCoeff() < 2e-3;
        if (same) {return;}   // arrived at the run's own start rest
      }
      replan = true;
    }
    if (replan) {startEePlanning();}
  }

  void startEePlanning()
  {
    RestSpec hold;
    EeShape shape;
    EeTrajectoryOptions opts;
    bool need_smax = false;
    double s_req = 1.0;
    unsigned gen = 0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (ee_shape_type_.empty()) {return;}
      if (!mode_direct_ || !hold_.has_value()) {
        publishEeStatus("NOT IN DIRECT");
        return;
      }
      hold = *hold_;
      shape = eeShape();
      opts = eeOptions();
      need_smax = !ee_s_max_.has_value();
      s_req = ee_time_scale_req_;
      gen = ++ee_gen_;
    }
    publishEeStatus("CALCULATING");
    if (ee_worker_.joinable()) {ee_worker_.join();}
    ee_worker_ = std::thread([this, hold, shape, opts, need_smax, s_req, gen]() {
        double s_max = 0.0;
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (ee_s_max_.has_value()) {s_max = *ee_s_max_;}
        }
        if (need_smax) {
          s_max = EeTrajectoryPlanner::maxTimeScale(*vehicle_, hold, shape, opts);
        }
        std::shared_ptr<Trajectory> traj;
        EeTrajectoryDiag diag;
        std::string err;
        EeTrajectoryOptions o = opts;
        o.time_scale = std::max(0.05, std::min(s_req, s_max > 0.0 ? s_max : s_req));
        if (s_max <= 0.0) {
          err = "no feasible time scale for this trajectory from the current hold";
        } else {
          try {
            traj = EeTrajectoryPlanner::plan(*vehicle_, hold, shape, o, &diag);
          } catch (const std::exception & e) {
            err = e.what();
          }
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != ee_gen_) {return;}          // superseded
          ee_s_max_ = s_max;
          ee_traj_ = traj;
          ee_diag_ = diag;
          ee_anchor_hold_ = hold;
        }
        if (!traj) {
          RCLCPP_WARN(get_logger(), "EE trajectory infeasible: %s", err.c_str());
          publishEeStatus("INFEASIBLE: " + err);
          publishEePath(nullptr);
          publishEeInfo(s_max, o.time_scale, diag);
          return;
        }
        RCLCPP_INFO(
          get_logger(), "EE trajectory READY (s_max %.2f): %s", s_max, diag.summary.c_str());
        std::ostringstream m;
        m.setf(std::ios::fixed);
        m.precision(2);
        m << "READY s=" << o.time_scale << " max=" << s_max << " T=" << diag.T_total
          << "s EE err " << std::setprecision(1) << diag.ee_pos_err_max * 1e3 << "mm/"
          << diag.ee_rot_err_max_deg << "deg";
        publishEeStatus(m.str());
        publishEeInfo(s_max, o.time_scale, diag);
        publishEePath(traj.get());
      });
  }

  void onEeGoToStart(Trigger::Response::SharedPtr resp)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {
        resp->success = false;
        resp->message = "not in whole-body DIRECT";
        return;
      }
      if (!ee_traj_) {
        resp->success = false;
        resp->message = "no EE trajectory is READY -- select one first";
        return;
      }
      if (state_ == "EXECUTING") {
        resp->success = false;
        resp->message = "executing -- wait for the transition to finish";
        return;
      }
      goal_override_ = ee_traj_->goalRest();
      auto_send_ = true;
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
    }
    RCLCPP_INFO(get_logger(), "EE trajectory: planning the compatible transition to its start rest.");
    startPlanning();
    resp->success = true;
    resp->message = "planning the compatible transition to the trajectory's start; it executes as soon as it is PLANNED";
  }

  // measured (x_b, phi, q) vs the run's start rest
  bool startErrors(double * pos, double * yaw, double * joint)
  {
    Vec3 x_b;
    Mat3 r0;
    if (!ee_traj_ || !odomPair(&x_b, &r0) || !q_meas_.has_value()) {return false;}
    const RestSpec r = ee_traj_->goalRest();
    const Mat3 r0_model = r0 * vehicle_->r_model;
    const double phi = std::atan2(r0_model(1, 0), r0_model(0, 0));
    *pos = (x_b - r.x_b).norm();
    *yaw = std::abs(wrapPi(phi - r.phi));
    *joint = (*q_meas_ - r.q).cwiseAbs().maxCoeff();
    return true;
  }

  bool atStart(double * pos, double * yaw, double * joint)
  {
    if (!startErrors(pos, yaw, joint)) {return false;}
    return *pos <= get_parameter("ee_traj_start_pos_tol").as_double() &&
           *yaw <= get_parameter("ee_traj_start_yaw_tol_deg").as_double() * M_PI / 180.0 &&
           *joint <= get_parameter("ee_traj_start_joint_tol_deg").as_double() * M_PI / 180.0;
  }

  void onEeStart(Trigger::Response::SharedPtr resp)
  {
    double T = 0.0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {
        resp->success = false;
        resp->message = "not in whole-body DIRECT";
        return;
      }
      if (!ee_traj_) {
        resp->success = false;
        resp->message = "no EE trajectory is READY";
        return;
      }
      if (state_ != "HOLD" && state_ != "PLANNED" && state_ != "PENDING" && state_ != "INFEASIBLE") {
        resp->success = false;
        resp->message = "planner is " + state_ + " -- wait for HOLD";
        return;
      }
      double pos = 0.0, yaw = 0.0, joint = 0.0;
      if (!atStart(&pos, &yaw, &joint)) {
        std::ostringstream m;
        m.setf(std::ios::fixed);
        m.precision(1);
        m << "not at the trajectory's start: position " << pos * 100.0 << " cm, yaw "
          << yaw * 180.0 / M_PI << " deg, joints " << joint * 180.0 / M_PI
          << " deg off -- press Go to start first";
        resp->success = false;
        resp->message = m.str();
        return;
      }
      plan_ = ee_traj_;
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      goal_override_.reset();
      auto_send_ = false;
      ++plan_gen_;
      exec_t0_ = Clock::now();
      state_ = "EXECUTING";
      T = plan_->duration();
    }
    publishStatus();
    publishVizPath(plan_.get());
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << "executing the EE trajectory, T = " << T << " s";
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "%s", resp->message.c_str());
  }

  void publishEeStartError()
  {
    Float64MultiArray m;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !ee_traj_) {return;}
      double pos = 0.0, yaw = 0.0, joint = 0.0;
      const bool ok = atStart(&pos, &yaw, &joint);
      const bool ready = ok && state_ == "HOLD";
      m.data = {pos, yaw * 180.0 / M_PI, joint * 180.0 / M_PI, ready ? 1.0 : 0.0};
    }
    ee_start_err_pub_->publish(m);
  }

  void publishEeStatus(const std::string & s)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      ee_status_ = s;
    }
    String m;
    m.data = s;
    ee_status_pub_->publish(m);
  }

  void publishEeInfo(double s_max, double s, const EeTrajectoryDiag & d)
  {
    Float64MultiArray m;
    const double type_id = ee_shape_type_ == "circle" ? 1.0 : (ee_shape_type_ == "figure8" ? 2.0 : 0.0);
    m.data = {s, s_max, d.T_total, d.T_lap, static_cast<double>(d.laps), d.ramp_time, type_id,
      d.ee_pos_err_max, d.ee_rot_err_max_deg, d.peak_v, d.peak_a, d.peak_qdot, d.peak_tau_joint,
      d.min_sigma_nd};
    ee_info_pub_->publish(m);
  }

  // The EE frame published for the operator: x = the claw axis (-R_e e3),
  // z = the gripper's up (R_e e2), y = z x x -- a right-handed triad whose x
  // points where the gripper points, in the shared world frame.
  static void eeFrameQuat(const Mat3 & R0, const Mat3 & Re, geometry_msgs::msg::Quaternion * q)
  {
    Mat3 F;
    F.col(0) = -Re.col(2);
    F.col(1) = -Re.col(0);
    F.col(2) = Re.col(1);
    const Eigen::Quaterniond qq(R0 * F);
    q->x = qq.x();
    q->y = qq.y();
    q->z = qq.z();
    q->w = qq.w();
  }

  void eePoseOf(const WbReference & ref, PoseStamped * m) const
  {
    const Vec3 ac = ref.x_cd_ddot + vehicle_->params.g * Vec3{0.0, 0.0, 1.0};
    Mat3 R0 = Mat3::Identity();
    if (ac.norm() > 1e-9) {R0 = buildR0(ac / ac.norm(), ref.b1_d);}
    Mat3 Re;
    armKinematics(ref.q_d, vehicle_->params, nullptr, nullptr, &Re);
    m->pose.position.x = ref.r_ed(0);
    m->pose.position.y = ref.r_ed(1);
    m->pose.position.z = ref.r_ed(2);
    eeFrameQuat(R0, Re, &m->pose.orientation);
  }

  void publishEeRefPose(const WbReference & ref)
  {
    PoseStamped m;
    m.header.stamp = now();
    m.header.frame_id = "world";
    eePoseOf(ref, &m);
    ee_ref_pose_pub_->publish(m);
  }

  // [t, x, y, z, speed, qx, qy, qz, qw] per sample, world frame; empty = clear
  void publishEePath(const Trajectory * traj)
  {
    Float64MultiArray m;
    if (traj != nullptr) {
      const int n = 400;
      for (int k = 0; k < n; ++k) {
        const double t = traj->duration() * k / (n - 1);
        const WbReference ref = traj->eval(t);
        PoseStamped ps;
        eePoseOf(ref, &ps);
        m.data.push_back(t);
        m.data.push_back(ps.pose.position.x);
        m.data.push_back(ps.pose.position.y);
        m.data.push_back(ps.pose.position.z);
        m.data.push_back(ref.r_ed_dot.norm());
        m.data.push_back(ps.pose.orientation.x);
        m.data.push_back(ps.pose.orientation.y);
        m.data.push_back(ps.pose.orientation.z);
        m.data.push_back(ps.pose.orientation.w);
      }
    }
    ee_path_pub_->publish(m);
  }

  // ---------------------------------------------------------------- status
  void publishStatus()
  {
    std::string s;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      s = state_;
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(1);
      if (state_ == "PLANNED" && plan_) {
        m << "PLANNED T=" << plan_->duration() << "s";
        s = m.str();
      } else if (state_ == "INFEASIBLE") {
        s = "INFEASIBLE: " + infeasible_reason_;
      } else if (state_ == "EXECUTING" && plan_) {
        m << "EXECUTING T=" << plan_->duration() << "s";
        s = m.str();
      }
    }
    String msg;
    msg.data = s;
    status_pub_->publish(msg);
  }

  // The base point the arm GS tab 2 interprets EE targets against: the
  // pending drone-GS target when one exists, else the current hold.
  // Published in the ACTUAL convention (yaw = heading of the mechanical front).
  void publishBaseAnchor()
  {
    Vec3 x_b;
    double phi = 0.0;
    std::string tag;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pending_base_.has_value()) {
        x_b = pending_base_->first;
        phi = pending_base_->second;
        tag = "pending";
      } else if (hold_.has_value()) {
        x_b = hold_->x_b;
        phi = hold_->phi;
        tag = "hold";
      } else {
        return;
      }
    }
    PoseStamped m;
    m.header.stamp = now();
    m.header.frame_id = tag;
    m.pose.position.x = x_b(0);
    m.pose.position.y = x_b(1);
    m.pose.position.z = x_b(2);
    const double yaw = phi + 0.5 * M_PI;  // model heading -> actual yaw
    m.pose.orientation.z = std::sin(0.5 * yaw);
    m.pose.orientation.w = std::cos(0.5 * yaw);
    base_pub_->publish(m);
  }

  // ---- configuration -------------------------------------------------------
  std::string vehicle_name_, planner_name_;
  double stream_rate_{100.0};
  VecN home_pose_{VecN::Zero()};
  VecN joint_sign_{VecN::Ones()};
  bool hold_ee_world_{false};
  std::shared_ptr<const VehicleModel> vehicle_;
  std::unique_ptr<TrajectoryPlanner> planner_;

  // ---- state (under lock_) -------------------------------------------------
  // RLock: status/anchor publishers are called both inside and outside
  // locked sections (the infeasible branch publishes under the lock).
  std::recursive_mutex lock_;
  bool mode_direct_{false};
  std::string state_{"IDLE"};  // IDLE HOLD PENDING CALCULATING PLANNED INFEASIBLE EXECUTING
  std::optional<RestSpec> hold_;
  std::optional<WbReference> hold_ref_;
  std::optional<std::pair<Vec3, double>> hold_anchor_;  // (p_e_world, model az)
  std::optional<std::pair<Vec3, double>> pending_base_;  // (x_b, phi)
  std::optional<std::pair<Vec3, double>> ee_target_;     // (p_e_world, az)
  bool home_goal_{false};
  std::shared_ptr<Trajectory> plan_;
  std::optional<Clock::time_point> exec_t0_;
  unsigned plan_gen_{0};
  std::string infeasible_reason_;
  std::optional<Clock::time_point> last_base_resolve_;
  std::thread worker_;
  // go-to-start: the transition goal replaces the GS targets, and PLANNED
  // executes without a Send
  std::optional<RestSpec> goal_override_;
  bool auto_send_{false};
  // EE trajectory mode
  std::string ee_shape_type_;
  double ee_time_scale_req_{1.0};
  std::optional<double> ee_s_max_;
  std::shared_ptr<Trajectory> ee_traj_;
  EeTrajectoryDiag ee_diag_;
  RestSpec ee_anchor_hold_;
  std::string ee_status_{"NONE"};
  unsigned ee_gen_{0};
  std::thread ee_worker_;

  // ---- live samples --------------------------------------------------------
  std::optional<Vec3> odom_p_;
  std::optional<Clock::time_point> odom_p_time_;
  std::optional<Mat3> att_R_;
  std::optional<Clock::time_point> att_time_;
  bool att_warned_{false};
  std::optional<VecN> q_meas_;
  VecN qdot_meas_{VecN::Zero()};
  int viz_decim_{0};
  int viz_decim_period_{5};

  // ---- ROS ------------------------------------------------------------------
  rclcpp::Publisher<WholeBodyReference>::SharedPtr ref_pub_;
  rclcpp::Publisher<String>::SharedPtr status_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr base_pub_, cur_ee_pub_, cur_ee_body_pub_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr joints_pub_, ws_pub_, viz_path_pub_, viz_pose_pub_;
  rclcpp::Publisher<JointTrajectory>::SharedPtr arm_ref_pub_;
  rclcpp::Subscription<String>::SharedPtr mode_sub_;
  rclcpp::Subscription<PositionControllerReference>::SharedPtr gs_sub_;
  rclcpp::Subscription<VehicleAttitude>::SharedPtr att_sub_;
  rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<JointState>::SharedPtr js_sub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr ee_sub_;
  rclcpp::Service<Trigger>::SharedPtr send_srv_, clear_srv_, home_srv_;
  rclcpp::TimerBase::SharedPtr stream_timer_, ee_timer_;
  rclcpp::Publisher<String>::SharedPtr ee_status_pub_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr ee_info_pub_, ee_path_pub_, ee_start_err_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr ee_ref_pose_pub_;
  rclcpp::Subscription<String>::SharedPtr ee_select_sub_;
  rclcpp::Subscription<Float64>::SharedPtr ee_scale_sub_;
  rclcpp::Service<Trigger>::SharedPtr ee_go_srv_, ee_start_srv_;
  rclcpp::TimerBase::SharedPtr ee_err_timer_;
};

}  // namespace fsc_trajectory_planner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<fsc_trajectory_planner::WholeBodyTrajectoryPlanner>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("whole_body_trajectory_planner"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
