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
//     PICK-AND-PLACE MODE (2026-10-01): six legs (go_to_start, execute_pick,
//                 go_to_place_start, execute_place, go_to_land_start,
//                 execute_land) between calibrated base poses and claw
//                 targets measured from mocap (obj_0 / drop_0). `plan` dry-
//                 runs the whole mission; each leg's service re-plans it from
//                 the CURRENT hold and executes it at once (pick_place.hpp).
//   Mode leaves DIRECT at ANY point -> streaming stops instantly.
//
// FRAMES: the ROS boundary is the ACTUAL world/FLU convention (odometry, GS
// yaw, EE targets); the planner and the streamed message are MODEL frame.
//     R0_model = R0_actual @ R_MODEL          phi_model = psi_actual - pi/2
//
// Every topic and service is RELATIVE, so `--ros-args -r __ns:=/uav_N` (the
// launch file's uav_prefix) namespaces the whole interface per vehicle.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <iomanip>
#include <limits>
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

#include <fsc_autopilot_ros2_msgs/msg/mocap.hpp>
#include <fsc_autopilot_ros2_msgs/msg/position_controller_reference.hpp>
#include <fsc_autopilot_ros2_msgs/msg/whole_body_reference.hpp>

#include "fsc_trajectory_planner/ee_trajectory_planner.hpp"
#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/pick_place.hpp"
#include "fsc_trajectory_planner/trajectory.hpp"
#include "fsc_trajectory_planner/vehicle_model.hpp"
#include "fsc_trajectory_planner/workspace.hpp"

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
using fsc_autopilot_ros2_msgs::msg::Mocap;
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

// Pick-and-place: the point each leg flies to, indexed like the legs
// (PickPlaceLeg). Names the pick_place_<point>_topic parameters and the
// capture_<point> services.
const char * const kPpPoint[kNumPickPlaceLegs] = {
  "start", "pick", "place_start", "place", "land_start", "land"};

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
    // "bspline" (flat outputs, constraints enforced) -- the only backend since
    // 2026-09-17, when the straight_line Picard backend was removed. Kept as a
    // parameter so the registry stays the extension point for a new one.
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
    // circle centred on the world origin at the current EE height (true) or
    // through the current EE point (false)
    declare_parameter<bool>("ee_traj_center_origin", true);
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

    // --- pick-and-place mode (pick_place.hpp) ------------------------------
    // Base poses [x, y, z, yaw_deg], WORLD frame, ACTUAL yaw (the drone GS's
    // convention). PLACEHOLDERS: every one is shifted by the Adjust offset,
    // and any one can be measured instead from a mocap body through its
    // pick_place_<point>_topic + capture_<point> (x, y measured; z and yaw
    // stay these).
    declare_parameter<std::vector<double>>("pick_place_start", std::vector<double>{0.0, 0.0, 1.0, 0.0});
    declare_parameter<std::vector<double>>(
      "pick_place_place_start", std::vector<double>{0.0, -1.0, 1.0, 0.0});
    declare_parameter<std::vector<double>>(
      "pick_place_land_start", std::vector<double>{-1.0, 0.0, 1.0, 0.0});
    // the hover over the landing spot: >= 0.5 m above the 0.305 m standing
    // height (a 0.4 m hover brushed the floor in Isaac, 2026-10-01)
    declare_parameter<std::vector<double>>("pick_place_land", std::vector<double>{-1.0, 0.0, 0.8, 0.0});
    // Arm poses [deg]. Pick / place: [0, 0, 0, 0] = upper arm straight down,
    // forearm out along the nose, claw straight down (the pitch joints at a
    // right angle each); carry: the folded home.
    declare_parameter<std::vector<double>>("pick_place_pick_pose_deg", std::vector<double>{0.0, 0.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>("pick_place_place_pose_deg", std::vector<double>{0.0, 0.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>(
      "pick_place_carry_pose_deg", std::vector<double>{0.0, 40.0, 40.0, 0.0});
    // claw target = measured mocap point + offset, world [m]
    declare_parameter<std::vector<double>>("pick_place_pick_ee_offset", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>("pick_place_place_ee_offset", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter<bool>("pick_place_face_target", true);
    // before go_to_place_start / go_to_land_start: climb, then back off [m]
    declare_parameter<double>("pick_place_retreat_dz", 0.15);
    declare_parameter<double>("pick_place_retreat_back", 0.20);
    // execute_place: the arm sweeps these bands [deg] (lo == hi: not swept)
    declare_parameter<std::vector<double>>(
      "pick_place_sweep_lo_deg", std::vector<double>{-25.0, 10.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>(
      "pick_place_sweep_hi_deg", std::vector<double>{25.0, 45.0, 0.0, 0.0});
    declare_parameter<std::vector<double>>(
      "pick_place_sweep_phase_deg", std::vector<double>{0.0, 90.0, 0.0, 0.0});
    declare_parameter<int>("pick_place_sweep_cycles", 2);
    declare_parameter<double>("pick_place_sweep_ramp_frac", 0.2);
    declare_parameter<double>("pick_place_sweep_qdot_max", 0.5);
    declare_parameter<double>("pick_place_sweep_t_min", 8.0);
    declare_parameter<double>("pick_place_sweep_t_max", 120.0);
    // claw (pick / place) or base (other legs) within this of its target:
    // the fine-correction gate reported on pick_place/arrival_error [m]
    declare_parameter<double>("pick_place_arrival_tol", 0.05);
    // Adjust shifts x, y; z and yaw too only when these are set
    declare_parameter<bool>("pick_place_adjust_z", false);
    declare_parameter<bool>("pick_place_adjust_yaw", false);
    // mocap bodies (fsc_autopilot_ros2_msgs/Mocap, ABSOLUTE topics); "" = none
    for (int k = 0; k < kNumPickPlaceLegs; ++k) {
      const char * dflt = k == kExecutePick ? "/obj_0/mocap" : (k == kExecutePlace ? "/drop_0/mocap" : "");
      pp_topic_[k] = declare_parameter<std::string>(
        std::string("pick_place_") + kPpPoint[k] + "_topic", dflt);
    }
    // a capture averages the samples of the last this-many seconds, and is
    // REFUSED when they scatter more than this: two publishers on one topic
    // (the Isaac emulator's phantom obj_0 at the origin did exactly that on
    // the first sim flight, 2026-10-01: 856 mm, averaged and flown) or a
    // moving / mis-tracked body must never become a goal [m]
    declare_parameter<double>("pick_place_capture_window", 0.5);
    declare_parameter<double>("pick_place_capture_max_spread", 0.02);

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
    pp_ws_pub_ = create_publisher<Float64MultiArray>(prefix + "/pick_place/workspace_rz", latched);
    // Isaac visualisation: planned path (latched) and current sample (~20 Hz),
    // 12 world-frame doubles per sample: x_cd, nose, r_ed, claw.
    viz_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/viz_path", latched);
    viz_pose_pub_ = create_publisher<Float64MultiArray>(prefix + "/viz_pose", 10);
    // The arm's CURRENT EE (grasp point) on the MEASURED joints, from THIS
    // node's model: inertial (gated on fresh odometry) and drone-body.
    cur_ee_pub_ = create_publisher<PoseStamped>(prefix + "/current_ee", 10);
    cur_ee_body_pub_ = create_publisher<PoseStamped>(prefix + "/current_ee_body", 10);
    // ... and its HEADING in the ee_target convention (ACTUAL yaw, rad): the
    // yaw an ee_target must carry to leave the wrist where it is. current_ee
    // carries no orientation, and a station seeding a target without this
    // asks for a wrist swing nobody commanded (the PS4 Remote tab did).
    cur_ee_heading_pub_ = create_publisher<Float64>(prefix + "/current_ee_heading", 10);
    // THE ARM REFERENCE IN DIRECT: this node is the arm's only reference
    // source while it streams (the arm planner owns it in SAFETY).
    arm_ref_pub_ = create_publisher<JointTrajectory>(arm_ref_topic, 10);

    // --- end-effector trajectory mode (arm GS "EE trajectory" tab) ---------
    ee_status_pub_ = create_publisher<String>(prefix + "/ee_trajectory/status", latched);
    ee_info_pub_ = create_publisher<Float64MultiArray>(prefix + "/ee_trajectory/info", latched);
    ee_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/ee_trajectory/path", latched);
    // The same run seen at the AIRFRAME. A separate topic rather than more
    // fields on /path: that array's 9-double stride is what its consumers
    // reshape by.
    ee_drone_path_pub_ = create_publisher<Float64MultiArray>(
      prefix + "/ee_trajectory/drone_path", latched);
    ee_start_err_pub_ = create_publisher<Float64MultiArray>(prefix + "/ee_trajectory/start_error", 10);
    ee_ref_pose_pub_ = create_publisher<PoseStamped>(prefix + "/ee_trajectory/reference_pose", 10);
    ee_drone_ref_pose_pub_ = create_publisher<PoseStamped>(
      prefix + "/ee_trajectory/drone_reference_pose", 10);
    // the run's start rest as a BASE pose (actual yaw), latched: what
    // go_to_start flies to and what the Start gate measures against
    ee_start_rest_pub_ = create_publisher<PoseStamped>(prefix + "/ee_trajectory/start_rest", latched);
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
    ee_pause_srv_ = create_service<Trigger>(
      prefix + "/ee_trajectory/pause",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onEePause(r);});
    ee_resume_srv_ = create_service<Trigger>(
      prefix + "/ee_trajectory/resume",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onEeResume(r);});
    ee_origin_srv_ = create_service<Trigger>(
      prefix + "/ee_trajectory/back_to_origin",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onEeBackToOrigin(r);});
    ee_err_timer_ = create_wall_timer(
      std::chrono::milliseconds(100), [this]() {publishEeStartError();});
    publishEeStatus("NONE");
    publishEePath(nullptr);

    // --- pick-and-place mode (arm GS "Pick & Place" tab) -------------------
    pp_status_pub_ = create_publisher<String>(prefix + "/pick_place/status", latched);
    pp_info_pub_ = create_publisher<Float64MultiArray>(prefix + "/pick_place/info", latched);
    // the planned mission (all six legs back to back), same 9-double stride
    // as ee_trajectory/path, at the claw and at the airframe
    pp_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/pick_place/path", latched);
    pp_drone_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/pick_place/drone_path", latched);
    pp_arrival_pub_ = create_publisher<Float64MultiArray>(prefix + "/pick_place/arrival_error", 10);
    const auto trigger = [this](const std::string & name, std::function<void(Trigger::Response::SharedPtr)> fn) {
        pp_srvs_.push_back(create_service<Trigger>(
          name, [fn](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {fn(r);}));
      };
    trigger(prefix + "/pick_place/adjust", [this](Trigger::Response::SharedPtr r) {onPpAdjust(r);});
    trigger(prefix + "/pick_place/plan", [this](Trigger::Response::SharedPtr r) {onPpPlan(r);});
    trigger(prefix + "/pick_place/reset", [this](Trigger::Response::SharedPtr r) {onPpReset(r);});
    for (int k = 0; k < kNumPickPlaceLegs; ++k) {
      trigger(
        prefix + "/pick_place/" + pickPlaceLegName(k),
        [this, k](Trigger::Response::SharedPtr r) {onPpFly(k, r);});
      trigger(
        prefix + "/pick_place/capture_" + kPpPoint[k],
        [this, k](Trigger::Response::SharedPtr r) {onPpCapture(k, r);});
      if (!pp_topic_[k].empty()) {
        // SensorData QoS (best effort) matches a reliable publisher as well
        pp_mocap_subs_.push_back(create_subscription<Mocap>(
          pp_topic_[k], rclcpp::SensorDataQoS(), [this, k](const Mocap & m) {onPpMocap(k, m);}));
      }
    }
    pp_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {publishPpArrival();});
    publishPpStatus();
    publishPpInfo();
    publishPpPaths(nullptr);

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
    if (pp_worker_.joinable()) {pp_worker_.join();}
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
        exec_pause_t_.reset();
        ++plan_gen_;
        goal_override_.reset();
        auto_send_ = false;
        ee_traj_.reset();
        ee_shape_type_.clear();
        ++ee_gen_;
        // pick-and-place: the plan and any leg in flight go; the captures,
        // the calibration and the progress stay (they are measurements and
        // the operator's bookkeeping), so a re-Plan resumes the mission
        pp_wp_.reset();
        pp_flying_.reset();
        pp_exec_plan_.reset();
        pp_planning_ = false;
        ++pp_gen_;
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
      publishPpPaths(nullptr);
    }
    publishPpStatus();
    publishPpInfo();
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
      gs_ref_z_ = x_b(2);
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
    bool heading_ok = false;
    const double az = clawAzimuth(vehicle_->params, r0 * vehicle_->r_model, *q, &heading_ok);
    if (heading_ok) {
      Float64 h;
      h.data = wrapPi(az + 0.5 * M_PI);   // model azimuth -> ACTUAL yaw, as ee_target reads it
      cur_ee_heading_pub_->publish(h);
    }
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
      // any other target supersedes a pick-and-place leg still planning
      pp_flying_.reset();
      pp_exec_plan_.reset();
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
    publishPpStatus();

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
              exec_pause_t_.reset();
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
      exec_pause_t_.reset();
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
      pp_flying_.reset();
      pp_exec_plan_.reset();
      if (mode_direct_) {state_ = "HOLD";}
    }
    publishStatus();
    publishPpStatus();
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
        const double t = exec_pause_t_.has_value()
          ? *exec_pause_t_
          : std::chrono::duration<double>(Clock::now() - *exec_t0_).count();
        if (t >= plan_->duration()) {
          finished = true;
        } else {
          ref = plan_->eval(t);
          have_ref = true;
        }
      }
      if (finished) {
        if (pp_flying_.has_value() && pp_exec_plan_ && pp_exec_plan_ == plan_) {
          pp_completed_ = *pp_flying_;
          RCLCPP_INFO(get_logger(), "pick-and-place: %s complete.", pickPlaceLegName(pp_completed_));
        }
        pp_flying_.reset();
        pp_exec_plan_.reset();
        setHold(plan_->goalRest());
        plan_.reset();
        pending_base_.reset();
        ee_target_.reset();
        home_goal_ = false;
        exec_pause_t_.reset();
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
      publishPpStatus();
      publishPpInfo();
    }
    publishRef(ref);
    // The arm reference, from the SAME sample the law gets, at the same rate.
    publishArmSync(ref.q_d, ref.qdot_d);
    viz_decim_ = (viz_decim_ + 1) % viz_decim_period_;
    if (viz_decim_ == 0) {
      publishVizPose(&ref);
      publishEeRefPose(ref);
      publishEeDroneRefPose(ref);
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
  // The reachable (r, z) set of the WHOLE-BODY chain as a filled grid
  // (workspace.hpp): [r_min, r_max, z_min, z_max, nr, nz, cells(nr*nz
  // row-major, rows = z from z_max down)], on two topics:
  //   workspace_rz             fold beta >= beta_min_deg, as it always was --
  //                            the EE Whole-Body and PS4 Remote tabs' envelope
  //   pick_place/workspace_rz  the floor lowered to just under the
  //                            pick-and-place poses when they fold below it
  //                            (the claw-down [0, 0, 0, 0] has beta = 0), those
  //                            extra poses keeping sigma_nd >= the margin: the
  //                            Pick & Place PS4 tab's envelope, without which
  //                            its fine correction could not move at the pick
  void publishWorkspace()
  {
    const double guard = vehicle_->beta_min_deg * M_PI / 180.0;
    double fold_min = guard;
    for (const char * name : {"pick_place_pick_pose_deg", "pick_place_place_pose_deg"}) {
      const auto q = get_parameter(name).as_double_array();
      if (q.size() == kNumJoints) {
        fold_min = std::min(fold_min, (q[1] + q[2] - 1.0) * M_PI / 180.0);
      }
    }
    const WorkspaceGrid g = usableWorkspace(*vehicle_, guard);
    const WorkspaceGrid gp = fold_min < guard ? usableWorkspace(*vehicle_, fold_min) : g;
    if (g.cells.empty() || gp.cells.empty()) {
      RCLCPP_ERROR(get_logger(), "usable workspace is EMPTY -- check the joint limits.");
      return;
    }
    ws_pub_->publish(gridMsg(g));
    pp_ws_pub_->publish(gridMsg(gp));
    RCLCPP_INFO(
      get_logger(), "usable workspace published: %dx%d grid, r [%.3f, %.3f] m, z "
      "[%.3f, %.3f] m, %.0f%% filled (fold beta >= %.0f deg); pick-and-place envelope "
      "down to fold %.0f deg (sigma_nd >= %.2f): z >= %.3f m", g.nr, g.nz, g.rs_min, g.rs_max,
      g.zs_min, g.zs_max, 100.0 * g.filledFraction(), vehicle_->beta_min_deg,
      fold_min * 180.0 / M_PI, vehicle_->sigma_nd_margin, gp.zs_min);
  }

  static Float64MultiArray gridMsg(const WorkspaceGrid & g)
  {
    Float64MultiArray m;
    m.data = {g.r_min, g.r_max, g.z_min, g.z_max, static_cast<double>(g.nr), static_cast<double>(g.nz)};
    m.data.reserve(6 + g.cells.size());
    for (uint8_t c : g.cells) {m.data.push_back(c);}
    return m;
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
    sh.center_origin = get_parameter("ee_traj_center_origin").as_bool();
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
        // ALWAYS re-measure the feasible time scale, even when the type is
        // unchanged: the shape parameters (ee_traj_circle_radius, fig8_a/b,
        // lap_time, laps) are live and a station edits them and re-selects to
        // apply them. Caching s_max across that would size the run by the
        // previous shape.
        ee_shape_type_ = type;
        ee_s_max_.reset();
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
        std::string smax_reason;
        if (need_smax) {
          s_max = EeTrajectoryPlanner::maxTimeScale(
            *vehicle_, hold, shape, opts, 6.0, 0.02, &smax_reason);
        }
        std::shared_ptr<Trajectory> traj;
        EeTrajectoryDiag diag;
        std::string err;
        EeTrajectoryOptions o = opts;
        o.time_scale = std::max(0.05, std::min(s_req, s_max > 0.0 ? s_max : s_req));
        if (s_max <= 0.0) {
          // name the bound that refuses even the slowest run: a bare "no
          // feasible time scale" leaves the operator with nothing to change
          err = smax_reason.empty()
            ? std::string("no feasible time scale for this trajectory from the current hold")
            : smax_reason;
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
        publishEeStartRest(traj->goalRest());
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
      pp_flying_.reset();
      pp_exec_plan_.reset();
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      goal_override_.reset();
      auto_send_ = false;
      ++plan_gen_;
      exec_t0_ = Clock::now();
      exec_pause_t_.reset();
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

  // PAUSE / RESUME. The run's clock stops; the reference stays on the point
  // of the planned run it had reached. Its velocity therefore STEPS to zero
  // (and back on resume) -- the reference is no longer dynamically
  // compatible across that instant, and the law absorbs the step. Sized for
  // an operator stopping a slow EE run to look at something, not for an
  // abort: to abort, revert to SAFETY.
  void onEePause(Trigger::Response::SharedPtr resp)
  {
    double t = 0.0, T = 0.0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (state_ != "EXECUTING" || !plan_ || !exec_t0_.has_value()) {
        resp->success = false;
        resp->message = "nothing is executing";
        return;
      }
      if (exec_pause_t_.has_value()) {
        resp->success = false;
        resp->message = "already paused";
        return;
      }
      t = std::chrono::duration<double>(Clock::now() - *exec_t0_).count();
      T = plan_->duration();
      exec_pause_t_ = std::min(std::max(t, 0.0), T);
    }
    publishStatus();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << "paused at t = " << t << " / " << T << " s -- the reference is held on the run";
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "EE trajectory: %s", resp->message.c_str());
  }

  void onEeResume(Trigger::Response::SharedPtr resp)
  {
    double t = 0.0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!exec_pause_t_.has_value()) {
        resp->success = false;
        resp->message = "not paused";
        return;
      }
      t = *exec_pause_t_;
      // rewind the clock's origin so the run continues from where it froze
      exec_t0_ = Clock::now() - std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(t));
      exec_pause_t_.reset();
    }
    publishStatus();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << "resumed from t = " << t << " s";
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "EE trajectory: %s", resp->message.c_str());
  }

  // BACK TO ORIGIN: the hover point [0, 0, z] with the arm folded home, z
  // being the drone GS's own commanded altitude (its last reference; the
  // current hold's height when it has sent none). Planned like any other
  // goal and left at PLANNED -- it is a room-crossing move, so it waits for
  // the operator's Start rather than flying on the button press the way
  // go_to_start does.
  void onEeBackToOrigin(Trigger::Response::SharedPtr resp)
  {
    double z = 0.0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {
        resp->success = false;
        resp->message = "not in whole-body DIRECT";
        return;
      }
      if (state_ == "EXECUTING") {
        resp->success = false;
        resp->message = "executing -- wait for the run to finish (or pause it) first";
        return;
      }
      z = gs_ref_z_.value_or(hold_->x_b(2));
      pending_base_ = std::make_pair(Vec3{0.0, 0.0, z}, hold_->phi);
      home_goal_ = true;
      ee_target_.reset();
      goal_override_.reset();
      auto_send_ = false;
      state_ = "PENDING";
    }
    publishBaseAnchor();
    publishStatus();
    startPlanning();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "planning the transition to the hover point [0, 0, " << z
      << "] with the arm home -- press Start once it reads PLANNED";
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "Back to origin: %s", resp->message.c_str());
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
      // [0..3] the three errors and the gate; [4..6] the tolerances each is
      // measured against, so a station can show the bound beside the value
      // instead of hard-coding this node's parameters.
      m.data = {pos, yaw * 180.0 / M_PI, joint * 180.0 / M_PI, ready ? 1.0 : 0.0,
        get_parameter("ee_traj_start_pos_tol").as_double(),
        get_parameter("ee_traj_start_yaw_tol_deg").as_double(),
        get_parameter("ee_traj_start_joint_tol_deg").as_double()};
    }
    ee_start_err_pub_->publish(m);
  }

  void publishEeStartRest(const RestSpec & r)
  {
    PoseStamped m;
    m.header.stamp = now();
    m.header.frame_id = "world";
    m.pose.position.x = r.x_b(0);
    m.pose.position.y = r.x_b(1);
    m.pose.position.z = r.x_b(2);
    const double yaw = r.phi + 0.5 * M_PI;   // model heading -> actual yaw
    m.pose.orientation.z = std::sin(0.5 * yaw);
    m.pose.orientation.w = std::cos(0.5 * yaw);
    ee_start_rest_pub_->publish(m);
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
      d.min_sigma_nd,
      // [14] the drone GS's commanded altitude -- the height this run
      // inherits, and where Back To Origin returns to. NaN until the GS has
      // sent a setpoint in DIRECT.
      gs_ref_z_.value_or(std::numeric_limits<double>::quiet_NaN())};
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

  // The AIRFRAME the same reference implies: body ORIGIN (not the system CoM
  // the law tracks -- x_b = x_cd - R0 r_0c(q_d), the base the odometry
  // reports) and the ACTUAL body attitude, R0 Rz(+pi/2), so its x axis is
  // the vehicle's nose.
  void dronePoseOf(const WbReference & ref, PoseStamped * m) const
  {
    const Vec3 ac = ref.x_cd_ddot + vehicle_->params.g * Vec3{0.0, 0.0, 1.0};
    Mat3 R0 = Mat3::Identity();
    if (ac.norm() > 1e-9) {R0 = buildR0(ac / ac.norm(), ref.b1_d);}
    Vec3 r0c;
    armKinematics(ref.q_d, vehicle_->params, &r0c, nullptr, nullptr);
    const Vec3 x_b = ref.x_cd - R0 * r0c;
    m->pose.position.x = x_b(0);
    m->pose.position.y = x_b(1);
    m->pose.position.z = x_b(2);
    const Eigen::Quaterniond q(R0 * Rz(0.5 * M_PI));
    m->pose.orientation.x = q.x();
    m->pose.orientation.y = q.y();
    m->pose.orientation.z = q.z();
    m->pose.orientation.w = q.w();
  }

  void publishEeDroneRefPose(const WbReference & ref)
  {
    PoseStamped m;
    m.header.stamp = now();
    m.header.frame_id = "world";
    dronePoseOf(ref, &m);
    ee_drone_ref_pose_pub_->publish(m);
  }

  // [t, x, y, z, speed, qx, qy, qz, qw] per sample, world frame; empty = clear
  void publishEePath(const Trajectory * traj)
  {
    Float64MultiArray m;
    if (traj != nullptr) {eePathRows(*traj, 400, &m.data);}
    ee_path_pub_->publish(m);
    publishEeDronePath(traj);
  }

  void publishEeDronePath(const Trajectory * traj)
  {
    Float64MultiArray m;
    if (traj != nullptr) {dronePathRows(*traj, 400, &m.data);}
    ee_drone_path_pub_->publish(m);
  }

  // n samples of the claw along a trajectory: [t, x, y, z, |v_ee|, quat].
  void eePathRows(const Trajectory & traj, int n, std::vector<double> * out) const
  {
    for (int k = 0; k < n; ++k) {
      const double t = traj.duration() * k / (n - 1);
      const WbReference ref = traj.eval(t);
      PoseStamped ps;
      eePoseOf(ref, &ps);
      out->push_back(t);
      out->push_back(ps.pose.position.x);
      out->push_back(ps.pose.position.y);
      out->push_back(ps.pose.position.z);
      out->push_back(ref.r_ed_dot.norm());
      out->push_back(ps.pose.orientation.x);
      out->push_back(ps.pose.orientation.y);
      out->push_back(ps.pose.orientation.z);
      out->push_back(ps.pose.orientation.w);
    }
  }

  // The same trajectory at the airframe, same [t, x, y, z, speed, quat]
  // layout. Speed is |dx_b/dt| by central difference on this grid: the body's
  // velocity is not a field of WbReference (x_cd_dot is the CoM's, and the
  // two differ by the arm's own motion), and this is a colour scale.
  void dronePathRows(const Trajectory & traj, int n, std::vector<double> * out) const
  {
    const double T = traj.duration();
    std::vector<double> t(n);
    std::vector<PoseStamped> pose(n);
    for (int k = 0; k < n; ++k) {
      t[k] = T * k / (n - 1);
      dronePoseOf(traj.eval(t[k]), &pose[k]);
    }
    for (int k = 0; k < n; ++k) {
      const int a = std::max(0, k - 1), b = std::min(n - 1, k + 1);
      const double dt = t[b] - t[a];
      double speed = 0.0;
      if (dt > 1e-9) {
        speed = std::hypot(
          std::hypot(
            pose[b].pose.position.x - pose[a].pose.position.x,
            pose[b].pose.position.y - pose[a].pose.position.y),
          pose[b].pose.position.z - pose[a].pose.position.z) / dt;
      }
      out->push_back(t[k]);
      out->push_back(pose[k].pose.position.x);
      out->push_back(pose[k].pose.position.y);
      out->push_back(pose[k].pose.position.z);
      out->push_back(speed);
      out->push_back(pose[k].pose.orientation.x);
      out->push_back(pose[k].pose.orientation.y);
      out->push_back(pose[k].pose.orientation.z);
      out->push_back(pose[k].pose.orientation.w);
    }
  }

  // ================================================= pick-and-place mode
  // Six legs (pick_place.hpp), each its own button. The goals come from the
  // nominal base poses (shifted by the Adjust offset) and the claw points
  // captured from mocap; `plan` dry-runs the whole mission from the hold, and
  // each leg's service re-plans that leg from the CURRENT hold -- the vehicle
  // may have been nudged since (the PS4 fine correction) -- and executes it
  // as soon as it is planned. Legs are flown in order; any earlier leg may be
  // flown again, none may be skipped.
  bool ppJoints(const std::string & name, VecN * q, std::string * err) const
  {
    const auto v = get_parameter(name).as_double_array();
    if (v.size() != kNumJoints) {
      *err = name + " needs 4 joint angles [deg]";
      return false;
    }
    for (int j = 0; j < kNumJoints; ++j) {(*q)(j) = v[j] * M_PI / 180.0;}
    return true;
  }

  bool ppVec3(const std::string & name, Vec3 * p, std::string * err) const
  {
    const auto v = get_parameter(name).as_double_array();
    if (v.size() != 3) {
      *err = name + " needs [x, y, z]";
      return false;
    }
    *p = Vec3{v[0], v[1], v[2]};
    return true;
  }

  // [x, y, z, yaw_deg] -> BasePose (ACTUAL yaw, rad)
  bool ppBase(const std::string & name, BasePose * b, std::string * err) const
  {
    const auto v = get_parameter(name).as_double_array();
    if (v.size() != 4) {
      *err = name + " needs [x, y, z, yaw_deg]";
      return false;
    }
    b->p = Vec3{v[0], v[1], v[2]};
    b->yaw = v[3] * M_PI / 180.0;
    return true;
  }

  bool ppConfig(PickPlaceConfig * c, std::string * err) const
  {
    c->home = home_pose_;
    if (!ppJoints("pick_place_carry_pose_deg", &c->carry, err) ||
      !ppJoints("pick_place_pick_pose_deg", &c->pick_pose, err) ||
      !ppJoints("pick_place_place_pose_deg", &c->place_pose, err) ||
      !ppVec3("pick_place_pick_ee_offset", &c->pick_ee_offset, err) ||
      !ppVec3("pick_place_place_ee_offset", &c->place_ee_offset, err) ||
      !ppJoints("pick_place_sweep_lo_deg", &c->sweep.lo, err) ||
      !ppJoints("pick_place_sweep_hi_deg", &c->sweep.hi, err) ||
      !ppJoints("pick_place_sweep_phase_deg", &c->sweep.phase, err))
    {
      return false;
    }
    c->face_target = get_parameter("pick_place_face_target").as_bool();
    c->retreat_dz = get_parameter("pick_place_retreat_dz").as_double();
    c->retreat_back = get_parameter("pick_place_retreat_back").as_double();
    ArmSweepOptions & o = c->sweep;
    o.cycles = static_cast<int>(get_parameter("pick_place_sweep_cycles").as_int());
    o.ramp_frac = get_parameter("pick_place_sweep_ramp_frac").as_double();
    o.qdot_max = get_parameter("pick_place_sweep_qdot_max").as_double();
    o.T_min = get_parameter("pick_place_sweep_t_min").as_double();
    o.T_max = get_parameter("pick_place_sweep_t_max").as_double();
    o.v_max = get_parameter("v_max").as_double();
    o.a_max = get_parameter("a_max").as_double();
    o.w_max = get_parameter("w_max").as_double();
    const double tjm = get_parameter("tau_joint_max").as_double();
    o.tau_joint_max = tjm > 0.0 ? tjm : -1.0;
    o.rotor_bounds = get_parameter("rotor_bounds").as_bool();
    o.sigma_nd_min = vehicle_->sigma_nd_margin;
    return true;
  }

  // The goals' inputs: the nominal base poses + the Adjust offset (x, y from
  // a mocap capture instead when one was taken) and the captured claw
  // points. Called under the lock.
  bool ppTargets(PickPlaceTargets * t, std::string * err) const
  {
    BasePose * base[kNumPickPlaceLegs] = {
      &t->start, nullptr, &t->place_start, nullptr, &t->land_start, &t->land};
    std::string missing;
    for (int k = 0; k < kNumPickPlaceLegs; ++k) {
      if (base[k] != nullptr) {
        if (!ppBase(std::string("pick_place_") + kPpPoint[k], base[k], err)) {return false;}
        base[k]->p += pp_offset_;
        base[k]->yaw += pp_yaw_offset_;
        if (pp_capture_[k].has_value()) {
          base[k]->p.head<2>() = pp_capture_[k]->head<2>();   // measured: no offset
        }
        continue;
      }
      if (!pp_capture_[k].has_value()) {
        missing += (missing.empty() ? "" : " and ") + std::string(kPpPoint[k]) + " (" +
          (pp_topic_[k].empty() ? "no topic set" : pp_topic_[k]) + ")";
        continue;
      }
      (k == kExecutePick ? t->pick : t->place) = *pp_capture_[k];
    }
    if (!missing.empty()) {
      *err = "capture " + missing + " first";
      return false;
    }
    return true;
  }

  void onPpMocap(int k, const Mocap & msg)
  {
    const Vec3 p{msg.pose.position.x, msg.pose.position.y, msg.pose.position.z};
    if (!p.allFinite()) {return;}
    std::lock_guard<std::recursive_mutex> lk(lock_);
    auto & d = pp_mocap_[k];
    d.emplace_back(Clock::now(), p);
    while (d.size() > 1 && (secondsSince(d.front().first) > 2.0 || d.size() > 2000)) {
      d.pop_front();
    }
  }

  // The inputs changed: a plan made from the old ones must not be flown.
  // Called under the lock.
  void ppInvalidate()
  {
    pp_wp_.reset();
    pp_planning_ = false;
    pp_error_.clear();
    ++pp_gen_;
  }

  void onPpCapture(int k, Trigger::Response::SharedPtr resp)
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pp_flying_.has_value()) {
        resp->success = false;
        resp->message = "a pick-and-place leg is in flight -- capture once it holds";
        return;
      }
      if (pp_topic_[k].empty()) {
        resp->success = false;
        resp->message = std::string("no mocap body for ") + kPpPoint[k] + " -- set pick_place_" +
          kPpPoint[k] + "_topic (the nominal pick_place_" + kPpPoint[k] + " is used meanwhile)";
        return;
      }
      const double window = get_parameter("pick_place_capture_window").as_double();
      Vec3 sum = Vec3::Zero();
      std::vector<Vec3> pts;
      for (const auto & s : pp_mocap_[k]) {
        if (secondsSince(s.first) <= window) {pts.push_back(s.second);}
      }
      if (pts.empty()) {
        m.precision(2);
        m << "no sample on " << pp_topic_[k] << " in the last " << window
          << " s -- is the body tracked and broadcast?";
        resp->success = false;
        resp->message = m.str();
        return;
      }
      for (const Vec3 & p : pts) {sum += p;}
      const Vec3 mean = sum / static_cast<double>(pts.size());
      double spread = 0.0;
      for (const Vec3 & p : pts) {spread = std::max(spread, (p - mean).norm());}
      const double max_spread = get_parameter("pick_place_capture_max_spread").as_double();
      if (spread > max_spread) {
        m.precision(1);
        m << pts.size() << " samples from " << pp_topic_[k] << " in " << window
          << " s scatter " << spread * 1e3 << " mm (> " << max_spread * 1e3
          << " mm): more than one publisher on the topic, or the body is moving or "
             "mis-tracked -- NOT captured";
        resp->success = false;
        resp->message = m.str();
        RCLCPP_WARN(get_logger(), "pick-and-place capture refused: %s", resp->message.c_str());
        return;
      }
      const bool was_planned = pp_wp_.has_value();
      pp_capture_[k] = mean;
      ppInvalidate();
      m.precision(3);
      m << "captured " << kPpPoint[k] << " at [" << mean(0) << ", " << mean(1) << ", " << mean(2)
        << "] m from " << pp_topic_[k] << " (" << pts.size() << " samples, spread "
        << std::setprecision(1) << spread * 1e3 << " mm)";
      if (k != kExecutePick && k != kExecutePlace) {m << "; x, y replace the nominal pose";}
      if (was_planned) {m << " -- the plan is stale, press Plan";}
    }
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "pick-and-place: %s", resp->message.c_str());
    publishPpStatus();
    publishPpInfo();
    publishPpPaths(nullptr);
  }

  // ADJUST: the vehicle is at the physical start mark (on the ground or in a
  // hover); its measured position minus the nominal start is the offset every
  // nominal point is shifted by -- the mocap centring changes between
  // sessions, the room does not.
  void onPpAdjust(Trigger::Response::SharedPtr resp)
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(3);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pp_flying_.has_value()) {
        resp->success = false;
        resp->message = "a pick-and-place leg is in flight -- adjust once it holds";
        return;
      }
      if (!odom_p_.has_value() || secondsSince(odom_p_time_) > kOdomFreshS) {
        resp->success = false;
        resp->message = "no fresh odometry -- cannot measure the vehicle's position";
        return;
      }
      BasePose nominal;
      std::string err;
      if (!ppBase("pick_place_start", &nominal, &err)) {
        resp->success = false;
        resp->message = err;
        return;
      }
      Vec3 off = *odom_p_ - nominal.p;
      if (!get_parameter("pick_place_adjust_z").as_bool()) {off(2) = 0.0;}
      double yoff = 0.0;
      if (get_parameter("pick_place_adjust_yaw").as_bool()) {
        if (!att_R_.has_value() || secondsSince(att_time_) > kOdomFreshS) {
          resp->success = false;
          resp->message = "pick_place_adjust_yaw is set but there is no fresh PX4 attitude";
          return;
        }
        yoff = wrapPi(std::atan2((*att_R_)(1, 0), (*att_R_)(0, 0)) - nominal.yaw);
      }
      pp_offset_ = off;
      pp_yaw_offset_ = yoff;
      ppInvalidate();
      m << "offset [" << off(0) << ", " << off(1) << ", " << off(2) << "] m, yaw "
        << std::setprecision(1) << yoff * 180.0 / M_PI
        << " deg -- every nominal point is shifted by it; press Plan";
    }
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "pick-and-place adjust: %s", resp->message.c_str());
    publishPpStatus();
    publishPpInfo();
    publishPpPaths(nullptr);
  }

  void onPpPlan(Trigger::Response::SharedPtr resp)
  {
    PickPlaceTargets tg;
    PickPlaceConfig cfg;
    PlanOptions opts;
    RestSpec from;
    unsigned gen = 0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {
        resp->success = false;
        resp->message = "not in whole-body DIRECT (the first leg is planned from the hold)";
        return;
      }
      if (pp_flying_.has_value() || state_ == "EXECUTING") {
        // the first leg is dry-run from the hold, which an execution is
        // about to replace
        resp->success = false;
        resp->message = "planner is " + state_ + " -- plan once it holds";
        return;
      }
      std::string err;
      if (!ppConfig(&cfg, &err) || !ppTargets(&tg, &err)) {
        resp->success = false;
        resp->message = err;
        return;
      }
      opts = planOptions();
      from = *hold_;
      ppInvalidate();
      gen = pp_gen_;
      pp_planning_ = true;
    }
    publishPpStatus();
    publishPpPaths(nullptr);
    if (pp_worker_.joinable()) {pp_worker_.join();}
    pp_worker_ = std::thread([this, tg, cfg, opts, from, gen]() {
        std::optional<PickPlaceWaypoints> wp;
        std::vector<std::shared_ptr<const Trajectory>> legs;
        std::shared_ptr<Trajectory> whole;
        std::string err;
        try {
          wp = pickPlaceWaypoints(*vehicle_, cfg, tg);
          legs = planPickPlaceMission(*vehicle_, *planner_, opts, cfg, *wp, from);
          whole = std::make_shared<SequenceTrajectory>(legs);
        } catch (const std::exception & e) {
          err = e.what();
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != pp_gen_) {return;}   // superseded
          pp_planning_ = false;
          if (!err.empty()) {
            pp_error_ = err;
            RCLCPP_WARN(get_logger(), "pick-and-place plan refused: %s", err.c_str());
          } else {
            pp_wp_ = wp;
            for (int k = 0; k < kNumPickPlaceLegs; ++k) {
              pp_leg_T_[k] = legs[k]->duration();
              RCLCPP_INFO(
                get_logger(), "pick-and-place %s: %s", pickPlaceLegName(k),
                legs[k]->diag().summary.c_str());
            }
            RCLCPP_INFO(get_logger(), "pick-and-place planned: %.1f s of flight in six legs.", whole->duration());
          }
        }
        publishPpStatus();
        publishPpInfo();
        publishPpPaths(whole.get());
      });
    resp->success = true;
    resp->message = "planning the six legs from the current hold -- the status reads READY when done";
  }

  void onPpFly(int leg, Trigger::Response::SharedPtr resp)
  {
    RestSpec from;
    PickPlaceWaypoints wp;
    PickPlaceConfig cfg;
    PlanOptions opts;
    unsigned gen = 0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::string err;
      if (!mode_direct_ || !hold_.has_value()) {
        err = "not in whole-body DIRECT";
      } else if (pp_planning_) {
        err = "the mission is still planning";
      } else if (!pp_wp_.has_value()) {
        err = "not planned -- capture the pick / place points and press Plan";
      } else if (state_ == "EXECUTING" || state_ == "CALCULATING") {
        err = "planner is " + state_ + " -- wait for HOLD";
      } else if (leg > pp_completed_ + 1) {
        err = std::string("out of order -- the next leg is ") + pickPlaceLegName(pp_completed_ + 1);
      }
      if (err.empty()) {ppConfig(&cfg, &err);}
      if (!err.empty()) {
        resp->success = false;
        resp->message = err;
        return;
      }
      from = *hold_;
      wp = *pp_wp_;
      opts = planOptions();
      gen = ++plan_gen_;
      state_ = "CALCULATING";
      plan_.reset();
      goal_override_.reset();
      auto_send_ = false;
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      exec_pause_t_.reset();
      pp_flying_ = leg;
      pp_exec_plan_.reset();
      pp_error_.clear();
    }
    publishStatus();
    publishPpStatus();
    publishPpInfo();
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, leg, from, wp, cfg, opts, gen]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          traj = planPickPlaceLeg(*vehicle_, *planner_, opts, cfg, wp, leg, from);
        } catch (const std::exception & e) {
          err = e.what();
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != plan_gen_ || state_ != "CALCULATING") {return;}   // superseded
          if (!traj) {
            state_ = "INFEASIBLE";
            infeasible_reason_ = err;
            pp_error_ = err;
            pp_flying_.reset();
            RCLCPP_WARN(get_logger(), "pick-and-place leg refused: %s", err.c_str());
          } else {
            plan_ = traj;
            pp_exec_plan_ = traj;
            exec_t0_ = Clock::now();
            exec_pause_t_.reset();
            state_ = "EXECUTING";
            RCLCPP_INFO(
              get_logger(), "pick-and-place %s: executing -- %s", pickPlaceLegName(leg),
              traj->diag().summary.c_str());
          }
        }
        publishStatus();
        publishPpStatus();
        publishPpInfo();
        publishVizPath(traj.get());
      });
    resp->success = true;
    resp->message = std::string("planning ") + pickPlaceLegName(leg) +
      " from the current hold; it executes as soon as it is planned";
    RCLCPP_INFO(get_logger(), "pick-and-place: %s", resp->message.c_str());
  }

  void onPpReset(Trigger::Response::SharedPtr resp)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pp_flying_.has_value()) {
        resp->success = false;
        resp->message = "a pick-and-place leg is in flight -- revert to SAFETY to abort it";
        return;
      }
      pp_completed_ = -1;
      ppInvalidate();
    }
    resp->success = true;
    resp->message = "pick-and-place progress and plan cleared (captures and the Adjust offset kept)";
    publishPpStatus();
    publishPpInfo();
    publishPpPaths(nullptr);
  }

  // 10 Hz: how far the leg in flight (else the last leg flown) is from its
  // target, measured -- the claw (FK on the encoders, odometry position, PX4
  // attitude) for execute_pick / execute_place, the body origin otherwise.
  // [0] leg, [1] 1 = claw target / 0 = base target, [2] |error| [m],
  // [3..5] error target - measured, world [m], [6] tolerance [m],
  // [7] within tolerance, [8] within AND holding: fine correction OK.
  void publishPpArrival()
  {
    Float64MultiArray m;
    bool gate_changed = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !pp_wp_.has_value()) {return;}
      const int leg = pp_flying_.has_value() ? *pp_flying_ : pp_completed_;
      if (leg < 0) {return;}
      Vec3 x_b;
      Mat3 r0;
      if (!odomPair(&x_b, &r0) || odomAge() > kOdomFreshS || !q_meas_.has_value()) {return;}
      const bool claw = pickPlaceLegIsClaw(leg);
      Vec3 meas = x_b, want = pp_wp_->goal[leg].x_b;
      if (claw) {
        Vec3 r0e;
        armKinematics(*q_meas_, vehicle_->params, nullptr, &r0e, nullptr);
        meas = x_b + r0 * (vehicle_->r_model * r0e);
        want = pp_wp_->ee[leg];
      }
      const Vec3 e = want - meas;
      const double tol = get_parameter("pick_place_arrival_tol").as_double();
      const bool within = e.norm() <= tol;
      const bool settled = within && !pp_flying_.has_value() && state_ == "HOLD";
      m.data = {static_cast<double>(leg), claw ? 1.0 : 0.0, e.norm(), e(0), e(1), e(2), tol,
        within ? 1.0 : 0.0, settled ? 1.0 : 0.0};
      if (settled != pp_settled_ || leg != pp_err_leg_) {
        pp_settled_ = settled;
        pp_err_leg_ = leg;
        gate_changed = true;
      }
    }
    pp_arrival_pub_->publish(m);
    if (gate_changed) {publishPpStatus();}
  }

  void publishPpStatus()
  {
    std::string s;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(1);
      if (!mode_direct_) {
        m << "NOT IN DIRECT";
      } else if (pp_planning_) {
        m << "PLANNING";
      } else if (pp_flying_.has_value()) {
        m << "FLYING " << pickPlaceLegName(*pp_flying_);
        if (state_ == "EXECUTING" && plan_) {m << " T=" << plan_->duration() << "s";}
      } else if (!pp_error_.empty()) {
        m << "INFEASIBLE: " << pp_error_;
      } else if (!pp_wp_.has_value()) {
        PickPlaceTargets t;
        std::string err;
        m << "NOT PLANNED: " << (ppTargets(&t, &err) ? std::string("press Plan") : err);
      } else if (pp_completed_ >= kNumPickPlaceLegs - 1) {
        m << "COMPLETE -- touch down with SAFETY -> land";
      } else if (pp_completed_ < 0) {
        m << "READY next=" << pickPlaceLegName(0);
      } else {
        m << "DONE " << pickPlaceLegName(pp_completed_) << " next="
          << pickPlaceLegName(pp_completed_ + 1);
        if (pickPlaceLegIsClaw(pp_completed_)) {
          m << (pp_settled_ ? " -- claw within tolerance: fine correction OK"
                            : " -- claw outside tolerance");
        }
      }
      s = m.str();
    }
    String msg;
    msg.data = s;
    pp_status_pub_->publish(msg);
  }

  // [0..2] Adjust offset [m], [3] yaw offset [deg], [4] planned, [5] last
  // leg completed (-1 none), [6] leg in flight (-1 none), [7] arrival
  // tolerance [m], [8..13] each leg's planned duration [s], [14..55] per leg
  // [goal base x, y, z, actual yaw deg, claw x, y, z], [56..79] per point
  // (start, pick, place_start, place, land_start, land) [captured x, y, z,
  // valid]. NaN where there is nothing.
  void publishPpInfo()
  {
    Float64MultiArray m;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      const double nan = std::numeric_limits<double>::quiet_NaN();
      m.data = {pp_offset_(0), pp_offset_(1), pp_offset_(2), pp_yaw_offset_ * 180.0 / M_PI,
        pp_wp_.has_value() ? 1.0 : 0.0, static_cast<double>(pp_completed_),
        pp_flying_.has_value() ? static_cast<double>(*pp_flying_) : -1.0,
        get_parameter("pick_place_arrival_tol").as_double()};
      for (int k = 0; k < kNumPickPlaceLegs; ++k) {
        m.data.push_back(pp_wp_.has_value() ? pp_leg_T_[k] : nan);
      }
      for (int k = 0; k < kNumPickPlaceLegs; ++k) {
        if (!pp_wp_.has_value()) {
          m.data.insert(m.data.end(), 7, nan);
          continue;
        }
        const RestSpec & g = pp_wp_->goal[k];
        const Vec3 & e = pp_wp_->ee[k];
        m.data.insert(
          m.data.end(), {g.x_b(0), g.x_b(1), g.x_b(2), wrapPi(g.phi + 0.5 * M_PI) * 180.0 / M_PI,
            e(0), e(1), e(2)});
      }
      for (int k = 0; k < kNumPickPlaceLegs; ++k) {
        if (pp_capture_[k].has_value()) {
          const Vec3 & p = *pp_capture_[k];
          m.data.insert(m.data.end(), {p(0), p(1), p(2), 1.0});
        } else {
          m.data.insert(m.data.end(), {nan, nan, nan, 0.0});
        }
      }
    }
    pp_info_pub_->publish(m);
  }

  void publishPpPaths(const Trajectory * whole)
  {
    Float64MultiArray a, b;
    if (whole != nullptr) {
      eePathRows(*whole, 600, &a.data);
      dronePathRows(*whole, 600, &b.data);
    }
    pp_path_pub_->publish(a);
    pp_drone_path_pub_->publish(b);
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
        // A SUFFIX, never a new state word: every consumer of this topic
        // tests the first token, and a pause is still an execution.
        m << "EXECUTING T=" << plan_->duration() << "s";
        if (exec_pause_t_.has_value()) {m << " PAUSED t=" << *exec_pause_t_ << "s";}
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
  // The drone GS's last commanded altitude, kept after pending_base_ is
  // consumed: it is the hover height Back To Origin returns to, and the arm
  // GS shows it as the height the trajectory inherits.
  std::optional<double> gs_ref_z_;
  std::optional<std::pair<Vec3, double>> ee_target_;     // (p_e_world, az)
  bool home_goal_{false};
  std::shared_ptr<Trajectory> plan_;
  std::optional<Clock::time_point> exec_t0_;
  // EE-trajectory PAUSE: the elapsed time the run is frozen at. While it is
  // set the stream keeps publishing plan_->eval(*exec_pause_t_), so the law
  // holds a reference that is STILL a point of the planned run rather than a
  // new hold -- resume is exact, at the cost of a step to zero in the
  // reference's velocity at the instant it is pressed.
  std::optional<double> exec_pause_t_;
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
  // pick-and-place mode
  std::array<std::string, kNumPickPlaceLegs> pp_topic_;          // mocap body per point
  Vec3 pp_offset_{Vec3::Zero()};                                  // Adjust
  double pp_yaw_offset_{0.0};
  std::array<std::optional<Vec3>, kNumPickPlaceLegs> pp_capture_;
  std::array<std::deque<std::pair<Clock::time_point, Vec3>>, kNumPickPlaceLegs> pp_mocap_;
  std::optional<PickPlaceWaypoints> pp_wp_;                       // set by Plan
  std::array<double, kNumPickPlaceLegs> pp_leg_T_{};
  bool pp_planning_{false};
  std::string pp_error_;
  int pp_completed_{-1};                                          // last leg flown to the end
  std::optional<int> pp_flying_;                                  // leg planning / executing
  std::shared_ptr<Trajectory> pp_exec_plan_;                      // ... and its trajectory
  bool pp_settled_{false};
  int pp_err_leg_{-1};
  unsigned pp_gen_{0};
  std::thread pp_worker_;

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
  rclcpp::Publisher<Float64>::SharedPtr cur_ee_heading_pub_;
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
  rclcpp::Publisher<Float64MultiArray>::SharedPtr ee_drone_path_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr ee_ref_pose_pub_, ee_start_rest_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr ee_drone_ref_pose_pub_;
  rclcpp::Subscription<String>::SharedPtr ee_select_sub_;
  rclcpp::Subscription<Float64>::SharedPtr ee_scale_sub_;
  rclcpp::Service<Trigger>::SharedPtr ee_go_srv_, ee_start_srv_;
  rclcpp::Service<Trigger>::SharedPtr ee_pause_srv_, ee_resume_srv_, ee_origin_srv_;
  rclcpp::TimerBase::SharedPtr ee_err_timer_;
  rclcpp::Publisher<String>::SharedPtr pp_status_pub_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr pp_info_pub_, pp_path_pub_, pp_drone_path_pub_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr pp_arrival_pub_, pp_ws_pub_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> pp_srvs_;
  std::vector<rclcpp::Subscription<Mocap>::SharedPtr> pp_mocap_subs_;
  rclcpp::TimerBase::SharedPtr pp_timer_;
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
