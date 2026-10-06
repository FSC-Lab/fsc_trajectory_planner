// MIT License
// Copyright (c) 2026 FSC Lab
//
// quadrotor_ccm_planner: streams the reference (x*, u*) of the torque-mode
// (R)CCM pipeline for a bare QUADROTOR (quadrotor_model.hpp) flying hover,
// circle or figure-8 (quadrotor_trajectory.hpp).
//
//   fmu/out/vehicle_attitude, state_estimator/local_position/odom  -> hold capture
//   fsc_autopilot_ros2/position_controller/ude (optional)          -> d_hat in the flatness map
//   fsc_autopilot_ros2/ccm_direct_actuation/mode                   -> hold when CCM is left
//   -> <prefix>/ccm_reference                   fsc_autopilot_ros2_msgs/CcmReference
//   -> fsc_autopilot_ros2/position_controller/reference  (optional; the SAFETY-mode
//      baseline flies the same path, so engaging CCM starts from a small error)
//   -> <prefix>/status (String), <prefix>/info (String, JSON, incl. the selection's XY
//      "path"), <prefix>/available_trajectories (String, JSON)   all latched
//
// Phases: IDLE -> HOLD <-> TRANSITION -> HOLD(at start) -> RUNNING -> FINISHED
// Services (std_srvs/Trigger) under <prefix>/, as wired to the ground station's
// "CCM Trajectory" tab:
//   hold           capture the current position/heading as the hover point, stream a hover there
//   go_to_start    fly (nonic rest-to-rest move) to the selected shape's start point, placed by
//                  `center` / `height`; captures the hover point first when IDLE
//   start          fly the selected shape (needs HOLD at its start point)
//   back_to_hover  ramp a running shape out, then fly back to the hover point
//   stop           ramp a running shape out and hold where it ends
//   release        stop streaming (IDLE)
// Topics in: <prefix>/select (String: circle | figure8 | circle_tangent | figure8_tangent),
// <prefix>/time_scale (Float64). The *_tangent shapes fly the same paths with the yaw along
// the path tangent: fly-to-start also turns the vehicle to the start heading, back-to-hover
// turns it back, and a start is refused above max_yaw_rate / max_yaw_accel.
//
// When the flight node leaves CCM for any reason (operator, abort, watchdog),
// a run or move is dropped and the hold is re-captured where the vehicle
// actually is (hold_on_ccm_exit), so the SAFETY baseline is never handed a
// moving target. The position-reference stream to the baseline also stops
// then (the flight node holds its own re-seeded point), so the operator's
// position commands -- the ground station's Move ENU, a landing -- are no
// longer overridden at 100 Hz; it resumes with the next hold / go_to_start /
// start. The CcmReference hold keeps streaming, ready for a re-engage.
//
// Rules shared with whole_body_trajectory_planner: the node never arms, never
// changes the PX4 mode and never commands the vehicle on its own -- nothing is
// published until an operator asks for a hold (or a fly-to-start).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "fsc_autopilot_ros2_msgs/msg/ccm_reference.hpp"
#include "fsc_autopilot_ros2_msgs/msg/position_controller_reference.hpp"
#include "fsc_autopilot_ros2_msgs/msg/ude_state.hpp"
#include "fsc_trajectory_planner/quadrotor_model.hpp"
#include "fsc_trajectory_planner/quadrotor_trajectory.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "px4_msgs/msg/vehicle_attitude.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/accel_stamped.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_srvs/srv/trigger.hpp"

namespace fsc_trajectory_planner
{

using CcmReference = fsc_autopilot_ros2_msgs::msg::CcmReference;
using PosRef = fsc_autopilot_ros2_msgs::msg::PositionControllerReference;
using UDEState = fsc_autopilot_ros2_msgs::msg::UDEState;
using Trigger = std_srvs::srv::Trigger;
using Clock = std::chrono::steady_clock;

namespace
{
constexpr double kFreshS = 0.5;
const Eigen::Matrix3d kRIE = (Eigen::Matrix3d() << 0, 1, 0, 1, 0, 0, 0, 0, -1).finished();
const Eigen::Matrix3d kRBV = (Eigen::Matrix3d() << 1, 0, 0, 0, -1, 0, 0, 0, -1).finished();

const char * phaseName(uint8_t p)
{
  switch (p) {
    case CcmReference::PHASE_HOLD: return "HOLD";
    case CcmReference::PHASE_RUNNING: return "RUNNING";
    case CcmReference::PHASE_FINISHED: return "FINISHED";
    case CcmReference::PHASE_TRANSITION: return "TRANSITION";
    default: return "IDLE";
  }
}

std::string vec(const Eigen::Vector3d & v)
{
  std::ostringstream o;
  o << std::fixed << std::setprecision(3) << "[" << v.x() << ", " << v.y() << ", " << v.z() << "]";
  return o.str();
}
}  // namespace

class QuadrotorCcmPlanner : public rclcpp::Node
{
public:
  QuadrotorCcmPlanner()
  : Node("quadrotor_ccm_planner")
  {
    model_ = makeQuadrotorModel(declare_parameter<std::string>("vehicle", "t650"));
    const double tau = declare_parameter<double>("rotor_time_constant", -1.0);
    if (tau >= 0.0) {model_.rotor_time_constant = tau;}
    shape_ = parseQuadShape(declare_parameter<std::string>("trajectory", "circle"));
    opts_.circle_radius = declare_parameter<double>("circle_radius", 1.0);
    opts_.fig8_a = declare_parameter<double>("fig8_a", 1.0);
    opts_.lap_time = declare_parameter<double>("lap_time", 8.0);
    opts_.laps = static_cast<int>(declare_parameter<int>("laps", 2));
    opts_.ramp_time = declare_parameter<double>("ramp_time", 3.0);
    opts_.ccw = declare_parameter<bool>("ccw", true);
    opts_.time_scale = declare_parameter<double>("time_scale", 1.0);
    ts_min_ = declare_parameter<double>("time_scale_min", 0.25);
    ts_max_ = declare_parameter<double>("time_scale_max", 1.5);
    const auto c = declare_parameter<std::vector<double>>("center", {0.0, 0.0});
    center_ = Eigen::Vector2d(c.size() > 0 ? c[0] : 0.0, c.size() > 1 ? c[1] : 0.0);
    height_ = declare_parameter<double>("height", -1.0);
    transition_speed_ = declare_parameter<double>("transition_speed", 0.4);
    transition_min_time_ = declare_parameter<double>("transition_min_time", 3.0);
    max_transition_distance_ = declare_parameter<double>("max_transition_distance", 4.0);
    start_tolerance_ = declare_parameter<double>("start_tolerance", 0.05);
    max_speed_ = declare_parameter<double>("max_speed", 2.0);
    max_accel_ = declare_parameter<double>("max_accel", 3.0);
    // Heading limits of the *_tangent shapes (body-rate / yaw-torque domain the CCM was certified on).
    max_yaw_rate_ = declare_parameter<double>("max_yaw_rate", 1.5);
    max_yaw_accel_ = declare_parameter<double>("max_yaw_accel", 2.0);
    transition_yaw_rate_ = declare_parameter<double>("transition_yaw_rate", 0.5);
    start_yaw_tolerance_ = declare_parameter<double>("start_yaw_tolerance", 0.05);
    use_ude_ = declare_parameter<bool>("use_disturbance_estimate", false);
    max_dist_ = declare_parameter<double>("max_disturbance_accel", 3.0);
    // Source of the folded-in estimate: "ude" = the SAFETY baseline's linear UDE force
    // (position_controller/ude, linear only); "ccm" = the CCM node's velocity-based estimator
    // (AccelStamped: linear [m/s^2] ENU + angular [rad/s^2] body FLU).
    dist_source_ = declare_parameter<std::string>("disturbance_source", "ccm");
    use_angular_ = declare_parameter<bool>("use_angular_disturbance", true);
    max_dist_ang_ = declare_parameter<double>("max_disturbance_angular_accel", 3.0);
    if (dist_source_ != "ude" && dist_source_ != "ccm") {
      throw std::runtime_error("disturbance_source must be 'ude' or 'ccm'");
    }
    publish_pos_ref_ = declare_parameter<bool>("publish_position_reference", true);
    hold_on_ccm_exit_ = declare_parameter<bool>("hold_on_ccm_exit", true);
    const double rate = declare_parameter<double>("stream_rate", 100.0);
    const auto prefix = declare_parameter<std::string>("topic_prefix", "quadrotor_planner");
    const auto odom_topic =
      declare_parameter<std::string>("odom_topic", "state_estimator/local_position/odom");
    const auto att_topic = declare_parameter<std::string>("attitude_topic", "fmu/out/vehicle_attitude");
    const auto ude_topic =
      declare_parameter<std::string>("ude_topic", "fsc_autopilot_ros2/position_controller/ude");
    const auto dist_topic = declare_parameter<std::string>(
      "disturbance_topic", "fsc_autopilot_ros2/ccm_direct_actuation/disturbance");
    const auto pos_ref_topic = declare_parameter<std::string>(
      "position_reference_topic", "fsc_autopilot_ros2/position_controller/reference");
    const auto mode_topic = declare_parameter<std::string>(
      "ccm_mode_topic", "fsc_autopilot_ros2/ccm_direct_actuation/mode");
    opts_.time_scale = std::clamp(opts_.time_scale, ts_min_, ts_max_);

    const auto latched = rclcpp::QoS(1).reliable().transient_local();
    const auto px4_qos = rclcpp::QoS(5).best_effort().durability_volatile();

    ref_pub_ = create_publisher<CcmReference>(prefix + "/ccm_reference", 10);
    pos_ref_pub_ = create_publisher<PosRef>(pos_ref_topic, 10);
    status_pub_ = create_publisher<std_msgs::msg::String>(prefix + "/status", latched);
    info_pub_ = create_publisher<std_msgs::msg::String>(prefix + "/info", latched);
    list_pub_ = create_publisher<std_msgs::msg::String>(prefix + "/available_trajectories", latched);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      odom_topic, 10, [this](const nav_msgs::msg::Odometry & m) {
        std::lock_guard<std::mutex> lk(lock_);
        odom_p_ = Eigen::Vector3d(m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z);
        odom_time_ = Clock::now();
      });
    att_sub_ = create_subscription<px4_msgs::msg::VehicleAttitude>(
      att_topic, px4_qos, [this](const px4_msgs::msg::VehicleAttitude & m) {
        Eigen::Quaterniond q(m.q[0], m.q[1], m.q[2], m.q[3]);
        if (!(q.norm() > 1e-9)) {return;}
        const Eigen::Matrix3d r_enu = kRIE * q.normalized().toRotationMatrix() * kRBV;
        std::lock_guard<std::mutex> lk(lock_);
        yaw_ = std::atan2(r_enu(1, 0), r_enu(0, 0));
        att_time_ = Clock::now();
      });
    if (use_ude_ && dist_source_ == "ccm") {
      dist_sub_ = create_subscription<geometry_msgs::msg::AccelStamped>(
        dist_topic, 10, [this](const geometry_msgs::msg::AccelStamped & m) {
          Eigen::Vector3d d(m.accel.linear.x, m.accel.linear.y, m.accel.linear.z);
          Eigen::Vector3d dw(m.accel.angular.x, m.accel.angular.y, m.accel.angular.z);
          if (!d.allFinite() || !dw.allFinite()) {return;}
          if (d.norm() > max_dist_) {d *= max_dist_ / d.norm();}
          if (dw.norm() > max_dist_ang_) {dw *= max_dist_ang_ / dw.norm();}
          std::lock_guard<std::mutex> lk(lock_);
          d_hat_ = d;
          dw_hat_ = dw;
          dist_time_ = Clock::now();
        });
    }
    if (use_ude_ && dist_source_ == "ude") {
      ude_sub_ = create_subscription<UDEState>(
        ude_topic, 10, [this](const UDEState & m) {
          // disturbance_estimate is a FORCE [N] (the flight node feeds -estimate
          // as its thrust feedforward) -> acceleration.
          Eigen::Vector3d d(m.disturbance_estimate.x, m.disturbance_estimate.y, m.disturbance_estimate.z);
          d /= model_.mass;
          if (!d.allFinite()) {return;}
          if (d.norm() > max_dist_) {d *= max_dist_ / d.norm();}
          std::lock_guard<std::mutex> lk(lock_);
          d_hat_ = m.is_active ? d : Eigen::Vector3d::Zero();
          dist_time_ = Clock::now();
        });
    }
    mode_sub_ = create_subscription<std_msgs::msg::String>(
      mode_topic, latched, [this](const std_msgs::msg::String & m) {onMode(m.data);});
    select_sub_ = create_subscription<std_msgs::msg::String>(
      prefix + "/select", 10, [this](const std_msgs::msg::String & m) {onSelect(m.data);});
    time_scale_sub_ = create_subscription<std_msgs::msg::Float64>(
      prefix + "/time_scale", 10, [this](const std_msgs::msg::Float64 & m) {onTimeScale(m.data);});

    const auto srv = [this, &prefix](const std::string & name, void (QuadrotorCcmPlanner::*fn)(Trigger::Response &)) {
        return create_service<Trigger>(
          prefix + "/" + name,
          [this, fn](Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {
            {
              std::lock_guard<std::mutex> lk(lock_);
              (this->*fn)(*r);
            }
            publishStatus();
            publishInfo();
          });
      };
    services_ = {srv("hold", &QuadrotorCcmPlanner::onHold), srv("go_to_start", &QuadrotorCcmPlanner::onGoToStart),
      srv("start", &QuadrotorCcmPlanner::onStart), srv("back_to_hover", &QuadrotorCcmPlanner::onBackToHover),
      srv("stop", &QuadrotorCcmPlanner::onStop), srv("release", &QuadrotorCcmPlanner::onRelease)};

    stream_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate), [this]() {streamTick();});
    status_timer_ = create_wall_timer(std::chrono::milliseconds(500), [this]() {
          publishStatus();
          publishInfo();
        });
    publishList();
    publishStatus();
    publishInfo();

    RCLCPP_INFO(
      get_logger(), "quadrotor_ccm_planner: model '%s' (m %.4f kg, tau %.4f s), trajectory '%s', "
      "R %.2f m, fig8 a %.2f m, lap %.2f s x %d, ramp %.2f s, time scale %.2f [%.2f, %.2f], centre "
      "(%.2f, %.2f), height %.2f, stream %.0f Hz, UDE %s (source %s, angular %s)",
      model_.name.c_str(), model_.mass, model_.rotor_time_constant, quadShapeName(shape_).c_str(),
      opts_.circle_radius, opts_.fig8_a, opts_.lap_time, opts_.laps, opts_.ramp_time, opts_.time_scale,
      ts_min_, ts_max_, center_.x(), center_.y(), height_, rate, use_ude_ ? "on" : "off", dist_source_.c_str(),
      (use_ude_ && dist_source_ == "ccm" && use_angular_) ? "on" : "off");
  }

private:
  static double elapsed(Clock::time_point since)
  {
    return std::chrono::duration<double>(Clock::now() - since).count();
  }

  bool fresh() const
  {
    return odom_time_ && att_time_ && elapsed(*odom_time_) <= kFreshS && elapsed(*att_time_) <= kFreshS;
  }

  // Where the selected shape starts: its centre at (center, height or the hover height).
  Eigen::Vector3d startPoint() const
  {
    const double z = height_ > 0.0 ? height_ : (home_valid_ ? home_p_.z() : hold_p_.z());
    return Eigen::Vector3d(center_.x(), center_.y(), z) + quadShapeStartOffset(shape_, opts_);
  }

  // Heading the selected shape starts with: the path tangent for the *_tangent shapes,
  // otherwise the heading currently held.
  double startYaw() const
  {
    if (!quadShapeYawTangent(shape_)) {return hold_yaw_;}
    return QuadTrajectory(shape_, opts_, startPoint()).startYaw(hold_yaw_);
  }

  bool atStart() const
  {
    return (phase_ == CcmReference::PHASE_HOLD || phase_ == CcmReference::PHASE_FINISHED) &&
           shape_ != QuadShape::kHover && (hold_p_ - startPoint()).norm() <= start_tolerance_ &&
           std::abs(wrapAngle(hold_yaw_ - startYaw())) <= start_yaw_tolerance_;
  }

  void beginMove(const Eigen::Vector3d & goal, double goal_yaw)
  {
    move_ = std::make_unique<QuadTransition>(hold_p_, goal, transition_speed_, transition_min_time_,
        hold_yaw_, goal_yaw, transition_yaw_rate_);
    t0_ = Clock::now();
    phase_ = CcmReference::PHASE_TRANSITION;
    pos_ref_active_ = true;
  }

  bool captureHold(Trigger::Response & res)
  {
    if (!fresh()) {
      res.success = false;
      res.message = "no fresh odometry / PX4 attitude to capture a hold from";
      return false;
    }
    hold_p_ = odom_p_;
    hold_yaw_ = yaw_;
    home_p_ = hold_p_;
    home_yaw_ = hold_yaw_;
    home_valid_ = true;
    traj_.reset();
    move_.reset();
    return_home_after_run_ = false;
    phase_ = CcmReference::PHASE_HOLD;
    pos_ref_active_ = true;
    return true;
  }

  void onMode(const std::string & mode)
  {
    std::lock_guard<std::mutex> lk(lock_);
    const bool left_ccm = last_mode_ == "CCM" && mode != "CCM";
    last_mode_ = mode;
    if (!left_ccm || !hold_on_ccm_exit_ || phase_ == CcmReference::PHASE_IDLE) {return;}
    const bool odom_fresh = odom_time_ && elapsed(*odom_time_) <= kFreshS;
    if (odom_fresh) {hold_p_ = odom_p_;}
    if (att_time_ && elapsed(*att_time_) <= kFreshS) {hold_yaw_ = yaw_;}
    traj_.reset();
    move_.reset();
    return_home_after_run_ = false;
    phase_ = CcmReference::PHASE_HOLD;
    pos_ref_active_ = false;
    RCLCPP_WARN(
      get_logger(), "flight node left CCM (now %s): run dropped, holding at %s%s; position reference "
      "released to the operator", mode.c_str(), vec(hold_p_).c_str(),
      odom_fresh ? "" : " (odometry stale: last hold point)");
  }

  void onSelect(const std::string & name)
  {
    {
      std::lock_guard<std::mutex> lk(lock_);
      if (phase_ == CcmReference::PHASE_RUNNING || phase_ == CcmReference::PHASE_TRANSITION) {
        RCLCPP_WARN(get_logger(), "select '%s' ignored while %s", name.c_str(), phaseName(phase_));
        return;
      }
      try {
        shape_ = parseQuadShape(name);
        RCLCPP_INFO(get_logger(), "trajectory selected: %s", name.c_str());
      } catch (const std::exception & e) {
        RCLCPP_WARN(get_logger(), "%s", e.what());
      }
    }
    publishInfo();
  }

  void onTimeScale(double s)
  {
    {
      std::lock_guard<std::mutex> lk(lock_);
      if (!std::isfinite(s)) {return;}
      opts_.time_scale = std::clamp(s, ts_min_, ts_max_);
      RCLCPP_INFO(get_logger(), "time scale %.2f (applies at the next start)%s", opts_.time_scale,
        phase_ == CcmReference::PHASE_RUNNING ? "; the running shape keeps its speed" : "");
    }
    publishInfo();
  }

  void onHold(Trigger::Response & res)
  {
    if (phase_ == CcmReference::PHASE_RUNNING || phase_ == CcmReference::PHASE_TRANSITION) {
      res.success = false;
      res.message = std::string("busy (") + phaseName(phase_) + "): use back_to_hover or stop";
      return;
    }
    if (!captureHold(res)) {return;}
    res.success = true;
    res.message = "holding at " + vec(hold_p_) + " (hover point)";
    RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
  }

  void onGoToStart(Trigger::Response & res)
  {
    if (phase_ == CcmReference::PHASE_RUNNING || phase_ == CcmReference::PHASE_TRANSITION) {
      res.success = false;
      res.message = std::string("busy (") + phaseName(phase_) + ")";
      return;
    }
    if (shape_ == QuadShape::kHover) {
      res.success = false;
      res.message = "select a trajectory (circle, figure8, circle_tangent, figure8_tangent) first";
      return;
    }
    if (phase_ == CcmReference::PHASE_IDLE && !captureHold(res)) {return;}
    const Eigen::Vector3d goal = startPoint();
    const double goal_yaw = startYaw();
    const double d = (goal - hold_p_).norm();
    if (d > max_transition_distance_) {
      res.success = false;
      res.message = "start point " + vec(goal) + " is " + std::to_string(d) + " m away (max_transition_distance)";
      return;
    }
    if (d <= start_tolerance_ && std::abs(wrapAngle(goal_yaw - hold_yaw_)) <= start_yaw_tolerance_) {
      res.success = true;
      res.message = "already at the start point " + vec(goal);
      return;
    }
    beginMove(goal, goal_yaw);
    std::ostringstream o;
    o << "flying to the " << quadShapeName(shape_) << " start " << vec(goal) << ", yaw " << std::setprecision(3)
      << goal_yaw * 180.0 / M_PI << " deg, " << move_->duration() << " s";
    res.success = true;
    res.message = o.str();
    RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
  }

  void onStart(Trigger::Response & res)
  {
    if (shape_ == QuadShape::kHover) {
      res.success = false;
      res.message = "select a trajectory (circle, figure8, circle_tangent, figure8_tangent) first";
      return;
    }
    if (!atStart()) {
      res.success = false;
      res.message = std::string("not holding at the start point (phase ") + phaseName(phase_) +
        "): press 'Fly to start' first";
      return;
    }
    try {
      auto traj = std::make_unique<QuadTrajectory>(shape_, opts_, startPoint());
      const double v = traj->cruiseSpeedMax(), a = traj->cruiseAccelMax();
      if (v > max_speed_ || a > max_accel_) {
        std::ostringstream o;
        o << std::setprecision(3) << "rejected at time scale " << opts_.time_scale << ": cruise speed " << v
          << " m/s / accel " << a << " m/s^2 exceed max_speed " << max_speed_ << " / max_accel " << max_accel_;
        res.success = false;
        res.message = o.str();
        return;
      }
      const double yr = traj->yawRateMax(), ya = traj->yawAccelMax();
      if (yr > max_yaw_rate_ || ya > max_yaw_accel_) {
        std::ostringstream o;
        o << std::setprecision(3) << "rejected at time scale " << opts_.time_scale << ": yaw rate " << yr
          << " rad/s / yaw accel " << ya << " rad/s^2 exceed max_yaw_rate " << max_yaw_rate_ << " / max_yaw_accel "
          << max_yaw_accel_ << " (lower the time scale)";
        res.success = false;
        res.message = o.str();
        return;
      }
      traj_ = std::move(traj);
      t0_ = Clock::now();
      phase_ = CcmReference::PHASE_RUNNING;
      return_home_after_run_ = false;
      pos_ref_active_ = true;
      std::ostringstream o;
      o << std::setprecision(3) << "running " << quadShapeName(shape_) << " x" << opts_.time_scale << ", "
        << traj_->duration() << " s, cruise " << v << " m/s, " << a << " m/s^2";
      res.success = true;
      res.message = o.str();
      RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
    } catch (const std::exception & e) {
      res.success = false;
      res.message = e.what();
    }
  }

  void onBackToHover(Trigger::Response & res)
  {
    if (!home_valid_) {
      res.success = false;
      res.message = "no hover point yet (hold or fly to start first)";
      return;
    }
    if (phase_ == CcmReference::PHASE_RUNNING && traj_) {
      traj_->requestStop(elapsed(t0_));
      return_home_after_run_ = true;
      res.success = true;
      res.message = "ramping out (ends t = " + std::to_string(traj_->duration()) +
        " s), then back to hover " + vec(home_p_);
      RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
      return;
    }
    if (phase_ == CcmReference::PHASE_TRANSITION) {
      res.success = false;
      res.message = "a move is in progress; wait for it to finish";
      return;
    }
    if (phase_ == CcmReference::PHASE_IDLE) {
      res.success = false;
      res.message = "planner idle";
      return;
    }
    if ((home_p_ - hold_p_).norm() <= 0.01 && std::abs(wrapAngle(home_yaw_ - hold_yaw_)) <= start_yaw_tolerance_) {
      res.success = true;
      res.message = "already at the hover point";
      return;
    }
    beginMove(home_p_, home_yaw_);
    res.success = true;
    res.message = "flying back to hover " + vec(home_p_) + ", " + std::to_string(move_->duration()) + " s";
    RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
  }

  void onStop(Trigger::Response & res)
  {
    if (phase_ != CcmReference::PHASE_RUNNING || !traj_) {
      res.success = false;
      res.message = "nothing running";
      return;
    }
    traj_->requestStop(elapsed(t0_));
    return_home_after_run_ = false;
    res.success = true;
    res.message = "ramping out; ends at t = " + std::to_string(traj_->duration()) + " s";
    RCLCPP_INFO(get_logger(), "%s", res.message.c_str());
  }

  void onRelease(Trigger::Response & res)
  {
    phase_ = CcmReference::PHASE_IDLE;
    traj_.reset();
    move_.reset();
    return_home_after_run_ = false;
    res.success = true;
    res.message = "released: reference streaming stopped";
    RCLCPP_WARN(get_logger(), "%s", res.message.c_str());
  }

  void streamTick()
  {
    std::lock_guard<std::mutex> lk(lock_);
    if (phase_ == CcmReference::PHASE_IDLE) {return;}
    double t = 0.0;
    if (phase_ == CcmReference::PHASE_RUNNING && traj_) {
      t = elapsed(t0_);
      if (traj_->finished(t)) {
        hold_p_ = traj_->jets(traj_->duration()).p;
        hold_yaw_ = traj_->yawJets(traj_->duration(), hold_yaw_).psi;   // the tangent shapes end turned
        traj_.reset();
        t = 0.0;
        if (return_home_after_run_) {
          return_home_after_run_ = false;
          beginMove(home_p_, home_yaw_);
          RCLCPP_INFO(get_logger(), "run finished; flying back to hover %s", vec(home_p_).c_str());
        } else {
          phase_ = CcmReference::PHASE_FINISHED;
          RCLCPP_INFO(get_logger(), "trajectory finished; holding at %s", vec(hold_p_).c_str());
        }
      }
    }
    if (phase_ == CcmReference::PHASE_TRANSITION && move_) {
      t = elapsed(t0_);
      if (t >= move_->duration()) {
        hold_p_ = move_->goal();
        hold_yaw_ = move_->goalYaw();
        move_.reset();
        phase_ = CcmReference::PHASE_HOLD;
        t = 0.0;
        RCLCPP_INFO(get_logger(), "move finished; holding at %s%s", vec(hold_p_).c_str(),
          atStart() ? " (trajectory start)" : "");
      }
    }
    // A stale estimate (publisher gone) is dropped rather than held.
    const bool dist_fresh = dist_time_ && elapsed(*dist_time_) <= kFreshS;
    const Eigen::Vector3d d = (use_ude_ && dist_fresh) ? d_hat_ : Eigen::Vector3d::Zero();
    const Eigen::Vector3d dw = (use_ude_ && use_angular_ && dist_source_ == "ccm" && dist_fresh) ?
      dw_hat_ : Eigen::Vector3d::Zero();
    CcmReferenceSample s;
    try {
      s = move_ ? sampleCcmReference(model_, *move_, t, d, 1e-3, dw) :
        sampleCcmReference(model_, traj_.get(), hold_p_, hold_yaw_, t, d, 1e-3, dw);
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "reference sample failed: %s", e.what());
      return;
    }
    const auto & r = s.ref;
    const auto stamp = now();

    CcmReference msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "map";
    msg.phase = phase_;
    msg.trajectory = move_ ? "transition" : (traj_ ? quadShapeName(traj_->shape()) : "hover");
    msg.t = t;
    msg.position.x = r.position.x(); msg.position.y = r.position.y(); msg.position.z = r.position.z();
    msg.velocity.x = r.velocity.x(); msg.velocity.y = r.velocity.y(); msg.velocity.z = r.velocity.z();
    msg.acceleration.x = r.acceleration.x(); msg.acceleration.y = r.acceleration.y();
    msg.acceleration.z = r.acceleration.z();
    const Eigen::Quaterniond q(r.R);
    msg.orientation.w = q.w(); msg.orientation.x = q.x(); msg.orientation.y = q.y(); msg.orientation.z = q.z();
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) {msg.rotation[3 * i + k] = r.R(i, k);}
    }
    msg.angular_velocity.x = r.omega.x(); msg.angular_velocity.y = r.omega.y();
    msg.angular_velocity.z = r.omega.z();
    msg.yaw = r.yaw;
    msg.thrust = r.thrust;
    msg.torque.x = r.torque.x(); msg.torque.y = r.torque.y(); msg.torque.z = r.torque.z();
    msg.thrust_command = s.thrust_command;
    msg.torque_command.x = s.torque_command.x(); msg.torque_command.y = s.torque_command.y();
    msg.torque_command.z = s.torque_command.z();
    msg.disturbance.x = d.x(); msg.disturbance.y = d.y(); msg.disturbance.z = d.z();
    msg.angular_disturbance.x = dw.x(); msg.angular_disturbance.y = dw.y(); msg.angular_disturbance.z = dw.z();
    ref_pub_->publish(msg);

    if (publish_pos_ref_ && pos_ref_active_) {
      PosRef p;
      p.header.stamp = stamp;
      p.header.frame_id = "map";
      p.position = msg.position;
      p.velocity = msg.velocity;
      p.acceleration = msg.acceleration;
      p.yaw = r.yaw;
      p.yaw_unit = PosRef::RADIANS;
      pos_ref_pub_->publish(p);
    }
  }

  void publishStatus()
  {
    std::lock_guard<std::mutex> lk(lock_);
    std::ostringstream oss;
    oss << phaseName(phase_);
    if (phase_ == CcmReference::PHASE_RUNNING && traj_) {
      oss << " " << quadShapeName(traj_->shape()) << " t=" << std::fixed << std::setprecision(1) << elapsed(t0_)
          << "/" << traj_->duration() << "s";
    } else if (phase_ == CcmReference::PHASE_TRANSITION && move_) {
      oss << " t=" << std::fixed << std::setprecision(1) << elapsed(t0_) << "/" << move_->duration() << "s";
    } else {
      oss << " (selected " << quadShapeName(shape_) << (atStart() ? ", at start" : "") << ")";
    }
    std_msgs::msg::String m;
    m.data = oss.str();
    status_pub_->publish(m);
  }

  // Everything the GUI shows about the selection, as flat JSON.
  void publishInfo()
  {
    std::lock_guard<std::mutex> lk(lock_);
    std::ostringstream o;
    o << std::fixed << std::setprecision(4) << "{\"phase\": \"" << phaseName(phase_) << "\", \"selected\": \""
      << quadShapeName(shape_) << "\", \"time_scale\": " << opts_.time_scale << ", \"time_scale_min\": " << ts_min_
      << ", \"time_scale_max\": " << ts_max_ << ", \"at_start\": " << (atStart() ? "true" : "false")
      << ", \"pos_ref\": " << ((publish_pos_ref_ && pos_ref_active_ && phase_ != CcmReference::PHASE_IDLE) ?
      "true" : "false");
    double t = 0.0, T = 0.0;
    if (phase_ == CcmReference::PHASE_RUNNING && traj_) {
      t = elapsed(t0_); T = traj_->duration();
    } else if (phase_ == CcmReference::PHASE_TRANSITION && move_) {
      t = elapsed(t0_); T = move_->duration();
    }
    o << ", \"t\": " << t << ", \"T\": " << T;
    if (shape_ != QuadShape::kHover) {
      try {
        const QuadTrajectory probe(shape_, opts_, startPoint());
        o << ", \"lap_time\": " << opts_.lap_time / opts_.time_scale << ", \"duration\": " << probe.duration()
          << ", \"cruise_speed\": " << probe.cruiseSpeedMax() << ", \"cruise_accel\": " << probe.cruiseAccelMax()
          << ", \"yaw_tangent\": " << (quadShapeYawTangent(shape_) ? "true" : "false")
          << ", \"yaw_rate_max\": " << probe.yawRateMax() << ", \"yaw_accel_max\": " << probe.yawAccelMax()
          << ", \"start_yaw\": " << probe.startYaw(hold_yaw_)
          << ", \"within_limits\": "
          << ((probe.cruiseSpeedMax() <= max_speed_ && probe.cruiseAccelMax() <= max_accel_ &&
          probe.yawRateMax() <= max_yaw_rate_ && probe.yawAccelMax() <= max_yaw_accel_) ? "true" : "false");
        // The XY path the selection would fly, for the ground station's preview.
        constexpr int kPathSamples = 128;
        o << ", \"path\": [";
        for (int i = 0; i < kPathSamples; ++i) {
          const Eigen::Vector3d p = probe.jets(probe.duration() * i / (kPathSamples - 1)).p;
          o << (i ? ", [" : "[") << p.x() << ", " << p.y() << "]";
        }
        o << "]";
      } catch (const std::exception &) {
        o << ", \"within_limits\": false";
      }
      const Eigen::Vector3d sp = startPoint();
      o << ", \"start\": [" << sp.x() << ", " << sp.y() << ", " << sp.z() << "]";
    }
    if (home_valid_) {o << ", \"hover\": [" << home_p_.x() << ", " << home_p_.y() << ", " << home_p_.z() << "]";}
    o << ", \"max_speed\": " << max_speed_ << ", \"max_accel\": " << max_accel_ << ", \"max_yaw_rate\": "
      << max_yaw_rate_ << ", \"max_yaw_accel\": " << max_yaw_accel_ << "}";
    std_msgs::msg::String m;
    m.data = o.str();
    info_pub_->publish(m);
  }

  void publishList()
  {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2) << "[{\"name\": \"circle\", \"label\": \"Circle (R " << opts_.circle_radius
      << " m)\", \"lap_time\": " << opts_.lap_time << ", \"laps\": " << opts_.laps
      << "}, {\"name\": \"figure8\", \"label\": \"Figure-8 (" << 2 * opts_.fig8_a << " x " << opts_.fig8_a
      << " m)\", \"lap_time\": " << opts_.lap_time << ", \"laps\": " << opts_.laps
      << "}, {\"name\": \"circle_tangent\", \"label\": \"Circle, yaw along path (R " << opts_.circle_radius
      << " m)\", \"lap_time\": " << opts_.lap_time << ", \"laps\": " << opts_.laps
      << "}, {\"name\": \"figure8_tangent\", \"label\": \"Figure-8, yaw along path (" << 2 * opts_.fig8_a
      << " x " << opts_.fig8_a << " m)\", \"lap_time\": " << opts_.lap_time << ", \"laps\": " << opts_.laps << "}]";
    std_msgs::msg::String m;
    m.data = o.str();
    list_pub_->publish(m);
  }

  QuadrotorModel model_;
  QuadShape shape_{QuadShape::kCircle};
  QuadTrajectoryOptions opts_;
  Eigen::Vector2d center_{Eigen::Vector2d::Zero()};
  double height_{-1.0};
  double ts_min_{0.25}, ts_max_{1.5};
  double transition_speed_{0.4}, transition_min_time_{3.0}, max_transition_distance_{4.0}, start_tolerance_{0.05};
  double max_speed_{2.0}, max_accel_{3.0}, max_dist_{3.0}, max_dist_ang_{3.0};
  std::string dist_source_{"ccm"};
  bool use_angular_{false};
  double max_yaw_rate_{1.5}, max_yaw_accel_{2.0}, transition_yaw_rate_{0.5}, start_yaw_tolerance_{0.05};
  bool use_ude_{false}, publish_pos_ref_{true}, hold_on_ccm_exit_{true};
  std::string last_mode_;

  std::mutex lock_;
  uint8_t phase_{CcmReference::PHASE_IDLE};
  std::unique_ptr<QuadTrajectory> traj_;
  std::unique_ptr<QuadTransition> move_;
  bool return_home_after_run_{false};
  // Whether the baseline's position reference follows this planner (see the header comment).
  bool pos_ref_active_{false};
  Clock::time_point t0_;
  Eigen::Vector3d hold_p_{Eigen::Vector3d::Zero()};
  double hold_yaw_{0.0};
  double home_yaw_{0.0};
  Eigen::Vector3d home_p_{Eigen::Vector3d::Zero()};
  bool home_valid_{false};
  Eigen::Vector3d odom_p_{Eigen::Vector3d::Zero()};
  double yaw_{0.0};
  std::optional<Clock::time_point> odom_time_, att_time_;
  Eigen::Vector3d d_hat_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d dw_hat_{Eigen::Vector3d::Zero()};
  std::optional<Clock::time_point> dist_time_;

  rclcpp::Publisher<CcmReference>::SharedPtr ref_pub_;
  rclcpp::Publisher<PosRef>::SharedPtr pos_ref_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_, info_pub_, list_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<px4_msgs::msg::VehicleAttitude>::SharedPtr att_sub_;
  rclcpp::Subscription<UDEState>::SharedPtr ude_sub_;
  rclcpp::Subscription<geometry_msgs::msg::AccelStamped>::SharedPtr dist_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr select_sub_, mode_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr time_scale_sub_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> services_;
  rclcpp::TimerBase::SharedPtr stream_timer_, status_timer_;
};

}  // namespace fsc_trajectory_planner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<fsc_trajectory_planner::QuadrotorCcmPlanner>());
  rclcpp::shutdown();
  return 0;
}
