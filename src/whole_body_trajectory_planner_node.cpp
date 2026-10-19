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
//     PUSH-AND-PULL MODE (2026-10-03): six steps (go_to_start, ready,
//                 push, exit, go_to_land_start, land; five until 2026-10-16)
//                 that grasp a box on a table by a
//                 handle on its side and slide it along the table with the
//                 arm held still; the law's contact phase flag is CONTACT
//                 from the claw's arrival on the handle (end of Ready) to the
//                 Exit press, the claw world-held throughout (push_pull.hpp).
//     TELEOP (2026-09-27): the PS4 pad drives the reference in real time --
//                 D-pad + action buttons the CoM and heading, the sticks the
//                 grasp point relative to the airframe and the wrist roll --
//                 integrated to setpoints and streamed as compatible
//                 WholeBodyReference samples (teleop_reference.hpp). Entered
//                 from HOLD through whole_body_planner/teleop/engage; on
//                 disengage (TELEOP_STOP) the smoothers settle and the final
//                 rest becomes the new HOLD. Every other target is refused
//                 while it runs.
//   Mode leaves DIRECT at ANY point -> streaming stops instantly.
//
// FRAMES: the ROS boundary is the ACTUAL world/FLU convention (odometry, GS
// yaw, EE targets); the planner and the streamed message are MODEL frame.
//     R0_model = R0_actual @ R_MODEL          phi_model = psi_actual - pi/2
//
// Every topic and service is RELATIVE, so `--ros-args -r __ns:=/uav_N` (the
// launch file's uav_prefix) namespaces the whole interface per vehicle.

#include <Eigen/Geometry>

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
#include <sensor_msgs/msg/joy.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <px4_msgs/msg/vehicle_attitude.hpp>

#include <fsc_autopilot_ros2_msgs/msg/mocap.hpp>
#include <fsc_autopilot_ros2_msgs/msg/position_controller_reference.hpp>
#include <fsc_autopilot_ros2_msgs/msg/whole_body_reference.hpp>

#include "fsc_trajectory_planner/ee_trajectory_planner.hpp"
#include "fsc_trajectory_planner/kinematics.hpp"
#include "fsc_trajectory_planner/pick_place.hpp"
#include "fsc_trajectory_planner/push_pull.hpp"
#include "fsc_trajectory_planner/teleop_reference.hpp"
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
using sensor_msgs::msg::Joy;
using std_msgs::msg::Float64;
using std_msgs::msg::Float64MultiArray;
using std_msgs::msg::String;
using std_srvs::srv::SetBool;
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
    // Servo armature structure. MUST equal the flight node's
    // wb_armature_joint_diag / wb_armature_j1..j4 (false = the flown link model).
    const bool armature_joint_diag = declare_parameter<bool>("armature_joint_diag", false);
    const auto armature = declare_parameter<std::vector<double>>(
      "armature", std::vector<double>{0.010, 0.0194, 0.0097, 0.0097});
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
    // figure-8 long axis on a WORLD azimuth (2026-10-05): true -> ee_traj_fig8_axis_deg
    // (0 = world x); false -> the legacy anchor, start tangent along the nose (long axis
    // ~45 deg off it for B = A/2).
    declare_parameter<bool>("ee_traj_fig8_world_axis", false);
    declare_parameter<double>("ee_traj_fig8_axis_deg", 0.0);
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
    // q2 sinusoid tied to the lap: > 0 sets ee_traj_q2_period_s = lap_time /
    // cycles on every plan, so a lap time edited from the ground station (its
    // Mean Velocity field) keeps the arm on whole cycles and the run ends on
    // its start rest. 0 = use ee_traj_q2_period_s as given.
    declare_parameter<int>("ee_traj_q2_cycles_per_lap", 0);
    // Rate bounds for the EE trajectory ONLY (> 0 overrides v_max / a_max /
    // w_max for this mode; 0 = inherit them). The shared bounds also size every
    // transition (go-to-start, back-to-origin, ground-station targets); a fast
    // figure-8 needs more CoM acceleration and yaw rate than those should get.
    declare_parameter<double>("ee_traj_v_max", 0.0);
    declare_parameter<double>("ee_traj_a_max", 0.0);
    declare_parameter<double>("ee_traj_w_max", 0.0);
    declare_parameter<double>("ee_traj_time_scale", 1.0);
    declare_parameter<double>("ee_traj_start_pos_tol", 0.05);
    declare_parameter<double>("ee_traj_start_yaw_tol_deg", 5.0);
    declare_parameter<double>("ee_traj_start_joint_tol_deg", 5.0);
    ee_time_scale_req_ = get_parameter("ee_traj_time_scale").as_double();

    // --- pick-and-place mode (pick_place.hpp) ------------------------------
    // Base poses [x, y, z, yaw_deg], WORLD frame, ACTUAL yaw (the drone GS's
    // convention). PLACEHOLDERS: Adjust WRITES its shift into all four (they
    // then read what the legs fly to; 2026-10-03), and any one can be measured instead from a mocap body through its
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
    // The PLACE pose (where the box CG is released, and its yaw -- the same
    // reference as the captured obj_0 pose, so ONE pick_place_ee_offset serves
    // both), typed in [x, y, z, yaw_deg], world [m] (yaw added 2026-10-03, user
    // decision; a 3-value [x, y, z] is still read, yaw 0) (2026-10-02,
    // user decision: entered on the ground station, not measured live). NOT
    // shifted by the Adjust offset (2026-10-03, user decision): it is read off
    // mocap before the flight -- the cap that carries the payload then covers
    // the mark -- so it is already in the mocap frame, like the captured pick
    // point. A mocap capture still wins when pick_place_place_topic names a
    // body and capture_place was called.
    declare_parameter<std::vector<double>>(
      "pick_place_place_point", std::vector<double>{0.6, -1.6, 0.66, 0.0});
    // The PICK pose as a parameter (2026-10-10, operator request after the
    // first hardware pick-and-place flights: the arm GS's Pick row became
    // editable boxes). It MIRRORS the pick capture both ways: [] = not
    // captured, else [x, y, z, yaw_deg], world [m] -- capture_pick writes its
    // result here, and a set from outside (the GS's boxes, a script) sets the
    // capture itself (onPpParams -> ppSetPickCapture); [] clears it. Like the
    // place point it is a mocap reading and is never shifted by Adjust.
    declare_parameter<std::vector<double>>("pick_place_pick_point", std::vector<double>{});
    {
      // a value from the launch file: the on-set callback is not registered yet
      const auto pk = get_parameter("pick_place_pick_point").as_double_array();
      std::string why;
      if (ppPickPointOk(pk, &why)) {
        ppSetPickCapture(pk);
        if (!pk.empty()) {
          RCLCPP_INFO(get_logger(), "pick-and-place: pick point from the parameters: "
            "[%.3f, %.3f, %.3f] m, yaw %.1f deg (not measured)", pk[0], pk[1], pk[2], pk[3]);
        }
      } else {
        RCLCPP_ERROR(get_logger(), "%s -- IGNORED, the pick point is not captured", why.c_str());
        set_parameter(rclcpp::Parameter("pick_place_pick_point", std::vector<double>{}));
      }
    }
    // THE EE offset [m], ONE for both claw legs (2026-10-03, user decision:
    // the same payload on the same pillars, so the claw goes to the same place
    // relative to the box at both): box CG -> grasp point in the captured
    // OBJECT's frame (obj_0 = the payload's bottom box, its CG and yaw). Pick:
    // claw = obj_0 + Rz(yaw_obj) * offset; place: claw = place point +
    // Rz(place yaw) * offset (PickPlaceConfig::ee_offset). Set by the arm
    // GS's "EE Offset" button. It REPLACES pick_place_pick_ee_offset /
    // pick_place_place_ee_offset: a file still setting those is refused below,
    // since an ignored offset would put the claw into the box.
    declare_parameter<std::vector<double>>("pick_place_ee_offset", std::vector<double>{0.0, 0.0, 0.0});
    {
      const auto & ov = get_node_parameters_interface()->get_parameter_overrides();
      for (const char * old : {"pick_place_pick_ee_offset", "pick_place_place_ee_offset"}) {
        if (ov.count(old) != 0) {
          throw std::runtime_error(
            std::string(old) + " was replaced by pick_place_ee_offset (one EE offset for the "
            "pick and the place, 2026-10-03): set that instead and delete " + old);
        }
      }
    }
    // the claw legs' drone yaw = the target pose's yaw (the captured object's
    // at the pick -- its handle aligned with the jaws --, the typed place
    // pose's at the place; 2026-10-03, user design). false: face_target.
    declare_parameter<bool>("pick_place_align_yaw", true);
    declare_parameter<bool>("pick_place_face_target", true);
    // before go_to_place_start / go_to_land_start: climb, then back off [m]
    declare_parameter<double>("pick_place_retreat_dz", 0.15);
    declare_parameter<double>("pick_place_retreat_back", 0.20);
    // execute_place: the arm sweeps these bands [deg] (lo == hi: not swept)
    // +-10 deg (2026-10-02; +-25 tripped the tilt watchdog in Isaac)
    declare_parameter<std::vector<double>>(
      "pick_place_sweep_lo_deg", std::vector<double>{-10.0, 17.5, 0.0, 0.0});
    declare_parameter<std::vector<double>>(
      "pick_place_sweep_hi_deg", std::vector<double>{10.0, 37.5, 0.0, 0.0});
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
    // mocap bodies (fsc_autopilot_ros2_msgs/Mocap, ABSOLUTE topics); "" = none.
    // Only the PICK point is measured by default (obj_0); every other point
    // is typed in. Fixed at launch: the subscriptions are made here.
    for (int k = 0; k < kNumPickPlaceLegs; ++k) {
      const char * dflt = k == kExecutePick ? "/obj_0/mocap" : "";
      pp_topic_[k] = declare_parameter<std::string>(
        std::string("pick_place_") + kPpPoint[k] + "_topic", dflt);
    }
    // 2026-10-02 audit. APPROACH: execute_pick / execute_place fly the claw to
    // this far above the target, hold, and descend vertically once the claw
    // has been within pick_place_arrival_tol for pick_place_settle_s; a wait
    // longer than pick_place_approach_wait_max [s] (0 = forever) ends the leg
    // incomplete, holding above.
    // THE SAFETY MARGIN (arm GS "Safety Margin", 2026-10-03: 0.10 -> 0.20 m):
    // also how far exit_pick / exit_place climb back up from the target.
    declare_parameter<double>("pick_place_approach_dz", 0.20);
    // Exit To Pick / Place duration [s]; 0 = the leg bounds (see PickPlaceConfig::exit_t)
    declare_parameter<double>("pick_place_exit_time_s", 0.0);
    // Trim first: a descent trim > pick_place_descent_trim_first_min is flown sideways at the hover height,
    // held this long [s], then straight down; 0 = in the descent itself
    declare_parameter<double>("pick_place_descent_trim_settle_s", 0.0);
    // ... for a horizontal trim above this [m]; 0 = trim first always, so the
    // descent's duration does not depend on the hover error (2026-10-09)
    declare_parameter<double>("pick_place_descent_trim_first_min", 0.005);
    // HOOK GRASP (2026-10-07, the box payload's hanger): > 0 = the pick
    // approaches from this far BEHIND the target at its own height and slides
    // in horizontally [m]; the place exit first backs off this far, then
    // climbs [m]. 0 = the vertical approach / exit (see PickPlaceConfig).
    declare_parameter<double>("pick_place_pick_approach_back", 0.0);
    declare_parameter<double>("pick_place_place_exit_back", 0.0);
    // DROP PLACE (2026-10-14, user design; see PickPlaceConfig::place_drop):
    // true = Ready To Place flies the claw straight to drop_height [m] above the
    // place claw target and the leg ends there; the arm GS's Place opens the
    // gripper (the payload drops into the container), Exit To Place backs off
    // and climbs. false = the set-down place (descent on request), unchanged.
    // drop_height = THE DROP MARGIN (2026-10-15, the arm GS's Pick & Drop
    // box): the place point is the payload standing on the container's cover
    // (Get), and the gripper opens with the payload's centre this far above it
    // [m]. 0 (default) = at the place point itself.
    declare_parameter<bool>("pick_place_place_drop", false);
    declare_parameter<double>("pick_place_drop_height", 0.0);
    // THE DROP PHASE AT ONE ALTITUDE + THE DROP EXIT (2026-10-14, user design;
    // see PickPlaceConfig::drop_altitude / drop_exit): the drop rest's BASE
    // altitude [m] (0 = the drop at the release point itself -- every yaml
    // since 2026-10-15: the drop margin sets the height), and Exit To
    // Place flying level out of the place region to the DROP EXIT ([] = the
    // back-off-and-climb exit). The exit is an OFFSET [dx, dy, yaw_deg] from the
    // place point's x, y in the room frame (2026-10-14, user: fixed relative to
    // the place pillar): it follows the place point -- the pillar, read off
    // mocap by Get -- and Adjust does NOT move it. yaw is the heading there;
    // the height is the drop's own + pick_place_drop_exit_climb.
    declare_parameter<double>("pick_place_drop_altitude", 0.0);
    // the fraction of Ready To Place over which a Place Start at another
    // altitude climbs (or descends) to it, in one smooth curve [0.2, 1]
    declare_parameter<double>("pick_place_drop_climb_lead", 0.5);
    declare_parameter<std::vector<double>>("pick_place_drop_exit_offset", std::vector<double>{});
    // ... and how far Exit To Place first climbs straight up from the drop point
    // before the level move [m] (user 2026-10-14: "climb up a little bit higher")
    declare_parameter<double>("pick_place_drop_exit_climb", 0.10);
    // EXIT STRAIGHT ON TO LAND START (2026-10-15, user design; see
    // PickPlaceConfig::drop_exit_to_land_start): with the drop place, Exit To
    // Place climbs pick_place_drop_exit_climb, then flies on to Land Start (the
    // arm folding home) and Go To Land Start is done. Overrides the drop exit.
    declare_parameter<bool>("pick_place_drop_exit_to_land_start", false);
    // THE BACK MARGIN (2026-10-15, user design; PickPlaceConfig::drop_exit_back):
    // with the drop place, Exit To Place backs out this far before it climbs
    // the Vertical Margin -- the arm GS's Pick & Drop "Back Margin" box. < 0 =
    // unset: it backs out pick_place_place_exit_back (the Side Margin).
    declare_parameter<double>("pick_place_drop_exit_back", -1.0);
    // THE AUTONOMOUS DROP (2026-10-16, user design after the 10-15 hardware
    // flights, where the Drop click came 0.67 s after the leg ended, with the
    // claw still ringing 46-48 mm off and the basket swinging on the wrist).
    // With the drop place, once Ready To Place has ended on the drop point the
    // planner releases the payload BY ITSELF when, for drop_auto_dwell_s, ALL of
    // these hold (publishPpArrival, 10 Hz):
    //   - at least drop_auto_wait_s since the leg ended (the arrival swing);
    //   - the claw within drop_auto_claw_tol of its drop target and moving
    //     slower than drop_auto_claw_speed [m/s];
    //   - the payload's OWN mocap body (pick_place_pick_topic, obj_0) tracked
    //     (fresh, not frozen, not a 180 deg flipped solve), horizontally within
    //     drop_auto_payload_tol of the drop point (pick_place_place_point x, y)
    //     and moving slower than drop_auto_payload_speed [m/s];
    //   - the wrist roll q4 within drop_auto_q4_tol_deg of the place pose's q4,
    //     so the payload hangs level.
    // It then publishes pick_place/drop_now (once per drop hold); the arm GS's
    // Pick & Drop tab opens the gripper on it exactly as its Drop button does
    // (the button stays the manual override). It never releases on a timer: a
    // gate that does not pass keeps it holding, and the status names the gate.
    // false (node default) = the Drop button only.
    declare_parameter<bool>("pick_place_drop_auto", false);
    declare_parameter<double>("pick_place_drop_auto_wait_s", 2.0);
    declare_parameter<double>("pick_place_drop_auto_dwell_s", 0.5);
    declare_parameter<double>("pick_place_drop_auto_claw_tol", 0.020);
    declare_parameter<double>("pick_place_drop_auto_claw_speed", 0.03);
    declare_parameter<double>("pick_place_drop_auto_payload_tol", 0.025);
    declare_parameter<double>("pick_place_drop_auto_payload_speed", 0.04);
    declare_parameter<double>("pick_place_drop_auto_q4_tol_deg", 3.0);
    // AIM THE DROP AT THE BASKET (2026-10-16, user decision after the 10-15
    // replay). The drop's claw target is place point + Rz(yaw)*ee_offset: a
    // MODEL of where the basket sits relative to the claw. On 10-15 D-1 the
    // basket hung 17 mm further back and 17 mm further right than that (the
    // hang, and the jaws closing ~20 mm off obj_0's origin), so with the claw on
    // target the basket was ~24 mm off the drop point. With this set, the
    // basket - claw offset is MEASURED (obj_0 - FK claw, in the vehicle frame)
    // while hovering at Drop Start with the arm at the drop pose, averaged over
    // the last _window_s, and Ready To Drop shifts the drop rest so that the
    // measured basket lands on the drop point. Refused (the model is used, and
    // the leg says why) with under _min_s of samples or a shift above
    // _max_shift. Sampling starts _settle_s after the hover began (the turn to
    // Drop Start excites the wrist swing: 10-15 D-1 broke loose at its end).
    // The window spans two periods of the ~0.5 Hz basket swing seen on 10-15
    // (+-40 mm), the minimum one: a part-period average of that swing is off by
    // up to ~10 mm, so HOVER >= _settle_s + _window_s (6 s) AT DROP START before
    // Ready To Drop (under _settle_s + _min_s, 4 s, it is not aimed). When
    // applied, the autonomous drop gates position on the basket only (the claw
    // keeps its stillness gate). Drop only (pick_place_place_drop); false
    // (default) = the model, as before. All read at Ready To Drop: setting one
    // live leaves the plan valid.
    declare_parameter<bool>("pick_place_drop_aim_payload", false);
    declare_parameter<double>("pick_place_drop_aim_window_s", 4.0);
    declare_parameter<double>("pick_place_drop_aim_min_s", 2.0);
    declare_parameter<double>("pick_place_drop_aim_max_shift", 0.06);
    declare_parameter<double>("pick_place_drop_aim_settle_s", 2.0);
    declare_parameter<double>("pick_place_approach_wait_max", 60.0);
    // true (2026-10-03, user design: Ready -> Pick -> Exit): a claw leg ends
    // HOLDING above its target; the descent waits for descend_pick /
    // descend_place (the arm GS's Pick / Place button), refused unless the
    // claw is within pick_place_arrival_tol of the point above the target,
    // and the hold above never times out. false: the descent starts by
    // itself once the claw has dwelt within the tolerance (2026-10-02).
    declare_parameter<bool>("pick_place_descent_on_request", true);
    // DESCENT TRIM (2026-10-03): a controller can hold the claw a steady few
    // cm off its reference -- the decoupled geometric + L1 law parks at
    // (unmatched force) / Kp, ~37 mm on the mirror plant, against +-8.75 mm of
    // jaw clearance on the handle. While holding above the target the planner
    // averages the claw's error over the last window, and the descent's goal
    // is shifted by it, so the MEASURED claw lands on the target. Refused (on
    // request) above trim_max: that much error means something else is wrong.
    declare_parameter<bool>("pick_place_descent_trim", true);
    declare_parameter<double>("pick_place_descent_trim_window_s", 1.0);
    // 0.06 -> 0.10 m (2026-10-03): the decoupled law's steady hover offset
    // varied 36..79 mm between two identical flights (geo_3 / geo_4)
    declare_parameter<double>("pick_place_descent_trim_max", 0.10);
    declare_parameter<double>("pick_place_settle_s", 1.0);
    // PHASE-DEPENDENT EE ANCHOR (2026-10-02, Isaac pick_place_tune_20261001):
    // hold the claw in the WORLD from the moment a claw leg reaches the point
    // above its target -- the wait, the descent and the grasp that follows --
    // and return to the law's own (CoM) anchor on the next request (any leg,
    // Abort, Reset, a wait timeout, teleop, any other motion). Per leg: the
    // PICK needs it (the open jaws have millimetres to spare around the
    // handle, and nothing touches until they close); the PLACE should not
    // (setting a payload down is a rigid contact a world-held claw fights).
    // Between the clamp and the next request the vehicle is coupled to a
    // payload resting on its table: LIFT PROMPTLY -- under either anchor that
    // closed chain drifts within ~1 s. Calls the whole-body node's
    // set_ee_anchor_com service; false = never call it. Needs
    // pick_place_approach_dz > 0 (the wait is where it switches).
    declare_parameter<bool>("pick_place_world_anchor_pick", false);
    declare_parameter<bool>("pick_place_world_anchor_place", false);
    declare_parameter<std::string>(
      "pick_place_anchor_service", "fsc_autopilot_ros2/whole_body_direct_actuation/set_ee_anchor_com");
    // a leg starts only when the measured body is within this of its hold [m]
    declare_parameter<double>("pick_place_leg_start_tol", 0.15);
    // Adjust refuses an offset larger than this [m] (pressed off the start mark)
    declare_parameter<double>("pick_place_adjust_max", 0.5);
    // GEOFENCE on every goal, approach, retreat and leg path (body origin,
    // world [m]): the body stands at 0.305 m on its gear and the law swings
    // ~+-0.1 m, hence the 0.6 m floor; the abort climb is clipped at the ceiling
    declare_parameter<double>("pick_place_fence_min_z", 0.6);
    declare_parameter<double>("pick_place_fence_max_z", 1.8);
    declare_parameter<std::vector<double>>(
      "pick_place_fence_xy", std::vector<double>{-2.5, 2.5, -2.5, 2.5});
    // ABORT (pick_place/abort): arm home and the vehicle this much higher [m],
    // x, y and heading kept -- the arm GS opens the gripper on the same press
    declare_parameter<double>("pick_place_abort_climb", 0.30);
    // ... and the TWO-STEP ARM (2026-10-09, user design): the arm first goes to
    // this RELEASE pose [deg] while the vehicle climbs, holds it
    // pick_place_abort_release_hold_s, then folds home in place (the arm GS
    // then closes the gripper). Claws pointing steeply down let a payload still
    // hanging on the open fingers -- the hook grasp's arch -- slide off. Empty
    // (the default) = straight home, the earlier behaviour.
    declare_parameter<std::vector<double>>("pick_place_abort_release_pose_deg", std::vector<double>{});
    declare_parameter<double>("pick_place_abort_release_hold_s", 2.0);
    {
      const auto rp = get_parameter("pick_place_abort_release_pose_deg").as_double_array();
      if (rp.size() == 4) {
        RCLCPP_INFO(get_logger(), "pick-and-place ABORT: arm to the release pose [%.0f, %.0f, %.0f, %.0f] deg "
          "while climbing, hold %.1f s, then home", rp[0], rp[1], rp[2], rp[3],
          get_parameter("pick_place_abort_release_hold_s").as_double());
      } else if (!rp.empty()) {
        RCLCPP_ERROR(get_logger(), "pick_place_abort_release_pose_deg has %zu values (needs 4) -- IGNORED, "
          "the abort folds straight home", rp.size());
      }
    }
    // SAFETY GUARD (2026-10-03, user design): an AUTOMATIC abort -- the same
    // response as the button: the arm GS opens the gripper (it sees the
    // ABORTING status), arm home, this climb, hover -- when the body TILT
    // (from PX4's attitude, which a mocap position flip does not move) stays
    // above this [deg] for pick_place_guard_persist_s, in whole-body DIRECT
    // once a pick-and-place leg has been flown. 15 deg: ~1.7x the worst
    // nominal tilt measured (8.7 deg carrying in the 2026-10-02 Isaac runs;
    // hardware whole-body flights <= 5.3 deg), below the whole-body node's own
    // 20 deg tilt watchdog, which hands over to PX4. Re-arms below it - 5 deg.
    // 0 disables.
    declare_parameter<double>("pick_place_guard_tilt_deg", 15.0);
    declare_parameter<double>("pick_place_guard_persist_s", 0.10);
    // a capture averages the samples of the last this-many seconds, and is
    // REFUSED when they scatter more than this: two publishers on one topic
    // (the Isaac emulator's phantom obj_0 at the origin did exactly that on
    // the first sim flight, 2026-10-01: 856 mm, averaged and flown) or a
    // moving / mis-tracked body must never become a goal [m]
    declare_parameter<double>("pick_place_capture_window", 0.5);
    declare_parameter<double>("pick_place_capture_max_spread", 0.02);
    // ... and (the pick / place bodies) when the body's YAW samples in the
    // window disagree by more than this [deg] (<= 0 disables). 2026-10-10,
    // from the first hardware flights:
    // obj_0's solved orientation flipped by ~180 deg several times a second
    // (a near-symmetric marker layout has two solutions), and the circular
    // mean of a flipping yaw is garbage -- the frame the EE offset is applied
    // in. A capture is also refused when every position sample in the window
    // is bit-identical: a mocap processor republishing the FROZEN last pose of
    // an untracked body with fresh stamps (11 423 identical obj_0 samples in
    // one flight). A real tracked body always jitters; a simulated one must
    // carry some position noise (the Isaac emulator's position_noise_std,
    // 0.5 mm in the pick-and-place sim yamls) or a resting payload is refused.
    declare_parameter<double>("pick_place_capture_max_yaw_spread_deg", 10.0);

    // --- push-and-pull mode (push_pull.hpp) --------------------------------
    // The box: a mocap body (CG + yaw), captured by capture_box (the arm GS's
    // Get); fixed at launch. The grasp point = the box + push_pull_ee_offset
    // in the box's frame; the drone's ACTUAL yaw there = the box's yaw +
    // push_pull_yaw_from_box_deg (-90: the handle on the box's +y side, the
    // push along the box's -y).
    declare_parameter<std::string>("push_pull_box_topic", "/obj_0/mocap");
    // (2026-10-16) the box capture's PARAMETER TWIN -- the arm GS's Box row
    // became editable boxes, as the Pick row did on 2026-10-10. It MIRRORS the
    // capture both ways: [] = not captured, else [x, y, z, yaw_deg], world [m]
    // -- capture_box writes its result here, and a set from outside (the GS's
    // boxes, a script) sets the capture itself (onPlParams -> plSetBoxCapture);
    // [] clears it. A mocap reading: never shifted by Adjust.
    declare_parameter<std::vector<double>>("push_pull_box_point", std::vector<double>{});
    {
      // a value from the launch file: the on-set callback is not registered yet
      const auto bp = get_parameter("push_pull_box_point").as_double_array();
      std::string why;
      if (plBoxPointOk(bp, &why)) {
        plSetBoxCapture(bp);
        if (!bp.empty()) {
          RCLCPP_INFO(get_logger(), "push-and-pull: box from the parameters: "
            "[%.3f, %.3f, %.3f] m, yaw %.1f deg (not measured)", bp[0], bp[1], bp[2], bp[3]);
        }
      } else {
        RCLCPP_ERROR(get_logger(), "%s -- IGNORED, the box is not captured", why.c_str());
        set_parameter(rclcpp::Parameter("push_pull_box_point", std::vector<double>{}));
      }
    }
    declare_parameter<std::vector<double>>("push_pull_ee_offset", std::vector<double>{0.0, 0.0, 0.0});
    declare_parameter<double>("push_pull_yaw_from_box_deg", -90.0);
    // the arm in the push: claw pitched 10 deg forward, 0.130 m ahead of and
    // 0.316 m under the body origin, sigma_nd 0.298 (the pick pose)
    declare_parameter<std::vector<double>>(
      "push_pull_push_pose_deg", std::vector<double>{0.0, -20.0, 30.0, 0.0});
    // THE SAFETY MARGIN: go_to_start holds the claw this far above the grasp
    // point, ready descends it, exit climbs it [m]
    declare_parameter<double>("push_pull_approach_dz", 0.20);
    // (2026-10-06) the approach also starts this far BEHIND the grasp along the
    // nose [m] (a forward-pitched claw slides onto a vertical post); 0 = straight
    // down, unchanged. The exit climb [m]; 0 = push_pull_approach_dz.
    declare_parameter<double>("push_pull_approach_back", 0.0);
    declare_parameter<double>("push_pull_exit_dz", 0.0);
    // (2026-10-06) WHEN the law's contact phase flag switches. Default = the
    // 2026-10-04 timing: CONTACT on at the end of Ready (claw on the handle, jaws
    // still open), off at Exit (after the jaws opened). The gripper timing:
    // contact_at_ready false -> CONTACT on at the Push press, which the GS sends
    // on the jaws' stall (the box joins the chain); contact_off_at_push_end true
    // -> free as soon as the push completes, before the GS opens the jaws.
    declare_parameter<bool>("push_pull_contact_at_ready", true);
    declare_parameter<bool>("push_pull_contact_off_at_push_end", false);
    // the push: this far along the nose [m] (< 0 pulls), no faster than this
    // [s], after holding the grasp this long [s] (the law's phase flag and the
    // EE anchor switch at the press; the anchor blends over the node's
    // wb_ee_anchor_blend_s, 1.0 s)
    declare_parameter<double>("push_pull_push_distance", 0.50);
    declare_parameter<double>("push_pull_push_time_s", 12.0);
    declare_parameter<double>("push_pull_push_settle_s", 1.5);
    // THE CRUISE (2026-10-17, PushPullConfig::push_ramp): > 0 = the straight
    // push speeds up over this many seconds, holds a constant speed
    // push_pull_push_distance / (push_pull_push_time_s - this), and slows down
    // over the same time; must be below push_pull_push_time_s / 2. 0 = the
    // min-snap move (unchanged: its speed peaks at 2.19 x the mean). A turn
    // ignores it.
    declare_parameter<double>("push_pull_push_ramp_s", 0.0);
    declare_parameter<double>("push_pull_exit_time_s", 3.0);
    // THE UNLOAD (2026-10-07, see PushPullConfig::unload_time): > 0 = after the
    // slide the push holds push_pull_unload_wait_s, the hold is then re-seated
    // on the claw's measured position -- the whole-body law's own task error
    // e_y (wb_control_debug [24..26], world) averaged over the last
    // push_pull_unload_avg_s of that hold -- over this many seconds, still
    // gripping and in CONTACT, and held unload_wait again. An error above
    // push_pull_unload_max is not applied (warned). 0 = off (unchanged).
    declare_parameter<double>("push_pull_unload_time_s", 0.0);
    declare_parameter<double>("push_pull_unload_wait_s", 1.0);
    declare_parameter<double>("push_pull_unload_avg_s", 0.5);
    declare_parameter<double>("push_pull_unload_max", 0.03);
    // THE TURN (2026-10-15, the valve-turning task; PushPullConfig::turn_angle):
    // non-zero = the push step ROTATES the claw, the base and the heading
    // together about the captured body's vertical axis (the "box" is the
    // valve's bearing axis + the beam's yaw) by this many degrees (+ =
    // counter-clockwise from above) over push_pull_push_time_s, the arm held;
    // push_pull_push_distance is then ignored. The nose must be along the
    // turn's tangent at the handle (yaw_from_box +90 for a + turn, -90 for -).
    // 0 = the straight push (unchanged).
    declare_parameter<double>("push_pull_turn_deg", 0.0);
    // the land hover [x, y, z, yaw_deg] (world, ACTUAL yaw, flown as typed:
    // from the push heading -90, -360 is 270 deg CLOCKWISE) and the start mark
    // Adjust measures against; Adjust writes its x, y shift into both
    declare_parameter<std::vector<double>>("push_pull_land", std::vector<double>{-0.5, 0.0, 0.8, -360.0});
    // (2026-10-16) LAND START [x, y, z, yaw_deg] (world, ACTUAL yaw, flown as
    // typed like the land's): go_to_land_start flies ONE transition from the
    // exit hold to it (keep it high, clear of the table); land then flies
    // straight down to push_pull_land. [] = the same pose as push_pull_land.
    // Adjust shifts it too.
    declare_parameter<std::vector<double>>("push_pull_land_start", std::vector<double>{});
    declare_parameter<std::vector<double>>("push_pull_start_mark", std::vector<double>{0.0, 0.0, 1.0, 0.0});
    declare_parameter<double>("push_pull_adjust_max", 0.5);
    // THE TABLE [x_min, x_max, y_min, y_max, z_top] (world, axis-aligned; all 0
    // = no table check): wherever the landing gear (push_pull_gear_depth under
    // the body origin, within push_pull_gear_radius of it) or a point of the
    // arm is over the top (+ push_pull_table_margin_xy), it must stay
    // push_pull_table_clearance above it
    declare_parameter<std::vector<double>>(
      "push_pull_table", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});
    // A SECOND keep-out with the table's rule (2026-10-15): the valve's swept
    // circle, whose BEAM top stands above the table it is bolted to. Same
    // layout, all 0 = none.
    declare_parameter<std::vector<double>>(
      "push_pull_obstacle", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});
    declare_parameter<double>("push_pull_table_clearance", 0.05);
    declare_parameter<double>("push_pull_table_margin_xy", 0.05);
    declare_parameter<double>("push_pull_gear_depth", 0.313);
    declare_parameter<double>("push_pull_gear_radius", 0.25);
    // THE BOX STAYS ON THE TABLE (2026-10-16): > 0 = the box's half length
    // along the push [m]; a plan whose pushed box would reach within
    // push_pull_box_edge_margin of the table's edge (push_pull_table) is
    // refused -- a box or mocap body turned 180 deg, or too long a distance.
    // 0 = off (unchanged).
    declare_parameter<double>("push_pull_box_half_length", 0.0);
    declare_parameter<double>("push_pull_box_edge_margin", 0.03);
    declare_parameter<double>("push_pull_fence_min_z", 0.6);
    declare_parameter<double>("push_pull_fence_max_z", 1.8);
    declare_parameter<std::vector<double>>(
      "push_pull_fence_xy", std::vector<double>{-2.5, 2.5, -2.5, 2.5});
    // Ready's gate: the claw within this of the point above the handle [m]
    // (less the descent trim, as the pick's), the trim no larger than max
    declare_parameter<double>("push_pull_arrival_tol", 0.05);
    declare_parameter<bool>("push_pull_descent_trim", true);
    declare_parameter<double>("push_pull_descent_trim_window_s", 1.0);
    declare_parameter<double>("push_pull_descent_trim_max", 0.10);
    declare_parameter<double>("push_pull_leg_start_tol", 0.15);
    // the claw held in the WORLD from the hover above the handle through the
    // grasp (the pick's anchor, through pick_place_anchor_service)
    declare_parameter<bool>("push_pull_world_anchor_ready", true);
    // (2026-10-07) with push_pull_world_anchor_ready FALSE (the hover above
    // and the Ready descent CoM-anchored): true = switch the claw to WORLD-held
    // on arrival at the handle, before the jaws close. World-held in free
    // flight the claw's vertical loop bobbed (height error 6-30 mm p-p hovering,
    // 20-57 mm in the descent; one flip); CoM-anchored it is calm but follows
    // the base's drift (one run 14 mm off-centre across a 5 mm fin, never
    // gripped). This keeps the free-flight world hold to the grasp alone.
    // Pair it with push_pull_descent_trim false: a trim measured CoM-anchored
    // is the base's wander, not the world-held claw's steady error.
    declare_parameter<bool>("push_pull_world_anchor_on_handle", false);
    // ... and through the PUSH (2026-10-04, after the first three flights):
    // true = the claw follows the PLANNED world path while the box is
    // clamped. With the CoM anchor a base offset drags the claw reference,
    // loads the arm against the stuck box, and the platform observers then
    // push the vehicle through the box's static friction until it breaks
    // away (pl_1..3: a growing 0.3 Hz stick-slip cycle, box yaw to +-90 deg).
    // World-held, base drift moves no reference: the arm's impedance against
    // the planned path is what moves the box (pl_6..8: 3 / 3 pushes of
    // 495..502 mm). false = the CoM anchor at the Push press (the first
    // design).
    declare_parameter<bool>("push_pull_world_anchor_push", true);
    // THE INTERACTION PHASE: the whole-body node's contact phase flag, CONTACT
    // from the claw's arrival on the handle (end of Ready; the jaws close
    // next) to the Exit / Abort press; false = never call it
    declare_parameter<bool>("push_pull_contact", true);
    declare_parameter<std::string>(
      "push_pull_contact_service", "fsc_autopilot_ros2/whole_body_direct_actuation/set_contact");
    // ABORT: hold this long when the claw is on the handle (the gripper
    // opens), climb this far, arm home
    declare_parameter<double>("push_pull_abort_climb", 0.30);
    declare_parameter<double>("push_pull_abort_release_wait_s", 0.5);
    // THE ABORT ON THE BOX (2026-10-16, archive/box_pushing_campaign_20261016): with the
    // claw on the box, from_measured holds the MEASURED pose (odometry, PX4 heading,
    // encoders) instead of the reference the push was cut at -- a jammed box leaves the
    // planned claw 40+ mm AHEAD of the real one, and holding the cut reference drags the
    // claw along the fin during the release and wedges it in the open jaw; abort_back
    // [m] retreats the base that far against the push after the release wait, before
    // the climb, so the fingers leave the fin horizontally. Isaac: without both, the
    // open claw climbing straight up LIFTED a jammed 1 kg box 58 mm and the vehicle
    // flipped when it dropped off. false / 0 = the cut-reference hold and the straight
    // climb (unchanged).
    declare_parameter<bool>("push_pull_abort_from_measured", false);
    declare_parameter<double>("push_pull_abort_back", 0.0);
    // THE ABORT ON A FREE HANDLE (2026-10-18, archive/valve_turning_20261015 README 14):
    // true = when the claw is on the handle, KEEP the world-held claw through the release
    // wait (the jaws open around a claw pinned in the room) and hand it back to the CoM
    // anchor only when the wait ends -- the valve's exit lesson (8.9): the open jaws riding
    // the base's drift into a fin that cannot move radially roll the wrist 30-45 deg. On a
    // TURN the retreat (push_pull_abort_back) is along the CURRENT nose, not the grasp's
    // (push_dir is the tangent at the grasp: mid-turn it points sideways or into the fin).
    // false = the anchor goes back to the CoM at once (unchanged).
    declare_parameter<bool>("push_pull_abort_release_world_held", false);
    // SAFETY GUARDS -> ABORT: the tilt (PX4 attitude) above this for that
    // long; the task force the whole-body node reads (wb_control_debug,
    // low-passed) above this for that long while in contact. 0 disables.
    declare_parameter<double>("push_pull_guard_tilt_deg", 15.0);
    declare_parameter<double>("push_pull_guard_persist_s", 0.10);
    // THE BOX GUARDS (2026-10-16, the box-pushing campaign's worst cases,
    // archive/box_pushing_campaign_20261016): a jammed or too-heavy box makes
    // the arm BUCKLE -- the base keeps flying its push while the clamped claw
    // stops, q2 runs onto its +45 stop and q3 through 0 into the elbow-singular
    // branch in under a second -- long before the force guard sees 5 N (Isaac:
    // the force tripped at 16.3 N, after the buckle, and the vehicle flipped).
    // The box itself shows it a second earlier: from the box's mocap body,
    //  - box_lag: |the box's displacement along the push - the planned claw's|
    //    above this [m] (nominal 0..25 mm behind, never > 2 mm ahead);
    //  - box_tilt_deg: the box tilted from its pose at the push start above this
    //    [deg] -- tipping instead of sliding (nominal <= 0.7 deg);
    // each for box_persist_s, while the push flies (lag) / while the claw is on
    // the handle or pushing (tilt). box_stale_s: no box sample for this long
    // while those are armed [s] -- the two are blind without it.
    // q3_min_deg (> 0 enables): q3 below this while on the handle / pushing --
    // the buckle itself, whatever caused it (nominal >= 19 deg).
    // All 0 = off (unchanged).
    declare_parameter<double>("push_pull_guard_box_lag", 0.0);
    declare_parameter<double>("push_pull_guard_box_tilt_deg", 0.0);
    declare_parameter<double>("push_pull_guard_box_persist_s", 0.20);
    declare_parameter<double>("push_pull_guard_box_stale_s", 0.50);
    declare_parameter<double>("push_pull_guard_q3_min_deg", 0.0);
    declare_parameter<double>("push_pull_guard_force_n", 5.0);
    declare_parameter<double>("push_pull_guard_force_persist_s", 0.30);
    // PATH FOLLOWING FOR THE TURN (2026-10-18; push_pull_turn_deg != 0 only):
    // true = the turn is flown at a GOVERNED PROGRESS (ProgressGovernor) in
    // place of the wall clock -- the same rigid rotation, its schedule slowed or
    // held while the measurements say so. Every 100 Hz stream tick the rate
    // target is the smallest of three channel factors, each 1 at or below its
    // [soft, hard][0], 0 at or above [1], linear between:
    //   lead  [deg]: the PLANNED rotation of the claw about the axis minus the
    //         MEASURED valve's (the box body's yaw) -- the reference never runs
    //         more than `hard` ahead of the real handle (a stuck or heavy
    //         valve: the turn waits instead of loading the arm);
    //   tilt  [deg]: the vehicle's tilt (PX4 attitude);
    //   wrist [deg]: |q4 - q4 planned|, the wrist roll.
    // Isaac 2026-10-18: in all six windowed failures the valve tracked the plan
    // to <= 6 deg up to the trip, while tilt > 7 deg and |q4| > 15 deg came 2-7 s
    // before it; the completed headless turns never exceed 6.4 / 14.3.
    // A channel whose feed is older than stale_s holds (factor 0); [] or
    // hard <= soft turns a channel off. bandwidth [rad/s]: the progress
    // filter (4 stages); recover_s: the rate climbs 0 -> 1 no faster than
    // this; hold_timeout_s > 0: a hold (rate < 0.02) that long ABORTS (the
    // push-and-pull abort: release, climb, fold). Read when the push starts.
    // false = the scheduled turn (unchanged).
    declare_parameter<bool>("push_pull_follow", false);
    // filter_s: every channel is low-passed (first order) before its factor
    // -- the contact phase rings at ~1 Hz (the airframe's attitude mode; the
    // lead jitters +-4 deg, the arm swings between its stops with it), and a
    // governor chasing that ring would modulate the turn at the ring's
    // frequency (pf_n1, 2026-10-18); the governor answers TRENDS: a valve
    // falling behind, a growing tilt envelope, a wrist walking away.
    declare_parameter<double>("push_pull_follow_filter_s", 1.0);
    declare_parameter<std::vector<double>>("push_pull_follow_lead_deg", {6.0, 12.0});
    declare_parameter<std::vector<double>>("push_pull_follow_tilt_deg", {7.0, 10.0});
    declare_parameter<std::vector<double>>("push_pull_follow_wrist_deg", {15.0, 25.0});
    declare_parameter<double>("push_pull_follow_stale_s", 0.3);
    declare_parameter<double>("push_pull_follow_bandwidth", 4.0);
    declare_parameter<double>("push_pull_follow_recover_s", 5.0);
    declare_parameter<double>("push_pull_follow_hold_timeout_s", 0.0);
    declare_parameter<std::string>(
      "push_pull_force_topic", "fsc_autopilot_ros2/whole_body_direct_actuation/wb_control_debug");
    declare_parameter<double>("push_pull_capture_window", 0.5);
    declare_parameter<double>("push_pull_capture_max_spread", 0.02);
    // (2026-10-19, the hardware box push) the box capture also refuses a FROZEN
    // feed -- every position in the window bit-identical: the mocap processor
    // republishing the last pose of a body it no longer tracks, with fresh
    // stamps (10-09: 11 423 identical obj_0 samples in one flight) -- and a yaw
    // varying by more than this [deg] over the window (an orientation flipping
    // between two solutions; the push follows the box's yaw). <= 0: no yaw check.
    declare_parameter<double>("push_pull_capture_max_yaw_spread_deg", 10.0);

    // --- PS4 teleoperation (2026-09-27) -----------------------------------
    // Rates are at FULL deflection and PHYSICAL; teleop_time_scale converts
    // them to this node's (wall) clock -- 1.0 on hardware, the sim's real-
    // time factor in Isaac (the same rule as ee_traj_time_scale).
    declare_parameter<std::string>("teleop_joy_topic", "rc/input");
    declare_parameter<double>("teleop_joy_timeout_s", 0.3);
    declare_parameter<double>("teleop_time_scale", 1.0);
    declare_parameter<double>("teleop_com_speed_xy", 0.20);
    declare_parameter<double>("teleop_com_speed_z", 0.15);
    declare_parameter<double>("teleop_yaw_rate_deg", 20.0);
    declare_parameter<double>("teleop_ee_speed", 0.04);
    declare_parameter<double>("teleop_roll_rate_deg", 20.0);
    declare_parameter<double>("teleop_home_rate_deg", 10.0);
    declare_parameter<double>("teleop_com_bandwidth", 6.0);
    declare_parameter<double>("teleop_yaw_bandwidth", 4.0);
    declare_parameter<double>("teleop_arm_bandwidth", 6.0);
    declare_parameter<double>("teleop_box_half_xy", 1.5);
    declare_parameter<double>("teleop_com_z_min", 0.70);
    declare_parameter<double>("teleop_com_z_max", 2.20);
    declare_parameter<double>("teleop_ee_z_min", 0.25);
    declare_parameter<double>("teleop_leash_m", 0.50);
    // each joint's range shrunk about its centre to this fraction (0.8 = 20 %
    // reserve) -- the pad can never drive the arm onto a hardware stop
    declare_parameter<double>("teleop_joint_range_frac", 0.8);
    // where PS folds the arm: the PAD'S home, inside that box [deg]
    declare_parameter<std::vector<double>>(
      "teleop_home_pose_deg", std::vector<double>{0.0, 30.0, 30.0, 0.0});
    declare_parameter<double>("teleop_q3_min_deg", 0.0);
    declare_parameter<double>("teleop_stick_deadzone", 0.08);
    // Joy AXES: [ee forward, ee left, ee up, wrist roll, D-pad x, D-pad y]
    // (DualShock 4 over joy_node: left stick 0/1, right stick 3/4, D-pad
    // 6/7; LEFT and UP read +1 on all of them). Signs multiply each.
    teleop_axes_ = declare_parameter<std::vector<int64_t>>(
      "teleop_axes", std::vector<int64_t>{1, 0, 4, 3, 6, 7});
    teleop_axis_signs_ = declare_parameter<std::vector<double>>(
      "teleop_axis_signs", std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    // Joy BUTTONS: [up, down, yaw left, yaw right, arm home] =
    // triangle, cross, square, circle, PS.
    teleop_buttons_ = declare_parameter<std::vector<int64_t>>(
      "teleop_buttons", std::vector<int64_t>{2, 0, 3, 1, 10});
    if (teleop_axes_.size() != 6 || teleop_axis_signs_.size() != 6 || teleop_buttons_.size() != 5) {
      throw std::runtime_error("teleop_axes/teleop_axis_signs need 6 values, teleop_buttons 5");
    }

    if (home.size() != kNumJoints || base_com.size() != 3 || sign.size() != kNumJoints ||
      armature.size() != kNumJoints)
    {
      throw std::runtime_error("home_pose/arm_joint_sign/armature need 4 values, base_com 3");
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
    vo.armature_joint_diag = armature_joint_diag;
    for (int j = 0; j < kNumJoints; ++j) {
      if (!(armature[j] >= 0.0)) {throw std::runtime_error("armature must be >= 0");}
      vo.armature(j) = armature[j];
    }
    vehicle_ = makeVehicleModel(vehicle_name_, vo);
    planner_ = makePlanner(planner_name_);
    RCLCPP_INFO(
      get_logger(), "vehicle: %s (whole-body model total %.6f kg, base_com "
      "[%+.5f %+.5f %+.5f] m model frame -- must match the node's wb_base_com_*)",
      vehicle_->name.c_str(), vehicle_->params.totalMass(), vo.base_com(0),
      vo.base_com(1), vo.base_com(2));
    if (armature_joint_diag) {
      RCLCPP_INFO(
        get_logger(), "armature: joint-diagonal [%.4f %.4f %.4f %.4f] kg m^2 -- must match "
        "the node's wb_armature_j1..j4", armature[0], armature[1], armature[2], armature[3]);
    } else {
      RCLCPP_INFO(get_logger(), "armature: link structure (the flown model)");
    }
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
    // yaw an ee_target must carry to leave the wrist where it is. current_ee's
    // orientation is the EE frame, not this convention, and a station seeding a
    // target without this asks for a wrist swing nobody commanded (the PS4
    // Remote tab did).
    cur_ee_heading_pub_ = create_publisher<Float64>(prefix + "/current_ee_heading", 10);
    // ...and the MEASURED airframe that current_ee was computed from (same
    // sample, same gate), so a station draws the two as one state.
    cur_base_pub_ = create_publisher<PoseStamped>(prefix + "/current_base", 10);
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
    // the autonomous drop's release signal (pick_place_drop_auto): once per drop
    // hold, the gate values as text; the arm GS's Pick & Drop tab opens the gripper
    pp_drop_now_pub_ = create_publisher<String>(prefix + "/pick_place/drop_now", 10);
    const auto trigger = [this](const std::string & name, std::function<void(Trigger::Response::SharedPtr)> fn) {
        pp_srvs_.push_back(create_service<Trigger>(
          name, [fn](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {fn(r);}));
      };
    trigger(prefix + "/pick_place/adjust", [this](Trigger::Response::SharedPtr r) {onPpAdjust(r);});
    trigger(prefix + "/pick_place/plan", [this](Trigger::Response::SharedPtr r) {onPpPlan(r);});
    trigger(prefix + "/pick_place/reset", [this](Trigger::Response::SharedPtr r) {onPpReset(r);});
    trigger(prefix + "/pick_place/abort", [this](Trigger::Response::SharedPtr r) {
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          pp_guard_tripped_ = false;    // the operator's abort, not the guard's
        }
        onPpAbort(r);
      });
    // the claw legs' EXIT: straight back up the safety margin from the target
    // (2026-10-03, user design: Ready -> gripper -> Exit)
    // ... and their DESCENT on request (pick_place_descent_on_request)
    trigger(prefix + "/pick_place/descend_pick",
      [this](Trigger::Response::SharedPtr r) {onPpDescend(kExecutePick, r);});
    trigger(prefix + "/pick_place/descend_place",
      [this](Trigger::Response::SharedPtr r) {onPpDescend(kExecutePlace, r);});
    trigger(prefix + "/pick_place/exit_pick",
      [this](Trigger::Response::SharedPtr r) {onPpExit(kExecutePick, r);});
    trigger(prefix + "/pick_place/exit_place",
      [this](Trigger::Response::SharedPtr r) {onPpExit(kExecutePlace, r);});
    // the PLACE point measured with the payload itself (2026-10-10): the PICK
    // body's pose -> pick_place_place_point x, y, z (the arm GS's Place-row Get)
    trigger(prefix + "/pick_place/get_place", [this](Trigger::Response::SharedPtr r) {onPpGetPlace(r);});
    anchor_cli_ = create_client<SetBool>(get_parameter("pick_place_anchor_service").as_string());
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

    // --- push-and-pull mode (arm GS "Push & Pull" tab) ---------------------
    pl_status_pub_ = create_publisher<String>(prefix + "/push_pull/status", latched);
    pl_info_pub_ = create_publisher<Float64MultiArray>(prefix + "/push_pull/info", latched);
    pl_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/push_pull/path", latched);
    pl_drone_path_pub_ = create_publisher<Float64MultiArray>(prefix + "/push_pull/drone_path", latched);
    pl_arrival_pub_ = create_publisher<Float64MultiArray>(prefix + "/push_pull/arrival_error", 10);
    {
      const auto pl_trigger = [this](const std::string & name, std::function<void(Trigger::Response::SharedPtr)> fn) {
          pl_srvs_.push_back(create_service<Trigger>(
            name, [fn](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {fn(r);}));
        };
      pl_trigger(prefix + "/push_pull/adjust", [this](Trigger::Response::SharedPtr r) {onPlAdjust(r);});
      pl_trigger(prefix + "/push_pull/capture_box", [this](Trigger::Response::SharedPtr r) {onPlCapture(r);});
      pl_trigger(prefix + "/push_pull/plan", [this](Trigger::Response::SharedPtr r) {onPlPlan(r);});
      pl_trigger(prefix + "/push_pull/reset", [this](Trigger::Response::SharedPtr r) {onPlReset(r);});
      pl_trigger(prefix + "/push_pull/abort", [this](Trigger::Response::SharedPtr r) {
          {
            std::lock_guard<std::recursive_mutex> lk(lock_);
            pl_guard_tripped_ = false;    // the operator's abort, not a guard's
          }
          onPlAbort(r);
        });
      // the release-window fix (2026-10-17): the station calls it as it opens the gripper
      pl_trigger(prefix + "/push_pull/release", [this](Trigger::Response::SharedPtr r) {onPlRelease(r);});
      for (int k = 0; k < kNumPushPullSteps; ++k) {
        pl_trigger(
          prefix + "/push_pull/" + pushPullStepName(k),
          [this, k](Trigger::Response::SharedPtr r) {onPlStep(k, r);});
      }
    }
    contact_cli_ = create_client<SetBool>(get_parameter("push_pull_contact_service").as_string());
    pl_box_topic_ = get_parameter("push_pull_box_topic").as_string();
    if (!pl_box_topic_.empty()) {
      pl_mocap_sub_ = create_subscription<Mocap>(
        pl_box_topic_, rclcpp::SensorDataQoS(), [this](const Mocap & m) {onPlMocap(m);});
    }
    {
      const auto ftopic = get_parameter("push_pull_force_topic").as_string();
      if (!ftopic.empty()) {
        pl_force_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
          ftopic, rclcpp::SensorDataQoS(),
          [this](const std_msgs::msg::Float32MultiArray & m) {onPlForce(m);});
      }
    }
    pl_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this]() {publishPlArrival();});
    publishPlStatus();
    publishPlInfo();
    publishPlPaths(nullptr);

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

    // --- PS4 teleoperation ----------------------------------------------------
    // The pad feed is taken at depth 1: a stick sample is worth nothing a
    // tick later, and a stall must not hand over a backlog of deflections.
    joy_sub_ = create_subscription<Joy>(
      get_parameter("teleop_joy_topic").as_string(), rclcpp::QoS(1),
      [this](const Joy & m) {onJoy(m);});
    teleop_engage_srv_ = create_service<SetBool>(
      prefix + "/teleop/engage",
      [this](const SetBool::Request::SharedPtr q, SetBool::Response::SharedPtr r) {
        onTeleopEngage(q->data, r);
      });
    teleop_home_srv_ = create_service<Trigger>(
      prefix + "/teleop/arm_home",
      [this](const Trigger::Request::SharedPtr, Trigger::Response::SharedPtr r) {onTeleopHome(r);});
    // [see publishTeleopState] targets, walls and rates, ~20 Hz while engaged
    teleop_state_pub_ = create_publisher<Float64MultiArray>(prefix + "/teleop/state", 10);
    // the last wall that refused a step, latched; "" once motion is free again
    teleop_note_pub_ = create_publisher<String>(prefix + "/teleop/note", latched);
    // The MEASURED arm as points (world): base origin, joints 2..4, grasp
    // point -- the planner's own chain, so a station draws the arm the law
    // sees. Same sample and gate as current_ee.
    skeleton_pub_ = create_publisher<Float64MultiArray>(prefix + "/current_skeleton", 10);

    stream_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / stream_rate_), [this]() {streamTick();});
    ee_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / 15.0), [this]() {currentEeTick();});
    viz_decim_period_ = std::max(1, static_cast<int>(std::lround(stream_rate_ / 20.0)));

    // Pick-and-place parameters edited live (the arm GS's typed-in points,
    // a driver's poses): checked here, and any change to what the legs fly
    // makes the plan STALE -- a plan from the old values must not be flown.
    pp_param_cb_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & ps) {return onPpParams(ps);});
    pl_param_cb_ = add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> & ps) {return onPlParams(ps);});

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
    if (pl_worker_.joinable()) {pl_worker_.join();}
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
      pp_anchor_world_ = false;          // the whole-body node restores its yaml anchor itself
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
        // pick-and-place: the plan and any leg in flight go; the captures,
        // the calibration and the progress stay (they are measurements and
        // the operator's bookkeeping), so a re-Plan resumes the mission
        pp_wp_.reset();
        clearDropAim();
        pp_flying_.reset();
        pp_phase_ = kPpNone;
        pp_exec_plan_.reset();
        pp_abort_plan_.reset();
        pp_aborting_ = false;
        pp_aborted_ = false;
        pp_guard_tripped_ = false;
        pp_planning_ = false;
        ++pp_gen_;
        // push-and-pull: the plan and any step in flight go, and the claw is
        // no longer on the handle (the arm planner folds it home on the
        // revert); the box capture and the typed points stay. The whole-body
        // node restores its yaml contact flag by itself.
        pl_wp_.reset();
        pl_flying_.reset();
        pl_exec_plan_.reset();
        pl_abort_plan_.reset();
        pl_aborting_ = false;
        pl_planning_ = false;
        pl_at_handle_ = false;
        pl_contact_ = false;
        pl_guard_tripped_ = false;
        pl_abort_keep_anchor_ = false;
        pl_abort_anchor_due_.reset();
        ++pl_gen_;
        if (teleop_) {
          RCLCPP_WARN(get_logger(), "SAFETY -- PS4 teleoperation dropped.");
        }
        teleop_.reset();
        if (plFollowing()) {plFollowEnd("left DIRECT");}
        pl_follow_.reset();
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
    publishPlStatus();
    publishPlInfo();
    if (!direct) {publishPlPaths(nullptr);}
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
      if (teleopActive()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "drone-GS target ignored -- PS4 teleoperation owns the reference.");
        return;
      }
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
      if (teleopActive()) {
        RCLCPP_WARN(get_logger(), "EE target ignored -- PS4 teleoperation owns the reference.");
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
    bool trip = false, pl_trip = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      att_R_ = kRIE * r_ned * kRBV;
      att_time_ = Clock::now();
      trip = guardCheck();
      pl_trip = !trip && plGuardCheck();
    }
    if (pl_trip) {
      auto resp = std::make_shared<Trigger::Response>();
      onPlAbort(resp);
      RCLCPP_ERROR(get_logger(), "push-and-pull SAFETY GUARD: %s -- %s", pl_guard_reason_.c_str(),
        resp->message.c_str());
    }
    if (trip) {
      // the button's path, outside the lock (it joins the planning worker)
      auto resp = std::make_shared<Trigger::Response>();
      onPpAbort(resp);
      RCLCPP_ERROR(get_logger(), "pick-and-place SAFETY GUARD: %s -- %s", pp_guard_reason_.c_str(),
        resp->message.c_str());
    }
  }

  // The tilt guard (pick_place_guard_tilt_deg), with lock_ held: true = abort now.
  bool guardCheck()
  {
    const double limit = get_parameter("pick_place_guard_tilt_deg").as_double();
    if (!(limit > 0.0) || !att_R_.has_value()) {return false;}
    const double tilt = std::acos(std::clamp((*att_R_)(2, 2), -1.0, 1.0)) * 180.0 / M_PI;
    if (tilt < limit - 5.0) {
      pp_guard_armed_ = true;     // re-armed once well below the limit
      pp_guard_since_.reset();
      return false;
    }
    const bool engaged = mode_direct_ && hold_.has_value() &&
      (pp_completed_ >= 0 || pp_flying_.has_value() || pp_exiting_.has_value());
    if (tilt <= limit || !engaged || !pp_guard_armed_ || pp_aborting_ || pp_abort_plan_) {
      pp_guard_since_.reset();
      return false;
    }
    if (!pp_guard_since_.has_value()) {pp_guard_since_ = Clock::now();}
    if (secondsSince(pp_guard_since_) < get_parameter("pick_place_guard_persist_s").as_double()) {
      return false;
    }
    pp_guard_armed_ = false;
    pp_guard_since_.reset();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << "tilt " << tilt << " deg > " << limit << " deg";
    pp_guard_reason_ = m.str();
    pp_guard_tripped_ = true;
    return true;
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
    q_meas_time_ = Clock::now();
    qdot_meas_ = joint_sign_.cwiseProduct(qd);
  }

  // Grasp-point EE at the MEASURED joints, and the measured airframe it hangs
  // from, 15 Hz (a display feed).
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
    const bool odom_fresh = have_odom && age <= kOdomFreshS;
    const auto stamp = now();
    // The MEASURED airframe: body origin and ACTUAL attitude (x = nose), the
    // very pair the world current_ee below is built from. Needs no arm.
    if (odom_fresh) {
      PoseStamped base;
      base.header.stamp = stamp;
      base.header.frame_id = "world";
      base.pose.position.x = x_b(0);
      base.pose.position.y = x_b(1);
      base.pose.position.z = x_b(2);
      const Eigen::Quaterniond qb(r0);
      base.pose.orientation.x = qb.x();
      base.pose.orientation.y = qb.y();
      base.pose.orientation.z = qb.z();
      base.pose.orientation.w = qb.w();
      cur_base_pub_->publish(base);
    }
    if (!q.has_value()) {return;}
    Vec3 r0e;
    Mat3 re;
    armKinematics(*q, vehicle_->params, nullptr, &r0e, &re);
    const Vec3 v = vehicle_->r_model * r0e;
    PoseStamped body;
    body.header.stamp = stamp;
    body.header.frame_id = "drone_body";
    body.pose.position.x = v(0);
    body.pose.position.y = v(1);
    body.pose.position.z = v(2);
    body.pose.orientation.w = 1.0;
    cur_ee_body_pub_->publish(body);
    if (!odom_fresh) {
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
    // The MEASURED end-effector frame, in the same convention as the
    // reference_pose (x = claw axis, z = gripper up), so the arm ground
    // station can draw "EE (FK)" beside the mocap marker's "EE (Meas)" and
    // the offset between them reads as the constant it is.
    eeFrameQuat(r0 * vehicle_->r_model, re, &world.pose.orientation);
    cur_ee_pub_->publish(world);

    // The measured arm as world points [base origin, joint 2, joint 3,
    // joint 4, grasp point] (15 doubles), on the same sample.
    std::array<Vec3, 5> pts;
    armChainPoints(*q, vehicle_->params, &pts);
    const Mat3 r0m = r0 * vehicle_->r_model;
    Float64MultiArray sk;
    sk.data.reserve(15);
    for (const Vec3 & pt : pts) {
      const Vec3 w = x_b + r0m * pt;
      sk.data.push_back(w(0));
      sk.data.push_back(w(1));
      sk.data.push_back(w(2));
    }
    skeleton_pub_->publish(sk);
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
      if (state_ == "EXECUTING" || teleopActive()) {return;}
      // any other target supersedes a pick-and-place leg still planning
      pp_flying_.reset();
      pp_phase_ = kPpNone;
      pp_exec_plan_.reset();
      pp_abort_plan_.reset();
      pp_aborting_ = false;
      pl_flying_.reset();
      pl_exec_plan_.reset();
      pl_abort_plan_.reset();
      pl_aborting_ = false;
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
              restoreAnchor();                  // a non-claw motion: back to the CoM anchor
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
      restoreAnchor();                  // a non-claw motion: back to the CoM anchor
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
      if (teleopActive()) {
        resp->success = false;
        resp->message = "PS4 teleoperation engaged -- use PS/Home (teleop/arm_home)";
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
      if (teleopActive()) {
        resp->success = false;
        resp->message = "PS4 teleoperation engaged -- disengage it first";
        return;
      }
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      plan_.reset();
      ++plan_gen_;
      pp_flying_.reset();
      pp_phase_ = kPpNone;
      pp_exec_plan_.reset();
      pl_flying_.reset();
      pl_exec_plan_.reset();
      if (mode_direct_) {state_ = "HOLD";}
    }
    publishStatus();
    publishPpStatus();
    publishBaseAnchor();
    publishVizPath(nullptr);
    resp->success = true;
    resp->message = "targets cleared, holding";
  }

  // ===================================================== PS4 teleoperation
  bool teleopActive() const {return state_ == "TELEOP" || state_ == "TELEOP_STOP";}

  TeleopOptions teleopOptions() const
  {
    const double d2r = M_PI / 180.0;
    TeleopOptions o;
    // Rates are live (the arm station edits them in flight): never negative.
    auto rate = [this](const char * n) {return std::max(0.0, get_parameter(n).as_double());};
    o.com_speed_xy = rate("teleop_com_speed_xy");
    o.com_speed_z = rate("teleop_com_speed_z");
    o.yaw_rate = rate("teleop_yaw_rate_deg") * d2r;
    o.ee_speed = rate("teleop_ee_speed");
    o.roll_rate = rate("teleop_roll_rate_deg") * d2r;
    o.home_rate = std::max(0.01, rate("teleop_home_rate_deg")) * d2r;
    o.com_bandwidth = get_parameter("teleop_com_bandwidth").as_double();
    o.yaw_bandwidth = get_parameter("teleop_yaw_bandwidth").as_double();
    o.arm_bandwidth = get_parameter("teleop_arm_bandwidth").as_double();
    o.box_half_xy = get_parameter("teleop_box_half_xy").as_double();
    o.com_z_min = get_parameter("teleop_com_z_min").as_double();
    o.com_z_max = get_parameter("teleop_com_z_max").as_double();
    o.ee_z_min = get_parameter("teleop_ee_z_min").as_double();
    o.leash = get_parameter("teleop_leash_m").as_double();
    o.joint_range_frac = get_parameter("teleop_joint_range_frac").as_double();
    const auto hp = get_parameter("teleop_home_pose_deg").as_double_array();
    if (hp.size() == static_cast<size_t>(kNumJoints)) {
      for (int j = 0; j < kNumJoints; ++j) {o.home(j) = hp[static_cast<size_t>(j)] * d2r;}
    }
    o.q3_min = get_parameter("teleop_q3_min_deg").as_double() * d2r;
    return o;
  }

  void onJoy(const Joy & m)
  {
    std::lock_guard<std::recursive_mutex> lk(lock_);
    joy_ = m;
    joy_time_ = Clock::now();
  }

  bool joyFresh() const
  {
    return joy_.has_value() &&
           secondsSince(joy_time_) <= get_parameter("teleop_joy_timeout_s").as_double();
  }

  // The pad as operator input (under the lock). A stale feed reads all zero:
  // a velocity must never be latched the way a position setpoint is.
  TeleopInput joyInput(bool * neutral)
  {
    TeleopInput in;
    // Only a FRESH pad reading all-centred counts as neutral: a silent feed
    // reads zero input but must never arm the inputs.
    *neutral = false;
    if (!joyFresh()) {
      home_prev_ = false;
      return in;
    }
    const Joy & m = *joy_;
    const double dz = get_parameter("teleop_stick_deadzone").as_double();
    auto axis = [&](int i) {
        const int64_t idx = teleop_axes_[static_cast<size_t>(i)];
        double v = (idx >= 0 && static_cast<size_t>(idx) < m.axes.size())
          ? static_cast<double>(m.axes[static_cast<size_t>(idx)]) : 0.0;
        v *= teleop_axis_signs_[static_cast<size_t>(i)];
        if (std::abs(v) < dz) {return 0.0;}
        return std::copysign(std::min(1.0, (std::abs(v) - dz) / (1.0 - dz)), v);
      };
    auto button = [&](int i) {
        const int64_t idx = teleop_buttons_[static_cast<size_t>(i)];
        return idx >= 0 && static_cast<size_t>(idx) < m.buttons.size() &&
               m.buttons[static_cast<size_t>(idx)] != 0;
      };
    // Sticks: the grasp point relative to the airframe + the wrist roll.
    in.ee = Vec3{axis(0), axis(1), axis(2)};
    in.roll = axis(3);
    // D-pad (a hat: exactly -1/0/+1): UP = forward, LEFT = left. A diagonal
    // is normalised so it is not sqrt(2) faster than a straight press.
    in.com(0) = axis(5);
    in.com(1) = axis(4);
    const double n = std::hypot(in.com(0), in.com(1));
    if (n > 1.0) {in.com(0) /= n; in.com(1) /= n;}
    // Action buttons: triangle up / cross down, square yaw left / circle right.
    in.com(2) = (button(0) ? 1.0 : 0.0) - (button(1) ? 1.0 : 0.0);
    in.yaw = (button(2) ? 1.0 : 0.0) - (button(3) ? 1.0 : 0.0);
    const bool home_now = button(4);
    in.home = home_now && !home_prev_;
    home_prev_ = home_now;
    *neutral = in.ee.cwiseAbs().maxCoeff() == 0.0 && in.roll == 0.0 &&
      in.com.cwiseAbs().maxCoeff() == 0.0 && in.yaw == 0.0 && !home_now;
    return in;
  }

  // Measured SYSTEM CoM (fresh odometry + attitude + encoders), for the leash.
  std::optional<Vec3> measuredCom()
  {
    Vec3 x_b;
    Mat3 r0;
    if (!odomPair(&x_b, &r0) || odomAge() > kOdomFreshS || !q_meas_.has_value()) {
      return std::nullopt;
    }
    Vec3 r0c;
    armKinematics(*q_meas_, vehicle_->params, &r0c, nullptr, nullptr);
    return x_b + r0 * vehicle_->r_model * r0c;
  }

  // One teleop sample (under the lock). Sets *done when a stop has settled
  // and the final rest became the HOLD.
  WbReference teleopTick(bool * done)
  {
    const auto now = Clock::now();
    double dt = teleop_last_tick_.has_value()
      ? std::chrono::duration<double>(now - *teleop_last_tick_).count() : 1.0 / stream_rate_;
    teleop_last_tick_ = now;
    dt = std::min(std::max(dt, 0.0), 0.05);
    const double s = std::min(1.0, std::max(0.05, get_parameter("teleop_time_scale").as_double()));

    bool neutral = false;
    TeleopInput in = joyInput(&neutral);
    if (teleop_armed_ && !joyFresh()) {
      // The pad went quiet: inputs already read zero; disarm too, so a button
      // still held when the feed returns cannot resume a motion by itself.
      teleop_armed_ = false;
      RCLCPP_WARN(get_logger(), "PS4 teleoperation: pad feed lost -- holding; centre the pad to resume.");
    }
    if (state_ == "TELEOP") {
      // Sticks count only after they have been seen at NEUTRAL since the
      // engage: a pad held deflected at that moment must not move anything.
      if (!teleop_armed_) {
        if (neutral) {
          teleop_armed_ = true;
          RCLCPP_INFO(get_logger(), "PS4 teleoperation: pad at neutral -- inputs live.");
        }
        in = TeleopInput{};
      }
      if (teleop_home_request_) {
        in.home = true;
        teleop_home_request_ = false;
      }
    } else {
      in = TeleopInput{};     // stopping: the smoothers settle on the targets
    }
    teleop_->setOptions(teleopOptions());
    teleop_->step(dt * s, in, measuredCom());
    WbReference ref;
    try {
      ref = teleop_->reference(s);
      teleop_last_ref_ = ref;
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "PS4 teleoperation stopped: reference failed (%s).", e.what());
      ref = teleop_last_ref_.value_or(*hold_ref_);
      setHold(teleop_->targetRest());
      teleop_.reset();
      state_ = "HOLD";
      *done = true;
      return ref;
    }
    const std::string wall = teleop_->status().wall;
    if (wall != teleop_note_) {
      teleop_note_ = wall;
      publishTeleopNote(wall);
    }
    if (state_ == "TELEOP_STOP" && teleop_->settled()) {
      setHold(teleop_->targetRest());
      teleop_.reset();
      state_ = "HOLD";
      logHold("PS4 teleoperation ended");
      *done = true;
    }
    return ref;
  }

  void onTeleopEngage(bool on, SetBool::Response::SharedPtr resp)
  {
    bool engaged_now = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (on) {
        if (!mode_direct_ || !hold_.has_value() || !hold_ref_.has_value()) {
          resp->success = false;
          resp->message = "not in whole-body DIRECT";
          return;
        }
        if (state_ == "TELEOP") {
          resp->success = true;
          resp->message = "already engaged";
          return;
        }
        if (state_ == "EXECUTING" || state_ == "CALCULATING") {
          resp->success = false;
          resp->message = "planner is " + state_ + " -- wait for HOLD";
          return;
        }
        if (pp_flying_.has_value()) {
          // a claw leg WAITING above its target holds in HOLD: a teleop
          // session now would leave the vehicle elsewhere and the descent
          // would then be planned from there (the fine correction is after it)
          resp->success = false;
          resp->message = std::string(pickPlaceLegName(*pp_flying_)) +
            " is waiting above its target -- let it descend, or Abort it";
          return;
        }
        restoreAnchor();
        if (state_ == "TELEOP_STOP" && teleop_) {
          // re-engaged while settling: carry on from the same targets
          state_ = "TELEOP";
        } else {
          teleop_ = std::make_unique<TeleopReference>(vehicle_, teleopOptions());
          // Seeded on the HOLD being streamed, so the first sample IS it.
          teleop_->seed(hold_ref_->x_cd, hold_->phi, hold_->q);
          pending_base_.reset();
          ee_target_.reset();
          home_goal_ = false;
          plan_.reset();
          ++plan_gen_;
          goal_override_.reset();
          auto_send_ = false;
          teleop_last_tick_.reset();
          teleop_last_ref_.reset();
          state_ = "TELEOP";
        }
        teleop_armed_ = false;
        teleop_home_request_ = false;
        home_prev_ = true;           // a PS button held at engage does not fire
        engaged_now = true;
        std::ostringstream m;
        m.setf(std::ios::fixed);
        m.precision(2);
        m << "PS4 teleoperation engaged (time scale "
          << get_parameter("teleop_time_scale").as_double()
          << ") -- centre the pad to make it live";
        resp->success = true;
        resp->message = m.str();
      } else {
        if (state_ == "TELEOP") {
          state_ = "TELEOP_STOP";
          resp->message = "stopping: settling on the last targets, then HOLD";
        } else if (state_ == "TELEOP_STOP") {
          resp->message = "already stopping";
        } else {
          resp->message = "not engaged";
        }
        resp->success = true;
      }
    }
    RCLCPP_INFO(get_logger(), "teleop/engage(%s): %s", on ? "true" : "false", resp->message.c_str());
    publishStatus();
    if (engaged_now) {
      publishBaseAnchor();
      publishVizPath(nullptr);
    }
  }

  void onTeleopHome(Trigger::Response::SharedPtr resp)
  {
    std::lock_guard<std::recursive_mutex> lk(lock_);
    if (state_ != "TELEOP") {
      resp->success = false;
      resp->message = "PS4 teleoperation is not engaged";
      return;
    }
    teleop_home_request_ = true;
    resp->success = true;
    resp->message = "arm folding home (platform held)";
  }

  void publishTeleopNote(const std::string & s)
  {
    String m;
    m.data = s;
    teleop_note_pub_->publish(m);
  }

  // [0] 1 engaged / 2 stopping   [1..3] CoM target   [4] base yaw target
  // (ACTUAL)   [5..7] base target position   [8..10] grasp-point target
  // [11..14] its quaternion (x = claw axis, same convention as current_ee)
  // [15..18] joint targets (model, rad)   [19..21] grasp point vs airframe,
  // ACTUAL body frame (fwd, left, up)   [22] CoM steps refused (bits x/y/z)
  // [23] arm steps refused (bits fwd/left/up/roll)   [24] leash   [25] homing
  // [26] sigma_nd   [27..31] rates: CoM xy, CoM z [m/s], yaw [deg/s], EE
  // [m/s], roll [deg/s]   [32] time scale   [33] pad fresh   [34] inputs live
  // Empty = not engaged.
  void publishTeleopState()
  {
    Float64MultiArray m;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!teleop_) {
        if (teleop_state_cleared_) {return;}
        teleop_state_cleared_ = true;
      } else {
        teleop_state_cleared_ = false;
        const TeleopStatus & st = teleop_->status();
        const TeleopOptions & o = teleop_->options();
        const RestSpec r = teleop_->targetRest();
        Vec3 pe;
        Mat3 R0, Re;
        teleop_->targetEe(&pe, &R0, &Re);
        geometry_msgs::msg::Quaternion qe;
        eeFrameQuat(R0, Re, &qe);
        const Vec3 & c = teleop_->comTarget();
        const VecN & q = teleop_->qTarget();
        const Vec3 s_act = vehicle_->r_model * teleop_->sTarget();
        m.data = {
          state_ == "TELEOP" ? 1.0 : 2.0,
          c(0), c(1), c(2), wrapPi(r.phi + 0.5 * M_PI),
          r.x_b(0), r.x_b(1), r.x_b(2),
          pe(0), pe(1), pe(2), qe.x, qe.y, qe.z, qe.w,
          q(0), q(1), q(2), q(3),
          s_act(0), s_act(1), s_act(2),
          static_cast<double>(st.com_blocked), static_cast<double>(st.arm_blocked),
          st.leash_active ? 1.0 : 0.0, st.homing ? 1.0 : 0.0,
          st.sigma_nd,
          o.com_speed_xy, o.com_speed_z, o.yaw_rate * 180.0 / M_PI, o.ee_speed,
          o.roll_rate * 180.0 / M_PI,
          get_parameter("teleop_time_scale").as_double(),
          joyFresh() ? 1.0 : 0.0, teleop_armed_ ? 1.0 : 0.0};
        // [35..38] / [39..42]: the pad's inner joint box, lower / upper [rad]
        for (int k = 0; k < 2; ++k) {
          for (int j = 0; j < kNumJoints; ++j) {
            double lo, hi;
            teleop_->jointBox(j, &lo, &hi);
            m.data.push_back(k == 0 ? lo : hi);
          }
        }
      }
    }
    teleop_state_pub_->publish(m);
  }

  // ------------------------------------------------------------- streaming
  void streamTick()
  {
    WbReference ref;
    bool finished = false;
    bool teleop_done = false;
    bool follow_abort = false;
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
      if (state_ == "EXECUTING" && plan_ && plFollowing()) {
        // the governed turn (push_pull_follow): evaluated at its progress
        const double tau = plFollowTick(&follow_abort);
        if (tau >= plan_->duration()) {
          finished = true;
        } else {
          ref = plan_->evalScaled(tau, pl_follow_->gov.rates());
          have_ref = true;
        }
      } else if (state_ == "EXECUTING" && plan_) {
        const double t = std::chrono::duration<double>(Clock::now() - *exec_t0_).count();
        if (t >= plan_->duration()) {
          finished = true;
        } else {
          ref = plan_->eval(t);
          have_ref = true;
        }
      }
      if (finished && plFollowing()) {plFollowEnd("complete");}
      if (finished && plUnloadStart()) {
        // the push flew to its end: the unload leg replaced it (still the push)
        finished = false;
        ref = plan_->eval(0.0);
        have_ref = true;
      }
      if (finished) {
        if (pp_abort_plan_ && pp_abort_plan_ == plan_) {
          RCLCPP_WARN(get_logger(), "pick-and-place: ABORT complete -- holding, arm home.");
          pp_at_table_ = false;   // climbed off, arm folded
        }
        pp_abort_plan_.reset();
        if (pp_exit_plan_ && pp_exit_plan_ == plan_ && pp_exiting_.has_value()) {
          pp_exited_[*pp_exiting_] = true;
          pp_at_table_ = false;   // back at the clearance point: the next leg needs no retreat
          RCLCPP_INFO(get_logger(), "pick-and-place: %s complete.", ppExitName(*pp_exiting_));
          if (pp_exit_to_land_) {
            // the drop's exit flew on to Land Start: Go To Land Start is done
            pp_completed_ = kGoToLandStart;
            RCLCPP_INFO(get_logger(), "pick-and-place: %s ended on Land Start -- %s done.",
              ppExitName(*pp_exiting_), pickPlaceLegName(kGoToLandStart));
          }
        }
        if (pp_exit_plan_ && pp_exit_plan_ == plan_) {pp_exit_to_land_ = false;}
        pp_exit_plan_.reset();
        pp_exiting_.reset();
        if (pp_flying_.has_value() && pp_exec_plan_ && pp_exec_plan_ == plan_) {
          const int leg = *pp_flying_;
          // the leg flew to its APPROACH rest when that differs from the goal
          const bool above = pp_wp_.has_value() &&
            (pp_wp_->approach[leg].x_b - pp_wp_->goal[leg].x_b).norm() > 1e-6;
          if (pickPlaceLegIsClaw(leg) && pp_phase_ == kPpTravel && above) {
            // above the target: hold there until the claw is within the
            // tolerance, then descend (publishPpArrival)
            pp_phase_ = kPpWait;
            pp_wait_t0_ = Clock::now();
            pp_within_since_.reset();
            RCLCPP_INFO(
              get_logger(), "pick-and-place: %s above its target -- descending once the claw is "
              "within tolerance.", pickPlaceLegName(leg));
            // from here through the descent and the grasp the claw is world-held
            // (pick_place_world_anchor_*); the next request restores the CoM anchor
            if ((leg == kExecutePick && get_parameter("pick_place_world_anchor_pick").as_bool()) ||
              (leg == kExecutePlace && get_parameter("pick_place_world_anchor_place").as_bool()))
            {
              requestAnchor(false);
            }
          } else {
            pp_completed_ = leg;
            pp_flying_.reset();
            pp_phase_ = kPpNone;
            pp_at_table_ = pickPlaceLegIsClaw(leg);
            if (leg == kExecutePlace) {
              // the drop hold starts here (the autonomous drop's wait counts from now)
              pp_drop_hold_t0_ = Clock::now();
              resetDropAuto();
            }
            RCLCPP_INFO(get_logger(), "pick-and-place: %s complete.", pickPlaceLegName(leg));
          }
        } else {
          pp_flying_.reset();
          pp_phase_ = kPpNone;
        }
        pp_exec_plan_.reset();
        // push-and-pull: the abort, or a step flown to its end
        if (pl_abort_plan_ && pl_abort_plan_ == plan_) {
          RCLCPP_WARN(get_logger(), "push-and-pull: ABORT complete -- holding, arm home.");
          pl_at_handle_ = false;
        }
        pl_abort_plan_.reset();
        if (pl_flying_.has_value() && pl_exec_plan_ && pl_exec_plan_ == plan_) {
          plStepDone(*pl_flying_);
        } else {
          pl_flying_.reset();
        }
        pl_exec_plan_.reset();
        setHold(plan_->goalRest());
        plan_.reset();
        pending_base_.reset();
        ee_target_.reset();
        home_goal_ = false;
        state_ = "HOLD";
        logHold("transition complete");
      }
      if (!have_ref && teleopActive() && teleop_) {
        ref = teleopTick(&teleop_done);
        have_ref = true;
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
      publishPlStatus();
      publishPlInfo();
    }
    if (teleop_done) {
      publishStatus();
      publishBaseAnchor();
      publishTeleopNote("");
    }
    if (follow_abort) {
      // the governed turn held past push_pull_follow_hold_timeout_s: the
      // abort's path, outside the lock (it joins the planning worker)
      auto resp = std::make_shared<Trigger::Response>();
      onPlAbort(resp);
      RCLCPP_ERROR(get_logger(), "push-and-pull PATH FOLLOWING: %s -- %s", pl_guard_reason_.c_str(),
        resp->message.c_str());
    }
    publishRef(ref);
    // The arm reference, from the SAME sample the law gets, at the same rate.
    publishArmSync(ref.q_d, ref.qdot_d);
    viz_decim_ = (viz_decim_ + 1) % viz_decim_period_;
    if (viz_decim_ == 0) {
      publishVizPose(&ref);
      publishEeRefPose(ref);
      publishEeDroneRefPose(ref);
      publishTeleopState();
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
    sh.fig8_world_axis = get_parameter("ee_traj_fig8_world_axis").as_bool();
    sh.fig8_axis = get_parameter("ee_traj_fig8_axis_deg").as_double() * M_PI / 180.0;
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
    const int64_t q2_cycles = get_parameter("ee_traj_q2_cycles_per_lap").as_int();
    if (q2_cycles > 0) {
      o.q2_period_s = get_parameter("ee_traj_lap_time").as_double() / static_cast<double>(q2_cycles);
    }
    o.qdot_max = get_parameter("ee_traj_qdot_max").as_double();
    auto bound = [this](const char * ee_key, const char * shared_key) {
        const double b = get_parameter(ee_key).as_double();
        return b > 0.0 ? b : get_parameter(shared_key).as_double();
      };
    o.v_max = bound("ee_traj_v_max", "v_max");
    o.a_max = bound("ee_traj_a_max", "a_max");
    o.w_max = bound("ee_traj_w_max", "w_max");
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
      if (teleopActive()) {
        resp->success = false;
        resp->message = "PS4 teleoperation engaged -- disengage it first";
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
      pp_phase_ = kPpNone;
      pp_exec_plan_.reset();
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      goal_override_.reset();
      auto_send_ = false;
      ++plan_gen_;
      restoreAnchor();                  // a non-claw motion: back to the CoM anchor
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
        resp->message = "executing -- wait for the run to finish, or revert to SAFETY to abort";
        return;
      }
      if (teleopActive()) {
        resp->success = false;
        resp->message = "PS4 teleoperation engaged -- disengage it first";
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
      gs_ref_z_.value_or(std::numeric_limits<double>::quiet_NaN()),
      // [15] run time at which the first lap ends: where a display cuts
      // drone_path to draw one airframe loop (see EeTrajectoryDiag).
      d.t_lap_end};
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
  // base-pose parameters (Adjust writes its shift into them) and the claw points
  // (pick captured, place typed; both mocap readings, not shifted); `plan` dry-runs the whole mission from the hold, and
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
      !ppVec3("pick_place_ee_offset", &c->ee_offset, err) ||
      !ppJoints("pick_place_sweep_lo_deg", &c->sweep.lo, err) ||
      !ppJoints("pick_place_sweep_hi_deg", &c->sweep.hi, err) ||
      !ppJoints("pick_place_sweep_phase_deg", &c->sweep.phase, err))
    {
      return false;
    }
    c->align_yaw = get_parameter("pick_place_align_yaw").as_bool();
    c->face_target = get_parameter("pick_place_face_target").as_bool();
    c->approach_dz = get_parameter("pick_place_approach_dz").as_double();
    c->exit_t = get_parameter("pick_place_exit_time_s").as_double();
    c->trim_settle_s = get_parameter("pick_place_descent_trim_settle_s").as_double();
    c->trim_first_min = get_parameter("pick_place_descent_trim_first_min").as_double();
    c->pick_approach_back = get_parameter("pick_place_pick_approach_back").as_double();
    c->place_exit_back = get_parameter("pick_place_place_exit_back").as_double();
    c->place_drop = get_parameter("pick_place_place_drop").as_bool();
    c->drop_height = get_parameter("pick_place_drop_height").as_double();
    c->drop_altitude = get_parameter("pick_place_drop_altitude").as_double();
    c->drop_climb_lead = get_parameter("pick_place_drop_climb_lead").as_double();
    {
      // the drop exit = the place point (x, y) + this offset, room frame; never
      // shifted by Adjust (the place point is not either)
      const auto de = get_parameter("pick_place_drop_exit_offset").as_double_array();
      if (!de.empty() && de.size() != 3) {
        *err = "pick_place_drop_exit_offset needs [] or [dx, dy, yaw_deg]";
        return false;
      }
      c->drop_exit_set = de.size() == 3;
      c->drop_exit_climb = get_parameter("pick_place_drop_exit_climb").as_double();
      c->drop_exit_to_land_start = get_parameter("pick_place_drop_exit_to_land_start").as_bool();
      c->drop_exit_back = get_parameter("pick_place_drop_exit_back").as_double();
      if (c->drop_exit_set) {
        const auto pp = get_parameter("pick_place_place_point").as_double_array();
        if (pp.size() != 3 && pp.size() != 4) {
          *err = "pick_place_drop_exit_offset is relative to pick_place_place_point, which needs [x, y, z, yaw_deg]";
          return false;
        }
        c->drop_exit.p = Vec3{pp[0] + de[0], pp[1] + de[1], 0.0};   // z: the drop altitude
        c->drop_exit.yaw = de[2] * M_PI / 180.0;
      }
    }
    c->fence.min_z = get_parameter("pick_place_fence_min_z").as_double();
    c->fence.max_z = get_parameter("pick_place_fence_max_z").as_double();
    const auto xy = get_parameter("pick_place_fence_xy").as_double_array();
    if (xy.size() != 4) {
      *err = "pick_place_fence_xy needs [x_min, x_max, y_min, y_max]";
      return false;
    }
    c->fence.x_min = xy[0];
    c->fence.x_max = xy[1];
    c->fence.y_min = xy[2];
    c->fence.y_max = xy[3];
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

  // The goals' inputs: the base-pose parameters as they stand -- Adjust has
  // already written its shift into them (x, y from a mocap capture instead
  // when one was taken), the captured pick point and
  // the typed (or captured) place point -- the two claw points are mocap
  // readings and are never shifted. Called under the lock.
  bool ppTargets(PickPlaceTargets * t, std::string * err) const
  {
    BasePose * base[kNumPickPlaceLegs] = {
      &t->start, nullptr, &t->place_start, nullptr, &t->land_start, &t->land};
    std::string missing;
    for (int k = 0; k < kNumPickPlaceLegs; ++k) {
      if (base[k] != nullptr) {
        // the parameters already carry the Adjust shift (onPpAdjust writes it in)
        if (!ppBase(std::string("pick_place_") + kPpPoint[k], base[k], err)) {return false;}
        if (pp_capture_[k].has_value()) {
          base[k]->p.head<2>() = pp_capture_[k]->head<2>();   // measured: no offset
        }
        continue;
      }
      if (k == kExecutePlace && !pp_capture_[k].has_value()) {
        // typed in, but NOT shifted (2026-10-03, user decision): the value is the
        // place mark read off mocap in THIS session -- before the flight, since
        // the cap that carries the payload then covers it -- so it is already in
        // the mocap frame, exactly like the captured pick point. The Adjust
        // offset is where the vehicle stands relative to the typed Start; it
        // says nothing about where the table is.
        const auto v = get_parameter("pick_place_place_point").as_double_array();
        if (v.size() != 3 && v.size() != 4) {
          *err = "pick_place_place_point needs [x, y, z, yaw_deg]";
          return false;
        }
        t->place = Vec3{v[0], v[1], v[2]};
        t->place_yaw = v.size() == 4 ? v[3] * M_PI / 180.0 : 0.0;
        continue;
      }
      if (!pp_capture_[k].has_value()) {
        missing += (missing.empty() ? "" : " and ") + std::string(kPpPoint[k]) + " (" +
          (pp_topic_[k].empty() ? "no topic set" : pp_topic_[k]) + ")";
        continue;
      }
      (k == kExecutePick ? t->pick : t->place) = *pp_capture_[k];
      // a capture carries the body's yaw too: the frame ee_offset is applied in
      (k == kExecutePick ? t->pick_yaw : t->place_yaw) = pp_capture_yaw_[k].value_or(0.0);
    }
    if (!missing.empty()) {
      *err = "capture " + missing + " first";
      return false;
    }
    return true;
  }

  rcl_interfaces::msg::SetParametersResult onPpParams(const std::vector<rclcpp::Parameter> & ps)
  {
    rcl_interfaces::msg::SetParametersResult r;
    r.successful = true;
    bool stale = false;
    // a new pick point: applied at the end, once every parameter of this set
    // is accepted (this callback runs BEFORE the value is stored)
    const rclcpp::Parameter * pick_point = nullptr;
    for (const auto & p : ps) {
      const std::string & n = p.get_name();
      if (n.rfind("pick_place_", 0) != 0) {continue;}
      if (n.size() > 6 && n.compare(n.size() - 6, 6, "_topic") == 0) {
        r.successful = false;
        r.reason = n + " is fixed at launch (the mocap subscription is made then)";
        return r;
      }
      if (n == "pick_place_anchor_service") {
        r.successful = false;
        r.reason = n + " is fixed at launch (the service client is made then)";
        return r;
      }
      const auto want = [&](size_t len) {
          if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY ||
            p.as_double_array().size() != len)
          {
            r.successful = false;
            r.reason = n + " needs " + std::to_string(len) + " numbers";
          }
        };
      if (n == "pick_place_start" || n == "pick_place_place_start" ||
        n == "pick_place_land_start" || n == "pick_place_land")
      {
        want(4);
      } else if (n == "pick_place_fence_xy") {
        want(4);
      } else if (n == "pick_place_drop_aim_window_s" || n == "pick_place_drop_aim_min_s" ||
        n == "pick_place_drop_aim_max_shift" || n == "pick_place_drop_aim_settle_s")
      {
        const bool settle = n == "pick_place_drop_aim_settle_s";
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
          !std::isfinite(p.as_double()) || (settle ? p.as_double() < 0.0 : !(p.as_double() > 0.0)))
        {
          r.successful = false;
          r.reason = n + (settle ? " needs a number >= 0 (a double)" : " needs a number > 0 (a double)");
        }
      } else if (n == "pick_place_ee_offset") {
        want(3);
      } else if (n == "pick_place_drop_exit_offset") {
        // [] = the back-off-and-climb exit, else [dx, dy, yaw_deg] from the place point
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY ||
          (!p.as_double_array().empty() && p.as_double_array().size() != 3))
        {
          r.successful = false;
          r.reason = n + " needs [] or [dx, dy, yaw_deg]";
        }
      } else if (n == "pick_place_abort_release_pose_deg") {
        // [] = straight home, else the 4 joints [deg]
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY ||
          (!p.as_double_array().empty() && p.as_double_array().size() != 4))
        {
          r.successful = false;
          r.reason = n + " needs [] or 4 joint angles [deg]";
        }
      } else if (n == "pick_place_place_point") {
        // [x, y, z, yaw_deg]; a 3-value [x, y, z] (before 2026-10-03) is yaw 0
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY ||
          (p.as_double_array().size() != 3 && p.as_double_array().size() != 4))
        {
          r.successful = false;
          r.reason = n + " needs [x, y, z, yaw_deg]";
        }
      } else if (n == "pick_place_pick_point") {
        // [] = not captured, else [x, y, z, yaw_deg], all finite (2026-10-10)
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
          r.successful = false;
          r.reason = n + " needs [x, y, z, yaw_deg], or [] for not captured";
        } else if (ppPickPointOk(p.as_double_array(), &r.reason)) {
          pick_point = &p;
        } else {
          r.successful = false;
        }
      }
      if (!r.successful) {return r;}
      // what only gates or reports leaves the plan valid
      if (n != "pick_place_arrival_tol" && n != "pick_place_capture_window" &&
        n != "pick_place_capture_max_spread" && n != "pick_place_capture_max_yaw_spread_deg" &&
        n != "pick_place_adjust_z" &&
        n != "pick_place_adjust_yaw" && n != "pick_place_settle_s" &&
        n != "pick_place_approach_wait_max" && n != "pick_place_leg_start_tol" &&
        n != "pick_place_adjust_max" && n != "pick_place_abort_climb" &&
        n != "pick_place_world_anchor_pick" && n != "pick_place_world_anchor_place" &&
        n != "pick_place_descent_on_request" && n != "pick_place_guard_tilt_deg" &&
        n != "pick_place_descent_trim" && n != "pick_place_descent_trim_window_s" &&
        n != "pick_place_descent_trim_max" &&
        n != "pick_place_guard_persist_s" && n != "pick_place_abort_release_pose_deg" &&
        n != "pick_place_abort_release_hold_s" &&
        // the autonomous drop's gates and the basket aim are read at the
        // drop hold / at Ready To Drop, never baked into the plan
        n.rfind("pick_place_drop_auto", 0) != 0 && n.rfind("pick_place_drop_aim_", 0) != 0)
      {
        stale = true;
      }
    }
    if (stale) {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pp_flying_.has_value()) {
        r.successful = false;
        r.reason = "a pick-and-place leg is in flight -- edit once it holds";
        return r;
      }
      if (pp_wp_.has_value() || pp_planning_) {
        RCLCPP_INFO(get_logger(), "pick-and-place: parameters changed -- the plan is stale, press Plan");
      }
      // the pick point IS the pick capture (2026-10-10). Nothing can refuse
      // the set after this: rclcpp runs the on-set callbacks newest first and
      // this one was registered first, so it is the last to run.
      if (pick_point != nullptr) {ppSetPickCapture(pick_point->as_double_array());}
      ppInvalidate();
      pp_params_dirty_ = true;   // published from the 10 Hz timer, once the values are set
    }
    return r;
  }

  // pick_place_pick_point: [] (not captured) or [x, y, z, yaw_deg], finite.
  static bool ppPickPointOk(const std::vector<double> & v, std::string * why)
  {
    if (!v.empty() && v.size() != 4) {
      *why = "pick_place_pick_point needs [x, y, z, yaw_deg], or [] for not captured (got " +
        std::to_string(v.size()) + " numbers)";
      return false;
    }
    for (double x : v) {
      if (!std::isfinite(x)) {
        *why = "pick_place_pick_point has a non-finite value";
        return false;
      }
    }
    return true;
  }

  // The pick capture from a (checked) pick_place_pick_point value. Called
  // under the lock (or from the constructor). The yaw is taken as typed, not
  // wrapped -- like the place pose's (a typed turn keeps its direction).
  void ppSetPickCapture(const std::vector<double> & v)
  {
    if (v.size() == 4) {
      pp_capture_[kExecutePick] = Vec3{v[0], v[1], v[2]};
      pp_capture_yaw_[kExecutePick] = v[3] * M_PI / 180.0;
    } else {
      pp_capture_[kExecutePick].reset();
      pp_capture_yaw_[kExecutePick].reset();
    }
  }

  void onPpMocap(int k, const Mocap & msg)
  {
    const Vec3 p{msg.pose.position.x, msg.pose.position.y, msg.pose.position.z};
    if (!p.allFinite()) {return;}
    // the body's yaw (ENU): the pick object's heading, which pick_place_ee_offset follows
    const auto & o = msg.pose.orientation;
    const double yaw = std::atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z));
    // the body's tilt (angle of its z axis from the vertical): a 180 deg flipped
    // solve reads > 90 deg (the autonomous drop skips those samples)
    const double cz = 1.0 - 2.0 * (o.x * o.x + o.y * o.y);
    const double tilt = std::acos(std::clamp(cz, -1.0, 1.0));
    std::lock_guard<std::recursive_mutex> lk(lock_);
    auto & d = pp_mocap_[k];
    d.push_back(PpSample{Clock::now(), p, std::isfinite(yaw) ? yaw : 0.0, std::isfinite(tilt) ? tilt : M_PI});
    while (d.size() > 1 && (secondsSince(d.front().t) > 2.0 || d.size() > 2000)) {
      d.pop_front();
    }
  }

  // The inputs changed: a plan made from the old ones must not be flown.
  // Called under the lock.
  void ppInvalidate()
  {
    pp_wp_.reset();
    clearDropAim();
    pp_planning_ = false;
    pp_error_.clear();
    pp_timed_out_ = false;
    ++pp_gen_;
  }

  // "/obj_0/mocap" -> "obj_0": the body as the operator knows it
  std::string ppBody(int k) const
  {
    std::string b = pp_topic_[k];
    const std::string tail = "/mocap";
    if (b.size() > tail.size() && b.compare(b.size() - tail.size(), tail.size(), tail) == 0) {
      b.erase(b.size() - tail.size());
    }
    if (!b.empty() && b.front() == '/') {b.erase(0, 1);}
    return b;
  }

  // A mocap body averaged over the capture window.
  struct PpAverage
  {
    Vec3 mean{Vec3::Zero()};
    double yaw{0.0};       // circular mean [rad]: no wrap at +-180 deg
    size_t n{0};
    double spread{0.0};    // max |sample - mean| [m]
  };

  // Average point k's mocap body over pick_place_capture_window -- what
  // capture_<point> and get_place both measure (factored out of onPpCapture,
  // 2026-10-10). REFUSES (false; *why names what to look at, ending in
  // `verdict` when samples were there but must not become a goal): no sample,
  // a FROZEN pose, scattered positions, a FLIPPING yaw. Called under the lock.
  bool ppAverageBody(int k, const char * verdict, PpAverage * out, std::string * why) const
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    const double window = get_parameter("pick_place_capture_window").as_double();
    Vec3 sum = Vec3::Zero();
    std::vector<Vec3> pts;
    std::vector<double> yaws;
    double ys = 0.0, yc = 0.0;
    for (const auto & s : pp_mocap_[k]) {
      if (secondsSince(s.t) <= window) {
        pts.push_back(s.p);
        yaws.push_back(s.yaw);
        ys += std::sin(s.yaw);
        yc += std::cos(s.yaw);
      }
    }
    if (pts.empty()) {
      m.precision(2);
      m << "no sample on " << pp_topic_[k] << " in the last " << window
        << " s -- is the body tracked and broadcast?";
      *why = m.str();
      return false;
    }
    const auto refuse = [&]() {
        m << " -- " << verdict;
        *why = m.str();
        RCLCPP_WARN(get_logger(), "pick-and-place capture refused: %s", why->c_str());
        return false;
      };
    // FROZEN (2026-10-10, the first hardware pick-and-place flights): every
    // position in the window bit-identical. A tracked body jitters by tens of
    // micrometres at least; identical samples are the mocap processor
    // republishing the last pose of a body it no longer tracks, with fresh
    // stamps (11 423 identical obj_0 samples in one flight).
    if (pts.size() >= 2 &&
      std::all_of(pts.begin() + 1, pts.end(), [&pts](const Vec3 & p) {return p == pts.front();}))
    {
      m.precision(2);
      m << ppBody(k) << "'s pose is frozen -- the body is not tracked (" << pts.size()
        << " identical samples from " << pp_topic_[k] << " in " << window
        << " s: the mocap processor is republishing its last pose)";
      return refuse();
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
           "mis-tracked";
      return refuse();
    }
    // FLIPPING YAW (2026-10-10, same flights): obj_0's solved orientation
    // jumped by ~180 deg several times a second, and the circular mean of two
    // opposite headings is garbage. The spread is taken about the first sample
    // (wrapped), so a steady yaw near +-180 deg reads small. Only the two CLAW
    // points' bodies: a base point's capture uses x, y alone, and a floor mark
    // may be a single marker with no orientation to speak of.
    const bool claw_body = k == kExecutePick || k == kExecutePlace;
    const double max_yaw =
      claw_body ? get_parameter("pick_place_capture_max_yaw_spread_deg").as_double() : 0.0;
    double lo = 0.0, hi = 0.0;
    for (double y : yaws) {
      const double d = wrapPi(y - yaws.front());
      lo = std::min(lo, d);
      hi = std::max(hi, d);
    }
    const double yaw_spread = (hi - lo) * 180.0 / M_PI;
    if (max_yaw > 0.0 && yaw_spread > max_yaw) {
      m.precision(0);
      m << ppBody(k) << "'s yaw varies by " << yaw_spread << " deg (> " << max_yaw << " deg) over "
        << pts.size() << " samples from " << pp_topic_[k]
        << ": its orientation is flipping between two solutions (or the body is turning) -- "
           "make the marker layout asymmetric, or press Get again";
      return refuse();
    }
    out->mean = mean;
    out->yaw = std::atan2(ys, yc);
    out->n = pts.size();
    out->spread = spread;
    return true;
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
      PpAverage a;
      if (!ppAverageBody(k, "NOT captured", &a, &resp->message)) {
        resp->success = false;
        return;
      }
      const bool was_planned = pp_wp_.has_value();
      if (k == kExecutePick) {
        // the pick capture has a parameter twin (2026-10-10): write it, as
        // onPpAdjust writes the drone points -- onPpParams sets the capture
        // from it and makes the plan stale, and the arm GS's Pick boxes follow
        // the parameter event
        const auto res = set_parameters({rclcpp::Parameter("pick_place_pick_point",
              std::vector<double>{a.mean(0), a.mean(1), a.mean(2), a.yaw * 180.0 / M_PI})});
        if (res.empty() || !res.front().successful) {
          resp->success = false;
          resp->message = "pick NOT captured: pick_place_pick_point not set" +
            (res.empty() ? std::string() : ": " + res.front().reason);
          return;
        }
      }
      pp_capture_[k] = a.mean;
      pp_capture_yaw_[k] = a.yaw;
      ppInvalidate();
      m.precision(3);
      m << "captured " << kPpPoint[k] << " at [" << a.mean(0) << ", " << a.mean(1) << ", " << a.mean(2)
        << "] m, yaw " << std::setprecision(1) << a.yaw * 180.0 / M_PI << " deg from " << pp_topic_[k]
        << " (" << a.n << " samples, spread " << a.spread * 1e3 << " mm)";
      if (k == kExecutePick) {m << "; the EE offset is applied in this object frame";}
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

  // GET PLACE (2026-10-10, operator request after the first hardware
  // pick-and-place flights). The place pose was measured by standing the
  // payload on the place platform, pressing the PICK row's Get and typing its
  // numbers into the Place row by hand. This does that copy: the PICK body
  // (obj_0, the payload) averaged exactly like a capture ->
  // pick_place_place_point x, y, z. The yaw is KEPT -- it is the drone's
  // heading at the place (typed, -180), not the payload's -- and z is the
  // payload's RESTING height: the operator lowers it by the release gap
  // (0.02 m) by hand, nothing is subtracted here. The pick capture is not
  // touched.
  void onPpGetPlace(Trigger::Response::SharedPtr resp)
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pp_flying_.has_value()) {
        resp->success = false;
        resp->message = "a pick-and-place leg is in flight -- get the place point once it holds";
        return;
      }
      const int k = kExecutePick;   // the payload's body
      if (pp_topic_[k].empty()) {
        resp->success = false;
        resp->message = "no mocap body for the payload -- set pick_place_pick_topic";
        return;
      }
      PpAverage a;
      if (!ppAverageBody(k, "Place NOT changed", &a, &resp->message)) {
        resp->success = false;
        return;
      }
      const auto old = get_parameter("pick_place_place_point").as_double_array();
      const double yaw_deg = old.size() == 4 ? old[3] : 0.0;   // a 3-value point is yaw 0
      const bool was_planned = pp_wp_.has_value();
      const auto res = set_parameters({rclcpp::Parameter("pick_place_place_point",
            std::vector<double>{a.mean(0), a.mean(1), a.mean(2), yaw_deg})});   // onPpParams: plan made stale
      if (res.empty() || !res.front().successful) {
        resp->success = false;
        resp->message = "Place NOT changed: pick_place_place_point not set" +
          (res.empty() ? std::string() : ": " + res.front().reason);
        return;
      }
      m.precision(3);
      m << "place point set to [" << a.mean(0) << ", " << a.mean(1) << ", " << a.mean(2)
        << "] m from " << pp_topic_[k] << " (" << a.n << " samples, spread "
        << std::setprecision(1) << a.spread * 1e3 << " mm); yaw kept at " << yaw_deg
        << " deg; z is the payload's RESTING height -- lower it by 0.02 m for the release";
      if (pp_capture_[kExecutePlace].has_value()) {
        m << "; NOTE a capture_place capture is active and wins over this point";
      }
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
      const double max_off = get_parameter("pick_place_adjust_max").as_double();
      if (off.norm() > max_off) {
        m << "offset [" << off(0) << ", " << off(1) << ", " << off(2) << "] m is larger than "
          << max_off << " m -- is the vehicle on the START mark? Not adjusted";
        resp->success = false;
        resp->message = m.str();
        return;
      }
      // WRITE the shift into the four drone-point parameters themselves
      // (2026-10-03, user request): the arm GS's typed fields then show exactly
      // what the legs fly to, and a second Adjust with the vehicle unmoved
      // measures ~0 against the moved Start. Place is never moved.
      std::vector<rclcpp::Parameter> moved;
      // (the drop exit is an offset from the place point: not moved either)
      for (const char * name : {"pick_place_start", "pick_place_place_start",
          "pick_place_land_start", "pick_place_land"})
      {
        auto v = get_parameter(name).as_double_array();
        if (v.size() != 4) {
          resp->success = false;
          resp->message = std::string(name) + " needs 4 numbers -- not adjusted";
          return;
        }
        v[0] += off(0);
        v[1] += off(1);
        v[2] += off(2);
        v[3] += yoff * 180.0 / M_PI;   // NOT wrapped: 0 -> 180 -> 360 is a typed turn direction
        moved.emplace_back(name, v);
      }
      const auto results = set_parameters(moved);   // onPpParams: checked, plan made stale
      for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].successful) {
          resp->success = false;
          resp->message = moved[i].get_name() + " not moved: " + results[i].reason;
          return;
        }
      }
      pp_offset_ = off;          // the LAST shift applied -- shown, not added again
      pp_yaw_offset_ = yoff;
      ppInvalidate();
      m << "drone points moved by [" << off(0) << ", " << off(1) << ", " << off(2) << "] m, yaw "
        << std::setprecision(1) << yoff * 180.0 / M_PI
        << " deg (written into Start, Place Start, Land Start, Land; Place and the drop exit not moved); press Plan";
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
      if (teleopActive()) {
        resp->success = false;
        resp->message = "PS4 teleoperation engaged -- disengage it first";
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
      pp_aborted_ = false;
      pp_guard_tripped_ = false;
    }
    publishPpStatus();
    publishPpPaths(nullptr);
    if (pp_worker_.joinable()) {pp_worker_.join();}
    pp_worker_ = std::thread([this, tg, cfg, opts, from, gen]() {
        std::optional<PickPlaceWaypoints> wp;
        std::vector<std::shared_ptr<const Trajectory>> legs;
        std::array<std::shared_ptr<const Trajectory>, kNumPickPlaceLegs> exits;
        std::shared_ptr<Trajectory> whole;
        std::string err;
        try {
          wp = pickPlaceWaypoints(*vehicle_, cfg, tg);
          legs = planPickPlaceMission(*vehicle_, *planner_, opts, cfg, *wp, from, &exits);
          std::vector<std::shared_ptr<const Trajectory>> all;   // drawn: each exit after its leg
          for (int k = 0; k < kNumPickPlaceLegs; ++k) {
            all.push_back(legs[k]);
            if (exits[k]) {all.push_back(exits[k]);}
          }
          whole = std::make_shared<SequenceTrajectory>(all);
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
            clearDropAim();               // the fresh plan's drop rest is the model's
            for (int k = 0; k < kNumPickPlaceLegs; ++k) {
              pp_leg_T_[k] = legs[k]->duration();
              pp_exit_T_[k] = exits[k] ? exits[k]->duration() : 0.0;
              RCLCPP_INFO(
                get_logger(), "pick-and-place %s: %s", pickPlaceLegName(k),
                legs[k]->diag().summary.c_str());
            }
            RCLCPP_INFO(get_logger(), "pick-and-place planned: %.1f s of flight in six legs and two exits.",
              whole->duration());
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
    std::string aim;          // the drop's aim at the basket, for the reply
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::string err;
      if (!mode_direct_ || !hold_.has_value()) {
        err = "not in whole-body DIRECT";
      } else if (pp_planning_) {
        err = "the mission is still planning";
      } else if (pp_flying_.has_value()) {
        err = std::string(pickPlaceLegName(*pp_flying_)) +
          (pp_phase_ == kPpWait ? " is waiting above its target" : " is in flight") +
          " -- Abort to cancel it";
      } else if (!pp_wp_.has_value()) {
        err = "not planned -- capture the pick / place points and press Plan";
      } else if (teleopActive()) {
        err = "PS4 teleoperation engaged -- disengage it first";
      } else if (state_ == "EXECUTING" || state_ == "CALCULATING") {
        err = "planner is " + state_ + " -- wait for HOLD";
      } else if (leg > pp_completed_ + 1) {
        err = std::string("out of order -- the next leg is ") + pickPlaceLegName(pp_completed_ + 1);
      } else if (!odom_p_.has_value() || secondsSince(odom_p_time_) > kOdomFreshS) {
        err = "no fresh odometry -- a leg must not start blind";
      } else {
        // the vehicle must actually be at its hold (audit M1): a leg started
        // while it still swings adds that error to the new reference's start
        const double d = (*odom_p_ - hold_->x_b).norm();
        const double tol = get_parameter("pick_place_leg_start_tol").as_double();
        if (d > tol) {
          std::ostringstream m;
          m.setf(std::ios::fixed);
          m.precision(0);
          m << "the body is " << d * 1e3 << " mm from its hold (> " << tol * 1e3
            << " mm) -- wait for it to settle";
          err = m.str();
        }
      }
      if (err.empty()) {ppConfig(&cfg, &err);}
      if (!err.empty()) {
        resp->success = false;
        resp->message = err;
        return;
      }
      from = *hold_;
      if (leg == kExecutePlace) {         // aim the drop rest at the measured basket
        applyDropAim();
        aim = pp_aim_text_;
      }
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
      pp_flying_ = leg;
      pp_phase_ = kPpTravel;
      pp_exited_[leg] = false;
      if (leg == kExecutePlace) {pp_drop_hold_t0_.reset();}
      resetDropAuto();
      pp_exec_plan_.reset();
      pp_error_.clear();
      pp_timed_out_ = false;
      pp_aborted_ = false;
      pp_guard_tripped_ = false;
      restoreAnchor();
    }
    const bool from_table = pp_at_table_;
    publishStatus();
    publishPpStatus();
    publishPpInfo();
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, leg, from, wp, cfg, opts, gen, from_table]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          traj = planPickPlaceLeg(*vehicle_, *planner_, opts, cfg, wp, leg, from, from_table);
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
            pp_phase_ = kPpNone;
            RCLCPP_WARN(get_logger(), "pick-and-place leg refused: %s", err.c_str());
          } else {
            plan_ = traj;
            pp_exec_plan_ = traj;
            exec_t0_ = Clock::now();
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
    if (!aim.empty()) {resp->message += " -- " + aim;}
    RCLCPP_INFO(get_logger(), "pick-and-place: %s", resp->message.c_str());
  }

  // ABORT: arm home and the vehicle pick_place_abort_climb higher, x, y and
  // heading kept, planned and flown at once. From a hold it starts there; in
  // the middle of a run it CUTS the run at the current reference point: the
  // CoM stays continuous (x_b is taken so that x_c matches), the reference's
  // velocity steps to zero once -- the trade the EE-trajectory pause makes --
  // and the cut point is held for the milliseconds the plan takes. The leg
  // that was cut is not complete: re-fly it or Reset. The gripper is the
  // arm GS's (its Abort button opens it on the same press).
  void onPpAbort(Trigger::Response::SharedPtr resp)
  {
    RestSpec from, goal;
    PlanOptions opts;
    unsigned gen = 0;
    bool cut = false;
    double climb = 0.0;
    std::optional<VecN> release;
    double release_hold = 0.0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {
        resp->success = false;
        resp->message = "not in whole-body DIRECT -- use the drone ground station";
        return;
      }
      {
        const auto rp = get_parameter("pick_place_abort_release_pose_deg").as_double_array();
        if (rp.size() == static_cast<std::size_t>(kNumJoints)) {
          VecN q;
          for (int j = 0; j < kNumJoints; ++j) {q(j) = rp[j] * M_PI / 180.0;}
          release = q;
          release_hold = std::max(0.0, get_parameter("pick_place_abort_release_hold_s").as_double());
        }
      }
      pp_abort_release_ = release.has_value();
      if (pp_abort_plan_) {
        resp->success = false;
        resp->message = "already aborting";
        return;
      }
      from = *hold_;
      std::optional<WbReference> cut_ref;
      if (state_ == "EXECUTING" && plan_ && exec_t0_.has_value()) {
        const double t = std::chrono::duration<double>(Clock::now() - *exec_t0_).count();
        cut_ref = plan_->eval(std::min(t, plan_->duration()));
      } else if (teleopActive() && teleop_last_ref_.has_value()) {
        // PS4 teleop (the Pick & Place PS4 fine correction included): hold_
        // is still the pose teleop started from, so cut at the reference
        // teleop is streaming, exactly as for a run -- else the reference
        // would step back by however far the pad has moved the vehicle
        cut_ref = *teleop_last_ref_;
        teleop_.reset();
        teleop_last_ref_.reset();
      }
      if (cut_ref.has_value()) {
        const WbReference & ref = *cut_ref;
        from.phi = std::atan2(ref.b1_d(1), ref.b1_d(0));
        from.q = ref.q_d;
        Vec3 r0c;
        armKinematics(ref.q_d, vehicle_->params, &r0c, nullptr, nullptr);
        from.x_b = ref.x_cd - Rz(from.phi) * r0c;   // x_c continuous
        cut = true;
      }
      // the climb is clipped at the fence ceiling (audit H3): repeated presses
      // never take the vehicle out of the volume
      climb = std::max(0.0, std::min(
          get_parameter("pick_place_abort_climb").as_double(),
          get_parameter("pick_place_fence_max_z").as_double() - from.x_b(2)));
      goal.x_b = from.x_b + Vec3{0.0, 0.0, climb};
      pp_abort_climb_ = climb;
      goal.phi = from.phi;
      goal.q = home_pose_;
      opts = planOptions();
      // supersede whatever was running or planning, hold the cut point
      gen = ++plan_gen_;
      plan_.reset();
      goal_override_.reset();
      auto_send_ = false;
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      setHold(from);
      state_ = "CALCULATING";
      pp_flying_.reset();
      pp_phase_ = kPpNone;
      pp_exec_plan_.reset();
      pp_error_.clear();
      pp_timed_out_ = false;
      pp_aborted_ = true;
      pp_aborting_ = true;
      pp_exit_plan_.reset();
      pp_exiting_.reset();
      restoreAnchor();                  // the climb is a flight motion: CoM anchor
    }
    publishStatus();
    publishPpStatus();
    publishPpInfo();
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, from, goal, opts, gen, release, release_hold]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          traj = planPickPlaceAbort(*vehicle_, *planner_, opts, from, goal,
            release.has_value() ? &*release : nullptr, release_hold);
        } catch (const std::exception & e) {
          err = e.what();
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != plan_gen_ || state_ != "CALCULATING") {return;}   // superseded
          pp_aborting_ = false;
          if (!traj) {
            state_ = "INFEASIBLE";
            infeasible_reason_ = "abort: " + err;
            pp_error_ = "abort refused: " + err + " -- holding where it was cut";
            RCLCPP_ERROR(get_logger(), "pick-and-place ABORT could not plan: %s", err.c_str());
          } else {
            plan_ = traj;
            pp_abort_plan_ = traj;
            exec_t0_ = Clock::now();
            state_ = "EXECUTING";
            RCLCPP_WARN(get_logger(), "pick-and-place ABORT: %s", traj->diag().summary.c_str());
          }
        }
        publishStatus();
        publishPpStatus();
        publishVizPath(traj.get());
      });
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "ABORT: " << (release.has_value() ? "arm to the release pose, then home," : "arm home")
      << " and up " << climb << " m from " << (cut ? "the point the run was cut at" : "the hold");
    if (climb < get_parameter("pick_place_abort_climb").as_double() - 1e-9) {
      m << " (clipped at the fence ceiling " << get_parameter("pick_place_fence_max_z").as_double() << " m)";
    }
    resp->success = true;
    resp->message = m.str();
    RCLCPP_WARN(get_logger(), "pick-and-place: %s", resp->message.c_str());
  }

  // EXIT (exit_pick / exit_place, 2026-10-03, user design): from the hold on a
  // claw target -- the gripper just closed (pick) or opened (place) on the arm
  // GS -- straight up by the safety margin (pick_place_approach_dz) to the
  // clearance point, heading and arm unchanged, CoM anchor (a flight motion).
  // The leg after it then starts there, with no retreat.
  void onPpExit(int leg, Trigger::Response::SharedPtr resp)
  {
    RestSpec from;
    PickPlaceConfig cfg;
    PlanOptions opts;
    std::optional<PickPlaceWaypoints> wp;
    double drop_shift = -1.0;   // the drop's exit: the vehicle's offset from the hold [m]
    unsigned gen = 0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::string err;
      if (!mode_direct_ || !hold_.has_value()) {
        err = "not in whole-body DIRECT";
      } else if (pp_planning_) {
        err = "the mission is still planning";
      } else if (pp_flying_.has_value() || pp_exiting_.has_value() || pp_aborting_ || pp_abort_plan_) {
        err = "a pick-and-place motion is in flight -- wait for it, or Abort";
      } else if (teleopActive()) {
        err = "PS4 teleoperation engaged -- disengage it first";
      } else if (state_ == "EXECUTING" || state_ == "CALCULATING") {
        err = "planner is " + state_ + " -- wait for HOLD";
      } else if (pp_completed_ != leg || !pp_at_table_) {
        err = std::string("not holding on the ") + (leg == kExecutePick ? "pick" : "place") +
          " target -- fly " + pickPlaceLegName(leg) + " first";
      } else if (!odom_p_.has_value() || secondsSince(odom_p_time_) > kOdomFreshS) {
        err = "no fresh odometry -- a climb must not start blind";
      }
      if (err.empty()) {ppConfig(&cfg, &err);}
      if (err.empty() && pickPlaceExitEndsAtLandStart(cfg, leg) && !pp_wp_.has_value()) {
        err = "no plan -- press Plan (the exit flies on to Land Start)";
      }
      if (!err.empty()) {
        resp->success = false;
        resp->message = std::string(ppExitName(leg)) + ": " + err;
        return;
      }
      if (pp_wp_.has_value()) {wp = *pp_wp_;}
      pp_exit_to_land_ = pickPlaceExitEndsAtLandStart(cfg, leg);
      from = *hold_;
      // THE DROP'S EXIT STARTS WHERE THE VEHICLE IS (2026-10-15, user: the claw
      // must not come back over the container): releasing the payload throws
      // the vehicle back 0.1-0.15 m (measured in Isaac), and an exit planned
      // from the hold would first pull it back to the release point. Heading
      // and arm stay the hold's; odometry is fresh (checked above).
      if (cfg.place_drop && leg == kExecutePlace && odom_p_.has_value()) {
        drop_shift = (*odom_p_ - from.x_b).norm();
        from.x_b = *odom_p_;
      }
      opts = planOptions();
      gen = ++plan_gen_;
      state_ = "CALCULATING";
      plan_.reset();
      goal_override_.reset();
      auto_send_ = false;
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      pp_exiting_ = leg;
      pp_exit_plan_.reset();
      pp_error_.clear();
      pp_timed_out_ = false;
      pp_aborted_ = false;
      pp_guard_tripped_ = false;
      restoreAnchor();                  // the climb is a flight motion: CoM anchor
    }
    publishStatus();
    publishPpStatus();
    publishPpInfo();
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, leg, from, cfg, opts, gen, wp]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          traj = planPickPlaceExit(*vehicle_, *planner_, opts, cfg, leg, from, wp ? &*wp : nullptr);
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
            pp_exiting_.reset();
            pp_exit_to_land_ = false;
            RCLCPP_WARN(get_logger(), "pick-and-place exit refused: %s", err.c_str());
          } else {
            plan_ = traj;
            pp_exit_plan_ = traj;
            exec_t0_ = Clock::now();
            state_ = "EXECUTING";
            RCLCPP_INFO(get_logger(), "pick-and-place %s: executing -- %s", ppExitName(leg),
              traj->diag().summary.c_str());
          }
        }
        publishStatus();
        publishPpStatus();
        publishPpInfo();
        publishVizPath(traj.get());
      });
    resp->success = true;
    if (pickPlaceExitEndsAtLandStart(cfg, leg)) {
      std::ostringstream dm;
      dm.setf(std::ios::fixed);
      dm.precision(2);
      const Vec3 & ls = wp->goal[kGoToLandStart].x_b;
      dm << ppExitName(leg) << ": climbing " << cfg.drop_exit_climb << " m, then flying straight on to "
         << "Land Start [" << ls(0) << ", " << ls(1) << ", " << ls(2) << "] m, the arm folding home -- "
         << pickPlaceLegName(kGoToLandStart) << " is then done";
      resp->message = dm.str();
      RCLCPP_INFO(get_logger(), "pick-and-place: %s", resp->message.c_str());
      return;
    }
    if (leg == kExecutePlace && cfg.place_drop && cfg.drop_exit_set) {
      std::ostringstream dm;
      dm.setf(std::ios::fixed);
      dm.precision(2);
      dm << ppExitName(leg) << ": climbing " << cfg.drop_exit_climb << " m, then flying level out of "
         << "the place region to the drop exit [" << cfg.drop_exit.p(0) << ", " << cfg.drop_exit.p(1)
         << "] m, heading " << std::setprecision(0) << cfg.drop_exit.yaw * 180.0 / M_PI
         << " deg (the arm folds on the way to Land Start)";
      resp->message = dm.str();
      RCLCPP_INFO(get_logger(), "pick-and-place: %s", resp->message.c_str());
      return;
    }
    const double back_d = pickPlaceExitBack(cfg);
    const bool back_out = leg == kExecutePlace && back_d > 0.0;
    resp->message = std::string(ppExitName(leg)) + ": " +
      (drop_shift >= 0.0 ? "from where the vehicle is now (" +
       std::to_string(static_cast<int>(std::lround(drop_shift * 1000.0))) +
       " mm from the release hold), " : std::string()) +
      (back_out ? "backing out " + std::to_string(static_cast<int>(std::lround(back_d * 100.0))) +
       (cfg.place_drop ? " cm (away from the drop point), then " : " cm (out from under the arch), then ")
       : std::string()) + "climbing " +
      std::to_string(static_cast<int>(std::lround(cfg.approach_dz * 100.0))) +
      " cm straight up to the clearance point";
    RCLCPP_INFO(get_logger(), "pick-and-place: %s", resp->message.c_str());
  }

  static const char * ppExitName(int leg) {return leg == kExecutePick ? "exit_pick" : "exit_place";}

  // What the descent gate checks (lock_ held): the claw's current error, less
  // the trim when it is on (the steady part the descent will cancel).
  double descentResidual() const
  {
    if (pp_trim_buf_.empty()) {return pp_last_err_;}
    return (pp_trim_buf_.back().second - descentTrim()).norm();
  }

  // The descent trim (lock_ held): the mean claw error over the hover window,
  // zero when disabled or with under half a window of samples.
  Vec3 descentTrim() const
  {
    if (!get_parameter("pick_place_descent_trim").as_bool() || pp_trim_buf_.size() < 2) {
      return Vec3::Zero();
    }
    const double span = std::chrono::duration<double>(
      pp_trim_buf_.back().first - pp_trim_buf_.front().first).count();
    if (span < 0.5 * get_parameter("pick_place_descent_trim_window_s").as_double()) {
      return Vec3::Zero();
    }
    Vec3 m = Vec3::Zero();
    for (const auto & s : pp_trim_buf_) {m += s.second;}
    return m / static_cast<double>(pp_trim_buf_.size());
  }

  // DESCENT ON REQUEST (descend_pick / descend_place, the arm GS's Pick /
  // Place button): from the hold above the target straight down onto it, the
  // claw leg then completing there as before. Refused unless the leg is
  // holding above its target with the claw within pick_place_arrival_tol of
  // the point above it.
  void onPpDescend(int leg, Trigger::Response::SharedPtr resp)
  {
    const char * name = leg == kExecutePick ? "descend_pick" : "descend_place";
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::string err;
      const double tol = get_parameter("pick_place_arrival_tol").as_double();
      if (!mode_direct_ || !hold_.has_value()) {
        err = "not in whole-body DIRECT";
      } else if (leg == kExecutePlace && get_parameter("pick_place_place_drop").as_bool()) {
        err = "drop place (pick_place_place_drop): no descent -- Ready To Place ends on the drop "
          "point, Place opens the gripper there";
      } else if (!get_parameter("pick_place_descent_on_request").as_bool()) {
        err = "the descent starts by itself (pick_place_descent_on_request is false)";
      } else if (!pp_flying_.has_value() || *pp_flying_ != leg || pp_phase_ != kPpWait ||
        state_ != "HOLD")
      {
        err = std::string("not holding above the ") + (leg == kExecutePick ? "pick" : "place") +
          " target -- fly " + pickPlaceLegName(leg) + " (Ready) first";
      } else if (teleopActive()) {
        err = "PS4 teleoperation engaged -- disengage it first";
      } else if (!(descentResidual() <= tol)) {
        // with the trim, a STEADY offset is fine (the trim cancels it): what
        // must be inside the tolerance is the claw's deviation from its own
        // average -- a claw still swinging waits
        std::ostringstream m;
        m.setf(std::ios::fixed);
        m.precision(0);
        m << "the claw is " << descentResidual() * 1e3 << " mm from "
          << (get_parameter("pick_place_descent_trim").as_bool() ? "its trimmed point above the target"
                                                                 : "the point above the target")
          << " (> " << tol * 1e3 << " mm) -- wait for it to settle";
        err = m.str();
      } else if (descentTrim().norm() > get_parameter("pick_place_descent_trim_max").as_double()) {
        std::ostringstream m;
        m.setf(std::ios::fixed);
        m.precision(0);
        m << "the claw holds " << descentTrim().norm() * 1e3 << " mm off the point above the target "
          << "on average (> pick_place_descent_trim_max " << get_parameter("pick_place_descent_trim_max").as_double() * 1e3
          << " mm) -- not trimming that far";
        err = m.str();
      }
      if (!err.empty()) {
        resp->success = false;
        resp->message = std::string(name) + ": " + err;
        return;
      }
    }
    startPpDescent();
    resp->success = true;
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      m << name << ": descending onto the target (goal trimmed by [" << pp_descent_trim_(0) * 1e3 << ", "
        << pp_descent_trim_(1) * 1e3 << ", " << pp_descent_trim_(2) * 1e3 << "] mm, the hover error)";
    }
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "pick-and-place: %s", resp->message.c_str());
  }

  // Ask the whole-body node for the CoM (com = true) or world anchor. Fire and
  // forget: the node blends over its own wb_ee_anchor_blend_s, refuses outside
  // DIRECT, and restores its yaml anchor on every mode change by itself.
  void requestAnchor(bool com)
  {
    pp_anchor_world_ = !com;
    if (!anchor_cli_ || !anchor_cli_->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "EE-anchor service %s not available -- anchor unchanged",
        anchor_cli_ ? anchor_cli_->get_service_name() : "?");
      return;
    }
    auto req = std::make_shared<SetBool::Request>();
    req->data = com;
    anchor_cli_->async_send_request(req);
    RCLCPP_INFO(get_logger(), "pick-and-place: EE anchor -> %s", com ? "CoM" : "WORLD");
  }

  // Any new motion that is not a claw leg's own wait / descent / grasp
  // returns to the CoM anchor.
  void restoreAnchor()
  {
    if (pp_anchor_world_) {requestAnchor(true);}
  }

  void onPpReset(Trigger::Response::SharedPtr resp)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pp_flying_.has_value()) {
        resp->success = false;
        resp->message = "a pick-and-place leg is in flight or waiting above its target -- Abort it first";
        return;
      }
      pp_completed_ = -1;
      pp_aborted_ = false;
      pp_guard_tripped_ = false;
      pp_at_table_ = false;
      pp_exited_.fill(false);
      pp_drop_hold_t0_.reset();
      resetDropAuto();
      ppInvalidate();
      restoreAnchor();
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
    if (pp_params_dirty_.exchange(false)) {
      publishPpStatus();
      publishPpInfo();
      publishPpPaths(nullptr);
    }
    Float64MultiArray m;
    bool gate_changed = false, descend = false, timed_out = false;
    std::string drop_now;     // the autonomous drop's release message, published below
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !pp_wp_.has_value()) {return;}
      const int leg = pp_flying_.has_value() ? *pp_flying_ : pp_completed_;
      if (leg < 0) {return;}
      Vec3 x_b;
      Mat3 r0;
      if (!odomPair(&x_b, &r0) || odomAge() > kOdomFreshS || !q_meas_.has_value()) {return;}
      sampleDropAim(x_b, r0);          // the basket - claw offset, while holding at Drop Start
      const bool claw = pickPlaceLegIsClaw(leg);
      // before / while waiting the claw's target is the point ABOVE the object
      const bool above = pp_flying_.has_value() && (pp_phase_ == kPpTravel || pp_phase_ == kPpWait);
      Vec3 meas = x_b, want = pp_wp_->goal[leg].x_b;
      if (claw) {
        Vec3 r0e;
        armKinematics(*q_meas_, vehicle_->params, nullptr, &r0e, nullptr);
        meas = x_b + r0 * (vehicle_->r_model * r0e);
        want = above ? pp_wp_->approach_ee[leg] : pp_wp_->ee[leg];
      }
      const Vec3 e = want - meas;
      // the descent trim's samples: the claw's error while holding above
      if (claw && pp_phase_ == kPpWait && state_ == "HOLD") {
        const auto now = Clock::now();
        pp_trim_buf_.push_back({now, e});
        const double win = get_parameter("pick_place_descent_trim_window_s").as_double();
        while (!pp_trim_buf_.empty() &&
          std::chrono::duration<double>(now - pp_trim_buf_.front().first).count() > win)
        {
          pp_trim_buf_.pop_front();
        }
      } else if (pp_phase_ != kPpWait) {
        pp_trim_buf_.clear();
      }
      const double tol = get_parameter("pick_place_arrival_tol").as_double();
      const bool within = e.norm() <= tol;
      // a DWELL inside the tolerance (audit L1): the claw settles THROUGH it
      if (!within) {
        pp_within_since_.reset();
      } else if (!pp_within_since_.has_value()) {
        pp_within_since_ = Clock::now();
      }
      const bool dwelt = within &&
        secondsSince(pp_within_since_) >= get_parameter("pick_place_settle_s").as_double();
      const bool settled = dwelt && !pp_flying_.has_value() && state_ == "HOLD";
      pp_last_err_ = e.norm();
      if (claw && evalDropAuto(meas, e, &drop_now)) {gate_changed = true;}
      if (pp_phase_ == kPpWait && state_ == "HOLD" &&
        !get_parameter("pick_place_descent_on_request").as_bool())
      {
        const double wait_max = get_parameter("pick_place_approach_wait_max").as_double();
        if (dwelt) {
          descend = true;
        } else if (wait_max > 0.0 && secondsSince(pp_wait_t0_) > wait_max) {
          std::ostringstream w;
          w.setf(std::ios::fixed);
          w.precision(0);
          w << pickPlaceLegName(leg) << ": the claw stayed " << e.norm() * 1e3
            << " mm from the point above the target for " << wait_max
            << " s -- descent not started; re-fly it or Abort";
          pp_error_ = w.str();
          pp_timed_out_ = true;
          pp_flying_.reset();
          pp_phase_ = kPpNone;
          gate_changed = true;
          timed_out = true;
          restoreAnchor();              // hovering above the target, nothing touched
          RCLCPP_WARN(get_logger(), "pick-and-place %s", pp_error_.c_str());
        }
      }
      m.data = {static_cast<double>(leg), claw ? 1.0 : 0.0, e.norm(), e(0), e(1), e(2), tol,
        within ? 1.0 : 0.0, settled ? 1.0 : 0.0, static_cast<double>(pp_phase_)};
      if (settled != pp_settled_ || within != pp_within_ || leg != pp_err_leg_ ||
        pp_phase_ == kPpWait)
      {
        pp_settled_ = settled;
        pp_within_ = within;
        pp_err_leg_ = leg;
        gate_changed = true;
      }
    }
    pp_arrival_pub_->publish(m);
    if (!drop_now.empty()) {
      String d;
      d.data = drop_now;
      pp_drop_now_pub_->publish(d);
    }
    if (descend) {startPpDescent();}
    if (gate_changed) {publishPpStatus();}
    if (timed_out) {publishPpInfo();}
  }

  // Called under the lock: a new drop hold (or none) -- nothing released yet.
  void resetDropAuto()
  {
    pp_drop_ok_since_.reset();
    pp_drop_fired_ = false;
    pp_drop_claw_prev_.reset();
    pp_drop_last_eval_.reset();
    pp_drop_text_.clear();
  }

  // ── AIM THE DROP AT THE BASKET (pick_place_drop_aim_payload) ───────────────
  // Forget any applied shift WITHOUT touching pp_wp_: called where pp_wp_ is
  // reset or replaced by a fresh plan, whose drop rest is the model's.
  void clearDropAim()
  {
    pp_aim_shift_.setZero();
    pp_aim_active_ = false;
    pp_aim_text_.clear();
    pp_aim_buf_.clear();
    pp_aim_hold_since_.reset();
  }

  // The newest tracked sample of a mocap body: un-flipped (tilt < 90 deg),
  // under 0.25 s old and not frozen (>= 3 bit-identical positions in 0.3 s)
  // -- the autonomous drop's own tests: its position, or none.
  std::optional<Vec3> ppFreshSample(int k) const
  {
    const auto & d = pp_mocap_[k];
    const PpSample * last = nullptr;
    for (auto it = d.rbegin(); it != d.rend(); ++it) {
      if (it->tilt < 0.5 * M_PI) {last = &*it; break;}
    }
    if (last == nullptr || secondsSince(last->t) > 0.25) {return std::nullopt;}
    size_t n_recent = 0;
    bool frozen = true;
    for (auto it = d.rbegin(); it != d.rend(); ++it) {
      if (it->tilt >= 0.5 * M_PI) {continue;}
      if (std::chrono::duration<double>(last->t - it->t).count() > 0.3) {break;}
      ++n_recent;
      if (!(it->p == last->p)) {frozen = false;}
    }
    if (n_recent >= 3 && frozen) {return std::nullopt;}
    return last->p;
  }

  // One basket - claw sample at the 10 Hz arrival tick, while HOLDING at Drop
  // Start (go_to_place_start done, the drop not started) with the arm at the
  // DROP pose: q1..q3 within 5 deg of the drop rest's (q4 is the swing, free).
  // The offset is kept in the VEHICLE frame (yaw only), so it carries to the
  // drop's heading. Outside that hover the buffer is KEPT (a re-flown Ready To
  // Drop aims with the same window); only a new plan clears it (clearDropAim).
  // Samples start pick_place_drop_aim_settle_s after the hover began (the turn
  // to Drop Start excites the wrist swing); a sample needs fresh joints and a
  // fresh, tracked, un-flipped obj_0. Called under the lock.
  void sampleDropAim(const Vec3 & x_b, const Mat3 & r0)
  {
    const bool on = get_parameter("pick_place_drop_aim_payload").as_bool() &&
      get_parameter("pick_place_place_drop").as_bool() && pp_wp_.has_value() &&
      pp_completed_ == kGoToPlaceStart && !pp_flying_.has_value() && state_ == "HOLD" &&
      !pp_topic_[kExecutePick].empty();
    if (!on) {pp_aim_hold_since_.reset(); return;}
    if (!pp_aim_hold_since_.has_value()) {pp_aim_hold_since_ = Clock::now();}
    if (secondsSince(pp_aim_hold_since_) <
      get_parameter("pick_place_drop_aim_settle_s").as_double())
    {
      return;
    }
    if (!q_meas_.has_value() || secondsSince(q_meas_time_) > 0.25) {return;}
    const auto & qd = pp_wp_->goal[kExecutePlace].q;
    for (int j = 0; j < 3; ++j) {
      if (std::abs((*q_meas_)(j) - qd(j)) > 5.0 * M_PI / 180.0) {return;}
    }
    const std::optional<Vec3> obj = ppFreshSample(kExecutePick);
    if (!obj.has_value()) {return;}
    Vec3 r0e;
    armKinematics(*q_meas_, vehicle_->params, nullptr, &r0e, nullptr);
    const Vec3 claw = x_b + r0 * (vehicle_->r_model * r0e);
    const double psi = std::atan2(r0(1, 0), r0(0, 0));
    const Vec3 body = Eigen::AngleAxisd(-psi, Vec3::UnitZ()) * (*obj - claw);
    const auto now = Clock::now();
    pp_aim_buf_.emplace_back(now, body);
    const double win = get_parameter("pick_place_drop_aim_window_s").as_double();
    while (!pp_aim_buf_.empty() &&
      std::chrono::duration<double>(now - pp_aim_buf_.front().first).count() > win)
    {
      pp_aim_buf_.pop_front();
    }
  }

  // Shift a pick-and-place leg's rest (base and claw together: the arm pose
  // does not change) by a world offset. Called under the lock.
  void shiftLegRest(int leg, const Vec3 & d)
  {
    pp_wp_->goal[leg].x_b += d;
    pp_wp_->ee[leg] += d;
    pp_wp_->approach[leg].x_b += d;
    pp_wp_->approach_ee[leg] += d;
  }

  // At Ready To Drop: undo the last shift, then aim the drop rest so the
  // MEASURED basket lands on the drop point (x, y), or keep the model and say
  // why. Sets pp_aim_text_ for the leg's reply. Called under the lock.
  void applyDropAim()
  {
    shiftLegRest(kExecutePlace, -pp_aim_shift_);
    pp_aim_shift_.setZero();
    pp_aim_active_ = false;
    pp_aim_text_.clear();
    if (!get_parameter("pick_place_drop_aim_payload").as_bool() ||
      !get_parameter("pick_place_place_drop").as_bool())
    {
      return;
    }
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(0);
    const double win = get_parameter("pick_place_drop_aim_window_s").as_double();
    const double min_s = get_parameter("pick_place_drop_aim_min_s").as_double();
    const double settle = get_parameter("pick_place_drop_aim_settle_s").as_double();
    const double max_shift = get_parameter("pick_place_drop_aim_max_shift").as_double();
    // only the hover that has just ended counts: drop what is older than the window
    const auto now = Clock::now();
    while (!pp_aim_buf_.empty() &&
      std::chrono::duration<double>(now - pp_aim_buf_.front().first).count() > win)
    {
      pp_aim_buf_.pop_front();
    }
    const double span = pp_aim_buf_.size() < 2 ? 0.0 :
      std::chrono::duration<double>(pp_aim_buf_.back().first - pp_aim_buf_.front().first).count();
    const double newest = pp_aim_buf_.empty() ? 1e9 :
      std::chrono::duration<double>(now - pp_aim_buf_.back().first).count();
    // the arrival tick is 10 Hz: a full span with a hole in it is not enough
    const double need_n = 8.0 * min_s;
    Vec3 mean = Vec3::Zero();
    for (const auto & s : pp_aim_buf_) {mean += s.second;}
    if (!pp_aim_buf_.empty()) {mean /= static_cast<double>(pp_aim_buf_.size());}
    PickPlaceTargets tg;
    std::string terr;
    if (!(min_s > 0.0) || !(win >= min_s) || !(max_shift > 0.0) || !(settle >= 0.0)) {
      m.precision(2);
      m << "NOT aimed at the basket: the aim's parameters are invalid (need min_s > 0, window_s >= "
        "min_s, max_shift > 0, settle_s >= 0; have " << min_s << ", " << win << ", " << max_shift
        << ", " << settle << ") -- the EE-offset model is used";
    } else if (span < min_s || static_cast<double>(pp_aim_buf_.size()) < need_n || newest > 0.5) {
      m.precision(1);
      m << "NOT aimed at the basket: " << span << " s / " << pp_aim_buf_.size()
        << " basket samples at Drop Start (need " << min_s << " s, " << need_n
        << ", the newest under 0.5 s old: hover >= " << settle + min_s << " s with the arm at the "
        "drop pose and " << ppBody(kExecutePick) << " tracked) -- the EE-offset model is used";
    } else if (!mean.allFinite()) {
      m << "NOT aimed at the basket: the measured offset is not finite -- the EE-offset model is used";
    } else if (!ppTargets(&tg, &terr)) {
      m << "NOT aimed at the basket: no drop point (" << terr << ")";
    } else {
      const double psi = pp_wp_->goal[kExecutePlace].phi + 0.5 * M_PI;   // model heading -> actual yaw
      const Vec3 basket = pp_wp_->ee[kExecutePlace] + Eigen::AngleAxisd(psi, Vec3::UnitZ()) * mean;
      Vec3 d = Vec3::Zero();
      d.head<2>() = tg.place.head<2>() - basket.head<2>();
      if (!d.allFinite()) {
        m << "NOT aimed at the basket: the shift is not finite -- the EE-offset model is used";
      } else if (d.norm() > max_shift) {
        m << "NOT aimed at the basket: the measured basket would need a " << d.norm() * 1e3
          << " mm shift (> " << max_shift * 1e3 << " mm) -- the EE-offset model is used";
      } else {
        shiftLegRest(kExecutePlace, d);
        pp_aim_shift_ = d;
        pp_aim_active_ = true;
        m << "aimed at the basket: the drop rest moved [" << d(0) * 1e3 << ", " << d(1) * 1e3
          << "] mm (basket measured [" << mean(0) * 1e3 << ", " << mean(1) * 1e3
          << "] mm from the claw, vehicle frame, over " << std::setprecision(1) << span << " s)";
      }
    }
    pp_aim_text_ = m.str();
    RCLCPP_INFO(get_logger(), "pick-and-place execute_place: %s", pp_aim_text_.c_str());
  }

  // THE AUTONOMOUS DROP's gates (pick_place_drop_auto: see its parameters), at
  // the 10 Hz arrival tick while the drop leg holds on the drop point. `claw`:
  // the measured claw; `e`: its error to the drop target (world). Sets *fire to
  // the release message once every gate has held for the dwell (once per drop
  // hold). Returns true when the status text changed. Called under the lock.
  bool evalDropAuto(const Vec3 & claw, const Vec3 & e, std::string * fire)
  {
    const std::string before = pp_drop_text_;
    const bool holding = get_parameter("pick_place_drop_auto").as_bool() &&
      get_parameter("pick_place_place_drop").as_bool() && pp_wp_.has_value() &&
      pp_completed_ == kExecutePlace && !pp_flying_.has_value() && !pp_exiting_.has_value() &&
      !pp_exited_[kExecutePlace] && pp_at_table_ && !pp_aborting_ && !pp_abort_plan_ &&
      !pp_aborted_ && pp_drop_hold_t0_.has_value() && state_ == "HOLD";
    if (!holding) {
      pp_drop_ok_since_.reset();
      pp_drop_claw_prev_.reset();
      pp_drop_last_eval_.reset();
      if (!pp_drop_fired_) {pp_drop_text_.clear();}
      return pp_drop_text_ != before;
    }
    if (pp_drop_fired_) {return false;}          // released: the text stays as it was
    const auto now = Clock::now();
    // a gap in the 10 Hz evaluation (stale odometry / attitude / joints skip
    // the tick) restarts the dwell and the claw speed: never release on one
    // tick after a gap, nor on a speed averaged across it
    if (pp_drop_last_eval_.has_value() && secondsSince(pp_drop_last_eval_) > 0.25) {
      pp_drop_ok_since_.reset();
      pp_drop_claw_prev_.reset();
    }
    pp_drop_last_eval_ = now;
    const double wait = get_parameter("pick_place_drop_auto_wait_s").as_double();
    const double dwell = get_parameter("pick_place_drop_auto_dwell_s").as_double();
    const double claw_tol = get_parameter("pick_place_drop_auto_claw_tol").as_double();
    const double claw_vmax = get_parameter("pick_place_drop_auto_claw_speed").as_double();
    const double pay_tol = get_parameter("pick_place_drop_auto_payload_tol").as_double();
    const double pay_vmax = get_parameter("pick_place_drop_auto_payload_speed").as_double();
    const double q4_tol = get_parameter("pick_place_drop_auto_q4_tol_deg").as_double();

    // the claw's speed, between two arrival ticks
    double claw_v = std::numeric_limits<double>::quiet_NaN();
    if (pp_drop_claw_prev_.has_value()) {
      const double dt = std::chrono::duration<double>(now - pp_drop_claw_prev_->first).count();
      if (dt > 1e-3) {claw_v = (claw - pp_drop_claw_prev_->second).norm() / dt;}
    }
    pp_drop_claw_prev_ = std::make_pair(now, claw);

    // the payload's own mocap body: its newest UN-flipped sample, fresh, not
    // frozen, and its speed over ~0.1 s
    std::string pay_why;
    double pay_err = std::numeric_limits<double>::quiet_NaN();
    double pay_v = std::numeric_limits<double>::quiet_NaN();
    const auto & d = pp_mocap_[kExecutePick];
    const PpSample * last = nullptr;
    for (auto it = d.rbegin(); it != d.rend(); ++it) {
      if (it->tilt < 0.5 * M_PI) {last = &*it; break;}
    }
    PickPlaceTargets tg;
    std::string terr;
    if (pp_topic_[kExecutePick].empty()) {
      pay_why = "no payload body (pick_place_pick_topic is empty)";
    } else if (last == nullptr || secondsSince(last->t) > 0.25) {
      pay_why = ppBody(kExecutePick) + " not tracked (no fresh sample)";
    } else if (!ppTargets(&tg, &terr)) {
      pay_why = "no drop point: " + terr;
    } else {
      const PpSample * old = nullptr;
      size_t n_recent = 0;
      bool frozen = true;
      for (auto it = d.rbegin(); it != d.rend(); ++it) {
        if (it->tilt >= 0.5 * M_PI) {continue;}
        const double age = std::chrono::duration<double>(last->t - it->t).count();
        if (age <= 0.3) {
          ++n_recent;
          if (!(it->p == last->p)) {frozen = false;}
        }
        if (age >= 0.1) {old = &*it; break;}
      }
      if (n_recent >= 3 && frozen) {
        pay_why = ppBody(kExecutePick) + "'s pose is frozen (not tracked)";
      } else if (old == nullptr) {
        pay_why = ppBody(kExecutePick) + " not tracked long enough to measure its speed";
      } else {
        const double dt = std::chrono::duration<double>(last->t - old->t).count();
        pay_v = (last->p - old->p).norm() / dt;
        pay_err = (last->p.head<2>() - tg.place.head<2>()).norm();
      }
    }
    // the wrist roll against the place pose's
    const auto place_pose = get_parameter("pick_place_place_pose_deg").as_double_array();
    const double q4_ref = place_pose.size() == 4 ? place_pose[3] : 0.0;
    const double q4_err = std::abs((*q_meas_)(3) * 180.0 / M_PI - q4_ref);

    std::ostringstream m;
    m.setf(std::ios::fixed);
    m << "auto drop: ";
    const double held = secondsSince(pp_drop_hold_t0_);
    bool ok = false;
    const double q_age = secondsSince(q_meas_time_);
    const bool aimed = pp_aim_active_ && get_parameter("pick_place_drop_aim_payload").as_bool();
    if (held < wait) {
      m << std::setprecision(1) << "waiting " << held << " of " << wait << " s";
    } else if (q_age > 0.25) {
      m << std::setprecision(2) << "arm joint states " << q_age << " s old (the claw and q4 are unknown)";
    } else if (!aimed && e.norm() > claw_tol) {
      // aimed at the basket, the payload gate below IS the position gate: the
      // claw only has to be still (its target already carries the measured hang)
      m << std::setprecision(0) << "claw " << e.norm() * 1e3 << " mm from the drop target (gate "
        << claw_tol * 1e3 << ")";
    } else if (!(claw_v <= claw_vmax)) {
      m << std::setprecision(2) << "claw moving " << claw_v << " m/s (gate " << claw_vmax << ")";
    } else if (!pay_why.empty()) {
      m << pay_why;
    } else if (pay_err > pay_tol) {
      m << std::setprecision(0) << "payload " << pay_err * 1e3 << " mm from the drop point (gate "
        << pay_tol * 1e3 << ")";
    } else if (!(pay_v <= pay_vmax)) {
      m << std::setprecision(2) << "payload moving " << pay_v << " m/s (gate " << pay_vmax << ")";
    } else if (q4_err > q4_tol) {
      m << std::setprecision(1) << "wrist q4 " << q4_err << " deg off (gate " << q4_tol << ")";
    } else {
      ok = true;
    }
    if (!ok) {
      pp_drop_ok_since_.reset();
    } else {
      if (!pp_drop_ok_since_.has_value()) {pp_drop_ok_since_ = now;}
      const double inside = secondsSince(pp_drop_ok_since_);
      if (inside + 1e-9 < dwell) {
        m << std::setprecision(1) << "all gates OK for " << inside << " of " << dwell << " s";
      } else {
        std::ostringstream r;
        r.setf(std::ios::fixed);
        r << std::setprecision(0) << "claw " << e.norm() * 1e3 << " mm, payload " << pay_err * 1e3
          << " mm from the drop point, " << std::setprecision(2) << "claw " << claw_v
          << " m/s, payload " << pay_v << " m/s, " << std::setprecision(1) << "q4 " << q4_err
          << " deg off, " << held << " s after the leg";
        *fire = r.str();
        pp_drop_fired_ = true;
        m << "RELEASE signalled (" << r.str() << ") -- the arm GS opens the gripper";
        RCLCPP_INFO(get_logger(), "pick-and-place: AUTO DROP -- %s", r.str().c_str());
      }
    }
    pp_drop_text_ = m.str();
    return pp_drop_text_ != before;
  }

  // The claw leg's vertical descent from the approach hold onto the target,
  // started by publishPpArrival once the claw has dwelt within tolerance.
  void startPpDescent()
  {
    RestSpec from;
    PickPlaceWaypoints wp;
    PickPlaceConfig cfg;
    PlanOptions opts;
    unsigned gen = 0;
    int leg = -1;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pp_phase_ != kPpWait || !pp_flying_.has_value() || !pp_wp_.has_value() ||
        state_ != "HOLD" || !hold_.has_value())
      {
        return;
      }
      std::string err;
      if (!ppConfig(&cfg, &err)) {
        pp_error_ = err;
        pp_flying_.reset();
        pp_phase_ = kPpNone;
        restoreAnchor();
        return;
      }
      leg = *pp_flying_;
      from = *hold_;
      wp = *pp_wp_;
      // the descent trim: shift the goal by the averaged hover error (the
      // arm pose is fixed, so moving the base moves the claw alike); the
      // arrival error stays measured against the TRUE target (pp_wp_)
      pp_descent_trim_ = descentTrim();
      wp.goal[leg].x_b += pp_descent_trim_;
      opts = planOptions();
      gen = ++plan_gen_;
      state_ = "CALCULATING";
      plan_.reset();
      pp_phase_ = kPpDescent;
      pp_exec_plan_.reset();
      RCLCPP_INFO(get_logger(), "pick-and-place %s: claw within tolerance above the target -- descending "
        "(goal trimmed by [%.1f, %.1f, %.1f] mm, the hover error).", pickPlaceLegName(leg),
        pp_descent_trim_(0) * 1e3, pp_descent_trim_(1) * 1e3, pp_descent_trim_(2) * 1e3);
    }
    publishStatus();
    publishPpStatus();
    publishPpInfo();
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, leg, from, wp, cfg, opts, gen]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          traj = planPickPlaceDescent(*vehicle_, *planner_, opts, cfg, wp, leg, from);
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
            pp_phase_ = kPpNone;
            restoreAnchor();            // hovering above the target, nothing touched
            RCLCPP_WARN(get_logger(), "pick-and-place descent refused: %s", err.c_str());
          } else {
            plan_ = traj;
            pp_exec_plan_ = traj;
            exec_t0_ = Clock::now();
            state_ = "EXECUTING";
          }
        }
        publishStatus();
        publishPpStatus();
        publishVizPath(traj.get());
      });
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
      } else if (pp_aborting_ || pp_abort_plan_) {
        m << "ABORTING" << (pp_guard_tripped_ ? " (SAFETY GUARD: " + pp_guard_reason_ + ")" : std::string())
          << " -- gripper open, arm " << (pp_abort_release_ ? "to the release pose then home" : "home")
          << ", climbing " << std::setprecision(2) << pp_abort_climb_ << " m";
      } else if (pp_planning_) {
        m << "PLANNING";
      } else if (pp_flying_.has_value() && pp_phase_ == kPpWait) {
        m << "WAITING " << pickPlaceLegName(*pp_flying_) << ": claw " << std::setprecision(0)
          << pp_last_err_ * 1e3 << " mm from the point above the target (tolerance "
          << get_parameter("pick_place_arrival_tol").as_double() * 1e3;
        if (get_parameter("pick_place_descent_on_request").as_bool()) {
          m << " mm) -- press " << (*pp_flying_ == kExecutePick ? "Pick" : "Place") << " to descend";
        } else {
          m << " mm) -- descends once inside";
        }
      } else if (pp_flying_.has_value()) {
        m << "FLYING " << pickPlaceLegName(*pp_flying_);
        if (pickPlaceLegIsClaw(*pp_flying_)) {
          m << (pp_phase_ == kPpDescent ? " (descent)" : " (approach)");
        }
        if (state_ == "EXECUTING" && plan_) {m << " T=" << plan_->duration() << "s";}
      } else if (pp_exiting_.has_value()) {
        m << "FLYING " << ppExitName(*pp_exiting_) <<
          (pp_exit_to_land_ ? " (take-off, then on to Land Start)" : " (up the vertical margin)");
        if (state_ == "EXECUTING" && plan_) {m << " T=" << plan_->duration() << "s";}
      } else if (!pp_error_.empty()) {
        m << (pp_timed_out_ ? "INCOMPLETE: " : "INFEASIBLE: ") << pp_error_;
      } else if (pp_aborted_) {
        m << "ABORTED" << (pp_guard_tripped_ ? " (SAFETY GUARD: " + pp_guard_reason_ + ")" : std::string())
          << " -- holding higher, arm home; ";
        if (!pp_wp_.has_value()) {
          m << "press Plan";
        } else if (pp_completed_ >= kNumPickPlaceLegs - 1) {
          m << "touch down with SAFETY -> land";
        } else {
          m << "re-fly " << pickPlaceLegName(pp_completed_ + 1) << " or Reset";
        }
      } else if (!pp_wp_.has_value()) {
        PickPlaceTargets t;
        std::string err;
        m << "NOT PLANNED: " << (ppTargets(&t, &err) ? std::string("press Plan") : err);
      } else if (pp_completed_ >= kNumPickPlaceLegs - 1) {
        m << "COMPLETE -- touch down with SAFETY -> land";
      } else if (pp_completed_ < 0) {
        m << "READY next=" << pickPlaceLegName(0);
      } else {
        const bool claw = pickPlaceLegIsClaw(pp_completed_);
        if (claw && pp_exited_[pp_completed_]) {
          m << "DONE " << ppExitName(pp_completed_) << " next=" << pickPlaceLegName(pp_completed_ + 1);
        } else if (claw && pp_at_table_) {
          m << "DONE " << pickPlaceLegName(pp_completed_) << " next="
            << (pp_completed_ == kExecutePick ? "exit_pick (close the gripper first)"
                                              : "exit_place (open the gripper first)");
        } else {
          m << "DONE " << pickPlaceLegName(pp_completed_) << " next="
            << pickPlaceLegName(pp_completed_ + 1);
        }
        if (claw && !pp_exited_[pp_completed_]) {
          m << (pp_settled_ ? " -- claw within tolerance: fine correction OK"
                : pp_within_ ? " -- claw inside tolerance, settling"
                             : " -- claw outside tolerance");
          if (!pp_drop_text_.empty()) {m << " -- " << pp_drop_text_;}
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
  // valid], [80] the captured pick object's yaw [deg] (2026-10-03; the frame
  // pick_place_ee_offset is applied in; [60..63] and [80] are also the
  // parameter pick_place_pick_point since 2026-10-10), [81] / [82] exit_pick / exit_place
  // planned duration [s], [83] / [84] exit_pick / exit_place flown (0 / 1),
  // [85] holding on a claw target (0 / 1), [86] the exit in flight (1 pick,
  // 3 place, -1 none), [87] the claw leg's phase (0 none, 1 to the point
  // above, 2 holding there, 3 the descent). NaN where there is nothing.
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
          m.data.end(), {g.x_b(0), g.x_b(1), g.x_b(2), (g.phi + 0.5 * M_PI) * 180.0 / M_PI,
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
      const auto & py = pp_capture_yaw_[kExecutePick];
      m.data.push_back(py.has_value() ? *py * 180.0 / M_PI : nan);
      for (int k : {kExecutePick, kExecutePlace}) {
        m.data.push_back(pp_wp_.has_value() ? pp_exit_T_[k] : nan);
      }
      for (int k : {kExecutePick, kExecutePlace}) {m.data.push_back(pp_exited_[k] ? 1.0 : 0.0);}
      m.data.push_back(pp_at_table_ ? 1.0 : 0.0);
      m.data.push_back(pp_exiting_.has_value() ? static_cast<double>(*pp_exiting_) : -1.0);
      m.data.push_back(static_cast<double>(pp_phase_));
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

  // ================================================== push-and-pull mode
  // Six steps (push_pull.hpp; five until 2026-10-16), one button each on the
  // arm GS's "Box Pushing" / "Valve Turning" tabs: go_to_start (the claw the safety margin straight above the box's
  // handle, the nose along the push), ready (straight down onto the handle --
  // the GS opens the gripper first), push (the GS has CLOSED the gripper on
  // the handle: the law's phase flag is CONTACT (since the end of ready), the
  // claw stays world-held (push_pull_world_anchor_push), and the base slides
  // the box push_distance along the nose
  // with the arm held still), exit (the GS has OPENED the gripper: the phase
  // flag back to free, straight up, arm home), go_to_land_start (along the
  // nose off the table, then an arc to Land Start, turning the typed way
  // round), land (straight to the land hover). The box is MEASURED (Get,
  // mocap obj_0: its CG and yaw); Land Start and the land point are typed and
  // moved by Adjust. Every step re-plans from the CURRENT hold and executes as
  // soon as it is planned; the steps go in order (go_to_start and
  // go_to_land_start whenever the claw is off the handle, land after
  // go_to_land_start).
  bool plConfig(PushPullConfig * c, std::string * err) const
  {
    c->home = home_pose_;
    if (!ppJoints("push_pull_push_pose_deg", &c->push_pose, err) ||
      !ppVec3("push_pull_ee_offset", &c->ee_offset, err))
    {
      return false;
    }
    c->yaw_from_box = get_parameter("push_pull_yaw_from_box_deg").as_double() * M_PI / 180.0;
    c->approach_dz = get_parameter("push_pull_approach_dz").as_double();
    c->approach_back = get_parameter("push_pull_approach_back").as_double();
    c->exit_dz = get_parameter("push_pull_exit_dz").as_double();
    c->push_distance = get_parameter("push_pull_push_distance").as_double();
    c->push_time = get_parameter("push_pull_push_time_s").as_double();
    c->push_settle = get_parameter("push_pull_push_settle_s").as_double();
    c->push_ramp = get_parameter("push_pull_push_ramp_s").as_double();
    c->exit_time = get_parameter("push_pull_exit_time_s").as_double();
    c->unload_time = get_parameter("push_pull_unload_time_s").as_double();
    c->unload_wait = get_parameter("push_pull_unload_wait_s").as_double();
    c->turn_angle = get_parameter("push_pull_turn_deg").as_double() * M_PI / 180.0;
    c->fence.min_z = get_parameter("push_pull_fence_min_z").as_double();
    c->fence.max_z = get_parameter("push_pull_fence_max_z").as_double();
    const auto xy = get_parameter("push_pull_fence_xy").as_double_array();
    if (xy.size() != 4) {
      *err = "push_pull_fence_xy needs [x_min, x_max, y_min, y_max]";
      return false;
    }
    c->fence.x_min = xy[0];
    c->fence.x_max = xy[1];
    c->fence.y_min = xy[2];
    c->fence.y_max = xy[3];
    const auto tb = get_parameter("push_pull_table").as_double_array();
    if (tb.size() != 5) {
      *err = "push_pull_table needs [x_min, x_max, y_min, y_max, z_top] (all 0 = no table check)";
      return false;
    }
    c->table.x_min = tb[0];
    c->table.x_max = tb[1];
    c->table.y_min = tb[2];
    c->table.y_max = tb[3];
    c->table.z_top = tb[4];
    const auto ob = get_parameter("push_pull_obstacle").as_double_array();
    if (ob.size() != 5) {
      *err = "push_pull_obstacle needs [x_min, x_max, y_min, y_max, z_top] (all 0 = none)";
      return false;
    }
    c->obstacle.x_min = ob[0];
    c->obstacle.x_max = ob[1];
    c->obstacle.y_min = ob[2];
    c->obstacle.y_max = ob[3];
    c->obstacle.z_top = ob[4];
    c->table_clearance = get_parameter("push_pull_table_clearance").as_double();
    c->table_margin_xy = get_parameter("push_pull_table_margin_xy").as_double();
    c->gear_depth = get_parameter("push_pull_gear_depth").as_double();
    c->gear_radius = get_parameter("push_pull_gear_radius").as_double();
    c->box_half_length = get_parameter("push_pull_box_half_length").as_double();
    c->box_edge_margin = get_parameter("push_pull_box_edge_margin").as_double();
    return true;
  }

  // The goals' inputs: the captured box and the typed (Adjust-shifted) land
  // hover and Land Start. Called under the lock.
  bool plTargets(PushPullTargets * t, std::string * err) const
  {
    if (!ppBase("push_pull_land", &t->land, err)) {return false;}
    t->land_start_set = !get_parameter("push_pull_land_start").as_double_array().empty();
    if (t->land_start_set && !ppBase("push_pull_land_start", &t->land_start, err)) {return false;}
    if (!pl_box_.has_value()) {
      *err = "Get the box first (" + (pl_box_topic_.empty()
        ? std::string("no push_pull_box_topic set") : "mocap " + pl_box_topic_) + ")";
      return false;
    }
    t->box = *pl_box_;
    t->box_yaw = pl_box_yaw_.value_or(0.0);
    return true;
  }

  rcl_interfaces::msg::SetParametersResult onPlParams(const std::vector<rclcpp::Parameter> & ps)
  {
    rcl_interfaces::msg::SetParametersResult r;
    r.successful = true;
    // only gate or report: the plan stays valid when these change
    static const char * const kGateOnly[] = {
      "push_pull_arrival_tol", "push_pull_leg_start_tol", "push_pull_adjust_max",
      "push_pull_abort_climb", "push_pull_abort_release_wait_s", "push_pull_guard_tilt_deg",
      "push_pull_abort_from_measured", "push_pull_abort_back", "push_pull_abort_release_world_held",
      "push_pull_guard_persist_s", "push_pull_guard_force_n", "push_pull_guard_force_persist_s",
      "push_pull_guard_box_lag", "push_pull_guard_box_tilt_deg", "push_pull_guard_box_persist_s",
      "push_pull_guard_box_stale_s", "push_pull_guard_q3_min_deg",
      "push_pull_descent_trim", "push_pull_descent_trim_window_s", "push_pull_descent_trim_max",
      "push_pull_world_anchor_ready", "push_pull_world_anchor_push", "push_pull_contact",
      "push_pull_world_anchor_on_handle",
      "push_pull_contact_at_ready", "push_pull_contact_off_at_push_end",
      "push_pull_capture_window",
      "push_pull_capture_max_spread", "push_pull_capture_max_yaw_spread_deg", "push_pull_start_mark",
      "push_pull_unload_avg_s", "push_pull_unload_max",
      "push_pull_follow", "push_pull_follow_filter_s", "push_pull_follow_lead_deg", "push_pull_follow_tilt_deg",
      "push_pull_follow_wrist_deg", "push_pull_follow_stale_s", "push_pull_follow_bandwidth",
      "push_pull_follow_recover_s", "push_pull_follow_hold_timeout_s"};
    bool stale = false;
    const rclcpp::Parameter * box_point = nullptr;
    for (const auto & p : ps) {
      const std::string & n = p.get_name();
      if (n.rfind("push_pull_", 0) != 0) {continue;}
      if (n == "push_pull_box_point") {
        // [] = not captured, else [x, y, z, yaw_deg], all finite (2026-10-16)
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
          r.successful = false;
          r.reason = n + " needs [x, y, z, yaw_deg], or [] for not captured";
          return r;
        }
        if (!plBoxPointOk(p.as_double_array(), &r.reason)) {
          r.successful = false;
          return r;
        }
        // the capture's own write-back (Get) or a clear after it was dropped:
        // nothing changes, the plan stays
        if (plBoxPointIsCapture(p.as_double_array())) {continue;}
        box_point = &p;
        stale = true;
        continue;
      }
      if (n == "push_pull_box_topic" || n == "push_pull_contact_service" ||
        n == "push_pull_force_topic")
      {
        r.successful = false;
        r.reason = n + " is fixed at launch (its subscription / client is made then)";
        return r;
      }
      size_t len = 0;
      if (n == "push_pull_land_start") {
        // [] = Land Start is the land pose, else [x, y, z, yaw_deg]
        if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY ||
          (!p.as_double_array().empty() && p.as_double_array().size() != 4))
        {
          r.successful = false;
          r.reason = n + " needs [x, y, z, yaw_deg], or [] for the land pose";
          return r;
        }
      }
      if (n == "push_pull_land" || n == "push_pull_start_mark" || n == "push_pull_push_pose_deg" ||
        n == "push_pull_fence_xy")
      {
        len = 4;
      } else if (n == "push_pull_ee_offset") {
        len = 3;
      } else if (n == "push_pull_table" || n == "push_pull_obstacle") {
        len = 5;
      }
      if (len > 0 && (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY ||
        p.as_double_array().size() != len))
      {
        r.successful = false;
        r.reason = n + " needs " + std::to_string(len) + " numbers";
        return r;
      }
      if (std::none_of(std::begin(kGateOnly), std::end(kGateOnly),
        [&n](const char * g) {return n == g;}))
      {
        stale = true;
      }
    }
    if (stale) {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pl_flying_.has_value()) {
        r.successful = false;
        r.reason = "a push-and-pull step is in flight -- edit once it holds";
        return r;
      }
      if (box_point != nullptr && (pl_aborting_ || pl_abort_plan_)) {
        r.successful = false;
        r.reason = "an abort is in flight -- edit the box once it holds";
        return r;
      }
      if (box_point != nullptr && pl_at_handle_) {
        r.successful = false;
        r.reason = "the box is in the gripper -- edit it before Ready, or after Exit";
        return r;
      }
      // (2026-10-19) ANY plan-invalidating edit while the claw is on the handle:
      // the plan would go, and with it Exit (it needs the plan) -- Plan and Reset
      // are refused on the handle too, so ABORT would be the only way out with
      // the jaws on the box. Gate-only keys (guards, tolerances) still pass.
      if (pl_at_handle_) {
        r.successful = false;
        r.reason = "the claw is on the handle -- this edit would drop the plan; change it after Exit";
        return r;
      }
      if (pl_wp_.has_value() || pl_planning_) {
        RCLCPP_INFO(get_logger(), "push-and-pull: parameters changed -- the plan is stale, press Plan");
      }
      // the box point IS the box capture (2026-10-16)
      if (box_point != nullptr) {plSetBoxCapture(box_point->as_double_array());}
      plInvalidate();
      pl_params_dirty_ = true;   // published from the 10 Hz timer, once the values are set
    }
    return r;
  }

  // push_pull_box_point: [] (not captured) or [x, y, z, yaw_deg], finite.
  static bool plBoxPointOk(const std::vector<double> & v, std::string * why)
  {
    if (!v.empty() && v.size() != 4) {
      *why = "push_pull_box_point needs [x, y, z, yaw_deg], or [] for not captured (got " +
        std::to_string(v.size()) + " numbers)";
      return false;
    }
    for (double x : v) {
      if (!std::isfinite(x)) {
        *why = "push_pull_box_point has a non-finite value";
        return false;
      }
    }
    return true;
  }

  // True when the value says what the capture already holds (both empty, or
  // the same pose to the micrometre / 1e-6 deg).
  bool plBoxPointIsCapture(const std::vector<double> & v)
  {
    std::lock_guard<std::recursive_mutex> lk(lock_);
    if (v.empty() || !pl_box_.has_value()) {return v.empty() && !pl_box_.has_value();}
    const Vec3 & b = *pl_box_;
    return std::abs(v[0] - b(0)) < 1e-6 && std::abs(v[1] - b(1)) < 1e-6 &&
      std::abs(v[2] - b(2)) < 1e-6 &&
      std::abs(v[3] - pl_box_yaw_.value_or(0.0) * 180.0 / M_PI) < 1e-6;
  }

  // The box capture from a (checked) push_pull_box_point value. Called under
  // the lock (or from the constructor). The yaw is taken as typed.
  void plSetBoxCapture(const std::vector<double> & v)
  {
    if (v.size() == 4) {
      pl_box_ = Vec3{v[0], v[1], v[2]};
      pl_box_yaw_ = v[3] * M_PI / 180.0;
    } else {
      pl_box_.reset();
      pl_box_yaw_.reset();
    }
  }

  void onPlMocap(const Mocap & msg)
  {
    const Vec3 p{msg.pose.position.x, msg.pose.position.y, msg.pose.position.z};
    if (!p.allFinite()) {return;}
    const auto & o = msg.pose.orientation;
    const double yaw = std::atan2(2.0 * (o.w * o.z + o.x * o.y), 1.0 - 2.0 * (o.y * o.y + o.z * o.z));
    bool trip = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      pl_mocap_.push_back(PpSample{Clock::now(), p, std::isfinite(yaw) ? yaw : 0.0, 0.0});
      while (pl_mocap_.size() > 1 &&
        (secondsSince(pl_mocap_.front().t) > 2.0 || pl_mocap_.size() > 2000))
      {
        pl_mocap_.pop_front();
      }
      const Eigen::Quaterniond qb(o.w, o.x, o.y, o.z);
      if (std::abs(qb.norm() - 1.0) < 0.1) {trip = plBoxGuardCheck(p, qb.normalized().toRotationMatrix());}
    }
    if (trip) {
      auto resp = std::make_shared<Trigger::Response>();
      onPlAbort(resp);
      RCLCPP_ERROR(get_logger(), "push-and-pull SAFETY GUARD: %s -- %s", pl_guard_reason_.c_str(),
        resp->message.c_str());
    }
  }

  // THE BOX GUARDS (push_pull_guard_box_lag / _box_tilt_deg), with lock_
  // held, on every box mocap sample: true = abort now (2026-10-16).
  bool plBoxGuardCheck(const Vec3 & p, const Mat3 & R)
  {
    const double lag_lim = get_parameter("push_pull_guard_box_lag").as_double();
    const double tilt_lim = get_parameter("push_pull_guard_box_tilt_deg").as_double();
    const bool pushing = pl_flying_.has_value() && *pl_flying_ == kPlPush;
    const bool engaged = mode_direct_ && hold_.has_value() && (pl_at_handle_ || pushing) &&
      !pl_aborting_ && !pl_abort_plan_ && pl_wp_.has_value();
    const bool executing = pushing && state_ == "EXECUTING" && plan_ && exec_t0_.has_value();
    if (!engaged || (!(lag_lim > 0.0) && !(tilt_lim > 0.0))) {
      pl_bg_lag_since_.reset();
      pl_bg_tilt_since_.reset();
      if (!pl_at_handle_ && !pushing) {pl_bg_box0_.reset();}
      return false;
    }
    // the reference pose: the box when the claw first sat on the handle (its
    // tilt) -- re-taken for the lag when the push starts executing
    if (!pl_bg_box0_.has_value()) {
      pl_bg_box0_ = p;
      pl_bg_R0_ = R;
    }
    const double persist = get_parameter("push_pull_guard_box_persist_s").as_double();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    // tilt: the angle between the box's z axis now and at the reference
    const Vec3 z0 = pl_bg_R0_.col(2), z1 = R.col(2);
    pl_bg_tilt_ = std::acos(std::clamp(z0.dot(z1), -1.0, 1.0)) * 180.0 / M_PI;
    if (tilt_lim > 0.0 && pl_bg_tilt_ > tilt_lim) {
      if (!pl_bg_tilt_since_.has_value()) {pl_bg_tilt_since_ = Clock::now();}
      if (secondsSince(pl_bg_tilt_since_) >= persist) {
        m.precision(1);
        m << "box tilted " << pl_bg_tilt_ << " deg > " << tilt_lim << " deg (tipping, not sliding)";
        pl_guard_reason_ = m.str();
        pl_guard_tripped_ = true;
        pl_bg_tilt_since_.reset();
        return true;
      }
    } else {
      pl_bg_tilt_since_.reset();
    }
    // lag: the box against the PLANNED claw, along the push, while it flies
    if (!(lag_lim > 0.0) || !executing || get_parameter("push_pull_turn_deg").as_double() != 0.0) {
      pl_bg_lag_since_.reset();
      return false;
    }
    if (!pl_bg_lag0_.has_value() || pl_bg_lag_gen_ != plan_gen_) {
      pl_bg_lag0_ = p;
      pl_bg_lag_gen_ = plan_gen_;
    }
    const double t = std::chrono::duration<double>(Clock::now() - *exec_t0_).count();
    const Vec3 plan = plan_->eval(std::min(t, plan_->duration())).r_ed -
      plan_->eval(0.0).r_ed;
    const double d = get_parameter("push_pull_push_distance").as_double();
    const Vec3 u = (d < 0.0 ? -1.0 : 1.0) * pl_wp_->push_dir;
    pl_bg_lag_ = plan.dot(u) - (p - *pl_bg_lag0_).dot(u);
    if (std::abs(pl_bg_lag_) > lag_lim) {
      if (!pl_bg_lag_since_.has_value()) {pl_bg_lag_since_ = Clock::now();}
      if (secondsSince(pl_bg_lag_since_) >= persist) {
        m.precision(0);
        m << "the box is " << std::abs(pl_bg_lag_) * 1e3 << " mm " << (pl_bg_lag_ > 0.0 ? "BEHIND" : "AHEAD OF")
          << " its plan (> " << lag_lim * 1e3 << " mm: " << (pl_bg_lag_ > 0.0 ? "jammed or too heavy" : "running away")
          << ")";
        pl_guard_reason_ = m.str();
        pl_guard_tripped_ = true;
        pl_bg_lag_since_.reset();
        return true;
      }
    } else {
      pl_bg_lag_since_.reset();
    }
    return false;
  }

  // The box feed and the arm's buckle (push_pull_guard_box_stale_s /
  // _q3_min_deg), with lock_ held, from the 10 Hz tick: true = abort now.
  bool plFeedGuardCheck()
  {
    const bool pushing = pl_flying_.has_value() && *pl_flying_ == kPlPush;
    const bool engaged = mode_direct_ && hold_.has_value() && (pl_at_handle_ || pushing) &&
      !pl_aborting_ && !pl_abort_plan_;
    if (!engaged) {
      pl_bg_q3_since_.reset();
      return false;
    }
    std::ostringstream m;
    m.setf(std::ios::fixed);
    const bool box_armed = get_parameter("push_pull_guard_box_lag").as_double() > 0.0 ||
      get_parameter("push_pull_guard_box_tilt_deg").as_double() > 0.0;
    const double stale = get_parameter("push_pull_guard_box_stale_s").as_double();
    const double age = pl_mocap_.empty() ? 1e9 : secondsSince(pl_mocap_.back().t);
    // A FROZEN feed counts as a silent one (2026-10-19, the hardware box push):
    // the mocap processor republishes the last pose of a body it lost with fresh
    // stamps (10-09: 11 423 identical obj_0 samples), so the age above never
    // grows -- the time since the pose last CHANGED does. At least three
    // bit-identical samples: a repeated VRPN frame (pairs, 2026-10-14) is not.
    double frozen = 0.0;
    if (pl_mocap_.size() >= 3) {
      const Vec3 & last = pl_mocap_.back().p;
      size_t n = 0;
      auto first = pl_mocap_.back().t;
      for (auto it = pl_mocap_.rbegin(); it != pl_mocap_.rend() && it->p == last; ++it) {
        first = it->t;
        ++n;
      }
      if (n >= 3) {frozen = secondsSince(first);}
    }
    if (box_armed && stale > 0.0 && std::max(age, frozen) > stale) {
      m.precision(2);
      m << "the box's mocap (" << pl_box_topic_ << ") " << (frozen > age ? "FROZEN" : "silent")
        << " for " << std::min(std::max(age, frozen), 99.0)
        << " s > " << stale << " s -- the box guards are blind";
      pl_guard_reason_ = m.str();
      pl_guard_tripped_ = true;
      return true;
    }
    const double q3_lim = get_parameter("push_pull_guard_q3_min_deg").as_double();
    if (q3_lim > 0.0 && q_meas_.has_value() && secondsSince(q_meas_time_) < 0.25) {
      const double q3 = (*q_meas_)(2) * 180.0 / M_PI;
      if (q3 < q3_lim) {
        if (!pl_bg_q3_since_.has_value()) {pl_bg_q3_since_ = Clock::now();}
        if (secondsSince(pl_bg_q3_since_) >= get_parameter("push_pull_guard_box_persist_s").as_double()) {
          m.precision(1);
          m << "q3 " << q3 << " deg < " << q3_lim << " deg (the arm buckling toward its "
            "elbow-singular branch)";
          pl_guard_reason_ = m.str();
          pl_guard_tripped_ = true;
          pl_bg_q3_since_.reset();
          return true;
        }
      } else {
        pl_bg_q3_since_.reset();
      }
    }
    return false;
  }

  // The inputs changed: a plan made from the old ones must not be flown.
  // Called under the lock.
  void plInvalidate()
  {
    pl_wp_.reset();
    pl_planning_ = false;
    pl_error_.clear();
    ++pl_gen_;
  }

  // Ask the whole-body node for the contact (true) or free (false) phase flag
  // of its 4-D attribution. Fire and forget: the node refuses outside DIRECT
  // and restores its yaml flag on every mode change by itself; a refusal is
  // logged here.
  void requestContact(bool on)
  {
    pl_contact_ = on;
    if (!get_parameter("push_pull_contact").as_bool()) {return;}
    if (!contact_cli_ || !contact_cli_->service_is_ready()) {
      RCLCPP_WARN(
        get_logger(), "push-and-pull: contact-flag service %s not available -- the law keeps "
        "its yaml phase flag (is this the whole-body 4-D L1 node?)",
        contact_cli_ ? contact_cli_->get_service_name() : "?");
      return;
    }
    auto req = std::make_shared<SetBool::Request>();
    req->data = on;
    contact_cli_->async_send_request(
      req, [this](rclcpp::Client<SetBool>::SharedFuture f) {
        const auto r = f.get();
        if (r->success) {
          RCLCPP_INFO(get_logger(), "push-and-pull: %s", r->message.c_str());
        } else {
          RCLCPP_WARN(get_logger(), "push-and-pull: the contact flag was REFUSED: %s", r->message.c_str());
        }
      });
  }

  void onPlCapture(Trigger::Response::SharedPtr resp)
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pl_flying_.has_value() || pl_aborting_ || pl_abort_plan_) {
        resp->success = false;
        resp->message = "a push-and-pull step is in flight -- Get once it holds";
        return;
      }
      if (pl_at_handle_) {
        resp->success = false;
        resp->message = "the box is in the gripper -- Get it before Ready, or after Exit";
        return;
      }
      if (pl_box_topic_.empty()) {
        resp->success = false;
        resp->message = "no mocap body for the box -- set push_pull_box_topic";
        return;
      }
      const double window = get_parameter("push_pull_capture_window").as_double();
      std::vector<Vec3> pts;
      std::vector<double> yaws;
      double ys = 0.0, yc = 0.0;
      for (const auto & s : pl_mocap_) {
        if (secondsSince(s.t) <= window) {
          pts.push_back(s.p);
          yaws.push_back(s.yaw);
          ys += std::sin(s.yaw);
          yc += std::cos(s.yaw);
        }
      }
      if (pts.empty()) {
        m.precision(2);
        m << "no sample on " << pl_box_topic_ << " in the last " << window
          << " s -- is the box tracked and broadcast?";
        resp->success = false;
        resp->message = m.str();
        return;
      }
      // FROZEN (2026-10-19, as the pick-and-place capture since 2026-10-10): a
      // tracked body jitters by tens of micrometres at least; identical samples
      // are the mocap processor republishing the last pose of a body it lost
      if (pts.size() >= 2 &&
        std::all_of(pts.begin() + 1, pts.end(), [&pts](const Vec3 & p) {return p == pts.front();}))
      {
        m.precision(2);
        m << "the box's pose is frozen -- the box is not tracked (" << pts.size()
          << " identical samples from " << pl_box_topic_ << " in " << window
          << " s: the mocap processor is republishing its last pose) -- NOT captured";
        resp->success = false;
        resp->message = m.str();
        RCLCPP_WARN(get_logger(), "push-and-pull capture refused: %s", resp->message.c_str());
        return;
      }
      Vec3 mean = Vec3::Zero();
      for (const Vec3 & p : pts) {mean += p;}
      mean /= static_cast<double>(pts.size());
      double spread = 0.0;
      for (const Vec3 & p : pts) {spread = std::max(spread, (p - mean).norm());}
      const double max_spread = get_parameter("push_pull_capture_max_spread").as_double();
      if (spread > max_spread) {
        m.precision(1);
        m << pts.size() << " samples from " << pl_box_topic_ << " in " << window << " s scatter "
          << spread * 1e3 << " mm (> " << max_spread * 1e3 << " mm): more than one publisher on the "
             "topic, or the box is moving or mis-tracked -- NOT captured";
        resp->success = false;
        resp->message = m.str();
        RCLCPP_WARN(get_logger(), "push-and-pull capture refused: %s", resp->message.c_str());
        return;
      }
      // FLIPPING YAW (2026-10-19): the push and the box-on-table check follow
      // the box's yaw, and the circular mean of two opposite headings is
      // garbage. Spread about the first sample, wrapped (a steady yaw near
      // +-180 deg reads small).
      const double max_yaw = get_parameter("push_pull_capture_max_yaw_spread_deg").as_double();
      double lo = 0.0, hi = 0.0;
      for (double y : yaws) {
        const double d = wrapPi(y - yaws.front());
        lo = std::min(lo, d);
        hi = std::max(hi, d);
      }
      const double yaw_spread = (hi - lo) * 180.0 / M_PI;
      if (max_yaw > 0.0 && yaw_spread > max_yaw) {
        m.precision(0);
        m << "the box's yaw varies by " << yaw_spread << " deg (> " << max_yaw << " deg) over "
          << pts.size() << " samples from " << pl_box_topic_ << ": its orientation is flipping "
             "between two solutions (or the box is turning) -- make the marker layout asymmetric, "
             "or press Get again -- NOT captured";
        resp->success = false;
        resp->message = m.str();
        RCLCPP_WARN(get_logger(), "push-and-pull capture refused: %s", resp->message.c_str());
        return;
      }
      const double yaw = std::atan2(ys, yc);   // circular mean: no wrap at +-180 deg
      // the capture has a parameter twin (2026-10-16): write it -- onPlParams
      // sets the capture from it and makes the plan stale, and the arm GS's
      // Box boxes follow the parameter event
      const auto res = set_parameters({rclcpp::Parameter("push_pull_box_point",
            std::vector<double>{mean(0), mean(1), mean(2), yaw * 180.0 / M_PI})});
      if (res.empty() || !res.front().successful) {
        resp->success = false;
        resp->message = "box NOT captured: push_pull_box_point not set" +
          (res.empty() ? std::string() : ": " + res.front().reason);
        return;
      }
      pl_box_ = mean;
      pl_box_yaw_ = yaw;
      plInvalidate();
      m.precision(3);
      m << "captured the box at [" << mean(0) << ", " << mean(1) << ", " << mean(2) << "] m, yaw "
        << std::setprecision(1) << yaw * 180.0 / M_PI << " deg from " << pl_box_topic_ << " ("
        << pts.size() << " samples, spread " << spread * 1e3
        << " mm); the EE offset is applied in this box frame -- press Plan";
    }
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "push-and-pull: %s", resp->message.c_str());
    publishPlStatus();
    publishPlInfo();
    publishPlPaths(nullptr);
  }

  // ADJUST: the vehicle is on (or over) its start mark; its measured position
  // minus push_pull_start_mark is written into the start mark and the land
  // point (x, y): the mocap centring changes between sessions, the room does
  // not. The box is measured and never moved.
  void onPlAdjust(Trigger::Response::SharedPtr resp)
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(3);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pl_flying_.has_value() || pl_aborting_ || pl_abort_plan_) {
        resp->success = false;
        resp->message = "a push-and-pull step is in flight -- adjust once it holds";
        return;
      }
      if (!odom_p_.has_value() || secondsSince(odom_p_time_) > kOdomFreshS) {
        resp->success = false;
        resp->message = "no fresh odometry -- cannot measure the vehicle's position";
        return;
      }
      BasePose mark;
      std::string err;
      if (!ppBase("push_pull_start_mark", &mark, &err)) {
        resp->success = false;
        resp->message = err;
        return;
      }
      Vec3 off = *odom_p_ - mark.p;
      off(2) = 0.0;
      const double max_off = get_parameter("push_pull_adjust_max").as_double();
      if (off.norm() > max_off) {
        m << "offset [" << off(0) << ", " << off(1) << "] m is larger than " << max_off
          << " m -- is the vehicle on its start mark? Not adjusted";
        resp->success = false;
        resp->message = m.str();
        return;
      }
      std::vector<rclcpp::Parameter> moved;
      for (const char * name : {"push_pull_start_mark", "push_pull_land", "push_pull_land_start"}) {
        auto v = get_parameter(name).as_double_array();
        if (v.empty() && std::string(name) == "push_pull_land_start") {continue;}   // = the land pose
        if (v.size() != 4) {
          resp->success = false;
          resp->message = std::string(name) + " needs 4 numbers -- not adjusted";
          return;
        }
        v[0] += off(0);
        v[1] += off(1);
        moved.emplace_back(name, v);
      }
      const auto results = set_parameters(moved);   // onPlParams: checked, plan made stale
      for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].successful) {
          resp->success = false;
          resp->message = moved[i].get_name() + " not moved: " + results[i].reason;
          return;
        }
      }
      pl_offset_ = off;
      plInvalidate();
      m << "start mark and land point moved by [" << off(0) << ", " << off(1)
        << "] m (the box is measured and not moved); press Plan";
    }
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "push-and-pull adjust: %s", resp->message.c_str());
    publishPlStatus();
    publishPlInfo();
    publishPlPaths(nullptr);
  }

  // Another mode's motion in flight (lock_ held): a step must not start under it.
  bool otherModeBusy() const
  {
    return pp_flying_.has_value() || pp_exiting_.has_value() || pp_aborting_ ||
           static_cast<bool>(pp_abort_plan_);
  }

  void onPlPlan(Trigger::Response::SharedPtr resp)
  {
    PushPullTargets tg;
    PushPullConfig cfg;
    PlanOptions opts;
    RestSpec from;
    unsigned gen = 0;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::string err;
      if (!mode_direct_ || !hold_.has_value()) {
        err = "not in whole-body DIRECT (the first step is planned from the hold)";
      } else if (teleopActive()) {
        err = "PS4 teleoperation engaged -- disengage it first";
      } else if (pl_flying_.has_value() || pl_aborting_ || pl_abort_plan_ ||
        state_ == "EXECUTING" || state_ == "CALCULATING")
      {
        err = "planner is " + state_ + " -- plan once it holds";
      } else if (pl_at_handle_) {
        err = "the claw is on the handle -- Exit first (Plan dry-runs from Go To Start)";
      } else if (otherModeBusy()) {
        err = "another mode is flying -- wait for it";
      }
      if (err.empty()) {plConfig(&cfg, &err);}
      if (err.empty()) {plTargets(&tg, &err);}
      if (!err.empty()) {
        resp->success = false;
        resp->message = err;
        return;
      }
      opts = planOptions();
      from = *hold_;
      plInvalidate();
      gen = pl_gen_;
      pl_planning_ = true;
      pl_aborted_ = false;
      pl_guard_tripped_ = false;
    }
    publishPlStatus();
    publishPlPaths(nullptr);
    if (pl_worker_.joinable()) {pl_worker_.join();}
    pl_worker_ = std::thread([this, tg, cfg, opts, from, gen]() {
        std::optional<PushPullWaypoints> wp;
        std::vector<std::shared_ptr<const Trajectory>> steps;
        std::shared_ptr<Trajectory> whole;
        std::string err;
        try {
          wp = pushPullWaypoints(*vehicle_, cfg, tg);
          steps = planPushPullMission(*vehicle_, *planner_, opts, cfg, *wp, from);
          whole = std::make_shared<SequenceTrajectory>(steps);
        } catch (const std::exception & e) {
          err = e.what();
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != pl_gen_) {return;}   // superseded
          pl_planning_ = false;
          if (!err.empty()) {
            pl_error_ = err;
            RCLCPP_WARN(get_logger(), "push-and-pull plan refused: %s", err.c_str());
          } else {
            pl_wp_ = wp;
            pl_leave_wp_.reset();
            for (int k = 0; k < kNumPushPullSteps; ++k) {
              pl_T_[k] = steps[k]->duration();
              RCLCPP_INFO(
                get_logger(), "push-and-pull %s: %s", pushPullStepName(k),
                steps[k]->diag().summary.c_str());
            }
            if (cfg.turn_angle != 0.0) {
              RCLCPP_INFO(
                get_logger(), "push-and-pull planned: %.1f s of flight in six steps; the push "
                "TURNS the handle %.0f deg about the captured axis [%.3f, %.3f] in %.1f s, the "
                "nose along yaw %.1f deg at the handle.", whole->duration(),
                cfg.turn_angle * 180.0 / M_PI, wp->axis(0), wp->axis(1), cfg.push_time,
                wp->drone_yaw * 180.0 / M_PI);
            } else {
              RCLCPP_INFO(
                get_logger(), "push-and-pull planned: %.1f s of flight in six steps; the push slides "
                "the box %.2f m along yaw %.1f deg.", whole->duration(), cfg.push_distance,
                wp->drone_yaw * 180.0 / M_PI);
            }
          }
        }
        publishPlStatus();
        publishPlInfo();
        publishPlPaths(whole.get());
      });
    resp->success = true;
    resp->message = "planning the six steps from the current hold -- the status reads READY when done";
  }

  // What the Ready descent gates on (lock_ held), as the pick's: the claw's
  // current error above the handle less the trim (its steady part), and the
  // trim itself -- the mean hover error over the window, zero when disabled or
  // with under half a window of samples.
  Vec3 plTrim() const
  {
    if (!get_parameter("push_pull_descent_trim").as_bool() || pl_trim_buf_.size() < 2) {
      return Vec3::Zero();
    }
    const double span = std::chrono::duration<double>(
      pl_trim_buf_.back().first - pl_trim_buf_.front().first).count();
    if (span < 0.5 * get_parameter("push_pull_descent_trim_window_s").as_double()) {
      return Vec3::Zero();
    }
    Vec3 m = Vec3::Zero();
    for (const auto & s : pl_trim_buf_) {m += s.second;}
    return m / static_cast<double>(pl_trim_buf_.size());
  }

  double plResidual() const
  {
    if (pl_trim_buf_.empty()) {return pl_last_err_;}
    return (pl_trim_buf_.back().second - plTrim()).norm();
  }

  void onPlStep(int s, Trigger::Response::SharedPtr resp)
  {
    RestSpec from;
    PushPullWaypoints wp;
    PushPullConfig cfg;
    PlanOptions opts;
    unsigned gen = 0;
    Vec3 trim = Vec3::Zero();
    const char * name = pushPullStepName(s);
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::string err;
      std::ostringstream w;
      w.setf(std::ios::fixed);
      w.precision(0);
      if (!mode_direct_ || !hold_.has_value()) {
        err = "not in whole-body DIRECT";
      } else if (pl_planning_) {
        err = "the mission is still planning";
      } else if (pl_flying_.has_value() || pl_aborting_ || pl_abort_plan_) {
        err = std::string(pl_flying_.has_value() ? pushPullStepName(*pl_flying_) : "the abort") +
          " is in flight -- wait for it, or Abort";
      } else if (!pl_wp_.has_value() &&
        !((s == kPlGoToLandStart || s == kPlLand) && pl_leave_wp_.has_value()))
      {
        err = "not planned -- Get the box and press Plan";
      } else if (teleopActive()) {
        err = "PS4 teleoperation engaged -- disengage it first";
      } else if (state_ == "EXECUTING" || state_ == "CALCULATING") {
        err = "planner is " + state_ + " -- wait for HOLD";
      } else if (otherModeBusy()) {
        err = "another mode is flying -- wait for it";
      } else if (!odom_p_.has_value() || secondsSince(odom_p_time_) > kOdomFreshS) {
        err = "no fresh odometry -- a step must not start blind";
      } else if ((*odom_p_ - hold_->x_b).norm() > get_parameter("push_pull_leg_start_tol").as_double()) {
        w << "the body is " << (*odom_p_ - hold_->x_b).norm() * 1e3 << " mm from its hold (> "
          << get_parameter("push_pull_leg_start_tol").as_double() * 1e3 << " mm) -- wait for it to settle";
        err = w.str();
      } else if (s == kPlGoToStart || s == kPlGoToLandStart) {
        if (pl_at_handle_) {err = "the claw is on the handle -- Exit first";}
      } else if (s == kPlLand) {
        if (pl_at_handle_) {
          err = "the claw is on the handle -- Exit first";
        } else if (pl_completed_ != kPlGoToLandStart) {
          err = "not at Land Start -- fly Go To Land Start first";
        }
      } else if (s == kPlReady) {
        const double tol = get_parameter("push_pull_arrival_tol").as_double();
        if (pl_at_handle_) {
          err = "the claw is already on the handle";
        } else if (pl_completed_ != kPlGoToStart) {
          err = "not holding above the handle -- fly Go To Start first";
        } else if (!(plResidual() <= tol)) {
          w << "the claw is " << plResidual() * 1e3 << " mm from "
            << (get_parameter("push_pull_descent_trim").as_bool() ? "its trimmed point above the handle"
                                                                   : "the point above the handle")
            << " (> " << tol * 1e3 << " mm) -- wait for it to settle";
          err = w.str();
        } else if (plTrim().norm() > get_parameter("push_pull_descent_trim_max").as_double()) {
          w << "the claw holds " << plTrim().norm() * 1e3 << " mm off the point above the handle on "
            "average (> push_pull_descent_trim_max " <<
            get_parameter("push_pull_descent_trim_max").as_double() * 1e3 << " mm) -- not trimming that far";
          err = w.str();
        }
      } else if (s == kPlPush) {
        if (!pl_at_handle_ || pl_completed_ != kPlReady) {
          err = pl_completed_ == kPlPush ? "already pushed -- Exit (open the gripper first)"
                                         : "not holding on the handle -- Ready first";
        }
      } else if (s == kPlExit) {
        if (!pl_at_handle_) {err = "the claw is not on the handle -- nothing to exit";}
      }
      if (err.empty()) {plConfig(&cfg, &err);}
      if (!err.empty()) {
        resp->success = false;
        resp->message = std::string(name) + ": " + err;
        return;
      }
      from = *hold_;
      wp = pl_wp_.has_value() ? *pl_wp_ : *pl_leave_wp_;   // leaving the table after an abort
      opts = planOptions();
      gen = ++plan_gen_;
      state_ = "CALCULATING";
      plan_.reset();
      goal_override_.reset();
      auto_send_ = false;
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      pl_flying_ = s;
      pl_exec_plan_.reset();
      pl_error_.clear();
      pl_aborted_ = false;
      pl_guard_tripped_ = false;
      switch (s) {
        case kPlReady:
          // the descent keeps the world-held claw requested on arrival above
          trim = plTrim();
          pl_trim_ = trim;
          break;
        case kPlPush:
          // THE INTERACTION PHASE (user, 2026-10-03): from the clamp on, the
          // law's phase flag is CONTACT -- the handle's reaction is the task
          // wrench, rendered on the EE impedance instead of rejected -- and
          // the EE reference follows the CoM (a world-held claw in contact
          // fights the base, 2026-10-02). The push holds push_settle first.
          // push_pull_world_anchor_push keeps the world-held claw instead.
          if (get_parameter("push_pull_world_anchor_push").as_bool()) {
            requestAnchor(false);
          } else {
            restoreAnchor();
          }
          requestContact(true);
          pl_unloaded_ = false;
          pl_ey_buf_.clear();
          pl_bg_box0_.reset();       // the box guards: taken when the push executes
          pl_bg_lag_since_.reset();
          break;
        case kPlExit:
          // the gripper is open: no contact any more
          requestContact(false);
          restoreAnchor();
          break;
        default:
          restoreAnchor();
      }
    }
    publishStatus();
    publishPlStatus();
    publishPlInfo();
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, s, from, wp, cfg, opts, gen, trim]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          switch (s) {
            case kPlGoToStart:
              traj = planPushPullGoToStart(*vehicle_, *planner_, opts, cfg, wp, from);
              break;
            case kPlReady:
              traj = planPushPullReady(*vehicle_, *planner_, opts, cfg, wp, from, trim);
              break;
            case kPlPush:
              traj = planPushPullPush(*vehicle_, *planner_, opts, cfg, from, wp.axis);
              break;
            case kPlExit:
              traj = planPushPullExit(*vehicle_, *planner_, opts, cfg, from);
              break;
            case kPlGoToLandStart:
              traj = planPushPullGoToLandStart(*vehicle_, *planner_, opts, cfg, wp, from);
              break;
            default:
              traj = planPushPullLand(*vehicle_, *planner_, opts, cfg, wp, from);
          }
        } catch (const std::exception & e) {
          err = e.what();
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != plan_gen_ || state_ != "CALCULATING") {return;}   // superseded
          if (!traj) {
            state_ = "INFEASIBLE";
            infeasible_reason_ = err;
            pl_error_ = err;
            pl_flying_.reset();
            RCLCPP_WARN(get_logger(), "push-and-pull step refused: %s", err.c_str());
          } else {
            plan_ = traj;
            pl_exec_plan_ = traj;
            exec_t0_ = Clock::now();
            state_ = "EXECUTING";
            RCLCPP_INFO(
              get_logger(), "push-and-pull %s: executing -- %s", pushPullStepName(s),
              traj->diag().summary.c_str());
            pl_follow_.reset();
            if (s == kPlPush && cfg.turn_angle != 0.0 && get_parameter("push_pull_follow").as_bool()) {
              plFollowStart(traj, wp.axis, cfg.turn_angle);
            }
          }
        }
        publishStatus();
        publishPlStatus();
        publishPlInfo();
        publishVizPath(traj.get());
      });
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << name << ": planning from the current hold; it executes as soon as it is planned";
    if (s == kPlReady) {
      m << " (goal trimmed by [" << trim(0) * 1e3 << ", " << trim(1) * 1e3 << ", " << trim(2) * 1e3
        << "] mm, the hover error)";
    } else if (s == kPlPush) {
      m << " -- CONTACT phase on, the push holds " << cfg.push_settle << " s, then ";
      if (cfg.turn_angle != 0.0) {
        m << "turns " << std::setprecision(0) << cfg.turn_angle * 180.0 / M_PI
          << " deg about the axis in " << std::setprecision(1) << cfg.push_time << " s";
      } else if (cfg.push_ramp > 0.0 && cfg.push_time > 2.0 * cfg.push_ramp) {
        m << "slides " << std::setprecision(2) << cfg.push_distance << " m at a constant "
          << std::setprecision(1) << std::abs(cfg.push_distance) / (cfg.push_time - cfg.push_ramp) * 1e3
          << " mm/s (" << cfg.push_ramp << " s ramps, " << cfg.push_time << " s)";
      } else {
        m << "slides " << std::setprecision(2) << cfg.push_distance << " m in "
          << std::setprecision(1) << cfg.push_time << " s";
      }
    }
    resp->success = true;
    resp->message = m.str();
    RCLCPP_INFO(get_logger(), "push-and-pull: %s", resp->message.c_str());
  }

  // Called from the stream tick, under the lock, when the plan in flight has
  // ended. THE UNLOAD (push_pull_unload_time_s > 0): when that plan is the
  // push, replace it with the unload leg -- the hold re-seated by the law's
  // averaged task error, still gripping and in CONTACT -- and return true (the
  // push continues). Planned here, synchronously: a few-mm rest-to-rest leg
  // plans in milliseconds and the stream must not see a gap.
  bool plUnloadStart()
  {
    if (!pl_flying_.has_value() || *pl_flying_ != kPlPush || !pl_exec_plan_ ||
      pl_exec_plan_ != plan_ || pl_unloaded_ || pl_aborting_ || pl_abort_plan_)
    {
      return false;
    }
    PushPullConfig cfg;
    std::string err;
    if (!plConfig(&cfg, &err) || cfg.unload_time <= 0.0) {return false;}
    pl_unloaded_ = true;
    const auto now = Clock::now();
    const double avg = get_parameter("push_pull_unload_avg_s").as_double();
    Vec3 shift = Vec3::Zero();
    int n = 0;
    for (const auto & e : pl_ey_buf_) {
      if (std::chrono::duration<double>(now - e.first).count() <= avg) {
        shift += e.second;
        ++n;
      }
    }
    if (n == 0) {
      RCLCPP_WARN(get_logger(), "push-and-pull unload: no task error in the last %.2f s -- "
        "skipped (is the whole-body node's wb_control_debug flowing?)", avg);
      return false;
    }
    shift /= static_cast<double>(n);
    const double max_shift = get_parameter("push_pull_unload_max").as_double();
    if (shift.norm() > max_shift) {
      RCLCPP_WARN(get_logger(), "push-and-pull unload: the claw's task error %.1f mm exceeds "
        "push_pull_unload_max %.1f mm -- not re-seated", shift.norm() * 1e3, max_shift * 1e3);
      return false;
    }
    const RestSpec from = plan_->goalRest();
    std::shared_ptr<Trajectory> traj;
    try {
      traj = planPushPullUnload(*vehicle_, *planner_, planOptions(), cfg, from, shift);
    } catch (const std::exception & e) {
      RCLCPP_WARN(get_logger(), "push-and-pull unload refused (%s) -- the push completes "
        "without it", e.what());
      return false;
    }
    setHold(from, true);
    plan_ = traj;
    pl_exec_plan_ = traj;
    exec_t0_ = now;
    RCLCPP_INFO(get_logger(), "push-and-pull unload: re-seating the hold by [%.1f %.1f %.1f] mm "
      "(the law's task error over %.2f s, %d samples) -- %s", shift(0) * 1e3, shift(1) * 1e3,
      shift(2) * 1e3, avg, n, traj->diag().summary.c_str());
    return true;
  }

  // Called from the stream tick, under the lock, when step s has been flown
  // to its end.
  void plStepDone(int s)
  {
    pl_completed_ = s;
    pl_flying_.reset();
    switch (s) {
      case kPlGoToStart:
        pl_at_handle_ = false;
        pl_trim_buf_.clear();
        // from the hover above the handle through the descent and the grasp
        // the claw is held in the WORLD (the pick's validated anchor) -- and
        // through the push too with push_pull_world_anchor_push
        if (get_parameter("push_pull_world_anchor_ready").as_bool()) {requestAnchor(false);}
        break;
      case kPlReady:
        // THE INTERACTION PHASE STARTS ON THE HANDLE (2026-10-04, first
        // flight): the jaws close next, and from the clamp on the box --
        // held on the table by static friction -- is part of the chain. With
        // the flag still free during the close, the whole-body node's platform
        // observer integrated that reaction for the ~1.5 s before the Push
        // press (51 mm of drift on a fixed reference). Push asks again
        // (idempotent); Exit and ABORT clear it.
        pl_at_handle_ = true;
        if (!get_parameter("push_pull_world_anchor_ready").as_bool() &&
          get_parameter("push_pull_world_anchor_on_handle").as_bool())
        {
          requestAnchor(false);
          RCLCPP_INFO(get_logger(), "push-and-pull: on the handle -- the claw now WORLD-held "
            "(push_pull_world_anchor_on_handle): pulled onto the grasp point before the jaws close");
        }
        if (get_parameter("push_pull_contact_at_ready").as_bool()) {
          requestContact(true);
        } else {
          RCLCPP_INFO(get_logger(), "push-and-pull: on the handle -- the contact phase waits for the "
            "jaws (push_pull_contact_at_ready false: CONTACT at the Push press)");
        }
        break;
      case kPlPush:
        pl_at_handle_ = true;
        if (get_parameter("push_pull_contact_off_at_push_end").as_bool()) {
          requestContact(false);
          RCLCPP_INFO(get_logger(), "push-and-pull: push complete -- contact phase free before the "
            "jaws open (push_pull_contact_off_at_push_end)");
        }
        break;
      default:
        pl_at_handle_ = false;
    }
    RCLCPP_INFO(get_logger(), "push-and-pull: %s complete.", pushPullStepName(s));
  }

  // RELEASE (2026-10-17, the release-window fix). Called by the station (and the
  // sim driver) the moment it commands the gripper OPEN after a push, BEFORE
  // Exit. From the Push press the claw is held in the ROOM
  // (push_pull_world_anchor_push) -- right while the jaws clamp a box held by
  // static friction. Once the jaws open nothing holds the claw, and a
  // world-held open claw makes the arm absorb all of the base's post-release
  // drift: in the 2026-10-17 results campaign the base drifted 27 mm back in
  // 1 s, q2 reached its +45 deg stop, q3 went through 0 into the
  // elbow-singular branch and the vehicle flipped before the q3 guard could
  // act. Release hands the EE reference back to the CoM anchor (blended over
  // the whole-body node's wb_ee_anchor_blend_s) and the phase flag to free. It
  // moves nothing: Exit -- started by the station when the jaws report open --
  // climbs and folds. Refused unless the claw is on the handle (after the push,
  // or after Ready when the grasp found nothing) and nothing is flying -- so a
  // station that opens the gripper only on success never opens it mid-push.
  // Without a call, Exit switches the anchor as before.
  void onPlRelease(Trigger::Response::SharedPtr resp)
  {
    std::lock_guard<std::recursive_mutex> lk(lock_);
    std::string err;
    if (!mode_direct_ || !hold_.has_value()) {
      err = "not in whole-body DIRECT";
    } else if (pl_flying_.has_value() || pl_aborting_ || pl_abort_plan_) {
      err = std::string(pl_flying_.has_value() ? pushPullStepName(*pl_flying_) : "the abort") +
        " is in flight -- wait for it, or Abort";
    } else if (!pl_at_handle_) {
      err = "nothing to release -- the claw is not on the handle";
    }
    if (!err.empty()) {
      resp->success = false;
      resp->message = "release: " + err;
      RCLCPP_WARN(get_logger(), "push-and-pull %s", resp->message.c_str());
      return;
    }
    const bool was_world = pp_anchor_world_;
    requestContact(false);
    restoreAnchor();
    resp->success = true;
    resp->message = std::string("release: ") +
      (was_world ? "the claw's reference now follows the vehicle (CoM anchor, blended)"
                 : "the claw's reference already follows the vehicle (CoM anchor)") +
      " and the contact phase is free -- open the gripper; Exit when it reports open";
    RCLCPP_INFO(get_logger(), "push-and-pull %s", resp->message.c_str());
  }

  // ABORT: the GS opens the gripper on the same press; here the phase flag
  // goes back to free, the EE reference to the CoM, and the vehicle -- held
  // still push_pull_abort_release_wait_s first when the claw is on the handle,
  // so the opening jaws do not lift the box -- climbs push_pull_abort_climb
  // with the arm unchanged and then folds it home. In the middle of a step it
  // CUTS the step at the current reference (x_c continuous). After an abort
  // with the box touched the capture and the plan are dropped: the box may have
  // moved -- Get and Plan again.
  void onPlAbort(Trigger::Response::SharedPtr resp)
  {
    RestSpec from;
    PushPullConfig cfg;
    PlanOptions opts;
    unsigned gen = 0;
    bool cut = false;
    double climb = 0.0, wait = 0.0;
    Vec3 retreat = Vec3::Zero();
    bool measured = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !hold_.has_value()) {
        resp->success = false;
        resp->message = "not in whole-body DIRECT -- use the drone ground station";
        return;
      }
      if (pl_abort_plan_ || pl_aborting_) {
        resp->success = false;
        resp->message = "already aborting";
        return;
      }
      // (2026-10-19) another mode's leg in flight (a push-and-pull step cannot
      // start under one, so nothing of this mode is engaged): its own ABORT
      // releases its payload the right way -- this one would cut its leg here
      if (otherModeBusy()) {
        resp->success = false;
        resp->message = "another mode is flying -- use that tab's ABORT";
        return;
      }
      std::string err;
      if (!plConfig(&cfg, &err)) {
        // an abort is never refused for a configuration error: the defaults
        cfg = PushPullConfig{};
        cfg.home = home_pose_;
        RCLCPP_WARN(get_logger(), "push-and-pull ABORT with the default configuration (%s)", err.c_str());
      }
      from = *hold_;
      std::optional<WbReference> cut_ref;
      if (state_ == "EXECUTING" && plan_ && exec_t0_.has_value()) {
        // the governed turn is cut where its PROGRESS is, not its clock
        const double t = plFollowing() ? pl_follow_->gov.tau() :
          std::chrono::duration<double>(Clock::now() - *exec_t0_).count();
        cut_ref = plan_->eval(std::min(t, plan_->duration()));
      }
      if (plFollowing()) {plFollowEnd("aborted");}
      if (cut_ref.has_value()) {
        const WbReference & ref = *cut_ref;
        from.phi = std::atan2(ref.b1_d(1), ref.b1_d(0));
        from.q = ref.q_d;
        Vec3 r0c;
        armKinematics(ref.q_d, vehicle_->params, &r0c, nullptr, nullptr);
        from.x_b = ref.x_cd - Rz(from.phi) * r0c;   // x_c continuous
        cut = true;
      }
      const bool touched = pl_at_handle_ ||
        (pl_flying_.has_value() && (*pl_flying_ == kPlReady || *pl_flying_ == kPlPush));
      // on the box: hold where the vehicle IS, and retreat against the push (2026-10-16)
      if (touched && get_parameter("push_pull_abort_from_measured").as_bool()) {
        if (auto m = restFromMeasurements()) {
          from = *m;
          cut = false;
          measured = true;
        }
      }
      const bool turning = get_parameter("push_pull_turn_deg").as_double() != 0.0;
      if (touched && pl_wp_.has_value()) {
        const double d = get_parameter("push_pull_push_distance").as_double();
        // a turn retreats along the nose it has NOW (the hold it aborts from)
        // the nose at model heading phi (push_pull.cpp's noseOf: the arm's forward axis)
        const Vec3 dir = turning ? Vec3{-std::sin(from.phi), std::cos(from.phi), 0.0} : pl_wp_->push_dir;
        retreat = -get_parameter("push_pull_abort_back").as_double() *
          (turning ? 1.0 : (d < 0.0 ? -1.0 : 1.0)) * dir;
      }
      climb = std::max(0.0, std::min(
          get_parameter("push_pull_abort_climb").as_double(), cfg.fence.max_z - from.x_b(2)));
      wait = touched ? get_parameter("push_pull_abort_release_wait_s").as_double() : 0.0;
      pl_abort_climb_ = climb;
      opts = planOptions();
      gen = ++plan_gen_;
      plan_.reset();
      goal_override_.reset();
      auto_send_ = false;
      pending_base_.reset();
      ee_target_.reset();
      home_goal_ = false;
      setHold(from);
      state_ = "CALCULATING";
      pl_flying_.reset();
      pl_exec_plan_.reset();
      pl_error_.clear();
      pl_aborted_ = true;
      pl_aborting_ = true;
      requestContact(false);
      // the free-handle abort keeps the claw world-held until the release wait ends
      // (the worker schedules the hand-back when the abort plan starts)
      pl_abort_keep_anchor_ = touched && wait > 0.0 && pp_anchor_world_ &&
        get_parameter("push_pull_abort_release_world_held").as_bool();
      pl_abort_anchor_due_.reset();
      if (!pl_abort_keep_anchor_) {restoreAnchor();}
      if (touched) {
        if (pl_wp_.has_value()) {pl_leave_wp_ = pl_wp_;}   // Go To Land Start stays flyable
        pl_box_.reset();
        pl_box_yaw_.reset();
        plInvalidate();
        pl_completed_ = -1;
        pl_box_point_clear_ = true;   // its parameter twin: cleared from the 10 Hz timer
      }
    }
    publishStatus();
    publishPlStatus();
    publishPlInfo();
    if (worker_.joinable()) {worker_.join();}
    worker_ = std::thread([this, from, cfg, opts, gen, climb, wait, retreat]() {
        std::shared_ptr<Trajectory> traj;
        std::string err;
        try {
          traj = planPushPullAbort(*vehicle_, *planner_, opts, cfg, from, climb, wait, retreat);
        } catch (const std::exception & e) {
          err = e.what();
        }
        {
          std::lock_guard<std::recursive_mutex> lk(lock_);
          if (gen != plan_gen_ || state_ != "CALCULATING") {return;}   // superseded
          pl_aborting_ = false;
          if (!traj) {
            state_ = "INFEASIBLE";
            infeasible_reason_ = "abort: " + err;
            pl_error_ = "abort refused: " + err + " -- holding where it was cut";
            RCLCPP_ERROR(get_logger(), "push-and-pull ABORT could not plan: %s", err.c_str());
            if (pl_abort_keep_anchor_) {pl_abort_keep_anchor_ = false; restoreAnchor();}
          } else {
            plan_ = traj;
            pl_abort_plan_ = traj;
            exec_t0_ = Clock::now();
            state_ = "EXECUTING";
            if (pl_abort_keep_anchor_) {
              pl_abort_anchor_due_ = *exec_t0_ + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double>(wait));
            }
            RCLCPP_WARN(get_logger(), "push-and-pull ABORT: %s", traj->diag().summary.c_str());
          }
        }
        publishStatus();
        publishPlStatus();
        publishVizPath(traj.get());
      });
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(2);
    m << "ABORT: contact off, " << (wait > 0.0 ? (pl_abort_keep_anchor_ ? "release (claw world-held) " : "release ") : "")
      << (retreat.norm() > 1e-3 ? "back " : "") << "then up " << climb << " m and the "
      << "arm home, from " << (measured ? "the MEASURED pose" : cut ? "the point the step was cut at" : "the hold");
    if (retreat.norm() > 1e-3) {m << " (back " << retreat.norm() << " m against the push first)";}
    resp->success = true;
    resp->message = m.str();
    RCLCPP_WARN(get_logger(), "push-and-pull: %s", resp->message.c_str());
  }

  void onPlReset(Trigger::Response::SharedPtr resp)
  {
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (pl_flying_.has_value() || pl_aborting_ || pl_abort_plan_) {
        resp->success = false;
        resp->message = "a push-and-pull step is in flight -- Abort it first";
        return;
      }
      if (pl_at_handle_) {
        resp->success = false;
        resp->message = "the claw is on the handle -- Exit (or Abort) first";
        return;
      }
      pl_completed_ = -1;
      pl_aborted_ = false;
      pl_guard_tripped_ = false;
      plInvalidate();
      restoreAnchor();
    }
    resp->success = true;
    resp->message = "push-and-pull progress and plan cleared (the box capture and the typed points kept)";
    publishPlStatus();
    publishPlInfo();
    publishPlPaths(nullptr);
  }

  // ---------------------------------------------------- PATH FOLLOWING
  // (2026-10-18, push_pull_follow). All under lock_.
  bool plFollowing() const
  {
    return pl_follow_.has_value() && plan_ && pl_follow_->plan == plan_;
  }

  // The push of a TURN starts executing with push_pull_follow true.
  void plFollowStart(const std::shared_ptr<const Trajectory> & traj, const Vec3 & axis, double angle)
  {
    if (!traj->timeScalable()) {
      RCLCPP_WARN(get_logger(), "push-and-pull PATH FOLLOWING: this push cannot be time-scaled -- "
        "the turn flies its schedule");
      return;
    }
    PlFollow f;
    f.plan = traj;
    f.gov.setup(get_parameter("push_pull_follow_bandwidth").as_double(),
      get_parameter("push_pull_follow_recover_s").as_double());
    f.t0 = f.last = f.last_log = Clock::now();
    f.axis = axis;
    f.sign = angle >= 0.0 ? 1.0 : -1.0;
    const WbReference r0 = traj->eval(0.0);
    f.alpha0 = f.alpha_prev = std::atan2(r0.r_ed(1) - axis(1), r0.r_ed(0) - axis(0));
    auto pair = [this](const char * name, double * out) {
        const auto v = get_parameter(name).as_double_array();
        out[0] = v.size() >= 2 ? v[0] : 0.0;
        out[1] = v.size() >= 2 ? v[1] : 0.0;
      };
    pair("push_pull_follow_lead_deg", f.lead);
    pair("push_pull_follow_tilt_deg", f.tilt);
    pair("push_pull_follow_wrist_deg", f.wrist);
    f.stale_s = get_parameter("push_pull_follow_stale_s").as_double();
    f.hold_timeout_s = get_parameter("push_pull_follow_hold_timeout_s").as_double();
    f.filter_s = get_parameter("push_pull_follow_filter_s").as_double();
    if (!pl_mocap_.empty() && secondsSince(pl_mocap_.back().t) < f.stale_s) {
      f.yaw0 = pl_mocap_.back().yaw;
      f.have_yaw0 = true;
    }
    pl_follow_ = f;
    auto on = [](const double * c) {return c[1] > c[0] && c[1] > 0.0;};
    RCLCPP_INFO(get_logger(), "push-and-pull PATH FOLLOWING on: the turn flies at a governed "
      "progress -- lead %s, tilt %s, wrist %s (soft/hard deg, low-passed %.1f s), progress filter "
      "%.1f rad/s, recover %.1f s, hold timeout %s; the valve's yaw %s",
      on(f.lead) ? (fmt1(f.lead[0]) + "/" + fmt1(f.lead[1])).c_str() : "off",
      on(f.tilt) ? (fmt1(f.tilt[0]) + "/" + fmt1(f.tilt[1])).c_str() : "off",
      on(f.wrist) ? (fmt1(f.wrist[0]) + "/" + fmt1(f.wrist[1])).c_str() : "off",
      f.filter_s, get_parameter("push_pull_follow_bandwidth").as_double(),
      get_parameter("push_pull_follow_recover_s").as_double(),
      f.hold_timeout_s > 0.0 ? (fmt1(f.hold_timeout_s) + " s").c_str() : "off",
      f.have_yaw0 ? "captured" : "NOT FRESH (the lead channel holds until it is)");
  }

  static std::string fmt1(double x)
  {
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << x;
    return m.str();
  }

  // One stream tick of the governed turn: returns the progress to evaluate
  // the plan at; *abort = true once a hold outlasts the timeout.
  double plFollowTick(bool * abort)
  {
    PlFollow & f = *pl_follow_;
    const auto now = Clock::now();
    const double dt = std::clamp(std::chrono::duration<double>(now - f.last).count(), 0.0, 0.05);
    f.last = now;
    // the planned claw's rotation about the axis where the progress is now
    const WbReference r = plan_->eval(f.gov.tau());
    const double a = std::atan2(r.r_ed(1) - f.axis(1), r.r_ed(0) - f.axis(0));
    f.alpha += wrapPi(a - f.alpha_prev);
    f.alpha_prev = a;
    auto on = [](const double * c) {return c[1] > c[0] && c[1] > 0.0;};
    // first-order low-pass of a channel, seeded by its first sample
    const double lp = f.filter_s > 0.0 ? 1.0 - std::exp(-dt / f.filter_s) : 1.0;
    auto low = [&f, lp](int k, double * state, double x) {
        if (!f.f_init[k]) {*state = x; f.f_init[k] = true;} else {*state += lp * (x - *state);}
      };
    bool stale = false;
    double fac[3] = {1.0, 1.0, 1.0};
    // lead: the plan ahead of the measured valve, the measured rotation taken
    // nearest the plan (no unwrapping to drift); a one-tick jump above 30 deg
    // is an orientation flip of the mocap body, not motion: kept out
    if (on(f.lead)) {
      if (pl_mocap_.empty() || secondsSince(pl_mocap_.back().t) > f.stale_s) {
        stale = true;
      } else {
        const double yaw = pl_mocap_.back().yaw;
        if (!f.have_yaw0) {
          f.yaw0 = yaw - f.alpha;   // a late feed: taken to sit on the plan
          f.have_yaw0 = true;
        }
        const double lead = f.sign * wrapPi(f.alpha - (yaw - f.yaw0)) * 180.0 / M_PI;
        if (f.have_lead && std::abs(lead - f.lead_prev) > 30.0) {
          ++f.flips;
        } else {
          f.lead_prev = lead;
          f.have_lead = true;
        }
        low(0, &f.e_lead, f.lead_prev);
        fac[0] = followFactor(f.e_lead, f.lead[0], f.lead[1]);
      }
    }
    if (on(f.tilt)) {
      if (!att_R_.has_value() || secondsSince(att_time_) > f.stale_s) {
        stale = true;
      } else {
        low(1, &f.e_tilt, std::acos(std::clamp((*att_R_)(2, 2), -1.0, 1.0)) * 180.0 / M_PI);
        fac[1] = followFactor(f.e_tilt, f.tilt[0], f.tilt[1]);
      }
    }
    if (on(f.wrist)) {
      if (!q_meas_.has_value() || secondsSince(q_meas_time_) > f.stale_s) {
        stale = true;
      } else {
        low(2, &f.e_wrist, std::abs((*q_meas_)(3) - r.q_d(3)) * 180.0 / M_PI);
        fac[2] = followFactor(f.e_wrist, f.wrist[0], f.wrist[1]);
      }
    }
    double target = std::min({fac[0], fac[1], fac[2]});
    int limit = 0;
    if (stale) {
      target = 0.0;
      limit = 4;
    } else if (target < 0.999) {
      limit = 1 + static_cast<int>(std::min_element(fac, fac + 3) - fac);
    }
    f.limit = limit;
    f.gov.step(target, dt);
    const double rate = f.gov.rate();
    f.min_rate = std::min(f.min_rate, rate);
    if (rate < 0.95) {f.slowed_s += dt;}
    f.worst[0] = std::max(f.worst[0], f.e_lead);
    f.worst[1] = std::max(f.worst[1], f.e_tilt);
    f.worst[2] = std::max(f.worst[2], f.e_wrist);
    static const char * const kName[] = {"none", "lead", "tilt", "wrist", "a stale feed"};
    if (rate < 0.02 && target < 0.02) {
      f.held_s += dt;
      if (!f.hold_since.has_value()) {f.hold_since = now;}
      if (f.hold_timeout_s > 0.0 && secondsSince(f.hold_since) >= f.hold_timeout_s &&
        !pl_aborting_ && !pl_abort_plan_)
      {
        std::ostringstream m;
        m.setf(std::ios::fixed);
        m.precision(1);
        m << "the turn held " << secondsSince(f.hold_since) << " s on " << kName[limit]
          << " (lead " << f.e_lead << ", tilt " << f.e_tilt << ", wrist " << f.e_wrist
          << " deg) -- releasing";
        pl_guard_reason_ = m.str();
        pl_guard_tripped_ = true;
        f.hold_since.reset();
        *abort = true;
      }
    } else {
      f.hold_since.reset();
    }
    const bool edge = (limit != 0) != (f.logged_limit > 0) || (limit != 0 && limit != f.logged_limit);
    if (edge || (rate < 0.95 && secondsSince(f.last_log) >= 2.0)) {
      f.logged_limit = limit;
      f.last_log = now;
      RCLCPP_INFO(get_logger(), "push-and-pull PATH FOLLOWING: rate %.2f, progress %.1f s, %s "
        "(lead %.1f, tilt %.1f, wrist %.1f deg)", rate, f.gov.tau(),
        limit == 0 ? "on schedule" : (std::string("limited by ") + kName[limit]).c_str(),
        f.e_lead, f.e_tilt, f.e_wrist);
    }
    return f.gov.tau();
  }

  void plFollowEnd(const char * why)
  {
    if (!pl_follow_.has_value()) {return;}
    const PlFollow & f = *pl_follow_;
    RCLCPP_INFO(get_logger(), "push-and-pull PATH FOLLOWING %s: %.1f s for a %.1f s schedule "
      "(progress %.1f s) -- rate min %.2f, slowed %.1f s, held %.1f s; worst lead %.1f, tilt %.1f, "
      "wrist %.1f deg; %d valve flip(s) rejected", why, secondsSince(f.t0), f.plan->duration(),
      f.gov.tau(), f.min_rate, f.slowed_s, f.held_s, f.worst[0], f.worst[1], f.worst[2], f.flips);
    pl_follow_.reset();
  }

  // The tilt guard (push_pull_guard_tilt_deg), with lock_ held: true = abort now.
  bool plGuardCheck()
  {
    const double limit = get_parameter("push_pull_guard_tilt_deg").as_double();
    if (!(limit > 0.0) || !att_R_.has_value()) {return false;}
    const double tilt = std::acos(std::clamp((*att_R_)(2, 2), -1.0, 1.0)) * 180.0 / M_PI;
    if (tilt < limit - 5.0) {
      pl_guard_armed_ = true;
      pl_guard_since_.reset();
      return false;
    }
    const bool engaged = mode_direct_ && hold_.has_value() &&
      (pl_completed_ >= 0 || pl_flying_.has_value() || pl_at_handle_);
    if (tilt <= limit || !engaged || !pl_guard_armed_ || pl_aborting_ || pl_abort_plan_) {
      pl_guard_since_.reset();
      return false;
    }
    if (!pl_guard_since_.has_value()) {pl_guard_since_ = Clock::now();}
    if (secondsSince(pl_guard_since_) < get_parameter("push_pull_guard_persist_s").as_double()) {
      return false;
    }
    pl_guard_armed_ = false;
    pl_guard_since_.reset();
    std::ostringstream m;
    m.setf(std::ios::fixed);
    m.precision(1);
    m << "tilt " << tilt << " deg > " << limit << " deg";
    pl_guard_reason_ = m.str();
    pl_guard_tripped_ = true;
    return true;
  }

  // THE PUSH-FORCE GUARD (2026-10-03): the whole-body node's own reading of
  // the task force (wb_control_debug [97..99], the 4-D attribution's raw F_hat,
  // world [N]) low-passed at kPlForceOmega; above push_pull_guard_force_n for
  // push_pull_guard_force_persist_s while the claw is on the handle in CONTACT
  // -- the box jammed, or the vehicle drifting against the clamp -- the same
  // response as ABORT. The bench's along-arm capacity is ~6 N (interaction
  // campaign, 2026-09-27); the box's friction is ~1.5 N.
  static constexpr double kPlForceOmega = 3.0;   // [rad/s]
  void onPlForce(const std_msgs::msg::Float32MultiArray & msg)
  {
    if (msg.data.size() < 100) {return;}
    const Vec3 F{msg.data[97], msg.data[98], msg.data[99]};
    if (!F.allFinite()) {return;}
    bool trip = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      const auto now = Clock::now();
      // the unload's input: the law's task error while the push flies
      const Vec3 ey{msg.data[24], msg.data[25], msg.data[26]};
      if (pl_flying_.has_value() && *pl_flying_ == kPlPush && ey.allFinite()) {
        pl_ey_buf_.emplace_back(now, ey);
        while (!pl_ey_buf_.empty() &&
          std::chrono::duration<double>(now - pl_ey_buf_.front().first).count() > 3.0)
        {
          pl_ey_buf_.pop_front();
        }
      }
      const double dt = pl_force_t_.has_value()
        ? std::clamp(std::chrono::duration<double>(now - *pl_force_t_).count(), 0.0, 0.1) : 0.0;
      pl_force_t_ = now;
      const double a = std::exp(-kPlForceOmega * dt);
      pl_force_f_ = a * pl_force_f_ + (1.0 - a) * F;
      const double limit = get_parameter("push_pull_guard_force_n").as_double();
      const bool armed = limit > 0.0 && mode_direct_ && pl_contact_ &&
        (pl_at_handle_ || (pl_flying_.has_value() && *pl_flying_ == kPlPush)) &&
        !pl_aborting_ && !pl_abort_plan_;
      if (!armed || pl_force_f_.norm() <= limit) {
        pl_force_since_.reset();
      } else {
        if (!pl_force_since_.has_value()) {pl_force_since_ = now;}
        if (secondsSince(pl_force_since_) >=
          get_parameter("push_pull_guard_force_persist_s").as_double())
        {
          std::ostringstream m;
          m.setf(std::ios::fixed);
          m.precision(1);
          m << "push force " << pl_force_f_.norm() << " N > " << limit << " N";
          pl_guard_reason_ = m.str();
          pl_guard_tripped_ = true;
          pl_force_since_.reset();
          trip = true;
        }
      }
    }
    if (trip) {
      auto resp = std::make_shared<Trigger::Response>();
      onPlAbort(resp);
      RCLCPP_ERROR(get_logger(), "push-and-pull SAFETY GUARD: %s -- %s", pl_guard_reason_.c_str(),
        resp->message.c_str());
    }
  }

  // 10 Hz: how far the step in flight (else the last one flown) is from its
  // target, measured -- the claw (FK on the encoders, odometry position, PX4
  // attitude) for go_to_start (the point above the handle), ready (the grasp
  // point) and push (the pushed grasp point), the body origin otherwise.
  // [0] step, [1] 1 = claw target / 0 = base target, [2] |error| [m],
  // [3..5] error target - measured, world [m], [6] tolerance [m], [7] within,
  // [8] holding (no step in flight, HOLD), [9] the claw's error along the
  // box's x axis -- the jaws' closing axis [m], [10] the low-passed |F_hat| [N],
  // [11] the box's lag behind its plan [m], [12] the box's tilt [deg] (2026-10-16).
  // PATH FOLLOWING (2026-10-18): [13] 1 = the governed turn in flight, [14] its
  // progress rate (0..1), [15] progress [s], [16] lead [deg], [17] tilt [deg],
  // [18] wrist excursion [deg], [19] the limiting channel (0 none, 1 lead,
  // 2 tilt, 3 wrist, 4 a stale feed), [20] time held so far [s].
  void publishPlArrival()
  {
    if (pl_box_point_clear_.exchange(false)) {
      // the capture was dropped (an abort that touched the box): [] is then a
      // no-op in onPlParams, which only tells the arm GS the box row is empty
      set_parameters({rclcpp::Parameter("push_pull_box_point", std::vector<double>{})});
    }
    if (pl_params_dirty_.exchange(false)) {
      publishPlStatus();
      publishPlInfo();
      publishPlPaths(nullptr);
    }
    bool feed_trip = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      feed_trip = plFeedGuardCheck();
      if (pl_abort_anchor_due_.has_value() && Clock::now() >= *pl_abort_anchor_due_) {
        pl_abort_anchor_due_.reset();
        pl_abort_keep_anchor_ = false;
        restoreAnchor();
        RCLCPP_INFO(get_logger(), "push-and-pull ABORT: released -- the claw back to the CoM anchor");
      }
    }
    if (feed_trip) {
      auto resp = std::make_shared<Trigger::Response>();
      onPlAbort(resp);
      RCLCPP_ERROR(get_logger(), "push-and-pull SAFETY GUARD: %s -- %s", pl_guard_reason_.c_str(),
        resp->message.c_str());
    }
    Float64MultiArray m;
    bool changed = false;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      if (!mode_direct_ || !pl_wp_.has_value()) {return;}
      const int s = pl_flying_.has_value() ? *pl_flying_ : pl_completed_;
      if (s < 0) {return;}
      Vec3 x_b;
      Mat3 r0;
      if (!odomPair(&x_b, &r0) || odomAge() > kOdomFreshS || !q_meas_.has_value()) {return;}
      const bool claw = s == kPlGoToStart || s == kPlReady || s == kPlPush;
      Vec3 meas = x_b, want = s == kPlExit ? pl_wp_->exited.x_b :
        s == kPlGoToLandStart ? pl_wp_->land_start.x_b : pl_wp_->land.x_b;
      if (claw) {
        Vec3 r0e;
        armKinematics(*q_meas_, vehicle_->params, nullptr, &r0e, nullptr);
        meas = x_b + r0 * (vehicle_->r_model * r0e);
        want = s == kPlGoToStart ? pl_wp_->approach_ee
                                 : s == kPlReady ? pl_wp_->grasp_ee : pl_wp_->pushed_ee;
      }
      const Vec3 e = want - meas;
      const bool holding = !pl_flying_.has_value() && state_ == "HOLD";
      // the descent trim's samples: the claw's error while holding above
      if (s == kPlGoToStart && holding) {
        const auto now = Clock::now();
        pl_trim_buf_.push_back({now, e});
        const double win = get_parameter("push_pull_descent_trim_window_s").as_double();
        while (!pl_trim_buf_.empty() &&
          std::chrono::duration<double>(now - pl_trim_buf_.front().first).count() > win)
        {
          pl_trim_buf_.pop_front();
        }
      } else {
        pl_trim_buf_.clear();
      }
      const double tol = get_parameter("push_pull_arrival_tol").as_double();
      const bool within = e.norm() <= tol;
      const double yb = pl_box_yaw_.value_or(0.0);
      const double axis = std::cos(yb) * e(0) + std::sin(yb) * e(1);
      pl_last_err_ = e.norm();
      // [11] the box's lag behind its plan [m], [12] its tilt [deg] (the box
      // guards, 2026-10-16; 0 until they are armed)
      m.data = {static_cast<double>(s), claw ? 1.0 : 0.0, e.norm(), e(0), e(1), e(2), tol,
        within ? 1.0 : 0.0, holding ? 1.0 : 0.0, axis, pl_force_f_.norm(), pl_bg_lag_, pl_bg_tilt_};
      const bool fl = plFollowing();
      m.data.insert(m.data.end(), {fl ? 1.0 : 0.0, fl ? pl_follow_->gov.rate() : 1.0,
          fl ? pl_follow_->gov.tau() : 0.0, fl ? pl_follow_->e_lead : 0.0,
          fl ? pl_follow_->e_tilt : 0.0, fl ? pl_follow_->e_wrist : 0.0,
          fl ? static_cast<double>(pl_follow_->limit) : 0.0, fl ? pl_follow_->held_s : 0.0});
      if (within != pl_within_ || s != pl_err_step_ || (s == kPlGoToStart && holding)) {
        pl_within_ = within;
        pl_err_step_ = s;
        changed = true;
      }
    }
    pl_arrival_pub_->publish(m);
    if (changed) {publishPlStatus();}
  }

  void publishPlStatus()
  {
    std::string s;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      std::ostringstream m;
      m.setf(std::ios::fixed);
      m.precision(1);
      const std::string guard = pl_guard_tripped_ ? " (SAFETY GUARD: " + pl_guard_reason_ + ")" : "";
      if (!mode_direct_) {
        m << "NOT IN DIRECT";
      } else if (pl_aborting_ || pl_abort_plan_) {
        m << "ABORTING" << guard << " -- gripper open, contact off, climbing " << std::setprecision(2)
          << pl_abort_climb_ << " m, then the arm home";
      } else if (pl_planning_) {
        m << "PLANNING";
      } else if (pl_flying_.has_value()) {
        m << "FLYING " << pushPullStepName(*pl_flying_);
        if (*pl_flying_ == kPlPush) {m << " (CONTACT)";}
        if (state_ == "EXECUTING" && plan_) {m << " T=" << plan_->duration() << "s";}
      } else if (!pl_error_.empty()) {
        m << "INFEASIBLE: " << pl_error_;
      } else if (pl_aborted_) {
        m << "ABORTED" << guard << " -- holding higher, arm home, contact off; "
          << (pl_wp_.has_value() ? "re-fly go_to_start, or go_to_land_start"
              : pl_leave_wp_.has_value() ? "go_to_land_start to leave the table, or Get the box and press Plan"
              : "Get the box and press Plan");
      } else if (!pl_wp_.has_value() && pl_leave_wp_.has_value() && pl_completed_ >= kPlGoToLandStart) {
        // leaving the table after an abort (2026-10-16): the cleared plan's Land Start / Land
        m << (pl_completed_ >= kPlLand ? std::string("COMPLETE -- touch down with SAFETY -> land")
                                       : std::string("DONE go_to_land_start next=land"))
          << " (off the table after the abort)";
      } else if (!pl_wp_.has_value()) {
        PushPullTargets t;
        std::string err;
        m << "NOT PLANNED: " << (plTargets(&t, &err) ? std::string("press Plan") : err);
      } else if (pl_completed_ >= kPlLand) {
        m << "COMPLETE -- touch down with SAFETY -> land";
      } else if (pl_completed_ < 0) {
        m << "READY next=go_to_start";
      } else if (pl_completed_ == kPlGoToStart) {
        m << "DONE go_to_start next=ready (the gripper opens first) -- claw " << std::setprecision(0)
          << pl_last_err_ * 1e3 << " mm from the point above the handle (tolerance "
          << get_parameter("push_pull_arrival_tol").as_double() * 1e3 << " mm)";
      } else if (pl_completed_ == kPlReady) {
        m << "DONE ready next=push (CONTACT on: the gripper closes first)";
      } else if (pl_completed_ == kPlPush) {
        m << "DONE push next=exit (the gripper opens first)";
      } else {
        m << "DONE " << pushPullStepName(pl_completed_) << " next=" << pushPullStepName(pl_completed_ + 1);
      }
      s = m.str();
    }
    String msg;
    msg.data = s;
    pl_status_pub_->publish(msg);
  }

  // With N = kNumPushPullSteps (6 since 2026-10-16, was 5 -- every index from
  // [13] moved): [0..2] Adjust offset [m], [3] 1 = Go To Land Start / Land are
  // flyable WITHOUT a plan (after an abort that touched the box, 2026-10-16), [4] planned, [5]
  // last step completed (-1 none), [6] step in flight (-1 none), [7] arrival
  // tolerance [m], [8 .. 7+N] each step's planned duration [s], [8+N] box
  // captured (0 / 1), [9+N .. 12+N] the box x, y, z [m], yaw [deg],
  // [13+N + 7k ..] per step k [goal base x, y, z, actual yaw deg, claw x, y,
  // z], then [13+8N] the claw on the handle (0 / 1), [14+8N] the contact
  // phase requested (0 / 1), [15+8N] the world-held claw requested (0 / 1),
  // [16+8N] the push distance [m], [17+8N] the low-passed |F_hat| [N],
  // [18+8N] the drone's yaw at the box [deg], [19+8N] the descent trim of the
  // last Ready [m, norm], [20+8N] the turn angle [deg]. N = 6: T [8..13], box
  // [14] + [15..18], goals [19..60], [61] at handle, [62] contact, [63] world,
  // [64] distance, [65] |F|, [66] drone yaw, [67] trim, [68] turn; 69 values.
  // NaN where there is nothing.
  void publishPlInfo()
  {
    Float64MultiArray m;
    {
      std::lock_guard<std::recursive_mutex> lk(lock_);
      const double nan = std::numeric_limits<double>::quiet_NaN();
      const bool planned = pl_wp_.has_value();
      m.data = {pl_offset_(0), pl_offset_(1), pl_offset_(2),
        (!planned && pl_leave_wp_.has_value()) ? 1.0 : 0.0, planned ? 1.0 : 0.0,
        static_cast<double>(pl_completed_),
        pl_flying_.has_value() ? static_cast<double>(*pl_flying_) : -1.0,
        get_parameter("push_pull_arrival_tol").as_double()};
      for (int k = 0; k < kNumPushPullSteps; ++k) {m.data.push_back(planned ? pl_T_[k] : nan);}
      if (pl_box_.has_value()) {
        m.data.insert(m.data.end(), {1.0, (*pl_box_)(0), (*pl_box_)(1), (*pl_box_)(2),
            pl_box_yaw_.value_or(0.0) * 180.0 / M_PI});
      } else {
        m.data.insert(m.data.end(), {0.0, nan, nan, nan, nan});
      }
      for (int k = 0; k < kNumPushPullSteps; ++k) {
        if (!planned) {
          m.data.insert(m.data.end(), 7, nan);
          continue;
        }
        const PushPullWaypoints & w = *pl_wp_;
        const RestSpec & g = k == kPlGoToStart ? w.approach : k == kPlReady ? w.grasp :
          k == kPlPush ? w.pushed : k == kPlExit ? w.exited :
          k == kPlGoToLandStart ? w.land_start : w.land;
        const Vec3 e = clawOf(*vehicle_, g);
        m.data.insert(
          m.data.end(), {g.x_b(0), g.x_b(1), g.x_b(2), (g.phi + 0.5 * M_PI) * 180.0 / M_PI,
            e(0), e(1), e(2)});
      }
      m.data.push_back(pl_at_handle_ ? 1.0 : 0.0);
      m.data.push_back(pl_contact_ ? 1.0 : 0.0);
      m.data.push_back(pp_anchor_world_ ? 1.0 : 0.0);
      m.data.push_back(get_parameter("push_pull_push_distance").as_double());
      m.data.push_back(pl_force_f_.norm());
      m.data.push_back(planned ? pl_wp_->drone_yaw * 180.0 / M_PI : nan);
      m.data.push_back(pl_trim_.norm());
      // [20+8N] the turn angle [deg] (0 = a straight push), 2026-10-15
      m.data.push_back(get_parameter("push_pull_turn_deg").as_double());
    }
    pl_info_pub_->publish(m);
  }

  void publishPlPaths(const Trajectory * whole)
  {
    Float64MultiArray a, b;
    if (whole != nullptr) {
      eePathRows(*whole, 600, &a.data);
      dronePathRows(*whole, 600, &b.data);
    }
    pl_path_pub_->publish(a);
    pl_drone_path_pub_->publish(b);
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
  // IDLE HOLD PENDING CALCULATING PLANNED INFEASIBLE EXECUTING TELEOP TELEOP_STOP
  std::string state_{"IDLE"};
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
  std::array<std::optional<double>, kNumPickPlaceLegs> pp_capture_yaw_;   // [rad], with each capture
  struct PpSample {Clock::time_point t; Vec3 p; double yaw; double tilt;};  // a mocap sample
  std::array<std::deque<PpSample>, kNumPickPlaceLegs> pp_mocap_;
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
  // PS4 teleoperation
  std::unique_ptr<TeleopReference> teleop_;
  std::optional<Clock::time_point> teleop_last_tick_;
  std::optional<WbReference> teleop_last_ref_;
  bool teleop_armed_{false};
  bool teleop_home_request_{false};
  bool home_prev_{false};
  std::string teleop_note_;
  bool teleop_state_cleared_{false};
  std::optional<Joy> joy_;
  std::optional<Clock::time_point> joy_time_;
  std::vector<int64_t> teleop_axes_, teleop_buttons_;
  std::vector<double> teleop_axis_signs_;
  std::atomic<bool> pp_params_dirty_{false};
  // a leg's phase: travel (a claw leg: to the point above its target), the
  // wait above it, the vertical descent
  enum PpPhase {kPpNone = 0, kPpTravel = 1, kPpWait = 2, kPpDescent = 3};
  PpPhase pp_phase_{kPpNone};
  Clock::time_point pp_wait_t0_{};
  std::optional<Clock::time_point> pp_within_since_;
  std::deque<std::pair<Clock::time_point, Vec3>> pp_trim_buf_;   // claw error while holding above
  Vec3 pp_descent_trim_{Vec3::Zero()};                          // the trim the last descent used
  double pp_last_err_{0.0};
  // the autonomous drop (pick_place_drop_auto): when the drop leg ended, since
  // when every gate has passed, whether this drop hold has already released,
  // the claw's last position (its speed), and the status text it shows
  std::optional<Clock::time_point> pp_drop_hold_t0_;
  std::optional<Clock::time_point> pp_drop_ok_since_;
  bool pp_drop_fired_{false};
  std::optional<std::pair<Clock::time_point, Vec3>> pp_drop_claw_prev_;
  std::optional<Clock::time_point> pp_drop_last_eval_;   // a gap restarts the dwell
  std::string pp_drop_text_;
  // aiming the drop at the basket (pick_place_drop_aim_payload): the basket -
  // claw offset in the VEHICLE frame (x forward, y left) sampled at Drop Start,
  // the world shift currently applied to the drop leg's rest (undone before a
  // new one, and zero on a fresh plan), whether it was applied (the claw
  // position gate stands down only then), and what was done, for the leg reply
  std::deque<std::pair<Clock::time_point, Vec3>> pp_aim_buf_;
  Vec3 pp_aim_shift_{Vec3::Zero()};
  bool pp_aim_active_{false};
  std::optional<Clock::time_point> pp_aim_hold_since_;   // the Drop Start hover began (settle)
  std::string pp_aim_text_;
  bool pp_at_table_{false};                     // holding at a table: the next leg retreats first
  bool pp_within_{false};                       // the claw inside the tolerance (dwell not yet done)
  bool pp_timed_out_{false};                    // pp_error_ is a wait timeout, not a refusal
  double pp_abort_climb_{0.0};                  // the (fence-clipped) climb of the abort in flight
  bool pp_abort_release_{false};                // that abort goes through the release pose
  std::shared_ptr<Trajectory> pp_abort_plan_;   // the abort transition in flight
  bool pp_guard_armed_{true};                    // the tilt guard may fire (re-armed below limit - 5 deg)
  bool pp_guard_tripped_{false};                 // the abort in flight / held was the guard's
  std::optional<Clock::time_point> pp_guard_since_;   // tilt above the limit since
  std::string pp_guard_reason_;
  std::optional<int> pp_exiting_;                // the claw leg whose exit is planning / flying
  bool pp_exit_to_land_{false};                  // ... and it ends on Land Start (the drop's)
  std::shared_ptr<Trajectory> pp_exit_plan_;     // ... and its climb
  std::array<bool, kNumPickPlaceLegs> pp_exited_{};   // a claw leg's exit flown
  std::array<double, kNumPickPlaceLegs> pp_exit_T_{}; // planned exit durations (Plan) [s]
  bool pp_aborting_{false};                     // ... or being planned
  bool pp_aborted_{false};                      // until the next pick-and-place action
  // phase-dependent EE anchor (pick_place_world_anchor_*)
  rclcpp::Client<SetBool>::SharedPtr anchor_cli_;
  bool pp_anchor_world_{false};                 // the whole-body node was asked for WORLD
  OnSetParametersCallbackHandle::SharedPtr pp_param_cb_;
  // push-and-pull mode (push_pull.hpp)
  std::string pl_box_topic_;                    // the box's mocap body
  std::deque<PpSample> pl_mocap_;
  // PATH FOLLOWING (2026-10-18, push_pull_follow): the governed turn in flight
  struct PlFollow
  {
    std::shared_ptr<const Trajectory> plan;   // the push plan it governs
    ProgressGovernor gov;
    Clock::time_point t0, last;
    Vec3 axis{Vec3::Zero()};
    double sign{1.0};          // the turn's direction
    double alpha0{0.0};        // the planned claw's bearing about the axis at tau = 0
    double alpha{0.0};         // the planned rotation so far [rad] (unwrapped)
    double alpha_prev{0.0};
    double yaw0{0.0};          // the valve's yaw at the start [rad]
    bool have_yaw0{false};
    double lead_prev{0.0};
    bool have_lead{false};
    int flips{0};              // valve samples rejected as orientation flips
    double lead[2]{0.0, 0.0}, tilt[2]{0.0, 0.0}, wrist[2]{0.0, 0.0};   // [soft, hard] [deg]
    double stale_s{0.3}, hold_timeout_s{0.0}, filter_s{1.0};
    double e_lead{0.0}, e_tilt{0.0}, e_wrist{0.0};   // the channels now, low-passed [deg]
    bool f_init[3]{false, false, false};
    int limit{0};              // 0 none, 1 lead, 2 tilt, 3 wrist, 4 stale
    double held_s{0.0}, slowed_s{0.0}, min_rate{1.0};
    double worst[3]{0.0, 0.0, 0.0};
    std::optional<Clock::time_point> hold_since;
    int logged_limit{-1};
    Clock::time_point last_log;
  };
  std::optional<PlFollow> pl_follow_;
  std::optional<Vec3> pl_box_;                  // captured box CG ...
  std::optional<double> pl_box_yaw_;            // ... and yaw [rad]
  Vec3 pl_offset_{Vec3::Zero()};                // the last Adjust shift
  std::optional<PushPullWaypoints> pl_wp_;      // set by Plan
  // the plan an abort that touched the box cleared, kept ONLY for leaving the table
  // (2026-10-16): Go To Land Start / Land need its Land Start, land point and push
  // heading, none of which a moved box makes stale -- re-planning the whole mission
  // from a box already pushed part-way can be refused (the box-on-table check), which
  // left the vehicle hovering beside the table with no planned way off it
  std::optional<PushPullWaypoints> pl_leave_wp_;
  std::array<double, kNumPushPullSteps> pl_T_{};
  bool pl_planning_{false};
  std::string pl_error_;
  int pl_completed_{-1};                        // the last step flown to its end
  std::optional<int> pl_flying_;                // the step planning / executing
  std::shared_ptr<Trajectory> pl_exec_plan_;    // ... and its trajectory
  bool pl_at_handle_{false};                    // the claw on the handle (ready / push done)
  bool pl_aborting_{false};                     // the abort being planned
  bool pl_aborted_{false};                      // until the next push-and-pull action
  double pl_abort_climb_{0.0};
  bool pl_abort_keep_anchor_{false};                         // abort_release_world_held in force
  std::optional<Clock::time_point> pl_abort_anchor_due_;     // when the claw goes back to the CoM
  std::shared_ptr<Trajectory> pl_abort_plan_;   // the abort in flight
  bool pl_contact_{false};                      // the contact phase flag requested
  bool pl_guard_armed_{true}, pl_guard_tripped_{false};
  std::optional<Clock::time_point> pl_guard_since_;
  std::string pl_guard_reason_;
  Vec3 pl_force_f_{Vec3::Zero()};               // the low-passed F_hat [N]
  // the box guards (2026-10-16): the box's pose when the push started
  // executing, and each condition's since-time
  std::optional<Vec3> pl_bg_box0_;
  Mat3 pl_bg_R0_{Mat3::Identity()};
  std::optional<Clock::time_point> pl_bg_lag_since_, pl_bg_tilt_since_, pl_bg_q3_since_;
  std::optional<Vec3> pl_bg_lag0_;               // the box when the push started executing
  unsigned pl_bg_lag_gen_{0};
  double pl_bg_lag_{0.0}, pl_bg_tilt_{0.0};      // the last values (for the info line)
  std::optional<Clock::time_point> pl_force_t_, pl_force_since_;
  std::deque<std::pair<Clock::time_point, Vec3>> pl_trim_buf_;   // claw error holding above
  std::deque<std::pair<Clock::time_point, Vec3>> pl_ey_buf_;     // the law's e_y in the push
  bool pl_unloaded_{false};                     // the push's unload has been started
  Vec3 pl_trim_{Vec3::Zero()};                  // the trim the last Ready used
  double pl_last_err_{0.0};
  bool pl_within_{false};
  int pl_err_step_{-1};
  unsigned pl_gen_{0};
  std::thread pl_worker_;
  std::atomic<bool> pl_params_dirty_{false};
  std::atomic<bool> pl_box_point_clear_{false};   ///< clear push_pull_box_point (box dropped)
  rclcpp::Client<SetBool>::SharedPtr contact_cli_;
  OnSetParametersCallbackHandle::SharedPtr pl_param_cb_;

  // ---- live samples --------------------------------------------------------
  std::optional<Vec3> odom_p_;
  std::optional<Clock::time_point> odom_p_time_;
  std::optional<Mat3> att_R_;
  std::optional<Clock::time_point> att_time_;
  bool att_warned_{false};
  std::optional<VecN> q_meas_;
  Clock::time_point q_meas_time_{};   // when q_meas_ arrived (the drop gates and the aim need it fresh)
  VecN qdot_meas_{VecN::Zero()};
  int viz_decim_{0};
  int viz_decim_period_{5};

  // ---- ROS ------------------------------------------------------------------
  rclcpp::Publisher<WholeBodyReference>::SharedPtr ref_pub_;
  rclcpp::Publisher<String>::SharedPtr status_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr base_pub_, cur_ee_pub_, cur_ee_body_pub_,
    cur_base_pub_;
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
  rclcpp::Service<Trigger>::SharedPtr ee_origin_srv_;
  rclcpp::TimerBase::SharedPtr ee_err_timer_;
  rclcpp::Publisher<String>::SharedPtr pp_status_pub_;
  rclcpp::Publisher<String>::SharedPtr pp_drop_now_pub_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr pp_info_pub_, pp_path_pub_, pp_drone_path_pub_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr pp_arrival_pub_, pp_ws_pub_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> pp_srvs_;
  std::vector<rclcpp::Subscription<Mocap>::SharedPtr> pp_mocap_subs_;
  rclcpp::TimerBase::SharedPtr pp_timer_;
  rclcpp::Subscription<Joy>::SharedPtr joy_sub_;
  rclcpp::Service<SetBool>::SharedPtr teleop_engage_srv_;
  rclcpp::Service<Trigger>::SharedPtr teleop_home_srv_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr teleop_state_pub_, skeleton_pub_;
  rclcpp::Publisher<String>::SharedPtr teleop_note_pub_;
  rclcpp::Publisher<String>::SharedPtr pl_status_pub_;
  rclcpp::Publisher<Float64MultiArray>::SharedPtr pl_info_pub_, pl_path_pub_, pl_drone_path_pub_,
    pl_arrival_pub_;
  std::vector<rclcpp::Service<Trigger>::SharedPtr> pl_srvs_;
  rclcpp::Subscription<Mocap>::SharedPtr pl_mocap_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr pl_force_sub_;
  rclcpp::TimerBase::SharedPtr pl_timer_;
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
