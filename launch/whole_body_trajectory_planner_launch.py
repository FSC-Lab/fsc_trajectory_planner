"""Launch the whole-body trajectory planner (rclcpp node) for one vehicle.

    ros2 launch fsc_trajectory_planner whole_body_trajectory_planner_launch.py \
        uav_prefix:=uav_0 [params_file:=<yaml>] [planner:=bspline]

`uav_prefix` becomes the node's namespace, so every topic and service the
node offers (whole_body_planner/*, the WholeBodyReference stream, the arm
reference) and every input it consumes lives under /<uav_prefix>/... -- one
planner per vehicle, side by side.

`params_file` may be ANY yaml that carries a `/**/whole_body_trajectory_planner:`
section: this package's config, or the flight node's own yaml so one file
describes the whole run. `planner:=`, `vehicle:=`, `hold_ee_world:=`,
`base_com:=` and `arm_joint_sign:=` override the file when given; an empty
value leaves the yaml's setting alone.

`mode_topic:=` and `arm_reference_topic:=` (2026-09-26) override the two
topics that differ between the WHOLE-BODY and the DECOUPLED flight stacks --
which flight node's /mode this planner follows, and which arm controller's
reference_joint_trajectory receives the joint half of the plan -- so ONE yaml
section (the whole-body flight yaml's) serves both rigs from this launch file:
    mode_topic:=fsc_autopilot_ros2/geometric_l1_direct_actuation/mode \
    arm_reference_topic:=fsc_open_manipulator/position_controller/reference_joint_trajectory
Empty (the default) leaves the yaml's whole-body wiring alone.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_node(context):
    def arg(name):
        return LaunchConfiguration(name).perform(context).strip()

    params = [arg("params_file")]
    overrides = {}
    if arg("planner"):
        overrides["planner"] = arg("planner")
    if arg("vehicle"):
        overrides["vehicle"] = arg("vehicle")
    if arg("hold_ee_world"):
        overrides["hold_ee_world"] = arg("hold_ee_world").lower() in ("1", "true")
    if arg("base_com"):
        overrides["base_com"] = [float(v) for v in
                                 arg("base_com").strip("[]").split(",")]
    if arg("arm_joint_sign"):
        overrides["arm_joint_sign"] = [float(v) for v in
                                       arg("arm_joint_sign").strip("[]").split(",")]
    if arg("mode_topic"):
        overrides["mode_topic"] = arg("mode_topic")
    if arg("arm_reference_topic"):
        overrides["arm_reference_topic"] = arg("arm_reference_topic")
    if overrides:
        params.append(overrides)
    return [Node(
        package="fsc_trajectory_planner",
        executable="whole_body_trajectory_planner",
        name="whole_body_trajectory_planner",
        namespace=arg("uav_prefix"),
        output="screen",
        emulate_tty=True,
        parameters=params,
    )]


def generate_launch_description():
    pkg = get_package_share_directory("fsc_trajectory_planner")
    default_params = os.path.join(
        pkg, "config", "whole_body_trajectory_planner_t650_aerial_manipulator.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("uav_prefix", default_value="uav_0",
                              description="vehicle namespace (uav_0, uav_1, ...)"),
        DeclareLaunchArgument("params_file", default_value=default_params),
        DeclareLaunchArgument("planner", default_value="",
                              description="override the yaml's planner backend"),
        DeclareLaunchArgument("vehicle", default_value="",
                              description="override the yaml's vehicle model"),
        DeclareLaunchArgument("hold_ee_world", default_value="",
                              description="override hold_ee_world (true/false)"),
        DeclareLaunchArgument("base_com", default_value="",
                              description="override base_com, e.g. [0.0, 0.0, 0.0]"),
        DeclareLaunchArgument("arm_joint_sign", default_value="",
                              description="override arm_joint_sign, e.g. [-1,1,1,-1]"),
        DeclareLaunchArgument("mode_topic", default_value="",
                              description="override mode_topic (the flight node's /mode "
                                          "this planner follows; the decoupled stack passes "
                                          "fsc_autopilot_ros2/geometric_l1_direct_actuation/mode)"),
        DeclareLaunchArgument("arm_reference_topic", default_value="",
                              description="override arm_reference_topic (the arm controller's "
                                          "reference_joint_trajectory; the decoupled stack passes "
                                          "fsc_open_manipulator/position_controller/"
                                          "reference_joint_trajectory)"),
        OpaqueFunction(function=_launch_node),
    ])
