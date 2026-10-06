"""quadrotor_ccm_planner: CCM reference (x*, u*) for the torque-mode CCM stack."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution(
        [FindPackageShare("fsc_trajectory_planner"), "config", "quadrotor_ccm_planner_t650.yaml"])
    return LaunchDescription([
        DeclareLaunchArgument("uav_prefix", default_value="uav_0",
                              description="Namespace of the vehicle"),
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="quadrotor_ccm_planner parameters"),
        Node(
            package="fsc_trajectory_planner",
            executable="quadrotor_ccm_planner",
            name="quadrotor_ccm_planner",
            namespace=LaunchConfiguration("uav_prefix"),
            parameters=[LaunchConfiguration("params_file")],
            output="screen",
        ),
    ])
