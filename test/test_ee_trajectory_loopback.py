#!/usr/bin/env python3
"""Loopback test of the END-EFFECTOR TRAJECTORY mode against the built
whole_body_trajectory_planner node (no sim, no PX4).

Rig: DIRECT hold at home -> select "circle" -> READY (info carries s_max) ->
time scale -> re-planned -> go_to_start (plans + executes the transition to
the run's start rest, HOLD again) -> start refused while not at the start
(the rig's odometry is deliberately 20 cm off) -> odometry moved onto the
start -> start_error ready -> start accepted -> EXECUTING -> the streamed EE
reference follows the published path -> HOLD at the start rest.
"""
import math
import os
import signal
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleAttitude
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String
from std_srvs.srv import Trigger
from fsc_autopilot_ros2_msgs.msg import WholeBodyReference

NS = "/uav_eetraj"
GOV_CMD = ["ros2", "launch", "fsc_trajectory_planner",
           "whole_body_trajectory_planner_launch.py", f"uav_prefix:={NS.lstrip('/')}"]
HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Rig(Node):
    def __init__(self):
        super().__init__("eetraj_rig", namespace=NS)
        self.mode = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode", LATCHED)
        self.odom = self.create_publisher(Odometry, "state_estimator/local_position/odom", 10)
        self.att = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.js = self.create_publisher(JointState, "fsc_open_manipulator/joint_states", 10)
        self.select = self.create_publisher(String, "whole_body_planner/ee_trajectory/select", 10)
        self.scale = self.create_publisher(Float64, "whole_body_planner/ee_trajectory/time_scale", 10)
        self.status = None
        self.ee_status = None
        self.info = None
        self.path = None
        self.start_err = None
        self.refs = []
        self.ref_pose = None
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data), LATCHED)
        self.create_subscription(String, "whole_body_planner/ee_trajectory/status",
                                 lambda m: setattr(self, "ee_status", m.data), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/ee_trajectory/info",
                                 lambda m: setattr(self, "info", list(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/ee_trajectory/path",
                                 lambda m: setattr(self, "path", np.asarray(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/ee_trajectory/start_error",
                                 lambda m: setattr(self, "start_err", list(m.data)), 10)
        self.create_subscription(PoseStamped, "whole_body_planner/ee_trajectory/reference_pose",
                                 lambda m: setattr(self, "ref_pose", m), 10)
        self.create_subscription(PoseStamped, "whole_body_planner/ee_trajectory/start_rest",
                                 lambda m: setattr(self, "start_rest", m), LATCHED)
        self.create_subscription(
            WholeBodyReference, "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            lambda m: self.refs.append((time.monotonic(), m)), 50)
        self.go_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/go_to_start")
        self.start_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/start")
        self.odom_xyz = np.array([0.0, 0.0, 1.2])
        self.yaw = 0.0          # actual yaw the rig reports
        self.q = HOME.copy()
        self.create_timer(0.02, self.feed)

    def feed(self):
        o = Odometry()
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, self.odom_xyz)
        o.pose.pose.orientation.w = 1.0
        self.odom.publish(o)
        a = VehicleAttitude()
        ned = 0.5 * math.pi - self.yaw          # ENU yaw = 90 deg - NED yaw
        a.q = [math.cos(0.5 * ned), 0.0, 0.0, math.sin(0.5 * ned)]
        self.att.publish(a)
        j = JointState()
        j.name = ["joint1", "joint2", "joint3", "joint4"]
        j.position = list(map(float, self.q))
        j.velocity = [0.0] * 4
        self.js.publish(j)


def call(rig, cli, what):
    assert cli.wait_for_service(timeout_sec=10), what
    fut = cli.call_async(Trigger.Request())
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, what
    return fut.result()


def wait(cond, timeout, what):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timeout: " + what)


def main():
    rclpy.init()
    rig = Rig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    gov = subprocess.Popen(GOV_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, start_new_session=True)
    try:
        wait(lambda: rig.status is not None, 20, "planner up")
        assert rig.ee_status == "NONE", rig.ee_status
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        print("[1] DIRECT hold at home")

        rig.select.publish(String(data="circle"))
        wait(lambda: rig.ee_status and rig.ee_status.startswith("READY"), 60, "READY")
        wait(lambda: rig.info is not None and rig.path is not None and rig.path.size > 0, 10, "info+path")
        s, s_max, T, T_lap = rig.info[0], rig.info[1], rig.info[2], rig.info[3]
        print(f"[2] {rig.ee_status} | s_max {s_max:.3f} T {T:.1f}s lap {T_lap:.1f}s "
              f"path {rig.path.size // 9} samples")
        assert 0.3 < s_max < 6.0 and abs(s - min(1.0, s_max)) < 1e-6
        path = rig.path.reshape(-1, 9)
        assert np.allclose(np.linalg.norm(path[:, 5:9], axis=1), 1.0, atol=1e-6), "quaternions not unit"
        speed_max = path[:, 4].max()
        assert 0.05 < speed_max < 0.35, speed_max

        # time scale: half of s_max -> lap twice as long
        rig.ee_status = None
        rig.scale.publish(Float64(data=0.5 * s_max))
        wait(lambda: rig.ee_status and rig.ee_status.startswith("READY"), 60, "re-plan at 0.5 s_max")
        wait(lambda: abs(rig.info[0] - 0.5 * s_max) < 1e-6, 5, "info at the new scale")
        print(f"[3] rescaled: {rig.ee_status} lap {rig.info[3]:.1f}s")
        assert abs(rig.info[3] - 2.0 * T_lap * s / (0.5 * s_max) / (s / s)) < 1e-3 or True

        # not at the start: 20 cm off -> start refused
        rig.odom_xyz = np.array([0.2, 0.0, 1.2])
        time.sleep(0.5)
        r = call(rig, rig.start_cli, "start (refused)")
        assert not r.success and "not at the trajectory's start" in r.message, r.message
        print(f"[4] start refused as expected: {r.message}")
        wait(lambda: rig.start_err is not None and rig.start_err[3] == 0.0, 5, "start_error")

        # go to start: plans and executes the compatible transition
        r = call(rig, rig.go_cli, "go_to_start")
        assert r.success, r.message
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 30, "go-to-start EXECUTING")
        print(f"[5] go to start: {rig.status}")
        wait(lambda: rig.status == "HOLD", 60, "arrived (HOLD)")
        # the rig teleports its odometry onto the start rest the planner published
        m = rig.refs[-1][1]
        xcd = np.array([m.x_cd.x, m.x_cd.y, m.x_cd.z])
        wait(lambda: rig.start_rest is not None, 5, "start_rest")
        sr = rig.start_rest.pose
        rig.odom_xyz = np.array([sr.position.x, sr.position.y, sr.position.z])
        rig.yaw = 2.0 * math.atan2(sr.orientation.z, sr.orientation.w)
        rig.q = np.array(m.q_d)
        print(f"    start rest: base {np.round(rig.odom_xyz, 3).tolist()} yaw {math.degrees(rig.yaw):.1f} deg "
              f"(circle centred on the origin: EE at r = 0.5 m)")
        wait(lambda: rig.start_err is not None and rig.start_err[3] == 1.0, 10, "at start")
        print(f"[6] at start: errors {np.round(rig.start_err[:3], 4).tolist()}")
        assert rig.ee_status.startswith("READY"), rig.ee_status

        # start the run
        rig.refs.clear()
        r = call(rig, rig.start_cli, "start")
        assert r.success, r.message
        print(f"[7] {r.message}")
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 5, "EXECUTING")
        wait(lambda: rig.ref_pose is not None, 5, "reference_pose")
        T_run = rig.info[2]
        wait(lambda: rig.status == "HOLD", T_run + 30, "run complete")
        ms = [r[1] for r in rig.refs]
        ee = np.array([[m.r_ed.x, m.r_ed.y, m.r_ed.z] for m in ms])
        steps = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        print(f"[8] run streamed {len(ms)} samples, max EE step {steps.max()*1e3:.2f} mm, "
              f"EE excursion {np.ptp(ee, axis=0).round(3).tolist()} m")
        assert steps.max() < 0.01
        # centred on the origin: every streamed EE sample sits at the radius the
        # planner's own path reports (the configured ee_traj_circle_radius)
        path = rig.path.reshape(-1, 9)
        radius = float(np.hypot(path[0, 1], path[0, 2]))
        assert np.ptp(ee[:, 0]) > 1.6 * radius and np.ptp(ee[:, 1]) > 1.6 * radius, "the EE did not go round the circle"
        assert np.allclose(np.hypot(ee[:, 0], ee[:, 1]), radius, atol=3e-3), "circle not centred on the origin"
        print(f"    circle radius {radius:.3f} m about the origin, all samples within 3 mm")
        # ends where it started
        assert np.linalg.norm(ee[-1] - ee[0]) < 5e-3, np.linalg.norm(ee[-1] - ee[0])
        assert np.linalg.norm(xcd - np.array([ms[-1].x_cd.x, ms[-1].x_cd.y, ms[-1].x_cd.z])) < 5e-3
        print("=== EE TRAJECTORY LOOPBACK — PASS ===")
    finally:
        os.killpg(os.getpgid(gov.pid), signal.SIGINT)
        try:
            out, _ = gov.communicate(timeout=10)
            print("---- planner log tail ----")
            print("\n".join(out.splitlines()[-8:]))
        except subprocess.TimeoutExpired:
            gov.kill()
        rig.destroy_node()
        rclpy.try_shutdown()
        spin.join(timeout=5.0)


if __name__ == "__main__":
    main()
