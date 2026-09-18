#!/usr/bin/env python3
"""Loopback test of the EE trajectory mode's PAUSE / RESUME / BACK TO ORIGIN
services against the built whole_body_trajectory_planner node (no sim, no PX4).

Sibling of test_ee_trajectory_loopback.py, which covers select -> go_to_start
-> start -> HOLD. This one covers what the arm ground station's button row adds
to that:

  pause refused while nothing is executing -> run started -> PAUSE ->
  the streamed reference FREEZES on a point of the run (not on a new hold) and
  the status carries the PAUSED suffix while its first token stays EXECUTING ->
  a second pause refused -> RESUME -> the reference continues from where it
  froze, without a jump -> run completes -> BACK TO ORIGIN plans [0, 0, z] with
  the arm home and STOPS at PLANNED (it does not fly on the button press) ->
  the generic send executes it.

The freeze is the point of the test: the reference has to stay on the planned
trajectory rather than fall back to a hold, because that is what makes resume
exact.
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
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from fsc_autopilot_ros2_msgs.msg import WholeBodyReference

NS = "/uav_eepause"
GOV_CMD = ["ros2", "launch", "fsc_trajectory_planner",
           "whole_body_trajectory_planner_launch.py", f"uav_prefix:={NS.lstrip('/')}"]
HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Rig(Node):
    def __init__(self):
        super().__init__("eepause_rig", namespace=NS)
        self.mode = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode", LATCHED)
        self.odom = self.create_publisher(Odometry, "state_estimator/local_position/odom", 10)
        self.att = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.js = self.create_publisher(JointState, "fsc_open_manipulator/joint_states", 10)
        self.select = self.create_publisher(String, "whole_body_planner/ee_trajectory/select", 10)
        self.status = None
        self.ee_status = None
        self.info = None
        self.start_err = None
        self.start_rest = None
        self.drone_path = None
        self.drone_pose = None
        self.refs = []
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data), LATCHED)
        self.create_subscription(String, "whole_body_planner/ee_trajectory/status",
                                 lambda m: setattr(self, "ee_status", m.data), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/ee_trajectory/info",
                                 lambda m: setattr(self, "info", list(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/ee_trajectory/start_error",
                                 lambda m: setattr(self, "start_err", list(m.data)), 10)
        self.create_subscription(PoseStamped, "whole_body_planner/ee_trajectory/start_rest",
                                 lambda m: setattr(self, "start_rest", m), LATCHED)
        self.create_subscription(
            Float64MultiArray, "whole_body_planner/ee_trajectory/drone_path",
            lambda m: setattr(self, "drone_path", np.asarray(m.data)), LATCHED)
        self.create_subscription(
            PoseStamped, "whole_body_planner/ee_trajectory/drone_reference_pose",
            lambda m: setattr(self, "drone_pose", m), 10)
        self.create_subscription(
            WholeBodyReference, "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            lambda m: self.refs.append((time.monotonic(), m)), 200)
        self.go_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/go_to_start")
        self.start_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/start")
        self.pause_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/pause")
        self.resume_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/resume")
        self.origin_cli = self.create_client(
            Trigger, "whole_body_planner/ee_trajectory/back_to_origin")
        self.send_cli = self.create_client(Trigger, "whole_body_planner/send")
        self.odom_xyz = np.array([0.0, 0.0, 1.2])
        self.yaw = 0.0
        self.q = HOME.copy()
        self.create_timer(0.02, self.feed)

    def feed(self):
        o = Odometry()
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = \
            map(float, self.odom_xyz)
        o.pose.pose.orientation.w = 1.0
        self.odom.publish(o)
        a = VehicleAttitude()
        ned = 0.5 * math.pi - self.yaw
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


def ee_of(m):
    return np.array([m.r_ed.x, m.r_ed.y, m.r_ed.z])


def main():
    rclpy.init()
    rig = Rig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    gov = subprocess.Popen(GOV_CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, start_new_session=True)
    try:
        wait(lambda: rig.status is not None, 20, "planner up")
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        print("[1] DIRECT hold at home")

        # pause is refused when there is nothing to pause
        r = call(rig, rig.pause_cli, "pause (idle)")
        assert not r.success and "nothing is executing" in r.message, r.message
        r = call(rig, rig.resume_cli, "resume (idle)")
        assert not r.success and "not paused" in r.message, r.message
        print("[2] pause and resume refused while idle")

        rig.select.publish(String(data="circle"))
        wait(lambda: rig.ee_status and rig.ee_status.startswith("READY"), 90, "READY")
        wait(lambda: rig.info is not None, 10, "info")
        print(f"[3] {rig.ee_status}")

        # the drone path is published beside the EE path, same 9-double stride
        wait(lambda: rig.drone_path is not None and rig.drone_path.size > 0, 10, "drone_path")
        dp = rig.drone_path.reshape(-1, 9)
        assert np.allclose(np.linalg.norm(dp[:, 5:9], axis=1), 1.0, atol=1e-6), "drone quats"
        assert dp[:, 4].min() >= 0.0 and dp[:, 4].max() < 1.0, dp[:, 4].max()
        print(f"[4] drone path {dp.shape[0]} samples, speed <= {dp[:, 4].max():.3f} m/s")

        # fly to the start, then teleport the rig onto it
        r = call(rig, rig.go_cli, "go_to_start")
        assert r.success, r.message
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 30, "go EXECUTING")
        wait(lambda: rig.status == "HOLD", 90, "arrived")
        sr = rig.start_rest.pose
        rig.odom_xyz = np.array([sr.position.x, sr.position.y, sr.position.z])
        rig.yaw = 2.0 * math.atan2(sr.orientation.z, sr.orientation.w)
        rig.q = np.array(rig.refs[-1][1].q_d)
        wait(lambda: rig.start_err is not None and rig.start_err[3] == 1.0, 15, "at start")
        # the tolerances now ride along with the errors (the GS shows them)
        assert len(rig.start_err) >= 7, rig.start_err
        assert rig.start_err[4] > 0.0 and rig.start_err[5] > 0.0 and rig.start_err[6] > 0.0
        print(f"[5] at start; tolerances {np.round(rig.start_err[4:7], 3).tolist()}")

        # start, let it run, then PAUSE
        r = call(rig, rig.start_cli, "start")
        assert r.success, r.message
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 5, "EXECUTING")
        time.sleep(4.0)
        r = call(rig, rig.pause_cli, "pause")
        assert r.success, r.message
        print(f"[6] {r.message}")
        wait(lambda: rig.status and "PAUSED" in rig.status, 3, "PAUSED status")
        # the first token must stay EXECUTING: every consumer tests that
        assert rig.status.split()[0] == "EXECUTING", rig.status

        r = call(rig, rig.pause_cli, "pause (twice)")
        assert not r.success and "already paused" in r.message, r.message

        # the reference must FREEZE, and freeze on the run
        rig.refs.clear()
        time.sleep(3.0)
        frozen = [ee_of(m) for _, m in rig.refs]
        assert len(frozen) > 100, len(frozen)
        spread = np.ptp(np.asarray(frozen), axis=0)
        assert spread.max() < 1e-9, spread
        held = frozen[-1]
        print(f"[7] reference frozen for {len(frozen)} samples, spread {spread.max():.2e} m")

        # RESUME continues from exactly there. Capture ACROSS the resume — from
        # still-frozen, through the instant itself, into the motion — because
        # the property that matters is that the reference does not jump there,
        # and a window opened after the call has already missed it.
        rig.refs.clear()
        time.sleep(0.5)
        r = call(rig, rig.resume_cli, "resume")
        assert r.success, r.message
        time.sleep(2.0)
        seq = np.asarray([ee_of(m) for _, m in rig.refs])
        assert len(seq) > 100, len(seq)
        assert np.linalg.norm(seq[0] - held) < 1e-9, "the tail of the pause moved"
        step = np.linalg.norm(np.diff(seq, axis=0), axis=1)
        assert step.max() < 0.01, f"jump on resume: {step.max()*1e3:.1f} mm"
        assert np.ptp(seq, axis=0).max() > 1e-3, "reference did not restart"
        assert "PAUSED" not in rig.status, rig.status
        print(f"[8] resumed with no jump: max step across the resume "
              f"{step.max()*1e3:.2f} mm over {len(seq)} samples")

        wait(lambda: rig.status == "HOLD", rig.info[2] + 60, "run complete")
        print("[9] run complete, HOLD")

        # BACK TO ORIGIN: plans, and stops at PLANNED
        r = call(rig, rig.origin_cli, "back_to_origin")
        assert r.success, r.message
        print(f"[10] {r.message}")
        wait(lambda: rig.status and rig.status.startswith("PLANNED"), 60, "PLANNED")
        time.sleep(1.5)
        assert rig.status.startswith("PLANNED"), f"it flew without Start: {rig.status}"
        r = call(rig, rig.send_cli, "send")
        assert r.success, r.message
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 10, "origin EXECUTING")
        wait(lambda: rig.status == "HOLD", 120, "origin reached")
        m = rig.refs[-1][1]
        xcd = np.array([m.x_cd.x, m.x_cd.y, m.x_cd.z])
        q_end = np.array(m.q_d)
        print(f"[11] arrived: CoM {xcd.round(3).tolist()}, "
              f"q {np.degrees(q_end).round(1).tolist()} deg")
        assert np.hypot(xcd[0], xcd[1]) < 0.15, xcd
        assert np.abs(np.degrees(q_end - HOME)).max() < 1.0, q_end
        print("=== EE TRAJECTORY PAUSE / RESUME / BACK TO ORIGIN — PASS ===")
    finally:
        os.killpg(os.getpgid(gov.pid), signal.SIGINT)
        try:
            out, _ = gov.communicate(timeout=10)
            print("---- planner log tail ----")
            print("\n".join(out.splitlines()[-6:]))
        except subprocess.TimeoutExpired:
            gov.kill()
        rig.destroy_node()
        rclpy.try_shutdown()
        spin.join(timeout=5.0)


if __name__ == "__main__":
    main()
