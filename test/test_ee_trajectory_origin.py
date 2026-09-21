#!/usr/bin/env python3
"""Loopback test of the EE trajectory mode's BACK TO ORIGIN service, and of the
two side channels the arm ground station's EE Trajectory tab draws from,
against the built whole_body_trajectory_planner node (no sim, no PX4).

Sibling of test_ee_trajectory_loopback.py, which covers select -> go_to_start
-> start -> HOLD. This one covers the rest of that tab:

  the DRONE path is published beside the EE path, same 9-double stride, and
  info [15] says where its first lap ends (the tab draws one airframe loop) ->
  current_base carries the MEASURED airframe the tab's Drone triad draws ->
  start_error carries its three tolerances so the tab can show the bound beside
  the value -> a run is started and allowed to finish -> BACK TO ORIGIN plans
  [0, 0, z] with the arm home and STOPS at PLANNED (it does not fly on the
  button press) -> the generic send executes it.

There is deliberately no pause/resume here: the EE run is not interruptible, and
an abort is a revert to SAFETY (the tab's Pause button and the planner's
pause/resume services were removed 2026-09-18 -- the watchdog owns that path).
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

NS = "/uav_eeorigin"
GOV_CMD = ["ros2", "launch", "fsc_trajectory_planner",
           "whole_body_trajectory_planner_launch.py", f"uav_prefix:={NS.lstrip('/')}"]
# optional: plan against a flight yaml (e.g. the 4-D sim yaml, whose arm fold
# sweeps over both laps) instead of the node's defaults
if os.environ.get("EE_ORIGIN_PARAMS_FILE"):
    GOV_CMD.append(f"params_file:={os.environ['EE_ORIGIN_PARAMS_FILE']}")
HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Rig(Node):
    def __init__(self):
        super().__init__("eeorigin_rig", namespace=NS)
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
        self.base = None
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
            PoseStamped, "whole_body_planner/current_base",
            lambda m: setattr(self, "base", m), 10)
        self.create_subscription(
            WholeBodyReference, "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            lambda m: self.refs.append((time.monotonic(), m)), 200)
        self.go_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/go_to_start")
        self.start_cli = self.create_client(Trigger, "whole_body_planner/ee_trajectory/start")
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

        rig.select.publish(String(data="circle"))
        wait(lambda: rig.ee_status and rig.ee_status.startswith("READY"), 90, "READY")
        wait(lambda: rig.info is not None, 10, "info")
        print(f"[2] {rig.ee_status}")

        # the drone path is published beside the EE path, same 9-double stride
        wait(lambda: rig.drone_path is not None and rig.drone_path.size > 0, 10, "drone_path")
        dp = rig.drone_path.reshape(-1, 9)
        assert np.allclose(np.linalg.norm(dp[:, 5:9], axis=1), 1.0, atol=1e-6), "drone quats"
        assert dp[:, 4].min() >= 0.0 and dp[:, 4].max() < 1.0, dp[:, 4].max()
        print(f"[3] drone path {dp.shape[0]} samples, speed <= {dp[:, 4].max():.3f} m/s")

        # where the first lap ends: laps >= 2 put it in the constant-rate part,
        # where tau(t) = s Tr / 2 + s (t - Tr), i.e. t = Tr / 2 + T_lap
        assert len(rig.info) >= 16, rig.info
        t1, T_lap, Tr, laps = rig.info[15], rig.info[3], rig.info[5], int(rig.info[4])
        if laps >= 2:
            assert abs(t1 - (0.5 * Tr + T_lap)) < 1e-6, (t1, Tr, T_lap)
        else:
            assert abs(t1 - rig.info[2]) < 1e-6, (t1, rig.info[2])
        # the first lap's airframe loop closes on itself (the arm is back on
        # the same q2 at its end): measured 0.4 mm on both the node defaults
        # and the 4-D sim yaml
        # (interpolated at exact times: the 400-sample grid is ~0.13 s, i.e.
        # up to ~20 mm of airframe travel between samples)
        def at(t):
            return np.array([np.interp(t, dp[:, 0], dp[:, 1 + i]) for i in range(3)])
        gap = np.linalg.norm(at(t1) - dp[0, 1:4])
        # and how far lap 2 rides from lap 1 at the same point of the shape:
        # MID-lap, where both samples are on the constant rate (lap 2's end
        # is inside the ramp-out, a different point of the shape)
        lap2 = 0.0
        if laps >= 2:
            ta = t1 - 0.5 * T_lap
            lap2 = np.linalg.norm(at(ta + T_lap) - at(ta))
        print(f"[3b] first lap ends t = {t1:.2f} s of {rig.info[2]:.2f}; "
              f"loop gap {gap * 1e3:.1f} mm; lap 2 vs lap 1 {lap2 * 1e3:.1f} mm")
        assert gap < 0.005, gap

        # the MEASURED airframe: exactly the odometry + attitude fed in
        wait(lambda: rig.base is not None, 5, "current_base")
        b = rig.base.pose
        pb = np.array([b.position.x, b.position.y, b.position.z])
        yaw_b = 2.0 * math.atan2(b.orientation.z, b.orientation.w)
        assert rig.base.header.frame_id == "world", rig.base.header.frame_id
        assert np.allclose(pb, rig.odom_xyz, atol=1e-9), (pb, rig.odom_xyz)
        assert abs(math.remainder(yaw_b - rig.yaw, 2.0 * math.pi)) < 1e-6, (yaw_b, rig.yaw)
        assert abs(b.orientation.x) < 1e-9 and abs(b.orientation.y) < 1e-9
        print(f"[3c] current_base {pb.round(3).tolist()} yaw {math.degrees(yaw_b):.2f} deg")

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
        # the Drone triad follows the MEASURED base, not the reference: it
        # moved with the teleport
        time.sleep(0.3)
        b = rig.base.pose
        assert np.allclose([b.position.x, b.position.y, b.position.z], rig.odom_xyz,
                           atol=1e-9), (b.position, rig.odom_xyz)
        assert abs(math.remainder(2.0 * math.atan2(b.orientation.z, b.orientation.w)
                                  - rig.yaw, 2.0 * math.pi)) < 1e-6
        # the tolerances now ride along with the errors (the GS shows them)
        assert len(rig.start_err) >= 7, rig.start_err
        assert rig.start_err[4] > 0.0 and rig.start_err[5] > 0.0 and rig.start_err[6] > 0.0
        print(f"[4] at start; tolerances {np.round(rig.start_err[4:7], 3).tolist()}")

        # start and let the run finish: it is not interruptible by design
        r = call(rig, rig.start_cli, "start")
        assert r.success, r.message
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 5, "EXECUTING")
        # the status word stays a bare EXECUTING now that the PAUSED suffix is gone
        assert rig.status.split()[0] == "EXECUTING", rig.status
        print(f"[5] {r.message}")
        wait(lambda: rig.status == "HOLD", rig.info[2] + 60, "run complete")
        print("[6] run complete, HOLD")

        # BACK TO ORIGIN: plans, and stops at PLANNED
        r = call(rig, rig.origin_cli, "back_to_origin")
        assert r.success, r.message
        print(f"[7] {r.message}")
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
        print(f"[8] arrived: CoM {xcd.round(3).tolist()}, "
              f"q {np.degrees(q_end).round(1).tolist()} deg")
        assert np.hypot(xcd[0], xcd[1]) < 0.15, xcd
        assert np.abs(np.degrees(q_end - HOME)).max() < 1.0, q_end
        print("=== EE TRAJECTORY DRONE PATH / CURRENT BASE / START TOLERANCES / "
              "BACK TO ORIGIN — PASS ===")
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
