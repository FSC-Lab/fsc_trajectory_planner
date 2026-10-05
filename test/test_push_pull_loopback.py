#!/usr/bin/env python3
"""Loopback test of the PUSH-AND-PULL mode against the built
whole_body_trajectory_planner node (no sim, no PX4), on the generated
push-and-pull sim yaml (fsc_autopilot_ros2/config/
params_single_aerial_manipulator_whole_body_l1_4d_direct_actuation_t650_sim_push_pull.yaml).

Rig: a PERFECT plant -- after every step it teleports its odometry, attitude
and joints onto the step's planned goal -- plus a fake mocap box, a fake task-
force feed (the whole-body node's wb_control_debug) and fake set_contact /
set_ee_anchor_com services that record what the planner asks for.

  [1] Adjust (on the start mark) moves the land point; a step is refused
      outside DIRECT
  [2] DIRECT; Plan refused before Get; Get the box; Plan -> READY; the grasp
      point = the box + the EE offset, the drone's yaw = the box's - 90 deg
  [3] steps out of order are refused
  [4] go_to_start: the claw ends approach_dz above the grasp point; the WORLD
      anchor is requested on arrival
  [5] ready: refused 80 mm off; then the vertical descent onto the handle
  [6] push: the planner asks for CONTACT and -- per push_pull_world_anchor_push
      -- keeps the WORLD anchor (true) or returns to the CoM (false); the reference
      holds still for the settle, then slides 0.50 m along -y with the arm and
      the heading FIXED
  [7] exit: CONTACT off; straight up the margin, then the arm home
  [8] go_to_land: the heading turns 270 deg CLOCKWISE; COMPLETE
  [9] again with the force guard: 6 N on the task-force feed while pushing ->
      ABORTING (SAFETY GUARD) -> contact off, a release hold, the climb, the
      arm home -> the box capture and the plan dropped
  [10] SAFETY silences the stream

Run with a scratch install first on the path and a private ROS domain:
  ROS_DOMAIN_ID=78 python3 test_push_pull_loopback.py
"""
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleAttitude
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from fsc_autopilot_ros2_msgs.msg import Mocap, WholeBodyReference

NS = "/uav_pushpull"
BOX_TOPIC = "/pl_rig/obj_0/mocap"
FORCE_TOPIC = "/pl_rig/wb_control_debug"
CONTACT_SRV = "/pl_rig/set_contact"
ANCHOR_SRV = "/pl_rig/set_ee_anchor_com"
YAML = os.environ.get("PL_YAML", os.path.expanduser(
    "~/ros2_ws/src/fsc_autopilot_ros2/config/"
    "params_single_aerial_manipulator_whole_body_l1_4d_direct_actuation_t650_sim_push_pull.yaml"))
GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={NS}", "--params-file", YAML,
           "-p", f"push_pull_box_topic:={BOX_TOPIC}",
           "-p", f"push_pull_force_topic:={FORCE_TOPIC}",
           "-p", f"push_pull_contact_service:={CONTACT_SRV}",
           "-p", f"pick_place_anchor_service:={ANCHOR_SRV}",
           # the rig is a PERFECT plant: no trim to measure
           "-p", "push_pull_descent_trim:=false"]
HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
PUSH_POSE = np.radians([0.0, -20.0, 30.0, 0.0])
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
STEPS = ["go_to_start", "ready", "push", "exit", "go_to_land"]
BOX = np.array([1.20, 0.25, 0.83])
EE_OFFSET = np.array([0.0, 0.155, 0.065])    # the yaml's push_pull_ee_offset
APPROACH_DZ = 0.20
# the yaml's push_pull_world_anchor_push (2026-10-04); absent = the node default
_m = re.search(r"^\s*push_pull_world_anchor_push:\s*(true|false)", open(YAML).read(), re.M)
WORLD_ANCHOR_PUSH = bool(_m and _m.group(1) == "true")
OFFSET = np.array([0.10, -0.07])             # the start mark, as the scene spawns


class Rig(Node):
    def __init__(self):
        super().__init__("pushpull_rig", namespace=NS)
        self.mode = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode", LATCHED)
        self.odom = self.create_publisher(Odometry, "state_estimator/local_position/odom", 10)
        self.att = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.js = self.create_publisher(JointState, "fsc_open_manipulator/joint_states", 10)
        self.box_pub = self.create_publisher(Mocap, BOX_TOPIC, 10)
        self.force_pub = self.create_publisher(Float32MultiArray, FORCE_TOPIC, 10)
        self.status = self.pl_status = self.info = self.arrival = None
        self.refs = []
        self.contact_calls, self.anchor_calls = [], []
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data), LATCHED)
        self.create_subscription(String, "whole_body_planner/push_pull/status",
                                 lambda m: setattr(self, "pl_status", m.data), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/push_pull/info",
                                 lambda m: setattr(self, "info", np.asarray(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/push_pull/arrival_error",
                                 lambda m: setattr(self, "arrival", list(m.data)), 10)
        self.create_subscription(
            WholeBodyReference, "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            lambda m: self.refs.append((time.monotonic(), m)), 50)
        self.create_service(SetBool, CONTACT_SRV, self.on_contact)
        self.create_service(SetBool, ANCHOR_SRV, self.on_anchor)
        self.cli = {n: self.create_client(Trigger, f"whole_body_planner/push_pull/{n}")
                    for n in STEPS + ["adjust", "plan", "reset", "capture_box", "abort"]}
        self.params = self.create_client(SetParameters, "whole_body_trajectory_planner/set_parameters")
        self.getp = self.create_client(GetParameters, "whole_body_trajectory_planner/get_parameters")
        self.odom_xyz = np.array([OFFSET[0], OFFSET[1], 0.0])
        self.yaw = 0.0
        self.q = HOME.copy()
        self.box = BOX.copy()
        self.force = np.zeros(3)
        self.rng = np.random.default_rng(0)
        self.create_timer(0.02, self.feed)
        self.create_timer(0.004, self.feed_force)    # the node's 250 Hz debug

    def on_contact(self, req, resp):
        self.contact_calls.append(req.data)
        resp.success, resp.message = True, f"contact -> {req.data}"
        return resp

    def on_anchor(self, req, resp):
        self.anchor_calls.append(req.data)
        resp.success, resp.message = True, f"anchor com -> {req.data}"
        return resp

    def feed(self):
        o = Odometry()
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = map(float, self.odom_xyz)
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
        m = Mocap()
        noisy = self.box + self.rng.normal(0.0, 0.0005, 3)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, noisy)
        m.pose.orientation.w = 1.0
        self.box_pub.publish(m)

    def feed_force(self):
        data = [0.0] * 115
        data[97:100] = list(map(float, self.force))
        d = Float32MultiArray()
        d.data = data
        self.force_pub.publish(d)

    def goal(self, step):
        g = self.info[18 + 7 * step: 25 + 7 * step]
        return np.array(g[0:3]), math.radians(g[3]), np.array(g[4:7])

    def teleport_ref(self, ref):
        """The perfect plant on the LAST streamed reference (body origin from
        x_cd and the arm), as a step ends."""
        q = np.array(ref.q_d)
        psi = math.atan2(ref.b1_d.y, ref.b1_d.x)       # model heading
        self.q = q
        self.yaw = psi + 0.5 * math.pi
        return q, psi

    def teleport(self, step, dz=0.0, off=(0.0, 0.0, 0.0)):
        base, yaw, _ = self.goal(step)
        self.odom_xyz = base + np.array([0.0, 0.0, dz]) + np.array(off)
        self.yaw = yaw
        if self.refs:
            self.q = np.array(self.refs[-1][1].q_d)
        time.sleep(0.2)


def call(rig, name):
    cli = rig.cli[name]
    assert cli.wait_for_service(timeout_sec=10), name
    fut = cli.call_async(Trigger.Request())
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    return fut.result()


def get_param(rig, name):
    assert rig.getp.wait_for_service(timeout_sec=10)
    fut = rig.getp.call_async(GetParameters.Request(names=[name]))
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    return np.array(fut.result().values[0].double_array_value)


def wait(cond, timeout, what):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if cond():
                return
        except (TypeError, IndexError):
            pass
        time.sleep(0.05)
    raise AssertionError("timeout: " + what)


def fly(rig, k, teleport=True):
    """Press step k, wait for EXECUTING then HOLD; the plant lands on the goal."""
    rig.refs.clear()
    r = call(rig, STEPS[k])
    assert r.success, f"{STEPS[k]}: {r.message}"
    wait(lambda: rig.status.startswith("EXECUTING"), 10, f"{STEPS[k]} EXECUTING")
    assert rig.pl_status.startswith("FLYING " + STEPS[k]), rig.pl_status
    T = float(rig.status.split("T=")[1].rstrip("s"))
    wait(lambda: rig.status == "HOLD", T + 20, f"{STEPS[k]} complete")
    wait(lambda: int(rig.info[5]) == k, 5, f"{STEPS[k]} recorded")
    refs = [m for _, m in rig.refs]
    if teleport:
        rig.teleport(k)
    return T, refs, r.message


def heading_deg(m):
    return math.degrees(math.atan2(m.b1_d.y, m.b1_d.x))


def main():
    rclpy.init()
    rig = Rig()
    threading.Thread(target=rclpy.spin, args=(rig,), daemon=True).start()
    log = tempfile.NamedTemporaryFile("w+", prefix="push_pull_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT, text=True,
                           start_new_session=True)
    try:
        wait(lambda: rig.pl_status is not None and rig.info is not None, 30, "planner up")
        assert rig.pl_status == "NOT IN DIRECT", rig.pl_status

        # [1] Adjust on the start mark (SAFETY): land and start mark move by the offset
        land0 = get_param(rig, "push_pull_land")
        time.sleep(0.5)
        r = call(rig, "adjust")
        assert r.success, r.message
        land = get_param(rig, "push_pull_land")
        assert np.allclose(land[:2] - land0[:2], OFFSET, atol=1e-3), (land, land0)
        assert land[2] == land0[2] and land[3] == land0[3], land
        r = call(rig, "go_to_start")
        assert not r.success and "DIRECT" in r.message, r.message
        print(f"[1] adjust: {r.message if not r.success else ''} land {land.round(3).tolist()}")

        # [2] DIRECT at the hover; Plan before Get refused; Get; Plan
        rig.odom_xyz = np.array([OFFSET[0], OFFSET[1], 1.0])
        time.sleep(0.3)
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 10, "DIRECT hold")
        r = call(rig, "plan")
        assert not r.success and "Get the box" in r.message, r.message
        r = call(rig, "capture_box")
        assert r.success, r.message
        print(f"[2] {r.message}")
        r = call(rig, "plan")
        assert r.success, r.message
        wait(lambda: rig.pl_status.startswith("READY"), 20, "READY")
        assert rig.info[4] == 1.0
        base, yaw, claw = rig.goal(1)                     # ready: the grasp
        grasp = BOX + EE_OFFSET
        assert np.linalg.norm(claw - grasp) < 2e-3, (claw, grasp)   # 0.5 mm mocap jitter, averaged
        assert abs(math.degrees(yaw) + 90.0) < 0.5, yaw
        assert abs(rig.info[58] + 90.0) < 0.5, rig.info[58]
        _, _, claw0 = rig.goal(0)
        assert np.linalg.norm(claw0 - (grasp + [0, 0, APPROACH_DZ])) < 2e-3, claw0
        _, _, claw2 = rig.goal(2)
        assert np.linalg.norm(claw2 - (grasp + [0, -0.50, 0])) < 2e-3, claw2
        print(f"[2] planned: {rig.pl_status}; step T {rig.info[8:13].round(1).tolist()} s; "
              f"grasp {claw.round(3).tolist()}, drone at the box {base.round(3).tolist()} yaw {math.degrees(yaw):.1f}")

        # [3] out of order
        for name, word in (("ready", "Go To Start first"), ("push", "Ready first"),
                           ("exit", "not on the handle")):
            r = call(rig, name)
            assert not r.success and word in r.message, (name, r.message)
        print("[3] out-of-order steps refused")

        # [4] go_to_start: the world anchor requested on arrival
        T, refs, _ = fly(rig, 0)
        assert rig.anchor_calls and rig.anchor_calls[-1] is False, rig.anchor_calls
        assert abs(heading_deg(refs[-1]) - (-180.0)) < 0.5 or abs(heading_deg(refs[-1]) - 180.0) < 0.5
        wait(lambda: rig.pl_status.startswith("DONE go_to_start"), 5, "DONE go_to_start")
        print(f"[4] go_to_start T={T:.1f}s, anchor calls {rig.anchor_calls}; {rig.pl_status}")

        # [5] ready: 80 mm off refused, then on
        rig.teleport(0, off=(0.08, 0.0, 0.0))
        time.sleep(0.4)
        r = call(rig, "ready")
        assert not r.success and "wait for it to settle" in r.message, r.message
        rig.teleport(0)
        time.sleep(0.4)
        T, refs, _ = fly(rig, 1)
        side = max(np.hypot(m.r_ed.x - grasp[0], m.r_ed.y - grasp[1]) for m in refs)
        assert side < 3e-3, side
        assert rig.info[53] == 1.0, "not at the handle"
        wait(lambda: rig.pl_status.startswith("DONE ready next=push"), 5, "DONE ready")
        print(f"[5] ready T={T:.1f}s, claw sideways {side * 1e3:.2f} mm on the way down; {rig.pl_status}")

        # [6] push: contact on, the anchor the yaml asks for (world-held: a
        # repeated WORLD request, or none), settle then slide along -y, arm and
        # heading fixed
        n_c, n_a = len(rig.contact_calls), len(rig.anchor_calls)
        T, refs, msg = fly(rig, 2)
        assert rig.contact_calls[n_c:] == [True], rig.contact_calls
        want_com = not WORLD_ANCHOR_PUSH
        assert all(c is want_com for c in rig.anchor_calls[n_a:]) and (
            want_com is False or rig.anchor_calls[n_a:] == [True]), rig.anchor_calls
        q_dev = max(np.max(np.abs(np.array(m.q_d) - PUSH_POSE)) for m in refs)
        h0 = heading_deg(refs[0])
        h_dev = max(abs((heading_deg(m) - h0 + 180) % 360 - 180) for m in refs)
        y = [m.r_ed.y for m in refs]
        assert q_dev < 1e-6 and h_dev < 1e-4, (q_dev, h_dev)
        assert abs((y[-1] - y[0]) + 0.50) < 2e-3, (y[0], y[-1])
        settle = [m for t_, m in zip(range(len(refs)), refs)][:100]   # the first 1.0 s at 100 Hz
        assert max(abs(m.r_ed.y - y[0]) for m in settle) < 1e-6, "moved during the settle"
        assert rig.info[54] == 1.0
        print(f"[6] push T={T:.1f}s: {msg}; arm dev {q_dev:.1e} rad, heading dev {h_dev:.1e} deg, "
              f"claw y {y[0]:.3f} -> {y[-1]:.3f}; contact calls {rig.contact_calls}")

        # [7] exit: contact off, climb with the arm fixed, then home
        n_c = len(rig.contact_calls)
        T, refs, _ = fly(rig, 3)
        assert rig.contact_calls[n_c:] == [False], rig.contact_calls
        z = [m.r_ed.z for m in refs]
        k_up = next(i for i, m in enumerate(refs) if np.max(np.abs(np.array(m.q_d) - PUSH_POSE)) > 1e-6)
        assert z[k_up] - z[0] > APPROACH_DZ - 2e-3, (z[0], z[k_up])
        assert np.allclose(refs[-1].q_d, HOME, atol=1e-6)
        assert rig.info[53] == 0.0
        print(f"[7] exit T={T:.1f}s: claw up {z[k_up] - z[0]:.3f} m before the arm moved, then home")

        # [8] go_to_land: 270 deg clockwise
        T, refs, _ = fly(rig, 4)
        h = [heading_deg(m) for m in refs]
        turn = sum(((b - a + 180) % 360 - 180) for a, b in zip(h, h[1:]))
        ccw = max(((b - a + 180) % 360 - 180) for a, b in zip(h, h[1:]))
        assert abs(turn + 270.0) < 1.0 and ccw < 1e-3, (turn, ccw)
        wait(lambda: rig.pl_status.startswith("COMPLETE"), 5, "COMPLETE")
        print(f"[8] go_to_land T={T:.1f}s: turned {turn:.1f} deg (clockwise only); {rig.pl_status}")

        # [9] the force guard mid-push
        r = call(rig, "reset")
        assert r.success, r.message
        rig.box = BOX + np.array([0.0, -0.50, 0.0])        # where the push left it
        time.sleep(0.6)
        assert call(rig, "capture_box").success
        assert call(rig, "plan").success
        wait(lambda: rig.pl_status.startswith("READY"), 20, "READY again")
        fly(rig, 0)
        time.sleep(0.4)
        fly(rig, 1)
        n_c = len(rig.contact_calls)
        r = call(rig, "push")
        assert r.success, r.message
        wait(lambda: rig.status.startswith("EXECUTING"), 10, "push executing")
        time.sleep(2.5)
        rig.force = np.array([0.0, 6.0, 0.0])
        wait(lambda: rig.pl_status.startswith("ABORTING"), 5, "force guard")
        assert "SAFETY GUARD" in rig.pl_status and "push force" in rig.pl_status, rig.pl_status
        rig.force = np.zeros(3)
        print(f"[9] {rig.pl_status}")
        wait(lambda: rig.status.startswith("EXECUTING"), 10, "abort executing")
        rig.refs.clear()
        T = float(rig.status.split("T=")[1].rstrip("s"))
        wait(lambda: rig.status == "HOLD", T + 20, "abort complete")
        refs = [m for _, m in rig.refs]
        assert rig.contact_calls[n_c:] == [True, False], rig.contact_calls
        hold_s = sum(1 for m in refs[:80] if abs(m.r_ed.z - refs[0].r_ed.z) < 1e-6) / 100.0
        assert hold_s >= 0.40, hold_s
        assert np.allclose(refs[-1].q_d, HOME, atol=1e-6)
        wait(lambda: rig.pl_status.startswith("ABORTED"), 5, "ABORTED")
        assert rig.info[4] == 0.0 and rig.info[13] == 0.0, "plan / capture kept after a touched abort"
        print(f"[9] abort held {hold_s:.2f} s for the release, climbed, arm home; {rig.pl_status}")

        # [10] SAFETY
        rig.mode.publish(String(data="SAFETY"))
        wait(lambda: rig.pl_status == "NOT IN DIRECT", 5, "NOT IN DIRECT")
        n = len(rig.refs)
        time.sleep(0.5)
        assert len(rig.refs) == n, "still streaming in SAFETY"
        print("[10] SAFETY: silent")
        print("ALL PASS")
    finally:
        # the whole process GROUP: terminating the `ros2 run` wrapper alone
        # leaves the node child running, and a second planner on the same
        # namespace answers every service twice in the next run
        try:
            os.killpg(gov.pid, signal.SIGTERM)
            gov.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(gov.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f"(node log: {log.name})")
        rig.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
