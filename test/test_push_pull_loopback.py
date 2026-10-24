#!/usr/bin/env python3
"""Loopback test of the PUSH-AND-PULL mode against the built
whole_body_trajectory_planner node (no sim, no PX4), on the generated
box-pushing sim yaml (fsc_autopilot_ros2/config/
params_single_aerial_manipulator_whole_body_l1_4d_direct_actuation_t650_sim_box_pushing.yaml).

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
  [4b] (2026-10-22) with push_pull_table_from_box: the table published in info [77..86]
       is the one the box places; with push_pull_go_to_start_bypass go_to_start comes
       near the table only at the approach height (round its box end, never across)
  [5] ready: refused 80 mm off; then the vertical descent onto the handle
  [6] push: the planner asks for CONTACT and -- per push_pull_world_anchor_push
      -- keeps the WORLD anchor (true) or returns to the CoM (false); the reference
      holds still for the settle, then slides 0.50 m along -y with the arm and
      the heading FIXED
  [6b] release (2026-10-17): refused off the handle ([4a]); after the push the CoM
      anchor is requested and CONTACT off, nothing moves
  [7] exit: CONTACT off; straight up the margin, then the arm home
  [8] go_to_land: the heading turns as typed from the push heading CLOCKWISE (-180: 90 deg;
      -360: 270); with the bypass ([8b], 2026-10-22) round the table; COMPLETE
  [9b] (2026-10-19) the force guard is armed on the handle before the Push press too
       (the contact flag still free): 6 N -> ABORTING
  [9c] (2026-10-19) the arm's joint states going silent on the handle (the q3 guard
       blind) -> ABORTING
  [9] again with the force guard: 6 N on the task-force feed while pushing ->
      ABORTING (SAFETY GUARD) -> contact off, a release hold, the climb, the
      arm home -> the box capture and the plan dropped
  [9d] (2026-10-21) the raw-mocap guard: BRIDGING on the handle aborts nothing,
       CONFIRMED aborts (release, climb, arm home), then 'HOVERING on EKF2' and every
       step / Plan waits for the mocap; the status recovers when the mocap does
  [9e] the guard's own status silent: steps refused
  [9f] (2026-10-21) the box-lag guard against the LAW's claw target: the box
       following the plan exactly, the law's free-axis shift (wb_control_debug [117])
       is the lag the guard reads -- 30 mm passes, 60 mm aborts (BEHIND, shift counted)
  [9g] (2026-10-21) the box sample gate: a 0.3 s run of the 10-21 obj_0 mis-solve (95 deg
       roll, origin 61 mm off) while pushing -- which the 0.2 s tilt guard would abort on --
       is dropped (no trip, the monitor's travel stays on the plan); a feed that keeps
       jumping ages out and aborts as JUMPING
  [9h] (2026-10-21) the unbuckle: the arm buckled on the handle (the 10-21 trip pose, q2 44.9 /
       q3 5 deg) trips the q3 guard; the abort holds the measured arm through the release
       wait, then the retreat brings q2 / q3 to the push pose's (q1 / q4 kept) before the
       climb, and the fold ends home
  [9i] (2026-10-22) the q3 RATE guard (push_pull_guard_q3_lookahead_s): a slow drift toward the
       limit and a fast dip that stops well above it abort nothing; a fast fold (150 deg/s)
       aborts while q3 is still above push_pull_guard_q3_min_deg
  [9k] (2026-10-24) the q2 wall in the rate guard (push_pull_guard_q2_max_deg): the 10-23 stretch
       -- q2 up / q3 down at 65 deg/s (faster under a low q3 limit, so the flown rule still trips)
       with the fold held, until q2 meets its wall at q3 = fold - wall (15 deg) -- aborts nothing;
       the same stretch with the key live-set to 0 (the flown rule) aborts; a fold with q2 pinned
       at the wall still aborts while q3 is above the limit
  [9j] (2026-10-22, the valve's drone-mocap freeze) a mocap-guard abort hands the claw to the
       CoM anchor AT ONCE even with push_pull_abort_release_world_held (the valve's free-handle
       release): "the world" is then EKF2 dead-reckoning; and a mocap fault confirmed DURING a
       world-held release (a force-guard abort) ends the world hold early
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
GUARD_TOPIC = "/pl_rig/mocap_guard"        # the estimator's raw-mocap guard (2026-10-21)
YAML = os.environ.get("PL_YAML", os.path.expanduser(
    "~/ros2_ws/src/fsc_autopilot_ros2/config/"
    "params_single_aerial_manipulator_whole_body_l1_4d_direct_actuation_t650_sim_box_pushing.yaml"))
GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={NS}", "--params-file", YAML,
           "-p", f"push_pull_box_topic:={BOX_TOPIC}",
           "-p", f"push_pull_force_topic:={FORCE_TOPIC}",
           "-p", f"push_pull_contact_service:={CONTACT_SRV}",
           "-p", f"pick_place_anchor_service:={ANCHOR_SRV}",
           "-p", f"push_pull_mocap_guard_topic:={GUARD_TOPIC}",
           # the rig is a PERFECT plant: no trim to measure
           "-p", "push_pull_descent_trim:=false",
           # ... and its box is a fixed mocap pose that never moves with the claw,
           # so the 2026-10-16 box guards (lag / tilt) would abort every push
           # (45 mm "behind its plan"); they are covered by the gtests and Isaac
           "-p", "push_pull_guard_box_lag:=0.0",
           "-p", "push_pull_guard_box_tilt_deg:=0.0"]
HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
_Y = open(YAML).read()


def _yaml_list(key):
    m = re.search(r"^\s*" + key + r":\s*\[([^\]]*)\]", _Y, re.M)
    return np.array([float(x) for x in m.group(1).split(",")]) if m else None


def _yaml_num(key, default):
    m = re.search(r"^\s*" + key + r":\s*([-+0-9.eE]+)", _Y, re.M)
    return float(m.group(1)) if m else default


def _yaml_bool(key, default):
    m = re.search(r"^\s*" + key + r":\s*(true|false)", _Y, re.M)
    return (m.group(1) == "true") if m else default


# 2026-10-16: read from the yaml (the 10-03 constants had gone stale on 10-07)
PUSH_POSE = np.radians(_yaml_list("push_pull_push_pose_deg"))
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
STEPS = ["go_to_start", "ready", "push", "exit", "go_to_land_start", "land"]   # six since 2026-10-16
EE_OFFSET = _yaml_list("push_pull_ee_offset")
APPROACH_DZ = _yaml_num("push_pull_approach_dz", 0.20)
APPROACH_BACK = _yaml_num("push_pull_approach_back", 0.0)
EXIT_DZ = _yaml_num("push_pull_exit_dz", 0.0) or APPROACH_DZ
TABLE = _yaml_list("push_pull_table")
# the scene's box on the table's centre line (2026-10-21: from the yaml's table, not a
# constant -- the layout moved 0.60 m back; it was x 1.275)
BOX = np.array([0.5 * (TABLE[0] + TABLE[1]), 0.25, 0.85])
WORLD_ANCHOR_PUSH = _yaml_bool("push_pull_world_anchor_push", True)
# 2026-10-22: the table placed from the box, Go To Start round it
TABLE_FROM_BOX = _yaml_bool("push_pull_table_from_box", False)
TABLE_BOX_GAP = _yaml_num("push_pull_table_box_gap", 0.10)
BYPASS = _yaml_bool("push_pull_go_to_start_bypass", False)
WORLD_ANCHOR_READY = _yaml_bool("push_pull_world_anchor_ready", True)
WORLD_ANCHOR_ON_HANDLE = _yaml_bool("push_pull_world_anchor_on_handle", False)
CONTACT_OFF_AT_PUSH_END = _yaml_bool("push_pull_contact_off_at_push_end", False)
# the push's profile (2026-10-17): > 0 = the cruise at push_distance / (T - ramp)
PUSH_RAMP = _yaml_num("push_pull_push_ramp_s", 0.0)
PUSH_DIST = _yaml_num("push_pull_push_distance", 0.50)
# 2026-10-21: a straight push's time is the yaml's mean speed's, max(|d| / v, floor)
PUSH_SPEED = _yaml_num("push_pull_push_mean_speed", 0.0)
PUSH_MIN_TIME = _yaml_num("push_pull_push_min_time_s", 0.0)
PUSH_TIME = max(abs(PUSH_DIST) / PUSH_SPEED, PUSH_MIN_TIME) if PUSH_SPEED > 0.0 else float("nan")
# push_pull/info with six steps: goals at [19 + 7k], at handle [61], contact [62], drone yaw [66]
I_BOX_VALID, I_GOAL, I_AT_HANDLE, I_CONTACT, I_DRONE_YAW = 14, 19, 61, 62, 66
OFFSET = np.array([0.10, -0.07])             # the spawn's shift from the start mark, as the scene spawns
START = _yaml_list("push_pull_start_mark")[:2]   # 2026-10-21: (-0.60, 0); (0, 0) before


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
        self.guard_pub = self.create_publisher(Float64MultiArray, GUARD_TOPIC, 10)
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
                    for n in STEPS + ["adjust", "plan", "reset", "capture_box", "abort", "release"]}
        self.params = self.create_client(SetParameters, "whole_body_trajectory_planner/set_parameters")
        self.getp = self.create_client(GetParameters, "whole_body_trajectory_planner/get_parameters")
        self.odom_xyz = np.array([*(START + OFFSET), 0.0])
        self.yaw = 0.0
        self.q = HOME.copy()
        self.box = BOX.copy()
        self.force = np.zeros(3)
        self.js_on = True          # [9c] stops the joint states
        self.mg = [0.0, 0.0, 0.01]  # [9d] the mocap guard: state, cause, age
        self.mg_on = True          # [9e] stops it
        self.free_off = 0.0        # [9f] the law's free-axis shift, wb_control_debug [117] [m]
        self.box_follow = None     # [9f] (box0, r_ed0): the box follows the streamed claw reference
        self.glitch_n = 0          # [9g] the next n box samples are the 10-21 mis-solve
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
        if self.js_on:
            j = JointState()
            j.header.stamp = self.get_clock().now().to_msg()
            j.name = ["joint1", "joint2", "joint3", "joint4"]
            j.position = list(map(float, self.q))
            j.velocity = [0.0] * 4
            self.js.publish(j)
        m = Mocap()
        box = self.box
        if self.box_follow is not None and self.refs:
            r = self.refs[-1][1].r_ed
            box = self.box_follow[0] + (np.array([r.x, r.y, r.z]) - self.box_follow[1])
        noisy = box + self.rng.normal(0.0, 0.0005, 3)
        q = (1.0, 0.0, 0.0, 0.0)
        if self.glitch_n > 0:      # [9g] a 95 deg roll, the origin 61 mm off
            self.glitch_n -= 1
            noisy = noisy + np.array([0.0, 0.040, -0.046])
            q = (math.cos(math.radians(47.5)), math.sin(math.radians(47.5)), 0.0, 0.0)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, noisy)
        m.pose.orientation.w, m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z = q
        self.box_pub.publish(m)
        if self.mg_on:
            g = Float64MultiArray()
            g.data = list(map(float, self.mg)) + [0.0] * 13
            self.guard_pub.publish(g)

    def feed_force(self):
        data = [0.0] * 118
        data[97:100] = list(map(float, self.force))
        data[117] = float(self.free_off)
        d = Float32MultiArray()
        d.data = data
        self.force_pub.publish(d)

    def goal(self, step):
        g = self.info[I_GOAL + 7 * step: I_GOAL + 7 + 7 * step]
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


def set_param(rig, name, value):
    assert rig.params.wait_for_service(timeout_sec=10)
    p = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE,
                                                  double_value=float(value)))
    fut = rig.params.call_async(SetParameters.Request(parameters=[p]))
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    res = fut.result().results[0]
    assert res.successful, (name, res.reason)


def set_param_bool(rig, name, value):
    assert rig.params.wait_for_service(timeout_sec=10)
    p = Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_BOOL,
                                                  bool_value=bool(value)))
    fut = rig.params.call_async(SetParameters.Request(parameters=[p]))
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    res = fut.result().results[0]
    assert res.successful, (name, res.reason)


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
    th = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    th.start()
    log = tempfile.NamedTemporaryFile("w+", prefix="push_pull_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT, text=True,
                           start_new_session=True)
    try:
        wait(lambda: rig.pl_status is not None and rig.info is not None, 30, "planner up")
        assert rig.pl_status == "NOT IN DIRECT", rig.pl_status

        # [1] Adjust on the start mark (SAFETY): land and start mark move by the offset
        land0 = get_param(rig, "push_pull_land")
        ls0 = get_param(rig, "push_pull_land_start")
        time.sleep(0.5)
        r = call(rig, "adjust")
        assert r.success, r.message
        land = get_param(rig, "push_pull_land")
        assert np.allclose(land[:2] - land0[:2], OFFSET, atol=1e-3), (land, land0)
        assert land[2] == land0[2] and land[3] == land0[3], land
        if ls0.size == 4:   # Land Start moves with it (2026-10-16)
            ls = get_param(rig, "push_pull_land_start")
            assert np.allclose(ls[:2] - ls0[:2], OFFSET, atol=1e-3), (ls, ls0)
        r = call(rig, "go_to_start")
        assert not r.success and "DIRECT" in r.message, r.message
        print(f"[1] adjust: {r.message if not r.success else ''} land {land.round(3).tolist()}")

        # [2] DIRECT at the hover; Plan before Get refused; Get; Plan
        rig.odom_xyz = np.array([*(START + OFFSET), 1.0])
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
        assert abs(rig.info[I_DRONE_YAW] + 90.0) < 0.5, rig.info[I_DRONE_YAW]
        _, _, claw0 = rig.goal(0)
        # the margins: APPROACH_DZ above and APPROACH_BACK behind (the nose is -y: behind = +y)
        assert np.linalg.norm(claw0 - (grasp + [0, APPROACH_BACK, APPROACH_DZ])) < 2e-3, claw0
        _, _, claw2 = rig.goal(2)
        assert np.linalg.norm(claw2 - (grasp + [0, -0.50, 0])) < 2e-3, claw2
        print(f"[2] planned: {rig.pl_status}; step T {rig.info[8:14].round(1).tolist()} s; "
              f"grasp {claw.round(3).tolist()}, drone at the box {base.round(3).tolist()} yaw {math.degrees(yaw):.1f}")

        # [3] out of order
        for name, word in (("ready", "Go To Start first"), ("push", "Ready first"),
                           ("exit", "not on the handle"), ("land", "Go To Land Start first")):
            r = call(rig, name)
            assert not r.success and word in r.message, (name, r.message)
        print("[3] out-of-order steps refused")

        # [4] go_to_start: the world anchor requested on arrival -- unless the
        # yaml holds the claw in the world only from the handle (2026-10-07)
        T, refs, _ = fly(rig, 0)
        if WORLD_ANCHOR_READY:
            assert rig.anchor_calls and rig.anchor_calls[-1] is False, rig.anchor_calls
        else:
            assert not rig.anchor_calls or rig.anchor_calls[-1] is True, rig.anchor_calls
        assert abs(heading_deg(refs[-1]) - (-180.0)) < 0.5 or abs(heading_deg(refs[-1]) - 180.0) < 0.5
        wait(lambda: rig.pl_status.startswith("DONE go_to_start"), 5, "DONE go_to_start")
        print(f"[4] go_to_start T={T:.1f}s, anchor calls {rig.anchor_calls}; {rig.pl_status}")
        # [4b] (2026-10-22) the table placed from the box (info [77..86]) and Go To Start round
        # it: seen from above, the vehicle comes within 0.65 m of the table (0.70 less the arm's
        # CoM offset) only once it is at the approach height -- never low, never across the table
        if TABLE_FROM_BOX:
            tb = rig.info[77:87]
            assert len(rig.info) >= 87 and tb[0] == 1.0 and tb[1] == 1.0, rig.info[77:87]
            yaw_t = math.radians(tb[4])
            ex = rig.box - np.array([math.cos(yaw_t), math.sin(yaw_t), 0.0]) * (-tb[5])
            assert abs(tb[5] + 0.12 + TABLE_BOX_GAP) < 1e-6 and abs(tb[9] - (BOX[2] - 0.05)) < 3e-3, tb   # the capture is noisy

            def table_dist(p):
                dx, dy = p[0] - tb[2], p[1] - tb[3]
                lx = math.cos(yaw_t) * dx + math.sin(yaw_t) * dy
                ly = -math.sin(yaw_t) * dx + math.cos(yaw_t) * dy
                return math.hypot(max(tb[5] - lx, 0.0, lx - tb[6]), max(tb[7] - ly, 0.0, ly - tb[8]))
            z_end = refs[-1].x_cd.z
            near = [(table_dist((m.x_cd.x, m.x_cd.y)), m.x_cd.z) for m in refs]
            low_near = [d for d, z in near if d < 0.65 and z < z_end - 0.005]
            assert not low_near, f"go_to_start came {min(low_near):.2f} m from the table below the approach height"
            if BYPASS:
                start_d = table_dist((refs[0].x_cd.x, refs[0].x_cd.y))
                assert start_d > 0.70, start_d
                print(f"[4b] the table placed from the box (box end at {ex[0]:.3f}, {ex[1]:.3f}); go_to_start "
                      f"from {start_d:.2f} m round it: within 0.65 m only at the approach height "
                      f"(closest {min(d for d, _ in near):.2f} m at the end)")

        # [4a] release (2026-10-17) is refused off the handle: nothing to release
        r = call(rig, "release")
        assert not r.success and "not on the handle" in r.message, r.message
        print(f"[4a] release above the handle refused: {r.message}")

        # [5] ready: 80 mm off refused, then on
        rig.teleport(0, off=(0.08, 0.0, 0.0))
        time.sleep(0.4)
        r = call(rig, "ready")
        assert not r.success and "wait for it to settle" in r.message, r.message
        rig.teleport(0)
        time.sleep(0.4)
        T, refs, _ = fly(rig, 1)
        # straight down onto the handle, or diagonally along the nose (-y) with
        # the side margin: never ACROSS it (x)
        side = max(abs(m.r_ed.x - grasp[0]) for m in refs)
        if APPROACH_BACK == 0.0:
            side = max(side, max(abs(m.r_ed.y - grasp[1]) for m in refs))
        assert side < 3e-3, side
        assert rig.info[I_AT_HANDLE] == 1.0, "not at the handle"
        if WORLD_ANCHOR_ON_HANDLE:
            assert rig.anchor_calls and rig.anchor_calls[-1] is False, rig.anchor_calls
        wait(lambda: rig.pl_status.startswith("DONE ready next=push"), 5, "DONE ready")
        print(f"[5] ready T={T:.1f}s, claw sideways {side * 1e3:.2f} mm on the way down; {rig.pl_status}")

        # [6] push: contact on, the anchor the yaml asks for (world-held: a
        # repeated WORLD request, or none), settle then slide along -y, arm and
        # heading fixed
        n_c, n_a = len(rig.contact_calls), len(rig.anchor_calls)
        T, refs, msg = fly(rig, 2)
        assert rig.contact_calls[n_c:] == ([True, False] if CONTACT_OFF_AT_PUSH_END else [True]), \
            rig.contact_calls
        want_com = not WORLD_ANCHOR_PUSH
        assert all(c is want_com for c in rig.anchor_calls[n_a:]) and (
            want_com is False or rig.anchor_calls[n_a:] == [True]), rig.anchor_calls
        q_dev = max(np.max(np.abs(np.array(m.q_d) - PUSH_POSE)) for m in refs)
        h0 = heading_deg(refs[0])
        h_dev = max(abs((heading_deg(m) - h0 + 180) % 360 - 180) for m in refs)
        y = [m.r_ed.y for m in refs]
        assert q_dev < 1e-6 and h_dev < 1e-4, (q_dev, h_dev)
        assert abs((y[-1] - y[0]) + PUSH_DIST) < 2e-3, (y[0], y[-1])
        settle = [m for t_, m in zip(range(len(refs)), refs)][:100]   # the first 1.0 s at 100 Hz
        assert max(abs(m.r_ed.y - y[0]) for m in settle) < 1e-6, "moved during the settle"
        assert rig.info[I_CONTACT] == (0.0 if CONTACT_OFF_AT_PUSH_END else 1.0)
        # the profile as streamed: with a ramp, the middle of the move at the
        # constant cruise speed (the claw's, which the acceleration tilt does
        # not touch while the speed is constant), never faster than it
        sp = np.array([math.hypot(m.r_ed_dot.x, m.r_ed_dot.y) for m in refs])
        prof = ""
        if PUSH_RAMP > 0.0:
            V = abs(PUSH_DIST) / (PUSH_TIME - PUSH_RAMP)
            k0 = int(round((T - PUSH_TIME + PUSH_RAMP + 0.5) * 100))
            k1 = int(round((T - PUSH_RAMP - 0.5) * 100))
            mid = sp[k0:k1]
            assert len(mid) > 100 and np.max(np.abs(mid - V)) < 1e-3 * V, (V, mid.min(), mid.max())
            # the claw (not the CoM) may run a little over V in the ramps: the
            # acceleration tilt swings it (0.13 % measured)
            assert sp.max() < V * (1 + 1e-2), (V, sp.max())
            prof = f", cruise {mid.mean() * 1e3:.2f} mm/s (want {V * 1e3:.2f}) for {len(mid) / 100:.1f} s"
        # THE PUSH SPEED (2026-10-21): the step = settle + max(|d| / v, floor), info [72]
        # carries that time, and the push monitor (arrival [21..29]) holds its summary --
        # the claw reference's full travel, the rig's fixed box none
        settle_s = _yaml_num("push_pull_push_settle_s", 1.5)
        assert abs(rig.info[72] - PUSH_TIME) < 1e-9, (rig.info[69:73], PUSH_TIME)
        # the tested ranges (2026-10-21, display only): info [73..76] = the yaml's, NaN if unset
        for i, key in ((73, "push_pull_push_distance_ok"), (75, "push_pull_push_speed_ok")):
            want = _yaml_list(key)
            got = rig.info[i:i + 2]
            if want is None or len(want) != 2:
                assert np.all(np.isnan(got)), (key, got)
            else:
                assert np.allclose(got, want), (key, got, want)
        assert abs(T - (settle_s + PUSH_TIME)) < 0.06, (T, settle_s, PUSH_TIME)   # T printed to 0.1 s
        wait(lambda: rig.arrival is not None and len(rig.arrival) >= 30 and rig.arrival[22] == 1.0,
             3, "the push monitor's summary")
        a = rig.arrival
        assert a[21] == 1.0 and abs(a[24] - T) < 0.06 and abs(a[25] - settle_s) < 1e-9, a[21:30]
        assert abs(a[27] - abs(PUSH_DIST)) < 2e-3 and abs(a[26]) < 5e-3 and a[29] == 0.0, a[21:30]
        print(f"[6] push T={T:.1f}s: {msg}; arm dev {q_dev:.1e} rad, heading dev {h_dev:.1e} deg, "
              f"claw y {y[0]:.3f} -> {y[-1]:.3f}{prof}; contact calls {rig.contact_calls}; "
              f"monitor t {a[23]:.1f}/{a[24]:.1f} s, travel box {a[26] * 1e3:.1f} / plan {a[27] * 1e3:.1f} mm")

        # [6b] release (2026-10-17, the release-window fix): as the station opens
        # the gripper the claw's reference goes back to the CoM anchor -- one CoM
        # request when it was world-held -- and the contact flag OFF; nothing moves,
        # the planner stays on the handle, and a second call is harmless
        n_c, n_a = len(rig.contact_calls), len(rig.anchor_calls)
        r = call(rig, "release")
        assert r.success, r.message
        time.sleep(0.3)
        assert rig.contact_calls[n_c:] and all(c is False for c in rig.contact_calls[n_c:]), rig.contact_calls
        assert rig.anchor_calls[n_a:] == ([True] if WORLD_ANCHOR_PUSH else []), rig.anchor_calls
        assert rig.info[I_AT_HANDLE] == 1.0 and rig.status == "HOLD", (rig.info[I_AT_HANDLE], rig.status)
        r2 = call(rig, "release")
        assert r2.success and "already follows" in r2.message, r2.message
        print(f"[6b] release: {r.message}; anchor calls {rig.anchor_calls[n_a:]}")

        # [7] exit: contact off, climb with the arm fixed, then home
        n_c = len(rig.contact_calls)
        T, refs, _ = fly(rig, 3)
        # contact off (again, when the push already ended it): only OFF requests
        new = rig.contact_calls[n_c:]
        assert all(c is False for c in new) and (CONTACT_OFF_AT_PUSH_END or new), rig.contact_calls
        z = [m.r_ed.z for m in refs]
        k_up = next(i for i, m in enumerate(refs) if np.max(np.abs(np.array(m.q_d) - PUSH_POSE)) > 1e-6)
        assert z[k_up] - z[0] > EXIT_DZ - 2e-3, (z[0], z[k_up])
        assert np.allclose(refs[-1].q_d, HOME, atol=1e-6)
        assert rig.info[I_AT_HANDLE] == 0.0
        print(f"[7] exit T={T:.1f}s: claw up {z[k_up] - z[0]:.3f} m before the arm moved, then home")

        # [8] go_to_land_start (2026-10-16): ONE transition to the typed Land
        # Start, turning 270 deg clockwise, the airframe above the table top
        # wherever it is over the table; land refused before it, then straight down
        r = call(rig, "land")
        assert not r.success and "Go To Land Start first" in r.message, r.message
        T, refs, _ = fly(rig, 4)
        h = [heading_deg(m) for m in refs]
        steps_h = [((b - a + 180) % 360 - 180) for a, b in zip(h, h[1:])]
        turn = sum(steps_h)
        # the typed Land Start yaw from the push heading -90 (2026-10-22: -180 -> 90 deg clockwise;
        # it was -360 -> 270): the box's yaw is 0 here, so relative and absolute agree
        want = _yaml_list("push_pull_land_start")[3] + 90.0
        assert abs(turn - want) < 1.0 and max(steps_h) < 1e-3, (turn, want, max(steps_h))
        if TABLE_FROM_BOX and BYPASS:
            # round the table: near it (0.65 m) only at the exit height, backing out over its end
            tb = rig.info[77:87]
            yaw_t = math.radians(tb[4])

            def tdist(p):
                dx, dy = p[0] - tb[2], p[1] - tb[3]
                lx = math.cos(yaw_t) * dx + math.sin(yaw_t) * dy
                ly = -math.sin(yaw_t) * dx + math.cos(yaw_t) * dy
                return math.hypot(max(tb[5] - lx, 0.0, lx - tb[6]), max(tb[7] - ly, 0.0, ly - tb[8]))
            z0 = refs[0].x_cd.z
            low = [tdist((m.x_cd.x, m.x_cd.y)) for m in refs if m.x_cd.z < z0 - 0.005]
            assert not low or min(low) > 0.65, min(low)
            print(f"[8b] go_to_land_start round the table: below the exit height only >= "
                  f"{min(low) if low else float('nan'):.2f} m from it")
        over = [m.x_cd.z for m in refs
                if TABLE[0] - 0.25 < m.x_cd.x < TABLE[1] + 0.25 and TABLE[2] - 0.25 < m.x_cd.y < TABLE[3] + 0.25]
        z_min = min(over) if over else float("nan")
        assert not over or z_min > TABLE[4] + 0.30, (z_min, TABLE[4])
        wait(lambda: rig.pl_status.startswith("DONE go_to_land_start next=land"), 5, "DONE go_to_land_start")
        print(f"[8] go_to_land_start T={T:.1f}s: turned {turn:.1f} deg (clockwise only), "
              f"CoM >= {z_min:.2f} m near the table (top {TABLE[4]:.2f}); {rig.pl_status}")
        T, refs, _ = fly(rig, 5)
        xy_dev = max(np.hypot(m.x_cd.x - refs[0].x_cd.x, m.x_cd.y - refs[0].x_cd.y) for m in refs)
        assert xy_dev < 1e-3, xy_dev
        wait(lambda: rig.pl_status.startswith("COMPLETE"), 5, "COMPLETE")
        print(f"[8] land T={T:.1f}s: straight down ({refs[0].x_cd.z:.2f} -> {refs[-1].x_cd.z:.2f} m, "
              f"{xy_dev * 1e3:.2f} mm sideways); {rig.pl_status}")

        # [9] the force guard mid-push
        r = call(rig, "reset")
        assert r.success, r.message
        # a second mission starts like the first, at the takeoff hover: the planner
        # plans from its HOLD, which after Land sits at the Land hover (0.30 m since
        # 2026-10-21), and from there Go To Start would climb through the table's
        # height on its way over it -- refused, rightly. Leave DIRECT, climb, re-enter
        # (the hold is captured from the odometry at DIRECT entry).
        rig.mode.publish(String(data="SAFETY"))
        wait(lambda: rig.pl_status == "NOT IN DIRECT", 10, "SAFETY")
        rig.odom_xyz = np.array([rig.odom_xyz[0], rig.odom_xyz[1], 1.3])
        time.sleep(0.3)
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 10, "DIRECT hold at the takeoff hover")
        # back at its start: from where the push left it a second push would end
        # 0.25 m past the table's end, which the 2026-10-16 box-on-table check refuses
        rig.box = BOX.copy()
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
        assert rig.info[4] == 0.0 and rig.info[I_BOX_VALID] == 0.0, "plan / capture kept after a touched abort"
        print(f"[9] abort held {hold_s:.2f} s for the release, climbed, arm home; {rig.pl_status}")

        def land_abort():
            # the perfect plant on the abort's end hold: from the MEASURED pose (where the
            # rig sat), 0.03 m back against the push (the nose is -y: back is +y), 0.30 m up,
            # the arm home (push_pull_abort_from_measured / _abort_back / _abort_climb)
            rig.odom_xyz = rig.odom_xyz + np.array([0.0, 0.03, 0.30])
            rig.q = HOME.copy()
            rig.yaw = -0.5 * math.pi
            time.sleep(0.3)

        def back_on_handle(tag):
            r = call(rig, "reset")
            assert r.success, r.message
            rig.box = BOX.copy()
            time.sleep(0.6)
            assert call(rig, "capture_box").success
            assert call(rig, "plan").success
            wait(lambda: rig.pl_status.startswith("READY"), 20, f"READY ({tag})")
            fly(rig, 0)
            time.sleep(0.4)
            fly(rig, 1)
            wait(lambda: rig.pl_status.startswith("DONE ready"), 5, f"on the handle ({tag})")

        def abort_done(tag):
            wait(lambda: rig.status.startswith("EXECUTING"), 10, f"abort executing ({tag})")
            T = float(rig.status.split("T=")[1].rstrip("s"))
            wait(lambda: rig.status == "HOLD", T + 20, f"abort complete ({tag})")
            wait(lambda: rig.pl_status.startswith("ABORTED"), 5, f"ABORTED ({tag})")
            land_abort()

        # [9b] the force guard on the handle BEFORE the Push press (contact still free)
        land_abort()                       # [9]'s abort
        back_on_handle("9b")
        assert not rig.contact_calls or rig.contact_calls[-1] is False, rig.contact_calls
        time.sleep(0.5)
        assert rig.pl_status.startswith("DONE ready"), rig.pl_status    # no trip at 0 N
        rig.force = np.array([0.0, 6.0, 0.0])
        wait(lambda: rig.pl_status.startswith("ABORTING"), 5, "force guard before the push")
        assert "push force" in rig.pl_status, rig.pl_status
        rig.force = np.zeros(3)
        print(f"[9b] {rig.pl_status}")
        abort_done("9b")

        # [9c] the arm's joint states going silent on the handle
        back_on_handle("9c")
        rig.js_on = False
        wait(lambda: rig.pl_status.startswith("ABORTING"), 5, "joint-feed guard")
        assert "joint states silent" in rig.pl_status, rig.pl_status
        rig.js_on = True
        print(f"[9c] {rig.pl_status}")
        abort_done("9c")

        # [9d] the raw-mocap guard (2026-10-21) on the handle: BRIDGING (a short
        # pause) aborts nothing; CONFIRMED (frozen > 0.15 s) aborts -- release,
        # climb, arm home -- and then every step waits for the mocap while the
        # vehicle hovers on EKF2 for the pilot
        back_on_handle("9d")
        rig.mg = [1.0, 2.0, 0.08]
        time.sleep(0.6)
        assert rig.pl_status.startswith("DONE ready"), rig.pl_status
        rig.mg = [2.0, 2.0, 0.20]
        wait(lambda: rig.pl_status.startswith("ABORTING"), 5, "mocap guard")
        assert "mocap CONFIRMED (frozen" in rig.pl_status, rig.pl_status
        print(f"[9d] {rig.pl_status}")
        wait(lambda: rig.status.startswith("EXECUTING"), 10, "abort executing (9d)")
        T = float(rig.status.split("T=")[1].rstrip("s"))
        wait(lambda: rig.status == "HOLD", T + 20, "abort complete (9d)")
        rig.mg = [3.0, 2.0, 3.5]
        wait(lambda: rig.pl_status.startswith("ABORTED") and "HOVERING on EKF2" in rig.pl_status, 5,
             "ABORTED, hovering on EKF2")
        print(f"[9d] {rig.pl_status}")
        for name in ("go_to_land_start", "plan"):
            r = call(rig, name)
            assert not r.success and "waits for the mocap" in r.message, (name, r.message)
        print(f"[9d] go_to_land_start / plan refused: {r.message}")
        rig.mg = [0.0, 0.0, 0.01]
        wait(lambda: rig.pl_status.startswith("ABORTED") and "HOVERING" not in rig.pl_status, 5,
             "mocap back")
        land_abort()
        print(f"[9d] mocap back: {rig.pl_status}")

        # [9e] the guard's status itself going silent: steps refused
        rig.mg_on = False
        time.sleep(0.7)
        r = call(rig, "go_to_land_start")
        assert not r.success and "no mocap-guard status" in r.message, r.message
        rig.mg_on = True
        time.sleep(0.3)
        print(f"[9e] guard silent: {r.message}")

        # [9f] the box-lag guard measured against the LAW's claw target (2026-10-21): with
        # wb_ee_anchor_free_axis_weight > 0 the law moves the target along the nose by its share
        # of the vehicle's drift (wb_control_debug [117]) and the clamped box follows it ON
        # PURPOSE. Here the rig's box follows the streamed plan exactly, so the bare lag is ~0
        # and the guard reads the shift alone: 30 mm (inside the 40 mm limit) passes, a target
        # 60 mm AHEAD of the box = the box 60 mm BEHIND -> ABORTING, the shift named
        set_param(rig, "push_pull_guard_box_lag", 0.040)
        back_on_handle("9f")
        rig.refs.clear()
        r = call(rig, "push")
        assert r.success, r.message
        wait(lambda: rig.status.startswith("EXECUTING") and rig.refs, 10, "push executing (9f)")
        r0 = rig.refs[-1][1].r_ed
        rig.box_follow = (rig.box.copy(), np.array([r0.x, r0.y, r0.z]))
        time.sleep(2.5)                       # the settle and the slide's start, shift 0
        assert rig.pl_status.startswith("FLYING push"), rig.pl_status
        lag_bare = rig.arrival[11]
        rig.free_off = 0.030
        time.sleep(1.5)
        assert rig.pl_status.startswith("FLYING push"), rig.pl_status    # no trip at 30 mm
        lag_30 = rig.arrival[11]
        assert abs(lag_bare) < 0.005 and abs(lag_30 - 0.030) < 0.005, (lag_bare, lag_30)
        rig.free_off = 0.060
        wait(lambda: rig.pl_status.startswith("ABORTING"), 3, "box-lag guard on the law's target")
        assert "BEHIND its plan" in rig.pl_status and "law free-axis shift" in rig.pl_status, rig.pl_status
        rig.free_off = 0.0
        rig.box_follow = None
        print(f"[9f] bare lag {lag_bare * 1e3:+.1f} mm, with a 30 mm shift {lag_30 * 1e3:+.1f} mm (no trip); "
              f"{rig.pl_status}")
        abort_done("9f")
        set_param(rig, "push_pull_guard_box_lag", 0.0)

        # [9g] the box sample gate (2026-10-21): the yaml's push_pull_box_gate_jump_m / _deg.
        # A 0.3 s run of the 10-21 mis-solve while pushing (15 samples at the rig's 50 Hz):
        # ungated it would trip the 2 deg / 0.2 s box-tilt guard; gated it is dropped -- no
        # trip, and the monitor's box travel stays on the plan's. A feed that keeps jumping
        # is dropped until it ages out: the 0.5 s stale guard aborts, naming it JUMPING.
        if _yaml_num("push_pull_box_gate_jump_m", 0.0) > 0.0:
            set_param(rig, "push_pull_guard_box_lag", 0.040)
            set_param(rig, "push_pull_guard_box_tilt_deg", 2.0)
            back_on_handle("9g")
            rig.refs.clear()
            r = call(rig, "push")
            assert r.success, r.message
            wait(lambda: rig.status.startswith("EXECUTING") and rig.refs, 10, "push executing (9g)")
            r0 = rig.refs[-1][1].r_ed
            rig.box_follow = (rig.box.copy(), np.array([r0.x, r0.y, r0.z]))
            time.sleep(2.5)                     # into the slide
            gap = []
            for _ in range(2):
                rig.glitch_n = 15
                t_g = time.monotonic()
                while time.monotonic() - t_g < 0.4:
                    a = rig.arrival
                    gap.append(abs(a[26] - a[27]))
                    time.sleep(0.05)
                assert rig.pl_status.startswith("FLYING push"), rig.pl_status    # no trip
                time.sleep(0.6)
            assert max(gap) < 0.005, max(gap)
            rig.glitch_n = 10 ** 6              # keeps jumping
            wait(lambda: rig.pl_status.startswith("ABORTING"), 3, "a jumping box feed")
            assert "JUMPING" in rig.pl_status, rig.pl_status
            rig.glitch_n = 0
            rig.box_follow = None
            print(f"[9g] two 0.3 s mis-solve runs while pushing: no trip, monitor travel within "
                  f"{max(gap) * 1e3:.1f} mm of the plan; then {rig.pl_status}")
            abort_done("9g")
            set_param(rig, "push_pull_guard_box_lag", 0.0)
            set_param(rig, "push_pull_guard_box_tilt_deg", 0.0)
        else:
            print("[9g] skipped: the yaml sets no box sample gate (push_pull_box_gate_jump_m)")

        # [9h] the unbuckle (2026-10-21): the yaml's push_pull_abort_unbuckle. The arm on the
        # handle buckles to the 10-21 trip pose (q2 44.9, q3 5 deg; 2 deg under a lower limit);
        # the q3 guard (the yaml's push_pull_guard_q3_min_deg) aborts. The abort holds that MEASURED arm through the
        # release wait (the jaws open), then the retreat -- before the climb -- brings q2 / q3 to
        # the push pose's with q1 / q4 untouched, and the fold ends home.
        q3_min = _yaml_num("push_pull_guard_q3_min_deg", 0.0)
        if _yaml_bool("push_pull_abort_unbuckle", False) and q3_min > 0.0:
            push_q = np.radians(_yaml_list("push_pull_push_pose_deg"))
            wait_s = _yaml_num("push_pull_abort_release_wait_s", 0.5)
            back_on_handle("9h")
            q_trip = rig.q.copy()
            q_trip[1], q_trip[2] = math.radians(44.9), math.radians(min(5.0, q3_min - 2.0))
            rig.q = q_trip
            wait(lambda: rig.pl_status.startswith("ABORTING"), 5, "q3 guard on a buckled arm")
            assert "q3" in rig.pl_status, rig.pl_status
            print(f"[9h] {rig.pl_status}")
            wait(lambda: rig.status.startswith("EXECUTING"), 10, "abort executing (9h)")
            rig.refs.clear()
            T = float(rig.status.split("T=")[1].rstrip("s"))
            wait(lambda: rig.status == "HOLD", T + 20, "abort complete (9h)")
            refs = [m for _, m in rig.refs]
            Q = np.array([list(m.q_d) for m in refs])
            Z = np.array([m.r_ed.z for m in refs])
            # the release: the measured arm (the stream runs at 100 Hz)
            held = int(np.argmax(np.abs(Q - q_trip).max(axis=1) > 1e-6))
            assert held >= int(0.9 * wait_s * 100), (held, wait_s)
            # q1 / q4 untouched until the fold; q2 / q3 at the push pose's while the claw is
            # still within 5 cm of its height (the retreat, not the climb, carried it)
            unb = q_trip.copy()
            unb[1], unb[2] = push_q[1], push_q[2]
            at_unb = np.where(np.abs(Q - unb).max(axis=1) < 1e-6)[0]
            assert len(at_unb), "the arm never reached the push pose's q2 / q3"
            i_u = int(at_unb[0])
            assert np.allclose(Q[:i_u + 1, [0, 3]], q_trip[[0, 3]], atol=1e-9), "q1 / q4 moved before the fold"
            assert Z[i_u] - Z[0] < 0.05, ("unbuckled only in the climb", Z[i_u] - Z[0])
            assert np.min(Z[:i_u + 1]) > Z[0] - 1e-3, "the open claw dropped while unbuckling"
            assert np.allclose(Q[-1], HOME, atol=1e-6), Q[-1]
            wait(lambda: rig.pl_status.startswith("ABORTED"), 5, "ABORTED (9h)")
            print(f"[9h] held the buckled arm {held / 100.0:.2f} s (the release), q2/q3 "
                  f"{math.degrees(q_trip[1]):.1f}/{math.degrees(q_trip[2]):.1f} -> "
                  f"{math.degrees(unb[1]):.1f}/{math.degrees(unb[2]):.1f} deg by {i_u / 100.0:.2f} s "
                  f"with the claw {1e3 * (Z[i_u] - Z[0]):+.1f} mm, then the climb and home")
            land_abort()
        else:
            print("[9h] skipped: the yaml sets no push_pull_abort_unbuckle (or no q3 guard)")

        # [9i] the q3 RATE guard (2026-10-22): the yaml's push_pull_guard_q3_lookahead_s.
        # q3 moves at a set rate on the handle (the rig publishes it at 50 Hz, stamped).
        tau = _yaml_num("push_pull_guard_q3_lookahead_s", 0.0)
        if tau > 0.0 and q3_min > 0.0:
            vmin = _yaml_num("push_pull_guard_q3_rate_min_deg_s", 5.0)

            def ramp(to_deg, rate_deg_s):
                """q3 from where it is toward to_deg at rate_deg_s; stops on an abort."""
                q = rig.q.copy()
                a = math.degrees(q[2])
                t0 = time.monotonic()
                while True:
                    if rig.pl_status.startswith("ABORTING"):
                        return True
                    x = a - rate_deg_s * (time.monotonic() - t0)
                    done = x <= to_deg
                    q[2] = math.radians(max(x, to_deg))
                    rig.q = q.copy()
                    if done:
                        return False
                    time.sleep(0.004)

            back_on_handle("9i")
            q_home = rig.q.copy()
            q3_0 = math.degrees(q_home[2])
            # a slow drift to 4 deg above the limit: below the rate floor, above the limit
            assert not ramp(q3_min + 4.0, 0.6 * vmin), rig.pl_status
            time.sleep(0.5)
            assert rig.pl_status.startswith("DONE ready"), rig.pl_status
            rig.q = q_home.copy()
            time.sleep(0.5)
            # a fast dip that stops where the look-ahead never reaches the limit
            dip = q3_min + tau * 60.0 + 3.0
            if dip < q3_0 - 2.0:
                assert not ramp(dip, 60.0), rig.pl_status
                time.sleep(0.5)
                assert rig.pl_status.startswith("DONE ready"), rig.pl_status
                rig.q = q_home.copy()
                time.sleep(0.5)
            # the fold: 150 deg/s toward the elbow branch
            assert ramp(-30.0, 150.0), "the fold never aborted"
            assert "falling at" in rig.pl_status, rig.pl_status
            m = re.search(r"q3 (-?[0-9.]+) deg falling", rig.pl_status)
            q3_trip = float(m.group(1))
            assert q3_trip > q3_min, (q3_trip, rig.pl_status)
            print(f"[9i] no trip on a {0.6 * vmin:.0f} deg/s drift to {q3_min + 4.0:.0f} deg or a 60 deg/s "
                  f"dip to {dip:.0f} deg; the 150 deg/s fold aborted at q3 {q3_trip:.1f} deg: {rig.pl_status}")
            abort_done("9i")
        else:
            print("[9i] skipped: the yaml sets no push_pull_guard_q3_lookahead_s (or no q3 guard)")

        # [9k] the q2 wall in the rate guard (2026-10-24): the yaml's push_pull_guard_q2_max_deg.
        wall = _yaml_num("push_pull_guard_q2_max_deg", 0.0)
        if tau > 0.0 and q3_min > 0.0 and wall > 0.0:
            def move(to_q2, to_q3, rate_deg_s):
                """q2 and q3 together toward (to_q2, to_q3) at rate_deg_s along the larger
                excursion; stops on an abort. True = it aborted."""
                q = rig.q.copy()
                a2, a3 = math.degrees(q[1]), math.degrees(q[2])
                span = max(abs(to_q2 - a2), abs(to_q3 - a3), 1e-9)
                t0 = time.monotonic()
                while True:
                    if rig.pl_status.startswith("ABORTING"):
                        return True
                    f = min(1.0, rate_deg_s * (time.monotonic() - t0) / span)
                    q[1] = math.radians(a2 + f * (to_q2 - a2))
                    q[2] = math.radians(a3 + f * (to_q3 - a3))
                    rig.q = q.copy()
                    if f >= 1.0:
                        time.sleep(0.5)
                        return rig.pl_status.startswith("ABORTING")
                    time.sleep(0.004)

            back_on_handle("9k")
            q_home = rig.q.copy()
            q2_0, q3_0 = math.degrees(q_home[1]), math.degrees(q_home[2])
            q3_end = q3_0 - (wall - q2_0)          # the fold held: q3 = fold - wall at the wall
            assert q3_end > q3_min + 2.0, ("the push pose leaves no stretch room", q2_0, q3_0, wall)
            # the 10-23 rate (65 deg/s), or faster when the limit is low: the rule WITHOUT the
            # wall must see q3 + tau*q3_dot under the limit for ~2x its 0.05 s persistence, i.e.
            # rate * (tau - 0.1) >= q3_end - q3_min
            vs = max(65.0, 1.2 * (q3_end - q3_min) / max(tau - 0.1, 0.02))
            # (a) the 10-23 stretch into the wall: no abort
            assert not move(wall, q3_end, vs), ("the stretch into the q2 wall aborted", rig.pl_status)
            assert rig.pl_status.startswith("DONE ready"), rig.pl_status
            rig.q = q_home.copy()
            time.sleep(0.5)
            # (b) the same stretch, the flown rule (key 0): aborts
            set_param(rig, "push_pull_guard_q2_max_deg", 0.0)
            time.sleep(0.2)
            aborted = move(wall, q3_end, vs)
            set_param(rig, "push_pull_guard_q2_max_deg", wall)
            assert aborted, "the flown rule did not abort the stretch -- the test does not exercise the wall"
            assert "falling at" in rig.pl_status and "wall" not in rig.pl_status, rig.pl_status
            print(f"[9k] stretch q2/q3 {q2_0:.0f}/{q3_0:.0f} -> {wall:.0f}/{q3_end:.0f} deg at {vs:.0f} deg/s: "
                  f"no abort with the wall; the flown rule aborted: {rig.pl_status}")
            abort_done("9k-b")
            # (c) a buckle: q2 pinned at the wall, q3 folding at 150 deg/s -> aborts above the limit
            back_on_handle("9k-c")
            assert not move(wall, q3_0, 20.0), ("q2 alone up to its wall aborted", rig.pl_status)
            time.sleep(0.3)
            assert move(wall, -30.0, 150.0), "the fold with q2 pinned never aborted"
            m = re.search(r"q3 (-?[0-9.]+) deg falling", rig.pl_status)
            q3_trip = float(m.group(1))
            assert q3_trip > q3_min, (q3_trip, rig.pl_status)
            assert "stopped by q2" not in rig.pl_status, rig.pl_status
            print(f"[9k] q2 pinned at {wall:.0f} deg, q3 folding at 150 deg/s: aborted at q3 {q3_trip:.1f} deg: "
                  f"{rig.pl_status}")
            abort_done("9k-c")
        else:
            print("[9k] skipped: the yaml sets no push_pull_guard_q2_max_deg (or no rate guard)")

        # [9j] (2026-10-22) the valve's world-held release vs a mocap fault. Live-set the valve's
        # push_pull_abort_release_world_held (read at the abort) and check two things against
        # the recorded set_ee_anchor_com calls (True = the CoM anchor):
        #   (a) a mocap-guard abort hands the claw to the CoM AT ONCE (the release is not
        #       world-held: the world is EKF2 dead-reckoning);
        #   (b) a force-guard abort keeps the claw world-held through its release wait -- and a
        #       mocap fault confirmed inside that wait ends the world hold early.
        WAIT_S = _yaml_num("push_pull_abort_release_wait_s", 0.0)
        if WAIT_S >= 0.8:
            set_param_bool(rig, "push_pull_abort_release_world_held", True)
            back_on_handle("9j-a")
            n_a = len(rig.anchor_calls)
            rig.mg = [2.0, 2.0, 0.20]
            wait(lambda: rig.pl_status.startswith("ABORTING"), 5, "mocap guard (9j-a)")
            t_ab = time.monotonic()
            wait(lambda: True in rig.anchor_calls[n_a:], 0.6, "the CoM anchor at once on a mocap abort")
            dt_a = time.monotonic() - t_ab
            wait(lambda: rig.status.startswith("EXECUTING"), 10, "abort executing (9j-a)")
            T = float(rig.status.split("T=")[1].rstrip("s"))
            wait(lambda: rig.status == "HOLD", T + 20, "abort complete (9j-a)")
            rig.mg = [0.0, 0.0, 0.01]
            wait(lambda: rig.pl_status.startswith("ABORTED") and "HOVERING" not in rig.pl_status, 5,
                 "mocap back (9j-a)")
            land_abort()
            print(f"[9j] (a) mocap abort with the world-held release on: the CoM anchor {dt_a:.2f} s "
                  f"after ABORTING (the release wait is {WAIT_S:.1f} s)")
            back_on_handle("9j-b")
            n_a = len(rig.anchor_calls)
            rig.force = np.array([0.0, 6.0, 0.0])
            wait(lambda: rig.pl_status.startswith("ABORTING"), 5, "force guard (9j-b)")
            rig.force = np.zeros(3)
            t_ab = time.monotonic()
            time.sleep(0.4)
            assert True not in rig.anchor_calls[n_a:], ("world hold dropped early", rig.anchor_calls[n_a:])
            rig.mg = [2.0, 2.0, 0.20]
            wait(lambda: True in rig.anchor_calls[n_a:], 0.5, "the CoM anchor once the mocap fails")
            dt_b = time.monotonic() - t_ab
            assert dt_b < WAIT_S - 0.2, dt_b
            rig.mg = [0.0, 0.0, 0.01]
            print(f"[9j] (b) force-guard abort: world-held for 0.4 s, the mocap fault then handed the "
                  f"claw to the CoM at {dt_b:.2f} s (the scheduled hand-back is at {WAIT_S:.1f} s)")
            abort_done("9j-b")
            set_param_bool(rig, "push_pull_abort_release_world_held", False)
        else:
            print("[9j] skipped: the yaml's push_pull_abort_release_wait_s is under 0.8 s")

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
        # stop the spin thread BEFORE the node it spins goes (the other order could
        # abort the interpreter after ALL PASS: "terminate called ...")
        rclpy.shutdown()
        th.join(timeout=5)


if __name__ == "__main__":
    main()
