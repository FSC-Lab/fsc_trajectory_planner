#!/usr/bin/env python3
"""Loopback test of the PICK-AND-PLACE mode against the built
whole_body_trajectory_planner node (no sim, no PX4).

Rig: SAFETY -> Adjust measures the vehicle on the start mark (offset) -> a
leg is refused outside DIRECT -> DIRECT hold -> Plan refused until obj_0 is
captured -> capture pick and (a measured) land point; the place point TYPED
IN (pick_place_place_point, bad edits refused) -> Plan -> READY -> an edit
makes the plan stale -> a land hover under the geofence floor is refused at
Plan -> a leg out of order is refused -> the six legs, one service each; after
every leg the rig teleports its odometry, attitude and joints onto the leg's
goal, as a perfect plant would. A claw leg (execute_pick / execute_place)
stops 0.10 m ABOVE its target and WAITS; the rig teleports onto that approach
rest, and once the claw has been inside 50 mm for pick_place_settle_s the
planner flies the vertical descent. Checked on the way: an Adjust larger than
pick_place_adjust_max is refused, the base goals carry the offset, the claw
ends ON the captured points with the arm in the pick pose, the wait does not
descend 80 mm off nor before the dwell, a wait that times out leaves the leg
INCOMPLETE, a leg is refused while the body is off its hold, the arrival
error gates "fine correction OK" at 50 mm, the retreat climbs before the
transit, execute_place sweeps q1 / q2 across their +-10 deg bands, ABORT
mid-sweep and from a hold, the abort climb clipped at the fence ceiling,
COMPLETE after execute_land, Reset, SAFETY silences the stream.

The mocap bodies are rig-private topics (passed as parameters), so a sim
publishing /obj_0/mocap cannot interfere.
"""
import math
import os
import signal
import subprocess
import tempfile
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
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from fsc_autopilot_ros2_msgs.msg import Mocap, WholeBodyReference
from ament_index_python.packages import get_package_share_directory

NS = "/uav_pickplace"
OBJ_TOPIC = "/pp_rig/obj_0/mocap"
LAND_TOPIC = "/pp_rig/land_0/mocap"
YAML = os.path.join(get_package_share_directory("fsc_trajectory_planner"), "config",
                    "whole_body_trajectory_planner_t650_aerial_manipulator.yaml")
GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={NS}", "--params-file", YAML,
           "-p", f"pick_place_pick_topic:={OBJ_TOPIC}",
           "-p", f"pick_place_land_topic:={LAND_TOPIC}"]
HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
LEGS = ["go_to_start", "execute_pick", "go_to_place_start", "execute_place",
        "go_to_land_start", "execute_land"]
CLAW_LEGS = (1, 3)
APPROACH_DZ = 0.10      # pick_place_approach_dz in the package yaml
SETTLE_S = 1.0          # pick_place_settle_s
NOMINAL_START = np.array([0.0, 0.0, 1.0])       # the package yaml's placeholders
NOMINAL_LAND = np.array([-1.0, 0.0, 0.8])
OBJ = np.array([1.0, 0.6, 0.70])
DROP = np.array([0.6, -1.6, 0.70])
LAND_MARK = np.array([-1.2, 0.3, 0.0])
# the place point is TYPED IN (pick_place_place_point) and, like every other
# room-frame point, shifted by the Adjust offset
PLACE_GOAL = DROP + np.array([0.10, -0.05, 0.0])
OFFSET = np.array([0.10, -0.05])


class Rig(Node):
    def __init__(self):
        super().__init__("pickplace_rig", namespace=NS)
        self.mode = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode", LATCHED)
        self.odom = self.create_publisher(Odometry, "state_estimator/local_position/odom", 10)
        self.att = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.js = self.create_publisher(JointState, "fsc_open_manipulator/joint_states", 10)
        self.mocap = {OBJ_TOPIC: (self.create_publisher(Mocap, OBJ_TOPIC, 10), OBJ),
                      LAND_TOPIC: (self.create_publisher(Mocap, LAND_TOPIC, 10), LAND_MARK)}
        self.status = None
        self.pp_status = None
        self.info = None
        self.path = None
        self.arrival = None
        self.base_anchor = None
        self.create_subscription(PoseStamped, "whole_body_planner/pending_base",
                                 lambda m: setattr(self, "base_anchor", m), LATCHED)
        self.heading = None
        self.ws = None
        self.pp_ws = None
        self.refs = []
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data), LATCHED)
        self.create_subscription(String, "whole_body_planner/pick_place/status",
                                 lambda m: setattr(self, "pp_status", m.data), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/pick_place/info",
                                 lambda m: setattr(self, "info", np.asarray(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/pick_place/path",
                                 lambda m: setattr(self, "path", np.asarray(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/pick_place/arrival_error",
                                 lambda m: setattr(self, "arrival", list(m.data)), 10)
        self.create_subscription(Float64, "whole_body_planner/current_ee_heading",
                                 lambda m: setattr(self, "heading", m.data), 10)
        self.create_subscription(Float64MultiArray, "whole_body_planner/workspace_rz",
                                 lambda m: setattr(self, "ws", np.asarray(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, "whole_body_planner/pick_place/workspace_rz",
                                 lambda m: setattr(self, "pp_ws", np.asarray(m.data)), LATCHED)
        self.create_subscription(
            WholeBodyReference, "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            lambda m: self.refs.append((time.monotonic(), m)), 50)
        self.cli = {}
        for name in LEGS + ["adjust", "plan", "reset", "capture_pick", "capture_place",
                            "capture_land", "capture_start", "abort"]:
            self.cli[name] = self.create_client(Trigger, f"whole_body_planner/pick_place/{name}")
        self.params = self.create_client(SetParameters, "whole_body_trajectory_planner/set_parameters")
        self.odom_xyz = np.array([OFFSET[0], OFFSET[1], 0.0])   # on the start mark, on the ground
        self.yaw = 0.0          # actual yaw the rig reports
        self.q = HOME.copy()
        self.rng = np.random.default_rng(0)
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
        for pub, p in self.mocap.values():
            m = Mocap()
            noisy = p + self.rng.normal(0.0, 0.0005, 3)   # 0.5 mm mocap jitter
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, noisy)
            m.pose.orientation.w = 1.0
            pub.publish(m)

    def goal(self, leg):
        """(base xyz, actual yaw rad, claw xyz) of a leg's planned goal."""
        g = self.info[14 + 7 * leg: 21 + 7 * leg]
        return g[0:3], math.radians(g[3]), g[4:7]

    def teleport(self, leg, dz=0.0, off=(0.0, 0.0, 0.0)):
        """A perfect plant: the measured pose becomes the leg's goal (dz above
        it: the approach rest; `off` a deliberate miss)."""
        base, yaw, _ = self.goal(leg)
        self.odom_xyz = np.array(base) + np.array([0.0, 0.0, dz]) + np.array(off)
        self.yaw = yaw
        self.q = np.array(self.refs[-1][1].q_d)
        time.sleep(0.15)      # a few 50 Hz feeds: the node measures the new pose


def ws_contains(ws, r, z):
    """Ps4Panel::reachable's cell lookup on the published workspace_rz."""
    r_min, r_max, z_min, z_max, nr, nz = ws[:6]
    nr, nz = int(nr), int(nz)
    if not (r_min <= r <= r_max and z_min <= z <= z_max):
        return False
    ri = int((r - r_min) / (r_max - r_min) * (nr - 1) + 0.5)
    zi = int((z_max - z) / (z_max - z_min) * (nz - 1) + 0.5)
    return bool(ws[6 + zi * nr + ri])


def set_param(rig, name, value):
    """What the Pick & Place tab does when a field is edited."""
    v = ParameterValue()
    if isinstance(value, str):
        v.type, v.string_value = ParameterType.PARAMETER_STRING, value
    elif isinstance(value, float):
        v.type, v.double_value = ParameterType.PARAMETER_DOUBLE, value
    else:
        v.type, v.double_array_value = ParameterType.PARAMETER_DOUBLE_ARRAY, [float(x) for x in value]
    assert rig.params.wait_for_service(timeout_sec=10)
    fut = rig.params.call_async(SetParameters.Request(parameters=[Parameter(name=name, value=v)]))
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    return fut.result().results[0]


def call(rig, name):
    cli = rig.cli[name]
    assert cli.wait_for_service(timeout_sec=10), name
    fut = cli.call_async(Trigger.Request())
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    return fut.result()


def wait(cond, timeout, what):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timeout: " + what)


def start_leg(rig, name):
    """Press a leg's button and wait until it executes; its planned T."""
    r = call(rig, name)
    assert r.success, f"{name}: {r.message}"
    wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 10, f"{name} EXECUTING")
    assert rig.pp_status.startswith("FLYING " + name), rig.pp_status
    return float(rig.status.split("T=")[1].rstrip("s"))


def wait_above(rig, leg_index, T):
    """A claw leg's approach: it ends HOLDING above its target, WAITING."""
    name = LEGS[leg_index]
    wait(lambda: rig.pp_status.startswith("WAITING " + name) and rig.status == "HOLD", T + 20,
         f"{name} waiting above its target")
    wait(lambda: rig.arrival is not None and len(rig.arrival) >= 10 and rig.arrival[9] == 2.0, 5,
         "arrival phase = wait")


def fly(rig, leg_index, above=None):
    """Fly one leg through its service and wait for the hold at its goal. A
    claw leg stops ABOVE its target and waits: `above(rig)` may check things
    there, then the rig teleports onto the approach rest and the descent
    follows once the claw has dwelt inside the tolerance."""
    name = LEGS[leg_index]
    rig.refs.clear()
    T = start_leg(rig, name)
    if leg_index in CLAW_LEGS:
        assert "(approach)" in rig.pp_status, rig.pp_status
        wait_above(rig, leg_index, T)
        if above is not None:
            above(rig)
        rig.teleport(leg_index, dz=APPROACH_DZ)
        wait(lambda: rig.status.startswith("EXECUTING") and "(descent)" in rig.pp_status, 10,
             f"{name} descent")
        Td = float(rig.status.split("T=")[1].rstrip("s"))
        T += Td
        wait(lambda: rig.status == "HOLD", Td + 20, f"{name} descent complete")
    else:
        wait(lambda: rig.status == "HOLD", T + 20, f"{name} complete")
    wait(lambda: rig.info is not None and int(rig.info[5]) == leg_index, 5, f"{name} recorded")
    refs = [m for _, m in rig.refs]
    rig.teleport(leg_index)
    return T, refs


def main():
    rclpy.init()
    rig = Rig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    # the node's log goes to a file: an undrained pipe would stall it on a long run
    log = tempfile.NamedTemporaryFile("w+", prefix="pick_place_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT,
                           text=True, start_new_session=True)
    try:
        wait(lambda: rig.pp_status is not None and rig.info is not None, 20, "planner up")
        assert rig.pp_status == "NOT IN DIRECT", rig.pp_status

        # [1] Adjust in SAFETY, the vehicle on the start mark: offset = measured - nominal (x, y);
        # 0.8 m off the mark is refused (pick_place_adjust_max 0.5 m)
        rig.odom_xyz = np.array([0.8, 0.0, 0.0])
        time.sleep(0.5)
        r = call(rig, "adjust")
        assert not r.success and "larger than" in r.message and "START mark" in r.message, r.message
        print(f"[1] 0.8 m off the mark: {r.message}")
        rig.odom_xyz = np.array([OFFSET[0], OFFSET[1], 0.0])
        time.sleep(0.5)
        r = call(rig, "adjust")
        assert r.success, r.message
        wait(lambda: abs(rig.info[0] - OFFSET[0]) < 1e-6, 5, "offset published")
        assert abs(rig.info[1] - OFFSET[1]) < 1e-6 and rig.info[2] == 0.0, rig.info[:3]
        print(f"[1] adjust: {r.message}")
        r = call(rig, "go_to_start")
        assert not r.success and "DIRECT" in r.message, r.message

        # [2] DIRECT; Plan needs both claw points
        rig.odom_xyz = np.array([OFFSET[0], OFFSET[1], 1.0])     # took off on the mark
        time.sleep(0.3)                     # the hold is captured from the airborne odometry
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        wait(lambda: rig.pp_status.startswith("NOT PLANNED"), 5, "NOT PLANNED")
        # only the pick point is measured; the place point is typed in
        assert "pick" in rig.pp_status and "place" not in rig.pp_status, rig.pp_status
        r = call(rig, "plan")
        assert not r.success and "capture" in r.message, r.message
        r = call(rig, "capture_start")
        assert not r.success and "pick_place_start_topic" in r.message, r.message
        print(f"[2] DIRECT hold; plan refused: {r.message}")

        # [3a] a second publisher on the pick topic (the Isaac emulator's
        # phantom obj_0 at the origin did this): the capture is REFUSED
        rig.mocap[OBJ_TOPIC + "#phantom"] = (rig.create_publisher(Mocap, OBJ_TOPIC, 10), np.zeros(3))
        time.sleep(0.6)
        r = call(rig, "capture_pick")
        assert not r.success and "scatter" in r.message, r.message
        rig.destroy_publisher(rig.mocap.pop(OBJ_TOPIC + "#phantom")[0])
        time.sleep(0.6)
        print(f"[3a] two publishers on the pick topic: {r.message}")
        # [3] captures: obj_0 and a measured landing mark; the place point is
        # typed in (no place topic: capture_place is refused)
        for name in ("capture_pick", "capture_land"):
            r = call(rig, name)
            assert r.success, r.message
            print(f"    {name}: {r.message}")
        r = call(rig, "capture_place")
        assert not r.success and "pick_place_place_topic" in r.message, r.message
        r = set_param(rig, "pick_place_place_point", [1.0, 2.0])
        assert not r.successful and "needs 3" in r.reason, r.reason
        r = set_param(rig, "pick_place_pick_topic", "/elsewhere")
        assert not r.successful and "fixed at launch" in r.reason, r.reason
        r = set_param(rig, "pick_place_place_point", DROP.tolist())
        assert r.successful, r.reason
        wait(lambda: rig.info[56 + 4 * 1 + 3] == 1.0, 5, "pick captured")
        assert np.linalg.norm(rig.info[60:63] - OBJ) < 1e-3, rig.info[60:63]
        wait(lambda: rig.pp_status == "NOT PLANNED: press Plan", 5, "NOT PLANNED")
        print("[3] pick / land captured; place typed in; bad edits refused")

        # [4] Plan: six legs, READY
        r = call(rig, "plan")
        assert r.success, r.message
        wait(lambda: rig.pp_status.startswith("READY"), 30, "READY")
        assert rig.pp_status == "READY next=go_to_start", rig.pp_status
        assert rig.info[4] == 1.0 and np.all(np.isfinite(rig.info[8:14])), rig.info[4:14]
        assert rig.path is not None and rig.path.size == 600 * 9, None if rig.path is None else rig.path.size
        start_base, start_yaw, _ = rig.goal(0)
        assert np.allclose(start_base, NOMINAL_START + np.array([OFFSET[0], OFFSET[1], 0.0]), atol=1e-9)
        land_base, _, _ = rig.goal(5)
        assert np.allclose(land_base[:2], LAND_MARK[:2], atol=1e-3) and abs(land_base[2] - NOMINAL_LAND[2]) < 1e-9
        _, _, claw_pick = rig.goal(1)
        _, _, claw_place = rig.goal(3)
        assert np.linalg.norm(claw_pick - OBJ) < 1e-3 and np.linalg.norm(claw_place - PLACE_GOAL) < 1e-9
        print(f"[4] {rig.pp_status}: legs {np.round(rig.info[8:14], 1).tolist()} s, "
              f"total {rig.info[8:14].sum():.1f} s")

        # [4b] editing a point makes the plan STALE; re-planning takes the edit
        r = set_param(rig, "pick_place_place_point", (DROP + [0.1, 0.0, 0.0]).tolist())
        assert r.successful, r.reason
        wait(lambda: rig.pp_status == "NOT PLANNED: press Plan" and rig.info[4] == 0.0, 5, "plan stale")
        assert call(rig, "go_to_start").success is False
        assert call(rig, "plan").success
        wait(lambda: rig.pp_status.startswith("READY"), 30, "READY again")
        assert np.linalg.norm(rig.goal(3)[2] - (PLACE_GOAL + [0.1, 0.0, 0.0])) < 1e-9
        assert set_param(rig, "pick_place_place_point", DROP.tolist()).successful
        wait(lambda: rig.info[4] == 0.0, 5, "stale again")
        assert call(rig, "plan").success
        wait(lambda: rig.pp_status.startswith("READY"), 30, "READY with the original point")
        print("[4b] a typed edit made the plan stale; re-plan took it")

        # [4c] a land hover at 0.4 m: under the geofence floor, refused at Plan
        assert set_param(rig, "pick_place_land", [-1.0, 0.0, 0.4, 0.0]).successful
        wait(lambda: rig.info[4] == 0.0, 5, "stale")
        assert call(rig, "plan").success
        wait(lambda: rig.pp_status.startswith("INFEASIBLE"), 30, "fence refusal")
        assert "execute_land goal" in rig.pp_status and "fence floor" in rig.pp_status, rig.pp_status
        print(f"[4c] {rig.pp_status}")
        assert set_param(rig, "pick_place_land", NOMINAL_LAND.tolist() + [0.0]).successful
        assert call(rig, "plan").success
        wait(lambda: rig.pp_status.startswith("READY"), 30, "READY after the fence refusal")

        # [5] out of order
        r = call(rig, "execute_pick")
        assert not r.success and "out of order" in r.message, r.message
        print(f"[5] refused: {r.message}")

        # [6] the six legs
        T, refs = fly(rig, 0)
        print(f"[6] go_to_start T {T:.1f}s -> {rig.pp_status}")
        assert rig.pp_status == "DONE go_to_start next=execute_pick", rig.pp_status

        # [7a] a wait that times out: the leg ends INCOMPLETE above the target
        assert set_param(rig, "pick_place_approach_wait_max", 2.0).successful
        T = start_leg(rig, "execute_pick")
        wait_above(rig, 1, T)
        wait(lambda: rig.pp_status.startswith("INCOMPLETE: execute_pick"), 10, "wait timeout")
        assert int(rig.info[5]) == 0 and int(rig.info[6]) == -1, rig.info[5:7]
        print(f"[7a] {rig.pp_status}")
        assert set_param(rig, "pick_place_approach_wait_max", 60.0).successful
        # [7b] the body is not at its hold (the rig never flew there): refused
        r = call(rig, "execute_pick")
        assert not r.success and "from its hold" in r.message, r.message
        print(f"[7b] leg-start gate: {r.message}")
        rig.teleport(1, dz=APPROACH_DZ)

        def at_pick_approach(rig):
            # 80 mm off the point above the target: no descent however long
            rig.teleport(1, dz=APPROACH_DZ, off=(0.08, 0.0, 0.0))
            wait(lambda: abs(rig.arrival[2] - 0.08) < 2e-3, 5, "80 mm error")
            time.sleep(SETTLE_S + 1.0)
            assert rig.status == "HOLD" and rig.pp_status.startswith("WAITING execute_pick: claw 80 mm"), \
                rig.pp_status
            # 30 mm off: inside the tolerance -- descends after the dwell, not before
            t0 = time.monotonic()
            rig.teleport(1, dz=APPROACH_DZ, off=(0.03, 0.0, 0.0))
            wait(lambda: rig.status.startswith(("CALCULATING", "EXECUTING")), 5, "descent started")
            dt = time.monotonic() - t0
            assert SETTLE_S - 0.15 < dt < SETTLE_S + 0.8, dt
            print(f"    waiting above obj_0: no descent at 80 mm; at 30 mm it began after {dt:.2f} s")

        # the re-fly from the approach hold: a zero-length move, then the wait
        T, refs = fly(rig, 1, above=at_pick_approach)
        last = refs[-1]
        assert np.allclose(np.array(last.q_d), 0.0, atol=1e-9), last.q_d     # the pick pose
        r_ed = np.array([last.r_ed.x, last.r_ed.y, last.r_ed.z])
        assert np.linalg.norm(r_ed - OBJ) < 1e-3, r_ed                   # claw on obj_0
        # nose turned to face the object: final heading vs the bearing start -> object
        b = math.atan2(OBJ[1] - start_base[1], OBJ[0] - start_base[0])
        _, pick_yaw, _ = rig.goal(1)
        assert abs(math.remainder(pick_yaw - b, 2 * math.pi)) < math.radians(1.0), (pick_yaw, b)
        wait(lambda: rig.arrival is not None and rig.arrival[0] == 1 and rig.arrival[8] == 1.0, 5,
             "fine-correction gate")
        wait(lambda: "fine correction OK" in rig.pp_status, 5, "status gate")
        # B2: the measured claw heading in the ee_target convention -- claw down
        # with q1 = q4 = 0, the claw's lateral axis is the body's: the heading
        # is the vehicle's own yaw
        wait(lambda: rig.heading is not None and
             abs(math.remainder(rig.heading - pick_yaw, 2 * math.pi)) < 1e-6, 5, "claw heading")
        # B1: the claw-down grasp point (0.155 m out, 0.340 m down) is inside
        # the PICK-AND-PLACE envelope, and so are small moves up / in; the
        # other tabs' workspace_rz is left as it was (beta >= 5 deg: outside)
        wait(lambda: rig.ws is not None and rig.pp_ws is not None, 5, "workspace grids")
        base, _, claw = rig.goal(1)
        r = math.hypot(claw[0] - base[0], claw[1] - base[1])
        z = claw[2] - base[2]
        assert ws_contains(rig.pp_ws, r, z) and ws_contains(rig.pp_ws, r, z + 0.01) and \
            ws_contains(rig.pp_ws, r - 0.01, z), (r, z)
        assert not ws_contains(rig.pp_ws, r, z - 0.02), "below the bottom of the reach"
        assert not ws_contains(rig.ws, r, z), "workspace_rz must stay the beta >= 5 deg envelope"
        print(f"    claw heading {math.degrees(rig.heading):.1f} deg; claw (r {r:.3f}, z {z:.3f}) "
              "inside pick_place/workspace_rz, outside the unchanged workspace_rz")
        print(f"[7] execute_pick T {T:.1f}s, claw on obj_0 ({np.linalg.norm(r_ed - OBJ)*1e3:.2f} mm), "
              f"heading {math.degrees(pick_yaw):.1f} deg -> {rig.pp_status}")
        rig.odom_xyz = rig.odom_xyz + np.array([0.08, 0.0, 0.0])          # drifted 8 cm
        wait(lambda: rig.arrival[8] == 0.0 and abs(rig.arrival[2] - 0.08) < 2e-3, 5, "gate opens")
        wait(lambda: "outside tolerance" in rig.pp_status, 5, "status: outside")
        print(f"    8 cm off: |err| {rig.arrival[2]*1e3:.0f} mm -> {rig.pp_status}")
        rig.teleport(1)

        T, refs = fly(rig, 2)
        z = np.array([m.x_cd.z for m in refs])
        assert z.max() > refs[0].x_cd.z + 0.12, (z.max(), refs[0].x_cd.z)   # the retreat climb
        print(f"[8] go_to_place_start T {T:.1f}s (retreat climb {z.max() - refs[0].x_cd.z:.3f} m)")

        # [9a] ABORT in the middle of execute_place: the leg is cut where it
        # is, the CoM stays continuous, the arm goes home and the vehicle climbs;
        # the leg is not done, a second press is refused, then it is re-flown
        rig.refs.clear()
        assert call(rig, "execute_place").success
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 10, "execute_place EXECUTING")
        time.sleep(5.0)
        r = call(rig, "abort")
        assert r.success and "cut" in r.message, r.message
        r2 = call(rig, "abort")
        assert not r2.success and "already aborting" in r2.message, r2.message
        wait(lambda: rig.pp_status.startswith(("ABORTING", "ABORTED")), 5, "ABORTING")
        wait(lambda: rig.status == "HOLD" and rig.pp_status.startswith("ABORTED"), 30, "ABORTED, holding")
        refs = [m for _, m in rig.refs]
        xc = np.array([[m.x_cd.x, m.x_cd.y, m.x_cd.z] for m in refs])
        step = np.linalg.norm(np.diff(xc, axis=0), axis=1).max()
        assert step < 0.01, f"CoM reference jumped {step*1e3:.1f} mm at the cut"
        assert np.allclose(np.array(refs[-1].q_d), HOME, atol=1e-6), refs[-1].q_d
        assert int(rig.info[5]) == 2, "the cut leg must not count as done"
        assert rig.pp_status == "ABORTED -- holding higher, arm home; re-fly execute_place or Reset", rig.pp_status
        wait(lambda: rig.base_anchor is not None and rig.base_anchor.pose.position.z > 1.2, 5, "new hold")
        b = rig.base_anchor.pose
        print(f"[9a] abort mid-sweep: max CoM step {step*1e3:.1f} mm, arm home, base now z "
              f"{b.position.z:.3f} m; second press refused; {rig.pp_status}")
        rig.odom_xyz = np.array([b.position.x, b.position.y, b.position.z])
        rig.yaw = 2.0 * math.atan2(b.orientation.z, b.orientation.w)
        rig.q = HOME.copy()
        time.sleep(0.15)

        T, refs = fly(rig, 3)
        q = np.array([m.q_d for m in refs])
        last = refs[-1]
        r_ed = np.array([last.r_ed.x, last.r_ed.y, last.r_ed.z])
        assert np.linalg.norm(r_ed - PLACE_GOAL) < 1e-3, r_ed
        assert abs(q[:, 0].max() - math.radians(10)) < math.radians(1.0), math.degrees(q[:, 0].max())
        assert abs(q[:, 0].min() + math.radians(10)) < math.radians(1.0), math.degrees(q[:, 0].min())
        # q2 sweeps [17.5, 37.5] on the plateau (the gtest locks the band); here
        # the leg starts from the abort hold with the arm home (q2 40 deg), so
        # the highest q2 is that start pose -- never the old band's 45 deg
        assert q[:, 1].max() < math.radians(40.5), math.degrees(q[:, 1].max())
        steps = np.abs(np.diff(q, axis=0)).max()
        assert steps < 0.02, steps
        print(f"[9] execute_place T {T:.1f}s: q1 [{math.degrees(q[:, 0].min()):.1f}, "
              f"{math.degrees(q[:, 0].max()):.1f}] q2 max {math.degrees(q[:, 1].max()):.1f} deg, "
              f"claw on the typed place point -> {rig.pp_status}")

        T, _ = fly(rig, 4)
        print(f"[10] go_to_land_start T {T:.1f}s")
        T, refs = fly(rig, 5)
        assert np.allclose(np.array(refs[-1].q_d), HOME, atol=1e-6), refs[-1].q_d
        assert abs(rig.odom_xyz[2] - NOMINAL_LAND[2]) < 1e-9      # the hover over the spot, not the ground
        wait(lambda: rig.pp_status.startswith("COMPLETE"), 5, "COMPLETE")
        print(f"[11] execute_land T {T:.1f}s -> {rig.pp_status}")

        # [11a] ABORT from a hold: exactly pick_place_abort_climb higher, same x, y
        land_base, land_yaw, _ = rig.goal(5)
        r = call(rig, "abort")
        assert r.success and "the hold" in r.message, r.message
        wait(lambda: rig.status == "HOLD" and rig.pp_status.startswith("ABORTED"), 30, "abort from hold")
        wait(lambda: rig.base_anchor is not None and abs(rig.base_anchor.pose.position.z - land_base[2] - 0.30) < 1e-6,
             5, "climbed 0.30 m")
        b = rig.base_anchor.pose
        assert abs(b.position.x - land_base[0]) < 1e-9 and abs(b.position.y - land_base[1]) < 1e-9
        assert rig.pp_status.endswith("touch down with SAFETY -> land"), rig.pp_status
        print(f"[11a] abort from the hold: base z {land_base[2]:.2f} -> {b.position.z:.2f} m, x, y kept")

        # [11b] the abort climb is clipped at the fence ceiling
        rig.odom_xyz = np.array([b.position.x, b.position.y, b.position.z])
        z0 = b.position.z
        assert set_param(rig, "pick_place_fence_max_z", z0 + 0.10).successful
        r = call(rig, "abort")
        assert r.success and "up 0.10 m" in r.message and "clipped at the fence ceiling" in r.message, r.message
        wait(lambda: rig.status.startswith("EXECUTING"), 10, "clipped abort flying")
        wait(lambda: rig.status == "HOLD" and rig.pp_status.startswith("ABORTED"), 30, "clipped abort")
        wait(lambda: abs(rig.base_anchor.pose.position.z - (z0 + 0.10)) < 1e-6, 5, "at the ceiling")
        rig.odom_xyz = rig.odom_xyz + np.array([0.0, 0.0, 0.10])
        r = call(rig, "abort")
        assert r.success and "up 0.00 m" in r.message, r.message
        wait(lambda: rig.status.startswith("EXECUTING"), 10, "abort at the ceiling flying")
        wait(lambda: rig.status == "HOLD" and rig.pp_status.startswith("ABORTED"), 30, "abort at the ceiling")
        assert abs(rig.base_anchor.pose.position.z - (z0 + 0.10)) < 1e-6
        assert set_param(rig, "pick_place_fence_max_z", 1.8).successful
        print(f"[11b] ceiling {z0 + 0.10:.2f} m: {r.message}")

        # [12] reset, SAFETY
        r = call(rig, "reset")
        assert r.success, r.message
        wait(lambda: rig.pp_status == "NOT PLANNED: press Plan", 5, "reset")
        rig.mode.publish(String(data="SAFETY"))
        wait(lambda: rig.pp_status == "NOT IN DIRECT", 5, "SAFETY")
        time.sleep(0.3)
        n = len(rig.refs)
        time.sleep(0.5)
        assert len(rig.refs) == n, "still streaming in SAFETY"
        print("[12] reset; SAFETY silences the stream")
        print("PASS")
    finally:
        os.killpg(os.getpgid(gov.pid), signal.SIGINT)
        try:
            gov.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(gov.pid), signal.SIGKILL)
        print(f"node log: {log.name}")
        rig.destroy_node()
        rclpy.try_shutdown()
        spin.join(timeout=5.0)


if __name__ == "__main__":
    main()
