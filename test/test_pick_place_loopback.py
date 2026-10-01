#!/usr/bin/env python3
"""Loopback test of the PICK-AND-PLACE mode against the built
whole_body_trajectory_planner node (no sim, no PX4).

Rig: SAFETY -> Adjust measures the vehicle on the start mark (offset) -> a
leg is refused outside DIRECT -> DIRECT hold -> Plan refused until obj_0 /
drop_0 are captured -> capture pick, place and (a measured) land point ->
Plan -> READY -> a leg out of order is refused -> the six legs, one service
each; after every leg the rig teleports its odometry, attitude and joints onto
the leg's goal, as a perfect plant would. Checked on the way: the base goals
carry the offset, the claw ends ON the captured points with the arm in the
pick pose, the arrival error gates "fine correction OK" at 50 mm, the
retreat climbs before the transit, execute_place sweeps q1 / q2 across their
bands, COMPLETE after execute_land, Reset, SAFETY silences the stream.

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

from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleAttitude
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float64MultiArray, String
from std_srvs.srv import Trigger
from fsc_autopilot_ros2_msgs.msg import Mocap, WholeBodyReference
from ament_index_python.packages import get_package_share_directory

NS = "/uav_pickplace"
OBJ_TOPIC = "/pp_rig/obj_0/mocap"
DROP_TOPIC = "/pp_rig/drop_0/mocap"
LAND_TOPIC = "/pp_rig/land_0/mocap"
YAML = os.path.join(get_package_share_directory("fsc_trajectory_planner"), "config",
                    "whole_body_trajectory_planner_t650_aerial_manipulator.yaml")
GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={NS}", "--params-file", YAML,
           "-p", f"pick_place_pick_topic:={OBJ_TOPIC}",
           "-p", f"pick_place_place_topic:={DROP_TOPIC}",
           "-p", f"pick_place_land_topic:={LAND_TOPIC}"]
HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
LEGS = ["go_to_start", "execute_pick", "go_to_place_start", "execute_place",
        "go_to_land_start", "execute_land"]
NOMINAL_START = np.array([0.0, 0.0, 1.0])       # the package yaml's placeholders
NOMINAL_LAND = np.array([-1.0, 0.0, 0.8])
OBJ = np.array([1.0, 0.6, 0.70])
DROP = np.array([0.6, -1.6, 0.70])
LAND_MARK = np.array([-1.2, 0.3, 0.0])
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
                      DROP_TOPIC: (self.create_publisher(Mocap, DROP_TOPIC, 10), DROP),
                      LAND_TOPIC: (self.create_publisher(Mocap, LAND_TOPIC, 10), LAND_MARK)}
        self.status = None
        self.pp_status = None
        self.info = None
        self.path = None
        self.arrival = None
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
                            "capture_land", "capture_start"]:
            self.cli[name] = self.create_client(Trigger, f"whole_body_planner/pick_place/{name}")
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

    def teleport(self, leg):
        """A perfect plant: the measured pose becomes the leg's goal."""
        base, yaw, _ = self.goal(leg)
        self.odom_xyz = np.array(base)
        self.yaw = yaw
        self.q = np.array(self.refs[-1][1].q_d)


def ws_contains(ws, r, z):
    """Ps4Panel::reachable's cell lookup on the published workspace_rz."""
    r_min, r_max, z_min, z_max, nr, nz = ws[:6]
    nr, nz = int(nr), int(nz)
    if not (r_min <= r <= r_max and z_min <= z <= z_max):
        return False
    ri = int((r - r_min) / (r_max - r_min) * (nr - 1) + 0.5)
    zi = int((z_max - z) / (z_max - z_min) * (nz - 1) + 0.5)
    return bool(ws[6 + zi * nr + ri])


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


def fly(rig, leg_index):
    """Fly one leg through its service and wait for the hold at its goal."""
    name = LEGS[leg_index]
    rig.refs.clear()
    r = call(rig, name)
    assert r.success, f"{name}: {r.message}"
    wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 10, f"{name} EXECUTING")
    assert rig.pp_status.startswith("FLYING " + name), rig.pp_status
    T = float(rig.status.split("T=")[1].rstrip("s"))
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

        # [1] Adjust in SAFETY, the vehicle on the start mark: offset = measured - nominal (x, y)
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
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        wait(lambda: rig.pp_status.startswith("NOT PLANNED"), 5, "NOT PLANNED")
        assert "pick" in rig.pp_status and "place" in rig.pp_status, rig.pp_status
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
        # [3] captures: obj_0, drop_0 and a measured landing mark
        for name, want in (("capture_pick", OBJ), ("capture_place", DROP), ("capture_land", LAND_MARK)):
            r = call(rig, name)
            assert r.success, r.message
            print(f"    {name}: {r.message}")
        wait(lambda: rig.info[56 + 4 * 1 + 3] == 1.0 and rig.info[56 + 4 * 3 + 3] == 1.0, 5, "captures")
        assert np.linalg.norm(rig.info[60:63] - OBJ) < 1e-3, rig.info[60:63]
        assert np.linalg.norm(rig.info[68:71] - DROP) < 1e-3, rig.info[68:71]
        assert rig.pp_status == "NOT PLANNED: press Plan", rig.pp_status
        print("[3] pick / place / land captured")

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
        assert np.linalg.norm(claw_pick - OBJ) < 1e-3 and np.linalg.norm(claw_place - DROP) < 1e-3
        print(f"[4] {rig.pp_status}: legs {np.round(rig.info[8:14], 1).tolist()} s, "
              f"total {rig.info[8:14].sum():.1f} s")

        # [5] out of order
        r = call(rig, "execute_pick")
        assert not r.success and "out of order" in r.message, r.message
        print(f"[5] refused: {r.message}")

        # [6] the six legs
        T, refs = fly(rig, 0)
        print(f"[6] go_to_start T {T:.1f}s -> {rig.pp_status}")
        assert rig.pp_status == "DONE go_to_start next=execute_pick", rig.pp_status

        T, refs = fly(rig, 1)
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

        T, refs = fly(rig, 3)
        q = np.array([m.q_d for m in refs])
        last = refs[-1]
        r_ed = np.array([last.r_ed.x, last.r_ed.y, last.r_ed.z])
        assert np.linalg.norm(r_ed - DROP) < 1e-3, r_ed
        assert abs(q[:, 0].max() - math.radians(25)) < math.radians(1.5), math.degrees(q[:, 0].max())
        assert abs(q[:, 0].min() + math.radians(25)) < math.radians(1.5), math.degrees(q[:, 0].min())
        assert abs(q[:, 1].max() - math.radians(45)) < math.radians(1.5), math.degrees(q[:, 1].max())
        steps = np.abs(np.diff(q, axis=0)).max()
        assert steps < 0.02, steps
        print(f"[9] execute_place T {T:.1f}s: q1 [{math.degrees(q[:, 0].min()):.1f}, "
              f"{math.degrees(q[:, 0].max()):.1f}] q2 max {math.degrees(q[:, 1].max()):.1f} deg, "
              f"claw on drop_0 -> {rig.pp_status}")

        T, _ = fly(rig, 4)
        print(f"[10] go_to_land_start T {T:.1f}s")
        T, refs = fly(rig, 5)
        assert np.allclose(np.array(refs[-1].q_d), HOME, atol=1e-6), refs[-1].q_d
        assert abs(rig.odom_xyz[2] - NOMINAL_LAND[2]) < 1e-9      # the hover over the spot, not the ground
        wait(lambda: rig.pp_status.startswith("COMPLETE"), 5, "COMPLETE")
        print(f"[11] execute_land T {T:.1f}s -> {rig.pp_status}")

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
