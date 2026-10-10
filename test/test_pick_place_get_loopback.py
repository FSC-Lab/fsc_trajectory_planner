#!/usr/bin/env python3
"""Loopback test of the pick-and-place MEASURED POINTS (2026-10-10) against the
built whole_body_trajectory_planner node (no sim, no PX4). Run it on a private
ROS domain (ROS_DOMAIN_ID=77), never beside a live stack.

What changed that day (operator request after the first hardware
pick-and-place flights), and what is checked here:

  pick_place_pick_point   the pick capture as a PARAMETER, both ways: [] until
                          captured; capture_pick writes [x, y, z, yaw_deg]; a
                          set from outside sets the capture (info [60..63],
                          [80]) and makes the plan stale; [] clears it; a wrong
                          length or a non-finite value is refused; refused
                          while a leg is in flight.
  pick_place/get_place    the PICK body (the payload, standing on the place
                          platform) -> pick_place_place_point x, y, z, the
                          typed yaw KEPT, nothing subtracted from z, the pick
                          capture untouched; refused while a leg is in flight.
  the shared capture gate (capture_pick and get_place alike): a FROZEN feed
                          (bit-identical positions) is refused; a FLIPPING yaw
                          is refused (pick_place_capture_max_yaw_spread_deg,
                          which does not make the plan stale; <= 0 disables),
                          while a steady yaw at +-180 deg is not.

The rig, its helpers and the mocap jitter are test_pick_place_loopback.py's;
the mocap body is a rig-private topic.
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
from std_msgs.msg import String

import test_pick_place_loopback as L
from test_pick_place_loopback import call, get_param, set_param, wait

L.NS = "/uav_ppget"                       # (Rig reads these at construction)
L.OBJ_TOPIC = "/ppget_rig/obj_0/mocap"
L.LAND_TOPIC = "/ppget_rig/land_0/mocap"
GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={L.NS}", "--params-file", L.YAML,
           "-p", f"pick_place_pick_topic:={L.OBJ_TOPIC}"]
PICK = np.array([1.0, 0.6, 0.70])         # the payload on the pick platform
PICK_YAW = 12.0                           # ... its yaw [deg]
PLACE = np.array([0.62, -1.58, 0.71])     # the payload on the place platform
PLACE_YAW_TYPED = 15.0                    # the drone's heading at the place, typed [deg]
INFO_PICK, INFO_PICK_YAW, INFO_PLANNED = 60, 80, 4


class GetRig(L.Rig):
    """The loopback rig with a payload that can turn, freeze and flip."""

    def __init__(self):
        self.obj_frozen = False
        self.obj_yaw = lambda n: math.radians(PICK_YAW)   # [rad], of the n-th sample
        self.obj_n = 0
        super().__init__()

    def mocap_msg(self, topic, p):
        m = super().mocap_msg(topic, p)
        if topic == L.OBJ_TOPIC:
            self.obj_n += 1
            if self.obj_frozen:           # the processor republishing its last pose
                m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
            yaw = self.obj_yaw(self.obj_n)
            m.pose.orientation.z, m.pose.orientation.w = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
        return m

    def move_payload(self, p):
        self.mocap[L.OBJ_TOPIC] = (self.mocap[L.OBJ_TOPIC][0], np.array(p, dtype=float))


def pick_state(rig):
    """(parameter, info capture block [x, y, z, valid], info yaw)."""
    return (get_param(rig, "pick_place_pick_point"), rig.info[INFO_PICK:INFO_PICK + 4].copy(),
            rig.info[INFO_PICK_YAW])


def planned(rig):
    assert call(rig, "plan").success
    wait(lambda: rig.pp_status.startswith("READY"), 30, "READY")
    assert rig.info[INFO_PLANNED] == 1.0


def main():
    rclpy.init()
    rig = GetRig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    log = tempfile.NamedTemporaryFile("w+", prefix="pick_place_get_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT,
                           text=True, start_new_session=True)
    try:
        wait(lambda: rig.pp_status is not None and rig.info is not None, 20, "planner up")
        rig.move_payload(PICK)
        time.sleep(0.8)

        # [1] nothing captured: the parameter is [], bad values are refused
        p, cap, _ = pick_state(rig)
        assert p.size == 0 and cap[3] == 0.0 and np.all(np.isnan(cap[:3])), (p, cap)
        r = set_param(rig, "pick_place_pick_point", [1.0, 2.0, 3.0])
        assert not r.successful and "needs [x, y, z, yaw_deg]" in r.reason, r.reason
        r = set_param(rig, "pick_place_pick_point", [1.0, 2.0, float("nan"), 0.0])
        assert not r.successful and "non-finite" in r.reason, r.reason
        assert get_param(rig, "pick_place_pick_point").size == 0
        print(f"[1] not captured: pick_place_pick_point = []; refused: {r.reason}")

        # [2] capture_pick fills the parameter: [x, y, z, yaw_deg] = the capture
        r = call(rig, "capture_pick")
        assert r.success, r.message
        wait(lambda: rig.info[INFO_PICK + 3] == 1.0, 5, "pick captured")
        p, cap, yaw = pick_state(rig)
        assert p.shape == (4,) and np.linalg.norm(p[:3] - PICK) < 1e-3, p
        assert abs(p[3] - PICK_YAW) < 1e-6, p
        assert np.allclose(p[:3], cap[:3], atol=1e-12) and abs(p[3] - yaw) < 1e-9, (p, cap, yaw)
        print(f"[2] capture_pick: {r.message}\n    pick_place_pick_point = {np.round(p, 4).tolist()}")

        # [3] DIRECT, a plan; a pick point SET FROM OUTSIDE (the GS's Pick boxes)
        # sets the capture and makes the plan stale; the re-plan flies to it
        rig.odom_xyz = np.array([0.3, 0.2, 1.0])     # off the start point: go_to_start [7] takes seconds
        time.sleep(0.3)
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        assert set_param(rig, "pick_place_place_point", [0.6, -1.6, 0.70, PLACE_YAW_TYPED]).successful
        planned(rig)
        assert np.linalg.norm(rig.goal(1)[2] - p[:3]) < 1e-9, rig.goal(1)[2]
        edit = [1.05, 0.55, 0.72, 20.0]
        r = set_param(rig, "pick_place_pick_point", edit)
        assert r.successful, r.reason
        wait(lambda: rig.info[INFO_PLANNED] == 0.0 and rig.pp_status == "NOT PLANNED: press Plan", 5,
             "plan stale after the pick edit")
        wait(lambda: np.allclose(rig.info[INFO_PICK:INFO_PICK + 3], edit[:3], atol=1e-12), 5,
             "info follows the pick edit")
        assert rig.info[INFO_PICK + 3] == 1.0 and abs(rig.info[INFO_PICK_YAW] - 20.0) < 1e-9, rig.info[60:81]
        assert call(rig, "go_to_start").success is False       # the old plan is not flown
        planned(rig)
        assert np.linalg.norm(rig.goal(1)[2] - np.array(edit[:3])) < 1e-9, rig.goal(1)[2]
        assert abs(math.degrees(rig.goal(1)[1]) - 20.0) < 1e-6, math.degrees(rig.goal(1)[1])
        print(f"[3] pick_place_pick_point set to {edit}: info [60..63], [80] follow, the plan went "
              f"stale, the re-plan puts the claw on it at yaw {math.degrees(rig.goal(1)[1]):.1f} deg")

        # [4] get_place: the payload carried to the place platform. Straight
        # after the move both positions are in the window: refused (scatter)
        rig.move_payload(PLACE)
        rig.obj_yaw = lambda n: math.radians(40.0)     # the payload's own yaw there: NOT used
        time.sleep(0.15)
        r = call(rig, "get_place")
        assert not r.success and "scatter" in r.message and "Place NOT changed" in r.message, r.message
        assert np.allclose(get_param(rig, "pick_place_place_point"), [0.6, -1.6, 0.70, PLACE_YAW_TYPED])
        print(f"[4] mid-move: {r.message}")
        time.sleep(0.8)
        pick_before = pick_state(rig)
        r = call(rig, "get_place")
        assert r.success, r.message
        assert "yaw kept" in r.message and "RESTING height" in r.message and "0.02" in r.message, r.message
        assert "the plan is stale" in r.message, r.message
        place = get_param(rig, "pick_place_place_point")
        assert place.shape == (4,) and np.linalg.norm(place[:3] - PLACE) < 1e-3, place
        assert place[3] == PLACE_YAW_TYPED, place                # the typed yaw, exactly
        assert abs(place[2] - PLACE[2]) < 1e-3, place            # nothing subtracted from z
        wait(lambda: rig.info[INFO_PLANNED] == 0.0 and rig.pp_status == "NOT PLANNED: press Plan", 5,
             "plan stale after get_place")
        pick_after = pick_state(rig)
        for a, b in zip(pick_before, pick_after):
            assert np.array_equal(a, b), (pick_before, pick_after)
        planned(rig)
        assert np.linalg.norm(rig.goal(3)[2] - place[:3]) < 1e-9, rig.goal(3)[2]
        print(f"    get_place: {r.message}\n    pick_place_place_point = {np.round(place, 4).tolist()}"
              "; the pick capture is untouched; the re-plan releases there")
        # ... a 3-value place point (before 2026-10-03) reads as yaw 0 and is kept as 0
        assert set_param(rig, "pick_place_place_point", [0.6, -1.6, 0.70]).successful
        assert call(rig, "get_place").success
        place3 = get_param(rig, "pick_place_place_point")
        assert place3.shape == (4,) and place3[3] == 0.0 and np.linalg.norm(place3[:3] - PLACE) < 1e-3
        assert set_param(rig, "pick_place_place_point", place.tolist()).successful

        # [5] a FROZEN feed (an untracked body republished with fresh stamps):
        # both refused, nothing changed
        planned(rig)
        rig.obj_frozen = True
        time.sleep(0.8)
        for name, verdict in (("capture_pick", "NOT captured"), ("get_place", "Place NOT changed")):
            r = call(rig, name)
            assert not r.success and "frozen" in r.message and "not tracked" in r.message \
                and verdict in r.message, r.message
            print(f"[5] frozen feed, {name}: {r.message}")
        assert np.array_equal(get_param(rig, "pick_place_place_point"), place)
        for a, b in zip(pick_after, pick_state(rig)):
            assert np.array_equal(a, b)
        assert rig.info[INFO_PLANNED] == 1.0, "a refused Get made the plan stale"
        rig.obj_frozen = False

        # [6] a FLIPPING yaw (two solutions of a near-symmetric marker layout,
        # ~180 deg apart, several times a second): both refused
        rig.obj_yaw = lambda n: math.radians(40.0) + (math.pi if (n // 5) % 2 else 0.0)
        time.sleep(0.8)
        for name in ("capture_pick", "get_place"):
            r = call(rig, name)
            assert not r.success and "flipping between two solutions" in r.message \
                and "asymmetric" in r.message, r.message
            print(f"[6] flipping yaw, {name}: {r.message}")
        assert np.array_equal(get_param(rig, "pick_place_place_point"), place)
        # the gate is a parameter that does NOT make the plan stale; <= 0 disables it
        assert set_param(rig, "pick_place_capture_max_yaw_spread_deg", 0.0).successful
        time.sleep(0.3)
        assert rig.info[INFO_PLANNED] == 1.0 and rig.pp_status.startswith("READY"), rig.pp_status
        r = call(rig, "get_place")
        assert r.success, r.message
        assert set_param(rig, "pick_place_capture_max_yaw_spread_deg", 10.0).successful
        # a STEADY yaw at +-180 deg (samples either side of the wrap) is not a flip
        rig.move_payload(PICK)
        rig.obj_yaw = lambda n: math.pi + (0.01 if n % 2 else -0.01)
        time.sleep(0.8)
        r = call(rig, "capture_pick")
        assert r.success, r.message
        p180 = get_param(rig, "pick_place_pick_point")
        assert abs(abs(p180[3]) - 180.0) < 0.1 and np.linalg.norm(p180[:3] - PICK) < 1e-3, p180
        print(f"[6] gate off: get_place accepted; yaw at +-180 deg: {r.message}")

        # [7] in flight: a new pick point and get_place are refused
        rig.obj_yaw = lambda n: math.radians(PICK_YAW)
        time.sleep(0.8)
        assert call(rig, "capture_pick").success
        planned(rig)
        T = L.start_leg(rig, "go_to_start")
        r = call(rig, "get_place")
        assert not r.success and "in flight" in r.message, r.message
        rs = set_param(rig, "pick_place_pick_point", edit)
        assert not rs.successful and "in flight" in rs.reason, rs.reason
        assert abs(get_param(rig, "pick_place_pick_point")[3] - PICK_YAW) < 1e-6
        wait(lambda: rig.status == "HOLD", T + 20, "go_to_start complete")
        print(f"[7] in flight: {r.message} / {rs.reason}")

        # [8] [] clears the capture: Plan asks for it again; Reset keeps a capture
        assert set_param(rig, "pick_place_pick_point", edit).successful
        assert call(rig, "reset").success
        p, cap, yaw = pick_state(rig)
        assert np.allclose(p, edit) and cap[3] == 1.0 and abs(yaw - 20.0) < 1e-9, (p, cap, yaw)
        assert set_param(rig, "pick_place_pick_point", []).successful
        wait(lambda: rig.info[INFO_PICK + 3] == 0.0, 5, "capture cleared")
        wait(lambda: rig.pp_status.startswith("NOT PLANNED: capture pick"), 5, "asks for the capture")
        assert get_param(rig, "pick_place_pick_point").size == 0
        r = call(rig, "plan")
        assert not r.success and "capture" in r.message and "pick" in r.message, r.message
        print(f"[8] Reset kept the pick point; [] cleared it: {rig.pp_status}")
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
