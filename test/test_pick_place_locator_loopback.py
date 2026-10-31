#!/usr/bin/env python3
"""Loopback test of the pick-and-place LOCATOR FRAME (2026-10-29) against the
built whole_body_trajectory_planner node (no sim, no PX4). Run it on a private
ROS domain (ROS_DOMAIN_ID=77), never beside a live stack.

The mail delivery's targets carry no markers of their own: a printed frame with
five markers sits on the pick hat, then on the mailbox's top, and each row's Get
measures it. Two parameters carry the frame-to-target offset, in the frame's
axes and turned with its measured yaw:

  pick_place_pick_locator   capture_pick writes obj_0 + Rz(yaw) d, yaw + dyaw
  pick_place_place_locator  get_place writes obj_0 + Rz(yaw) d AND the yaw
                            (yaw + dyaw), written as the value NEAREST the typed
                            yaw so a typed turn keeps its direction

What is checked: [] / 3 / 4 values accepted, other lengths and non-finite values
refused; a change does not make the plan stale; both Gets go through the offset
at any frame yaw; the place yaw follows the frame, nearest the typed one; [] is
the old behaviour (obj_0 itself, the typed place yaw kept); the frozen-feed and
yaw-flip gates still refuse.

The rig and its helpers are test_pick_place_get_loopback.py's.
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

import test_pick_place_get_loopback as G
import test_pick_place_loopback as L
from test_pick_place_loopback import call, get_param, set_param, wait

L.NS = "/uav_pploc"                       # (Rig reads these at construction; after G's own)
L.OBJ_TOPIC = "/pploc_rig/obj_0/mocap"
L.LAND_TOPIC = "/pploc_rig/land_0/mocap"

GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={L.NS}", "--params-file", L.YAML,
           "-p", f"pick_place_pick_topic:={L.OBJ_TOPIC}"]
FRAME_HAT = np.array([1.0, 0.6, 0.78])     # the frame's marker centroid on the pick hat
FRAME_BOX = np.array([1.7, 0.25, 1.15])    # ... on the mailbox's top
D_PICK = [-0.1012, 0.0034, -0.0410]        # frame -> envelope centre [m]
D_PLACE = [-0.0594, 0.0034, -0.0915, 0.0]  # frame -> envelope centre at the release [m, deg]


def through(c, yaw_deg, d):
    y = math.radians(yaw_deg)
    return c + np.array([math.cos(y) * d[0] - math.sin(y) * d[1],
                         math.sin(y) * d[0] + math.cos(y) * d[1], d[2]])


def main():
    rclpy.init()
    rig = G.GetRig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    log = tempfile.NamedTemporaryFile("w+", prefix="pick_place_locator_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT,
                           text=True, start_new_session=True)
    try:
        wait(lambda: rig.pp_status is not None and rig.info is not None, 20, "planner up")
        rig.move_payload(FRAME_HAT)
        rig.obj_yaw = lambda n: math.radians(30.0)
        time.sleep(0.8)

        # [1] defaults off; the accepted and refused values
        for name in ("pick_place_pick_locator", "pick_place_place_locator"):
            assert get_param(rig, name).size == 0, name
            for bad in ([0.1, 0.2], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, float("nan"), 0.0]):
                r = set_param(rig, name, bad)
                assert not r.successful and "all finite" in r.reason, (name, bad, r.reason)
            for good in ([0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 5.0], []):
                assert set_param(rig, name, good).successful, (name, good)
        print(f"[1] defaults []; refused: {r.reason}")

        # [2] capture_pick through the pick locator, frame yaw 30 deg
        assert set_param(rig, "pick_place_pick_locator", D_PICK).successful
        r = call(rig, "capture_pick")
        assert r.success, r.message
        assert "LOCATOR FRAME" in r.message and "pick_place_pick_locator" in r.message, r.message
        p = get_param(rig, "pick_place_pick_point")
        want = through(FRAME_HAT, 30.0, D_PICK)
        assert np.linalg.norm(p[:3] - want) < 1e-3 and abs(p[3] - 30.0) < 1e-6, (p, want)
        wait(lambda: np.allclose(rig.info[G.INFO_PICK:G.INFO_PICK + 3], p[:3], atol=1e-12), 5,
             "info follows the located pick")
        print(f"[2] capture_pick: {r.message}\n    pick_place_pick_point = {np.round(p, 4).tolist()}")
        # ... a 4-value locator turns the pick yaw, and a frame across +-180 deg wraps it
        assert set_param(rig, "pick_place_pick_locator", D_PICK[:3] + [10.0]).successful
        rig.obj_yaw = lambda n: math.radians(175.0)
        time.sleep(0.8)
        assert call(rig, "capture_pick").success
        p = get_param(rig, "pick_place_pick_point")
        want = through(FRAME_HAT, 175.0, D_PICK)
        assert np.linalg.norm(p[:3] - want) < 1e-3 and abs(p[3] - (-175.0)) < 1e-6, (p, want)
        print(f"    + dyaw 10 at frame yaw 175: pick yaw {p[3]:.1f} deg")

        # [3] DIRECT, a plan; a locator change does not make it stale
        assert set_param(rig, "pick_place_pick_locator", D_PICK).successful
        rig.obj_yaw = lambda n: math.radians(30.0)
        time.sleep(0.8)
        assert call(rig, "capture_pick").success
        rig.odom_xyz = np.array([0.3, 0.2, 1.0])
        time.sleep(0.3)
        rig.mode.publish(G.String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        assert set_param(rig, "pick_place_place_point", [1.65, 0.25, 1.06, -330.0]).successful
        G.planned(rig)
        assert set_param(rig, "pick_place_place_locator", [0.0, 0.0, 0.0]).successful
        assert set_param(rig, "pick_place_place_locator", D_PLACE).successful
        time.sleep(0.3)
        assert rig.info[G.INFO_PLANNED] == 1.0 and rig.pp_status.startswith("READY"), rig.pp_status
        print("[3] locator changes keep the plan READY")

        # [4] get_place through the place locator: position AND yaw from the
        # frame; the yaw is written nearest the typed -330 (frame 40 -> -320)
        rig.move_payload(FRAME_BOX)
        rig.obj_yaw = lambda n: math.radians(40.0)
        time.sleep(0.8)
        r = call(rig, "get_place")
        assert r.success, r.message
        assert "LOCATOR FRAME" in r.message and "nothing to correct by hand" in r.message, r.message
        assert "the plan is stale" in r.message, r.message
        place = get_param(rig, "pick_place_place_point")
        want = through(FRAME_BOX, 40.0, D_PLACE)
        assert np.linalg.norm(place[:3] - want) < 1e-3, (place, want)
        assert abs(place[3] - (-320.0)) < 1e-6, place
        G.planned(rig)
        assert np.linalg.norm(rig.goal(3)[2] - place[:3]) < 1e-9, rig.goal(3)[2]
        print(f"[4] get_place: {r.message}\n    pick_place_place_point = {np.round(place, 4).tolist()}"
              "; the re-plan releases there")

        # [5] the locator off: obj_0 itself, the typed yaw kept (the old behaviour)
        assert set_param(rig, "pick_place_place_locator", []).successful
        r = call(rig, "get_place")
        assert r.success and "yaw kept" in r.message and "LOCATOR" not in r.message, r.message
        place0 = get_param(rig, "pick_place_place_point")
        assert np.linalg.norm(place0[:3] - FRAME_BOX) < 1e-3 and place0[3] == -320.0, place0
        print(f"[5] locator off: {r.message}")

        # [6] the gates still refuse through a locator
        assert set_param(rig, "pick_place_place_locator", D_PLACE).successful
        rig.obj_frozen = True
        time.sleep(0.8)
        for name in ("capture_pick", "get_place"):
            r = call(rig, name)
            assert not r.success and "frozen" in r.message, r.message
        rig.obj_frozen = False
        rig.obj_yaw = lambda n: math.radians(40.0) + (math.pi if (n // 5) % 2 else 0.0)
        time.sleep(0.8)
        for name in ("capture_pick", "get_place"):
            r = call(rig, name)
            assert not r.success and "flipping" in r.message, r.message
        assert np.array_equal(get_param(rig, "pick_place_place_point"), place0)
        print("[6] frozen and flipping feeds still refused")
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
