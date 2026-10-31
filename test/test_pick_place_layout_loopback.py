#!/usr/bin/env python3
"""Loopback test of a MOVED mail-delivery layout (2026-10-30) against the built
whole_body_trajectory_planner node (no sim, no PX4). Run it on a private ROS
domain (ROS_DOMAIN_ID=77), never beside a live stack.

On the day the pillar and the mailbox stand roughly where designed, their
layout relative to each other the same, and the drone may start elsewhere.
Here: the drone on the ground at (-1, 0) instead of (0, 0), and the pillar and
the mailbox both moved by (-0.2, -0.1) m. Everything the mission flies must
follow:

  Adjust              Start, Land Start and Land move to where the drone stands
                      (pick_place_adjust_max 2.0 allows the 1 m); Mailbox Start
                      is NOT moved by it
  Envelope Get        the pick point = the localization frame on the hat + the
                      pick locator
  Mailbox Get         the place point = the frame on the mailbox + the place
                      locator, AND Mailbox Start follows it
                      (pick_place_place_start_offset, turned by its yaw)
  Plan                refused until the Mailbox is measured (or typed); then the
                      planned goals sit on all of the above; a hand-typed place
                      point drags Mailbox Start along at Plan

and pick_place_place_start_offset [] keeps the old behaviour (Mailbox Start
typed, moved by Adjust).

Uses the simulation mail yaml (MAIL_YAML overrides) and the rig of
test_pick_place_get_loopback.py.
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

L.NS = "/uav_pplay"                       # (Rig reads these at construction)
L.OBJ_TOPIC = "/pplay_rig/obj_0/mocap"
L.LAND_TOPIC = "/pplay_rig/land_0/mocap"
MAIL_YAML = os.environ.get("MAIL_YAML", os.path.expanduser(
    "~/ros2_ws/src/fsc_autopilot_ros2/config/"
    "params_single_aerial_manipulator_whole_body_l1_4d_direct_actuation_t650_sim_mail_delivery.yaml"))
GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={L.NS}", "--params-file", MAIL_YAML,
           "-p", f"pick_place_pick_topic:={L.OBJ_TOPIC}"]
# the frame's marker centroid from its underside centre O, frame axes [m]
# (Pegasus archive/mail_delivery_20261023/cad/localization_frame.json)
CEN = np.array([-0.0035832, 0.0063756, 0.0410992])
SHIFT = np.array([-0.2, -0.1, 0.0])                      # the pillar and the mailbox, moved together
HAT_O = np.array([1.0, 1.0, 0.778]) + SHIFT              # the frame on the pick hat
BOX_O = np.array([1.7625, 0.25, 1.0725]) + SHIFT         # ... on the red mailbox's top
DRONE = np.array([-1.0, 0.0])                            # where the drone stands


def rz(deg, v):
    y = math.radians(deg)
    return np.array([math.cos(y) * v[0] - math.sin(y) * v[1], math.sin(y) * v[0] + math.cos(y) * v[1], v[2]])


def get_double(rig, name):
    """A double parameter of the planner."""
    assert rig.getp.wait_for_service(timeout_sec=10)
    fut = rig.getp.call_async(L.GetParameters.Request(names=[name]))
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    return fut.result().values[0].double_value


def frame_at(rig, O, yaw_deg):
    rig.move_payload(O + rz(yaw_deg, CEN))
    rig.obj_yaw = lambda n: math.radians(yaw_deg)
    time.sleep(0.8)


def close(a, b, tol, what):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and np.max(np.abs(a - b)) < tol, (what, a.tolist(), b.tolist())


def main():
    rclpy.init()
    rig = G.GetRig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    log = tempfile.NamedTemporaryFile("w+", prefix="pick_place_layout_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        wait(lambda: rig.pp_status is not None and rig.info is not None, 20, "planner up")
        off = get_param(rig, "pick_place_place_start_offset")
        ploc = get_param(rig, "pick_place_pick_locator")
        qloc = get_param(rig, "pick_place_place_locator")
        assert off.size == 4 and get_double(rig, "pick_place_adjust_max") >= 1.5, (off, MAIL_YAML)
        assert ploc.size >= 3 and qloc.size >= 3, "the mail yaml carries no frame locators"
        typed = {n: get_param(rig, "pick_place_" + n) for n in ("start", "place_start", "land_start", "land")}
        for bad in ([0.1, 0.2, 0.3], [0.1, float("nan"), 0.0, 0.0]):
            r = set_param(rig, "pick_place_place_start_offset", bad)
            assert not r.successful and "all finite" in r.reason, (bad, r.reason)
        print(f"[1] {os.path.basename(MAIL_YAML)}: place-start offset {off.tolist()}, adjust_max "
              f"{get_double(rig, 'pick_place_adjust_max')}; bad offsets refused")

        # [2] the drone on the ground at (-1, 0): Adjust moves Start / Land Start / Land, not Mailbox Start
        rig.odom_xyz = np.array([DRONE[0], DRONE[1], 0.24])
        time.sleep(0.3)
        r = call(rig, "adjust")
        assert r.success, r.message
        d = np.array([DRONE[0] - typed["start"][0], DRONE[1] - typed["start"][1], 0.0, 0.0])
        for n in ("start", "land_start", "land"):
            close(get_param(rig, "pick_place_" + n), typed[n] + d, 2e-3, n)
        close(get_param(rig, "pick_place_place_start"), typed["place_start"], 1e-9, "place_start (Adjust)")
        print(f"[2] Adjust at {DRONE.tolist()}: Start {np.round(get_param(rig, 'pick_place_start'), 3).tolist()}, "
              f"Land {np.round(get_param(rig, 'pick_place_land'), 3).tolist()}; Mailbox Start not moved")

        # [3] Envelope Get: the frame on the moved hat
        frame_at(rig, HAT_O, 0.0)
        r = call(rig, "capture_pick")
        assert r.success, r.message
        pick = get_param(rig, "pick_place_pick_point")
        close(pick[:3], HAT_O + CEN + ploc[:3], 2e-3, "pick point")
        print(f"[3] Envelope Get: pick {np.round(pick, 4).tolist()}")
        # [3a] the WRONG-FRAME mistake: the Mailbox Get with the frame still on the hat
        r = call(rig, "get_place")
        assert not r.success and "still on the pick hat" in r.message, r.message
        print(f"[3a] Mailbox Get with the frame on the hat: refused -- {r.message}")

        # [3b] DIRECT over the start: Plan is REFUSED while the Mailbox is the configured
        # one (the 2026-10-30 sim crash: no Mailbox Get, the release point inside the
        # moved mailbox)
        rig.odom_xyz = np.array([DRONE[0], DRONE[1], 1.0])
        time.sleep(0.3)
        rig.mode.publish(G.String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        r = call(rig, "plan")
        assert not r.success and "Mailbox Get" in r.message, r.message
        print(f"[3b] Plan before the Mailbox Get: refused -- {r.message}")

        # [4] Mailbox Get: the frame on the moved mailbox; Mailbox Start follows
        frame_at(rig, BOX_O, 0.0)
        r = call(rig, "get_place")
        assert r.success and "Place Start follows it" in r.message, r.message
        place = get_param(rig, "pick_place_place_point")
        close(place[:3], BOX_O + CEN + qloc[:3], 2e-3, "place point")
        ps = get_param(rig, "pick_place_place_start")
        close(ps, [place[0] + off[0], place[1] + off[1], place[2] + off[2], place[3] + off[3]], 1e-4, "place_start")
        print(f"[4] Mailbox Get: place {np.round(place, 4).tolist()} -> Mailbox Start {np.round(ps, 4).tolist()}")
        # [4b] ... and the Envelope Get with the frame on the mailbox
        r = call(rig, "capture_pick")
        assert not r.success and "still on the mailbox" in r.message, r.message
        close(get_param(rig, "pick_place_pick_point"), pick, 1e-9, "pick point after the refused Get")
        print(f"[4b] Envelope Get with the frame on the mailbox: refused -- {r.message}")

        # [5] DIRECT hover over the start, Plan: every goal on the moved layout
        rig.odom_xyz = np.array([DRONE[0], DRONE[1], 1.0])
        time.sleep(0.3)
        rig.mode.publish(G.String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        G.planned(rig)
        close(rig.goal(0)[0], get_param(rig, "pick_place_start")[:3], 1e-6, "go_to_start goal")
        ee = get_param(rig, "pick_place_ee_offset")              # the claw: the point + the EE offset
        close(rig.goal(1)[2], pick[:3] + rz(pick[3], ee), 1e-6, "pick claw goal")
        close(rig.goal(2)[0], ps[:3], 1e-6, "go_to_place_start goal")
        close(rig.goal(3)[2], place[:3] + rz(place[3], ee), 1e-6, "place claw goal")
        close(rig.goal(4)[0], get_param(rig, "pick_place_land_start")[:3], 1e-6, "go_to_land_start goal")
        close(rig.goal(5)[0], get_param(rig, "pick_place_land")[:3], 1e-6, "land goal")
        for k in range(6):
            b, y, c = rig.goal(k)
            print(f"    leg {k} {L.LEGS[k]:18s} base {np.round(b, 3).tolist()} yaw {math.degrees(y):6.1f}"
                  f"  claw {np.round(c, 3).tolist()}")
        print("[5] Plan READY on the moved layout")

        # [6] a mailbox turned 20 deg: Mailbox Start turns with it
        frame_at(rig, BOX_O, 20.0)
        r = call(rig, "get_place")
        assert r.success, r.message
        place = get_param(rig, "pick_place_place_point")
        ps = get_param(rig, "pick_place_place_start")
        assert abs(place[3] - 20.0) < 0.5, place
        close(ps[:3], place[:3] + rz(place[3], off[:3]), 1e-4, "turned place_start")
        assert abs(ps[3] - (place[3] + off[3])) < 1e-6, ps
        G.planned(rig)
        close(rig.goal(2)[0], ps[:3], 1e-6, "turned go_to_place_start goal")
        print(f"[6] mailbox at yaw {place[3]:.1f}: Mailbox Start {np.round(ps, 4).tolist()}, planned")

        # [7] a hand-typed place point: Plan drags Mailbox Start along
        frame_at(rig, BOX_O, 0.0)
        assert call(rig, "get_place").success
        typed_place = [1.45, 0.12, 1.03, 0.0]
        assert set_param(rig, "pick_place_place_point", typed_place).successful
        G.planned(rig)
        ps = get_param(rig, "pick_place_place_start")
        close(ps, [typed_place[0] + off[0], typed_place[1] + off[1], typed_place[2] + off[2], off[3]], 1e-4,
              "place_start after a typed place point")
        close(rig.goal(2)[0], ps[:3], 1e-6, "go_to_place_start goal (typed place)")
        print(f"[7] typed place {typed_place} -> at Plan Mailbox Start {np.round(ps, 4).tolist()}")

        # [8] the offset off: the old behaviour, Adjust moves the typed Mailbox Start
        assert set_param(rig, "pick_place_place_start_offset", []).successful
        before = get_param(rig, "pick_place_place_start")
        rig.odom_xyz = np.array([DRONE[0] + 0.1, DRONE[1] + 0.05, 1.0])
        time.sleep(0.3)
        assert call(rig, "adjust").success
        close(get_param(rig, "pick_place_place_start"), before + np.array([0.1, 0.05, 0.0, 0.0]), 2e-3,
              "place_start, offset off")
        print("[8] offset [] -> Adjust moves Mailbox Start with the drone again")
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
