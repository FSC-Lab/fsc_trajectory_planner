#!/usr/bin/env python3
"""Loopback test of the MAIL ROUTE (pick_place_route, 2026-11-03) against the
built whole_body_trajectory_planner node (no sim, no PX4). Run it on a private
ROS domain (ROS_DOMAIN_ID=77), never beside a live stack.

The user's layout: three hats in a line in front of two mailboxes, extending
away from them (the nearest 1 m out, 0.6 m apart), each letter turned 90 deg
on its hat, letters posted alternately RED, BLUE, RED; the sim mail yaml's
insert mission. What is checked:

  the Gets       capture_letter_<k> / get_box_<b> write the localization frame
                 through the pick / place locator into the route's points; a
                 Get with the frame still on another hat is refused
  Plan           refused until every letter and box is measured; refused when a
                 pick would stand the airframe on another hat; READY with the
                 route fields in info [88..]
  letter 1       the six-leg insert mission's legs + exits, go_to_start = its Pick
                 Start; its place exit POSTS it and the route waits (no automatic
                 switch); re-selecting it is refused (its hat is empty)
  letter 2       selected by the operator; an abort while waiting above it
  letter 3       selected instead; with it in the claw, selecting another letter
                 and landing are refused; posted into RED; the land legs;
                 COMPLETE with the route tally
  Reset          back to letter 1; route off = the single mission again

The rig is test_pick_place_loopback.py's perfect plant (it teleports onto each
planned rest), its mocap body the localization frame, here with a yaw.
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
from std_srvs.srv import Trigger
from fsc_autopilot_ros2_msgs.msg import Mocap

import test_pick_place_loopback as L

LOC_PICK = np.array([-0.0914, -0.0064, -0.039])
LOC_PLACE = np.array([-0.0499, -0.0064, -0.0896])
HAT_X = [0.70, 0.10, -0.50]          # nearest first, 0.6 m apart, on y = 0
HAT_TOP = 0.80
BOX = {"RED": np.array([1.709, 0.25, 1.024]), "BLUE": np.array([1.709, -0.25, 1.024])}
PICK_BACK, PLACE_BACK, PITCH, EXIT_BACK, DZ = 0.10, 0.15, math.radians(25.0), 0.10, 0.20
PICK_START = np.array([-0.5, 0.0, 0.35])
HOVER = np.array([-0.5, -1.0, 1.0])  # DIRECT engaged here, off the line of hats


def arr(v):
    return "[" + ",".join(f"{float(x):.4f}" for x in v) + "]"


GOV_CMD = L.GOV_CMD + [
    "-p", "pick_place_route:=true",
    "-p", "pick_place_place_insert:=true",
    "-p", f"pick_place_place_approach_back:={PLACE_BACK}",
    "-p", "pick_place_place_insert_pitch_deg:=25.0",
    "-p", f"pick_place_pick_approach_back:={PICK_BACK}",
    "-p", f"pick_place_place_exit_back:={EXIT_BACK}",
    "-p", f"pick_place_approach_dz:={DZ}",
    "-p", "pick_place_pick_pose_deg:=[0.0,36.0,44.0,90.0]",
    "-p", "pick_place_place_pose_deg:=[0.0,20.0,45.0,90.0]",
    "-p", "pick_place_carry_pose_deg:=[0.0,36.0,44.0,90.0]",
    "-p", "pick_place_sweep_lo_deg:=[0.0,36.0,44.0,90.0]",
    "-p", "pick_place_sweep_hi_deg:=[0.0,36.0,44.0,90.0]",
    "-p", "pick_place_ee_offset:=[-0.06,0.0,0.0]",
    "-p", "pick_place_fence_min_z:=0.30",
    "-p", "pick_place_fence_max_z:=2.2",
    "-p", "pick_place_pick_locator:=" + arr(list(LOC_PICK) + [0.0]),
    "-p", "pick_place_place_locator:=" + arr(list(LOC_PLACE) + [0.0]),
    "-p", "pick_place_place_start_offset:=[-0.709,0.0,0.196,0.0]",
    "-p", "pick_place_min_pick_place_dist:=0.30",
    "-p", "pick_place_insert_descent_trim:=false",
    "-p", "pick_place_start:=[0.70,-0.70,1.10,90.0]",
    "-p", "pick_place_land_start:=[-0.5,-1.0,1.0,0.0]",
    "-p", "pick_place_land:=[-0.5,-1.0,0.8,0.0]"]


def rz(yaw, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def hat_frame(k):
    """The localization frame's markers on hat k (the letter turned 90 deg)."""
    return np.array([HAT_X[k], 0.0, HAT_TOP + 0.039]), math.radians(90.0)


def letter_point(k):
    p, yaw = hat_frame(k)
    return p + rz(yaw, LOC_PICK)


def box_frame(name):
    return BOX[name] - LOC_PLACE, 0.0


class RouteRig(L.Rig):
    """The pick-and-place rig with a yawed mocap body (the frame) and the
    route's services."""

    def __init__(self):
        super().__init__()
        self.frame_yaw = 0.0
        for name in ["capture_letter_1", "capture_letter_2", "capture_letter_3",
                     "get_box_1", "get_box_2", "select_letter_1", "select_letter_2", "select_letter_3"]:
            self.cli[name] = self.create_client(Trigger, f"whole_body_planner/pick_place/route/{name}")

    def mocap_msg(self, topic, p):
        m = Mocap()
        noisy = p + self.rng.normal(0.0, 0.0005, 3)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, noisy)
        m.pose.orientation.z = math.sin(0.5 * self.frame_yaw)
        m.pose.orientation.w = math.cos(0.5 * self.frame_yaw)
        return m

    def put_frame(self, p, yaw):
        self.mocap[L.OBJ_TOPIC] = (self.mocap[L.OBJ_TOPIC][0], np.array(p))
        self.frame_yaw = yaw
        time.sleep(0.8)       # a full capture window of the new pose

    def at(self, base, yaw):
        """A perfect plant: the measured pose is this rest."""
        self.odom_xyz = np.array(base)
        self.yaw = yaw
        if self.refs:
            self.q = np.array(self.refs[-1][1].q_d)
        time.sleep(0.15)


def nose(yaw):
    return np.array([math.cos(yaw), math.sin(yaw), 0.0])


def fly_claw(rig, leg, above=None):
    """A claw leg of the insert mission: to its approach rest (behind the
    envelope / back and up along the insertion line), WAIT, the rig onto it, the
    descent (the slide in) by itself, the hold on the goal."""
    name = L.LEGS[leg]
    rig.refs.clear()
    T = L.start_leg(rig, name)
    L.wait_above(rig, leg, T)
    if above is not None:
        return above(rig)
    base, yaw, _ = rig.goal(leg)
    if leg == 1:
        approach = np.array(base) - PICK_BACK * nose(yaw)
    else:
        approach = np.array(base) - PLACE_BACK * math.cos(PITCH) * nose(yaw) + \
            np.array([0.0, 0.0, PLACE_BACK * math.sin(PITCH)])
    rig.at(approach, yaw)
    L.wait(lambda: rig.status.startswith("EXECUTING") and "(descent)" in rig.pp_status, 15, f"{name} descent")
    L.wait(lambda: rig.status == "HOLD", 30, f"{name} descent complete")
    L.wait(lambda: rig.info is not None and int(rig.info[5]) == leg, 5, f"{name} recorded")
    rig.teleport(leg)


def exit_leg(rig, leg):
    """Exit To Pick / Place; the rig onto its end (the place's backs out first)."""
    base, yaw, _ = rig.goal(leg)
    end = np.array(base) + np.array([0.0, 0.0, DZ]) - (EXIT_BACK * nose(yaw) if leg == 3 else 0.0)
    rig.refs.clear()
    name = "exit_pick" if leg == 1 else "exit_place"
    r = L.call(rig, name)
    assert r.success, r.message
    L.wait(lambda: rig.status.startswith("EXECUTING") and f"FLYING {name}" in rig.pp_status, 10, f"{name} flying")
    L.wait(lambda: rig.status == "HOLD" and not rig.pp_status.startswith("FLYING"), 30, f"{name} complete")
    rig.at(end, yaw)


def route(rig):
    """info [88..]: on, letter, N, its box, states, boxes, T."""
    i = rig.info
    n = int(i[90])
    return {"on": i[88], "letter": int(i[89]), "n": n, "box": int(i[91]),
            "state": [int(x) for x in i[92:92 + n]], "boxes": [int(x) for x in i[92 + n:92 + 2 * n]],
            "T": i[92 + 2 * n]}


def main():
    rclpy.init()
    rig = RouteRig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    log = tempfile.NamedTemporaryFile("w+", prefix="mail_route_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        L.wait(lambda: rig.pp_status is not None and rig.info is not None, 20, "planner up")
        assert rig.pp_status == "NOT IN DIRECT", rig.pp_status
        assert route(rig)["on"] == 1.0 and route(rig)["n"] == 3 and route(rig)["boxes"] == [0, 1, 0], route(rig)
        print(f"[1] route on: 3 letters, boxes {route(rig)['boxes']} (RED, BLUE, RED)")

        # [2] the Gets, on the ground: the frame on each hat, then on each box
        for k in range(3):
            rig.put_frame(*hat_frame(k))
            r = L.call(rig, f"capture_letter_{k + 1}")
            assert r.success, r.message
            got = L.get_param(rig, "pick_place_route_letter_points")[4 * k:4 * k + 4]
            assert np.linalg.norm(got[:3] - letter_point(k)) < 2e-3 and abs(got[3] - 90.0) < 0.5, (got, letter_point(k))
            if k == 0:
                # the frame left on hat 1: letter 2's Get is refused
                r = L.call(rig, "capture_letter_2")
                assert not r.success and "is the frame still on" in r.message, r.message
                print(f"[2] frame still on hat 1: {r.message}")
        for b, name in enumerate(["RED", "BLUE"]):
            rig.put_frame(*box_frame(name))
            r = L.call(rig, f"get_box_{b + 1}")
            assert r.success, r.message
            got = L.get_param(rig, "pick_place_route_box_points")[4 * b:4 * b + 4]
            assert np.linalg.norm(got[:3] - BOX[name]) < 2e-3 and abs(got[3]) < 0.5, (got, BOX[name])
        rig.put_frame(np.array([5.0, 5.0, 0.0]), 0.0)      # the frame lifted off
        print("[2] three letters and two mailboxes measured through the locators (< 2 mm)")

        # [3] DIRECT at the hover off the line, then Plan; first a layout that
        # stands the airframe on a hat: letter 2's hat moved under letter 1's
        # approach hold -> refused by the hat check
        rig.odom_xyz = HOVER.copy()
        rig.yaw = 0.0
        time.sleep(0.5)       # the hover measured before DIRECT captures the hold
        rig.mode.publish(String(data="DIRECT"))
        L.wait(lambda: rig.status == "HOLD", 10, "planner HOLD")
        good = L.get_param(rig, "pick_place_route_letter_points")
        bad = good.copy()
        bad[4:7] = np.array([0.70, -0.40, HAT_TOP + 0.039]) + rz(math.radians(90.0), LOC_PICK)
        assert L.set_param(rig, "pick_place_route_letter_points", list(bad)).successful
        assert L.call(rig, "plan").success
        L.wait(lambda: rig.pp_status.startswith("INFEASIBLE"), 30, "plan refused")
        assert "letter 1" in rig.pp_status and "hat 2" in rig.pp_status, rig.pp_status
        print(f"[3] {rig.pp_status}")
        assert L.set_param(rig, "pick_place_route_letter_points", list(good)).successful
        t0 = time.monotonic()
        assert L.call(rig, "plan").success
        L.wait(lambda: rig.pp_status.startswith("READY"), 60, "route READY")
        R = route(rig)
        assert R["letter"] == 0 and R["state"] == [0, 0, 0] and R["T"] > 100.0, R
        assert rig.pp_status.endswith("[letter 1/3 -> mailbox RED]"), rig.pp_status
        assert rig.path is not None and len(rig.path) > 0
        print(f"[3] READY in {time.monotonic() - t0:.1f} s: {R['T']:.1f} s of flight -- {rig.pp_status}")

        # [4] letter 1: the whole insert mission from its Pick Start; its place exit
        # POSTS it and the route WAITS for the operator's next letter
        L.fly(rig, 0)
        base, yaw, _ = rig.goal(0)
        got = L.get_param(rig, "pick_place_route_letter_points")[0:4]
        assert np.linalg.norm(np.array(base) - (got[:3] + rz(math.radians(got[3]), PICK_START))) < 1e-6, base
        fly_claw(rig, 1)
        exit_leg(rig, 1)
        assert route(rig)["state"][0] == 1, route(rig)
        L.fly(rig, 2)
        _, _, claw = rig.goal(3)
        assert np.linalg.norm(np.array(claw) - (BOX["RED"] + np.array([-0.06, 0.0, 0.0]))) < 2e-3, claw   # the measured box
        fly_claw(rig, 3)
        exit_leg(rig, 3)
        L.wait(lambda: route(rig)["state"][0] == 2, 5, "letter 1 posted")
        time.sleep(0.5)
        R = route(rig)
        assert R["letter"] == 0 and R["state"] == [2, 0, 0] and int(rig.info[5]) == 3, R
        assert "letter 1 posted into mailbox RED -- select the next letter" in rig.pp_status, rig.pp_status
        print(f"[4] letter 1 posted, the route waits -> {rig.pp_status}")
        r = L.call(rig, "select_letter_1")
        assert not r.success and "already posted" in r.message, r.message

        # [5] the operator selects letter 2; an abort while waiting above it
        r = L.call(rig, "select_letter_2")
        assert r.success, r.message
        assert rig.pp_status == "READY next=go_to_start [letter 2/3 -> mailbox BLUE]", rig.pp_status
        L.fly(rig, 0)
        fly_claw(rig, 1, above=lambda rg: None)
        r = L.call(rig, "abort")
        assert r.success, r.message
        L.wait(lambda: rig.pp_status.startswith("ABORTED") and rig.status == "HOLD", 30, "aborted")
        assert "select another letter" in rig.pp_status, rig.pp_status
        base, yaw, _ = rig.goal(1)
        rig.at(np.array(base) - PICK_BACK * nose(yaw) + np.array([0.0, 0.0, 0.30]), yaw)
        assert route(rig)["state"] == [2, 0, 0], route(rig)     # never lifted: still to do
        print(f"[5] letter 2 selected, aborted above it -> {rig.pp_status}")

        # [6] letter 3 instead: in the claw, no switching and no landing; then posted
        r = L.call(rig, "select_letter_3")
        assert r.success, r.message
        assert route(rig)["letter"] == 2 and route(rig)["box"] == 0, route(rig)
        L.fly(rig, 0)
        fly_claw(rig, 1)
        exit_leg(rig, 1)
        r = L.call(rig, "select_letter_2")
        assert not r.success and "in the claw" in r.message, r.message
        r = L.call(rig, "go_to_land_start")
        assert not r.success and "in the claw" in r.message, r.message
        print(f"[6] letter 3 in the claw: {r.message}")
        L.fly(rig, 2)
        fly_claw(rig, 3)
        exit_leg(rig, 3)
        L.wait(lambda: route(rig)["state"] == [2, 0, 2], 5, "letter 3 posted")
        L.fly(rig, 4)
        L.fly(rig, 5)
        L.wait(lambda: rig.pp_status.startswith("COMPLETE"), 5, "COMPLETE")
        assert rig.pp_status.endswith("[route: 2 of 3 letters posted]"), rig.pp_status
        print(f"[6] {rig.pp_status}")

        # [7] Reset: letter 1 again; the route off = the single mission's Plan
        assert L.call(rig, "reset").success
        R = route(rig)
        assert R["letter"] == 0 and R["state"] == [0, 0, 0], R
        assert L.set_param(rig, "pick_place_route", False).successful
        L.wait(lambda: rig.pp_status.startswith("NOT PLANNED: capture pick"), 5,
               "the single mission's status (its own pick capture missing)")
        print(f"[7] reset to letter 1; route off -> {rig.pp_status}")
        print("PASS")
    finally:
        os.killpg(gov.pid, signal.SIGINT)
        try:
            gov.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(gov.pid, signal.SIGKILL)
        print(f"node log: {log.name}")
        rig.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=5.0)


if __name__ == "__main__":
    main()
