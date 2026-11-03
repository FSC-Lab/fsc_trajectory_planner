#!/usr/bin/env python3
"""OPERATOR FUZZER for the MAIL ROUTE (2026-11-03): a random operator and
safety pilot against the built whole_body_trajectory_planner node, looking for
the sequences a human can produce at the arm GS's Mail Route tab that leave the
planner wrong, stuck or unsafe. No sim, no PX4: test_mail_route_loopback.py's
perfect-plant rig (each motion's goal becomes the measured pose).

    ROS_DOMAIN_ID=81 /usr/bin/python3 fuzz_mail_route_loopback.py --seed 1 --steps 60

Each step is one random action, weighted like a session: the Gets (the frame
sometimes left on the wrong hat), Plan (the route's, or the single mission's
from another tab), a letter button, a leg, an exit, ABORT (from a hold or
mid-motion), Reset, a margin edit, the route parameter flipped, a double press,
and the safety pilot's SAFETY -> DIRECT (from a hold or mid-motion). The rig
keeps its OWN record of where every letter physically is (the GS closes the
gripper after a pick descent and opens it for the insert, an abort opens it),
and after every step the invariants are checked against it:

  CARRY    a letter in the claw: landing, another letter, Reset and the
           single mission's Plan are all refused
  POSTED   a posted letter is never selectable again (until Reset)
  PHANTOM  the status never reads FLYING / ABORTING for long while the
           planner holds with nothing to fly (a stuck motion)
  INFO     info [88..] is well formed: N 3, letter in range, states 0..3
  ALIVE    the node answers its services and its log has no crash

A violation is printed with the step log that led to it; the run continues
(or stops at --stop-on-first). Exit 0 = no violation.
"""
import argparse
import math
import os
import random
import re
import signal
import subprocess
import tempfile
import threading
import time

import numpy as np
import rclpy
from std_msgs.msg import String

import test_pick_place_loopback as L
import test_mail_route_loopback as M

N = 3
LEGS = L.LEGS
BOXES = ["RED", "BLUE"]

GOV_CMD = M.GOV_CMD + [
    # the rig cannot know where an abort or an exit ended: the leg start check off
    "-p", "pick_place_leg_start_tol:=5.0",
    "-p", "pick_place_approach_wait_max:=20.0"]


def get_param_bool(rig, name):
    """A bool parameter of the planner (L.get_param reads double arrays only)."""
    from rcl_interfaces.srv import GetParameters
    assert rig.getp.wait_for_service(timeout_sec=10)
    fut = rig.getp.call_async(GetParameters.Request(names=[name]))
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, name
    return bool(fut.result().values[0].bool_value)


class Truth:
    """Where things physically are, as the operator's own eyes would say."""

    def __init__(self):
        self.where = ["hat"] * N     # hat / claw / posted / dropped / unknown
        self.claw = None             # the letter in the claw
        self.at_pick_of = None       # the claw closed on this letter at its hat (not yet lifted)
        self.selected = 0
        self.direct = False

    def carrying(self):
        return self.claw is not None or self.at_pick_of is not None


class Fuzzer:
    def __init__(self, rig, seed, log_path):
        self.rig = rig
        self.rng = random.Random(seed)
        self.t = Truth()
        self.steps = []
        self.violations = []
        self.log_path = log_path
        self.since_motion = time.monotonic()

    # ----------------------------------------------------------- utilities
    def note(self, s):
        line = f"[{len(self.steps):3d}] {s}"
        self.steps.append(line)
        print(line, flush=True)

    def violate(self, rule, what):
        v = f"VIOLATION {rule}: {what}\n      planner: {self.rig.status} | {self.rig.pp_status}\n" \
            f"      truth: where {self.t.where} claw {self.t.claw} at_pick {self.t.at_pick_of} " \
            f"selected {self.t.selected} direct {self.t.direct}"
        self.violations.append((rule, what, list(self.steps[-12:])))
        print(v, flush=True)

    def call(self, name):
        try:
            return L.call(self.rig, name)
        except AssertionError as e:
            self.violate("ALIVE", f"{name} did not answer ({e})")
            return None

    def route(self):
        i = self.rig.info
        if i is None or len(i) < 92:
            return None
        return M.route(self.rig)

    def idle(self, timeout=60.0):
        """Wait until nothing moves: HOLD and no FLYING/ABORTING/PLANNING."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            s, p = self.rig.status or "", self.rig.pp_status or ""
            if s.startswith("HOLD") and not p.startswith(("FLYING", "ABORTING", "PLANNING")):
                return True
            if s == "IDLE":
                return True
            time.sleep(0.1)
        return False

    # ----------------------------------------------------------- invariants
    def check(self):
        R = self.route()
        if R is None:
            self.violate("INFO", "no pick_place/info")
            return
        if R["n"] != N or not (-1 <= R["letter"] < N) or any(s not in (0, 1, 2, 3) for s in R["state"]):
            self.violate("INFO", f"malformed route info {R}")
        # PHANTOM: a motion reported while the planner holds with nothing streaming
        s, p = self.rig.status or "", self.rig.pp_status or ""
        if s == "HOLD" and p.startswith(("FLYING", "ABORTING")):
            t0 = time.monotonic()
            while time.monotonic() - t0 < 5.0:
                time.sleep(0.2)
                s, p = self.rig.status or "", self.rig.pp_status or ""
                if not (s == "HOLD" and p.startswith(("FLYING", "ABORTING"))):
                    break
            else:
                self.violate("PHANTOM", f"'{p[:90]}' for 5 s while the planner HOLDs")
        # ALIVE: the node log
        try:
            log = open(self.log_path).read()
        except OSError:
            log = ""
        for pat in ("terminate called", "Segmentation", "core dumped", "what():"):
            if pat in log:
                self.violate("ALIVE", f"node log: {pat}")
                break
        # CARRY: probe the refusals the operator must get with a letter in the claw (DIRECT only)
        if self.t.direct and self.t.carrying() and self.idle(5.0) and (not self.chaos or self.rng.random() < 0.9):
            self.probe_carrying()
        # POSTED
        if self.t.direct and self.idle(5.0):
            for k in range(N):
                if self.t.where[k] == "posted" and self.rng.random() < 0.3:
                    r = self.call(f"select_letter_{k + 1}")
                    if r is not None and r.success and "selected" in r.message:
                        self.violate("POSTED", f"posted letter {k + 1} selected again: {r.message}")
                        self.t.selected = k

    def probe_carrying(self):
        k = self.t.claw if self.t.claw is not None else self.t.at_pick_of
        probes = [("go_to_land_start", "landing"), ("reset", "Reset")]
        other = [j for j in range(N) if j != k]
        probes.append((f"select_letter_{self.rng.choice(other) + 1}", "another letter"))
        name, what = self.rng.choice(probes)
        r = self.call(name)
        if r is None:
            return
        if r.success:
            self.violate("CARRY", f"letter {k + 1} in the claw, yet {what} was accepted ({name}): {r.message}")
            if name == "go_to_land_start":
                self.idle(60.0)
                self.rig.at(np.array(self.rig.odom_xyz), self.rig.yaw)
            if name == "reset":
                self.t.selected = 0
                self.t.where = ["hat" if w in ("hat", "posted", "unknown", "dropped") else w for w in self.t.where]
            if name.startswith("select_letter_"):
                self.t.selected = int(name[-1]) - 1
        # the single mission's Plan (another tab's), with the route parameter off
        if self.rng.random() < 0.1:
            # (flipping the parameter makes the plan stale, as any edit does: restored after)
            was = get_param_bool(self.rig, "pick_place_route")
            L.set_param(self.rig, "pick_place_route", False)
            r = self.call("plan")
            if r is not None and r.success:
                self.violate("CARRY", f"letter {k + 1} in the claw, yet the single-mission Plan was accepted: "
                             f"{r.message}")
            L.set_param(self.rig, "pick_place_route", was)

    # ----------------------------------------------------------- the plant / the GS
    def settle_motion(self, leg=None, timeout=90.0):
        """Follow a motion as a perfect plant would; returns how it ended."""
        rig = self.rig
        t0 = time.monotonic()
        descended = False
        while time.monotonic() - t0 < timeout:
            s, p = rig.status or "", rig.pp_status or ""
            if p.startswith("WAITING") and s == "HOLD" and leg in L.CLAW_LEGS:
                # the claw holds above its target: the plant goes onto the approach rest
                base, yaw, _ = rig.goal(leg)
                if leg == 1:
                    ap = np.array(base) - M.PICK_BACK * M.nose(yaw)
                else:
                    ap = np.array(base) - M.PLACE_BACK * math.cos(M.PITCH) * M.nose(yaw) + \
                        np.array([0.0, 0.0, M.PLACE_BACK * math.sin(M.PITCH)])
                rig.at(ap, yaw)
                time.sleep(0.5)
                continue
            if "(descent)" in p:
                descended = True
            if s.startswith("HOLD") and not p.startswith(("FLYING", "WAITING", "ABORTING", "PLANNING")):
                return "aborted" if p.startswith("ABORTED") else ("descended" if descended else "done")
            if s == "IDLE":
                return "safety"
            time.sleep(0.1)
        return "timeout"

    # ----------------------------------------------------------- actions
    def a_get(self):
        box = self.rng.random() < 0.35
        if box:
            b = self.rng.randrange(2)
            on = b if self.rng.random() < 0.85 else 1 - b
            self.rig.put_frame(*M.box_frame(BOXES[on]))
            r = self.call(f"get_box_{b + 1}")
            self.note(f"Get mailbox {BOXES[b]} with the frame on {BOXES[on]}: "
                      f"{'ok' if r and r.success else 'refused'} {r.message[:80] if r else ''}")
        else:
            k = self.rng.randrange(N)
            on = k if self.rng.random() < 0.8 else self.rng.randrange(N)
            self.rig.put_frame(*M.hat_frame(on))
            r = self.call(f"capture_letter_{k + 1}")
            self.note(f"Get letter {k + 1} with the frame on hat {on + 1}: "
                      f"{'ok' if r and r.success else 'refused'} {r.message[:80] if r else ''}")
        self.rig.put_frame(np.array([5.0, 5.0, 0.0]), 0.0)

    def a_plan(self):
        if self.rng.random() < 0.8:
            r = self.call("route_plan")
            self.note(f"route Plan: {r.message[:100] if r else ''}")
        else:
            r = self.call("plan")
            self.note(f"/plan (another tab's Plan): {r.message[:100] if r else ''}")
        if r is not None and r.success:
            t0 = time.monotonic()
            while time.monotonic() - t0 < 60:
                p = self.rig.pp_status or ""
                if not p.startswith(("PLANNING", "NOT PLANNED")) or "INFEASIBLE" in p:
                    break
                time.sleep(0.1)

    def a_select(self):
        k = self.rng.randrange(N)
        r = self.call(f"select_letter_{k + 1}")
        self.note(f"Letter {k + 1}: {'ok' if r and r.success else 'refused'} {r.message[:90] if r else ''}")
        if r is not None and r.success and "selected" in r.message:
            self.t.selected = k
            self.t.at_pick_of = None

    def a_leg(self, abort_after=None, safety_after=None, force=None):
        # an operator mostly presses the next leg, sometimes any
        done = int(self.rig.info[5]) if self.rig.info is not None else -1
        if force is not None:
            leg = force
        else:
            leg = min(done + 1, 5) if self.rng.random() < 0.75 else self.rng.randrange(6)
        r = self.call(LEGS[leg])
        ok = r is not None and r.success
        self.note(f"leg {LEGS[leg]}: {'ok' if ok else 'refused'} {r.message[:90] if r else ''}")
        if not ok:
            return
        if leg != 1 and self.t.at_pick_of is not None:
            # a leg flown off the pick with the jaws closed lifts the letter, as the exit does
            self.t.claw = self.t.at_pick_of
            self.t.where[self.t.claw] = "claw"
            self.t.at_pick_of = None
        if abort_after is not None:
            time.sleep(abort_after)
            self.a_abort(mid=True)
            return
        if safety_after is not None:
            time.sleep(safety_after)
            self.a_safety(mid=True)
            return
        how = self.settle_motion(leg)
        self.note(f"   -> {how}: {self.rig.pp_status[:90]}")
        if how in ("done", "descended"):
            base, yaw, _ = self.rig.goal(leg)
            self.rig.at(np.array(base), yaw)
            # the claw went to the planner's letter in progress: that is the hat it is on
            R = self.route()
            sel = R["letter"] if R is not None and R["letter"] >= 0 else self.t.selected
            self.t.selected = sel
            if leg == 1 and how == "descended" and self.t.where[sel] == "hat":
                self.t.at_pick_of = sel          # the GS closes the gripper on it
            if leg == 3 and how == "descended" and self.t.claw is not None:
                pass                             # in the mouth; posted when the GS opens + exits

    def a_exit(self, safety_after=None):
        name = getattr(self, "_exit_name", None) or ("exit_pick" if self.rng.random() < 0.5 else "exit_place")
        self._exit_name = None
        leg = 1 if name == "exit_pick" else 3
        r = self.call(name)
        ok = r is not None and r.success
        self.note(f"{name}: {'ok' if ok else 'refused'} {r.message[:90] if r else ''}")
        if not ok:
            return
        if safety_after is not None:
            # the safety pilot takes SAFETY during the exit and hands DIRECT back; the letter
            # stays wherever the gripper had it (an exit after the release: posted)
            if leg == 3 and self.t.claw is not None:
                self.t.where[self.t.claw] = "posted"
                self.t.claw = None
            time.sleep(safety_after)
            self.a_safety(mid=True)
            return
        if leg == 3 and self.t.claw is not None:
            # the GS opened the gripper (Insert) before Exit To Insert
            self.t.where[self.t.claw] = "posted"
            self.t.claw = None
        how = self.settle_motion(None)
        self.note(f"   -> {how}: {self.rig.pp_status[:90]}")
        if how == "done" and leg == 1 and self.t.at_pick_of is not None:
            self.t.claw = self.t.at_pick_of
            self.t.where[self.t.claw] = "claw"
            self.t.at_pick_of = None
        self.rig.at(np.array(self.rig.odom_xyz) + np.array([0.0, 0.0, 0.2]), self.rig.yaw)

    def a_abort(self, mid=False):
        r = self.call("abort")
        self.note(f"ABORT{' mid-motion' if mid else ''}: {'ok' if r and r.success else 'refused'} "
                  f"{r.message[:80] if r else ''}")
        if r is None or not r.success:
            self.settle_motion(None)
            return
        how = self.settle_motion(None)
        self.note(f"   -> {how}: {self.rig.pp_status[:90]}")
        # the abort opens the gripper: a letter in the claw falls (at its hat it stays there)
        if self.t.at_pick_of is not None:
            self.t.where[self.t.at_pick_of] = "hat"
            self.t.at_pick_of = None
        if self.t.claw is not None:
            self.t.where[self.t.claw] = "unknown"
            self.t.claw = None
        self.rig.at(np.array(self.rig.odom_xyz) + np.array([0.0, 0.0, 0.3]), self.rig.yaw)

    def a_reset(self):
        r = self.call("reset")
        self.note(f"Reset: {'ok' if r and r.success else 'refused'} {r.message[:90] if r else ''}")
        if r is not None and r.success:
            self.t.selected = 0
            self.t.where = ["hat" if w in ("hat", "posted", "unknown", "dropped") else w for w in self.t.where]
            # (the operator puts the letters back for a new round)

    def a_margin(self):
        name, lo, hi = self.rng.choice([("pick_place_approach_dz", 0.10, 0.30),
                                        ("pick_place_place_approach_back", 0.10, 0.25),
                                        ("pick_place_route_hat_margin", 0.0, 0.06)])
        v = round(self.rng.uniform(lo, hi), 3)
        r = L.set_param(self.rig, name, v)
        self.note(f"edit {name} = {v}: {'ok' if r.successful else 'refused ' + r.reason[:60]}")

    def a_route_param(self):
        v = self.rng.random() < 0.5
        r = L.set_param(self.rig, "pick_place_route", v)
        self.note(f"pick_place_route := {v}: {'ok' if r.successful else r.reason[:60]}")

    def a_double(self):
        name = self.rng.choice(["route_plan", "abort", "exit_pick", "exit_place", "reset",
                                f"select_letter_{self.rng.randrange(N) + 1}"] + LEGS)
        r1 = self.call(name)
        r2 = self.call(name)
        self.note(f"double {name}: {r1.success if r1 else None}/{r2.success if r2 else None} "
                  f"{(r2.message if r2 else '')[:70]}")
        if name.startswith("select_letter_") and r2 is not None and r2.success and "selected" in r2.message:
            self.t.selected = int(name[-1]) - 1
        if name == "reset" and ((r1 and r1.success) or (r2 and r2.success)):
            self.t.selected = 0
        if name == "exit_place" and r1 is not None and r1.success and self.t.claw is not None:
            self.t.where[self.t.claw] = "posted"
            self.t.claw = None
        how = self.settle_motion(None)
        if name == "exit_pick" and r1 is not None and r1.success and how == "done" and \
                self.t.at_pick_of is not None:
            self.t.claw = self.t.at_pick_of
            self.t.where[self.t.claw] = "claw"
            self.t.at_pick_of = None
        if name == "abort" and r1 is not None and r1.success:
            if self.t.at_pick_of is not None:
                self.t.at_pick_of = None
            if self.t.claw is not None:
                self.t.where[self.t.claw] = "unknown"
                self.t.claw = None

    def a_safety(self, mid=False):
        self.rig.mode.publish(String(data="SAFETY"))
        self.t.direct = False
        time.sleep(self.rng.uniform(0.5, 3.0))
        self.rig.mode.publish(String(data="DIRECT"))
        self.t.direct = True
        time.sleep(0.5)
        self.note(f"SAFETY -> DIRECT{' mid-motion' if mid else ''}: {self.rig.status} | {self.rig.pp_status[:80]}")

    def a_sensible(self):
        """What a competent operator presses next, from the tab's own state."""
        R = self.route()
        i = self.rig.info
        p = self.rig.pp_status or ""
        if R is None or i is None:
            return self.a_plan()
        if p.startswith(("NOT PLANNED", "INFEASIBLE")) or math.isnan(R["T"]):
            return self.a_plan()
        done, exited_pick, exited_place = int(i[5]), i[83] > 0.5, i[84] > 0.5
        st = R["state"]
        cur = R["letter"] if R["letter"] >= 0 else 0
        if done == 1 and not exited_pick:
            return self.a_exit_named("exit_pick")
        if done == 3 and not exited_place:
            return self.a_exit_named("exit_place")
        if done >= 5:
            return self.a_reset()
        if cur < len(st) and st[cur] in (2, 3) and done < 4:
            todo = [k for k in range(N) if st[k] in (0, 3) and k != cur]
            if todo:
                k = self.rng.choice(todo)
                r = self.call(f"select_letter_{k + 1}")
                self.note(f"Letter {k + 1} (next): {'ok' if r and r.success else 'refused'} "
                          f"{r.message[:90] if r else ''}")
                if r is not None and r.success and "selected" in r.message:
                    self.t.selected = k
                return
            return self.a_leg(force=4)        # every letter dealt with: land
        return self.a_leg(force=min(done + 1, 5) if done != 3 else 4)

    def a_exit_named(self, name):
        self._exit_name = name
        return self.a_exit()

    chaos = False

    def step(self):
        if self.rng.random() < (0.5 if self.chaos else 0.7):
            self.a_sensible()
            self.check()
            return
        w = [("get", 4), ("plan", 6), ("select", 8), ("leg", 30), ("exit", 10), ("abort", 4),
             ("abort_mid", 4), ("reset", 2), ("margin", 5), ("route_param", 2), ("double", 5),
             ("safety", 2), ("safety_mid", 3), ("safety_exit", 3)]
        if self.chaos:
            # the risky events, several times as likely: mid-motion aborts and SAFETY, edits
            w = [("get", 3), ("plan", 4), ("select", 6), ("leg", 10), ("exit", 6), ("abort", 4),
                 ("abort_mid", 14), ("reset", 3), ("margin", 8), ("route_param", 2), ("double", 6),
                 ("safety", 3), ("safety_mid", 12), ("safety_exit", 12)]
        a = self.rng.choices([x for x, _ in w], weights=[y for _, y in w])[0]
        if a == "get":
            self.a_get()
        elif a == "plan":
            self.a_plan()
        elif a == "select":
            self.a_select()
        elif a == "leg":
            self.a_leg()
        elif a == "exit":
            self.a_exit()
        elif a == "abort":
            self.a_abort()
        elif a == "abort_mid":
            self.a_leg(abort_after=self.rng.uniform(0.2, 4.0))
        elif a == "reset":
            self.a_reset()
        elif a == "margin":
            self.a_margin()
        elif a == "route_param":
            self.a_route_param()
        elif a == "double":
            self.a_double()
        elif a == "safety":
            self.a_safety()
        elif a == "safety_mid":
            self.a_leg(safety_after=self.rng.uniform(0.2, 4.0))
        elif a == "safety_exit":
            self.a_exit(safety_after=self.rng.uniform(0.2, 2.0))
        self.check()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--stop-on-first", action="store_true")
    ap.add_argument("--scenario", default="", help="safety_exit: SAFETY during an exit; skip_exit: Go To Mailbox Start without Exit To Pick")
    ap.add_argument("--chaos", action="store_true", help="the risky events several times as likely")
    a = ap.parse_args()
    rclpy.init()
    rig = M.RouteRig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()
    log = tempfile.NamedTemporaryFile("w+", prefix=f"fuzz_mail_route_{a.seed}_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    fz = Fuzzer(rig, a.seed, log.name)
    fz.chaos = a.chaos
    try:
        L.wait(lambda: rig.pp_status is not None and rig.info is not None, 20, "planner up")
        # the session's start, as an operator does it: the Gets on the ground, DIRECT at the hover
        for k in range(N):
            rig.put_frame(*M.hat_frame(k))
            assert L.call(rig, f"capture_letter_{k + 1}").success
        for b, name in enumerate(BOXES):
            rig.put_frame(*M.box_frame(name))
            assert L.call(rig, f"get_box_{b + 1}").success
        rig.put_frame(np.array([5.0, 5.0, 0.0]), 0.0)
        rig.odom_xyz = M.HOVER.copy()
        rig.yaw = 0.0
        time.sleep(0.5)
        rig.mode.publish(String(data="DIRECT"))
        fz.t.direct = True
        L.wait(lambda: rig.status == "HOLD", 10, "planner HOLD")
        fz.note("session start: 5 Gets, DIRECT at the hover")
        if a.scenario == "safety_exit":
            # the stuck-exit bug: SAFETY during Exit To Pick, DIRECT handed back
            fz.a_plan()
            fz._exit_name = None
            fz.a_leg(force=0)
            fz.a_leg(force=1)
            r = fz.call("exit_pick")
            fz.note(f"exit_pick: {r.message[:80] if r else ''}")
            time.sleep(0.8)
            fz.a_safety(mid=True)
            fz.check()
            r = fz.call("exit_pick")
            fz.note(f"exit_pick again after DIRECT: {'ok' if r and r.success else 'refused'} "
                    f"{r.message[:90] if r else ''}")
            if r is None or not r.success:
                fz.violate("PHANTOM", f"the exit cannot be pressed again after SAFETY: {r.message if r else ''}")
            else:
                how = fz.settle_motion(None)
                fz.note(f"   -> {how}: {rig.pp_status[:90]}")
                R = fz.route()
                if R["state"][R["letter"]] != 1:
                    fz.violate("CARRY", f"after the re-pressed exit the letter is not recorded in the claw: {R}")
                r = fz.call("go_to_land_start")
                if r is not None and r.success:
                    fz.violate("CARRY", "landing accepted with the letter in the claw after the SAFETY exit")
            a.steps = 0
        if a.scenario == "skip_exit":
            # Go To Mailbox Start pressed straight after the pick (no Exit To Pick): its retreat
            # lifts the letter; landing, Reset and another letter must stay refused
            fz.a_plan()
            fz.a_leg(force=0)
            fz.a_leg(force=1)
            fz.a_leg(force=2)
            R = fz.route()
            fz.note(f"after Go To Mailbox Start without the exit: states {R['state']}")
            if R["state"][R["letter"]] != 1:
                fz.violate("CARRY", f"the letter lifted by the retreat is not recorded in the claw: {R}")
            for name in ("go_to_land_start", "reset", "select_letter_2"):
                r = fz.call(name)
                fz.note(f"{name}: {'ok' if r and r.success else 'refused'} {r.message[:80] if r else ''}")
                if r is not None and r.success:
                    fz.violate("CARRY", f"{name} accepted with the letter in the claw")
                    break
            a.steps = 0
        for _ in range(a.steps):
            fz.step()
            if a.stop_on_first and fz.violations:
                break
    except Exception as e:   # the harness itself: report, do not hide
        fz.violate("HARNESS", f"{type(e).__name__}: {e}")
    finally:
        os.killpg(gov.pid, signal.SIGINT)
        try:
            gov.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(gov.pid, signal.SIGKILL)
        rig.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=5.0)
    print(f"\nseed {a.seed}: {len(fz.steps)} steps, {len(fz.violations)} violation(s); node log {log.name}")
    seen = set()
    for rule, what, steps in fz.violations:
        key = (rule, re.sub(r"[0-9.]+", "#", what)[:60])
        if key in seen:
            continue
        seen.add(key)
        print(f"--- {rule}: {what}")
        for s in steps:
            print("    " + s)
    raise SystemExit(1 if fz.violations else 0)


if __name__ == "__main__":
    main()
