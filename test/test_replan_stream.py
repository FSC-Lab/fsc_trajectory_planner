#!/usr/bin/env python3
"""Regression for the 2026-08-23 re-plan oscillation.

Symptom: after a drone-GS target had moved the setpoint, pressing Trajectory
Planning and then Send made the aerial manipulator lurch/oscillate.

Cause, in two halves:
  * the whole-body planner's Picard solve holds the GIL for ~0.26 s in one block, which
    is LONGER than the node's `wb_streamed_ref_timeout_s` (0.25 s), so the
    streamed reference went stale during every re-plan; and
  * on staleness the node fell back to its internal builder, whose reference
    tracks the RAW ground-station setpoint — a metre away once the operator had
    sent a new drone target. The law was flicked onto that and back.

This test covers the whole-body planner half, which is observable from outside: while a
re-plan runs, the reference stream must keep flowing well inside the staleness
window, and the setpoint it carries must not move (the whole-body planner streams a
static hold while it solves).

Run with the workspace sourced (fsc_trajectory_planner built); no Pegasus needed.
"""
import os
import signal
import subprocess
import sys
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
from std_msgs.msg import String
from fsc_autopilot_ros2_msgs.msg import (PositionControllerReference,
                                         WholeBodyReference)

NS = "/uav_rp"
GOV_CMD = ["ros2", "launch", "fsc_trajectory_planner",
           "whole_body_trajectory_planner_launch.py",
           f"uav_prefix:={NS.lstrip('/')}"]

# Which transition planner the whole-body planner under test should load. Every check
# here is backend-agnostic by construction — both planners take the same
# options and return the same reference keys — so the same suite is what
# qualifies a new backend.  WB_GOV_PLANNER=bspline exercises the flat B-spline
# one; unset keeps the whole-body planner's own default (straight_line).
_PLANNER = os.environ.get("WB_GOV_PLANNER", "").strip()
GOV_ARGS = ([f"planner:={_PLANNER}"] if _PLANNER else [])

HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
# The node's own staleness window. The stream must stay comfortably inside it.
TIMEOUT_S = 0.25


class Rig(Node):
    def __init__(self):
        super().__init__("replan_rig", namespace=NS)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode",
            latched)
        self.odom = self.create_publisher(
            Odometry, "state_estimator/local_position/odom", 10)
        # The whole-body planner now takes ATTITUDE from PX4, as the law does, so the
        # rig must feed it: NED q (w,x,y,z) = (0.70711, 0, 0, 0.70711) is
        # exactly identity in ENU/FLU, i.e. the actual yaw = 0 this rig means.
        self.att = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        # The arm's MEASURED state -- since 2026-09-04 this, not the
        # smoothed reference, is what the whole-body planner captures.
        self.js = self.create_publisher(
            JointState, "fsc_open_manipulator/joint_states", 10)
        self.sm = self.create_publisher(
            JointState, "fsc_open_manipulator/external_torque_controller/"
                        "smoothed_reference_joint_trajectory", 10)
        self.gs = self.create_publisher(
            PositionControllerReference,
            "fsc_autopilot_ros2/position_controller/reference", 10)
        self.ee = self.create_publisher(
            PoseStamped, "whole_body_planner/ee_target", 10)
        self.stamps = []
        self.xcd = []
        self.status = None
        self.create_subscription(
            WholeBodyReference,
            "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            self._ref, 100)
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data),
                                 latched)
        self.create_timer(0.02, self._feed)

    def _ref(self, m):
        self.stamps.append(time.monotonic())
        self.xcd.append((m.x_cd.x, m.x_cd.y, m.x_cd.z))

    def _feed(self):
        o = Odometry()
        o.pose.pose.position.z = 1.2
        o.pose.pose.orientation.w = 1.0
        _a = VehicleAttitude()
        _a.q = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
        self.att.publish(_a)
        self.odom.publish(o)
        j = JointState()
        j.name = ["joint1", "joint2", "joint3", "joint4"]
        j.position = list(HOME)
        j.velocity = [0.0] * 4
        self.js.publish(j)   # the capture reads THIS (encoders)
        self.sm.publish(j)


def wait(cond, t, what, repeat=None):
    """Wait for `cond`. `repeat` is re-called each poll: a ONE-SHOT publish can
    be dropped when DDS discovery has not finished matching the pair yet (the
    known trap — the publisher exists, the subscription is not connected), so
    the command is re-sent until it visibly lands."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < t:
        if cond():
            return
        if repeat is not None:
            repeat()
        time.sleep(0.05)
    raise AssertionError("timeout: " + what)


def main():
    rclpy.init()
    rig = Rig()
    threading.Thread(target=rclpy.spin, args=(rig,), daemon=True).start()
    gov = subprocess.Popen(GOV_CMD + GOV_ARGS, start_new_session=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
    try:
        wait(lambda: rig.status is not None, 25, "planner up")
        wait(lambda: rig.status == "HOLD", 20, "HOLD",
             repeat=lambda: rig.mode.publish(String(data="DIRECT")))

        # The operator's drone-GS target: +1 m in x. This is what made the
        # builder's fallback reference a metre away from the streamed one.
        gs = PositionControllerReference()
        gs.position.x, gs.position.y, gs.position.z = 1.0, 0.0, 1.2
        gs.yaw = 0.0
        gs.yaw_unit = PositionControllerReference.DEGREES
        wait(lambda: rig.status and rig.status.startswith("PLANNED"), 60,
             "ride-along PLANNED", repeat=lambda: rig.gs.publish(gs))
        print(f"[1] drone-GS target sent, auto-plan: {rig.status}")

        # The repeats above are the point: a ground station re-sending an
        # UNCHANGED setpoint must not restart the solve. Before this was
        # de-duplicated the status flickered CALCULATING <-> PLANNED and a
        # stable PLANNED could not be caught at all.
        for _ in range(10):
            rig.gs.publish(gs)
            time.sleep(0.05)
        assert rig.status.startswith("PLANNED"), (
            f"a repeated identical drone target restarted planning "
            f"(status {rig.status})")
        print("[1b] repeated identical drone target ignored — still PLANNED")

        # Now the reported action: press Trajectory Planning (an explicit EE
        # target) and watch the stream across the whole re-plan.
        rig.stamps.clear()
        rig.xcd.clear()
        rig.status = None
        ee = PoseStamped()
        ee.pose.position.x = 1.22
        ee.pose.position.y = 0.0
        ee.pose.position.z = 1.02
        ee.pose.orientation.w = 1.0
        rig.ee.publish(ee)
        t_clear = time.monotonic()
        wait(lambda: rig.status and (rig.status.startswith("PLANNED") or
                                     "INFEASIBLE" in rig.status), 90,
             "re-plan verdict")
        print(f"[2] re-plan: {rig.status}")
        # The C++ planner answers in a few ms, so "until PLANNED" is a window
        # of ~3-5 samples and says nothing about the stream. Score a FIXED
        # 1 s window that contains the solve instead: the gap and hold-motion
        # checks below are what the regression is about, and they hold over
        # any window that includes the re-plan.
        while time.monotonic() - t_clear < 1.0:
            time.sleep(0.02)

        st = np.array(rig.stamps)
        assert len(st) > 5, f"only {len(st)} reference samples during re-plan"
        gaps = np.diff(st)
        worst = float(gaps.max())
        span = float(st[-1] - st[0])
        rate = (len(st) - 1) / span
        # The count follows from the window length, so the metrics that matter
        # are the RATE and the WORST GAP, not how many samples fitted.
        print(f"[3] stream during the re-plan: {len(st)} samples over "
              f"{span:.2f} s = {rate:.0f} Hz, worst gap {worst * 1e3:.0f} ms "
              f"(staleness window {TIMEOUT_S * 1e3:.0f} ms)")
        assert worst < TIMEOUT_S, (
            f"stream stalled {worst * 1e3:.0f} ms — the node would have "
            "declared the reference stale mid-engagement")
        assert rate > 60.0, f"stream degraded to {rate:.0f} Hz while solving"

        # And the setpoint must not have moved while solving: the whole-body planner
        # streams a static hold, so a frozen sample and a streamed one agree.
        x = np.array(rig.xcd)
        span = float(np.linalg.norm(x - x[0], axis=1).max())
        print(f"[4] setpoint motion while solving: {span * 1e3:.3f} mm "
              "(must be ~0 — it is a hold)")
        assert span < 1e-6, "the hold setpoint moved during planning"
        print("=== RE-PLAN STREAM CONTINUITY — PASS ===")
    finally:
        os.killpg(os.getpgid(gov.pid), signal.SIGINT)
        try:
            out, _ = gov.communicate(timeout=5)
            print("---- whole-body planner log ----")
            print("\n".join(out.splitlines()[-14:]))
        except subprocess.TimeoutExpired:
            gov.kill()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
