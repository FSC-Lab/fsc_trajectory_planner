#!/usr/bin/env python3
"""Loopback regression test of the C++ whole_body_trajectory_planner node
(no sim, no PX4, no Pegasus).

Run it after any planner change -- ~40 s, needs only a sourced workspace with
fsc_trajectory_planner and fsc_autopilot_ros2_msgs built:

    python3 test_planner_loopback.py                 # the yaml default (bspline)

Emulates the WB node's mode topic, odometry, PX4 attitude, the arm's
measured joint_states and a DECOY smoothed reference 2.3 deg away from it;
drives: SAFETY-silence -> DIRECT hold -> drone target (PENDING->PLANNED
ride-along) -> EE target (replan) -> Send -> EXECUTING -> completion HOLD ->
SAFETY silence. Asserts stream content and continuity at every stage.
"""
import math
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
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import JointTrajectory
from std_srvs.srv import Trigger
from fsc_autopilot_ros2_msgs.msg import (PositionControllerReference,
                                         WholeBodyReference)

NS = "/uav_gov_test"
import os
# The node under test is the installed C++ executable, launched through the
# package launch file so the yaml defaults apply exactly as in a flight.
GOV_CMD = ["ros2", "launch", "fsc_trajectory_planner",
           "whole_body_trajectory_planner_launch.py",
           f"uav_prefix:={NS.lstrip('/')}"]

# Which transition planner the whole-body planner under test should load. Every check
# here is backend-agnostic by construction — both planners take the same
# options and return the same reference keys — so the same suite is what
# qualifies a new backend.  WB_GOV_PLANNER=bspline exercises the flat B-spline
# one; unset keeps the whole-body planner's own default (bspline).
_PLANNER = os.environ.get("WB_GOV_PLANNER", "").strip()
GOV_ARGS = ([f"planner:={_PLANNER}"] if _PLANNER else [])


HOME = np.array([0.0, np.deg2rad(40.0), np.deg2rad(40.0), 0.0])
# DECOY (2026-09-04): the capture must come from the ENCODERS, never from the
# arm controller's smoothed reference. The rig therefore publishes the two at
# DIFFERENT poses -- 2.3 deg apart on joint 2, deliberately inside the node's
# wb_gate_arm_rad (0.05 rad) so it is a realistic disagreement and not an
# impossible one. If anyone re-introduces the smoothed preference, stage [3]'s
# q_d assertion reads 42.3 deg and fails.
SMOOTH_DECOY = HOME + np.array([0.0, np.deg2rad(2.3), 0.0, 0.0])


class Rig(Node):
    def __init__(self):
        super().__init__("gov_test_rig", namespace=NS)
        latched = QoSProfile(depth=1,
                             reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode",
            latched)
        self.odom_pub = self.create_publisher(
            Odometry, "state_estimator/local_position/odom", 10)
        # The whole-body planner now takes ATTITUDE from PX4, as the law does, so the
        # rig must feed it: NED q (w,x,y,z) = (0.70711, 0, 0, 0.70711) is
        # exactly identity in ENU/FLU, i.e. the actual yaw = 0 this rig means.
        self.att_pub = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.smooth_pub = self.create_publisher(
            JointState,
            "fsc_open_manipulator/external_torque_controller/"
            "smoothed_reference_joint_trajectory", 10)
        # The arm's MEASURED state -- what the capture must use.
        self.js_pub = self.create_publisher(
            JointState, "fsc_open_manipulator/joint_states", 10)
        self.gs_pub = self.create_publisher(
            PositionControllerReference,
            "fsc_autopilot_ros2/position_controller/reference", 10)
        self.ee_pub = self.create_publisher(
            PoseStamped, "whole_body_planner/ee_target", 10)

        self.refs = []
        self.status = None
        self.base_anchor = None
        self.arm_syncs = []
        self.create_subscription(
            WholeBodyReference,
            "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            lambda m: self.refs.append((time.monotonic(), m)), 50)
        self.create_subscription(
            String, "whole_body_planner/status",
            lambda m: setattr(self, "status", m.data), latched)
        self.create_subscription(
            PoseStamped, "whole_body_planner/pending_base",
            lambda m: setattr(self, "base_anchor", m), latched)
        self.create_subscription(
            JointTrajectory,
            "fsc_open_manipulator/external_torque_controller/"
            "reference_joint_trajectory",
            lambda m: self.arm_syncs.append(list(m.points[0].positions)), 50)

        self.send_cli = self.create_client(Trigger, "whole_body_planner/send")
        self.create_timer(0.02, self._feed)

    def _feed(self):
        od = Odometry()
        od.pose.pose.position.x = 0.0
        od.pose.pose.position.y = 0.0
        od.pose.pose.position.z = 1.2
        od.pose.pose.orientation.w = 1.0  # ignored by the whole-body planner now
        _a = VehicleAttitude()
        _a.q = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
        self.att_pub.publish(_a)
        self.odom_pub.publish(od)
        js = JointState()
        js.name = ["joint1", "joint2", "joint3", "joint4"]
        js.position = list(HOME)
        js.velocity = [0.0] * 4
        self.js_pub.publish(js)
        sm = JointState()
        sm.name = ["joint1", "joint2", "joint3", "joint4"]
        sm.position = list(SMOOTH_DECOY)     # deliberately NOT the encoders
        sm.velocity = [0.0] * 4
        self.smooth_pub.publish(sm)


def wait_for(cond, timeout, what):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


def main():
    rclpy.init()
    rig = Rig()
    spin = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    spin.start()

    gov = subprocess.Popen(
        GOV_CMD + GOV_ARGS,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    try:
        wait_for(lambda: rig.status is not None, 15, "planner status")
        print(f"[1] whole-body planner up, status={rig.status}")
        assert rig.status == "IDLE"

        # SAFETY: must be silent
        rig.mode_pub.publish(String(data="SAFETY"))
        time.sleep(1.0)
        rig.refs.clear()
        time.sleep(1.0)
        assert len(rig.refs) == 0, "whole-body planner streamed in SAFETY"
        print("[2] silent in SAFETY OK")

        # DIRECT: hold stream
        rig.mode_pub.publish(String(data="DIRECT"))
        wait_for(lambda: len(rig.refs) > 20, 10, "hold stream")
        wait_for(lambda: rig.status == "HOLD", 5, "HOLD status")
        m = rig.refs[-1][1]
        # rate ~100 Hz
        t_first, t_last = rig.refs[0][0], rig.refs[-1][0]
        rate = (len(rig.refs) - 1) / max(1e-9, t_last - t_first)
        # hold checks: actual yaw 0 -> b1_d azimuth = -90 deg
        az_b1 = math.degrees(math.atan2(m.b1_d.y, m.b1_d.x))
        print(f"[3] hold stream {rate:.0f} Hz; x_cd=({m.x_cd.x:.4f},"
              f"{m.x_cd.y:.4f},{m.x_cd.z:.4f}) r_ed=({m.r_ed.x:.4f},"
              f"{m.r_ed.y:.4f},{m.r_ed.z:.4f}) b1_d az={az_b1:.1f} deg "
              f"q_d={[round(math.degrees(v),1) for v in m.q_d]}")
        assert 60 < rate < 140
        assert abs(az_b1 - (-90.0)) < 1e-6
        assert [round(math.degrees(v), 1) for v in m.q_d] == [0.0, 40.0,
                                                              40.0, 0.0]
        assert abs(m.x_cd_dot.x) < 1e-12 and abs(m.r_ed_dot.z) < 1e-12
        hold_ee = np.array([m.r_ed.x, m.r_ed.y, m.r_ed.z])
        hold_xc = np.array([m.x_cd.x, m.x_cd.y, m.x_cd.z])

        # drone GS target -> PENDING -> auto ride-along plan -> PLANNED
        gs = PositionControllerReference()
        gs.position.x, gs.position.y, gs.position.z = 0.5, 0.2, 1.4
        gs.yaw = 15.0
        gs.yaw_unit = PositionControllerReference.DEGREES
        rig.gs_pub.publish(gs)
        wait_for(lambda: rig.status and rig.status.startswith("PLANNED"), 30,
                 "ride-along PLANNED")
        print(f"[4] {rig.status} (ride-along); base anchor "
              f"{rig.base_anchor.header.frame_id} "
              f"({rig.base_anchor.pose.position.x:.2f},"
              f"{rig.base_anchor.pose.position.y:.2f},"
              f"{rig.base_anchor.pose.position.z:.2f})")
        assert rig.base_anchor.header.frame_id == "pending"

        # reference must STILL be the hold while planned (not executing)
        m = rig.refs[-1][1]
        assert abs(m.r_ed.x - hold_ee[0]) < 1e-9, "hold moved without Send"

        # EE target (inertial), derived REACHABLE from the pending base:
        # take the hold's base-relative EE offset, re-anchor it at the
        # pending base/yaw (the ride-along pose), then perturb a few cm and
        # +15 deg of heading -- well inside q1/q4 authority.
        phi0, phi1 = math.radians(-90.0), math.radians(15.0 - 90.0)
        x_b0 = np.array([0.0, 0.0, 1.2])
        x_b1 = np.array([0.5, 0.2, 1.4])
        def rz(a):
            return np.array([[math.cos(a), -math.sin(a), 0],
                             [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        r_rel = rz(phi0).T @ (hold_ee - x_b0)
        ride_ee = x_b1 + rz(phi1) @ r_rel
        # the grasp point (0.108 m further out), the same downward step
        # STALE FIXTURE, fixed 2026-09-04: (0.04, 0.03, -0.06) was authored
        # when the model's EE was the WRIST. Since GRIPPER_OFF_WRIST moved
        # r_e to the grasp point (0.108 m further out) the same step needs
        # q2 = 52.4 deg against its +50 hard stop, so this assert had been
        # failing on BASELINE too -- it was a stale test, not a regression.
        # Re-solved offline against transition_planner.ik_world: this one
        # gives q = [2.9, 43.7, 23.3, 28.7] deg, 6.3 deg inside the worst
        # limit, and still moves all three axes plus the 15 deg heading.
        tgt = ride_ee + np.array([0.02, 0.02, -0.03])
        ee = PoseStamped()
        ee.pose.position.x, ee.pose.position.y, ee.pose.position.z = (
            float(tgt[0]), float(tgt[1]), float(tgt[2]))
        # On the wire the heading is the ACTUAL world yaw, as the arm GS
        # publishes it (the whole-body planner converts to the model azimuth itself);
        # +15 deg of heading beyond the base's: model phi1 + 15 -> actual +90.
        az = phi1 + 0.5 * math.pi + math.radians(15.0)
        ee.pose.orientation.z = math.sin(0.5 * az)
        ee.pose.orientation.w = math.cos(0.5 * az)
        rig.status = None
        rig.ee_pub.publish(ee)
        wait_for(lambda: rig.status and (rig.status.startswith("PLANNED")
                                         or "INFEASIBLE" in rig.status), 30,
                 "EE-target plan verdict")
        assert rig.status.startswith("PLANNED"), rig.status
        print(f"[5] {rig.status} (EE target {np.round(tgt,3).tolist()})")

        # Send -> EXECUTING -> completion
        wait_for(lambda: rig.send_cli.service_is_ready(), 5, "send service")
        rig.arm_syncs.clear()
        fut = rig.send_cli.call_async(Trigger.Request())
        wait_for(lambda: fut.done(), 5, "send response")
        assert fut.result().success, fut.result().message
        print(f"[6] send: {fut.result().message}")
        wait_for(lambda: rig.status and rig.status.startswith("EXECUTING"), 5,
                 "EXECUTING")
        n0 = len(rig.refs)
        wait_for(lambda: rig.status == "HOLD", 60, "completion HOLD")
        ms = [r[1] for r in rig.refs[n0:]]
        print(f"[7] executed: {len(ms)} samples, arm syncs "
              f"{len(rig.arm_syncs)}")
        assert len(rig.arm_syncs) > 5, "no arm reference sync during exec"

        # continuity of the streamed EE reference across the whole execution
        ee_path = np.array([[m.r_ed.x, m.r_ed.y, m.r_ed.z] for m in ms])
        steps = np.linalg.norm(np.diff(ee_path, axis=0), axis=1)
        print(f"    max inter-sample EE step {steps.max()*1000:.2f} mm; "
            f"start {ee_path[0]} end {ee_path[-1]}")
        assert steps.max() < 0.02, "EE reference jumped"
        assert np.linalg.norm(ee_path[0] - hold_ee) < 5e-3, "start != hold"
        want_end = np.array([ee.pose.position.x, ee.pose.position.y,
                             ee.pose.position.z])
        assert np.linalg.norm(ee_path[-1] - want_end) < 5e-3, "end != target"

        # new hold: base anchor back to 'hold' at the pending base
        wait_for(lambda: rig.base_anchor.header.frame_id == "hold", 5,
                 "hold anchor")
        bp = rig.base_anchor.pose.position
        assert abs(bp.x - 0.5) < 1e-6 and abs(bp.z - 1.4) < 1e-6
        m = rig.refs[-1][1]
        print(f"[8] new hold: base ({bp.x:.2f},{bp.y:.2f},{bp.z:.2f}), "
              f"q_d={[round(math.degrees(v),1) for v in m.q_d]}, "
              f"EE=({m.r_ed.x:.3f},{m.r_ed.y:.3f},{m.r_ed.z:.3f})")
        assert np.linalg.norm(
            np.array([m.r_ed.x, m.r_ed.y, m.r_ed.z]) - want_end) < 1e-6

        # SAFETY: silence again
        rig.mode_pub.publish(String(data="SAFETY"))
        time.sleep(0.5)
        rig.refs.clear()
        time.sleep(1.0)
        assert len(rig.refs) == 0, "still streaming after SAFETY"
        wait_for(lambda: rig.status == "IDLE", 5, "IDLE")
        print("[9] SAFETY revert: silent, IDLE")

        print("=== ALL PLANNER LOOPBACK CHECKS PASSED ===")
    finally:
        import signal
        os.killpg(os.getpgid(gov.pid), signal.SIGINT)
        try:
            out, _ = gov.communicate(timeout=10)
            print("---- whole-body planner log tail ----")
            print("\n".join(out.splitlines()[-12:]))
        except subprocess.TimeoutExpired:
            gov.kill()
        rig.destroy_node()
        rclpy.try_shutdown()
        spin.join(timeout=5.0)


if __name__ == "__main__":
    main()
