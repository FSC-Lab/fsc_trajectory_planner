#!/usr/bin/env python3
"""Check the planned-trajectory visualisation feed (2026-09-04).

The Isaac side draws BLUE = drone reference path + heading arrow, RED =
end-effector path + heading arrow, from two plain Float64MultiArray topics the
whole-body planner publishes in the WORLD frame:

    whole_body_planner/viz_path   latched, the whole planned transition,
                                   12 doubles per sample
    whole_body_planner/viz_pose   ~20 Hz, the current reference sample

Everything the drawing depends on is checked here, because the drawing itself
cannot be: it lives in Isaac Sim behind a GPU. What this covers is the part
that can silently be wrong — layout, frame, unit headings, the endpoints
matching the plan, and both topics CLEARING when there is nothing to draw (a
stale curve left on screen after an abort is worse than no curve).

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

from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleAttitude
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger
from fsc_autopilot_ros2_msgs.msg import PositionControllerReference

NS = "/uav_viz"
GOV_CMD = ["ros2", "launch", "fsc_trajectory_planner",
           "whole_body_trajectory_planner_launch.py",
           f"uav_prefix:={NS.lstrip('/')}"]
_PLANNER = os.environ.get("WB_GOV_PLANNER", "").strip()
GOV_ARGS = ([f"planner:={_PLANNER}"] if _PLANNER else [])

START = np.array([0.0, 0.698132, 0.698132, 0.0])     # folded home
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Rig(Node):
    def __init__(self):
        super().__init__("viz_rig", namespace=NS)
        self.mode = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode",
            LATCHED)
        self.odom = self.create_publisher(
            Odometry, "state_estimator/local_position/odom", 10)
        self.att = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.js = self.create_publisher(
            JointState, "fsc_open_manipulator/joint_states", 10)
        self.sm = self.create_publisher(
            JointState, "fsc_open_manipulator/external_torque_controller/"
                        "smoothed_reference_joint_trajectory", 10)
        self.gs = self.create_publisher(
            PositionControllerReference,
            "fsc_autopilot_ros2/position_controller/reference", 10)
        self.status = None
        self.path = None
        self.pose = None
        self.pose_stamps = []
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data),
                                 LATCHED)
        self.create_subscription(Float64MultiArray,
                                 "whole_body_planner/viz_path",
                                 self._on_path, LATCHED)
        self.create_subscription(Float64MultiArray,
                                 "whole_body_planner/viz_pose",
                                 self._on_pose, 10)
        self.send_cli = self.create_client(Trigger, "whole_body_planner/send")
        self.create_timer(0.02, self.feed)

    def _on_path(self, m):
        self.path = np.asarray(m.data, float)

    def _on_pose(self, m):
        self.pose = np.asarray(m.data, float)
        self.pose_stamps.append(time.monotonic())

    def feed(self):
        o = Odometry()
        o.pose.pose.position.z = 1.2
        o.pose.pose.orientation.w = 1.0
        self.odom.publish(o)
        a = VehicleAttitude()
        a.q = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
        self.att.publish(a)
        j = JointState()
        j.name = ["joint1", "joint2", "joint3", "joint4"]
        j.position = list(START)
        j.velocity = [0.0] * 4
        self.js.publish(j)
        self.sm.publish(j)


def wait(cond, timeout, what, repeat=None):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        if repeat is not None:
            repeat()
        time.sleep(0.05)
    raise AssertionError("timeout: " + what)


def check_rows(d, what):
    assert d is not None, f"{what}: never published"
    assert d.size % 12 == 0 and d.size > 0, f"{what}: size {d.size} not 12*N"
    r = d.reshape(-1, 12)
    assert np.all(np.isfinite(r)), f"{what}: non-finite entries"
    for name, o in (("b1_d", 3), ("b1_de", 9)):
        n = np.linalg.norm(r[:, o:o + 3], axis=1)
        assert np.allclose(n, 1.0, atol=1e-6), \
            f"{what}: {name} not unit (min {n.min():.6f}, max {n.max():.6f})"
    return r


def main():
    rclpy.init()
    rig = Rig()
    threading.Thread(target=rclpy.spin, args=(rig,), daemon=True).start()
    gov = subprocess.Popen(GOV_CMD + GOV_ARGS, start_new_session=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
    try:
        wait(lambda: rig.status is not None, 20, "planner up")
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")

        # nothing planned yet -> the curve topic must be EMPTY, not absent
        wait(lambda: rig.path is not None, 10, "viz_path latched")
        assert rig.path.size == 0, f"path not empty at HOLD ({rig.path.size})"
        print("[1] HOLD: viz_path empty (nothing to draw)")

        # the arrows stream while holding, so they are visible before any plan
        wait(lambda: rig.pose is not None and rig.pose.size == 12, 10,
             "viz_pose while holding")
        p0 = check_rows(rig.pose, "viz_pose@HOLD")[0]
        print(f"[2] HOLD arrows: drone {np.round(p0[0:3], 3).tolist()} "
              f"heading {np.round(p0[3:6], 3).tolist()} | EE "
              f"{np.round(p0[6:9], 3).tolist()} heading "
              f"{np.round(p0[9:12], 3).tolist()}")

        # THE ARROWS MUST POINT WHERE THE VEHICLE POINTS (regression for the
        # 2026-09-04 report "the heading directions are wrong, it should be
        # forward"). The law's own b1_d/b1_de are MODEL-frame x-axes and the
        # model frame is the actual one yawed by -90, so publishing them raw
        # drew the drone's heading 90 deg off its nose: [0,-1,0] with this rig
        # at yaw 0, while the arm reaches along +x.
        #
        # Tied to the ARM's measured direction, not to a hard-coded [1,0,0]:
        # that is the invariant that actually failed, and a constant would pass
        # for a vehicle whose arm had moved.
        reach = p0[6:9] - p0[0:3]                      # drone -> EE, world
        reach_h = reach[:2] / np.linalg.norm(reach[:2])
        nose, claw = p0[3:6], p0[9:12]
        assert abs(np.linalg.norm(nose) - 1.0) < 1e-9, "nose not unit"
        assert nose[2] == 0.0, "nose must be horizontal"
        assert float(nose[:2] @ reach_h) > 0.99, (
            f"nose {np.round(nose, 3)} is not along the arm's reach "
            f"{np.round(reach_h, 3)} -- 90 deg out is the model-frame bug")
        claw_h = claw[:2] / (np.linalg.norm(claw[:2]) + 1e-12)
        assert float(claw_h @ reach_h) > 0.99, (
            f"claw {np.round(claw, 3)} does not point along the arm")
        assert claw[2] < 0.0, "claw should aim slightly DOWN at the home pose"
        print(f"    both arrows lie along the arm's reach "
              f"{np.round(reach_h, 3).tolist()}; claw tilts "
              f"{np.degrees(np.arcsin(-claw[2])):.1f} deg down")

        gs = PositionControllerReference()
        gs.position.x, gs.position.y, gs.position.z = 0.9, 0.4, 1.45
        gs.yaw, gs.yaw_unit = 20.0, PositionControllerReference.DEGREES
        wait(lambda: rig.status and rig.status.startswith("PLANNED"), 90,
             "PLANNED", repeat=lambda: rig.gs.publish(gs))
        wait(lambda: rig.path is not None and rig.path.size > 0, 20,
             "viz_path after PLANNED")
        path = check_rows(rig.path, "viz_path")
        assert len(path) >= 50, f"only {len(path)} path samples"
        # It must START where the vehicle is holding: a curve that begins
        # somewhere else is the frame being wrong, which is the whole risk in
        # this feed. 1 mm, NOT machine precision — straight_line fits its CoM
        # with an unconstrained polynomial and lands ~1e-5 m off its own
        # endpoint (bspline is exact there). A frame error is metres or a
        # swapped axis, so this still catches every failure worth catching.
        d_com = float(np.linalg.norm(path[0, 0:3] - p0[0:3]))
        d_ee = float(np.linalg.norm(path[0, 6:9] - p0[6:9]))
        assert d_com < 1e-3, f"path starts {d_com:.2e} m from the hold"
        assert d_ee < 1e-3, f"EE path starts {d_ee:.2e} m from the hold"
        print(f"    curve starts on the hold: drone {d_com:.1e} m, "
              f"EE {d_ee:.1e} m")
        span_com = np.linalg.norm(path[-1, 0:3] - path[0, 0:3])
        span_ee = np.linalg.norm(path[-1, 6:9] - path[0, 6:9])
        print(f"[3] PLANNED: {len(path)} samples, drone path spans "
              f"{span_com:.3f} m, EE path {span_ee:.3f} m")
        assert span_com > 0.5, "drone path did not move toward the target"

        # 20 Hz arrows (100 Hz stream, decimated by 5)
        rig.pose_stamps.clear()
        time.sleep(2.0)
        st = rig.pose_stamps
        rate = (len(st) - 1) / (st[-1] - st[0]) if len(st) > 2 else 0.0
        print(f"[4] viz_pose rate {rate:.1f} Hz ({len(st)} samples in 2 s)")
        assert 12.0 < rate < 30.0, f"arrow rate {rate:.1f} Hz off 20 Hz"

        rig.send_cli.wait_for_service(timeout_sec=10)
        rig.send_cli.call_async(Trigger.Request())
        wait(lambda: rig.status and rig.status.startswith("EXECUTING"), 10,
             "EXECUTING")
        time.sleep(2.0)
        mid = check_rows(rig.pose, "viz_pose@EXECUTING")[0]
        moved = np.linalg.norm(mid[0:3] - p0[0:3])
        print(f"[5] EXECUTING: arrows have moved {moved:.3f} m along the plan")
        assert moved > 0.01, "arrows did not follow the executing plan"
        # and they must be ON the planned curve, not drifting off it
        off = np.min(np.linalg.norm(path[:, 0:3] - mid[0:3], axis=1))
        assert off < 0.02, f"arrow sits {off:.3f} m off the drawn curve"
        print(f"[6] arrows track the drawn curve to {off * 1e3:.1f} mm")

        # SAFETY must clear BOTH, or the last plan stays on screen all flight
        rig.mode.publish(String(data="SAFETY"))
        wait(lambda: rig.path is not None and rig.path.size == 0, 10,
             "viz_path cleared on SAFETY")
        wait(lambda: rig.pose is not None and rig.pose.size == 0, 10,
             "viz_pose cleared on SAFETY")
        print("[7] SAFETY: both topics cleared")
        print("=== TRAJECTORY VISUALISATION FEED — PASS ===")
    finally:
        os.killpg(os.getpgid(gov.pid), signal.SIGINT)
        try:
            gov.wait(timeout=5)
        except subprocess.TimeoutExpired:
            gov.kill()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
