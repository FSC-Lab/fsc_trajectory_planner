#!/usr/bin/env python3
"""Loopback test of the planner's PS4 TELEOPERATION (no sim, no PX4).

    ROS_DOMAIN_ID=77 python3 test_teleop_loopback.py

Run it on a SPARE domain beside a live stack: the rig publishes a DIRECT mode
and pad messages of its own. It launches the installed planner under
/uav_tele_test, fakes the vehicle at rest in DIRECT, and drives a synthetic
DualShock 4 on rc/input through every channel: D-pad (forward/left), triangle
(up), square (yaw left), the sticks (grasp point vs airframe, wrist roll), PS
(arm home), a dropped pad, disengage -> HOLD, SAFETY -> silence. Asserts the
direction of every move, the stream's continuity and the hand-back.
"""
import math
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleAttitude
from sensor_msgs.msg import JointState, Joy
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from fsc_autopilot_ros2_msgs.msg import WholeBodyReference

NS = "/uav_tele_test"
CMD = ["ros2", "launch", "fsc_trajectory_planner", "whole_body_trajectory_planner_launch.py",
       f"uav_prefix:={NS.lstrip('/')}"]
HOME = np.array([0.0, np.deg2rad(40.0), np.deg2rad(40.0), 0.0])
# where PS folds the arm: the PAD's home (teleop_home_pose_deg), inside the
# 20 % inner joint box -- NOT the stowed flight home above
PAD_HOME = np.array([0.0, np.deg2rad(30.0), np.deg2rad(30.0), 0.0])

# DualShock 4 on joy_node: 8 axes (0/1 left stick, 2 L2, 3/4 right stick,
# 5 R2, 6/7 D-pad), 13 buttons (0 cross, 1 circle, 2 triangle, 3 square, 10 PS).
def pad(**kw):
    j = Joy()
    j.axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    j.buttons = [0] * 13
    for k, v in kw.items():
        if k.startswith("a"):
            j.axes[int(k[1:])] = float(v)
        else:
            j.buttons[int(k[1:])] = int(v)
    return j


class Rig(Node):
    def __init__(self):
        super().__init__("teleop_test_rig", namespace=NS)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode", latched)
        self.odom_pub = self.create_publisher(Odometry, "state_estimator/local_position/odom", 10)
        self.att_pub = self.create_publisher(
            VehicleAttitude, "fmu/out/vehicle_attitude",
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT,
                       durability=DurabilityPolicy.VOLATILE))
        self.js_pub = self.create_publisher(JointState, "fsc_open_manipulator/joint_states", 10)
        self.joy_pub = self.create_publisher(Joy, "rc/input", 10)
        self.refs = []
        self.status = None
        self.note = None
        self.state = []
        self.skel = []
        self.joy = None           # the pad state being streamed (None = silent)
        # base - CoM at the hold: once set, the fake vehicle FOLLOWS the
        # streamed CoM (perfect tracking), so the leash stays out of the way
        self.offset = None
        self.create_subscription(
            WholeBodyReference, "fsc_autopilot_ros2/whole_body_direct_actuation/reference",
            lambda m: self.refs.append((time.monotonic(), m)), 200)
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data), latched)
        self.create_subscription(String, "whole_body_planner/teleop/note",
                                 lambda m: setattr(self, "note", m.data), latched)
        self.create_subscription(Float64MultiArray, "whole_body_planner/teleop/state",
                                 lambda m: setattr(self, "state", list(m.data)), 10)
        self.create_subscription(Float64MultiArray, "whole_body_planner/current_skeleton",
                                 lambda m: setattr(self, "skel", list(m.data)), 10)
        self.engage = self.create_client(SetBool, "whole_body_planner/teleop/engage")
        self.home_cli = self.create_client(Trigger, "whole_body_planner/teleop/arm_home")
        self.create_timer(0.02, self._feed)
        self.create_timer(0.05, self._pad)

    def _feed(self):
        od = Odometry()
        od.pose.pose.position.z = 1.2
        if self.offset is not None and self.refs:
            p = v(self.refs[-1][1].x_cd) - self.offset
            od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = map(float, p)
        od.pose.pose.orientation.w = 1.0
        a = VehicleAttitude()
        a.q = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]   # actual yaw 0
        self.att_pub.publish(a)
        self.odom_pub.publish(od)
        js = JointState()
        js.name = ["joint1", "joint2", "joint3", "joint4"]
        js.position = list(HOME)
        js.velocity = [0.0] * 4
        self.js_pub.publish(js)

    def _pad(self):
        if self.joy is not None:
            self.joy_pub.publish(self.joy)


def wait_for(cond, timeout, what):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


def call(cli, req):
    assert cli.wait_for_service(timeout_sec=5.0)
    f = cli.call_async(req)
    t0 = time.monotonic()
    while not f.done() and time.monotonic() - t0 < 5.0:
        time.sleep(0.02)
    return f.result()


def v(x):
    return np.array([x.x, x.y, x.z])


def last(rig):
    return rig.refs[-1][1]


def hold_for(rig, seconds, **buttons):
    rig.joy = pad(**buttons)
    time.sleep(seconds)
    rig.joy = pad()


def settle(rig, seconds=2.5):
    rig.joy = pad()
    time.sleep(seconds)


def main():
    rclpy.init()
    rig = Rig()
    threading.Thread(target=rclpy.spin, args=(rig,), daemon=True).start()
    proc = subprocess.Popen(CMD, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    ok = False
    try:
        wait_for(lambda: rig.status is not None, 15, "planner status")
        rig.mode_pub.publish(String(data="DIRECT"))
        wait_for(lambda: rig.status == "HOLD" and len(rig.refs) > 20, 10, "HOLD stream")
        h = last(rig)
        xc0, re0, q0 = v(h.x_cd), v(h.r_ed), np.array(h.q_d)
        rig.offset = xc0 - np.array([0.0, 0.0, 1.2])
        print(f"[1] HOLD: x_cd {np.round(xc0, 4)}  r_ed {np.round(re0, 4)}")

        # Engage with the pad HELD: nothing may move until it is centred.
        rig.joy = pad(a7=1.0)
        r = call(rig.engage, SetBool.Request(data=True))
        assert r.success, r.message
        wait_for(lambda: rig.status == "TELEOP", 5, "TELEOP")
        time.sleep(1.0)
        assert np.linalg.norm(v(last(rig).x_cd) - xc0) < 1e-9, "moved before the pad was centred"
        print(f"[2] engaged ({r.message}); held D-pad ignored until neutral OK")
        settle(rig, 0.5)
        assert rig.state and rig.state[34] == 1.0, "inputs never went live"

        # D-pad UP = forward. Actual yaw 0: forward is world +x.
        rig.refs.clear()
        hold_for(rig, 2.0, a7=1.0)
        settle(rig)
        d = v(last(rig).x_cd) - xc0
        steps = [np.linalg.norm(v(b.x_cd) - v(a.x_cd)) for (_, a), (_, b) in zip(rig.refs, rig.refs[1:])]
        print(f"[3] D-pad up 2 s: x_cd moved {np.round(d, 3)} m, max step {max(steps)*1e3:.2f} mm")
        assert d[0] > 0.2 and abs(d[1]) < 1e-3 and abs(d[2]) < 1e-3
        assert max(steps) < 0.005
        assert np.allclose(np.array(last(rig).q_d), q0, atol=1e-9), "arm moved on a platform command"

        # D-pad LEFT = left = world +y at yaw 0.
        x1 = v(last(rig).x_cd)
        hold_for(rig, 1.0, a6=1.0)
        settle(rig)
        d = v(last(rig).x_cd) - x1
        print(f"[4] D-pad left 1 s: {np.round(d, 3)} m")
        assert d[1] > 0.1 and abs(d[0]) < 1e-3

        # TRIANGLE = up, CROSS = down.
        x1 = v(last(rig).x_cd)
        hold_for(rig, 1.0, b2=1)
        settle(rig)
        d = v(last(rig).x_cd) - x1
        print(f"[5] triangle 1 s: dz {d[2]:+.3f} m")
        assert d[2] > 0.08
        x1 = v(last(rig).x_cd)
        hold_for(rig, 0.5, b0=1)
        settle(rig)
        assert v(last(rig).x_cd)[2] < x1[2] - 0.04
        print("[5b] cross: down OK")

        # SQUARE = yaw LEFT (CCW): the model heading azimuth grows.
        az = lambda m: math.atan2(m.b1_d.y, m.b1_d.x)
        a0 = az(last(rig))
        hold_for(rig, 1.0, b3=1)
        settle(rig)
        da = math.degrees(az(last(rig)) - a0)
        print(f"[6] square 1 s: heading {da:+.1f} deg")
        assert da > 10.0
        a0 = az(last(rig))
        hold_for(rig, 1.0, b1=1)
        settle(rig)
        assert math.degrees(az(last(rig)) - a0) < -10.0
        print("[6b] circle: yaw right OK")

        # LEFT STICK UP (axes[1] = +1): the grasp point FORWARD of the airframe,
        # the CoM untouched.
        m0 = last(rig)
        x1, q1 = v(m0.x_cd), np.array(m0.q_d)
        s0 = np.array(rig.state[19:22])
        hold_for(rig, 1.0, a4=-1.0)       # right stick DOWN: EE down first (room there)
        settle(rig)
        s1 = np.array(rig.state[19:22])
        print(f"[7] right stick down 1 s: grasp point vs airframe {np.round(s1 - s0, 4)} m (fwd,left,up)")
        assert s1[2] - s0[2] < -0.02 and np.linalg.norm(v(last(rig).x_cd) - x1) < 1e-3
        hold_for(rig, 1.0, a0=1.0)        # left stick LEFT: EE left
        settle(rig)
        s2 = np.array(rig.state[19:22])
        print(f"[8] left stick left 1 s: {np.round(s2 - s1, 4)} m")
        assert s2[1] - s1[1] > 0.01
        q2 = np.array(last(rig).q_d)
        hold_for(rig, 1.0, a3=1.0)        # right stick LEFT: wrist roll +q4
        settle(rig)
        dq4 = math.degrees(last(rig).q_d[3] - q2[3])
        print(f"[9] right stick left 1 s: q4 {dq4:+.1f} deg")
        assert dq4 > 5.0

        # Push the grasp point FORWARD until the arm refuses: a wall, not a jump.
        rig.refs.clear()
        hold_for(rig, 6.0, a1=1.0)
        note = rig.note
        settle(rig)
        qs = np.array([m.q_d for _, m in rig.refs])
        print(f"[10] left stick up 6 s: wall '{note}', max joint step "
              f"{np.degrees(np.abs(np.diff(qs, axis=0)).max()):.3f} deg/sample")
        assert note, "no wall reported"
        assert np.degrees(np.abs(np.diff(qs, axis=0)).max()) < 0.5

        # PS = arm home; the platform holds.
        xh = v(last(rig).x_cd)
        hold_for(rig, 0.3, b10=1)
        wait_for(lambda: rig.state and rig.state[25] == 0.0 and
                 np.allclose(rig.state[15:19], PAD_HOME, atol=1e-9), 30, "arm home")
        settle(rig)
        print(f"[11] PS: arm home {np.round(np.degrees(last(rig).q_d), 2)} deg, CoM held "
              f"{np.linalg.norm(v(last(rig).x_cd) - xh)*1e3:.3f} mm")
        assert np.allclose(np.array(last(rig).q_d), PAD_HOME, atol=1e-4)
        assert np.linalg.norm(v(last(rig).x_cd) - xh) < 1e-9

        # The pad DROPS while a direction is held: motion stops.
        rig.joy = pad(a7=1.0)
        time.sleep(0.5)
        rig.joy = None
        time.sleep(3.0)
        xa = v(last(rig).x_cd)
        time.sleep(1.0)
        drift = np.linalg.norm(v(last(rig).x_cd) - xa)
        print(f"[12] pad silent: drift over 1 s {drift*1e3:.4f} mm")
        assert drift < 1e-4

        # Disengage: settle, then HOLD on the same point (no step).
        rig.joy = pad()
        before = last(rig)
        r = call(rig.engage, SetBool.Request(data=False))
        wait_for(lambda: rig.status == "HOLD", 15, "HOLD after disengage")
        time.sleep(0.5)
        after = last(rig)
        jump = np.linalg.norm(v(after.x_cd) - v(before.x_cd)) + np.linalg.norm(v(after.r_ed) - v(before.r_ed))
        print(f"[13] disengaged ({r.message}) -> HOLD, jump {jump*1e3:.3f} mm")
        assert jump < 1e-3
        assert len(rig.skel) == 15
        print(f"[14] current_skeleton: 5 points, grasp point {np.round(rig.skel[12:], 3)}")

        rig.mode_pub.publish(String(data="SAFETY"))
        time.sleep(1.0)
        rig.refs.clear()
        time.sleep(1.0)
        assert not rig.refs, "streamed in SAFETY"
        print("[15] SAFETY: silent OK")
        ok = True
    finally:
        import os
        import signal
        os.killpg(proc.pid, signal.SIGINT)
        try:
            out, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, _ = proc.communicate()
        if not ok:
            print("---- planner log ----")
            print(out[-4000:])
        rig.destroy_node()
        rclpy.shutdown()
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
