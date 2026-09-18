#!/usr/bin/env python3
"""Check the whole-body planner's Go Home: from an arm pose away from home, the service
must plan a compatible transition whose goal joints ARE the home pose."""
import os, signal, subprocess, sys, threading, time
import numpy as np, rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleAttitude
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import Trigger

NS = "/uav_gh"
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

HOME = np.array([0.0, 0.698132, 0.698132, 0.0])
# start the arm somewhere else entirely
START = np.array([0.30, 0.20, 0.55, 1.20])


class Rig(Node):
    def __init__(self):
        super().__init__("gh_rig", namespace=NS)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode = self.create_publisher(
            String, "fsc_autopilot_ros2/whole_body_direct_actuation/mode", latched)
        self.odom = self.create_publisher(Odometry, "state_estimator/local_position/odom", 10)
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
        self.status = None; self.jt = None
        self.create_subscription(String, "whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data), latched)
        self.create_subscription(Float64MultiArray, "whole_body_planner/target_joints",
                                 lambda m: setattr(self, "jt", list(m.data)), latched)
        self.home_cli = self.create_client(Trigger, "whole_body_planner/go_home")
        self.create_timer(0.02, self.feed)

    def feed(self):
        o = Odometry(); o.pose.pose.position.z = 1.2
        o.pose.pose.orientation.w = 1.0; self.odom.publish(o)
        _a = VehicleAttitude()
        _a.q = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]
        self.att.publish(_a)
        j = JointState(); j.name = ["joint1", "joint2", "joint3", "joint4"]
        j.position = list(START); j.velocity = [0.0]*4
        self.js.publish(j)   # the capture reads THIS (encoders)
        self.sm.publish(j)


def wait(c, t, what):
    t0 = time.monotonic()
    while time.monotonic()-t0 < t:
        if c(): return
        time.sleep(0.05)
    raise AssertionError("timeout: "+what)


def main():
    rclpy.init(); rig = Rig()
    threading.Thread(target=rclpy.spin, args=(rig,), daemon=True).start()
    gov = subprocess.Popen(GOV_CMD + GOV_ARGS, start_new_session=True,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        wait(lambda: rig.status is not None, 20, "planner up")
        rig.mode.publish(String(data="DIRECT"))
        wait(lambda: rig.status == "HOLD", 15, "HOLD")
        print(f"[1] holding at arm pose {np.round(np.degrees(START),1).tolist()} deg")
        # target_joints is published from its own callback, so HOLD can be
        # observed before its first message lands -- wait for it rather than
        # asserting on None. (Seen once, 2026-09-04, with the faster B-spline
        # planner: a race in this test, not in the whole-body planner.)
        wait(lambda: rig.jt is not None, 5, "target_joints")
        assert np.allclose(rig.jt, START, atol=1e-6), f"hold joints {rig.jt}"

        wait(lambda: rig.home_cli.service_is_ready(), 10, "go_home service")
        rig.status = None
        fut = rig.home_cli.call_async(Trigger.Request())
        wait(lambda: fut.done(), 10, "go_home response")
        assert fut.result().success, fut.result().message
        print(f"[2] go_home: {fut.result().message}")

        wait(lambda: rig.status and rig.status.startswith("PLANNED"), 60, "PLANNED")
        print(f"[3] {rig.status}")
        got = np.array(rig.jt)
        print(f"[4] goal joints published: {np.round(np.degrees(got),2).tolist()} deg"
              f"   (home = {np.round(np.degrees(HOME),2).tolist()})")
        assert np.allclose(got, HOME, atol=1e-6), "goal is not the home pose"
        print("=== GO HOME PLANS TO THE HOME POSE — PASS ===")
    finally:
        os.killpg(os.getpgid(gov.pid), signal.SIGINT)
        try: gov.communicate(timeout=5)
        except subprocess.TimeoutExpired: gov.kill()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
