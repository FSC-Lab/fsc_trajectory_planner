#!/usr/bin/env python3
"""Loopback test of push_pull_box_point (2026-10-16) against the built
whole_body_trajectory_planner node (no sim, no PX4). Run it on a private ROS
domain, never beside a live stack:

    ROS_DOMAIN_ID=78 python3 test_push_pull_box_point_loopback.py

The box capture as a PARAMETER, both ways (the arm GS's Box Pushing tab shows
it in editable boxes, as the Pick row of 2026-10-10 does):
  [1] [] until captured; push_pull/info [13] = 0
  [2] capture_box writes [x, y, z, yaw_deg] from the mocap body, and info
      [13..17] says the same
  [3] a set from outside sets the capture itself
  [4] a wrong length or a non-finite value is refused, the capture unchanged
  [5] [] clears the capture
  [6] capture_box again writes the measured pose back
  [7] (2026-10-19) a FROZEN feed -- the mocap processor republishing the last
      pose of a body it lost -- is refused, the capture unchanged
  [8] (2026-10-19) a yaw flipping between two solutions is refused
The yaml is the box-pushing sim yaml (PL_YAML overrides); the mocap body is a
rig-private topic.
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
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from fsc_autopilot_ros2_msgs.msg import Mocap

NS = "/uav_plbox"
BOX_TOPIC = "/plbox_rig/obj_0/mocap"
YAML = os.environ.get("PL_YAML", os.path.expanduser(
    "~/ros2_ws/src/fsc_autopilot_ros2/config/"
    "params_single_aerial_manipulator_whole_body_l1_4d_direct_actuation_t650_sim_box_pushing.yaml"))
GOV_CMD = ["ros2", "run", "fsc_trajectory_planner", "whole_body_trajectory_planner",
           "--ros-args", "-r", f"__ns:={NS}", "--params-file", YAML,
           "-p", f"push_pull_box_topic:={BOX_TOPIC}"]
BOX = np.array([1.275, 0.25, 0.85])
BOX_YAW = 7.0                       # [deg]
INFO_VALID, INFO_BOX = 14, 15   # (six steps since 2026-10-16: 13, 14 before)
PARAM = "push_pull_box_point"


class Rig(Node):
    def __init__(self):
        super().__init__("plbox_rig")
        self.box = self.create_publisher(Mocap, BOX_TOPIC, 10)
        self.info = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float64MultiArray, f"{NS}/whole_body_planner/push_pull/info",
                                 lambda m: setattr(self, "info", list(m.data)), latched)
        self.capture = self.create_client(Trigger, f"{NS}/whole_body_planner/push_pull/capture_box")
        node = f"{NS}/whole_body_trajectory_planner"
        self.setp = self.create_client(SetParameters, f"{node}/set_parameters")
        self.getp = self.create_client(GetParameters, f"{node}/get_parameters")
        self.rng = np.random.default_rng(1)
        self.mode = "live"     # "frozen": one pose repeated; "flip": yaw +180 every other sample
        self.n = 0
        self.frozen_p = BOX + np.array([0.04, -0.03, 0.0])
        self.create_timer(0.01, self.tick)

    def tick(self):
        m = Mocap()
        self.n += 1
        if self.mode == "frozen":
            p = self.frozen_p
        else:
            p = BOX + self.rng.normal(0.0, 3e-4, 3)   # 0.3 mm of mocap jitter
        yaw = BOX_YAW + (180.0 if self.mode == "flip" and self.n % 2 else 0.0)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
        m.pose.orientation.w = math.cos(math.radians(yaw) / 2)
        m.pose.orientation.z = math.sin(math.radians(yaw) / 2)
        self.box.publish(m)


def done(fut, what):
    t0 = time.monotonic()
    while not fut.done():
        time.sleep(0.05)
        assert time.monotonic() - t0 < 10, what
    return fut.result()


def set_param(rig, value):
    v = ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=[float(x) for x in value])
    assert rig.setp.wait_for_service(timeout_sec=10)
    return done(rig.setp.call_async(SetParameters.Request(parameters=[Parameter(name=PARAM, value=v)])),
                "set").results[0]


def get_param(rig):
    assert rig.getp.wait_for_service(timeout_sec=10)
    return np.array(done(rig.getp.call_async(GetParameters.Request(names=[PARAM])), "get")
                    .values[0].double_array_value)


def capture(rig):
    assert rig.capture.wait_for_service(timeout_sec=10)
    return done(rig.capture.call_async(Trigger.Request()), "capture_box")


def wait(cond, timeout, what):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("timeout: " + what)


def info_box(rig):
    i = rig.info
    return None if i is None or i[INFO_VALID] < 0.5 else np.array(i[INFO_BOX:INFO_BOX + 4])


def main():
    log = tempfile.NamedTemporaryFile(prefix="plbox_node_", suffix=".log", delete=False)
    gov = subprocess.Popen(GOV_CMD, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rclpy.init()
    rig = Rig()
    th = threading.Thread(target=rclpy.spin, args=(rig,), daemon=True)
    th.start()
    ok = False
    try:
        wait(lambda: rig.info is not None, 30, "the planner's push_pull/info")
        # [1]
        assert get_param(rig).size == 0, get_param(rig)
        assert info_box(rig) is None
        print("[1] not captured: [] and info [13] = 0")
        # [2]
        time.sleep(1.0)                                # a capture window of samples
        r = capture(rig)
        assert r.success, r.message
        v = get_param(rig)
        assert v.size == 4 and np.linalg.norm(v[:3] - BOX) < 2e-3 and abs(v[3] - BOX_YAW) < 0.1, v
        wait(lambda: info_box(rig) is not None and np.allclose(info_box(rig), v, atol=1e-6), 2, "info = param")
        print(f"[2] Get: {r.message}\n    param {np.round(v, 4).tolist()} = info")
        # [3]
        typed = [1.30, 0.20, 0.86, -3.0]
        res = set_param(rig, typed)
        assert res.successful, res.reason
        wait(lambda: info_box(rig) is not None and np.allclose(info_box(rig), typed, atol=1e-9), 2,
             "info = the typed box")
        print(f"[3] typed {typed}: the capture follows")
        # [4]
        for bad in ([1.0, 2.0, 3.0], [1.0, 2.0, float("nan"), 0.0]):
            res = set_param(rig, bad)
            assert not res.successful, bad
            assert np.allclose(get_param(rig), typed) and np.allclose(info_box(rig), typed)
            print(f"[4] refused {bad}: {res.reason}")
        # [5]
        res = set_param(rig, [])
        assert res.successful, res.reason
        wait(lambda: info_box(rig) is None, 2, "info cleared")
        print("[5] [] clears the capture")
        # [6]
        r = capture(rig)
        assert r.success, r.message
        v = get_param(rig)
        assert v.size == 4 and np.linalg.norm(v[:3] - BOX) < 2e-3, v
        wait(lambda: info_box(rig) is not None and np.allclose(info_box(rig), v, atol=1e-6), 2, "info = param")
        print("[6] Get again writes the measured pose back")
        # [7]
        rig.mode = "frozen"
        time.sleep(1.0)
        r = capture(rig)
        assert not r.success and "frozen" in r.message, r.message
        assert np.allclose(get_param(rig), v), "a refused Get must not move the capture"
        print(f"[7] frozen feed refused: {r.message}")
        # [8]
        rig.mode = "flip"
        time.sleep(1.0)
        r = capture(rig)
        assert not r.success and "flipping" in r.message, r.message
        assert np.allclose(get_param(rig), v), "a refused Get must not move the capture"
        print(f"[8] flipping yaw refused: {r.message}")
        ok = True
        print("PASS")
    finally:
        os.killpg(gov.pid, signal.SIGINT)
        try:
            gov.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(gov.pid, signal.SIGKILL)
        # stop the spin thread BEFORE destroying the node it spins (the other
        # order aborted the interpreter after PASS: "terminate called ...")
        rclpy.shutdown()
        th.join(timeout=5)
        if not ok:
            print(f"(node log: {log.name})")


if __name__ == "__main__":
    main()
