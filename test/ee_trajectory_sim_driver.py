#!/usr/bin/env python3
"""Fly the END-EFFECTOR TRAJECTORY mission on the Command.md 7.15.1 IsaacSim
rig and record it (the ground stations' ROS interface, driven by a script).

    /usr/bin/python3 ee_trajectory_sim_driver.py --shape circle --scale 0.8 --out run.npz

    offboard -> arm -> SAFETY climb to hover_z -> settle -> DIRECT -> settle
      -> select <shape>            (what the arm GS combo box publishes)
      -> wait READY, set time scale = <scale> x s_max   (the GS slider)
      -> go_to_start               (the GS button: compatible transition)
      -> wait HOLD + AT START
      -> start                     (the GS button)
      -> wait for the run to finish (HOLD)
      -> SAFETY -> land by reference -> disarm

Records odometry, the planner's EE reference pose and its measured EE
(current_ee) so the EE tracking error of the run can be scored offline.
Refuses to start against an already-armed vehicle (the rig never disarms in
the air; every run is a clean relaunch).
"""
import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger

from fsc_autopilot_ros2_msgs.msg import PositionControllerReference
from px4_msgs.msg import VehicleStatus

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
DA = "fsc_autopilot_ros2/whole_body_direct_actuation"
EE = "whole_body_planner/ee_trajectory"


class Driver(Node):
    def __init__(self, a):
        super().__init__("ee_trajectory_sim_driver")
        self.a = a
        ns = a.namespace.rstrip("/")
        self.t0 = time.time()
        self.phase = "WAIT"
        self.tp = self.t0
        self.done = False
        self.aborted = False
        self.reason = ""
        self.events = []
        self.odom = None
        self.mode = ""
        self.armed = None
        self.plan_status = ""
        self.ee_status = ""
        self.info = None
        self.start_err = None
        self.ref_pose = None
        self.cur_ee = None
        self.log = []
        self.settle_since = None
        self.ref = np.array([0.0, 0.0, a.hover_z])
        self.run_T = 0.0

        self.create_subscription(Odometry, f"{ns}/state_estimator/local_position/odom", self.on_odom, 10)
        self.create_subscription(String, f"{ns}/{DA}/mode", lambda m: setattr(self, "mode", m.data), LATCHED)
        self.create_subscription(VehicleStatus, f"{ns}/fmu/out/vehicle_status_v1",
                                 lambda m: setattr(self, "armed", m.arming_state == 2), PX4_QOS)
        self.create_subscription(String, f"{ns}/whole_body_planner/status",
                                 lambda m: setattr(self, "plan_status", m.data), LATCHED)
        self.create_subscription(String, f"{ns}/{EE}/status",
                                 lambda m: setattr(self, "ee_status", m.data), LATCHED)
        self.create_subscription(Float64MultiArray, f"{ns}/{EE}/info",
                                 lambda m: setattr(self, "info", list(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, f"{ns}/{EE}/start_error",
                                 lambda m: setattr(self, "start_err", list(m.data)), 10)
        self.create_subscription(PoseStamped, f"{ns}/{EE}/reference_pose", self.on_ref_pose, 10)
        self.create_subscription(PoseStamped, f"{ns}/whole_body_planner/current_ee", self.on_cur_ee, 10)
        self.ref_pub = self.create_publisher(PositionControllerReference,
                                             f"{ns}/fsc_autopilot_ros2/position_controller/reference", 10)
        self.select_pub = self.create_publisher(String, f"{ns}/{EE}/select", 10)
        self.scale_pub = self.create_publisher(Float64, f"{ns}/{EE}/time_scale", 10)
        self.offboard = self.create_client(Trigger, f"{ns}/rc/offboard")
        self.arm = self.create_client(Trigger, f"{ns}/rc/arm")
        self.disarm = self.create_client(Trigger, f"{ns}/rc/disarm")
        self.direct = self.create_client(SetBool, f"{ns}/{DA}/set_direct_mode")
        self.go = self.create_client(Trigger, f"{ns}/{EE}/go_to_start")
        self.start = self.create_client(Trigger, f"{ns}/{EE}/start")
        self.futs = []
        self.create_timer(1.0 / a.rate, self.tick)

    # ---- callbacks
    def on_odom(self, m):
        p, v = m.pose.pose.position, m.twist.twist.linear
        self.odom = np.array([p.x, p.y, p.z, v.x, v.y, v.z])
        self.log.append([self.now(), p.x, p.y, p.z, v.x, v.y, v.z, 1.0 if self.mode == "DIRECT" else 0.0])

    def on_ref_pose(self, m):
        self.ref_pose = np.array([self.now(), m.pose.position.x, m.pose.position.y, m.pose.position.z])

    def on_cur_ee(self, m):
        self.cur_ee = np.array([self.now(), m.pose.position.x, m.pose.position.y, m.pose.position.z])
        if self.ref_pose is not None and self.phase == "RUN":
            self.ee_log.append([self.now(), *self.cur_ee[1:], *self.ref_pose[1:]])

    ee_log = []

    # ---- helpers
    def now(self):
        return time.time() - self.t0

    def ev(self, s):
        line = f"[{self.now():7.2f}s] {s}"
        self.events.append(line)
        print(line, flush=True)

    def goto(self, ph):
        self.phase = ph
        self.tp = time.time()
        self.settle_since = None
        self.ev(f"PHASE {ph}  mode={self.mode} plan={self.plan_status} ee={self.ee_status}")

    def call(self, cli, req, name):
        if not cli.service_is_ready():
            self.ev(f"service {name} not ready")
            return
        fut = cli.call_async(req)
        fut.add_done_callback(lambda f: self.ev(f"service {name} -> {f.result()}"))
        self.futs.append(fut)

    def send_ref(self):
        m = PositionControllerReference()
        m.header.stamp = self.get_clock().now().to_msg()
        m.position.x, m.position.y, m.position.z = map(float, self.ref)
        m.yaw, m.yaw_unit = 0.0, PositionControllerReference.DEGREES
        self.ref_pub.publish(m)

    def settled(self, tol_m=0.08, tol_v=0.10, hold=3.0):
        if self.odom is None:
            return False
        err = np.linalg.norm(self.odom[:3] - self.ref)
        spd = np.linalg.norm(self.odom[3:6])
        if err < tol_m and spd < tol_v:
            if self.settle_since is None:
                self.settle_since = time.time()
            return time.time() - self.settle_since > hold
        self.settle_since = None
        return False

    def abort(self, why):
        self.aborted = True
        self.reason = why
        self.ev("ABORT: " + why)
        self.call(self.direct, SetBool.Request(data=False), "set_direct_mode(false)")
        self.goto("LAND")

    # ---- the state machine
    def tick(self):
        a = self.a
        el = time.time() - self.tp
        if self.mode != "DIRECT":
            self.send_ref()   # SAFETY needs the full-rate stream
        if self.odom is not None and self.mode == "DIRECT":
            tilt_ok = True  # the node's own watchdog covers tilt
        if self.phase == "WAIT":
            if self.odom is not None and self.mode and self.armed is not None:
                if self.armed:
                    self.ev("vehicle already ARMED -- refusing (clean relaunch needed)")
                    self.done = True
                    return
                self.ref = np.array([self.odom[0], self.odom[1], self.odom[2]])
                self.goto("OFFBOARD")
        elif self.phase == "OFFBOARD":
            if el > 1.0:
                self.call(self.offboard, Trigger.Request(), "rc/offboard")
                self.goto("ARM")
        elif self.phase == "ARM":
            if el > 3.0:
                self.call(self.arm, Trigger.Request(), "rc/arm")
                self.goto("CLIMB")
        elif self.phase == "CLIMB":
            if el > 2.0:
                self.ref = np.array([self.odom[0], self.odom[1], a.hover_z])
                self.goto("TAKEOFF")
        elif self.phase == "TAKEOFF":
            if self.settled():
                self.goto("ENTER_DIRECT")
            elif el > 60.0:
                self.abort("takeoff did not settle")
        elif self.phase == "ENTER_DIRECT":
            self.call(self.direct, SetBool.Request(data=True), "set_direct_mode(true)")
            self.goto("CONFIRM")
        elif self.phase == "CONFIRM":
            if self.mode == "DIRECT":
                self.goto("DIRECT_SETTLE")
            elif el > 5.0:
                self.abort("DIRECT was refused")
        elif self.phase == "DIRECT_SETTLE":
            if el > a.direct_settle:
                self.select_pub.publish(String(data=a.shape))
                self.goto("SELECT")
        elif self.phase == "SELECT":
            if self.ee_status.startswith("READY") and self.info:
                s_max = self.info[1]
                s = max(0.05, a.scale * s_max)
                self.ev(f"READY: s_max {s_max:.3f} -> requesting s = {s:.3f}")
                self.scale_pub.publish(Float64(data=s))
                self.ee_status = "PENDING_RESCALE"
                self.goto("RESCALE")
            elif self.ee_status.startswith("INFEASIBLE") or el > 60.0:
                self.abort(f"trajectory not READY: {self.ee_status}")
        elif self.phase == "RESCALE":
            if self.ee_status.startswith("READY") and self.info and abs(self.info[0] - max(0.05, a.scale * self.info[1])) < 1e-3:
                self.run_T = self.info[2]
                self.ev(f"{self.ee_status}  T_total {self.run_T:.1f}s lap {self.info[3]:.1f}s")
                self.call(self.go, Trigger.Request(), "ee_trajectory/go_to_start")
                self.goto("GO_TO_START")
            elif el > 60.0:
                self.abort(f"rescale did not settle: {self.ee_status}")
        elif self.phase == "GO_TO_START":
            if el > 2.0 and self.plan_status == "HOLD" and self.start_err and self.start_err[3] > 0.5:
                if el > 6.0:
                    self.call(self.start, Trigger.Request(), "ee_trajectory/start")
                    self.goto("START")
            elif el > 90.0:
                self.abort(f"never at start: plan={self.plan_status} err={self.start_err}")
        elif self.phase == "START":
            if self.plan_status.startswith("EXECUTING"):
                self.goto("RUN")
            elif el > 10.0:
                self.abort(f"start did not begin: plan={self.plan_status}")
        elif self.phase == "RUN":
            if self.plan_status == "HOLD" and el > 5.0:
                self.ev("run complete")
                self.goto("POST_HOLD")
            elif el > self.run_T + 60.0:
                self.abort("run overran")
        elif self.phase == "POST_HOLD":
            if el > a.post_hold:
                self.ev("mission complete -> SAFETY")
                self.call(self.direct, SetBool.Request(data=False), "set_direct_mode(false)")
                self.goto("ABORT_SETTLE")
        elif self.phase == "ABORT_SETTLE":
            if el > 1.0 and self.odom is not None and self.mode != "DIRECT":
                if el < 1.5:
                    self.ref = np.array([self.odom[0], self.odom[1], self.odom[2]])
                if el > a.abort_settle:
                    self.goto("LAND")
        elif self.phase == "LAND":
            if self.mode == "DIRECT":
                return
            if el < 0.2 and self.odom is not None:
                self.ref = np.array([self.odom[0], self.odom[1], self.odom[2]])
            # descend by reference at 0.2 m/s to land_z
            self.ref[2] = max(a.land_z, self.ref[2] - 0.2 / a.rate)
            if self.ref[2] <= a.land_z + 1e-6 and el > a.land_wait:
                self.call(self.disarm, Trigger.Request(), "rc/disarm")
                self.goto("DONE")
        elif self.phase == "DONE":
            if el > 2.0:
                self.done = True

    def save(self, path):
        log = np.array(self.log) if self.log else np.zeros((0, 8))
        ee = np.array(self.ee_log) if self.ee_log else np.zeros((0, 7))
        np.savez(path, log=log, ee=ee, events=np.array(self.events), info=np.array(self.info or []),
                 shape=self.a.shape, scale=self.a.scale, aborted=self.aborted, reason=self.reason)
        if ee.shape[0] > 10:
            err = np.linalg.norm(ee[:, 1:4] - ee[:, 4:7], axis=1)
            print(f"EE tracking during the run: {ee.shape[0]} samples, mean {err.mean()*1e3:.1f} mm, "
                  f"p95 {np.percentile(err, 95)*1e3:.1f} mm, max {err.max()*1e3:.1f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="/uav_0")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--hover-z", type=float, default=1.0)
    ap.add_argument("--land-z", type=float, default=0.35)
    ap.add_argument("--shape", default="circle", choices=["circle", "figure8"])
    ap.add_argument("--scale", type=float, default=0.8, help="fraction of the planner's s_max")
    ap.add_argument("--direct-settle", type=float, default=15.0)
    ap.add_argument("--post-hold", type=float, default=8.0)
    ap.add_argument("--abort-settle", type=float, default=12.0)
    ap.add_argument("--land-wait", type=float, default=20.0)
    ap.add_argument("--out", default="ee_trajectory_run.npz")
    a = ap.parse_args()
    rclpy.init()
    d = Driver(a)
    try:
        while rclpy.ok() and not d.done:
            rclpy.spin_once(d, timeout_sec=0.1)
    except KeyboardInterrupt:
        d.ev("interrupted")
    finally:
        d.save(a.out)
        print("ABORTED: " + d.reason if d.aborted else "completed")
        d.destroy_node()
        rclpy.shutdown()
    return 1 if d.aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
