#!/usr/bin/env python3
"""Fly the PICK-AND-PLACE mission on the IsaacSim rig and record it (the arm
ground station's Pick & Place / Pick & Place PS4 interfaces, driven by a
script).

    /usr/bin/python3 pick_place_sim_driver.py --out run.npz

    offboard -> arm -> SAFETY climb to hover_z -> settle -> DIRECT -> settle
      -> Adjust, type the place point, Get obj_0, Plan     (Pick & Place tab)
      -> go_to_start -> execute_pick: to 0.10 m above obj_0, WAIT there until
         the claw has stayed inside 50 mm for pick_place_settle_s (the
         planner's own gate), the vertical descent
      -> wait for the 50 mm arrival gate; fine correction: engage the
         Pick & Place PS4 tab, one right-stick push up, disengage
      -> go_to_place_start (retreat first) -> execute_place (arm sweep to
         above the place point, wait, descent)
      -> arrival gate; fine correction again
      -> go_to_land_start -> execute_land (the hover over the landing spot)
      -> SAFETY -> land by reference -> disarm

THE PICK OBJECT IS A STAND-IN. The Isaac scene has no pickup body: the
OptiTrack emulator publishes obj_0 only for the gripper marker cube (off in
the 4-D yaml). This driver publishes a static pick point itself (--obj-topic)
and TYPES the place point into the planner (pick_place_place_point), as the
Pick & Place tab does, so every leg, the sweep, the gate and the fine
correction fly for real, but nothing is grasped.

Records odometry, the planner's EE reference (ee_trajectory/reference_pose,
published for every streamed sample), the measured EE (current_ee) and the
arrival error, tagged by leg, so each leg's claw tracking can be scored.
Refuses to start against an already-armed vehicle.
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

from fsc_autopilot_ros2_msgs.msg import Mocap, PositionControllerReference
from px4_msgs.msg import VehicleStatus

PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                     durability=DurabilityPolicy.VOLATILE,
                     history=HistoryPolicy.KEEP_LAST, depth=10)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
DA = "fsc_autopilot_ros2/whole_body_direct_actuation"
PP = "whole_body_planner/pick_place"
LEGS = ["go_to_start", "execute_pick", "go_to_place_start", "execute_place",
        "go_to_land_start", "execute_land"]


class Abort(Exception):
    pass


class Driver(Node):
    def __init__(self, a):
        super().__init__("pick_place_sim_driver")
        self.a = a
        ns = self.ns = a.namespace.rstrip("/")
        gs = f"{ns}/fsc_open_manipulator"
        self.t0 = time.time()
        self.events = []
        self.odom = None
        self.mode = ""
        self.armed = None
        self.status = ""
        self.pp_status = ""
        self.info = None
        self.arrival = None
        self.ref_pose = None
        self.fine_engaged = None
        self.leg = -1
        self.ref = np.array([0.0, 0.0, a.hover_z])
        self.stream_ref = True          # SAFETY needs the position reference stream
        self.joy = None
        self.lock = threading.Lock()
        self.log, self.ee_log, self.arr_log = [], [], []

        self.create_subscription(Odometry, f"{ns}/state_estimator/local_position/odom", self.on_odom, 10)
        self.create_subscription(String, f"{ns}/{DA}/mode", lambda m: setattr(self, "mode", m.data), LATCHED)
        self.create_subscription(VehicleStatus, f"{ns}/fmu/out/vehicle_status_v1",
                                 lambda m: setattr(self, "armed", m.arming_state == 2), PX4_QOS)
        self.create_subscription(String, f"{ns}/whole_body_planner/status",
                                 lambda m: setattr(self, "status", m.data), LATCHED)
        self.create_subscription(String, f"{ns}/{PP}/status", self.on_pp_status, LATCHED)
        self.create_subscription(Float64MultiArray, f"{ns}/{PP}/info",
                                 lambda m: setattr(self, "info", np.asarray(m.data)), LATCHED)
        self.create_subscription(Float64MultiArray, f"{ns}/{PP}/arrival_error", self.on_arrival, 10)
        self.create_subscription(PoseStamped, f"{ns}/whole_body_planner/ee_trajectory/reference_pose",
                                 self.on_ref_pose, 10)
        self.create_subscription(PoseStamped, f"{ns}/whole_body_planner/current_ee", self.on_cur_ee, 10)
        self.create_subscription(Bool, f"{gs}/pick_place_fine/engaged",
                                 lambda m: setattr(self, "fine_engaged", m.data), LATCHED)
        self.ref_pub = self.create_publisher(PositionControllerReference,
                                             f"{ns}/fsc_autopilot_ros2/position_controller/reference", 10)
        self.joy_pub = self.create_publisher(Joy, f"{ns}/rc/input", 1)
        self.mocap_pub = [(self.create_publisher(Mocap, a.obj_topic, 10), np.array(a.obj))]
        self.cli = {n: self.create_client(Trigger, f"{ns}/rc/{n}") for n in ("offboard", "arm", "disarm")}
        self.direct = self.create_client(SetBool, f"{ns}/{DA}/set_direct_mode")
        for n in LEGS + ["adjust", "plan", "capture_pick"]:
            self.cli[n] = self.create_client(Trigger, f"{ns}/{PP}/{n}")
        self.fine = self.create_client(SetBool, f"{gs}/pick_place_fine/set_engaged")
        self.set_params = self.create_client(SetParameters, f"{ns}/whole_body_trajectory_planner/set_parameters")
        self.create_timer(1.0 / a.rate, self.tick)
        self.create_timer(0.01, self.feed_mocap)

    # ---- callbacks ---------------------------------------------------------
    def now(self):
        return time.time() - self.t0

    def on_odom(self, m):
        p, v = m.pose.pose.position, m.twist.twist.linear
        self.odom = np.array([p.x, p.y, p.z, v.x, v.y, v.z])
        self.log.append([self.now(), p.x, p.y, p.z, v.x, v.y, v.z,
                         1.0 if self.mode == "DIRECT" else 0.0, self.leg])

    def on_pp_status(self, m):
        # WAITING repeats with the live claw error: log it once per leg
        same_wait = m.data.startswith("WAITING") and \
            m.data.split(":")[0] == self.pp_status.split(":")[0]
        if m.data != self.pp_status and not same_wait:
            self.ev(f"pick_place: {m.data}")
        self.pp_status = m.data

    def on_arrival(self, m):
        self.arrival = list(m.data)
        self.arr_log.append([self.now(), *self.arrival[:9]])

    def on_ref_pose(self, m):
        self.ref_pose = np.array([m.pose.position.x, m.pose.position.y, m.pose.position.z])

    def on_cur_ee(self, m):
        if self.ref_pose is not None and self.mode == "DIRECT":
            p = m.pose.position
            self.ee_log.append([self.now(), p.x, p.y, p.z, *self.ref_pose, self.leg])

    def tick(self):
        if self.stream_ref and self.mode != "DIRECT":
            m = PositionControllerReference()
            m.header.stamp = self.get_clock().now().to_msg()
            m.position.x, m.position.y, m.position.z = map(float, self.ref)
            m.yaw, m.yaw_unit = 0.0, PositionControllerReference.DEGREES
            self.ref_pub.publish(m)
        if self.joy is not None:
            self.joy_pub.publish(Joy(axes=self.joy[0], buttons=self.joy[1]))

    def feed_mocap(self):
        for pub, p in self.mocap_pub:
            m = Mocap()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = "map"
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, p)
            m.pose.orientation.w = 1.0
            pub.publish(m)

    # ---- helpers -----------------------------------------------------------
    def ev(self, s):
        line = f"[{self.now():7.2f}s] {s}"
        self.events.append(line)
        print(line, flush=True)

    def wait(self, cond, timeout, what):
        t = time.time()
        while time.time() - t < timeout:
            if cond():
                return time.time() - t
            time.sleep(0.05)
        raise Abort(f"timeout ({timeout:.0f} s): {what}  [mode={self.mode} plan={self.status} "
                    f"pick_place={self.pp_status}]")

    def call(self, cli, req, name, must=True):
        if not cli.wait_for_service(timeout_sec=10.0):
            raise Abort(f"service {name} not available")
        fut = cli.call_async(req)
        self.wait(lambda: fut.done(), 10, f"{name} response")
        r = fut.result()
        self.ev(f"{name}: success={r.success} {r.message}")
        if must and not r.success:
            raise Abort(f"{name} refused: {r.message}")
        return r

    def trig(self, name, must=True):
        return self.call(self.cli[name], Trigger.Request(), name, must)

    def settled(self, tol_m=0.08, tol_v=0.10, hold=3.0, timeout=60.0):
        since = None
        t = time.time()
        while time.time() - t < timeout:
            if self.odom is not None:
                ok = (np.linalg.norm(self.odom[:3] - self.ref) < tol_m and
                      np.linalg.norm(self.odom[3:6]) < tol_v)
                since = (since or time.time()) if ok else None
                if since and time.time() - since > hold:
                    return
            time.sleep(0.05)
        raise Abort("hover did not settle")

    def pad(self, ry=0.0):
        self.joy = ([0.0, 0.0, 1.0, 0.0, ry, 1.0, 0.0, 0.0], [0] * 13)

    # ---- the mission -------------------------------------------------------
    def fly_leg(self, k):
        self.leg = k
        name = LEGS[k]
        t_call = self.now()
        self.trig(name)
        self.wait(lambda: self.status.startswith("EXECUTING"), 15, f"{name} EXECUTING")
        T = float(self.status.split("T=")[1].split("s")[0])
        self.ev(f"{name}: executing, T = {T:.1f} s")
        if k in (1, 3):
            # a claw leg stops ABOVE its target and waits for the claw to
            # settle inside the tolerance; the planner then flies the descent
            self.wait(lambda: self.pp_status.startswith("WAITING " + name), T + 30, f"{name} above")
            t_wait = self.now()
            self.ev(f"{name}: above the target, waiting (claw {self.arrival[2]*1e3:.0f} mm)")
            self.wait(lambda: "(descent)" in self.pp_status or self.pp_status.startswith("INCOMPLETE"),
                      self.a.approach_wait + 15, f"{name} descent")
            if self.pp_status.startswith("INCOMPLETE"):
                raise Abort(self.pp_status)
            self.wait(lambda: self.status.startswith("EXECUTING"), 10, f"{name} descent EXECUTING")
            Td = float(self.status.split("T=")[1].split("s")[0])
            self.ev(f"{name}: descending after {self.now() - t_wait:.1f} s above, T = {Td:.1f} s")
            T += Td
        self.wait(lambda: self.status == "HOLD" and self.info is not None and int(self.info[5]) == k,
                  T + 30, f"{name} complete")
        self.ev(f"{name}: complete after {self.now() - t_call:.1f} s")
        return T

    def gate_and_fine(self, k):
        """After a claw leg: time the 50 mm gate, then one fine correction."""
        # the gate must STAY open (the claw settles through it, not onto it)
        t, since = time.time(), None
        try:
            while True:
                if time.time() - t > self.a.gate_timeout:
                    raise Abort("gate")
                ok = self.arrival is not None and int(self.arrival[0]) == k and self.arrival[8] > 0.5
                since = (since or time.time()) if ok else None
                if since and time.time() - since >= self.a.gate_hold:
                    break
                time.sleep(0.05)
            self.ev(f"{LEGS[k]}: claw inside the {self.arrival[6]*1e3:.0f} mm gate for "
                    f"{self.a.gate_hold:.0f} s, {time.time() - t:.1f} s after the leg "
                    f"(error {self.arrival[2]*1e3:.1f} mm)")
        except Abort:
            e = self.arrival[2] * 1e3 if self.arrival else float("nan")
            self.ev(f"{LEGS[k]}: claw NOT inside the gate within {self.a.gate_timeout:.0f} s "
                    f"(error {e:.1f} mm) -- fine correction skipped")
            return
        if not self.a.fine:
            return
        self.pad(0.0)                      # the pad goes live before engaging
        time.sleep(0.5)
        r = self.call(self.fine, SetBool.Request(data=True), "pick_place_fine/set_engaged(true)", must=False)
        if not r.success:
            self.ev("fine correction: engage refused -- skipped")
            self.joy = None
            return
        claw_z = self.ref_pose[2] if self.ref_pose is not None else float("nan")
        self.pad(ry=1.0)                   # right stick up
        try:
            self.wait(lambda: self.status.startswith(("CALCULATING", "PLANNED", "EXECUTING")), 5,
                      "fine-correction target")
            self.pad(0.0)
            self.wait(lambda: self.status.startswith("EXECUTING"), 10, "fine move EXECUTING")
            self.wait(lambda: self.status == "HOLD", 20, "fine move complete")
            rise = (self.ref_pose[2] - claw_z) * 1e3
            self.ev(f"fine correction: one push flown, claw reference rose {rise:.1f} mm, "
                    f"still engaged={self.fine_engaged}")
        finally:
            self.pad(0.0)
            self.call(self.fine, SetBool.Request(data=False), "pick_place_fine/set_engaged(false)", must=False)
            self.joy = None
        time.sleep(3.0)

    def apply_planner_params(self):
        """--planner-param name=value: set live on the planner before Plan (the
        pick-and-place parameters are read at Plan time)."""
        params = []
        # the place point is TYPED IN on the ground station, not measured
        self.a.planner_param = list(self.a.planner_param) + [
            "pick_place_place_point=[" + ",".join(f"{v:.4f}" for v in self.a.drop) + "]"]
        for kv in self.a.planner_param:
            name, val = kv.split("=", 1)
            v = ParameterValue()
            if val.startswith("["):
                v.type, v.double_array_value = ParameterType.PARAMETER_DOUBLE_ARRAY, \
                    [float(x) for x in val.strip("[]").split(",")]
            elif val.lower() in ("true", "false"):
                v.type, v.bool_value = ParameterType.PARAMETER_BOOL, val.lower() == "true"
            elif "." in val or "e" in val.lower():
                v.type, v.double_value = ParameterType.PARAMETER_DOUBLE, float(val)
            else:
                v.type, v.integer_value = ParameterType.PARAMETER_INTEGER, int(val)
            params.append(Parameter(name=name, value=v))
        if not self.set_params.wait_for_service(timeout_sec=10.0):
            raise Abort("planner set_parameters not available")
        fut = self.set_params.call_async(SetParameters.Request(parameters=params))
        self.wait(lambda: fut.done(), 10, "set_parameters")
        for p, r in zip(params, fut.result().results):
            self.ev(f"planner param {p.name}: {'set' if r.successful else 'REFUSED ' + r.reason}")
            if not r.successful:
                raise Abort(f"planner refused {p.name}: {r.reason}")

    def mission(self):
        a = self.a
        self.wait(lambda: self.odom is not None and self.mode and self.armed is not None, 60, "vehicle feeds")
        if self.armed:
            raise Abort("vehicle already ARMED -- refusing (clean relaunch needed)")
        self.ref = self.odom[:3].copy()
        time.sleep(1.0)
        self.trig("offboard")
        time.sleep(3.0)
        self.trig("arm")
        time.sleep(2.0)
        self.ref = np.array([self.odom[0], self.odom[1], a.hover_z])
        self.ev(f"climbing to {a.hover_z} m")
        self.settled()
        self.call(self.direct, SetBool.Request(data=True), "set_direct_mode(true)")
        self.wait(lambda: self.mode == "DIRECT", 5, "DIRECT")
        self.stream_ref = False
        time.sleep(a.direct_settle)
        self.wait(lambda: self.status == "HOLD", 10, "planner HOLD")
        self.apply_planner_params()
        # the Pick & Place tab's setup row
        self.trig("adjust")
        time.sleep(1.0)
        self.trig("capture_pick")
        self.trig("plan")
        self.wait(lambda: self.pp_status.startswith("READY"), 30, "mission READY")
        self.ev(f"planned leg durations {np.round(self.info[8:14], 1).tolist()} s")
        for k in range(6):
            self.fly_leg(k)
            if k in (1, 3):
                self.gate_and_fine(k)
            else:
                time.sleep(a.leg_pause)
        self.wait(lambda: self.pp_status.startswith("COMPLETE"), 10, "COMPLETE")
        time.sleep(a.post_hold)

    def land(self):
        self.leg = -1
        self.joy = None
        if self.mode == "DIRECT":
            self.ref = self.odom[:3].copy() if self.odom is not None else self.ref
            self.stream_ref = True
            try:
                self.call(self.direct, SetBool.Request(data=False), "set_direct_mode(false)", must=False)
                self.wait(lambda: self.mode != "DIRECT", 5, "SAFETY")
            except Abort as e:
                self.ev(str(e))
        self.stream_ref = True
        time.sleep(self.a.abort_settle)
        self.ref = self.odom[:3].copy()
        self.ev("landing by reference")
        while self.ref[2] > self.a.land_z:
            self.ref[2] = max(self.a.land_z, self.ref[2] - 0.2 / 50.0)
            time.sleep(0.02)
        time.sleep(self.a.land_wait)
        try:
            self.trig("disarm", must=False)
        except Abort as e:
            self.ev(str(e))

    def save(self, path, aborted, reason):
        ee = np.array(self.ee_log) if self.ee_log else np.zeros((0, 8))
        np.savez(path, log=np.array(self.log) if self.log else np.zeros((0, 9)), ee=ee,
                 arrival=np.array(self.arr_log) if self.arr_log else np.zeros((0, 10)),
                 events=np.array(self.events), info=self.info if self.info is not None else np.zeros(0),
                 obj=np.array(self.a.obj), drop=np.array(self.a.drop), aborted=aborted, reason=reason)
        if ee.shape[0] > 10:
            print("claw tracking (measured current_ee vs streamed reference), per leg:")
            for k in range(6):
                s = ee[ee[:, 7] == k]
                if s.shape[0] < 5:
                    continue
                err = np.linalg.norm(s[:, 1:4] - s[:, 4:7], axis=1)
                print(f"  {LEGS[k]:18s} {s.shape[0]:5d} samples  mean {err.mean()*1e3:6.1f}  "
                      f"p95 {np.percentile(err, 95)*1e3:6.1f}  max {err.max()*1e3:6.1f}  "
                      f"final {err[-1]*1e3:6.1f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--namespace", default="/uav_0")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--hover-z", type=float, default=1.0)
    ap.add_argument("--land-z", type=float, default=0.30)
    ap.add_argument("--obj", type=float, nargs=3, default=[0.8, 0.6, 0.66],
                    help="stand-in pickup point published as /obj_0/mocap [m]")
    ap.add_argument("--drop", type=float, nargs=3, default=[0.6, -1.6, 0.66],
                    help="the place point, typed into the planner (pick_place_place_point) [m]")
    ap.add_argument("--obj-topic", default="/obj_0/mocap",
                    help="where to publish the stand-in pickup point (= the planner's pick_place_pick_topic)")
    ap.add_argument("--gate-timeout", type=float, default=45.0)
    ap.add_argument("--approach-wait", type=float, default=60.0,
                    help="= the planner's pick_place_approach_wait_max [s]")
    ap.add_argument("--gate-hold", type=float, default=2.0,
                    help="the arrival gate must stay open this long before the fine correction engages")
    ap.add_argument("--no-fine", dest="fine", action="store_false")
    ap.add_argument("--direct-settle", type=float, default=15.0)
    ap.add_argument("--leg-pause", type=float, default=3.0)
    ap.add_argument("--post-hold", type=float, default=5.0)
    ap.add_argument("--abort-settle", type=float, default=8.0)
    ap.add_argument("--land-wait", type=float, default=15.0)
    ap.add_argument("--planner-param", action="append", default=[],
                    help="name=value set on the planner before Plan, e.g. "
                         "pick_place_pick_pose_deg=[0,-10,20,0] (repeatable)")
    ap.add_argument("--out", default="pick_place_run.npz")
    a = ap.parse_args()
    rclpy.init()
    d = Driver(a)
    spin = threading.Thread(target=rclpy.spin, args=(d,), daemon=True)
    spin.start()
    aborted, reason = False, ""
    try:
        d.mission()
        d.ev("mission complete")
    except Abort as e:
        aborted, reason = True, str(e)
        d.ev("ABORT: " + reason)
    except KeyboardInterrupt:
        aborted, reason = True, "interrupted"
    finally:
        if d.armed:
            d.land()
        d.save(a.out, aborted, reason)
        print("ABORTED: " + reason if aborted else "completed")
        d.destroy_node()
        rclpy.try_shutdown()
        spin.join(timeout=5.0)
    return 1 if aborted else 0


if __name__ == "__main__":
    raise SystemExit(main())
