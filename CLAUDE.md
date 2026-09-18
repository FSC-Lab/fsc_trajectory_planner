# CLAUDE.md — fsc_trajectory_planner

Guidance for Claude Code (and anyone else) working in this package. Written
2026-09-14, the day the package was created.

## What this is

`fsc_trajectory_planner` is a **self-contained C++17 / Eigen ROS 2 (ament_cmake,
rclcpp) package** holding the whole-body trajectory planning of the FSC Lab
aerial manipulator, and the `whole_body_trajectory_planner` node that streams
the whole-body reference to `fsc_autopilot_ros2`'s
`single_aerial_manipulator_whole_body_direct_actuation` flight node in DIRECT
mode.

It replaces the Python `whole_body_planner.py` that lived in
`fsc_autopilot_ros2/.../single_aerial_manipulator_whole_body_direct_actuation/planner/`
and imported its maths from `fsc_PegasusSimulator`'s
`extensions/fsc_aerial_manipulation/.../robotic_arm/utils_planner`. That
coupling — a flight-stack node importing a simulator extension by filesystem
path — is gone. **Nothing here depends on Pegasus.** The whole-body model and
the flat B-spline planner are NOT copied either (they were, for one day):
since 2026-09-14 (evening) the package links the flight stack's exported
library **`fsc_autopilot_ros2::wb_law`** — the very `wb_model.cpp` /
`flat_planner.cpp` the whole-body node flies and its parity tests lock — so a
model fix in the autopilot is a fix here on the next build, and there is no
second copy to drift. `include/fsc_trajectory_planner/wb_law.hpp` is the one
place that dependency is named; it imports the types into this namespace.
The other workspace dependencies are `fsc_autopilot_ros2_msgs` (the
`WholeBodyReference` and `PositionControllerReference` messages) and
`px4_msgs`. Build order: `fsc_autopilot_ros2_msgs` → `fsc_autopilot_ros2` →
this package (package.xml declares it, colcon orders it).

The Python planner directory was DELETED from the autopilot repo the same day
(nothing launched it any more); its four rig tests were ported here
(`test/test_planner_loopback.py`, `test_go_home.py`, `test_replan_stream.py`,
`test_traj_viz.py`) and pass against this node. The Python *maths* still
exists in Pegasus (`utils_planner`, imported by its demos and truth
generators) and is what `scripts/dump_python_fixtures.py` reads.

Why C++: a plan is **~3 ms** here (flat B-spline 2.9 ms, Release build)
against 45-260 ms in Python. The whole-body
model alone was 6 µs vs 1 ms per `dynamics()` call.

## Layout

```
include/fsc_trajectory_planner/
  wb_law.hpp              THE dependency: includes fsc_autopilot_ros2's wb_types/wb_model/
                          flat_planner/wb_reference_builder headers and imports the types
                          (Vec3.., WbReference, WholeBodyParams, RestSpec, RotorModel, FlatPlan*,
                          kArmQMin/kArmQMax = WbReferenceBuilder::kQMin/kQMax) into this namespace
  kinematics.hpp          Rz/Rx, minsnap3, buildR0, zxzAngles, armKinematics, armTaskJacobian,
                          sigmaNd, restReference, ikPositionAzimuth, ikWorld, kSigmaNdMargin
  vehicle_model.hpp       VehicleModel + VehicleOptions + the VEHICLE REGISTRY (name -> factory)
  trajectory.hpp          Trajectory / HoldTrajectory / PlanOptions / PlanRequest /
                          TrajectoryPlanner + the PLANNER REGISTRY (name -> factory)
  transition_planner.hpp  FlatBSplineTransitionPlanner (adapter over wb_law's
                          planFlatTransition) -- the only transition backend
  ee_trajectory_planner.hpp  EeShape / EeTrajectoryOptions / EeTrajectoryDiag / EeTrajectoryPlanner
                          (the periodic END-EFFECTOR TRAJECTORY mode, see its section)
  bspline_fit.hpp         clamped uniform B-spline with sparse least-squares fitting and pinned ends
src/
  kinematics.cpp                           port of transition_planner.py / compatible_trajectory.py helpers
  transition_planner.cpp                   the bspline adapter over wb_law's planFlatTransition
  ee_trajectory_planner.cpp, bspline_fit.cpp   the EE trajectory mode
  vehicle_model.cpp                        the registry: "t650_aerial_manipulator" (WholeBodyParams::t650Defaults + RotorModel::t650 from wb_law)
  trajectory.cpp                           HoldTrajectory + the registry: "bspline"
  whole_body_trajectory_planner_node.cpp   the rclcpp node (port of whole_body_planner.py's state machine)
launch/whole_body_trajectory_planner_launch.py     uav_prefix -> namespace, params_file, overrides
config/whole_body_trajectory_planner_t650_aerial_manipulator.yaml   the standalone default config
test/  test_kinematics.cpp, test_transition_planner.cpp (gtest, parity vs Python, fixtures under data/)
       test_flat_planner.cpp (gtest, parity vs Python through the registry, reading the AUTOPILOT's
         installed fixture share/fsc_autopilot_ros2/test_data/flat_plan_t650.txt -- path compiled in)
       test_planner_loopback.py, test_go_home.py, test_replan_stream.py, test_traj_viz.py
         (rclpy rigs driving the built node end to end, no sim -- ported from the Python planner)
       data/python_kinematics_t650.txt, python_transition_t650.txt (fixtures)
scripts/dump_python_fixtures.py   DEV-ONLY: regenerates the python_* fixtures from Pegasus
```

Library target: `fsc_trajectory_planner::planner_lib` (static, PIC, exported)
— pure Eigen on top of `fsc_autopilot_ros2::wb_law`, no ROS. Executable:
`whole_body_trajectory_planner`.

## Build / test

```bash
cd ~/Workspaces/fsc_autopilot_ws
colcon build --packages-select fsc_trajectory_planner     # Release by default (see CMakeLists)
colcon test  --packages-select fsc_trajectory_planner && colcon test-result --verbose
# loopback against the built node (no sim, no PX4, ~40 s):
source install/setup.bash && cd src/fsc_trajectory_planner/test
python3 test_planner_loopback.py                          # yaml default backend (bspline)
```

`CMakeLists.txt` forces `CMAKE_BUILD_TYPE=Release` when none is given. Do not
remove that: the 2026-08-27 finding in the autopilot repo was that an
unoptimised Eigen build of this same model is ~160x slower, and the stream
tick here runs at 100 Hz.

### What the tests lock (all passing 2026-09-17, 10 gtests)

| test | against | tolerance / result |
|---|---|---|
| `Kinematics.ParityWithPython` | `transition_planner.arm_fk_model/rest_ref/_sigma_nd/ik_world` on 6 random rests | 1e-12 m (FK), 1e-7 rad (IK), mass 1e-12 |
| `PlannerRegistry.NamesAndUnknown` | the registry's contract | one backend, `bspline`; an unknown name throws |
| `FlatBSpline.ParityWithPython` | `flat_bspline_planner.py` fixture (2 cases, 37 samples) | 1e-9, duration 1e-9, defect < 1e-9; 2.9 ms/plan |
| loopback (`test_planner_loopback.py`) | the built node | SAFETY-silence → DIRECT hold @100 Hz → PENDING → PLANNED (ride-along) → EE target → Send → EXECUTING → completion HOLD → SAFETY silence |
| `test_go_home.py` | the built node | Go Home plans to the home pose from an arbitrary arm pose |
| `test_replan_stream.py` | the built node | unchanged drone target ignored; stream stays 100 Hz with no gap near the 250 ms staleness window while re-planning; the hold does not move |
| `test_traj_viz.py` | the built node | viz_path/viz_pose layout, unit headings, nose/claw along the arm, curve starts on the hold, 20 Hz arrows, cleared on SAFETY |

**THE STRAIGHT-LINE BACKEND WAS REMOVED 2026-09-17 (user request).** It was
the C++ port of Pegasus's `utils_planner/transition_planner.py`: a straight EE
line with the CoM on a degree-16 polynomial solved by a Picard fixed point and
the joints recovered per sample. Every shipped config had already moved to
`bspline` (sim and hardware alike, same day), which enforces what it only
verified and additionally bounds rotor force and joint torque. Gone with it:
its registry entry, `StraightLine.*` gtests, `test/data/python_transition_t650
.txt`, the transition half of `scripts/dump_python_fixtures.py`, the
`straight_line`-only `PlanOptions` knobs, and the Python `plan_transition` in
Pegasus. Recover from git if a transit ever needs a guaranteed straight EE
line -- it was also the backend of the 2026-09-12 hardware flight.

## The node — `whole_body_trajectory_planner`

Behaviourally identical to the retired Python planner (same states, topics,
services, semantics), so the arm ground station's "EE Whole-Body" tab, the
drone ground station, the Isaac visualiser (`viz_path`/`viz_pose`) and
`wb_l1_campaign_driver.py` all work unchanged.

Run under a vehicle namespace — that is what "one UAV = one namespace" means
here; every name below is relative:

```bash
ros2 launch fsc_trajectory_planner whole_body_trajectory_planner_launch.py uav_prefix:=uav_0 \
    [params_file:=<yaml with a /**/whole_body_trajectory_planner section>] \
    [planner:=bspline] [vehicle:=t650_aerial_manipulator] \
    [base_com:="[x,y,z]"] [arm_joint_sign:="[-1,1,1,-1]"] [hold_ee_world:=false]
```

| | name (under `/<uav_prefix>/`) |
|---|---|
| subscribes | `fsc_autopilot_ros2/whole_body_direct_actuation/mode` (String, latched), `fsc_autopilot_ros2/position_controller/reference` (drone GS), `fmu/out/vehicle_attitude` (PX4, best-effort), `state_estimator/local_position/odom` (position ONLY), `fsc_open_manipulator/joint_states`, `whole_body_planner/ee_target` (arm GS) |
| publishes | `fsc_autopilot_ros2/whole_body_direct_actuation/reference` (WholeBodyReference, 100 Hz in DIRECT), `whole_body_planner/{status,pending_base,target_joints,workspace_rz,viz_path}` (latched), `whole_body_planner/{viz_pose,current_ee,current_ee_body}`, `fsc_open_manipulator/external_torque_controller/reference_joint_trajectory` (the arm reference, same sample as the law's) |
| services | `whole_body_planner/{send,clear,go_home}` (std_srvs/Trigger) |

The `whole_body_planner/` prefix is the `topic_prefix` parameter; it is kept
so the two ground stations and the Isaac visualiser keep resolving. Every
input topic is a parameter too (see the config yaml).

States, printed verbatim on `status`: `IDLE` (SAFETY) → `HOLD` → `PENDING`
(drone-GS target captured, never executed) → `CALCULATING` → `PLANNED T=..s`
/ `INFEASIBLE: <reason>` → (Send) `EXECUTING T=..s` → `HOLD` at the goal.
Any mode change to SAFETY drops everything instantly.

Frames at the ROS boundary, exactly as the Python did and as the C++
`frame_adapter` does: `R0_model = R0_actual · R_MODEL`, `phi_model =
psi_actual − π/2`, GS/EE yaws are ACTUAL on the wire, the streamed message is
MODEL frame. The hold is captured from MEASUREMENT throughout (odometry
position, PX4 EKF2 attitude, encoder joints) — the 2026-09-04 decision.

`hold_ee_world` stays **false** (default and in every launcher): the world-EE
re-solve moves `x_cd` with base drift and leaves the position loop
effectively open (2026-08-31 measurement). Do not flip it on a vehicle.

The plan runs on a `std::thread` with a generation counter so a superseded
result is discarded, and the 100 Hz stream never waits on a solve. A plan
takes milliseconds; the thread is kept because a stalled solver must still
never stall the stream.

## Extending it

**A new vehicle**: add one factory in `src/vehicle_model.cpp` returning a
`VehicleModel` (whole-body params, rotor force limits, joint box, home pose,
`sigma_nd` margin, fold guard, `r_model`), register it in
`vehicleFactories()`, select it with the `vehicle` parameter. Nothing in the
planners or the node names a vehicle. `WholeBodyParams` currently fixes the
arm at 4 joints (`kNumJoints`), so a different arm is a bigger change than a
different airframe.

**Periodic end-effector trajectories** (circle, figure-8) are a separate MODE
now — see "End-effector trajectory mode" below; a new periodic shape is one
more branch of `localShape()` in `ee_trajectory_planner.cpp` (position and
its τ-derivative in the shape's local frame, τ=0 tangent along +x).

**A new rest-to-rest backend**: implement
`TrajectoryPlanner::plan(vehicle, request, options)` returning a `Trajectory`
(`duration()`, `eval(t)` → `WbReference`, `goalRest()`, `diag()`), register
it in `plannerFactories()` (`src/trajectory.cpp`), select it with the
`planner` parameter. `PlanRequest` carries `rest0`, an optional `rest1`, and a
free-form `shape` map (radius, period, laps, ...) so the node does not change
when a shape needs more than two endpoints. `ee_trajectory_planner.cpp`'s
`localShape()` / `eeAt()` show the shape contract — prescribe every channel
with analytic derivatives against the phase, then let the Picard loop solve the
compatible CoM. Reference: `compatible_trajectory.py`'s
`_prescribed_task_showcase` in Pegasus is the multi-segment version of this
(it fits one polynomial per rest-to-rest segment; a long periodic path needs
that, a single degree-16 polynomial does not fit a lap). Today only the
rest-to-rest transitions are wired from the ground stations; a shape would
also need a trigger (a service taking name + parameters) in the node.

**Rules that carry over from the Python planner and must not be lost:**

- The node **never arms, never changes PX4 mode, never publishes in SAFETY**.
- Min-snap (septic) phasing is REQUIRED wherever a task is prescribed against a
  phase (the EE-trajectory ramps), not a nicety: the solved CoM velocity depends
  on the prescribed jerk, so min-jerk endpoints step it at the hold joins.
- `GRIPPER_OFF_WRIST` (0.108 m, inside `t650Defaults`), the joint box, `home_pose`
  and `tau_joint_max` each exist in several places (the flight node — which this
  package now shares by linking — the arm controller, the Isaac plant, the arm
  GS). Change one, change all.
- `base_com` MUST equal the flight node's `wb_base_com_*` (the hardware
  launcher cross-checks and refuses); the node builds `x_cd` from this model
  while the law computes `x_c` from its own.
- Unchanged drone-GS targets are ignored only while already
  PENDING/CALCULATING/PLANNED/INFEASIBLE/EXECUTING — from HOLD an unchanged
  target re-plans. Drivers must publish on CHANGE in DIRECT (Command.md §7.15.5).

## Provenance and parity

The whole-body model, the flat B-spline planner, `RestSpec`, `RotorModel` and
the joint box are **the flight node's, linked** (`fsc_autopilot_ros2::wb_law`,
exported by `fsc_autopilot_ros2/CMakeLists.txt`; headers under
`include/single_aerial_manipulator_whole_body_direct_actuation/`). Do not
copy any of them back into this package: the day they were copied
(2026-09-14, morning) is exactly the drift risk the user asked to remove.
`kinematics.cpp` and `transition_planner.cpp` are fresh ports of the Python;
`scripts/dump_python_fixtures.py` regenerates their fixtures from a Pegasus
checkout (`FSC_PEGASUS_ROOT`, default `~/Source/fsc_PegasusSimulator`, run
with `PYTHONNOUSERSITE=1 /usr/bin/python3`). The flat-planner parity test
reads the autopilot's installed fixture, so the two packages can never hold
two different truths for the same code.

## Where it is wired in

- `fsc_autopilot_ros2/scripts/isaacsim/start_whole_body_{,l1_}direct_actuation_t650_aerial_manipulator_stack.sh`
  — the `planner` tmux window now runs this node (`ros2 launch ... params_file:=<the flight yaml>`).
- `fsc_autopilot_ros2/scripts/indoor_exp/start_whole_body_direct_actuation_stack_t650_aerial_manipulator.sh`
  — same, passing the measured `base_com` and `arm_joint_sign:=[-1,1,1,-1]`; no Pegasus root, no interpreter probe.
- The four whole-body yamls in `fsc_autopilot_ros2/config/` carry a
  `/**/whole_body_trajectory_planner:` section (mirroring the old
  `whole_body_planner:` one) so one file still describes a run.
- `stop_isaacsim_stack.sh`, `stop_autopilot_stack.sh`, `status_autopilot_stack.sh`
  know the executable name.

## Simulation validation (Command.md §7.15.1 rig, 2026-09-14)

See the section "Simulation validation record" at the end of this file.

## Simulation validation record

**2026-09-14, first flight of this package, Command.md §7.15.1 rig (AM-T650
whole-body + L1, `params_single_aerial_manipulator_whole_body_l1_direct_actuation_t650_sim.yaml`,
backend `bspline`), fsc_lab_machine, headless Isaac.** Run end to end by
`fsc_PegasusSimulator/application/robotic_arm/utils/wb_l1_tune_cycle.sh l1 cpp_planner fsc_lab_machine`:
clean slate → L1 stack (this node in the `planner` window, launched from the
flight yaml) → Pegasus/PX4/arm → offboard → arm → SAFETY climb to 1 m →
DIRECT → 20 s soak → the standard mission driven through the ground stations'
ROS interface (`wb_l1_campaign_driver.py`: drone-GS reference topic for the
x/y/yaw steps, arm-GS `whole_body_planner/ee_target` + `whole_body_planner/send`
for the compatible trajectories) → SAFETY → land → disarm.

Result: **10/10 legs completed, every target PLANNED within 10-20 ms, no
INFEASIBLE, no abort, no watchdog trip**; the flight node reported the
streamed reference fresh (debug[56] = 1) at 100.0 Hz for the whole DIRECT
phase. Per-leg peak / settled CoM error: x steps 352-386 / 50-61 mm, y steps
356-389 / 73-94 mm, yaw steps 70-166 / 40-45 mm, **compatible EE trajectory 34 /
11 mm (back: 15 / 5 mm)**, whole-system base+arm move 199-211 / 69-71 mm.
Record: `fsc_PegasusSimulator/docs/docs_aerial_manipulator/trajectory_planner_cpp_20260914/`
(npz, metrics, stack/pegasus/driver logs) and Command.md §7.15.11. Not
compared like-for-like against the Python planner in flight (the plant config
has moved since the 2026-09-06 run E table); the reference streams are
identical to 1e-9 by the parity tests, which is the claim that matters.

**Second flight, same evening, after re-basing the package on the linked
`fsc_autopilot_ros2::wb_law`** (tag `cpp_planner_wblaw`): 10/10 legs, no
refusal, no abort, stream fresh 100 %, u1 47.60 N. Per-leg peak / settled
CoM error: x 318-356 / 79-84 mm, y 328-389 / 79-89 mm, yaw 77-167 / 46-53 mm,
compatible EE trajectory 46 / 21 mm (back 35 / 15 mm), whole-system move
209-220 / 71-76 mm — same profile as the first flight within the rig's
run-to-run scatter. Score in the same record directory.

Not yet done: a hardware flight with this node (the hardware launcher is
wired and passes `base_com` / `arm_joint_sign`, unflown), and the figure-8 /
circle shapes (registry hooks only).

## End-effector trajectory mode (2026-09-15)

A second planning mode beside the rest-to-rest transitions: a PERIODIC
end-effector pose trajectory (circle or figure-8) selected from the arm
ground station's new **EE trajectory** tab, turned into a compatible
whole-body reference and flown as one run.

### What the operator sees (arm GS, `utils_custom_ground_station`, tab "EE trajectory")

The Sine Test and Demos tabs were removed; this tab took their place
(`src/ee_trajectory_panel.{hpp,cpp}`). Row 1: trajectory type (None / Circle /
Figure-8), the time-scale slider (0 .. the planner's feasible maximum), **Go to
start** (a compatible transition to the run's start rest, planned by the
transition planner and executed without a Send), **Start trajectory** (enabled
only while the planner reports vehicle position, yaw and joints within
tolerance of that rest AND the planner is in HOLD). Below: a 3-D view of the
planned EE path coloured by |v_ee| with a colour bar, the world triad and the
planner's current reference EE frame (x = the claw axis, z = the gripper's up),
drag to orbit, wheel to zoom; status lines are overlaid on the view.

### Interface (`whole_body_planner/ee_trajectory/*`, relative to the vehicle namespace)

| | name | type |
|---|---|---|
| in | `select` | `std_msgs/String` `circle` / `figure8` / `none` |
| in | `time_scale` | `std_msgs/Float64` s (clamped to [0.05, s_max]) |
| out | `status` (latched) | `NONE` / `NOT IN DIRECT` / `CALCULATING` / `READY s=.. max=.. T=..s EE err ..` / `INFEASIBLE: reason` |
| out | `info` (latched) | `[s, s_max, T_total, T_lap, laps, ramp, type_id, ee_pos_err, ee_rot_err_deg, peak_v, peak_a, peak_qdot, peak_tau_j, min_sigma_nd]` |
| out | `path` (latched) | 400 × `[t, x, y, z, speed, qx, qy, qz, qw]`, world frame |
| out | `start_error` (10 Hz) | `[pos_err_m, yaw_err_deg, joint_err_deg, ready]` |
| out | `reference_pose` (20 Hz while streaming) | `PoseStamped`, the current EE reference |
| out | `start_rest` (latched) | `PoseStamped`, the run's start BASE pose (actual yaw): what go_to_start flies to |

`maxTimeScale` also carries the refusal from its slowest probe, so when
nothing is feasible the status names the binding bound (a geometry one, since
rates vanish as the run slows) instead of a bare "no feasible time scale".
| srv | `go_to_start`, `start` | `std_srvs/Trigger` |

The run is a `Trajectory`, so the node's existing EXECUTING machinery streams
it (100 Hz `WholeBodyReference` + the arm reference) and re-holds at its end.
Selecting a shape anchors it on the CURRENT hold (the circle on the origin at
the held EE height, the figure-8 at the held EE point); a new hold re-anchors
it, arriving at the run's own start rest does not. SAFETY drops everything.

### The maths (`ee_trajectory_planner.{hpp,cpp}`, `bspline_fit.{hpp,cpp}`)

Following the MATLAB task-space planner (`~/Downloads/Task-space Planner`,
`main_redundant_zyxx.m` + `recover_motion_redundant.m`), adapted to the
z-x-x-z OM-X chain:

1. **EE pose curve** p_e(τ): the circle is **centred on the world origin** at
   the current EE height (`ee_traj_center_origin`, default true); the run
   starts at the point of that circle on the current EE's bearing, tangent
   ccw/cw there — so Go-to-start carries the vehicle onto the circle. The
   figure-8 `(A sin ωτ, B sin 2ωτ)` (or the circle with `center_origin`
   false) is rotated so its τ=0 tangent lies along the drone's nose and
   translated onto the held EE point. Yaw = the curve's tangent (ACTUAL
   azimuth), R_e = Rz(ψ_tan − π/2)·Rx(β_e) in the MODEL frame. The run's
   start rest is published latched on `ee_trajectory/start_rest`.
   **The assigned q₂ and the fold are coupled and are checked before the
   solve:** at rest q₃ = β_e − q₂ exactly, so the q₂ sinusoid's whole range
   must leave q₃ inside its box. At the 80° fold that pins q₂ to [30°, 50°],
   i.e. a centre of 40° with an amplitude up to 9° (10° grazes the +50° stop
   once the thrust tilt is added). A split that does not fit is refused in
   0.1 ms naming the fold, the implied q₃ range and the admissible centres,
   rather than after several seconds of failed time-scale probes.
   **β_e (`ee_traj_fold_deg`, default 80° = the flown home fold) is the EE's
   roll about its heading.** A literally level EE (β_e = 0) is this arm's
   wrist singularity — joints 1 and 4 share the vertical axis — and violates
   the β ≥ 5° / σ_nd ≥ 0.10 constraints the mode enforces, so it is refused
   (the unit test `LevelEndEffectorIsRefusedAsSingular` locks that). In the
   MATLAB model (z-y-x-x, arm hanging straight down at zero angles) the same
   attitude is regular; the difference is the chain, not the maths.
2. **Time scaling** τ(t): min-snap ramp-in over `ramp_time`, constant rate s
   for `laps` whole periods minus the ramp phase, min-snap ramp-out — the run
   starts and ends AT REST on the same pose (`startRest()`), which is what
   makes Go-to-start a rest-to-rest transition.
3. **Flat outputs** at every grid sample: thrust direction r → R2(r) (the
   MATLAB `R2_from_t` minimal tilt), A = R2ᵀR_e, (ψ, β, γ) = zxz(A): the first
   z angle is the DRONE yaw (q1 is fixed at 0), β = q2 + q3 with q2 the
   ASSIGNED slow sinusoid (`q2_center_deg` + `q2_amp_deg` sin(2πτ/P)), γ = q4;
   ψ is then refined onto the law's own `build(b3, b1)` attitude so `flatState`
   reproduces exactly this R0; x_c = p_e + R0(r0c − r0e)(q) (A6).
4. **Feasibility**: where the MATLAB script minimises the residual
   r − normalize(ẍ_c + g e3) with fmincon over a trig-series r, this solves the
   same equation as a **Picard fixed point** with x_c a clamped B-spline (degree
   7, 0.5 s spans, 5 coincident end points ⇒ rest ends) and **under-relaxed
   thrust updates (0.5)**. The relaxation is not cosmetic: the map amplifies a
   thrust perturbation at frequency ω by L·ω²/g (L ≈ 0.2 m lever), i.e. it is
   NOT a contraction above ~1 Hz, and the first version diverged from a 1.8×/
   iteration mode at the ramp-out junction. A coarse basis plus relaxation
   makes every representable mode contract (21 iterations, 7e-5 m).
5. **Fitting**: ψ (deg 5, unwrapped) and q (deg 5, 4 channels) as B-splines
   with 3 coincident end points; then `flatState()` (wb_law) on the fitted
   (x_c⁽⁰⁻⁴⁾, ψ⁽⁰⁻²⁾, q⁽⁰⁻²⁾) is the 16-field reference.
6. **Checks on the whole run** (the artifact's constraint set + rates):
   joint box, β ≥ 5°, σ_nd ≥ 0.10, |v| ≤ v_max, |a| ≤ a_max, |ψ̇| ≤ w_max,
   |q̇| ≤ `qdot_max`, |τ_j| ≤ `tau_joint_max` and rotor forces inside the
   T650 model via `inverseInputs()`, rest at both ends, and the **FK round
   trip**: the EE pose recomputed from the fitted flat outputs against the
   prescribed curve (position and rotation angle; circle 0.1 mm / 0.0°,
   figure-8 0.0 mm / 0.0°). The first violated bound is the refusal reason.
7. **s_max** (`maxTimeScale`): bracket then bisect to 2 % on the full check;
   ~0.2 s. The GS slider spans [0, s_max]. On the T650 defaults the circle
   (0.5 m, 24 s lap) is limited by the yaw rate at s_max 1.13, the figure-8
   (0.5 × 0.25 m) by the acceleration at 0.36.

Tests: `test_ee_trajectory_planner.cpp` (4 gtests: both shapes compatible and
bounded at 0.9 s_max, s_max positive and binding, level EE refused) and
`test/test_ee_trajectory_loopback.py` (the whole ROS flow against the built
node: select → READY → rescale → start refused 20 cm off → go-to-start → at
start → start → 8888 streamed samples round the circle → HOLD at the start).
Sim: `test/ee_trajectory_sim_cycle.sh <tag> [cfg] [shape] [scale]` flies it on
the Command.md 7.15.1 rig with `test/ee_trajectory_sim_driver.py`.

**Sim validation of the EE trajectory mode (2026-09-15, 7.15.1 rig, L1
stack):** two complete circle flights (0.5 m, 2 laps) at s = 0.90 and 0.45 of
the 1.13 maximum — select → Go-to-start → Start → run → HOLD → SAFETY → land.
The reference's FK round trip was 0.0-0.1 mm; the measured EE trailed it by a
~2 s first-order lag with gain < 1 (raw error 214 mm at 0.118 m/s, 138 mm at
0.059 m/s; 102 / 56 mm after removing the lag; flown radius 0.40 / 0.45 m of
0.50). That is the whole-body law's position-loop bandwidth, the same
behaviour §7.15.5 measured on steps, not a planning error. Record: Command.md
§7.15.12, data in `trajectory_planner_cpp_20260914/ee_circle_*.npz`.

**Third flight, 2026-09-15, the ORIGIN-CENTRED circle** (`ee_circle_origin_circle.npz`),
same rig and speed: the reference sits at 0.500 m from the world origin with a
standard deviation of **0 mm**, the flown circle is concentric to ~5 cm at
0.387 m, and the error decomposes exactly as before (1.90 s lag, 104 mm
residual). It is also the first flight in which **Go-to-start was a real
transition** — 0.56 m and 90° of yaw, complete in 15 s — and the Start gate's
5 cm / 5° / 3° tolerance passed on the first try. That gate is the thing to
watch when the circle is moved away from the vehicle: §7.15.5 measures
50-90 mm of settled error after a 0.5 m leg, the same order as the tolerance.
