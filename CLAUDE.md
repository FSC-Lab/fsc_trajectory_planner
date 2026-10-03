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
  arm_sweep_planner.hpp   ArmSweepOptions / ArmSweepDiag / ArmSweepPlanner + nonicPhase: a rest-to-rest
                          leg with joints sweeping between limits (pick-and-place execute_place)
  pick_place.hpp          PickPlaceLeg / Geofence / PickPlaceConfig / PickPlaceTargets / PickPlaceWaypoints,
                          pickPlaceWaypoints, retreatRest, planPickPlaceLeg, planPickPlaceDescent,
                          checkPathInFence, planPickPlaceMission
                          (the PICK-AND-PLACE mode, see its section)
  workspace.hpp           WorkspaceGrid / usableWorkspace: the (r, z) envelope published on workspace_rz
src/
  kinematics.cpp                           port of transition_planner.py / compatible_trajectory.py helpers
  transition_planner.cpp                   the bspline adapter over wb_law's planFlatTransition
  ee_trajectory_planner.cpp, bspline_fit.cpp   the EE trajectory mode
  arm_sweep_planner.cpp, pick_place.cpp        the pick-and-place mode
  workspace.cpp                            the workspace_rz grid (moved out of the node 2026-10-01, unchanged)
  vehicle_model.cpp                        the registry: "t650_aerial_manipulator" (WholeBodyParams::t650Defaults + RotorModel::t650 from wb_law)
  trajectory.cpp                           HoldTrajectory, SequenceTrajectory (rest-to-rest legs back to back) + the registry: "bspline"
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
| publishes | `fsc_autopilot_ros2/whole_body_direct_actuation/reference` (WholeBodyReference, 100 Hz in DIRECT), `whole_body_planner/{status,pending_base,target_joints,workspace_rz,viz_path}` (latched), `whole_body_planner/{viz_pose,current_ee,current_ee_body,current_base}` (current_base = the MEASURED airframe current_ee is built from, same tick and gate — the arm GS's Drone triad), `whole_body_planner/current_ee_heading` (Float64, the measured claw heading in the ee_target yaw convention, 2026-10-01), `fsc_open_manipulator/external_torque_controller/reference_joint_trajectory` (the arm reference, same sample as the law's) |
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


## Pick-and-place mode (2026-10-01)

A third planning mode beside the transitions and the EE trajectories: SIX
operator-triggered legs, each a compatible whole-body move that ends AT REST,
one service (one GS button) per leg, flown in order (any leg flown while the
vehicle holds at a table -- after execute_pick / execute_place, whichever leg
is next, a re-fly included -- retreats first):

| # | service (`whole_body_planner/pick_place/...`) | goal |
|---|---|---|
| 0 | `go_to_start` | `pick_place_start` + Adjust offset, arm home |
| 1 | `execute_pick` | claw `approach_dz` ABOVE the captured `obj_0` point (+ `pick_place_pick_ee_offset`), arm in `pick_place_pick_pose_deg`, nose turned to face it; WAIT; then the vertical DESCENT onto the point |
| 2 | `go_to_place_start` | RETREAT (climb `retreat_dz`, back off `retreat_back` along -nose), then `pick_place_place_start` + offset, arm in the carry pose |
| 3 | `execute_place` | claw above the TYPED place point (`pick_place_place_point` + offset), **the arm sweeping its bands on the way** (`ArmSweepPlanner`); WAIT; DESCENT onto the point |
| 4 | `go_to_land_start` | retreat, then `pick_place_land_start` + offset, arm home |
| 5 | `execute_land` | `pick_place_land` + offset: the hover over the landing spot (0.8 m by default -- NOT lower, see the Isaac record below). The node never lands / arms / changes PX4 mode -- touchdown is the operator's SAFETY -> land |

Base poses are `[x, y, z, yaw_deg]`, world frame, ACTUAL yaw; they are
PLACEHOLDERS in the yaml. **Adjust** (`pick_place/adjust`, works in SAFETY too):
with the vehicle on the physical start mark, `measured - nominal start` (x, y;
z / yaw only with `pick_place_adjust_z` / `_yaw`) becomes the offset every
nominal pose is shifted by -- the mocap centring changes between sessions.
**Typed points (2026-10-02, user decision).** Only the PICK point is measured;
start, place_start, land_start, land (`[x, y, z, yaw_deg]`) and the place
point `pick_place_place_point` (`[x, y, z]`) are typed in -- the arm GS's
Pick & Place tab writes these parameters live -- and are shifted by the Adjust
offset like every room-frame point. An `add_on_set_parameters_callback` checks
their sizes, refuses edits while a leg is in flight and any `*_topic` change
(fixed at launch), and makes the plan STALE on every waypoint edit (status
`NOT PLANNED: press Plan`). **Captures** (`pick_place/capture_<point>`, point =
start | pick | place_start | place | land_start | land): average the point's mocap body
(`pick_place_<point>_topic`, `fsc_autopilot_ros2_msgs/Mocap`, ABSOLUTE topic,
SensorData QoS) over `pick_place_capture_window` (0.5 s). pick = `/obj_0/mocap`
is the only topic set by default (place too can be measured if
`pick_place_place_topic` is set at launch and `capture_place` called; a capture
wins over the typed point); the base points have no topic (refused, the typed
pose is used) -- set one to measure
that pose's x, y from a marker (z and yaw stay nominal, no offset applied).
Mocap and odometry share the ENU world frame (the indoor bridge passes mocap
through); an EKF2-fused stack must keep its local origin on the mocap origin.

**Plan** (`pick_place/plan`, DIRECT only) dry-runs the whole mission on a
worker thread (leg 0 from the hold, each later leg from the previous goal) and
publishes it; any change to a capture or the offset invalidates it. Each leg
service then RE-PLANS that leg from the CURRENT hold (the PS4 fine correction
may have moved it) and executes it as soon as it is planned (like
Go-to-start). A leg may be flown again, none skipped (`leg <= completed + 1`).
`reset` forgets progress + plan (captures and offset stay). SAFETY drops the
plan and any leg in flight, keeps captures, offset and progress.

Geometry. A claw goal puts the grasp point on the target with the arm's
horizontal reach on the bearing from the previous leg's base to the target, so
the nose turns to face the object over the whole leg (the flat planner's
heading channel is smooth and `w_max`-bounded). **The pick / place pose
`[0, 0, 0, 0]`** (user spec "both joints at 90 deg, arm facing down, EE away
from the drone"): upper arm straight down, forearm out along the nose, claw
straight DOWN; grasp point 0.155 m ahead of and 0.340 m below the body origin,
sigma_nd 0.169 (> 0.10; the beta >= 5 deg guard is an EE-trajectory-mode rule,
the transitions plan joints directly). **It is the bottom of the arm's
reach**: with the base fixed (the PS4 / ee_target path) IK cannot lower the
claw even 2 cm nor push it 5 cm further out from there -- only up, in, or
sideways. Lowering onto an object is the base's job.

**The arm-sweep leg** (`arm_sweep_planner.{hpp,cpp}`): every flat output in
closed form -- x_c and psi on a NONIC phase (rest through snap: a septic one
steps snap, i.e. the body-torque reference, at the joins), q = (1 - w) q_lin +
w (c + a sin(omega (t - r) + phase)) for swept joints with w a nonic plateau
window -- mapped through `flatState()`; no fixed point (nothing is prescribed in
task space). Defaults **q1 [-10, 10] deg, q2 [17.5, 37.5] deg** (2026-10-02,
user decision after +-25 deg tripped the Isaac tilt watchdog; was q1 +-25, q2
[10, 45]), 90 deg apart, 2 cycles. T = the shortest duration (bracket +
bisection to 2 %) passing joint box, sigma_nd, v/a/yaw rate,
`pick_place_sweep_qdot_max`, joint torque and rotor force on a 401-point grid;
on the test scene 15.0 s, bound by the joint rate, 40 ms to plan.

**Approach, wait, descent (2026-10-02, audit M2; user wording "approach above
the target, then when the error is within the 50 mm threshold start the
descent").** A claw leg's service flies to the APPROACH rest -- the goal
rest raised by `pick_place_approach_dz` (0.10 m), same heading and arm, so the
claw is 0.10 m straight above the target -- and the node then WAITS there
(`HOLD`, the leg still in flight). `publishPpArrival` measures the claw
against the point above the target; once it has stayed inside
`pick_place_arrival_tol` for `pick_place_settle_s` (1 s: the claw settles
THROUGH the tolerance in the law's oscillation, it must dwell, audit L1),
`startPpDescent` plans the approach -> goal transition from the hold
(`planPickPlaceDescent`; the claw stays on the vertical to 1e-10 m, 3 s) and
flies it; only its end completes the leg. No descent within
`pick_place_approach_wait_max` (60 s, 0 = forever) ends the leg INCOMPLETE,
holding above the target (re-fly it or Abort). `approach_dz` 0 flies straight
onto the target as before. Plan's dry run chains approach + descent.

**Abort** (`pick_place/abort`, the GS's red button, which also opens the
gripper): arm home and the vehicle `pick_place_abort_climb` (0.30 m) higher,
x, y and heading kept, planned and flown at once. A running leg is CUT at the
current reference (phi from b1_d, q = q_d, x_b chosen so x_c is continuous:
1.7 mm max CoM step in the loopback), a waiting leg cancelled; the cut leg
does not count as done. The climb is clipped at the fence ceiling, so repeated
presses never leave the volume (audit H3). A second press while aborting is
refused.

**Gates (2026-10-02 audit).** H1 `Geofence` (`pick_place_fence_min_z` 0.6,
`_max_z` 1.8, `_xy` [-2.5, 2.5, -2.5, 2.5], on the BODY origin): every goal and
approach rest at Plan, and every planned leg's path (200 samples, level
attitude) -- a leg may START outside (DIRECT engaged low) but no sample may lie
further outside than its start, per bound. H2 Adjust refuses an offset larger
than `pick_place_adjust_max` (0.5 m: not on the start mark). M1 a leg starts
only with fresh odometry and the body within `pick_place_leg_start_tol`
(0.15 m) of its hold. M3 the retreat follows the vehicle being AT a table
(`pp_at_table_`, set when a claw leg completes, cleared by an abort or Reset),
not the leg index, so re-flying execute_pick from the object retreats too. M4
the sweep band +-10 deg (above).

Outputs: `pick_place/status` (latched; `NOT IN DIRECT` / `NOT PLANNED: <what
is missing>` / `PLANNING` / `READY next=<leg>` / `FLYING <leg> [(approach) |
(descent)] T=..s` / `WAITING <leg>: claw N mm from the point above the target
(tolerance 50 mm) -- descends once inside` / `DONE <leg> next=<leg> [-- claw
within tolerance: fine correction OK | inside tolerance, settling | outside
tolerance]` / `INFEASIBLE: <leg>: <reason>` / `INCOMPLETE: <leg>: ...` (wait
timeout) / `ABORTING -- ...` / `ABORTED -- holding higher, arm home; ...` /
`COMPLETE -- touch down with SAFETY -> land`),
`pick_place/info` (latched, 80 doubles: [0..2] offset, [3] yaw offset deg,
[4] planned, [5] last leg completed, [6] leg in flight, [7] tolerance,
[8..13] leg durations, [14..55] per leg [goal base x y z, actual yaw deg, claw
x y z], [56..79] per point [captured x y z, valid]), `pick_place/path` and
`pick_place/drone_path` (latched, the planned mission at the claw / airframe,
600 x the ee_trajectory/path 9-double stride), `pick_place/arrival_error`
(10 Hz: [leg, claw?, |e|, e_xyz, tol, within, settled, phase] -- the claw
by FK on the measured state for the execute legs, the body otherwise; the
target is the point ABOVE the object while phase is 1 (approach) / 2 (wait),
the object itself in 3 (descent) / 0; settled = inside for settle_s AND the
leg done AND HOLD: the gate on which the arm GS's separate "Pick & Place PS4"
tab may engage the gamepad fine correction).

Two node changes made for that fine correction (2026-10-01):
- **`pick_place/workspace_rz`** (latched, the workspace_rz layout). The
  plain `workspace_rz` keeps only folds beta >= 5 deg (the EE-trajectory
  guard), which puts the claw-down grasp point (z -0.340 m) OUTSIDE it (lowest
  point -0.331 m): a tab saturating on it refuses every stick increment there.
  This second grid lowers the floor to 1 deg under `pick_place_pick_pose_deg` /
  `_place_pose_deg` when they fold below the guard, those extra poses also
  needing sigma_nd >= 0.10; lowest point -0.345 m, no original cell lost
  (`usableWorkspace`, `workspace.hpp`). Only the Pick & Place PS4 tab reads
  it; `workspace_rz` itself is unchanged, so the EE Whole-Body and PS4 Remote
  tabs behave exactly as before.
- **`current_ee_heading`** (Float64, 15 Hz, same freshness gate as
  current_ee): the measured claw heading as ee_target reads it (actual yaw;
  `clawAzimuth` + pi/2). current_ee carries no orientation, and a target
  seeded without it asks for a wrist swing; ikWorld with this heading returns
  the current joints (gtest).

Tests: `test_pick_place.cpp` (14 gtests, 2026-10-02: the approach rest straight
above with the descent on the vertical and never under the target, any leg
from a table retreats, a zero-length re-fly from the approach hold, the fence
-- goal / approach / path refused, a leg starting outside may come back but
not go further out --, the +-10 deg sweep default; and from 2026-10-01: the workspace floor admits the
claw-down pose -- up / in / sideways yes, down / out no -- and loses no
original cell (the node publishes it on pick_place/workspace_rz); ikWorld with clawAzimuth returns the current joints, claw-down
included; goals put the claw on the target
facing it, an impossible arm pose is refused naming the leg, the retreat, the
sweep leg compatible / at rest / analytic derivatives = central differences /
visits its band / every bound / shortest T, a band outside the box refused,
the six-leg mission dry run chained through the retreat points,
SequenceTrajectory refuses a gap) and `test/test_pick_place_loopback.py` (the
whole ROS flow against the built node with rig-private mocap topics: Adjust in
SAFETY (an 0.8 m offset refused) -> captures -> Plan -> a 0.4 m land refused
by the fence -> out-of-order refused -> six legs with the perfect-plant
teleport, each claw leg WAITING above its target until the rig teleports onto
the approach rest (no descent at 80 mm; at 30 mm it begins after the 1 s
dwell), a 2 s wait timeout -> INCOMPLETE, the leg-start gate -> fine-correction
gate at 50 mm both ways -> retreat climb 0.150 m -> abort mid-sweep -> sweep q1
+-10 deg -> COMPLETE -> abort from the hold, then clipped at a 1.20 m ceiling
-> reset -> SAFETY silence, plus current_ee_heading and the claw-down point
inside pick_place/workspace_rz but still outside the unchanged workspace_rz;
PASS 2026-10-02). The arm GS (fsc_om_ws, utils_custom_ground_station) has the
**Pick & Place** tab (the buttons, typed points, ABORT) and the separate **Pick
& Place PS4** tab (the fine correction); its
`test/test_pick_place_fine_integration.py` drives both against this node
(17 checks, 15 pass 2026-10-02; the 2 PS4 Remote ones need the planner's
`teleop/engage`, not on this branch). Flown once in Isaac with the approach /
wait / descent (pp7, 2026-10-02, record below): it did what it is meant to,
and the vehicle crashed at the pick anyway.

### Isaac Sim record (2026-10-01, `test/pick_place_sim_cycle.sh`)

The sim twin of the indoor run sheet: the 4-D L1 EKF2-fused stack
(`isaacsim/start_whole_body_l1_4d_direct_actuation_t650_aerial_manipulator_stack_fused.sh`)
plus Pegasus `start_t650_aerial_manipulator_whole_body_L1_adaptive_4D_fused_direct_actuation_sitl.sh`,
on the STRESSED `..._l1_4d_..._t650_sim.yaml` (mass/inertia x1.10, CoM shift,
arm friction/mass x1.05, +17.6 % allocator kf), driven by
`test/pick_place_sim_driver.py`. The scene has no table objects, so the driver
publishes stand-in pick / drop points on private topics
(`FSC_PICK_PLACE_PICK_TOPIC` / `_PLACE_TOPIC` -> the launch file's
`pick_place_*_topic` overrides). Logs and npz under `fsc_autopilot_ws/log/pick_place_sim/`.

- **Two full missions flown** (all six legs, the 19.2 s sweep, retreats, the
  hover over the landing spot) with no refusal or abort from the flight node.
- **Phantom obj_0.** The emulator on this machine (master, without the
  skip-until-first-pose change) publishes obj_0 at the origin when no marker
  cube is spawned; with the stand-in on the same topic the first capture
  averaged two publishers (856 mm scatter) and flew it. Hence
  `pick_place_capture_max_spread` (20 mm): such a capture is now refused.
- **Vertical oscillation in DIRECT.** SAFETY hover 36-49 mm peak to peak;
  the moment DIRECT is engaged, arm folded and no leg flown, 161 mm at
  0.36 Hz, and 115-226 mm at 0.32-0.49 Hz in every hold after a leg, while
  the planner's hold reference is dead still (0.000 mm). It is the closed
  loop, not the plan; the long holds' 0.32 Hz equals the L1 translational
  filter bandwidth `wb_l1_omega_c_t` = 2 rad/s. **Confirmed by the matched
  plant** (`..._sim_ps4test.yaml` swapped in, pp3): DIRECT hover 161 -> 6 mm
  peak to peak, claw-down hold 226 -> 17 mm, go_to_start claw error 5.8 mm
  mean -- the oscillation is the law against the injected mismatch.
- **The 50 mm gate flickers in it.** The claw swings through the tolerance
  (30-190 mm), so the Pick & Place PS4 tab's engage was refused each time
  (correctly) and no fine correction was flown in sim.
- **Fine correction flown in Isaac** on the matched plant: the gate held for
  2 s at 3.7 mm (pick) and 30.8 mm (place), the Pick & Place PS4 tab engaged,
  one stick push was planned, sent by the tab and flown (+9.5 / +8.7 mm).
- **Tilt-watchdog trips with the arm extended (open).** pp3 (matched): the
  translational L1 estimate F_hat rang and GREW while holding claw-down at
  the place (1.5 -> 9 -> 13.7 N, sign alternating along one direction in the
  arm's vertical plane, starting before the fine correction) and reached
  66.9 N in the retreat -> "DIRECT WATCHDOG TRIPPED: excess tilt 20.4/20.0
  deg". The same pose at the pick was stable in the same flight. pp4
  (stressed): the same trip 8 s into the sweep (F_hat 12 N); pp1/pp2 swept
  through. The watchdog reverted to SAFETY, and PX4's attitude control
  caught the vehicle in pp4 but NOT in pp3 nor the 2026-10-02 live run on the
  stressed plant (same trip, 20.4 deg at 279 deg/s, F_hat to 24.6 N in the
  claw-down hold at the place): both fell from 1 m and ended on their side.
  The SAFETY fallback does not always recover that tilt rate.
  A flight-law question (wb_l1_*), not a plan one; the planner can only
  excite it less (slower / narrower sweep, a better-conditioned pick pose).
- **The 0.4 m land hover crashed.** The body stands at 0.305 m on its gear;
  the hover dipped to 0.318 m, brushed the floor, saturated the allocator, and
  the vehicle tumbled on the switch to SAFETY. `pick_place_land` defaults to
  0.8 m since -- not yet flown (pp3 / pp4 tripped before execute_land).
- **pp7 (2026-10-02), the first flight with approach / wait / descent**,
  matched plant, pick / place pose [0, -10, 20, 0], sweep +-10 deg at 0.3
  rad/s (same as pp6 apart from the new logic). execute_pick flew to 0.10 m
  above obj_0, WAITED, and the planner started the 3 s descent itself 7.2 s
  later (claw inside 50 mm for 1 s); the leg completed and the gate opened.
  5 s later the tilt watchdog tripped (20.2 deg, 87 deg/s), SAFETY did not
  recover, and the vehicle fell onto its side. The body's vertical swing
  (~0.4 Hz) had grown: 50 mm peak to peak holding above, 144 mm in the
  descent, 124 -> 169 mm after it -- where pp6 held the SAME pose at the same
  spot at a steady 18 mm. Reading: the 3 s descent's energy sits near that
  0.3-0.4 Hz mode (the L1 bandwidth), the 1 s dwell is shorter than one
  ~2.5 s swing so the descent started mid-swing, and the law's swing grows
  once it is large (as in pp3). Proposed, not yet done: a dwell longer than
  one swing (`pick_place_settle_s` 3 s) and a slower descent (>= 8 s, a new
  `pick_place_descent_t_min`). Data: `log/pick_place_sim/*pp7_matched_approach*`.
