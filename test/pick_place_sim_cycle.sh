#!/usr/bin/env bash
# One PICK-AND-PLACE flight on the IsaacSim rig, clean slate to npz -- the
# simulation twin of the indoor run sheet:
#
#   [Orin]   MicroXRCEAgent + start_whole_body_l1_4d_direct_actuation_stack_t650_aerial_manipulator_fused.sh
#     ->     fsc_autopilot_ros2/scripts/isaacsim/start_whole_body_l1_4d_direct_actuation_t650_aerial_manipulator_stack_fused.sh
#            (agent, OptiTrack emulator, whole-body L1 4-D node, EKF2-fused estimator,
#             drone ground station, virtual remote, and the trajectory planner window)
#   [Orin]   start_open_manipulator_inverted_wb_torque.sh + [Laptop] arm GS + gamepad
#     ->     fsc_PegasusSimulator/scripts/indoor_sim/start_t650_aerial_manipulator_whole_body_L1_adaptive_4D_fused_direct_actuation_sitl.sh
#            (Isaac + PX4 SITL, the Isaac torque-mode arm stack, the inverted arm GS
#             with its Pick & Place tabs, the gamepad if /dev/input/js0 exists)
#   [Laptop] mocap  ->  the emulator (uav_0) + test/pick_place_sim_driver.py's
#            stand-in pickup / drop points (the scene has no table objects), on
#            private topics -- see FSC_PICK_PLACE_*_TOPIC below
#
#   pick_place_sim_cycle.sh <run-tag> [machine-config]
#
# Run from a script (never inline): the launchers' pgrep guards match any
# shell whose command line names a node.
set -uo pipefail
TAG="${1:?usage: pick_place_sim_cycle.sh <run-tag> [machine-config]}"
CFG="${2:-fsc_lab_machine}"
shift 2 2>/dev/null || shift $#
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PEG="${FSC_PEGASUS_ROOT:-$HOME/Source/fsc_PegasusSimulator}"
# shellcheck source=/dev/null
source "$PEG/scripts/config/${CFG}.conf"
# PP_AUTOPILOT_WS: fly a different fsc_autopilot_ros2 (e.g. an overlay
# workspace holding a git worktree of another branch, built on top of this
# one). The machine config sets FSC_AUTOPILOT_WS unconditionally, hence the
# separate name.
WS="${PP_AUTOPILOT_WS:-${FSC_AUTOPILOT_WS:-$HOME/Workspaces/fsc_autopilot_ws}}"
AUT="$WS/src/fsc_autopilot_ros2"
STACK="$AUT/scripts/isaacsim/start_whole_body_l1_4d_direct_actuation_t650_aerial_manipulator_stack_fused.sh"
SITL="$PEG/scripts/indoor_sim/start_t650_aerial_manipulator_whole_body_L1_adaptive_4D_fused_direct_actuation_sitl.sh"
NODE="autopilot_whole_body_l1_direct_actuation_node"
OUT="${PP_SIM_OUT:-${FSC_AUTOPILOT_WS:-$HOME/Workspaces/fsc_autopilot_ws}/log/pick_place_sim}"
# One yaml for the controller AND the plant. Stacks with WB_SIM_PROFILE
# (dev_robotic_arm: mirror = the hardware gains on the identified plant,
# robustness = the stress plant) read WB_SIM_YAML for the controller; the
# Pegasus launcher reads it for the plant either way.
if grep -q WB_SIM_PROFILE "$STACK"; then
  export WB_SIM_PROFILE="${WB_SIM_PROFILE:-mirror}"
  case "$WB_SIM_PROFILE" in robustness) SUF=_sim_robustness.yaml ;; *) SUF=_sim.yaml ;; esac
else
  SUF=_sim.yaml
fi
export WB_SIM_YAML="${WB_SIM_YAML:-$AUT/config/params_single_aerial_manipulator_whole_body_l1_4d_direct_actuation_t650$SUF}"
echo "fsc_autopilot_ros2: $AUT ($(git -C "$AUT" log --oneline -1 2>/dev/null))"
echo "controller + plant yaml: $WB_SIM_YAML"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
# The stand-in pick object gets a PRIVATE topic: the emulator (master, without
# the skip-until-first-pose change) publishes a phantom obj_0 at the origin
# when no gripper marker cube is spawned. The planner's launch file reads
# it (its pick_place_pick_topic override); the stack's tmux server inherits
# it from this shell. The place point is typed in (the driver sets it).
export FSC_PICK_PLACE_PICK_TOPIC="${FSC_PICK_PLACE_PICK_TOPIC:-/sim_pick_place/obj_0/mocap}"
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "$WS/install/setup.bash"
set -u

echo "=== [$TAG] 0. clean slate ==="
"$AUT/scripts/isaacsim/stop_isaacsim_stack.sh" >/dev/null 2>&1
"$PEG/scripts/kill_stale_sim_processes.sh" -y >/dev/null 2>&1
sleep 5
echo "=== [$TAG] 1. controller stack (4-D L1, EKF2-fused) ==="
# a tmux server that is already up does not hand new sessions this shell's
# environment: push the two variables into its global environment as well
tmux setenv -g FSC_PICK_PLACE_PICK_TOPIC "$FSC_PICK_PLACE_PICK_TOPIC" 2>/dev/null || true
setsid nohup "$STACK" "$CFG" uav_0 > "$LOGS/stack_$TAG.log" 2>&1 < /dev/null &
for _ in $(seq 60); do pgrep -x MicroXRCEAgent >/dev/null && break; sleep 2; done
pgrep -x MicroXRCEAgent >/dev/null || { echo "FAILED: agent never came up"; exit 1; }
for _ in $(seq 60); do pgrep -f "$NODE" >/dev/null && break; sleep 1; done
pgrep -f "$NODE" >/dev/null || { echo "FAILED: $NODE never came up"; exit 1; }
for _ in $(seq 30); do pgrep -f whole_body_trajectory_planner >/dev/null && break; sleep 1; done
pgrep -f whole_body_trajectory_planner >/dev/null || { echo "FAILED: the planner never came up"; exit 1; }
sleep 8
got=$(timeout 15 ros2 param get /uav_0/whole_body_trajectory_planner pick_place_pick_topic 2>&1 | sed 's/.*: //')
echo "planner pick_place_pick_topic = $got (wanted $FSC_PICK_PLACE_PICK_TOPIC)"
[ "$got" = "$FSC_PICK_PLACE_PICK_TOPIC" ] || { echo "FAILED: the planner is not on the stand-in topic"; exit 1; }
echo "=== [$TAG] 2. Pegasus / PX4 / arm stack / arm GS ==="
DISPLAY="${DISPLAY:-:0}" setsid nohup "$SITL" --in-terminal "$CFG" > "$LOGS/pegasus_$TAG.log" 2>&1 < /dev/null &
echo "=== [$TAG] 3. waiting for odometry / EKF / arm ==="
ok=0
for _ in $(seq 120); do
  timeout 5 ros2 topic echo --once /uav_0/state_estimator/local_position/odom >/dev/null 2>&1 && { ok=1; break; }
  sleep 5
done
[ "$ok" = 1 ] || { echo "FAILED: no odometry"; exit 1; }
ok=0
for _ in $(seq 60); do
  f=$(timeout 5 ros2 topic echo --once /uav_0/fmu/out/estimator_status_flags 2>/dev/null)
  if grep -q "cs_yaw_align: true" <<<"$f" && grep -q "cs_ev_pos: true" <<<"$f" && grep -q "cs_ev_yaw: true" <<<"$f"; then ok=1; break; fi
  sleep 5
done
[ "$ok" = 1 ] || { echo "FAILED: EKF never aligned"; exit 1; }
for _ in $(seq 60); do
  timeout 5 ros2 topic echo --once /uav_0/fsc_open_manipulator/joint_states >/dev/null 2>&1 && break
  sleep 5
done
for _ in $(seq 60); do
  ros2 service type /uav_0/fsc_open_manipulator/pick_place_fine/set_engaged >/dev/null 2>&1 && break
  sleep 2
done
ros2 service type /uav_0/fsc_open_manipulator/pick_place_fine/set_engaged >/dev/null 2>&1 \
  || echo "WARNING: the arm GS's Pick & Place PS4 tab is not up -- the fine-correction steps will be skipped"
sleep 10
echo "=== [$TAG] 4. flying the pick-and-place mission ==="
/usr/bin/python3 "$HERE/pick_place_sim_driver.py" --out "$OUT/pick_place_$TAG.npz" \
    --obj-topic "$FSC_PICK_PLACE_PICK_TOPIC" "$@" \
    > "$LOGS/driver_$TAG.log" 2>&1
rc=$?
echo "=== [$TAG] done (driver rc=$rc) ==="
tail -14 "$LOGS/driver_$TAG.log"
exit $rc
