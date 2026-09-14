#!/usr/bin/env bash
# One END-EFFECTOR TRAJECTORY flight on the Command.md 7.15.1 IsaacSim rig,
# clean slate to npz -- the same bring-up as fsc_PegasusSimulator's
# wb_l1_tune_cycle.sh, then test/ee_trajectory_sim_driver.py instead of the
# step campaign.
#
#   ee_trajectory_sim_cycle.sh <run-tag> [machine-config] [circle|figure8] [scale-fraction]
#
# Run from a script (never inline): the launchers' pgrep guards match any
# shell whose command line names a node.
set -uo pipefail
TAG="${1:?usage: ee_trajectory_sim_cycle.sh <run-tag> [machine-config] [shape] [scale]}"
CFG="${2:-fsc_lab_machine}"
SHAPE="${3:-circle}"
SCALE="${4:-0.8}"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PEG="${FSC_PEGASUS_ROOT:-$HOME/Source/fsc_PegasusSimulator}"
# shellcheck source=/dev/null
source "$PEG/scripts/config/${CFG}.conf"
AUT="${FSC_AUTOPILOT_WS:-$HOME/Workspaces/fsc_autopilot_ws}/src/fsc_autopilot_ros2"
STACK="$AUT/scripts/isaacsim/start_whole_body_l1_direct_actuation_t650_aerial_manipulator_stack.sh"
SITL="$PEG/scripts/indoor_sim/start_t650_aerial_manipulator_whole_body_L1_adaptive_direct_actuation_sitl.sh"
NODE="autopilot_whole_body_l1_direct_actuation_node"
OUT="${EE_TRAJ_OUT:-$PEG/docs/docs_aerial_manipulator/trajectory_planner_cpp_20260914}"
LOGS="$OUT/logs"
mkdir -p "$OUT" "$LOGS"
set +u
source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
source "${FSC_AUTOPILOT_WS:-$HOME/Workspaces/fsc_autopilot_ws}/install/setup.bash"
set -u

echo "=== [$TAG] 0. clean slate ==="
"$AUT/scripts/isaacsim/stop_isaacsim_stack.sh" >/dev/null 2>&1
"$PEG/scripts/kill_stale_sim_processes.sh" -y >/dev/null 2>&1
sleep 5
echo "=== [$TAG] 1. controller stack ==="
setsid nohup "$STACK" "$CFG" uav_0 > "$LOGS/stack_$TAG.log" 2>&1 < /dev/null &
for _ in $(seq 60); do pgrep -x MicroXRCEAgent >/dev/null && break; sleep 2; done
pgrep -x MicroXRCEAgent >/dev/null || { echo "FAILED: agent never came up"; exit 1; }
for _ in $(seq 60); do pgrep -f "$NODE" >/dev/null && break; sleep 1; done
pgrep -f "$NODE" >/dev/null || { echo "FAILED: $NODE never came up"; exit 1; }
sleep 8
echo "=== [$TAG] 2. Pegasus / PX4 / arm ==="
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
sleep 10
echo "=== [$TAG] 4. flying the $SHAPE at $SCALE x s_max ==="
/usr/bin/python3 "$HERE/ee_trajectory_sim_driver.py" --shape "$SHAPE" --scale "$SCALE" \
    --out "$OUT/ee_${SHAPE}_${TAG}.npz" > "$LOGS/ee_${SHAPE}_${TAG}.log" 2>&1
rc=$?
echo "=== [$TAG] done (driver rc=$rc) ==="
tail -6 "$LOGS/ee_${SHAPE}_${TAG}.log"
exit $rc
