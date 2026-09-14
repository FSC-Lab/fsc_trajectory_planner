#!/usr/bin/env python3
"""Regenerate test/data/python_*.txt from the Pegasus Python planners.

DEV-ONLY: needs a fsc_PegasusSimulator checkout (FSC_PEGASUS_ROOT, default
~/Source/fsc_PegasusSimulator) and numpy. The C++ tests read the committed
fixtures and never import Python. Run with the system interpreter:

    PYTHONNOUSERSITE=1 /usr/bin/python3 scripts/dump_python_fixtures.py
"""
import os
import sys

import numpy as np

root = os.environ.get("FSC_PEGASUS_ROOT",
                      os.path.expanduser("~/Source/fsc_PegasusSimulator"))
sys.path.insert(0, os.path.join(root, "extensions", "fsc_aerial_manipulation"))
from fsc_aerial_manipulation.robotic_arm.utils_planner import (  # noqa: E402
    transition_planner as TP)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "test", "data")
os.makedirs(OUT, exist_ok=True)
np.set_printoptions(precision=17)


def fmt(v):
    return " ".join(repr(float(x)) for x in np.asarray(v, float).ravel())


def dump_kinematics():
    P = TP.make_params_t650()
    rng = np.random.default_rng(7)
    lines = []
    lo, hi = TP.Q_MIN, TP.Q_MAX
    cases = []
    for _ in range(12):
        q = lo + (hi - lo) * rng.random(4)
        if q[1] + q[2] < np.deg2rad(6.0):
            continue
        x_b = rng.uniform(-1.0, 1.0, 3) + np.array([0, 0, 1.5])
        phi = rng.uniform(-np.pi, np.pi)
        cases.append((x_b, phi, q))
    cases.append((np.array([0.0, 0.0, 1.2]), 0.0,
                  np.array([0.0, np.deg2rad(40.0), np.deg2rad(40.0), 0.0])))
    lines.append(f"cases {len(cases)}")
    for x_b, phi, q in cases:
        r0c, r0e, re = TP.arm_fk_model(q, P)
        rr = TP.rest_ref(P, x_b, phi, q)
        az = np.arctan2(rr["b1_de"][1], rr["b1_de"][0])
        q_ik, info = TP.ik_world(P, x_b, phi, rr["r_ed"], az,
                                 np.array([0.0, 0.7, 0.7, 0.0]))
        sig = TP._sigma_nd(q, P)
        lines.append("x_b " + fmt(x_b))
        lines.append(f"phi {phi!r}")
        lines.append("q " + fmt(q))
        lines.append("r0c " + fmt(r0c))
        lines.append("r0e " + fmt(r0e))
        lines.append("Re " + fmt(re))
        lines.append("x_cd " + fmt(rr["x_cd"]))
        lines.append("r_ed " + fmt(rr["r_ed"]))
        lines.append("b1_d " + fmt(rr["b1_d"]))
        lines.append("b1_de " + fmt(rr["b1_de"]))
        lines.append(f"sigma_nd {sig!r}")
        lines.append("q_ik " + fmt(q_ik) + f" {int(info['ok'])}")
    lines.append(f"total_mass {sum(P['m_i'])!r}")
    with open(os.path.join(OUT, "python_kinematics_t650.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"kinematics: {len(cases)} cases")


def dump_transition():
    P = TP.make_params_t650()
    home = np.array([0.0, np.deg2rad(40.0), np.deg2rad(40.0), 0.0])
    q_star = np.array([np.deg2rad(15.0), np.deg2rad(30.0),
                       np.deg2rad(35.0), np.deg2rad(-20.0)])
    cases = {
        "combined": ({"x_b": np.array([0.0, 0.0, 1.2]), "phi": 0.0, "q": home},
                     {"x_b": np.array([0.4, 0.3, 1.4]),
                      "phi": np.deg2rad(20.0), "q": q_star}),
        "yaw_only": ({"x_b": np.array([0.0, 0.0, 1.2]), "phi": 0.0, "q": home},
                     {"x_b": np.array([0.0, 0.0, 1.2]),
                      "phi": np.deg2rad(30.0), "q": home}),
        "arm_only": ({"x_b": np.array([0.0, 0.0, 1.2]), "phi": 0.0, "q": home},
                     {"x_b": np.array([0.0, 0.0, 1.2]), "phi": 0.0,
                      "q": q_star}),
        "model_heading": ({"x_b": np.array([0.3, -0.2, 1.0]),
                           "phi": -0.5 * np.pi, "q": home},
                          {"x_b": np.array([0.8, 0.1, 1.3]),
                           "phi": -0.5 * np.pi + np.deg2rad(15.0),
                           "q": np.array([0.05, 0.76, 0.41, 0.5])}),
    }
    keys = ("x_cd", "x_cd_dot", "x_cd_ddot", "x_cd_d3", "x_cd_d4",
            "b1_d", "b1_d_dot", "b1_d_ddot", "r_ed", "r_ed_dot", "r_ed_ddot",
            "b1_de", "b1_de_dot", "b1_de_ddot", "q_d", "qdot_d")
    lines = [f"cases {len(cases)}"]
    for name, (r0, r1) in cases.items():
        plan = TP.plan_transition(P, r0, r1)
        d = plan["diag"]
        lines.append(f"case {name}")
        lines.append("rest0 " + fmt(r0["x_b"]) + f" {r0['phi']!r} " + fmt(r0["q"]))
        lines.append("rest1 " + fmt(r1["x_b"]) + f" {r1['phi']!r} " + fmt(r1["q"]))
        lines.append(f"T {plan['T']!r}")
        lines.append(f"diag {d['iterations']} {d['max_dyn_defect']!r} "
                     f"{d['min_sigma_nd']!r} {d['peak_com_speed']!r} "
                     f"{d['endpoint_mismatch']!r}")
        ts = np.linspace(0.0, plan["T"], 41)
        lines.append(f"samples {len(ts)}")
        for t in ts:
            ref = plan["ref"](t)
            lines.append(f"{t!r} " + " ".join(fmt(ref[k]) for k in keys))
        print(f"transition '{name}': T={plan['T']:.4f} defect "
              f"{d['max_dyn_defect']:.2e}")
    with open(os.path.join(OUT, "python_transition_t650.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    dump_kinematics()
    dump_transition()
