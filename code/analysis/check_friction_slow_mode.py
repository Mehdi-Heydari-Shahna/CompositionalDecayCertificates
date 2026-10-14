"""Slowest certified motion of the friction-driven (Husky) teams at zero slip and standing loads (Section 7.8).

Recomputes the rate of Proposition 23(b) at rest, as exp5_friction_pointwise.py of the numerical-checks folder does
(slowest_rate_at_rest: gains K = 10 diag(L^+T Mbar L^+), the smoothed Coulomb law of the compiled model at zero slip and
the standing wheel loads), and reports the generalized eigenvector of the smallest rate: the planar velocity of every
base and of the payload, and the rates of the wheel joints (normalized to the largest entry), together with the
next rates and the wheel gain that the rule gives. The paper's statement that this slowest motion is the team rolling
on gripping wheels, damped only by the wheel gains, rests on this output.

Usage: python3 check_friction_slow_mode.py
Output: check_friction_slow_mode.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.linalg import eigh

from teamdata import CHECKS, PKG, RESULTS  # noqa: F401  (puts the package on sys.path)

os.environ.setdefault('TEAM_CMG_PACDM', str(PKG))           # lib.py of the numerical-checks folder reads it
sys.path.insert(0, str(CHECKS))
from lib import Team  # noqa: E402  (the numerical-checks folder)

out = {}
for name in ('husky2', 'husky3'):
    T = Team(name)
    model = T.model
    q = T.q0
    lift = model.lift(q, T.mode())
    E = lift.E
    M = T.M(q)
    Mr = E.T @ M @ E
    L = T.B.T @ E
    seen = np.diag(np.linalg.pinv(L).T @ Mr @ np.linalg.pinv(L))
    K = np.diag(10. * np.maximum(seen, 1e-6 * seen.max()))
    Dm = T.B @ K @ T.B.T + T.D
    kin = T.tree.sweep(q)
    N0 = {rid: model.standing_loads(q, rid, kin) for rid in T.robots}
    Df = np.zeros((T.n, T.n))                        # tangent friction damping at zero slip (= the secant one there)
    for rid in T.robots:
        slip, rows, cols = model.wheel_slip(q, np.zeros(T.n), rid, kin)
        t = model._tyre[rid]
        for k in range(len(slip)):
            S = rows[cols[k]]
            r2 = slip[k] @ slip[k] + t['slip_speed'] ** 2
            Df += t['mu'][k] * max(N0[rid][k], 0.) * S.T @ ((np.eye(2) * r2 - np.outer(slip[k], slip[k])) / r2 ** 1.5) @ S
    w, V = eigh(E.T @ (Dm + Df) @ E, Mr)
    x = E @ V[:, 0]
    x = x / np.max(np.abs(x))
    robots = {}
    wheel_gains = []
    for rid in T.robots:
        r = T.comp.robots[rid]
        robots[rid] = dict(base_velocity=[float(x[r.base[0]]), float(x[r.base[1]])], base_yaw_rate=float(x[r.base[3]]),
                           heading=float(q[r.base[3]]), wheel_rates=[float(x[wh['index']]) for wh in r.wheels])
        for wh in r.wheels:
            j = int(np.argmax(np.abs(T.B[wh['index']])))
            wheel_gains.append(float(K[j, j]))
    pl = T.payload
    # shares of the dissipation of this motion: wheel ports, other actuator ports, joint damping, friction
    wheel_cols = {int(np.argmax(np.abs(T.B[wh['index']]))) for rid in T.robots for wh in T.comp.robots[rid].wheels}
    y = T.B.T @ x
    parts = dict(wheel_ports=sum(K[j, j] * y[j] ** 2 for j in range(len(y)) if j in wheel_cols),
                 other_ports=sum(K[j, j] * y[j] ** 2 for j in range(len(y)) if j not in wheel_cols),
                 joint_damping=float(x @ T.D @ x), friction=float(x @ Df @ x))
    total = sum(parts.values())
    out[name] = dict(slowest_rates=[float(v) for v in w[:4]], payload_velocity=[float(x[pl[0]]), float(x[pl[1]])],
                     payload_yaw_rate=float(x[pl[3]]), robots=robots,
                     wheel_gain_range=[min(wheel_gains), max(wheel_gains)],
                     dissipation_shares={k: float(v / total) for k, v in parts.items()},
                     rate_check=float(total / (x @ M @ x)))
    print(name, json.dumps(out[name]), flush=True)
(RESULTS / 'check_friction_slow_mode.json').write_text(json.dumps(out, indent=1))
