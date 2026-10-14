"""Theorem 19(a) (references in actuator space) checked pointwise on the compiled teams with ideal edges.

Law of joint-level loops:   tau = tau_f - K (B^T v - w),   w in R^p arbitrary required actuator rates.
With  u_r = (L^T K L)^-1 L^T K w  (the fit of w by admissible rates in the metric K),  w_perp = w - L u_r:
  (a)  L^T K w_perp = 0                                  (K w_perp is an internal effort)
       tau = (tau_f + K w_perp) - K B^T (v - E u_r)      (the law of the theory for the admissible reference)
  (b)  the acceleration is the same with and without K w_perp
  (c)  B K w_perp lies in range A^T, and the total constraint force changes by - B K w_perp
Also recorded: how far a random w is from the admissible rates, and that u_r minimises |w - L u|_K.
"""
import json
import time

import numpy as np

from lib import RESULTS, ROLLING, Team

rng = np.random.default_rng(20261008)
out = {}
for name in ROLLING:
    t0 = time.time()
    T = Team(name)
    mode = T.mode()
    q = T.q0.copy()
    worst = dict(internal=0., law=0., accel=0., in_range=0., reaction=0., minimiser=0.)
    share = []
    for trial in range(12):
        if trial in (4, 8):                                  # another closed configuration
            u0 = .15 * rng.standard_normal(T.E(q, mode).shape[1])
            dt = .02
            for _ in range(10):
                k1 = T.E(q, mode) @ u0
                k2 = T.E(q + .5 * dt * k1, mode) @ u0
                k3 = T.E(q + .5 * dt * k2, mode) @ u0
                k4 = T.E(q + dt * k3, mode) @ u0
                q = q + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        A, con, keep = T.rows(q, mode)
        E = T.E(q, mode)
        d = E.shape[1]
        L = T.B.T @ E
        assert np.linalg.matrix_rank(L) == d
        M, g, D = T.M(q), T.g(q), T.D
        K = T.gains(q) * np.diag(rng.uniform(.3, 3., T.nf))  # any positive diagonal gain
        u = .3 * rng.standard_normal(d)
        v = E @ u
        Ed = T.Edot(q, v, mode)
        w = .3 * rng.standard_normal(T.nf)                   # required actuator rates, not admissible in general
        tau_f = T.model.effort_bounds[:, 1] * .2 * rng.standard_normal(T.nf)
        LKL = L.T @ K @ L
        ur = np.linalg.solve(LKL, L.T @ K @ w)
        wp = w - L @ ur
        internal = K @ wp
        nK = lambda x: float(np.sqrt(x @ K @ x))
        worst['internal'] = max(worst['internal'], float(np.max(np.abs(L.T @ internal)) / np.max(np.abs(L.T @ (K @ w)))))
        tau_joint = tau_f - K @ (T.B.T @ v - w)
        tau_theory = (tau_f + internal) - K @ (T.B.T @ (v - E @ ur))
        worst['law'] = max(worst['law'], float(np.max(np.abs(tau_joint - tau_theory)) / np.max(np.abs(tau_joint))))
        hv = T.b(q, v) + g + D @ v
        Mr = E.T @ M @ E

        def plant(tau):
            ud = np.linalg.solve(Mr, E.T @ (T.B @ tau - hv - M @ (Ed @ u)))
            a = E @ ud + Ed @ u
            return a, M @ a + hv - T.B @ tau                  # acceleration and total constraint force A^T lam

        a1, c1 = plant(tau_joint)
        a0, c0 = plant(tau_joint - internal)                 # the same law with w replaced by L u_r
        worst['accel'] = max(worst['accel'], float(np.max(np.abs(a1 - a0)) / np.max(np.abs(a1))))
        f = T.B @ internal
        mu = np.linalg.lstsq(A.T, f, rcond=None)[0]
        worst['in_range'] = max(worst['in_range'], float(np.max(np.abs(A.T @ mu - f)) / np.max(np.abs(f))))
        worst['reaction'] = max(worst['reaction'], float(np.max(np.abs((c1 - c0) + f)) / np.max(np.abs(f))))
        # u_r minimises |w - L u|_K: no admissible perturbation lowers the distance
        base = nK(wp)
        for _ in range(20):
            du = 1e-3 * rng.standard_normal(d)
            worst['minimiser'] = max(worst['minimiser'], (base - nK(w - L @ (ur + du))) / base)
        share.append(base / nK(w))
    out[name] = dict(n=int(T.n), p=int(T.nf), d=int(d), trials=12, worst=worst,
                     inadmissible_share=[float(min(share)), float(max(share))], seconds=round(time.time() - t0, 1))
    print(name, json.dumps(out[name]))
(RESULTS / 'exp7_actuator_reference.json').write_text(json.dumps(out, indent=1))
