"""Proposition 3 (error-power identity) checked pointwise on the compiled teams with ideal edges.

For random states on the constraint manifold and random references:
  consistent reference  (A(q) v_r = 0):   d/dt (1/2 |s|_M^2) = - s^T (B K B^T + D) s
  any reference:                          d/dt (1/2 |s|_M^2) = - s^T (B K B^T + D) s - <A v_r, lam - lam_r>
  lower bound:                            |s|_M >= |A v_r|_{N^+},  N = A M^-1 A^T
The reactions lam (plant) and lam_r (required balance) are obtained independently by least squares.
"""
import json
import time

import numpy as np

from lib import RESULTS, ROLLING, Team

rng = np.random.default_rng(20261007)
out = {}
for name in ROLLING:
    t0 = time.time()
    T = Team(name)
    mode = T.mode()
    rows = []
    q = T.q0.copy()
    for trial in range(12):
        if trial in (4, 8):
            # another closed configuration: follow a random admissible motion for a short time (RK4 on q' = E(q) u)
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
        u, ur, urd = .3 * rng.standard_normal(d), .3 * rng.standard_normal(d), rng.standard_normal(d)
        v = E @ u
        Ed = T.Edot(q, v, mode)
        M, g, K, D = T.M(q), T.g(q), T.gains(q), T.D
        Dm = T.B @ K @ T.B.T + D
        BA = np.hstack([T.B, A.T])

        def run(vr, ar):
            rhs = M @ ar + T.C(q, v, vr) + g + D @ vr
            x = np.linalg.lstsq(BA, rhs, rcond=None)[0]
            tau_r, lam_r = x[:T.nf], x[T.nf:]
            res_r = float(np.max(np.abs(BA @ x - rhs)) / np.max(np.abs(rhs)))
            s = v - vr
            tau = tau_r - K @ (T.B.T @ s)
            hv = T.b(q, v) + g + D @ v
            ud = np.linalg.solve(E.T @ M @ E, E.T @ (T.B @ tau - hv - M @ (Ed @ u)))
            a = E @ ud + Ed @ u
            gen = M @ a + hv - T.B @ tau
            lam = np.linalg.lstsq(A.T, gen, rcond=None)[0]
            res_p = float(np.max(np.abs(A.T @ lam - gen)) / np.max(np.abs(hv)))
            sd = a - ar
            Wdot = float(s @ (M @ sd) + s @ T.C(q, v, s))
            diss = float(s @ Dm @ s)
            react = float((A @ vr) @ (lam - lam_r))
            N = A @ np.linalg.solve(M, A.T)
            lower = float(np.sqrt(max((A @ vr) @ np.linalg.pinv(N, rcond=1e-12) @ (A @ vr), 0.)))
            return dict(Wdot=Wdot, diss=diss, react=react, res_r=res_r, res_p=res_p, sM=float(np.sqrt(s @ M @ s)),
                        lower=lower, Avr=float(np.max(np.abs(A @ vr))), s=s)

        c = run(E @ ur, E @ urd + Ed @ ur)
        # skew symmetry used in the proof: s^T Mdot s = 2 s^T C(q,v) s (Mdot by a centred difference along v)
        h = 1e-6
        Mdot = (T.M(q + h * v) - T.M(q - h * v)) / (2 * h)
        s = c['s']
        skew = abs(float(s @ Mdot @ s - 2 * s @ T.C(q, v, s))) / abs(float(s @ Mdot @ s) + 1e-300)
        # a reference that violates the constraints at the actual configuration
        dv = .05 * rng.standard_normal(T.n)
        i = run(E @ ur + dv, E @ urd + Ed @ ur + rng.standard_normal(T.n))
        rows.append(dict(trial=trial, consistent_rel=abs(c['Wdot'] + c['diss']) / c['diss'], consistent_Avr=c['Avr'],
                         skew_rel=skew, balance_res=max(c['res_r'], c['res_p'], i['res_r'], i['res_p']),
                         inconsistent_rel=abs(i['Wdot'] + i['diss'] + i['react']) / (abs(i['diss']) + abs(i['react'])),
                         inconsistent_reaction_over_dissipation=abs(i['react']) / i['diss'],
                         lower_bound_ratio=i['sM'] / i['lower'], constraint_residual=float(np.max(np.abs(con.residual)))))
    out[name] = dict(trials=rows,
                     max_consistent_rel=max(r['consistent_rel'] for r in rows),
                     max_inconsistent_rel=max(r['inconsistent_rel'] for r in rows),
                     max_skew_rel=max(r['skew_rel'] for r in rows),
                     max_balance_res=max(r['balance_res'] for r in rows),
                     min_lower_bound_ratio=min(r['lower_bound_ratio'] for r in rows),
                     reaction_term_range=[min(r['inconsistent_reaction_over_dissipation'] for r in rows),
                                          max(r['inconsistent_reaction_over_dissipation'] for r in rows)],
                     seconds=round(time.time() - t0, 1))
    print(name, {k: v for k, v in out[name].items() if k != 'trials'}, flush=True)
(RESULTS / 'exp1_identity.json').write_text(json.dumps(out, indent=1))
