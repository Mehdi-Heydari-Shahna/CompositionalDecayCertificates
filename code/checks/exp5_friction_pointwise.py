"""Proposition 23 (friction-driven bases) checked pointwise on the compiled Husky teams.

  * rank test on the ideal edges: rank L = d - 3; ker L = planar rigid motions of the whole formation
  * friction law of the package at given wheel loads: monotone in the slip, wheel by wheel
  * identity with the reference friction at the ACTUAL loads:
        d/dt (1/2 |s|_M^2) = - s^T (B K B^T + D) s - s^T D_f s,   D_f = sum mu N S^T [int_0^1 DPhi] S  >= 0
  * with the reference friction at other loads the extra term has no sign
  * rate with friction: lambda_min(E^T (B K B^T + D + D_f(v)) E, E^T M E) > 0
"""
import json

import numpy as np

from lib import RESULTS, Team, gen_eig

rng = np.random.default_rng(3)
out = {}
from scipy.integrate import quad_vec


def formation_basis(T, q, ind):
    """Planar rigid motions of the whole team (actuators locked), in the independent coordinates."""
    Z = np.zeros((len(ind), 3))
    pos = {i: k for k, i in enumerate(ind)}
    groups = [T.comp.robots[r].base for r in T.robots] + [T.payload]
    for base in groups:
        x, y = q[base[0]], q[base[1]]
        Z[pos[base[0]]] = [1., 0., -y]
        Z[pos[base[1]]] = [0., 1., x]
        Z[pos[base[3]]] = [0., 0., 1.]
    return Z


for name in ('husky2', 'husky3'):
    T = Team(name)
    model = T.model
    q = T.q0
    mode = T.mode()
    lift = model.lift(q, mode)
    E, ind = lift.E, lift.independent
    d = E.shape[1]
    A, con, keep = T.rows(q, mode)
    M = T.M(q)
    Mr = E.T @ M @ E
    L = T.B.T @ E
    sv = np.linalg.svd(L, compute_uv=False)
    rankL = int(np.sum(sv > 1e-10 * sv[0]))
    Z = formation_basis(T, q, ind)
    kin = T.tree.sweep(q)
    K = np.eye(T.nf)
    seen = np.diag(np.linalg.pinv(L).T @ Mr @ np.linalg.pinv(L))
    K = np.diag(10. * np.maximum(seen, 1e-6 * seen.max()))
    Dm = T.B @ K @ T.B.T + T.D
    N0 = {rid: model.standing_loads(q, rid, kin) for rid in T.robots}
    tyre = model._tyre

    def Q(v, loads):
        return sum(model.friction_loads(q, v, rid, loads[rid], kin)[0] for rid in T.robots)

    def secant(vr, s, loads):
        """D_f, the secant damping of Proposition 23(a): sum mu N S^T [int_0^1 DPsi(S (v_r + theta s)) d theta] S, by adaptive quadrature.

        The integrand of a wheel peaks where its slip passes closest to zero (width: the slip speed of the law),
        so that point is given to the quadrature.
        """
        Df = np.zeros((T.n, T.n))
        for rid in T.robots:
            s0, rows, cols = model.wheel_slip(q, vr, rid, kin)
            ds = model.wheel_slip(q, s, rid, kin)[0]
            t = tyre[rid]
            for k in range(len(s0)):
                S = rows[cols[k]]
                d2 = t['slip_speed'] ** 2

                def jac(theta, a=s0[k], b=ds[k]):
                    x = a + theta * b
                    r2 = x @ x + d2
                    return ((np.eye(2) * r2 - np.outer(x, x)) / r2 ** 1.5).ravel()

                near = -float(s0[k] @ ds[k]) / max(float(ds[k] @ ds[k]), 1e-300)
                pts = [near] if 0. < near < 1. else None
                J = quad_vec(jac, 0., 1., epsabs=1e-12, epsrel=1e-11, limit=2000, points=pts)[0].reshape(2, 2)
                Df += t['mu'][k] * max(loads[rid][k], 0.) * S.T @ J @ S
        return Df

    def tangent(v, loads):
        """Tangent form sum mu N S^T DPsi(S v) S for the smoothed Coulomb law of the package (at one velocity)."""
        Df = np.zeros((T.n, T.n))
        for rid in T.robots:
            slip, rows, cols = model.wheel_slip(q, v, rid, kin)
            t = tyre[rid]
            for k in range(len(slip)):
                S = rows[cols[k]]
                r2 = slip[k] @ slip[k] + t['slip_speed'] ** 2
                J = (np.eye(2) * r2 - np.outer(slip[k], slip[k])) / r2 ** 1.5
                Df += t['mu'][k] * max(loads[rid][k], 0.) * S.T @ J @ S
        return Df

    worst = dict(monotone_max=-np.inf, identity=0., secant=0., other_loads_sign=[np.inf, -np.inf])
    rates, rates_tangent = [], []
    for trial in range(20):
        loads = {rid: N0[rid] * rng.uniform(.3, 1.7, len(N0[rid])) for rid in T.robots}
        u, ur = .3 * rng.standard_normal(d), .3 * rng.standard_normal(d)
        v, vr = E @ u, E @ ur
        Ed = T.Edot(q, v, mode)
        g = T.g(q)
        # wheel by wheel monotonicity of the law at the same loads
        for rid in T.robots:
            s1, rows, cols = model.wheel_slip(q, v, rid, kin)
            s2 = model.wheel_slip(q, vr, rid, kin)[0]
            t = tyre[rid]
            from teamc.skid import friction_forces
            F1, F2 = friction_forces(s1, loads[rid], t['mu'], t['slip_speed']), friction_forces(s2, loads[rid], t['mu'], t['slip_speed'])
            worst['monotone_max'] = max(worst['monotone_max'], float(np.max(((s1 - s2) * (F1 - F2)).sum(axis=1))))
        # a reference that solves the virtual balance: the acceleration along ker L is given by the friction of the reference
        urd0 = rng.standard_normal(d)
        Qr = Q(vr, loads)
        rest = E.T @ (M @ (Ed @ ur) + T.C(q, v, vr) + g + T.D @ vr - Qr)
        c = -np.linalg.solve(Z.T @ Mr @ Z, Z.T @ (Mr @ urd0 + rest))
        urd = urd0 + Z @ c
        ar = E @ urd + Ed @ ur
        rhs = M @ ar + T.C(q, v, vr) + g + T.D @ vr - Qr
        x = np.linalg.lstsq(np.hstack([T.B, A.T]), rhs, rcond=None)[0]
        res = float(np.max(np.abs(np.hstack([T.B, A.T]) @ x - rhs)) / np.max(np.abs(rhs)))
        tau_r = x[:T.nf]
        s = v - vr
        tau = tau_r - K @ (T.B.T @ s)
        Qv = Q(v, loads)
        hv = T.b(q, v) + g + T.D @ v
        ud = np.linalg.solve(Mr, E.T @ (T.B @ tau - hv - M @ (Ed @ u) + Qv))
        a = E @ ud + Ed @ u
        Wdot = float(s @ (M @ (a - ar)) + s @ T.C(q, v, s))
        fric = float(s @ (Qv - Qr))
        worst['identity'] = max(worst['identity'], abs(Wdot + s @ Dm @ s - fric) / (abs(s @ Dm @ s) + abs(fric)), res)
        # secant form of the friction term
        Dsec = secant(vr, s, loads)
        worst['secant'] = max(worst['secant'], abs(fric + s @ Dsec @ s) / abs(fric))
        # the reference friction at other loads: the extra term has no sign
        other = {rid: loads[rid] * rng.uniform(.5, 1.5, len(loads[rid])) for rid in T.robots}
        extra = float(s @ (Qr - Q(vr, other)))
        worst['other_loads_sign'] = [min(worst['other_loads_sign'][0], extra), max(worst['other_loads_sign'][1], extra)]
        rates.append(float(gen_eig(E.T @ (Dm + Dsec) @ E, Mr)[0]))                 # kappa_mu of Proposition 23(b)
        rates_tangent.append(float(gen_eig(E.T @ (Dm + tangent(v, loads)) @ E, Mr)[0]))
    ev0 = gen_eig(E.T @ Dm @ E, Mr)
    stiff = gen_eig(E.T @ (Dm + tangent(np.zeros(T.n), N0)) @ E, Mr)
    out[name] = dict(d=int(d), nf=T.nf, rank_L=rankL, deficiency=int(d - rankL), LZ=float(np.max(np.abs(L @ Z))),
                     rate_without_friction=float(ev0[0]), zero_rates_without_friction=int(np.sum(ev0 < 1e-9 * ev0[-1])),
                     rate_with_friction_range=[min(rates), max(rates)], rate_with_tangent_friction_range=[min(rates_tangent), max(rates_tangent)], fastest_rate_at_rest=float(stiff[-1]),
                     slowest_rate_at_rest=float(stiff[0]),
                     slip_speed={rid: float(tyre[rid]['slip_speed']) for rid in T.robots},
                     mu={rid: [float(x) for x in tyre[rid]['mu']] for rid in T.robots},
                     standing_loads={rid: [float(x) for x in N0[rid]] for rid in T.robots}, worst=worst)
    print(name, json.dumps(out[name]), flush=True)
(RESULTS / 'exp5_friction_pointwise.json').write_text(json.dumps(out, indent=1))
