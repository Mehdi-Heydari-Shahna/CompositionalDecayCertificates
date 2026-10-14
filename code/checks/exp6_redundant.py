"""Dependent constraint rows on a compiled team (Lemma 2, the pseudo-inverse form of P, Proposition 3 for rank-deficient A).

The six compiled teams have independent ideal rows: the compiler describes the ground under a four-wheel base by
three rows of a fitted plane.  Here the ground of team2 is described by one contact row per wheel instead (the
height of the base-fixed point under every wheel centre), which gives 28 rows of rank 26.  Checked with these rows:
  * the compiled lift still spans their kernel;
  * P = I - M^-1 A^T (A M^-1 A^T)^+ A equals E (E^T M E)^-1 E^T M;
  * Lemma 2: which rows are identifiable (rank test), and that A^T lam is the same for two different multipliers;
  * Proposition 3(i): the identity with reactions from two different selections of the multipliers;
  * Lemma 21: the singular impulse equations give P+ v-.
"""
import json

import numpy as np

from lib import RESULTS, Team, m_projector

rng = np.random.default_rng(31)
T = Team('team2')
model, comp = T.model, T.comp
q = T.q0
G = T.mode()
lift = model.lift(q, G)
E, con = lift.E, lift.constraints
M = T.M(q)
n, d = T.n, E.shape[1]
ez = np.array([0., 0., 1.])
Tw, S = T.tree.sweep(q)
keep, contact = [], []
for b in con.blocks:
    if b['kind'] == 'ground':
        rid = b['robot']
        ib = model.base_body[rid]
        Xb = model._twist_map(ib, S)
        R, o = Tw[ib][:3, :3], Tw[ib][:3, 3]
        for w in comp.robots[rid].wheels:
            p = o + R @ np.asarray(w['centre_base'], float) - w['radius_m'] * R[:, 2]
            contact.append(np.cross(p, ez) @ Xb[:3] + ez @ Xb[3:])
    else:
        keep += list(range(b['rows'].start, b['rows'].stop))
A = np.vstack([con.J[keep], np.array(contact)])
m = A.shape[0]
sv = np.linalg.svd(A, compute_uv=False)
r = int(np.sum(sv > 1e-10 * sv[0]))
labels = ['grasp/wheel'] * len(keep) + ['contact'] * len(contact)
ident = [bool(r == np.linalg.matrix_rank(np.delete(A, j, axis=0), tol=1e-10 * sv[0]) + 1) for j in range(m)]
N = A @ np.linalg.solve(M, A.T)
P1 = np.eye(n) - np.linalg.solve(M, A.T) @ np.linalg.pinv(N, rcond=1e-12) @ A
P2 = m_projector(M, E)
null = np.linalg.svd(A.T)[2][r:].T                         # basis of ker A^T
assert null.shape[1] == m - r
K, D = T.gains(q, 10., G), T.D
Dm = T.B @ K @ T.B.T + D
BA = np.hstack([T.B, A.T])
worst = dict(identity_consistent=0., identity_inconsistent=0., reaction_force_two_selections=0., identity_two_selections=0., impulse=0.)
for trial in range(12):
    u, ur, urd = .3 * rng.standard_normal(d), .3 * rng.standard_normal(d), rng.standard_normal(d)
    v = E @ u
    Ed = T.Edot(q, v, G)
    g = T.g(q)
    for bad in (0., .05):
        vr = E @ ur + bad * rng.standard_normal(n)
        ar = E @ urd + Ed @ ur + bad * rng.standard_normal(n)
        rhs = M @ ar + T.C(q, v, vr) + g + D @ vr
        x = np.linalg.lstsq(BA, rhs, rcond=None)[0]
        tau_r, lam_r = x[:T.nf], x[T.nf:]
        s = v - vr
        tau = tau_r - K @ (T.B.T @ s)
        hv = T.b(q, v) + g + D @ v
        ud = np.linalg.solve(E.T @ M @ E, E.T @ (T.B @ tau - hv - M @ (Ed @ u)))
        a = E @ ud + Ed @ u
        gen = M @ a + hv - T.B @ tau
        lam = np.linalg.lstsq(A.T, gen, rcond=None)[0]
        assert np.max(np.abs(A.T @ lam - gen)) < 1e-8 * np.max(np.abs(hv))
        Wdot = float(s @ (M @ (a - ar)) + s @ T.C(q, v, s))
        diss = float(s @ Dm @ s)
        react = float((A @ vr) @ (lam - lam_r))
        key = 'identity_inconsistent' if bad else 'identity_consistent'
        worst[key] = max(worst[key], abs(Wdot + diss + react) / (abs(diss) + abs(react)))
        # other multipliers: add elements of ker A^T to both; nothing may change
        lam2 = lam + 1e3 * null @ rng.standard_normal(m - r)
        lam_r2 = lam_r + 1e3 * null @ rng.standard_normal(m - r)
        worst['reaction_force_two_selections'] = max(worst['reaction_force_two_selections'],
                                                     float(np.max(np.abs(A.T @ (lam2 - lam_r2) - A.T @ (lam - lam_r))) / np.max(np.abs(A.T @ lam))))
        react2 = float((A @ vr) @ (lam2 - lam_r2))
        worst['identity_two_selections'] = max(worst['identity_two_selections'], abs(react2 - react) / (abs(diss) + abs(react)))
    # plastic closing onto these rows from the mode in which the last robot has released
    Rm = T.mode(grasps=tuple(g_ for g_ in G.grasps if comp.connections[g_]['robot'] != T.robots[-1]))
    vm = model.lift(q, Rm).E @ rng.standard_normal(model.lift(q, Rm).E.shape[1])
    KKT = np.block([[M, -A.T], [A, np.zeros((m, m))]])
    sol = np.linalg.lstsq(KKT, np.r_[M @ vm, np.zeros(m)], rcond=None)[0]
    worst['impulse'] = max(worst['impulse'], float(np.max(np.abs(sol[:n] - P1 @ vm)) / np.max(np.abs(vm))))
out = dict(team='team2', rows=int(m), rank=r, d=int(d), kernel_defect=float(np.max(np.abs(A @ E))),
           projector_defect=float(np.max(np.abs(P1 - P2))),
           identifiable_rows=int(sum(ident)), not_identifiable=sorted(set(l for l, i in zip(labels, ident) if not i)),
           not_identifiable_count=int(m - sum(ident)), contact_rows=len(contact), worst=worst)
assert out['not_identifiable'] == ['contact'] and out['not_identifiable_count'] == len(contact)
print(json.dumps(out))
(RESULTS / 'exp6_redundant.json').write_text(json.dumps(out, indent=1))
