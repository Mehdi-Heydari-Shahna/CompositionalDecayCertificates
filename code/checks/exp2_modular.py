"""Theorems 2 and 3 on the compiled teams: rank tests, exact rate, modular certificates.

For every team with ideal edges, at its carrying configuration:
  * the compile-time tests: rank A, rank L = d (locked-actuator rigidity), dim ker L^T, [B A^T] onto;
  * gains designed robot by robot (each robot alone with a share of the payload inertia);
  * exact team rate  k = lambda_min(E^T (B K B^T + D) E, E^T M E);
  * modular lower bounds: equal shares, optimal scalar shares (rates equalised by bisection);
  * exact decomposition with symmetric matrix shares (Schur complement construction): sum S_i = M_0 and every
    module certificate positive semidefinite at the exact team rate;
  * plug-and-play: rate of the team against min(rate of the team without one robot, zero-share rate of that robot).
"""
import json

import numpy as np

from lib import RESULTS, ROLLING, Team, gen_eig, psd_min

RATE = 10.
out = {}


def module_data(T, q, mode, grasps):
    """Reduced matrices in the coordinates (payload velocity; own independent velocities) of every robot that grasps."""
    lift = T.model.lift(q, mode)
    E, ind = lift.E, list(lift.independent)
    M = T.M(q)
    pay = [ind.index(i) for i in T.payload]
    assert len(pay) == 6, 'the payload coordinates are independent in a carrying mode'
    mods = {}
    for rid in T.robots:
        own = [k for k, i in enumerate(ind) if i in set(T.coords[rid])]
        c = T.coords[rid]
        Ei = E[np.ix_(c, pay + own)]
        mods[rid] = dict(E=Ei, M=M[np.ix_(c, c)], B=T.B[np.ix_(c, T.actuators[rid])], D=T.D[np.ix_(c, c)], own=own, coords=c,
                         attached=any(T.comp.connections[g]['robot'] == rid for g in grasps))
        # no coupling of a robot's coordinates with the others in the lift
        other = [k for k in range(len(ind)) if k not in pay + own]
        mods[rid]['leak'] = float(np.max(np.abs(E[np.ix_(c, other)]))) if other else 0.
    M0 = M[np.ix_(T.payload, T.payload)]
    return lift, E, M, pay, mods, M0


def reduced_inertia(m, S):
    Mr = m['E'].T @ m['M'] @ m['E']
    Mr[:6, :6] += S
    return Mr


def local_gain(m, S):
    """Gain of one robot from its own model and its share S of the payload inertia."""
    Mr = reduced_inertia(m, S)
    Lp = np.linalg.pinv(m['B'].T @ m['E'])
    seen = np.diag(Lp.T @ Mr @ Lp)
    return np.diag(RATE * np.maximum(seen, 1e-6 * seen.max()))


def module_rate(m, K, S):
    Dr = m['E'].T @ (m['B'] @ K @ m['B'].T + m['D']) @ m['E']
    return float(gen_eig(Dr, reduced_inertia(m, S))[0]), Dr


for name in ROLLING:
    T = Team(name)
    q = T.q0
    mode = T.mode()
    lift, E, M, pay, mods, M0 = module_data(T, q, mode, mode.grasps)
    A, con, keep = T.rows(q, mode)
    N = len(T.robots)
    d = E.shape[1]
    L = T.B.T @ E
    sv = np.linalg.svd(L, compute_uv=False)
    tests = dict(n=T.n, rows=int(A.shape[0]), rank_A=int(np.linalg.matrix_rank(A, tol=1e-9)), d=int(d), nf=T.nf,
                 rank_L=int(np.sum(sv > 1e-10 * sv[0])), sigma_min_L=float(sv[-1]), dim_ker_LT=int(T.nf - np.sum(sv > 1e-10 * sv[0])),
                 rank_BA=int(np.linalg.matrix_rank(np.hstack([T.B, A.T]), tol=1e-9)),
                 lift_leak=max(m['leak'] for m in mods.values()))
    # decentralized gain design: every robot alone, with an equal share of the payload inertia
    K = np.zeros((T.nf, T.nf))
    Ki = {}
    for rid, m in mods.items():
        Ki[rid] = local_gain(m, M0 / N)
        K[np.ix_(T.actuators[rid], T.actuators[rid])] = Ki[rid]
    Dm = T.B @ K @ T.B.T + T.D
    Mr, Dr = E.T @ M @ E, E.T @ Dm @ E
    ev = gen_eig(Dr, Mr)
    k_team = float(ev[0])
    # block-arrow structure: no coupling between the own coordinates of different robots
    arrow = 0.
    for a in T.robots:
        for b in T.robots:
            if a != b:
                arrow = max(arrow, float(np.max(np.abs(Mr[np.ix_(mods[a]['own'], mods[b]['own'])]))),
                            float(np.max(np.abs(Dr[np.ix_(mods[a]['own'], mods[b]['own'])]))))
    equal = {rid: module_rate(m, Ki[rid], M0 / N)[0] for rid, m in mods.items()}
    full = {rid: module_rate(m, Ki[rid], M0)[0] for rid, m in mods.items()}
    zero = {rid: module_rate(m, Ki[rid], 0 * M0)[0] for rid, m in mods.items()}

    # optimal scalar shares: equalise the module rates (each is decreasing in its share)
    def share_for(rid, k):
        lo, hi = 0., 1.
        if module_rate(mods[rid], Ki[rid], lo * M0)[0] < k:
            return None
        if module_rate(mods[rid], Ki[rid], hi * M0)[0] >= k:
            return 1.
        for _ in range(60):
            mid = .5 * (lo + hi)
            if module_rate(mods[rid], Ki[rid], mid * M0)[0] >= k:
                lo = mid
            else:
                hi = mid
        return lo

    lo, hi = 0., min(zero.values())
    for _ in range(60):
        mid = .5 * (lo + hi)
        sh = [share_for(rid, mid) for rid in T.robots]
        if all(x is not None for x in sh) and sum(sh) >= 1.:
            lo = mid
        else:
            hi = mid
    k_scalar = lo
    shares = [share_for(rid, lo) for rid in T.robots]
    # exact decomposition with symmetric matrix shares at the exact team rate
    X = Dr - k_team * Mr
    X00 = X[np.ix_(pay, pay)]
    refl = {}
    for rid, m in mods.items():
        Xi0 = X[np.ix_(m['own'], pay)]
        refl[rid] = Xi0.T @ np.linalg.solve(X[np.ix_(m['own'], m['own'])], Xi0)
    S0 = X00 - sum(refl.values())
    Ssum, worst, share_eigs = np.zeros((6, 6)), np.inf, {}
    for rid, m in mods.items():
        Dri = module_rate(m, Ki[rid], 0 * M0)[1]
        Mri = reduced_inertia(m, 0 * M0)
        Y00 = refl[rid] + S0 / N
        S = ((Dri - k_team * Mri)[:6, :6] - Y00) / k_team
        Ssum += S
        Y = Dri - k_team * reduced_inertia(m, S)
        worst = min(worst, psd_min(Y) / np.linalg.norm(Y, 2))
        share_eigs[rid] = [float(x) for x in np.linalg.eigvalsh(.5 * (S + S.T))]
    out[name] = dict(tests=tests, rate_team=k_team, rate_team_max=float(ev[-1]), block_arrow_defect=arrow,
                     module_rate_equal_share=equal, module_rate_full_payload=full, module_rate_zero_share=zero,
                     bound_equal_shares=min(equal.values()), bound_full_payload=min(full.values()),
                     bound_optimal_scalar_shares=k_scalar, optimal_scalar_shares=shares,
                     schur_complement_min_eig=psd_min(S0) / np.linalg.norm(X00, 2),
                     matrix_shares_sum_defect=float(np.max(np.abs(Ssum - M0)) / np.max(np.abs(M0))),
                     matrix_shares_min_module_eig_rel=float(worst), matrix_share_eigenvalues=share_eigs,
                     payload_inertia_eigenvalues=[float(x) for x in np.linalg.eigvalsh(M0)])
    # plug-and-play: the team without its last robot (that robot released, the others carry)
    if N == 3:
        last = T.robots[-1]
        others = tuple(g for g in mode.grasps if T.comp.connections[g]['robot'] != last)
        m2 = T.mode(grasps=others)
        l2 = T.model.lift(q, m2)
        ind2 = list(l2.independent)
        keep_cols = [k for k, i in enumerate(ind2) if i not in set(T.coords[last])]
        E2 = l2.E[:, keep_cols]                                 # the released robot at rest in the comparison
        k_sub = float(gen_eig(E2.T @ Dm @ E2, E2.T @ M @ E2)[0])
        out[name]['plug_and_play'] = dict(rate_without_last=k_sub, zero_share_rate_of_last=zero[last],
                                          bound=min(k_sub, zero[last]), rate_team=k_team)
    print(name, json.dumps({k: v for k, v in out[name].items() if k not in ('matrix_share_eigenvalues',)}), flush=True)
(RESULTS / 'exp2_modular.json').write_text(json.dumps(out, indent=1))
