"""Refresh and membership (Section 7.2): payload-space test along the 24 snapshots at 90% of the first rate, change of
the aggregate when the previous snapshot's summaries are used, and removal of one robot by the cached-summary update.
"""
import json, numpy as np
from teamdata import RESULTS, Star, mission_gains
out = {}
for name in ['team2', 'team3', 'ridgeback2', 'ridgeback3']:
    K, *_ = mission_gains(name)
    qs = np.load(RESULTS / f'snapshots_{name}.npy')
    stars = [Star(name, q=q) for q in qs]
    k_first = stars[0].rate(K)
    kap = 0.9 * k_first
    feas, stale = [], []
    for j, st in enumerate(stars):
        Xis, S, ok = st.aggregate(K, kap)
        feas.append(bool(ok and np.linalg.eigvalsh(S).min() >= 0))
        if j > 0:
            prev = stars[j - 1]
            Xp, Sp, _ = prev.aggregate(K, kap)
            # stale: previous-frame summaries with the current payload inertia
            S_stale = sum(Xp.values()) - kap * st.M0
            stale.append(float(np.linalg.norm(S_stale - S) / np.linalg.norm(S)))
    # interface-only maximum vs global at every snapshot
    dev = max(abs(st.interface_rate(K) - st.rate(K)) / st.rate(K) for st in stars[::6])
    out[name] = dict(target=kap, all_feasible=all(feas), max_stale_change=max(stale), interface_vs_global_rel=dev)
    print(name, json.dumps(out[name]), flush=True)
# membership: remove each robot at nominal configuration, target 0.95 of team rate
mem = {}
for name in ['team3', 'ridgeback3', 'team2', 'ridgeback2']:
    K, *_ = mission_gains(name); st = Star(name); kt = st.rate(K); tgt = 0.95 * kt
    res = {}
    for r in st.robots:
        Xis, S, ok = st.aggregate(K, tgt)
        S_rem = S - Xis[r]
        # maximum rate of remaining cells by bisection on the aggregate without r
        lo, hi = 0., 2 * kt
        for _ in range(80):
            mid = .5 * (lo + hi)
            X2, S2, ok2 = st.aggregate(K, mid)
            S2r = S2 - X2[r]
            if ok2 and np.linalg.eigvalsh(S2r).min() >= -1e-10: lo = mid
            else: hi = mid
        res[r] = dict(max_rate_without=lo, target=tgt, retained=lo >= tgt, lmin_remove=float(np.linalg.eigvalsh(np.linalg.solve(np.linalg.cholesky(st.M0), np.linalg.solve(np.linalg.cholesky(st.M0), S_rem).T)).min()))
    mem[name] = res
    print(name, {r: (round(v['max_rate_without'], 4), round(v['target'], 4), v['retained']) for r, v in res.items()}, flush=True)
json.dump(dict(refresh=out, membership=mem), open(RESULTS / 'refresh_membership.json', 'w'), indent=1)
