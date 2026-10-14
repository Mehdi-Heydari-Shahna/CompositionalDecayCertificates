"""Section 7.2 and Fig. 6(b): centralized rate, largest rate passing the payload-space test (Proposition 7), equal and optimized
scalar shares (Theorem 8(a)), and checks of the equality-normalized supplies, at the nominal carrying configurations.
"""
import json, time
import numpy as np
from teamdata import RESULTS, Star, mission_gains, TEAMS

out = {}
for name in TEAMS:
    t0 = time.time()
    st = Star(name)
    K, Kp, jp, wp = mission_gains(name)
    k = st.rate(K)
    ki = st.interface_rate(K)
    N = len(st.robots)
    eq = min(st.scalar_share_rate(r, K, 1. / N) for r in st.robots)
    # optimized scalar shares: equalize standalone rates by bisection on the common rate
    def share_for(r, kap):
        lo, hi = 0., 1.
        if st.scalar_share_rate(r, K, 1e-12) < kap: return None
        if st.scalar_share_rate(r, K, 1.) >= kap: return 1.
        for _ in range(60):
            mid = .5 * (lo + hi)
            if st.scalar_share_rate(r, K, mid) >= kap: lo = mid
            else: hi = mid
        return lo
    lo, hi = 0., min(st.scalar_share_rate(r, K, 1e-12) for r in st.robots)
    for _ in range(60):
        mid = .5 * (lo + hi); sh = [share_for(r, mid) for r in st.robots]
        if all(s is not None for s in sh) and sum(sh) >= 1.: lo = mid
        else: hi = mid
    real = all(st.realizable(r) for r in st.robots)
    f = {r: st.summary(r, K, k)[2] for r in st.robots}
    # equality-normalized supplies at the centralized rate (Theorem 8(b)): Theta_i = Xi_i - S/N, sum = kl M0
    Xis, S, ok = st.aggregate(K, k)
    Th = {r: Xis[r] - S / N for r in st.robots}
    defect = float(np.linalg.norm(sum(Th.values()) - k * st.M0) / np.linalg.norm(k * st.M0))
    local_min, indefinite = {}, {}
    for r in st.robots:
        T = st.local[r]['T']
        Q = np.zeros((T.shape[0], T.shape[0]))
        Q[:6, :6] = -Th[r]
        Q[6:, 6:] = st.local_form(r, K, k)
        F = T.T @ Q @ T
        F = .5 * (F + F.T)
        ev = np.linalg.eigvalsh(F)
        local_min[r] = float(ev.min() / np.abs(ev).max())
        et = np.linalg.eigvalsh(.5 * (Th[r] + Th[r].T))
        indefinite[r] = bool(et.min() < 0. < et.max())
    supplies = dict(kappa=k, aggregate_feasible=bool(ok), min_eig_S=float(np.linalg.eigvalsh(S).min()), sum_defect=defect,
                    normalized_local_min_eig=local_min, indefinite=indefinite)
    out[name] = dict(d=st.d, p=st.nf, rate=k, interface_rate=ki, equal_shares=eq, optimized_scalar=lo, realizable=real,
                     self_motion_orders=f, supplies_at_centralized_rate=supplies)
    print(name, json.dumps(out[name]), round(time.time() - t0, 1), 's', flush=True)
json.dump(out, open(RESULTS / 'table2_check.json', 'w'), indent=1)
