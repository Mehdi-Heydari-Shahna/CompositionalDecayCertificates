"""Edge arity (Remark 15): cells = all (m-1)-subsets of m modules (no join tree), every edge joining TWO modules.

For random assemblies we compare the centralized rate kl = lambda_min(E'DE, E'ME) with the best rate of a modular
certificate (SDP over all cell forms). Remark 15 states that they coincide (certificates exist for every kappa < kl).
Contrast: with the (m-1)-module edges of Theorem 13, case (B), the best modular rate is 0 (verify_new_theorems.py).
"""
import itertools
import json
import os
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh, null_space

RESULTS = Path(os.environ.get('ANALYSIS_RESULTS', Path(__file__).resolve().parents[2] / 'results' / 'analysis')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def spd(n):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q @ np.diag(rng.uniform(.5, 3., n)) @ Q.T


def best_modular(cells, rows_of_cell, D, M, offs):
    n = D.shape[0]
    k = cp.Variable()
    tot, cons = 0, []
    for c, cell in enumerate(cells):
        idx = np.concatenate([np.arange(offs[j], offs[j + 1]) for j in cell])
        P = np.zeros((len(idx), n))
        P[np.arange(len(idx)), idx] = 1.
        th = cp.Variable((len(idx), len(idx)), symmetric=True)
        R = rows_of_cell.get(c)
        N = null_space(R[:, idx]) if R is not None else np.eye(len(idx))
        if N.shape[1]:
            cons.append(N.T @ th @ N >> 0)
        tot = tot + P.T @ th @ P
    cons += [tot == D - k * M, k <= 1e3]
    prob = cp.Problem(cp.Maximize(k), cons)
    for solver, opts in (('CLARABEL', {}), ('SCS', dict(eps=1e-9, max_iters=200000))):
        try:
            prob.solve(solver=solver, **opts)
            if k.value is not None:
                return solver, prob.status, float(k.value)
        except cp.error.SolverError:
            continue
    return None, 'failed', float('nan')


def trial(m, dim=2):
    offs = np.cumsum([0] + [dim] * m)
    n = offs[-1]
    M, D = np.zeros((n, n)), np.zeros((n, n))
    for j in range(m):
        s = slice(offs[j], offs[j + 1])
        M[s, s] = spd(dim)
        G = rng.standard_normal((dim, int(rng.integers(1, dim + 1))))     # rank-deficient damping allowed
        D[s, s] = G @ G.T
    cells = [list(c) for c in itertools.combinations(range(m), m - 1)]
    pairs = list(itertools.combinations(range(m), 2))
    rows_of_cell = {}
    for i in rng.choice(len(pairs), int(rng.integers(3, len(pairs) + 1)), replace=False):
        a, b = pairs[i]
        R = np.zeros((1, n))
        R[0, offs[a]:offs[a + 1]] = rng.standard_normal(dim)
        R[0, offs[b]:offs[b + 1]] = rng.standard_normal(dim)
        c = int(rng.choice([c for c, cell in enumerate(cells) if a in cell and b in cell]))
        rows_of_cell[c] = np.vstack([rows_of_cell[c], R]) if c in rows_of_cell else R
    E = null_space(np.vstack(list(rows_of_cell.values())))
    if E.shape[1] == 0:
        return None
    kl = float(eigh(E.T @ D @ E, E.T @ M @ E, eigvals_only=True)[0])
    if kl < 1e-3:
        return None
    solver, status, kb = best_modular(cells, rows_of_cell, D, M, offs)
    return dict(m=m, edges=int(sum(len(r) for r in rows_of_cell.values())), d=int(E.shape[1]), kappa=kl,
                best_modular=kb, relative_gap=(kl - kb) / kl, solver=solver, status=status)


if __name__ == '__main__':
    out = []
    for m, trials in ((4, 200), (5, 60)):
        for _ in range(trials):
            r = trial(m)
            if r is not None:
                out.append(r)
    ok = [r for r in out if r['solver'] is not None]
    summary = dict(instances=len(out), solved=len(ok),
                   per_m={m: sum(r['m'] == m for r in ok) for m in (4, 5)},
                   max_relative_gap=max(r['relative_gap'] for r in ok),
                   min_relative_gap=min(r['relative_gap'] for r in ok),
                   statuses=sorted({f"{r['solver']}:{r['status']}" for r in ok}))
    print(json.dumps(summary, indent=1))
    (RESULTS / 'verify_edge_arity.json').write_text(json.dumps(dict(summary=summary, records=out), indent=1))
