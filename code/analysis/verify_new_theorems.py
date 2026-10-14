"""Checks of the new statements on synthetic assemblies (NumPy/SciPy/CVXPY only).

(1) Theorem 11(b), join trees with separator realizability: the leaf-to-root generalized-Schur construction gives a
    modular certificate AT the centralized rate; we check sum identity and local nonnegativity on T_c.
(2) Theorem 13, architectures without a join tree: (i) chordless cycles of the primal graph (twisted rings) and
    (ii) minimal uncovered cliques with frustrated sum constraints; the best certified rate (SDP over all modular
    certificates) is ~0 while the centralized rate is kz; a single enclosing cell recovers kz.
(3) Proposition 12 (endpoint failure) is checked in verify_endpoint.py, and Remark 15 (two-module edges) in
    verify_edge_arity.py.
"""
import json
import os
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh, null_space

RESULTS = Path(os.environ.get('ANALYSIS_RESULTS', Path(__file__).resolve().parents[2] / 'results' / 'analysis')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)


def spd(n, lo=1., hi=5.):
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return Q @ np.diag(rng.uniform(lo, hi, n)) @ Q.T


def lmin(A, B):
    return eigh(.5 * (A + A.T), .5 * (B + B.T), eigvals_only=True)[0]


# --------------------------------------------------------------------------------- (1) join-tree endpoint
def random_tree_assembly(n_cells=6, dims=(2, 3)):
    """Random join tree of cells. Every cell has private modules and shares one 'body' module with its parent.
    Constraints of a cell act on (separator, private) and are chosen so that every separator velocity is realizable
    by the cell alone AND by the rest (separator realizability)."""
    modules = []                       # module dimensions
    cells = []                         # list of module index lists
    parent = [-1]
    # root cell: one body module + private module
    body = len(modules); modules.append(int(rng.integers(*dims) + 1))
    priv = len(modules); modules.append(int(rng.integers(*dims) + 2))
    cells.append([body, priv])
    bodies = [body]
    for c in range(1, n_cells):
        p = int(rng.integers(0, c))
        parent.append(p)
        sep = int(rng.choice(cells[p]))            # separator: one module of the parent cell
        priv = len(modules); modules.append(int(rng.integers(*dims) + 3))
        cells.append([sep, priv])
    offs = np.cumsum([0] + modules)
    n = offs[-1]
    # constraints of each cell: rows m_c = dim(private) - 1 acting on (sep, priv) with full row rank on priv
    rows = []
    for c, cell in enumerate(cells):
        a, b = cell
        na, nb = modules[a], modules[b]
        m = max(1, nb - 1) if c > 0 else max(1, nb - 1)
        Ab = rng.standard_normal((m, nb))         # full row rank on the private module -> any separator velocity realizable
        Aa = rng.standard_normal((m, na))
        R = np.zeros((m, n))
        R[:, offs[a]:offs[a + 1]] = Aa
        R[:, offs[b]:offs[b + 1]] = Ab
        rows.append(R)
    M = np.zeros((n, n)); D = np.zeros((n, n))
    for j, nj in enumerate(modules):
        s = slice(offs[j], offs[j + 1])
        M[s, s] = spd(nj)
        r = int(rng.integers(0, nj + 1))           # some modules undamped (bodies), others partially damped
        G = rng.standard_normal((nj, r))
        D[s, s] = G @ G.T
    return modules, offs, cells, parent, rows, M, D


def leaf_to_root_certificate(modules, offs, cells, parent, rows, M, D, kappa):
    """Leaf-to-root elimination of Theorem 11(b). Returns cell forms on the full coordinates and checks."""
    n = M.shape[0]
    Q = D - kappa * M
    # cell-supported forms p_c, start: module forms assigned to the first cell that contains the module
    owner = {}
    for c, cell in enumerate(cells):
        for j in cell:
            owner.setdefault(j, c)
    p = [np.zeros((n, n)) for _ in cells]
    for j, c in owner.items():
        s = slice(offs[j], offs[j + 1])
        p[c][s, s] += Q[s, s]
    theta = [None] * len(cells)
    alive = set(range(len(cells)))
    children = {c: [k for k in range(len(cells)) if parent[k] == c] for c in range(len(cells))}
    order = []
    # post-order: leaves first
    def visit(c):
        for k in children[c]:
            visit(k)
        order.append(c)
    visit(0)
    for c in order:
        if c == 0:
            theta[c] = p[c]
            continue
        par = parent[c]
        sep = [j for j in cells[c] if j in cells[par]]
        prv = [j for j in cells[c] if j not in cells[par]]
        Sidx = np.concatenate([np.arange(offs[j], offs[j + 1]) for j in sep]) if sep else np.zeros(0, int)
        Pidx = np.concatenate([np.arange(offs[j], offs[j + 1]) for j in prv])
        A = rows[c]
        AS, AP = A[:, Sidx], A[:, Pidx]
        X = -np.linalg.pinv(AP) @ AS if len(Sidx) else np.zeros((len(Pidx), 0))
        assert np.linalg.norm(AS + AP @ X) < 1e-9, 'separator realizability by the leaf'
        Z = null_space(AP)
        V = np.concatenate([Sidx, Pidx])
        pc = p[c][np.ix_(V, V)]
        ns = len(Sidx)
        # p_c(s, X s + Z x) = x'F x + 2 x' Lam s + s' G s
        T = np.zeros((len(V), ns + Z.shape[1]))
        T[:ns, :ns] = np.eye(ns)
        T[ns:, :ns] = X
        T[ns:, ns:] = Z
        H = T.T @ pc @ T
        F, Lam, G = H[ns:, ns:], H[ns:, :ns], H[:ns, :ns]
        w, U = np.linalg.eigh(.5 * (F + F.T))
        tol = 1e-10 * max(1., np.abs(w).max())
        assert w.min() >= -tol, f'F not PSD ({w.min()})'
        Fp = (U[:, w > tol] / w[w > tol]) @ U[:, w > tol].T
        assert np.linalg.norm(Lam - F @ Fp @ Lam) <= 1e-7 * max(1., np.linalg.norm(Lam)), 'range condition'
        Xi = G - Lam.T @ Fp @ Lam
        th = p[c].copy()
        th[np.ix_(Sidx, Sidx)] -= Xi
        theta[c] = th
        p[par][np.ix_(Sidx, Sidx)] += Xi
    # checks: sum identity and local nonnegativity on T_c (cell constraint kernels, on cell coordinates)
    total = sum(theta)
    defect = np.linalg.norm(total - Q) / max(1., np.linalg.norm(Q))
    worst = np.inf
    for c, cell in enumerate(cells):
        V = np.concatenate([np.arange(offs[j], offs[j + 1]) for j in cell])
        outside = np.setdiff1d(np.arange(n), V)
        assert np.abs(theta[c][np.ix_(outside, np.arange(n))]).max() < 1e-12, 'form supported on the cell'
        A = rows[c][:, V]
        N = null_space(A)
        th = theta[c][np.ix_(V, V)]
        loc = N.T @ th @ N
        scale = max(1., np.abs(loc).max())
        worst = min(worst, np.linalg.eigvalsh(.5 * (loc + loc.T)).min() / scale)
    return defect, worst


def check_join_trees(trials=400):
    res = []
    for _ in range(trials):
        modules, offs, cells, parent, rows, M, D = random_tree_assembly(n_cells=int(rng.integers(3, 8)))
        A = np.vstack(rows)
        E = null_space(A)
        if E.shape[1] == 0:
            continue
        # separator realizability: T projects onto every separator
        ok = True
        for c in range(1, len(cells)):
            sep = [j for j in cells[c] if j in cells[parent[c]]]
            S = np.concatenate([np.arange(offs[j], offs[j + 1]) for j in sep])
            ok &= np.linalg.matrix_rank(E[S], tol=1e-9) == len(S)
        if not ok:
            continue
        kap = lmin(E.T @ D @ E, E.T @ M @ E)
        if kap <= 1e-6:
            continue
        defect, worst = leaf_to_root_certificate(modules, offs, cells, parent, rows, M, D, kap)
        res.append(dict(cells=len(cells), d=int(E.shape[1]), kappa=float(kap), sum_defect=float(defect), min_local_eig=float(worst)))
    return res


# --------------------------------------------------------------------------------- (2) architectures without join tree
def best_modular_rate(cells, edge_rows, D, M, dims):
    n = D.shape[0]
    offs = np.cumsum([0] + dims)
    k = cp.Variable()
    tot = 0
    cons = []
    for c, cell in enumerate(cells):
        idx = np.concatenate([np.arange(offs[j], offs[j + 1]) for j in cell])
        P = np.zeros((len(idx), n)); P[np.arange(len(idx)), idx] = 1.
        th = cp.Variable((len(idx), len(idx)), symmetric=True)
        R = edge_rows.get(c)
        N = null_space(R[:, idx]) if R is not None else np.eye(len(idx))
        cons.append(N.T @ th @ N >> 0)
        tot = tot + P.T @ th @ P
    cons += [tot == D - k * M, k <= 1e3]
    prob = cp.Problem(cp.Maximize(k), cons)
    prob.solve(solver='CLARABEL')
    return prob.status, float(k.value)


def frustrated_clique(m, kz=2.):
    """Cells = all (m-1)-subsets of m modules (every proper subset covered, the m-clique not): constraints
    sum_{i != j} x_i = 0 assigned to the cell K \\ {j}."""
    dims = [2] * m
    n = 2 * m
    M = np.eye(n)
    D = np.zeros((n, n))
    for j in range(m):
        D[2 * j + 1, 2 * j + 1] = kz
    cells, edge_rows = [], {}
    for j in range(m):
        cell = [i for i in range(m) if i != j]
        R = np.zeros((1, n))
        for i in cell:
            R[0, 2 * i] = 1.
        edge_rows[len(cells)] = R
        cells.append(cell)
    A = np.vstack(list(edge_rows.values()))
    E = null_space(A)
    kap = lmin(E.T @ D @ E, E.T @ M @ E)
    st, kb = best_modular_rate(cells, edge_rows, D, M, dims)
    st1, k1 = best_modular_rate([list(range(m))], {0: A}, D, M, dims)
    return dict(m=m, centralized=float(kap), best_with_given_cells=kb, status=st, single_cell=k1)


def twisted_ring_with_larger_cells(k=5, kz=2.):
    """Theorem 13, case (A): chordless k-cycle of the primal graph. Cells are the pairs {j, j+1} with the equalities
    x_j = x_{j+1} and the twisted closing edge x_{k-1} = -x_0, plus one extra module k attached to module 0 by a cell
    without edges (so that the primal graph has a vertex outside the cycle)."""
    dims = [2] * (k + 1)                           # k ring modules and one extra module
    n = 2 * (k + 1)
    M = np.eye(n)
    D = np.zeros((n, n))
    for j in range(k):
        D[2 * j + 1, 2 * j + 1] = kz
    D[2 * k:, 2 * k:] = kz * np.eye(2)
    cells, edge_rows = [], {}
    for i in range(k):
        a, b = i, (i + 1) % k
        cell = [a, b]
        R = np.zeros((1, n))
        R[0, 2 * a] = 1.
        R[0, 2 * b] = -1. if i < k - 1 else 1.
        edge_rows[len(cells)] = R
        cells.append(cell)
    cells.append([k, 0])                            # an extra module attached to the ring by a cell without edge
    A = np.vstack(list(edge_rows.values()))
    E = null_space(A)
    kap = lmin(E.T @ D @ E, E.T @ M @ E)
    st, kb = best_modular_rate(cells, edge_rows, D, M, dims)
    return dict(k=k, centralized=float(kap), best_with_given_cells=kb, status=st)


if __name__ == '__main__':
    out = {}
    jt = check_join_trees()
    out['join_trees'] = dict(instances=len(jt), max_sum_defect=max(r['sum_defect'] for r in jt),
                             min_local_eig=min(r['min_local_eig'] for r in jt), records=jt)
    print('join trees with separator realizability at the centralized rate:', len(jt), 'instances; max sum defect',
          f"{out['join_trees']['max_sum_defect']:.1e}; min normalized local eigenvalue {out['join_trees']['min_local_eig']:.1e}")
    out['frustrated_cliques'] = [frustrated_clique(m) for m in (3, 4, 5)]
    for r in out['frustrated_cliques']:
        print('minimal uncovered clique m=%d: centralized %.4f, best with (m-1)-cells %.2e (%s), one m-cell %.4f'
              % (r['m'], r['centralized'], r['best_with_given_cells'], r['status'], r['single_cell']))
    out['rings'] = [twisted_ring_with_larger_cells(k) for k in (4, 6)]
    for r in out['rings']:
        print('twisted chordless ring k=%d: centralized %.4f, best modular %.2e (%s)' % (r['k'], r['centralized'], r['best_with_given_cells'], r['status']))
    json.dump(out, open(RESULTS / 'verify_new_theorems.json', 'w'), indent=1)
