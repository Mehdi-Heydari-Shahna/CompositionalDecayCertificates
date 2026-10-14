"""Proposition 12 (endpoint failure without realizability): the data (eq. endpoint-data) of the paper.

Coordinates (y | x, u | z, w): payload y, robot 1 (x, u), robot 2 (z, w).
Checks d = 2, rank L = d, centralized rate kl = 1, and the best rate certified by supplies with |Theta_i| <= bound.
"""
import json
import os
from pathlib import Path

import cvxpy as cp
import numpy as np
from scipy.linalg import eigh, null_space
from scipy.optimize import brentq

RESULTS = Path(os.environ.get('ANALYSIS_RESULTS', Path(__file__).resolve().parents[2] / 'results' / 'analysis')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)

M = np.zeros((5, 5))
D = np.zeros((5, 5))
M[0, 0] = 1.
M[1:3, 1:3] = [[1., -1.], [-1., 2.]]
M[3:, 3:] = np.eye(2)
D[1:3, 1:3] = np.diag([1., 2.])
D[3:, 3:] = 2. * np.eye(2)
A1 = np.array([[-1., 0., 1.]])                       # robot 1 on (y, x, u): u = y
A2 = np.array([[-1., 0., 1.], [0., 0., 1.]])         # robot 2 on (y, z, w): w = y, w = 0
A = np.zeros((3, 5))
A[0, [0, 1, 2]] = A1[0]
A[1:, [0, 3, 4]] = A2
E = null_space(A)
B = np.vstack([np.zeros((1, 4)), np.eye(4)])         # payload unactuated
kl = float(eigh(E.T @ D @ E, E.T @ M @ E, eigvals_only=True)[0])
N1, N2 = null_space(A1), null_space(A2)


def best_rate(bound):
    k, t1, t2 = cp.Variable(), cp.Variable(), cp.Variable()
    z12, z21 = np.zeros((1, 2)), np.zeros((2, 1))
    Q1 = cp.bmat([[cp.reshape(-t1, (1, 1), order='F'), z12], [z21, D[1:3, 1:3] - k * M[1:3, 1:3]]])
    Q2 = cp.bmat([[cp.reshape(-t2, (1, 1), order='F'), z12], [z21, D[3:, 3:] - k * M[3:, 3:]]])
    cons = [N1.T @ Q1 @ N1 >> 0, N2.T @ Q2 @ N2 >> 0, t1 + t2 >= k * M[0, 0], cp.abs(t1) <= bound, cp.abs(t2) <= bound]
    prob = cp.Problem(cp.Maximize(k), cons)
    prob.solve(solver='CLARABEL')
    return prob.status, float(k.value), float(t1.value), float(t2.value)


def analytic_rate(bound):
    """Largest kappa with Theta_2 = bound, Theta_1 = kappa - bound <= 2 delta - kappa^2/delta, delta = 1 - kappa."""
    f = lambda k: 2. * (1. - k) - k ** 2 / (1. - k) - (k - bound)
    return brentq(f, 0., 1. - 1e-15)


if __name__ == '__main__':
    out = dict(d=int(E.shape[1]), rank_L=int(np.linalg.matrix_rank(B.T @ E)), centralized_rate=kl, bounded_supplies=[])
    for bound in (1e1, 1e2, 1e3):
        status, k, t1, t2 = best_rate(bound)
        out['bounded_supplies'].append(dict(bound=bound, status=status, best_rate=k, analytic=analytic_rate(bound),
                                            theta1=t1, theta2=t2))
    print(json.dumps(out, indent=1))
    (RESULTS / 'verify_endpoint.json').write_text(json.dumps(out, indent=1))
