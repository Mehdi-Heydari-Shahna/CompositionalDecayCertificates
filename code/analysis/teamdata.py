"""Per-robot interface data of the compiled teams (reads the framework package; writes nothing into it).

For a team at a carrying configuration this module returns, robot by robot, the data that the robot itself holds
in the payload-star certificate:
    M_i, R_i, B_i       inertia, joint damping and actuator map of its own coordinates,
    A_i                 its grasp and ground/rolling rows, restricted to (payload coordinates, own coordinates),
    ports_i             its actuator ports (rate gains K act on these),
and the shared payload inertia M_0, together with the assembled lift E for the centralized comparison.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import eigh, null_space

_REPO = Path(__file__).resolve().parents[2]
PKG = Path(os.environ.get('TEAM_CMG_PACDM', _REPO / 'code' / 'Team_CMG_PACDM')).resolve()        # the framework package
CHECKS = Path(os.environ.get('CHECKS_CODE', _REPO / 'code' / 'checks')).resolve()               # scripts of the numerical checks
CHK = Path(os.environ.get('CHECKS_RESULTS', _REPO / 'results' / 'checks')).resolve()            # their results
RESULTS = Path(os.environ.get('ANALYSIS_RESULTS', _REPO / 'results' / 'analysis')).resolve()   # results of these scripts
RESULTS.mkdir(parents=True, exist_ok=True)
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from teamc.interfaces import TeamModel  # noqa: E402
from teamc.io import INPUTS, load_team, read_json  # noqa: E402

TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3')


def gen_eig(Dm, Mm):
    return eigh(.5 * (Dm + Dm.T), .5 * (Mm + Mm.T), eigvals_only=True)


@lru_cache(maxsize=None)
def controller(name):
    from mission.controller import TeamController
    comp = load_team(name)
    mission = read_json(INPUTS / 'missions' / f'{name}.json')
    ctrl = TeamController(comp, mission, .002)
    return comp, mission, ctrl


def configuration(name, at=None):
    """Carrying configuration: the stored nominal one (route parameter 2 m) or the one at route parameter ``at``."""
    if at is None:
        return np.load(CHK / f'q_carry_{name}.npy')
    from mission.runner import carry_configuration
    comp, mission, ctrl = controller(name)
    return carry_configuration(comp, mission, ctrl, at)


def mission_gains(name):
    """Rate gains K of the mission loops per actuator port (kd on joints, wheel gain on wheels) and the joint stiffness."""
    comp, mission, ctrl = controller(name)
    model = ctrl.model
    K, Kp = np.zeros(model.nf), np.zeros(model.nf)
    joint_ports, wheel_ports = {}, {}
    for rid, ix in ctrl.ix.items():
        K[ix['joint_ports']], Kp[ix['joint_ports']] = ctrl.kd[rid], ctrl.kp[rid]
        K[ix['wheel_ports']] = ctrl.opt['wheel_gain']
        joint_ports[rid] = np.asarray(ix['joint_ports'], int)
        wheel_ports[rid] = np.asarray(ix['wheel_ports'], int)
    assert np.all(K > 0)
    return K, Kp, joint_ports, wheel_ports


class Star:
    """Payload-star data of one team at one configuration."""

    def __init__(self, name, q=None, at=None):
        self.name = name
        comp, mission, ctrl = controller(name)
        self.comp = comp
        self.model = TeamModel(comp)
        self.q = configuration(name, at) if q is None else q
        self.mode = self.model.mode()
        m = self.model
        self.n, self.nf = m.n, m.nf
        self.B = m.B
        self.R = np.diag(m.damping)
        self.M = m.tree.mass(self.q)
        self.ids = list(m.ids)
        self.payload = np.asarray(m.payload, int)
        self.robots = list(comp.robots)
        self.coords = {rid: np.array([i for i, k in enumerate(self.ids) if k.startswith(rid + '_')], int) for rid in self.robots}
        self.ports = {rid: np.array([k for k, p in enumerate(m.ports) if p['index'] in set(self.coords[rid])], int)
                      for rid in self.robots}
        assert len(self.payload) + sum(len(c) for c in self.coords.values()) == self.n
        con = m.constraints(self.q, self.mode)
        assert not any(b.get('slip') for b in con.blocks), 'rolling teams only'
        self.J = con.J
        self.blocks = con.blocks
        self.E = m.lift(self.q, self.mode).E
        self.d = self.E.shape[1]
        # local rows of every robot: its grasp and its ground/rolling blocks, on (payload, own) columns
        self.local = {}
        used = np.zeros(self.J.shape[0], bool)
        for rid in self.robots:
            rows = []
            for b in self.blocks:
                bid = b['id']
                mine = bid == f'grasp_{rid}' or bid.startswith(f'{rid}:')
                if mine:
                    rows.extend(range(b['rows'].start, b['rows'].stop))
            rows = np.array(rows, int)
            used[rows] = True
            cols = np.r_[self.payload, self.coords[rid]]
            other = np.setdiff1d(np.arange(self.n), cols)
            leak = float(np.max(np.abs(self.J[np.ix_(rows, other)]))) if len(other) else 0.
            assert leak < 1e-12, f'{name} {rid}: rows act outside (payload, own) columns ({leak})'
            A = self.J[np.ix_(rows, cols)]
            T = null_space(A, rcond=1e-10)
            self.local[rid] = dict(A=A, T=T, cols=cols, coords=self.coords[rid], ports=self.ports[rid],
                                   M=self.M[np.ix_(self.coords[rid], self.coords[rid])],
                                   R=self.R[np.ix_(self.coords[rid], self.coords[rid])],
                                   B=self.B[np.ix_(self.coords[rid], self.ports[rid])])
        assert used.all(), 'every row belongs to one robot cell'
        self.M0 = self.M[np.ix_(self.payload, self.payload)]
        # block-diagonal inertia (modules are separate trees)
        off = 0.
        for a in self.robots:
            off = max(off, float(np.max(np.abs(self.M[np.ix_(self.coords[a], self.payload)]))))
            for b in self.robots:
                if a != b:
                    off = max(off, float(np.max(np.abs(self.M[np.ix_(self.coords[a], self.coords[b])]))))
        assert off < 1e-10, f'inertia couples modules ({off})'

    # ------------------------------------------------------------------ rates
    def D(self, K):
        return self.B @ np.diag(K) @ self.B.T + self.R

    def rate(self, K):
        """Centralized rate kl = lambda_min(E^T D E, E^T M E)."""
        E = self.E
        return float(gen_eig(E.T @ self.D(K) @ E, E.T @ self.M @ E)[0])

    def local_form(self, rid, K, kappa):
        """Q_i = D_i - kappa M_i on the robot's coordinates."""
        l = self.local[rid]
        Di = l['B'] @ np.diag(K[l['ports']]) @ l['B'].T + l['R']
        return Di - kappa * l['M']

    def realizable(self, rid):
        """Every payload velocity can be followed by the robot alone (projection of T_i onto the payload is onto)."""
        T = self.local[rid]['T']
        return int(np.linalg.matrix_rank(T[:6], tol=1e-9)) == 6

    def summary(self, rid, K, kappa):
        """Generalized Schur summary Xi_i of Proposition 7 and its local feasibility test."""
        l = self.local[rid]
        A = l['A']
        Ap, Ai = A[:, :6], A[:, 6:]
        X = -np.linalg.pinv(Ai) @ Ap                         # particular local motion for a payload velocity
        Z = null_space(Ai, rcond=1e-10)                      # self-motions with the payload at rest
        assert np.linalg.norm(Ap + Ai @ X) < 1e-9 * max(1., np.linalg.norm(Ap)), 'payload realizability'
        Q = self.local_form(rid, K, kappa)
        F, Lam = Z.T @ Q @ Z, Z.T @ Q @ X
        w, V = np.linalg.eigh(.5 * (F + F.T))
        tol = 1e-11 * max(1., np.abs(w).max())
        ok = bool(w.min() >= -tol)
        Fp = (V[:, w > tol] / w[w > tol]) @ V[:, w > tol].T
        ok = ok and np.linalg.norm(Lam - F @ Fp @ Lam) <= 1e-8 * max(1., np.linalg.norm(Lam))
        Xi = X.T @ Q @ X - Lam.T @ Fp @ Lam
        return .5 * (Xi + Xi.T), ok, Z.shape[1]

    def aggregate(self, K, kappa):
        Xis, ok = {}, True
        for rid in self.robots:
            Xis[rid], oki, _ = self.summary(rid, K, kappa)
            ok = ok and oki
        S = sum(Xis.values()) - kappa * self.M0
        return Xis, .5 * (S + S.T), ok

    def interface_rate(self, K, lo=0., hi=None, iters=80):
        """Largest kappa passing the payload-space test (bisection on Proposition 7)."""
        hi = 2. * self.rate(K) if hi is None else hi
        for _ in range(iters):
            mid = .5 * (lo + hi)
            _, S, ok = self.aggregate(K, mid)
            if ok and np.linalg.eigvalsh(S).min() >= -1e-10 * max(1., np.abs(S).max()):
                lo = mid
            else:
                hi = mid
        return lo

    def scalar_share_rate(self, rid, K, share):
        """Standalone rate of robot rid holding inertia share*M0 (Definition 6 with J = share*M0)."""
        l = self.local[rid]
        T = l['T']
        Di = l['B'] @ np.diag(K[l['ports']]) @ l['B'].T + l['R']
        Dl = np.zeros((len(l['cols']),) * 2)
        Ml = np.zeros_like(Dl)
        Dl[6:, 6:] = Di
        Ml[:6, :6] = share * self.M0
        Ml[6:, 6:] = l['M']
        return float(gen_eig(T.T @ Dl @ T, T.T @ Ml @ T)[0])
