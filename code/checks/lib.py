"""Numerical checks of the stability theory on the compiled teams: shared helpers.

Everything here only *reads* the package (compiled graph, constraint rows, tangent lift, tree dynamics);
no file of the package is changed and nothing is written into it.

Notation of the paper:  M v' + C(q,v) v + g + D v = B tau + A^T lam,   A(q) v = 0,   v = E(q) u.
The tree coordinates of the package are scalar joint coordinates (floating bases are chains of six scalar
joints), so v = q' and the Christoffel form of C is symmetric in its two velocity slots:
C(q,v) y = C(q,y) v = (b(v+y) - b(v-y)) / 4 with b(w) the velocity-product effort of the tree.
"""
import sys
from pathlib import Path

import numpy as np

import os

# root of the framework package (code/Team_CMG_PACDM); set TEAM_CMG_PACDM to use another copy
ROOT = Path(os.environ.get('TEAM_CMG_PACDM', Path(__file__).resolve().parent.parent / 'Team_CMG_PACDM')).resolve()
if not (ROOT / 'teamc').is_dir():
    raise SystemExit(f'framework package not found at {ROOT}: set the environment variable TEAM_CMG_PACDM to its folder')
# results of the checks (read and written): results/checks of the repository, or CHECKS_RESULTS
RESULTS = Path(os.environ.get('CHECKS_RESULTS', Path(__file__).resolve().parents[2] / 'results' / 'checks')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teamc.interfaces import TeamModel  # noqa: E402
from teamc.io import INPUTS, load_team, read_json  # noqa: E402

TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3', 'husky2', 'husky3')
ROLLING = ('team2', 'team3', 'ridgeback2', 'ridgeback3')


class Team:
    def __init__(self, name):
        self.name = name
        self.comp = load_team(name)
        self.model = TeamModel(self.comp)
        self.tree = self.model.tree
        self.n = self.model.n
        self.B = self.model.B
        self.nf = self.model.nf
        self.D = np.diag(self.model.damping)
        self.q0 = np.load(RESULTS / f'q_carry_{name}.npy')
        self.ids = list(self.model.ids)
        self.payload = np.asarray(self.model.payload, int)
        self.robots = list(self.comp.robots)
        self.coords = {rid: np.array([i for i, k in enumerate(self.ids) if k.startswith(rid + '_')], int)
                       for rid in self.robots}
        assert len(self.payload) + sum(len(c) for c in self.coords.values()) == self.n
        self.actuators = {rid: np.array([k for k, p in enumerate(self.model.ports) if p['index'] in set(self.coords[rid])], int)
                          for rid in self.robots}

    # ---------------------------------------------------------------- rows and lift
    def mode(self, **kw):
        return self.model.mode(**kw)

    def rows(self, q, mode):
        """Ideal rows A (slip rows of friction-driven wheels left out), the constraint record and their row indices."""
        con = self.model.constraints(q, mode)
        keep = np.ones(con.J.shape[0], bool)
        for b in con.blocks:
            if b.get('slip'):
                keep[b['rows']] = False
        return con.J[keep], con, np.flatnonzero(keep)

    def E(self, q, mode):
        return self.model.lift(q, mode).E

    def Edot(self, q, v, mode, h0=1e-6):
        h = h0 / max(1., float(np.max(np.abs(v))))
        return (self.E(q + h * v, mode) - self.E(q - h * v, mode)) / (2 * h)

    # ---------------------------------------------------------------- dynamics terms
    def M(self, q):
        return self.tree.mass(q)

    def b(self, q, w):
        return self.tree.rnea(q, w, np.zeros(self.n), gravity=False)

    def C(self, q, v, y):
        """C(q, v) y in the Christoffel form (symmetric bilinear in v and y)."""
        return .25 * (self.b(q, v + y) - self.b(q, v - y))

    def g(self, q):
        return self.tree.rnea(q, np.zeros(self.n), np.zeros(self.n))

    def gains(self, q, rate=10., mode=None):
        """Actuator-port damping K = rate * diag(inertia seen at the actuator ports), a diagonal positive matrix.

        The inertia seen at the ports is (L^+)^T M_r L^+ with L = B^T E; its diagonal gives one scalar gain per
        actuator (a joint-level, decentralized law).  Any positive diagonal K satisfies the theorems; this choice
        only makes the rates comparable across teams.
        """
        mode = self.mode() if mode is None else mode
        E = self.E(q, mode)
        Lp = np.linalg.pinv(self.B.T @ E)
        seen = np.diag(Lp.T @ (E.T @ self.M(q) @ E) @ Lp)
        return np.diag(rate * np.maximum(seen, 1e-6 * seen.max()))


def gen_eig(Dm, Mm):
    """Generalized eigenvalues of (Dm, Mm), Mm positive definite, ascending."""
    from scipy.linalg import eigh
    return eigh(.5 * (Dm + Dm.T), .5 * (Mm + Mm.T), eigvals_only=True)


def m_projector(M, E):
    """M-orthogonal projector onto range(E)."""
    return E @ np.linalg.solve(E.T @ M @ E, E.T @ M)


def psd_min(X):
    return float(np.min(np.linalg.eigvalsh(.5 * (X + X.T))))


def make_reference(T, ind, seed=7, scale=1.):
    """A smooth reference on the independent coordinates of the carrying mode, and an initial velocity.

    Payload and bases move by a few centimetres / hundredths of a radian around the carrying configuration; the
    lift joints only move upwards from their start (they stand close to their lower limit there).  The plant
    starts off the reference in position (offset) and in velocity.
    Returns ``(yd, u0)`` with ``yd(t) -> (y, y', y'')``.
    """
    rng = np.random.default_rng(seed)
    d = len(ind)
    names = [T.ids[i] for i in ind]
    y0 = T.q0[ind].copy()
    lift = np.array(['lift' in k for k in names])
    amp = np.array([.04 if ('_x' in k or '_y' in k or '_z' in k) else .03 for k in names])
    amp[lift] = .02
    amp = scale * amp                                         # scale < 1: a smaller motion (same phases, same start)
    phase0 = rng.uniform(0, 2 * np.pi, d)
    phase0[lift] = -.5 * np.pi                                # amp (1 - cos(w t)) >= 0
    freq = rng.uniform(.25, .5, d) * 2 * np.pi
    offset = .005 * rng.standard_normal(d)
    offset[lift] = np.abs(offset[lift])

    def yd(t):
        return (y0 + offset + amp * (np.sin(freq * t + phase0) - np.sin(phase0)), amp * freq * np.cos(freq * t + phase0),
                -amp * freq ** 2 * np.sin(freq * t + phase0))

    u0 = yd(0.)[1] + .015 * rng.standard_normal(d)
    return yd, u0
