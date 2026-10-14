"""Generated connection (chord) evaluators and the per-robot PACDM closure modules.

New extension; ``PACDM`` (correct, acquire, mapping) is the unchanged kernel.

A connection joins a frame on body 1 (minus side) to a frame on body 2 (plus
side).  Its residual and differential use the kernel convention exactly:

    T- = T_body1 T1 prod_k exp(xi_k c_k),      T+ = T_body2 T2,
    r  = log((T- D)^-1 T+),                    J = inv_left(r) Ad((T- D)^-1) (E+ - E-),

where ``xi_k`` are the free relative twists of the connection type (none for a
weld) with chart coordinates ``c_k``, and ``D`` is the artificial defect of the
PACDM homotopy.  At closure ``J v`` is the relative twist of the plus frame with
respect to the minus frame in minus-frame axes, so the multiplier of these rows
is the wrench the connection applies to body 2, in the minus frame.

What is generated from the compiled graph: only the two world-rooted joint paths
of a connection are traversed and only their coordinate columns are formed.
"""
from __future__ import annotations

import numpy as np

from vendor.pacdm_original import PACDM

from .se3 import adj_fast, exp, inv, inv_fast, inv_left, log
from .tree import PRISMATIC, REVOLUTE


_I3 = np.eye(3)


class Chain:
    """One world-rooted joint path ending in a body-fixed frame."""

    def __init__(self, tree, body, T_body_frame, column_of):
        """``column_of`` maps a global coordinate index to a local column (missing -> not differentiated)."""
        i = tree.index[body]
        path = []
        while i != 0:
            path.append(i)
            i = tree.parent[i]
        eye = np.eye(4)
        self.steps = []
        for i in reversed(path):
            k = int(tree.coord[i])
            enter, leave = tree.enter[i], tree.leave[i]
            self.steps.append((None if np.array_equal(enter, eye) else (enter[:3, :3].copy(), enter[:3, 3].copy()),
                               int(tree.kind[i]), tree.axis[i].copy(), tree.K[i].copy(), tree.K2[i].copy(),
                               None if np.array_equal(leave, eye) else (leave[:3, :3].copy(), leave[:3, 3].copy()),
                               k, column_of.get(k, -1) if k >= 0 else -1))
        self.end = np.asarray(T_body_frame, float)
        self.columns = sorted({s[7] for s in self.steps if s[7] >= 0})

    def evaluate(self, q, ncols, need_jac=True):
        """End-frame world pose and its 6 x ncols world spatial twist map (angular-first)."""
        R, p = _I3, np.zeros(3)
        X = np.zeros((6, ncols)) if need_jac else None
        for enter, kind, axis, K, K2, leave, k, col in self.steps:
            if enter is not None:
                p = p + R @ enter[1]
                R = R @ enter[0]
            if kind == REVOLUTE:
                if need_jac and col >= 0:
                    w = R @ axis
                    X[0, col], X[1, col], X[2, col] = w
                    X[3, col] = p[1] * w[2] - p[2] * w[1]
                    X[4, col] = p[2] * w[0] - p[0] * w[2]
                    X[5, col] = p[0] * w[1] - p[1] * w[0]
                a = q[k]
                R = R @ (_I3 + np.sin(a) * K + (1. - np.cos(a)) * K2)
            elif kind == PRISMATIC:
                w = R @ axis
                if need_jac and col >= 0:
                    X[3:, col] = w
                p = p + w * q[k]
            if leave is not None:
                p = p + R @ leave[1]
                R = R @ leave[0]
        T = np.eye(4)
        T[:3, :3] = R @ self.end[:3, :3]
        T[:3, 3] = p + R @ self.end[:3, 3]
        return T, X


def chord_residual(Tm, Xm, Tp, Xp, free, chart, chart_columns, defect=None):
    """Kernel residual/Jacobian of one connection from its two evaluated frames.

    ``Xm``/``Xp`` are world spatial twist maps (6 x ncols) of the minus and plus
    frames.  ``chart`` holds the free-twist coordinates and ``chart_columns``
    their local columns.
    """
    Xm = Xm.copy() if (free and Xm is not None) else Xm
    for xi, c, col in zip(free, chart, chart_columns):
        if Xm is not None:
            Xm[:, col] += adj_fast(Tm) @ xi
        Tm = Tm @ exp(np.asarray(xi) * c)
    reverse = inv_fast(Tm if defect is None else Tm @ defect)
    delta = reverse @ Tp
    r = log(delta)
    J = None if Xm is None else inv_left(r) @ adj_fast(reverse) @ (Xp - Xm)
    return r, J, delta


class GraspModule:
    """PACDM graph of one robot's connection to the payload (kernel graph protocol).

    Local coordinates ``x``: inputs first (payload chart 6, base chart 6, the
    robot's commanded tool-path coordinates), then the dependent tool-path
    coordinates, then the connection's chart coordinates (none for a weld).
    ``offset`` right-multiplies the payload frame; it is the identity for a
    physical grasp and an approach offset while the tool is only tracking the
    grasp frame.
    """

    def __init__(self, comp, connection_id):
        c = comp.connections[connection_id]
        robot = comp.robots[c['robot']]
        tree = comp.tree
        self.comp, self.id, self.robot = comp, connection_id, robot
        self.free = [np.asarray(f, float) for f in c['free']]
        inputs = list(comp.payload['base']) + list(robot.base) + [robot.joint[j] for j in robot.tool_path
                                                                  if j in robot.commanded]
        dependent = [robot.joint[j] for j in robot.dependent]
        self.global_index = np.array(inputs + dependent, int)
        nt = len(self.global_index)
        self.nt = nt
        self.n = nt + len(self.free)
        self.active = np.arange(len(inputs))
        self.passive = np.arange(len(inputs), self.n)
        column = {int(g): i for i, g in enumerate(self.global_index)}
        self.minus = Chain(tree, c['body1'], c['T1'], column)
        self.plus = Chain(tree, c['body2'], c['T2'], column)
        self.T2 = np.asarray(c['T2'], float)
        self.chart_columns = list(range(nt, self.n))
        ids = comp.cmg['coordinate_ids']
        lim = comp.cmg['limits']
        self.lower = np.r_[[lim[ids[g]]['lower'] for g in self.global_index], np.full(len(self.free), -np.pi)]
        self.upper = np.r_[[lim[ids[g]]['upper'] for g in self.global_index], np.full(len(self.free), np.pi)]
        self.nglobal = tree.n
        self.calls = 0
        self._scratch = np.zeros(tree.n)

    def set_offset(self, T=None):
        self.plus.end = self.T2 if T is None else self.T2 @ np.asarray(T, float)

    def set_grasp_frame(self, T2):
        """Model reconfiguration: re-register the payload-side frame (for example as measured at latching)."""
        self.T2 = np.asarray(T2, float).copy()
        self.plus.end = self.T2

    def pack(self, q, chart=None):
        x = np.zeros(self.n)
        x[:self.nt] = np.asarray(q, float)[self.global_index]
        if chart is not None:
            x[self.nt:] = chart
        return x

    def unpack(self, x, q):
        q[self.global_index[len(self.active):]] = x[len(self.active):self.nt]
        return q

    def _evaluate(self, x, defects, need_jac):
        self.calls += 1
        q = self._scratch
        q[self.global_index] = x[:self.nt]
        Tm, Xm = self.minus.evaluate(q, self.n, need_jac)
        Tp, Xp = self.plus.evaluate(q, self.n, need_jac)
        r, J, delta = chord_residual(Tm, Xm, Tp, Xp, self.free, x[self.nt:], self.chart_columns,
                                     None if defects is None else defects[0])
        return r, J, delta[None]

    def residual(self, x, defects=None):
        return self._evaluate(x, defects, True)

    def residual_only(self, x):
        return self._evaluate(x, None, False)[0]

    def frames(self, x):
        """World poses of the tool (minus) and target (plus) frames."""
        q = self._scratch
        q[self.global_index] = x[:self.nt]
        return self.minus.evaluate(q, self.n, False)[0], self.plus.evaluate(q, self.n, False)[0]


class TeamAssembler:
    """Modular PACDM assembly: one closure module per robot, solved only when its inputs change."""

    def __init__(self, comp, q0, connection_ids=None, predictor=True, reuse=True):
        self.comp = comp
        self.predictor, self.reuse = predictor, reuse
        ids = [k for k, c in comp.connections.items() if c['kind'] == 'grasp'] if connection_ids is None \
            else list(connection_ids)
        self.modules = {}
        for cid in ids:
            g = GraspModule(comp, cid)
            self.modules[cid] = dict(g=g, solver=PACDM(g), x=None, N=None, info=None)
        self.stats = dict(solved=0, skipped=0, fallbacks=0)
        self.q = np.asarray(q0, float).copy()

    def module(self, cid):
        return self.modules[cid]['g']

    def acquire(self, cid, q, offset=None):
        """Close one module from an arbitrary seed with the kernel's defect homotopy."""
        m = self.modules[cid]
        g = m['g']
        g.set_offset(offset)
        x0 = g.pack(q)
        x0[g.active] = np.clip(x0[g.active], g.lower[g.active], g.upper[g.active])
        x, info = m['solver'].acquire(x0[g.active].copy(), x0)
        if not info['success']:
            raise RuntimeError(f'{cid}: PACDM acquisition failed: {info.get("message")}')
        N, mi = m['solver'].mapping(x)
        if not mi['success']:
            raise RuntimeError(f'{cid}: PACDM mapping failed after acquisition: {mi}')
        m.update(x=x, N=N, info=mi)
        return g.unpack(x, np.asarray(q, float).copy()), info

    def solve(self, q, connection_ids=None, offsets=None):
        """Dependent coordinates of the listed modules for the inputs contained in ``q``."""
        q = np.asarray(q, float).copy()
        report = {}
        for cid in (self.modules if connection_ids is None else connection_ids):
            m = self.modules[cid]
            g, solver = m['g'], m['solver']
            g.set_offset(None if offsets is None else offsets.get(cid))
            if m['x'] is None:
                q, _ = self.acquire(cid, q, None if offsets is None else offsets.get(cid))
                report[cid] = dict(fallback=False, reused=False, acquired=True, rcond=m['info']['rcond'])
                continue
            active = g.pack(q)[g.active]
            # measured inputs may sit a hair outside a declared limit (a lift resting on its end stop)
            if np.any(active < g.lower[g.active] - 1e-3) or np.any(active > g.upper[g.active] + 1e-3):
                raise RuntimeError(f'{cid}: closure input outside its declared limits')
            active = np.clip(active, g.lower[g.active], g.upper[g.active])
            offset_key = None if offsets is None or offsets.get(cid) is None else np.asarray(offsets[cid]).tobytes()
            if self.reuse and np.array_equal(active, m['x'][g.active]) and m.get('offset_key') == offset_key:
                self.stats['skipped'] += 1
                g.unpack(m['x'], q)
                report[cid] = dict(fallback=False, reused=True, rcond=m['info']['rcond'])
                continue
            pred = m['x'][g.passive].copy()
            if self.predictor and m.get('offset_key') == offset_key:
                pred += m['N'][g.passive] @ (active - m['x'][g.active])
                pred = np.clip(pred, g.lower[g.passive], g.upper[g.passive])
            x, ci = solver.correct(active, pred, None, np.array(m['info']['rows']), maxiter=12)
            fallback = False
            if not ci['success']:
                seed = m['x'].copy()
                x, ci = solver.acquire(active, seed)
                fallback = True
                self.stats['fallbacks'] += 1
            if not ci['success']:
                raise RuntimeError(f'{cid}: closure failed: {ci.get("message")}')
            N, mi = solver.mapping(x)
            if not mi['success']:
                raise RuntimeError(f'{cid}: {mi["message"]} (rcond {mi["rcond"]:.2e})')
            m.update(x=x, N=N, info=mi, offset_key=offset_key)
            self.stats['solved'] += 1
            g.unpack(x, q)
            report[cid] = dict(fallback=fallback, reused=False, rcond=mi['rcond'], residual=ci.get('residual_inf'))
        self.q = q
        return q, report

    def gap(self, cid, q, offset=None):
        """Physical gap of a connection at ``q`` (translation m, rotation rad)."""
        g = self.modules[cid]['g']
        g.set_offset(offset)
        Tm, Tp = g.frames(g.pack(q))
        d = inv(Tm) @ Tp
        R = d[:3, :3]
        # the sine resolves a small angle; the arccosine of the trace alone cannot see anything below 2e-8 rad
        sine = .5 * np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        return float(np.linalg.norm(d[:3, 3])), float(np.arctan2(sine, .5 * (np.trace(R) - 1.)))
