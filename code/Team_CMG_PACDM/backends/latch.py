"""Compliant clamps and the wrist sensors they imply (NumPy only).

Used by the plants that have no latch constraint of their own (Isaac Sim, and the
MuJoCo plant in its ``latch='compliant'`` mode, which exists to check this module
against MuJoCo's own force/torque sensor).

Clamp model: when a clamp closes it holds the payload where it is.  From then on the
tool frame on the clamp and the frame it had on the payload at that instant are tied
by a six-dimensional spring-damper (declared stiffness and damping, isotropic in
translation and in rotation).  The wrench is evaluated from the state at the start of
a step and applied as an external load during that step: equal and opposite on the
payload and on the clamp body, at the tool-frame origin, so the pair adds no net
force or moment.

Wrist sensor: an ideal force/torque sensor in the sensor joint reads the wrench the
arm side applies to the tool-side body.  By the Newton-Euler balance of that body it
equals what the body needs for its own motion and weight minus the external loads on
it (the clamp load and, if any, contact forces).  With the motion taken from the
measured state before and after a step this is the mean interface wrench of the step.
"""
from __future__ import annotations

import numpy as np

from teamc.se3 import inv_fast
from teamc.tree import wrench_to_frame

# declared clamp compliance (per clamp); see LatchSet.stability_ratio for the timestep it allows.  A robot record
# may declare the compliance of its own clamp under ports.tool.clamp (a lighter arm and payload need a softer one).
DEFAULTS = dict(stiffness_N_m=2.0e5, damping_N_s_m=8.0e2, rot_stiffness_Nm_rad=1.5e3, rot_damping_Nm_s_rad=10.)
STABILITY_LIMIT = .6           # largest accepted stability ratio of the explicit clamp loads (1 is the linear limit)


def rotvec(R):
    """Rotation vector of a rotation matrix (angle below pi)."""
    w = .5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = float(np.linalg.norm(w))
    if s < 1e-12:
        return w
    return w * (np.arctan2(s, .5 * (np.trace(R) - 1.)) / s)


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class LatchSet:
    """The clamps of a compiled team, evaluated in compiled coordinates."""

    def __init__(self, comp, params=None):
        self.comp = comp
        self.tree = comp.tree
        t = self.tree
        self.grasps = {}
        for cid, c in comp.connections.items():
            if c['kind'] != 'grasp':
                continue
            p = dict(DEFAULTS)
            p.update(comp.robots[c['robot']].record['ports']['tool'].get('clamp', {}))      # declared with the clamp
            p.update(params or {})                                                          # given for this run
            self.grasps[cid] = dict(robot=c['robot'], body1=c['body1'], body2=c['body2'], i1=t.index[c['body1']],
                                    i2=t.index[c['body2']], T1=np.asarray(c['T1'], float), T2=None, p=p)
        # the compliance in use: one set of values where every clamp has the same, else one per clamp
        sets = [g['p'] for g in self.grasps.values()]
        self.p = dict(sets[0]) if sets and all(x == sets[0] for x in sets) else {cid: dict(g['p']) for cid, g in self.grasps.items()}
        self.sensors_of = {}
        for rid, r in comp.robots.items():
            port = r.record['ports'].get('sensor')
            if port:
                body, T_s = r.frames[port['frame']]
                self.sensors_of[rid] = dict(body=body, index=t.index[body], T=np.asarray(T_s, float))
        self.closed = {cid: False for cid in self.grasps}

    # ------------------------------------------------------------------ commands
    def close(self, cid, q, kin=None):
        """Close a clamp at the current relative pose; returns the as-latched tool frame in the payload body."""
        T, _ = kin if kin is not None else self.tree.sweep(q)
        g = self.grasps[cid]
        g['T2'] = inv_fast(T[g['i2']]) @ T[g['i1']] @ g['T1']
        self.closed[cid] = True
        return g['T2'].copy()

    def open(self, cid):
        self.closed[cid] = False

    def reset(self):
        for cid in self.closed:
            self.closed[cid] = False

    def stability_ratio(self, q, dt, kin=None):
        """Stability ratio of the explicit clamp loads at a configuration (below 1: stable; linear analysis).

        The loads are evaluated at the start of a step and held during it.  For the closed clamps,
        with ``W = J M^-1 J^T`` the inverse inertia seen by their relative twists and ``G`` the
        diagonal of ``2 dt D + dt^2 K``, the discrete system is stable while the largest eigenvalue
        of ``W G`` stays below 4.  ``M`` is the unconstrained team inertia, which is the least
        favourable case (constraints only add inertia).
        """
        closed = [cid for cid in self.grasps if self.closed[cid]]
        if not closed:
            return 0.
        T, S = kin if kin is not None else self.tree.sweep(q)
        t = self.tree
        rows, gain = [], []
        for cid in closed:
            g = self.grasps[cid]
            p = g['p']
            x = (T[g['i1']] @ g['T1'])[:3, 3]
            J = np.zeros((6, t.n))
            for sign, i in ((1., g['i1']), (-1., g['i2'])):
                for k in t.support[i]:
                    J[:3, t.coord[k]] += sign * S[k][:3]
                    J[3:, t.coord[k]] += sign * (S[k][3:] + _cross(S[k][:3], x))
            rows.append(J)
            gain.append(np.r_[[2. * dt * p['rot_damping_Nm_s_rad'] + dt * dt * p['rot_stiffness_Nm_rad']] * 3,
                              [2. * dt * p['damping_N_s_m'] + dt * dt * p['stiffness_N_m']] * 3])
        J = np.vstack(rows)
        W = J @ np.linalg.solve(t.mass(q, kinematics=(T, S)), J.T)
        return float(np.max(np.abs(np.linalg.eigvals(W * np.concatenate(gain)[None, :]))) / 4.)

    # ------------------------------------------------------------------ loads
    def _twist(self, i, S, v):
        V = np.zeros(6)
        for k in self.tree.support[i]:
            V += S[k] * v[self.tree.coord[k]]
        return V

    def loads(self, q, v, kin=None):
        """Clamp loads of the closed clamps at a state.

        Returns ``{cid: dict(point, force, moment, on, by)}``: world force and moment (about
        ``point``, the tool-frame origin) applied to the payload body ``on``; the clamp body
        ``by`` receives the opposite.
        """
        T, S = kin if kin is not None else self.tree.sweep(q)
        out = {}
        for cid, g in self.grasps.items():
            if not self.closed[cid]:
                continue
            p = g['p']
            Ta = T[g['i1']] @ g['T1']                          # tool frame on the clamp
            Tb = T[g['i2']] @ g['T2']                          # the frame it had on the payload when it closed
            Va, Vb = self._twist(g['i1'], S, v), self._twist(g['i2'], S, v)
            x = Ta[:3, 3]
            rel = (Va[3:] + _cross(Va[:3], x)) - (Vb[3:] + _cross(Vb[:3], x))
            force = p['stiffness_N_m'] * (x - Tb[:3, 3]) + p['damping_N_s_m'] * rel
            moment = p['rot_stiffness_Nm_rad'] * rotvec(Ta[:3, :3] @ Tb[:3, :3].T) \
                + p['rot_damping_Nm_s_rad'] * (Va[:3] - Vb[:3])
            out[cid] = dict(point=x.copy(), force=force, moment=moment, on=g['body2'], by=g['body1'],
                            deflection=float(np.linalg.norm(x - Tb[:3, 3])))
        return out

    @staticmethod
    def spatial(point, force, moment):
        """World spatial wrench ``[moment about the world origin; force]`` of a load at a point."""
        return np.r_[moment + _cross(point, force), force]

    def payload_wrench(self, load, origin):
        """``(force, moment about origin)`` of a clamp load on the payload (audit)."""
        return load['force'].copy(), load['moment'] + _cross(load['point'] - origin, load['force'])

    # ------------------------------------------------------------------ sensors
    def sensors(self, q, v, a, loads, extra=None, kin=None):
        """Wrist sensor readings ``{robot: [moment; force]}`` in the sensor frames.

        ``loads``: result of :meth:`loads` (the loads applied during the step); ``extra``:
        additional world spatial wrenches on bodies ``{body: wrench}`` (contacts on a clamp).
        """
        kin = kin if kin is not None else self.tree.sweep(q)
        fext = {}
        for cid, load in loads.items():
            w = -self.spatial(load['point'], load['force'], load['moment'])       # on the clamp body
            fext[load['by']] = fext.get(load['by'], 0.) + w
        for body, w in (extra or {}).items():
            fext[body] = fext.get(body, 0.) + np.asarray(w, float)
        _, F, T = self.tree.rnea(q, v, a, fext=fext, return_forces=True, kinematics=kin)
        return {rid: wrench_to_frame(T[s['index']] @ s['T'], F[s['index']]) for rid, s in self.sensors_of.items()}


def check_latch_stability(latches, q, dt, limit=STABILITY_LIMIT):
    """Stability ratio of the closed clamps; raises when the declared compliance is too stiff for the timestep."""
    ratio = latches.stability_ratio(q, dt)
    if ratio > limit:
        raise RuntimeError(f'the declared clamp compliance needs a smaller timestep: stability ratio {ratio:.2f} '
                           f'at dt = {dt:g} s exceeds {limit:g} (reduce the timestep or the clamp stiffness and damping)')
    return ratio
