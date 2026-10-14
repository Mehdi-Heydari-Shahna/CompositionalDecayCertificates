"""Ideal-constraint plant on Pinocchio dynamics.

The team moves under exact constraints: latched grasps are rigid, wheels roll
without slip on the support plane of each base, the payload sits rigidly on its
supports until the supports would have to pull, and lands on them when it
reaches them.  There is no contact solver, no compliance and no slip, so this
plant shows what the controller does when the mechanics are exactly the ideal
ones, and it is fast.

A base on conventional wheels (skid steering) is the exception to "no slip" and
to "no compliance".  Its wheels are not constrained to roll, and it stands on its
tyres:

* ``step``: every tyre carries its wheel through the declared normal stiffness and
  damping of its contact (a load, no ground row for such a base), and its friction
  is the tyre model of ``backends/tyre.py`` at that load.  Both are functions of
  the state at the start of the step;
* ``evaluate`` (accelerations and reactions at a given state, for the validation
  of the compiled interfaces) is the declared model of the compiler: the ground
  carries the base rigidly, and the friction is Coulomb's, built up over the
  declared slip speed, at wheel loads that are consistent with it.

Why such a base is not stepped on rigid ground.  The friction of a sliding wheel
is proportional to its load, and the friction forces shift load between the
wheels through the height of the centre of mass.  In a skid turn this loop
favours one diagonal of the base: a wheel whose friction points backwards and
to one side pushes load forwards and to the other side, onto itself.  For the
Husky with its arm the gain of the loop is about 0.85, so that a change of the
loads is amplified about sevenfold, and with front and rear wheels pushing
against each other it reaches ``mu h / (L / 2)`` = 0.8 x 0.38 / 0.256 = 1.18,
beyond which rigid-body mechanics has no answer.  On rigid ground the loads
follow the friction without any time scale of their own, and what a time
stepping scheme then does with the loop is its own property, not the
platform's: with the loads of the step before, three bases that hold one beam
rocked from one diagonal to the other at about 5 Hz in the turn in front of the
supports.  A tyre is not rigid.  Its normal compliance gives the loads the
dynamics of the base on its tyres, and with the declared tyre that turn is
steady, as it is on the soft contacts of the MuJoCo plant.

It is an independent implementation, not the compiled interfaces run again:

* inertia and bias come from Pinocchio;
* grasp and support rows are relative frame twists from Pinocchio frame
  Jacobians, rolling rows are the velocity of the wheel body's ground point;
* accelerations and multipliers come from a constrained (KKT) solve with a
  numerical constraint-drift term, followed by a projection onto the constraints;
* the slip and the deflection of a tyre are evaluated from the Pinocchio frame
  placement and Jacobian of its wheel body;
* the wrist sensor is the Newton-Euler balance of the tool-side body.

Only the declared ideal ground model is shared with the compiler: the
ground-following relation ``[z, pitch, roll] = support(x, y, yaw)`` of a base that
follows a fitted plane, or the terrain plane a wheel touches where every wheel
has its own contact row (``Terrain.wheel_contact``).  The contact rows themselves
are written here from Pinocchio frame Jacobians.  Declared and shared as well:
the friction law, the tyre model and, in ``evaluate``, the rule that gives every
wheel of a rigid four-wheel base its load (normal load affine in the wheel
position).
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.linalg import cho_factor, cho_solve

from teamc.se3 import inv
from teamc.stability import pad_loads
from teamc.terrain import Terrain

from .pin_model import PinTeam, _se3
from .tyre import TyreSet, check_tyre_stability

LWA = pin.LOCAL_WORLD_ALIGNED


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class PinPlant:
    name = 'pinocchio'

    def __init__(self, comp, timestep=.002, skid_ground='tyres'):
        """``skid_ground``: how a base on conventional wheels is stepped.  ``'tyres'``: it stands on the normal
        compliance of its tyres.  ``'rigid'``: rigid ground rows, tyre friction at the wheel loads of the step
        before; kept to reproduce what that does to three bases that hold one beam (module docstring)."""
        if skid_ground not in ('tyres', 'rigid'):
            raise ValueError("skid_ground must be 'tyres' or 'rigid'")
        self.skid_ground = skid_ground
        self.comp = comp
        self.pt = PinTeam(comp)
        self.model = self.pt.model
        self.n = self.pt.n
        self.dt = float(timestep)
        self.terrain = Terrain(comp.scene)
        ports = comp.cmg['ports']
        self.B = np.zeros((self.n, len(ports)))
        for k, p in enumerate(ports):
            self.B[p['index'], k] = p['gear']
        self._bounds = np.array([p['bounds'] for p in ports], float)
        self.damping = np.array([comp.cmg['damping'].get(k, 0.) for k in comp.cmg['coordinate_ids']])
        self.payload = np.asarray(comp.payload['base'], int)
        self.grasps, self.rests, self.robots = {}, {}, {}
        for cid, c in comp.connections.items():
            if c['kind'] == 'grasp':
                self.grasps[cid] = dict(f1=self.pt.frame(c['body1'], c['T1']), body2=c['body2'], f2=None, robot=c['robot'])
            else:
                self.rests[cid] = dict(f2=self.pt.frame(c['body2'], c['T2']), M=_se3(c['T1']), nominal=_se3(c['T1']), c=c)
        for rid, r in comp.robots.items():
            wheels = []
            friction = r.drive == 'friction'
            for w in r.wheels:
                if friction:
                    ty = w.get('tyre', {})
                    wheels.append(dict(id=w['id'], frame=self.pt.frame(r.prefix + w['body']), radius=w['radius_m'],
                                       index=w['index'], axis_base=np.asarray(w['axis_base'], float), mu=float(w.get('friction', .8)),
                                       slip_speed=float(ty.get('slip_speed_m_s', 1e-3)),
                                       kn=float(ty.get('contact_stiffness_N_m', 2.0e5)), cn=float(ty.get('contact_damping_N_s_m', 4.0e3))))
                    continue
                wheels.append(dict(id=w['id'], frame=self.pt.frame(r.prefix + w['body']), radius=w['radius_m'],
                                   g=np.asarray(w['traction_direction_base'], float), index=w['index'],
                                   carrier=self.pt.frame(w['carrier']), centre=np.asarray(w['centre_carrier'], float),
                                   axis=np.asarray(w['axis_carrier'], float),
                                   g_carrier=np.asarray(w['traction_direction_carrier'], float)))
            sensor = r.record['ports']['sensor']
            s_body, T_s = r.frames[sensor['frame']]
            body = next(b for b in r.record['bodies'] if r.prefix + b['id'] == s_body)
            Y = pin.Inertia(float(body['mass_kg']), np.asarray(body['com_m'], float), np.asarray(body['inertia_kg_m2'], float))
            self.robots[rid] = dict(base=r.base, base_frame=self.pt.frame(r.prefix + r.record['floating_root']), wheels=wheels,
                                    ground=r.ground, friction=friction,
                                    centres=np.array([w['centre_base'] for w in r.wheels]), radius=r.wheels[0]['radius_m'],
                                    sensor=self.pt.frame(s_body, T_s), Y=_se3(inv(np.asarray(T_s))).act(Y),
                                    T_sensor_tool=inv(np.asarray(T_s)) @ r.frames[r.record['ports']['tool']['frame']][1])
        self.data = self.model.createData()
        self.pt.data = self.data
        self.q, self.v = comp.seed(), np.zeros(self.n)
        self.f = np.zeros(len(ports))
        self.time = 0.
        self.latched = {cid: False for cid in self.grasps}
        self.rest = None
        self.saturated = 0
        self.obstacle = None
        self._lam = {}
        self._wheel = {}                                  # friction-driven bases: wheel loads and friction of the last step
        self.tyres = TyreSet(comp, self.terrain) or None
        self.tyre_ratio = 0.
        self._rigid_ground = False                        # True while the declared rigid model is evaluated (``evaluate``, ``balance``)
        self._wrench = {rid: np.zeros(6) for rid in self.robots}
        self.stats = dict(position_error=0., velocity_error=0.)

    # ------------------------------------------------------------------ state and inputs
    def reset(self, q, v=None):
        self.q = np.asarray(q, float).copy()
        self.v = np.zeros(self.n) if v is None else np.asarray(v, float).copy()
        self.time = 0.
        for cid in self.latched:
            self.latched[cid] = False
        self.rest = None
        # the payload starts on the supports it sits on
        self.pt.kinematics(self.q)
        for cid, r in self.rests.items():
            e = pin.log6(r['nominal'].actInv(self.data.oMf[r['f2']])).vector
            if np.linalg.norm(e) < 1e-6:
                self.rest = cid
                r['M'] = r['nominal']
        self._lam = {}
        self._wheel = {}
        if self.tyres:
            self.tyres.reset()
            self.tyre_ratio = check_tyre_stability(self.tyres, self.q, self.dt)
            if self.skid_ground == 'tyres':
                self._stand_on_tyres()
        self._wrench = {rid: np.zeros(6) for rid in self.robots}

    def _stand_on_tyres(self):
        """Lower every base on conventional wheels onto its tyres: the deflections that carry its own weight at rest."""
        tree = self.comp.tree
        g = float(np.linalg.norm(tree.gravity))
        for _ in range(3):
            T = tree.sweep(self.q)[0]
            frames = self._tyre_frames(self.q, update=True)
            for rid, wheels in frames.items():
                r = self.robots[rid]
                prefix = self.comp.robots[rid].prefix
                bodies = [i for i in tree.massive if tree.names[i].startswith(prefix)]
                m = tree.mass_kg[bodies]
                com = (m[:, None] * np.array([T[i, :3, 3] + T[i, :3, :3] @ tree.com[i] for i in bodies])).sum(axis=0) / m.sum()
                Tb = T[tree.index[prefix + self.comp.robots[rid].record['floating_root']]]
                c = Tb[:3, :3].T @ (com - Tb[:3, 3])
                Phi = np.c_[np.ones(len(r['centres'])), r['centres'][:, :2]]
                loads = Phi @ np.linalg.solve(Phi.T @ Phi, m.sum() * g * np.array([1., c[0], c[1]]))
                want = loads / np.array([w['kn'] for w in r['wheels']])
                have = np.array([x['deflection'] for x in wheels])
                D = np.array([x['Jn'][r['base'][[2, 4, 5]]] for x in wheels])         # wheel centre along the normal per z, pitch, roll
                self.q[r['base'][[2, 4, 5]]] += np.linalg.lstsq(D, have - want, rcond=None)[0]

    def kick(self, dv):
        """Add ``dv`` (compiled rates) to the velocity of the plant: what an impulse does (disturbance tests)."""
        self.v = self.v + np.asarray(dv, float)

    def state(self):
        return self.q.copy(), self.v.copy()

    def measure(self):
        return dict(t=self.time, q=self.q.copy(), v=self.v.copy(), wrench={rid: w.copy() for rid, w in self._wrench.items()})

    def set_efforts(self, f):
        f = np.asarray(f, float)
        clipped = np.clip(f, self._bounds[:, 0], self._bounds[:, 1])
        self.saturated += int(np.any(clipped != f))
        self.f = clipped
        return clipped

    def set_obstacle(self, position):
        self.obstacle = np.asarray(position, float).copy()

    def latch(self, cid):
        g = self.grasps[cid]
        self.pt.kinematics(self.q)
        jid, B = self.pt.body[g['body2']]
        T_body = self.data.oMi[jid].homogeneous @ B
        T2 = inv(T_body) @ self.data.oMf[g['f1']].homogeneous
        g['f2'] = self.pt.frame(g['body2'], T2, name=('latched', cid, self.time))
        self.data = self.pt.data
        self.latched[cid] = True
        return T2

    def unlatch(self, cid):
        self.latched[cid] = False

    # ------------------------------------------------------------------ constraints
    def _constraints(self, q, v):
        """Stacked rows ``J``, position errors ``e`` (zero for velocity-level rows) and the row blocks."""
        model, data = self.model, self.data
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        rows, err, blocks = [], [], []
        m = 0

        def add(key, J, e, holonomic):
            nonlocal m
            rows.append(J)
            err.append(e)
            blocks.append((key, slice(m, m + len(e)), holonomic))
            m += len(e)

        for cid, g in self.grasps.items():
            if not self.latched[cid]:
                continue
            M12 = data.oMf[g['f1']].actInv(data.oMf[g['f2']])
            J = M12.toActionMatrix() @ pin.getFrameJacobian(model, data, g['f2'], pin.LOCAL) \
                - pin.getFrameJacobian(model, data, g['f1'], pin.LOCAL)
            add(cid, J, pin.log6(M12).vector, True)
        if self.rest is not None:
            r = self.rests[self.rest]
            Mrel = r['M'].actInv(data.oMf[r['f2']])
            add(self.rest, Mrel.toActionMatrix() @ pin.getFrameJacobian(model, data, r['f2'], pin.LOCAL), pin.log6(Mrel).vector, True)
        for rid, r in self.robots.items():
            if r['ground'] == 'contact':
                # one row per wheel: the point of the wheel carrier at the contact does not move along the terrain normal;
                # and the point of the wheel there does not move along the roller axis
                nw = len(r['wheels'])
                Jg, eg, Jw = np.zeros((nw, self.n)), np.zeros(nw), np.zeros((nw, self.n))
                for k, w in enumerate(r['wheels']):
                    Mc = data.oMf[w['carrier']]
                    centre = Mc.translation + Mc.rotation @ w['centre']
                    eg[k], normal, point = self.terrain.wheel_contact(centre, Mc.rotation @ w['axis'], w['radius'])
                    Jc = pin.getFrameJacobian(model, data, w['carrier'], LWA)      # carrier twist at its origin, world axes
                    Jg[k] = normal @ Jc[:3] + _cross(point - Mc.translation, normal) @ Jc[3:]
                    Jf = pin.getFrameJacobian(model, data, w['frame'], LWA)        # wheel twist at its centre, world axes
                    g = Mc.rotation @ w['g_carrier']
                    g = g - (g @ normal) * normal
                    g /= np.linalg.norm(g)
                    Jw[k] = g @ Jf[:3] + _cross(point - data.oMf[w['frame']].translation, g) @ Jf[3:]
                add(f'{rid}:ground', Jg, eg, True)
                add(f'{rid}:wheels', Jw, np.zeros(nw), False)
                continue
            if r['friction'] and not self._rigid_ground and self.skid_ground == 'tyres':
                continue                                                      # the base stands on its tyres: loads, not rows
            base = r['base']
            xy_yaw = q[base[[0, 1, 3]]]
            value, normal, _ = self.terrain.support(xy_yaw, r['centres'], r['radius'])
            r['normal'] = normal
            G = self.terrain.support_jacobian(xy_yaw, r['centres'], r['radius'])
            J = np.zeros((3, self.n))
            J[np.arange(3), base[[2, 4, 5]]] = 1.
            J[:, base[[0, 1, 3]]] -= G
            add(f'{rid}:ground', J, q[base[[2, 4, 5]]] - value, True)
            if r['friction']:                                                 # wheels that slide: friction, not constraints
                continue
            Rb = data.oMf[r['base_frame']].rotation
            Jw = np.zeros((len(r['wheels']), self.n))
            for k, w in enumerate(r['wheels']):
                Jf = pin.getFrameJacobian(model, data, w['frame'], LWA)      # wheel body twist at its centre, world axes
                g = Rb @ w['g']
                g = g - (g @ normal) * normal
                g /= np.linalg.norm(g)
                arm = -w['radius'] * normal                                   # centre -> ground point
                Jw[k] = g @ Jf[:3] + _cross(arm, g) @ Jf[3:]
            add(f'{rid}:wheels', Jw, np.zeros(len(r['wheels'])), False)
        if not rows:
            return np.zeros((0, self.n)), np.zeros(0), []
        return np.vstack(rows), np.concatenate(err), blocks

    # ------------------------------------------------------------------ friction-driven bases
    def _slip_rows(self, q):
        """Per friction-driven base: slip rows (two per wheel: rolling direction, axle) and the load map of its wheels.

        Uses the joint Jacobians and the support normals of the last :meth:`_constraints` call, which must have
        been made at the same ``q``.  The load map ``A`` (3 x wheels) takes wheel normal loads to the generalized
        forces on z, pitch and roll of the base.
        """
        model, data = self.model, self.data
        out = {}
        for rid, r in self.robots.items():
            if not r['friction']:
                continue
            base, normal = r['base'], r['normal']
            Mb = data.oMf[r['base_frame']]
            Jb = pin.getFrameJacobian(model, data, r['base_frame'], LWA)
            nw = len(r['wheels'])
            Js, A, frames = np.zeros((2 * nw, self.n)), np.zeros((3, nw)), []
            for k, w in enumerate(r['wheels']):
                Mw = data.oMf[w['frame']]
                Jf = pin.getFrameJacobian(model, data, w['frame'], LWA)       # wheel body twist at its centre, world axes
                axis = Mb.rotation @ w['axis_base']
                arm = -w['radius'] * normal                                    # centre -> ground point
                roll = _cross(axis, normal)
                side = axis - (axis @ normal) * normal
                roll, side = roll / np.linalg.norm(roll), side / np.linalg.norm(side)
                for i, g in enumerate((roll, side)):
                    Js[2 * k + i] = g @ Jf[:3] + _cross(arm, g) @ Jf[3:]
                point = Mw.translation + arm - Mb.translation                  # ground point from the base origin
                A[:, k] = (normal @ Jb[:3] + _cross(point, normal) @ Jb[3:])[base[[2, 4, 5]]]
                frames.append((Mw.translation + arm, normal, roll, side))
            out[rid] = (Js, A, frames)
        return out

    def _load_map(self, rid, A):
        """``P`` (wheels x 3): the normal load of every wheel from the ground reaction of its base, affine in the wheel position."""
        centres = self.robots[rid]['centres']
        Phi = np.c_[np.ones(len(centres)), centres[:, :2]]
        return Phi @ np.linalg.inv(A @ Phi)

    def _tyre_frames(self, q, update=False):
        """Per base on conventional wheels, per wheel: the ground kinematics of its tyre at ``q``.

        Uses the joint Jacobians of the last :meth:`_constraints` call, which must have been made at the same
        ``q`` (``update``: compute them here).  Each entry has the ground point, the terrain normal under the
        wheel, the rolling and the axle direction in the ground plane, the deflection of the tyre (how far the
        wheel centre is closer to the terrain than its radius) and the rows of the wheel body's ground point
        on the coordinates: along the normal (``Jn``) and along the two ground directions (``Js``).
        """
        model, data = self.model, self.data
        if update:
            pin.computeJointJacobians(model, data, q)
            pin.updateFramePlacements(model, data)
        out = {}
        for rid, r in self.robots.items():
            if not r['friction']:
                continue
            Mb = data.oMf[r['base_frame']]
            wheels = []
            for w in r['wheels']:
                centre = data.oMf[w['frame']].translation
                patch, _ = self.terrain.plane_at(float(centre[0]), float(centre[1]))
                normal = np.asarray(patch['n'], float)
                Jf = pin.getFrameJacobian(model, data, w['frame'], LWA)       # wheel body twist at its centre, world axes
                axis = Mb.rotation @ w['axis_base']
                arm = -w['radius'] * normal                                    # centre -> ground point
                roll = _cross(axis, normal)
                side = axis - (axis @ normal) * normal
                roll, side = roll / np.linalg.norm(roll), side / np.linalg.norm(side)
                Js = np.array([g @ Jf[:3] + _cross(arm, g) @ Jf[3:] for g in (roll, side)])
                wheels.append(dict(point=centre + arm, normal=normal, roll=roll, side=side, Js=Js, Jn=normal @ Jf[:3],
                                   deflection=float(w['radius'] - (normal @ centre - patch['d']))))
            out[rid] = wheels
        return out

    def _tyres(self, frames, v):
        """Generalized force of the tyres for this step and, per base, ``(loads, friction forces)``.

        A tyre carries its wheel with the declared normal stiffness and damping of its contact (no pull), and
        its friction is the tyre model of ``backends/tyre.py`` at that load.  Both are functions of the state
        at the start of the step.
        """
        state, loads, normal_force = {}, {}, np.zeros(self.n)
        for rid, wheels in frames.items():
            for w, x in zip(self.robots[rid]['wheels'], wheels):
                N = max(w['kn'] * x['deflection'] - w['cn'] * float(x['Jn'] @ v), 0.) if x['deflection'] > 0. else 0.
                s = x['Js'] @ v
                state[(rid, w['id'])] = (x['point'], x['normal'], s[0] * x['roll'] + s[1] * x['side'], float(v[w['index']]))
                loads[(rid, w['id'])] = N
                normal_force += x['Jn'] * N
        out = self.tyres.advance([state[w['key']] for w in self.tyres.wheels], loads, self.dt)
        Q, detail = normal_force, {}
        for rid, wheels in frames.items():
            N, F = np.zeros(len(wheels)), np.zeros((len(wheels), 2))
            for k, (w, x) in enumerate(zip(self.robots[rid]['wheels'], wheels)):
                on_wheel = -out[('tyre', rid, w['id'])]['force']
                N[k] = loads[(rid, w['id'])]
                F[k] = on_wheel @ x['roll'], on_wheel @ x['side']
                Q = Q + x['Js'].T @ F[k]
            detail[rid] = (N, F)
        return Q, detail

    def _tyre_loads(self, slip_rows, v, blocks, reactions):
        """Rigid ground (``skid_ground='rigid'``): generalized force of the tyres and, per base, ``(loads, forces)``.

        The wheel loads are those of the ground reactions of the step before; at the first step, those of
        this state without friction (``reactions`` returns the multipliers for an added generalized force).
        The slip of every tyre and its ground point come from the Pinocchio frame Jacobians.
        """
        ground = {key: self._lam[key] for key, _, _ in blocks if key in self._lam}
        if any(f'{rid}:ground' not in ground for rid in slip_rows):
            lam0 = reactions(np.zeros(self.n))
            ground = {key: lam0[rows] for key, rows, _ in blocks}
        state, loads, order = {}, {}, []
        for rid, (Js, A, frames) in slip_rows.items():
            N = np.maximum(self._load_map(rid, A) @ ground[f'{rid}:ground'], 0.)
            s = (Js @ v).reshape(-1, 2)
            for k, w in enumerate(self.robots[rid]['wheels']):
                point, normal, roll, side = frames[k]
                state[(rid, w['id'])] = (point, normal, s[k, 0] * roll + s[k, 1] * side, float(v[w['index']]))
                loads[(rid, w['id'])] = float(N[k])
            order.append((rid, N))
        out = self.tyres.advance([state[w['key']] for w in self.tyres.wheels], loads, self.dt)
        Q, detail = np.zeros(self.n), {}
        for rid, N in order:
            Js, _, frames = slip_rows[rid]
            F = np.zeros((len(N), 2))
            for k, w in enumerate(self.robots[rid]['wheels']):
                on_wheel = -out[('tyre', rid, w['id'])]['force']
                F[k] = on_wheel @ frames[k][2], on_wheel @ frames[k][3]
            Q += Js.T @ F.ravel()
            detail[rid] = (N, F)
        return Q, detail

    def _friction(self, slip_rows, v, blocks, reactions):
        """Declared friction law of the wheels of the friction-driven bases at the rates ``v`` (for ``evaluate``).

        The friction of a wheel is proportional to its normal load, and the loads follow from the ground
        reactions, which the friction forces change in turn (they act below the base and shift load between
        the wheels).  ``reactions(Q)`` returns the multipliers of all rows for an added generalized force
        ``Q``; they are affine in it, so the loads that are consistent with their own friction solve a small
        linear system.  A wheel whose load comes out negative has left the ground: it is taken out and the
        others are solved again.

        Returns the generalized force and, per base, ``(loads, forces (wheels, 2))``.
        """
        order, unit, columns, maps = [], [], [], {}
        ground = {key: rows for key, rows, _ in blocks}
        for rid, (Js, A, _) in slip_rows.items():
            s = (Js @ v).reshape(-1, 2)
            maps[rid] = self._load_map(rid, A)
            for k, w in enumerate(self.robots[rid]['wheels']):
                rho = np.sqrt(s[k] @ s[k] + w['slip_speed'] ** 2)
                order.append((rid, k))
                unit.append(-w['mu'] * s[k] / rho)                                 # friction per unit load
                columns.append(Js[2 * k:2 * k + 2].T @ unit[-1])
        lam0 = reactions(np.zeros(self.n))
        N0 = np.concatenate([maps[rid] @ lam0[ground[f'{rid}:ground']] for rid in slip_rows])
        offsets = np.cumsum([0] + [len(self.robots[rid]['wheels']) for rid in slip_rows])
        G = np.zeros((len(order), len(order)))
        for j, column in enumerate(columns):
            d = reactions(column) - lam0
            G[:, j] = np.concatenate([maps[rid] @ d[ground[f'{rid}:ground']] for rid in slip_rows])
        active = np.ones(len(order), bool)
        for _ in range(len(order)):
            N = np.zeros(len(order))
            N[active] = np.linalg.solve((np.eye(len(order)) - G)[np.ix_(active, active)], N0[active])
            if N.min() >= 0.:
                break
            active &= N > 0.
        Q, detail = np.zeros(self.n), {}
        for i, rid in enumerate(slip_rows):
            Js = slip_rows[rid][0]
            Nr = N[offsets[i]:offsets[i + 1]]
            F = np.array([Nr[k] * unit[offsets[i] + k] for k in range(len(Nr))])
            Q += Js.T @ F.ravel()
            detail[rid] = (Nr, F)
        return Q, detail

    # ------------------------------------------------------------------ step
    def _solve(self, q, v):
        J, e, blocks = self._constraints(q, v)
        M, h = self.pt.terms(q, v)
        tau = self.B @ self.f - self.damping * v - h
        chol = cho_factor(M)
        hol = np.concatenate([np.arange(b.start, b.stop) for _, b, holonomic in blocks if holonomic]) if blocks else np.zeros(0, int)
        return J, e, blocks, M, tau, chol, hol

    def evaluate(self, q, v, f, latched=(), rest=None, frames=None):
        """Constrained accelerations and multipliers at a given state (no integration, no projection).

        ``latched`` grasps use the compiled payload-side frames (or ``frames``), ``rest`` names an
        active support.  Multipliers are returned per row block in the compiled convention
        ``[moment; force]`` for connections.
        """
        saved = (self.q, self.v, self.f, dict(self.latched), self.rest)
        try:
            for cid, g in self.grasps.items():
                self.latched[cid] = cid in latched
                if cid in latched:
                    T2 = self.comp.connections[cid]['T2'] if frames is None else frames[cid]
                    g['f2'] = self.pt.frame(g['body2'], T2)
            self.data = self.pt.data
            self.rest = rest
            if rest is not None:
                self.rests[rest]['M'] = self.rests[rest]['nominal']
            self.f = np.asarray(f, float)
            self._rigid_ground = True
            J, e, blocks, M, tau, chol, hol = self._solve(q, v)
            slip_rows = self._slip_rows(q)
            eps = 1e-6 / max(1., float(np.max(np.abs(v))))
            Jp, _, _ = self._constraints(q + eps * v, v)
            Jm, _, _ = self._constraints(q - eps * v, v)
            gamma = ((Jp - Jm) @ v) / (2. * eps)
            MiJt = cho_solve(chol, J.T)
            lam = np.linalg.solve(J @ MiJt, -gamma - J @ cho_solve(chol, tau))
            wheels = {}
            if slip_rows:
                # friction at this state, at the wheel loads that are consistent with it
                K = J @ MiJt
                Q, detail = self._friction(slip_rows, v, blocks,
                                           lambda extra: np.linalg.solve(K, -gamma - J @ cho_solve(chol, tau + extra)))
                tau = tau + Q
                lam = np.linalg.solve(K, -gamma - J @ cho_solve(chol, tau))
                wheels = {f'{rid}:wheels': F.ravel() for rid, (_, F) in detail.items()}
            a = cho_solve(chol, tau + J.T @ lam)
            ports = dict(wheels)
            for key, rows, _ in blocks:
                x = lam[rows]
                ports[key] = np.r_[x[3:], x[:3]] if len(x) == 6 else x.copy()
            return dict(a=a, ports=ports, J=J, residual=e, velocity_residual=J @ v)
        finally:
            self._rigid_ground = False
            self.q, self.v, self.f, self.latched, self.rest = saved[0], saved[1], saved[2], saved[3], saved[4]

    def balance(self, q, v, a, f, ports, latched=(), rest=None, frames=None):
        """Unbalanced generalized force of a motion with given efforts and port reactions, in this plant's equations.

        ``ports`` are reactions in the compiled convention (``[moment; force]`` for connections; for the wheels
        of a friction-driven base the two tangential forces of every wheel).  Returns the largest component of
        ``M a + h - B f - J^T lam`` with inertia and bias from Pinocchio and the rows written here: a check that
        does not solve anything.
        """
        saved = (self.q, self.v, self.f, dict(self.latched), self.rest)
        try:
            for cid, g in self.grasps.items():
                self.latched[cid] = cid in latched
                if cid in latched:
                    T2 = self.comp.connections[cid]['T2'] if frames is None else frames[cid]
                    g['f2'] = self.pt.frame(g['body2'], T2)
            self.data = self.pt.data
            self.rest = rest
            if rest is not None:
                self.rests[rest]['M'] = self.rests[rest]['nominal']
            self._rigid_ground = True
            J, _, blocks = self._constraints(q, v)
            slip_rows = self._slip_rows(q)
            M, h = self.pt.terms(q, v)
            out = M @ a + h + self.damping * v - self.B @ np.asarray(f, float)
            for key, rows, _ in blocks:
                x = np.asarray(ports[key], float)
                out -= J[rows].T @ (np.r_[x[3:], x[:3]] if len(x) == 6 else x)
            for rid, (Js, _, _) in slip_rows.items():
                out -= Js.T @ np.asarray(ports[f'{rid}:wheels'], float)
            return float(np.max(np.abs(out)))
        finally:
            self._rigid_ground = False
            self.q, self.v, self.f, self.latched, self.rest = saved[0], saved[1], saved[2], saved[3], saved[4]

    def step(self):
        dt = self.dt
        q, v = self.q, self.v
        J, e, blocks, M, tau, chol, hol = self._solve(q, v)
        sprung = bool(self.tyres) and self.skid_ground == 'tyres'
        frames = self._tyre_frames(q) if sprung else {}
        slip_rows = self._slip_rows(q) if self.tyres and not sprung else {}
        # ---- projection of the state onto the constraints (removes integration drift)
        if len(hol):
            Jh = J[hol]
            MiJt = cho_solve(chol, Jh.T)
            q = q - MiJt @ np.linalg.solve(Jh @ MiJt, e[hol])
            self.stats['position_error'] = max(self.stats['position_error'], float(np.max(np.abs(e[hol]))))
        rows = J.shape[0] > 0                           # (a team on tyres with nothing latched and no support has none)
        gamma = np.zeros(0)
        if rows:
            MiJt = cho_solve(chol, J.T)
            A = J @ MiJt
            vel = J @ v
            self.stats['velocity_error'] = max(self.stats['velocity_error'], float(np.max(np.abs(vel))))
            v = v - MiJt @ np.linalg.solve(A, vel)
            # ---- constraint drift (J_dot v) by a centred difference along the motion
            eps = 1e-6 / max(1., float(np.max(np.abs(v))))
            Jp, _, _ = self._constraints(q + eps * v, v)
            Jm, _, _ = self._constraints(q - eps * v, v)
            gamma = ((Jp - Jm) @ v) / (2. * eps)
        # ---- tyres of conventional wheels: they carry the base and supply its friction (explicit loads)
        detail = {}
        if frames or slip_rows:
            kept = self.tyres.snapshot()
            if frames:
                Q, detail = self._tyres(frames, v)
            else:
                Q, detail = self._tyre_loads(slip_rows, v, blocks,
                                             lambda extra: np.linalg.solve(A, -gamma - J @ cho_solve(chol, tau + extra)))
            tau = tau + Q
        # ---- constrained accelerations and multipliers
        lam = np.linalg.solve(A, -gamma - J @ cho_solve(chol, tau)) if rows else np.zeros(0)
        if self.rest is not None:
            key, rows, _ = next(b for b in blocks if b[0] == self.rest)
            if lam[rows][2] < 0.:                      # the supports would have to pull: the payload leaves them
                self.rest = None
                self.q, self.v = q, v
                if frames or slip_rows:
                    self.tyres.restore(kept)           # the step is taken again: the tyres have not moved yet
                return self.step()
        a = cho_solve(chol, tau + J.T @ lam)
        self._lam = {key: lam[rows].copy() for key, rows, _ in blocks}
        self._wheel = detail
        self._sensors(q, v, a)
        self.v = v + a * dt
        self.q = q + self.v * dt
        self.time += dt
        self._land()

    def _land(self):
        """The payload reaches a support from above: it sits on it from now on (as-landed pose kept in the plane)."""
        if self.rest is not None:
            return
        self.pt.kinematics(self.q, self.v)
        for cid, r in self.rests.items():
            Mrel = r['nominal'].actInv(self.data.oMf[r['f2']])
            p = Mrel.translation
            vz = (r['nominal'].rotation.T @ pin.getFrameVelocity(self.model, self.data, r['f2'], LWA).linear)[2]
            if p[2] <= 0. and abs(p[0]) < .05 and abs(p[1]) < .05 and vz < 0. and abs(pin.log3(Mrel.rotation)).max() < .05:
                yaw = np.arctan2(Mrel.rotation[1, 0], Mrel.rotation[0, 0])
                c, s = np.cos(yaw), np.sin(yaw)
                landed = pin.SE3(np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]), np.array([p[0], p[1], 0.]))
                r['M'] = r['nominal'] * landed
                self.rest = cid
                return

    # ------------------------------------------------------------------ measurements
    def _sensors(self, q, v, a):
        model, data = self.model, self.data
        pin.forwardKinematics(model, data, q, v, a)
        pin.updateFramePlacements(model, data)
        for rid, r in self.robots.items():
            fid = r['sensor']
            nu = pin.getFrameVelocity(model, data, fid, pin.LOCAL)
            alpha = pin.getFrameAcceleration(model, data, fid, pin.LOCAL)
            grav = pin.Motion(data.oMf[fid].rotation.T @ model.gravity.linear, np.zeros(3))
            w = r['Y'] * (alpha - grav) + r['Y'].vxiv(nu)                 # what the tool-side body needs
            w = np.r_[w.linear, w.angular]
            cid = next(c for c, g in self.grasps.items() if g['robot'] == rid)
            if self.latched[cid] and cid in self._lam:
                X = _se3(r['T_sensor_tool'])                              # tool frame in the sensor frame
                lam = pin.Force(self._lam[cid][:3], self._lam[cid][3:])   # on the payload, tool axes
                on_payload = X.act(lam)
                w = w + np.r_[on_payload.linear, on_payload.angular]      # the tool-side body passes it on
            self._wrench[rid] = np.r_[w[3:], w[:3]]                       # [moment; force]

    def sensor_wrench(self, rid):
        return self._wrench[rid].copy()

    def weld_wrench(self, cid):
        """Wrench a latched grasp applies to the payload: force (world) and moment about the payload origin (world)."""
        if not self.latched[cid] or cid not in self._lam:
            return np.zeros(3), np.zeros(3)
        g = self.grasps[cid]
        T = self.data.oMf[g['f1']]
        f, m = T.rotation @ self._lam[cid][:3], T.rotation @ self._lam[cid][3:]
        jid, B = self.pt.body[g['body2']]
        origin = (self.data.oMi[jid].homogeneous @ B)[:3, 3]
        return f, m + np.cross(T.translation - origin, f)

    def contacts(self):
        """Support loads implied by the plant's own multipliers (there is no contact solver here)."""
        out = dict(pads={}, wheels={}, guides=0., forbidden=[], tool=0.)
        if self.rest is not None and self.rest in self._lam:
            lam = self._lam[self.rest]
            c = dict(self.rests[self.rest]['c'], T1=self.rests[self.rest]['M'].homogeneous)
            out['pads'] = pad_loads(c, np.r_[lam[3:], lam[:3]])['pad_normal_N']
        for rid, (N, F) in self._wheel.items():        # friction-driven bases: wheel loads and friction of the last step
            for k, w in enumerate(self.robots[rid]['wheels']):
                out['wheels'][(rid, w['id'])] = [float(N[k]), float(np.linalg.norm(F[k]))]
        return out
