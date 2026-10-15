"""Tyres of conventional wheels as explicit loads (NumPy only).

A base on conventional wheels turns by sliding them (``teamc/skid.py``).  The
friction of a sliding tyre must then be Coulomb's in every direction of the
ground plane at once.  Rigid-contact friction that is solved as two independent
rows per contact point (PhysX) does not give that: a four-wheel base whose load
is shifted towards one axle turns at a fraction of the rate Coulomb friction
gives, and the fraction changes with the rate (``tests/test_skid_physx.py``).
Vehicle simulation therefore computes tyre forces from the slip and applies them
as loads.  This module is that tyre model for the plants: the Isaac Sim plant
always uses it for conventional wheels (by default the tyre also carries the
wheel, which then has no collider; on a PhysX wheel contact the collider is
frictionless), and the MuJoCo plant uses it by default (``tyre='model'``);
MuJoCo's own friction (``tyre='native'``) is the comparison
(``tests/test_skid_plant.py``).

Tyre model: Coulomb friction with an elastic range and relaxation.  The ground
point of a tyre holds on to the ground through the tangential stiffness of the
carcass: a deflection ``e`` (a vector in the ground plane) with the force

    F = -k e - c s,        de/dt = s - (r |w| / L) e,

``s`` being the velocity of the wheel's material point at the ground (the slip),
``r |w|`` the speed at which the tread runs through the contact and ``L`` the
relaxation length: a turning wheel brings undeflected tread into the contact, so
a deflection is renewed over the rolled distance ``L``.  While ``|F| <= mu N`` the
tyre grips.  When the force would exceed ``mu N`` the tyre slides: the force
stays on the friction circle, ``F = mu N F / |F|``, and the deflection is the one
that force stands for.

Three regimes follow.  A wheel at rest holds a force with its deflection (static
friction).  A rolling wheel with a small slip settles at ``e = L s / (r |w|)``,
a force proportional to the slip angle (cornering stiffness ``k L``).  A wheel
with a large slip slides with ``F = -mu N s / |s|``, the law of the compiled
model; a base that turns by skidding is in this regime at every wheel.

The force is evaluated from the state at the start of a step and applied during
the step at the ground point of the wheel.  The explicit loads limit the
stiffness and the damping a timestep allows (``stability_ratio``, as for the
clamps of ``backends/latch.py``).

Normal load.  Where the plant carries the wheel on a rigid contact (``carry``
False), the normal load is the one the plant reported for the step before.  With
``carry`` True the tyre carries its wheel itself: the wheel has no collider, and
the normal load is the declared normal spring and damper of the tyre,

    N = max(kn d + cn d', 0),

``d`` being the depth of the wheel (a disc of the wheel radius, as in the
compiled model) below the plane of the compiled terrain it touches.  A rigid
chassis on four rigid contacts leaves the share of its two wheel diagonals to
the contact solver; on four equal springs it is determined, and it is the
equal-stiffness distribution of the compiled model.  The normal force acts at
the same point as the friction, along the terrain normal; both are one load of
the wheel.

Not modelled: a contact patch and its aligning moment, rolling resistance, a
dependence of the friction coefficient on load or slip speed.
"""
from __future__ import annotations

import numpy as np

from teamc.terrain import Terrain

STABILITY_LIMIT = .6           # largest accepted stability ratio of the explicit tyre loads (1 is the linear limit)
STAND_TOLERANCE_M = .005       # a base whose wheels are all within this of the ground is lowered onto its tyres at reset


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class TyreSet:
    """The tyres of the friction-driven bases of a compiled team, evaluated in compiled coordinates.

    ``carry``: the tyres carry their wheels (normal spring and damper) instead of a contact of the plant.
    """

    def __init__(self, comp, terrain=None, carry=False):
        self.comp = comp
        self.tree = comp.tree
        self.terrain = Terrain(comp.scene) if terrain is None else terrain
        self.carry = bool(carry)
        self.wheels = []
        for rid, r in comp.robots.items():
            if r.drive != 'friction':
                continue
            for w in r.wheels:
                ty = w.get('tyre', {})
                i = self.tree.index[r.prefix + w['body']]
                # wheel axis in the wheel body frame: the joint axis carried through the frame that leaves the joint
                axis = self.tree.leave[i][:3, :3].T @ self.tree.axis[i]
                self.wheels.append(dict(key=(rid, w['id']), body=r.prefix + w['body'], index=i,
                                        coordinate=int(w['index']), radius=float(w['radius_m']), mu=float(w.get('friction', .8)),
                                        k=float(ty.get('stiffness_N_m', 1.5e5)), c=float(ty.get('damping_N_s_m', 1.5e3)),
                                        relax=float(ty.get('relaxation_length_m', 0.)),
                                        kn=float(ty.get('contact_stiffness_N_m', 2.0e5)),
                                        cn=float(ty.get('contact_damping_N_s_m', 4.0e3)),
                                        axis=axis / np.linalg.norm(axis)))
        self.e = {w['key']: np.zeros(3) for w in self.wheels}
        self.sliding = {w['key']: False for w in self.wheels}
        self.loaded = set()                                    # tyres that have carried load (carry mode)

    def __bool__(self):
        return bool(self.wheels)

    def reset(self):
        for key in self.e:
            self.e[key] = np.zeros(3)
            self.sliding[key] = False
        self.loaded = set()

    def snapshot(self):
        """The state of the tyres (deflections, sliding flags), to take a step again from the same state."""
        return {key: e.copy() for key, e in self.e.items()}, dict(self.sliding)

    def restore(self, kept):
        self.e = {key: e.copy() for key, e in kept[0].items()}
        self.sliding = dict(kept[1])

    def _contact(self, w, T):
        """Ground point and terrain normal of a tyre (a sphere about the wheel centre)."""
        return self.ground_point(w, T[w['index'], :3, 3])

    def ground_point(self, w, centre):
        """Ground point and terrain normal of a tyre whose wheel centre is at ``centre``."""
        patch, _ = self.terrain.plane_at(float(centre[0]), float(centre[1]))
        return centre - w['radius'] * patch['n'], patch['n']

    def disc_contact(self, w, T):
        """Contact of a tyre that carries its wheel: ``(clearance, normal, point)`` of the wheel disc against the terrain.

        The disc of the wheel radius about the wheel axis, as in the compiled model (``Terrain.wheel_contact``):
        the clearance is negative where the tyre is pressed into the ground.
        """
        R = T[w['index'], :3, :3]
        return self.terrain.wheel_contact(T[w['index'], :3, 3], R @ w['axis'], w['radius'])

    def _velocity(self, w, S, v, point):
        V = np.zeros(6)
        for k in self.tree.support[w['index']]:
            V += S[k] * v[self.tree.coord[k]]
        return V[3:] + _cross(V[:3], point)                    # velocity of the wheel's material point at ``point``

    def _slip(self, w, S, v, point, normal):
        s = self._velocity(w, S, v, point)
        return s - (s @ normal) * normal

    def loads(self, q, v, normal_loads, dt, kin=None):
        """Tyre loads at a state for the given normal loads ``{(robot, wheel): N}``; advances the deflections by ``dt``.

        Returns ``{key: dict(point, force, moment, on, by)}`` in the form of ``LatchSet.loads``: ``by`` is the wheel
        body, which receives ``-force`` (``force`` is what the tyre applies to the ground); ``on`` is ``None``.
        """
        T, S = kin if kin is not None else self.tree.sweep(q)
        state = []
        for w in self.wheels:
            point, normal = self._contact(w, T)
            state.append((point, normal, self._slip(w, S, v, point, normal), float(v[w['coordinate']])))
        return self.advance(state, normal_loads, dt)

    def carried_loads(self, q, v, dt, kin=None):
        """Loads of tyres that carry their wheels: normal spring and damper, and the friction at that load.

        The normal load of a tyre follows from its depth below the terrain plane and the rate of that depth; the
        friction is the tyre model at this load (of the same step).  Both act at the contact point of the wheel
        disc, as one load per wheel: ``force`` is what the tyre applies to the ground, ``tangential`` its part in
        the ground plane, ``load`` the normal load.  Advances the deflections by ``dt``.
        """
        T, S = kin if kin is not None else self.tree.sweep(q)
        state, normal_loads, normals = [], {}, {}
        for w in self.wheels:
            clearance, normal, point = self.disc_contact(w, T)
            velocity = self._velocity(w, S, v, point)
            depth, rate = -clearance, -float(normal @ velocity)
            N = max(w['kn'] * depth + w['cn'] * rate, 0.) if depth > 0. else 0.
            normal_loads[w['key']] = N
            normals[w['key']] = (N, normal, depth)
            state.append((point, normal, velocity - (velocity @ normal) * normal, float(v[w['coordinate']])))
        out = self.advance(state, normal_loads, dt)
        for key, load in out.items():
            N, normal, depth = normals[key[1:]]
            load['force'] = load['tangential'] - N * normal       # the ground pushes the wheel out of it with N
            load['depth'] = depth
            if N > 0.:
                self.loaded.add(key[1:])
        return out

    def advance(self, state, normal_loads, dt):
        """Tyre loads from the ground kinematics of every wheel; advances the deflections by ``dt``.

        ``state``: per wheel, in the order of ``self.wheels``, ``(ground point, terrain normal, slip, spin rate)``
        with the slip in the ground plane.  A plant that has these from its own state passes them here;
        :meth:`loads` computes them from compiled coordinates.
        """
        out = {}
        for w, (point, normal, s, spin) in zip(self.wheels, state):
            key = w['key']
            N = float(normal_loads.get(key, 0.))
            if N <= 0.:
                self.e[key] = np.zeros(3)
                self.sliding[key] = False
                F = np.zeros(3)
            else:
                e, gain = self.e[key] - (self.e[key] @ normal) * normal, dt
                if w['relax'] > 0.:                           # the tread in the contact is renewed as the wheel turns
                    a = abs(spin) * w['radius'] / w['relax']
                    if a * dt > 1e-9:
                        decay = np.exp(-a * dt)
                        e, gain = e * decay, (1. - decay) / a
                e = e + s * gain
                F = -w['k'] * e - w['c'] * s                  # on the wheel
                size, limit = float(np.linalg.norm(F)), w['mu'] * N
                self.sliding[key] = size > limit
                if size > limit:
                    F = F * (limit / size)
                    e = -F / w['k']
                self.e[key] = e
            out[('tyre',) + key] = dict(point=point, force=-F, tangential=-F, moment=np.zeros(3), on=None, by=w['body'],
                                        deflection=float(np.linalg.norm(self.e[key])), slip=s, normal=normal, load=N)
        return out

    def stability_ratio(self, q, dt, kin=None):
        """Stability ratio of the explicit tyre loads while every tyre grips (below 1: stable; linear analysis).

        With ``W = J M^-1 J^T`` the inverse inertia seen by the slips and ``g = 2 dt c + dt^2 k`` per tyre,
        the discrete system is stable while the largest eigenvalue of ``W g`` stays below 4.  ``M`` is the
        unconstrained team inertia.  Tyres that carry their wheels add the normal spring and damper, one row
        along the terrain normal with ``g = 2 dt cn + dt^2 kn``.
        """
        if not self.wheels:
            return 0.
        T, S = kin if kin is not None else self.tree.sweep(q)
        t = self.tree
        rows, gain = [], []
        for w in self.wheels:
            point, normal = self._contact(w, T)
            J = np.zeros((3, t.n))
            for k in t.support[w['index']]:
                J[:, t.coord[k]] = S[k][3:] + _cross(S[k][:3], point)
            a = np.cross(normal, [1., 0., 0.]) if abs(normal[0]) < .9 else np.cross(normal, [0., 1., 0.])
            a /= np.linalg.norm(a)
            rows += [a @ J, np.cross(normal, a) @ J]
            gain += [2. * dt * w['c'] + dt * dt * w['k']] * 2
            if self.carry:
                rows.append(normal @ J)
                gain.append(2. * dt * w['cn'] + dt * dt * w['kn'])
        J = np.array(rows)
        W = J @ np.linalg.solve(t.mass(q, kinematics=(T, S)), J.T)
        return float(np.max(np.abs(np.linalg.eigvals(W * np.asarray(gain)[None, :]))) / 4.)

    def standing_loads(self, q, rid, kin=None):
        """Wheel loads of a base that carries only its own weight on level ground: the equal-stiffness distribution
        (normal load affine in the wheel position), which equal tyre springs under a rigid chassis give."""
        robot = self.comp.robots[rid]
        tree = self.tree
        T = (tree.sweep(q) if kin is None else kin)[0]
        bodies = [i for i in tree.massive if tree.names[i].startswith(robot.prefix)]
        m = tree.mass_kg[bodies]
        com = (m[:, None] * np.array([T[i, :3, 3] + T[i, :3, :3] @ tree.com[i] for i in bodies])).sum(axis=0) / m.sum()
        Tb = T[tree.index[robot.prefix + robot.record['floating_root']]]
        c = Tb[:3, :3].T @ (com - Tb[:3, 3])
        weight = float(m.sum() * np.linalg.norm(tree.gravity))
        centres = np.array([w['centre_base'] for w in robot.wheels], float)
        Phi = np.c_[np.ones(len(centres)), centres[:, :2]]
        return Phi @ np.linalg.solve(Phi.T @ Phi, weight * np.array([1., c[0], c[1]]))

    def _clearances(self, q, idx, x, wheels):
        """Clearances of the tyre discs of ``wheels`` with the coordinates ``idx`` of ``q`` set to ``x``."""
        qq = q.copy()
        qq[idx] = x
        T = self.tree.sweep(qq)[0]
        return np.array([self.disc_contact(w, T)[0] for w in wheels])

    def stand(self, q, tolerance=STAND_TOLERANCE_M):
        """``q`` with every friction-driven base that stands on the terrain lowered onto its tyres.

        A base whose wheels all touch the terrain within ``tolerance`` (rigid-wheel coordinates, as the compiled
        model gives them) is moved along its height, pitch and roll until every tyre is pressed in by the depth that
        carries its share of the weight of the robot (``standing_loads``).  A base clear of the ground is left as it
        is.  Only for tyres that carry their wheels; the start of a run then has no drop onto the springs.
        """
        q = np.asarray(q, float).copy()
        if not self.carry:
            return q
        for rid, robot in self.comp.robots.items():
            wheels = [w for w in self.wheels if w['key'][0] == rid]
            if not wheels:
                continue
            idx = robot.base[[2, 4, 5]]
            x = q[idx].copy()
            if np.max(np.abs(self._clearances(q, idx, x, wheels))) > tolerance:
                continue
            target = -self.standing_loads(q, rid) / np.array([w['kn'] for w in wheels])
            for _ in range(8):
                h = self._clearances(q, idx, x, wheels)
                J = np.zeros((len(wheels), 3))
                for k in range(3):
                    d = np.zeros(3)
                    d[k] = 1e-6
                    J[:, k] = (self._clearances(q, idx, x + d, wheels) - self._clearances(q, idx, x - d, wheels)) / 2e-6
                step = np.linalg.lstsq(J, target - h, rcond=None)[0]
                x = x + step
                if np.max(np.abs(step)) < 1e-12:
                    break
            q[idx] = x
        return q

    def summary(self, loads):
        """Normal and tangential force of every tyre of a load set, as ``contacts()['wheels']`` reports them."""
        return {key[1:]: [float(load['load']), float(np.linalg.norm(load.get('tangential', load['force'])))]
                for key, load in loads.items()}


def check_tyre_stability(tyres, q, dt, limit=STABILITY_LIMIT):
    """Stability ratio of the explicit tyre loads; raises when the declared tyre is too stiff for the timestep."""
    ratio = tyres.stability_ratio(q, dt)
    if ratio > limit:
        raise RuntimeError(f'the declared tyre stiffness and damping need a smaller timestep: stability ratio {ratio:.2f} '
                           f'at dt = {dt:g} s exceeds {limit:g}')
    return ratio
