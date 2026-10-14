"""Tyres of conventional wheels as explicit loads (NumPy only).

A base on conventional wheels turns by sliding them (``teamc/skid.py``).  The
friction of a sliding tyre must then be Coulomb's in every direction of the
ground plane at once.  Rigid-contact friction that is solved as two independent
rows per contact point (PhysX) does not give that: a four-wheel base whose load
is shifted towards one axle turns at a fraction of the rate Coulomb friction
gives, and the fraction changes with the rate (``tests/test_skid_physx.py``).
Vehicle simulation therefore computes tyre forces from the slip and applies them
as loads.  This module is that tyre model for the plants: the Isaac Sim plant
always uses it for conventional wheels (where PhysX has a collider under them
it is frictionless), and the MuJoCo plant uses it by default (``tyre='model'``);
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

The normal load.  The friction is proportional to the load of every single
wheel, and a rigid base on four rigid contacts does not have one: rigid-body
mechanics fixes the total load and its centre and leaves open how much one
wheel diagonal carries at the cost of the other.  A contact solver fills that
in by its own rules.  PhysX 5.3.1 does it smoothly.  In Isaac Sim 6.1.0.0
(PhysX 5.9.1) the share moved between successive 20 ms means while a base stood
on the ground plane, by 55 N rms and by up to 330 N for wheels that carry 286 N
on average.  The friction of a wheel is taken at its reported load, and the
wheel torques of that mission peaked at 0.91 of their bound.  On PhysX 5.3.1
the same mission needs 0.77 on rigid contacts, 0.88 when the friction is taken
at the wheel loads recorded in Isaac Sim, and 0.63 with tyres that carry their
wheels (``reference_results/study_isaac_rigid_tyre_contact``).  A tyre is not
rigid.  With ``springs=True`` every tyre carries its wheel itself,
through the declared normal stiffness and damping of its contact
(``normal_loads``): the load is a function of the state like the friction, at
the same instant, and the plant has no contact under these wheels.  That is how
the Pinocchio plant steps such a base (``backends/pin_plant.py`` says why;
``reference_results/study_husky_rigid_ground`` shows what rigid ground does
there), and it is the default of the Isaac Sim plant.  With ``springs=False`` the plant
supplies the normal loads from its own contacts: MuJoCo's soft contact, or a
PhysX contact whose loads are those of the step before.

Not modelled: a contact patch and its aligning moment, rolling resistance, a
dependence of the friction coefficient on load or slip speed.
"""
from __future__ import annotations

import numpy as np

from teamc.terrain import Terrain

STABILITY_LIMIT = .6           # largest accepted stability ratio of the explicit tyre loads (1 is the linear limit)


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


CONTACT_BAND = 1.0e-3          # m; a base whose tyres are within this of touching is one that stands on the ground


class TyreSet:
    """The tyres of the friction-driven bases of a compiled team, evaluated in compiled coordinates.

    ``springs``: the tyres carry their wheels through their declared normal stiffness and damping
    (:meth:`normal_loads`); otherwise the plant gives the normal loads.
    """

    def __init__(self, comp, terrain=None, springs=False):
        self.comp = comp
        self.tree = comp.tree
        self.terrain = Terrain(comp.scene) if terrain is None else terrain
        self.springs = bool(springs)
        self.wheels = []
        for rid, r in comp.robots.items():
            if r.drive != 'friction':
                continue
            for w in r.wheels:
                ty = w.get('tyre', {})
                self.wheels.append(dict(key=(rid, w['id']), body=r.prefix + w['body'], index=self.tree.index[r.prefix + w['body']],
                                        coordinate=int(w['index']), radius=float(w['radius_m']), mu=float(w.get('friction', .8)),
                                        k=float(ty.get('stiffness_N_m', 1.5e5)), c=float(ty.get('damping_N_s_m', 1.5e3)),
                                        relax=float(ty.get('relaxation_length_m', 0.)),
                                        kn=float(ty.get('contact_stiffness_N_m', 2.0e5)), cn=float(ty.get('contact_damping_N_s_m', 4.0e3))))
        self.e = {w['key']: np.zeros(3) for w in self.wheels}
        self.sliding = {w['key']: False for w in self.wheels}

    def __bool__(self):
        return bool(self.wheels)

    def reset(self):
        for key in self.e:
            self.e[key] = np.zeros(3)
            self.sliding[key] = False

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
        if self.springs:                                      # the plane the sphere touches (continuous over an edge)
            normal = self.terrain.sphere_contact(centre, w['radius'])[1]
        else:
            normal = self.terrain.plane_at(float(centre[0]), float(centre[1]))[0]['n']
        return centre - w['radius'] * normal, normal

    def _twist(self, w, S, v):
        V = np.zeros(6)
        for k in self.tree.support[w['index']]:
            V += S[k] * v[self.tree.coord[k]]
        return V

    def normal_loads(self, q, v, kin=None):
        """Normal load of every tyre from its own spring at a state: ``{(robot, wheel): N}``.

        ``N = kn d - cn u`` with ``d`` the depth by which the tyre (a sphere about the wheel centre) has sunk into
        the terrain plane it touches and ``u`` the rate of the wheel centre along the normal of that plane.  A tyre
        does not pull: no load while it is clear of the terrain, and never a negative one.
        """
        T, S = kin if kin is not None else self.tree.sweep(q)
        out = {}
        for w in self.wheels:
            centre = T[w['index'], :3, 3]
            clearance, normal, _ = self.terrain.sphere_contact(centre, w['radius'])
            N = 0.
            if clearance < 0.:
                V = self._twist(w, S, v)
                rate = float((V[3:] + _cross(V[:3], centre)) @ normal)
                N = max(-w['kn'] * clearance - w['cn'] * rate, 0.)
            out[w['key']] = N
        return out

    def stand(self, q, iterations=30):
        """``q`` with every base that stands on the ground lowered onto its tyres.

        The compiled model and the start configurations of a mission have the wheel centres one radius above the
        terrain.  On its springs a base rests lower: this returns the configuration in which height, pitch and roll
        of every such base are those at which its tyres carry it at rest, all other coordinates as given.  A base
        stands on the ground if one of its tyres is in the terrain or within ``CONTACT_BAND`` of it; a base in the
        air is left where it is.  A tyre does not pull, so on ground that is not one plane a base may come to rest
        on three tyres.  The friction that holds a base on a slope is left out: there its pitch settles by a
        fraction of a milliradian in the first steps of a plant.
        """
        q = np.array(q, float)
        t = self.tree
        zero = np.zeros(t.n)
        for _ in range(iterations):
            T, S = t.sweep(q)
            weight = t.rnea(q, zero, zero, kinematics=(T, S))       # what holds every coordinate at rest
            moved = 0.
            for rid, r in self.comp.robots.items():
                mine = [w for w in self.wheels if w['key'][0] == rid]
                if not mine:
                    continue
                b = np.asarray(r.base)[[2, 4, 5]]
                Q, K, K_all, nearest = np.zeros(3), np.zeros((3, 3)), np.zeros((3, 3)), np.inf
                for w in mine:
                    centre = T[w['index'], :3, 3]
                    clearance, normal, _ = self.terrain.sphere_contact(centre, w['radius'])
                    nearest = min(nearest, clearance)
                    J = np.zeros(t.n)
                    for k in t.support[w['index']]:
                        J[t.coord[k]] = normal @ (S[k][3:] + _cross(S[k][:3], centre))
                    stiffness = w['kn'] * np.outer(J[b], J[b])
                    K_all += stiffness
                    if clearance < -1e-9:                           # this tyre carries (not one that just touches)
                        Q += J[b] * w['kn'] * -clearance
                        K += stiffness
                if nearest > CONTACT_BAND:
                    continue
                # Newton step on the tyres that carry; while fewer than three do (the first step from rigid wheels that
                # just touch, or a base rocking onto its third tyre), the stiffness of all of them
                if np.linalg.eigvalsh(K)[0] < 1e-6 * np.linalg.eigvalsh(K_all)[-1]:
                    K = K_all
                step = np.linalg.solve(K, Q - weight[b])
                q[b] += step
                moved = max(moved, float(np.abs(step).max()))
            if moved < 1e-12:
                break
        return q

    def _slip(self, w, S, v, point, normal):
        V = self._twist(w, S, v)
        s = V[3:] + _cross(V[:3], point)                      # velocity of the wheel's material point at the ground
        return s - (s @ normal) * normal

    def loads(self, q, v, normal_loads, dt, kin=None):
        """Tyre loads at a state for the given normal loads ``{(robot, wheel): N}``; advances the deflections by ``dt``.

        ``normal_loads`` None: those of the tyres' own springs at this state (:meth:`normal_loads`).

        Returns ``{key: dict(point, force, moment, on, by)}`` in the form of ``LatchSet.loads``: ``by`` is the wheel
        body, which receives ``-force`` (``force`` is what the tyre applies to the ground in its plane: the
        friction); ``on`` is ``None``.  ``load`` and ``normal`` are the normal load and the terrain normal;
        :meth:`carried` gives the loads with the normal force included, for a plant that leaves it to the tyres.
        """
        T, S = kin if kin is not None else self.tree.sweep(q)
        if normal_loads is None:
            normal_loads = self.normal_loads(q, v, (T, S))
        state = []
        for w in self.wheels:
            point, normal = self._contact(w, T)
            state.append((point, normal, self._slip(w, S, v, point, normal), float(v[w['coordinate']])))
        return self.advance(state, normal_loads, dt)

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
            out[('tyre',) + key] = dict(point=point, force=-F, moment=np.zeros(3), on=None, by=w['body'],
                                        deflection=float(np.linalg.norm(self.e[key])), slip=s, normal=normal, load=N)
        return out

    @staticmethod
    def carried(loads):
        """The loads of :meth:`loads` with the normal force of every tyre added to its friction: what a plant applies
        when the tyres carry their wheels (the wheel receives ``-force``, friction and ``load`` along ``normal``)."""
        return {key: dict(load, force=load['force'] - load['load'] * load['normal']) for key, load in loads.items()}

    def stability_ratio(self, q, dt, kin=None):
        """Stability ratio of the explicit tyre loads while every tyre grips (below 1: stable; linear analysis).

        With ``W = J M^-1 J^T`` the inverse inertia seen by the slips and ``g = 2 dt c + dt^2 k`` per tyre,
        the discrete system is stable while the largest eigenvalue of ``W g`` stays below 4.  ``M`` is the
        unconstrained team inertia.  Tyres that carry their wheels (``springs``) add a row along the terrain
        normal each, with their normal stiffness and damping.
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
            if self.springs:
                rows.append(normal @ J)
                gain.append(2. * dt * w['cn'] + dt * dt * w['kn'])
        J = np.array(rows)
        W = J @ np.linalg.solve(t.mass(q, kinematics=(T, S)), J.T)
        return float(np.max(np.abs(np.linalg.eigvals(W * np.asarray(gain)[None, :]))) / 4.)

    def summary(self, loads):
        """Normal and tangential force of every tyre of a load set, as ``contacts()['wheels']`` reports them."""
        return {key[1:]: [float(load['load']), float(np.linalg.norm(load['force']))] for key, load in loads.items()}


def check_tyre_stability(tyres, q, dt, limit=STABILITY_LIMIT):
    """Stability ratio of the explicit tyre loads; raises when the declared tyre is too stiff for the timestep."""
    ratio = tyres.stability_ratio(q, dt)
    if ratio > limit:
        raise RuntimeError(f'the declared tyre stiffness and damping need a smaller timestep: stability ratio {ratio:.2f} '
                           f'at dt = {dt:g} s exceeds {limit:g}')
    return ratio
