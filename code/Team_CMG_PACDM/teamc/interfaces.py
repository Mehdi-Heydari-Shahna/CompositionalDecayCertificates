"""Compiled mechanical interfaces of a team in one mode.

From the compiled team graph and the active connections of a mode this module
generates, at a configuration:

* the stacked constraint Jacobian ``C`` (connection rows, ground-coupling rows,
  wheel-rolling rows) with its port table;
* the rank-checked partition and tangent lift ``v = E u``;
* the constraint-curvature term ``c = E_dot u``;
* the actuator-port map ``L = B^T E`` and its dual ``tau_r = L^T f``;
* reduced dynamics ``M_r u_dot + h_r = L^T f + E^T tau_ext``;
* the constraint reactions that the reduction eliminates, recovered per port by a
  transpose solve with the dependent block already used for the lift, together
  with a rank test saying which reactions rigid-body mechanics determines;
* inverse dynamics for prescribed grasp wrenches (load sharing is an input).

Ground coupling and wheel rolling are ideal: no friction limit, unilateral
condition or slip is implied here.  Those are checked on the recovered
reactions in ``stability.py``.

Two ground models, chosen by the compiler from the mechanism of a base:

* ``fit``: ``[z, pitch, roll] = support(x, y, yaw)`` of a plane fitted to the
  ground under the wheels (three rows; their multipliers are the generalized
  forces on those three coordinates);
* ``contact``: one row per wheel, the height of the wheel above the terrain
  plane it touches (their multipliers are the wheel normal loads).  The passive
  joints of the base are dependent coordinates together with z, pitch and roll.

Two ways wheels drive a base, also chosen by the compiler:

* ``rolling`` (mecanum wheels): the wheel rows are ideal constraints; wheel
  rates are dependent coordinates;
* ``friction`` (conventional wheels on fixed axles, skid steering): the wheel
  rows are contact rows.  They give the slip of every wheel and carry its
  tangential ground force, but they do not constrain the motion: the wheel
  rates stay independent, the friction law of ``teamc/skid.py`` supplies the
  forces, and the quasi-static skid model supplies the motion of the base for
  given wheel speeds (``skid``, ``base_motion``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr

from . import skid as _skid
from .se3 import adj_fast as adj, chart_pose, exp, inv_fast as inv, inv_left, log, rot
from .terrain import Terrain

RANK_TOL = 1e-10
RCOND_MIN = 1e-10


@dataclass(frozen=True)
class Mode:
    """Active physical connections.  ``grasps`` and ``rest`` are connection ids."""
    grasps: tuple = ()
    rest: str | None = None
    ground: bool = True
    rolling: bool = True
    name: str = ''

    def label(self):
        return self.name or ('+'.join(self.grasps) or 'free') + (f'+{self.rest}' if self.rest else '')


@dataclass
class Constraints:
    J: np.ndarray                 # m x n       rows on tree coordinates
    Jc: np.ndarray                # m x nc      rows on connection chart coordinates
    residual: np.ndarray          # m           position-level residual (zero for velocity-level rows)
    blocks: list                  # dict(id, kind, rows (slice), holonomic)
    chart: np.ndarray             # nc          chart coordinates lifted from the body poses
    info: dict = field(default_factory=dict)

    def rows(self, port):
        return next(b['rows'] for b in self.blocks if b['id'] == port)


@dataclass
class Lift:
    E: np.ndarray                 # n x d       tree rows of the tangent lift
    independent: np.ndarray       # d           tree coordinate indices
    dependent: np.ndarray         # tree coordinate indices (chart coordinates follow them in Cp)
    constraints: Constraints
    Cp: np.ndarray                # selected rows x dependent block
    rows: np.ndarray              # selected constraint rows
    lu: tuple
    rank: int
    rcond: float
    redundant_rows: int


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def _rank(A):
    s = np.linalg.svd(A, compute_uv=False)
    return int(np.sum(s > RANK_TOL * s[0])) if s.size and s[0] > 0 else 0


class TeamModel:
    def __init__(self, comp, terrain=None):
        self.comp = comp
        self.tree = comp.tree
        self.n = self.tree.n
        self.ids = comp.cmg['coordinate_ids']
        self.terrain = Terrain(comp.scene) if terrain is None else terrain
        self.ports = comp.cmg['ports']
        self.nf = len(self.ports)
        self.B = np.zeros((self.n, self.nf))
        for k, p in enumerate(self.ports):
            self.B[p['index'], k] = p['gear']
        self.effort_bounds = np.array([p['bounds'] for p in self.ports], float)
        self.damping = np.array([comp.cmg['damping'].get(k, 0.) for k in self.ids])
        self.payload = np.asarray(comp.payload['base'], int)
        self.grasp_ids = [k for k, c in comp.connections.items() if c['kind'] == 'grasp']
        self.rest_ids = [k for k, c in comp.connections.items() if c['kind'] == 'rest']
        self._paths = {}
        for cid, c in comp.connections.items():
            t = self.tree
            self._paths[cid] = (t.index[c['body1']], t.index[c['body2']])
        self.wheel_centres = {rid: np.array([w['centre_base'] for w in r.wheels]) for rid, r in comp.robots.items()}
        self.wheel_radius = {rid: (r.wheels[0]['radius_m'] if r.wheels else 0.) for rid, r in comp.robots.items()}
        self.base_body = {rid: self.tree.index[r.prefix + r.record['floating_root']] for rid, r in comp.robots.items()}
        self._ground_memo = {}
        # coordinates the ground coupling determines: z, pitch, roll and the passive joints of the base
        self.passive = {rid: np.array([r.joint[j] for j in r.passive], int) for rid, r in comp.robots.items()}
        self.ground_dep = {rid: np.r_[r.base[[2, 4, 5]], self.passive[rid]].astype(int) for rid, r in comp.robots.items()}
        self.carrier = {rid: [self.tree.index[w.get('carrier', r.prefix + r.record['floating_root'])] for w in r.wheels]
                        for rid, r in comp.robots.items()}
        # height of the base origin above flat ground
        self.base_height = {rid: (self.wheel_radius[rid] - float(self.wheel_centres[rid][:, 2].mean()) if r.wheels else 0.)
                            for rid, r in comp.robots.items()}
        # friction-driven bases (conventional wheels): data of the friction law
        self.friction = {rid: r.drive == 'friction' for rid, r in comp.robots.items()}
        self._tyre = {}
        for rid, r in comp.robots.items():
            if self.friction[rid]:
                self._tyre[rid] = dict(
                    mu=np.array([w.get('friction', .8) for w in r.wheels], float),
                    radius=np.array([w['radius_m'] for w in r.wheels], float),
                    slip_speed=float(min(w.get('tyre', {}).get('slip_speed_m_s', _skid.SLIP_SPEED) for w in r.wheels)),
                    index=np.array([w['index'] for w in r.wheels], int))
        self._skid_memo = {}
        self._suspension = {}
        for rid, r in comp.robots.items():
            if r.ground != 'contact':
                continue
            joints = {j['id']: j for j in comp.cmg['joints']}
            chains = []
            for w in r.wheels:                      # joints between the floating root and the body that carries the wheel
                chain, body = [], w['carrier']
                while body != r.prefix + r.record['floating_root']:
                    j = next(j for j in comp.cmg['joints'] if j['follower_body'] == body)
                    k = self.ids.index(j['id']) if j['type'] != 'fixed' else -1
                    chain.append((np.asarray(j['T_BJ'], float), j['type'], np.asarray(j.get('axis', [0., 0., 1.]), float),
                                  inv(np.asarray(j['T_FJ'], float)), int(np.flatnonzero(self.passive[rid] == k)[0]) if k >= 0 else -1))
                    body = j['base_body']
                chains.append(chain[::-1])
            self._suspension[rid] = chains
        self._sensor = {}
        for rid, r in comp.robots.items():
            port = r.record['ports'].get('sensor')
            if port:
                body, T_s = r.frames[port['frame']]
                tool_body, T_tool = r.frames[r.record['ports']['tool']['frame']]
                if tool_body != body:
                    raise ValueError(f'{rid}: the tool frame must be on the tool-side body of the sensor joint')
                self._sensor[rid] = dict(body=self.tree.index[body], T_sensor=np.asarray(T_s, float), T_tool=np.asarray(T_tool, float))

    # ------------------------------------------------------------------ modes
    def mode(self, grasps=None, rest=None, ground=True, rolling=True, name=''):
        grasps = tuple(self.grasp_ids if grasps is None else grasps)
        for g in grasps:
            if g not in self.grasp_ids:
                raise KeyError(g)
        if rest is not None and rest not in self.rest_ids:
            raise KeyError(rest)
        return Mode(grasps=grasps, rest=rest, ground=ground, rolling=rolling, name=name)

    def sensor_to_grasp(self, rid, w_sensor, q, kin, v=None, a=None):
        """Wrench a robot applies to the payload through its grasp (tool axes) from its wrist sensor.

        The sensor reads what its robot side applies to the tool-side body.  Removing
        what that body needs for itself (its weight; with ``v`` and ``a`` also its inertia
        force) leaves what is passed on to the payload.
        """
        s = self._sensor[rid]
        T = kin[0]
        Ts = T[s['body']] @ s['T_sensor']
        if a is None:
            v = a = np.zeros(self.n)
        _, F, _ = self.tree.rnea(q, v, a, return_forces=True, kinematics=kin)
        own = adj(Ts).T @ self._subtree_own(s['body'], F)          # joint wrench the tool-side body needs for itself
        w = np.asarray(w_sensor, float) - own
        return adj(inv(s['T_sensor']) @ s['T_tool']).T @ w

    def _subtree_own(self, body, F):
        """World spatial wrench across the parent joint of ``body`` when nothing external acts on its subtree."""
        return F[body]

    def register_grasp(self, cid, T2):
        """Model reconfiguration: replace the payload-side frame of a grasp by its as-latched value."""
        c = self.comp.connections[cid]
        if c['kind'] != 'grasp':
            raise KeyError(cid)
        c['T2'] = np.asarray(T2, float).copy()

    # ------------------------------------------------------------------ rows
    def _twist_map(self, body_index, S):
        X = np.zeros((6, self.n))
        for k in self.tree.support[body_index]:
            X[:, self.tree.coord[k]] = S[k]
        return X

    def _chart(self, Tm, Tp, free):
        """Free-twist coordinates of a closed connection from its two frames (Gauss-Newton on the kernel residual)."""
        c = np.zeros(len(free))
        for _ in range(20):
            T, cols = Tm.copy(), []
            for xi, ck in zip(free, c):
                cols.append(adj(T) @ xi)
                T = T @ exp(xi * ck)
            rev = inv(T)
            r = log(rev @ Tp)
            A = -(inv_left(r) @ adj(rev) @ np.column_stack(cols))
            step = np.linalg.lstsq(A, -r, rcond=None)[0]
            c += step
            if np.max(np.abs(step)) < 1e-13:
                break
        return c

    def connection_rows(self, cid, kin):
        c = self.comp.connections[cid]
        T, S = kin
        i1, i2 = self._paths[cid]
        Tm, Tp = T[i1] @ c['T1'], T[i2] @ c['T2']
        X1, X2 = self._twist_map(i1, S), self._twist_map(i2, S)
        free = [np.asarray(f, float) for f in c['free']]
        chart = self._chart(Tm, Tp, free) if free else np.zeros(0)
        cols = []
        for xi, ck in zip(free, chart):
            cols.append(adj(Tm) @ xi)
            Tm = Tm @ exp(xi * ck)
        rev = inv(Tm)
        r = log(rev @ Tp)
        P = inv_left(r) @ adj(rev)
        Jc = -(P @ np.column_stack(cols)) if free else np.zeros((6, 0))
        return r, P @ (X2 - X1), Jc, chart

    def ground_state(self, q, rid):
        """Ideal ground-following values, their Jacobian and the support-plane normal of one base (``fit`` model)."""
        base = self.comp.robots[rid].base
        xy_yaw = np.asarray(q, float)[base[[0, 1, 3]]]
        key = (rid, xy_yaw.tobytes())
        hit = self._ground_memo.get(rid)
        if hit is not None and hit[0] == key:
            return hit[1]
        value, normal, misfit = self.terrain.support(xy_yaw, self.wheel_centres[rid], self.wheel_radius[rid])
        G = self.terrain.support_jacobian(xy_yaw, self.wheel_centres[rid], self.wheel_radius[rid])
        self._ground_memo[rid] = (key, (value, G, normal, misfit))
        return value, G, normal, misfit

    def wheel_contacts(self, q, rid, kin):
        """Ideal terrain contact of every wheel of a ``contact`` base: heights, normals, points, wheel axes."""
        robot = self.comp.robots[rid]
        T = kin[0]
        out = []
        for w, ic in zip(robot.wheels, self.carrier[rid]):
            Tc = T[ic]
            centre = Tc[:3, 3] + Tc[:3, :3] @ w['centre_carrier']
            axis = Tc[:3, :3] @ w['axis_carrier']
            height, normal, point = self.terrain.wheel_contact(centre, axis, w['radius_m'])
            out.append(dict(height=height, normal=normal, point=point, axis=axis, centre=centre))
        return out

    def ground_rows(self, q, rid, kin=None):
        """Rows of the ground coupling of one base: residual, Jacobian and the ground geometry used.

        ``fit``: three rows ``[z, pitch, roll] - support(x, y, yaw)``; the geometry is the normal of the
        fitted plane.  ``contact``: one row per wheel, the velocity along the terrain normal of the
        point of the wheel carrier that is at the contact point; the geometry is the list of contacts.
        """
        robot = self.comp.robots[rid]
        base = robot.base
        if robot.ground == 'contact':
            kin = self.tree.sweep(q) if kin is None else kin
            contacts = self.wheel_contacts(q, rid, kin)
            J = np.zeros((len(contacts), self.n))
            for k, (c, ic) in enumerate(zip(contacts, self.carrier[rid])):
                X = self._twist_map(ic, kin[1])
                J[k] = _cross(c['point'], c['normal']) @ X[:3] + c['normal'] @ X[3:]
            return np.array([c['height'] for c in contacts]), J, contacts
        value, G, normal, _ = self.ground_state(q, rid)
        J = np.zeros((3, self.n))
        J[np.arange(3), base[[2, 4, 5]]] = 1.
        J[:, base[[0, 1, 3]]] -= G
        return np.asarray(q, float)[base[[2, 4, 5]]] - value, J, normal

    def wheel_rows(self, q, rid, kin, normal=None):
        """One row per traction direction: velocity of the wheel's ground point along it is zero.

        ``normal`` is the ground geometry returned by :meth:`ground_rows` (the normal of the fitted plane,
        or the list of wheel contacts).  Without it the ground is taken parallel to the base (``fit``) or
        to the body that carries the wheel (``contact``).
        """
        robot = self.comp.robots[rid]
        T, S = kin
        rows, labels = [], []
        if robot.ground == 'contact':
            contacts = normal
            for k, (w, ic) in enumerate(zip(robot.wheels, self.carrier[rid])):
                Rc = T[ic][:3, :3]
                if contacts is None:
                    n = Rc[:, 2]
                    axis = Rc @ w['axis_carrier']
                    centre = T[ic][:3, 3] + Rc @ w['centre_carrier']
                else:
                    n, axis, centre = contacts[k]['normal'], contacts[k]['axis'], contacts[k]['centre']
                na = float(n @ axis)
                u = (n - na * axis) / np.sqrt(max(1. - na * na, 1e-300))      # from the ground point towards the centre
                c = centre - w['radius_m'] * u
                drive = _cross(axis, u)                                       # rolling direction of the wheel bottom per unit spin
                Xc = self._twist_map(ic, S)
                directions = []
                if w['type'] == 'mecanum':
                    directions.append(('traction', Rc @ w['traction_direction_carrier']))
                else:                                                         # conventional wheel
                    directions.append(('rolling', drive / np.linalg.norm(drive)))
                    if w.get('lateral', 'constraint') == 'constraint':
                        directions.append(('lateral', axis))
                for label, g in directions:
                    g = g - (g @ n) * n
                    g = g / np.sqrt(g @ g)
                    row = _cross(c, g) @ Xc[:3] + g @ Xc[3:]
                    row[w['index']] += -w['radius_m'] * (g @ drive)
                    rows.append(row)
                    labels.append(f"{w['id']}:{label}")
            return np.array(rows).reshape(-1, self.n), labels
        ib = self.base_body[rid]
        Tb = T[ib]
        R, origin = Tb[:3, :3], Tb[:3, 3]
        Xb = self._twist_map(ib, S)
        if normal is None:
            normal = R[:, 2]
        for w in robot.wheels:
            centre = origin + R @ w['centre_base']
            axis = R @ w['axis_base']
            c = centre - w['radius_m'] * normal
            drive = _cross(axis, normal)                        # rolling direction of the wheel bottom per unit spin
            directions = []
            if w['type'] == 'mecanum':
                directions.append(('traction', R @ w['traction_direction_base']))
            else:                                               # conventional wheel
                directions.append(('rolling', drive / np.linalg.norm(drive)))
                if w.get('lateral', 'constraint') == 'constraint':
                    directions.append(('lateral', axis))
            for label, g in directions:
                g = g - (g @ normal) * normal
                g = g / np.sqrt(g @ g)
                row = _cross(c, g) @ Xb[:3] + g @ Xb[3:]
                row[w['index']] += -w['radius_m'] * (g @ drive)
                rows.append(row)
                labels.append(f"{w['id']}:{label}")
        return np.array(rows).reshape(-1, self.n), labels

    # ------------------------------------------------------------------ a base on the terrain
    def _wheel_heights(self, rid, planar, x):
        """Wheel clearances of a ``contact`` base at a planar pose with ``x = [z, pitch, roll, passive...]``."""
        robot = self.comp.robots[rid]
        Tb = chart_pose([planar[0], planar[1], x[0], planar[2], x[1], x[2]])
        out = np.zeros(len(robot.wheels))
        for k, (w, chain) in enumerate(zip(robot.wheels, self._suspension[rid])):
            Tc = Tb
            for T_BJ, kind, axis, leave, slot in chain:
                Tc = Tc @ T_BJ
                if kind == 'revolute':
                    R = np.eye(4)
                    R[:3, :3] = rot(axis, x[3 + slot])
                    Tc = Tc @ R
                elif kind == 'prismatic':
                    P = np.eye(4)
                    P[:3, 3] = axis * x[3 + slot]
                    Tc = Tc @ P
                Tc = Tc @ leave
            centre = Tc[:3, 3] + Tc[:3, :3] @ w['centre_carrier']
            out[k] = self.terrain.wheel_contact(centre, Tc[:3, :3] @ w['axis_carrier'], w['radius_m'])[0]
        return out

    def ground_pose(self, rid, planar):
        """Coordinates of a base that stands on the terrain at the planar pose ``[x, y, yaw]``.

        Returns ``(indices, values)`` of the coordinates the ground coupling determines: z, pitch and
        roll of the base chart, followed by the passive joints of a ``contact`` base.
        """
        robot = self.comp.robots[rid]
        planar = np.asarray(planar, float)
        value, _, _ = self.terrain.support(planar, self.wheel_centres[rid], self.wheel_radius[rid])
        if robot.ground != 'contact':
            return self.ground_dep[rid], value
        x = np.r_[value, np.zeros(len(self.passive[rid]))]          # exact on one plane; Newton across a crease
        for _ in range(40):
            h = self._wheel_heights(rid, planar, x)
            if np.max(np.abs(h)) < 1e-12:
                return self.ground_dep[rid], x
            J = np.zeros((len(h), len(x)))
            for k in range(len(x)):
                d = np.zeros(len(x))
                d[k] = 1e-7
                J[:, k] = (self._wheel_heights(rid, planar, x + d) - self._wheel_heights(rid, planar, x - d)) / 2e-7
            x = x - np.linalg.solve(J, h)
        if np.max(np.abs(self._wheel_heights(rid, planar, x))) > 1e-9:
            raise np.linalg.LinAlgError(f'{rid}: no pose with every wheel on the terrain at {planar}')
        return self.ground_dep[rid], x

    def base_motion(self, q, rid, kin, loads=None):
        """Rates that follow from the planar rates of a base under ground coupling and rolling.

        Returns ``(indices, S, W)``: ``S`` maps ``[x, y, yaw]`` rates to the rates of ``indices`` (the six
        chart coordinates and the passive joints), ``W`` maps them to the wheel rates, in wheel order.

        For a friction-driven base ``W`` comes from the quasi-static skid model at the wheel loads ``loads``
        (default: the base standing alone): the wheel rates at which the base moves forward and turns at the
        given rates.  Such a base cannot be given a sideways rate; the sideways motion that goes with a turn
        is ``skid(...)['centre']``.
        """
        robot = self.comp.robots[rid]
        base = robot.base
        wheels = np.array([w['index'] for w in robot.wheels], int)
        if self.friction[rid]:
            _, G, geometry, _ = self.ground_state(q, rid)
            S = np.zeros((6, 3))
            S[0, 0] = S[1, 1] = S[3, 2] = 1.
            S[[2, 4, 5]] = G
            rows, labels = self.wheel_rows(q, rid, kin, geometry)
            rolling = [k for k, label in enumerate(labels) if label.endswith(':rolling')]
            W = -(rows[rolling][:, base] @ S) / rows[rolling, wheels][:, None]     # rates of wheels that roll without slip
            m = self.skid(q, rid, kin, loads)
            centres = self.wheel_centres[rid]
            # a turn needs more than rolling: the wheels of the two sides differ by 2 / c[2] in surface speed
            W[:, 2] += (centres[:, 1] - m['centre'][1] - m['side'] / m['c'][2]) / self._tyre[rid]['radius']
            return base, S, W
        if robot.ground == 'contact':
            _, A, geometry = self.ground_rows(q, rid, kin)
            idx = np.r_[base, self.passive[rid]].astype(int)
            S = np.zeros((len(idx), 3))
            S[0, 0] = S[1, 1] = S[3, 2] = 1.
            dep = np.r_[[2, 4, 5], 6 + np.arange(len(self.passive[rid]))].astype(int)
            S[dep] = -np.linalg.solve(A[:, self.ground_dep[rid]], A[:, base[[0, 1, 3]]])
        else:
            _, G, geometry, _ = self.ground_state(q, rid)
            idx = base
            S = np.zeros((6, 3))
            S[0, 0] = S[1, 1] = S[3, 2] = 1.
            S[[2, 4, 5]] = G
        rows, _ = self.wheel_rows(q, rid, kin, geometry)
        W = -(rows[:, idx] @ S) / rows[np.arange(len(wheels)), wheels][:, None]
        return idx, S, W

    # ------------------------------------------------------------------ friction-driven bases
    def wheel_loads(self, q, rid, lam_ground, kin=None, normal=None):
        """Normal load of every wheel of a base from the multipliers of its ground rows."""
        from .stability import wheel_normal_loads
        return wheel_normal_loads(self, q, rid, lam_ground, kin, normal)

    def standing_loads(self, q, rid, kin=None):
        """Wheel loads of a ``fit`` base that carries only its own weight on level ground (equal-stiffness rule)."""
        robot = self.comp.robots[rid]
        T = (self.tree.sweep(q) if kin is None else kin)[0]
        tree = self.tree
        bodies = [i for i in tree.massive if tree.names[i].startswith(robot.prefix)]
        m = tree.mass_kg[bodies]
        com = (m[:, None] * np.array([T[i, :3, 3] + T[i, :3, :3] @ tree.com[i] for i in bodies])).sum(axis=0) / m.sum()
        Tb = T[self.base_body[rid]]
        c = Tb[:3, :3].T @ (com - Tb[:3, 3])
        weight = float(m.sum() * np.linalg.norm(tree.gravity))
        centres = self.wheel_centres[rid]
        Phi = np.c_[np.ones(len(centres)), centres[:, :2]]
        return Phi @ np.linalg.solve(Phi.T @ Phi, weight * np.array([1., c[0], c[1]]))

    def skid(self, q, rid, kin=None, loads=None):
        """Quasi-static skid model of a friction-driven base at given wheel loads (``teamc.skid.skid_map``).

        ``loads`` are the normal loads of the wheels (default: the base standing alone).  They are rounded
        to a quarter of a newton, so that a base whose loads do not change is not solved again.
        """
        t = self._tyre[rid]
        loads = self.standing_loads(q, rid, kin) if loads is None else np.asarray(loads, float)
        key = np.round(loads * 4.)
        memo = self._skid_memo.get(rid)
        if memo is not None and np.array_equal(memo[0], key):
            return memo[1]
        robot = self.comp.robots[rid]
        axes = np.array([w['axis_base'][:2] for w in robot.wheels], float)
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        rolling = np.c_[axes[:, 1], -axes[:, 0]]                      # axis x normal, in the plane of the base
        out = _skid.skid_map(self.wheel_centres[rid][:, :2], rolling, axes, t['mu'], key / 4., t['radius'],
                             start=None if memo is None else memo[1]['c'])
        out['loads'] = key / 4.
        self._skid_memo[rid] = (key, out)
        return out

    def planar_twist(self, q, v, rid, kin=None):
        """Twist ``[vx, vy, wz]`` of a base in its own axes (velocity of its origin, yaw rate about its z axis)."""
        kin = self.tree.sweep(q) if kin is None else kin
        ib = self.base_body[rid]
        R = kin[0][ib][:3, :3]
        V = self._twist_map(ib, kin[1]) @ np.asarray(v, float)        # world twist about the world origin
        origin = V[3:] + _cross(V[:3], kin[0][ib][:3, 3])
        return np.array([R[:, 0] @ origin, R[:, 1] @ origin, R[:, 2] @ V[:3]])

    def wheel_slip(self, q, v, rid, kin=None, geometry=None):
        """Slip of every wheel of a friction-driven base: ``(n, 2)``, along its rolling direction and along its axle.

        Returns ``(slip, rows, columns)``: ``rows`` are the wheel rows and ``columns`` the row numbers of the
        two components of every wheel.
        """
        kin = self.tree.sweep(q) if kin is None else kin
        if geometry is None:
            geometry = self.ground_state(q, rid)[2]
        rows, labels = self.wheel_rows(q, rid, kin, geometry)
        robot = self.comp.robots[rid]
        columns = np.array([[labels.index(f"{w['id']}:rolling"), labels.index(f"{w['id']}:lateral")] for w in robot.wheels])
        return (rows @ np.asarray(v, float))[columns], rows, columns

    def friction_loads(self, q, v, rid, loads, kin=None, geometry=None):
        """Friction of the wheels of a friction-driven base at a state, for given wheel loads.

        Returns ``(Q, lam)``: the generalized force of the friction forces and the forces themselves in
        the order of the wheel rows (their multipliers).
        """
        t = self._tyre[rid]
        slip, rows, columns = self.wheel_slip(q, v, rid, kin, geometry)
        F = _skid.friction_forces(slip, loads, t['mu'], t['slip_speed'])
        lam = np.zeros(len(rows))
        lam[columns] = F
        return rows.T @ lam, lam

    def skid_twist(self, q, v, rid, loads, kin=None):
        """Twist the skid model gives a base for the present rates of its wheels (base axes ``[vx, vy, wz]``)."""
        t = self._tyre[rid]
        m = self.skid(q, rid, kin, loads)
        w = t['mu'] * np.maximum(np.asarray(loads, float), 0.)
        return _skid.twist(m['A'], w, t['radius'] * np.asarray(v, float)[t['index']])[0]

    def _friction(self, q, v, mode, kin, info, ground, wheel_loads=None):
        """Friction loads of every friction-driven base at the rates ``v``.

        The friction of a wheel is proportional to its normal load.  The loads are those given in
        ``wheel_loads`` (per robot); for a robot not listed there they follow from its ground reaction,
        which the friction forces change in turn: they act below the base and shift load between its wheels.
        ``ground(Q)`` returns the ground reactions (per robot) for an added generalized force ``Q``; they are
        affine in it, so the loads that are consistent with their own friction solve a small linear system.
        A wheel whose load comes out negative has left the ground: it is taken out and the others are solved
        again.

        Returns the generalized force, the forces per wheel-row port and the wheel loads used.
        """
        from .stability import wheel_load_map
        Q, lam, used = np.zeros(self.n), {}, {}
        if not (mode.ground and mode.rolling):
            return Q, lam, used
        robots = [rid for rid in self.comp.robots if self.friction[rid]]
        unit, rows_of, maps = {}, {}, {}
        for rid in robots:
            geometry = info.get(f'{rid}:normal')
            slip, rows, columns = self.wheel_slip(q, v, rid, kin, geometry)
            t = self._tyre[rid]
            unit[rid] = -(t['mu'] / np.sqrt((slip * slip).sum(axis=1) + t['slip_speed'] ** 2))[:, None] * slip    # per unit load
            rows_of[rid] = (rows, columns)
            if wheel_loads is not None and wheel_loads.get(rid) is not None:
                used[rid] = np.asarray(wheel_loads[rid], float)
            else:
                maps[rid] = wheel_load_map(self, q, rid, kin, geometry)

        def force(rid, loads):
            rows, columns = rows_of[rid]
            x = np.zeros(len(rows))
            x[columns] = loads[:, None] * unit[rid]
            return rows.T @ x, x

        for rid in used:
            Q += force(rid, used[rid])[0]
        if maps:
            order = [(rid, k) for rid in maps for k in range(len(unit[rid]))]
            lam0 = ground(Q)
            stack = lambda g: np.concatenate([maps[rid] @ g[rid] for rid in maps])
            N0 = stack(lam0)
            G = np.zeros((len(order), len(order)))
            for j, (rid, k) in enumerate(order):
                one = np.zeros(len(unit[rid]))
                one[k] = 1.
                d = ground(Q + force(rid, one)[0])
                G[:, j] = stack({r: d[r] - lam0[r] for r in maps})
            active = np.ones(len(order), bool)
            for _ in range(len(order)):
                N = np.zeros(len(order))
                N[active] = np.linalg.solve((np.eye(len(order)) - G)[np.ix_(active, active)], N0[active])
                if N.min() >= 0.:
                    break
                active &= N > 0.
            start = 0
            for rid in maps:
                used[rid] = N[start:start + len(unit[rid])]
                start += len(unit[rid])
                Q += force(rid, used[rid])[0]
        for rid in robots:
            lam[f'{rid}:wheels'] = force(rid, used[rid])[1]
        return Q, lam, used

    def constraints(self, q, mode, kin=None):
        q = np.asarray(q, float)
        kin = self.tree.sweep(q) if kin is None else kin
        J, Jc_blocks, res, blocks, charts, info = [], [], [], [], [], {}
        m = 0
        for cid in list(mode.grasps) + ([mode.rest] if mode.rest else []):
            r, Jt, Jc, chart = self.connection_rows(cid, kin)
            J.append(Jt)
            res.append(r)
            Jc_blocks.append((slice(m, m + 6), Jc))
            charts.append(chart)
            blocks.append(dict(id=cid, kind=self.comp.connections[cid]['kind'], rows=slice(m, m + 6), holonomic=True))
            m += 6
        for rid in self.comp.robots:
            normal = None
            if mode.ground:
                r, Jt, normal = self.ground_rows(q, rid, kin)
                J.append(Jt)
                res.append(r)
                blocks.append(dict(id=f'{rid}:ground', kind='ground', rows=slice(m, m + len(r)), holonomic=True, robot=rid))
                m += len(r)
                info[f'{rid}:normal'] = normal
            if mode.rolling and self.comp.robots[rid].wheels:
                Jt, labels = self.wheel_rows(q, rid, kin, normal)
                J.append(Jt)
                res.append(np.zeros(len(Jt)))
                blocks.append(dict(id=f'{rid}:wheels', kind='wheels', rows=slice(m, m + len(Jt)), holonomic=False,
                                   robot=rid, labels=labels))
                if self.friction[rid]:
                    blocks[-1]['slip'] = True                      # contact rows of a friction-driven base: not constraints
                m += len(Jt)
        nc = sum(len(c) for c in charts)
        Jc = np.zeros((m, nc))
        col = 0
        for rows, block in Jc_blocks:
            Jc[rows, col:col + block.shape[1]] = block
            col += block.shape[1]
        return Constraints(J=np.vstack(J) if J else np.zeros((0, self.n)), Jc=Jc,
                           residual=np.concatenate(res) if res else np.zeros(0), blocks=blocks,
                           chart=np.concatenate(charts) if charts else np.zeros(0), info=info)

    # ------------------------------------------------------------------ partition and lift
    def partition(self, mode):
        """Dependent tree coordinates implied by the mode (inputs stay independent)."""
        dep = []
        for cid in mode.grasps:
            robot = self.comp.robots[self.comp.connections[cid]['robot']]
            dep += [robot.joint[j] for j in robot.dependent]
        if mode.rest:
            dep += list(self.payload)
        for rid, robot in self.comp.robots.items():
            if mode.ground:
                dep += list(self.ground_dep[rid])
            if mode.rolling and not self.friction[rid]:
                dep += [w['index'] for w in robot.wheels]
        dep = np.array(sorted(set(int(i) for i in dep)), int)
        ind = np.setdiff1d(np.arange(self.n), dep)
        return ind, dep

    def lift(self, q, mode, kin=None):
        con = self.constraints(q, mode, kin)
        ind, dep = self.partition(mode)
        Cp = np.hstack([con.J[:, dep], con.Jc])
        Ca = con.J[:, ind]
        slip = [b['rows'] for b in con.blocks if b.get('slip')]
        if slip:
            # the wheel rows of a friction-driven base are not constraints: they take no part in the lift
            ideal = np.setdiff1d(np.arange(Cp.shape[0]), np.concatenate([np.arange(r.start, r.stop) for r in slip]))
            Cp_all, Cp = Cp, Cp[ideal]
        m, p = Cp.shape
        rk = _rank(Cp) if m else 0
        if rk != p:
            raise np.linalg.LinAlgError(f'{mode.label()}: dependent block has rank {rk} for {p} dependent coordinates')
        if m == p:
            rows = np.arange(m)
        else:   # redundant rows: keep a maximal independent subset, check the others afterwards
            rows = np.sort(qr(Cp.T, pivoting=True, mode='economic')[2][:p])
        if slip:
            rows, Cp, m = ideal[rows], Cp_all, len(ideal)
        A = Cp[rows]
        rcond = 1. / np.linalg.cond(A, 1) if p else 1.
        if rcond < RCOND_MIN:
            raise np.linalg.LinAlgError(f'{mode.label()}: dependent block is ill-conditioned (rcond {rcond:.2e})')
        lu = lu_factor(A) if p else None
        E = np.zeros((self.n, len(ind)))
        E[ind] = np.eye(len(ind))
        if p:
            X = -lu_solve(lu, Ca[rows])
            E[dep] = X[:len(dep)]
        return Lift(E=E, independent=ind, dependent=dep, constraints=con, Cp=A, rows=rows, lu=lu, rank=rk,
                    rcond=float(rcond), redundant_rows=int(m - p))

    def curvature(self, q, u, mode, lift=None, h0=1e-5):
        """``c = E_dot u`` by a centred directional difference of the stacked constraint Jacobian."""
        lift = self.lift(q, mode) if lift is None else lift
        v = lift.E @ np.asarray(u, float)
        c = np.zeros(self.n)
        if not len(lift.dependent) and not lift.constraints.Jc.shape[1]:
            return c, v
        h = h0 / max(1., float(np.max(np.abs(v))))
        cp = self.constraints(q + h * v, mode)
        cm = self.constraints(q - h * v, mode)
        # chart rates follow from the lift: they are the trailing dependent speeds
        vc = np.zeros(lift.constraints.Jc.shape[1])
        if vc.size:
            full = -lu_solve(lift.lu, (lift.constraints.J[:, lift.independent] @ np.asarray(u, float))[lift.rows])
            vc = full[len(lift.dependent):]
        gamma = ((cp.J - cm.J) / (2 * h)) @ v + ((cp.Jc - cm.Jc) / (2 * h)) @ vc
        sol = -lu_solve(lift.lu, gamma[lift.rows])
        c[lift.dependent] = sol[:len(lift.dependent)]
        return c, v

    # ------------------------------------------------------------------ dynamics
    def terms(self, q, v, kin=None):
        kin = self.tree.sweep(q) if kin is None else kin
        M = self.tree.mass(q, kinematics=kin)
        h = self.tree.bias(q, v, kinematics=kin) + self.damping * np.asarray(v, float)
        return M, h

    def reduced(self, q, u, mode):
        """Reduced inertia, bias and actuator-port map at ``(q, u)``."""
        lift = self.lift(q, mode)
        c, v = self.curvature(q, u, mode, lift)
        M, h = self.terms(q, v)
        Mr = lift.E.T @ M @ lift.E
        hr = lift.E.T @ (h + M @ c)
        return dict(Mr=Mr, hr=hr, L=self.B.T @ lift.E, E=lift.E, c=c, v=v, M=M, h=h, lift=lift)

    def reactions(self, lift, generalized_reaction):
        """Multipliers of the selected rows from the generalized reaction ``M a + h - tau``."""
        rhs = np.r_[generalized_reaction[lift.dependent], np.zeros(lift.constraints.Jc.shape[1])]
        lam = np.zeros(lift.constraints.J.shape[0])
        if lift.lu is not None:
            lam[lift.rows] = lu_solve(lift.lu, rhs, trans=1)
        return lam

    def split(self, lift, lam):
        """Reactions per port."""
        return {b['id']: lam[b['rows']].copy() for b in lift.constraints.blocks}

    def forward(self, q, u, f, mode, tau_ext=None, wheel_loads=None):
        """Reduced forward dynamics: independent accelerations, full accelerations and reactions.

        ``wheel_loads`` (friction-driven bases): the normal loads of the wheels the friction law is evaluated
        with, per robot.  Without them they are the loads that are consistent with the friction they give
        (``_friction``).  The loads used are returned.
        """
        r = self.reduced(q, u, mode)
        tau = self.B @ np.asarray(f, float) + (0. if tau_ext is None else np.asarray(tau_ext, float))
        used = {}
        if any(self.friction.values()):
            # friction of the wheels of friction-driven bases at this state; their forces are the multipliers of
            # the wheel rows
            con = r['lift'].constraints
            factor = lu_factor(r['Mr'])

            def ground(extra):
                udd = lu_solve(factor, r['E'].T @ (tau + extra - r['h'] - r['M'] @ r['c']))
                lam = self.reactions(r['lift'], r['M'] @ (r['E'] @ udd + r['c']) + r['h'] - tau - extra)
                return {rid: lam[con.rows(f'{rid}:ground')] for rid in self.comp.robots if self.friction[rid]}

            Q, wheel_forces, used = self._friction(q, r['v'], mode, self.tree.sweep(q), con.info, ground, wheel_loads)
            udd = np.linalg.solve(r['Mr'], r['E'].T @ (tau + Q - r['h'] - r['M'] @ r['c']))
            a = r['E'] @ udd + r['c']
            resid = r['M'] @ a + r['h'] - tau
            lam = self.reactions(r['lift'], resid - Q)
            for key, value in wheel_forces.items():
                lam[con.rows(key)] = value
        else:
            udd = np.linalg.solve(r['Mr'], r['E'].T @ (tau - r['h'] - r['M'] @ r['c']))
            a = r['E'] @ udd + r['c']
            resid = r['M'] @ a + r['h'] - tau
            lam = self.reactions(r['lift'], resid)
        balance = float(np.max(np.abs(r['lift'].constraints.J.T @ lam - resid))) if lam.size else float(np.max(np.abs(resid)))
        return dict(udd=udd, a=a, v=r['v'], lam=lam, ports=self.split(r['lift'], lam), lift=r['lift'],
                    Mr=r['Mr'], reaction_balance=balance, wheel_loads=used,
                    virtual_power_defect=abs(float(tau @ r['v'] - (r['E'].T @ tau) @ np.asarray(u, float))))

    def inverse(self, q, v, a, mode, grasp_wrenches=None, tau_ext=None, kin=None, constraints=None, effort=None,
                wheel_loads=None):
        """Actuator efforts and remaining reactions for a motion and prescribed grasp wrenches.

        ``grasp_wrenches`` maps active grasp ids to the wrench applied to the payload
        (tool axes, ``[moment; force]``).  Grasps not listed are treated as unknown
        reactions.  Among the solutions the one with the smallest wheel traction
        forces is returned (the wheel-internal effort mode is not excited).

        Friction-driven bases: the friction of the requested motion is a load, evaluated with the wheel
        loads in ``wheel_loads`` (per robot) or, without them, with the loads that are consistent with
        that friction (``_friction``).  The wheel rows carry what the law does not supply: the grip of
        wheels that do not slide.  The wheel loads used are returned.
        """
        kin = self.tree.sweep(q) if kin is None else kin
        con = self.constraints(q, mode, kin) if constraints is None else constraints
        rhs = (self.tree.rnea(q, v, a, kinematics=kin) if effort is None else np.asarray(effort, float)) \
            + self.damping * np.asarray(v, float)
        if tau_ext is not None:
            rhs = rhs - np.asarray(tau_ext, float)
        given = grasp_wrenches or {}
        free_blocks, lam = [], np.zeros(con.J.shape[0])
        for b in con.blocks:
            if b['id'] in given:
                lam[b['rows']] = given[b['id']]
            else:
                free_blocks.append(b)
        rhs = rhs - con.J.T @ lam
        cols = [self.B]
        weight = [np.full(self.nf, 1e4)]
        for b in free_blocks:
            cols.append(con.J[b['rows']].T)
            weight.append(np.full(b['rows'].stop - b['rows'].start, 1. if b['kind'] == 'wheels' else 1e4))
        A = np.hstack(cols)
        w = np.concatenate(weight)
        P = np.linalg.pinv(A * w, rcond=1e-11)
        given_lam = lam

        def solve(rhs):
            z = w * (P @ rhs)
            lam = given_lam.copy()
            k = self.nf
            for b in free_blocks:
                size = b['rows'].stop - b['rows'].start
                lam[b['rows']] = z[k:k + size]
                k += size
            return z, lam

        used = {}
        if any(self.friction.values()):
            # the wheels of a friction-driven base slide when it turns: the friction of the requested motion is
            # a load; the rows keep what the law does not supply
            def ground(extra):
                x = solve(rhs - extra)[1]
                return {rid: x[con.rows(f'{rid}:ground')] for rid in self.comp.robots if self.friction[rid]}

            Q, wheel_forces, used = self._friction(q, v, mode, kin, con.info, ground, wheel_loads)
            rhs = rhs - Q
            z, lam = solve(rhs)
            for key, value in wheel_forces.items():
                lam[con.rows(key)] += value
        else:
            z, lam = solve(rhs)
        residual = float(np.max(np.abs(A @ z - rhs)))
        return dict(f=z[:self.nf], lam=lam, ports={b['id']: lam[b['rows']].copy() for b in con.blocks}, residual=residual,
                    constraints=con, wheel_loads=used)

    def recover(self, q, v, a, f, mode, known=None, kin=None):
        """Constraint reactions implied by a measured motion and measured actuator efforts.

        Solves ``J^T lam = M a + h - B f`` for the reactions of the mode.  ``known`` maps
        ports to reactions taken as given (for example wrist-sensor wrenches while the
        payload also rests on supports, where the split is not determined by rigid-body
        mechanics alone).  Returns the reactions per port, the largest unbalanced
        generalized force and the rank deficiency of the solved rows.
        """
        kin = self.tree.sweep(q) if kin is None else kin
        con = self.constraints(q, mode, kin)
        resid = self.tree.rnea(q, v, a, kinematics=kin) + self.damping * np.asarray(v, float) - self.B @ np.asarray(f, float)
        lam = np.zeros(con.J.shape[0])
        free = []
        for b in con.blocks:
            if known and b['id'] in known:
                lam[b['rows']] = known[b['id']]
            else:
                free += list(range(b['rows'].start, b['rows'].stop))
        free = np.array(free, int)
        rhs = resid - con.J.T @ lam
        A = con.J[free].T
        shared = self._shared_axle_forces(con, free)
        if shared is not None:
            # wheels on one axle of a friction-driven base: their forces along the axle act on one line, so
            # only their sum is determined.  It is one unknown, split equally, instead of one unknown per wheel
            # that a measured state slightly off the ideal ground could trade against the others without bound.
            sol, _, rank, _ = np.linalg.lstsq(A @ shared, rhs, rcond=1e-10)
            sol, rank = shared @ sol, rank + (shared.shape[0] - shared.shape[1])
        else:
            sol, _, rank, _ = np.linalg.lstsq(A, rhs, rcond=1e-10)
        lam[free] = sol
        return dict(ports={b['id']: lam[b['rows']].copy() for b in con.blocks}, constraints=con,
                    unbalanced=float(np.max(np.abs(con.J.T @ lam - resid))), deficiency=int(len(free) - rank))

    def _shared_axle_forces(self, con, free):
        """Map from reduced to free multipliers that gives the wheels of one axle one common axle force (or None)."""
        position = {int(row): k for k, row in enumerate(free)}
        columns, merged = [], set()
        for b in con.blocks:
            if not b.get('slip') or b['rows'].start not in position:
                continue
            robot = self.comp.robots[b['robot']]
            lateral = {label.split(':')[0]: b['rows'].start + k for k, label in enumerate(b['labels']) if label.endswith(':lateral')}
            axles = {}
            for w in robot.wheels:
                # an axle: the line through the wheel centre along the wheel axis
                e = np.asarray(w['axis_base'], float) / np.linalg.norm(w['axis_base'])
                c = np.asarray(w['centre_base'], float)
                foot = c - (c @ e) * e
                key = tuple(np.round(np.r_[foot, e if e[np.argmax(np.abs(e))] > 0 else -e], 6))
                axles.setdefault(key, []).append(lateral[w['id']])
            for rows in axles.values():
                if len(rows) > 1:
                    column = np.zeros(len(free))
                    column[[position[r] for r in rows]] = 1. / len(rows)
                    columns.append(column)
                    merged.update(position[r] for r in rows)
        if not columns:
            return None
        keep = [k for k in range(len(free)) if k not in merged]
        return np.column_stack([np.eye(len(free))[:, keep]] + columns)

    # ------------------------------------------------------------------ determinacy
    def identifiability(self, q, mode, constraints=None):
        """Per port: are its reactions uniquely determined by rigid-body mechanics at ``q``?"""
        con = self.constraints(q, mode) if constraints is None else constraints
        C = np.hstack([con.J, con.Jc])
        total = _rank(C)
        out = {}
        for b in con.blocks:
            rows = np.arange(b['rows'].start, b['rows'].stop)
            others = np.setdiff1d(np.arange(C.shape[0]), rows)
            unique = total == (_rank(C[others]) if len(others) else 0) + len(rows)
            components = [bool(total == (_rank(np.delete(C, r, axis=0)) if C.shape[0] > 1 else 0) + 1) for r in rows]
            out[b['id']] = dict(determinate=bool(unique), components=components)
        return dict(ports=out, rows=int(C.shape[0]), rank=int(total), redundant_rows=int(C.shape[0] - total))

    # ------------------------------------------------------------------ effort space
    def effort_split(self, q, mode, lift=None):
        """Actuator efforts that produce no motion, and their map to the reactions.

        Returns a basis ``Nf`` of the null space of ``L^T`` and ``Gamma`` with
        ``delta_lam = Gamma @ phi`` for ``delta_f = Nf @ phi``.
        """
        lift = self.lift(q, mode) if lift is None else lift
        LT = lift.E.T @ self.B
        U, s, Vt = np.linalg.svd(LT)
        rk = int(np.sum(s > 1e-10 * s[0])) if s.size else 0
        Nf = Vt[rk:].T
        rhs = np.vstack([-(self.B @ Nf)[lift.dependent], np.zeros((lift.constraints.Jc.shape[1], Nf.shape[1]))])
        Gamma = np.zeros((lift.constraints.J.shape[0], Nf.shape[1]))
        if lift.lu is not None and Nf.shape[1]:
            Gamma[lift.rows] = lu_solve(lift.lu, rhs, trans=1)
        return dict(Nf=Nf, Gamma=Gamma, rank=rk, dimension=Nf.shape[1], L=LT.T)
