"""MuJoCo plant of a compiled team (state in compiled coordinates, efforts at the compiled ports).

The plant is physics only.  It exposes what a real cell would expose: joint and
base states, payload pose (declared as measured), wrist force/torque sensors, a
latch command per clamp, and the efforts of the actuator ports.  Contact forces
are reported for the audit, not for control.
"""
from __future__ import annotations

import mujoco
import numpy as np

from teamc.se3 import chart_pose, chart_rate_map, from_quat_wxyz, inv, pose_chart, quat_wxyz, transform

from .latch import LatchSet, check_latch_stability
from .mujoco_export import build_mjcf
from .tyre import TyreSet, check_tyre_stability

ALLOWED_PAD = ('pad',)
FORBIDDEN_ROLES = ('neighbour', 'rack', 'wall', 'door', 'mount')


class MujocoPlant:
    name = 'mujoco'
    mujoco_version = mujoco.__version__

    def __init__(self, comp, timestep=.002, q0=None, roller_hull=(41, 64), latch='connect', sensor='native',
                 latch_params=None, tyre='model', **export):
        """``latch``: ``'connect'`` (three point constraints of the solver) or ``'compliant'`` (the clamp
        model of ``backends/latch.py`` applied as external loads, as in the Isaac Sim plant).
        ``sensor``: ``'native'`` (MuJoCo force/torque sensors) or ``'model'`` (Newton-Euler balance of
        the tool-side body, compliant latch only).
        ``tyre`` (conventional wheels only): ``'model'`` (their contacts are frictionless and the tyre model
        of ``backends/tyre.py`` is applied as external loads, as in the Isaac Sim plant) or ``'native'``
        (their friction is that of the MuJoCo contacts; see ``backends/mujoco_export.py`` for what that
        friction does to a base that turns while it drives)."""
        if latch not in ('connect', 'compliant') or sensor not in ('native', 'model'):
            raise ValueError('latch must be connect|compliant and sensor native|model')
        if sensor == 'model' and latch != 'compliant':
            raise ValueError("sensor='model' needs latch='compliant'")
        self.comp = comp
        self.latch_mode, self.sensor_mode = latch, sensor
        self.latches = LatchSet(comp, latch_params) if latch == 'compliant' else None
        self.tyres = (TyreSet(comp) or None) if tyre == 'model' else None
        self.tyre_mode = tyre if any(r.drive == 'friction' for r in comp.robots.values()) else None
        self._tyre_normal, self._tyre_loads, self.tyre_ratio = {}, {}, 0.
        self._loads, self._model_wrench, self._state_cache = {}, {}, None
        self.latch_ratio = 0.
        self.xml, self.meta = build_mjcf(comp, timestep=timestep, q0=q0, roller_hull=roller_hull, tyre=tyre, **export)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        m, meta = self.model, self.meta
        self.n = comp.tree.n
        jid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        self._qadr = np.full(self.n, -1)
        self._vadr = np.full(self.n, -1)
        for index, name in meta['coordinate_joint'].items():
            j = jid(name)
            self._qadr[index], self._vadr[index] = m.jnt_qposadr[j], m.jnt_dofadr[j]
        self._scalar = np.flatnonzero(self._qadr >= 0)
        self._floating = []
        for item in list(meta['robots'].values()) + [meta['payload']]:
            j = jid(item['free'])
            self._floating.append((np.asarray(item['base'], int), int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])))
        self._act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, p) for p in meta['ports']])
        self._bounds = m.actuator_ctrlrange[self._act].copy()
        self._sensor = {}
        for rid, r in meta['robots'].items():
            if 'sensor' in r:
                f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, r['sensor']['force'])
                t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, r['sensor']['torque'])
                self._sensor[rid] = (int(m.sensor_adr[f]), int(m.sensor_adr[t]))
        self._weld = {cid: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, name) for name in w['names']]
                      for cid, w in meta['welds'].items()}
        self._body = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        self._geom_tag = [meta['geoms'].get(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), dict(group='?', role='?'))
                          for g in range(m.ngeom)]
        self._geom_name = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)]
        self._payload_dof = self._floating[-1][2]
        self._tree_body = {name: (comp.tree.index[name], self._body(name)) for name in
                           {w['body1'] for w in meta['welds'].values()} | {w['body2'] for w in meta['welds'].values()}
                           | ({w['body'] for w in self.tyres.wheels} if self.tyres else set())}
        self._tyre_body = []                                     # per tyre: MuJoCo body and the dof of its wheel joint
        if self.tyres:
            for w in self.tyres.wheels:
                rid, wid = w['key']
                joint = jid(meta['robots'][rid]['wheels'][wid]['joint'])
                self._tyre_body.append((self._body(w['body']), int(m.jnt_dofadr[joint])))
        self.latched = {cid: False for cid in meta['welds']}
        self.saturated = 0
        self._renderer = None
        self._yaw = (comp.seed() if q0 is None else np.asarray(q0, float)).copy()
        self._model_wrench = {rid: np.zeros(6) for rid in self._sensor}
        mujoco.mj_forward(m, self.data)

    # ------------------------------------------------------------------ state
    @property
    def time(self):
        return float(self.data.time)

    def reset(self, q, v=None):
        """Set the plant to a compiled configuration (rollers at zero)."""
        q = np.asarray(q, float)
        v = np.zeros(self.n) if v is None else np.asarray(v, float)
        d = self.data
        mujoco.mj_resetData(self.model, d)
        self._yaw = q.copy()
        d.qpos[self._qadr[self._scalar]] = q[self._scalar]
        d.qvel[self._vadr[self._scalar]] = v[self._scalar]
        for idx, qa, va in self._floating:
            T = chart_pose(q[idx])
            d.qpos[qa:qa + 3] = T[:3, 3]
            d.qpos[qa + 3:qa + 7] = quat_wxyz(T[:3, :3])
            d.qvel[va:va + 3] = v[idx[:3]]
            d.qvel[va + 3:va + 6] = T[:3, :3].T @ (chart_rate_map(q[idx]) @ v[idx[3:]])
        for cid in self.latched:
            d.eq_active[self._weld[cid]] = 0
            self.latched[cid] = False
        if self.latches is not None:
            self.latches.reset()
        self._loads, self._state_cache = {}, None
        self._model_wrench = {rid: np.zeros(6) for rid in self._sensor}
        mujoco.mj_forward(self.model, d)
        if self.tyres:
            self.tyres.reset()
            self._tyre_loads = {}
            self._tyre_normal = self._wheel_normals()
            self.tyre_ratio = check_tyre_stability(self.tyres, q, self.dt)

    def kick(self, dv):
        """Add ``dv`` (compiled rates) to the velocity of the plant: what an impulse does (disturbance tests)."""
        dv = np.asarray(dv, float)
        d = self.data
        q = self.state()[0]
        d.qvel[self._vadr[self._scalar]] += dv[self._scalar]
        for idx, qa, va in self._floating:
            R = from_quat_wxyz(d.qpos[qa + 3:qa + 7])
            d.qvel[va:va + 3] += dv[idx[:3]]
            d.qvel[va + 3:va + 6] += R.T @ (chart_rate_map(q[idx]) @ dv[idx[3:]])
        self._state_cache = None
        mujoco.mj_forward(self.model, d)

    def state(self):
        """Compiled coordinates and rates ``(q, v)`` of the current plant state (yaw kept continuous)."""
        d = self.data
        if self._state_cache is not None and self._state_cache[0] == d.time:
            return self._state_cache[1].copy(), self._state_cache[2].copy()
        q, v = np.zeros(self.n), np.zeros(self.n)
        q[self._scalar] = d.qpos[self._qadr[self._scalar]]
        v[self._scalar] = d.qvel[self._vadr[self._scalar]]
        for idx, qa, va in self._floating:
            R = from_quat_wxyz(d.qpos[qa + 3:qa + 7])
            c = pose_chart(transform(R, d.qpos[qa:qa + 3]), self._yaw[idx[3]])
            q[idx] = c
            v[idx[:3]] = d.qvel[va:va + 3]
            v[idx[3:]] = np.linalg.solve(chart_rate_map(c), R @ d.qvel[va + 3:va + 6])
            self._yaw[idx[3]] = c[3]
        self._state_cache = (d.time, q.copy(), v.copy())
        return q, v

    def body_pose(self, name):
        i = self._body(name)
        return transform(self.data.xmat[i].reshape(3, 3), self.data.xpos[i])

    # ------------------------------------------------------------------ inputs
    def set_efforts(self, f):
        """Actuator port efforts in compiled port order (clipped to the declared bounds)."""
        f = np.asarray(f, float)
        clipped = np.clip(f, self._bounds[:, 0], self._bounds[:, 1])
        self.saturated += int(np.any(clipped != f))
        self.data.ctrl[self._act] = clipped
        return clipped

    def set_obstacle(self, position):
        if 'obstacle' in self.meta:
            self.data.mocap_pos[0] = position

    def measure(self):
        """What the cell measures: state, wrist sensors, time."""
        q, v = self.state()
        return dict(t=self.time, q=q, v=v, wrench={rid: self.sensor_wrench(rid) for rid in self._sensor})

    def _step_tyres(self):
        """One step with the tyre model alone: the ground kinematics of the wheels come straight from the MuJoCo state."""
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)                                # poses and velocities of the state as it is now
        mujoco.mj_comPos(m, d)                                    # (after a step they are still those of its start)
        mujoco.mj_comVel(m, d)
        state, vel = [], np.zeros(6)
        for w, (mi, dof) in zip(self.tyres.wheels, self._tyre_body):
            centre = d.xpos[mi]
            point, normal = self.tyres.ground_point(w, centre)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, mi, vel, 0)     # [angular, linear] at the body origin
            slip = vel[3:] + np.cross(vel[:3], point - centre)
            state.append((point, normal, slip - (slip @ normal) * normal, float(d.qvel[dof])))
        loads = self.tyres.advance(state, self._tyre_normal, self.dt)
        for load, (mi, _) in zip(loads.values(), self._tyre_body):
            f = -load['force']                                    # on the wheel, at its ground point
            d.xfrc_applied[mi, :3] = f
            d.xfrc_applied[mi, 3:] = np.cross(load['point'] - d.xipos[mi], f)
        mujoco.mj_step(m, d)
        self._tyre_loads, self._tyre_normal = loads, self._wheel_normals()

    def step(self, n=1):
        if self.latches is None and not self.tyres:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)
            return
        if self.latches is None:
            for _ in range(n):
                self._step_tyres()
            return
        tree, d = self.comp.tree, self.data
        for _ in range(n):
            q, v = self.state()
            kin = tree.sweep(q)
            loads = self.latches.loads(q, v, kin) if self.latches is not None else {}
            tyre_loads = self.tyres.loads(q, v, self._tyre_normal, self.dt, kin) if self.tyres else {}
            for ti, mi in self._tree_body.values():
                d.xfrc_applied[mi] = 0.
            for load in list(loads.values()) + list(tyre_loads.values()):
                # clamp: equal and opposite, at the tool-frame origin; tyre: on the wheel, at its ground point
                for body, sign in ((load['on'], 1.), (load['by'], -1.)):
                    if body is None:
                        continue
                    ti, mi = self._tree_body[body]
                    com = kin[0][ti, :3, 3] + kin[0][ti, :3, :3] @ tree.com[ti]
                    f = sign * load['force']
                    d.xfrc_applied[mi, :3] += f
                    d.xfrc_applied[mi, 3:] += sign * load['moment'] + np.cross(load['point'] - com, f)
            mujoco.mj_step(self.model, d)
            if self.tyres:
                self._tyre_loads, self._tyre_normal = tyre_loads, self._wheel_normals()
            if self.latches is None:
                continue
            self._loads = dict(loads, origin=kin[0][tree.index[self.meta['payload']['body']], :3, 3].copy())
            if self.sensor_mode == 'model':
                _, v1 = self.state()
                self._model_wrench = self.latches.sensors(q, v, (v1 - v) / self.dt, loads, kin=kin)

    def _wheel_normals(self):
        """Normal load of every wheel from the contacts of the present state: ``{(robot, wheel): N}``."""
        d, m = self.data, self.model
        out, f6 = {}, np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                t = self._geom_tag[g]
                if t['role'] == 'roller':
                    mujoco.mj_contactForce(m, d, i, f6)
                    key = (t['group'], t['wheel'])
                    out[key] = out.get(key, 0.) + float(f6[0])
        return out

    # ------------------------------------------------------------------ grasp connections
    def tool_pose(self, cid):
        w = self.meta['welds'][cid]
        return self.body_pose(w['body1']) @ w['T1']

    def latch(self, cid):
        """Close the clamp of a grasp connection at the current relative pose.

        Returns the as-latched payload-side frame (tool frame expressed in the payload body).
        """
        w = self.meta['welds'][cid]
        if self.latches is not None:
            self.latched[cid] = True
            q = self.state()[0]
            T2 = self.latches.close(cid, q)
            self.latch_ratio = max(self.latch_ratio, check_latch_stability(self.latches, q, self.dt))
            return T2
        T1, T2 = self.body_pose(w['body1']), self.body_pose(w['body2'])
        for eq, point in zip(self._weld[cid], w['points']):
            data = self.model.eq_data[eq]
            data[0:3] = point                                   # anchor in body 1 (clamp)
            data[3:6] = (inv(T2) @ T1 @ np.r_[point, 1.])[:3]   # the same point in body 2 (payload)
            self.data.eq_active[eq] = 1
        self.latched[cid] = True
        return inv(T2) @ T1 @ w['T1']

    def unlatch(self, cid):
        self.data.eq_active[self._weld[cid]] = 0
        self.latched[cid] = False
        if self.latches is not None:
            self.latches.open(cid)

    def weld_wrench(self, cid):
        """Wrench a latched connection applies to the payload, from the solver (audit only).

        Returns ``(force_world, moment_world about the payload body origin)``.
        """
        d, m = self.data, self.model
        if self.latches is not None:
            if cid not in self._loads:
                return np.zeros(3), np.zeros(3)
            return self.latches.payload_wrench(self._loads[cid], self._loads['origin'])
        rows = np.flatnonzero((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                              & np.isin(d.efc_id[:d.nefc], self._weld[cid]))
        if not len(rows):
            return np.zeros(3), np.zeros(3)
        if mujoco.mj_isSparse(m):
            J = np.zeros((d.nefc, m.nv))
            mujoco.mju_sparse2dense(J, d.efc_J, d.efc_J_rownnz, d.efc_J_rowadr, d.efc_J_colind)
        else:
            J = d.efc_J.reshape(d.nefc, m.nv)
        a = self._payload_dof
        g = J[rows][:, a:a + 6].T @ d.efc_force[rows]
        R = self.body_pose(self.meta['payload']['body'])[:3, :3]
        return g[:3].copy(), R @ g[3:]

    # ------------------------------------------------------------------ measurements
    def sensor_wrench(self, rid):
        """Raw force/torque sensor reading ``[moment; force]`` in the sensor frame.

        Sign: the wrench the sensor-side body applies to the tool-side body (see
        ``tests/test_mujoco_plant.py`` for the check against a known load).
        """
        if self.sensor_mode == 'model':
            return self._model_wrench[rid].copy()
        return self.native_sensor_wrench(rid)

    def native_sensor_wrench(self, rid):
        fa, ta = self._sensor[rid]
        s = self.data.sensordata
        return np.r_[s[ta:ta + 3], s[fa:fa + 3]]

    def contacts(self):
        """Contact forces by category (audit).

        ``pads``: normal load per support pad; ``wheels``: resultant ground force on each
        wheel as ``[normal, tangential]`` (vector sum over its roller contacts);
        ``guides``: payload on the mount guides; ``forbidden``: any other contact of a
        robot or the payload with the scene, another robot or the cart.
        """
        d, m = self.data, self.model
        out = dict(pads={}, wheels={}, guides=0., forbidden=[], tool=0.)
        vec = {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            mujoco.mj_contactForce(m, d, i, f6)
            if f6[0] <= 0.:
                continue
            ta, tb = self._geom_tag[c.geom1], self._geom_tag[c.geom2]
            na, nb = self._geom_name[c.geom1], self._geom_name[c.geom2]
            roles = {ta['role'], tb['role']}
            if 'roller' in roles:
                t = ta if ta['role'] == 'roller' else tb
                frame = c.frame.reshape(3, 3)
                sign = 1. if tb['role'] == 'roller' else -1.          # force on the second geom is along +normal
                e = vec.setdefault((t['group'], t['wheel']), [np.zeros(3), np.zeros(3)])
                e[0] += sign * (frame.T @ f6[:3])
                e[1] += frame[0] * f6[0] * sign
            elif roles == {'payload', 'pad'}:
                pad = na if ta['role'] == 'pad' else nb
                out['pads'][pad] = out['pads'].get(pad, 0.) + float(f6[0])
            elif roles == {'payload', 'guide'}:
                out['guides'] += float(f6[0])
            elif roles == {'payload', 'robot'}:
                name = na if ta['role'] == 'robot' else nb
                if 'clamp' in name:
                    out['tool'] += float(f6[0])
                else:
                    out['forbidden'].append((na, nb, float(f6[0])))
            elif 'terrain' in roles and 'robot' not in roles and 'payload' not in roles:
                continue
            else:
                out['forbidden'].append((na, nb, float(f6[0])))
        for key, (force, normal) in vec.items():
            n = float(np.linalg.norm(normal))
            out['wheels'][key] = [n, float(np.linalg.norm(force - normal))] if n > 0. else [0., 0.]
        if self.tyres:                                           # frictionless contacts: the tangential force is the tyre load
            for key, load in self._tyre_loads.items():
                entry = out['wheels'].setdefault(key[1:], [0., 0.])
                entry[1] = float(np.linalg.norm(load['force']))
        return out

    # ------------------------------------------------------------------ rendering (optional)
    def render(self, lookat, distance, azimuth, elevation, width=1280, height=720):
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (width, height):
            self._renderer = mujoco.Renderer(self.model, height, width)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render().copy()
