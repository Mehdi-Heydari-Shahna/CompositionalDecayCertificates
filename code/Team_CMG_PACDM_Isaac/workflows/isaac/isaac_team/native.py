"""CPU PhysX tensor boundary of a team scene.

One articulation view per robot, a rigid-body view for the payload and one for the
cart, and the contact-report observer.  Only frame and unit conversions happen
here; mechanics are computed by the compiled model.  Conventions of the PhysX
tensor API used below (the same as in the other RoboCompiler Isaac workflows):

* transforms are ``[x, y, z, qx, qy, qz, qw]`` of the body origin, world frame;
* velocities are ``[linear velocity of the centre of mass, angular velocity]``, world frame;
* external forces are applied with their application point (world frame);
* joint efforts are generalized forces in SI units per degree of freedom;
* names of links and degrees of freedom are the prim names of bodies and joints.

Every call uses float32 arrays of the documented shapes; values are converted to
float64 on the way in.
"""
from __future__ import annotations

import numpy as np

from teamc.se3 import chart_pose, chart_rate_map, pose_chart, quat_wxyz, transform

from .runtime import to_numpy


# ---------------------------------------------------------------------- conversions
def rot_xyzw(q):
    """Rotation matrix of a quaternion ``(x, y, z, w)``."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1. - 2. * (y * y + z * z), 2. * (x * y - z * w), 2. * (x * z + y * w)],
                     [2. * (x * y + z * w), 1. - 2. * (x * x + z * z), 2. * (y * z - x * w)],
                     [2. * (x * z - y * w), 2. * (y * z + x * w), 1. - 2. * (x * x + y * y)]])


def quat_xyzw(R):
    w, x, y, z = quat_wxyz(R)
    return np.array([x, y, z, w])


def pose_matrix(pose):
    """4x4 transform of a tensor-API pose ``[x, y, z, qx, qy, qz, qw]``."""
    return transform(rot_xyzw(pose[3:7]), pose[:3])


def to_chart(pose, velocity, com_local, yaw_near=None):
    """Chart coordinates and rates of a free body from its tensor-API pose and velocity."""
    R = rot_xyzw(pose[3:7])
    c = pose_chart(transform(R, pose[:3]), yaw_near)
    omega = np.asarray(velocity[3:6], float)
    origin_velocity = np.asarray(velocity[:3], float) - np.cross(omega, R @ com_local)
    return c, np.r_[origin_velocity, np.linalg.solve(chart_rate_map(c), omega)]


def from_chart(c, rates, com_local):
    """Tensor-API pose and velocity of a free body from its chart coordinates and rates."""
    T = chart_pose(c)
    omega = chart_rate_map(c) @ np.asarray(rates[3:6], float)
    com_velocity = np.asarray(rates[:3], float) + np.cross(omega, T[:3, :3] @ com_local)
    return np.r_[T[:3, 3], quat_xyzw(T[:3, :3])], np.r_[com_velocity, omega]


def wrap(angle):
    return (np.asarray(angle, float) + np.pi) % (2. * np.pi) - np.pi


def _f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


class _Robot:
    pass


class NativeTeam:
    """Tensor views of a team scene authored by ``backends.usd_export.build_usd``."""

    def __init__(self, sim, meta, comp, dt):
        self.sim, self.meta, self.comp, self.dt = sim, meta, comp, float(dt)
        self.n = comp.tree.n
        self.index = np.array([0], dtype=np.int32)
        self.robots = {}
        self.writes = 0
        self.joint_force_note = None
        for rid, r in meta['robots'].items():
            view = sim.create_articulation_view(r['root'])
            if view is None or int(view.count) != 1:
                raise RuntimeError(f'{rid}: expected exactly one articulation at {r["root"]}')
            mt = view.shared_metatype
            dofs, links = [str(x) for x in mt.dof_names], [str(x) for x in mt.link_names]
            if sorted(dofs) != sorted(r['dofs']):
                raise RuntimeError(f'{rid}: degrees of freedom differ from the authored joints: missing '
                                   f'{sorted(set(r["dofs"]) - set(dofs))[:5]}, unexpected {sorted(set(dofs) - set(r["dofs"]))[:5]}')
            if sorted(links) != sorted(r['links']):
                raise RuntimeError(f'{rid}: links differ from the authored bodies: missing '
                                   f'{sorted(set(r["links"]) - set(links))[:5]}, unexpected {sorted(set(links) - set(r["links"]))[:5]}')
            if bool(getattr(mt, 'fixed_base', False)):
                raise RuntimeError(f'{rid}: the floating robot was fixed to the world')
            x = _Robot()
            x.id, x.view, x.dofs, x.links = rid, view, dofs, links
            x.ndof, x.nlink = len(dofs), len(links)
            joints = list(r['joints'].values())
            x.native = np.array([dofs.index(j['dof']) for j in joints])                # compiled joints in native order
            x.coord = np.array([j['index'] for j in joints])                           # their compiled coordinates
            free = np.array([r['limits'][j['dof']] is None and j['type'] == 'revolute' for j in joints])
            x.turning = np.flatnonzero(free)                                           # joints PhysX may wrap (wheels)
            x.rollers = np.array([dofs.index(d) for d in r['rollers']], dtype=int)
            x.damping = np.array([r['damping'][d] for d in dofs])
            x.base = np.asarray(r['base'], int)
            x.base_link = links.index(r['root'].rsplit('/', 1)[-1])
            x.link_of = {name: links.index(path.rsplit('/', 1)[-1]) for name, path in r['bodies'].items()}
            x.tool_link = x.link_of[r['tool']['body']]
            coms = np.asarray(to_numpy(view.get_coms()), float).reshape(1, x.nlink, 7)[0]
            x.coms = coms[:, :3].copy()
            x.com = x.coms[x.base_link].copy()
            x.effort = np.zeros((1, x.ndof), np.float32)
            x.force = np.zeros((1, x.nlink, 3), np.float32)
            x.torque = np.zeros((1, x.nlink, 3), np.float32)
            x.point = np.zeros((1, x.nlink, 3), np.float32)
            x.dof_velocity = np.zeros(x.ndof)
            x.raw = np.zeros(len(x.turning))
            x.turns = np.zeros(len(x.turning))
            x.yaw = None
            self.robots[rid] = x
        p = meta['payload']
        self.payload = sim.create_rigid_body_view(p['path'])
        if self.payload is None or len(list(self.payload.prim_paths)) != 1:
            raise RuntimeError(f'expected exactly one payload body at {p["path"]}')
        self.payload_base = np.asarray(p['base'], int)
        self.payload_com = np.asarray(to_numpy(self.payload.get_coms()), float).reshape(-1, 7)[0, :3].copy()
        self.payload_yaw = None
        self.payload_pose = None
        self._pf, self._pt, self._pp = (np.zeros((1, 3), np.float32) for _ in range(3))
        self.cart, self.cart_mode, self.cart_position = None, 'absent', None
        if meta.get('obstacle'):
            try:
                self.cart = sim.create_rigid_body_view(meta['obstacle']['path'])
                if self.cart is None or len(list(self.cart.prim_paths)) != 1:
                    raise RuntimeError('no body view')
                self.cart_mode = 'kinematic_targets' if callable(getattr(self.cart, 'set_kinematic_targets', None)) else 'transforms'
                self.cart_position = np.asarray(meta['obstacle']['start'], float).copy()
            except Exception as exc:                                  # noqa: BLE001 (the cart is not essential to the physics)
                self.cart, self.cart_mode = None, f'unavailable ({type(exc).__name__}: {exc})'
        sim.update_articulations_kinematic()

    # ------------------------------------------------------------------ state
    def set_state(self, q, v=None):
        """Initialisation only: write a compiled configuration (rollers at zero, wheels within one turn)."""
        q = np.asarray(q, float)
        v = np.zeros(self.n) if v is None else np.asarray(v, float)
        for x in self.robots.values():
            pose, velocity = from_chart(q[x.base], v[x.base], x.com)
            jp, jv = np.zeros((1, x.ndof), np.float32), np.zeros((1, x.ndof), np.float32)
            values = q[x.coord].copy()
            values[x.turning] = wrap(values[x.turning])
            jp[0, x.native] = values
            jv[0, x.native] = v[x.coord]
            x.view.set_root_transforms(_f32(pose[None]), self.index)
            x.view.set_root_velocities(_f32(velocity[None]), self.index)
            x.view.set_dof_positions(jp, self.index)
            x.view.set_dof_velocities(jv, self.index)
            x.effort[:] = 0.
            x.view.set_dof_actuation_forces(x.effort, self.index)
            x.raw = jp[0, x.native][x.turning].astype(float)
            x.turns = q[x.coord][x.turning] - x.raw
            x.yaw = float(q[x.base[3]])
            self.writes += 1
        pose, velocity = from_chart(q[self.payload_base], v[self.payload_base], self.payload_com)
        self.payload.set_transforms(_f32(pose[None]), self.index)
        self.payload.set_velocities(_f32(velocity[None]), self.index)
        self.payload_yaw = float(q[self.payload_base[3]])
        self.sim.update_articulations_kinematic()

    def state(self):
        """Compiled coordinates and rates of the native state (yaw and wheel angles kept continuous)."""
        q, v = np.zeros(self.n), np.zeros(self.n)
        for x in self.robots.values():
            pose = np.array(to_numpy(x.view.get_root_transforms())[0], float)
            velocity = np.array(to_numpy(x.view.get_root_velocities())[0], float)
            jp = np.array(to_numpy(x.view.get_dof_positions())[0], float)
            jv = np.array(to_numpy(x.view.get_dof_velocities())[0], float)
            c, rates = to_chart(pose, velocity, x.com, x.yaw)
            x.yaw = float(c[3])
            q[x.base], v[x.base] = c, rates
            values = jp[x.native]
            raw = values[x.turning]
            x.turns += wrap(raw - x.raw) - (raw - x.raw)              # whole turns PhysX removed (or added)
            x.raw = raw.copy()
            values[x.turning] = raw + x.turns
            q[x.coord], v[x.coord] = values, jv[x.native]
            x.dof_velocity = jv
        pose = np.array(to_numpy(self.payload.get_transforms()), float).reshape(-1, 7)[0]
        velocity = np.array(to_numpy(self.payload.get_velocities()), float).reshape(-1, 6)[0]
        c, rates = to_chart(pose, velocity, self.payload_com, self.payload_yaw)
        self.payload_yaw = float(c[3])
        self.payload_pose = pose
        q[self.payload_base], v[self.payload_base] = c, rates
        if not (np.isfinite(q).all() and np.isfinite(v).all()):
            raise FloatingPointError('non-finite PhysX state')
        return q, v

    # ------------------------------------------------------------------ inputs
    def set_efforts(self, tau):
        """Generalized efforts by compiled coordinate; the declared joint damping is added here (explicit)."""
        for x in self.robots.values():
            x.effort[0] = -x.damping * x.dof_velocity
            x.effort[0, x.native] += tau[x.coord]
            x.view.set_dof_actuation_forces(x.effort, self.index)

    def apply_loads(self, loads, payload_com_world):
        """External loads of one step.

        A clamp load acts on the payload (``on``) and with the opposite sign on the clamp body (``by``).
        A tyre load has no ``on`` body: the wheel (``by``) receives the opposite of what the tyre applies
        to the ground.
        """
        if not loads:
            return
        force, moment = np.zeros(3), np.zeros(3)
        by_body, on_payload = {}, False
        for load in loads.values():
            if load.get('on') is not None:
                force += load['force']
                moment += load['moment'] + np.cross(load['point'] - payload_com_world, load['force'])
                on_payload = True
            # several loads on one body (a clamp load and a known load of a check, say) add up: one force at the point
            # of the first, and the moment of the others about it
            entry = by_body.get(load['by'])
            if entry is None:
                by_body[load['by']] = [np.asarray(load['point'], float).copy(), np.asarray(load['force'], float).copy(),
                                       np.asarray(load['moment'], float).copy()]
            else:
                entry[1] = entry[1] + load['force']
                entry[2] = entry[2] + load['moment'] + np.cross(np.asarray(load['point'], float) - entry[0], load['force'])
        if on_payload:
            self._pf[0], self._pt[0], self._pp[0] = force, moment, payload_com_world
            self.payload.apply_forces_and_torques_at_position(self._pf, self._pt, self._pp, self.index, True)
        for x in self.robots.values():
            hit = False
            x.force[:] = 0.
            x.torque[:] = 0.
            x.point[:] = 0.
            for body, (point, f, m) in by_body.items():
                link = x.link_of.get(body)
                if link is None:
                    continue
                x.force[0, link], x.torque[0, link], x.point[0, link] = -f, -m, point
                hit = True
            if hit:
                x.view.apply_forces_and_torques_at_position(x.force, x.torque, x.point, self.index, True)

    def set_cart(self, position):
        """Move the kinematic cart.  Never fatal: the scenario owns the cart position, the body only shows it and collides."""
        if self.cart is None:
            return False
        position = np.asarray(position, float)
        if self.cart_position is not None and np.max(np.abs(position - self.cart_position)) < 1e-9:
            return True
        pose = _f32(np.r_[position, 0., 0., 0., 1.][None])
        for mode in ([self.cart_mode] + [m for m in ('kinematic_targets', 'transforms') if m != self.cart_mode]):
            try:
                if mode == 'kinematic_targets':
                    self.cart.set_kinematic_targets(pose, self.index)
                else:
                    self.cart.set_transforms(pose, self.index)
                self.cart_mode = mode
                self.cart_position = position.copy()
                return True
            except Exception as exc:                                  # noqa: BLE001
                failure = f'{type(exc).__name__}: {exc}'
        self.cart, self.cart_mode = None, f'unavailable ({failure})'
        return False

    def cart_pose(self):
        return None if self.cart is None else np.array(to_numpy(self.cart.get_transforms()), float).reshape(-1, 7)[0]

    # ------------------------------------------------------------------ readback (preflight)
    def link_poses(self, rid):
        """World poses of the links of a robot by compiled body name (after a kinematic update)."""
        x = self.robots[rid]
        self.sim.update_articulations_kinematic()
        poses = np.asarray(to_numpy(x.view.get_link_transforms()), float).reshape(1, x.nlink, 7)[0]
        return {name: pose_matrix(poses[k]) for name, k in x.link_of.items()}

    def properties(self, rid):
        """Mass properties and joint properties PhysX actually uses (by link name and by degree of freedom)."""
        x = self.robots[rid]
        out = dict(links=x.links, dofs=x.dofs, unavailable=[])
        masses = np.asarray(to_numpy(x.view.get_masses()), float).reshape(1, x.nlink)[0]
        coms = np.asarray(to_numpy(x.view.get_coms()), float).reshape(1, x.nlink, 7)[0]
        # column-major inertia about the centre of mass, in the axes of the body prim
        inertias = np.asarray(to_numpy(x.view.get_inertias()), float).reshape(1, x.nlink, 3, 3)[0].transpose(0, 2, 1)
        out.update(mass=masses, com=coms[:, :3], inertia=inertias)

        def optional(key, getter, shape):
            try:
                out[key] = np.asarray(to_numpy(getter()), float).reshape(shape)[0]
            except Exception as exc:                                  # noqa: BLE001
                out['unavailable'].append(f'{key}: {type(exc).__name__}: {exc}')

        optional('armature', lambda: x.view.get_dof_armatures(), (1, x.ndof))
        optional('stiffness', lambda: x.view.get_dof_stiffnesses(), (1, x.ndof))
        optional('drive_damping', lambda: x.view.get_dof_dampings(), (1, x.ndof))
        optional('limits', lambda: x.view.get_dof_limits(), (1, x.ndof, 2))
        if callable(getattr(x.view, 'get_dof_friction_properties', None)):
            optional('friction', lambda: x.view.get_dof_friction_properties(), (1, -1))
        else:
            optional('friction', lambda: x.view.get_dof_friction_coefficients(), (1, -1))
        return out

    def payload_properties(self):
        mass = float(np.asarray(to_numpy(self.payload.get_masses()), float).reshape(-1)[0])
        com = np.asarray(to_numpy(self.payload.get_coms()), float).reshape(-1, 7)[0, :3]
        inertia = np.asarray(to_numpy(self.payload.get_inertias()), float).reshape(-1, 3, 3)[0].T
        return dict(mass=mass, com=com, inertia=inertia)

    def gravity(self):
        try:
            return np.asarray(to_numpy(self.sim.get_gravity()), float).reshape(3)
        except Exception:                                             # noqa: BLE001
            return None

    def incoming_joint_force(self, rid, body):
        """PhysX's own reaction in the incoming joint of a link: ``[force, torque]`` in the child joint frame.

        Used only as an independent reference for the wrist sensor balance; ``None`` where the call is not available.
        """
        if self.joint_force_note is not None:
            return None
        x = self.robots[rid]
        try:
            values = np.asarray(to_numpy(x.view.get_link_incoming_joint_force()), float).reshape(1, x.nlink, 6)
            return values[0, x.link_of[body]].copy()
        except Exception as exc:                                      # noqa: BLE001
            self.joint_force_note = f'{type(exc).__name__}: {exc}'
            return None

    def cooked_hull(self, path, stage_id):
        """Vertices of the convex hull PhysX cooked for a collider prim (in the units of the authored mesh)."""
        from omni.physx import get_physx_cooking_interface
        from pxr import PhysicsSchemaTools
        found = {}

        def on_result(result, convexes):
            found['result'] = str(result)
            found['hulls'] = [np.array([[v.x, v.y, v.z] for v in c.vertices], float) for c in (convexes or [])]

        get_physx_cooking_interface().request_convex_collision_representation(
            stage_id=int(stage_id), collision_prim_id=PhysicsSchemaTools.sdfPathToInt(path), run_asynchronously=False,
            on_result=on_result)
        if 'hulls' not in found or len(found['hulls']) != 1:
            raise RuntimeError(f"no single cooked hull for {path}: {found.get('result')}")
        return found['hulls'][0]

    def close(self):
        """Drop the tensor views while their stage is still alive."""
        for x in self.robots.values():
            x.view = None
        self.payload = self.cart = self.sim = None


class ContactMonitor:
    """Contact forces of the last physics step by category, from the PhysX contact report.

    The report gives, per contact point, the normal and the normal impulse, and per pair of
    colliders the friction anchors with their impulses; force is impulse / dt.  Where only the
    plain report is available the tangential part of a wheel load is not reported.  The sign
    convention of the impulses (which collider they act on) is learned from the roller/terrain
    contacts, where the force on the roller must point out of the terrain, and from the payload
    on its support pads, where the force on the payload must point up (a team whose tyres carry
    its wheels has no wheel contact).  A collider the scene generator did not author is an
    error: the classification would be incomplete.
    """

    FORCE_FLOOR = 1e-3            # N; smaller reported forces are numerical zeros

    def __init__(self, meta, dt):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self.dt = float(dt)
        self.tags = meta['colliders']
        self._decode = PhysicsSchemaTools.intToSdfPath
        self._paths = {}
        self.error = None
        self.callbacks = self.points = 0
        self._votes = 0.
        self.wheels_seen = set()
        self.pads_seen = set()
        self.unsigned_tool_steps = 0      # steps with clamp contacts before the sign of the impulses was learned
        self.friction_note = None
        self.begin_step()
        interface = get_physx_simulation_interface()
        full = getattr(interface, 'subscribe_full_contact_report_events', None)
        self.friction = callable(full)
        if self.friction:
            self.subscription = full(lambda headers, data, anchors=None, *more: self._callback(headers, data, anchors))
        else:
            self.subscription = interface.subscribe_contact_report_events(lambda headers, data, *more: self._callback(headers, data, None))

    def begin_step(self):
        self.wheels, self.pads, self.guides, self.tool = {}, {}, 0., 0.
        self.forbidden, self.tool_points = {}, []

    def _path(self, value):
        path = self._paths.get(value)
        if path is None:
            path = self._paths[value] = str(self._decode(value))
        return path

    def _callback(self, headers, data, anchors):
        try:
            self.callbacks += 1
            for h in headers:
                count = int(h.num_contact_data)
                if count == 0:
                    continue                                          # contact lost
                pa, pb = self._path(h.collider0), self._path(h.collider1)
                ta, tb = self.tags.get(pa), self.tags.get(pb)
                if ta is None or tb is None:
                    raise RuntimeError(f'contact between colliders the scene generator did not author: {pa}, {pb}')
                tx = ty = tz = normal_sum = 0.
                points = []
                first = int(h.contact_data_offset)
                self.points += count
                for k in range(first, first + count):                 # plain floats: a report has hundreds of points per step
                    c = data[k]
                    impulse = c.impulse
                    ix, iy, iz = float(impulse[0]), float(impulse[1]), float(impulse[2])
                    if ix == 0. and iy == 0. and iz == 0.:
                        continue                                      # within the contact offset without force: most points
                    normal = c.normal
                    tx, ty, tz = tx + ix, ty + iy, tz + iz
                    normal_sum += abs(ix * float(normal[0]) + iy * float(normal[1]) + iz * float(normal[2]))
                    points.append((c, ix, iy, iz))
                normal_sum /= self.dt
                if normal_sum <= self.FORCE_FLOOR:
                    continue
                total = np.array([tx, ty, tz]) / self.dt
                roles = {ta['role'], tb['role']}
                if 'roller' in roles:
                    roller_first = ta['role'] == 'roller'
                    t, other = (ta, tb) if roller_first else (tb, ta)
                    if other['role'] == 'terrain':
                        self._votes += total[2] * (1. if roller_first else -1.)
                        friction = np.zeros(3)
                        if anchors is None and self.friction:
                            self.friction, self.friction_note = False, 'the full contact report came without friction anchors'
                        if self.friction:
                            try:
                                first = int(h.friction_anchors_offset)
                                for k in range(first, first + int(h.num_friction_anchors_data)):
                                    a = anchors[k].impulse
                                    friction += (float(a[0]), float(a[1]), float(a[2]))
                                friction /= self.dt
                            except Exception as exc:                  # noqa: BLE001
                                # the anchors are not laid out as expected: go on without the traction of the wheels
                                # (compared with the compiled one, nothing else); the normal loads above do not depend on them
                                self.friction, self.friction_note = False, f'{type(exc).__name__}: {exc}'
                                friction = np.zeros(3)
                        sign = 1. if total[2] >= 0. else -1.          # on the roller: out of the terrain
                        key = (t['group'], t['wheel'])
                        entry = self.wheels.setdefault(key, [np.zeros(3), np.zeros(3)])
                        entry[0] += sign * total
                        entry[1] += sign * friction
                        self.wheels_seen.add(key)
                    else:
                        self._forbid(ta, tb, normal_sum)
                elif roles == {'payload', 'pad'}:
                    pad = ta['name'] if ta['role'] == 'pad' else tb['name']
                    self.pads[pad] = self.pads.get(pad, 0.) + normal_sum
                    self.pads_seen.add(pad)
                    # a support pad pushes the payload up: the same vote as a roller pushed out of the terrain
                    self._votes += total[2] * (1. if ta['role'] == 'payload' else -1.)
                elif roles == {'payload', 'guide'}:
                    self.guides += normal_sum
                elif roles == {'payload', 'robot'}:
                    robot_first = ta['role'] == 'robot'
                    t = ta if robot_first else tb
                    if 'clamp' in t['name']:
                        self.tool += normal_sum
                        for c, ix, iy, iz in points:
                            position = getattr(c, 'position', None)
                            if position is not None:
                                self.tool_points.append((t['body'], robot_first, np.array([ix, iy, iz]) / self.dt,
                                                         np.array([position[0], position[1], position[2]], float)))
                    else:
                        self._forbid(ta, tb, normal_sum)
                elif 'terrain' in roles and 'robot' not in roles and 'payload' not in roles:
                    continue
                else:
                    self._forbid(ta, tb, normal_sum)
        except Exception as exc:                                      # noqa: BLE001
            # an exception inside a PhysX callback would be swallowed; the stepping loop raises it
            self.error = repr(exc)

    def _forbid(self, ta, tb, force):
        key = (ta['name'], tb['name'])
        self.forbidden[key] = max(self.forbidden.get(key, 0.), force)

    @property
    def impulse_on_first(self):
        """+1 when the reported impulse acts on the first collider of a pair, -1 otherwise, 0 while unknown."""
        return 0. if self._votes == 0. else float(np.sign(self._votes))

    def tool_wrenches(self):
        """World spatial wrenches ``[moment about the world origin; force]`` of contacts on clamp bodies."""
        sign = self.impulse_on_first
        out = {}
        if sign == 0.:
            if self.tool_points:
                self.unsigned_tool_steps += 1                         # left out of the wrist sensors: see positive_control
            return out
        for body, robot_first, force, position in self.tool_points:
            f = force * (sign if robot_first else -sign)
            out[body] = out.get(body, 0.) + np.r_[np.cross(position, f), f]
        return out

    def wheel_normals(self):
        """Normal load of every wheel during the last step: ``{(robot, wheel): N}`` (input of the tyre model)."""
        return {key: float(np.linalg.norm(normal)) for key, (normal, _) in self.wheels.items()}

    def result(self):
        if self.error:
            raise RuntimeError('contact reporting failed: ' + self.error)
        wheels = {}
        for key, (normal, friction) in self.wheels.items():
            wheels[key] = [float(np.linalg.norm(normal)), float(np.linalg.norm(friction)) if self.friction else float('nan')]
        return dict(pads=dict(self.pads), wheels=wheels, guides=float(self.guides), tool=float(self.tool),
                    forbidden=[(a, b, float(f)) for (a, b), f in self.forbidden.items()])

    def close(self):
        self.subscription = None
