"""Team controller of the beam mission, built on the compiled interfaces.

What comes from the compiled team graph:

* arm references: each robot's closure module gives its dependent arm joints (and
  their rates through the module tangent map) for the payload pose, its own base
  pose and its commanded coordinates (a lift, where the robot has one);
* base references: the compiled ground and rolling rows give the rates of the base
  chart, of its passive joints and of the wheels for a planar base twist; for a
  friction-driven base (skid steering) the wheel rates come from the compiled skid
  model at the present wheel loads, which also says about which point the base turns;
* efforts: inverse dynamics with *prescribed* grasp wrenches gives the actuator
  efforts and the remaining reactions; the grasp wrenches come from the declared
  load distribution of the payload wrench;
* force regulation: the wrist sensors give the grasp wrenches actually applied to
  the payload; their deviation from the prescribed wrenches in the directions the
  mode leaves free (the internal wrench while carrying, every direction while the
  payload rests on supports) drives a tool-frame admittance.

What is declared here and is not a compiler output: gains, durations, thresholds
and the order of the phases.

Assumptions about measurement: joint positions and rates, base poses, the payload
pose and the wrist force/torque sensors are measured.  The controller never reads
contact forces of the plant.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d

from teamc.chords import TeamAssembler
from teamc.interfaces import TeamModel
from teamc.loads import LoadSharing
from teamc.se3 import adj, exp, inv, log, pose_chart, transform
from teamc.stability import pad_loads

from .route import PlanarPath, SpeedZones, compose, relative
from .timing import Governor, blend


def arm_joints(robot):
    """Local ids of the joints a grasp closure solves (the arm), in tool-path order."""
    return list(robot.dependent)


def lift_joint(robot):
    """Local id of the commanded tool-path coordinate (the lift), or ``None`` for a robot without one."""
    commanded = [j for j in robot.tool_path if j in robot.commanded]
    if len(commanded) > 1:
        raise ValueError(f'{robot.id}: one commanded tool-path coordinate is supported')
    return commanded[0] if commanded else None


DEFAULTS = dict(
    arm_stiffness=[14000., 14000., 10000., 3000., 3000., 3000.],   # N m/rad
    arm_damping_ratio=1.0, lift_stiffness=2.5e5, lift_damping_ratio=1.0,
    wheel_gain=60.,                         # N m s/rad on the wheel speed error
    base_kp=[1.6, 1.6, 1.6], base_ki=[.4, .4, .4], base_integral_limit=.06,
    # a base that cannot move sideways (skid steering): along its heading (1/s), across it (1/m^2, scaled by the
    # speed), heading (1/s); integral gains along the heading and of the heading.  The heading has none: an error
    # across the path is removed by turning towards it, and an integral of the heading error would undo that turn
    track_gain=[2., 20., 3.], track_ki=[.5, 0.],
    # such a base corrects its heading by sliding its wheels; it does that while it moves, not at rest: the heading
    # loop fades out below this reference speed (m/s), a turn counting as this length per radian (m)
    track_rest=[.02, .30],
    wheel_integral=0.,                      # N m/rad on the integral of the wheel speed error (0: proportional only)
    wheel_integral_leak=.25,                # s; the wheel integrals of a friction-driven base decay while it rests
    admittance_force=1.5e-4,                # m/s per N of wrench error
    admittance_moment=1.5e-3,               # rad/s per N m
    admittance_limit=[.03, .08],            # m, rad
    wrench_filter_hz=6.,
    velocity_filter_hz=15.,                 # low-pass of the joint rates used in the damping term (0: off)
    internal_reference=None,                # optional internal wrench reference per grasp (tool axes)
)


def _translation(v):
    return transform(p=v)


class TeamController:
    def __init__(self, comp, mission, dt, slow_every=5, options=None):
        self.comp, self.mission = comp, mission
        self.model = TeamModel(comp)
        self.tree = comp.tree
        self.n = self.tree.n
        self.dt, self.slow_every, self.dt_slow = float(dt), int(slow_every), float(dt) * int(slow_every)
        self.opt = dict(DEFAULTS, **mission.get('controller', {}))
        self.opt.update(options or {})
        self.robots = list(comp.robots)
        self.grasp_of = {c['robot']: cid for cid, c in comp.connections.items() if c['kind'] == 'grasp'}
        self.asm = TeamAssembler(comp, comp.seed())
        self.payload = np.asarray(comp.payload['base'], int)
        self.payload_body = self.tree.index[comp.payload['body']]
        self.mass = float(mission['payload']['mass_kg'])
        self.weight = self.mass * float(np.linalg.norm(self.tree.gravity))
        port_of = {p['index']: k for k, p in enumerate(self.model.ports)}
        self.ix = {}
        for rid, r in comp.robots.items():
            arm = r.idx(arm_joints(r))
            name = lift_joint(r)
            lift = None if name is None else r.joint[name]
            wheels = np.array([w['index'] for w in r.wheels])
            joints = arm if lift is None else np.r_[arm, lift]
            joint_ports = np.array([port_of[int(i)] for i in joints])
            wheel_ports = np.array([port_of[int(i)] for i in wheels])
            tool_body, T_tool = r.frames[r.record['ports']['tool']['frame']]
            sensor = r.record['ports']['sensor']
            s_body, T_s = r.frames[sensor['frame']]
            body = next(b for b in r.record['bodies'] if r.prefix + b['id'] == s_body)
            self.ix[rid] = dict(arm=arm, lift=lift, wheels=wheels, joints=joints, joint_ports=joint_ports,
                                wheel_ports=wheel_ports, base=r.base, planar=r.base[[0, 1, 3]],
                                tool_body=self.tree.index[tool_body], T_tool=np.asarray(T_tool, float),
                                sensor_body=self.tree.index[s_body], T_sensor=np.asarray(T_s, float),
                                tool_mass=float(body['mass_kg']), tool_com=np.asarray(body['com_m'], float),
                                arm_base=(self.tree.index[r.frames['arm_base_link'][0]], r.frames['arm_base_link'][1]))
        self._gains()
        self.wheel_bounds = {rid: .5 * self.model.effort_bounds[ix['wheel_ports']] for rid, ix in self.ix.items()}
        ls = mission['load_sharing']
        self._ls_args = dict(shares=ls['shares'], moment_scale=ls['moment_scale_Nm'], force_scale=ls['force_scale_N'])
        self._ls = {}
        self._route()
        # ---------------------------------------------------------------- state
        self.mode = self.model.mode(grasps=(), rest=mission['rests']['pickup'])
        self.latched = []
        self.share = 0.
        self.payload_source = 'measured'
        self.delta = {cid: np.zeros(6) for cid in self.grasp_of.values()}
        self.lam_meas = {cid: np.zeros(6) for cid in self.grasp_of.values()}
        self.lam_raw = {cid: np.zeros(6) for cid in self.grasp_of.values()}
        self.lam_ref = {cid: np.zeros(6) for cid in self.grasp_of.values()}
        self.carried = 0.
        self.q_ref, self.v_ref = comp.seed(), np.zeros(self.n)
        self.f_ff = np.zeros(self.model.nf)
        self.wheel_ref = {rid: np.zeros(len(self.ix[rid]['wheels'])) for rid in self.robots}
        self.wheel_int = {rid: np.zeros(len(self.ix[rid]['wheels'])) for rid in self.robots}
        self.wheel_load = {rid: None for rid in self.robots}     # wheel loads of the last feedforward (friction-driven bases)
        self._skid_good, self._skid_noted = {}, {}               # the last wheel loads the skid model answered for
        self.base_moving = {rid: 1. for rid in self.robots}      # 0 while the reference of a friction-driven base rests
        self.base_cmd = {rid: np.zeros(3) for rid in self.robots}
        self.base_int = {rid: np.zeros(3) for rid in self.robots}
        self.base_ref = {rid: (np.zeros(3), np.zeros(3), np.zeros(3)) for rid in self.robots}
        self.lift_ref = {rid: (0., 0., 0.) for rid in self.robots}
        self.arm_cmd = {rid: ('joint', None, None) for rid in self.robots}
        self.pref, self.pref_v, self.pref_a = np.zeros(6), np.zeros(6), np.zeros(6)
        self.commands, self.events = [], []
        self.reactions, self.pad = {}, None
        self.t_slow, self.t = 0., 0.
        self._v_arm_prev = np.zeros(self.n)
        self._v_filt = None
        self.phases = self._phase_list()
        self.k, self.t0, self.first = 0, 0., True
        self.ctx = {}
        self.status = 'running'
        self.verification = {}

    # ------------------------------------------------------------------ setup
    def _gains(self):
        q = self.asm.solve(self.comp.seed())[0]
        M = np.diag(self.tree.mass(q))
        o = self.opt
        self.kp, self.kd = {}, {}
        for rid, ix in self.ix.items():
            n_lift = 0 if ix['lift'] is None else 1
            k = np.r_[o['arm_stiffness'], np.full(n_lift, o['lift_stiffness'])]
            # inertia seen by each joint, with the robot's share of the payload at a typical lever arm
            m_eff = M[ix['joints']] + np.r_[np.full(3, self.mass / len(self.robots) * .5), np.zeros(3),
                                            np.full(n_lift, self.mass / len(self.robots))]
            zeta = np.r_[np.full(6, o['arm_damping_ratio']), np.full(n_lift, o['lift_damping_ratio'])]
            self.kp[rid], self.kd[rid] = k, 2. * zeta * np.sqrt(k * m_eff)

    def _route(self):
        r = self.mission['route']
        self.path = PlanarPath(np.asarray(r['samples']), r['rho'])
        grid = np.arange(0., self.path.length + .025, .05)
        grid[-1] = self.path.length
        zones = SpeedZones(r['cruise_speed'], r['zones'])
        curv = np.array([np.linalg.norm(self.path(l)[2] * [1., 1., r['rho']]) for l in grid])
        limit = np.minimum([zones(l) for l in grid], np.sqrt(.8 * r['accel'] / np.maximum(curv, 1e-6)))
        limit = np.maximum(limit, .03)
        # base poses along the route: a fixed offset from the payload, or, for a team whose bases cannot move
        # sideways, the poses the mission designed for every base
        self.base_route = None
        if 'bases' in r:
            ls = self.path.sample_l                               # the base poses are given at the route samples
            if np.any(np.diff(ls) <= 0.):
                raise ValueError('base routes need route samples that advance along the route')
            self.base_route = {rid: CubicSpline(ls, np.asarray(r['bases'][rid], float)) for rid in self.robots}
            # a base that turns slides its wheels: the route speed is also bounded by the yaw rate of every base
            turn = np.max([np.abs(route(grid, 1)[:, 2]) for route in self.base_route.values()], axis=0)
            limit = np.minimum(limit, r['base_yaw_rate'] / np.maximum(turn, 1e-6))
        for k in range(len(grid) - 2, -1, -1):          # speeds from which every limit ahead can still be met
            limit[k] = min(limit[k], np.sqrt(limit[k + 1] ** 2 + 2. * .7 * r['accel'] * (grid[k + 1] - grid[k])))
        self.speed_limit = lambda l: float(np.interp(l, grid, limit))
        self.governor = Governor(self.path.length, self.speed_limit, accel=r['accel'], decel=r['accel'])
        # ground height under each base along the route, payload height and lift schedules
        h = {}
        for rid in self.robots:
            z = [self.model.ground_pose(rid, self.base_pose_at(rid, l))[1][0] - self.model.base_height[rid] for l in grid]
            h[rid] = np.asarray(z)
        top = gaussian_filter1d(np.max([h[rid] for rid in self.robots], axis=0), 4., mode='nearest')
        self.z_table = CubicSpline(grid, self.mission['heights']['z_carry'] + top)
        # a robot with a lift keeps its arm posture over uneven ground with it; without one the arm takes the difference
        self.lift_table = {rid: CubicSpline(grid, np.maximum(self.mission['lifts']['carry'][rid]
                                                             + gaussian_filter1d(top - h[rid], 4., mode='nearest'), 0.))
                           for rid in self.robots if self.ix[rid]['lift'] is not None}
        self.ground_end = float(top[-1])

    def base_pose_at(self, rid, l):
        """Planar pose of a base at route parameter ``l``."""
        if self.base_route is not None:
            return self.base_route[rid](float(np.clip(l, 0., self.path.length)))
        return compose(self.path(l)[0], self.mission['formation']['carry'][rid])

    def _phase_list(self):
        d = self.mission['durations']
        out = [('settle', d['settle'], None), ('approach', None, None), ('dock', None, None), ('pregrasp', d['pregrasp'], None),
               ('over', d['over'], None), ('descend', d['descend'], None), ('latch', d['latch'], None),
               ('take_weight', d['take_weight'], None), ('raise', d['raise_'], None), ('extract', d['extract'], None),
               ('stow', d['stow'], None), ('navigate', None, None), ('present', d['present'], None),
               ('insert', d['insert'], None), ('lower', None, None), ('transfer', d['transfer'], None),
               ('verify', d['verify'], None)]
        for rid in self.mission['release_order']:
            out += [('release', d['release'], rid), ('retract_up', d['retract'], rid), ('retract_back', d['retract'], rid),
                    ('fold', d['fold'], rid)]
        out += [('retreat', d['retreat'], None), ('done', None, None)]
        return out

    @property
    def phase(self):
        return self.phases[self.k][0]

    def _event(self, kind, **data):
        self.events.append(dict(t=self.t, phase=self.phase, event=kind, **data))

    def load(self):
        key = tuple(self.mode.grasps)
        if key not in self._ls:
            self._ls[key] = LoadSharing(self.model, key, **dict(self._ls_args, shares={g: self._ls_args['shares'][g] for g in key}))
        return self._ls[key]

    # ------------------------------------------------------------------ commands to the plant
    def on_latched(self, cid, T2):
        """The plant closed a clamp; register the as-latched grasp frame and add the connection to the mode."""
        self.model.register_grasp(cid, T2)
        self.asm.module(cid).set_grasp_frame(T2)
        self.latched.append(cid)
        self.delta[cid][:] = 0.
        self.mode = self.model.mode(grasps=sorted(self.latched), rest=self.mode.rest)
        self._event('latched', id=cid, T2=np.asarray(T2, float).tolist())

    def on_unlatched(self, cid):
        self.latched.remove(cid)
        self.delta[cid][:] = 0.
        self.mode = self.model.mode(grasps=sorted(self.latched), rest=self.mode.rest)
        self._event('unlatched', id=cid)

    # ------------------------------------------------------------------ measurement
    def start(self, meas):
        q = meas['q']
        self.q_ref = q.copy()
        self.t = self.t_slow = self.t0 = meas['t']
        for rid, ix in self.ix.items():
            self.base_ref[rid] = (q[ix['planar']].copy(), np.zeros(3), np.zeros(3))
            if ix['lift'] is not None:
                self.lift_ref[rid] = (float(q[ix['lift']]), 0., 0.)
            self.arm_cmd[rid] = ('joint', q[ix['arm']].copy(), np.zeros(6))
        self.pref = q[self.payload].copy()
        self._event('start')

    def start_carrying(self, meas, at=0.):
        """Start with the payload already grasped and carried at route parameter ``at`` (tests and partial runs)."""
        self.start(meas)
        self.k = [p[0] for p in self.phases].index('navigate')
        self.first = True
        self.share, self.payload_source = None, 'ref'
        self.mode = self.model.mode(grasps=sorted(self.latched), rest=None)
        self.governor.l = float(at)
        for rid in self.robots:
            self.arm_cmd[rid] = ('closure', None)
            if rid in self.lift_table:
                self.lift_ref[rid] = (float(self.lift_table[rid](at)), 0., 0.)
        self._set_payload(meas['q'][self.payload])

    def grasp_wrench(self, rid, w_sensor, q, kin):
        """Wrench applied to the payload through a grasp (tool axes) from the wrist sensor reading.

        The weight of the tool-side body is removed; its inertia force is neglected here.
        """
        return self.model.sensor_to_grasp(rid, w_sensor, q, kin)

    def _measure(self, meas, kin):
        alpha = 1. - np.exp(-2. * np.pi * self.opt['wrench_filter_hz'] * self.dt_slow)
        up = 0.
        for rid, cid in self.grasp_of.items():
            lam = self.grasp_wrench(rid, meas['wrench'][rid], meas['q'], kin)
            self.lam_raw[cid] = lam
            self.lam_meas[cid] += alpha * (lam - self.lam_meas[cid])
            if cid in self.latched:
                ix = self.ix[rid]
                R = (kin[0][ix['tool_body']] @ ix['T_tool'])[:3, :3]
                up += float((R @ self.lam_meas[cid][3:])[2])
        self.carried = up / self.weight

    # ------------------------------------------------------------------ references of the phases
    def _hold_bases(self):
        for rid in self.robots:
            p, _, _ = self.base_ref[rid]
            self.base_ref[rid] = (p, np.zeros(3), np.zeros(3))

    def _formation(self, pose, rate, acc, offsets):
        """Base references that keep planar offsets from a moving payload planar pose."""
        for rid in self.robots:
            o = np.asarray(offsets[rid], float)
            p = compose(pose, o)
            d = p[:2] - pose[:2]
            perp = np.array([-d[1], d[0]])
            v = np.r_[rate[:2] + rate[2] * perp, rate[2]]
            a = np.r_[acc[:2] + acc[2] * perp - rate[2] ** 2 * d, acc[2]]
            self.base_ref[rid] = (p, v, a)

    def _follow(self, ctx, paths, tau):
        """Base references along designed pose paths: every base runs its segments one after the other, each from
        rest to rest.  Returns True when every base has finished."""
        if 'segments' not in ctx:
            ctx['segments'] = {}
            for rid in self.robots:
                t0, out = 0., []
                for seg in paths[rid]:
                    path = PlanarPath(np.asarray(seg['samples'], float), seg['rho'])
                    T = max(1.875 * path.length / seg['speed'], 1.)
                    out.append((t0, T, path))
                    t0 += T + seg.get('pause', .3)
                ctx['segments'][rid] = out
            ctx['T'] = {rid: (segs[-1][0] + segs[-1][1]) if segs else 0. for rid, segs in ctx['segments'].items()}
        for rid, segs in ctx['segments'].items():
            if not segs:
                continue
            t0, T, path = next((x for x in segs if tau < x[0] + x[1]), segs[-1])
            w, dw, ddw = blend(tau - t0, T)
            p, dp, ddp = path(path.length * w)
            L = path.length
            self.base_ref[rid] = (p, dp * (L * dw), ddp * (L * dw) ** 2 + dp * (L * ddw))
        return tau >= max(ctx['T'].values()) + .5

    def _rest_pose(self, rest_id, yaw_near):
        """Payload chart pose at which its rest frame sits on the support frame of a rest connection."""
        c = self.comp.connections[rest_id]
        return pose_chart(np.asarray(c['T1']) @ inv(c['T2']), yaw_near=yaw_near)

    def _payload_planar(self, q):
        c = q[self.payload]
        return np.array([c[0], c[1], c[3]])

    def _set_payload(self, pose, rate=None, acc=None):
        self.pref = np.asarray(pose, float).copy()
        self.pref_v = np.zeros(6) if rate is None else np.asarray(rate, float)
        self.pref_a = np.zeros(6) if acc is None else np.asarray(acc, float)

    def _lifts(self, target, s, ds, dds, start):
        for rid in start:
            a, b = start[rid], target[rid]
            self.lift_ref[rid] = (a + (b - a) * s, (b - a) * ds, (b - a) * dds)

    def _lifted(self):
        """Robots that have a lift."""
        return [rid for rid in self.robots if self.ix[rid]['lift'] is not None]

    def _closure_target(self, rid, q, offset, lift):
        """Arm joints that put the tool on its (offset) grasp frame for the measured base and payload."""
        cid = self.grasp_of[rid]
        q_in = q.copy()
        if self.ix[rid]['lift'] is not None:
            q_in[self.ix[rid]['lift']] = lift
        q_out, _ = self.asm.solve(q_in, [cid], offsets={cid: offset})
        return q_out[self.ix[rid]['arm']]

    def _advance(self):
        self.k += 1
        self.t0 = self.t
        self.first = True
        self.ctx = {}
        self._event('phase', name=self.phase)

    def _phase(self, meas, kin):
        """References of the current phase; returns True when the phase is complete."""
        name, T, who = self.phases[self.k]
        tau = self.t - self.t0
        q, v = meas['q'], meas['v']
        m, ctx, first = self.mission, self.ctx, self.first
        self.first = False
        s, ds, dds = blend(tau, T) if T else (1., 0., 0.)
        station, carry = m['formation']['station'], m['formation']['carry']
        app = m['approach']

        if name == 'settle':
            return tau >= T

        if name in ('approach', 'dock') and 'base_paths' in m:
            # bases that cannot move sideways follow the pose paths the mission designed for them
            return self._follow(ctx, m['base_paths'][name], tau)

        if name in ('approach', 'dock'):
            # two legs: to a point beside the station, then a straight sideways move into it
            if first:
                rest = self._payload_planar(q)
                ctx['from'] = {rid: (q[self.ix[rid]['planar']] if name == 'approach' else self.base_ref[rid][0]).copy()
                               for rid in self.robots}
                back = m['dock_distance'] if name == 'approach' else 0.
                ctx['to'] = {rid: compose(rest, np.asarray(station[rid]) + [0., np.sign(station[rid][1]) * back, 0.])
                             for rid in self.robots}
                speed = m['approach_speed'] if name == 'approach' else m['dock_speed']
                ctx['T'] = {rid: max(1.875 * np.linalg.norm((ctx['to'][rid] - ctx['from'][rid]) * [1., 1., .9]) / speed, 2.)
                            for rid in self.robots}
            for rid in self.robots:
                a, b = ctx['from'][rid], ctx['to'][rid]
                p, dp, ddp = blend(tau, ctx['T'][rid])
                self.base_ref[rid] = (a + (b - a) * p, (b - a) * dp, (b - a) * ddp)
            return tau >= max(ctx['T'].values()) + .5

        if name == 'pregrasp':
            if first:
                self._hold_bases()
                ctx['lift0'] = {rid: float(q[self.ix[rid]['lift']]) for rid in self._lifted()}
                ctx['q0'] = {rid: q[self.ix[rid]['arm']].copy() for rid in self.robots}
                ctx['q1'] = {rid: self._closure_target(rid, q, _translation(app[rid]['retracted']), m['lifts']['station'].get(rid))
                             for rid in self.robots}
            self._lifts(m['lifts']['station'], s, ds, dds, ctx['lift0'])
            for rid in self.robots:
                a, b = ctx['q0'][rid], ctx['q1'][rid]
                self.arm_cmd[rid] = ('joint', a + (b - a) * s, (b - a) * ds)
            return tau >= T

        if name in ('over', 'descend'):
            for rid in self.robots:
                a = np.asarray(app[rid]['retracted' if name == 'over' else 'above'], float)
                b = np.asarray(app[rid]['above'], float) if name == 'over' else np.zeros(3)
                self.arm_cmd[rid] = ('closure', _translation(a + (b - a) * s))
            return tau >= T

        if name == 'latch':
            if first:
                self.share = 0.
                self._set_payload(q[self.payload])       # latched arms hold the payload where it rests
                self.payload_source = 'ref'
                for rid in self.robots:
                    self.arm_cmd[rid] = ('closure', None)
                    self.commands.append(('latch', self.grasp_of[rid]))
            return tau >= T and len(self.latched) == len(self.robots)

        if name == 'take_weight':
            self.share = s
            if tau >= T and self.carried >= m['thresholds']['carried_fraction_taken']:
                self._event('weight_taken', carried_fraction=self.carried, pads=self.pad)
                return True
            if tau >= T + 3.:
                # the wrist sensors do not show the weight in the clamps: do not lift; hold with the payload on its supports
                self.share = 0.
                self.status = 'held: weight not taken'
                self._event('weight_not_taken', carried_fraction=self.carried, pads=self.pad)
                self.k = len(self.phases) - 2
                return True
            return False

        if name == 'raise':
            if first:
                self.mode = self.model.mode(grasps=sorted(self.latched), rest=None)
                self.share = None
                ctx['p0'] = self.pref.copy()
                rest = self._rest_pose(m['rests']['pickup'], ctx['p0'][3])
                ctx['p1'] = rest + np.array([0., 0., m['heights']['raise_at_pickup'], 0., 0., 0.])
                self._event('mode', mode=self.mode.label())
            d = ctx['p1'] - ctx['p0']
            self._set_payload(ctx['p0'] + d * s, d * ds, d * dds)
            return tau >= T

        if name == 'extract':
            if first:
                ctx['p0'] = self.pref.copy()
                side = np.sign(np.mean([station[rid][1] for rid in self.robots]))
                step = compose([ctx['p0'][0], ctx['p0'][1], ctx['p0'][3]], [0., side * m['pull'], 0.])
                ctx['p1'] = ctx['p0'].copy()
                ctx['p1'][:2] = step[:2]
            d = ctx['p1'] - ctx['p0']
            self._set_payload(ctx['p0'] + d * s, d * ds, d * dds)
            return tau >= T

        if name == 'stow':
            if first:
                ctx['p0'] = self.pref.copy()
                ctx['p1'] = ctx['p0'].copy()
                start = self.path(0.)[0]
                ctx['p1'][[0, 1, 3]] = start
                ctx['p1'][2] = float(self.z_table(0.))
                ctx['lift0'] = {rid: self.lift_ref[rid][0] for rid in self._lifted()}
            d = ctx['p1'] - ctx['p0']
            self._set_payload(ctx['p0'] + d * s, d * ds, d * dds)
            self._lifts({rid: float(self.lift_table[rid](0.)) for rid in self._lifted()}, s, ds, dds, ctx['lift0'])
            return tau >= T

        if name == 'navigate':
            stops = []
            ob = meas.get('obstacle')
            for gate in m['route']['gates']:
                if ob is None:
                    continue
                w = gate['closed_while']
                x = ob['position'][w['axis']]
                if w['entered'] < x < w['cleared'] and self.governor.l <= gate['stop_l'] + .3:
                    stops.append(gate['stop_l'])
            was_waiting = self.governor.waiting
            l, rate, acc = self.governor.step(self.dt_slow, stops)
            if self.governor.waiting is not None and was_waiting is None:
                self._event('pause', l=l, reason='gate closed')
            if self.governor.waiting is None and was_waiting is not None:
                self._event('resume', l=l, stop_overrun_m=self.governor.overrun)
            p, dp, ddp = self.path(l)
            pose = np.array([p[0], p[1], float(self.z_table(l)), p[2], 0., 0.])
            dz, ddz = float(self.z_table(l, 1)), float(self.z_table(l, 2))
            vel = np.array([dp[0], dp[1], dz, dp[2], 0., 0.]) * rate
            accel = np.array([ddp[0], ddp[1], ddz, ddp[2], 0., 0.]) * rate ** 2 + np.array([dp[0], dp[1], dz, dp[2], 0., 0.]) * acc
            self._set_payload(pose, vel, accel)
            if self.base_route is None:
                self._formation(p, dp * rate, ddp * rate ** 2 + dp * acc, carry)
            else:
                for rid, route in self.base_route.items():
                    b, db, ddb = route(l), route(l, 1), route(l, 2)
                    self.base_ref[rid] = (b, db * rate, ddb * rate ** 2 + db * acc)
            for rid, tab in self.lift_table.items():
                self.lift_ref[rid] = (float(tab(l)), float(tab(l, 1)) * rate, float(tab(l, 2)) * rate ** 2 + float(tab(l, 1)) * acc)
            ctx['l'] = l
            return self.governor.done

        if name == 'present':
            if first:
                self._hold_bases()
                ctx['p0'] = self.pref.copy()
                ctx['p1'] = ctx['p0'].copy()
                place = self._rest_pose(m['rests']['place'], ctx['p0'][3])
                ctx['p1'][2] = place[2] + m['heights']['place_above']
                ctx['lift0'] = {rid: self.lift_ref[rid][0] for rid in self._lifted()}
            d = ctx['p1'] - ctx['p0']
            self._set_payload(ctx['p0'] + d * s, d * ds, d * dds)
            self._lifts(m['lifts']['station'], s, ds, dds, ctx['lift0'])
            return tau >= T

        if name == 'insert':
            if first:
                ctx['p0'] = self.pref.copy()
                place = self._rest_pose(m['rests']['place'], ctx['p0'][3])
                ctx['p1'] = place + np.array([0., 0., m['heights']['place_above'], 0., 0., 0.])
            d = ctx['p1'] - ctx['p0']
            self._set_payload(ctx['p0'] + d * s, d * ds, d * dds)
            return tau >= T

        if name == 'lower':
            if first:
                ctx['p0'] = self.pref.copy()
                ctx['z_end'] = self._rest_pose(m['rests']['place'], ctx['p0'][3])[2] - .006
                ctx['carried0'] = self.carried
            speed = m['lower_speed'] * min(tau / .5, 1.)
            pose = self.pref.copy()
            pose[2] = max(pose[2] - speed * self.dt_slow, ctx['z_end'])
            touched = self.carried < ctx['carried0'] - m['thresholds']['contact_drop_fraction']
            if touched or pose[2] <= ctx['z_end']:
                self._set_payload(self.pref)
                self._event('touchdown', carried_fraction=self.carried, by_force=bool(touched),
                            payload_z=float(q[self.payload][2]))
                return True
            self._set_payload(pose, np.array([0., 0., -speed, 0., 0., 0.]))
            return False

        if name == 'transfer':
            if first:
                self.mode = self.model.mode(grasps=sorted(self.latched), rest=m['rests']['place'])
                self._set_payload(q[self.payload])       # hold the payload where it touched down
                ctx['share0'] = float(np.clip(self.carried, 0., 1.))
                self._event('mode', mode=self.mode.label())
            self.share = ctx['share0'] * (1. - s)
            return tau >= T and (abs(self.carried) <= m['thresholds']['released_fraction'] or tau >= T + 3.)

        if name == 'verify':
            self.share = 0.
            if tau < T:
                return False
            th = m['thresholds']
            place = self.comp.connections[m['rests']['place']]
            T_p = kin[0][self.payload_body] @ place['T2']
            err = log(inv(place['T1']) @ T_p)
            pads = self.pad or {}
            checks = dict(
                grasps_unloaded=bool(abs(self.carried) <= th['released_fraction']),
                payload_at_rest=bool(np.linalg.norm(v[self.payload][:3]) <= th['settle_speed_m_s']),
                pads_loaded=bool(pads and min(pads['pad_normal_N'].values()) > .25 * self.weight),
                placed_within_tolerance=bool(np.linalg.norm(err[3:]) <= th['place_tolerance_m']
                                             and np.linalg.norm(err[:3]) <= th['place_tolerance_rad']))
            self.verification = dict(checks=checks, carried_fraction=self.carried, position_error_m=float(np.linalg.norm(err[3:])),
                                     rotation_error_rad=float(np.linalg.norm(err[:3])), error_in_support_frame=err.tolist(),
                                     pad_normal_N=pads.get('pad_normal_N'), accepted=all(checks.values()))
            self._event('verification', **self.verification)
            if not self.verification['accepted']:
                self.status = 'held: supports not verified'
                self.k = len(self.phases) - 2          # skip release; hold in the final phase
            ctx['pose'] = q[self.payload].copy()
            return True

        if name == 'release':
            cid = self.grasp_of[who]
            if first:
                ctx['pose'] = q[self.payload].copy()
                self.commands.append(('unlatch', cid))
            if tau < T or cid in self.latched:
                return False
            moved = float(np.linalg.norm((q[self.payload] - ctx['pose'])[:3]))
            self._event('released', id=cid, payload_moved_m=moved, still=bool(moved < 1e-3))
            return True

        if name in ('retract_up', 'retract_back'):
            a = np.zeros(3) if name == 'retract_up' else np.asarray(app[who]['above'], float)
            b = np.asarray(app[who]['above' if name == 'retract_up' else 'retracted'], float)
            self.arm_cmd[who] = ('closure', _translation(a + (b - a) * s))
            return tau >= T

        if name == 'fold':
            if first:
                ctx['q0'] = q[self.ix[who]['arm']].copy()
                ctx['q1'] = np.array([m['travel_posture'][who][j] for j in arm_joints(self.comp.robots[who])])
                ctx['lift0'] = self.lift_ref[who][0]
            a, b = ctx['q0'], ctx['q1']
            self.arm_cmd[who] = ('joint', a + (b - a) * s, (b - a) * ds)
            if self.ix[who]['lift'] is not None:
                l0, l1 = ctx['lift0'], m['lifts']['carry'][who]
                self.lift_ref[who] = (l0 + (l1 - l0) * s, (l1 - l0) * ds, (l1 - l0) * dds)
            return tau >= T

        if name == 'retreat' and 'base_paths' in m:
            if first:
                # straight back from where every base stands, along its own heading
                back = m['base_paths']['retreat']
                ctx['paths'] = {}
                for rid in self.robots:
                    a = self.base_ref[rid][0]
                    b = a - back['distance'] * np.array([np.cos(a[2]), np.sin(a[2]), 0.])
                    ctx['paths'][rid] = [dict(samples=[a.tolist(), b.tolist()], rho=back['rho'], speed=back['speed'])]
            self._follow(ctx, ctx['paths'], tau)
            return tau >= T

        if name == 'retreat':
            if first:
                rest = self._payload_planar(q)
                ctx['from'] = {rid: self.base_ref[rid][0].copy() for rid in self.robots}
                ctx['to'] = {}
                for rid in self.robots:
                    o = np.asarray(station[rid], float)
                    ctx['to'][rid] = compose(rest, o + np.array([0., np.sign(o[1]) * .5, 0.]))
            for rid in self.robots:
                a, b = ctx['from'][rid], ctx['to'][rid]
                self.base_ref[rid] = (a + (b - a) * s, (b - a) * ds, (b - a) * dds)
            return tau >= T

        if name == 'done':
            if first:
                self._hold_bases()
                if self.status == 'running':
                    self.status = 'complete'
            return False
        raise RuntimeError(name)

    # ------------------------------------------------------------------ slow loop
    def slow(self, meas):
        """Update references, feedforward efforts and the force regulation from a measurement."""
        self.t = meas['t']
        q = meas['q']
        kin = self.tree.sweep(q)
        self._measure(meas, kin)
        while self._phase(meas, kin):
            self._advance()
        self._references(meas, kin)
        self._feedforward(meas, kin)
        self._admittance()
        self.t_slow = self.t

    def _references(self, meas, kin):
        q, v = meas['q'], meas['v']
        o = self.opt
        v_ref = np.zeros(self.n)
        a_ref = np.zeros(self.n)
        if self.payload_source == 'ref' and self.mode.rest is None:
            v_ref[self.payload], a_ref[self.payload] = self.pref_v, self.pref_a
        for rid in self.robots:
            ix = self.ix[rid]
            # ---------------- base: planar pose loop -> twist command -> wheel speeds from the compiled rolling rows
            p_ref, u_ref, du_ref = self.base_ref[rid]
            e = p_ref - q[ix['planar']]
            if self.model.friction[rid]:
                u_cmd, u_ref, du_ref, loads = self._track(rid, q, kin, p_ref, u_ref, du_ref, e)
                moving, S3, W3 = self.model.base_motion(q, rid, kin, loads)
            else:
                self.base_int[rid] = np.clip(self.base_int[rid] + np.asarray(o['base_ki']) * e * self.dt_slow,
                                             -o['base_integral_limit'], o['base_integral_limit'])
                u_cmd = u_ref + np.asarray(o['base_kp']) * e + self.base_int[rid]
                moving, S3, W3 = self.model.base_motion(q, rid, kin)  # base chart (and passive joints), wheels
            self.base_cmd[rid] = u_cmd
            self.wheel_ref[rid] = W3 @ u_cmd
            v_ref[moving], a_ref[moving] = S3 @ u_ref, S3 @ du_ref
            v_ref[ix['wheels']], a_ref[ix['wheels']] = W3 @ u_ref, W3 @ du_ref
            # ---------------- lift (commanded coordinate)
            lift_rate = np.zeros(0)
            if ix['lift'] is not None:
                l_ref, dl_ref, ddl_ref = self.lift_ref[rid]
                self.q_ref[ix['lift']], v_ref[ix['lift']], a_ref[ix['lift']] = l_ref, dl_ref, ddl_ref
                lift_rate = np.array([dl_ref])
            # ---------------- arm: joint target or closure module
            cmd = self.arm_cmd[rid]
            if cmd[0] == 'joint':
                self.q_ref[ix['arm']], v_ref[ix['arm']] = cmd[1], cmd[2]
            else:
                cid = self.grasp_of[rid]
                offset = cmd[1]
                if cid in self.latched:
                    offset = exp(self.delta[cid])
                q_in = q.copy()
                rates = np.zeros(6)
                if self.payload_source == 'ref' and cid in self.latched:
                    q_in[self.payload] = self.pref
                    rates = self.pref_v
                q_out, _ = self.asm.solve(q_in, [cid], offsets={cid: offset})
                self.q_ref[ix['arm']] = q_out[ix['arm']]
                mod = self.asm.modules[cid]
                N = mod['N'][mod['g'].passive[:6]]
                v_ref[ix['arm']] = N @ np.r_[rates, v_ref[ix['base']], lift_rate]
        a_ref_arm = np.clip((v_ref - self._v_arm_prev) / self.dt_slow, -3., 3.)
        for rid in self.robots:
            a_ref[self.ix[rid]['arm']] = a_ref_arm[self.ix[rid]['arm']]
        self._v_arm_prev = v_ref.copy()
        self.v_ref, self.a_ref = v_ref, a_ref

    def _track(self, rid, q, kin, p_ref, u_ref, du_ref, e):
        """Twist command of a base that cannot move sideways, and its reference as a twist it can follow.

        The base drives along its heading and turns about the point the skid model gives (``x0`` ahead of
        its origin on its centre line): the velocity of its origin across the heading is ``-x0`` times the
        yaw rate and cannot be commanded.  The error along the heading acts on the speed; the error across
        it, taken at the turning point and scaled by the speed, and the heading error act on the yaw rate.
        A base turns by sliding its wheels, so the yaw-rate terms act only while the reference moves: a base
        at rest keeps the heading it stopped with (its arm takes up the difference).
        Returns the commanded rates, the reference rates and accelerations in world axes, and the wheel loads
        used.
        """
        o = self.opt
        loads = self._skid_loads(rid, q, kin)
        x0 = float(self.model.skid(q, rid, kin, loads)['centre'][0])
        yaw = q[self.ix[rid]['planar'][2]]
        c, s_ = np.cos(yaw), np.sin(yaw)
        ex, ey = c * e[0] + s_ * e[1], -s_ * e[0] + c * e[1]
        cr, sr = np.cos(p_ref[2]), np.sin(p_ref[2])
        v_r, w_r = cr * u_ref[0] + sr * u_ref[1], u_ref[2]
        a_r, dw_r = cr * du_ref[0] + sr * du_ref[1], du_ref[2]
        kx, ky, kw = o['track_gain']
        limit = o['base_integral_limit']
        rest_speed, rest_length = o['track_rest']
        moving = min((abs(v_r) + rest_length * abs(w_r)) / rest_speed, 1.)
        self.base_moving[rid] = moving
        self.base_int[rid][[0, 2]] = np.clip(self.base_int[rid][[0, 2]]
                                             + np.asarray(o['track_ki']) * [ex, moving * e[2]] * self.dt_slow, -limit, limit)
        v_c = v_r * np.cos(e[2]) + kx * ex + self.base_int[rid][0]
        w_c = w_r + moving * (kw * e[2] + ky * v_r * (ey + x0 * np.sin(e[2])) + self.base_int[rid][2])

        def world(v, w):
            return np.array([c * v + s_ * x0 * w, s_ * v - c * x0 * w, w])

        return world(v_c, w_c), world(v_r, w_r), world(a_r, dw_r), loads

    def _skid_loads(self, rid, q, kin):
        """Wheel loads the skid model of a friction-driven base is evaluated with.

        Those of the last feedforward.  Where they leave the skid model without an answer (all the load on one side
        of the base or on one wheel, which a transient of the feedforward can give for one update), the last loads
        it answered for are kept, so that one such update does not end the mission; the first time this happens to
        a robot it is an event of the record.  ``None``: the base standing alone (the default of the model).
        """
        loads = self.wheel_load[rid]
        if loads is None:
            return None
        if np.all(np.isfinite(loads)):
            try:
                self.model.skid(q, rid, kin, loads)
                self._skid_good[rid] = loads
                return loads
            except (ValueError, np.linalg.LinAlgError):
                pass
        if not self._skid_noted.get(rid):
            self._skid_noted[rid] = True
            self._event('skid_loads_kept', robot=rid, loads=np.asarray(loads, float).tolist())
        return self._skid_good.get(rid)

    def _feedforward(self, meas, kin):
        q = meas['q']
        con = self.model.constraints(q, self.mode, kin)
        effort = self.tree.rnea(q, self.v_ref, self.a_ref, kinematics=kin)
        given = {}
        if self.mode.grasps:
            ls = self.load()
            G = ls.grasp_matrix(con)
            w = effort[self.payload]
            if self.mode.rest is not None:                       # payload on its supports: the robots carry a share
                w = self.tree.rnea(q, np.zeros(self.n), np.zeros(self.n), kinematics=kin)[self.payload] * (self.share or 0.)
            internal = self.opt['internal_reference']
            given = ls.distribute(G, w, None if internal is None else np.concatenate([internal[g] for g in ls.grasps]))
            self._G, self._ls_now = G, ls
        for cid in self.lam_ref:
            self.lam_ref[cid] = given.get(cid, np.zeros(6))
        out = self.model.inverse(q, self.v_ref, self.a_ref, self.mode, grasp_wrenches=given, kin=kin, constraints=con,
                                 effort=effort)
        self.f_ff = out['f']
        self.reactions = out['ports']
        self.ff_residual = out['residual']
        for rid in self.robots:                                  # wheel loads for the skid model of the next update
            if self.model.friction[rid]:
                self.wheel_load[rid] = self.model.wheel_loads(q, rid, out['ports'][f'{rid}:ground'], kin,
                                                              con.info.get(f'{rid}:normal'))
        self.pad = None
        if self.mode.rest is not None:
            # support loads implied by the *measured* grasp wrenches: the rest reaction balances what the robots do not carry
            rows = con.rows(self.mode.rest)
            need = self.tree.rnea(q, np.zeros(self.n), np.zeros(self.n), kinematics=kin)[self.payload]
            for g in self.mode.grasps:
                need = need - con.J[con.rows(g)][:, self.payload].T @ self.lam_meas[g]
            lam_rest = np.linalg.solve(con.J[rows][:, self.payload].T, need)
            self.pad = pad_loads(self.comp.connections[self.mode.rest], lam_rest)

    def _admittance(self):
        """Tool-frame admittance on the wrench error in the directions the mode leaves free."""
        if not self.mode.grasps:
            return
        o = self.opt
        gain = np.r_[np.full(3, o['admittance_moment']), np.full(3, o['admittance_force'])]
        limit = np.r_[np.full(3, o['admittance_limit'][1]), np.full(3, o['admittance_limit'][0])]
        ls = self._ls_now
        lam = np.concatenate([self.lam_meas[g] for g in ls.grasps])
        ref = np.concatenate([self.lam_ref[g] for g in ls.grasps])
        err = ref - lam
        if self.mode.rest is None:                                # carrying: only the internal part is regulated
            err = ls.internal_projector(self._G) @ err
        for k, g in enumerate(ls.grasps):
            self.delta[g] = np.clip(self.delta[g] + gain * err[6 * k:6 * k + 6] * self.dt_slow, -limit, limit)

    # ------------------------------------------------------------------ fast loop
    def fast(self, meas):
        """Actuator port efforts: compiled feedforward plus joint and wheel-speed feedback."""
        q, v = meas['q'], meas['v']
        lead = meas['t'] - self.t_slow
        f = self.f_ff.copy()
        fc = self.opt['velocity_filter_hz']
        if fc > 0.:
            if self._v_filt is None:
                self._v_filt = v.copy()
            self._v_filt += (1. - np.exp(-2. * np.pi * fc * self.dt)) * (v - self._v_filt)
            v = self._v_filt
        for rid, ix in self.ix.items():
            j = ix['joints']
            f[ix['joint_ports']] += self.kp[rid] * (self.q_ref[j] + self.v_ref[j] * lead - q[j]) + self.kd[rid] * (self.v_ref[j] - v[j])
            err = self.wheel_ref[rid] - v[ix['wheels']]
            f[ix['wheel_ports']] += self.opt['wheel_gain'] * err
            if self.opt['wheel_integral']:
                self.wheel_int[rid] = np.clip(self.wheel_int[rid] + self.opt['wheel_integral'] * err * self.dt,
                                              self.wheel_bounds[rid][:, 0], self.wheel_bounds[rid][:, 1])
                if self.base_moving[rid] < 1.:                 # at rest the wheels let go of what they wound up
                    self.wheel_int[rid] *= np.exp(-self.dt * (1. - self.base_moving[rid]) / self.opt['wheel_integral_leak'])
                f[ix['wheel_ports']] += self.wheel_int[rid]
        return f

    # ------------------------------------------------------------------ record
    def snapshot(self):
        """Controller quantities for the recorder."""
        out = dict(phase=self.k, share=np.nan if self.share is None else self.share, carried=self.carried,
                   payload_ref=self.pref.copy(), l=self.governor.l, rate=self.governor.rate,
                   mode=self.mode.label(), mode_grasps=','.join(self.mode.grasps), mode_rest=self.mode.rest or '',
                   ff_residual=getattr(self, 'ff_residual', 0.), waiting=float(self.governor.waiting is not None))
        for cid in self.lam_meas:
            out[f'lam_meas:{cid}'] = self.lam_meas[cid].copy()
            out[f'lam_raw:{cid}'] = self.lam_raw[cid].copy()
            out[f'lam_ref:{cid}'] = self.lam_ref[cid].copy()
            out[f'delta:{cid}'] = self.delta[cid].copy()
        for rid in self.robots:
            out[f'base_ref:{rid}'] = self.base_ref[rid][0].copy()
            if self.model.friction[rid]:
                out[f'base_cmd:{rid}'] = self.base_cmd[rid].copy()
                out[f'wheel_ref:{rid}'] = self.wheel_ref[rid].copy()
        return out
