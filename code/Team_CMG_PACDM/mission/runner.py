"""Mission loop shared by every plant, with the scenario (cart) and the recorder.

A plant provides: ``dt``, ``reset(q)``, ``measure()``, ``set_efforts(f)``,
``step()``, ``latch(id)`` returning the as-latched grasp frame, ``unlatch(id)``,
``set_obstacle(position)`` and, for the audit only, ``contacts()`` and
``weld_wrench(id)``.
"""
from __future__ import annotations

import time

import numpy as np

from teamc.se3 import inv, pose_chart

from .controller import TeamController, arm_joints, lift_joint


def start_configuration(comp, mission, model):
    """Compiled configuration at the start: bases at their start poses, arms folded, payload on its pickup supports."""
    q = comp.seed()
    for rid, r in comp.robots.items():
        x, y, yaw = mission['starts'][rid]
        idx, value = model.ground_pose(rid, [x, y, yaw])
        q[r.base[[0, 1, 3]]] = [x, y, yaw]
        q[idx] = value
        for j in arm_joints(r):
            q[r.joint[j]] = mission['travel_posture'][rid][j]
        if lift_joint(r) is not None:
            q[r.joint[lift_joint(r)]] = mission['lifts']['carry'][rid]
        for w in r.wheels:
            q[w['index']] = 0.
    c = comp.connections[mission['rests']['pickup']]
    q[comp.payload['base']] = pose_chart(np.asarray(c['T1']) @ inv(c['T2']))
    return q


class Scenario:
    """The crossing cart: it starts when the payload passes its trigger line and moves at a declared speed."""

    def __init__(self, comp):
        self.ob = comp.scene.get('obstacle')
        self.comp = comp
        self.started = None
        self.position = None if not self.ob else np.asarray(self.ob['start'], float).copy()
        self.velocity = np.zeros(3)
        self.speed = 0.
        self.half = float(comp.payload['record']['geometry'][0]['half_size'][0]) if comp.payload['record'].get('geometry') else 0.

    def update(self, t, dt, payload_chart):
        if not self.ob:
            return None
        tr = self.ob['trigger']
        axis = 'xyz'.index(tr['axis'])
        direction = np.array([np.cos(payload_chart[3]), np.sin(payload_chart[3]), 0.])
        lead = payload_chart[axis] + self.half * abs(direction[axis])
        if self.started is None and lead >= tr['value']:
            self.started = t
        if self.started is not None:
            end = np.asarray(self.ob['end'], float)
            gap = end - self.position
            dist = float(np.linalg.norm(gap))
            if dist > 1e-6:
                a = self.ob.get('acceleration_m_s2', .5)
                target = min(self.ob['speed_m_s'], np.sqrt(2. * a * dist))
                self.speed = min(self.speed + a * dt, target)
                step = min(self.speed * dt, dist)
                self.position = self.position + gap / dist * step
                self.velocity = gap / dist * self.speed
            else:
                self.speed, self.velocity = 0., np.zeros(3)
        return dict(position=self.position.copy(), velocity=self.velocity.copy())


class Recorder:
    """Record rows at a fixed interval; step quantities are averaged over the interval.

    A row holds the state at the start of its interval, the mean acceleration over it
    (velocity difference), the mean applied efforts, the mean wrist-sensor readings and,
    for the audit, the mean contact and latch forces reported by the plant.  The mean
    generalized reaction of the interval follows from these by one inverse-dynamics
    evaluation, so reactions and margins are those of the interval, not of one step.
    """

    def __init__(self, every):
        self.every = every
        self.rows = []
        self.contacts = []
        self._row = None

    def open(self, **row):
        self._row = row
        self._n = 0
        self._sum = {}
        self._peak = None
        self._wheels, self._pads, self._guides, self._tool, self._forbidden = {}, {}, 0., 0., {}

    def step(self, applied, command, wrenches, contacts=None, extra=None):
        self._n += 1
        acc = self._sum
        acc['f'] = acc.get('f', 0.) + applied
        self._peak = np.abs(command) if self._peak is None else np.maximum(self._peak, np.abs(command))
        for rid, w in wrenches.items():
            acc[f'sensor:{rid}'] = acc.get(f'sensor:{rid}', 0.) + w
        for key, value in (extra or {}).items():
            acc[key] = acc.get(key, 0.) + value
        if contacts is not None:
            for key, value in contacts['wheels'].items():
                self._wheels[key] = self._wheels.get(key, 0.) + np.asarray(value)
            for key, value in contacts['pads'].items():
                self._pads[key] = self._pads.get(key, 0.) + value
            self._guides += contacts['guides']
            self._tool += contacts['tool']
            for a, b, force in contacts['forbidden']:
                self._forbidden[(a, b)] = max(self._forbidden.get((a, b), 0.), force)

    def close(self, v_end, dt, with_contacts):
        row, n = self._row, max(self._n, 1)
        row['a'] = (v_end - row['v']) / (n * dt)
        for key, value in self._sum.items():
            row[key] = value / n
        row['f_peak'] = self._peak
        self.rows.append(row)
        if with_contacts:
            self.contacts.append(dict(t=row['t'], wheels={k: (v / n).tolist() for k, v in self._wheels.items()},
                                      pads={k: v / n for k, v in self._pads.items()}, guides=self._guides / n,
                                      tool=self._tool / n, forbidden=[(a, b, f) for (a, b), f in self._forbidden.items()]))
        self._row = None

    def arrays(self):
        out = {}
        for key in self.rows[0]:
            vals = [r[key] for r in self.rows]
            if isinstance(vals[0], str):
                out[key] = np.array(vals)
            else:
                out[key] = np.asarray(vals, float)
        return out


def carry_configuration(comp, mission, ctrl, at=0.):
    """Compiled configuration of the team carrying the payload at route parameter ``at``.

    The arms start from the travel posture, which is the posture at the start of the route.  Where every base
    keeps its place and heading on the payload that posture is a good seed anywhere.  Bases that turn under
    the payload along the route (``route.bases`` of the mission) leave it far behind, so for them the closure
    is followed along the route from its start.
    """
    q = comp.seed()
    for rid, r in comp.robots.items():
        for j in arm_joints(r):
            q[r.joint[j]] = mission['travel_posture'][rid][j]
    stations = [at] if ctrl.base_route is None else np.linspace(0., at, int(np.ceil(at / .2)) + 1)
    for l in stations:
        p = ctrl.path(l)[0]
        q[comp.payload['base']] = [p[0], p[1], float(ctrl.z_table(l)), p[2], 0., 0.]
        for rid, r in comp.robots.items():
            b = ctrl.base_pose_at(rid, l)
            idx, value = ctrl.model.ground_pose(rid, b)
            q[r.base[[0, 1, 3]]] = b
            q[idx] = value
            if rid in ctrl.lift_table:
                q[r.joint[lift_joint(r)]] = float(ctrl.lift_table[rid](l))
        q, _ = ctrl.asm.solve(q)
    return q


def run_mission(plant, comp, mission, options=None, record_hz=50., max_time=400., progress=None, on_record=None):
    """Run the mission on a plant.  Returns ``(controller, record arrays, info)``.

    Options: ``slow_every`` (physics steps per reference update), ``stop_after`` (phase name),
    ``start`` (``'navigate'`` starts with the payload already carried, at route parameter ``start_l``),
    ``duration`` (stop after this simulated time), ``controller`` (controller options).
    """
    options = dict(options or {})
    slow_every = int(options.pop('slow_every', 5))
    stop_phase = options.pop('stop_after', None)
    start = options.pop('start', None)
    duration = options.pop('duration', None)
    ctrl = TeamController(comp, mission, plant.dt, slow_every=slow_every, options=options.pop('controller', None))
    scenario = Scenario(comp)
    rec = Recorder(max(int(round(1. / (record_hz * plant.dt))), 1))
    if start == 'navigate':
        at = float(options.pop('start_l', 0.))
        q0 = carry_configuration(comp, mission, ctrl, at)
        plant.reset(q0)
        for cid in ctrl.grasp_of.values():
            ctrl.on_latched(cid, plant.latch(cid))
        meas = plant.measure()
        meas['obstacle'] = None
        ctrl.start_carrying(meas, at)
    else:
        q0 = start_configuration(comp, mission, ctrl.model)
        plant.reset(q0)
        meas = plant.measure()
        meas['obstacle'] = None
        ctrl.start(meas)
    if scenario.ob:
        plant.set_obstacle(scenario.position)
    wall = dict(control=0., plant=0., start=time.perf_counter())
    k = 0
    has_contacts = hasattr(plant, 'contacts')
    has_weld = hasattr(plant, 'weld_wrench')
    while True:
        t = meas['t']
        tc = time.perf_counter()
        if k % slow_every == 0:
            meas['obstacle'] = scenario.update(t, plant.dt * slow_every, meas['q'][comp.payload['base']])
            if scenario.ob:
                plant.set_obstacle(scenario.position)
            ctrl.slow(meas)
            for kind, cid in ctrl.commands:
                if kind == 'latch':
                    ctrl.on_latched(cid, plant.latch(cid))
                else:
                    plant.unlatch(cid)
                    ctrl.on_unlatched(cid)
            ctrl.commands.clear()
        f = ctrl.fast(meas)
        applied = np.asarray(plant.set_efforts(f), float)
        wall['control'] += time.perf_counter() - tc
        if k % rec.every == 0:
            if rec._row is not None:
                rec.close(meas['v'], plant.dt, has_contacts)
                if on_record is not None:
                    on_record(t, plant, ctrl)
            row = dict(t=t, q=meas['q'], v=meas['v'], **ctrl.snapshot())
            if scenario.ob:
                row['obstacle'] = scenario.position.copy()
            rec.open(**row)
        tp = time.perf_counter()
        plant.step()
        wall['plant'] += time.perf_counter() - tp
        k += 1
        meas = plant.measure()
        meas['obstacle'] = None
        # quantities of the step just taken (they belong to the interval of the open row)
        extra = {'v_mean': meas['v']}                  # rates after every step: their mean gives the displacement of the interval
        if has_weld:
            for cid in ctrl.grasp_of.values():
                fw, mw = plant.weld_wrench(cid) if cid in ctrl.latched else (np.zeros(3), np.zeros(3))
                extra[f'weld:{cid}'] = np.r_[mw, fw]
        rec.step(applied, f, meas['wrench'], plant.contacts() if has_contacts else None, extra)
        if progress is not None and k % int(round(5. / plant.dt)) == 0:
            progress(meas['t'], ctrl)
        if ctrl.phase == 'done' and meas['t'] - ctrl.t0 > 1.0:
            break
        if stop_phase is not None and ctrl.phase == stop_phase:
            break
        if duration is not None and meas['t'] >= duration:
            break
        if meas['t'] > max_time:
            ctrl.status = 'timeout'
            break
    wall['total'] = time.perf_counter() - wall.pop('start')
    info = dict(status=ctrl.status, sim_time=float(meas['t']), steps=k, wall=wall, events=ctrl.events,
                verification=ctrl.verification, saturated_steps=int(getattr(plant, 'saturated', 0)),
                phases=[p[0] + (f':{p[2]}' if p[2] else '') for p in ctrl.phases], q_start=q0, team=mission.get('team'),
                # the loops as they ran (options resolved): what a report of their quantities is about
                controller=dict(feedforward=ctrl.opt['feedforward'], feedforward_rate=ctrl.opt['feedforward_rate'],
                                advance=ctrl.opt['advance'], velocity_filter_hz=float(ctrl.opt['velocity_filter_hz']),
                                reference_update_s=ctrl.dt_slow, loop_step_s=ctrl.dt))
    return ctrl, rec, info
