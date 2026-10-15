"""Isaac Sim plant of a compiled team (state in compiled coordinates, efforts at the compiled ports).

This module contains no Isaac import: it drives a ``native`` object (the PhysX tensor
boundary of ``workflows/isaac/isaac_team/native.py``), a contact ``monitor`` and a
``stepper`` that advances PhysX by one step.  The offline tests drive the same class
through a stand-in for those three objects.

Physics is PhysX: articulations, rollers on the terrain, contacts.  These elements are
supplied by this adapter, as stated in ``backends/usd_export.LIMITATIONS``:

* the clamps are the compliant clamps of ``backends/latch.py``, applied as external
  loads evaluated at the start of each step;
* the wrist sensors are the Newton-Euler balance of the tool-side bodies over the
  step (measured motion before and after, clamp load and clamp contacts);
* the friction of conventional wheels is the tyre model of ``backends/tyre.py``,
  applied as an external force on every wheel.  What carries such a wheel depends on
  ``wheel_load``: ``'tyre'`` (the default of the workflow for these wheels) has no
  collider under the wheel, and the declared normal spring and damper of the tyre
  carry it, applied by this adapter together with the friction; ``'contact'`` stands
  the wheel on a PhysX contact and takes the normal load PhysX reported for the step
  before (``backends/usd_export.TYRE_LIMITATION``).
"""
from __future__ import annotations

import numpy as np

from teamc.se3 import adj, inv

from .latch import LatchSet, check_latch_stability
from .tyre import TyreSet, check_tyre_stability


class IsaacPlant:
    name = 'isaac'

    def __init__(self, comp, native, monitor, stepper, dt, latch_params=None, wheel_load='contact'):
        if wheel_load not in ('contact', 'tyre'):
            raise ValueError(f'wheel_load must be contact or tyre, not {wheel_load!r}')
        self.comp, self.native, self.monitor, self._advance = comp, native, monitor, stepper
        self.dt = float(dt)
        self.tree = comp.tree
        self.n = self.tree.n
        ports = comp.cmg['ports']
        self._index = np.array([p['index'] for p in ports])
        self._gear = np.array([p['gear'] for p in ports], float)
        self._bounds = np.array([p['bounds'] for p in ports], float)
        self.latches = LatchSet(comp, latch_params)
        self.latched = {cid: False for cid in self.latches.grasps}
        self.tyres = TyreSet(comp, carry=wheel_load == 'tyre') or None
        if wheel_load == 'tyre' and self.tyres is None:
            raise ValueError("wheel_load 'tyre' needs conventional wheels (a friction-driven base)")
        self.wheel_load = wheel_load
        self._tyre_normal, self._tyre_loads, self.tyre_ratio = {}, {}, 0.
        self._payload = self.tree.index[comp.payload['body']]
        self.time, self.steps, self.saturated, self.latch_ratio = 0., 0, 0, 0.
        self._q = self._v = None
        self._loads = {}
        self._wrench = {rid: np.zeros(6) for rid in self.latches.sensors_of}
        self.max_deflection = 0.
        self.max_tyre_depth = 0.
        self.cart_moved = None
        self._injected = None
        # PhysX's own reaction in the sensor joint, where the tensor API provides it: an independent reference
        # for the sensor balance (never used by the controller)
        joints = {j['id']: j for j in comp.cmg['joints']}
        self._reference = {}
        for rid, sensor in self.latches.sensors_of.items():
            r = comp.robots[rid]
            T_fj = np.asarray(joints[r.prefix + r.record['ports']['sensor']['joint_id']]['T_FJ'], float)
            X = inv(sensor['T']) @ T_fj                                # child joint frame in the sensor frame
            self._reference[rid] = dict(body=sensor['body'], map=adj(inv(X)).T, n=0, same=np.zeros(2), opposite=np.zeros(2),
                                        scale=np.zeros(2))

    @property
    def tyres_carry(self):
        return bool(self.tyres is not None and self.tyres.carry)

    # ------------------------------------------------------------------ state
    def reset(self, q, v=None):
        """Initialisation only: write a compiled configuration to PhysX and read the state back.

        With the tyres carrying the wheels, a base that stands on the terrain is lowered onto its tyres first
        (``TyreSet.stand``), so that a run does not start with a drop onto the springs.
        """
        q = np.asarray(q, float)
        if self.tyres_carry:
            q = self.tyres.stand(q)
        self.native.set_state(q, v)
        self.latches.reset()
        for cid in self.latched:
            self.latched[cid] = False
        self.time, self.steps = 0., 0
        self._loads = {}
        self._wrench = {rid: np.zeros(6) for rid in self._wrench}
        self._q, self._v = self.native.state()
        if self.tyres:
            self.tyres.reset()
            self._tyre_normal, self._tyre_loads = {}, {}
            self.tyre_ratio = check_tyre_stability(self.tyres, self._q, self.dt)

    def state(self):
        return self._q.copy(), self._v.copy()

    def measure(self):
        """What the cell measures: state, wrist sensors, time."""
        return dict(t=self.time, q=self._q.copy(), v=self._v.copy(), wrench={rid: w.copy() for rid, w in self._wrench.items()})

    # ------------------------------------------------------------------ inputs
    def set_efforts(self, f):
        """Actuator port efforts in compiled port order (clipped to the declared bounds)."""
        f = np.asarray(f, float)
        clipped = np.clip(f, self._bounds[:, 0], self._bounds[:, 1])
        self.saturated += int(np.any(clipped != f))
        tau = np.zeros(self.n)
        np.add.at(tau, self._index, self._gear * clipped)
        self.native.set_efforts(tau)
        return clipped

    def set_obstacle(self, position):
        self.cart_moved = self.native.set_cart(position)

    def step(self, n=1):
        tree = self.tree
        for _ in range(n):
            q, v = self._q, self._v
            kin = tree.sweep(q)
            T = kin[0]
            loads = self.latches.loads(q, v, kin)
            if self._injected:                                         # known loads of a check, treated like clamp loads
                loads.update(self._injected)
                self._injected = None
            origin = T[self._payload, :3, 3].copy()
            applied = loads
            if self.tyres:
                if self.tyres.carry:                                   # the tyres carry the wheels: normal load and friction
                    self._tyre_loads = self.tyres.carried_loads(q, v, self.dt, kin)
                    for load in self._tyre_loads.values():
                        self.max_tyre_depth = max(self.max_tyre_depth, load['depth'])
                else:                                                  # tyre friction, with the wheel loads of the step before
                    self._tyre_loads = self.tyres.loads(q, v, self._tyre_normal, self.dt, kin)
                applied = {**self._tyre_loads, **loads}                # a known load of a check replaces the tyre load of its wheel
            self.native.apply_loads(applied, origin + T[self._payload, :3, :3] @ tree.com[self._payload])
            self.monitor.begin_step()
            self._advance()
            if self.monitor.error:
                raise RuntimeError('contact reporting failed: ' + self.monitor.error)
            if self.tyres and not self.tyres.carry:
                self._tyre_normal = self.monitor.wheel_normals()
            q1, v1 = self.native.state()
            self._wrench = self.latches.sensors(q, v, (v1 - v) / self.dt, loads, self.monitor.tool_wrenches(), kin)
            self._loads = dict(loads, origin=origin)
            for load in loads.values():
                if 'deflection' in load:
                    self.max_deflection = max(self.max_deflection, load['deflection'])
            self._q, self._v = q1, v1
            self.steps += 1
            self.time = self.steps * self.dt
            for rid, ref in self._reference.items():
                w = self.native.incoming_joint_force(rid, ref['body'])
                if w is None:
                    break
                w = ref['map'] @ np.r_[w[3:], w[:3]]                   # [moment; force] in the sensor frame
                mine = self._wrench[rid]
                for key, other in (('same', w), ('opposite', -w)):
                    d = mine - other
                    ref[key] += (d[:3] @ d[:3], d[3:] @ d[3:])
                ref['scale'] += (mine[:3] @ mine[:3], mine[3:] @ mine[3:])
                ref['n'] += 1

    def inject(self, loads):
        """Known external loads for the next step only, in the form of ``LatchSet.loads`` (preflight checks)."""
        self._injected = dict(loads)

    def reset_sensor_reference(self):
        for ref in self._reference.values():
            ref.update(n=0, same=np.zeros(2), opposite=np.zeros(2), scale=np.zeros(2))

    # ------------------------------------------------------------------ grasp connections
    def latch(self, cid):
        """Close a clamp at the current relative pose; returns the as-latched tool frame in the payload body."""
        T2 = self.latches.close(cid, self._q)
        self.latched[cid] = True
        self.latch_ratio = max(self.latch_ratio, check_latch_stability(self.latches, self._q, self.dt))
        return T2

    def unlatch(self, cid):
        self.latches.open(cid)
        self.latched[cid] = False

    def weld_wrench(self, cid):
        """Wrench a closed clamp applied to the payload during the last step: force and moment about the payload origin (world)."""
        if cid not in self._loads:
            return np.zeros(3), np.zeros(3)
        return self.latches.payload_wrench(self._loads[cid], self._loads['origin'])

    # ------------------------------------------------------------------ measurements
    def sensor_wrench(self, rid):
        return self._wrench[rid].copy()

    def sensor_reference(self):
        """Sensor balance against PhysX's incoming joint force of the tool-side link, per robot.

        The sign convention of the native quantity is not assumed: both are evaluated and the closer one is
        reported with its sign.  Returns a note instead where the native quantity is not available.
        """
        note = getattr(self.native, 'joint_force_note', None)
        out = {}
        for rid, ref in self._reference.items():
            if not ref['n']:
                out[rid] = dict(available=False, note=note or 'no step taken')
                continue
            key = 'same' if ref['same'].sum() <= ref['opposite'].sum() else 'opposite'
            rms = np.sqrt(ref[key] / ref['n'])
            size = np.sqrt(ref['scale'] / ref['n'])
            out[rid] = dict(available=True, steps=ref['n'], native_sign=1 if key == 'same' else -1,
                            moment_rms_Nm=float(rms[0]), force_rms_N=float(rms[1]),
                            sensor_moment_rms_Nm=float(size[0]), sensor_force_rms_N=float(size[1]))
        return out

    def contacts(self):
        """Contact forces of the last step by category.  Wheels: ``{(robot, wheel): [normal load, tangential force]}``.

        Conventional wheels: the tangential force is the tyre load.  Where the tyres carry the wheels there is no
        wheel contact in PhysX: every tyre that carried load in the step is reported with the loads the adapter
        applied.
        """
        out = self.monitor.result()
        if self.tyres:
            if self.tyres.carry:
                out['wheels'] = {key[1:]: [float(load['load']), float(np.linalg.norm(load['tangential']))]
                                 for key, load in self._tyre_loads.items() if load['load'] > 0.}
            else:                                                      # frictionless colliders: the tangential force is the tyre load
                for key, load in self._tyre_loads.items():
                    if key[1:] in out['wheels']:                       # wheels PhysX reports on the terrain
                        out['wheels'][key[1:]][1] = float(np.linalg.norm(load['tangential']))
        return out
