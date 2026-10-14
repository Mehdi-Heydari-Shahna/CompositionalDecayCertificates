"""Grasp moments at every plant step of a complete MuJoCo mission (Section 7.6).

Runs the mission of TEAM with the controller KIND on the MuJoCo plant with the settings of the package's mission
workflow (2 ms steps, references renewed every 10 ms) and audits it as the workflow does. In addition, at every plant
step in which a grasp is latched, the wrench that the robot applies to the payload is inferred from its wrist sensor
with the tool-side inertia removed (TeamModel.sensor_to_grasp of the package, with the sensor reading of the step, the
configuration and rates at its start, and the acceleration of the step). Reported per grasp: the largest per-step
moment, the largest moment of the 20 ms vector means of the same series (the record intervals of the workflow), and
the audit's value (interval means of the sensor with an interval-level inertia correction), which missions.json of the
numerical-checks folder records for the same mission.

KIND (as in missions.py of the numerical-checks folder):
    fitted05   F/2: fitted feedforward, arm and lift damping ratios 0.5, no rate filter (default)
    fitted     F:   fitted feedforward, no rate filter
    desired0   D:   desired-motion feedforward, no rate filter
    desired    the package's default controller: desired-motion feedforward with the 15 Hz rate filter (Husky teams)

Usage: python3 run_native_loading.py TEAM [KIND]
Output: native_loading_<team>_<kind>.json, and the workflow's report folder native_loading_<team>_<kind>/
"""
from __future__ import annotations

import os

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('MUJOCO_GL', 'disable')

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from teamdata import PKG, RESULTS  # noqa: E402,F401  (puts the package on sys.path)

KINDS = {
    'fitted05': dict(feedforward='fitted', monitor=True, arm_damping_ratio=.5, lift_damping_ratio=.5),
    'fitted': dict(feedforward='fitted', monitor=True),
    'desired0': dict(feedforward='desired', monitor=True, velocity_filter_hz=0.),
    'desired': dict(),                     # the default controller of the package (also the only one for friction-driven bases)
}


def make_plant(comp, dt=.002):
    from backends.mujoco_plant import MujocoPlant
    from teamc.interfaces import TeamModel

    class PerStepGraspPlant(MujocoPlant):
        """The MuJoCo plant of the workflow, recording the inferred grasp wrench at every step with a latched grasp."""

        def __init__(self, comp_, **kw):
            super().__init__(comp_, **kw)
            self._infer = TeamModel(comp_)
            self._robot_of = {c: comp_.connections[c]['robot'] for c in self.latched
                              if comp_.connections[c]['kind'] == 'grasp'}
            self.k = 0
            self.rows = []                                  # (step, time at its start, grasp, wrench [moment; force])

        def step(self, n=1):
            for _ in range(n):
                closed = [c for c in self._robot_of if self.latched[c]]
                if closed:
                    q0, v0 = self.state()
                    t0 = self.time
                super().step(1)
                if closed:
                    _, v1 = self.state()
                    a = (v1 - v0) / self.dt                  # acceleration of the step
                    kin = self.comp.tree.sweep(q0)
                    for c in closed:
                        rid = self._robot_of[c]
                        lam = self._infer.sensor_to_grasp(rid, self.sensor_wrench(rid), q0, kin, v0, a)
                        self.rows.append((self.k, t0, c, np.asarray(lam, float)))
                self.k += 1

    return PerStepGraspPlant(comp, timestep=dt)


if __name__ == '__main__':
    team = sys.argv[1]
    kind = sys.argv[2] if len(sys.argv) > 2 else 'fitted05'
    from mission.workflow import run_and_report
    from teamc.io import INPUTS, load_team, read_json
    comp = load_team(team)
    mission = read_json(INPUTS / 'missions' / f'{team}.json')
    plant = make_plant(comp)
    every = max(int(round(1. / (50. * plant.dt))), 1)         # record interval of the workflow (50 Hz)
    slow_every = max(int(round(.01 / plant.dt)), 1)
    out = RESULTS / f'native_loading_{team}_{kind}'
    t0 = time.time()
    report, A, rec, ctrl, info = run_and_report(plant, comp, mission, out, f'MuJoCo {team} {kind}:',
                                                options=dict(slow_every=slow_every, controller=dict(KINDS[kind])),
                                                quiet=True, figures=False)
    checks = report['bounds']['checks']
    res = dict(team=team, kind=kind, controller=KINDS[kind], dt=plant.dt, record_interval_steps=every,
               status=report['status'], sim_time_s=report['sim_time_s'], seconds=time.time() - t0,
               payload_mm=1e3 * report['tracking']['payload_position_error_m']['max'],
               payload_rms_mm=1e3 * report['tracking']['payload_position_error_m']['rms'],
               audit_grasp_moment_max=checks['grasp_moment']['value'], grasp_moment_bound=checks['grasp_moment']['bound'],
               saturated_steps=report['efforts']['saturated_steps'], by_grasp={})
    bound = checks['grasp_moment']['bound']
    for c in sorted({r[2] for r in plant.rows}):
        rows = [r for r in plant.rows if r[2] == c]
        k = np.array([r[0] for r in rows])
        t = np.array([r[1] for r in rows])
        lam = np.array([r[3] for r in rows])
        m = np.linalg.norm(lam[:, :3], axis=1)
        f = np.linalg.norm(lam[:, 3:], axis=1)
        # 20 ms vector means over the record intervals in which the grasp is latched at every step
        iv = k // every
        means = []
        for j in np.unique(iv):
            sel = iv == j
            if sel.sum() == every:
                means.append(lam[sel].mean(axis=0))
        means = np.array(means)
        mm = np.linalg.norm(means[:, :3], axis=1) if len(means) else np.zeros(0)
        i = int(m.argmax())
        res['by_grasp'][c] = dict(steps=int(len(m)), moment_max_per_step=float(m.max()), t_of_max=float(t[i]),
                                  moment_max_20ms_mean=float(mm.max()) if len(mm) else None,
                                  force_max_per_step=float(f.max()), steps_above_bound=int((m > bound).sum()),
                                  share_above_bound=float((m > bound).mean()), moment_p99_per_step=float(np.percentile(m, 99)))
    g = res['by_grasp']
    res['moment_max_per_step'] = max(v['moment_max_per_step'] for v in g.values())
    res['moment_max_20ms_mean'] = max(v['moment_max_20ms_mean'] for v in g.values())
    res['steps_above_bound'] = sum(v['steps_above_bound'] for v in g.values())
    (RESULTS / f'native_loading_{team}_{kind}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k_: v for k_, v in res.items() if k_ != 'by_grasp'}), flush=True)
