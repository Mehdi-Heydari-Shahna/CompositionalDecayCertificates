"""Check of the per-step grasp-moment inference of run_native_loading.py against the MuJoCo solver (Sections 7.6, 7.8).

On the carrying segment of run_segment_compare.py (MuJoCo, 2 ms steps), the grasp wrench inferred at every step from
the wrist sensor (TeamModel.sensor_to_grasp, tool-side inertia removed with the acceleration of the step) is compared
with the wrench that the solver's grasp constraints apply to the payload (MujocoPlant.weld_wrench), moved to the
tool-frame origin. Reported per team: the largest difference of the moment norms, absolute and relative to the largest
moment, over all steps and grasps. The controller is F/2 (fitted05 of run_native_loading.py) for the full-rank teams
and the package's default controller (desired) for the friction-driven Husky teams, for which the fitted feedforward
is not available; another kind can be given as TEAM:KIND.

Usage: python3 check_native_inference.py [TEAM[:KIND] ...]   (default: team3 ridgeback3 husky2 husky3)
Output: check_native_inference.json
"""
from __future__ import annotations

import os

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('MUJOCO_GL', 'disable')

import json  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402

from teamdata import PKG, RESULTS  # noqa: E402,F401  (puts the package on sys.path)
from run_native_loading import KINDS, make_plant  # noqa: E402


def check(team, kind='fitted05', at=1.0, hold=2.0, move=4.0, dt=.002):
    from mission.controller import TeamController
    from mission.runner import carry_configuration
    from teamc.io import INPUTS, load_team, read_json
    comp = load_team(team)
    mission = read_json(INPUTS / 'missions' / f'{team}.json')
    plant = make_plant(comp, dt)
    slow_every = int(round(.01 / dt))
    ctrl = TeamController(comp, mission, plant.dt, slow_every=slow_every, options=dict(KINDS[kind]))
    speed = ctrl.governor.speed_limit
    ctrl.governor.speed_limit = lambda l: 0.
    q0 = carry_configuration(comp, mission, ctrl, at)
    plant.reset(q0)
    for cid in ctrl.grasp_of.values():
        ctrl.on_latched(cid, plant.latch(cid))
    meas = plant.measure()
    meas['obstacle'] = None
    ctrl.start_carrying(meas, at)
    model = plant._infer
    payload_body = plant.meta['payload']['body']
    diffs, moments = [], []
    steps = int(round((hold + move) / dt))
    for k in range(steps):
        if k == int(round(hold / dt)):
            ctrl.governor.speed_limit = speed
        if k % slow_every == 0:
            meas['obstacle'] = None
            ctrl.slow(meas)
        q_start, _ = plant.state()
        n_rows = len(plant.rows)
        plant.set_efforts(ctrl.fast(meas))
        plant.step()
        kin = comp.tree.sweep(q_start)
        origin = plant.body_pose(payload_body)[:3, 3]           # payload pose of the state the solver used
        for _, _, cid, lam in plant.rows[n_rows:]:
            force, moment = plant.weld_wrench(cid)
            s = model._sensor[comp.connections[cid]['robot']]
            p_tool = (kin[0][s['body']] @ s['T_tool'])[:3, 3]
            m_tool = moment - np.cross(p_tool - origin, force)
            diffs.append(abs(np.linalg.norm(m_tool) - np.linalg.norm(lam[:3])))
            moments.append(np.linalg.norm(lam[:3]))
        meas = plant.measure()
    diffs, moments = np.array(diffs), np.array(moments)
    return dict(team=team, kind=kind, steps=steps, samples=int(len(diffs)), moment_max=float(moments.max()),
                largest_difference_Nm=float(diffs.max()), largest_difference_relative=float(diffs.max() / moments.max()))


if __name__ == '__main__':
    teams = sys.argv[1:] or ['team3', 'ridgeback3', 'husky2', 'husky3']
    out = []
    for arg in teams:
        team, _, kind = arg.partition(':')
        out.append(check(team, kind or ('desired' if team.startswith('husky') else 'fitted05')))
    for r in out:
        print(json.dumps(r), flush=True)
    (RESULTS / 'check_native_inference.json').write_text(json.dumps(out, indent=1))
