"""Carrying segment on the ideal-constraint (Pinocchio) plant or on MuJoCo: desired-motion and fitted-reference loops at
several plant steps (Section 7.6, Table 3).

The team starts at route station AT (m) in its carrying mode, holds the payload for HOLD s, and then accelerates along
the route for MOVE s under the mission's speed governor. References, requests and feedforward are renewed every 10 ms,
as in the missions; actuator feedback is applied in every plant step. Controllers:
    D    desired-motion feedforward, arm/lift rate gains k_d, no rate filter        (single-joint damping ratio 1)
    F    fitted-reference feedforward, k_d, no rate filter                          (single-joint damping ratio 1.25)
    F/2  fitted-reference feedforward, k_d/2 (damping ratios 0.5), no rate filter   (single-joint damping ratio 1)
For one joint of design inertia m with k_d = 2 sqrt(k_p m), the fitted loop adds the rate feedback m k_p/K, so D and F/2
have the same single-joint damping (Remark 20). The payload position error |p - p_ref| is sampled at every renewal; the
peak and rms values over the motion (t >= HOLD) are reported, and over the whole segment as well.

Usage: python3 run_segment_compare.py TEAM [DT ...]           Pinocchio plant (default steps: 0.002 0.001 0.0005)
       python3 run_segment_compare.py mujoco TEAM [DT ...]    MuJoCo plant with the settings of the missions
Output: segment_<team>.json/.npz (Pinocchio), segment_mujoco_<team>.json/.npz (MuJoCo).
"""
from __future__ import annotations

import os

# small matrices: one BLAS thread is faster, and two runs can share a machine (set before NumPy is imported)
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
os.environ.setdefault('MUJOCO_GL', 'disable')                 # physics only, no OpenGL

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from teamdata import PKG, RESULTS  # noqa: E402,F401  (puts the package on sys.path)

VARIANTS = {
    'D': dict(feedforward='desired', velocity_filter_hz=0.),
    'F': dict(feedforward='fitted'),
    'F/2': dict(feedforward='fitted', arm_damping_ratio=.5, lift_damping_ratio=.5),
}


def make_plant(engine, comp, dt):
    if engine == 'mujoco':
        from backends.mujoco_plant import MujocoPlant
        return MujocoPlant(comp, timestep=dt)                 # defaults of workflows/mujoco/run_mission.py
    from backends.pin_plant import PinPlant
    return PinPlant(comp, timestep=dt)


def run(team, variant, dt, engine='pinocchio', at=1.0, hold=2.0, move=4.0, renew=.01):
    from mission.controller import TeamController
    from mission.runner import carry_configuration
    from teamc.io import INPUTS, load_team, read_json
    comp = load_team(team)
    mission = read_json(INPUTS / 'missions' / f'{team}.json')
    plant = make_plant(engine, comp, dt)
    slow_every = max(int(round(renew / dt)), 1)
    ctrl = TeamController(comp, mission, plant.dt, slow_every=slow_every, options=dict(VARIANTS[variant]))
    speed = ctrl.governor.speed_limit
    ctrl.governor.speed_limit = lambda l: 0.
    q0 = carry_configuration(comp, mission, ctrl, at)
    plant.reset(q0)
    for cid in ctrl.grasp_of.values():
        ctrl.on_latched(cid, plant.latch(cid))
    meas = plant.measure()
    meas['obstacle'] = None
    ctrl.start_carrying(meas, at)
    pb = np.asarray(comp.payload['base'], int)
    steps = int(round((hold + move) / plant.dt))
    hold_steps = int(round(hold / plant.dt))
    rows = []
    t0 = time.time()
    for k in range(steps):
        if k == hold_steps:
            ctrl.governor.speed_limit = speed
        if k % slow_every == 0:
            meas['obstacle'] = None
            ctrl.slow(meas)
            e = meas['q'][pb] - ctrl.pref
            rows.append((meas['t'], float(np.linalg.norm(e[:3])), ctrl.governor.l, ctrl.governor.rate))
        plant.set_efforts(ctrl.fast(meas))
        plant.step()
        meas = plant.measure()
    A = np.array(rows)
    t = A[:, 0] - A[0, 0]
    err = A[:, 1]
    moving = t >= hold - 1e-9
    return dict(team=team, engine=engine, variant=variant, dt=dt, renew=renew, at=at, hold=hold, move=move,
                peak_mm=1e3 * float(err.max()), rms_mm=1e3 * float(np.sqrt(np.mean(err ** 2))),
                peak_moving_mm=1e3 * float(err[moving].max()), rms_moving_mm=1e3 * float(np.sqrt(np.mean(err[moving] ** 2))),
                distance_m=float(A[-1, 2] - A[0, 2]), speed_end=float(A[-1, 3]), seconds=time.time() - t0), \
        dict(t=t, err=err, l=A[:, 2], rate=A[:, 3])


if __name__ == '__main__':
    args = sys.argv[1:]
    engine = 'pinocchio'
    if args and args[0] == 'mujoco':
        engine, args = 'mujoco', args[1:]
    team = args[0]
    dts = [float(x) for x in args[1:]] or [.002, .001, .0005]
    stem = f'segment_{team}' if engine == 'pinocchio' else f'segment_mujoco_{team}'
    out, arrays = [], {}
    for dt in dts:
        for variant in VARIANTS:
            s, a = run(team, variant, dt, engine)
            out.append(s)
            arrays.update({f'{variant}:{dt}:{k}': v for k, v in a.items()})
            print(json.dumps(s), flush=True)
            (RESULTS / f'{stem}.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(RESULTS / f'{stem}.npz', **arrays)
