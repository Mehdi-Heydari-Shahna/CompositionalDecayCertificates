#!/usr/bin/env python
"""Decay of the error energy of the mission controller on a plant, for the two feedforwards.

    python tools/run_decay.py --team team2 --plant pinocchio --out results/decay_team2

The team holds the payload at a point of its route (the reference rests).  After a time to
settle on its contacts the plant is given a velocity jump the team can have in the carrying
mode (what an impulse does), so the only thing that happens afterwards is the decay of that
velocity under the joint and wheel loops of the mission controller.  The quantity looked at
is the one of the stability statement (``mission/certificate.py``):

    W = 1/2 |v - v_r|_M^2,   v_r: the fitted reference of the loops at the measured state.

On the compiled model, with the effort of the fitted reference as feedforward,
``-2 k_max W <= W' <= -2 k_min W`` with the two rates the gains and the inertia give.
With the inverse dynamics of the desired motion as feedforward the statement does not
apply; whether ``W`` then rises depends on the gains and the direction.

The jump is along the direction in which the form ``D_r - M_r G_p`` is smallest
(``--direction worst``; the direction in which the loops with the desired-motion feedforward
do worst), or a random admissible one (``--direction random --seed N``).

Written: ``decay.json`` (summary of every run) and ``decay.npz`` (time histories, every step).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402


def start_velocity(ctrl, q, mode, direction, size, seed):
    """Admissible velocity jump of M-norm ``size`` and the form of the loops at ``q``."""
    model, tree = ctrl.model, ctrl.tree
    lift = model.lift(q, mode)
    E = lift.E
    L = E[ctrl.port_coordinate]
    Mr = E.T @ tree.mass(q) @ E
    H = L.T @ (ctrl.K[:, None] * L)
    Gp = np.linalg.solve(H, L.T @ ((ctrl.K * ctrl.rate_ratio)[:, None] * L))
    Dr = H + E.T @ (model.damping[:, None] * E)
    form = Dr - Mr @ Gp
    values, vectors = eigh(.5 * (form + form.T), Mr)
    if direction == 'worst':
        u = vectors[:, 0]
    else:
        u = np.random.default_rng(seed).standard_normal(E.shape[1])
    u = size * u / np.sqrt(u @ Mr @ u)
    rate = eigh(Dr, Mr, eigvals_only=True)
    return E @ u, dict(form_min=float(values[0]), form_max=float(values[-1]), rate=[float(rate[0]), float(rate[-1])],
                       d=int(E.shape[1]))


def run(team, plant_name, feedforward, rate, args):
    from mission.controller import TeamController
    from mission.runner import carry_configuration
    from teamc.io import INPUTS, load_team, read_json
    comp = load_team(team)
    mission = read_json(INPUTS / 'missions' / f'{team}.json')
    if plant_name == 'pinocchio':
        from backends.pin_plant import PinPlant
        plant = PinPlant(comp, timestep=args.timestep)
    else:
        from backends.gl import prepare_mujoco_gl
        prepare_mujoco_gl(render=False)
        from backends.mujoco_plant import MujocoPlant
        plant = MujocoPlant(comp, timestep=args.timestep)
    slow_every = max(int(round(.01 / args.timestep)), 1)
    options = dict(feedforward=feedforward, feedforward_rate=rate, monitor=True)
    if args.filter is not None:
        options['velocity_filter_hz'] = args.filter
    options.update(args.options)
    ctrl = TeamController(comp, mission, plant.dt, slow_every=slow_every, options=options)
    ctrl.governor.speed_limit = lambda l: 0.                      # the reference rests at its place on the route
    q0 = carry_configuration(comp, mission, ctrl, args.at)
    plant.reset(q0)
    for cid in ctrl.grasp_of.values():
        ctrl.on_latched(cid, plant.latch(cid))
    meas = plant.measure()
    meas['obstacle'] = None
    ctrl.start_carrying(meas, args.at)
    ref = ctrl.reference
    bound = np.max(np.abs(ctrl.model.effort_bounds), axis=1)
    settle = int(round(args.settle / (plant.dt * slow_every))) * slow_every
    floor_before = 0.
    for k in range(settle):                                       # the plant settles on its contacts
        if k % slow_every == 0:
            meas['obstacle'] = None
            ctrl.slow(meas)
            if k >= settle // 2:
                floor_before = max(floor_before, ctrl.certificate['W'])
        plant.set_efforts(ctrl.fast(meas))
        plant.step()
        meas = plant.measure()
    q0 = meas['q'].copy()
    dv, start = start_velocity(ctrl, q0, ctrl.mode, args.direction, args.size, args.seed)
    start['W_before'] = floor_before
    plant.kick(dv)
    meas = plant.measure()
    rows = []
    steps = int(round(args.duration / plant.dt))
    for k in range(steps + 1):
        if k % slow_every == 0:
            meas['obstacle'] = None
            ctrl.slow(meas)
        f = ctrl.fast(meas)
        # the quantities of the statement at this very step, evaluated apart from the controller
        q, v = meas['q'], meas['v']
        fit = ref.fit(q, ctrl.mode, ctrl.K, ctrl._requested(q, v, meas['t'] - ctrl.t_slow))
        fit['M'] = ctrl.tree.mass(q, kinematics=fit['kin'])
        Q = ref.quantities(fit, v, ctrl.K)
        rows.append(dict(t=meas['t'], W=Q['W'], W_admissible=Q['W_admissible'], dissipation=Q['dissipation'], kmin=Q['rate'][0],
                         kmax=Q['rate'][1], effort=float(np.max(np.abs(f) / bound)), internal=float(np.max(np.abs(Q['internal']) / bound)),
                         displacement=float(np.max(np.abs(q - q0)))))
        if k == steps:
            break
        plant.set_efforts(f)
        plant.step()
        meas = plant.measure()
    A = {key: np.array([r[key] for r in rows]) for key in rows[0]}
    t, W = A['t'] - A['t'][0], A['W']
    ratio = W / W[0]
    trap = lambda y: np.concatenate([[0.], np.cumsum(.5 * (y[1:] + y[:-1]) * np.diff(t))])
    upper, lower = np.exp(-2. * trap(A['kmin'])), np.exp(-2. * trap(A['kmax']))
    A.update(upper=upper, lower=lower, t=t)
    i = int(np.argmax(ratio))
    # the decay is followed down to a floor the sampled loops and the plant leave; the statement is compared above it
    above = ratio > args.floor
    last = int(np.argmin(above)) if not above.all() else len(t)
    summary = dict(team=team, plant=plant_name, feedforward=feedforward, feedforward_rate=rate, timestep=plant.dt,
                   velocity_filter_hz=ctrl.opt['velocity_filter_hz'], advance=ctrl.opt['advance'],
                   arm_damping_ratio=ctrl.opt['arm_damping_ratio'], lift_damping_ratio=ctrl.opt['lift_damping_ratio'],
                   options=dict(args.options), start=start, W0=float(W[0]),
                   W_max_ratio=float(ratio[i]), t_of_max=float(t[i]), W_end_ratio=float(ratio[-1]),
                   largest_rise_between_steps=float(np.max(np.diff(ratio))),
                   above_upper_bound=float(np.max((ratio / upper - 1.)[:last])) if last else float('nan'),
                   time_to_floor=float(t[last]) if last < len(t) else None, floor=args.floor,
                   rate_fitted=float(-np.polyfit(t[:last], np.log(ratio[:last]), 1)[0] / 2.) if last > 5 else float('nan'),
                   balance=float((W[-1] - W[0] + trap(A['dissipation'])[-1]) / W[0]),
                   max_effort_ratio=float(A['effort'].max()), max_internal_ratio=float(A['internal'].max()),
                   largest_displacement=float(A['displacement'].max()))
    return summary, A


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team', default='team2')
    ap.add_argument('--plant', choices=('pinocchio', 'mujoco'), default='pinocchio')
    ap.add_argument('--out', default=None)
    ap.add_argument('--feedforward', nargs='+', default=['desired', 'fitted'], choices=('desired', 'fitted'))
    ap.add_argument('--rate', nargs='+', default=['slow'], choices=('slow', 'fast'), help='evaluation of the fitted effort')
    ap.add_argument('--timestep', type=float, default=.002)
    ap.add_argument('--duration', type=float, default=.3)
    ap.add_argument('--at', type=float, default=1., help='route parameter of the place held (m)')
    ap.add_argument('--settle', type=float, default=None, help='time before the jump (s); default 0.1 (pinocchio), 2 (mujoco)')
    ap.add_argument('--size', type=float, default=.1, help='M-norm of the velocity jump (sqrt(kg) m/s)')
    ap.add_argument('--direction', choices=('worst', 'random'), default='worst')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--floor', type=float, default=1e-3, help='the bounds are compared while W is above this share of W(0)')
    ap.add_argument('--filter', type=float, default=None, help='velocity filter of the damping term (Hz); default: that of the feedforward')
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE',
                    help='another option of the mission controller (mission/controller.py, DEFAULTS), the value in JSON; '
                         'for example --option arm_damping_ratio=0.5.  May be given several times')
    args = ap.parse_args(argv)
    args.options = {}
    for item in args.option:
        key, _, value = item.partition('=')
        if not key or not value:
            ap.error(f'--option needs KEY=VALUE, got {item!r}')
        args.options[key] = json.loads(value)
    if args.settle is None:
        args.settle = .1 if args.plant == 'pinocchio' else 2.
    out = Path(args.out or ROOT / 'results' / f'decay_{args.plant}_{args.team}')
    out.mkdir(parents=True, exist_ok=True)
    summaries, arrays = [], {}
    for ff in args.feedforward:
        for rate in (args.rate if ff == 'fitted' else ['slow']):
            t0 = time.perf_counter()
            s, A = run(args.team, args.plant, ff, rate, args)
            s['seconds'] = round(time.perf_counter() - t0, 1)
            summaries.append(s)
            key = ff if ff == 'desired' else f'{ff}_{rate}'
            arrays.update({f'{key}:{name}': value for name, value in A.items()})
            print(json.dumps(s), flush=True)
    (out / 'decay.json').write_text(json.dumps(summaries, indent=1), encoding='utf-8')
    np.savez_compressed(out / 'decay.npz', **arrays)
    print('written to', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
