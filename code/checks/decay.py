"""Collect the decay tests of the mission controller on the two plants (tools/run_decay.py of the package).

Reads <package>/results/v7/decay/<plant>_<team>_<direction><seed>[ _nofilter]/decay.json and decay.npz.
Written here: decay.json (one entry per run) and decay.npz (the time histories of the runs along the worst
direction, for the figure).  Nothing of the package is run or changed.

A run: the team holds the payload at rest; after settling, the plant is given an admissible velocity jump; then
W = 1/2 |v - v_r|_M^2 (v_r: fitted reference of the loops) is followed at every step, with the two exponentials
of the stability statement.  Variants: feedforward of the desired motion with the rate filter of the missions
(15 Hz) and without it; fitted feedforward evaluated with the references (10 ms) and in every step (2 ms); fitted
feedforward with the rate gains K = sqrt(k_p m) on the arm and lift joints (half those of the other variants).
Checked: from the second step after the jump on, no effort reaches 95 % of its bound (the loops are linear); a
first step at or beyond the bound is recorded (clipped_steps, steps_above_95).
"""
import json
import sys
from pathlib import Path

import numpy as np

from lib import RESULTS, ROOT, ROLLING

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'v7' / 'decay'
runs, arrays = {}, {}
for plant in ('pinocchio', 'mujoco'):
    for team in ROLLING:
        for tag in ('worst0', 'random1', 'random2', 'random3'):
            folder = RUNS / f'{plant}_{team}_{tag}'
            if not (folder / 'decay.json').exists():
                continue
            entries = json.loads((folder / 'decay.json').read_text())
            nofilter = json.loads((RUNS / f'{plant}_{team}_{tag}_nofilter' / 'decay.json').read_text())
            assert len(nofilter) == 1 and nofilter[0]['feedforward'] == 'desired' and nofilter[0]['velocity_filter_hz'] == 0.
            named = {}
            for e in entries:
                if e['feedforward'] == 'desired':
                    assert e['velocity_filter_hz'] == 15.
                    named['desired'] = e
                else:
                    assert e['velocity_filter_hz'] == 0.
                    named['fitted_' + e['feedforward_rate']] = e
            named['desired_nofilter'] = nofilter[0]
            half = json.loads((RUNS / f'{plant}_{team}_{tag}_half' / 'decay.json').read_text())
            assert len(half) == 1 and half[0]['feedforward'] == 'fitted' and half[0]['velocity_filter_hz'] == 0.
            assert half[0]['arm_damping_ratio'] == half[0]['lift_damping_ratio'] == .5
            named['fitted_half'] = half[0]
            for e in entries + nofilter:
                assert e['arm_damping_ratio'] == e['lift_damping_ratio'] == 1. and e['advance'] == 'positions'
            assert set(named) == {'desired', 'desired_nofilter', 'fitted_slow', 'fitted_fast', 'fitted_half'}, (folder, sorted(named))
            source = dict(desired=(folder, 'desired'), fitted_slow=(folder, 'fitted_slow'), fitted_fast=(folder, 'fitted_fast'),
                          desired_nofilter=(RUNS / f'{plant}_{team}_{tag}_nofilter', 'desired'),
                          fitted_half=(RUNS / f'{plant}_{team}_{tag}_half', 'fitted_slow'))
            for variant, e in named.items():
                key = f'{plant}_{team}_{tag}_{variant}'
                where, name = source[variant]
                effort = np.load(where / 'decay.npz')[f'{name}:effort']          # largest effort/bound at every step
                assert abs(effort.max() - e['max_effort_ratio']) < 1e-12, key
                runs[key] = dict(plant=plant, team=team, direction=tag, variant=variant, **{k: e[k] for k in (
                    'W0', 'W_max_ratio', 't_of_max', 'W_end_ratio', 'largest_rise_between_steps', 'above_upper_bound', 'time_to_floor',
                    'floor', 'rate_fitted', 'balance', 'max_effort_ratio', 'max_internal_ratio', 'timestep', 'velocity_filter_hz')},
                    form_min=e['start']['form_min'], kmin=e['start']['rate'][0], kmax=e['start']['rate'][1], d=e['start']['d'],
                    W_before=e['start']['W_before'], steps=len(effort), clipped_steps=int((effort > 1.).sum()),
                    steps_above_95=int((effort > .95).sum()), effort_after_first=float(effort[1:].max()))
                # no actuator at its bound, so that the loops are linear, except at the first step after the jump (where
                # the plant clips what exceeds a bound); such runs are recorded and named in the text
                assert effort[1:].max() < .95, key
            if tag == 'worst0':
                z = np.load(folder / 'decay.npz')
                zn = np.load(RUNS / f'{plant}_{team}_{tag}_nofilter' / 'decay.npz')
                zh = np.load(RUNS / f'{plant}_{team}_{tag}_half' / 'decay.npz')
                for variant, src, name in (('desired', z, 'desired'), ('fitted_slow', z, 'fitted_slow'), ('fitted_fast', z, 'fitted_fast'),
                                           ('desired_nofilter', zn, 'desired'), ('fitted_half', zh, 'fitted_slow')):
                    for q in ('t', 'W', 'upper', 'lower'):
                        arrays[f'{plant}_{team}_{variant}:{q}'] = src[f'{name}:{q}']
(RESULTS / 'decay.json').write_text(json.dumps(runs, indent=1))
np.savez_compressed(RESULTS / 'decay.npz', **arrays)
print('runs', len(runs))
for key, r in runs.items():
    if r['steps_above_95']:
        print(f"{key}: effort/bound {r['max_effort_ratio']:.3f} at the first step after the jump ({r['clipped_steps']} step clipped), "
              f"then at most {r['effort_after_first']:.3f}")
for key, r in runs.items():
    if r['direction'] == 'worst0':
        print(f"{key:44s} max {r['W_max_ratio']:.3f} at {1e3 * r['t_of_max']:4.0f} ms  above bound {r['above_upper_bound']:8.3f}  "
              f"rate {r['rate_fitted']:6.2f} (kmin {r['kmin']:.2f})  form {r['form_min']:+.2f}  effort {r['max_effort_ratio']:.2f}")
