"""The mission controller with the fitted feedforward under what the statement leaves out: sampling, model error,
noisy and delayed rate measurements.  Plant with ideal constraints (the compiled model up to its time stepping).

    python exp12_robustness.py <team> [--duration 8] [--only sampling,mass,noise,delay]

Every run is a carry stage of the mission of the package: the team starts at rest with the beam in its clamps at
1 m of its route and follows the route (speeding up to 0.45 m/s, then at constant speed), with the loops of the
mission controller and the fitted feedforward (``advance='positions'``), or the feedforward of the desired motion
with the 15 Hz rate filter of the earlier controller for comparison.  Nothing of the package is changed: the
variations are made here.

sampling  reference update every 2, 4, 10, 20 ms (the loops every 2 ms).  Section 6.3 of the paper, (26):
          the error W should scale like the square of the update interval once the transient has passed.
mass      the controller's model of the beam has 0.7, 0.9, 1.1, 1.3 times its mass (inertia scaled with it);
          the plant has the true beam.  Proposition 5 (residuals with a part linear in the error): the error energy
          settles at a level set by the residual, and the rate is that of D_r + sym(Psi).
noise     white noise of standard deviation 1e-3 and 5e-3 (rad/s, m/s) on every measured rate the controller sees;
          also the fitted feedforward with the 15 Hz rate filter of the earlier controller in the damping term.
delay     the controller sees the measurement of 1 or 2 loop steps earlier; for 2 steps also the fitted feedforward
          with the 15 Hz rate filter and the feedforward of the desired motion without it.
half      the fitted feedforward with half the rate gains on the arm and lift joints (damping ratio 0.5 of the joint
          design, K = sqrt(k_p m)), nominal and under the mass errors, noise and delays above (labels 'half ...').

Recorded per run (over the cruise, t > 3 s): W of the loops (median, 99th percentile, largest), the position error
of the payload (true plant state against the reference: largest, rms, the norm of its mean vector -- a steady sag --
and the largest deviation from that mean), grasp loads measured by the controller, efforts.

Written: exp12_<team>.json (one entry per run), after every run.  A run whose label or whose settings are already
there is not made again (the nominal run with the fitted feedforward is the sampling run at 10 ms).
"""
import argparse
import copy
import json
import time

import numpy as np

from lib import RESULTS, ROOT  # noqa: F401  (sets up the import path of the package)
from backends.pin_plant import PinPlant
from mission.runner import run_mission
from teamc.io import INPUTS, load_team, read_json

ap = argparse.ArgumentParser()
ap.add_argument('team')
ap.add_argument('--duration', type=float, default=8.)
ap.add_argument('--only', default='sampling,mass,noise,delay', help='also: half')
ap.add_argument('--masses', default='0.7,0.9,1.1,1.3', help='mass scales of the beam in the model of the controller')
ap.add_argument('--redo', default='', help='labels starting with one of these comma-separated prefixes are run again')
ap.add_argument('--resume-log', default=None, help='take the finished runs from the output of an earlier call '
                '(one JSON line per run, same team and duration)')
args = ap.parse_args()
only = set(args.only.split(','))
MISSION = read_json(INPUTS / 'missions' / f'{args.team}.json')


class Measured:
    """The plant as the controller sees it: rates with noise, measurements of earlier steps."""

    def __init__(self, plant, noise=0., delay=0, seed=1):
        self._plant, self._noise, self._delay = plant, noise, int(delay)
        self._rng = np.random.default_rng(seed)
        self._past = []

    def __getattr__(self, name):
        return getattr(self._plant, name)

    def reset(self, q0):
        self._past = []
        return self._plant.reset(q0)

    def measure(self):
        m = self._plant.measure()
        if self._noise:
            m = dict(m, v=m['v'] + self._noise * self._rng.standard_normal(m['v'].shape))
        if self._delay:
            self._past.append(m)
            if len(self._past) > self._delay + 1:
                self._past.pop(0)
            m = dict(self._past[0], t=m['t'])         # what was measured delay steps ago, at the time of now
        return m


def believed(comp, mission, scale):
    """The team and mission as the controller models them: the beam with ``scale`` times its mass."""
    if scale == 1.:
        return comp, mission
    tree = comp.tree
    i = tree.index[comp.payload['body']]
    tree.mass_kg[i] *= scale
    tree.inertia[i] = tree.inertia[i] * scale
    tree._inertia_massive = tree.inertia[tree.massive]
    mission = copy.deepcopy(mission)
    mission['payload']['mass_kg'] = float(mission['payload']['mass_kg']) * scale
    return comp, mission


def one(label, feedforward='fitted', slow_every=5, mass=1., noise=0., delay=0, velocity_filter=None, damping_ratio=None):
    true = load_team(args.team)
    plant = Measured(PinPlant(true), noise=noise, delay=delay)
    comp, mission = believed(load_team(args.team), MISSION, mass)
    controller = dict(feedforward=feedforward, monitor=True)
    if velocity_filter is not None:                 # rate filter of the damping term (default: 15 Hz desired, none fitted)
        controller['velocity_filter_hz'] = velocity_filter
    if damping_ratio is not None:                   # rate gains of the arm and lift joints (default: damping ratio 1)
        controller['arm_damping_ratio'] = controller['lift_damping_ratio'] = damping_ratio
    t0 = time.perf_counter()
    seen = []                                       # the plant's own state (the record holds what the controller saw)

    def on_record(t, plant_, ctrl_):
        q_true = plant_._plant.state()[0]
        seen.append((t, *(q_true[ctrl_.payload] - ctrl_.pref)[:3]))

    ctrl, rec, info = run_mission(plant, comp, mission, options=dict(start='navigate', start_l=1., duration=args.duration,
                                                                    slow_every=slow_every, controller=controller),
                                  on_record=on_record)
    A = rec.arrays()
    t = A['t']
    late = t > 3.                                   # the team cruises from about 2.5 s on
    W = A['fit:W']
    seen = np.array(seen)
    ts, vec = seen[:, 0], seen[:, 1:][seen[:, 0] > 3.]
    payload = np.linalg.norm(vec, axis=1)
    offset = vec.mean(axis=0)                       # the steady part of the error (a sag, for instance)
    moving = np.linalg.norm(vec - offset, axis=1)   # what varies about it
    moments = np.max([np.linalg.norm(A[f'lam_raw:{c}'][:, :3], axis=1) for c in ctrl.grasp_of.values()], axis=0)
    forces = np.max([np.linalg.norm(A[f'lam_raw:{c}'][:, 3:], axis=1) for c in ctrl.grasp_of.values()], axis=0)
    bound = np.max(np.abs(ctrl.model.effort_bounds), axis=1)
    effort = np.max(np.abs(A['f']) / bound, axis=1)
    out = dict(label=label, team=args.team, feedforward=feedforward, slow_every=slow_every, update_s=slow_every * plant.dt,
               mass_scale=mass, noise=noise, delay_steps=delay, duration_s=args.duration, velocity_filter_hz=velocity_filter,
               damping_ratio=damping_ratio,
               status=info['status'],
               seconds=round(time.perf_counter() - t0, 1),
               W_median=float(np.nanmedian(W[late])), W_p99=float(np.nanpercentile(W[late], 99)), W_max=float(np.nanmax(W[late])),
               W_median_all=float(np.nanmedian(W)), payload_max=float(payload.max()), payload_rms=float(np.sqrt(np.mean(payload ** 2))),
               payload_offset=float(np.linalg.norm(offset)), payload_varying_max=float(moving.max()),
               grasp_moment_rms=float(np.sqrt(np.mean(moments[late] ** 2))), grasp_force_max=float(forces[late].max()),
               internal_effort_max=float(np.nanmax(A['fit:internal'][late])), effort_max=float(effort.max()),
               effort_rms=float(np.sqrt(np.mean(effort[late] ** 2))), speed_end=float(A['rate'][-1]),
               rate_min=float(np.nanmin(A['fit:rate'][:, 0])))
    print(json.dumps(out), flush=True)
    return out


path = RESULTS / f'exp12_{args.team}.json'
done = json.loads(path.read_text()) if path.exists() else []
if args.resume_log:
    for line in open(args.resume_log):
        if line.startswith('{'):
            r = json.loads(line)
            r.setdefault('duration_s', args.duration)       # the earlier call ran with the same --duration
            if r['team'] == args.team and r['label'] not in {d['label'] for d in done}:
                done.append(r)
SETTINGS = ('feedforward', 'slow_every', 'mass_scale', 'noise', 'delay_steps', 'duration_s', 'velocity_filter_hz', 'damping_ratio')


REDO = tuple(x for x in args.redo.split(',') if x)


def run(label, feedforward='fitted', slow_every=5, mass=1., noise=0., delay=0, velocity_filter=None, damping_ratio=None):
    want = (feedforward, slow_every, mass, noise, delay, args.duration, velocity_filter, damping_ratio)
    again = bool(REDO) and label.startswith(REDO)
    if again:
        done[:] = [r for r in done if r['label'] != label]
    for r in done:
        if r['label'] == label:
            assert tuple(r.get(k) for k in SETTINGS) == want, (label, r)
            return r
    for r in done:
        if not again and tuple(r.get(k) for k in SETTINGS) == want:     # the same run under another label
            r = dict(r, label=label, same_run_as=r['label'])
            break
    else:
        r = one(label, feedforward, slow_every, mass, noise, delay, velocity_filter, damping_ratio)
    done.append(r)
    path.write_text(json.dumps(done, indent=1))
    return r


if 'sampling' in only:
    for k in (1, 2, 5, 10):
        run(f'sampling {2 * k} ms', slow_every=k)
if 'mass' in only or 'noise' in only or 'delay' in only:
    run('nominal')
    run('nominal, desired', feedforward='desired')
if 'mass' in only:
    for s in (float(x) for x in args.masses.split(',')):
        run(f'mass x{s}', mass=s)
        run(f'mass x{s}, desired', feedforward='desired', mass=s)
if 'noise' in only:
    for sd in (1e-3, 5e-3):
        run(f'noise {sd}', noise=sd)
        run(f'noise {sd}, desired', feedforward='desired', noise=sd)
        run(f'noise {sd}, filter', noise=sd, velocity_filter=15.)      # fitted, with the 15 Hz filter of the earlier controller
    run('nominal, filter', velocity_filter=15.)
if 'delay' in only:
    for d in (1, 2):
        run(f'delay {d}', delay=d)
        run(f'delay {d}, desired', feedforward='desired', delay=d)
    # which part of the controller decides under the longer delay: the feedforward or the rate filter
    run('delay 2, filter', delay=2, velocity_filter=15.)
    run('delay 2, desired, no filter', feedforward='desired', delay=2, velocity_filter=0.)
if 'half' in only:
    run('half nominal', damping_ratio=.5)
    for s in (.7, 1.3):
        run(f'half mass x{s}', mass=s, damping_ratio=.5)
    for sd in (1e-3, 5e-3):
        run(f'half noise {sd}', noise=sd, damping_ratio=.5)
    for d in (1, 2):
        run(f'half delay {d}', delay=d, damping_ratio=.5)
    run('half noise 0.001, filter', noise=1e-3, velocity_filter=15., damping_ratio=.5)
print('written', path)
