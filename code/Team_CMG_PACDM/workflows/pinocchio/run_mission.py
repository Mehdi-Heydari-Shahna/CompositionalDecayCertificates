#!/usr/bin/env python
"""Pinocchio workflow, part 2: the mission on the ideal-constraint plant.

    python workflows/pinocchio/run_mission.py --team team2 --out results/pinocchio_team2

The plant integrates Pinocchio dynamics under exact constraints (rigid grasps,
rolling without slip, rigid supports that release when they would have to pull).
It has no contact solver, so the audit of this run shows the controller and the
compiled interfaces without contact compliance, vibration or slip.

A base on conventional wheels (skid steering) is the exception.  Its wheels are
not constrained to roll: they turn under their motors and the tyre model of
``backends/tyre.py``.  And it is not carried by rigid ground rows: it stands on
the declared normal stiffness and damping of its tyres (``--skid-ground rigid``
steps it on rigid ground instead; DESIGN.md says what that does).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# small matrices: one BLAS thread is faster, and several runs can share a machine (set before NumPy is imported)
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

from backends.pin_plant import PinPlant  # noqa: E402
from mission.workflow import run_and_report  # noqa: E402
from teamc.io import INPUTS, load_team, read_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team', default='team2')
    ap.add_argument('--out', default=None)
    ap.add_argument('--timestep', type=float, default=.002)
    ap.add_argument('--slow-every', type=int, default=None, help='physics steps per reference/feedforward update (default: 10 ms)')
    ap.add_argument('--stop-after', default=None)
    ap.add_argument('--skid-ground', choices=('tyres', 'rigid'), default='tyres',
                    help='bases on conventional wheels: on the normal compliance of their tyres, or on rigid ground with the '
                         'wheel loads of the step before (kept to reproduce the rocking of husky3 described in DESIGN.md)')
    ap.add_argument('--feedforward', choices=('desired', 'fitted'), default=None,
                    help='desired: inverse dynamics of the desired motion (the default of the mission controller); '
                         'fitted: the effort required by the fitted reference of the loops (rolling wheels; DESIGN.md)')
    ap.add_argument('--feedforward-rate', choices=('slow', 'fast'), default=None,
                    help='fitted effort: evaluated with the references (default) or in every step')
    ap.add_argument('--monitor', action='store_true',
                    help='record the quantities of the stability statement (rolling wheels; always with --feedforward fitted)')
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE',
                    help='another option of the mission controller (mission/controller.py, DEFAULTS), the value in JSON; '
                         'for example --option arm_damping_ratio=0.7.  May be given several times')
    ap.add_argument('--velocity-filter', type=float, default=None, metavar='HZ',
                    help='low-pass of the joint rates in the damping term (default: 15 Hz with --feedforward desired, none '
                         'with fitted, where the stability statement is for the measured rates)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    controller = {key: value for key, value in (('feedforward', args.feedforward), ('feedforward_rate', args.feedforward_rate))
                  if value is not None}
    if args.monitor or args.feedforward == 'fitted':
        controller['monitor'] = True
    if args.velocity_filter is not None:
        controller['velocity_filter_hz'] = args.velocity_filter
    for item in args.option:
        key, _, value = item.partition('=')
        if not key or not value:
            ap.error(f'--option needs KEY=VALUE, got {item!r}')
        import json
        controller[key] = json.loads(value)
    out = Path(args.out or ROOT / 'results' / f'pinocchio_{args.team}')
    comp = load_team(args.team)
    mission = read_json(INPUTS / 'missions' / f'{args.team}.json')
    plant = PinPlant(comp, timestep=args.timestep, skid_ground=args.skid_ground)
    import pinocchio
    slow_every = args.slow_every or max(int(round(.01 / args.timestep)), 1)
    report, *_ = run_and_report(plant, comp, mission, out, f'Pinocchio (ideal constraints) {args.team}:',
                                options=dict(slow_every=slow_every, stop_after=args.stop_after, controller=controller),
                                quiet=args.quiet,
                                plant_info=dict(engine=f'Pinocchio {pinocchio.__version__}, ideal constraints',
                                                **(dict(skid_ground=args.skid_ground) if plant.tyres else {}),
                                                **({'controller': controller} if controller else {})))
    print('largest constraint errors before projection:', plant.stats)
    return 0 if report['bounds']['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
