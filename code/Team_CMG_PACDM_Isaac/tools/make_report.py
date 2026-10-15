#!/usr/bin/env python
"""Audit a recorded run again and draw its figures (any plant; no simulator needed).

    python tools/make_report.py --record results/isaac_team2/.../mission/record.npz

The record written by every workflow holds the measured states, the applied efforts,
the wrist-sensor readings, the plant contact rows and the run information.  This tool
recompiles the team, registers the as-latched grasp frames from the recorded events
and repeats the audit of ``mission/audit.py``.  It is the way to get figures for a run
made where matplotlib is not installed (the Isaac Sim Python, for example).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission.audit import audit  # noqa: E402
from mission.report import load_record, save  # noqa: E402
from teamc.interfaces import TeamModel  # noqa: E402
from teamc.io import INPUTS, load_team, read_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', required=True, type=Path)
    ap.add_argument('--team', default=None, help='team name (default: read from the record)')
    ap.add_argument('--out', default=None, type=Path, help='output folder (default: <record folder>/reaudit)')
    ap.add_argument('--title', default=None)
    args = ap.parse_args(argv)
    A, info, contacts = load_record(args.record)
    if info is None:
        ap.error('this record has no run information (written by an older version)')
    team = args.team or info.get('team')
    if not team:
        ap.error('the record does not name its team; pass --team')
    comp = load_team(team)
    mission = read_json(INPUTS / 'missions' / f'{team}.json')
    model = TeamModel(comp)
    for e in info['events']:
        if e['event'] == 'latched' and 'T2' in e:
            model.register_grasp(e['id'], e['T2'])
    report, series = audit(comp, mission, A, info, contacts, model=model)
    report['plant'] = info.get('plant', {})
    report['team'] = mission['team']
    out = args.out or args.record.parent / 'reaudit'
    plant = info.get('plant', {})
    title = args.title or f"{plant.get('engine') or plant.get('name', 'plant')}, {team}:"
    lines, files = save(out, comp, mission, A, report, series, info, contacts, title=title, arrays=False)
    print('\n'.join(lines))
    print('written to', out)
    return 0 if report['bounds']['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
