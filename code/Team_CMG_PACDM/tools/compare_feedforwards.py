#!/usr/bin/env python
"""Table of mission runs made with the two feedforwards of the controller (Markdown).

    python tools/compare_feedforwards.py --runs reference_results --out reference_results/FEEDFORWARDS.md

Looks for run folders ``<plant>_<team>_desired``, ``<plant>_<team>_fitted``, ``<plant>_<team>_fitted05`` (fitted,
with the rate gains of the arm and lift joints halved: ``--option arm_damping_ratio=0.5 --option lift_damping_ratio=0.5``)
and ``<plant>_<team>_fitted15`` (fitted, with the 15 Hz rate filter in the damping term), and the comparison runs
``<plant>_<team>_desired0`` (the default controller without its rate filter: ``--monitor --velocity-filter 0``),
``<plant>_<team>_fitted0515`` (fitted, halved rate gains and the 15 Hz filter) and ``<plant>_<team>_desiredR`` (the
default controller with the damping ratios of the arm joints raised for the inertia they see in the closed chain:
``--option arm_damping_ratio=[...]``) under ``--runs`` and reads their ``report.json``.  Every
number of the table is one of the report; nothing is simulated.  Where the folder ``<plant>_<team>`` of an earlier
release is found beside them, the report of the ``desired`` run is compared with it entry by entry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3')
PLANTS = (('pinocchio', 'Pinocchio, ideal constraints'), ('mujoco', 'MuJoCo'))
KINDS = (('desired', 'desired motion'), ('fitted', 'fitted reference'),
         ('fitted05', 'fitted reference, rate gains sqrt(k_p m) on the arm and lift joints'),
         ('fitted15', 'fitted reference, rates of the damping term low-passed at 15 Hz'),
         ('desired0', 'desired motion, damping term without the 15 Hz filter'),
         ('fitted0515', 'fitted reference, rate gains sqrt(k_p m), rates of the damping term low-passed at 15 Hz'),
         ('desiredR', 'desired motion, arm damping designed for the inertia the joints see'))


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + '.'))
        else:
            out[pre + k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True, type=Path)
    ap.add_argument('--earlier', type=Path, default=None, help='folder with the reports of an earlier release (default: --runs)')
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args(argv)
    earlier = args.earlier or args.runs
    lines = ['| Plant | Team | Controller | Payload error max / rms (mm) | Tilt max (mrad) | Base error max (mm) | Internal force max / rms (N) | '
             'Internal moment max (N m) | Effort ratio max | Error energy of the loops, median / 99 % (J) | Bounds |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    same, found = [], 0
    for plant, plant_name in PLANTS:
        for team in TEAMS:
            for kind, kind_name in KINDS:
                path = args.runs / f'{plant}_{team}_{kind}' / 'report.json'
                if not path.exists():
                    continue
                found += 1
                r = json.loads(path.read_text(encoding='utf-8'))
                tr, g, lp = r['tracking'], r['grasp'], r.get('loops') or {}
                e = lp.get('error_energy_J') or {}
                base = max(b['position_error_m']['max'] for b in tr['base'].values())
                unmet = [f"{k} ({c['value']:.3g} against {c['bound']:.3g})" for k, c in r['bounds']['checks'].items() if not c['ok']]
                lines.append(f"| {plant_name} | `{team}` | {kind_name} | {1e3 * tr['payload_position_error_m']['max']:.2f} / "
                             f"{1e3 * tr['payload_position_error_m']['rms']:.2f} | "
                             f"{1e3 * tr['payload_tilt_rad']['max']:.2f} | {1e3 * base:.1f} | "
                             f"{g['internal_force_N']['max']:.1f} / {g['internal_force_N']['rms']:.2f} | {g['internal_moment_Nm']['max']:.1f} | "
                             f"{r['efforts']['max_ratio']:.2f} | "
                             + (f"{e['median']:.1e} / {e['p99']:.1e}" if e else '-') + ' | '
                             + ('all met' if not unmet else 'NOT met: ' + ', '.join(unmet)) + ' |')
                if kind == 'desired' and (earlier / f'{plant}_{team}' / 'report.json').exists():
                    old = flat(json.loads((earlier / f'{plant}_{team}' / 'report.json').read_text(encoding='utf-8')))
                    new = flat(r)
                    common = [k for k in old if k in new and not k.startswith('timing')]
                    differ = [k for k in common if old[k] != new[k]]
                    same.append((plant, team, len(common), differ, [k for k in old if k not in new]))
    if not found:
        ap.error(f'no run folders found under {args.runs}')
    lines.append('')
    for plant, team, count, differ, lost in same:
        lines.append(f'- `{plant}_{team}_desired` against the stored report `{plant}_{team}`: '
                     + (f'{count} common entries (timing apart), all equal' if not differ and not lost else
                        f'{len(differ)} of {count} common entries DIFFER ({differ[:5]}), {len(lost)} entries missing'))
    text = '\n'.join(lines) + '\n'
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    print(text)
    return 1 if any(differ or lost for *_, differ, lost in same) else 0


if __name__ == '__main__':
    sys.exit(main())
