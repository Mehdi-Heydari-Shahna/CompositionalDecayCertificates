#!/usr/bin/env python
"""Write the trajectories of a recorded run as a table (CSV) and as arrays (NPZ), at a chosen rate.

    python tools/export_trajectory.py --record results/mujoco_team2/record.npz --rate 10

A record holds every compiled coordinate at 50 Hz and is tens of megabytes.  This tool keeps what describes the
motion of the team, with one column per quantity and a header that names it:

* time, phase and mode;
* the payload: its pose (chart x, y, z, yaw, pitch, roll) and the reference pose;
* every robot: the planar pose of its base (x, y, yaw) and its reference, the lift and the arm joints;
* the wrenches the wrist sensors give at the grasps (tool axes, moment then force);
* the applied efforts of the actuator ports (interval means);
* where the run recorded them, the quantities of the stability statement of the loops (``fit:*``).

Written beside the record (or to ``--out``): ``trajectory.csv``, ``trajectory.npz`` and ``trajectory_columns.txt``.
No simulator is needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission.controller import arm_joints, lift_joint  # noqa: E402
from mission.report import load_record  # noqa: E402
from teamc.io import load_team  # noqa: E402


def columns(comp, A):
    """``(name, values)`` for every column, in the order of the table."""
    ids = comp.cmg['coordinate_ids']
    pb = np.asarray(comp.payload['base'], int)
    out = [('t_s', A['t']), ('phase', np.asarray(A['phase'], float))]
    chart = ('x_m', 'y_m', 'z_m', 'yaw_rad', 'pitch_rad', 'roll_rad')
    out += [(f'payload_{c}', A['q'][:, pb[k]]) for k, c in enumerate(chart)]
    out += [(f'payload_ref_{c}', A['payload_ref'][:, k]) for k, c in enumerate(chart)]
    for rid, r in comp.robots.items():
        planar = r.base[[0, 1, 3]]
        for k, c in enumerate(('x_m', 'y_m', 'yaw_rad')):
            out.append((f'{rid}_base_{c}', A['q'][:, planar[k]]))
        for k, c in enumerate(('x_m', 'y_m', 'yaw_rad')):
            out.append((f'{rid}_base_ref_{c}', A[f'base_ref:{rid}'][:, k]))
        name = lift_joint(r)
        if name is not None:
            out.append((f'{rid}_{name}_m', A['q'][:, r.joint[name]]))
        for j in arm_joints(r):
            out.append((f'{rid}_{j}_rad', A['q'][:, r.joint[j]]))
    for cid, c in comp.connections.items():
        if c['kind'] == 'grasp' and f'lam_meas:{cid}' in A:
            for k, name in enumerate(('mx_Nm', 'my_Nm', 'mz_Nm', 'fx_N', 'fy_N', 'fz_N')):
                out.append((f'{cid}_{name}', A[f'lam_meas:{cid}'][:, k]))
    for k, p in enumerate(comp.cmg['ports']):
        out.append((f"effort_{p['id']}", A['f'][:, k]))
    if 'fit:W' in A:
        out += [('loops_error_energy_J', A['fit:W']), ('loops_rate_min_1_s', A['fit:rate'][:, 0]),
                ('loops_rate_max_1_s', A['fit:rate'][:, 1]), ('loops_internal_effort_ratio', A['fit:internal'])]
    assert len({name for name, _ in out}) == len(out)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', required=True, type=Path)
    ap.add_argument('--team', default=None, help='team name (default: read from the record)')
    ap.add_argument('--rate', type=float, default=10., help='rows per second of simulated time (at most that of the record)')
    ap.add_argument('--out', default=None, type=Path, help='output folder (default: the folder of the record)')
    args = ap.parse_args(argv)
    A, info, _ = load_record(args.record)
    team = args.team or (info or {}).get('team')
    if not team:
        ap.error('the record does not name its team; pass --team')
    comp = load_team(team)
    interval = float(np.median(np.diff(A['t'])))
    step = max(int(round(1. / (args.rate * interval))), 1)
    cols = columns(comp, A)
    table = np.column_stack([np.asarray(values, float)[::step] for _, values in cols])
    names = [name for name, _ in cols]
    out = args.out or args.record.parent
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / 'trajectory.csv', table, delimiter=',', header=','.join(names), comments='', fmt='%.9g')
    np.savez_compressed(out / 'trajectory.npz', columns=np.array(names), table=table,
                        mode=np.asarray(A['mode'])[::step], team=np.array(team))
    phases = (info or {}).get('phases') or []
    lines = [f'team {team}; {len(table)} rows at {1. / (step * interval):.3g} rows per second; '
             f'{len(names)} columns', '', 'phase numbers: ' + ', '.join(f'{k} {name}' for k, name in enumerate(phases)), '']
    lines += [f'{k + 1:4d}  {name}' for k, name in enumerate(names)]
    (out / 'trajectory_columns.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(table)} rows, {len(names)} columns written to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
