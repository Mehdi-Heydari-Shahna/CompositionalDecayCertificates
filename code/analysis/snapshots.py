"""Closed carrying configurations reconstructed from a recorded ideal-plant mission (fitted controller).

Exported payload pose, base poses and arm/lift coordinates anchor each snapshot; the ground model fills the base
height and tilt, and the assembler closes the grasps.  Snapshots are taken at evenly spaced times of the
navigation phase (all grasps latched, no payload support).
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from teamdata import RESULTS, TEAMS, Star, configuration, controller, mission_gains

TRAJ = Path(os.environ.get('TRAJECTORIES', Path(__file__).resolve().parents[2] / 'results' / 'trajectories'))
NAVIGATE = 11


def load_rows(name, run='fitted'):
    with open(TRAJ / f'pinocchio_{name}_{run}' / 'trajectory.csv') as fh:
        return list(csv.DictReader(fh))


def snapshot(name, row):
    from mission.controller import arm_joints, lift_joint
    comp, mission, ctrl = controller(name)
    q = configuration(name).copy()
    q[comp.payload['base']] = [float(row[f'payload_{k}']) for k in ('x_m', 'y_m', 'z_m', 'yaw_rad', 'pitch_rad', 'roll_rad')]
    anchored = {}
    for rid, r in comp.robots.items():
        b = np.array([float(row[f'{rid}_base_{k}']) for k in ('x_m', 'y_m', 'yaw_rad')])
        idx, value = ctrl.model.ground_pose(rid, b)
        q[r.base[[0, 1, 3]]] = b
        q[idx] = value
        for j in arm_joints(r):
            q[r.joint[j]] = float(row[f'{rid}_{j}_rad'])
            anchored[r.joint[j]] = q[r.joint[j]]
        lj = lift_joint(r)
        if lj is not None:
            q[r.joint[lj]] = float(row[f'{rid}_{lj}_m'])
    q1, _ = ctrl.asm.solve(q)
    change = max(abs(q1[i] - v) for i, v in anchored.items())
    return q1, float(change)


def snapshots(name, count=24, run='fitted'):
    rows = [r for r in load_rows(name, run) if int(float(r['phase'])) == NAVIGATE]
    pick = np.linspace(0, len(rows) - 1, count).round().astype(int)
    out = []
    for k in pick:
        q, change = snapshot(name, rows[k])
        out.append(dict(t=float(rows[k]['t_s']), q=q, closure_change=change))
    return out


if __name__ == '__main__':
    report = {}
    for name in TEAMS:
        K, *_ = mission_gains(name)
        snaps = snapshots(name)
        rates = []
        for s in snaps:
            st = Star(name, q=s['q'])
            rates.append(st.rate(K))
        np.save(RESULTS / f'snapshots_{name}.npy', np.array([s['q'] for s in snaps]))
        report[name] = dict(times=[s['t'] for s in snaps], rate_range=[min(rates), max(rates)],
                            max_closure_change=max(s['closure_change'] for s in snaps))
        print(name, json.dumps(report[name]['rate_range']), 'closure change', report[name]['max_closure_change'], flush=True)
    (RESULTS / 'snapshots.json').write_text(json.dumps(report, indent=1))
