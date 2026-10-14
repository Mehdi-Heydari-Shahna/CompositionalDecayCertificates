#!/usr/bin/env python
"""Pinocchio workflow, part 1: independent check of the compiled interfaces.

For team configurations along the mission (rack station, carrying on the floor,
on the ramp and on the plateau, mount station) and for the modes the mission
goes through, random admissible velocities and random actuator efforts are
applied and three computations are compared (for a team on conventional wheels
the velocities include wheels that slide in every direction, and the first two
computations each evaluate the declared friction law on their own):

* the compiled reduced model (tangent lift, curvature term, reduced dynamics and
  the reaction recovery of ``teamc.interfaces``);
* a constrained solve on Pinocchio inertia and bias with constraint rows written
  independently from Pinocchio frame Jacobians (``backends.pin_plant``);
* for grasp-only modes, Pinocchio's own ``constraintDynamics`` with 6-D
  constraint models between the tool and payload frames.

It also checks the inverse dynamics with prescribed grasp wrenches: the efforts
it returns, applied in the independent solve, must reproduce the prescribed
wrenches and the requested motion.

For a team on conventional wheels that last check has another form.  Such a base
gets its ground forces from friction, which the wheel speeds of the moment fix:
grasp wrenches other than those of the motion would move the bases, so they
cannot be prescribed together with the motion.  The inverse dynamics lets the
wheel rows carry what the friction law does not supply (the grip of wheels that
do not slide).  Checked are: with the grasp wrenches and the wheel loads of the
motion itself it returns the efforts that were applied; and with prescribed
grasp wrenches its efforts and reactions balance the motion in the independent
equations (Pinocchio inertia and bias, rows from Pinocchio frame Jacobians).

    python workflows/pinocchio/validate_interfaces.py --team team2 --out results/pinocchio_team2
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

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402

from backends.pin_model import _se3  # noqa: E402
from backends.pin_plant import PinPlant  # noqa: E402
from mission.controller import TeamController  # noqa: E402
from mission.runner import carry_configuration, start_configuration  # noqa: E402
from mission.route import compose  # noqa: E402
from teamc.chords import TeamAssembler  # noqa: E402
from teamc.io import INPUTS, load_team, read_json, write_json  # noqa: E402
from teamc.loads import LoadSharing  # noqa: E402

TOL = 1e-7


def configurations(comp, mission, ctrl):
    """Named closed configurations of the mission."""
    out = {}
    asm = TeamAssembler(comp, comp.seed())
    out['rack station'] = asm.solve(comp.seed())[0]
    route = mission['route']
    marks = {'carry, floor': .15, 'carry, aisle turn': .25, 'carry, ramp': None, 'carry, plateau turn': .84}
    site = comp.scene['site']
    for name, frac in marks.items():
        if frac is None:
            # as many robots as possible on the ramp, none of them across a crease (there the fitted ground
            # relation is an approximation with a numerical derivative, and a wheel changes its plane)
            terrain, best = ctrl.model.terrain, (-1, None)
            for l in np.arange(route['zones'][2][0], route['zones'][2][1], .01):
                p = ctrl.path(l)[0]
                patches = []
                for rid in comp.robots:
                    b = ctrl.base_pose_at(rid, l)
                    # the margin keeps every wheel's ground point, not only its centre, on the patch
                    margin = 0. if comp.robots[rid].ground == 'fit' else ctrl.model.wheel_radius[rid]
                    centres = ctrl.model.wheel_centres[rid]
                    corners = np.vstack([centres + [dx, dy, 0.] for dx in (-margin, margin) for dy in (-margin, margin)]) \
                        if margin else centres
                    patches.append(terrain._single_plane(b[0], b[1], b[2], corners, ctrl.model.wheel_radius[rid]))
                if all(x is not None for x in patches):
                    n = sum(x['id'] == 'ramp' for x in patches)
                    if n > best[0]:
                        best = (n, l)
            at = best[1]
        else:
            at = frac * route['length_m']
        out[name] = carry_configuration(comp, mission, ctrl, at)
    # mount station: payload on the supports, bases at their stations.  A team whose bases keep their place on
    # the payload stands there as at the rack; bases that cannot move sideways have turned to face the supports
    # (formation.place), and their arms are seeded from the end of the route
    steered = 'place' in mission['formation']
    q = (carry_configuration(comp, mission, ctrl, route['length_m']) if steered else out['rack station']).copy()
    c = comp.connections[mission['rests']['place']]
    from teamc.se3 import inv, pose_chart
    pose = pose_chart(np.asarray(c['T1']) @ inv(c['T2']), yaw_near=q[comp.payload['base'][3]] if steered else None)
    q[comp.payload['base']] = pose
    for rid, r in comp.robots.items():
        b = compose([pose[0], pose[1], pose[3]], mission['formation']['place' if steered else 'station'][rid])
        idx, value = ctrl.model.ground_pose(rid, b)
        q[r.base[[0, 1, 3]]] = b
        q[idx] = value
    out['mount station'] = TeamAssembler(comp, q).solve(q)[0]
    return out


def compare(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b)) / max(1., float(np.max(np.abs(b))))) if a.size else 0.


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team', default='team2')
    ap.add_argument('--out', default=None)
    ap.add_argument('--samples', type=int, default=3)
    args = ap.parse_args(argv)
    out = Path(args.out or ROOT / 'results' / f'pinocchio_{args.team}')
    comp = load_team(args.team)
    mission = read_json(INPUTS / 'missions' / f'{args.team}.json')
    ctrl = TeamController(comp, mission, .002)
    model, tree = ctrl.model, comp.tree
    plant = PinPlant(comp)
    grasps = model.grasp_ids
    rng = np.random.default_rng(7)
    rows, worst = [], {}

    def note(key, value):
        worst[key] = max(worst.get(key, 0.), value)

    for cname, q in configurations(comp, mission, ctrl).items():
        on_rack, on_mount = cname == 'rack station', cname == 'mount station'
        modes = [('carry', model.mode())]
        if on_rack or on_mount:
            rest = mission['rests']['pickup' if on_rack else 'place']
            modes = [('supports + all grasps', model.mode(rest=rest)), ('supports + one grasp', model.mode(grasps=grasps[-1:], rest=rest)),
                     ('supports, arms free', model.mode(grasps=(), rest=rest))]
        for mname, mode in modes:
            for _ in range(args.samples):
                lift = model.lift(q, mode)
                u = rng.uniform(-.25, .25, lift.E.shape[1])
                f = rng.uniform(-1., 1., model.nf) * .15 * model.effort_bounds[:, 1]
                c = model.forward(q, u, f, mode)
                p = plant.evaluate(q, c['v'], f, latched=mode.grasps, rest=mode.rest)
                e = dict(configuration=cname, mode=mname, mobility=int(lift.E.shape[1]), rows=int(lift.constraints.J.shape[0]),
                         rcond=lift.rcond, closure_residual=float(np.max(np.abs(p['residual']))),
                         velocity_residual=float(np.max(np.abs(p['velocity_residual']))),
                         acceleration=compare(c['a'], p['a']))
                for key, lam in c['ports'].items():
                    e[f'reaction:{key}'] = compare(lam, p['ports'][key])
                # inverse dynamics with prescribed grasp wrenches -> efforts -> independent forward solve
                if mode.grasps:
                    ls = LoadSharing(model, mode.grasps)
                    G = ls.grasp_matrix(lift.constraints)
                    internal = rng.normal(size=6 * len(mode.grasps)) * 8.
                    need = tree.rnea(q, c['v'], c['a'])[model.payload] if mode.rest is None else \
                        .6 * tree.rnea(q, np.zeros(tree.n), np.zeros(tree.n))[model.payload]
                    given = ls.distribute(G, need, internal)
                    inv_ = model.inverse(q, c['v'], c['a'], mode, grasp_wrenches=given)
                    if any(model.friction.values()):
                        same = model.inverse(q, c['v'], c['a'], mode, grasp_wrenches={g: c['ports'][g] for g in mode.grasps},
                                             wheel_loads=c['wheel_loads'])
                        e['inverse:efforts'] = compare(same['f'], f)
                        e['inverse:balance'] = plant.balance(q, c['v'], c['a'], inv_['f'], inv_['ports'], latched=mode.grasps,
                                                             rest=mode.rest) / max(1., float(np.max(np.abs(inv_['f']))))
                    else:
                        back = plant.evaluate(q, c['v'], inv_['f'], latched=mode.grasps, rest=mode.rest)
                        e['inverse:acceleration'] = compare(back['a'], c['a'])
                        e['inverse:grasp_wrenches'] = max(compare(back['ports'][g], given[g]) for g in mode.grasps)
                    e['inverse:residual'] = inv_['residual']
                rows.append(e)
                for key, value in e.items():
                    if key.startswith(('acceleration', 'reaction', 'inverse')) and key != 'inverse:residual':
                        note(key.split(':')[0] if key.startswith('reaction') else key, value)
                        if key.startswith('reaction'):
                            kind = 'grasp' if key.split(':', 1)[1] in grasps else key.split(':')[-1] if ':' in key.split(':', 1)[1] else 'rest'
                            note(f'reaction ({kind})', value)

    # ------------------------------------------------------------------ Pinocchio's own constraint dynamics (grasps only)
    native = []
    q = configurations(comp, mission, ctrl)['carry, floor']
    mode = model.mode(ground=False, rolling=False)
    pm = plant.model
    cmodels = []
    for cid in mode.grasps:
        c = comp.connections[cid]
        j1, B1 = plant.pt.body[c['body1']]
        j2, B2 = plant.pt.body[c['body2']]
        cmodels.append(pin.RigidConstraintModel(pin.ContactType.CONTACT_6D, pm, j1, _se3(B1 @ c['T1']), j2, _se3(B2 @ c['T2']), pin.LOCAL))
    cdatas = [cm.createData() for cm in cmodels]
    data = pm.createData()
    pin.initConstraintDynamics(pm, data, cmodels)
    prox = pin.ProximalSettings(1e-13, 1e-10, 100)
    for _ in range(max(args.samples, 3)):
        lift = model.lift(q, mode)
        u = rng.uniform(-.25, .25, lift.E.shape[1])
        f = rng.uniform(-1., 1., model.nf) * .15 * model.effort_bounds[:, 1]
        c = model.forward(q, u, f, mode)
        tau = model.B @ f - model.damping * c['v']
        ddq = pin.constraintDynamics(pm, data, q, c['v'], tau, cmodels, cdatas, prox)
        native.append(compare(ddq, c['a']))
    worst['pinocchio constraintDynamics: acceleration'] = max(native)

    report = dict(team=args.team, tolerance=TOL, worst_relative_difference=worst, cases=rows,
                  ok=bool(all(v <= TOL for k, v in worst.items())),
                  versions=dict(pinocchio=pin.__version__, numpy=np.__version__),
                  note='differences are max|a - b| / max(1, max|b|) between the compiled model and the independent solve')
    write_json(out / 'interfaces.json', report)
    print(f'{args.team}: {len(rows)} cases over {len(set(r["configuration"] for r in rows))} configurations')
    for key, value in sorted(worst.items()):
        print(f'  {key:48s} {value:.2e}')
    print('  smallest dependent-block rcond', f"{min(r['rcond'] for r in rows):.3e}")
    print('PASS' if report['ok'] else 'FAIL', '(tolerance %.0e)' % TOL, '->', out / 'interfaces.json')
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
