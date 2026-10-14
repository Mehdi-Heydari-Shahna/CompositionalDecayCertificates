"""Wheel constraint rows against the motion of the wheel's material point at the ground.

A row of ``TeamModel.wheel_rows`` must be the velocity, along one ground direction, of the
point of the wheel body that is at the ground point at this instant.  That velocity is taken
here from central differences of the compiled forward kinematics, independently of the twist
maps the rows are built from.

For a mecanum wheel the row is a rolling constraint.  For the conventional wheels of a base
that turns by sliding them (Husky) the same rows are not constraints: they are the slip of
the wheel, which the friction law takes (``teamc/skid.py``).  Conventional wheels as
constraints are compiled here from a mecanum record with the wheel type changed, which
checks the rows only.
"""
from copy import deepcopy

import numpy as np
import pytest

from teamc.compiler import compile_team
from teamc.interfaces import TeamModel
from teamc.io import load_records


def material_velocity(tree, q, v, body, point, h=1e-6):
    """World velocity of the point of ``body`` that is at ``point`` now, moving with the generalized velocity ``v``."""
    local = np.linalg.solve(tree.poses(q)[body], np.r_[point, 1.])
    return ((tree.poses(q + h * v)[body] - tree.poses(q - h * v)[body]) @ local)[:3] / (2. * h)


def unit_in_plane(g, normal):
    g = g - (g @ normal) * normal
    return g / np.linalg.norm(g)


def expected_directions(wheel, R, normal):
    """Ground directions a wheel constrains, in the order of its rows (declared independently of the rows)."""
    axis = R @ wheel['axis_base']
    if wheel['type'] == 'mecanum':
        return [('traction', unit_in_plane(R @ wheel['traction_direction_base'], normal))]
    out = [('rolling', unit_in_plane(np.cross(axis, normal), normal))]
    if wheel.get('lateral', 'constraint') == 'constraint':
        out.append(('lateral', unit_in_plane(axis, normal)))
    return out


def check_rows(comp, q, seed):
    model, tree = TeamModel(comp), comp.tree
    rng = np.random.default_rng(seed)
    kin = tree.sweep(q)
    worst = 0.
    for rid, robot in comp.robots.items():
        Tb = kin[0][model.base_body[rid]]
        R, origin = Tb[:3, :3], Tb[:3, 3]
        normal = R[:, 2]
        rows, labels = model.wheel_rows(q, rid, kin)
        want = []
        for w in robot.wheels:
            ground_point = origin + R @ w['centre_base'] - w['radius_m'] * normal
            body = tree.names[tree.body_of_joint[robot.prefix + w['joint_id']]]
            want += [(f"{w['id']}:{label}", g, body, ground_point) for label, g in expected_directions(w, R, normal)]
        assert labels == [x[0] for x in want]
        for _ in range(4):
            v = rng.normal(size=tree.n)
            for row, (_, g, body, point) in zip(rows, want):
                worst = max(worst, abs(row @ v - g @ material_velocity(tree, q, v, body, point)))
    return worst


def test_mecanum_rows_are_the_ground_point_velocity_along_the_roller_axis(closed, mecanum_team):
    c = closed(mecanum_team)
    # the hub point at the ground and the roller point there differ by a spin about the roller axis,
    # which has no component along that axis: the hub suffices
    assert check_rows(c['comp'], c['carry'], 11) < 1e-7


def test_slip_rows_of_a_skid_steered_base_are_the_ground_point_velocity(closed, skid_team):
    c = closed(skid_team)
    assert check_rows(c['comp'], c['carry'], 13) < 1e-7              # two per wheel: along the rolling direction and the axle
    model, comp = c['model'], c['comp']
    rows, labels = model.wheel_rows(c['carry'], 'r1', comp.tree.sweep(c['carry']))
    assert len(labels) == 8
    # they are not constraints of the mode: every base keeps its three planar freedoms and its four wheel rates
    lift = model.lift(c['carry'], model.mode())
    blocks = [b for b in lift.constraints.blocks if b['id'].endswith(':wheels')]
    assert len(blocks) == len(comp.robots) and all(b['slip'] and not b['holonomic'] for b in blocks)
    for robot in comp.robots.values():
        free = set(int(i) for i in robot.base[[0, 1, 3]]) | {int(w['index']) for w in robot.wheels}
        assert free <= set(int(i) for i in lift.independent)
    assert not set(np.concatenate([np.arange(b['rows'].start, b['rows'].stop) for b in blocks])) & set(int(i) for i in lift.rows)


@pytest.mark.parametrize('lateral', ['constraint', 'free'])
def test_conventional_wheel_rows(lateral):
    team, robots, payload, scene = load_records('team2')
    robots = deepcopy(robots)
    for record in robots.values():
        for w in record['wheels']:
            w['type'] = 'fixed'
            w['lateral'] = lateral
            w.pop('traction_direction_base', None)
    comp = compile_team(team, robots, payload, scene)
    q = comp.seed()
    rng = np.random.default_rng(2)
    for robot in comp.robots.values():                       # a tilted base, turned wheels: no special configuration
        q[robot.base[3:]] += rng.uniform(-.2, .2, 3)
        q[[w['index'] for w in robot.wheels]] += rng.uniform(-1., 1., len(robot.wheels))
    assert check_rows(comp, q, 12) < 1e-7
    rows, labels = TeamModel(comp).wheel_rows(q, 'r1', comp.tree.sweep(q))
    assert len(labels) == (8 if lateral == 'constraint' else 4)
    # pure spin of one wheel moves its ground point along the rolling direction by radius times rate, and not sideways
    w = comp.robots['r1'].wheels[0]
    spin = np.zeros(comp.tree.n)
    spin[w['index']] = 1.
    values = dict(zip(labels, rows @ spin))
    assert abs(abs(values[f"{w['id']}:rolling"]) - w['radius_m']) < 1e-12
    if lateral == 'constraint':
        assert abs(values[f"{w['id']}:lateral"]) < 1e-12
