"""A base whose wheel contacts are determinate: wheels on rockers, one contact row per wheel.

The Ridgeback has its four wheels on two rockers, the front one on a passive pivot.  Its ground
coupling is compiled as one ideal contact row per wheel, with the rocker angle among the
dependent coordinates, and the reactions of those rows are the wheel loads.
"""
from copy import deepcopy

import numpy as np
import pytest

from teamc.compiler import ModelError, compile_team
from teamc.interfaces import TeamModel
from teamc.io import load_records, load_team
from teamc.se3 import chart_pose
from teamc.stability import ground_loads, support_points
from teamc.terrain import Terrain

TEAM = 'ridgeback2'


@pytest.fixture(scope='module')
def rb():
    comp = load_team(TEAM)
    return comp, TeamModel(comp)


def test_the_compiler_takes_the_rocker_as_a_passive_joint(rb):
    comp, model = rb
    for robot in comp.robots.values():
        assert robot.passive == ['front_rocker'] and robot.ground == 'contact'
        assert robot.commanded == [] and len(robot.dependent) == 6          # no lift: the arm closes the grasp alone
        front = [w for w in robot.wheels if w['suspension']]
        rear = [w for w in robot.wheels if not w['suspension']]
        assert len(front) == 2 and len(rear) == 2
        assert all(w['carrier'].endswith('front_rocker_link') for w in front)
        assert all(w['carrier'].endswith('rear_rocker_link') for w in rear)
        assert not any(p['index'] == robot.joint['front_rocker'] for p in comp.cmg['ports'])
    assert comp.plan['robots']['r1']['ground_model'] == 'contact'
    # the first platform is untouched by this: a rigid base, a fitted plane
    old = load_team('team2')
    assert all(r.ground == 'fit' and not r.passive for r in old.robots.values())
    assert 'ground_model' not in old.plan['robots']['r1']


def test_what_the_compiler_refuses():
    team, robots, payload, scene = load_records(TEAM)
    locked = deepcopy(robots)
    record = locked['ridgeback_ur10e']
    record['wheels'] = record['wheels'][:3]                                 # three wheels and a rocker: one contact short
    with pytest.raises(ModelError, match='statically determinate'):
        compile_team(team, locked, payload, scene)
    loose = deepcopy(robots)
    record = loose['ridgeback_ur10e']
    record['actuators'] = [a for a in record['actuators'] if a['joint_id'] != 'arm_wrist_3_joint']
    with pytest.raises(ModelError, match='carry no wheel'):                 # an unactuated arm joint is not a suspension
        compile_team(team, loose, payload, scene)
    fixed = deepcopy(robots)
    record = fixed['ridgeback_ur10e']
    next(j for j in record['joints'] if j['id'] == 'front_rocker')['type'] = 'fixed'
    record['limits'].pop('front_rocker')
    record['seed']['joints'].pop('front_rocker')
    record['drive'].pop('front_rocker')
    comp = compile_team(team, fixed, payload, scene)                        # both rockers fixed: a rigid four-wheel base
    assert all(r.ground == 'fit' and not r.passive for r in comp.robots.values())
    assert comp.tree.n == load_team(TEAM).tree.n - len(comp.robots)


def test_wheel_contact_on_planes_and_creases():
    comp = load_team(TEAM)
    terrain = Terrain(comp.scene)
    site = comp.scene['site']
    r = .0759
    axis = np.array([1., 0., 0.])
    h, n, p = terrain.wheel_contact([0., 0., r + .01], axis, r)
    assert abs(h - .01) < 1e-15 and np.allclose(n, [0, 0, 1]) and np.allclose(p, [0., 0., .01])
    # a tilted wheel touches with its rim: the clearance of the disc, not of its centre
    tilt = .05
    h, _, p = terrain.wheel_contact([0., 0., r], [np.cos(tilt), 0., np.sin(tilt)], r)
    assert abs(h - r * (1. - np.cos(tilt))) < 1e-15 and abs(p[2] - h) < 1e-15
    # concave crease (floor to ramp): the wheel meets the ramp before its centre is over it, and never goes below either plane
    x, y0 = site['x_c'], site['y_ramp0']
    slope = np.arctan(site['ramp_rise'] / (site['y_ramp1'] - site['y_ramp0']))
    meet = r * np.tan(slope / 2.)
    before = terrain.wheel_contact([x, y0 - 3. * meet, r], axis, r)
    assert abs(before[0]) < 1e-15 and before[1][2] == 1.
    at = terrain.wheel_contact([x, y0 - .5 * meet, r], axis, r)
    assert at[0] < 0. and at[1][1] < 0.                                     # resting on the floor here would cut into the ramp
    heights = [terrain.wheel_contact([x, y, r], axis, r)[0] for y in np.linspace(y0 - .01, y0 + .01, 201)]
    assert np.max(np.abs(np.diff(heights))) < 2e-5                          # continuous across the crease
    # convex edge (ramp to plateau): the two planes are joined with an error far below a tenth of a millimetre
    y1, rise = site['y_ramp1'], site['ramp_rise']
    heights = [terrain.wheel_contact([x, y, rise + r], axis, r)[0] for y in np.linspace(y1 - .01, y1 + .01, 201)]
    assert np.max(np.abs(np.diff(heights))) < 2e-5
    with pytest.raises(ValueError, match='No terrain patch'):
        terrain.wheel_contact([x + 50., 100., r], axis, r)


def _on_the_crease(comp, model, yaw):
    """A base that straddles the foot of the ramp at an angle: its wheels stand on two planes."""
    site = comp.scene['site']
    planar = np.array([site['x_c'], site['y_ramp0'] + .1, yaw])
    idx, value = model.ground_pose('r1', planar)
    q = comp.seed()
    r = comp.robots['r1']
    q[r.base[[0, 1, 3]]] = planar
    q[idx] = value
    return q, value


def test_the_rocker_keeps_four_wheels_on_two_planes(rb):
    comp, model = rb
    q, value = _on_the_crease(comp, model, .5)
    kin = comp.tree.sweep(q)
    contacts = model.wheel_contacts(q, 'r1', kin)
    assert max(abs(c['height']) for c in contacts) < 1e-11                  # every wheel touches
    assert len({tuple(np.round(c['normal'], 9)) for c in contacts}) == 2    # on two different planes
    assert abs(value[3]) > 2e-3                                             # which takes rocker travel ...
    limits = comp.cmg['limits']['r1_front_rocker']
    assert limits['lower'] < value[3] < limits['upper']                     # ... inside its stops
    # squarely across the same crease the four ground points are coplanar and the rocker stays level
    _, square = _on_the_crease(comp, model, np.pi / 2)
    assert abs(square[3]) < 1e-9
    # on one plane the pose is the closed form of the fitted model with the rocker at zero
    idx, flat = model.ground_pose('r1', [0., 0., .7])
    assert np.allclose(flat, [model.base_height['r1'], 0., 0., 0.], atol=1e-13)
    assert list(idx) == list(model.ground_dep['r1'])


def test_ground_rows_are_the_rates_of_the_wheel_heights(rb):
    comp, model = rb
    q, _ = _on_the_crease(comp, model, .5)
    tree = comp.tree
    rng = np.random.default_rng(5)
    residual, J, _ = model.ground_rows(q, 'r1', tree.sweep(q))
    assert np.max(np.abs(residual)) < 1e-11
    r = comp.robots['r1']
    own = np.r_[r.base, model.passive['r1'], [w['index'] for w in r.wheels]]
    for _ in range(4):
        v = np.zeros(tree.n)
        v[own] = rng.normal(size=len(own))
        h = 1e-7
        up = model.ground_rows(q + h * v, 'r1')[0]
        down = model.ground_rows(q - h * v, 'r1')[0]
        assert np.allclose(J @ v, (up - down) / (2. * h), atol=2e-6)
    assert np.max(np.abs(J[:, [w['index'] for w in r.wheels]])) == 0.       # spinning a wheel does not lift it


def test_wheel_rows_are_the_ground_point_velocity_along_the_roller_axis(rb):
    comp, model = rb
    q, _ = _on_the_crease(comp, model, .5)
    q[comp.robots['r1'].base[5]] += .01                                     # not even a consistent pose: the rows are kinematics
    tree = comp.tree
    kin = tree.sweep(q)
    robot = comp.robots['r1']
    contacts = model.wheel_contacts(q, 'r1', kin)
    rows, labels = model.wheel_rows(q, 'r1', kin, contacts)
    assert labels == [f"{w['id']}:traction" for w in robot.wheels]
    rng = np.random.default_rng(6)
    for row, w, c in zip(rows, robot.wheels, contacts):
        body = robot.prefix + w['body']
        Rc = kin[0][tree.index[w['carrier']]][:3, :3]
        g = Rc @ w['traction_direction_carrier']
        g = g - (g @ c['normal']) * c['normal']
        g /= np.linalg.norm(g)
        local = np.linalg.solve(tree.poses(q)[body], np.r_[c['point'], 1.])
        for _ in range(3):
            v = rng.normal(size=tree.n)
            h = 1e-6
            velocity = ((tree.poses(q + h * v)[body] - tree.poses(q - h * v)[body]) @ local)[:3] / (2. * h)
            assert abs(row @ v - g @ velocity) < 1e-7


def test_base_motion_keeps_every_wheel_on_the_ground_and_rolling(rb):
    comp, model = rb
    q, _ = _on_the_crease(comp, model, .5)
    kin = comp.tree.sweep(q)
    idx, S, W = model.base_motion(q, 'r1', kin)
    robot = comp.robots['r1']
    assert list(idx) == list(robot.base) + [robot.joint['front_rocker']]
    v = np.zeros(comp.tree.n)
    u = np.array([.3, -.2, .4])
    v[idx] = S @ u
    v[[w['index'] for w in robot.wheels]] = W @ u
    _, J, contacts = model.ground_rows(q, 'r1', kin)
    assert np.max(np.abs(J @ v)) < 1e-12
    rows, _ = model.wheel_rows(q, 'r1', kin, contacts)
    assert np.max(np.abs(rows @ v)) < 1e-12
    assert np.allclose(v[robot.base[[0, 1, 3]]], u)
    # the planar rates are the rates of the dependent coordinates the pose solver follows
    h = 1e-6
    planar = q[robot.base[[0, 1, 3]]]
    change = (model.ground_pose('r1', planar + h * u)[1] - model.ground_pose('r1', planar - h * u)[1]) / (2. * h)
    assert np.allclose(v[model.ground_dep['r1']], change, atol=1e-6)


def test_wheel_loads_are_determinate_and_the_support_is_a_triangle(closed):
    c = closed(TEAM)
    comp, model = c['comp'], c['model']
    q = c['carry']
    zero = np.zeros(model.n)
    mode = model.mode()
    ident = model.identifiability(q, mode)
    assert ident['redundant_rows'] == 0
    assert all(ident['ports'][f'{rid}:ground']['determinate'] for rid in comp.robots)
    # for comparison, the rigid four-wheel base: its three ground rows are determinate, the load of each wheel is not a row at all
    out = model.inverse(q, zero, zero, mode)
    kin = comp.tree.sweep(q)
    con = out['constraints']
    weight = 0.
    for rid, robot in comp.robots.items():
        N = out['ports'][f'{rid}:ground']
        assert len(N) == 4 and N.min() > 300.
        front = [k for k, w in enumerate(robot.wheels) if w['suspension']]
        assert abs(N[front[0]] - N[front[1]]) < 1e-6 * N.sum()             # a free rocker shares its load equally at rest
        loads = ground_loads(model, q, rid, N, out['ports'][f'{rid}:wheels'], kin, con.info[f'{rid}:normal'])
        assert np.allclose(loads['normal_loads_N'], N) and 'determinate' in loads['assumption']
        points, wheels = support_points(model, rid, kin[0])
        assert len(points) == 3
        pivot = points[[len(w) == 2 for w in wheels].index(True)]
        assert np.allclose(pivot, [.319, 0.], atol=1e-9)                    # the front pair acts at the rocker pivot
        # the margin is taken to that triangle, which is narrower than the rectangle of the four wheels
        cop = loads['cop_base_m']
        assert 0. < loads['tipping_margin_m'] < .2755 - abs(cop[1])
        weight += N.sum()
    carried = sum(comp.tree.mass_kg) * 9.81
    assert abs(weight - carried) < 1e-6 * carried                           # the wheels carry robots and payload


def test_lift_and_reactions_with_the_contact_rows(closed):
    c = closed(TEAM)
    comp, model = c['comp'], c['model']
    mode = model.mode()
    lift = model.lift(c['carry'], mode)
    n_robots = len(comp.robots)
    assert lift.redundant_rows == 0
    assert lift.constraints.J.shape[0] == 6 * n_robots + 8 * n_robots       # grasps; four contact and four rolling rows per base
    assert len(lift.independent) == 6 + 3 * n_robots                       # the payload, and each base in its plane
    rng = np.random.default_rng(2)
    u = .2 * rng.normal(size=len(lift.independent))
    f = .1 * rng.uniform(-1, 1, model.nf) * model.effort_bounds[:, 1]
    fwd = model.forward(c['carry'], u, f, mode)
    assert fwd['reaction_balance'] < 1e-6 and fwd['virtual_power_defect'] < 1e-8
    rec = model.recover(c['carry'], fwd['v'], fwd['a'], f, mode)
    assert rec['unbalanced'] < 1e-6 and rec['deficiency'] == 0
    for key, value in fwd['ports'].items():
        assert np.allclose(rec['ports'][key], value, atol=1e-5), key


def test_start_and_carry_configurations_without_a_lift(closed):
    from mission.runner import start_configuration
    c = closed(TEAM)
    comp, model, mission = c['comp'], c['model'], c['mission']
    assert mission['lifts'] == dict(station={}, carry={})
    assert c['ctrl'].lift_table == {}
    q0 = start_configuration(comp, mission, model)
    for rid, robot in comp.robots.items():
        assert abs(chart_pose(q0[robot.base])[2, 3] - model.base_height[rid]) < 1e-12
        assert q0[robot.joint['front_rocker']] == 0.
    kin = comp.tree.sweep(c['carry'])
    for rid in comp.robots:
        assert max(abs(x['height']) for x in model.wheel_contacts(c['carry'], rid, kin)) < 1e-11


def test_declared_clamp_compliance_keeps_the_explicit_loads_stable(closed):
    from backends.latch import DEFAULTS, STABILITY_LIMIT, LatchSet
    c = closed(TEAM)
    declared = c['comp'].robots['r1'].record['ports']['tool']['clamp']
    latches = LatchSet(c['comp'])
    assert latches.p == dict(DEFAULTS, **declared) and declared['stiffness_N_m'] < DEFAULTS['stiffness_N_m']
    for cid in latches.grasps:
        latches.close(cid, c['carry'])
    assert 0. < latches.stability_ratio(c['carry'], .002) < STABILITY_LIMIT
    stiff = LatchSet(c['comp'], DEFAULTS)                                   # the clamp of the heavier robot on this light team
    for cid in stiff.grasps:
        stiff.close(cid, c['carry'])
    assert stiff.stability_ratio(c['carry'], .002) > 1.


@pytest.mark.parametrize('yaw_offset, along', ((0., 0.), (.3, 0.), (.6, 0.), (.9, .15)))
def test_the_rocker_angle_of_the_compiled_model_is_the_one_a_roller_plant_settles_at(rb, missions, yaw_offset, along):
    """A base standing across the foot of the ramp, square to it and obliquely.

    The compiled contact rows give its height, pitch, roll and rocker angle.  The MuJoCo plant knows
    nothing of them: its wheels are wide, carry rollers and rest on soft contacts.
    """
    pytest.importorskip('mujoco')
    from backends.mujoco_plant import MujocoPlant
    from mission.runner import run_mission, start_configuration
    comp, model = rb
    site = comp.scene['site']
    mission = deepcopy(missions(TEAM))
    mission['starts']['r1'] = [site['x_c'], site['y_ramp0'] + along, np.pi / 2 - yaw_offset]
    q0 = start_configuration(comp, mission, model)
    plant = MujocoPlant(comp, roller_hull=(21, 30))
    run_mission(plant, comp, mission, options=dict(duration=.9))             # the settle phase: the team stands still
    q1 = plant.state()[0]
    robot = comp.robots['r1']
    rocker, base = robot.joint['front_rocker'], robot.base
    slope = np.arctan2(site['ramp_rise'], site['y_ramp1'] - site['y_ramp0'])
    if yaw_offset == 0.:
        assert abs(q0[rocker]) < 1e-9                                        # square to the edge: the pivot is not needed
    else:
        assert .25 * slope < q0[rocker] < .8 * slope                         # oblique: the front pair rolls against the rear pair
    assert abs(q1[rocker] - q0[rocker]) < 1.5e-3                             # rad
    assert abs(q1[base[2]] - q0[base[2]]) < 3e-4                             # m
    assert np.abs(q1[base[4:]] - q0[base[4:]]).max() < 1e-3                  # pitch and roll, rad
