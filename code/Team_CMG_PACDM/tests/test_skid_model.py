"""Friction-driven bases: the quasi-static skid model, the friction law and what the compiler makes of them."""
from copy import deepcopy

import numpy as np
import pytest

from teamc import skid
from teamc.compiler import ModelError, compile_team
from teamc.interfaces import TeamModel
from teamc.io import load_records

L, B, R = .512, .5658, .1651                                  # wheelbase, track and tyre radius of the Husky A300
XY = np.array([[L / 2, B / 2], [L / 2, -B / 2], [-L / 2, B / 2], [-L / 2, -B / 2]])
ROLL, AXLE = np.tile([1., 0.], (4, 1)), np.tile([0., 1.], (4, 1))


def model(loads, mu=.8):
    return skid.skid_map(XY, ROLL, AXLE, np.full(4, mu), loads, np.full(4, R))


def power(A, weight, z, speeds):
    s = A @ z
    s[:, 0] -= speeds
    return float(weight @ np.linalg.norm(s, axis=1))


def test_equal_loads_give_the_closed_form():
    m = model(np.full(4, 300.))
    assert abs(m['track_factor'] - (1. + (L / B) ** 2)) < 1e-9           # the equivalent differential drive is wider
    assert np.max(np.abs(m['centre'])) < 1e-9                             # and turns about the middle of the wheels
    # every wheel slides at the same speed: half the wheelbase sideways and (chi - 1) B / 2 along its rolling direction
    slip = np.hypot(L / 2, L ** 2 / (2 * B))
    assert np.allclose(m['slip_per_yaw_m'], slip, atol=1e-9)
    assert abs(m['moment_Nm'] - .8 * 1200. * slip) < 1e-6
    # Clearpath's controller of the A300 uses 1.75 for this factor (clearpath_control, diff_4wd.yaml)
    assert abs(m['track_factor'] - 1.75) < .08


def test_the_twist_minimizes_the_dissipated_power():
    rng = np.random.default_rng(0)
    for _ in range(6):
        loads = rng.uniform(100., 500., 4)
        speeds = rng.uniform(-.5, .5, 4)
        A = skid.slip_rows(XY, ROLL, AXLE)
        z, _ = skid.twist(A, .8 * loads, speeds)
        best = power(A, .8 * loads, z, speeds)
        for _ in range(40):
            assert power(A, .8 * loads, z + 1e-3 * rng.normal(size=3), speeds) >= best - 1e-12


def test_the_friction_forces_balance_at_the_skid_twist():
    """At the twist of the model the Coulomb forces of the sliding wheels have no resultant and no moment."""
    loads = np.array([345., 138., 444., 237.])
    A = skid.slip_rows(XY, ROLL, AXLE)
    z, slips = skid.twist(A, .8 * loads, np.array([-.2, .2, -.2, .2]))
    assert np.linalg.norm(slips, axis=1).min() > 1e-3                      # every wheel slides here
    F = skid.friction_forces(slips, loads, np.full(4, .8), 1e-9)
    assert np.max(np.abs(np.einsum('ijk,ij->k', A, F))) < 1e-6             # generalized force on [vx, vy, wz]
    assert np.allclose(np.linalg.norm(F, axis=1), .8 * loads)              # each at its limit


def test_uneven_loads_move_the_turning_centre():
    even = model(np.full(4, 300.))
    rear = model(np.array([200., 200., 400., 400.]))
    left = model(np.array([400., 200., 400., 200.]))
    assert rear['centre'][0] < -.03 and abs(rear['centre'][1]) < 1e-9      # towards the heavier axle
    assert left['centre'][1] < -.03 and abs(left['centre'][0]) < 1e-9      # away from the heavier side
    assert left['track_factor'] > even['track_factor'] + .05
    assert rear['moment_Nm'] < even['moment_Nm']                           # the heavy axle slides less


def test_the_map_is_linear_in_speed_and_speed_difference():
    loads = np.array([345., 138., 444., 237.])
    m = model(loads)
    A = skid.slip_rows(XY, ROLL, AXLE)
    for forward, delta in ((.3, .1), (-.2, .25), (.4, -.05)):
        z, _ = skid.twist(A, .8 * loads, forward - m['side'] * delta)
        assert np.allclose(z, np.array([forward, 0., 0.]) + delta * m['c'], atol=1e-8)


def test_wheel_rates_of_the_map_give_the_requested_motion():
    """Driving the wheels at ``wheel_rate @ [v, w]`` moves the turning point forward at ``v`` and turns the base at ``w``."""
    loads = np.array([276., 188., 394., 306.])
    m = model(loads)
    A = skid.slip_rows(XY, ROLL, AXLE)
    for v, w in ((.3, .15), (0., .25), (.45, -.1)):
        z, _ = skid.twist(A, .8 * loads, R * (m['wheel_rate'] @ [v, w]))
        x0 = m['centre'][0]
        assert np.allclose(z, [v, -x0 * w, w], atol=1e-8)                  # the origin slides sideways by -x0 w


def test_a_warm_start_gives_the_same_map():
    loads = np.array([345., 138., 444., 237.])
    cold = model(loads)
    warm = skid.skid_map(XY, ROLL, AXLE, np.full(4, .8), loads + .25, np.full(4, R), start=cold['c'])
    again = model(loads + .25)
    assert np.allclose(warm['c'], again['c'], atol=1e-9)


def test_loads_that_leave_the_motion_open_are_refused():
    """With one side unloaded, or one wheel alone carrying load, the friction forces do not determine the twist."""
    for loads in ([300., 0., 300., 0.], [0., 250., 0., 250.], [0., 0., 0., 0.],
                  [300., 0., 0., 0.], [0., 0., 0., 250.], [0., 0., 300., 0.]):
        with pytest.raises(ValueError, match='skid model'):
            model(np.array(loads))
    # one wheel on each side is enough: the heavier one grips, and the base turns about a point of its axle
    two = model(np.array([300., 0., 0., 250.]))
    assert abs(two['centre'][0] - L / 2) < 1e-4


def test_friction_law_builds_up_over_the_slip_speed():
    loads, mu = np.array([300.]), np.array([.8])
    slow = skid.friction_forces(np.array([[1e-4, 0.]]), loads, mu, 1e-3)
    fast = skid.friction_forces(np.array([[.05, .05]]), loads, mu, 1e-3)
    assert abs(slow[0, 0] + 240. * 1e-4 / np.hypot(1e-4, 1e-3)) < 1e-9     # a tenth of the slip speed: a tenth of the limit
    assert abs(np.linalg.norm(fast) - 240.) < .1 and np.allclose(fast[0, 0], fast[0, 1])   # at the limit, against the slip


# ---------------------------------------------------------------------- the compiler and the compiled interfaces
def test_the_compiler_finds_the_drive_from_the_axles(compiled):
    assert {r.drive for r in compiled('husky2').robots.values()} == {'friction'}
    assert {r.drive for r in compiled('team2').robots.values()} == {'rolling'}
    assert {r.drive for r in compiled('ridgeback2').robots.values()} == {'rolling'}
    assert compiled('husky2').plan['robots']['r1']['drive'] == 'friction'
    assert 'drive' not in compiled('team2').plan['robots']['r1']           # the compiled plan of the other teams is unchanged


def test_one_line_of_axles_rolls_and_two_lines_skid():
    team, robots, payload, scene = load_records('husky2')
    two_axles = compile_team(team, robots, payload, scene)
    assert two_axles.robots['r1'].drive == 'friction'
    # the same base with its four wheels on one line across it: it can drive and turn without sliding
    robots = deepcopy(robots)
    for record in robots.values():
        for j in record['joints']:
            if j['id'].endswith('_wheel_joint'):
                T = np.asarray(j['T_AJ'], float)
                T[0, 3] = 0.
                T[1, 3] *= 1. if j['id'].startswith('front') else 1.6
                j['T_AJ'] = T.tolist()
    assert compile_team(team, robots, payload, scene).robots['r1'].drive == 'rolling'


def test_mixed_wheels_are_refused():
    team, robots, payload, scene = load_records('husky2')
    robots = deepcopy(robots)
    for record in robots.values():
        record['wheels'][0]['lateral'] = 'free'
    with pytest.raises(ModelError):
        compile_team(team, robots, payload, scene)


def test_wheel_rates_stay_independent_and_the_slip_rows_are_not_constraints(closed):
    c = closed('husky2')
    model_, q = c['model'], c['carry']
    mode = model_.mode()
    lift = model_.lift(q, mode)
    # payload chart; per robot the planar motion of the base and its four wheels
    assert lift.E.shape[1] == 6 + 2 * (3 + 4)
    blocks = {b['id']: b for b in lift.constraints.blocks}
    assert blocks['r1:wheels'].get('slip') and blocks['r1:wheels']['rows'].stop - blocks['r1:wheels']['rows'].start == 8
    slip_rows = np.r_[tuple(np.arange(blocks[f'{rid}:wheels']['rows'].start, blocks[f'{rid}:wheels']['rows'].stop)
                            for rid in ('r1', 'r2'))]
    assert not set(lift.rows) & set(slip_rows)
    # an admissible velocity satisfies every ideal row and slides the wheels
    v = lift.E @ np.random.default_rng(0).normal(size=lift.E.shape[1])
    ideal = np.setdiff1d(np.arange(lift.constraints.J.shape[0]), slip_rows)
    chart = np.linalg.lstsq(lift.constraints.Jc[ideal], -lift.constraints.J[ideal] @ v, rcond=None)[0]
    assert np.max(np.abs(lift.constraints.J[ideal] @ v + lift.constraints.Jc[ideal] @ chart)) < 1e-8
    assert np.max(np.abs(lift.constraints.J[slip_rows] @ v)) > 1e-2


def test_base_motion_follows_the_skid_map(closed):
    """The wheel rates the model gives for a planar base motion move the base that way in its own skid model."""
    c = closed('husky2')
    model_, q = c['model'], c['carry']
    kin = c['comp'].tree.sweep(q)
    rid = 'r1'
    loads = np.array([334., 241., 341., 248.])
    base, S, W = model_.base_motion(q, rid, kin, loads)
    m = model_.skid(q, rid, kin, loads)
    yaw = q[c['comp'].robots[rid].base[3]]
    for v, w in ((.3, .12), (0., .2)):
        x0 = m['centre'][0]
        rates = np.array([np.cos(yaw) * v + np.sin(yaw) * x0 * w, np.sin(yaw) * v - np.cos(yaw) * x0 * w, w])   # world axes
        qd = np.zeros(model_.n)
        qd[base] = S @ rates
        qd[[x['index'] for x in c['comp'].robots[rid].wheels]] = W @ rates
        assert np.allclose(model_.skid_twist(q, qd, rid, loads, kin), [v, -x0 * w, w], atol=1e-7)
        assert np.allclose(model_.planar_twist(q, qd, rid, kin), [v, -x0 * w, w], atol=1e-9)


def test_forward_and_inverse_dynamics_agree_for_the_wrenches_of_the_motion(closed):
    c = closed('husky2')
    model_, q = c['model'], c['carry']
    mode = model_.mode()
    rng = np.random.default_rng(4)
    lift = model_.lift(q, mode)
    u = .2 * rng.normal(size=lift.E.shape[1])
    f = .1 * rng.uniform(-1, 1, model_.nf) * model_.effort_bounds[:, 1]
    fwd = model_.forward(q, u, f, mode)
    assert fwd['reaction_balance'] < 1e-6
    # the multipliers of the wheel rows are the friction forces of the declared law at the loads used
    for rid in ('r1', 'r2'):
        _, lam = model_.friction_loads(q, fwd['v'], rid, fwd['wheel_loads'][rid])
        assert np.allclose(fwd['ports'][f'{rid}:wheels'], lam, atol=1e-9)
    inv = model_.inverse(q, fwd['v'], fwd['a'], mode, grasp_wrenches={g: fwd['ports'][g] for g in mode.grasps},
                         wheel_loads=fwd['wheel_loads'])
    assert inv['residual'] < 1e-6
    assert np.allclose(inv['f'], f, atol=1e-6)


def test_a_turn_on_the_spot_needs_the_friction_torque_at_every_wheel(closed, missions):
    """Feedforward of a turn about the turning point: each wheel motor balances the friction of its own wheel."""
    c = closed('husky2')
    comp, model_ = c['comp'], c['model']
    from mission.runner import start_configuration
    q = start_configuration(comp, missions('husky2'), model_)
    kin = comp.tree.sweep(q)
    mode = model_.mode(grasps=(), rest='rack')
    rid = 'r1'
    robot = comp.robots[rid]
    loads = model_.standing_loads(q, rid, kin)
    m = model_.skid(q, rid, kin, loads)
    base, S, W = model_.base_motion(q, rid, kin, loads)
    yaw, w, x0 = q[robot.base[3]], .25, m['centre'][0]
    rates = np.array([np.sin(yaw) * x0 * w, -np.cos(yaw) * x0 * w, w])
    v = np.zeros(model_.n)
    wheels = np.array([x['index'] for x in robot.wheels])
    v[base], v[wheels] = S @ rates, W @ rates
    out = model_.inverse(q, v, np.zeros(model_.n), mode, wheel_loads={rid: loads, 'r2': model_.standing_loads(q, 'r2', kin)})
    ports = {p['index']: k for k, p in enumerate(model_.ports)}
    torque = out['f'][[ports[i] for i in wheels]]
    forces = out['ports'][f'{rid}:wheels'].reshape(4, 2)
    assert np.allclose(np.linalg.norm(forces, axis=1), .8 * loads, rtol=2e-3)             # every wheel slides at its limit
    assert np.allclose(torque, robot.wheels[0]['radius_m'] * forces[:, 0], atol=.2)       # motor torque = radius x friction along the tyre
    assert abs(np.sum(forces[:, 1])) < 1. and abs(np.sum(forces[:, 0])) < 1.               # no resultant on the base
    assert np.abs(torque).max() < .9 * model_.effort_bounds[ports[wheels[0]], 1]
