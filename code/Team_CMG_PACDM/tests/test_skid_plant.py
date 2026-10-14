"""A skid-steered base in the plants against the compiled skid model.

MuJoCo, ``tyre='model'``: frictionless contacts and the tyre model of ``backends/tyre.py`` (also what the Isaac Sim
plant uses).  MuJoCo, ``tyre='native'``: the friction of the MuJoCo contacts, with the two properties described in
``backends/mujoco_export.py``.  Pinocchio: the base on the declared normal compliance of its tyres with the same
tyre model (``'pinocchio'``), or on rigid ground with the wheel loads of the step before (``'pinocchio-rigid'``).
"""
import numpy as np
import pytest

from backends.tyre import TyreSet
from teamc.interfaces import TeamModel
from teamc.io import load_team

mujoco = pytest.importorskip('mujoco')

ARM_KP = np.array([3000., 3000., 2000., 300., 300., 300.])
ARM_KD = np.array([200., 200., 100., 15., 15., 15.])


def drive(tyre, forward, yaw_rate, pose=(-3., -1., .3), steps=2500, hold=False):
    """One Husky of ``husky2`` driven by wheel speed loops at the rates of its skid map; the mean motion of the second half.

    ``tyre``: ``'model'`` or ``'native'`` (MuJoCo plant), ``'pinocchio'`` or ``'pinocchio-rigid'`` (Pinocchio plant).
    Returns the measured planar twist of the base (its own axes), the twist the skid model gives for the measured
    wheel rates and wheel loads, the turning point of the model and the displacement of the base origin.
    """
    comp = load_team('husky2')
    model = TeamModel(comp)
    if tyre.startswith('pinocchio'):
        from backends.pin_plant import PinPlant
        plant = PinPlant(comp, skid_ground='rigid' if tyre.endswith('rigid') else 'tyres')
    else:
        from backends.mujoco_plant import MujocoPlant
        plant = MujocoPlant(comp, tyre=tyre)
    rid = 'r1'
    r = comp.robots[rid]
    q = comp.seed()
    q[r.base[[0, 1, 3]]] = pose
    idx, value = model.ground_pose(rid, pose)
    q[idx] = value
    plant.reset(q)
    ports = model.ports
    wheels = np.array([w['index'] for w in r.wheels])
    wheel_ports = np.array([next(k for k, p in enumerate(ports) if p['index'] == i) for i in wheels])
    arm_ports = np.array([k for k, p in enumerate(ports) if 'wheel' not in p['id']])
    arm = np.array([ports[k]['index'] for k in arm_ports])
    gains = np.tile(ARM_KP, len(comp.robots)), np.tile(ARM_KD, len(comp.robots))
    sk = model.skid(q, rid)
    target = np.zeros(4) if hold else sk['wheel_rate'] @ [forward, yaw_rate]
    integral, rows = np.zeros(4), []
    zero = np.zeros(comp.tree.n)
    for k in range(steps):
        meas = plant.measure()
        qm, vm = meas['q'], meas['v']
        f = np.zeros(model.nf)
        f[arm_ports] = comp.tree.rnea(qm, zero, zero)[arm] + gains[0] * (q[arm] - qm[arm]) - gains[1] * vm[arm]
        error = target - vm[wheels]
        integral += 2000. * error * plant.dt
        f[wheel_ports] = 60. * error + integral
        plant.set_efforts(f)
        plant.step()
        if k >= steps // 2:
            loads = plant.contacts()['wheels']
            # (a wheel whose contact is open in this step has no entry: MuJoCo's own friction lets a sliding wheel hop)
            rows.append(np.r_[model.planar_twist(qm, vm, rid), vm[wheels],
                              [loads.get((rid, w['id']), [0., 0.])[0] for w in r.wheels]])
    mean = np.mean(rows, axis=0)
    v = np.zeros(comp.tree.n)
    v[wheels] = mean[3:7]
    final = plant.measure()['q']
    return dict(twist=mean[:3], model=model.skid_twist(final, v, rid, mean[7:]), centre=model.skid(final, rid, None, mean[7:])['centre'],
                moved=float(np.linalg.norm(final[r.base[:2]] - q[r.base[:2]])), loads=mean[7:],
                spread=np.std(np.array(rows)[:, 7:], axis=0), sink=float(q[r.base[2]] - final[r.base[2]]))


# ---------------------------------------------------------------------- the tyre model by itself
def tyres():
    comp = load_team('husky2')
    return TyreSet(comp), comp


def test_tyre_grips_at_rest_and_slides_at_its_limit():
    ts, comp = tyres()
    w = ts.wheels[0]
    point, normal = np.array([0., 0., 0.]), np.array([0., 0., 1.])
    others = [(point, normal, np.zeros(3), 0.)] * (len(ts.wheels) - 1)
    load = {w['key']: 300.}
    dt = .002
    # a short slip of 0.5 mm in all, then rest: the tyre holds the force of that deflection
    for _ in range(5):
        out = ts.advance([(point, normal, np.array([.05, 0., 0.]), 0.)] + others, load, dt)
    held = ts.advance([(point, normal, np.zeros(3), 0.)] + others, load, dt)[('tyre',) + w['key']]
    assert np.allclose(held['force'], [w['k'] * 5e-4, 0., 0.], rtol=1e-9)             # on the ground: along the slip
    for _ in range(50):
        again = ts.advance([(point, normal, np.zeros(3), 0.)] + others, load, dt)[('tyre',) + w['key']]
    assert np.allclose(again['force'], held['force'])                                 # and keeps holding it
    # a steady slip: the force is on the friction circle, against the slip of the wheel
    for _ in range(200):
        out = ts.advance([(point, normal, np.array([.03, -.04, 0.]), 0.)] + others, load, dt)
    sliding = out[('tyre',) + w['key']]
    assert abs(np.linalg.norm(sliding['force']) - w['mu'] * 300.) < 1e-9
    assert np.allclose(sliding['force'] / np.linalg.norm(sliding['force']), [.6, -.8, 0.])
    assert ts.sliding[w['key']]
    # a wheel that leaves the ground forgets its deflection
    off = ts.advance([(point, normal, np.zeros(3), 0.)] + others, {}, dt)[('tyre',) + w['key']]
    assert np.allclose(off['force'], 0.) and off['deflection'] == 0.


def test_a_rolling_tyre_gives_a_force_proportional_to_its_slip_angle():
    ts, comp = tyres()
    w = ts.wheels[0]
    point, normal = np.zeros(3), np.array([0., 0., 1.])
    others = [(point, normal, np.zeros(3), 0.)] * (len(ts.wheels) - 1)
    spin = .3 / w['radius']                                                            # rolling at 0.3 m/s
    side = np.array([0., .003, 0.])                                                    # 3 mm/s sideways: an angle of 0.01 rad
    for _ in range(2000):
        out = ts.advance([(point, normal, side, spin)] + others, {w['key']: 300.}, .002)
    force = out[('tyre',) + w['key']]['force']
    cornering = w['k'] * w['relax']                                                    # N per radian of slip angle
    assert abs(force[1] - (cornering * .01 + w['c'] * .003)) < 1e-6 and abs(force[0]) < 1e-12
    assert np.linalg.norm(force) < w['mu'] * 300. and not ts.sliding[w['key']]
    # and after the slip has ended the force decays over the relaxation length (a fifth of it rolled here)
    for _ in range(10):
        out = ts.advance([(point, normal, np.zeros(3), spin)] + others, {w['key']: 300.}, .002)
    assert abs(out[('tyre',) + w['key']]['force'][1] - cornering * .01 * np.exp(-.3 * .02 / w['relax'])) < 1e-6


# ---------------------------------------------------------------------- tyres that carry their wheels
def on_the_ground(comp, pose=(-3., -1., .3)):
    """A configuration of ``husky2`` with ``r1`` on the ground at a planar pose (wheel centres one radius above it)."""
    model = TeamModel(comp)
    q = comp.seed()
    q[comp.robots['r1'].base[[0, 1, 3]]] = pose
    idx, value = model.ground_pose('r1', pose)
    q[idx] = value
    return q, model


def robot_weight(comp, rid='r1'):
    prefix = comp.robots[rid].prefix
    return 9.81 * sum(comp.tree.mass_kg[i] for i in comp.tree.massive if comp.tree.names[i].startswith(prefix))


def test_tyre_springs_carry_a_base_with_determinate_wheel_loads():
    """On its tyre springs a base has one set of wheel loads: that of the declared equal-stiffness rule of the model."""
    comp = load_team('husky2')
    ts = TyreSet(comp, springs=True)
    q, model = on_the_ground(comp)
    r = comp.robots['r1']
    zero = np.zeros(comp.tree.n)
    assert max(ts.normal_loads(q, zero).values()) == 0.                               # rigid wheels just touch: no load yet
    rest = ts.stand(q)
    N = np.array([ts.normal_loads(rest, zero)[('r1', w['id'])] for w in r.wheels])
    assert abs(N.sum() / robot_weight(comp) - 1.) < 1e-9                              # the springs carry the robot
    assert np.abs(N - model.standing_loads(q, 'r1')).max() < 1.                       # as the model shares its weight (N of 190 to 360;
    #                                                                                   the base leans by 0.6 mrad on its springs)
    assert abs(N[0] - N[1] - N[2] + N[3]) < 1e-6                                      # no load on one diagonal at the cost of the other
    stiffness = r.wheels[0]['tyre']['contact_stiffness_N_m']
    assert abs((q - rest)[r.base[2]] - robot_weight(comp) / 4. / stiffness) < 2e-5    # about 1.4 mm on the declared tyre
    moved = np.flatnonzero(np.abs(rest - q) > 0.)
    assert set(moved) <= set(r.base[[2, 4, 5]]) | set(comp.robots['r2'].base[[2, 4, 5]])   # height, pitch and roll only
    assert np.allclose(ts.stand(rest), rest, atol=1e-12)                              # and that is where it stays
    lifted = q.copy()
    lifted[r.base[2]] += .5
    assert np.array_equal(ts.stand(lifted)[r.base], lifted[r.base])                   # a base in the air is left there
    assert ts.normal_loads(lifted, zero)[('r1', r.wheels[0]['id'])] == 0.


def test_tyre_springs_bring_a_base_to_rest_on_ground_that_is_not_one_plane():
    """Across an edge of the ramp at an angle, rigid wheels fit the ground only in the mean: two are in it and two above
    it.  On its springs the base comes to rest all the same, on three tyres where the fourth does not reach."""
    from backends.latch import LatchSet
    comp = load_team('husky2')
    ts = TyreSet(comp, springs=True)
    site = comp.scene['site']
    r = comp.robots['r1']
    tree, zero = comp.tree, np.zeros(comp.tree.n)
    held = np.asarray(r.base)[[2, 4, 5]]
    three = 0
    for edge, shift, turn in ((site['y_ramp0'], -.12, .45), (site['y_ramp0'], .2, -.5), (site['y_ramp1'], -.3, .35),
                              (site['y_ramp1'], .05, -.25), (site['y_ramp0'], -.2, .3)):
        q, model = on_the_ground(comp, (site['x_c'], edge + shift, np.pi / 2 + turn))
        kin = tree.sweep(q)
        gaps = [ts.terrain.sphere_contact(kin[0][w['index'], :3, 3], w['radius'])[0] for w in ts.wheels if w['key'][0] == 'r1']
        assert min(gaps) < -3e-4 and max(gaps) > 3e-4                                 # in the ground and clear of it
        rest = ts.stand(q)
        kin = tree.sweep(rest)
        N = ts.normal_loads(rest, zero, kin)
        fext = {}
        for w in ts.wheels:
            point, normal = ts.ground_point(w, kin[0][w['index'], :3, 3])
            fext[w['body']] = LatchSet.spatial(point, N[w['key']] * normal, np.zeros(3))
        assert np.abs(tree.rnea(rest, zero, zero, fext=fext, kinematics=kin)[held]).max() < 1e-6   # N, N m: at rest
        mine = np.array([N[('r1', w['id'])] for w in r.wheels])
        assert mine.min() >= 0. and abs(mine.sum() / robot_weight(comp) - 1.) < 2e-3
        three += int((mine == 0.).sum() == 1)
        assert np.allclose(ts.stand(rest), rest, atol=1e-10)
    assert three >= 1


def test_tyre_spring_load_follows_the_declared_stiffness_and_damping_and_never_pulls():
    comp = load_team('husky2')
    ts = TyreSet(comp, springs=True)
    q, _ = on_the_ground(comp)
    r = comp.robots['r1']
    w = ts.wheels[0]
    key = w['key']
    low = q.copy()
    low[r.base[2]] -= .002                                                            # every tyre 2 mm into the floor
    v = np.zeros(comp.tree.n)
    assert abs(ts.normal_loads(low, v)[key] - w['kn'] * .002) < 1e-6
    v[r.base[2]] = -.01                                                               # and sinking at 10 mm/s
    assert abs(ts.normal_loads(low, v)[key] - (w['kn'] * .002 + w['cn'] * .01)) < 1e-6
    v[r.base[2]] = .5                                                                 # leaving faster than the spring relaxes
    assert ts.normal_loads(low, v)[key] == 0.
    # the load a plant applies: friction and normal force together, on the wheel at its ground point
    loads = ts.loads(low, np.zeros(comp.tree.n), None, .002)
    applied = TyreSet.carried(loads)[('tyre',) + key]
    assert loads[('tyre',) + key]['load'] == pytest.approx(w['kn'] * .002)
    assert np.allclose(-applied['force'], [0., 0., w['kn'] * .002]) and np.allclose(loads[('tyre',) + key]['force'], 0.)
    centre = comp.tree.sweep(low)[0][w['index'], :3, 3]
    assert np.allclose(applied['point'], centre - [0., 0., w['radius']])


def test_tyre_springs_are_within_the_stability_limit_of_the_timestep():
    from backends.tyre import STABILITY_LIMIT, check_tyre_stability
    comp = load_team('husky2')
    q, _ = on_the_ground(comp)
    friction, springs = TyreSet(comp), TyreSet(comp, springs=True)
    rest = springs.stand(q)
    assert friction.stability_ratio(rest, .002) < springs.stability_ratio(rest, .002) < STABILITY_LIMIT
    assert springs.stability_ratio(rest, .002) == pytest.approx(.454, abs=.01)
    with pytest.raises(RuntimeError, match='smaller timestep'):
        check_tyre_stability(springs, rest, .004)


# ---------------------------------------------------------------------- the base in the plant
@pytest.mark.parametrize('tyre', ('model', 'native', 'pinocchio', 'pinocchio-rigid'))
def test_turn_on_the_spot_follows_the_skid_model(tyre):
    if tyre.startswith('pinocchio'):
        pytest.importorskip('pinocchio')
    rate = .20 if tyre == 'native' else .25                                            # (the next test: why not faster)
    out = drive(tyre, 0., rate)
    assert abs(out['twist'][2] / out['model'][2] - 1.) < .02                           # yaw rate
    assert abs(out['twist'][2] - rate) < .02                                           # which is the one asked for
    assert np.max(np.abs(out['twist'][:2] - out['model'][:2])) < 2e-3                  # and the same turning point
    assert abs(out['twist'][1] + out['centre'][0] * out['twist'][2]) < 2e-3            # the origin slides sideways by -x0 w
    assert out['spread'].max() < 1.                                                    # steady wheel loads (N)


def test_native_mujoco_friction_gives_way_in_a_faster_turn_on_the_spot():
    """Above a slip of about 0.1 m/s the contacts of the sliding wheels open and close and the base no longer turns as asked."""
    out = drive('native', 0., .25)
    assert out['twist'][2] < .6 * out['model'][2]                                      # it turned at 0.07 rad/s instead of 0.25
    assert out['spread'].min() > 20.                                                   # with wheel loads that jump (N, standard deviation)


@pytest.mark.parametrize('tyre', ('model', 'pinocchio', 'pinocchio-rigid'))
def test_driving_through_a_turn_follows_the_skid_model_with_the_tyre_model(tyre):
    if tyre.startswith('pinocchio'):
        pytest.importorskip('pinocchio')
    out = drive(tyre, .30, .15)
    assert abs(out['twist'][2] / out['model'][2] - 1.) < .02
    assert np.max(np.abs(out['twist'][:2] - out['model'][:2])) < 2e-3
    assert out['spread'].max() < 1.                                                    # a steady turn: steady wheel loads (N)


def test_the_three_plants_agree_on_the_wheel_loads_of_a_turning_base():
    """Soft contacts (MuJoCo), tyre springs and rigid ground rows (Pinocchio) arrive at the same wheel loads."""
    pytest.importorskip('pinocchio')
    loads = {tyre: drive(tyre, .20, .12)['loads'] for tyre in ('model', 'pinocchio', 'pinocchio-rigid')}
    comp = load_team('husky2')
    r = comp.robots['r1']
    weight = 9.81 * sum(b['mass_kg'] for b in r.record['bodies'])
    for tyre, N in loads.items():
        assert abs(N.sum() / weight - 1.) < 2e-3, tyre
        assert np.abs(N - loads['pinocchio']).max() < 6., tyre                         # of 190 to 360 N
    # in a steady turn of a base that nothing else holds the friction forces balance each other and shift no load
    standing = TeamModel(comp).standing_loads(comp.seed(), 'r1')
    assert np.abs(loads['pinocchio'] - standing).max() < 3.


def test_the_pinocchio_plant_stands_a_base_on_its_tyres():
    pytest.importorskip('pinocchio')
    out = drive('pinocchio', 0., 0., steps=500, hold=True)
    comp = load_team('husky2')
    r = comp.robots['r1']
    weight = 9.81 * sum(b['mass_kg'] for b in r.record['bodies'])
    assert abs(out['loads'].sum() / weight - 1.) < 1e-3 and out['spread'].max() < .5   # at rest from the first step
    stiffness = r.wheels[0]['tyre']['contact_stiffness_N_m']
    assert abs(out['sink'] - weight / 4. / stiffness) < 2e-4 and out['moved'] < 1e-4   # about 1.4 mm on the declared tyre


def test_native_mujoco_friction_drifts_outwards_in_a_turn():
    """The property of MuJoCo's friction rows described in ``backends/mujoco_export.py``, measured."""
    comp = load_team('husky2')
    constant = comp.robots['r1'].wheels[0]['tyre']['friction_time_constant_s']
    out = drive('native', .30, .15)
    drift = out['twist'][1] - out['model'][1]
    expected = -.30 * .15 * .95 * constant / 2.                                        # v w / b, to the outside of a left turn
    assert abs(drift / expected - 1.) < .35
    assert abs(out['twist'][2] / out['model'][2] - 1.) < .05                           # the yaw rate is hardly affected


@pytest.mark.parametrize('tyre', ('model', 'pinocchio'))
def test_the_tyre_model_holds_a_base_on_the_ramp(tyre):
    if tyre.startswith('pinocchio'):
        pytest.importorskip('pinocchio')
    comp = load_team('husky2')
    site = comp.scene['site']
    out = drive(tyre, 0., 0., pose=(site['x_c'], .5 * (site['y_ramp0'] + site['y_ramp1']), np.pi / 2), steps=1500, hold=True)
    assert out['moved'] < 1e-3                                                         # static friction: no creep down the slope
    weight = 9.81 * sum(comp.tree.mass_kg[i] for i in comp.tree.massive if comp.tree.names[i].startswith(comp.robots['r1'].prefix))
    assert abs(out['loads'].sum() / weight - 1.) < .01                                 # and the wheels carry the robot
    assert out['loads'][2:].sum() > out['loads'][:2].sum()                             # more on the wheels down the slope
