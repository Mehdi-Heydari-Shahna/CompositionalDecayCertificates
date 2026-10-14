"""The world-frame tree dynamics against Pinocchio and against itself."""
import numpy as np
import pytest


def _random_state(comp, seed):
    rng = np.random.default_rng(seed)
    n = comp.tree.n
    q = comp.seed() + rng.uniform(-.3, .3, n)
    lim = comp.cmg['limits']
    lo = np.array([lim[k]['lower'] for k in comp.cmg['coordinate_ids']])
    hi = np.array([lim[k]['upper'] for k in comp.cmg['coordinate_ids']])
    q = np.clip(q, lo + 1e-3, hi - 1e-3)
    return q, rng.uniform(-1., 1., n), rng.uniform(-1., 1., n)


def test_inverse_dynamics_is_mass_times_acceleration_plus_bias(compiled, team):
    comp = compiled(team)
    tree = comp.tree
    for seed in range(3):
        q, v, a = _random_state(comp, seed)
        M, h = tree.mass(q), tree.bias(q, v)
        assert np.allclose(M, M.T, atol=1e-10)
        assert np.linalg.eigvalsh(M).min() > 0.
        assert np.allclose(tree.rnea(q, v, a), M @ a + h, rtol=1e-10, atol=1e-8)


def test_against_pinocchio(compiled, team):
    pin = pytest.importorskip('pinocchio')
    from backends.pin_model import PinTeam
    comp = compiled(team)
    pt = PinTeam(comp)
    for seed in range(3):
        q, v, a = _random_state(comp, seed + 10)
        M, h = pt.terms(q, v)
        assert np.allclose(comp.tree.mass(q), M, rtol=1e-10, atol=1e-9)
        assert np.allclose(comp.tree.bias(q, v), h, rtol=1e-10, atol=1e-8)
        assert np.allclose(comp.tree.rnea(q, v, a), pin.rnea(pt.model, pt.data, q, v, a), rtol=1e-10, atol=1e-8)   # Pinocchio includes the armature
        pt.kinematics(q)
        poses = comp.tree.poses(q)
        for name in list(poses)[::7]:
            jid, B = pt.body[name]
            assert np.allclose(pt.data.oMi[jid].homogeneous @ B, poses[name], atol=1e-11)


def test_joint_wrench_of_a_static_tool_is_its_weight(compiled):
    comp = compiled('team2')
    tree = comp.tree
    q = comp.seed()
    zero = np.zeros(tree.n)
    r = comp.robots['r1']
    sensor = r.record['ports']['sensor']
    body, T_s = r.frames[sensor['frame']]
    w = tree.joint_wrench(q, zero, zero, r.prefix + sensor['joint_id'], T_s)
    mass = tree.mass_kg[tree.index[body]]
    assert abs(np.linalg.norm(w[3:]) - mass * 9.81) < 1e-9          # the sensor carries the tool-side body


def test_external_wrench_enters_with_the_right_sign(compiled):
    comp = compiled('team2')
    tree = comp.tree
    q = comp.seed()
    zero = np.zeros(tree.n)
    body = comp.payload['body']
    lift = np.r_[0., 0., 0., 0., 0., tree.mass_kg[tree.index[body]] * 9.81]      # weight carried at the world origin ...
    T = tree.poses(q)[body]
    lift[:3] = np.cross(T[:3, 3] + T[:3, :3] @ tree.com[tree.index[body]], lift[3:])  # ... moved to the centre of mass
    tau = tree.rnea(q, zero, zero, fext={body: lift})
    assert np.max(np.abs(tau[np.asarray(comp.payload['base'])])) < 1e-9
