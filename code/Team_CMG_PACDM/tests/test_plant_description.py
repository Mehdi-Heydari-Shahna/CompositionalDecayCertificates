"""The simulator-neutral plant description: rollers, wheels, inertia bookkeeping."""
import numpy as np

from backends.plantspec import build_plantspec, robot_mass, roller_profile, roller_vertices


def test_roller_profile_lies_on_the_wheel_cylinder():
    R, r, L, angle = .127, .022, .118, np.pi / 4
    z, rad = roller_profile(R, r, L, angle, rings=41)
    assert abs(z[0] + L / 2) < 1e-9 and abs(z[-1] - L / 2) < 1e-9
    assert abs(rad.max() - r) < 1e-12 and rad.min() > 0.
    # roller axis: through (0, 0, -(R - r)) of the wheel frame (wheel axis y), tilted by `angle` about z
    axis = np.array([np.sin(angle), np.cos(angle), 0.])
    origin = np.array([0., 0., -(R - r)])
    # a point of the profile, turned about the roller axis until it is farthest from the wheel axis, must be at radius R
    for zi, ri in zip(z[::5], rad[::5]):
        centre = origin + zi * axis
        phi = np.linspace(0., 2 * np.pi, 20001)
        e1 = np.cross(axis, [0., 0., 1.])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)
        pts = centre + ri * (np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2))
        dist = np.hypot(pts[:, 0], pts[:, 2])                           # distance from the wheel axis (y)
        assert abs(dist.max() - R) < 2e-7


def test_roller_vertices_layout():
    V = roller_vertices(.127, .022, .118, rings=7, around=12, stagger=True).reshape(7, 12, 3)
    W = roller_vertices(.127, .022, .118, rings=7, around=12, stagger=False).reshape(7, 12, 3)
    assert np.allclose(V[::2], W[::2])                                  # even rings coincide, odd rings are turned by half a step
    assert np.allclose(np.arctan2(V[1, 0, 1], V[1, 0, 0]), np.pi / 12)
    assert np.allclose(np.linalg.norm(V[:, :, :2], axis=2), np.linalg.norm(W[:, :, :2], axis=2))


def test_wheel_inertia_is_split_between_hub_and_rollers(compiled, mecanum_team):
    comp = compiled(mecanum_team)
    spec = build_plantspec(comp, (7, 12))
    ro = spec['roller']
    Ir = np.diag(ro['inertia'])
    for rid, r in spec['robots'].items():
        bodies = {b['name']: b for b in r['bodies']}
        assert abs(robot_mass(spec, rid) - sum(comp.tree.mass_kg[comp.tree.index[n]] for n in bodies)) < 1e-9
        for w in r['wheels']:
            declared = bodies[w['body']]
            total_m = w['hub_mass'] + len(w['rollers']) * ro['mass']
            total_I = w['hub_inertia'].copy()
            for k in w['rollers']:
                Rk, c = k['T'][:3, :3], k['T'][:3, 3]
                total_I += Rk @ Ir @ Rk.T + ro['mass'] * ((c @ c) * np.eye(3) - np.outer(c, c))
                assert abs(np.linalg.norm(c) - (w['radius'] - ro['ellipsoid'][0])) < 1e-12   # roller axes at R - r from the wheel axis
            assert abs(total_m - declared['mass']) < 1e-12
            assert np.allclose(total_I, declared['inertia'], atol=1e-12)
            assert np.linalg.eigvalsh(w['hub_inertia']).min() > 0.


def test_a_conventional_wheel_keeps_its_declared_inertia_and_gets_a_tyre(compiled, skid_team):
    comp = compiled(skid_team)
    spec = build_plantspec(comp, (7, 12))
    assert spec['roller'] is None
    for rid, r in spec['robots'].items():
        bodies = {b['name']: b for b in r['bodies']}
        assert abs(robot_mass(spec, rid) - sum(comp.tree.mass_kg[comp.tree.index[n]] for n in bodies)) < 1e-9
        for w, declared in zip(r['wheels'], comp.robots[rid].wheels):
            assert w['rollers'] == [] and 'hub_mass' not in w
            tyre = w['tyre']                                            # a sphere of the wheel radius about the wheel centre
            assert tyre['radius'] == declared['radius_m'] and np.allclose(tyre['centre'], 0.)
            assert tyre['stiffness'] == declared['tyre']['stiffness_N_m']
            assert tyre['relaxation_length'] == declared['tyre']['relaxation_length_m']
            assert abs(abs(w['axis_body'] @ [0., 1., 0.]) - 1.) < 1e-12   # the axle is the y axis of the wheel body


def test_roller_axes_follow_the_declared_traction_directions(compiled):
    comp = compiled('team2')
    spec = build_plantspec(comp, (7, 12))
    r = spec['robots']['r1']
    robot = comp.robots['r1']
    bodies = {b['name']: b for b in r['bodies']}
    for w, declared in zip(r['wheels'], robot.wheels):
        R_base_wheel = bodies[w['body']]['T_parent_body'][:3, :3]
        lowest = min(w['rollers'], key=lambda k: (R_base_wheel @ k['T'][:3, 3])[2])       # the roller on the ground
        axis = R_base_wheel @ lowest['T'][:3, 2]
        g = np.asarray(declared['traction_direction_base'], float)
        assert abs(abs(axis @ g) - 1.) < 1e-12                                            # traction acts along the roller axis
