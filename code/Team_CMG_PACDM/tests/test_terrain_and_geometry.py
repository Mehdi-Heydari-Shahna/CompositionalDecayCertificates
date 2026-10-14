"""Terrain following and the distance queries used by the clearance audit."""
import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from teamc.geometry import Geometry, Placed, Shape, distance
from teamc.terrain import Terrain


def test_support_on_floor_ramp_and_plateau(compiled):
    comp = compiled('team2')
    terrain = Terrain(comp.scene)
    r = comp.robots['r1']
    C = np.array([w['centre_base'] for w in r.wheels])
    radius = r.wheels[0]['radius_m']
    site = comp.scene['site']
    value, normal, misfit = terrain.support([0., 0., .3], C, radius)
    assert np.allclose(value, [radius, 0., 0.], atol=1e-12) and misfit == 0.
    y = .5 * (site['y_ramp0'] + site['y_ramp1'])
    value, normal, _ = terrain.support([site['x_c'], y, np.pi / 2], C, radius)       # heading up the ramp
    slope = np.arctan(site['ramp_rise'] / (site['y_ramp1'] - site['y_ramp0']))
    assert abs(value[1] + slope) < 1e-9 and abs(value[2]) < 1e-9                       # nose up is negative pitch
    value, _, _ = terrain.support([site['x_c'], site['y_ramp1'] + 3., 0.], C, radius)
    assert np.allclose(value, [site['ramp_rise'] + radius, 0., 0.], atol=1e-9)


def test_support_jacobian_matches_differences(compiled):
    comp = compiled('team2')
    terrain = Terrain(comp.scene)
    r = comp.robots['r1']
    C = np.array([w['centre_base'] for w in r.wheels])
    radius = r.wheels[0]['radius_m']
    site = comp.scene['site']
    rng = np.random.default_rng(0)
    for _ in range(5):
        p = np.array([site['x_c'] + rng.uniform(-.3, .3), rng.uniform(site['y_ramp0'] + 1.1, site['y_ramp1'] - 1.1), rng.uniform(-3, 3)])
        G = terrain.support_jacobian(p, C, radius)
        h = 1e-6
        num = np.column_stack([(terrain.support(p + h * e, C, radius)[0] - terrain.support(p - h * e, C, radius)[0]) / (2 * h)
                               for e in np.eye(3)])
        assert np.allclose(G, num, atol=1e-7)


def _box(rng, shift):
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    T[:3, 3] = rng.uniform(-.8, .8, 3) + shift
    half = rng.uniform(.05, .4, 3)
    return Placed(Shape('x', 'box', -1, 'g', 'r', np.eye(4), half=half), T), T, half


def test_sphere_contact_is_continuous_over_the_edges_of_the_ramp(compiled):
    """The plane a tyre stands on changes at the foot and at the top of the ramp; its depth in the terrain does not jump."""
    comp = compiled('husky2')
    terrain = Terrain(comp.scene)
    site = comp.scene['site']
    radius = .1651
    by_id = {p['id']: p for p in terrain.patches}
    slope = float(np.arccos(by_id['ramp']['n'][2]))
    assert abs(np.degrees(slope) - 3.) < .01
    step = 1e-4
    ys = np.arange(site['y_ramp0'] - .5, site['y_ramp1'] + .5, step)
    for offset in (-.002, 0., .005):                                 # sunk in, touching, lifted
        # the centre follows the height of the terrain under it (a path with a kink at each edge, as a rolling wheel takes)
        heights = np.array([terrain.height(site['x_c'], y) for y in ys]) + radius + offset
        out = [terrain.sphere_contact((site['x_c'], y, z), radius) for y, z in zip(ys, heights)]
        clearance = np.array([x[0] for x in out])
        names = [x[2]['id'] for x in out]
        assert names[0] == 'floor' and names[len(ys) // 2] == 'ramp' and names[-1] == 'plateau'
        # the path has slopes of 0 and tan(3 degrees): the clearance changes by no more than that between two samples
        assert np.abs(np.diff(clearance)).max() < 1.05 * step * np.tan(slope) + 1e-12
        # away from the edges it is the distance to the one plane below: the offset, seen along the normal of that plane
        assert abs(clearance[0] - offset) < 1e-12 and abs(clearance[-1] - offset) < 1e-12
        assert abs(clearance[len(ys) // 2] - ((radius + offset) * np.cos(slope) - radius)) < 1e-9
        # at the foot of the ramp (a concave crease) the sphere meets the rising plane before its centre is over the crease
        first = ys[names.index('ramp')]
        assert site['y_ramp0'] - 1.2 * radius * np.tan(slope) < first < site['y_ramp0']
    normal = terrain.sphere_contact((site['x_c'], .5 * (site['y_ramp0'] + site['y_ramp1']), 1.), radius)[1]
    assert np.allclose(normal, by_id['ramp']['n'])
    with pytest.raises(ValueError, match='No terrain patch'):
        terrain.sphere_contact((500., 500., 0.), radius)


def test_gjk_box_box_against_a_direct_minimisation():
    rng = np.random.default_rng(4)
    checked = 0
    for _ in range(40):
        (a, Ta, ha), (b, Tb, hb) = _box(rng, np.zeros(3)), _box(rng, np.array([1.4, 0., 0.]))
        d, xa, xb = distance(a, b)
        if d <= 0.:
            continue

        def gap(z):
            return np.sum((Ta[:3, 3] + Ta[:3, :3] @ (z[:3] * ha) - Tb[:3, 3] - Tb[:3, :3] @ (z[3:] * hb)) ** 2)

        best = min(minimize(gap, rng.uniform(-1, 1, 6), bounds=[(-1, 1)] * 6).fun for _ in range(6))
        assert abs(np.sqrt(best) - d) < 1e-5
        assert abs(np.linalg.norm(xa - xb) - d) < 1e-9
        checked += 1
    assert checked > 20


def test_gjk_capsule_and_sphere_cases():
    eye = np.eye(4)
    T = eye.copy()
    T[:3, 3] = [1., 0., 0.]
    a = Placed(Shape('a', 'sphere', -1, 'g', 'r', eye, radius=.2), eye)
    b = Placed(Shape('b', 'sphere', -1, 'g', 'r', eye, radius=.3), T)
    assert abs(distance(a, b)[0] - .5) < 1e-9
    c = Placed(Shape('c', 'capsule', -1, 'g', 'r', eye, radius=.1, half_length=.5), eye)          # axis along z
    T2 = eye.copy()
    T2[:3, 3] = [0., .6, .9]
    assert abs(distance(c, Placed(Shape('d', 'sphere', -1, 'g', 'r', eye, radius=.05), T2))[0] - (np.hypot(.6, .4) - .15)) < 1e-9


def test_clearance_categories_at_the_rack(closed):
    c = closed('team2')
    comp = c['comp']
    geo = Geometry(comp)
    cl = geo.clearance(comp.tree.sweep(c['station'])[0], cutoff=1.)
    assert cl['payload:pad'][0] < 1e-6                 # the beam lies on its rack pads
    assert cl['payload:neighbour'][0] > .05            # and is clear of the neighbouring beams
    assert cl['robot:robot'][0] > .2
