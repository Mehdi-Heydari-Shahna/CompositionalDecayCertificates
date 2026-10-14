"""Route geometry and the speed governor."""
import numpy as np

from mission.route import PlanarPath, compose, relative
from mission.timing import Governor, quintic


def test_compose_and_relative_are_inverse():
    rng = np.random.default_rng(0)
    for _ in range(10):
        a, b = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(relative(a, compose(a, b)), b, atol=1e-12)


def test_quintic_profile():
    s = [quintic(x) for x in (0., .5, 1.)]
    assert np.allclose([x[0] for x in s], [0., .5, 1.]) and abs(s[0][1]) < 1e-12 and abs(s[2][1]) < 1e-12


def test_mission_route_is_smooth_and_long_enough(missions, compiled):
    mission = missions('team2')
    route = mission['route']
    path = PlanarPath(np.asarray(route['samples']), route['rho'])
    assert path.length > 30.
    l = np.linspace(0., path.length, 4000)
    poses = np.array([path(x)[0] for x in l])
    tangent = np.array([path(x)[1] for x in l])
    step = np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)
    assert step.max() < 1.5 * (l[1] - l[0])                           # no jumps
    assert np.abs(np.diff(tangent, axis=0)).max() < .05                # the tangent does not jump at the joints of the pieces


def test_governor_stops_at_a_gate_and_resumes():
    g = Governor(20., lambda l: .45, accel=.25, decel=.25)
    dt, t = .01, 0.
    peak_decel, last = 0., 0.
    while t < 60. and g.waiting is None:
        g.step(dt, stops=[8.])
        peak_decel = max(peak_decel, (last - g.rate) / dt)
        last = g.rate
        t += dt
    assert g.waiting == 8. and g.l <= 8. + 1e-9 and g.overrun == 0.     # waiting at the gate, never past it
    assert 8. - g.l < 4e-3 and g.rate < 4e-3
    assert peak_decel <= 1.6 * .25 + 1e-9                               # within the declared deceleration margin
    while t < 140. and not g.done:
        g.step(dt)
        t += dt
    assert g.done and abs(g.l - 20.) < 2e-3


def test_governor_respects_the_speed_limit():
    limit = lambda l: .45 if l < 5. else .2
    g = Governor(12., limit, accel=.25, decel=.25)
    worst = 0.
    for _ in range(9000):
        g.step(.01)
        worst = max(worst, g.rate - .45)
    assert worst < 1e-3 and g.done


def test_a_late_gate_is_reported_as_an_overrun():
    g = Governor(20., lambda l: .45)
    for _ in range(1500):
        g.step(.01)                                                      # cruising
    here = g.l
    for _ in range(600):
        g.step(.01, stops=[here + .05])                                  # a gate closes 5 cm ahead at full speed
    assert g.overrun > 0. and g.rate < .01
