"""Paths of bases that cannot move sideways (planar geometry, no platform)."""
import numpy as np
import pytest

from mission.route import compose, line, smooth_poses
from mission.steering import (arc_length, drawn_headings, drawn_lag, follow_tangents, hermite, led_poses, line_and_arc,
                              tangent_headings, turn_in_place, unit, yaw_on_line)


def curvature(points):
    d = np.gradient(points, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    return np.gradient(heading) / np.linalg.norm(d, axis=1)


def test_hermite_leaves_and_arrives_straight():
    a, b = np.array([0., 0.]), np.array([1.2, 3.5])
    for weights in ((1., 1.), (1., 1.6)):
        P = hermite(a, np.pi / 2, b, 1.71, 400, weights)
        assert np.allclose(P[0], a) and np.allclose(P[-1], b)
        h, _ = tangent_headings(P, np.pi / 2)
        assert abs(h[1] - np.pi / 2) < 1e-3 and abs(h[-2] - 1.71) < 1e-3
        k = curvature(P)
        assert max(abs(k[2]), abs(k[-3])) < .1 * np.abs(k).max() and np.abs(k).max() > .2   # no curvature at the ends


def test_line_and_arc_reaches_the_goal_pose_on_the_shorter_side():
    goal = np.array([3., 2.5, np.pi / 2])
    for start in ([0., 0.], [6., -1.], [3.2, -3.]):
        P = line_and_arc(start, goal, 1.5)
        assert np.allclose(P[0, :2], start) and np.allclose(P[-1], goal, atol=1e-9)
        h, step = tangent_headings(P[:, :2], P[0, 2])
        assert np.max(np.abs((h - P[:, 2])[1:-1])) < .02                             # the heading is the path direction
        assert np.abs(curvature(P[:, :2])[3:-3]).max() < 1. / 1.5 + .02              # nowhere tighter than the circle
        assert np.abs(np.diff(P[:, 2])).max() < .02                                  # no jump of the heading where the arc begins
    inside = line_and_arc([2.2, 2.4], goal, 1.5)                                     # inside the left circle: round the right one
    assert np.allclose(inside[-1, :2], goal[:2], atol=1e-9) and inside[:, 0].max() > 5.5
    assert abs((inside[-1, 2] - goal[2] + np.pi) % (2. * np.pi) - np.pi) < 1e-9      # the heading is continuous along the way
    with pytest.raises(ValueError):
        line_and_arc(goal[:2], goal, 1.5)                                            # on both circles: nowhere to come from


def test_turn_in_place_keeps_the_turning_point():
    pose = np.array([1., 2., .3])
    for x0 in (-.13, 0., .05):
        P = turn_in_place(pose, 2.1, x0)
        pivot = P[:, :2] + x0 * np.c_[np.cos(P[:, 2]), np.sin(P[:, 2])]
        assert np.allclose(pivot, pivot[0]) and np.allclose(P[0], pose) and abs(P[-1, 2] - 2.1) < 1e-12


def test_yaw_on_line_keeps_the_second_base_on_its_lane():
    lane_x, apart = 9.69, np.array([3.6, 0.])
    lead = hermite([lane_x, 19.5], np.pi / 2, [10.9, 24.], 1.71, 300, (1., 1.6))
    yaw = yaw_on_line(lead, apart, [lane_x, 0.], np.pi / 2, np.pi / 2)
    second = lead - np.c_[np.cos(yaw) * apart[0], np.sin(yaw) * apart[0]]
    assert np.max(np.abs(second[:, 0] - lane_x)) < 1e-12                             # on the lane
    assert np.all(np.diff(second[:, 1]) > 0.)                                        # and driving forwards on it
    away = np.diff(lead[:, 0]) > 0.
    assert abs(yaw[0] - np.pi / 2) < 1e-9 and np.all(np.diff(yaw)[away] < 0.)        # it turns as the leading base leaves the lane
    assert abs(np.cos(yaw[-1]) - (lead[-1, 0] - lane_x) / apart[0]) < 1e-12
    with pytest.raises(ValueError):
        yaw_on_line(lead + [4., 0.], apart, [lane_x, 0.], np.pi / 2, np.pi / 2)      # further from the lane than the bases are apart


def test_led_poses_put_the_leading_base_on_its_path():
    lead = hermite([0., 0.], 0., [4., 1.], .4, 200)
    yaw = np.linspace(0., -1.2, len(lead))
    offset = np.array([1.8, -.45])
    poses = led_poses(lead, offset, yaw)
    back = np.array([compose(p, [offset[0], offset[1], 0.])[:2] for p in poses])
    assert np.allclose(back, lead, atol=1e-12) and np.allclose(poses[:, 2], yaw)


def test_follow_tangents_reports_the_tightest_turn():
    # the payload turns on the spot about its centre: a base 1.8 m from it drives a circle of that radius
    n = 400
    poses = np.c_[np.zeros(n), np.zeros(n), np.linspace(0., 1.5, n)]
    bases, info = follow_tangents(poses, dict(a=np.array([1.8, 0.]), b=np.array([-.9, 0.])), dict(a=np.pi / 2, b=-np.pi / 2))
    assert abs(info['a']['max_turn_rad_per_m'] - 1. / 1.8) < 1e-3 and abs(info['b']['max_turn_rad_per_m'] - 1. / .9) < 1e-3
    assert abs(info['a']['path_m'] - 1.8 * 1.5) < 1e-3
    assert np.allclose(bases['a'][:, 2] - poses[:, 2], np.pi / 2, atol=3e-3)         # it heads across the payload throughout


def test_drawn_heading_is_that_of_a_base_that_turns_about_a_point_behind_its_origin():
    P = line_and_arc([0., 0.], [3., 2.5, np.pi / 2], 1.5, spacing=.002)
    x0 = -.13
    heading = drawn_headings(P[:, :2], x0, P[0, 2])
    d = np.gradient(P[:, :2], axis=0)
    sideways = -np.sin(heading) * d[:, 0] + np.cos(heading) * d[:, 1]                # of the origin, in base axes, per step
    assert np.abs((sideways + x0 * np.gradient(heading))[2:-2]).max() < 1e-5 * np.linalg.norm(d, axis=1).mean() / .002
    # on the circle the heading lags the path direction by asin(|x0| / R), like a trailer behind its hitch
    assert abs((P[:, 2] - heading)[-300] - np.arcsin(-x0 / 1.5)) < 2e-3
    assert abs(heading[0] - P[0, 2]) < 1e-12


def test_drawn_lag_is_the_solution_of_the_trailer_equation():
    from scipy.integrate import solve_ivp
    rng = np.random.default_rng(0)
    cases = 0
    for _ in range(300):
        length, step = 10. ** rng.uniform(-3., -.5), 10. ** rng.uniform(-3., -.7)
        # curvature times length: gentle, near the limit, and tighter than a settled lag allows; and straight on
        k = rng.uniform(-1., 1.) * rng.choice([.2, .9, 1.5, 3.]) / length * rng.choice([1., .1, 0.])
        if abs(k * step) > 2.5:
            continue
        e0 = rng.uniform(-1.2, 1.2)
        exact = solve_ivp(lambda s, e: k - np.sin(e) / length, (0., step), [e0], rtol=1e-11, atol=1e-13, method='LSODA').y[0, -1]
        assert abs((drawn_lag(e0, k * step, step, length) - exact + np.pi) % (2. * np.pi) - np.pi) < 1e-8
        cases += 1
    assert cases > 200
    assert drawn_lag(.3, .2, 0., .05) == .5                                          # no travel: the path direction alone changes
    # on a circle the lag settles at asin(length / radius) however long the steps are against the length
    for step in (.002, .02, .2):
        e = 0.
        for _ in range(int(3. / step)):
            e = drawn_lag(e, step / 1.5, step, .008)
        assert abs(e - np.arcsin(.008 / 1.5)) < 1e-12


def test_drawn_heading_does_not_depend_on_the_spacing_of_the_points():
    # the way into a station: an arc of 1.5 m radius, the turning point 8 mm behind the origin, points 20 mm apart
    x0 = -.008
    coarse = line_and_arc([0., 0.], [3., 2.5, np.pi / 2], 1.5, spacing=.02)
    fine = line_and_arc([0., 0.], [3., 2.5, np.pi / 2], 1.5, spacing=.0005)
    lag_coarse = (coarse[:, 2] - drawn_headings(coarse[:, :2], x0, coarse[0, 2]))[-40]
    lag_fine = (fine[:, 2] - drawn_headings(fine[:, :2], x0, fine[0, 2]))[-1600]
    assert abs(lag_fine - np.arcsin(-x0 / 1.5)) < 2e-5 and abs(lag_coarse - lag_fine) < 2e-5


def test_drawn_heading_runs_ahead_when_the_turning_point_is_in_front():
    P = line_and_arc([0., 0.], [3., 2.5, np.pi / 2], 1.5, spacing=.002)
    straight = np.vstack([P[:, :2], P[-1, :2] + np.linspace(.002, 1., 500)[:, None] * unit(np.pi / 2)])
    heading = drawn_headings(straight, .05, P[0, 2], end=np.pi / 2)
    phi, _ = tangent_headings(straight, P[0, 2])
    k = len(P) - 300
    assert abs((heading - phi)[k] - np.arcsin(.05 / 1.5)) < 2e-3                     # ahead of the path direction on the circle
    assert abs(heading[-1] - np.pi / 2) < 1e-12 and abs(heading[0] - phi[0]) < 1e-6  # straight at both ends: the path direction
    d = np.gradient(straight, axis=0)
    sideways = -np.sin(heading) * d[:, 0] + np.cos(heading) * d[:, 1]
    assert np.abs((sideways + .05 * np.gradient(heading))[2:-2]).max() < 1e-5


def test_drawn_heading_meets_the_path_direction_where_the_turning_point_passes_the_origin():
    P = line_and_arc([0., 0.], [3., 2.5, np.pi / 2], 1.5, spacing=.002)
    ahead = np.linspace(-.13, .05, len(P))
    heading = drawn_headings(P[:, :2], ahead, P[0, 2])
    phi, _ = tangent_headings(P[:, :2], P[0, 2])
    k = int(np.argmin(np.abs(ahead)))
    assert abs(heading[k] - phi[k]) < 2e-3
    assert np.abs(np.diff(heading)).max() < 3e-3                                     # continuous through the change of sign
    d = np.gradient(P[:, :2], axis=0)
    sideways = -np.sin(heading) * d[:, 0] + np.cos(heading) * d[:, 1]
    assert np.abs((sideways + ahead * np.gradient(heading))[2:-2]).max() < 1e-5


def test_smoothing_leaves_straight_runs_and_end_poses():
    corner = np.vstack([line([0., 0., 0.], [3., 0., 0.]), line([3., 0., 0.], [3., .3, np.pi / 2])[1:],
                        line([3., .3, np.pi / 2], [3., 4., np.pi / 2])[1:]])
    S = smooth_poses(corner, 2.5, .5)
    assert np.allclose(S[0], corner[0]) and np.allclose(S[-1], corner[-1])
    assert np.allclose(S[:100, 1:], 0., atol=1e-6)                                   # the first metre: still on the first line
    assert np.all(np.diff(S[:, 2]) >= -1e-12) and np.abs(np.diff(S[:, 2])).max() < .01   # the turn is spread out
    assert abs(arc_length(S[:, :2])[-1] - 7.) < .4
