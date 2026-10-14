"""Paths for bases that cannot move sideways (skid steering, differential drive).

Such a base drives along its heading and turns; the velocity of its origin has no
component across the heading that could be chosen.  A team of them cannot move as
a rigid formation unless the formation moves along the common heading.  What
keeps the task possible is that the arm, not the base, holds the payload: a base
may turn under its arm while the grasp stays where it is.

The paths here are paths of the base origin.  The heading follows from them: the
direction of the path for a base that turns about its origin, and the solution of
:func:`drawn_headings` for a base that turns about another point of its centre
line, as a skid-steered base under uneven wheel loads does.

* :func:`line_and_arc`: from a point straight to a circle and round it into a goal
  pose (the way into a station);
* :func:`turn_in_place`: a turn on the spot about the turning centre the skid
  model gives;
* :func:`hermite`: a smooth curve between two poses with no curvature at its ends;
* :func:`yaw_on_line`: the payload yaw that keeps another base on a straight line
  while the leading base follows its path;
* :func:`led_poses`: payload poses for a base that leads on a designed path while
  the payload turns about it;
* :func:`follow_tangents`: the path of every base origin, and the direction it
  moves in, when the origin keeps a fixed position on a moving payload;
* :func:`drawn_headings`: the heading of a base along the path of its origin when
  it turns about a point in front of or behind the origin.

Everything is planar geometry in NumPy; nothing here knows a platform.
"""
from __future__ import annotations

import numpy as np


def smoothstep(s):
    """Quintic blend 0 -> 1 with zero first and second derivatives at both ends."""
    s = np.clip(np.asarray(s, float), 0., 1.)
    return s * s * s * (10. + s * (-15. + 6. * s))


def unit(angle):
    return np.array([np.cos(angle), np.sin(angle)])


def hermite(a, heading_a, b, heading_b, n=200, weights=(1., 1.)):
    """Quintic curve from ``a`` to ``b`` that leaves and arrives along the given headings without curvature.

    ``weights`` scale the two end tangents (in units of the distance between the points): a larger weight
    keeps the curve longer on its end direction.  Returns ``(n, 2)`` points.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    ta, tb = weights[0] * L * unit(heading_a), weights[1] * L * unit(heading_b)
    s = np.linspace(0., 1., n)[:, None]
    s3, s4, s5 = s ** 3, s ** 4, s ** 5
    return ((1. - 10. * s3 + 15. * s4 - 6. * s5) * a + (s - 6. * s3 + 8. * s4 - 3. * s5) * ta
            + (-4. * s3 + 7. * s4 - 3. * s5) * tb + (10. * s3 - 15. * s4 + 6. * s5) * b)


def arc_length(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(np.asarray(points, float), axis=0), axis=1))]


def tangent_headings(points, start):
    """Heading of a base that follows a point path forwards: the direction of the path, continuous from ``start``.

    Where the path stands still the heading of the last moving sample is kept.  Returns ``(headings, speed)``
    with the step lengths, so that the caller can see where a base hardly moves.
    """
    P = np.asarray(points, float)
    d = np.gradient(P, axis=0)
    step = np.linalg.norm(d, axis=1)
    out = np.zeros(len(P))
    last = float(start)
    floor = 1e-3 * float(np.median(step[step > 0.])) if np.any(step > 0.) else 0.
    for k in range(len(P)):
        if step[k] > floor:
            a = np.arctan2(d[k, 1], d[k, 0])
            last = last + (a - last + np.pi) % (2. * np.pi) - np.pi
        out[k] = last
    return out, step


def turn_in_place(pose, yaw_to, x0, n=40):
    """Poses ``(n, 3)`` of a base that turns on the spot from ``pose`` to the heading ``yaw_to``.

    The base turns about its turning centre, ``x0`` ahead of its origin on its centre line; the origin moves
    on a circle of that radius.
    """
    pose = np.asarray(pose, float)
    pivot = pose[:2] + x0 * unit(pose[2])
    yaw = np.linspace(pose[2], yaw_to, n)
    return np.c_[pivot[0] - x0 * np.cos(yaw), pivot[1] - x0 * np.sin(yaw), yaw]


def line_and_arc(start, goal, radius, spacing=.02):
    """Straight line from the point ``start`` to a circle of ``radius`` and round it into the pose ``goal``.

    Of the two circles that touch the goal line at the goal (one on each side) the one giving the shorter way
    is taken.  Returns the poses ``(n, 3)`` with the heading along the path.
    """
    start, goal = np.asarray(start, float)[:2], np.asarray(goal, float)
    best = None
    for side in (1., -1.):                                  # +1: centre to the left of the goal heading (counter-clockwise arc)
        centre = goal[:2] + side * radius * unit(goal[2] + np.pi / 2)
        d = centre - start
        D = float(np.linalg.norm(d))
        if D <= radius * 1.0001:
            continue
        alpha = np.arcsin(radius / D)
        heading = np.arctan2(d[1], d[0]) - side * alpha     # the centre stays on the side the arc turns to
        touch = start + np.sqrt(D * D - radius * radius) * unit(heading)
        sweep = (side * (goal[2] - heading)) % (2. * np.pi)  # angle turned on the circle, in its own sense
        length = float(np.linalg.norm(touch - start) + radius * sweep)
        if best is None or length < best[0]:
            best = (length, side, centre, heading, touch, sweep)
    if best is None:
        raise ValueError('line_and_arc: the start lies inside both circles of the goal; use a smaller radius')
    _, side, centre, heading, touch, sweep = best
    n1 = max(int(np.ceil(np.linalg.norm(touch - start) / spacing)), 1)
    line = start + (touch - start) * np.linspace(0., 1., n1 + 1)[:, None]
    n2 = max(int(np.ceil(radius * sweep / spacing)), 1)
    a = heading + side * sweep * np.linspace(0., 1., n2 + 1)[1:]
    arc = centre[None, :] - side * radius * np.c_[np.cos(a + np.pi / 2), np.sin(a + np.pi / 2)]
    return np.vstack([np.c_[line, np.full(len(line), heading)], np.c_[arc, a]])


def yaw_on_line(lead, delta, point, heading, yaw_start):
    """Payload yaw ``(n,)`` that keeps a second base on a straight line while the leading base is at ``lead``.

    ``delta``: position of the leading base less that of the second one, in payload axes.  The line passes
    through ``point`` along ``heading``.  Of the two solutions the one continuous from ``yaw_start`` is taken.
    The second base is at ``lead - R(yaw) delta``; it is on the line when the component of that point across
    the line vanishes.
    """
    lead = np.asarray(lead, float)
    normal = unit(heading + np.pi / 2)
    e = (lead - np.asarray(point, float)[:2]) @ normal
    A, B = delta[0] * normal[0] + delta[1] * normal[1], delta[0] * normal[1] - delta[1] * normal[0]
    size = np.hypot(A, B)
    if np.any(np.abs(e) > size):
        raise ValueError('yaw_on_line: the leading base is further from the line than the two bases are apart')
    centre, spread = np.arctan2(B, A), np.arccos(np.clip(e / size, -1., 1.))
    out, last = np.zeros(len(lead)), float(yaw_start)
    for k in range(len(lead)):
        cand = [centre + spread[k], centre - spread[k]]
        cand = [last + (x - last + np.pi) % (2. * np.pi) - np.pi for x in cand]
        last = min(cand, key=lambda x: abs(x - last))
        out[k] = last
    return out


def led_poses(lead, offset, yaw):
    """Payload poses ``(n, 3)`` when the base that holds it at ``offset`` (payload axes) follows the path ``lead``.

    ``yaw``: the payload yaw at every point of the path.
    """
    lead, psi = np.asarray(lead, float), np.asarray(yaw, float)
    c, s_ = np.cos(psi), np.sin(psi)
    return np.c_[lead[:, 0] - (c * offset[0] - s_ * offset[1]), lead[:, 1] - (s_ * offset[0] + c * offset[1]), psi]


def follow_tangents(poses, offsets, start_headings):
    """Base poses ``{id: (n, 3)}`` along payload poses ``(n, 3)`` when every base origin keeps its ``offset``.

    The heading of a base is the direction its origin moves in.  Also returns diagnostics per base: the
    smallest step of its origin relative to its median step (a base that almost stops while the others move
    turns sharply), and its largest heading change per metre of its own path.
    """
    P = np.asarray(poses, float)
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    out, info = {}, {}
    for rid, o in offsets.items():
        xy = np.c_[P[:, 0] + c * o[0] - s * o[1], P[:, 1] + s * o[0] + c * o[1]]
        heading, step = tangent_headings(xy, start_headings[rid])
        out[rid] = np.c_[xy, heading]
        moving = step > 1e-9
        rate = np.abs(np.gradient(heading)[moving]) / step[moving]
        info[rid] = dict(slowest_step_ratio=float(step[moving].min() / np.median(step[moving])),
                         max_turn_rad_per_m=float(rate.max()), path_m=float(arc_length(xy)[-1]),
                         relative_heading_rad=[float((heading - P[:, 2]).min()), float((heading - P[:, 2]).max())])
    return out, info


def drawn_lag(e, turn, step, length):
    """Path direction less heading after one step of the origin path, for a base that turns ``length`` behind its origin.

    ``e`` is that difference at the start of the step, ``turn`` the change of the path direction over the step
    (taken at a constant rate) and ``step`` its arc length.  The difference obeys

        de/ds = k - sin(e) / length,        k = turn / step .

    In ``t = tan(e / 2)`` this is the Riccati equation ``dt/ds = (k / 2)(t^2 + 1) - t / length`` with constant
    coefficients, which is solved in closed form here, so the step may be long against ``length``.  With
    ``c = k length`` and ``|c| < 1`` the two constant solutions are ``t = (1 -+ sqrt(1 - c^2)) / c``; the first is
    the settled lag of a trailer on a circle, ``sin(e) = c``, and the solution approaches it as
    ``exp(-sqrt(1 - c^2) s / length)``.  With ``|c| > 1`` the path turns tighter than the base can follow at a
    settled lag, and the difference keeps growing.
    """
    if step <= 1e-12:
        return e + turn                                          # the path direction changes on the spot
    t0 = np.tan(.5 * e)
    c = turn / step * length
    if abs(c) < 1e-9:                                            # straight on: t = c / 2 is the settled value
        return 2. * np.arctan(.5 * c + (t0 - .5 * c) * np.exp(-step / length))
    if abs(c) < 1.:
        root = np.sqrt(1. - c * c)
        settled, other = (1. - root) / c, (1. + root) / c
        if t0 == other:
            return e
        r = (t0 - settled) / (t0 - other) * np.exp(-root * step / length)
        return 2. * np.arctan((settled - r * other) / (1. - r))
    if abs(c) == 1.:
        w0 = t0 - 1. / c
        return 2. * np.arctan(1. / c + w0 / (1. - .5 * turn / step * step * w0))
    m = np.sqrt(1. - 1. / (c * c))
    w = m * np.tan(.5 * m * turn + np.arctan((t0 - 1. / c) / m))
    return 2. * np.arctan(1. / c + w)


def drawn_headings(points, ahead, start, end=None):
    """Heading ``(n,)`` of a base whose origin follows ``points`` while it turns about a point of its centre line.

    A base on conventional wheels turns about a point ``ahead`` metres in front of its origin (behind it when
    negative); the skid model gives that point for the wheel loads.  It moves along the heading, so the origin,
    which lies off it, also moves sideways when the base turns.  With ``phi`` the direction of the origin path,
    ``theta`` the heading and ``s`` the length of the origin path,

        d theta / ds = - sin(phi - theta) / ahead .

    With the turning point behind the origin the base follows its origin as a trailer follows its hitch: the
    heading lags the path direction (on a circle of radius ``R`` by ``asin(|ahead| / R)``), and the equation is
    integrated forwards from ``start``.  With the turning point in front the equation is stable backwards: it
    is integrated from ``end`` (default: the path direction there), and the heading runs ahead of the path
    direction.  ``ahead`` is one number or one per point; where it changes sign the origin itself is the
    turning point and the heading is the path direction.

    Between two points the path direction is taken to turn at a constant rate and ``ahead`` to be constant
    (the mean of its two values); the step itself is then exact (:func:`drawn_lag`), whatever its length
    against ``|ahead|``.
    """
    P = np.asarray(points, float)
    n = len(P)
    phi, _ = tangent_headings(P, start)
    a = np.broadcast_to(np.asarray(ahead, float), (n,))
    ds = np.linalg.norm(np.diff(P, axis=0), axis=1)
    sign = np.where(np.abs(a) < 1e-6, 0, np.sign(a)).astype(int)
    theta = phi.copy()
    k = 0
    while k < n:
        j = k
        while j + 1 < n and sign[j + 1] == sign[k]:
            j += 1
        if sign[k] < 0:                                          # run k..j, forwards
            theta[k] = float(start) if k == 0 else phi[k]
            for i in range(k, j):
                theta[i + 1] = phi[i + 1] - drawn_lag(phi[i] - theta[i], phi[i + 1] - phi[i], ds[i], .5 * abs(a[i] + a[i + 1]))
        elif sign[k] > 0:                                        # run k..j, backwards
            theta[j] = (phi[j] if end is None else float(end)) if j == n - 1 else phi[j]
            for i in range(j, k, -1):
                theta[i - 1] = phi[i - 1] - drawn_lag(phi[i] - theta[i], phi[i - 1] - phi[i], ds[i - 1], .5 * abs(a[i] + a[i - 1]))
        k = j + 1
    return theta
