"""Friction-driven bases: wheel slip, the quasi-static skid model and the friction loads (NumPy only).

A conventional wheel on a fixed axle rolls in one direction only.  A base that
stands on such wheels at both ends cannot turn by rolling: it turns by sliding
its wheels over the ground (skid steering).  For such a base the tangential
forces between wheels and ground are not reactions of rolling constraints.
They follow a friction law, and the motion of the base follows from it.

Declared law: isotropic Coulomb friction at the ground point of every wheel,

    F_i = - mu_i N_i s_i / |s_i|,

where ``s_i`` is the velocity of the wheel's material point at the ground (two
components in the ground plane: along the rolling direction and along the axle)
and ``N_i`` the normal load of the wheel.

Quasi-static model.  At the speeds of a transport task the inertia forces of a
base are small against its friction capacity, so the planar twist ``z = [vx, vy,
wz]`` of the base for given wheel speeds is the one at which the friction forces
balance.  The balance equations are the stationarity conditions of the
dissipated power

    D(z) = sum_i mu_i N_i |s_i(z)|,      s_i(z) = A_i z - [u_i, 0],

with ``u_i`` the surface speed of wheel ``i``.  ``D`` is convex, so the twist is
its minimizer: the base moves the way that dissipates the least power.  With
equal loads on four wheels (wheelbase ``L``, track ``B``) the minimizer is the
motion of a differential drive whose track is ``chi B`` with ``chi = 1 +
(L / B)^2``, turning about its centre.  Unequal loads change that: the turning
centre moves towards the heavier axle and away from the heavier side, and
``chi`` changes with it.  That is why the model takes the wheel loads of the
present configuration from the compiled reactions instead of one tuned factor.

The twist is homogeneous of degree one in the wheel speeds.  With the wheels of
each side at one speed, ``u = m - side * delta`` (``side`` +1 left, -1 right),
it is linear in ``(m, delta)``:

    z = [m, 0, 0] + delta * c,      c = argmin D for m = 0, delta = 1,

so three numbers describe the base at given loads: the yaw rate per unit speed
difference ``c[2]`` and the turning centre ``(x0, y0) = (-c[1], c[0]) / c[2]``.

What is not in the model: planar loads on the base other than the friction of its
own wheels (slope, inertia, forces from the arm).  They are small against the
friction capacity in the missions here and are taken up by the base loop; the
audit compares the measured twist of every base with the one of this model.
"""
from __future__ import annotations

import numpy as np

SLIP_SPEED = 1e-3          # m/s; default slip speed over which the friction force of the compiled loads builds up


def slip_rows(xy, rolling, lateral):
    """``A`` (n, 2, 3): slip of the base-fixed ground points for a unit planar twist ``[vx, vy, wz]`` (base axes).

    ``xy`` are the ground points in the base frame, ``rolling`` and ``lateral`` the two unit
    directions of every wheel in the ground plane.  The velocity of a base-fixed point is
    ``[vx - wz y, vy + wz x]``.
    """
    xy = np.asarray(xy, float)
    A = np.zeros((len(xy), 2, 3))
    for i, (p, g, e) in enumerate(zip(xy, np.asarray(rolling, float), np.asarray(lateral, float))):
        point = np.array([[1., 0., -p[1]], [0., 1., p[0]]])
        A[i, 0] = g @ point
        A[i, 1] = e @ point
    return A


def twist(A, weight, surface_speed, tol=1e-12, start=None):
    """Planar twist of a base whose wheels turn at given surface speeds: the minimizer of the dissipated power.

    ``weight`` is ``mu_i N_i``.  Newton steps on the smoothed power ``sum w sqrt(|s|^2 + eps^2)`` with
    ``eps`` reduced to ``tol`` times the largest wheel speed; at a wheel that does not slip the power has
    a kink, which the continuation approaches from the smooth side.  ``start`` is an optional first guess
    (the twist of a neighbouring load case).  Returns ``(z, slips)``.
    """
    w = np.maximum(np.asarray(weight, float), 0.)
    u = np.asarray(surface_speed, float)
    b = np.zeros((len(u), 2))
    b[:, 0] = u
    scale = float(np.max(np.abs(u)))
    if scale == 0. or w.sum() <= 0.:
        return np.zeros(3), np.zeros((len(u), 2))
    AA = np.einsum('ijk,ijl->ikl', A, A)
    Ab = np.einsum('ijk,ij->ik', A, b)
    if start is None:
        # weighted least squares: the minimizer for a friction force proportional to the slip
        z = np.linalg.solve(np.tensordot(w, AA, 1), w @ Ab)
        stages = [1e-2, 1e-4, 1e-6, 1e-9, tol]
    else:
        z = np.asarray(start, float).copy()
        stages = [1e-6, 1e-9, tol]
    for stage in stages:
        eps = stage * scale
        e2 = eps * eps
        for _ in range(60):
            s = A @ z - b
            n = np.sqrt((s * s).sum(axis=1) + e2)
            wn = w / n
            As = AA @ z - Ab                                           # A_i^T s_i
            g = wn @ As
            H = np.tensordot(wn, AA, 1) - (As * (wn / (n * n))[:, None]).T @ As
            step = np.linalg.solve(H, -g)
            f0, slope, t = float(w @ n), float(g @ step), 1.
            while t > 1e-10:
                s1 = A @ (z + t * step) - b
                if float(w @ np.sqrt((s1 * s1).sum(axis=1) + e2)) <= f0 + 1e-4 * t * slope:
                    break
                t *= .5
            z = z + t * step
            if np.max(np.abs(t * step)) < 1e-13 * scale:
                break
    return z, A @ z - b


def skid_map(xy, rolling, lateral, friction, loads, radius, start=None):
    """Quasi-static skid model of a base at given wheel loads.

    Returns a dict with

    * ``c``: twist ``[vx, vy, wz]`` of the base per unit speed difference ``delta`` (left wheels at surface
      speed ``-delta``, right wheels at ``+delta``);
    * ``track_factor``: ``chi``, the track of the equivalent differential drive over the track of the wheels;
    * ``centre``: the turning centre ``[x0, y0]`` in the base frame (the point that stays at rest when the
      wheels of the two sides turn at opposite speeds);
    * ``moment_Nm``: the moment the wheels must overcome to turn the base (dissipated power per unit yaw rate);
    * ``slip_per_yaw_m``: slip speed of every wheel per unit yaw rate;
    * ``wheel_rate``: ``(n, 2)`` map from ``[vx, wz]`` (base axes) to the wheel rates;
    * ``side``: +1 for the wheels left of the centre line of the wheels, -1 for the others.

    ``start`` is an optional first guess of ``c`` (the value at neighbouring loads).
    """
    xy = np.asarray(xy, float)
    A = slip_rows(xy, rolling, lateral)
    mid = float(xy[:, 1].mean())
    side = np.where(xy[:, 1] >= mid, 1., -1.)
    if abs(side.sum()) == len(side):
        raise ValueError('skid model: the wheels of a base must lie on both sides of its centre line')
    track = float(xy[side > 0, 1].mean() - xy[side < 0, 1].mean())
    w = np.asarray(friction, float) * np.maximum(np.asarray(loads, float), 0.)
    if np.count_nonzero(w > 0.) == 1:
        # the base can turn about the one wheel that grips at any rate: the dissipated power has no single minimizer
        raise ValueError('skid model: one wheel alone carries load, which leaves the motion of the base open')
    try:
        c, s = twist(A, w, -side, start=start)
    except np.linalg.LinAlgError:
        raise ValueError('skid model: these wheel loads leave the motion of the base open') from None
    if abs(c[2]) < 1e-9:
        raise ValueError('skid model: the wheels cannot turn the base (no friction on one side?)')
    slip = np.sqrt((s * s).sum(axis=1)) / abs(c[2])
    centre = np.array([-c[1], c[0]]) / c[2]
    radius = np.asarray(radius, float)
    rate = np.zeros((len(xy), 2))
    rate[:, 0] = 1. / radius
    rate[:, 1] = (-centre[1] - side / c[2]) / radius
    return dict(c=c, track_factor=float(2. / (c[2] * track)), centre=centre, track_m=track,
                moment_Nm=float(w @ slip), slip_per_yaw_m=slip, wheel_rate=rate, side=side, A=A)


def friction_forces(slip, loads, friction, slip_speed=SLIP_SPEED):
    """Friction force of every wheel ``(n, 2)`` for its slip ``(n, 2)``: Coulomb, built up over ``slip_speed``."""
    s = np.asarray(slip, float)
    w = np.asarray(friction, float) * np.maximum(np.asarray(loads, float), 0.)
    return -(w / np.sqrt((s * s).sum(axis=1) + slip_speed ** 2))[:, None] * s
