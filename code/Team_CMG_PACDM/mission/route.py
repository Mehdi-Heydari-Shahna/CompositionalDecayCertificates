"""Planar route of the carried payload: geometry, speed zones and clearance-aware design.

A route is a curve of payload planar poses ``(x, y, yaw)``.  It is parameterized
by a generalized length ``l`` in which a rotation counts as ``rho * |d yaw|``
(``rho`` is the payload half-length), so a unit rate bounds the speed of the
payload centre and of its ends at the same time.

The team moves along it as a formation: every base keeps a fixed planar offset
from the payload.  Clearance of a formation pose is evaluated on the full
three-dimensional shapes of the robots and the payload against the scene.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


def planar(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    T[:2, 3] = [x, y]
    return T


def planar_inverse(T):
    out = np.eye(4)
    out[:2, :2] = T[:2, :2].T
    out[:2, 3] = -T[:2, :2].T @ T[:2, 3]
    return out


def compose(a, b):
    """Planar pose ``a`` followed by the relative planar pose ``b`` (both ``[x, y, yaw]``)."""
    c, s = np.cos(a[2]), np.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]])


def relative(a, b):
    """Planar pose of ``b`` seen from ``a``."""
    c, s = np.cos(a[2]), np.sin(a[2])
    d = np.asarray(b[:2], float) - np.asarray(a[:2], float)
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], b[2] - a[2]])


class PlanarPath:
    """Smooth planar pose curve with unit generalized speed."""

    def __init__(self, samples, rho, smooth=0., step=.01):
        P = np.asarray(samples, float)
        chord = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0) * [1., 1., rho], axis=1))]    # of every sample given
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0) * [1., 1., rho], axis=1) > 1e-9]
        P = P[keep]
        self.rho = float(rho)
        u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0) * [1., 1., rho], axis=1))]
        fine = np.arange(0., u[-1] + step / 2, step)
        if smooth > 0.:
            # analytic pieces (lines, arcs): resample linearly, then round their junctions;
            # the end poses are kept exactly
            F0 = np.column_stack([np.interp(fine, u, P[:, k]) for k in range(3)])
            pad = int(round(4 * smooth / step))
            G = np.vstack([np.repeat(F0[:1], pad, axis=0), F0, np.repeat(F0[-1:], pad, axis=0)])
            F = gaussian_filter1d(G, smooth / step, axis=0, mode='nearest')[pad:-pad]
            w = np.clip(np.minimum(fine, fine[-1] - fine) / (3 * smooth), 0., 1.)[:, None]
            w = w * w * (3. - 2. * w)
            F = w * F + (1. - w) * F0
        else:
            # samples of an already smooth curve: interpolate them smoothly (a linear resampling
            # would leave a piecewise-constant tangent, i.e. steps in the velocity reference)
            F = CubicSpline(u, P, bc_type='natural')(fine)
        l = np.r_[0., np.cumsum(np.linalg.norm(np.diff(F, axis=0) * [1., 1., rho], axis=1))]
        keep = np.r_[True, np.diff(l) > 1e-12]
        self.spline = CubicSpline(l[keep], F[keep], bc_type='natural')
        self.length = float(l[-1])
        self.sample_l = np.interp(chord, fine, l)                     # path parameter of every sample given

    def __call__(self, l):
        """Pose, tangent and curvature vector ``(p, dp/dl, d2p/dl2)`` at ``l``."""
        l = float(np.clip(l, 0., self.length))
        return self.spline(l), self.spline(l, 1), self.spline(l, 2)

    def sample(self, n):
        l = np.linspace(0., self.length, n)
        return l, self.spline(l)

    def locate(self, predicate, start=0., step=.01):
        """First path parameter at which ``predicate(pose)`` holds (None if never)."""
        for l in np.arange(start, self.length + step / 2, step):
            if predicate(self.spline(min(l, self.length))):
                return float(min(l, self.length))
        return None


class SpeedZones:
    """Piecewise speed limit along a path with linear ramps between zones."""

    def __init__(self, default, zones=(), ramp=.6):
        self.default, self.zones, self.ramp = float(default), [tuple(z) for z in zones], float(ramp)

    def __call__(self, l):
        v = self.default
        for a, b, limit in self.zones:
            if a - self.ramp <= l <= b + self.ramp:
                d = max(a - l, l - b, 0.) / self.ramp
                v = min(v, limit + (self.default - limit) * d) if limit < self.default else v
        return v


class FormationClearance:
    """Clearance of the team, moved rigidly as a formation, against the scene."""

    def __init__(self, geometry, T_bodies, reference_pose, roles=('wall', 'door', 'rack', 'neighbour', 'mount', 'guide', 'pad'),
                 cutoff=1.):
        self.geometry, self.T, self.cutoff = geometry, T_bodies, cutoff
        self.inv_ref = planar_inverse(planar(*reference_pose))
        self.roles = set(roles)
        self.calls = 0

    def __call__(self, pose, detail=False):
        self.calls += 1
        placed = self.geometry.place(self.T, planar=planar(*pose) @ self.inv_ref)
        items = self.geometry.against_statics(placed, self.cutoff, skip=lambda a, b: b.role not in self.roles)
        if not items:
            return (self.cutoff, None) if detail else self.cutoff
        best = min(items, key=lambda item: item[0])
        return (best[0], best) if detail else best[0]

    def top(self, groups=None):
        return self.geometry.top(self.T, groups)


def max_clearance_turn(clearance, entry, exit_, steps=25, cap=2., pull=.02, nominal=None):
    """Formation poses through a turn that keep the largest clearance at every heading.

    ``entry`` and ``exit_`` are the poses on the two lanes where the turn starts and
    ends.  For each intermediate heading the position maximizing the smallest
    clearance (capped at ``cap``) is found by a local search started from the
    previous solution; a weak pull towards a nominal arc regularizes the position
    where the clearance is flat.  The positions are then made monotone along both
    lane directions, so the formation never backs up.  Returns the poses and their
    clearances.
    """
    entry, exit_ = np.asarray(entry, float), np.asarray(exit_, float)
    yaws = np.linspace(entry[2], exit_[2], steps)
    if nominal is None:
        # quarter ellipse tangent to both lanes
        def nominal(s):
            a = .5 * np.pi * s
            return np.array([entry[0] + (exit_[0] - entry[0]) * np.sin(a), exit_[1] - (exit_[1] - entry[1]) * np.cos(a)])
    poses = [entry.copy()]
    guess = entry[:2].copy()
    for k in range(1, steps - 1):
        nom = nominal(k / (steps - 1))

        def cost(p, yaw=yaws[k], nom=nom):
            return -min(clearance([p[0], p[1], yaw]), cap) + pull * float((p - nom) @ (p - nom))

        best = None
        for start in (guess, nom):
            res = minimize(cost, start, method='Nelder-Mead', options=dict(xatol=2e-3, fatol=1e-4, maxiter=120,
                                                                         initial_simplex=start + np.array([[0, 0], [.15, 0], [0, .15]])))
            if best is None or res.fun < best.fun:
                best = res
        guess = best.x
        poses.append(np.array([best.x[0], best.x[1], yaws[k]]))
    poses.append(exit_.copy())
    poses = np.array(poses)
    for axis in (0, 1):
        sign = np.sign(exit_[axis] - entry[axis]) or 1.
        poses[:, axis] = sign * np.minimum(np.maximum.accumulate(sign * poses[:, axis]), sign * exit_[axis])
    return poses, np.array([clearance(p) for p in poses])


def bottleneck_turn(clearance, entry, exit_, steps=25, spacing=.15, step_max=.6, keep=.85):
    """Formation poses through a turn with the largest possible smallest clearance, by dynamic programming.

    The local search of :func:`max_clearance_turn` follows the best position from one heading to the
    next; a formation whose best position jumps between headings can be led into a place that later
    headings close.  Here the rectangle between the two lanes is scanned on a grid at every heading.
    Among the sequences of grid positions that never back up and move at most ``step_max`` per heading
    step, the largest attainable smallest clearance is found first; then, among the positions that keep
    at least ``keep`` of it, the sequence with the smallest sum of squared steps is taken (an even
    advance).  Returns the poses and their clearances.
    """
    entry, exit_ = np.asarray(entry, float), np.asarray(exit_, float)
    yaws = np.linspace(entry[2], exit_[2], steps)
    n = [max(int(round(abs(exit_[a] - entry[a]) / spacing)), 1) + 1 for a in (0, 1)]
    X, Y = np.linspace(entry[0], exit_[0], n[0]), np.linspace(entry[1], exit_[1], n[1])
    reach = [max(int(np.floor(step_max / max(abs(g[1] - g[0]), 1e-9) + 1e-9)), 1) for g in (X, Y)]
    C = np.zeros((steps, n[0], n[1]))
    for k in range(1, steps - 1):
        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                C[k, i, j] = clearance([x, y, yaws[k]])
    C[0], C[-1] = -np.inf, -np.inf
    C[0, 0, 0] = clearance(entry)
    C[-1, -1, -1] = clearance(exit_)

    def windows(i, j):
        return slice(max(i - reach[0], 0), i + 1), slice(max(j - reach[1], 0), j + 1)

    # largest attainable smallest clearance
    B = np.full_like(C, -np.inf)
    B[0, 0, 0] = C[0, 0, 0]
    for k in range(1, steps):
        for i in range(n[0]):
            for j in range(n[1]):
                if np.isfinite(C[k, i, j]):
                    wi, wj = windows(i, j)
                    B[k, i, j] = min(C[k, i, j], B[k - 1, wi, wj].max())
    best = B[-1, -1, -1]
    if not np.isfinite(best) or best <= 0.:
        raise RuntimeError('turn: no sequence of positions keeps the formation clear')
    # smallest sum of squared steps among the positions that keep most of it
    ok = C >= keep * best
    ok[0, 0, 0] = ok[-1, -1, -1] = True
    D = np.full_like(C, np.inf)
    D[0, 0, 0] = 0.
    back = np.zeros(C.shape + (2,), int)
    for k in range(1, steps):
        for i in range(n[0]):
            for j in range(n[1]):
                if not ok[k, i, j]:
                    continue
                wi, wj = windows(i, j)
                step = (X[i] - X[wi])[:, None] ** 2 + (Y[j] - Y[wj])[None, :] ** 2
                total = D[k - 1, wi, wj] + step
                a = np.unravel_index(int(np.argmin(total)), total.shape)
                D[k, i, j] = total[a]
                back[k, i, j] = (wi.start + a[0], wj.start + a[1])
    if not np.isfinite(D[-1, -1, -1]):
        raise RuntimeError('turn: no sequence of positions keeps the required share of the best clearance')
    cells = [(n[0] - 1, n[1] - 1)]
    for k in range(steps - 1, 0, -1):
        cells.append(tuple(back[k][cells[-1]]))
    cells = cells[::-1]
    poses = np.array([[X[i], Y[j], yaw] for (i, j), yaw in zip(cells, yaws)])
    return poses, np.array([clearance(p) for p in poses])


def smooth_poses(samples, rho, sigma, step=.01):
    """Pose samples resampled at ``step`` of the generalized length and smoothed over ``sigma`` (same length).

    For pose sequences that are not smooth in themselves (positions found on a grid).  Both ends must be
    straight runs longer than ``3 sigma``: the smoothing does not change those, and the end poses are kept.
    """
    P = np.asarray(samples, float)
    u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0) * [1., 1., rho], axis=1))]
    keep = np.r_[True, np.diff(u) > 1e-9]
    fine = np.linspace(0., u[-1], max(int(round(u[-1] / step)), 1) + 1)                # even steps, the end pose included
    step = fine[1] - fine[0]
    F0 = np.column_stack([np.interp(fine, u[keep], P[keep, k]) for k in range(3)])
    pad = int(round(4 * sigma / step))
    ahead = np.arange(1, pad + 1)[:, None]
    G = np.vstack([F0[0] - ahead[::-1] * (F0[1] - F0[0]), F0, F0[-1] + ahead * (F0[-1] - F0[-2])])
    return gaussian_filter1d(G, sigma / step, axis=0, mode='nearest')[pad:-pad]


def densify(poses, n=200):
    """Shape-preserving interpolation of a pose sequence (parameter: cumulative planar distance and heading)."""
    P = np.asarray(poses, float)
    u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(P, axis=0) * [1., 1., 2.5], axis=1))]
    f = PchipInterpolator(u, P, axis=0)
    return f(np.linspace(0., u[-1], n))


def line(a, b, spacing=.05):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.ceil(np.linalg.norm((b - a) * [1., 1., 2.5]) / spacing)), 1)
    return a + (b - a) * np.linspace(0., 1., n + 1)[:, None]


def lane_change(a, b, spacing=.05):
    """Straight run from ``a`` to ``b`` in which the offset across the run direction blends smoothly."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(int(np.ceil(np.linalg.norm((b - a)[:2]) / spacing)), 1)
    s = np.linspace(0., 1., n + 1)
    w = s * s * s * (10. + s * (-15. + 6. * s))
    major = int(np.argmax(np.abs((b - a)[:2])))
    out = a + (b - a) * w[:, None]
    out[:, major] = a[major] + (b[major] - a[major]) * s
    return out
