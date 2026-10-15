"""Piecewise-planar terrain and the ideal ground-following pose of a wheeled base.

Physical input: terrain patches of the scene record (a plane through a point
with a normal, valid over a rectangle in the world x-y plane).  The support
surface under a point is the highest patch covering it.

The ground coupling of a base is the holonomic relation

    [z, pitch, roll] = support(x, y, yaw),

obtained by fitting a plane to the terrain heights under the wheel centres.  It
is exact on a single plane and an explicit smooth approximation across a patch
boundary.  It contains no friction, unilateral or stability statement.
"""
from __future__ import annotations

import numpy as np

from .se3 import pose_chart, rot


def patch_mean_z(C):
    return float(C[:, 2].sum() / len(C))


class Terrain:
    def __init__(self, scene):
        self.patches = []
        for p in scene.get('terrain', {}).get('patches', []):
            n = np.asarray(p['normal'], float)
            n = n / np.linalg.norm(n)
            if n[2] <= .5:
                raise ValueError(f"terrain patch {p['id']}: normal must point up")
            lo, hi = np.asarray(p['region'][0], float), np.asarray(p['region'][1], float)
            self.patches.append(dict(id=p['id'], n=n, d=float(n @ np.asarray(p['point'], float)), lo=lo, hi=hi))
        if not self.patches:
            self.patches.append(dict(id='floor', n=np.array([0., 0., 1.]), d=0., lo=np.array([-1e9, -1e9]),
                                     hi=np.array([1e9, 1e9])))
        self._fast = [(float(p['lo'][0]), float(p['lo'][1]), float(p['hi'][0]), float(p['hi'][1]),
                       float(p['n'][0]), float(p['n'][1]), float(p['n'][2]), p['d'], p) for p in self.patches]

    def plane_at(self, x, y):
        best, z_best = None, -np.inf
        for p in self.patches:
            if p['lo'][0] <= x <= p['hi'][0] and p['lo'][1] <= y <= p['hi'][1]:
                z = (p['d'] - p['n'][0] * x - p['n'][1] * y) / p['n'][2]
                if z > z_best:
                    best, z_best = p, z
        if best is None:
            raise ValueError(f'No terrain patch under ({x:.3f}, {y:.3f})')
        return best, z_best

    def height(self, x, y):
        return self.plane_at(x, y)[1]

    def wheel_contact(self, centre, axis, radius):
        """Ideal contact of a thin wheel with the terrain.

        The wheel is a disc of ``radius`` about the unit ``axis`` through ``centre``.  Returns
        ``(height, normal, point)``: the clearance of the disc above the plane it touches (zero in
        contact), the unit normal of that plane and the point of the disc nearest to it.

        Candidates are the planes of the patches near the wheel.  A candidate is valid if its nearest
        disc point lies on the patch and the patch is the support surface there; the smallest clearance
        among the valid candidates is returned (at a concave crease the wheel touches the plane it meets
        first).  Where no candidate is valid, the nearest points lie beyond a convex edge; the largest
        clearance among the near patches is then used, which differs from rolling over the edge by
        ``radius (1 / cos(angle / 2) - 1)``: 0.03 mm for a 76 mm wheel on a 3 degree edge.
        """
        cx, cy, cz = float(centre[0]), float(centre[1]), float(centre[2])
        ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        valid, near = None, None
        for p in self._fast:                               # plain floats: this runs for every wheel at every evaluation
            lox, loy, hix, hiy, nx, ny, nz, d, patch = p
            if not (lox - radius <= cx <= hix + radius and loy - radius <= cy <= hiy + radius):
                continue
            na = nx * ax + ny * ay + nz * az
            s = (max(1. - na * na, 1e-300)) ** .5
            k = radius / s
            px, py, pz = cx - k * (nx - na * ax), cy - k * (ny - na * ay), cz - k * (nz - na * az)
            h = nx * cx + ny * cy + nz * cz - d - radius * s
            if near is None or h > near[0]:
                near = (h, patch, px, py, pz)
            if lox <= px <= hix and loy <= py <= hiy and (valid is None or h < valid[0]) and self._top(px, py) is patch:
                valid = (h, patch, px, py, pz)
        if near is None:
            raise ValueError(f'No terrain patch under ({cx:.3f}, {cy:.3f})')
        h, patch, px, py, pz = valid if valid is not None else near
        return h, patch['n'], np.array([px, py, pz])

    def _top(self, x, y):
        """The patch that is the support surface at ``(x, y)`` (plain floats), or ``None``."""
        best, z_best = None, -np.inf
        for lox, loy, hix, hiy, nx, ny, nz, d, patch in self._fast:
            if lox <= x <= hix and loy <= y <= hiy:
                z = (d - nx * x - ny * y) / nz
                if z > z_best:
                    best, z_best = patch, z
        return best

    def fit(self, points_xy):
        """Least-squares plane ``z = a x + b y + c`` through the terrain heights under the points."""
        P = np.asarray(points_xy, float)
        h = np.array([self.height(x, y) for x, y in P])
        A = np.c_[P, np.ones(len(P))]
        (a, b, c), *_ = np.linalg.lstsq(A, h, rcond=None)
        n = np.array([-a, -b, 1.])
        return n / np.linalg.norm(n), np.array([a, b, c]), float(np.max(np.abs(A @ np.array([a, b, c]) - h)))

    def _single_plane(self, x, y, yaw, C, radius):
        """Patch under all wheel contact points if they share one, else ``None``."""
        c, s = np.cos(yaw), np.sin(yaw)
        first = None
        for cx, cy in C[:, :2]:
            patch, _ = self.plane_at(x + c * cx - s * cy, y + s * cx + c * cy)
            if first is None:
                first = patch
            elif patch is not first:
                return None
        return first

    @staticmethod
    def _on_plane(patch, x, y, yaw, C, radius):
        n = patch['n']
        nx, ny, nz = n
        c, s = np.cos(yaw), np.sin(yaw)
        d = c * nx + s * ny                               # heading . n
        xb = np.array([c - d * nx, s - d * ny, -d * nz])
        xb /= np.sqrt(xb @ xb)
        yb_z = nx * xb[1] - ny * xb[0]                    # z component of n x xb
        pitch = np.arcsin(min(max(-xb[2], -1.), 1.))
        roll = np.arctan2(yb_z, nz)
        z = (patch['d'] - nx * x - ny * y) / nz + (radius - patch_mean_z(C)) / nz
        return np.array([z, pitch, roll])

    def support(self, xy_yaw, wheel_centres_base, radius):
        """Ground-following ``[z, pitch, roll]`` of a base whose wheel centres lie in its x-y plane.

        Returns the three coordinates, the unit normal of the fitted support plane
        and the largest wheel height misfit (zero on a single plane).
        """
        x, y, yaw = map(float, xy_yaw)
        C = np.asarray(wheel_centres_base, float)
        patch = self._single_plane(x, y, yaw, C, radius)
        if patch is not None:
            return self._on_plane(patch, x, y, yaw, C, radius), patch['n'], 0.
        R = rot([0, 0, 1.], yaw)
        z = self.height(x, y) + radius
        n, misfit = np.array([0., 0., 1.]), 0.
        for _ in range(4):
            centres = np.array([x, y, z]) + C @ R.T
            n, (a, b, c), misfit = self.fit(centres[:, :2] - radius * n[:2])
            heading = np.array([np.cos(yaw), np.sin(yaw), 0.])
            xb = heading - (heading @ n) * n
            xb /= np.linalg.norm(xb)
            R = np.column_stack([xb, np.cross(n, xb), n])
            # the wheel-centre plane is the support plane shifted by the radius along its normal
            z = a * x + b * y + c + (radius - C[:, 2].mean()) / n[2]
        chart = pose_chart(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]), yaw_near=yaw)
        return np.array([z, chart[4], chart[5]]), n, misfit

    def support_jacobian(self, xy_yaw, wheel_centres_base, radius, h=1e-5):
        """``G`` (3x3) with ``d[z, pitch, roll] = G d[x, y, yaw]``.

        Closed form in x and y on a single plane; central differences otherwise.
        """
        base = np.asarray(xy_yaw, float)
        C = np.asarray(wheel_centres_base, float)
        patch = self._single_plane(base[0], base[1], base[2], C, radius)
        G = np.zeros((3, 3))
        if patch is not None:
            nx, ny, nz = patch['n']
            G[0, 0], G[0, 1] = -nx / nz, -ny / nz
            # derivative of pitch and roll with respect to the heading (closed form)
            c, s = np.cos(base[2]), np.sin(base[2])
            d, dd = c * nx + s * ny, -s * nx + c * ny
            w = np.array([c - d * nx, s - d * ny, -d * nz])
            dw = np.array([-s - dd * nx, c - dd * ny, -dd * nz])
            norm = np.sqrt(w @ w)
            xb = w / norm
            dxb = (dw - xb * (xb @ dw)) / norm
            yb_z, dyb_z = nx * xb[1] - ny * xb[0], nx * dxb[1] - ny * dxb[0]
            G[1, 2] = -dxb[2] / np.sqrt(max(1. - xb[2] ** 2, 1e-300))
            G[2, 2] = dyb_z * nz / (yb_z ** 2 + nz ** 2)
            return G
        for k in range(3):
            d = np.zeros(3)
            d[k] = h
            # Where the wheel centres leave a single patch the support changes from that patch to the fitted plane,
            # and the two differ by a fraction of a millimetre near a convex crease.  A central difference across
            # that step is no derivative (tens of m/m): the difference is then taken on the side of ``base``.
            ahead = self._single_plane(base[0] + d[0], base[1] + d[1], base[2] + d[2], C, radius)
            behind = self._single_plane(base[0] - d[0], base[1] - d[1], base[2] - d[2], C, radius)
            if ahead is not None and behind is None:
                G[:, k] = (self.support(base, wheel_centres_base, radius)[0]
                           - self.support(base - d, wheel_centres_base, radius)[0]) / h
            elif behind is not None and ahead is None:
                G[:, k] = (self.support(base + d, wheel_centres_base, radius)[0]
                           - self.support(base, wheel_centres_base, radius)[0]) / h
            else:
                G[:, k] = (self.support(base + d, wheel_centres_base, radius)[0]
                           - self.support(base - d, wheel_centres_base, radius)[0]) / (2 * h)
        return G
