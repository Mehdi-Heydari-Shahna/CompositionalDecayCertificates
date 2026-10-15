"""Convex collision primitives of the team and the scene, and their separation distances.

Physical input: the ``geometry`` lists of the robot and payload records, the
``statics`` of the scene record and its ``obstacle``.  Shapes are boxes,
cylinders, capsules and spheres (declared primitives, not CAD).

Distances are computed by GJK on support functions (no simulator is imported), so
the same clearance audit runs on states recorded from any plant and inside the
route planner.  Capsules and spheres are a core segment/point plus a radius.

Scope: separation distance of disjoint convex shapes and its witness points.
Overlapping shapes report distance zero; penetration depth is not computed.
"""
from __future__ import annotations

from math import hypot, sqrt

import numpy as np

BOX, CYLINDER, CAPSULE, SPHERE = 'box', 'cylinder', 'capsule', 'sphere'


# ---------------------------------------------------------------------------- small vector helpers (floats)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _comb(points, lam):
    x = y = z = 0.
    for p, l in zip(points, lam):
        x += l * p[0]
        y += l * p[1]
        z += l * p[2]
    return (x, y, z)


# ---------------------------------------------------------------------------- closest point of a simplex to the origin
def _segment(a, b):
    ab = _sub(b, a)
    den = _dot(ab, ab)
    if den <= 0.:
        return (1., 0.)
    t = -_dot(a, ab) / den
    if t <= 0.:
        return (1., 0.)
    if t >= 1.:
        return (0., 1.)
    return (1. - t, t)


def _triangle(a, b, c):
    """Barycentric coordinates of the point of triangle abc closest to the origin (Ericson, RTCD 5.1.5)."""
    ab, ac = _sub(b, a), _sub(c, a)
    d1, d2 = -_dot(ab, a), -_dot(ac, a)
    if d1 <= 0. and d2 <= 0.:
        return (1., 0., 0.)
    d3, d4 = -_dot(ab, b), -_dot(ac, b)
    if d3 >= 0. and d4 <= d3:
        return (0., 1., 0.)
    vc = d1 * d4 - d3 * d2
    if vc <= 0. and d1 >= 0. and d3 <= 0.:
        v = d1 / (d1 - d3) if d1 != d3 else 0.
        return (1. - v, v, 0.)
    d5, d6 = -_dot(ab, c), -_dot(ac, c)
    if d6 >= 0. and d5 <= d6:
        return (0., 0., 1.)
    vb = d5 * d2 - d1 * d6
    if vb <= 0. and d2 >= 0. and d6 <= 0.:
        w = d2 / (d2 - d6) if d2 != d6 else 0.
        return (1. - w, 0., w)
    va = d3 * d6 - d5 * d4
    if va <= 0. and (d4 - d3) >= 0. and (d5 - d6) >= 0.:
        den = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / den if den != 0. else 0.
        return (0., 1. - w, w)
    den = va + vb + vc
    if den == 0.:                                   # degenerate triangle: fall back to its edges
        best, out = None, None
        for (i, j), (p, q) in (((0, 1), (a, b)), ((0, 2), (a, c)), ((1, 2), (b, c))):
            l = _segment(p, q)
            x = _comb((p, q), l)
            n = _dot(x, x)
            if best is None or n < best:
                lam = [0., 0., 0.]
                lam[i], lam[j] = l
                best, out = n, tuple(lam)
        return out
    v, w = vb / den, vc / den
    return (1. - v - w, v, w)


def _tetrahedron(a, b, c, d):
    """Closest point of tetrahedron abcd to the origin; returns (barycentric, inside)."""
    pts = (a, b, c, d)
    best, out = None, None
    inside = True
    for face in ((0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2), (1, 3, 2, 0)):
        p, q, r, o = (pts[i] for i in face)
        n = _cross(_sub(q, p), _sub(r, p))
        sp, so = -_dot(p, n), _dot(_sub(o, p), n)
        if sp * so < 0. or so == 0.:                # origin on the outer side of this face (or flat simplex)
            inside = False
            l = _triangle(p, q, r)
            x = _comb((p, q, r), l)
            nn = _dot(x, x)
            if best is None or nn < best:
                lam = [0., 0., 0., 0.]
                lam[face[0]], lam[face[1]], lam[face[2]] = l
                best, out = nn, tuple(lam)
    if inside:
        return (.25, .25, .25, .25), True
    return out, False


def gjk(support_a, support_b, direction=(1., 0., 0.), rel_tol=1e-9, abs_tol=1e-12, max_iter=64):
    """Distance between two convex sets given by world support functions.

    Returns ``(distance, point_a, point_b, iterations)``; distance zero means the
    sets touch or overlap (the points are then not meaningful).
    """
    d = direction
    a = support_a((-d[0], -d[1], -d[2]))
    b = support_b(d)
    W, A, B = [_sub(a, b)], [a], [b]
    lam = (1.,)
    v = W[0]
    for it in range(max_iter):
        vv = _dot(v, v)
        if vv <= abs_tol:
            return 0., A[0], B[0], it
        a = support_a((-v[0], -v[1], -v[2]))
        b = support_b(v)
        w = _sub(a, b)
        if vv - _dot(v, w) <= rel_tol * vv + abs_tol or any(_dot(_sub(w, x), _sub(w, x)) <= 1e-24 for x in W):
            break
        W.append(w)
        A.append(a)
        B.append(b)
        k = len(W)
        if k == 2:
            lam = _segment(W[0], W[1])
        elif k == 3:
            lam = _triangle(W[0], W[1], W[2])
        else:
            lam, inside = _tetrahedron(W[0], W[1], W[2], W[3])
            if inside:
                return 0., A[0], B[0], it
        keep = [i for i in range(k) if lam[i] > 0.]
        W, A, B = [W[i] for i in keep], [A[i] for i in keep], [B[i] for i in keep]
        lam = tuple(lam[i] for i in keep)
        v_new = _comb(W, lam)
        if _dot(v_new, v_new) >= vv:                # no progress (round-off): keep the previous estimate
            break
        v = v_new
    return sqrt(_dot(v, v)), _comb(A, lam), _comb(B, lam), it


# ---------------------------------------------------------------------------- shapes
class Shape:
    """One convex primitive attached to a tree body (``body`` index) or to the world (``body`` -1)."""

    __slots__ = ('name', 'kind', 'body', 'group', 'role', 'T', 'half', 'radius', 'half_length', 'margin', 'tags')

    def __init__(self, name, kind, body, group, role, T, half=None, radius=0., half_length=0., tags=()):
        self.name, self.kind, self.body, self.group, self.role = name, kind, int(body), group, role
        self.T = np.asarray(T, float)
        self.half = None if half is None else tuple(float(x) for x in half)
        self.radius, self.half_length = float(radius), float(half_length)
        self.margin = self.radius if kind in (CAPSULE, SPHERE) else 0.
        self.tags = tuple(tags)


class Placed:
    """A shape at a world pose: support function, axis-aligned bounds."""

    __slots__ = ('shape', 'R', 'p', 'lo', 'hi', 'support', 'margin')

    def __init__(self, shape, T):
        self.shape = shape
        R, p = T[:3, :3], T[:3, 3]
        self.R, self.p = R, p
        c0, c1, c2 = (tuple(float(x) for x in R[:, i]) for i in range(3))
        px, py, pz = (float(x) for x in p)
        kind = shape.kind
        self.margin = shape.margin
        if kind == BOX:
            hx, hy, hz = shape.half
            ext = np.abs(R) @ np.array(shape.half)

            def support(d):
                sx = hx if _dot(c0, d) >= 0. else -hx
                sy = hy if _dot(c1, d) >= 0. else -hy
                sz = hz if _dot(c2, d) >= 0. else -hz
                return (px + sx * c0[0] + sy * c1[0] + sz * c2[0], py + sx * c0[1] + sy * c1[1] + sz * c2[1],
                        pz + sx * c0[2] + sy * c1[2] + sz * c2[2])
        elif kind == CYLINDER:
            r, h = shape.radius, shape.half_length
            ext = r * np.sqrt(np.clip(1. - R[:, 2] ** 2, 0., 1.)) + h * np.abs(R[:, 2])

            def support(d):
                lx, ly, lz = _dot(c0, d), _dot(c1, d), _dot(c2, d)
                rho = hypot(lx, ly)
                sx, sy = (r * lx / rho, r * ly / rho) if rho > 1e-300 else (0., 0.)
                sz = h if lz >= 0. else -h
                return (px + sx * c0[0] + sy * c1[0] + sz * c2[0], py + sx * c0[1] + sy * c1[1] + sz * c2[1],
                        pz + sx * c0[2] + sy * c1[2] + sz * c2[2])
        elif kind == CAPSULE:
            h = shape.half_length
            ext = h * np.abs(R[:, 2]) + shape.radius

            def support(d):
                sz = h if _dot(c2, d) >= 0. else -h
                return (px + sz * c2[0], py + sz * c2[1], pz + sz * c2[2])
        elif kind == SPHERE:
            ext = np.full(3, shape.radius)

            def support(d):
                return (px, py, pz)
        else:
            raise ValueError(f'{shape.name}: unsupported shape {kind}')
        self.support = support
        self.lo, self.hi = p - ext, p + ext


def distance(a, b):
    """Separation of two placed shapes and its witness points (on the surfaces)."""
    d0 = (float(b.p[0] - a.p[0]), float(b.p[1] - a.p[1]), float(b.p[2] - a.p[2]))
    if d0 == (0., 0., 0.):
        d0 = (1., 0., 0.)
    # GJK works on A - B: start from the direction pointing from B to A
    dist, pa, pb, _ = gjk(a.support, b.support, (-d0[0], -d0[1], -d0[2]))
    margin = a.margin + b.margin
    if margin and dist > 0.:
        n = ((pb[0] - pa[0]) / dist, (pb[1] - pa[1]) / dist, (pb[2] - pa[2]) / dist)
        pa = (pa[0] + a.margin * n[0], pa[1] + a.margin * n[1], pa[2] + a.margin * n[2])
        pb = (pb[0] - b.margin * n[0], pb[1] - b.margin * n[1], pb[2] - b.margin * n[2])
    return max(0., dist - margin), np.array(pa), np.array(pb)


def _record_shape(g, body, group, role, name, T):
    kind = g['type']
    if kind == BOX:
        return Shape(name, BOX, body, group, role, T, half=g['half_size'])
    if kind in (CYLINDER, CAPSULE):
        return Shape(name, kind, body, group, role, T, radius=g['radius'], half_length=g['half_length'])
    if kind == SPHERE:
        return Shape(name, SPHERE, body, group, role, T, radius=g['radius'])
    raise ValueError(f'{name}: unsupported geometry type {kind}')


class Geometry:
    """Collision primitives of a compiled team (moving) and its scene (static)."""

    def __init__(self, comp, wheels=True):
        tree = comp.tree
        self.comp = comp
        self.moving, self.statics = [], []
        for rid, robot in comp.robots.items():
            for g in robot.record.get('geometry', []):
                if not g.get('collide', True):
                    continue
                tags = ('tool',) if g['body'] == robot.frames[robot.record['ports']['tool']['frame']][0][len(robot.prefix):] else ()
                self.moving.append(_record_shape(g, tree.index[robot.prefix + g['body']], rid, 'robot',
                                                 f"{rid}/{g.get('name') or g['body']}", g['T_body_geom']))
                self.moving[-1].tags = tags
            if wheels:
                for w in robot.wheels:
                    axis = np.asarray(tree.axis[tree.body_of_joint[robot.prefix + w['joint_id']]], float)
                    z = axis / np.linalg.norm(axis)
                    x = np.cross([1., 0, 0], z) if abs(z[0]) < .9 else np.cross([0, 1., 0], z)
                    x /= np.linalg.norm(x)
                    T = np.eye(4)
                    T[:3, :3] = np.column_stack([x, np.cross(z, x), z])
                    self.moving.append(Shape(f"{rid}/{w['id']}_wheel", CYLINDER, tree.index[robot.prefix + w['body']],
                                             rid, 'robot', T, radius=w['radius_m'], half_length=w['width_m'] / 2,
                                             tags=('wheel',)))
        pid = comp.payload['id']
        for g in comp.payload['record'].get('geometry', []):
            if g.get('collide', True):
                self.moving.append(_record_shape(g, tree.index[comp.payload['prefix'] + g['body']], pid, 'payload',
                                                 f"{pid}/{g.get('name') or g['body']}", g['T_body_geom']))
        for s in comp.scene.get('statics', []):
            if s.get('collide', True):
                self.statics.append(Placed(_record_shape(s, -1, 'scene', s.get('role', 'static'), s['id'], np.eye(4)),
                                           np.asarray(s['T_world_geom'], float)))
        self.static_lo = np.array([s.lo for s in self.statics]).reshape(-1, 3)
        self.static_hi = np.array([s.hi for s in self.statics]).reshape(-1, 3)
        self.payload_id = pid
        ob = comp.scene.get('obstacle')
        self.obstacle = None if not ob else Shape(ob['id'], BOX, -1, 'obstacle', 'obstacle', np.eye(4), half=ob['half_size'])

    # ------------------------------------------------------------------ placement
    def place(self, T_bodies, planar=None, groups=None):
        """World placement of the moving shapes for tree body poses ``T_bodies`` (nb,4,4).

        ``planar`` (4x4) moves the whole team rigidly (used by the route planner).
        """
        out = []
        for s in self.moving:
            if groups is not None and s.group not in groups:
                continue
            T = T_bodies[s.body] @ s.T
            out.append(Placed(s, T if planar is None else planar @ T))
        return out

    def place_obstacle(self, position, yaw=0.):
        if self.obstacle is None:
            return None
        T = np.eye(4)
        c, s = np.cos(yaw), np.sin(yaw)
        T[:3, :3] = [[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]
        T[:3, 3] = position
        return Placed(self.obstacle, T)

    # ------------------------------------------------------------------ queries
    def against_statics(self, placed, cutoff=.5, skip_roles=('terrain',), skip=None):
        """Smallest separation of each placed shape from the statics, by static role.

        Pairs whose bounding boxes are further apart than ``cutoff`` are not refined;
        ``cutoff`` is then reported as a lower bound.  Returns a list of
        ``(distance, moving name, static name, role, point_moving, point_static)``.
        """
        out = []
        if not self.statics:
            return out
        for a in placed:
            gap = np.maximum(np.maximum(self.static_lo - a.hi, a.lo - self.static_hi), 0.)
            near = np.flatnonzero((gap * gap).sum(axis=1) < cutoff * cutoff)
            for k in near:
                b = self.statics[k]
                if b.shape.role in skip_roles or (skip is not None and skip(a.shape, b.shape)):
                    continue
                d, pa, pb = distance(a, b)
                if d < cutoff:
                    out.append((d, a.shape.name, b.shape.name, b.shape.role, pa, pb))
        return out

    def between(self, placed_a, placed_b, cutoff=.5):
        out = []
        for a in placed_a:
            for b in placed_b:
                gap = np.maximum(np.maximum(b.lo - a.hi, a.lo - b.hi), 0.)
                if (gap * gap).sum() < cutoff * cutoff:
                    d, pa, pb = distance(a, b)
                    if d < cutoff:
                        out.append((d, a.shape.name, b.shape.name, b.shape.role, pa, pb))
        return out

    def clearance(self, T_bodies, cutoff=.5, obstacle=None, planar=None):
        """Clearance summary of one team configuration.

        Categories: payload against scene roles, robots against scene roles, robots
        against each other, robots against the payload (the gripping tool excluded),
        and everything against the moving obstacle.  Values are capped at ``cutoff``.
        """
        placed = self.place(T_bodies, planar)
        out = {}

        def note(key, item):
            if key not in out or item[0] < out[key][0]:
                out[key] = item

        wheel = lambda s, t: 'wheel' in s.tags and t.role in ('terrain',)
        for item in self.against_statics(placed, cutoff, skip=wheel):
            owner = 'payload' if item[1].startswith(self.payload_id + '/') else 'robot'
            note(f'{owner}:{item[3]}', item)
        groups = {}
        for p in placed:
            groups.setdefault(p.shape.group, []).append(p)
        ids = [g for g in groups if g != self.payload_id]
        for i, gi in enumerate(ids):
            for gj in ids[i + 1:]:
                for item in self.between(groups[gi], groups[gj], cutoff):
                    note('robot:robot', item)
            body = [p for p in groups[gi] if 'tool' not in p.shape.tags]
            for item in self.between(body, groups.get(self.payload_id, []), cutoff):
                note('robot:payload', item)
        if obstacle is not None:
            for item in self.between(placed, [obstacle], cutoff):
                note('team:obstacle', item)
        return out

    def top(self, T_bodies, groups=None):
        """Highest point of the moving shapes (m)."""
        return max(float(p.hi[2]) for p in self.place(T_bodies, groups=groups))
