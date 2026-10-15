"""Ground loads, traction and tipping margins from the recovered reactions.

``fit`` ground model (a rigid base on four wheels): rigid-body mechanics
determines the resultant ground wrench of the base, not the normal load of each
wheel.  The per-wheel normal loads therefore use a declared equal-stiffness
assumption (normal load affine in the wheel position), which is the distribution
of identical linear tyre springs under a rigid chassis.

``contact`` ground model (a base whose passive joints make the wheel contacts
determinate, such as a rocker): the reactions of the ground rows are the wheel
normal loads themselves.  The tipping margin is then taken to the polygon the
loads can actually reach: wheels on a passive joint act on the base through
that joint, so a pair of wheels on a rocker supports the base at one point.

Nothing here feeds back into the constraints: the outputs are margins.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def support_points(model, rid, T):
    """Points at which the ground supports a ``contact`` base, in the base frame (x, y).

    A wheel on a body fixed to the base supports it at its own ground point.  Wheels behind a
    passive revolute joint support it where the line through their ground points passes the joint
    axis: a normal force there has no moment about that axis.
    """
    robot = model.comp.robots[rid]
    Tb = T[model.base_body[rid]]
    Rb, pb = Tb[:3, :3], Tb[:3, 3]
    groups = {}
    for k, w in enumerate(robot.wheels):
        groups.setdefault(tuple(w['suspension']), []).append(k)
    points, wheels = [], []
    centres = np.array([w['centre_base'] for w in robot.wheels], float)
    for joints, members in groups.items():
        if not joints:
            for k in members:
                points.append(centres[k, :2])
                wheels.append([k])
            continue
        if len(joints) != 1 or len(members) != 2:
            raise ValueError(f'{rid}: support points are defined for wheel pairs behind one passive joint')
        ij = model.tree.body_of_joint[robot.prefix + joints[0]]
        if model.tree.kind[ij] != 1:
            raise ValueError(f'{rid}: support points are defined for a passive revolute joint')
        # joint axis and a point on it, in the base frame, at the present configuration
        Tp = T[model.tree.parent[ij]] @ model.tree.enter[ij]
        axis = Rb.T @ (Tp[:3, :3] @ model.tree.axis[ij])
        origin = Rb.T @ (Tp[:3, 3] - pb)
        a, b = centres[members[0]], centres[members[1]]
        n = np.array([0., 0., 1.])
        # moment about the axis of a unit normal force at a + s (b - a):  axis . ((p - origin) x n) = 0
        f = lambda p: float(axis @ np.cross(p - origin, n))
        fa, fb = f(a), f(b)
        if abs(fb - fa) < 1e-12:
            raise ValueError(f'{rid}: the wheels behind {joints[0]} do not straddle its axis')
        s = -fa / (fb - fa)
        points.append((a + s * (b - a))[:2])
        wheels.append(list(members))
    return np.array(points), wheels


def _contact_loads(model, q, rid, lam_ground, lam_wheels, T, S, contacts):
    robot = model.comp.robots[rid]
    N = np.asarray(lam_ground, float).copy()
    centres = np.array([w['centre_base'] for w in robot.wheels], float)
    total = float(N.sum())
    cop = (N[:, None] * centres[:, :2]).sum(axis=0) / total if abs(total) > 1e-9 else np.zeros(2)
    points, _ = support_points(model, rid, T)
    hull = ConvexHull(points)
    margin = float(np.min(-(hull.equations[:, :2] @ cop + hull.equations[:, 2])))
    mu = np.array([w.get('friction', .8) for w in robot.wheels])
    traction = np.zeros(len(centres))
    labels = model.wheel_rows(q, rid, (T, S), contacts)[1]
    lam_wheels = np.asarray(lam_wheels, float)
    for i, w in enumerate(robot.wheels):
        rows = [k for k, label in enumerate(labels) if label.startswith(w['id'] + ':')]
        traction[i] = float(np.linalg.norm(lam_wheels[rows])) if rows else 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        use = np.where(N > 1e-9, traction / (mu * N), np.inf)
    lifted = robot.wheels[int(N.argmin())]['id'] if N.min() <= 0. else None
    return dict(normal_loads_N=N, total_normal_N=total, cop_base_m=cop, tipping_margin_m=margin,
                min_normal_load_N=float(N.min()), traction_N=traction, traction_utilization=use,
                max_traction_utilization=float(np.max(np.where(np.isfinite(use), use, 0.))) if lifted else float(use.max()),
                unloaded_wheel=lifted, support_polygon_base_m=points,
                assumption='determinate: one contact row per wheel, passive joints free')


def _fit_loads(model, rid, T, S, lam_ground, normal):
    """Wheel normal loads of a ``fit`` base from the three multipliers of its ground rows (equal-stiffness rule)."""
    robot = model.comp.robots[rid]
    ib = model.base_body[rid]
    Tb = T[ib]
    Xb = model._twist_map(ib, S)[:, robot.base[[2, 4, 5]]]
    centres = np.array([w['centre_base'] for w in robot.wheels], float)
    A = np.zeros((3, len(centres)))
    for i, w in enumerate(robot.wheels):
        c = Tb[:3, 3] + Tb[:3, :3] @ centres[i] - w['radius_m'] * normal
        A[:, i] = np.r_[np.cross(c, normal), normal] @ Xb
    Phi = np.c_[np.ones(len(centres)), centres[:, :2]]
    theta, *_ = np.linalg.lstsq(A @ Phi, np.asarray(lam_ground, float), rcond=None)
    N = Phi @ theta
    lifted = None
    if N.min() < 0. and len(centres) == 4:
        # a wheel cannot pull on the ground: with one wheel unloaded the other three are determinate
        best = None
        for j in range(4):
            keep = [i for i in range(4) if i != j]
            try:
                n3 = np.linalg.solve(A[:, keep], np.asarray(lam_ground, float))
            except np.linalg.LinAlgError:
                continue
            if best is None or n3.min() > best[1].min():
                best = (j, n3, keep)
        if best is not None and best[1].min() >= 0.:
            N = np.zeros(4)
            N[best[2]] = best[1]
            lifted = robot.wheels[best[0]]['id']
    return N, lifted, centres


def wheel_load_map(model, q, rid, kin=None, normal=None):
    """``P`` (wheels x 3) with ``loads = P @ lam_ground`` for a ``fit`` base while every wheel carries load.

    The linear part of :func:`wheel_normal_loads` (equal-stiffness rule): what the friction of a
    friction-driven base needs to find the loads that are consistent with it.
    """
    robot = model.comp.robots[rid]
    if robot.ground != 'fit':
        raise ValueError(f'{rid}: the wheel loads of this base are its ground reactions themselves')
    T, S = model.tree.sweep(q) if kin is None else kin
    ib = model.base_body[rid]
    Tb = T[ib]
    if normal is None:
        normal = Tb[:3, 2]
    Xb = model._twist_map(ib, S)[:, robot.base[[2, 4, 5]]]
    centres = np.array([w['centre_base'] for w in robot.wheels], float)
    A = np.zeros((3, len(centres)))
    for i, w in enumerate(robot.wheels):
        c = Tb[:3, 3] + Tb[:3, :3] @ centres[i] - w['radius_m'] * normal
        A[:, i] = np.r_[np.cross(c, normal), normal] @ Xb
    Phi = np.c_[np.ones(len(centres)), centres[:, :2]]
    return Phi @ np.linalg.inv(A @ Phi)


def wheel_normal_loads(model, q, rid, lam_ground, kin=None, normal=None):
    """Normal load of every wheel of a base from the multipliers of its ground rows."""
    robot = model.comp.robots[rid]
    if robot.ground == 'contact':
        return np.asarray(lam_ground, float).copy()
    T, S = model.tree.sweep(q) if kin is None else kin
    if normal is None:
        normal = T[model.base_body[rid]][:3, 2]
    return _fit_loads(model, rid, T, S, lam_ground, normal)[0]


def ground_loads(model, q, rid, lam_ground, lam_wheels, kin=None, normal=None):
    """Wheel normal loads, centre of pressure, tipping and traction margins of one base.

    ``normal`` is the ground geometry of the constraint rows (``Constraints.info['<robot>:normal']``).

    For a friction-driven base (conventional wheels, skid steering) the tangential force of a wheel is a
    friction force: it is at its limit whenever the wheel slides, which is how such a base turns.  The
    traction that says whether the base keeps its grip is therefore the resultant the wheels pass to the
    ground, reported as ``net_traction_utilization``: that resultant over the friction capacity of the base.
    """
    robot = model.comp.robots[rid]
    T, S = model.tree.sweep(q) if kin is None else kin
    if robot.ground == 'contact':
        contacts = model.wheel_contacts(q, rid, (T, S)) if normal is None else normal
        return _contact_loads(model, q, rid, lam_ground, lam_wheels, T, S, contacts)
    ib = model.base_body[rid]
    Tb = T[ib]
    if normal is None:
        normal = Tb[:3, 2]
    N, lifted, centres = _fit_loads(model, rid, T, S, lam_ground, normal)
    total = float(N.sum())
    cop = (N[:, None] * centres[:, :2]).sum(axis=0) / total if abs(total) > 1e-9 else np.zeros(2)
    hull = ConvexHull(centres[:, :2])
    margin = float(np.min(-(hull.equations[:, :2] @ cop + hull.equations[:, 2])))
    mu = np.array([w.get('friction', .8) for w in robot.wheels])
    traction = np.zeros(len(centres))
    labels = model.wheel_rows(q, rid, (T, S), normal)[1]
    lam_wheels = np.asarray(lam_wheels, float)
    for i, w in enumerate(robot.wheels):
        rows = [k for k, label in enumerate(labels) if label.startswith(w['id'] + ':')]
        traction[i] = float(np.linalg.norm(lam_wheels[rows])) if rows else 0.
    with np.errstate(divide='ignore', invalid='ignore'):
        use = np.where(N > 1e-9, traction / (mu * N), np.inf)
    out = dict(normal_loads_N=N, total_normal_N=total, cop_base_m=cop, tipping_margin_m=margin,
               min_normal_load_N=float(N.min()), traction_N=traction, traction_utilization=use,
               max_traction_utilization=float(np.max(np.where(np.isfinite(use), use, 0.))) if lifted else float(use.max()),
               unloaded_wheel=lifted, assumption='equal-stiffness wheel compliance')
    if robot.drive == 'friction':
        # resultant of the wheel forces in the ground plane: the rolling directions of the wheels are parallel,
        # and so are their axles, so the multipliers of each kind add up
        along = sum(lam_wheels[k] for k, label in enumerate(labels) if label.endswith(':rolling'))
        across = sum(lam_wheels[k] for k, label in enumerate(labels) if label.endswith(':lateral'))
        capacity = float(np.sum(mu * np.maximum(N, 0.)))
        out['net_traction_N'] = float(np.hypot(along, across))
        out['net_traction_utilization'] = float(np.hypot(along, across) / capacity) if capacity > 1e-9 else np.inf
    return out


def pad_loads(connection, lam_rest):
    """Pad normal loads and friction demand of a rest connection from its total support wrench.

    ``lam_rest`` is the wrench the supports apply to the payload, in the rest frame
    (``[moment; force]``, moment about the rest-frame origin).  Pads are point
    supports with unilateral normals along their local z axes; the moment about the
    line through collinear pads is carried by the pad width and reported.
    """
    T0 = np.asarray(connection['T1'], float)
    pads = connection['pads']
    lam = np.asarray(lam_rest, float)
    # equilibrium rows: resultant force (3) and moment about the rest-frame origin (3)
    A = np.zeros((6, len(pads)))
    for k, pad in enumerate(pads):
        Tp = np.linalg.inv(T0) @ np.asarray(pad['T'], float)
        n = Tp[:3, 2]
        A[:, k] = np.r_[np.cross(Tp[:3, 3], n), n]
    N, *_ = np.linalg.lstsq(A, lam, rcond=None)
    remainder = lam - A @ N
    n_mean = A[3:].mean(axis=1)
    n_mean /= np.linalg.norm(n_mean)
    tangential = remainder[3:] - (remainder[3:] @ n_mean) * n_mean
    capacity = float(sum(max(n, 0.) * pad['friction'] for n, pad in zip(N, pads)))
    width_capacity = float(sum(max(n, 0.) * pad['half_width_m'] for n, pad in zip(N, pads)))
    return dict(pad_normal_N={pad['id']: float(n) for n, pad in zip(N, pads)}, min_pad_normal_N=float(N.min()),
                total_normal_N=float(N.sum()), friction_demand_N=float(np.linalg.norm(tangential)),
                friction_capacity_N=capacity,
                friction_utilization=float(np.linalg.norm(tangential) / capacity) if capacity > 1e-9 else np.inf,
                residual_moment_Nm=float(np.linalg.norm(remainder[:3])), width_moment_capacity_Nm=width_capacity)
