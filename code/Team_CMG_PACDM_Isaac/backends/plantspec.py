"""Simulator-neutral plant description generated from the compiled team graph.

The compiled graph is a control and reference model: world-rooted chart joints
for every floating root, wheels as single bodies with ideal rolling rows.  A
simulated plant needs the physical side of the same data: free bodies, joints
with their drive data, collision primitives, and wheels that actually roll.
This module derives that description once, so the MuJoCo and USD exporters
cannot drift apart.

Wheel model of the plants.  A conventional wheel is a sphere of the wheel radius
(``tyre``): it touches the ground at one point, as the thin wheel of the compiled
model does, and its friction is what moves and turns the base.  Every mecanum
wheel carries its declared number of free-spinning barrel rollers.  A roller is a body of revolution about its own
axis whose envelope, as the wheel turns, is the cylinder of the wheel radius
(exact profile, see :func:`roller_profile`).  Its collision shape is the convex
hull of a declared number of profile vertices.  The wheel mass and inertia of
the robot record are split between the hub and the rollers, so the plant and the
compiled model carry the same total inertia.

NumPy only.
"""
from __future__ import annotations

import numpy as np

from teamc.se3 import inv, rot, transform

# collision classes (bit masks shared by the exporters)
STATIC, PAYLOAD, ROLLER, OBSTACLE, TERRAIN = 1, 2, 4, 8, 16
ROBOT_BITS = (32, 64, 128, 256, 512, 1024)


def roller_profile(wheel_radius, max_radius, length, angle=np.pi / 4, rings=7):
    """Axial positions and radii of a mecanum roller whose envelope is the wheel cylinder.

    The roller axis passes at ``wheel_radius - max_radius`` from the wheel axis and
    makes ``angle`` with it.  For the axis point at parameter ``s`` the wheel-cylinder
    normal through it meets the cylinder at the contact point; its axial coordinate
    along the roller and its distance from the roller axis give the exact profile.
    """
    d0, sg2 = wheel_radius - max_radius, np.sin(angle) ** 2

    def point(s):
        rho = np.sqrt(d0 ** 2 + s ** 2 * sg2)
        return s * (1. + (wheel_radius - rho) * sg2 / rho), (wheel_radius - rho) * np.sqrt(max(1. - s ** 2 * sg2 ** 2 / rho ** 2, 0.))

    lo, hi = 0., length
    for _ in range(80):                                   # invert the monotone map s -> axial coordinate
        mid = .5 * (lo + hi)
        lo, hi = (mid, hi) if point(mid)[0] < length / 2 else (lo, mid)
    s_max = .5 * (lo + hi)
    out = np.array([point(s) for s in np.linspace(-s_max, s_max, rings)])
    if out[:, 1].min() <= 0.:
        raise ValueError('roller: the declared length is too large for its radius')
    return out[:, 0], out[:, 1]


def roller_vertices(wheel_radius, max_radius, length, rings=5, around=12, angle=np.pi / 4, stagger=True):
    """Hull vertices of a roller, local z along its axis, ring by ring (``rings * around`` points).

    With ``stagger`` alternate rings are turned by half a step (triangular faces); without it the
    hull between two neighbouring rings is a frustum with plane quadrilateral sides.
    """
    z, r = roller_profile(wheel_radius, max_radius, length, angle, rings)
    pts = []
    for i, (zi, ri) in enumerate(zip(z, r)):
        phase = (i % 2) * np.pi / around if stagger else 0.
        for k in range(around):
            a = phase + 2. * np.pi * k / around
            pts.append([ri * np.cos(a), ri * np.sin(a), zi])
    return np.array(pts)


def _frame_with_z(z, origin):
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    return transform(np.column_stack([x, np.cross(z, x), z]), origin)


def build_plantspec(comp, roller_hull=(5, 12)):
    """Plant description of a compiled team: robots, payload, scene, wheels with rollers, ports."""
    cmg = comp.cmg
    records = {b['id']: b for b in cmg['bodies']}
    joints = {j['id']: j for j in cmg['joints']}
    children = {}
    for j in cmg['joints']:
        if not j.get('generated'):
            children.setdefault(j['base_body'], []).append(j)
    ids = cmg['coordinate_ids']

    def body_entry(name, parent, joint):
        b = records[name]
        e = dict(name=name, parent=parent, mass=float(b['mass_kg']), com=np.asarray(b['com_m'], float),
                 inertia=np.asarray(b['inertia_kg_m2'], float), geoms=[], joint=None, T_parent_body=np.eye(4))
        if joint is not None:
            T_BJ, T_FJ = np.asarray(joint['T_BJ'], float), np.asarray(joint['T_FJ'], float)
            e['T_parent_body'] = T_BJ @ inv(T_FJ)
            e['T_parent_joint'], e['T_body_joint'] = T_BJ, T_FJ
            if joint['type'] != 'fixed':
                axis = np.asarray(joint['axis'], float)
                e['joint'] = dict(name=joint['id'], type=joint['type'], axis_joint=axis,
                                  axis_body=T_FJ[:3, :3] @ axis, anchor_body=T_FJ[:3, 3].copy(),
                                  lower=float(joint['limits']['lower']), upper=float(joint['limits']['upper']),
                                  armature=float(cmg['armature'].get(joint['id'], 0.)),
                                  damping=float(cmg['damping'].get(joint['id'], 0.)),
                                  index=ids.index(joint['id']))
            else:
                e['fixed'] = joint['id']
        return e

    def walk(root):
        out = [body_entry(root, None, None)]
        stack = [root]
        while stack:
            parent = stack.pop(0)
            for j in children.get(parent, []):
                out.append(body_entry(j['follower_body'], parent, j))
                stack.append(j['follower_body'])
        return out

    spec = dict(gravity=np.asarray(cmg['gravity_m_s2'], float), robots={}, ports=cmg['ports'],
                coordinate_ids=ids, roller=None)
    for k, (rid, robot) in enumerate(comp.robots.items()):
        rec = robot.record
        prefix = robot.prefix
        bodies = walk(prefix + rec['floating_root'])
        by_name = {b['name']: b for b in bodies}
        for g in rec.get('geometry', []):
            e = dict(g, name=f"{rid}/{g.get('name') or g['body']}", T=np.asarray(g['T_body_geom'], float))
            by_name[prefix + g['body']]['geoms'].append(e)
        wheels = []
        for w in robot.wheels:
            wb = by_name[prefix + w['body']]
            R_bw = wb['T_parent_body'][:3, :3]                  # wheel axes in the base frame, joints above the wheel at zero
            above = wb['parent']
            while by_name[above]['parent'] is not None:         # a wheel on a rocker or another body below the base
                R_bw = by_name[above]['T_parent_body'][:3, :3] @ R_bw
                above = by_name[above]['parent']
            axis_w = wb['joint']['axis_body'] / np.linalg.norm(wb['joint']['axis_body'])
            entry = dict(id=w['id'], body=wb['name'], joint=wb['joint']['name'], radius=w['radius_m'], width=w['width_m'],
                         friction=w.get('friction', .8), type=w['type'], axis_body=axis_w, rollers=[])
            if w['type'] != 'mecanum':
                ty = w.get('tyre', {})
                entry['tyre'] = dict(
                    name=f"{rid}/{w['id']}_tyre_contact", centre=np.zeros(3), radius=w['radius_m'],
                    slip_speed=float(ty.get('slip_speed_m_s', 1e-3)),
                    stiffness=float(ty.get('stiffness_N_m', 1.5e5)), damping=float(ty.get('damping_N_s_m', 1.5e3)),
                    relaxation_length=float(ty.get('relaxation_length_m', 0.)),
                    contact_time_constant=float(ty.get('contact_time_constant_s', .015)),
                    contact_damping_ratio=float(ty.get('contact_damping_ratio', 1.)),
                    friction_time_constant=float(ty.get('friction_time_constant_s', .5)),
                    contact_stiffness=float(ty.get('contact_stiffness_N_m', 2.0e5)),
                    contact_damping=float(ty.get('contact_damping_N_s_m', 4.0e3)))
            if w['type'] == 'mecanum':
                ro = w['rollers']
                g_w = R_bw.T @ np.asarray(w['traction_direction_base'], float)
                down_w = R_bw.T @ np.array([0., 0., -1.])
                cosang = abs(float(g_w @ axis_w))
                angle = float(np.arccos(np.clip(cosang, -1., 1.)))
                d0 = w['radius_m'] - ro['max_radius_m']
                for i in range(ro['count']):
                    Rk = rot(axis_w, 2. * np.pi * i / ro['count'])
                    entry['rollers'].append(dict(name=f"{wb['name']}_roller_{i}", joint=f"{wb['name']}_roller_{i}_joint",
                                                 T=_frame_with_z(Rk @ g_w, Rk @ (d0 * down_w))))
                if spec['roller'] is None:
                    r_eq = .75 * ro['max_radius_m']
                    spec['roller'] = dict(
                        vertices=roller_vertices(w['radius_m'], ro['max_radius_m'], ro['length_m'], roller_hull[0],
                                                 roller_hull[1], angle, stagger=(len(roller_hull) < 3 or bool(roller_hull[2]))),
                        profile=roller_profile(w['radius_m'], ro['max_radius_m'], ro['length_m'], angle, 41),
                        mass=ro['mass_kg'], angle=angle,
                        inertia=np.array([ro['mass_kg'] * (3 * r_eq ** 2 + ro['length_m'] ** 2) / 12] * 2
                                         + [ro['mass_kg'] * r_eq ** 2 / 2]),
                        damping=float(ro.get('damping', 2e-4)), armature=float(ro.get('armature', 2e-5)),
                        contact_time_constant=float(ro.get('contact_time_constant_s', .005)),
                        contact_damping_ratio=float(ro.get('contact_damping_ratio', 1.)),
                        contact_stiffness=float(ro.get('contact_stiffness_N_m', 1.4e6)),
                        contact_damping=float(ro.get('contact_damping_N_s_m', 1.4e4)),
                        # spheroid with the same crown radius and the same curvature along the axis at the crown
                        ellipsoid=[ro['max_radius_m'], ro['max_radius_m'],
                                   float(np.sqrt(ro['max_radius_m'] * (w['radius_m'] - ro['max_radius_m'])) / np.sin(angle))],
                        hull=tuple(roller_hull))
                # hub = declared wheel inertia minus its rollers (same total mass and inertia as the compiled model)
                Ir = np.diag(spec['roller']['inertia'])
                m_r = spec['roller']['mass']
                sub = np.zeros((3, 3))
                for r in entry['rollers']:
                    R, c = r['T'][:3, :3], r['T'][:3, 3]
                    sub += R @ Ir @ R.T + m_r * ((c @ c) * np.eye(3) - np.outer(c, c))
                hub_inertia = wb['inertia'] - sub
                hub_mass = wb['mass'] - len(entry['rollers']) * m_r
                moments = np.linalg.eigvalsh(hub_inertia)
                if hub_mass <= 0 or moments.min() <= 0:
                    raise ValueError(f"{wb['name']}: declared rollers are heavier than the wheel")
                if moments.max() > moments.sum() - moments.max() + 1e-12:
                    raise ValueError(f"{wb['name']}: the wheel inertia less its rollers is not the inertia of a body "
                                     f'(principal moments {moments}); the declared wheel is too flat for rollers of this length')
                entry.update(hub_mass=hub_mass, hub_inertia=hub_inertia)
            wheels.append(entry)
        sensor = rec['ports'].get('sensor')
        tool_body, T_tool = robot.frames[rec['ports']['tool']['frame']]
        s = dict(id=rid, prefix=prefix, root=bodies[0]['name'], bodies=bodies, wheels=wheels, base=robot.base.tolist(),
                 collision_bit=ROBOT_BITS[k],
                 tool=dict(body=tool_body, T=np.asarray(T_tool, float), latch_points=rec['ports']['tool'].get('latch_points')),
                 sensor=None)
        if sensor:
            sb, T_s = robot.frames[sensor['frame']]
            s['sensor'] = dict(body=sb, T=np.asarray(T_s, float), joint=prefix + sensor['joint_id'],
                               range=sensor.get('range', {}))
        spec['robots'][rid] = s

    payload = comp.payload
    prec = payload['record']
    pb = body_entry(payload['body'], None, None)
    for g in prec.get('geometry', []):
        pb['geoms'].append(dict(g, name=f"{payload['id']}/{g.get('name') or g['body']}", T=np.asarray(g['T_body_geom'], float)))
    spec['payload'] = dict(id=payload['id'], body=pb, base=payload['base'].tolist())
    spec['statics'] = [dict(s, T=np.asarray(s['T_world_geom'], float)) for s in comp.scene.get('statics', [])]
    spec['terrain_friction'] = float(comp.scene.get('terrain', {}).get('friction', .8))
    spec['obstacle'] = comp.scene.get('obstacle')
    spec['grasps'] = {cid: dict(body1=c['body1'], T1=np.asarray(c['T1'], float), body2=c['body2'],
                                T2=np.asarray(c['T2'], float), robot=c['robot'])
                      for cid, c in comp.connections.items() if c['kind'] == 'grasp'}
    spec['rests'] = {cid: dict(pads=[p['id'] for p in c['pads']]) for cid, c in comp.connections.items()
                     if c['kind'] == 'rest'}
    return spec


def robot_mass(spec, rid):
    return float(sum(b['mass'] for b in spec['robots'][rid]['bodies']))
