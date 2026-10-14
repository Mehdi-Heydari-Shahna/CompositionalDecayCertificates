"""MJCF of the simulated plant, generated from the plant description.

Robots and the payload are free bodies; every actuator port becomes a torque or
force motor; mecanum wheels carry free-spinning rollers with convex-hull
collision; each grasp connection is a weld equality that is inactive until the
clamp latches; a force and a torque sensor sit at each robot's sensor joint; the
crossing cart is a mocap body.  Nothing here is a controller.

Conventional wheels (skid steering) are spheres that meet the terrain through
explicit contact pairs.  By default (``tyre='model'``) the pairs are frictionless
and carry the wheel loads only; the friction of the tyres is the tyre model of
``backends/tyre.py``, which the plant applies as loads, as the Isaac Sim plant
does.

With ``tyre='native'`` the friction is that of the MuJoCo contacts.  A wheel of
such a base slides whenever the base turns, and MuJoCo's soft friction rows
carry a body that slides steadily only up to a slip speed that grows with their
time constant: above it the contact lifts off and lands again every few steps,
and the mean friction is no longer Coulomb's.  The pairs therefore give the
friction rows their own, longer time constant (``friction_time_constant_s`` of
the wheel record).  Three things follow and are measured in
``tests/test_skid_plant.py``.  Below a slip of a few centimetres per second the
friction force grows with the slip instead of being at its limit.  The friction
of a sliding wheel opposes ``b v + a`` (slip velocity and acceleration of the
ground point, ``b = 2 / (0.95 time constant)``), so a base that turns while it
drives drifts outwards at about ``v w / b``: its turning point lies ``v / b``
further ahead than the skid model says.  And the longer time constant only moves
the limit: a turn on the spot follows the skid model up to 0.2 rad/s (slips of
0.1 m/s); at 0.25 rad/s the contacts of the sliding wheels open and close and
the base turned at 0.07 rad/s.  This mode is kept as a comparison with a
friction the simulator solves itself; the missions are not run with it.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from teamc.se3 import chart_pose, quat_wxyz

from .plantspec import OBSTACLE, PAYLOAD, ROLLER, STATIC, TERRAIN, build_plantspec


# clamp contact points on the payload, in the tool frame (jaw faces and the opposite jaw); a robot record
# may declare its own under ports.tool.latch_points
LATCH_POINTS = ((.07, .049, 0.), (-.07, .049, 0.), (0., -.049, 0.))


def _f(x):
    return ' '.join(f'{float(v):.10g}' for v in np.asarray(x, float).ravel())


def _pose(T):
    T = np.asarray(T, float)
    return dict(pos=_f(T[:3, 3]), quat=_f(quat_wxyz(T[:3, :3])))


def _inertial(parent, mass, com, inertia):
    I = np.asarray(inertia, float)
    ET.SubElement(parent, 'inertial', pos=_f(com), mass=f'{mass:.10g}',
                  fullinertia=_f([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]))


def _geom(parent, g, contype, conaffinity, friction=None, **extra):
    kind = g['type']
    attr = dict(name=g['name'], type=kind, rgba=_f(g.get('rgba', [.6, .6, .6, 1.])), **_pose(g['T']))
    if kind == 'box':
        attr['size'] = _f(g['half_size'])
    elif kind in ('cylinder', 'capsule'):
        attr['size'] = _f([g['radius'], g['half_length']])
    elif kind == 'sphere':
        attr['size'] = _f([g['radius']])
    else:
        raise ValueError(f"{g['name']}: unsupported geometry {kind}")
    collide = g.get('collide', True)
    attr.update(contype=str(contype if collide else 0), conaffinity=str(conaffinity if collide else 0),
                group='0' if collide else '1')
    mu = g.get('friction', friction)
    if mu is not None:
        attr['friction'] = _f([mu, .005, .0001])
    attr.update(extra)
    return ET.SubElement(parent, 'geom', attr)


def build_mjcf(comp, timestep=.002, roller_hull=(5, 12), contact_solref=(.005, 1.), weld_solref=(.004, 1.),
               impratio=10., q0=None, spec=None, offscreen=(1920, 1440), roller_shape='hull', floor='plane',
               joint_damping=True, integrator='implicitfast', tyre='model'):
    """Return ``(xml, meta)`` for a compiled team.  ``meta`` names every element the plant adapter needs.

    ``joint_damping=False`` leaves the declared joint damping out of the model, for an adapter that
    applies it as an explicit effort (the Isaac Sim adapter and its offline stand-in).
    ``tyre``: ``'model'`` (the contacts of conventional wheels are frictionless and the plant applies the
    tyre model of ``backends/tyre.py``) or ``'native'`` (their friction is MuJoCo's).
    """
    spec = build_plantspec(comp, roller_hull) if spec is None else spec
    q0 = comp.seed() if q0 is None else np.asarray(q0, float)
    if tyre not in ('native', 'model'):
        raise ValueError('tyre must be native|model')
    root = ET.Element('mujoco', model=comp.cmg.get('name', 'team') or 'team')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', boundmass='0', boundinertia='0')
    ET.SubElement(root, 'option', timestep=f'{timestep:.10g}', gravity=_f(spec['gravity']), integrator=integrator,
                  cone='elliptic', impratio=f'{impratio:.6g}', solver='Newton', iterations='60', tolerance='1e-10')
    ET.SubElement(root, 'size', memory='200M')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(offscreen[0]), offheight=str(offscreen[1]), azimuth='120', elevation='-25')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.55 .55 .55', specular='.1 .1 .1')
    ET.SubElement(visual, 'quality', shadowsize='4096')
    ET.SubElement(visual, 'map', znear='0.02', zfar='80')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', condim='3', solref=_f(contact_solref), solimp='0.9 0.95 0.001', margin='0')
    ET.SubElement(default, 'joint', limited='false')
    ET.SubElement(default, 'motor', ctrllimited='true')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='grid', type='2d', builtin='checker', rgb1='.78 .79 .80', rgb2='.72 .73 .74',
                  width='512', height='512')
    ET.SubElement(asset, 'material', name='grid', texture='grid', texrepeat='20 20', reflectance='0')
    ET.SubElement(asset, 'texture', name='sky', type='skybox', builtin='gradient', rgb1='.85 .9 .95', rgb2='.55 .65 .8',
                  width='256', height='256')
    if spec['roller'] is not None:
        ET.SubElement(asset, 'mesh', name='roller', vertex=_f(spec['roller']['vertices']))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', name='sun', pos='4 8 12', dir='-.2 -.3 -1', directional='true', diffuse='.5 .5 .5',
                  castshadow='true')
    robot_bits = sum(r['collision_bit'] for r in spec['robots'].values())
    meta = dict(robots={}, geoms={}, statics=[], coordinate_joint={}, ports=[p['id'] for p in spec['ports']],
                welds={}, timestep=timestep, roller_hull=tuple(roller_hull))
    terrain_geoms, tyres = ['floor'], []

    def tag(name, group, role, **kw):
        meta['geoms'][name] = dict(group=group, role=role, **kw)

    # ------------------------------------------------------------------ terrain and statics
    mu_t = spec['terrain_friction']
    if floor == 'plane':
        ET.SubElement(world, 'geom', name='floor', type='plane', size='40 40 .1', material='grid', contype=str(TERRAIN | STATIC),
                      conaffinity=str(ROLLER | robot_bits | PAYLOAD), friction=_f([mu_t, .005, .0001]))
    else:   # a slab: roller hulls then meet every terrain surface through the same convex-convex routine (one contact point)
        ET.SubElement(world, 'geom', name='floor', type='box', size='40 40 .5', pos='0 0 -.5', material='grid',
                      contype=str(TERRAIN | STATIC), conaffinity=str(ROLLER | robot_bits | PAYLOAD),
                      friction=_f([mu_t, .005, .0001]))
    tag('floor', 'scene', 'terrain')
    for s in spec['statics']:
        terrain = s.get('role') == 'terrain'
        g = dict(s, name=s['id'])
        _geom(world, g, (TERRAIN | STATIC) if terrain else STATIC,
              (ROLLER | robot_bits | PAYLOAD) if terrain else (robot_bits | PAYLOAD))
        tag(s['id'], 'scene', s.get('role', 'static'))
        meta['statics'].append(s['id'])
        if terrain:
            terrain_geoms.append(s['id'])
    ob = spec['obstacle']
    if ob:
        cart = ET.SubElement(world, 'body', name=ob['id'], mocap='true', pos=_f(ob['start']))
        ET.SubElement(cart, 'geom', name=ob['id'], type='box', size=_f(ob['half_size']), rgba=_f(ob.get('rgba', [.2, .6, .3, 1.])),
                      contype=str(OBSTACLE), conaffinity=str(robot_bits | PAYLOAD))
        tag(ob['id'], 'obstacle', 'obstacle')
        meta['obstacle'] = ob['id']

    # ------------------------------------------------------------------ robots
    ro = spec['roller']
    for rid, r in spec['robots'].items():
        bit = r['collision_bit']
        affinity = STATIC | PAYLOAD | OBSTACLE | (robot_bits & ~bit)
        wheels = {w['body']: w for w in r['wheels']}
        elements = {}
        rmeta = dict(free=f'{rid}_free', root=r['root'], rollers=[], wheels={}, base=r['base'])
        for b in r['bodies']:
            if b['parent'] is None:
                T0 = chart_pose(q0[r['base']])
                e = ET.SubElement(world, 'body', name=b['name'], **_pose(T0))
                ET.SubElement(e, 'freejoint', name=rmeta['free'])
            else:
                e = ET.SubElement(elements[b['parent']], 'body', name=b['name'], **_pose(b['T_parent_body']))
                j = b['joint']
                if j is not None:
                    attr = dict(name=j['name'], type='hinge' if j['type'] == 'revolute' else 'slide', pos=_f(j['anchor_body']),
                                axis=_f(j['axis_body']), armature=f"{j['armature']:.10g}",
                                damping=f"{j['damping'] if joint_damping else 0.:.10g}")
                    if j['upper'] - j['lower'] < 1e3:
                        attr.update(limited='true', range=_f([j['lower'], j['upper']]))
                    ET.SubElement(e, 'joint', attr)
                    meta['coordinate_joint'][j['index']] = j['name']
            elements[b['name']] = e
            w = wheels.get(b['name'])
            if w is not None and w['rollers']:
                _inertial(e, w['hub_mass'], b['com'], w['hub_inertia'])
                axis = w['axis_body']
                hub = dict(name=f"{rid}/{w['id']}_hub", type='cylinder', radius=w['radius'] - 2.2 * ro['profile'][1].max(),
                           half_length=w['width'] / 2, rgba=[.12, .12, .13, 1.], collide=False,
                           T=np.block([[_z_frame(axis), np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))
                _geom(e, hub, 0, 0)
                names = []
                for k in w['rollers']:
                    rb = ET.SubElement(e, 'body', name=k['name'], **_pose(k['T']))
                    ET.SubElement(rb, 'joint', name=k['joint'], type='hinge', axis='0 0 1',
                                  damping=f"{ro['damping'] if joint_damping else 0.:.10g}", armature=f"{ro['armature']:.10g}")
                    ET.SubElement(rb, 'inertial', pos='0 0 0', mass=f"{ro['mass']:.10g}", diaginertia=_f(ro['inertia']))
                    shape = dict(type='mesh', mesh='roller') if roller_shape == 'hull' else \
                        dict(type='ellipsoid', size=_f(ro['ellipsoid']))
                    ET.SubElement(rb, 'geom', name=k['name'], rgba='.25 .25 .27 1', contype=str(ROLLER), conaffinity='0',
                                  friction=_f([w['friction'], .005, .0001]), solref=_f([ro['contact_time_constant'], ro['contact_damping_ratio']]),
                                  priority='1', **shape)
                    tag(k['name'], rid, 'roller', wheel=w['id'])
                    names.append(k['joint'])
                rmeta['rollers'] += names
                rmeta['wheels'][w['id']] = dict(joint=w['joint'], rollers=names, body=b['name'])
            else:
                _inertial(e, b['mass'], b['com'], b['inertia'])
                if w is not None:                               # conventional wheel: a sphere, met through explicit pairs
                    ty = w['tyre']
                    ET.SubElement(e, 'geom', name=ty['name'], type='sphere', size=_f([ty['radius']]), pos=_f(ty['centre']),
                                  rgba='.12 .12 .13 .25', contype='0', conaffinity='0', group='3')
                    tag(ty['name'], rid, 'roller', wheel=w['id'])
                    tyres.append((ty, w))
                    rmeta['wheels'][w['id']] = dict(joint=w['joint'], rollers=[], body=b['name'], tyre=ty['name'])
            for g in b['geoms']:
                _geom(e, g, bit, affinity)
                tag(g['name'], rid, 'robot', body=b['name'])
        ET.SubElement(elements[r['tool']['body']], 'site', name=f'{rid}_tool', size='.012', rgba='0 1 0 1', **_pose(r['tool']['T']))
        if r['sensor']:
            ET.SubElement(elements[r['sensor']['body']], 'site', name=f'{rid}_ft', size='.012', rgba='1 1 0 1',
                          **_pose(r['sensor']['T']))
            rmeta['sensor'] = dict(site=f'{rid}_ft', force=f'{rid}_ft_force', torque=f'{rid}_ft_torque',
                                   body=r['sensor']['body'])
        meta['robots'][rid] = rmeta

    # ------------------------------------------------------------------ payload
    p = spec['payload']
    pb = p['body']
    e = ET.SubElement(world, 'body', name=pb['name'], **_pose(chart_pose(q0[p['base']])))
    ET.SubElement(e, 'freejoint', name=f"{p['id']}_free")
    _inertial(e, pb['mass'], pb['com'], pb['inertia'])
    for g in pb['geoms']:
        _geom(e, g, PAYLOAD, STATIC | OBSTACLE)
        tag(g['name'], p['id'], 'payload', body=pb['name'])
    meta['payload'] = dict(free=f"{p['id']}_free", body=pb['name'], base=p['base'], id=p['id'])

    # ------------------------------------------------------------------ grasp welds, motors, sensors
    # A latched clamp holds the payload at three points (three connect constraints).  Unlike a weld
    # equality this transmits forces only, which MuJoCo reports exactly in the body interaction forces
    # read by the wrist sensors (its weld torque rows are not reported exactly there).
    eq = ET.SubElement(root, 'equality')
    for cid, c in spec['grasps'].items():
        names = []
        latch_points = spec['robots'][c['robot']]['tool'].get('latch_points') or LATCH_POINTS
        for k, a in enumerate(latch_points):
            names.append(f'{cid}#{k}')
            ET.SubElement(eq, 'connect', name=names[-1], body1=c['body1'], body2=c['body2'],
                          anchor=_f((c['T1'] @ np.r_[a, 1.])[:3]), active='false', solref=_f(weld_solref),
                          solimp='0.95 0.99 0.001')
        meta['welds'][cid] = dict(body1=c['body1'], body2=c['body2'], T1=c['T1'], robot=c['robot'], names=names,
                                  points=[(c['T1'] @ np.r_[a, 1.])[:3] for a in latch_points])
    if tyres:
        contact = ET.SubElement(root, 'contact')
        for ty, w in tyres:
            for ground in terrain_geoms:
                attr = dict(geom1=ground, geom2=ty['name'], solref=_f([ty['contact_time_constant'], ty['contact_damping_ratio']]),
                            solimp='0.9 0.95 0.001')
                if tyre == 'native':
                    attr.update(condim='3', friction=_f([w['friction'], w['friction'], .005, .0001, .0001]),
                                solreffriction=_f([ty['friction_time_constant'], 1.]))
                else:
                    attr.update(condim='1')
                ET.SubElement(contact, 'pair', attr)
        meta['tyre'] = tyre
    act = ET.SubElement(root, 'actuator')
    for port in spec['ports']:
        ET.SubElement(act, 'motor', name=port['id'], joint=spec['coordinate_ids'][port['index']], gear=f"{port['gear']:.10g}",
                      ctrlrange=_f(port['bounds']))
    sens = ET.SubElement(root, 'sensor')
    for rid, rmeta in meta['robots'].items():
        if 'sensor' in rmeta:
            ET.SubElement(sens, 'force', name=rmeta['sensor']['force'], site=rmeta['sensor']['site'])
            ET.SubElement(sens, 'torque', name=rmeta['sensor']['torque'], site=rmeta['sensor']['site'])
    ET.indent(root)
    return ET.tostring(root, encoding='unicode'), meta


def _z_frame(z):
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0., 1., 0.], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])
