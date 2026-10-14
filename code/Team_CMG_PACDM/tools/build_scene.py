#!/usr/bin/env python
"""Author the payload, scene and team records of the beam mission (physical data only).

All values here are declared design data for the demonstration: a storage rack
with occupied neighbouring levels, a route with a right-angle aisle turn, a
doorway, a cross aisle with a cart, a ramp onto a plateau, and two elevated
mounting supports with lateral guides.  Nothing here is measured from a real site.

Six configurations are written, two per platform:

* ``team2``: two RB-ROBOUT+ / UR30 facing each other, arms at the outer ends, 5 m steel beam;
* ``team3``: three of them in line with a longer and heavier beam;
* ``ridgeback2``: two Ridgeback / UR10e in line, both facing the direction of travel, 5 m aluminium beam;
* ``ridgeback3``: three of them in line with a longer and heavier beam;
* ``husky2``: two Husky A300 / UR10e in line at the rack, the same aluminium beam;
* ``husky3``: three of them in line with the longer beam.

The site is the same for all of them; what differs between the platforms is what
the robots can reach and carry: the heights of the rack level and of the mounting
supports, the reach at the stations and the beams (``SITE2`` to ``SITE_H3``).

The task itself (start poses, stations, route, tolerances) is written by
``tools/build_mission.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teamc import SCHEMA_PAYLOAD, SCHEMA_SCENE, SCHEMA_TEAM  # noqa: E402
from teamc.se3 import rot, transform  # noqa: E402

STEEL = 7850.
ALUMINIUM = 2700.
WHEEL_RADIUS = .127
ARM_X = -.76                       # arm base on the platform (upstream RB-ROBOUT+ lift position)

# Arm seeds of the elbow-up family used at a station (beam to the left / to the right of the base).
ARM_LEFT = dict(arm_shoulder_pan_joint=1.36, arm_shoulder_lift_joint=-.70, arm_elbow_joint=1.68,
                arm_wrist_1_joint=-2.54, arm_wrist_2_joint=-1.5708, arm_wrist_3_joint=-.21)
ARM_RIGHT = dict(ARM_LEFT, arm_shoulder_pan_joint=-1.78)


def hollow_box(length, width, height, wall, density=STEEL):
    """Mass and centroidal inertia of a rectangular hollow section, x along its length."""
    def solid(m, lx, ly, lz):
        return np.diag([m * (ly ** 2 + lz ** 2) / 12, m * (lx ** 2 + lz ** 2) / 12, m * (lx ** 2 + ly ** 2) / 12])
    outer = density * width * height * length
    inner = density * (width - 2 * wall) * (height - 2 * wall) * length
    return outer - inner, solid(outer, length, width, height) - solid(inner, length, width - 2 * wall, height - 2 * wall)


def box(sid, lo, hi, rgba, role, friction=.6, collide=True):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return dict(id=sid, type='box', half_size=((hi - lo) / 2).tolist(), T_world_geom=transform(p=(lo + hi) / 2).tolist(),
                rgba=rgba, role=role, friction=friction, collide=collide)


def placed_box(sid, T_frame, lo, hi, rgba, role, friction=.6, collide=True):
    """A box given by its corners in a local frame ``T_frame`` (used for the yawed mounting supports)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return dict(id=sid, type='box', half_size=((hi - lo) / 2).tolist(),
                T_world_geom=(np.asarray(T_frame) @ transform(p=(lo + hi) / 2)).tolist(),
                rgba=rgba, role=role, friction=friction, collide=collide)


def payload_record(name, length, width, height, wall, grasps, pads_x, density=STEEL, material='steel'):
    mass, inertia = hollow_box(length, width, height, wall, density)
    down = rot([1., 0, 0], np.pi)                      # tool z points into the beam from above
    frames = {f'grasp_{k}': dict(body='beam', T_body_frame=transform(down, [x, 0, 0]).tolist())
              for k, x in grasps.items()}
    frames['rest'] = dict(body='beam', T_body_frame=np.eye(4).tolist())
    for k, x in zip('ab', pads_x):
        frames[f'pad_{k}'] = dict(body='beam', T_body_frame=transform(p=[x, 0, -height / 2]).tolist())
    return dict(schema=SCHEMA_PAYLOAD, name=name, units='SI', root='beam',
                bodies=[dict(id='beam', mass_kg=mass, com_m=[0., 0., 0.], inertia_kg_m2=inertia.tolist())],
                joints=[], frames=frames,
                geometry=[dict(body='beam', name='beam', type='box', half_size=[length / 2, width / 2, height / 2],
                               T_body_geom=np.eye(4).tolist(), rgba=[.86, .30, .12, 1.], collide=True, friction=.5)],
                section=dict(type='rectangular hollow section', length_m=length, width_m=width, height_m=height,
                             wall_m=wall, density_kg_m3=density, mass_per_m=mass / length),
                declared=f'Design data: {material} rectangular hollow section, mass and inertia from its geometry.')


def scene_record(name, p):
    """``p`` holds the site parameters (see ``SITE2``/``SITE3``)."""
    grey, wall_c, rack_c, steel = [.50, .52, .55, 1.], [.80, .82, .84, 1.], [.20, .33, .55, 1.], [.62, .64, .68, 1.]
    beam_c, mount_c, stub_c = [.45, .22, .14, 1.], [.85, .65, .10, 1.], [.74, .76, .78, 1.]
    hb, wb = p['beam_height'], p['beam_width']
    half = p['beam_length'] / 2
    H, t = p['wall_height'], .10
    statics, frames = [], {}

    # ------------------------------------------------------------------ rack on the north wall of the storage hall
    yr, zr = p['rack_y'], p['rack_z']                       # beam centroid in the rack
    pad_top = zr - hb / 2
    for k, x in zip('ab', p['pads_x']):
        statics.append(box(f'rack_upright_{k}', [x - .05, yr + .30, 0.], [x + .05, yr + .40, zr + p['level_gap'] + .55],
                           rack_c, 'rack'))
        for level, top in (('low', pad_top - p['level_gap']), ('mid', pad_top), ('high', pad_top + p['level_gap'])):
            statics.append(box(f'rack_arm_{level}_{k}', [x - .04, yr - .15, top - .06], [x + .04, yr + .30, top], rack_c,
                               'pad' if level == 'mid' else 'rack', friction=.5))
            statics.append(box(f'rack_lip_{level}_{k}', [x - .04, yr - .15, top], [x + .04, yr - .13, top + .02], rack_c,
                               'rack'))
        frames[f'rack_pad_{k}'] = transform(p=[x, yr, pad_top]).tolist()
    frames['rack_rest'] = transform(p=[0., yr, zr]).tolist()
    for level, dz in (('low', -p['level_gap']), ('high', p['level_gap'])):
        statics.append(box(f'neighbour_{level}', [-half, yr - wb / 2, zr + dz - hb / 2], [half, yr + wb / 2, zr + dz + hb / 2],
                           beam_c, 'neighbour'))
    statics.append(box('neighbour_behind', [-half, yr + .14 - wb / 2, zr - hb / 2], [half, yr + .14 + wb / 2, zr + hb / 2],
                       beam_c, 'neighbour'))

    # ------------------------------------------------------------------ storage hall, aisles, corner
    w, xc = p['aisle'], p['x_c']                              # aisle width, centre line of the north-going aisle
    x_w, x_e = xc - w / 2, xc + w / 2                        # walls of the north-going aisle
    y_s, y_n = p['aisle1_y'] - w / 2, p['aisle1_y'] + w / 2  # walls of the east-going aisle
    hx0, hy0, x_a = p['hall_x0'], p['hall_y0'], p['x_a']
    y0, y1, rise = p['y_ramp0'], p['y_ramp1'], p['ramp_rise']
    yc, gap = p['cart_y'], p['cart_gap']
    statics += [
        box('wall_rack', [hx0 - t, yr + .40, 0.], [x_a, yr + .50, H], wall_c, 'wall'),
        box('wall_hall_west', [hx0 - t, hy0 - t, 0.], [hx0, yr + .40, H], wall_c, 'wall'),
        box('wall_hall_south', [hx0, hy0 - t, 0.], [x_a, hy0, H], wall_c, 'wall'),
        box('wall_hall_east', [x_a, hy0 - t, 0.], [x_a + t, y_s, H], wall_c, 'wall'),
        box('wall_aisle1_south', [x_a, y_s - t, 0.], [x_e + t, y_s, H], wall_c, 'wall'),
        box('wall_aisle1_north', [x_a, y_n, 0.], [x_w, yr + .50, H], wall_c, 'wall'),
        box('wall_aisle2_east_a', [x_e, y_s, 0.], [x_e + t, yc - gap / 2, H], wall_c, 'wall'),
        box('wall_aisle2_east_b', [x_e, yc + gap / 2, 0.], [x_e + t, y1, H + rise], wall_c, 'wall'),
        box('wall_aisle2_west_a', [x_w - t, y_n, 0.], [x_w, yc - gap / 2, H], wall_c, 'wall'),
        box('wall_aisle2_west_b', [x_w - t, yc + gap / 2, 0.], [x_w, y1, H + rise], wall_c, 'wall'),
    ]
    for side, xa, xb in (('west', x_w - 2.2, x_w - t), ('east', x_e + t, x_e + 2.2)):      # cross aisle of the cart
        statics += [box(f'wall_cross_{side}_s', [xa, yc - gap / 2 - t, 0.], [xb, yc - gap / 2, H], stub_c, 'wall'),
                    box(f'wall_cross_{side}_n', [xa, yc + gap / 2, 0.], [xb, yc + gap / 2 + t, H], stub_c, 'wall')]
    yd, dw, xd = p['door_y'], p['door_width'], xc + p['door_offset']
    statics += [
        box('door_jamb_west', [x_w, yd - .10, 0.], [xd - dw / 2, yd + .10, H], wall_c, 'door'),
        box('door_jamb_east', [xd + dw / 2, yd - .10, 0.], [x_e, yd + .10, H], wall_c, 'door'),
        box('door_lintel', [xd - dw / 2, yd - .10, p['door_height']], [xd + dw / 2, yd + .10, H], wall_c, 'door'),
    ]

    # ------------------------------------------------------------------ ramp and plateau
    slope = np.arctan2(rise, y1 - y0)
    length = np.hypot(y1 - y0, rise)
    normal = np.array([0., -np.sin(slope), np.cos(slope)])
    statics.append(dict(id='ramp', type='box', half_size=[w / 2, length / 2, .05],
                        T_world_geom=transform(rot([1., 0, 0], slope),
                                               np.array([xc, (y0 + y1) / 2, rise / 2]) - .05 * normal).tolist(),
                        rgba=grey, role='terrain', friction=.8, collide=True))
    px0, px1, y_end = p['plateau_x'][0], p['plateau_x'][1], p['y_end']
    statics.append(box('plateau', [px0, y1, rise - .10], [px1, y_end, rise], grey, 'terrain', friction=.8))
    statics += [box('wall_plateau_south_w', [px0 - t, y1 - t, 0.], [x_w - t, y1, H + rise], wall_c, 'wall'),
                box('wall_plateau_south_e', [x_e + t, y1 - t, 0.], [px1 + t, y1, H + rise], wall_c, 'wall'),
                box('wall_plateau_west', [px0 - t, y1, 0.], [px0, y_end, H + rise], wall_c, 'wall'),
                box('wall_plateau_east', [px1, y1, 0.], [px1 + t, y_end, H + rise], wall_c, 'wall'),
                box('wall_plateau_north', [px0 - t, y_end, 0.], [px1 + t, y_end + t, H + rise], wall_c, 'wall')]
    terrain = dict(friction=.8, patches=[
        dict(id='floor', point=[0., 0., 0.], normal=[0., 0., 1.], region=[[hx0 - 2., hy0 - 2.], [px1 + 2., y0]]),
        dict(id='ramp', point=[xc, y0, 0.], normal=normal.tolist(), region=[[x_w, y0], [x_e, y1]]),
        dict(id='plateau', point=[xc, y1, rise], normal=[0., 0., 1.], region=[[px0, y1], [px1, y_end]])])

    # ------------------------------------------------------------------ mounting supports on the plateau
    xm, ym, zm = p['mount']                                   # beam centroid when installed
    T_m = transform(rot([0, 0, 1.], p['mount_yaw']), [xm, ym, zm])
    top = -hb / 2                                             # pad top in the mount frame
    g = wb / 2 + p['guide_gap']
    for k, x in zip('ab', p['pads_x']):
        statics += [placed_box(f'mount_post_{k}', T_m, [x - .06, -.06, rise - zm], [x + .06, .06, top - .03], steel, 'mount'),
                    placed_box(f'mount_pad_{k}', T_m, [x - .08, -.10, top - .03], [x + .08, .10, top], mount_c, 'pad',
                               friction=.5),
                    placed_box(f'mount_guide_{k}_s', T_m, [x - .08, -g - .02, top], [x + .08, -g, top + .04], mount_c, 'guide'),
                    placed_box(f'mount_guide_{k}_n', T_m, [x - .08, g, top], [x + .08, g + .02, top + .04], mount_c, 'guide')]
        frames[f'mount_pad_{k}'] = (T_m @ transform(p=[x, 0., top])).tolist()
    frames['mount_rest'] = T_m.tolist()

    cart = p['cart']
    return dict(schema=SCHEMA_SCENE, name=name, units='SI', terrain=terrain, frames=frames, statics=statics,
                obstacle=dict(id='cart', half_size=cart['half_size'], rgba=[.15, .55, .25, 1.],
                              start=[x_w - 1.6, yc, cart['half_size'][2]], end=[x_e + 1.6, yc, cart['half_size'][2]],
                              speed_m_s=cart['speed'], acceleration_m_s2=cart['acceleration'],
                              trigger=dict(type='payload_point_reaches', axis='y', value=cart['trigger_y'],
                                           note='the cart starts to cross when any part of the payload passes this line'),
                              note='A cart crossing the north-going aisle.  The plant moves it kinematically.'),
                site=p, declared='Design data for the demonstration; not a surveyed site.')


def team_record(name, members, files, payload_seed, bases, model='rbrobout_ur30', commanded=('lift_top_joint',)):
    return dict(
        schema=SCHEMA_TEAM, name=name, gravity_m_s2=[0., 0., -9.81], files=files,
        robots=[dict(id=rid, model=model, commanded=list(commanded)) for rid in members],
        payload=dict(id='beam'),
        grasps=[dict(id=f'grasp_{rid}', robot=rid, tool_frame='tool', payload_frame=f'grasp_{rid}', type='weld',
                     share=1.) for rid in members],
        rests=[dict(id='rack', payload_frame='rest', scene_frame='rack_rest',
                    pads=[dict(id=f'rack_pad_{k}', scene_frame=f'rack_pad_{k}', friction=.5, half_width_m=.03) for k in 'ab']),
               dict(id='mount', payload_frame='rest', scene_frame='mount_rest',
                    pads=[dict(id=f'mount_pad_{k}', scene_frame=f'mount_pad_{k}', friction=.5, half_width_m=.03) for k in 'ab'])],
        seed=dict(beam=payload_seed, **{rid: dict(base=b['base'], joints=dict(b['arm'], **{j: b['lift'] for j in commanded}))
                                        for rid, b in bases.items()}),
        declared='Seeds are the station configurations at the rack (bases beside the rack, tools on the beam).')


def station(arm_xy, yaw, lift, beam_side, z=WHEEL_RADIUS, arm_x=ARM_X, seeds=None):
    """Base chart value that puts the arm base at ``arm_xy`` with heading ``yaw``.

    ``beam_side`` is the world direction from the arm base to the beam; it selects the arm seed
    (``seeds``: the left and right seeds of the platform; the UR30 ones by default).
    """
    R = rot([0, 0, 1.], yaw)[:2, :2]
    centre = np.asarray(arm_xy, float) - R @ np.array([arm_x, 0.])
    side = np.cross(np.r_[R[:, 0], 0.], np.r_[np.asarray(beam_side, float), 0.])[2]
    seeds = dict(left=ARM_LEFT, right=ARM_RIGHT) if seeds is None else seeds
    return dict(base=[float(centre[0]), float(centre[1]), z, float(yaw), 0., 0.], lift=lift,
                arm=dict(seeds['left' if side > 0 else 'right']))


SITE2 = dict(
    beam_length=5.0, beam_width=.06, beam_height=.10, beam_wall=.0036, pads_x=[-1.5, 1.5],
    rack_y=3.0, rack_z=1.25, level_gap=.75, wall_height=2.6,
    hall_x0=-6.5, hall_y0=-3.0, x_a=5.0, aisle=3.0, aisle1_y=1.5, x_c=9.6,
    door_y=6.8, door_width=1.7, door_offset=.0, door_height=2.20,
    cart_y=13.0, cart_gap=1.6, cart=dict(half_size=[.45, .30, .45], speed=.35, acceleration=.4, trigger_y=10.6),
    y_ramp0=14.2, y_ramp1=17.2, ramp_rise=.157, plateau_x=[4.6, 13.0], y_end=27.0,
    mount=[8.9, 25.3, .157 + 1.40], mount_yaw=float(np.deg2rad(8.)), guide_gap=.006,
    reach_station=.95, lift_station=.30,
    grasps=dict(r1=dict(x=-2.0, yaw=0.), r2=dict(x=2.0, yaw=float(np.pi))))
SITE3 = dict(
    SITE2, beam_length=5.5, beam_width=.06, beam_height=.12, beam_wall=.004,
    grasps=dict(r1=dict(x=-2.61, yaw=0.), r2=dict(x=2.61, yaw=float(np.pi)), r3=dict(x=-.76, yaw=0.)))


# ---------------------------------------------------------------------- Clearpath Ridgeback with a UR10e
# The Ridgeback has no lift column: the arm takes every height.  Its arm base is 0.31 m above the floor,
# so the rack level and the mounting supports are lower than for the RB-ROBOUT+, and the beams are
# aluminium sections whose share stays inside the range of the UR10e's wrist sensor.
# The robots stand in line beside the beam and face the direction of travel, as the RB-ROBOUT+ teams do:
# they drive (and do not strafe) along the aisles and over the two edges of the ramp.  The rear rocker of
# the base is fixed and the front one pivots, so the two rear wheels and the front pivot carry the base;
# an arm that works to the side moves the centre of pressure towards one edge of that triangle, which is
# why the stations are closer to the beam than for the RB-ROBOUT+ (README, "Ridgeback formation").
RIDGEBACK_Z = .0759 - .05          # chassis frame above flat ground: wheel radius minus the axle offset
RIDGEBACK_ARM = dict(left=dict(arm_shoulder_pan_joint=1.36, arm_shoulder_lift_joint=-1.34, arm_elbow_joint=1.28,
                               arm_wrist_1_joint=-1.51, arm_wrist_2_joint=-1.5708, arm_wrist_3_joint=-.21),
                     right=dict(arm_shoulder_pan_joint=-1.78, arm_shoulder_lift_joint=-1.34, arm_elbow_joint=1.28,
                                arm_wrist_1_joint=-1.51, arm_wrist_2_joint=-1.5708, arm_wrist_3_joint=-.21))
SITE_R2 = dict(
    SITE2, beam_length=5.0, beam_width=.05, beam_height=.10, beam_wall=.003,
    rack_z=.90, mount=[8.9, 25.3, .157 + 1.00], reach_station=.80, lift_station=None,
    grasps=dict(r1=dict(x=-1.8, yaw=0.), r2=dict(x=1.8, yaw=0.)))
SITE_R3 = dict(
    SITE_R2, beam_length=5.5, beam_width=.05, beam_height=.12, beam_wall=.003,
    grasps=dict(r1=dict(x=-2.2, yaw=0.), r2=dict(x=2.2, yaw=0.), r3=dict(x=-.6, yaw=0.)))


# ---------------------------------------------------------------------- Clearpath Husky A300 with a UR10e
# The same arm, beams and heights as for the Ridgeback, so that the two Clearpath teams differ in the base alone.
# The arm stands on the centre of the top plate, over the middle of the four wheels.  At the rack the robots
# stand in line beside the beam, as the Ridgeback teams do; what a base that cannot move sideways changes is
# how they get there, how they leave and how they reach the mounting supports (tools/build_mission.py).
HUSKY_Z = .1651 - .02913           # chassis frame above flat ground: wheel radius minus the height of the axles in it
# The arm reaches the same tool pose on two shoulder branches; upper arm and forearm lie 0.16 m to one side of the
# arm base or to the other.  With the branch of the Ridgeback teams they lie behind the arm base of a base that
# stands in line beside the beam, and the rear wheels of the Husky, whose own centre of mass is behind its middle,
# carry 60 % of the weight.  A skid turn needs a wheel torque that grows with the load of the wheel, so the other
# branch is taken here: it puts the links ahead of the arm base and the loads of the two axles within 8 %.
HUSKY_ARM = dict(left=dict(arm_shoulder_pan_joint=-1.36, arm_shoulder_lift_joint=-1.80, arm_elbow_joint=-1.28,
                           arm_wrist_1_joint=-1.63, arm_wrist_2_joint=1.5708, arm_wrist_3_joint=.21),
                 right=dict(arm_shoulder_pan_joint=1.78, arm_shoulder_lift_joint=-1.80, arm_elbow_joint=-1.28,
                            arm_wrist_1_joint=-1.63, arm_wrist_2_joint=1.5708, arm_wrist_3_joint=.21))
SITE_H2 = dict(SITE_R2)
SITE_H3 = dict(SITE_R3)


def build():
    out = {}
    for tag, p, payload_file, payload_name, team_name in (
            ('2', SITE2, 'payloads/beam_rhs_100x60_5m.json', 'Steel RHS 100 x 60 x 3.6, 5 m',
             'Two RB-ROBOUT+ / UR30 and a 5 m beam'),
            ('3', SITE3, 'payloads/beam_rhs_120x60_5p5m.json', 'Steel RHS 120 x 60 x 4, 5.5 m',
             'Three RB-ROBOUT+ / UR30 in line and a 5.5 m beam')):
        gx = {rid: g['x'] for rid, g in p['grasps'].items()}
        out[payload_file] = payload_record(payload_name, p['beam_length'], p['beam_width'], p['beam_height'],
                                           p['beam_wall'], gx, p['pads_x'])
        out[f'scenes/hall{tag}.json'] = scene_record(f'Hall route, team of {tag}', p)
        bases = {rid: station([g['x'], p['rack_y'] - p['reach_station']], g['yaw'], p['lift_station'], [0., 1.])
                 for rid, g in p['grasps'].items()}
        out[f'teams/team{tag}.json'] = team_record(
            team_name, list(p['grasps']),
            dict(payload=payload_file, scene=f'scenes/hall{tag}.json', robots=dict(rbrobout_ur30='robots/rbrobout_ur30.json')),
            [0., p['rack_y'], p['rack_z'], 0., 0., 0.], bases)
    for tag, p, payload_file, payload_name, team_name in (
            ('2', SITE_R2, 'payloads/beam_al_rhs_100x50_5m.json', 'Aluminium RHS 100 x 50 x 3, 5 m',
             'Two Ridgeback / UR10e and a 5 m beam'),
            ('3', SITE_R3, 'payloads/beam_al_rhs_120x50_5p5m.json', 'Aluminium RHS 120 x 50 x 3, 5.5 m',
             'Three Ridgeback / UR10e in line and a 5.5 m beam')):
        gx = {rid: g['x'] for rid, g in p['grasps'].items()}
        out[payload_file] = payload_record(payload_name, p['beam_length'], p['beam_width'], p['beam_height'],
                                           p['beam_wall'], gx, p['pads_x'], ALUMINIUM, 'aluminium')
        out[f'scenes/hall{tag}_ridgeback.json'] = scene_record(f'Hall route, team of {tag} (Ridgeback heights)', p)
        bases = {rid: station([g['x'], p['rack_y'] - p['reach_station']], g['yaw'], None, [0., 1.], RIDGEBACK_Z, 0.,
                              RIDGEBACK_ARM) for rid, g in p['grasps'].items()}
        out[f'teams/ridgeback{tag}.json'] = team_record(
            team_name, list(p['grasps']),
            dict(payload=payload_file, scene=f'scenes/hall{tag}_ridgeback.json',
                 robots=dict(ridgeback_ur10e='robots/ridgeback_ur10e.json')),
            [0., p['rack_y'], p['rack_z'], 0., 0., 0.], bases, 'ridgeback_ur10e', ())
    for tag, p, payload_file, team_name in (
            ('2', SITE_H2, 'payloads/beam_al_rhs_100x50_5m.json', 'Two Husky A300 / UR10e and a 5 m beam'),
            ('3', SITE_H3, 'payloads/beam_al_rhs_120x50_5p5m.json', 'Three Husky A300 / UR10e in line and a 5.5 m beam')):
        # the beams are those of the Ridgeback teams (the same grasp positions): their records are written above
        out[f'scenes/hall{tag}_husky.json'] = scene_record(f'Hall route, team of {tag} (Husky heights)', p)
        bases = {rid: station([g['x'], p['rack_y'] - p['reach_station']], g['yaw'], None, [0., 1.], HUSKY_Z, 0.,
                              HUSKY_ARM) for rid, g in p['grasps'].items()}
        out[f'teams/husky{tag}.json'] = team_record(
            team_name, list(p['grasps']),
            dict(payload=payload_file, scene=f'scenes/hall{tag}_husky.json',
                 robots=dict(husky_a300_ur10e='robots/husky_a300_ur10e.json')),
            [0., p['rack_y'], p['rack_z'], 0., 0., 0.], bases, 'husky_a300_ur10e', ())
    return out


def main():
    for rel, record in build().items():
        path = ROOT / 'inputs' / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8', newline='\n')
        print('wrote', path.relative_to(ROOT))


if __name__ == '__main__':
    main()
