#!/usr/bin/env python
"""Design the mission of a team: stations, carrying formation, route, speed zones and gates.

Everything written here is task data (what the team should do), derived from the
physical records and a few declared choices listed in ``CHOICES``.  The route is
designed on the full three-dimensional shapes of the formation: lanes are placed
so that the formation is centred in each aisle, and the positions through the
aisle turn maximize the smallest clearance at every heading.

A team whose bases cannot move sideways (conventional wheels, skid steering) gets
the same task with the legs such a base cannot drive replaced (``build_steered``):
the bases reach their stations along the rack instead of sideways into them; on
the route every base heads the way its own origin moves, so its heading leaves
that of the payload wherever the payload turns or shifts; and on the plateau the
payload is turned and brought to the supports while the bases keep driving
forwards, each on its own path, and arrive facing the supports.

    python tools/build_mission.py team2
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import betainc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mission.controller import arm_joints, lift_joint  # noqa: E402
from mission.route import (FormationClearance, PlanarPath, bottleneck_turn, compose, densify, lane_change, line,  # noqa: E402
                           max_clearance_turn, planar, planar_inverse, relative, smooth_poses)
from mission.steering import (arc_length, drawn_headings, follow_tangents, hermite, led_poses, line_and_arc,  # noqa: E402
                              turn_in_place, unit, yaw_on_line)
from teamc.chords import TeamAssembler  # noqa: E402
from teamc.geometry import Geometry  # noqa: E402
from teamc.interfaces import TeamModel  # noqa: E402
from teamc.io import INPUTS, load_team, write_json  # noqa: E402
from teamc.loads import LoadSharing  # noqa: E402
from teamc.se3 import chart_pose, inv  # noqa: E402

SCHEMA_MISSION = 'cmg.team-mission/1.0'

CHOICES = dict(
    reach_carry=.60,             # arm base to payload axis while carrying (m)
    lift_carry=.02,              # lift position while carrying on level ground (m); kept off the end stop
    z_carry=1.25,                # payload centroid above the ground under the team while carrying (m)
    raise_at_rack=.08,           # lift-off height before the payload is pulled out of the rack (m)
    place_above=.10,             # payload centroid above its installed height when it is presented (m)
    approach_above=.15,          # tool above the grasp frame before it descends (m)
    cruise_speed=.45, turn_speed=.28, door_speed=.30, ramp_speed=.30, rotate_speed=.28, final_speed=.15,
    accel=.25, clearance_margin=.10, stop_margin=.60,
    starts=dict(team2=dict(r1=[-4.4, -.4, .9], r2=[3.4, -1.2, 2.4]),
                team3=dict(r1=[-4.9, -.2, .9], r2=[3.4, -1.2, 2.4], r3=[.3, -1.5, .5])),
    approach_speed=.40, dock_distance=1.0, dock_speed=.25,   # the last leg into the station is a straight sideways move
)
# Declared admissible bounds checked by the audit.
BOUNDS = dict(grasp_force_N=450., grasp_moment_Nm=45., effort_ratio=.90, traction_utilization=.60,
              tipping_margin_m=.10, min_wheel_load_N=150., clearance_m=.02, payload_tilt_rad=.012,
              payload_position_m=.02,
              note='declared admissible bounds checked by the audit; clearance excludes the supports the payload rests on')

# Ridgeback / UR10e teams: what differs from the choices above.  The arm takes every height (there is no
# lift), the beam is carried lower and closer, and the bounds follow the smaller robot: the grasp force and
# moment bounds come from the range of the UR10e wrist sensor, which also carries the clamp, and the tipping
# margin is measured to the support triangle of the rocker base (two rear wheels, front pivot).
RIDGEBACK = dict(
    choices=dict(reach_carry=.45, z_carry=.90,
                 starts=dict(ridgeback2=dict(r1=[-4.4, -.4, .9], r2=[3.4, -1.2, -.7]),
                             ridgeback3=dict(r1=[-4.9, -.2, .9], r2=[3.4, -1.2, -.7], r3=[.3, -1.5, .5]))),
    drop=('lift_carry',),
    bounds=dict(BOUNDS, grasp_force_N=85., grasp_moment_Nm=10., tipping_margin_m=.06, min_wheel_load_N=100.,
                note='declared admissible bounds checked by the audit; clearance excludes the supports the payload rests on. '
                     'Grasp force: the 100 N range of the UR10e wrist sensor less the weight of the clamp it also carries '
                     '(11.8 N), rounded down; grasp moment: its 10 N m range. '
                     'Tipping margin: distance of the centre of pressure from the triangle of the two rear wheels '
                     'and the front rocker pivot.'),
    # gains for the lighter arm and base; a firmer base position loop, because the rollers of this wheel make it
    # move sideways about 5 % more slowly than the ideal rolling rows of the compiled model say
    controller=dict(arm_stiffness=[9000., 9000., 6000., 1500., 1500., 1500.], wheel_gain=12.,
                    base_kp=[3., 3., 3.], base_ki=[1., 1., 1.]),
)
# Husky A300 / UR10e teams: the arm, the beams and the heights of the Ridgeback teams on a skid-steered base.
# The bounds on the grasp are those of the UR10e wrist sensor again.  Traction is the resultant the wheels pass
# to the ground over the friction capacity of the base: its wheels slide whenever it turns, so the friction of
# a single wheel is at its limit by design and says nothing about grip.  The yaw rate of a base is bounded by
# the task (``base_yaw_rate`` while carrying, ``turn_rate`` for a turn on the spot without load): the user
# manual allows the full payload only for "wide turns", and a turn on the spot on a high-friction floor only
# with the all-terrain payload, which arm and beam share stay under.
HUSKY = dict(
    choices=dict(reach_carry=.45, z_carry=.90,
                 starts=dict(husky2=dict(r1=[-4.6, -.6, 2.2], r2=[-1.0, -1.4, -.7]),
                             husky3=dict(r1=[-5.2, -1.0, 2.2], r2=[-.6, -1.6, -.7], r3=[-3.0, -1.9, 2.6])),
                 approach_speed=.35, dock_speed=.25, dock_distance=1.0,
                 via_spacing=2.4,             # the bases enter the lane along the rack at least this far apart (m)
                 arc_radius=1.5,              # of the curve into the lane along the rack (m)
                 turn_rate=.25,               # largest yaw rate of a turn on the spot (rad/s)
                 base_yaw_rate=.15,           # largest yaw rate of a base while the team carries (rad/s)
                 rotation_length=.60,         # a turn of a base counts as this length per radian in its path length (m)
                 lane_change=3.4,             # length over which the team moves from the rack to its lane (m)
                 turn_smooth=.50,             # length over which the positions of the aisle turn are smoothed (m)
                 swing_advance=4.5,           # the leading base leaves the lane at most this far before its last straight (m)
                 swing_weights=(1., 1.6),     # end tangents of the leading base's curve, in units of its chord
                 swing_shape=(6., 2.2),       # exponents of the blend of the payload yaw to that of the supports
                 final_straight=.80,          # the bases drive this far straight towards the supports at the end (m)
                 min_turn_radius=.80,         # smallest radius of the path of a base while the team carries (m)
                 crease_heading=.01,          # a base crosses the ends of the ramp square to them within this (rad)
                 retreat=.5, retreat_speed=.20),
    drop=('lift_carry',),
    bounds=dict(BOUNDS, grasp_force_N=85., grasp_moment_Nm=10., min_wheel_load_N=100.,
                note='declared admissible bounds checked by the audit; clearance excludes the supports the payload rests on. '
                     'Grasp force: the 100 N range of the UR10e wrist sensor less the weight of the clamp it also carries '
                     '(11.8 N), rounded down; grasp moment: its 10 N m range. Traction: resultant ground force of the wheels '
                     'of a base over its friction capacity (the wheels of a skid-steered base slide whenever it turns).'),
    # the wheel speed loop is that of a drive: stiff, with integral action, because a wheel that slides needs its
    # friction torque whatever the speed
    controller=dict(arm_stiffness=[9000., 9000., 6000., 1500., 1500., 1500.], wheel_gain=60., wheel_integral=2000.),
)
PLATFORM = dict(ridgeback2=RIDGEBACK, ridgeback3=RIDGEBACK, husky2=HUSKY, husky3=HUSKY)


def payload_planar(q, comp):
    c = q[comp.payload['base']]
    return np.array([c[0], c[1], c[3]])


def base_planar(q, robot):
    c = q[robot.base]
    return np.array([c[0], c[1], c[3]])


def configure(comp, asm, q_from, payload_pose, offsets, lifts, steps=12):
    """Assemble the team at a payload chart pose with given base offsets and lifts (continuation from ``q_from``)."""
    q0 = np.asarray(q_from, float).copy()
    target = q0.copy()
    target[comp.payload['base']] = payload_pose
    planar_p = np.array([payload_pose[0], payload_pose[1], payload_pose[3]])
    for rid, r in comp.robots.items():
        b = compose(planar_p, offsets[rid])
        target[r.base[[0, 1, 3]]] = b
        if lift_joint(r) is not None:
            target[r.joint[lift_joint(r)]] = lifts[rid]
    q = q0
    for s in np.linspace(0., 1., steps + 1)[1:]:
        q_s = q.copy()
        for idx in [comp.payload['base']] + [r.base for r in comp.robots.values()]:
            q_s[idx] = q0[idx] + s * (target[idx] - q0[idx])
        for r in comp.robots.values():
            if lift_joint(r) is not None:
                k = r.joint[lift_joint(r)]
                q_s[k] = q0[k] + s * (target[k] - q0[k])
        q, _ = asm.solve(q_s)
    return q


def build(team_name, verbose=True):
    comp = load_team(team_name)
    if any(r.drive == 'friction' for r in comp.robots.values()):
        return build_steered(team_name, comp, verbose)
    site = comp.scene['site']
    geo = Geometry(comp)
    tree = comp.tree
    c = dict(CHOICES)
    platform = PLATFORM.get(team_name, {})
    c.update(platform.get('choices', {}))
    for key in platform.get('drop', ()):
        c.pop(key)
    say = print if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------------ station and carrying formation
    q_seed = comp.seed()
    asm = TeamAssembler(comp, q_seed)
    q_station, _ = asm.solve(q_seed)
    rack = payload_planar(q_station, comp)
    station = {rid: relative(rack, base_planar(q_station, r)) for rid, r in comp.robots.items()}
    shift = site['reach_station'] - c['reach_carry']
    carry = {rid: x + np.array([0., -np.sign(x[1]) * shift, 0.]) for rid, x in station.items()}
    lifted = [rid for rid, r in comp.robots.items() if lift_joint(r) is not None]
    lifts_station = {rid: site['lift_station'] for rid in lifted}
    lifts_carry = {rid: c['lift_carry'] for rid in lifted}
    carry_pose = np.array([rack[0], rack[1], c['z_carry'], rack[2], 0., 0.])
    q_carry = configure(comp, asm, q_station, carry_pose, carry, lifts_carry)
    T_carry, _ = tree.sweep(q_carry)
    clear = FormationClearance(geo, T_carry, rack)
    placed = geo.place(T_carry)
    Pinv = planar_inverse(planar(*rack))
    lo = np.min([(Pinv[:3, :3] @ np.array([[p.lo[0], p.lo[1], 0.], [p.hi[0], p.hi[1], 0.], [p.lo[0], p.hi[1], 0.], [p.hi[0], p.lo[1], 0.]]).T
                  + Pinv[:3, 3:4]).min(axis=1) for p in placed], axis=0)
    hi = np.max([(Pinv[:3, :3] @ np.array([[p.lo[0], p.lo[1], 0.], [p.hi[0], p.hi[1], 0.], [p.lo[0], p.hi[1], 0.], [p.hi[0], p.lo[1], 0.]]).T
                  + Pinv[:3, 3:4]).max(axis=1) for p in placed], axis=0)
    half_length = float(max(abs(lo[0]), abs(hi[0])))
    reach = float(np.sqrt(max(abs(lo[0]), abs(hi[0])) ** 2 + max(abs(lo[1]), abs(hi[1])) ** 2))   # swept radius about the payload centre
    mid = .5 * (lo[1] + hi[1])
    top = geo.top(T_carry)
    say(f'formation: x [{lo[0]:.2f}, {hi[0]:.2f}], y [{lo[1]:.2f}, {hi[1]:.2f}] (width {hi[1] - lo[1]:.2f} m), top {top:.2f} m')

    # ------------------------------------------------------------------ lanes
    xc, w = site['x_c'], site['aisle']
    x_w, y_n = xc - w / 2, site['aisle1_y'] + w / 2
    lane1_y = site['aisle1_y'] - mid                      # payload axis while heading east
    lane2_x = xc + mid                                    # payload axis while heading north
    extracted = np.array([rack[0], rack[1] - shift, 0.])  # payload after it has been pulled out of the rack
    entry = np.array([x_w - 2.0, lane1_y, 0.])
    exit_ = np.array([lane2_x, y_n + 2.0, np.pi / 2])
    say('designing the aisle turn ...')
    turn, turn_clear = max_clearance_turn(clear, entry, exit_)
    if turn_clear.min() < c['clearance_margin']:
        say('  the local search closed the turn (clearance', np.round(turn_clear.min(), 3), 'm); scanning the corner at every heading ...')
        turn, turn_clear = bottleneck_turn(clear, entry, exit_)
        c['turn_search'] = 'scan of the corner on a 0.15 m grid at every heading (mission.route.bottleneck_turn)'
    say('  turn clearance by heading (m):', np.round(turn_clear, 3))
    mount = np.array([site['mount'][0], site['mount'][1], site['mount_yaw']])
    preplace = compose(mount, [0., -shift, 0.])
    y_rot = site['y_ramp1'] + reach + .6                 # the whole formation is on the plateau while it turns
    if preplace[1] - reach < y_rot - shift - 1e-9 and np.linalg.norm(preplace[:2] - [lane2_x, y_rot]) < reach + .5:
        raise RuntimeError('site: the mounting supports are inside the turning circle of the formation')
    pieces = [lane_change(extracted, [rack[0] + 2.6, lane1_y, 0.]),
              line([rack[0] + 2.6, lane1_y, 0.], entry),
              densify(turn, 240),
              line(exit_, [lane2_x, y_rot, np.pi / 2]),
              line([lane2_x, y_rot, np.pi / 2], [lane2_x, y_rot, mount[2]]),
              line([lane2_x, y_rot, mount[2]], preplace)]
    samples = np.vstack(pieces)
    path = PlanarPath(samples, rho=half_length, smooth=.18)
    l, P = path.sample(int(path.length / .05) + 1)

    # ------------------------------------------------------------------ clearance along the route (formation on level ground)
    profile = np.array([clear(p) for p in P])
    worst = int(np.argmin(profile))
    d, item = clear(P[worst], detail=True)
    say(f'route length {path.length:.2f} m; smallest clearance {d:.3f} m at l = {l[worst]:.2f} m ({item[1]} <-> {item[2]})')
    if d < c['clearance_margin']:
        raise RuntimeError('route: clearance below the declared margin')

    # ------------------------------------------------------------------ speed zones and gates
    half = site['beam_length'] / 2

    def ends(p):
        d_ = half * np.array([np.cos(p[2]), np.sin(p[2])])
        return p[:2] - d_, p[:2] + d_

    def span_y(p):
        a, b = ends(p)
        return min(a[1], b[1]), max(a[1], b[1])

    def zone(predicate, limit):
        inside = np.array([predicate(p) for p in P])
        if not inside.any():
            return None
        k = np.flatnonzero(inside)
        return [float(l[k[0]]), float(l[k[-1]]), limit]

    yd = site['door_y']
    turning = np.abs(np.gradient(P[:, 2], l)) * half_length > .05
    zones = [zone(lambda p: bool(turning[int(np.argmin(np.linalg.norm(P - p, axis=1)))]) and p[1] < site['y_ramp0'], c['turn_speed']),
             zone(lambda p: span_y(p)[0] - .4 < yd < span_y(p)[1] + .4 and p[2] > 1.2, c['door_speed']),
             zone(lambda p: span_y(p)[1] > site['y_ramp0'] - .3 and span_y(p)[0] < site['y_ramp1'] + .3 and p[1] < y_rot - .05,
                  c['ramp_speed']),
             zone(lambda p: p[1] >= y_rot - .05 and bool(turning[int(np.argmin(np.linalg.norm(P - p, axis=1)))]), c['rotate_speed']),
             zone(lambda p: np.linalg.norm(p[:2] - preplace[:2]) < 1.2 and abs(p[2] - mount[2]) < .02, c['final_speed'])]
    zones = [z for z in zones if z is not None]
    ob = comp.scene.get('obstacle')
    gates = []
    if ob:
        y_stop = ob['start'][1] - ob['half_size'][1] - c['stop_margin']
        l_stop = path.locate(lambda p: span_y(p)[1] >= y_stop and p[2] > 1.2)
        gates.append(dict(id='cart_crossing', obstacle=ob['id'], stop_l=l_stop - .01, leading_y=y_stop,
                          closed_while=dict(axis=0, entered=ob['start'][0] + .05, cleared=xc + w / 2 + ob['half_size'][0] + .3),
                          note='closed from the moment the cart leaves its start position until it has left the aisle'))

    # ------------------------------------------------------------------ approach, offsets, postures
    T_station, _ = tree.sweep(q_station)
    approach, travel = {}, {}
    for rid, r in comp.robots.items():
        con = comp.connections[f'grasp_{rid}']
        G = T_station[tree.index[con['body2']]] @ con['T2']                    # grasp frame (world)
        arm = (T_station[tree.index[r.frames['arm_base_link'][0]]] @ r.frames['arm_base_link'][1])[:3, 3]
        d_world = np.r_[arm[:2] - G[:2, 3], 0.]
        d_world *= shift / np.linalg.norm(d_world)
        back = G[:3, :3].T @ d_world                                            # towards the arm base, grasp axes
        up = G[:3, :3].T @ np.array([0., 0., c['approach_above']])
        approach[rid] = dict(above=up.tolist(), retracted=(up + back).tolist())
        travel[rid] = {j: float(q_carry[r.joint[j]]) for j in arm_joints(r)}
    starts = {rid: c['starts'][team_name][rid] for rid in comp.robots}
    order = sorted(comp.robots)

    # ------------------------------------------------------------------ approach: clearance of the folded robots on their way in
    from mission.timing import blend
    q_a = q_station.copy()
    for rid, r in comp.robots.items():
        for j in arm_joints(r):
            q_a[r.joint[j]] = travel[rid][j]
        if rid in lifts_carry:
            q_a[r.joint[lift_joint(r)]] = lifts_carry[rid]
    goal = {rid: compose(rack, station[rid]) for rid in comp.robots}
    via = {rid: compose(rack, station[rid] + np.array([0., np.sign(station[rid][1]) * c['dock_distance'], 0.])) for rid in comp.robots}
    T_app = {rid: max(1.875 * np.linalg.norm((via[rid] - np.asarray(starts[rid])) * [1., 1., .9]) / c['approach_speed'], 2.)
             for rid in comp.robots}
    T_dock = 1.875 * c['dock_distance'] / c['dock_speed']
    worst_app = (np.inf, None)
    t_end = max(T_app.values()) + .5
    for t in np.arange(0., t_end + T_dock + .25, .25):
        for rid, r in comp.robots.items():
            if t <= t_end:
                a, b, w = np.asarray(starts[rid], float), via[rid], blend(t, T_app[rid])[0]
            else:
                a, b, w = via[rid], goal[rid], blend(t - t_end, T_dock)[0]
            q_a[r.base[[0, 1, 3]]] = a + (b - a) * w
        cl = geo.clearance(tree.sweep(q_a)[0], cutoff=1.)
        for key, item in cl.items():
            if key.startswith('robot:') and key != 'robot:payload' and item[0] < worst_app[0]:
                worst_app = (item[0], (key, item[1], item[2], float(t)))
    say(f'approach: smallest clearance {worst_app[0]:.3f} m {worst_app[1]}')
    if worst_app[0] < c['clearance_margin']:
        raise RuntimeError('approach: clearance below the declared margin; revise the start poses')

    mission = dict(
        schema=SCHEMA_MISSION, team=team_name, name=f'Beam extraction, transport and installation ({team_name})',
        payload=dict(id=comp.payload['id'], half_length_m=half, mass_kg=float(comp.payload['record']['bodies'][0]['mass_kg'])),
        rests=dict(pickup='rack', place='mount'),
        formation=dict(station={k: v.tolist() for k, v in station.items()}, carry={k: v.tolist() for k, v in carry.items()},
                       extent=dict(lo=lo[:2].tolist(), hi=hi[:2].tolist(), top_m=top),
                       note='base planar pose [x, y, yaw] in the payload planar frame'),
        heights=dict(z_carry=c['z_carry'], raise_at_pickup=c['raise_at_rack'], place_above=c['place_above']),
        lifts=dict(station=lifts_station, carry=lifts_carry),
        starts=starts, travel_posture=travel, approach=approach, pull=shift,
        approach_clearance=dict(min_m=float(worst_app[0]), at=worst_app[1]),
        route=dict(rho=half_length, samples=P.tolist(), length_m=path.length, zones=zones, gates=gates,
                   cruise_speed=c['cruise_speed'], accel=c['accel'],
                   clearance=dict(min_m=float(d), at_l=float(l[worst]), pair=[item[1], item[2]],
                                  profile=[[float(a), float(b)] for a, b in zip(l[::4], profile[::4])]),
                   turn=dict(headings=turn[:, 2].tolist(), clearance=turn_clear.tolist())),
        durations=dict(settle=1.0, pregrasp=5.0, over=4.0, descend=4.0, latch=.6, take_weight=5.0, raise_=3.0, extract=5.0,
                       stow=6.0, present=6.0, insert=5.0, transfer=5.0, verify=1.0, release=.6, retract=4.0, fold=4.0,
                       retreat=5.0),
        approach_speed=c['approach_speed'], dock_distance=c['dock_distance'], dock_speed=c['dock_speed'], lower_speed=.012,
        release_order=order,
        load_sharing=dict(shares={f'grasp_{rid}': 1. for rid in comp.robots}, moment_scale_Nm=20., force_scale_N=300.,
                          note='weighted minimum-norm distribution; the internal wrench is defined relative to it'),
        thresholds=dict(carried_fraction_taken=.97, contact_drop_fraction=.12, released_fraction=.04,
                        settle_speed_m_s=.004, place_tolerance_m=.004, place_tolerance_rad=.004),
        bounds=dict(platform.get('bounds', BOUNDS)),
        declared='Task data designed by tools/build_mission.py from the physical records and the listed choices.',
        choices=c)
    if 'controller' in platform:
        mission['controller'] = dict(platform['controller'])
    return mission, dict(comp=comp, q_station=q_station, q_carry=q_carry, path=path, clear=clear, P=P, l=l, profile=profile)


def build_steered(team_name, comp, verbose=True):
    """Mission of a team whose bases cannot move sideways (conventional wheels on fixed axles)."""
    site = comp.scene['site']
    geo = Geometry(comp)
    tree = comp.tree
    model = TeamModel(comp)
    c = dict(CHOICES)
    platform = PLATFORM.get(team_name, {})
    c.update(platform.get('choices', {}))
    for key in platform.get('drop', ()):
        c.pop(key)
    say = print if verbose else (lambda *a, **k: None)
    robots = list(comp.robots)
    roles = ('wall', 'door', 'rack', 'neighbour', 'mount', 'guide', 'pad')

    # ------------------------------------------------------------------ station and carrying formation (in line, as the other teams)
    q_seed = comp.seed()
    asm = TeamAssembler(comp, q_seed)
    q_station, _ = asm.solve(q_seed)
    rack = payload_planar(q_station, comp)
    station = {rid: relative(rack, base_planar(q_station, r)) for rid, r in comp.robots.items()}
    if any(lift_joint(r) is not None for r in comp.robots.values()):
        raise RuntimeError('build_steered: robots with a commanded lift are not covered')
    shift = site['reach_station'] - c['reach_carry']
    carry = {rid: x + np.array([0., -np.sign(x[1]) * shift, 0.]) for rid, x in station.items()}
    carry_pose = np.array([rack[0], rack[1], c['z_carry'], rack[2], 0., 0.])
    q_carry = configure(comp, asm, q_station, carry_pose, carry, {})
    T_carry, _ = tree.sweep(q_carry)
    clear = FormationClearance(geo, T_carry, rack)
    placed = geo.place(T_carry)
    Pinv = planar_inverse(planar(*rack))
    corners = [(Pinv[:3, :3] @ np.array([[p.lo[0], p.lo[1], 0.], [p.hi[0], p.hi[1], 0.], [p.lo[0], p.hi[1], 0.], [p.hi[0], p.lo[1], 0.]]).T
                + Pinv[:3, 3:4]) for p in placed]
    lo, hi = np.min([x.min(axis=1) for x in corners], axis=0), np.max([x.max(axis=1) for x in corners], axis=0)
    half_length = float(max(abs(lo[0]), abs(hi[0])))
    mid = .5 * (lo[1] + hi[1])
    top = geo.top(T_carry)
    say(f'formation in line: x [{lo[0]:.2f}, {hi[0]:.2f}], y [{lo[1]:.2f}, {hi[1]:.2f}] (width {hi[1] - lo[1]:.2f} m), top {top:.2f} m')

    # ------------------------------------------------------------------ route of the payload
    xc, w = site['x_c'], site['aisle']
    x_w, y_n = xc - w / 2, site['aisle1_y'] + w / 2
    lane1_y = site['aisle1_y'] - mid
    lane2_x = xc + mid
    extracted = np.array([rack[0], rack[1] - shift, 0.])
    entry = np.array([x_w - 2.0, lane1_y, 0.])
    exit_ = np.array([lane2_x, y_n + 2.0, np.pi / 2])
    say('designing the aisle turn ...')
    turn, turn_clear = max_clearance_turn(clear, entry, exit_)
    if turn_clear.min() < c['clearance_margin']:
        say('  the local search closed the turn (clearance', np.round(turn_clear.min(), 3), 'm); scanning the corner at every heading ...')
        turn, turn_clear = bottleneck_turn(clear, entry, exit_)
        c['turn_search'] = 'scan of the corner on a 0.15 m grid at every heading (mission.route.bottleneck_turn)'
    say('  turn clearance by heading, bases in line (m):', np.round(turn_clear, 3))
    mount = np.array([site['mount'][0], site['mount'][1], site['mount_yaw']])
    preplace = compose(mount, [0., -shift, 0.])
    # The turn positions come from a search on a grid.  For the rigid formation the small steps between them do not
    # matter; a base that heads the way it moves would have to follow each of them, so the run through the turn is
    # smoothed over a longer length (the clearance is evaluated again below, on the route as it is driven).
    aisle = smooth_poses(np.vstack([line([rack[0] + c['lane_change'], lane1_y, 0.], entry), densify(turn, 240)[1:],
                                    line(exit_, [lane2_x, exit_[1] + 4., np.pi / 2])[1:]]), half_length, c['turn_smooth'])
    # On the plateau the payload has to turn from the heading of the aisle to that of the supports and to come to
    # them broadside, with every base facing them.  The base that leads (the one furthest ahead on the payload) leaves
    # the lane on a curve to its place in front of the supports and drives the last stretch straight.  While the last
    # base is still on the ramp the payload turns just as much as keeps that base on the lane, straight up the slope;
    # from there on its yaw is blended to that of the supports, and every base follows the point it holds.
    lead = max(robots, key=lambda rid: carry[rid][0])
    last = min(robots, key=lambda rid: carry[rid][0])
    facing = mount[2] + np.pi / 2                                 # the heading in which the bases face the supports
    lead_end = compose(preplace, carry[lead])[:2]
    lead_knee = lead_end - c['final_straight'] * unit(facing)
    y_clear = site['y_ramp1'] + .45                               # a base that heads north has left the ramp here
    y_lead = max(y_clear + carry[lead][0] - min(carry[rid][0] for rid in robots if rid != last),   # all but the last are clear
                 lead_knee[1] - c['swing_advance'])
    swing_from = np.array([lane2_x, y_lead - carry[lead][0], np.pi / 2])
    n_curve = 240
    track = np.vstack([hermite(compose(swing_from, carry[lead])[:2], np.pi / 2, lead_knee, facing, n_curve, c['swing_weights']),
                       lead_knee + (lead_end - lead_knee) * np.linspace(0., 1., 41)[1:, None]])
    apart = carry[lead][:2] - carry[last][:2]
    keep = yaw_on_line(track, apart, compose(swing_from, carry[last])[:2], np.pi / 2, np.pi / 2)
    last_y = track[:, 1] - (np.sin(keep) * apart[0] + np.cos(keep) * apart[1])
    if not (last_y[:n_curve] >= y_clear).any():
        raise RuntimeError('site: the last base is still on the ramp when the leading base reaches the supports')
    s_track = arc_length(track)
    free = int(np.argmax(last_y >= y_clear))
    blend_ = betainc(c['swing_shape'][0], c['swing_shape'][1],
                     np.clip((s_track - s_track[free]) / (s_track[n_curve - 1] - s_track[free]), 0., 1.))
    swing = led_poses(track, carry[lead][:2], (1. - blend_) * keep + blend_ * mount[2])
    pieces = [lane_change(extracted, [rack[0] + c['lane_change'], lane1_y, 0.]),
              aisle[1:],
              line(aisle[-1], swing_from)[1:],
              swing[1:]]
    path = PlanarPath(np.vstack(pieces), rho=half_length, smooth=.18)
    l, P = path.sample(int(path.length / .05) + 1)

    # ------------------------------------------------------------------ the bases under the payload
    # The origin of every base (where its arm stands) keeps its place on the payload.  A base on conventional wheels
    # turns about a point of its centre line that depends on its wheel loads: the skid model gives it for the team
    # as it stands at every point of the route.  The heading follows from that point (mission.steering.drawn_headings),
    # and the wheel loads from the heading, so the two are found in turn.
    offsets = {rid: carry[rid][:2] for rid in robots}
    headings0 = {rid: rack[2] + carry[rid][2] for rid in robots}
    bases, steering = follow_tangents(P, offsets, headings0)
    grasp_ids = sorted(cid for cid, x in comp.connections.items() if x['kind'] == 'grasp')
    carrying = model.mode(grasps=grasp_ids, rest=None)
    sharing = dict(shares={cid: 1. for cid in grasp_ids}, moment_scale_Nm=20., force_scale_N=300.)
    share = LoadSharing(model, grasp_ids, shares=sharing['shares'], moment_scale=sharing['moment_scale_Nm'],
                        force_scale=sharing['force_scale_N'])
    zero = np.zeros(tree.n)

    def stand(k, q):
        """The team at sample ``k`` of the route: bases on the terrain, payload at its height, arms closed on it."""
        ground = []
        for rid, r in comp.robots.items():
            idx, value = model.ground_pose(rid, bases[rid][k])
            q[r.base[[0, 1, 3]]] = bases[rid][k]
            q[idx] = value
            ground.append(value[0] - model.base_height[rid])
        q[comp.payload['base']] = [P[k, 0], P[k, 1], c['z_carry'] + max(ground), P[k, 2], 0., 0.]
        return asm.solve(q)[0]

    def turning_points(q):
        """Turning point of every base (ahead of its origin, m) and its wheel loads, for the team standing at ``q``."""
        kin = tree.sweep(q)
        con = model.constraints(q, carrying, kin)
        effort = tree.rnea(q, zero, zero, kinematics=kin)
        given = share.distribute(share.grasp_matrix(con), effort[model.payload])
        out = model.inverse(q, zero, zero, carrying, grasp_wrenches=given, kin=kin, constraints=con, effort=effort)
        result = {}
        for rid in robots:
            loads = model.wheel_loads(q, rid, out['ports'][f'{rid}:ground'], kin, con.info.get(f'{rid}:normal'))
            result[rid] = (float(model.skid(q, rid, kin, loads)['centre'][0]), loads)
        return result

    coarse = np.unique(np.r_[np.arange(0, len(P), 4), len(P) - 1])
    ahead = {}
    for _ in range(2):
        q, found = q_carry.copy(), {rid: [] for rid in robots}
        for k in coarse:
            q = stand(k, q)
            for rid, (x0, _) in turning_points(q).items():
                found[rid].append(x0)
        for rid in robots:
            ahead[rid] = np.interp(np.arange(len(P)), coarse, found[rid])
            bases[rid][:, 2] = drawn_headings(bases[rid][:, :2], ahead[rid], headings0[rid], facing)
    for rid in robots:
        info, B = steering[rid], bases[rid]
        step = np.linalg.norm(np.gradient(B[:, :2], axis=0), axis=1)
        moving = step > 1e-9
        info.update(max_turn_rad_per_m=float((np.abs(np.gradient(B[:, 2])[moving]) / step[moving]).max()),
                    relative_heading_rad=[float((B[:, 2] - P[:, 2]).min()), float((B[:, 2] - P[:, 2]).max())],
                    turning_point_ahead_m=[float(ahead[rid].min()), float(ahead[rid].max())])
        # at the two ends of the ramp a base without suspension has to stand square to them
        near = np.min(np.abs(B[:, 1:2] - [site['y_ramp0'], site['y_ramp1']]), axis=1) < .40
        info['heading_at_ramp_ends_rad'] = float(np.abs(B[near, 2] - np.pi / 2).max())
        say(f"  {rid}: path {info['path_m']:.2f} m, heading against the payload {np.degrees(info['relative_heading_rad'][0]):+.0f} to "
            f"{np.degrees(info['relative_heading_rad'][1]):+.0f} deg, tightest turn {1. / info['max_turn_rad_per_m']:.2f} m, "
            f"slowest step {info['slowest_step_ratio']:.2f} of its median, turning point {1e3 * info['turning_point_ahead_m'][0]:+.0f} "
            f"to {1e3 * info['turning_point_ahead_m'][1]:+.0f} mm ahead of the origin, "
            f"{np.degrees(info['heading_at_ramp_ends_rad']):.2f} deg off square at the ends of the ramp")
        if info['slowest_step_ratio'] < .15 or info['max_turn_rad_per_m'] * c['min_turn_radius'] > 1.:
            raise RuntimeError(f'route: base {rid} would have to turn more tightly than the declared radius; revise the route')
        if info['heading_at_ramp_ends_rad'] > c['crease_heading']:
            raise RuntimeError(f'route: base {rid} does not cross the ends of the ramp squarely; revise the route')
        if abs(B[0, 2] - headings0[rid]) > 1e-6 or abs(B[-1, 2] - facing) > 2e-3:
            raise RuntimeError(f'route: base {rid} cannot start or end in its station heading; the route must start and end straight')
        B[-1, 2] = facing

    # ------------------------------------------------------------------ clearance along the route: the team as it really stands
    q = q_carry.copy()
    profile, worst = np.zeros(len(P)), (np.inf, None)
    for k in range(len(P)):
        q = stand(k, q)
        items = geo.against_statics(geo.place(tree.sweep(q)[0]), 1., skip=lambda a, b: b.role not in roles)
        best = min(items, key=lambda item: item[0]) if items else (1., None, None)
        profile[k] = best[0]
        if best[0] < worst[0]:
            worst = (best[0], best, float(l[k]))
    d, item = worst[0], worst[1]
    say(f'route length {path.length:.2f} m; smallest clearance {d:.3f} m at l = {worst[2]:.2f} m ({item[1]} <-> {item[2]})')
    if d < c['clearance_margin']:
        raise RuntimeError('route: clearance below the declared margin')
    q_place = q.copy()                                            # the team at the end of the route

    # ------------------------------------------------------------------ speed zones and gates
    half = site['beam_length'] / 2

    def ends(p):
        d_ = half * np.array([np.cos(p[2]), np.sin(p[2])])
        return p[:2] - d_, p[:2] + d_

    def span_y(p):
        a, b = ends(p)
        return min(a[1], b[1]), max(a[1], b[1])

    def zone(predicate, limit):
        inside = np.array([predicate(p) for p in P])
        if not inside.any():
            return None
        k = np.flatnonzero(inside)
        return [float(l[k[0]]), float(l[k[-1]]), limit]

    yd = site['door_y']
    turning = np.abs(np.gradient(P[:, 2], l)) * half_length > .05
    zones = [zone(lambda p: bool(turning[int(np.argmin(np.linalg.norm(P - p, axis=1)))]) and p[1] < site['y_ramp0'], c['turn_speed']),
             zone(lambda p: span_y(p)[0] - .4 < yd < span_y(p)[1] + .4 and p[2] > 1.2, c['door_speed']),
             zone(lambda p: span_y(p)[1] > site['y_ramp0'] - .3 and span_y(p)[0] < site['y_ramp1'] + .3 and p[1] < swing_from[1] - .05,
                  c['ramp_speed']),
             zone(lambda p: p[1] >= swing_from[1] - .05 and bool(turning[int(np.argmin(np.linalg.norm(P - p, axis=1)))]), c['rotate_speed']),
             zone(lambda p: np.linalg.norm(p[:2] - preplace[:2]) < 1.2 and abs(p[2] - mount[2]) < .02, c['final_speed'])]
    zones = [z for z in zones if z is not None]
    ob = comp.scene.get('obstacle')
    gates = []
    if ob:
        y_stop = ob['start'][1] - ob['half_size'][1] - c['stop_margin']
        l_stop = path.locate(lambda p: span_y(p)[1] >= y_stop and p[2] > 1.2)
        gates.append(dict(id='cart_crossing', obstacle=ob['id'], stop_l=l_stop - .01, leading_y=y_stop,
                          closed_while=dict(axis=0, entered=ob['start'][0] + .05, cleared=xc + w / 2 + ob['half_size'][0] + .3),
                          note='closed from the moment the cart leaves its start position until it has left the aisle'))

    # ------------------------------------------------------------------ arm offsets and postures
    T_station, _ = tree.sweep(q_station)
    approach, travel = {}, {}
    for rid, r in comp.robots.items():
        con = comp.connections[f'grasp_{rid}']
        G = T_station[tree.index[con['body2']]] @ con['T2']                    # grasp frame (world)
        arm = (T_station[tree.index[r.frames['arm_base_link'][0]]] @ r.frames['arm_base_link'][1])[:3, 3]
        d_world = np.r_[arm[:2] - G[:2, 3], 0.]
        d_world *= shift / np.linalg.norm(d_world)
        back = G[:3, :3].T @ d_world                                            # towards the arm base, grasp axes
        up = G[:3, :3].T @ np.array([0., 0., c['approach_above']])
        approach[rid] = dict(above=up.tolist(), retracted=(up + back).tolist())
        travel[rid] = {j: float(q_carry[r.joint[j]]) for j in arm_joints(r)}
    starts = {rid: c['starts'][team_name][rid] for rid in comp.robots}

    # ------------------------------------------------------------------ the way into the stations
    # A turn on the spot towards the first leg (about the turning centre the skid model gives the folded robot),
    # a straight leg and a curve into the lane along the rack, then straight along the rack into the station.
    q_a = q_station.copy()
    for rid, r in comp.robots.items():
        for j in arm_joints(r):
            q_a[r.joint[j]] = travel[rid][j]
    rho_b = c['rotation_length']
    goal = {rid: compose(rack, station[rid]) for rid in comp.robots}
    # the points at which the bases enter the lane lie one behind the other, far enough apart for their curves
    dock, ahead = {}, None
    for rid in sorted(robots, key=lambda k: -station[k][0]):
        dock[rid] = c['dock_distance'] if ahead is None else max(
            c['dock_distance'], c['via_spacing'] - (station[ahead][0] - dock[ahead]) + station[rid][0])
        ahead = rid
    paths_in, paths_dock, centres = {}, {}, {}
    for rid, r in comp.robots.items():
        start = np.asarray(starts[rid], float)
        q_s = q_a.copy()
        q_s[r.base[[0, 1, 3]]] = start
        x0 = float(model.skid(q_s, rid)['centre'][0])                           # folded arm, own weight
        via = goal[rid] - dock[rid] * np.r_[unit(goal[rid][2]), 0.]
        first = start.copy()
        for _ in range(8):                                                       # the turn moves the origin a little: iterate
            drive = line_and_arc(first[:2], via, c['arc_radius'])
            spin = turn_in_place(start, start[2] + (drive[0, 2] - start[2] + np.pi) % (2. * np.pi) - np.pi, x0)
            if np.linalg.norm(spin[-1, :2] - first[:2]) < 1e-9:
                break
            first = spin[-1].copy()
        # the heading along the way: the base turns about the point x0 of its centre line, not about its origin
        n_dock = max(int(np.ceil(dock[rid] / .02)), 1)
        way = np.vstack([drive[:, :2], via[:2] + (goal[rid][:2] - via[:2]) * np.linspace(0., 1., n_dock + 1)[1:, None]])
        way = np.c_[way, drawn_headings(way, x0, spin[-1, 2])]
        n_in = len(drive)
        if abs(way[-1, 2] - goal[rid][2]) > 2e-3:
            raise RuntimeError(f'approach: {rid} does not reach its station in the station heading; revise its start pose')
        way[-1, 2] = goal[rid][2]
        thin = lambda x: x[::2].tolist() + ([x[-1].tolist()] if (len(x) - 1) % 2 else [])
        paths_in[rid] = [dict(samples=spin.tolist(), rho=rho_b, speed=c['turn_rate'] * rho_b),
                         dict(samples=thin(way[:n_in]), rho=rho_b, speed=c['approach_speed'])]
        paths_dock[rid] = [dict(samples=thin(way[n_in - 1:]), rho=rho_b, speed=c['dock_speed'])]
        centres[rid] = x0
        say(f"  {rid}: turn on the spot {np.degrees(spin[-1, 2] - start[2]):+.0f} deg about a point {x0 * 1e3:+.0f} mm ahead of its "
            f"origin, then {arc_length(drive[:, :2])[-1]:.2f} m to the lane and {dock[rid]:.2f} m along it")

    # clearance of the folded robots on their way in (the time law of mission.controller._follow)
    from mission.timing import blend

    def schedule(segments):
        t0, out = 0., []
        for seg in segments:
            sp = PlanarPath(np.asarray(seg['samples'], float), seg['rho'])
            T = max(1.875 * sp.length / seg['speed'], 1.)
            out.append((t0, T, sp))
            t0 += T + .3
        return out, t0 - .3

    timed = {rid: schedule(paths_in[rid]) for rid in comp.robots}
    docks = {rid: schedule(paths_dock[rid]) for rid in comp.robots}
    t_in = max(v[1] for v in timed.values()) + .5
    t_dock = max(v[1] for v in docks.values()) + .5

    def pose_at(sched, t):
        t0, T, sp = next((x for x in sched if t < x[0] + x[1]), sched[-1])
        return sp(sp.length * blend(t - t0, T)[0])[0]

    worst_app = (np.inf, None)
    for t in np.arange(0., t_in + t_dock + .25, .25):
        for rid, r in comp.robots.items():
            q_a[r.base[[0, 1, 3]]] = pose_at(timed[rid][0], t) if t <= t_in else pose_at(docks[rid][0], t - t_in)
        cl = geo.clearance(tree.sweep(q_a)[0], cutoff=1.)
        for key, item_ in cl.items():
            if key.startswith('robot:') and key != 'robot:payload' and item_[0] < worst_app[0]:
                worst_app = (item_[0], (key, item_[1], item_[2], float(t)))
    say(f'approach: smallest clearance {worst_app[0]:.3f} m {worst_app[1]}')
    if worst_app[0] < c['clearance_margin']:
        raise RuntimeError('approach: clearance below the declared margin; revise the start poses')

    # ------------------------------------------------------------------ the stations in front of the supports
    place = {rid: relative(mount, bases[rid][-1]) for rid in robots}
    mission = dict(
        schema=SCHEMA_MISSION, team=team_name, name=f'Beam extraction, transport and installation ({team_name})',
        payload=dict(id=comp.payload['id'], half_length_m=half, mass_kg=float(comp.payload['record']['bodies'][0]['mass_kg'])),
        rests=dict(pickup='rack', place='mount'),
        formation=dict(station={k: v.tolist() for k, v in station.items()}, carry={k: v.tolist() for k, v in carry.items()},
                       place={k: v.tolist() for k, v in place.items()},
                       extent=dict(lo=lo[:2].tolist(), hi=hi[:2].tolist(), top_m=top),
                       note='base planar pose [x, y, yaw] in the payload planar frame: at the rack, in line while carrying, '
                            'and in front of the supports (in their frame). On the route every base has its own poses '
                            '(route.bases).'),
        heights=dict(z_carry=c['z_carry'], raise_at_pickup=c['raise_at_rack'], place_above=c['place_above']),
        lifts=dict(station={}, carry={}),
        starts=starts, travel_posture=travel, approach=approach, pull=shift,
        approach_clearance=dict(min_m=float(worst_app[0]), at=worst_app[1]),
        base_paths=dict(approach=paths_in, dock=paths_dock,
                        retreat=dict(distance=c['retreat'], speed=c['retreat_speed'], rho=rho_b),
                        turning_centre_m=centres, dock_distance_m=dock,
                        note='pose paths [x, y, yaw] of the base origins: a turn on the spot about the turning point of the '
                             'skid model, a straight leg with a curve into the lane along the rack, and the straight run '
                             'into the station; each from rest to rest. The heading is that of a base that turns about its '
                             'turning point. rho: length a turn counts for, per radian'),
        route=dict(rho=half_length, samples=P.tolist(), length_m=path.length, zones=zones, gates=gates,
                   cruise_speed=c['cruise_speed'], accel=c['accel'],
                   bases={rid: bases[rid].tolist() for rid in robots}, base_yaw_rate=c['base_yaw_rate'],
                   steering=dict(by_base=steering, leading=lead,
                                 note='the origin of every base keeps the planar offset formation.carry[:2] from the '
                                      'payload; the base turns about the point of its centre line that the skid model gives '
                                      'for its wheel loads (turning_point_ahead_m), and its heading follows from that'),
                   clearance=dict(min_m=float(d), at_l=float(worst[2]), pair=[item[1], item[2]],
                                  profile=[[float(a), float(b)] for a, b in zip(l[::4], profile[::4])]),
                   turn=dict(headings=turn[:, 2].tolist(), clearance=turn_clear.tolist())),
        durations=dict(settle=1.0, pregrasp=5.0, over=4.0, descend=4.0, latch=.6, take_weight=5.0, raise_=3.0, extract=5.0,
                       stow=6.0, present=6.0, insert=5.0, transfer=5.0, verify=1.0, release=.6, retract=4.0, fold=4.0,
                       retreat=5.0),
        lower_speed=.012,
        release_order=sorted(comp.robots),
        load_sharing=dict(sharing, note='weighted minimum-norm distribution; the internal wrench is defined relative to it'),
        thresholds=dict(carried_fraction_taken=.97, contact_drop_fraction=.12, released_fraction=.04,
                        settle_speed_m_s=.004, place_tolerance_m=.004, place_tolerance_rad=.004),
        bounds=dict(platform.get('bounds', BOUNDS)),
        declared='Task data designed by tools/build_mission.py from the physical records and the listed choices.',
        choices=c)
    if 'controller' in platform:
        mission['controller'] = dict(platform['controller'])
    return mission, dict(comp=comp, q_station=q_station, q_carry=q_carry, q_place=q_place, path=path, clear=clear, P=P, l=l,
                         profile=profile, bases=bases)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'team2'
    t0 = time.perf_counter()
    mission, _ = build(name)
    path = write_json(INPUTS / 'missions' / f'{name}.json', mission)
    print('wrote', path.relative_to(ROOT), f"({time.perf_counter() - t0:.1f} s)")
    print('zones', [[round(a, 2), round(b, 2), v] for a, b, v in mission['route']['zones']])
    print('gates', [(g['id'], round(g['stop_l'], 2)) for g in mission['route']['gates']])


if __name__ == '__main__':
    main()
