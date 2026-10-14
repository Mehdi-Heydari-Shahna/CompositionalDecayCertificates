"""Team graph compiler: robot graphs + one payload body + connections -> compiled team graph.

New extension; the PACDM kernel is used unchanged.

Input (physical records only):

* robot records (``cmg.physical-mobile-manipulator/1.0``): bodies with inertias,
  *undirected* joints, a floating root, named frames, actuator and wheel
  declarations, drive data and a named seed;
* one payload record (``cmg.physical-payload/1.0``): bodies, floating root, named
  frames;
* a scene record (``cmg.physical-scene/1.0``): world-fixed frames, terrain;
* a team record (``cmg.physical-team-graph/1.0``): which robots, which coordinates
  are commanded, grasp connections (tool frame <-> payload frame, joint type) and
  rest connections (payload frame <-> scene frame).

Not supplied and rejected if present: tree directions, coordinate orders,
connection paths, Jacobians, modules, partitions.

Generated here: one world-rooted tree (a six-coordinate chart for every floating
root, as in the Go2 and Kangaroo compilers), the coordinate order, one chord per
connection with its two root paths, the per-robot closure modules and their
dependent sets, actuator and reaction port tables, and the mode table.

Scope: robots that are trees of fixed, revolute and prismatic joints on one
floating root; a single rigid payload; connections of the types listed in
``CONNECTION_FREEDOMS``.  Robot-internal loops are not compiled here.

Wheels may sit on the floating root or on bodies below it.  A moving joint
without an actuator that is not a wheel is a passive joint; it is accepted
where it carries wheels (a rocker, a suspension link).  The ground model of a
base follows from its mechanism:

* ``contact``: one ideal contact row per wheel.  It needs as many wheels as the
  wheel contacts can determine, three plus the number of passive joints, so that
  every wheel load is statically determinate;
* ``fit``: a base without passive joints follows a plane fitted to the ground
  under its wheels (three rows); with more than three wheels the loads of the
  individual wheels are then not determined by rigid-body mechanics.

How the wheels drive a base also follows from the mechanism:

* ``rolling``: mecanum wheels.  Every wheel has one ideal rolling row, and the
  base can move in every planar direction without slip;
* ``friction``: conventional wheels on fixed axles that do not all lie on one
  line.  Such a base cannot turn by rolling; its wheels slide (skid steering).
  Their tangential ground forces follow a friction law (``teamc/skid.py``) and
  are not constraint reactions.  The test is the rank of the rows that keep the
  ground points of the wheels from moving along their axles, taken on the
  planar twist of the base: one line of axles (rank one) leaves the base free
  to drive and to turn, more than one does not.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from . import SCHEMA_COMPILED, SCHEMA_PAYLOAD, SCHEMA_ROBOT, SCHEMA_SCENE, SCHEMA_TEAM
from .se3 import chart_pose, inv, rigid
from .tree import Tree

CHART = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
CHART_AXES = [[1., 0, 0], [0, 1., 0], [0, 0, 1.], [0, 0, 1.], [0, 1., 0], [1., 0, 0]]
CHART_LIMIT = [1e6, 1e6, 1e6, 1e6, 1.45, np.pi]
FORBIDDEN = {'tree', 'coordinate_ids', 'independent_ids', 'dependent_ids', 'modules', 'paths', 'closures',
             'jacobian', 'chords', 'cuts', 'partitions', 'rows', 'sparsity'}

# Free relative twists of a connection, in the connection frame on its first body
# (angular-first unit twists).  Rows left constrained are the reaction components.
RX, RY, RZ, TX, TY, TZ = np.eye(6)
CONNECTION_FREEDOMS = {
    'weld': [],
    'revolute_x': [RX], 'revolute_y': [RY], 'revolute_z': [RZ],
    'prismatic_x': [TX], 'prismatic_y': [TY], 'prismatic_z': [TZ],
    'cylindrical_x': [RX, TX],
    'universal_xy': [RX, RY],
    'spherical': [RX, RY, RZ],
}


class ModelError(ValueError):
    pass


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _check_inertia(b, name):
    m = float(b['mass_kg'])
    c = np.asarray(b['com_m'], float)
    I = np.asarray(b['inertia_kg_m2'], float)
    if not np.isfinite(m) or m <= 0 or c.shape != (3,) or I.shape != (3, 3) or not np.all(np.isfinite(np.r_[c, I.ravel()])):
        raise ModelError(f'{name}: finite positive mass and finite COM/inertia required')
    eig = np.linalg.eigvalsh(.5 * (I + I.T))
    if not np.allclose(I, I.T, rtol=0, atol=1e-9) or eig.min() <= 0 or eig.max() > eig.sum() - eig.max() + 1e-9:
        raise ModelError(f'{name}: inertia must be symmetric positive definite with physical principal moments')


def direct_graph(record, root_key, label):
    """Directed tree of one physical record by depth-first search from its floating root.

    Returns directed joints (``base_body``, ``follower_body``, ``T_BJ``, ``T_FJ``,
    axis sign-corrected) in deterministic preorder, and the joint path to every body.
    """
    forbidden = FORBIDDEN & set(record)
    if forbidden:
        raise ModelError(f'{label}: input must not pre-author computational structure: {sorted(forbidden)}')
    bodies = record.get('bodies', [])
    bmap = {b['id']: b for b in bodies}
    root = record.get(root_key)
    if not bodies or len(bmap) != len(bodies) or root not in bmap:
        raise ModelError(f'{label}: unique bodies and a valid {root_key} are required')
    for b in bodies:
        _check_inertia(b, f"{label}/{b['id']}")
    joints = record.get('joints', [])
    if len({j['id'] for j in joints}) != len(joints):
        raise ModelError(f'{label}: unique joint ids required')
    if len(joints) != len(bodies) - 1:
        raise ModelError(f'{label}: joints must form a tree over the bodies (internal loops are not compiled here)')
    adjacency = {b: [] for b in bmap}
    for j in joints:
        a, b = j['body_a'], j['body_b']
        if a not in bmap or b not in bmap or a == b:
            raise ModelError(f"{label}/{j['id']}: invalid joint endpoints")
        if j['type'] not in ('fixed', 'revolute', 'prismatic'):
            raise ModelError(f"{label}/{j['id']}: unsupported joint type {j['type']}")
        rigid(j['T_AJ'], j['id'] + '.T_AJ')
        rigid(j['T_BJ'], j['id'] + '.T_BJ')
        if j['type'] != 'fixed':
            axis = np.asarray(j['axis'], float)
            if axis.shape != (3,) or abs(np.linalg.norm(axis) - 1.) > 1e-9:
                raise ModelError(f"{label}/{j['id']}: axis must be a unit vector")
        adjacency[a].append((j, b, 1))
        adjacency[b].append((j, a, -1))
    ordered, paths, seen = [], {root: []}, {root}

    def walk(body):
        for j, child, direction in sorted(adjacency[body], key=lambda item: item[0]['id']):
            if child in seen:       # the edge we came through; a tree has no other back edge
                continue
            seen.add(child)
            e = dict(id=j['id'], type=j['type'], base_body=body, follower_body=child,
                     T_BJ=deepcopy(j['T_AJ'] if direction == 1 else j['T_BJ']),
                     T_FJ=deepcopy(j['T_BJ'] if direction == 1 else j['T_AJ']))
            if j['type'] != 'fixed':
                e['axis'] = (direction * np.asarray(j['axis'], float)).tolist()
            ordered.append(e)
            paths[child] = paths[body] + [j['id']]
            walk(child)

    walk(root)
    if seen != set(bmap):
        raise ModelError(f'{label}: disconnected graph')
    return ordered, paths


@dataclass
class Robot:
    """Index view of one robot inside the compiled team graph."""
    id: str
    prefix: str
    base: np.ndarray                    # six chart coordinate indices
    joint: dict                         # local joint id -> coordinate index
    commanded: list                     # local joint ids declared as inputs
    tool_path: list                     # local moving joint ids on the root -> tool path
    dependent: list                     # local joint ids solved by a grasp closure
    actuators: list                     # dict(id, joint_id, index, gear, bounds, unit, velocity_limit)
    wheels: list                        # wheel records with coordinate index
    frames: dict                        # frame name -> (global body name, T_body_frame)
    record: dict = field(repr=False, default=None)
    passive: list = field(default_factory=list)     # local ids of unactuated joints that carry wheels
    ground: str = 'fit'                 # ground model: 'fit' (plane under the wheels) or 'contact' (one row per wheel)
    drive: str = 'rolling'              # 'rolling' (ideal rolling rows) or 'friction' (wheel forces from the friction law)

    def idx(self, joint_ids):
        return np.array([self.joint[j] for j in joint_ids], int)


@dataclass
class TeamCompilation:
    cmg: dict
    plan: dict
    source: dict
    tree: Tree
    robots: dict
    payload: dict
    connections: dict
    scene: dict

    def robot_ids(self):
        return list(self.robots)

    def seed(self):
        return np.asarray(self.cmg['q_seed'], float).copy()

    def coordinate(self, name):
        return self.cmg['coordinate_ids'].index(name)

    def frame(self, owner, name):
        """Global body name and body-fixed transform of a named frame of a robot or of the payload."""
        if owner in self.robots:
            return self.robots[owner].frames[name]
        if owner == self.payload['id']:
            return self.payload['frames'][name]
        raise KeyError(owner)


def _chart(prefix, child, seed):
    """Generated chart frames and scalar joints carrying a floating root (world XYZ, intrinsic ZYX)."""
    bodies, joints, parent = [], [], 'world'
    for i, (name, axis, limit) in enumerate(zip(CHART, CHART_AXES, CHART_LIMIT)):
        follower = child if i == 5 else f'{prefix}chart_{i}'
        if i < 5:
            bodies.append(dict(id=follower, kind='generated_frame', mass_kg=0., com_m=[0., 0., 0.],
                               inertia_kg_m2=np.zeros((3, 3)).tolist()))
        joints.append(dict(id=f'{prefix}base_{name}', type='prismatic' if i < 3 else 'revolute',
                           base_body=parent, follower_body=follower, axis=list(axis), T_BJ=np.eye(4).tolist(),
                           T_FJ=np.eye(4).tolist(), limits=dict(lower=-limit, upper=limit), generated=True))
        parent = follower
    return bodies, joints, [f'{prefix}base_{name}' for name in CHART], list(map(float, seed))


def compile_team(team, robots, payload, scene):
    """Join the physical records into one compiled team graph.

    ``robots`` maps a model name to its physical robot record.
    """
    team, payload, scene = deepcopy(team), deepcopy(payload), deepcopy(scene)
    robots = {k: deepcopy(v) for k, v in robots.items()}
    for record, schema, label in ((team, SCHEMA_TEAM, 'team'), (payload, SCHEMA_PAYLOAD, 'payload'),
                                  (scene, SCHEMA_SCENE, 'scene')):
        if record.get('schema') != schema:
            raise ModelError(f'{label}: expected schema {schema}')
        bad = FORBIDDEN & set(record)
        if bad:
            raise ModelError(f'{label}: input must not pre-author computational structure: {sorted(bad)}')
    members = team.get('robots', [])
    if len({m['id'] for m in members}) != len(members) or not members:
        raise ModelError('team: distinct robot ids required')
    gravity = team.get('gravity_m_s2', [0., 0., -9.81])

    bodies = [dict(id='world', kind='reference_frame', mass_kg=0., com_m=[0., 0., 0.],
                   inertia_kg_m2=np.zeros((3, 3)).tolist())]
    joints, ids, seed, armature, damping, limits = [], [], [], {}, {}, {}
    views, ports, structure = {}, [], {}

    def add_tree(prefix, record, root, chart_seed, joint_seed, label):
        directed, paths = direct_graph(record, 'floating_root' if 'floating_root' in record else 'root', label)
        cb, cj, cids, cseed = _chart(prefix, prefix + root, chart_seed)
        bodies.extend(cb)
        joints.extend(cj)
        ids.extend(cids)
        seed.extend(cseed)
        for j in cj:
            limits[j['id']] = j['limits']
        for b in record['bodies']:
            e = {k: deepcopy(b[k]) for k in ('mass_kg', 'com_m', 'inertia_kg_m2')}
            e.update(id=prefix + b['id'], kind='rigid_body')
            bodies.append(e)
        local = {}
        declared = record.get('limits', {})
        for j in directed:
            e = deepcopy(j)
            e.update(id=prefix + j['id'], base_body=prefix + j['base_body'], follower_body=prefix + j['follower_body'])
            if j['type'] != 'fixed':
                if j['id'] not in declared:
                    raise ModelError(f"{label}/{j['id']}: joint limits required")
                lo, hi = float(declared[j['id']]['lower']), float(declared[j['id']]['upper'])
                value = float(joint_seed.get(j['id'], np.nan))
                if not np.isfinite(value) or not lo <= value <= hi:
                    raise ModelError(f"{label}/{j['id']}: named seed missing or outside the declared limits")
                e['limits'] = dict(lower=lo, upper=hi)
                limits[e['id']] = e['limits']
                local[j['id']] = len(ids)
                ids.append(e['id'])
                seed.append(value)
                drive = record.get('drive', {}).get(j['id'], {})
                armature[e['id']] = float(drive.get('armature', 0.))
                damping[e['id']] = float(drive.get('damping', 0.))
            joints.append(e)
        if set(joint_seed) - set(local):
            raise ModelError(f'{label}: seed names unknown joints {sorted(set(joint_seed) - set(local))}')
        frames = {name: (prefix + f['body'], rigid(f['T_body_frame'], name)) for name, f in record.get('frames', {}).items()}
        for b in record['bodies']:
            frames.setdefault(b['id'], (prefix + b['id'], np.eye(4)))
        return directed, paths, local, frames

    seeds = team.get('seed', {})
    for member in members:
        rid, model = member['id'], member['model']
        if model not in robots:
            raise ModelError(f'team: unknown robot model {model}')
        record = robots[model]
        if record.get('schema') != SCHEMA_ROBOT:
            raise ModelError(f'{model}: expected schema {SCHEMA_ROBOT}')
        prefix = rid + '_'
        s = seeds.get(rid, {})
        base_seed = s.get('base')
        if base_seed is None or len(base_seed) != 6:
            raise ModelError(f'team: seed for {rid} needs a six-coordinate base chart value')
        joint_seed = dict(record.get('seed', {}).get('joints', {}))
        joint_seed.update(s.get('joints', {}))
        first = len(ids)
        directed, paths, local, frames = add_tree(prefix, record, record['floating_root'], base_seed, joint_seed, rid)
        base = np.arange(first, first + 6)
        acts = []
        for a in sorted(record.get('actuators', []), key=lambda a: a['id']):
            if a['joint_id'] not in local:
                raise ModelError(f"{rid}/{a['id']}: actuator must drive a moving joint")
            lo, hi = map(float, a['effort_bounds'])
            if not lo < hi or not np.isfinite(float(a['gear'])) or float(a['gear']) == 0.:
                raise ModelError(f"{rid}/{a['id']}: invalid actuator gear or bounds")
            acts.append(dict(id=prefix + a['id'], joint_id=a['joint_id'], index=local[a['joint_id']],
                             gear=float(a['gear']), bounds=[lo, hi], unit=a.get('unit', ''),
                             velocity_limit=float(a.get('velocity_limit', np.inf))))
        if len({a['joint_id'] for a in acts}) != len(acts):
            raise ModelError(f'{rid}: at most one actuator per joint')
        wheels = []
        by_id = {j['id']: j for j in directed}
        root = record['floating_root']

        def nominal(body):
            """Pose of a body in the root frame with every joint between them at zero."""
            T = np.eye(4)
            for jid in paths[body]:
                T = T @ np.asarray(by_id[jid]['T_BJ'], float) @ inv(np.asarray(by_id[jid]['T_FJ'], float))
            return T

        for w in sorted(record.get('wheels', []), key=lambda w: w['id']):
            if w['joint_id'] not in local or w['type'] not in ('mecanum', 'fixed'):
                raise ModelError(f"{rid}/{w['id']}: wheel needs a moving joint and a supported type")
            joint = by_id[w['joint_id']]
            if joint['follower_body'] != w['body']:
                raise ModelError(f"{rid}/{w['id']}: the wheel body must be the follower of the wheel joint")
            carrier = joint['base_body']
            T_BJ = np.asarray(joint['T_BJ'], float)
            e = deepcopy(w)
            e.update(index=local[w['joint_id']], carrier=prefix + carrier, centre_carrier=T_BJ[:3, 3].copy(),
                     axis_carrier=T_BJ[:3, :3] @ np.asarray(joint['axis'], float),
                     suspension=[j for j in paths[carrier] if j in local])
            if carrier == root:
                e.update(centre_base=T_BJ[:3, 3].copy(), axis_base=T_BJ[:3, :3] @ np.asarray(joint['axis'], float))
            else:                                      # geometry in the root frame with the joints above the wheel at zero
                Tc = nominal(carrier)
                e.update(centre_base=Tc[:3, :3] @ T_BJ[:3, 3] + Tc[:3, 3],
                         axis_base=Tc[:3, :3] @ (T_BJ[:3, :3] @ np.asarray(joint['axis'], float)))
            if 'traction_direction_base' in e:
                e['traction_direction_base'] = np.asarray(e['traction_direction_base'], float)
                e['traction_direction_carrier'] = e['traction_direction_base'] if carrier == root \
                    else nominal(carrier)[:3, :3].T @ e['traction_direction_base']
            elif w['type'] == 'mecanum':
                raise ModelError(f"{rid}/{w['id']}: a mecanum wheel needs its traction direction")
            wheels.append(e)
        # unactuated joints that are not wheels: accepted where they carry wheels
        driven = {a['joint_id'] for a in acts} | {w['joint_id'] for w in wheels}
        passive = [j for j in local if j not in driven]
        carrying = {j for w in wheels for j in w['suspension']}
        if set(passive) - carrying:
            raise ModelError(f'{rid}: unactuated joints {sorted(set(passive) - carrying)} carry no wheel')
        if any(len(w['suspension']) > 0 and set(w['suspension']) - set(passive) for w in wheels):
            raise ModelError(f'{rid}: actuated joints between the floating root and a wheel are not supported')
        ground = 'fit'
        if passive:
            ground = 'contact'
            if len(wheels) != 3 + len(passive):
                raise ModelError(f'{rid}: {len(wheels)} wheels and {len(passive)} passive joints: the wheel contacts are '
                                 'not statically determinate (three wheels plus one per passive joint are needed)')
        elif wheels and len(wheels) < 3:
            raise ModelError(f'{rid}: a base needs at least three wheels')
        # how the wheels drive the base
        kinds = {w['type'] for w in wheels}
        if len(kinds) > 1:
            raise ModelError(f'{rid}: mecanum and conventional wheels on one base are not supported')
        drive = 'rolling'
        held = [w for w in wheels if w['type'] == 'fixed' and w.get('lateral', 'constraint') == 'constraint']
        if held:
            # rows of "no motion of the ground point along the axle" on the planar twist [vx, vy, wz] of the base
            rows = []
            for w in held:
                e, c = w['axis_base'][:2] / np.linalg.norm(w['axis_base'][:2]), w['centre_base']
                rows.append([e[0], e[1], e[1] * c[0] - e[0] * c[1]])
            if np.linalg.matrix_rank(np.array(rows), tol=1e-9) > 1:
                drive = 'friction'
                if len(held) != len(wheels):
                    raise ModelError(f'{rid}: a base that turns by skidding needs the axle row of every wheel')
                if passive:
                    raise ModelError(f'{rid}: conventional wheels on passive joints are not supported')
        commanded = list(member.get('commanded', []))
        tool = record.get('ports', {}).get('tool', {}).get('frame')
        if tool not in frames:
            raise ModelError(f'{rid}: a tool port frame is required')
        tool_body = frames[tool][0][len(prefix):]
        tool_path = [j for j in paths[tool_body] if j in local]
        if any(c not in local for c in commanded) or len(set(commanded)) != len(commanded):
            raise ModelError(f'{rid}: commanded coordinates must be distinct moving joints')
        dependent = [j for j in tool_path if j not in commanded]
        if set(passive) & set(tool_path):
            raise ModelError(f'{rid}: a passive joint on the tool path is not supported')
        if set(commanded) - set(tool_path):
            raise ModelError(f'{rid}: commanded coordinates must lie on the tool path')
        views[rid] = Robot(id=rid, prefix=prefix, base=base, joint=local, commanded=commanded, tool_path=tool_path,
                           dependent=dependent, actuators=acts, wheels=wheels, frames=frames, record=record,
                           passive=passive, ground=ground, drive=drive)
        structure[rid] = dict(model=model, base_coordinates=base.tolist(), coordinates={k: int(v) for k, v in local.items()},
                              tool_path=tool_path, commanded=commanded, grasp_dependent=dependent,
                              actuators=[a['id'] for a in acts], wheels=[w['id'] for w in wheels])
        if passive:
            structure[rid].update(passive=passive, ground_model=ground)
        if drive != 'rolling':
            structure[rid].update(drive=drive)
        for a in acts:
            ports.append(dict(id=a['id'], robot=rid, index=a['index'], gear=a['gear'], bounds=a['bounds'], unit=a['unit'],
                              velocity_limit=a['velocity_limit']))

    pid = team.get('payload', {}).get('id', 'payload')
    if pid in views:
        raise ModelError('team: payload id collides with a robot id')
    pseed = seeds.get(pid)
    if pseed is None or len(pseed) != 6:
        raise ModelError('team: a six-coordinate payload chart seed is required')
    first = len(ids)
    _, ppaths, plocal, pframes = add_tree(pid + '_', payload, payload['root'], pseed, {}, pid)
    if plocal:
        raise ModelError('payload: a single rigid payload is supported (no internal joints)')
    payload_view = dict(id=pid, prefix=pid + '_', base=np.arange(first, first + 6), frames=pframes,
                        body=pid + '_' + payload['root'], record=payload)

    scene_frames = {name: rigid(T, name) for name, T in scene.get('frames', {}).items()}
    connections = {}
    for g in sorted(team.get('grasps', []), key=lambda g: g['id']):
        if g['robot'] not in views or g.get('type', 'weld') not in CONNECTION_FREEDOMS:
            raise ModelError(f"grasp {g['id']}: unknown robot or connection type")
        robot = views[g['robot']]
        frame = g.get('tool_frame', robot.record['ports']['tool']['frame'])
        if frame not in robot.frames or g['payload_frame'] not in pframes:
            raise ModelError(f"grasp {g['id']}: unknown frame")
        free = CONNECTION_FREEDOMS[g.get('type', 'weld')]
        if len(robot.dependent) != 6 - len(free):
            raise ModelError(f"grasp {g['id']}: {len(robot.dependent)} dependent tool-path coordinates cannot close "
                             f"a connection with {6 - len(free)} constrained directions; revise 'commanded'")
        b1, T1 = robot.frames[frame]
        b2, T2 = pframes[g['payload_frame']]
        connections[g['id']] = dict(id=g['id'], kind='grasp', type=g.get('type', 'weld'), robot=g['robot'],
                                    body1=b1, T1=T1, body2=b2, T2=T2, free=[f.tolist() for f in free],
                                    share=float(g.get('share', 1.)), payload_frame=g['payload_frame'],
                                    note='minus side: tool frame on the robot; plus side: payload frame. The '
                                         'reaction is the wrench applied to the payload, in tool axes.')
    for r in sorted(team.get('rests', []), key=lambda r: r['id']):
        if r['payload_frame'] not in pframes or r['scene_frame'] not in scene_frames:
            raise ModelError(f"rest {r['id']}: unknown frame")
        b2, T2 = pframes[r['payload_frame']]
        pads = []
        for pad in r.get('pads', []):
            if pad['scene_frame'] not in scene_frames:
                raise ModelError(f"rest {r['id']}: unknown pad frame {pad['scene_frame']}")
            pads.append(dict(id=pad['id'], T=scene_frames[pad['scene_frame']], friction=float(pad.get('friction', .5)),
                             half_width_m=float(pad.get('half_width_m', .03))))
        connections[r['id']] = dict(id=r['id'], kind='rest', type='weld', robot=None, body1='world',
                                    T1=scene_frames[r['scene_frame']], body2=b2, T2=T2, free=[], pads=pads,
                                    note='ideal sticking support: the payload frame rests on a world frame; the '
                                         'reaction is the total support wrench on the payload, in that world frame. '
                                         'Unilateral and friction limits are checked on the pad loads.')
    ids_g = [k for k, c in connections.items() if c['kind'] == 'grasp']
    if len({connections[k]['robot'] for k in ids_g}) != len(ids_g):
        raise ModelError('team: at most one grasp connection per robot')

    cmg = dict(schema=SCHEMA_COMPILED, name=team.get('name', ''), root_body='world', bodies=bodies, joints=joints,
               coordinate_ids=ids, independent_ids=ids, closures=[], gravity_m_s2=list(map(float, gravity)),
               q_seed=seed, armature=armature, damping=damping, limits=limits,
               charts={rid: [ids[i] for i in v.base] for rid, v in views.items()} | {pid: [ids[i] for i in payload_view['base']]})
    tree = Tree(cmg)
    # world paths of every connection (lowest common ancestor is the generated world root)
    chord_plans = {}
    for cid, c in connections.items():
        path1 = [tree.joint_id[k] for k in tree.support[tree.index[c['body1']]]]
        path2 = [tree.joint_id[k] for k in tree.support[tree.index[c['body2']]]]
        chord_plans[cid] = dict(kind=c['kind'], type=c['type'], body1=c['body1'], body2=c['body2'], lca='world',
                                path1=path1, path2=path2, rows=6, free=len(c['free']),
                                constrained=6 - len(c['free']))
    modules = {}
    for cid in ids_g:
        robot = views[connections[cid]['robot']]
        modules[cid] = dict(
            robot=robot.id, connection=cid,
            inputs=[ids[i] for i in payload_view['base']] + [ids[i] for i in robot.base]
                   + [robot.prefix + j for j in robot.tool_path if j in robot.commanded],
            dependent=[robot.prefix + j for j in robot.dependent],
            note='One closure module per robot: its dependent coordinates are disjoint from every other '
                 'module, given the payload pose, its own base pose and its commanded coordinates.')
    plan = dict(robots=structure, payload=dict(id=pid, coordinates=payload_view['base'].tolist()),
                bodies=len(bodies), physical_bodies=sum(b['mass_kg'] > 0 for b in bodies), joints=len(joints),
                coordinates=len(ids), generated_chart_frames=5 * (len(views) + 1), actuator_ports=len(ports),
                connections=chord_plans, modules=modules, total_mass_kg=float(sum(b['mass_kg'] for b in bodies)),
                input_sha256=canonical_hash(dict(team=team, payload=payload, scene={k: v for k, v in scene.items() if k != 'statics'},
                                                 robots={m['model']: canonical_hash(robots[m['model']]) for m in members})))
    cmg['ports'] = ports
    return TeamCompilation(cmg=cmg, plan=plan, source=dict(team=team, payload=payload, scene=scene, robots=robots),
                           tree=tree, robots=views, payload=payload_view, connections=connections, scene=scene)


def pose_of(comp, q, body, T_body_frame=None, kinematics=None):
    T, _ = kinematics if kinematics is not None else comp.tree.sweep(q)
    P = T[comp.tree.index[body]]
    return P if T_body_frame is None else P @ np.asarray(T_body_frame, float)


def base_pose(comp, q, rid):
    return chart_pose(np.asarray(q, float)[comp.robots[rid].base])
