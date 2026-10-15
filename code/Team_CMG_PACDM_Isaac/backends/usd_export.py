"""USD / PhysX scene of the simulated plant, generated from the plant description.

Same content as the MJCF of ``backends/mujoco_export``: free robots and payload,
wheels with free-spinning rollers, the site, the crossing cart.  Differences that
follow from the engine are listed in ``LIMITATIONS``.

The stage is authored with the public USD physics schema.  A PhysX API schema is
applied through the ``PhysxSchema`` bindings where they exist and have it, and as the
same schema token otherwise; PhysX attributes are written under their names on every
host alike.  The module therefore runs under plain ``usd-core`` (authoring and audit
only) and inside Isaac Sim (where the scene is simulated).  Nothing here starts or
steps a simulation.

Robots: one floating articulation each (root API on the base body), one rigid body
per compiled body and per roller, joints in a ``joints`` scope with the joint axis
on the local X axis, no drives (efforts and the declared joint damping are applied
by the runner), declared armature on every joint.

Conventional wheels (skid steering), ``contact='tyre'``: the wheel has no collider;
the tyre model of ``backends/tyre.py`` carries it (declared normal spring and
damper) and supplies its friction, applied by the runner (``TYRE_CARRY_LIMITATION``).
With ``contact='rigid'`` or ``'compliant'`` the tyre is a sphere collider of the
wheel radius, and it and the terrain are frictionless: PhysX carries the wheels and
the runner applies the friction of the tyre model (``TYRE_LIMITATION``).
"""
from __future__ import annotations

import re

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

try:
    from pxr import PhysxSchema
except ImportError:                                    # usd-core: authoring and audit only
    PhysxSchema = None

from teamc.se3 import chart_pose, inv, quat_wxyz

from .plantspec import build_plantspec

LIMITATIONS = [
    'PhysX contacts are not MuJoCo contacts: rigid Coulomb contacts with the max friction of the pair.  With '
    "contact='compliant' the roller/terrain pair becomes a PhysX compliant contact with the declared roller stiffness and "
    'damping per contact point; that option has not been exercised on any PhysX.',
    'A roller is the union of convex hulls of neighbouring profile rings (PhysX limits a hull to 64 vertices).  The hull '
    'points are authored in millimetres under a 0.001 scale: PhysX convex cooking drops points closer than 0.0007 mesh units '
    '(0.7 mm on a mesh in metres) to the hull built so far, which would coarsen a 22 mm roller.',
    'A compliant roller stiffness acts per contact point, so the wheel as a whole is stiffer than one such spring; every '
    'run reports the base height against rigid wheels.',
    'There is no latch constraint and no force/torque sensor in the scene: the clamp is the compliant clamp of '
    'backends/latch.py applied by the runner, and the wrist sensor is the Newton-Euler balance of the tool-side body.',
    'No joint drives, no PhysX joint friction and no rigid-body damping are authored; the declared joint damping is '
    'applied by the runner as an explicit effort.',
    'Authored mass properties and joint frames are audited offline; what PhysX does with them is checked by the '
    'preflight of the Isaac Sim workflow (property readback, kinematics, free-flight dynamics).',
]

TYRE_LIMITATION = (
    'Conventional wheels: the tyre is a frictionless sphere on frictionless terrain, and its friction is the tyre model of '
    'backends/tyre.py, applied by the runner as an external force at the ground point of the wheel, with the normal load '
    'PhysX reported for the step before.  PhysX solves contact friction as two independent rows per contact point, which is '
    'not isotropic Coulomb friction for a wheel that slides along and across its rolling direction at once: in '
    'tests/test_skid_physx.py (PhysX 5.3.1) a four-wheel base with its load shifted towards one axle turns at 0.36 to 0.75 '
    'of the rate Coulomb friction gives, depending on the rate, and with the TGS solver at a sixth of it even with equal '
    'loads.  PhysX is therefore not a check of the friction law of these wheels.  A rigid base on four rigid wheel contacts '
    'is statically indeterminate: how much load each wheel diagonal carries is left to the contact solver, and the tyre '
    'friction follows that load.')

TYRE_CARRY_LIMITATION = (
    'Conventional wheels: the wheel has no collider.  The tyre model of backends/tyre.py carries it, with the declared '
    'normal stiffness and damping of the tyre against the plane of the compiled terrain, and supplies its friction, '
    'applied by the runner as one explicit load at the contact point of the wheel.  The wheel loads are those of equal '
    'tyre springs under a rigid chassis; PhysX checks neither the friction law nor the normal contact of these wheels.  '
    'A wheel meeting anything but the terrain is left to the robot colliders of the base and to the clearance check of '
    'the audit.')

CONTACTS = ('rigid', 'compliant', 'tyre')

DEFAULTS = dict(
    roller_hull=(21, 30, True),          # profile rings, vertices per ring, alternate rings staggered
    contact='rigid',                     # wheel contact: 'rigid', 'compliant' (declared compliance), 'tyre' (no wheel collider)
    solver='PGS', position_iterations=64, velocity_iterations=8,
    contact_offset=dict(robot=.001, roller=.001, terrain=.001, payload=.001, scene=.0018, guide=.0005, obstacle=.002),
    rest_offset=0.,
    hull_vertex_limit=64, hull_min_thickness=.0005,
    hull_unit_m=1.0e-3,                  # length unit of the authored hull points (the mesh prim carries this scale)
    max_angular_velocity_deg_s=1.0e5, max_joint_velocity=1.0e6,
    physics_scene=None,                  # path of a physics scene to author (None: the runtime owns the scene)
    lights=True,
)


# ---------------------------------------------------------------------- small helpers
def _name(text):
    out = re.sub(r'[^A-Za-z0-9_]', '_', str(text))
    return out if re.match(r'[A-Za-z_]', out) else '_' + out


def _v3f(x):
    return Gf.Vec3f(*(float(v) for v in x))


def _quatf(R):
    w, x, y, z = quat_wxyz(R)
    return Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))


def _set_pose(prim, T, scale=None):
    T = np.asarray(T, float)
    w, x, y, z = quat_wxyz(T[:3, :3])
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(float(v) for v in T[:3, 3])))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
    if scale is not None:
        xform.AddScaleOp().Set(_v3f(scale))


# the PhysX API schemas this module applies
PHYSX_APIS = ('PhysxSceneAPI', 'PhysxArticulationAPI', 'PhysxRigidBodyAPI', 'PhysxJointAPI', 'PhysxCollisionAPI',
              'PhysxConvexHullCollisionAPI', 'PhysxMaterialAPI', 'PhysxContactReportAPI')


def schema_bindings():
    """Which of ``PHYSX_APIS`` the installed ``PhysxSchema`` bindings apply (none on a USD-only host)."""
    return [name for name in PHYSX_APIS if getattr(PhysxSchema, name, None) is not None] if PhysxSchema is not None else []


def _physx(prim, api_name, attributes):
    """Apply a PhysX API schema and set attributes.

    The schema is applied through the PhysxSchema bindings where they have it (Isaac Sim), and as its
    token otherwise (a USD-only host, or bindings without that API).  The attributes are written under
    their names with the types given here on every host alike, so the stage does not depend on the host.
    """
    schema = getattr(PhysxSchema, api_name, None) if PhysxSchema is not None else None
    if schema is not None:
        schema.Apply(prim)
    else:
        prim.AddAppliedSchema(api_name)
    for name, value_type, value in attributes:
        prim.CreateAttribute(name, value_type, custom=False).Set(value)


def _x_frame(axis):
    """Rotation whose X axis is ``axis`` (USD joints turn or slide about a canonical local axis)."""
    x = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([0., 0., 1.]) if abs(x[2]) < .9 else np.array([1., 0., 0.])
    y = np.cross(ref, x)
    y /= np.linalg.norm(y)
    return np.column_stack([x, y, np.cross(x, y)])


def _principal(inertia):
    d, R = np.linalg.eigh(np.asarray(inertia, float))
    if np.linalg.det(R) < 0.:
        R[:, 0] *= -1.
    return d, R


def hull_faces(points):
    """Outward triangles of the convex hull of a point set (counts, indices)."""
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    centre = points.mean(axis=0)
    tri = hull.simplices.copy()
    for k, (a, b, c) in enumerate(tri):
        n = np.cross(points[b] - points[a], points[c] - points[a])
        if n @ (points[a] - centre) < 0.:
            tri[k] = (a, c, b)
    return np.full(len(tri), 3, dtype=np.int32), tri.astype(np.int32).ravel()


def roller_segments(vertices, rings, around):
    """Point sets of the convex pieces of a roller: one hull per pair of neighbouring rings."""
    V = np.asarray(vertices, float).reshape(rings, around, 3)
    return [np.vstack([V[i], V[i + 1]]) for i in range(rings - 1)]


# ---------------------------------------------------------------------- scene
def build_usd(stage, comp, q0=None, options=None, spec=None):
    """Author the plant on ``stage``.  Returns ``meta``: every path, name and expected property the runtime needs."""
    opt = dict(DEFAULTS, **(options or {}))
    opt['contact_offset'] = dict(DEFAULTS['contact_offset'], **(options or {}).get('contact_offset', {}))
    if opt['contact'] not in CONTACTS or opt['solver'] not in ('PGS', 'TGS'):
        raise ValueError('contact must be rigid|compliant|tyre and solver PGS|TGS')
    hull = tuple(opt['roller_hull'])
    rings, around = int(hull[0]), int(hull[1])
    if 2 * around > opt['hull_vertex_limit']:
        raise ValueError(f'two rings of {around} vertices exceed the hull vertex limit {opt["hull_vertex_limit"]}')
    spec = build_plantspec(comp, hull) if spec is None else spec
    q0 = comp.seed() if q0 is None else np.asarray(q0, float)
    if stage.GetPrimAtPath('/World/Scene'):
        raise ValueError('/World/Scene already exists; build on a fresh stage')
    poses = comp.tree.poses(q0)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, '/World')
    if not stage.GetDefaultPrim():
        stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, '/World/Looks')
    UsdGeom.Xform.Define(stage, '/World/Scene')
    npos, nvel = int(opt['position_iterations']), int(opt['velocity_iterations'])
    offsets = opt['contact_offset']
    meta = dict(robots={}, colliders={}, expected={}, options=dict(opt, roller_hull=list(hull)), limitations=list(LIMITATIONS),
                physx_schema_bindings=PhysxSchema is not None, physx_schemas_applied_by_bindings=schema_bindings(),
                gravity=[float(v) for v in spec['gravity']],
                counts=dict(bodies=0, joints=0, colliders=0, hulls=0))

    if opt['physics_scene']:
        scene = UsdPhysics.Scene.Define(stage, opt['physics_scene'])
        author_scene_settings(scene.GetPrim(), spec['gravity'], opt)
        meta['physics_scene'] = opt['physics_scene']

    # ------------------------------------------------------------------ materials
    materials = {}

    def material(friction, compliant=False):
        """``compliant``: False, True (the declared roller compliance) or ``(stiffness, damping)``."""
        if compliant is True:
            compliant = (float(spec['roller']['contact_stiffness']), float(spec['roller']['contact_damping']))
        key = (round(float(friction), 6), compliant if compliant else False)
        if key in materials:
            return materials[key]
        path = f'/World/Looks/contact_{len(materials)}'
        mat = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr(float(friction))
        api.CreateDynamicFrictionAttr(float(friction))
        api.CreateRestitutionAttr(0.)
        attributes = [('physxMaterial:frictionCombineMode', Sdf.ValueTypeNames.Token, 'max'),
                      ('physxMaterial:restitutionCombineMode', Sdf.ValueTypeNames.Token, 'max'),
                      ('physxMaterial:dampingCombineMode', Sdf.ValueTypeNames.Token, 'max')]
        if compliant:                                   # stiffness first: PhysX warns when damping is set on a rigid material
            attributes += [('physxMaterial:compliantContactStiffness', Sdf.ValueTypeNames.Float, float(compliant[0])),
                           ('physxMaterial:compliantContactDamping', Sdf.ValueTypeNames.Float, float(compliant[1]))]
        _physx(mat.GetPrim(), 'PhysxMaterialAPI', attributes)
        materials[key] = mat
        return mat

    def collider(prim, role, friction, tag, compliant=False):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        _physx(prim, 'PhysxCollisionAPI', [
            ('physxCollision:contactOffset', Sdf.ValueTypeNames.Float, float(offsets[role])),
            ('physxCollision:restOffset', Sdf.ValueTypeNames.Float, float(opt['rest_offset']))])
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(friction, compliant), UsdShade.Tokens.weakerThanDescendants, 'physics')
        meta['colliders'][str(prim.GetPath())] = tag
        meta['counts']['colliders'] += 1

    def shape(path, g, T):
        kind = g['type']
        scale = None
        if kind == 'box':
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(2.0)                    # half extents enter as the scale
            scale = g['half_size']
        elif kind == 'sphere':
            geom = UsdGeom.Sphere.Define(stage, path)
            geom.CreateRadiusAttr(float(g['radius']))
        elif kind in ('cylinder', 'capsule'):
            geom = (UsdGeom.Cylinder if kind == 'cylinder' else UsdGeom.Capsule).Define(stage, path)
            geom.CreateRadiusAttr(float(g['radius']))
            geom.CreateHeightAttr(float(2. * g['half_length']))
            geom.CreateAxisAttr(UsdGeom.Tokens.z)
        else:
            raise ValueError(f'{path}: unsupported geometry {kind}')
        _set_pose(geom.GetPrim(), T, scale)
        geom.CreateDisplayColorAttr([_v3f(g.get('rgba', [.6, .6, .6, 1.])[:3])])
        return geom.GetPrim()

    def rigid_body(path, T, mass, com, inertia, kinematic=False):
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        _set_pose(prim, T)
        rb = UsdPhysics.RigidBodyAPI.Apply(prim)
        rb.CreateRigidBodyEnabledAttr(True)
        rb.CreateKinematicEnabledAttr(bool(kinematic))
        _physx(prim, 'PhysxRigidBodyAPI', [
            ('physxRigidBody:linearDamping', Sdf.ValueTypeNames.Float, 0.),
            ('physxRigidBody:angularDamping', Sdf.ValueTypeNames.Float, 0.),
            ('physxRigidBody:sleepThreshold', Sdf.ValueTypeNames.Float, 0.),
            ('physxRigidBody:stabilizationThreshold', Sdf.ValueTypeNames.Float, 0.),
            ('physxRigidBody:maxAngularVelocity', Sdf.ValueTypeNames.Float, float(opt['max_angular_velocity_deg_s'])),
            ('physxRigidBody:enableGyroscopicForces', Sdf.ValueTypeNames.Bool, True),
            ('physxRigidBody:solverPositionIterationCount', Sdf.ValueTypeNames.Int, npos),
            ('physxRigidBody:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, nvel)])
        _physx(prim, 'PhysxContactReportAPI', [('physxContactReport:threshold', Sdf.ValueTypeNames.Float, 0.)])
        if not kinematic:
            if mass <= 0.:
                raise ValueError(f'{path}: a simulated body needs a positive mass')
            d, R = _principal(inertia)
            if d.min() <= 0.:
                raise ValueError(f'{path}: a simulated body needs a positive definite inertia')
            api = UsdPhysics.MassAPI.Apply(prim)
            api.CreateMassAttr(float(mass))
            api.CreateCenterOfMassAttr(_v3f(com))
            api.CreateDiagonalInertiaAttr(_v3f(d))
            api.CreatePrincipalAxesAttr(_quatf(R))
        meta['counts']['bodies'] += 1
        return prim

    def joint(path, kind, parent, child, T_parent, T_child, axis=None, lower=None, upper=None, armature=0.):
        cls = dict(fixed=UsdPhysics.FixedJoint, revolute=UsdPhysics.RevoluteJoint, prismatic=UsdPhysics.PrismaticJoint)[kind]
        j = cls.Define(stage, path)
        j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
        Ra = np.eye(3) if axis is None else _x_frame(axis)
        for side, T in enumerate((np.asarray(T_parent, float), np.asarray(T_child, float))):
            getattr(j, f'CreateLocalPos{side}Attr')(_v3f(T[:3, 3]))
            getattr(j, f'CreateLocalRot{side}Attr')(_quatf(T[:3, :3] @ Ra))
        j.CreateCollisionEnabledAttr(False)
        j.CreateExcludeFromArticulationAttr(False)
        j.CreateJointEnabledAttr(True)
        if kind != 'fixed':
            j.CreateAxisAttr('X')
            if lower is not None:
                unit = 180. / np.pi if kind == 'revolute' else 1.
                j.CreateLowerLimitAttr(float(lower * unit))
                j.CreateUpperLimitAttr(float(upper * unit))
            _physx(j.GetPrim(), 'PhysxJointAPI', [
                ('physxJoint:armature', Sdf.ValueTypeNames.Float, float(armature)),
                ('physxJoint:jointFriction', Sdf.ValueTypeNames.Float, 0.),
                ('physxJoint:maxJointVelocity', Sdf.ValueTypeNames.Float, float(opt['max_joint_velocity']))])
        meta['counts']['joints'] += 1

    # ------------------------------------------------------------------ site
    mu_t = spec['terrain_friction']
    compliant = opt['contact'] == 'compliant' and spec['roller'] is not None
    tyres = [w['tyre'] for r in spec['robots'].values() for w in r['wheels'] if 'tyre' in w]
    carried = opt['contact'] == 'tyre'
    if carried and not tyres:
        raise ValueError("contact='tyre' is for conventional wheels; this team has none")
    if tyres:
        if spec['roller'] is not None:
            raise ValueError('rollers and tyres in one scene are not supported')
        if carried:
            # no wheel touches the terrain: it keeps its declared friction (see TYRE_CARRY_LIMITATION)
            meta['limitations'].append(TYRE_CARRY_LIMITATION)
            meta['tyres'] = dict(carried_by='tyre model of backends/tyre.py applied by the runner (normal spring and damper)',
                                 friction='tyre model of backends/tyre.py applied by the runner', wheel_collider=False,
                                 terrain_friction_authored=float(mu_t), terrain_friction_declared=float(mu_t))
        else:
            meta['limitations'].append(TYRE_LIMITATION)
            meta['tyres'] = dict(carried_by='PhysX contact of a sphere collider', wheel_collider=True,
                                 friction='tyre model of backends/tyre.py applied by the runner', terrain_friction_authored=0.,
                                 terrain_friction_declared=float(mu_t))
            mu_t = 0.                                   # see TYRE_LIMITATION: tyres and terrain are frictionless
            if opt['contact'] == 'compliant':
                compliant = (tyres[0]['contact_stiffness'], tyres[0]['contact_damping'])
    plane_schema = getattr(UsdPhysics, 'Plane', None) or getattr(UsdGeom, 'Plane', None)
    floor = (plane_schema.Define(stage, '/World/Scene/floor').GetPrim() if plane_schema is not None
             else stage.DefinePrim('/World/Scene/floor', 'Plane'))
    floor.CreateAttribute('axis', Sdf.ValueTypeNames.Token, custom=False).Set('Z')
    floor.CreateAttribute('width', Sdf.ValueTypeNames.Double, custom=False).Set(80.)
    floor.CreateAttribute('length', Sdf.ValueTypeNames.Double, custom=False).Set(80.)
    _set_pose(floor, np.eye(4))
    collider(floor, 'terrain', mu_t, dict(group='scene', role='terrain', name='floor'), compliant)
    UsdGeom.Imageable(floor).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    xy = np.array([s['T'][:2, 3] for s in spec['statics']] + [[0., 0.]])
    lo, hi = xy.min(axis=0) - 6., xy.max(axis=0) + 6.
    quad = UsdGeom.Mesh.Define(stage, '/World/Scene/floor_visual')
    quad.CreatePointsAttr([(float(lo[0]), float(lo[1]), 0.), (float(hi[0]), float(lo[1]), 0.),
                           (float(hi[0]), float(hi[1]), 0.), (float(lo[0]), float(hi[1]), 0.)])
    quad.CreateFaceVertexCountsAttr([4])
    quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    quad.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    quad.CreateDoubleSidedAttr(True)
    quad.CreateExtentAttr([(float(lo[0]), float(lo[1]), 0.), (float(hi[0]), float(hi[1]), 0.)])
    quad.CreateDisplayColorAttr([Gf.Vec3f(.74, .75, .76)])
    for s in spec['statics']:
        role = s.get('role', 'static')
        prim = shape('/World/Scene/' + _name(s['id']), s, s['T'])
        if s.get('collide', True):
            terrain = role == 'terrain'
            # terrain under tyres is frictionless (mu_t); everything else keeps its declared friction
            collider(prim, 'terrain' if terrain else ('guide' if role == 'guide' else 'scene'),
                     mu_t if (terrain and tyres) else s.get('friction', spec['terrain_friction']),
                     dict(group='scene', role=role, name=s['id']), compliant if terrain else False)
    ob = spec['obstacle']
    if ob:
        UsdGeom.Xform.Define(stage, '/World/Cart')
        T = np.eye(4)
        T[:3, 3] = ob['start']
        path = '/World/Cart/' + _name(ob['id'])
        rigid_body(path, T, 0., None, None, kinematic=True)
        prim = shape(path + '/box', dict(ob, type='box'), np.eye(4))
        collider(prim, 'obstacle', .6, dict(group='obstacle', role='obstacle', name=ob['id']))
        meta['obstacle'] = dict(path=path, id=ob['id'], start=[float(v) for v in ob['start']])

    # ------------------------------------------------------------------ robots
    ro = spec['roller']
    segments = None
    if ro is not None:
        segments = []
        unit = float(opt['hull_unit_m'])
        for pts in roller_segments(ro['vertices'], rings, around):
            counts, indices = hull_faces(pts)
            scaled = pts / unit
            segments.append((scaled.astype(np.float32), counts, indices, scaled.min(axis=0), scaled.max(axis=0)))
        meta['roller'] = dict(rings=rings, around=around, staggered=(len(hull) < 3 or bool(hull[2])), pieces=len(segments),
                              vertices_per_piece=2 * around, damping=ro['damping'], armature=ro['armature'], mass=ro['mass'],
                              inertia=[float(v) for v in ro['inertia']], mesh_unit_m=unit,
                              max_radius_m=float(np.hypot(ro['vertices'][:, 0], ro['vertices'][:, 1]).max()),
                              piece_radius_m=[float(np.hypot(pts[:, 0], pts[:, 1]).max()) for pts in
                                              roller_segments(ro['vertices'], rings, around)])
    for rid, r in spec['robots'].items():
        root = f'/World/{_name(rid)}'
        UsdGeom.Xform.Define(stage, root)
        UsdGeom.Scope.Define(stage, root + '/joints')
        wheels = {w['body']: w for w in r['wheels']}
        rmeta = dict(root=None, path=root, bodies={}, links=[], joints={}, dofs=[], rollers=[], wheels={}, base=r['base'],
                     armature={}, damping={}, limits={}, fixed={})
        expected = {}
        for b in r['bodies']:
            name = _name(b['name'])
            path = f'{root}/{name}'
            w = wheels.get(b['name'])
            mass, inertia = (w['hub_mass'], w['hub_inertia']) if (w is not None and w['rollers']) else (b['mass'], b['inertia'])
            T = poses[b['name']]
            prim = rigid_body(path, T, mass, b['com'], inertia)
            rmeta['bodies'][b['name']] = path
            rmeta['links'].append(name)
            expected[name] = dict(mass=float(mass), com=[float(v) for v in b['com']], inertia=np.asarray(inertia, float).tolist())
            if b['parent'] is None:
                UsdPhysics.ArticulationRootAPI.Apply(prim)
                _physx(prim, 'PhysxArticulationAPI', [
                    ('physxArticulation:enabledSelfCollisions', Sdf.ValueTypeNames.Bool, False),
                    ('physxArticulation:solverPositionIterationCount', Sdf.ValueTypeNames.Int, npos),
                    ('physxArticulation:solverVelocityIterationCount', Sdf.ValueTypeNames.Int, nvel),
                    ('physxArticulation:sleepThreshold', Sdf.ValueTypeNames.Float, 0.),
                    ('physxArticulation:stabilizationThreshold', Sdf.ValueTypeNames.Float, 0.)])
                rmeta['root'] = path
            else:
                parent = rmeta['bodies'][b['parent']]
                j = b['joint']
                if j is None:
                    jname = _name(b['fixed'])
                    joint(f'{root}/joints/{jname}', 'fixed', parent, path, b['T_parent_joint'], b['T_body_joint'])
                    rmeta['fixed'][b['fixed']] = jname
                else:
                    jname = _name(j['name'])
                    limited = j['upper'] - j['lower'] < 1e3
                    joint(f'{root}/joints/{jname}', j['type'], parent, path, b['T_parent_joint'], b['T_body_joint'], j['axis_joint'],
                          j['lower'] if limited else None, j['upper'] if limited else None, j['armature'])
                    rmeta['joints'][j['name']] = dict(dof=jname, index=int(j['index']), type=j['type'])
                    rmeta['dofs'].append(jname)
                    rmeta['armature'][jname] = float(j['armature'])
                    rmeta['damping'][jname] = float(j['damping'])
                    rmeta['limits'][jname] = [float(j['lower']), float(j['upper'])] if limited else None
            for g in b['geoms']:
                gname = _name(g['name'].split('/', 1)[-1])
                prim_g = shape(f'{path}/{gname}', g, g['T'])
                if g.get('collide', True):
                    collider(prim_g, 'robot', g.get('friction', .6), dict(group=rid, role='robot', name=g['name'], body=b['name']))
            if w is not None and w['rollers']:
                names = []
                for k in w['rollers']:
                    kname = _name(k['name'])
                    kpath = f'{root}/{kname}'
                    rigid_body(kpath, T @ k['T'], ro['mass'], np.zeros(3), np.diag(ro['inertia']))
                    rmeta['bodies'][k['name']] = kpath
                    rmeta['links'].append(kname)
                    expected[kname] = dict(mass=float(ro['mass']), com=[0., 0., 0.], inertia=np.diag(ro['inertia']).tolist())
                    jname = _name(k['joint'])
                    joint(f'{root}/joints/{jname}', 'revolute', path, kpath, k['T'], np.eye(4), [0., 0., 1.], None, None, ro['armature'])
                    rmeta['dofs'].append(jname)
                    rmeta['rollers'].append(jname)
                    rmeta['armature'][jname] = float(ro['armature'])
                    rmeta['damping'][jname] = float(ro['damping'])
                    rmeta['limits'][jname] = None
                    for n, (pts, counts, indices, lo_, hi_) in enumerate(segments):
                        mesh = UsdGeom.Mesh.Define(stage, f'{kpath}/hull_{n:02d}')
                        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
                        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(counts))
                        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
                        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
                        mesh.CreateExtentAttr([_v3f(lo_), _v3f(hi_)])
                        mesh.CreateDisplayColorAttr([Gf.Vec3f(.2, .2, .22)])
                        _set_pose(mesh.GetPrim(), np.eye(4), [unit, unit, unit])
                        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
                        _physx(mesh.GetPrim(), 'PhysxConvexHullCollisionAPI', [
                            ('physxConvexHullCollision:hullVertexLimit', Sdf.ValueTypeNames.Int, int(opt['hull_vertex_limit'])),
                            ('physxConvexHullCollision:minThickness', Sdf.ValueTypeNames.Float, float(opt['hull_min_thickness']))])
                        collider(mesh.GetPrim(), 'roller', w['friction'], dict(group=rid, role='roller', name=k['name'], wheel=w['id']),
                                 compliant)
                        meta['counts']['hulls'] += 1
                    names.append(jname)
                rmeta['wheels'][w['id']] = dict(joint=_name(w['joint']), rollers=names, body=b['name'], radius=float(w['radius']))
            elif w is not None:
                Tc = np.eye(4)
                Tc[:3, 3] = w['tyre']['centre']
                tyre = shape(f'{path}/tyre', dict(type='sphere', radius=w['tyre']['radius'], rgba=[.12, .12, .13, 1.]), Tc)
                UsdGeom.Imageable(tyre).CreateVisibilityAttr(UsdGeom.Tokens.invisible)   # the wheel is drawn by its own shape
                if not carried:                         # with contact='tyre' the sphere only marks the tyre: no collider
                    collider(tyre, 'roller', 0., dict(group=rid, role='roller', name=w['id'], wheel=w['id']), compliant)
                rmeta['wheels'][w['id']] = dict(joint=_name(w['joint']), rollers=[], body=b['name'], radius=float(w['radius']),
                                                tyre=f'{path}/tyre', friction=float(w['friction']), collider=not carried)
        rmeta['tool'] = dict(body=r['tool']['body'], T=np.asarray(r['tool']['T'], float).tolist())
        if r['sensor']:
            rmeta['sensor'] = dict(body=r['sensor']['body'], joint=_name(r['sensor']['joint']), T=np.asarray(r['sensor']['T'], float).tolist())
        meta['robots'][rid] = rmeta
        meta['expected'][rid] = expected

    # ------------------------------------------------------------------ payload
    p = spec['payload']
    pb = p['body']
    UsdGeom.Xform.Define(stage, '/World/Payload')
    path = '/World/Payload/' + _name(pb['name'])
    rigid_body(path, chart_pose(q0[p['base']]), pb['mass'], pb['com'], pb['inertia'])
    for g in pb['geoms']:
        prim_g = shape(f"{path}/{_name(g['name'].split('/', 1)[-1])}", g, g['T'])
        if g.get('collide', True):
            collider(prim_g, 'payload', g.get('friction', .5), dict(group=p['id'], role='payload', name=g['name'], body=pb['name']))
    meta['payload'] = dict(path=path, body=pb['name'], base=p['base'], id=p['id'],
                           expected=dict(mass=float(pb['mass']), com=[float(v) for v in pb['com']],
                                         inertia=np.asarray(pb['inertia'], float).tolist()))

    if opt['lights']:
        UsdGeom.Scope.Define(stage, '/World/Lighting')
        UsdLux.DomeLight.Define(stage, '/World/Lighting/Dome').CreateIntensityAttr(600.)
        sun = UsdLux.DistantLight.Define(stage, '/World/Lighting/Sun')
        sun.CreateIntensityAttr(1800.)
        from teamc.se3 import rpy, transform
        _set_pose(sun.GetPrim(), transform(rpy(np.radians(35.), np.radians(-30.), np.radians(15.)), [0., 0., 8.]))
    meta['q0'] = [float(v) for v in q0]
    return meta


def author_scene_settings(prim, gravity, options=None):
    """Gravity and PhysX solver settings on a physics scene prim (the scene of the runtime or one authored here)."""
    opt = dict(DEFAULTS, **(options or {}))
    g = np.asarray(gravity, float)
    scene = UsdPhysics.Scene(prim)
    scene.CreateGravityDirectionAttr(_v3f(g / np.linalg.norm(g)))
    scene.CreateGravityMagnitudeAttr(float(np.linalg.norm(g)))
    npos, nvel = int(opt['position_iterations']), int(opt['velocity_iterations'])
    _physx(prim, 'PhysxSceneAPI', [
        ('physxScene:enableGPUDynamics', Sdf.ValueTypeNames.Bool, False),
        ('physxScene:broadphaseType', Sdf.ValueTypeNames.Token, 'MBP'),
        ('physxScene:solverType', Sdf.ValueTypeNames.Token, opt['solver']),
        ('physxScene:enableStabilization', Sdf.ValueTypeNames.Bool, False),
        ('physxScene:enableEnhancedDeterminism', Sdf.ValueTypeNames.Bool, True),
        ('physxScene:minPositionIterationCount', Sdf.ValueTypeNames.UInt, npos),
        ('physxScene:maxPositionIterationCount', Sdf.ValueTypeNames.UInt, npos),
        ('physxScene:minVelocityIterationCount', Sdf.ValueTypeNames.UInt, nvel),
        ('physxScene:maxVelocityIterationCount', Sdf.ValueTypeNames.UInt, nvel)])
    if 'timestep' in opt:
        _physx(prim, 'PhysxSceneAPI', [('physxScene:timeStepsPerSecond', Sdf.ValueTypeNames.UInt, int(round(1. / opt['timestep'])))])
    return dict(solver=opt['solver'], position_iterations=npos, velocity_iterations=nvel)


def read_scene_settings(prim):
    """The physics-scene attributes as they are on the stage (authored values only)."""
    out = {}
    for attribute in prim.GetAttributes():
        name = attribute.GetName()
        if (name.startswith('physxScene:') or name.startswith('physics:')) and attribute.HasAuthoredValue():
            value = attribute.Get()
            out[name] = [float(v) for v in value] if hasattr(value, '__len__') and not isinstance(value, str) else value
    return out


# ---------------------------------------------------------------------- audit of the authored stage
def _matrix(prim):
    M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    return M.T                                         # USD stores row vectors


def _rot(q):
    from teamc.se3 import from_quat_wxyz
    return from_quat_wxyz([q.GetReal(), *q.GetImaginary()])


def audit_stage(stage, comp, meta, tol=2e-5):
    """Compare what is on the stage with the compiled graph (no simulation).  Returns a report with ``passed``."""
    failures, checks = [], {}
    q0 = np.asarray(meta['q0'], float)
    poses = comp.tree.poses(q0)
    cmg_bodies = {b['id']: b for b in comp.cmg['bodies']}
    ids = comp.cmg['coordinate_ids']
    worst = dict(pose=0., inertia=0., mass=0., joint_gap=0., joint_axis=0., joint_value=0.)
    for rid, r in meta['robots'].items():
        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI) and str(p.GetPath()).startswith(r['path'] + '/')]
        if roots != [r['root']]:
            failures.append(f'{rid}: articulation roots {roots}')
        total = 0.
        for name, path in r['bodies'].items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                failures.append(f'missing body {path}')
                continue
            api = UsdPhysics.MassAPI(prim)
            mass = float(api.GetMassAttr().Get())
            total += mass
            D = np.asarray(api.GetDiagonalInertiaAttr().Get(), float)
            R = _rot(api.GetPrincipalAxesAttr().Get())
            e = meta['expected'][rid][_name(name)]
            worst['mass'] = max(worst['mass'], abs(mass - e['mass']) / e['mass'])
            worst['inertia'] = max(worst['inertia'], float(np.abs(R @ np.diag(D) @ R.T - np.asarray(e['inertia'])).max()
                                                           / max(np.abs(e['inertia']).max(), 1e-12)))
            if name in poses:
                worst['pose'] = max(worst['pose'], float(np.abs(_matrix(prim) - poses[name]).max()))
        compiled = sum(cmg_bodies[b]['mass_kg'] for b in cmg_bodies if b.startswith(comp.robots[rid].prefix))
        if abs(total - compiled) > 1e-6 * compiled:
            failures.append(f'{rid}: total mass {total} differs from the compiled {compiled}')
        checks[f'{rid}_mass_kg'] = total
        # joints: both local frames coincide up to the joint motion about X, which must be the compiled coordinate
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint) or not str(prim.GetPath()).startswith(r['path'] + '/joints/'):
                continue
            j = UsdPhysics.Joint(prim)
            b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            if len(b0) != 1 or len(b1) != 1:
                failures.append(f'{prim.GetPath()}: joint without two bodies')
                continue
            frames = []
            for side, body in enumerate((b0[0], b1[0])):
                L = np.eye(4)
                L[:3, :3] = _rot(getattr(j, f'GetLocalRot{side}Attr')().Get())
                L[:3, 3] = np.asarray(getattr(j, f'GetLocalPos{side}Attr')().Get(), float)
                frames.append(_matrix(stage.GetPrimAtPath(body)) @ L)
            rel = inv(frames[0]) @ frames[1]
            name = prim.GetName()
            compiled_id = next((k for k, v in r['joints'].items() if v['dof'] == name), None)
            if prim.IsA(UsdPhysics.RevoluteJoint):
                value = np.arctan2(rel[2, 1], rel[1, 1])
                worst['joint_gap'] = max(worst['joint_gap'], float(np.linalg.norm(rel[:3, 3])))
                worst['joint_axis'] = max(worst['joint_axis'], float(np.linalg.norm(rel[:3, 0] - [1., 0., 0.])))
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                value = rel[0, 3]
                worst['joint_gap'] = max(worst['joint_gap'], float(np.linalg.norm(rel[1:3, 3])))
                worst['joint_axis'] = max(worst['joint_axis'], float(np.abs(rel[:3, :3] - np.eye(3)).max()))
            else:
                value = None
                worst['joint_gap'] = max(worst['joint_gap'], float(np.abs(rel - np.eye(4)).max()))
            if value is not None and compiled_id is not None:
                want = q0[ids.index(compiled_id)]
                err = value - want
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    err = (err + np.pi) % (2. * np.pi) - np.pi
                worst['joint_value'] = max(worst['joint_value'], abs(float(err)))
            if prim.HasAPI(UsdPhysics.DriveAPI):
                failures.append(f'{prim.GetPath()}: unexpected drive')
    hulls = 0
    for path, tag in meta['colliders'].items():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
            failures.append(f'missing collider {path}')
        elif prim.IsA(UsdGeom.Mesh):
            hulls += 1
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
            if len(points) > meta['options']['hull_vertex_limit']:
                failures.append(f'{path}: {len(points)} hull vertices')
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != 'convexHull':
                failures.append(f'{path}: not a convex hull collider')
            if hulls <= 40:                           # the pieces are the same for every roller: check the first two rollers
                local = _matrix(prim.GetParent())
                world = (_matrix(prim) @ np.c_[points, np.ones(len(points))].T).T[:, :3]
                radius = np.linalg.norm((world - local[:3, 3]) @ local[:3, :2], axis=1).max()
                want = meta['roller']['piece_radius_m'][int(prim.GetName().rsplit('_', 1)[-1])]   # of this piece of the profile
                if abs(radius - want) > 2e-6:
                    failures.append(f'{path}: hull radius {radius:.5f} m does not fit the roller ({want:.5f} m)')
    if hulls != meta['counts']['hulls']:
        failures.append('convex hull count differs')
    limits = dict(pose=tol, inertia=2e-6, mass=2e-7, joint_gap=tol, joint_axis=tol, joint_value=tol)
    for key, value in worst.items():
        checks[key] = value
        if not value <= limits[key]:
            failures.append(f'{key}: {value:.3g} exceeds {limits[key]:.3g}')
    return dict(passed=not failures, failures=failures, checks=checks, counts=dict(meta['counts']),
                scope='authored USD against the compiled graph; no PhysX behaviour is checked here')
