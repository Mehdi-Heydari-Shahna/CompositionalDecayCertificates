#!/usr/bin/env python
"""One-time extraction of the physical robot records from the upstream descriptions.

    python tools/build_robot.py                     # every platform
    python tools/build_robot.py ridgeback_ur10e     # one of them

Platforms and their sources (unchanged copies in ``upstream/``):

``rbrobout_ur30``  RB-ROBOUT+ with a UR30

* ``robotnik_description`` (Robotnik Automation, BSD-3-Clause): base, top cover,
  mecanum wheel, Ewellix lift and arm-mount macros;
* ``ur_description`` (Universal Robots, BSD-3-Clause): UR macro and the UR30
  kinematic, inertial and joint-limit configuration.

``ridgeback_ur10e``  Clearpath Ridgeback with a UR10e

* ``ridgeback_description`` (Clearpath Robotics, BSD-3-Clause): the Ridgeback
  with its two rockers and four wheels;
* ``ur_description``: UR macro and the UR10e configuration.

``husky_a300_ur10e``  Clearpath Husky A300 with a UR10e

* ``clearpath_platform_description`` (Clearpath Robotics, BSD-3-Clause): the A300
  in its skid-steer configuration with bumpers and top plate;
* ``ur_description``: UR macro and the UR10e configuration.

The URDF is produced by ``xacro`` from those files, then reduced to physical
records by ``teamc.urdf_import``.  Everything the public descriptions do not
contain is listed under ``declared`` in the record and is a synthetic engineering
value, not a manufacturer parameter.

Needs ``xacro`` and ``pyyaml`` (development only).  The generated files are
shipped, so running the package does not need them.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from teamc import SCHEMA_ROBOT  # noqa: E402
from teamc.se3 import origin, rot, transform  # noqa: E402
from teamc.urdf_import import physical_records  # noqa: E402

ARM = ['arm_shoulder_pan_joint', 'arm_shoulder_lift_joint', 'arm_elbow_joint',
       'arm_wrist_1_joint', 'arm_wrist_2_joint', 'arm_wrist_3_joint']
WHEELS = ['front_left', 'front_right', 'back_left', 'back_right']


def make_urdf(top='inputs/xacro/rbrobout_plus_ur30.urdf.xacro', packages=('robotnik_description', 'ur_description'),
              staged_top=None, extras=None, stubs=(), env_prefix=None, mappings=None):
    """URDF text from the upstream xacro files.

    ``top`` is a top-level xacro of this package; with ``staged_top`` the top-level file is an upstream
    one (path below the staging folder) and ``extras`` maps an environment variable to a xacro of this
    package that the upstream file includes through ``$(optenv ...)``.  ``stubs`` are packages an
    upstream file includes but does not use here; they are staged as empty files.  Environment
    variables starting with ``env_prefix`` are hidden while xacro runs, so the result does not depend
    on the shell it is run from.  ``mappings`` are xacro arguments.
    """
    import xacro
    stage = Path(tempfile.mkdtemp(prefix='team_xacro_'))
    saved = dict(os.environ)
    try:
        for package in packages:
            shutil.copytree(ROOT / 'upstream' / package, stage / package)
        for rel in stubs:
            path = stage / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('<robot xmlns:xacro="http://ros.org/wiki/xacro"/>\n', encoding='utf-8', newline='\n')
        if env_prefix:
            for key in [k for k in os.environ if k.startswith(env_prefix)]:
                del os.environ[key]
        for variable, rel in (extras or {}).items():
            shutil.copy(ROOT / rel, stage / Path(rel).name)
            os.environ[variable] = (stage / Path(rel).name).as_posix()
        if staged_top is None:
            entry = stage / 'top.urdf.xacro'
            shutil.copy(ROOT / top, entry)
        else:
            entry = stage / staged_top
        names = list(packages) + [Path(rel).parts[0] for rel in stubs]
        for path in stage.rglob('*'):
            if not path.is_file() or path.suffix not in ('.xacro', '.gazebo', '.urdf'):
                continue
            if staged_top is None and path.suffix != '.xacro':
                continue
            text = path.read_text(encoding='utf-8')
            for package in names:
                text = text.replace(f'$(find {package})', (stage / package).as_posix())
            path.write_text(text, encoding='utf-8', newline='\n')
        urdf = xacro.process_file(str(entry), mappings=dict(mappings or {})).toprettyxml(indent='  ')
    finally:
        os.environ.clear()
        os.environ.update(saved)
        shutil.rmtree(stage, ignore_errors=True)
    # paths of the temporary staging folder must not leak into the shipped file
    return urdf.replace(stage.as_posix(), 'package:/')


def box(body, lo, hi, rgba, collide=True, name=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return dict(body=body, name=name, type='box', half_size=((hi - lo) / 2).tolist(),
                T_body_geom=transform(p=(lo + hi) / 2).tolist(), rgba=rgba, collide=collide)


def capsule(body, a, b, radius, rgba, collide=True, name=None):
    """Capsule between two points given in the body frame (local z along a -> b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    z = (b - a) / np.linalg.norm(b - a)
    x = np.cross([0, 1., 0], z) if abs(z[1]) < .9 else np.cross([1., 0, 0], z)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    return dict(body=body, name=name, type='capsule', radius=radius, half_length=float(np.linalg.norm(b - a) / 2),
                T_body_geom=transform(R, (a + b) / 2).tolist(), rgba=rgba, collide=collide)


def cylinder(body, center, axis, radius, half_length, rgba, collide=True, name=None):
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.cross([0, 1., 0], z) if abs(z[1]) < .9 else np.cross([1., 0, 0], z)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    return dict(body=body, name=name, type='cylinder', radius=radius, half_length=half_length,
                T_body_geom=transform(R, center).tolist(), rgba=rgba, collide=collide)


def build():
    urdf = make_urdf()
    rec = physical_records(urdf, drop_frames=('base_docking_contact', 'arm_base', 'lift_base_link'))
    bodies, joints, frames = rec['bodies'], rec['joints'], rec['frames']
    if rec['root'] != 'base_link':
        raise RuntimeError('Unexpected URDF root body ' + rec['root'])

    # ---------------------------------------------------------------- joints from the URDF
    limits, actuators = {}, []
    for j in joints:
        if j['type'] == 'fixed':
            continue
        u = j.pop('urdf_limit')
        continuous = j.pop('continuous')
        if j['id'] in ARM:
            limits[j['id']] = dict(lower=u['lower'], upper=u['upper'])
            actuators.append(dict(id=j['id'].replace('_joint', '_motor'), joint_id=j['id'], gear=1., unit='N m',
                                  effort_bounds=[-u['effort'], u['effort']], velocity_limit=u['velocity'],
                                  source='ur_description config/ur30/joint_limits.yaml'))
        elif j['id'] == 'lift_top_joint':
            limits[j['id']] = dict(lower=u['lower'], upper=u['upper'])
            actuators.append(dict(id='lift_motor', joint_id=j['id'], gear=1., unit='N',
                                  effort_bounds=[-1500., 1500.], velocity_limit=u['velocity'],
                                  source='declared: 1500 N (distributor listing of the 900 mm LIFTKIT-UR column). '
                                         'The upstream URDF placeholder is 700 N, below the weight of a UR30 '
                                         'with payload; velocity limit 0.088 m/s is the upstream URDF value.'))
        else:
            if not continuous:
                raise RuntimeError('Unexpected bounded wheel joint ' + j['id'])
            limits[j['id']] = dict(lower=-1e9, upper=1e9)

    # ---------------------------------------------------------------- declared tooling (synthetic)
    T_flange = np.asarray(frames['arm_tool0']['T_body_frame'])          # tool0 on wrist_3 (z out of the flange)
    sensor_h, clamp_h = .035, .160
    bodies.append(dict(id='ft_sensor', mass_kg=.35, com_m=[0, 0, sensor_h / 2],
                       inertia_kg_m2=np.diag([.35 * (3 * .04 ** 2 + sensor_h ** 2) / 12] * 2 + [.35 * .04 ** 2 / 2]).tolist()))
    m_c, sx, sy, sz = 3.2, .14, .128, clamp_h
    bodies.append(dict(id='clamp', mass_kg=m_c, com_m=[0, 0, .065],
                       inertia_kg_m2=np.diag([m_c * (sy ** 2 + sz ** 2) / 12, m_c * (sx ** 2 + sz ** 2) / 12,
                                              m_c * (sx ** 2 + sy ** 2) / 12]).tolist()))
    joints.append(dict(id='sensor_mount', type='fixed', body_a='arm_wrist_3_link', body_b='ft_sensor',
                       T_AJ=T_flange.tolist(), T_BJ=np.eye(4).tolist()))
    joints.append(dict(id='sensor_joint', type='fixed', body_a='ft_sensor', body_b='clamp',
                       T_AJ=transform(p=[0, 0, sensor_h]).tolist(), T_BJ=np.eye(4).tolist()))
    tool_z = .120                                                       # clamp frame -> jaw centre
    frames['tool'] = dict(body='clamp', T_body_frame=transform(p=[0, 0, tool_z]).tolist())
    frames['ft_sensor_frame'] = dict(body='clamp', T_body_frame=np.eye(4).tolist())
    frames['footprint'] = frames.pop('base_footprint')

    # ---------------------------------------------------------------- wheels
    wheel_records = []
    # Roller-axis (traction) direction at the ground, base frame: the Gazebo friction
    # direction fdir1 of the upstream mecanum wheel macro (reflect -> "1 -1 0", else "1 1 0").
    reflect = dict(front_right=False, front_left=True, back_left=False, back_right=True)
    for name in WHEELS:
        g = np.array([1., -1., 0.]) if reflect[name] else np.array([1., 1., 0.])
        wheel_records.append(dict(
            id=name, joint_id=name + '_wheel_joint', body=name + '_wheel_link', type='mecanum',
            radius_m=.127, width_m=.087, traction_direction_base=(g / np.linalg.norm(g)).tolist(),
            friction=.8, rollers=dict(count=10, max_radius_m=.022, length_m=.118, mass_kg=.12,
                                       contact_time_constant_s=.010, contact_damping_ratio=.5,
                                       contact_stiffness_N_m=1.4e6, contact_damping_N_s_m=1.3e4),
            source='radius, width, mass and traction direction: upstream mecanum wheel macro; '
                   'friction, roller geometry and roller compliance: declared. The compliance is given twice, for two '
                   'contact models that are not equivalent: a soft-contact time constant and damping ratio (MuJoCo), and a '
                   'spring and damper per contact point (PhysX; one such spring deflects about 0.9 mm under a quarter of the '
                   'robot weight). Every run reports the base height against rigid wheels'))
        actuators.append(dict(id=name + '_wheel_motor', joint_id=name + '_wheel_joint', gear=1., unit='N m',
                              effort_bounds=[-150., 150.], velocity_limit=10.,
                              source='declared: 150 N m; velocity limit 10 rad/s from the upstream ros2_control file '
                                     '(the upstream URDF effort limit 100000 is a placeholder)'))

    # ---------------------------------------------------------------- drive data (synthetic)
    drive = {}
    for jid, armature, damping in zip(ARM, [2.5, 2.5, 1.2, .25, .25, .25], [8., 8., 5., 1.5, 1.5, 1.5]):
        drive[jid] = dict(armature=armature, damping=damping)
    drive['lift_top_joint'] = dict(armature=40., damping=400.)
    for name in WHEELS:
        drive[name + '_wheel_joint'] = dict(armature=.15, damping=.05)

    # ---------------------------------------------------------------- primitive geometry (not CAD)
    grey, dark, blue, steel, orange = [.62, .64, .67, 1.], [.16, .17, .19, 1.], [.36, .55, .78, 1.], [.75, .77, .8, 1.], [.95, .55, .12, 1.]
    geometry = [
        box('base_link', [-.8585, -.425, -.093], [.8585, .425, .207], dark, name='chassis'),
        box('top_cover_base_link', [-.859, -.47, .307], [.849, .425, .40], grey, name='deck'),
        box('top_cover_base_link', [-.35, -.46, .40], [.85, .425, .70], grey, name='cabinet'),
        box('base_link', [-.86, -.1, .18], [-.66, .1, .8676], steel, name='lift_column'),
        box('lift_top_link', [-.074, -.074, -.9924], [.074, .074, 0.], steel, collide=False, name='lift_tube'),
        cylinder('arm_base_link_inertia', [0, 0, .05], [0, 0, 1], .12, .05, blue, name='arm_base'),
        cylinder('arm_shoulder_link', [0, 0, 0], [0, 0, 1], .11, .12, blue, name='shoulder'),
        capsule('arm_upper_arm_link', [0, 0, .215], [-.637, 0, .215], .075, blue, name='upper_arm'),
        cylinder('arm_upper_arm_link', [0, 0, .11], [0, 0, 1], .10, .13, blue, collide=False, name='shoulder_cap'),
        capsule('arm_forearm_link', [0, 0, .063], [-.5037, 0, .063], .055, blue, name='forearm'),
        cylinder('arm_forearm_link', [0, 0, .10], [0, 0, 1], .08, .11, blue, collide=False, name='elbow_cap'),
        cylinder('arm_wrist_1_link', [0, 0, 0], [0, 0, 1], .055, .075, blue, name='wrist_1'),
        cylinder('arm_wrist_2_link', [0, 0, 0], [0, 0, 1], .055, .075, blue, name='wrist_2'),
        cylinder('arm_wrist_3_link', [0, 0, -.03], [0, 0, 1], .05, .03, blue, name='wrist_3'),
    ]
    T_s = np.asarray(frames['arm_tool0']['T_body_frame'])
    geometry.append(dict(body='ft_sensor', name='ft_sensor', type='cylinder', radius=.045, half_length=sensor_h / 2,
                         T_body_geom=transform(p=[0, 0, sensor_h / 2]).tolist(), rgba=steel, collide=False))
    # C-shaped clamp: top plate and two jaws around a 60 mm wide section (4 mm clearance per side).
    # The tool frame (jaw centre) is 70 mm below the plate, so sections up to 120 mm high fit under it.
    geometry += [box('clamp', [-.07, -.064, 0.], [.07, .064, .050], orange, name='clamp_top'),
                 box('clamp', [-.07, -.064, .050], [.07, -.034, clamp_h], orange, name='clamp_jaw_a'),
                 box('clamp', [-.07, .034, .050], [.07, .064, clamp_h], orange, name='clamp_jaw_b')]

    seed = {jid: q for jid, q in zip(ARM, [0., -1.2, -1.7, -1.8, 1.5708, 0.])}
    seed['lift_top_joint'] = .40
    seed.update({name + '_wheel_joint': 0. for name in WHEELS})

    record = dict(
        schema=SCHEMA_ROBOT, name='RB-ROBOUT+ with UR30', units='SI', floating_root='base_link',
        bodies=bodies, joints=joints, frames=frames, limits=limits, actuators=actuators, wheels=wheel_records,
        ports=dict(tool=dict(frame='tool'), sensor=dict(joint_id='sensor_joint', frame='ft_sensor_frame',
                                                     range=dict(force_N=600., torque_Nm=60.),
                                                     builtin_ur30_range=dict(force_N=200., torque_Nm=20.))),
        drive=drive, geometry=geometry, seed=dict(joints=seed), gravity_m_s2=[0., 0., -9.81],
        declared=dict(
            tooling='ft_sensor (0.35 kg) and clamp (3.2 kg) bodies, their inertias, the tool frame and the '
                    'sensor range are synthetic. The built-in UR30 force/torque sensor range (200 N, 20 N m, '
                    'UR30 technical specification) is recorded for the load audit only.',
            drive='armature and damping are synthetic values for numerical realism, not UR or Robotnik data.',
            actuators='arm effort bounds are the ur_description UR30 limits; lift and wheel bounds are declared.',
            geometry='boxes, cylinders and capsules placed from the kinematic frames, the published '
                     'centres of mass and the bounding boxes of the upstream chassis and cover meshes; not CAD.',
            masses='upstream URDF values, several of them marked TODO upstream (base 200 kg, top cover 200 kg); '
                   'the lift base link has zero mass upstream and is treated as a frame.',
            omitted='IMU, laser scanners, RGB-D camera and the signal mast of the upstream robot.'),
        provenance=dict(
            robotnik_description='https://github.com/RobotnikAutomation/robotnik_description commit 4bc73425d090ead4591a7091e7ef7e7dc4fe862a (BSD-3-Clause)',
            ur_description='https://github.com/UniversalRobots/Universal_Robots_ROS2_Description commit 6662e15f32c23c12ece57d0050aee9a716e4fc41 (BSD-3-Clause)',
            xacro_top='inputs/xacro/rbrobout_plus_ur30.urdf.xacro',
            urdf_sha256=hashlib.sha256(urdf.encode()).hexdigest()))
    return record, urdf


# ---------------------------------------------------------------------- tooling of the UR10e (both Clearpath platforms)
def ur10e_clamp(bodies, joints, frames):
    """Add the declared clamp of the UR10e records; returns its dimensions.

    The UR10e measures force and torque in its tool flange: the sensor is the joint between the
    last arm link and the clamp, at the flange frame.
    """
    T_flange = np.asarray(frames['arm_tool0']['T_body_frame'])          # tool0 on wrist_3 (z out of the flange)
    clamp_h, plate, jaw_in, jaw_out, half_x = .140, .030, .029, .054, .060
    m_c = 1.2
    sx, sy, sz = 2 * half_x, 2 * jaw_out, clamp_h
    bodies.append(dict(id='clamp', mass_kg=m_c, com_m=[0, 0, .055],
                       inertia_kg_m2=np.diag([m_c * (sy ** 2 + sz ** 2) / 12, m_c * (sx ** 2 + sz ** 2) / 12,
                                              m_c * (sx ** 2 + sy ** 2) / 12]).tolist()))
    joints.append(dict(id='sensor_joint', type='fixed', body_a='arm_wrist_3_link', body_b='clamp',
                       T_AJ=T_flange.tolist(), T_BJ=np.eye(4).tolist()))
    tool_z = plate + .070                                               # clamp frame -> jaw centre
    frames['tool'] = dict(body='clamp', T_body_frame=transform(p=[0, 0, tool_z]).tolist())
    frames['ft_sensor_frame'] = dict(body='clamp', T_body_frame=np.eye(4).tolist())
    return clamp_h, plate, jaw_in, jaw_out, half_x, .5 * (jaw_in + jaw_out)


def ur10e_geometry(clamp_h, plate, jaw_in, jaw_out, half_x):
    """Primitive geometry of the UR10e and its clamp (not CAD)."""
    blue, orange = [.36, .55, .78, 1.], [.95, .55, .12, 1.]
    return [
        cylinder('arm_base_link_inertia', [0, 0, .05], [0, 0, 1], .095, .05, blue, name='arm_base'),
        cylinder('arm_shoulder_link', [0, 0, 0], [0, 0, 1], .09, .10, blue, name='shoulder'),
        capsule('arm_upper_arm_link', [0, 0, .1762], [-.6127, 0, .1762], .065, blue, name='upper_arm'),
        cylinder('arm_upper_arm_link', [0, 0, .09], [0, 0, 1], .085, .11, blue, collide=False, name='shoulder_cap'),
        capsule('arm_forearm_link', [0, 0, .0393], [-.57155, 0, .0393], .05, blue, name='forearm'),
        cylinder('arm_forearm_link', [0, 0, .075], [0, 0, 1], .065, .095, blue, collide=False, name='elbow_cap'),
        cylinder('arm_wrist_1_link', [0, 0, 0], [0, 0, 1], .047, .065, blue, name='wrist_1'),
        cylinder('arm_wrist_2_link', [0, 0, 0], [0, 0, 1], .047, .065, blue, name='wrist_2'),
        cylinder('arm_wrist_3_link', [0, 0, -.03], [0, 0, 1], .045, .03, blue, name='wrist_3'),
        # C-shaped clamp: top plate and two jaws around a 50 mm wide section (4 mm clearance per side).
        # The tool frame (jaw centre) is 70 mm below the plate, so sections up to 120 mm high fit under it.
        box('clamp', [-half_x, -jaw_out, 0.], [half_x, jaw_out, plate], orange, name='clamp_top'),
        box('clamp', [-half_x, -jaw_out, plate], [half_x, -jaw_in, clamp_h], orange, name='clamp_jaw_a'),
        box('clamp', [-half_x, jaw_in, plate], [half_x, jaw_out, clamp_h], orange, name='clamp_jaw_b'),
    ]


def ur10e_ports(half_x, mid):
    """Tool and sensor ports of the UR10e records (clamp compliance and sensor range)."""
    return dict(tool=dict(frame='tool', latch_points=[[half_x, mid, 0.], [-half_x, mid, 0.], [0., -mid, 0.]],
                          clamp=dict(stiffness_N_m=8.0e4, damping_N_s_m=3.0e2, rot_stiffness_Nm_rad=4.0e2,
                                     rot_damping_Nm_s_rad=2.5)),
                sensor=dict(joint_id='sensor_joint', frame='ft_sensor_frame',
                            range=dict(force_N=100., torque_Nm=10.)))


# ---------------------------------------------------------------------- Clearpath Ridgeback with a UR10e
RIDGEBACK_WHEELS = ['front_left', 'front_right', 'rear_left', 'rear_right']
# Arm posture of the record (a folded, elbow-up posture inside the footprint of the base).
RIDGEBACK_ARM_SEED = [3.1416, -1.9, 2.3, -1.97, -1.5708, 0.]


def build_ridgeback_ur10e():
    urdf = make_urdf(packages=('ridgeback_description', 'ur_description'),
                     staged_top='ridgeback_description/urdf/ridgeback.urdf.xacro',
                     extras=dict(RIDGEBACK_URDF_EXTRAS='inputs/xacro/ridgeback_ur10e_extras.urdf.xacro'),
                     stubs=('lms1xx/urdf/sick_lms1xx.urdf.xacro',), env_prefix='RIDGEBACK_')
    rec = physical_records(urdf, drop_frames=('arm_base',))
    bodies, joints, frames = rec['bodies'], rec['joints'], rec['frames']
    if rec['root'] != 'chassis_link':
        raise RuntimeError('Unexpected first massive body ' + rec['root'])
    # the 1 g dummy link of the IMU is a frame here, not a body
    imu = next(j for j in joints if j['body_b'] == 'imu_link')
    if imu['type'] != 'fixed' or imu['body_a'] != 'chassis_link':
        raise RuntimeError('Unexpected IMU attachment upstream')
    frames['imu_link'] = dict(body='chassis_link', T_body_frame=imu['T_AJ'])
    joints.remove(imu)
    bodies[:] = [b for b in bodies if b['id'] != 'imu_link']

    # ---------------------------------------------------------------- joints from the URDF
    limits, actuators = {}, []
    wheel_joint = {name: name + '_wheel' for name in RIDGEBACK_WHEELS}
    speed = 1.1 / .0759                                                 # published top speed over the wheel radius
    for j in joints:
        if j['type'] == 'fixed':
            j.pop('axis', None)
            continue
        u = j.pop('urdf_limit')
        continuous = j.pop('continuous')
        if j['id'] in ARM:
            limits[j['id']] = dict(lower=u['lower'], upper=u['upper'])
            actuators.append(dict(id=j['id'].replace('_joint', '_motor'), joint_id=j['id'], gear=1., unit='N m',
                                  effort_bounds=[-u['effort'], u['effort']], velocity_limit=u['velocity'],
                                  source='ur_description config/ur10e/joint_limits.yaml'))
        elif j['id'] == 'front_rocker':
            limits[j['id']] = dict(lower=u['lower'], upper=u['upper'])          # passive: no actuator
        elif j['id'] in wheel_joint.values():
            if not continuous:
                raise RuntimeError('Unexpected bounded wheel joint ' + j['id'])
            limits[j['id']] = dict(lower=-1e9, upper=1e9)
        else:
            raise RuntimeError('Unexpected moving joint ' + j['id'])
    if 'front_rocker' not in limits or next(j for j in joints if j['id'] == 'rear_rocker')['type'] != 'fixed':
        raise RuntimeError('Expected a revolute front rocker and a fixed rear rocker upstream')

    # ---------------------------------------------------------------- declared tooling (synthetic)
    clamp_h, plate, jaw_in, jaw_out, half_x, mid = ur10e_clamp(bodies, joints, frames)

    # ---------------------------------------------------------------- wheels
    # Upstream gives the wheel the inertia of a thin disc (ixx = izz = iyy / 2).  Nothing with an axial
    # extent, such as the rollers of the simulated plants, can be taken from a thin disc and leave a
    # physical hub.  The two transverse moments are therefore those of a solid cylinder of the upstream
    # mass, radius and width, the formula the wheel macro of clearpath_common uses; mass and axial moment
    # are the upstream values.
    wheel_r, wheel_w = .0759, .079
    for b in bodies:
        if b['id'].endswith('_wheel_link'):
            I = np.asarray(b['inertia_kg_m2'], float)
            if not np.allclose(I, np.diag([3.3212e-3, 6.6424e-3, 3.3212e-3]), rtol=0, atol=1e-12):
                raise RuntimeError('Unexpected wheel inertia upstream')
            transverse = b['mass_kg'] * (3. * wheel_r ** 2 + wheel_w ** 2) / 12.
            b['inertia_kg_m2'] = np.diag([transverse, I[1, 1], transverse]).tolist()
    wheel_records = []
    # Roller-axis (traction) direction at the ground, base frame.  The description used here has no
    # roller data; the directions are those of the Ridgeback mecanum wheel macro of clearpath_common
    # (Gazebo friction direction fdir1 "1 side 0": front left -1, front right 1, rear left 1, rear right -1),
    # which is the arrangement the mecanum drive controller of both Clearpath stacks assumes.
    side = dict(front_left=-1., front_right=1., rear_left=1., rear_right=-1.)
    for name in RIDGEBACK_WHEELS:
        g = np.array([1., side[name], 0.])
        wheel_records.append(dict(
            id=name, joint_id=wheel_joint[name], body=name + '_wheel_link', type='mecanum',
            radius_m=wheel_r, width_m=wheel_w, traction_direction_base=(g / np.linalg.norm(g)).tolist(),
            friction=.8, rollers=dict(count=10, max_radius_m=.0145, length_m=.104, mass_kg=.05,
                                       contact_time_constant_s=.010, contact_damping_ratio=.5,
                                       contact_stiffness_N_m=7.0e5, contact_damping_N_s_m=6.0e3),
            source='radius, width and mass: upstream ridgeback.urdf.xacro; traction direction: mecanum wheel macro of '
                   'clearpath_common (see provenance); friction, roller geometry and roller compliance: declared. The '
                   'compliance is given twice, for two contact models that are not equivalent: a soft-contact time constant '
                   'and damping ratio (MuJoCo), and a spring and damper per contact point (PhysX; one such spring deflects '
                   'about 0.8 mm under a quarter of the robot weight). Every run reports the base height against rigid wheels'))
        actuators.append(dict(id=name + '_wheel_motor', joint_id=wheel_joint[name], gear=1., unit='N m',
                              effort_bounds=[-40., 40.], velocity_limit=float(round(speed, 3)),
                              source='declared: 40 N m (no public drive torque); velocity limit from the published top '
                                     'speed 1.1 m/s and the wheel radius'))

    # ---------------------------------------------------------------- drive data (synthetic)
    drive = {}
    for jid, armature, damping in zip(ARM, [1.2, 1.2, .6, .12, .12, .12], [5., 5., 3., 1., 1., 1.]):
        drive[jid] = dict(armature=armature, damping=damping)
    for name in RIDGEBACK_WHEELS:
        drive[wheel_joint[name]] = dict(armature=.05, damping=.02)
    drive['front_rocker'] = dict(armature=0., damping=2.)

    # ---------------------------------------------------------------- primitive geometry (not CAD)
    dark, yellow = [.16, .17, .19, 1.], [.85, .72, .12, 1.]
    geometry = [
        box('chassis_link', [-.4663, -.395, .0038], [.4663, .395, .2755], dark, name='chassis'),
        box('chassis_link', [-.4658, -.395, .2737], [.4658, .395, .28], dark, name='deck'),
        box('chassis_link', [-.235, -.3966, .0052], [.235, .3966, .2202], yellow, collide=False, name='side_covers'),
        box('front_rocker_link', [-.06, -.2362, -.0275], [.06, .2362, .0275], dark, collide=False, name='front_rocker'),
        box('rear_rocker_link', [-.06, -.2362, -.0275], [.06, .2362, .0275], dark, collide=False, name='rear_rocker'),
    ] + ur10e_geometry(clamp_h, plate, jaw_in, jaw_out, half_x)

    seed = {jid: q for jid, q in zip(ARM, RIDGEBACK_ARM_SEED)}
    seed['front_rocker'] = 0.
    seed.update({wheel_joint[name]: 0. for name in RIDGEBACK_WHEELS})

    platform_mass = sum(b['mass_kg'] for b in bodies if not b['id'].startswith('arm_') and b['id'] != 'clamp')
    record = dict(
        schema=SCHEMA_ROBOT, name='Clearpath Ridgeback with UR10e', units='SI', floating_root='chassis_link',
        bodies=bodies, joints=joints, frames=frames, limits=limits, actuators=actuators, wheels=wheel_records,
        ports=ur10e_ports(half_x, mid),
        drive=drive, geometry=geometry, seed=dict(joints=seed), gravity_m_s2=[0., 0., -9.81],
        declared=dict(
            tooling='the clamp (1.2 kg), its inertia, the tool frame, the points at which it holds a section and its '
                    'compliance are synthetic. The wrist sensor is modelled in the tool flange with the range of the '
                    'force/torque sensor built into the UR10e (100 N, 10 N m; UR10e data sheet).',
            drive='armature and damping, including the damping of the rocker pivot, are synthetic values for '
                  'numerical realism, not UR or Clearpath data.',
            actuators='arm effort bounds are the ur_description UR10e limits; the wheel effort bound is declared, the '
                      'wheel speed limit follows from the published top speed of 1.1 m/s.',
            rocker='the front rocker is a passive revolute joint with limits of +-0.08726 rad and the rear rocker is '
                   'fixed, as in the upstream description used here (ridgeback_description 0.3.5). The newer '
                   'clearpath_common description fixes both rockers.',
            arm_mount='the arm base is on the link mid_mount with zero offset, the default of Clearpath\'s '
                      'ridgeback_ur10_e_description; upstream the offset is an environment variable, and the user '
                      'manual describes mounting holes at the front end of the plate.',
            geometry='boxes, cylinders and capsules placed from the kinematic frames, the published centres of mass '
                     'and the bounding boxes of the upstream chassis, deck, cover and rocker meshes; not CAD.',
            masses=f'upstream URDF values. The platform bodies sum to {platform_mass:.1f} kg (chassis 165.304, two rockers '
                   'of 10.267, four wheels of 2.3); the Ridgeback user manual of 2016 gives a weight of 135 kg. The base '
                   'link mass of the UR10e is marked "might be incorrect" upstream.',
            wheel_inertia='upstream gives each wheel the inertia of a thin disc (3.3212e-3, 6.6424e-3, 3.3212e-3 kg m2), '
                          'which leaves no physical hub once rollers are taken from it. The two transverse moments are '
                          'replaced by those of a solid cylinder of the upstream mass, radius and width (4.509e-3 kg m2), '
                          'the formula of the wheel macro of clearpath_common; mass and axial moment are unchanged.',
            omitted='the dummy mass of the IMU link (1 g), laser scanners and other accessories of the upstream '
                    'description; the UR control box (12 kg in the UR10e data sheet) and the arm cable.'),
        provenance=dict(
            ridgeback_description='https://github.com/ridgeback/ridgeback commit 7f2489993f1369906e808402e8f45e4c2aebf5b3 (BSD-3-Clause)',
            ur_description='https://github.com/UniversalRobots/Universal_Robots_ROS2_Description commit 6662e15f32c23c12ece57d0050aee9a716e4fc41 (BSD-3-Clause)',
            xacro_top='upstream/ridgeback_description/urdf/ridgeback.urdf.xacro with RIDGEBACK_URDF_EXTRAS = '
                      'inputs/xacro/ridgeback_ur10e_extras.urdf.xacro; the laser description the accessory mounts '
                      'include (package lms1xx) is replaced by an empty file, no accessory is enabled',
            references=dict(
                arm_mount='https://github.com/ridgeback/ridgeback_manipulation commit 748d284978610bab37984847700497ca35a71575, '
                          'ridgeback_ur_description/urdf/ridgeback_ur10_e_description.urdf.xacro (read, not copied)',
                traction_directions='https://github.com/clearpathrobotics/clearpath_common commit '
                                    '726d7fba367f18e53899d3ac03f0b276380ae847, clearpath_platform_description/urdf/r100/'
                                    'drivetrain/wheels/mecanum.urdf.xacro (read, not copied)',
                platform='Ridgeback user manual and integration pages, https://docs.clearpathrobotics.com/docs_robots/'
                         'indoor_robots/ridgeback/user_manual_ridgeback/ (maximum payload 100 kg, maximum speed 1.1 m/s); '
                         'weight 135 kg: Ridgeback user manual, 2016',
                arm='UR10e data sheet, https://www.universal-robots.com/manuals/latest/en/datasheets/ur10e/ (payload '
                    '12.5 kg, reach 1300 mm, force/torque sensor range 100 N and 10 N m, control box 12 kg)'),
            urdf_sha256=hashlib.sha256(urdf.encode()).hexdigest()))
    return record, urdf


# ---------------------------------------------------------------------- Clearpath Husky A300 with a UR10e
HUSKY_WHEELS = ['front_left', 'front_right', 'rear_left', 'rear_right']
# Arm posture of the record (the folded, elbow-up posture of the Ridgeback record).
HUSKY_ARM_SEED = [3.1416, -1.9, 2.3, -1.97, -1.5708, 0.]
# Husky A300 user manual, 40 Ah (two-battery) configuration, with reference to base_link.
HUSKY_MASS, HUSKY_COM = 78.5, [-.041, .003, .070]
# Wheel torque bound (declared; Clearpath publishes none): one axle alone holds the robot with its all-terrain payload
# (51.5 kg, user manual) on the published maximum grade of 30 degrees: 130 kg * 9.81 * sin(30 deg) * 0.1651 m / 2 = 52.6.
HUSKY_WHEEL_TORQUE = 52.5
HUSKY_TRACK = .566


def build_husky_a300_ur10e():
    urdf = make_urdf(top='inputs/xacro/husky_a300_ur10e.urdf.xacro', packages=('clearpath_platform_description', 'ur_description'),
                     stubs=('clearpath_control/config/a300/control.yaml',), mappings=dict(use_platform_controllers='false'))
    rec = physical_records(urdf, drop_frames=('arm_base', 'base_footprint'))
    bodies, joints, frames = rec['bodies'], rec['joints'], rec['frames']
    if rec['root'] != 'chassis_link':
        raise RuntimeError('Unexpected first massive body ' + rec['root'])
    if np.abs(np.asarray(frames['base_link']['T_body_frame']) - np.eye(4)).max() > 0.:
        raise RuntimeError('Expected chassis_link to coincide with base_link upstream')

    # ---------------------------------------------------------------- joints from the URDF
    limits, actuators = {}, []
    wheel_joint = {name: name + '_wheel_joint' for name in HUSKY_WHEELS}
    speed = 2. / .1651                                                  # published top speed over the wheel radius
    for j in joints:
        if j['type'] == 'fixed':
            j.pop('axis', None)
            continue
        u = j.pop('urdf_limit')
        continuous = j.pop('continuous')
        if j['id'] in ARM:
            limits[j['id']] = dict(lower=u['lower'], upper=u['upper'])
            actuators.append(dict(id=j['id'].replace('_joint', '_motor'), joint_id=j['id'], gear=1., unit='N m',
                                  effort_bounds=[-u['effort'], u['effort']], velocity_limit=u['velocity'],
                                  source='ur_description config/ur10e/joint_limits.yaml'))
        elif j['id'] in wheel_joint.values():
            if not continuous:
                raise RuntimeError('Unexpected bounded wheel joint ' + j['id'])
            limits[j['id']] = dict(lower=-1e9, upper=1e9)
        else:
            raise RuntimeError('Unexpected moving joint ' + j['id'])

    # ---------------------------------------------------------------- declared corrections of the platform description
    # (1) Track.  Upstream puts the wheel mounts of BOTH suspension beams at y = +0.0095 of the beam link; on the right
    #     side only the visual of the beam is turned, so the right wheels end up 19 mm closer to the centre line than the
    #     left ones (y = -0.2639 against +0.2829).  The user manual gives a track width of 566 mm, which is twice 0.2829
    #     to the millimetre: the right wheels are put at y = -0.2829.
    for name in HUSKY_WHEELS:
        j = next(j for j in joints if j['id'] == wheel_joint[name])
        T = np.asarray(j['T_AJ'], float)
        expected = [.256 if name.startswith('front') else -.256, .2829 if name.endswith('left') else -.2639, .02913]
        if not np.allclose(T[:3, 3], expected, rtol=0, atol=1e-9) or not np.allclose(T[:3, :3], np.eye(3), rtol=0, atol=1e-12):
            raise RuntimeError('Unexpected wheel position upstream: ' + name)
        if name.endswith('right'):
            T[1, 3] = -.2829
            j['T_AJ'] = T.tolist()
    for name in ('right_suspension_beam_front_mount', 'right_suspension_beam_rear_mount', 'front_right_motor_link',
                 'rear_right_motor_link', 'front_right_motor_mount', 'rear_right_motor_mount'):
        T = np.asarray(frames[name]['T_body_frame'], float)             # the frames that carry the right wheels move with them
        T[1, 3] -= .2829 - .2639
        frames[name]['T_body_frame'] = T.tolist()
    if abs(2 * .2829 - HUSKY_TRACK) > 5e-4:
        raise RuntimeError('The corrected track is not the one of the user manual')
    # (2) Mass and centre of mass.  Upstream gives the chassis 70 kg as a box centred in the chassis volume (80 kg with
    #     the four wheels, centre of mass on the centre line).  The user manual gives 78.5 kg for the two-battery
    #     configuration, the one of the sample configuration followed here, and its centre of mass in base_link.  The
    #     chassis takes the mass and the centre of mass that make the platform meet both; its inertia keeps the upstream
    #     box formula with the new mass.
    chassis = next(b for b in bodies if b['id'] == 'chassis_link')
    box_dim = np.array([.860, .378, .229559])
    I_up = 70. / 12. * np.array([box_dim[1] ** 2 + box_dim[2] ** 2, box_dim[0] ** 2 + box_dim[2] ** 2, box_dim[0] ** 2 + box_dim[1] ** 2])
    if chassis['mass_kg'] != 70. or not np.allclose(chassis['com_m'], [0., 0., box_dim[2] / 2], rtol=0, atol=1e-12) \
            or not np.allclose(chassis['inertia_kg_m2'], np.diag(I_up), rtol=0, atol=1e-9):
        raise RuntimeError('Unexpected chassis inertial upstream')
    wheel_bodies = [b for b in bodies if b['id'].endswith('_wheel_link')]
    if len(wheel_bodies) != 4 or any(b['mass_kg'] != 2.5 for b in wheel_bodies):
        raise RuntimeError('Unexpected wheel bodies upstream')
    m_chassis = HUSKY_MASS - sum(b['mass_kg'] for b in wheel_bodies)
    first = np.zeros(3)                                                 # first moment of the wheels in base_link
    for name in HUSKY_WHEELS:
        j = next(j for j in joints if j['id'] == wheel_joint[name])
        first += 2.5 * np.asarray(j['T_AJ'], float)[:3, 3]
    chassis.update(mass_kg=m_chassis, com_m=((HUSKY_MASS * np.asarray(HUSKY_COM) - first) / m_chassis).tolist(),
                   inertia_kg_m2=np.diag(I_up * m_chassis / 70.).tolist())

    # ---------------------------------------------------------------- declared tooling (synthetic)
    clamp_h, plate, jaw_in, jaw_out, half_x, mid = ur10e_clamp(bodies, joints, frames)

    # ---------------------------------------------------------------- wheels
    # Four conventional wheels on fixed axles: the base turns by skidding.  Their tangential ground forces are
    # friction, not rolling constraints (see teamc/skid.py); what the friction law and the plants need beyond
    # the upstream radius, width and mass is declared here.
    wheel_r, wheel_w = .1651, .1143
    for b in wheel_bodies:
        I = np.asarray(b['inertia_kg_m2'], float)
        transverse = 2.5 * (3. * wheel_r ** 2 + wheel_w ** 2) / 12.
        if not np.allclose(I, np.diag([transverse, 2.5 * wheel_r ** 2 / 2., transverse]), rtol=0, atol=1e-9):
            raise RuntimeError('Unexpected wheel inertia upstream')
    wheel_records = []
    for name in HUSKY_WHEELS:
        wheel_records.append(dict(
            id=name, joint_id=wheel_joint[name], body=name + '_wheel_link', type='fixed',
            radius_m=wheel_r, width_m=wheel_w, friction=.8,
            tyre=dict(slip_speed_m_s=.001,
                      stiffness_N_m=1.5e5, damping_N_s_m=1.5e3, relaxation_length_m=.03,
                      contact_time_constant_s=.015, contact_damping_ratio=1., friction_time_constant_s=.5,
                      contact_stiffness_N_m=2.0e5, contact_damping_N_s_m=4.0e3),
            source='radius, width and mass: upstream outdoor wheel macro (the user manual gives a nominal tyre diameter of '
                   '336 mm and an effective radius of 162.5 mm). Friction and every tyre value are declared: isotropic Coulomb '
                   'friction; slip_speed is the slip speed over which the friction force of the compiled model and of the '
                   'Pinocchio plant builds up; stiffness and damping are those of the tangential carcass deflection in the '
                   'explicit tyre model (backends/tyre.py: 1.6 mm at the friction limit under a quarter of the robot weight) '
                   'and the relaxation length is the rolled distance over which that deflection is renewed (stiffness times '
                   'relaxation length is the cornering stiffness, 4.5 kN/rad); '
                   'the contact time constant and damping ratio are the normal soft contact of MuJoCo (about 2 mm of '
                   'deflection), the friction time constant is the reference of its friction rows; contact stiffness and '
                   'damping are the normal spring of a compliant PhysX contact. In the plants the tyre is a sphere of the '
                   'wheel radius: it touches the ground at one point'))
        actuators.append(dict(id=name + '_wheel_motor', joint_id=wheel_joint[name], gear=1., unit='N m',
                              effort_bounds=[-HUSKY_WHEEL_TORQUE, HUSKY_WHEEL_TORQUE], velocity_limit=float(round(speed, 3)),
                              source=f'declared: {HUSKY_WHEEL_TORQUE} N m. Clearpath publishes no drive torque. The value is '
                                     'the torque at which one axle alone holds the robot with its all-terrain payload on the '
                                     'published maximum grade (78.5 kg + 51.5 kg, 30 degrees, tyre radius 0.1651 m: 52.6 N m '
                                     'per wheel, rounded down); with the four wheels sharing equally that climb needs 26 N m '
                                     'per wheel. The data sheet names that climb with a high-mounted payload and turning in '
                                     'place on a high-friction surface as the conditions of the all-terrain payload, and '
                                     'gives 1080 W continuous drive power for the four motors. Velocity limit from the '
                                     'published top speed 2 m/s and the wheel radius'))

    # ---------------------------------------------------------------- drive data (synthetic)
    drive = {}
    for jid, armature, damping in zip(ARM, [1.2, 1.2, .6, .12, .12, .12], [5., 5., 3., 1., 1., 1.]):
        drive[jid] = dict(armature=armature, damping=damping)
    for name in HUSKY_WHEELS:
        drive[wheel_joint[name]] = dict(armature=.10, damping=.05)

    # ---------------------------------------------------------------- primitive geometry (not CAD)
    dark, yellow, tyre = [.16, .17, .19, 1.], [.85, .72, .12, 1.], [.10, .10, .11, 1.]
    geometry = [
        box('chassis_link', [-.451, -.2305, 0.], [.4475, .2305, .2296], yellow, name='chassis'),
        box('chassis_link', [-.37, -.21, .2296], [.37, .21, .2423], dark, name='top_plate'),
        box('chassis_link', [.3697, -.1889, .0176], [.4977, .1889, .0501], dark, name='front_bumper'),
        box('chassis_link', [-.4977, -.1889, .0176], [-.3697, .1889, .0501], dark, name='rear_bumper'),
        # the space the wheels of one side sweep, from 30 mm above flat ground to their top: the wheels themselves
        # touch the ground, so they cannot be clearance shapes; these two boxes stand for them towards the scene
        box('chassis_link', [-.424, .229, -.106], [.424, .349, .197], dark, name='wheels_left'),
        box('chassis_link', [-.424, -.349, -.106], [.424, -.229, .197], dark, name='wheels_right'),
    ]
    for name in HUSKY_WHEELS:
        geometry.append(cylinder(name + '_wheel_link', [0, 0, 0], [0, 1, 0], wheel_r, wheel_w / 2, tyre, collide=False,
                                 name=name + '_tyre'))
    geometry += ur10e_geometry(clamp_h, plate, jaw_in, jaw_out, half_x)

    seed = {jid: q for jid, q in zip(ARM, HUSKY_ARM_SEED)}
    seed.update({wheel_joint[name]: 0. for name in HUSKY_WHEELS})

    record = dict(
        schema=SCHEMA_ROBOT, name='Clearpath Husky A300 with UR10e', units='SI', floating_root='chassis_link',
        bodies=bodies, joints=joints, frames=frames, limits=limits, actuators=actuators, wheels=wheel_records,
        ports=ur10e_ports(half_x, mid),
        drive=drive, geometry=geometry, seed=dict(joints=seed), gravity_m_s2=[0., 0., -9.81],
        declared=dict(
            tooling='the clamp (1.2 kg), its inertia, the tool frame, the points at which it holds a section and its '
                    'compliance are synthetic. The wrist sensor is modelled in the tool flange with the range of the '
                    'force/torque sensor built into the UR10e (100 N, 10 N m; UR10e data sheet).',
            drive='armature and damping are synthetic values for numerical realism, not UR or Clearpath data.',
            actuators='arm effort bounds are the ur_description UR10e limits; the wheel effort bound is declared (see the '
                      'wheel actuators), the wheel speed limit follows from the published top speed of 2 m/s.',
            track='the right wheels are at y = -0.2829 m. Upstream (clearpath_platform_description 2.9.17) they are at '
                  'y = -0.2639 m: the wheel mounts of both suspension beams are placed at y = +0.0095 of the beam, and on '
                  'the right side only the visual is turned. The user manual gives a track width of 566 mm (2 x 0.2829).',
            masses=f'the platform has the mass ({HUSKY_MASS} kg) and the centre of mass ({HUSKY_COM[0] * 1e3:.0f}, '
                   f'{HUSKY_COM[1] * 1e3:.0f}, {HUSKY_COM[2] * 1e3:.0f} mm in base_link) the user manual gives for the '
                   'two-battery (40 Ah) configuration. Upstream the chassis is a 70 kg box with its centre of mass in the '
                   f'middle of its volume (80 kg with the wheels); here it has {m_chassis:.1f} kg, the centre of mass that '
                   'puts the platform centre of mass where the manual has it, and the upstream box inertia scaled to that '
                   'mass. Wheel masses and inertias are upstream. The base link mass of the UR10e is marked "might be '
                   'incorrect" upstream.',
            wheels='conventional wheels on fixed axles (skid steering). Friction coefficient, the tyre data and the tyre '
                   'shape of the plants (a sphere of the wheel radius) are declared.',
            suspension='none: the user manual says the A300 has no suspension system, and upstream fixes the "suspension '
                       'beams" to the chassis.',
            arm_mount='the arm base is on the link top_plate_default_mount (centre of the top plate) with zero offset. '
                      'Clearpath\'s configuration accepts a universal_robots arm of type ur10e on any mount; its sample '
                      'for the Husky A200 uses default_mount, which on the A300 lies under the top plate.',
            geometry='boxes, cylinders and capsules placed from the kinematic frames and the bounding boxes of the upstream '
                     'chassis, bumper and wheel meshes; not CAD. Two boxes fixed to the chassis stand for the wheels of each '
                     'side towards the scene.',
            omitted='suspension beams, motors, bumpers and top plate have no mass upstream (the manual mass covers them); '
                    'the UR control box (12 kg in the UR10e data sheet) and the arm cable.'),
        provenance=dict(
            clearpath_platform_description='https://github.com/clearpathrobotics/clearpath_common commit '
                                           '726d7fba367f18e53899d3ac03f0b276380ae847 (BSD-3-Clause)',
            ur_description='https://github.com/UniversalRobots/Universal_Robots_ROS2_Description commit 6662e15f32c23c12ece57d0050aee9a716e4fc41 (BSD-3-Clause)',
            xacro_top='inputs/xacro/husky_a300_ur10e.urdf.xacro (use_platform_controllers:=false)',
            references=dict(
                configuration='https://github.com/clearpathrobotics/clearpath_config commit '
                              '4897cd1311deabadb887293eb300dcb31e74d666: sample/a300/a300_default.yaml (platform, bumpers, top '
                              'plate, battery S1P2), sample/manipulators/arms/universal_robots.yaml and the accepted ur_type '
                              'values in clearpath_config/manipulators/types/arms.py (read, not copied)',
                arm_macro='clearpath_common commit 726d7fba367f18e53899d3ac03f0b276380ae847, clearpath_manipulators_description/'
                          'urdf/arm/universal_robots.urdf.xacro (read, not copied)',
                skid_steering='clearpath_common commit 726d7fba367f18e53899d3ac03f0b276380ae847, clearpath_control/config/a300/'
                              'control/diff_4wd.yaml: wheel_separation 0.562, wheel_separation_multiplier 1.75, wheel_radius '
                              '0.1625 (read, not copied; compared with the compiled slip model in the README)',
                platform='Husky A300 user manual, https://docs.clearpathrobotics.com/docs_robots/outdoor_robots/husky/a300/'
                         'user_manual_husky/ (mass 78.5 kg with 40 Ah, centre of mass -41, 3, 70 mm in base_link, wheelbase '
                         '512 mm, track width 566 mm, effective tyre radius 162.5 mm, ground clearance 130 mm, maximum speed '
                         '2 m/s, maximum climb grade 30 degrees, allowable payload 101.5 kg for flat terrain with wide turns and '
                         '51.5 kg for all terrain, which includes turning in place on high-friction ground; four brushless '
                         'motors, 1080 W in all; no suspension system)',
                arm='UR10e data sheet, https://www.universal-robots.com/manuals/latest/en/datasheets/ur10e/ (payload '
                    '12.5 kg, reach 1300 mm, force/torque sensor range 100 N and 10 N m, control box 12 kg)'),
            urdf_sha256=hashlib.sha256(urdf.encode()).hexdigest()))
    return record, urdf


PLATFORMS = dict(rbrobout_ur30=build, ridgeback_ur10e=build_ridgeback_ur10e, husky_a300_ur10e=build_husky_a300_ur10e)


def main(argv=None):
    names = list(sys.argv[1:] if argv is None else argv) or list(PLATFORMS)
    out = ROOT / 'inputs/robots'
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        record, urdf = PLATFORMS[name]()
        (out / f'{name}.urdf').write_text(urdf, encoding='utf-8', newline='\n')      # the same bytes on every platform
        (out / f'{name}.json').write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8', newline='\n')
        mass = sum(b['mass_kg'] for b in record['bodies'])
        print(f"{name}: bodies {len(record['bodies'])}, joints {len(record['joints'])}, frames {len(record['frames'])}, "
              f"actuators {len(record['actuators'])}, mass {mass:.3f} kg")


if __name__ == '__main__':
    main()
