"""URDF -> physical robot records (bodies, undirected joints, named frames).

One-time, provenance-preserving extraction used by ``tools/build_inputs.py``.
It reads only what a physical record needs: link inertials, joint origins, axes
and limits.  Visual and collision geometry is not imported.

Links without mass are not physical bodies.  A massless link attached by a fixed
joint becomes a named frame of the nearest massive ancestor; a massless link
attached by a moving joint is rejected.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from .se3 import origin


def _floats(text, default):
    return np.array([float(x) for x in text.split()], float) if text else np.array(default, float)


def _origin(element):
    o = element.find('origin') if element is not None else None
    if o is None:
        return np.eye(4)
    return origin(_floats(o.get('xyz'), [0, 0, 0]), _floats(o.get('rpy'), [0, 0, 0]))


def read_urdf(text):
    """Return ``(root_link, links, joints)`` with transforms as arrays."""
    robot = ET.fromstring(text)
    links, joints, children = {}, [], set()
    for link in robot.findall('link'):
        inertial = link.find('inertial')
        mass, com, inertia = 0., np.zeros(3), np.zeros((3, 3))
        if inertial is not None:
            mass = float(inertial.find('mass').get('value'))
            T = _origin(inertial)
            i = inertial.find('inertia')
            I = np.array([[float(i.get('ixx')), float(i.get('ixy')), float(i.get('ixz'))],
                          [float(i.get('ixy')), float(i.get('iyy')), float(i.get('iyz'))],
                          [float(i.get('ixz')), float(i.get('iyz')), float(i.get('izz'))]])
            com, inertia = T[:3, 3], T[:3, :3] @ I @ T[:3, :3].T
        links[link.get('name')] = dict(mass=mass, com=com, inertia=inertia)
    for joint in robot.findall('joint'):
        kind = joint.get('type')
        if kind not in ('fixed', 'revolute', 'continuous', 'prismatic'):
            raise ValueError(f"{joint.get('name')}: unsupported URDF joint type {kind}")
        axis = joint.find('axis')
        limit = joint.find('limit')
        record = dict(id=joint.get('name'), type='revolute' if kind == 'continuous' else kind,
                      parent=joint.find('parent').get('link'), child=joint.find('child').get('link'),
                      T=_origin(joint), axis=_floats(axis.get('xyz') if axis is not None else None, [1, 0, 0]),
                      continuous=kind == 'continuous')
        if limit is not None:
            record.update(lower=float(limit.get('lower', 'nan')), upper=float(limit.get('upper', 'nan')),
                          effort=float(limit.get('effort', 'nan')), velocity=float(limit.get('velocity', 'nan')))
        joints.append(record)
        children.add(record['child'])
    roots = [name for name in links if name not in children]
    if len(roots) != 1:
        raise ValueError(f'Expected one URDF root link, found {roots}')
    return roots[0], links, joints


def physical_records(text, drop_frames=()):
    """Massive bodies, joints between them and named frames, from URDF text.

    Joints are returned with ``body_a``/``body_b`` and attachment frames
    ``T_AJ``/``T_BJ`` (the undirected physical-record convention of the
    RoboCompiler inputs).  ``frames`` maps every link name to
    ``(massive body, T_body_link)``.
    """
    root, links, joints = read_urdf(text)
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j['parent'], []).append(j)
    frames, bodies, out_joints = {}, [], []

    def visit(link, owner, T_owner_link):
        """``owner`` is the massive body this link belongs to (None above the first massive link)."""
        data = links[link]
        if data['mass'] > 0:
            owner, T_owner_link = link, np.eye(4)
            bodies.append(dict(id=link, mass_kg=data['mass'], com_m=data['com'].tolist(),
                               inertia_kg_m2=data['inertia'].tolist()))
        frames[link] = (owner, T_owner_link)
        for j in by_parent.get(link, []):
            child = links[j['child']]
            T_owner_joint = T_owner_link @ j['T']
            if child['mass'] > 0:
                if owner is None:
                    if j['type'] != 'fixed':
                        raise ValueError(f"{j['id']}: a moving joint above the first massive link is not supported")
                    visit(j['child'], None, T_owner_joint)
                    # the massless ancestors become frames of the first massive body
                    continue
                rec = dict(id=j['id'], type=j['type'], body_a=owner, body_b=j['child'],
                           T_AJ=T_owner_joint.tolist(), T_BJ=np.eye(4).tolist())
                if j['type'] != 'fixed':
                    rec['axis'] = j['axis'].tolist()
                    rec['urdf_limit'] = {k: j.get(k) for k in ('lower', 'upper', 'effort', 'velocity')}
                    rec['continuous'] = j['continuous']
                out_joints.append(rec)
                visit(j['child'], owner, T_owner_joint)
            else:
                if j['type'] != 'fixed':
                    raise ValueError(f"{j['id']}: massless link {j['child']} on a moving joint")
                visit(j['child'], owner, T_owner_joint)

    visit(root, None, np.eye(4))
    first = bodies[0]['id']
    resolved = {}
    # pose of the first massive body in the URDF root frame (frames above it are re-expressed in it)
    chain = {root: np.eye(4)}
    stack = [root]
    while stack:
        link = stack.pop()
        for j in by_parent.get(link, []):
            if j['type'] == 'fixed' or links[j['child']]['mass'] == 0:
                chain[j['child']] = chain[link] @ j['T']
                stack.append(j['child'])
    T_root_first = chain.get(first, np.eye(4))
    for name, (owner, T) in frames.items():
        if name in drop_frames:
            continue
        if owner is None:
            resolved[name] = dict(body=first, T_body_frame=(np.linalg.inv(T_root_first) @ T).tolist())
        elif name != owner:
            resolved[name] = dict(body=owner, T_body_frame=T.tolist())
    return dict(root=first, bodies=bodies, joints=out_joints, frames=resolved)
