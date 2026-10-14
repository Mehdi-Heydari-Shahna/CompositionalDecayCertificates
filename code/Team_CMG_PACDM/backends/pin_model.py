"""Pinocchio model of the compiled team graph.

Built directly from the compiled graph through the Pinocchio model API (no URDF):
every scalar joint of the world-rooted tree becomes a one-degree-of-freedom
Pinocchio joint in the compiled coordinate order, bodies behind fixed joints are
attached to the nearest moving ancestor.  Because every joint has one coordinate,
the Pinocchio configuration and velocity vectors are the compiled ones.

Used as an independent rigid-body engine: reference for ``teamc.tree`` and
dynamics of the ideal-constraint plant.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from teamc.se3 import inv


def _se3(T):
    T = np.asarray(T, float)
    return pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())


class PinTeam:
    def __init__(self, comp):
        cmg = comp.cmg
        self.comp = comp
        ids = cmg['coordinate_ids']
        bodies = {b['id']: b for b in cmg['bodies']}
        children = {}
        for j in cmg['joints']:
            children.setdefault(j['base_body'], []).append(j)
        model = pin.Model()
        model.name = cmg.get('name', 'team') or 'team'
        model.gravity.linear = np.asarray(cmg['gravity_m_s2'], float)
        # every body: (pinocchio joint id, pose of the body frame in that joint frame)
        self.body = {cmg['root_body']: (0, np.eye(4))}
        # moving joints must be added in the compiled coordinate order
        order = {name: k for k, name in enumerate(ids)}
        pending = [cmg['root_body']]
        added = {}
        queue = []
        while pending:
            parent = pending.pop(0)
            for j in children.get(parent, []):
                queue.append(j)
                pending.append(j['follower_body'])
        moving = sorted([j for j in queue if j['type'] != 'fixed'], key=lambda j: order[j['id']])
        fixed = [j for j in queue if j['type'] == 'fixed']
        done = False
        while not done:                                  # resolve in passes: a joint needs its parent body placed
            done = True
            for j in list(moving) + list(fixed):
                if j['id'] in added or j['base_body'] not in self.body:
                    continue
                pj, B_p = self.body[j['base_body']]
                T_BJ, T_FJ = np.asarray(j['T_BJ'], float), np.asarray(j['T_FJ'], float)
                b = bodies[j['follower_body']]
                inertia = None
                if b['mass_kg'] > 0:
                    inertia = pin.Inertia(float(b['mass_kg']), np.asarray(b['com_m'], float), np.asarray(b['inertia_kg_m2'], float))
                if j['type'] == 'fixed':
                    B = B_p @ T_BJ @ inv(T_FJ)
                    jid = pj
                else:
                    if order[j['id']] != model.nv:       # keep the compiled order: wait for earlier coordinates
                        continue
                    axis = np.asarray(j['axis'], float)
                    jm = pin.JointModelRevoluteUnaligned(axis) if j['type'] == 'revolute' else pin.JointModelPrismaticUnaligned(axis)
                    jid = model.addJoint(pj, jm, _se3(B_p @ T_BJ), j['id'])
                    B = inv(T_FJ)
                if inertia is not None:
                    model.appendBodyToJoint(jid, inertia, _se3(B))
                self.body[j['follower_body']] = (jid, B)
                added[j['id']] = jid
                done = False
        if len(added) != len(queue) or model.nv != len(ids):
            raise RuntimeError('pinocchio model: the compiled tree could not be reproduced')
        if [model.names[i] for i in range(1, model.njoints)] != list(ids):
            raise RuntimeError('pinocchio model: coordinate order differs from the compiled order')
        model.armature[:] = [float(cmg['armature'].get(k, 0.)) for k in ids]
        lim = cmg['limits']
        model.lowerPositionLimit[:] = [lim[k]['lower'] for k in ids]
        model.upperPositionLimit[:] = [lim[k]['upper'] for k in ids]
        self.model = model
        self.data = model.createData()
        self.n = model.nv
        self.frames = {}

    # ------------------------------------------------------------------ frames
    def frame(self, body, T_body_frame=None, name=None):
        """Operational frame fixed to a compiled body (created once, cached)."""
        T = np.eye(4) if T_body_frame is None else np.asarray(T_body_frame, float)
        key = name or (body, T.tobytes())
        if key not in self.frames:
            jid, B = self.body[body]
            fid = self.model.addFrame(pin.Frame(f'f{len(self.frames)}_{body}', jid, 0, _se3(B @ T), pin.FrameType.OP_FRAME))
            self.frames[key] = fid
            self.data = self.model.createData()
        return self.frames[key]

    # ------------------------------------------------------------------ dynamics
    def terms(self, q, v):
        """Mass matrix (armature included) and bias ``h`` (Coriolis, centrifugal, gravity)."""
        M = pin.crba(self.model, self.data, q)
        M = np.triu(M) + np.triu(M, 1).T
        h = pin.nonLinearEffects(self.model, self.data, q, v)
        return M, h.copy()

    def kinematics(self, q, v=None, a=None):
        if v is None:
            pin.forwardKinematics(self.model, self.data, q)
        elif a is None:
            pin.forwardKinematics(self.model, self.data, q, v)
        else:
            pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)

    def jacobian(self, q, fid, frame=pin.LOCAL):
        """6 x n frame Jacobian, linear rows first (Pinocchio convention)."""
        return pin.computeFrameJacobian(self.model, self.data, q, fid, frame)

    def pose(self, fid):
        return self.data.oMf[fid].homogeneous
