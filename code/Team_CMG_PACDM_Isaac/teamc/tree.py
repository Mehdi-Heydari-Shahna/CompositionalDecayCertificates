"""World-frame spatial rigid-tree kinematics and dynamics (NumPy only).

Feedforward and reference model of the team compiler.  It is not a state
integrator and imports no simulator.  It is checked against Pinocchio and the
repository's ``NumpyTree`` in ``tests/``.

Input: a directed CMG tree (``root_body``, ``bodies``, ``joints`` with
``base_body``/``follower_body``/``T_BJ``/``T_FJ``/``axis``, ``coordinate_ids``),
scalar revolute/prismatic joints and fixed joints.  Body inertias are about the
centre of mass in body axes (``inertia_kg_m2``).

All spatial quantities are expressed in world axes about the world origin and are
angular-first: twists ``[omega; v_O]``, wrenches ``[moment_O; force]``.
"""
from __future__ import annotations

import numpy as np

from .se3 import adj, crm, inv, skew

FIXED, REVOLUTE, PRISMATIC = 0, 1, 2
_I3 = np.eye(3)
KIND = {'fixed': FIXED, 'revolute': REVOLUTE, 'prismatic': PRISMATIC}


def wrench_to_world(T_frame, w_local):
    """World spatial wrench of a wrench given in a frame (moment about its origin, its axes)."""
    return adj(inv(np.asarray(T_frame, float))).T @ np.asarray(w_local, float)


def wrench_to_frame(T_frame, w_world):
    """Wrench in a frame (moment about its origin, its axes) of a world spatial wrench."""
    return adj(np.asarray(T_frame, float)).T @ np.asarray(w_world, float)


class Tree:
    def __init__(self, cmg):
        self.cmg = cmg
        self.ids = list(cmg['coordinate_ids'])
        self.n = len(self.ids)
        self.gravity = np.asarray(cmg.get('gravity_m_s2', [0., 0., -9.81]), float)
        records = {b['id']: b for b in cmg['bodies']}
        root = cmg['root_body']
        self.names = [root]
        index = {root: 0}
        remaining = list(cmg['joints'])
        edges = []
        while remaining:
            progressed = False
            for j in remaining[:]:
                if j['base_body'] not in index:
                    continue
                if j['follower_body'] in index:
                    raise ValueError(f"{j['id']}: body {j['follower_body']} has two parents (not a tree)")
                index[j['follower_body']] = len(self.names)
                self.names.append(j['follower_body'])
                edges.append(j)
                remaining.remove(j)
                progressed = True
            if not progressed:
                raise ValueError('CMG tree is disconnected: ' + ', '.join(j['id'] for j in remaining[:5]))
        if set(index) != set(records):
            raise ValueError('CMG contains bodies outside the tree: ' + ', '.join(sorted(set(records) - set(index))[:5]))
        self.index = index
        nb = self.nb = len(self.names)
        self.parent = np.zeros(nb, int)
        self.kind = np.zeros(nb, int)
        self.coord = -np.ones(nb, int)
        self.enter = np.tile(np.eye(4), (nb, 1, 1))
        self.leave = np.tile(np.eye(4), (nb, 1, 1))
        self.axis = np.zeros((nb, 3))
        self.K = np.zeros((nb, 3, 3))
        self.K2 = np.zeros((nb, 3, 3))
        self.joint_id = [None] * nb
        self.body_of_joint = {}
        for i, j in enumerate(edges, 1):
            self.parent[i] = index[j['base_body']]
            self.kind[i] = KIND[j['type']]
            self.enter[i] = np.asarray(j['T_BJ'], float)
            self.leave[i] = inv(np.asarray(j['T_FJ'], float))
            self.joint_id[i] = j['id']
            self.body_of_joint[j['id']] = i
            if j['type'] != 'fixed':
                axis = np.asarray(j['axis'], float)
                if abs(np.linalg.norm(axis) - 1.) > 1e-9:
                    raise ValueError(f"{j['id']}: axis must be a unit vector")
                self.axis[i] = axis
                self.K[i] = skew(axis)
                self.K2[i] = self.K[i] @ self.K[i]
                self.coord[i] = self.ids.index(j['id'])
        if sorted(c for c in self.coord if c >= 0) != list(range(self.n)):
            raise ValueError('coordinate_ids must be exactly the moving joints of the tree')
        self.mass_kg = np.zeros(nb)
        self.com = np.zeros((nb, 3))
        self.inertia = np.zeros((nb, 6, 6))          # body-origin spatial inertia, body axes
        for i, name in enumerate(self.names):
            b = records[name]
            m, c, Ic = float(b['mass_kg']), np.asarray(b['com_m'], float), np.asarray(b['inertia_kg_m2'], float)
            if m < 0 or not np.allclose(Ic, Ic.T, atol=1e-10) or (m > 0 and np.linalg.eigvalsh(Ic).min() < -1e-12):
                raise ValueError(f'{name}: invalid mass or inertia')
            C = skew(c)
            self.mass_kg[i], self.com[i] = m, c
            self.inertia[i] = np.block([[Ic + m * C @ C.T, m * C], [m * C.T, m * np.eye(3)]])
        self.massive = np.flatnonzero(self.mass_kg > 0)
        arm = cmg.get('armature', {})
        self.armature = np.array([float(arm.get(k, 0.)) for k in self.ids]) if isinstance(arm, dict) \
            else np.asarray(arm, float)
        # ancestors (movable joints on the path to the root) of every body, root first
        self.support = [[] for _ in range(nb)]
        for i in range(1, nb):
            self.support[i] = self.support[self.parent[i]] + ([i] if self.coord[i] >= 0 else [])
        # evaluation tables: joint frames split into rotation and translation, identity flags,
        # and the ancestor matrix (row i marks body i and all its ancestors)
        self._Re, self._pe = self.enter[:, :3, :3].copy(), self.enter[:, :3, 3].copy()
        self._Rl, self._pl = self.leave[:, :3, :3].copy(), self.leave[:, :3, 3].copy()
        eye = np.eye(4)
        self._enter_id = [bool(np.array_equal(self.enter[i], eye)) for i in range(nb)]
        self._leave_id = [bool(np.array_equal(self.leave[i], eye)) for i in range(nb)]
        self._steps = [(i, int(self.parent[i]), int(self.kind[i]), int(self.coord[i]), self._enter_id[i], self._leave_id[i])
                       for i in range(1, nb)]
        self.ancestors = np.zeros((nb, nb))
        for i in range(nb):
            j = i
            while True:
                self.ancestors[i, j] = 1.
                if j == 0:
                    break
                j = self.parent[j]
        self._moving = np.flatnonzero(self.coord >= 0)
        self._moving_coord = self.coord[self._moving]
        self._inertia_massive = self.inertia[self.massive]

    # ------------------------------------------------------------------ kinematics
    def _vec(self, x, name):
        x = np.asarray(x, float)
        if x.shape != (self.n,) or not np.all(np.isfinite(x)):
            raise ValueError(f'{name}: expected {self.n} finite values')
        return x

    def sweep(self, q):
        """World poses ``T`` (nb,4,4) and world joint twists ``S`` (nb,6) of every body."""
        q = self._vec(q, 'q')
        T = np.zeros((self.nb, 4, 4))
        T[:, 3, 3] = 1.
        S = np.zeros((self.nb, 6))
        Rw = [None] * self.nb
        pw = [None] * self.nb
        Rw[0], pw[0] = _I3, np.zeros(3)
        Re, pe, Rl, pl, axis, K, K2 = self._Re, self._pe, self._Rl, self._pl, self.axis, self.K, self.K2
        sin, cos = np.sin(q), np.cos(q)
        for i, parent, kind, k, enter_id, leave_id in self._steps:
            R, p = Rw[parent], pw[parent]
            if not enter_id:
                p = p + R @ pe[i]
                R = R @ Re[i]
            if kind == REVOLUTE:
                w = R @ axis[i]
                S[i, 0], S[i, 1], S[i, 2] = w
                S[i, 3] = p[1] * w[2] - p[2] * w[1]
                S[i, 4] = p[2] * w[0] - p[0] * w[2]
                S[i, 5] = p[0] * w[1] - p[1] * w[0]
                R = R @ (_I3 + sin[k] * K[i] + (1. - cos[k]) * K2[i])
            elif kind == PRISMATIC:
                w = R @ axis[i]
                S[i, 3:] = w
                p = p + w * q[k]
            if not leave_id:
                p = p + R @ pl[i]
                R = R @ Rl[i]
            Rw[i], pw[i] = R, p
            T[i, :3, :3] = R
            T[i, :3, 3] = p
        T[0, :3, :3] = _I3
        return T, S

    def poses(self, q):
        """Dictionary body name -> world pose."""
        T, _ = self.sweep(q)
        return dict(zip(self.names, T))

    def twist_maps(self, q, S=None):
        """``E`` (nb,6,n): world spatial twist of every body is ``E[i] @ v``."""
        if S is None:
            _, S = self.sweep(q)
        E = np.zeros((self.nb, 6, self.n))
        for i in range(1, self.nb):
            E[i] = E[self.parent[i]]
            if self.coord[i] >= 0:
                E[i, :, self.coord[i]] += S[i]
        return E

    def frame_jacobian(self, q, body, T_body_frame=None, kinematics=None):
        """Pose of a body-fixed frame and its 6xn Jacobian of the twist expressed in that frame."""
        T, S = kinematics if kinematics is not None else self.sweep(q)
        i = self.index[body]
        F = T[i] if T_body_frame is None else T[i] @ np.asarray(T_body_frame, float)
        J = np.zeros((6, self.n))
        for k in self.support[i]:
            J[:, self.coord[k]] = S[k]
        return F, adj(inv(F)) @ J

    # ------------------------------------------------------------------ dynamics
    def _world_inertia(self, T):
        """Spatial inertia of every body about the world origin, world axes (zero for massless frames)."""
        out = np.zeros((self.nb, 6, 6))
        m = self.massive
        Rt = np.transpose(T[m, :3, :3], (0, 2, 1))
        p = T[m, :3, 3]
        P = np.zeros((len(m), 3, 3))
        P[:, 0, 1], P[:, 0, 2], P[:, 1, 0] = -p[:, 2], p[:, 1], p[:, 2]
        P[:, 1, 2], P[:, 2, 0], P[:, 2, 1] = -p[:, 0], -p[:, 1], p[:, 0]
        X = np.zeros((len(m), 6, 6))                 # world twist -> body-frame twist
        X[:, :3, :3] = Rt
        X[:, 3:, 3:] = Rt
        X[:, 3:, :3] = -Rt @ P
        out[m] = np.transpose(X, (0, 2, 1)) @ self._inertia_massive @ X
        return out

    def rnea(self, q, v, a, fext=None, gravity=True, return_forces=False, kinematics=None):
        """Generalized effort ``M a + h - J^T fext`` (armature included).

        ``fext`` maps body name -> world spatial wrench applied to that body.
        With ``return_forces`` the wrench transmitted across every body's parent
        joint (world spatial, acting on the body's subtree) is returned as well;
        it includes the constrained directions of fixed joints.
        """
        v, a = self._vec(v, 'v'), self._vec(a, 'a')
        T, S = kinematics if kinematics is not None else self.sweep(q)
        Iw = self._world_inertia(T)
        mv, mc = self._moving, self._moving_coord
        Sv = np.zeros((self.nb, 6))
        Sv[mv] = S[mv] * v[mc][:, None]
        V = self.ancestors @ Sv                       # world twists of all bodies
        Vp = V[self.parent]
        # joint terms of the acceleration: S a + (V_parent x S) v
        term = np.zeros((self.nb, 6))
        term[mv] = S[mv] * a[mc][:, None]
        term[:, :3] += np.cross(Vp[:, :3], Sv[:, :3])
        term[:, 3:] += np.cross(Vp[:, :3], Sv[:, 3:]) + np.cross(Vp[:, 3:], Sv[:, :3])
        A = self.ancestors @ term
        if gravity:
            A[:, 3:] -= self.gravity
        m = self.massive
        IV = np.einsum('nij,nj->ni', Iw[m], V[m])
        F = np.zeros((self.nb, 6))
        F[m] = np.einsum('nij,nj->ni', Iw[m], A[m])
        # - crm(V)^T (I V): moment part  w x n + v x f, force part  w x f
        F[m, :3] += np.cross(V[m, :3], IV[:, :3]) + np.cross(V[m, 3:], IV[:, 3:])
        F[m, 3:] += np.cross(V[m, :3], IV[:, 3:])
        if fext:
            for name, w in fext.items():
                F[self.index[name]] -= np.asarray(w, float)
        F = self.ancestors.T @ F                      # wrench transmitted to each subtree
        tau = np.zeros(self.n)
        tau[mc] = np.einsum('ni,ni->n', S[mv], F[mv])
        tau += self.armature * a
        return (tau, F, T) if return_forces else tau

    def bias(self, q, v, kinematics=None):
        """Coriolis, centrifugal and gravity effort ``h(q, v)``."""
        return self.rnea(q, v, np.zeros(self.n), kinematics=kinematics)

    def mass(self, q, kinematics=None):
        """Joint-space inertia (composite rigid body algorithm), armature included."""
        T, S = kinematics if kinematics is not None else self.sweep(q)
        Ic = self._world_inertia(T)
        for i in range(self.nb - 1, 0, -1):
            Ic[self.parent[i]] += Ic[i]
        M = np.zeros((self.n, self.n))
        for i in range(1, self.nb):
            k = self.coord[i]
            if k < 0:
                continue
            F = Ic[i] @ S[i]
            M[k, k] = S[i] @ F
            for j in self.support[self.parent[i]]:
                M[k, self.coord[j]] = M[self.coord[j], k] = S[j] @ F
        M[np.diag_indices(self.n)] += self.armature
        return M

    def joint_wrench(self, q, v, a, joint_id, T_body_frame=None, fext=None):
        """Wrench transmitted across a tree joint, expressed in a frame fixed to its follower body.

        This is the wrench the base-side body applies to the follower-side subtree
        (moment about the frame origin, frame axes).  It is defined for fixed joints
        too, which is what a wrist force/torque sensor measures.
        """
        _, F, T = self.rnea(q, v, a, fext=fext, return_forces=True)
        i = self.body_of_joint[joint_id]
        frame = T[i] if T_body_frame is None else T[i] @ np.asarray(T_body_frame, float)
        return wrench_to_frame(frame, F[i])

    def total_mass(self):
        return float(self.mass_kg.sum())

    def com_world(self, q, bodies=None):
        T, _ = self.sweep(q)
        idx = self.massive if bodies is None else [self.index[b] for b in bodies if self.mass_kg[self.index[b]] > 0]
        m = self.mass_kg[idx]
        c = np.array([T[i, :3, 3] + T[i, :3, :3] @ self.com[i] for i in idx])
        return (m[:, None] * c).sum(axis=0) / m.sum(), float(m.sum())
