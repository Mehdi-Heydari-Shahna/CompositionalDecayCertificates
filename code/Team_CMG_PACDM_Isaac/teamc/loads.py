"""Load sharing and internal wrenches of the grasp connections.

The grasp matrix ``G`` (6 x 6N) is the payload block of the compiled grasp rows:
``G @ lam`` is the generalized force the grasp wrenches ``lam`` apply to the
payload chart coordinates.  Rigid-body mechanics fixes only ``G @ lam``; how the
robots share it is a design choice.  It is declared here as a weighted
minimum-norm distribution

    lam = W^-1 G^T (G W^-1 G^T)^-1 w,        W = blkdiag(W_i / share_i),

with ``W_i = diag(1/moment_scale^2 (x3), 1/force_scale^2 (x3))``.  A larger share
makes a robot's wrench cheaper, a small moment scale keeps the load out of the
wrists.  The internal wrench of a set of grasp wrenches is defined relative to
this declared distribution: ``lam_int = lam - D G lam``.  It produces no net
force on the payload (``G lam_int = 0``).
"""
from __future__ import annotations

import numpy as np


class LoadSharing:
    def __init__(self, model, grasps, shares=None, moment_scale=20., force_scale=300.):
        self.model = model
        self.grasps = list(grasps)
        comp = model.comp
        shares = {g: comp.connections[g].get('share', 1.) for g in self.grasps} if shares is None else dict(shares)
        total = sum(shares[g] for g in self.grasps)
        self.shares = {g: shares[g] / total for g in self.grasps}
        self.scale = np.r_[np.full(3, moment_scale), np.full(3, force_scale)]

    def weights_inverse(self):
        return np.concatenate([self.shares[g] * self.scale ** 2 for g in self.grasps])

    def grasp_matrix(self, constraints):
        """``G`` with columns ordered as ``self.grasps`` (six per grasp)."""
        payload = self.model.payload
        return np.hstack([constraints.J[constraints.rows(g)][:, payload].T for g in self.grasps])

    def distribution(self, G):
        """``D`` (6N x 6) with ``lam = D w`` for a payload generalized force ``w``."""
        Wi = self.weights_inverse()
        A = (G * Wi) @ G.T
        return (Wi[:, None] * G.T) @ np.linalg.inv(A)

    def distribute(self, G, w, internal=None):
        """Grasp wrenches carrying ``w`` (and an optional internal wrench), per grasp id."""
        lam = self.distribution(G) @ np.asarray(w, float)
        if internal is not None:
            lam = lam + self.internal_projector(G) @ np.asarray(internal, float)
        return {g: lam[6 * k:6 * k + 6] for k, g in enumerate(self.grasps)}

    def internal_projector(self, G):
        return np.eye(G.shape[1]) - self.distribution(G) @ G

    def decompose(self, G, wrenches):
        """Resultant on the payload and internal part of grasp wrenches given per grasp id."""
        lam = np.concatenate([np.asarray(wrenches[g], float) for g in self.grasps])
        w = G @ lam
        internal = lam - self.distribution(G) @ w
        return dict(resultant=w, internal={g: internal[6 * k:6 * k + 6] for k, g in enumerate(self.grasps)},
                    internal_force_norm=float(np.linalg.norm(internal.reshape(-1, 6)[:, 3:])),
                    internal_moment_norm=float(np.linalg.norm(internal.reshape(-1, 6)[:, :3])))

    def internal_basis(self, G):
        """Orthonormal basis of the internal-wrench space, dimension 6(N-1) for N rigid grasps."""
        _, s, Vt = np.linalg.svd(G)
        rank = int(np.sum(s > 1e-10 * s[0]))
        return Vt[rank:].T


def world_wrench(T_frame, lam):
    """Force (world axes) and moment about the frame origin (world axes) of a frame-axis wrench."""
    R = np.asarray(T_frame, float)[:3, :3]
    return R @ lam[3:], R @ lam[:3]
