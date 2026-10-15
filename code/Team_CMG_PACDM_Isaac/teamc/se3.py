"""SE(3) helpers shared by the team compiler.

The exponential, logarithm, adjoint and inverse-left-Jacobian are the ones of the
unchanged PACDM kernel (``vendor/pacdm_original.py``); this module only adds
small conveniences.  Conventions are those of the kernel:

* homogeneous 4x4 transforms ``aTb`` map coordinates of frame b to frame a;
* twists are angular-first ``[omega; v]`` and wrenches ``[moment; force]``;
* ``adj(T)`` maps a twist expressed in the frame of ``T`` to the frame ``T`` is
  expressed in, so ``adj(T).T`` maps wrenches the other way.
"""
from __future__ import annotations

import numpy as np

from vendor.pacdm_original import adj, exp, inv, inv_left, log, skew  # noqa: F401  (re-exported, unchanged)


def rot(axis, angle):
    """Rotation matrix about a unit axis (Rodrigues)."""
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1. - np.cos(angle)) * (K @ K)


def rpy(roll, pitch, yaw):
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot([0, 0, 1.], yaw) @ rot([0, 1., 0], pitch) @ rot([1., 0, 0], roll)


def transform(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = np.asarray(R, float)
    if p is not None:
        T[:3, 3] = np.asarray(p, float)
    return T


def origin(xyz=(0., 0., 0.), rpy_=(0., 0., 0.)):
    """Homogeneous transform of a URDF ``<origin xyz rpy>``."""
    return transform(rpy(*rpy_), xyz)


def rigid(value, name='transform'):
    """Validate a proper rigid transform and return it as an array."""
    T = np.asarray(value, float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)) or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-11, rtol=0):
        raise ValueError(f'{name}: expected a finite homogeneous transform')
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-9, rtol=0) or abs(np.linalg.det(R) - 1.) > 1e-9:
        raise ValueError(f'{name}: rotation is not proper orthonormal')
    return T


def chart_pose(c):
    """World pose of a body carried by the generated six-coordinate chart.

    ``c = [x, y, z, yaw, pitch, roll]``: world translation, then intrinsic ZYX
    rotations.  This is the chart convention of the Go2 and Kangaroo compilers.
    """
    x, y, z, yaw, pitch, roll = np.asarray(c, float)
    return transform(rot([0, 0, 1.], yaw) @ rot([0, 1., 0], pitch) @ rot([1., 0, 0], roll), [x, y, z])


def pose_chart(T, yaw_near=None):
    """Inverse of :func:`chart_pose` on the regular domain ``|pitch| < pi/2``."""
    R = np.asarray(T, float)[:3, :3]
    pitch = np.arcsin(np.clip(-R[2, 0], -1., 1.))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    if yaw_near is not None:
        yaw += 2. * np.pi * np.round((yaw_near - yaw) / (2. * np.pi))
    return np.r_[np.asarray(T, float)[:3, 3], yaw, pitch, roll]


def chart_rate_map(c):
    """``A`` with world angular velocity ``omega = A @ [yaw_dot, pitch_dot, roll_dot]``."""
    yaw, pitch = float(c[3]), float(c[4])
    return np.array([[0., -np.sin(yaw), np.cos(yaw) * np.cos(pitch)],
                     [0., np.cos(yaw), np.sin(yaw) * np.cos(pitch)],
                     [1., 0., -np.sin(pitch)]])


def crm(V):
    """Spatial motion cross product matrix for an angular-first twist."""
    X = np.zeros((6, 6))
    X[:3, :3] = skew(V[:3])
    X[3:, 3:] = X[:3, :3]
    X[3:, :3] = skew(V[3:])
    return X


def rotation_angle(R):
    """Angle of a rotation matrix (rad)."""
    return float(np.arccos(np.clip((np.trace(R) - 1.) / 2., -1., 1.)))


def quat_wxyz(R):
    """Unit quaternion (w, x, y, z) of a rotation matrix."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(np.asarray(R, float)).as_quat()
    return np.array([w, x, y, z])


def from_quat_wxyz(q):
    """Rotation matrix of a quaternion (w, x, y, z); the quaternion is normalized first."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1. - 2. * (y * y + z * z), 2. * (x * y - w * z), 2. * (x * z + w * y)],
                     [2. * (x * y + w * z), 1. - 2. * (x * x + z * z), 2. * (y * z - w * x)],
                     [2. * (x * z - w * y), 2. * (y * z + w * x), 1. - 2. * (x * x + y * y)]])


def adj_fast(T):
    """Same map as the kernel ``adj`` (checked in the tests), without block assembly."""
    R, p = T[:3, :3], T[:3, 3]
    X = np.zeros((6, 6))
    X[:3, :3] = R
    X[3:, 3:] = R
    X[3, :3] = p[1] * R[2] - p[2] * R[1]
    X[4, :3] = p[2] * R[0] - p[0] * R[2]
    X[5, :3] = p[0] * R[1] - p[1] * R[0]
    return X


def inv_fast(T):
    """Same map as the kernel ``inv`` (checked in the tests)."""
    out = np.zeros((4, 4))
    Rt = T[:3, :3].T
    out[:3, :3] = Rt
    out[:3, 3] = -Rt @ T[:3, 3]
    out[3, 3] = 1.
    return out
