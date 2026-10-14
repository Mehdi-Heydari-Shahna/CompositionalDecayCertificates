"""The vendored PACDM kernel is the published one, and the fast helpers are the same maps."""
import hashlib
from pathlib import Path

import numpy as np

from teamc import se3
from vendor import pacdm_original as kernel

ROOT = Path(__file__).resolve().parents[1]


def test_kernel_is_byte_identical():
    assert hashlib.md5((ROOT / 'vendor' / 'pacdm_original.py').read_bytes()).hexdigest() == 'ef147635e7d799f5be94772ff0fa08c8'


def test_fast_adjoint_and_inverse_equal_kernel():
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = kernel.exp(rng.normal(size=6))
        assert np.allclose(se3.adj_fast(T), kernel.adj(T), atol=1e-14)
        assert np.allclose(se3.inv_fast(T), kernel.inv(T), atol=1e-14)


def test_chart_round_trip_and_rate_map():
    rng = np.random.default_rng(2)
    for _ in range(20):
        c = np.r_[rng.normal(size=3), rng.uniform(-3, 3), rng.uniform(-1.2, 1.2), rng.uniform(-3, 3)]
        assert np.allclose(se3.pose_chart(se3.chart_pose(c), c[3]), c, atol=1e-12)
        rates = rng.normal(size=3)
        h = 1e-6
        R0, R1 = se3.chart_pose(c)[:3, :3], se3.chart_pose(c + h * np.r_[0, 0, 0, rates])[:3, :3]
        W = (R1 - R0) / h @ R0.T                                   # skew of the world angular velocity
        omega = np.array([W[2, 1], W[0, 2], W[1, 0]])
        assert np.allclose(omega, se3.chart_rate_map(c) @ rates, atol=1e-5)


def test_quaternion_helpers():
    rng = np.random.default_rng(3)
    for _ in range(10):
        R = kernel.exp(np.r_[rng.normal(size=3), 0, 0, 0])[:3, :3]
        assert np.allclose(se3.from_quat_wxyz(se3.quat_wxyz(R)), R, atol=1e-12)
