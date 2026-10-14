"""The compliant clamp model and the wrist sensors it implies."""
import numpy as np
import pytest

from backends.latch import DEFAULTS, STABILITY_LIMIT, LatchSet, check_latch_stability, rotvec
from teamc.se3 import rot


def test_rotvec():
    for axis, angle in (([0, 0, 1.], .3), ([1., 0, 0], -1.2), ([0, 1., 0], 1e-9), ([.6, 0., .8], 2.5)):
        assert np.allclose(rotvec(rot(axis, angle)), np.asarray(axis) * angle, atol=1e-12)


def _closed(c):
    latches = LatchSet(c['comp'])
    for cid in latches.grasps:
        latches.close(cid, c['carry'])
    return latches


def test_no_load_at_the_closing_pose_and_restoring_load_after_a_displacement(closed):
    c = closed('team2')
    comp, q = c['comp'], c['carry'].copy()
    latches = _closed(c)
    zero = np.zeros(comp.tree.n)
    for load in latches.loads(q, zero).values():
        assert np.linalg.norm(load['force']) < 1e-6 and np.linalg.norm(load['moment']) < 1e-6
    pb = np.asarray(comp.payload['base'])
    q[pb[2]] -= .001                                                    # the payload sags by 1 mm
    loads = latches.loads(q, zero)
    for load in loads.values():
        assert abs(load['force'][2] - DEFAULTS['stiffness_N_m'] * .001) < 1e-6      # pulled up by both clamps
        assert abs(load['deflection'] - .001) < 1e-9
    v = zero.copy()
    v[pb[2]] = -.01                                                     # and it is falling: damping adds to the pull
    assert all(l['force'][2] > DEFAULTS['stiffness_N_m'] * .001 for l in latches.loads(q, v).values())


def test_sensor_reads_tool_weight_plus_the_clamp_load(closed):
    c = closed('team2')
    comp, q = c['comp'], c['carry'].copy()
    model = c['model']
    latches = _closed(c)
    zero = np.zeros(comp.tree.n)
    q[np.asarray(comp.payload['base'])[2]] -= .0005
    kin = comp.tree.sweep(q)
    loads = latches.loads(q, zero, kin)
    readings = latches.sensors(q, zero, zero, loads, kin=kin)
    for cid, load in loads.items():
        rid = comp.connections[cid]['robot']
        grasp = model.sensor_to_grasp(rid, readings[rid], q, kin)       # the compiled inverse of the sensor balance
        i1 = comp.tree.index[load['by']]
        R = (kin[0][i1] @ comp.connections[cid]['T1'])[:3, :3]
        assert np.allclose(R @ grasp[3:], load['force'], atol=1e-9)     # what the clamp applies to the payload
        assert np.allclose(R @ grasp[:3], load['moment'], atol=1e-9)


def test_stability_ratio_scales_with_the_timestep_and_guards_the_plant(closed):
    c = closed('team2')
    latches = _closed(c)
    r2, r1 = latches.stability_ratio(c['carry'], .002), latches.stability_ratio(c['carry'], .001)
    assert 0. < r1 < r2 < STABILITY_LIMIT
    stiff = LatchSet(c['comp'], {k: 5. * v for k, v in DEFAULTS.items()})
    for cid in stiff.grasps:
        stiff.close(cid, c['carry'])
    with pytest.raises(RuntimeError, match='smaller timestep'):
        check_latch_stability(stiff, c['carry'], .002)
