"""The MuJoCo plant against the compiled model."""
import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from backends.mujoco_plant import MujocoPlant  # noqa: E402
from mission.runner import run_mission  # noqa: E402
from teamc.io import load_team  # noqa: E402

HULL = (7, 32)          # coarse rollers: these tests are about bookkeeping, not ride quality


@pytest.mark.parametrize('team', ('team2', 'ridgeback2', 'husky2'))
def test_state_round_trip_and_mass_matrix(closed, team):
    c = closed(team)
    comp, q = c['comp'], c['carry']
    plant = MujocoPlant(comp, roller_hull=HULL)
    v = np.random.default_rng(0).uniform(-.2, .2, comp.tree.n)
    plant.reset(q, v)
    q1, v1 = plant.state()
    assert np.allclose(q1, q, atol=1e-10) and np.allclose(v1, v, atol=1e-10)
    m, d = plant.model, plant.data
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    # kinetic energy of the same motion in both models (rollers at rest)
    assert abs(.5 * d.qvel @ M @ d.qvel - .5 * v @ comp.tree.mass(q) @ v) < 1e-8 * (v @ comp.tree.mass(q) @ v)
    simulated = sum(m.body_mass[i] for i in range(m.nbody) if m.body_mocapid[i] < 0)        # the cart is moved kinematically
    assert abs(simulated - comp.tree.total_mass()) < 1e-9


@pytest.mark.parametrize('team', ('team2', 'ridgeback2', 'husky2'))
def test_wrist_sensor_reads_the_tool_weight_when_nothing_is_in_the_clamp(missions, team):
    comp = load_team(team)
    plant = MujocoPlant(comp, roller_hull=HULL)
    ctrl, rec, info = run_mission(plant, comp, missions(team), options=dict(duration=.8))    # the team stands still (settle phase)
    meas = plant.measure()
    kin = comp.tree.sweep(meas['q'])
    sensor = comp.robots['r1'].record['ports']['sensor']
    tool_mass = comp.tree.mass_kg[comp.tree.index[comp.robots['r1'].frames[sensor['frame']][0]]]
    for rid in comp.robots:
        w = meas['wrench'][rid]
        assert abs(np.linalg.norm(w[3:]) - tool_mass * 9.81) < 1.5                   # the sensor carries the clamp ...
        assert np.linalg.norm(ctrl.model.sensor_to_grasp(rid, w, meas["q"], kin)[3:]) < 1.5  # ... and passes nothing on (the arm still settles)


@pytest.mark.parametrize('team, carried', (('team2', 150.), ('ridgeback2', 55.), ('husky2', 55.)))
def test_compliant_clamp_sensor_equals_the_native_sensor(missions, team, carried):
    comp = load_team(team)
    plant = MujocoPlant(comp, roller_hull=HULL, latch='compliant', sensor='model')
    diffs = []

    def on_record(t, plant_, ctrl):
        for rid in comp.robots:
            diffs.append(plant_.native_sensor_wrench(rid) - plant_.sensor_wrench(rid))

    ctrl, rec, info = run_mission(plant, comp, missions(team), options=dict(start='navigate', start_l=1., duration=1.5),
                                  on_record=on_record)
    diffs = np.array(diffs)
    assert len(diffs) > 100
    assert np.sqrt((diffs ** 2).mean()) < .02                           # N and N m
    assert 0. < plant.latch_ratio < .6
    A = rec.arrays()
    sensed = np.abs(A['sensor:r1'][-20:, 3:]).max()
    assert sensed > carried                                             # about half of the beam weight plus the clamp
