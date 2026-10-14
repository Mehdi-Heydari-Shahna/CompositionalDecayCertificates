"""Simulation runs of tens of seconds to minutes (``pytest -m slow``)."""
import numpy as np
import pytest

pytestmark = pytest.mark.slow


def _mission(name):
    from teamc.io import INPUTS, load_team, read_json
    return load_team(name), read_json(INPUTS / 'missions' / f'{name}.json')


@pytest.mark.parametrize('team', ('team2', 'ridgeback2', 'husky2'))
def test_pickup_in_mujoco_takes_the_weight_gradually(team):
    pytest.importorskip('mujoco')
    from backends.mujoco_plant import MujocoPlant
    from mission.audit import audit
    from mission.runner import run_mission
    comp, mission = _mission(team)
    plant = MujocoPlant(comp)
    ctrl, rec, info = run_mission(plant, comp, mission, options=dict(stop_after='extract'))
    events = [e['event'] for e in info['events']]
    assert 'weight_taken' in events and info['saturated_steps'] == 0
    report, _ = audit(comp, mission, rec.arrays(), info, rec.contacts, model=ctrl.model)
    transfer = report['transfer']['pickup']
    weight = mission['payload']['mass_kg'] * 9.81
    assert transfer['closure_error_N']['max'] < .05 * weight          # pad loads + sensed load = weight, throughout the hand-over
    assert np.max(transfer['carried_fraction']) > .97
    unmet = [k for k, c in report['bounds']['checks'].items() if not c['ok'] and k not in ('mission_complete', 'supports_verified')]
    assert not unmet


def test_the_team_does_not_lift_when_the_clamps_hold_nothing():
    pytest.importorskip('mujoco')
    from backends.mujoco_plant import MujocoPlant
    from mission.runner import run_mission

    class NoGrip(MujocoPlant):
        def latch(self, cid):
            T2 = super().latch(cid)
            self.data.eq_active[self._weld[cid]] = 0                   # the clamp closes on nothing
            return T2

    comp, mission = _mission('team2')
    plant = NoGrip(comp)
    ctrl, rec, info = run_mission(plant, comp, mission, max_time=80.)
    assert info['status'] == 'held: weight not taken'
    assert any(e['event'] == 'weight_not_taken' for e in info['events'])
    A = rec.arrays()
    pb = np.asarray(comp.payload['base'])
    assert np.abs(A['q'][-1, pb] - A['q'][0, pb]).max() < 2e-3         # the beam is still on its rack supports


@pytest.mark.parametrize('team', ('team2', 'ridgeback2', 'husky2'))
def test_pinocchio_interface_validation(tmp_path, team):
    pytest.importorskip('pinocchio')
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'workflows' / 'pinocchio' / 'validate_interfaces.py'
    spec = importlib.util.spec_from_file_location('validate_interfaces', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(['--team', team, '--out', str(tmp_path)]) == 0


def test_three_skid_steered_bases_rock_on_rigid_ground_and_not_on_their_tyres():
    """The turn in front of the supports, three Huskies holding one beam (DESIGN.md, wheel loads and friction in a skid turn).

    On rigid ground the wheel loads follow the friction forces without a time scale of their own, and the bases rock
    from one diagonal to the other.  On the normal compliance of the declared tyres the same turn is steady.
    """
    pytest.importorskip('pinocchio')
    from backends.pin_plant import PinPlant
    from mission.runner import run_mission
    comp, mission = _mission('husky3')
    peak, low = {}, {}
    for ground in ('rigid', 'tyres'):
        plant = PinPlant(comp, skid_ground=ground)
        ctrl, rec, info = run_mission(plant, comp, mission, options=dict(start='navigate', start_l=28.3, duration=4.))
        A = rec.arrays()
        late = A['t'] > A['t'][0] + 2.
        peak[ground] = max(np.linalg.norm(A[f'sensor:{rid}'][late, 3:], axis=1).max() for rid in comp.robots)
        low[ground] = min(v[0] for c in rec.contacts[len(rec.contacts) // 2:] for v in c['wheels'].values())
    assert peak['tyres'] < 75. and low['tyres'] > 150.              # N: the clamps carry their share of the beam, nothing more
    assert peak['rigid'] > 110. and low['rigid'] < 120.             # N: beyond the range of the wrist sensors; wheels unload
