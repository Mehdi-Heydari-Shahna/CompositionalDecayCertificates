"""The raw record of a run can be stored, read back and audited again without the plant."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('mujoco')

from backends.mujoco_plant import MujocoPlant  # noqa: E402
from mission.audit import audit, summary_lines  # noqa: E402
from mission.report import load_record, save_record  # noqa: E402
from mission.runner import run_mission  # noqa: E402
from teamc.interfaces import TeamModel  # noqa: E402
from teamc.io import load_team  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def short_run(tmp_path_factory):
    from teamc.io import INPUTS, read_json
    comp = load_team('team2')
    mission = read_json(INPUTS / 'missions' / 'team2.json')
    plant = MujocoPlant(comp, roller_hull=(7, 32))
    ctrl, rec, info = run_mission(plant, comp, mission, options=dict(start='navigate', start_l=1., duration=2.))
    path = tmp_path_factory.mktemp('record') / 'record.npz'
    A = rec.arrays()
    save_record(path, A, info, rec.contacts)
    report, _ = audit(comp, mission, A, info, rec.contacts, model=ctrl.model)
    return dict(path=path, A=A, info=info, contacts=rec.contacts, report=report, mission=mission)


def test_record_round_trip(short_run):
    A, info, contacts = load_record(short_run['path'])
    assert set(A) == set(short_run['A'])
    for key, value in short_run['A'].items():
        assert (A[key] == value).all() if value.dtype.kind in 'US' else np.array_equal(A[key], value, equal_nan=True)
    assert info['team'] == 'team2' and info['status'] == short_run['info']['status']
    assert len(contacts) == len(short_run['contacts'])
    assert contacts[5]['wheels'].keys() == short_run['contacts'][5]['wheels'].keys()
    latched = [e for e in info['events'] if e['event'] == 'latched']
    assert len(latched) == 2 and np.asarray(latched[0]['T2']).shape == (4, 4)


def test_audit_from_the_record_alone_reproduces_the_audit(short_run):
    A, info, contacts = load_record(short_run['path'])
    comp = load_team('team2')
    model = TeamModel(comp)
    for e in info['events']:
        if e['event'] == 'latched':
            model.register_grasp(e['id'], e['T2'])
    report, _ = audit(comp, short_run['mission'], A, info, contacts, model=model)
    first = short_run['report']
    assert report['tracking']['payload_position_error_m']['max'] == pytest.approx(first['tracking']['payload_position_error_m']['max'], rel=1e-9)
    assert report['grasp']['internal_force_N']['max'] == pytest.approx(first['grasp']['internal_force_N']['max'], rel=1e-6)
    for cid, v in first['consistency']['recovered_vs_sensor'].items():
        assert report['consistency']['recovered_vs_sensor'][cid]['force_N']['rms'] == pytest.approx(v['force_N']['rms'], rel=1e-6)
    assert first['consistency']['state_integration']['floating_translation_m']['max'] < 1e-7      # MuJoCo integrates what it reports
    assert any('declared bounds' in line for line in summary_lines(report))


def test_audit_reports_a_base_the_skid_model_cannot_turn(missions):
    """Wheels of one side without load leave the skid model without a twist: the audit says so and does not fail."""
    comp = load_team('husky2')
    mission = missions('husky2')
    plant = MujocoPlant(comp)
    ctrl, rec, info = run_mission(plant, comp, mission, options=dict(start='navigate', start_l=1., duration=1.5))
    A = rec.arrays()
    first, _ = audit(comp, mission, A, info, rec.contacts, model=ctrl.model)
    assert all('skid_model_not_evaluated' not in first['ground'][rid]['skid'] for rid in comp.robots)

    model, seen = ctrl.model, []
    skid_twist = model.skid_twist

    def one_side_unloaded(q, v, rid, loads, kin=None):
        seen.append(rid)
        if rid == 'r1' and seen.count('r1') % 3 == 0:
            raise ValueError('skid model: the wheels cannot turn the base (no friction on one side?)')
        return skid_twist(q, v, rid, loads, kin)

    model.skid_twist = one_side_unloaded
    try:
        report, _ = audit(comp, mission, A, info, rec.contacts, model=model)
    finally:
        del model.skid_twist
    missed = report['ground']['r1']['skid']['skid_model_not_evaluated']
    assert 0 < missed['samples'] < missed['of']
    assert 'skid_model_not_evaluated' not in report['ground']['r2']['skid']
    for key in ('max_slip_speed_m_s', 'friction_energy_J', 'turned_rad'):      # what does not need the skid model is unchanged
        assert report['ground']['r1']['skid'][key] == first['ground']['r1']['skid'][key]
    assert report['bounds'] == first['bounds']
    assert any('no skid model there' in line for line in summary_lines(report))


def test_make_report_tool(short_run, tmp_path):
    pytest.importorskip('matplotlib')
    out = subprocess.run([sys.executable, str(ROOT / 'tools' / 'make_report.py'), '--record', str(short_run['path']), '--out', str(tmp_path)],
                         capture_output=True, text=True, timeout=600)
    assert 'payload while carried' in out.stdout, out.stdout[-1500:] + out.stderr[-1500:]
    assert (tmp_path / 'report.json').is_file() and (tmp_path / 'grasp_loads.png').is_file()
