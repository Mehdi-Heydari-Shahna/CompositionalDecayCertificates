"""Collect the mission runs of the four teams on rolling wheels: both feedforwards, both plants.

Reads the run folders written by the workflows of the package (report.json, record.npz) under
    <package>/results/v7/<plant>_<team>_<kind>          kind: fitted | fitted05 | fitted15 (and desired, see below)
    <package>/results/v7/earlier/desired_d9384fd/<plant>_<team>_desired
and the stored reports of the sixth release under <package>/reference_results/<plant>_<team>.  The runs with the
feedforward of the desired motion were made with commit d9384fd (the default controller of the sixth release, with
the monitor); those repeated with the final files (results/v7/<plant>_<team>_desired) must give the same report value
for value.
Nothing of the package is run or changed.  Written here: missions.json (one entry per run) and missions.npz
(decimated time histories for the figures).

kind       feedforward of the mission controller
desired    inverse dynamics of the desired motion (the controller of the earlier releases), quantities recorded
fitted     effort required by the fitted reference of the loops; measured rates in the damping term
fitted05   the same with the rate gains K = sqrt(k_p m) on the arm and lift joints (damping ratio 1/2 of the design,
           half the rate gains of the other runs; --option arm_damping_ratio=0.5 --option lift_damping_ratio=0.5)
fitted15   the fitted feedforward with the rates of the damping term low-passed at 15 Hz (outside the statement)
comparison runs (MuJoCo):
desired0   the controller of the earlier releases without its 15 Hz rate filter (--monitor --velocity-filter 0)
fitted0515 the fitted feedforward with K = sqrt(k_p m) on the arm and lift joints and the 15 Hz rate filter
desiredR   the controller of the earlier releases (rate filter included) with the damping of every arm joint designed
           at least for the inertia the joint sees in the closed chain at the carrying configuration: damping ratio
           max(1, sqrt(seen / design)) per arm joint, the largest over the robots of the team (DAMPING_R)

Checked on the way (assertions):
  * every run is complete, without a saturated step; the declared bounds that a run does not meet are recorded
    (they are NOT asserted to be met);
  * a run with the feedforward of the desired motion gives the stored report of the sixth release value for value
    (the monitor changes nothing, and the default controller is the one of that release);
  * a run with the fitted feedforward found a fitted reference at every update and ran with the options its
    folder name says.
From the records: the stretches of the carry in which the bases travel sideways, straight ahead, or otherwise,
with the error energy W of the loops and the largest grasp moment that the controller measured in each; the
stretches in which a base is at an edge of the ramp (its origin within 0.6 m of the foot or the top line of the ramp
region of the scene), where the ground rows of its wheels change abruptly; W around the events of the mode (grasps
latched, weight taken, touchdown, releases); and the part of W that the measured velocity has outside the admissible
space of the ideal edges, W - W_admissible = 1/2 |(I - P) v|_M^2 (Proposition 3(iii)), which no admissible reference
can remove.
"""
import json
import sys
from pathlib import Path

import numpy as np

from lib import RESULTS, ROOT, ROLLING, load_team

RUNS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'v7'
PLANTS = ('pinocchio', 'mujoco')
KINDS = ('desired', 'fitted', 'fitted05', 'fitted15', 'desired0', 'fitted0515', 'desiredR')
EXPECT = dict(desired=dict(feedforward='desired', velocity_filter_hz=15.), fitted=dict(feedforward='fitted', velocity_filter_hz=0.),
              fitted05=dict(feedforward='fitted', velocity_filter_hz=0.), fitted15=dict(feedforward='fitted', velocity_filter_hz=15.),
              desired0=dict(feedforward='desired', velocity_filter_hz=0.), fitted0515=dict(feedforward='fitted', velocity_filter_hz=15.),
              desiredR=dict(feedforward='desired', velocity_filter_hz=15.))
DAMPING = dict(desired=1., fitted=1., fitted05=.5, fitted15=1., desired0=1., fitted0515=.5, desiredR=None)   # damping ratio of the joint design
DAMPING_R = dict(ridgeback3=[1, 1, 1, 2.2, 1.87, 2.17], team2=[1, 1, 1, 2.15, 4.05, 2.88], team3=[1, 1, 1, 2.64, 3.13, 2.39],
                 ridgeback2=[1, 1, 1, 2.24, 2.37, 2.61])               # desiredR: damping ratios of the arm joints (--option arm_damping_ratio)
RAMP_X, RAMP_EDGES_Y, NEAR = (8.1, 11.1), (14.2, 17.2), .6                # ramp region of the four scenes (inputs/scenes)
MODE_EVENTS = ('latched', 'weight_taken', 'touchdown', 'unlatched', 'released')


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + '.'))
        else:
            out[pre + k] = v
    return out


def one(folder, kind, earlier=False):
    r = json.loads((folder / 'report.json').read_text())
    tr, g, lp = r['tracking'], r['grasp'], r.get('loops')
    assert r['status'] == 'complete' and r['efforts']['saturated_steps'] == 0, folder
    checks = r['bounds']['checks']
    unmet = {k: [c['value'], c['bound']] for k, c in checks.items() if not c['ok']}
    assert r['bounds']['all_ok'] == (not unmet)
    for k in ('mission_complete', 'supports_verified', 'no_forbidden_contact'):
        assert checks[k]['ok'], (folder, k)
    out = dict(sim_time=r['sim_time_s'], bounds_met=not unmet, unmet=unmet, checks=len(checks),
               payload_mm=1e3 * tr['payload_position_error_m']['max'], payload_rms_mm=1e3 * tr['payload_position_error_m']['rms'],
               tilt_mrad=1e3 * tr['payload_tilt_rad']['max'], tilt_rms_mrad=1e3 * tr['payload_tilt_rad']['rms'],
               base_mm=1e3 * max(b['position_error_m']['max'] for b in tr['base'].values()),
               base_rms_mm=1e3 * max(b['position_error_m']['rms'] for b in tr['base'].values()),
               internal_force_max=g['internal_force_N']['max'], internal_force_rms=g['internal_force_N']['rms'],
               internal_moment_max=g['internal_moment_Nm']['max'], internal_moment_rms=g['internal_moment_Nm']['rms'],
               grasp_force_max=checks['grasp_force']['value'], grasp_force_bound=checks['grasp_force']['bound'],
               grasp_moment_max=checks['grasp_moment']['value'], grasp_moment_bound=checks['grasp_moment']['bound'],
               effort=r['efforts']['max_ratio'], effort_port=r['efforts']['worst']['port'],
               placement_mm=1e3 * r['verification']['position_error_m'],
               control_seconds=r['timing'].get('control'), plant_seconds=r['timing'].get('plant'),
               engine=r['plant'].get('engine'))
    given = r['plant'].get('controller') or {}
    if earlier:
        # a run of commit d9384fd, made before the report recorded the loops as they ran: the default controller with
        # the monitor (its options are those of the sixth release; checked value for value below)
        assert kind == 'desired' and given == {'monitor': True} and lp['feedforward'] == 'desired', folder
        how = dict(velocity_filter_hz=15.)
    else:
        assert lp is not None and lp.get('controller'), folder
        how = lp['controller']
        for k, v in EXPECT[kind].items():
            assert how[k] == v, (folder, k, how[k])
        assert how['advance'] == 'positions' and how['feedforward_rate'] == 'slow', folder
        assert (how['reference_update_s'], how['loop_step_s']) == (.01, .002), folder
    if kind == 'desiredR':
        assert given.get('arm_damping_ratio') == DAMPING_R[folder.name.split('_')[1]] and 'lift_damping_ratio' not in given, (folder, given)
    else:
        for key in ('arm_damping_ratio', 'lift_damping_ratio'):
            assert float(given.get(key, 1.)) == DAMPING[kind], (folder, key, given)
    out.update(feedforward=lp['feedforward'], velocity_filter_hz=how['velocity_filter_hz'],
               W_median=lp['error_energy_J']['median'], W_p99=lp['error_energy_J']['p99'],
               W_max=lp['error_energy_J']['max'], W_max_phase=lp['error_energy_J']['phase'], W_max_t=lp['error_energy_J']['at'],
               rate_min=lp['rate_1_s']['smallest'], rate_max=lp['rate_1_s']['largest'],
               internal_effort_max=lp['internal_effort_ratio']['max'], internal_effort_rms=lp['internal_effort_ratio']['rms'],
               unfitted=lp['updates_without_fitted_reference'], modes=lp['rate_1_s']['by_mode'],
               W_by_phase=lp['error_energy_by_phase_J'])
    assert out['unfitted'] == 0, folder
    return out, r


def record(folder, comp, events):
    """From the record: W in half-second windows, the carry split by the way the bases travel, the stretches at the
    edges of the ramp, W around the events of the mode, and the part of W outside the admissible space."""
    z = np.load(folder / 'record.npz', allow_pickle=False)
    t, W = z['t'], z['fit__W']
    W_adm = z['fit__W_admissible']
    edges = np.arange(0., t[-1] + .5, .5)
    idx = np.digitize(t, edges) - 1
    med = np.array([np.nanmedian(W[idx == k]) if np.any(idx == k) else np.nan for k in range(len(edges))])
    top = np.array([np.nanmax(W[idx == k]) if np.any(idx == k) else np.nan for k in range(len(edges))])
    carrying = (z['mode_grasps'] != '') & (z['mode_rest'] == '')
    # position error of the payload (as the audit takes it) and the largest grasp moment that the controller measured
    pb = np.asarray(comp.payload['base'], int)
    perr = np.linalg.norm((z['q'][:, pb] - z['payload_ref'])[:, :3], axis=1)
    gm = np.max([np.linalg.norm(z[f'lam_raw__{c}'][:, :3], axis=1) for c in comp.connections if f'lam_raw__{c}' in z.files], axis=0)
    window_max = lambda y, m: np.array([np.nanmax(np.where(m & (idx == k), y, np.nan)) if np.any(m & (idx == k)) else np.nan
                                        for k in range(len(edges))])
    carry_rows = (z['mode_grasps'] != '') & (z['mode_rest'] == '')
    arrays = dict(t=edges, W_median=med, W_max=top, phase_t=t[np.r_[0, 1 + np.flatnonzero(np.diff(z['phase']) != 0)]],
                  payload_error_max=window_max(perr, carry_rows), grasp_moment_max=window_max(gm, carry_rows),
                  carry=np.array([t[carrying][0], t[carrying][-1]]) if carrying.any() else np.zeros(2),
                  kmin=np.array([np.nanmin(z['fit__rate'][idx == k, 0]) if np.any(idx == k) else np.nan for k in range(len(edges))]))
    # how the bases travel: speed of each base over the ground and its part across the heading (mean over the robots)
    speed, across = [], []
    at_edge = np.zeros(len(t), bool)
    for r in comp.robots.values():
        b = r.base
        vx, vy, yaw = z['v'][:, b[0]], z['v'][:, b[1]], z['q'][:, b[3]]
        speed.append(np.hypot(vx, vy))
        across.append(np.abs(-np.sin(yaw) * vx + np.cos(yaw) * vy))
        x, y = z['q'][:, b[0]], z['q'][:, b[1]]
        inside = (x > RAMP_X[0] - NEAR) & (x < RAMP_X[1] + NEAR)
        at_edge |= inside & (np.min([np.abs(y - e) for e in RAMP_EDGES_Y], axis=0) < NEAR)
    speed, across = np.mean(speed, axis=0), np.mean(across, axis=0)
    moment = np.max([np.linalg.norm(z[f'lam_raw__{c}'][:, :3], axis=1) for c in comp.connections if f'lam_raw__{c}' in z.files], axis=0)
    moving = carrying & (speed > .05)
    sideways = moving & (across > .9 * speed)
    ahead = moving & (across < .1 * speed)
    other = carrying & ~sideways & ~ahead
    seg = {}
    for name, m in (('sideways', sideways), ('ahead', ahead), ('other', other), ('carry', carrying)):
        ok = m & np.isfinite(W)
        seg[name] = dict(seconds=float(m.sum() * np.median(np.diff(t))),
                         W_median=float(np.median(W[ok])) if ok.any() else None, W_max=float(W[ok].max()) if ok.any() else None,
                         grasp_moment_max=float(moment[m].max()) if m.any() else None,
                         speed_max=float(speed[m].max()) if m.any() else None)
    ok = carrying & np.isfinite(W)
    k = np.flatnonzero(ok)[np.argsort(W[ok])[-50:]]                # the fifty largest values of W while carrying (1 s of record rows)
    seg['largest_W_sideways_share'] = float(np.mean(sideways[k]))
    edge = carrying & at_edge
    seg['ramp_edges'] = dict(seconds=float(edge.sum() * np.median(np.diff(t))), share_of_carry=float(edge.sum() / max(carrying.sum(), 1)),
                             W_median=float(np.nanmedian(W[edge])) if edge.any() else None,
                             W_max=float(np.nanmax(W[edge])) if edge.any() else None)
    seg['largest_W_ramp_share'] = float(np.mean(at_edge[k]))
    seg['largest_W_t'] = float(t[k[-1]])
    seg['largest_W_at_ramp_edge'] = bool(at_edge[k[-1]])
    # the part of W that the measured velocity has outside the admissible space (no admissible reference removes it)
    fin = carrying & np.isfinite(W) & np.isfinite(W_adm)
    out = np.clip(W[fin] - W_adm[fin], 0., None)
    seg['inadmissible'] = dict(share_of_sum=float(out.sum() / W[fin].sum()), median_share=float(np.median(out / np.maximum(W[fin], 1e-300))),
                               median_J=float(np.median(out)), max_J=float(out.max()))
    # W around the events of the mode: largest W from one record row before an event to 0.2 s after it
    around = {}
    for e in events:
        if e['event'] in MODE_EVENTS:
            m = (t >= e['t'] - .02) & (t <= e['t'] + .2)
            if m.any():
                around.setdefault(e['event'], []).append(float(np.nanmax(W[m])))
    seg['events'] = {name: dict(count=len(v), W_max=max(v)) for name, v in around.items()}
    seg['events_W_max'] = max((max(v) for v in around.values()), default=None)
    j = int(np.argmax(np.where(carrying, moment, -np.inf)))
    seg['largest_grasp_moment'] = dict(t=float(t[j]), sideways=bool(sideways[j]), ahead=bool(ahead[j]), speed=float(speed[j]))
    return arrays, seg


EARLIER = RUNS / 'earlier' / 'desired_d9384fd'
checked_with_final_files = {}
runs, arrays = {}, {}
for plant in PLANTS:
    for team in ROLLING:
        comp = load_team(team)
        for kind in KINDS:
            folder = RUNS / f'{plant}_{team}_{kind}'
            earlier = False
            if kind == 'desired':
                # the runs with the feedforward of the desired motion are those of commit d9384fd, whose default controller
                # is that of the sixth release and of the final files; where a run with the final files exists, the two
                # reports must agree value for value (timing and the record of the loops as they ran apart)
                if (folder / 'report.json').exists():
                    final = flat(json.loads((folder / 'report.json').read_text()))
                    first = flat(json.loads((EARLIER / f'{plant}_{team}_desired' / 'report.json').read_text()))
                    common = [k for k in first if k in final and not k.startswith('timing')]
                    differ = [k for k in common if first[k] != final[k]]
                    extra = sorted(set(final) - set(first))
                    assert not differ and set(first) - set(final) <= {k for k in first if k.startswith('timing')}, (folder, differ[:5])
                    assert all(k.startswith(('loops.controller.', 'timing')) for k in extra), extra
                    checked_with_final_files[f'{plant}_{team}'] = len(common)
                folder, earlier = EARLIER / f'{plant}_{team}_desired', True
            if not (folder / 'report.json').exists():
                continue
            key = f'{plant}_{team}_{kind}'
            runs[key], report = one(folder, kind, earlier)
            runs[key].update(plant=plant, team=team, kind=kind, folder=str(folder.relative_to(ROOT)))
            if kind == 'desired':
                # the stored report of the sixth release, value for value (timing and the entries of the monitor apart)
                stored = flat(json.loads((ROOT / 'reference_results' / f'{plant}_{team}' / 'report.json').read_text()))
                new = flat(report)
                common = [k for k in stored if k in new and not k.startswith('timing')]
                differ = [k for k in common if stored[k] != new[k]]
                only_old = [k for k in stored if k not in new]
                assert not differ and not only_old and len(common) > 200, (key, differ[:5], only_old[:5])
                runs[key]['equal_to_release_6'] = len(common)
            hist, seg = record(folder, comp, report['events'])
            runs[key]['segments'] = seg
            arrays.update({f'{key}:{name}': value for name, value in hist.items()})
            print(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in runs[key].items()
                        if k in ('payload_mm', 'base_mm', 'internal_force_max', 'internal_force_rms', 'grasp_moment_max', 'effort',
                                 'W_median', 'W_p99', 'equal_to_release_6', 'unmet')},
                  'sideways', {k: (float(f'{v:.3g}') if isinstance(v, float) else v) for k, v in seg['sideways'].items()},
                  'ahead W', seg['ahead']['W_median'], 'share', seg['largest_W_sideways_share'], seg['largest_grasp_moment'],
                  'ramp edges', {k: (float(f'{v:.3g}') if isinstance(v, float) else v) for k, v in seg['ramp_edges'].items()},
                  'largest W at ramp edges', seg['largest_W_ramp_share'], 'inadmissible', {k: float(f'{v:.3g}') for k, v in seg['inadmissible'].items()},
                  'events', seg['events_W_max'], flush=True)
runs['_checks'] = dict(desired_runs_repeated_with_final_files=checked_with_final_files)
print('desired runs of d9384fd repeated with the final files, equal value for value:', checked_with_final_files)
(RESULTS / 'missions.json').write_text(json.dumps(runs, indent=1))
np.savez_compressed(RESULTS / 'missions.npz', **arrays)
print('runs', len(runs) - 1, 'with every declared bound met:', sum(r['bounds_met'] for k, r in runs.items() if not k.startswith('_')))
