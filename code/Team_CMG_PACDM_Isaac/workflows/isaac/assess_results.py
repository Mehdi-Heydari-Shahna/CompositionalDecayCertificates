#!/usr/bin/env python
"""Assessment of runs of the Isaac Sim workflow: what passed, on which engine, and how it compares.

    python workflows/isaac/assess_results.py results/isaac_campaign_<id>        # a campaign folder
    python workflows/isaac/assess_results.py campaign_bundle.zip               # the bundle of a campaign
    python workflows/isaac/assess_results.py results/isaac_husky2_<id> ...     # run folders of single teams

For every team it reads the records the workflow wrote (``manifest.json``, the ``case.json`` of
every stage, ``preflight.json``, the ``report.json`` of the mission) and states

* the engine of the run: Isaac Sim with its versions, or a stand-in, which is not Isaac Sim;
* whether the files that produced the run are those of this package (the hashes in ``case.json``);
* every stage with its outcome, the preflight checks, the declared bounds of the mission;
* the figures of the mission beside those of the reference runs of this package (PhysX 5.3.1
  stand-in, MuJoCo, Pinocchio plant), with the differences that deserve a look marked.

The verdict of a team follows from its stages and its declared bounds alone.  The comparison with
the reference runs is information: another PhysX version need not give the same numbers, and the
thresholds that mark a difference are review thresholds, not acceptance criteria.

``ASSESSMENT.md`` and ``assessment.json`` are written into the first input folder, or beside the first
input when it is a bundle (``--output`` elsewhere).
Exit status: 0 every team shown passed every stage in Isaac Sim; 3 everything passed, but not in
Isaac Sim (a stand-in); 4 nothing failed but stages or teams are missing; 1 something failed;
2 the input could not be read.

Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

STAGES = ('selftest', 'preflight', 'settle', 'carry', 'mission')
TEAMS = ('team2', 'ridgeback2', 'husky2', 'team3', 'ridgeback3', 'husky3')
PLATFORM = dict(team2='RB-ROBOUT+ / UR30', team3='RB-ROBOUT+ / UR30', ridgeback2='Ridgeback / UR10e',
                ridgeback3='Ridgeback / UR10e', husky2='Husky A300 / UR10e', husky3='Husky A300 / UR10e')

# name, heading, unit, digits, review threshold against the PhysX stand-in: absolute part, relative part
METRICS = (
    ('sim_time_s', 'simulated time', 's', 0, 1.0, 0.),
    ('payload_position_error_mm', 'payload position error, max', 'mm', 1, 1.0, 0.),
    ('payload_tilt_mrad', 'payload tilt, max', 'mrad', 1, .5, 0.),
    ('internal_force_max_N', 'internal force, max', 'N', 1, 10., .5),
    ('grasp_force_max_N', 'grasp force, max', 'N', 0, 5., .1),
    ('grasp_moment_max_Nm', 'grasp moment, max', 'N m', 1, 1.5, 0.),
    ('effort_ratio_max', 'effort ratio, max', '', 2, .08, 0.),
    ('min_wheel_load_N', 'wheel load, min', 'N', 0, 0., .15),
    ('traction_use_max', 'traction use, max', '', 2, .15, 0.),
    ('tipping_margin_min_m', 'tipping margin, min', 'm', 3, .02, 0.),
    ('base_tracking_max_mm', 'base tracking error, max', 'mm', 0, 15., 0.),
    ('placement_error_mm', 'placement error', 'mm', 2, .5, 0.),
)
# what the compiled model says against what the plant reported (the audit computes these for every plant): name, heading, digits
CONSISTENCY = (
    ('unbalanced_rms', 'equations of motion of the compiled model with the measured motion: unbalanced generalized force, rms, N or N m', 2),
    ('payload_balance_rms_N', 'payload: the loads applied to it against its measured motion, rms, N', 2),
    ('recovered_force_rms_N', 'grasp reactions recovered from the model against the wrist sensors: force, rms, N', 2),
    ('recovered_moment_rms_Nm', 'the same, moment, rms, N m', 2),
    ('normal_load_rms_N', 'ground load of the model against the contact report of the plant, rms, N', 2),
    ('integration_um', 'displacement of the floating bodies against their summed rates, per record interval, max, um', 1),
)
# checks of the preflight by what they compare (part of the check name, heading): the worst value of each is shown
PREFLIGHT_GROUPS = (
    ('link masses', 'link masses, relative error'),
    ('link centres of mass', 'link centres of mass, m'),
    ('link inertia tensors', 'link inertia tensors, relative error'),
    ('joint armature', 'joint armature, SI'),
    ('joint limits', 'joint limits, SI'),
    ('payload: inertia', 'payload inertia tensor, relative error'),
    ('gravity', 'gravity, m/s2'),
    ('roller hulls', 'roller hulls: authored points outside the cooked hull, m'),
    ('configuration round trip', 'configuration written and read back, m or rad'),
    ('velocity round trip', 'rates written and read back, SI'),
    ('link positions against', 'link positions against the compiled kinematics, m'),
    ('link orientations against', 'link orientations against the compiled kinematics, rad'),
    ('free-flight forward dynamics', 'free-flight accelerations against the compiled dynamics, relative error'),
    ('effect of external', 'effect of a known external load on the accelerations, relative error'),
    ('do not act in the following step', 'share of that load left in the following step'),
    ('step displacement against', 'displacement of a step outside the reported rates, m or rad'),
)
PREFLIGHT_REFERENCE = 'physx_standin_campaign/{team}/preflight/preflight.json'
REFERENCES = (('standin', 'PhysX 5.3.1 stand-in', 'physx_standin_campaign/{team}/mission/report.json'),
              ('mujoco_compliant', 'MuJoCo, compliant clamps', 'mujoco_{team}_compliant_clamp/report.json'),
              ('mujoco', 'MuJoCo', 'mujoco_{team}/report.json'),
              ('pinocchio', 'Pinocchio plant', 'pinocchio_{team}/report.json'))
NOT_TESTED = (
    'The compliant wheel contact (`--contact compliant`), unless the run was made with it.',
    'Hardware: every plant here is a simulation, with declared drive, tyre and clamp data.',
)
TYRE_NOT_TESTED = dict(
    tyre='The friction law and the normal contact of the Husky wheels: with `--contact tyre` (what `auto`, the default, '
         'gives these wheels) both are the declared tyre model, applied by the adapter; the wheels have no collider.',
    rigid='The friction law of the Husky wheels: in this workflow it is the declared tyre model, applied by the adapter. '
          'With `--contact rigid` PhysX carries the wheels and the terrain without friction, and the share of the two '
          'wheel diagonals of a base is left to its contact solver.',
    compliant='The friction law of the Husky wheels: in this workflow it is the declared tyre model, applied by the adapter. '
              'With `--contact compliant` PhysX carries the wheels on compliant contacts, without friction.')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ====================================================================== input
def unpack(path, holder):
    """A folder for an input path: the folder itself, or a bundle extracted to a temporary folder."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        return path
    if path.is_file() and zipfile.is_zipfile(path):
        target = Path(tempfile.mkdtemp(prefix='isaac_assessment_'))
        holder.append(target)
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.startswith(('/', '\\')) or '..' in Path(name).parts:
                    raise ValueError(f'{path}: unsafe path in the archive: {name}')
            z.extractall(target)
        return target
    raise ValueError(f'{path} is neither a folder nor a zip archive')


def is_run(folder):
    try:
        m = read_json(folder / 'manifest.json')
    except (OSError, ValueError):
        return False
    return isinstance(m, dict) and 'team' in m and 'stages' in m


def find_runs(folder):
    """``(campaign record or None, environment run folder or None, {team: run folder})`` of an input folder."""
    campaign, environment, runs = None, None, {}

    def add(run):
        m = read_json(run / 'manifest.json')
        counted = latest_stages(m)
        if set(counted) == {'environment'}:
            return run
        if m['team'] in runs:
            raise ValueError(f"two runs of {m['team']}: {runs[m['team']]} and {run}")
        runs[m['team']] = run
        return None

    if (folder / 'campaign.json').is_file():
        campaign = read_json(folder / 'campaign.json')
        if campaign.get('environment') and is_run(folder / campaign['environment']['folder']):
            environment = folder / campaign['environment']['folder']
        for team, entry in campaign.get('teams', {}).items():
            if is_run(folder / entry['folder']):
                add(folder / entry['folder'])
    elif is_run(folder):
        environment = add(folder)
    else:
        inside = sorted(p for p in folder.iterdir() if p.is_dir())
        campaigns = [p for p in inside if (p / 'campaign.json').is_file()]
        if len(campaigns) == 1 and not any(is_run(p) for p in inside):   # an archive that holds the campaign folder itself
            return find_runs(campaigns[0])
        if campaigns:
            raise ValueError(f'{folder} holds {len(campaigns)} campaign folder(s) among other things: name the one to assess')
        for sub in inside:
            if is_run(sub):
                environment = add(sub) or environment
    return campaign, environment, runs


def latest_stages(manifest):
    latest = {}
    for entry in manifest.get('stages', []):
        if not entry.get('superseded'):
            latest[entry['stage']] = entry
    return latest


# ====================================================================== one run
def engine_of(case):
    """What simulated: ``(kind, text)`` with kind ``isaac``, ``stand-in``, ``dry`` (nothing simulated) or ``unknown``."""
    v = case.get('versions') or {}
    label = str(case.get('engine', ''))
    stand_in = case.get('stand_in') or {}
    if stand_in.get('dry'):
        return 'dry', str(stand_in.get('label') or 'nothing simulated (dry run of the coordinators)')
    physx = str(stand_in.get('physx') or '')
    if 'stand-in' in label.lower() or 'stand_in' in case or 'stand-in' in str(v.get('kit_version', '')).lower():
        return 'stand-in', label + (f' (PhysX {physx})' if physx and physx not in label else '')
    isaac = v.get('isaacsim') or '.'.join(str(x) for x in (v.get('isaac_version') or [])[:1] if x)
    if isaac or v.get('kit_version'):
        parts = [f'Isaac Sim {isaac}' if isaac else 'Isaac Sim (version not reported)']
        if v.get('kit_version'):
            parts.append(f"Kit {v['kit_version']}")
        if v.get('physx_extension'):
            parts.append(str(v['physx_extension']))
        return 'isaac', ', '.join(parts)
    return 'unknown', label or 'not recorded'


def source_state(case, package):
    """The files of the run against the files of this package: ``(identical, differing, missing here)``."""
    recorded = case.get('source_sha256') or {}
    differing, missing = [], []
    for name, digest in sorted(recorded.items()):
        path = package / name
        if not path.is_file():
            missing.append(name)
        elif sha256(path) != digest:
            differing.append(name)
    return bool(recorded) and not differing and not missing, differing, missing


def mission_metrics(report):
    """The figures of a mission report that the results tables of the package show."""
    out = {}
    try:
        tr, g = report['tracking'], report['grasp']
        ground = {k: v for k, v in report['ground'].items() if isinstance(v, dict)}
        grasps = g['by_grasp'].values()
        out.update(
            sim_time_s=report['sim_time_s'],
            payload_position_error_mm=1e3 * tr['payload_position_error_m']['max'],
            payload_tilt_mrad=1e3 * tr['payload_tilt_rad']['max'],
            internal_force_max_N=g['internal_force_N']['max'], internal_force_rms_N=g['internal_force_N']['rms'],
            grasp_force_max_N=max(x['force_N']['max'] for x in grasps),
            grasp_moment_max_Nm=max(x['moment_Nm']['max'] for x in grasps),
            effort_ratio_max=report['efforts']['max_ratio'],
            min_wheel_load_N=min(x['min_wheel_load_N'] for x in ground.values()),
            traction_use_max=max(x.get('net_traction_utilization_max', x.get('max_traction_utilization')) for x in ground.values()),
            tipping_margin_min_m=min(x['min_tipping_margin_m'] for x in ground.values()),
            base_tracking_max_mm=1e3 * max(b['position_error_m']['max'] for b in tr['base'].values()),
            base_yaw_error_max_mrad=1e3 * max(b['yaw_error_rad']['max'] for b in tr['base'].values()),
            clearance_min_m=report['clearance']['smallest_m'])
        position = (report.get('verification') or {}).get('position_error_m')
        if number(position):
            out['placement_error_mm'] = 1e3 * position
        ratios = [x['skid']['against_skid_model']['yaw_rate_ratio'] for x in ground.values()
                  if 'skid' in x and 'against_skid_model' in x['skid']]
        if ratios:
            out['skid_yaw_rate_ratio'] = ratios
    except (KeyError, TypeError, ValueError) as exc:
        out['unreadable'] = f'{type(exc).__name__}: {exc}'
    c = report.get('consistency') or {}

    def part(value, key='rms'):
        value = value.get(key) if isinstance(value, dict) else None
        return value if number(value) else None

    recovered = [g for g in (c.get('recovered_vs_sensor') or {}).values() if isinstance(g, dict)]
    step = part((c.get('state_integration') or {}).get('floating_translation_m'), 'max')
    out['consistency'] = dict(
        unbalanced_rms=part(c.get('unbalanced_generalized_force')), payload_balance_rms_N=part(c.get('payload_wrench_balance_N')),
        recovered_force_rms_N=max((x for x in (part(g.get('force_N')) for g in recovered) if x is not None), default=None),
        recovered_moment_rms_Nm=max((x for x in (part(g.get('moment_Nm')) for g in recovered) if x is not None), default=None),
        normal_load_rms_N=part(c.get('total_normal_load_vs_plant_N')), integration_um=None if step is None else 1e6 * step)
    return out


def preflight_worst(pre):
    """The worst value of every group of preflight checks: ``{part of the name: dict(value, limit)}``."""
    out = {}
    for key, _ in PREFLIGHT_GROUPS:
        rows = [c for c in pre.get('checks', []) if key in str(c.get('name', '')) and number(c.get('value'))]
        if rows:
            out[key] = dict(value=max(c['value'] for c in rows), limit=max((c['limit'] for c in rows if number(c.get('limit'))), default=None),
                            checks=len(rows))
    return out


def assess_run(folder, package, reference):
    manifest = read_json(folder / 'manifest.json')
    team = manifest['team']
    counted = latest_stages(manifest)
    out = dict(team=team, platform=PLATFORM.get(team, 'unknown'), folder=folder.name, run_id=manifest.get('run_id'),
               manifest_status=manifest.get('status'), stages={}, problems=[], notes=[])
    cases = {}
    for stage in STAGES:
        entry = counted.get(stage)
        if entry is None:
            out['stages'][stage] = dict(status='not run')
            continue
        row = dict(status=entry.get('status'), problems=list(entry.get('problems') or []),
                   attempts=1 + sum(1 for e in manifest['stages'] if e['stage'] == stage and e.get('superseded') and e.get('status') != 'not run'))
        if entry.get('other_error_lines_in_log'):
            row['other_error_lines_in_log'] = entry['other_error_lines_in_log']
        if entry.get('closing_problems'):                               # after the record was written: reported, not decisive
            row['closing_problems'] = list(entry['closing_problems'])
            out['notes'].append(f"{stage}: the record was complete; while the application closed afterwards: " + '; '.join(row['closing_problems']))
            if any(word in line.lower() for line in row['closing_problems'] for word in ('physx', 'physics')):
                # an application may write its own lines late: such a line is worth reading in the log of the stage
                out['notes'].append(f'{stage}: one of these lines is of a physics extension. It stands after the line that marks the '
                                    f'record as written; read logs/{stage}.log to see whether it belongs to the closing')
        try:
            case = cases[stage] = read_json(folder / stage / 'case.json')
        except (OSError, ValueError):
            case = None
            if entry.get('status') == 'passed':
                row['status'] = 'failed'
                row['problems'].append('the manifest says passed, but the stage has no case.json')
        if case is not None:
            if entry.get('status') == 'passed' and case.get('passed') is not True:
                row['status'] = 'failed'
                row['problems'].append('the manifest says passed, case.json does not')
            for key in ('wall_s', 'simulated_s', 'status', 'failure'):
                if case.get(key) is not None:
                    row['mission_status' if key == 'status' else key] = case[key]
            if case.get('physx_errors'):
                row['physx_errors'] = len(case['physx_errors'])
                row['status'] = 'failed'
                row['problems'].append(f"PhysX reported {len(case['physx_errors'])} error event(s)")
        # a stage that was interrupted (Ctrl+C, or the run ended from outside) did not fail: it was not finished
        if row['status'] == 'failed' and (row['problems'] == ['interrupted'] or (case or {}).get('failure') == 'interrupted'):
            row['status'] = 'interrupted'
        out['stages'][stage] = row

    # ---------------------------------------------------------------- engine, settings, files
    physics = [cases[s] for s in ('mission', 'carry', 'settle', 'preflight') if s in cases]
    kinds = {engine_of(c) for c in physics}
    if len(kinds) > 1:
        out['problems'].append('the stages of this run were made on different engines: ' + '; '.join(sorted(t for _, t in kinds)))
    out['engine_kind'], out['engine'] = engine_of(physics[0]) if physics else ('unknown', 'no physics stage was run')
    first = physics[0] if physics else cases.get('selftest')
    if first:
        v = first.get('versions') or {}
        out['versions'] = {k: v.get(k) for k in ('python', 'numpy', 'scipy', 'platform', 'isaacsim', 'isaac_version', 'kit_version',
                                                 'physx_extension', 'tensors_extension', 'physx_tensors_extension')}
        out['settings'] = first.get('settings')
        settings = first.get('settings') or {}
        out['contact_used'] = settings.get('contact_used') or settings.get('contact')      # runs before 'auto' named the contact used
    out['tyres'] = any(bool(c.get('tyre')) for c in physics)
    if physics:
        runtime = physics[0].get('physics_runtime') or {}
        out['physics_runtime'] = {k: runtime.get(k) for k in ('engine', 'tensor_route', 'device')}
    # what the stages noted about the installation (the same note of several stages once)
    remarks = []
    for stage in STAGES:
        case = cases.get(stage) or {}
        changed = (case.get('solver') or {}).get('changed_by_reset') or {}
        if changed:
            remarks.append('the reset of the World changed physics scene settings: '
                           + ', '.join(f'{key} {before!r} to {after!r}' for key, (before, after) in sorted(changed.items())))
        for note in case.get('physics_context_notes') or []:
            remarks.append(f'a call of the physics context failed (the setting is authored on the scene as well): {note}')
        if case.get('tensor_extension_note'):
            remarks.append(f"enabling the tensor extensions: {case['tensor_extension_note']}")
        if str(case.get('physx_error_stream', '')).startswith('unavailable'):
            remarks.append(f"PhysX error events were not collected: {case['physx_error_stream']}")
        if (case.get('contact_report') or {}).get('friction_note'):
            remarks.append(f"the friction anchors of the contact report were not used: {case['contact_report']['friction_note']}")
        if str(case.get('cart_mode', '')).startswith('unavailable'):
            remarks.append(f"the crossing cart could not be moved: {case['cart_mode']}")
    out['notes'] += list(dict.fromkeys(remarks))
    identical = True
    for stage, case in cases.items():
        same, differing, missing = source_state(case, package)
        identical = identical and same
        if differing or missing:
            out['source_differs'] = dict(stage=stage, differing=differing, not_in_this_package=missing)
            break
    out['source_identical'] = bool(cases) and identical
    if cases and not identical:
        out['notes'].append('the run was made with files that differ from this package (see source_differs)')

    # ---------------------------------------------------------------- preflight
    try:
        pre = read_json(folder / 'preflight' / 'preflight.json')
        checks = pre.get('checks', [])
        failed = [c for c in checks if not c.get('passed')]
        details = pre.get('details', {})
        out['preflight'] = dict(checks=len(checks), passed=len(checks) - len(failed),
                                failed=[dict(name=c['name'], value=c.get('value'), limit=c.get('limit'), relation=c.get('relation')) for c in failed],
                                unavailable=details.get('unavailable', []), load=details.get('load'))
        hulls = details.get('cooked_hulls')
        if hulls:
            out['preflight']['cooked_hull_vertices'] = sorted({(h['authored_vertices'], h['cooked_vertices']) for h in hulls})
        out['preflight']['worst'] = preflight_worst(pre)
        try:
            out['preflight']['worst_standin'] = preflight_worst(read_json(reference / PREFLIGHT_REFERENCE.format(team=team)))
        except (OSError, ValueError):
            out['preflight']['worst_standin'] = None
        if details.get('unavailable'):
            out['notes'].append('preflight checks that this installation could not make: ' + '; '.join(details['unavailable']))
    except (OSError, ValueError, KeyError):
        pass

    # ---------------------------------------------------------------- mission
    case = cases.get('mission')
    if case is not None:
        mission = dict(status=case.get('status'), unmet_bounds=case.get('unmet_bounds'), saturated_steps=case.get('saturated_steps'),
                       steps=case.get('steps'), clamp_stability_ratio=case.get('clamp_stability_ratio'),
                       clamp_max_deflection_mm=1e3 * case['clamp_max_deflection_m'] if number(case.get('clamp_max_deflection_m')) else None,
                       tyre_stability_ratio=(case.get('tyre') or {}).get('stability_ratio'), contact_report=case.get('contact_report'),
                       supports_accepted=(case.get('verification') or {}).get('accepted'))
        sensors = [r for r in (case.get('sensor_reference') or {}).values() if r.get('available')]
        if sensors:
            worst = max(sensors, key=lambda r: r['force_rms_N'])
            mission['sensor_balance_vs_physx_joint_force'] = dict(force_rms_N=worst['force_rms_N'], of_N=worst['sensor_force_rms_N'],
                                                                  native_sign=sorted({r['native_sign'] for r in sensors}))
        else:
            out['notes'].append("PhysX's own joint reaction was not available: the wrist-sensor balance has no independent reference")
        try:
            report = read_json(folder / 'mission' / 'report.json')
        except (OSError, ValueError):
            report = None
            if out['stages']['mission']['status'] != 'interrupted':      # an interrupted mission did not get to its report
                out['problems'].append('the mission has no report.json')
        if report is not None:
            mission['bounds'] = {k: dict(value=c.get('value'), bound=c.get('bound'), ok=bool(c.get('ok')))
                                 for k, c in report.get('bounds', {}).get('checks', {}).items()}
            mission['all_bounds_met'] = bool(report.get('bounds', {}).get('all_ok'))
            mission['metrics'] = values = mission_metrics(report)
            mission['plant_engine'] = (report.get('plant') or {}).get('engine')
            if 'unreadable' in values:
                out['problems'].append('the mission report could not be read completely: ' + values['unreadable'])
            if out['stages']['mission']['status'] == 'passed' and not mission['all_bounds_met']:
                out['stages']['mission']['status'] = 'failed'
                out['stages']['mission']['problems'].append('the stage is recorded as passed, but its report has unmet bounds')
            comparison, flagged, contacts = {}, [], {}
            for key, label, pattern in REFERENCES:
                try:
                    ref = read_json(reference / pattern.format(team=team))
                    comparison[key] = mission_metrics(ref)
                    contacts[key] = (ref.get('plant') or {}).get('contact')
                except (OSError, ValueError):
                    comparison[key] = None
            mission['references'] = comparison
            mission['contact'] = (report.get('plant') or {}).get('contact') or (case.get('settings') or {}).get('contact_used')
            base = comparison.get('standin')
            # the stand-in is a reference for the same wheel contact only: on another one the ground loads, the wheel
            # efforts and the base tracking of a team on conventional wheels are not expected to agree
            same = contacts.get('standin') in (None, mission['contact']) or not (case.get('tyre') or {})
            mission['standin_contact'] = contacts.get('standin')
            mission['standin_comparable'] = bool(same)
            if base and out['engine_kind'] != 'dry' and same:
                for name, heading, unit, digits, absolute, relative in METRICS:
                    if number(values.get(name)) and number(base.get(name)):
                        if abs(values[name] - base[name]) > absolute + relative * abs(base[name]):
                            flagged.append(dict(metric=name, heading=heading, unit=unit, value=values[name], standin=base[name]))
            mission['differs_from_standin'] = flagged
        out['mission'] = mission

    # ---------------------------------------------------------------- verdict
    statuses = [out['stages'][s]['status'] for s in STAGES]
    if any(s == 'failed' for s in statuses) or out['problems']:
        out['verdict'] = 'not passed'
    elif all(s == 'passed' for s in statuses):
        out['verdict'] = {'isaac': 'confirmed in Isaac Sim', 'stand-in': 'passed on a stand-in (not Isaac Sim)',
                          'dry': 'nothing was simulated (dry run)'}.get(out['engine_kind'], 'passed, engine not identified')
    else:
        done = [s for s in STAGES if out['stages'][s]['status'] == 'passed']
        out['verdict'] = 'incomplete: ' + (', '.join(done) + ' passed' if done else 'nothing passed') + '; not run: ' + \
                         ', '.join(s + (' (interrupted)' if out['stages'][s]['status'] == 'interrupted' else '')
                                   for s in STAGES if out['stages'][s]['status'] != 'passed')
    return out


def assess_environment(folder):
    out = dict(folder=folder.name)
    try:
        manifest = read_json(folder / 'manifest.json')
        out['status'] = latest_stages(manifest).get('environment', {}).get('status')
        case = read_json(folder / 'environment' / 'case.json')
        out['engine_kind'], out['engine'] = engine_of(case)
        out['versions'] = case.get('versions')
        if case.get('failure'):
            out['failure'] = case['failure']
        report = read_json(folder / 'environment' / 'environment.json')
        out.update(missing_required=report.get('missing_required', []), missing_optional=report.get('missing_optional', []),
                   notes=report.get('notes', []))
    except (OSError, ValueError, KeyError) as exc:
        out['unreadable'] = f'{type(exc).__name__}: {exc}'
    return out


# ====================================================================== all runs
def assess(paths, package=ROOT, reference=None, expected=None):
    """Assess the runs under ``paths``.  ``expected``: the teams that should be there (default: those of the campaign,
    or the teams found).  Returns the assessment (a plain dict)."""
    package = Path(package)
    reference = Path(reference) if reference else package / 'reference_results'
    temporary, campaigns, environments, runs = [], [], [], {}
    try:
        for path in paths:
            campaign, environment, found = find_runs(unpack(path, temporary))
            if campaign:
                campaigns.append(campaign)
            if environment:
                environments.append(environment)
            for team, run in found.items():
                if team in runs:
                    raise ValueError(f'two runs of {team}: {runs[team]} and {run}')
                runs[team] = run
        if not runs and not environments:
            raise ValueError('no run of the Isaac Sim workflow was found (no manifest.json with stages)')
        wanted = list(expected) if expected else list(dict.fromkeys(t for c in campaigns for t in c.get('teams', {}))) or list(runs)
        teams = {team: assess_run(runs[team], package, reference) for team in runs}
        # names, not paths: the assessment of a bundle reads the same wherever it is made
        result = dict(schema_version=1, inputs=[Path(p).name for p in paths],
                      campaigns=[dict(campaign_id=c.get('campaign_id'), status=c.get('status'), started_utc=c.get('started_utc'),
                                      finished_utc=c.get('finished_utc'), options=c.get('options')) for c in campaigns],
                      environment=[assess_environment(e) for e in environments],
                      teams={t: teams[t] for t in sorted(teams, key=lambda t: (TEAMS.index(t) if t in TEAMS else 99, t))},
                      teams_expected=wanted, teams_missing=[t for t in wanted if t not in teams])
    finally:
        for folder in temporary:
            import shutil
            shutil.rmtree(folder, ignore_errors=True)
    kinds = sorted({t['engine_kind'] for t in teams.values()})
    verdicts = [t['verdict'] for t in teams.values()]
    failed = [t['team'] for t in teams.values() if t['verdict'] == 'not passed']
    incomplete = [t['team'] for t in teams.values() if t['verdict'].startswith('incomplete')]
    complete = not failed and not incomplete and not result['teams_missing'] and bool(teams)
    no_start = [e for e in result['environment'] if e.get('status') != 'passed']
    if failed:
        headline, code = 'NOT PASSED: ' + ', '.join(failed), 1
    elif no_start and not teams:
        headline, code = 'NOT PASSED: the environment stage did not pass and no stage of a team was run', 1
    elif not complete:
        parts = []
        if incomplete:
            parts.append('stages not run for ' + ', '.join(incomplete))
        if result['teams_missing']:
            parts.append('no run of ' + ', '.join(result['teams_missing']))
        if not parts:
            parts.append('no stage of a team was run')
        headline, code = 'INCOMPLETE, NOTHING FAILED: ' + '; '.join(parts) + (' (a dry run: nothing was simulated)' if 'dry' in kinds else ''), 4
    elif 'dry' in kinds:
        headline, code = 'NOTHING WAS SIMULATED: a dry run of the coordinators (' + ', '.join(teams) + ')', 3
    elif kinds == ['isaac']:
        headline, code = f'CONFIRMED IN ISAAC SIM: {len(teams)} team(s), every stage passed and every declared bound met', 0
    elif kinds == ['stand-in']:
        headline, code = (f'PASSED ON A STAND-IN, NOT IN ISAAC SIM: {len(teams)} team(s), every stage passed and every declared '
                          'bound met'), 3
    else:
        headline, code = 'PASSED, BUT NOT ALL IN ISAAC SIM: ' + ', '.join(f"{t['team']} ({t['engine_kind']})" for t in teams.values()), 3
    engines = sorted({t['engine'] for t in teams.values()} or {e['engine'] for e in result['environment'] if e.get('engine')})
    closing = sum(1 for t in teams.values() for row in t['stages'].values() if row.get('closing_problems'))
    result.update(headline=headline, exit_code=code, engines=engines, verdicts=verdicts, stages_with_closing_problems=closing)
    return result


# ====================================================================== report
def fmt(x, digits=1):
    return '-' if not number(x) else f'{x:.{digits}f}'


def readme_row(label, m):
    """A row of the results tables of the README."""
    return (f"| {label} | {fmt(m.get('sim_time_s'), 0)} s | {fmt(m.get('payload_position_error_mm'))} mm, "
            f"{fmt(m.get('payload_tilt_mrad'))} mrad | {fmt(m.get('internal_force_max_N'))} N ({fmt(m.get('internal_force_rms_N'))}) | "
            f"{fmt(m.get('grasp_force_max_N'), 0)} N, {fmt(m.get('grasp_moment_max_Nm'))} N m | {fmt(m.get('effort_ratio_max'), 2)} | "
            f"{fmt(m.get('min_wheel_load_N'), 0)} N | {fmt(m.get('traction_use_max'), 2)} | {fmt(m.get('tipping_margin_min_m'), 3)} m | "
            f"{fmt(m.get('placement_error_mm'), 2)} mm |")


def markdown(a):
    lines = ['# Assessment of the Isaac Sim workflow runs', '', f"**{a['headline']}**", '']
    lines += ['Engine: ' + ('; '.join(a['engines']) or 'none'), '']
    if any(t['engine_kind'] == 'stand-in' for t in a['teams'].values()) or any(e.get('engine_kind') == 'stand-in' for e in a['environment']):
        lines += ['A stand-in is not Isaac Sim: it shows that the workflow runs and what it gives on that engine, nothing about '
                  'Isaac Sim.', '']
    for env in a['environment']:
        if 'unreadable' in env:
            lines += [f"Environment stage: {env.get('status') or 'no record'}; its records could not be read ({env['unreadable']})", '']
            continue
        lines.append(f"Environment stage: {env.get('status')} ({env.get('engine')}); required and missing: "
                     f"{', '.join(env['missing_required']) or 'nothing'}; optional and missing: "
                     f"{'; '.join(env['missing_optional']) or 'nothing'}")
        if env.get('failure'):
            lines.append(f"Environment stage ended with: {env['failure']}")
        lines.append('')
    if a['teams_missing']:
        lines += ['No run of: ' + ', '.join(a['teams_missing']), '']
    if a.get('stages_with_closing_problems'):
        lines += [f"The application reported errors while it closed after {a['stages_with_closing_problems']} stage(s) had written "
                  'their records (under Notes). The records of those stages were complete before, and they count as they are.', '']

    lines += ['## Stages', '', '| Team | Platform | ' + ' | '.join(STAGES) + ' | Verdict |', '| --- | --- |' + ' --- |' * (len(STAGES) + 1)]
    for t in a['teams'].values():
        cells = []
        for s in STAGES:
            row = t['stages'][s]
            text = row['status']
            if s == 'preflight' and 'preflight' in t:
                text += f" ({t['preflight']['passed']} of {t['preflight']['checks']} checks)"
            if s == 'mission' and number(row.get('simulated_s')):
                text += f" ({row['simulated_s']:.0f} s simulated)"
            if row.get('attempts', 1) > 1:
                text += f", attempt {row['attempts']}"
            cells.append(text)
        lines.append(f"| `{t['team']}` | {t['platform']} | " + ' | '.join(cells) + f" | {t['verdict']} |")
    lines.append('')
    for t in a['teams'].values():
        for s in STAGES:
            for problem in t['stages'][s].get('problems', []):
                lines.append(f"- `{t['team']}`, {s}: {problem}")
        for problem in t['problems']:
            lines.append(f"- `{t['team']}`: {problem}")
        for check in (t.get('preflight') or {}).get('failed', []):
            lines.append(f"- `{t['team']}`, preflight check failed: {check['name']}: {check['value']} "
                         f"(required {check['relation']} {check['limit']})")
    if lines[-1] != '':
        lines.append('')

    with_preflight = [t for t in a['teams'].values() if (t.get('preflight') or {}).get('worst')]
    if with_preflight:
        lines += ['## Preflight: what PhysX uses and does against the compiled graph', '',
                  'The worst value of every group of checks, without contact; in brackets the value of the PhysX 5.3.1 stand-in.', '',
                  '| | ' + ' | '.join(f"`{t['team']}`" for t in with_preflight) + ' | limit |', '| --- |' + ' --- |' * (len(with_preflight) + 1)]
        for key, heading in PREFLIGHT_GROUPS:
            cells, limit = [], None
            for t in with_preflight:
                row, ref = t['preflight']['worst'].get(key), (t['preflight'].get('worst_standin') or {}).get(key)
                if row is None:
                    cells.append('-')
                    continue
                limit = row['limit'] if limit is None else limit
                cells.append(f"{row['value']:.1e}" + (f" ({ref['value']:.1e})" if ref else ''))
            if any(c != '-' for c in cells):
                lines.append(f"| {heading} | " + ' | '.join(cells) + f" | {'-' if limit is None else format(limit, '.0e')} |")
        lines.append('')
        hulls = {t['team']: t['preflight'].get('cooked_hull_vertices') for t in with_preflight if t['preflight'].get('cooked_hull_vertices')}
        if hulls:
            lines += ['Roller hulls, vertices authored and cooked: ' + '; '.join(
                f"`{team}` " + ', '.join(f'{a_} and {c_}' for a_, c_ in rows) for team, rows in hulls.items()) + '.', '']

    with_mission = [t for t in a['teams'].values() if 'metrics' in t.get('mission', {})]
    if with_mission:
        lines += ['## Missions', '',
                  '| Run | Simulated | Payload error while carried | Internal force, max (rms) | Largest grasp force and moment | '
                  'Largest effort ratio | Smallest wheel load | Largest traction use | Smallest tipping margin | Placement error |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for t in with_mission:
            label = {'isaac': 'Isaac Sim', 'stand-in': 'Stand-in', 'dry': 'Not simulated'}.get(t['engine_kind'], 'Unknown engine')
            lines.append(readme_row(f"{label}, `{t['team']}`", t['mission']['metrics']))
        lines += ['', 'Declared bounds of the missions:', '']
        for t in with_mission:
            unmet = [f"{k} ({b['value']} against {b['bound']})" for k, b in t['mission'].get('bounds', {}).items() if not b['ok']]
            lines.append(f"- `{t['team']}`: " + ('all met' if not unmet else 'NOT met: ' + ', '.join(unmet))
                         + f"; mission status {t['mission'].get('status')}; saturated steps {t['mission'].get('saturated_steps')}")
        lines += ['', '### Beside the reference runs of the package', '']
        for t in with_mission:
            m, refs = t['mission']['metrics'], t['mission'].get('references', {})
            lines += [f"`{t['team']}`", '', '| | this run | ' + ' | '.join(label for _, label, _ in REFERENCES) + ' |',
                      '| --- | --- |' + ' --- |' * len(REFERENCES)]
            for name, heading, unit, digits, _, _ in METRICS:
                cells = [fmt((refs.get(key) or {}).get(name), digits) for key, _, _ in REFERENCES]
                lines.append(f"| {heading}{', ' + unit if unit else ''} | {fmt(m.get(name), digits)} | " + ' | '.join(cells) + ' |')
            if 'skid_yaw_rate_ratio' in m:
                cells = [', '.join(f'{x:.3f}' for x in (refs.get(key) or {}).get('skid_yaw_rate_ratio', [])) or '-' for key, _, _ in REFERENCES]
                lines.append('| yaw rate over that of the skid model, by base | ' + ', '.join(f'{x:.3f}' for x in m['skid_yaw_rate_ratio'])
                             + ' | ' + ' | '.join(cells) + ' |')
            lines += ['', 'The compiled model against what the plant reported:', '',
                      '| | this run | ' + ' | '.join(label for _, label, _ in REFERENCES) + ' |', '| --- | --- |' + ' --- |' * len(REFERENCES)]
            for name, heading, digits in CONSISTENCY:
                cells = [fmt(((refs.get(key) or {}).get('consistency') or {}).get(name), digits) for key, _, _ in REFERENCES]
                lines.append(f"| {heading} | {fmt((m.get('consistency') or {}).get(name), digits)} | " + ' | '.join(cells) + ' |')
            lines.append('')
            x = t['mission']
            facts = []
            if x.get('sensor_balance_vs_physx_joint_force'):
                s = x['sensor_balance_vs_physx_joint_force']
                facts.append(f"wrist-sensor balance against PhysX's joint reaction {s['force_rms_N']:.4f} N rms of {s['of_N']:.0f} N")
            if number(x.get('clamp_stability_ratio')):
                facts.append(f"clamp stability ratio {x['clamp_stability_ratio']:.2f}, largest deflection {fmt(x.get('clamp_max_deflection_mm'))} mm")
            if number(x.get('tyre_stability_ratio')):
                facts.append(f"tyre model stability ratio {x['tyre_stability_ratio']:.2f}")
            report = x.get('contact_report') or {}
            if report and report.get('wheel_collider') is False:
                facts.append(f"contact report: no wheel collider (the tyres carry the wheels), payload on "
                             f"{report.get('pads_in_contact')} support pads")
            elif report:
                facts.append(f"contact report: {report.get('wheels_in_contact')} wheels, friction anchors "
                             f"{'reported' if report.get('friction_reported') else 'not reported'}")
            if facts:
                lines += ['; '.join(facts) + '.', '']
            if not x.get('standin_comparable', True):
                lines += [f"The PhysX 5.3.1 stand-in ran this team with `{x.get('standin_contact')}` wheel contacts, this run with "
                          f"`{x.get('contact')}`: its ground loads, wheel efforts and base tracking are not a reference for this "
                          'run, and no difference is marked.', '']
            for f in x.get('differs_from_standin', []):
                lines.append(f"- differs from the stand-in beyond the review threshold: {f['heading']} {f['value']:.4g} "
                             f"against {f['standin']:.4g} {f['unit']}".rstrip())
            if x.get('differs_from_standin'):
                lines.append('')

    lines += ['## What the runs were made with', '']
    for t in a['teams'].values():
        v, s = t.get('versions') or {}, t.get('settings') or {}
        lines.append(f"- `{t['team']}`: {t['engine']}; Python {v.get('python')}, NumPy {v.get('numpy')}, SciPy {v.get('scipy')}, "
                     f"{v.get('platform')}; dt {s.get('dt')}, {s.get('solver')} {s.get('position_iterations')}/"
                     f"{s.get('velocity_iterations')}, {t.get('contact_used') or s.get('contact')} wheel contact; files "
                     + ('identical to this package' if t['source_identical'] else 'DIFFERENT from this package: '
                        + ', '.join((t.get('source_differs') or {}).get('differing', []) + (t.get('source_differs') or {}).get('not_in_this_package', []))))
    lines.append('')
    notes = [f"- `{t['team']}`: {n}" for t in a['teams'].values() for n in t['notes']]
    if notes:
        lines += ['## Notes', ''] + notes + ['']
    used = sorted({t.get('contact_used') for t in a['teams'].values() if t.get('tyres') and t.get('contact_used')})
    tyre_lines = [TYRE_NOT_TESTED[c] for c in used if c in TYRE_NOT_TESTED]
    lines += ['## What these runs do not test', ''] + [f'- {x}' for x in tyre_lines + list(NOT_TESTED)] + ['']
    return '\n'.join(lines)


def write(a, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'assessment.json').write_text(json.dumps(a, indent=1) + '\n', encoding='utf-8')
    (folder / 'ASSESSMENT.md').write_text(markdown(a), encoding='utf-8')
    return folder / 'ASSESSMENT.md'


def summary(a):
    """A few console lines: the headline and one line per team."""
    lines = [a['headline']]
    for t in a['teams'].values():
        states = ', '.join(f"{s} {t['stages'][s]['status']}" for s in STAGES)
        lines.append(f"  {t['team']:<11} {t['verdict']}  [{states}]")
    for team in a['teams_missing']:
        lines.append(f'  {team:<11} no run')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='+', type=Path, help='campaign folders, campaign bundles (zip) or run folders of single teams')
    ap.add_argument('--reference', type=Path, default=None, help='folder of the reference results (default: reference_results of this package)')
    ap.add_argument('--output', type=Path, default=None, help='folder for ASSESSMENT.md and assessment.json (default: the first input folder, or the folder of the first bundle)')
    ap.add_argument('--teams', nargs='+', default=None, help='the teams that should be there (default: those of the campaign)')
    args = ap.parse_args(argv)
    try:
        a = assess(args.paths, reference=args.reference, expected=args.teams)
    except (OSError, ValueError) as exc:
        print(f'The input cannot be assessed: {exc}', file=sys.stderr)
        return 2
    first = args.paths[0].expanduser().resolve()
    target = args.output or (first if first.is_dir() else first.parent)
    path = write(a, target)
    print(summary(a))
    print(f'Report: {path}')
    return a['exit_code']


if __name__ == '__main__':
    sys.exit(main())
