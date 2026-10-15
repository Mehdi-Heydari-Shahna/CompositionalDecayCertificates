#!/usr/bin/env python
"""Isaac Sim workflow of the cooperative team: staged runs of the beam mission in PhysX.

Run it with the Python of an Isaac Sim installation (where ``from isaacsim import
SimulationApp`` works), or point it at one:

    python workflows/isaac/run_isaac.py --team team2 --headless
    python workflows/isaac/run_isaac.py --team team2 --headless --isaac-python <path to Isaac's python>

Without ``--stage`` the stages run in order, each in its own process, and stop at the
first one that fails:

    selftest   the controller stack reproduces its reference numbers in this Python (no Isaac)
    preflight  what PhysX uses is read back and compared with the compiled graph; kinematics;
               one-step free-flight forward dynamics
    settle     first seconds of the mission: the team at rest on its wheels, then starting to move
    carry      a stretch of the route with the payload already in the clamps
    mission    the complete mission, audited like the MuJoCo and Pinocchio runs

``--stages`` runs several named stages, and ``--run-folder`` names the folder of the run:
a new one, or one of an earlier run of the same team, which is then continued (a stage
that is run again keeps its earlier attempt under ``attempts/``).  Two more stages are
not part of the sequence: ``--stage environment`` starts Isaac Sim, builds nothing, and
lists the modules, functions and classes of the installation that the workflow uses;
``--stage pickup`` runs the mission up to the end of the extraction.

``workflows/isaac/run_campaign.py`` runs this for all teams of the three platforms.

Everything is written to a new folder under ``results/`` (or to the folder named with
``--run-folder``) and bundled into ``results_bundle.zip`` (without the scene files),
which is what to keep or send on.  The controller, the audit and the scene generator are
the simulator-neutral ones of this package; they need NumPy and SciPy only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# small matrices: one BLAS thread is faster and leaves the cores to PhysX (set before NumPy is imported)
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

STAGES = ('selftest', 'preflight', 'settle', 'carry', 'mission')
EXTRA_STAGES = ('pickup', 'environment')
CONTACT_CHOICES = ('auto', 'tyre', 'rigid', 'compliant')
MISSION_OPTIONS = dict(settle=dict(duration=3.0), carry=dict(start='navigate', start_l=1.0, duration=14.0),
                       pickup=dict(stop_after='stow'), mission={})
PARTIAL_EXEMPT = ('mission_complete', 'supports_verified')        # bounds that only a complete mission can meet
SELFTEST_TOLERANCE = 1e-7
# printed by a stage once its record is on disk; what the application reports after this line happened while it closed
CLOSING_MARK = 'Stage record written; closing the application.'
CLOSING_PATIENCE_S = 180.                                          # for the application to close after that line


def utc():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    def convert(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        raise TypeError(type(x).__name__)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=1, default=convert) + '\n', encoding='utf-8')
    for attempt in range(6):                                           # a scanner or a sync client may hold the file briefly
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.2 * (attempt + 1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprints():
    """sha256 of the files a stage executes or reads: the kernel, this file, the team compiler, the mission, the
    plants and scene generators, the Isaac adapters, the inputs and the reference values of the self-test."""
    files = [ROOT / 'vendor/pacdm_original.py', HERE / 'run_isaac.py']
    for folder, pattern in ((ROOT / 'teamc', '*.py'), (ROOT / 'mission', '*.py'), (ROOT / 'backends', '*.py'),
                            (HERE / 'isaac_team', '*.py'), (ROOT / 'inputs', '**/*.json'), (HERE / 'reference', '*.json')):
        files += sorted(folder.glob(pattern))
    return {str(p.relative_to(ROOT)).replace(os.sep, '/'): sha256(p) for p in files if p.is_file()}


# ====================================================================== arguments
def parse_args(argv=None):
    # no abbreviations: the campaign sets some options itself and refuses them after --, by their full names
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument('--team', default='team2',
                   help='team record in inputs/teams (team2, team3, ridgeback2, ridgeback3, husky2, husky3)')
    p.add_argument('--stage', default='all', choices=('all',) + STAGES + EXTRA_STAGES)
    p.add_argument('--stages', nargs='+', default=None, choices=STAGES + EXTRA_STAGES, metavar='STAGE',
                   help='several stages, run in the order given (instead of --stage)')
    p.add_argument('--headless', action='store_true', help='no Isaac window (still needs the Isaac runtime)')
    p.add_argument('--output', type=Path, default=ROOT / 'results', help='parent folder of the new run folder')
    p.add_argument('--run-folder', type=Path, default=None,
                   help='folder of the run instead of a new one under --output: created, or continued if it holds a run of this team')
    runtime = p.add_mutually_exclusive_group()
    runtime.add_argument('--isaac-python', type=Path, help='Python executable of an Isaac Sim environment (default: this Python)')
    runtime.add_argument('--isaac-root', type=Path, help='Isaac Sim folder containing python.bat / python.sh')
    p.add_argument('--keep-going', action='store_true', help='run the remaining stages after a failed one')
    p.add_argument('--stall-timeout', type=float, default=30., metavar='MINUTES',
                   help='end a stage whose process has printed nothing for this long, or whose application has not closed '
                        '3 minutes after the record of the stage was written (default 30; 0: wait for ever)')
    p.add_argument('--no-bundle', action='store_true', help='do not write results_bundle.zip')
    # physics
    p.add_argument('--dt', type=float, default=.002, help='physics timestep in seconds')
    p.add_argument('--solver', choices=('PGS', 'TGS'), default='PGS')
    p.add_argument('--position-iterations', type=int, default=None, help='default: 64 with PGS, 8 with TGS')
    p.add_argument('--velocity-iterations', type=int, default=None, help='default: 8 with PGS, 1 with TGS')
    p.add_argument('--contact', choices=CONTACT_CHOICES, default='auto',
                   help='wheel contact: auto (the default: tyre for conventional wheels, rigid for mecanum rollers); tyre '
                        '(conventional wheels carried by the declared tyre springs, no wheel collider); rigid (PhysX contacts '
                        'under the rollers or the tyres); compliant (PhysX compliant contacts with the declared compliance)')
    p.add_argument('--roller-hull', type=int, nargs=2, default=(21, 30), metavar=('RINGS', 'AROUND'),
                   help='profile rings and vertices per ring of the roller collision hulls (two rings per hull, at most 32 around)')
    p.add_argument('--clamp', type=float, nargs=4, default=None, metavar=('K', 'D', 'KR', 'DR'),
                   help='clamp stiffness N/m, damping N s/m, rotational stiffness N m/rad, rotational damping N m s/rad')
    p.add_argument('--physx-threads', type=int, default=None, help='CPU PhysX worker threads (default: Isaac setting)')
    p.add_argument('--max-time', type=float, default=400., help='largest simulated time of a mission stage in seconds')
    # presentation and lifecycle
    p.add_argument('--render-hz', type=float, default=25., help='viewport refresh per simulated second (with a window only)')
    p.add_argument('--export-scene', action='store_true', help='write scene.usda in every stage (default: preflight only)')
    p.add_argument('--no-figures', action='store_true', help='do not draw figures even if matplotlib is installed')
    p.add_argument('--shutdown-mode', choices=('auto', 'fast', 'graceful'), default='auto',
                   help='how the application is closed: auto (the fast way where it keeps an explicit exit status, the graceful '
                        'way elsewhere), graceful, or fast (refused where the application has no fast way)')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--run-id', default=None, help=argparse.SUPPRESS)
    p.add_argument('--make-reference', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if not 1e-4 <= a.dt <= .004 or abs(round(1. / a.dt) - 1. / a.dt) > 1e-9:
        p.error('--dt must be between 0.0001 and 0.004 s and a whole number of steps per second (PhysX takes steps per second)')
    if a.position_iterations is None:
        a.position_iterations = 64 if a.solver == 'PGS' else 8
    if a.velocity_iterations is None:
        a.velocity_iterations = 8 if a.solver == 'PGS' else 1
    if a.roller_hull[0] < 3 or not 8 <= a.roller_hull[1] <= 32:
        p.error('--roller-hull needs at least 3 rings and 8 to 32 vertices per ring')
    if a.position_iterations < 1 or a.velocity_iterations < 0 or max(a.position_iterations, a.velocity_iterations) > 255:
        p.error('iteration counts must be within 1..255 (position) and 0..255 (velocity)')
    if a.stall_timeout < 0.:
        p.error('--stall-timeout must not be negative')
    if a.stages and a.stage != 'all':
        p.error('give --stage or --stages, not both')
    if a.stages and len(set(a.stages)) != len(a.stages):
        p.error('--stages names a stage twice')
    if a.worker and (a.stage == 'all' or a.stages or a.run_folder or not a.run_id):
        p.error('--worker needs one --stage and a --run-id')
    return a


def physics_cli(a):
    cli = ['--dt', repr(a.dt), '--solver', a.solver, '--position-iterations', str(a.position_iterations),
           '--velocity-iterations', str(a.velocity_iterations), '--contact', a.contact,
           '--roller-hull', str(a.roller_hull[0]), str(a.roller_hull[1]), '--max-time', repr(a.max_time),
           '--render-hz', repr(a.render_hz), '--shutdown-mode', a.shutdown_mode]
    if a.clamp:
        cli += ['--clamp'] + [repr(x) for x in a.clamp]
    if a.physx_threads is not None:
        cli += ['--physx-threads', str(a.physx_threads)]
    for flag in ('headless', 'export_scene', 'no_figures'):
        if getattr(a, flag):
            cli.append('--' + flag.replace('_', '-'))
    return cli


# ====================================================================== self-test (no Isaac)
def selftest_values(team, dt=.002):
    """Numbers of the controller stack at the start of the mission (compared across Python environments)."""
    import numpy as np
    from mission.controller import TeamController
    from mission.runner import carry_configuration, start_configuration
    from teamc.io import INPUTS, load_team, read_json

    comp = load_team(team)
    mission = read_json(INPUTS / 'missions' / f'{team}.json')
    ctrl = TeamController(comp, mission, dt, slow_every=max(int(round(.01 / dt)), 1))
    q0 = start_configuration(comp, mission, ctrl.model)
    zero = np.zeros(comp.tree.n)
    meas = dict(t=0., q=q0, v=zero, wrench={rid: np.zeros(6) for rid in comp.robots}, obstacle=None)
    ctrl.start(meas)
    ctrl.slow(meas)
    efforts = ctrl.fast(meas)
    M = comp.tree.mass(q0)
    qc = carry_configuration(comp, mission, ctrl, 0.)
    return dict(coordinates=comp.tree.n, ports=len(comp.cmg['ports']), q_start=q0, mass_diagonal=np.diag(M),
                mass_frobenius=float(np.linalg.norm(M)), bias=comp.tree.bias(q0, zero), efforts=np.asarray(efforts, float),
                q_carry=qc, total_mass_kg=comp.tree.total_mass())


def selftest(args, result):
    import numpy as np
    import scipy
    result['versions'] = dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                              platform=platform.platform())
    kernel = sha256(ROOT / 'vendor/pacdm_original.py')
    reference = json.loads((HERE / 'reference' / f'{args.team}.json').read_text(encoding='utf-8'))
    values = selftest_values(args.team, reference['dt'])
    checks = [dict(name='PACDM kernel unchanged (sha256)', passed=kernel == reference['kernel_sha256'], value=kernel)]
    for key, want in reference['values'].items():
        got, want = np.asarray(values[key], float), np.asarray(want, float)
        if got.shape != want.shape:
            checks.append(dict(name=key, passed=False, value=f'shape {got.shape} instead of {want.shape}'))
            continue
        error = float(np.max(np.abs(got - want) / np.maximum(1., np.abs(want))))
        checks.append(dict(name=key, passed=error <= SELFTEST_TOLERANCE, value=error, limit=SELFTEST_TOLERANCE))
    result.update(checks=checks, passed=all(c['passed'] for c in checks),
                  reference=dict(numpy=reference['numpy'], scipy=reference['scipy'], python=reference['python']),
                  scope='the compiled model and the controller reproduce the reference numbers of the package in this Python')
    for c in checks:
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {c['name']}: {c['value']}", flush=True)


def make_reference(args):
    import numpy as np
    import scipy
    values = selftest_values(args.team, args.dt)
    out = HERE / 'reference' / f'{args.team}.json'
    out.parent.mkdir(exist_ok=True)
    save_json(out, dict(team=args.team, dt=args.dt, kernel_sha256=sha256(ROOT / 'vendor/pacdm_original.py'),
                        python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, values=values))
    print('written', out)
    return 0


# ====================================================================== Isaac stages
def resolve_contact(requested, comp):
    """The wheel contact a team is simulated with: ``(contact, note)``.

    ``auto`` and ``tyre``: the tyres carry conventional wheels (``tyre``); rollers of mecanum wheels stand on rigid
    contacts (``rigid``), which is what ``tyre`` means for a team without conventional wheels.  ``rigid`` and
    ``compliant`` apply to every wheel as asked.
    """
    conventional = any(r.drive == 'friction' for r in comp.robots.values())
    if requested in ('auto', 'tyre'):
        if conventional:
            return 'tyre', None
        return 'rigid', (None if requested == 'auto' else 'tyre asked for a team without conventional wheels: rigid roller contacts')
    return requested, None


def positive_control(comp, plant, monitor, report, A, result):
    """Checks that the plant reported what a stage must see; returns the unmet ones.

    Without a positive control "no forbidden contact" would mean nothing.  Wheels on PhysX contacts: every wheel
    reported a terrain contact, the sign of the impulses was learned, and the wheel loads add up to the compiled
    ground load.  Tyres that carry their wheels: every tyre carried load and the tyre loads add up to the compiled
    ground load; the contact report shows the payload on its supports wherever the stage has it resting there (the
    ``carry`` stage holds the payload in the clamps throughout and has no contact to show), and no clamp contact came
    before the sign of the impulses was known (it would be missing from the wrist sensors).
    """
    import numpy as np
    unmet = []
    wheels = sum(len(r.wheels) for r in comp.robots.values())
    weight = comp.tree.total_mass() * float(np.linalg.norm(comp.tree.gravity))
    total = report['consistency'].get('total_normal_load_vs_plant_N') or {}
    rested = bool(np.any(np.asarray(A['mode_rest']) != ''))
    if plant.tyres_carry:
        tyres = plant.tyres
        control = 'tyre loads' + (' and the payload on its supports' if rested else '')
        if len(tyres.loaded) != len(tyres.wheels):
            unmet.append(f'tyres: {len(tyres.loaded)} of {len(tyres.wheels)} carried load')
        elif total.get('rms') is None or total['rms'] > .05 * weight:
            unmet.append(f"tyres: tyre loads differ from the compiled ground load by {total.get('rms')} N rms")
        if rested:
            if not monitor.pads_seen:
                unmet.append('contact report: the payload on its supports was not reported')
            elif monitor.impulse_on_first == 0.:
                unmet.append('contact report: the sign of the impulses was not learned')
        if monitor.unsigned_tool_steps:
            unmet.append(f'contact report: clamp contacts in {monitor.unsigned_tool_steps} steps before the sign of the '
                         'impulses was learned (left out of the wrist sensors)')
    else:
        control = 'wheel contacts'
        if monitor.points == 0:
            unmet.append('contact report: no contact point was reported')
        elif monitor.impulse_on_first == 0. or len(monitor.wheels_seen) != wheels:
            unmet.append(f'contact report: {len(monitor.wheels_seen)} of {wheels} wheels reported a terrain contact')
        elif total.get('rms') is None or total['rms'] > .05 * weight:
            unmet.append(f"contact report: wheel loads differ from the compiled ground load by {total.get('rms')} N rms")
    result['positive_control'] = dict(of=control, unmet=list(unmet))
    return unmet


def enable_tensor_extensions(result):
    """The tensor API and its PhysX side, by their extension names (the World enables them as well; never fatal)."""
    notes = []
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
    except Exception as exc:                                          # noqa: BLE001
        result['tensor_extension_note'] = f'{type(exc).__name__}: {exc}'
        return
    for name in ('omni.physics.tensors', 'omni.physx.tensors'):
        try:
            manager.set_extension_enabled_immediate(name, True)
        except Exception as exc:                                      # noqa: BLE001
            notes.append(f'{name}: {type(exc).__name__}: {exc}')
    if notes:
        result['tensor_extension_note'] = '; '.join(notes)


def execute(args, app, result, out):
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from pxr import UsdPhysics

    from backends import usd_export
    from backends.isaac_plant import IsaacPlant
    from isaac_team import preflight
    from isaac_team.native import ContactMonitor, NativeTeam
    from isaac_team.runtime import cleanup_actions, create_numpy_view, current_stage_id, select_physx_engine, version_info
    from mission.runner import start_configuration
    from mission.workflow import run_and_report
    from teamc.interfaces import TeamModel
    from teamc.io import INPUTS, load_team, read_json

    dt = float(args.dt)
    comp = load_team(args.team)
    mission = read_json(INPUTS / 'missions' / f'{args.team}.json')
    q_start = start_configuration(comp, mission, TeamModel(comp))
    contact, contact_note = resolve_contact(args.contact, comp)
    result['settings']['contact_used'] = contact
    if contact_note:
        result['contact_note'] = contact_note
        print(f'Contact: {contact_note}', flush=True)
    print(f'Wheel contact: {contact}' + (' (the tyres carry the wheels; no wheel collider)' if contact == 'tyre' else ''), flush=True)
    native = monitor = world = None
    errors = []
    result['versions'] = version_info()
    try:
        result['execution_phase'] = 'physics_setup'
        result['physics_runtime'] = select_physx_engine()
        enable_tensor_extensions(result)
        # rendering_dt equal to physics_dt: one World.step is one physics step, with or without a rendered frame
        world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1., backend='numpy', device='cpu')
        # The same settings are authored on the physics scene below, which is what PhysX reads.  The calls of the
        # physics context come first because they are the way of the application; one that fails is noted, not fatal.
        notes = []
        try:
            context = world.get_physics_context()
            for name, value in (('enable_gpu_dynamics', False), ('set_broadphase_type', 'MBP'), ('set_solver_type', args.solver)):
                try:
                    getattr(context, name)(value)
                except Exception as exc:                              # noqa: BLE001
                    notes.append(f'{name}({value!r}): {type(exc).__name__}: {exc}')
        except Exception as exc:                                      # noqa: BLE001
            notes.append(f'get_physics_context: {type(exc).__name__}: {exc}')
        if notes:
            result['physics_context_notes'] = notes
        stage = omni.usd.get_context().get_stage()
        # PhysX reports problems of a scene (parsing, cooking, articulation setup) on its error stream
        pump = None
        try:
            from omni.physx import get_physx_interface
            stream = get_physx_interface().get_error_event_stream()
            subscription = stream.create_subscription_to_pop(lambda e: errors.append(dict(type=str(e.type), payload=str(e.payload))))
            pump = getattr(stream, 'pump', None)
            result['physx_error_stream'] = 'subscribed'
        except Exception as exc:                                      # noqa: BLE001
            subscription = None
            result['physx_error_stream'] = f'unavailable: {type(exc).__name__}: {exc}'

        def physx_errors(where):
            if callable(pump):
                pump()
            if errors:
                raise RuntimeError(f'PhysX reported an error ({where}): {errors[-1]}')

        result['execution_phase'] = 'scene_authoring'
        options = dict(roller_hull=(args.roller_hull[0], args.roller_hull[1], True), contact=contact, solver=args.solver,
                       position_iterations=args.position_iterations, velocity_iterations=args.velocity_iterations)
        meta = usd_export.build_usd(stage, comp, q_start, options)
        scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
        if len(scenes) > 1:
            raise RuntimeError(f'expected the one physics scene of the World, found {[str(p.GetPath()) for p in scenes]}')
        scene = scenes[0] if scenes else UsdPhysics.Scene.Define(stage, '/physicsScene').GetPrim()
        result['solver'] = dict(usd_export.author_scene_settings(scene, comp.cmg['gravity_m_s2'], dict(options, timestep=dt)),
                                scene=str(scene.GetPath()), scene_created_by='World' if scenes else 'this workflow')
        layer_data = dict(stage.GetRootLayer().customLayerData)
        layer_data.update(generator='Team_CMG_PACDM backends/usd_export.py', team=args.team, run_id=args.run_id, stage=args.stage)
        stage.GetRootLayer().customLayerData = layer_data
        audit = usd_export.audit_stage(stage, comp, meta)
        save_json(out / 'usd_audit.json', audit)
        save_json(out / 'scene_meta.json', {k: v for k, v in meta.items() if k not in ('colliders', 'expected')})
        if not audit['passed']:
            raise RuntimeError('the authored USD differs from the compiled graph: ' + '; '.join(audit['failures'][:5]))
        if args.export_scene or args.stage == 'preflight':
            stage.GetRootLayer().Export(str(out / 'scene.usda'))
        result['scene'] = dict(counts=meta['counts'], roller=meta.get('roller'), physx_schema_bindings=meta['physx_schema_bindings'],
                               physx_schemas_applied_by_bindings=meta['physx_schemas_applied_by_bindings'],
                               limitations=meta['limitations'])
        if not args.headless:
            try:
                from isaacsim.core.utils.viewports import set_camera_view
                c = q_start[np.asarray(comp.payload['base'])[:3]]
                set_camera_view(eye=np.array([c[0] + 9., c[1] - 11., 8.]), target=np.array([c[0], c[1] + 2., 1.]),
                                camera_prim_path='/OmniverseKit_Persp')
            except Exception as exc:                                  # noqa: BLE001 (presentation only)
                result['camera_note'] = f'{type(exc).__name__}: {exc}'

        result['execution_phase'] = 'physics_initialization'
        authored = usd_export.read_scene_settings(scene)
        world.reset()
        physx_errors('while the scene was loaded')
        stage_id = current_stage_id(expected_stage=stage)
        print(f'Binding NumPy PhysX tensors to USD stage {stage_id}', flush=True)
        sim, route = create_numpy_view(stage_id, world)
        sim.set_subspace_roots('/')
        result['physics_runtime'].update(stage_id=stage_id, tensor_route=route, device='cpu')
        after = result['solver']['after_reset'] = usd_export.read_scene_settings(scene)
        # what the reset of the World changed on the physics scene (reported; the stages show what it means)
        result['solver']['changed_by_reset'] = {key: [authored.get(key), after.get(key)]
                                                for key in sorted(set(authored) | set(after)) if authored.get(key) != after.get(key)}
        native = NativeTeam(sim, meta, comp, dt)
        result['native'] = {rid: dict(links=x.nlink, dofs=x.ndof, dof_order=x.dofs, link_order=x.links)
                            for rid, x in native.robots.items()}
        result['native']['cart'] = native.cart_mode
        monitor = ContactMonitor(meta, dt)
        physx_errors('while the tensor views were created')

        render_stride = max(int(round(1. / (args.render_hz * dt))), 1)
        clock = dict(k=0, time=float(world.current_time), index=int(world.current_time_step_index))

        def advance():
            if not app.is_running():
                raise RuntimeError('Isaac Sim was closed before the stage completed')
            if not world.is_playing():
                raise RuntimeError('the Isaac timeline stopped during the stage')
            clock['k'] += 1
            world.step(render=(not args.headless and clock['k'] % render_stride == 0))
            if clock['k'] == 1 or clock['k'] % 250 == 0:
                physx_errors(f'at step {clock["k"]}')
                steps = int(world.current_time_step_index) - clock['index']
                elapsed = float(world.current_time) - clock['time']
                if steps != clock['k'] or abs(elapsed - clock['k'] * dt) > max(1e-6, 1e-6 * clock['k'] * dt):
                    raise RuntimeError(f'the PhysX clock and the step count differ: {steps} steps, {elapsed} s after {clock["k"]} calls')

        latch_params = None if not args.clamp else dict(zip(
            ('stiffness_N_m', 'damping_N_s_m', 'rot_stiffness_Nm_rad', 'rot_damping_Nm_s_rad'), args.clamp))
        plant = IsaacPlant(comp, native, monitor, advance, dt, latch_params, wheel_load='tyre' if contact == 'tyre' else 'contact')
        result['clamp'] = dict(plant.latches.p)
        plant.reset(q_start)
        if plant.tyres:                                                # conventional wheels: the tyre model applied by the plant
            w = plant.tyres.wheels[0]
            result['tyre'] = dict(model='backends/tyre.py', friction=w['mu'], stiffness_N_m=w['k'], damping_N_s_m=w['c'],
                                  relaxation_length_m=w['relax'], stability_ratio=plant.tyre_ratio,
                                  carries_wheels=plant.tyres_carry)
            if plant.tyres_carry:
                result['tyre'].update(normal_stiffness_N_m=w['kn'], normal_damping_N_s_m=w['cn'])

        preflight_reference = None
        if args.stage == 'preflight':
            result['execution_phase'] = 'preflight'
            print('Native PhysX checks against the compiled graph...', flush=True)
            bounds = mission['bounds']
            report = preflight.run(plant, native, comp, meta, q_start, stage=stage, stage_id=stage_id,
                                   load=(bounds['grasp_force_N'] / 3., bounds['grasp_moment_Nm'] * 8. / 45.))
            report.update(run_id=args.run_id, team=args.team, generated_utc=utc())
            save_json(out / 'preflight.json', report)
            for c in report['checks']:
                if not c['passed']:
                    print(f"  FAIL  {c['name']}: {c['value']} (required {c['relation']} {c['limit']})", flush=True)
            print(f"preflight: {sum(c['passed'] for c in report['checks'])} of {len(report['checks'])} checks passed", flush=True)
            for line in report['details']['unavailable']:
                print(f'  not available here (check skipped): {line}', flush=True)
            result.update(passed=report['passed'], failed_checks=[c['name'] for c in report['checks'] if not c['passed']],
                          cart=report['details']['cart'], unavailable=report['details']['unavailable'],
                          cooked_hulls=report['details'].get('cooked_hulls'))
            preflight_reference = report['details']['sensor_reference']
        else:
            result['execution_phase'] = 'rollout'
            opts = dict(MISSION_OPTIONS[args.stage], slow_every=max(int(round(.01 / dt)), 1))
            started = time.perf_counter()
            report, A, rec, ctrl, info = run_and_report(
                plant, comp, mission, out, f"{result['engine']}, {args.team} ({args.stage}):", options=opts,
                plant_info=dict(engine=result['engine'], solver=result['solver'], contact=contact,
                                roller_hull=list(args.roller_hull), clamp=dict(plant.latches.p)),
                figures=False if args.no_figures else 'auto', max_time=args.max_time)
            checks = report['bounds']['checks']
            exempt = () if args.stage == 'mission' else PARTIAL_EXEMPT
            unmet = [k for k, c in checks.items() if not c['ok'] and k not in exempt]
            if info['status'] not in ('running', 'complete'):          # held by a guard of the controller, or out of time
                unmet.append(f"status: {info['status']}")
            unmet += positive_control(comp, plant, monitor, report, A, result)
            result.update(passed=not unmet, unmet_bounds=unmet, status=info['status'], simulated_s=info['sim_time'],
                          steps=info['steps'], wall_s=time.perf_counter() - started, saturated_steps=info['saturated_steps'],
                          events=[dict(t=e['t'], phase=e['phase'], event=e['event']) for e in info['events']],
                          consistency=report['consistency'], verification=info.get('verification'))
            if plant.tyres_carry:
                result['tyre'].update(carried_load=sorted('/'.join(k) for k in plant.tyres.loaded),
                                      largest_depth_m=plant.max_tyre_depth)
        if callable(pump):
            pump()
        result.update(clamp_stability_ratio=plant.latch_ratio, clamp_max_deflection_m=plant.max_deflection,
                      cart_follows=plant.cart_moved, cart_mode=native.cart_mode, state_writes=native.writes,
                      contact_report=dict(callbacks=monitor.callbacks, points=monitor.points,
                                          wheels_in_contact=len(monitor.wheels_seen), pads_in_contact=len(monitor.pads_seen),
                                          wheel_collider=not plant.tyres_carry,
                                          clamp_contact_steps_unsigned=monitor.unsigned_tool_steps,
                                          friction_reported=monitor.friction,
                                          friction_note=monitor.friction_note,
                                          impulse_on_first_collider=monitor.impulse_on_first),
                      sensor_reference=preflight_reference if args.stage == 'preflight' else plant.sensor_reference(),
                      world_steps=clock['k'])
        for rid, ref in result['sensor_reference'].items():
            if ref.get('available'):
                print(f"  wrist sensor balance vs PhysX incoming joint force {rid}: force rms {ref['force_rms_N']:.3f} N of "
                      f"{ref['sensor_force_rms_N']:.1f}, moment rms {ref['moment_rms_Nm']:.3f} N m of {ref['sensor_moment_rms_Nm']:.2f} "
                      f"(native sign {ref['native_sign']:+d})", flush=True)
        if errors:
            result['passed'] = False
        result['execution_phase'] = 'finished'
    finally:
        active = sys.exc_info()[0] is not None
        result['physx_errors'] = list(errors)
        actions = []
        if monitor is not None:
            actions.append(('contact_monitor.close', monitor.close))
        if native is not None:
            actions.append(('tensor_views.close', native.close))
        if world is not None:
            actions += [('world.stop', world.stop), ('world.clear_instance', World.clear_instance)]
        result['resource_cleanup'] = cleanup_actions(actions)
        if not result['resource_cleanup']['passed'] and not active:
            raise RuntimeError('resource cleanup failed: ' + '; '.join(result['resource_cleanup']['errors']))


def environment(result, out):
    """What the workflow uses of this Isaac Sim installation, looked up without a scene (``isaac_team/environment.py``)."""
    from isaac_team.environment import probe
    from isaac_team.runtime import version_info

    result['execution_phase'] = 'environment'
    enable_tensor_extensions(result)                                   # as a stage with a scene does before it looks them up
    result['versions'] = version_info()
    report = probe()
    save_json(out / 'environment.json', report)
    for name in report['missing_required']:
        print(f'  MISSING  {name}', flush=True)
    for name in report['missing_optional']:
        print(f'  not available: {name}', flush=True)
    for note in report['notes']:
        print(f'  note: {note}', flush=True)
    looked_up = len(report['required']) + len(report['optional'])
    print(f"environment: {looked_up - len(report['missing_required']) - len(report['missing_optional'])} of {looked_up} modules and "
          f"functions found; missing and required: {len(report['missing_required'])}", flush=True)
    result.update(passed=report['passed'], failed_checks=report['missing_required'], missing_optional=report['missing_optional'],
                  scope='modules, functions and classes of the installation that the workflow uses; nothing is simulated')
    result['execution_phase'] = 'finished'


def worker(args):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(HERE))
    out = args.output / args.stage
    out.mkdir(parents=True, exist_ok=False)
    result = dict(schema_version=1, stage=args.stage, team=args.team, run_id=args.run_id, started_utc=utc(),
                  engine='none (controller self-test)' if args.stage == 'selftest' else 'Isaac Sim / PhysX',
                  settings=dict(dt=args.dt, solver=args.solver, position_iterations=args.position_iterations,
                                velocity_iterations=args.velocity_iterations, contact=args.contact,
                                roller_hull=list(args.roller_hull), clamp=args.clamp, headless=args.headless,
                                physx_threads=args.physx_threads),
                  execution_phase='initialization', passed=False, failure=None)
    app, code = None, 1
    try:
        result['source_sha256'] = source_fingerprints()
        if args.stage == 'selftest':
            selftest(args, result)
        else:
            from isaacsim import SimulationApp
            from isaac_team.runtime import shutdown_configuration
            result['shutdown_policy'] = shutdown_configuration(SimulationApp, args.shutdown_mode)
            config = dict(headless=args.headless, width=1280, height=720, renderer='RayTracedLighting', sync_loads=True,
                          anti_aliasing=0, fast_shutdown=result['shutdown_policy']['fast_shutdown'])
            if args.headless:
                config['disable_viewport_updates'] = True
            if args.physx_threads is not None:
                config['extra_args'] = [f'--/persistent/physics/numThreads={int(args.physx_threads)}']
            result['simulation_app_config'] = config
            app = SimulationApp(config)
            # a stand-in for the application (offline tests) names itself; Isaac Sim has no such attribute
            result['engine'] = getattr(app, 'engine_label', result['engine'])
            if args.stage == 'environment':
                environment(result, out)
            else:
                execute(args, app, result, out)
        code = 0 if result.get('passed') else 3
    except KeyboardInterrupt:
        code = 130
        result.update(passed=False, failure='interrupted', traceback=traceback.format_exc())
    except Exception as exc:                                          # noqa: BLE001 (everything is recorded)
        result.update(passed=False, failure=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        result['finished_utc'] = utc()
        result['execution_exit_code'] = code                           # status before the application closes
        saved = False
        try:
            save_json(out / 'case.json', result)
            saved = True
        except Exception:                                             # noqa: BLE001
            code = code or 1
            traceback.print_exc()
        if app is not None:
            from isaac_team.runtime import close_simulation_app
            if saved:
                print(CLOSING_MARK, flush=True)
            try:
                fast = bool((result.get('shutdown_policy') or {}).get('fast_shutdown'))
                close_simulation_app(app, code, settle_updates=0 if fast else 3)
            except Exception:                                         # noqa: BLE001
                code = code or 1
                traceback.print_exc()
    return code


# ====================================================================== coordinator (standard library only)
def _inside(path, root):
    try:
        path, root = os.path.normcase(os.path.abspath(path)), os.path.normcase(os.path.abspath(root))
        return os.path.commonpath((path, root)) == root
    except (OSError, ValueError):
        return False


def isaac_environment(source, runtime_python):
    """Environment for an Isaac Python that is not this Python: no Conda or virtual-environment influence."""
    env = dict(source)
    prefixes = [v for k, v in env.items() if (k == 'CONDA_PREFIX' or k.startswith('CONDA_PREFIX_')) and v]
    if env.get('VIRTUAL_ENV'):
        prefixes.append(env['VIRTUAL_ENV'])
    for key in ('PATH', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
        if key in env:
            env[key] = os.pathsep.join(p for p in env[key].split(os.pathsep) if p and not any(_inside(p, x) for x in prefixes))
    for key in list(env):
        if key.startswith('CONDA_') or key in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'PYTHONUSERBASE', 'VIRTUAL_ENV'):
            env.pop(key, None)
    env.update(PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    parent = Path(runtime_python).parent
    paths = [str(parent)] + ([str(parent / 'Scripts'), str(parent / 'Library' / 'bin')] if os.name == 'nt' else [])
    env['PATH'] = os.pathsep.join(paths + [env.get('PATH', '')])
    return env


def runtime_command(runtime, arguments):
    if os.name == 'nt' and Path(runtime).suffix.lower() in ('.bat', '.cmd'):
        values = [str(runtime), *map(str, arguments)]
        if any(any(ch in v for ch in '&|<>^%!"\r\n') for v in values):
            raise ValueError('paths passed to a Windows batch launcher must not contain & | < > ^ % ! or quotes')
        return '"' + os.environ.get('COMSPEC', 'cmd.exe') + '" /d /s /c "' + ' '.join('"' + v + '"' for v in values) + '"'
    return [str(runtime), *map(str, arguments)]


def echo(text):
    """Print a line of a child process; a console that cannot show a character must not end the run."""
    try:
        print(text, end='', flush=True)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'), end='', flush=True)


def end_tree(process):
    """End a stage process and everything it started: first the way Ctrl+C would, then without asking."""
    if os.name == 'nt':                                               # a batch launcher: the whole process tree
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
    else:                                                             # the stage has a process group of its own
        for sign, patience in ((signal.SIGINT, 20.), (signal.SIGKILL, 10.)):
            try:
                os.killpg(process.pid, sign)
            except (ProcessLookupError, PermissionError):
                break
            try:
                process.wait(timeout=patience)
            except subprocess.TimeoutExpired:
                pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_logged(command, log_path, env, stall_s=0.):
    """Run a stage process with its output on the console and in its log.

    ``stall_s``: the process is ended when it has printed nothing for this long (0: never), and, once it has
    printed ``CLOSING_MARK``, when it has not ended ``CLOSING_PATIENCE_S`` later (or ``stall_s``, if that is less).
    Returns ``(exit status, why it was ended: None, 'silent' or 'closing')``."""
    with open(log_path, 'w', encoding='utf-8', buffering=1) as log:
        log.write('Command: ' + json.dumps(command) + '\n')
        # a process group of its own (POSIX): the launcher of an Isaac installation starts Python as a child of a shell
        options = dict(start_new_session=True) if os.name != 'nt' else {}
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', **options)
        clock = dict(last=time.monotonic(), closing=None)
        ended = []

        def watch():
            while process.poll() is None:
                now = time.monotonic()
                if clock['closing'] is not None and now - clock['closing'] > min(CLOSING_PATIENCE_S, stall_s):
                    ended.append('closing')
                elif now - clock['last'] > stall_s:
                    ended.append('silent')
                if ended:
                    end_tree(process)
                    return
                time.sleep(min(1., stall_s / 4.))

        if stall_s > 0.:
            threading.Thread(target=watch, daemon=True).start()
        try:
            for line in process.stdout:
                clock['last'] = time.monotonic()
                if clock['closing'] is None and line.strip() == CLOSING_MARK:
                    clock['closing'] = clock['last']
                log.write(line)
                echo(line)
            return process.wait(), (ended[0] if ended else None)
        except BaseException:
            end_tree(process)
            raise
        finally:
            process.stdout.close()


PHYSICS_ERROR = ('[error]', '[fatal]')


def scan_log(path, limit=20):
    """Error lines of a stage log: ``dict(physics, other, closing, marked)``.

    Before the line that marks the record as written (``CLOSING_MARK``), error lines of the physics extensions and
    native faults are problems of the stage (``physics``), and the other error lines are counted (``other``).  After it,
    every error line is something the application reported while it closed (``closing``): the record was complete."""
    physics, closing, other, marked = [], [], 0, False
    try:
        with open(path, encoding='utf-8', errors='replace') as stream:
            for number, line in enumerate(stream, 1):
                if not marked and line.strip() == CLOSING_MARK:
                    marked = True
                    continue
                low = line.lower()
                if any(tag in low for tag in PHYSICS_ERROR) or 'fatal python error' in low or 'windows fatal exception' in low:
                    text = f'log line {number}: {line.strip()[:400]}'
                    if marked:
                        if len(closing) < limit:
                            closing.append(text)
                    elif 'physx' in low or 'physics' in low or 'fatal' in low:
                        if len(physics) < limit:
                            physics.append(text)
                    else:
                        other += 1
    except OSError as exc:
        return dict(physics=[f'the stage log could not be read: {exc}'], other=0, closing=[], marked=False)
    return dict(physics=physics, other=other, closing=closing, marked=marked)


def exit_text(code):
    """An exit status in words; the large ones of Windows also in hexadecimal (0xC0000005: access violation)."""
    return f'{code} (0x{code & 0xffffffff:08X})' if isinstance(code, int) and (code > 255 or code < -255) else str(code)


def bundle(folder):
    path = folder / 'results_bundle.zip'
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and f != path and f.suffix not in ('.usda', '.usdc', '.zip', '.tmp'):
                z.write(f, f.relative_to(folder))
    return path


def latest_stages(manifest):
    """The entry that counts for every stage of a run: the last one that was not superseded by a later attempt."""
    latest = {}
    for entry in manifest.get('stages', []):
        if not entry.get('superseded'):
            latest[entry['stage']] = entry
    return latest


def open_run(args, runtime, stages):
    """The folder and the manifest of the run: a new run, or the run in ``--run-folder`` continued."""
    now = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    if args.run_folder is None:
        folder = args.output.expanduser().resolve() / f'isaac_{args.team}_{now}'
        folder.mkdir(parents=True, exist_ok=False)
    else:
        folder = args.run_folder.expanduser().resolve()
        if (folder / 'manifest.json').is_file():
            manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('team') != args.team:
                raise ValueError(f"{folder} holds a run of {manifest.get('team')!r}, not of {args.team!r}")
            manifest.setdefault('invocations', []).append(dict(started_utc=utc(), requested_stages=stages, runtime=str(runtime)))
            manifest['requested_stages'] = list(dict.fromkeys(list(manifest.get('requested_stages', [])) + stages))
            manifest.update(status='running', runtime=str(runtime), headless=args.headless)
            manifest.pop('error', None)
            (folder / 'logs').mkdir(exist_ok=True)
            return folder, manifest
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f'{folder} is not empty and holds no run (no manifest.json)')
        folder.mkdir(parents=True, exist_ok=True)
    (folder / 'logs').mkdir()
    manifest = dict(schema_version=1, run_id=now, team=args.team, started_utc=utc(), folder=str(folder), package=str(ROOT),
                    runtime=str(runtime), coordinator_python=sys.executable, platform=platform.platform(),
                    requested_stages=stages, headless=args.headless, stages=[], status='running',
                    invocations=[dict(started_utc=utc(), requested_stages=stages, runtime=str(runtime))])
    return folder, manifest


def move(source, target):
    for attempt in range(6):                                           # a scanner or a sync client may hold a file briefly
        try:
            source.rename(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.2 * (attempt + 1))


def set_aside(folder, manifest, stage):
    """Before a stage is run again: its earlier attempt moves to ``attempts/`` and its entries no longer count."""
    earlier = [e for e in manifest['stages'] if e['stage'] == stage and not e.get('superseded')]
    kept = None
    if (folder / stage).exists() or (folder / 'logs' / f'{stage}.log').exists():
        (folder / 'attempts').mkdir(exist_ok=True)
        number = 1 + sum(1 for x in (folder / 'attempts').iterdir() if x.name.startswith(stage + '_') and x.is_dir())
        kept = f'attempts/{stage}_{number}'
        (folder / kept).mkdir()
        if (folder / stage).exists():
            move(folder / stage, folder / kept / stage)
        if (folder / 'logs' / f'{stage}.log').exists():
            move(folder / 'logs' / f'{stage}.log', folder / kept / f'{stage}.log')
    for entry in earlier:
        entry['superseded'] = True
        if kept:
            entry['kept_in'] = kept


def coordinator(args):
    if args.isaac_root:
        runtime = args.isaac_root.expanduser().resolve() / ('python.bat' if os.name == 'nt' else 'python.sh')
    elif args.isaac_python:
        runtime = args.isaac_python.expanduser().absolute()
    else:
        runtime = Path(sys.executable)
    if not runtime.is_file():
        print(f'The Isaac Python does not exist: {runtime}', file=sys.stderr)
        return 2
    separate = bool(args.isaac_root or args.isaac_python)
    env = isaac_environment(os.environ, runtime) if separate else dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    stages = list(args.stages) if args.stages else (list(STAGES) if args.stage == 'all' else [args.stage])
    try:
        folder, manifest = open_run(args, runtime, stages)
    except (OSError, ValueError) as exc:
        print(f'The run folder cannot be used: {exc}', file=sys.stderr)
        return 2
    run_id = manifest['run_id']
    save_json(folder / 'manifest.json', manifest)
    print(f'Run {run_id}\nResults: {folder}\nIsaac Python: {runtime}', flush=True)
    failed = interrupted = False
    try:
        for number, stage in enumerate(stages):
            set_aside(folder, manifest, stage)
            entry = dict(stage=stage, started_utc=utc(), status='running', log=f'logs/{stage}.log')
            manifest['stages'].append(entry)
            save_json(folder / 'manifest.json', manifest)
            cli = [str(HERE / 'run_isaac.py'), '--worker', '--stage', stage, '--team', args.team, '--output', str(folder),
                   '--run-id', run_id] + physics_cli(args)
            print(f'\n[{number + 1}/{len(stages)}] {stage}', flush=True)
            code, stalled = run_logged(runtime_command(runtime, cli), folder / 'logs' / f'{stage}.log', env,
                                       stall_s=60. * args.stall_timeout)
            record, problems = {}, []
            try:
                record = json.loads((folder / stage / 'case.json').read_text(encoding='utf-8'))
            except (OSError, ValueError) as exc:
                problems.append(f'no usable case.json: {exc}')
            if record:
                if record.get('run_id') != run_id or record.get('stage') != stage:
                    problems.append('case.json does not belong to this stage of this run')
                if record.get('failure'):
                    problems.append(str(record['failure']))
                if record.get('passed') is not True:
                    problems.append('the stage did not pass: ' + ', '.join(map(str, record.get('failed_checks') or record.get('unmet_bounds') or ['see case.json'])))
            log = scan_log(folder / 'logs' / f'{stage}.log')
            problems += log['physics']
            # The record is written before the application closes.  Once it is on disk and says passed, what the
            # application does while it closes cannot change the result of the stage: an error line, an exit status
            # or a hang of that moment is reported (closing_problems) and does not make the stage fail.  Before that
            # moment nothing is forgiven: an application can end with status 0 after a Python exception, and a
            # record can say passed in a process that did not get to its end.
            complete = bool(record) and record.get('passed') is True and record.get('execution_exit_code') == 0 and log['marked']
            closing = list(log['closing'])
            trouble = []
            if stalled == 'silent':
                trouble.append(f'the stage printed nothing for {args.stall_timeout:g} minutes and was ended (--stall-timeout)')
            elif stalled == 'closing':
                patience = min(CLOSING_PATIENCE_S / 60., args.stall_timeout)
                trouble.append(f'the application had not closed {patience:g} minutes after the record was written and was ended')
            if code != 0:
                trouble.append(f'the stage process ended with exit status {exit_text(code)}')
            if complete:
                closing += trouble
            else:
                problems = trouble + problems + closing
                closing = []
            ok = not problems and (code == 0 or complete)
            entry.update(process_exit_code=code, problems=problems, closing_problems=closing,
                         other_error_lines_in_log=log['other'], finished_utc=utc(), status='passed' if ok else 'failed')
            for line in problems:
                print(f'{stage}: {line}', file=sys.stderr, flush=True)
            for line in closing:
                print(f'{stage}: while the application closed, after the record was written: {line}', file=sys.stderr, flush=True)
            save_json(folder / 'manifest.json', manifest)
            if not ok:
                failed = True
                if not args.keep_going:
                    for blocked in stages[number + 1:]:
                        manifest['stages'].append(dict(stage=blocked, status='not run', reason=f'{stage} failed'))
                    break
    except KeyboardInterrupt:
        interrupted = failed = True
    except BaseException as exc:                                      # nothing that ends the coordinator early is a pass
        failed = True
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        for entry in manifest['stages']:
            if entry.get('status') == 'running':
                entry.update(status='failed', problems=[manifest['error']])
        print(manifest['error'], file=sys.stderr, flush=True)
    finally:
        counted = latest_stages(manifest)
        if any(e.get('status') != 'passed' for e in counted.values()) or any(stage not in counted for stage in stages):
            failed = True
        for entry in manifest['stages']:
            if entry.get('status') == 'running':                      # interrupted before the stage reported
                entry.update(status='failed', problems=['interrupted'], finished_utc=utc())
        manifest['finished_utc'] = utc()
        manifest['status'] = 'interrupted' if interrupted else ('failed' if failed else 'passed')
        save_json(folder / 'manifest.json', manifest)
        if not args.no_bundle:
            try:
                print(f'Bundle: {bundle(folder)}', flush=True)
            except Exception as exc:                                  # noqa: BLE001
                print(f'bundle not written: {exc}', file=sys.stderr)
    print(f"\n{'PASSED' if not failed else 'NOT PASSED'}: " + ', '.join(f"{s['stage']} {s['status']}" for s in counted.values()), flush=True)
    print(f'Results: {folder}', flush=True)
    return 130 if interrupted else (1 if failed else 0)


def main(argv=None):
    args = parse_args(argv)
    if args.make_reference:
        sys.path.insert(0, str(ROOT))
        return make_reference(args)
    return worker(args) if args.worker else coordinator(args)


if __name__ == '__main__':
    sys.exit(main())
