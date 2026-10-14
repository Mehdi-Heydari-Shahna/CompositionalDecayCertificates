#!/usr/bin/env python
"""MuJoCo workflow: run the beam mission of a team and audit it.

    python workflows/mujoco/run_mission.py --team team2 --out results/mujoco_team2

The plant is generated from the compiled team graph (``backends/mujoco_export``):
free bodies, wheels in contact with the terrain (mecanum wheels with their
rollers, conventional wheels with a tyre), latching clamps, wrist sensors.  The controller and the audit are the simulator-neutral ones in
``mission/``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# small matrices: one BLAS thread is faster, and several runs can share a machine (set before NumPy is imported)
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

from backends.gl import prepare_mujoco_gl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--team', default='team2')
    ap.add_argument('--out', default=None)
    ap.add_argument('--timestep', type=float, default=.002)
    ap.add_argument('--roller-hull', type=int, nargs=2, default=(41, 64), metavar=('RINGS', 'AROUND'),
                    help='vertex rings and vertices per ring of the roller collision hull')
    ap.add_argument('--slow-every', type=int, default=None, help='physics steps per reference/feedforward update (default: 10 ms)')
    ap.add_argument('--latch', choices=('connect', 'compliant'), default='connect',
                    help='clamp model: constraints of the MuJoCo solver, or the compliant clamp of backends/latch.py '
                         '(the one the Isaac Sim plant uses)')
    ap.add_argument('--sensor', choices=('native', 'model'), default='native',
                    help='wrist sensor: MuJoCo force/torque sensors, or the Newton-Euler balance of the tool-side body '
                         '(compliant clamp only; what the Isaac Sim plant uses)')
    ap.add_argument('--tyre', choices=('model', 'native'), default='model',
                    help='conventional wheels (skid steering): frictionless contacts with the tyre model of backends/tyre.py '
                         'applied as loads (what the Isaac Sim plant uses), or the friction of the MuJoCo contacts')
    ap.add_argument('--stop-after', default=None, help='stop when this phase is reached (debugging)')
    ap.add_argument('--frames', type=float, default=0., help='render frames at this rate (Hz) into <out>/frames')
    ap.add_argument('--video', action='store_true', help='also write <out>/mission.mp4 from the frames')
    ap.add_argument('--video-speed', type=float, default=4., help='playback speed of the video relative to real time')
    ap.add_argument('--camera', type=float, nargs=3, default=None, metavar=('DISTANCE', 'AZIMUTH', 'ELEVATION'),
                    help='viewpoint of the frames (default: chosen from the height of the team, mission.report.video_camera)')
    ap.add_argument('--feedforward', choices=('desired', 'fitted'), default=None,
                    help='desired: inverse dynamics of the desired motion (the default of the mission controller); '
                         'fitted: the effort required by the fitted reference of the loops (rolling wheels; DESIGN.md)')
    ap.add_argument('--feedforward-rate', choices=('slow', 'fast'), default=None,
                    help='fitted effort: evaluated with the references (default) or in every step')
    ap.add_argument('--monitor', action='store_true',
                    help='record the quantities of the stability statement (rolling wheels; always with --feedforward fitted)')
    ap.add_argument('--option', action='append', default=[], metavar='KEY=VALUE',
                    help='another option of the mission controller (mission/controller.py, DEFAULTS), the value in JSON; '
                         'for example --option arm_damping_ratio=0.7.  May be given several times')
    ap.add_argument('--velocity-filter', type=float, default=None, metavar='HZ',
                    help='low-pass of the joint rates in the damping term (default: 15 Hz with --feedforward desired, none '
                         'with fitted, where the stability statement is for the measured rates)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    controller = {key: value for key, value in (('feedforward', args.feedforward), ('feedforward_rate', args.feedforward_rate))
                  if value is not None}
    if args.monitor or args.feedforward == 'fitted':
        controller['monitor'] = True
    if args.velocity_filter is not None:
        controller['velocity_filter_hz'] = args.velocity_filter
    for item in args.option:
        key, _, value = item.partition('=')
        if not key or not value:
            ap.error(f'--option needs KEY=VALUE, got {item!r}')
        import json
        controller[key] = json.loads(value)
    if args.video and args.frames <= 0:
        ap.error('--video needs --frames (for example --frames 6.25)')
    gl = prepare_mujoco_gl(render=args.frames > 0)                  # before mujoco is imported: it reads MUJOCO_GL once
    from backends.mujoco_plant import MujocoPlant
    from mission.workflow import run_and_report
    from teamc.io import INPUTS, load_team, read_json
    out = Path(args.out or ROOT / 'results' / f'mujoco_{args.team}')
    out.mkdir(parents=True, exist_ok=True)

    comp = load_team(args.team)
    mission = read_json(INPUTS / 'missions' / f'{args.team}.json')
    plant = MujocoPlant(comp, timestep=args.timestep, roller_hull=tuple(args.roller_hull), latch=args.latch, sensor=args.sensor,
                        tyre=args.tyre)
    slow_every = args.slow_every or max(int(round(.01 / args.timestep)), 1)
    (out / 'plant.xml').write_text(plant.xml, encoding='utf-8')

    on_record, writer, shots = None, None, []
    if args.frames > 0:
        import imageio.v2 as imageio
        print(f'rendering with the MuJoCo OpenGL backend: {gl}', flush=True)
        every = max(int(round(50. / args.frames)), 1)
        counter = dict(n=0, phase=None)
        (out / 'frames').mkdir(exist_ok=True)
        pb = comp.payload['base']
        from mission.report import video_camera
        look_z, *camera = video_camera(comp, mission)
        camera = args.camera or camera
        if args.video:
            writer = imageio.get_writer(out / 'mission.mp4', fps=args.frames * args.video_speed, macro_block_size=8, quality=6)

        def on_record(t, plant_, ctrl):
            counter['n'] += 1
            new_phase = ctrl.phase != counter['phase']
            if counter['n'] % every and not new_phase:
                return
            c = plant_.state()[0][pb]
            img = plant_.render([c[0], c[1], look_z], camera[0], camera[1], camera[2], width=960, height=540)
            if writer is not None and counter['n'] % every == 0:
                writer.append_data(img)
            if new_phase:                                   # one still per phase
                counter['phase'] = ctrl.phase
                shots.append(ctrl.phase)
                imageio.imwrite(out / 'frames' / f'{len(shots):02d}_{ctrl.phase}.png', img)

    report, A, rec, ctrl, info = run_and_report(
        plant, comp, mission, out, f'MuJoCo {args.team}:', options=dict(slow_every=slow_every, stop_after=args.stop_after, controller=controller),
        quiet=args.quiet, on_record=on_record,
        plant_info=dict(engine=f'MuJoCo {plant.mujoco_version}', roller_hull=list(args.roller_hull), latch=args.latch, sensor=args.sensor,
                        clamp=None if plant.latches is None else dict(plant.latches.p), tyre=plant.tyre_mode,
                        **({'controller': controller} if controller else {})))
    if plant.latches is not None:
        print(f'compliant clamps: stability ratio {plant.latch_ratio:.2f}')
    if plant.tyres is not None:
        print(f'tyre model: stability ratio {plant.tyre_ratio:.2f}')
    if writer is not None:
        writer.close()
    return 0 if report['bounds']['all_ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
