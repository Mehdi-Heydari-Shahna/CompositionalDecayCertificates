#!/usr/bin/env python
"""Render a recorded run from its saved states (any plant) with the MuJoCo renderer.

    python tools/replay_video.py --record results/isaac_team2_.../mission/record.npz --out mission_replay.mp4

This is a replay: the recorded compiled coordinates are written into the MuJoCo scene of the same
team and drawn; nothing is simulated.  It gives a video of a run made where no camera was used
(the Isaac Sim workflow renders nothing during physics) and of the Pinocchio plant, which has no
renderer.  Wheel rollers are drawn at their zero angles.  Needs ``mujoco`` and ``imageio`` with
``imageio-ffmpeg``.  On a Linux machine without a display the OpenGL backend is OSMesa if that library
is installed, else EGL; set ``MUJOCO_GL`` to choose it yourself.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--record', required=True, type=Path)
    ap.add_argument('--out', type=Path, default=None, help='video file (default: replay.mp4 beside the record)')
    ap.add_argument('--team', default=None)
    ap.add_argument('--fps', type=float, default=25.)
    ap.add_argument('--speed', type=float, default=4., help='playback speed relative to real time')
    ap.add_argument('--size', type=int, nargs=2, default=(960, 540), metavar=('WIDTH', 'HEIGHT'))
    ap.add_argument('--stills', type=Path, default=None, help='also write one PNG per mission phase into this folder')
    ap.add_argument('--camera', type=float, nargs=3, default=None, metavar=('DISTANCE', 'AZIMUTH', 'ELEVATION'),
                    help='viewpoint (default: chosen from the height of the team, mission.report.video_camera)')
    args = ap.parse_args(argv)
    from backends.gl import prepare_mujoco_gl
    print(f'rendering with the MuJoCo OpenGL backend: {prepare_mujoco_gl(render=True)}', flush=True)   # before mujoco is imported
    import imageio.v2 as imageio
    import numpy as np

    from backends.mujoco_plant import MujocoPlant
    from mission.report import load_record, video_camera
    from teamc.io import INPUTS, load_team, read_json

    A, info, _ = load_record(args.record)
    team = args.team or (info or {}).get('team')
    if not team:
        ap.error('the record does not name its team; pass --team')
    comp = load_team(team)
    look_z, *camera = video_camera(comp, read_json(INPUTS / 'missions' / f'{team}.json'))
    camera = args.camera or camera
    plant = MujocoPlant(comp, roller_hull=(7, 16))                        # drawing only
    out = args.out or args.record.parent / 'replay.mp4'
    t = A['t']
    frames = np.searchsorted(t, np.arange(t[0], t[-1], args.speed / args.fps))
    pb = np.asarray(comp.payload['base'])
    phases = (info or {}).get('phases')
    width = (args.size[0] // 8) * 8
    height = (args.size[1] // 8) * 8
    writer = imageio.get_writer(out, fps=args.fps, macro_block_size=8, quality=6)
    seen = set()
    if args.stills:
        args.stills.mkdir(parents=True, exist_ok=True)
    try:
        for i in frames:
            plant.reset(A['q'][i])
            if 'obstacle' in A:
                plant.set_obstacle(A['obstacle'][i])
                import mujoco
                mujoco.mj_forward(plant.model, plant.data)
            c = A['q'][i, pb]
            img = plant.render([c[0], c[1], look_z], camera[0], camera[1], camera[2], width=width, height=height)
            writer.append_data(img)
            k = int(A['phase'][i])
            if args.stills and k not in seen:
                seen.add(k)
                name = phases[k].replace(':', '_') if phases and k < len(phases) else f'phase{k}'
                imageio.imwrite(args.stills / f'{len(seen):02d}_{name}.png', img)
    finally:
        writer.close()
    print(f'{len(frames)} frames at {args.speed:g}x written to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
