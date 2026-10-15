"""Figures and files of a mission run (matplotlib only; no simulator)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from teamc.io import write_json


def _shade(ax, phases, names=None):
    for i, p in enumerate(phases):
        if names is None or p['name'].split(':')[0] in names:
            ax.axvspan(p['start'], p['end'], color='#888', alpha=.07 if i % 2 else .0, lw=0)


def site_plan(ax, comp, alpha=.9):
    from matplotlib.patches import Polygon
    colors = dict(terrain='#ecebe4', wall='#5a5a5a', door='#5a5a5a', pad='#c89000', mount='#c89000', guide='#c89000',
                  rack='#3355aa', neighbour='#8a4a30')
    for s in comp.scene['statics']:
        T, h = np.asarray(s['T_world_geom']), s['half_size']
        c = np.array([[-h[0], -h[1]], [h[0], -h[1]], [h[0], h[1]], [-h[0], h[1]]])
        color = '#bdbdbd' if s['id'] == 'door_lintel' else colors.get(s.get('role'), '#999')
        ax.add_patch(Polygon((T[:2, :2] @ c.T).T + T[:2, 3], closed=True, fc=color, ec='none', alpha=alpha,
                             zorder=1 if s.get('role') == 'terrain' else 3))
    ax.set_aspect('equal')


def video_camera(comp, mission):
    """Viewpoint of the videos, which follow the payload: ``(height looked at, distance, azimuth, elevation)``.

    A team that stands lower than 0.6 of the wall height of the site is looked at from higher up and
    from nearer; from the flatter viewpoint the walls of an aisle hide it.
    """
    walls = float(comp.scene.get('site', {}).get('wall_height', 0.))
    if float(mission['formation']['extent']['top_m']) < .6 * walls:
        return .6, 9., 130., -48.
    return 1., 10.5, 130., -28.


def figures(out, comp, mission, A, report, series, contacts=None, title=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = A['t']
    phases = report['phases']
    pb = np.asarray(comp.payload['base'], int)
    grasps = [c for c in comp.connections if comp.connections[c]['kind'] == 'grasp']
    weight = mission['payload']['mass_kg'] * 9.81
    files = []

    # ---------------------------------------------------------------- route
    fig, ax = plt.subplots(figsize=(8.5, 12))
    site_plan(ax, comp)
    P = np.asarray(mission['route']['samples'])
    ax.plot(P[:, 0], P[:, 1], '--', color='#c00', lw=.8, label='designed route (payload centre)', zorder=5)
    ax.plot(A['q'][:, pb[0]], A['q'][:, pb[1]], '-', color='#e06000', lw=1.4, label='payload centre', zorder=6)
    for rid, r in comp.robots.items():
        ax.plot(A['q'][:, r.base[0]], A['q'][:, r.base[1]], '-', lw=1., label=f'{rid} base', zorder=6)
    half = mission['payload']['half_length_m']
    for i in np.linspace(0, len(t) - 1, 28).astype(int):
        c = A['q'][i, pb]
        d = half * np.array([np.cos(c[3]), np.sin(c[3])])
        ax.plot([c[0] - d[0], c[0] + d[0]], [c[1] - d[1], c[1] + d[1]], '-', color='#e06000', lw=.6, alpha=.6, zorder=5)
    if 'obstacle' in A:
        ax.plot(A['obstacle'][:, 0], A['obstacle'][:, 1], ':', color='g', lw=1.2, label='cart', zorder=5)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'{title} route')
    ax.autoscale_view()
    fig.savefig(out / 'route.png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    files.append('route.png')

    # ---------------------------------------------------------------- load transfer
    tr = report['transfer']
    if tr:
        fig, axes = plt.subplots(1, len(tr), figsize=(6.2 * len(tr), 4.2), squeeze=False)
        for ax, (label, e) in zip(axes[0], tr.items()):
            tt = np.asarray(e['t'])
            ax.plot(tt, np.asarray(e['carried_fraction']) * weight, label='carried by the grasps (wrist sensors)', lw=1.6)
            ax.plot(tt, np.asarray(e['share_command']) * weight, '--', label='commanded', lw=1.)
            total = np.zeros(len(tt))
            for name, v in e.get('pads_plant_N', {}).items():
                if max(v) > 1e-9:
                    ax.plot(tt, v, lw=1., label=f'{name} (plant contact)')
                    total += np.asarray(v)
            if 'pads_plant_N' in e:
                ax.plot(tt, total + np.asarray(e['carried_fraction']) * weight, ':', color='k', lw=1., label='sum')
            ax.axhline(weight, color='#999', lw=.6)
            ax.set_title(f'{label}: load hand-over')
            ax.set_xlabel('t (s)')
            ax.set_ylabel('vertical load (N)')
            ax.grid(alpha=.3)
            ax.legend(fontsize=7)
        fig.savefig(out / 'load_transfer.png', dpi=110, bbox_inches='tight')
        plt.close(fig)
        files.append('load_transfer.png')

    # ---------------------------------------------------------------- grasp wrenches
    ts = series['t']
    fig, ax = plt.subplots(4, 1, figsize=(11, 10.5), sharex=True)
    for c in grasps:
        ax[0].plot(ts, series['share'][c] * weight, label=f'{c} vertical force (sensor)')
        m = series['grasp_model'][c]
        s = series['grasp_meas'][c]
        ax[3].plot(ts, np.linalg.norm(m[:, 3:] - s[:, 3:], axis=1), label=f'{c} force (N)')
        ax[3].plot(ts, np.linalg.norm(m[:, :3] - s[:, :3], axis=1), '--', label=f'{c} moment (N m)')
        ax[1].plot(ts, np.linalg.norm(s[:, :3], axis=1), label=f'{c} moment norm (N m)')
    ax[0].axhline(weight / max(len(grasps), 1), color='#999', lw=.6)
    ax[0].set_ylabel('N')
    ax[2].plot(ts, series['internal_force'], label='internal force norm (N)')
    ax[2].plot(ts, series['internal_moment'], label='internal moment norm (N m)')
    ax[3].set_title('compiled reaction recovery minus wrist sensor (while carrying)', fontsize=9)
    ax[3].set_xlabel('t (s)')
    for a in ax:
        _shade(a, phases)
        a.grid(alpha=.3)
        a.legend(fontsize=7, ncol=3)
    ax[0].set_title(f'{title} grasp loads')
    fig.savefig(out / 'grasp_loads.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    files.append('grasp_loads.png')

    # ---------------------------------------------------------------- ground margins, efforts, clearance
    fig, ax = plt.subplots(5, 1, figsize=(11, 13), sharex=True)
    for rid, g in series['ground'].items():
        ax[0].plot(ts, g[:, 0], label=f'{rid} smallest wheel load (N)')
        ax[1].plot(ts, g[:, 1], label=f'{rid} traction utilization' if rid not in series.get('skid', {})
                   else f'{rid} net traction utilization')
        ax[2].plot(ts, g[:, 2], label=f'{rid} tipping margin (m)')
    b = mission['bounds']
    ax[0].axhline(b['min_wheel_load_N'], color='r', lw=.6, ls='--')
    ax[1].axhline(b['traction_utilization'], color='r', lw=.6, ls='--')
    ax[2].axhline(b['tipping_margin_m'], color='r', lw=.6, ls='--')
    hi = np.array([p['bounds'][1] for p in comp.cmg['ports']])
    ratio = np.abs(A['f_peak']) / hi
    role = {}                                      # coordinate -> group of its actuator port, from the structure
    for rid, r in comp.robots.items():
        role.update({int(w['index']): f'{rid} wheels' for w in r.wheels})
        role.update({int(r.joint[j]): f'{rid} arm' for j in r.dependent})
        role.update({int(r.joint[j]): f'{rid} lift' for j in r.tool_path if j in r.commanded})
    groups = {}
    for kk, p in enumerate(comp.cmg['ports']):
        groups.setdefault(role.get(int(p['index']), f"{p['robot']} other"), []).append(kk)
    for key, cols in groups.items():
        ax[3].plot(t, ratio[:, cols].max(axis=1), lw=.8, label=key)
    ax[3].axhline(b['effort_ratio'], color='r', lw=.6, ls='--')
    ax[3].set_ylabel('effort / bound')
    keys = sorted({key for c in series['clearance'] for key in c})
    for key in keys:
        if key == 'payload:pad':
            continue
        ax[4].plot(ts, [c.get(key, 1.) for c in series['clearance']], lw=.9, label=key)
    ax[4].axhline(b['clearance_m'], color='r', lw=.6, ls='--')
    ax[4].set_ylabel('clearance (m)')
    ax[4].set_xlabel('t (s)')
    for a in ax:
        _shade(a, phases)
        a.grid(alpha=.3)
        a.legend(fontsize=7, ncol=4)
    ax[0].set_title(f'{title} margins (dashed red: declared bounds)')
    fig.savefig(out / 'margins.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    files.append('margins.png')

    # ---------------------------------------------------------------- tracking
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    e = A['q'][:, pb] - A['payload_ref']
    carrying = np.array([(g != '') and (r == '') for g, r in zip(A['mode_grasps'], A['mode_rest'])])
    en = np.where(carrying, np.linalg.norm(e[:, :3], axis=1), np.nan)
    ax[0].plot(t, 1e3 * en, label='payload position error (mm)')
    ax[1].plot(t, 1e3 * np.where(carrying, A['q'][:, pb[4]], np.nan), label='payload pitch (mrad)')
    ax[1].plot(t, 1e3 * np.where(carrying, A['q'][:, pb[5]], np.nan), label='payload roll (mrad)')
    ax[1].plot(t, 1e3 * np.where(carrying, e[:, 3], np.nan), label='payload yaw error (mrad)')
    for rid, r in comp.robots.items():
        be = A[f'base_ref:{rid}'] - A['q'][:, r.base[[0, 1, 3]]]
        ax[2].plot(t, 1e3 * np.linalg.norm(be[:, :2], axis=1), label=f'{rid} base position error (mm)')
    ax[2].set_xlabel('t (s)')
    for a in ax:
        _shade(a, phases)
        a.grid(alpha=.3)
        a.legend(fontsize=7, ncol=3)
    ax[0].set_title(f'{title} tracking')
    fig.savefig(out / 'tracking.png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    files.append('tracking.png')
    return files


def save_record(path, A, info=None, contacts=None):
    """The raw record of a run (arrays, run information and plant contact rows) in one ``.npz`` file."""
    import json
    from teamc.io import to_plain
    extra = {}
    if info is not None:
        extra['info_json'] = np.array(json.dumps(to_plain(info)))
    if contacts:
        rows = [dict(t=c['t'], wheels=[[list(k), v] for k, v in c['wheels'].items()], pads=c['pads'], guides=c['guides'],
                     tool=c['tool'], forbidden=c['forbidden']) for c in contacts]
        extra['contacts_json'] = np.array(json.dumps(to_plain(rows)))
    np.savez_compressed(path, **{k.replace(':', '__'): v for k, v in A.items()}, **extra)


def load_record(path):
    """Inverse of :func:`save_record`: ``(arrays, info, contacts)``."""
    import json
    with np.load(path, allow_pickle=False) as z:
        A = {k.replace('__', ':'): z[k] for k in z.files if k not in ('info_json', 'contacts_json')}
        info = json.loads(str(z['info_json'])) if 'info_json' in z.files else None
        contacts = None
        if 'contacts_json' in z.files:
            contacts = []
            for c in json.loads(str(z['contacts_json'])):
                # a load that was not reported is stored as null
                wheels = {tuple(k): [float('nan') if x is None else x for x in v] for k, v in c['wheels']}
                contacts.append(dict(c, wheels=wheels, forbidden=[tuple(f) for f in c['forbidden']]))
    return A, info, contacts


def save(out, comp, mission, A, report, series, info, contacts=None, title='', arrays=True, with_figures=True):
    """Write ``report.json``, ``summary.txt``, the figures and (optionally) the raw record.

    ``with_figures='auto'`` draws the figures only where matplotlib is installed.
    """
    from .audit import summary_lines
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'report.json', report)
    lines = summary_lines(report)
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if arrays:
        save_record(out / 'record.npz', A, info, contacts)
    files = []
    if with_figures == 'auto':
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            with_figures = False
    if with_figures:
        files = figures(out, comp, mission, A, report, series, contacts, title)
    return lines, files
