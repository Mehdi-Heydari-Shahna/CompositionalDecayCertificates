"""Fig. 10: MuJoCo missions of the four full-rank teams: (a) peak and (b) rms carried-payload position error, (c) largest
grasp moment and (d) largest actuator effort relative to their declared bounds, for the controllers of Section 7.1:
D (desired-motion feedforward, k_d, no rate filter), F (fitted loops, k_d, no rate filter), F/2 (fitted loops, k_d/2 on
the arm and lift joints) and D_R (desired-motion feedforward with retuned arm damping and the 15 Hz rate filter).
Grasp moments are maxima of 20 ms interval means of the wrist measurements after tool-inertia removal.

Data: results/checks/missions.json (runs mujoco_<team>_desired0, _fitted, _fitted05, _desiredR) and, for the declared
effort-ratio bound, the reports of the same runs in code/Team_CMG_PACDM/reference_results/fitted_feedforward/.
"""
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from paths import CHECKS, FIGS, REPO

REPORTS = REPO / 'code' / 'Team_CMG_PACDM' / 'reference_results' / 'fitted_feedforward'
TEAMS = (('team2', 'RB-ROBOUT+\n2 robots'), ('team3', 'RB-ROBOUT+\n3 robots'), ('ridgeback2', 'Ridgeback\n2 robots'),
         ('ridgeback3', 'Ridgeback\n3 robots'))
KINDS = (('desired0', 'D: desired-motion implementation', dict(facecolor='#656d78', edgecolor='#656d78')),
         ('fitted', 'F: fitted-reference implementation', dict(facecolor='#0072b2', edgecolor='#0072b2')),
         ('fitted05', 'F/2: fitted, half arm/lift rate gains', dict(facecolor='#d55e00', edgecolor='#d55e00')),
         ('desiredR', r'$D_R$: retuned desired-motion implementation', dict(facecolor='#009e73', edgecolor='white', hatch='///')))
BOUND = '#a3123a'

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.linewidth': .6,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'hatch.linewidth': .5, 'pdf.fonttype': 42})
mis = json.loads((CHECKS / 'missions.json').read_text())
values = {name: np.zeros((len(TEAMS), len(KINDS))) for name in ('peak', 'rms', 'moment', 'effort')}
for i, (team, _) in enumerate(TEAMS):
    for j, (kind, _, _) in enumerate(KINDS):
        r = mis[f'mujoco_{team}_{kind}']
        report = json.loads((REPORTS / f'mujoco_{team}_{kind}' / 'report.json').read_text())
        effort = report['bounds']['checks']['effort_ratio']
        assert abs(effort['value'] - r['effort']) < 1e-12
        assert (r['velocity_filter_hz'] > 0) == (kind == 'desiredR')       # D, F, F/2 without the 15 Hz filter
        values['peak'][i, j] = r['payload_mm']
        values['rms'][i, j] = r['payload_rms_mm']
        values['moment'][i, j] = r['grasp_moment_max'] / r['grasp_moment_bound']
        values['effort'][i, j] = effort['value'] / effort['bound']

fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
PANELS = (('peak', '(a) Peak payload-position error', 'error (mm)', False),
          ('rms', '(b) RMS payload-position error', 'error (mm)', False),
          ('moment', '(c) Grasp-moment utilization', 'reported maximum / bound', True),
          ('effort', '(d) Actuator-effort utilization', 'reported maximum / bound', True))
x = np.arange(len(TEAMS))
w = .19
for ax, (name, title, ylabel, bounded) in zip(axes.ravel(), PANELS):
    v = values[name]
    top = max(v.max(), 1. if bounded else 0.)
    for j, (kind, label, style) in enumerate(KINDS):
        xs = x + (j - 1.5) * w
        ax.bar(xs, v[:, j], w * .92, label=label, lw=0, zorder=2, **style)
        for xi, vi in zip(xs, v[:, j]):
            above = bounded and vi > 1.
            ax.text(xi, vi + (.07 if above else .015) * top, f'{vi:.2f}', ha='center', va='bottom', fontsize=6,
                    rotation=90 if bounded else 0)
            if above:
                ax.plot([xi], [vi], marker='x', ms=5, mew=1.2, color=BOUND, zorder=4)
    if bounded:
        ax.axhline(1., color=BOUND, lw=.9, ls='--', zorder=3)
        ax.text(.995, 1.01, 'bound', color=BOUND, fontsize=6.5, ha='right', va='bottom', transform=ax.get_yaxis_transform())
    ax.set_ylim(0., (1.25 if bounded else 1.2) * top)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in TEAMS], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left', fontsize=9)
    ax.grid(axis='y', alpha=.35, lw=.4, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
handles = [Patch(label=label, **style) for _, label, style in KINDS]
fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.0))
fig.text(.5, .893, 'D and F: identical gains, 15 Hz filter disabled. F/2 and $D_R$: separate tuning comparisons.',
         ha='center', fontsize=7)
fig.tight_layout(rect=(0, 0, 1, .875), h_pad=1.6, w_pad=1.5)
fig.savefig(FIGS / 'fig_matched_missions.pdf')
fig.savefig(FIGS / 'fig_matched_missions.png', dpi=220)
print('written', FIGS / 'fig_matched_missions.pdf')
