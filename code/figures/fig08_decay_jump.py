"""Fig. 8: normalized error storage of the fitted reference after an adverse velocity jump with the mission controllers
D, F and F/2 (2 ms steps, 10 ms renewal) on (a) the ideal-constraint plant and (b) MuJoCo.

D is the desired-motion feedforward without the 15 Hz rate filter, F the fitted loops with k_d, F/2 the fitted loops
with k_d/2 on the arm and lift joints.  Shaded: the band certified for F, between exp(-2 int kappa_max) and
exp(-2 int kappa_min) evaluated along the run, drawn until W/W(0) of F reaches the cutoff (F/2 has other gains; D is
not certified); dotted: the two bounds over the whole window.  Cutoffs: 1e-3 (a), 0.05 (b).

Data: results/checks/decay.npz (time histories along the adverse direction, with the two bounds normalized to 1 at the
jump) and decay.json, written by code/checks/decay.py from the decay runs in
code/Team_CMG_PACDM/reference_results/fitted_feedforward/decay/.
"""
import json

import matplotlib
matplotlib.use('Agg')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

from paths import CHECKS, FIGS

TEAMS = (('team2', 'RB-ROBOUT+ / 2'), ('team3', 'RB-ROBOUT+ / 3'), ('ridgeback2', 'Ridgeback / 2'), ('ridgeback3', 'Ridgeback / 3'))
ROWS = (('pinocchio', '(a) ideal-constraint plant', (1e-8, 3.), (1e-8, 1e-6, 1e-4, 1e-2, 1.)),
        ('mujoco', '(b) MuJoCo', (1e-3, 3.), (1e-3, 1e-2, 1e-1, 1.)))
RUNS = (('desired_nofilter', 'D (unfiltered)', dict(color='#656d78', ls=(0, (3.7, 1.6)), lw=1.1)),
        ('fitted_slow', 'F (full gains)', dict(color='#0072b2', ls='-', lw=1.1)),
        ('fitted_half', 'F/2 (half arm/lift rate gains)', dict(color='#d55e00', ls=(0, (6.4, 1.6, 1., 1.6)), lw=1.1)))
BAND, BOUND, CUT, CUTFILL = '#c2c8d0', '#8f97a3', '#b8a984', '#f5f1e6'
T_END = .3

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.linewidth': .6,
                     'xtick.major.width': .6, 'ytick.major.width': .6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'pdf.fonttype': 42})
runs = json.loads((CHECKS / 'decay.json').read_text())
z = np.load(CHECKS / 'decay.npz')

fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0), sharex=True)
for i, (plant, row_title, ylim, yticks) in enumerate(ROWS):
    for j, (team, title) in enumerate(TEAMS):
        ax = axes[i, j]
        key = f'{plant}_{team}_fitted_slow'
        t, W, upper, lower = z[key + ':t'], z[key + ':W'], z[key + ':upper'], z[key + ':lower']
        rec = runs[f'{plant}_{team}_worst0_fitted_slow']
        assert abs(W[0] - rec['W0']) <= 1e-12 * rec['W0'] and abs(upper[0] - 1.) < 1e-12 and abs(lower[0] - 1.) < 1e-12
        floor, t_cut = rec['floor'], rec['time_to_floor']
        ax.axhspan(ylim[0], floor, color=CUTFILL, lw=0, zorder=0)
        ax.axhline(floor, color=CUT, lw=.6, ls=':', zorder=1)
        keep = t <= (t_cut if t_cut is not None else T_END) + 1e-9
        ax.fill_between(t[keep], np.maximum(lower[keep], ylim[0]), np.minimum(upper[keep], ylim[1]), color=BAND, lw=0, zorder=1)
        ax.semilogy(t, np.maximum(upper, 1e-300), color=BOUND, lw=.6, ls=':', zorder=2)
        ax.semilogy(t, np.maximum(lower, 1e-300), color=BOUND, lw=.6, ls=':', zorder=2)
        for variant, label, style in RUNS:
            k = f'{plant}_{team}_{variant}'
            tv, Wv = z[k + ':t'], z[k + ':W']
            r = runs[f'{plant}_{team}_worst0_{variant}']
            assert abs(Wv.max() / Wv[0] - r['W_max_ratio']) < 1e-9 and r['timestep'] == .002
            ax.semilogy(tv, Wv / Wv[0], zorder=3, **style)
        dmax = runs[f'{plant}_{team}_worst0_desired_nofilter']['W_max_ratio']
        ax.text(.97, .95, f'D max = {dmax:.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
        if j == 0:
            ax.text(.02, .03, ('0.1%' if floor == 1e-3 else f'{100 * floor:g}%') + ' cutoff', transform=ax.transAxes,
                    color='#8a7a52', fontsize=7)
        ax.set_xlim(0., T_END)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_title(title, fontsize=8.5, loc='left')
        ax.grid(alpha=.35, lw=.4)
        ax.spines[['top', 'right']].set_visible(False)
        if j == 0:
            ax.set_ylabel(r'$W(t)/W(0)$')
        else:
            ax.tick_params(labelleft=False)
        if i == 1:
            ax.set_xlabel('time after jump (s)')
    axes[i, 0].annotate(row_title, xy=(0, 1), xycoords='axes fraction', xytext=(0, 15), textcoords='offset points',
                        fontsize=9, fontweight='bold', ha='left', va='bottom')
handles = [Line2D([], [], label=label, **style) for _, label, style in RUNS] + \
          [Patch(color=BAND, lw=0, label='band certified for F')]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, .93), h_pad=2.2, w_pad=.6)
fig.savefig(FIGS / 'fig_matched_decay.pdf')
fig.savefig(FIGS / 'fig_matched_decay.png', dpi=220)
print('written', FIGS / 'fig_matched_decay.pdf')
