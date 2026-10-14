"""Fig. 7: normalized error storage after an admissible velocity jump along the slowest certified direction, on the
ideal-constraint plant (0.5 ms steps, renewal in every step), for the mission gains and the minimum-gain (P1) and
maximum-rate (P2) designs of Table 2.  Dashed: the certified bounds exp(-2 int kappa) of Proposition 4(b).

Data: results/analysis/decay_gains_<team>.npz, written by code/analysis/run_decay_gains.py <team>.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import ANALYSIS, FIGS

OUT = FIGS / 'fig_synth_decay.pdf'
TITLES = {'team2': 'RB-ROBOUT+ / 2', 'team3': 'RB-ROBOUT+ / 3', 'ridgeback2': 'Ridgeback / 2', 'ridgeback3': 'Ridgeback / 3'}
COLORS = {'mission': '#2a78d6', 'P1': '#eb6834', 'P2': '#1baf7a'}        # categorical colours
LABELS = {'mission': 'mission gains', 'P1': 'min. gain (P1)', 'P2': 'max. rate (P2)'}

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': .5,
                     'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.05), sharex=True, sharey=True)
for ax, team in zip(axes.ravel(), TITLES):
    A = np.load(ANALYSIS / f'decay_gains_{team}.npz')
    for key in ('mission', 'P1', 'P2'):
        t, r, up = A[f'{key}:t'], A[f'{key}:ratio'], A[f'{key}:upper']
        ax.semilogy(t, up, ls=(0, (3, 2)), lw=.8, color=COLORS[key], alpha=.9)
        style = dict(lw=2.6, alpha=.45) if key == 'mission' else dict(lw=1.0)
        ax.semilogy(t, r, color=COLORS[key], label=LABELS[key], solid_capstyle='butt', **style)
    ax.set_title(TITLES[team], fontsize=7.5, pad=2)
    ax.set_ylim(1e-3, 1.5)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_xlim(0, .25)
    ax.grid(True, which='major', lw=.3, color='#d9d8d4')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel('time after jump (s)')
for ax in axes[:, 0]:
    ax.set_ylabel(r'$W(t)/W(0)$')
h, l = axes[0, 0].get_legend_handles_labels()
from matplotlib.lines import Line2D
h.append(Line2D([], [], color='#52514e', lw=.8, ls=(0, (3, 2))))
l.append(r'certified bound $e^{-2\int\underline{\kappa}}$')
fig.legend(h, l, loc='upper center', ncol=4, frameon=False, fontsize=6.1, handlelength=1.5, columnspacing=.7,
           handletextpad=.4, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, .92), h_pad=.4, w_pad=.6)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix('.png'), dpi=220)
summary = {t: json.loads((ANALYSIS / f'decay_gains_{t}.json').read_text()) for t in TITLES}
print(json.dumps({t: {k: (round(v['rate_at_jump'][0], 3), v['excess_over_upper'], v['time_to_1e3'], round(v['rate_fitted'], 2)) for k, v in s.items()} for t, s in summary.items()}, indent=0))
