"""Fig. 6: rates certified at the nominal carrying configurations by the payload-space test (equal to the centralized
rate) and by equal and optimized scalar shares, (a) for gains designed robot by robot for a target of 10 1/s with equal
payload shares and (b) for the mission gains.

Data: results/analysis/exp2_share_gains.json (a copy of results/checks/exp2_modular.json, written by
code/checks/exp2_modular.py) and results/analysis/table2_check.json, written by code/analysis/check_table2.py.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import ANALYSIS, FIGS

OUT = FIGS / 'fig_composition.pdf'
TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3')
NAMES = ('RB-ROBOUT+\n2 robots', 'RB-ROBOUT+\n3 robots', 'Ridgeback\n2 robots', 'Ridgeback\n3 robots')
COLORS = ('#2a78d6', '#eb6834', '#1baf7a')        # categorical colours (as in Fig. 7)
LABELS = ('payload-space test = centralized', 'equal scalar shares', 'optimized scalar shares')

share = json.loads((ANALYSIS / 'exp2_share_gains.json').read_text())
mission = json.loads((ANALYSIS / 'table2_check.json').read_text())
data = {
    'a': [[share[t]['rate_team'] for t in TEAMS], [share[t]['bound_equal_shares'] for t in TEAMS],
          [share[t]['bound_optimal_scalar_shares'] for t in TEAMS]],
    'b': [[mission[t]['rate'] for t in TEAMS], [mission[t]['equal_shares'] for t in TEAMS],
          [mission[t]['optimized_scalar'] for t in TEAMS]],
}
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': .5,
                     'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.15), sharex=True)
titles = {'a': '(a) gains designed robot by robot for 10 s$^{-1}$ with equal shares',
          'b': '(b) mission gains (Table 1)'}
x = np.arange(len(TEAMS))
w = .26
for ax, key in zip(axes, ('a', 'b')):
    for j in range(3):
        vals = np.asarray(data[key][j])
        ax.bar(x + (j - 1) * w, vals, w * .92, color=COLORS[j], label=LABELS[j], zorder=3)
        for xi, v in zip(x + (j - 1) * w, vals):
            ax.text(xi, v * 1.15, f'{v:.3g}', ha='center', va='bottom', fontsize=5.4, color='#0b0b0b', rotation=90)
    ax.set_yscale('log')
    ax.set_ylim(.1, 600 if key == 'b' else 80)
    ax.set_yticks([.1, 1, 10, 100] if key == 'b' else [.1, 1, 10])
    ax.set_yticklabels(['0.1', '1', '10', '100'] if key == 'b' else ['0.1', '1', '10'])
    ax.minorticks_off()
    ax.set_xticks(x)
    ax.set_ylabel(r'rate (s$^{-1}$)')
    ax.set_title(titles[key], fontsize=7.0, pad=2, loc='left')
    ax.grid(True, axis='y', which='major', lw=.3, color='#d9d8d4', zorder=0)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
axes[1].set_xticklabels(NAMES, fontsize=6.4)
axes[0].tick_params(axis='x', length=0)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=6.0, handlelength=1.0, columnspacing=.8,
           handletextpad=.4, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, .935), h_pad=.3)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix('.png'), dpi=220)
for key in ('a', 'b'):
    ex, eq, op = (np.asarray(v) for v in data[key])
    print(key, 'exact/optimized ratio', np.round(ex / op, 2), 'exact/equal', np.round(ex / eq, 2))
