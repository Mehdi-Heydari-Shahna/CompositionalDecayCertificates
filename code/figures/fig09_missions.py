"""Fig. 9: error storage W of the fitted reference along the complete missions on the ideal-constraint plant (top) and
on MuJoCo (bottom), for the desired-motion feedforward with the 15 Hz rate filter and the fitted loops with k_d and
k_d/2 (largest W over every 0.5 s of the record).  Shaded: carrying phase.

Data: results/checks/missions.npz (decimated time histories) and missions.json, written by code/checks/missions.py from
the run folders of the mission workflows (their records are not included; the reports are in
code/Team_CMG_PACDM/reference_results/fitted_feedforward/).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import CHECKS, FIGS

ROLL = ('team2', 'team3', 'ridgeback2', 'ridgeback3')
PLANTS = ('pinocchio', 'mujoco')
MJ_KINDS = ('desired', 'desired0', 'desiredR', 'fitted15', 'fitted', 'fitted0515', 'fitted05')
KIND_FF = dict(desired='desired', desired0='desired', desiredR='desired', fitted15='fitted', fitted='fitted',
               fitted0515='fitted', fitted05='fitted')
KIND_FILTER = dict(desired=True, desired0=False, desiredR=True, fitted15=True, fitted=False, fitted0515=True, fitted05=False)
KIND_GAIN = dict(desired=r'$k_d$', desired0=r'$k_d$', desiredR=r'$k_R$', fitted15=r'$k_d$', fitted=r'$k_d$',
                 fitted0515=r'$k_d/2$', fitted05=r'$k_d/2$')
KIND_NAMES = {k: ('desired' if KIND_FF[k] == 'desired' else 'fitted') + ', ' + ('filter' if KIND_FILTER[k] else 'no filter')
              + ', ' + KIND_GAIN[k] for k in MJ_KINDS}
plt.rcParams.update({'font.size': 8, 'axes.linewidth': .6, 'lines.linewidth': 1., 'legend.fontsize': 7, 'pdf.fonttype': 42})

mis = json.loads((CHECKS / 'missions.json').read_text())
mz = np.load(CHECKS / 'missions.npz')
for plant in PLANTS:
    for t in ROLL:
        for kind in ('desired', 'fitted', 'fitted05'):
            r = mis[f'{plant}_{t}_{kind}']
            assert r['feedforward'] == KIND_FF[kind] and (r['velocity_filter_hz'] > 0) == KIND_FILTER[kind], (plant, t, kind)


def mission_panel(ax, plant, team, kinds):
    for kind, color, ls in (('desired', '#c44', '-'), ('fitted', '#1f4e9c', '-'), ('fitted05', '#1f4e9c', '--')):
        if kind in kinds:
            key = f'{plant}_{team}_{kind}'
            ax.semilogy(mz[f'{key}:t'], np.maximum(mz[f'{key}:W_max'], 1e-13), color=color, lw=.6, ls=ls, label=KIND_NAMES[kind])
    c = mz[f'{plant}_{team}_fitted:carry']
    ax.axvspan(c[0], c[1], color='#888', alpha=.10, lw=0)
    ax.grid(alpha=.3, lw=.4)
    ax.set_xlim(0., mz[f'{plant}_{team}_fitted:t'][-1])


fig, axes = plt.subplots(2, 4, figsize=(7.16, 3.2), sharex='col', sharey='row')
for i, plant in enumerate(PLANTS):
    for j, t in enumerate(ROLL):
        mission_panel(axes[i, j], plant, t, ('desired', 'fitted', 'fitted05'))
        if i == 0:
            axes[i, j].set_title(t, fontsize=8, family='monospace')
        else:
            axes[i, j].set_xlabel('time (s)')
    axes[i, 0].set_ylabel(('ideal constraints' if plant == 'pinocchio' else 'MuJoCo') + '\n$W$ (J)')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, .95), pad=.4)
fig.savefig(FIGS / 'fig_mission_all.pdf')
fig.savefig(FIGS / 'fig_mission_all.png', dpi=220)
print('written', FIGS / 'fig_mission_all.pdf')
