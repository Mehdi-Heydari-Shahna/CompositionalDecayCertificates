"""Fig. 11: carry stage on the ideal-constraint plant under conditions outside the statements: (a) median storage over
the cruise against the renewal interval h (team2, ridgeback3); for ridgeback3, (b) varying part of the payload error when
the controller's model of the beam has 0.7-1.3 times its mass, (c) rms grasp moment under white noise on the measured
rates, and (d) median storage with measurement delays of 2 and 4 ms.

Data: results/checks/exp12_team2.json and exp12_ridgeback3.json, written by code/checks/exp12_robustness.py; the runs
in these files are those of ``--only sampling,mass,noise,delay,half`` (team2 with ``--masses 0.7,1.3``).
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import CHECKS, FIGS

plt.rcParams.update({'font.size': 8, 'axes.linewidth': .6, 'lines.linewidth': 1., 'legend.fontsize': 7, 'pdf.fonttype': 42})
ROB_TEAMS = ('ridgeback3', 'team2')
rob = {t: {r['label']: r for r in json.loads((CHECKS / f'exp12_{t}.json').read_text())} for t in ROB_TEAMS}
Rb = rob['ridgeback3']
for t, runs_t in rob.items():
    for r in runs_t.values():
        assert r['status'] == 'running' and r['duration_s'] == 8. and abs(r['speed_end'] - .45) < 1e-6, (t, r['label'])
HS = (2, 4, 10, 20)
um = lambda x: 1e6 * x  # noqa: E731

BLUE, LIGHT, RED, GREEN = '#1f4e9c', '#6a9fd8', '#c44', '#2a8a4a'
fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.0))
ax = axes[0]
for t, mk in zip(ROB_TEAMS, ('o', 's')):
    ax.loglog(HS, [rob[t][f'sampling {h} ms']['W_median'] for h in HS], marker=mk, ms=3, lw=.9, color=BLUE, label=t)
h2 = np.array([8., 25.])
ax.loglog(h2, Rb['sampling 10 ms']['W_median'] * (h2 / 10.) ** 2 * .45, color='k', lw=.6, ls='--', label=r'$\propto h^2$')
ax.set_xlabel('update interval $h$ (ms)'); ax.set_ylabel('median $W$ (J)')
ax.set_xticks(HS); ax.set_xticklabels([str(h) for h in HS]); ax.minorticks_off()
ax.legend(frameon=False, fontsize=6, loc='upper left')
ax.set_title('(a) sampling', fontsize=7.5)
ax = axes[1]
MS = [.7, 1., 1.3]


def mkey(s_, kind):
    base = 'nominal' if s_ == 1. else f'mass x{s_:g}'
    return {'fitted': base, 'desired': base + ', desired', 'half': 'half ' + base}[kind]


for kind, color, mk, name in (('fitted', BLUE, 'o', r'fitted, $K=k_d$'), ('half', GREEN, 's', r'fitted, $K=k_d/2$'),
                              ('desired', RED, 'x', 'desired motion')):
    ax.plot(MS, [um(Rb[mkey(s_, kind)]['payload_varying_max']) for s_ in MS], marker=mk, ms=3.5, lw=.9, color=color, label=name)
ax.set_xlabel('mass of the beam in the model'); ax.set_ylabel(r'varying payload error ($\mu$m)')
ax.set_xticks(MS)
ax.set_ylim(0., 1.45 * max(um(Rb[mkey(s_, 'desired')]['payload_varying_max']) for s_ in MS))
ax.legend(frameon=False, fontsize=5.5, loc='upper left', ncol=1)
ax.set_title('(b) mass error', fontsize=7.5)
ax = axes[2]
SD = [0., .001, .005]


def nl(sd, prefix, suffix):
    return prefix + ('nominal' if sd == 0. else f'noise {sd:g}') + suffix


for prefix, suffix, color, mk, name in (('', '', BLUE, 'o', r'fitted, $K=k_d$'), ('', ', filter', LIGHT, '^', 'fitted, filter'),
                                        ('half ', '', GREEN, 's', r'fitted, $K=k_d/2$'), ('', ', desired', RED, 'x', 'desired motion')):
    keys = [nl(sd, prefix, suffix) for sd in SD]
    if all(k in Rb for k in keys):
        ax.plot([1e3 * sd for sd in SD], [1e3 * Rb[k]['grasp_moment_rms'] for k in keys], marker=mk, ms=3.5, lw=.9, color=color, label=name)
ax.set_xlabel(r'rate noise ($10^{-3}$ rad/s, m/s)'); ax.set_ylabel('grasp moment, rms (mN m)')
ax.legend(frameon=False, fontsize=5.5, loc='upper left')
ax.set_title('(c) noise', fontsize=7.5)
ax = axes[3]
DL = [0, 1, 2]


def dl(d, prefix, suffix):
    return prefix + ('nominal' if d == 0 else f'delay {d}') + suffix


for prefix, suffix, color, mk, name in (('', '', BLUE, 'o', r'fitted, $K=k_d$'), ('half ', '', GREEN, 's', r'fitted, $K=k_d/2$'),
                                        ('', ', desired', RED, 'x', 'desired motion, filter')):
    keys = [dl(d, prefix, suffix) for d in DL]
    ax.semilogy([2 * d for d in DL], [Rb[k]['W_median'] for k in keys], marker=mk, ms=3.5, lw=.9, color=color, label=name)
ax.semilogy([4.3], [Rb['delay 2, filter']['W_median']], marker='^', ms=4.5, lw=0, mfc='none', mec=LIGHT, mew=1.,
            label='fitted, filter', zorder=5)
ax.semilogy([3.7], [Rb['delay 2, desired, no filter']['W_median']], marker='+', ms=6, lw=0, color=RED, label='desired, no filter')
ax.set_xlabel('delay (ms)'); ax.set_ylabel('median $W$ (J)')
ax.set_xticks([0, 2, 4])
ax.legend(frameon=False, fontsize=5, loc='center left')
ax.set_title('(d) delay', fontsize=7.5)
for ax in axes:
    ax.grid(alpha=.3, lw=.4)
fig.tight_layout(pad=.35, w_pad=.6)
fig.savefig(FIGS / 'fig_robust.pdf')
fig.savefig(FIGS / 'fig_robust.png', dpi=220)
print('written', FIGS / 'fig_robust.pdf')
