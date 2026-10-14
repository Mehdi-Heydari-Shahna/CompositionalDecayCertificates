"""Fig. 3: error storage of team3 across a release (t = 1 s) and a plastic latch at the present relative pose (t = 2 s)
on the compiled ideal-constraint model, with a reference that is continuous at the events and with a retreat of the
released arm that starts with a reference jump.  Circles: the values after each event predicted by Proposition 22.

Data: results/checks/sim_team3_events.{json,npz} and sim_team3_events_jump.{json,npz}, written by
    python code/checks/sim.py team3 --T 3 --amp-scale 0.5 --events
    python code/checks/sim.py team3 --T 3 --amp-scale 0.5 --events --release-jump 0.15 --tag events_jump
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from paths import CHECKS, FIGS

plt.rcParams.update({'font.size': 8, 'axes.linewidth': .6, 'lines.linewidth': 1., 'legend.fontsize': 7, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(3.45, 2.0))
for label, tag, col in (('continuous reference', 'events', 'C0'), ('retreat starts with a jump', 'events_jump', 'C3')):
    z = np.load(CHECKS / f'sim_team3_{tag}.npz')
    run = json.loads((CHECKS / f'sim_team3_{tag}.json').read_text())
    ax.semilogy(z['t'], z['W'], col, label=label)
    for e in run['events']:
        assert abs(e['W_after'] - e['W_predicted']) <= 1e-9 * max(e['W_after'], 1e-30) + 1e-18    # Proposition 22(a)
        ax.plot([e['t']], [e['W_predicted']], 'o', ms=3.5, mfc='none', mec='k', mew=.7)
ax.set_xlabel('time (s)')
ax.set_ylabel(r'$W$ (J)')
ax.grid(alpha=.3, lw=.4)
ax.legend(frameon=False)
fig.tight_layout(pad=.4)
fig.savefig(FIGS / 'fig_events.pdf')
fig.savefig(FIGS / 'fig_events.png', dpi=220)
print('written', FIGS / 'fig_events.pdf')
