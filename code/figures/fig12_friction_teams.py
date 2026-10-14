"""Fig. 12: friction-driven (Husky) teams (Section 7.8).

(a)-(c): frames of the MuJoCo mission of husky3 with the supplied controller (run mujoco_husky3 of the reference
results), taken from its mission video (rendered by the MuJoCo workflow with --video, 25 frames/s at four times real
time) at simulated times t = 82, 170 and 245 s:
    ffmpeg -ss <t/4> -i mujoco_husky3/mission.mp4 -frames:v 1 results/figures/frames/husky3_t<t>.png
(d): the three friction runs of husky2 (results/checks/exp5_husky2_floor025.npz, exp5_husky2_scrub.npz and
exp5_husky2_floor080.npz, written by code/checks/exp5_friction_sim.py): W(t)/W(0), with the decay at the rate of
Proposition 23(b) at zero slip and the standing loads (results/checks/exp5_friction_pointwise.json,
slowest_rate_at_rest) for comparison.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from paths import CHECKS as CHK, FIGS, FRAMES as FRAME_DIR

OUT = FIGS / 'fig_husky.pdf'
FRAMES = ((82, (150, 50, 760, 480), '(a) extraction'), (170, (150, 40, 760, 470), '(b) transport'),
          (245, (240, 50, 850, 480), '(c) delivery'))
RUNS = (('floor025', r'$\mu=0.25$', '#2a78d6'), ('scrub', r'$\mu=0.25$, scrub', '#eb6834'),
        ('floor080', r'$\mu=0.8$', '#1baf7a'))

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 7.5, 'axes.linewidth': .5,
                     'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.edgecolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.labelcolor': '#0b0b0b'})
fig = plt.figure(figsize=(7.16, 1.36))
widths = [1.62, 1.62, 1.62, 1.9]
left, gap = .005, .012
x = left
axes = []
for w in widths:
    axes.append(fig.add_axes([x, .03 if len(axes) < 3 else .24, w / 7.16 - (0 if len(axes) < 3 else .055), .82 if len(axes) < 3 else .62]))
    x += w / 7.16 + gap
for ax, (t, box, title) in zip(axes[:3], FRAMES):
    im = Image.open(FRAME_DIR / f'husky3_t{t}.png').convert('RGB').crop(box)
    ax.imshow(np.asarray(im), interpolation='lanczos')
    ax.set_axis_off()
    ax.set_title(f'{title}, $t\\approx{t}$ s', fontsize=7.2, pad=2)
ax = axes[3]
ax.set_position([axes[3].get_position().x0 + .055, .24, 1.9 / 7.16 - .065, .62])
rate0 = json.loads((CHK / 'exp5_friction_pointwise.json').read_text())['husky2']['slowest_rate_at_rest']
for key, label, color in RUNS:
    a = np.load(CHK / f'exp5_husky2_{key}.npz')
    style = dict(lw=2.6, alpha=.45) if key == 'floor025' else dict(lw=1.0)      # floor025 and floor080 nearly coincide
    ax.semilogy(a['t'], a['W'] / a['W'][0], color=color, label=label, solid_capstyle='butt', **style)
tt = np.linspace(.3, 1., 50)
ref = np.load(CHK / 'exp5_husky2_floor025.npz')
anchor = float(np.interp(.3, ref['t'], ref['W'] / ref['W'][0]))
ax.semilogy(tt, anchor * np.exp(-2. * rate0 * (tt - .3)) * 1.6, color='#52514e', lw=.8, ls=(0, (3, 2)),
            label=f'rate {rate0:.3f} s$^{{-1}}$, zero slip')
ax.set_xlim(0, 1)
ax.set_ylim(1e-3, 1.5)
ax.set_xlabel('time (s)', labelpad=1)
ax.set_ylabel(r'$W(t)/W(0)$', labelpad=1)
ax.set_title('(d) friction runs, husky2', fontsize=7.2, pad=2)
ax.grid(True, which='major', lw=.3, color='#d9d8d4')
for s_ in ('top', 'right'):
    ax.spines[s_].set_visible(False)
ax.legend(frameon=False, fontsize=5.6, handlelength=1.6, loc='lower left', borderaxespad=.1, labelspacing=.25)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix('.png'), dpi=220)
print('written', OUT)
