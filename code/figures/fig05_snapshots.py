"""Fig. 5: recorded snapshots of one three-robot RB-ROBOUT+/UR30 MuJoCo mission with the fitted-reference controller
(run mujoco_team3_fitted): (a) extraction from the rack, (b) transport with three active grasps, (c) delivery and load
transfer to the mounting supports.

The frames are taken from the mission video of that run (rendered by the MuJoCo workflow with --video, 25 frames/s at
four times real time) at simulated times t = 48, 176 and 220 s:
    ffmpeg -ss <t/4> -i mujoco_team3_fitted/mission.mp4 -frames:v 1 results/figures/frames/team3_t<t>.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from paths import FIGS, FRAMES as FRAME_DIR

SHOTS = ((48, (130, 60, 750, 435), '(a) Pick  $t\\approx48$ s', 'Extract beam from rack'),
         (176, (150, 40, 770, 415), '(b) Carry  $t\\approx176$ s', 'Transport with three active grasps'),
         (220, (140, 60, 760, 435), '(c) Deliver  $t\\approx220$ s', 'Transfer to mounting supports'))

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.85))
for ax, (t, box, title, label) in zip(axes, SHOTS):
    im = Image.open(FRAME_DIR / f'team3_t{t}.png').convert('RGB').crop(box)
    ax.imshow(im, interpolation='lanczos')
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xlabel(label, fontsize=8, labelpad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(.4); side.set_color('#888888')
fig.tight_layout(pad=.3, w_pad=.6)
fig.savefig(FIGS / 'fig_pick_carry_deliver.pdf', dpi=300)
fig.savefig(FIGS / 'fig_pick_carry_deliver.png', dpi=220)
print('written', FIGS / 'fig_pick_carry_deliver.pdf')
