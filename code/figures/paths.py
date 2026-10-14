"""Folders of the figure scripts: the recorded results they read and the folder they write to.

Every folder can be replaced by an environment variable: ANALYSIS_RESULTS, CHECKS_RESULTS, FIG_DIR.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = Path(os.environ.get('ANALYSIS_RESULTS', REPO / 'results' / 'analysis'))
CHECKS = Path(os.environ.get('CHECKS_RESULTS', REPO / 'results' / 'checks'))
FIGS = Path(os.environ.get('FIG_DIR', REPO / 'results' / 'figures'))
FRAMES = REPO / 'results' / 'figures' / 'frames'          # video frames used by Figs. 5 and 12
FIGS.mkdir(parents=True, exist_ok=True)
