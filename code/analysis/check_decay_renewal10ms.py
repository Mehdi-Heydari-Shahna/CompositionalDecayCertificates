"""Decay runs of the sampled implementation: references, requests and fitted feedforward renewed every 10 ms.

Same protocol as run_decay_gains.py, but with the 10 ms renewal of the package (feedforward_rate='slow') and plant
steps of 2, 1 and 0.5 ms. Reports, per run and step, the time for W/W(0) to reach 1e-3, the fitted decay rate, the
certified rate at the jump, and the largest relative excess of W/W(0) over its certified bound before 1e-3
(zero means the bound holds). The renewal is a residual in the sense of Theorem 19(c); the bound is not guaranteed.

Usage: python3 check_decay_renewal10ms.py [TEAM:GAINS ...]      (default: all four teams, mission/P1/P2 gains)
"""
import json
import os
import sys

import numpy as np

from run_decay_gains import run
from teamdata import RESULTS, TEAMS, mission_gains


def gains_for(team, key):
    if key == 'mission':
        return None
    synth = json.loads((RESULTS / 'synth_tables.json').read_text())[team]
    K0, _, jp, _ = mission_gains(team)
    ratios = np.asarray(synth[key]['ratios'])
    gains, off = {}, 0
    for rid in jp:
        n = len(jp[rid])
        gains[rid] = K0[jp[rid]] * ratios[off:off + n]
        off += n
    return gains


if __name__ == '__main__':
    cases = [tuple(a.split(':')) for a in sys.argv[1:]] or [(t, k) for t in TEAMS for k in ('mission', 'P1', 'P2')]
    records = []
    for team, key in cases:
        for dt in (.002, .001, .0005):
            s, _ = run(team, gains_for(team, key), dt=dt, renew=.01, ff_rate='slow')
            rec = dict(team=team, gains=key, dt=dt, time_to_1e3=s['time_to_1e3'], rate_fitted=s['rate_fitted'],
                       rate_at_jump=s['rate_at_jump'][0], excess_over_upper=s['excess_over_upper'])
            records.append(rec)
            print(json.dumps(rec), flush=True)
    out = dict(protocol='references, requests and fitted feedforward renewed every 10 ms (feedforward_rate=slow), '
                        'plant steps 2, 1, 0.5 ms; P1/P2 gains of synth_tables.json', records=records)
    (RESULTS / os.environ.get('RENEW10_OUT', 'check_decay_renewal10ms.json')).write_text(json.dumps(out, indent=1))
