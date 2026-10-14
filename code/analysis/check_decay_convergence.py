"""Step-size convergence of the decay runs of Fig. 7 (renewal in every plant step), from the 0.5 ms and 1 ms runs.

For each run: time to W/W(0) = 1e-3 at both steps; excess of the fitted decay rate over the certified rate at the jump
at both steps; and the margin m(t) = log(bound(t)) - log(W(t)/W(0)) (non-negative: the bound holds) at both steps and
its first-order Richardson extrapolation 2 m_0.5(t) - m_1(t), on the common time grid up to W/W(0) = 1e-3 (0.5 ms run).
"""
import json
import os
from pathlib import Path

import numpy as np

RESULTS = Path(os.environ.get('ANALYSIS_RESULTS', Path(__file__).resolve().parents[2] / 'results' / 'analysis')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3')

if __name__ == '__main__':
    out = {}
    for team in TEAMS:
        a, b = np.load(RESULTS / f'decay_gains_{team}.npz'), np.load(RESULTS / f'decay_gains_{team}_1ms.npz')
        sa = json.loads((RESULTS / f'decay_gains_{team}.json').read_text())
        sb = json.loads((RESULTS / f'decay_gains_{team}_1ms.json').read_text())
        out[team] = {}
        for key in ('mission', 'P1', 'P2'):
            ta, ra, ua = a[f'{key}:t'], a[f'{key}:ratio'], a[f'{key}:upper']
            tb, rb, ub = b[f'{key}:t'], b[f'{key}:ratio'], b[f'{key}:upper']
            end = sa[key]['time_to_1e3']
            ia = {round(t, 6): i for i, t in enumerate(ta)}
            common = [(ia[round(t, 6)], j) for j, t in enumerate(tb) if round(t, 6) in ia and t <= end + 1e-9]
            i_a = np.array([c[0] for c in common])
            i_b = np.array([c[1] for c in common])
            ma = np.log(ua[i_a]) - np.log(ra[i_a])
            mb = np.log(ub[i_b]) - np.log(rb[i_b])
            mx = 2. * ma - mb
            out[team][key] = dict(
                time_to_1e3={'0.5ms': sa[key]['time_to_1e3'], '1ms': sb[key]['time_to_1e3']},
                fitted_over_certified={'0.5ms': sa[key]['rate_fitted'] / sa[key]['rate_at_jump'][0] - 1.,
                                       '1ms': sb[key]['rate_fitted'] / sb[key]['rate_at_jump'][0] - 1.},
                min_margin={'0.5ms': float(ma.min()), '1ms': float(mb.min()), 'extrapolated': float(mx.min())},
                final_margin={'0.5ms': float(ma[-1]), '1ms': float(mb[-1]), 'extrapolated': float(mx[-1])},
                points=len(common))
            r = out[team][key]
            print(team, key, json.dumps({k: v for k, v in r.items() if k != 'points'}), flush=True)
    (RESULTS / 'check_decay_convergence.json').write_text(json.dumps(out, indent=1))
