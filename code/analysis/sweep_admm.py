"""Choice of the ADMM penalty and over-relaxation for P1 on team2, on 3 of the 24 snapshots.
"""
import json

import numpy as np
from synth import Design
from teamdata import RESULTS
name = 'team2'
qs = np.load(RESULTS / f'snapshots_{name}.npy')[::8]
des = Design(name, qs)
opt = des.centralized('P1', kappa=des.kappa_mis)['value']
res = []
for rho, alpha, bal in [(0.3, 1.6, False), (1.0, 1.6, False), (3.0, 1.6, False), (1.0, 1.0, False), (1.0, 1.6, True)]:
    a = des.admm(des.kappa_mis, rho=rho, iters=300, tol=1e-7, log_every=0, alpha=alpha, balance=bal)
    gaps = [h['cost'] / opt - 1 for h in a['history']]
    r = dict(rho=rho, alpha=alpha, balance=bal, gap_100=gaps[min(99, len(gaps)-1)], gap_300=gaps[-1], cert=des.certified(a['k']), kappa=des.kappa_mis, sec=a['seconds'])
    res.append(r); print(json.dumps(r), flush=True)
json.dump(res, open(RESULTS / 'sweep_admm.json', 'w'), indent=1)
