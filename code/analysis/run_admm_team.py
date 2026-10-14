"""Sharing ADMM for problem P1 (Remark 17): python3 run_admm_team.py TEAM ITERATIONS.
"""
import json
import sys

import numpy as np
from synth import Design
from teamdata import RESULTS
name = sys.argv[1]; iters = int(sys.argv[2])
qs = np.load(RESULTS / f'snapshots_{name}.npy')
des = Design(name, qs)
opt = des.centralized('P1', kappa=des.kappa_mis)['value']
a = des.admm(des.kappa_mis, rho=0.3, iters=iters, tol=1e-8, log_every=25, alpha=1.6, balance=False)
gaps = [h['cost'] / opt - 1 for h in a['history']]
K = des.certified(a['k'])
res = dict(team=name, optimum=opt, cost=a['cost'], gap=gaps[-1], gap_at={str(k): gaps[k - 1] for k in (25, 50, 100, 200, 300) if k <= len(gaps)},
           certified=K, kappa=des.kappa_mis, iterations=a['iterations'], seconds=a['seconds'], numbers_per_robot_per_iteration=a['numbers_per_robot_per_iteration'],
           history=a['history'], k_ratio=(a['k'] / des.kmis).tolist())
json.dump(res, open(RESULTS / f'admm_{name}.json', 'w'), indent=1)
print('DONE', name, 'gap', gaps[-1], 'certified', K, 'kappa', des.kappa_mis, flush=True)
