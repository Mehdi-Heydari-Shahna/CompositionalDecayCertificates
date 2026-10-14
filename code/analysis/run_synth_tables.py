"""Table 2: problems P1 and P2 of Section 7.4 over the 24 snapshots, centralized and robot-level (Theorem 16(d)),
with every returned gain set rechecked by generalized eigenvalues.
"""
import json, time, numpy as np
from synth import Design
from teamdata import RESULTS
out = {}
for name in ['team2', 'team3', 'ridgeback2', 'ridgeback3']:
    t0 = time.time()
    qs = np.load(RESULTS / f'snapshots_{name}.npy')
    des = Design(name, qs)
    c1 = des.centralized('P1', kappa=des.kappa_mis); d1 = des.distributed_joint('P1', kappa=des.kappa_mis)
    c2 = des.centralized('P2'); d2 = des.distributed_joint('P2')
    r = dict(snapshots=len(qs), joint_ports=des.nJ, kappa_mission=des.kappa_mis,
             P1=dict(central=c1['value'], central_status=c1['status'], distributed=d1['value'], distributed_status=d1['status'],
                     certified_central=des.certified(c1['k']), certified_distributed=des.certified(d1['k']),
                     ratios=(c1['k'] / des.kmis).tolist(), at_lower=int(np.sum(c1['k'] <= 0.1001 * des.kmis)), at_upper=int(np.sum(c1['k'] >= 1.9999 * des.kmis))),
             P2=dict(central=c2['kappa'], central_status=c2['status'], distributed=d2['kappa'], distributed_status=d2['status'],
                     certified_central=des.certified(c2['k']), certified_distributed=des.certified(d2['k']),
                     ratios=(c2['k'] / des.kmis).tolist()),
             seconds=time.time() - t0)
    out[name] = r
    print(name, f"kmis={des.kappa_mis:.4f} nJ={des.nJ} | P1 cen {c1['value']:.5f} ({c1['status']}) dis {d1['value']:.5f} ({d1['status']}) cert {r['P1']['certified_central']:.4f}/{r['P1']['certified_distributed']:.4f} | "
          f"P2 cen {c2['kappa']:.4f} ({c2['status']}) dis {d2['kappa']:.4f} ({d2['status']}) cert {r['P2']['certified_central']:.4f}/{r['P2']['certified_distributed']:.4f} [{r['seconds']:.0f}s]", flush=True)
json.dump(out, open(RESULTS / 'synth_tables.json', 'w'), indent=1)
