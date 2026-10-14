"""Payload realizability (hypothesis of Proposition 7, Theorem 8(b), Theorem 16(b)): every robot's own rows admit every
payload velocity, at the nominal carrying configuration and at the 24 snapshots of each team."""
import json

import numpy as np

from teamdata import RESULTS, TEAMS, Star


if __name__ == '__main__':
    out = {}
    for name in TEAMS:
        nominal = Star(name)
        rec = dict(nominal=all(nominal.realizable(r) for r in nominal.robots), snapshots=[])
        for q in np.load(RESULTS / f'snapshots_{name}.npy'):
            s = Star(name, q=q)
            rec['snapshots'].append(all(s.realizable(r) for r in s.robots))
        rec['all_snapshots'] = all(rec['snapshots'])
        out[name] = rec
        print(name, 'nominal:', rec['nominal'], 'snapshots:', sum(rec['snapshots']), '/', len(rec['snapshots']), flush=True)
    (RESULTS / 'check_realizability.json').write_text(json.dumps(out, indent=1))
