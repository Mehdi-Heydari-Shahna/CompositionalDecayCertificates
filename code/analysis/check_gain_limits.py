"""Rate limits of the joint and wheel loops (Section 7.4): centralized rate at the nominal carrying configuration when the
arm/lift rate gains or the wheel gains are scaled (mission gains otherwise)."""
import json

import numpy as np

from teamdata import RESULTS, TEAMS, Star, mission_gains


if __name__ == '__main__':
    out = {}
    for name in TEAMS:
        st = Star(name)
        K, _, jp, wp = mission_gains(name)
        J = np.concatenate([jp[r] for r in jp])
        Wp = np.concatenate([wp[r] for r in wp])
        rows = []
        for arm, wheel in [(1, 1), (100, 1), (1, 2), (100, 2), (100, 100)]:
            K2 = K.copy()
            K2[J] *= arm
            K2[Wp] *= wheel
            rows.append(dict(arm_scale=arm, wheel_scale=wheel, rate=st.rate(K2)))
        out[name] = rows
        print(name, [(r['arm_scale'], r['wheel_scale'], round(r['rate'], 4)) for r in rows], flush=True)
    (RESULTS / 'check_gain_limits.json').write_text(json.dumps(out, indent=1))
