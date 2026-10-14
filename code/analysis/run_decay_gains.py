"""Decay of the fitted-reference storage W on the ideal-constraint (Pinocchio) plant for a given set of joint gains.

Protocol of tools/run_decay.py of the package (the team holds the payload at a route station, settles, and receives an
admissible velocity jump of M-norm 0.1; fitted feedforward), with these changes:
  * the rate gains of the arm/lift joint loops can be replaced (the joint stiffnesses k_p are kept, so the loops
    request w = v_ref + (k_p/k_d)(q_ref - q) with the new k_d);
  * the jump is along the slowest certified direction of the gains used (the generalized eigenvector of
    (D_r, M_r) for kl), where the certificate is tight;
  * the plant step dt and the renewal interval of the references, requests and fitted feedforward are arguments.
    Renewal in every plant step (renew = dt, the default of the command line) approximates the continuous-time loops
    of Theorem 19(b); renew = 10 ms is the sampled implementation of the package (Theorem 19(c) applies to it).

Usage: python3 run_decay_gains.py TEAM [DT [RENEW [TAG]]]     (defaults: DT = 0.0005 s, RENEW = DT, TAG = '')
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.linalg import eigh

from teamdata import PKG, RESULTS  # noqa: F401  (puts the package on sys.path)


def run(team, joint_gains=None, at=2.0, size=.1, duration=.3, settle=.1, dt=.002, ff_rate='slow', renew=.01):
    from backends.pin_plant import PinPlant
    from mission.controller import TeamController
    from mission.runner import carry_configuration
    from teamc.io import INPUTS, load_team, read_json
    comp = load_team(team)
    mission = read_json(INPUTS / 'missions' / f'{team}.json')
    plant = PinPlant(comp, timestep=dt)
    slow_every = max(int(round(renew / dt)), 1)
    ctrl = TeamController(comp, mission, plant.dt, slow_every=slow_every,
                          options=dict(feedforward='fitted', feedforward_rate=ff_rate, monitor=True))
    if joint_gains is not None:
        for rid, ix in ctrl.ix.items():
            kd = np.asarray(joint_gains[rid], float)
            assert kd.shape == ctrl.kd[rid].shape
            ctrl.kd[rid] = kd
            ctrl.K[ix['joint_ports']] = kd
            ctrl.rate_ratio[ix['joint_ports']] = ctrl.kp[rid] / kd
    ctrl.governor.speed_limit = lambda l: 0.
    q0 = carry_configuration(comp, mission, ctrl, at)
    plant.reset(q0)
    for cid in ctrl.grasp_of.values():
        ctrl.on_latched(cid, plant.latch(cid))
    meas = plant.measure()
    meas['obstacle'] = None
    ctrl.start_carrying(meas, at)
    ref = ctrl.reference
    nsettle = int(round(settle / (plant.dt * slow_every))) * slow_every
    for k in range(nsettle):
        if k % slow_every == 0:
            meas['obstacle'] = None
            ctrl.slow(meas)
        plant.set_efforts(ctrl.fast(meas))
        plant.step()
        meas = plant.measure()
    q0 = meas['q'].copy()
    lift = ctrl.model.lift(q0, ctrl.mode)
    E = lift.E
    L = E[ctrl.port_coordinate]
    Mr = E.T @ ctrl.tree.mass(q0) @ E
    Dr = L.T @ (ctrl.K[:, None] * L) + E.T @ (ctrl.model.damping[:, None] * E)
    vals, vecs = eigh(Dr, Mr)
    u = vecs[:, 0]
    u = size * u / np.sqrt(u @ Mr @ u)
    plant.kick(E @ u)
    meas = plant.measure()
    rows = []
    steps = int(round(duration / plant.dt))
    for k in range(steps + 1):
        if k % slow_every == 0:
            meas['obstacle'] = None
            ctrl.slow(meas)
        f = ctrl.fast(meas)
        q, v = meas['q'], meas['v']
        fit = ref.fit(q, ctrl.mode, ctrl.K, ctrl._requested(q, v, meas['t'] - ctrl.t_slow))
        fit['M'] = ctrl.tree.mass(q, kinematics=fit['kin'])
        Q = ref.quantities(fit, v, ctrl.K)
        rows.append((meas['t'], Q['W'], Q['rate'][0], Q['rate'][1]))
        if k == steps:
            break
        plant.set_efforts(f)
        plant.step()
        meas = plant.measure()
    A = np.array(rows)
    t = A[:, 0] - A[:, 0][0]
    W, kmin, kmax = A[:, 1], A[:, 2], A[:, 3]
    trap = lambda y: np.concatenate([[0.], np.cumsum(.5 * (y[1:] + y[:-1]) * np.diff(t))])
    upper, lower = np.exp(-2. * trap(kmin)), np.exp(-2. * trap(kmax))
    ratio = W / W[0]
    above = ratio > 1e-3
    last = int(np.argmin(above)) if not above.all() else len(t)
    summary = dict(team=team, at=at, rate_at_jump=[float(vals[0]), float(vals[-1])], W0=float(W[0]),
                   max_ratio=float(ratio.max()), excess_over_upper=float(np.max((ratio / upper - 1.)[:last])) if last else float('nan'),
                   time_to_1e3=float(t[last]) if last < len(t) else None,
                   rate_fitted=float(-np.polyfit(t[:last], np.log(ratio[:last]), 1)[0] / 2.) if last > 5 else float('nan'))
    return summary, dict(t=t, ratio=ratio, upper=upper, lower=lower, kmin=kmin, kmax=kmax)


if __name__ == '__main__':
    team = sys.argv[1]
    dt = float(sys.argv[2]) if len(sys.argv) > 2 else .0005
    renew = float(sys.argv[3]) if len(sys.argv) > 3 else dt
    tag = sys.argv[4] if len(sys.argv) > 4 else ''
    variants = {'mission': None}
    synth = json.loads((RESULTS / 'synth_tables.json').read_text())[team]
    from teamdata import mission_gains
    K0, Kp, jp, wp = mission_gains(team)
    for key in ('P1', 'P2'):
        ratios = np.asarray(synth[key]['ratios'])
        gains, off = {}, 0
        for rid in jp:
            n = len(jp[rid])
            gains[rid] = K0[jp[rid]] * ratios[off:off + n]
            off += n
        variants[key] = gains
    out, arrays = {}, {}
    for name, gains in variants.items():
        s, A = run(team, gains, dt=dt, renew=renew, ff_rate='fast')
        s.update(dt=dt, renew=renew)
        out[name] = s
        arrays.update({f'{name}:{k}': v for k, v in A.items()})
        print(team, name, json.dumps(s), flush=True)
    (RESULTS / f'decay_gains_{team}{tag}.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(RESULTS / f'decay_gains_{team}{tag}.npz', **arrays)
