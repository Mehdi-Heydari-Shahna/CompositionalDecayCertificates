"""Lemma 21 and Proposition 22 (constraint events) checked on the six compiled teams, pointwise.

Mode G: every robot grasps the payload.  Mode R: the last robot has released it.  ker A_G is a subspace of ker A_R.
P_m is the M-orthogonal projector onto ker A_m (from the compiled lift: P = E (E^T M E)^-1 E^T M).

  attach  (R -> G):  v+ = P_G v-   (checked against the impulse equations with the stacked rows, by least squares)
                     Carnot:  T- - T+ = 1/2 |v- - v+|_M^2
                     error reset:  s+ = P_G (s- + v_r- - v_r+)                       (any consistent references)
                     projected raw reference:  W+ = W- - 1/2 |(P_R - P_G) s^-|_M^2    (s^ = v - v^)
                     reference continuous and admissible in both modes:  W- - W+ = Carnot loss
  release (G -> R):  v+ = v-,  projected raw reference:  W+ = W- + 1/2 |(P_R - P_G) v^|_M^2
  rate:              lambda_min on ker A_G  >=  lambda_min on ker A_R  (same gains)
"""
import json

import numpy as np

from lib import RESULTS, TEAMS, Team, gen_eig, m_projector

rng = np.random.default_rng(11)
out = {}
for name in TEAMS:
    T = Team(name)
    q = T.q0
    G = T.mode()
    last = T.robots[-1]
    R = T.mode(grasps=tuple(g for g in G.grasps if T.comp.connections[g]['robot'] != last))
    M = T.M(q)
    EG, ER = T.E(q, G), T.E(q, R)
    AG = T.rows(q, G)[0]
    PG, PR = m_projector(M, EG), m_projector(M, ER)
    nrm = lambda x: float(np.sqrt(x @ M @ x))
    rel = lambda a, b: float(abs(a - b) / max(abs(a), abs(b), 1e-300))
    worst = dict(nested=float(np.max(np.abs(PG @ PR - PG)) + np.max(np.abs(PR @ PG - PG))), projector_AG=float(np.max(np.abs(AG @ PG))))
    keys = ('impulse', 'carnot', 'reset', 'attach_projected', 'attach_continuous', 'release_projected', 'nonexpansive')
    for k in keys:
        worst[k] = 0.
    n = T.n
    for _ in range(20):
        vm = ER @ rng.standard_normal(ER.shape[1])
        vp = PG @ vm
        # the impulse equations M (v+ - v-) = A^T Lam, A v+ = 0 with the stacked rows of the mode
        KKT = np.block([[M, -AG.T], [AG, np.zeros((AG.shape[0], AG.shape[0]))]])
        sol = np.linalg.lstsq(KKT, np.r_[M @ vm, np.zeros(AG.shape[0])], rcond=None)[0]
        worst['impulse'] = max(worst['impulse'], float(np.max(np.abs(sol[:n] - vp)) / np.max(np.abs(vm))))
        worst['carnot'] = max(worst['carnot'], rel(.5 * vm @ M @ vm - .5 * vp @ M @ vp, .5 * nrm(vm - vp) ** 2))
        vrm, vrp = ER @ rng.standard_normal(ER.shape[1]), EG @ rng.standard_normal(EG.shape[1])
        sm, sp = vm - vrm, vp - vrp
        worst['reset'] = max(worst['reset'], float(np.max(np.abs(sp - PG @ (sm + vrm - vrp))) / np.max(np.abs(sp))))
        worst['nonexpansive'] = max(worst['nonexpansive'], nrm(sp) / (nrm(sm) + nrm(PG @ (vrm - vrp))))
        vhat = rng.standard_normal(n)
        shat = vm - vhat
        Wm, Wp = .5 * nrm(PR @ shat) ** 2, .5 * nrm(vp - PG @ vhat) ** 2
        worst['attach_projected'] = max(worst['attach_projected'], rel(Wm - Wp, .5 * nrm((PR - PG) @ shat) ** 2))
        vrc = EG @ rng.standard_normal(EG.shape[1])                 # admissible in both modes, continuous
        worst['attach_continuous'] = max(worst['attach_continuous'],
                                         rel(.5 * nrm(vm - vrc) ** 2 - .5 * nrm(vp - vrc) ** 2, .5 * nrm(vm - vp) ** 2))
        vg = EG @ rng.standard_normal(EG.shape[1])                  # release: the velocity does not jump
        Wm, Wp = .5 * nrm(vg - PG @ vhat) ** 2, .5 * nrm(vg - PR @ vhat) ** 2
        worst['release_projected'] = max(worst['release_projected'], rel(Wp - Wm, .5 * nrm((PR - PG) @ vhat) ** 2))
    K = T.gains(q, 10., G) if name not in ('husky2', 'husky3') else np.eye(T.nf)
    Dm = T.B @ K @ T.B.T + T.D
    kG = float(gen_eig(EG.T @ Dm @ EG, EG.T @ M @ EG)[0])
    kR = float(gen_eig(ER.T @ Dm @ ER, ER.T @ M @ ER)[0])
    out[name] = dict(dims=dict(n=n, d_grasped=int(EG.shape[1]), d_released=int(ER.shape[1])), worst=worst,
                     rate_grasped=kG, rate_released=kR)
    print(name, json.dumps(out[name]), flush=True)
(RESULTS / 'exp3_events.json').write_text(json.dumps(out, indent=1))
