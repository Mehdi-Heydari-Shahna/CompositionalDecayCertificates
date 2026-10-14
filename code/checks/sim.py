"""The certified law on a compiled team with ideal edges, in the time domain (Propositions 3, 4 and 22).

Plant:      M v' + C v + g + D v = B tau + A_m^T lam,  A_m(q) v = 0      (compiled model, reduced form, RK4)
Reference:  u_r = y_d' - Lam (y - y_d) on the independent coordinates of the carrying mode,
            v_r = E(q) u_r at the ACTUAL configuration
Law:        tau = tau_r - K B^T (v - v_r);  (tau_r, lam_r): the solution of the required balance with the
            smallest efforts relative to the actuator bounds (a declared distribution; any other gives the same motion)

Checked along the run:
  W(t) - W(0) + int s^T (B K B^T + D) s + int <A v_r, lam - lam_r> = 0     (jumps at events accounted for)
  exp(-int k_max) <= |s|_M / |s(0)|_M <= exp(-int k_min)                    (between events)
  sum over modules of the error power through the edges = 0
  position error and reaction error -> 0
  Theorem 19: with u_fit = (L^T K L)^-1 L^T K B^T v_r (the admissible fit of the required actuator rates) and
  w_perp = B^T v_r - L u_fit, the error |v - E u_fit|_M, the internal effort K w_perp and the part of the reaction
  error that is not - B K w_perp are recorded (they matter with --inconsistent; otherwise w_perp = 0).
  With --inconsistent also the balance of Theorem 19(c) for the fitted reference v_fit = E u_fit,
      W_fit(t) - W_fit(0) + int s_fit^T (B K B^T + D) s_fit + int s_fit^T delta = 0,
      delta = M v_fit' + C(q,v) v_fit + g + D v_fit - B tau_f,   tau_f = tau_r of the law,
  with v_fit' taken along the motion, and the size |E^T delta|_{M_r^-1} of this residual.
  The reaction error is recorded separately for the rows whose multipliers are forces (N) and moments (N m).
Options:
  --inconsistent a     the reference of the last robot's arm violates its grasp rows by a field of size a
  --events             t = 1: the last robot releases;  t = 2: it latches at the current relative pose (plastic)
  --release-jump r     with --events: the retreat of the released arm starts with joint rate r (a reference jump)
"""
import argparse
import json
import time

import numpy as np

from lib import RESULTS, Team, gen_eig, m_projector, make_reference

ap = argparse.ArgumentParser()
ap.add_argument('team')
ap.add_argument('--T', type=float, default=3.)
ap.add_argument('--dt', type=float, default=.002)
ap.add_argument('--rate', type=float, default=10.)
ap.add_argument('--gain-scale', type=float, default=1.)
ap.add_argument('--inconsistent', type=float, default=0.)
ap.add_argument('--events', action='store_true')
ap.add_argument('--release-jump', type=float, default=0.)
ap.add_argument('--t-rel', type=float, default=1.)
ap.add_argument('--t-att', type=float, default=2.)
ap.add_argument('--amp-scale', type=float, default=1.)
ap.add_argument('--tag', default='')
args = ap.parse_args()

T = Team(args.team)
model, comp = T.model, T.comp
G = T.mode()
last = T.robots[-1]
cid_last = next(g for g in G.grasps if comp.connections[g]['robot'] == last)
R = T.mode(grasps=tuple(g for g in G.grasps if g != cid_last))
liftG0 = model.lift(T.q0, G)
indG = liftG0.independent
indR = model.lift(T.q0, R).independent
d = len(indG)
rng = np.random.default_rng(7)
K = args.gain_scale * T.gains(T.q0, args.rate, G)
Dm = T.B @ K @ T.B.T + T.D
bound = model.effort_bounds[:, 1]
LAM = 4.
T_REL, T_ATT = args.t_rel, args.t_att
yd, U0 = make_reference(T, indG, scale=args.amp_scale)


arm_last = np.array([i for i in T.coords[last] if i in set(liftG0.dependent) and 'wheel' not in T.ids[i]
                     and 'base' not in T.ids[i]], int)
field = np.zeros(T.n)
field[arm_last] = np.cos(np.arange(len(arm_last)))
i1, i2 = model._paths[cid_last]


def follow(q):
    """Register the grasp of the last robot at the present relative pose (the frame a latch would take)."""
    Tw = T.tree.sweep(q)[0]
    c = comp.connections[cid_last]
    model.register_grasp(cid_last, np.linalg.inv(Tw[i2]) @ Tw[i1] @ c['T1'])


def lift_of(q, key):
    if key == 'F':                                          # carrying mode with the grasp frame of the present pose
        follow(q)
        return model.lift(q, G)
    return model.lift(q, G if key == 'G' else R)


def Edot(q, v, key, h0=1e-6):
    h = h0 / max(1., float(np.max(np.abs(v))))
    return (lift_of(q + h * v, key).E - lift_of(q - h * v, key).E) / (2 * h)


def retreat(t):
    """Joint-rate profile of the released arm (added to the reference that follows the payload)."""
    if not args.events or not (T_REL <= t < T_ATT):
        return 0., 0.
    x = (t - T_REL) / (T_ATT - T_REL)
    a = .08
    if args.release_jump:                                   # starts with a jump of the reference, ends at rest
        return args.release_jump * np.cos(.5 * np.pi * x) ** 2, -args.release_jump * np.pi * np.sin(np.pi * x) / 2 / (T_ATT - T_REL)
    return a * np.sin(2 * np.pi * x) * np.sin(np.pi * x) ** 2, a * (2 * np.pi * np.cos(2 * np.pi * x) * np.sin(np.pi * x) ** 2
                                                                   + np.sin(2 * np.pi * x) * np.pi * np.sin(2 * np.pi * x)) / (T_ATT - T_REL)


def moment_rows(con):
    """True for the rows whose multipliers are moments (N m): the three angular rows of a grasp, pitch and roll of a
    fitted ground plane.  All other rows are velocities of points (multipliers in N)."""
    mom = np.zeros(con.J.shape[0], bool)
    for b in con.blocks:
        r = np.arange(b['rows'].start, b['rows'].stop)
        if b['kind'] == 'ground':
            if comp.robots[b['robot']].ground != 'contact':
                assert len(r) == 3
                mom[r[1:]] = True                           # rows [z, pitch, roll]
        elif b['kind'] != 'wheels':
            assert len(r) == 6
            mom[r[:3]] = True                               # twist rows: angular part first
    return mom


def ufit_at(t, q):
    """The admissible fit u_fit of the required actuator rates of the carrying mode, as a function of t and q."""
    E = model.lift(q, G).E
    L = T.B.T @ E
    y, yv, ya = yd(t)
    vr = E @ (yv - LAM * (q[indG] - y)) + args.inconsistent * field * np.sin(1.3 * t)
    return np.linalg.solve(L.T @ K @ L, L.T @ K @ (T.B.T @ vr))


def weighted_balance(A, rhs):
    """(tau_r, lam_r) with B tau_r + A^T lam_r = rhs and the smallest efforts relative to the actuator bounds."""
    cols = np.hstack([T.B * bound, A.T * 1e4])
    x = np.linalg.lstsq(cols, rhs, rcond=None)[0]
    tau_r, lam_r = bound * x[:T.nf], 1e4 * x[T.nf:]
    return tau_r, lam_r, float(np.max(np.abs(T.B @ tau_r + A.T @ lam_r - rhs)))


def law(t, q, u, key, full=False):
    """key: 'G' carrying, 'R' last robot released."""
    lift = lift_of(q, key)
    E, con = lift.E, lift.constraints
    A = con.J
    v = E @ u
    Ed = Edot(q, v, key)
    M, g = T.M(q), T.g(q)
    y, yv, ya = yd(t)
    e = q[indG] - y
    ur = yv - LAM * e
    urd = ya - LAM * (v[indG] - yv)
    if key == 'G':
        w, wd = (np.sin(1.3 * t), 1.3 * np.cos(1.3 * t)) if args.inconsistent else (0., 0.)
        vr = E @ ur + args.inconsistent * field * w
        ar = E @ urd + Ed @ ur + args.inconsistent * field * wd
    else:                                                   # the released arm follows the payload reference and retreats
        Ef = lift_of(q, 'F').E
        Efd = Edot(q, v, 'F')
        rho, rhod = retreat(t)
        vr = Ef @ ur + rho * field
        ar = Ef @ urd + Efd @ ur + rhod * field
    rhs = M @ ar + T.C(q, v, vr) + g + T.D @ vr
    tau_r, lam_r, bal = weighted_balance(A, rhs)
    s = v - vr
    tau = tau_r - K @ (T.B.T @ s)
    hv = T.b(q, v) + g + T.D @ v
    Mr = E.T @ M @ E
    ud = np.linalg.solve(Mr, E.T @ (T.B @ tau - hv - M @ (Ed @ u)))
    a = E @ ud + Ed @ u
    gen = M @ a + hv - T.B @ tau
    lam = np.linalg.lstsq(A.T, gen, rcond=None)[0]
    diss, react = float(s @ Dm @ s), float((A @ vr) @ (lam - lam_r))
    dfit = rfit = rho_fit = 0.
    if args.inconsistent and key == 'G':                    # Theorem 19(c): balance for the fitted reference
        h = 1e-6 / max(1., float(np.max(np.abs(v))))
        L_ = T.B.T @ E
        uf = np.linalg.solve(L_.T @ K @ L_, L_.T @ K @ (T.B.T @ vr))
        ufd = (ufit_at(t + h, q + h * v) - ufit_at(t - h, q - h * v)) / (2 * h)
        vf = E @ uf
        delta = M @ (E @ ufd + Ed @ uf) + T.C(q, v, vf) + g + T.D @ vf - T.B @ tau_r
        sf = v - vf
        dfit, rfit = float(sf @ Dm @ sf), float(sf @ delta)
        Ed_ = E.T @ delta
        rho_fit = float(np.sqrt(max(Ed_ @ np.linalg.solve(Mr, Ed_), 0.)))
    if not full:
        return v, ud, np.array([diss, react, dfit, rfit])
    ev = gen_eig(E.T @ Dm @ E, Mr)
    N = A @ np.linalg.solve(M, A.T)
    Avr = A @ vr
    flow = A.T @ (lam - lam_r)                              # error force of the edges on every coordinate
    p = [float(s[c] @ flow[c]) for c in [T.payload] + [T.coords[r] for r in T.robots]]
    Wm = [.5 * float(s[c] @ M[np.ix_(c, c)] @ s[c]) for c in [T.payload] + [T.coords[r] for r in T.robots]]
    # Theorem 19: the law only sees the required actuator rates w = B^T v_r
    L = T.B.T @ E
    wreq = T.B.T @ vr
    ufit = np.linalg.solve(L.T @ K @ L, L.T @ K @ wreq)
    wperp = wreq - L @ ufit
    sfit = v - E @ ufit
    internal = K @ wperp
    mom = moment_rows(con)
    dl = np.abs(lam - lam_r)
    fit = dict(rho_fit=rho_fit, dlam_force=float(dl[~mom].max()), dlam_moment=float(dl[mom].max()) if mom.any() else 0.,
               sfit=float(np.sqrt(max(sfit @ M @ sfit, 0.))), internal=float(np.max(np.abs(internal))),
               internal_ratio=float(np.max(np.abs(internal) / bound)), flow=float(np.max(np.abs(flow))),
               flow_rest=float(np.max(np.abs(flow + T.B @ internal))),
               wperp_K=float(np.sqrt(max(wperp @ K @ wperp, 0.))), wreq_K=float(np.sqrt(max(wreq @ K @ wreq, 0.))))
    return dict(**fit, W=.5 * float(s @ M @ s), diss=diss, react=react, e=float(np.max(np.abs(e))), kmin=float(ev[0]), kmax=float(ev[-1]),
                Av=float(np.max(np.abs(A @ v))), res=float(np.max(np.abs(con.residual))),
                dlam=float(np.max(np.abs(lam - lam_r))), lam=float(np.max(np.abs(lam))),
                lower=float(np.sqrt(max(Avr @ np.linalg.pinv(N, rcond=1e-12) @ Avr, 0.))), balance=bal,
                effort=float(np.max(np.abs(tau) / bound)), effort_ref=float(np.max(np.abs(tau_r) / bound)),
                p_sum=float(sum(p)), p_max=float(max(abs(x) for x in p)), W_modules_sum=float(sum(Wm)),
                s=s, v=v, vr=vr, M=M, E=E)


q = T.q0.copy()
u = U0.copy()                                              # initial velocity error
t, dt = 0., args.dt
key = 'G'
rec, events = [], []
I = np.zeros(4)                                             # int diss, int react, and the two integrals of Theorem 19(c)
jumps = 0.                                                  # sum of the jumps of W at the events
t0 = time.time()
nsteps = int(round(args.T / dt))
k_rel, k_att = int(round(T_REL / dt)), int(round(T_ATT / dt))
SCAL = ('W', 'diss', 'react', 'e', 'kmin', 'kmax', 'Av', 'res', 'dlam', 'lam', 'lower', 'balance', 'effort', 'effort_ref',
        'p_sum', 'p_max', 'W_modules_sum', 'sfit', 'internal', 'internal_ratio', 'flow', 'flow_rest', 'wperp_K', 'wreq_K',
        'rho_fit', 'dlam_force', 'dlam_moment')
for k in range(nsteps + 1):
    if args.events and k in (k_rel, k_att):
        before = law(t, q, u, key, full=True)
        nrm = lambda x: float(np.sqrt(x @ before['M'] @ x))
        if k == k_rel:                                      # release: nothing jumps in the plant
            key = 'R'
            u = before['v'][indR]
            after = law(t, q, u, key, full=True)
            P = m_projector(after['M'], after['E'])
            kind = 'release'
        else:                                               # latch at the present relative pose: plastic attach
            follow(q)
            key = 'G'
            Eg = model.lift(q, G).E
            P = m_projector(before['M'], Eg)
            vplus = P @ before['v']
            u = vplus[indG]
            after = law(t, q, u, key, full=True)
            kind = 'attach'
        dr = before['vr'] - after['vr']
        predicted = .5 * nrm(P @ (before['s'] + dr)) ** 2
        events.append(dict(kind=kind, t=t, W_before=before['W'], W_after=after['W'], W_predicted=predicted,
                           reset_defect=float(np.max(np.abs(after['s'] - P @ (before['s'] + dr))) / max(np.max(np.abs(after['s'])), 1e-300)),
                           reference_jump_M=nrm(P @ dr), carnot_loss=.5 * nrm(before['v'] - after['v']) ** 2,
                           error_lost=.5 * nrm(before['s'] + dr - P @ (before['s'] + dr)) ** 2,
                           kmin_before=before['kmin'], kmin_after=after['kmin']))
        jumps += after['W'] - before['W']
        r = {x: before[x] for x in SCAL}
        r.update(t=t, I_diss=float(I[0]), I_react=float(I[1]), I_dfit=float(I[2]), I_rfit=float(I[3]), jumps=jumps - (after['W'] - before['W']), seg=len(events) - 1)
        rec.append(r)
    if k % 5 == 0 or (args.events and k in (k_rel, k_att)):
        full = law(t, q, u, key, full=True)
        r = {x: full[x] for x in SCAL}
        r.update(t=t, I_diss=float(I[0]), I_react=float(I[1]), I_dfit=float(I[2]), I_rfit=float(I[3]), jumps=jumps, seg=len(events))
        rec.append(r)
    if k == nsteps:
        break
    k1 = law(t, q, u, key)
    k2 = law(t + .5 * dt, q + .5 * dt * k1[0], u + .5 * dt * k1[1], key)
    k3 = law(t + .5 * dt, q + .5 * dt * k2[0], u + .5 * dt * k2[1], key)
    k4 = law(t + dt - 1e-12, q + dt * k3[0], u + dt * k3[1], key)
    q = q + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
    u = u + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
    I = I + dt * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]) / 6
    t += dt
arr = {x: np.array([r[x] for r in rec]) for x in rec[0]}
tt = arr['t']
bal = arr['W'] - arr['W'][0] + arr['I_diss'] + arr['I_react'] - arr['jumps']
# two-sided envelope, segment by segment
viol_up = viol_lo = 0.
env_up, env_lo = np.ones(len(tt)), np.ones(len(tt))
for seg in np.unique(arr['seg']):
    m = np.flatnonzero(arr['seg'] == seg)
    if len(m) < 2:
        continue
    trap = lambda f: np.concatenate([[0.], np.cumsum(.5 * (f[m][1:] + f[m][:-1]) * np.diff(tt[m]))])
    ratio = np.sqrt(arr['W'][m] / arr['W'][m][0])
    env_up[m], env_lo[m] = np.exp(-trap(arr['kmin'])), np.exp(-trap(arr['kmax']))
    if not args.inconsistent:
        viol_up, viol_lo = max(viol_up, float(np.max(ratio / env_up[m] - 1.))), max(viol_lo, float(np.max(env_lo[m] / ratio - 1.)))
summary = dict(team=args.team, T=args.T, dt=dt, d=d, amp_scale=args.amp_scale, gain_scale=args.gain_scale, inconsistent=args.inconsistent, events=events,
               W0=float(arr['W'][0]), W_end=float(arr['W'][-1]), balance_defect_rel=float(np.max(np.abs(bal)) / arr['W'][0]),
               envelope_upper_violation=viol_up, envelope_lower_violation=viol_lo,
               kmin=[float(arr['kmin'].min()), float(arr['kmin'].max())], kmax=[float(arr['kmax'].min()), float(arr['kmax'].max())],
               e0=float(arr['e'][0]), e_end=float(arr['e'][-1]), dlam0=float(arr['dlam'][0]), dlam_end=float(arr['dlam'][-1]),
               dlam_max_late=float(arr['dlam'][tt > .5 * args.T].max()),
               lam_scale=float(arr['lam'].max()), max_Av=float(arr['Av'].max()), max_residual=float(arr['res'].max()),
               max_balance=float(arr['balance'].max()), max_effort_ratio=float(arr['effort'].max()),
               max_reference_effort_ratio=float(arr['effort_ref'].max()),
               module_power_sum_max=float(np.max(np.abs(arr['p_sum']))), module_power_max=float(arr['p_max'].max()),
               module_storage_sum_defect=float(np.max(np.abs(arr['W_modules_sum'] - arr['W']) / arr['W'][0])),
               sM_late=[float(np.sqrt(2 * arr['W'][tt > .5 * args.T]).min()), float(np.sqrt(2 * arr['W'][tt > .5 * args.T]).max())],
               lower_late=[float(arr['lower'][tt > .5 * args.T].min()), float(arr['lower'][tt > .5 * args.T].max())],
               min_sM_over_lower=float(np.min(np.sqrt(2 * arr['W'][1:]) / np.maximum(arr['lower'][1:], 1e-300))) if args.inconsistent else None,
               dlam_force_late=[float(arr['dlam_force'][tt > .5 * args.T].min()), float(arr['dlam_force'][tt > .5 * args.T].max())],
               dlam_moment_late=[float(arr['dlam_moment'][tt > .5 * args.T].min()), float(arr['dlam_moment'][tt > .5 * args.T].max())],
               fit_balance_defect_rel=float(np.max(np.abs(.5 * arr['sfit'] ** 2 - .5 * arr['sfit'][0] ** 2 + arr['I_dfit'] + arr['I_rfit'])) / (.5 * arr['sfit'][0] ** 2)) if args.inconsistent else None,
               rho_fit_late=[float(arr['rho_fit'][tt > .5 * args.T].min()), float(arr['rho_fit'][tt > .5 * args.T].max())] if args.inconsistent else None,
               sfit0=float(arr['sfit'][0]), sfit_late=[float(arr['sfit'][tt > .5 * args.T].min()), float(arr['sfit'][tt > .5 * args.T].max())],
               internal_max=float(arr['internal'].max()), internal_late=float(arr['internal'][tt > .5 * args.T].max()),
               internal_ratio_max=float(arr['internal_ratio'].max()),
               flow_late=float(arr['flow'][tt > .5 * args.T].max()), flow_rest_late=float(arr['flow_rest'][tt > .5 * args.T].max()),
               wperp_share_max=float(np.max(arr['wperp_K'] / np.maximum(arr['wreq_K'], 1e-300))),
               seconds=round(time.time() - t0, 1))
print(json.dumps(summary))
tag = args.tag or ('events' if args.events else 'inconsistent' if args.inconsistent else 'consistent')
np.savez(RESULTS / f'sim_{args.team}_{tag}.npz', **arr, env_up=env_up, env_lo=env_lo)
(RESULTS / f'sim_{args.team}_{tag}.json').write_text(json.dumps(summary, indent=1))
