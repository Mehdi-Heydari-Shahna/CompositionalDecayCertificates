"""Proposition 23 in the time domain: a team on friction-driven wheels (Husky bases), certified law, friction at the actual loads.

Plant (compiled model of the package):  M v' + C v + g + D v = B tau + A^T lam + Q(v; N),  A(q) v = 0
   Q: smoothed Coulomb friction of the wheels (law and data of the package), N: wheel loads that are consistent with
   the ground reaction of this very state (equal-stiffness rule of the package), solved exactly at every evaluation.
Reference:  u_r = (wheel rates w_r(t)) + Z(q) xi_r,  Z: planar rigid motions of the formation (= ker L)
   xi_r is a state of the controller: it integrates the virtual balance along Z with the friction of the REFERENCE
   slip at the ACTUAL loads N(t) and the actual configuration.
Law:  tau = tau_r - K B^T (v - v_r).
Checked:  W(t) = 1/2 |v - v_r|_M^2 never increases;
          W(t) - W(0) + int s^T (B K B^T + D) s + int <s, Q(v_r; N) - Q(v; N)> = 0,  both integrands >= 0.
"""
import json
import time

import numpy as np
from scipy.integrate import solve_ivp

from lib import RESULTS, Team
from teamc.stability import wheel_load_map

import argparse
ap = argparse.ArgumentParser()
ap.add_argument('team', nargs='?', default='husky2')
ap.add_argument('--T', type=float, default=1.)
ap.add_argument('--scale', type=float, default=1., help='size of the initial error')
ap.add_argument('--mu', type=float, default=None, help='floor friction of this run (default: the data of the team)')
ap.add_argument('--turn', type=float, default=0., help='yaw rate (rad/s) for which the wheels of every base are driven '
                'while the formation is asked to drive straight: the wheels then slide all the time')
ap.add_argument('--scrub', type=float, default=0., help='surface-speed offset (m/s): the front wheels of every base are driven '
                'faster and the rear wheels slower by this amount (the offset grows with the smooth start): the pair of '
                'wheels with the smaller load then slides while the forces of the two pairs on the base cancel')
ap.add_argument('--tag', default='')
args = ap.parse_args()
name, TEND, SCALE, MU = args.team, args.T, args.scale, args.mu
T = Team(name)
model, comp = T.model, T.comp
if MU is not None:
    for rid in model._tyre:                                 # in memory only; no file of the package changes
        model._tyre[rid]['mu'] = np.full(len(model._tyre[rid]['mu']), MU)
mode = T.mode()
lift0 = model.lift(T.q0, mode)
ind = lift0.independent
pos = {i: k for k, i in enumerate(ind)}
d, n = len(ind), T.n
rng = np.random.default_rng(5)
L0 = T.B.T @ lift0.E
Mr0 = lift0.E.T @ T.M(T.q0) @ lift0.E
seen = np.diag(np.linalg.pinv(L0).T @ Mr0 @ np.linalg.pinv(L0))
K = np.diag(10. * np.maximum(seen, 1e-6 * seen.max()))
Dm = T.B @ K @ T.B.T + T.D
bound = model.effort_bounds[:, 1]
wheel_idx = np.concatenate([[w['index'] for w in comp.robots[rid].wheels] for rid in T.robots])
Wsel = np.zeros((d, len(wheel_idx)))
for k, i in enumerate(wheel_idx):
    Wsel[pos[i], k] = 1.
groups = [comp.robots[r].base for r in T.robots] + [T.payload]
nw = {rid: len(comp.robots[rid].wheels) for rid in T.robots}


def formation(q, v=None):
    """Z (planar rigid motions of the formation in the independent coordinates) and its rate along v."""
    Z, Zd = np.zeros((d, 3)), np.zeros((d, 3))
    for base in groups:
        Z[pos[base[0]]] = [1., 0., -q[base[1]]]
        Z[pos[base[1]]] = [0., 1., q[base[0]]]
        Z[pos[base[3]]] = [0., 0., 1.]
        if v is not None:
            Zd[pos[base[0]], 2] = -v[base[1]]
            Zd[pos[base[1]], 2] = v[base[0]]
    return Z, Zd


def smooth(t, T1=.8):
    x = min(max(t / T1, 0.), 1.)
    return x ** 3 * (10 - 15 * x + 6 * x ** 2), (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / T1 if 0. < t < T1 else 0.


# wheel-rate reference: the rates at which every base would roll along a formation motion (forward and a slow turn)
kin0 = T.tree.sweep(T.q0)
centre = np.mean([T.q0[b[[0, 1]]] for b in groups[:-1]], axis=0)
heading = T.q0[groups[0][3]]
VF, WZ = .15, 0.                                           # the formation is asked to drive straight
w_unit, w_scrub = [], []
for rid in T.robots:
    base, S, Wm = model.base_motion(T.q0, rid, kin0)
    p = T.q0[comp.robots[rid].base[[0, 1]]] - centre
    planar = np.array([VF * np.cos(heading) - WZ * p[1], VF * np.sin(heading) + WZ * p[0], WZ + args.turn])
    w_unit.append(Wm @ planar)
    forward = Wm @ np.array([np.cos(T.q0[comp.robots[rid].base[3]]), np.sin(T.q0[comp.robots[rid].base[3]]), 0.])   # wheel rates per m/s of forward speed
    xs = np.array([w['centre_base'][0] for w in comp.robots[rid].wheels])
    w_scrub.append(args.scrub * np.where(xs > xs.mean(), 1., -1.) * forward)
w_unit = np.concatenate(w_unit)
w_scrub = np.concatenate(w_scrub)
FRONT = np.concatenate([(lambda xs: xs > xs.mean())(np.array([w['centre_base'][0] for w in comp.robots[rid].wheels])) for rid in T.robots])


def unit_friction(q, v, kin):
    """Generalized friction force per unit load of every wheel, at the slips of v (columns in wheel order)."""
    cols = []
    for rid in T.robots:
        slip, rows, columns = model.wheel_slip(q, v, rid, kin)
        t = model._tyre[rid]
        unit = -(t['mu'] / np.sqrt((slip * slip).sum(axis=1) + t['slip_speed'] ** 2))[:, None] * slip
        for k in range(len(slip)):
            cols.append(rows[columns[k]].T @ unit[k])
    return np.array(cols).T                                  # n x wheels


def rhs(t, X, full=False):
    q, u, xi = X[:n], X[n:n + d], X[n + d:n + d + 3]
    lift = model.lift(q, mode)
    E, con = lift.E, lift.constraints
    ideal = np.ones(con.J.shape[0], bool)
    for b in con.blocks:
        if b.get('slip'):
            ideal[b['rows']] = False
    A = con.J[ideal]
    v = E @ u
    Ed = T.Edot(q, v, mode)
    kin = T.tree.sweep(q)
    M, g = T.M(q), T.g(q)
    Mr = E.T @ M @ E
    Z, Zd = formation(q, v)
    sm, smd = smooth(t)
    ur = Wsel @ (sm * (w_unit + w_scrub)) + Z @ xi
    vr = E @ ur
    s = v - vr
    Uv, Ur = unit_friction(q, v, kin), unit_friction(q, vr, kin)
    hv = T.b(q, v) + g + T.D @ v
    base_r = M @ (Ed @ ur) + T.C(q, v, vr) + g + T.D @ vr
    BA = np.hstack([T.B * bound, A.T * 1e4])
    BAp = np.linalg.pinv(BA, rcond=1e-12)
    ZMZ = Z.T @ Mr @ Z
    ground_rows = [np.flatnonzero(ideal[:con.rows(f'{rid}:ground').stop])[-3:] for rid in T.robots]  # rows of A
    # map from ideal-row multipliers to ground multipliers per robot
    idx_ideal = np.flatnonzero(ideal)
    gmap = [np.array([int(np.flatnonzero(idx_ideal == r)[0]) for r in range(con.rows(f'{rid}:ground').start, con.rows(f'{rid}:ground').stop)])
            for rid in T.robots]
    P = [wheel_load_map(model, q, rid, kin, con.info.get(f'{rid}:normal')) for rid in T.robots]
    ATp = np.linalg.pinv(A.T, rcond=1e-12)

    def inner(N):
        Qr, Qv = Ur @ N, Uv @ N
        # virtual balance along Z: acceleration of the formation reference
        urd0 = Wsel @ (smd * (w_unit + w_scrub)) + Zd @ xi
        xid = -np.linalg.solve(ZMZ, Z.T @ (Mr @ urd0 + E.T @ (base_r - Qr)))
        urd = urd0 + Z @ xid
        ar = E @ urd + Ed @ ur
        rr = M @ (E @ urd) + base_r - Qr
        x = BAp @ rr
        tau_r = bound * x[:T.nf]
        bal = float(np.max(np.abs(BA @ x - rr)))
        tau = tau_r - K @ (T.B.T @ s)
        ud = np.linalg.solve(Mr, E.T @ (T.B @ tau - hv - M @ (Ed @ u) + Qv))
        a = E @ ud + Ed @ u
        lam = ATp @ (M @ a + hv - T.B @ tau - Qv)
        Nn = np.concatenate([P[k] @ lam[gmap[k]] for k in range(len(T.robots))])
        return Nn, dict(ud=ud, xid=xid, tau=tau, tau_r=tau_r, Qv=Qv, Qr=Qr, bal=bal, a=a, ar=ar)

    m = Uv.shape[1]
    N0 = inner(np.zeros(m))[0]
    Gm = np.column_stack([inner(np.eye(m)[:, k])[0] - N0 for k in range(m)])
    active = np.ones(m, bool)                                # a wheel whose load comes out negative has left the ground
    for _ in range(m):
        N = np.zeros(m)
        N[active] = np.linalg.solve((np.eye(m) - Gm)[np.ix_(active, active)], N0[active])
        if N.min() >= 0.:
            break
        active &= N > 0.
    Nn, o = inner(N)
    diss = float(s @ Dm @ s)
    fric = float(s @ (o['Qr'] - o['Qv']))
    dX = np.concatenate([v, o['ud'], o['xid'], [diss, fric]])
    if not full:
        return dX
    return dict(W=.5 * float(s @ M @ s), diss=diss, fric=fric, N=N, Nmin=float(N.min()), loop=float(np.max(np.abs((Nn - N)[active]))) if active.any() else float('nan'), lifted=int(m - active.sum()),
                gain=float(np.max(np.abs(np.linalg.eigvals(Gm)))), bal=o['bal'], effort=float(np.max(np.abs(o['tau']) / bound)),
                sB=float(np.max(np.abs(T.B.T @ s))), xi=xi.copy(), twist_err=float(np.linalg.norm(np.linalg.lstsq(Z, u - ur, rcond=None)[0])),
                Av=float(np.max(np.abs(A @ v))), res=float(np.max(np.abs(con.residual[ideal]))),
                slip=float(max(np.max(np.abs(model.wheel_slip(q, v, rid, kin)[0])) for rid in T.robots)),
                # slip speed of every wheel (m/s), in the wheel order of the run, and of the reference
                slip_wheels=np.concatenate([np.sqrt((model.wheel_slip(q, v, rid, kin)[0] ** 2).sum(axis=1)) for rid in T.robots]),
                slip_wheels_ref=np.concatenate([np.sqrt((model.wheel_slip(q, vr, rid, kin)[0] ** 2).sum(axis=1)) for rid in T.robots]))


Z0 = formation(T.q0)[0]
# initial errors: the formation slides and yaws, every wheel turns too fast by the same amount
u0 = Z0 @ (SCALE * np.array([.02, -.02, .02])) + Wsel @ (SCALE * .1 * np.where(np.arange(len(wheel_idx)) % 2, 1., -1.))
X0 = np.concatenate([T.q0, u0, np.zeros(3), [0., 0.]])
t0 = time.time()
first = rhs(0., X0, full=True)
print('start', json.dumps({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in first.items()}), flush=True)
sol = solve_ivp(rhs, (0., TEND), X0, method='BDF', rtol=1e-8, atol=1e-11, dense_output=True, max_step=.02)
print('integrated', sol.status, sol.message, 'steps', len(sol.t), 'rhs', sol.nfev, 'jac', sol.njev, round(time.time() - t0, 1), 's', flush=True)
TEND = float(sol.t[-1]) if sol.status != 0 else TEND
tt = np.linspace(0., TEND, int(round(TEND / .01)) + 1)
rec = []
for t in tt:
    X = sol.sol(t)
    r = rhs(t, X, full=True)
    r.update(t=float(t), I_diss=float(X[-2]), I_fric=float(X[-1]))
    rec.append(r)
arr = {k: np.array([r[k] for r in rec]) for k in rec[0]}
bal = arr['W'] - arr['W'][0] + arr['I_diss'] + arr['I_fric']
summary = dict(team=name, T=TEND, d=d, mu=[float(x) for x in model._tyre[T.robots[0]]['mu']], slip_speed=float(model._tyre[T.robots[0]]['slip_speed']), solver_status=int(sol.status), steps=int(len(sol.t)), rhs=int(sol.nfev), W0=float(arr['W'][0]), W_end=float(arr['W'][-1]),
               balance_defect_rel=float(np.max(np.abs(bal)) / arr['W'][0]),
               max_increase_of_W=float(np.max(np.diff(arr['W'])) / arr['W'][0]),
               min_friction_term=float(arr['fric'].min()), I_diss_end=float(arr['I_diss'][-1]), I_fric_end=float(arr['I_fric'][-1]),
               min_load_N=float(arr['Nmin'].min()), wheels_lifted_max=int(arr['lifted'].max()), load_loop_gain_max=float(arr['gain'].max()), load_loop_defect=float(arr['loop'].max()),
               max_balance=float(arr['bal'].max()), max_effort_ratio=float(arr['effort'].max()),
               actuated_error=[float(arr['sB'][0]), float(arr['sB'][-1])], twist_error=[float(arr['twist_err'][0]), float(arr['twist_err'][-1])],
               max_slip_m_s=float(arr['slip'].max()), max_Av=float(arr['Av'].max()), max_residual=float(arr['res'].max()),
               xi_end=arr['xi'][-1].tolist(), turn=args.turn, scrub=args.scrub, scale=SCALE,
               slip_late_m_s=[float(arr['slip'][arr['t'] >= .5 * TEND].min()), float(arr['slip'][arr['t'] >= .5 * TEND].max())],
               share_of_time_sliding=float(np.mean(arr['slip'] > model._tyre[T.robots[0]]['slip_speed'])),
               # rate of |s|_M (half the rate of W) fitted over the second half of the run
               W_late_rate=float(-np.polyfit(arr['t'][arr['t'] >= .5 * TEND], np.log(arr['W'][arr['t'] >= .5 * TEND]), 1)[0] / 2),
               front_wheels=[bool(x) for x in FRONT],
               slip_late_front_m_s=[float(arr['slip_wheels'][arr['t'] >= .5 * TEND][:, FRONT].min()), float(arr['slip_wheels'][arr['t'] >= .5 * TEND][:, FRONT].max())],
               slip_late_rear_m_s=[float(arr['slip_wheels'][arr['t'] >= .5 * TEND][:, ~FRONT].min()), float(arr['slip_wheels'][arr['t'] >= .5 * TEND][:, ~FRONT].max())],
               share_of_time_all_sliding=float(np.mean(arr['slip_wheels'].min(axis=1) > model._tyre[T.robots[0]]['slip_speed'])),
               share_of_time_any_sliding=float(np.mean(arr['slip_wheels'].max(axis=1) > model._tyre[T.robots[0]]['slip_speed'])),
               load_late_front_N=[float(arr['N'][arr['t'] >= .5 * TEND][:, FRONT].min()), float(arr['N'][arr['t'] >= .5 * TEND][:, FRONT].max())],
               load_late_rear_N=[float(arr['N'][arr['t'] >= .5 * TEND][:, ~FRONT].min()), float(arr['N'][arr['t'] >= .5 * TEND][:, ~FRONT].max())],
               seconds=round(time.time() - t0, 1))
print(json.dumps(summary))
tag = f'_{args.tag}' if args.tag else ''
np.savez(RESULTS / f'exp5_{name}{tag}.npz', **arr)
(RESULTS / f'exp5_{name}{tag}.json').write_text(json.dumps(summary, indent=1))
