"""Compositional gain synthesis on the compiled teams (Theorem 16 of the paper).

Decision: rate gains k_j of the arm and lift joint loops (wheel speed gains kept at their mission value), within
[LO, HI] times the mission gains.  Constraints: certified velocity-error rate >= kappa at every configuration of a
set (the 24 snapshots of the recorded navigation phase).

  centralized  : E_s^T (D(K) - kappa M_s) E_s >= 0                         (one LMI of order d per snapshot)
  distributed  : per robot i and snapshot s, its local LMI in (k_i, Theta_is) on its own admissible space,
                 coupled only by  sum_i Theta_is >= kappa M_0s                 (payload-sized coupling)
  ADMM         : the distributed problem solved robot by robot (sharing form); each robot solves its own small SDP and
                 exchanges one symmetric 6x6 supply per snapshot with a coordinator.
Problems: P1  minimum normalized gain sum_j k_j / k_j^mis at the rate certified by the mission gains;
          P2  largest certified rate at the mission's normalized gain budget.
"""
from __future__ import annotations

import json
import sys
import time

import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm

from teamdata import RESULTS, TEAMS, Star, mission_gains

LO, HI = 0.1, 2.0
SOLVER = dict(solver='CLARABEL')


def sym(X):
    return .5 * (X + X.T)


class Design:
    def __init__(self, name, qs):
        self.name = name
        self.K0, Kp, jp, wp = mission_gains(name)
        self.stars = [Star(name, q=q) for q in qs]
        st = self.stars[0]
        self.robots = st.robots
        self.jports = {r: np.asarray(jp[r], int) for r in self.robots}        # global port indices of joint loops
        self.all_j = np.concatenate([self.jports[r] for r in self.robots])
        self.kmis = self.K0[self.all_j]
        self.nJ = len(self.all_j)
        self.Kfixed = self.K0.copy()
        self.Kfixed[self.all_j] = 0.
        self.kappa_mis = min(s.rate(self.K0) for s in self.stars)
        self._central_data()
        self._local_data()

    # ---------------------------------------------------------------- centralized LMIs, M_r-normalized
    def _central_data(self):
        self.cen = []
        for st in self.stars:
            E = st.E
            Mr = E.T @ st.M @ E
            W = np.linalg.inv(np.linalg.cholesky(Mr)).T                          # W^T Mr W = I
            C0 = W.T @ E.T @ st.D(self.Kfixed) @ E @ W                             # fixed part: wheels and joint damping
            G = [W.T @ E.T @ np.outer(st.B[:, j], st.B[:, j]) @ E @ W for j in self.all_j]
            self.cen.append(dict(C0=sym(C0), G=[sym(g) for g in G], d=E.shape[1]))

    # ---------------------------------------------------------------- local LMIs, metric-normalized
    def _local_data(self):
        self.loc = {r: [] for r in self.robots}
        self.S0 = []
        for st in self.stars:
            S0 = np.real(sqrtm(st.M0))
            self.S0.append(S0)
            for r in self.robots:
                l = st.local[r]
                n0, ni = 6, len(l['coords'])
                Mfull = np.zeros((n0 + ni,) * 2)
                Mfull[:n0, :n0] = st.M0
                Mfull[n0:, n0:] = l['M']
                T = l['T']
                Tm = T @ np.linalg.inv(np.linalg.cholesky(T.T @ Mfull @ T)).T      # M-orthonormal basis of T_i
                Tp, Ti = Tm[:n0], Tm[n0:]
                Dfix = l['B'] @ np.diag(self.Kfixed[l['ports']]) @ l['B'].T + l['R']
                C0 = Ti.T @ Dfix @ Ti
                Mi = Ti.T @ l['M'] @ Ti
                own = {g: k for k, g in enumerate(l['ports'])}
                G = [Ti.T @ np.outer(l['B'][:, own[j]], l['B'][:, own[j]]) @ Ti for j in self.jports[r]]
                P = Tp.T @ S0                                                       # Theta_i = S0 Th S0  ->  P Th P^T
                self.loc[r].append(dict(C0=sym(C0), Mi=sym(Mi), G=[sym(g) for g in G], P=P, m=Tm.shape[1]))

    # ---------------------------------------------------------------- problems
    def _cost(self, k):
        return cp.sum(cp.multiply(1. / self.kmis, k))

    def centralized(self, problem, kappa=None):
        # decision variables: gain ratios r = k / k_mis (well scaled; the LMIs use k_mis_j G_j)
        r = cp.Variable(self.nJ)
        kap = cp.Variable() if problem == 'P2' else kappa
        cons = [r >= LO, r <= HI]
        for c in self.cen:
            cons.append(c['C0'] + sum(r[j] * (self.kmis[j] * c['G'][j]) for j in range(self.nJ)) - kap * np.eye(c['d']) >> 0)
        if problem == 'P1':
            prob = cp.Problem(cp.Minimize(cp.sum(r)), cons)
        else:
            cons.append(cp.sum(r) <= self.nJ)
            prob = cp.Problem(cp.Maximize(kap), cons)
        prob.solve(**SOLVER)
        return dict(status=prob.status, value=float(prob.value), k=self.kmis * np.asarray(r.value).ravel(),
                    kappa=float(kap.value) if problem == 'P2' else float(kappa))

    def distributed_joint(self, problem, kappa=None):
        """The distributed formulation solved as one SDP (to compare optimal values with the centralized one)."""
        kap = cp.Variable() if problem == 'P2' else kappa
        cons, cost, kvars = [], 0, {}
        off = 0
        Th = {r: [cp.Variable((6, 6), symmetric=True) for _ in self.stars] for r in self.robots}
        for r in self.robots:
            nr = len(self.jports[r])
            rr = cp.Variable(nr)                                   # gain ratios k / k_mis of robot r
            km = self.kmis[off:off + nr]
            kvars[r] = (rr, km)
            off += nr
            cons += [rr >= LO, rr <= HI]
            cost = cost + cp.sum(rr)
            for s, L in enumerate(self.loc[r]):
                lmi = -L['P'] @ Th[r][s] @ L['P'].T + L['C0'] + sum(rr[j] * (km[j] * L['G'][j]) for j in range(nr)) - kap * L['Mi']
                cons.append(sym_expr(lmi) >> 0)
        for s in range(len(self.stars)):
            cons.append(sum(Th[r][s] for r in self.robots) - kap * np.eye(6) >> 0)
        if problem == 'P1':
            prob = cp.Problem(cp.Minimize(cost), cons)
        else:
            cons.append(cost <= self.nJ)
            prob = cp.Problem(cp.Maximize(kap), cons)
        prob.solve(**SOLVER)
        k = np.concatenate([kvars[r][1] * np.asarray(kvars[r][0].value).ravel() for r in self.robots])
        return dict(status=prob.status, value=float(prob.value), k=k, kappa=float(kap.value) if problem == 'P2' else float(kappa))

    def certified(self, k):
        K = self.Kfixed.copy()
        K[self.all_j] = k
        return min(st.rate(K) for st in self.stars)

    # ---------------------------------------------------------------- ADMM (sharing form)
    def admm(self, kappa, rho=1., iters=2000, tol=1e-5, log_every=50, alpha=1.0, balance=True, init='equal'):
        S = len(self.stars)
        N = len(self.robots)
        nv = S * 21
        iu = np.triu_indices(6)
        wvec = np.where(iu[0] == iu[1], 1., np.sqrt(2.))                          # isometry sym(6) -> R^21

        def vec(Ms):
            return np.concatenate([M[iu] * wvec for M in Ms])

        def mat(x):
            out = []
            for s in range(S):
                v = x[21 * s:21 * (s + 1)] / wvec
                M = np.zeros((6, 6))
                M[iu] = v
                out.append(M + np.triu(M, 1).T)
            return out

        # local problems: robot r solves  min c^T k_r + rho/2 |x_r - v_r|^2  s.t. its local LMIs
        local = {}
        off = 0
        for r in self.robots:
            nr = len(self.jports[r])
            km = self.kmis[off:off + nr]
            off += nr
            kr = cp.Variable(nr)
            Th = [cp.Variable((6, 6), symmetric=True) for _ in range(S)]
            xv = cp.hstack([cp.hstack([Th[s][i, j] * wvec[t] for t, (i, j) in enumerate(zip(*iu))]) for s in range(S)])
            vpar = cp.Parameter(nv)
            rpar = cp.Parameter(nonneg=True)
            cons = [kr >= LO * km, kr <= HI * km]
            for s, L in enumerate(self.loc[r]):
                lmi = -L['P'] @ Th[s] @ L['P'].T + L['C0'] + sum(kr[j] * L['G'][j] for j in range(nr)) - kappa * L['Mi']
                cons.append(sym_expr(lmi) >> 0)
            obj = cp.sum(cp.multiply(1. / km, kr)) + rpar / 2 * cp.sum_squares(xv - vpar)
            local[r] = dict(prob=cp.Problem(cp.Minimize(obj), cons), k=kr, x=xv, v=vpar, rho=rpar, km=km)
        target = vec([kappa * np.eye(6) / N] * S)
        if init == 'equal':
            x = {r: target.copy() for r in self.robots}
            zbar, u, xbar = target.copy(), np.zeros(nv), target.copy()
        else:
            x = {r: np.zeros(nv) for r in self.robots}
            zbar, u, xbar = np.zeros(nv), np.zeros(nv), np.zeros(nv)
        hist = []
        t0 = time.time()
        for it in range(iters):
            for r in self.robots:
                L = local[r]
                L['v'].value = x[r] - xbar + zbar - u
                L['rho'].value = rho
                L['prob'].solve(**SOLVER, warm_start=True)
                x[r] = np.asarray(L['x'].value).ravel()
            xbar_new = sum(x.values()) / N
            xhat = alpha * xbar_new + (1. - alpha) * zbar                         # over-relaxation
            w = mat(u + xhat - target)
            zproj = []
            for Wm in w:
                ev, V = np.linalg.eigh(sym(Wm))
                zproj.append((V * np.maximum(ev, 0.)) @ V.T)
            z_new = vec(zproj) + target
            u = u + xhat - z_new
            # residuals of the sharing form (z_i = x_i + zbar - xbar)
            r_pri = np.sqrt(N) * np.linalg.norm(xbar_new - z_new)
            zi_new = {r: x[r] + z_new - xbar_new for r in self.robots}
            if it == 0:
                zi_old = {r: np.zeros(nv) for r in self.robots}
            r_dual = rho * np.sqrt(sum(np.linalg.norm(zi_new[r] - zi_old[r]) ** 2 for r in self.robots))
            zi_old = zi_new
            zbar, xbar = z_new, xbar_new
            cost = sum(float(np.sum(np.asarray(local[r]['k'].value).ravel() / local[r]['km'])) for r in self.robots)
            hist.append(dict(it=it + 1, cost=cost, r_pri=float(r_pri), r_dual=float(r_dual), rho=rho))
            if log_every and (it + 1) % log_every == 0:
                print(f'    admm {self.name} it {it + 1}: cost {cost:.6f} r_pri {r_pri:.2e} r_dual {r_dual:.2e} rho {rho:g} '
                      f'({time.time() - t0:.0f} s)', flush=True)
            eps_pri = tol * max(1., np.sqrt(N) * np.linalg.norm(zbar))
            eps_dual = tol * max(1., rho * np.sqrt(N) * np.linalg.norm(u))
            if it >= 10 and r_pri < eps_pri and r_dual < eps_dual:
                break
            # residual balancing
            if not balance:
                continue
            if r_pri > 10 * r_dual:
                rho *= 2.
                u /= 2.
            elif r_dual > 10 * r_pri:
                rho /= 2.
                u *= 2.
        k = np.concatenate([np.asarray(local[r]['k'].value).ravel() for r in self.robots])
        return dict(k=k, cost=hist[-1]['cost'], iterations=len(hist), history=hist, seconds=time.time() - t0,
                    numbers_per_robot_per_iteration=nv)


def sym_expr(X):
    return .5 * (X + X.T)


def run(name, qs, do_admm=True):
    t0 = time.time()
    des = Design(name, qs)
    out = dict(team=name, snapshots=len(qs), joint_ports=des.nJ, kappa_mission=des.kappa_mis,
               mission_cost=float(des.nJ))
    # P1
    c1 = des.centralized('P1', kappa=des.kappa_mis)
    d1 = des.distributed_joint('P1', kappa=des.kappa_mis)
    out['P1'] = dict(central=dict(status=c1['status'], cost=c1['value'], certified=des.certified(c1['k']),
                                  gain_ratio=(c1['k'] / des.kmis).tolist()),
                     distributed=dict(status=d1['status'], cost=d1['value'], certified=des.certified(d1['k'])))
    # P2
    c2 = des.centralized('P2')
    d2 = des.distributed_joint('P2')
    out['P2'] = dict(central=dict(status=c2['status'], kappa=c2['kappa'], certified=des.certified(c2['k']),
                                  gain_ratio=(c2['k'] / des.kmis).tolist()),
                     distributed=dict(status=d2['status'], kappa=d2['kappa'], certified=des.certified(d2['k'])))
    print(f"  {name}: kappa_mis={des.kappa_mis:.4f}  P1 central {c1['value']:.6f} distributed {d1['value']:.6f} "
          f"(of {des.nJ})  P2 central {c2['kappa']:.4f} distributed {d2['kappa']:.4f}  [{time.time() - t0:.0f} s]", flush=True)
    if do_admm:
        a = des.admm(des.kappa_mis)
        out['P1']['admm'] = dict(cost=a['cost'], iterations=a['iterations'], seconds=a['seconds'],
                                 certified=des.certified(a['k']),
                                 numbers_per_robot_per_iteration=a['numbers_per_robot_per_iteration'],
                                 history=a['history'])
        print(f"  {name}: ADMM cost {a['cost']:.6f} after {a['iterations']} iterations, certified "
              f"{out['P1']['admm']['certified']:.4f} [{a['seconds']:.0f} s]", flush=True)
    return out


if __name__ == '__main__':
    teams = sys.argv[1:] or list(TEAMS)
    results = {}
    for name in teams:
        qs = np.load(RESULTS / f'snapshots_{name}.npy')
        results[name] = run(name, qs)
        (RESULTS / f'synth_{name}.json').write_text(json.dumps(results[name], indent=1))
