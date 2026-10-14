"""The fitted reference of joint-level loops and the effort it requires.

Joint and wheel loops ask every actuator for a rate.  With the gains ``K`` of their
rate feedback they apply

    f = f_ff - K (a_dot - w),        a_dot = B^T v  (the actuator rates),

where ``w`` holds the rates the loops ask for: the rate of a joint reference plus
its position feedback divided by the rate gain, and the reference rate of a wheel.
In a mode of the compiled team the actuators cannot follow arbitrary ``w``: their
rates are ``a_dot = L u`` with ``L = B^T E`` for the independent rates ``u`` of the
mode.  The *fitted reference* is the admissible motion closest to what the loops
ask for, in the metric of their own gains,

    u_r = (L^T K L)^-1 L^T K w,      v_r = E u_r,      w = L u_r + w_perp.

The loops are then the feedback ``-K B^T (v - v_r)`` of the velocity error with
respect to the fitted reference, plus the effort ``K w_perp``, which moves nothing
(``L^T K w_perp = 0``): it only loads the connections.

The effort the fitted reference requires is the solution of

    M v_r' + C(q, v) v_r + g + R v_r = B f_r + A^T lam_r                       (*)

with ``v_r' = E u_r' + E' u_r`` taken along the measured motion.  With this effort
as feedforward the error ``s = v - v_r`` obeys

    d/dt (1/2 s^T M s) = - s^T (B K B^T + R) s

on the compiled model: it decays between the two rates that the gains and the
inertia give in the mode, for any positive gains.  The inverse dynamics of the
desired motion is a different effort; with it the same loops are not covered by
this statement (DESIGN.md, "The feedforward of the fitted reference").

This module evaluates the fitted reference, its rate along the motion, the left
side of (*) and the quantities of the statement.  It needs what the compiled
interfaces provide (rows, tangent lift, tree dynamics) and nothing else.

Requirements: every wheel row of the mode is an ideal constraint (rolling wheels)
and ``L`` has full column rank (with every actuator locked nothing moves).  The
second is tested at every evaluation: the factorization of ``L^T K L`` fails
otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh, lu_solve


def tangent_rate(model, q, v, mode, lift, h0=1e-5):
    """``E'``: the rate of the tangent lift along the motion ``v`` (``n x d``).

    ``d/dt (E u) = E u' + E' u``.  The rows of the dependent coordinates follow from the rate of
    the constraint rows, taken by a centred difference along ``v`` as in ``TeamModel.curvature``;
    the rows of the independent coordinates are zero.
    """
    n, d = lift.E.shape
    out = np.zeros((n, d))
    dep, ind = lift.dependent, lift.independent
    nc = lift.constraints.Jc.shape[1]
    if lift.lu is None or (not len(dep) and not nc):
        return out
    h = h0 / max(1., float(np.max(np.abs(v))))
    cp = model.constraints(q + h * v, mode)
    cm = model.constraints(q - h * v, mode)
    rows = lift.rows
    G = ((cp.J - cm.J) / (2. * h))[rows] @ lift.E
    if nc:
        X = -lu_solve(lift.lu, lift.constraints.J[rows][:, ind])     # rates of the dependent and chart coordinates
        G += ((cp.Jc - cm.Jc) / (2. * h))[rows] @ X[len(dep):]
    out[dep] = -lu_solve(lift.lu, G)[:len(dep)]
    return out


class FittedReference:
    """Fitted reference of actuator-rate requests in a mode and the effort it requires."""

    def __init__(self, model):
        self.model, self.tree = model, model.tree
        self.n, self.nf = model.n, model.nf
        self.B = model.B
        if not np.all(np.isin(self.B, (0., 1.))) or np.any(self.B.sum(axis=0) != 1.):
            raise ValueError('the fitted reference is written for actuator ports that act on one coordinate each with gear 1')
        self.coordinate = np.array([p['index'] for p in model.ports], int)       # coordinate of every port
        self.damping = model.damping

    def fit(self, q, mode, K, w, kin=None, lift=None):
        """Fitted reference of the requested actuator rates ``w`` for the rate gains ``K`` (both per port)."""
        kin = self.tree.sweep(q) if kin is None else kin
        lift = self.model.lift(q, mode, kin) if lift is None else lift
        E = lift.E
        L = E[self.coordinate]                                     # B^T E for ports of gear 1
        KL = K[:, None] * L
        H = L.T @ KL
        try:
            cho = cho_factor(H)
        except np.linalg.LinAlgError as err:
            raise np.linalg.LinAlgError(f'{mode.label()}: the actuators do not hold every motion of this mode '
                                        '(L = B^T E has no full column rank); no fitted reference') from err
        ur = cho_solve(cho, KL.T @ w)
        return dict(kin=kin, lift=lift, E=E, L=L, KL=KL, H=H, cho=cho, ur=ur, vr=E @ ur, perp=w - L @ ur)

    def required(self, f, q, v, K, wdot):
        """Rate of the fitted reference along the motion ``v`` and the left side of (*) without the damping term.

        ``f`` is the result of :meth:`fit` with ``f['Ed']`` set to the rate of the tangent lift along ``v``
        (:func:`tangent_rate`).  ``wdot`` is the rate of the requested rates along the motion: their explicit
        rate plus what the measured rates change in them.  Adds ``urd`` (``u_r'``), ``ar`` (``v_r'``), ``M``,
        ``gravity`` and ``effort = M v_r' + C(q, v) v_r + g`` to ``f`` and returns it.
        """
        tree, kin = self.tree, f['kin']
        E, Ed, ur, vr = f['E'], f['Ed'], f['ur'], f['vr']
        Ld = Ed[self.coordinate]
        urd = cho_solve(f['cho'], Ld.T @ (K * f['perp']) + f['KL'].T @ (wdot - Ld @ ur))
        ar = E @ urd + Ed @ ur
        M = tree.mass(q, kinematics=kin)
        zero = np.zeros(self.n)
        gravity = tree.rnea(q, zero, zero, kinematics=kin)
        # C(q, v) v_r: the velocity-product effort is quadratic in the rates and the coordinates are scalar joint
        # coordinates (v = q'), so its polar form is the Christoffel product, symmetric in its two rates
        product = .25 * (tree.rnea(q, v + vr, zero, gravity=False, kinematics=kin)
                         - tree.rnea(q, v - vr, zero, gravity=False, kinematics=kin))
        f.update(urd=urd, ar=ar, M=M, gravity=gravity, effort=M @ ar + product + gravity)
        return f

    def rate_feedback(self, f):
        """``Gamma`` with ``L^T Gamma = M_r (L^T K L)^-1 L^T K``: the actuator efforts that give the reduced
        acceleration a change of the requested rates' rate asks for (symmetric, positive semidefinite).

        The effort of the fitted reference contains ``M E (L^T K L)^-1 L^T K w'``, and ``w'`` contains
        the measured joint rates through the position feedback of the loops.  Between two evaluations
        of (*) this part follows the measured rates through ``Gamma``.
        """
        Y = cho_solve(f['cho'], f['KL'].T).T                        # K L (L^T K L)^-1
        Mr = f['E'].T @ f['M'] @ f['E']
        return Y @ Mr @ Y.T

    def quantities(self, f, v, K):
        """Quantities of the statement at this instant (``f`` from :meth:`required` or :meth:`fit` with ``M`` added).

        ``W = 1/2 |v - v_r|_M^2``; ``dissipation = s^T (B K B^T + R) s``; ``rate`` the smallest and the largest
        generalized eigenvalue of ``(E^T (B K B^T + R) E, E^T M E)``: on the compiled model
        ``-2 rate[1] W <= W' <= -2 rate[0] W``; ``W_admissible`` the same energy for the measured independent
        rates alone (it leaves out what the plant moves against its ideal constraints); ``internal`` the
        effort ``K w_perp`` of the loops that moves nothing.
        """
        E, M = f['E'], f['M']
        s = v - f['vr']
        Ms = M @ s
        Mr = E.T @ M @ E
        Dr = f['H'] + E.T @ (self.damping[:, None] * E)
        rate = eigh(.5 * (Dr + Dr.T), .5 * (Mr + Mr.T), eigvals_only=True)
        sigma = v[f['lift'].independent] - f['ur']
        a = s[self.coordinate]
        return dict(W=.5 * float(s @ Ms), W_admissible=.5 * float(sigma @ Mr @ sigma),
                    dissipation=float(a @ (K * a) + s @ (self.damping * s)),
                    rate=(float(rate[0]), float(rate[-1])), internal=K * f['perp'], d=int(E.shape[1]))
