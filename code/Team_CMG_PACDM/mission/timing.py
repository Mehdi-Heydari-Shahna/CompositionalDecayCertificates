"""Time laws: rest-to-rest profiles and a reactive speed governor for a path parameter."""
from __future__ import annotations

import numpy as np


def quintic(s):
    """Rest-to-rest blend on [0, 1]: value, first and second derivative with respect to ``s``."""
    s = float(np.clip(s, 0., 1.))
    return (s * s * s * (10. + s * (-15. + 6. * s)), 30. * s * s * (1. + s * (-2. + s)),
            60. * s * (1. + s * (-3. + 2. * s)))


def blend(t, duration):
    """Rest-to-rest blend in time: value in [0, 1], rate (1/s) and acceleration (1/s^2)."""
    if duration <= 0.:
        return 1., 0., 0.
    p, dp, ddp = quintic(t / duration)
    return p, dp / duration, ddp / duration ** 2


class Governor:
    """Speed governor of a path parameter ``l`` (m of generalized path length).

    Two accelerations compete and the smaller one is applied: one tracks the local
    speed limit, the other is a critically damped approach to the next stop point
    (``w^2 gap - 2 w rate``), which is inactive far from it and brings rate and
    acceleration to zero together at it.  Stop points are the end of the path and
    any closed gate ahead; a gate that opens while the team waits releases it.  The
    applied acceleration is bounded and passed through a short first-order lag, so
    the references handed to the bases and arms have no jumps.
    """

    def __init__(self, length, speed_limit, accel=.25, decel=.25, gain=2.5, approach=1.1, lag=.12):
        self.length = float(length)
        self.speed_limit = speed_limit            # callable l -> m/s
        self.accel, self.decel, self.gain, self.approach, self.lag = accel, decel, gain, approach, lag
        self.l, self.rate, self.acc = 0., 0., 0.
        self.waiting = None
        self.overrun = 0.

    def step(self, dt, stops=(), scale=1.):
        """Advance by ``dt``.  ``stops`` are path parameters that must not be passed now."""
        stop = min(list(stops) + [self.length])
        gap = max(stop - self.l, 0.)
        w = self.approach
        wanted = min(self.gain * (self.speed_limit(self.l) * scale - self.rate), w * w * gap - 2. * w * self.rate)
        wanted = float(np.clip(wanted, -1.6 * self.decel, self.accel))
        acc = self.acc + (wanted - self.acc) * min(dt / self.lag, 1.)
        rate = max(self.rate + acc * dt, 0.)
        if stop >= self.length and rate * dt > gap:
            rate = gap / dt
        self.acc = (rate - self.rate) / dt
        self.rate = rate
        self.l = min(self.l + rate * dt, self.length)
        # a stop that closes too late is passed with the largest braking; the overrun is reported
        self.overrun = max(self.overrun, self.l - stop) if stop < self.length else self.overrun
        self.waiting = stop if (stop < self.length - 1e-9 and stop - self.l < 4e-3 and rate < 4e-3) else None
        return self.l, self.rate, self.acc

    @property
    def done(self):
        return self.l >= self.length - 2e-3 and self.rate < 3e-3
