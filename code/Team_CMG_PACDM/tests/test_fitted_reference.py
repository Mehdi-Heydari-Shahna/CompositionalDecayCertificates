"""The fitted reference of the joint and wheel loops and the effort it requires (``mission/certificate.py``).

What is tested:

* with the effort of the fitted reference as feedforward the power balance of the stability statement closes on
  the compiled model, pointwise, in the modes of a mission;
* the rates the controller hands to that effort are the derivatives they are meant to be;
* on the plant with ideal constraints the error energy of the mission controller decays between the two rates of
  the statement, and with the inverse dynamics of the desired motion it rises first (for this team and these gains);
* the default controller is the one of the earlier releases, and recording the quantities does not change a run.
"""
import numpy as np
import pytest

from mission.certificate import FittedReference, tangent_rate
from mission.controller import TeamController
from mission.runner import carry_configuration, run_mission, start_configuration
from teamc.se3 import exp, transform


def _controller(compiled, missions, name, **options):
    return TeamController(compiled(name), missions(name), .002, options=options or None)


def _modes(ctrl, mission):
    comp, model = ctrl.comp, ctrl.model
    grasps = sorted(ctrl.grasp_of.values())
    yield 'carry', carry_configuration(comp, mission, ctrl, 1.), model.mode(grasps=grasps, rest=None)
    yield 'free, payload on its supports', start_configuration(comp, mission, model), model.mode(grasps=(), rest=mission['rests']['pickup'])


def test_the_effort_of_the_fitted_reference_closes_the_power_balance(mecanum_team, compiled, missions):
    """E^T [M (v' - v_r') + C(q, v) s + (B K B^T + R) s] = 0 for the acceleration of the compiled model, so W' = -dissipation."""
    ctrl = _controller(compiled, missions, mecanum_team)
    model, tree, K = ctrl.model, ctrl.tree, ctrl.K
    ref = FittedReference(model)
    rng = np.random.default_rng(3)
    zero = np.zeros(tree.n)
    for label, q, mode in _modes(ctrl, missions(mecanum_team)):
        lift = model.lift(q, mode)
        d = lift.E.shape[1]
        assert np.linalg.matrix_rank(lift.E[ref.coordinate]) == d, label          # locked actuators hold every motion
        for _ in range(2):
            u = .2 * rng.standard_normal(d)
            v = lift.E @ u
            w, wdot = .3 * rng.standard_normal(model.nf), 2. * rng.standard_normal(model.nf)
            fit = ref.fit(q, mode, K, w, lift=lift)
            fit['Ed'] = tangent_rate(model, q, v, mode, lift)
            assert fit['Ed'] @ u == pytest.approx(model.curvature(q, u, mode, lift)[0], abs=1e-9)
            ref.required(fit, q, v, K, wdot)
            assert fit['L'].T @ (K * fit['perp']) == pytest.approx(0., abs=1e-8)   # what the loops ask beyond the fit moves nothing
            out = model.inverse(q, fit['vr'], None, mode, constraints=lift.constraints, effort=fit['effort'])
            assert out['residual'] < 1e-7                                          # the balance (*) has a solution
            f = out['f'] - K * (v[ref.coordinate] - w)                             # the loops
            a = model.forward(q, u, f, mode)['a']
            s = v - fit['vr']
            product = .25 * (tree.rnea(q, v + s, zero, gravity=False) - tree.rnea(q, v - s, zero, gravity=False))
            damping = model.B @ (K * (model.B.T @ s)) + model.damping * s
            balance = lift.E.T @ (fit['M'] @ (a - fit['ar']) + product + damping)
            assert np.max(np.abs(balance)) < 1e-9 * np.max(np.abs(lift.E.T @ damping)), label
            quantities = ref.quantities(fit, v, K)
            h = 1e-6
            Mdot = (tree.mass(q + h * v) - tree.mass(q - h * v)) / (2. * h)
            Wdot = float(s @ (fit['M'] @ (a - fit['ar']))) + .5 * float(s @ Mdot @ s)
            assert Wdot == pytest.approx(-quantities['dissipation'], rel=1e-6)
            assert -2. * quantities['rate'][1] * quantities['W'] <= Wdot <= -2. * quantities['rate'][0] * quantities['W']


def test_the_rate_of_the_fitted_reference_is_its_derivative_along_the_motion(mecanum_team, compiled, missions):
    """u_r' of FittedReference.required against a centred difference of the fit along q + t v, w + t w'."""
    ctrl = _controller(compiled, missions, mecanum_team)
    model, K = ctrl.model, ctrl.K
    ref = FittedReference(model)
    rng = np.random.default_rng(13)
    for label, q, mode in _modes(ctrl, missions(mecanum_team)):
        lift = model.lift(q, mode)
        v = lift.E @ (.2 * rng.standard_normal(lift.E.shape[1]))
        w, wdot = .3 * rng.standard_normal(model.nf), 2. * rng.standard_normal(model.nf)
        fit = ref.fit(q, mode, K, w, lift=lift)
        fit['Ed'] = tangent_rate(model, q, v, mode, lift)
        ref.required(fit, q, v, K, wdot)
        h = 1e-5
        ahead, behind = (ref.fit(q + sign * h * v, mode, K, w + sign * h * wdot) for sign in (1., -1.))
        assert fit['urd'] == pytest.approx((ahead['ur'] - behind['ur']) / (2. * h), abs=1e-6 * (1. + np.max(np.abs(fit['urd'])))), label
        assert fit['ar'] == pytest.approx((ahead['vr'] - behind['vr']) / (2. * h), abs=1e-6 * (1. + np.max(np.abs(fit['ar'])))), label


def test_the_wheel_rows_of_the_tangent_lift_are_the_map_of_the_base_motion(mecanum_team, compiled, missions):
    """The rate of a wheel reference is taken from these rows (TeamController._fitted)."""
    ctrl = _controller(compiled, missions, mecanum_team)
    for _, q, mode in _modes(ctrl, missions(mecanum_team)):
        lift = ctrl.model.lift(q, mode)
        column = {int(i): k for k, i in enumerate(lift.independent)}
        kin = ctrl.tree.sweep(q)
        for rid, ix in ctrl.ix.items():
            W3 = ctrl.model.base_motion(q, rid, kin)[2]
            rows = lift.E[np.ix_(ix['wheels'], [column[int(i)] for i in ix['planar']])]
            assert rows == pytest.approx(W3, abs=1e-10)
            other = np.delete(lift.E[ix['wheels']], [column[int(i)] for i in ix['planar']], axis=1)
            assert np.max(np.abs(other)) < 1e-10                                  # nothing else moves these wheels


def test_the_rate_of_a_closure_reference_along_the_motion(compiled, missions):
    """Payload, base and lift rates and the rate of the offset move the arm reference as the controller says."""
    name = 'team2'
    ctrl = _controller(compiled, missions, name)
    comp, mission = compiled(name), missions(name)
    q = carry_configuration(comp, mission, ctrl, 1.)
    rng = np.random.default_rng(5)
    h = 1e-4                                                      # well above what the closure solver leaves
    for rid, cid in ctrl.grasp_of.items():
        ix = ctrl.ix[rid]
        for kind in ('translation', 'twist'):
            base = .01 * rng.standard_normal(6)
            offset = (lambda x: transform(p=x[3:])) if kind == 'translation' else exp
            rate = .1 * rng.standard_normal(6) * (np.r_[0., 0., 0., 1., 1., 1.] if kind == 'translation' else 1.)
            payload, moved, lift = .1 * rng.standard_normal(6), .1 * rng.standard_normal(6), .05 * rng.standard_normal(1)

            def reference(sign):
                q_in = q.copy()
                q_in[ctrl.payload] += sign * h * payload
                q_in[ix['base']] += sign * h * moved
                q_in[ix['lift']] += sign * h * lift[0]
                return ctrl.asm.solve(q_in, [cid], offsets={cid: offset(base + sign * h * rate)})[0][ix['arm']]

            numeric = (reference(1.) - reference(-1.)) / (2. * h)
            q_out, _ = ctrl.asm.solve(q, [cid], offsets={cid: offset(base)})
            mod = ctrl.asm.modules[cid]
            N = mod['N'][mod['g'].passive[:6]]
            twist = rate if kind == 'translation' else ctrl._exp_rate(base, rate)   # of the offset frame, in its own axes
            analytic = N @ np.r_[payload, moved, lift] + N[:, :6] @ ctrl._offset_as_payload_rate(mod['g'], q, twist)
            assert analytic == pytest.approx(numeric, abs=2e-5 * np.max(np.abs(numeric)))


def test_the_requested_rates_move_at_the_rate_the_effort_is_given(compiled, missions):
    name = 'ridgeback2'
    ctrl = _controller(compiled, missions, name, feedforward='fitted')
    rng = np.random.default_rng(7)
    n = ctrl.n
    ctrl.q_ref, ctrl.v_ref, ctrl.a_ref, ctrl.r_dot = (rng.standard_normal(n) for _ in range(4))
    for rid in ctrl.robots:
        ctrl.wheel_ref[rid], ctrl.wheel_rate[rid] = rng.standard_normal(4), rng.standard_normal(4)
    q, v, h = rng.standard_normal(n), rng.standard_normal(n), 1e-6
    numeric = (ctrl._requested(q + h * v, v, .003 + h) - ctrl._requested(q - h * v, v, .003 - h)) / (2. * h)
    assert ctrl._requested_rate(v) == pytest.approx(numeric, abs=1e-7)


def test_the_requested_rates_follow_the_measured_configuration_between_two_updates(compiled, missions):
    """With ``advance='positions'`` the derivative of the requests along any motion is the rate the effort is given."""
    name = 'team2'
    ctrl = _controller(compiled, missions, name, feedforward='fitted')
    assert ctrl.opt['advance'] == 'positions'
    rng = np.random.default_rng(11)
    n = ctrl.n
    ctrl.q_ref, ctrl.v_ref, ctrl.a_ref, ctrl.r_dot, ctrl.r_acc = (rng.standard_normal(n) for _ in range(5))
    q0, v0 = rng.standard_normal(n), rng.standard_normal(n)       # state at the update
    for rid, ix in ctrl.ix.items():
        ctrl.wheel_ref[rid], ctrl.wheel_rate[rid], ctrl.wheel_acc[rid] = (rng.standard_normal(4) for _ in range(3))
        at = np.r_[ix['base'], ix['lift']]
        ctrl.arm_follow[rid] = (rng.standard_normal((6, 7)), at, q0[at].copy(), v0[at].copy())
        ctrl.wheel_follow[rid] = (rng.standard_normal((4, 3)), q0[ix['planar']].copy(), v0[ix['planar']].copy())
    lead, h = .004, 1e-6
    q, v = q0 + lead * v0 + 1e-3 * rng.standard_normal(n), rng.standard_normal(n)     # another state, other rates
    numeric = (ctrl._requested(q + h * v, v, lead + h) - ctrl._requested(q - h * v, v, lead - h)) / (2. * h)
    assert ctrl._requested_rate(v, lead) == pytest.approx(numeric, abs=1e-6)
    # at the update itself nothing is added: the requests and their rates are those of the references
    plain = _controller(compiled, missions, name, feedforward='fitted', advance='rates')
    plain.q_ref, plain.v_ref, plain.a_ref, plain.r_dot = ctrl.q_ref, ctrl.v_ref, ctrl.a_ref, ctrl.r_dot
    plain.wheel_ref, plain.wheel_rate = ctrl.wheel_ref, ctrl.wheel_rate
    assert np.array_equal(ctrl._requested(q0, v0, 0.), plain._requested(q0, v0, 0.))
    assert ctrl._requested_rate(v0) == pytest.approx(plain._requested_rate(v0), abs=1e-12)
    # along the rates of the update the two ways of advancing differ by the second order in time alone
    second = np.zeros(ctrl.model.nf)
    for rid, ix in ctrl.ix.items():
        second[ix['joint_ports']] = ctrl.rate_ratio[ix['joint_ports']] * ctrl.r_acc[ix['joints']] * .5 * lead ** 2
        second[ix['wheel_ports']] = ctrl.wheel_acc[rid] * .5 * lead ** 2
    assert ctrl._requested(q0 + lead * v0, v0, lead) - plain._requested(q0 + lead * v0, v0, lead) == pytest.approx(second, abs=1e-12)
    assert np.max(np.abs(ctrl._requested(q, v, lead) - plain._requested(q, v, lead))) > 1e-4


def test_the_rate_of_a_reference_as_its_samples_have_it(compiled, missions):
    """``_centred``: a reference stepped with the rate of the step (the speed governor) and one whose three values
    belong to one instant both give a rate with which the reference arrives at its next sample."""
    ctrl = _controller(compiled, missions, 'ridgeback2', feedforward='fitted')
    dt = ctrl.dt_slow
    # the governor: acceleration -> rate -> position with the new rate
    x, rate, acc = np.array([.3, -.2]), np.array([.10, .05]), np.array([.25, -.1])
    assert ctrl._centred('a', x, rate, acc) is rate                # no sample before this one
    for _ in range(3):
        rate = rate + acc * dt
        x = x + rate * dt
        r = ctrl._centred('a', x, rate, acc)
        assert r == pytest.approx(rate + .5 * acc * dt, abs=1e-14)
        ahead = x + r * dt + .5 * acc * dt ** 2                    # where the advanced reference is at the next update
        assert ahead == pytest.approx(x + (rate + acc * dt) * dt, abs=1e-14)     # the next sample of the governor
    # a time law: position, rate and acceleration of one instant (constant jerk here)
    law = lambda t: (np.array([.1 * t ** 3 + .2 * t ** 2]), np.array([.3 * t ** 2 + .4 * t]), np.array([.6 * t + .4]))
    ctrl._centred('b', *law(1.))
    for k in (1, 2, 3):
        x, rate, acc = law(1. + k * dt)
        r = ctrl._centred('b', x, rate, acc)
        assert r == pytest.approx(rate, abs=2e-5)                  # the rate handed over, to the second order of the interval
        assert x + r * dt + .5 * acc * dt ** 2 == pytest.approx(law(1. + (k + 1) * dt)[0], abs=1e-6)
    # a jump of the reference: the rate handed over
    x, rate, acc = law(2.)
    assert ctrl._centred('b', x + .01, rate, acc) is rate


def _stepped(missions, name, advance, kick_at=None, steps=45):
    """A carry stage on the plant with ideal constraints, stepped by hand: how far the requests advanced to the time
    of an update are from the ones that update then computes afresh (largest over the ports, per update)."""
    from backends.pin_plant import PinPlant
    from teamc.io import load_team
    comp, mission = load_team(name), missions(name)
    plant = PinPlant(comp)
    ctrl = TeamController(comp, mission, plant.dt, slow_every=5, options=dict(feedforward='fitted', advance=advance))
    plant.reset(carry_configuration(comp, mission, ctrl, 1.))
    for cid in ctrl.grasp_of.values():
        ctrl.on_latched(cid, plant.latch(cid))
    meas = plant.measure()
    meas['obstacle'] = None
    ctrl.start_carrying(meas, 1.)
    errors = []
    for k in range(steps):
        if k % 5 == 0:
            ahead = None if k == 0 else ctrl._requested(meas['q'], meas['v'], meas['t'] - ctrl.t_slow)
            ctrl.slow(meas)
            if ahead is not None:
                errors.append(np.max(np.abs(ahead - ctrl._requested(meas['q'], meas['v'], 0.))))
        plant.set_efforts(ctrl.fast(meas))
        plant.step()
        if kick_at is not None and k == kick_at:                  # an admissible velocity jump between two updates
            lift = ctrl.model.lift(plant.measure()['q'], ctrl.mode)
            plant.kick(lift.E @ (.1 * np.random.default_rng(2).standard_normal(lift.E.shape[1])))
        meas = plant.measure()
        meas['obstacle'] = None
    return np.array(errors)


@pytest.mark.parametrize('name', ['ridgeback2', 'team2'])
def test_after_a_velocity_jump_the_requests_follow_the_measured_positions(missions, name):
    """A velocity jump right after an update: advanced with the measured displacement the requests are, at the next
    update, as close to the ones computed afresh as without the jump; advanced with the rates of the update they miss
    them by what the jump moved."""
    pytest.importorskip('pinocchio')
    smooth = _stepped(missions, name, 'positions')
    jumped = _stepped(missions, name, 'positions', kick_at=30)
    stale = _stepped(missions, name, 'rates', kick_at=30)
    k = 30 // 5                                                    # index of the update that follows the jump
    assert np.array_equal(jumped[:k], smooth[:k])
    assert jumped[k] < 1.5 * smooth[k] and smooth[k] < 5e-3        # what is left is the sampling of the references
    assert stale[k] > 20. * jumped[k]


@pytest.mark.slow
def test_requests_that_arrive_at_the_next_update_leave_less_error_energy(missions):
    """A team that speeds up from rest with the payload, on the plant with ideal constraints: with the requests
    advanced to the samples of the next update (``advance='positions'``) the error energy of the loops while the team
    accelerates is a fraction of what it is when they are advanced at their rates at the update."""
    pytest.importorskip('pinocchio')
    median = {}
    for advance in ('positions', 'rates'):
        A = _carry(None, missions, 'team2', duration=3., feedforward='fitted', advance=advance)[1].arrays()
        speeding_up = (A['t'] > .5) & (A['t'] < 2.5)
        assert np.all(np.diff(A['rate'][speeding_up]) > 0.)
        median[advance] = float(np.nanmedian(A['fit:W'][speeding_up]))
    assert median['positions'] < .25 * median['rates'], median


def test_options_of_the_feedforward(compiled, missions):
    ctrl = _controller(compiled, missions, 'team2')
    assert ctrl.opt['feedforward'] == 'desired' and not ctrl.fitted and ctrl.reference is None       # as in the earlier releases
    assert ctrl.opt['velocity_filter_hz'] == 15.
    fitted = _controller(compiled, missions, 'team2', feedforward='fitted')
    assert fitted.fitted and fitted.opt['velocity_filter_hz'] == 0. and fitted.reference is not None
    assert _controller(compiled, missions, 'team2', feedforward='fitted', velocity_filter_hz=15.).opt['velocity_filter_hz'] == 15.
    assert np.all(fitted.K > 0.) and np.all(fitted.rate_ratio >= 0.)
    wheels = np.concatenate([ix['wheel_ports'] for ix in fitted.ix.values()])
    assert np.all(fitted.rate_ratio[wheels] == 0.) and np.all(np.delete(fitted.rate_ratio, wheels) > 0.)
    with pytest.raises(ValueError, match='rolling wheels'):
        _controller(compiled, missions, 'husky2', feedforward='fitted')
    with pytest.raises(ValueError, match='rolling wheels'):
        _controller(compiled, missions, 'husky2', monitor=True)
    with pytest.raises(ValueError, match='unknown controller options'):
        _controller(compiled, missions, 'team2', feedforwards='fitted')
    with pytest.raises(ValueError, match="'desired' or 'fitted'"):
        _controller(compiled, missions, 'team2', feedforward='certified')
    with pytest.raises(ValueError, match="'positions' or 'rates'"):
        _controller(compiled, missions, 'team2', advance='held')


def _carry(compiled, missions, name, duration=.4, **controller):
    from backends.pin_plant import PinPlant
    from teamc.io import load_team
    comp = load_team(name)                  # a run registers its as-latched grasp frames in the compiled team
    plant = PinPlant(comp)
    return run_mission(plant, comp, missions(name), options=dict(start='navigate', start_l=1., duration=duration, controller=controller))


def test_recording_the_quantities_does_not_change_the_run(compiled, missions):
    pytest.importorskip('pinocchio')
    plain = _carry(compiled, missions, 'ridgeback2')[1].arrays()
    watched = _carry(compiled, missions, 'ridgeback2', monitor=True)[1].arrays()
    assert not any(key.startswith('fit:') for key in plain)
    for key in ('q', 'v', 'f', 'f_peak'):
        assert np.array_equal(plain[key], watched[key]), key
    assert np.all(np.isfinite(watched['fit:W'])) and np.all(watched['fit:rate'][:, 0] > 5.)
    assert watched['fit:vr'].shape == plain['v'].shape and np.all(watched['fit:d'] == 12.)
    assert np.all(np.diff(watched['fit:dissipated']) >= 0.)


def test_a_carry_stage_with_the_fitted_effort(compiled, missions):
    pytest.importorskip('pinocchio')
    ctrl, rec, info = _carry(compiled, missions, 'team2', duration=1., feedforward='fitted')
    A = rec.arrays()
    assert info['status'] == 'running' and info['saturated_steps'] == 0
    assert np.max(np.abs(A['q'][:, ctrl.payload] - A['payload_ref'])[:, :3]) < 5e-4
    assert np.max(A['ff_residual']) < 1e-6                                          # the balance (*) had a solution throughout
    assert np.nanmax(A['fit:W']) < 1e-3 and np.nanmax(A['fit:internal']) < .01
    # the same stage with the fitted effort evaluated in every step
    again = _carry(compiled, missions, 'team2', duration=1., feedforward='fitted', feedforward_rate='fast')[1].arrays()
    assert np.max(np.abs(again['q'] - A['q'])) < 2e-5 and np.max(np.abs(again['f'] - A['f'])) < .02 * np.max(np.abs(A['f']))


def test_the_error_energy_decays_between_the_two_rates(compiled, missions, tmp_path):
    """The mission controller on the plant with ideal constraints, after a velocity jump (tools/run_decay.py)."""
    pytest.importorskip('pinocchio')
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import run_decay
    assert run_decay.main(['--team', 'team2', '--plant', 'pinocchio', '--out', str(tmp_path), '--duration', '.2']) == 0
    import json
    desired, fitted = json.loads((tmp_path / 'decay.json').read_text())
    assert desired['feedforward'] == 'desired' and fitted['feedforward'] == 'fitted'
    assert fitted['start']['form_min'] < 0. < fitted['start']['rate'][0]            # the form of the loops is indefinite for this team
    assert fitted['W_max_ratio'] == 1. and fitted['largest_rise_between_steps'] < 0.     # never rises
    assert fitted['above_upper_bound'] < 1e-3                                       # under the bound of the smallest rate
    assert fitted['rate_fitted'] > .99 * fitted['start']['rate'][0]
    assert desired['W_max_ratio'] > 1.2 and desired['above_upper_bound'] > 1.       # rises first and leaves the bound
    assert max(desired['max_effort_ratio'], fitted['max_effort_ratio']) < .9        # no actuator at its bound


def test_a_kick_is_a_velocity_jump(compiled):
    mujoco = pytest.importorskip('mujoco')
    from backends.mujoco_plant import MujocoPlant
    comp = compiled('ridgeback2')
    plant = MujocoPlant(comp, roller_hull=(7, 32))
    q = comp.seed()
    plant.reset(q)
    before = plant.state()[1]
    dv = .1 * np.random.default_rng(1).standard_normal(plant.n)
    plant.kick(dv)
    q_after, after = plant.state()
    assert after - before == pytest.approx(dv, abs=1e-12) and q_after == pytest.approx(plant.state()[0])


def test_the_report_and_the_trajectory_table_of_a_run_with_the_quantities(compiled, missions, tmp_path):
    """The audit reports the loops, and tools/export_trajectory.py writes the motion of the team as a table."""
    pytest.importorskip('pinocchio')
    import sys
    from pathlib import Path
    from mission.audit import audit, summary_lines
    from mission.report import save_record
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
    import export_trajectory
    ctrl, rec, info = _carry(compiled, missions, 'ridgeback2', duration=1., feedforward='fitted')
    A = rec.arrays()
    info['plant'] = dict(name='pinocchio', controller=dict(feedforward='fitted', monitor=True))
    report, _ = audit(ctrl.comp, missions('ridgeback2'), A, info, rec.contacts or None, model=ctrl.model)
    loops = report['loops']
    assert loops['feedforward'] == 'fitted' and loops['updates_without_fitted_reference'] == 0
    assert 0. < loops['error_energy_J']['median'] <= loops['error_energy_J']['p99'] <= loops['error_energy_J']['max'] < 1e-3
    assert 20. < loops['rate_1_s']['smallest'] < loops['rate_1_s']['largest'] and list(loops['rate_1_s']['by_mode']) == [ctrl.mode.label()]
    assert any(line.startswith('loops, feedforward of the fitted reference') for line in summary_lines(report))
    save_record(tmp_path / 'record.npz', A, info, rec.contacts)
    assert export_trajectory.main(['--record', str(tmp_path / 'record.npz'), '--rate', '10']) == 0
    names = (tmp_path / 'trajectory.csv').read_text().splitlines()[0].split(',')
    table = np.loadtxt(tmp_path / 'trajectory.csv', delimiter=',', skiprows=1)
    assert table.shape == (10, len(names)) and names[:3] == ['t_s', 'phase', 'payload_x_m']
    assert {'r1_base_x_m', 'r2_base_ref_yaw_rad', 'grasp_r1_fz_N', 'loops_error_energy_J'} <= set(names)
    assert sum(name.startswith('effort_') for name in names) == ctrl.model.nf
    assert table[:, names.index('payload_x_m')] == pytest.approx(A['q'][::5, ctrl.payload[0]])
