"""Team compilation, closure modules and the generated interfaces."""
import numpy as np
import pytest

from teamc.chords import TeamAssembler
from teamc.loads import LoadSharing


def test_structure(compiled, team):
    comp = compiled(team)
    n_robots = len(comp.robots)
    assert n_robots == int(team[-1])
    grasps = [c for c in comp.connections.values() if c['kind'] == 'grasp']
    rests = [c for c in comp.connections.values() if c['kind'] == 'rest']
    assert len(grasps) == n_robots and len(rests) == 2
    # payload chart; per robot: chart, wheels, arm, and a lift (RB-ROBOUT+), a passive rocker (Ridgeback) or neither (Husky)
    lift, rocker, drive, ground = dict(team=(1, 0, 'rolling', 'fit'), ridgeback=(0, 1, 'rolling', 'contact'),
                                       husky=(0, 0, 'friction', 'fit'))[team[:-1]]
    assert comp.tree.n == 6 + n_robots * (6 + 4 + 6 + lift + rocker)
    lifts = sum(len(r.commanded) for r in comp.robots.values())
    rockers = sum(len(r.passive) for r in comp.robots.values())
    assert (lifts, rockers) == (n_robots * lift, n_robots * rocker)
    assert len(comp.cmg['ports']) == n_robots * 10 + lifts          # every joint but the rocker has an actuator
    assert all((r.drive, r.ground) == (drive, ground) for r in comp.robots.values())
    assert abs(comp.tree.total_mass() - sum(b['mass_kg'] for b in comp.cmg['bodies'])) < 1e-9


def test_closure_modules_close_the_grasps(closed, team):
    c = closed(team)
    asm = TeamAssembler(c['comp'], c['station'])
    for q in (c['station'], c['carry']):
        for cid, con in c['comp'].connections.items():
            if con['kind'] == 'grasp':
                assert max(asm.gap(cid, q)) < 1e-9                   # translation and rotation gap


def test_lift_spans_the_constraint_null_space(closed, team):
    c = closed(team)
    model = c['model']
    mode = model.mode()
    lift = model.lift(c['carry'], mode)
    con = lift.constraints
    assert np.max(np.abs(con.residual)) < 1e-8
    assert lift.rank == lift.Cp.shape[1]
    # every admissible velocity satisfies every constraint row, including the redundant ones
    u = np.random.default_rng(0).normal(size=len(lift.independent))
    full = np.linalg.lstsq(np.hstack([con.J, con.Jc]), np.zeros(con.J.shape[0]), rcond=None)[0]   # shape check only
    assert full.shape[0] == con.J.shape[1] + con.Jc.shape[1]
    v = lift.E @ u
    chart_rates = np.linalg.lstsq(con.Jc, -con.J @ v, rcond=None)[0] if con.Jc.size else np.zeros(0)
    defect = np.abs(con.J @ v + (con.Jc @ chart_rates if con.Jc.size else 0.))
    slip = np.zeros(len(defect), bool)
    for block in con.blocks:                                         # wheel rows of a friction-driven base are not constraints:
        slip[block['rows']] = bool(block.get('slip'))                # an admissible velocity slides the wheels
    assert defect[~slip].max() < 1e-8
    assert slip.any() == team.startswith('husky') and (not slip.any() or defect[slip].max() > .1)


def test_forward_and_inverse_dynamics_are_inverse_to_each_other(closed, team):
    c = closed(team)
    model = c['model']
    mode = model.mode()
    q = c['carry']
    rng = np.random.default_rng(1)
    lift = model.lift(q, mode)
    u = .2 * rng.normal(size=len(lift.independent))
    f = .1 * rng.uniform(-1, 1, model.nf) * model.effort_bounds[:, 1]
    fwd = model.forward(q, u, f, mode)
    assert fwd['reaction_balance'] < 1e-6
    assert fwd['virtual_power_defect'] < 1e-8
    grasp_wrenches = {cid: fwd['ports'][cid] for cid in mode.grasps}
    inv = model.inverse(q, fwd['v'], fwd['a'], mode, grasp_wrenches=grasp_wrenches)
    assert inv['residual'] < 1e-6
    again = model.forward(q, u, inv['f'], mode)                     # the efforts differ by a wheel-internal mode at most
    assert np.allclose(again['a'], fwd['a'], atol=1e-6)
    for cid in mode.grasps:
        assert np.allclose(again['ports'][cid], fwd['ports'][cid], atol=1e-5)


def test_static_carry_balances_the_weights(closed, team):
    c = closed(team)
    comp, model = c['comp'], c['model']
    mode = model.mode()
    q = c['carry']
    zero = np.zeros(model.n)
    ls = LoadSharing(model, mode.grasps)
    con = model.constraints(q, mode)
    kin = comp.tree.sweep(q)
    G = ls.grasp_matrix(con)
    weight = comp.tree.mass_kg[comp.tree.index[comp.payload['body']]] * 9.81
    Tp = kin[0][comp.tree.index[comp.payload['body']]]
    need = comp.tree.rnea(q, zero, zero, kinematics=kin)[np.asarray(comp.payload['base'])]
    wrenches = ls.distribute(G, need)
    out = model.inverse(q, zero, zero, mode, grasp_wrenches=wrenches, kin=kin, constraints=con)
    assert out['residual'] < 1e-6
    up = 0.
    for cid in mode.grasps:
        i1, _ = model._paths[cid]
        R = (kin[0][i1] @ comp.connections[cid]['T1'])[:3, :3]
        up += (R @ wrenches[cid][3:])[2]
    assert abs(up - weight) < 1e-6                                   # the grasps carry the payload ...
    total = comp.tree.total_mass() * 9.81
    # the z row of a fitted plane, or the sum of the wheel loads where every wheel has its own row
    ground = sum(out['ports'][f'{rid}:ground'].sum() if comp.robots[rid].ground == 'contact' else out['ports'][f'{rid}:ground'][0]
                 for rid in comp.robots)
    assert abs(ground - total) / total < 2e-3                        # ... and the ground carries everything (small slopes)
    assert np.all(np.abs(out['f']) <= model.effort_bounds[:, 1])


def test_declared_shares_are_reproduced(closed):
    c = closed('team3')
    comp, model = c['comp'], c['model']
    mode = model.mode()
    q = c['carry']
    shares = c['mission']['load_sharing']['shares']
    ls = LoadSharing(model, mode.grasps, shares={g: shares[g] for g in mode.grasps})
    con = model.constraints(q, mode)
    kin = comp.tree.sweep(q)
    need = comp.tree.rnea(q, np.zeros(model.n), np.zeros(model.n), kinematics=kin)[np.asarray(comp.payload['base'])]
    wrenches = ls.distribute(ls.grasp_matrix(con), need)
    parts = ls.decompose(ls.grasp_matrix(con), wrenches)
    assert max(np.linalg.norm(w) for w in parts['internal'].values()) < 1e-8      # the distribution itself has no internal part


def test_reaction_recovery_returns_the_reactions(closed):
    c = closed('team2')
    model = c['model']
    mode = model.mode()
    q = c['carry']
    rng = np.random.default_rng(3)
    lift = model.lift(q, mode)
    u = .1 * rng.normal(size=len(lift.independent))
    f = .1 * rng.uniform(-1, 1, model.nf) * model.effort_bounds[:, 1]
    fwd = model.forward(q, u, f, mode)
    rec = model.recover(q, fwd['v'], fwd['a'], f, mode)
    assert rec['unbalanced'] < 1e-6
    for cid in mode.grasps:
        assert np.allclose(rec['ports'][cid], fwd['ports'][cid], atol=1e-5)
