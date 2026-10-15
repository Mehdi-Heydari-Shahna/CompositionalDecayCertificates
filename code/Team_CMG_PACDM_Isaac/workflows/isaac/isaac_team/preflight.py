"""Native PhysX checks against the compiled model, before any mission step.

What PhysX actually uses is read back through the tensor API and compared with the
compiled graph: mass properties, joint properties, gravity, state round trip,
link kinematics, the convex hulls PhysX cooked for the rollers, and single free-flight
steps (robots and payload lifted clear of the site, so no contact acts): the
accelerations against the compiled forward dynamics, the last samples with a known
external load applied the way the clamp loads are applied (and, for conventional
wheels, one more with known forces applied the way the tyre loads are applied: on
every wheel at its ground point), and the displacement of the step against the
reported rates, which fixes the velocity conventions.  No contact behaviour is
certified here; the ``settle`` and ``carry`` stages of the workflow do that with the
mission controller.
"""
from __future__ import annotations

import numpy as np

from backends.latch import LatchSet
from teamc.se3 import rotation_angle


def cooked_hulls(native, meta, stage, stage_id, limit=2e-5):
    """Compare the hulls PhysX cooked for one roller with the authored points.  Returns ``(rows, worst)``."""
    from pxr import UsdGeom
    from scipy.spatial import ConvexHull
    r = next(iter(meta['robots'].values()))
    roller = next(path for name, path in r['bodies'].items() if 'roller' in name)
    pieces = int(meta['roller']['pieces'])
    unit = float(meta['roller']['mesh_unit_m'])
    rows, worst = [], 0.
    for n in sorted({0, pieces // 2, pieces - 1}):
        path = f'{roller}/hull_{n:02d}'
        authored = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get(), float)
        cooked = native.cooked_hull(path, stage_id)
        ratio = np.ptp(authored, axis=0).max() / np.ptp(cooked, axis=0).max()
        factor = next((f for f in (1., 1. / unit, unit) if .8 < ratio / f < 1.25), None)   # cooked units: the mesh's or metres
        if factor is None:
            raise RuntimeError(f'{path}: the cooked hull has another size (extent ratio {ratio:.4g})')
        hull = ConvexHull(cooked * factor)
        outside = float((authored @ hull.equations[:, :3].T + hull.equations[:, 3]).max()) * unit
        worst = max(worst, outside)
        rows.append(dict(path=path, authored_vertices=len(authored), cooked_vertices=len(cooked), unit_factor=factor,
                         authored_point_outside_cooked_hull_m=outside))
    return rows, worst


def run(plant, native, comp, meta, q_start, stage=None, stage_id=None, samples=6, seed=48271, lift=2.0, load=(150., 8.)):
    """``load``: size of the known clamp load of the loaded samples, force in N and moment in N m.

    The workflow passes a third of the grasp-force bound of the mission and 8/45 of its moment bound
    (150 N and 8 N m for the RB-ROBOUT+ teams): a load of the size the clamps carry in the mission,
    so that the one step it acts for stays small for a light payload as well.

    A team on conventional wheels gets one more dynamic sample, with a known force on every wheel at
    its ground point (``backends/tyre.py`` applies the tyre friction there): half the friction limit
    of a wheel that carries an even share of the weight of its robot.
    """
    checks, details = [], dict(unavailable=[], samples=[], load=dict(force_N=float(load[0]), moment_Nm=float(load[1])))
    tyres = getattr(plant, 'tyres', None)
    tyre_force = {}
    if tyres:
        g = float(np.linalg.norm(comp.tree.gravity))
        for rid, r in comp.robots.items():
            mass = sum(comp.tree.mass_kg[i] for i in comp.tree.massive if comp.tree.names[i].startswith(r.prefix))
            for w in tyres.wheels:
                if w['key'][0] == rid:
                    tyre_force[w['key']] = .5 * w['mu'] * mass * g / len(r.wheels)
        details['load']['tyre_force_N'] = float(max(tyre_force.values()))
    dynamic = samples + (1 if tyres else 0)                         # the extra sample carries the known tyre forces

    def check(name, value, limit, relation='<='):
        value = float(value)
        ok = bool(np.isfinite(value) and (value <= limit if relation == '<=' else value >= limit))
        checks.append(dict(name=name, value=value if np.isfinite(value) else None, limit=float(limit), relation=relation, passed=ok))
        return ok

    tree = comp.tree
    # ------------------------------------------------------------------ properties PhysX uses
    for rid, r in meta['robots'].items():
        p = native.properties(rid)
        details['unavailable'] += [f'{rid}: {x}' for x in p['unavailable']]
        exp = meta['expected'][rid]
        mass = np.array([exp[name]['mass'] for name in p['links']])
        com = np.array([exp[name]['com'] for name in p['links']])
        inertia = np.array([exp[name]['inertia'] for name in p['links']])
        check(f'{rid}: link masses, largest relative error', np.max(np.abs(p['mass'] - mass) / mass), 2e-6)
        check(f'{rid}: link centres of mass / m', np.max(np.abs(p['com'] - com)), 2e-6)
        scale = np.abs(inertia).max(axis=(1, 2))[:, None, None]
        check(f'{rid}: link inertia tensors, largest relative error', np.max(np.abs(p['inertia'] - inertia) / scale), 5e-5)
        check(f'{rid}: total mass / kg, relative error', abs(p['mass'].sum() - mass.sum()) / mass.sum(), 2e-6)
        if 'armature' in p:
            want = np.array([r['armature'][d] for d in p['dofs']])
            check(f'{rid}: joint armature / SI', np.max(np.abs(p['armature'] - want)), 1e-6)
        if 'stiffness' in p:
            check(f'{rid}: no drive stiffness', np.max(np.abs(p['stiffness'])), 0.)
        if 'drive_damping' in p:
            check(f'{rid}: no drive damping', np.max(np.abs(p['drive_damping'])), 0.)
        if 'friction' in p:
            check(f'{rid}: no native joint friction', np.max(np.abs(p['friction'])), 0.)
        if 'limits' in p:
            err = 0.
            for k, d in enumerate(p['dofs']):
                if r['limits'][d] is not None:
                    err = max(err, float(np.max(np.abs(p['limits'][k] - np.asarray(r['limits'][d])))))
            check(f'{rid}: joint limits / SI', err, 5e-6)
    pp, pe = native.payload_properties(), meta['payload']['expected']
    check('payload: mass, relative error', abs(pp['mass'] - pe['mass']) / pe['mass'], 2e-6)
    check('payload: centre of mass / m', np.max(np.abs(pp['com'] - np.asarray(pe['com']))), 2e-6)
    check('payload: inertia tensor, relative error', np.max(np.abs(pp['inertia'] - np.asarray(pe['inertia']))) / np.abs(pe['inertia']).max(), 5e-5)
    g = native.gravity()
    if g is None:
        details['unavailable'].append('gravity readback')
    else:
        check('gravity / m s-2', np.max(np.abs(g - tree.gravity)), 1e-5)
    if meta.get('roller') and stage is not None:
        try:
            details['cooked_hulls'], worst = cooked_hulls(native, meta, stage, stage_id)
            check('roller hulls: authored points left outside the hull PhysX cooked / m', worst, 2e-5)
        except Exception as exc:                                      # noqa: BLE001 (the query itself is optional)
            details['unavailable'].append(f'cooked roller hulls: {type(exc).__name__}: {exc}')

    # ------------------------------------------------------------------ free-flight samples
    rng = np.random.default_rng(seed)
    plant.reset_sensor_reference()
    ports = comp.cmg['ports']
    B = np.zeros((tree.n, len(ports)))
    for k, p in enumerate(ports):
        B[p['index'], k] = p['gear']
    bounds = np.array([p['bounds'][1] for p in ports])
    damping = np.array([comp.cmg['damping'].get(k, 0.) for k in comp.cmg['coordinate_ids']])
    floating = np.concatenate([r.base for r in comp.robots.values()] + [np.asarray(comp.payload['base'])])
    joints = np.setdiff1d(np.arange(tree.n), floating)
    wheels = np.array([w['index'] for r in comp.robots.values() for w in r.wheels], dtype=int)
    lower = np.array([comp.cmg['limits'][k]['lower'] for k in comp.cmg['coordinate_ids']])
    upper = np.array([comp.cmg['limits'][k]['upper'] for k in comp.cmg['coordinate_ids']])
    writes_before = native.writes
    try:
        for i in range(dynamic + 1):
            kinematic = i == dynamic                               # one more sample, for the velocity conventions only
            q = np.asarray(q_start, float).copy()
            for r in comp.robots.values():
                q[r.base[2]] += lift
                q[r.base[3:]] += rng.uniform(-.15, .15, 3)
            pb = np.asarray(comp.payload['base'])
            q[pb[2]] += lift + 1.
            q[pb[3:]] += rng.uniform(-.15, .15, 3)
            moving = np.setdiff1d(joints, wheels)
            q[moving] = np.clip(q[moving] + rng.uniform(-.08, .08, len(moving)), lower[moving] + .02, upper[moving] - .02)
            v = np.zeros(tree.n) if i < samples // 2 else rng.uniform(-.2, .2, tree.n)
            f = rng.uniform(-.2, .2, len(ports)) * bounds
            if kinematic:
                # A marked tilt rate of every base and no efforts.  The centre of mass of a base is not at its origin, so
                # the displacement of this step shows which point of the body the reported linear velocity belongs to.
                for r in comp.robots.values():
                    v[r.base[4:]] = rng.choice([-1., 1.], 2) * rng.uniform(.6, 1., 2)
                f = np.zeros(len(ports))
            plant.reset(q, v)
            q0, v0 = plant.state()
            check(f'sample {i}: configuration round trip', np.max(np.abs(q0 - q)), 2e-5)
            check(f'sample {i}: velocity round trip', np.max(np.abs(v0 - v)), 2e-5)
            poses = tree.poses(q0)
            dp = dr = 0.
            for rid in comp.robots:
                for name, T in native.link_poses(rid).items():
                    if name in poses:
                        dp = max(dp, float(np.linalg.norm(T[:3, 3] - poses[name][:3, 3])))
                        dr = max(dr, rotation_angle(T[:3, :3].T @ poses[name][:3, :3]))
            check(f'sample {i}: link positions against the compiled kinematics / m', dp, 5e-5)
            check(f'sample {i}: link orientations against the compiled kinematics / rad', dr, 2e-4)
            # The last two dynamic samples carry a known load, applied exactly as the clamp loads of a mission are: on each
            # clamp link at its tool point, and the opposite on the payload.  One sample has forces only, the other moments
            # only (``load``: mission scale), and each is stepped twice from the same state, without and with the load, so
            # that the difference of the two accelerations is the effect of the load alone.  Conventional wheels: a third
            # loaded sample has a known force on every wheel at its ground point, where the tyre loads of a mission act.
            loaded = not kinematic and i >= samples - 2
            kind = None if not loaded else ('forces', 'moments', 'tyre forces')[i - (samples - 2)]
            Q = np.zeros(tree.n)
            M = tree.mass(q0)
            unloaded = None
            if loaded:
                plant.set_efforts(f)
                plant.step()
                unloaded = (plant.state()[1] - v0) / plant.dt
                plant.reset(q, v)
                again = plant.state()
                check(f'sample {i}: the same state again for the loaded step',
                      max(np.max(np.abs(again[0] - q0)), np.max(np.abs(again[1] - v0))), 1e-6)
                kin = tree.sweep(q0)
                T = kin[0]
                loads, fext = {}, {}
                for w in (tyres.wheels if kind == 'tyre forces' else ()):
                    point, _ = tyres.ground_point(w, T[w['index'], :3, 3])
                    size = rng.choice([-1., 1.], 3) * rng.uniform(.5, 1., 3)
                    load_ = dict(point=point, force=tyre_force[w['key']] * size, moment=np.zeros(3), on=None, by=w['body'],
                                 deflection=0.)
                    loads[('tyre',) + w['key']] = load_          # the key of the tyre load it stands in for
                    fext[w['body']] = fext.get(w['body'], 0.) - LatchSet.spatial(point, load_['force'], load_['moment'])
                for cid, c in comp.connections.items():
                    if c['kind'] != 'grasp' or kind == 'tyre forces':
                        continue
                    point = (T[tree.index[c['body1']]] @ np.asarray(c['T1'], float))[:3, 3]
                    size = rng.choice([-1., 1.], 3) * rng.uniform(.5, 1., 3)
                    load_ = dict(point=point, force=load[0] * size if kind == 'forces' else np.zeros(3),
                                 moment=load[1] * size if kind == 'moments' else np.zeros(3), on=c['body2'], by=c['body1'], deflection=0.)
                    loads[cid] = load_
                    w = LatchSet.spatial(load_['point'], load_['force'], load_['moment'])
                    fext[load_['on']] = fext.get(load_['on'], 0.) + w
                    fext[load_['by']] = fext.get(load_['by'], 0.) - w
                zero = np.zeros(tree.n)
                Q = tree.rnea(q0, v0, zero, kinematics=kin) - tree.rnea(q0, v0, zero, fext=fext, kinematics=kin)
                plant.inject(loads)
            expected = np.linalg.solve(M, B @ f - damping * v0 - tree.bias(q0, v0) + Q)
            plant.set_efforts(f)
            plant.step()
            q1, v1 = plant.state()
            measured = (v1 - v0) / plant.dt
            sample = dict(sample=i, dt_s=plant.dt, external_load=kind, purpose='velocity conventions' if kinematic else 'dynamics')
            if not kinematic:
                for label, idx in (('floating coordinates', floating), ('joints', joints)):
                    rel = float(np.linalg.norm(measured[idx] - expected[idx]) / max(1., np.linalg.norm(expected[idx])))
                    check(f'sample {i}: free-flight forward dynamics{" with an external load" if loaded else ""}, {label}', rel, .02)
                    sample[label] = dict(relative_error=rel, expected=expected[idx].tolist(), measured=measured[idx].tolist())
            if loaded:
                effect = np.linalg.solve(M, Q)
                sample['external_load_effect'] = {}
                for label, idx in (('floating coordinates', floating), ('joints', joints)):
                    rel = float(np.linalg.norm(measured[idx] - unloaded[idx] - effect[idx]) / np.linalg.norm(effect[idx]))
                    check(f'sample {i}: effect of external {kind} on the accelerations (with minus without), {label}', rel, .03)
                    sample['external_load_effect'][label] = dict(relative_error=rel, size=float(np.linalg.norm(effect[idx])))
                # the load was given for one step: the following step, with the same efforts, must be free of it
                following = np.linalg.solve(tree.mass(q1), B @ f - damping * v1 - tree.bias(q1, v1))
                plant.set_efforts(f)
                plant.step()
                after = (plant.state()[1] - v1) / plant.dt
                for label, idx in (('floating coordinates', floating), ('joints', joints)):
                    rel = float(np.linalg.norm(after[idx] - following[idx]) / np.linalg.norm(effect[idx]))
                    check(f'sample {i}: the external {kind} do not act in the following step, {label} (share of their effect left)', rel, .3)
                    sample['external_load_effect'][label]['left_in_following_step'] = rel
            # The step as a displacement.  Whatever the integrator (rates before the step, after it, or sub-steps in
            # between), every coordinate moves by dt times a rate between its rate before and after the step.  A
            # displacement outside that interval means the reported rates are not the rates of the compiled coordinates:
            # another reference point of the linear velocity, another angular convention or unit, or positions and
            # velocities in different orders.
            moved = q1 - q0
            a, b = plant.dt * v0, plant.dt * v1
            gap = np.maximum(np.maximum(np.minimum(a, b) - moved, moved - np.maximum(a, b)), 0.)
            check(f'sample {i}: step displacement against the reported rates, floating coordinates / m, rad', gap[floating].max(), 2e-5)
            check(f'sample {i}: step displacement against the reported rates, joints / rad, m', gap[joints].max(), 2e-5)
            sample['step_displacement_gap'] = dict(floating=float(gap[floating].max()), joints=float(gap[joints].max()))
            contacts = plant.contacts()
            touching = len(contacts['forbidden']) + len(contacts['wheels']) + len(contacts['pads'])
            check(f'sample {i}: no contact while lifted', touching, 0.)
            details['samples'].append(sample)
    finally:
        plant.set_efforts(np.zeros(len(ports)))
        plant.reset(q_start)
    # the wrist sensor balance against PhysX's own incoming joint force, over the free-flight samples (reported, not required:
    # the native quantity is an independent reference whose convention is not assumed)
    details['sensor_reference'] = plant.sensor_reference()
    plant.reset_sensor_reference()

    # ------------------------------------------------------------------ the cart (not essential: the scenario owns its position)
    cart = dict(mode=native.cart_mode)
    if native.cart is not None:
        try:
            start = np.asarray(meta['obstacle']['start'], float)
            target = start + np.array([.01, 0., 0.])
            plant.set_obstacle(target)
            plant.step()
            pose = native.cart_pose()
            cart['error_m'] = float(np.linalg.norm(pose[:3] - target))
            cart['follows'] = bool(cart['error_m'] < 1e-4)
            plant.set_obstacle(start)
            plant.step()
        except Exception as exc:                                      # noqa: BLE001
            cart.update(follows=False, error=f'{type(exc).__name__}: {exc}')
        plant.reset(q_start)
    details['cart'] = cart
    details['state_writes'] = native.writes - writes_before
    return dict(passed=all(c['passed'] for c in checks), checks=checks, details=details,
                scope='PhysX property readback, kinematics, velocity conventions and free-flight forward dynamics against the '
                      'compiled graph; no contact, clamp or sensor behaviour is checked here')
