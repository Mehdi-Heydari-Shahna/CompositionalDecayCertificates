"""Audit of a recorded mission (the same for every plant).

Everything is computed from the record: measured states, applied efforts, wrist
sensor wrenches and the controller log.  Where a plant also reports contact
forces or solver forces they are used only as an independent reference.

Sections of the report:

* ``tracking``    payload and base tracking, payload orientation while carried;
* ``grasp``       measured grasp wrenches (every record interval), load share and
                  internal wrench, wrist-sensor readings against the sensor range;
* ``efforts``     actuator demand against the declared bounds;
* ``ground``      wheel loads, traction utilization and tipping margins from the
                  compiled reaction recovery; the travel of passive joints; for a
                  friction-driven base (skid steering) the slip of its wheels, the
                  power they dissipate and its motion against the skid model;
* ``clearance``   separation distances by category and forbidden contacts;
* ``transfer``    load hand-over at pickup and placement and the support check;
* ``consistency`` compiled model against the plant: reaction recovery against the
                  wrist sensors and contact forces, payload wrench balance,
                  connection gaps;
* ``bounds``      the declared admissible bounds and whether they were met.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from teamc.geometry import Geometry
from teamc.interfaces import Mode, TeamModel
from teamc.loads import LoadSharing
from teamc.se3 import inv, log
from teamc.stability import ground_loads, pad_loads


def _stats(x):
    x = np.asarray(x, float)
    return dict(max=float(np.max(x)), rms=float(np.sqrt(np.mean(x ** 2)))) if x.size else dict(max=None, rms=None)


def _mode(model, grasps, rest):
    g = tuple(x for x in str(grasps).split(',') if x)
    return Mode(grasps=g, rest=(str(rest) or None))


def phase_table(A, info):
    t, k = A['t'], A['phase'].astype(int)
    names = info['phases']
    out = []
    for i in np.unique(k):
        sel = np.flatnonzero(k == i)
        out.append(dict(index=int(i), name=names[i], start=float(t[sel[0]]), end=float(t[sel[-1]])))
    return out


def audit(comp, mission, A, info, contacts=None, sample_hz=10., model=None):
    model = TeamModel(comp) if model is None else model
    tree = comp.tree
    t = A['t']
    n = tree.n
    dt_rec = float(np.median(np.diff(t)))
    stride = max(int(round(1. / (sample_hz * dt_rec))), 1)
    idx = np.arange(0, len(t), stride)
    phases = phase_table(A, info)
    name_of = {p['index']: p['name'] for p in phases}
    k = A['phase'].astype(int)
    carrying = np.array([(g != '') and (r == '') for g, r in zip(A['mode_grasps'], A['mode_rest'])])
    pb = np.asarray(comp.payload['base'], int)
    grasps = [cid for cid, c in comp.connections.items() if c['kind'] == 'grasp']
    robot_of = {cid: comp.connections[cid]['robot'] for cid in grasps}
    weight = float(mission['payload']['mass_kg']) * float(np.linalg.norm(tree.gravity))
    bounds = mission['bounds']
    report = dict(status=info['status'], sim_time_s=info['sim_time'], phases=phases, events=info['events'],
                  verification=info.get('verification', {}))

    # ------------------------------------------------------------------ tracking
    e = A['q'][:, pb] - A['payload_ref']
    carry_e = e[carrying]
    tr = dict(payload_position_error_m=_stats(np.linalg.norm(carry_e[:, :3], axis=1)),
              payload_yaw_error_rad=_stats(np.abs(carry_e[:, 3])),
              payload_tilt_rad=_stats(np.hypot(A['q'][carrying][:, pb[4]], A['q'][carrying][:, pb[5]])),
              base={})
    for rid, r in comp.robots.items():
        be = A[f'base_ref:{rid}'] - A['q'][:, r.base[[0, 1, 3]]]
        tr['base'][rid] = dict(position_error_m=_stats(np.linalg.norm(be[:, :2], axis=1)), yaw_error_rad=_stats(np.abs(be[:, 2])))
    report['tracking'] = tr

    # ------------------------------------------------------------------ efforts
    ports = comp.cmg['ports']
    hi = np.array([p['bounds'][1] for p in ports])
    ratio = np.abs(A['f_peak']) / hi
    report['efforts'] = dict(max_ratio=float(ratio.max()), by_port={p['id']: float(r) for p, r in zip(ports, ratio.max(axis=0))},
                             saturated_steps=info.get('saturated_steps', 0),
                             worst=dict(port=ports[int(ratio.max(axis=0).argmax())]['id'],
                                        t=float(t[int(ratio.max(axis=1).argmax())])))

    # ------------------------------------------------------------------ sampled model quantities
    if 'a' in A:                         # one-step accelerations recorded with the efforts that produced them
        acc = A['a']
    else:
        win = max(int(round(.3 / dt_rec)) | 1, 5)
        acc = savgol_filter(A['v'], win, 2, deriv=1, delta=dt_rec, axis=0, mode='interp')
    sensor_of = {c: f'sensor:{robot_of[c]}' for c in grasps}
    geo = Geometry(comp)
    ls_cache = {}
    S = dict(t=[], grasp_meas={c: [] for c in grasps}, grasp_model={c: [] for c in grasps}, internal_force=[], internal_moment=[],
             share={c: [] for c in grasps}, ground={rid: [] for rid in comp.robots}, unbalanced=[], balance=[], gap=[],
             clearance=[], pads_model=[], carry=[], height={rid: [] for rid in comp.robots},
             skid={rid: [] for rid in comp.robots if model.friction[rid]})
    worst_clear = {}
    ob = comp.scene.get('obstacle')
    for i in idx:
        q, v, a, f = A['q'][i], A['v'][i], acc[i], A['f'][i]
        mode = _mode(model, A['mode_grasps'][i], A['mode_rest'][i])
        kin = tree.sweep(q)
        # grasp wrenches from the wrist sensors of this step (tool-side inertia removed)
        lam_s = {c: model.sensor_to_grasp(robot_of[c], A[sensor_of[c]][i], q, kin, v, a) for c in mode.grasps}
        known = lam_s if mode.rest else None
        rec = model.recover(q, v, a, f, mode, known=known, kin=kin)
        con = rec['constraints']
        S['t'].append(t[i])
        S['unbalanced'].append(rec['unbalanced'])
        S['carry'].append(bool(mode.grasps and not mode.rest))
        for c in grasps:
            S['grasp_meas'][c].append(lam_s[c] if c in mode.grasps else np.full(6, np.nan))
            S['grasp_model'][c].append(rec['ports'][c] if (c in mode.grasps and not mode.rest) else np.full(6, np.nan))
        if mode.grasps:
            key = tuple(mode.grasps)
            if key not in ls_cache:
                sh = mission['load_sharing']
                ls_cache[key] = LoadSharing(model, key, shares={g: sh['shares'][g] for g in key},
                                            moment_scale=sh['moment_scale_Nm'], force_scale=sh['force_scale_N'])
            ls = ls_cache[key]
            G = ls.grasp_matrix(con)
            lam = lam_s
            if not mode.rest:
                dec = ls.decompose(G, lam)
                S['internal_force'].append(dec['internal_force_norm'])
                S['internal_moment'].append(dec['internal_moment_norm'])
                need = (tree.rnea(q, v, a, kinematics=kin))[pb]
                S['balance'].append(float(np.linalg.norm(dec['resultant'] - need)))
            else:
                S['internal_force'].append(np.nan)
                S['internal_moment'].append(np.nan)
                S['balance'].append(np.nan)
            gap = 0.
            for c in key:
                r, *_ = model.connection_rows(c, kin)
                gap = max(gap, float(np.linalg.norm(r[3:])))
            S['gap'].append(gap)
        else:
            for key_ in ('internal_force', 'internal_moment', 'balance', 'gap'):
                S[key_].append(np.nan)
        up = {}
        for c in grasps:
            if c in mode.grasps:
                ib, T1 = model._paths[c][0], comp.connections[c]['T1']
                R = (kin[0][ib] @ T1)[:3, :3]
                up[c] = float((R @ lam_s[c][3:])[2])
        for c in grasps:
            S['share'][c].append(up.get(c, np.nan) / weight)
        for rid in comp.robots:
            gl = ground_loads(model, q, rid, rec['ports'][f'{rid}:ground'], rec['ports'][f'{rid}:wheels'], kin,
                              con.info.get(f'{rid}:normal'))
            # traction that the declared bound is about: of the most used wheel, or, for a base whose wheels slide
            # when it turns, of the resultant the wheels pass to the ground
            use = gl['net_traction_utilization'] if model.friction[rid] else gl['max_traction_utilization']
            S['ground'][rid].append([gl['min_normal_load_N'], use, gl['tipping_margin_m'],
                                     gl['total_normal_N']] + list(gl['normal_loads_N']) + list(gl['traction_N']))
            if model.friction[rid]:
                tyre = model._tyre[rid]
                loads = np.maximum(gl['normal_loads_N'], 0.)
                slip = model.wheel_slip(q, v, rid, kin, con.info.get(f'{rid}:normal'))[0]
                speed = np.linalg.norm(slip, axis=1)
                law = tyre['mu'] * loads * speed / np.sqrt(speed ** 2 + tyre['slip_speed'] ** 2)
                try:
                    modelled = model.skid_twist(q, v, rid, loads, kin)
                except ValueError:                 # the wheels of one side carry nothing: the skid model has no twist to give
                    modelled = np.full(3, np.nan)
                S['skid'][rid].append(np.r_[model.planar_twist(q, v, rid, kin), modelled,
                                            float(tyre['mu'] * loads @ speed), speed, law])
            base = comp.robots[rid].base
            dep, rigid = model.ground_pose(rid, q[base[[0, 1, 3]]])
            S['height'][rid].append(q[dep] - rigid)       # measured minus rigid-wheel support: z, pitch, roll (, passive joints)
        S['pads_model'].append(pad_loads(comp.connections[mode.rest], rec['ports'][mode.rest])['pad_normal_N'] if mode.rest else {})
        obstacle = geo.place_obstacle(A['obstacle'][i]) if (ob and 'obstacle' in A) else None
        cl = geo.clearance(kin[0], cutoff=1., obstacle=obstacle)
        S['clearance'].append({key_: item[0] for key_, item in cl.items()})
        for key_, item in cl.items():
            if key_ not in worst_clear or item[0] < worst_clear[key_][0]:
                worst_clear[key_] = (item[0], item[1], item[2], float(t[i]), name_of.get(int(k[i]), ''))
    ts = np.asarray(S['t'])
    carry = np.asarray(S['carry'])

    # ------------------------------------------------------------------ grasp loads
    # Force and moment of every grasp from the wrist sensor of EVERY record interval in which its clamp is closed
    # (the sampled quantities above cover every ``stride``-th interval; a peak must not depend on which ones).
    closed = {c: np.array([c in str(g).split(',') for g in A['mode_grasps']]) for c in grasps}
    every = {c: np.full((len(t), 6), np.nan) for c in grasps}
    for i in np.flatnonzero(np.any([closed[c] for c in grasps], axis=0)):
        kin = tree.sweep(A['q'][i])
        for c in grasps:
            if closed[c][i]:
                every[c][i] = model.sensor_to_grasp(robot_of[c], A[sensor_of[c]][i], A['q'][i], kin, A['v'][i], acc[i])
    grasp = dict(internal_force_N=_stats(np.asarray(S['internal_force'])[carry]),
                 internal_moment_Nm=_stats(np.asarray(S['internal_moment'])[carry]), by_grasp={},
                 note='force and moment of each grasp: all record intervals with the clamp closed (interval means of the wrist '
                      f'sensor, tool-side inertia removed); internal wrench and shares: every {stride}th interval')
    for c in grasps:
        lam = every[c][closed[c]]
        sh = np.asarray(S['share'][c])
        grasp['by_grasp'][c] = dict(force_N=_stats(np.linalg.norm(lam[:, 3:], axis=1)),
                                    moment_Nm=_stats(np.linalg.norm(lam[:, :3], axis=1)),
                                    share_while_carrying=dict(mean=float(np.nanmean(sh[carry])), min=float(np.nanmin(sh[carry])),
                                                              max=float(np.nanmax(sh[carry]))) if carry.any() else None)
    # raw wrist-sensor readings against the declared range of the sensor (the sensor also carries the tool-side body)
    readings = {}
    for rid, r in comp.robots.items():
        if f'sensor:{rid}' not in A:
            continue
        w = np.asarray(A[f'sensor:{rid}'], float)
        declared = (r.record['ports'].get('sensor') or {}).get('range') or None
        entry = dict(force_N=float(np.linalg.norm(w[:, 3:], axis=1).max()), moment_Nm=float(np.linalg.norm(w[:, :3], axis=1).max()),
                     range=declared)
        if declared:
            entry['within_range'] = bool(entry['force_N'] <= declared['force_N'] and entry['moment_Nm'] <= declared['torque_Nm'])
        readings[rid] = entry
    if readings:
        grasp['sensor_reading'] = dict(by_robot=readings, note='largest norm of the force and of the moment read by each wrist sensor over '
                                                               'the whole run (means over one record interval), with the range declared '
                                                               'in the robot record; reported, not a declared bound')
    report['grasp'] = grasp

    # ------------------------------------------------------------------ ground
    ground = {}
    settled = ts >= ts[0] + .5                     # the plant settles onto its contacts at the start
    for rid in comp.robots:
        g = np.asarray(S['ground'][rid])[settled]
        tg = ts[settled]
        names = ('net_traction_utilization_max', 'net_traction_utilization_p99') if model.friction[rid] \
            else ('max_traction_utilization', 'traction_utilization_p99')
        ground[rid] = dict(min_wheel_load_N=float(g[:, 0].min()), **{names[0]: float(g[:, 1].max()),
                                                                     names[1]: float(np.percentile(g[:, 1], 99.))},
                           min_tipping_margin_m=float(g[:, 2].min()),
                           mean_wheel_loads_N=g[:, 4:4 + len(comp.robots[rid].wheels)].mean(axis=0).tolist(),
                           at=dict(min_wheel_load=float(tg[g[:, 0].argmin()]), traction=float(tg[g[:, 1].argmax()]),
                                   tipping=float(tg[g[:, 2].argmin()])))
        h = np.asarray(S['height'][rid])[settled]
        ground[rid]['against_rigid_wheels'] = dict(height_m=dict(mean=float(h[:, 0].mean()), min=float(h[:, 0].min()), max=float(h[:, 0].max())),
                                                   tilt_rad_max=float(np.abs(h[:, 1:3]).max()))
        if comp.robots[rid].passive:
            # passive joints of the base: the travel they used and how far they stayed from their limits
            lim = comp.cmg['limits']
            ground[rid]['passive_joints'] = {}
            for name in comp.robots[rid].passive:
                kj = comp.robots[rid].joint[name]
                x = A['q'][:, kj]
                lo, hi = lim[comp.cmg['coordinate_ids'][kj]]['lower'], lim[comp.cmg['coordinate_ids'][kj]]['upper']
                ground[rid]['passive_joints'][name] = dict(min=float(x.min()), max=float(x.max()), limits=[lo, hi],
                                                           margin_to_limit=float(min(x.min() - lo, hi - x.max())))
            ground[rid]['against_rigid_wheels']['passive_rad_max'] = float(np.abs(h[:, 3:]).max())
    for rid in S['skid']:
        k_ = np.asarray(S['skid'][rid])[settled]
        tg = ts[settled]
        nw = len(comp.robots[rid].wheels)
        meas, mod, power, speed = k_[:, :3], k_[:, 3:6], k_[:, 6], k_[:, 7:7 + nw]
        known = np.isfinite(mod[:, 2])
        turning = known & (np.abs(np.where(known, mod[:, 2], 0.)) > .02)           # the skid model has something to say
        entry = dict(max_slip_speed_m_s=float(speed.max()), friction_power_W=dict(max=float(power.max()), mean=float(power.mean())),
                     friction_energy_J=float(np.sum(.5 * (power[1:] + power[:-1]) * np.diff(tg))),
                     turned_rad=float(np.sum(np.abs(.5 * (meas[1:, 2] + meas[:-1, 2])) * np.diff(tg))))
        if not known.all():
            entry['skid_model_not_evaluated'] = dict(
                samples=int((~known).sum()), of=int(len(known)),
                note='samples at which the wheels of one side of the base carried no load: the skid model gives no twist there')
        if turning.any():
            entry['against_skid_model'] = dict(
                yaw_rate_ratio=float(meas[turning, 2] @ mod[turning, 2] / (mod[turning, 2] @ mod[turning, 2])),
                yaw_rate_rad_s=_stats(meas[turning, 2] - mod[turning, 2]),
                forward_m_s=_stats(meas[turning, 0] - mod[turning, 0]),
                sideways_m_s=_stats(meas[turning, 1] - mod[turning, 1]),
                share_of_time=float(turning.mean()),
                note='measured twist of the base less the twist the quasi-static skid model gives for the measured wheel '
                     'rates and the compiled wheel loads, while that model turns the base faster than 0.02 rad/s; ratio: '
                     'least-squares ratio of the measured to the modelled yaw rate')
        ground[rid]['skid'] = entry
    if S['skid']:
        ground['note_friction_drive'] = (
            'bases on conventional wheels turn by sliding them: the friction of a wheel is at its limit whenever it slides, '
            'so the traction reported and bounded for these bases is the resultant ground force of their wheels over their '
            'friction capacity; friction power and energy are those of the declared Coulomb law at the measured slip')
    if all(r.ground == 'contact' for r in comp.robots.values()):
        ground['note'] = ('wheel loads are the reactions of the wheel contact rows (determinate: the passive joints of the base '
                          'are free); the tipping margin is taken to the polygon of the fixed wheels and the passive pivots; '
                          'values are means over one record interval, the first 0.5 s is excluded')
    else:
        ground['note'] = ('wheel loads use the declared equal-stiffness distribution of the recovered ground wrench; '
                          'values are means over one record interval, the first 0.5 s is excluded')
    report['ground'] = ground

    # ------------------------------------------------------------------ clearance and contacts
    allowed = {'payload:pad', 'payload:guide', 'payload:mount'}
    report['clearance'] = dict(
        by_category={key_: dict(min_m=float(v_[0]), pair=[v_[1], v_[2]], t=v_[3], phase=v_[4]) for key_, v_ in sorted(worst_clear.items())},
        smallest_m=float(min((v_[0] for key_, v_ in worst_clear.items() if key_ not in allowed), default=np.nan)),
        note='the smallest value excludes the payload against its own supports and guides (pad, guide, mount); '
             'the clamp against the payload is excluded by construction')
    if contacts is not None:
        forb = [(c['t'], a, b, f_) for c in contacts for a, b, f_ in c['forbidden']]
        report['clearance']['forbidden_contacts'] = dict(
            count=len(forb), first=[dict(t=float(x[0]), a=x[1], b=x[2], force_N=float(x[3])) for x in forb[:10]],
            pairs=sorted({(x[1], x[2]) for x in forb}))
        report['clearance']['guide_contact_max_N'] = float(max((c['guides'] for c in contacts), default=0.))

    # ------------------------------------------------------------------ load transfer
    def window(names):
        sel = np.flatnonzero([name_of.get(int(x), '') in names for x in k])
        return sel

    transfer = {}
    for label, names, rest in (('pickup', ('latch', 'take_weight', 'raise'), mission['rests']['pickup']),
                               ('placement', ('lower', 'transfer', 'verify'), mission['rests']['place'])):
        sel = window(names)
        if not len(sel):
            continue
        entry = dict(t=t[sel][::5].tolist(), carried_fraction=A['carried'][sel][::5].tolist(),
                     share_command=np.nan_to_num(A['share'][sel][::5], nan=1.).tolist())
        if contacts is not None:
            pads = [p['id'] for p in comp.connections[rest]['pads']]
            static_of = {p['id']: p['id'] for p in comp.connections[rest]['pads']}
            names_plant = sorted({name for c in contacts for name in c['pads']})
            entry['pads_plant_N'] = {name: [contacts[j]['pads'].get(name, 0.) for j in sel[::5]] for name in names_plant}
            total = np.array([sum(contacts[j]['pads'].values()) for j in sel])
            carried = A['carried'][sel]
            entry['closure_error_N'] = _stats(np.abs(total + carried * weight - weight))
            entry['note'] = 'closure error: plant pad loads + wrist-sensor carried load - payload weight'
        transfer[label] = entry
    report['transfer'] = transfer

    # ------------------------------------------------------------------ consistency of the compiled model with the plant
    cons = dict(unbalanced_generalized_force=_stats(S['unbalanced']),
                payload_wrench_balance_N=_stats(np.asarray(S['balance'])[carry]),
                connection_gap_m=_stats(np.asarray(S['gap'])[~np.isnan(S['gap'])]) if np.any(~np.isnan(S['gap'])) else None,
                recovered_vs_sensor={})
    for c in grasps:
        m_, s_ = np.asarray(S['grasp_model'][c]), np.asarray(S['grasp_meas'][c])
        ok = ~np.isnan(m_[:, 0])
        if ok.any():
            d = m_[ok] - s_[ok]
            cons['recovered_vs_sensor'][c] = dict(force_N=_stats(np.linalg.norm(d[:, 3:], axis=1)),
                                                  moment_Nm=_stats(np.linalg.norm(d[:, :3], axis=1)))
    if contacts is not None and any(c['wheels'] for c in contacts):
        total_err, mean_model, mean_plant, trac_err, wheel_err, slide_err = [], {}, {}, [], [], []
        for rid, r in comp.robots.items():
            nw = len(r.wheels)
            g = np.asarray(S['ground'][rid])[settled]
            plant = np.array([[contacts[i]['wheels'].get((rid, w['id']), [0., 0.]) for w in r.wheels] for i in idx])[settled]
            total_err += list(np.abs(plant[:, :, 0].sum(axis=1) - g[:, 3]))
            if model.friction[rid]:
                # rigid-body mechanics leaves the side forces of the single wheels open; what can be compared is the
                # friction of the wheels that slide: the declared law at the compiled loads against the plant
                k_ = np.asarray(S['skid'][rid])[settled]
                speed, law = k_[:, 7:7 + nw], k_[:, 7 + nw:7 + 2 * nw]
                sliding = speed > 20. * model._tyre[rid]['slip_speed']
                if np.isfinite(plant[:, :, 1]).all() and sliding.any():
                    slide_err += list(np.abs(plant[:, :, 1] - law)[sliding])
            elif np.isfinite(plant[:, :, 1]).all():              # a plant may report normal loads only
                trac_err += list(np.abs(plant[:, :, 1] - np.abs(g[:, 4 + nw:4 + 2 * nw])).ravel())
            if r.ground == 'contact':                            # determinate wheel loads: compared wheel by wheel
                wheel_err += list(np.abs(plant[:, :, 0] - g[:, 4:4 + nw]).ravel())
            mean_model[rid] = g[:, 4:4 + nw].mean(axis=0).tolist()
            mean_plant[rid] = plant[:, :, 0].mean(axis=0).tolist()
        cons['total_normal_load_vs_plant_N'] = _stats(total_err)
        cons['wheel_traction_vs_plant_N'] = _stats(trac_err) if trac_err else None
        if slide_err:
            cons['sliding_friction_vs_plant_N'] = _stats(slide_err)
        cons['mean_wheel_normal_load_N'] = dict(model=mean_model, plant=mean_plant)
        if wheel_err:
            cons['wheel_normal_load_vs_plant_N'] = _stats(wheel_err)
            cons['note'] = ('the wheel contacts of these bases are statically determinate (passive joints free), so the model '
                            'gives the load of every wheel and it is compared wheel by wheel with the plant')
        else:
            cons['note'] = ('rigid-body mechanics determines the total ground force and its centre, not the load of each wheel; '
                            'the model distributes it with the declared equal-stiffness rule, so wheel loads are compared in the mean')
    if any(f'weld:{c}' in A for c in grasps):
        d = []
        for i in idx:
            mode = _mode(model, A['mode_grasps'][i], A['mode_rest'][i])
            if not mode.grasps:
                continue
            kin = tree.sweep(A['q'][i])
            Tp = kin[0][tree.index[comp.payload['body']]]
            for c in mode.grasps:
                ib, T1 = model._paths[c][0], comp.connections[c]['T1']
                Tt = kin[0][ib] @ T1
                w = A[f'weld:{c}'][i]
                m_tool = Tt[:3, :3].T @ (w[:3] - np.cross(Tt[:3, 3] - Tp[:3, 3], w[3:]))
                f_tool = Tt[:3, :3].T @ w[3:]
                d.append(np.r_[m_tool, f_tool] - model.sensor_to_grasp(robot_of[c], A[sensor_of[c]][i], A['q'][i], kin,
                                                                      A['v'][i], acc[i]))
        d = np.asarray(d)
        if len(d):
            cons['sensor_vs_solver_latch'] = dict(force_N=_stats(np.linalg.norm(d[:, 3:], axis=1)),
                                                  moment_Nm=_stats(np.linalg.norm(d[:, :3], axis=1)))
    # measured displacement against the sum of the measured rates of the same steps (a plant that integrates
    # in single precision moves in quanta and can stall at low speed; this is where it shows)
    if len(t) > 2 and 'v_mean' in A:
        step = np.diff(t)[:, None]
        drift = np.diff(A['q'], axis=0) - A['v_mean'][:-1] * step
        lin = np.concatenate([r.base[:3] for r in comp.robots.values()] + [np.asarray(comp.payload['base'][:3])])
        ang = np.concatenate([r.base[3:] for r in comp.robots.values()] + [np.asarray(comp.payload['base'][3:])])
        joints = np.setdiff1d(np.arange(A['q'].shape[1]), np.r_[lin, ang])
        cons['state_integration'] = dict(
            floating_translation_m=_stats(np.abs(drift[:, lin]).max(axis=1)),
            floating_rotation_rad=_stats(np.abs(drift[:, ang]).max(axis=1)),
            joints=_stats(np.abs(drift[:, joints]).max(axis=1)), interval_s=float(np.median(step)),
            note='largest |q(t+T) - q(t) - T mean(v after each step)| over the coordinates, per record interval T')
    report['consistency'] = cons

    # ------------------------------------------------------------------ declared bounds
    fmax = max((v_['force_N']['max'] or 0.) for v_ in grasp['by_grasp'].values())
    mmax = max((v_['moment_Nm']['max'] or 0.) for v_ in grasp['by_grasp'].values())
    def check(value, bound, at_most=True):
        if isinstance(value, (float, np.floating)) and np.isinf(value):
            # an infinite traction use (a wheel without load) is not a pass
            return dict(value=None, bound=bound, ok=False, note=f'not a finite value ({value})')
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return dict(value=None, bound=bound, ok=True, note='not exercised in this run')
        return dict(value=value, bound=bound, ok=bool(value <= bound if at_most else value >= bound))

    robots_ground = [g for g in ground.values() if isinstance(g, dict)]
    checks = dict(
        grasp_force=check(fmax, bounds['grasp_force_N']),
        grasp_moment=check(mmax, bounds['grasp_moment_Nm']),
        effort_ratio=check(report['efforts']['max_ratio'], bounds['effort_ratio']),
        traction=check(max(g.get('net_traction_utilization_max', g.get('max_traction_utilization')) for g in robots_ground),
                       bounds['traction_utilization']),
        tipping=check(min(g['min_tipping_margin_m'] for g in robots_ground), bounds['tipping_margin_m'], False),
        wheel_load=check(min(g['min_wheel_load_N'] for g in robots_ground), bounds['min_wheel_load_N'], False),
        clearance=check(report['clearance']['smallest_m'], bounds['clearance_m'], False),
        payload_tilt=check(tr['payload_tilt_rad']['max'], bounds['payload_tilt_rad']),
        payload_position=check(tr['payload_position_error_m']['max'], bounds['payload_position_m']))
    if contacts is not None:
        n_forb = report['clearance']['forbidden_contacts']['count']
        checks['no_forbidden_contact'] = dict(value=n_forb, bound=0, ok=n_forb == 0)
    checks['mission_complete'] = dict(value=info['status'], bound='complete', ok=info['status'] == 'complete')
    checks['supports_verified'] = dict(value=bool(info.get('verification', {}).get('accepted', False)), bound=True,
                                       ok=bool(info.get('verification', {}).get('accepted', False)))
    report['bounds'] = dict(checks=checks, all_ok=bool(all(c['ok'] for c in checks.values())))
    report['timing'] = info.get('wall', {})
    series = dict(t=ts, carry=carry, internal_force=np.asarray(S['internal_force']), internal_moment=np.asarray(S['internal_moment']),
                  ground={rid: np.asarray(v_) for rid, v_ in S['ground'].items()},
                  skid={rid: np.asarray(v_) for rid, v_ in S['skid'].items()},
                  share={c: np.asarray(v_) for c, v_ in S['share'].items()},
                  grasp_meas={c: np.asarray(v_) for c, v_ in S['grasp_meas'].items()},
                  grasp_model={c: np.asarray(v_) for c, v_ in S['grasp_model'].items()},
                  clearance=S['clearance'], pads_model=S['pads_model'])
    return report, series


def _n(x, scale=1., fmt='.1f'):
    """A number of the report, or a dash when the run did not exercise it."""
    return '-' if x is None else format(scale * x, fmt)


def summary_lines(report):
    """Short human-readable summary of an audit report."""
    r = report
    out = [f"status: {r['status']}  (simulated {r['sim_time_s']:.1f} s)"]
    tr = r['tracking']
    out.append(f"payload while carried: position error max {_n(tr['payload_position_error_m']['max'], 1e3)} mm, "
               f"tilt max {_n(tr['payload_tilt_rad']['max'], 1e3)} mrad, yaw error max {_n(tr['payload_yaw_error_rad']['max'], 1e3)} mrad")
    out.append('base tracking: ' + ', '.join(f"{rid} {_n(b['position_error_m']['max'], 1e3)} mm / {_n(b['yaw_error_rad']['max'], 1e3)} mrad"
                                             for rid, b in tr['base'].items()))
    g = r['grasp']
    out.append(f"internal wrench while carried: force max {_n(g['internal_force_N']['max'])} N (rms {_n(g['internal_force_N']['rms'])}), "
               f"moment max {_n(g['internal_moment_Nm']['max'])} N m (rms {_n(g['internal_moment_Nm']['rms'])})")
    for c, v in g['by_grasp'].items():
        sh = v['share_while_carrying']
        share = f"vertical share {sh['mean']:.3f} [{sh['min']:.3f}, {sh['max']:.3f}]" if sh else 'not carrying'
        out.append(f"  {c}: force max {_n(v['force_N']['max'])} N, moment max {_n(v['moment_Nm']['max'])} N m, {share}")
    sr = g.get('sensor_reading', {}).get('by_robot')
    if sr:
        rng = next((v['range'] for v in sr.values() if v.get('range')), None)
        out.append('  wrist sensor readings, largest: ' + ', '.join(f"{rid} {v['force_N']:.1f} N / {v['moment_Nm']:.1f} N m" for rid, v in sr.items())
                   + (f" (declared range {rng['force_N']:.0f} N / {rng['torque_Nm']:.0f} N m)" if rng else ''))
    out.append(f"actuator demand: max ratio {r['efforts']['max_ratio']:.2f} ({r['efforts']['worst']['port']}), "
               f"saturated steps {r['efforts']['saturated_steps']}")
    for rid, v in r['ground'].items():
        if rid == 'note':
            continue
        if rid == 'note_friction_drive':
            continue
        hr = v.get('against_rigid_wheels')
        if 'net_traction_utilization_max' in v:
            traction = (f"net traction utilization max {v['net_traction_utilization_max']:.3f} "
                        f"(p99 {v['net_traction_utilization_p99']:.3f})")
        else:
            traction = f"traction utilization max {v['max_traction_utilization']:.3f} (p99 {v['traction_utilization_p99']:.3f})"
        out.append(f"  {rid}: min wheel load {v['min_wheel_load_N']:.0f} N, {traction}, "
                   f"tipping margin min {v['min_tipping_margin_m']:.3f} m"
                   + (f", base height vs rigid wheels {1e3 * hr['height_m']['mean']:+.2f} mm [{1e3 * hr['height_m']['min']:+.2f}, "
                      f"{1e3 * hr['height_m']['max']:+.2f}]" if hr else '')
                   + ''.join(f", {name} [{1e3 * pj['min']:+.1f}, {1e3 * pj['max']:+.1f}] mrad of +-{1e3 * pj['limits'][1]:.0f}"
                             for name, pj in v.get('passive_joints', {}).items()))
        sk = v.get('skid')
        if sk:
            am = sk.get('against_skid_model')
            out.append(f"    {rid} skid steering: turned {sk['turned_rad']:.1f} rad in all, wheel slip up to "
                       f"{1e3 * sk['max_slip_speed_m_s']:.0f} mm/s, friction power max {sk['friction_power_W']['max']:.0f} W, "
                       f"energy {1e-3 * sk['friction_energy_J']:.2f} kJ"
                       + (f"; against the skid model: yaw rate ratio {am['yaw_rate_ratio']:.3f}, error max "
                          f"{1e3 * am['yaw_rate_rad_s']['max']:.1f} mrad/s (rms {1e3 * am['yaw_rate_rad_s']['rms']:.1f}), "
                          f"sideways max {1e3 * am['sideways_m_s']['max']:.1f} mm/s (rms {1e3 * am['sideways_m_s']['rms']:.1f})"
                          if am else '')
                       + (f"; one side of the base without wheel load in {sk['skid_model_not_evaluated']['samples']} of "
                          f"{sk['skid_model_not_evaluated']['of']} samples (no skid model there)"
                          if sk.get('skid_model_not_evaluated') else ''))
    cl = r['clearance']
    out.append(f"smallest clearance {_n(cl['smallest_m'], fmt='.3f')} m; by category: "
               + ', '.join(f"{key} {v['min_m']:.3f}" for key, v in cl['by_category'].items()))
    if 'forbidden_contacts' in cl:
        out.append(f"forbidden contacts reported by the plant: {cl['forbidden_contacts']['count']} {cl['forbidden_contacts']['pairs'][:4]}")
    c = r['consistency']
    for key, v in c.get('recovered_vs_sensor', {}).items():
        out.append(f"  compiled reaction recovery vs wrist sensor {key}: force max {_n(v['force_N']['max'])} N (rms {_n(v['force_N']['rms'])}), "
                   f"moment max {_n(v['moment_Nm']['max'])} N m (rms {_n(v['moment_Nm']['rms'])})")
    if 'total_normal_load_vs_plant_N' in c:
        wt = c.get('wheel_traction_vs_plant_N')
        wn = c.get('wheel_normal_load_vs_plant_N')
        out.append(f"  compiled ground loads vs plant contacts: total normal max {_n(c['total_normal_load_vs_plant_N']['max'])} N "
                   f"(rms {_n(c['total_normal_load_vs_plant_N']['rms'])}), "
                   + (f"each wheel normal max {wn['max']:.1f} N (rms {wn['rms']:.1f}), " if wn else '')
                   + (f"wheel traction max {wt['max']:.1f} N (rms {wt['rms']:.1f})" if wt else
                      ('wheel traction not reported by the plant' if 'sliding_friction_vs_plant_N' not in c else 'friction of sliding '
                       f"wheels max {c['sliding_friction_vs_plant_N']['max']:.1f} N (rms {c['sliding_friction_vs_plant_N']['rms']:.1f})")))
        for rid in c['mean_wheel_normal_load_N']['model']:
            out.append(f"    {rid} mean wheel loads (N): model {np.round(c['mean_wheel_normal_load_N']['model'][rid], 0).tolist()} "
                       f"plant {np.round(c['mean_wheel_normal_load_N']['plant'][rid], 0).tolist()}")
    si = c.get('state_integration')
    if si:
        out.append(f"  displacement vs summed rates per {1e3 * si['interval_s']:.0f} ms: floating bodies max "
                   f"{_n(si['floating_translation_m']['max'], 1e6, '.2f')} um / {_n(si['floating_rotation_rad']['max'], 1e6, '.2f')} urad, "
                   f"joints max {_n(si['joints']['max'], 1e6, '.2f')} u(rad|m)")
    v = r.get('verification') or {}
    if v:
        out.append(f"supports verified: {v.get('accepted')}  checks {v.get('checks')}  placement error {1e3 * v.get('position_error_m', 0):.2f} mm, "
                   f"{1e3 * v.get('rotation_error_rad', 0):.2f} mrad")
    bad = [key for key, ch in r['bounds']['checks'].items() if not ch['ok']]
    out.append('declared bounds: ' + ('all met' if not bad else 'NOT met: ' + ', '.join(
        f"{key} ({r['bounds']['checks'][key]['value']} vs {r['bounds']['checks'][key]['bound']})" for key in bad)))
    return out
