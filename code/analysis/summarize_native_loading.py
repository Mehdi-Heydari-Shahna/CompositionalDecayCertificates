"""Section 7.6: the repeated F/2 MuJoCo missions of run_native_loading.py against the recorded ones (missions.json of the
numerical-checks folder): changes of the peak and rms payload errors and of the grasp-moment maxima (20 ms means), and
the per-step grasp moments (maxima, and the share of latched grasp steps above the declared bound).
Writes native_loading_summary.json.
"""
import json

from teamdata import CHK, RESULTS

TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3')

if __name__ == '__main__':
    rec = json.loads((CHK / 'missions.json').read_text())
    out = {}
    for t in TEAMS:
        r = json.loads((RESULTS / f'native_loading_{t}_fitted05.json').read_text())
        m = rec[f'mujoco_{t}_fitted05']
        steps = sum(v['steps'] for v in r['by_grasp'].values())
        above = sum(v['steps_above_bound'] for v in r['by_grasp'].values())
        out[t] = dict(recorded=dict(payload_mm=m['payload_mm'], payload_rms_mm=m['payload_rms_mm'], grasp_moment_max=m['grasp_moment_max']),
                      repeated=dict(payload_mm=r['payload_mm'], payload_rms_mm=r['payload_rms_mm'], grasp_moment_max=r['audit_grasp_moment_max'],
                                    grasp_moment_max_20ms_vector_mean=r['moment_max_20ms_mean']),
                      change=dict(payload_peak=r['payload_mm'] / m['payload_mm'] - 1, payload_rms=r['payload_rms_mm'] / m['payload_rms_mm'] - 1,
                                  grasp_moment_max=r['audit_grasp_moment_max'] / m['grasp_moment_max'] - 1),
                      per_step=dict(moment_max=r['moment_max_per_step'], bound=r['grasp_moment_bound'],
                                    share_of_latched_grasp_steps_above_bound=above / steps,
                                    ratio_to_20ms_max=r['moment_max_per_step'] / r['audit_grasp_moment_max']))
    rng = lambda key, sub: [min(out[t][key][sub] for t in TEAMS), max(out[t][key][sub] for t in TEAMS)]
    out['ranges'] = dict(change_payload_peak=rng('change', 'payload_peak'), change_payload_rms=rng('change', 'payload_rms'),
                         change_grasp_moment_max=rng('change', 'grasp_moment_max'))
    (RESULTS / 'native_loading_summary.json').write_text(json.dumps(out, indent=1))
    for t in TEAMS:
        o = out[t]
        print(f"{t:11s} peak {o['recorded']['payload_mm']:.3f}->{o['repeated']['payload_mm']:.3f} ({100*o['change']['payload_peak']:+.1f}%)"
              f"  rms {o['recorded']['payload_rms_mm']:.4f}->{o['repeated']['payload_rms_mm']:.4f} ({100*o['change']['payload_rms']:+.1f}%)"
              f"  moment20ms {o['recorded']['grasp_moment_max']:.2f}->{o['repeated']['grasp_moment_max']:.2f} ({100*o['change']['grasp_moment_max']:+.1f}%)"
              f"  per-step max {o['per_step']['moment_max']:.2f} (bound {o['per_step']['bound']:.0f}, above {100*o['per_step']['share_of_latched_grasp_steps_above_bound']:.2f}%)")
    print(json.dumps(out['ranges']))
