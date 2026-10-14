"""Table 3 and the step-refinement statements of Section 7.6 from the carrying-segment runs of run_segment_compare.py.

Reads segment_<team>.json (ideal-constraint plant) and segment_mujoco_<team>.json (MuJoCo) and reports, over the 4 s of
motion: peak and rms payload position error at every plant step; at the finest step, the reductions of F/2 and F
relative to D; the change of each value from the 2 ms step to the finest step and between the two finest steps; and the
change of the ratios F/D and F/2 over D between the coarsest and the finest step. Writes segment_summary.json.
"""
import json
import os
from pathlib import Path

RESULTS = Path(os.environ.get('ANALYSIS_RESULTS', Path(__file__).resolve().parents[2] / 'results' / 'analysis')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3')
PLANTS = {'ideal': 'segment_{}.json', 'mujoco': 'segment_mujoco_{}.json'}


def load(plant, team):
    rows = json.loads((RESULTS / PLANTS[plant].format(team)).read_text())
    out = {}
    for r in rows:
        out.setdefault(r['dt'], {})[r['variant']] = (r['peak_moving_mm'], r['rms_moving_mm'])
    return out


def rel(a, b):
    return a / b - 1.


if __name__ == '__main__':
    summary = {}
    for plant in PLANTS:
        summary[plant] = {}
        for team in TEAMS:
            try:
                data = load(plant, team)
            except FileNotFoundError:
                continue
            steps = sorted(data, reverse=True)                      # coarse to fine
            complete = [dt for dt in steps if len(data[dt]) == 3]
            if not complete:
                continue
            fine, coarse = complete[-1], complete[0]
            prev = complete[-2] if len(complete) > 1 else None
            f = data[fine]
            s = dict(steps=complete, finest=fine,
                     table={v: dict(peak=f[v][0], rms=f[v][1]) for v in f},
                     reduction_F2_vs_D=dict(peak=1 - f['F/2'][0] / f['D'][0], rms=1 - f['F/2'][1] / f['D'][1]),
                     reduction_F_vs_D=dict(peak=1 - f['F'][0] / f['D'][0], rms=1 - f['F'][1] / f['D'][1]),
                     change_coarse_to_fine={v: dict(peak=rel(f[v][0], data[coarse][v][0]), rms=rel(f[v][1], data[coarse][v][1]))
                                            for v in f},
                     ratio_change_coarse_to_fine={v: dict(peak=rel(f[v][0] / f['D'][0], data[coarse][v][0] / data[coarse]['D'][0]),
                                                          rms=rel(f[v][1] / f['D'][1], data[coarse][v][1] / data[coarse]['D'][1]))
                                                  for v in ('F', 'F/2')})
            if prev is not None:
                s['change_last_halving'] = {v: dict(peak=rel(f[v][0], data[prev][v][0]), rms=rel(f[v][1], data[prev][v][1]))
                                            for v in f}
            s['all'] = {str(dt): {v: dict(peak=data[dt][v][0], rms=data[dt][v][1]) for v in data[dt]} for dt in steps}
            s['ordering_F2_F_D_all_steps'] = all(data[dt]['F/2'][i] < data[dt]['F'][i] < data[dt]['D'][i]
                                                 for dt in complete for i in (0, 1))
            summary[plant][team] = s
    # ranges over the teams, as stated in the text
    ranges = {}
    for plant, teams in summary.items():
        if not teams:
            continue
        def span(values):
            return [min(values), max(values)]
        tt = teams.values()
        ranges[plant] = dict(
            teams=list(teams), finest=sorted({s['finest'] for s in tt}),
            reduction_F2_vs_D_peak=span([s['reduction_F2_vs_D']['peak'] for s in tt]),
            reduction_F2_vs_D_rms=span([s['reduction_F2_vs_D']['rms'] for s in tt]),
            reduction_F_vs_D_peak=span([s['reduction_F_vs_D']['peak'] for s in tt]),
            reduction_F_vs_D_rms=span([s['reduction_F_vs_D']['rms'] for s in tt]),
            change_coarse_to_fine=span([c[k] for s in tt for c in s['change_coarse_to_fine'].values() for k in c]),
            ratio_change_coarse_to_fine_abs_max=max(abs(c[k]) for s in tt for c in s['ratio_change_coarse_to_fine'].values() for k in c),
            change_last_halving=span([c[k] for s in tt if 'change_last_halving' in s for c in s['change_last_halving'].values() for k in c]),
            change_last_halving_abs_max=max([abs(c[k]) for s in tt if 'change_last_halving' in s for c in s['change_last_halving'].values() for k in c] or [0.]),
            ordering_F2_F_D_all_runs=all(s['ordering_F2_F_D_all_steps'] for s in tt))
    summary['ranges'] = ranges
    (RESULTS / 'segment_summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(ranges, indent=1))
    summary.pop('ranges')
    for plant, teams in summary.items():
        print(f'== {plant}')
        for team, s in teams.items():
            t = s['table']
            print(f"{team:11s} finest {s['finest']*1e3:g} ms  D {t['D']['peak']:.3f}/{t['D']['rms']:.3f}  F {t['F']['peak']:.3f}/{t['F']['rms']:.3f}"
                  f"  F/2 {t['F/2']['peak']:.3f}/{t['F/2']['rms']:.3f}   F/2 vs D -{100*s['reduction_F2_vs_D']['peak']:.1f}/-{100*s['reduction_F2_vs_D']['rms']:.1f}%"
                  f"   F vs D -{100*s['reduction_F_vs_D']['peak']:.1f}/-{100*s['reduction_F_vs_D']['rms']:.1f}%")
            cc = s['change_coarse_to_fine']
            print('            2 ms -> finest peak change: ' + ', '.join(f"{v} {100*cc[v]['peak']:+.1f}%" for v in cc)
                  + '   rms: ' + ', '.join(f"{v} {100*cc[v]['rms']:+.1f}%" for v in cc))
            rc = s['ratio_change_coarse_to_fine']
            print('            ratio to D change: ' + ', '.join(f"{v} peak {100*rc[v]['peak']:+.1f}% rms {100*rc[v]['rms']:+.1f}%" for v in rc))
            if 'change_last_halving' in s:
                lh = s['change_last_halving']
                print('            last halving: ' + ', '.join(f"{v} {100*lh[v]['peak']:+.2f}%/{100*lh[v]['rms']:+.2f}%" for v in lh))
