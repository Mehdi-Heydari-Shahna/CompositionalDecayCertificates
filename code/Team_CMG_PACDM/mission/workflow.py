"""Run a mission on a plant, audit it and write the report (shared by all workflows)."""
from __future__ import annotations

import time
from pathlib import Path

from .audit import audit
from .report import save, save_record
from .runner import run_mission


def run_and_report(plant, comp, mission, out, title, options=None, quiet=False, on_record=None, plant_info=None, figures=True,
                   max_time=400.):
    """Returns ``(report, record arrays, recorder, controller, info)``.

    ``figures``: ``True``, ``False`` or ``'auto'`` (only where matplotlib is installed).  The report,
    the summary and the raw record are always written.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    def progress(t, ctrl):
        if not quiet:
            print(f'  t {t:6.1f} s  {ctrl.phase:13s} carried {ctrl.carried:6.3f}  l {ctrl.governor.l:6.2f} m  '
                  f'wall {time.perf_counter() - t0:6.1f} s', flush=True)

    ctrl, rec, info = run_mission(plant, comp, mission, options=options, progress=progress, on_record=on_record, max_time=max_time)
    A = rec.arrays()
    info['plant'] = dict(name=getattr(plant, 'name', '?'), timestep=plant.dt, **(plant_info or {}))
    contacts = rec.contacts if rec.contacts else None
    try:
        report, series = audit(comp, mission, A, info, contacts, model=ctrl.model)
    except Exception:
        save_record(out / 'record.npz', A, info, contacts)          # a run is not lost to a failure of its audit
        print('the audit failed; the record is in', out / 'record.npz', flush=True)
        raise
    report['plant'] = info['plant']
    report['team'] = mission['team']
    lines, files = save(out, comp, mission, A, report, series, info, contacts, title=title, with_figures=figures)
    print('\n'.join(lines), flush=True)
    print('written to', out, flush=True)
    return report, A, rec, ctrl, info
