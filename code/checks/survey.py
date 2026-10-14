"""Survey of the six compiled teams: dimensions, rank tests, redundancy (reads the package, changes nothing)."""
import sys, time, json
from pathlib import Path
import numpy as np
import os
ROOT = Path(os.environ.get('TEAM_CMG_PACDM', Path(__file__).resolve().parent.parent / 'Team_CMG_PACDM')).resolve()
if not (ROOT / 'teamc').is_dir():
    raise SystemExit(f'framework package not found at {ROOT}: set the environment variable TEAM_CMG_PACDM to its folder')
sys.path.insert(0, str(ROOT))
RESULTS = Path(os.environ.get('CHECKS_RESULTS', Path(__file__).resolve().parents[2] / 'results' / 'checks')).resolve()
RESULTS.mkdir(parents=True, exist_ok=True)
from teamc.interfaces import TeamModel
from teamc.io import INPUTS, load_team, read_json
from mission.controller import TeamController
from mission.runner import carry_configuration

out = {}
for name in ('team2', 'team3', 'ridgeback2', 'ridgeback3', 'husky2', 'husky3'):
    t0 = time.time()
    comp = load_team(name)
    mission = read_json(INPUTS / 'missions' / f'{name}.json')
    ctrl = TeamController(comp, mission, .002)
    q = carry_configuration(comp, mission, ctrl, 2.0)
    model = TeamModel(comp)
    mode = model.mode()
    lift = model.lift(q, mode)
    con = lift.constraints
    ident = model.identifiability(q, mode, con)
    es = model.effort_split(q, mode, lift)
    info = dict(n=int(model.n), nf=int(model.nf), d=int(lift.E.shape[1]), rows=int(con.J.shape[0]),
                chart=int(con.Jc.shape[1]), rank_dep=int(lift.rank), redundant_lift=int(lift.redundant_rows),
                rankL=int(es['rank']), dim_kerLT=int(es['dimension']), ident_rank=ident['rank'], ident_red=ident['redundant_rows'],
                residual=float(np.max(np.abs(con.residual))),
                blocks=[(b['id'], b['kind'], b['rows'].stop - b['rows'].start, bool(b.get('slip', False))) for b in con.blocks],
                determinate={k: v['determinate'] for k, v in ident['ports'].items()},
                ground={rid: r.ground for rid, r in comp.robots.items()}, drive={rid: r.drive for rid, r in comp.robots.items()},
                independent=[model.ids[i] for i in lift.independent], seconds=round(time.time() - t0, 1))
    out[name] = info
    print(name, json.dumps({k: v for k, v in info.items() if k != 'independent'}), flush=True)
    print('   independent:', info['independent'], flush=True)
    np.save(RESULTS / f'q_carry_{name}.npy', q)
(RESULTS / 'survey.json').write_text(json.dumps(out, indent=1))
