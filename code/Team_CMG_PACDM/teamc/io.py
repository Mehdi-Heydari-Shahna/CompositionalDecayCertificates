"""Loading of the physical input records and JSON output helpers (standard library and NumPy only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compiler import compile_team

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'inputs'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_plain(obj), indent=indent, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    return path


def to_plain(obj):
    """Convert NumPy containers and non-finite floats to plain JSON values."""
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_plain(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, slice):
        return [obj.start, obj.stop]
    return obj


def team_path(name):
    """Resolve a team name (``team2``) or a path to a team record."""
    p = Path(name)
    if p.suffix == '.json' and p.exists():
        return p
    q = INPUTS / 'teams' / f'{name}.json'
    if not q.exists():
        raise FileNotFoundError(f'No team record {name!r} (looked for {q})')
    return q


def load_records(name):
    """Team record and the payload, scene and robot records it names (paths relative to ``inputs/``)."""
    path = team_path(name)
    team = read_json(path)
    files = team.get('files')
    if not files:
        raise ValueError(f'{path}: the team record must name its payload, scene and robot files')
    base = path.resolve().parents[1]
    payload = read_json(base / files['payload'])
    scene = read_json(base / files['scene'])
    robots = {model: read_json(base / rel) for model, rel in files['robots'].items()}
    return team, robots, payload, scene


def load_team(name):
    """Compile a team from its input records."""
    team, robots, payload, scene = load_records(name)
    return compile_team(team, robots, payload, scene)
