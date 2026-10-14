"""Team compiler: joins robot mechanism graphs through a shared payload body.

New extension of the RoboCompiler packages.  The PACDM kernel in ``vendor/`` is
used unchanged.  Nothing in this package imports a simulator; MuJoCo, Pinocchio
and Isaac Sim are reached only through ``backends/``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # make ``vendor`` importable when used as a plain folder
    sys.path.insert(0, str(ROOT))

SCHEMA_ROBOT = 'cmg.physical-mobile-manipulator/1.0'
SCHEMA_PAYLOAD = 'cmg.physical-payload/1.0'
SCHEMA_SCENE = 'cmg.physical-scene/1.0'
SCHEMA_TEAM = 'cmg.physical-team-graph/1.0'
SCHEMA_COMPILED = 'cmg.team.compiled-graph/1.0'
