"""Shared fixtures.  The package root goes on the path; optional engines are skipped where they are missing."""
import os
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

# Started in the background of a shell script, pytest is told to ignore SIGINT, and every process the tests start
# would inherit that, so it gets its handler back here.
if os.name != 'nt' and signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
    signal.signal(signal.SIGINT, signal.default_int_handler)

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from teamc.interfaces import TeamModel  # noqa: E402
from teamc.io import INPUTS, load_team, read_json  # noqa: E402

MECANUM_TEAMS = ('team2', 'team3', 'ridgeback2', 'ridgeback3')  # bases that roll: RB-ROBOUT+ / UR30, Ridgeback / UR10e
SKID_TEAMS = ('husky2', 'husky3')                               # bases that turn by sliding their wheels: Husky A300 / UR10e
TEAMS = MECANUM_TEAMS + SKID_TEAMS                              # three platforms, teams of two and three


@pytest.fixture(scope='session', params=TEAMS)
def team(request):
    return request.param


@pytest.fixture(scope='session', params=MECANUM_TEAMS)
def mecanum_team(request):
    return request.param


@pytest.fixture(scope='session', params=SKID_TEAMS)
def skid_team(request):
    return request.param


@pytest.fixture(scope='session')
def compiled():
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = load_team(name)
        return cache[name]
    return get


@pytest.fixture(scope='session')
def missions():
    return lambda name: read_json(INPUTS / 'missions' / f'{name}.json')


@pytest.fixture()
def comp2():
    """A fresh compilation of team2 (tests that register grasp frames must not share it)."""
    return load_team('team2')


@pytest.fixture(scope='session')
def closed():
    """Closed configurations of a team: the rack station and a carrying configuration."""
    cache = {}

    def get(name):
        if name not in cache:
            from mission.controller import TeamController
            from mission.runner import carry_configuration
            from teamc.chords import TeamAssembler
            comp = load_team(name)
            mission = read_json(INPUTS / 'missions' / f'{name}.json')
            ctrl = TeamController(comp, mission, .002)
            station = TeamAssembler(comp, comp.seed()).solve(comp.seed())[0]
            cache[name] = dict(comp=comp, mission=mission, model=TeamModel(comp), station=station,
                               carry=carry_configuration(comp, mission, ctrl, 2.0), ctrl=ctrl)
        return cache[name]
    return get


def rng(seed=0):
    return np.random.default_rng(seed)
