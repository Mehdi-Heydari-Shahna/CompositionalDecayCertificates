#!/usr/bin/env python
"""Isaac Sim verification campaign: the staged workflow for the teams of the three platforms, in one command.

    python workflows/isaac/run_campaign.py --headless                                  # with the Python of Isaac Sim
    python workflows/isaac/run_campaign.py --headless --isaac-python <Isaac's python>  # from another Python 3
    python workflows/isaac/run_campaign.py --headless --quick                          # everything but the missions
    python workflows/isaac/run_campaign.py --headless --resume results/isaac_campaign_<id>

What it does, in this order:

    environment    one start of Isaac Sim that builds nothing and lists what the workflow uses of the installation
                   (the campaign ends here if the application does not start or its Python lacks NumPy or SciPy)
    quick stages   selftest, preflight, settle and carry of every team, the teams of two robots first
    missions       the complete mission of every team whose quick stages passed
    assessment     ASSESSMENT.md and assessment.json (workflows/isaac/assess_results.py)
    bundle         campaign_bundle.zip: the records of every stage, without scenes and time series

The teams: team2, team3 (RB-ROBOUT+ / UR30), ridgeback2, ridgeback3 (Ridgeback / UR10e), husky2, husky3
(Husky A300 / UR10e).  Every team has a run folder of its own inside the campaign folder, written by
``run_isaac.py`` exactly as for a single team.  The quick stages of a team are all run, also after one of
them failed: each has something to tell.  The mission of a team is run only if its four quick stages
passed.  A failure of one team does not hold the others, and the assessment and the bundle are written
whatever happened.

``--resume`` continues a campaign: it runs what has not passed yet (after ``--quick``: the missions; after an
interruption or a failure: the stages that are left) and leaves the stages that passed alone, provided they
were made with the files the package has now.  A stage made with other files is run again.

Options after ``--`` go to every stage: ``-- --solver TGS``, ``-- --contact rigid``, ``-- --dt 0.001``
(see ``run_isaac.py --help``).  A campaign is made with one set of them.  Without ``--contact`` the wheel
contact is ``auto``: the declared tyre springs carry the conventional wheels of the Husky bases, and the
rollers of the mecanum wheels stand on rigid PhysX contacts; ``-- --contact rigid`` stands the Husky wheels
on rigid PhysX contacts as well.

Exit status: 0 every stage asked for passed; 1 a stage failed or was not run; 130 interrupted; 2 arguments.
Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import assess_results  # noqa: E402
import run_isaac  # noqa: E402
from run_isaac import STAGES, latest_stages, save_json, utc  # noqa: E402

QUICK = STAGES[:-1]
TEAMS = assess_results.TEAMS                                             # the teams of two robots first
PLATFORMS = dict(rbrobout=('team2', 'team3'), ridgeback=('ridgeback2', 'ridgeback3'), husky=('husky2', 'husky3'))
# options of run_isaac.py that the campaign sets itself
RESERVED = ('--team', '--stage', '--stages', '--run-folder', '--output', '--isaac-python', '--isaac-root', '--headless',
            '--no-bundle', '--worker', '--run-id', '--make-reference')
NOT_BUNDLED = ('.usda', '.usdc', '.zip', '.tmp', '.npz', '.mp4')
GRACE_S = 30.                                                            # for stages to end by themselves after an interruption


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ====================================================================== arguments
def parse_args(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if '--' in argv:
        extra = argv[argv.index('--') + 1:]
        argv = argv[:argv.index('--')]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument('--teams', nargs='+', choices=TEAMS, metavar='TEAM', help='teams to run (default: all six, or those of --resume)')
    p.add_argument('--platforms', nargs='+', choices=sorted(PLATFORMS), metavar='PLATFORM',
                   help='only the teams of these platforms: ' + ', '.join(sorted(PLATFORMS)))
    p.add_argument('--quick', action='store_true', help='the stages before the mission only (selftest, preflight, settle, carry)')
    p.add_argument('--stages', nargs='+', choices=STAGES, metavar='STAGE', help='only these stages (default: all five)')
    p.add_argument('--headless', action='store_true', help='no Isaac window')
    p.add_argument('--output', type=Path, default=ROOT / 'results', help='parent folder of the new campaign folder')
    p.add_argument('--resume', type=Path, default=None, metavar='FOLDER', help='continue the campaign in this folder')
    p.add_argument('--jobs', type=int, default=1,
                   help='teams run at the same time (default 1; every job is an Isaac Sim process of its own)')
    runtime = p.add_mutually_exclusive_group()
    runtime.add_argument('--isaac-python', type=Path, help='Python executable of an Isaac Sim environment (default: this Python)')
    runtime.add_argument('--isaac-root', type=Path, help='Isaac Sim folder containing python.bat / python.sh')
    p.add_argument('--with-records', action='store_true',
                   help='also write campaign_records.zip with the time series of every stage (tens of megabytes per mission)')
    a = p.parse_args(argv)
    if a.quick and a.stages:
        p.error('give --quick or --stages, not both')
    if a.jobs < 1:
        p.error('--jobs must be at least 1')
    for token in extra:
        if token.split('=')[0] in RESERVED:
            p.error(f'{token.split("=")[0]} is set by the campaign and cannot be passed on after --')
    try:
        run_isaac.parse_args(['--team', TEAMS[0], *extra])
    except SystemExit:
        p.error('the options after -- are not options of run_isaac.py (see above)')
    a.teams_given = bool(a.teams or a.platforms)
    selected = [t for t in TEAMS if (not a.teams or t in a.teams)
                and (not a.platforms or any(t in PLATFORMS[name] for name in a.platforms))]
    if not selected:
        p.error('no team is left: --teams and --platforms name different teams')
    a.teams = selected
    return a, extra


# ====================================================================== state of a team
def digest(fingerprints):
    return hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode('utf-8')).hexdigest()


def stage_states(folder, fingerprints):
    """State of every stage of the run in ``folder``: ``passed``, ``failed``, ``not run``, or ``other files``
    (passed, but with files of the package that differ from the present ones)."""
    states = {stage: 'not run' for stage in STAGES}
    try:
        manifest = read_json(Path(folder) / 'manifest.json')
    except (OSError, ValueError):
        return states
    for stage, entry in latest_stages(manifest).items():
        if stage not in states:
            continue
        status = entry.get('status')
        if status == 'passed':
            try:
                case = read_json(Path(folder) / stage / 'case.json')
            except (OSError, ValueError):
                case = {}
            if case.get('passed') is not True:
                status = 'failed'
            elif case.get('source_sha256') != fingerprints:
                status = 'other files'
        elif status == 'running':                                       # the coordinator of that run did not get to finish
            status = 'failed'
        states[stage] = status if status in ('passed', 'failed', 'other files') else 'not run'
    return states


def pending(states, wanted):
    """The stages of ``wanted`` that are left to run, in the order of the workflow."""
    return [stage for stage in STAGES if stage in wanted and states[stage] != 'passed']


def environment_outcome(folder):
    """``(kind, text)`` of the environment run: ``passed``; ``missing`` (the installation lacks something the workflow
    uses, or the lookup itself failed); ``no packages`` (its Python lacks NumPy or SciPy: nothing of the workflow can
    run); or ``no start`` (the application did not come up: no other stage can run)."""
    try:
        case = read_json(Path(folder) / 'environment' / 'case.json')
    except (OSError, ValueError):
        return 'no start', 'the stage left no record (see environment/logs/environment.log)'
    kind, engine = assess_results.engine_of(case)
    if case.get('passed') is True:
        return 'passed', engine
    if case.get('execution_phase') == 'initialization':
        return 'no start', str(case.get('failure') or 'the application did not start')
    lacking = [str(x) for x in case.get('failed_checks') or []]
    missing = ', '.join(lacking) or str(case.get('failure') or 'see environment/case.json')
    if any(name.split('.')[0] in ('numpy', 'scipy') for name in lacking):
        return 'no packages', missing
    return 'missing', f'{engine}; required and missing: {missing}'


# ====================================================================== the campaign
class Campaign:
    def __init__(self, args, extra):
        self.args, self.extra = args, list(extra)
        self.lock = threading.Lock()
        self.children = set()
        self.stopping = False
        self.fingerprints = run_isaac.source_fingerprints()
        self.steps = []
        if args.isaac_root:
            self.runtime = args.isaac_root.expanduser().resolve() / ('python.bat' if os.name == 'nt' else 'python.sh')
        elif args.isaac_python:
            self.runtime = args.isaac_python.expanduser().absolute()
        else:
            self.runtime = Path(sys.executable)
        self.wanted = list(args.stages) if args.stages else (list(QUICK) if args.quick else list(STAGES))

    # ------------------------------------------------------------------ folder and record
    def open(self):
        args = self.args
        if args.resume is not None:
            self.folder = args.resume.expanduser().resolve()
            try:
                self.record = read_json(self.folder / 'campaign.json')
            except (OSError, ValueError) as exc:
                raise ValueError(f'{self.folder} holds no campaign: {exc}')
            earlier = list(self.record.get('options', {}).get('worker_arguments', []))
            if self.extra and self.extra != earlier:
                raise ValueError('this campaign was made with the stage options ' + (' '.join(earlier) or '(none)')
                                 + '; other options need a campaign of their own')
            self.extra = earlier
            self.teams = list(args.teams) if args.teams_given else [t for t in TEAMS if t in self.record.get('teams', {})]
            self.record['status'] = 'running'
        else:
            name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
            self.folder = args.output.expanduser().resolve() / f'isaac_campaign_{name}'
            self.folder.mkdir(parents=True, exist_ok=False)
            self.teams = list(args.teams)
            self.record = dict(schema_version=1, campaign_id=name, started_utc=utc(), status='running',
                               package=dict(kernel_sha256=self.fingerprints.get('vendor/pacdm_original.py'),
                                            source_digest=digest(self.fingerprints), files=len(self.fingerprints)),
                               options=dict(worker_arguments=self.extra),
                               environment=dict(folder='environment', team=self.teams[0]),   # a run folder belongs to one team
                               teams={}, invocations=[])
        for team in self.teams:
            self.record['teams'].setdefault(team, dict(folder=team, platform=assess_results.PLATFORM[team]))
        self.invocation = dict(started_utc=utc(), teams=self.teams, stages=self.wanted, headless=self.args.headless,
                               jobs=self.args.jobs, runtime=str(self.runtime), coordinator_python=sys.version.split()[0],
                               platform=platform.platform(), resumed=self.args.resume is not None,
                               source_digest=digest(self.fingerprints), steps=self.steps)
        self.record.setdefault('invocations', []).append(self.invocation)
        self.save()

    def save(self):
        save_json(self.folder / 'campaign.json', self.record)

    def log(self, text=''):
        with self.lock:
            run_isaac.echo(text + '\n')
            with open(self.folder / 'campaign.log', 'a', encoding='utf-8') as stream:
                stream.write(text + '\n')

    def states(self, team):
        return stage_states(self.folder / self.record['teams'][team]['folder'], self.fingerprints)

    # ------------------------------------------------------------------ one call of run_isaac.py
    def call(self, label, team, folder, stages, phase):
        """Run ``stages`` of ``team`` in ``folder`` with the coordinator of the single-team workflow."""
        command = [sys.executable, str(HERE / 'run_isaac.py'), '--team', team, '--stages', *stages, '--run-folder', str(folder),
                   '--no-bundle']
        if self.args.headless:
            command.append('--headless')
        if self.args.isaac_root:                                        # absolute: the stages run from the package root
            command += ['--isaac-root', str(self.runtime.parent)]
        elif self.args.isaac_python:
            command += ['--isaac-python', str(self.runtime)]
        command += self.extra
        if phase == 'quick stages' and '--keep-going' not in self.extra:
            command.append('--keep-going')                              # every quick stage has something to tell
        step = dict(phase=phase, team=team, stages=list(stages), started_utc=utc())
        started = time.perf_counter()
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        # a process group of its own (POSIX): an interruption is then passed on once, to the stage and its coordinator
        options = dict(start_new_session=True) if os.name != 'nt' else {}
        with self.lock:                                                 # started and registered in one piece: see stop_children
            if self.stopping:
                return None
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', **options)
            self.children.add(process)
        try:
            for line in process.stdout:
                with self.lock:
                    run_isaac.echo(f'[{label}] {line}')
            code = process.wait()
        finally:
            process.stdout.close()
            with self.lock:
                self.children.discard(process)
        step.update(exit_code=code, finished_utc=utc(), wall_s=round(time.perf_counter() - started, 1))
        with self.lock:
            self.steps.append(step)
        return step

    def run_all(self, tasks):
        """Run the calls of a phase, ``--jobs`` at a time.  ``tasks``: argument tuples of ``call``."""
        if not tasks:
            return
        with ThreadPoolExecutor(max_workers=self.args.jobs) as pool:
            futures = [pool.submit(self.call, *task) for task in tasks]
            try:
                while not all(f.done() for f in futures):               # a plain wait cannot be interrupted on Windows
                    time.sleep(.2)
            except KeyboardInterrupt:
                self.stop_children()
                raise
        for future in futures:
            future.result()                                             # an error of a call is an error of the campaign
        self.save()

    def stop_children(self):
        """After an interruption: no new stage starts, and the running ones end (by themselves, or they are ended)."""
        with self.lock:
            self.stopping = True
            children = list(self.children)
        if os.name != 'nt':
            for process in children:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except (ProcessLookupError, PermissionError):
                        pass
        deadline = time.monotonic() + GRACE_S
        while time.monotonic() < deadline and any(p.poll() is None for p in children):
            time.sleep(.2)
        for process in children:
            if process.poll() is None:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass

    # ------------------------------------------------------------------ phases
    def environment(self):
        folder = self.folder / self.record['environment']['folder']
        if self.args.resume is not None and environment_outcome(folder)[0] == 'passed':
            try:                                                        # the same installation and the same files as then?
                same = (read_json(folder / 'environment' / 'case.json').get('source_sha256') == self.fingerprints
                        and read_json(folder / 'manifest.json').get('runtime') == str(self.runtime))
            except (OSError, ValueError):
                same = False
            if same:
                self.log('\n== environment: passed earlier')
                return True
        self.log('\n== environment')
        team = self.record['environment'].setdefault('team', self.teams[0])
        self.run_all([('environment', team, folder, ['environment'], 'environment')])
        kind, text = environment_outcome(folder)
        self.record['environment'].update(outcome=kind, text=text)
        if kind == 'passed':
            self.log(f'environment: passed ({text})')
        elif kind == 'missing':
            self.log(f'environment: NOT PASSED ({text}).\nThe stages are run all the same: they show what this means for them.')
        elif kind == 'no packages':
            self.log(f'environment: the Python of Isaac Sim lacks what the controller needs: {text}\nNo other stage was run. '
                     'Install it there with: python -m pip install -r workflows/isaac/requirements-isaac.txt')
        else:
            self.log(f'environment: Isaac Sim did not start in {self.runtime}:\n  {text}\nNo other stage was run. '
                     'Run the campaign with the Python of Isaac Sim, or name it with --isaac-python.')
        return kind in ('passed', 'missing')

    def describe(self, team):
        states = self.states(team)
        return ', '.join(f'{stage} {states[stage]}' for stage in STAGES if states[stage] != 'not run') or 'nothing was run'

    def quick(self):
        tasks = []
        for team in self.teams:
            stages = [s for s in pending(self.states(team), self.wanted) if s in QUICK]
            if stages:
                tasks.append((team, team, self.folder / self.record['teams'][team]['folder'], stages, 'quick stages'))
        if not tasks:
            return
        self.log('\n== quick stages: ' + ', '.join(f'{t[1]} ({" ".join(t[3])})' for t in tasks))
        started = time.perf_counter()
        self.run_all(tasks)
        for team in self.teams:
            self.log(f'{team}: {self.describe(team)}')
        self.log(f'quick stages: {(time.perf_counter() - started) / 60.:.1f} minutes')

    def missions(self):
        tasks, held = [], []
        for team in self.teams:
            states = self.states(team)
            if 'mission' not in pending(states, self.wanted):
                continue
            before = [s for s in QUICK if states[s] != 'passed']
            if before:
                held.append(f'{team}: the mission was not run ({", ".join(f"{s} {states[s]}" for s in before)})')
            else:
                tasks.append((team, team, self.folder / self.record['teams'][team]['folder'], ['mission'], 'missions'))
        if not tasks and not held:
            return
        self.log('\n== missions: ' + (', '.join(t[1] for t in tasks) or 'none'))
        for line in held:
            self.log(line)
        started = time.perf_counter()
        self.run_all(tasks)
        for task in tasks:
            self.log(f'{task[1]}: {self.describe(task[1])}')
        if tasks:
            self.log(f'missions: {(time.perf_counter() - started) / 60.:.1f} minutes')

    # ------------------------------------------------------------------ end
    def finish(self, interrupted, error):
        states = {team: self.states(team) for team in self.teams}
        for team in self.teams:
            self.record['teams'][team]['stages'] = states[team]
        left = {team: pending(states[team], self.wanted) for team in self.teams}
        failed = [team for team in self.teams if any(states[team][s] == 'failed' for s in STAGES)]
        if interrupted:
            status = 'interrupted'
        elif error or failed:
            status = 'failed'
        elif any(left.values()):
            status = 'incomplete'
        else:
            status = 'passed'
        self.invocation.update(finished_utc=utc(), status=status)
        self.record.update(status=status, finished_utc=utc())
        if error:
            self.record['error'] = self.invocation['error'] = error
        else:
            self.record.pop('error', None)
        self.save()
        self.log('\n== assessment')
        try:
            assessment = assess_results.assess([self.folder], expected=[t for t in TEAMS if t in self.record['teams']])
            report = assess_results.write(assessment, self.folder)
            self.log(assess_results.summary(assessment))
            self.log(f'Report: {report}')
        except Exception as exc:                                         # noqa: BLE001 (the bundle is written all the same)
            self.log(f'the assessment could not be written: {type(exc).__name__}: {exc}')
        try:
            path, size = bundle(self.folder)
            self.log(f'Bundle: {path} ({size / 1e6:.1f} MB)')
            if self.args.with_records:
                path, size = bundle(self.folder, records=True)
                self.log(f'Records: {path} ({size / 1e6:.1f} MB)')
        except Exception as exc:                                         # noqa: BLE001
            self.log(f'the bundle could not be written: {type(exc).__name__}: {exc}')
        words = dict(passed='every stage asked for passed', failed='NOT PASSED', incomplete='INCOMPLETE: stages are left to run',
                     interrupted='INTERRUPTED')
        self.log(f'\nCampaign {self.record["campaign_id"]}: {words[status]}')
        for team in self.teams:
            if left[team]:
                self.log(f'  {team}: left to run: {", ".join(left[team])}')
        later = [team for team in self.teams if pending(states[team], STAGES)]
        if status != 'passed' or later:
            if status == 'passed':
                self.log('Stages of the workflow that this campaign has not run yet: '
                         + '; '.join(f'{team}: {", ".join(pending(states[team], STAGES))}' for team in later))
            self.log('To run what is left: ' + self.resume_command())
        self.log(f'Folder: {self.folder}')
        return 130 if interrupted else (0 if status == 'passed' else 1)

    def resume_command(self):
        words = ['python', 'workflows/isaac/run_campaign.py', '--resume', f'"{self.folder}"']
        if self.args.headless:
            words.append('--headless')
        if self.args.isaac_root:
            words += ['--isaac-root', f'"{self.args.isaac_root}"']
        elif self.args.isaac_python:
            words += ['--isaac-python', f'"{self.args.isaac_python}"']
        if self.args.jobs > 1:
            words += ['--jobs', str(self.args.jobs)]
        return ' '.join(words)


def bundle(folder, records=False):
    """``campaign_bundle.zip``: every record of the campaign except scenes, time series, videos and archives.
    ``records``: ``campaign_records.zip`` with the time series (``record.npz``) instead."""
    path = folder / ('campaign_records.zip' if records else 'campaign_bundle.zip')
    temporary = path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_STORED if records else zipfile.ZIP_DEFLATED) as z:
        for f in sorted(folder.rglob('*')):
            if f.is_file() and ((f.name == 'record.npz') if records else (f.suffix not in NOT_BUNDLED)):
                z.write(f, f.relative_to(folder))
    temporary.replace(path)
    return path, path.stat().st_size


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def _ignored(signum, frame):
    """A signal the campaign was started to ignore: it goes on ignoring it."""


def answer_signals():
    """How the campaign answers the signals that end it.  Returns the names of those that end it in order.

    SIGTERM, SIGHUP and Ctrl+Break end the campaign as Ctrl+C does, stages included.  A signal the campaign was
    started to ignore stays ignored: SIGHUP under ``nohup``, and SIGINT in the background of a shell script.  The
    stages, however, must not inherit an ignored SIGINT, because that signal is what ends them in order when the
    campaign is told to end.  An ignored signal is inherited by a child process and a handled one is not, so a
    handler that does nothing takes its place here.
    """
    ending = []
    for name in ('SIGTERM', 'SIGHUP', 'SIGBREAK'):
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            if signal.getsignal(number) != signal.SIG_IGN:
                signal.signal(number, _interrupt)
                ending.append(name)
        except (OSError, ValueError):
            pass
    try:
        if signal.getsignal(signal.SIGINT) == signal.SIG_IGN:
            if os.name != 'nt':
                signal.signal(signal.SIGINT, _ignored)
        else:
            ending.insert(0, 'SIGINT')
    except (OSError, ValueError):
        pass
    return ending


def main(argv=None):
    args, extra = parse_args(argv)
    campaign = Campaign(args, extra)
    if not campaign.runtime.is_file():
        print(f'The Isaac Python does not exist: {campaign.runtime}', file=sys.stderr)
        return 2
    try:
        campaign.open()
    except (OSError, ValueError) as exc:
        print(f'The campaign cannot be started: {exc}', file=sys.stderr)
        return 2
    ending = answer_signals()
    campaign.log(f"Campaign {campaign.record['campaign_id']}" + (' (continued)' if args.resume else ''))
    campaign.log(f'Folder: {campaign.folder}\nIsaac Python: {campaign.runtime}')
    campaign.log(f"Teams: {', '.join(campaign.teams)}\nStages: {', '.join(campaign.wanted)}"
                 + (f"\nStage options: {' '.join(campaign.extra)}" if campaign.extra else ''))
    if 'SIGINT' in ending:
        campaign.log('Ctrl+C ends the campaign in order; --resume continues it later.')
    else:                                                                # the background of a shell script
        campaign.log('Started to ignore the interrupt signal; ' + (' or '.join(ending) or 'nothing') + ' ends the campaign in order; '
                     '--resume continues it later.')
    interrupted, error = False, None
    try:
        if campaign.environment():
            campaign.quick()
            campaign.missions()
        else:
            error = ('NumPy or SciPy is missing in the Python of Isaac Sim: ' if campaign.record['environment'].get('outcome') == 'no packages'
                     else 'Isaac Sim did not start: ') + campaign.record['environment'].get('text', '')
    except KeyboardInterrupt:
        interrupted = True
        campaign.stop_children()
        campaign.log('\ninterrupted')
    except Exception as exc:                                             # noqa: BLE001 (recorded; the folder is closed properly)
        error = f'{type(exc).__name__}: {exc}'
        campaign.stop_children()
        campaign.log(f'\nthe campaign ended with an error: {error}')
    return campaign.finish(interrupted, error)


if __name__ == '__main__':
    sys.exit(main())
