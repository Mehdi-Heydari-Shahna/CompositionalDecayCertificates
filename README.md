# CompositionalDecayCertificates

**Robot Teams Sharing a Load: Compositional Decay Certificates**

Mehdi Heydari Shahna · Soyi Jung · Soohyun Park · Joongheon Kim · Jouni Mattila

This repository is the supplementary material of the paper: the source code of every numerical result, the recorded
results of the runs behind its figures and tables, the trajectories of the 40 complete missions, and the scripts that
draw the figures from the recorded results.

Robots carrying a shared payload form a constrained mechanical system whose velocity-error decay rate depends on all
robots and constraints. The paper studies when this rate can be certified, and gains achieving it designed, from
inequalities on each robot's own constraint space: local certificates for cell architectures with a join tree, the
endpoint condition under which they attain the centralized rate, robot-level linear matrix inequalities for
minimum-gain and maximum-rate designs, and the realization of the certified rates by joint-level loops with a fitted
reference, including events, sampled renewal and skid-steered bases driven through wheel friction. The results are
evaluated on six compiled mobile-manipulator teams (two and three RB-ROBOUT+/UR30, Ridgeback/UR10e and Husky
A300/UR10e robots carrying a beam) on a Pinocchio-based plant with ideal constraints and on MuJoCo 3.3.7.
[RoboCompiler](https://github.com/Mehdi-Heydari-Shahna/RoboCompiler) supplies the compiled team graphs; its package for
robot teams, `Team_CMG_PACDM`, is included in `code/`.

## Repository contents

| Path | Content |
|---|---|
| `code/Team_CMG_PACDM/` | Team compiler, mission controller (desired-motion and fitted feedforward), MuJoCo and Pinocchio plants, mission workflows, tools and tests, with the reports of its reference runs (`reference_results/`); see its `README.md` and `DESIGN.md` |
| `code/checks/` | Numerical checks of the statements on the compiled teams: identity, events, redundant rows, fitted reference, friction-driven bases, decay tests, missions, robustness |
| `code/analysis/` | Payload-space test, refresh and membership, synthetic assemblies, gain synthesis, decay runs, carrying segment, per-step grasp loads, friction-driven teams |
| `code/figures/` | Scripts of Figs. 3 and 5-12 |
| `results/checks/`, `results/analysis/` | Recorded results of the scripts in `code/checks/` and `code/analysis/` |
| `results/trajectories/` | Trajectories of the 40 missions at 10 Hz (CSV) |
| `results/figures/` | Figures drawn from the recorded results (PDF, with PNG previews), and the video frames used in Figs. 5 and 12 |
| `reproduce.sh` | Regenerates the results and figures into `results/` |
| `requirements.txt` | Pinned Python packages |

## Setup

Use Python 3.13 on Linux (or WSL2 on Windows). Open a terminal in the repository folder and run:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Keep the folder structure intact: the scripts find the framework package and the recorded results through it. The
environment variables `TEAM_CMG_PACDM`, `CHECKS_CODE`, `CHECKS_RESULTS`, `ANALYSIS_RESULTS`, `TRAJECTORIES` and
`FIG_DIR` point them elsewhere. The
workflows of the framework package were run with its own pins (`code/Team_CMG_PACDM/requirements-linux.txt`, Python
3.12); its test suite also passes with `requirements.txt`.

## Run single examples

```bash
# Pointwise checks: Proposition 3 at random states, and the events of Fig. 3 in time
python code/checks/exp1_identity.py
python code/checks/sim.py team3 --T 3 --amp-scale 0.5 --events

# Payload-space test at the nominal configurations, synthesis of Table 2, a decay run of Fig. 7
python code/analysis/check_table2.py
python code/analysis/run_synth_tables.py
python code/analysis/run_decay_gains.py team2

# A figure from the recorded results
python code/figures/fig08_decay_jump.py

# A complete mission of three RB-ROBOUT+/UR30 robots with the fitted loops F/2, on MuJoCo and on the ideal-constraint plant
python code/Team_CMG_PACDM/workflows/mujoco/run_mission.py --team team3 --feedforward fitted --option arm_damping_ratio=0.5 --option lift_damping_ratio=0.5
python code/Team_CMG_PACDM/workflows/pinocchio/run_mission.py --team team3 --feedforward fitted --option arm_damping_ratio=0.5 --option lift_damping_ratio=0.5
```

The scripts of `code/checks/` and `code/analysis/` write their results into `results/checks/` and
`results/analysis/`, replacing the recorded results of the same name (Git shows what changed).
`check_realizability.py`, `check_refresh.py`, `run_synth_tables.py` and the ADMM scripts read the snapshots written by
`snapshots.py`, and the decay runs read the gains of `synth_tables.json`. The mission workflows write to
`code/Team_CMG_PACDM/results/<plant>_<team>/` (`summary.txt`, `report.json`, `record.npz`, figures). The recorded
missions took between 7 and 79 minutes of wall-clock time each on two CPU cores (see their logs in
`code/Team_CMG_PACDM/reference_results/`).

## Reproduce the paper

```bash
bash reproduce.sh checks       # pointwise checks and runs on the compiled models (Sections 6.3, 7.1, 7.2, 7.5, 7.8)
bash reproduce.sh analysis     # certificates, refresh, synthesis and decay runs (Sections 7.1-7.5, 7.8)
bash reproduce.sh segments     # carrying segment at plant steps of 2 to 0.25 ms on both plants (Table 3)
bash reproduce.sh loading      # MuJoCo missions with the grasp loads of every plant step (Sections 7.6, 7.8)
bash reproduce.sh robustness   # carry stage under sampling, model error, noise and delay (Section 7.7, Fig. 11)
bash reproduce.sh figures      # Figs. 3 and 5-12 from the results
bash reproduce.sh tests        # test suite of the framework package
```

`bash reproduce.sh` runs all steps in this order; the console output of every script goes to `results/logs/`. On two
CPU cores, `checks` takes about 7 minutes, `analysis` about 16 minutes, `figures` less than a minute and `tests` about
10 minutes; the run times recorded in the results of the other steps add up to about 2 hours (`segments`), 1.5 hours
(`loading`) and 5.3 hours (`robustness`). The 40 missions and the decay tests of the mission controller are runs of
the framework's workflows (see "Missions and decay tests" below); the `checks` step recomputes
`results/checks/decay.json` and `decay.npz` from the stored decay runs.

| Paper item | Script | Output |
|---|---|---|
| Fig. 3 and the event values of Section 6.3 | `code/checks/sim.py team3 --T 3 --amp-scale 0.5 --events` (and `--release-jump 0.15 --tag events_jump`); `code/figures/fig03_events.py` | `results/checks/sim_team3_events*.{json,npz}`, `results/figures/fig_events.pdf` |
| Fig. 4: n, d, p and rank L of the six teams | `code/checks/survey.py` | `results/checks/survey.json` |
| Table 1 | inputs of the framework: `code/Team_CMG_PACDM/inputs/missions/<team>.json` (grasp bounds; gains of the Ridgeback teams), `inputs/payloads/` (beam masses) and the controller defaults in `mission/controller.py` (gains of the RB-ROBOUT+ teams) | |
| Fig. 5 | `code/figures/fig05_snapshots.py` (frames of the mission video of `mujoco_team3_fitted`) | `results/figures/fig_pick_carry_deliver.pdf` |
| Section 7.1: the 24 navigation snapshots and the closure change of the arm coordinates | `code/analysis/snapshots.py` (from `results/trajectories/pinocchio_<team>_fitted/`) | `results/analysis/snapshots.json`, `snapshots_<team>.npy` |
| Section 7.2: Proposition 3, Theorem 19(a), Lemma 21 and Proposition 22 at random states, Lemma 2 with 28 rows of rank 26 | `code/checks/exp1_identity.py`, `exp7_actuator_reference.py`, `exp3_events.py`, `exp6_redundant.py` | `results/checks/exp1_identity.json`, `exp7_actuator_reference.json`, `exp3_events.json`, `exp6_redundant.json` |
| Section 7.2: centralized rates, payload-space test, supplies; Fig. 6(b) | `code/analysis/check_table2.py` | `results/analysis/table2_check.json` |
| Fig. 6(a): scalar shares for gains designed robot by robot | `code/checks/exp2_modular.py` | `results/checks/exp2_modular.json` (and its copy `results/analysis/exp2_share_gains.json`) |
| Fig. 6 | `code/figures/fig06_composition.py` | `results/figures/fig_composition.pdf` |
| Section 7.2: rate ranges along the snapshots, payload realizability | `code/analysis/snapshots.py`, `check_realizability.py` | `results/analysis/snapshots.json`, `check_realizability.json` |
| Section 7.2: refresh (target of 90 %, change of the aggregate) and membership | `code/analysis/check_refresh.py` | `results/analysis/refresh_membership.json` |
| Section 7.3: Proposition 12 with bounded supplies | `code/analysis/verify_endpoint.py` | `results/analysis/verify_endpoint.json` |
| Section 7.3: Theorem 11(b) on random join trees, the constructions of Theorem 13 | `code/analysis/verify_new_theorems.py` | `results/analysis/verify_new_theorems.json` |
| Section 7.3: Remark 15, two-module edges on random assemblies | `code/analysis/verify_edge_arity.py` | `results/analysis/verify_edge_arity.json` |
| Table 2: P1 and P2, centralized and robot-level | `code/analysis/run_synth_tables.py` (uses `synth.py`) | `results/analysis/synth_tables.json` |
| Section 7.4: sharing ADMM for P1 on team2 (Remark 17) and the choice of its parameters | `code/analysis/run_admm_team.py team2 200`, `sweep_admm.py` | `results/analysis/admm_team2.json`, `sweep_admm.json` |
| Section 7.4: rate limits of the joint and wheel loops | `code/analysis/check_gain_limits.py` | `results/analysis/check_gain_limits.json` |
| Fig. 7: decay runs, 0.5 ms steps, renewal in every step | `code/analysis/run_decay_gains.py <team>`; `code/figures/fig07_synthesis_decay.py` | `results/analysis/decay_gains_<team>.{json,npz}`, `results/figures/fig_synth_decay.pdf` |
| Section 7.5: step check at 1 ms, margins and their extrapolation to zero step | `code/analysis/run_decay_gains.py <team> 0.001 0.001 _1ms`, `check_decay_convergence.py` | `results/analysis/decay_gains_<team>_1ms.{json,npz}`, `check_decay_convergence.json` |
| Section 7.5: 10 ms renewal at plant steps of 2, 1 and 0.5 ms | `code/analysis/check_decay_renewal10ms.py` | `results/analysis/check_decay_renewal10ms.json` |
| Fig. 8 and the jump tests with D, F and F/2 (adverse eigenvalues, rises of D, bands) | `code/checks/decay.py` (from the stored decay runs); `code/figures/fig08_decay_jump.py` | `results/checks/decay.{json,npz}`, `results/figures/fig_matched_decay.pdf` |
| Section 7.6: the 40 missions (times, placement, storage), grasp-moment maxima; Figs. 9 and 10 | `code/checks/missions.py` (from the mission records); `code/figures/fig09_missions.py`, `fig10_mission_comparison.py` | `results/checks/missions.{json,npz}`, `results/figures/fig_mission_all.pdf`, `fig_matched_missions.pdf`; reports in `code/Team_CMG_PACDM/reference_results/fitted_feedforward/` |
| Table 3 and the step refinement of Section 7.6 | `code/analysis/run_segment_compare.py [mujoco] <team> 0.002 0.001 0.0005 0.00025`, `summarize_segments.py` | `results/analysis/segment_<team>.{json,npz}`, `segment_mujoco_<team>.{json,npz}`, `segment_summary.json` |
| Section 7.6: repeated F/2 missions on MuJoCo and their grasp moments at every 2 ms step | `code/analysis/run_native_loading.py <team> fitted05`, `summarize_native_loading.py`, `check_native_inference.py` | `results/analysis/native_loading_<team>_fitted05.json` (and folder), `native_loading_summary.json`, `check_native_inference.json` |
| Fig. 11 and Section 7.7 | `code/checks/exp12_robustness.py`; `code/figures/fig11_robustness.py` | `results/checks/exp12_team2.json`, `exp12_ridgeback3.json`, `results/figures/fig_robust.pdf` |
| Section 7.8: ker L, rates at zero slip and at random sliding states, Proposition 23(a), the residual of (c) | `code/checks/exp5_friction_pointwise.py` | `results/checks/exp5_friction_pointwise.json` |
| Section 7.8: slowest certified motion at zero slip and the shares of its dissipation | `code/analysis/check_friction_slow_mode.py` | `results/analysis/check_friction_slow_mode.json` |
| Fig. 12(d): the three friction runs of husky2 | `code/checks/exp5_friction_sim.py husky2 --mu 0.25 --tag floor025`, `... --mu 0.25 --scrub 0.04 --tag scrub`, `... --tag floor080` | `results/checks/exp5_husky2_{floor025,scrub,floor080}.{json,npz}` |
| Section 7.8: Husky missions (times, payload errors, placement, 20 ms grasp moments, wheel slip, friction energy) | MuJoCo and Pinocchio workflows of the framework | `code/Team_CMG_PACDM/reference_results/{mujoco,pinocchio}_husky{2,3}/report.json` |
| Section 7.8: grasp moments of the Husky missions at every 2 ms step | `code/analysis/run_native_loading.py husky2 desired`, `... husky3 desired`, `check_native_inference.py` | `results/analysis/native_loading_husky{2,3}_desired.json` (and folders), `check_native_inference.json` |
| Section 7.8: the Husky teams on rigid ground | `code/Team_CMG_PACDM/workflows/pinocchio/run_mission.py --team <team> --skid-ground rigid` | `code/Team_CMG_PACDM/reference_results/study_husky_rigid_ground/` |
| Fig. 12 | `code/figures/fig12_friction_teams.py` (frames of the mission video of `mujoco_husky3`) | `results/figures/fig_husky.pdf` |

Figs. 1, 2 and 4 are diagrams and renderings without a data script; the numbers of Fig. 4 are those of
`results/checks/survey.json`.

## Missions and decay tests

The 40 missions of Section 7.6 are runs of `code/Team_CMG_PACDM/workflows/mujoco/run_mission.py` and
`workflows/pinocchio/run_mission.py` with `--team <team>` and the options of the controller:

| Controller | In the paper | Options |
|---|---|---|
| `desired` | D with the 15 Hz rate filter | `--monitor` |
| `desired0` | D | `--monitor --velocity-filter 0` |
| `desiredR` | D_R | `--monitor --option "arm_damping_ratio=[...]"` with the damping ratios of the arm joints of `DAMPING_R` in `code/checks/missions.py` |
| `fitted` | F | `--feedforward fitted` |
| `fitted05` | F/2 | `--feedforward fitted --option arm_damping_ratio=0.5 --option lift_damping_ratio=0.5` |
| `fitted15` | F with the 15 Hz rate filter | `--feedforward fitted --velocity-filter 15` |
| `fitted0515` | F/2 with the 15 Hz rate filter | the options of `fitted05` and `--velocity-filter 15` |

The ideal-constraint plant has the runs `desired`, `fitted` and `fitted05`, MuJoCo all seven. The report, summary,
figures and console log of every run are in `code/Team_CMG_PACDM/reference_results/fitted_feedforward/`, with the
table of all runs in `FEEDFORWARDS.md` there (written by `tools/compare_feedforwards.py`). `code/checks/missions.py`
collects the runs from their folders (`--out code/Team_CMG_PACDM/results/v7/<plant>_<team>_<controller>`; the runs of
`desired` under `results/v7/earlier/desired_d9384fd/`) and needs their records (`record.npz`, 29 to 54 MB each), which
are not included; `results/trajectories/` has the trajectories exported from them.

The decay tests of Section 7.5 (Fig. 8) are runs of `code/Team_CMG_PACDM/tools/run_decay.py` along the adverse direction
and three random admissible directions per team and plant. Their records are in
`code/Team_CMG_PACDM/reference_results/fitted_feedforward/decay/`, with the settings of every run in its `decay.json`;
`python code/checks/decay.py code/Team_CMG_PACDM/reference_results/fitted_feedforward/decay` collects them into
`results/checks/decay.json` and `decay.npz`.

## Recorded results

- `results/checks/`: the results of `code/checks/` (JSON summaries, NPZ time histories), and the carrying
  configurations `q_carry_<team>.npy` that every script starts from.
- `results/analysis/`: the results of `code/analysis/`. For the runs of `run_native_loading.py` the folders hold the
  workflow's `report.json` and `summary.txt`; their records (29 to 47 MB each) are not included.
- `results/trajectories/`: the 40 missions at 10 Hz, one folder per run (see the README there).
- `results/figures/`: the figures drawn from these results; `frames/` holds the video frames of Figs. 5 and 12.
- `code/Team_CMG_PACDM/reference_results/`: the reports, summaries, figures and console logs of the framework's
  reference runs: the missions of the six teams with the default controller on both plants, the runs with the fitted
  feedforward and the decay tests (`fitted_feedforward/`), and the studies its `README.md` describes.

## Notes

- The scripts and their docstrings number the statements, sections and tables as in the paper.
- A rerun of the `checks` and `analysis` steps reproduces the recorded values to the digits reported in the paper.
  Rounding errors differ between machines and linear-algebra builds: residuals of order 1e-18 to 1e-12 change, the
  coordinates of the snapshots of `snapshots.py` change by about 1e-14, and with them the synthesized gains change
  within the solver tolerance (the relative shortfall of a certified rate in `synth_tables.json` can then reach a few
  1e-10). Simulations on MuJoCo can differ in the last digits; Section 7.6 of the paper reports how much the F/2
  missions changed when they were repeated on another computer (`run_native_loading.py` repeats them here).
- The synthesis LMIs of `run_synth_tables.py` use the gain ratios to the mission gains as decision variables (bounds 0.1
  and 2). Solver statuses are recorded; every returned gain set is rechecked by generalized eigenvalues at every
  snapshot (the `certified_*` fields of `synth_tables.json`).
- The carrying-segment runs renew references, requests and feedforward every 10 ms and apply feedback in every plant
  step; on MuJoCo they use the settings of the mission workflow. Errors are taken over the 4 s of motion that follow the
  2 s hold (`peak_moving_mm`, `rms_moving_mm`); the values over the whole segment are recorded as well.
- `run_native_loading.py` repeats MuJoCo missions with the workflow of the package and infers the grasp wrench at every
  plant step with the package's own sensor model; the means of that series over the 20 ms record intervals reproduce the
  workflow's audit of the same run. The Husky teams have no fitted loops (the controller refuses the fitted feedforward
  for friction-driven bases), so their runs use the default controller.
- Figs. 3, 9 and 11 were drawn for the paper with Matplotlib 3.10.8, and so are their files in `results/figures/`; with
  that version their scripts reproduce them pixel for pixel. With Matplotlib 3.11.2 (`requirements.txt`) the plotted
  data are the same and the fonts and tick labels differ slightly.
- The frames of Figs. 5 and 12 are taken from the mission videos of `mujoco_team3_fitted` and `mujoco_husky3` (25
  frames per second at four times real time), at simulated time t with
  `ffmpeg -ss <t/4> -i mission.mp4 -frames:v 1 <frame>.png`.

## Videos

The mission videos accompany the paper separately. The MuJoCo workflow renders the video of a run with
`--frames 6.25 --video`, and `code/Team_CMG_PACDM/tools/replay_video.py` draws one from the record of any run.

## Paper and citation

Please cite the paper when using this work:

Mehdi Heydari Shahna, Soyi Jung, Soohyun Park, Joongheon Kim, and Jouni Mattila. **Robot Teams Sharing a Load:
Compositional Decay Certificates.** 2026.

This work builds on RoboCompiler: Mehdi Heydari Shahna, Joongheon Kim, and Jouni Mattila. **RoboCompiler: Graph-Native
Compilation of Closed-Chain Robots for Consistent Modeling, Control, and Simulation.** 2026.
https://doi.org/10.48550/arXiv.2609.35717

## Acknowledgment

This work was supported by the Korean Ministry of Trade, Industry and Energy (MOTIE) under Grant RS-2024-00442168,
through the Mechanical Equipment Industry Technology Development Program administered by KEIT.

## License

Released under the Apache License 2.0; see [LICENSE](LICENSE). The robot description files in
`code/Team_CMG_PACDM/upstream/` and the records derived from them keep their BSD-3-Clause licences, and the PACDM kernel
in `code/Team_CMG_PACDM/vendor/` is included unchanged; see `code/Team_CMG_PACDM/THIRD_PARTY_NOTICES.md`.
