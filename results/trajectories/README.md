# Trajectories of the 40 missions

The complete missions of Section 7.6 of the paper, sampled at 10 Hz: on the ideal-constraint plant (Pinocchio) with
three controllers per team and on MuJoCo with seven. One folder per run, named `<plant>_<team>_<controller>` as in
`code/Team_CMG_PACDM/reference_results/fitted_feedforward/`, where the report and the figures of every run are.

| Controller | Feedforward | Rate filter in the damping term | Rate gains of the arm and lift joints | In the paper |
| --- | --- | --- | --- | --- |
| `desired` | desired motion (the default controller) | 15 Hz | k_d | D with the filter |
| `desired0` | desired motion | none | k_d | D |
| `desiredR` | desired motion | 15 Hz | arm damping designed for the inertia the joints see | D_R |
| `fitted15` | fitted reference | 15 Hz | k_d | F with the filter |
| `fitted` | fitted reference | none | k_d | F |
| `fitted0515` | fitted reference | 15 Hz | k_d/2 | F/2 with the filter |
| `fitted05` | fitted reference | none | k_d/2 | F/2 |

The ideal-constraint plant has the runs `desired`, `fitted` and `fitted05`. All runs use noise-free rate measurements.

- `trajectory.csv`: one row per 0.1 s, one column per quantity.
- `trajectory_columns.txt`: the columns in order (the unit is the suffix of each name) and the numbers of the mission
  phases in the column `phase`.

The files were written by `code/Team_CMG_PACDM/tools/export_trajectory.py` from the record (`record.npz`) of each run.
The records (29 to 54 MB each) are not included; the mission workflows write them again, for example

    python code/Team_CMG_PACDM/workflows/mujoco/run_mission.py --team team3 --feedforward fitted
    python code/Team_CMG_PACDM/tools/export_trajectory.py --record code/Team_CMG_PACDM/results/mujoco_team3/record.npz --rate 10

(the options of every controller: `code/Team_CMG_PACDM/tools/compare_feedforwards.py` and `code/Team_CMG_PACDM/README.md`,
"The feedforward of the fitted reference").
`code/analysis/snapshots.py` takes the 24 navigation snapshots of Section 7.1 from the runs `pinocchio_<team>_fitted`.
