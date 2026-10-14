# Reference results

Outputs of the runs made when the package was built: 2 to 6 October 2026, Python 3.12.3 on Linux
x86-64 (two CPU cores), NumPy 2.3.5, SciPy 1.17.0, MuJoCo 3.3.7, Pinocchio 3.8.0. The records
(`record.npz`) are not included; the commands below write them again, into `results/`. `<team>` is `team2`, `team3`, `ridgeback2`,
`ridgeback3`, `husky2` or `husky3`.

| Folder | Command (from the package root) |
| --- | --- |
| `mujoco_<team>` | `python workflows/mujoco/run_mission.py --team <team> --frames 6.25 --video` |
| `mujoco_team2_compliant_clamp`, `mujoco_ridgeback2_compliant_clamp`, `mujoco_husky2_compliant_clamp` | `python workflows/mujoco/run_mission.py --team <team> --latch compliant --sensor model --out results/mujoco_<team>_compliant_clamp` |
| `pinocchio_<team>` | `python workflows/pinocchio/validate_interfaces.py --team <team>` then `python workflows/pinocchio/run_mission.py --team <team>` |
| `fitted_feedforward` | the 40 missions and the decay tests of the fitted feedforward (seventh release); the options of every run are in the main README, "The feedforward of the fitted reference", and the table of all runs is `fitted_feedforward/FEEDFORWARDS.md` |
| `design_study_rear_to_beam` | a formation of the Ridgeback team that was run and rejected; see the README in that folder |
| `study_husky_rigid_ground` | `python workflows/pinocchio/run_mission.py --team <husky2 or husky3> --skid-ground rigid --out results/pinocchio_<team>_rigid_ground`: the Husky teams stepped on rigid ground, where they do not meet their bounds; see the README in that folder |

In every run folder:

- `summary.txt`: the audit in words; `report.json`: all its numbers.
- `route.png`, `tracking.png`, `grasp_loads.png`, `load_transfer.png`, `margins.png`.
- MuJoCo: `plant.xml`, the generated model (six main runs). Pinocchio: `interfaces.json`, the interface
  validation.

`logs/` has the console output of the MuJoCo and Pinocchio runs.

The videos that the workflows render (`mission.mp4`, with one still per phase in `frames/`) or draw
from a record (`replay.mp4`, `tools/replay_video.py`) are not included; nothing in the package reads
them.

## Which package state the runs belong to

### Husky teams

The MuJoCo and Pinocchio runs were made on 3 October. While they were in progress the package
changed in one respect: the audit reports the samples at which the skid model has no twist to give
(a base without load on one side) instead of failing on them, and the skid model refuses a base
that stands on one wheel alone (`mission/audit.py`, `teamc/skid.py`). The rigid-ground runs of the
study showed the need.

- MuJoCo and Pinocchio runs of `husky2` and `husky3`: made before it. The shipped audit, run on
  their stored records (`tools/make_report.py`), gives every number of their `report.json` files
  here. Their console logs are those of the runs. The interface validation and the Pinocchio
  mission of `husky2` were also run again from a clean copy of the shipped package, with the same
  result, number for number.
- On 6 October the tyre model and the terrain gained what the tyre springs of the PhysX plant
  need (`backends/tyre.py`, `teamc/terrain.py`). The MuJoCo and the Pinocchio mission of `husky2`
  were run again with the present files: their reports are the stored ones, value for value
  (2303 and 2291 values; the run times aside).
- Rigid-ground runs of the study: made after the change. They had been run once before it, which
  is what showed the need: the audit of that moment failed on their records.

`tools/build_mission.py`, run again in a clean copy of the shipped package, writes the two Husky
missions byte for byte.

### Ridgeback and RB-ROBOUT+ teams

Their MuJoCo and Pinocchio runs were not repeated. Those of the Ridgeback teams were made on
3 October with the package as it was before the third platform was added, those of the RB-ROBOUT+
teams on 2 and 3 October, before the second platform was added. For these four teams the shipped
package computes what it computed then:

- `tools/build_robot.py`, `tools/build_scene.py` and `tools/build_mission.py` still write their
  robot, payload, scene, team and mission records byte for byte;
- the Pinocchio interface validation gives every number of the four `interfaces.json` files here;
- the shipped audit, run on the stored records of the MuJoCo and Pinocchio missions of these teams
  (`tools/make_report.py`), gives every number of their `report.json` files here.

The reports of the RB-ROBOUT+ runs are not those of the runs themselves: their `summary.txt`,
`report.json` and figures were written from the stored records when the second platform was
added. Against the audit of the runs two things differ: the largest force and moment of a grasp
are taken over every record interval instead of every fifth one (so these maxima are a little
larger), and the wrist-sensor readings are reported. The console logs of these runs in `logs/` show
the audit of the runs. The `plant.xml` files of all four teams are those of the runs.
