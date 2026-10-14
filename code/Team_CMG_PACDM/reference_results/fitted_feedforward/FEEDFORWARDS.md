| Plant | Team | Controller | Payload error max / rms (mm) | Tilt max (mrad) | Base error max (mm) | Internal force max / rms (N) | Internal moment max (N m) | Effort ratio max | Error energy of the loops, median / 99 % (J) | Bounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pinocchio, ideal constraints | `team2` | desired motion | 0.51 / 0.14 | 0.56 | 1.0 | 24.3 / 0.88 | 7.8 | 0.63 | 1.9e-06 / 2.3e-03 | all met |
| Pinocchio, ideal constraints | `team2` | fitted reference | 0.51 / 0.14 | 0.66 | 0.5 | 11.5 / 0.34 | 4.1 | 0.71 | 1.9e-08 / 2.4e-05 | all met |
| Pinocchio, ideal constraints | `team2` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.51 / 0.14 | 0.58 | 0.5 | 5.9 / 0.18 | 2.6 | 0.68 | 8.5e-08 / 3.6e-05 | all met |
| Pinocchio, ideal constraints | `team3` | desired motion | 0.51 / 0.13 | 0.44 | 1.1 | 43.2 / 1.54 | 13.4 | 0.63 | 3.1e-06 / 2.8e-03 | all met |
| Pinocchio, ideal constraints | `team3` | fitted reference | 0.28 / 0.13 | 0.47 | 0.5 | 18.7 / 0.78 | 6.3 | 0.69 | 2.2e-08 / 3.7e-05 | all met |
| Pinocchio, ideal constraints | `team3` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.51 / 0.13 | 0.42 | 0.5 | 9.6 / 0.42 | 3.3 | 0.66 | 1.1e-07 / 5.4e-05 | all met |
| Pinocchio, ideal constraints | `ridgeback2` | desired motion | 0.19 / 0.05 | 0.30 | 0.7 | 1.4 / 0.10 | 0.6 | 0.67 | 9.1e-07 / 1.7e-03 | all met |
| Pinocchio, ideal constraints | `ridgeback2` | fitted reference | 0.11 / 0.05 | 0.22 | 0.3 | 1.7 / 0.09 | 0.5 | 0.67 | 1.1e-08 / 7.8e-06 | all met |
| Pinocchio, ideal constraints | `ridgeback2` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.12 / 0.05 | 0.22 | 0.3 | 1.2 / 0.05 | 0.4 | 0.67 | 2.7e-08 / 9.0e-06 | all met |
| Pinocchio, ideal constraints | `ridgeback3` | desired motion | 0.14 / 0.05 | 0.23 | 0.8 | 31.3 / 1.06 | 7.2 | 0.65 | 1.3e-06 / 1.8e-03 | all met |
| Pinocchio, ideal constraints | `ridgeback3` | fitted reference | 0.10 / 0.04 | 0.17 | 0.3 | 13.1 / 0.38 | 3.1 | 0.65 | 1.5e-08 / 1.3e-05 | all met |
| Pinocchio, ideal constraints | `ridgeback3` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.10 / 0.04 | 0.17 | 0.3 | 7.1 / 0.22 | 1.7 | 0.64 | 4.4e-08 / 1.4e-05 | all met |
| MuJoCo | `team2` | desired motion | 0.78 / 0.19 | 0.87 | 7.2 | 25.0 / 4.78 | 8.3 | 0.64 | 6.9e-03 / 1.3e-01 | all met |
| MuJoCo | `team2` | fitted reference | 0.60 / 0.17 | 0.62 | 7.0 | 42.2 / 5.06 | 13.5 | 0.68 | 7.0e-03 / 1.2e-01 | all met |
| MuJoCo | `team2` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.41 / 0.15 | 0.56 | 7.2 | 23.3 / 3.19 | 7.7 | 0.66 | 6.9e-03 / 1.2e-01 | all met |
| MuJoCo | `team2` | fitted reference, rates of the damping term low-passed at 15 Hz | 0.76 / 0.17 | 0.71 | 7.2 | 22.4 / 4.36 | 7.9 | 0.64 | 6.3e-03 / 1.2e-01 | all met |
| MuJoCo | `team2` | desired motion, damping term without the 15 Hz filter | 0.66 / 0.19 | 0.78 | 7.1 | 34.4 / 5.03 | 11.8 | 0.66 | 6.4e-03 / 1.2e-01 | all met |
| MuJoCo | `team2` | fitted reference, rate gains sqrt(k_p m), rates of the damping term low-passed at 15 Hz | 0.52 / 0.15 | 0.61 | 7.2 | 17.5 / 2.86 | 6.2 | 0.64 | 7.4e-03 / 1.3e-01 | all met |
| MuJoCo | `team2` | desired motion, arm damping designed for the inertia the joints see | 0.65 / 0.18 | 0.61 | 7.1 | 28.9 / 5.01 | 10.2 | 0.64 | 5.5e-03 / 1.3e-01 | all met |
| MuJoCo | `team3` | desired motion | 0.70 / 0.22 | 0.84 | 7.6 | 67.2 / 13.38 | 20.4 | 0.64 | 1.1e-02 / 1.7e-01 | all met |
| MuJoCo | `team3` | fitted reference | 0.47 / 0.16 | 0.48 | 7.5 | 85.1 / 14.37 | 21.9 | 0.67 | 1.0e-02 / 1.6e-01 | all met |
| MuJoCo | `team3` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.40 / 0.14 | 0.41 | 7.4 | 50.9 / 9.50 | 13.6 | 0.66 | 1.7e-02 / 1.6e-01 | all met |
| MuJoCo | `team3` | fitted reference, rates of the damping term low-passed at 15 Hz | 0.49 / 0.17 | 0.52 | 7.5 | 55.3 / 12.76 | 16.0 | 0.65 | 1.2e-02 / 1.6e-01 | all met |
| MuJoCo | `team3` | desired motion, damping term without the 15 Hz filter | 0.65 / 0.17 | 0.69 | 7.7 | 73.9 / 14.58 | 21.3 | 0.69 | 1.3e-02 / 1.6e-01 | all met |
| MuJoCo | `team3` | fitted reference, rate gains sqrt(k_p m), rates of the damping term low-passed at 15 Hz | 0.40 / 0.14 | 0.44 | 7.6 | 33.3 / 7.89 | 10.7 | 0.64 | 1.4e-02 / 1.7e-01 | all met |
| MuJoCo | `team3` | desired motion, arm damping designed for the inertia the joints see | 0.66 / 0.18 | 0.64 | 7.7 | 71.2 / 13.91 | 22.2 | 0.64 | 1.3e-02 / 1.7e-01 | all met |
| MuJoCo | `ridgeback2` | desired motion | 0.52 / 0.11 | 0.97 | 10.2 | 27.0 / 5.43 | 5.9 | 0.66 | 4.5e-03 / 3.9e-01 | all met |
| MuJoCo | `ridgeback2` | fitted reference | 0.38 / 0.09 | 0.46 | 10.2 | 33.5 / 6.02 | 7.8 | 0.69 | 4.8e-03 / 3.8e-01 | all met |
| MuJoCo | `ridgeback2` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.27 / 0.07 | 0.44 | 10.2 | 31.0 / 4.95 | 7.6 | 0.68 | 5.8e-03 / 3.9e-01 | all met |
| MuJoCo | `ridgeback2` | fitted reference, rates of the damping term low-passed at 15 Hz | 0.45 / 0.10 | 0.47 | 10.4 | 25.0 / 4.63 | 5.9 | 0.70 | 4.8e-03 / 3.8e-01 | all met |
| MuJoCo | `ridgeback2` | desired motion, damping term without the 15 Hz filter | 0.52 / 0.11 | 0.95 | 10.4 | 36.5 / 6.61 | 8.6 | 0.69 | 4.0e-03 / 3.9e-01 | all met |
| MuJoCo | `ridgeback2` | fitted reference, rate gains sqrt(k_p m), rates of the damping term low-passed at 15 Hz | 0.26 / 0.07 | 0.40 | 10.2 | 33.0 / 4.40 | 7.8 | 0.69 | 5.4e-03 / 3.8e-01 | all met |
| MuJoCo | `ridgeback2` | desired motion, arm damping designed for the inertia the joints see | 0.52 / 0.11 | 0.95 | 10.4 | 29.7 / 5.25 | 7.5 | 0.67 | 4.1e-03 / 3.9e-01 | all met |
| MuJoCo | `ridgeback3` | desired motion | 0.47 / 0.09 | 0.71 | 10.6 | 38.5 / 8.16 | 8.6 | 0.64 | 6.7e-03 / 5.7e-01 | all met |
| MuJoCo | `ridgeback3` | fitted reference | 0.39 / 0.08 | 0.39 | 10.6 | 50.1 / 10.43 | 11.5 | 0.67 | 5.5e-03 / 5.5e-01 | NOT met: grasp_moment (10.6 against 10) |
| MuJoCo | `ridgeback3` | fitted reference, rate gains sqrt(k_p m) on the arm and lift joints | 0.21 / 0.06 | 0.29 | 10.5 | 46.8 / 8.09 | 11.2 | 0.65 | 8.5e-03 / 5.5e-01 | all met |
| MuJoCo | `ridgeback3` | fitted reference, rates of the damping term low-passed at 15 Hz | 0.41 / 0.08 | 0.38 | 10.7 | 37.0 / 8.06 | 9.1 | 0.67 | 7.4e-03 / 5.7e-01 | all met |
| MuJoCo | `ridgeback3` | desired motion, damping term without the 15 Hz filter | 0.43 / 0.09 | 0.69 | 10.6 | 75.0 / 11.02 | 16.5 | 0.69 | 8.1e-03 / 5.6e-01 | NOT met: grasp_moment (13 against 10) |
| MuJoCo | `ridgeback3` | fitted reference, rate gains sqrt(k_p m), rates of the damping term low-passed at 15 Hz | 0.19 / 0.06 | 0.22 | 10.6 | 45.1 / 6.30 | 10.3 | 0.67 | 8.0e-03 / 5.7e-01 | all met |
| MuJoCo | `ridgeback3` | desired motion, arm damping designed for the inertia the joints see | 0.47 / 0.09 | 0.70 | 10.8 | 36.4 / 8.37 | 8.3 | 0.65 | 7.5e-03 / 5.7e-01 | all met |

- `pinocchio_team2_desired` against the stored report `pinocchio_team2`: 271 common entries (timing apart), all equal
- `pinocchio_team3_desired` against the stored report `pinocchio_team3`: 314 common entries (timing apart), all equal
- `pinocchio_ridgeback2_desired` against the stored report `pinocchio_ridgeback2`: 275 common entries (timing apart), all equal
- `pinocchio_ridgeback3_desired` against the stored report `pinocchio_ridgeback3`: 326 common entries (timing apart), all equal
- `mujoco_team2_desired` against the stored report `mujoco_team2`: 284 common entries (timing apart), all equal
- `mujoco_team3_desired` against the stored report `mujoco_team3`: 329 common entries (timing apart), all equal
- `mujoco_ridgeback2_desired` against the stored report `mujoco_ridgeback2`: 290 common entries (timing apart), all equal
- `mujoco_ridgeback3_desired` against the stored report `mujoco_ridgeback3`: 343 common entries (timing apart), all equal
