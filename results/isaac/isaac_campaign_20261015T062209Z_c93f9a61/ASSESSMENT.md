# Assessment of the Isaac Sim workflow runs

**CONFIRMED IN ISAAC SIM: 6 team(s), every stage passed and every declared bound met**

Engine: Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2

Environment stage: passed (Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2); required and missing: nothing; optional and missing: nothing

## Stages

| Team | Platform | selftest | preflight | settle | carry | mission | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `team2` | RB-ROBOUT+ / UR30 | passed | passed (94 of 94 checks) | passed | passed | passed (251 s simulated) | confirmed in Isaac Sim |
| `ridgeback2` | Ridgeback / UR10e | passed | passed (94 of 94 checks) | passed | passed | passed (247 s simulated) | confirmed in Isaac Sim |
| `husky2` | Husky A300 / UR10e | passed | passed (107 of 107 checks) | passed | passed | passed (269 s simulated) | confirmed in Isaac Sim |
| `team3` | RB-ROBOUT+ / UR30 | passed | passed (103 of 103 checks) | passed | passed | passed (266 s simulated) | confirmed in Isaac Sim |
| `ridgeback3` | Ridgeback / UR10e | passed | passed (103 of 103 checks) | passed | passed | passed (264 s simulated) | confirmed in Isaac Sim |
| `husky3` | Husky A300 / UR10e | passed | passed (116 of 116 checks) | passed | passed | passed (298 s simulated) | confirmed in Isaac Sim |

## Preflight: what PhysX uses and does against the compiled graph

The worst value of every group of checks, without contact; in brackets the value of the PhysX 5.3.1 stand-in.

| | `team2` | `ridgeback2` | `husky2` | `team3` | `ridgeback3` | `husky3` | limit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| link masses, relative error | 8.4e-08 | 1.0e-07 | 1.0e-07 | 8.4e-08 | 1.0e-07 | 1.0e-07 | 2e-06 |
| link centres of mass, m | 4.1e-09 | 7.3e-09 | 7.3e-09 | 4.1e-09 | 7.3e-09 | 7.3e-09 | 2e-06 |
| link inertia tensors, relative error | 3.0e-07 | 4.2e-07 | 4.2e-07 | 3.0e-07 | 4.2e-07 | 4.2e-07 | 5e-05 |
| joint armature, SI | 4.8e-08 | 4.8e-08 | 4.8e-08 | 4.8e-08 | 4.8e-08 | 4.8e-08 | 1e-06 |
| joint limits, SI | 1.7e-07 | 1.7e-07 | 1.7e-07 | 1.7e-07 | 1.7e-07 | 1.7e-07 | 5e-06 |
| payload inertia tensor, relative error | 2.4e-07 | 3.1e-07 | 3.1e-07 | 2.4e-07 | 2.4e-07 | 2.4e-07 | 5e-05 |
| gravity, m/s2 | 4.2e-07 | 4.2e-07 | 4.2e-07 | 4.2e-07 | 4.2e-07 | 4.2e-07 | 1e-05 |
| roller hulls: authored points outside the cooked hull, m | 3.6e-18 | 3.6e-18 | - | 3.6e-18 | 3.6e-18 | - | 2e-05 |
| configuration written and read back, m or rad | 3.3e-07 | 5.7e-07 | 1.2e-07 | 3.3e-07 | 5.7e-07 | 1.9e-07 | 2e-05 |
| rates written and read back, SI | 4.8e-08 | 3.7e-08 | 4.7e-08 | 5.2e-08 | 1.7e-07 | 2.9e-08 | 2e-05 |
| link positions against the compiled kinematics, m | 1.4e-06 | 1.3e-06 | 9.5e-07 | 1.5e-06 | 1.1e-06 | 1.4e-06 | 5e-05 |
| link orientations against the compiled kinematics, rad | 7.6e-07 | 6.5e-07 | 9.3e-07 | 8.0e-07 | 6.4e-07 | 7.2e-07 | 2e-04 |
| free-flight accelerations against the compiled dynamics, relative error | 1.1e-03 | 2.4e-04 | 4.4e-04 | 1.7e-04 | 6.4e-05 | 3.9e-04 | 2e-02 |
| effect of a known external load on the accelerations, relative error | 1.1e-03 | 2.4e-04 | 4.5e-04 | 3.3e-05 | 1.9e-05 | 3.8e-04 | 3e-02 |
| share of that load left in the following step | 1.3e-03 | 4.3e-04 | 2.2e-04 | 8.8e-04 | 3.9e-04 | 6.0e-04 | 3e-01 |
| displacement of a step outside the reported rates, m or rad | 4.6e-06 | 2.8e-06 | 1.1e-06 | 1.8e-06 | 9.9e-07 | 2.5e-06 | 2e-05 |

Roller hulls, vertices authored and cooked: `team2` 60 and 60; `ridgeback2` 60 and 60; `team3` 60 and 60; `ridgeback3` 60 and 60.

## Missions

| Run | Simulated | Payload error while carried | Internal force, max (rms) | Largest grasp force and moment | Largest effort ratio | Smallest wheel load | Largest traction use | Smallest tipping margin | Placement error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Isaac Sim, `team2` | 251 s | 1.6 mm, 0.9 mrad | 12.7 N (2.8) | 227 N, 1.7 N m | 0.64 | 399 N | 0.27 | 0.185 m | 0.16 mm |
| Isaac Sim, `ridgeback2` | 247 s | 1.7 mm, 1.1 mrad | 13.0 N (2.3) | 60 N, 0.6 N m | 0.68 | 347 N | 0.26 | 0.076 m | 0.10 mm |
| Isaac Sim, `husky2` | 269 s | 0.9 mm, 0.4 mrad | 6.8 N (0.6) | 60 N, 0.6 N m | 0.68 | 173 N | 0.09 | 0.179 m | 0.14 mm |
| Isaac Sim, `team3` | 266 s | 1.4 mm, 0.6 mrad | 43.4 N (9.6) | 241 N, 2.0 N m | 0.65 | 381 N | 0.28 | 0.160 m | 0.12 mm |
| Isaac Sim, `ridgeback3` | 264 s | 1.3 mm, 0.7 mrad | 21.0 N (4.8) | 58 N, 0.6 N m | 0.67 | 358 N | 0.29 | 0.079 m | 0.17 mm |
| Isaac Sim, `husky3` | 298 s | 0.9 mm, 0.3 mrad | 12.5 N (1.0) | 56 N, 0.7 N m | 0.69 | 174 N | 0.09 | 0.181 m | 0.15 mm |

Declared bounds of the missions:

- `team2`: all met; mission status complete; saturated steps 0
- `ridgeback2`: all met; mission status complete; saturated steps 0
- `husky2`: all met; mission status complete; saturated steps 0
- `team3`: all met; mission status complete; saturated steps 0
- `ridgeback3`: all met; mission status complete; saturated steps 0
- `husky3`: all met; mission status complete; saturated steps 0

### Beside the reference runs of the package

`team2`

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| simulated time, s | 251 | - | - | - | - |
| payload position error, max, mm | 1.6 | - | - | - | - |
| payload tilt, max, mrad | 0.9 | - | - | - | - |
| internal force, max, N | 12.7 | - | - | - | - |
| grasp force, max, N | 227 | - | - | - | - |
| grasp moment, max, N m | 1.7 | - | - | - | - |
| effort ratio, max | 0.64 | - | - | - | - |
| wheel load, min, N | 399 | - | - | - | - |
| traction use, max | 0.27 | - | - | - | - |
| tipping margin, min, m | 0.185 | - | - | - | - |
| base tracking error, max, mm | 34 | - | - | - | - |
| placement error, mm | 0.16 | - | - | - | - |

The compiled model against what the plant reported:

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| equations of motion of the compiled model with the measured motion: unbalanced generalized force, rms, N or N m | 9.32 | - | - | - | - |
| payload: the loads applied to it against its measured motion, rms, N | 1.19 | - | - | - | - |
| grasp reactions recovered from the model against the wrist sensors: force, rms, N | 0.83 | - | - | - | - |
| the same, moment, rms, N m | 0.44 | - | - | - | - |
| ground load of the model against the contact report of the plant, rms, N | 0.49 | - | - | - | - |
| displacement of the floating bodies against their summed rates, per record interval, max, um | 54.4 | - | - | - | - |

wrist-sensor balance against PhysX's joint reaction 0.0088 N rms of 203 N; clamp stability ratio 0.32, largest deflection 1.2 mm; contact report: 8 wheels, friction anchors reported.

`ridgeback2`

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| simulated time, s | 247 | - | - | - | - |
| payload position error, max, mm | 1.7 | - | - | - | - |
| payload tilt, max, mrad | 1.1 | - | - | - | - |
| internal force, max, N | 13.0 | - | - | - | - |
| grasp force, max, N | 60 | - | - | - | - |
| grasp moment, max, N m | 0.6 | - | - | - | - |
| effort ratio, max | 0.68 | - | - | - | - |
| wheel load, min, N | 347 | - | - | - | - |
| traction use, max | 0.26 | - | - | - | - |
| tipping margin, min, m | 0.076 | - | - | - | - |
| base tracking error, max, mm | 31 | - | - | - | - |
| placement error, mm | 0.10 | - | - | - | - |

The compiled model against what the plant reported:

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| equations of motion of the compiled model with the measured motion: unbalanced generalized force, rms, N or N m | 5.55 | - | - | - | - |
| payload: the loads applied to it against its measured motion, rms, N | 0.45 | - | - | - | - |
| grasp reactions recovered from the model against the wrist sensors: force, rms, N | 0.27 | - | - | - | - |
| the same, moment, rms, N m | 0.11 | - | - | - | - |
| ground load of the model against the contact report of the plant, rms, N | 0.15 | - | - | - | - |
| displacement of the floating bodies against their summed rates, per record interval, max, um | 48.2 | - | - | - | - |

wrist-sensor balance against PhysX's joint reaction 0.0034 N rms of 58 N; clamp stability ratio 0.31, largest deflection 0.8 mm; contact report: 8 wheels, friction anchors reported.

`husky2`

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| simulated time, s | 269 | - | - | - | - |
| payload position error, max, mm | 0.9 | - | - | - | - |
| payload tilt, max, mrad | 0.4 | - | - | - | - |
| internal force, max, N | 6.8 | - | - | - | - |
| grasp force, max, N | 60 | - | - | - | - |
| grasp moment, max, N m | 0.6 | - | - | - | - |
| effort ratio, max | 0.68 | - | - | - | - |
| wheel load, min, N | 173 | - | - | - | - |
| traction use, max | 0.09 | - | - | - | - |
| tipping margin, min, m | 0.179 | - | - | - | - |
| base tracking error, max, mm | 4 | - | - | - | - |
| placement error, mm | 0.14 | - | - | - | - |
| yaw rate over that of the skid model, by base | 1.000, 1.000 | - | - | - | - |

The compiled model against what the plant reported:

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| equations of motion of the compiled model with the measured motion: unbalanced generalized force, rms, N or N m | 0.07 | - | - | - | - |
| payload: the loads applied to it against its measured motion, rms, N | 0.39 | - | - | - | - |
| grasp reactions recovered from the model against the wrist sensors: force, rms, N | 0.17 | - | - | - | - |
| the same, moment, rms, N m | 0.06 | - | - | - | - |
| ground load of the model against the contact report of the plant, rms, N | 0.13 | - | - | - | - |
| displacement of the floating bodies against their summed rates, per record interval, max, um | 73.6 | - | - | - | - |

wrist-sensor balance against PhysX's joint reaction 0.0003 N rms of 55 N; clamp stability ratio 0.31, largest deflection 0.8 mm; tyre model stability ratio 0.45; contact report: no wheel collider (the tyres carry the wheels), payload on 4 support pads.

`team3`

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| simulated time, s | 266 | - | - | - | - |
| payload position error, max, mm | 1.4 | - | - | - | - |
| payload tilt, max, mrad | 0.6 | - | - | - | - |
| internal force, max, N | 43.4 | - | - | - | - |
| grasp force, max, N | 241 | - | - | - | - |
| grasp moment, max, N m | 2.0 | - | - | - | - |
| effort ratio, max | 0.65 | - | - | - | - |
| wheel load, min, N | 381 | - | - | - | - |
| traction use, max | 0.28 | - | - | - | - |
| tipping margin, min, m | 0.160 | - | - | - | - |
| base tracking error, max, mm | 37 | - | - | - | - |
| placement error, mm | 0.12 | - | - | - | - |

The compiled model against what the plant reported:

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| equations of motion of the compiled model with the measured motion: unbalanced generalized force, rms, N or N m | 10.24 | - | - | - | - |
| payload: the loads applied to it against its measured motion, rms, N | 1.61 | - | - | - | - |
| grasp reactions recovered from the model against the wrist sensors: force, rms, N | 2.36 | - | - | - | - |
| the same, moment, rms, N m | 0.93 | - | - | - | - |
| ground load of the model against the contact report of the plant, rms, N | 0.68 | - | - | - | - |
| displacement of the floating bodies against their summed rates, per record interval, max, um | 72.1 | - | - | - | - |

wrist-sensor balance against PhysX's joint reaction 0.0077 N rms of 209 N; clamp stability ratio 0.30, largest deflection 1.2 mm; contact report: 12 wheels, friction anchors reported.

`ridgeback3`

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| simulated time, s | 264 | - | - | - | - |
| payload position error, max, mm | 1.3 | - | - | - | - |
| payload tilt, max, mrad | 0.7 | - | - | - | - |
| internal force, max, N | 21.0 | - | - | - | - |
| grasp force, max, N | 58 | - | - | - | - |
| grasp moment, max, N m | 0.6 | - | - | - | - |
| effort ratio, max | 0.67 | - | - | - | - |
| wheel load, min, N | 358 | - | - | - | - |
| traction use, max | 0.29 | - | - | - | - |
| tipping margin, min, m | 0.079 | - | - | - | - |
| base tracking error, max, mm | 32 | - | - | - | - |
| placement error, mm | 0.17 | - | - | - | - |

The compiled model against what the plant reported:

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| equations of motion of the compiled model with the measured motion: unbalanced generalized force, rms, N or N m | 5.50 | - | - | - | - |
| payload: the loads applied to it against its measured motion, rms, N | 0.55 | - | - | - | - |
| grasp reactions recovered from the model against the wrist sensors: force, rms, N | 0.68 | - | - | - | - |
| the same, moment, rms, N m | 0.15 | - | - | - | - |
| ground load of the model against the contact report of the plant, rms, N | 0.15 | - | - | - | - |
| displacement of the floating bodies against their summed rates, per record interval, max, um | 53.7 | - | - | - | - |

wrist-sensor balance against PhysX's joint reaction 0.0031 N rms of 55 N; clamp stability ratio 0.29, largest deflection 0.7 mm; contact report: 12 wheels, friction anchors reported.

`husky3`

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| simulated time, s | 298 | - | - | - | - |
| payload position error, max, mm | 0.9 | - | - | - | - |
| payload tilt, max, mrad | 0.3 | - | - | - | - |
| internal force, max, N | 12.5 | - | - | - | - |
| grasp force, max, N | 56 | - | - | - | - |
| grasp moment, max, N m | 0.7 | - | - | - | - |
| effort ratio, max | 0.69 | - | - | - | - |
| wheel load, min, N | 174 | - | - | - | - |
| traction use, max | 0.09 | - | - | - | - |
| tipping margin, min, m | 0.181 | - | - | - | - |
| base tracking error, max, mm | 6 | - | - | - | - |
| placement error, mm | 0.15 | - | - | - | - |
| yaw rate over that of the skid model, by base | 1.000, 1.000, 1.000 | - | - | - | - |

The compiled model against what the plant reported:

| | this run | PhysX 5.3.1 stand-in | MuJoCo, compliant clamps | MuJoCo | Pinocchio plant |
| --- | --- | --- | --- | --- | --- |
| equations of motion of the compiled model with the measured motion: unbalanced generalized force, rms, N or N m | 0.09 | - | - | - | - |
| payload: the loads applied to it against its measured motion, rms, N | 0.51 | - | - | - | - |
| grasp reactions recovered from the model against the wrist sensors: force, rms, N | 0.16 | - | - | - | - |
| the same, moment, rms, N m | 0.06 | - | - | - | - |
| ground load of the model against the contact report of the plant, rms, N | 0.11 | - | - | - | - |
| displacement of the floating bodies against their summed rates, per record interval, max, um | 47.4 | - | - | - | - |

wrist-sensor balance against PhysX's joint reaction 0.0002 N rms of 52 N; clamp stability ratio 0.29, largest deflection 0.7 mm; tyre model stability ratio 0.45; contact report: no wheel collider (the tyres carry the wheels), payload on 4 support pads.

## What the runs were made with

- `team2`: Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2; Python 3.12.14, NumPy 2.3.1, SciPy 1.17.0, Windows-11-10.0.26100-SP0; dt 0.002, PGS 64/8, rigid wheel contact; files identical to this package
- `ridgeback2`: Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2; Python 3.12.14, NumPy 2.3.1, SciPy 1.17.0, Windows-11-10.0.26100-SP0; dt 0.002, PGS 64/8, rigid wheel contact; files identical to this package
- `husky2`: Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2; Python 3.12.14, NumPy 2.3.1, SciPy 1.17.0, Windows-11-10.0.26100-SP0; dt 0.002, PGS 64/8, tyre wheel contact; files identical to this package
- `team3`: Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2; Python 3.12.14, NumPy 2.3.1, SciPy 1.17.0, Windows-11-10.0.26100-SP0; dt 0.002, PGS 64/8, rigid wheel contact; files identical to this package
- `ridgeback3`: Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2; Python 3.12.14, NumPy 2.3.1, SciPy 1.17.0, Windows-11-10.0.26100-SP0; dt 0.002, PGS 64/8, rigid wheel contact; files identical to this package
- `husky3`: Isaac Sim 6.1.0.0, Kit 110.3.0+feature.371399.00c488ae.gl, omni.physx-110.3.2; Python 3.12.14, NumPy 2.3.1, SciPy 1.17.0, Windows-11-10.0.26100-SP0; dt 0.002, PGS 64/8, tyre wheel contact; files identical to this package

## What these runs do not test

- The friction law and the normal contact of the Husky wheels: with `--contact tyre` (what `auto`, the default, gives these wheels) both are the declared tyre model, applied by the adapter; the wheels have no collider.
- The compliant wheel contact (`--contact compliant`), unless the run was made with it.
- Hardware: every plant here is a simulation, with declared drive, tyre and clamp data.
