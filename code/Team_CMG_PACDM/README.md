# Team_CMG_PACDM: cooperative mobile manipulation

Each robot keeps its own mechanism graph. A team compiler joins the graphs through one payload
body and explicit grasp and support connections into one compiled team graph, and generates from
it the motion, effort and reaction interfaces a cooperative task needs. Two or three mobile
manipulators take a beam out of a rack, carry it through a confined route and install it on two
supports. The same compiler, controller, mission design and audit run for three platforms,

- the **RB-ROBOUT+ with a lift column and a UR30** (`team2`, `team3`),
- the **Clearpath Ridgeback with a UR10e** (`ridgeback2`, `ridgeback3`), and
- the **Clearpath Husky A300 with a UR10e** (`husky2`, `husky3`),

on two plants: MuJoCo and an ideal-constraint plant on Pinocchio dynamics.
The first two platforms stand on mecanum wheels and move in every planar direction. The Husky is
skid-steered: it turns by sliding its wheels and cannot move sideways, so its teams do the same
task with another route, another base loop and a tyre model in the plants
("What the third platform changes").

`DESIGN.md` gives the method, the conventions and the limits. The PACDM kernel in `vendor/` is the
published one, unchanged; everything else here is new.

This copy of the package accompanies the paper "Robot Teams Sharing a Load: Compositional Decay
Certificates" and contains what its results use: the compiler, the controller, the MuJoCo and
Pinocchio workflows, the tools and the reference results of these workflows. The Isaac Sim / PhysX
workflow of the package (`workflows/isaac/`, `backends/isaac_plant.py`, `backends/usd_export.py`, its
tests and its records) is not included; `DESIGN.md` and some comments in the code still describe it.

## Status

| Part | State |
| --- | --- |
| Tests (compiler, skid model, interfaces, controller, fitted feedforward, audit, plants, tyre model) | 198 tests pass (`pytest`) and 9 longer ones (`pytest -m slow`) with Python 3.13 and the versions of `requirements.txt` at the root of the repository; tests that need an optional package are skipped without it. |
| Fitted feedforward (seventh release) | `--feedforward fitted` in the MuJoCo and Pinocchio workflows (teams on rolling wheels): decay tests and complete missions of the four teams on both plants, all complete; the default controller is the one of the sixth release, bit for bit. See "The feedforward of the fitted reference". |
| Pinocchio workflow | Interface validation passes for the six teams (largest deviation 2.5e-8, tolerance 1e-7). The six missions complete with all declared bounds met. |
| MuJoCo workflow | The six missions complete with all declared bounds met. |

The workflow runs behind this table were made with Python 3.12 on Linux and the versions pinned in
`requirements-linux.txt`. Their reports are in `reference_results/`.

## Quick start

Linux, Python 3.12, from this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-validation.txt

python -m pytest                    # about 3 minutes on two CPU cores
python -m pytest -m slow            # about 8 minutes on two CPU cores

python workflows/pinocchio/validate_interfaces.py --team ridgeback2
python workflows/mujoco/run_mission.py --team ridgeback2 --frames 6.25 --video
python workflows/pinocchio/run_mission.py --team ridgeback2
```

`--team` takes `team2`, `team3`, `ridgeback2`, `ridgeback3`, `husky2` or `husky3`. A mission took
between 13 and 79 minutes on two CPU cores shared by several runs, and writes to
`results/<workflow>_<team>/`: `summary.txt`, `report.json`, `record.npz`, figures, and with
`--frames` one still per phase and optionally `mission.mp4`. Rendering without a display uses OSMesa if it is installed (`libosmesa6`
on Ubuntu), else EGL; set `MUJOCO_GL` to choose yourself.

Two options exist for the Husky teams only, for the comparisons in `DESIGN.md`: `--tyre native`
(MuJoCo workflow) leaves the friction of the wheels to the MuJoCo contacts, and `--skid-ground
rigid` (Pinocchio workflow) steps the bases on rigid ground. The results below use neither.

Afterwards:

```bash
python tools/make_report.py --record <run folder>/record.npz     # audit and figures again from a record
python tools/replay_video.py --record <run folder>/record.npz    # video of any record, drawn by MuJoCo
```

The pinned environment is Linux. On Windows, MuJoCo installs from PyPI and Pinocchio from
conda-forge; that combination has not been tested.

## The demonstration

| | `team2` | `team3` | `ridgeback2` | `ridgeback3` | `husky2` | `husky3` |
| --- | --- | --- | --- | --- | --- | --- |
| Robots | 2 RB-ROBOUT+ / UR30 | 3 in line | 2 Ridgeback / UR10e | 3 in line | 2 Husky A300 / UR10e | 3 in line |
| Robot mass | 495 kg | 495 kg | 229 kg | 229 kg | 112.8 kg | 112.8 kg |
| Payload | steel RHS 100 x 60 x 3.6, 5 m, 43.2 kg | steel RHS 120 x 60 x 4, 5.5 m, 59.4 kg | aluminium RHS 100 x 50 x 3, 5 m, 11.7 kg | aluminium RHS 120 x 50 x 3, 5.5 m, 14.6 kg | as `ridgeback2` | as `ridgeback3` |
| Compiled coordinates / actuator ports | 40 / 22 | 57 / 33 | 40 / 20 | 57 / 30 | 38 / 20 | 54 / 30 |
| Team mass | 1033 kg | 1544 kg | 470 kg | 703 kg | 237 kg | 353 kg |
| Rack level and carrying height of the beam | 1.25 m | 1.25 m | 0.90 m | 0.90 m | 0.90 m | 0.90 m |
| Route length | 36.7 m | 37.2 m | 35.9 m | 37.0 m | 33.7 m | 34.6 m |

The site is the same for the six teams: a rack with neighbouring levels, an aisle turn, a 1.7 m
doorway, a cart that crosses the route, a 3 degree ramp, and two mounting pads, turned by 8 degrees,
with guides that leave 6 mm. Only the heights of the rack level and of the mounting pads differ
(beam at 1.56 m on the pads for the RB-ROBOUT+ teams, at 1.16 m for the Ridgeback and Husky teams).

Mission phases: settle, approach, dock, pregrasp, over, descend, latch, take_weight, raise, extract,
stow, navigate, present, insert, lower, transfer, verify, then for each robot in turn release,
retract and fold, then retreat. The set of active constraints changes at latch, at lift-off, at
touchdown and at every release; the compiled interfaces are generated for each of these modes.

### What the second platform changes

Nothing in `teamc/`, `mission/` or `backends/` names a platform. The differences are read from the
robot record and lead to different compiled structure:

| | RB-ROBOUT+ / UR30 | Ridgeback / UR10e |
| --- | --- | --- |
| Tool path | lift column (commanded) and six arm joints (dependent) | six arm joints (dependent); no lift, the arm takes every height |
| Base | rigid, four mecanum wheels | four mecanum wheels on two rockers; the front rocker pivots freely (a passive joint), the rear one is fixed |
| Ground model chosen by the compiler | `fit`: three rows, a plane under the wheels | `contact`: one row per wheel; the rocker angle is a dependent coordinate |
| Wheel loads | resultant only (four wheels on a rigid base are indeterminate); distributed by a declared rule | determinate: the reactions of the four contact rows |
| Support polygon for the tipping margin | rectangle of the four wheels | triangle of the two rear wheels and the front pivot |
| Wrist sensor | declared, 600 N and 60 N m | built into the UR10e flange, 100 N and 10 N m |
| Clamp compliance (compliant clamp, `--latch compliant`) | package default | declared in the robot record (softer) |
| Declared bounds | 450 N, 45 N m, tipping margin 0.10 m, wheel load 150 N | 85 N, 10 N m, tipping margin 0.06 m, wheel load 100 N |

The remaining bounds are common: effort ratio 0.9, traction use 0.6, clearance 0.02 m, payload tilt
12 mrad and position error 20 mm.

### What the third platform changes

The Husky A300 is a skid-steered base: four conventional wheels on two fixed axles, no steering and
no suspension. It moves along its heading and turns by sliding its wheels over the ground; it
cannot move sideways. The first two platforms do every leg of the mission that needs a sideways
move (into the stations, the lane changes, the shift in front of the supports) by simply making
it. For the Husky the same task had to be designed, modelled and simulated differently. The site,
the beams (those of the Ridgeback teams), the phases and the audit are the same, and so are the
declared bounds except for two rows of the table below (traction and tipping margin).

| | Ridgeback / UR10e | Husky A300 / UR10e |
| --- | --- | --- |
| Base | four mecanum wheels on two rockers; moves in every planar direction | four conventional wheels on two fixed axles; drives and turns, no sideways move |
| Drive chosen by the compiler | `rolling`: the wheel rows are ideal constraints, the wheel rates dependent | `friction`: the wheel rows give the slip of the wheels and carry their friction; the wheel rates are independent |
| Tangential ground forces | reactions of the rolling rows | isotropic Coulomb friction at every wheel, a declared law |
| Motion of the base for given wheel speeds | from the rolling rows | from the quasi-static skid model at the present wheel loads: the twist that dissipates the least power |
| Ground model | `contact` (determinate wheel loads) | `fit`; the wheel loads follow a declared rule and enter the skid model, with the load shift of the friction forces solved consistently |
| Formation on the route | rigid: every base keeps one offset and one heading against the beam | the origin of every base keeps its offset; its heading is that of a base that turns about the point the skid model gives |
| Into the stations | sideways | a turn on the spot, a straight leg, a curve into the lane along the rack and a straight run into the station |
| In front of the supports | the formation turns and shifts sideways | the beam is turned and brought to the supports while every base keeps driving forwards and arrives facing them |
| Base loop | position loop in three planar directions | along the heading on the speed; across it (at the turning point) and heading on the yaw rate |
| Wheel friction in the plants | contact friction of the simulator, on rollers | a tyre model applied as loads, in every plant. What carries the wheels has no friction of its own: MuJoCo's soft contact, and the declared tyre springs in the Pinocchio plant |
| Traction bound | per wheel | of the base as a whole (its wheels are at their friction limit in every turn) |
| Other declared bounds | 85 N, 10 N m, tipping margin 0.06 m, wheel load 100 N | 85 N, 10 N m, tipping margin 0.10 m, wheel load 100 N |
| Robot mass | 229 kg | 112.8 kg |

What the compiler, the skid model, the tyre model and the mission design do is in `DESIGN.md`
("Drive of a base", "Tyres of conventional wheels", "Wheel loads and friction in a skid turn",
"Bases that cannot move sideways"). In short:

- **Skid model.** For given wheel speeds the base moves the way that dissipates the least friction
  power. That gives the yaw rate per speed difference and the point about which the base turns,
  from the wheel loads of the present configuration. With equal loads the Husky turns about its
  centre like a differential drive of 1.82 times its track. With the arm and the beam the turning
  point lies between 100 mm behind the base origin and 50 mm in front of it along these routes. For
  the bare platform the model gives an equivalent track of 0.967 m; Clearpath's own controller
  configuration for the A300 uses 0.562 m x 1.75 = 0.984 m (read for this comparison, not used).
- **Route.** A base that turns about a point off its origin moves its origin sideways while it
  turns. The heading of every base along the route is the one that goes with the path of its
  origin and that turning point, the way a trailer follows its hitch. Every base drives forwards
  all the way and crosses the two edges of the ramp squarely. The tightest turn of a base on the
  route has a radius of 1.62 m for `husky2` and 0.88 m for `husky3`; the builder refuses a route
  that needs less than 0.8 m.
- **Tyres.** A wheel of a turning base slides along and across its rolling direction at once.
  PhysX solves the friction of a contact point as two independent rows, which is not Coulomb
  friction for such a slip: a four-wheel box with its load towards one axle turned at 0.36 to 0.75
  of the rate Coulomb friction gives, depending on the rate (PhysX 5.3.1). MuJoCo's
  own friction follows the skid model at low rates only. The plants therefore apply a tyre model as
  loads (Coulomb friction with an elastic range and relaxation, `backends/tyre.py`), as vehicle
  simulation does, and what carries these wheels has no friction of its own.
- **Wheel loads.** The friction of a sliding wheel is proportional to its load, and the friction
  forces shift load between the wheels. For this platform with its arm that loop amplifies a load
  change six- to sevenfold in a skid turn. Stepped on rigid ground, three bases holding one beam
  rocked from one diagonal to the other in the turn in front of the supports. On the normal
  compliance of the declared tyres and on MuJoCo's soft contacts the turn is steady. Rigid ground
  has a second defect: a rigid base on four rigid contacts has no determinate wheel loads, and how
  much one wheel diagonal carries at the cost of the other is left to the contact solver.
- **Arm posture.** The UR10e works in the shoulder configuration that puts its links ahead of its
  base. In the compiled model of the carrying team, the other configuration puts 118 N more on
  each rear wheel than on the front wheel of its side, and a skid turn then takes 36 to 37 N m at
  the rear wheels; with the configuration used here it takes 25 to 26 N m at every wheel.
- **Wheel torque.** Clearpath publishes no drive torque for the A300. The record declares 52.5 N m
  per wheel, the torque at which one axle alone holds the robot with its all-terrain payload on the
  published 30 degree grade. Every skid turn needs the full friction torque of the sliding wheels
  whatever its radius. The missions use up to 0.69 of the bound in MuJoCo and on the Pinocchio
  plant.

## The feedforward of the fitted reference (seventh release)

The joint and wheel loops of the mission controller ask every actuator for a rate. In a mode of the team these
requests are in general not those of one admissible motion: the arms of several robots ask for the same motion of
the payload through different joints, at the measured poses of bases that the wheel loops move. The seventh release
adds a second feedforward to the controller (`--feedforward fitted`, teams on rolling wheels): the effort that the
**fitted reference** requires, the admissible motion closest to the requests in the metric of the rate gains of the
loops. With it, on the compiled model, the energy of the velocity error of the loops obeys an exact dissipation
equality for any positive gains (`DESIGN.md`, "The feedforward of the fitted reference"; the paper that goes with this
release states and proves it). The default controller is the one of the sixth release, bit for bit: every run with
it reproduces the stored reports of the sixth release value for value (timing apart).

```bash
python workflows/mujoco/run_mission.py --team ridgeback3 --feedforward fitted                       # rate gains K = k_d
python workflows/mujoco/run_mission.py --team ridgeback3 --feedforward fitted \
    --option arm_damping_ratio=0.5 --option lift_damping_ratio=0.5                                    # K = k_d / 2 = sqrt(k_p m)
python workflows/mujoco/run_mission.py --team ridgeback3 --feedforward fitted --velocity-filter 15  # with the rate filter
python workflows/mujoco/run_mission.py --team ridgeback3 --monitor                                  # default controller, same quantities recorded
python workflows/mujoco/run_mission.py --team ridgeback3 --monitor --velocity-filter 0              # default controller without its rate filter
python workflows/mujoco/run_mission.py --team ridgeback3 --monitor \
    --option "arm_damping_ratio=[1,1,1,2.2,1.87,2.17]"                                               # default controller, arm damping per joint
python tools/run_decay.py --team ridgeback3 --plant pinocchio --feedforward desired fitted          # decay after a velocity jump
python tools/compare_feedforwards.py --runs results --out results/FEEDFORWARDS.md                   # table of runs
python tools/export_trajectory.py --record results/mujoco_ridgeback3/record.npz --rate 10           # trajectories as a table
```

The same options exist in the Pinocchio workflow.

What the runs of this release show (`reference_results/fitted_feedforward/`, the table below; the missions and the
decay tests with noise-free rate measurements):

- **Plant with ideal constraints (Pinocchio), the controller's own model.** After a velocity jump the energy `W`
  of the velocity error never exceeded its value after the jump or the bound of the statement in the 32 decay runs,
  and `|s|_M` decayed at 1.01 to 3.4 times the certified rate. In the missions the fitted feedforward lowers the
  median of `W` by factors of 83 to 143 (rate gains `K = k_d`) and 22 to 34 (`K = k_d/2`), at least halves the
  position error of the bases and lowers the rms internal force at the grasps on all four teams; the largest effort
  ratio rises on `team2` and `team3` (from 0.63 to at most 0.71), and with `K = k_d` the peak internal force of
  `ridgeback2` from 1.4 to 1.7 N.
- **MuJoCo (compliant rolling contacts), seven controllers per team.** With only the feedforward changed, the largest
  payload error is lower on all four teams: by 9 to 28 % without a rate filter, and by 3 to 30 % with the 15 Hz filter
  of the default controller in the damping term (which leaves the rates inside the fitted effort unfiltered). With half the rate gains on the arm and lift joints it is 33 to 60 % (with
  the filter) and 38 to 51 % (without) lower than with the default controller with the same filter setting, and every
  declared bound is met. Without the filter the peak internal force is 10 to 95 % (default feedforward) and 34 to 88 %
  (fitted) higher, and on `ridgeback3` the grasp moment exceeds its declared bound of 10 N m with either feedforward
  and `K = k_d` (10.6 and 13.0 N m). The default controller with the damping of its arm joints designed for the inertia they see
  (`k_R`) is 17 % better on `team2` than with its own gains and beats the fitted feedforward with the filter and
  `K = k_d` there (0.65 against 0.76 mm; 0.60 mm without the filter); the
  fitted feedforward with `K = k_d/2` and the filter is better on all four teams. On MuJoCo the fitted feedforward does
  not lower the median of `W` (within 18 % of the default controller with `K = k_d`, up to 54 % higher with
  `K = k_d/2`). Every run placed the beam within 0.25 mm of its target.
- **What the statements leave out** (carry stage on the plant with ideal constraints, `ridgeback3` and `team2`). The
  fitted effort contains the measured rates and passes noise in them on to the actuators: the rms grasp moment is 3.9
  to 4.1 times that of the default controller, and at a noise of 5e-3 (rad/s, m/s) the payload error is larger
  (103.7 against 80.4 um and 144.0 against 123.5 um). With measurements 4 ms old the loops oscillate with either
  feedforward when the damping term uses the 15 Hz filter (efforts at their bounds, both teams) and, on `team2`, also
  without it (below the bounds); there, with `K = k_d/2`, they stay stable. Every run of these is in `results/checks/exp12_team2.json` and `results/checks/exp12_ridgeback3.json` of the repository.
- **Computation time** (Python implementation, one core): 16 to 49 ms per 10 ms of mission with the default
  feedforward and 25 to 72 ms with the fitted one; the implementation is not real time.

| MuJoCo, largest payload error while carried (mm) | default, filter, k_d | default, no filter, k_d | default, filter, k_R | fitted, filter, k_d | fitted, no filter, k_d | fitted, filter, k_d/2 | fitted, no filter, k_d/2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `team2` | 0.78 | 0.66 | 0.65 | 0.76 | 0.60 | 0.52 | 0.41 |
| `team3` | 0.70 | 0.65 | 0.66 | 0.49 | 0.47 | 0.40 | 0.40 |
| `ridgeback2` | 0.52 | 0.52 | 0.52 | 0.45 | 0.38 | 0.26 | 0.27 |
| `ridgeback3` | 0.47 | 0.43 (bound not met) | 0.47 | 0.41 | 0.39 (bound not met) | 0.19 | 0.21 |

Every run, on both plants, with the internal forces, grasp moments, effort ratios and the error energy of the loops:
`reference_results/fitted_feedforward/FEEDFORWARDS.md` (written by `tools/compare_feedforwards.py` from the stored
reports, which it also compares with the reports of the sixth release). The reports, summaries and logs of the 40 runs
and the decay tests are in `reference_results/fitted_feedforward/`; the MuJoCo model of every run is the one stored
for the sixth release (`reference_results/mujoco_<team>/plant.xml`, checked byte for byte), and the figures that the
workflows draw for every run are in its folder. The default-controller runs reproduce the stored reports of the
sixth release value for value (271 to 343 entries per run, timing apart).

## Results of the reference runs

| Run | Simulated | Payload error while carried | Internal force, max (rms) | Largest grasp force and moment | Largest effort ratio | Smallest wheel load | Largest traction use | Smallest tipping margin | Placement error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuJoCo, `team2` | 251 s | 0.8 mm, 0.9 mrad | 25.0 N (4.8) | 222 N, 14.7 N m | 0.64 | 584 N | 0.17 | 0.204 m | 0.25 mm |
| MuJoCo, `team3` | 266 s | 0.7 mm, 0.8 mrad | 67.2 N (13.4) | 235 N, 18.4 N m | 0.64 | 589 N | 0.17 | 0.202 m | 0.22 mm |
| MuJoCo, `team2`, compliant clamp and model sensor | 251 s | 1.0 mm, 0.6 mrad | 16.2 N (2.0) | 225 N, 1.4 N m | 0.64 | 546 N | 0.18 | 0.201 m | 0.13 mm |
| Pinocchio, `team2` | 251 s | 0.5 mm, 0.6 mrad | 24.3 N (0.9) | 224 N, 10.4 N m | 0.63 | 589 N | 0.17 | 0.206 m | 0.10 mm |
| Pinocchio, `team3` | 266 s | 0.5 mm, 0.4 mrad | 43.2 N (1.5) | 238 N, 12.5 N m | 0.63 | 608 N | 0.21 | 0.204 m | 0.09 mm |
| MuJoCo, `ridgeback2` | 247 s | 0.5 mm, 1.0 mrad | 27.0 N (5.4) | 66 N, 7.4 N m | 0.66 | 353 N | 0.17 | 0.077 m | 0.04 mm |
| MuJoCo, `ridgeback3` | 264 s | 0.5 mm, 0.7 mrad | 38.5 N (8.2) | 62 N, 7.8 N m | 0.64 | 358 N | 0.16 | 0.079 m | 0.05 mm |
| MuJoCo, `ridgeback2`, compliant clamp and model sensor | 247 s | 0.5 mm, 0.6 mrad | 6.5 N (1.5) | 59 N, 0.3 N m | 0.68 | 352 N | 0.18 | 0.077 m | 0.11 mm |
| Pinocchio, `ridgeback2` | 247 s | 0.2 mm, 0.3 mrad | 1.4 N (0.1) | 64 N, 5.5 N m | 0.67 | 354 N | 0.12 | 0.078 m | 0.01 mm |
| Pinocchio, `ridgeback3` | 264 s | 0.1 mm, 0.2 mrad | 31.3 N (1.1) | 65 N, 7.7 N m | 0.65 | 359 N | 0.12 | 0.079 m | 0.00 mm |
| MuJoCo, `husky2` | 269 s | 0.8 mm, 0.8 mrad | 37.5 N (2.0) | 65 N, 7.2 N m | 0.66 | 173 N | 0.10 | 0.178 m | 0.05 mm |
| MuJoCo, `husky3` | 298 s | 0.7 mm, 0.6 mrad | 30.7 N (2.5) | 61 N, 9.1 N m | 0.69 | 175 N | 0.08 | 0.181 m | 0.04 mm |
| MuJoCo, `husky2`, compliant clamp and model sensor | 269 s | 1.0 mm, 0.4 mrad | 8.1 N (0.7) | 60 N, 0.6 N m | 0.68 | 173 N | 0.10 | 0.180 m | 0.11 mm |
| Pinocchio, `husky2` | 269 s | 0.7 mm, 0.3 mrad | 35.7 N (1.6) | 64 N, 7.5 N m | 0.68 | 173 N | 0.07 | 0.180 m | 0.01 mm |
| Pinocchio, `husky3` | 298 s | 0.7 mm, 0.3 mrad | 28.1 N (2.1) | 59 N, 9.3 N m | 0.69 | 174 N | 0.08 | 0.181 m | 0.01 mm |

All fifteen runs are complete, the supports were verified before the first release, the plants
reported no forbidden contact, and no actuator saturated. All declared bounds are met; the smallest
clearance is 0.044 m for the RB-ROBOUT+ teams, 0.040 m for the Ridgeback teams and 0.040 m for the
Husky teams.

The grasp moment of `husky3` comes closest to its bound: 9.1 and 9.3 N m of 10 N m, the range of
the UR10e sensor. For both Husky teams the largest moment is one 20 ms record interval, the one in
which the rear axle of a base meets the foot of the ramp (the base has no suspension, and the
clamps of these two plants are rigid). In 99.9 % of the intervals with the beam carried every
grasp moment is 4.0 N m or less. With the compliant clamp the largest moment of `husky2` is 0.6 N m
instead of 7.2 N m. The bound is checked on the interval means; single samples are larger (below).

How to read the columns:

- Payload error: position and tilt of the payload against its reference while it is carried.
- Internal force: the part of the grasp wrenches that does not move the payload, relative to the
  declared load distribution (every fifth record interval).
- Grasp force and moment: the largest over every 20 ms record interval with the clamp closed. A
  value is the mean of the wrist-sensor reading over the interval, with the inertia of the
  tool-side body removed. Single 2 ms samples of the MuJoCo plant with its rigid clamp constraints
  are larger: for `ridgeback2` up to 117 N and 22 N m among the recorded samples (one step in
  ten), which is above the range of the UR10e sensor; with the compliant clamp, 62 N and 0.7 N m.
  For `husky3` they reach 70 N and 13.2 N m (two samples above 10 N m, rear axles at the foot of
  the ramp; Pinocchio plant: 66 N and 10.9 N m), for `husky2` 66 N and 6.6 N m, and 61 N and
  0.7 N m with the compliant clamp.
- Effort ratio: actuator effort over its declared bound.
- Traction use: wheel traction over friction coefficient times wheel load, from the recovered
  ground reactions. For the Husky it is the resultant ground force of a base over the friction
  capacity of its wheel loads: its wheels are at their friction limit whenever it turns.
- Placement error: payload pose against the support frame when the supports are verified.

Consistency of the compiled model with the plants, from the same reports:

- RB-ROBOUT+ and Ridgeback: grasp forces recovered by the compiled model from the measured motion
  and efforts agree with the wrist sensors to 0.1 to 1.0 N rms in MuJoCo and to 0.6 N rms or better
  on the Pinocchio plant. The largest differences in a single interval are 1 N for the teams of two
  in MuJoCo (18 N and 4 N with the compliant clamp), 21 N and 6 N for `team3` and `ridgeback3` in
  MuJoCo, and on the Pinocchio plant up to 5 N, except 17 N for `ridgeback3`.
- RB-ROBOUT+ and Ridgeback: the total ground load recovered by the model agrees with the MuJoCo
  contact forces to 0.5 N rms or better. For the RB-ROBOUT+ the load of each wheel is statically
  indeterminate; it is reported (up to about 110 N difference in the mean wheel loads) and not
  bounded. For the Ridgeback the model gives the load of each wheel: the mean wheel loads agree
  with MuJoCo to 6 N, and the loads of single intervals to 13 N rms, of about 570 N.
- Ridgeback rocker: the route crosses the two edges of the ramp squarely, so the pivot stays
  within 0.5 mrad of its middle during the missions. A base standing obliquely across the foot of
  the ramp is the case that needs it: there the compiled contact rows give a rocker angle of
  29.5 mrad and the MuJoCo plant settles at 29.5 mrad (12 poses tried: within 1.1 mrad, base height
  within 0.15 mm; four of them are in `tests/test_contact_ground.py`).
- The largest wrist-sensor readings of the Ridgeback teams, clamp weight included, are 77 N and
  4.7 N m as interval means (range of the UR10e sensor: 100 N, 10 N m).
- Husky, skid model. While a base turns (about a quarter of a mission), the audit compares its
  measured twist with the twist the skid model gives for the measured wheel rates and the compiled
  wheel loads. The yaw rates agree in the mean to 0.1 % (ratio 0.9995 to 1.0008 over all bases and
  both plants), with an rms difference of 2.5 to 4.1 mrad/s in MuJoCo and 0.5 to 1.1 mrad/s on the
  Pinocchio plant. The bases turn at 0.25 rad/s on the spot in the approach and at up to 0.16 rad/s
  (`husky2`) and 0.20 rad/s (`husky3`) on the route. The sideways speed of the base origin differs
  by 2.0 to 2.9 mm/s rms in MuJoCo and 0.5 to 0.8 mm/s on the Pinocchio plant.
- Husky, tyres. The wheels slide at up to 95 mm/s and dissipate up to 77 W per base. Over a mission
  a base turns by 6.3 to 9.0 rad in all and its tyres dissipate 2.0 to 2.9 kJ. The friction forces
  of the sliding wheels in the compiled model differ from the tyre forces the plants applied by 1.3
  to 3.7 N rms. The stability ratio of the explicit tyre loads is 0.44 (limit 0.6).
- Husky, wheel loads. The total ground load agrees with the plants to 0.32 N rms or better. The
  mean load of each wheel under the declared rule agrees with MuJoCo to 5 N, of 250 to 320 N, and
  with the Pinocchio plant to 1 N, where the tyres are the equal springs the rule assumes. On its
  tyres a base stands 1.4 mm lower than on rigid wheels on the Pinocchio plant and 0.2 mm lower on
  MuJoCo's contacts.
- Husky, grasp forces recovered from motion and efforts against the wrist sensors: 0.34 N rms or
  better on both plants; the largest difference in one sampled interval is 11 to 14 N for `husky2`
  (7 N with the compliant clamp) and about 1 N for `husky3`. The largest wrist-sensor readings,
  clamp weight included, are 75 N and 5.5 N m (range of the UR10e sensor: 100 N, 10 N m).
- The compliant-clamp runs use the compliant clamp of `backends/latch.py` and the model sensor
  (`--latch compliant --sensor model`) inside MuJoCo. The clamps deflect by about 1 mm; the stability ratio of the explicit clamp load is 0.32
  for `team2` and 0.31 for `ridgeback2` and `husky2` (limit 0.6).
- With rollers, a mecanum wheel moves sideways more slowly than the ideal rolling rows of the
  compiled model say: 5 % for the Ridgeback wheel in MuJoCo, nothing when driving straight. The
  base position loop takes this up. In MuJoCo the bases of the Ridgeback teams are up to 11 mm
  from their references, where the route moves them sideways (RB-ROBOUT+ teams: up to 8 mm).
- The Pinocchio plant has no contact solver: constraints are exact and the state is projected onto
  them, so its line "displacement vs summed rates" shows that projection where the terrain has a
  crease and at touchdown (up to 0.17 mm per 20 ms). In MuJoCo that line is zero.

`reference_results/` holds, for every run, `summary.txt`, `report.json`, the figures and the console
log, and for the six main MuJoCo runs the generated `plant.xml`. The records (`record.npz`, 23 to 47
MB each), the videos and the stills are not included; the commands above write them again.

### Ridgeback formation

The Ridgeback stands on a triangle (two rear wheels and the front pivot), so the first formation
tried put each base with its rear to the beam, the arm working over the fixed rear axle. Standing,
that is the better one: tipping margin 0.123 m carrying and 0.138 m at the rack, against 0.108 m
and 0.080 m for bases in line with the beam and the arm working to the side (each at the reach it
uses). But bases at right angles to the beam move sideways along the whole route and cross the
two edges of the ramp sideways. That mission was run in MuJoCo and rejected: it completed, but
with a grasp force of 96.5 N (bound 85), a grasp moment of 19.0 N m (bound 10) and a tipping
margin of 0.049 m (bound 0.06). Taken over every record interval, its loads stayed at 65.7 N and
8.1 N m outside the ramp zone and reached 131 N and 25.8 N m inside it. In line, the bases drive
over the edges with their wheel axes parallel to them, and the loads in the ramp zone are at most
59.9 N and 6.0 N m; the largest loads of the shipped `ridgeback2` mission, 65.8 N and 7.4 N m,
occur where the route moves the bases sideways on the plateau. The report of the rejected run,
its inputs and the comparison by part of the route are in
`reference_results/design_study_rear_to_beam/`.

## Layout

```
teamc/        team compiler, closure modules (PACDM), interfaces, loads, margins, terrain, geometry;
              skid.py: friction law and skid model of bases that turn by sliding their wheels
mission/      route and timing, controller, runner, audit, report;
              steering.py: paths of bases that cannot move sideways;
              certificate.py: the fitted reference, its rate and the quantities of the stability statement
backends/     plant description; MuJoCo and Pinocchio plants; clamp and sensor model;
              tyre.py: the tyres of conventional wheels (their friction, and the springs that carry them)
workflows/    mujoco/  pinocchio/        entry points
tools/        build_robot, build_scene, build_mission (inputs); make_report, replay_video;
              run_decay (decay after a velocity jump), compare_feedforwards (table of runs),
              export_trajectory (trajectories of a record as a table)
inputs/       robots, payloads, scenes, teams (physical records); missions (designed task data);
              xacro (top-level files for tools/build_robot.py)
upstream/     unchanged upstream robot description files, with PROVENANCE.json
vendor/       PACDM kernel, unchanged
tests/        unit tests
reference_results/   reports of the runs listed above; fitted_feedforward/: the runs of the seventh release
```

`teamc/` and `mission/` import no simulator.

## Declared data and limits

- The robot records are extracted from the public Robotnik, Clearpath and Universal Robots
  descriptions. Tooling (clamp, sensor body), armature and damping, lift and wheel effort bounds,
  roller geometry and compliance, tyre friction and tyre data and all collision primitives are
  synthetic and listed under `declared` in each record. Several upstream masses are placeholders
  upstream. Nothing here is a manufacturer model.
- RB-ROBOUT+ / UR30: the wrist sensor is a declared one (600 N, 60 N m). The grasp forces of these
  missions (222 to 238 N) are above the 200 N range the record lists for the sensor built into the
  UR30.
- Ridgeback / UR10e:
  - The platform bodies of the upstream description sum to 195 kg; Clearpath's data give a weight
    of 135 kg. The upstream values are used unchanged. A lighter base would have smaller tipping
    margins than those reported here.
  - Clearpath's user manual says "Avoid operating the robot on slopes and grades." The site keeps
    the 3 degree ramp so that both platforms run the same mission; that part of the route is
    outside the manufacturer's advice for the real platform.
  - The chassis collision box (bounding box of the upstream chassis mesh) is 30 mm above the floor;
    the manual gives a clearance of 18 mm. Driving over the edges of the ramp takes 8.3 mm of it.
  - The arm sits on the link `mid_mount` with zero offset, the default of Clearpath's UR10e
    package. The UR control box (12 kg) and the arm cable are not modelled.
  - The upstream description used here (`ridgeback_description` 0.3.5) has a pivoting front rocker
    and a fixed rear one; the newer `clearpath_common` description fixes both.
  - Each wheel has ten rollers of declared size; upstream gives the wheel the inertia of a thin
    disc, which is replaced by that of a solid cylinder of the same mass (details in the record).
  - The beams are aluminium sections chosen so that the share of a robot (at most 57 N) stays
    inside the 100 N range of the UR10e's sensor with the 1.2 kg clamp on it.
- Husky A300 / UR10e:
  - The platform has the mass and the centre of mass of the user manual (78.5 kg with two
    batteries, 41 mm behind and 70 mm above the origin of `base_link`). Upstream the chassis is a
    70 kg box with its centre of mass in the middle; the record keeps the upstream wheel masses and
    gives the chassis the mass and centre of mass that reproduce the manual's values.
  - Upstream places the right wheels 19 mm nearer the centre line than the left ones (the wheel
    mount is not mirrored). The record puts them at the track of the user manual (566 mm).
  - Clearpath publishes no drive torque. The wheel effort bound is declared: 52.5 N m, the torque
    at which one axle alone holds the robot with its all-terrain payload on the published 30 degree
    grade. Everything about the tyre is declared as well: friction 0.8 in every direction,
    tangential and normal stiffness and damping, a relaxation length of 30 mm.
  - No suspension is modelled, as the user manual says for the A300; upstream fixes the
    "suspension beams" to the chassis. A base standing obliquely on an edge of the ramp would stand
    on three wheels, so the routes cross the two edges squarely.
  - The arm sits on the centre of the top plate with zero offset. The UR control box (12 kg) and
    the arm cable are not modelled. The arm, its clamp and its share of the beam are about 40 kg;
    the user manual allows 51.5 kg on all terrain, which it says includes turning in place on
    high-friction ground.
  - Skid steering is modelled with one friction coefficient and a quasi-static skid model, and
    simulated with a tyre model that the plants apply as loads. No plant here solves the friction
    of these wheels in its own contact solver (PhysX cannot, MuJoCo only at low rates). A real tyre
    on a real floor turns the base about a somewhat different point, and its friction depends on
    load, speed and surface.
  - Whether a turning base rocks from one wheel diagonal to the other depends on the normal
    stiffness and damping of its tyres, which are declared. With the declared tyre (the Pinocchio
    plant) and on MuJoCo's soft contacts the turns are steady; on rigid ground with rigid grasps the
    three-robot team rocks (`DESIGN.md`, "Wheel loads and friction in a skid turn"). The margin of a
    real machine against it is not known from this.
- The site, the beams and the missions are design data for the demonstration.
- Three platforms are modelled: two with mecanum wheels and one with conventional wheels on two
  fixed axles. A base with steered or castor wheels, or with conventional wheels on a suspension,
  is not covered.
- Ground following and rolling are ideal in the compiled model. Slip of mecanum rollers, contact
  compliance, the width of a wheel and its rollers exist only in the plants and in the reported
  margins. For the Husky the slip of the wheels is part of the compiled model (friction law and
  skid model); tyre elasticity and relaxation exist only in the plants.
- The payload is rigid and its pose is taken as measured. There is no perception, no localisation
  error and no communication delay.
- Simulation only; nothing was run on hardware.

## Provenance and licences

`THIRD_PARTY_NOTICES.md` and `upstream/PROVENANCE.json` cover the robot description sources
(BSD-3-Clause) and the PACDM kernel. `tools/build_robot.py`, `tools/build_scene.py` and
`tools/build_mission.py` regenerate the shipped inputs byte for byte (the first needs
`requirements-tools.txt`).
