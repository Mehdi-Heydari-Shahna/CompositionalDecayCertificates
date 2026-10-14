# Study: the Husky teams stepped on rigid ground

The shipped missions of `husky3` and `husky2` on the Pinocchio plant, with the bases stepped on
rigid ground. This is not how the plant steps these bases by default, and both runs fail their
declared bounds. `DESIGN.md` refers to this study under "Wheel loads and friction in a skid turn".

    python workflows/pinocchio/run_mission.py --team husky3 --skid-ground rigid --out results/pinocchio_husky3_rigid_ground
    python workflows/pinocchio/run_mission.py --team husky2 --skid-ground rigid --out results/pinocchio_husky2_rigid_ground

`husky3/` and `husky2/` hold `summary.txt`, `report.json` and the figures of the two runs.

## What was different

| | this study (`--skid-ground rigid`) | shipped (`--skid-ground tyres`) |
| --- | --- | --- |
| ground under a base | three rigid rows: the base follows the plane under its wheels | none: every tyre carries its wheel through its declared normal stiffness and damping (2.0e5 N/m, 4.0e3 N s/m) |
| wheel loads | from the reactions of those rows, by the declared rule | the spring forces of the tyres |
| friction of a tyre | tyre model at the load of the step before | tyre model at the load of this state |

Inputs, controller, tyre model and timestep are the shipped ones.

## What was found

Both missions run to the end and place the beam. Both exceed their declared bounds on the grasp
loads (85 N, 10 N m) and on the smallest wheel load (100 N), in two kinds of places and nowhere
else. For `husky2` the audit also reports the tipping margin as unmet, with a value that has no
meaning (end of this section).

The table gives the largest grasp force and moment (interval means of 20 ms, as in the audit) and
the range of the wheel loads the plant reported, by part of the route, with the same parts of the
shipped runs beside them.

| Team | Part of the route | Rigid ground | On tyres (shipped) | MuJoCo |
| --- | --- | --- | --- | --- |
| `husky3` | foot of the ramp, rear axle of `r3` (t = 178.6 s) | 377 N, 87.5 N m; 197 to 1784 N | 59.5 N, 9.3 N m; 218 to 366 N | 58.8 N, 9.1 N m; 217 to 358 N |
| `husky3` | turn in front of the supports (t = 198.5 to 204.5 s) | 235 N, 51.9 N m; 0 to 575 N | 55.0 N, 0.8 N m; 206 to 375 N | 55.0 N, 1.0 N m; 204 to 374 N |
| `husky3` | rest of the time the beam is carried (165 s) | 64.1 N, 7.1 N m; 175 to 406 N | 58.1 N, 6.0 N m; 174 to 406 N | 61.5 N, 7.0 N m; 174 to 415 N |
| `husky2` | foot of the ramp, rear axle of `r2` (t = 157.5 s) | 247 N, 59.6 N m; 95 to 1546 N | 61.7 N, 7.3 N m; 213 to 371 N | 61.8 N, 6.7 N m; 215 to 362 N |
| `husky2` | foot of the ramp, rear axle of `r1` (t = 169.5 s) | 226 N, 55.2 N m; 112 to 1451 N | 64.0 N, 7.5 N m; 215 to 371 N | 64.7 N, 7.2 N m; 214 to 362 N |
| `husky2` | rest of the time the beam is carried (160 s) | 62.0 N, 5.6 N m; 173 to 409 N | 59.8 N, 3.3 N m; 173 to 409 N | 63.3 N, 5.3 N m; 173 to 408 N |

1. **Rocking in the turn in front of the supports, three bases.** From t = 199.0 s, for 4.6 s
   (route length 28.7 to 29.6 m), the three bases rock from one wheel diagonal to the other. The
   front left and the rear right wheel of every base swing against each other at 4.8 Hz (standard
   deviation of their loads 43 to 122 N), the other two wheels hardly (16 to 25 N). A wheel of
   `r3` leaves the ground, the grasp forces reach 235 N, and the grasp loads are beyond their
   bounds for 2.9 s in all. The two-robot team drives the same turn steadily: it is part of the
   last row of the table.
2. **Foot of the ramp, both teams.** On rigid ground a rear axle that meets the foot of the ramp
   can be an impact: in one record interval the wheel loads of the plant reach 1.5 to 1.8 kN
   (about 300 N standing), the grasp forces 226 to 377 N and the grasp moments 55 to 88 N m;
   `husky2` then stays beyond its bounds for 0.2 s. It happened at three of the five crossings of
   a rear axle (both bases of `husky2`, and `r3` of `husky3`) and not at the other two. On
   tyres the same instants give the largest grasp moments of the shipped runs (7.5 and 9.3 N m).

Outside these places the rigid-ground runs agree with the shipped ones (last row of each team).

At the impacts the model quantities of the audit lose their meaning: the report of `husky2` shows
recovered wheel loads down to -1000 N and a tipping margin of -19.8 m there. At a few samples a
base has no load on one side, where the skid model gives no twist; the reports count them.

## Why the bases rock

The friction of a sliding wheel is proportional to its load, and the friction forces shift load
between the wheels through the height of the centre of mass: a wheel whose friction points
backwards and to one side pushes load forwards and to the other side, onto itself and off the
wheel diagonally opposite. For this platform with its arm the gain of that loop is 0.83 to 0.85
while all wheels slide, so a change of the loads is amplified six- to sevenfold. On rigid ground
the loads follow the friction with no time scale of their own, and with the loads of the step
before the loop closes within a few steps. A base then alternates between gripping with its front
and with its rear wheels, and the stiff arms pass that on to the beam and to the other bases.

A tyre is not rigid, and its normal compliance is what gives the loop its dynamics. With the
declared tyre the same turn is steady (`reference_results/pinocchio_husky3/`), as it is on MuJoCo's
soft contacts and on the rigid contacts of PhysX 5.3.1 with the compliant clamps (stand-in
campaign of 5 October, no longer kept in the package).

The impacts at the foot of the ramp were not analysed further. The plane that is fitted under four
wheels has a kink where an axle crosses a crease, and rigid rows pass it on within one step. The
mecanum bases, which the Pinocchio plant does carry on rigid rows, stay within their bounds over
the same creases.

## What this does not say

How a real Husky behaves. Normal stiffness and damping of its tyres are declared here, not
measured, and the margin against the rocking depends on them. A real floor has no sharp crease
either.
