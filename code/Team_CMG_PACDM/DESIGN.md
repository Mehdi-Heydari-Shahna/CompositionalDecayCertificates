# Design

This copy of the package does not include its Isaac Sim / PhysX workflow (`workflows/isaac/`,
`backends/isaac_plant.py`, `backends/usd_export.py`), the tests of that workflow or its records; the
passages below that describe them are kept for reference.

## Purpose

Each robot keeps its own mechanism graph. The team compiler joins the graphs of several mobile
manipulators through one shared payload body and explicit connections (grasps, supports) into one
compiled team graph, and generates from it the kinematic and force interfaces a cooperative task
needs. A beam mission (pick from a rack, carry through a confined route, place on two supports)
exercises those interfaces on three plants: MuJoCo, an ideal-constraint plant on Pinocchio dynamics,
and Isaac Sim / PhysX. Three platforms run it: the RB-ROBOUT+ with a lift column and a UR30, the
Clearpath Ridgeback with a UR10e, and the Clearpath Husky A300 with a UR10e. Nothing in the
compiler, the controller or the audit names a platform; what differs between them is read from the
robot records. The first two stand on mecanum wheels and move in every planar direction. The Husky
stands on four conventional wheels on two fixed axles: it turns by sliding its wheels over the
ground and cannot move sideways. The compiler finds that in the mechanism, and the model, the
mission and the plants follow from it (sections "Drive of a base", "Tyres of conventional wheels"
and "Bases that cannot move sideways").

The PACDM kernel (`vendor/pacdm_original.py`) is the published one, byte for byte. Everything else
in this package is new.

## Data flow

```
inputs/robots  inputs/payloads  inputs/scenes  inputs/teams        physical records only
        \            |               |             /
                 teamc.compiler.compile_team                        one world-rooted tree, chords,
                          |                                         closure modules, ports, modes
     +--------------------+----------------------+
     |                    |                      |
teamc.chords         teamc.interfaces       backends.plantspec      plant description (rollers,
closure modules      constraints, lift,          |                  hub inertia, colliders)
(PACDM kernel)       reduced dynamics,     +-----+--------------------+
     |               reactions, inverse    |            |             |
     |               dynamics              mujoco_export  pin_model   usd_export
     +----------+---------+                mujoco_plant   pin_plant   isaac_plant + workflows/isaac
                |
        mission.controller  <-- inputs/missions (task data designed by tools/build_mission.py)
        mission.runner      --> record --> mission.audit --> report, figures
```

`teamc/` and `mission/` import no simulator. A plant is reached only through `backends/`.

Three modules are used only for bases that turn by sliding their wheels: `teamc/skid.py` (friction
law and quasi-static skid model), `mission/steering.py` (paths of bases that cannot move sideways)
and `backends/tyre.py` (the tyre model of the plants).

## Conventions

- Homogeneous transforms `aTb` map coordinates of frame b to frame a.
- Twists are angular first `[omega; v]`, wrenches `[moment; force]` (the kernel convention).
- `adj(T)` maps a twist from the frame of `T` to the frame `T` is expressed in; `adj(T).T` maps
  wrenches the other way.
- A floating root is carried by a generated six-coordinate chart `[x, y, z, yaw, pitch, roll]`
  (world translation, intrinsic ZYX), as in the Go2 and Kangaroo compilers. Chart rates are the
  generalized velocities.
- The multiplier of a grasp connection is the wrench the connection applies to the payload,
  expressed in the tool frame (minus side). The wrist sensor reads the wrench the arm side applies
  to the tool-side body, in the sensor frame.
- SI units throughout.

## Inputs

Physical records only: bodies with inertias, undirected joints, a floating root, named frames,
actuator and wheel declarations, drive data, collision primitives; a payload; a scene (terrain
patches, static boxes, support frames, a cart); a team record naming robots, commanded coordinates,
grasp connections and rest connections. Tree directions, coordinate orders, Jacobians, modules and
partitions are not accepted as input; they are generated.

The robot records are extracted by `tools/build_robot.py` from public descriptions:
`inputs/robots/rbrobout_ur30.json` from those of Robotnik and Universal Robots,
`inputs/robots/ridgeback_ur10e.json` and `inputs/robots/husky_a300_ur10e.json` from those of
Clearpath and Universal Robots. What the descriptions do not contain (tooling, armature and
damping, lift and wheel effort bounds, roller geometry and compliance, tyre friction and tyre data,
collision primitives) is listed under `declared` in each record and is a synthetic engineering
value. The site, the beams and the missions are design data for the demonstration.

## What the compiler generates

`teamc/compiler.py`

- one world-rooted tree: a chart for every floating root (robots, payload), robot bodies and joints
  below it, with a fixed coordinate order;
- one chord per connection with its two root paths and its free relative twists (weld, revolute,
  prismatic, cylindrical, universal, spherical);
- one closure module per robot: inputs are the payload chart, the robot's base chart and its
  commanded coordinates on the tool path (the lift of the RB-ROBOUT+; none for the Ridgeback); the
  dependent set is the remaining six tool-path joints, the arm;
- the role of every joint from the structure: wheel joints, actuated joints, and passive joints
  (moving joints without an actuator that are not wheels). A passive joint is accepted only where it
  carries wheels; an unactuated joint anywhere else is refused;
- the ground model of each base and the way its wheels drive it (next two sections);
- actuator ports and reaction ports; the mode table.

### Ground models

The compiler chooses how a base is coupled to the ground from its mechanism.

| | `fit` | `contact` |
| --- | --- | --- |
| Chosen for | a base without passive joints (RB-ROBOUT+) | a base with passive joints that carry wheels (Ridgeback: front rocker) |
| Constraint rows | 3: `[z, pitch, roll] = support(x, y, yaw)`, a plane fitted to the ground under the wheels | one per wheel: the wheel, a thin disc of its radius, touches the terrain |
| Dependent coordinates | z, pitch, roll | z, pitch, roll and the passive joints |
| Condition | at least three wheels | wheels = 3 + passive joints, so that the contacts are statically determinate; refused otherwise |
| Wheel loads | resultant only; the load of each wheel is indeterminate with four wheels | the multipliers of the rows are the wheel loads |

For a disc with centre `c`, axis `a` and radius `r` on a terrain plane `n . x = d`, the height of
its lowest point is `h = n . c - d - r sqrt(1 - (n . a)^2)`, the contact point is `c - r u` with
`u = (n - (n . a) a) / sqrt(1 - (n . a)^2)`, and the constraint row is the normal velocity of the
wheel carrier at that point. On piecewise-planar terrain a plane counts if the nearest point of
the disc lies on its patch; the smallest height among those is taken (at a concave edge the wheel
touches the plane it meets first). Beyond a convex edge no plane counts and the largest height
among the neighbouring planes is used, which differs from rolling over the edge by
`r (1 / cos(angle / 2) - 1)`: 0.03 mm for the Ridgeback wheel on the 3 degree ramp. The width of a
wheel and its rollers exist only in the plants.

### Drive of a base

The compiler also decides from the mechanism how the wheels move a base.

| | `rolling` | `friction` |
| --- | --- | --- |
| Chosen for | mecanum wheels (RB-ROBOUT+, Ridgeback); conventional wheels whose axles lie on one line | conventional wheels on fixed axles that do not all lie on one line (Husky: two axles) |
| Test | | the rows "the ground point of a wheel does not move along its axle", written on the planar twist of the base, have rank above one: the base cannot turn without sliding a wheel |
| Wheel rows | ideal constraints (one per mecanum wheel) | no constraints. Two rows per wheel, along its rolling direction and along its axle, give the slip of the wheel and carry its tangential ground force |
| Wheel rates | dependent coordinates | independent coordinates |
| Tangential ground forces | reactions of the rolling rows | a friction law |
| Motion of the base for given wheel speeds | from the rolling rows | from the quasi-static skid model |

`teamc/skid.py` holds what a friction-driven base needs.

- Declared law: isotropic Coulomb friction at the ground point of every wheel,
  `F = -mu N s / |s|`, with `s` the velocity of the wheel's material point at the ground and `N`
  the normal load of the wheel. In the compiled model the force builds up over a declared slip
  speed (1 mm/s), so that it is a function of the state.
- Skid model. At the speeds of a transport task the inertia forces of a base are small against
  its friction capacity, so the planar twist of the base for given wheel speeds is the one at which
  the friction forces balance. That twist minimises the dissipated power `sum mu N |s|`, a convex
  function, and is found by Newton steps on a smoothed form of it. With the wheels of each side at
  one speed the twist is linear in the mean speed and the speed difference, so three numbers
  describe the base at given loads: the yaw rate per unit speed difference and the turning point
  `(x0, y0)`, the point of the base that stays at rest when the two sides turn at opposite speeds.
- Wheel loads. With equal loads the base turns about its centre like a differential drive whose
  track is `1 + (L / B)^2` times the track of the wheels (wheelbase `L`, track `B`: 1.82 for the
  Husky). The arm and the payload shift the loads, and the turning point moves towards the heavier
  axle; along the routes here it lies between 100 mm behind the base origin and 50 mm in front of
  it. The model therefore takes the wheel loads of the present configuration from the compiled
  ground reactions and not one tuned factor.
- Loads and friction depend on each other. The friction forces act at the ground, below the centre
  of mass, and shift load between the wheels; the friction of a wheel is in turn proportional to
  its load. The ground reactions are affine in the friction forces, so the loads that are
  consistent with their own friction solve a small linear system (`(I - G) N = N0`, one unknown per
  wheel). `forward` and `inverse` use these loads.

What the skid model leaves out: planar loads on a base other than the friction of its own wheels
(slope, inertia, the arm). They are small against the friction capacity in these missions; the
base loop takes them up, and the audit compares the measured twist of every base with the model.

For the bare platform (78.5 kg, centre of mass 41 mm behind the middle of the wheels, as the user
manual gives it) the model gives an equivalent track of 0.967 m. Clearpath's own controller
configuration for the A300 uses a wheel separation of 0.562 m with a multiplier of 1.75, that is
0.984 m: 1.7 % more. That configuration was read for this comparison and is not used anywhere.

`teamc/chords.py` evaluates a chord and its differential on the two paths only and wraps each
robot's module for the unchanged `PACDM` (`correct`, `acquire`, `mapping`).

`teamc/interfaces.py`, for a mode (set of closed grasps, an optional support, ground coupling,
rolling) at a configuration:

- stacked constraint rows with their port table: grasp rows (6 each), support rows (6), the ground
  rows of each base (3 for `fit`, one per wheel for `contact`), mecanum rolling (1 per wheel: no
  slip along the roller axis at the contact point; for a wheel on a rocker the row is written on
  the body that carries it);
- the ground pose of a base from its planar pose (`ground_pose`: closed form or plane fit for
  `fit`, a Newton solve of the wheel heights for `contact`) and the rates of its chart, its passive
  joints and its wheels for a planar twist (`base_motion`; for a friction-driven base the wheel
  rates of the skid model for a forward speed and a yaw rate, since it has no sideways rate to give);
- for a friction-driven base: the slip of every wheel (`wheel_slip`), the friction forces at given
  wheel loads (`friction_loads`), the skid model at given loads (`skid`: turning point, yaw rate per
  speed difference, turning moment), the twist it gives for the present wheel rates (`skid_twist`),
  and the wheel loads from the ground reactions (`wheel_loads`);
- the rank-checked partition and tangent lift `v = E u`, the curvature term `c = E_dot u`;
- reduced dynamics `Mr u_dot + hr = L^T f` with `L = B^T E`;
- the reactions of every port, recovered with the dependent block already factored for the lift;
- inverse dynamics with prescribed grasp wrenches (smallest wheel traction among the solutions; for
  a friction-driven base the friction forces are known from the state, and the wheel efforts follow);
- reaction recovery from a measured motion and measured efforts (least squares, rank reported);
- which reactions rigid-body mechanics determines (identifiability).

`teamc/loads.py` declares load sharing: a weighted minimum-norm distribution of the payload wrench
over the grasps, and the internal wrench as the remainder relative to that distribution.

`teamc/stability.py` turns recovered reactions into margins: wheel normal loads, traction
utilization, tipping margin, support-pad loads. For a `fit` base the wheel loads follow a declared
equal-stiffness distribution of the resultant (with one wheel unloaded, the three-wheel solution)
and the tipping margin is the distance of the centre of pressure from the polygon of the wheels.
For a `contact` base the wheel loads are the reactions of the contact rows, and the margin is
taken to the polygon the loads can reach: a pair of wheels behind a passive pivot supports the
base at one point, where the line through their ground points passes the pivot axis, so the
Ridgeback stands on the triangle of its two rear wheels and the front pivot. Ground coupling and
rolling are ideal in the constraints; friction limits and unilateral conditions are checked on the
recovered reactions. The wheels of a friction-driven base are at their friction limit whenever the
base turns, so for it the traction margin is that of the base as a whole: the resultant ground
force of its wheels over the friction capacity of their loads.

`teamc/geometry.py` computes separation distances of the declared primitives with GJK in pure
Python, so the clearance audit is the same for every plant and for the route designer.

## Plants

All plants expose the same interface: `dt`, `reset(q)`, `measure()` (compiled coordinates and rates,
wrist sensor wrenches), `set_efforts(f)` at the compiled actuator ports, `step()`, `latch(id)`
returning the as-latched grasp frame, `unlatch(id)`, `set_obstacle(position)`, and for the audit
only `contacts()` and `weld_wrench(id)`.

`backends/plantspec.py` derives the physical plant description once: every mecanum wheel carries
ten free-spinning barrel rollers whose envelope is the wheel cylinder (exact profile); the declared
wheel inertia is split between hub and rollers so the totals are unchanged. A conventional wheel
keeps its declared inertia and touches the ground as a sphere of the wheel radius: at one point,
like the thin wheel of the compiled model.

| Plant | Dynamics and contacts | Conventional wheels | Clamps | Wrist sensors |
| --- | --- | --- | --- | --- |
| MuJoCo (`mujoco_plant`) | MuJoCo; roller hulls on the terrain, soft contacts | sphere on the terrain, soft contact; friction from the tyre model (`tyre='model'`, the default) or from the MuJoCo contact (`tyre='native'`) | three point constraints per clamp (forces only), or the compliant clamp (`latch='compliant'`) | MuJoCo force/torque sensors, or the model balance (`sensor='model'`) |
| Pinocchio (`pin_plant`) | Pinocchio inertia and bias; exact constraints from Pinocchio frame Jacobians (for a `contact` base one ground row and one rolling row per wheel, written independently of the compiled rows), KKT solve, projection; no contact solver | no ground rows and no wheel rows: the base stands on the declared normal stiffness and damping of its tyres; friction from the tyre model | rigid | Newton-Euler balance of the tool-side body |
| Isaac Sim (`isaac_plant` + `workflows/isaac`) | PhysX articulations, roller hulls on the terrain, rigid contacts (a compliant roller contact is optional) | no collider under the tyres: the base stands on the declared normal stiffness and damping of its tyres, with the friction of the tyre model, both applied by the adapter (`--tyres springs`, the default). Or a frictionless sphere on frictionless terrain, carried by PhysX, with the tyre friction at the wheel loads PhysX reported for the step before (`--tyres contact`) | compliant clamp (`backends/latch.py`) applied as external loads | Newton-Euler balance of the tool-side body |

The Isaac Sim plant has run in Isaac Sim 6.1.0.0 for `team2`, with the first state of the package:
every stage passed (`reference_results/isaac_sim_first_package/`). For `husky2` it ran there with
the third package, in which PhysX carried the wheels on rigid contacts: the mission was completed
with one of its declared bounds unmet, and that run is the reason the tyres now carry the wheels
("Wheel loads and friction in a skid turn"; `reference_results/study_isaac_rigid_tyre_contact/`).
The present files have not run there. Its adapter runs in the tests on a stand-in for the Isaac API with MuJoCo or with PhysX 5
as the engine (`tests/fake_isaac`, `tests/physx_standin`). For a run in Isaac Sim the workflow is
staged (the controller stack alone, PhysX against the compiled graph without contact, the first
seconds, a stretch of the route, the mission), and
`workflows/isaac/run_campaign.py` runs these stages for the six teams and writes an assessment of
the records (`workflows/isaac/assess_results.py`): which engine made them, with which files, what
passed, and the figures of every mission beside those of the other plants.

The compliant clamp is a six-dimensional spring-damper between the tool frame and the frame it had
on the payload when it closed. Its loads are explicit (evaluated at the start of a step), so the
plant checks a stability ratio when a clamp closes: with `W = J M^-1 J^T` the inverse inertia seen
by the relative twists of the closed clamps and `G = 2 dt D + dt^2 K`, the largest eigenvalue of
`W G` must stay below 4; the plant refuses above 0.6 of that. A robot record may declare the
compliance of its own clamp (`ports.tool.clamp`); the Ridgeback record does, because the default
values, chosen for the heavier UR30 and steel beam, give a ratio above the limit for it.

### Tyres of conventional wheels

A base that turns by skidding needs Coulomb friction in every direction of the ground plane at
once: each wheel slides along and across its rolling direction, and the direction of the friction
force decides about which point the base turns. `backends/tyre.py` is the tyre the plants use for
that: Coulomb friction with an elastic range and relaxation. The ground point of a tyre holds on to
the ground through the tangential stiffness of the carcass, a deflection `e` in the ground plane,

    F = -k e - c s,        de/dt = s - (r |w| / L) e,

with `s` the slip, `r |w|` the speed at which the tread runs through the contact and `L` the
relaxation length. While `|F| <= mu N` the tyre grips; beyond that it slides with `|F| = mu N`.
Three regimes follow. A wheel at rest holds a force with its deflection, which is static friction:
a base stays where it is on the ramp. A rolling wheel with a small slip gives a force proportional
to its slip angle (cornering stiffness `k L`, 4.5 kN/rad as declared). A wheel with a large slip
slides with `F = -mu N s / |s|`, the law of the compiled model. On a wide curve the tyres of a
base stay in the second regime; in the tighter turns of these routes every wheel is in the third.
Stiffness, damping, relaxation length and friction coefficient are declared in the robot record.
The force is evaluated at the start of a step, with the normal load the plant has for the tyre
(the contact force of the step before in MuJoCo, the spring force of the tyre at the same state in
the Pinocchio plant and in the PhysX plant), and applied during the step at the ground point of
the wheel; as for the clamps the plant computes a stability ratio of these explicit loads (at 2 ms
0.44 for the friction alone and 0.45 with the normal springs; limit 0.6).

Why the friction of these wheels is not left to the contact solvers:

- PhysX solves the friction of a contact point as two independent rows. A four-wheel box with its
  load shifted towards one axle then turns at 0.36 of the rate Coulomb friction gives at one speed
  and at 0.75 of it at half that speed (PGS, 64/8 iterations), so the result is not a friction law
  of this form at all; with the TGS solver it turns at a sixth of the rate even with equal loads
  (`tests/test_skid_physx.py`, PhysX 5.3.1). The PhysX plant therefore takes no friction of these
  wheels from PhysX: the tyre model supplies it (and where PhysX has a collider under a tyre, tyre
  and terrain are frictionless). This is how vehicle simulation is usually built: tyre forces
  computed from the slip and applied as loads.
- MuJoCo's friction is isotropic (an elliptic cone), and at low rates a base follows the skid model
  with it. But its friction rows are soft constraints, and they carry a wheel that slides steadily
  only up to a slip speed that grows with their time constant. With the time constant that lets a
  wheel slide at the speeds of the route (0.5 s) the friction opposes `b v + a` instead of `v`, so
  the turning point of a base that drives through a turn lies `0.24 s` times the speed further
  ahead than the skid model says (a lateral drift of 11 mm/s at 0.3 m/s and 0.15 rad/s), and a
  turn on the spot follows the model only up to 0.2 rad/s: at 0.25 rad/s the contacts open and
  close and the base turned at 0.07 rad/s (`tests/test_skid_plant.py`). `tyre='native'` is kept
  for that comparison; the missions are not run with it.

### Wheel loads and friction in a skid turn

The friction of a sliding wheel is proportional to its load, and the friction forces shift load
between the wheels through the height of the centre of mass. In a skid turn this loop favours one
diagonal of the base: a wheel whose friction points backwards and to one side pushes load forwards
and to the other side, onto itself and off the wheel diagonally opposite. For the Husky with its
arm (centre of mass 0.377 m high, wheelbase 0.512 m, track 0.566 m, friction 0.8) the gain of the
loop was 0.83 to 0.85 while all wheels slid in the turn in front of the supports: a change of the
loads is amplified six- to sevenfold. With front and rear wheels pushing against each other the
gain is `mu h / (L / 2)` = 1.18, beyond which rigid-body mechanics has no answer. (A comment in
`backends/pin_plant.py` shows this product with the height rounded to 0.38 m.)

The compiled model solves the loop at a state (the consistent loads above). A plant has to step
through it, and on rigid ground the loads follow the friction with no time scale of their own;
what a time stepping scheme does with the loop is then its own property. The Pinocchio plant was
first written that way (rigid ground rows, tyre friction at the loads of the step before). With
three bases holding one beam, the bases then rock from one diagonal to the other at 4.8 Hz in that
turn: grasp forces up to 235 N (bound 85 N), and a wheel leaves the ground. Two bases drive the
same turn steadily. A tyre is not rigid, and its normal compliance is what gives the loop its
dynamics. The Pinocchio plant therefore steps a friction-driven base on its tyres: no ground rows,
every tyre carries its wheel through the declared normal stiffness and damping of its contact
(2.0e5 N/m, 4.0e3 N s/m), and its friction is the tyre model at that load, all functions of the
state at the start of the step. With that, the turn is steady for both teams, as it is on the soft
contacts of MuJoCo, and on PhysX 5.3.1 with rigid contacts, where the clamps are compliant and do
not tie the three bases together rigidly.

Rigid contacts under such a base have a defect of their own, and the PhysX plant met it in Isaac
Sim. Rigid-body mechanics fixes the total load of a base and its centre; how much one wheel
diagonal carries at the cost of the other is not determined while base and ground are rigid, and a
contact solver fills it in by its own rules. The tyre friction is proportional to the load of
every single wheel, so it follows whatever the solver chose. PhysX 5.3.1 (the stand-in) chose
smoothly: for the bases of `husky2` a share of 51 to 58 N in the mean on the floor and 11 to 12 N
on the plateau, moving by 6 N rms from one 20 ms mean to the next, and wheel torques up to 0.77 of
their bound. PhysX 5.9.1 in Isaac Sim 6.1.0.0 moved the share between successive 20 ms means by
55 N rms, and by up to 330 N, while a base stood on the ground plane, for wheels that carry 286 N
on average; on the boxes of the ramp and the plateau by 10 to 13 N rms. The mission was completed
there with the beam placed; a wheel reached 0.908 of its torque bound and was above the declared
0.9 in two record intervals. With the tyre friction taken at the wheel loads recorded in that
run, the mission on PhysX 5.3.1 needs 0.88 instead of 0.77, at the same wheel; that the excess in
Isaac Sim comes from these loads alone is not shown by a run there. The PhysX plant therefore
steps a friction-driven base as the Pinocchio plant does:
the scene has no collider under the tyres, and the adapter applies to every wheel one force at its
ground point, the normal load of the declared tyre spring (`N = max(k d - c u, 0)` for the depth
`d` of the tyre in the terrain plane it touches and the rate `u` of the wheel centre along the
normal) and the tyre friction at that load, both from the state at the start of the step. The
wheel loads of the PhysX plant are then those of the equal-stiffness rule of the compiled model:
on PhysX 5.3.1 the share of the diagonals stayed within 2.5 N of zero over the missions
of both teams, and the wheel torques within 0.63 (`husky2`) and 0.69 (`husky3`) of their bound, as
on the Pinocchio plant. PhysX's own compliant contact with the same stiffness and damping does
the same on PhysX 5.3.1 (`--tyres contact --contact compliant`: wheel torques of `husky2` within
0.66 of their bound, the share within 18 N of zero, at the edges of the ramp); it leaves the wheel
contact to PhysX. Neither the tyre springs nor the compliant contact has run in Isaac Sim. The
records, the analysis and the comparison runs are in
`reference_results/study_isaac_rigid_tyre_contact/`.

Rigid rows under such a base have a second effect, at the foot of the ramp: a rear axle that meets
the crease can be an impact, with wheel loads of 1.5 to 1.8 kN and grasp forces of 226 to 377 N in
one record interval, for the team of two as well. On tyres the same instants give the largest
grasp moments of the missions (7.5 and 9.3 N m, bound 10 N m).

The declared rigid model is kept for the validation of the compiled interfaces (`evaluate`), and
`PinPlant(..., skid_ground='rigid')` (`--skid-ground rigid` in the workflow) steps the bases the
first way. The two missions run that way are in `reference_results/study_husky_rigid_ground/`.

## Mission and controller

`tools/build_mission.py` designs the task from the physical records: stations, carrying formation,
a route for the payload pose in a generalized length (rotation counted with the payload half
length), lanes centred in the aisles, the aisle turn placed for the largest clearance at every
heading, speed zones, a gate before the cart crossing, the approach in two legs, thresholds and
declared admissible bounds.

### Bases that cannot move sideways

The mecanum teams move as a rigid formation: every base keeps one offset and one heading against
the payload, sideways where the route says so. A friction-driven base moves along its heading and
turns, so the same task is designed differently for it (`build_steered`, with the planar geometry
of `mission/steering.py`). The site, the stations, the payload route through the aisles and the
phases stay the same.

- The origin of every base still keeps its place in the formation. Its heading is free: the arm,
  not the base, holds the payload, and a base may turn under its arm while the grasp stays where it
  is. The mission record carries the pose of every base along the route (`route.bases`).
- Heading. A base that turns about a point `x0` in front of its origin (behind it when negative)
  moves its origin sideways at `-x0` times its yaw rate. For the path of the origin the heading
  follows from `d theta / ds = -sin(phi - theta) / x0`, with `phi` the direction of the path: with
  the turning point behind the origin the base follows its origin as a trailer follows its hitch,
  with the turning point in front the equation is integrated backwards from the end
  (`drawn_headings`; each step is the closed-form solution for a path that turns at a constant
  rate, so the points may be further apart than `|x0|`). `x0` comes from the skid model at the
  wheel loads of the static carrying configuration at that place of the route.
- The positions through the aisle turn come from a search on a grid. A rigid formation does not
  mind the small steps between them; a base that heads the way it moves would have to follow each
  one, so the run through the turn is smoothed (a Gaussian of 0.5 m of route), and the clearance is
  evaluated again on the team as it really stands. The builder refuses a route on which a base
  would have to turn more tightly than a declared radius (0.8 m).
- Stations. The bases cannot enter their stations sideways. They turn on the spot towards a point
  of the lane along the rack, drive there, curve into the lane and run the last metre straight into
  the station; each base has its own point of entry so that they do not meet.
- Plateau. The mecanum formation turns and shifts sideways in front of the supports. Here the
  payload is turned and brought to the supports while the bases keep driving forwards. Prescribing
  the tracks of both end bases is singular where both head across the beam (two sliders and a bar
  at dead centre), so the leading base gets a designed track and the payload yaw is the free
  quantity: it keeps the last base on its lane until that base has left the ramp, then blends into
  the yaw of the supports. Every base crosses the two edges of the ramp squarely (the platform has
  no suspension) and arrives facing the supports.
- The speed limit along the route also keeps the yaw rate of every base at 0.15 rad/s or less,
  because a turning base slides its wheels. The speed governor follows its limit with a lag, as it
  does for every team, so where the limit falls quickly the route speed is above it for a moment
  (by up to 29 % in these runs) and a base turns faster: up to 0.16 rad/s for `husky2` and
  0.20 rad/s for `husky3`. The turns on the spot of the approach, without load, are made at
  0.25 rad/s.

`mission/controller.py` runs two rates: a fast loop (every physics step: joint PD on the arm and lift
references, wheel speed loop) and a slow loop (10 ms: references, feedforward, force regulation).
Phases: settle, approach, dock, pregrasp, over, descend, latch, take_weight, raise, extract, stow,
navigate, present, insert, lower, transfer, verify, then per robot release, retract_up, retract_back,
fold, then retreat, done.

- Arm references come from the closure modules (payload pose, own base pose, and the commanded
  lift where the robot has one; without a lift the arm takes every change of height).
- Wheel speeds come from the compiled rolling rows. A friction-driven base has no sideways rate to
  command: the error along its heading acts on its speed, and the error across it (taken at the
  turning point and scaled by the speed) and the heading error act on its yaw rate. The wheel
  speeds for that forward speed and yaw rate come from the skid model at the wheel loads of the
  last feedforward. The yaw-rate terms act only while the reference moves: a base at rest keeps the
  heading it stopped with, and its arm takes up the difference.
- Feedforward efforts come from the compiled inverse dynamics with the grasp wrenches of the
  declared load distribution.
- The wrist sensors give the grasp wrenches actually applied; a tool-frame admittance acts on their
  deviation in the directions the mode leaves free (the internal wrench while carrying, every
  direction while the payload rests on supports).
- The weight is taken and handed back as a ramp of the commanded share; `take_weight` ends when the
  sensors show the weight in the clamps, and holds with the payload on its supports if they do not.
  `verify` accepts the placement only if the grasps are unloaded, the payload is at rest, the
  supports carry it and it lies within tolerance; otherwise the team holds without releasing.
- The speed governor follows the zone limits and stops at a closed gate with a bounded deceleration.

Gains, durations and thresholds are declared in the controller and the mission record; they are not
compiler outputs. A mission record may carry controller options for its platform (`controller`);
the Ridgeback missions set a lower arm stiffness and wheel gain than the defaults, for the lighter
arm and base, and a firmer base position loop. The Husky missions add an integral term to the
wheel speed loops: a wheel that slides needs its full friction torque at any speed.

### The feedforward of the fitted reference

The loops of the controller ask every actuator for a rate. A joint loop with gains `k_p`, `k_d` applies
`-k_d (rate - w)` with `w` = rate of its reference + `k_p / k_d` times its position error; a wheel loop applies
`-k_w (rate - w)` with its reference rate `w`. In a mode of the team the actuators cannot follow arbitrary
requests: their rates are `L u` for the independent rates `u` of the mode (`L = B^T E`, the actuator-port map of
the compiler). The arms of several robots ask for the same motion of the payload through different joints, at
the measured poses of bases that the wheel loops move, so the requests `w` are in general not of this form.

`mission/certificate.py` computes the **fitted reference** of the loops: the admissible motion closest to their
requests in the metric of their own rate gains `K`,

    u_r = (L^T K L)^-1 L^T K w,     v_r = E u_r,     w = L u_r + w_perp .

Two facts follow from the normal equations. The feedback of the loops is `-K B^T (v - v_r)` plus the effort
`K w_perp`, which moves nothing (`L^T K w_perp = 0`) and only loads the connections. And if the feedforward is
the effort that the fitted reference requires,

    M v_r' + C(q, v) v_r + g + R v_r = B f_r + A^T lam_r,     v_r' = E u_r' + E' u_r  along the measured motion,

then on the compiled model the energy `W = 1/2 |v - v_r|_M^2` obeys `W' = -(v - v_r)^T (B K B^T + R) (v - v_r)`:
it decays between twice the smallest and twice the largest generalized eigenvalue of
`(E^T (B K B^T + R) E, E^T M E)`, for any positive gains. The inverse dynamics of the desired motion is a
different effort. With it the same loops are not covered by this statement, and for position loops the damping
of the fitted error is reduced by `M_r (L^T K L)^-1 L^T K_p L`; with the gains of `team2` and `team3` the
difference has one negative direction at the carrying configuration (the wrist joints, whose rate gains were
designed for the inertia of the wrist alone, see up to 17 times as much in the closed chain). The loops are stable
there all the same; what is lost is the statement and its rate.

The controller has both feedforwards (`feedforward` in the controller options, `--feedforward` in the three
workflows):

- `desired` (the default, and the only one up to the sixth release): `inverse(q, v_ref, a_ref)` with the declared
  distribution of the payload wrench. Nothing of this path was changed: 20 short stages of the six teams on both
  plants give the same records bit for bit as the sixth release, and the eight missions of the teams on rolling
  wheels give its stored reports value for value.
- `fitted`: the effort of the fitted reference, for teams on rolling wheels. Loops, references, load distribution
  and admittance are the same; the rates of the damping term are not filtered unless `velocity_filter_hz` is
  given (below).

The statement holds for any positive rate gains, so they are a choice of their own. With the gains of the earlier
controller (`k_d = 2 sqrt(k_p m)` for the estimated inertia `m` of a joint, damping ratio 1) the fitted effort adds
rate feedback: for one joint it is `m (a_d'' + (k_p / k_d)(a_d' - a'))`, so the loop feeds the rate error back with
`k_d + m k_p / k_d` in all, a quarter more than the critically damped `2 sqrt(k_p m)`. With
`--option arm_damping_ratio=0.5 --option lift_damping_ratio=0.5` the rate gains of the arm and lift joints are
`sqrt(k_p m)`, and each loop with its fitted effort is, for its own inertia, the critically damped loop of the earlier
controller with the feedforward of the desired motion; the loops differ from the earlier ones only in what couples
the joints. The runs of the seventh release use both (README).

How `fitted` is evaluated:

- *With the references, every 10 ms* (`_fitted`): the lift `E` of the present mode at the measured configuration;
  the fit; `E'` along the measured motion (`tangent_rate`: a centred difference of the compiled rows, the
  dependent block of the partition reused); the rate of the requests along the motion; `u_r'`; the left side of
  the balance from the tree dynamics, with `C(q, v) v_r` as the polar form of the velocity-product effort; and
  `TeamModel.inverse` with the grasp wrenches of the declared distribution.
- *The rate of the requests* is where the measured motion enters. An arm reference is solved by the closure
  module at the measured base and lift, so it moves with their measured rates through the tangent map of the
  module, with the rate of its approach offset, and with the rate at which the admittance moves its offset
  (`_offset_as_payload_rate`, `_exp_rate`). A wheel reference moves with the rate of the twist command of the
  base loop, which feeds the measured rates back, and with the rate of the map from the planar rates of a base to
  its wheel rates (the wheel rows of `E'`). None of this needs an acceleration.
- *In every step, 2 ms* (`_fast_fitted`): the loops, with requests that have moved since the update, and the
  fitted effort of the last update plus `Gamma (w'(now) - w'(update))`,
  `Gamma = K L (L^T K L)^-1 M_r (L^T K L)^-1 L^T K`, for the rate `w'` of the requests. This follows the part of
  the fitted effort that varies as fast as the loops: the measured rates in the rate of the requests.
  `feedforward_rate='fast'` evaluates everything in every step instead; on the plants the two agree (see the
  README).
- *How the requests move between two updates* (`advance`, `_requested`, `_requested_rate`). The requests depend on
  the time through the references and on the measured configuration: an arm reference is solved at the measured
  base and lift, and the twist command of a base feeds its measured pose back. With `advance='positions'` (the
  default) they follow the measured configuration by what it has moved since the update, through the maps of the
  update (the tangent map of the closure module, the gain of the pose loop times the wheel rows of the lift), and
  the time to second order. The rate of a reference in this advance is the rate its own samples have at the update
  (`_centred`): the speed governor of the route steps its position with the rate of the step, so its rate is half
  an update behind its position, and the time laws of the other phases hand over values of one instant; the last
  difference of the samples plus half the acceleration times the interval is right for both, and with it the
  advanced reference arrives at the next sample. With `advance='rates'` the requests move at their rates along the
  measured motion at the update times the time, as in the first form of this controller. The two differ at second
  order while the rates stay what they were at the update. They differ at first order when the rates change within
  an update interval: after a velocity jump right behind an update, the requests advanced with the positions are
  at the next update as close to the freshly computed ones as without the jump, and those advanced with the rates
  miss them by more than twenty times as much; and while a team speeds up with the payload on the plant with ideal
  constraints, the error energy of the loops is less than a quarter (tests).
- The damping term uses the measured rates as they are (`velocity_filter_hz` is 0 with `fitted` unless it is
  given): the statement is for the measured rates. When the filter is given it acts on the damping term only; the
  fitted effort, whose rate of the requests contains the measured rates, keeps them unfiltered. That variant is
  outside the statement.
- If a lift or a fit cannot be formed at a state (a rank is lost), that update uses the inverse dynamics of the
  desired motion and the record counts it (`fit:unfitted`); in the runs stored here the count is zero.

What stays outside the statement: the two sampling rates (every reference update is a small jump of the fitted
reference, by what the advance of the requests missed), the bound of 3 rad/s^2 on the rate of the arm reference
rates, and whatever the plant does that the ideal rows do not describe (contact compliance, rollers, slip). The
first two are small on the plant with ideal constraints, where the error energy of a mission is two orders of
magnitude below that of the other feedforward; the third sets the level in MuJoCo, the same for both (the numbers
are in the README).

Friction-driven bases (the Husky teams) are refused with `fitted`: with every actuator locked the formation can
still slide over the floor, so `L^T K L` is singular, and the reference of such a team would have to integrate
its own balance along those directions with the friction of its slip at the measured wheel loads. That
controller is not written.

With `monitor` (always on with `fitted`) the record holds, at every reference update: `W` (`fit:W`), the same
for the measured independent rates alone, the dissipation of the statement and its integral, the two rates of
the present mode, the largest share of an actuator bound taken by `K w_perp`, the fitted reference and the
requests. The audit reports them (`loops` in `report.json`, one line of the summary, `loops.png`).

`tools/run_decay.py` shows the statement on a plant: the team holds the payload, the plant is given a velocity
jump it can have in the carrying mode, and `W` is followed at every step with the two exponentials.

## Audit

`mission/audit.py` works on the record alone (measured states, applied efforts, sensor readings,
events), the same for every plant: tracking, grasp loads and internal wrench, actuator demand,
ground margins, clearances by category, load hand-over at pickup and placement, declared bounds, and
the consistency of the compiled model with the plant:

- grasp wrenches recovered by the compiled model from the measured motion and efforts, against the
  wrist sensors;
- total ground load recovered by the model against the plant's contact forces, where it reports
  them, and for `contact` bases the load of every wheel against the plant's;
- the travel of passive joints and their distance from their limits;
- for friction-driven bases: the largest wheel slip, the friction power and the energy the tyres
  dissipate, the measured twist of the base against the twist of the skid model for the measured
  wheel rates and loads (yaw rate ratio, yaw rate and sideways errors), and the friction of the
  sliding wheels against the plant's. The skid model has no twist to give where one wheel alone, or
  one side alone, carries load; such samples are counted in the report;
- the largest wrist-sensor reading against the range the robot record declares (reported, not a bound);
- payload wrench balance and connection gaps;
- measured displacement against the sum of the measured rates (shows a plant that integrates in
  single precision).

`tools/make_report.py` repeats the audit from a stored record; `tools/replay_video.py` draws a record.

## Limits

- Robots are trees of fixed, revolute and prismatic joints on one floating root; one rigid payload.
- Passive joints are supported where they carry wheels and make the wheel contacts determinate
  (three wheels plus one per passive joint). A passive joint on the tool path and an actuated joint
  between the base and a wheel are refused; springs on passive joints are not modelled.
- Ground following and rolling are ideal in the compiled model; contact compliance, slip, the
  width of a wheel and its rollers exist only in the plants and in the margins. With rollers, a
  mecanum wheel moves sideways more slowly than the ideal rolling row says (5 % for the Ridgeback
  wheel in the MuJoCo plant, none when driving straight); the base position loop takes this up.
- Friction-driven bases: isotropic Coulomb friction with one coefficient, declared; four wheels on
  a rigid base without passive joints. The skid model is quasi-static. The tyre model has no
  contact patch, no aligning moment, no rolling resistance and no dependence of the friction
  coefficient on load or speed; a real tyre on a real floor will turn the base about a somewhat
  different point, which the base loop has to take up as it takes up the differences here.
  Neither the friction nor, by default, the normal load of these wheels is simulated by PhysX
  (see "Tyres of conventional wheels" and "Wheel loads and friction in a skid turn").
- The payload pose is taken as measured.
- Wheel normal loads of a rigid four-wheel base are statically indeterminate; the model reports
  them under a declared equal-stiffness rule and compares them with a plant in the mean. For the
  Husky they also enter the skid model. MuJoCo's soft contacts follow the rule, and so do the tyre
  springs of the Pinocchio and the PhysX plant; rigid PhysX contacts (`--tyres contact`) leave the
  share of the two wheel diagonals to the solver.
- Drive data, tooling, roller data and several upstream masses are not manufacturer values (see
  `declared` in the robot records).
