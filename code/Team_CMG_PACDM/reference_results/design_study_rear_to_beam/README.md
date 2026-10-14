# Design study: Ridgeback bases with their rear to the beam (rejected)

This folder documents a formation of the `ridgeback2` team that was run and rejected. It is not
one of the shipped teams. The main README refers to it under "Ridgeback formation".

## What was different

Each base stood at right angles to the beam with its rear towards it, and the arm worked over the
fixed rear axle. The team then moves sideways along the whole route.

| | this study | shipped `ridgeback2` |
| --- | --- | --- |
| base heading relative to the beam | -90 degrees (rear to the beam) | 0 (in line, facing the direction of travel) |
| arm seed | shoulder pan 3.1416 (over the rear) | shoulder pan 1.36 (to the left) |
| arm base to beam axis at the rack / carrying | 0.85 m / 0.50 m | 0.80 m / 0.45 m |
| start poses of the bases (x, y, yaw) | (-4.4, -0.4, -0.9), (3.4, -1.2, -2.4) | (-4.4, -0.4, 0.9), (3.4, -1.2, -0.7) |
| base position gains | controller defaults (1.6 1/s, 0.4 1/s2) | 3.0 1/s, 1.0 1/s2 |

Robot record, beam, site and bounds were the shipped ones. `inputs/` here holds the team, scene and
mission records of the study; copied over the shipped files of the same names they reproduce the
run with `python workflows/mujoco/run_mission.py --team ridgeback2`.

## What was found

Static margins from the compiled model (payload on level ground, both formations at the reach
they use):

| | rear to the beam | in line |
| --- | --- | --- |
| tipping margin, carrying | 0.123 m | 0.108 m |
| tipping margin, at the rack | 0.138 m | 0.080 m |
| smallest wheel load, carrying / at the rack | 548 N / 495 N | 491 N / 365 N |

The MuJoCo mission of the study (`summary.txt`, `report.json`, figures) completed, with the
supports verified and no forbidden contact, but did not meet three declared bounds:

| | study | bound |
| --- | --- | --- |
| largest grasp force | 96.5 N | 85 N |
| largest grasp moment | 19.0 N m | 10 N m |
| smallest tipping margin | 0.049 m | 0.06 m |

Those numbers are from the audit of that time, which evaluated every fifth record interval.
`by_route_section.txt` gives the largest loads over every interval by part of the mission, for the
study and for the shipped in-line team: in the study the loads are inside the bounds everywhere
except in the ramp zone, where they reach 131 N and 25.8 N m.

## Reading

Fact: the loads rise when a base crosses an edge of the ramp sideways, and they do not when the
bases cross the same edges driving straight.

Explanation offered, not measured separately: with its left and right wheels on different planes
a base rolls, and each of its mecanum wheels stands tilted against the plane it is on. The contact
of a mecanum wheel runs along a roller across the width of the wheel and jumps back at the next
roller; on a tilted wheel that travel (48 mm of the width per roller for this wheel on level
ground) becomes a change of height of the wheel centre, up to about 2.5 mm at 3 degrees, once per
roller. Driving straight over the edge, the wheel axes stay parallel to it and the wheels are not
tilted against their ground.

The stored report and figures were written by the package as it was at the time of the study; the
shipped audit also reports the wrist-sensor readings and evaluates the grasp loads over every
interval.
