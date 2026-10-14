# Third-party notices

## Robot description sources

`upstream/robotnik_description/` contains files of the Robotnik description package,
<https://github.com/RobotnikAutomation/robotnik_description>, commit
`4bc73425d090ead4591a7091e7ef7e7dc4fe862a`, Copyright (c) 2025 Robotnik Automation S.L.,
BSD-3-Clause. The licence text is in `upstream/robotnik_description/LICENSE`.

`upstream/ur_description/` contains files of the Universal Robots ROS 2 description,
<https://github.com/UniversalRobots/Universal_Robots_ROS2_Description>, commit
`6662e15f32c23c12ece57d0050aee9a716e4fc41`, BSD-3-Clause. The licence text is in
`upstream/ur_description/LICENSE`.

`upstream/ridgeback_description/` contains files of the Clearpath Ridgeback description (package
`ridgeback_description` 0.3.5), <https://github.com/ridgeback/ridgeback>, commit
`7f2489993f1369906e808402e8f45e4c2aebf5b3`, Copyright 2021 Clearpath Robotics Inc., BSD-3-Clause.
The licence text is in `upstream/ridgeback_description/LICENSE`.

`upstream/clearpath_platform_description/` contains files of Clearpath's platform description
(package `clearpath_platform_description` 2.9.17 of
<https://github.com/clearpathrobotics/clearpath_common>), commit
`726d7fba367f18e53899d3ac03f0b276380ae847`, Copyright (c) 2023, clearpathrobotics, BSD-3-Clause.
The licence text, the licence file at the root of that repository, is in
`upstream/clearpath_platform_description/LICENSE`.

The files are unchanged copies. `upstream/PROVENANCE.json` records their Git blob ids, sizes and
SHA-256 hashes. No meshes are included.

`inputs/robots/rbrobout_ur30.urdf` and `inputs/robots/rbrobout_ur30.json` are generated from the
Robotnik and Universal Robots files, `inputs/robots/ridgeback_ur10e.urdf` and
`inputs/robots/ridgeback_ur10e.json` from the Clearpath and Universal Robots files, by
`tools/build_robot.py` (xacro, then reduction to physical records); each is derived from both of its
sources. `inputs/xacro/ridgeback_ur10e_extras.urdf.xacro` was written for this package: it places
the UR10e on the Ridgeback the way `ridgeback_ur10_e_description.urdf.xacro` of Clearpath's
`ridgeback_manipulation` repository does (commit `748d284978610bab37984847700497ca35a71575`, read,
not copied). `inputs/robots/husky_a300_ur10e.urdf` and `inputs/robots/husky_a300_ur10e.json` are
generated from the Clearpath platform description and the Universal Robots files.
`inputs/xacro/husky_a300_ur10e.urdf.xacro` was written for this package: it assembles the A300 the
way Clearpath's description generator does for the sample configuration `a300_default.yaml` of
`clearpath_config` (commit `4897cd1311deabadb887293eb300dcb31e74d666`) and adds a UR10e as the arm
macro of `clearpath_manipulators_description` does (both read, not copied). Clearpath's controller
configuration for the A300 (`clearpath_control/config/a300/control/diff_4wd.yaml`) was read for a
comparison with the skid model and is not used. The URDF files still name upstream mesh files;
nothing in this package loads them.
Values that the upstream descriptions do not contain are listed under `declared` in the records and
are synthetic engineering values, not manufacturer data. Preserve these notices when redistributing
the files.

RB-ROBOUT+, Ridgeback, Husky, UR10e, UR30 and Ewellix are product names of their owners. The robot
records are not manufacturer models.

## PACDM

The PACDM implementation (M. Dastranj and J. Mattila, arXiv:2609.11338) is included unchanged in
`vendor/pacdm_original.py` (SHA-256 `bbd1fb482e7529d70e05be3c3533d6d1076dada79f6b121e70424d138a9be8de`,
the same file as `Go2_CMG_PACDM/vendor/pacdm_original.py` of the RoboCompiler repository).

## Software dependencies

MuJoCo, Pinocchio, Coal, NumPy, SciPy, Matplotlib, Pillow, imageio, imageio-ffmpeg (whose
wheel contains an FFmpeg binary), pytest, xacro and PyYAML are installed separately under their own
licences.
