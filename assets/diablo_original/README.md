# Diablo Original visualization assets

These files were downloaded from
[`DDTRobot/diablo-sim-env`](https://github.com/DDTRobot/diablo-sim-env/tree/3c9abbcaeb93911932b9c14777c4e75cc8c1470e/diablo_A1/urdfs/diablo_robot/diablo_original)
at commit `3c9abbcaeb93911932b9c14777c4e75cc8c1470e`.

Only the assets needed by the Rerun visualization are included:

- `urdf/diablo_forest_nav.urdf`
- all STL files referenced by that URDF
- the wheel texture referenced by that URDF

The copied URDF differs from the upstream `diablo_mesh.urdf` only by an
identity `body -> base_link` fixed joint. The recorded bags already publish
`camera_init -> body`, so this connects the robot geometry to the dataset's
actual TF tree without changing the mesh coordinate system.

The upstream repository does not contain a top-level license file. Its README
displays an LGPL-2.1 badge linking to the Diablo SDK license; verify the
applicable asset licensing with DDTRobot before redistributing these files.
