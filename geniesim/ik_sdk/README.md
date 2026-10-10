# G2 IK-SDK files

`G2_NO_GRIPPER.urdf` and `g2_solver.yaml` are copied unmodified from genie_sim
(`source/geniesim_benchmark/src/geniesim_benchmark/utils/IK-SDK/`, commit 6ca11c7),
which is under the Mozilla Public License 2.0. They are the kinematic model the
simulator's own IK/FK uses, so `g2_ik.py` solves on exactly the chain that produced
the dataset's EEF fields and the payload's `end_pose`.

Kept here so an eval machine without a genie_sim checkout gets them with the
code. The mesh references in the URDF are not needed: only the kinematic chain is
loaded.

The `ik_solver` Python package these files go with is a compiled cp311 wheel that
ships inside the geniesim container (`/tmp/ik_solver-0.4.3-cp311-cp311-linux_x86_64.whl`);
it is not vendored. `geniesim/setup_env.sh` installs it when `IK_SOLVER_WHEEL` points at it.
