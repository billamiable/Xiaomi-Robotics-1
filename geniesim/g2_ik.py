"""Server-side IK for the G2 arms, so an EEF policy can send JOINT_ABS.

Why not let the simulator do it (`EEF_ABS`): its `eef_actions_to_joint` runs
the IK-SDK's `solve()` 3 times per step, seeded with the measured joints, and
`solve()` takes velocity-limited steps. On 6,000 dataset targets
(geniesim/ik_check.py) the solved pose misses the target by a median 1.4 mm
but p90 15.7 mm / p99 25.8 mm -- about 10% of steps land 1.5 cm off. A
JOINT_ABS sender has no such error.

Here the same IK-SDK provides exact FK (the function the dataset's EEF fields
and the sim's `end_pose` both come from), and Levenberg-Marquardt with a
numeric Jacobian solves each step. Steps are chained (step k is seeded with
step k-1's solution), with a retry from the chunk's measured joints when a
step doesn't converge. Same 6,000 targets: median 0.013 mm, p99 0.26 mm,
0.1% of steps still ~14 mm (near joint limits); ~37 ms per 30-step chunk for
both arms.

Needs the `ik_solver` wheel (in the geniesim container at
/tmp/ik_solver-*-cp311-*.whl) and the IK-SDK files G2_NO_GRIPPER.urdf +
g2_solver.yaml. A copy of the latter ships in geniesim/ik_sdk/ and is the
default; --ik-sdk or $GENIESIM_IK_SDK point elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Joint limits as the IK-SDK prints them for both G2 arms.
LOWER = np.array([-3.1067, -2.0944, -3.1067, -2.5307, -3.1067, -1.0472, -1.5708])
UPPER = np.array([3.1067, 2.0944, 3.1067, 1.0472, 3.1067, 1.0472, 1.5708])
# Residual weights: 1 m of position against 0.2 m per radian of rotation
# (1 mm ~ 0.29 deg).
WEIGHT = np.array([1.0, 1.0, 1.0, 0.2, 0.2, 0.2])
POS_TOL, ROT_TOL = 5e-5, 2e-4          # converged: 0.05 mm and ~0.01 deg
RETRY_POS, RETRY_ROT = 1e-3, 2e-3      # retry from the chunk seed above 1 mm / ~0.1 deg


class G2ArmIK:
    def __init__(self, sdk_dir: str | os.PathLike | None = None) -> None:
        import ik_solver

        sdk = Path(sdk_dir or os.environ.get("GENIESIM_IK_SDK") or Path(__file__).resolve().parent / "ik_sdk")
        urdf, cfg = sdk / "G2_NO_GRIPPER.urdf", sdk / "g2_solver.yaml"
        if not urdf.is_file() or not cfg.is_file():
            raise FileNotFoundError(f"IK-SDK files not found in {sdk!s}; pass --ik-sdk or set GENIESIM_IK_SDK")
        self.solvers = {}
        for side, part in (("left", ik_solver.RobotPart.LEFT_ARM), ("right", ik_solver.RobotPart.RIGHT_ARM)):
            solver = ik_solver.Solver(part=part, urdf_path=str(urdf), config_path=str(cfg))
            solver.set_debug_mode(False)
            self.solvers[side] = solver

    def fk(self, side: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = np.asarray(self.solvers[side].compute_fk(np.asarray(q, dtype=np.float32)), dtype=np.float64).reshape(4, 4)
        return m[:3, 3], m[:3, :3]

    def _residual(self, side: str, q: np.ndarray, pos: np.ndarray, rotm: np.ndarray) -> np.ndarray:
        p, r = self.fk(side, q)
        return np.concatenate([pos - p, R.from_matrix(rotm @ r.T).as_rotvec()]) * WEIGHT

    def _solve_step(self, side: str, q: np.ndarray, pos: np.ndarray, rotm: np.ndarray,
                    iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
        lam = 1e-3
        e = self._residual(side, q, pos, rotm)
        for _ in range(iters):
            if np.linalg.norm(e[:3]) < POS_TOL and np.linalg.norm(e[3:]) < ROT_TOL:
                break
            jac = np.empty((6, 7))
            for j in range(7):
                dq = q.copy()
                dq[j] += 1e-5
                jac[:, j] = (e - self._residual(side, dq, pos, rotm)) / 1e-5      # d(pose)/dq
            step = np.linalg.solve(jac.T @ jac + lam * np.eye(7), jac.T @ e)
            q_new = np.clip(q + step, LOWER, UPPER)
            e_new = self._residual(side, q_new, pos, rotm)
            if e_new @ e_new < e @ e:
                q, e, lam = q_new, e_new, max(lam / 3, 1e-7)
            else:
                lam *= 5
        return q, e

    def solve_chunk(self, side: str, q_measured: np.ndarray, pos: np.ndarray,
                    rotm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Joint targets (T, 7) for EEF targets pos (T, 3), rotm (T, 3, 3), and
        the per-step position error in metres (T,)."""
        q0 = np.asarray(q_measured, dtype=np.float64)
        q = q0.copy()
        out, err = np.empty((len(pos), 7)), np.empty(len(pos))
        for k in range(len(pos)):
            qa, ea = self._solve_step(side, q, pos[k], rotm[k])
            if np.linalg.norm(ea[:3]) > RETRY_POS or np.linalg.norm(ea[3:]) > RETRY_ROT:
                qb, eb = self._solve_step(side, q0, pos[k], rotm[k])
                if eb @ eb < ea @ ea:
                    qa, ea = qb, eb
            q = qa
            out[k], err[k] = q, np.linalg.norm(ea[:3])
        return out, err
