#!/usr/bin/env python3
"""Measure the simulator's EEF_ABS -> joint IK on real dataset targets.

XR-1 sends EEF targets and the sim turns them into joint targets with its own
IK (`IKFKSolver.eef_actions_to_joint`: 3 iterations per step, seeded with the
measured joints). That step can't be checked from the policy server, and a
closed-loop error mixes IK with controller lag and with the policy's own
quality. This isolates the IK:

    # 1. on the host (XR-1 venv): sample targets from the dataset
    python geniesim/ik_check.py extract --data <instruction suite> --out ik_samples.npz

    # 2. inside the Genie Sim container: run the sim's solver on them
    docker exec geniesim3 bash -lc 'cd /workspace && python <path>/ik_check.py run --samples ik_samples.npz'

For each sampled frame t and step k it calls the solver exactly as
`corobotpolicy._post_process_action` does (target pose as xyz + euler "xyz",
seeded with the measured joints at t + k), then compares:

* FK(IK joints) with the target pose -- IK error proper
* IK joints with the dataset's joint target -- since in this data the EEF
  target is FK of the joint target, this should be near zero too

A JOINT_ABS sender has neither error, so anything here is the price of the
EEF_ABS path.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

H = 30


def extract(args) -> None:
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lerobot_to_xr1 import ACTION_FIELDS, STATE_FIELDS, field_indices, sample_ranges

    rng = random.Random(args.seed)
    seed_joints, targets, joint_targets = [], [], []
    for task_dir in sorted(p for p in args.data.iterdir() if (p / "meta/info.json").is_file()):
        info = json.load(open(task_dir / "meta/info.json"))
        s_idx = field_indices(info, "observation.state", STATE_FIELDS)
        a_idx = field_indices(info, "action", {**ACTION_FIELDS, "action/joint/position": None})
        lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in open(task_dir / "meta/episodes.jsonl")}
        for ep in rng.sample(sorted(lengths), args.per_task):
            T = lengths[ep]
            frames = [i for s, e in sample_ranges(info["instruction_segments"][str(ep)])
                      for i in range(s, min(e, T - 1) + 1)]
            t = rng.choice(frames)
            df = pd.read_parquet(task_dir / info["data_path"].format(episode_chunk=ep // info["chunks_size"],
                                                                     episode_index=ep))
            S = np.stack(df["observation.state"].values)
            A = np.stack(df["action"].values)
            idx = np.minimum(t + np.arange(H), T - 1)
            seed_joints.append(S[idx][:, s_idx["state/joint/position"]])
            pos, quat = A[idx][:, a_idx["action/end/position"]], A[idx][:, a_idx["action/end/orientation"]]
            targets.append(np.concatenate([pos[:, 0:3], quat[:, 0:4], pos[:, 3:6], quat[:, 4:8]], axis=1))
            joint_targets.append(A[idx][:, a_idx["action/joint/position"]])
    np.savez(args.out, seed_joints=np.stack(seed_joints), targets=np.stack(targets),
             joint_targets=np.stack(joint_targets))
    print(f"wrote {args.out}: {len(targets)} samples x {H} steps")


def run(args) -> None:
    from geniesim_benchmark.utils.ikfk_utils import IKFKSolver

    d = np.load(args.samples)
    solver = IKFKSolver([0.0] * 14, [0.0] * 3, [0.0] * 5, robot_cfg="G2_omnipicker")
    print(f"{len(d['targets'])} samples x {d['targets'].shape[1]} steps x 2 arms")
    for iters in args.iters:
        report(solver, d, iters)


def report(solver, d, iters: int) -> None:
    pos_err, rot_err, joint_err = [], [], []
    for seeds, tgts, jts in zip(d["seed_joints"], d["targets"], d["joint_targets"]):
        for seed, tgt, jt in zip(seeds, tgts, jts):
            left, right = tgt[0:7], tgt[7:14]
            # Same packing as corobotpolicy._post_process_action for eef_frame=arm_base_link.
            eef = np.concatenate([left[:3], R.from_quat(left[3:7]).as_euler("xyz"),
                                  right[:3], R.from_quat(right[3:7]).as_euler("xyz"), [0.0, 0.0]])
            joints = np.asarray(solver.eef_actions_to_joint([eef.tolist()], list(seed), [0.0, 0.0],
                                                            max_ik_iters=iters)[0][:14])
            fk = solver.compute_eef(joints.tolist())
            for side, want in (("left", left), ("right", right)):
                got = np.asarray(fk[side], dtype=np.float64)          # [x, y, z, qw, qx, qy, qz]
                pos_err.append(np.linalg.norm(got[:3] - want[:3]) * 1000)
                got_q = got[[4, 5, 6, 3]]                               # wxyz -> xyzw
                rot_err.append(np.degrees((R.from_quat(got_q).inv() * R.from_quat(want[3:7])).magnitude()))
            joint_err.append(np.abs(joints - jt).max())
    p, r, j = np.asarray(pos_err), np.asarray(rot_err), np.degrees(np.asarray(joint_err))
    print(f"max_ik_iters={iters}" + ("  (what the sim uses)" if iters == 3 else ""))
    for name, v, unit in (("FK(IK) vs target, pos", p, "mm"), ("FK(IK) vs target, rot", r, "deg"),
                          ("IK joints vs dataset joint target (max over 14)", j, "deg")):
        print(f"  {name:<48} median {np.median(v):8.3f}  p90 {np.percentile(v, 90):8.3f}  "
              f"p99 {np.percentile(v, 99):8.3f}  max {v.max():8.3f} {unit}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--per-task", type=int, default=10)
    e.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--samples", type=Path, required=True)
    r.add_argument("--iters", type=int, nargs="+", default=[3], help="IK iterations per step (sim: 3)")
    args = ap.parse_args()
    extract(args) if args.cmd == "extract" else run(args)


if __name__ == "__main__":
    main()
