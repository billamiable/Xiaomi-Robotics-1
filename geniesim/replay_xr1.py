#!/usr/bin/env python3
"""Open-loop replay: feed dataset frames to the eval backend and score its actions.

    python geniesim/replay_xr1.py --ckpt <ckpt> \
        --data /path/to/instruction [--per-task 20] [--tasks pick_block_color_500 ...]

Each request is built the way the simulator builds one (corobotpolicy.py):
the three camera frames JPEG-encoded at quality 95, `arm_joint_states` /
`gripper_states`, `end_pose.arm_base_link` from the dataset's
`state/end/arm_*` (the same FK the sim runs), and the episode's instruction as
the prompt. The response goes through the real `XR1Policy.infer`, so this
tests the eval path end to end -- decoding, resize, prompt, state
normalization, de-normalization, delta recovery, quaternion packing -- against
the ground-truth action targets, without a simulator.

Frames are drawn from the training sample ranges (sub-task segments), so the
numbers are comparable with what the model was trained on. `--dump-payload` writes the first request as a corobot `.pkl` that
`geniesim benchmark check-inference <pkl> --host ... --port ...` accepts. Its
bundled payload is a G1 request without `end_pose`, which this backend
rejects by design.

A "hold" baseline
(target = current pose for every step) is printed alongside: a working policy
should beat it by a wide margin at later steps.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lerobot_to_xr1 import ACTION_FIELDS, STATE_FIELDS, field_indices, instruction_text, sample_ranges  # noqa: E402
from xr1_policy import XR1Policy  # noqa: E402

CAMERA_KEYS = {"head": "observation.images.top_head",
               "hand_left": "observation.images.hand_left",
               "hand_right": "observation.images.hand_right"}


def encode_jpeg(rgb: np.ndarray) -> dict:
    import cv2

    _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"encoding": "JPEG", "image_data": buf.tobytes(), "height": rgb.shape[0], "width": rgb.shape[1]}


def rot_err_deg(q_pred: np.ndarray, q_true: np.ndarray) -> np.ndarray:
    return np.degrees((R.from_quat(q_pred).inv() * R.from_quat(q_true)).magnitude())


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data", required=True, type=Path, help="LeRobot suite root (task dirs inside)")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--per-task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump-payload", type=Path, default=None,
                    help="save the first request as a check-inference .pkl")
    args = ap.parse_args()

    from decord import VideoReader

    # EEF output: the replay scores the predicted poses. IK accuracy is measured
    # separately (ik_check.py, and the server IK residual in xr1_policy logs).
    policy = XR1Policy(args.ckpt, config=args.config, execute_steps=30, arm_output="eef")
    rng = random.Random(args.seed)
    task_dirs = sorted(p for p in args.data.iterdir() if (p / "meta/info.json").is_file())
    if args.tasks:
        task_dirs = [p for p in task_dirs if p.name in args.tasks]

    steps = np.arange(30)
    rows = []
    for task_dir in task_dirs:
        info = json.load(open(task_dir / "meta/info.json"))
        s_idx = field_indices(info, "observation.state", STATE_FIELDS)
        a_idx = field_indices(info, "action", ACTION_FIELDS)
        segments = info["instruction_segments"]
        anns = {int(v["episode_index"]): v for v in json.load(open(task_dir / "meta/annotations.json")).values()}
        lengths = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in open(task_dir / "meta/episodes.jsonl")}
        eps = rng.sample(sorted(lengths), min(args.per_task, len(lengths)))
        for ep in eps:
            T = lengths[ep]
            frames = sorted({i for s, e in sample_ranges(segments[str(ep)]) for i in range(s, min(e, T - 1) + 1)})
            t = rng.choice(frames)
            chunk = ep // info["chunks_size"]
            df = pd.read_parquet(task_dir / info["data_path"].format(episode_chunk=chunk, episode_index=ep),
                                 columns=["observation.state", "action"])
            S = np.stack(df["observation.state"].values).astype(np.float64)
            A = np.stack(df["action"].values).astype(np.float64)
            images = {}
            for bench_key, video_key in CAMERA_KEYS.items():
                vr = VideoReader(str(task_dir / info["video_path"].format(
                    episode_chunk=chunk, video_key=video_key, episode_index=ep)), num_threads=2)
                images[bench_key] = encode_jpeg(vr.get_batch([t]).asnumpy()[0])
            pos = S[t, s_idx["state/end/arm_position"]]
            quat = S[t, s_idx["state/end/arm_orientation"]]
            params = {
                "images": images,
                "states": {
                    "arm_joint_states": S[t, s_idx["state/joint/position"]].tolist(),
                    "gripper_states": [S[t, s_idx["state/left_effector/position"][0]],
                                       S[t, s_idx["state/right_effector/position"][0]]],
                    "end_pose": {"arm_base_link": {
                        "left_arm": {"position": pos[0:3].tolist(), "orientation": quat[0:4].tolist()},
                        "right_arm": {"position": pos[3:6].tolist(), "orientation": quat[4:8].tolist()},
                    }},
                },
                "prompt": instruction_text(anns[ep]),
                "task_name": task_dir.name,
            }
            if args.dump_payload and not args.dump_payload.exists():
                import pickle

                with open(args.dump_payload, "wb") as f:
                    pickle.dump({"payload": {"method": "infer", "params": {
                        **params, "robot_type": "G2_omnipicker", "episode_idx": 0, "episode_done": False,
                    }}, "model_arch": "corobot"}, f)
                print(f"wrote {args.dump_payload}")
            t0 = time.monotonic()
            res = policy.infer(params)
            dt = time.monotonic() - t0

            idx = np.minimum(t + steps, T - 1)       # LeRobot-style hold past the end
            a_pos, a_quat = A[idx][:, a_idx["action/end/position"]], A[idx][:, a_idx["action/end/orientation"]]
            for side, sl_p, sl_q, g_key in (("left", slice(0, 3), slice(0, 4), "action/left_effector/position"),
                                            ("right", slice(3, 6), slice(4, 8), "action/right_effector/position")):
                pred = np.asarray(res[f"{side}_arm"]["values"])
                true_p, true_q = a_pos[:, sl_p], a_quat[:, sl_q]
                cur_p = pos[sl_p]
                cur_q = quat[sl_q]
                rows.append({
                    "task": task_dir.name, "side": side, "latency": dt,
                    "pos_mm": np.linalg.norm(pred[:, :3] - true_p, axis=1) * 1000,
                    "rot_deg": rot_err_deg(pred[:, 3:7], true_q),
                    "grip": np.abs(np.asarray(res[f"{side}_effector"])[:, 0] - A[idx][:, a_idx[g_key][0]]),
                    "hold_pos_mm": np.linalg.norm(cur_p[None] - true_p, axis=1) * 1000,
                    "hold_rot_deg": rot_err_deg(np.repeat(cur_q[None], 30, 0), true_q),
                    "moving": float(np.linalg.norm(true_p[-1] - cur_p)) > 0.02,
                })
        print(f"{task_dir.name:<28} done ({len(eps)} samples)", flush=True)

    def table(sel, label):
        if not sel:
            return
        P = np.stack([r["pos_mm"] for r in sel]); Q = np.stack([r["rot_deg"] for r in sel])
        G = np.stack([r["grip"] for r in sel])
        HP = np.stack([r["hold_pos_mm"] for r in sel]); HQ = np.stack([r["hold_rot_deg"] for r in sel])
        print(f"\n{label}  ({len(sel)} arm-samples)   median over samples")
        print(f"  {'step':>4} {'pos mm':>8} {'hold mm':>8} {'rot deg':>8} {'hold deg':>8} {'grip':>7}")
        for k in (0, 4, 9, 14, 19, 29):
            print(f"  {k + 1:>4} {np.median(P[:, k]):>8.1f} {np.median(HP[:, k]):>8.1f} "
                  f"{np.median(Q[:, k]):>8.2f} {np.median(HQ[:, k]):>8.2f} {np.median(G[:, k]):>7.3f}")

    table(rows, "all")
    table([r for r in rows if r["moving"]], "arm moving > 2 cm over the chunk")
    lat = np.array([r["latency"] for r in rows[::2]])
    print(f"\nlatency per request (incl. decode/preprocess): median {np.median(lat) * 1000:.0f} ms, "
          f"p90 {np.percentile(lat, 90) * 1000:.0f} ms; model only avg "
          f"{policy.t_model / max(policy.n_infer, 1) * 1000:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
