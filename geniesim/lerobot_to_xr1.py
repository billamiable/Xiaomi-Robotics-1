#!/usr/bin/env python3
"""Convert Genie Sim LeRobot v2.1 task directories into Xiaomi-Robotics-1 JSON episodes.

    python geniesim/lerobot_to_xr1.py \
        --src /path/to/instruction/pick_block_color_500 [...] \
        --dst /path/to/instruction_xr1

One JSON per episode, written to <dst>/<task>/episode_XXXXXX.json. Videos are
referenced by absolute path, never copied or re-encoded.

Three things this file is careful about, each of which fails silently if wrong:

1. Field positions come from meta/info.json `field_descriptions`, looked up by
   name. The 109-dim release was re-indexed from a fuller field set (see
   meta/.trimmed_to_md_v1), so a hard-coded index from any other release, or
   from the official 159-dim converter, points at the wrong field.

2. XR-1 learns end-effector deltas relative to the current end-effector pose,
   so proprio and action poses must be in the same frame. The dataset has two:
   `state/end/position` is base_link, `state/end/arm_position` is
   arm_base_link, and `action/end/position` is arm_base_link. The state and
   action fields share the name `end/position` but not the frame -- pairing by
   name is off by ~0.49 m. We take proprio from `state/end/arm_*` and assert
   the frame from the description text, and check per episode that the action
   target lands on the proprio pose a few frames later.

3. The instruction is not in tasks.jsonl, which holds only the task name
   ("pick_block_color"). The per-episode sentence lives in
   annotations.json -> action_steps[].action_text; multi-step tasks are joined
   with ", ", which is exactly the string the evaluation sends.

Training frames follow the π0.5 Genie Sim baseline (openpi
`dataloader_sampler="subtask"`, src/openpi/training/sampler.py): a sample may
only start inside a sub-task segment from meta/info.json `instruction_segments`,
which drops the idle head and the tail after the last segment (~19% of frames
in instruction/). The episode itself is written whole and the segments go into
`sample_ranges`, so an action chunk ending past a segment still reads the real
future, as it does in openpi. Normalization statistics, like openpi's, still use
every frame.

Output directories must contain episode JSONs only: XR-1's JsonDataset globs
**/*.json under the training path and treats every hit as an episode. The run
summary goes to stdout, not to a file under <dst>.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

# XR-1's multi-view prompt, verbatim from xr1/docs/data_format.md. The deploy
# side must build the identical string.
PROMPT = (
    "The following observations are captured from multiple views.\n"
    "# Ego View\n<image>\n# Left-Wrist View\n<image>\n# Right-Wrist View\n<image>\n"
    "Generate robot actions for the task:\n{task}"
)
CAMERAS = {  # XR-1 view -> LeRobot video key
    "ego": "observation.images.top_head",
    "wrist_left": "observation.images.hand_left",
    "wrist_right": "observation.images.hand_right",
}

# Fields we read, and the frame each must be in (checked against the dataset's
# own description text, so a release that redefines a field fails loudly).
STATE_FIELDS = {
    "state/joint/position": None,
    "state/left_effector/position": None,
    "state/right_effector/position": None,
    "state/end/arm_position": "arm_base_link",
    "state/end/arm_orientation": "arm_base_link",
}
ACTION_FIELDS = {
    "action/end/position": "arm_base_link",
    "action/end/orientation": "arm_base_link",
    "action/left_effector/position": None,
    "action/right_effector/position": None,
}

# Per-episode sanity thresholds. Measured on this release: action target vs
# proprio two frames later agrees to ~1.1 mm median; the wrong frame pairing
# disagrees by ~490 mm. 1 cm leaves an order of magnitude on both sides.
CHECK_LAG = 2
MAX_TRACK_ERR_M = 0.01
MAX_QUAT_NORM_ERR = 1e-3

# openpi sample_subtask() caps a sub-task whose instruction reads like a reset at
# 45 frames once it runs past 90. The match is a bare substring, so it also
# catches "...returns it to the original position" in straighten_object;
# reproduced as is, since matching the baseline is the point.
RESET_WORDS = ("reset", "return", "default")


def field_indices(info: dict, feature: str, wanted: dict[str, str | None]) -> dict[str, list[int]]:
    fd = info["features"][feature].get("field_descriptions")
    if not fd:
        raise SystemExit(f"{feature} has no field_descriptions; this converter only supports "
                         "releases that document their layout")
    out = {}
    for name, frame in wanted.items():
        if name not in fd:
            raise SystemExit(f"{feature}: field {name!r} missing (have {sorted(fd)})")
        desc = fd[name].get("description", "")
        if frame and frame not in desc:
            raise SystemExit(f"{name}: expected frame {frame!r}, description says: {desc!r}")
        if "orientation" in name and "x/y/z/w" not in desc:
            raise SystemExit(f"{name}: expected an x/y/z/w quaternion, description says: {desc!r}")
        out[name] = fd[name]["indices"]
    return out


def instruction_text(ann: dict) -> str:
    steps = ann.get("action_steps") or []
    texts = [s["action_text"].strip() for s in steps if s.get("action_text", "").strip()]
    if not texts:
        raise ValueError("no action_text in annotations")
    return ", ".join(texts)


def sample_ranges(segments: list[dict]) -> list[list[int]]:
    """Inclusive [start, end] frame ranges, exactly as openpi sample_subtask() builds them."""
    out = []
    for seg in segments:
        start = seg["start_frame_index"]
        end = seg["success_frame_index"] if "success_frame_index" in seg else seg["end_frame_index"]
        if any(w in seg["instruction"].lower() for w in RESET_WORDS) and end - start > 90:
            end = start + 45
        out.append([int(start), int(end)])
    return out


def quat_to_rotm(q: np.ndarray) -> np.ndarray:
    """(T, 4) xyzw -> (T, 9) row-major rotation matrices."""
    return R.from_quat(q).as_matrix().reshape(len(q), 9)


def r6(a: np.ndarray) -> list:
    return np.round(a.astype(np.float64), 6).tolist()


def convert_episode(job: tuple) -> dict:
    (task_dir, task, ep, length, info, s_idx, a_idx, text, ranges, dst) = job
    chunk = ep // info["chunks_size"]
    pq = task_dir / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    df = pd.read_parquet(pq, columns=["observation.state", "action"])
    S = np.stack(df["observation.state"].values).astype(np.float64)
    A = np.stack(df["action"].values).astype(np.float64)
    T = len(S)
    if T != length:
        return {"ep": ep, "error": f"parquet has {T} rows, episodes.jsonl says {length}"}

    joint = S[:, s_idx["state/joint/position"]]
    s_pos = S[:, s_idx["state/end/arm_position"]]
    s_quat = S[:, s_idx["state/end/arm_orientation"]]
    a_pos = A[:, a_idx["action/end/position"]]
    a_quat = A[:, a_idx["action/end/orientation"]]

    # Quaternions: unit norm, both arms.
    for name, q in (("state", s_quat), ("action", a_quat)):
        n = np.linalg.norm(q.reshape(T, 2, 4), axis=-1)
        if np.abs(n - 1).max() > MAX_QUAT_NORM_ERR:
            return {"ep": ep, "error": f"{name} quaternion norm off by {np.abs(n - 1).max():.4f}"}

    # Frame: the action target must land on the proprio pose a few frames on.
    if T > CHECK_LAG + 1:
        err = np.linalg.norm((a_pos[:-CHECK_LAG] - s_pos[CHECK_LAG:]).reshape(-1, 2, 3), axis=-1)
        track = float(np.median(err))
        if track > MAX_TRACK_ERR_M:
            return {"ep": ep, "error": f"action vs proprio EEF median {track * 1000:.1f} mm "
                                       f"(> {MAX_TRACK_ERR_M * 1000:.0f} mm): frame mismatch?"}
    else:
        track = float("nan")

    zeros1 = np.zeros((T, 1))
    vid = {k: str((task_dir / info["video_path"].format(
        episode_chunk=chunk, video_key=v, episode_index=ep)).resolve())
        for k, v in CAMERAS.items()}
    for k, p in vid.items():
        if not os.path.exists(p):
            return {"ep": ep, "error": f"missing video {k}: {p}"}

    out = {
        "trajectory_type": "success",
        "time": f"{task}/episode_{ep:06d}",
        "num_frames": T,
        "sample_ranges": ranges,
        "instruction": {"general": [{
            "images": [f"observations.{k}" for k in CAMERAS],
            "conversations": [
                {"from": "human", "value": PROMPT.format(task=text)},
                {"from": "gpt", "value": ""},
            ],
        }]},
        "observations": {k: [{"path": p, "start": 0, "crop_bbox": None}] for k, p in vid.items()},
        "proprios": {
            "left_arm_joint": r6(joint[:, 0:7]),
            "right_arm_joint": r6(joint[:, 7:14]),
            "left_gripper_pos": r6(S[:, s_idx["state/left_effector/position"]]),
            "right_gripper_pos": r6(S[:, s_idx["state/right_effector/position"]]),
            "left_ee_pos": r6(s_pos[:, 0:3]),
            "left_ee_rotm": r6(quat_to_rotm(s_quat[:, 0:4])),
            "right_ee_pos": r6(s_pos[:, 3:6]),
            "right_ee_rotm": r6(quat_to_rotm(s_quat[:, 4:8])),
            # Waist is static in this suite (action std 0, state jitter ~1e-4).
            # Zero both sides so the learned delta is exactly 0, not noise.
            "waist_pos": r6(zeros1),
        },
        "actions": {
            "left_ee_pos": r6(a_pos[:, 0:3]),
            "left_ee_rotm": r6(quat_to_rotm(a_quat[:, 0:4])),
            "right_ee_pos": r6(a_pos[:, 3:6]),
            "right_ee_rotm": r6(quat_to_rotm(a_quat[:, 4:8])),
            "left_gripper_pos": r6(A[:, a_idx["action/left_effector/position"]]),
            "right_gripper_pos": r6(A[:, a_idx["action/right_effector/position"]]),
            "waist_pos": r6(zeros1),
            "base_vel": r6(np.zeros((T, 3))),
        },
    }
    path = dst / task / f"episode_{ep:06d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    sampled = len({i for s, e in ranges for i in range(max(0, s), min(T - 1, e) + 1)})
    return {"ep": ep, "frames": T, "sampled": sampled, "track_mm": track * 1000}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", nargs="+", required=True, type=Path, help="LeRobot task directories")
    ap.add_argument("--dst", required=True, type=Path, help="output root (episode JSONs only)")
    ap.add_argument("--episodes", type=int, default=None, help="convert only the first N per task")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()

    total_ok, total_bad, total_frames, total_sampled = 0, 0, 0, 0
    for task_dir in args.src:
        task_dir = task_dir.resolve()
        task = task_dir.name
        info = json.load(open(task_dir / "meta/info.json"))
        # Large and per-episode; pull it out so each job ships only its own.
        segments = info.pop("instruction_segments", None)
        if not segments:
            raise SystemExit(f"{task}: meta/info.json has no instruction_segments")
        if info.get("fps") != 30:
            raise SystemExit(f"{task}: fps {info.get('fps')} (expected 30)")
        s_idx = field_indices(info, "observation.state", STATE_FIELDS)
        a_idx = field_indices(info, "action", ACTION_FIELDS)
        anns = json.load(open(task_dir / "meta/annotations.json"))
        by_ep = {int(v["episode_index"]): v for v in anns.values()}
        eps = [json.loads(l) for l in open(task_dir / "meta/episodes.jsonl")]
        if args.episodes:
            eps = eps[: args.episodes]
        # openpi keys segments by position in the episode list; that equals
        # episode_index only when indices run 0..N-1.
        if [int(e["episode_index"]) for e in eps] != list(range(len(eps))):
            raise SystemExit(f"{task}: episode indices are not 0..N-1")

        jobs = []
        for e in eps:
            i = int(e["episode_index"])
            if i not in by_ep:
                raise SystemExit(f"{task}: episode {i} has no annotation")
            # openpi skips an episode with no segments; an empty range list
            # makes JsonDataset do the same.
            jobs.append((task_dir, task, i, int(e["length"]), info, s_idx, a_idx,
                         instruction_text(by_ep[i]), sample_ranges(segments.get(str(i), [])), args.dst))

        with ProcessPoolExecutor(args.workers) as pool:
            results = list(pool.map(convert_episode, jobs, chunksize=8))
        bad = [r for r in results if "error" in r]
        ok = [r for r in results if "error" not in r]
        total_ok += len(ok)
        total_bad += len(bad)
        tr = np.array([r["track_mm"] for r in ok if r["track_mm"] == r["track_mm"]])
        frames, sampled = sum(r["frames"] for r in ok), sum(r["sampled"] for r in ok)
        total_frames += frames
        total_sampled += sampled
        print(f"{task:<28} ok {len(ok):>4}  bad {len(bad):>3}  frames {frames:>7}  "
              f"sampled {sampled:>7} ({sampled / max(frames, 1):.1%})  "
              f"track median {np.median(tr):.2f} mm  p99 {np.percentile(tr, 99):.2f} mm", flush=True)
        for r in bad[:5]:
            print(f"    ep {r['ep']}: {r['error']}")

    print(f"total ok {total_ok}  bad {total_bad}  frames {total_frames}  "
          f"sampled {total_sampled} ({total_sampled / max(total_frames, 1):.1%})")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
