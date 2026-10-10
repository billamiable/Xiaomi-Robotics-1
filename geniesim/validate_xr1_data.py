#!/usr/bin/env python3
"""Load converted episodes through XR-1's own JsonDataset and check them end to end.

    cd xr1 && ../.venv/bin/python ../geniesim/validate_xr1_data.py \
        --data /tmp/xr1_smoke --norm /tmp/xr1_smoke_norm.json

Three checks, each aimed at a way the conversion could be wrong without
anything raising:

1. Sample shape -- three images per sample, action (30, 60), state (1, 60),
   prompt carrying the episode's instruction.

2. Round trip -- take the normalized action the dataset produces, undo the
   normalization, rebuild absolute end-effector targets with XR-1's own
   recover_action() from the proprio pose at that frame, and compare to the
   targets written by the converter. This is the inverse of what the deploy
   side will do, so a match means training and inference agree on frames,
   quaternion order and the local-frame convention. Expect float32 noise only.

3. Loader throughput -- every sample opens and seeks three native-resolution
   HEVC videos, which is the likeliest training bottleneck.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

import numpy as np

from mibot.data.datasets.json_dataset import JsonDataset
from mibot.utils.io import denormalize_action, recover_action

ARM_KEYS = ("ee_pos", "ee_rotm", "gripper_pos")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--norm", required=True)
    ap.add_argument("--samples", type=int, default=200, help="round-trip samples")
    ap.add_argument("--timed", type=int, default=40, help="samples to time for throughput")
    args = ap.parse_args()

    norm = json.load(open(args.norm))
    # JsonDataset sizes itself to max_steps * batch * world, so ask for exactly
    # one pass over every sampled frame -- otherwise it only materializes
    # max_steps samples and the round trip below draws from a handful of frames.
    frames = 0
    for f in JsonDataset._json_files([args.data]):
        traj = json.load(open(f))
        frames += len(JsonDataset._sample_frames(traj, traj["num_frames"]))
    params = {"max_steps": frames, "train_datasets": {
        "action_length": 30, "batch_size": 1, "paths": [args.data], **norm}}
    ds = JsonDataset(params)
    mean, std = np.asarray(norm["mean"]), np.asarray(norm["std"])
    print(f"dataset: {len(ds.files)} episodes, {len(ds.samples)} sampled frames (π0.5 sub-task ranges)")

    # 1. shape
    s = ds[0]
    imgs = [c["image"] for m in s["messages"] for c in m["content"]
            if isinstance(c, dict) and c.get("type") == "image"]
    text = "".join(c.get("text", "") for m in s["messages"] for c in m["content"]
                   if isinstance(c, dict) and c.get("type") == "text")
    print(f"[1] images {len(imgs)}: {[im.size for im in imgs]}  "
          f"action {tuple(s['action'].shape)}  state {tuple(s['state'].shape)}")
    print(f"    prompt tail: ...{text[text.find('task:') - 1:][:110]!r}")
    assert len(imgs) == 3 and tuple(s["action"].shape) == (30, 60)

    # 2. round trip
    rng = random.Random(0)
    worst = {"pos": 0.0, "rot": 0.0, "grip": 0.0}
    for i in rng.sample(range(len(ds.source_samples) if hasattr(ds, "source_samples")
                              else len(ds.samples)), min(args.samples, len(ds.samples))):
        smp = ds.samples[i]
        traj = JsonDataset._read_json(smp["file"])
        t, T = smp["frame_index"], traj["num_frames"]
        steps = min(30, T - t)
        act = denormalize_action(ds[i]["action"].numpy(), mean, std)[:steps]
        # recover_action handles both arms in one call, so it needs both.
        robot = {f"{side}_{k}": np.asarray(traj["proprios"][f"{side}_{k}"][t])
                 for side in ("left", "right") for k in ARM_KEYS}
        rec = recover_action(act, robot)
        for side in ("left", "right"):
            want_p = np.asarray(traj["actions"][f"{side}_ee_pos"][t:t + steps])
            want_r = np.asarray(traj["actions"][f"{side}_ee_rotm"][t:t + steps]).reshape(-1, 3, 3)
            want_g = np.asarray(traj["actions"][f"{side}_gripper_pos"][t:t + steps])
            worst["pos"] = max(worst["pos"], np.abs(rec[f"{side}_ee_pos"] - want_p).max())
            worst["rot"] = max(worst["rot"], np.abs(rec[f"{side}_ee_rotm"] - want_r).max())
            worst["grip"] = max(worst["grip"], np.abs(rec[f"{side}_gripper_pos"] - want_g).max())
    n = min(args.samples, len(ds.samples))
    print(f"[2] round trip over {n} samples x 2 arms, worst abs error: "
          f"pos {worst['pos'] * 1000:.4f} mm  rotm {worst['rot']:.2e}  gripper {worst['grip']:.2e}")
    # Rotation tolerance is set by XR-1, not by the conversion: rotm2aa_batch
    # casts to float32 and takes arccos((trace - 1) / 2), whose resolution near
    # zero is sqrt(2 * eps32) ~ 4.9e-4 rad. Anything at that floor is XR-1's own
    # quantization, present in its pretraining too; a frame or quaternion-order
    # bug shows up as errors of order 1e-1 or larger.
    ok = worst["pos"] < 1e-4 and worst["rot"] < 2e-3 and worst["grip"] < 1e-4

    # 3. throughput
    idx = [rng.randrange(len(ds.samples)) for _ in range(args.timed)]
    t0 = time.time()
    for i in idx:
        ds[i]
    dt = (time.time() - t0) / args.timed
    print(f"[3] loader: {dt * 1000:.0f} ms / sample (single process, 3 native-res videos)")

    print("PASS" if ok else "FAIL: round trip exceeds float32 noise")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
