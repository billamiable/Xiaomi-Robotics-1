#!/usr/bin/env python3
"""Serve a post-trained XR-1 checkpoint to the Genie Sim benchmark.

    # local benchmark: a websocket server the simulator connects to
    python geniesim/xr1_policy.py --ckpt <ckpt> [--port 8999] [--execute-steps 30]

    # remote challenge: any agent that loads a backend file and calls
    # build_backend(model_path, argv) -> object with load(), __call__(session_id, payload), pack
    build_backend(<ckpt>, ["--execute-steps", "30"])

<ckpt> is any of: a Lightning `*.ckpt` directory, its
`checkpoint/mp_rank_00_model_states.pt`, or a bare `model_states.pt`. The
run's `config.py` (model params plus the inlined normalization stats) is found
by walking up from there, or given with --config.

Request and response follow `corobotpolicy.py` in genie_sim:

  request   {"method": "infer", "params": {
               "images": {"head"|"hand_left"|"hand_right": {"encoding": "JPEG", "image_data": ...}},
               "states": {"arm_joint_states": [14], "gripper_states": [2],
                          "end_pose": {"arm_base_link": {"left_arm"|"right_arm":
                                       {"position": [3], "orientation": [x, y, z, w]}}}, ...},
               "prompt": str, "task_name": str, ...}}
  response  {"result": {"left_arm":  {"kind": "JOINT_ABS", "values": [[7]] * k},
                        "right_arm": {...}, "left_effector": [[g]] * k, "right_effector": [[g]] * k}}
            (or, with --arm-output eef, {"kind": "EEF_ABS", "values": [[x,y,z,qx,qy,qz,qw]] * k,
             "base_link": "arm_base_link"})

Everything the model sees is built exactly as in training (see
geniesim/lerobot_to_xr1.py and xr1/mibot/data/datasets/json_dataset.py):

* images: native frames through the same `resize_image` (keep aspect, no
  letterbox), in the order ego, left wrist, right wrist
* prompt: XR-1's multi-view template + " /no_cot", assistant turn "<cot></cot>"
* state: 14 joints + 2 grippers in the fixed 60-dim slots, quantile-normalized
  and clipped to [-1, 1]
* actions: deltas in the current EE frame, turned back into absolute targets
  around `states.end_pose.arm_base_link` -- the same FK the dataset's
  `state/end/arm_*` came from

Arm output. XR-1 predicts EEF poses; the sim accepts joints or poses.

* `--arm-output joint` (default): solve IK here (geniesim/g2_ik.py, exact FK
  from the sim's own IK-SDK) and send JOINT_ABS. Median 0.013 mm / p99
  0.26 mm on dataset targets.
* `--arm-output eef`: send EEF_ABS and let the sim solve. Its 3-iteration,
  velocity-limited IK lands p90 15.7 mm / p99 25.8 mm off on the same targets
  (geniesim/ik_check.py). EEF_ABS defaults to base_link, so we declare
  arm_base_link, the frame the model was trained in.

With no `end_pose` in the payload there is nothing to add the deltas to. We
fail the request rather than invent a pose.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

XR1_DIR = Path(__file__).resolve().parents[1] / "xr1"
sys.path.insert(0, str(XR1_DIR))
# g2_ik.py sits next to this file; an agent may load us by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger("xr1-geniesim")

# Benchmark camera key -> position in the training prompt (ego, left wrist, right wrist).
CAMERAS = ("head", "hand_left", "hand_right")
ARM_DIM = 7

# XR-1's prompt, verbatim from xr1/mibot/server/runtime/client.py. The training
# side (geniesim/lerobot_to_xr1.py PROMPT + JsonDataset._prompt) produces the
# identical token sequence.
TEXT_BEFORE = (
    "The following observations are captured from multiple views.\n# Ego View\n",
    "\n# Left-Wrist View\n",
    "\n# Right-Wrist View\n",
)
TEXT_AFTER = "\nGenerate robot actions for the task:\n{instruction} /no_cot"

# Gripper targets in dataset units: open = -0.785, closed toward 0 (state and
# action alike, every episode starts at -0.785). Recovered targets are state +
# predicted delta, so a small overshoot past either end is possible; clamp to
# the range the robot was ever commanded in.
GRIPPER_RANGE = (-0.785, 0.0)

TRACK_LAG = 2


# ---------------------------------------------------------------------------
# wire format
# ---------------------------------------------------------------------------


def make_pack_unpack():
    """msgpack with numpy support, matching the benchmark client.

    String values may arrive as bytes, keys are decoded, and genie_sim's own
    `__ndarray__` objects are understood alongside msgpack_numpy's.
    """
    import msgpack
    import msgpack_numpy as msgpack_np

    def _decode(value: Any) -> Any:
        if isinstance(value, dict) and b"__ndarray__" in value:
            return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
        if isinstance(value, dict) and b"__npgeneric__" in value:
            return np.dtype(value[b"dtype"]).type(value[b"data"])
        return msgpack_np.decode(value)

    def _str_keys(value: Any) -> Any:
        if isinstance(value, dict):
            return {(k.decode("utf-8", "replace") if isinstance(k, (bytes, bytearray)) else k): _str_keys(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [_str_keys(v) for v in value]
        return value

    def pack(value: Any) -> bytes:
        return msgpack.packb(value, default=msgpack_np.encode)

    def unpack(data: bytes) -> Any:
        return _str_keys(msgpack.unpackb(data, raw=True, object_hook=_decode, strict_map_key=False))

    return pack, unpack


def _text(value: Any) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def decode_image(entry: Any) -> Image.Image:
    """RGB PIL image from one benchmark image entry (JPEG bytes, or an array)."""
    if isinstance(entry, np.ndarray):
        return Image.fromarray(entry)
    if not isinstance(entry, dict) or entry.get("image_data") is None:
        raise ValueError(f"unsupported image entry: {type(entry)}")
    encoding = _text(entry.get("encoding", "JPEG")).upper()
    if encoding not in ("JPEG", "JPG"):
        raise ValueError(f"unsupported image encoding: {encoding}")
    import cv2

    bgr = cv2.imdecode(np.frombuffer(entry["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        for candidate in (path / "checkpoint" / "mp_rank_00_model_states.pt", path / "mp_rank_00_model_states.pt"):
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"no mp_rank_00_model_states.pt under {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def find_config(ckpt_file: Path) -> Path:
    for parent in ckpt_file.parents:
        if (parent / "config.py").is_file():
            return parent / "config.py"
    raise FileNotFoundError(f"no config.py above {ckpt_file}; pass --config")


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


class XR1Policy:
    def __init__(self, ckpt: str, config: str | None = None, device: str = "cuda:0",
                 execute_steps: int = 30, seed: int | None = 42, arm_output: str = "joint",
                 ik_sdk: str | None = None) -> None:
        import torch
        from mmengine import Config
        from transformers import AutoProcessor

        from mibot.models import MIMODEL
        from mibot.utils.io import build_action_mask, validate_quantiles, validate_stats

        self.torch = torch
        self.device = device
        ckpt_file = resolve_checkpoint(ckpt)
        config_file = Path(config) if config else find_config(ckpt_file)
        cfg = Config.fromfile(str(config_file))
        data = cfg.data.params.train_datasets
        self.action_length = int(data.get("action_length", 30))
        if not 1 <= execute_steps <= self.action_length:
            raise ValueError(f"--execute-steps must be in [1, {self.action_length}], got {execute_steps}")
        self.execute_steps = execute_steps
        if arm_output not in ("joint", "eef"):
            raise ValueError(f"--arm-output must be joint or eef, got {arm_output}")
        self.arm_output = arm_output
        self.ik = None
        if arm_output == "joint":
            from g2_ik import G2ArmIK

            self.ik = G2ArmIK(ik_sdk)
        self._ik_err: list[float] = []

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t0 = time.monotonic()
        model = MIMODEL.build(cfg.model.params.model).to(torch.bfloat16)
        state = torch.load(ckpt_file, map_location="cpu", mmap=True, weights_only=False)["module"]
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(device)
        logger.info("loaded %s (config %s) in %.1fs", ckpt_file, config_file, time.monotonic() - t0)

        mean, std = validate_stats(data.mean, data.std, self.action_length)
        q01, q99 = validate_quantiles(data.q01, data.q99)
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)
        self.q01 = torch.tensor(q01, device=device)
        self.q99 = torch.tensor(q99, device=device)
        self.action_mask = torch.from_numpy(build_action_mask(self.action_length)).to(device)

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        self.processor.tokenizer.padding_side = "right"

        self.n_infer = 0
        self.t_model = 0.0
        self._first_logged = False
        self._seen_prompts: dict[str, str] = {}
        # IK tracking: targets of the previous chunk vs where the arm actually
        # is when the next request arrives. Two references: the last executed
        # step, and the step TRACK_LAG before it -- in the dataset the measured
        # EEF reaches a target about 2 frames after it is commanded (~1 mm),
        # so the lagged error isolates IK/controller accuracy from that lag.
        self._last_target: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] | None = None
        self._episode_key: tuple | None = None
        self._track: dict[int, tuple[list[float], list[float]]] = {0: ([], []), TRACK_LAG: ([], [])}

    # -- inputs ---------------------------------------------------------------

    def _messages(self, instruction: str, images: list[Image.Image]) -> list[dict]:
        content = []
        for text, image in zip(TEXT_BEFORE, images):
            content += [{"type": "text", "text": text}, {"type": "image", "image": image}]
        content.append({"type": "text", "text": TEXT_AFTER.format(instruction=instruction)})
        return [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    @staticmethod
    def robot_state(states: dict[str, Any]) -> dict[str, np.ndarray]:
        arm = np.asarray(states.get("arm_joint_states", []), dtype=np.float32).reshape(-1)
        grip = np.asarray(states.get("gripper_states", []), dtype=np.float32).reshape(-1)
        if arm.size < 2 * ARM_DIM or grip.size < 2:
            raise ValueError(f"need {2 * ARM_DIM} arm and 2 gripper states, got {arm.size} and {grip.size}")
        pose = (states.get("end_pose") or {}).get("arm_base_link")
        if not pose:
            raise ValueError("payload has no states.end_pose.arm_base_link; XR-1 needs the current EEF pose "
                             "to turn its deltas into targets (the sim sends None when FK is unavailable)")
        out = {
            "left_arm_joint": arm[:ARM_DIM],
            "right_arm_joint": arm[ARM_DIM:2 * ARM_DIM],
            "left_gripper_pos": grip[0:1],
            "right_gripper_pos": grip[1:2],
        }
        for side in ("left", "right"):
            p = pose[f"{side}_arm"]
            out[f"{side}_ee_pos"] = np.asarray(p["position"], dtype=np.float64).reshape(3)
            # xyzw, same convention as the converter's R.from_quat on state/end/arm_orientation
            out[f"{side}_ee_rotm"] = R.from_quat(np.asarray(p["orientation"], dtype=np.float64)).as_matrix()
        return out

    # -- diagnostics ----------------------------------------------------------

    def _log_first(self, params: dict, images: list[Image.Image], robot: dict, prompt: str) -> None:
        if self._first_logged:
            return
        self._first_logged = True
        raw = params.get("images") or {}
        sizes = {k: (raw[k].get("width"), raw[k].get("height")) if isinstance(raw.get(k), dict) else None
                 for k in CAMERAS}
        logger.info("first request: source sizes %s -> model sizes %s", sizes, [im.size for im in images])
        logger.info("first request: gripper_states %s (dataset: open = -0.785)",
                    np.round(np.concatenate([robot["left_gripper_pos"], robot["right_gripper_pos"]]), 4).tolist())
        logger.info("first request: end_pose arm_base_link L %s R %s",
                    np.round(robot["left_ee_pos"], 4).tolist(), np.round(robot["right_ee_pos"], 4).tolist())
        logger.info("first request: prompt %r", prompt)

    def _track_step(self, params: dict, robot: dict) -> None:
        # A new episode starts from a reset pose, not from our last target.
        key = (_text(params.get("task_name")), params.get("episode_idx"), _text(params.get("prompt")))
        if key != self._episode_key:
            self._episode_key, self._last_target = key, None
        if self._last_target is None:
            return
        for side in ("left", "right"):
            for lag, (pos, rotm) in self._last_target[side].items():
                errs_p, errs_r = self._track[lag]
                errs_p.append(float(np.linalg.norm(robot[f"{side}_ee_pos"] - pos)) * 1000)
                errs_r.append(float(np.degrees(R.from_matrix(rotm.T @ robot[f"{side}_ee_rotm"]).magnitude())))

    def tracking_summary(self) -> str:
        parts = []
        for lag, (p, r) in self._track.items():
            if not p:
                return "no tracking samples yet"
            p, r = np.asarray(p), np.asarray(r)
            ref = "last step" if lag == 0 else f"step -{lag}"
            parts.append(f"vs {ref}: pos median {np.median(p):.1f} / p90 {np.percentile(p, 90):.1f} / max "
                         f"{p.max():.1f} mm, rot median {np.median(r):.2f} / p90 {np.percentile(r, 90):.2f} deg")
        return f"EEF tracking over {len(self._track[0][0])} arm-chunks | " + " | ".join(parts)

    # -- inference ------------------------------------------------------------

    def infer(self, params: dict[str, Any]) -> dict[str, Any]:
        torch = self.torch
        from mibot.utils.io import ACTION_EPS, compose_state, denormalize_action, recover_action, resize_image

        images_raw = params.get("images") or {}
        missing = [k for k in CAMERAS if k not in images_raw]
        if missing:
            raise ValueError(f"payload missing images {missing}")
        images = [resize_image(decode_image(images_raw[k]), factor=32, max_pixels=160000) for k in CAMERAS]
        robot = self.robot_state(params.get("states") or {})
        prompt = _text(params.get("prompt") or params.get("task_name"))
        self._log_first(params, images, robot, prompt)
        task = _text(params.get("task_name"))
        if self._seen_prompts.get(task) != prompt:
            self._seen_prompts[task] = prompt
            logger.info("task %s prompt %r", task, prompt)
        self._track_step(params, robot)

        batch = self.processor.apply_chat_template(
            [self._messages(prompt, images)], tokenize=True, return_dict=True, return_tensors="pt",
            padding=True, images_kwargs={"do_resize": False},
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        state = torch.from_numpy(compose_state(
            left_gripper=robot["left_gripper_pos"], left_joint=robot["left_arm_joint"],
            right_gripper=robot["right_gripper_pos"], right_joint=robot["right_arm_joint"],
        ))[None].to(self.device)
        # Same normalization as mibot/server/runtime/server.py.
        valid = self.q99 > self.q01
        norm = torch.zeros_like(state)
        norm[..., valid[0]] = (2.0 * (state[..., valid[0]] - self.q01[..., valid[0]])
                               / (self.q99[..., valid[0]] - self.q01[..., valid[0]] + ACTION_EPS) - 1.0)
        mask = self.action_mask.unsqueeze(0)
        batch.update(
            state=norm.clamp(-1.0, 1.0),
            action=torch.zeros((1, *self.mean.shape), device=self.device, dtype=torch.bfloat16),
            action_mask=mask,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.no_grad():
            action = self.model.generate(batch)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.monotonic() - t0
        self.n_infer += 1
        self.t_model += dt

        action = denormalize_action(action.float() * mask, self.mean, self.std) * mask
        targets = recover_action(action[0].cpu().numpy(), {
            k: robot[k] for k in ("left_ee_pos", "left_ee_rotm", "left_gripper_pos",
                                  "right_ee_pos", "right_ee_rotm", "right_gripper_pos")
        })
        result = self.to_result(targets, robot)

        last = self.execute_steps - 1
        self._last_target = {
            s: {lag: (targets[f"{s}_ee_pos"][max(last - lag, 0)].astype(np.float64),
                      targets[f"{s}_ee_rotm"][max(last - lag, 0)].astype(np.float64))
                for lag in self._track}
            for s in ("left", "right")
        }
        if self.n_infer == 1 or self.n_infer % 50 == 0:
            logger.info("infer #%d  model %.0f ms (avg %.0f ms)  %s%s", self.n_infer, dt * 1000,
                        self.t_model / self.n_infer * 1000, self.tracking_summary(), self.ik_summary())
        return result

    def ik_summary(self) -> str:
        if not self._ik_err:
            return ""
        e = np.asarray(self._ik_err) * 1000
        return (f" | server IK residual over {len(e)} steps: median {np.median(e):.3f} p99 "
                f"{np.percentile(e, 99):.3f} max {e.max():.2f} mm")

    def to_result(self, targets: dict[str, np.ndarray], robot: dict[str, np.ndarray]) -> dict[str, Any]:
        n = self.execute_steps
        result: dict[str, Any] = {}
        for side in ("left", "right"):
            pos = targets[f"{side}_ee_pos"][:n].astype(np.float64)
            rotm = targets[f"{side}_ee_rotm"][:n].astype(np.float64)
            if not (np.isfinite(pos).all() and np.isfinite(rotm).all()):
                raise FloatingPointError(f"non-finite {side} EEF target")
            if self.ik is not None:
                joints, err = self.ik.solve_chunk(side, robot[f"{side}_arm_joint"], pos, rotm)
                self._ik_err.extend(err.tolist())
                result[f"{side}_arm"] = {"kind": "JOINT_ABS", "values": joints.tolist()}
            else:
                quat = R.from_matrix(rotm).as_quat()  # xyzw
                result[f"{side}_arm"] = {
                    "kind": "EEF_ABS",
                    "values": np.concatenate([pos, quat], axis=1).tolist(),
                    "base_link": "arm_base_link",
                }
            grip = np.clip(targets[f"{side}_gripper_pos"][:n].astype(np.float64), *GRIPPER_RANGE)
            result[f"{side}_effector"] = grip.reshape(n, 1).tolist()
        return result


# ---------------------------------------------------------------------------
# challenge agent backend
# ---------------------------------------------------------------------------


class Backend:
    """The interface a challenge agent drives: lazy `load()`,
    `__call__(session_id, payload) -> bytes`, and `pack` for error replies.
    An empty payload is the gateway's warmup probe."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self.model_path = model_path
        self.kwargs = kwargs
        self.policy: XR1Policy | None = None
        self.pack, self.unpack = make_pack_unpack()

    def load(self) -> None:
        if self.policy is None:
            self.policy = XR1Policy(self.model_path, **self.kwargs)

    def __call__(self, session_id: str, payload: bytes) -> bytes:
        self.load()
        if not payload:
            return b""
        msg = self.unpack(payload)
        params = msg.get("params", msg) if isinstance(msg, dict) else None
        if not isinstance(params, dict):
            raise TypeError(f"expected dict payload, got {type(msg)}")
        return self.pack({"result": self.policy.infer(params)})


def add_policy_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=None, help="run config.py (default: found above the checkpoint)")
    ap.add_argument("--execute-steps", type=int, default=30,
                    help="actions of each 30-step chunk the sim replays before asking again")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm-output", choices=("joint", "eef"), default="joint",
                    help="joint: solve IK here and send JOINT_ABS; eef: send EEF_ABS for the sim's IK")
    ap.add_argument("--ik-sdk", default=None,
                    help="IK-SDK dir (G2_NO_GRIPPER.urdf, g2_solver.yaml); default $GENIESIM_IK_SDK, "
                         "else the copy in geniesim/ik_sdk")


def build_backend(model_path: str, argv: list[str]) -> Backend:
    """Entry point for an agent that loads this file as its model backend."""
    ap = argparse.ArgumentParser(prog="xr1 backend")
    add_policy_args(ap)
    ns = ap.parse_args(argv)
    return Backend(model_path, config=ns.config, execute_steps=ns.execute_steps, seed=ns.seed,
                   arm_output=ns.arm_output, ik_sdk=ns.ik_sdk)


# ---------------------------------------------------------------------------
# local benchmark server
# ---------------------------------------------------------------------------


def serve(policy: XR1Policy, host: str, port: int) -> None:
    import websockets.asyncio.server as ws_server

    pack, unpack = make_pack_unpack()

    async def handler(ws) -> None:
        # No metadata frame on connect: the benchmark's corobot client does not
        # read one and would take it as the first reply.
        while True:
            try:
                raw = await ws.recv()
            except Exception as exc:  # noqa: BLE001
                if exc.__class__.__name__.startswith("ConnectionClosed"):
                    logger.info("connection closed; %s", policy.tracking_summary())
                    return
                raise
            try:
                msg = unpack(raw)
                params = msg.get("params", msg) if isinstance(msg, dict) else None
                if not isinstance(params, dict):
                    raise TypeError(f"expected dict payload, got {type(msg)}")
                result = await asyncio.get_running_loop().run_in_executor(None, policy.infer, params)
                await ws.send(pack({"result": result}))
            except Exception:  # noqa: BLE001
                logger.exception("inference error")
                await ws.send(pack({"error": "inference_failed", "traceback": traceback.format_exc()}))

    async def run() -> None:
        async with ws_server.serve(handler, host, port, compression=None, max_size=2**28) as server:
            await server.serve_forever()

    asyncio.run(run())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8999)
    ap.add_argument("--device", default="cuda:0")
    add_policy_args(ap)
    ns = ap.parse_args()
    policy = XR1Policy(ns.ckpt, config=ns.config, device=ns.device, execute_steps=ns.execute_steps,
                       seed=ns.seed, arm_output=ns.arm_output, ik_sdk=ns.ik_sdk)
    logger.info("serving on %s:%d, %d of %d steps per chunk, arm output %s", ns.host, ns.port,
                ns.execute_steps, policy.action_length, ns.arm_output)
    serve(policy, ns.host, ns.port)


if __name__ == "__main__":
    main()
