#!/usr/bin/env bash
# Time a short XR-1 training run and record peak GPU memory, then delete its
# checkpoints (each is ~64 GB with optimizer state).
#
#   bash geniesim/bench_step_time.sh <exp_name> [hydra overrides...]
#   bash geniesim/bench_step_time.sh bench_sync model.params.model.async_train=false
#
# Prints the steady-state step time (median of steps 6..N, skipping start-up and
# the first-step FusedAdam compile) and the peak memory across all GPUs.
set -euo pipefail

EXP=${1:?usage: bench_step_time.sh <exp_name> [hydra overrides...]}
shift
STEPS=${STEPS:-30}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT=${OUT_ROOT:-$REPO/outputs}
OUT="$OUT_ROOT/$EXP"
mkdir -p "$OUT"
# Timing runs are throwaway; keep them out of the wandb project.
export WANDB_MODE=${WANDB_MODE:-offline}

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 1000 > "$OUT/mem.log" &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

# Lightning's ModelCheckpoint has no "never"; push the interval past the run.
OUT_ROOT="$OUT_ROOT" bash "$REPO/geniesim/train_geniesim.sh" "$EXP" \
    trainer.max_steps="$STEPS" trainer.save_interval=1000000 \
    +trainer.log_every_n_steps=1 "$@" > /dev/null

kill $SMI 2>/dev/null || true
rm -rf "$OUT/project_"*/"$EXP"/*.ckpt

# tqdm prints the running step count and elapsed time; difference consecutive
# updates to get per-step wall time.
python3 - "$OUT/train.log" "$OUT/mem.log" <<'PY'
import re, sys, statistics
log, mem = sys.argv[1:3]
text = open(log, errors="ignore").read().replace("\r", "\n")
seen = {}
for m in re.finditer(r"Epoch \d+:\s+\d+%\|[^|]*\|\s*(\d+)/\d+ \[(\d+):(\d+)(?::(\d+))?", text):
    step = int(m.group(1))
    a, b, c = m.group(2), m.group(3), m.group(4)
    secs = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
    seen.setdefault(step, secs)
steps = sorted(seen)
dts = [seen[s] - seen[p] for p, s in zip(steps, steps[1:]) if s == p + 1 and p >= 5]
peak = max(int(x) for x in open(mem).read().split() if x.strip().isdigit())
if dts:
    print(f"steps timed {len(dts)}  median {statistics.median(dts):.2f} s/step  "
          f"mean {statistics.mean(dts):.2f}  peak mem {peak} MiB")
else:
    print(f"no step timings parsed (last step seen {steps[-1] if steps else None}); peak mem {peak} MiB")
PY
