#!/usr/bin/env bash
# Post-train Xiaomi-Robotics-1 on converted Genie Sim episodes, one node.
#
#   bash geniesim/train_geniesim.sh <exp_name> [hydra overrides...]
#   bash geniesim/train_geniesim.sh smoke trainer.max_steps=15
#
# Environment knobs (defaults in brackets):
#   GPUS        number of local GPUs                      [8]
#   DATA        configs/data/<DATA>.yaml                  [geniesim_instruction]
#   DATA_ROOT   where the pretrained weights live         [<repo>/data]
#   CKPT        pretrained model_states.pt                [$DATA_ROOT/Xiaomi-Robotics-1-5B/model_states.pt]
#   OUT_ROOT    checkpoints + logs go to $OUT_ROOT/<exp>  [<repo>/outputs]
#   HF_HUB_OFFLINE  set 1 once Qwen/Qwen3-VL-4B-Instruct (config + tokenizer) is
#               cached, so eight ranks don't each re-check the hub   [unset]
#   WANDB_MODE  tools/train.py hard-wires WandbLogger        [online if WANDB_API_KEY is set, else offline]
#   WANDB_ENTITY  wandb team/user to log under               [the logged-in account's default]
set -euo pipefail

EXP=${1:?usage: train_geniesim.sh <exp_name> [hydra overrides...]}
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS=${GPUS:-8}
DATA=${DATA:-geniesim_instruction}
DATA_ROOT=${DATA_ROOT:-$REPO/data}
CKPT=${CKPT:-$DATA_ROOT/Xiaomi-Robotics-1-5B/model_states.pt}
OUT_ROOT=${OUT_ROOT:-$REPO/outputs}
OUT="$OUT_ROOT/$EXP"

if [ "${HF_HUB_OFFLINE:-0}" = 1 ]; then export TRANSFORMERS_OFFLINE=1; fi
export WANDB_MODE=${WANDB_MODE:-$([ -n "${WANDB_API_KEY:-}" ] && echo online || echo offline)}
export WANDB_DIR="$OUT"
# DeepSpeed's FusedAdam JIT-compiles a CUDA op on first use; nvcc is in
# /usr/local/cuda but not on PATH by default.
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$REPO/.venv/bin:$PATH"

[ -f "$CKPT" ] || { echo "checkpoint not found: $CKPT"; exit 1; }
[ -f "$REPO/xr1/configs/data/$DATA.yaml" ] || { echo "no data config: $DATA (run geniesim/make_data_config.py)"; exit 1; }
mkdir -p "$OUT"

echo "exp=$EXP gpus=$GPUS data=$DATA out=$OUT"
cd "$REPO/xr1"   # scripts/train.sh puts $(pwd) on PYTHONPATH
RESOURCE_GPU=$GPUS bash scripts/train.sh \
    trainer.project=xiaomi-robotics-1 \
    trainer.exp_name="$EXP" \
    trainer.default_root_dir="$OUT" \
    data="$DATA" \
    model=posttrain \
    model.params.pretrained="$CKPT" \
    "$@" 2>&1 | tee "$OUT/train.log"
