#!/usr/bin/env bash
# Build the XR-1 training and eval environment in <repo>/.venv.
#
# TORCH_INDEX picks the torch build. The default cu126 suits a driver that
# ships CUDA 12.6; with a 12.8 driver the plain PyPI wheel works too
# (TORCH_INDEX=https://pypi.org/simple). nvcc must be under CUDA_HOME for
# flash-attn if no prebuilt wheel matches.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}

cd "$REPO"
[ -d .venv ] || "$UV" venv --python 3.11 .venv
PY="$REPO/.venv/bin/python"

echo "== torch ($TORCH_INDEX) =="
"$UV" pip install --python "$PY" torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX"

echo "== xr1 + requirements =="
"$UV" pip install --python "$PY" -e xr1

echo "== converter deps (geniesim/lerobot_to_xr1.py) =="
"$UV" pip install --python "$PY" pandas pyarrow scipy

echo "== eval deps (geniesim/xr1_policy.py) =="
"$UV" pip install --python "$PY" msgpack msgpack-numpy websockets opencv-python-headless
# The sim's IK solver (geniesim/g2_ik.py) is a compiled wheel that only ships
# inside the geniesim container; copy it out and point IK_SOLVER_WHEEL at it.
#
# The wheel is built on Ubuntu 24.04 but only needs glibc 2.35, so it also
# runs on 22.04 -- given two system libraries 22.04 lacks. Without root, unpack
# the jammy packages into a private dir and put it on LD_LIBRARY_PATH for
# every process that imports ik_solver:
#   export LD_LIBRARY_PATH=$SYSLIBS/root/usr/lib/x86_64-linux-gnu
# (A 24.04 libtinyxml needs glibc 2.38; don't copy one over.) On 24.04 the
# libraries come from apt: libtinyxml2.6.2v5 libminizip1.
SYSLIBS=${SYSLIBS:-$REPO/.syslibs}
if [ -n "${IK_SOLVER_WHEEL:-}" ]; then
    "$UV" pip install --python "$PY" "$IK_SOLVER_WHEEL"
    mkdir -p "$SYSLIBS/deb"
    (cd "$SYSLIBS/deb" && apt-get download libtinyxml2.6.2v5 libminizip1 \
        && for f in *.deb; do dpkg -x "$f" "$SYSLIBS/root"; done)
    echo "   ik_solver needs: export LD_LIBRARY_PATH=$SYSLIBS/root/usr/lib/x86_64-linux-gnu"
else
    echo "   IK_SOLVER_WHEEL not set: server-side IK (--arm-output joint) unavailable"
fi

echo "== flash-attn =="
"$UV" pip install --python "$PY" setuptools wheel ninja packaging psutil
MAX_JOBS=16 "$UV" pip install --python "$PY" flash-attn --no-build-isolation

echo "== check =="
"$PY" - <<'PY'
import torch, transformers, deepspeed, decord, flash_attn, pandas, pyarrow, scipy
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, " deepspeed", deepspeed.__version__,
      " decord", decord.__version__, " flash_attn", flash_attn.__version__)
PY
