#!/usr/bin/env bash
# Every application dependency belongs to this project.
PHP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PHP_BASE_ENV="$PHP_ROOT/.runtime/b200/env"
export PHP_ROOT PHP_BASE_ENV
export PATH="$PHP_BASE_ENV/bin:$PHP_ROOT/.runtime/b200/host-tools"
export PYTHONPATH="$PHP_ROOT/thirdparty/holosoma/src/holosoma:$PHP_ROOT:$PHP_ROOT/.runtime/b200/IsaacLab/source/isaaclab:$PHP_ROOT/.runtime/b200/IsaacLab/source/isaaclab_assets:$PHP_ROOT/.runtime/b200/IsaacLab/source/isaaclab_tasks:$PHP_ROOT/.runtime/b200/IsaacLab/source/isaaclab_rl"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export ENABLE_CAMERAS=0 LIVESTREAM=0
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export XDG_CACHE_HOME="$PHP_ROOT/.runtime/b200/cache"
export XDG_CONFIG_HOME="$PHP_ROOT/.runtime/b200/config"
export TMPDIR="$PHP_ROOT/.runtime/b200/tmp"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export XDG_RUNTIME_DIR="$PHP_ROOT/.runtime/b200/run"
export WARP_CACHE_PATH="$XDG_CACHE_HOME/warp" CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export MPLCONFIGDIR="$XDG_CONFIG_HOME/matplotlib"
export WANDB_DIR="$PHP_ROOT/outputs/wandb" WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$XDG_CONFIG_HOME/wandb" WANDB_DATA_DIR="$XDG_CACHE_HOME/wandb-data"
export WANDB_ENTITY=humanoid-parkour WANDB_PROJECT=php-parkour
export LOGURU_LEVEL=WARNING
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER
# Load the project's private credential copy without displaying it.
if [[ -z ${WANDB_API_KEY:-} && -r "$PHP_ROOT/.runtime/b200/secrets/wandb-api-key" ]]; then
    WANDB_API_KEY=$(< "$PHP_ROOT/.runtime/b200/secrets/wandb-api-key")
    export WANDB_API_KEY
fi
if [[ -z ${WANDB_BASE_URL:-} && -r "$PHP_ROOT/.runtime/b200/secrets/wandb-base-url" ]]; then
    WANDB_BASE_URL=$(< "$PHP_ROOT/.runtime/b200/secrets/wandb-base-url")
    export WANDB_BASE_URL
fi
# Reuse the license acceptance already recorded on this server; do not prompt
# or write a new acceptance as part of environment activation.
if [[ $(cat "$PHP_ROOT/.runtime/b200/isaac-sim-eula.accepted" 2>/dev/null) == ACCEPT ]]; then
    export OMNI_KIT_ACCEPT_EULA=YES
fi
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR" "$XDG_RUNTIME_DIR"
cd "$PHP_ROOT"
