#!/usr/bin/env bash
set -euo pipefail

PERSIST_BASE=/lustre/fsw/portfolios/coreai/projects/coreai_dlcompiler_agent/users/mfrisella
REPO=$PERSIST_BASE/piper
CODEX_PERSIST=$PERSIST_BASE/.codex
CUDA_COMPAT_PATH=${CUDA_COMPAT_PATH:-/usr/local/cuda-12.8/compat}

# CUDA Green Contexts require newer user-space driver entry points than the
# R535 host driver exports.  The image supplies them through CUDA forward
# compatibility; keep that libcuda ahead of the host libraries mounted by the
# NVIDIA container runtime.
if [[ -r "$CUDA_COMPAT_PATH/libcuda.so.1" ]]; then
    export LD_LIBRARY_PATH="$CUDA_COMPAT_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$CODEX_PERSIST"

# Recreate container-side ~/.codex every time the container starts
rm -rf "$HOME/.codex"
ln -sfn "$CODEX_PERSIST" "$HOME/.codex"

cd "$REPO"

exec bash -i
