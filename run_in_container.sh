#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
