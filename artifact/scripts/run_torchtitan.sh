#!/usr/bin/env bash
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

NNODE=1
NGPU=8
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
MODULE=qwen3
CONFIG=qwen3_9b
LOG_RANK=0
NSIGHT=false
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnode) NNODE="$2"; shift 2 ;;
        --ngpu) NGPU="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --module) MODULE="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --log-rank) LOG_RANK="$2"; shift 2 ;;
        --nsight) NSIGHT=true; shift ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

RUNNER=(
    torchrun
    "--nnodes=${NNODE}"
    "--nproc_per_node=${NGPU}"
    "--node_rank=${NODE_RANK}"
    "--master_addr=${MASTER_ADDR}"
    "--master_port=${MASTER_PORT}"
    "--local-ranks-filter" "${LOG_RANK}"
    "--role" "rank"
    "--tee" "3"
    -m torchtitan.train
    --module "${MODULE}"
    --config "${CONFIG}"
    "${TRAIN_ARGS[@]}"
)

if $NSIGHT; then
    mkdir -p /workspace/eval-out/nsight
    RUNNER=(nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true --output "/workspace/eval-out/nsight/torchtitan_node${NODE_RANK}" "${RUNNER[@]}")
fi

cd /workspace/torchtitan
exec conda run -n torchtitan env \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    TORCHFT_LIGHTHOUSE=http://localhost:29510 \
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}" \
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}" \
    "${RUNNER[@]}"
