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
MASTER_PORT=29501
MODEL=qwen3_1b
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnode) NNODE="$2"; shift 2 ;;
        --ngpu) NGPU="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

cd /workspace/piper
exec conda run -n deepspeed env \
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}" \
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}" \
    torchrun \
        "--nnodes=${NNODE}" \
        "--nproc_per_node=${NGPU}" \
        "--node_rank=${NODE_RANK}" \
        "--master_addr=${MASTER_ADDR}" \
        "--master_port=${MASTER_PORT}" \
        /workspace/artifact/run_deepspeed.py \
        --model "${MODEL}" \
        "${TRAIN_ARGS[@]}"
