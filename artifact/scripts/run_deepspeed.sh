#!/usr/bin/env bash
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
