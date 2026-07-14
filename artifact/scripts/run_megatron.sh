#!/usr/bin/env bash
set -euo pipefail

NNODE=1
NGPU=8
NODE_RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
MODEL=qwen3_1b
TP=1
PP=1
DP=1
CP=1
EP=1
NSIGHT=false
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --nnode) NNODE="$2"; shift 2 ;;
        --ngpu) NGPU="$2"; shift 2 ;;
        --node-rank) NODE_RANK="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --tp) TP="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --dp) DP="$2"; shift 2 ;;
        --cp) CP="$2"; shift 2 ;;
        --ep) EP="$2"; shift 2 ;;
        --nsight) NSIGHT=true; shift ;;
        --) shift; TRAIN_ARGS=("$@"); break ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

NSIGHT_ARGS=()
if $NSIGHT; then
    NSIGHT_ARGS=(--nsight --tensorboard-dir "/tmp/megatron_profiler/${MODEL}_pp${PP}_dp${DP}_ep${EP}")
fi

cd /workspace/Megatron-LM
exec conda run -n megatron env \
    NODE_RANK="${NODE_RANK}" \
    NCCL_P2P_DISABLE=1 \
    NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}" \
    GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-}" \
    python /workspace/artifact/run_megatron.py \
        --model "${MODEL}" \
        --nnodes "${NNODE}" \
        --nproc-per-node "${NGPU}" \
        --master-addr "${MASTER_ADDR}" \
        --master-port "${MASTER_PORT}" \
        --disable-background-mode \
        --tp "${TP}" --pp "${PP}" --dp "${DP}" --cp "${CP}" --ep "${EP}" \
        "${NSIGHT_ARGS[@]}" \
        "${TRAIN_ARGS[@]}"
