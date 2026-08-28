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

RAY_BIN="${RAY_BIN:-/opt/piper-venv/bin/ray}"
PYTHON_BIN="${PYTHON_BIN:-/opt/piper-venv/bin/python}"
PORT="${RAY_PORT:-6379}"
NUM_GPUS="${RAY_NUM_GPUS:-}"
TEMP_DIR="${RAY_TEMP_DIR:-/tmp/piper/ray_tmp}"
NETWORK_INTERFACE="${RAY_NETWORK_INTERFACE:-}"
STOP_EXISTING=true
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: scripts/start_ray_head.sh [options]

Start a Ray head inside the current host/container.

Options:
  --port PORT               Ray head port (default: 6379).
  --num-gpus COUNT          GPUs advertised by this node (default: auto-detect).
  --temp-dir PATH           Ray temporary directory (default: /tmp/piper/ray_tmp).
  --network-interface NAME  Export NAME for NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME.
  --no-stop                 Do not stop an existing local Ray daemon first.
  --dry-run                 Print the resolved command without changing Ray state.
  -h, --help                Show this help.

Environment equivalents: RAY_PORT, RAY_NUM_GPUS, RAY_TEMP_DIR,
RAY_NETWORK_INTERFACE, RAY_BIN, and PYTHON_BIN.
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

detect_node_ip() {
    if [[ -n "$NETWORK_INTERFACE" ]]; then
        command -v ip >/dev/null 2>&1 || die "the ip command is required with --network-interface"
        ip -o -4 addr show dev "$NETWORK_INTERFACE" scope global \
            | awk 'NR == 1 { split($4, parts, "/"); print parts[1] }'
        return
    fi
    local detected
    if detected="$("$PYTHON_BIN" -c 'import ray; print(ray.util.get_node_ip_address())' 2>/dev/null)" \
        && [[ -n "$detected" ]]; then
        echo "$detected"
        return
    fi
    if command -v hostname >/dev/null 2>&1 \
        && detected="$(hostname -I 2>/dev/null | awk '{ print $1 }')" \
        && [[ -n "$detected" ]]; then
        echo "$detected"
        return
    fi
    return 1
}

detect_num_gpus() {
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" \
          && "${CUDA_VISIBLE_DEVICES}" != "-1" \
          && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        local visible
        visible="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
        local devices
        IFS=',' read -r -a devices <<<"$visible"
        echo "${#devices[@]}"
        return
    fi
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required to auto-detect GPUs"
    nvidia-smi -L | awk 'END { print NR }'
}

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --temp-dir) TEMP_DIR="$2"; shift 2 ;;
        --network-interface) NETWORK_INTERFACE="$2"; shift 2 ;;
        --no-stop) STOP_EXISTING=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -x "$RAY_BIN" ]] || die "Ray executable not found: $RAY_BIN"
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) \
    || die "invalid port: $PORT"

if [[ -n "$NETWORK_INTERFACE" && ! -d "/sys/class/net/$NETWORK_INTERFACE" ]]; then
    die "network interface does not exist: $NETWORK_INTERFACE"
fi

if ! NODE_IP="$(detect_node_ip)"; then
    die "could not detect a cluster-reachable node IP"
fi
[[ -n "$NODE_IP" ]] || die "could not detect a cluster-reachable node IP"

if [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS="$(detect_num_gpus)"
fi
[[ "$NUM_GPUS" =~ ^[0-9]+$ ]] && (( NUM_GPUS > 0 )) \
    || die "invalid GPU count: $NUM_GPUS"

if [[ -n "$NETWORK_INTERFACE" ]]; then
    export NCCL_SOCKET_IFNAME="$NETWORK_INTERFACE"
    export GLOO_SOCKET_IFNAME="$NETWORK_INTERFACE"
fi

START_COMMAND=(
    "$RAY_BIN" start
    --head
    "--node-ip-address=$NODE_IP"
    "--port=$PORT"
    "--num-gpus=$NUM_GPUS"
    --disable-usage-stats
    --include-dashboard=false
    "--temp-dir=$TEMP_DIR"
)

echo "Ray head configuration: node=$NODE_IP port=$PORT gpus=$NUM_GPUS temp_dir=$TEMP_DIR"
if [[ -n "$NETWORK_INTERFACE" ]]; then
    echo "Communication interface: $NETWORK_INTERFACE"
fi

if [[ "$STOP_EXISTING" == true ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        print_command "$RAY_BIN" stop -f
    else
        "$RAY_BIN" stop -f >/dev/null 2>&1 || true
    fi
fi

if [[ "$DRY_RUN" == true ]]; then
    print_command "${START_COMMAND[@]}"
    exit 0
fi

mkdir -p "$TEMP_DIR"
"${START_COMMAND[@]}"

cat <<EOF

Ray head started at $NODE_IP:$PORT.

On each worker, run:
  scripts/join_ray_worker.sh --head-address $NODE_IP:$PORT --node-ip <worker-private-ip>

Verify the cluster:
  $RAY_BIN status --address=$NODE_IP:$PORT

Pass this address to the Piper harness:
  --address $NODE_IP --port $PORT
EOF
