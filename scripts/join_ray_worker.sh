#!/usr/bin/env bash
set -euo pipefail

RAY_BIN="${RAY_BIN:-/opt/piper-venv/bin/ray}"
PYTHON_BIN="${PYTHON_BIN:-/opt/piper-venv/bin/python}"
HEAD_ADDRESS="${RAY_HEAD_ADDRESS:-}"
PORT="${RAY_PORT:-6379}"
NODE_IP="${RAY_NODE_IP:-}"
NUM_GPUS="${RAY_NUM_GPUS:-}"
NETWORK_INTERFACE="${RAY_NETWORK_INTERFACE:-}"
MIN_WORKER_PORT="${RAY_MIN_WORKER_PORT:-0}"
MAX_WORKER_PORT="${RAY_MAX_WORKER_PORT:-0}"
STOP_EXISTING=true
DRY_RUN=false

usage() {
    cat <<'EOF'
Usage: scripts/join_ray_worker.sh --head-address HOST[:PORT] [options]

Join the current host/container to an existing Ray head.

Options:
  --head-address ADDRESS    Required Ray head hostname/IP, optionally with port.
  --port PORT               Port appended when ADDRESS omits it (default: 6379).
  --node-ip ADDRESS         IP advertised by this worker (default: auto-detect).
  --num-gpus COUNT          GPUs advertised by this node (default: auto-detect).
  --network-interface NAME  Export NAME for NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME.
  --min-worker-port PORT    Lowest worker port, or 0 for random (default: 0).
  --max-worker-port PORT    Highest worker port, or 0 for random (default: 0).
  --no-stop                 Do not stop an existing local Ray daemon first.
  --dry-run                 Print the resolved command without changing Ray state.
  -h, --help                Show this help.

Environment equivalents: RAY_HEAD_ADDRESS, RAY_PORT, RAY_NODE_IP,
RAY_NUM_GPUS, RAY_NETWORK_INTERFACE, RAY_MIN_WORKER_PORT,
RAY_MAX_WORKER_PORT, RAY_BIN, and PYTHON_BIN.
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
        --head-address) HEAD_ADDRESS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --node-ip) NODE_IP="$2"; shift 2 ;;
        --num-gpus) NUM_GPUS="$2"; shift 2 ;;
        --network-interface) NETWORK_INTERFACE="$2"; shift 2 ;;
        --min-worker-port) MIN_WORKER_PORT="$2"; shift 2 ;;
        --max-worker-port) MAX_WORKER_PORT="$2"; shift 2 ;;
        --no-stop) STOP_EXISTING=false; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *)
            if [[ -z "$HEAD_ADDRESS" && "$1" != -* ]]; then
                HEAD_ADDRESS="$1"
                shift
            else
                die "unknown argument: $1"
            fi
            ;;
    esac
done

[[ -x "$RAY_BIN" ]] || die "Ray executable not found: $RAY_BIN"
[[ -x "$PYTHON_BIN" ]] || die "Python executable not found: $PYTHON_BIN"
[[ -n "$HEAD_ADDRESS" ]] || die "--head-address is required"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) \
    || die "invalid port: $PORT"
[[ "$MIN_WORKER_PORT" =~ ^[0-9]+$ ]] && (( MIN_WORKER_PORT <= 65535 )) \
    || die "invalid minimum worker port: $MIN_WORKER_PORT"
[[ "$MAX_WORKER_PORT" =~ ^[0-9]+$ ]] && (( MAX_WORKER_PORT <= 65535 )) \
    || die "invalid maximum worker port: $MAX_WORKER_PORT"
if (( MIN_WORKER_PORT == 0 || MAX_WORKER_PORT == 0 )); then
    (( MIN_WORKER_PORT == 0 && MAX_WORKER_PORT == 0 )) \
        || die "minimum and maximum worker ports must both be 0 for random ports"
else
    (( MIN_WORKER_PORT >= 1024 && MAX_WORKER_PORT > MIN_WORKER_PORT )) \
        || die "worker port range must be within 1024-65535 and maximum must exceed minimum"
fi

if [[ "$HEAD_ADDRESS" != *:* ]]; then
    HEAD_ADDRESS="$HEAD_ADDRESS:$PORT"
fi

if [[ -n "$NETWORK_INTERFACE" && ! -d "/sys/class/net/$NETWORK_INTERFACE" ]]; then
    die "network interface does not exist: $NETWORK_INTERFACE"
fi

if [[ -z "$NODE_IP" ]]; then
    if ! NODE_IP="$(detect_node_ip)"; then
        die "could not detect a cluster-reachable node IP"
    fi
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
    "--address=$HEAD_ADDRESS"
    "--node-ip-address=$NODE_IP"
    "--num-gpus=$NUM_GPUS"
    "--min-worker-port=$MIN_WORKER_PORT"
    "--max-worker-port=$MAX_WORKER_PORT"
    --disable-usage-stats
)

if (( MIN_WORKER_PORT == 0 )); then
    WORKER_PORTS=random
else
    WORKER_PORTS="$MIN_WORKER_PORT-$MAX_WORKER_PORT"
fi
echo "Ray worker configuration: node=$NODE_IP head=$HEAD_ADDRESS gpus=$NUM_GPUS worker_ports=$WORKER_PORTS"
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

"${START_COMMAND[@]}"

cat <<EOF

Ray worker $NODE_IP joined $HEAD_ADDRESS.

On the head, verify registration with:
  $RAY_BIN status --address=$HEAD_ADDRESS
EOF
