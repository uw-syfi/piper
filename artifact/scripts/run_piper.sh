#!/usr/bin/env bash
set -euo pipefail

MODEL=1B
SCHEDULE=1f1b
PP=1
DP=1
EP=false
ZERO_STAGE=1
BUCKET_SIZE=""
BATCH_SIZE=4
SEQ_LEN=512
MBS=1
WARMUP=3
ITERS=10
ITERATION_SLEEP=0
RAY_ADDRESS=""
RAY_PORT=6379
TEMP_DIR=/tmp/piper/ray_tmp
METRICS_OUT=""
NSIGHT=false
USE_INDUCTOR_ARG=--use-inductor
PP_OUTER_ARG=--no-pp-outer

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --schedule) SCHEDULE="$2"; shift 2 ;;
        --pp) PP="$2"; shift 2 ;;
        --dp) DP="$2"; shift 2 ;;
        --ep) EP=true; shift ;;
        --zero-stage) ZERO_STAGE="$2"; shift 2 ;;
        --bucket-size) BUCKET_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seq-len) SEQ_LEN="$2"; shift 2 ;;
        --mbs) MBS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --iteration-sleep) ITERATION_SLEEP="$2"; shift 2 ;;
        --address) RAY_ADDRESS="$2"; shift 2 ;;
        --port) RAY_PORT="$2"; shift 2 ;;
        --temp-dir) TEMP_DIR="$2"; shift 2 ;;
        --metrics-out) METRICS_OUT="$2"; shift 2 ;;
        --nsight) NSIGHT=true; shift ;;
        --use-inductor) USE_INDUCTOR_ARG=--use-inductor; shift ;;
        --no-use-inductor) USE_INDUCTOR_ARG=--no-use-inductor; shift ;;
        --gradient-accumulation|--no-gradient-accumulation) shift ;;
        --ar-a2a-same-stream|--no-ar-a2a-same-stream) shift ;;
        --overlap-chunks|--no-overlap-chunks) shift ;;
        --pp-outer) PP_OUTER_ARG=--pp-outer; shift ;;
        --no-pp-outer) PP_OUTER_ARG=--no-pp-outer; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$SCHEDULE" in
    1f1b) HARNESS_SCHEDULE=1f1b; VIRTUAL_STAGES=1 ;;
    interleaved1f1b|interleaved-1f1b|interleaved_1f1b) HARNESS_SCHEDULE=interleaved_1f1b; VIRTUAL_STAGES=2 ;;
    zerobubble|interleaved-zerobubble|interleavedzerobubble) HARNESS_SCHEDULE=zerobubble; VIRTUAL_STAGES=1 ;;
    dualpipe|dualpipev) HARNESS_SCHEDULE=dualpipev; VIRTUAL_STAGES=2 ;;
    *) echo "Unsupported Piper schedule: $SCHEDULE" >&2; exit 2 ;;
esac

BASE_SCHEDULE="$(mktemp /tmp/piper-base-schedule.XXXXXX.json)"
BASE_ARGS=(
    --pp "$PP"
    --dp "$DP"
    --virtual-stages "$VIRTUAL_STAGES"
    --layout "$([[ "$HARNESS_SCHEDULE" == "dualpipev" ]] && echo v || echo linear)"
    --zero-stage "$ZERO_STAGE"
    --output "$BASE_SCHEDULE"
)
if $EP; then
    BASE_ARGS+=(--ep)
fi
if [[ -n "$BUCKET_SIZE" ]]; then
    BASE_ARGS+=(--bucket-size "$BUCKET_SIZE")
fi
python /workspace/artifact/scripts/make_piper_base_schedule.py "${BASE_ARGS[@]}"

HARNESS_ARGS=(
    examples/test_harness.py
    --test-file examples/test_qwen.py
    --base-schedule "$BASE_SCHEDULE"
    --schedule "$HARNESS_SCHEDULE"
    --ranks "$PP"
    --mbs "$MBS"
    --temp-dir "$TEMP_DIR"
)
if [[ -n "$RAY_ADDRESS" ]]; then
    HARNESS_ARGS+=(--address "$RAY_ADDRESS" --port "$RAY_PORT")
fi

TEST_ARGS=(
    --model "$MODEL"
    --batch-size "$BATCH_SIZE"
    --seq-len "$SEQ_LEN"
    --warmup "$WARMUP"
    --iters "$ITERS"
    --iteration-sleep "$ITERATION_SLEEP"
    "$USE_INDUCTOR_ARG"
    "$PP_OUTER_ARG"
)
if $NSIGHT; then
    TEST_ARGS+=(--nsight)
fi

cd /workspace/piper
export PYTHONPATH="/workspace/piper:/workspace/piper/examples:${PYTHONPATH:-}"
set +e
conda run -n piper python "${HARNESS_ARGS[@]}" "${TEST_ARGS[@]}"
STATUS=$?
set -e

if [[ -n "$METRICS_OUT" ]]; then
    mkdir -p "$(dirname "$METRICS_OUT")"
    LATEST_RUN="$(ls -td out/* 2>/dev/null | head -1 || true)"
    if [[ -n "$LATEST_RUN" && -f "$LATEST_RUN/results.csv" ]]; then
        cp "$LATEST_RUN/results.csv" "$METRICS_OUT"
        echo "Piper metrics copied to $METRICS_OUT"
    else
        echo "Piper metrics were not found under /workspace/piper/out" >&2
    fi
fi

rm -f "$BASE_SCHEDULE"
exit "$STATUS"
