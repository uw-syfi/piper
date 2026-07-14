#!/usr/bin/env python3
"""Run the Qwen example schedule set with the normal execution path."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


QWEN_MODELS = ("9M", "1B", "9B", "48B", "30B-A3B", "30-A3B-half", "72B")

# name, base schedule, generated schedule, pp ranks, microbatches
RUNS = (
    (
        "dualpipev",
        "examples/base-schedules/pp4_dp2_ep2_v_placement.json",
        "dualpipev",
        2,
        4,
    ),
    ("zero3_1f1b", "examples/base-schedules/pp2_dp2_ep2_zero3.json", "1f1b", 2, 4),
    (
        "bucket100_1f1b",
        "examples/base-schedules/pp2_dp2_ep2_bucket100.json",
        "1f1b",
        2,
        4,
    ),
    (
        "interleaved_1f1b",
        "examples/base-schedules/pp4_dp2_ep2_interleaved.json",
        "interleaved_1f1b",
        2,
        4,
    ),
    ("zero2_1f1b", "examples/base-schedules/pp2_dp2_ep2_zero2.json", "1f1b", 2, 4),
    (
        "custom_order",
        "examples/base-schedules/pp2_dp2_ep2_custom_order.json",
        "custom",
        2,
        2,
    ),
    ("zerobubble", "examples/base-schedules/pp2_dp2_ep2.json", "zerobubble", 2, 4),
)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    python_bin = _python_bin(repo_root)
    forwarded_args = _forwarded_args(args)

    print(f"Python: {python_bin}", flush=True)
    if forwarded_args:
        print(f"Forwarded args: {' '.join(forwarded_args)}", flush=True)

    status = 0
    for name, base_schedule, schedule, ranks, mbs in RUNS:
        print(flush=True)
        print(
            f"==> {name}: {schedule} {base_schedule} --ranks {ranks} --mbs {mbs}",
            flush=True,
        )
        cmd = [
            python_bin,
            "examples/test_harness.py",
            "--test-file",
            "examples/test_qwen.py",
            "--base-schedule",
            base_schedule,
            "--schedule",
            schedule,
            "--ranks",
            str(ranks),
            "--mbs",
            str(mbs),
            *forwarded_args,
        ]
        result = subprocess.run(cmd, cwd=repo_root, check=False)
        if result.returncode:
            print(f"FAILED: {name}", file=sys.stderr, flush=True)
            status = 1
        else:
            print(f"PASSED: {name}", flush=True)

    return status


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all Qwen example schedules with the normal execution path."
    )
    parser.add_argument("--model", choices=QWEN_MODELS)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--iteration-sleep", type=float)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--nsight", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument(
        "--use-inductor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Compile actor stage modules with torch.compile.",
    )
    parser.add_argument(
        "--pp-outer",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use PP as the outer placement dimension.",
    )
    parser.add_argument("--pytorch-profiler", action="store_true")
    parser.add_argument("--pytorch-profiler-iters", type=int)
    parser.add_argument("--address")
    parser.add_argument("--port", type=int)
    parser.add_argument("--temp-dir", help="Ray temp directory.")
    parser.add_argument("--ray-namespace")
    return parser.parse_args(argv)


def _forwarded_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    _append_value(forwarded, "--model", args.model)
    _append_value(forwarded, "--batch-size", args.batch_size)
    _append_value(forwarded, "--seq-len", args.seq_len)
    _append_value(forwarded, "--warmup", args.warmup)
    _append_value(forwarded, "--iters", args.iters)
    _append_value(forwarded, "--iteration-sleep", args.iteration_sleep)
    if args.activation_checkpointing:
        forwarded.append("--activation-checkpointing")
    if args.nsight:
        forwarded.append("--nsight")
    if args.viz:
        forwarded.append("--viz")
    if args.use_inductor is not None:
        forwarded.append("--use-inductor" if args.use_inductor else "--no-use-inductor")
    if args.pp_outer is not None:
        forwarded.append("--pp-outer" if args.pp_outer else "--no-pp-outer")
    if args.pytorch_profiler:
        forwarded.append("--pytorch-profiler")
    _append_value(forwarded, "--pytorch-profiler-iters", args.pytorch_profiler_iters)
    _append_value(forwarded, "--address", args.address)
    _append_value(forwarded, "--port", args.port)
    _append_value(forwarded, "--temp-dir", args.temp_dir)
    _append_value(forwarded, "--ray-namespace", args.ray_namespace)
    return forwarded


def _append_value(out: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        out.extend([flag, str(value)])


def _python_bin(repo_root: Path) -> str:
    if os.environ.get("PYTHON"):
        return os.environ["PYTHON"]

    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    return sys.executable


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
