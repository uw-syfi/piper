#!/usr/bin/env python3
"""Capture one active PyTorch-profiler iteration for three selected schedules."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX_ROOT = ROOT / "out" / "20260813_qwen_sharding_matrix_rerun"
OUTPUT_ROOT = ROOT / "out" / "20260813_qwen_selected_pytorch_profiles"
PYTHON = Path("/opt/piper-venv/bin/python")
CASES = (
    "dualpipev__none__bucket_off",
    "dualpipev__none__bucket_128mb",
    "dualpipev__grad__bucket_off",
)


def command(case: str) -> list[str]:
    return [
        str(PYTHON),
        "examples/test_harness.py",
        "--test-file",
        "examples/test_qwen.py",
        "--base-schedule",
        str((MATRIX_ROOT / "base_schedules" / f"{case}.json").resolve()),
        "--schedule",
        "dualpipev",
        "--ranks",
        "2",
        "--mbs",
        "4",
        "--address",
        "10.65.3.25",
        "--port",
        "6379",
        "--ray-namespace",
        f"qwen-profile-20260813-{case}".replace("_", "-"),
        "--pytorch-profiler",
        "--model",
        "Qwen3-30B-A3B-empirical1to1-L4",
        "--batch-size",
        "1",
        "--seq-len",
        "12288",
        "--warmup",
        "2",
        "--iters",
        "0",
        "--pytorch-profiler-warmup-iters",
        "2",
        "--pytorch-profiler-iters",
        "1",
        "--pp-outer",
    ]


def main() -> int:
    import shlex

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = []
    for index, case in enumerate(CASES, start=1):
        case_dir = OUTPUT_ROOT / case
        case_dir.mkdir(parents=True, exist_ok=True)
        done_path = case_dir / "trace_manifest.json"
        if done_path.is_file():
            print(f"[{index}/3] SKIP {case} (already complete)", flush=True)
            manifest.append(json.loads(done_path.read_text(encoding="utf-8")))
            continue
        cmd = command(case)
        (case_dir / "command.txt").write_text(
            shlex.join(cmd) + "\n", encoding="utf-8"
        )
        log_path = case_dir / "harness.log"
        print(f"[{index}/3] START {case}", flush=True)
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if result.returncode:
            failures.append(case)
            print(f"[{index}/3] FAIL {case}; see {log_path}", flush=True)
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"metrics written to (.+?/results\.csv)", log_text)
        if not matches:
            failures.append(case)
            print(f"[{index}/3] FAIL {case}; no run directory in log", flush=True)
            continue
        results_path = Path(matches[-1].strip())
        if not results_path.is_absolute():
            results_path = ROOT / results_path
        run_dir = results_path.resolve().parent
        traces = sorted(run_dir.glob("pytorch_profile_*.json"))
        if len(traces) != 8:
            failures.append(case)
            print(
                f"[{index}/3] FAIL {case}; expected 8 combined DP traces, got {len(traces)}",
                flush=True,
            )
            continue
        record = {
            "case": case,
            "run_dir": str(run_dir),
            "normal_warmup_iters": 2,
            "normal_timed_iters": 0,
            "profiler_warmup_iters": 2,
            "profiler_active_iters": 1,
            "traces": [str(path.resolve()) for path in traces],
            "trace_sizes_bytes": [path.stat().st_size for path in traces],
        }
        done_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        manifest.append(record)
        print(
            f"[{index}/3] DONE {case}: {len(traces)} traces in {run_dir}",
            flush=True,
        )

    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
