#!/usr/bin/env python3
"""Run the selected DualPipeV/interleaved Qwen matrix and plot throughput."""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE_ROOT = ROOT / "out" / "20260813_qwen_sharding_matrix_rerun" / "base_schedules"
OUTPUT_NAME = os.environ.get(
    "QWEN_MATRIX_OUTPUT_NAME",
    "20260814_qwen_dualpipev_interleaved_matrix_10_65_9_209",
)
OUTPUT_ROOT = ROOT / "out" / OUTPUT_NAME
CASE_ROOT = OUTPUT_ROOT / "cases"
PYTHON = Path("/opt/piper-venv/bin/python")
HARNESS = ROOT / "examples" / "test_harness.py"
MODEL = "Qwen3-30B-A3B-empirical1to1-L4"
RAY_HOST = os.environ.get("QWEN_MATRIX_RAY_HOST", "10.65.9.209")
RAY_PORT = int(os.environ.get("QWEN_MATRIX_RAY_PORT", "6379"))
SEQ_LEN = 12_288
BATCH_SIZE = 1
PP = 2
DP = 8
EP = 8
MICROBATCHES = 4
WARMUP = 3
ITERS = int(os.environ.get("QWEN_MATRIX_TIMED_ITERS", "5"))
ERROR_BAR_MODE = os.environ.get("QWEN_MATRIX_ERROR_BAR_MODE", "sample_sd")
USE_INDUCTOR = os.environ.get("QWEN_MATRIX_USE_INDUCTOR", "1") != "0"
MODEL_SEED = 42
INPUT_SEED = 20_260_814
TOKENS_PER_STEP = BATCH_SIZE * MICROBATCHES * SEQ_LEN * DP


@dataclass(frozen=True)
class Case:
    schedule: str
    sharding: str
    bucket_mb: int | None

    @property
    def slug(self) -> str:
        bucket = "bucket_off" if self.bucket_mb is None else f"bucket_{self.bucket_mb}mb"
        return f"{self.schedule}__{self.sharding}__{bucket}"

    @property
    def base_schedule(self) -> Path:
        return (BASE_ROOT / f"{self.slug}.json").resolve()


CASES = [
    Case(schedule, sharding, bucket)
    for sharding, bucket in (
        ("none", None),
        ("none", 128),
        ("grad", None),
        ("param_grad", None),
    )
    for schedule in ("dualpipev", "interleaved_1f1b")
]


def shell_join(parts: list[str]) -> str:
    import shlex

    return shlex.join(parts)


def command(case: Case) -> list[str]:
    namespace = f"qwen-selected-20260814-n{ITERS}-{case.slug}".replace("_", "-")
    return [
        str(PYTHON),
        str(HARNESS.relative_to(ROOT)),
        "--test-file",
        "examples/test_qwen.py",
        "--base-schedule",
        str(case.base_schedule),
        "--schedule",
        case.schedule,
        "--ranks",
        str(PP),
        "--mbs",
        str(MICROBATCHES),
        "--address",
        RAY_HOST,
        "--port",
        str(RAY_PORT),
        "--ray-namespace",
        namespace,
        "--model",
        MODEL,
        "--batch-size",
        str(BATCH_SIZE),
        "--seq-len",
        str(SEQ_LEN),
        "--warmup",
        str(WARMUP),
        "--iters",
        str(ITERS),
        "--model-seed",
        str(MODEL_SEED),
        "--input-seed",
        str(INPUT_SEED),
        "--pp-outer",
        "--use-inductor" if USE_INDUCTOR else "--no-use-inductor",
    ]


def find_run_dir(log_text: str) -> Path:
    matches = re.findall(r"metrics written to (.+?/results\.csv)", log_text)
    if not matches:
        raise RuntimeError("harness output did not identify results.csv")
    result_path = Path(matches[-1].strip())
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    return result_path.parent.resolve()


def aggregate(case: Case, raw_path: Path, run_dir: Path) -> dict[str, object]:
    rows = sorted(
        json.loads(raw_path.read_text(encoding="utf-8")),
        key=lambda row: int(row["dp_rank"]),
    )
    if len(rows) != DP:
        raise ValueError(f"expected {DP} DP rows, got {len(rows)}")
    if {len(row["iter_times_s"]) for row in rows} != {ITERS}:
        raise ValueError(f"expected {ITERS} samples from every DP rank")
    if {int(row["pp"]) for row in rows} != {PP} or {int(row["dp"]) for row in rows} != {DP}:
        raise ValueError("raw metrics contain unexpected PP/DP degrees")

    fingerprints = {str(row["input_data_sha256"]) for row in rows}
    input_seeds = {int(row["input_seed"]) for row in rows}
    model_seeds = {int(row["model_seed"]) for row in rows}
    if len(fingerprints) != 1 or input_seeds != {INPUT_SEED} or model_seeds != {MODEL_SEED}:
        raise ValueError("input/model identity differs across DP ranks")

    # The latency of one distributed step is the slowest corresponding DP rank.
    global_steps = [
        max(float(row["iter_times_s"][sample]) for row in rows)
        for sample in range(ITERS)
    ]
    step_mean = statistics.fmean(global_steps)
    step_std = statistics.stdev(global_steps)
    throughput = TOKENS_PER_STEP / step_mean
    throughput_samples = [TOKENS_PER_STEP / value for value in global_steps]
    throughput_delta_method_se = throughput * step_std / (step_mean * ITERS**0.5)

    return {
        "schedule": case.schedule,
        "sharding": case.sharding,
        "bucket_mb": "off" if case.bucket_mb is None else case.bucket_mb,
        "pp": PP,
        "dp": DP,
        "ep": EP,
        "microbatches": MICROBATCHES,
        "warmup_iterations": WARMUP,
        "timed_iterations": ITERS,
        "use_inductor": USE_INDUCTOR,
        "tokens_per_step": TOKENS_PER_STEP,
        "step_time_mean_s": step_mean,
        "step_time_sample_std_s": step_std,
        "throughput_tokens_per_s": throughput,
        "throughput_sample_std_tokens_per_s": statistics.stdev(throughput_samples),
        "throughput_delta_method_se_tokens_per_s": throughput_delta_method_se,
        "throughput_95ci_halfwidth_tokens_per_s": 1.96 * throughput_delta_method_se,
        "global_step_samples_s": global_steps,
        "model_seed": MODEL_SEED,
        "input_seed": INPUT_SEED,
        "input_data_sha256": fingerprints.pop(),
        "run_dir": str(run_dir),
    }


def existing_summary(case: Case) -> dict[str, object] | None:
    path = CASE_ROOT / case.slug / "summary.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def write_summaries(summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    scalar_rows = [
        {key: value for key, value in row.items() if key != "global_step_samples_s"}
        | {"global_step_samples_s": json.dumps(row["global_step_samples_s"])}
        for row in summaries
    ]
    with (OUTPUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def plot(summaries: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    group_specs = [
        ("none", "off", "No sharding\nNo bucket"),
        ("none", 128, "No sharding\n128 MB"),
        ("grad", "off", "Grad sharding\nNo bucket"),
        ("param_grad", "off", "Param + grad\nNo bucket"),
    ]
    by_key = {(row["schedule"], row["sharding"], row["bucket_mb"]): row for row in summaries}
    sample_counts = {int(row["timed_iterations"]) for row in summaries}
    if len(sample_counts) != 1:
        raise ValueError(f"plot inputs have inconsistent sample counts: {sample_counts}")
    sample_count = sample_counts.pop()
    x = np.arange(len(group_specs), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    styles = [
        ("dualpipev", "DualPipeV", "#2563eb", ""),
        ("interleaved_1f1b", "Interleaved 1F1B", "#f59e0b", "//"),
    ]
    for index, (schedule, label, color, hatch) in enumerate(styles):
        rows = [by_key[(schedule, sharding, bucket)] for sharding, bucket, _ in group_specs]
        values = [float(row["throughput_tokens_per_s"]) / 1000 for row in rows]
        if ERROR_BAR_MODE == "sample_sd":
            errors = [float(row["throughput_sample_std_tokens_per_s"]) / 1000 for row in rows]
        elif ERROR_BAR_MODE == "ci95":
            errors = [float(row["throughput_95ci_halfwidth_tokens_per_s"]) / 1000 for row in rows]
        else:
            raise ValueError(f"unknown error-bar mode: {ERROR_BAR_MODE!r}")
        positions = x + (index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            yerr=errors,
            capsize=4,
            label=label,
            color=color,
            hatch=hatch,
            edgecolor="#1f2937",
            linewidth=0.7,
            error_kw={
                "ecolor": "#111827",
                "elinewidth": 1.3,
                "capthick": 1.3,
                "zorder": 4,
            },
        )
        ax.bar_label(bars, labels=[f"{value:.0f}k" for value in values], padding=5, fontsize=9)

    ax.set_xticks(x, [label for _, _, label in group_specs])
    ax.set_ylabel("Throughput (thousand tokens/s)")
    ax.set_xlabel("(Sharding, bucketing) configuration")
    execution_mode = "Inductor" if USE_INDUCTOR else "No Inductor"
    ax.set_title(
        f"Qwen3 30B-L4, 12K sequence — PP=2, DP/EP=8 — {execution_mode}"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    ax.legend(frameon=False, ncols=2, loc="upper center")
    error_label = (
        "sample SD of per-step throughput"
        if ERROR_BAR_MODE == "sample_sd"
        else "approximate 95% CI for throughput (delta method)"
    )
    ax.text(
        0.5,
        -0.17,
        f"Bars: {TOKENS_PER_STEP:,} tokens / mean distributed step time; "
        f"error bars: {error_label} (n={sample_count})",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        color="#4b5563",
    )
    fig.savefig(OUTPUT_ROOT / "throughput_by_shard_bucket.png", dpi=200)
    fig.savefig(OUTPUT_ROOT / "throughput_by_shard_bucket.svg")
    plt.close(fig)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        if not case.base_schedule.is_file():
            raise FileNotFoundError(case.base_schedule)

    commands = [f"# {case.slug}\n{shell_join(command(case))}" for case in CASES]
    (OUTPUT_ROOT / "commands.txt").write_text("\n\n".join(commands) + "\n", encoding="utf-8")

    failures: list[str] = []
    for index, case in enumerate(CASES, start=1):
        if existing_summary(case) is not None:
            print(f"[{index}/8] SKIP {case.slug} (complete)", flush=True)
            continue
        case_dir = CASE_ROOT / case.slug
        case_dir.mkdir(parents=True, exist_ok=True)
        cmd = command(case)
        (case_dir / "command.txt").write_text(shell_join(cmd) + "\n", encoding="utf-8")
        log_path = case_dir / "harness.log"
        print(f"[{index}/8] START {case.slug}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            failures.append(case.slug)
            print(f"[{index}/8] FAIL {case.slug} rc={completed.returncode}; {log_path}", flush=True)
            continue
        try:
            run_dir = find_run_dir(log_path.read_text(encoding="utf-8", errors="replace"))
            shutil.copy2(run_dir / "results.csv", case_dir / "results.csv")
            shutil.copy2(run_dir / "raw_metrics.json", case_dir / "raw_metrics.json")
            for schedule_path in run_dir.glob("*.json"):
                if schedule_path.name != "raw_metrics.json":
                    shutil.copy2(schedule_path, case_dir / schedule_path.name)
            summary = aggregate(case, run_dir / "raw_metrics.json", run_dir)
            (case_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[{index}/8] DONE {case.slug}: "
                f"{summary['step_time_mean_s']:.4f} +/- {summary['step_time_sample_std_s']:.4f} s, "
                f"{summary['throughput_tokens_per_s']:.1f} tok/s",
                flush=True,
            )
        except Exception as error:
            failures.append(case.slug)
            print(f"[{index}/8] POSTPROCESS FAIL {case.slug}: {error}", flush=True)

        completed_summaries = [summary for item in CASES if (summary := existing_summary(item))]
        write_summaries(completed_summaries)

    summaries = [summary for case in CASES if (summary := existing_summary(case))]
    write_summaries(summaries)
    (OUTPUT_ROOT / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    if len(summaries) == len(CASES):
        fingerprints = {str(row["input_data_sha256"]) for row in summaries}
        if len(fingerprints) != 1:
            raise ValueError(f"inputs differ across cases: {fingerprints}")
        plot(summaries)
        print(f"chart={OUTPUT_ROOT / 'throughput_by_shard_bucket.png'}", flush=True)
    print(f"complete: {len(summaries)}/{len(CASES)} successful", flush=True)
    return 1 if failures or len(summaries) != len(CASES) else 0


if __name__ == "__main__":
    sys.exit(main())
