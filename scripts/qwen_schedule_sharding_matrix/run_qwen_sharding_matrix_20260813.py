#!/usr/bin/env python3
"""Rerun the report-backed Qwen sharding matrix on a 16-GPU Ray cluster.

This runner deliberately keeps the benchmark invocation boring: three warmup
steps, five timed steps, no profiler, and no optional instrumentation.  It also
materializes every exact command before execution and is safe to resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_BASE_ROOT = ROOT.parent / "piper" / "examples" / "base-schedules"
OUTPUT_NAME = os.environ.get(
    "QWEN_FULL_MATRIX_OUTPUT_NAME", "20260813_qwen_sharding_matrix_rerun"
)
OUTPUT_ROOT = ROOT / "out" / OUTPUT_NAME
SCHEDULE_ROOT = OUTPUT_ROOT / "base_schedules"
CASE_ROOT = OUTPUT_ROOT / "cases"
HARNESS = ROOT / "examples" / "test_harness.py"
PYTHON = Path("/opt/piper-venv/bin/python")
MODEL = "Qwen3-30B-A3B-empirical1to1-L4"
RAY_HOST = os.environ.get("QWEN_FULL_MATRIX_RAY_HOST", "10.65.3.25")
RAY_PORT = int(os.environ.get("QWEN_FULL_MATRIX_RAY_PORT", "6379"))
SEQ_LEN = 12_288
BATCH_SIZE = 1
WARMUP = 3
ITERS = int(os.environ.get("QWEN_FULL_MATRIX_TIMED_ITERS", "5"))
MODEL_SEED = int(os.environ.get("QWEN_FULL_MATRIX_MODEL_SEED", "42"))
INPUT_SEED = int(os.environ.get("QWEN_FULL_MATRIX_INPUT_SEED", "20260814"))
USE_INDUCTOR = os.environ.get("QWEN_FULL_MATRIX_USE_INDUCTOR", "1") != "0"


@dataclass(frozen=True)
class Case:
    variant: str
    schedule: str
    sharding: str
    bucket_mb: int | None
    pp: int
    dp: int
    ep: int
    mbs: int

    @property
    def slug(self) -> str:
        bucket = "bucket_off" if self.bucket_mb is None else f"bucket_{self.bucket_mb}mb"
        return f"{self.variant}__{self.sharding}__{bucket}"

    @property
    def source_base_name(self) -> str:
        shard_suffix = {
            "none": "unsharded",
            "grad": "zero2",
            "param_grad": "zero3",
        }[self.sharding]
        if self.variant == "dualpipev":
            return f"pp4_dp8_ep8_v_placement_{shard_suffix}.json"
        if self.variant == "interleaved_1f1b":
            return f"pp4_dp8_ep8_interleaved_{shard_suffix}.json"
        if self.variant == "1f1b":
            stem = "pp2_dp8_ep8" if self.sharding == "none" else f"pp2_dp8_ep8_{shard_suffix}"
            return f"{stem}.json"
        if self.variant == "no_pp":
            return f"pp1_dp16_ep16_{shard_suffix}.json"
        raise ValueError(f"unknown variant {self.variant!r}")


def _cases() -> list[Case]:
    cases: list[Case] = []
    for variant, schedule in (
        ("dualpipev", "dualpipev"),
        ("1f1b", "1f1b"),
        ("interleaved_1f1b", "interleaved_1f1b"),
    ):
        for sharding in ("none", "grad", "param_grad"):
            for bucket_mb in (None, 128):
                cases.append(
                    Case(
                        variant=variant,
                        schedule=schedule,
                        sharding=sharding,
                        bucket_mb=bucket_mb,
                        pp=2,
                        dp=8,
                        ep=8,
                        mbs=4,
                    )
                )

    for sharding, buckets in (
        ("none", (None, 128)),
        ("grad", (None, 128)),
        ("param_grad", (None, 128)),
    ):
        for bucket_mb in buckets:
            cases.append(
                Case(
                    variant="no_pp",
                    schedule="1f1b",
                    sharding=sharding,
                    bucket_mb=bucket_mb,
                    pp=1,
                    dp=16,
                    ep=16,
                    mbs=1,
                )
            )
    assert len(cases) == 24
    return cases


def _materialize_base_schedule(case: Case) -> Path:
    source = SOURCE_BASE_ROOT / case.source_base_name
    if not source.is_file():
        raise FileNotFoundError(f"missing source base schedule: {source}")
    directives = json.loads(source.read_text(encoding="utf-8"))
    replicate_count = 0
    for directive in directives:
        if directive.get("op") != "replicate":
            continue
        replicate_count += 1
        if case.bucket_mb is None:
            directive.pop("bucket_size", None)
        else:
            directive["bucket_size"] = case.bucket_mb
    if replicate_count == 0:
        raise ValueError(f"no replicate directive in {source}")

    SCHEDULE_ROOT.mkdir(parents=True, exist_ok=True)
    destination = (SCHEDULE_ROOT / f"{case.slug}.json").resolve()
    destination.write_text(json.dumps(directives, indent=2) + "\n", encoding="utf-8")
    return destination


def _command(case: Case, base_schedule: Path) -> list[str]:
    namespace = f"qwen-matrix-20260813-{case.slug}".replace("_", "-")
    return [
        str(PYTHON),
        str(HARNESS.relative_to(ROOT)),
        "--test-file",
        "examples/test_qwen.py",
        "--base-schedule",
        str(base_schedule),
        "--schedule",
        case.schedule,
        "--ranks",
        str(case.pp),
        "--mbs",
        str(case.mbs),
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


def _shell_join(parts: list[str]) -> str:
    import shlex

    return shlex.join(parts)


def prepare(cases: list[Case]) -> dict[str, tuple[Path, list[str]]]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CASE_ROOT.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, tuple[Path, list[str]]] = {}
    command_lines = []
    manifest = []
    for case in cases:
        base_schedule = _materialize_base_schedule(case)
        command = _command(case, base_schedule)
        prepared[case.slug] = (base_schedule, command)
        command_lines.append(f"# {case.slug}\n{_shell_join(command)}")
        manifest.append(
            {
                **asdict(case),
                "slug": case.slug,
                "source_base_schedule": str(SOURCE_BASE_ROOT / case.source_base_name),
                "materialized_base_schedule": str(base_schedule),
                "command": command,
                "tokens_per_step": BATCH_SIZE * case.mbs * SEQ_LEN * case.dp,
            }
        )
    (OUTPUT_ROOT / "commands.txt").write_text(
        "\n\n".join(command_lines) + "\n", encoding="utf-8"
    )
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return prepared


def _find_run_dir(log_text: str) -> Path:
    matches = re.findall(r"metrics written to (.+?/results\.csv)", log_text)
    if not matches:
        raise RuntimeError("harness output did not identify results.csv")
    result_path = Path(matches[-1].strip())
    if not result_path.is_absolute():
        result_path = ROOT / result_path
    if not result_path.is_file():
        raise FileNotFoundError(f"harness-reported results do not exist: {result_path}")
    return result_path.parent.resolve()


def _aggregate(case: Case, raw_path: Path, run_dir: Path) -> dict[str, object]:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if len(raw) != case.dp:
        raise ValueError(f"{case.slug}: expected {case.dp} DP rows, got {len(raw)}")
    rows = sorted(raw, key=lambda row: int(row["dp_rank"]))
    sample_counts = {len(row["iter_times_s"]) for row in rows}
    if sample_counts != {ITERS}:
        raise ValueError(f"{case.slug}: expected {ITERS} samples/DP rank, got {sample_counts}")
    if {int(row["pp"]) for row in rows} != {case.pp}:
        raise ValueError(f"{case.slug}: raw metrics contain unexpected PP degree")
    if {int(row["dp"]) for row in rows} != {case.dp}:
        raise ValueError(f"{case.slug}: raw metrics contain unexpected DP degree")
    input_hashes = {str(row["input_data_sha256"]) for row in rows}
    if len(input_hashes) != 1:
        raise ValueError(f"{case.slug}: input hashes differ across DP ranks")
    if {int(row["model_seed"]) for row in rows} != {MODEL_SEED}:
        raise ValueError(f"{case.slug}: unexpected model seed")
    if {int(row["input_seed"]) for row in rows} != {INPUT_SEED}:
        raise ValueError(f"{case.slug}: unexpected input seed")

    # A distributed step is complete when its slowest DP driver completes.
    # Corresponding samples are synchronized, so take the max across DP ranks
    # for each iteration rather than averaging independent rank latencies.
    global_step_samples = [
        max(float(row["iter_times_s"][sample]) for row in rows)
        for sample in range(ITERS)
    ]
    mean_step = statistics.fmean(global_step_samples)
    std_step = statistics.stdev(global_step_samples)
    tokens_per_step = BATCH_SIZE * case.mbs * SEQ_LEN * case.dp
    throughput_samples = [tokens_per_step / sample for sample in global_step_samples]
    mean_throughput = tokens_per_step / mean_step
    throughput_se = mean_throughput * std_step / (mean_step * ITERS**0.5)

    peaks_by_pp = {pp_rank: 0 for pp_rank in range(case.pp)}
    for row in rows:
        memory_samples = row.get("peak_memory_samples_by_rank") or {}
        for global_rank_text, samples in memory_samples.items():
            physical_pp_rank = int(global_rank_text) % case.pp
            if samples:
                peaks_by_pp[physical_pp_rank] = max(
                    peaks_by_pp[physical_pp_rank], max(int(value) for value in samples)
                )

    summary: dict[str, object] = {
        "variant": case.variant,
        "sharding": case.sharding,
        "bucket_mb": "off" if case.bucket_mb is None else case.bucket_mb,
        "pp": case.pp,
        "dp": case.dp,
        "ep": case.ep,
        "microbatches": case.mbs,
        "warmup_iterations": WARMUP,
        "timed_iterations": ITERS,
        "use_inductor": USE_INDUCTOR,
        "tokens_per_step": tokens_per_step,
        "step_time_mean_s": mean_step,
        "step_time_std_s": std_step,
        "throughput_tokens_per_s": mean_throughput,
        "throughput_sample_std_tokens_per_s": statistics.stdev(throughput_samples),
        "throughput_95ci_halfwidth_tokens_per_s": 1.96 * throughput_se,
        "model_seed": MODEL_SEED,
        "input_seed": INPUT_SEED,
        "input_data_sha256": input_hashes.pop(),
        "run_dir": str(run_dir),
        "global_step_samples_s": json.dumps(global_step_samples),
    }
    for pp_rank, peak_bytes in peaks_by_pp.items():
        summary[f"peak_memory_pp{pp_rank}_gb"] = peak_bytes / 1024**3
    return summary


def _write_summaries(summaries: list[dict[str, object]]) -> None:
    if not summaries:
        return
    all_fields: list[str] = []
    for row in summaries:
        for field in row:
            if field not in all_fields:
                all_fields.append(field)
    with (OUTPUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(summaries)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )


def _existing_summary(case: Case) -> dict[str, object] | None:
    path = CASE_ROOT / case.slug / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def import_completed_cases(
    cases: list[Case],
    prepared: dict[str, tuple[Path, list[str]]],
    import_root: Path,
) -> int:
    imported = 0
    for case in cases:
        if _existing_summary(case) is not None:
            continue
        source_case = import_root / "cases" / case.slug
        source_raw = source_case / "raw_metrics.json"
        source_summary = source_case / "summary.json"
        if not source_raw.is_file() or not source_summary.is_file():
            continue

        source_metadata = json.loads(source_summary.read_text(encoding="utf-8"))
        run_dir = Path(source_metadata["run_dir"])
        case_dir = CASE_ROOT / case.slug
        case_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_raw, case_dir / "raw_metrics.json")
        source_results = source_case / "results.csv"
        if source_results.is_file():
            shutil.copy2(source_results, case_dir / "results.csv")
        source_log = source_case / "harness.log"
        if source_log.is_file():
            shutil.copy2(source_log, case_dir / "harness.log")
        for source_schedule in source_case.glob("*.json"):
            if source_schedule.name in {"raw_metrics.json", "summary.json"}:
                continue
            shutil.copy2(source_schedule, case_dir / source_schedule.name)
        _, command = prepared[case.slug]
        (case_dir / "command.txt").write_text(
            _shell_join(command) + "\n", encoding="utf-8"
        )
        summary = _aggregate(case, source_raw, run_dir)
        summary["imported_from"] = str(source_case.resolve())
        (case_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        imported += 1
    return imported


def run(cases: list[Case], prepared: dict[str, tuple[Path, list[str]]]) -> int:
    summaries: list[dict[str, object]] = []
    failures: list[str] = []
    for index, case in enumerate(cases, start=1):
        existing = _existing_summary(case)
        if existing is not None:
            summaries.append(existing)
            print(f"[{index:02d}/{len(cases)}] SKIP {case.slug} (already complete)", flush=True)
            continue

        _, command = prepared[case.slug]
        case_dir = CASE_ROOT / case.slug
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "command.txt").write_text(_shell_join(command) + "\n", encoding="utf-8")
        log_path = case_dir / "harness.log"
        print(f"[{index:02d}/{len(cases)}] START {case.slug}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode != 0:
            failures.append(case.slug)
            print(
                f"[{index:02d}/{len(cases)}] FAIL {case.slug} rc={completed.returncode}; "
                f"see {log_path}",
                flush=True,
            )
            continue

        try:
            run_dir = _find_run_dir(log_path.read_text(encoding="utf-8", errors="replace"))
            shutil.copy2(run_dir / "results.csv", case_dir / "results.csv")
            shutil.copy2(run_dir / "raw_metrics.json", case_dir / "raw_metrics.json")
            generated_schedules = [
                path
                for path in run_dir.glob("*.json")
                if path.name != "raw_metrics.json"
            ]
            for generated_schedule in generated_schedules:
                shutil.copy2(generated_schedule, case_dir / generated_schedule.name)
            summary = _aggregate(case, run_dir / "raw_metrics.json", run_dir)
            (case_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            summaries.append(summary)
            _write_summaries(summaries)
            peak_text = ", ".join(
                f"PP{rank}={summary[f'peak_memory_pp{rank}_gb']:.2f} GiB"
                for rank in range(case.pp)
            )
            print(
                f"[{index:02d}/{len(cases)}] DONE {case.slug}: "
                f"{summary['step_time_mean_s']:.4f} +/- {summary['step_time_std_s']:.4f} s, "
                f"{summary['throughput_tokens_per_s']:.1f} tok/s, {peak_text}",
                flush=True,
            )
        except Exception as error:  # Keep the expensive matrix resumable.
            failures.append(case.slug)
            print(
                f"[{index:02d}/{len(cases)}] POSTPROCESS FAIL {case.slug}: {error}",
                flush=True,
            )

    # Reassemble in canonical order, including successful results from earlier
    # resumed attempts.
    canonical = []
    for case in cases:
        summary = _existing_summary(case)
        if summary is not None:
            canonical.append(summary)
    _write_summaries(canonical)
    (OUTPUT_ROOT / "failures.json").write_text(
        json.dumps(failures, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"matrix complete: {len(canonical)}/{len(cases)} successful; "
        f"summary={OUTPUT_ROOT / 'summary.csv'}",
        flush=True,
    )
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="materialize schedules and exact commands without running benchmarks",
    )
    parser.add_argument(
        "--import-root",
        type=Path,
        help="Import matching completed case metrics from another matrix output.",
    )
    args = parser.parse_args()
    cases = _cases()
    prepared = prepare(cases)
    print(f"prepared {len(cases)} commands in {OUTPUT_ROOT / 'commands.txt'}", flush=True)
    if args.import_root is not None:
        imported = import_completed_cases(cases, prepared, args.import_root.resolve())
        print(f"imported {imported} completed case(s) from {args.import_root}", flush=True)
    if args.prepare_only:
        return 0
    return run(cases, prepared)


if __name__ == "__main__":
    sys.exit(main())
