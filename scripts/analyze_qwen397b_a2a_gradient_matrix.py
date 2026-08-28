#!/usr/bin/env python3
"""Summarize the Qwen-397B A2A x gradient-collective matrix.

The benchmark JSON already contains sample-aligned, world-critical summaries.
This script turns those summaries into a flat CSV and a compact JSON document,
with A2A critical-path delay as the primary metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


CONCURRENT_MODES = {
    "concurrent_a2a_first": "a2a_first",
    "concurrent_allreduce_first": "gradient_first",
}


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def metric_median(world: dict[str, Any], metric: str) -> float:
    return float(world[metric]["median"])


def signed_start_offsets_ms(
    rank_results: list[dict[str, Any]], mode: str
) -> list[float]:
    """Positive means that A2A started after the gradient collective."""
    return [
        float(sample["a2a_gate_to_start_ms"])
        - float(sample["allreduce_gate_to_start_ms"])
        for result in rank_results
        for sample in result["samples"][mode]
    ]


def validation_ok(rank_results: list[dict[str, Any]]) -> bool:
    for result in rank_results:
        validation = result["validation"]
        if validation["a2a_observed"] != validation["a2a_expected"]:
            return False
        expected = float(validation["allreduce_expected"])
        if any(
            abs(float(observed) - expected) > 1e-5
            for observed in validation["allreduce_observed"].values()
        ):
            return False
    return True


def load_cases(input_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    cases = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "matrix_manifest.json":
            continue
        document = json.loads(path.read_text())
        if "rank_results" not in document or "aggregate" not in document:
            continue
        cases.append((path, document))
    return cases


def make_rows(cases: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, document in cases:
        args = document["command_args"]
        buckets = [float(value) for value in args["gradient_bucket_mib"].split(",")]
        collective = args["gradient_collective"]
        rank_results = document["rank_results"]
        phase = path.stem.split("_", 1)[1].split("_", 1)[0]
        if path.stem.startswith("ar_pair_"):
            phase = "two_bucket"
        elif path.stem.startswith("ar_large_"):
            phase = "large"
        elif path.stem.startswith("ar_depth_"):
            phase = "depth"
        elif path.stem.startswith("ar_offset_"):
            phase = "offset"
        elif path.stem.startswith("rs_"):
            phase = "reduce_scatter"

        for mode, launch_order in CONCURRENT_MODES.items():
            world = document["aggregate"]["derived"][mode]["world_critical"]
            offsets = signed_start_offsets_ms(rank_results, mode)
            solo_a2a = metric_median(world, "a2a_solo_ms")
            solo_gradient = metric_median(world, "allreduce_solo_ms")
            a2a_delay = metric_median(world, "a2a_critical_delay_ms")
            a2a_slowdown = metric_median(world, "a2a_slowdown")
            gradient_slowdown = metric_median(world, "allreduce_slowdown")
            rows.append(
                {
                    "phase": phase,
                    "case": path.stem,
                    "gradient_collective": collective,
                    "gradient_buckets_mib": ",".join(f"{x:g}" for x in buckets),
                    "first_bucket_mib": buckets[0],
                    "bucket_count": len(buckets),
                    "total_gradient_mib": sum(buckets),
                    "launch_order": launch_order,
                    "configured_a2a_delay_cycles": args["a2a_delay_cycles"],
                    "configured_gradient_delay_cycles": args[
                        "gradient_delay_cycles"
                    ],
                    "actual_a2a_minus_gradient_start_ms_median": median(offsets),
                    "solo_a2a_ms": solo_a2a,
                    "concurrent_a2a_ms": solo_a2a + a2a_delay,
                    "a2a_critical_delay_ms": a2a_delay,
                    "a2a_slowdown_x": a2a_slowdown,
                    "a2a_delay_fraction_of_concurrent": (
                        a2a_delay / (solo_a2a + a2a_delay)
                    ),
                    "solo_gradient_ms": solo_gradient,
                    "concurrent_gradient_ms": solo_gradient * gradient_slowdown,
                    "gradient_slowdown_x": gradient_slowdown,
                    "normal_pair_ms": metric_median(world, "normal_pair_ms"),
                    "ideal_pair_ms": metric_median(
                        world, "ideal_isolated_pair_ms"
                    ),
                    "pair_interference_removal_speedup_x": metric_median(
                        world, "interference_removal_speedup_ceiling"
                    ),
                    "valid": validation_ok(rank_results),
                    "rank_count": len(rank_results),
                    "samples_per_mode_per_rank": len(
                        rank_results[0]["samples"][mode]
                    ),
                    "source": str(path),
                }
            )
    return rows


def add_queue_predictions(rows: list[dict[str, Any]]) -> None:
    single_solo: dict[tuple[str, float], float] = {}
    for row in rows:
        if (
            row["bucket_count"] == 1
            and row["configured_a2a_delay_cycles"] == 0
            and row["configured_gradient_delay_cycles"] == 0
        ):
            single_solo[
                (row["gradient_collective"], row["first_bucket_mib"])
            ] = row["solo_gradient_ms"]

    for row in rows:
        buckets = [float(value) for value in row["gradient_buckets_mib"].split(",")]
        prefix_references = [
            single_solo.get((row["gradient_collective"], bucket))
            for bucket in buckets[:-1]
        ]
        if (
            row["launch_order"] == "gradient_first"
            and row["bucket_count"] > 1
            and row["configured_a2a_delay_cycles"] == 0
            and row["configured_gradient_delay_cycles"] == 0
            and all(value is not None for value in prefix_references)
        ):
            # NCCL overlaps A2A with the final queued gradient collective, so
            # the device-side queue delay is approximately the sum of every
            # earlier bucket. For two buckets this reduces to the first-bucket
            # rule observed in the original experiment.
            reference = sum(float(value) for value in prefix_references)
            row["gradient_prefix_queue_prediction_ms"] = reference
            row["queue_prediction_error_ms"] = (
                row["a2a_critical_delay_ms"] - reference
            )
            row["queue_prediction_relative_error"] = (
                row["queue_prediction_error_ms"] / reference
            )
        else:
            row["gradient_prefix_queue_prediction_ms"] = ""
            row["queue_prediction_error_ms"] = ""
            row["queue_prediction_relative_error"] = ""


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gradient_first = [
        row
        for row in rows
        if row["launch_order"] == "gradient_first"
        and row["configured_a2a_delay_cycles"] == 0
        and row["configured_gradient_delay_cycles"] == 0
    ]
    a2a_first = [
        row
        for row in rows
        if row["launch_order"] == "a2a_first"
        and row["configured_a2a_delay_cycles"] == 0
        and row["configured_gradient_delay_cycles"] == 0
    ]
    predicted = [
        row for row in rows if row["queue_prediction_relative_error"] != ""
    ]
    predicted_x = [
        float(row["gradient_prefix_queue_prediction_ms"]) for row in predicted
    ]
    measured_y = [float(row["a2a_critical_delay_ms"]) for row in predicted]
    mean_x = statistics.mean(predicted_x)
    mean_y = statistics.mean(measured_y)
    covariance = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(predicted_x, measured_y, strict=True)
    )
    variance_x = sum((x - mean_x) ** 2 for x in predicted_x)
    variance_y = sum((y - mean_y) ** 2 for y in measured_y)
    slope = covariance / variance_x
    intercept = mean_y - slope * mean_x
    r_squared = covariance**2 / (variance_x * variance_y)
    absolute_relative_errors = [
        abs(row["queue_prediction_relative_error"]) for row in predicted
    ]
    return {
        "case_count": len({row["case"] for row in rows}),
        "row_count": len(rows),
        "all_valid": all(row["valid"] for row in rows),
        "gradient_first": {
            "max_a2a_critical_delay_case": max(
                gradient_first,
                key=lambda row: row["a2a_critical_delay_ms"],
            ),
            "max_a2a_slowdown_case": max(
                gradient_first, key=lambda row: row["a2a_slowdown_x"]
            ),
        },
        "a2a_first": {
            "max_a2a_critical_delay_case": max(
                a2a_first, key=lambda row: row["a2a_critical_delay_ms"]
            ),
            "median_a2a_critical_delay_ms": median(
                [row["a2a_critical_delay_ms"] for row in a2a_first]
            ),
        },
        "queue_prediction": {
            "case_count": len(predicted),
            "slope": slope,
            "intercept_ms": intercept,
            "r_squared": r_squared,
            "median_absolute_relative_error": median(absolute_relative_errors),
            "p95_absolute_relative_error": percentile(
                absolute_relative_errors, 0.95
            ),
            "max_absolute_relative_error": max(absolute_relative_errors),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "out/interference_benchmarks/"
            "qwen397b_a2a_gradient_matrix/ep8_dp2"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "out/interference_benchmarks/"
            "qwen397b_a2a_gradient_matrix/ep8_dp2/analysis"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = load_cases(args.input_dir)
    rows = make_rows(cases)
    add_queue_predictions(rows)
    summary = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "case_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.output_dir / "analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"Wrote {len(rows)} rows for {summary['case_count']} cases to {csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
