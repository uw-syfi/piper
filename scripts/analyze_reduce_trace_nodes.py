#!/usr/bin/env python3
"""Compare reduction-node profiles with their PyTorch trace kernels."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_UIDS = (
    "reduce.173",
    "reduce.179",
    "reduce.176",
    "reduce.182",
    "reduce.95",
    "reduce.71",
    "reduce.119",
    "reduce.47",
    "reduce.68",
    "reduce.170",
    "reduce.44",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--uids", nargs="+", default=list(DEFAULT_UIDS))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _repeated_period(values: list[tuple[int, str]]) -> int:
    for period in range(1, len(values) + 1):
        if len(values) % period:
            continue
        if all(value == values[index % period] for index, value in enumerate(values)):
            return period
    raise AssertionError("the complete sequence is always a valid period")


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _find_node(value: Any, uid: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("uid") == uid:
            return value
        for child in value.values():
            found = _find_node(child, uid)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_node(child, uid)
            if found is not None:
                return found
    return None


def _profile_summary(node: dict[str, Any], mode: str) -> dict[str, Any]:
    by_rank = node["profile_times_by_mode"][mode]
    rank_estimates = {
        rank: float(profile["estimate_us"]) for rank, profile in by_rank.items()
    }
    return {
        "rank_estimates_us": rank_estimates,
        "min_rank_estimate_us": min(rank_estimates.values()),
        "median_rank_estimate_us": statistics.median(rank_estimates.values()),
        "max_rank_estimate_us": max(rank_estimates.values()),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    wanted = set(args.uids)

    ratio_rows: dict[str, dict[str, str]] = {}
    with (output_dir / "node_profile_time_ratios.csv").open(
        newline="", encoding="utf-8"
    ) as ratio_file:
        for row in csv.DictReader(ratio_file):
            if row["node_uid"] in wanted:
                ratio_rows[row["node_uid"]] = row

    with (output_dir / "step_time_profiles.json").open(encoding="utf-8") as file:
        step_profiles = json.load(file)

    grouped_kernels: dict[
        tuple[str, Any, str], list[dict[str, Any]]
    ] = defaultdict(list)
    kernels_by_source: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    trace_paths = sorted(output_dir.glob("pytorch_profile_*_dprank*.json"))
    for trace_path in trace_paths:
        with trace_path.open(encoding="utf-8") as trace_file:
            events = json.load(trace_file)["traceEvents"]
        for event in events:
            if event.get("ph") != "X" or event.get("cat") != "kernel":
                continue
            args_dict = event.get("args") or {}
            label = args_dict.get("piper_dag_node")
            if not isinstance(label, str):
                continue
            source = (trace_path.name, event.get("pid"))
            kernels_by_source[source].append(event)
            for uid in wanted:
                if label.endswith(f"uid{uid}"):
                    grouped_kernels[(trace_path.name, event.get("pid"), uid)].append(
                        event
                    )
                    break

    summaries: list[dict[str, Any]] = []
    for uid in args.uids:
        node = _find_node(step_profiles.get("per_pp_dags"), uid)
        if node is None:
            raise SystemExit(f"Could not find {uid} in step_time_profiles.json")
        occurrences: list[dict[str, Any]] = []
        signatures: set[tuple[tuple[int, str], ...]] = set()
        for (trace_name, pid, grouped_uid), kernels in grouped_kernels.items():
            if grouped_uid != uid:
                continue
            kernels.sort(key=lambda event: float(event["ts"]))
            sequence = [
                (
                    int((event.get("args") or {})["In msg nelems"]),
                    str((event.get("args") or {}).get("dtype")),
                )
                for event in kernels
            ]
            period = _repeated_period(sequence)
            signature = tuple(sequence[:period])
            signatures.add(signature)
            source_kernels = kernels_by_source[(trace_name, pid)]
            for occurrence_index in range(0, len(kernels), period):
                chunk = kernels[occurrence_index : occurrence_index + period]
                start_us = float(chunk[0]["ts"])
                end_us = float(chunk[-1]["ts"]) + float(chunk[-1]["dur"])
                own_label = (chunk[0].get("args") or {})["piper_dag_node"]
                intersections = []
                overlapping_labels: set[str] = set()
                for other in source_kernels:
                    other_label = str(
                        (other.get("args") or {}).get("piper_dag_node", "")
                    )
                    if other_label == own_label:
                        continue
                    other_start = float(other["ts"])
                    other_end = other_start + float(other["dur"])
                    intersection_start = max(start_us, other_start)
                    intersection_end = min(end_us, other_end)
                    if intersection_end > intersection_start:
                        intersections.append((intersection_start, intersection_end))
                        if other_label:
                            overlapping_labels.add(other_label.split(":", 1)[0])
                kernel_sum_us = sum(float(event["dur"]) for event in chunk)
                span_us = end_us - start_us
                occurrences.append(
                    {
                        "trace": trace_name,
                        "pid": pid,
                        "index": occurrence_index // period,
                        "kernel_sum_us": kernel_sum_us,
                        "span_us": span_us,
                        "gap_us": span_us - kernel_sum_us,
                        "other_kernel_overlap_union_us": _union_duration(intersections),
                        "overlapping_node_types": sorted(overlapping_labels),
                    }
                )
        if len(signatures) != 1:
            raise SystemExit(f"Expected one payload signature for {uid}, got {signatures}")
        signature = next(iter(signatures))
        ratio = ratio_rows[uid]
        summaries.append(
            {
                "uid": uid,
                "tag": node.get("tag"),
                "participants": node.get("node_meta", {}).get("participants"),
                "payload_signature": [
                    {"numel": numel, "dtype": dtype} for numel, dtype in signature
                ],
                "payload_bytes": sum(
                    numel * (2 if dtype == "BFloat16" else 0)
                    for numel, dtype in signature
                ),
                "csv": {
                    "normal_us": float(ratio["median_normal_time_us"]),
                    "isolated_us": float(
                        ratio["median_isolated_time_us"]
                    ),
                    "ratio": float(ratio["normal_to_isolated_ratio"]),
                },
                "profiles": {
                    mode: _profile_summary(node, mode)
                    for mode in ("normal", "isolated")
                },
                "trace": {
                    "occurrence_count": len(occurrences),
                    "median_kernel_sum_us": _median(
                        [value["kernel_sum_us"] for value in occurrences]
                    ),
                    "median_span_us": _median(
                        [value["span_us"] for value in occurrences]
                    ),
                    "median_gap_us": _median(
                        [value["gap_us"] for value in occurrences]
                    ),
                    "median_other_kernel_overlap_union_us": _median(
                        [
                            value["other_kernel_overlap_union_us"]
                            for value in occurrences
                        ]
                    ),
                    "min_span_us": min(value["span_us"] for value in occurrences),
                    "max_span_us": max(value["span_us"] for value in occurrences),
                    "overlapping_node_types": sorted(
                        {
                            node_type
                            for occurrence in occurrences
                            for node_type in occurrence["overlapping_node_types"]
                        }
                    ),
                    "occurrences": occurrences,
                },
            }
        )

    output_path = args.output or output_dir / "selected_reduce_trace_analysis.json"
    output_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
