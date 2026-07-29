#!/usr/bin/env python3
"""Separate NCCL launch/admission delay from kernel execution time in traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return (
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "median": math.nan,
            "mean": math.nan,
            "p10": math.nan,
            "p90": math.nan,
            "min": math.nan,
            "max": math.nan,
        }
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "min": min(values),
        "max": max(values),
    }


def classify_kernel(event: dict[str, Any]) -> str | None:
    name = event.get("name", "")
    args = event.get("args", {})
    collective = str(args.get("Collective name", "")).lower()
    if collective in {"all_to_all", "all_to_allv", "alltoall", "alltoallv"}:
        return "a2a"
    if "allgather" in collective or "AllGather" in name:
        return "allgather"
    if "SendRecv" in name:
        return "p2p"
    return None


def rank_from_path(path: Path) -> int:
    match = re.search(r"_rank(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot resolve rank from {path}")
    return int(match.group(1))


def enclosing_mode(
    timestamp: float, windows: list[dict[str, Any]]
) -> str | None:
    for window in windows:
        if window["ts"] <= timestamp <= window["ts"] + window["dur"]:
            return window["name"].split("::", 1)[1]
    return None


def trace_rows(path: Path, case_name: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload["traceEvents"] if isinstance(payload, dict) else payload
    windows = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and str(event.get("name", "")).startswith("benchmark_mode::")
    ]
    launches: dict[int, dict[str, Any]] = {}
    for event in events:
        correlation = event.get("args", {}).get("correlation")
        if correlation is None:
            continue
        if event.get("cat") not in {"cuda_driver", "cuda_runtime"}:
            continue
        if "LaunchKernel" not in str(event.get("name", "")):
            continue
        launches[int(correlation)] = event

    rows: list[dict[str, Any]] = []
    ordinal_by_mode_op: dict[tuple[str, str], int] = {}
    for event in sorted(events, key=lambda item: item.get("ts", 0.0)):
        if event.get("cat") != "kernel":
            continue
        operation = classify_kernel(event)
        if operation is None:
            continue
        mode = enclosing_mode(float(event["ts"]), windows)
        if mode is None:
            continue
        key = (mode, operation)
        ordinal = ordinal_by_mode_op.get(key, 0)
        ordinal_by_mode_op[key] = ordinal + 1
        correlation = int(event.get("args", {}).get("correlation", -1))
        launch = launches.get(correlation)
        launch_to_start_us = math.nan
        launch_end_us = math.nan
        if launch is not None:
            launch_end_us = float(launch["ts"]) + float(launch.get("dur", 0.0))
            launch_to_start_us = float(event["ts"]) - launch_end_us
        rows.append(
            {
                "case": case_name,
                "trace": str(path),
                "rank": rank_from_path(path),
                "mode": mode,
                "operation": operation,
                "ordinal": ordinal,
                "kernel_name": event["name"],
                "collective_name": event.get("args", {}).get(
                    "Collective name", ""
                ),
                "in_msg_nelems": event.get("args", {}).get(
                    "In msg nelems", ""
                ),
                "out_msg_nelems": event.get("args", {}).get(
                    "Out msg nelems", ""
                ),
                "kernel_start_us": float(event["ts"]),
                "kernel_duration_us": float(event["dur"]),
                "launch_end_us": launch_end_us,
                "launch_to_start_us": launch_to_start_us,
            }
        )

    mode_names = {row["mode"] for row in rows}
    solo_mode = {
        "a2a": "a2a_solo",
        "p2p": "p2p_solo",
        "allgather": (
            "allgather_solo"
            if "allgather_solo" in mode_names
            else "allreduce_solo"
        ),
    }
    baselines = {
        (row["operation"], row["ordinal"]): row
        for row in rows
        if row["mode"] == solo_mode[row["operation"]]
    }
    for row in rows:
        baseline = baselines.get((row["operation"], row["ordinal"]))
        if baseline is None:
            row["baseline_kernel_duration_us"] = math.nan
            row["kernel_duration_ratio"] = math.nan
            row["baseline_launch_to_start_us"] = math.nan
            row["extra_launch_to_start_us"] = math.nan
            continue
        row["baseline_kernel_duration_us"] = baseline["kernel_duration_us"]
        row["kernel_duration_ratio"] = (
            row["kernel_duration_us"] / baseline["kernel_duration_us"]
        )
        row["baseline_launch_to_start_us"] = baseline["launch_to_start_us"]
        row["extra_launch_to_start_us"] = (
            row["launch_to_start_us"] - baseline["launch_to_start_us"]
        )
    return rows


def group_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["case"],
            row["mode"],
            row["operation"],
            row["ordinal"],
            row["out_msg_nelems"],
        )
        grouped.setdefault(key, []).append(row)
    output = []
    for key, group in sorted(grouped.items()):
        result = {
            "case": key[0],
            "mode": key[1],
            "operation": key[2],
            "ordinal": key[3],
            "out_msg_nelems": key[4],
            "rank_count": len(group),
        }
        for metric in (
            "kernel_duration_us",
            "kernel_duration_ratio",
            "launch_to_start_us",
            "extra_launch_to_start_us",
        ):
            values = [
                float(row[metric])
                for row in group
                if not math.isnan(float(row[metric]))
            ]
            summary = summarize(values)
            for statistic, value in summary.items():
                result[f"{metric}_{statistic}"] = value
        output.append(result)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="Case name and trace glob; may be repeated",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for item in args.case:
        name, separator, pattern = item.partition("=")
        if not separator:
            raise ValueError(f"Expected NAME=GLOB, received {item!r}")
        paths = sorted(Path().glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        for path in paths:
            rows.extend(trace_rows(path, name))
    summaries = group_summary(rows)
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "kernel_events.csv", rows)
    write_csv(output_dir / "mechanism_summary.csv", summaries)
    (output_dir / "mechanism_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} kernel rows and {len(summaries)} summaries")


if __name__ == "__main__":
    main()
