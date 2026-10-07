#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Summarize NCCL kernel spans in isolated reduction-bucket traces."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--kernels-per-iteration", type=int, default=9)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "maximum": max(values),
    }


def main() -> None:
    args = parse_args()
    traces = sorted(args.trace_dir.glob("isolated_reduce_bucket_rank*.json"))
    if not traces:
        raise SystemExit(f"No isolated traces found in {args.trace_dir}")

    occurrences: list[dict[str, Any]] = []
    for trace_path in traces:
        with trace_path.open(encoding="utf-8") as trace_file:
            events = json.load(trace_file)["traceEvents"]
        kernels = sorted(
            (
                event
                for event in events
                if event.get("ph") == "X"
                and event.get("cat") == "kernel"
                and str(event.get("name", "")).startswith("ncclDevKernel_AllReduce")
            ),
            key=lambda event: float(event["ts"]),
        )
        if len(kernels) % args.kernels_per_iteration:
            raise SystemExit(
                f"{trace_path} has {len(kernels)} NCCL kernels, not divisible by "
                f"{args.kernels_per_iteration}"
            )
        for start_index in range(0, len(kernels), args.kernels_per_iteration):
            chunk = kernels[start_index : start_index + args.kernels_per_iteration]
            start_us = float(chunk[0]["ts"])
            end_us = float(chunk[-1]["ts"]) + float(chunk[-1]["dur"])
            kernel_sum_us = sum(float(event["dur"]) for event in chunk)
            occurrences.append(
                {
                    "trace": trace_path.name,
                    "iteration": start_index // args.kernels_per_iteration,
                    "span_us": end_us - start_us,
                    "kernel_sum_us": kernel_sum_us,
                    "gap_us": end_us - start_us - kernel_sum_us,
                    "kernels": [
                        {
                            "index": index,
                            "numel": int((event.get("args") or {})["In msg nelems"]),
                            "dtype": (event.get("args") or {}).get("dtype"),
                            "duration_us": float(event["dur"]),
                            "name": event["name"],
                        }
                        for index, event in enumerate(chunk)
                    ],
                }
            )

    steady = [value for value in occurrences if value["iteration"] > 0]
    per_kernel = []
    for kernel_index in range(args.kernels_per_iteration):
        samples = [value["kernels"][kernel_index] for value in steady]
        signatures = {(sample["numel"], sample["dtype"]) for sample in samples}
        if len(signatures) != 1:
            raise SystemExit(f"Kernel {kernel_index} has inconsistent signatures")
        numel, dtype = next(iter(signatures))
        per_kernel.append(
            {
                "index": kernel_index,
                "numel": numel,
                "dtype": dtype,
                "duration_us": summarize(
                    [sample["duration_us"] for sample in samples]
                ),
            }
        )

    result = {
        "trace_count": len(traces),
        "occurrence_count": len(occurrences),
        "steady_occurrence_count": len(steady),
        "note": "Iteration zero is excluded from steady statistics because profiler startup introduced rank skew.",
        "all_occurrences": {
            field: summarize([value[field] for value in occurrences])
            for field in ("span_us", "kernel_sum_us", "gap_us")
        },
        "steady_occurrences": {
            field: summarize([value[field] for value in steady])
            for field in ("span_us", "kernel_sum_us", "gap_us")
        },
        "per_kernel_steady": per_kernel,
        "occurrences": occurrences,
    }
    output_path = args.output or args.trace_dir / "isolated_reduce_trace_analysis.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
