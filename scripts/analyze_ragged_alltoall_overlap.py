#!/usr/bin/env python3
"""Summarize same-GPU kernel overlap for the ragged all-to-all kernel.

Chrome/Perfetto GPU trace timestamps and durations are interpreted as microseconds,
which is the unit used by PyTorch/XLA JSON traces even when displayTimeUnit asks the
viewer to display nanoseconds.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TARGET_FRAGMENT = "RaggedAllToAllWithSymmetricMemoryKernelImpl"
# Inferred from this trace: one-CTA/SM kernels report occupancy_min_grid_size=152,
# and the ragged-dot dW kernel's grid=148 with sm_margin_4 confirms 152 - 4.
TRACE_GB200_SMS = 152
KERNEL_DETAILS_RE = re.compile(
    r"regs:(?P<regs>\d+) "
    r"static_shared:(?P<static_shared>\d+) "
    r"dynamic_shared:(?P<dynamic_shared>\d+) "
    r"grid:(?P<grid_x>\d+),(?P<grid_y>\d+),(?P<grid_z>\d+) "
    r"block:(?P<block_x>\d+),(?P<block_y>\d+),(?P<block_z>\d+) "
    r"occ_pct:(?P<occupancy>[0-9.]+)"
)


def classify_resources(name: str) -> tuple[str, str]:
    lowered = name.lower()
    if TARGET_FRAGMENT.lower() in lowered:
        return (
            "data: NVLink/NVSwitch symmetric-memory traffic",
            "fabric data movement + HBM/L2",
        )
    if "nccl" in lowered and "barrier" in lowered:
        return (
            "control: NCCL/NVLink synchronization",
            "small fabric/control traffic + HBM/L2",
        )
    if "nccldevkernel" in lowered:
        return (
            "data: NCCL over NVLink/NVSwitch",
            "fabric data movement + HBM/L2",
        )
    if "ragged_dot" in lowered or "nvjet" in lowered:
        return (
            "none indicated",
            "matmul: tensor cores + HBM/L2 + registers/shared memory",
        )
    if any(
        token in lowered
        for token in ("memcpy", "transpose", "scatter", "concatenate", "slice")
    ):
        return (
            "none indicated",
            "local data movement: HBM/L2 + registers/shared memory",
        )
    if "sort" in lowered or "reduce" in lowered:
        return (
            "none indicated",
            "local reduction/sort: HBM/L2 + registers/shared memory",
        )
    return (
        "none indicated",
        "local compute/fusion: HBM/L2 + registers/shared memory",
    )


def parse_kernel_details(details: str) -> dict[str, int | float]:
    match = KERNEL_DETAILS_RE.fullmatch(details)
    if not match:
        return {}
    values: dict[str, int | float] = {
        key: (float(value) if key == "occupancy" else int(value))
        for key, value in match.groupdict().items()
    }
    values["grid_ctas"] = (
        int(values["grid_x"])
        * int(values["grid_y"])
        * int(values["grid_z"])
    )
    values["threads_per_cta"] = (
        int(values["block_x"])
        * int(values["block_y"])
        * int(values["block_z"])
    )
    values["max_sm_coverage"] = min(
        int(values["grid_ctas"]), TRACE_GB200_SMS
    )
    return values


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def union_length(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start > end:
            total += end - start
            start, end = next_start, next_end
        else:
            end = max(end, next_end)
    return total + end - start


def summarize(
    label: str,
    rows: list[dict[str, Any]],
    isolated_mean_us: float,
    isolated_mean_by_baseline_key: dict[str, float],
) -> dict[str, Any]:
    durations = [row["target_duration_us"] for row in rows]
    overlap_durations = [row["overlap_duration_us"] for row in rows]
    overlap_fractions = [row["overlap_fraction"] for row in rows]
    mean_duration = statistics.fmean(durations)
    matched_baselines = [
        isolated_mean_by_baseline_key.get(
            row["target_baseline_key"], isolated_mean_us
        )
        for row in rows
    ]
    matched_baseline = statistics.fmean(matched_baselines)
    return {
        "partner_kernel": label,
        "target_launches": len(rows),
        "gpu_count": len({row["pid"] for row in rows}),
        "mean_target_duration_us": mean_duration,
        "median_target_duration_us": statistics.median(durations),
        "p95_target_duration_us": percentile(durations, 0.95),
        "min_target_duration_us": min(durations),
        "max_target_duration_us": max(durations),
        "mean_overlap_duration_us": statistics.fmean(overlap_durations),
        "median_overlap_duration_us": statistics.median(overlap_durations),
        "mean_overlap_fraction": statistics.fmean(overlap_fractions),
        "duration_vs_alone_ratio": (
            mean_duration / isolated_mean_us
            if isolated_mean_us and not math.isnan(isolated_mean_us)
            else math.nan
        ),
        "matched_alone_mean_duration_us": matched_baseline,
        "duration_vs_matched_alone_ratio": (
            mean_duration / matched_baseline
            if matched_baseline and not math.isnan(matched_baseline)
            else math.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV (default: <trace stem>.ragged_alltoall_overlap.csv)",
    )
    args = parser.parse_args()

    output = args.output or args.trace.with_name(
        args.trace.name.removesuffix(".trace.json")
        + ".ragged_alltoall_overlap.csv"
    )

    with args.trace.open() as handle:
        trace = json.load(handle)
    events = trace["traceEvents"]

    process_names = {
        event["pid"]: event.get("args", {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    gpu_pids = {
        pid
        for pid, name in process_names.items()
        if isinstance(name, str) and name.startswith("/device:GPU:")
    }

    # kernel_details distinguishes actual GPU kernels from memcpy/memset activity
    # and from host-side launch annotations that also use complete events.
    kernels_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    target_events: list[dict[str, Any]] = []
    for event in events:
        if (
            event.get("pid") not in gpu_pids
            or event.get("ph") != "X"
            or "kernel_details" not in event.get("args", {})
        ):
            continue
        event["_end"] = float(event["ts"]) + float(event["dur"])
        kernels_by_pid[event["pid"]].append(event)
        if TARGET_FRAGMENT in event.get("name", ""):
            target_events.append(event)

    for pid in kernels_by_pid:
        kernels_by_pid[pid].sort(key=lambda event: float(event["ts"]))

    # Prefix maximum end times let us stop a backwards interval scan once no
    # earlier-starting event can still overlap the target interval.
    starts_by_pid: dict[int, list[float]] = {}
    prefix_max_end_by_pid: dict[int, list[float]] = {}
    for pid, kernels in kernels_by_pid.items():
        starts_by_pid[pid] = [float(event["ts"]) for event in kernels]
        prefix: list[float] = []
        running_max = -math.inf
        for event in kernels:
            running_max = max(running_max, event["_end"])
            prefix.append(running_max)
        prefix_max_end_by_pid[pid] = prefix

    isolated: list[dict[str, Any]] = []
    any_overlap: list[dict[str, Any]] = []
    by_partner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sole_partner: dict[str, list[dict[str, Any]]] = defaultdict(list)
    overlapping_partner_events: dict[
        str, dict[tuple[Any, ...], dict[str, Any]]
    ] = defaultdict(dict)
    target_details: list[dict[str, Any]] = []

    for target in target_events:
        pid = target["pid"]
        target_start = float(target["ts"])
        target_end = target["_end"]
        target_duration = float(target["dur"])
        kernels = kernels_by_pid[pid]
        starts = starts_by_pid[pid]
        prefix_max_end = prefix_max_end_by_pid[pid]
        stop = bisect.bisect_left(starts, target_end)

        overlaps_by_name: dict[str, list[tuple[float, float]]] = defaultdict(list)
        index = stop - 1
        while index >= 0 and prefix_max_end[index] > target_start:
            other = kernels[index]
            if other is not target and other["_end"] > target_start:
                overlap_start = max(target_start, float(other["ts"]))
                overlap_end = min(target_end, other["_end"])
                if overlap_end > overlap_start:
                    other_name = other.get("name", "<unnamed>")
                    overlaps_by_name[other_name].append(
                        (overlap_start, overlap_end)
                    )
                    event_key = (
                        other.get("pid"),
                        other.get("tid"),
                        other.get("ts"),
                        other.get("dur"),
                        other_name,
                    )
                    overlapping_partner_events[other_name][event_key] = other
            index -= 1

        base = {
            "pid": pid,
            "gpu": process_names[pid],
            "target_ts_us": target_start,
            "target_duration_us": target_duration,
            "target_hlo_op": target.get("args", {}).get("hlo_op", ""),
            "target_signature": target.get("args", {}).get("kernel_details", ""),
            # Match comparisons to isolated launches with the same launch shape
            # on the same physical GPU. This controls both A2A payload class and
            # stable device-to-device timing differences visible in the trace.
            "target_baseline_key": (
                f'{pid}|{target.get("args", {}).get("kernel_details", "")}'
            ),
        }
        if not overlaps_by_name:
            row = {
                **base,
                "overlap_duration_us": 0.0,
                "overlap_fraction": 0.0,
            }
            isolated.append(row)
            target_details.append({**row, "partner_kernel": "__ALONE__"})
            continue

        all_intervals = [
            interval
            for intervals in overlaps_by_name.values()
            for interval in intervals
        ]
        total_overlap_duration = union_length(all_intervals)
        any_overlap.append(
            {
                **base,
                "overlap_duration_us": total_overlap_duration,
                "overlap_fraction": total_overlap_duration / target_duration,
            }
        )
        for name, intervals in overlaps_by_name.items():
            overlap_duration = union_length(intervals)
            row = {
                **base,
                "overlap_duration_us": overlap_duration,
                "overlap_fraction": overlap_duration / target_duration,
            }
            by_partner[name].append(row)
            target_details.append({**row, "partner_kernel": name})
            if len(overlaps_by_name) == 1:
                by_sole_partner[name].append(row)

    isolated_mean_us = (
        statistics.fmean(row["target_duration_us"] for row in isolated)
        if isolated
        else math.nan
    )
    isolated_by_baseline_key: dict[str, list[float]] = defaultdict(list)
    for row in isolated:
        isolated_by_baseline_key[row["target_baseline_key"]].append(
            row["target_duration_us"]
        )
    isolated_mean_by_baseline_key = {
        baseline_key: statistics.fmean(durations)
        for baseline_key, durations in isolated_by_baseline_key.items()
    }
    summaries: list[dict[str, Any]] = []
    if isolated:
        summaries.append(
            summarize(
                "__ALONE__",
                isolated,
                isolated_mean_us,
                isolated_mean_by_baseline_key,
            )
        )
    if any_overlap:
        summaries.append(
            summarize(
                "__ANY_OVERLAP__",
                any_overlap,
                isolated_mean_us,
                isolated_mean_by_baseline_key,
            )
        )
    for name, rows in by_partner.items():
        summary = summarize(
            name, rows, isolated_mean_us, isolated_mean_by_baseline_key
        )
        sole_rows = by_sole_partner.get(name, [])
        summary["sole_partner_launches"] = len(sole_rows)
        if sole_rows:
            sole_mean = statistics.fmean(
                row["target_duration_us"] for row in sole_rows
            )
            sole_baseline = statistics.fmean(
                isolated_mean_by_baseline_key.get(
                    row["target_baseline_key"], isolated_mean_us
                )
                for row in sole_rows
            )
            summary["mean_target_duration_sole_partner_us"] = sole_mean
            summary["duration_vs_matched_alone_sole_partner_ratio"] = (
                sole_mean / sole_baseline
            )
        else:
            summary["mean_target_duration_sole_partner_us"] = math.nan
            summary["duration_vs_matched_alone_sole_partner_ratio"] = math.nan
        summaries.append(summary)

    for summary in summaries:
        summary.setdefault("sole_partner_launches", 0)
        summary.setdefault("mean_target_duration_sole_partner_us", math.nan)
        summary.setdefault(
            "duration_vs_matched_alone_sole_partner_ratio", math.nan
        )

    # Put the baseline first, then the most frequently observed partners.
    summaries.sort(
        key=lambda row: (
            {"__ALONE__": 0, "__ANY_OVERLAP__": 1}.get(
                row["partner_kernel"], 2
            ),
            -row["target_launches"],
            row["partner_kernel"],
        )
    )
    fieldnames = [
        "partner_kernel",
        "target_launches",
        "gpu_count",
        "mean_target_duration_us",
        "median_target_duration_us",
        "p95_target_duration_us",
        "min_target_duration_us",
        "max_target_duration_us",
        "mean_overlap_duration_us",
        "median_overlap_duration_us",
        "mean_overlap_fraction",
        "duration_vs_alone_ratio",
        "matched_alone_mean_duration_us",
        "duration_vs_matched_alone_ratio",
        "sole_partner_launches",
        "mean_target_duration_sole_partner_us",
        "duration_vs_matched_alone_sole_partner_ratio",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    detail_output = output.with_name(output.stem + ".instances.csv")
    detail_fields = [
        "partner_kernel",
        "pid",
        "gpu",
        "target_ts_us",
        "target_duration_us",
        "target_hlo_op",
        "target_signature",
        "target_baseline_key",
        "overlap_duration_us",
        "overlap_fraction",
    ]
    with detail_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(target_details)

    resource_output = output.with_name(output.stem + ".resources.csv")
    resource_fields = [
        "kernel",
        "is_ragged_a2a_target",
        "overlapping_kernel_launches",
        "resource_config_launches",
        "resource_config_share",
        "regs_per_thread",
        "static_shared_bytes_per_cta",
        "dynamic_shared_bytes_per_cta",
        "grid_x",
        "grid_y",
        "grid_z",
        "grid_ctas",
        "block_x",
        "block_y",
        "block_z",
        "threads_per_cta",
        "theoretical_occupancy_pct",
        "max_sm_coverage_of_152",
        "covers_all_152_sms",
        "inter_gpu_fabric_use",
        "memory_compute_resource_class",
        "hlo_ops",
        "raw_kernel_details",
    ]
    resource_rows: list[dict[str, Any]] = []
    resource_event_groups: list[tuple[str, list[dict[str, Any]], bool]] = [
        ("__RAGGED_A2A_TARGET__", target_events, True)
    ]
    resource_event_groups.extend(
        (
            name,
            list(event_map.values()),
            False,
        )
        for name, event_map in sorted(overlapping_partner_events.items())
    )
    for name, kernel_events, is_target in resource_event_groups:
        detail_counts = Counter(
            event.get("args", {}).get("kernel_details", "")
            for event in kernel_events
        )
        network_class, memory_class = classify_resources(
            target_events[0]["name"] if is_target else name
        )
        total = len(kernel_events)
        for details, count in detail_counts.most_common():
            parsed = parse_kernel_details(details)
            hlo_ops = sorted(
                {
                    str(event.get("args", {}).get("hlo_op"))
                    for event in kernel_events
                    if event.get("args", {}).get("kernel_details", "")
                    == details
                    and event.get("args", {}).get("hlo_op") is not None
                }
            )
            resource_rows.append(
                {
                    "kernel": name,
                    "is_ragged_a2a_target": is_target,
                    "overlapping_kernel_launches": total,
                    "resource_config_launches": count,
                    "resource_config_share": count / total,
                    "regs_per_thread": parsed.get("regs", ""),
                    "static_shared_bytes_per_cta": parsed.get(
                        "static_shared", ""
                    ),
                    "dynamic_shared_bytes_per_cta": parsed.get(
                        "dynamic_shared", ""
                    ),
                    "grid_x": parsed.get("grid_x", ""),
                    "grid_y": parsed.get("grid_y", ""),
                    "grid_z": parsed.get("grid_z", ""),
                    "grid_ctas": parsed.get("grid_ctas", ""),
                    "block_x": parsed.get("block_x", ""),
                    "block_y": parsed.get("block_y", ""),
                    "block_z": parsed.get("block_z", ""),
                    "threads_per_cta": parsed.get("threads_per_cta", ""),
                    "theoretical_occupancy_pct": parsed.get(
                        "occupancy", ""
                    ),
                    "max_sm_coverage_of_152": parsed.get(
                        "max_sm_coverage", ""
                    ),
                    "covers_all_152_sms": (
                        parsed.get("max_sm_coverage") == TRACE_GB200_SMS
                        if parsed
                        else ""
                    ),
                    "inter_gpu_fabric_use": network_class,
                    "memory_compute_resource_class": memory_class,
                    "hlo_ops": ";".join(hlo_ops),
                    "raw_kernel_details": details,
                }
            )
    with resource_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resource_fields)
        writer.writeheader()
        writer.writerows(resource_rows)

    report_output = output.with_suffix(".md")
    with report_output.open("w") as handle:
        handle.write("# Ragged all-to-all kernel overlap analysis\n\n")
        handle.write(
            f"Trace-local scope: {len(gpu_pids)} GPU lanes, "
            f"{len(target_events):,} main ragged A2A launches. "
            "Host launch annotations and non-kernel GPU activities are excluded.\n\n"
        )
        handle.write(
            "`__ALONE__` means no positive-duration overlap with any other GPU "
            "kernel on the same physical GPU. Partner rows are multi-label: one "
            "target launch can appear in more than one row. Durations are the "
            "ragged A2A kernel's full duration, not just the overlapped portion. "
            "The matched baseline controls for GPU and ragged-A2A launch grid.\n\n"
        )
        handle.write(
            "| Concurrent partner | Target launches | Mean target (µs) | "
            "Median (µs) | P95 (µs) | Mean overlap | Matched-alone (µs) | "
            "Matched ratio | Sole-partner launches | Sole-partner mean (µs) |\n"
        )
        handle.write(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for row in summaries:
            sole_mean = row["mean_target_duration_sole_partner_us"]
            sole_mean_text = (
                "—" if math.isnan(sole_mean) else f"{sole_mean:.3f}"
            )
            handle.write(
                f'| `{row["partner_kernel"]}` | {row["target_launches"]} | '
                f'{row["mean_target_duration_us"]:.3f} | '
                f'{row["median_target_duration_us"]:.3f} | '
                f'{row["p95_target_duration_us"]:.3f} | '
                f'{100 * row["mean_overlap_fraction"]:.2f}% | '
                f'{row["matched_alone_mean_duration_us"]:.3f} | '
                f'{row["duration_vs_matched_alone_ratio"]:.3f}× | '
                f'{row["sole_partner_launches"]} | {sole_mean_text} |\n'
            )
        handle.write("\n## Resource data\n\n")
        handle.write(
            f"See `{resource_output.name}` for every target/partner launch "
            "configuration: registers, shared memory, grid/CTA geometry, "
            "theoretical occupancy, maximum SM coverage out of 152 SMs, and "
            "inferred local-memory versus NVLink/NVSwitch use. SM coverage is a "
            "launch-geometry upper bound, not a hardware-counter measurement.\n"
        )

    print(f"GPU processes: {len(gpu_pids)}")
    print(f"Ragged A2A launches: {len(target_events)}")
    print(f"Alone launches: {len(isolated)}")
    print(f"Distinct overlapping kernel names: {len(by_partner)}")
    print(f"Summary: {output}")
    print(f"Instances: {detail_output}")
    print(f"Resources: {resource_output}")
    print(f"Report: {report_output}")


if __name__ == "__main__":
    main()
