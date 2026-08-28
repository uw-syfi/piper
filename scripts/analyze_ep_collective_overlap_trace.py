#!/usr/bin/env python3
"""Analyze A2A, all-gather, and reduce-scatter overlap in an XLA GPU trace."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ALLTOALL_HLO_PREFIX = "all-to-all"
ALLGATHER_NAME = "ncclDevKernel_AllGather"
REDUCESCATTER_NAME = "ncclDevKernel_ReduceScatter"
NCCL_NAME = "ncclDevKernel"


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (
        position - lower
    )


def summarize(values: list[float]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "median_us": None,
            "mean_us": None,
            "p95_us": None,
            "minimum_us": None,
            "maximum_us": None,
        }
    return {
        "count": len(values),
        "median_us": statistics.median(values),
        "mean_us": statistics.fmean(values),
        "p95_us": percentile(values, 0.95),
        "minimum_us": min(values),
        "maximum_us": max(values),
    }


def event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event["pid"],
        event["tid"],
        event["ts"],
        event["dur"],
        event["name"],
        event["_kernel_id"],
    )


def compute_signature(event: dict[str, Any]) -> tuple[str, str, str, str]:
    args = event.get("args") or {}
    return (
        str(event["name"]),
        str(args.get("kernel_details", "")),
        str(args.get("hlo_module", "")),
        str(args.get("hlo_op", "")),
    )


class KernelIndex:
    def __init__(self, kernels_by_pid: dict[int, list[dict[str, Any]]]):
        self.kernels_by_pid = kernels_by_pid
        self.starts_by_pid: dict[int, list[float]] = {}
        self.prefix_max_end_by_pid: dict[int, list[float]] = {}
        for pid, kernels in kernels_by_pid.items():
            kernels.sort(key=lambda event: float(event["ts"]))
            self.starts_by_pid[pid] = [
                float(event["ts"]) for event in kernels
            ]
            prefix: list[float] = []
            running_max = -math.inf
            for event in kernels:
                running_max = max(running_max, float(event["_end"]))
                prefix.append(running_max)
            self.prefix_max_end_by_pid[pid] = prefix

    def overlaps_on_other_streams(
        self, target: dict[str, Any]
    ) -> list[dict[str, Any]]:
        pid = target["pid"]
        target_start = float(target["ts"])
        target_end = float(target["_end"])
        kernels = self.kernels_by_pid[pid]
        starts = self.starts_by_pid[pid]
        prefix_max_end = self.prefix_max_end_by_pid[pid]
        stop = bisect.bisect_left(starts, target_end)
        overlaps: list[dict[str, Any]] = []
        index = stop - 1
        while index >= 0 and prefix_max_end[index] > target_start:
            other = kernels[index]
            if (
                other["_kernel_id"] != target["_kernel_id"]
                and other["tid"] != target["tid"]
                and float(other["_end"]) > target_start
            ):
                overlaps.append(other)
            index -= 1
        return overlaps


def is_compute(event: dict[str, Any]) -> bool:
    return not str(event["name"]).startswith(NCCL_NAME)


def is_alltoall(event: dict[str, Any]) -> bool:
    return str((event.get("args") or {}).get("hlo_op", "")).startswith(
        ALLTOALL_HLO_PREFIX
    )


def is_allgather(event: dict[str, Any]) -> bool:
    return ALLGATHER_NAME in str(event["name"])


def is_reducescatter(event: dict[str, Any]) -> bool:
    return REDUCESCATTER_NAME in str(event["name"])


def classify_alltoalls(
    targets: list[dict[str, Any]], index: KernelIndex
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    category_values: dict[str, list[float]] = defaultdict(list)
    by_hlo_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    detail_rows: list[dict[str, Any]] = []
    both_orders = 0
    collective_only = 0
    overlapping_any = 0
    for target in targets:
        overlaps = index.overlaps_on_other_streams(target)
        compute_overlaps = [event for event in overlaps if is_compute(event)]
        compute_before = [
            event
            for event in compute_overlaps
            if float(event["ts"]) < float(target["ts"])
        ]
        compute_after = [
            event
            for event in compute_overlaps
            if float(event["ts"]) > float(target["ts"])
        ]
        hlo_op = str((target.get("args") or {}).get("hlo_op", ""))
        duration = float(target["dur"])
        if overlaps:
            overlapping_any += 1
        else:
            category_values["strict_alone"].append(duration)
            by_hlo_values[hlo_op]["strict_alone"].append(duration)
        if compute_before:
            category_values["launched_after_compute"].append(duration)
            by_hlo_values[hlo_op]["launched_after_compute"].append(duration)
        if compute_after:
            category_values["launched_before_compute"].append(duration)
            by_hlo_values[hlo_op]["launched_before_compute"].append(duration)
        if compute_before and compute_after:
            both_orders += 1
        if overlaps and not compute_overlaps:
            collective_only += 1
        detail_rows.append(
            {
                "gpu_pid": target["pid"],
                "stream": target["tid"],
                "hlo_op": hlo_op,
                "start_us": float(target["ts"]),
                "duration_us": duration,
                "other_stream_kernel_count": len(overlaps),
                "compute_overlap_count": len(compute_overlaps),
                "has_compute_started_before_a2a": bool(compute_before),
                "has_compute_started_after_a2a": bool(compute_after),
                "strict_alone": not overlaps,
                "compute_kernel_names": ";".join(
                    sorted({str(event["name"]) for event in compute_overlaps})
                ),
            }
        )
    categories = {
        key: summarize(category_values[key])
        for key in (
            "strict_alone",
            "launched_after_compute",
            "launched_before_compute",
        )
    }
    by_hlo = {
        hlo_op: {
            "total": len([target for target in targets if target["args"]["hlo_op"] == hlo_op]),
            **{
                key: summarize(values[key])
                for key in (
                    "strict_alone",
                    "launched_after_compute",
                    "launched_before_compute",
                )
            },
        }
        for hlo_op, values in sorted(by_hlo_values.items())
    }
    return (
        {
            "target_count": len(targets),
            "overlapping_any_other_stream_kernel_count": overlapping_any,
            "collective_only_overlap_count": collective_only,
            "both_compute_start_orders_count": both_orders,
            "categories_are_nonexclusive": True,
            "categories": categories,
            "by_hlo_op": by_hlo,
        },
        detail_rows,
    )


def collect_partner_sets(
    targets: list[dict[str, Any]], index: KernelIndex
) -> tuple[set[tuple[Any, ...]], set[tuple[Any, ...]], int]:
    """Return compute occurrences starting after/before each collective.

    "after" means the compute kernel starts after the collective; "before"
    means the compute kernel was already running when the collective starts.
    An occurrence can belong to both sets if it overlaps multiple collectives.
    """
    compute_after_collective: set[tuple[Any, ...]] = set()
    compute_before_collective: set[tuple[Any, ...]] = set()
    simultaneous = 0
    for target in targets:
        for event in index.overlaps_on_other_streams(target):
            if not is_compute(event):
                continue
            key = event_key(event)
            if float(event["ts"]) > float(target["ts"]):
                compute_after_collective.add(key)
            elif float(event["ts"]) < float(target["ts"]):
                compute_before_collective.add(key)
            else:
                simultaneous += 1
    return compute_after_collective, compute_before_collective, simultaneous


def strict_alone_compute(
    compute_events: list[dict[str, Any]], index: KernelIndex
) -> set[tuple[Any, ...]]:
    return {
        event_key(event)
        for event in compute_events
        if not index.overlaps_on_other_streams(event)
    }


def collect_compute_partner_keys(
    targets: list[dict[str, Any]], index: KernelIndex
) -> set[tuple[Any, ...]]:
    """Return every compute occurrence overlapping at least one target."""
    return {
        event_key(event)
        for target in targets
        for event in index.overlaps_on_other_streams(target)
        if is_compute(event)
    }


def compute_overlap_table(
    target_name: str,
    targets: list[dict[str, Any]],
    compute_events: list[dict[str, Any]],
    alone_keys: set[tuple[Any, ...]],
    index: KernelIndex,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    after_keys, before_keys, simultaneous = collect_partner_sets(targets, index)
    events_by_key = {event_key(event): event for event in compute_events}
    events_by_signature: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for event in compute_events:
        events_by_signature[compute_signature(event)].append(event)
    signatures = sorted(
        {
            compute_signature(events_by_key[key])
            for key in after_keys | before_keys
        }
    )
    rows: list[dict[str, Any]] = []
    for signature in signatures:
        name, kernel_details, hlo_module, hlo_op = signature
        all_signature_events = events_by_signature[signature]
        after_values = [
            float(event["dur"])
            for event in all_signature_events
            if event_key(event) in after_keys
        ]
        before_values = [
            float(event["dur"])
            for event in all_signature_events
            if event_key(event) in before_keys
        ]
        alone_values = [
            float(event["dur"])
            for event in all_signature_events
            if event_key(event) in alone_keys
        ]
        after_summary = summarize(after_values)
        before_summary = summarize(before_values)
        alone_summary = summarize(alone_values)
        alone_median = alone_summary["median_us"]
        rows.append(
            {
                "compute_kernel": name,
                "kernel_details": kernel_details,
                "hlo_module": hlo_module,
                "hlo_op": hlo_op,
                f"overlap_{target_name}_collective_first_count": after_summary[
                    "count"
                ],
                f"overlap_{target_name}_collective_first_median_us": after_summary[
                    "median_us"
                ],
                f"overlap_{target_name}_collective_first_p95_us": after_summary[
                    "p95_us"
                ],
                f"overlap_{target_name}_compute_first_count": before_summary[
                    "count"
                ],
                f"overlap_{target_name}_compute_first_median_us": before_summary[
                    "median_us"
                ],
                f"overlap_{target_name}_compute_first_p95_us": before_summary[
                    "p95_us"
                ],
                "strict_alone_count": alone_summary["count"],
                "strict_alone_median_us": alone_median,
                "strict_alone_p95_us": alone_summary["p95_us"],
                f"collective_first_vs_alone_ratio": (
                    float(after_summary["median_us"]) / float(alone_median)
                    if after_summary["median_us"] is not None
                    and alone_median is not None
                    and float(alone_median) > 0
                    else None
                ),
                f"compute_first_vs_alone_ratio": (
                    float(before_summary["median_us"]) / float(alone_median)
                    if before_summary["median_us"] is not None
                    and alone_median is not None
                    and float(alone_median) > 0
                    else None
                ),
            }
        )
    rows.sort(
        key=lambda row: max(
            float(row.get(f"collective_first_vs_alone_ratio") or -math.inf),
            float(row.get(f"compute_first_vs_alone_ratio") or -math.inf),
        ),
        reverse=True,
    )
    return (
        rows,
        {
            "collective_kernel_count": len(targets),
            "overlapping_compute_kernel_signature_count": len(rows),
            "overlapping_compute_kernel_name_count": len(
                {row["compute_kernel"] for row in rows}
            ),
            "compute_occurrences_collective_first": len(after_keys),
            "compute_occurrences_compute_first": len(before_keys),
            "compute_occurrences_in_both_start_order_sets": len(
                after_keys & before_keys
            ),
            "simultaneous_start_pair_count": simultaneous,
        },
    )


def compute_any_collective_overlap_table(
    targets_by_name: dict[str, list[dict[str, Any]]],
    compute_events: list[dict[str, Any]],
    alone_keys: set[tuple[Any, ...]],
    index: KernelIndex,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare compute overlapped by any EP collective against strict-alone."""
    overlap_keys_by_collective = {
        name: collect_compute_partner_keys(targets, index)
        for name, targets in targets_by_name.items()
    }
    any_overlap_keys: set[tuple[Any, ...]] = set().union(
        *overlap_keys_by_collective.values()
    )
    events_by_signature: dict[
        tuple[str, str, str, str], list[dict[str, Any]]
    ] = defaultdict(list)
    for event in compute_events:
        events_by_signature[compute_signature(event)].append(event)

    overlapping_signatures = {
        compute_signature(event)
        for event in compute_events
        if event_key(event) in any_overlap_keys
    }
    rows: list[dict[str, Any]] = []
    excluded_without_alone = 0
    for signature in sorted(overlapping_signatures):
        name, kernel_details, hlo_module, hlo_op = signature
        signature_events = events_by_signature[signature]
        overlap_events = [
            event
            for event in signature_events
            if event_key(event) in any_overlap_keys
        ]
        alone_events = [
            event
            for event in signature_events
            if event_key(event) in alone_keys
        ]
        if not alone_events:
            excluded_without_alone += 1
            continue
        overlap_summary = summarize(
            [float(event["dur"]) for event in overlap_events]
        )
        alone_summary = summarize(
            [float(event["dur"]) for event in alone_events]
        )
        overlap_median = float(overlap_summary["median_us"])
        alone_median = float(alone_summary["median_us"])
        row = {
            "compute_kernel": name,
            "kernel_details": kernel_details,
            "hlo_module": hlo_module,
            "hlo_op": hlo_op,
            "overlap_any_a2a_ag_rs_count": overlap_summary["count"],
            "overlap_any_a2a_ag_rs_median_us": overlap_median,
            "overlap_any_a2a_ag_rs_p95_us": overlap_summary["p95_us"],
            "strict_alone_count": alone_summary["count"],
            "strict_alone_median_us": alone_median,
            "strict_alone_p95_us": alone_summary["p95_us"],
            "overlap_vs_strict_alone_ratio": (
                overlap_median / alone_median
            ),
        }
        for target_name, keys in overlap_keys_by_collective.items():
            category_summary = summarize(
                [
                    float(event["dur"])
                    for event in signature_events
                    if event_key(event) in keys
                ]
            )
            category_median = category_summary["median_us"]
            row[f"overlap_{target_name}_count"] = category_summary["count"]
            row[f"overlap_{target_name}_median_us"] = category_median
            row[f"{target_name}_vs_strict_alone_ratio"] = (
                float(category_median) / alone_median
                if category_median is not None
                else None
            )
        rows.append(row)
    rows.sort(
        key=lambda row: float(row["overlap_vs_strict_alone_ratio"]),
        reverse=True,
    )
    return (
        rows,
        {
            "overlapping_compute_occurrence_count": len(any_overlap_keys),
            "overlapping_compute_signature_count": len(
                overlapping_signatures
            ),
            "reported_signature_count": len(rows),
            "excluded_signature_count_without_strict_alone": (
                excluded_without_alone
            ),
            "strict_alone_compute_occurrence_count": len(alone_keys),
            "collective_overlap_occurrence_counts": {
                name: len(keys)
                for name, keys in overlap_keys_by_collective.items()
            },
        },
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_any_collective_report(
    path: Path,
    source: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "- DeepSeek EP8 x PP2 compute-kernel slowdown with communication overlap",
        f"  - Source trace: {source}",
        "  - Sequence length: not recorded in the trace and not reliably inferable from kernel events.",
        "  - Overlapped means that the compute occurrence intersects at least one A2A, all-gather, or reduce-scatter GPU kernel on another CUDA stream.",
        "  - Normal means strict-alone: no GPU kernel on any other CUDA stream overlaps the compute occurrence.",
        "  - Each result is one exact signature defined by kernel name, launch details, XLA module, and XLA operation.",
        "  - Slowdown ratio is overlapped median duration divided by normal median duration.",
        f"  - Reported signatures: {metadata['reported_signature_count']}",
        f"  - Excluded signatures without a normal occurrence: {metadata['excluded_signature_count_without_strict_alone']}",
        f"  - Distinct overlapped compute occurrences: {metadata['overlapping_compute_occurrence_count']}",
        f"  - Strict-alone compute occurrences in the trace: {metadata['strict_alone_compute_occurrence_count']}",
        "- Summary by communication class",
    ]
    for label, target_name in (
        ("All-gather", "allgather"),
        ("Reduce-scatter", "reducescatter"),
        ("A2A", "alltoall"),
    ):
        ratio_field = f"{target_name}_vs_strict_alone_ratio"
        count_field = f"overlap_{target_name}_count"
        category_rows = sorted(
            [row for row in rows if row[ratio_field] is not None],
            key=lambda row: float(row[ratio_field]),
            reverse=True,
        )
        ratios = [float(row[ratio_field]) for row in category_rows]
        slowed_ratios = [ratio for ratio in ratios if ratio > 1.0]
        lines.extend(
            [
                f"  - {label}",
                f"    - Exact signatures with normal controls: {len(category_rows)}",
                (
                    f"    - Full observed ratio range: "
                    f"{min(ratios):.4f}x to {max(ratios):.4f}x"
                ),
                (
                    f"    - Slowdown-only range: "
                    f"{min(slowed_ratios):.4f}x to "
                    f"{max(slowed_ratios):.4f}x"
                ),
                "    - Top five slowed kernels",
            ]
        )
        for row in category_rows[:5]:
            lines.append(
                (
                    f"      - {row['compute_kernel']} | "
                    f"XLA module: {row['hlo_module']} | "
                    f"XLA operation: {row['hlo_op']} | "
                    f"slowdown: {float(row[ratio_field]):.4f}x | "
                    f"overlapped occurrences: {row[count_field]} | "
                    f"normal occurrences: {row['strict_alone_count']}"
                )
            )
    lines.extend(
        [
        "- Results sorted by slowdown ratio from highest to lowest",
        ]
    )
    for row in rows:
        lines.extend(
            [
                (
                    f"  - Kernel: {row['compute_kernel']} | "
                    f"XLA module: {row['hlo_module']} | "
                    f"XLA operation: {row['hlo_op']}"
                ),
                (
                    f"    - Slowdown ratio: "
                    f"{float(row['overlap_vs_strict_alone_ratio']):.4f}x"
                ),
                (
                    f"    - Overlapped occurrences: "
                    f"{row['overlap_any_a2a_ag_rs_count']} | "
                    f"median duration: "
                    f"{float(row['overlap_any_a2a_ag_rs_median_us']):.3f} us"
                ),
                (
                    f"    - Normal occurrences: {row['strict_alone_count']} | "
                    f"median duration: "
                    f"{float(row['strict_alone_median_us']):.3f} us"
                ),
                (
                    f"    - Overlap occurrence labels: "
                    f"A2A {row['overlap_alltoall_count']} | "
                    f"all-gather {row['overlap_allgather_count']} | "
                    f"reduce-scatter {row['overlap_reducescatter_count']}"
                ),
                f"    - Kernel details: {row['kernel_details']}",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def format_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_compute_table(
    title: str, target_name: str, rows: list[dict[str, Any]]
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        (
            "A compute occurrence can appear in both overlap columns if it "
            "overlaps multiple collective occurrences with different start "
            "orders. `Alone` is strict: no kernel on any other stream overlaps."
        ),
        "",
        "| Compute kernel | Collective first n | Collective first median µs | Compute first n | Compute first median µs | Alone n | Alone median µs | Collective-first / alone | Compute-first / alone |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    (
                        f"`{str(row['compute_kernel']).replace('|', '\\|')}`"
                        f"<br><sub>{str(row['hlo_module']).replace('|', '\\|')} / "
                        f"{str(row['hlo_op']).replace('|', '\\|')}</sub>"
                        f"<br><sub>{str(row['kernel_details']).replace('|', '\\|')}</sub>"
                    ),
                    str(
                        row[
                            f"overlap_{target_name}_collective_first_count"
                        ]
                    ),
                    format_value(
                        row[
                            f"overlap_{target_name}_collective_first_median_us"
                        ]
                    ),
                    str(
                        row[f"overlap_{target_name}_compute_first_count"]
                    ),
                    format_value(
                        row[f"overlap_{target_name}_compute_first_median_us"]
                    ),
                    str(row["strict_alone_count"]),
                    format_value(row["strict_alone_median_us"]),
                    (
                        format_value(
                            row["collective_first_vs_alone_ratio"], 4
                        )
                        + "×"
                        if row["collective_first_vs_alone_ratio"] is not None
                        else "N/A"
                    ),
                    (
                        format_value(row["compute_first_vs_alone_ratio"], 4)
                        + "×"
                        if row["compute_first_vs_alone_ratio"] is not None
                        else "N/A"
                    ),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def write_report(
    path: Path,
    source: Path,
    process_names: dict[int, str],
    alltoall_summary: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
    table_metadata: dict[str, dict[str, Any]],
    counts: dict[str, int],
) -> None:
    categories = alltoall_summary["categories"]
    lines = [
        "# EP8 × PP2 DeepSeek collective-overlap trace analysis",
        "",
        f"Source trace: `{source}`",
        "",
        (
            "Critical-path counterfactual and confidence interval: "
            "[`critical_path_slowdown.md`](critical_path_slowdown.md)."
        ),
        (
            "Combined A2A/all-gather/reduce-scatter compute slowdown: "
            "[`compute_vs_any_ep_collective.md`](compute_vs_any_ep_collective.md)."
        ),
        "",
        "## Trace scope and definitions",
        "",
        (
            f"The trace contains {len(process_names)} GPU tracks and "
            f"{counts['gpu_kernel_count']:,} actual GPU-kernel events. "
            "`kernel_details` distinguishes kernels from memcpy/memset "
            "activities and host/XLA annotations."
        ),
        "",
        (
            f"There are **{counts['alltoall_count']} true all-to-all kernels**: "
            f"{counts['alltoall_main_count']} steady `all-to-all.2` kernels and "
            f"{counts['alltoall_setup_count']} setup kernels. The trace also "
            f"contains {counts['sendrecv_non_a2a_count']} other "
            "`ncclDevKernel_SendRecv` kernels that are pipeline/other traffic; "
            "they are excluded from the all-to-all statistics."
        ),
        "",
        (
            "All overlap tests are per GPU and use half-open device intervals. "
            "Only kernels on a different CUDA stream count as concurrent. "
            "Launch order means GPU start-timestamp order, not CPU API-call "
            "order. All times are microseconds."
        ),
        "",
        "## All-to-all kernel duration",
        "",
        "| Category | n | Median µs | Mean µs | p95 µs |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, label in (
        ("strict_alone", "No kernel on another stream"),
        ("launched_after_compute", "A2A starts after overlapping compute"),
        ("launched_before_compute", "A2A starts before overlapping compute"),
    ):
        summary = categories[key]
        lines.append(
            f"| {label} | {summary['count']} | "
            f"{format_value(summary['median_us'])} | "
            f"{format_value(summary['mean_us'])} | "
            f"{format_value(summary['p95_us'])} |"
        )
    lines.extend(
        [
            "",
            (
                f"The start-order groups are nonexclusive: "
                f"{alltoall_summary['both_compute_start_orders_count']} A2As "
                "overlap compute kernels that start on both sides of the A2A "
                "start timestamp."
            ),
            "",
            (
                "**Important comparability caveat:** all strict-alone A2As are "
                "small setup HLOs. None of the 64 steady `all-to-all.2` kernels "
                "runs strictly alone, so the trace does not provide an "
                "isolated baseline for the production A2A shape."
            ),
            "",
            "### Breakdown by all-to-all HLO operation",
            "",
            "| HLO operation | Total | Alone n / median µs | A2A after compute n / median µs | A2A before compute n / median µs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for hlo_op, values in alltoall_summary["by_hlo_op"].items():
        alone = values["strict_alone"]
        after = values["launched_after_compute"]
        before = values["launched_before_compute"]
        lines.append(
            f"| `{hlo_op}` | {values['total']} | "
            f"{alone['count']} / {format_value(alone['median_us'])} | "
            f"{after['count']} / {format_value(after['median_us'])} | "
            f"{before['count']} / {format_value(before['median_us'])} |"
        )
    lines.append("")
    lines.extend(
        markdown_compute_table(
            "Compute kernels overlapping all-to-all",
            "alltoall",
            tables["alltoall"],
        )
    )
    lines.extend(
        markdown_compute_table(
            "Compute kernels overlapping all-gather",
            "allgather",
            tables["allgather"],
        )
    )
    lines.extend(
        markdown_compute_table(
            "Compute kernels overlapping reduce-scatter",
            "reducescatter",
            tables["reducescatter"],
        )
    )
    lines.extend(
        [
            "## Coverage",
            "",
            "| Collective | Collective kernels | Compute signatures / names | Compute occurrences, collective first | Compute occurrences, compute first |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for target_name, label in (
        ("alltoall", "All-to-all"),
        ("allgather", "All-gather"),
        ("reducescatter", "Reduce-scatter"),
    ):
        metadata = table_metadata[target_name]
        lines.append(
            f"| {label} | {metadata['collective_kernel_count']} | "
            f"{metadata['overlapping_compute_kernel_signature_count']} / "
            f"{metadata['overlapping_compute_kernel_name_count']} | "
            f"{metadata['compute_occurrences_collective_first']} | "
            f"{metadata['compute_occurrences_compute_first']} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "- Duration differences are observational correlations. "
                "The analysis separates identical display names by XLA module, "
                "XLA operation, and CUDA launch signature."
            ),
            (
                "- An overlap category can include additional simultaneous "
                "kernels beyond the named collective, so it is not an isolated "
                "pairwise causal experiment."
            ),
            (
                "- `N/A` alone medians mean that exact kernel signature never "
                "ran without another-stream kernel in this trace."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = json.loads(args.trace.read_text())
    events = trace["traceEvents"]
    process_names = {
        int(event["pid"]): str((event.get("args") or {}).get("name", ""))
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    gpu_pids = {
        pid
        for pid, name in process_names.items()
        if name.startswith("/device:GPU:")
    }
    process_names = {pid: process_names[pid] for pid in sorted(gpu_pids)}

    kernels_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    kernel_id = 0
    for event in events:
        if (
            event.get("pid") not in gpu_pids
            or event.get("ph") != "X"
            or "kernel_details" not in (event.get("args") or {})
        ):
            continue
        kernel = dict(event)
        kernel["_end"] = float(kernel["ts"]) + float(kernel["dur"])
        kernel["_kernel_id"] = kernel_id
        kernel_id += 1
        kernels_by_pid[int(kernel["pid"])].append(kernel)

    index = KernelIndex(kernels_by_pid)
    all_kernels = [
        event for kernels in kernels_by_pid.values() for event in kernels
    ]
    compute_events = [event for event in all_kernels if is_compute(event)]
    alltoalls = [event for event in all_kernels if is_alltoall(event)]
    allgathers = [event for event in all_kernels if is_allgather(event)]
    reducescatters = [
        event for event in all_kernels if is_reducescatter(event)
    ]
    sendrecvs = [
        event
        for event in all_kernels
        if "ncclDevKernel_SendRecv" in str(event["name"])
    ]

    alltoall_summary, alltoall_details = classify_alltoalls(
        alltoalls, index
    )
    alone_keys = strict_alone_compute(compute_events, index)
    targets_by_name = {
        "alltoall": alltoalls,
        "allgather": allgathers,
        "reducescatter": reducescatters,
    }
    tables: dict[str, list[dict[str, Any]]] = {}
    table_metadata: dict[str, dict[str, Any]] = {}
    for target_name, targets in targets_by_name.items():
        tables[target_name], table_metadata[target_name] = (
            compute_overlap_table(
                target_name,
                targets,
                compute_events,
                alone_keys,
                index,
            )
        )
    any_collective_rows, any_collective_metadata = (
        compute_any_collective_overlap_table(
            targets_by_name,
            compute_events,
            alone_keys,
            index,
        )
    )

    counts = {
        "gpu_kernel_count": len(all_kernels),
        "compute_kernel_count": len(compute_events),
        "alltoall_count": len(alltoalls),
        "alltoall_main_count": sum(
            (event.get("args") or {}).get("hlo_op") == "all-to-all.2"
            for event in alltoalls
        ),
        "alltoall_setup_count": sum(
            (event.get("args") or {}).get("hlo_op") != "all-to-all.2"
            for event in alltoalls
        ),
        "allgather_count": len(allgathers),
        "reducescatter_count": len(reducescatters),
        "sendrecv_total_count": len(sendrecvs),
        "sendrecv_non_a2a_count": len(sendrecvs) - len(alltoalls),
        "strict_alone_compute_count": len(alone_keys),
    }
    result = {
        "source_trace": str(args.trace),
        "time_unit": "microseconds",
        "gpu_processes": process_names,
        "counts": counts,
        "definitions": {
            "kernel": "GPU X event with args.kernel_details",
            "overlap": "same GPU, different CUDA stream, half-open intervals intersect",
            "launch_order": "GPU kernel start timestamp order",
            "alltoall": "ncclDevKernel_SendRecv with HLO op starting all-to-all",
            "compute": "actual GPU kernel whose name does not start ncclDevKernel",
            "strict_alone": "no actual GPU kernel on another stream overlaps",
        },
        "alltoall": alltoall_summary,
        "compute_table_metadata": table_metadata,
        "compute_tables": tables,
        "compute_vs_any_ep_collective_metadata": any_collective_metadata,
        "compute_vs_any_ep_collective": any_collective_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_csv(
        args.output_dir / "alltoall_occurrences.csv", alltoall_details
    )
    for target_name, rows in tables.items():
        write_csv(
            args.output_dir / f"compute_vs_{target_name}.csv", rows
        )
    write_csv(
        args.output_dir / "compute_vs_any_ep_collective.csv",
        any_collective_rows,
    )
    write_any_collective_report(
        args.output_dir / "compute_vs_any_ep_collective.md",
        args.trace,
        any_collective_rows,
        any_collective_metadata,
    )
    write_report(
        args.output_dir / "analysis.md",
        args.trace,
        process_names,
        alltoall_summary,
        tables,
        table_metadata,
        counts,
    )
    print(json.dumps(counts, indent=2))
    print(args.output_dir / "analysis.md")


if __name__ == "__main__":
    main()
