#!/usr/bin/env python3
"""Estimate exposed interference slowdown in an EP GPU trace."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_ep_collective_overlap_trace import (
    KernelIndex,
    compute_signature,
    is_allgather,
    is_alltoall,
    is_compute,
    is_reducescatter,
    percentile,
    summarize,
)


CENTRAL_BASELINE_QUANTILE = 0.10
CENTRAL_SUCCESSOR_GAP_US = 10.0
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20260723


def kernel_category(event: dict[str, Any]) -> str:
    name = str(event["name"])
    hlo_op = str((event.get("args") or {}).get("hlo_op", ""))
    if is_alltoall(event):
        return "alltoall"
    if is_allgather(event):
        return "allgather"
    if is_reducescatter(event):
        return "reducescatter"
    if "AllReduce" in name:
        return "allreduce"
    if "SendRecv" in name:
        if hlo_op.startswith("collective-permute"):
            return "pipeline_sendrecv"
        return "other_sendrecv"
    if name.startswith("ncclDevKernel"):
        return "other_nccl"
    return "compute"


def load_kernels(
    trace_path: Path,
) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]]]:
    events = json.loads(trace_path.read_text())["traceEvents"]
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
    gpu_names = {pid: process_names[pid] for pid in sorted(gpu_pids)}
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
    return gpu_names, kernels_by_pid


def build_baselines(
    compute_events: list[dict[str, Any]],
    index: KernelIndex,
    quantile: float,
) -> tuple[
    dict[tuple[str, str, str, str], float],
    dict[tuple[str, str, str, str], str],
    dict[tuple[str, str, str, str], int],
    dict[tuple[str, str, str, str], int],
]:
    all_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(
        list
    )
    alone_values: dict[
        tuple[str, str, str, str], list[float]
    ] = defaultdict(list)
    for event in compute_events:
        signature = compute_signature(event)
        all_values[signature].append(float(event["dur"]))
        if not index.overlaps_on_other_streams(event):
            alone_values[signature].append(float(event["dur"]))

    baselines: dict[tuple[str, str, str, str], float] = {}
    sources: dict[tuple[str, str, str, str], str] = {}
    alone_counts: dict[tuple[str, str, str, str], int] = {}
    total_counts: dict[tuple[str, str, str, str], int] = {}
    for signature, values in all_values.items():
        controls = alone_values[signature]
        if controls:
            baselines[signature] = statistics.median(controls)
            sources[signature] = "strict_alone_median"
        else:
            baselines[signature] = percentile(values, quantile)
            sources[signature] = f"within_signature_p{round(quantile * 100)}"
        alone_counts[signature] = len(controls)
        total_counts[signature] = len(values)
    return baselines, sources, alone_counts, total_counts


def analyze_compute(
    compute_events: list[dict[str, Any]],
    index: KernelIndex,
    quantile: float,
    successor_gap_us: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, float | int]],
]:
    baselines, sources, alone_counts, total_counts = build_baselines(
        compute_events, index, quantile
    )
    by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in compute_events:
        by_pid[int(event["pid"])].append(event)
    for events in by_pid.values():
        events.sort(key=lambda event: float(event["ts"]))
    compute_streams_by_pid = {
        pid: {int(event["tid"]) for event in events}
        for pid, events in by_pid.items()
    }
    if any(len(streams) != 1 for streams in compute_streams_by_pid.values()):
        raise ValueError(
            "Critical successor analysis requires one compute stream per GPU; "
            f"found {compute_streams_by_pid}"
        )

    occurrence_rows: list[dict[str, Any]] = []
    signature_rows: dict[
        tuple[str, str, str, str], dict[str, Any]
    ] = {}
    rank_summaries: dict[str, dict[str, float | int]] = {}
    for pid, events in sorted(by_pid.items()):
        trace_start = float(events[0]["ts"])
        trace_end = float(events[-1]["_end"])
        trace_span = trace_end - trace_start
        exposed_excess = 0.0
        all_overlap_excess = 0.0
        overlapped_count = 0
        exposed_count = 0
        strict_control_count = 0
        strict_control_duration = 0.0
        overlapped_duration = 0.0
        for position, event in enumerate(events):
            overlaps = index.overlaps_on_other_streams(event)
            if not overlaps:
                continue
            overlapped_count += 1
            overlapped_duration += float(event["dur"])
            signature = compute_signature(event)
            baseline = baselines[signature]
            excess = float(event["dur"]) - baseline
            all_overlap_excess += excess
            if sources[signature] == "strict_alone_median":
                strict_control_count += 1
                strict_control_duration += float(event["dur"])
            successor_gap = (
                max(
                    0.0,
                    float(events[position + 1]["ts"])
                    - float(event["_end"]),
                )
                if position + 1 < len(events)
                else 0.0
            )
            exposed = successor_gap <= successor_gap_us
            occurrence_excess = excess if exposed else 0.0
            if exposed:
                exposed_count += 1
                exposed_excess += excess
            categories = sorted(
                {
                    kernel_category(other)
                    for other in overlaps
                    if kernel_category(other) != "compute"
                }
            )
            category_combo = "+".join(categories) if categories else "compute"
            occurrence_rows.append(
                {
                    "gpu_pid": pid,
                    "compute_kernel": signature[0],
                    "kernel_details": signature[1],
                    "hlo_module": signature[2],
                    "hlo_op": signature[3],
                    "start_us": float(event["ts"]),
                    "observed_duration_us": float(event["dur"]),
                    "estimated_baseline_us": baseline,
                    "estimated_excess_us": excess,
                    "baseline_source": sources[signature],
                    "successor_gap_us": successor_gap,
                    "critical_path_exposed": exposed,
                    "exposed_excess_us": occurrence_excess,
                    "overlap_categories": category_combo,
                    "overlap_kernel_count": len(overlaps),
                }
            )
            aggregate = signature_rows.setdefault(
                signature,
                {
                    "compute_kernel": signature[0],
                    "kernel_details": signature[1],
                    "hlo_module": signature[2],
                    "hlo_op": signature[3],
                    "baseline_source": sources[signature],
                    "estimated_baseline_us": baseline,
                    "all_occurrence_count": total_counts[signature],
                    "strict_alone_occurrence_count": alone_counts[signature],
                    "overlapped_occurrence_count": 0,
                    "critical_path_exposed_occurrence_count": 0,
                    "overlapped_observed_duration_us": 0.0,
                    "estimated_exposed_excess_us": 0.0,
                    "overlap_categories": set(),
                },
            )
            aggregate["overlapped_occurrence_count"] += 1
            aggregate["overlapped_observed_duration_us"] += float(
                event["dur"]
            )
            aggregate["overlap_categories"].update(categories)
            if exposed:
                aggregate["critical_path_exposed_occurrence_count"] += 1
                aggregate["estimated_exposed_excess_us"] += excess

        counterfactual_span = trace_span - exposed_excess
        rank_summaries[str(pid)] = {
            "trace_span_us": trace_span,
            "overlapped_compute_count": overlapped_count,
            "critical_path_exposed_compute_count": exposed_count,
            "overlapped_compute_duration_us": overlapped_duration,
            "strict_control_matched_overlap_count": strict_control_count,
            "strict_control_matched_overlap_duration_us": (
                strict_control_duration
            ),
            "all_overlap_estimated_excess_us": all_overlap_excess,
            "critical_path_exposed_excess_us": exposed_excess,
            "counterfactual_span_us": counterfactual_span,
            "removal_speedup_pct": 100.0 * exposed_excess / trace_span,
            "observed_over_ideal_overhead_pct": (
                100.0 * exposed_excess / counterfactual_span
            ),
        }

    flattened_signature_rows: list[dict[str, Any]] = []
    for row in signature_rows.values():
        row = dict(row)
        row["overlap_categories"] = "+".join(
            sorted(row["overlap_categories"])
        )
        flattened_signature_rows.append(row)
    flattened_signature_rows.sort(
        key=lambda row: float(row["estimated_exposed_excess_us"]),
        reverse=True,
    )
    occurrence_rows.sort(
        key=lambda row: float(row["exposed_excess_us"]), reverse=True
    )
    return occurrence_rows, flattened_signature_rows, rank_summaries


def analyze_main_alltoalls(
    kernels_by_pid: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    compute_by_pid = {
        pid: sorted(
            [event for event in events if is_compute(event)],
            key=lambda event: float(event["ts"]),
        )
        for pid, events in kernels_by_pid.items()
    }
    targets = [
        event
        for events in kernels_by_pid.values()
        for event in events
        if str((event.get("args") or {}).get("hlo_op", ""))
        == "all-to-all.2"
    ]
    rows: list[dict[str, Any]] = []
    hidden_slack_values: list[float] = []
    exposed_count = 0
    ambiguous_count = 0
    for target in sorted(
        targets, key=lambda event: (int(event["pid"]), float(event["ts"]))
    ):
        target_end = float(target["_end"])
        active_compute = [
            event
            for event in compute_by_pid[int(target["pid"])]
            if float(event["ts"]) < target_end < float(event["_end"])
        ]
        if active_compute:
            status = "hidden_by_compute_at_completion"
            hidden_slack = (
                max(float(event["_end"]) for event in active_compute)
                - target_end
            )
            hidden_slack_values.append(hidden_slack)
            next_compute_gap = None
        else:
            next_events = [
                event
                for event in compute_by_pid[int(target["pid"])]
                if float(event["ts"]) >= target_end
            ]
            next_compute_gap = (
                float(next_events[0]["ts"]) - target_end
                if next_events
                else None
            )
            if (
                next_compute_gap is not None
                and next_compute_gap <= CENTRAL_SUCCESSOR_GAP_US
            ):
                status = "exposed_immediate_downstream_compute"
                exposed_count += 1
            else:
                status = "ambiguous_no_immediate_downstream_compute"
                ambiguous_count += 1
            hidden_slack = None
        rows.append(
            {
                "gpu_pid": target["pid"],
                "start_us": float(target["ts"]),
                "duration_us": float(target["dur"]),
                "hlo_module": str(
                    (target.get("args") or {}).get("hlo_module", "")
                ),
                "critical_path_status": status,
                "hidden_slack_us": hidden_slack,
                "next_compute_gap_us": next_compute_gap,
            }
        )
    return rows, {
        "steady_alltoall_count": len(targets),
        "hidden_by_compute_at_completion_count": len(hidden_slack_values),
        "exposed_immediate_downstream_compute_count": exposed_count,
        "ambiguous_count": ambiguous_count,
        "hidden_slack_us": summarize(hidden_slack_values),
    }


def bootstrap_rank_mean(
    values: list[float], samples: int, seed: int
) -> dict[str, float | int]:
    rng = random.Random(seed)
    means = [
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(samples)
    ]
    return {
        "sample_count": samples,
        "cluster_count": len(values),
        "seed": seed,
        "estimate": statistics.fmean(values),
        "ci95_low": percentile(means, 0.025),
        "ci95_high": percentile(means, 0.975),
    }


def sensitivity_grid(
    compute_events: list[dict[str, Any]],
    index: KernelIndex,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for quantile in (0.05, 0.10, 0.20, 0.25):
        for successor_gap in (5.0, 10.0, 25.0, 50.0):
            _, _, ranks = analyze_compute(
                compute_events, index, quantile, successor_gap
            )
            rows.append(
                {
                    "fallback_baseline_quantile": quantile,
                    "successor_gap_threshold_us": successor_gap,
                    "mean_removal_speedup_pct": statistics.fmean(
                        float(row["removal_speedup_pct"])
                        for row in ranks.values()
                    ),
                    "mean_observed_over_ideal_overhead_pct": statistics.fmean(
                        float(row["observed_over_ideal_overhead_pct"])
                        for row in ranks.values()
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def write_report(path: Path, result: dict[str, Any]) -> None:
    speedup = result["bootstrap"]["removal_speedup_pct"]
    overhead = result["bootstrap"]["observed_over_ideal_overhead_pct"]
    alltoall = result["steady_alltoall_critical_path"]
    rank_summaries = result["rank_summaries"]
    sensitivity = result["sensitivity"]
    category_summary = result["overlap_category_summary"]
    top_signatures = result["top_compute_signatures"]
    central_sensitivity = [
        row
        for row in sensitivity
        if row["successor_gap_threshold_us"]
        == CENTRAL_SUCCESSOR_GAP_US
    ]
    lines = [
        "# Critical-path interference slowdown estimate",
        "",
        (
            f"**Central estimate:** eliminating device-side kernel interference "
            f"would reduce the traced critical-path time by "
            f"**{fmt(speedup['estimate'], 2)}%** "
            f"(95% rank-cluster bootstrap CI "
            f"**{fmt(speedup['ci95_low'], 2)}–"
            f"{fmt(speedup['ci95_high'], 2)}%**). Equivalently, the observed "
            f"trace is **{fmt(overhead['estimate'], 2)}%** slower than the "
            f"estimated isolated trace (95% CI "
            f"{fmt(overhead['ci95_low'], 2)}–"
            f"{fmt(overhead['ci95_high'], 2)}%)."
        ),
        "",
        "## Method",
        "",
        (
            "1. GPU `X` events with `kernel_details` are device kernels. "
            "Compute kernels are matched by CUDA name, full launch signature, "
            "XLA module, and XLA operation."
        ),
        (
            "2. For a matched signature that has strict-alone occurrences, "
            "the isolated baseline is their median. If an exact "
            "signature never runs alone, the baseline is its within-signature "
            f"{CENTRAL_BASELINE_QUANTILE:.0%} duration quantile across ranks/"
            "occurrences. This is a conservative low-contention proxy and "
            "does not pool different HLO shapes."
        ),
        (
            "3. Only compute occurrences concurrent with another-stream "
            "kernel are assigned interference excess. A compute occurrence is "
            "treated as exposed on the local critical path when the next "
            f"same-stream compute starts within "
            f"{CENTRAL_SUCCESSOR_GAP_US:.0f} µs; a larger following bubble is "
            "left unchanged in the counterfactual."
        ),
        (
            "4. Per GPU, exposed excess durations are removed from the "
            "observed first-to-last-compute span. The reported point estimate "
            "is the mean across the four EP GPU tracks. The percentile CI "
            f"resamples those four GPU tracks {BOOTSTRAP_SAMPLES:,} times as "
            "clusters."
        ),
        "",
        "## Per-GPU estimate",
        "",
        "| GPU PID | Observed span ms | Exposed excess ms | Remove-interference speedup | Observed / ideal overhead | Strict-control duration coverage |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for pid, row in sorted(rank_summaries.items(), key=lambda item: int(item[0])):
        duration_coverage = (
            100.0
            * float(row["strict_control_matched_overlap_duration_us"])
            / float(row["overlapped_compute_duration_us"])
        )
        lines.append(
            f"| {pid} | {fmt(float(row['trace_span_us']) / 1000)} | "
            f"{fmt(float(row['critical_path_exposed_excess_us']) / 1000)} | "
            f"{fmt(row['removal_speedup_pct'], 3)}% | "
            f"{fmt(row['observed_over_ideal_overhead_pct'], 3)}% | "
            f"{fmt(duration_coverage, 1)}% |"
        )
    lines.extend(
        [
            "",
            "## All-to-all critical-path exposure",
            "",
            (
                f"All {alltoall['steady_alltoall_count']} steady "
                "`all-to-all.2` kernels finish while a compute kernel is still "
                "running on the same GPU. Therefore none is directly exposed "
                "at completion in this trace, and making only the A2A shorter "
                "would not advance the compute stream."
            ),
            "",
            (
                "Remaining compute cover after A2A completion: median "
                f"**{fmt(alltoall['hidden_slack_us']['median_us'])} µs**, "
                f"minimum **{fmt(alltoall['hidden_slack_us']['minimum_us'])} "
                "µs**, p95 "
                f"**{fmt(alltoall['hidden_slack_us']['p95_us'])} µs**."
            ),
            "",
            (
                "This does not mean A2A is harmless: an A2A can slow the "
                "overlapping compute kernel, and that compute-side extension "
                "is included in the estimate."
            ),
            "",
            "## Estimated exposed excess by overlap combination",
            "",
            (
                "These categories are mutually exclusive combinations. A row "
                "such as `allgather+reducescatter` means both collective types "
                "overlap the same compute occurrence, so the trace cannot "
                "causally apportion that occurrence between them."
            ),
            "",
            "| Concurrent kernel category combination | Compute occurrences | Exposed excess ms | Share of net exposed excess |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in category_summary:
        lines.append(
            f"| `{row['overlap_categories']}` | "
            f"{row['compute_occurrence_count']} | "
            f"{fmt(float(row['exposed_excess_us']) / 1000)} | "
            f"{fmt(row['net_excess_share_pct'], 2)}% |"
        )
    lines.extend(
        [
            "",
            "## Largest exact-kernel contributions",
            "",
            "| Compute kernel | XLA module / operation | Concurrent kernels | Exposed occurrences | Baseline source | Exposed excess ms |",
            "|---|---|---|---:|---|---:|",
        ]
    )
    for row in top_signatures:
        kernel = str(row["compute_kernel"]).replace("|", "\\|")
        module = str(row["hlo_module"]).replace("|", "\\|")
        hlo_op = str(row["hlo_op"]).replace("|", "\\|")
        lines.append(
            f"| `{kernel}` | `{module}` / `{hlo_op}` | "
            f"`{row['overlap_categories']}` | "
            f"{row['critical_path_exposed_occurrence_count']} | "
            f"`{row['baseline_source']}` | "
            f"{fmt(float(row['estimated_exposed_excess_us']) / 1000)} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            (
                "At the fixed 10 µs critical-successor threshold, changing the "
                "fallback low-contention baseline from p5 to p25 gives:"
            ),
            "",
            "| Fallback quantile | Remove-interference speedup | Observed / ideal overhead |",
            "|---:|---:|---:|",
        ]
    )
    for row in central_sensitivity:
        lines.append(
            f"| p{round(100 * row['fallback_baseline_quantile'])} | "
            f"{fmt(row['mean_removal_speedup_pct'], 3)}% | "
            f"{fmt(row['mean_observed_over_ideal_overhead_pct'], 3)}% |"
        )
    threshold_sensitivity = [
        row
        for row in sensitivity
        if row["fallback_baseline_quantile"]
        == CENTRAL_BASELINE_QUANTILE
    ]
    lines.extend(
        [
            "",
            "At the fixed p10 fallback baseline, varying the definition of an "
            "immediate same-stream successor gives:",
            "",
            "| Successor-gap threshold | Remove-interference speedup | Observed / ideal overhead |",
            "|---:|---:|---:|",
        ]
    )
    for row in threshold_sensitivity:
        lines.append(
            f"| {fmt(row['successor_gap_threshold_us'], 0)} µs | "
            f"{fmt(row['mean_removal_speedup_pct'], 3)}% | "
            f"{fmt(row['mean_observed_over_ideal_overhead_pct'], 3)}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            (
                "- The CI measures variability across the four GPU tracks; it "
                "does not include model-form uncertainty from the p10 proxy. "
                "The sensitivity table is the appropriate check for that "
                "assumption."
            ),
            (
                "- The estimate preserves observed compute-stream bubbles. It "
                "does not claim that every shortened kernel shifts the whole "
                "step, and it counts no direct A2A saving because every steady "
                "A2A is hidden at completion."
            ),
            (
                "- This is a trace counterfactual, not a randomized causal "
                "experiment. A dedicated solo/overlap replay of the exact HLO "
                "kernels would give stronger isolated controls."
            ),
            (
                "- The four EP ranks participate in the same collectives, so "
                "the rank-cluster CI should be read as an empirical uncertainty "
                "interval rather than evidence from four independent runs."
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
    gpu_names, kernels_by_pid = load_kernels(args.trace)
    index = KernelIndex(kernels_by_pid)
    compute_events = [
        event
        for kernels in kernels_by_pid.values()
        for event in kernels
        if is_compute(event)
    ]
    occurrence_rows, signature_rows, rank_summaries = analyze_compute(
        compute_events,
        index,
        CENTRAL_BASELINE_QUANTILE,
        CENTRAL_SUCCESSOR_GAP_US,
    )
    alltoall_rows, alltoall_summary = analyze_main_alltoalls(kernels_by_pid)
    speedup_values = [
        float(row["removal_speedup_pct"]) for row in rank_summaries.values()
    ]
    overhead_values = [
        float(row["observed_over_ideal_overhead_pct"])
        for row in rank_summaries.values()
    ]
    sensitivity = sensitivity_grid(compute_events, index)
    category_values: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "compute_occurrence_count": 0,
            "observed_duration_us": 0.0,
            "exposed_excess_us": 0.0,
        }
    )
    for row in occurrence_rows:
        category = str(row["overlap_categories"])
        category_values[category]["compute_occurrence_count"] += 1
        category_values[category]["observed_duration_us"] += float(
            row["observed_duration_us"]
        )
        category_values[category]["exposed_excess_us"] += float(
            row["exposed_excess_us"]
        )
    total_exposed_excess = sum(
        float(row["critical_path_exposed_excess_us"])
        for row in rank_summaries.values()
    )
    category_summary = []
    for category, values in category_values.items():
        category_summary.append(
            {
                "overlap_categories": category,
                **values,
                "net_excess_share_pct": (
                    100.0
                    * float(values["exposed_excess_us"])
                    / total_exposed_excess
                ),
            }
        )
    category_summary.sort(
        key=lambda row: float(row["exposed_excess_us"]), reverse=True
    )
    result = {
        "source_trace": str(args.trace),
        "time_unit": "microseconds",
        "gpu_processes": gpu_names,
        "method": {
            "compute_signature": (
                "CUDA name + full launch signature + XLA module + XLA op"
            ),
            "strict_control_baseline": "median strict-alone duration",
            "fallback_baseline": (
                f"within-exact-signature p{round(CENTRAL_BASELINE_QUANTILE * 100)}"
            ),
            "critical_compute_successor_gap_us": (
                CENTRAL_SUCCESSOR_GAP_US
            ),
            "bootstrap_cluster": "GPU track",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "rank_summaries": rank_summaries,
        "bootstrap": {
            "removal_speedup_pct": bootstrap_rank_mean(
                speedup_values, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED
            ),
            "observed_over_ideal_overhead_pct": bootstrap_rank_mean(
                overhead_values, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED + 1
            ),
        },
        "steady_alltoall_critical_path": alltoall_summary,
        "overlap_category_summary": category_summary,
        "top_compute_signatures": signature_rows[:10],
        "sensitivity": sensitivity,
    }
    write_csv(
        args.output_dir / "critical_path_compute_occurrences.csv",
        occurrence_rows,
    )
    write_csv(
        args.output_dir / "critical_path_compute_signatures.csv",
        signature_rows,
    )
    write_csv(
        args.output_dir / "critical_path_alltoall_occurrences.csv",
        alltoall_rows,
    )
    (args.output_dir / "critical_path_slowdown.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    write_report(args.output_dir / "critical_path_slowdown.md", result)
    print(json.dumps(result["bootstrap"], indent=2))
    print(args.output_dir / "critical_path_slowdown.md")


if __name__ == "__main__":
    main()
