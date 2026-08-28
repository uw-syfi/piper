#!/usr/bin/env python3
"""Attribute per-node normal-profile slowdowns using PyTorch CPU/GPU traces.

This is an offline profile-and-replay diagnostic.

The profile ratio artifact contains CUDA-event timings from the normal and
isolated profiling passes.  The PyTorch traces capture normal
execution only.  This script correlates every labeled device activity in the
normal trace back to its CPU runtime launch and enclosing Piper NVTX range,
then decomposes the normal device span into associated GPU activity, device
gaps, CPU-launch-exposed gaps, and otherwise unexplained time.

Because there is no isolated PyTorch trace, attribution of GPU
activity-duration inflation is conservative: the normal trace's associated
GPU activity is compared with the *entire* isolated node duration.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}

OUTPUT_FIELDS = (
    "node_uid",
    "median_normal_time_us",
    "median_isolated_time_us",
    "normal_to_isolated_ratio",
    "normal_minus_isolated_us",
    "trace_task_type",
    "trace_cpu_occurrence_count",
    "trace_gpu_occurrence_count",
    "trace_gpu_match_fraction",
    "trace_unmatched_device_activity_count",
    "median_cpu_node_duration_us",
    "median_cpu_launch_envelope_us",
    "median_cpu_interlaunch_gap_us",
    "fraction_all_launches_before_first_gpu",
    "median_last_launch_to_first_gpu_us",
    "median_activity_launch_lead_us",
    "median_gpu_activity_launch_match_fraction",
    "median_gpu_span_us",
    "median_gpu_activity_union_us",
    "median_gpu_kernel_union_us",
    "median_gpu_kernel_sum_us",
    "median_gpu_memcpy_memset_sum_us",
    "median_device_gpu_gap_us",
    "median_cpu_exposed_gpu_gap_us",
    "median_other_gpu_overlap_in_gap_us",
    "median_other_gpu_overlap_in_span_us",
    "gpu_gap_other_overlap_fraction",
    "normal_profile_to_trace_gpu_span_ratio",
    "normal_trace_gpu_span_relative_error",
    "normal_profile_minus_trace_gpu_span_us",
    "inferred_profile_edge_gap_us",
    "total_gpu_gap_budget_us",
    "gpu_activity_excess_over_isolated_total_us",
    "profile_delta_unexplained_by_trace_us",
    "fraction_occurrences_cpu_gap_ge_60pct_delta",
    "primary_interference_attribution",
    "associated_activity_interpretation",
    "gpu_gap_mechanism",
    "attribution_confidence",
    "trace_status",
    "overlapping_node_types",
    "analysis",
)


@dataclass(frozen=True)
class RuntimeLaunch:
    pid: int
    ts: float
    dur: float
    name: str
    correlation: int | str | None
    external_id: int | str | None

    @property
    def end(self) -> float:
        return self.ts + self.dur


@dataclass(frozen=True)
class DeviceActivity:
    event_id: int
    uid: str
    label: str
    task_type: str
    category: str
    pid: int
    stream: int | str | None
    ts: float
    dur: float
    launch: RuntimeLaunch | None

    @property
    def end(self) -> float:
        return self.ts + self.dur


@dataclass
class Occurrence:
    uid: str
    label: str
    task_type: str
    trace_name: str
    cpu_pid: int
    ts: float
    dur: float
    activities: list[DeviceActivity] = field(default_factory=list)

    @property
    def end(self) -> float:
        return self.ts + self.dur


@dataclass
class NodeTraceAggregate:
    task_types: set[str] = field(default_factory=set)
    cpu_durations: list[float] = field(default_factory=list)
    occurrence_metrics: list[dict[str, float | bool]] = field(default_factory=list)
    overlapping_types: set[str] = field(default_factory=set)
    unmatched_device_activity_count: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--ratios",
        type=Path,
        help="Defaults to OUTPUT_DIR/node_profile_time_ratios.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to OUTPUT_DIR/node_profile_interference_analysis.csv",
    )
    return parser.parse_args()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    return float(statistics.median(materialized)) if materialized else None


def _csv_number(value: float | None) -> str | float:
    if value is None or not math.isfinite(value):
        return ""
    return value


def _uid_from_label(label: Any, wanted: set[str]) -> str | None:
    if not isinstance(label, str) or ":uid" not in label:
        return None
    uid = label.rsplit(":uid", 1)[1]
    return uid if uid in wanted else None


def _task_type(label: str) -> str:
    return label.split(":", 1)[0]


def _merge_intervals(
    intervals: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return merged


def _interval_sum(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _gaps(merged: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [
        (left[1], right[0])
        for left, right in zip(merged, merged[1:])
        if right[0] > left[1]
    ]


def _clipped_intersections(
    activities: Iterable[DeviceActivity],
    windows: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    window_list = list(windows)
    intersections: list[tuple[float, float]] = []
    for activity in activities:
        for start, end in window_list:
            clipped_start = max(start, activity.ts)
            clipped_end = min(end, activity.end)
            if clipped_end > clipped_start:
                intersections.append((clipped_start, clipped_end))
    return intersections


def _runtime_name_matches(category: str, name: str) -> bool:
    lowered = name.lower()
    if category == "kernel":
        return "launchkernel" in lowered
    if category == "gpu_memcpy":
        return "memcpy" in lowered
    if category == "gpu_memset":
        return "memset" in lowered
    return False


def _find_enclosing_occurrence(
    occurrences: list[Occurrence], launch_ts: float
) -> Occurrence | None:
    if not occurrences:
        return None
    starts = [occurrence.ts for occurrence in occurrences]
    index = bisect.bisect_right(starts, launch_ts) - 1
    if index < 0:
        return None
    occurrence = occurrences[index]
    if occurrence.ts <= launch_ts <= occurrence.end:
        return occurrence
    return None


def _find_launch(
    candidates: list[RuntimeLaunch], category: str, device_ts: float
) -> RuntimeLaunch | None:
    matching = [
        launch
        for launch in candidates
        if launch.ts <= device_ts and _runtime_name_matches(category, launch.name)
    ]
    return max(matching, key=lambda launch: launch.ts) if matching else None


def _cpu_exposed_gap_us(activities: list[DeviceActivity]) -> float:
    """Return device-gap time for which the next launch was not yet submitted."""

    exposed = 0.0
    covered_end: float | None = None
    for activity in sorted(activities, key=lambda value: (value.ts, value.end)):
        if covered_end is None:
            covered_end = activity.end
            continue
        if activity.ts > covered_end and activity.launch is not None:
            exposed_end = min(activity.ts, activity.launch.end)
            if exposed_end > covered_end:
                exposed += exposed_end - covered_end
        covered_end = max(covered_end, activity.end)
    return exposed


def _occurrence_metrics(
    occurrence: Occurrence,
    all_activities: list[DeviceActivity],
    all_activity_starts: list[float],
    max_activity_duration: float,
) -> tuple[dict[str, float | bool] | None, set[str]]:
    own = sorted(occurrence.activities, key=lambda value: (value.ts, value.end))
    if not own:
        return None, set()

    own_intervals = _merge_intervals((activity.ts, activity.end) for activity in own)
    span_start = own_intervals[0][0]
    span_end = own_intervals[-1][1]
    span = span_end - span_start
    activity_union = sum(end - start for start, end in own_intervals)
    gap_windows = _gaps(own_intervals)
    gpu_gap = sum(end - start for start, end in gap_windows)

    own_ids = {activity.event_id for activity in own}
    lower = bisect.bisect_left(
        all_activity_starts, span_start - max_activity_duration
    )
    upper = bisect.bisect_left(all_activity_starts, span_end)
    others = [
        activity
        for activity in all_activities[lower:upper]
        if activity.event_id not in own_ids
        and activity.end > span_start
        and activity.ts < span_end
    ]
    span_overlap = _interval_sum(
        _clipped_intersections(others, [(span_start, span_end)])
    )
    gap_overlap = _interval_sum(_clipped_intersections(others, gap_windows))
    overlapping_types = {activity.task_type for activity in others}

    launches = {
        (
            activity.launch.pid,
            activity.launch.ts,
            activity.launch.dur,
            activity.launch.name,
            activity.launch.correlation,
            activity.launch.external_id,
        ): activity.launch
        for activity in own
        if activity.launch is not None
    }
    ordered_launches = sorted(launches.values(), key=lambda value: value.ts)
    launch_envelope = None
    interlaunch_gap = None
    all_before_first = False
    last_launch_to_first = None
    if ordered_launches:
        launch_envelope = ordered_launches[-1].end - ordered_launches[0].ts
        launch_union = _interval_sum(
            (launch.ts, launch.end) for launch in ordered_launches
        )
        interlaunch_gap = launch_envelope - launch_union
        last_launch_end = max(launch.end for launch in ordered_launches)
        all_before_first = last_launch_end <= span_start
        last_launch_to_first = span_start - last_launch_end

    launch_leads = [
        activity.ts - activity.launch.end
        for activity in own
        if activity.launch is not None
    ]
    return (
        {
            "gpu_span_us": span,
            "gpu_activity_union_us": activity_union,
            "gpu_kernel_union_us": _interval_sum(
                (activity.ts, activity.end)
                for activity in own
                if activity.category == "kernel"
            ),
            "gpu_kernel_sum_us": sum(
                activity.dur for activity in own if activity.category == "kernel"
            ),
            "gpu_memcpy_memset_sum_us": sum(
                activity.dur for activity in own if activity.category != "kernel"
            ),
            "device_gpu_gap_us": gpu_gap,
            "cpu_exposed_gpu_gap_us": _cpu_exposed_gap_us(own),
            "other_gpu_overlap_in_gap_us": gap_overlap,
            "other_gpu_overlap_in_span_us": span_overlap,
            "cpu_launch_envelope_us": launch_envelope
            if launch_envelope is not None
            else math.nan,
            "cpu_interlaunch_gap_us": interlaunch_gap
            if interlaunch_gap is not None
            else math.nan,
            "all_launches_before_first_gpu": all_before_first,
            "last_launch_to_first_gpu_us": last_launch_to_first
            if last_launch_to_first is not None
            else math.nan,
            "activity_launch_lead_us": _median(launch_leads)
            if launch_leads
            else math.nan,
            "matched_activity_fraction": (
                len(launch_leads) / len(own) if own else 0.0
            ),
        },
        overlapping_types,
    )


def _finite_metric_median(
    metrics: list[dict[str, float | bool]], key: str
) -> float | None:
    values = [
        float(metric[key])
        for metric in metrics
        if isinstance(metric.get(key), (int, float))
        and math.isfinite(float(metric[key]))
    ]
    return _median(values)


def _analyze_trace(
    trace_path: Path,
    wanted: set[str],
    aggregates: dict[str, NodeTraceAggregate],
) -> None:
    with trace_path.open(encoding="utf-8") as trace_file:
        events = json.load(trace_file).get("traceEvents", [])

    process_names: dict[int, str] = {}
    for event in events:
        if event.get("ph") == "M" and event.get("name") == "process_name":
            process_names[int(event["pid"])] = str((event.get("args") or {}).get("name", ""))

    occurrences: list[Occurrence] = []
    occurrences_by_key: dict[tuple[int, str], list[Occurrence]] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "user_annotation":
            continue
        label = event.get("name")
        uid = _uid_from_label(label, wanted)
        if uid is None:
            continue
        occurrence = Occurrence(
            uid=uid,
            label=str(label),
            task_type=_task_type(str(label)),
            trace_name=trace_path.name,
            cpu_pid=int(event["pid"]),
            ts=_as_float(event.get("ts")),
            dur=_as_float(event.get("dur")),
        )
        occurrences.append(occurrence)
        occurrences_by_key.setdefault((occurrence.cpu_pid, uid), []).append(occurrence)
    for values in occurrences_by_key.values():
        values.sort(key=lambda occurrence: occurrence.ts)

    cpu_pids_by_process: dict[str, set[int]] = {}
    for occurrence in occurrences:
        process_name = process_names.get(occurrence.cpu_pid, "")
        cpu_pids_by_process.setdefault(process_name, set()).add(occurrence.cpu_pid)

    runtime_by_key: dict[
        tuple[int, int | str | None, int | str | None], list[RuntimeLaunch]
    ] = {}
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {
            "cuda_driver",
            "cuda_runtime",
        }:
            continue
        args = event.get("args") or {}
        correlation = args.get("correlation")
        if correlation is None:
            continue
        launch = RuntimeLaunch(
            pid=int(event["pid"]),
            ts=_as_float(event.get("ts")),
            dur=_as_float(event.get("dur")),
            name=str(event.get("name", "")),
            correlation=correlation,
            external_id=args.get("External id"),
        )
        runtime_by_key.setdefault(
            (launch.pid, launch.correlation, launch.external_id), []
        ).append(launch)

    all_activities_by_gpu: dict[int, list[DeviceActivity]] = {}
    event_id = 0
    for event in events:
        category = str(event.get("cat", ""))
        if event.get("ph") != "X" or category not in DEVICE_CATEGORIES:
            continue
        args = event.get("args") or {}
        label = args.get("piper_dag_node")
        uid = _uid_from_label(label, wanted)
        if uid is None:
            continue
        gpu_pid = int(event["pid"])
        process_name = process_names.get(gpu_pid, "")
        cpu_candidates = cpu_pids_by_process.get(process_name, set())
        if len(cpu_candidates) == 1:
            cpu_pid = next(iter(cpu_candidates))
        elif gpu_pid - 2 in {pid for pids in cpu_pids_by_process.values() for pid in pids}:
            cpu_pid = gpu_pid - 2
        else:
            cpu_pid = -1

        candidates = runtime_by_key.get(
            (cpu_pid, args.get("correlation"), args.get("External id")), []
        )
        launch = _find_launch(candidates, category, _as_float(event.get("ts")))
        activity = DeviceActivity(
            event_id=event_id,
            uid=uid,
            label=str(label),
            task_type=_task_type(str(label)),
            category=category,
            pid=gpu_pid,
            stream=args.get("stream", event.get("tid")),
            ts=_as_float(event.get("ts")),
            dur=_as_float(event.get("dur")),
            launch=launch,
        )
        event_id += 1
        all_activities_by_gpu.setdefault(gpu_pid, []).append(activity)

        occurrence = (
            _find_enclosing_occurrence(
                occurrences_by_key.get((cpu_pid, uid), []), launch.ts
            )
            if launch is not None
            else None
        )
        if occurrence is not None:
            occurrence.activities.append(activity)
        else:
            aggregates[uid].unmatched_device_activity_count += 1

    for activities in all_activities_by_gpu.values():
        activities.sort(key=lambda activity: activity.ts)

    for occurrence in occurrences:
        aggregate = aggregates[occurrence.uid]
        aggregate.task_types.add(occurrence.task_type)
        aggregate.cpu_durations.append(occurrence.dur)
        if not occurrence.activities:
            continue
        gpu_pids = {activity.pid for activity in occurrence.activities}
        if len(gpu_pids) != 1:
            # This is not expected for a rank-local Piper node.  Retain the CPU
            # match but do not invent a cross-device span.
            continue
        gpu_pid = next(iter(gpu_pids))
        all_activities = all_activities_by_gpu[gpu_pid]
        starts = [activity.ts for activity in all_activities]
        max_duration = max((activity.dur for activity in all_activities), default=0.0)
        metrics, overlapping_types = _occurrence_metrics(
            occurrence, all_activities, starts, max_duration
        )
        if metrics is not None:
            aggregate.occurrence_metrics.append(metrics)
            aggregate.overlapping_types.update(overlapping_types)

    del events


def _fallback_task_type(uid: str) -> str:
    prefix = uid.split(".", 1)[0]
    if prefix in {"a2a", "recv", "reduce", "send", "update"}:
        return prefix
    if ".bwd" in uid:
        return "backward"
    return "forward"


def _classify(
    normal: float,
    isolated: float,
    aggregate: NodeTraceAggregate,
) -> dict[str, Any]:
    metrics = aggregate.occurrence_metrics
    cpu_count = len(aggregate.cpu_durations)
    gpu_count = len(metrics)
    match_fraction = gpu_count / cpu_count if cpu_count else 0.0
    normal_delta = normal - isolated

    result: dict[str, Any] = {
        "trace_cpu_occurrence_count": cpu_count,
        "trace_gpu_occurrence_count": gpu_count,
        "trace_gpu_match_fraction": match_fraction,
        "median_cpu_node_duration_us": _median(aggregate.cpu_durations),
    }
    metric_columns = {
        "median_cpu_launch_envelope_us": "cpu_launch_envelope_us",
        "median_cpu_interlaunch_gap_us": "cpu_interlaunch_gap_us",
        "median_last_launch_to_first_gpu_us": "last_launch_to_first_gpu_us",
        "median_activity_launch_lead_us": "activity_launch_lead_us",
        "median_gpu_span_us": "gpu_span_us",
        "median_gpu_activity_union_us": "gpu_activity_union_us",
        "median_gpu_kernel_union_us": "gpu_kernel_union_us",
        "median_gpu_kernel_sum_us": "gpu_kernel_sum_us",
        "median_gpu_memcpy_memset_sum_us": "gpu_memcpy_memset_sum_us",
        "median_device_gpu_gap_us": "device_gpu_gap_us",
        "median_cpu_exposed_gpu_gap_us": "cpu_exposed_gpu_gap_us",
        "median_other_gpu_overlap_in_gap_us": "other_gpu_overlap_in_gap_us",
        "median_other_gpu_overlap_in_span_us": "other_gpu_overlap_in_span_us",
        "median_gpu_activity_launch_match_fraction": "matched_activity_fraction",
    }
    for output_name, metric_name in metric_columns.items():
        result[output_name] = _finite_metric_median(metrics, metric_name)
    result["fraction_all_launches_before_first_gpu"] = (
        statistics.fmean(
            1.0 if bool(metric["all_launches_before_first_gpu"]) else 0.0
            for metric in metrics
        )
        if metrics
        else None
    )

    span = result["median_gpu_span_us"]
    work = result["median_gpu_activity_union_us"]
    gap = result["median_device_gpu_gap_us"]
    cpu_exposed = result["median_cpu_exposed_gpu_gap_us"]
    other_gap = result["median_other_gpu_overlap_in_gap_us"]
    gap_overlap_fraction = (
        min(1.0, other_gap / gap)
        if isinstance(gap, float)
        and isinstance(other_gap, float)
        and gap > 0.0
        else None
    )
    profile_to_span = (
        normal / span
        if isinstance(span, float) and span > 0.0
        else None
    )
    activity_excess = (
        max(0.0, work - isolated)
        if isinstance(work, float)
        else None
    )
    profile_minus_span = (
        normal - span if isinstance(span, float) else None
    )
    trace_span_relative_error = (
        abs(span - normal) / normal
        if isinstance(span, float) and normal > 0.0
        else None
    )
    edge_gap = (
        max(0.0, profile_minus_span)
        if isinstance(profile_minus_span, float)
        else None
    )
    gap_budget = (
        gap + edge_gap
        if isinstance(gap, float) and isinstance(edge_gap, float)
        else None
    )
    unexplained = (
        normal_delta - activity_excess - gap_budget
        if isinstance(activity_excess, float) and isinstance(gap_budget, float)
        else None
    )
    cpu_dominant_fraction = (
        statistics.fmean(
            1.0
            if float(metric["cpu_exposed_gpu_gap_us"]) >= 0.60 * normal_delta
            else 0.0
            for metric in metrics
        )
        if metrics and normal_delta > 0.0
        else 0.0
    )
    result.update(
        {
            "gpu_gap_other_overlap_fraction": gap_overlap_fraction,
            "normal_profile_to_trace_gpu_span_ratio": profile_to_span,
            "normal_trace_gpu_span_relative_error": trace_span_relative_error,
            "normal_profile_minus_trace_gpu_span_us": profile_minus_span,
            "inferred_profile_edge_gap_us": edge_gap,
            "total_gpu_gap_budget_us": gap_budget,
            "gpu_activity_excess_over_isolated_total_us": activity_excess,
            "profile_delta_unexplained_by_trace_us": unexplained,
            "fraction_occurrences_cpu_gap_ge_60pct_delta": cpu_dominant_fraction,
        }
    )

    if cpu_count == 0:
        trace_status = "no_piper_cpu_annotation"
    elif gpu_count == 0:
        trace_status = (
            "expected_no_device_activity"
            if normal < 10.0 and isolated < 10.0
            else "cpu_annotation_only"
        )
    elif match_fraction < 0.8:
        trace_status = "partial_gpu_match"
    elif profile_to_span is None:
        trace_status = "matched_no_gpu_span"
    elif isinstance(trace_span_relative_error, float) and trace_span_relative_error <= 0.25:
        trace_status = "matched_profile_agrees"
    else:
        trace_status = "matched_profile_trace_mismatch"

    significant_threshold = max(50.0, 0.05 * isolated)
    if normal_delta < -significant_threshold:
        attribution = "isolated_slower"
    elif normal_delta <= significant_threshold:
        attribution = "no_significant_slowdown"
    elif not isinstance(work, float) or not isinstance(gap, float):
        attribution = "other_or_trace_unmatched"
    else:
        cpu_score = (
            min(
                normal_delta,
                max(0.0, cpu_exposed if isinstance(cpu_exposed, float) else 0.0),
            )
            if cpu_dominant_fraction >= (2.0 / 3.0)
            else 0.0
        )
        device_gap_score = min(
            normal_delta,
            max(0.0, (gap_budget or 0.0) - cpu_score),
        )
        activity_score = min(normal_delta, max(0.0, activity_excess or 0.0))
        residual_score = max(
            0.0, normal_delta - activity_score - cpu_score - device_gap_score
        )
        scores = {
            "associated_gpu_activity_longer": activity_score,
            "device_side_gpu_gaps": device_gap_score,
            "cpu_side_launch_gaps": cpu_score,
            "other_or_unresolved": residual_score,
        }
        fractions = {
            name: score / normal_delta for name, score in scores.items()
        }
        positive = sorted(
            ((fraction, name) for name, fraction in fractions.items() if fraction > 0.0),
            reverse=True,
        )
        if not positive:
            attribution = "other_or_unresolved"
        elif positive[0][0] >= 0.75 and not (
            len(positive) > 1 and positive[1][0] >= 0.25
        ):
            attribution = positive[0][1]
        elif len(positive) >= 2:
            names = {positive[0][1], positive[1][1]}
            if names == {
                "associated_gpu_activity_longer",
                "device_side_gpu_gaps",
            }:
                attribution = "mixed_gpu_activity_and_device_gaps"
            elif "cpu_side_launch_gaps" in names:
                attribution = "mixed_cpu_and_gpu_effects"
            else:
                attribution = "mixed_with_unresolved_time"
        else:
            attribution = positive[0][1]

    if isinstance(gap_budget, float) and gap_budget > 10.0:
        cpu_fraction = (
            min(1.0, cpu_exposed / gap_budget)
            if isinstance(cpu_exposed, float)
            else 0.0
        )
        if cpu_fraction >= 0.5:
            gap_mechanism = "cpu_launch_starvation"
        elif (
            isinstance(gap_overlap_fraction, float)
            and gap_overlap_fraction >= 0.5
            and isinstance(edge_gap, float)
            and edge_gap <= 0.25 * gap_budget
        ):
            gap_mechanism = "concurrent_gpu_work"
        elif (
            isinstance(gap_overlap_fraction, float)
            and gap_overlap_fraction >= 0.5
            and isinstance(edge_gap, float)
            and edge_gap > 0.25 * gap_budget
        ):
            gap_mechanism = "concurrent_gpu_work_plus_unlocated_edge_gap"
        elif isinstance(gap_overlap_fraction, float) and gap_overlap_fraction >= 0.1:
            gap_mechanism = "mixed_overlap_and_device_wait"
        else:
            gap_mechanism = "device_sync_or_unobserved_activity"
    else:
        gap_mechanism = "no_material_gpu_gap"

    agreement_error = (
        trace_span_relative_error
        if isinstance(trace_span_relative_error, float)
        else math.inf
    )
    if gpu_count >= 2 and match_fraction >= 0.8 and agreement_error <= 0.25:
        confidence = "high"
    elif gpu_count >= 1 and match_fraction >= 0.5 and agreement_error <= 0.75:
        confidence = "medium"
    else:
        confidence = "low"

    task_types = aggregate.task_types
    if "recv" in task_types:
        activity_interpretation = (
            "NCCL recv residency, commonly peer-arrival/network wait rather than "
            "compute; the replay collapses the send/recv endpoints, so this raw "
            "recv duration is not itself a compute-stream service time"
        )
    elif "send" in task_types:
        activity_interpretation = (
            "NCCL send residency; the replay collapses the send/recv endpoints "
            "and uses one rendezvous service time"
        )
    elif task_types & {"all_reduce", "all_to_all"}:
        activity_interpretation = (
            "NCCL kernel residency; network, peer-arrival, protocol, and SM effects "
            "are not separable from a normal-only trace"
        )
    elif task_types:
        activity_interpretation = "compute/memory GPU activity duration"
    else:
        activity_interpretation = "no matched GPU activity"

    if isinstance(span, float):
        analysis = (
            f"normal excess={normal_delta:.3f} us; normal trace median="
            f"{work:.3f} us associated GPU activity + {gap:.3f} us internal device "
            f"gaps + {(edge_gap or 0.0):.3f} us unlocated profile-edge gap; "
            f"CPU-launch-exposed gap={cpu_exposed:.3f} us; other GPU work overlaps "
            f"{other_gap:.3f} us of gaps; associated activity exceeds the entire "
            f"isolated node time by {(activity_excess or 0.0):.3f} us; trace/profile "
            f"span ratio={(profile_to_span if profile_to_span is not None else math.nan):.3f}"
        )
    else:
        analysis = (
            f"normal excess={normal_delta:.3f} us; no matched normal GPU activity "
            "was available for decomposition"
        )
    result.update(
        {
            "primary_interference_attribution": attribution,
            "associated_activity_interpretation": activity_interpretation,
            "gpu_gap_mechanism": gap_mechanism,
            "attribution_confidence": confidence,
            "trace_status": trace_status,
            "analysis": analysis,
        }
    )
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    ratio_path = (args.ratios or output_dir / "node_profile_time_ratios.csv").resolve()
    output_path = (
        args.output or output_dir / "node_profile_interference_analysis.csv"
    ).resolve()

    with ratio_path.open(newline="", encoding="utf-8") as ratio_file:
        ratio_rows = list(csv.DictReader(ratio_file))
    wanted = {row["node_uid"] for row in ratio_rows}
    aggregates = {uid: NodeTraceAggregate() for uid in wanted}

    trace_paths = sorted(output_dir.glob("pytorch_profile_*_dprank*.json"))
    if not trace_paths:
        raise SystemExit(f"No dprank PyTorch traces found in {output_dir}")
    for trace_path in trace_paths:
        print(f"Analyzing {trace_path.name}", flush=True)
        _analyze_trace(trace_path, wanted, aggregates)

    output_rows: list[dict[str, Any]] = []
    for ratio_row in ratio_rows:
        uid = ratio_row["node_uid"]
        normal = float(ratio_row["median_normal_time_us"])
        isolated = float(ratio_row["median_isolated_time_us"])
        aggregate = aggregates[uid]
        analysis = _classify(normal, isolated, aggregate)
        task_types = sorted(aggregate.task_types)
        row: dict[str, Any] = {
            "node_uid": uid,
            "median_normal_time_us": normal,
            "median_isolated_time_us": isolated,
            "normal_to_isolated_ratio": float(
                ratio_row["normal_to_isolated_ratio"]
            ),
            "normal_minus_isolated_us": normal - isolated,
            "trace_task_type": ";".join(task_types)
            if task_types
            else _fallback_task_type(uid),
            "overlapping_node_types": ";".join(sorted(aggregate.overlapping_types)),
            "trace_unmatched_device_activity_count": (
                aggregate.unmatched_device_activity_count
            ),
            **analysis,
        }
        output_rows.append(
            {
                field_name: _csv_number(row.get(field_name))
                if isinstance(row.get(field_name), float)
                else row.get(field_name, "")
                for field_name in OUTPUT_FIELDS
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
