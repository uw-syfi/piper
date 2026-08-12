#!/usr/bin/env python3
"""Summarize cross-stream GPU overlap across all DP-rank traces in one run.

The input is an output directory produced by ``examples/test_harness.py``.
Every ``pytorch_profile_*_dprank*.json`` trace directly inside that directory
is analyzed. The report pools every qualifying event from every GPU and DP
rank; no per-GPU or low-count statistics are excluded.

Usage:
    python scripts/profile_and_replay/diagnostics/analyze_gpu_interference.py PATH/TO/OUTPUT_DIRECTORY
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import re
from pathlib import Path
from typing import Any


EVENT_PREFIXES = (
    "forward:",
    "backward:",
    "forward_a2a:",
    "backward_a2a:",
    "all_reduce:",
    "send:",
    "recv:",
)
GPU_EVENT_CATEGORIES = {
    "kernel",
    "gpu_memcpy",
    "gpu_memset",
    "gpu_user_annotation",
}
STATISTICS = ("all", "no_overlap", "overlap")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Run output directory containing combined DP-rank trace JSON files.",
    )
    return parser.parse_args()


def event_prefix(name: object) -> str | None:
    if not isinstance(name, str):
        return None
    return next((prefix for prefix in EVENT_PREFIXES if name.startswith(prefix)), None)


def gpu_pids(events: list[dict[str, Any]]) -> list[object]:
    """Return process IDs containing GPU-side complete events."""
    return sorted(
        {
            event.get("pid")
            for event in events
            if event.get("ph") == "X"
            and event.get("cat") in GPU_EVENT_CATEGORIES
        },
        key=str,
    )


def pp_rank_by_pid(events: list[dict[str, Any]], pids: list[object]) -> dict[object, int]:
    """Resolve merged-trace GPU processes to their PP/GPU rank.

    ``examples/test_harness.py`` prefixes ``process_name`` metadata with the
    DP and PP rank while merging actor traces. Each GPU process can therefore
    be matched to the PP rank that owns the corresponding local GPU.
    """
    process_names = {
        event.get("pid"): (event.get("args") or {}).get("name", "")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    labels: dict[object, int] = {}
    for index, pid in enumerate(pids):
        match = re.search(r"\bpp(\d+)\b", str(process_names.get(pid, "")))
        # The PID order is stable in Piper's merged traces. Fall back to it if
        # a trace lacks process-name metadata.
        labels[pid] = int(match.group(1)) if match else index
    return labels


def per_gpu_statistics(
    events: list[dict[str, Any]], pid: object
) -> dict[str, dict[str, tuple[float, int]]]:
    """Return duration sums (microseconds) and counts by event/statistic."""
    gpu_events: list[tuple[float, float, object]] = []
    targets: list[tuple[str, float, float, object, float]] = []

    for event in events:
        if (
            event.get("pid") != pid
            or event.get("ph") != "X"
            or event.get("cat") not in GPU_EVENT_CATEGORIES
        ):
            continue

        ts = event.get("ts")
        duration = event.get("dur")
        stream = event.get("tid")
        if (
            not isinstance(ts, (int, float))
            or not isinstance(duration, (int, float))
            or duration <= 0
        ):
            continue

        start = float(ts)
        duration_us = float(duration)
        end = start + duration_us
        gpu_events.append((start, end, stream))

        prefix = event_prefix(event.get("name"))
        if prefix is not None:
            targets.append((prefix, start, end, stream, duration_us))

    stream_indexes = build_stream_indexes(gpu_events)
    results = {
        prefix: {statistic: [0.0, 0] for statistic in STATISTICS}
        for prefix in EVENT_PREFIXES
    }

    for prefix, start, end, stream, duration_us in targets:
        overlaps = overlaps_other_stream(start, end, stream, stream_indexes)
        for statistic in ("all", "overlap" if overlaps else "no_overlap"):
            results[prefix][statistic][0] += duration_us
            results[prefix][statistic][1] += 1

    return {
        prefix: {
            statistic: (float(values[0]), int(values[1]))
            for statistic, values in statistics.items()
        }
        for prefix, statistics in results.items()
    }


def build_stream_indexes(
    gpu_events: list[tuple[float, float, object]],
) -> dict[object, tuple[list[float], list[float]]]:
    """Index each stream for interval overlap queries.

    For an interval ending at ``end``, an overlap exists if an interval on a
    different stream starts before ``end`` and has an end after its start.
    Prefix maxima make that query logarithmic per other stream.
    """
    by_stream: dict[object, list[tuple[float, float]]] = collections.defaultdict(list)
    for start, end, stream in gpu_events:
        by_stream[stream].append((start, end))

    indexes: dict[object, tuple[list[float], list[float]]] = {}
    for stream, intervals in by_stream.items():
        intervals.sort()
        starts: list[float] = []
        maximum_ends: list[float] = []
        maximum_end = float("-inf")
        for start, end in intervals:
            starts.append(start)
            maximum_end = max(maximum_end, end)
            maximum_ends.append(maximum_end)
        indexes[stream] = (starts, maximum_ends)
    return indexes


def overlaps_other_stream(
    start: float,
    end: float,
    stream: object,
    stream_indexes: dict[object, tuple[list[float], list[float]]],
) -> bool:
    """Return whether [start, end) overlaps a GPU event on another stream."""
    for other_stream, (starts, maximum_ends) in stream_indexes.items():
        if other_stream == stream:
            continue
        candidate_index = bisect.bisect_left(starts, end) - 1
        if candidate_index >= 0 and maximum_ends[candidate_index] > start:
            return True
    return False


def aggregate_across_sources(
    source_statistics: list[dict[str, dict[str, tuple[float, int]]]],
) -> dict[str, dict[str, tuple[float, int, int]]]:
    """Pool all event durations and counts across GPU/DP-rank sources."""
    aggregate = {
        prefix: {statistic: [0.0, 0, 0] for statistic in STATISTICS}
        for prefix in EVENT_PREFIXES
    }
    for statistics_for_source in source_statistics:
        for prefix in EVENT_PREFIXES:
            for statistic in STATISTICS:
                duration_sum_us, count = statistics_for_source[prefix][statistic]
                if count == 0:
                    continue
                aggregate[prefix][statistic][0] += duration_sum_us
                aggregate[prefix][statistic][1] += count
                aggregate[prefix][statistic][2] += 1
    return {
        prefix: {
            statistic: (float(values[0]), int(values[1]), int(values[2]))
            for statistic, values in statistics.items()
        }
        for prefix, statistics in aggregate.items()
    }


def format_duration(statistic: tuple[float, int, int]) -> str:
    duration_sum_us, count, contributing_sources = statistic
    if count == 0:
        return "N/A (n=0)"
    mean_ms = duration_sum_us / count / 1000.0
    return f"{mean_ms:.6f} ms (n={count}; sources={contributing_sources})"


def format_ratio(
    no_overlap: tuple[float, int, int], overlap: tuple[float, int, int]
) -> str:
    no_sum_us, no_count, _ = no_overlap
    overlap_sum_us, overlap_count, _ = overlap
    if no_count == 0 or overlap_count == 0:
        return "N/A"
    no_mean_us = no_sum_us / no_count
    overlap_mean_us = overlap_sum_us / overlap_count
    return f"{overlap_mean_us / no_mean_us:.6f}x"


def render_table(aggregate: dict[str, dict[str, tuple[float, int, int]]]) -> list[str]:
    lines = [
        "| Statistic | forward: | backward: | forward_a2a: | backward_a2a: | all_reduce: | send: | recv: |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for statistic, label in (
        ("all", "All"),
        ("no_overlap", "No overlap"),
        ("overlap", "Overlap"),
    ):
        cells = [format_duration(aggregate[prefix][statistic]) for prefix in EVENT_PREFIXES]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    ratios = [
        format_ratio(aggregate[prefix]["no_overlap"], aggregate[prefix]["overlap"])
        for prefix in EVENT_PREFIXES
    ]
    lines.append("| Overlap / no overlap | " + " | ".join(ratios) + " |")
    return lines


def render_report(
    trace_paths: list[Path],
    aggregate: dict[str, dict[str, tuple[float, int, int]]],
    aggregate_by_pp_rank: dict[int, dict[str, dict[str, tuple[float, int, int]]]],
) -> str:
    lines = [
        "# Event-weighted GPU interference statistics across DP ranks",
        "",
        "Source traces:",
        "",
    ]
    lines.extend(f"- `{trace_path.name}`" for trace_path in trace_paths)
    lines.extend(
        [
            "",
            "Every qualifying GPU event from every listed trace is pooled "
            "directly. Duration cells contain `mean wall duration in ms "
            "(event count; contributing GPU/DP-rank sources)`. The ratio is "
            "`overlap mean / no-overlap mean`.",
            "",
        ]
    )
    lines.extend(render_table(aggregate))
    for pp_rank, pp_aggregate in aggregate_by_pp_rank.items():
        lines.extend(["", f"## GPU {pp_rank} (PP rank {pp_rank})", ""])
        lines.extend(render_table(pp_aggregate))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not output_dir.is_dir():
        raise SystemExit(f"Output directory does not exist: {output_dir}")

    trace_paths = sorted(output_dir.glob("pytorch_profile_*_dprank*.json"))
    if not trace_paths:
        raise SystemExit(f"No DP-rank PyTorch profile traces found in: {output_dir}")

    source_statistics = []
    source_statistics_by_pp_rank: dict[int, list[dict[str, dict[str, tuple[float, int]]]]] = (
        collections.defaultdict(list)
    )
    for trace_path in trace_paths:
        with trace_path.open(encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
        events = trace.get("traceEvents")
        if not isinstance(events, list):
            raise SystemExit(f"Missing traceEvents list in: {trace_path}")
        pids = gpu_pids(events)
        pp_ranks = pp_rank_by_pid(events, pids)
        for pid in pids:
            statistics = per_gpu_statistics(events, pid)
            source_statistics.append(statistics)
            source_statistics_by_pp_rank[pp_ranks[pid]].append(statistics)

    if not source_statistics:
        raise SystemExit(f"No GPU complete events found in: {output_dir}")

    report_path = output_dir / "interference_stats_weighted_all_dpranks.md"
    aggregate_by_pp_rank = {
        pp_rank: aggregate_across_sources(statistics)
        for pp_rank, statistics in sorted(source_statistics_by_pp_rank.items())
    }
    report_path.write_text(
        render_report(
            trace_paths,
            aggregate_across_sources(source_statistics),
            aggregate_by_pp_rank,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
