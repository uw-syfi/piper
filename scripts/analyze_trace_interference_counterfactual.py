#!/usr/bin/env python3
"""Reconstruct normal and isolated Piper steps from PyTorch traces.

The analysis matches logical GPU kernels by ``(PP rank, External id, ordinal)``
across DP-rank traces.  A communication kernel's normal duration is the minimum
of its DP samples; all other kernels use the median.  Isolated controls
come from matching kernel signatures which have at least one occurrence with no
positive-duration overlap with any other GPU kernel.

All-gather kernels deliberately retain their normal duration even if a trace
happens to contain an apparent isolated occurrence.  Send/recv or other kernels
without a strict isolated control also retain their normal duration.

Per-node durations are rebuilt from representative kernel durations plus the
node's observed internal kernel-gap budget.  The finalized TrainingDAG is then
replayed twice with Piper's dependency-aware replay, allowing downstream
nodes to start earlier when their upstream dependencies complete sooner.
"""

from __future__ import annotations

import argparse
import bisect
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.profiling import (
    ISOLATED_PROFILE_MODE,
    NORMAL_PROFILE_MODE,
    NodeProfile,
    ProfiledTrainingPlan,
    get_prologue_profile_times,
    set_node_profile_times,
    set_prologue_profile_times,
)
from src.replay import ReplayResult, replay_training_step


_TRACE_DP_RE = re.compile(r"dprank(\d+)")
_PROCESS_PP_RE = re.compile(r"\bpp(\d+):")
_DEVICE_CATEGORIES = {"kernel"}


@dataclass(frozen=True, order=True)
class NodeKey:
    pp_rank: int
    uid: str


@dataclass(frozen=True, order=True)
class LogicalKernelKey:
    pp_rank: int
    external_id: int
    ordinal: int


@dataclass
class KernelOccurrence:
    event_id: int
    dp_rank: int
    pp_rank: int
    gpu_pid: int
    stream: int | str
    external_id: int
    ordinal: int
    node: NodeKey | None
    task_type: str
    name: str
    signature: tuple[Any, ...]
    start_us: float
    duration_us: float
    isolated: bool = False

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us

    @property
    def logical_key(self) -> LogicalKernelKey:
        return LogicalKernelKey(
            self.pp_rank,
            self.external_id,
            self.ordinal,
        )


@dataclass(frozen=True)
class LogicalKernelTiming:
    key: LogicalKernelKey
    node: NodeKey
    node_kind: str
    task_type: str
    name: str
    signature: tuple[Any, ...]
    samples_us: tuple[float, ...]
    isolated_dp_ranks: tuple[int, ...]
    normal_us: float
    isolated_us: float
    isolated_source: str
    control_count: int


@dataclass(frozen=True)
class NodeTiming:
    key: NodeKey
    node_kind: str
    kernel_count: int
    controlled_kernel_count: int
    fallback_kernel_count: int
    internal_gap_us: float
    normal_us: float
    isolated_us: float

    @property
    def reduction_us(self) -> float:
        return self.normal_us - self.isolated_us

    @property
    def reduction_pct(self) -> float:
        return (
            100.0 * self.reduction_us / self.normal_us
            if self.normal_us > 0.0
            else 0.0
        )


class KernelIndex:
    """Find positive-duration overlaps on one GPU in O(log n + matches)."""

    def __init__(self, occurrences: Iterable[KernelOccurrence]):
        self.events = sorted(
            occurrences,
            key=lambda event: (
                event.start_us,
                event.end_us,
                event.event_id,
            ),
        )
        self.starts = [event.start_us for event in self.events]
        self.prefix_max_end: list[float] = []
        running = -math.inf
        for event in self.events:
            running = max(running, event.end_us)
            self.prefix_max_end.append(running)

    def is_isolated(self, target: KernelOccurrence) -> bool:
        stop = bisect.bisect_left(self.starts, target.end_us)
        index = stop - 1
        while index >= 0 and self.prefix_max_end[index] > target.start_us:
            other = self.events[index]
            if (
                other.event_id != target.event_id
                and other.end_us > target.start_us
                and other.start_us < target.end_us
            ):
                return False
            index -= 1
        return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing combined pytorch_profile_*_dprank*.json traces.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="Compatible step_time_profiles.json containing the finalized DAGs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Summary JSON (default: TRACE_DIR/trace_interference_counterfactual.json).",
    )
    parser.add_argument(
        "--nodes-csv",
        type=Path,
        help="Per-node CSV (default: TRACE_DIR/trace_interference_nodes.csv).",
    )
    parser.add_argument(
        "--kernels-csv",
        type=Path,
        help="Per-logical-kernel CSV (default: TRACE_DIR/trace_interference_kernels.csv).",
    )
    return parser.parse_args()


def _median(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("median requires at least one value")
    return float(statistics.median(materialized))


def _representative(values: Iterable[float], *, communication: bool) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("representative duration requires at least one value")
    return float(min(materialized) if communication else statistics.median(materialized))


def _uid_from_label(label: str) -> str:
    if ":uid" not in label:
        raise ValueError(f"malformed Piper DAG label: {label!r}")
    return label.rsplit(":uid", 1)[1]


def _kernel_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    args = event.get("args") or {}
    full_name = str(args.get("original_cuda_name") or event.get("name", ""))
    name = full_name.split("::", 1)[-1]
    return (
        name,
        tuple(args.get("grid", ())),
        tuple(args.get("block", ())),
        str(args.get("Collective name", "")),
        int(args.get("In msg nelems", -1)),
        int(args.get("Out msg nelems", -1)),
        int(args.get("Group size", -1)),
        str(args.get("dtype", "")),
    )


def _process_pp_ranks(events: list[dict[str, Any]]) -> dict[int, int]:
    result: dict[int, int] = {}
    for event in events:
        if event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        name = str((event.get("args") or {}).get("name", ""))
        match = _PROCESS_PP_RE.search(name)
        if match:
            result[int(event["pid"])] = int(match.group(1))
    if not result:
        raise ValueError("trace contains no DP/PP-prefixed process_name metadata")
    return result


def _load_trace(
    path: Path,
) -> tuple[int, list[KernelOccurrence], dict[NodeKey, float]]:
    match = _TRACE_DP_RE.search(path.name)
    if match is None:
        raise ValueError(f"trace filename has no dprank suffix: {path}")
    dp_rank = int(match.group(1))
    with path.open(encoding="utf-8") as trace_file:
        events = json.load(trace_file).get("traceEvents", [])
    pp_by_pid = _process_pp_ranks(events)

    kernels: list[KernelOccurrence] = []
    by_external_id: dict[tuple[int, int], list[KernelOccurrence]] = {}
    by_gpu_pid: dict[int, list[KernelOccurrence]] = {}
    for event_id, event in enumerate(events):
        if event.get("ph") != "X" or event.get("cat") not in _DEVICE_CATEGORIES:
            continue
        args = event.get("args") or {}
        gpu_pid = int(event["pid"])
        if gpu_pid not in pp_by_pid:
            continue
        raw_external_id = args.get("External id")
        if raw_external_id is None:
            continue
        label = str(args.get("piper_dag_node", ""))
        node = (
            None
            if not label
            else NodeKey(pp_by_pid[gpu_pid], _uid_from_label(label))
        )
        task_type = label.split(":", 1)[0] if label else ""
        full_name = str(args.get("original_cuda_name") or event.get("name", ""))
        occurrence = KernelOccurrence(
            event_id=event_id,
            dp_rank=dp_rank,
            pp_rank=pp_by_pid[gpu_pid],
            gpu_pid=gpu_pid,
            stream=args.get("stream", event.get("tid")),
            external_id=int(raw_external_id),
            ordinal=-1,
            node=node,
            task_type=task_type,
            name=full_name.split("::", 1)[-1],
            signature=_kernel_signature(event),
            start_us=float(event["ts"]),
            duration_us=float(event["dur"]),
        )
        kernels.append(occurrence)
        by_gpu_pid.setdefault(gpu_pid, []).append(occurrence)
        by_external_id.setdefault(
            (occurrence.pp_rank, occurrence.external_id),
            [],
        ).append(occurrence)

    for group in by_external_id.values():
        group.sort(
            key=lambda occurrence: (
                occurrence.start_us,
                occurrence.name,
                str(occurrence.stream),
            )
        )
        for ordinal, occurrence in enumerate(group):
            occurrence.ordinal = ordinal

    for gpu_kernels in by_gpu_pid.values():
        index = KernelIndex(gpu_kernels)
        for occurrence in gpu_kernels:
            occurrence.isolated = index.is_isolated(occurrence)

    gap_by_node: dict[NodeKey, float] = {}
    node_kernels: dict[NodeKey, list[KernelOccurrence]] = {}
    for occurrence in kernels:
        if occurrence.node is not None:
            node_kernels.setdefault(occurrence.node, []).append(occurrence)
    for node, occurrences in node_kernels.items():
        intervals = _merge_intervals(
            (occurrence.start_us, occurrence.end_us)
            for occurrence in occurrences
        )
        span = intervals[-1][1] - intervals[0][0]
        union = sum(end - start for start, end in intervals)
        gap_by_node[node] = max(0.0, span - union)

    del events
    return dp_rank, kernels, gap_by_node


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


def _plan_nodes(
    plan: ProfiledTrainingPlan,
) -> tuple[dict[NodeKey, Any], dict[NodeKey, str]]:
    nodes: dict[NodeKey, Any] = {}
    kinds: dict[NodeKey, str] = {}
    for pp_rank, dag in enumerate(plan.per_pp_dags):
        for uid, node in dag.nodes.items():
            key = NodeKey(pp_rank, uid)
            nodes[key] = node
            kinds[key] = node.node_kind
    return nodes, kinds


def _derive_timings(
    plan: ProfiledTrainingPlan,
    traces: list[tuple[int, list[KernelOccurrence], dict[NodeKey, float]]],
) -> tuple[
    dict[LogicalKernelKey, LogicalKernelTiming],
    dict[NodeKey, NodeTiming],
    dict[str, Any],
]:
    plan_nodes, node_kinds = _plan_nodes(plan)
    dp_ranks = tuple(sorted(dp_rank for dp_rank, _kernels, _gaps in traces))
    if len(dp_ranks) < 2:
        raise ValueError("cross-DP matching requires at least two DP-rank traces")
    if dp_ranks != tuple(range(len(dp_ranks))):
        raise ValueError(f"DP-rank traces must be contiguous from zero: {dp_ranks}")

    samples_by_kernel: dict[
        LogicalKernelKey,
        dict[int, KernelOccurrence],
    ] = {}
    kernel_keys_by_node: dict[NodeKey, set[LogicalKernelKey]] = {}
    gaps_by_node: dict[NodeKey, dict[int, float]] = {}
    for dp_rank, kernels, gaps in traces:
        for occurrence in kernels:
            if occurrence.node is None:
                continue
            samples = samples_by_kernel.setdefault(occurrence.logical_key, {})
            if dp_rank in samples:
                raise ValueError(
                    f"duplicate logical kernel {occurrence.logical_key} on DP rank {dp_rank}"
                )
            samples[dp_rank] = occurrence
            kernel_keys_by_node.setdefault(occurrence.node, set()).add(
                occurrence.logical_key
            )
        for node, gap in gaps.items():
            gaps_by_node.setdefault(node, {})[dp_rank] = gap

    normal_by_kernel: dict[LogicalKernelKey, float] = {}
    template_by_kernel: dict[LogicalKernelKey, KernelOccurrence] = {}
    control_signature_by_kernel: dict[
        LogicalKernelKey,
        tuple[Any, ...],
    ] = {}
    isolated_control_normals: dict[tuple[Any, ...], list[float]] = {}
    for key, samples in samples_by_kernel.items():
        if set(samples) != set(dp_ranks):
            raise ValueError(
                f"logical kernel {key} has DP samples {sorted(samples)}, "
                f"expected {list(dp_ranks)}"
            )
        identities = {
            (
                sample.node,
                sample.task_type,
                sample.name,
            )
            for sample in samples.values()
        }
        if len(identities) != 1:
            raise ValueError(
                f"logical kernel {key} does not match across DP ranks: {identities}"
            )
        template = samples[dp_ranks[0]]
        assert template.node is not None
        if template.node not in node_kinds:
            raise ValueError(
                f"trace kernel {key} references unknown plan node {template.node}"
            )
        communication = node_kinds[template.node].endswith("_COMM")
        normal = _representative(
            (sample.duration_us for sample in samples.values()),
            communication=communication,
        )
        normal_by_kernel[key] = normal
        template_by_kernel[key] = template
        # Ragged expert routing can legitimately change grid sizes across DP
        # ranks for one logical kernel.  The eight-rank multiset is itself the
        # workload signature: repeated microbatch occurrences are controls only
        # when their complete cross-rank shape distribution matches.
        control_signature = (
            template.name,
            tuple(
                sorted(
                    (sample.signature for sample in samples.values()),
                    key=repr,
                )
            ),
        )
        control_signature_by_kernel[key] = control_signature
        if any(sample.isolated for sample in samples.values()):
            isolated_control_normals.setdefault(control_signature, []).append(
                normal
            )

    kernel_timings: dict[LogicalKernelKey, LogicalKernelTiming] = {}
    for key, normal in normal_by_kernel.items():
        samples = samples_by_kernel[key]
        template = template_by_kernel[key]
        assert template.node is not None
        node_kind = node_kinds[template.node]
        communication = node_kind.endswith("_COMM")
        control_signature = control_signature_by_kernel[key]
        controls = isolated_control_normals.get(control_signature, [])
        if node_kind == "ALL_GATHER_COMM":
            isolated = normal
            source = "normal_fallback_all_gather"
        elif controls:
            isolated = _representative(
                controls,
                communication=communication,
            )
            source = "strict_isolated_signature_control"
        else:
            isolated = normal
            source = "normal_fallback_no_isolated_control"
        kernel_timings[key] = LogicalKernelTiming(
            key=key,
            node=template.node,
            node_kind=node_kind,
            task_type=template.task_type,
            name=template.name,
            signature=control_signature,
            samples_us=tuple(samples[rank].duration_us for rank in dp_ranks),
            isolated_dp_ranks=tuple(
                rank for rank in dp_ranks if samples[rank].isolated
            ),
            normal_us=normal,
            isolated_us=isolated,
            isolated_source=source,
            control_count=len(controls),
        )

    node_timings: dict[NodeKey, NodeTiming] = {}
    for key, node in plan_nodes.items():
        logical_keys = sorted(kernel_keys_by_node.get(key, ()))
        if not logical_keys:
            # ORDER_DUMMY nodes and empty BWD_W buckets perform no GPU work.
            node_timings[key] = NodeTiming(
                key=key,
                node_kind=node.node_kind,
                kernel_count=0,
                controlled_kernel_count=0,
                fallback_kernel_count=0,
                internal_gap_us=0.0,
                normal_us=0.0,
                isolated_us=0.0,
            )
            continue
        gap_samples = gaps_by_node.get(key, {})
        if set(gap_samples) != set(dp_ranks):
            raise ValueError(
                f"node {key} has internal-gap samples {sorted(gap_samples)}, "
                f"expected {list(dp_ranks)}"
            )
        communication = node.node_kind.endswith("_COMM")
        internal_gap = _representative(
            gap_samples.values(),
            communication=communication,
        )
        timings = [kernel_timings[logical_key] for logical_key in logical_keys]
        controlled = sum(
            timing.isolated_source
            == "strict_isolated_signature_control"
            for timing in timings
        )
        normal = internal_gap + sum(timing.normal_us for timing in timings)
        isolated = internal_gap + sum(
            timing.isolated_us for timing in timings
        )
        node_timings[key] = NodeTiming(
            key=key,
            node_kind=node.node_kind,
            kernel_count=len(timings),
            controlled_kernel_count=controlled,
            fallback_kernel_count=len(timings) - controlled,
            internal_gap_us=internal_gap,
            normal_us=normal,
            isolated_us=isolated,
        )

    coverage_by_kind: dict[str, dict[str, int | float]] = {}
    for timing in kernel_timings.values():
        row = coverage_by_kind.setdefault(
            timing.node_kind,
            {
                "logical_kernel_count": 0,
                "kernel_count_with_direct_isolated_dp_occurrence": 0,
                "kernel_count_with_signature_control": 0,
                "all_gather_forced_fallback_count": 0,
                "no_control_fallback_count": 0,
            },
        )
        row["logical_kernel_count"] += 1
        row["kernel_count_with_direct_isolated_dp_occurrence"] += bool(
            timing.isolated_dp_ranks
        )
        if timing.isolated_source == "strict_isolated_signature_control":
            row["kernel_count_with_signature_control"] += 1
        elif timing.isolated_source == "normal_fallback_all_gather":
            row["all_gather_forced_fallback_count"] += 1
        else:
            row["no_control_fallback_count"] += 1
    for row in coverage_by_kind.values():
        total = int(row["logical_kernel_count"])
        controlled = int(row["kernel_count_with_signature_control"])
        row["signature_control_coverage_pct"] = (
            100.0 * controlled / total if total else 0.0
        )

    matching = {
        "dp_ranks": list(dp_ranks),
        "sample_count_per_logical_kernel": len(dp_ranks),
        "logical_kernel_count": len(kernel_timings),
        "external_id_group_count": len(
            {
                (key.pp_rank, key.external_id)
                for key in kernel_timings
            }
        ),
        "plan_node_count": len(plan_nodes),
        "trace_node_count": sum(
            timing.kernel_count > 0 for timing in node_timings.values()
        ),
        "zero_kernel_plan_node_count": sum(
            timing.kernel_count == 0 for timing in node_timings.values()
        ),
        "coverage_by_node_kind": coverage_by_kind,
    }
    return kernel_timings, node_timings, matching


def _install_trace_profiles(
    plan: ProfiledTrainingPlan,
    node_timings: dict[NodeKey, NodeTiming],
) -> None:
    for pp_rank, dag in enumerate(plan.per_pp_dags):
        normal_prologue = get_prologue_profile_times(dag, NORMAL_PROFILE_MODE)
        # Keep prologue identical in both replays so the comparison isolates
        # kernel interference rather than a separately profiled setup phase.
        set_prologue_profile_times(
            dag,
            ISOLATED_PROFILE_MODE,
            normal_prologue,
        )
        for uid, node in dag.nodes.items():
            timing = node_timings[NodeKey(pp_rank, uid)]
            if node.node_kind == "ORDER_DUMMY":
                continue
            ranks = tuple(int(rank) for rank in (node.device or ()))
            if not ranks:
                raise ValueError(f"plan node {uid} has no device ranks")
            normal_profile = NodeProfile.from_samples(
                (timing.normal_us,),
                timing_source="pytorch_trace_kernel_reconstruction",
                isolation_scope="normal_cross_dp_representative",
            )
            isolated_profile = NodeProfile.from_samples(
                (timing.isolated_us,),
                timing_source="pytorch_trace_strict_isolated_control",
                isolation_scope="strict_no_other_gpu_kernel_overlap",
            )
            set_node_profile_times(
                node,
                NORMAL_PROFILE_MODE,
                {rank: normal_profile for rank in ranks},
            )
            set_node_profile_times(
                node,
                ISOLATED_PROFILE_MODE,
                {rank: isolated_profile for rank in ranks},
            )


def _replay_summary(result: ReplayResult) -> dict[str, Any]:
    return {
        "step_time_us": result.step_time_us,
        "step_time_s": result.step_time_s,
        "per_rank_makespan_us": {
            str(rank): duration
            for rank, duration in sorted(result.per_rank_makespan_us.items())
        },
        "critical_path": [
            {
                "operation_id": entry.operation_id,
                "members": [
                    {"global_rank": member.global_rank, "uid": member.uid}
                    for member in entry.members
                ],
                "start_us": entry.start_us,
                "end_us": entry.end_us,
                "duration_us": entry.effective_duration_us,
            }
            for entry in result.critical_path
        ],
        "warnings": list(result.warnings),
    }


def _load_reference_sample_medians(plan_path: Path) -> dict[str, Any] | None:
    """Load the profile-and-replay medians adjacent to the profiled plan."""
    results_path = plan_path.with_name("results.csv")
    if not results_path.exists():
        return None
    with results_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        return None
    normal_values = {
        float(row["normal_replay_step_time_median_s"])
        for row in rows
        if row.get("normal_replay_step_time_median_s")
    }
    isolated_values = {
        float(row["perfect_overlap_replay_step_time_median_s"])
        for row in rows
        if row.get("perfect_overlap_replay_step_time_median_s")
    }
    if len(normal_values) != 1 or len(isolated_values) != 1:
        return None
    normal_us = normal_values.pop() * 1_000_000.0
    isolated_us = isolated_values.pop() * 1_000_000.0
    reduction_us = normal_us - isolated_us
    return {
        "results_csv": str(results_path.resolve()),
        "normal_step_median_us": normal_us,
        "isolated_step_median_us": isolated_us,
        "step_reduction_median_us": reduction_us,
        "step_reduction_median_pct": (
            100.0 * reduction_us / normal_us if normal_us else 0.0
        ),
    }


def _write_nodes_csv(path: Path, timings: dict[NodeKey, NodeTiming]) -> None:
    fields = (
        "pp_rank",
        "node_uid",
        "node_kind",
        "kernel_count",
        "controlled_kernel_count",
        "fallback_kernel_count",
        "internal_gap_us",
        "normal_us",
        "isolated_us",
        "reduction_us",
        "reduction_pct",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key, timing in sorted(timings.items()):
            writer.writerow(
                {
                    "pp_rank": key.pp_rank,
                    "node_uid": key.uid,
                    "node_kind": timing.node_kind,
                    "kernel_count": timing.kernel_count,
                    "controlled_kernel_count": timing.controlled_kernel_count,
                    "fallback_kernel_count": timing.fallback_kernel_count,
                    "internal_gap_us": timing.internal_gap_us,
                    "normal_us": timing.normal_us,
                    "isolated_us": timing.isolated_us,
                    "reduction_us": timing.reduction_us,
                    "reduction_pct": timing.reduction_pct,
                }
            )


def _write_kernels_csv(
    path: Path,
    timings: dict[LogicalKernelKey, LogicalKernelTiming],
) -> None:
    fields = (
        "pp_rank",
        "external_id",
        "ordinal",
        "node_uid",
        "node_kind",
        "task_type",
        "kernel_name",
        "samples_us",
        "isolated_dp_ranks",
        "normal_us",
        "isolated_us",
        "reduction_us",
        "isolated_source",
        "control_count",
        "signature",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key, timing in sorted(timings.items()):
            writer.writerow(
                {
                    "pp_rank": key.pp_rank,
                    "external_id": key.external_id,
                    "ordinal": key.ordinal,
                    "node_uid": timing.node.uid,
                    "node_kind": timing.node_kind,
                    "task_type": timing.task_type,
                    "kernel_name": timing.name,
                    "samples_us": json.dumps(timing.samples_us),
                    "isolated_dp_ranks": json.dumps(timing.isolated_dp_ranks),
                    "normal_us": timing.normal_us,
                    "isolated_us": timing.isolated_us,
                    "reduction_us": timing.normal_us - timing.isolated_us,
                    "isolated_source": timing.isolated_source,
                    "control_count": timing.control_count,
                    "signature": json.dumps(timing.signature),
                }
            )


def main() -> None:
    args = _parse_args()
    trace_dir = args.trace_dir.resolve()
    plan_path = args.plan.resolve()
    output_path = args.output or (
        trace_dir / "trace_interference_counterfactual.json"
    )
    nodes_csv_path = args.nodes_csv or (
        trace_dir / "trace_interference_nodes.csv"
    )
    kernels_csv_path = args.kernels_csv or (
        trace_dir / "trace_interference_kernels.csv"
    )

    trace_paths = sorted(trace_dir.glob("pytorch_profile_*_dprank*.json"))
    if not trace_paths:
        raise SystemExit(f"no combined DP-rank traces found in {trace_dir}")
    traces = [_load_trace(path) for path in trace_paths]

    plan = ProfiledTrainingPlan.load_json(plan_path)
    reference_normal = replay_training_step(
        plan,
        profile_mode=NORMAL_PROFILE_MODE,
    )
    reference_isolated = replay_training_step(
        plan,
        profile_mode=ISOLATED_PROFILE_MODE,
    )
    kernel_timings, node_timings, matching = _derive_timings(plan, traces)
    _install_trace_profiles(plan, node_timings)
    reconstructed_normal = replay_training_step(
        plan,
        profile_mode=NORMAL_PROFILE_MODE,
    )
    reconstructed_isolated = replay_training_step(
        plan,
        profile_mode=ISOLATED_PROFILE_MODE,
    )

    reduction_us = (
        reconstructed_normal.step_time_us
        - reconstructed_isolated.step_time_us
    )
    reduction_pct = (
        100.0 * reduction_us / reconstructed_normal.step_time_us
        if reconstructed_normal.step_time_us
        else 0.0
    )
    reference_reduction_pct = (
        100.0
        * (
            reference_normal.step_time_us
            - reference_isolated.step_time_us
        )
        / reference_normal.step_time_us
    )
    reference_sample_medians = _load_reference_sample_medians(plan_path)
    total_normal_node_work = sum(
        timing.normal_us for timing in node_timings.values()
    )
    total_isolated_node_work = sum(
        timing.isolated_us for timing in node_timings.values()
    )
    top_node_reductions = [
        {
            "pp_rank": timing.key.pp_rank,
            "node_uid": timing.key.uid,
            "node_kind": timing.node_kind,
            "normal_us": timing.normal_us,
            "isolated_us": timing.isolated_us,
            "reduction_us": timing.reduction_us,
            "reduction_pct": timing.reduction_pct,
            "kernel_count": timing.kernel_count,
            "controlled_kernel_count": timing.controlled_kernel_count,
        }
        for timing in sorted(
            node_timings.values(),
            key=lambda value: (-value.reduction_us, value.key),
        )[:50]
    ]

    summary = {
        "inputs": {
            "trace_dir": str(trace_dir),
            "trace_files": [str(path.resolve()) for path in trace_paths],
            "profiled_plan": str(plan_path),
        },
        "method": {
            "logical_kernel_identity": "(PP rank, External id, ordinal)",
            "communication_normal_estimator": "minimum across DP ranks",
            "compute_normal_estimator": "median across DP ranks",
            "isolated_control": (
                "same kernel signature with at least one occurrence having no "
                "positive-duration overlap with any other GPU kernel"
            ),
            "all_gather_policy": "always retain normal duration",
            "missing_control_policy": "retain normal duration",
            "node_duration": (
                "sum of reconstructed logical-kernel durations plus unchanged "
                "representative internal kernel-gap budget"
            ),
            "step_replay": (
                "finalized TrainingDAG dependencies, stream FIFO, collective "
                "rendezvous, and directional P2P semantics"
            ),
            "prologue_policy": (
                "normal profiled prologue retained identically in both replays"
            ),
        },
        "matching": matching,
        "results": {
            "normal": _replay_summary(reconstructed_normal),
            "isolated": _replay_summary(
                reconstructed_isolated
            ),
            "step_reduction_us": reduction_us,
            "step_reduction_pct": reduction_pct,
            "total_normal_node_work_us": total_normal_node_work,
            "total_isolated_node_work_us": (
                total_isolated_node_work
            ),
            "total_node_work_reduction_us": (
                total_normal_node_work - total_isolated_node_work
            ),
            "total_node_work_reduction_pct": (
                100.0
                * (
                    total_normal_node_work
                    - total_isolated_node_work
                )
                / total_normal_node_work
            ),
        },
        "reference_profile_and_replay": {
            "normal_step_us": reference_normal.step_time_us,
            "isolated_step_us": (
                reference_isolated.step_time_us
            ),
            "step_reduction_pct": reference_reduction_pct,
            "raw_sample_medians": reference_sample_medians,
        },
        "top_node_reductions": top_node_reductions,
        "artifacts": {
            "nodes_csv": str(nodes_csv_path.resolve()),
            "kernels_csv": str(kernels_csv_path.resolve()),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_nodes_csv(nodes_csv_path, node_timings)
    _write_kernels_csv(kernels_csv_path, kernel_timings)

    print(f"normal_step_us={reconstructed_normal.step_time_us:.3f}")
    print(
        "isolated_step_us="
        f"{reconstructed_isolated.step_time_us:.3f}"
    )
    print(f"step_reduction_us={reduction_us:.3f}")
    print(f"step_reduction_pct={reduction_pct:.6f}")
    print(f"reference_step_reduction_pct={reference_reduction_pct:.6f}")
    print(f"summary={output_path.resolve()}")
    print(f"nodes_csv={nodes_csv_path.resolve()}")
    print(f"kernels_csv={kernels_csv_path.resolve()}")


if __name__ == "__main__":
    main()
