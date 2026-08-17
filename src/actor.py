import ray
import torch
import os
import re
import time
import hashlib
import json
import tempfile
from typing import Any
import gc
import statistics
from datetime import timedelta
from torch.autograd.graph import set_warn_on_accumulate_grad_stream_mismatch
import torch.distributed as dist
from collections import defaultdict

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .state import (
    create_logger,
    LOG_LEVEL,
)
from .fx import _deserialize_graphmodule, _serialize_graphmodule
from .model_state import (
    DEFAULT_MODEL_SEED,
    dynamo_buffer_name_to_state_dict_key,
    dynamo_parameter_name_to_state_dict_key,
    initialize_parameter_,
)
from .executors import (
    CommunicationExecutor,
    ComputeExecutor,
    DagExecutor,
    PreparedDagExecution,
)
from .ordering import _serial_topological_order
from .profiling import (
    ActorProfileResult,
    DEFAULT_NORMAL_SCOPE,
    DEFAULT_ISOLATION_SCOPE,
    INTERFERENCE_FREE_PROFILE_MODE,
    NodeProfile,
    PROFILE_MODES,
    finalized_dag_fingerprint,
    normalize_profile_mode,
)
from .runtime import (
    BufferStore,
    EventStore,
    ParamStorage,
    RuntimeState,
    StageStore,
)
from .tasks import training_dag_task_type as _training_dag_task_type

CLEANUP_MEMORY = False

logger = create_logger("actor", LOG_LEVEL)


def _kineto_node_attribution_audit(
    profiler, *, trace_path: str | None = None, detailed: bool = True
) -> dict[str, dict[str, Any]]:
    """Audit CUDA launch/activity correlation for every Piper node range.

    The compact path is linear in captured activity and is sufficient for
    simulator timing and exact kernel-signature checks.  ``detailed=True``
    additionally attributes cross-node overlap and CPU launch gaps; that
    diagnostic is intentionally opt-in because it compares every node range
    with every other range.
    """

    def merge_intervals(
        intervals: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        merged: list[tuple[float, float]] = []
        for start, end in sorted(intervals):
            if end <= start:
                continue
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        return merged

    def interval_total(intervals: list[tuple[float, float]]) -> float:
        return sum(end - start for start, end in intervals)

    def intersection_total(
        left: list[tuple[float, float]], right: list[tuple[float, float]]
    ) -> float:
        total = 0.0
        left_index = 0
        right_index = 0
        while left_index < len(left) and right_index < len(right):
            left_start, left_end = left[left_index]
            right_start, right_end = right[right_index]
            total += max(0.0, min(left_end, right_end) - max(left_start, right_start))
            if left_end <= right_end:
                left_index += 1
            else:
                right_index += 1
        return total
    delete_trace = trace_path is None
    if trace_path is None:
        with tempfile.NamedTemporaryFile(suffix=".kineto.json", delete=False) as tmp:
            trace_path = tmp.name
    else:
        os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)
    try:
        profiler.export_chrome_trace(trace_path)
        with open(trace_path, "r", encoding="utf-8") as handle:
            events = json.load(handle).get("traceEvents", [])
    finally:
        if delete_trace:
            try:
                os.unlink(trace_path)
            except FileNotFoundError:
                pass

    ranges_by_thread: dict[
        tuple[Any, Any], list[tuple[float, float, str, int]]
    ] = {}
    range_records: dict[int, tuple[float, float, str]] = {}
    for event in events:
        name = str(event.get("name", ""))
        if (
            event.get("ph") != "X"
            or event.get("cat") != "user_annotation"
            or ":uid" not in name
        ):
            continue
        start = event.get("ts")
        duration = event.get("dur")
        if not isinstance(start, (int, float)) or not isinstance(duration, (int, float)):
            continue
        range_key = len(range_records)
        record = (float(start), float(start) + float(duration), name)
        range_records[range_key] = record
        ranges_by_thread.setdefault((event.get("pid"), event.get("tid")), []).append(
            (*record, range_key)
        )
    for ranges in ranges_by_thread.values():
        ranges.sort()

    external_id_to_range: dict[Any, int] = {}
    ranges_by_label: dict[str, int] = defaultdict(int)
    for ranges in ranges_by_thread.values():
        for _, _, label, _ in ranges:
            ranges_by_label[label] += 1

    # Node ranges are sequential on each actor dispatch thread. Associate each
    # nested CPU/CUDA-runtime event's external correlation ID with that range.
    for thread, ranges in ranges_by_thread.items():
        thread_events = sorted(
            (
                event for event in events
                if (event.get("pid"), event.get("tid")) == thread
                and event.get("ph") == "X"
            ),
            key=lambda event: float(event.get("ts", -1.0) or -1.0),
        )
        range_index = 0
        for event in thread_events:
            timestamp = event.get("ts")
            if not isinstance(timestamp, (int, float)):
                continue
            while range_index < len(ranges) and ranges[range_index][1] < float(timestamp):
                range_index += 1
            if range_index >= len(ranges):
                break
            start, end, _, range_key = ranges[range_index]
            if start <= float(timestamp) <= end:
                external_id = (event.get("args") or {}).get("External id")
                if external_id is not None:
                    external_id_to_range[external_id] = range_key

    launch_correlations_by_label: dict[str, set[Any]] = defaultdict(set)
    launch_records_by_correlation: dict[
        Any, list[tuple[float, float, str, str, int | None]]
    ] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        name = str(event.get("name", "")).lower()
        if not any(token in name for token in ("launch", "memcpy", "memset")):
            continue
        args = event.get("args") or {}
        range_key = external_id_to_range.get(args.get("External id"))
        label = None if range_key is None else range_records[range_key][2]
        correlation = args.get("correlation")
        start = event.get("ts")
        duration = event.get("dur")
        if (
            correlation is not None
            and isinstance(start, (int, float))
            and isinstance(duration, (int, float))
        ):
            launch_records_by_correlation[correlation].append((
                float(start),
                float(start) + float(duration),
                str(event.get("name", "")),
                str(event.get("cat", "")),
                range_key,
            ))
        if label is not None and correlation is not None:
            launch_correlations_by_label[label].add(correlation)

    activity_correlations_by_label: dict[str, set[Any]] = defaultdict(set)
    activity_count_by_label: dict[str, int] = defaultdict(int)
    activity_time_by_label: dict[str, float] = defaultdict(float)
    activity_types_by_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    activity_signatures_by_range: dict[
        int, dict[tuple[str, str], int]
    ] = defaultdict(lambda: defaultdict(int))
    gpu_activity_records: list[
        tuple[float, float, int | None, str, str, Any | None]
    ] = []
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        args = event.get("args") or {}
        range_key = external_id_to_range.get(args.get("External id"))
        start = event.get("ts")
        duration = event.get("dur")
        if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
            gpu_activity_records.append((
                float(start),
                float(start) + float(duration),
                range_key,
                str(category),
                str(event.get("name", "")),
                args.get("correlation"),
            ))
        if range_key is None:
            continue
        label = range_records[range_key][2]
        activity_count_by_label[label] += 1
        activity_types_by_label[label][str(category)] += 1
        activity_signatures_by_range[range_key][
            (str(category), str(event.get("name", "")))
        ] += 1
        if isinstance(duration, (int, float)):
            activity_time_by_label[label] += float(duration)
        correlation = args.get("correlation")
        if correlation is not None:
            activity_correlations_by_label[label].add(correlation)

    audit: dict[str, dict[str, Any]] = {}
    gpu_activity_records_by_range: dict[
        int | None, list[tuple[float, float, int | None, str, str, Any | None]]
    ] = defaultdict(list)
    for record in gpu_activity_records:
        gpu_activity_records_by_range[record[2]].append(record)
    for label, range_count in ranges_by_label.items():
        launches = launch_correlations_by_label.get(label, set())
        activities = activity_correlations_by_label.get(label, set())
        unmatched = launches - activities
        range_activity_signatures = []
        label_range_keys = sorted(
            (
                range_key for range_key, (start, _, record_label) in range_records.items()
                if record_label == label
            ),
            key=lambda range_key: (range_records[range_key][0], range_key),
        )
        for occurrence, range_key in enumerate(label_range_keys):
            def activity_launch_order(
                record: tuple[
                    float, float, int | None, str, str, Any | None
                ],
            ) -> tuple[float, float, float, str, str, str]:
                correlation = record[5]
                launch_records = launch_records_by_correlation.get(
                    correlation, ()
                )
                # Cross-stream GPU start order is scheduling-dependent.  CPU
                # launch order is the stable invariant that validates the two
                # profile modes issued the same activity sequence.
                launch_start = min(
                    (launch[0] for launch in launch_records),
                    default=record[0],
                )
                return (
                    launch_start,
                    record[0],
                    record[1],
                    record[3],
                    record[4],
                    "" if correlation is None else str(correlation),
                )

            target_records = sorted(
                gpu_activity_records_by_range.get(range_key, []),
                key=activity_launch_order,
            )
            target_intervals = [
                (start, end) for start, end, _, _, _, _ in target_records
            ]
            target_union = merge_intervals(target_intervals)
            target_sum_us = interval_total(target_intervals)
            target_union_us = interval_total(target_union)
            target_envelope_us = (
                0.0 if not target_union
                else target_union[-1][1] - target_union[0][0]
            )
            target_envelope = (
                [] if not target_union
                else [(target_union[0][0], target_union[-1][1])]
            )
            entries = [
                {
                    "category": category,
                    "name": name,
                    "count": int(count),
                }
                for (category, name), count in sorted(
                    activity_signatures_by_range.get(range_key, {}).items()
                )
            ]
            encoded = json.dumps(
                entries, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            activity_sequence = [
                {"category": category, "name": name}
                for _, _, _, category, name, _ in target_records
            ]
            sequence_encoded = json.dumps(
                activity_sequence, separators=(",", ":")
            ).encode("utf-8")
            compact_signature = {
                "occurrence": int(occurrence),
                "activity_count": sum(entry["count"] for entry in entries),
                "signature_sha256": hashlib.sha256(encoded).hexdigest(),
                "activity_sequence_sha256": hashlib.sha256(
                    sequence_encoded
                ).hexdigest(),
                "activity_time_sum_us": target_sum_us,
                "activity_union_time_us": target_union_us,
                "activity_intra_node_overlap_us": max(
                    0.0, target_sum_us - target_union_us
                ),
                "activity_envelope_time_us": target_envelope_us,
                "activity_idle_gap_us": max(
                    0.0, target_envelope_us - target_union_us
                ),
                # Consumed centrally for the exact IF/normal activity audit,
                # then removed from the durable profile artifact.
                "activities": entries,
                "activity_sequence": activity_sequence,
            }
            if not detailed:
                range_activity_signatures.append(compact_signature)
                continue
            other_range_keys = [
                other_range_key
                for other_range_key in gpu_activity_records_by_range
                if other_range_key is not None and other_range_key != range_key
            ]
            other_records = [
                record
                for other_range_key in other_range_keys
                for record in gpu_activity_records_by_range[other_range_key]
            ]
            unattributed_records = gpu_activity_records_by_range.get(None, [])
            other_union = merge_intervals([
                (start, end) for start, end, _, _, _, _ in other_records
            ])
            unattributed_union = merge_intervals([
                (start, end) for start, end, _, _, _, _ in unattributed_records
            ])
            other_overlap_us = intersection_total(target_union, other_union)
            other_within_envelope_us = intersection_total(
                target_envelope, other_union
            )
            unattributed_overlap_us = intersection_total(
                target_union, unattributed_union
            )
            unattributed_within_envelope_us = intersection_total(
                target_envelope, unattributed_union
            )
            overlap_by_label: dict[str, dict[str, float]] = defaultdict(
                lambda: {"concurrent_overlap_us": 0.0, "within_envelope_us": 0.0}
            )
            for other_range_key in sorted(other_range_keys):
                label_records = gpu_activity_records_by_range[other_range_key]
                label_union = merge_intervals([
                    (start, end)
                    for start, end, _, _, _, _ in label_records
                ])
                label_overlap_us = intersection_total(target_union, label_union)
                label_within_envelope_us = intersection_total(
                    target_envelope, label_union
                )
                if label_overlap_us <= 0.0 and label_within_envelope_us <= 0.0:
                    continue
                other_label = range_records[other_range_key][2]
                overlap_by_label[other_label]["concurrent_overlap_us"] += (
                    label_overlap_us
                )
                overlap_by_label[other_label]["within_envelope_us"] += (
                    label_within_envelope_us
                )

            def preferred_launch_record(
                correlation: Any,
            ) -> tuple[float, float, str, str, int | None] | None:
                candidates = launch_records_by_correlation.get(correlation, ())
                if not candidates:
                    return None
                local = [record for record in candidates if record[4] == range_key]
                selected = local or list(candidates)
                runtime = [record for record in selected if record[3] == "cuda_runtime"]
                # The runtime callback is the application-visible launch API;
                # a nested driver callback for the same correlation would
                # otherwise double-count submission time.
                return min(runtime or selected, key=lambda record: (record[0], record[1]))

            kernel_records = sorted(
                (record for record in target_records if record[3] == "kernel"),
                key=lambda record: (record[0], record[1]),
            )
            launches_for_kernels = [
                preferred_launch_record(record[5]) for record in kernel_records
            ]
            matched_launches = [
                launch for launch in launches_for_kernels if launch is not None
            ]
            launches_in_cpu_order = sorted(
                matched_launches, key=lambda launch: (launch[0], launch[1])
            )
            inter_launch_cpu_gaps = [
                max(0.0, current[0] - previous[1])
                for previous, current in zip(
                    launches_in_cpu_order, launches_in_cpu_order[1:]
                )
            ]
            gpu_idle_gap_us = 0.0
            idle_before_launch_begin_us = 0.0
            idle_inside_launch_api_us = 0.0
            idle_after_launch_return_us = 0.0
            unclassified_gpu_idle_us = 0.0
            gap_details = []
            for index in range(1, len(kernel_records)):
                previous = kernel_records[index - 1]
                current = kernel_records[index]
                previous_end = previous[1]
                current_start = current[0]
                gpu_gap = max(0.0, current_start - previous_end)
                if gpu_gap <= 0.0:
                    continue
                gpu_idle_gap_us += gpu_gap
                launch = launches_for_kernels[index]
                if launch is None:
                    unclassified_gpu_idle_us += gpu_gap
                    gap_details.append({
                        "gpu_gap_us": gpu_gap,
                        "previous_kernel": previous[4][:240],
                        "next_kernel": current[4][:240],
                        "launch_record_missing": True,
                    })
                    continue
                launch_start, launch_end, launch_name, _, _ = launch
                before_launch = max(
                    0.0, min(current_start, launch_start) - previous_end
                )
                inside_launch = max(
                    0.0,
                    min(current_start, launch_end)
                    - max(previous_end, launch_start),
                )
                after_launch = max(
                    0.0, current_start - max(previous_end, launch_end)
                )
                classified = before_launch + inside_launch + after_launch
                idle_before_launch_begin_us += before_launch
                idle_inside_launch_api_us += inside_launch
                idle_after_launch_return_us += after_launch
                unclassified_gpu_idle_us += max(0.0, gpu_gap - classified)
                gap_details.append({
                    "gpu_gap_us": gpu_gap,
                    "previous_kernel_end_ts_us": previous_end,
                    "next_kernel_start_ts_us": current_start,
                    "previous_kernel": previous[4][:240],
                    "next_kernel": current[4][:240],
                    "next_launch_api": launch_name,
                    "next_launch_start_ts_us": launch_start,
                    "next_launch_end_ts_us": launch_end,
                    "before_launch_begin_us": before_launch,
                    "inside_launch_api_us": inside_launch,
                    "after_launch_return_us": after_launch,
                    "next_launch_api_duration_us": launch_end - launch_start,
                })
            largest_gpu_gaps = sorted(
                gap_details, key=lambda item: -float(item["gpu_gap_us"])
            )[:5]
            range_activity_signatures.append({
                **compact_signature,
                "overlap_with_other_node_activity_us": other_overlap_us,
                "other_node_activity_within_envelope_us": (
                    other_within_envelope_us
                ),
                "overlap_with_unattributed_activity_us": unattributed_overlap_us,
                "unattributed_activity_within_envelope_us": (
                    unattributed_within_envelope_us
                ),
                "overlapping_node_ranges": [
                    {"label": other_label, **overlap}
                    for other_label, overlap in sorted(
                        overlap_by_label.items(),
                        key=lambda item: (
                            -item[1]["within_envelope_us"], item[0]
                        ),
                    )
                ],
                "cpu_launch_timing": {
                    "kernel_activity_count": len(kernel_records),
                    "matched_kernel_launch_count": len(matched_launches),
                    "launch_api_time_sum_us": sum(
                        launch[1] - launch[0] for launch in matched_launches
                    ),
                    "inter_launch_cpu_gap_sum_us": sum(inter_launch_cpu_gaps),
                    "inter_launch_cpu_gap_max_us": (
                        max(inter_launch_cpu_gaps) if inter_launch_cpu_gaps else 0.0
                    ),
                    "gpu_idle_between_kernels_us": gpu_idle_gap_us,
                    "gpu_idle_before_next_launch_begin_us": (
                        idle_before_launch_begin_us
                    ),
                    "gpu_idle_inside_next_launch_api_us": idle_inside_launch_api_us,
                    "gpu_idle_after_next_launch_return_us": (
                        idle_after_launch_return_us
                    ),
                    "gpu_idle_unclassified_us": unclassified_gpu_idle_us,
                    "largest_gpu_gaps": largest_gpu_gaps,
                },
            })
        audit[label] = {
            "range_count": int(range_count),
            "launch_count": len(launches),
            "matched_launch_count": len(launches - unmatched),
            "unmatched_launch_count": len(unmatched),
            "launch_coverage_pct": (
                100.0 if not launches else 100.0 * len(launches - unmatched) / len(launches)
            ),
            "gpu_activity_count": int(activity_count_by_label.get(label, 0)),
            "gpu_activity_time_total_us": float(activity_time_by_label.get(label, 0.0)),
            "gpu_activity_time_mean_us": (
                float(activity_time_by_label.get(label, 0.0)) / range_count
            ),
            "gpu_activity_types": dict(activity_types_by_label.get(label, {})),
            "range_activity_signatures": range_activity_signatures,
        }
    return audit


def _kineto_correlated_activity_envelope_samples_us(
    measured: dict[str, Any],
    *,
    node_uid: str,
) -> tuple[float, ...]:
    """Return one correlated-GPU-activity envelope per node range.

    ``torch.profiler.key_averages()`` does not recursively include every GPU
    activity launched by nested Inductor/NCCL CPU operators in an enclosing
    Piper ``record_function`` range.  The attribution audit follows CUPTI
    external/correlation IDs explicitly.  Its per-occurrence first-activity-
    start to last-activity-end envelope captures both the kernels and the
    GPU-side gaps between them, which together determine node completion.
    P2P nodes are replaced separately by an isolated directional microprofile
    so their rendezvous waiting is not folded into transfer latency.
    """
    unmatched_launches = int(measured.get("unmatched_launch_count", 0))
    if unmatched_launches:
        raise RuntimeError(
            f"Kineto attribution for node {node_uid} left "
            f"{unmatched_launches} CUDA launches unmatched"
        )

    occurrences = tuple(measured.get("range_activity_signatures", ()))
    range_count = int(measured.get("range_count", len(occurrences)))
    if range_count <= 0 or len(occurrences) != range_count:
        raise RuntimeError(
            f"Kineto attribution for node {node_uid} has inconsistent range "
            f"counts: range_count={range_count}, occurrences={len(occurrences)}"
        )
    try:
        return tuple(
            float(occurrence["activity_envelope_time_us"])
            for occurrence in occurrences
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Kineto attribution for node {node_uid} has no correlated "
            "GPU-activity envelope"
        ) from exc


class _CudaEventTimingRecorder:
    """Collect one CUDA-event sample for every dispatched DAG node."""

    def __init__(
        self,
        *,
        coordinator_clock_offset_ns: int = 0,
        clock_sync_uncertainty_ns: int = 0,
    ) -> None:
        self._node_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._active_node_starts: dict[str, torch.cuda.Event] = {}
        self._prologue_events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None
        self._prologue_start: torch.cuda.Event | None = None
        self.node_samples_us: dict[str, float] = {}
        self.node_start_offsets_us: dict[str, float] = {}
        self.prologue_sample_us: float | None = None
        self.timeline_origin_us: float | None = None
        self.timeline_origin_uncertainty_us: float | None = None
        self._coordinator_clock_offset_ns = int(coordinator_clock_offset_ns)
        self._clock_sync_uncertainty_ns = int(clock_sync_uncertainty_ns)

    @staticmethod
    def _event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def before_prologue(self, stream: torch.cuda.Stream) -> None:
        start = self._event()
        host_before_ns = time.monotonic_ns()
        start.record(stream)
        # Profiling starts from an idle device.  Synchronizing this one anchor
        # event maps the actor-local CUDA timeline to the coordinator's
        # calibrated monotonic clock without synchronizing any timed node.
        start.synchronize()
        host_after_ns = time.monotonic_ns()
        host_midpoint_ns = (host_before_ns + host_after_ns) // 2
        self.timeline_origin_us = (
            host_midpoint_ns + self._coordinator_clock_offset_ns
        ) / 1000.0
        self.timeline_origin_uncertainty_us = (
            (host_after_ns - host_before_ns) / 2.0
            + self._clock_sync_uncertainty_ns
        ) / 1000.0
        self._prologue_start = start

    def after_prologue(self, stream: torch.cuda.Stream) -> None:
        if self._prologue_start is None:
            raise RuntimeError("profile recorder saw prologue end before start")
        end = self._event()
        end.record(stream)
        self._prologue_events = (self._prologue_start, end)
        self._prologue_start = None

    def before_node(self, node: Any, stream: torch.cuda.Stream) -> None:
        start = self._event()
        start.record(stream)
        self._active_node_starts[str(node.uid)] = start

    def after_node(self, node: Any, stream: torch.cuda.Stream) -> None:
        uid = str(node.uid)
        try:
            start = self._active_node_starts.pop(uid)
        except KeyError as exc:
            raise RuntimeError(f"profile recorder saw node end before start: {uid}") from exc
        end = self._event()
        end.record(stream)
        self._node_events[uid] = (start, end)

    def finish(self) -> None:
        if self._active_node_starts:
            raise RuntimeError(
                "profile recorder has unfinished nodes: "
                f"{sorted(self._active_node_starts)}"
            )
        self.node_samples_us = {
            uid: float(start.elapsed_time(end)) * 1000.0
            for uid, (start, end) in self._node_events.items()
        }
        if self._prologue_events is not None:
            start, end = self._prologue_events
            self.prologue_sample_us = float(start.elapsed_time(end)) * 1000.0
            # CUDA events carry a device-wide timestamp even when recorded on
            # different streams.  Reuse the existing events to capture exact
            # rank-local GPU start positions without adding callbacks, events,
            # or per-node synchronization to the hot path.
            self.node_start_offsets_us = {
                uid: float(start.elapsed_time(node_start)) * 1000.0
                for uid, (node_start, _node_end) in self._node_events.items()
            }

def _disable_functorch_donated_buffers() -> None:
    import importlib

    config = importlib.import_module("torch._functorch.config")
    config.donated_buffer = False


def _get_rank(pp_rank, dp_rank, pp_degree):
    return pp_rank + dp_rank * pp_degree


def _create_actors(
    num_actors,
    optim_class,
    profile=False,
    no_nvtx: bool = False,
    pg=None,
    temp_dir: str = None,
    use_inductor: bool = False,
    pp_outer: bool = False,
    model_seed: int = DEFAULT_MODEL_SEED,
):
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    world_size = int(os.environ["PIPER_WORLD_SIZE"])
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])
    pp_degree = int(os.environ["PIPER_PP_DEGREE"])

    from .state import piper_metadata

    for pp_rank in range(num_actors):
        global_rank = _get_rank(pp_rank, dp_rank, pp_degree)
        nsight_env = {"nsight": {
            "t": "cuda,cudnn,cublas,nvtx",
            "sample": "process-tree",
            "backtrace": "dwarf",
            "cudabacktrace": "sync:0,memory:0",
            "python-backtrace": "cuda",
            "stop-on-exit": "true",
        }} if profile else {}
        forwarded_env = {
            name: os.environ[name]
            for name in (
                "CPATH",
                "C_INCLUDE_PATH",
                "CPLUS_INCLUDE_PATH",
                "LIBRARY_PATH",
                "TRITON_CACHE_DIR",
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
                "NCCL_CUMEM_ENABLE",
                "NCCL_DEBUG",
                "NCCL_DEBUG_FILE",
                "NCCL_DEBUG_SUBSYS",
                "NCCL_NET",
                "NCCL_SOCKET_IFNAME",
                "GLOO_SOCKET_IFNAME",
                "PIPER_PROCESS_GROUP_TIMEOUT_MINUTES",
            )
            if os.environ.get(name)
        }
        nccl_env = {
            "env_vars": {
                # "NCCL_SOCKET_IFNAME": "ens32",
                # "GLOO_SOCKET_IFNAME": "ens32",
                # "NCCL_P2P_DISABLE": "1",
                # "NCCL_DEBUG": "INFO",
                **forwarded_env,
                **({"TMPDIR": temp_dir} if (profile and temp_dir) else {}),
            }
        }
        # The shared placement plan is authoritative when an innermost group
        # spans multiple node-sized bundles. Keep accepting raw placement groups
        # for callers that rely on Piper's original one-bundle-per-group layout.
        ray_pg = getattr(pg, "placement_group", pg)
        placement_plan = getattr(pg, "plan", None)
        bundle_index = (
            placement_plan.actor_bundle_index(pp_rank, dp_rank)
            if placement_plan is not None
            else (pp_rank if pp_outer else dp_rank)
        )
        actor = PiperActor.options(
            num_gpus=0.6,
            runtime_env={**nsight_env, **nccl_env},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=ray_pg,
                placement_group_bundle_index=bundle_index,
            ),
        ).remote(
            pp_rank,
            optim_class,
            world_size,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
            no_nvtx=no_nvtx,
            use_inductor=use_inductor,
            model_seed=model_seed,
        )
        piper_metadata.actors[pp_rank] = actor


@ray.remote
class PiperActor:
    def __init__(
        self,
        pp_rank,
        optim_class,
        world_size,
        dp_rank=0,
        dp_degree=1,
        pp_degree=1,
        no_nvtx: bool = False,
        use_inductor: bool = False,
        model_seed: int = DEFAULT_MODEL_SEED,
    ):
        self.logger = create_logger("actor", LOG_LEVEL)

        # BWD_I uses torch.autograd.grad (not .backward), so AccumulateGrad nodes are
        # traversed for stream-sync bookkeeping but never accumulate to p.grad.
        # Suppress the spurious stream-mismatch warning.
        set_warn_on_accumulate_grad_stream_mismatch(False)

        self.optim_class = optim_class
        self.use_inductor = bool(use_inductor)
        self.model_seed = int(model_seed)
        if self.use_inductor:
            _disable_functorch_donated_buffers()
        self.runtime = RuntimeState(
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
            world_size=world_size,
            no_nvtx=no_nvtx,
        )

        self.logger.debug(
            f"Initializing Ray actor {self.runtime.global_rank} GPU {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        self.inputs = None
        self.labels = None

        self.stages = StageStore()
        # accumuate loss for each microbatch
        self.loss = []

        # DAG execution state
        self.dag = None
        self.sorted_dag_nodes = None
        self.prepared_dag_execution: PreparedDagExecution | None = None
        self.profile_prepared_dag_executions: dict[
            str, PreparedDagExecution
        ] = {}
        # Interference-free and normal profiles must never be combined:
        # they answer different simulation questions from the same DAG.
        self._profile_node_samples: dict[
            str, dict[str, list[float]]
        ] = {}
        self._profile_node_start_offsets: dict[
            str, dict[str, list[float]]
        ] = {}
        self._profile_prologue_samples: dict[str, list[float]] = {}
        self._profile_timeline_origins_us: dict[str, list[float]] = {}
        self._profile_timeline_origin_uncertainties_us: dict[
            str, list[float]
        ] = {}
        self._profile_coordinator_clock_offset_ns = 0
        self._profile_clock_sync_uncertainty_ns = 0

        self.buffers = BufferStore()
        self.events = EventStore()
        self.communication = CommunicationExecutor(self.runtime, self.stages, self.logger)
        self.compute = ComputeExecutor(self.runtime, self.stages, self.logger)
        self.params = ParamStorage(self.runtime, self.stages, self.logger)
        self.dag_executor = DagExecutor(
            self.runtime,
            self.stages,
            self.buffers,
            self.events,
            self.params,
            self.communication,
            self.compute,
            self.logger,
        )

        # Non-trainable constant tensor attributes (e.g. freqs_cis, mask) pushed
        # from the coordinator before compilation so _load_stage can fill them in
        # instead of zero-initializing.  Keyed by bare attribute name (e.g. "freqs_cis").
        self.model_const_attrs: dict = {}

    def get_and_reset_peak_memory_stats(self) -> tuple:
        """Return (global_rank, max_memory_allocated_bytes) and reset peak stats."""
        max_alloc = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self.runtime.global_rank, max_alloc

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    def _nvtx_push(self, label: str) -> None:
        self.runtime.nvtx_push(label)

    def _nvtx_pop(self) -> None:
        self.runtime.nvtx_pop()

    def start_pytorch_profiler(
        self,
        *,
        warmup_iters: int = 0,
        active_iters: int = 1,
        activities: tuple[str, ...] | None = None,
    ) -> None:
        """Begin a scheduled torch.profiler session at DAG boundaries.

        Each node's execution in DagExecutor is wrapped in a record_function
        labelled the same as its NVTX range, so the resulting trace identifies
        per-node work.
        """
        if int(warmup_iters) < 0 or int(active_iters) <= 0:
            raise ValueError("profiler warmup must be non-negative and active iterations positive")
        activity_names = activities or ("cpu", "cuda")
        activity_by_name = {
            "cpu": torch.profiler.ProfilerActivity.CPU,
            "cuda": torch.profiler.ProfilerActivity.CUDA,
        }
        unknown = sorted(set(activity_names) - set(activity_by_name))
        if unknown:
            raise ValueError(f"unknown PyTorch profiler activities: {unknown}")
        selected_activities = [activity_by_name[name] for name in activity_names]
        if not selected_activities:
            raise ValueError("at least one PyTorch profiler activity is required")
        self.runtime.torch_profiler = torch.profiler.profile(
            activities=selected_activities,
            schedule=torch.profiler.schedule(
                wait=0, warmup=int(warmup_iters), active=int(active_iters), repeat=1
            ),
        )
        self.runtime.torch_profiler.__enter__()
        # ``profile_run_dag`` records allocator deltas whenever a profiler is
        # active.  Initialize the sink here as well as in the CUPTI convenience
        # wrapper so CPU-only profiler experiments follow the same execution
        # path without depending on which start method was used.
        self._cupti_allocator_iteration_stats = []
        self.runtime.pytorch_profiler_enabled = True
        self.logger.debug(f"Actor {self.runtime.global_rank}: PyTorch profiler started")

    def stop_pytorch_profiler_experiment(
        self, *, trace_path: str | None = None
    ) -> str | None:
        """Stop a profiler used by a timing experiment, optionally exporting it.

        Unlike :meth:`stop_pytorch_profiler`, this does not require every actor
        to export a trace.  That keeps repeated profiler-overhead measurements
        from spending time and disk space serializing traces that are not part
        of the experiment result.  Profiler teardown and export happen after
        the synchronized step timer has stopped.
        """

        if not self.runtime.pytorch_profiler_enabled:
            raise RuntimeError("PyTorch profiler is not active")
        profiler = self.runtime.torch_profiler
        self.runtime.pytorch_profiler_enabled = False
        profiler.__exit__(None, None, None)
        if trace_path is not None:
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            profiler.export_chrome_trace(trace_path)
        self.runtime.torch_profiler = None
        return trace_path

    def step_pytorch_profiler(self) -> None:
        """Advance an active profiler after externally timed GPU work."""

        if not self.runtime.pytorch_profiler_enabled:
            raise RuntimeError("PyTorch profiler is not active")
        self.runtime.torch_profiler.step()

    def start_cupti_node_profiler(
        self, *, warmup_iters: int = 0, active_iters: int = 1
    ) -> None:
        """Start CUPTI/Kineto capture for node ``record_function`` ranges.

        Unlike Piper's CUDA-event profiler, this adds no timing events to the
        DAG streams.  Kineto attributes CUDA activity to the existing node
        ranges through CUPTI correlation IDs.
        """
        self._cupti_allocator_iteration_stats = []
        self.start_pytorch_profiler(
            warmup_iters=warmup_iters, active_iters=active_iters
        )

    @staticmethod
    def _cuda_allocator_stats_snapshot() -> dict[str, int]:
        stats = torch.cuda.memory_stats()
        keys = (
            "num_alloc_retries",
            "num_ooms",
            "num_sync_all_streams",
            "num_device_alloc",
            "num_device_free",
            "allocated_bytes.all.current",
            "active_bytes.all.current",
            "reserved_bytes.all.current",
            "inactive_split_bytes.all.current",
            "segment.all.current",
        )
        return {key: int(stats[key]) for key in keys if key in stats}

    @staticmethod
    def _cuda_allocator_stream_snapshot() -> dict[str, dict[str, int]]:
        """Summarize caching-allocator segments by their owning CUDA stream."""

        summary: dict[str, dict[str, int]] = {}
        for segment in torch.cuda.memory_snapshot():
            stream = str(int(segment.get("stream", 0)))
            values = summary.setdefault(
                stream,
                {
                    "segments": 0,
                    "reserved_bytes": 0,
                    "allocated_bytes": 0,
                    "active_bytes": 0,
                    "inactive_bytes": 0,
                    "active_awaiting_free_bytes": 0,
                },
            )
            values["segments"] += 1
            values["reserved_bytes"] += int(segment.get("total_size", 0))
            values["allocated_bytes"] += int(segment.get("allocated_size", 0))
            values["active_bytes"] += int(segment.get("active_size", 0))
            for block in segment.get("blocks", ()):
                state = block.get("state")
                size = int(block.get("size", 0))
                if state == "inactive":
                    values["inactive_bytes"] += size
                elif state == "active_awaiting_free":
                    values["active_awaiting_free_bytes"] += size
        return summary

    def stop_pytorch_profiler(self, profile_dir: str) -> str:
        """End the profiler session and export a chrome trace.

        The file is named ``dp{dp_rank}_pp{pp_rank}.json`` inside *profile_dir*
        so the harness can group same-dp-rank actors. Returns the written path.
        """
        self.runtime.pytorch_profiler_enabled = False
        self.runtime.torch_profiler.__exit__(None, None, None)
        os.makedirs(profile_dir, exist_ok=True)
        filepath = os.path.join(
            profile_dir, f"dp{self.runtime.dp_rank}_pp{self.runtime.pp_rank}.json"
        )
        self.runtime.torch_profiler.export_chrome_trace(filepath)
        self.runtime.torch_profiler = None
        self.logger.info(
            f"Actor {self.runtime.global_rank}: PyTorch profiler trace written to {filepath}"
        )
        return filepath

    def stop_cupti_node_profiler(
        self, *, trace_path: str | None = None, detailed_audit: bool = True
    ) -> dict[str, dict[str, Any]]:
        """Return CUPTI-attributed CUDA totals keyed by Piper node range."""
        if not self.runtime.pytorch_profiler_enabled:
            raise RuntimeError("CUPTI node profiler is not active")
        profiler = self.runtime.torch_profiler
        self.runtime.pytorch_profiler_enabled = False
        profiler.__exit__(None, None, None)
        attribution_audit = _kineto_node_attribution_audit(
            profiler, trace_path=trace_path, detailed=detailed_audit
        )
        results: dict[str, dict[str, Any]] = {}
        for event in profiler.key_averages():
            key = str(event.key)
            if ":uid" not in key:
                continue
            count = max(1, int(event.count))
            results[key] = {
                "count": count,
                "device_time_total_us": float(event.device_time_total),
                "device_time_mean_us": float(event.device_time_total) / count,
                "self_device_time_total_us": float(event.self_device_time_total),
                "self_device_time_mean_us": (
                    float(event.self_device_time_total) / count
                ),
            }
            results[key].update(attribution_audit.get(key, {}))
        self._last_cupti_attribution_audit = attribution_audit
        self._last_cupti_trace_path = trace_path
        self.runtime.torch_profiler = None
        return results

    def stop_cupti_profile_result(
        self,
        *,
        estimator: str = "median",
        profile_mode: str = "interference_free",
        p2p_baselines_by_comm_id: dict[str, float] | None = None,
        trace_output_dir: str | None = None,
        trace_dp_ranks: tuple[int, ...] | None = None,
    ) -> ActorProfileResult:
        """Convert a Kineto/CUPTI capture into the normal profile shard type.

        Kineto's CUDA activity durations are associated with the ``record_function``
        range that dispatches each Piper node.  No timing events are inserted on
        the workload streams, and replay derives all starts from the DAG.
        """
        mode = normalize_profile_mode(profile_mode)
        if self.dag is None:
            raise RuntimeError("load_training_dag() must precede profiling")
        capture_trace = (
            trace_output_dir is not None
            and (
                trace_dp_ranks is None
                or int(self.runtime.dp_rank) in set(trace_dp_ranks)
            )
        )
        trace_path = (
            os.path.join(
                str(trace_output_dir),
                f"{mode}_rank{self.runtime.global_rank}_dp{self.runtime.dp_rank}"
                f"_pp{self.runtime.pp_rank}.json",
            )
            if capture_trace
            else None
        )
        captured = self.stop_cupti_node_profiler(
            trace_path=trace_path, detailed_audit=False
        )
        allocator_iteration_stats = list(
            getattr(self, "_cupti_allocator_iteration_stats", ())
        )
        node_profiles: dict[str, NodeProfile] = {}
        node_attribution_audit: dict[str, dict[str, Any]] = {}
        for uid, node in self.dag.nodes.items():
            if getattr(node, "node_kind", None) == "ORDER_DUMMY":
                node_profiles[uid] = NodeProfile.skipped(
                    "order_dummy",
                    timing_source="kineto_cupti",
                    isolation_scope=(
                        DEFAULT_ISOLATION_SCOPE
                        if mode == "interference_free"
                        else DEFAULT_NORMAL_SCOPE
                    ),
                )
                continue
            baseline_us = (p2p_baselines_by_comm_id or {}).get(
                str(node.node_meta.get("comm_id", ""))
            )
            matching = [
                value for key, value in captured.items()
                if key.endswith(f":uid{uid}")
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"expected one Kineto range for node {uid}, found {len(matching)}"
                )
            measured = dict(matching[0])
            duration_samples_us = _kineto_correlated_activity_envelope_samples_us(
                measured, node_uid=uid
            )
            duration_source = "correlated_gpu_activity_envelope"
            if (
                baseline_us is not None
                and getattr(node, "node_kind", None) in {"SEND_COMM", "RECV_COMM"}
            ):
                # Scheduled P2P ranges include rendezvous waiting.  Preserve
                # the separately measured transfer latency while keeping its
                # samples aligned with the captured profile iterations.
                duration_samples_us = tuple(
                    float(baseline_us) for _ in duration_samples_us
                )
                duration_source = "isolated_directional_p2p_microprofile"
            node_attribution_audit[uid] = {
                key: value for key, value in matching[0].items()
                if key in {
                    "range_count", "launch_count", "matched_launch_count",
                    "unmatched_launch_count", "launch_coverage_pct",
                    "gpu_activity_count", "gpu_activity_time_total_us",
                    "gpu_activity_time_mean_us", "gpu_activity_types",
                    "range_activity_signatures",
                    "device_time_total_us", "device_time_mean_us",
                }
            }
            node_attribution_audit[uid]["simulator_duration_source"] = (
                duration_source
            )
            node_attribution_audit[uid]["simulator_duration_samples_us"] = list(
                duration_samples_us
            )
            if getattr(node, "node_kind", None) in {"SEND_COMM", "RECV_COMM"}:
                isolation_scope = "normal_directional_p2p"
            elif mode == "interference_free":
                isolation_scope = DEFAULT_ISOLATION_SCOPE
            else:
                isolation_scope = DEFAULT_NORMAL_SCOPE
            node_profiles[uid] = NodeProfile.from_samples(
                duration_samples_us,
                estimator=estimator,
                timing_source="kineto_cupti",
                isolation_scope=isolation_scope,
            )
        props = torch.cuda.get_device_properties(self.runtime.device)
        return ActorProfileResult(
            profile_mode=mode,
            graph_fingerprint=finalized_dag_fingerprint(self.dag),
            global_rank=self.runtime.global_rank,
            dp_rank=self.runtime.dp_rank,
            pp_rank=self.runtime.pp_rank,
            device_metadata={
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": int(props.total_memory),
                "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda),
                "timing_source": "kineto_cupti",
                "kineto_trace_path": trace_path,
                "cuda_stream_ids": {
                    **{
                        f"logical:{name}": int(stream.cuda_stream)
                        for name, stream in self.runtime.streams.items()
                    },
                    **{
                        f"p2p:{name}": int(stream.cuda_stream)
                        for name, stream in self.runtime.directional_p2p_streams.items()
                    },
                },
                "allocator_iteration_stats": allocator_iteration_stats,
                "kineto_attribution_audit": node_attribution_audit,
            },
            node_profiles=node_profiles,
        )

    def load_input(self, inputs):
        self.inputs = [inp.to(self.runtime.device) for inp in inputs]
        self.logger.debug(f"Actor {self.runtime.global_rank} loaded inputs {len(self.inputs)}")

    def load_labels(self, labels):
        self.labels = labels.to(self.runtime.device)
        self.logger.debug(f"Actor {self.runtime.global_rank} loaded labels {self.labels.shape}")

    def load_const_attrs(self, const_attrs: dict) -> None:
        """Store non-trainable constant tensor attributes (e.g. freqs_cis, mask).

        *const_attrs* maps bare attribute name → CPU tensor.  Values are moved to
        the actor's device so ``_load_stage`` can copy them directly.
        """
        self.model_const_attrs = {k: v.to(self.runtime.device) for k, v in const_attrs.items()}

    def export_model_parameter_state(self) -> dict[str, torch.Tensor]:
        """Return this actor's current eager-compatible parameter state.

        ZeRO-3 parameters are reconstructed collectively from the shards held
        by the same PP rank in every DP replica. All DP workers must therefore
        invoke this method in the same bucket order.
        """
        self.params.drain_pending_frees()
        torch.cuda.synchronize()
        state: dict[str, torch.Tensor] = {}
        stream = self.runtime.default_stream()

        for ubid, bucket in sorted(
            self.stages.buckets.items(), key=lambda item: str(item[0])
        ):
            name_by_arg_idx = dict(zip(bucket.param_idxs, bucket.param_names))
            if ubid in self.stages.param_sharded_ubids:
                shard_param = bucket.shard_param
                shard_info = bucket.param_shard_info
                if shard_param is None or shard_info is None:
                    raise RuntimeError(
                        f"cannot export sharded parameters for bucket {ubid}: "
                        "missing shard storage metadata"
                    )
                _shard_start, shard_size, orig_numel = shard_info
                gathered = torch.empty(
                    shard_size * self.runtime.dp_degree,
                    dtype=shard_param.dtype,
                    device=shard_param.device,
                )
                with torch.cuda.stream(stream):
                    dist.all_gather_into_tensor(
                        gathered, shard_param, group=self.runtime.dp_group
                    )
                stream.synchronize()
                for arg_idx, (_param, offset, numel, shape) in zip(
                    bucket.trainable_param_idxs, bucket.param_view_specs
                ):
                    if offset + numel > orig_numel:
                        raise RuntimeError(
                            f"parameter slice exceeds unpadded bucket {ubid}: "
                            f"offset={offset} numel={numel} orig_numel={orig_numel}"
                        )
                    value = gathered[offset : offset + numel].view(shape)
                    self._add_exported_parameter(
                        state, name_by_arg_idx.get(arg_idx), value
                    )
                continue

            for arg_idx in bucket.trainable_param_idxs:
                value = bucket.forward_args[arg_idx]
                if value is None:
                    raise RuntimeError(
                        f"cannot export parameter index {arg_idx} in bucket {ubid}: "
                        "parameter storage is missing"
                    )
                self._add_exported_parameter(
                    state, name_by_arg_idx.get(arg_idx), value
                )

        torch.cuda.synchronize()
        return state

    @staticmethod
    def _add_exported_parameter(
        state: dict[str, torch.Tensor],
        dynamo_name: str | None,
        value: torch.Tensor,
    ) -> None:
        if dynamo_name is None:
            raise RuntimeError("parameter bucket is missing its Dynamo placeholder name")
        state_key = dynamo_parameter_name_to_state_dict_key(dynamo_name)
        if state_key is None:
            raise RuntimeError(
                f"cannot translate Dynamo parameter placeholder {dynamo_name!r}"
            )
        if state_key in state:
            raise RuntimeError(f"duplicate exported model parameter {state_key!r}")
        state[state_key] = value.detach().cpu().clone()

    def get_node_ip_and_free_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]
        return ray.util.get_node_ip_address(), port

    def _join_process_groups(self, master_addr, master_port):

        self.logger.debug(f"Actor {self.runtime.global_rank} using GPU {os.environ['CUDA_VISIBLE_DEVICES']}, master addr {master_addr}:{master_port}")

        init_method = f"tcp://{master_addr}:{master_port}"

        self.runtime.device = f"cuda:{self.runtime.global_rank % torch.cuda.device_count()}"
        torch.cuda.set_device(self.runtime.device)

        if self.runtime.pp_degree > 1 or self.runtime.dp_degree > 1:
            timeout_minutes = float(
                os.environ.get("PIPER_PROCESS_GROUP_TIMEOUT_MINUTES", "10")
            )
            if timeout_minutes <= 0:
                raise ValueError(
                    "PIPER_PROCESS_GROUP_TIMEOUT_MINUTES must be positive, "
                    f"got {timeout_minutes}"
                )
            self._process_group_timeout = timedelta(minutes=timeout_minutes)
            dist.init_process_group(
                "nccl",
                init_method=init_method,
                rank=self.runtime.global_rank,
                world_size=self.runtime.world_size,
                timeout=self._process_group_timeout,
            )

            if self.runtime.dp_degree > 1:
                self._join_dp_process_group()
            if self.runtime.pp_degree > 1:
                self._join_pp_process_group()

            self.logger.debug(f"Actor {self.runtime.global_rank} joined process groups")

    def _join_dp_process_group(self):
        num_dp_groups = self.runtime.world_size // self.runtime.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [
                (dp_group_id + num_dp_groups * i) for i in range(self.runtime.dp_degree)
            ]
            # Two separate NCCL communicators over the same ranks: one for allreduce,
            # one for all2all.  Sharing a communicator causes both op types to run on
            # the same internal NCCL proxy stream, which prevents true overlap.
            process_group = dist.new_group(
                ranks=group_ranks,
                backend="nccl",
                timeout=self._process_group_timeout,
            )
            ep_process_group = dist.new_group(
                ranks=group_ranks,
                backend="nccl",
                timeout=self._process_group_timeout,
            )
            if self.runtime.global_rank % num_dp_groups == dp_group_id:
                self.runtime.dp_group = process_group
                self.runtime.ep_group = ep_process_group

    def _join_pp_process_group(self):
        num_pp_groups = self.runtime.world_size // self.runtime.pp_degree

        for pp_group_id in range(num_pp_groups):
            group_ranks = [
                (pp_group_id * self.runtime.pp_degree + i) for i in range(self.runtime.pp_degree)
            ]
            lo_hi_group = dist.new_group(
                ranks=group_ranks,
                backend="nccl",
                timeout=self._process_group_timeout,
            )
            hi_lo_group = dist.new_group(
                ranks=group_ranks,
                backend="nccl",
                timeout=self._process_group_timeout,
            )

            if self.runtime.global_rank in group_ranks:
                self.runtime.pp_lo_hi = lo_hi_group
                self.runtime.pp_hi_lo = hi_lo_group

    def _derive_dag_bucket_modes(self, training_dag: Any) -> None:
        self.stages.param_sharded_ubids = set()
        self.stages.grad_sharded_ubids = set()
        for node in training_dag.nodes.values():
            meta = getattr(node, "node_meta", {}) or {}
            ubid = meta.get("bucket_key")
            if ubid is None:
                continue
            if node.node_kind == "ALL_GATHER_COMM":
                self.stages.param_sharded_ubids.add(ubid)
            elif node.node_kind == "REDUCE_SCATTER_COMM":
                self.stages.grad_sharded_ubids.add(ubid)
        self.stages.zero_managed_ubids = self.stages.param_sharded_ubids | self.stages.grad_sharded_ubids

    def _relocate_meta_devices(self, gm) -> None:
        """Rewrite baked ``torch.device('meta')`` literals to the actor device.

        Meta-device compilation captures whatever device was current at trace
        time (inside ``type_as`` / ``.to(device=x.device)`` / device-aware
        factory ops like ``torch.zeros(..., device=...)``). That literal
        round-trips through serialization as ``meta`` and would otherwise make
        those ops emit meta tensors at runtime, mismatching the CUDA inputs.
        Lifted params/buffers/inputs are unaffected (placed on device directly),
        so this only touches device literals embedded in node args/kwargs.
        """
        import torch.fx as fx

        device = self.runtime.device
        replaced = 0

        def _fix(value):
            nonlocal replaced
            if isinstance(value, torch.device):
                if value.type == "meta":
                    replaced += 1
                    return torch.device(device)
                return value
            if isinstance(value, tuple):
                return tuple(_fix(v) for v in value)
            if isinstance(value, list):
                return [_fix(v) for v in value]
            if isinstance(value, dict):
                return {k: _fix(v) for k, v in value.items()}
            return value

        for module in gm.modules():
            if not isinstance(module, fx.GraphModule):
                continue
            before = replaced
            for node in module.graph.nodes:
                node.args = _fix(node.args)
                node.kwargs = _fix(node.kwargs)
            if replaced != before:
                module.recompile()

    def _load_stage(
        self,
        stage_id: int,
        modules_data: list,
        a2a_boundaries: dict = None,
        use_activation_checkpointing: bool = False,
    ) -> None:
        """Load a (possibly bucketed) stage.

        *modules_data* is a list of dicts, one per module/bucket, each with keys:
        ``gm_data``, ``graphargs``, ``input_idxs``, ``param_idxs``, ``bucket_key``.

        A non-bucketed stage is represented as a single-element list.
        All per-bucket data structures are keyed by ``bucket_key`` so run_dag
        can look up bucket data without knowing which stage owns a bucket.
        """
        self.logger.debug(
            f"Loading stage {stage_id} ({len(modules_data)} module(s)) on actor {self.runtime.global_rank}"
        )

        first_gm = None

        for b_idx, bd in enumerate(modules_data):
            ubid: Any = bd["bucket_key"]
            bucket = self.stages.ensure_bucket(ubid)
            ac_num_subgraphs = int(bd.get("ac_num_subgraphs", 1))
            ac_requested_subgraphs = int(bd.get("ac_requested_subgraphs", ac_num_subgraphs))
            gms = [_deserialize_graphmodule(gm_data) for gm_data in bd["gm_data_list"]] if "gm_data_list" in bd else [_deserialize_graphmodule(bd["gm_data"])]
            for gm in gms:
                self._relocate_meta_devices(gm)
            if self.use_inductor:
                compiled_gms = []
                for subgraph_idx, gm in enumerate(gms):
                    is_expert = bool(bd.get("is_expert"))
                    # EP changes the token dimension entering the expert
                    # segment. Compile that segment dynamically from its
                    # first invocation, matching TorchTitan's expert compiler
                    # wrapper and avoiding one specialization per routing load.
                    compiled_gm = torch.compile(
                        gm,
                        fullgraph=is_expert,
                        dynamic=True if is_expert else None,
                    )
                    compiled_gms.append(compiled_gm)
                    self.logger.debug(
                        f"[load_stage_compile] rank={self.runtime.global_rank} stage={stage_id} "
                        f"ubid={ubid} subgraph={subgraph_idx} compiled=True"
                    )
                gms = compiled_gms

            if b_idx == 0:
                first_gm = gms[0]

            forward_args = list(bd["graphargs"])
            b_input_idxs = list(bd["input_idxs"])
            b_param_idxs = list(bd["param_idxs"])
            apply_zero = bool(bd.get("apply_zero", True))
            shared_placeholder_names = list(
                bd.get("shared_placeholder_names", bd.get("placeholder_names", []))
            )
            # Extract FX placeholder names for each param index.
            bucket.param_names = [
                shared_placeholder_names[i] if i < len(shared_placeholder_names) else f"ubid{ubid}_p{i}"
                for i in b_param_idxs
            ]

            # Save input tensor metadata for pre-allocating FWD recv buffers.
            # Stored as a list of (shape, dtype, requires_grad) in input-slot order.
            recv_meta = []
            for i in b_input_idxs:
                meta = forward_args[i]
                if meta is not None:
                    recv_meta.append((tuple(meta.shape), meta.dtype,
                                      bool(getattr(meta, "requires_grad", False))))
                forward_args[i] = None  # clear slot; run_dag will fill it at execution time
            bucket.forward_input_meta = recv_meta

            bucket.input_idxs = b_input_idxs

            # Realize parameter tensors.
            realized = [None] * len(forward_args)
            for i, arg in enumerate(forward_args):
                if arg is None:
                    continue
                t = torch.empty(arg.shape, dtype=arg.dtype, device=self.runtime.device)
                if arg.requires_grad:
                    ph_name = (
                        shared_placeholder_names[i]
                        if i < len(shared_placeholder_names)
                        else ""
                    )
                    state_key = dynamo_parameter_name_to_state_dict_key(ph_name)
                    if state_key is None:
                        raise RuntimeError(
                            "cannot deterministically initialize trainable "
                            f"placeholder {ph_name!r}: unsupported Dynamo name"
                        )
                    initialize_parameter_(
                        t,
                        state_key,
                        self.model_seed,
                    )
                    t.requires_grad_(True)
                else:
                    # Non-trainable slot: fill lifted model buffers (freqs_cis,
                    # rope_cache, mask, ...) from the eager value supplied by
                    # piper_setup before falling back to zeros.
                    ph_name = (
                        shared_placeholder_names[i]
                        if i < len(shared_placeholder_names)
                        else ""
                    )
                    buffer_key = dynamo_buffer_name_to_state_dict_key(ph_name)
                    const_val = self.model_const_attrs.get(buffer_key)
                    if (
                        const_val is not None
                        and tuple(const_val.shape) == tuple(arg.shape)
                        and const_val.dtype == arg.dtype
                    ):
                        t.copy_(const_val)
                    else:
                        t.zero_()
                realized[i] = t

            shared_name_to_idx = {
                name: i for i, name in enumerate(shared_placeholder_names)
            }
            subgraph_specs = []
            for gm in gms:
                sub_placeholder_names = [
                    n.name for n in gm.graph.nodes if n.op == "placeholder"
                ]
                dynamic_names = [
                    name for name in sub_placeholder_names
                    if name not in shared_name_to_idx
                ]
                subgraph_specs.append((gm.forward, sub_placeholder_names, dynamic_names))

            def _bucket_forward_runner(
                shared_args,
                _specs=subgraph_specs,
                _name_to_idx=shared_name_to_idx,
                _use_ac=use_activation_checkpointing,
                _stage_id=stage_id,
                _ubid=ubid,
                _shared_placeholder_names=tuple(shared_placeholder_names),
            ):
                out = None
                for subgraph_idx, (forward_impl, placeholder_names, dynamic_names) in enumerate(_specs):
                    if len(dynamic_names) == 0:
                        dynamic_values = []
                    elif len(dynamic_names) == 1:
                        dyn_arg = out
                        if isinstance(dyn_arg, (tuple, list)) and len(dyn_arg) == 1:
                            dyn_arg = dyn_arg[0]
                        dynamic_values = [dyn_arg]
                    else:
                        if not isinstance(out, (tuple, list)):
                            raise RuntimeError(
                                f"Bucket forward expected {len(dynamic_names)} dynamic inputs "
                                f"but previous subgraph produced {type(out).__name__}"
                            )
                        dynamic_values = list(out)
                        if len(dynamic_values) != len(dynamic_names):
                            raise RuntimeError(
                                f"Bucket forward expected {len(dynamic_names)} dynamic inputs "
                                f"but previous subgraph produced {len(dynamic_values)} values"
                            )
                    dynamic_name_to_value = {
                        name: value for name, value in zip(dynamic_names, dynamic_values)
                    }
                    call_args = []
                    for name in placeholder_names:
                        if name in dynamic_name_to_value:
                            call_args.append(dynamic_name_to_value[name])
                        else:
                            call_args.append(shared_args[_name_to_idx[name]])
                    if _use_ac:
                        out = torch.utils.checkpoint.checkpoint(
                            forward_impl,
                            *call_args,
                            use_reentrant=False,
                        )
                    else:
                        out = forward_impl(*call_args)
                return out

            bucket.forward_fn = _bucket_forward_runner
            bucket.forward_args = realized
            bucket.param_idxs = b_param_idxs
            bucket.activation_checkpoint_subgraph_count = ac_num_subgraphs
            trainable_idxs = [
                i for i in b_param_idxs
                if realized[i] is not None and realized[i].requires_grad
            ]
            bucket.trainable_param_idxs = trainable_idxs

            zero_managed = (
                self.runtime.dp_degree > 1
                and apply_zero
                and bool(trainable_idxs)
                and ubid in self.stages.zero_managed_ubids
            )
            params_sharded = ubid in self.stages.param_sharded_ubids
            grads_sharded = ubid in self.stages.grad_sharded_ubids

            if zero_managed:
                trainable = [realized[i] for i in trainable_idxs]
                flat_params = torch.cat([p.detach().view(-1) for p in trainable]).contiguous()
                flat_params.requires_grad_(False)
                orig_numel = flat_params.numel()
                dp = self.runtime.dp_degree
                shard_size = (orig_numel + dp - 1) // dp
                padded_numel = shard_size * dp
                if padded_numel > orig_numel:
                    padded = flat_params.new_zeros(padded_numel)
                    padded[:orig_numel].copy_(flat_params)
                    flat_params = padded

                offset = 0
                view_specs = []
                for idx, p in zip(trainable_idxs, trainable):
                    numel = p.numel()
                    realized[idx] = realized[idx].detach()
                    realized[idx].data = flat_params[offset:offset + numel].view(p.shape)
                    realized[idx].requires_grad_(True)
                    realized[idx].grad_dtype = self.params.grad_buffer_dtype
                    view_specs.append((realized[idx], offset, numel, tuple(p.shape)))
                    offset += numel

                shard_start = self.runtime.dp_rank * shard_size
                if params_sharded:
                    shard_param = flat_params[shard_start:shard_start + shard_size].detach().clone()
                else:
                    shard_param = flat_params[shard_start:shard_start + shard_size]
                shard_param.requires_grad_(True)
                # Keep ZeRO comm/storage buffers in fp32, but let the optimizer consume
                # grads in the shard-param dtype to match Adam's bf16 foreach path.
                shard_param.grad_dtype = None

                bucket.flat_params = flat_params
                bucket.flat_grads = (
                    None
                    if grads_sharded
                    else torch.zeros(padded_numel, dtype=self.params.grad_buffer_dtype, device=self.runtime.device)
                )
                bucket.shard_param = shard_param
                bucket.shard_optimizer = self.optim_class([shard_param])
                bucket.reduce_scatter_grads = (
                    torch.zeros(shard_size, dtype=self.params.grad_buffer_dtype, device=self.runtime.device)
                    if grads_sharded
                    else None
                )
                bucket.param_shard_info = (shard_start, shard_size, orig_numel)
                bucket.param_view_specs = view_specs
                bucket.full_params_fresh = False

                if params_sharded:
                    storage = flat_params.untyped_storage()
                    if storage.size() != 0:
                        storage.resize_(0)

                optim = None
            else:
                # Keep trainable params as separate tensors for the non-ZeRO path.
                bucket.param_view_specs = []
                bucket.flat_params = None
                bucket.flat_grads = None
                bucket.shard_param = None
                bucket.shard_optimizer = None
                bucket.reduce_scatter_grads = None
                bucket.param_shard_info = None
                bucket.full_params_fresh = False
                trainable_for_optim = [realized[i] for i in trainable_idxs]
                optim = self.optim_class(trainable_for_optim, fused=True) if trainable_for_optim else None
            bucket.optimizer = optim

        # Keep first GraphModule for compatibility with external inspection tools.
        self.stages.graph_modules[stage_id] = first_gm

    def load_training_dag(self, training_dag: Any) -> None:
        """Load per-PP-rank TrainingDAG compute nodes using existing _load_stage logic.

        This is an adapter for the new TrainingDAG representation where each
        COMPUTE/FWD node with ``node_meta['gm']`` is equivalent to one bucket/module.
        The runtime executes TrainingDAG nodes via ``run_dag``; this method
        only populates actor-side bucket/module state.
        """
        if training_dag is None:
            raise ValueError("load_training_dag requires non-None training_dag")
        if not hasattr(training_dag, "nodes"):
            raise TypeError("load_training_dag expected object with 'nodes' field")

        self._derive_dag_bucket_modes(training_dag)
        self.runtime.initialize_streams_for_training_dag(training_dag)

        # Clear any prior module state before loading a new training DAG.
        self.stages.clear_loaded_modules()
        self.dag_executor.expert_a2a_states.clear()

        stage_to_modules: dict[int, list[dict]] = defaultdict(list)

        # Deterministic order by stage/segment then uid.
        compute_nodes = [
            n for n in training_dag.nodes.values()
            if getattr(n, "node_kind", None) == "COMPUTE"
            and getattr(n, "compute_subkind", None) == "FWD"
        ]
        compute_nodes.sort(
            key=lambda n: (
                int(getattr(n, "node_meta", {}).get("stage_id", 10**9)),
                int(getattr(n, "node_meta", {}).get("segment_id", 10**9)),
                str(getattr(n, "uid", "")),
            )
        )

        seen_bucket_keys: set[Any] = set()
        skipped_dupe_fwd_nodes = 0
        for node in compute_nodes:
            meta = getattr(node, "node_meta", {}) or {}
            bucket_key = meta.get("bucket_key", getattr(node, "uid", None))
            if bucket_key in seen_bucket_keys:
                skipped_dupe_fwd_nodes += 1
                continue
            seen_bucket_keys.add(bucket_key)
            gm = meta.get("gm")
            gm_data = meta.get("gm_data")
            stage_id = meta.get("stage_id")
            input_idxs = meta.get("input_idxs")
            param_idxs = meta.get("param_idxs")
            graphargs = meta.get("graphargs")
            input_names = meta.get("input_names", [])
            output_names = meta.get("output_names", [])

            if gm_data is None and gm is None:
                raise ValueError(
                    f"load_training_dag: compute node {getattr(node, 'uid', '<unknown>')} "
                    f"is missing required metadata field 'gm_data' (or fallback 'gm')"
                )
            if stage_id is None or input_idxs is None or param_idxs is None or graphargs is None:
                raise ValueError(
                    f"load_training_dag: compute node {getattr(node, 'uid', '<unknown>')} "
                    f"is missing required metadata fields"
                )

            module_data = {
                "gm_data": gm_data if gm_data is not None else _serialize_graphmodule(gm),
                "graphargs": list(graphargs),
                "input_idxs": list(input_idxs),
                "param_idxs": list(param_idxs),
                "placeholder_names": list(input_names),
                "output_names": list(output_names),
                "shared_placeholder_names": list(input_names),
                "bucket_key": bucket_key,
                "ac_num_subgraphs": 1,
                "ac_requested_subgraphs": 1,
                "apply_zero": bool(meta.get("apply_zero", True)),
                "is_expert": "EP" in getattr(node, "tag", {}),
                "training_dag_uid": getattr(node, "uid", None),
                "triton_constant_args": dict(meta.get("triton_constant_args", {})),
            }
            stage_to_modules[int(stage_id)].append(module_data)

        assert stage_to_modules, (
            "load_training_dag: no FWD compute nodes with gm metadata found"
        )

        for stage_id in sorted(stage_to_modules.keys()):
            self._load_stage(
                stage_id=stage_id,
                modules_data=stage_to_modules[stage_id],
                a2a_boundaries={},
                use_activation_checkpointing=False,
            )

        # Build runtime adjacency on TrainingDAG nodes so run_dag can execute directly.
        for n in training_dag.nodes.values():
            n.data_preds = []
            n.data_succs = []
            n.temporal_preds = []
            n.temporal_succs = []
            if n.node_kind == "SEND_COMM":
                n.peer_pp_rank = n.node_meta.get("peer_pp_rank")
            elif n.node_kind == "RECV_COMM":
                n.peer_pp_rank = n.node_meta.get("peer_pp_rank")
            else:
                n.peer_pp_rank = None
        for e in training_dag.edges:
            if e.src_uid not in training_dag.nodes or e.dst_uid not in training_dag.nodes:
                continue
            src = training_dag.nodes[e.src_uid]
            dst = training_dag.nodes[e.dst_uid]
            if e.dep_kind == "temporal":
                src.temporal_succs.append(dst)
                dst.temporal_preds.append(src)
            else:
                src.data_succs.append(dst)
                dst.data_preds.append(src)

        for n in training_dag.nodes.values():
            n.task_type = _training_dag_task_type(n)
            mb = n.tag.get("MB", 0)
            st = n.tag.get("PP", 0)
            n.batches = [type("RuntimeBatch", (), {"stage_id": st, "mb_idx": mb})()]

        self.dag = training_dag
        self.sorted_dag_nodes = [
            training_dag.nodes[uid] for uid in _serial_topological_order(training_dag)
        ]
        self.prepared_dag_execution = self.dag_executor.prepare(
            training_dag,
            self.sorted_dag_nodes,
            stream_override=self.runtime.execution_stream_for_task,
        )
        # Normal profiling uses the ordinary physical-stream policy exactly;
        # only interference-free profiling needs a remapped stream table.
        self.profile_prepared_dag_executions = {
            "normal": self.prepared_dag_execution,
        }
        for profile_mode in sorted(PROFILE_MODES - {"normal"}):
            self.profile_prepared_dag_executions[profile_mode] = (
                self.dag_executor.remap_streams(
                    self.prepared_dag_execution,
                    stream_override=(
                        lambda node, logical_stream, mode=profile_mode: (
                            self._profile_stream_override(
                                node,
                                logical_stream,
                                profile_mode=mode,
                            )
                        )
                    ),
                )
            )

    def _execute_dag(
        self,
        loss_fn=None,
        *,
        prepared_execution: PreparedDagExecution | None = None,
        timing_recorder=None,
        serialize_nodes: bool = False,
        iteration_label: str = "iter",
    ):
        if prepared_execution is None:
            prepared_execution = self.prepared_dag_execution
        if prepared_execution is None:
            raise RuntimeError(
                "load_training_dag() must be called before DAG execution"
            )
        # Mark the entire iteration boundary for the NVTX timeline.
        iter_idx = getattr(self, "_iter_counter", 0)
        self._iter_counter = iter_idx + 1
        self._nvtx_push(
            f"{iteration_label}_{iter_idx}_rank_{self.runtime.global_rank}"
        )
        try:
            return self.dag_executor.run(
                prepared_execution,
                self.inputs,
                self.labels,
                self.loss,
                loss_fn=loss_fn,
                timing_recorder=timing_recorder,
                serialize_nodes=serialize_nodes,
            )
        finally:
            self._nvtx_pop()

    def run_dag(self, loss_fn=None, return_losses: bool = True):
        try:
            result = self._execute_dag(loss_fn=loss_fn)
        finally:
            if getattr(self.runtime, "pytorch_profiler_enabled", False):
                self.runtime.torch_profiler.step()
        # Timing/profiling callers do not consume losses. Avoid serializing CUDA
        # tensors back to CPU-only DP driver tasks in that common path.
        return result if return_losses else None

    def reset_profile_samples(
        self, *, profile_mode: str = "interference_free"
    ) -> None:
        """Discard samples from an earlier session for one profiling policy."""

        mode = normalize_profile_mode(profile_mode)
        self._profile_node_samples[mode] = defaultdict(list)
        self._profile_node_start_offsets[mode] = defaultdict(list)
        self._profile_prologue_samples[mode] = []
        self._profile_timeline_origins_us[mode] = []
        self._profile_timeline_origin_uncertainties_us[mode] = []

    def calibrate_profile_clock(self, clock_actor: Any, samples: int = 9) -> dict:
        """Calibrate this actor's monotonic clock to one Ray coordinator.

        This is the standard midpoint/round-trip estimator used by NTP.  The
        minimum-RTT exchange minimizes queueing asymmetry; half its RTT is
        retained as a conservative alignment uncertainty.
        """

        if int(samples) <= 0:
            raise ValueError("clock calibration requires at least one sample")
        candidates: list[tuple[int, int]] = []
        for _ in range(int(samples)):
            local_before_ns = time.monotonic_ns()
            coordinator_ns = int(ray.get(clock_actor.monotonic_time_ns.remote()))
            local_after_ns = time.monotonic_ns()
            round_trip_ns = local_after_ns - local_before_ns
            local_midpoint_ns = (local_before_ns + local_after_ns) // 2
            candidates.append(
                (round_trip_ns, coordinator_ns - local_midpoint_ns)
            )
        best_round_trip_ns, best_offset_ns = min(candidates)
        self._profile_coordinator_clock_offset_ns = best_offset_ns
        self._profile_clock_sync_uncertainty_ns = (
            best_round_trip_ns + 1
        ) // 2
        return {
            "method": "ray_ntp_midpoint_min_rtt",
            "samples": int(samples),
            "coordinator_clock_offset_ns": int(best_offset_ns),
            "round_trip_ns": int(best_round_trip_ns),
            "uncertainty_ns": int(self._profile_clock_sync_uncertainty_ns),
        }

    def _profile_stream_override(
        self,
        node: Any,
        logical_stream: torch.cuda.Stream,
        *,
        profile_mode: str = "interference_free",
    ) -> torch.cuda.Stream:
        return self.runtime.profile_stream_for_task(
            node,
            logical_stream,
            profile_mode=profile_mode,
        )

    def profile_run_dag(
        self,
        loss_fn=None,
        *,
        record: bool = True,
        synchronize_before: bool = True,
        synchronize: bool = False,
        profile_mode: str = "interference_free",
        advance_profiler: bool = True,
        capture_allocator_stats: bool = True,
    ) -> None:
        """Run one timed iteration using interference-free or normal streams.

        Both modes preserve each node's normal logical stream (including the
        directional P2P streams). ``interference_free`` additionally chains
        nodes with CUDA events outside their CUPTI ranges so their work is
        serialized without changing its stream affinity.
        """
        mode = normalize_profile_mode(profile_mode)
        if synchronize_before:
            torch.cuda.synchronize(self.runtime.device)
        collect_allocator_stats = (
            self.runtime.pytorch_profiler_enabled
            and bool(capture_allocator_stats)
        )
        allocator_before = (
            self._cuda_allocator_stats_snapshot()
            if collect_allocator_stats
            else None
        )
        allocator_streams_before = (
            self._cuda_allocator_stream_snapshot()
            if collect_allocator_stats
            else None
        )
        recorder = (
            _CudaEventTimingRecorder(
                coordinator_clock_offset_ns=getattr(
                    self, "_profile_coordinator_clock_offset_ns", 0
                ),
                clock_sync_uncertainty_ns=getattr(
                    self, "_profile_clock_sync_uncertainty_ns", 0
                ),
            )
            if record
            else None
        )
        try:
            self._execute_dag(
                loss_fn=loss_fn,
                prepared_execution=self.profile_prepared_dag_executions[mode],
                timing_recorder=recorder,
                serialize_nodes=(mode == INTERFERENCE_FREE_PROFILE_MODE),
                iteration_label=f"profile_{mode}_iter",
            )
        finally:
            if self.runtime.pytorch_profiler_enabled:
                if collect_allocator_stats:
                    allocator_after = self._cuda_allocator_stats_snapshot()
                    allocator_streams_after = self._cuda_allocator_stream_snapshot()
                    before = allocator_before or {}
                    self._cupti_allocator_iteration_stats.append({
                        "before": before,
                        "after": allocator_after,
                        "delta": {
                            key: int(value) - int(before.get(key, 0))
                            for key, value in allocator_after.items()
                        },
                        "stream_pools_before": allocator_streams_before or {},
                        "stream_pools_after": allocator_streams_after,
                    })
                if advance_profiler:
                    self.runtime.torch_profiler.step()
        if recorder is None:
            if synchronize:
                torch.cuda.synchronize(self.runtime.device)
            return
        missing = set(self.dag.nodes) - set(recorder.node_samples_us)
        if missing:
            raise RuntimeError(
                f"profile iteration did not record nodes: {sorted(missing)}"
            )
        node_samples = self._profile_node_samples.setdefault(mode, defaultdict(list))
        for uid, sample_us in recorder.node_samples_us.items():
            node_samples[uid].append(sample_us)
        node_start_offsets = self._profile_node_start_offsets.setdefault(
            mode, defaultdict(list)
        )
        for uid, offset_us in recorder.node_start_offsets_us.items():
            node_start_offsets[uid].append(offset_us)
        timeline_origins = getattr(
            self, "_profile_timeline_origins_us", None
        )
        if timeline_origins is None:
            timeline_origins = {}
            self._profile_timeline_origins_us = timeline_origins
        timeline_uncertainties = getattr(
            self, "_profile_timeline_origin_uncertainties_us", None
        )
        if timeline_uncertainties is None:
            timeline_uncertainties = {}
            self._profile_timeline_origin_uncertainties_us = timeline_uncertainties
        if getattr(recorder, "timeline_origin_us", None) is not None:
            timeline_origins.setdefault(mode, []).append(
                recorder.timeline_origin_us
            )
            timeline_uncertainties.setdefault(mode, []).append(
                float(
                    getattr(recorder, "timeline_origin_uncertainty_us", 0.0)
                    or 0.0
                )
            )
        if recorder.prologue_sample_us is not None:
            self._profile_prologue_samples.setdefault(mode, []).append(
                recorder.prologue_sample_us
            )

    def synchronize_device(self) -> None:
        """Wait for all work on this actor's CUDA device."""

        torch.cuda.synchronize(self.runtime.device)

    @staticmethod
    def _clone_profile_state_value(value: Any) -> Any:
        """Clone optimizer state into host memory for paired restoration."""

        if torch.is_tensor(value):
            # Snapshot storage is profiler control state, not part of the
            # represented training workload.  Keeping it on the device can
            # consume several GiB and make otherwise valid schedules OOM.
            return value.detach().to(device="cpu", copy=True)
        if isinstance(value, dict):
            return {
                key: PiperActor._clone_profile_state_value(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [PiperActor._clone_profile_state_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(
                PiperActor._clone_profile_state_value(item) for item in value
            )
        return value

    @staticmethod
    def _restore_profile_state_value(current: Any, saved: Any) -> Any:
        """Restore a cloned state value, preserving existing tensor storage."""

        if torch.is_tensor(saved):
            if (
                torch.is_tensor(current)
                and current.shape == saved.shape
                and current.dtype == saved.dtype
            ):
                # Preserve the optimizer tensor's device and storage.  copy_
                # handles the host-to-device transfer when the snapshot was
                # offloaded to CPU.
                current.copy_(saved)
                return current
            return saved.detach().clone()
        if isinstance(saved, dict):
            target = current if isinstance(current, dict) else {}
            for key in tuple(target):
                if key not in saved:
                    del target[key]
            for key, item in saved.items():
                target[key] = PiperActor._restore_profile_state_value(
                    target.get(key), item
                )
            return target
        if isinstance(saved, list):
            current_items = current if isinstance(current, list) else []
            return [
                PiperActor._restore_profile_state_value(
                    current_items[index] if index < len(current_items) else None,
                    item,
                )
                for index, item in enumerate(saved)
            ]
        if isinstance(saved, tuple):
            current_items = current if isinstance(current, tuple) else ()
            return tuple(
                PiperActor._restore_profile_state_value(
                    current_items[index] if index < len(current_items) else None,
                    item,
                )
                for index, item in enumerate(saved)
            )
        return saved

    def capture_profile_training_state(self) -> dict[str, int]:
        """Snapshot mutable training and RNG state for paired policy profiles.

        Normal and interference-free profiles must start from identical model
        and optimizer state.  This matters especially for MoE models: allowing
        the first policy to update router weights before profiling the second
        policy can change expert token counts and therefore the GPU kernels
        selected by grouped GEMM.
        """

        self.params.drain_pending_frees()
        torch.cuda.synchronize(self.runtime.device)

        parameters: list[tuple[torch.Tensor, torch.Tensor]] = []
        seen_parameters: set[int] = set()
        optimizers: list[tuple[Any, dict[Any, Any]]] = []
        seen_optimizers: set[int] = set()
        for bucket in self.stages.buckets.values():
            candidate_parameters = (
                [bucket.shard_param]
                if bucket.shard_param is not None
                else bucket.trainable_params()
            )
            for parameter in candidate_parameters:
                if parameter is None or id(parameter) in seen_parameters:
                    continue
                seen_parameters.add(id(parameter))
                parameters.append(
                    (
                        parameter,
                        parameter.detach().to(device="cpu", copy=True),
                    )
                )

            for optimizer in (bucket.shard_optimizer, bucket.optimizer):
                if optimizer is None or id(optimizer) in seen_optimizers:
                    continue
                seen_optimizers.add(id(optimizer))
                optimizers.append(
                    (
                        optimizer,
                        {
                            parameter: self._clone_profile_state_value(state)
                            for parameter, state in optimizer.state.items()
                        },
                    )
                )

        self._profile_training_state = {
            "parameters": parameters,
            "optimizers": optimizers,
            "cpu_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.runtime.device).clone(),
        }
        return {
            "parameters": len(parameters),
            "optimizers": len(optimizers),
        }

    def restore_profile_training_state(self) -> None:
        """Restore the state captured by :meth:`capture_profile_training_state`."""

        saved = getattr(self, "_profile_training_state", None)
        if saved is None:
            raise RuntimeError("profile training state has not been captured")
        self.params.drain_pending_frees()
        torch.cuda.synchronize(self.runtime.device)
        with torch.no_grad():
            for parameter, value in saved["parameters"]:
                parameter.copy_(value)
            for optimizer, optimizer_state in saved["optimizers"]:
                for parameter in tuple(optimizer.state):
                    if parameter not in optimizer_state:
                        del optimizer.state[parameter]
                for parameter, state in optimizer_state.items():
                    optimizer.state[parameter] = self._restore_profile_state_value(
                        optimizer.state.get(parameter), state
                    )
        torch.set_rng_state(saved["cpu_rng_state"])
        torch.cuda.set_rng_state(saved["cuda_rng_state"], self.runtime.device)
        self.clear_profile_transient_state()
        torch.cuda.synchronize(self.runtime.device)

    def clear_profile_transient_state(self) -> None:
        """Clear gradients and per-step buffers after restoring training state."""

        self.params.clear_param_grads()
        self.buffers.reset()
        self.events.reset()
        for bucket in self.stages.buckets.values():
            if bucket.flat_grads is not None:
                bucket.flat_grads.zero_()
            if bucket.reduce_scatter_grads is not None:
                bucket.reduce_scatter_grads.zero_()
            if bucket.shard_param is not None:
                bucket.shard_param.grad = None

    def p2p_receive_specs(self) -> dict[str, list[tuple[tuple[int, ...], str]]]:
        """Return the receiver-owned payload metadata keyed by P2P transaction."""
        if self.dag is None:
            raise RuntimeError("load_training_dag() must precede P2P profiling")
        specs: dict[str, list[tuple[tuple[int, ...], str]]] = {}
        for node in self.dag.nodes.values():
            if (
                getattr(node, "node_kind", None) != "RECV_COMM"
                or node.tag.get("PASS") != "F"
            ):
                continue
            comm_id = str(node.node_meta["comm_id"])
            bucket = self.stages.bucket(node.node_meta["bucket_key"])
            specs[comm_id] = [
                (tuple(shape), str(dtype))
                for shape, dtype, _ in bucket.forward_input_meta
            ]
        return specs

    def profile_p2p_transfer_baselines(
        self,
        specs_by_comm_id: dict[str, list[tuple[tuple[int, ...], str]]],
        *,
        repeats: int = 5,
    ) -> dict[str, float]:
        """Measure matched P2P transfers with both endpoints already posted.

        Every PP rank takes the same barriers, while only the two endpoints of
        a transaction issue its directional NCCL operation.  This removes DAG
        predecessor and pipeline-rendezvous delay from the timed transfer.
        """
        if self.dag is None:
            raise RuntimeError("load_training_dag() must precede P2P profiling")
        if repeats <= 0:
            raise ValueError("P2P profiling repeats must be positive")
        local_nodes = {
            str(node.node_meta["comm_id"]): node
            for node in self.dag.nodes.values()
            if getattr(node, "node_kind", None) in {"SEND_COMM", "RECV_COMM"}
        }
        results: dict[str, float] = {}
        # pp_lo_hi has every PP rank in this DP lane; barriers must be issued
        # in identical order even when a rank is not an endpoint.
        for comm_id in sorted(specs_by_comm_id):
            node = local_nodes.get(comm_id)
            payload_specs = specs_by_comm_id[comm_id]
            dist.barrier(group=self.runtime.pp_lo_hi)
            samples: list[float] = []
            if node is not None:
                stream = self.runtime.directional_p2p_stream_for_task(node)
                tensors = [
                    torch.empty(shape, dtype=getattr(torch, dtype.split(".")[-1]), device=self.runtime.device)
                    for shape, dtype in payload_specs
                ]
                peer = int(node.peer_pp_rank)
            for _ in range(int(repeats)):
                # Every PP rank must enter every barrier even though only the
                # two endpoints of this transaction issue P2P work.
                torch.cuda.synchronize(self.runtime.device)
                dist.barrier(group=self.runtime.pp_lo_hi)
                if node is not None:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    if node.node_kind == "SEND_COMM":
                        self.communication.send(tensors, peer, stream)
                    else:
                        # Allocate the exact receive buffers here instead of
                        # reusing payload tensors, matching normal recv setup.
                        recv = [torch.empty_like(tensor) for tensor in tensors]
                        with torch.cuda.stream(stream):
                            source = self.runtime.pipeline_peer_global_rank(peer)
                            group = self.runtime.pp_hi_lo if source > self.runtime.global_rank else self.runtime.pp_lo_hi
                            for tensor in recv:
                                dist.recv(tensor, src=source, group=group)
                    end.record(stream)
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end)) * 1000.0)
            if samples:
                results[comm_id] = float(statistics.median(samples))
            dist.barrier(group=self.runtime.pp_lo_hi)
        return results

    def clear_profile_allocator_cache(self) -> None:
        """Release unused cached blocks before a recorded profile series."""

        torch.cuda.synchronize(self.runtime.device)
        gc.collect()
        torch.cuda.empty_cache()

    def get_profile_result(
        self,
        *,
        estimator: str = "median",
        profile_mode: str = "interference_free",
    ) -> ActorProfileResult:
        """Aggregate samples into the detached, rank-keyed profile interface."""
        mode = normalize_profile_mode(profile_mode)
        if self.dag is None:
            raise RuntimeError("load_training_dag() must precede profiling")
        mode_node_samples = self._profile_node_samples.get(mode, {})
        mode_node_start_offsets = self._profile_node_start_offsets.get(mode, {})
        node_profiles: dict[str, NodeProfile] = {}
        for uid, node in self.dag.nodes.items():
            if getattr(node, "node_kind", None) == "ORDER_DUMMY":
                node_profiles[uid] = NodeProfile.skipped(
                    "order_dummy",
                    isolation_scope=(
                        DEFAULT_ISOLATION_SCOPE
                        if mode == "interference_free"
                        else DEFAULT_NORMAL_SCOPE
                    ),
                )
                continue
            samples = mode_node_samples.get(uid, [])
            if not samples:
                raise RuntimeError(
                    f"no {mode} profile samples collected for node {uid}"
                )
            if getattr(node, "node_kind", None) in {"SEND_COMM", "RECV_COMM"}:
                isolation_scope = "normal_directional_p2p"
            elif mode == "interference_free":
                isolation_scope = DEFAULT_ISOLATION_SCOPE
            else:
                isolation_scope = DEFAULT_NORMAL_SCOPE
            node_profiles[uid] = NodeProfile.from_samples(
                samples,
                estimator=estimator,
                isolation_scope=isolation_scope,
                start_offset_samples_us=mode_node_start_offsets.get(uid, ()),
            )

        prologue_samples = self._profile_prologue_samples.get(mode, [])
        prologue_profile = (
            NodeProfile.from_samples(
                prologue_samples,
                estimator=estimator,
                isolation_scope=(
                    DEFAULT_ISOLATION_SCOPE
                    if mode == "interference_free"
                    else DEFAULT_NORMAL_SCOPE
                ),
            )
            if prologue_samples
            else None
        )
        props = torch.cuda.get_device_properties(self.runtime.device)
        return ActorProfileResult(
            profile_mode=mode,
            graph_fingerprint=finalized_dag_fingerprint(self.dag),
            global_rank=self.runtime.global_rank,
            dp_rank=self.runtime.dp_rank,
            pp_rank=self.runtime.pp_rank,
            device_metadata={
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": int(props.total_memory),
                "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda),
            },
            node_profiles=node_profiles,
            prologue_profile=prologue_profile,
            timeline_origin_samples_us=tuple(
                getattr(self, "_profile_timeline_origins_us", {}).get(mode, ())
            ),
            timeline_origin_uncertainty_samples_us=tuple(
                getattr(
                    self,
                    "_profile_timeline_origin_uncertainties_us",
                    {},
                ).get(mode, ())
            ),
        )
