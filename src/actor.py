import ray
import torch
import os
import re
import hashlib
import json
import tempfile
import time
from typing import Any, Sequence
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
from .dag import restore_triton_side_table
from .model_state import (
    DEFAULT_MODEL_SEED,
    dynamo_buffer_name_to_state_dict_key,
    dynamo_parameter_name_to_state_dict_key,
    initialize_parameter_,
    initialize_route_parameter_,
)
from .executors import (
    CommunicationExecutor,
    ComputeExecutor,
    DagExecutor,
    P2P_POST_PROFILER_MARKER,
    PreparedDagExecution,
)
from .ordering import _dispatch_order
from .profiling import (
    ActorProfileResult,
    DEFAULT_NORMAL_SCOPE,
    DEFAULT_ISOLATION_SCOPE,
    ISOLATED_PROFILE_MODE,
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

_KINETO_ITERATION_ANCHOR_PREFIX = "piper_profile_timeline_anchor:"


def _kineto_node_attribution_audit(
    profiler,
    *,
    trace_path: str | None = None,
    timeline_anchor_origins: dict[str, tuple[float, float]] | None = None,
    expected_timeline_anchors: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[float, ...]]]:
    """Audit CUDA launch/activity correlation for every Piper node range."""

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

    def normalized_launch_operation(name: str) -> str | None:
        lowered = name.lower()
        if "memcpy" in lowered:
            return "memcpy"
        if "memset" in lowered:
            return "memset"
        if "launch" in lowered:
            return "kernel"
        return None
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

    supplied_timeline_anchors = dict(timeline_anchor_origins or {})
    captured_timeline_anchors: list[
        tuple[float, float, str, Any, Any]
    ] = []
    seen_timeline_labels: set[str] = set()
    for event in events:
        name = str(event.get("name", ""))
        if (
            event.get("ph") != "X"
            or event.get("cat") != "user_annotation"
            or not name.startswith(_KINETO_ITERATION_ANCHOR_PREFIX)
        ):
            continue
        start = event.get("ts")
        duration = event.get("dur")
        if not isinstance(start, (int, float)) or not isinstance(
            duration, (int, float)
        ):
            raise RuntimeError(
                f"Kineto timeline anchor {name!r} has no numeric interval"
            )
        if name in seen_timeline_labels:
            raise RuntimeError(f"duplicate Kineto timeline anchor {name!r}")
        if name not in supplied_timeline_anchors:
            raise RuntimeError(
                f"Kineto captured unknown timeline anchor {name!r}"
            )
        seen_timeline_labels.add(name)
        captured_timeline_anchors.append(
            (
                float(start),
                float(start) + float(duration),
                name,
                event.get("pid"),
                event.get("tid"),
            )
        )
    captured_timeline_anchors.sort(key=lambda item: (item[0], item[2]))
    if supplied_timeline_anchors:
        if not captured_timeline_anchors:
            raise RuntimeError("Kineto capture contains no profiling timeline anchors")
        if (
            expected_timeline_anchors is not None
            and len(captured_timeline_anchors) != int(expected_timeline_anchors)
        ):
            raise RuntimeError(
                "Kineto timeline anchor count does not match active iterations: "
                f"captured={len(captured_timeline_anchors)}, "
                f"expected={int(expected_timeline_anchors)}"
            )
    timeline_index_by_label = {
        label: index
        for index, (_start, _end, label, _pid, _tid) in enumerate(
            captured_timeline_anchors
        )
    }
    timeline_anchors_by_thread: dict[
        tuple[Any, Any], list[tuple[float, float, str]]
    ] = defaultdict(list)
    for start, end, label, pid, tid in captured_timeline_anchors:
        timeline_anchors_by_thread[(pid, tid)].append((start, end, label))

    ranges_by_thread: dict[
        tuple[Any, Any], list[tuple[float, float, str, int]]
    ] = {}
    range_records: dict[int, tuple[float, float, str]] = {}
    timeline_label_by_range: dict[int, str] = {}
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
        thread = (event.get("pid"), event.get("tid"))
        enclosing_anchors = [
            label
            for anchor_start, anchor_end, label in timeline_anchors_by_thread.get(
                thread, ()
            )
            if anchor_start <= record[0] and record[1] <= anchor_end
        ]
        if supplied_timeline_anchors and len(enclosing_anchors) != 1:
            raise RuntimeError(
                f"Kineto node range {name!r} must belong to exactly one timeline "
                f"anchor, found {enclosing_anchors!r}"
            )
        if enclosing_anchors:
            timeline_label_by_range[range_key] = enclosing_anchors[0]
        ranges_by_thread.setdefault(thread, []).append(
            (*record, range_key)
        )
    for ranges in ranges_by_thread.values():
        ranges.sort()

    p2p_post_markers_by_range: dict[int, list[float]] = defaultdict(list)
    for event in events:
        if (
            event.get("ph") != "X"
            or event.get("cat") != "user_annotation"
            or str(event.get("name", "")) != P2P_POST_PROFILER_MARKER
        ):
            continue
        start = event.get("ts")
        duration = event.get("dur")
        if not isinstance(start, (int, float)) or not isinstance(
            duration, (int, float)
        ):
            raise RuntimeError("Kineto P2P post marker has no numeric interval")
        marker_start = float(start)
        marker_end = marker_start + float(duration)
        thread = (event.get("pid"), event.get("tid"))
        enclosing_ranges = [
            range_key
            for range_start, range_end, _label, range_key in ranges_by_thread.get(
                thread, ()
            )
            if range_start <= marker_start and marker_end <= range_end
        ]
        if len(enclosing_ranges) != 1:
            raise RuntimeError(
                "Kineto P2P post marker must belong to exactly one node range, "
                f"found {enclosing_ranges!r}"
            )
        p2p_post_markers_by_range[enclosing_ranges[0]].append(marker_start)

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

    launch_records_by_correlation: dict[
        Any, list[tuple[float, float, str, str, int | None]]
    ] = defaultdict(list)
    launch_records_by_range: dict[
        int | None, list[tuple[float, float, str, str, Any | None]]
    ] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in {"cuda_runtime", "cuda_driver"}:
            continue
        event_name = str(event.get("name", ""))
        if normalized_launch_operation(event_name) is None:
            continue
        args = event.get("args") or {}
        range_key = external_id_to_range.get(args.get("External id"))
        correlation = args.get("correlation")
        start = event.get("ts")
        duration = event.get("dur")
        if isinstance(start, (int, float)) and isinstance(duration, (int, float)):
            record = (
                float(start),
                float(start) + float(duration),
                event_name,
                str(event.get("cat", "")),
                range_key,
            )
            launch_records_by_range[range_key].append((*record[:4], correlation))
            if correlation is not None:
                launch_records_by_correlation[correlation].append(record)

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
    audit: dict[str, dict[str, Any]] = {}
    gpu_activity_records_by_range: dict[
        int | None, list[tuple[float, float, int | None, str, str, Any | None]]
    ] = defaultdict(list)
    for record in gpu_activity_records:
        gpu_activity_records_by_range[record[2]].append(record)
    for label, range_count in ranges_by_label.items():
        range_activity_signatures = []
        label_launch_count = 0
        label_matched_launch_count = 0
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

            raw_launches = launch_records_by_range.get(range_key, ())
            launches_by_correlation: dict[
                Any, list[tuple[float, float, str, str, Any | None]]
            ] = defaultdict(list)
            uncorrelated_launches = []
            for launch in raw_launches:
                correlation = launch[4]
                if correlation is None:
                    uncorrelated_launches.append(launch)
                else:
                    launches_by_correlation[correlation].append(launch)

            activities_by_correlation: dict[
                Any, list[tuple[float, float, int | None, str, str, Any | None]]
            ] = defaultdict(list)
            unmapped_activities = []
            for activity in target_records:
                correlation = activity[5]
                if correlation is None or correlation not in launches_by_correlation:
                    unmapped_activities.append({
                        "category": activity[3],
                        "name": activity[4],
                        "reason": (
                            "missing_correlation"
                            if correlation is None
                            else "no_occurrence_local_launch"
                        ),
                    })
                else:
                    activities_by_correlation[correlation].append(activity)

            canonical_launches = []
            for correlation, candidates in launches_by_correlation.items():
                runtime_candidates = [
                    launch for launch in candidates
                    if launch[3] == "cuda_runtime"
                ]
                selected_candidates = runtime_candidates or candidates
                selected = min(
                    selected_candidates,
                    key=lambda launch: (launch[0], launch[1], launch[2]),
                )
                operations = {
                    normalized_launch_operation(launch[2])
                    for launch in candidates
                }
                ambiguous_reasons = []
                if len(selected_candidates) != 1:
                    ambiguous_reasons.append("duplicate_canonical_callbacks")
                if len(operations) != 1 or None in operations:
                    ambiguous_reasons.append("inconsistent_operation_class")
                activities_for_launch = sorted(
                    activities_by_correlation.get(correlation, ()),
                    key=lambda activity: (
                        activity[0], activity[1], activity[3], activity[4]
                    ),
                )
                canonical_launches.append({
                    "_start": selected[0],
                    "operation": normalized_launch_operation(selected[2]),
                    "api_name": selected[2],
                    "mapping_status": (
                        "ambiguous" if ambiguous_reasons else "unambiguous"
                    ),
                    "ambiguity_reasons": ambiguous_reasons,
                    "activities": [
                        {"category": activity[3], "name": activity[4]}
                        for activity in activities_for_launch
                    ],
                })
            for launch in uncorrelated_launches:
                canonical_launches.append({
                    "_start": launch[0],
                    "operation": normalized_launch_operation(launch[2]),
                    "api_name": launch[2],
                    "mapping_status": "ambiguous",
                    "ambiguity_reasons": ["missing_correlation"],
                    "activities": [],
                })
            canonical_launches.sort(
                key=lambda launch: (
                    launch["_start"], launch["operation"] or "", launch["api_name"]
                )
            )
            for launch_index, launch in enumerate(canonical_launches):
                launch.pop("_start")
                launch["launch_index"] = launch_index
            cpu_launch_sequence = [
                str(launch["operation"]) for launch in canonical_launches
            ]
            launch_sequence_encoded = json.dumps(
                cpu_launch_sequence, separators=(",", ":")
            ).encode("utf-8")
            launch_mapping_unambiguous = (
                not unmapped_activities
                and all(
                    launch["mapping_status"] == "unambiguous"
                    for launch in canonical_launches
                )
            )
            matched_launch_count = sum(
                bool(launch["activities"]) for launch in canonical_launches
            )
            label_launch_count += len(canonical_launches)
            label_matched_launch_count += matched_launch_count

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
            timeline_label = timeline_label_by_range.get(range_key)
            timeline_occurrence = (
                None
                if timeline_label is None
                else timeline_index_by_label[timeline_label]
            )
            if timeline_label is None:
                activity_start_offset_us = None
                activity_end_offset_us = None
                activity_timestamp_source = None
                p2p_post_offset_us = None
            else:
                anchor_start = captured_timeline_anchors[
                    timeline_occurrence
                ][0]
                activity_start = (
                    range_records[range_key][0]
                    if not target_union
                    else target_union[0][0]
                )
                activity_end = (
                    activity_start if not target_union else target_union[-1][1]
                )
                activity_start_offset_us = activity_start - anchor_start
                activity_end_offset_us = activity_end - anchor_start
                if activity_start_offset_us < 0.0:
                    raise RuntimeError(
                        f"Kineto activity for node range {label!r} starts before "
                        f"its iteration anchor by {-activity_start_offset_us:.3f} us"
                    )
                activity_timestamp_source = (
                    "cpu_range_zero_activity"
                    if not target_union
                    else "correlated_gpu_activity"
                )
                p2p_post_markers = p2p_post_markers_by_range.get(range_key, ())
                if len(p2p_post_markers) > 1:
                    raise RuntimeError(
                        f"Kineto node range {label!r} contains multiple P2P "
                        f"first-call markers: {p2p_post_markers!r}"
                    )
                p2p_post_offset_us = (
                    None
                    if not p2p_post_markers
                    else float(p2p_post_markers[0]) - anchor_start
                )
                if p2p_post_offset_us is not None and p2p_post_offset_us < 0.0:
                    raise RuntimeError(
                        f"Kineto P2P post for node range {label!r} precedes its "
                        f"iteration anchor by {-p2p_post_offset_us:.3f} us"
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
                "timeline_occurrence": timeline_occurrence,
                "activity_start_offset_us": activity_start_offset_us,
                "activity_end_offset_us": activity_end_offset_us,
                "activity_timestamp_source": activity_timestamp_source,
                "p2p_post_offset_us": p2p_post_offset_us,
                "cpu_launch_count": len(canonical_launches),
                "matched_cpu_launch_count": matched_launch_count,
                "cpu_launch_sequence_sha256": hashlib.sha256(
                    launch_sequence_encoded
                ).hexdigest(),
                "launch_mapping_unambiguous": launch_mapping_unambiguous,
                # Consumed centrally for per-launch normal/isolated validation, then
                # removed from the durable profile artifact.
                "cpu_launch_sequence": cpu_launch_sequence,
                "cpu_launches": canonical_launches,
                "unmapped_activities": unmapped_activities,
                # Consumed centrally for the exact isolated/normal activity audit,
                # then removed from the durable profile artifact.
                "activities": entries,
                "activity_sequence": activity_sequence,
            }
            range_activity_signatures.append(compact_signature)
        audit[label] = {
            "range_count": int(range_count),
            "launch_count": label_launch_count,
            "matched_launch_count": label_matched_launch_count,
            "unmatched_launch_count": (
                label_launch_count - label_matched_launch_count
            ),
            "launch_coverage_pct": (
                100.0
                if not label_launch_count
                else 100.0 * label_matched_launch_count / label_launch_count
            ),
            "gpu_activity_count": int(activity_count_by_label.get(label, 0)),
            "gpu_activity_time_total_us": float(activity_time_by_label.get(label, 0.0)),
            "gpu_activity_time_mean_us": (
                float(activity_time_by_label.get(label, 0.0)) / range_count
            ),
            "gpu_activity_types": dict(activity_types_by_label.get(label, {})),
            "range_activity_signatures": range_activity_signatures,
        }
        if captured_timeline_anchors:
            occurrences = [
                item.get("timeline_occurrence")
                for item in range_activity_signatures
            ]
            expected_occurrences = list(range(len(captured_timeline_anchors)))
            if occurrences != expected_occurrences:
                raise RuntimeError(
                    f"Kineto node range {label!r} does not cover every captured "
                    f"timeline occurrence: got={occurrences!r}, "
                    f"expected={expected_occurrences!r}"
                )

    timeline_origins_us = tuple(
        float(supplied_timeline_anchors[label][0])
        for _start, _end, label, _pid, _tid in captured_timeline_anchors
    )
    timeline_uncertainties_us = tuple(
        float(supplied_timeline_anchors[label][1])
        for _start, _end, label, _pid, _tid in captured_timeline_anchors
    )
    return audit, {
        "timeline_origin_samples_us": timeline_origins_us,
        "timeline_origin_uncertainty_samples_us": timeline_uncertainties_us,
    }


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
    Communication rendezvous waiting is removed later, using aligned endpoint
    starts from the same profiling policy.
    """
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


def _disable_functorch_donated_buffers() -> None:
    import importlib

    config = importlib.import_module("torch._functorch.config")
    config.donated_buffer = False


def _create_actors(
    actor_groups: tuple[tuple[int, ...], ...],
    optim_class,
    profile=False,
    no_nvtx: bool = False,
    pg=None,
    temp_dir: str = None,
    use_inductor: bool = False,
    model_seed: int = DEFAULT_MODEL_SEED,
):
    from .state import piper_metadata

    actor_groups = tuple(
        tuple(sorted(int(global_rank) for global_rank in group))
        for group in actor_groups
    )
    if not actor_groups or any(not group for group in actor_groups):
        raise ValueError(f"actor_groups must contain non-empty groups, got {actor_groups}")
    global_ranks = [rank for group in actor_groups for rank in group]
    if len(global_ranks) != len(set(global_ranks)):
        raise ValueError(
            "each global rank must belong to exactly one ActorGroup, got "
            f"{actor_groups}"
        )
    world_size = len(global_ranks)
    if sorted(global_ranks) != list(range(world_size)):
        raise ValueError(
            f"ActorGroups must cover global ranks [0, {world_size}), got "
            f"{sorted(global_ranks)}"
        )
    pp_degree = len(actor_groups)
    actor_specs = [
        (group_id, global_rank, local_rank, group, actor_groups)
        for group_id, group in enumerate(actor_groups)
        for local_rank, global_rank in enumerate(group)
    ]

    piper_metadata.actors = {}
    for (
        pp_rank,
        global_rank,
        local_rank,
        actor_group_devices,
        all_actor_groups,
    ) in actor_specs:
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
                "KINETO_LOG_LEVEL",
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
            placement_plan.actor_global_rank_bundle_index(global_rank)
            if placement_plan is not None
            else global_rank
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
            global_rank=global_rank,
            pp_degree=pp_degree,
            no_nvtx=no_nvtx,
            use_inductor=use_inductor,
            model_seed=model_seed,
            actor_group_id=pp_rank,
            actor_group_local_rank=local_rank,
            actor_group_devices=actor_group_devices,
            actor_groups=all_actor_groups,
        )
        piper_metadata.actors[global_rank] = actor


@ray.remote
class PiperActor:
    def __init__(
        self,
        pp_rank,
        optim_class,
        world_size,
        global_rank: int,
        pp_degree=1,
        no_nvtx: bool = False,
        use_inductor: bool = False,
        model_seed: int = DEFAULT_MODEL_SEED,
        actor_group_id: int = 0,
        actor_group_local_rank: int = 0,
        actor_group_devices: tuple[int, ...] = (),
        actor_groups: tuple[tuple[int, ...], ...] = (),
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
            global_rank=global_rank,
            pp_rank=pp_rank,
            pp_degree=pp_degree,
            world_size=world_size,
            no_nvtx=no_nvtx,
            actor_group_id=actor_group_id,
            actor_group_local_rank=actor_group_local_rank,
            actor_group_devices=tuple(actor_group_devices),
            actor_groups=tuple(tuple(group) for group in actor_groups),
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
        warmup_iters: int = 2,
        active_iters: int = 1,
        activities: tuple[str, ...] | None = None,
    ) -> None:
        """Begin a scheduled torch.profiler session at DAG boundaries.

        Each node's execution in DagExecutor is wrapped in a record_function
        labelled the same as its NVTX range, so the resulting trace identifies
        per-node work.
        """
        if self.runtime.torch_profiler is not None:
            raise RuntimeError("PyTorch profiler is already running")
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
        self.logger.debug(
            f"Actor {self.runtime.global_rank}: PyTorch profiler started with "
            f"{warmup_iters} warmup iterations"
        )

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
        """Advance an active profiler at a completed GPU iteration boundary."""

        if not self.runtime.pytorch_profiler_enabled:
            raise RuntimeError("PyTorch profiler is not active")
        # ``torch.profiler.step()`` changes the Kineto collection phase
        # immediately.  Drain work launched in the current phase first so CUDA
        # activity from a discarded warmup iteration cannot leak into the
        # active trace without its CPU-side record_function attribution.
        torch.cuda.synchronize(self.runtime.device)
        self.runtime.torch_profiler.step()

    def start_cupti_node_profiler(
        self,
        *,
        warmup_iters: int = 2,
        active_iters: int = 1,
        coordinator_clock_offset_ns: int = 0,
        clock_sync_uncertainty_ns: int = 0,
    ) -> None:
        """Start CUPTI/Kineto capture for node ``record_function`` ranges.

        This adds no per-node timing events to the DAG streams. Kineto
        attributes CUDA activity to the existing node ranges through CUPTI
        correlation IDs.
        """
        self._cupti_allocator_iteration_stats = []
        self._cupti_timeline_anchor_origins: dict[
            str, tuple[float, float]
        ] = {}
        self._cupti_timeline_anchor_counter = 0
        self._cupti_expected_active_iters = int(active_iters)
        self._cupti_coordinator_clock_offset_ns = int(
            coordinator_clock_offset_ns
        )
        self._cupti_clock_sync_uncertainty_ns = int(
            clock_sync_uncertainty_ns
        )
        self.start_pytorch_profiler(
            warmup_iters=warmup_iters, active_iters=active_iters
        )

    def profile_clock_sample(self) -> int:
        """Return this actor's monotonic clock for driver midpoint calibration."""

        return time.monotonic_ns()

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

        The file is named by its authoritative global actor rank.
        """
        self.runtime.pytorch_profiler_enabled = False
        self.runtime.torch_profiler.__exit__(None, None, None)
        os.makedirs(profile_dir, exist_ok=True)
        filepath = os.path.join(profile_dir, f"rank{self.runtime.global_rank}.json")
        self.runtime.torch_profiler.export_chrome_trace(filepath)
        self.runtime.torch_profiler = None
        self.logger.info(
            f"Actor {self.runtime.global_rank}: PyTorch profiler trace written to {filepath}"
        )
        return filepath

    def stop_cupti_node_profiler(
        self, *, trace_path: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Return CUPTI-attributed CUDA totals keyed by Piper node range."""
        if not self.runtime.pytorch_profiler_enabled:
            raise RuntimeError("CUPTI node profiler is not active")
        profiler = self.runtime.torch_profiler
        self.runtime.pytorch_profiler_enabled = False
        profiler.__exit__(None, None, None)
        attribution_audit, timeline = _kineto_node_attribution_audit(
            profiler,
            trace_path=trace_path,
            timeline_anchor_origins=getattr(
                self, "_cupti_timeline_anchor_origins", {}
            ),
            expected_timeline_anchors=getattr(
                self, "_cupti_expected_active_iters", None
            ),
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
        self._last_cupti_timeline = timeline
        self._last_cupti_trace_path = trace_path
        self.runtime.torch_profiler = None
        return results

    def stop_cupti_profile_result(
        self,
        *,
        estimator: str = "median",
        profile_mode: str = "isolated",
        trace_output_dir: str | None = None,
        trace_ranks: tuple[int, ...] | None = None,
    ) -> ActorProfileResult:
        """Convert a Kineto/CUPTI capture into one actor's profile result.

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
                trace_ranks is None
                or int(self.runtime.global_rank) in set(trace_ranks)
            )
        )
        trace_path = (
            os.path.join(
                str(trace_output_dir),
                f"{mode}_rank{self.runtime.global_rank}"
                f"_group{self.runtime.actor_group_id}.json",
            )
            if capture_trace
            else None
        )
        captured = self.stop_cupti_node_profiler(trace_path=trace_path)
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
                        if mode == "isolated"
                        else DEFAULT_NORMAL_SCOPE
                    ),
                )
                continue
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
            occurrence_records = tuple(
                measured.get("range_activity_signatures", ())
            )
            try:
                start_offset_samples_us = tuple(
                    float(record["activity_start_offset_us"])
                    for record in occurrence_records
                )
            except (KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"Kineto attribution for node {uid} has no aligned activity "
                    "start offsets"
                ) from exc
            if len(start_offset_samples_us) != len(duration_samples_us):
                raise RuntimeError(
                    f"Kineto attribution for node {uid} has mismatched duration "
                    f"and start samples: durations={len(duration_samples_us)}, "
                    f"starts={len(start_offset_samples_us)}"
                )
            node_kind = str(getattr(node, "node_kind", ""))
            p2p_post_offset_samples_us: tuple[float, ...] = ()
            if node_kind in {"SEND_COMM", "RECV_COMM"}:
                try:
                    p2p_post_offset_samples_us = tuple(
                        float(record["p2p_post_offset_us"])
                        for record in occurrence_records
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Kineto attribution for P2P node {uid} has no host "
                        "post offsets"
                    ) from exc
                if len(p2p_post_offset_samples_us) != len(
                    duration_samples_us
                ):
                    raise RuntimeError(
                        f"Kineto attribution for P2P node {uid} has mismatched "
                        f"duration and post samples: durations="
                        f"{len(duration_samples_us)}, posts="
                        f"{len(p2p_post_offset_samples_us)}"
                    )
            duration_source = "correlated_gpu_activity_envelope"
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
            node_attribution_audit[uid]["node_kind"] = node_kind
            node_attribution_audit[uid]["comm_id"] = node.node_meta.get(
                "comm_id"
            )
            node_attribution_audit[uid]["replay_duration_source"] = (
                duration_source
            )
            node_attribution_audit[uid]["replay_duration_samples_us"] = list(
                duration_samples_us
            )
            if mode == "isolated":
                isolation_scope = DEFAULT_ISOLATION_SCOPE
            else:
                isolation_scope = DEFAULT_NORMAL_SCOPE
            node_profiles[uid] = NodeProfile.from_samples(
                duration_samples_us,
                estimator=estimator,
                timing_source="kineto_cupti",
                isolation_scope=isolation_scope,
                start_offset_samples_us=start_offset_samples_us,
                p2p_post_offset_samples_us=p2p_post_offset_samples_us,
            )
        props = torch.cuda.get_device_properties(self.runtime.device)
        return ActorProfileResult(
            profile_mode=mode,
            graph_fingerprint=finalized_dag_fingerprint(self.dag),
            global_rank=self.runtime.global_rank,
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
                "actor_group_id": int(self.runtime.actor_group_id),
                "actor_group_devices": list(self.runtime.actor_group_devices),
                "allocator_iteration_stats": allocator_iteration_stats,
                "kineto_attribution_audit": node_attribution_audit,
            },
            node_profiles=node_profiles,
            timeline_origin_samples_us=tuple(
                getattr(self, "_last_cupti_timeline", {}).get(
                    "timeline_origin_samples_us", ()
                )
            ),
            timeline_origin_uncertainty_samples_us=tuple(
                getattr(self, "_last_cupti_timeline", {}).get(
                    "timeline_origin_uncertainty_samples_us", ()
                )
            ),
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
        by one ActorGroup. Every group member enters each all-gather in the same
        bucket order, while only local rank zero returns the reconstructed state.
        """
        self.params.drain_pending_frees()
        torch.cuda.synchronize()
        representative = self.runtime.is_actor_group_representative
        state: dict[str, torch.Tensor] = {}
        stream = self.runtime.default_stream()

        for ubid, bucket in sorted(
            self.stages.buckets.items(), key=lambda item: str(item[0])
        ):
            name_by_arg_idx = dict(zip(bucket.param_idxs, bucket.param_names))
            route_specs = bucket.route_sharded_param_specs
            if route_specs:
                if ubid in self.stages.param_sharded_ubids:
                    raise RuntimeError(
                        f"route-owned bucket {ubid} cannot also use ZeRO "
                        "parameter sharding across distinct route owners"
                    )
                group_size = self.runtime.actor_group_size
                sync_group = self.runtime.actor_group_sync_group
                if group_size > 1 and sync_group is None:
                    raise RuntimeError(
                        f"cannot export route-owned parameters for bucket "
                        f"{ubid}: ActorGroup communicator is missing"
                    )
                canonical_devices = tuple(
                    sorted(int(rank) for rank in self.runtime.actor_group_devices)
                )
                for arg_idx in sorted(route_specs):
                    value = bucket.forward_args[arg_idx]
                    if value is None:
                        raise RuntimeError(
                            f"route-owned parameter index {arg_idx} in bucket "
                            f"{ubid} has no storage"
                        )
                    spec = route_specs[arg_idx]
                    if group_size == 1:
                        gathered = value.detach()
                    else:
                        gathered = torch.empty(
                            (value.shape[0] * group_size, *value.shape[1:]),
                            dtype=value.dtype,
                            device=value.device,
                        )
                        with torch.cuda.stream(stream):
                            dist.all_gather_into_tensor(
                                gathered, value.detach(), group=sync_group
                            )
                        stream.synchronize()
                    if representative:
                        shard_devices = tuple(
                            int(device) for device in spec["shard_devices"]
                        )
                        if shard_devices != canonical_devices:
                            raise RuntimeError(
                                f"route-owned export for bucket {ubid} requires "
                                "canonical ascending shard_devices; got "
                                f"{shard_devices}, expected {canonical_devices}"
                            )
                        global_value = gathered
                        expected_shape = tuple(spec["global_shape"])
                        if tuple(global_value.shape) != expected_shape:
                            raise RuntimeError(
                                f"route-owned export for bucket {ubid} "
                                f"produced shape={tuple(global_value.shape)}, "
                                f"expected={expected_shape}"
                            )
                        self._add_exported_parameter(
                            state,
                            name_by_arg_idx.get(arg_idx),
                            global_value,
                        )
                if representative:
                    for arg_idx in bucket.trainable_param_idxs:
                        if arg_idx in route_specs:
                            continue
                        value = bucket.forward_args[arg_idx]
                        if value is None:
                            raise RuntimeError(
                                f"cannot export parameter index {arg_idx} in "
                                f"bucket {ubid}: parameter storage is missing"
                            )
                        self._add_exported_parameter(
                            state, name_by_arg_idx.get(arg_idx), value
                        )
                continue
            if ubid in self.stages.param_sharded_ubids:
                group_size = self.runtime.actor_group_size
                sync_group = self.runtime.actor_group_sync_group
                if group_size > 1 and sync_group is None:
                    raise RuntimeError(
                        f"cannot export sharded parameters for bucket {ubid}: "
                        "ActorGroup synchronization communicator is missing"
                    )
                shard_param = bucket.shard_param
                shard_info = bucket.param_shard_info
                if shard_param is None or shard_info is None:
                    raise RuntimeError(
                        f"cannot export sharded parameters for bucket {ubid}: "
                        "missing shard storage metadata"
                    )
                _shard_start, shard_size, orig_numel = shard_info
                gathered = torch.empty(
                    shard_size * group_size,
                    dtype=shard_param.dtype,
                    device=shard_param.device,
                )
                with torch.cuda.stream(stream):
                    dist.all_gather_into_tensor(
                        gathered, shard_param, group=sync_group
                    )
                stream.synchronize()
                if not representative:
                    continue
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

            if not representative:
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

        if self.runtime.world_size > 1:
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

            self._join_actor_group_process_groups()

            self.logger.debug(f"Actor {self.runtime.global_rank} joined process groups")

    def _join_actor_group_process_groups(self):
        """Create deterministic communicators for AFD ActorGroups.

        The DAG is not loaded yet when distributed initialization runs, so the
        schedule topology is the source of truth. Create one communicator for
        every within-group participant set, three routed communicators for every
        pairwise group union, and two P2P communicators for each aligned rank
        pair of equal-width groups. Lowering can then request exact participants
        and a traffic channel. All global ranks call ``new_group`` in the same
        deterministic order.
        """
        actor_groups = tuple(
            tuple(sorted(int(rank) for rank in group))
            for group in self.runtime.actor_groups
        )
        # Keep layout, M-to-N, and N-to-M on independent NCCL communicator
        # channels even when their participant set is equal. Their CUDA issue
        # order is finalized separately from the schedule-declared stream.
        for participants, channel in self.runtime.process_group_creation_specs():
            process_group = dist.new_group(
                ranks=list(participants),
                backend="nccl",
                timeout=self._process_group_timeout,
            )
            if self.runtime.global_rank in participants:
                key = (
                    participants
                    if channel is None
                    else (participants, channel)
                )
                self.runtime.process_groups_by_participants[key] = process_group

        # Force the directional P2P communicators to initialize in the same
        # global order before schedule execution. A collective on the process
        # group does not initialize NCCL's separate lazy P2P connection.
        p2p_warmup = torch.zeros(1, dtype=torch.uint8, device=self.runtime.device)
        for participants, channel in self.runtime.required_p2p_process_group_specs():
            if self.runtime.global_rank not in participants:
                continue
            process_group = self.runtime.process_group_for_participants(
                participants,
                channel=channel,
            )
            low_rank, high_rank = participants
            if channel in {"pp_lo_hi", "dense_state_lo_hi"}:
                source_rank, destination_rank = low_rank, high_rank
            elif channel in {"pp_hi_lo", "dense_state_hi_lo"}:
                source_rank, destination_rank = high_rank, low_rank
            else:
                raise RuntimeError(f"unknown P2P channel {channel!r}")
            if self.runtime.global_rank == source_rank:
                dist.send(p2p_warmup, dst=destination_rank, group=process_group)
            else:
                dist.recv(p2p_warmup, src=source_rank, group=process_group)

        own_group = tuple(sorted(self.runtime.actor_group_devices))
        if own_group:
            self.runtime.actor_group_routed_group = (
                self.runtime.process_group_for_participants(own_group)
            )

        # Replicated dense weights and routed expert traffic use distinct NCCL
        # communicators so reductions and activation collectives can overlap.
        for participants in sorted(actor_groups):
            sync_group = dist.new_group(
                ranks=list(participants),
                backend="nccl",
                timeout=self._process_group_timeout,
            )
            if self.runtime.global_rank in participants:
                self.runtime.actor_group_sync_group = sync_group

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
            restore_triton_side_table(
                gms,
                kernels=dict(bd.get("triton_kernels", {})),
                constant_args=dict(bd.get("triton_constant_args", {})),
            )
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
            raw_route_specs = list(bd.get("route_sharded_param_specs", ()))
            route_spec_by_arg_idx: dict[int, dict[str, Any]] = {}
            if raw_route_specs:
                shard_devices = tuple(int(rank) for rank in bd["shard_devices"])
                if shard_devices != tuple(sorted(shard_devices)):
                    raise RuntimeError(
                        f"route-owned bucket {ubid} requires canonical ascending "
                        f"shard_devices, got {shard_devices}"
                    )
                if set(shard_devices) != set(self.runtime.actor_group_devices):
                    raise RuntimeError(
                        f"route-owned bucket {ubid} shard_devices={shard_devices} "
                        "do not match its ActorGroup "
                        f"{self.runtime.actor_group_devices}"
                    )
                try:
                    ownership_index = shard_devices.index(self.runtime.global_rank)
                except ValueError as exc:
                    raise RuntimeError(
                        f"actor rank {self.runtime.global_rank} has no route "
                        f"partition in shard_devices={shard_devices}"
                    ) from exc
                partitions = list(bd["route_partitions"])
                partition = partitions[ownership_index]
                if int(partition["device"]) != self.runtime.global_rank:
                    raise RuntimeError(
                        f"route partition {partition} does not describe actor "
                        f"rank {self.runtime.global_rank}"
                    )
                route_start = int(partition["start"])
                route_stop = int(partition["stop"])
                for raw_spec in raw_route_specs:
                    spec = dict(raw_spec)
                    arg_idx = int(spec["arg_idx"])
                    if arg_idx in route_spec_by_arg_idx:
                        raise RuntimeError(
                            f"duplicate route-owned parameter index {arg_idx} "
                            f"in bucket {ubid}"
                        )
                    if int(spec.get("route_axis", -1)) != 0:
                        raise RuntimeError(
                            "only leading-axis route-owned parameters are "
                            f"supported, got {spec}"
                        )
                    local_shape = tuple(int(dim) for dim in spec["local_shape"])
                    if not local_shape or local_shape[0] != route_stop - route_start:
                        raise RuntimeError(
                            f"route-owned parameter {spec} disagrees with "
                            f"partition {partition}"
                        )
                    spec["global_route_start"] = route_start
                    spec["shard_devices"] = shard_devices
                    route_spec_by_arg_idx[arg_idx] = spec
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
                    route_spec = route_spec_by_arg_idx.get(i)
                    if route_spec is None:
                        initialize_parameter_(
                            t,
                            state_key,
                            self.model_seed,
                        )
                    else:
                        initialize_route_parameter_(
                            t,
                            state_key,
                            global_route_start=route_spec["global_route_start"],
                            model_seed=self.model_seed,
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
            bucket.route_sharded_param_specs = route_spec_by_arg_idx
            bucket.activation_checkpoint_subgraph_count = ac_num_subgraphs
            trainable_idxs = [
                i for i in b_param_idxs
                if realized[i] is not None and realized[i].requires_grad
            ]
            bucket.trainable_param_idxs = trainable_idxs

            zero_requested = (
                apply_zero
                and bool(trainable_idxs)
                and ubid in self.stages.zero_managed_ubids
            )
            if zero_requested:
                self.runtime.validate_actor_group_metadata(
                    context=f"load ZeRO bucket {ubid!r}"
                )
            actor_group_size = self.runtime.actor_group_size
            zero_managed = zero_requested and actor_group_size > 1
            params_sharded = ubid in self.stages.param_sharded_ubids
            grads_sharded = ubid in self.stages.grad_sharded_ubids

            if zero_managed:
                trainable = [realized[i] for i in trainable_idxs]
                flat_params = torch.cat([p.detach().view(-1) for p in trainable]).contiguous()
                flat_params.requires_grad_(False)
                orig_numel = flat_params.numel()
                shard_size = (
                    orig_numel + actor_group_size - 1
                ) // actor_group_size
                padded_numel = shard_size * actor_group_size
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

                shard_start = self.runtime.actor_group_local_rank * shard_size
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

    def load_training_dag(
        self,
        training_dag: Any,
        isolated_order: Sequence[str],
    ) -> None:
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

        isolated_order = tuple(isolated_order)
        if (
            len(isolated_order) != len(training_dag.nodes)
            or len(set(isolated_order)) != len(isolated_order)
            or set(isolated_order) != set(training_dag.nodes)
        ):
            raise ValueError(
                "isolated order must contain every TrainingDAG UID "
                f"exactly once: order={isolated_order!r}, "
                f"nodes={sorted(training_dag.nodes)!r}"
            )
        isolated_index = {uid: index for index, uid in enumerate(isolated_order)}
        for edge in training_dag.edges:
            if isolated_index[edge.src_uid] >= isolated_index[edge.dst_uid]:
                raise ValueError(
                    "isolated order violates TrainingDAG edge "
                    f"{edge.src_uid}->{edge.dst_uid} ({edge.dep_kind})"
                )

        self._derive_dag_bucket_modes(training_dag)
        self.runtime.initialize_streams_for_training_dag(training_dag)

        # Clear any prior module state before loading a new training DAG.
        self.stages.clear_loaded_modules()
        self.dag_executor.expert_a2a_states.clear()
        self.dag_executor.routed_cross_group_states.clear()

        routed_expert_fwd_uids = {
            meta["expert_fwd_uid"]
            for dag_node in training_dag.nodes.values()
            for meta in [getattr(dag_node, "node_meta", {}) or {}]
            if getattr(dag_node, "node_kind", None)
            in {"A2A_COMM", "M_TO_N_COMM", "N_TO_M_COMM"}
            and isinstance(meta.get("expert_fwd_uid"), str)
        }

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
                # Routed/expert identity comes from the communication
                # contract inserted by `shard`, never from a tag spelling.
                "is_expert": getattr(node, "uid", None)
                in routed_expert_fwd_uids,
                "training_dag_uid": getattr(node, "uid", None),
                "triton_kernels": dict(meta.get("triton_kernels", {})),
                "triton_constant_args": dict(meta.get("triton_constant_args", {})),
                "route_sharded_param_specs": list(
                    meta.get("route_sharded_param_specs", ())
                ),
                "shard_devices": list(meta.get("shard_devices", ())),
                "route_partitions": list(meta.get("route_partitions", ())),
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
            elif e.dep_kind == "data":
                src.data_succs.append(dst)
                dst.data_preds.append(src)
            elif e.dep_kind != "issue":
                raise ValueError(
                    f"unsupported dependency kind {e.dep_kind!r} on "
                    f"{e.src_uid} -> {e.dst_uid}"
                )

        for n in training_dag.nodes.values():
            n.task_type = _training_dag_task_type(n)
            mb = n.tag.get("MB", 0)
            st = n.tag.get("PP", 0)
            n.batches = [type("RuntimeBatch", (), {"stage_id": st, "mb_idx": mb})()]

        self.dag = training_dag
        self.sorted_dag_nodes = [
            training_dag.nodes[uid]
            for uid in _dispatch_order(training_dag)
        ]
        self.prepared_dag_execution = self.dag_executor.prepare(
            training_dag,
            self.sorted_dag_nodes,
            stream_override=self.runtime.execution_stream_for_task,
        )
        # Normal profiling uses the ordinary issue order exactly. The
        # isolated policy uses the transaction-consistent projection
        # computed once from all ActorGroup DAGs.
        self.profile_prepared_dag_executions = {
            "normal": self.prepared_dag_execution,
        }
        for profile_mode in sorted(PROFILE_MODES - {"normal"}):
            self.profile_prepared_dag_executions[profile_mode] = (
                self.dag_executor.prepare(
                    training_dag,
                    [training_dag.nodes[uid] for uid in isolated_order],
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
                self.step_pytorch_profiler()
        # Timing/profiling callers do not consume losses. Avoid serializing CUDA
        # tensors back to the CPU-only program driver in that common path.
        return result if return_losses else None

    def _profile_stream_override(
        self,
        node: Any,
        logical_stream: torch.cuda.Stream,
        *,
        profile_mode: str = "isolated",
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
        synchronize_before: bool = True,
        synchronize: bool = False,
        profile_mode: str = "isolated",
        advance_profiler: bool = True,
        capture_allocator_stats: bool = True,
    ) -> None:
        """Run one timed iteration using isolated or normal streams.

        Both modes preserve each node's normal logical stream (including the
        directional P2P streams). ``isolated`` additionally chains
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
        timeline_anchor = None
        if (
            self.runtime.pytorch_profiler_enabled
            and hasattr(self, "_cupti_timeline_anchor_origins")
        ):
            anchor_index = int(self._cupti_timeline_anchor_counter)
            self._cupti_timeline_anchor_counter = anchor_index + 1
            anchor_label = (
                f"{_KINETO_ITERATION_ANCHOR_PREFIX}{mode}:{anchor_index}:"
                f"rank{self.runtime.global_rank}"
            )
            local_before_ns = time.monotonic_ns()
            timeline_anchor = torch.profiler.record_function(anchor_label)
            timeline_anchor.__enter__()
            local_after_ns = time.monotonic_ns()
            local_midpoint_ns = (local_before_ns + local_after_ns) // 2
            coordinator_origin_us = (
                local_midpoint_ns
                + int(self._cupti_coordinator_clock_offset_ns)
            ) / 1000.0
            uncertainty_us = (
                (local_after_ns - local_before_ns) / 2.0
                + int(self._cupti_clock_sync_uncertainty_ns)
            ) / 1000.0
            self._cupti_timeline_anchor_origins[anchor_label] = (
                coordinator_origin_us,
                uncertainty_us,
            )
        try:
            self._execute_dag(
                loss_fn=loss_fn,
                prepared_execution=self.profile_prepared_dag_executions[mode],
                serialize_nodes=(mode == ISOLATED_PROFILE_MODE),
                iteration_label=f"profile_{mode}_iter",
            )
        finally:
            if timeline_anchor is not None:
                timeline_anchor.__exit__(None, None, None)
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
                    self.step_pytorch_profiler()
        if synchronize:
            torch.cuda.synchronize(self.runtime.device)

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

        Normal and isolated profiles must start from identical model
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
