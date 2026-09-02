# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Isolated, node-level replay for profiled TrainingDAGs.

The replay is deliberately CUDA-free. It uses the selected policy's profiled
duration for each non-communication node, preserves
TrainingDAG dependencies and CUDA stream FIFO order, and otherwise provides no
shared GPU or network capacity.  Consequently, independent streams and
communication transactions overlap completely.  Communication endpoint
timings can include time spent waiting for other ranks to arrive.  The
replay therefore separates rank-local endpoint arrival and completion with
a shared zero-duration rendezvous. For each policy it derives every endpoint's
post-rendezvous service time from that policy's observed, cross-actor-aligned
timeline. Symmetric collectives and A2A use an all-participant rendezvous
followed by rank-local aggregate envelope tails. M-to-N and N-to-M nodes are
not replayed because their aggregate node envelopes do not identify the
service times of their internal peer transfers.
Communication profiles with observed cross-actor offsets fail closed when
timeline alignment is unavailable.

Each matched P2P SEND/RECV node pair is one compound transaction, even when its
runtime protocol makes multiple P2P calls. The aggregate CUDA envelope supplies
independent endpoint service tails. Send completion is local: it waits only for
its own first call and send-source readiness. Receive completion waits for both
first calls plus send-source and receive-destination readiness. Successors wait
only for their local endpoint's completion.
Every finalized data and temporal dependency is preserved. Issue-only edges
control executor dispatch but do not constrain replay timing. Directional P2P
physical streams remain independent, while their DAG-encoded FIFO and
completion dependencies remain authoritative.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import heapq
import json
import math
from numbers import Real
import statistics
from typing import Any

from .dag import TrainingDAG, TrainingDAGNode
from .execution_plan import (
    PlannedOperation,
    build_execution_plan,
)
from .profiling import (
    NORMAL_PROFILE_MODE,
    ISOLATED_PROFILE_MODE,
    estimate_profile_time,
    get_node_profile_times,
    get_prologue_profile_times,
    normalize_profile_mode,
)
from .profile_statistics import (
    MeanStatistics,
    PairedSavingsStatistics,
    PredictionError,
    TimingStatistics,
    predicted_reduction,
    prediction_error,
    summarize_mean_samples,
    summarize_paired_savings,
    summarize_timing_samples,
)


_PROLOGUE_UID = "__step_prologue__"
_P2P_KINDS = frozenset({"SEND_COMM", "RECV_COMM"})
_A2A_COLLECTIVE_KINDS = frozenset({"A2A_COMM"})
_UNSUPPORTED_ROUTED_COMM_KINDS = frozenset(
    {"M_TO_N_COMM", "N_TO_M_COMM"}
)
_SYMMETRIC_COLLECTIVE_KINDS = frozenset(
    {
        "REDUCE_COMM",
        "ALL_REDUCE_COMM",
        "ALL_GATHER_COMM",
        "REDUCE_SCATTER_COMM",
    }
) | _A2A_COLLECTIVE_KINDS
_COLLECTIVE_KINDS = _SYMMETRIC_COLLECTIVE_KINDS
_COMM_KINDS = _P2P_KINDS | _COLLECTIVE_KINDS


class ReplayValidationError(ValueError):
    """Raised when an input cannot define an unambiguous replay."""


@dataclass(frozen=True, order=True)
class NodeInstance:
    """A rank-local occurrence of a finalized TrainingDAG node."""

    global_rank: int
    uid: str


@dataclass(frozen=True)
class TimelineEvent:
    """The replayed interval for one rank-local node occurrence.

    ``start_us`` is the earliest legal service start implied by the DAG,
    physical-stream FIFO constraints, and communication gates. Observed CUDA
    start offsets are retained only for validation. P2P host-post offsets are
    distinct release constraints; the normal post schedule is shared by both
    timing modes, so normal and isolated replays differ solely by service
    times.
    """

    operation: NodeInstance
    node_kind: str
    stream: str
    start_us: float
    end_us: float
    profile_duration_us: float
    effective_duration_us: float
    profile_start_offset_us: float = 0.0
    observed_profile_start_offset_us: float = 0.0
    arrival_start_us: float | None = None
    communication_id: str | None = None
    critical_predecessor: NodeInstance | None = None

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class CriticalPathEntry:
    """One (possibly multi-rank) operation on the replayed critical path."""

    operation_id: str
    members: tuple[NodeInstance, ...]
    start_us: float
    end_us: float
    effective_duration_us: float


@dataclass(frozen=True)
class ReplayResult:
    """One policy's step-time estimate and its supporting timeline."""

    step_time_us: float
    per_rank_makespan_us: dict[int, float]
    timeline: tuple[TimelineEvent, ...]
    critical_path: tuple[CriticalPathEntry, ...]
    warnings: tuple[str, ...] = ()
    profile_mode: str = ISOLATED_PROFILE_MODE
    sample_iteration_id: int | None = None

    @property
    def step_time_s(self) -> float:
        return self.step_time_us / 1_000_000.0

    def event(self, global_rank: int, uid: str) -> TimelineEvent:
        key = NodeInstance(global_rank, uid)
        for event in self.timeline:
            if event.operation == key:
                return event
        raise KeyError(key)


@dataclass(frozen=True)
class ReplayComparisonResult:
    """Isolated and normal replays plus uncertainty and calibration."""

    isolated: ReplayResult
    normal: ReplayResult
    isolated_samples_us: tuple[float, ...]
    normal_samples_us: tuple[float, ...]
    observed_normal_envelope_samples_us: tuple[float, ...]
    isolated_statistics: MeanStatistics
    normal_statistics: MeanStatistics
    perfect_overlap_savings_statistics: PairedSavingsStatistics
    observed_normal_envelope_statistics: TimingStatistics | None = None
    actual_statistics: TimingStatistics | None = None
    instrumented_normal_statistics: TimingStatistics | None = None
    instrumented_isolated_statistics: TimingStatistics | None = None
    uninstrumented_normal_statistics: TimingStatistics | None = None
    uninstrumented_isolated_statistics: TimingStatistics | None = None
    normal_replay_error: PredictionError | None = None
    instrumented_normal_replay_error: PredictionError | None = None
    instrumented_isolated_replay_error: PredictionError | None = None
    prediction_error_pct: float | None = None
    perfect_overlap_absolute_savings_us: float | None = None
    interference_overhead_pct: float | None = None
    instrumented_policy_reduction_pct: float | None = None
    observed_interference_reduction_pct: float | None = None
    envelope_modeled_interference_reduction_pct: float | None = None
    normal_dependency_replay_excess_pct: float | None = None
    profile_event_overhead_pct: float | None = None
    mode_profile_event_overhead_pct: float | None = None
    kineto_cupti_instrumentation_error: PredictionError | None = None


@dataclass(frozen=True)
class EarliestLegalStartComparison:
    """Validation of DAG/stream readiness against observed CUDA starts."""

    samples: int
    mean_signed_error_us: float
    median_signed_error_us: float
    median_absolute_error_us: float
    p95_absolute_error_us: float


def compare_earliest_legal_starts(
    plan_or_dags: Any,
    *,
    profile_mode: str = NORMAL_PROFILE_MODE,
    strict: bool = True,
) -> EarliestLegalStartComparison:
    """Compare replayed earliest legal starts with recorded CUDA starts.

    The comparison is per node instance and raw profile iteration.  A positive
    error means the observed CUDA start was later than its DAG/stream-ready
    time, i.e. residual launch or resource-queue delay not explained by the
    modeled ordering constraints.
    """
    mode = normalize_profile_mode(profile_mode)
    errors: list[float] = []
    for iteration_id in profile_iteration_ids(plan_or_dags, profile_mode=mode):
        result = replay_training_step(
            plan_or_dags,
            profile_mode=mode,
            sample_iteration_id=iteration_id,
            strict=strict,
        )
        for event in result.timeline:
            observed = event.observed_profile_start_offset_us
            earliest_start = (
                event.start_us
                if event.arrival_start_us is None
                else event.arrival_start_us
            )
            errors.append(float(observed) - float(earliest_start))
    if not errors:
        raise ReplayValidationError(
            f"profile mode {mode!r} has no events to compare"
        )
    absolute_errors = sorted(abs(error) for error in errors)
    p95_index = min(
        len(absolute_errors) - 1,
        max(0, math.ceil(0.95 * len(absolute_errors)) - 1),
    )
    return EarliestLegalStartComparison(
        samples=len(errors),
        mean_signed_error_us=statistics.fmean(errors),
        median_signed_error_us=statistics.median(errors),
        median_absolute_error_us=statistics.median(absolute_errors),
        p95_absolute_error_us=absolute_errors[p95_index],
    )

@dataclass(frozen=True)
class _RankLayout:
    ranks_by_pp: tuple[tuple[int, ...], ...]
    lane_by_rank: dict[int, int]
    pp_by_rank: dict[int, int]

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(rank for ranks in self.ranks_by_pp for rank in ranks)


@dataclass(frozen=True)
class _InstanceRecord:
    key: NodeInstance
    pp_index: int
    actor_group_local_rank: int
    node: TrainingDAGNode
    profile: Any | None
    profile_duration_us: float
    profile_start_offset_us: float = 0.0
    observed_profile_start_offset_us: float = 0.0
    p2p_post_offset_us: float | None = None


@dataclass
class _Vertex:
    vertex_id: str
    members: list[_InstanceRecord]
    duration_us: float
    release_us: float = 0.0
    communication_id: str | None = None
    emit_timeline: bool = True
    endpoint_arrival_vertex_id: str | None = None
    endpoint_rendezvous_vertex_id: str | None = None
    endpoint_observed_service_us: float | None = None
    rendezvous_wait_us: float = 0.0
    preds: set[str] = field(default_factory=set)
    succs: set[str] = field(default_factory=set)
    start_us: float = 0.0
    end_us: float = 0.0
    critical_pred: str | None = None


def _identity_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def _profile_duration_us(
    profile: Any,
    *,
    context: str,
    sample_iteration_id: int | None = None,
) -> float:
    if isinstance(profile, bool):
        raise ReplayValidationError(f"{context}: boolean is not a valid profile duration")
    if isinstance(profile, Real):
        value = float(profile)
    elif isinstance(profile, Mapping):
        if sample_iteration_id is not None and profile.get("samples_us"):
            raw_ids = profile.get("sample_iteration_ids")
            samples = tuple(float(value) for value in profile["samples_us"])
            ids = (
                tuple(range(len(samples)))
                if raw_ids is None
                else tuple(int(value) for value in raw_ids)
            )
            try:
                value = samples[ids.index(int(sample_iteration_id))]
            except ValueError as exc:
                raise ReplayValidationError(
                    f"{context}: no raw sample for iteration {sample_iteration_id}"
                ) from exc
        elif "estimate_us" in profile:
            value = float(profile["estimate_us"])
        elif "duration_us" in profile:
            value = float(profile["duration_us"])
        else:
            raise ReplayValidationError(
                f"{context}: profile mapping has neither estimate_us nor duration_us"
            )
    else:
        samples = getattr(profile, "samples_us", ())
        if sample_iteration_id is not None and samples:
            ids = getattr(profile, "sample_iteration_ids", ()) or tuple(
                range(len(samples))
            )
            try:
                value = float(samples[tuple(ids).index(int(sample_iteration_id))])
            except ValueError as exc:
                raise ReplayValidationError(
                    f"{context}: no raw sample for iteration {sample_iteration_id}"
                ) from exc
            estimate = value
        else:
            estimate = getattr(profile, "estimate_us", None)
        if estimate is None:
            estimate = getattr(profile, "duration_us", None)
        if estimate is None:
            raise ReplayValidationError(
                f"{context}: profile has neither estimate_us nor duration_us"
            )
        value = float(estimate)
    if not math.isfinite(value) or value < 0.0:
        raise ReplayValidationError(
            f"{context}: duration must be finite and non-negative, got {value!r}"
        )
    return value


def _profile_start_offset_us(
    profile: Any,
    *,
    context: str,
    sample_iteration_id: int | None = None,
) -> float:
    """Read an optional iteration-aligned GPU timeline release offset."""

    if isinstance(profile, Mapping):
        raw_samples = profile.get("start_offset_samples_us", ())
        estimate = profile.get("start_offset_estimate_us")
        raw_ids = profile.get("sample_iteration_ids", ())
    else:
        raw_samples = getattr(profile, "start_offset_samples_us", ())
        estimate = getattr(profile, "start_offset_estimate_us", None)
        raw_ids = getattr(profile, "sample_iteration_ids", ())
    samples = tuple(float(value) for value in (raw_samples or ()))
    if sample_iteration_id is not None and samples:
        ids = tuple(int(value) for value in (raw_ids or range(len(samples))))
        try:
            value = samples[ids.index(int(sample_iteration_id))]
        except ValueError as exc:
            raise ReplayValidationError(
                f"{context}: no start-offset sample for iteration "
                f"{sample_iteration_id}"
            ) from exc
    elif estimate is not None:
        value = float(estimate)
    elif samples:
        ordered = sorted(samples)
        midpoint = len(ordered) // 2
        value = (
            float(ordered[midpoint])
            if len(ordered) % 2
            else float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)
        )
    else:
        return 0.0
    if not math.isfinite(value) or value < 0.0:
        raise ReplayValidationError(
            f"{context}: start offset must be finite and non-negative, got {value!r}"
        )
    return value


def _profile_p2p_post_offset_us(
    profile: Any,
    *,
    context: str,
    sample_iteration_id: int | None = None,
) -> float | None:
    """Read an optional offset for entry to a node's first P2P call."""

    if isinstance(profile, Mapping):
        raw_samples = profile.get("p2p_post_offset_samples_us", ())
        estimate = profile.get("p2p_post_offset_estimate_us")
        raw_ids = profile.get("sample_iteration_ids", ())
    else:
        raw_samples = getattr(profile, "p2p_post_offset_samples_us", ())
        estimate = getattr(profile, "p2p_post_offset_estimate_us", None)
        raw_ids = getattr(profile, "sample_iteration_ids", ())
    samples = tuple(float(value) for value in (raw_samples or ()))
    if sample_iteration_id is not None and samples:
        ids = tuple(int(value) for value in (raw_ids or range(len(samples))))
        try:
            value = samples[ids.index(int(sample_iteration_id))]
        except ValueError as exc:
            raise ReplayValidationError(
                f"{context}: no P2P post-offset sample for iteration "
                f"{sample_iteration_id}"
            ) from exc
    elif estimate is not None:
        value = float(estimate)
    elif samples:
        value = estimate_profile_time(samples, _profile_estimator(profile))
    else:
        return None
    if not math.isfinite(value) or value < 0.0:
        raise ReplayValidationError(
            f"{context}: P2P post offset must be finite and non-negative, "
            f"got {value!r}"
        )
    return value


def _profile_for_rank(
    node: TrainingDAGNode, rank: int, profile_mode: str
) -> Any:
    profiles = get_node_profile_times(node, profile_mode)
    if not isinstance(profiles, Mapping):
        raise ReplayValidationError(
            f"node {node.uid!r} has no rank-keyed {profile_mode} profile mapping"
        )
    if rank in profiles:
        return profiles[rank]
    # Serialized artifacts may have passed through JSON without using the
    # ProfiledTrainingPlan loader yet.
    if str(rank) in profiles:
        return profiles[str(rank)]
    raise ReplayValidationError(
        f"missing profile for node {node.uid!r} on global rank {rank} "
        f"in mode {profile_mode!r}"
    )


def _optional_profile_for_rank(
    node: TrainingDAGNode, rank: int, profile_mode: str
) -> Any | None:
    profiles = get_node_profile_times(node, profile_mode)
    if not isinstance(profiles, Mapping):
        return None
    if rank in profiles:
        return profiles[rank]
    return profiles.get(str(rank))


def _profile_iteration_ids_for_communication(
    profile: Any, *, context: str
) -> tuple[int, ...]:
    if isinstance(profile, Mapping):
        samples = tuple(profile.get("samples_us", ()))
        raw_ids = tuple(profile.get("sample_iteration_ids", ()))
    else:
        samples = tuple(getattr(profile, "samples_us", ()))
        raw_ids = tuple(getattr(profile, "sample_iteration_ids", ()))
    if not samples:
        raise ReplayValidationError(
            f"{context}: aligned communication profile has no raw samples"
        )
    return tuple(int(value) for value in (raw_ids or range(len(samples))))


def _profile_estimator(profile: Any) -> str:
    if isinstance(profile, Mapping):
        return str(profile.get("estimator", "median"))
    return str(getattr(profile, "estimator", "median"))


def _profile_timing_source(profile: Any) -> str:
    if isinstance(profile, Mapping):
        return str(profile.get("timing_source", ""))
    return str(getattr(profile, "timing_source", ""))


def _communication_observed_services(
    members: Sequence[_InstanceRecord],
    *,
    communication_id: str,
    sample_iteration_id: int | None,
    pairwise_alignment_uncertainty_us: float,
) -> dict[NodeInstance, float]:
    """Derive wait-free endpoint service from aligned profile occurrences.

    Collectives become serviceable after every participant's GPU activity has
    arrived. A compound P2P send can complete after its own first call and GPU
    readiness without waiting for the peer. Receive data becomes serviceable
    only after both first calls and both endpoints' GPU readiness. Aggregate
    envelopes may contain additional internal P2P calls.
    """

    is_p2p = all(member.node.node_kind in _P2P_KINDS for member in members)
    if sample_iteration_id is None:
        if any(member.profile is None for member in members):
            raise ReplayValidationError(
                f"communication {communication_id!r} has no endpoint profile"
            )
        endpoint_has_samples = tuple(
            bool(
                member.profile.get("samples_us", ())
                if isinstance(member.profile, Mapping)
                else getattr(member.profile, "samples_us", ())
            )
            for member in members
        )
        if is_p2p and not any(endpoint_has_samples):
            # Duration-only test and legacy profiles have no observed offsets;
            # treating every endpoint start as zero applies the same service
            # formula without inventing cross-actor timing data.
            iteration_ids: tuple[int | None, ...] = (None,)
        else:
            endpoint_ids = {
                _profile_iteration_ids_for_communication(
                    member.profile,
                    context=(
                        f"communication {communication_id!r} rank "
                        f"{member.key.global_rank} uid={member.key.uid!r}"
                    ),
                )
                for member in members
            }
            if len(endpoint_ids) != 1:
                raise ReplayValidationError(
                    f"communication {communication_id!r} endpoint iteration IDs are "
                    f"not aligned: {sorted(endpoint_ids)!r}"
                )
            iteration_ids = next(iter(endpoint_ids))
    else:
        iteration_ids = (int(sample_iteration_id),)

    service_samples: dict[NodeInstance, list[float]] = {
        member.key: [] for member in members
    }
    for iteration_id in iteration_ids:
        durations = {
            member.key: _profile_duration_us(
                member.profile,
                context=(
                    f"communication {communication_id!r} rank "
                    f"{member.key.global_rank} uid={member.key.uid!r}"
                ),
                sample_iteration_id=iteration_id,
            )
            for member in members
        }
        zero_activity_states = {duration == 0.0 for duration in durations.values()}
        permits_zero_activity_endpoints = all(
            member.node.node_kind in _A2A_COLLECTIVE_KINDS
            for member in members
        ) or is_p2p
        if len(zero_activity_states) > 1 and not permits_zero_activity_endpoints:
            raise ReplayValidationError(
                f"communication {communication_id!r} iteration {iteration_id} "
                "mixes zero-activity and nonzero-activity endpoint profiles"
            )
        if zero_activity_states == {True}:
            for member in members:
                service_samples[member.key].append(0.0)
            continue

        starts = {
            member.key: _profile_start_offset_us(
                member.profile,
                context=(
                    f"communication {communication_id!r} rank "
                    f"{member.key.global_rank} uid={member.key.uid!r}"
                ),
                sample_iteration_id=iteration_id,
            )
            for member in members
        }
        if is_p2p:
            posts = {
                member.key: _profile_p2p_post_offset_us(
                    member.profile,
                    context=(
                        f"P2P {communication_id!r} rank "
                        f"{member.key.global_rank} uid={member.key.uid!r}"
                    ),
                    sample_iteration_id=iteration_id,
                )
                for member in members
            }
            permits_synthetic_zero_posts = all(
                _profile_timing_source(member.profile) != "kineto_cupti"
                for member in members
            )
            if all(value is None for value in posts.values()) and (
                iteration_id is None or permits_synthetic_zero_posts
            ):
                # Duration-only and synthetic programmatic profiles have no
                # real host timestamps. Treat both operations as posted at
                # zero. A durable Kineto profile fails closed below because
                # replaying it without the new post markers is ambiguous.
                observed_match_us = 0.0
            elif any(value is None for value in posts.values()):
                missing = sorted(
                    member.key
                    for member in members
                    if posts[member.key] is None
                )
                raise ReplayValidationError(
                    f"P2P {communication_id!r} has no aligned host post "
                    f"timestamps for endpoints {missing!r}; re-profile the plan"
                )
            else:
                observed_match_us = max(
                    float(value) for value in posts.values() if value is not None
                )
            send_members = [
                member for member in members
                if member.node.node_kind == "SEND_COMM"
            ]
            if len(send_members) != 1:
                raise ReplayValidationError(
                    f"P2P {communication_id!r} must have exactly one send endpoint"
                )
            observed_send_ready_us = starts[send_members[0].key]
        else:
            observed_match_us = max(starts.values())
            observed_send_ready_us = 0.0
        for member in members:
            if durations[member.key] == 0.0:
                service_samples[member.key].append(0.0)
                continue
            observed_end_us = starts[member.key] + durations[member.key]
            service_origin_us = observed_match_us
            if is_p2p and member.node.node_kind == "SEND_COMM":
                local_post_us = posts[member.key]
                service_origin_us = max(
                    starts[member.key],
                    0.0 if local_post_us is None else float(local_post_us),
                )
            elif is_p2p:
                service_origin_us = max(
                    service_origin_us,
                    starts[member.key],
                    observed_send_ready_us,
                )
            raw_service_us = observed_end_us - service_origin_us
            if raw_service_us < -pairwise_alignment_uncertainty_us:
                raise ReplayValidationError(
                    f"communication {communication_id!r} iteration {iteration_id} "
                    "has an inconsistent aligned endpoint at rank "
                    f"{member.key.global_rank}, uid={member.key.uid!r}: "
                    f"start={starts[member.key]:.3f} us, "
                    f"end={observed_end_us:.3f} us, service_origin="
                    f"{service_origin_us:.3f} us, pairwise_uncertainty="
                    f"{pairwise_alignment_uncertainty_us:.3f} us"
                )
            service_samples[member.key].append(max(0.0, raw_service_us))

    if sample_iteration_id is not None:
        return {key: values[0] for key, values in service_samples.items()}
    return {
        member.key: estimate_profile_time(
            service_samples[member.key], _profile_estimator(member.profile)
        )
        for member in members
    }


def _mapping_int(mapping: Mapping[str, Any] | None, *names: str) -> int | None:
    if not mapping:
        return None
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return int(value)
    return None


def _extract_inputs(
    plan_or_dags: Any,
) -> tuple[list[TrainingDAG], Mapping[str, Any]]:
    if hasattr(plan_or_dags, "per_pp_dags"):
        dags = list(plan_or_dags.per_pp_dags)
    elif hasattr(plan_or_dags, "per_pp_training_dags"):
        dags = list(plan_or_dags.per_pp_training_dags)
    elif isinstance(plan_or_dags, Sequence) and not isinstance(
        plan_or_dags, (str, bytes, bytearray)
    ):
        dags = list(plan_or_dags)
    else:
        raise TypeError(
            "replay_training_step expects a ProfiledTrainingPlan-like object "
            "or a sequence of TrainingDAGs"
        )
    if not dags:
        raise ReplayValidationError("at least one per-PP TrainingDAG is required")
    if any(not isinstance(dag, TrainingDAG) for dag in dags):
        raise TypeError("every per-PP DAG must be a TrainingDAG")

    topology = getattr(plan_or_dags, "topology", None)
    if not isinstance(topology, Mapping):
        topology = getattr(plan_or_dags, "schedule_info", None)
    if not isinstance(topology, Mapping):
        topology = {}

    return dags, topology


def _dag_device_ranks(dag: TrainingDAG) -> tuple[int, ...] | None:
    layouts: set[tuple[int, ...]] = set()
    for node in dag.nodes.values():
        if node.device is None:
            continue
        ranks = tuple(int(rank) for rank in node.device)
        if not ranks:
            raise ReplayValidationError(f"node {node.uid!r} has an empty device list")
        if len(set(ranks)) != len(ranks):
            raise ReplayValidationError(
                f"node {node.uid!r} has duplicate ranks in device={node.device!r}"
            )
        layouts.add(ranks)
    if not layouts:
        return None
    if len(layouts) != 1:
        raise ReplayValidationError(
            "a per-PP DAG must have one consistent ordered device list; got "
            f"{sorted(layouts)!r}"
        )
    return next(iter(layouts))


def _profile_ranks(dag: TrainingDAG, profile_mode: str) -> tuple[int, ...]:
    ranks: set[int] = set()
    for node in dag.nodes.values():
        profiles = get_node_profile_times(node, profile_mode)
        if isinstance(profiles, Mapping):
            for rank in profiles:
                try:
                    ranks.add(int(rank))
                except (TypeError, ValueError) as exc:
                    raise ReplayValidationError(
                        f"node {node.uid!r} has non-integer profile rank {rank!r}"
                    ) from exc
    prologue = get_prologue_profile_times(dag, profile_mode)
    if isinstance(prologue, Mapping):
        for rank in prologue:
            ranks.add(int(rank))
    return tuple(sorted(ranks))


def _explicit_ranks_by_pp(
    topology: Mapping[str, Any], num_pp_dags: int
) -> tuple[tuple[int, ...], ...] | None:
    raw = topology.get("ranks_by_pp")
    if raw is None:
        raw = topology.get("pp_rank_members")
    if raw is None:
        raw_actor_groups = topology.get("actor_groups")
        if raw_actor_groups is not None:
            raw = [
                group.get("devices") if isinstance(group, Mapping) else group
                for group in raw_actor_groups
            ]
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        groups = [raw.get(i, raw.get(str(i))) for i in range(num_pp_dags)]
    else:
        groups = list(raw)
    if len(groups) != num_pp_dags or any(group is None for group in groups):
        raise ReplayValidationError(
            f"topology ranks_by_pp must contain {num_pp_dags} groups"
        )
    return tuple(tuple(int(rank) for rank in group) for group in groups)


def _build_rank_layout(
    dags: Sequence[TrainingDAG],
    topology: Mapping[str, Any],
    *,
    pp_degree: int | None,
    profile_mode: str,
) -> _RankLayout:
    topology_pp = _mapping_int(topology, "pp_degree")
    topology_world = _mapping_int(topology, "world_size")
    pp_degree = int(pp_degree if pp_degree is not None else topology_pp or len(dags))
    if pp_degree != len(dags):
        raise ReplayValidationError(
            f"pp_degree={pp_degree} does not match {len(dags)} per-PP DAGs"
        )

    explicit = _explicit_ranks_by_pp(topology, len(dags))
    device_layouts = tuple(_dag_device_ranks(dag) for dag in dags)
    if explicit is not None:
        mismatches = [
            (index, explicit[index], device_ranks)
            for index, device_ranks in enumerate(device_layouts)
            if device_ranks is not None and explicit[index] != device_ranks
        ]
        if mismatches:
            raise ReplayValidationError(
                "DAG device sets do not match explicit ActorGroups: "
                + ", ".join(
                    f"group {index}: topology={expected!r}, DAG={actual!r}"
                    for index, expected, actual in mismatches
                )
            )
        ranks_by_pp = explicit
    elif all(ranks is not None for ranks in device_layouts):
        ranks_by_pp = tuple(ranks for ranks in device_layouts if ranks is not None)
    else:
        profile_layouts = tuple(_profile_ranks(dag, profile_mode) for dag in dags)
        if all(profile_layouts):
            ranks_by_pp = profile_layouts
        else:
            # Bare DAG input without placement or profile metadata represents
            # one actor per local DAG.  Replicated/sharded execution must make
            # its ActorGroups explicit; a rectangular world size is not enough
            # to recover schedule-defined group membership.
            ranks_by_pp = tuple((pp_index,) for pp_index in range(pp_degree))

    lengths = {len(ranks) for ranks in ranks_by_pp}
    if not lengths or 0 in lengths:
        raise ReplayValidationError(
            f"all ActorGroup rank lists must be non-empty, got {ranks_by_pp!r}"
        )
    flat = [rank for ranks in ranks_by_pp for rank in ranks]
    if len(flat) != len(set(flat)):
        raise ReplayValidationError(f"global ranks occur in multiple PP groups: {ranks_by_pp!r}")
    if any(rank < 0 for rank in flat):
        raise ReplayValidationError(f"global ranks must be non-negative: {ranks_by_pp!r}")
    if topology_world is not None and int(topology_world) != len(flat):
        raise ReplayValidationError(
            f"world_size={topology_world} does not match {len(flat)} replayed ranks"
        )

    lane_by_rank: dict[int, int] = {}
    pp_by_rank: dict[int, int] = {}
    for pp_index, ranks in enumerate(ranks_by_pp):
        for lane, rank in enumerate(ranks):
            lane_by_rank[rank] = lane
            pp_by_rank[rank] = pp_index
    return _RankLayout(tuple(ranks_by_pp), lane_by_rank, pp_by_rank)


def _comm_id(node: TrainingDAGNode) -> str:
    meta = node.node_meta
    if "comm_id" in meta:
        return _identity_text(meta["comm_id"])
    if node.node_kind in _P2P_KINDS:
        from_uid = meta.get("from_uid")
        to_uid = meta.get("to_uid")
        if from_uid is None or to_uid is None:
            raise ReplayValidationError(
                f"P2P node {node.uid!r} needs comm_id or from_uid/to_uid metadata"
            )
        return f"{from_uid}->{to_uid}"
    # Communication UIDs are globally assigned before the DAG is split, so
    # they are a safe compatibility identity for existing finalized DAGs.
    return node.uid


def _add_vertex_edge(vertices: dict[str, _Vertex], src: str, dst: str) -> None:
    if src == dst:
        return
    vertices[src].succs.add(dst)
    vertices[dst].preds.add(src)


def _merge_prologue_profiles(
    dags: Sequence[TrainingDAG],
    central: Mapping[int, Any] | None,
    profile_mode: str,
) -> dict[int, Any]:
    profiles: dict[int, Any] = {}
    sources: list[Mapping[Any, Any]] = []
    if central:
        sources.append(central)
    for dag in dags:
        candidate = get_prologue_profile_times(dag, profile_mode)
        if candidate:
            if not isinstance(candidate, Mapping):
                raise ReplayValidationError("DAG prologue_profile_times must be a mapping")
            sources.append(candidate)
    for source in sources:
        for raw_rank, profile in source.items():
            rank = int(raw_rank)
            if rank in profiles:
                old = _profile_duration_us(profiles[rank], context=f"prologue rank {rank}")
                new = _profile_duration_us(profile, context=f"prologue rank {rank}")
                if old != new:
                    raise ReplayValidationError(
                        f"conflicting prologue profiles for rank {rank}: {old} vs {new} us"
                    )
            profiles[rank] = profile
    return profiles


def _expected_participants(node: TrainingDAGNode) -> tuple[int, ...] | None:
    raw = node.node_meta.get("participants")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ReplayValidationError(
            f"node {node.uid!r} participants must be a rank sequence"
        )
    result = tuple(int(rank) for rank in raw)
    if len(result) != len(set(result)):
        raise ReplayValidationError(f"node {node.uid!r} has duplicate participants")
    return result


def _validate_p2p_vertex(
    vertex: _Vertex,
) -> None:
    sends = [member for member in vertex.members if member.node.node_kind == "SEND_COMM"]
    recvs = [member for member in vertex.members if member.node.node_kind == "RECV_COMM"]
    if len(sends) != 1 or len(recvs) != 1:
        descriptions = [
            f"rank {member.key.global_rank}:{member.node.node_kind}:{member.node.uid}"
            for member in vertex.members
        ]
        raise ReplayValidationError(
            f"P2P transaction {vertex.communication_id!r} must have exactly one SEND and "
            f"one RECV endpoint; got {descriptions!r}"
        )
    send, recv = sends[0], recvs[0]
    if send.actor_group_local_rank != recv.actor_group_local_rank:
        raise ReplayValidationError(
            f"P2P transaction {vertex.communication_id!r} crosses ActorGroup-local "
            f"ranks: {send.actor_group_local_rank} vs "
            f"{recv.actor_group_local_rank}"
        )
    transaction_ranks = {
        send.key.global_rank,
        recv.key.global_rank,
    }
    # Piper inserts one SEND/RECV node per pair of ActorGroups, then the
    # replay expands that node once per same-position ActorGroup member.
    # Consequently P2P
    # ``participants`` describes the full pair of device groups (for example
    # ranks 0,1,2,3), while this vertex is one ActorGroup-local transaction (for
    # example ranks 0,1).  Hand-built/legacy DAGs may instead store only the
    # two transaction ranks, so accept both precise representations while
    # still rejecting unrelated or incomplete participant sets.
    device_group_ranks = {
        int(rank)
        for member in (send, recv)
        for rank in (member.node.device or ())
    }
    valid_participant_sets = {frozenset(transaction_ranks)}
    if device_group_ranks:
        valid_participant_sets.add(frozenset(device_group_ranks))

    for endpoint, counterpart in ((send, recv), (recv, send)):
        meta = endpoint.node.node_meta
        peer_global = meta.get("peer_global_rank", meta.get("peer_rank"))
        if peer_global is not None and int(peer_global) != counterpart.key.global_rank:
            raise ReplayValidationError(
                f"P2P endpoint {endpoint.key} expects peer rank {peer_global}, got "
                f"{counterpart.key.global_rank}"
            )
        peer_pp = meta.get("peer_pp_rank")
        if peer_pp is not None and int(peer_pp) != counterpart.pp_index:
            raise ReplayValidationError(
                f"P2P endpoint {endpoint.key} expects peer PP rank {peer_pp}, got "
                f"{counterpart.pp_index}"
            )
        participants = _expected_participants(endpoint.node)
        if (
            participants is not None
            and frozenset(participants) not in valid_participant_sets
        ):
            raise ReplayValidationError(
                f"P2P endpoint {endpoint.key} participants={participants!r} do not match "
                f"transaction ranks {(send.key.global_rank, recv.key.global_rank)!r} "
                f"or endpoint device-group ranks {tuple(sorted(device_group_ranks))!r}"
            )
    groups = {_p2p_group(member) for member in vertex.members}
    if len(groups) != 1:
        raise ReplayValidationError(
            f"P2P transaction {vertex.communication_id!r} endpoints disagree on "
            f"comm_group_kind: {sorted(groups)!r}"
        )


def _p2p_group(record: _InstanceRecord) -> str:
    explicit = record.node.node_meta.get("comm_group_kind")
    if explicit is not None:
        group = str(explicit)
        if not group:
            raise ReplayValidationError(
                f"P2P endpoint {record.key} has an empty comm_group_kind"
            )
        return group
    peer_pp = record.node.node_meta.get("peer_pp_rank")
    if peer_pp is None:
        # Compatibility for hand-built/older DAGs.  Such DAGs cannot express
        # opposite directional groups, so conservatively preserve one FIFO.
        return "pp_p2p"
    peer_pp = int(peer_pp)
    if record.node.node_kind == "SEND_COMM":
        source_pp, destination_pp = record.pp_index, peer_pp
    else:
        source_pp, destination_pp = peer_pp, record.pp_index
    return "pp_lo_hi" if source_pp < destination_pp else "pp_hi_lo"


def _representative(vertex: _Vertex, preferred_rank: int | None = None) -> NodeInstance:
    members = sorted((member.key for member in vertex.members))
    if preferred_rank is not None:
        for member in members:
            if member.global_rank == preferred_rank:
                return member
    return members[0]


def _replay_vertices(vertices: dict[str, _Vertex]) -> tuple[str, ...]:
    indegree = {vertex_id: len(vertex.preds) for vertex_id, vertex in vertices.items()}
    ready = [vertex_id for vertex_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        vertex_id = heapq.heappop(ready)
        vertex = vertices[vertex_id]
        order.append(vertex_id)
        if vertex.preds:
            best_pred = min(
                vertex.preds,
                key=lambda pred: (-vertices[pred].end_us, pred),
            )
            vertex.critical_pred = best_pred
            vertex.start_us = vertices[best_pred].end_us
        if vertex.release_us > vertex.start_us:
            vertex.start_us = vertex.release_us
            # A host/GPU release constraint, rather than a replayed vertex,
            # determines this start.  Do not claim a shorter dependency-only
            # chain is critical through the release gap.
            vertex.critical_pred = None
        if vertex.endpoint_arrival_vertex_id is not None:
            arrival = vertices[vertex.endpoint_arrival_vertex_id]
            observed_service_us = vertex.endpoint_observed_service_us
            if observed_service_us is None:
                raise ReplayValidationError(
                    f"endpoint completion {vertex_id!r} has no communication timing"
                )
            vertex.rendezvous_wait_us = max(
                0.0, vertex.start_us - arrival.end_us
            )
            vertex.duration_us = observed_service_us
        vertex.end_us = vertex.start_us + vertex.duration_us
        for succ in sorted(vertex.succs):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                heapq.heappush(ready, succ)
    if len(order) != len(vertices):
        cyclic = sorted(vertex_id for vertex_id, degree in indegree.items() if degree > 0)
        raise ReplayValidationError(
            "communication replay graph contains a cycle (often inconsistent collective "
            f"ordering across ranks): {cyclic!r}"
        )
    return tuple(order)


def replay_training_step(
    plan_or_dags: Any,
    *,
    pp_degree: int | None = None,
    prologue_profile_times: Mapping[int, Any] | None = None,
    profile_mode: str = ISOLATED_PROFILE_MODE,
    sample_iteration_id: int | None = None,
    strict: bool = True,
    _normal_p2p_post_profiles: Mapping[NodeInstance, Any] | None = None,
) -> ReplayResult:
    """Replay one training step using the selected timing profile.

    Args:
        plan_or_dags: A ``ProfiledTrainingPlan`` (or compatible object) or a
            sequence of finalized per-PP ``TrainingDAG`` objects.
        pp_degree: Optional physical pipeline degree for bare DAG input.
        prologue_profile_times: Optional rank-keyed profiles for gradient-buffer
            initialization.  DAG-local profiles are merged with this mapping.
        strict: Reject surplus rank profiles and incomplete prologue profile
            sets.  Missing node profiles and malformed communication are always
            errors because no meaningful duration can be inferred for them.

    Returns:
        A deterministic timeline, critical path, and lower-bound step time in
        microseconds.
    """

    profile_mode = normalize_profile_mode(profile_mode)
    dags, topology = _extract_inputs(plan_or_dags)
    unsupported_routed_nodes = sorted(
        (pp_index, str(node.uid), str(node.node_kind))
        for pp_index, dag in enumerate(dags)
        for node in dag.nodes.values()
        if node.node_kind in _UNSUPPORTED_ROUTED_COMM_KINDS
    )
    if unsupported_routed_nodes:
        descriptions = ", ".join(
            f"PP {pp_index} {uid!r} ({node_kind})"
            for pp_index, uid, node_kind in unsupported_routed_nodes
        )
        raise ReplayValidationError(
            "M-to-N/N-to-M communication replay is unsupported because "
            "profiles contain only aggregate node service times, not the "
            "service times of individual peer sends and receives; nodes: "
            f"{descriptions}"
        )
    profile_metadata = getattr(plan_or_dags, "profile_metadata", {}) or {}
    alignment = (
        profile_metadata.get("timeline_alignment", {})
        if isinstance(profile_metadata, Mapping)
        else {}
    )
    aligned_modes = alignment.get("modes", {}) if isinstance(alignment, Mapping) else {}
    cross_actor_offsets_aligned = profile_mode in aligned_modes
    has_rendezvous_collectives = any(
        node.node_kind in _COLLECTIVE_KINDS
        for dag in dags
        for node in dag.nodes.values()
    )
    if has_rendezvous_collectives and not cross_actor_offsets_aligned:
        raise ReplayValidationError(
            f"profile mode {profile_mode!r} contains rendezvous collectives but "
            "has no cross-actor timeline alignment; re-profile this plan with "
            "aligned CUPTI iteration anchors"
        )
    mode_alignment = (
        aligned_modes.get(profile_mode, {})
        if isinstance(aligned_modes, Mapping)
        else {}
    )
    raw_pairwise_uncertainty = (
        mode_alignment.get("max_pairwise_uncertainty_us", 0.0)
        if isinstance(mode_alignment, Mapping)
        else 0.0
    )
    pairwise_alignment_uncertainty_us = float(
        0.0 if raw_pairwise_uncertainty is None else raw_pairwise_uncertainty
    )
    if (
        not math.isfinite(pairwise_alignment_uncertainty_us)
        or pairwise_alignment_uncertainty_us < 0.0
    ):
        raise ReplayValidationError(
            f"profile mode {profile_mode!r} has invalid pairwise timeline "
            f"uncertainty {raw_pairwise_uncertainty!r}"
        )
    empty_dags = [index for index, dag in enumerate(dags) if not dag.nodes]
    if empty_dags:
        raise ReplayValidationError(
            f"per-PP TrainingDAGs must be non-empty; empty indices={empty_dags!r}"
        )
    layout = _build_rank_layout(
        dags,
        topology,
        pp_degree=pp_degree,
        profile_mode=profile_mode,
    )
    warnings: list[str] = []

    merged_prologue = _merge_prologue_profiles(dags, None, profile_mode)
    if prologue_profile_times:
        supplied = _merge_prologue_profiles(
            [], prologue_profile_times, profile_mode
        )
        for rank, profile in supplied.items():
            if rank in merged_prologue:
                old = _profile_duration_us(
                    merged_prologue[rank], context=f"prologue rank {rank}"
                )
                new = _profile_duration_us(profile, context=f"prologue rank {rank}")
                if old != new:
                    raise ReplayValidationError(
                        f"conflicting prologue profiles for rank {rank}: {old} vs {new} us"
                    )
            merged_prologue[rank] = profile

    rank_set = set(layout.ranks)
    if merged_prologue:
        missing = sorted(rank_set - set(merged_prologue))
        extra = sorted(set(merged_prologue) - rank_set)
        if missing:
            message = f"prologue profiles are missing replayed ranks {missing!r}"
            if strict:
                raise ReplayValidationError(message)
            warnings.append(message)
        if extra:
            message = f"prologue profiles contain non-replayed ranks {extra!r}"
            if strict:
                raise ReplayValidationError(message)
            warnings.append(message)

    instances: dict[NodeInstance, _InstanceRecord] = {}
    dag_plans: list[tuple[PlannedOperation, ...]] = []
    for pp_index, dag in enumerate(dags):
        if _PROLOGUE_UID in dag.nodes:
            raise ReplayValidationError(
                f"node UID {_PROLOGUE_UID!r} is reserved by the replay"
            )
        try:
            plan = tuple(build_execution_plan(dag))
        except ValueError as exc:
            raise ReplayValidationError(str(exc)) from exc
        dag_plans.append(plan)
        expected_ranks = set(layout.ranks_by_pp[pp_index])
        for operation in plan:
            uid = operation.uid
            node = dag.nodes[uid]
            if not isinstance(node.stream, str) or not node.stream:
                raise ReplayValidationError(
                    f"node {uid!r} must have a non-empty logical stream name"
                )
            profiles = get_node_profile_times(node, profile_mode)
            if strict and isinstance(profiles, Mapping):
                try:
                    attached_ranks = {int(rank) for rank in profiles}
                except (TypeError, ValueError) as exc:
                    raise ReplayValidationError(
                        f"node {uid!r} has a non-integer profile rank"
                    ) from exc
                extra = sorted(attached_ranks - expected_ranks)
                if extra:
                    raise ReplayValidationError(
                        f"node {uid!r} in PP DAG {pp_index} has profiles for "
                        f"non-member ranks {extra!r}"
                    )
            for rank in layout.ranks_by_pp[pp_index]:
                key = NodeInstance(rank, uid)
                if key in instances:
                    raise ReplayValidationError(f"duplicate node instance {key}")
                if node.node_kind == "ORDER_DUMMY":
                    profile = None
                    profile_duration = 0.0
                    profile_start_offset = 0.0
                    p2p_post_offset = None
                else:
                    profile = _profile_for_rank(node, rank, profile_mode)
                    profile_duration = _profile_duration_us(
                        profile,
                        context=f"node {uid!r} rank {rank}",
                        sample_iteration_id=sample_iteration_id,
                    )
                    observed_profile_start_offset = _profile_start_offset_us(
                        profile,
                        context=f"node {uid!r} rank {rank}",
                        sample_iteration_id=sample_iteration_id,
                    )
                    # Both modes replay the same dependency and physical
                    # stream ordering.  Observed starts are diagnostics: they
                    # may include host latency or resource queueing, which
                    # must not make normal and isolated counterfactuals use
                    # different release policies.
                    profile_start_offset = 0.0
                    if cross_actor_offsets_aligned and not tuple(
                        getattr(profile, "start_offset_samples_us", ())
                    ):
                        raise ReplayValidationError(
                            f"aligned profile mode {profile_mode!r} is missing "
                            f"start offsets for node {uid!r} rank {rank}"
                        )
                    p2p_post_offset = None
                    if node.node_kind in _P2P_KINDS:
                        # Host dispatch is the intended normal schedule in
                        # both counterfactual replays.  Isolated capture has a
                        # different profiling issue order, so it is used only
                        # to derive service, never to schedule P2P posts when a
                        # normal post profile is available.
                        normal_post_profile = (
                            _normal_p2p_post_profiles.get(key)
                            if _normal_p2p_post_profiles is not None
                            else None
                        )
                        if normal_post_profile is None:
                            normal_post_profile = _optional_profile_for_rank(
                                node, rank, NORMAL_PROFILE_MODE
                            )
                        post_profile = (
                            normal_post_profile
                            if normal_post_profile is not None
                            and _profile_p2p_post_offset_us(
                                normal_post_profile,
                                context=(
                                    f"normal P2P post for node {uid!r} rank {rank}"
                                ),
                                sample_iteration_id=sample_iteration_id,
                            )
                            is not None
                            else profile
                        )
                        p2p_post_offset = _profile_p2p_post_offset_us(
                            post_profile,
                            context=f"P2P post for node {uid!r} rank {rank}",
                            sample_iteration_id=sample_iteration_id,
                        )
                        post_profile_mode = (
                            NORMAL_PROFILE_MODE
                            if post_profile is normal_post_profile
                            else profile_mode
                        )
                        if (
                            p2p_post_offset is not None
                            and post_profile_mode not in aligned_modes
                        ):
                            raise ReplayValidationError(
                                f"P2P post timestamps for node {uid!r} rank "
                                f"{rank} in mode {post_profile_mode!r} are not "
                                "cross-actor aligned"
                            )
                if node.node_kind == "ORDER_DUMMY":
                    observed_profile_start_offset = 0.0
                instances[key] = _InstanceRecord(
                    key=key,
                    pp_index=pp_index,
                    actor_group_local_rank=layout.lane_by_rank[rank],
                    node=node,
                    profile=profile,
                    profile_duration_us=profile_duration,
                    profile_start_offset_us=profile_start_offset,
                    observed_profile_start_offset_us=(
                        observed_profile_start_offset
                    ),
                    p2p_post_offset_us=p2p_post_offset,
                )

    comm_groups: dict[tuple[Any, ...], list[_InstanceRecord]] = defaultdict(list)
    for record in instances.values():
        kind = record.node.node_kind
        if kind in _P2P_KINDS:
            comm_groups[
                (
                    "p2p",
                    _comm_id(record.node),
                    record.actor_group_local_rank,
                )
            ].append(record)
        elif kind in _COLLECTIVE_KINDS:
            comm_groups[("collective", kind, _comm_id(record.node))].append(record)

    vertices: dict[str, _Vertex] = {}
    # Completion vertices represent when rank-local consumers may proceed.
    # Start vertices represent when an operation has satisfied its local
    # dependencies.  They differ for communication endpoints because every
    # arrival must first join a shared rendezvous.
    instance_vertex: dict[NodeInstance, str] = {}
    instance_start_vertex: dict[NodeInstance, str] = {}

    # Communication vertices are constructed first so local dependency edges
    # can target endpoint arrivals while outgoing edges originate at distinct
    # endpoint completions.
    for group_key in sorted(comm_groups, key=repr):
        members = comm_groups[group_key]
        communication_id = ":".join(_identity_text(part) for part in group_key[1:])
        rendezvous_id = f"comm:{group_key[0]}:{communication_id}:rendezvous"
        transaction = _Vertex(
            vertex_id=rendezvous_id,
            members=list(members),
            duration_us=0.0,
            communication_id=communication_id,
            emit_timeline=False,
        )
        if rendezvous_id in vertices:
            raise ReplayValidationError(
                f"duplicate communication identity {rendezvous_id!r}"
            )
        vertices[rendezvous_id] = transaction

        if group_key[0] == "p2p":
            _validate_p2p_vertex(transaction)
        else:
            actual = {member.key.global_rank for member in members}
            expected_sets = {
                expected
                for member in members
                for expected in [_expected_participants(member.node)]
                if expected is not None
            }
            if len(expected_sets) > 1:
                raise ReplayValidationError(
                    f"collective {communication_id!r} endpoints disagree on participants: "
                    f"{sorted(expected_sets)!r}"
                )
            if group_key[1] in _A2A_COLLECTIVE_KINDS and not expected_sets:
                raise ReplayValidationError(
                    f"A2A collective {communication_id!r} has no declared participants"
                )
            if expected_sets and actual != set(next(iter(expected_sets))):
                raise ReplayValidationError(
                    f"collective {communication_id!r} has ranks {sorted(actual)!r}, expected "
                    f"{sorted(next(iter(expected_sets)))!r}"
                )

        if group_key[0] == "p2p" and not cross_actor_offsets_aligned:
            has_observed_offsets = any(
                bool(
                    member.profile.get("start_offset_samples_us", ())
                    if isinstance(member.profile, Mapping)
                    else getattr(member.profile, "start_offset_samples_us", ())
                )
                for member in members
            )
            if has_observed_offsets:
                raise ReplayValidationError(
                    f"profile mode {profile_mode!r} contains P2P communication "
                    "with endpoint start offsets but has no cross-actor timeline "
                    "alignment"
                )
        observed_service_by_endpoint = _communication_observed_services(
            members,
            communication_id=communication_id,
            sample_iteration_id=sample_iteration_id,
            pairwise_alignment_uncertainty_us=(
                pairwise_alignment_uncertainty_us
            ),
        )

        arrival_ids: dict[NodeInstance, str] = {}
        post_ids: dict[NodeInstance, str] = {}
        completion_ids: dict[NodeInstance, str] = {}
        for member in members:
            endpoint_suffix = (
                f"{member.key.global_rank}:{member.key.uid}"
            )
            arrival_id = (
                f"comm:{group_key[0]}:{communication_id}:arrival:{endpoint_suffix}"
            )
            completion_id = (
                f"comm:{group_key[0]}:{communication_id}:completion:{endpoint_suffix}"
            )
            post_id = (
                f"comm:p2p:{communication_id}:post:{endpoint_suffix}"
            )
            if (
                arrival_id in vertices
                or completion_id in vertices
                or (group_key[0] == "p2p" and post_id in vertices)
            ):
                raise ReplayValidationError(
                    f"duplicate communication endpoint {member.key!r}"
                )
            vertices[arrival_id] = _Vertex(
                vertex_id=arrival_id,
                members=[member],
                duration_us=0.0,
                release_us=member.profile_start_offset_us,
                communication_id=communication_id,
                emit_timeline=False,
            )
            if group_key[0] == "p2p":
                vertices[post_id] = _Vertex(
                    vertex_id=post_id,
                    members=[member],
                    duration_us=0.0,
                    release_us=float(member.p2p_post_offset_us or 0.0),
                    communication_id=communication_id,
                    emit_timeline=False,
                )
                post_ids[member.key] = post_id
            vertices[completion_id] = _Vertex(
                vertex_id=completion_id,
                members=[member],
                # Calibrated profiles provide a fixed tail derived from the
                # endpoint's observed service origin.
                duration_us=0.0,
                communication_id=communication_id,
                endpoint_arrival_vertex_id=arrival_id,
                endpoint_observed_service_us=observed_service_by_endpoint[
                    member.key
                ],
            )
            arrival_ids[member.key] = arrival_id
            completion_ids[member.key] = completion_id
            instance_start_vertex[member.key] = arrival_id
            instance_vertex[member.key] = completion_id

        if group_key[0] == "p2p":
            send_member = next(
                member
                for member in members
                if member.node.node_kind == "SEND_COMM"
            )
            send_arrival_id = arrival_ids[send_member.key]
            for member in members:
                _add_vertex_edge(vertices, post_ids[member.key], rendezvous_id)
            for member in members:
                _add_vertex_edge(
                    vertices,
                    arrival_ids[member.key],
                    completion_ids[member.key],
                )
                if member.node.node_kind == "RECV_COMM":
                    # Delivery requires both calls plus source and destination
                    # readiness. Send completion is local and may precede the
                    # receive call, as ProcessGroupNCCL permits in practice.
                    _add_vertex_edge(
                        vertices,
                        rendezvous_id,
                        completion_ids[member.key],
                    )
                    _add_vertex_edge(
                        vertices,
                        send_arrival_id,
                        completion_ids[member.key],
                    )
                    vertices[
                        completion_ids[member.key]
                    ].endpoint_rendezvous_vertex_id = rendezvous_id
                else:
                    _add_vertex_edge(
                        vertices,
                        post_ids[member.key],
                        completion_ids[member.key],
                    )
        else:
            for member in members:
                _add_vertex_edge(vertices, arrival_ids[member.key], rendezvous_id)
                _add_vertex_edge(
                    vertices,
                    rendezvous_id,
                    completion_ids[member.key],
                )
                vertices[
                    completion_ids[member.key]
                ].endpoint_rendezvous_vertex_id = rendezvous_id

    for record in instances.values():
        if record.key in instance_vertex:
            continue
        vertex_id = f"node:{record.key.global_rank}:{record.key.uid}"
        vertices[vertex_id] = _Vertex(
            vertex_id=vertex_id,
            members=[record],
            duration_us=record.profile_duration_us,
            release_us=record.profile_start_offset_us,
        )
        instance_vertex[record.key] = vertex_id
        instance_start_vertex[record.key] = vertex_id

    # A rank-local synthetic prologue gates all local streams.  Absence means
    # the caller intentionally chose not to model prologue work.
    for rank in sorted(rank_set & set(merged_prologue)):
        key = NodeInstance(rank, _PROLOGUE_UID)
        record = _InstanceRecord(
            key=key,
            pp_index=layout.pp_by_rank[rank],
            actor_group_local_rank=layout.lane_by_rank[rank],
                node=TrainingDAGNode(
                uid=_PROLOGUE_UID,
                node_kind="STEP_PROLOGUE",
                compute_subkind=None,
                tag={},
                device=[rank],
                    stream="__prologue__",
                ),
                profile=merged_prologue[rank],
                profile_duration_us=_profile_duration_us(
                merged_prologue[rank],
                context=f"prologue rank {rank}",
                sample_iteration_id=sample_iteration_id,
            ),
            profile_start_offset_us=0.0,
        )
        vertex_id = f"prologue:{rank}"
        vertices[vertex_id] = _Vertex(
            vertex_id=vertex_id,
            members=[record],
            duration_us=record.profile_duration_us,
        )
        instance_vertex[key] = vertex_id
        instance_start_vertex[key] = vertex_id

    # Lower the same completion dependencies used by DagExecutor.  The shared
    # plan contains every finalized DAG dependency. Directional P2P
    # communicators retain dedicated physical streams, but explicit edges
    # between those streams are preserved.
    for pp_index, dag in enumerate(dags):
        plan = dag_plans[pp_index]
        for rank in layout.ranks_by_pp[pp_index]:
            for operation in plan:
                node = dag.nodes[operation.uid]
                dst = instance_start_vertex[NodeInstance(rank, operation.uid)]
                for predecessor_uid in operation.completion_predecessors:
                    src = instance_vertex[NodeInstance(rank, predecessor_uid)]
                    _add_vertex_edge(vertices, src, dst)

    # Every prologue precedes every real operation on its rank.  Adding all
    # such edges is equivalent to gating the first operation of each stream.
    for rank in sorted(rank_set & set(merged_prologue)):
        prologue_vertex = instance_vertex[NodeInstance(rank, _PROLOGUE_UID)]
        for record in instances.values():
            if record.key.global_rank == rank:
                _add_vertex_edge(
                    vertices,
                    prologue_vertex,
                    instance_start_vertex[record.key],
                )

    replay_order = _replay_vertices(vertices)

    timeline: list[TimelineEvent] = []
    per_rank: dict[int, float] = {rank: 0.0 for rank in layout.ranks}
    for vertex_id in replay_order:
        vertex = vertices[vertex_id]
        if not vertex.emit_timeline:
            continue
        for member in sorted(vertex.members, key=lambda item: item.key):
            critical_predecessor = None
            if vertex.critical_pred is not None:
                critical_predecessor = _representative(
                    vertices[vertex.critical_pred], member.key.global_rank
                )
            event = TimelineEvent(
                operation=member.key,
                node_kind=member.node.node_kind,
                stream=member.node.stream,
                start_us=vertex.start_us,
                end_us=vertex.end_us,
                profile_duration_us=member.profile_duration_us,
                effective_duration_us=vertex.duration_us,
                profile_start_offset_us=member.profile_start_offset_us,
                observed_profile_start_offset_us=(
                    member.observed_profile_start_offset_us
                ),
                arrival_start_us=vertices[
                    instance_start_vertex[member.key]
                ].start_us,
                communication_id=vertex.communication_id,
                critical_predecessor=critical_predecessor,
            )
            timeline.append(event)
            per_rank[member.key.global_rank] = max(
                per_rank[member.key.global_rank], vertex.end_us
            )
    timeline.sort(
        key=lambda event: (
            event.start_us,
            event.end_us,
            event.operation.global_rank,
            event.operation.uid,
        )
    )

    final_vertex = min(
        vertices,
        key=lambda vertex_id: (-vertices[vertex_id].end_us, vertex_id),
    )
    critical_vertex_ids: list[str] = []
    cursor: str | None = final_vertex
    while cursor is not None:
        critical_vertex_ids.append(cursor)
        cursor = vertices[cursor].critical_pred
    critical_vertex_ids.reverse()
    critical_path = tuple(
        CriticalPathEntry(
            operation_id=vertex_id,
            members=tuple(sorted(member.key for member in vertices[vertex_id].members)),
            start_us=vertices[vertex_id].start_us,
            end_us=vertices[vertex_id].end_us,
            effective_duration_us=vertices[vertex_id].duration_us,
        )
        for vertex_id in critical_vertex_ids
    )

    return ReplayResult(
        step_time_us=max(per_rank.values(), default=0.0),
        per_rank_makespan_us=per_rank,
        timeline=tuple(timeline),
        critical_path=critical_path,
        warnings=tuple(warnings),
        profile_mode=profile_mode,
        sample_iteration_id=sample_iteration_id,
    )


def profile_iteration_ids(
    plan_or_dags: Any,
    *,
    profile_mode: str = ISOLATED_PROFILE_MODE,
) -> tuple[int, ...]:
    """Return raw sample iterations shared by every executed node profile."""

    mode = normalize_profile_mode(profile_mode)
    dags, _ = _extract_inputs(plan_or_dags)
    expected: set[int] | None = None
    for dag in dags:
        profile_mappings = [
            get_node_profile_times(node, mode)
            for node in dag.nodes.values()
            if node.node_kind != "ORDER_DUMMY"
        ]
        prologue = get_prologue_profile_times(dag, mode)
        if prologue:
            profile_mappings.append(prologue)
        for profiles in profile_mappings:
            for profile in profiles.values():
                if not bool(getattr(profile, "executed", True)):
                    continue
                samples = tuple(getattr(profile, "samples_us", ()))
                if not samples:
                    raise ReplayValidationError(
                        f"{mode} profile has no raw samples for uncertainty analysis"
                    )
                ids = tuple(
                    int(value)
                    for value in (
                        getattr(profile, "sample_iteration_ids", ())
                        or tuple(range(len(samples)))
                    )
                )
                current = set(ids)
                if expected is None:
                    expected = current
                elif current != expected:
                    raise ReplayValidationError(
                        f"{mode} raw profile iteration IDs are not aligned: "
                        f"expected={sorted(expected)!r}, got={sorted(current)!r}"
                    )
    if not expected:
        raise ReplayValidationError(
            f"profile mode {mode!r} has no raw timing samples"
        )
    return tuple(sorted(expected))


def replay_training_step_samples(
    plan_or_dags: Any,
    *,
    profile_mode: str = ISOLATED_PROFILE_MODE,
    **replay_kwargs,
) -> tuple[float, ...]:
    """Replay every aligned raw profile iteration and return step times in us."""

    mode = normalize_profile_mode(profile_mode)
    return tuple(
        replay_training_step(
            plan_or_dags,
            profile_mode=mode,
            sample_iteration_id=iteration_id,
            **replay_kwargs,
        ).step_time_us
        for iteration_id in profile_iteration_ids(
            plan_or_dags, profile_mode=mode
        )
    )


def observed_profile_step_time_samples(
    plan_or_dags: Any,
    *,
    profile_mode: str = NORMAL_PROFILE_MODE,
) -> tuple[float, ...]:
    """Return the observed CUDA timeline envelope for each profile iteration.

    Normal profiling records every node's duration and its start offset from
    that actor's iteration origin.  New profile plans translate those offsets
    to a shared origin, so the maximum ``offset + duration`` reconstructs the
    global observed GPU envelope.  It is not a counterfactual scheduler: it
    deliberately does not collapse collective endpoints or move downstream
    work.
    """

    mode = normalize_profile_mode(profile_mode)
    iteration_ids = profile_iteration_ids(plan_or_dags, profile_mode=mode)
    dags, _topology = _extract_inputs(plan_or_dags)
    samples: list[float] = []
    for iteration_id in iteration_ids:
        ends_us: list[float] = []
        for dag in dags:
            for node in dag.nodes.values():
                if node.node_kind == "ORDER_DUMMY":
                    continue
                profiles = get_node_profile_times(node, mode)
                for rank, profile in profiles.items():
                    if not bool(getattr(profile, "executed", True)):
                        continue
                    offsets = tuple(
                        getattr(profile, "start_offset_samples_us", ())
                    )
                    if not offsets:
                        raise ReplayValidationError(
                            f"node {node.uid!r} rank {rank} in mode {mode!r} "
                            "has no observed start-offset samples"
                        )
                    duration_us = _profile_duration_us(
                        profile,
                        context=f"node {node.uid!r} rank {rank}",
                        sample_iteration_id=iteration_id,
                    )
                    offset_us = _profile_start_offset_us(
                        profile,
                        context=f"node {node.uid!r} rank {rank}",
                        sample_iteration_id=iteration_id,
                    )
                    ends_us.append(offset_us + duration_us)
        if not ends_us:
            raise ReplayValidationError(
                f"profile mode {mode!r} has no executed timeline events"
            )
        samples.append(max(ends_us))
    return tuple(samples)


def replay_profile_modes(
    plan_or_dags: Any,
    *,
    actual_step_samples_us: Sequence[float] | None = None,
    strict: bool = True,
) -> ReplayComparisonResult:
    """Replay isolated and normal profiles and quantify uncertainty."""

    isolated = replay_training_step(
        plan_or_dags,
        profile_mode=ISOLATED_PROFILE_MODE,
        strict=strict,
    )
    normal = replay_training_step(
        plan_or_dags,
        profile_mode=NORMAL_PROFILE_MODE,
        strict=strict,
    )
    isolated_iteration_ids = profile_iteration_ids(
        plan_or_dags, profile_mode=ISOLATED_PROFILE_MODE
    )
    normal_iteration_ids = profile_iteration_ids(
        plan_or_dags, profile_mode=NORMAL_PROFILE_MODE
    )
    if isolated_iteration_ids != normal_iteration_ids:
        raise ReplayValidationError(
            "normal and isolated profile iterations must align for "
            "paired isolated-savings estimation: "
            f"normal={normal_iteration_ids!r}, "
            f"isolated={isolated_iteration_ids!r}"
        )
    isolated_samples = replay_training_step_samples(
        plan_or_dags,
        profile_mode=ISOLATED_PROFILE_MODE,
        strict=strict,
    )
    normal_samples = replay_training_step_samples(
        plan_or_dags,
        profile_mode=NORMAL_PROFILE_MODE,
        strict=strict,
    )
    isolated_statistics = summarize_mean_samples(
        isolated_samples
    )
    normal_statistics = summarize_mean_samples(normal_samples)
    perfect_overlap_savings_statistics = summarize_paired_savings(
        normal_samples, isolated_samples
    )
    observed_normal_envelope_samples: tuple[float, ...] = ()
    observed_normal_envelope_statistics = None
    try:
        observed_normal_envelope_samples = observed_profile_step_time_samples(
            plan_or_dags,
            profile_mode=NORMAL_PROFILE_MODE,
        )
    except ReplayValidationError:
        # Duration-only artifacts predate start-offset capture and retain the
        # dependency-replay behavior without an envelope diagnostic.
        pass
    if observed_normal_envelope_samples:
        observed_normal_envelope_statistics = summarize_timing_samples(
            observed_normal_envelope_samples,
            estimator="median",
        )

    actual_statistics = None
    normal_error = None
    prediction_error_pct = None
    event_overhead_pct = None
    instrumented_statistics = None
    instrumented_isolated_statistics = None
    uninstrumented_normal_statistics = None
    uninstrumented_isolated_statistics = None
    instrumented_normal_replay_error = None
    instrumented_isolated_replay_error = None
    observed_reduction_pct = None
    dependency_replay_reduction_pct = (
        perfect_overlap_savings_statistics.percent
    )
    instrumented_policy_reduction_pct = None
    envelope_modeled_reduction_pct = None
    normal_dependency_replay_excess_pct = None
    mode_profile_event_overhead_pct = None
    metadata = getattr(plan_or_dags, "profile_metadata", {}) or {}
    instrumented_by_mode = metadata.get(
        "instrumented_global_step_samples_us_by_mode", {}
    )
    instrumented_samples = instrumented_by_mode.get(NORMAL_PROFILE_MODE, ())
    if instrumented_samples:
        instrumented_statistics = summarize_timing_samples(
            instrumented_samples,
            estimator="median",
        )
        instrumented_normal_replay_error = prediction_error(
            normal_statistics.estimate,
            instrumented_statistics.mean,
        )
        prediction_error_pct = instrumented_normal_replay_error.signed_pct
    instrumented_isolated_samples = instrumented_by_mode.get(
        ISOLATED_PROFILE_MODE, ()
    )
    if instrumented_isolated_samples:
        instrumented_isolated_statistics = summarize_timing_samples(
            instrumented_isolated_samples,
            estimator="median",
        )
        instrumented_isolated_replay_error = prediction_error(
            isolated_statistics.estimate,
            instrumented_isolated_statistics.estimate,
        )
    if (
        instrumented_statistics is not None
        and instrumented_isolated_statistics is not None
    ):
        instrumented_policy_reduction_pct = predicted_reduction(
            instrumented_isolated_statistics.estimate,
            instrumented_statistics.estimate,
        )

    uninstrumented_by_mode = metadata.get(
        "uninstrumented_global_step_samples_us_by_mode", {}
    )
    uninstrumented_normal_samples = uninstrumented_by_mode.get(
        NORMAL_PROFILE_MODE, ()
    )
    uninstrumented_isolated_samples = uninstrumented_by_mode.get(
        ISOLATED_PROFILE_MODE, ()
    )
    if uninstrumented_normal_samples:
        uninstrumented_normal_statistics = summarize_timing_samples(
            uninstrumented_normal_samples,
            estimator="median",
        )
    if uninstrumented_isolated_samples:
        uninstrumented_isolated_statistics = summarize_timing_samples(
            uninstrumented_isolated_samples,
            estimator="median",
        )
    if (
        uninstrumented_normal_statistics is not None
        and uninstrumented_isolated_statistics is not None
    ):
        observed_reduction_pct = predicted_reduction(
            uninstrumented_isolated_statistics.estimate,
            uninstrumented_normal_statistics.estimate,
        )
    if observed_normal_envelope_statistics is not None:
        envelope_modeled_reduction_pct = predicted_reduction(
            isolated_statistics.estimate,
            observed_normal_envelope_statistics.estimate,
        )
        normal_dependency_replay_excess_pct = prediction_error(
            normal_statistics.estimate,
            observed_normal_envelope_statistics.estimate,
        ).signed_pct
    if (
        instrumented_statistics is not None
        and uninstrumented_normal_statistics is not None
    ):
        mode_profile_event_overhead_pct = prediction_error(
            instrumented_statistics.estimate,
            uninstrumented_normal_statistics.estimate,
        ).signed_pct

    if actual_step_samples_us:
        actual_statistics = summarize_timing_samples(
            actual_step_samples_us,
            estimator="median",
        )
        normal_error = prediction_error(
            normal_statistics.estimate, actual_statistics.estimate
        )

        if instrumented_samples:
            event_overhead_pct = prediction_error(
                instrumented_statistics.estimate,
                actual_statistics.estimate,
            ).signed_pct

    return ReplayComparisonResult(
        isolated=isolated,
        normal=normal,
        isolated_samples_us=isolated_samples,
        normal_samples_us=normal_samples,
        observed_normal_envelope_samples_us=(
            observed_normal_envelope_samples
        ),
        isolated_statistics=isolated_statistics,
        normal_statistics=normal_statistics,
        perfect_overlap_savings_statistics=(
            perfect_overlap_savings_statistics
        ),
        observed_normal_envelope_statistics=(
            observed_normal_envelope_statistics
        ),
        actual_statistics=actual_statistics,
        instrumented_normal_statistics=instrumented_statistics,
        instrumented_isolated_statistics=(
            instrumented_isolated_statistics
        ),
        uninstrumented_normal_statistics=uninstrumented_normal_statistics,
        uninstrumented_isolated_statistics=(
            uninstrumented_isolated_statistics
        ),
        normal_replay_error=normal_error,
        instrumented_normal_replay_error=instrumented_normal_replay_error,
        instrumented_isolated_replay_error=(
            instrumented_isolated_replay_error
        ),
        prediction_error_pct=prediction_error_pct,
        perfect_overlap_absolute_savings_us=(
            perfect_overlap_savings_statistics.absolute_mean
        ),
        interference_overhead_pct=(
            dependency_replay_reduction_pct
        ),
        instrumented_policy_reduction_pct=instrumented_policy_reduction_pct,
        observed_interference_reduction_pct=observed_reduction_pct,
        envelope_modeled_interference_reduction_pct=(
            envelope_modeled_reduction_pct
        ),
        normal_dependency_replay_excess_pct=(
            normal_dependency_replay_excess_pct
        ),
        profile_event_overhead_pct=event_overhead_pct,
        mode_profile_event_overhead_pct=mode_profile_event_overhead_pct,
        kineto_cupti_instrumentation_error=(
            None
            if instrumented_statistics is None or actual_statistics is None
            else prediction_error(
                instrumented_statistics.estimate,
                actual_statistics.estimate,
            )
        ),
    )


__all__ = [
    "CriticalPathEntry",
    "NodeInstance",
    "ReplayResult",
    "ReplayComparisonResult",
    "ReplayValidationError",
    "TimelineEvent",
    "profile_iteration_ids",
    "replay_training_step",
    "replay_training_step_samples",
    "observed_profile_step_time_samples",
    "replay_profile_modes",
]
