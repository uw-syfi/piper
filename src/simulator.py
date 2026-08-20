"""Interference-free, node-level simulation for profiled TrainingDAGs.

The simulator is deliberately CUDA-free.  It treats each profiled compute
duration as the service time the node would observe in isolation, preserves
TrainingDAG dependencies and CUDA stream FIFO order, and otherwise provides no
shared GPU or network capacity.  Consequently, independent streams and
communication transactions overlap completely.  Communication endpoint
timings can include time spent waiting for other ranks to arrive.  The
simulator therefore separates rank-local endpoint arrival and completion with
a shared zero-duration rendezvous.  For calibrated profiles it derives each
endpoint's post-rendezvous tail once from the observed, cross-actor-aligned
timeline and reuses that tail in both replays.  Symmetric collectives use an
all-participant rendezvous.  Routed A2A, M-to-N, and N-to-M nodes use one
logical rendezvous per DAG node followed by rank-local aggregate envelope
tails; internal metadata/payload phases are not modeled separately.
Collective replay fails closed when aligned endpoint timestamps are
unavailable.

P2P SEND/RECV endpoints retain distinct completion vertices.  Both arrivals
gate their shared rendezvous, but send-side successors wait only for send
completion and receive-side consumers wait only for receive completion.
Every finalized DAG dependency is preserved. Directional P2P physical streams
remain independent, but an explicit edge between them is never relaxed.
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
    INTERFERENCE_FREE_PROFILE_MODE,
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
_ROUTED_COLLECTIVE_KINDS = frozenset(
    {"A2A_COMM", "M_TO_N_COMM", "N_TO_M_COMM"}
)
_SYMMETRIC_COLLECTIVE_KINDS = frozenset(
    {
        "REDUCE_COMM",
        "ALL_REDUCE_COMM",
        "ALL_GATHER_COMM",
        "REDUCE_SCATTER_COMM",
    }
) | _ROUTED_COLLECTIVE_KINDS
_COLLECTIVE_KINDS = _SYMMETRIC_COLLECTIVE_KINDS
_COMM_KINDS = _P2P_KINDS | _COLLECTIVE_KINDS


class SimulationValidationError(ValueError):
    """Raised when an input cannot define an unambiguous simulation."""


@dataclass(frozen=True, order=True)
class NodeInstance:
    """A rank-local occurrence of a finalized TrainingDAG node."""

    global_rank: int
    uid: str


@dataclass(frozen=True)
class TimelineEvent:
    """The simulated interval for one rank-local node occurrence.

    ``start_us`` is the earliest legal start implied by the DAG and physical
    stream FIFO constraints.  Observed CUDA start offsets are retained only
    for validation; they are not replay release constraints in either timing
    mode, so normal and isolated replays differ solely by service times.
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
    """One (possibly multi-rank) operation on the simulated critical path."""

    operation_id: str
    members: tuple[NodeInstance, ...]
    start_us: float
    end_us: float
    effective_duration_us: float


@dataclass(frozen=True)
class SimulationResult:
    """Interference-free step-time estimate and its supporting timeline."""

    step_time_us: float
    per_rank_makespan_us: dict[int, float]
    timeline: tuple[TimelineEvent, ...]
    critical_path: tuple[CriticalPathEntry, ...]
    warnings: tuple[str, ...] = ()
    profile_mode: str = INTERFERENCE_FREE_PROFILE_MODE
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
class SimulationComparisonResult:
    """Interference-free and normal replays plus uncertainty and calibration."""

    interference_free: SimulationResult
    normal: SimulationResult
    interference_free_samples_us: tuple[float, ...]
    normal_samples_us: tuple[float, ...]
    observed_normal_envelope_samples_us: tuple[float, ...]
    interference_free_statistics: MeanStatistics
    normal_statistics: MeanStatistics
    dependency_replay_savings_statistics: PairedSavingsStatistics
    observed_normal_envelope_statistics: TimingStatistics | None = None
    actual_statistics: TimingStatistics | None = None
    instrumented_normal_statistics: TimingStatistics | None = None
    instrumented_interference_free_statistics: TimingStatistics | None = None
    uninstrumented_normal_statistics: TimingStatistics | None = None
    uninstrumented_interference_free_statistics: TimingStatistics | None = None
    normal_prediction_error: PredictionError | None = None
    instrumented_normal_replay_error: PredictionError | None = None
    instrumented_interference_free_replay_error: PredictionError | None = None
    predicted_interference_reduction_pct: float | None = None
    dependency_replay_absolute_savings_us: float | None = None
    dependency_replay_interference_reduction_pct: float | None = None
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
    """Compare simulated earliest legal starts with recorded CUDA starts.

    The comparison is per node instance and raw profile iteration.  A positive
    error means the observed CUDA start was later than its DAG/stream-ready
    time, i.e. residual launch or resource-queue delay not explained by the
    modeled ordering constraints.
    """
    mode = normalize_profile_mode(profile_mode)
    errors: list[float] = []
    for iteration_id in profile_iteration_ids(plan_or_dags, profile_mode=mode):
        result = simulate_training_step(
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
        raise SimulationValidationError(
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
        raise SimulationValidationError(f"{context}: boolean is not a valid profile duration")
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
                raise SimulationValidationError(
                    f"{context}: no raw sample for iteration {sample_iteration_id}"
                ) from exc
        elif "estimate_us" in profile:
            value = float(profile["estimate_us"])
        elif "duration_us" in profile:
            value = float(profile["duration_us"])
        else:
            raise SimulationValidationError(
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
                raise SimulationValidationError(
                    f"{context}: no raw sample for iteration {sample_iteration_id}"
                ) from exc
            estimate = value
        else:
            estimate = getattr(profile, "estimate_us", None)
        if estimate is None:
            estimate = getattr(profile, "duration_us", None)
        if estimate is None:
            raise SimulationValidationError(
                f"{context}: profile has neither estimate_us nor duration_us"
            )
        value = float(estimate)
    if not math.isfinite(value) or value < 0.0:
        raise SimulationValidationError(
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
            raise SimulationValidationError(
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
        raise SimulationValidationError(
            f"{context}: start offset must be finite and non-negative, got {value!r}"
        )
    return value


def _profile_for_rank(
    node: TrainingDAGNode, rank: int, profile_mode: str
) -> Any:
    profiles = get_node_profile_times(node, profile_mode)
    if not isinstance(profiles, Mapping):
        raise SimulationValidationError(
            f"node {node.uid!r} has no rank-keyed {profile_mode} profile mapping"
        )
    if rank in profiles:
        return profiles[rank]
    # Serialized artifacts may have passed through JSON without using the
    # ProfiledTrainingPlan loader yet.
    if str(rank) in profiles:
        return profiles[str(rank)]
    raise SimulationValidationError(
        f"missing profile for node {node.uid!r} on global rank {rank} "
        f"in mode {profile_mode!r}"
    )


def _profile_iteration_ids_for_collective(
    profile: Any, *, context: str
) -> tuple[int, ...]:
    if isinstance(profile, Mapping):
        samples = tuple(profile.get("samples_us", ()))
        raw_ids = tuple(profile.get("sample_iteration_ids", ()))
    else:
        samples = tuple(getattr(profile, "samples_us", ()))
        raw_ids = tuple(getattr(profile, "sample_iteration_ids", ()))
    if not samples:
        raise SimulationValidationError(
            f"{context}: aligned collective profile has no raw samples"
        )
    return tuple(int(value) for value in (raw_ids or range(len(samples))))


def _profile_estimator(profile: Any) -> str:
    if isinstance(profile, Mapping):
        return str(profile.get("estimator", "median"))
    return str(getattr(profile, "estimator", "median"))


def _symmetric_collective_observed_services(
    members: Sequence[_InstanceRecord],
    *,
    communication_id: str,
    sample_iteration_id: int | None,
    pairwise_alignment_uncertainty_us: float,
) -> dict[NodeInstance, float]:
    """Derive endpoint service from paired aligned activity occurrences."""

    if sample_iteration_id is None:
        if any(member.profile is None for member in members):
            raise SimulationValidationError(
                f"collective {communication_id!r} has no endpoint profile"
            )
        endpoint_ids = {
            _profile_iteration_ids_for_collective(
                member.profile,
                context=(
                    f"collective {communication_id!r} rank "
                    f"{member.key.global_rank} uid={member.key.uid!r}"
                ),
            )
            for member in members
        }
        if len(endpoint_ids) != 1:
            raise SimulationValidationError(
                f"collective {communication_id!r} endpoint iteration IDs are "
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
                    f"collective {communication_id!r} rank "
                    f"{member.key.global_rank} uid={member.key.uid!r}"
                ),
                sample_iteration_id=iteration_id,
            )
            for member in members
        }
        zero_activity_states = {duration == 0.0 for duration in durations.values()}
        routed_collective = all(
            member.node.node_kind in _ROUTED_COLLECTIVE_KINDS
            for member in members
        )
        if len(zero_activity_states) > 1 and not routed_collective:
            raise SimulationValidationError(
                f"collective {communication_id!r} iteration {iteration_id} "
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
                    f"collective {communication_id!r} rank "
                    f"{member.key.global_rank} uid={member.key.uid!r}"
                ),
                sample_iteration_id=iteration_id,
            )
            for member in members
        }
        observed_rendezvous_us = max(starts.values())
        for member in members:
            if durations[member.key] == 0.0:
                service_samples[member.key].append(0.0)
                continue
            observed_end_us = starts[member.key] + durations[member.key]
            raw_service_us = observed_end_us - observed_rendezvous_us
            if raw_service_us < -pairwise_alignment_uncertainty_us:
                raise SimulationValidationError(
                    f"collective {communication_id!r} iteration {iteration_id} "
                    "has an inconsistent aligned endpoint at rank "
                    f"{member.key.global_rank}, uid={member.key.uid!r}: "
                    f"start={starts[member.key]:.3f} us, "
                    f"end={observed_end_us:.3f} us, rendezvous="
                    f"{observed_rendezvous_us:.3f} us, pairwise_uncertainty="
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
            "simulate_training_step expects a ProfiledTrainingPlan-like object "
            "or a sequence of TrainingDAGs"
        )
    if not dags:
        raise SimulationValidationError("at least one per-PP TrainingDAG is required")
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
            raise SimulationValidationError(f"node {node.uid!r} has an empty device list")
        if len(set(ranks)) != len(ranks):
            raise SimulationValidationError(
                f"node {node.uid!r} has duplicate ranks in device={node.device!r}"
            )
        layouts.add(ranks)
    if not layouts:
        return None
    if len(layouts) != 1:
        raise SimulationValidationError(
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
                    raise SimulationValidationError(
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
        raise SimulationValidationError(
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
        raise SimulationValidationError(
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
            raise SimulationValidationError(
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
        raise SimulationValidationError(
            f"all ActorGroup rank lists must be non-empty, got {ranks_by_pp!r}"
        )
    flat = [rank for ranks in ranks_by_pp for rank in ranks]
    if len(flat) != len(set(flat)):
        raise SimulationValidationError(f"global ranks occur in multiple PP groups: {ranks_by_pp!r}")
    if any(rank < 0 for rank in flat):
        raise SimulationValidationError(f"global ranks must be non-negative: {ranks_by_pp!r}")
    if topology_world is not None and int(topology_world) != len(flat):
        raise SimulationValidationError(
            f"world_size={topology_world} does not match {len(flat)} simulated ranks"
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
            raise SimulationValidationError(
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
                raise SimulationValidationError("DAG prologue_profile_times must be a mapping")
            sources.append(candidate)
    for source in sources:
        for raw_rank, profile in source.items():
            rank = int(raw_rank)
            if rank in profiles:
                old = _profile_duration_us(profiles[rank], context=f"prologue rank {rank}")
                new = _profile_duration_us(profile, context=f"prologue rank {rank}")
                if old != new:
                    raise SimulationValidationError(
                        f"conflicting prologue profiles for rank {rank}: {old} vs {new} us"
                    )
            profiles[rank] = profile
    return profiles


def _expected_participants(node: TrainingDAGNode) -> tuple[int, ...] | None:
    raw = node.node_meta.get("participants")
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise SimulationValidationError(
            f"node {node.uid!r} participants must be a rank sequence"
        )
    result = tuple(int(rank) for rank in raw)
    if len(result) != len(set(result)):
        raise SimulationValidationError(f"node {node.uid!r} has duplicate participants")
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
        raise SimulationValidationError(
            f"P2P transaction {vertex.communication_id!r} must have exactly one SEND and "
            f"one RECV endpoint; got {descriptions!r}"
        )
    send, recv = sends[0], recvs[0]
    if send.actor_group_local_rank != recv.actor_group_local_rank:
        raise SimulationValidationError(
            f"P2P transaction {vertex.communication_id!r} crosses ActorGroup-local "
            f"ranks: {send.actor_group_local_rank} vs "
            f"{recv.actor_group_local_rank}"
        )
    transaction_ranks = {
        send.key.global_rank,
        recv.key.global_rank,
    }
    # Piper inserts one SEND/RECV node per pair of ActorGroups, then the
    # simulator expands that node once per same-position ActorGroup member.
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
            raise SimulationValidationError(
                f"P2P endpoint {endpoint.key} expects peer rank {peer_global}, got "
                f"{counterpart.key.global_rank}"
            )
        peer_pp = meta.get("peer_pp_rank")
        if peer_pp is not None and int(peer_pp) != counterpart.pp_index:
            raise SimulationValidationError(
                f"P2P endpoint {endpoint.key} expects peer PP rank {peer_pp}, got "
                f"{counterpart.pp_index}"
            )
        participants = _expected_participants(endpoint.node)
        if (
            participants is not None
            and frozenset(participants) not in valid_participant_sets
        ):
            raise SimulationValidationError(
                f"P2P endpoint {endpoint.key} participants={participants!r} do not match "
                f"transaction ranks {(send.key.global_rank, recv.key.global_rank)!r} "
                f"or endpoint device-group ranks {tuple(sorted(device_group_ranks))!r}"
            )
    groups = {_p2p_group(member) for member in vertex.members}
    if len(groups) != 1:
        raise SimulationValidationError(
            f"P2P transaction {vertex.communication_id!r} endpoints disagree on "
            f"comm_group_kind: {sorted(groups)!r}"
        )


def _p2p_group(record: _InstanceRecord) -> str:
    explicit = record.node.node_meta.get("comm_group_kind")
    if explicit is not None:
        group = str(explicit)
        if not group:
            raise SimulationValidationError(
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


def _simulate_vertices(vertices: dict[str, _Vertex]) -> tuple[str, ...]:
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
            # A host/GPU release constraint, rather than a simulated vertex,
            # determines this start.  Do not claim a shorter dependency-only
            # chain is critical through the release gap.
            vertex.critical_pred = None
        if vertex.endpoint_arrival_vertex_id is not None:
            arrival = vertices[vertex.endpoint_arrival_vertex_id]
            observed_service_us = vertex.endpoint_observed_service_us
            if observed_service_us is None:
                raise SimulationValidationError(
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
        raise SimulationValidationError(
            "communication simulation graph contains a cycle (often inconsistent collective "
            f"ordering across ranks): {cyclic!r}"
        )
    return tuple(order)


def simulate_training_step(
    plan_or_dags: Any,
    *,
    pp_degree: int | None = None,
    prologue_profile_times: Mapping[int, Any] | None = None,
    profile_mode: str = INTERFERENCE_FREE_PROFILE_MODE,
    sample_iteration_id: int | None = None,
    strict: bool = True,
) -> SimulationResult:
    """Simulate one interference-free training step.

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
        raise SimulationValidationError(
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
        raise SimulationValidationError(
            f"profile mode {profile_mode!r} has invalid pairwise timeline "
            f"uncertainty {raw_pairwise_uncertainty!r}"
        )
    empty_dags = [index for index, dag in enumerate(dags) if not dag.nodes]
    if empty_dags:
        raise SimulationValidationError(
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
                    raise SimulationValidationError(
                        f"conflicting prologue profiles for rank {rank}: {old} vs {new} us"
                    )
            merged_prologue[rank] = profile

    rank_set = set(layout.ranks)
    if merged_prologue:
        missing = sorted(rank_set - set(merged_prologue))
        extra = sorted(set(merged_prologue) - rank_set)
        if missing:
            message = f"prologue profiles are missing simulated ranks {missing!r}"
            if strict:
                raise SimulationValidationError(message)
            warnings.append(message)
        if extra:
            message = f"prologue profiles contain non-simulated ranks {extra!r}"
            if strict:
                raise SimulationValidationError(message)
            warnings.append(message)

    instances: dict[NodeInstance, _InstanceRecord] = {}
    dag_plans: list[tuple[PlannedOperation, ...]] = []
    for pp_index, dag in enumerate(dags):
        if _PROLOGUE_UID in dag.nodes:
            raise SimulationValidationError(
                f"node UID {_PROLOGUE_UID!r} is reserved by the simulator"
            )
        try:
            plan = tuple(build_execution_plan(dag))
        except ValueError as exc:
            raise SimulationValidationError(str(exc)) from exc
        dag_plans.append(plan)
        expected_ranks = set(layout.ranks_by_pp[pp_index])
        for operation in plan:
            uid = operation.uid
            node = dag.nodes[uid]
            if not isinstance(node.stream, str) or not node.stream:
                raise SimulationValidationError(
                    f"node {uid!r} must have a non-empty logical stream name"
                )
            profiles = get_node_profile_times(node, profile_mode)
            if strict and isinstance(profiles, Mapping):
                try:
                    attached_ranks = {int(rank) for rank in profiles}
                except (TypeError, ValueError) as exc:
                    raise SimulationValidationError(
                        f"node {uid!r} has a non-integer profile rank"
                    ) from exc
                extra = sorted(attached_ranks - expected_ranks)
                if extra:
                    raise SimulationValidationError(
                        f"node {uid!r} in PP DAG {pp_index} has profiles for "
                        f"non-member ranks {extra!r}"
                    )
            for rank in layout.ranks_by_pp[pp_index]:
                key = NodeInstance(rank, uid)
                if key in instances:
                    raise SimulationValidationError(f"duplicate node instance {key}")
                if node.node_kind == "ORDER_DUMMY":
                    profile = None
                    profile_duration = 0.0
                    profile_start_offset = 0.0
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
                        raise SimulationValidationError(
                            f"aligned profile mode {profile_mode!r} is missing "
                            f"start offsets for node {uid!r} rank {rank}"
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
            raise SimulationValidationError(
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
                raise SimulationValidationError(
                    f"collective {communication_id!r} endpoints disagree on participants: "
                    f"{sorted(expected_sets)!r}"
                )
            if group_key[1] in _ROUTED_COLLECTIVE_KINDS and not expected_sets:
                raise SimulationValidationError(
                    f"routed collective {communication_id!r} has no declared participants"
                )
            if expected_sets and actual != set(next(iter(expected_sets))):
                raise SimulationValidationError(
                    f"collective {communication_id!r} has ranks {sorted(actual)!r}, expected "
                    f"{sorted(next(iter(expected_sets)))!r}"
                )

        observed_service_by_endpoint: dict[NodeInstance, float]
        if group_key[0] == "collective":
            observed_service_by_endpoint = _symmetric_collective_observed_services(
                members,
                communication_id=communication_id,
                sample_iteration_id=sample_iteration_id,
                pairwise_alignment_uncertainty_us=(
                    pairwise_alignment_uncertainty_us
                ),
            )
        else:
            # Directional P2P durations are isolated transfer-service
            # microprofiles, so no observed-wait decomposition is needed.
            observed_service_by_endpoint = {
                member.key: member.profile_duration_us for member in members
            }

        arrival_ids: dict[NodeInstance, str] = {}
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
            if arrival_id in vertices or completion_id in vertices:
                raise SimulationValidationError(
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
            vertices[completion_id] = _Vertex(
                vertex_id=completion_id,
                members=[member],
                # Calibrated profiles provide a fixed tail derived from the
                # observed rendezvous.
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
                for predecessor_uid in operation.issue_predecessors:
                    src = instance_start_vertex[
                        NodeInstance(rank, predecessor_uid)
                    ]
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

    simulation_order = _simulate_vertices(vertices)

    timeline: list[TimelineEvent] = []
    per_rank: dict[int, float] = {rank: 0.0 for rank in layout.ranks}
    for vertex_id in simulation_order:
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

    return SimulationResult(
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
    profile_mode: str = INTERFERENCE_FREE_PROFILE_MODE,
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
                    raise SimulationValidationError(
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
                    raise SimulationValidationError(
                        f"{mode} raw profile iteration IDs are not aligned: "
                        f"expected={sorted(expected)!r}, got={sorted(current)!r}"
                    )
    if not expected:
        raise SimulationValidationError(
            f"profile mode {mode!r} has no raw timing samples"
        )
    return tuple(sorted(expected))


def simulate_training_step_samples(
    plan_or_dags: Any,
    *,
    profile_mode: str = INTERFERENCE_FREE_PROFILE_MODE,
    **simulation_kwargs,
) -> tuple[float, ...]:
    """Replay every aligned raw profile iteration and return step times in us."""

    mode = normalize_profile_mode(profile_mode)
    return tuple(
        simulate_training_step(
            plan_or_dags,
            profile_mode=mode,
            sample_iteration_id=iteration_id,
            **simulation_kwargs,
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
                        raise SimulationValidationError(
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
            raise SimulationValidationError(
                f"profile mode {mode!r} has no executed timeline events"
            )
        samples.append(max(ends_us))
    return tuple(samples)


def simulate_profile_modes(
    plan_or_dags: Any,
    *,
    actual_step_samples_us: Sequence[float] | None = None,
    strict: bool = True,
) -> SimulationComparisonResult:
    """Replay interference-free and normal profiles and quantify uncertainty."""

    interference_free = simulate_training_step(
        plan_or_dags,
        profile_mode=INTERFERENCE_FREE_PROFILE_MODE,
        strict=strict,
    )
    normal = simulate_training_step(
        plan_or_dags,
        profile_mode=NORMAL_PROFILE_MODE,
        strict=strict,
    )
    interference_free_iteration_ids = profile_iteration_ids(
        plan_or_dags, profile_mode=INTERFERENCE_FREE_PROFILE_MODE
    )
    normal_iteration_ids = profile_iteration_ids(
        plan_or_dags, profile_mode=NORMAL_PROFILE_MODE
    )
    if interference_free_iteration_ids != normal_iteration_ids:
        raise SimulationValidationError(
            "normal and interference-free profile iterations must align for "
            "paired IF-savings estimation: "
            f"normal={normal_iteration_ids!r}, "
            f"interference_free={interference_free_iteration_ids!r}"
        )
    interference_free_samples = simulate_training_step_samples(
        plan_or_dags,
        profile_mode=INTERFERENCE_FREE_PROFILE_MODE,
        strict=strict,
    )
    normal_samples = simulate_training_step_samples(
        plan_or_dags,
        profile_mode=NORMAL_PROFILE_MODE,
        strict=strict,
    )
    interference_free_statistics = summarize_mean_samples(
        interference_free_samples
    )
    normal_statistics = summarize_mean_samples(normal_samples)
    dependency_replay_savings_statistics = summarize_paired_savings(
        normal_samples, interference_free_samples
    )
    observed_normal_envelope_samples: tuple[float, ...] = ()
    observed_normal_envelope_statistics = None
    try:
        observed_normal_envelope_samples = observed_profile_step_time_samples(
            plan_or_dags,
            profile_mode=NORMAL_PROFILE_MODE,
        )
    except SimulationValidationError:
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
    reduction_pct = None
    event_overhead_pct = None
    instrumented_statistics = None
    instrumented_interference_free_statistics = None
    uninstrumented_normal_statistics = None
    uninstrumented_interference_free_statistics = None
    instrumented_normal_replay_error = None
    instrumented_interference_free_replay_error = None
    observed_reduction_pct = None
    dependency_replay_reduction_pct = (
        dependency_replay_savings_statistics.percent
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
            instrumented_statistics.estimate,
        )
    instrumented_interference_free_samples = instrumented_by_mode.get(
        INTERFERENCE_FREE_PROFILE_MODE, ()
    )
    if instrumented_interference_free_samples:
        instrumented_interference_free_statistics = summarize_timing_samples(
            instrumented_interference_free_samples,
            estimator="median",
        )
        instrumented_interference_free_replay_error = prediction_error(
            interference_free_statistics.estimate,
            instrumented_interference_free_statistics.estimate,
        )
    if (
        instrumented_statistics is not None
        and instrumented_interference_free_statistics is not None
    ):
        instrumented_policy_reduction_pct = predicted_reduction(
            instrumented_interference_free_statistics.estimate,
            instrumented_statistics.estimate,
        )

    uninstrumented_by_mode = metadata.get(
        "uninstrumented_global_step_samples_us_by_mode", {}
    )
    uninstrumented_normal_samples = uninstrumented_by_mode.get(
        NORMAL_PROFILE_MODE, ()
    )
    uninstrumented_interference_free_samples = uninstrumented_by_mode.get(
        INTERFERENCE_FREE_PROFILE_MODE, ()
    )
    if uninstrumented_normal_samples:
        uninstrumented_normal_statistics = summarize_timing_samples(
            uninstrumented_normal_samples,
            estimator="median",
        )
    if uninstrumented_interference_free_samples:
        uninstrumented_interference_free_statistics = summarize_timing_samples(
            uninstrumented_interference_free_samples,
            estimator="median",
        )
    if (
        uninstrumented_normal_statistics is not None
        and uninstrumented_interference_free_statistics is not None
    ):
        observed_reduction_pct = predicted_reduction(
            uninstrumented_interference_free_statistics.estimate,
            uninstrumented_normal_statistics.estimate,
        )
    if observed_normal_envelope_statistics is not None:
        envelope_modeled_reduction_pct = predicted_reduction(
            interference_free_statistics.estimate,
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
        reduction_pct = predicted_reduction(
            interference_free_statistics.estimate, actual_statistics.estimate
        )

        if instrumented_samples:
            event_overhead_pct = prediction_error(
                instrumented_statistics.estimate,
                actual_statistics.estimate,
            ).signed_pct

    return SimulationComparisonResult(
        interference_free=interference_free,
        normal=normal,
        interference_free_samples_us=interference_free_samples,
        normal_samples_us=normal_samples,
        observed_normal_envelope_samples_us=(
            observed_normal_envelope_samples
        ),
        interference_free_statistics=interference_free_statistics,
        normal_statistics=normal_statistics,
        dependency_replay_savings_statistics=(
            dependency_replay_savings_statistics
        ),
        observed_normal_envelope_statistics=(
            observed_normal_envelope_statistics
        ),
        actual_statistics=actual_statistics,
        instrumented_normal_statistics=instrumented_statistics,
        instrumented_interference_free_statistics=(
            instrumented_interference_free_statistics
        ),
        uninstrumented_normal_statistics=uninstrumented_normal_statistics,
        uninstrumented_interference_free_statistics=(
            uninstrumented_interference_free_statistics
        ),
        normal_prediction_error=normal_error,
        instrumented_normal_replay_error=instrumented_normal_replay_error,
        instrumented_interference_free_replay_error=(
            instrumented_interference_free_replay_error
        ),
        predicted_interference_reduction_pct=reduction_pct,
        dependency_replay_absolute_savings_us=(
            dependency_replay_savings_statistics.absolute_mean
        ),
        dependency_replay_interference_reduction_pct=(
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
    "SimulationResult",
    "SimulationComparisonResult",
    "SimulationValidationError",
    "TimelineEvent",
    "profile_iteration_ids",
    "simulate_training_step",
    "simulate_training_step_samples",
    "observed_profile_step_time_samples",
    "simulate_profile_modes",
]
