"""Durable multi-mode node profiling records for stream simulation.

This module intentionally contains no CUDA or Ray code.  Actors produce
``ActorProfileResult`` shards, the coordinator merges those shards into the
finalized DAGs, and the simulator consumes the resulting
``ProfiledTrainingPlan``.  Keeping the interchange types here prevents live FX,
CUDA event, and executor state from leaking into saved profile artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode


PROFILE_SCHEMA_VERSION = 2
PROFILE_PLAN_SCHEMA_VERSION = 4
INTERFERENCE_FREE_PROFILE_MODE = "interference_free"
NORMAL_PROFILE_MODE = "normal"
PROFILE_MODES = frozenset({INTERFERENCE_FREE_PROFILE_MODE, NORMAL_PROFILE_MODE})
DEFAULT_ISOLATION_SCOPE = "event_serialized_logical_streams"
DEFAULT_NORMAL_SCOPE = "logical_streams_with_directional_p2p"


def normalize_profile_mode(mode: str) -> str:
    """Validate and canonicalize a named timing dataset."""

    normalized = str(mode).strip().lower()
    if normalized not in PROFILE_MODES:
        raise ValueError(
            f"unsupported profile mode {mode!r}; expected one of {sorted(PROFILE_MODES)}"
        )
    return normalized


@dataclass(frozen=True)
class TensorProfileSpec:
    """Shape information needed to replay a communication operation."""

    shape: tuple[int, ...]
    dtype: str
    bytes: int

    def __post_init__(self) -> None:
        if any(int(dim) < 0 for dim in self.shape):
            raise ValueError(f"tensor profile shape must be non-negative, got {self.shape}")
        if int(self.bytes) < 0:
            raise ValueError(f"tensor profile bytes must be non-negative, got {self.bytes}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TensorProfileSpec:
        return cls(
            shape=tuple(int(dim) for dim in value.get("shape", ())),
            dtype=str(value.get("dtype", "")),
            bytes=int(value.get("bytes", 0)),
        )


@dataclass(frozen=True)
class NodeProfile:
    """Repeated, rank-local measurements for one finalized DAG node."""

    schema_version: int = PROFILE_SCHEMA_VERSION
    samples_us: tuple[float, ...] = ()
    # Explicit IDs make raw samples pairable across nodes/ranks.
    sample_iteration_ids: tuple[int, ...] = ()
    # Optional GPU timeline position for each duration sample.  Newly collected
    # plans align these offsets to a shared per-iteration origin across actors;
    # legacy artifacts contain rank-local offsets.  ``profile_metadata`` says
    # which representation a plan contains.
    start_offset_samples_us: tuple[float, ...] = ()
    start_offset_estimate_us: float | None = None
    estimate_us: float = 0.0
    estimator: str = "median"
    timing_source: str = "cuda_event"
    isolation_scope: str = DEFAULT_ISOLATION_SCOPE
    executed: bool = True
    tensor_specs: tuple[TensorProfileSpec, ...] = ()
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if int(self.schema_version) != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported node profile schema_version={self.schema_version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )
        for sample in self.samples_us:
            if not math.isfinite(float(sample)) or float(sample) < 0:
                raise ValueError(f"profile samples must be finite and non-negative: {self.samples_us}")
        if self.samples_us and not self.sample_iteration_ids:
            object.__setattr__(
                self, "sample_iteration_ids", tuple(range(len(self.samples_us)))
            )
        if len(self.sample_iteration_ids) != len(self.samples_us):
            raise ValueError(
                "sample_iteration_ids must have one entry per raw sample: "
                f"ids={len(self.sample_iteration_ids)}, samples={len(self.samples_us)}"
            )
        if len(set(self.sample_iteration_ids)) != len(self.sample_iteration_ids):
            raise ValueError("sample_iteration_ids must be unique")
        if any(int(iteration) < 0 for iteration in self.sample_iteration_ids):
            raise ValueError("sample_iteration_ids must be non-negative")
        if not math.isfinite(float(self.estimate_us)) or float(self.estimate_us) < 0:
            raise ValueError(
                f"profile estimate must be finite and non-negative, got {self.estimate_us}"
            )
        if self.start_offset_samples_us:
            if len(self.start_offset_samples_us) != len(self.samples_us):
                raise ValueError(
                    "start_offset_samples_us must have one entry per raw sample: "
                    f"offsets={len(self.start_offset_samples_us)}, "
                    f"samples={len(self.samples_us)}"
                )
            for offset in self.start_offset_samples_us:
                if not math.isfinite(float(offset)) or float(offset) < 0:
                    raise ValueError(
                        "profile start offsets must be finite and non-negative: "
                        f"{self.start_offset_samples_us}"
                    )
            if self.start_offset_estimate_us is None:
                object.__setattr__(
                    self,
                    "start_offset_estimate_us",
                    estimate_profile_time(self.start_offset_samples_us, self.estimator),
                )
        if self.start_offset_estimate_us is not None and (
            not math.isfinite(float(self.start_offset_estimate_us))
            or float(self.start_offset_estimate_us) < 0
        ):
            raise ValueError(
                "profile start-offset estimate must be finite and non-negative, "
                f"got {self.start_offset_estimate_us}"
            )
        if not self.start_offset_samples_us and self.start_offset_estimate_us is not None:
            raise ValueError(
                "start_offset_estimate_us requires start_offset_samples_us"
            )
        if not self.estimator:
            raise ValueError("profile estimator must be non-empty")
        if self.executed and self.skip_reason is not None:
            raise ValueError("an executed profile cannot have a skip_reason")

    @classmethod
    def from_samples(
        cls,
        samples_us: Iterable[float],
        *,
        estimator: str = "median",
        timing_source: str = "cuda_event",
        isolation_scope: str = DEFAULT_ISOLATION_SCOPE,
        tensor_specs: Sequence[TensorProfileSpec] = (),
        sample_iteration_ids: Sequence[int] | None = None,
        start_offset_samples_us: Sequence[float] = (),
    ) -> NodeProfile:
        samples = tuple(float(sample) for sample in samples_us)
        start_offsets = tuple(float(offset) for offset in start_offset_samples_us)
        if not samples:
            raise ValueError("cannot estimate a node duration from zero samples")
        estimate = estimate_profile_time(samples, estimator)
        return cls(
            samples_us=samples,
            sample_iteration_ids=(
                tuple(range(len(samples)))
                if sample_iteration_ids is None
                else tuple(int(iteration) for iteration in sample_iteration_ids)
            ),
            estimate_us=estimate,
            start_offset_samples_us=start_offsets,
            start_offset_estimate_us=(
                estimate_profile_time(start_offsets, estimator)
                if start_offsets
                else None
            ),
            estimator=estimator,
            timing_source=timing_source,
            isolation_scope=isolation_scope,
            tensor_specs=tuple(tensor_specs),
        )

    @classmethod
    def skipped(
        cls,
        reason: str,
        *,
        timing_source: str = "synthetic_zero",
        isolation_scope: str = DEFAULT_ISOLATION_SCOPE,
        tensor_specs: Sequence[TensorProfileSpec] = (),
    ) -> NodeProfile:
        if not reason:
            raise ValueError("a skipped profile requires a reason")
        return cls(
            samples_us=(),
            estimate_us=0.0,
            timing_source=timing_source,
            isolation_scope=isolation_scope,
            executed=False,
            tensor_specs=tuple(tensor_specs),
            skip_reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["samples_us"] = list(self.samples_us)
        value["sample_iteration_ids"] = list(self.sample_iteration_ids)
        value["start_offset_samples_us"] = list(self.start_offset_samples_us)
        value["tensor_specs"] = [asdict(spec) for spec in self.tensor_specs]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NodeProfile:
        source_schema_version = int(value["schema_version"])
        if source_schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported node profile schema_version={source_schema_version}; "
                f"expected {PROFILE_SCHEMA_VERSION}"
            )
        samples = tuple(float(sample) for sample in value.get("samples_us", ()))
        start_offset_samples = tuple(
            float(offset) for offset in value.get("start_offset_samples_us", ())
        )
        raw_start_offset_estimate = value.get("start_offset_estimate_us")
        return cls(
            schema_version=PROFILE_SCHEMA_VERSION,
            samples_us=samples,
            sample_iteration_ids=tuple(
                int(iteration) for iteration in value["sample_iteration_ids"]
            ),
            start_offset_samples_us=start_offset_samples,
            start_offset_estimate_us=(
                None
                if not start_offset_samples or raw_start_offset_estimate is None
                else float(raw_start_offset_estimate)
            ),
            estimate_us=float(value.get("estimate_us", 0.0)),
            estimator=str(value.get("estimator", "median")),
            timing_source=str(value.get("timing_source", "cuda_event")),
            isolation_scope=str(value.get("isolation_scope", DEFAULT_ISOLATION_SCOPE)),
            executed=bool(value.get("executed", True)),
            tensor_specs=tuple(
                TensorProfileSpec.from_dict(spec) for spec in value.get("tensor_specs", ())
            ),
            skip_reason=value.get("skip_reason"),
        )


def estimate_profile_time(samples_us: Sequence[float], estimator: str = "median") -> float:
    """Reduce repeated measurements to the duration used by simulation."""

    if not samples_us:
        raise ValueError("profile estimator requires at least one sample")
    samples = tuple(float(sample) for sample in samples_us)
    if estimator == "median":
        return float(statistics.median(samples))
    if estimator == "mean":
        return float(statistics.fmean(samples))
    if estimator == "min":
        return float(min(samples))
    if estimator in {"p10", "percentile_10"}:
        ordered = sorted(samples)
        # Nearest-rank p10.  Avoid numpy in the artifact/schema layer.
        return float(ordered[max(0, math.ceil(0.10 * len(ordered)) - 1)])
    raise ValueError(f"unsupported profile estimator: {estimator!r}")


@dataclass(frozen=True)
class ProfileConfig:
    warmup_iters: int = 5
    # Minimum number of measured iterations for this profile mode.  When no
    # CI target is requested this remains the exact number of iterations.
    repeat_iters: int = 30
    # Optionally keep sampling until the bootstrap 95% CI of the chosen step
    # estimator is this tight (percent), subject to ``max_repeat_iters``.
    target_ci95_error_pct: float | None = None
    max_repeat_iters: int | None = None
    estimator: str = "median"
    network_policy: str = "directional_p2p"
    isolation_scope: str = DEFAULT_ISOLATION_SCOPE

    def __post_init__(self) -> None:
        if int(self.warmup_iters) < 0:
            raise ValueError("warmup_iters must be non-negative")
        if int(self.repeat_iters) <= 0:
            raise ValueError("repeat_iters must be positive")
        if self.target_ci95_error_pct is not None and float(self.target_ci95_error_pct) <= 0.0:
            raise ValueError("target_ci95_error_pct must be positive when set")
        if self.max_repeat_iters is not None and int(self.max_repeat_iters) < int(self.repeat_iters):
            raise ValueError("max_repeat_iters must be at least repeat_iters")
        # Validate the name without requiring measurements yet.
        estimate_profile_time((0.0,), self.estimator)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProfileConfig:
        return cls(
            warmup_iters=int(value.get("warmup_iters", 5)),
            repeat_iters=int(value.get("repeat_iters", 30)),
            target_ci95_error_pct=(
                None
                if value.get("target_ci95_error_pct") is None
                else float(value["target_ci95_error_pct"])
            ),
            max_repeat_iters=(
                None
                if value.get("max_repeat_iters") is None
                else int(value["max_repeat_iters"])
            ),
            estimator=str(value.get("estimator", "median")),
            network_policy=str(value.get("network_policy", "directional_p2p")),
            isolation_scope=str(value.get("isolation_scope", DEFAULT_ISOLATION_SCOPE)),
        )


@dataclass(frozen=True)
class ActorProfileResult:
    """Detached profile shard returned by one global-rank actor."""

    graph_fingerprint: str
    global_rank: int
    node_profiles: dict[str, NodeProfile]
    profile_mode: str = INTERFERENCE_FREE_PROFILE_MODE
    dp_rank: int = 0
    pp_rank: int = 0
    device_metadata: dict[str, Any] = field(default_factory=dict)
    prologue_profile: NodeProfile | None = None
    # Coordinator-clock timestamps corresponding to each local CUDA prologue
    # start event.  These are merge-time calibration data and are normalized
    # out before the durable plan is written.
    timeline_origin_samples_us: tuple[float, ...] = ()
    timeline_origin_uncertainty_samples_us: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_mode", normalize_profile_mode(self.profile_mode))

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_fingerprint": self.graph_fingerprint,
            "global_rank": int(self.global_rank),
            "profile_mode": self.profile_mode,
            "dp_rank": int(self.dp_rank),
            "pp_rank": int(self.pp_rank),
            "device_metadata": _json_safe_value(self.device_metadata),
            "timeline_origin_samples_us": list(self.timeline_origin_samples_us),
            "timeline_origin_uncertainty_samples_us": list(
                self.timeline_origin_uncertainty_samples_us
            ),
            "node_profiles": {
                uid: profile.to_dict() for uid, profile in sorted(self.node_profiles.items())
            },
            "prologue_profile": (
                None if self.prologue_profile is None else self.prologue_profile.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ActorProfileResult:
        prologue = value.get("prologue_profile")
        return cls(
            graph_fingerprint=str(value["graph_fingerprint"]),
            global_rank=int(value["global_rank"]),
            profile_mode=str(value["profile_mode"]),
            dp_rank=int(value.get("dp_rank", 0)),
            pp_rank=int(value.get("pp_rank", 0)),
            device_metadata=dict(value.get("device_metadata", {})),
            timeline_origin_samples_us=tuple(
                float(item)
                for item in value.get("timeline_origin_samples_us", ())
            ),
            timeline_origin_uncertainty_samples_us=tuple(
                float(item)
                for item in value.get(
                    "timeline_origin_uncertainty_samples_us", ()
                )
            ),
            node_profiles={
                str(uid): NodeProfile.from_dict(profile)
                for uid, profile in value.get("node_profiles", {}).items()
            },
            prologue_profile=None if prologue is None else NodeProfile.from_dict(prologue),
        )


_FINGERPRINT_EXCLUDED_META = frozenset(
    {
        "events",
        "event",
        "gm_data",
        "batch",
        "batches",
        "buffers",
        "runtime",
        "runtime_state",
        "completion_event",
    }
)
_FINGERPRINT_META_OVERRIDE = "__profile_fingerprint_node_meta__"


def _qualified_type(value: Any) -> str:
    typ = type(value)
    return f"{typ.__module__}.{typ.__qualname__}"


def _fingerprint_value(value: Any) -> Any:
    """Canonicalize graph metadata without object addresses or live state."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
            if str(key) not in _FINGERPRINT_EXCLUDED_META
        }
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical = [_fingerprint_value(item) for item in value]
        return sorted(canonical, key=lambda item: json.dumps(item, sort_keys=True))

    # Avoid importing torch here solely for isinstance checks.  These
    # attributes describe both real and fake/meta tensors.
    if all(hasattr(value, attr) for attr in ("shape", "dtype")):
        try:
            return {
                "type": _qualified_type(value),
                "shape": [int(dim) for dim in value.shape],
                "dtype": str(value.dtype),
                "requires_grad": bool(getattr(value, "requires_grad", False)),
            }
        except (TypeError, ValueError):
            pass

    graph = getattr(value, "graph", None)
    if graph is not None:
        # FX graph formatting is stable for a finalized graph and, unlike
        # repr(GraphModule), does not contain a process-local address.
        return {"type": _qualified_type(value), "graph": str(graph)}
    return {"type": _qualified_type(value)}


def _dag_fingerprint_payload(dag: TrainingDAG) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "uid": node.uid,
                "node_kind": node.node_kind,
                "compute_subkind": node.compute_subkind,
                "tag": _fingerprint_value(node.tag),
                "device": None if node.device is None else sorted(int(rank) for rank in node.device),
                "stream": node.stream,
                "node_meta": node.node_meta.get(
                    _FINGERPRINT_META_OVERRIDE, _fingerprint_value(node.node_meta)
                ),
            }
            for node in sorted(dag.nodes.values(), key=lambda item: item.uid)
        ],
        "edges": [
            {
                "src_uid": edge.src_uid,
                "dst_uid": edge.dst_uid,
                "dep_kind": edge.dep_kind,
                "tensor_name": edge.tensor_name,
            }
            for edge in sorted(
                dag.edges,
                key=lambda item: (
                    item.src_uid,
                    item.dst_uid,
                    item.dep_kind,
                    "" if item.tensor_name is None else item.tensor_name,
                ),
            )
        ],
    }


def finalized_dag_fingerprint(dag: TrainingDAG) -> str:
    """Return a deterministic fingerprint of a finalized, unprofiled DAG."""

    encoded = json.dumps(
        _dag_fingerprint_payload(dag), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def finalized_training_plan_fingerprint(per_pp_dags: Sequence[TrainingDAG]) -> str:
    encoded = json.dumps(
        [finalized_dag_fingerprint(dag) for dag in per_pp_dags],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_UNSERIALIZABLE = object()


def _json_safe_value(value: Any) -> Any:
    """Keep plain metadata while dropping live executor/FX/tensor objects."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda kv: str(kv[0])):
            if str(key) in _FINGERPRINT_EXCLUDED_META or str(key) in {
                "gm",
                "graphargs",
                "triton_constant_args",
                _FINGERPRINT_META_OVERRIDE,
            }:
                continue
            converted = _json_safe_value(item)
            if converted is not _UNSERIALIZABLE:
                result[str(key)] = converted
        return result
    if isinstance(value, (list, tuple)):
        converted = [_json_safe_value(item) for item in value]
        return [item for item in converted if item is not _UNSERIALIZABLE]
    if isinstance(value, (set, frozenset)):
        converted = [_json_safe_value(item) for item in value]
        kept = [item for item in converted if item is not _UNSERIALIZABLE]
        return sorted(kept, key=lambda item: json.dumps(item, sort_keys=True))
    return _UNSERIALIZABLE


def get_node_profile_times(
    node: TrainingDAGNode, mode: str = INTERFERENCE_FREE_PROFILE_MODE
) -> dict[int, NodeProfile]:
    """Return rank-keyed node profiles for ``mode``.

    Profiles must be installed explicitly under their semantic mode name.
    """

    mode = normalize_profile_mode(mode)
    by_mode = getattr(node, "profile_times_by_mode", {})
    return by_mode.get(mode, {})


def set_node_profile_times(
    node: TrainingDAGNode,
    mode: str,
    profiles: Mapping[int, NodeProfile],
) -> None:
    """Install one complete rank-keyed timing dataset on a node."""

    mode = normalize_profile_mode(mode)
    installed = {int(rank): profile for rank, profile in profiles.items()}
    node.profile_times_by_mode[mode] = installed


def get_prologue_profile_times(
    dag: TrainingDAG, mode: str = INTERFERENCE_FREE_PROFILE_MODE
) -> dict[int, NodeProfile]:
    """Return rank-keyed prologue profiles for ``mode``."""

    mode = normalize_profile_mode(mode)
    by_mode = getattr(dag, "prologue_profile_times_by_mode", {})
    return by_mode.get(mode, {})


def set_prologue_profile_times(
    dag: TrainingDAG,
    mode: str,
    profiles: Mapping[int, NodeProfile],
) -> None:
    """Install one complete rank-keyed prologue timing dataset."""

    mode = normalize_profile_mode(mode)
    installed = {int(rank): profile for rank, profile in profiles.items()}
    dag.prologue_profile_times_by_mode[mode] = installed


def profile_modes_in_dags(per_pp_dags: Sequence[TrainingDAG]) -> tuple[str, ...]:
    """Return the named datasets present anywhere in a training plan."""

    modes: set[str] = set()
    for dag in per_pp_dags:
        modes.update(getattr(dag, "prologue_profile_times_by_mode", {}))
        for node in dag.nodes.values():
            modes.update(getattr(node, "profile_times_by_mode", {}))
    return tuple(sorted(modes))


def training_dag_to_slim_dict(dag: TrainingDAG) -> dict[str, Any]:
    return {
        "fingerprint": finalized_dag_fingerprint(dag),
        "nodes": [
            {
                "uid": node.uid,
                "node_kind": node.node_kind,
                "compute_subkind": node.compute_subkind,
                "tag": dict(node.tag),
                "device": None if node.device is None else list(node.device),
                "stream": node.stream,
                "node_meta": _json_safe_value(node.node_meta),
                # The slim artifact excludes GraphModules/tensors.  Retain
                # their canonical signature separately so loading and saving
                # does not weaken or change the finalized-graph fingerprint.
                "fingerprint_node_meta": node.node_meta.get(
                    _FINGERPRINT_META_OVERRIDE, _fingerprint_value(node.node_meta)
                ),
                "profile_times_by_mode": {
                    mode: {
                        str(rank): profile.to_dict()
                        for rank, profile in sorted(
                            get_node_profile_times(node, mode).items()
                        )
                    }
                    for mode in sorted(node.profile_times_by_mode)
                },
            }
            for node in sorted(dag.nodes.values(), key=lambda item: item.uid)
        ],
        "edges": [
            asdict(edge)
            for edge in sorted(
                dag.edges,
                key=lambda item: (
                    item.src_uid,
                    item.dst_uid,
                    item.dep_kind,
                    "" if item.tensor_name is None else item.tensor_name,
                ),
            )
        ],
        "prologue_profile_times_by_mode": {
            mode: {
                str(rank): profile.to_dict()
                for rank, profile in sorted(
                    get_prologue_profile_times(dag, mode).items()
                )
            }
            for mode in sorted(dag.prologue_profile_times_by_mode)
        },
    }


def training_dag_from_slim_dict(value: Mapping[str, Any]) -> TrainingDAG:
    dag = TrainingDAG()
    for raw in value.get("nodes", ()):
        node_meta = dict(raw.get("node_meta", {}))
        if "fingerprint_node_meta" in raw:
            node_meta[_FINGERPRINT_META_OVERRIDE] = raw["fingerprint_node_meta"]
        raw_by_mode = raw["profile_times_by_mode"]
        parsed_by_mode = {
            normalize_profile_mode(mode): {
                int(rank): NodeProfile.from_dict(profile)
                for rank, profile in profiles.items()
            }
            for mode, profiles in raw_by_mode.items()
        }
        dag.add_node(
            TrainingDAGNode(
                uid=str(raw["uid"]),
                node_kind=str(raw["node_kind"]),
                compute_subkind=raw.get("compute_subkind"),
                tag=dict(raw.get("tag", {})),
                device=(
                    None
                    if raw.get("device") is None
                    else [int(rank) for rank in raw.get("device", ())]
                ),
                stream=str(raw.get("stream", "default_stream")),
                node_meta=node_meta,
                profile_times_by_mode=parsed_by_mode,
            )
        )
    for raw in value.get("edges", ()):
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=str(raw["src_uid"]),
                dst_uid=str(raw["dst_uid"]),
                dep_kind=str(raw["dep_kind"]),
                tensor_name=raw.get("tensor_name"),
            )
        )
    raw_prologue_by_mode = value["prologue_profile_times_by_mode"]
    dag.prologue_profile_times_by_mode = {
        normalize_profile_mode(mode): {
            int(rank): NodeProfile.from_dict(profile)
            for rank, profile in profiles.items()
        }
        for mode, profiles in raw_prologue_by_mode.items()
    }
    expected = value.get("fingerprint")
    actual = finalized_dag_fingerprint(dag)
    if expected is not None and str(expected) != actual:
        raise ValueError(
            "slim TrainingDAG fingerprint mismatch: "
            f"artifact={expected}, reconstructed={actual}"
        )
    return dag


@dataclass
class ProfiledTrainingPlan:
    """Versioned, portable input to normal/interference-free simulation."""

    per_pp_dags: list[TrainingDAG]
    graph_fingerprint: str
    profile_configs: dict[str, ProfileConfig] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)
    profile_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = PROFILE_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if int(self.schema_version) != PROFILE_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profiled plan schema_version={self.schema_version}; "
                f"expected {PROFILE_PLAN_SCHEMA_VERSION}"
            )
        normalized_configs = {
            normalize_profile_mode(mode): config
            for mode, config in self.profile_configs.items()
        }
        self.profile_configs = normalized_configs
        actual = finalized_training_plan_fingerprint(self.per_pp_dags)
        if self.graph_fingerprint != actual:
            raise ValueError(
                "profiled plan graph fingerprint mismatch: "
                f"provided={self.graph_fingerprint}, actual={actual}"
            )

    @classmethod
    def from_dags(
        cls,
        per_pp_dags: Sequence[TrainingDAG],
        *,
        profile_configs: Mapping[str, ProfileConfig] | None = None,
        topology: Mapping[str, Any] | None = None,
        profile_metadata: Mapping[str, Any] | None = None,
    ) -> ProfiledTrainingPlan:
        dags = list(per_pp_dags)
        configs = dict(profile_configs or {})
        for mode in profile_modes_in_dags(dags):
            configs.setdefault(
                mode,
                ProfileConfig(
                    isolation_scope=(
                        DEFAULT_NORMAL_SCOPE
                        if mode == NORMAL_PROFILE_MODE
                        else DEFAULT_ISOLATION_SCOPE
                    )
                ),
            )
        return cls(
            per_pp_dags=dags,
            graph_fingerprint=finalized_training_plan_fingerprint(dags),
            profile_configs=configs,
            topology=dict(topology or {}),
            profile_metadata=dict(profile_metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "graph_fingerprint": self.graph_fingerprint,
            "profile_configs": {
                mode: asdict(config)
                for mode, config in sorted(self.profile_configs.items())
            },
            "topology": _json_safe_value(self.topology),
            "profile_metadata": _json_safe_value(self.profile_metadata),
            "per_pp_dags": [training_dag_to_slim_dict(dag) for dag in self.per_pp_dags],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProfiledTrainingPlan:
        schema_version = int(value["schema_version"])
        if schema_version != PROFILE_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported profiled plan schema_version={schema_version}; "
                f"expected {PROFILE_PLAN_SCHEMA_VERSION}"
            )
        configs = {
            normalize_profile_mode(mode): ProfileConfig.from_dict(config)
            for mode, config in value["profile_configs"].items()
        }
        return cls(
            schema_version=PROFILE_PLAN_SCHEMA_VERSION,
            graph_fingerprint=str(value["graph_fingerprint"]),
            profile_configs=configs,
            topology=dict(value.get("topology", {})),
            profile_metadata=dict(value.get("profile_metadata", {})),
            per_pp_dags=[
                training_dag_from_slim_dict(dag) for dag in value.get("per_pp_dags", ())
            ],
        )

    @property
    def available_profile_modes(self) -> tuple[str, ...]:
        """Named timing datasets actually present in the plan's DAGs."""

        return profile_modes_in_dags(self.per_pp_dags)

    def config_for_mode(self, mode: str) -> ProfileConfig:
        """Return profiling configuration for one named timing dataset."""

        mode = normalize_profile_mode(mode)
        try:
            return self.profile_configs[mode]
        except KeyError as error:
            raise ValueError(f"profile plan has no config for mode={mode!r}") from error

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> ProfiledTrainingPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _dag_for_actor_rank(
    per_pp_dags: Sequence[TrainingDAG], result: ActorProfileResult
) -> TrainingDAG:
    # pp_rank is the unambiguous primary key.  Check rank membership as a
    # topology guard, but tolerate small unit-test DAGs without placement.
    if 0 <= result.pp_rank < len(per_pp_dags):
        candidate = per_pp_dags[result.pp_rank]
        placed_ranks = {
            int(rank)
            for node in candidate.nodes.values()
            for rank in (node.device or ())
        }
        if not placed_ranks or result.global_rank in placed_ranks:
            return candidate

    matches = [
        dag
        for dag in per_pp_dags
        if any(
            result.global_rank in (node.device or ())
            for node in dag.nodes.values()
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"could not resolve exactly one PP DAG for global_rank={result.global_rank}, "
            f"pp_rank={result.pp_rank}; matches={len(matches)}"
        )
    return matches[0]


def merge_actor_profiles(
    per_pp_dags: Sequence[TrainingDAG],
    results: Sequence[ActorProfileResult],
    *,
    profile_configs: Mapping[str, ProfileConfig] | None = None,
    topology: Mapping[str, Any] | None = None,
) -> ProfiledTrainingPlan:
    """Validate actor shards and attach rank-specific profiles to DAG nodes.

    The operation mutates ``per_pp_dags`` only after a shard has passed its
    rank, fingerprint, and node-coverage checks.  Missing profile shards are
    intentionally not filled with zero durations.
    """

    dags = list(per_pp_dags)
    results = list(results)

    # Normalize every actor's CUDA-event offsets to the earliest calibrated
    # CUDA origin in the same mode/iteration.  The epoch itself is irrelevant;
    # only cross-actor differences survive in the durable profile.
    results_by_mode: dict[str, list[ActorProfileResult]] = {}
    for result in results:
        results_by_mode.setdefault(
            normalize_profile_mode(result.profile_mode), []
        ).append(result)
    aligned_bases_by_mode: dict[str, tuple[float, ...]] = {}
    alignment_metadata: dict[str, Any] = {}
    for mode, mode_results in sorted(results_by_mode.items()):
        has_origins = [bool(result.timeline_origin_samples_us) for result in mode_results]
        if any(has_origins) and not all(has_origins):
            raise ValueError(
                f"profile mode {mode!r} mixes calibrated and uncalibrated actor timelines"
            )
        if not all(has_origins):
            continue
        sample_counts = {
            len(result.timeline_origin_samples_us) for result in mode_results
        }
        if len(sample_counts) != 1:
            raise ValueError(
                f"profile mode {mode!r} has mismatched calibrated-origin sample counts: "
                f"{sorted(sample_counts)}"
            )
        sample_count = next(iter(sample_counts))
        if sample_count <= 0:
            continue
        for result in mode_results:
            uncertainties = result.timeline_origin_uncertainty_samples_us
            if uncertainties and len(uncertainties) != sample_count:
                raise ValueError(
                    f"profile mode {mode!r} rank {result.global_rank} has "
                    "mismatched origin and uncertainty sample counts"
                )
        bases = tuple(
            min(
                float(result.timeline_origin_samples_us[index])
                for result in mode_results
            )
            for index in range(sample_count)
        )
        aligned_bases_by_mode[mode] = bases
        uncertainty_values = [
            float(value)
            for result in mode_results
            for value in result.timeline_origin_uncertainty_samples_us
        ]
        alignment_metadata[mode] = {
            "actors": len(mode_results),
            "samples": sample_count,
            "max_origin_uncertainty_us": (
                max(uncertainty_values) if uncertainty_values else None
            ),
        }

    def aligned_profile(
        result: ActorProfileResult, profile: NodeProfile
    ) -> NodeProfile:
        bases = aligned_bases_by_mode.get(result.profile_mode)
        if bases is None or not profile.executed:
            return profile
        local_offsets = profile.start_offset_samples_us
        origins = result.timeline_origin_samples_us
        if len(local_offsets) != len(origins):
            raise ValueError(
                f"profile mode {result.profile_mode!r} rank {result.global_rank} "
                f"node sample count={len(profile.samples_us)} has "
                f"start offsets={len(local_offsets)} and calibrated origins={len(origins)}"
            )
        global_offsets = tuple(
            float(origin) - float(base) + float(local_offset)
            for origin, base, local_offset in zip(origins, bases, local_offsets)
        )
        return replace(
            profile,
            start_offset_samples_us=global_offsets,
            start_offset_estimate_us=estimate_profile_time(
                global_offsets, profile.estimator
            ),
        )
    seen_shards: set[tuple[str, int]] = set()
    device_metadata: dict[str, dict[str, Any]] = {}
    for result in sorted(
        results, key=lambda item: (item.profile_mode, item.global_rank)
    ):
        mode = normalize_profile_mode(result.profile_mode)
        rank = int(result.global_rank)
        shard_key = (mode, rank)
        if shard_key in seen_shards:
            raise ValueError(
                "duplicate actor profile result for "
                f"profile_mode={mode}, global_rank={rank}"
            )
        seen_shards.add(shard_key)
        dag = _dag_for_actor_rank(dags, result)
        expected_fingerprint = finalized_dag_fingerprint(dag)
        if result.graph_fingerprint != expected_fingerprint:
            raise ValueError(
                f"profile fingerprint mismatch for global_rank={rank}: "
                f"actor={result.graph_fingerprint}, dag={expected_fingerprint}"
            )

        supplied = set(result.node_profiles)
        expected = set(dag.nodes)
        missing = sorted(expected - supplied)
        unknown = sorted(supplied - expected)
        if missing or unknown:
            raise ValueError(
                f"profile node coverage mismatch for global_rank={rank}: "
                f"missing={missing}, unknown={unknown}"
            )

        for uid, raw_profile in result.node_profiles.items():
            profile = aligned_profile(result, raw_profile)
            node = dag.nodes[uid]
            if node.device is not None and rank not in node.device:
                raise ValueError(
                    f"profile rank {rank} does not execute node {uid}; node devices={node.device}"
                )
            profiles = get_node_profile_times(node, mode)
            if mode not in node.profile_times_by_mode:
                set_node_profile_times(node, mode, profiles)
                profiles = get_node_profile_times(node, mode)
            if rank in profiles:
                raise ValueError(
                    f"duplicate profile for mode={mode}, node={uid}, global_rank={rank}"
                )
            profiles[rank] = profile
        if result.prologue_profile is not None:
            prologue_profiles = get_prologue_profile_times(dag, mode)
            if mode not in dag.prologue_profile_times_by_mode:
                set_prologue_profile_times(dag, mode, prologue_profiles)
                prologue_profiles = get_prologue_profile_times(dag, mode)
            if rank in prologue_profiles:
                raise ValueError(
                    "duplicate prologue profile for "
                    f"mode={mode}, global_rank={rank}"
                )
            prologue_profiles[rank] = result.prologue_profile
        device_metadata.setdefault(mode, {})[str(rank)] = _json_safe_value(
            result.device_metadata
        )

    profile_metadata: dict[str, Any] = {"actors": device_metadata}
    if alignment_metadata:
        profile_metadata["timeline_alignment"] = {
            "method": "ray_ntp_midpoint_plus_cuda_anchor",
            "origin": "earliest_actor_cuda_prologue_per_mode_iteration",
            "modes": alignment_metadata,
        }

    return ProfiledTrainingPlan.from_dags(
        dags,
        profile_configs=profile_configs,
        topology=topology,
        profile_metadata=profile_metadata,
    )
