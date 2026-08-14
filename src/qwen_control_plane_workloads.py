"""Realistic Qwen3.5-397B DualPipeV DAGs for CPU-only control-plane tests.

The model is traced on PyTorch's meta device.  Piper's normal TrainingDAG
builder and schedule-directive lowering then produce the finalized per-actor
DAGs, including PP P2P, DP ZeRO-3, EP A2A, bucketing, microbatch expansion,
DualPipeV ordering, stream serialization, and ZeRO lifetime dependencies.

Before a DAG leaves the factory, all FX/model/tensor payload is removed.  The
remaining graph is a compact control-plane workload: real node kinds, tags,
streams, communication metadata, and dependencies without executable model
state.
"""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
from typing import Any

import torch
import torch.fx as fx

from examples.build_schedule import build_dualpipev_schedule
from examples.models.qwen3 import PiperQwen3Model, create_qwen3_config

from .dag import TrainingDAG, build_training_dag
from .directives import (
    _validate_schedule_tags_exist,
    apply_communication_directives_to_rank_dags,
    apply_order_directives,
    apply_schedule_directives,
)
from .fx import split_gm_by_annotations
from .ordering import resolve_total_order_per_stream
from .piper import (
    _reset_annotation_state,
    _split_global_training_dag_by_pp_rank,
)
from .zero import _add_inter_chain_temporal_edges, _prune_zero_lifetime_metadata


DEFAULT_QWEN_MODEL = "Qwen3.5-397B-A17B"
COLLECTIVE_STREAM_POLICIES = ("shared", "separate")
_CONTROL_META_KEYS = frozenset(
    {
        "bucket_key",
        "comm_group_kind",
        "comm_id",
        "participants",
        "peer_pp_rank",
        "direction",
        "bucket_idx",
        "num_buckets",
    }
)


@dataclass(frozen=True)
class QwenDualPipeWorkloadSpec:
    """One realistic schedule point in the control-plane matrix.

    Piper's current placement language represents DP and EP over the same
    per-stage device group.  Consequently ``ep_degree`` may be 1 (replicated
    MoE) or equal to ``dp_degree`` (the DP=EP co-group used by Piper's checked-in
    PP/DP/EP schedules).
    """

    pp_degree: int
    dp_degree: int
    ep_degree: int
    bucket_size_mb: int | None
    collective_stream_policy: str
    model_name: str = DEFAULT_QWEN_MODEL
    microbatch_count: int | None = None
    shard_grads: bool = True
    shard_params: bool = True

    @property
    def virtual_stage_count(self) -> int:
        return 2 * self.pp_degree

    @property
    def resolved_microbatch_count(self) -> int:
        return (
            2 * self.pp_degree
            if self.microbatch_count is None
            else self.microbatch_count
        )

    @property
    def world_size(self) -> int:
        # EP is a different partitioning of the same stage device group, not
        # another Cartesian dimension in Piper's current schedule language.
        return self.pp_degree * self.dp_degree

    @property
    def case_name(self) -> str:
        bucket = "none" if self.bucket_size_mb is None else str(self.bucket_size_mb)
        return (
            f"qwen397b_dualpipev_pp{self.pp_degree}_dp{self.dp_degree}_"
            f"ep{self.ep_degree}_bucket{bucket}_{self.collective_stream_policy}"
        )

    def validate(self, *, model_layers: int | None = None) -> None:
        for name, value in (
            ("pp_degree", self.pp_degree),
            ("dp_degree", self.dp_degree),
            ("ep_degree", self.ep_degree),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.ep_degree not in {1, self.dp_degree}:
            raise ValueError(
                "Piper's current schedule IR supports ep_degree=1 or "
                "ep_degree=dp_degree for this workload"
            )
        if self.bucket_size_mb is not None and self.bucket_size_mb <= 0:
            raise ValueError("bucket_size_mb must be positive or None")
        if self.collective_stream_policy not in COLLECTIVE_STREAM_POLICIES:
            raise ValueError(
                "collective_stream_policy must be one of "
                f"{COLLECTIVE_STREAM_POLICIES}"
            )
        if self.resolved_microbatch_count < 2 * self.pp_degree:
            raise ValueError(
                "DualPipeV requires microbatch_count >= 2 * pp_degree"
            )
        if model_layers is not None and self.virtual_stage_count > model_layers:
            raise ValueError(
                f"Qwen proxy has {model_layers} traced layers, but PP={self.pp_degree} "
                f"requires {self.virtual_stage_count} non-empty V stages"
            )


@dataclass
class QwenControlPlaneWorkload:
    spec: QwenDualPipeWorkloadSpec
    per_actor_dags: list[TrainingDAG]
    metadata: dict[str, Any]


_DAG_CACHE_SCHEMA_VERSION = 1


def _dag_cache_paths(
    cache_dir: Path,
    spec: QwenDualPipeWorkloadSpec,
) -> tuple[Path, Path]:
    return (
        cache_dir / f"{spec.case_name}.training_dag.pkl",
        cache_dir / f"{spec.case_name}.manifest.json",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_qwen_control_plane_workload(
    workload: QwenControlPlaneWorkload,
    cache_dir: Path,
) -> dict[str, Any]:
    """Atomically persist one stripped physical-PP-rank-0 TrainingDAG."""

    serialization_start = time.perf_counter()
    if len(workload.per_actor_dags) != 1:
        raise ValueError("DAG cache expects exactly physical PP rank 0")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dag_path, manifest_path = _dag_cache_paths(cache_dir, workload.spec)
    payload = {
        "schema_version": _DAG_CACHE_SCHEMA_VERSION,
        "spec": asdict(workload.spec),
        "metadata": copy.deepcopy(workload.metadata),
        "physical_pp_rank": 0,
        "training_dag": workload.per_actor_dags[0],
    }
    tmp_dag = dag_path.with_name(f".{dag_path.name}.{os.getpid()}.tmp")
    with tmp_dag.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_dag, dag_path)
    cache_info = {
        "schema_version": _DAG_CACHE_SCHEMA_VERSION,
        "case_name": workload.spec.case_name,
        "spec": asdict(workload.spec),
        "physical_pp_rank": 0,
        "model_layers": workload.metadata["model_config"]["n_layers"],
        "node_count": len(workload.per_actor_dags[0].nodes),
        "edge_count": len(workload.per_actor_dags[0].edges),
        "pickle_file": dag_path.name,
        "pickle_bytes": dag_path.stat().st_size,
        "pickle_sha256": _sha256_file(dag_path),
        "serialization_time_s": time.perf_counter() - serialization_start,
    }
    tmp_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    tmp_manifest.write_text(
        json.dumps(cache_info, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_manifest, manifest_path)
    workload.metadata["dag_cache"] = {
        **cache_info,
        "cache_dir": str(cache_dir.resolve()),
    }
    return workload.metadata["dag_cache"]


def load_qwen_control_plane_workload(
    spec: QwenDualPipeWorkloadSpec,
    cache_dir: Path,
) -> QwenControlPlaneWorkload:
    """Load and validate one previously persisted rank-0 DAG."""

    dag_path, manifest_path = _dag_cache_paths(cache_dir, spec)
    if not dag_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing cached DAG for {spec.case_name}; run --mode generate first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != _DAG_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported DAG cache manifest: {manifest_path}")
    if manifest.get("spec") != asdict(spec):
        raise ValueError(f"cached DAG spec mismatch: {manifest_path}")
    actual_sha256 = _sha256_file(dag_path)
    if actual_sha256 != manifest.get("pickle_sha256"):
        raise ValueError(f"cached DAG checksum mismatch: {dag_path}")
    with dag_path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("schema_version") != _DAG_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported cached DAG payload: {dag_path}")
    if payload.get("spec") != asdict(spec) or payload.get("physical_pp_rank") != 0:
        raise ValueError(f"cached DAG payload metadata mismatch: {dag_path}")
    dag = payload["training_dag"]
    if len(dag.nodes) != manifest.get("node_count"):
        raise ValueError(f"cached DAG node count mismatch: {dag_path}")
    if len(dag.edges) != manifest.get("edge_count"):
        raise ValueError(f"cached DAG edge count mismatch: {dag_path}")
    metadata = payload["metadata"]
    metadata["dag_cache"] = {
        **manifest,
        "cache_dir": str(cache_dir.resolve()),
    }
    return QwenControlPlaneWorkload(
        spec=spec,
        per_actor_dags=[dag],
        metadata=metadata,
    )


def build_qwen_dualpipe_schedule(
    spec: QwenDualPipeWorkloadSpec,
) -> list[dict[str, Any]]:
    """Build the same directive sequence used by a real DualPipeV run."""

    config = create_qwen3_config(spec.model_name)
    spec.validate(model_layers=int(config.n_layers))
    stage_devices: dict[int, list[int]] = {}
    for stage in range(spec.virtual_stage_count):
        physical_pp_rank = (
            stage
            if stage < spec.pp_degree
            else 2 * spec.pp_degree - 1 - stage
        )
        stage_devices[stage] = [
            physical_pp_rank + replica * spec.pp_degree
            for replica in range(spec.dp_degree)
        ]

    directives: list[dict[str, Any]] = []
    for stage, devices in stage_devices.items():
        directives.append(
            {
                "op": "place",
                "filter": {"PP": stage},
                "devices": devices,
                "stream": "pp_stream",
            }
        )

    if spec.dp_degree > 1:
        if spec.collective_stream_policy == "shared":
            dp_stream = "collective_stream"
        else:
            dp_stream = "dp_stream"
        for stage, devices in stage_devices.items():
            replicate: dict[str, Any] = {
                "op": "replicate",
                "filter": {
                    "PP": stage,
                    **({"EP": "-"} if spec.ep_degree > 1 else {}),
                },
                "devices": devices,
                "gather_stream": dp_stream,
                "reduce_stream": dp_stream,
                "shard_grads": bool(spec.shard_grads),
                "shard_params": bool(spec.shard_params),
            }
            if spec.bucket_size_mb is not None:
                replicate["bucket_size"] = spec.bucket_size_mb
            directives.append(replicate)

    if spec.ep_degree > 1:
        ep_stream = (
            "collective_stream"
            if spec.collective_stream_policy == "shared"
            else "ep_stream"
        )
        for stage, devices in stage_devices.items():
            directives.append(
                {
                    "op": "shard",
                    "filter": {"PP": stage, "EP": "*"},
                    "devices": devices,
                    "stream": ep_stream,
                }
            )

    directives.append(
        {
            "op": "split",
            "filter": {},
            "dim_name": "MB",
            "num_microbatches": spec.resolved_microbatch_count,
        }
    )
    directives.extend(
        build_dualpipev_schedule(
            spec.pp_degree,
            spec.resolved_microbatch_count,
        )
    )
    return directives


def strip_training_dag_payload(training_dag: TrainingDAG) -> TrainingDAG:
    """Remove every model/FX/tensor payload while retaining control topology."""

    for node in training_dag.nodes.values():
        meta = getattr(node, "node_meta", {}) or {}
        node.node_meta = {
            key: copy.deepcopy(value)
            for key, value in meta.items()
            if key in _CONTROL_META_KEYS
        }
        node.profile_times_by_mode.clear()
    training_dag.prologue_profile_times_by_mode.clear()
    return training_dag


def summarize_training_dags(dags: list[TrainingDAG]) -> dict[str, Any]:
    node_counts = [len(dag.nodes) for dag in dags]
    edge_counts = [len(dag.edges) for dag in dags]
    node_kinds: Counter[str] = Counter()
    compute_subkinds: Counter[str] = Counter()
    streams: Counter[str] = Counter()
    edge_kinds: Counter[str] = Counter()
    for dag in dags:
        for node in dag.nodes.values():
            node_kinds[str(node.node_kind)] += 1
            if node.compute_subkind is not None:
                compute_subkinds[str(node.compute_subkind)] += 1
            streams[str(node.stream)] += 1
        edge_kinds.update(edge.dep_kind for edge in dag.edges)
    return {
        "actor_dag_count": len(dags),
        "nodes_per_physical_pp_rank": node_counts,
        "edges_per_physical_pp_rank": edge_counts,
        "pp_rank_0_nodes": node_counts[0],
        "pp_rank_0_edges": edge_counts[0],
        "total_nodes_across_unique_pp_dags": sum(node_counts),
        "total_edges_across_unique_pp_dags": sum(edge_counts),
        "nodes_per_actor_min": min(node_counts),
        "nodes_per_actor_max": max(node_counts),
        "nodes_per_actor_mean": sum(node_counts) / len(node_counts),
        "edges_per_actor_min": min(edge_counts),
        "edges_per_actor_max": max(edge_counts),
        "edges_per_actor_mean": sum(edge_counts) / len(edge_counts),
        "node_kinds": dict(sorted(node_kinds.items())),
        "compute_subkinds": dict(sorted(compute_subkinds.items())),
        "streams": dict(sorted(streams.items())),
        "edge_kinds": dict(sorted(edge_kinds.items())),
    }


class QwenTrainingDAGFactory:
    """Trace each V-stage count once and lower many schedule variants."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_QWEN_MODEL,
        batch_size: int = 1,
        sequence_length: int = 32768,
    ) -> None:
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")
        self.model_name = model_name
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self._base_dags: dict[tuple[int, int], TrainingDAG] = {}
        self._trace_times_s: dict[tuple[int, int], float] = {}
        self._lowered_templates: dict[tuple[Any, ...], QwenControlPlaneWorkload] = {}

    @staticmethod
    def _template_key(spec: QwenDualPipeWorkloadSpec) -> tuple[Any, ...]:
        return (
            spec.model_name,
            spec.pp_degree,
            spec.bucket_size_mb,
            spec.collective_stream_policy,
            spec.resolved_microbatch_count,
            bool(spec.dp_degree > 1),
            bool(spec.ep_degree > 1),
            spec.shard_grads,
            spec.shard_params,
        )

    @staticmethod
    def _retarget_template(
        template: QwenControlPlaneWorkload,
        spec: QwenDualPipeWorkloadSpec,
        *,
        build_start: float,
    ) -> QwenControlPlaneWorkload:
        retarget_start = time.perf_counter()
        workload = copy.deepcopy(template)
        source_spec = workload.spec
        if source_spec.pp_degree != spec.pp_degree:
            raise ValueError("structural template PP degree mismatch")

        def device_group(physical_pp_rank: int) -> list[int]:
            return [
                physical_pp_rank + replica * spec.pp_degree
                for replica in range(spec.dp_degree)
            ]

        for node in workload.per_actor_dags[0].nodes.values():
            if node.device is None:
                continue
            physical_pp_rank = min(int(device) for device in node.device) % spec.pp_degree
            node.device = device_group(physical_pp_rank)
            if "participants" not in node.node_meta:
                continue
            if node.node_kind in {"SEND_COMM", "RECV_COMM"}:
                peer_pp_rank = int(node.node_meta["peer_pp_rank"])
                node.node_meta["participants"] = sorted(
                    set(device_group(physical_pp_rank) + device_group(peer_pp_rank))
                )
            else:
                node.node_meta["participants"] = list(node.device)

        workload.spec = spec
        metadata = workload.metadata
        metadata["case_name"] = spec.case_name
        metadata["spec"] = asdict(spec)
        metadata["world_size"] = spec.world_size
        metadata["trace_cache_hit"] = True
        metadata["structural_template_cache_hit"] = True
        metadata["structural_template_source_case"] = source_spec.case_name
        metadata["structural_template_source_lowering_time_s"] = metadata.get(
            "lowering_time_s"
        )
        metadata["lowering_time_s"] = time.perf_counter() - retarget_start
        metadata["dag_build_time_s"] = time.perf_counter() - build_start
        return workload

    def _trace_base_dag(
        self,
        virtual_stage_count: int,
        expert_shard_degree: int,
    ) -> TrainingDAG:
        trace_key = (virtual_stage_count, expert_shard_degree)
        cached = self._base_dags.get(trace_key)
        if cached is not None:
            return cached

        config = create_qwen3_config(self.model_name)
        if virtual_stage_count > int(config.n_layers):
            raise ValueError(
                f"virtual_stage_count={virtual_stage_count} exceeds traced "
                f"layers={config.n_layers}"
            )
        captured: dict[str, TrainingDAG] = {}

        def capture_backend(
            gm: fx.GraphModule,
            _example_inputs: list[Any],
        ) -> Any:
            _top_level_gm, segments = split_gm_by_annotations(gm)
            if not segments:
                raise RuntimeError("Qwen trace produced no Piper annotation segments")
            captured["dag"] = build_training_dag(segments)
            return gm.forward

        start = time.perf_counter()
        torch._dynamo.reset()
        with torch.device("meta"):
            model = PiperQwen3Model(
                config,
                virtual_stage_count,
                expert_shard_degree,
            ).to(torch.bfloat16)
            tokens = torch.empty(
                (self.batch_size, self.sequence_length),
                dtype=torch.int64,
                device="meta",
            )
            torch._dynamo.mark_dynamic(tokens, 1)
        # This is a meta-device graph capture with a custom backend. Dynamo
        # still must recognize Triton JITFunctions so it records wrapper HOPs,
        # but no driver or kernel execution is needed. A CPU PyTorch wheel
        # reports ``has_triton() == False`` because no capable device is
        # active, which makes Dynamo fall through into an attempted Triton
        # driver call. Enable only the capture-time classification here.
        from torch.utils import _triton

        previous_has_triton = _triton.has_triton
        _triton.has_triton = lambda: True
        try:
            compiled = torch.compile(model, backend=capture_backend, fullgraph=True)
            _reset_annotation_state()
            compiled(tokens)
        finally:
            _triton.has_triton = previous_has_triton
        try:
            base_dag = captured["dag"]
        except KeyError as exc:
            raise RuntimeError("torch.compile did not invoke the capture backend") from exc
        self._trace_times_s[trace_key] = time.perf_counter() - start
        self._base_dags[trace_key] = base_dag
        return base_dag

    def build(self, spec: QwenDualPipeWorkloadSpec) -> QwenControlPlaneWorkload:
        build_start = time.perf_counter()
        if spec.model_name != self.model_name:
            raise ValueError(
                f"factory model={self.model_name!r}, workload model={spec.model_name!r}"
            )
        config = create_qwen3_config(spec.model_name)
        spec.validate(model_layers=int(config.n_layers))
        template_key = self._template_key(spec)
        template = self._lowered_templates.get(template_key)
        if template is not None and template.spec != spec:
            return self._retarget_template(
                template,
                spec,
                build_start=build_start,
            )
        trace_key = (spec.virtual_stage_count, spec.ep_degree)
        trace_cache_hit = trace_key in self._base_dags
        base_dag = self._trace_base_dag(
            spec.virtual_stage_count,
            spec.ep_degree,
        )
        directives = build_qwen_dualpipe_schedule(spec)

        lower_start = time.perf_counter()
        training_dag = copy.deepcopy(base_dag)
        _validate_schedule_tags_exist(training_dag, directives)
        apply_schedule_directives(
            training_dag,
            directives,
            defer_order=True,
            defer_communication=True,
        )
        all_pp_dags = _split_global_training_dag_by_pp_rank(
            training_dag,
            merge_same_device_components=True,
        )
        if len(all_pp_dags) != spec.pp_degree:
            raise RuntimeError(
                f"expected {spec.pp_degree} physical PP DAGs, got "
                f"{len(all_pp_dags)}"
            )
        apply_communication_directives_to_rank_dags(all_pp_dags, directives)

        # The scalability study intentionally controls for rank-to-rank DAG
        # variation by timing only physical PP rank 0.  The complete global
        # DualPipeV schedule is still lowered and split above, so this DAG owns
        # its real pair of logical V stages (0 and 2*PP-1).
        rank_zero_dag = all_pp_dags[0]
        apply_order_directives(
            rank_zero_dag,
            directives,
            skip_unmatched_directives=True,
        )
        zero_chains = _prune_zero_lifetime_metadata(rank_zero_dag)
        resolve_total_order_per_stream(rank_zero_dag)
        _add_inter_chain_temporal_edges(rank_zero_dag, zero_chains)
        strip_training_dag_payload(rank_zero_dag)
        per_actor_dags = [rank_zero_dag]
        lowering_time_s = time.perf_counter() - lower_start
        summary = summarize_training_dags(per_actor_dags)
        directive_counts = Counter(
            str(directive.get("op", "<missing>")) for directive in directives
        )
        metadata = {
            "case_name": spec.case_name,
            "spec": asdict(spec),
            "model_config": {
                "name": spec.model_name,
                "dim": int(config.dim),
                "n_layers": int(config.n_layers),
                "n_heads": int(config.n_heads),
                "num_experts": int(config.moe_args.num_experts),
                "top_k": int(config.moe_args.top_k),
            },
            "world_size": spec.world_size,
            "declared_physical_pp_rank_count": spec.pp_degree,
            "generated_physical_pp_ranks": [0],
            "virtual_stage_count": spec.virtual_stage_count,
            "microbatch_count": spec.resolved_microbatch_count,
            "trace_time_s": self._trace_times_s[trace_key],
            "trace_cache_hit": trace_cache_hit,
            "structural_template_cache_hit": False,
            "lowering_time_s": lowering_time_s,
            "directive_count": len(directives),
            "directive_counts": dict(sorted(directive_counts.items())),
            "dag_summary": summary,
            "contains_model_state": False,
            "contains_fx_graphs": False,
            "uses_cuda": False,
        }
        metadata["dag_build_time_s"] = time.perf_counter() - build_start
        workload = QwenControlPlaneWorkload(
            spec=spec,
            per_actor_dags=per_actor_dags,
            metadata=metadata,
        )
        self._lowered_templates.setdefault(template_key, copy.deepcopy(workload))
        return workload


__all__ = [
    "COLLECTIVE_STREAM_POLICIES",
    "DEFAULT_QWEN_MODEL",
    "QwenControlPlaneWorkload",
    "QwenDualPipeWorkloadSpec",
    "QwenTrainingDAGFactory",
    "build_qwen_dualpipe_schedule",
    "load_qwen_control_plane_workload",
    "save_qwen_control_plane_workload",
    "strip_training_dag_payload",
    "summarize_training_dags",
]
