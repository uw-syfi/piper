from __future__ import annotations

import hashlib
import json
from typing import Any

import torch.fx as fx

from examples.build_schedule import (
    build_1f1b_schedule,
    build_dualpipev_schedule,
    build_interleaved_1f1b_schedule,
    build_zerobubble_schedule,
)
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
import src.directives as directives
from src.ordering import resolve_total_order_per_stream
from src.piper import _split_global_training_dag_by_pp_rank
from src.zero import _add_inter_chain_temporal_edges, _prune_zero_lifetime_metadata


_LIVE_META_KEYS = {
    "gm",
    "graphargs",
    "input_idxs",
    "param_idxs",
    "triton_constant_args",
}


def _dummy_gm() -> fx.GraphModule:
    graph = fx.Graph()
    value = graph.placeholder("value")
    graph.output(value)
    return fx.GraphModule({}, graph)


def _base_training_dag(
    logical_stages: int,
    *,
    routed_a2a: bool = False,
) -> TrainingDAG:
    dag = TrainingDAG()
    fwd_uids: list[str] = []
    previous_uid: str | None = None
    for pp in range(logical_stages):
        for segment, ep in enumerate((None, 0, None)):
            uid = f"s{pp}.seg{segment}"
            fwd_uids.append(uid)
            boundary_after = {"tensor_idx": 0}
            if routed_a2a and segment == 0:
                boundary_after["routing_counts_idx"] = 1
            dag.add_node(
                TrainingDAGNode(
                    uid=uid,
                    node_kind="COMPUTE",
                    compute_subkind="FWD",
                    tag={"PP": pp, "EP": ep, "PASS": "F"},
                    device=None,
                    stream="default_stream",
                    node_meta={
                        "stage_id": pp,
                        "segment_id": segment,
                        "bucket_key": uid,
                        "gm": _dummy_gm(),
                        "graphargs": [],
                        "input_idxs": [],
                        "param_idxs": [],
                        "a2a_boundary_after": boundary_after,
                    },
                )
            )
            if previous_uid is not None:
                dag.add_edge(TrainingDAGEdge(previous_uid, uid, "data"))
            previous_uid = uid

    for fwd_uid in fwd_uids:
        fwd = dag.nodes[fwd_uid]
        dag.add_node(
            TrainingDAGNode(
                uid=f"{fwd_uid}.bwd",
                node_kind="COMPUTE",
                compute_subkind="BWD",
                tag={**fwd.tag, "PASS": "B"},
                device=None,
                stream="default_stream",
                node_meta={
                    "fwd_uid": fwd_uid,
                    "bucket_key": fwd_uid,
                    "compute_loss": False,
                },
            )
        )
    for edge in list(dag.edges):
        dag.add_edge(
            TrainingDAGEdge(
                f"{edge.dst_uid}.bwd",
                f"{edge.src_uid}.bwd",
                "data",
            )
        )
    dag.add_edge(TrainingDAGEdge(fwd_uids[-1], f"{fwd_uids[-1]}.bwd", "data"))
    dag.nodes[f"{fwd_uids[-1]}.bwd"].node_meta["compute_loss"] = True

    dag.add_node(
        TrainingDAGNode(
            uid="upd.0",
            node_kind="UPD",
            compute_subkind=None,
            tag={"PASS": None},
            device=None,
            stream="default_stream",
            node_meta={},
        )
    )
    for fwd_uid in fwd_uids:
        dag.add_edge(TrainingDAGEdge(f"{fwd_uid}.bwd", "upd.0", "temporal"))
    return dag


def _stage_devices(
    logical_stages: int,
    physical_pp: int,
    dp: int,
    placement: str,
) -> dict[int, list[int]]:
    if placement == "normal":
        physical_rank = lambda stage: stage
    elif placement == "interleaved":
        physical_rank = lambda stage: stage % physical_pp
    elif placement == "v":
        physical_rank = lambda stage: (
            stage if stage < physical_pp else 2 * physical_pp - 1 - stage
        )
    else:  # pragma: no cover - guarded by the case table
        raise ValueError(placement)
    return {
        stage: [physical_rank(stage) + replica * physical_pp for replica in range(dp)]
        for stage in range(logical_stages)
    }


def _schedule(case: dict[str, Any]) -> list[dict[str, Any]]:
    physical_pp = case["physical_pp"]
    placement = case["placement"]
    logical_stages = physical_pp if placement == "normal" else 2 * physical_pp
    devices = _stage_devices(logical_stages, physical_pp, case["dp"], placement)
    schedule: list[dict[str, Any]] = [
        {
            "op": "place",
            "filter": {"PP": stage},
            "devices": stage_devices,
            "stream": "pp_stream",
        }
        for stage, stage_devices in devices.items()
    ]
    if case["dp"] > 1:
        for stage, stage_devices in devices.items():
            replicate: dict[str, Any] = {
                "op": "replicate",
                "filter": {"PP": stage},
                "devices": stage_devices,
                "gather_stream": "dp_stream",
                "reduce_stream": "dp_stream",
            }
            if case["sharding"] == "grads":
                replicate["shard_grads"] = True
            elif case["sharding"] == "params_grads":
                replicate.update(shard_params=True, shard_grads=True)
            if case.get("bucket"):
                replicate["bucket_size"] = 16
            schedule.append(replicate)
    if case["ep"]:
        for stage, stage_devices in devices.items():
            schedule.append(
                {
                    "op": "shard",
                    "filter": {"PP": stage, "EP": "*"},
                    "devices": stage_devices,
                    "stream": "ep_stream",
                }
            )
    schedule.append(
        {
            "op": "split",
            "filter": {},
            "dim_name": "MB",
            "num_microbatches": case["microbatches"],
        }
    )
    if case["schedule"] == "1f1b":
        schedule.extend(build_1f1b_schedule(physical_pp, case["microbatches"]))
    elif case["schedule"] == "zerobubble":
        schedule.extend(build_zerobubble_schedule(physical_pp, case["microbatches"]))
    elif case["schedule"] == "interleaved":
        schedule.extend(
            build_interleaved_1f1b_schedule(
                physical_pp,
                case["microbatches"],
                2,
            )
        )
    elif case["schedule"] == "dualpipev":
        schedule.extend(build_dualpipev_schedule(physical_pp, case["microbatches"]))
    else:  # pragma: no cover - guarded by the case table
        raise ValueError(case["schedule"])
    return schedule


CASES: dict[str, dict[str, Any]] = {
    "pp_only_1f1b": dict(physical_pp=2, dp=1, ep=False, sharding="none", placement="normal", schedule="1f1b", microbatches=4),
    "dp_only_unsharded": dict(physical_pp=1, dp=2, ep=False, sharding="none", placement="normal", schedule="1f1b", microbatches=3),
    "pp_dp_unsharded": dict(physical_pp=2, dp=2, ep=False, sharding="none", placement="normal", schedule="1f1b", microbatches=4),
    "pp_dp_grad_sharded": dict(physical_pp=2, dp=2, ep=False, sharding="grads", placement="normal", schedule="1f1b", microbatches=4),
    "pp_dp_zero3_zerobubble": dict(physical_pp=2, dp=2, ep=False, sharding="params_grads", placement="normal", schedule="zerobubble", microbatches=4),
    "bucketed_zero3_zerobubble": dict(physical_pp=2, dp=2, ep=False, sharding="params_grads", placement="normal", schedule="zerobubble", microbatches=4, bucket=True),
    "pp_dp_ep_1f1b": dict(physical_pp=2, dp=2, ep=True, sharding="params_grads", placement="normal", schedule="1f1b", microbatches=4),
    "interleaved_placement": dict(physical_pp=2, dp=2, ep=True, sharding="grads", placement="interleaved", schedule="interleaved", microbatches=4),
    "v_placement_dualpipev": dict(physical_pp=2, dp=2, ep=True, sharding="params_grads", placement="v", schedule="dualpipev", microbatches=4),
}


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if key not in _LIVE_META_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_normalize(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _canonical_dag(dag: TrainingDAG) -> dict[str, Any]:
    return {
        "nodes": {
            uid: {
                "node_kind": node.node_kind,
                "compute_subkind": node.compute_subkind,
                "tag": _normalize(node.tag),
                "device": node.device,
                "stream": node.stream,
                "node_meta": _normalize(node.node_meta),
            }
            for uid, node in sorted(dag.nodes.items())
        },
        "edges": sorted(
            (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
            for edge in set(dag.edges)
        ),
    }


def _case_digest(
    case: dict[str, Any],
    *,
    rank_local_order: bool = False,
    rank_local_communication: bool = False,
) -> str:
    placement = case["placement"]
    logical_stages = (
        case["physical_pp"] if placement == "normal" else 2 * case["physical_pp"]
    )
    dag = _base_training_dag(logical_stages, routed_a2a=bool(case["ep"]))
    original_bucket_stage = directives.bucket_stage
    if case.get("bucket"):
        directives.bucket_stage = lambda *_args, **_kwargs: [
            (_dummy_gm(), [], [], []),
            (_dummy_gm(), [], [], []),
        ]
    try:
        schedule = _schedule(case)
        directives.apply_schedule_directives(
            dag,
            schedule,
            defer_order=rank_local_order or rank_local_communication,
            defer_communication=rank_local_communication,
        )
    finally:
        directives.bucket_stage = original_bucket_stage

    per_rank = _split_global_training_dag_by_pp_rank(
        dag,
        merge_same_device_components=rank_local_order or rank_local_communication,
    )
    if rank_local_communication:
        directives.apply_communication_directives_to_rank_dags(per_rank, schedule)
    for rank_dag in per_rank:
        if rank_local_order or rank_local_communication:
            directives.apply_order_directives(
                rank_dag,
                schedule,
                skip_unmatched_directives=True,
            )
        chains = _prune_zero_lifetime_metadata(rank_dag)
        resolve_total_order_per_stream(rank_dag)
        _add_inter_chain_temporal_edges(rank_dag, chains)
    payload = [_canonical_dag(rank_dag) for rank_dag in per_rank]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


LEGACY_DIGESTS: dict[str, str] = {
    # Bucketing now preserves the bucket GraphModule's output and
    # differentiability metadata. The executable graph structure is unchanged.
    "bucketed_zero3_zerobubble": "1adb5801bd07d6442592aee4fedbea985ab661ea265e5c1c2718d4017869ebf3",
    "dp_only_unsharded": "5e248c6eb273dbb10c6261aafb3cee2fe55e0da998cb1e6f0977081986f72cf2",
    "interleaved_placement": "59a2117379e4e44a63e8599c001cf5e57841cbcc2dbdd07913a396e96aa8a6e8",
    "pp_dp_ep_1f1b": "6a3845d839c766795a4a799ae47d5dcfcf4111acbb931197f2fa60c6f07dcc2e",
    "pp_dp_grad_sharded": "fdd15fd8eca5897402e89f482197a53e9f209c12bc0a61634343fa3c9f79d969",
    "pp_dp_unsharded": "d81344f46b34e8c8f3cba9238a8a37f9f5b8b382ee88821df65579f7a3263bd3",
    "pp_dp_zero3_zerobubble": "b8759170e8e02a53dabc58e6eddfc10d6cd416f5d0bb706621838d2818688e34",
    "pp_only_1f1b": "eac89ef8094e8acbd18cc296384effa13d22b91de6a59856d9a7bea36c2e79f0",
    "v_placement_dualpipev": "b9510ebd6d7eb5a1c0aa1582805e9495898244ed93536281c4972490a490012a",
}


def test_builder_matches_legacy_schedule_matrix() -> None:
    assert {
        name: _case_digest(case)
        for name, case in CASES.items()
    } == LEGACY_DIGESTS


def test_rank_local_order_lowering_matches_legacy_schedule_matrix() -> None:
    assert {
        name: _case_digest(case, rank_local_order=True)
        for name, case in CASES.items()
    } == LEGACY_DIGESTS


def test_rank_local_communication_lowering_matches_legacy_schedule_matrix() -> None:
    assert {
        name: _case_digest(case, rank_local_communication=True)
        for name, case in CASES.items()
    } == LEGACY_DIGESTS


if __name__ == "__main__":
    print(json.dumps({name: _case_digest(case) for name, case in CASES.items()}, indent=2, sort_keys=True))
