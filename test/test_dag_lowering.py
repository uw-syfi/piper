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

"""Representative schedule and actor-DAG lowering tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from ray import cloudpickle
import torch
import torch.fx as fx
import torch.nn as nn

from examples.build_schedule import (
    build_1f1b_schedule,
    build_dualpipev_schedule,
    build_interleaved_1f1b_schedule,
    build_zerobubble_schedule,
    build_afpipe_baseline_schedule,
    infer_afpipe_num_layers,
)
from examples.models.qwen3 import AFDQwen3Model, create_qwen3_config
from src.compile import _prepare_training_dags_for_rpc
from src.dag import (
    TrainingDAG,
    TrainingDAGEdge,
    TrainingDAGNode,
    _ReachabilityIndex,
    build_training_dag,
)
import src.directives as directives
from src.directives import _validate_schedule_tags_exist, apply_schedule_directives
from src.fx import split_gm_by_annotations
from src.ordering import (
    _resolve_default_stream_order,
    assign_dispatch_levels,
    execution_stream_domain,
    resolve_total_order_per_stream,
)
from src.piper import (
    _materialize_global_communication_issue_edges,
    _order_communication_transactions,
    _reset_annotation_state,
    _split_global_training_dag_by_pp_rank,
    _validate_dense_state_sequences,
    _validate_global_communication_order,
    _validate_p2p_sequences,
    _validate_routed_collective_sequences,
)
from src.schedule import load_schedule_directives
from src.zero import _add_inter_chain_temporal_edges, _prune_zero_lifetime_metadata


_LIVE_META_KEYS = {
    "gm",
    "graphargs",
    "input_idxs",
    "param_idxs",
    "triton_constant_args",
    "triton_kernels",
}

_ROUTED_COUNTS_SCHEMA = {
    "shape": (1, 8),
    "dtype": "torch.int64",
    "requires_grad": False,
}
_ROUTED_PAYLOAD_SCHEMA = {
    "shape": (8, 4),
    "dtype": "torch.float32",
    "requires_grad": True,
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
                boundary_after.update(
                    routing_counts_idx=1,
                    payload_schema=dict(_ROUTED_PAYLOAD_SCHEMA),
                    routing_counts_schema=dict(_ROUTED_COUNTS_SCHEMA),
                )
            elif routed_a2a and segment == 1:
                boundary_after["payload_schema"] = dict(
                    _ROUTED_PAYLOAD_SCHEMA
                )
            dag.add_node(
                TrainingDAGNode(
                    uid=uid,
                    node_kind="COMPUTE",
                    compute_subkind="FWD",
                    tag={
                        "PP": pp,
                        **(
                            {"EP": ep}
                            if ep is not None or not routed_a2a
                            else {}
                        ),
                        "PASS": "F",
                    },
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
                        **(
                            {
                                "output_names": (
                                    ["routed_payload", "routing_counts"]
                                    if segment == 0
                                    else ["activation"]
                                ),
                                "runtime_outputs_by_destination": {},
                            }
                            if routed_a2a
                            else {}
                        ),
                        "a2a_boundary_after": boundary_after,
                    },
                )
            )
            if previous_uid is not None:
                if routed_a2a:
                    producer = dag.nodes[previous_uid]
                    mapping = [
                        {
                            "value_id": "activation",
                            "source_output_idx": 0,
                            "destination_input_idx": 0,
                            "is_tensor": True,
                        }
                    ]
                    boundary = producer.node_meta["a2a_boundary_after"]
                    if "routing_counts_idx" in boundary:
                        mapping.append(
                            {
                                "value_id": "routing_counts",
                                "source_output_idx": int(
                                    boundary["routing_counts_idx"]
                                ),
                                "destination_input_idx": 1,
                                "is_tensor": True,
                            }
                        )
                    producer.node_meta["runtime_outputs_by_destination"][uid] = mapping
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
                "filter": {
                    "PP": stage,
                    **({"EP": "-"} if case["ep"] else {}),
                },
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


def test_dualpipev_never_splits_virtual_stage_zero_backward() -> None:
    for physical_pp in (1, 2, 4):
        microbatches = 2 * physical_pp
        directives = build_dualpipev_schedule(physical_pp, microbatches)
        stage_zero_passes = [
            tag_filter["PASS"]
            for directive in directives
            for slot in directive["filters"]
            for tag_filter in slot
            if tag_filter["PP"] == 0 and tag_filter["PASS"] != "F"
        ]

        assert stage_zero_passes == ["B"] * microbatches


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
    chains_by_rank = []
    for rank_dag in per_rank:
        if rank_local_order or rank_local_communication:
            directives.apply_order_directives(
                rank_dag,
                schedule,
                skip_unmatched_directives=True,
            )
        _resolve_default_stream_order(rank_dag)
        chains_by_rank.append(_prune_zero_lifetime_metadata(rank_dag))
    for rank_dag, chains in zip(per_rank, chains_by_rank):
        _add_inter_chain_temporal_edges(rank_dag, chains)
    communication_order = _order_communication_transactions(per_rank)
    for rank_dag in per_rank:
        resolve_total_order_per_stream(rank_dag)
    _materialize_global_communication_issue_edges(
        per_rank, communication_order
    )
    for rank_dag in per_rank:
        assign_dispatch_levels(rank_dag)
    payload = [_canonical_dag(rank_dag) for rank_dag in per_rank]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_global_and_rank_local_lowering_paths_match() -> None:
    for name, case in CASES.items():
        global_digest = _case_digest(case)
        assert _case_digest(case, rank_local_order=True) == global_digest, name
        assert (
            _case_digest(case, rank_local_communication=True) == global_digest
        ), name


_BASE_SCHEDULES = Path(__file__).parents[1] / "examples" / "base-schedules"
_ROUTED_KINDS = {"M_TO_N_COMM", "N_TO_M_COMM"}


class _ToyExperts(nn.Module):
    def forward(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        scale = num_tokens_per_expert.sum().to(routed_input.dtype) * 0 + 1
        return routed_input * scale


def _trace_afd_segments(num_stages: int, expert_shard_degree: int):
    captured = {}

    def backend(gm, example_inputs):
        del example_inputs
        captured["gm"] = gm
        return gm.forward

    torch._dynamo.reset()
    _reset_annotation_state()
    with torch.device("meta"):
        config = create_qwen3_config("9M")
        model = AFDQwen3Model(
            config,
            num_stages=num_stages,
            expert_shard_degree=expert_shard_degree,
        ).to(torch.bfloat16)
        for layer in model.layers.values():
            layer.moe.experts = _ToyExperts()
        tokens = torch.zeros((1, 8), dtype=torch.int64, device="meta")
        output = torch.compile(model, backend=backend, fullgraph=True)(tokens)

    assert tuple(output.shape) == (1, 8, config.vocab_size)
    _top_level, segments = split_gm_by_annotations(captured["gm"])
    return segments


def _full_afd_schedule(base_name: str, *, actor_groups: int) -> list[dict]:
    base = load_schedule_directives(str(_BASE_SCHEDULES / base_name))
    n_layers = infer_afpipe_num_layers(base, fallback=4)
    return [
        *base,
        {
            "op": "split",
            "filter": {},
            "dim_name": "MB",
            "num_microbatches": 2,
        },
        *build_afpipe_baseline_schedule(
            base,
            n_mbs=2,
            n_layers=n_layers,
            expected_actor_groups=actor_groups,
        ),
    ]


def _lower_afd_schedule(base_name: str, *, actor_groups: int) -> tuple[TrainingDAG, list[TrainingDAG]]:
    directives = _full_afd_schedule(base_name, actor_groups=actor_groups)
    shard_degrees = {
        len(directive["devices"])
        for directive in directives
        if directive.get("op") == "shard"
    }
    assert len(shard_degrees) == 1
    segments = _trace_afd_segments(
        num_stages=actor_groups,
        expert_shard_degree=next(iter(shard_degrees)),
    )
    global_dag = build_training_dag(segments)
    _validate_schedule_tags_exist(global_dag, directives)
    apply_schedule_directives(global_dag, directives)

    # This intentionally exercises the default path: order edges must connect
    # repeated local A/F chunks, without placement-based component merging.
    local_dags = _split_global_training_dag_by_pp_rank(global_dag)
    for local_dag in local_dags:
        _resolve_default_stream_order(local_dag)
    zero_chains_by_dag = [
        _prune_zero_lifetime_metadata(local_dag) for local_dag in local_dags
    ]
    for local_dag, zero_chains in zip(local_dags, zero_chains_by_dag):
        _add_inter_chain_temporal_edges(local_dag, zero_chains)
    communication_order = _order_communication_transactions(local_dags)
    for local_dag in local_dags:
        resolve_total_order_per_stream(local_dag)
    _materialize_global_communication_issue_edges(
        local_dags, communication_order
    )
    for local_dag in local_dags:
        assign_dispatch_levels(local_dag)
    _validate_global_communication_order(local_dags, communication_order)
    _validate_routed_collective_sequences(local_dags)
    _validate_dense_state_sequences(local_dags)
    _validate_p2p_sequences(local_dags)
    return global_dag, local_dags


def _device_key(dag: TrainingDAG) -> tuple[int, ...]:
    keys = {
        tuple(sorted(int(device) for device in node.device))
        for node in dag.nodes.values()
        if node.device is not None
    }
    assert len(keys) == 1
    return next(iter(keys))


def _assert_connected(dag: TrainingDAG) -> None:
    assert dag.nodes
    first_uid = next(iter(dag.nodes))
    visited = {first_uid}
    pending = [first_uid]
    while pending:
        uid = pending.pop()
        neighbors = dag.preds.get(uid, set()) | dag.succs.get(uid, set())
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                pending.append(neighbor)
    assert visited == set(dag.nodes)


@pytest.mark.parametrize(
    (
        "base_name",
        "actor_groups",
        "expected_devices",
        "dense_degree",
        "routed_degree",
    ),
    [
        (
            "afd_placement_pp2.json",
            2,
            [(0, 2, 4, 5), (1, 3)],
            4,
            2,
        ),
        (
            "afd_placement_pp4.json",
            4,
            [(0, 4, 8, 10), (1, 5), (2, 6, 9, 11), (3, 7)],
            4,
            2,
        ),
    ],
)
def test_traced_afd_qwen_full_schedule_lowers_to_connected_actor_dags(
    base_name: str,
    actor_groups: int,
    expected_devices: list[tuple[int, ...]],
    dense_degree: int,
    routed_degree: int,
) -> None:
    global_dag, local_dags = _lower_afd_schedule(
        base_name,
        actor_groups=actor_groups,
    )

    assert len(local_dags) == actor_groups
    assert [_device_key(dag) for dag in local_dags] == expected_devices
    for dag in local_dags:
        _assert_connected(dag)
        for edge in dag.edges:
            if edge.dep_kind != "issue":
                continue
            assert execution_stream_domain(
                dag.nodes[edge.src_uid]
            ) != execution_stream_domain(dag.nodes[edge.dst_uid])

    rpc_dags = _prepare_training_dags_for_rpc(local_dags)
    cloudpickle.dumps(rpc_dags)
    assert all(
        all(
            isinstance(dim, int)
            for arg in node.node_meta.get("graphargs", ())
            if isinstance(arg, torch.Tensor)
            for dim in arg.shape
        )
        for dag in rpc_dags
        for node in dag.nodes.values()
        if node.node_kind == "COMPUTE" and node.compute_subkind == "FWD"
    )

    routed = [
        node for node in global_dag.nodes.values() if node.node_kind in _ROUTED_KINDS
    ]
    # Four layers x two routed boundaries x two passes x two microbatches,
    # represented by one local node on each side of every collective.
    assert len(routed) == 64
    assert sum(node.node_kind == "M_TO_N_COMM" for node in routed) == 32
    assert sum(node.node_kind == "N_TO_M_COMM" for node in routed) == 32
    assert not any(
        node.node_kind in {"A2A_COMM", "SCALE_GRADS"}
        for node in global_dag.nodes.values()
    )
    dense_reductions = [
        node
        for node in global_dag.nodes.values()
        if node.node_kind == "REDUCE_COMM"
    ]
    assert dense_reductions
    assert all(len(node.device or ()) == dense_degree for node in dense_reductions)
    assert not any(
        len(node.device or ()) == routed_degree for node in dense_reductions
    )
    p2p_nodes = [
        node
        for node in global_dag.nodes.values()
        if node.node_kind in {"SEND_COMM", "RECV_COMM"}
    ]
    assert all(node.node_meta.get("dense_state_transfer") for node in p2p_nodes)
    assert len(p2p_nodes) == (0 if actor_groups == 2 else 32)
    routed_fwd_nodes = [
        node
        for node in global_dag.nodes.values()
        if node.node_kind == "COMPUTE"
        and node.compute_subkind == "FWD"
        and "FFN" in node.tag
    ]
    assert all(
        node.node_meta.get("dense_state_spec", {}).get("arity", 0) > 0
        for node in routed_fwd_nodes
    )
    if actor_groups == 2:
        assert all(
            node.node_meta.get("dense_state_mode") == "local"
            for node in routed_fwd_nodes
        )
    else:
        assert all(
            node.node_meta.get("dense_state_mode") == "direct"
            for node in routed_fwd_nodes
        )
        for expert_fwd_uid in {
            node.node_meta["expert_fwd_uid"]
            for node in routed
            if node.node_kind == "M_TO_N_COMM"
            and node.node_meta["role"] == "source"
            and node.tag["PASS"] == "F"
        }:
            dispatch_source = next(
                node
                for node in routed
                if node.node_kind == "M_TO_N_COMM"
                and node.node_meta["role"] == "source"
                and node.tag["PASS"] == "F"
                and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
            )
            combine_destination = next(
                node
                for node in routed
                if node.node_kind == "N_TO_M_COMM"
                and node.node_meta["role"] == "destination"
                and node.tag["PASS"] == "F"
                and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
            )
            state_source = next(
                node
                for node in p2p_nodes
                if node.node_kind == "SEND_COMM"
                and node.tag["PASS"] == "F"
                and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
            )
            state_destination = next(
                node
                for node in p2p_nodes
                if node.node_kind == "RECV_COMM"
                and node.tag["PASS"] == "F"
                and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
            )
            assert dispatch_source.device == state_source.device
            assert combine_destination.device == state_destination.device
            assert combine_destination.device != dispatch_source.device

    by_comm_id: dict[str, list] = {}
    for node in routed:
        by_comm_id.setdefault(node.node_meta["comm_id"], []).append(node)
        source = node.node_meta["source_devices"]
        destination = node.node_meta["destination_devices"]
        if node.node_kind == "M_TO_N_COMM":
            assert len(source) == dense_degree
            assert len(destination) == routed_degree
        else:
            assert len(source) == routed_degree
            assert len(destination) == dense_degree

    assert len(by_comm_id) == 32
    assert all(
        {node.node_meta["role"] for node in pair} == {"source", "destination"}
        for pair in by_comm_id.values()
    )

    expected_local_routed = 32 if actor_groups == 2 else 16
    for dag in local_dags:
        local_routed = [
            node for node in dag.nodes.values() if node.node_kind in _ROUTED_KINDS
        ]
        assert len(local_routed) == expected_local_routed
        assert sum(node.node_kind == "M_TO_N_COMM" for node in local_routed) == (
            expected_local_routed // 2
        )
        assert sum(node.node_kind == "N_TO_M_COMM" for node in local_routed) == (
            expected_local_routed // 2
        )

    if actor_groups == 2:
        dense_dag = next(dag for dag in local_dags if _device_key(dag) == expected_devices[0])
        first_dispatch = next(
            node
            for node in dense_dag.nodes.values()
            if node.node_kind == "M_TO_N_COMM"
            and node.node_meta["role"] == "source"
            and node.tag.get("PASS") == "F"
            and node.tag.get("ATTN") == 0
            and node.tag.get("MB") == 0
        )
        next_attention = next(
            node
            for node in dense_dag.nodes.values()
            if node.node_kind == "COMPUTE"
            and node.compute_subkind == "FWD"
            and node.tag.get("ATTN") == 0
            and node.tag.get("MB") == 1
        )
        reachability = _ReachabilityIndex(dense_dag)
        assert not reachability.has_path(first_dispatch.uid, next_attention.uid)
