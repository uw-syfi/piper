from pathlib import Path
from types import SimpleNamespace

import pytest
from ray import cloudpickle
import torch
import torch.nn as nn

from examples.build_schedule import (
    build_afpipe_baseline_schedule,
    infer_afpipe_num_layers,
)
from examples.models.qwen3 import AFDQwen3Model, create_qwen3_config
from src.dag import TrainingDAG, _ReachabilityIndex, build_training_dag
from src.compile import _prepare_training_dags_for_rpc
from src.directives import _validate_schedule_tags_exist, apply_schedule_directives
from src.fx import split_gm_by_annotations
from src.ordering import (
    _resolve_default_stream_order,
    assign_dispatch_levels,
    execution_stream_domain,
    resolve_total_order_per_stream,
)
from src.profiling import NodeProfile
from src.replay import replay_training_step
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


def _assert_cpu_replay_completes(dags: list[TrainingDAG]) -> None:
    mode = "isolated"
    for dag in dags:
        for node in dag.nodes.values():
            if node.node_kind == "ORDER_DUMMY":
                continue
            assert node.device is not None
            node.profile_times_by_mode[mode] = {
                int(rank): NodeProfile.from_samples(
                    (1.0,),
                    start_offset_samples_us=(0.0,),
                )
                for rank in node.device
            }
    plan = SimpleNamespace(
        per_pp_dags=dags,
        topology={},
        profile_metadata={
            "timeline_alignment": {
                "modes": {
                    mode: {
                        "samples": 1,
                        "max_pairwise_uncertainty_us": 0.0,
                    }
                }
            }
        },
    )
    assert replay_training_step(plan, profile_mode=mode).step_time_us > 0


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
        (
            "afd_e2e_pp2.json",
            2,
            [(0, 1), (2,)],
            2,
            1,
        ),
        (
            "afd_e2e_pp4.json",
            4,
            [(0, 1), (2,), (3, 4), (5,)],
            2,
            1,
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

    if base_name == "afd_placement_pp4.json":
        _assert_cpu_replay_completes(local_dags)

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
