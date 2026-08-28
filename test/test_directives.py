"""Core placement, ordering, split-backward, and ZeRO directive tests."""

import pytest
import torch
import torch.fx as fx

from src.piper import (
    TrainingDAG,
    TrainingDAGEdge,
    TrainingDAGNode,
    apply_schedule_directives,
    _apply_order_directive,
    _apply_split_backward_stencil,
    _apply_split_directive,
    _bucket_matched_fwd_nodes,
    _parse_order_directive,
    resolve_total_order_per_stream,
    _serial_topological_order,
    _validate_schedule_tags_exist,
    _validate_split_backward_order_stencil,
)
import src.directives as directives
from src.dag import _remove_edge, _remove_nodes_and_incident_edges
from src.ordering import _dispatch_order, assign_dispatch_levels
from src.zero import (
    _ZeroLifetimeChains,
    _add_inter_chain_temporal_edges,
    _prune_zero_lifetime_metadata,
)


def _compute(uid: str, tag: dict) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag=tag,
        device=[0],
        stream="default_stream",
        node_meta={},
    )


def _bwd(uid: str, fwd_uid: str, tag: dict) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="BWD",
        tag=tag,
        device=[0],
        stream="default_stream",
        node_meta={"fwd_uid": fwd_uid, "bucket_key": fwd_uid},
    )


def _dummy_gm() -> fx.GraphModule:
    graph = fx.Graph()
    x = graph.placeholder("x")
    graph.output(x)
    return fx.GraphModule({}, graph)


def _routed_output_gm() -> fx.GraphModule:
    graph = fx.Graph()
    passthrough = graph.placeholder("passthrough")
    aux = graph.placeholder("aux")
    counts = graph.placeholder("routing_counts")
    payload = graph.placeholder("routed_payload")
    for node in (passthrough, aux, payload):
        node.meta["example_value"] = torch.empty(
            2, 4, device="meta", requires_grad=True
        )
    counts.meta["example_value"] = torch.empty(
        4, device="meta", dtype=torch.int64
    )
    graph.output((passthrough, aux, counts, payload))
    return fx.GraphModule({}, graph)




def _single_bucket_training_dag() -> TrainingDAG:
    dag = TrainingDAG()
    fwd = _compute("fwd", {"PP": 0, "PASS": "F"})
    fwd.node_meta["bucket_key"] = "bucket0"
    bwd = _bwd("bwd", "fwd", {"PP": 0, "PASS": "B"})
    bwd.node_meta["bucket_key"] = "bucket0"
    upd = TrainingDAGNode(
        uid="upd",
        node_kind="UPD",
        compute_subkind=None,
        tag={"PASS": None},
        device=None,
        stream="default_stream",
        node_meta={},
    )
    for node in (fwd, bwd, upd):
        dag.add_node(node)
    dag.add_edge(TrainingDAGEdge("fwd", "bwd", "data"))
    dag.add_edge(TrainingDAGEdge("bwd", "upd", "temporal"))
    return dag


def _three_microbatch_directives(**replicate_options) -> list[dict]:
    return [
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 2]},
        {
            "op": "replicate",
            "filter": {"PP": 0},
            "devices": [0, 2],
            "reduce_stream": "dp_stream",
            **replicate_options,
        },
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 3},
        {
            "op": "order",
            "filters": [
                [{"PP": 0, "MB": 0, "PASS": "F"}],
                [{"PP": 0, "MB": 1, "PASS": "F"}],
                [{"PP": 0, "MB": 2, "PASS": "F"}],
                [{"PP": 0, "MB": 0, "PASS": "B"}],
                [{"PP": 0, "MB": 1, "PASS": "B"}],
                [{"PP": 0, "MB": 2, "PASS": "B"}],
            ],
        },
    ]


def _weight_backward_uids(dag: TrainingDAG) -> set[str]:
    return {
        uid
        for uid, node in dag.nodes.items()
        if node.node_kind == "COMPUTE"
        and node.compute_subkind in ("BWD", "BWD_W")
    }




def test_place_rejects_conflicting_repeated_compute_match() -> None:
    dag = TrainingDAG()
    node = _compute("a", {"REGION": 0})
    node.device = None
    dag.add_node(node)

    with pytest.raises(ValueError, match="conflicting place directives"):
        apply_schedule_directives(
            dag,
            [
                {"op": "place", "filter": {"REGION": 0}, "devices": [0, 2]},
                {"op": "place", "filter": {"REGION": "*"}, "devices": [1, 3]},
            ],
        )




def test_place_rejects_unplaced_compute_node() -> None:
    dag = TrainingDAG()
    placed = _compute("placed", {"REGION": 0})
    unplaced = _compute("unplaced", {"REGION": 1})
    placed.device = None
    unplaced.device = None
    dag.add_node(placed)
    dag.add_node(unplaced)

    with pytest.raises(ValueError, match="assign every COMPUTE node"):
        apply_schedule_directives(
            dag,
            [{"op": "place", "filter": {"REGION": 0}, "devices": [0]}],
        )




def test_backward_p2p_receive_explicitly_depends_on_local_forward_shape() -> None:
    dag = TrainingDAG()
    local_forward = _compute("local_forward", {"PP": 0, "PASS": "F"})
    remote_backward = _compute("remote_backward", {"PP": 1, "PASS": "B"})
    local_backward = _compute("local_backward", {"PP": 0, "PASS": "B"})
    local_forward.compute_subkind = "FWD"
    remote_backward.compute_subkind = "BWD"
    local_backward.compute_subkind = "BWD"
    local_backward.node_meta.update(
        fwd_uid=local_forward.uid,
        bucket_key=local_forward.uid,
    )
    local_forward.device = [0, 2]
    local_backward.device = [0, 2]
    remote_backward.device = [1, 3]
    for node in (local_forward, remote_backward, local_backward):
        dag.add_node(node)
    dag.add_edge(
        TrainingDAGEdge(
            remote_backward.uid,
            local_backward.uid,
            "data",
            "activation_grad",
        )
    )

    directives._insert_send_recv_comm_nodes(dag)

    recv = next(
        node for node in dag.nodes.values() if node.node_kind == "RECV_COMM"
    )
    assert recv.node_meta["receive_kind"] == "backward"
    assert recv.node_meta["shape_reference_fwd_uid"] == local_forward.uid
    assert any(
        edge.src_uid == local_forward.uid
        and edge.dst_uid == recv.uid
        and edge.dep_kind == "temporal"
        for edge in dag.edges
    )






def test_unsharded_dp_accumulates_before_one_all_reduce_per_bucket() -> None:
    dag = _single_bucket_training_dag()

    apply_schedule_directives(dag, _three_microbatch_directives())

    reductions = [
        node for node in dag.nodes.values() if node.node_kind == "REDUCE_COMM"
    ]
    assert len(reductions) == 1
    reduction = reductions[0]
    backwards = _weight_backward_uids(dag)
    assert set(reduction.node_meta["bwd_uids"]) == backwards
    assert reduction.tag == {"PP": 0, "PASS": "B"}

    reduction_preds = {
        edge.src_uid
        for edge in dag.edges
        if edge.dst_uid == reduction.uid
        and edge.dep_kind in ("data", "temporal")
    }
    assert reduction_preds == backwards
    assert sum(
        edge.dst_uid == reduction.uid and edge.dep_kind == "data"
        for edge in dag.edges
    ) == 1
    assert any(
        edge.src_uid == reduction.uid
        and edge.dst_uid == "upd"
        and edge.dep_kind == "temporal"
        for edge in dag.edges
    )

    topo = _serial_topological_order(dag)
    assert all(topo.index(uid) < topo.index(reduction.uid) for uid in backwards)




@pytest.mark.parametrize(
    "shard_options",
    [
        {"shard_grads": True},
        {"shard_params": True},
        {"shard_grads": True, "shard_params": True},
    ],
)
def test_sharded_gradients_accumulate_before_one_reduce_scatter_per_bucket(
    shard_options: dict,
) -> None:
    dag = _single_bucket_training_dag()

    apply_schedule_directives(
        dag,
        _three_microbatch_directives(**shard_options),
    )
    backwards = _weight_backward_uids(dag)
    reductions = [
        node
        for node in dag.nodes.values()
        if node.node_kind == "REDUCE_SCATTER_COMM"
    ]
    assert len(reductions) == 1
    reduction = reductions[0]
    assert set(reduction.node_meta["bwd_uids"]) == backwards
    assert {
        edge.src_uid
        for edge in dag.edges
        if edge.dst_uid == reduction.uid
        and edge.dep_kind in ("data", "temporal")
    } == backwards
    assert sum(
        edge.dst_uid == reduction.uid and edge.dep_kind == "data"
        for edge in dag.edges
    ) == 1
    assert all(
        dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before") is True
        for uid in backwards
    )
    if shard_options.get("shard_params"):
        assert all(
            dag.nodes[uid].node_meta.get("zero_free_full_params_after") is True
            for uid in backwards
        )
        assert not reduction.node_meta.get("zero_free_full_params_after")

    zero_chains = _prune_zero_lifetime_metadata(dag)
    _add_inter_chain_temporal_edges(dag, zero_chains)
    resolve_total_order_per_stream(dag)

    assert len(
        [
            node
            for node in dag.nodes.values()
            if node.node_kind == "REDUCE_SCATTER_COMM"
        ]
    ) == 1
    assert {
        edge.src_uid
        for edge in dag.edges
        if edge.dst_uid == reduction.uid
        and edge.dep_kind in ("data", "temporal")
    } == backwards
    alloc_roots = {
        uid
        for uid in backwards
        if dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before")
    }
    assert len(alloc_roots) == 1
    reduction_data_preds = {
        edge.src_uid
        for edge in dag.edges
        if edge.dst_uid == reduction.uid and edge.dep_kind == "data"
    }
    assert len(reduction_data_preds) == 1
    topo = _serial_topological_order(dag)
    assert next(iter(reduction_data_preds)) == max(
        backwards,
        key=topo.index,
    )
    assert all(topo.index(uid) < topo.index(reduction.uid) for uid in backwards)
    _serial_topological_order(dag)




def _two_parameter_lifetime_chains(
    boundary_order: str,
) -> tuple[TrainingDAG, _ZeroLifetimeChains]:
    dag = TrainingDAG()
    first = _compute("first", {"PP": 0, "MB": 0, "PASS": "F"})
    second = _compute("second", {"PP": 0, "MB": 1, "PASS": "F"})
    for node in (first, second):
        node.node_meta["bucket_key"] = "bucket0"
        node.node_meta["zero_free_full_params_after"] = True

    def _gather(uid: str, compute_uid: str) -> TrainingDAGNode:
        return TrainingDAGNode(
            uid=uid,
            node_kind="ALL_GATHER_COMM",
            compute_subkind=None,
            tag={"PP": 0},
            device=[0],
            stream="dp_stream",
            node_meta={"bucket_key": "bucket0", "compute_uid": compute_uid},
        )

    def _dummy(uid: str) -> TrainingDAGNode:
        return TrainingDAGNode(
            uid=uid,
            node_kind="ORDER_DUMMY",
            compute_subkind=None,
            tag={"PP": 0},
            device=[0],
            stream="delay_stream",
            node_meta={},
        )

    first_gather = _gather("all_gather.0", first.uid)
    second_gather = _gather("all_gather.1", second.uid)
    delay0 = _dummy("delay.0")
    delay1 = _dummy("delay.1")
    nodes = [first, second, first_gather, second_gather, delay0, delay1]
    if boundary_order == "reverse":
        nodes.append(_dummy("delay.2"))
    for node in nodes:
        dag.add_node(node)

    dag.add_edge(TrainingDAGEdge(first_gather.uid, first.uid, "data"))
    dag.add_edge(TrainingDAGEdge(delay0.uid, delay1.uid, "temporal"))
    dag.add_edge(TrainingDAGEdge(delay1.uid, second_gather.uid, "temporal"))
    dag.add_edge(TrainingDAGEdge(second_gather.uid, second.uid, "data"))
    if boundary_order == "forward":
        dag.add_edge(TrainingDAGEdge(first.uid, second_gather.uid, "temporal"))
    elif boundary_order == "reverse":
        dag.add_edge(TrainingDAGEdge(second_gather.uid, first.uid, "temporal"))
        dag.add_edge(TrainingDAGEdge(second_gather.uid, "delay.2", "temporal"))
        dag.add_edge(TrainingDAGEdge("delay.2", second.uid, "temporal"))
    elif boundary_order != "unordered":
        raise ValueError(boundary_order)
    return dag, _ZeroLifetimeChains(
        gradient=[],
        parameter=[[first.uid], [second.uid]],
    )






def test_zero_parameter_lifetimes_select_feasible_reverse_order() -> None:
    dag, chains = _two_parameter_lifetime_chains("reverse")

    _add_inter_chain_temporal_edges(dag, chains)

    assert any(
        edge.src_uid == "second"
        and edge.dst_uid == "all_gather.0"
        and edge.dep_kind == "temporal"
        for edge in dag.edges
    )


def test_zero_parameter_lifetimes_reject_cyclic_interval_order() -> None:
    dag, chains = _two_parameter_lifetime_chains("unordered")
    dag.add_edge(TrainingDAGEdge("all_gather.1", "first", "temporal"))
    dag.add_edge(TrainingDAGEdge("all_gather.0", "second", "temporal"))

    with pytest.raises(
        ValueError,
        match="cyclic parameter lifetime order.*bucket='bucket0'",
    ):
        _add_inter_chain_temporal_edges(dag, chains)










def test_expert_shard_retains_accumulated_grad_sync_for_replicated_weights() -> None:
    dag = TrainingDAG()
    backwards = []
    for mb in range(3):
        node = _bwd(
            f"bwd.{mb}",
            "fwd",
            {"PP": 0, "EP": 0, "MB": mb, "PASS": "B"},
        )
        node.compute_subkind = "BWD_W"
        node.device = [0, 2]
        node.node_meta["bucket_key"] = "bucket0"
        dag.add_node(node)
        backwards.append(node)

    directives._insert_reduce_scatter_comm_nodes(
        dag,
        [{"PP": 0}],
        [0, 2],
    )
    assert all(
        node.node_meta.get("zero_alloc_full_grads_before") is True
        for node in backwards
    )

    directives._insert_shard_a2a_comm_nodes(
        dag,
        [{"PP": 0, "EP": 0}],
        [0, 2],
    )

    assert any(
        node.node_kind == "REDUCE_SCATTER_COMM"
        for node in dag.nodes.values()
    )
    assert all(
        node.node_meta.get("zero_alloc_full_grads_before") is True
        for node in backwards
    )




def test_parse_order_directive_accepts_nested_filter_groups() -> None:
    directive = {
        "op": "order",
        "filters": [
            [
                {"PP": 0, "MB": 0, "PASS": "F"},
                {"PP": 0, "MB": 1, "PASS": "F"},
            ],
            [
                {"PP": 0, "MB": 0, "PASS": "B"},
            ],
        ],
    }

    groups = _parse_order_directive(directive)

    assert groups == [
        [
            {"PP": 0, "MB": 0, "PASS": "F"},
            {"PP": 0, "MB": 1, "PASS": "F"},
        ],
        [
            {"PP": 0, "MB": 0, "PASS": "B"},
        ],
    ]






def test_split_backward_stencil_allows_mixed_fused_and_split_microbatches() -> None:
    directive = {
        "op": "order",
        "filters": [
            [{"PP": 0, "MB": 0, "PASS": "B"}],
            [{"PP": 0, "MB": 1, "PASS": "BI"}],
            [{"PP": 0, "MB": 1, "PASS": "BW"}],
        ],
    }
    dag = TrainingDAG()
    fwd = _compute("fwd", {"PP": 0, "PASS": "F"})
    fwd.node_meta.update(
        {"gm": _dummy_gm(), "graphargs": [], "input_idxs": [], "param_idxs": []}
    )
    dag.add_node(fwd)
    dag.add_node(_bwd("bwd", "fwd", {"PP": 0, "PASS": "B"}))
    dag.add_edge(TrainingDAGEdge("fwd", "bwd", "data"))

    split_keys = _validate_split_backward_order_stencil([directive])
    _apply_split_directive(dag, {}, "MB", 2)
    _apply_split_backward_stencil(dag, split_keys)

    assert dag.nodes["bwd"].compute_subkind == "BWD"
    assert dag.nodes["bwd"].tag["PASS"] == "B"
    split_bwd_uid = "bwd.splitMB1"
    assert dag.nodes[split_bwd_uid].compute_subkind == "BWD_I"
    assert dag.nodes[split_bwd_uid].tag["PASS"] == "BI"
    assert dag.nodes[f"{split_bwd_uid}.bw"].compute_subkind == "BWD_W"




def test_bucket_rewrite_preserves_final_output_schema_and_routed_boundary(
    monkeypatch,
) -> None:
    dag = TrainingDAG()
    fwd = _compute("fwd", {"PP": 0, "PASS": "F"})
    routed_gm = _routed_output_gm()
    fwd.node_meta.update({
        "stage_id": 0,
        "segment_id": 0,
        "gm": routed_gm,
        "graphargs": [],
        "input_idxs": [],
        "param_idxs": [],
        "a2a_boundary_after": {
            "tensor_idx": 3,
            "routing_counts_idx": 2,
        },
    })
    dag.add_node(fwd)
    dag.add_node(_bwd("bwd", "fwd", {"PP": 0, "PASS": "B"}))
    dag.add_edge(TrainingDAGEdge("fwd", "bwd", "data"))

    monkeypatch.setattr(
        directives,
        "bucket_stage",
        lambda *_args, **_kwargs: [
            (_dummy_gm(), [], [], []),
            (routed_gm, [], [], []),
        ],
    )

    _bucket_matched_fwd_nodes(dag, [{"PP": 0}], 25)

    final_meta = dag.nodes["fwd.bucket1"].node_meta
    assert final_meta["output_names"] == [
        "passthrough",
        "aux",
        "routing_counts",
        "routed_payload",
    ]
    assert final_meta["tensor_output_indices"] == [0, 1, 2, 3]
    assert final_meta["differentiable_output_indices"] == [0, 1, 3]
    assert final_meta["a2a_boundary_after"] == {
        "tensor_idx": 3,
        "routing_counts_idx": 2,
    }










def test_apply_order_directive_groups_nested_subdags_with_dummy_source_sink() -> None:
    dag = TrainingDAG()
    for node in [
        _compute("prev", {"slot": 0}),
        _compute("lane0.first", {"slot": 1, "lane": 0}),
        _compute("lane0.last", {"slot": 1, "lane": 0}),
        _compute("lane1", {"slot": 1, "lane": 1}),
        _compute("next", {"slot": 2}),
    ]:
        dag.add_node(node)
    dag.add_edge(TrainingDAGEdge("lane0.first", "lane0.last", "data"))

    _apply_order_directive(
        dag,
        [
            [{"slot": 0}],
            [{"slot": 1, "lane": 0}, {"slot": 1, "lane": 1}],
            [{"slot": 2}],
        ],
        directive_idx=7,
    )

    source_uid = "order.7.group1.source"
    sink_uid = "order.7.group1.sink"
    assert dag.nodes[source_uid].node_kind == "ORDER_DUMMY"
    assert dag.nodes[sink_uid].node_kind == "ORDER_DUMMY"

    temporal_edges = {
        (edge.src_uid, edge.dst_uid)
        for edge in dag.edges
        if edge.dep_kind == "temporal"
    }
    assert temporal_edges == {
        ("prev", source_uid),
        (source_uid, "lane0.first"),
        (source_uid, "lane1"),
        ("lane0.last", sink_uid),
        ("lane1", sink_uid),
        (sink_uid, "next"),
    }

    topo = _serial_topological_order(dag)
    assert topo.index("prev") < topo.index(source_uid)
    assert topo.index(source_uid) < topo.index("lane0.first")
    assert topo.index(source_uid) < topo.index("lane1")
    assert topo.index("lane0.last") < topo.index(sink_uid)
    assert topo.index("lane1") < topo.index(sink_uid)
    assert topo.index(sink_uid) < topo.index("next")


def test_apply_order_directive_rejects_reverse_model_dataflow() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("producer", {"slot": 0}))
    dag.add_node(_compute("consumer", {"slot": 1}))
    dag.add_edge(TrainingDAGEdge("producer", "consumer", "data"))

    with pytest.raises(ValueError, match="violates model dataflow"):
        _apply_order_directive(
            dag,
            [
                [{"slot": 1}],
                [{"slot": 0}],
            ],
            directive_idx=3,
        )








def test_shard_uses_routed_contract_for_all_ep_collectives() -> None:
    dag = TrainingDAG()
    pre = _compute("pre.fwd", {"PP": 0, "PASS": "F"})
    expert = _compute(
        "expert.fwd",
        {"PP": 0, "EP": 0, "PASS": "F"},
    )
    post = _compute("post.fwd", {"PP": 0, "PASS": "F"})
    pre.device = expert.device = post.device = [0, 2]
    pre.node_meta["a2a_boundary_after"] = {
        "tensor_idx": 0,
        "routing_counts_idx": 1,
    }
    pre.node_meta["output_names"] = ["payload", "counts"]
    pre.node_meta["runtime_outputs_by_destination"] = {
        "expert.fwd": [
            {
                "value_id": "payload",
                "source_output_idx": 0,
                "destination_input_idx": 0,
            },
            {
                "value_id": "counts",
                "source_output_idx": 1,
                "destination_input_idx": 1,
            },
        ]
    }
    expert.node_meta["a2a_boundary_after"] = {"tensor_idx": 0}
    expert.node_meta["output_names"] = ["combined"]
    expert.node_meta["runtime_outputs_by_destination"] = {
        "post.fwd": [
            {
                "value_id": "combined",
                "source_output_idx": 0,
                "destination_input_idx": 0,
            }
        ]
    }

    post_bwd = _bwd(
        "post.bwd",
        "post.fwd",
        {"PP": 0, "PASS": "B"},
    )
    expert_bwd = _bwd(
        "expert.bwd",
        "expert.fwd",
        {"PP": 0, "EP": 0, "PASS": "B"},
    )
    pre_bwd = _bwd(
        "pre.bwd",
        "pre.fwd",
        {"PP": 0, "PASS": "B"},
    )
    post_bwd.device = expert_bwd.device = pre_bwd.device = [0, 2]

    for node in (pre, expert, post, post_bwd, expert_bwd, pre_bwd):
        dag.add_node(node)
    dag.add_edge(TrainingDAGEdge("pre.fwd", "expert.fwd", "data"))
    dag.add_edge(TrainingDAGEdge("expert.fwd", "post.fwd", "data"))
    dag.add_edge(TrainingDAGEdge("post.bwd", "expert.bwd", "data"))
    dag.add_edge(TrainingDAGEdge("expert.bwd", "pre.bwd", "data"))

    directives._insert_shard_a2a_comm_nodes(
        dag,
        [{"PP": 0, "EP": 0}],
        [0, 2],
    )

    collectives = [
        node
        for node in dag.nodes.values()
        if node.node_kind == "A2A_COMM"
    ]
    assert len(collectives) == 4
    assert all("ragged" not in node.node_meta for node in collectives)
    assert {
        node.node_meta["expert_fwd_uid"] for node in collectives
    } == {"expert.fwd"}

    count_collectives = [
        node
        for node in collectives
        if node.node_meta.get("routing_counts_idx") is not None
    ]
    assert len(count_collectives) == 1
    assert count_collectives[0].tag["PASS"] == "F"
    assert count_collectives[0].node_meta["direction"] == "incoming"
    assert count_collectives[0].node_meta["routing_counts_idx"] == 1










def test_apply_schedule_directives_rejects_unmatched_place_directive() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))

    with pytest.raises(ValueError, match="matched zero nodes"):
        apply_schedule_directives(
            dag,
            [{"op": "place", "filter": {"PP": 1}, "devices": [0]}],
        )




