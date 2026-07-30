import pytest
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
from src.zero import _add_inter_chain_temporal_edges, _prune_zero_lifetime_metadata


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


def test_singleton_replica_group_accumulates_locally_without_collective() -> None:
    dag = _single_bucket_training_dag()
    directives_without_order = _three_microbatch_directives()
    directives_without_order[0]["devices"] = [0]
    directives_without_order[1]["devices"] = [0]

    apply_schedule_directives(dag, directives_without_order)

    assert not any(
        node.node_kind in ("REDUCE_COMM", "REDUCE_SCATTER_COMM")
        for node in dag.nodes.values()
    )
    assert {
        node.node_meta.get("bucket_key")
        for node in dag.nodes.values()
        if node.node_kind == "COMPUTE"
    } == {"bucket0"}
    assert {
        edge.src_uid
        for edge in dag.edges
        if edge.dst_uid == "upd" and edge.dep_kind == "temporal"
    } == _weight_backward_uids(dag)


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
    resolve_total_order_per_stream(dag)
    _add_inter_chain_temporal_edges(dag, zero_chains)

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


def test_sharded_gradient_lifetime_pruning_still_coalesces_within_microbatch() -> None:
    dag = TrainingDAG()
    first = _bwd("bwd.0", "fwd", {"PP": 0, "MB": 0, "PASS": "B"})
    second = _bwd("bwd.1", "fwd", {"PP": 0, "MB": 0, "PASS": "B"})
    for node in (first, second):
        node.device = [0, 2]
        node.node_meta["bucket_key"] = "bucket0"
        dag.add_node(node)
    dag.add_edge(TrainingDAGEdge(first.uid, second.uid, "temporal"))
    directives._insert_reduce_scatter_comm_nodes(dag, [{"PP": 0}], [0, 2])

    _prune_zero_lifetime_metadata(dag)

    reductions = [
        node
        for node in dag.nodes.values()
        if node.node_kind == "REDUCE_SCATTER_COMM"
    ]
    assert len(reductions) == 1
    assert any(
        edge.src_uid == second.uid and edge.dst_uid == reductions[0].uid
        for edge in dag.edges
    )
    assert first.node_meta["zero_alloc_full_grads_before"] is True
    assert "zero_alloc_full_grads_before" not in second.node_meta


def test_expert_shard_clears_accumulated_grad_markers_for_all_microbatches() -> None:
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

    assert not any(
        node.node_kind == "REDUCE_SCATTER_COMM"
        for node in dag.nodes.values()
    )
    assert all(
        "zero_alloc_full_grads_before" not in node.node_meta
        for node in backwards
    )


def test_stream_order_anchors_accumulated_reduction_to_latest_contributor() -> None:
    dag = TrainingDAG()
    first = _compute("first", {"PP": 0})
    middle = _compute("middle", {"PP": 0})
    last = _compute("last", {"PP": 0})
    gather = TrainingDAGNode(
        uid="gather",
        node_kind="ALL_GATHER_COMM",
        compute_subkind=None,
        tag={"PP": 0},
        device=[0, 2],
        stream="dp_stream",
        node_meta={},
    )
    reduction = TrainingDAGNode(
        uid="reduction",
        node_kind="REDUCE_SCATTER_COMM",
        compute_subkind=None,
        tag={"PP": 0},
        device=[0, 2],
        stream="dp_stream",
        node_meta={},
    )
    for node in (first, middle, last, gather, reduction):
        dag.add_node(node)
    dag.add_edge(TrainingDAGEdge(first.uid, middle.uid, "temporal"))
    dag.add_edge(TrainingDAGEdge(middle.uid, last.uid, "temporal"))
    dag.add_edge(TrainingDAGEdge(gather.uid, middle.uid, "data"))
    dag.add_edge(TrainingDAGEdge(first.uid, reduction.uid, "temporal"))
    dag.add_edge(TrainingDAGEdge(last.uid, reduction.uid, "data"))

    resolve_total_order_per_stream(dag)

    topo = _serial_topological_order(dag)
    assert topo.index(gather.uid) < topo.index(reduction.uid)


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


def test_parse_order_directive_rejects_flat_filter_groups() -> None:
    directive = {
        "op": "order",
        "filters": [
            {"PP": 0, "MB": 0, "PASS": "F"},
            {"PP": 0, "MB": 0, "PASS": "B"},
        ],
    }

    with pytest.raises(ValueError, match="must be a non-empty list"):
        _parse_order_directive(directive)


def test_split_backward_stencil_handles_nested_filter_groups() -> None:
    directive = {
        "op": "order",
        "filters": [
            [
                {"PP": 0, "MB": 0, "PASS": "BI"},
                {"PP": 0, "MB": 0, "PASS": "BW"},
            ],
            [
                {"PP": 0, "MB": 1, "PASS": "F"},
            ],
        ],
    }

    split_keys = _validate_split_backward_order_stencil([directive])

    assert split_keys == {(("MB", 0), ("PP", 0))}


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
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))
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


def test_bucket_rewrite_resolves_split_microbatch_bwd_by_metadata(monkeypatch) -> None:
    dag = TrainingDAG()
    fwd = _compute("fwd", {"PP": 0, "PASS": "F"})
    fwd.node_meta.update({
        "stage_id": 0,
        "segment_id": 0,
        "gm": _dummy_gm(),
        "graphargs": [],
        "input_idxs": [],
        "param_idxs": [],
    })
    dag.add_node(fwd)
    dag.add_node(_bwd("bwd", "fwd", {"PP": 0, "PASS": "B"}))
    dag.add_edge(TrainingDAGEdge("fwd", "bwd", "data"))

    _apply_split_directive(dag, {}, "MB", 2)

    def fake_bucket_stage(*_args, **_kwargs):
        return [
            (_dummy_gm(), [], [], []),
            (_dummy_gm(), [], [], []),
        ]

    monkeypatch.setattr(directives, "bucket_stage", fake_bucket_stage)

    _bucket_matched_fwd_nodes(dag, [{"PP": 0}], 25)

    assert "fwd.bucket0" in dag.nodes
    assert "fwd.bucket0.bwd" in dag.nodes
    assert "fwd.splitMB1.bucket0" in dag.nodes
    assert "fwd.splitMB1.bucket0.bwd" in dag.nodes
    assert dag.nodes["fwd.bucket0"].node_meta["bucket_key"] == "fwd.bucket0"
    assert (
        dag.nodes["fwd.splitMB1.bucket0"].node_meta["bucket_key"]
        == "fwd.bucket0"
    )
    assert dag.nodes["fwd.bucket1"].node_meta["bucket_key"] == "fwd.bucket1"
    assert (
        dag.nodes["fwd.splitMB1.bucket1"].node_meta["bucket_key"]
        == "fwd.bucket1"
    )

    for node in dag.nodes.values():
        if node.node_kind == "COMPUTE":
            node.device = [0, 2]
    directives._insert_reduce_comm_nodes(dag, [{"PP": 0}], [0, 2])

    reductions = [
        node for node in dag.nodes.values() if node.node_kind == "REDUCE_COMM"
    ]
    assert len(reductions) == 2
    assert sorted(len(node.node_meta["bwd_uids"]) for node in reductions) == [2, 2]


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


def test_apply_order_directive_rejects_cross_device_order_edge() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("rank0", {"slot": 0}))
    dag.add_node(_compute("rank1", {"slot": 1}))
    dag.nodes["rank1"].device = [1]

    with pytest.raises(ValueError, match="crosses device placement"):
        _apply_order_directive(
            dag,
            [
                [{"slot": 0}],
                [{"slot": 1}],
            ],
            directive_idx=4,
        )


def test_shard_removes_zero_grad_lifetime_metadata_with_reduce_scatter() -> None:
    dag = TrainingDAG()
    node = TrainingDAGNode(
        uid="expert.bwd",
        node_kind="COMPUTE",
        compute_subkind="BWD",
        tag={"PP": 0, "EP": 0, "PASS": "B"},
        device=[0, 2],
        stream="default_stream",
        node_meta={"bucket_key": "expert", "fwd_uid": "expert.fwd"},
    )
    dag.add_node(node)

    directives._insert_reduce_scatter_comm_nodes(dag, [{"PP": 0, "EP": 0}], [0, 2])

    assert node.node_meta["zero_alloc_full_grads_before"] is True
    assert any(n.node_kind == "REDUCE_SCATTER_COMM" for n in dag.nodes.values())

    directives._insert_shard_a2a_comm_nodes(dag, [{"PP": 0, "EP": 0}], [0, 2])

    assert "zero_alloc_full_grads_before" not in node.node_meta
    assert not any(n.node_kind == "REDUCE_SCATTER_COMM" for n in dag.nodes.values())


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
    expert.node_meta["a2a_boundary_after"] = {"tensor_idx": 0}

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


def test_shard_requires_routing_counts_on_forward_expert_boundary() -> None:
    dag = TrainingDAG()
    pre = _compute("pre.fwd", {"PP": 0, "PASS": "F"})
    expert = _compute(
        "expert.fwd",
        {"PP": 0, "EP": 0, "PASS": "F"},
    )
    pre.device = expert.device = [0, 2]
    pre.node_meta["a2a_boundary_after"] = {"tensor_idx": 0}
    dag.add_node(pre)
    dag.add_node(expert)
    dag.add_edge(TrainingDAGEdge("pre.fwd", "expert.fwd", "data"))

    with pytest.raises(ValueError, match="requires a routing-count tensor"):
        directives._insert_shard_a2a_comm_nodes(
            dag,
            [{"PP": 0, "EP": 0}],
            [0, 2],
        )


def test_validate_schedule_tags_exist_rejects_tags_missing_from_model() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))

    directives = [
        {"op": "place", "filter": {"TP": 0}, "devices": [0]},
        {"op": "order", "filters": [[{"PP": 0, "MB": 0, "PASS": "F"}]]},
    ]

    with pytest.raises(ValueError, match="not found in model annotations"):
        _validate_schedule_tags_exist(dag, directives)


def test_validate_schedule_tags_exist_ignores_runtime_tags() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))

    _validate_schedule_tags_exist(
        dag,
        [
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
            {"op": "order", "filters": [[{"PP": 0, "MB": 0, "PASS": "F"}]]},
        ],
    )


def test_apply_schedule_directives_rejects_unmatched_place_directive() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))

    with pytest.raises(ValueError, match="matched zero nodes"):
        apply_schedule_directives(
            dag,
            [{"op": "place", "filter": {"PP": 1}, "devices": [0]}],
        )
