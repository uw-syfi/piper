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
    _serial_topological_order,
    _validate_schedule_tags_exist,
    _validate_split_backward_order_stencil,
)
import src.directives as directives


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
