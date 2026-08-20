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


def _backward_split_gm(
    *,
    parameter_only_output: bool,
    unknown_tail_after_boundary: bool = False,
) -> fx.GraphModule:
    graph = fx.Graph()
    activation = graph.placeholder("activation")
    parameter = graph.placeholder("parameter")
    activation.meta["example_value"] = torch.empty(
        2, 2, device="meta", requires_grad=True
    )
    parameter.meta["example_value"] = torch.empty(
        2, 2, device="meta", requires_grad=True
    )
    coupled = graph.call_function(torch.mul, (activation, parameter))
    coupled.meta["example_value"] = torch.empty(
        2, 2, device="meta", requires_grad=True
    )
    if unknown_tail_after_boundary:
        coupled = graph.call_function(torch.sin, (coupled,))
    if parameter_only_output:
        independent = graph.call_function(torch.square, (parameter,))
        independent.meta["example_value"] = torch.empty(
            2, 2, device="meta", requires_grad=True
        )
        graph.output((coupled, independent))
    else:
        graph.output(coupled)
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


def test_indexed_edge_removal_preserves_parallel_edge_adjacency() -> None:
    dag = TrainingDAG()
    for uid in ("a", "b", "c"):
        dag.add_node(_compute(uid, {"name": uid}))
    data_edge = TrainingDAGEdge("a", "b", "data", "activation")
    temporal_edge = TrainingDAGEdge("a", "b", "temporal")
    retained_edge = TrainingDAGEdge("a", "c", "data")
    for edge in (data_edge, temporal_edge, retained_edge):
        dag.add_edge(edge)

    assert dag.outgoing_edges("a") == [data_edge, temporal_edge, retained_edge]
    assert dag.incoming_edges("b", dep_kind="data") == [data_edge]
    _remove_edge(dag, data_edge)

    assert dag.succs["a"] == {"b", "c"}
    assert dag.preds["b"] == {"a"}
    assert dag.incoming_edges("b") == [temporal_edge]

    _remove_nodes_and_incident_edges(dag, {"b"})

    assert set(dag.nodes) == {"a", "c"}
    assert dag.edges == [retained_edge]
    assert dag.succs == {"a": {"c"}, "c": set()}
    assert dag.preds == {"a": set(), "c": {"a"}}


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


def test_place_accepts_repeated_normalized_device_set() -> None:
    dag = TrainingDAG()
    node = _compute("a", {"REGION": 0})
    node.device = None
    dag.add_node(node)

    apply_schedule_directives(
        dag,
        [
            {"op": "place", "filter": {"REGION": 0}, "devices": [2, 0]},
            {"op": "place", "filter": {"REGION": "*"}, "devices": [0, 2]},
        ],
    )

    assert dag.nodes["a"].device == [0, 2]


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


@pytest.mark.parametrize(
    ("source_devices", "destination_devices", "error"),
    [
        ([0, 2], [1], "equal-width actor groups"),
        ([0, 1], [1, 2], "disjoint actor groups"),
    ],
)
def test_generic_p2p_rejects_invalid_actor_group_pair(
    source_devices: list[int],
    destination_devices: list[int],
    error: str,
) -> None:
    dag = TrainingDAG()
    source = _compute("source", {"REGION": 0})
    destination = _compute("destination", {"REGION": 1})
    source.device = source_devices
    destination.device = destination_devices
    dag.add_node(source)
    dag.add_node(destination)
    dag.add_edge(TrainingDAGEdge("source", "destination", "data", "activation"))

    with pytest.raises(ValueError, match=error):
        directives._insert_send_recv_comm_nodes(dag)


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


def test_tag_index_matches_legacy_filter_semantics_and_node_order() -> None:
    dag = TrainingDAG()
    for uid, tag in (
        ("first", {"PP": 0, "EP": None, "PASS": "F"}),
        ("second", {"PP": 0, "EP": 3, "PASS": "F"}),
        ("third", {"PP": 1, "EP": 3, "PASS": "B"}),
        ("fourth", {"PP": 1, "PASS": "F"}),
    ):
        dag.add_node(_compute(uid, tag))

    filters = (
        {},
        {"PP": 0},
        {"EP": directives._ANY_TAG_INDEX},
        {"PP": 0, "EP": directives._ANY_TAG_INDEX},
        {"EP": directives._ABSENT_TAG_INDEX},
        {"PP": 1, "EP": directives._ABSENT_TAG_INDEX},
        {"PASS": "missing"},
    )
    for tag_filter in filters:
        expected = [
            uid
            for uid, node in dag.nodes.items()
            if directives._match_filter(node.tag, tag_filter)
        ]
        assert dag.matching_uids(
            tag_filter,
            wildcard_value=directives._ANY_TAG_INDEX,
            absent_value=directives._ABSENT_TAG_INDEX,
        ) == expected

    dag.update_node_tag("first", {"PP": 1, "EP": 2, "PASS": "F"})
    assert dag.matching_uids(
        {"PP": 1},
        wildcard_value=directives._ANY_TAG_INDEX,
        absent_value=directives._ABSENT_TAG_INDEX,
    ) == ["first", "third", "fourth"]


def test_place_filter_dash_matches_only_nodes_without_tag() -> None:
    dag = TrainingDAG()
    for uid, tag in (
        ("input", {"PP": 0, "PASS": "F"}),
        ("attn", {"PP": 0, "ATTN": 0, "PASS": "F"}),
        ("ffn", {"PP": 0, "FFN": 0, "PASS": "F"}),
        ("output", {"PP": 1, "PASS": "F"}),
    ):
        dag.add_node(_compute(uid, tag))

    apply_schedule_directives(
        dag,
        [
            {
                "op": "place",
                "filter": {"PP": 0, "ATTN": "-", "FFN": "-"},
                "devices": [0],
            },
            {
                "op": "place",
                "filter": {"PP": 1, "ATTN": "-", "FFN": "-"},
                "devices": [1],
            },
            {"op": "place", "filter": {"ATTN": "*"}, "devices": [0]},
            {"op": "place", "filter": {"FFN": "*"}, "devices": [1]},
        ],
    )

    assert dag.nodes["input"].device == [0]
    assert dag.nodes["attn"].device == [0]
    assert dag.nodes["ffn"].device == [1]
    assert dag.nodes["output"].device == [1]


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


def test_zero_parameter_lifetimes_add_missing_reachability_edge() -> None:
    dag, chains = _two_parameter_lifetime_chains("unordered")

    _add_inter_chain_temporal_edges(dag, chains)

    assert any(
        edge.src_uid == "first"
        and edge.dst_uid == "all_gather.1"
        and edge.dep_kind == "temporal"
        for edge in dag.edges
    )
    assign_dispatch_levels(dag)
    dispatch_order = _dispatch_order(dag)
    assert dispatch_order.index("first") < dispatch_order.index("all_gather.1")


def test_zero_parameter_lifetimes_retain_existing_reachability() -> None:
    dag, chains = _two_parameter_lifetime_chains("forward")
    existing_edges = list(dag.edges)

    _add_inter_chain_temporal_edges(dag, chains)

    assert dag.edges == existing_edges


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


def _alternating_bucket_lifetime_dag(
    *,
    same_bucket: bool,
) -> TrainingDAG:
    dag = TrainingDAG()
    uids = ("a0", "b0", "a1", "b1")
    for index, uid in enumerate(uids):
        node = _compute(uid, {"PP": 0, "INDEX": index, "PASS": "F"})
        node.node_meta["bucket_key"] = (
            "shared" if same_bucket else uid[0]
        )
        dag.add_node(node)
    for src_uid, dst_uid in zip(uids, uids[1:]):
        dag.add_edge(TrainingDAGEdge(src_uid, dst_uid, "temporal"))
    directives._insert_all_gather_comm_nodes(dag, [{}], [0])
    return dag


def test_zero_parameter_lifetimes_do_not_connect_through_other_buckets() -> None:
    dag = _alternating_bucket_lifetime_dag(same_bucket=False)

    chains = _prune_zero_lifetime_metadata(dag)

    assert chains.gradient == []
    assert sorted(chains.parameter) == [["a0"], ["a1"], ["b0"], ["b1"]]
    assert sum(
        node.node_kind == "ALL_GATHER_COMM" for node in dag.nodes.values()
    ) == 4
    assert all(
        dag.nodes[uid].node_meta.get("zero_free_full_params_after") is True
        for uid in ("a0", "b0", "a1", "b1")
    )


def test_zero_parameter_lifetimes_coalesce_direct_same_bucket_chain() -> None:
    dag = _alternating_bucket_lifetime_dag(same_bucket=True)

    chains = _prune_zero_lifetime_metadata(dag)

    assert chains.gradient == []
    assert chains.parameter == [["a0", "b0", "a1", "b1"]]
    assert sum(
        node.node_kind == "ALL_GATHER_COMM" for node in dag.nodes.values()
    ) == 1
    assert dag.nodes["b1"].node_meta["zero_free_full_params_after"] is True
    assert all(
        "zero_free_full_params_after" not in dag.nodes[uid].node_meta
        for uid in ("a0", "b0", "a1")
    )


def test_zero_parameter_lifetimes_split_branched_component_into_direct_chains() -> None:
    dag = TrainingDAG()
    for uid in ("first", "second", "third", "fourth"):
        node = _compute(uid, {"PP": 0, "PASS": "F"})
        node.node_meta["bucket_key"] = "shared"
        dag.add_node(node)
    for src_uid, dst_uid in (
        ("first", "third"),
        ("second", "fourth"),
        ("third", "fourth"),
    ):
        dag.add_edge(TrainingDAGEdge(src_uid, dst_uid, "temporal"))
    directives._insert_all_gather_comm_nodes(dag, [{}], [0])

    chains = _prune_zero_lifetime_metadata(dag)

    assert {uid for chain in chains.parameter for uid in chain} == {
        "first", "second", "third", "fourth",
    }
    assert len(chains.parameter) > 1
    assert sum(
        node.node_kind == "ALL_GATHER_COMM" for node in dag.nodes.values()
    ) == len(chains.parameter)
    for chain in chains.parameter:
        for src_uid, dst_uid in zip(chain, chain[1:]):
            assert any(
                edge.src_uid == src_uid
                and edge.dst_uid == dst_uid
                and edge.dep_kind in ("data", "temporal")
                for edge in dag.edges
            )


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


def test_bucketed_split_backward_weight_phase_stays_bucket_local(monkeypatch) -> None:
    dag = _single_bucket_training_dag()
    fwd = dag.nodes["fwd"]
    fwd.node_meta.update(
        {
            "stage_id": 0,
            "segment_id": 0,
            "gm": _dummy_gm(),
            "graphargs": [],
            "input_idxs": [],
            "param_idxs": [],
        }
    )
    monkeypatch.setattr(
        directives,
        "bucket_stage",
        lambda *_args, **_kwargs: [
            (_dummy_gm(), [], [], []),
            (_dummy_gm(), [], [], []),
        ],
    )

    _bucket_matched_fwd_nodes(dag, [{"PP": 0}], 1)
    split_keys = _validate_split_backward_order_stencil(
        [
            {
                "op": "order",
                "filters": [
                    [{"PP": 0, "PASS": "BI"}],
                    [{"PP": 0, "PASS": "BW"}],
                ],
            }
        ]
    )
    _apply_split_backward_stencil(dag, split_keys)

    weight_nodes = [
        node for node in dag.nodes.values()
        if node.compute_subkind == "BWD_W"
    ]
    assert len(weight_nodes) == 2
    for weight_node in weight_nodes:
        input_edges = dag.incoming_edges(weight_node.uid, dep_kind="data")
        assert len(input_edges) == 1
        input_node = dag.nodes[input_edges[0].src_uid]
        assert input_node.compute_subkind == "BWD_I"
        assert input_node.node_meta["bucket_key"] == weight_node.node_meta["bucket_key"]
        assert input_node.node_meta["fwd_uid"] == weight_node.node_meta["fwd_uid"]
        assert any(
            edge.dst_uid == "upd" and edge.dep_kind == "temporal"
            for edge in dag.outgoing_edges(weight_node.uid)
        )


@pytest.mark.parametrize(
    (
        "parameter_only_output",
        "differentiable_input",
        "provable_path",
        "eligible",
    ),
    [
        (False, True, True, True),
        (True, True, True, False),
        (False, False, True, False),
        (False, True, False, False),
    ],
)
def test_split_backward_requires_capturable_parameter_paths(
    parameter_only_output: bool,
    differentiable_input: bool,
    provable_path: bool,
    eligible: bool,
) -> None:
    dag = _single_bucket_training_dag()
    gm = _backward_split_gm(
        parameter_only_output=parameter_only_output
    )
    if not provable_path:
        for node in gm.graph.nodes:
            if node.op not in ("placeholder", "output"):
                node.meta.pop("example_value", None)
    dag.nodes["fwd"].node_meta.update(
        {
            "gm": gm,
            "graphargs": [
                torch.empty(
                    2,
                    2,
                    device="meta",
                    requires_grad=differentiable_input,
                ),
                torch.empty(2, 2, device="meta", requires_grad=True),
            ],
            "input_idxs": [0],
            "param_idxs": [1],
        }
    )
    split_keys = {(('PP', 0),)}

    if not eligible:
        with pytest.raises(
            ValueError,
            match=r"cannot split backward node 'bwd'.*parameter",
        ):
            _apply_split_backward_stencil(dag, split_keys)
    else:
        _apply_split_backward_stencil(dag, split_keys)
        assert dag.nodes["bwd"].compute_subkind == "BWD_I"
        assert dag.nodes["bwd.bw"].compute_subkind == "BWD_W"


def test_split_backward_does_not_require_proof_after_captured_boundary() -> None:
    dag = _single_bucket_training_dag()
    dag.nodes["fwd"].node_meta.update(
        {
            "gm": _backward_split_gm(
                parameter_only_output=False,
                unknown_tail_after_boundary=True,
            ),
            "graphargs": [
                torch.empty(2, 2, device="meta", requires_grad=True),
                torch.empty(2, 2, device="meta", requires_grad=True),
            ],
            "input_idxs": [0],
            "param_idxs": [1],
        }
    )

    _apply_split_backward_stencil(dag, {(('PP', 0),)})

    assert dag.nodes["bwd"].compute_subkind == "BWD_I"
    assert dag.nodes["bwd.bw"].compute_subkind == "BWD_W"


def test_apply_schedule_buckets_base_graph_before_microbatch_split(monkeypatch) -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))
    dag.add_node(_bwd("bwd", "fwd", {"PP": 0, "PASS": "B"}))
    dag.add_edge(TrainingDAGEdge("fwd", "bwd", "data"))
    calls: list[tuple[list[str], list[dict], int]] = []

    def record_bucket_call(actual_dag, filters, bucket_size):
        assert all("MB" not in node.tag for node in actual_dag.nodes.values())
        calls.append((list(actual_dag.nodes), filters, bucket_size))

    monkeypatch.setattr(
        directives,
        "_bucket_matched_fwd_nodes",
        record_bucket_call,
    )

    directives.apply_schedule_directives(
        dag,
        [
            {
                "op": "place",
                "filter": {"PP": 0},
                "devices": [0],
                "stream": "pp_stream",
            },
            {
                "op": "replicate",
                "filter": {"PP": 0},
                "devices": [0],
                "reduce_stream": "dp_stream",
                "bucket_size": 16,
            },
            {
                "op": "split",
                "filter": {},
                "dim_name": "MB",
                "num_microbatches": 2,
            },
        ],
    )

    assert calls == [(["fwd", "bwd"], [{"PP": 0}], 16)]
    assert {node.tag["MB"] for node in dag.nodes.values()} == {0, 1}


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


def test_apply_order_directive_rejects_disjoint_device_order_edge() -> None:
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


def test_apply_order_directive_allows_overlapping_device_sets() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("attention", {"slot": 0}))
    dag.add_node(_compute("embedding", {"slot": 1}))
    dag.nodes["attention"].device = [0, 2, 4, 5]
    dag.nodes["embedding"].device = [0]

    _apply_order_directive(
        dag,
        [
            [{"slot": 0}],
            [{"slot": 1}],
        ],
        directive_idx=4,
    )

    assert any(
        edge.src_uid == "attention"
        and edge.dst_uid == "embedding"
        and edge.dep_kind == "temporal"
        for edge in dag.edges
    )


def test_shard_retains_zero_grad_lifetime_metadata_with_reduce_scatter() -> None:
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

    assert node.node_meta["zero_alloc_full_grads_before"] is True
    assert any(n.node_kind == "REDUCE_SCATTER_COMM" for n in dag.nodes.values())


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


def test_validate_schedule_tags_exist_allows_absent_tag_filter() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))

    _validate_schedule_tags_exist(
        dag,
        [{"op": "place", "filter": {"EP": "-"}, "devices": [0]}],
    )


def test_apply_schedule_directives_rejects_unmatched_place_directive() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("fwd", {"PP": 0, "PASS": "F"}))

    with pytest.raises(ValueError, match="matched zero nodes"):
        apply_schedule_directives(
            dag,
            [{"op": "place", "filter": {"PP": 1}, "devices": [0]}],
        )
