from __future__ import annotations

import pytest

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.execution_plan import build_execution_plan
from src.ordering import assign_dispatch_levels, resolve_total_order_per_stream
from src.piper import (
    _collect_communication_transactions,
    _materialize_global_communication_issue_edges,
    _order_communication_transactions,
    _validate_global_communication_order,
)


def _compute(uid: str, rank: int) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={},
        device=[rank],
        stream="default_stream",
    )


def _collective(
    uid: str,
    kind: str,
    comm_id: str,
    devices: list[int],
    participants: list[int],
    *,
    stream: str,
    role: str | None = None,
) -> TrainingDAGNode:
    meta = {
        "comm_id": comm_id,
        "participants": participants,
        "comm_group_kind": (
            "routed_cross_group"
            if kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
            else "dp"
        ),
    }
    if role is not None:
        meta["role"] = role
    return TrainingDAGNode(
        uid=uid,
        node_kind=kind,
        compute_subkind=None,
        tag={},
        device=devices,
        stream=stream,
        node_meta=meta,
    )


def _p2p_pair(
    source_dag: TrainingDAG,
    destination_dag: TrainingDAG,
    *,
    comm_id: str,
    source_rank: int,
    destination_rank: int,
    source_compute_uid: str,
    destination_compute_uid: str,
    channel: str = "dense_state_lo_hi",
) -> tuple[str, str]:
    participants = sorted((source_rank, destination_rank))
    common = {
        "comm_id": comm_id,
        "participants": participants,
        "comm_group_kind": channel,
        "source_compute_uid": source_compute_uid,
        "destination_compute_uid": destination_compute_uid,
    }
    send_uid = f"send.{comm_id}"
    recv_uid = f"recv.{comm_id}"
    source_dag.add_node(
        TrainingDAGNode(
            uid=send_uid,
            node_kind="SEND_COMM",
            compute_subkind=None,
            tag={},
            device=[source_rank],
            stream="pp_stream",
            node_meta={**common, "peer_pp_rank": destination_rank},
        )
    )
    destination_dag.add_node(
        TrainingDAGNode(
            uid=recv_uid,
            node_kind="RECV_COMM",
            compute_subkind=None,
            tag={},
            device=[destination_rank],
            stream="pp_stream",
            node_meta={**common, "peer_pp_rank": source_rank},
        )
    )
    source_dag.add_edge(
        TrainingDAGEdge(source_compute_uid, send_uid, "data")
    )
    destination_dag.add_edge(
        TrainingDAGEdge(recv_uid, destination_compute_uid, "data")
    )
    return send_uid, recv_uid


def _mixed_triangle() -> tuple[list[TrainingDAG], dict[str, str]]:
    # D spans actor groups 0 -> 2, M spans 0 -> 3, and N spans 3 -> 2.
    # Only M -> N is mandatory. A common transaction order must orient D
    # consistently at both of its endpoints rather than forming D -> M -> N -> D.
    dag0, dag2, dag3 = TrainingDAG(), TrainingDAG(), TrainingDAG()
    for dag, node in (
        (dag0, _compute("producer.0", 0)),
        (dag2, _compute("consumer.2", 2)),
        (dag3, _compute("between.3", 3)),
    ):
        dag.add_node(node)
    send_d, recv_d = _p2p_pair(
        dag0,
        dag2,
        comm_id="dense",
        source_rank=0,
        destination_rank=2,
        source_compute_uid="producer.0",
        destination_compute_uid="consumer.2",
    )
    m_source = _collective(
        "m.source",
        "M_TO_N_COMM",
        "m",
        [0],
        [0, 3],
        stream="ep_stream",
        role="source",
    )
    m_destination = _collective(
        "m.destination",
        "M_TO_N_COMM",
        "m",
        [3],
        [0, 3],
        stream="ep_stream",
        role="destination",
    )
    n_source = _collective(
        "n.source",
        "N_TO_M_COMM",
        "n",
        [3],
        [2, 3],
        stream="ep_stream",
        role="source",
    )
    n_destination = _collective(
        "n.destination",
        "N_TO_M_COMM",
        "n",
        [2],
        [2, 3],
        stream="ep_stream",
        role="destination",
    )
    dag0.add_node(m_source)
    dag3.add_node(m_destination)
    dag3.add_node(n_source)
    dag2.add_node(n_destination)
    dag3.add_edge(TrainingDAGEdge("m.destination", "between.3", "data"))
    dag3.add_edge(TrainingDAGEdge("between.3", "n.source", "data"))
    return [dag0, dag2, dag3], {
        "d_source": send_d,
        "d_destination": recv_d,
        "m_source": "m.source",
        "m_destination": "m.destination",
        "n_source": "n.source",
        "n_destination": "n.destination",
    }


def _finalize(dags: list[TrainingDAG]):
    order = _order_communication_transactions(dags)
    for dag in dags:
        resolve_total_order_per_stream(dag)
    _materialize_global_communication_issue_edges(dags, order)
    for dag in dags:
        assign_dispatch_levels(dag)
    _validate_global_communication_order(dags, order)
    return order


def test_mixed_global_order_adds_only_required_cross_stream_issue_edges() -> None:
    dags, _uids = _mixed_triangle()

    order = _finalize(dags)

    issue_edges = [
        edge for dag in dags for edge in dag.edges if edge.dep_kind == "issue"
    ]
    assert len(issue_edges) == 2
    assert all(
        dags[dag_index].nodes[edge.src_uid].stream
        != dags[dag_index].nodes[edge.dst_uid].stream
        for dag_index, dag in enumerate(dags)
        for edge in dag.edges
        if edge.dep_kind == "issue"
    )
    global_index = {
        identity: index for index, identity in enumerate(order.identities)
    }
    transactions, _ = _collect_communication_transactions(dags)
    for dag_index, dag in enumerate(dags):
        local = sorted(
            (
                global_index[identity],
                endpoint.uid,
            )
            for identity, transaction in transactions.items()
            for endpoint in transaction.endpoints
            if endpoint.dag_index == dag_index
        )
        dispatch = {
            operation.uid: operation.dispatch_index
            for operation in build_execution_plan(dag)
        }
        assert [
            dag.nodes[uid].node_meta["global_communication_index"]
            for _index, uid in local
        ] == [index for index, _uid in local]
        assert [dispatch[uid] for _index, uid in local] == sorted(
            dispatch[uid] for _index, uid in local
        )


def test_mixed_mandatory_cycle_is_rejected_with_edge_provenance() -> None:
    dags, uids = _mixed_triangle()
    dags[0].add_edge(
        TrainingDAGEdge(uids["d_source"], uids["m_source"], "temporal")
    )
    dags[1].add_edge(
        TrainingDAGEdge(uids["n_destination"], uids["d_destination"], "temporal")
    )

    with pytest.raises(
        ValueError,
        match=r"global communication transaction graph is cyclic:.*dense.*m.*n",
    ):
        _order_communication_transactions(dags)


def test_existing_reachability_avoids_redundant_issue_edge() -> None:
    dag = TrainingDAG()
    first = _collective(
        "first", "A2A_COMM", "first", [0], [0], stream="stream.0"
    )
    second = _collective(
        "second", "REDUCE_COMM", "second", [0], [0], stream="stream.1"
    )
    dag.add_node(first)
    dag.add_node(second)
    dag.add_edge(TrainingDAGEdge("first", "second", "temporal"))

    _finalize([dag])

    assert not any(edge.dep_kind == "issue" for edge in dag.edges)


def test_same_stream_order_uses_temporal_fifo_not_issue_edge() -> None:
    dag = TrainingDAG()
    dag.add_node(
        _collective("a", "A2A_COMM", "a", [0], [0], stream="ep")
    )
    dag.add_node(
        _collective("b", "REDUCE_COMM", "b", [0], [0], stream="ep")
    )

    _finalize([dag])

    assert any(edge.dep_kind == "temporal" for edge in dag.edges)
    assert not any(edge.dep_kind == "issue" for edge in dag.edges)


def test_incompatible_textual_comm_id_reuse_is_rejected() -> None:
    dag = TrainingDAG()
    dag.add_node(
        _collective("a", "A2A_COMM", "shared", [0], [0], stream="ep")
    )
    dag.add_node(
        _collective(
            "reduce", "REDUCE_COMM", "shared", [0], [0], stream="dp"
        )
    )

    with pytest.raises(ValueError, match="comm_id 'shared' is reused"):
        _collect_communication_transactions([dag])


@pytest.mark.parametrize(
    "kind",
    [
        "A2A_COMM",
        "REDUCE_COMM",
        "ALL_REDUCE_COMM",
        "ALL_GATHER_COMM",
        "REDUCE_SCATTER_COMM",
    ],
)
def test_supported_collective_kinds_normalize_without_model_metadata(
    kind: str,
) -> None:
    dag = TrainingDAG()
    dag.add_node(
        _collective("endpoint", kind, "transaction", [0], [0], stream="comm")
    )

    transactions, endpoints = _collect_communication_transactions([dag])

    identity = (kind, "transaction")
    assert transactions[identity].participants == (0,)
    assert endpoints[(0, "endpoint")] == identity


def test_execution_plan_keeps_issue_separate_from_completion() -> None:
    dag = TrainingDAG()
    dag.add_node(_compute("first", 0))
    dag.add_node(_compute("second", 0))
    dag.add_edge(TrainingDAGEdge("first", "second", "issue"))
    assign_dispatch_levels(dag)

    plan = {operation.uid: operation for operation in build_execution_plan(dag)}

    assert plan["second"].issue_predecessors == ("first",)
    assert plan["second"].completion_predecessors == ()
