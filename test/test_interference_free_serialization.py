import pytest

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.ordering import _dispatch_order
from src.piper import _interference_free_serial_orders


def _compute(uid: str, device: int) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={"PASS": "F"},
        device=[device],
        stream="default_stream",
    )


def _p2p(
    uid: str,
    kind: str,
    *,
    device: int,
    comm_id: str,
    source_compute_uid: str,
    destination_compute_uid: str,
    direction: str,
) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind=kind,
        compute_subkind=None,
        tag={"PASS": "F"},
        device=[device],
        stream="pp_stream",
        node_meta={
            "comm_id": comm_id,
            "source_compute_uid": source_compute_uid,
            "destination_compute_uid": destination_compute_uid,
            "comm_group_kind": direction,
            "participants": [0, 1],
        },
    )


def _opposite_direction_p2p_dags(
    *,
    collapsed_cycle: bool = False,
) -> tuple[TrainingDAG, TrainingDAG]:
    dag0 = TrainingDAG()
    dag1 = TrainingDAG()

    a_source = _compute("a.source", 1)
    a_destination = _compute("a.destination", 0)
    b_source = _compute("b.source", 0)
    b_destination = _compute("b.destination", 1)
    a_send = _p2p(
        "a.send",
        "SEND_COMM",
        device=1,
        comm_id="a",
        source_compute_uid=a_source.uid,
        destination_compute_uid=a_destination.uid,
        direction="pp_hi_lo",
    )
    a_recv = _p2p(
        "a.recv",
        "RECV_COMM",
        device=0,
        comm_id="a",
        source_compute_uid=a_source.uid,
        destination_compute_uid=a_destination.uid,
        direction="pp_hi_lo",
    )
    b_send = _p2p(
        "b.send",
        "SEND_COMM",
        device=0,
        comm_id="b",
        source_compute_uid=b_source.uid,
        destination_compute_uid=b_destination.uid,
        direction="pp_lo_hi",
    )
    b_recv = _p2p(
        "b.recv",
        "RECV_COMM",
        device=1,
        comm_id="b",
        source_compute_uid=b_source.uid,
        destination_compute_uid=b_destination.uid,
        direction="pp_lo_hi",
    )

    for node in (a_destination, b_source, a_recv, b_send):
        dag0.add_node(node)
    for node in (a_source, b_destination, a_send, b_recv):
        dag1.add_node(node)
    dag0.add_edge(TrainingDAGEdge(a_recv.uid, a_destination.uid, "data"))
    dag0.add_edge(TrainingDAGEdge(b_source.uid, b_send.uid, "data"))
    dag1.add_edge(TrainingDAGEdge(a_source.uid, a_send.uid, "data"))
    dag1.add_edge(TrainingDAGEdge(b_recv.uid, b_destination.uid, "data"))
    if collapsed_cycle:
        dag0.add_edge(TrainingDAGEdge(a_recv.uid, b_send.uid, "temporal"))
        dag1.add_edge(TrainingDAGEdge(b_recv.uid, a_send.uid, "temporal"))
    return dag0, dag1


def _communication_ids(dag: TrainingDAG, order: tuple[str, ...]) -> list[str]:
    return [
        dag.nodes[uid].node_meta["comm_id"]
        for uid in order
        if dag.nodes[uid].node_kind in {"SEND_COMM", "RECV_COMM"}
    ]


def test_global_if_order_reconciles_opposite_local_p2p_orders() -> None:
    dag0, dag1 = _opposite_direction_p2p_dags()
    ordinary = (
        _communication_ids(dag0, tuple(_dispatch_order(dag0))),
        _communication_ids(dag1, tuple(_dispatch_order(dag1))),
    )
    assert ordinary[0] != ordinary[1]

    orders = _interference_free_serial_orders((dag0, dag1))

    assert _communication_ids(dag0, orders[0]) == _communication_ids(
        dag1, orders[1]
    )
    for dag, order in zip((dag0, dag1), orders):
        index = {uid: position for position, uid in enumerate(order)}
        assert set(order) == set(dag.nodes)
        assert all(index[edge.src_uid] < index[edge.dst_uid] for edge in dag.edges)


def test_global_if_order_rejects_transaction_collapsed_cycle() -> None:
    dags = _opposite_direction_p2p_dags(collapsed_cycle=True)

    with pytest.raises(
        ValueError,
        match="interference-free transaction graph is cyclic",
    ) as exc_info:
        _interference_free_serial_orders(dags)

    assert "('p2p', 'a', '')" in str(exc_info.value)
    assert "('p2p', 'b', '')" in str(exc_info.value)
    assert "actor_dag_members" in str(exc_info.value)


def test_global_if_order_groups_all_collective_participants() -> None:
    dags = []
    for device in (0, 1):
        dag = TrainingDAG()
        local = _compute(f"local.{device}", device)
        collective = TrainingDAGNode(
            uid=f"collective.{device}",
            node_kind="ALL_GATHER_COMM",
            compute_subkind=None,
            tag={"PASS": "F"},
            device=[device],
            stream="collective_stream",
            node_meta={"comm_id": "gather", "participants": [0, 1]},
        )
        dag.add_node(local)
        dag.add_node(collective)
        dags.append(dag)

    orders = _interference_free_serial_orders(tuple(dags))

    assert tuple(set(order) for order in orders) == (
        {"local.0", "collective.0"},
        {"local.1", "collective.1"},
    )
