"""Global P2P and collective transaction ordering tests."""

from __future__ import annotations

import pytest

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.execution_plan import build_execution_plan
from src.ordering import (
    _serial_topological_order,
    assign_dispatch_levels,
    execution_stream_domain,
    resolve_total_order_per_stream,
)
from src.piper import (
    _collect_communication_transactions,
    _materialize_global_communication_issue_edges,
    _order_communication_transactions,
    _validate_global_communication_order,
    _validate_global_p2p_transaction_graph,
    _validate_p2p_sequences,
    _validate_routed_collective_sequences,
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








def _routed_actor_node(
    uid: str,
    *,
    node_kind: str,
    role: str,
    comm_id: str,
    local_devices: list[int],
    source_devices: list[int],
    destination_devices: list[int],
) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind=node_kind,
        compute_subkind=None,
        tag={"PASS": "F"},
        device=list(local_devices),
        stream="routed_stream",
        node_meta={
            "role": role,
            "direction": (
                "incoming" if node_kind == "M_TO_N_COMM" else "outgoing"
            ),
            "source_devices": list(source_devices),
            "destination_devices": list(destination_devices),
            "source_degree": len(source_devices),
            "destination_degree": len(destination_devices),
            "participants": sorted(source_devices + destination_devices),
            "expert_fwd_uid": f"expert.{comm_id}",
            "expert_input_arity": 2,
            "expert_output_arity": 1,
            "a2a_tensor_idx": 0,
            "routing_counts_idx": (
                1 if node_kind == "M_TO_N_COMM" else None
            ),
            "bucket_key": f"expert.{comm_id}",
            "comm_group_kind": "routed_cross_group",
            "comm_id": comm_id,
        },
    )


def _routed_actor_dag(
    local_devices: list[int],
    specs: list[tuple[str, str, str, str, list[int], list[int]]],
    order: list[str],
) -> TrainingDAG:
    dag = TrainingDAG()
    for uid, node_kind, role, comm_id, source_devices, destination_devices in specs:
        dag.add_node(
            _routed_actor_node(
                uid,
                node_kind=node_kind,
                role=role,
                comm_id=comm_id,
                local_devices=local_devices,
                source_devices=source_devices,
                destination_devices=destination_devices,
            )
        )
    for src_uid, dst_uid in zip(order, order[1:]):
        dag.add_edge(TrainingDAGEdge(src_uid, dst_uid, "temporal"))
    return dag










def _p2p_compute_node(uid: str, device: int) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={"PASS": "F"},
        device=[device],
        stream="default_stream",
    )


def _add_p2p_transaction(
    source_dag: TrainingDAG,
    destination_dag: TrainingDAG,
    *,
    comm_id: str,
    source_compute_uid: str,
    destination_compute_uid: str,
    source_rank: int,
    destination_rank: int,
) -> tuple[str, str]:
    direction = "lo_hi" if source_rank < destination_rank else "hi_lo"
    common_meta = {
        "comm_id": comm_id,
        "comm_group_kind": f"pp_{direction}",
        "participants": sorted([source_rank, destination_rank]),
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
            tag={"PASS": "F"},
            device=[source_rank],
            stream="pp_stream",
            node_meta={**common_meta, "peer_pp_rank": destination_rank},
        )
    )
    destination_dag.add_node(
        TrainingDAGNode(
            uid=recv_uid,
            node_kind="RECV_COMM",
            compute_subkind=None,
            tag={"PASS": "F"},
            device=[destination_rank],
            stream="pp_stream",
            node_meta={**common_meta, "peer_pp_rank": source_rank},
        )
    )
    source_dag.add_edge(
        TrainingDAGEdge(source_compute_uid, send_uid, "data")
    )
    destination_dag.add_edge(
        TrainingDAGEdge(recv_uid, destination_compute_uid, "data")
    )
    return send_uid, recv_uid


def test_p2p_order_prefers_earlier_consumer_over_comm_id() -> None:
    destination_dag = TrainingDAG()
    source_dag = TrainingDAG()
    for dag, device, prefix in (
        (destination_dag, 0, "destination"),
        (source_dag, 1, "source"),
    ):
        dag.add_node(_p2p_compute_node(f"{prefix}.early", device))
        dag.add_node(_p2p_compute_node(f"{prefix}.late", device))
        dag.add_edge(
            TrainingDAGEdge(
                f"{prefix}.early",
                f"{prefix}.late",
                "temporal",
            )
        )
    early_send, early_recv = _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="z.early",
        source_compute_uid="source.early",
        destination_compute_uid="destination.early",
        source_rank=1,
        destination_rank=0,
    )
    late_send, late_recv = _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="a.late",
        source_compute_uid="source.late",
        destination_compute_uid="destination.late",
        source_rank=1,
        destination_rank=0,
    )

    _order_communication_transactions([destination_dag, source_dag])
    for dag in (destination_dag, source_dag):
        resolve_total_order_per_stream(dag)
    _validate_global_p2p_transaction_graph(
        [destination_dag, source_dag]
    )
    _validate_p2p_sequences([destination_dag, source_dag])

    assert any(
        edge.src_uid == early_send and edge.dst_uid == late_send
        for edge in source_dag.edges
    )
    assert any(
        edge.src_uid == early_recv and edge.dst_uid == late_recv
        for edge in destination_dag.edges
    )
    assert [
        node.node_meta["comm_id"]
        for node in (
            destination_dag.nodes[uid]
            for uid in _serial_topological_order(destination_dag)
        )
        if node.node_kind == "RECV_COMM"
        and execution_stream_domain(node) == "pp_hi_lo"
    ] == ["z.early", "a.late"]

def test_p2p_order_keeps_opposite_directions_independent() -> None:
    dag_0 = TrainingDAG()
    dag_1 = TrainingDAG()
    for uid in ("compute.0.source", "compute.0.destination"):
        dag_0.add_node(_p2p_compute_node(uid, 0))
    for uid in ("compute.1.source", "compute.1.destination"):
        dag_1.add_node(_p2p_compute_node(uid, 1))
    one_to_zero_send, one_to_zero_recv = _add_p2p_transaction(
        dag_1,
        dag_0,
        comm_id="one_to_zero",
        source_compute_uid="compute.1.source",
        destination_compute_uid="compute.0.destination",
        source_rank=1,
        destination_rank=0,
    )
    zero_to_one_send, zero_to_one_recv = _add_p2p_transaction(
        dag_0,
        dag_1,
        comm_id="zero_to_one",
        source_compute_uid="compute.0.source",
        destination_compute_uid="compute.1.destination",
        source_rank=0,
        destination_rank=1,
    )

    _order_communication_transactions([dag_0, dag_1])

    edges = {
        (edge.src_uid, edge.dst_uid)
        for dag in (dag_0, dag_1)
        for edge in dag.edges
    }
    assert (one_to_zero_send, zero_to_one_recv) not in edges
    assert (zero_to_one_send, one_to_zero_recv) not in edges


def test_p2p_order_rejects_cyclic_directional_transaction_graph() -> None:
    dag_0, dag_1, dag_2 = TrainingDAG(), TrainingDAG(), TrainingDAG()
    for dag, device, uids in (
        (dag_0, 0, ("a.source", "c.source")),
        (dag_1, 1, ("a.destination", "b.source")),
        (dag_2, 2, ("b.destination", "c.destination")),
    ):
        for uid in uids:
            dag.add_node(_p2p_compute_node(uid, device))
    send_a, recv_a = _add_p2p_transaction(
        dag_0,
        dag_1,
        comm_id="a",
        source_compute_uid="a.source",
        destination_compute_uid="a.destination",
        source_rank=0,
        destination_rank=1,
    )
    send_b, recv_b = _add_p2p_transaction(
        dag_1,
        dag_2,
        comm_id="b",
        source_compute_uid="b.source",
        destination_compute_uid="b.destination",
        source_rank=1,
        destination_rank=2,
    )
    send_c, recv_c = _add_p2p_transaction(
        dag_0,
        dag_2,
        comm_id="c",
        source_compute_uid="c.source",
        destination_compute_uid="c.destination",
        source_rank=0,
        destination_rank=2,
    )
    dag_0.add_edge(TrainingDAGEdge(send_c, send_a, "temporal"))
    dag_1.add_edge(TrainingDAGEdge(recv_a, send_b, "temporal"))
    dag_2.add_edge(TrainingDAGEdge(recv_b, recv_c, "temporal"))

    with pytest.raises(
        ValueError,
        match="global communication transaction graph is cyclic.*a.*b.*c",
    ):
        _order_communication_transactions([dag_0, dag_1, dag_2])




def test_p2p_validation_rejects_participant_mismatch() -> None:
    destination_dag = TrainingDAG()
    source_dag = TrainingDAG()
    destination_dag.add_node(_p2p_compute_node("destination", 0))
    source_dag.add_node(_p2p_compute_node("source", 1))
    send_uid, _recv_uid = _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="transfer",
        source_compute_uid="source",
        destination_compute_uid="destination",
        source_rank=1,
        destination_rank=0,
    )
    source_dag.nodes[send_uid].node_meta["participants"] = [0, 1, 2]

    with pytest.raises(ValueError, match="participants"):
        _validate_global_p2p_transaction_graph(
            [destination_dag, source_dag]
        )






def test_routed_collective_sequence_validation_rejects_mismatch() -> None:
    dense = [0, 2]
    expert = [1, 3]
    dense_specs = [
        ("m0.src", "M_TO_N_COMM", "source", "m0", dense, expert),
        ("m1.src", "M_TO_N_COMM", "source", "m1", dense, expert),
    ]
    expert_specs = [
        ("m1.dst", "M_TO_N_COMM", "destination", "m1", dense, expert),
        ("m0.dst", "M_TO_N_COMM", "destination", "m0", dense, expert),
    ]
    dense_dag = _routed_actor_dag(
        dense,
        dense_specs,
        ["m0.src", "m1.src"],
    )
    expert_dag = _routed_actor_dag(
        expert,
        expert_specs,
        ["m1.dst", "m0.dst"],
    )

    with pytest.raises(ValueError, match="call-sequence mismatch"):
        _validate_routed_collective_sequences([dense_dag, expert_dag])







