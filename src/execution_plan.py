"""Lower a :class:`TrainingDAG` into actor-local stream dependencies.

The CUDA executor and the interference-free simulator both consume this
representation.  Keeping the lowering CUDA-free makes the dependency model
testable without a GPU and, more importantly, prevents the simulator from
quietly drifting away from the ordering implemented by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .dag import TrainingDAG
from .ordering import _serial_topological_order


_BACKWARD_SUBKINDS = {"BWD", "BWD_I", "BWD_W"}
_UPDATE_SYNC_KINDS = {"REDUCE_COMM", "REDUCE_SCATTER_COMM"}
_P2P_KINDS = {"SEND_COMM", "RECV_COMM"}
_ROUTED_KINDS = {"M_TO_N_COMM", "N_TO_M_COMM"}
_RELAXABLE_DIRECTIONAL_COMM_REASONS = {
    "stream_fifo",
    "same_stream_temporal",
    "cross_stream_temporal",
}
_P2P_DIRECTIONS = {"pp_lo_hi", "pp_hi_lo"}
_ROUTED_DIRECTION_BY_KIND = {
    "M_TO_N_COMM": "m_to_n",
    "N_TO_M_COMM": "n_to_m",
}


@dataclass(frozen=True)
class PlannedOperation:
    """One dispatched operation and the completions that gate its start.

    ``completion_predecessors`` and ``dependency_reasons`` are parallel
    tuples.  A reason can contain ``+`` when the same predecessor establishes
    more than one ordering relation (for example, a temporal edge and stream
    FIFO order).
    """

    uid: str
    dispatch_index: int
    logical_stream: str
    completion_predecessors: tuple[str, ...]
    dependency_reasons: tuple[str, ...]

    def reason_for(self, predecessor_uid: str) -> str:
        """Return the dependency reason for ``predecessor_uid``."""
        try:
            index = self.completion_predecessors.index(predecessor_uid)
        except ValueError as exc:
            raise KeyError(predecessor_uid) from exc
        return self.dependency_reasons[index]


def is_relaxable_directional_communication_ordering_dependency(
    predecessor: object,
    node: object,
    dependency_reason: str,
) -> bool:
    """Whether a logical ordering wait may be omitted across physical streams.

    Schedule lowering can materialize logical ``pp_stream`` order as temporal
    DAG edges in addition to stream-FIFO dependencies. P2P and cross-group
    routed communication run on dedicated physical streams for their opposite
    directions. Retaining synthetic serialization between those streams can
    create a rendezvous progress cycle. Only ordering-only reasons are
    relaxable; data and future synchronization dependencies are preserved.
    """
    reason_components = set(str(dependency_reason).split("+"))
    if (
        not reason_components
        or not reason_components <= _RELAXABLE_DIRECTIONAL_COMM_REASONS
    ):
        return False

    predecessor_kind = getattr(predecessor, "node_kind", None)
    node_kind = getattr(node, "node_kind", None)
    if predecessor_kind in _P2P_KINDS and node_kind in _P2P_KINDS:
        predecessor_direction = (
            getattr(predecessor, "node_meta", {}) or {}
        ).get("comm_group_kind")
        node_direction = (getattr(node, "node_meta", {}) or {}).get(
            "comm_group_kind"
        )
        return (
            predecessor_direction in _P2P_DIRECTIONS
            and node_direction in _P2P_DIRECTIONS
            and predecessor_direction != node_direction
        )

    if predecessor_kind in _ROUTED_KINDS and node_kind in _ROUTED_KINDS:
        return (
            _ROUTED_DIRECTION_BY_KIND[predecessor_kind]
            != _ROUTED_DIRECTION_BY_KIND[node_kind]
        )
    return False


def is_relaxable_p2p_ordering_dependency(
    predecessor: object,
    node: object,
    dependency_reason: str,
) -> bool:
    """Compatibility wrapper for all directional communication dependencies."""
    return is_relaxable_directional_communication_ordering_dependency(
        predecessor,
        node,
        dependency_reason,
    )


def _execution_stream_key(node: object) -> str:
    """Return the FIFO domain used by the ordinary CUDA executor."""
    if getattr(node, "node_kind", None) in _P2P_KINDS:
        direction = (getattr(node, "node_meta", {}) or {}).get(
            "comm_group_kind"
        )
        if direction in _P2P_DIRECTIONS:
            return str(direction)
    routed_direction = _ROUTED_DIRECTION_BY_KIND.get(
        getattr(node, "node_kind", None)
    )
    if routed_direction is not None:
        return routed_direction
    return str(getattr(node, "stream", None) or "default_stream")


def _validated_dispatch_order(
    dag: TrainingDAG,
    dispatch_order: Iterable[str] | None,
) -> list[str]:
    order = (
        list(_serial_topological_order(dag))
        if dispatch_order is None
        else list(dispatch_order)
    )
    expected = set(dag.nodes)
    actual = set(order)
    if len(order) != len(actual):
        raise ValueError("execution-plan dispatch order contains duplicate node uids")
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "execution-plan dispatch order does not match DAG nodes: "
            f"missing={missing}, unknown={unknown}"
        )

    dispatch_index = {uid: index for index, uid in enumerate(order)}
    for edge in dag.edges:
        if edge.dep_kind not in {"data", "temporal"}:
            raise ValueError(
                f"unsupported dependency kind {edge.dep_kind!r} on "
                f"{edge.src_uid} -> {edge.dst_uid}"
            )
        if dispatch_index[edge.src_uid] >= dispatch_index[edge.dst_uid]:
            raise ValueError(
                "execution-plan dispatch order violates DAG dependency "
                f"{edge.src_uid} -> {edge.dst_uid} ({edge.dep_kind})"
            )
    return order


def build_execution_plan(
    dag: TrainingDAG,
    dispatch_order: Iterable[str] | None = None,
) -> list[PlannedOperation]:
    """Materialize the local CUDA completion dependencies for ``dag``.

    The plan contains:

    * FIFO order between consecutive operations on a logical stream;
    * every data or temporal edge crossing logical streams;
    * the executor's RECV guard against the most recently dispatched compute
      operation on the receiving consumer's stream; and
    * all backward/reduction completions relevant to an update operation.

    Same-stream DAG edges are also retained as reasons.  They are redundant
    with FIFO when the logical and physical stream assignments match, but
    distinguish a real DAG constraint from FIFO when the two P2P directions
    are mapped onto separate physical streams.
    """
    order = _validated_dispatch_order(dag, dispatch_order)
    dispatch_index = {uid: index for index, uid in enumerate(order)}

    incoming_edges: dict[str, list[tuple[str, str]]] = {
        uid: [] for uid in order
    }
    for edge in dag.edges:
        src = dag.nodes[edge.src_uid]
        dst = dag.nodes[edge.dst_uid]
        locality = (
            "cross_stream"
            if _execution_stream_key(src) != _execution_stream_key(dst)
            else "same_stream"
        )
        incoming_edges[edge.dst_uid].append(
            (edge.src_uid, f"{locality}_{edge.dep_kind}")
        )
    for uid in incoming_edges:
        incoming_edges[uid].sort(key=lambda item: dispatch_index[item[0]])

    previous_on_stream: dict[str, str] = {}
    latest_compute_on_stream: dict[str, str] = {}
    previously_dispatched_update_dependencies: list[str] = []
    result: list[PlannedOperation] = []

    for index, uid in enumerate(order):
        node = dag.nodes[uid]
        logical_stream = _execution_stream_key(node)
        reasons_by_predecessor: dict[str, list[str]] = {}

        def add_dependency(predecessor_uid: str | None, reason: str) -> None:
            if predecessor_uid is None or predecessor_uid == uid:
                return
            reasons = reasons_by_predecessor.setdefault(predecessor_uid, [])
            if reason not in reasons:
                reasons.append(reason)

        add_dependency(previous_on_stream.get(logical_stream), "stream_fifo")
        for predecessor_uid, edge_reason in incoming_edges[uid]:
            add_dependency(predecessor_uid, edge_reason)

        if node.node_kind == "RECV_COMM":
            consumers = [
                dag.nodes[edge.dst_uid]
                for edge in dag.edges
                if edge.src_uid == uid
                and edge.dep_kind == "data"
                and dag.nodes[edge.dst_uid].node_kind == "COMPUTE"
            ]
            # A RECV normally has exactly one compute consumer.  Supporting
            # multiple consumers costs nothing and makes malformed/diagnostic
            # DAGs deterministic instead of depending on adjacency order.
            consumers.sort(key=lambda consumer: dispatch_index[consumer.uid])
            for consumer in consumers:
                consumer_stream = str(consumer.stream or "default_stream")
                add_dependency(
                    latest_compute_on_stream.get(consumer_stream),
                    "recv_consumer_stream_guard",
                )

        if node.node_kind == "UPD":
            for predecessor_uid in previously_dispatched_update_dependencies:
                predecessor = dag.nodes[predecessor_uid]
                if predecessor.node_kind in _UPDATE_SYNC_KINDS:
                    reason = "update_gradient_sync"
                else:
                    reason = "update_backward"
                add_dependency(predecessor_uid, reason)

        ordered_predecessors = sorted(
            reasons_by_predecessor,
            key=lambda predecessor_uid: dispatch_index[predecessor_uid],
        )
        result.append(
            PlannedOperation(
                uid=uid,
                dispatch_index=index,
                logical_stream=logical_stream,
                completion_predecessors=tuple(ordered_predecessors),
                dependency_reasons=tuple(
                    "+".join(reasons_by_predecessor[predecessor_uid])
                    for predecessor_uid in ordered_predecessors
                ),
            )
        )

        previous_on_stream[logical_stream] = uid
        if node.node_kind == "COMPUTE":
            latest_compute_on_stream[logical_stream] = uid
            if node.compute_subkind in _BACKWARD_SUBKINDS:
                previously_dispatched_update_dependencies.append(uid)
        elif node.node_kind in _UPDATE_SYNC_KINDS:
            previously_dispatched_update_dependencies.append(uid)

    return result


__all__ = [
    "PlannedOperation",
    "build_execution_plan",
    "is_relaxable_directional_communication_ordering_dependency",
    "is_relaxable_p2p_ordering_dependency",
]
