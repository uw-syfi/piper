import heapq

from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    _ReachabilityIndex,
    _topological_levels,
    _topological_order,
)

_DEFAULT_STREAM = "default_stream"
_CRITICAL_PATH_COMM_KINDS = {
    "ALL_GATHER_COMM",
    "A2A_COMM",
    "M_TO_N_COMM",
    "N_TO_M_COMM",
}
_REDUCTION_COMM_KINDS = {
    "REDUCE_COMM",
    "REDUCE_SCATTER_COMM",
}
_DIRECTIONAL_P2P_DOMAINS = {
    "pp_lo_hi",
    "pp_hi_lo",
    "dense_state_lo_hi",
    "dense_state_hi_lo",
}


def execution_stream_domain(node: object) -> str:
    """Return the physical FIFO domain represented by a finalized DAG node.

    Schedule-declared streams are authoritative except for pipeline P2P, whose
    opposite directions deliberately use independent physical streams.
    """
    if getattr(node, "node_kind", None) in {"SEND_COMM", "RECV_COMM"}:
        direction = (getattr(node, "node_meta", {}) or {}).get(
            "comm_group_kind"
        )
        if direction in _DIRECTIONAL_P2P_DOMAINS:
            return str(direction)
    return str(getattr(node, "stream", None) or _DEFAULT_STREAM)


def _serial_topological_order(
    dag: TrainingDAG,
    topo_levels: dict[str, int] | None = None,
) -> list[str]:
    """Serialize topological levels into a deterministic dispatch order.

    Nodes with lower topological levels always come first. Within a level,
    priority order is: SEND > critical-path comm > reduction comm > compute/other > RECV.
    """
    if topo_levels is None:
        topo_levels = _topological_levels(dag)

    base_order = _topological_order(dag)
    topo_idx = {uid: i for i, uid in enumerate(base_order)}

    return sorted(
        dag.nodes,
        key=lambda uid: (
            topo_levels[uid],
            _serial_priority(dag, uid),
            topo_idx[uid],
        ),
    )


def _serial_priority(dag: TrainingDAG, uid: str) -> int:
    kind = dag.nodes[uid].node_kind
    if kind == "SEND_COMM":
        return 0
    if kind in _CRITICAL_PATH_COMM_KINDS:
        return 1
    if kind in _REDUCTION_COMM_KINDS:
        return 2
    if kind == "RECV_COMM":
        return 4
    return 3


def _dispatch_priority(dag: TrainingDAG, uid: str) -> int:
    kind = dag.nodes[uid].node_kind
    if kind == "SEND_COMM":
        return 0
    if kind.endswith("_COMM") and kind != "RECV_COMM":
        return 1
    if kind == "RECV_COMM":
        return 3
    return 2


def _compute_dispatch_levels(
    dag: TrainingDAG,
    *,
    require_p2p_anchors: bool,
) -> dict[str, int]:
    """Place P2P endpoints near compute anchors without changing DAG edges."""
    topo_levels = _topological_levels(dag)
    base_order = _topological_order(dag)
    dispatch_levels = dict(topo_levels)

    def anchor(uid: str, field: str) -> str | None:
        value = dag.nodes[uid].node_meta.get(field)
        anchor_node = dag.nodes.get(value) if isinstance(value, str) else None
        if anchor_node is not None and anchor_node.node_kind == "COMPUTE":
            return value
        if require_p2p_anchors:
            raise ValueError(
                f"P2P node {uid!r} requires local COMPUTE anchor {field}, "
                f"got {value!r}"
            )
        return None

    # Moving a SEND earlier can only threaten its incoming edges. Process
    # sources forward so every predecessor already has its final level.
    for uid in base_order:
        if dag.nodes[uid].node_kind != "SEND_COMM":
            continue
        producer_uid = anchor(uid, "source_compute_uid")
        if producer_uid is None:
            continue
        predecessor_floor = max(
            (dispatch_levels[pred_uid] + 1 for pred_uid in dag.preds[uid]),
            default=0,
        )
        target = dispatch_levels[producer_uid] + 1
        dispatch_levels[uid] = max(target, predecessor_floor)

    # Moving a RECV later can only threaten its outgoing edges. Process sinks
    # backward so every successor already has its final level.
    for uid in reversed(base_order):
        if dag.nodes[uid].node_kind != "RECV_COMM":
            continue
        consumer_uid = anchor(uid, "destination_compute_uid")
        if consumer_uid is None:
            continue
        successor_ceiling = min(
            (dispatch_levels[succ_uid] - 1 for succ_uid in dag.succs[uid]),
            default=dispatch_levels[consumer_uid] - 1,
        )
        target = dispatch_levels[consumer_uid] - 1
        dispatch_levels[uid] = min(target, successor_ceiling)

    for edge in dag.edges:
        src_level = dispatch_levels[edge.src_uid]
        dst_level = dispatch_levels[edge.dst_uid]
        if src_level >= dst_level:
            raise ValueError(
                "dispatch-level assignment violates DAG edge "
                f"{edge.src_uid}->{edge.dst_uid} ({edge.dep_kind}): "
                f"levels={src_level}->{dst_level}"
            )
    return dispatch_levels


def assign_dispatch_levels(dag: TrainingDAG) -> None:
    """Persist the finalized static issue level on every actor-DAG node."""
    dispatch_levels = _compute_dispatch_levels(
        dag,
        require_p2p_anchors=True,
    )
    for uid, level in dispatch_levels.items():
        dag.nodes[uid].node_meta["dispatch_level"] = int(level)


def _dispatch_order(dag: TrainingDAG) -> list[str]:
    """Return the static executor/simulator issue order for a finalized DAG."""
    stored_levels: dict[str, int] = {}
    for uid, node in dag.nodes.items():
        value = node.node_meta.get("dispatch_level")
        if isinstance(value, bool) or not isinstance(value, int):
            stored_levels = {}
            break
        stored_levels[uid] = value
    dispatch_levels = stored_levels or _compute_dispatch_levels(
        dag,
        require_p2p_anchors=False,
    )
    for edge in dag.edges:
        if dispatch_levels[edge.src_uid] >= dispatch_levels[edge.dst_uid]:
            raise ValueError(
                "persisted dispatch levels violate DAG edge "
                f"{edge.src_uid}->{edge.dst_uid} ({edge.dep_kind}): "
                f"levels={dispatch_levels[edge.src_uid]}->"
                f"{dispatch_levels[edge.dst_uid]}"
            )
    base_order = _topological_order(dag)
    topo_idx = {uid: index for index, uid in enumerate(base_order)}
    ordered = sorted(
        dag.nodes,
        key=lambda uid: (
            dispatch_levels[uid],
            _dispatch_priority(dag, uid),
            topo_idx[uid],
        ),
    )
    issue_index = {uid: index for index, uid in enumerate(ordered)}
    for edge in dag.edges:
        if issue_index[edge.src_uid] >= issue_index[edge.dst_uid]:
            raise ValueError(
                "dispatch order violates DAG edge "
                f"{edge.src_uid}->{edge.dst_uid} ({edge.dep_kind})"
            )
    return ordered


def _resolve_default_stream_order(
    dag: TrainingDAG,
    reachability: _ReachabilityIndex | None = None,
) -> None:
    """Serialize default-stream compute nodes with incremental graph indexes.

    Conflicts are resolved by downstream-count ordering. Topological levels and
    transitive reachability are updated incrementally as chain edges are added.
    """

    def _is_default_compute(uid: str) -> bool:
        node = dag.nodes[uid]
        return node.stream == _DEFAULT_STREAM and node.node_kind == "COMPUTE"

    compute_uids = [uid for uid in dag.nodes if _is_default_compute(uid)]
    if len(compute_uids) < 2:
        return

    if reachability is None:
        reachability = _ReachabilityIndex(dag)

    topo_levels = _topological_levels(dag)

    def _propagate_level_increases(starts: set[str]) -> None:
        pending = set(starts)
        work_heap = [(topo_levels[uid], uid) for uid in pending]
        heapq.heapify(work_heap)
        while work_heap:
            _level, uid = heapq.heappop(work_heap)
            if uid not in pending:
                continue
            pending.remove(uid)
            pred_levels = [
                topo_levels[pred_uid]
                for pred_uid in dag.preds.get(uid, set())
            ]
            required_level = max(pred_levels) + 1 if pred_levels else 0
            if required_level <= topo_levels[uid]:
                continue
            topo_levels[uid] = required_level
            for succ_uid in dag.succs.get(uid, set()):
                if succ_uid in pending:
                    continue
                pending.add(succ_uid)
                heapq.heappush(
                    work_heap,
                    (topo_levels[succ_uid], succ_uid),
                )

    while True:
        by_level: dict[int, list[str]] = {}
        for uid in compute_uids:
            by_level.setdefault(topo_levels[uid], []).append(uid)

        conflict_level = next(
            (
                level
                for level in sorted(by_level)
                if len(by_level[level]) > 1
            ),
            None,
        )
        if conflict_level is None:
            return

        group = by_level[conflict_level]
        ordered = sorted(
            group,
            key=lambda uid: (-reachability.descendant_count(uid), uid),
        )
        changed_dsts: set[str] = set()
        for src_uid, dst_uid in zip(ordered, ordered[1:]):
            before = len(dag.edges)
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=src_uid,
                    dst_uid=dst_uid,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )
            if len(dag.edges) != before:
                reachability.add_edge(src_uid, dst_uid)
                changed_dsts.add(dst_uid)
        _propagate_level_increases(changed_dsts)


def resolve_total_order_per_stream(dag: TrainingDAG) -> None:
    """Materialize one DAG-compatible FIFO order for every physical stream.

    Default-stream compute keeps its established downstream-aware ordering.
    Every stream domain is then chained in a deterministic topological order,
    so the executor and simulator consume the same FIFO without inventing
    dependencies independently.
    """
    reachability = _ReachabilityIndex(dag)
    _resolve_default_stream_order(dag, reachability)
    _materialize_execution_stream_fifo(dag)


def _materialize_execution_stream_fifo(dag: TrainingDAG) -> None:
    """Represent every physical-stream FIFO relation as a temporal DAG edge."""
    reachability = _ReachabilityIndex(dag)
    ordered_by_domain: dict[str, list[str]] = {}
    for uid in _serial_topological_order(dag):
        ordered_by_domain.setdefault(
            execution_stream_domain(dag.nodes[uid]), []
        ).append(uid)

    for domain, uids in sorted(ordered_by_domain.items()):
        for src_uid, dst_uid in zip(uids, uids[1:]):
            if reachability.has_path(dst_uid, src_uid):
                raise ValueError(
                    "physical stream order conflicts with DAG reachability: "
                    f"stream={domain!r}, requested={src_uid}->{dst_uid}"
                )
            if reachability.has_path(src_uid, dst_uid):
                continue
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=src_uid,
                    dst_uid=dst_uid,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )
            reachability.add_edge(src_uid, dst_uid)
