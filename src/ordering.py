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
}
_REDUCTION_COMM_KINDS = {
    "REDUCE_COMM",
    "REDUCE_SCATTER_COMM",
}


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

    def node_priority(uid: str) -> int:
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

    return sorted(
        dag.nodes,
        key=lambda uid: (topo_levels[uid], node_priority(uid), topo_idx[uid]),
    )


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
    """Serialize the default stream, then strictly chain non-default streams.

    The default-stream pass runs first so the per-non-default-stream anchors
    below see the post-serialization topological order of default-stream nodes.

    Non-default stream nodes are ordered by their nearest ready point on the
    default stream.  A post-compute operation with multiple default-stream
    predecessors (for example an accumulated reduce-scatter) is anchored to
    its latest predecessor, while a pure pre-compute operation (for example an
    all-gather) is anchored to its earliest default-stream successor.
    """
    reachability = _ReachabilityIndex(dag)
    _resolve_default_stream_order(dag, reachability)

    topo = _topological_order(dag)
    topo_idx = {uid: i for i, uid in enumerate(topo)}
    topo_levels = _topological_levels(dag)
    default_uids = [
        uid for uid in topo
        if dag.nodes[uid].stream == _DEFAULT_STREAM
    ]
    streams = sorted({
        n.stream for n in dag.nodes.values()
        if n.stream != _DEFAULT_STREAM
    })
    if not default_uids or not streams:
        return

    for stream in streams:
        associated_by_default: dict[str, list[str]] = {uid: [] for uid in default_uids}
        default_preds_by_stream_uid: dict[str, list[str]] = {}
        default_succs_by_stream_uid: dict[str, list[str]] = {}
        for edge in dag.edges:
            src = dag.nodes[edge.src_uid]
            dst = dag.nodes[edge.dst_uid]
            if src.stream == _DEFAULT_STREAM and dst.stream == stream:
                default_preds_by_stream_uid.setdefault(edge.dst_uid, []).append(
                    edge.src_uid
                )
            elif src.stream == stream and dst.stream == _DEFAULT_STREAM:
                default_succs_by_stream_uid.setdefault(edge.src_uid, []).append(
                    edge.dst_uid
                )

        for stream_uid in {
            uid for uid, node in dag.nodes.items()
            if node.stream == stream
        }:
            default_preds = default_preds_by_stream_uid.get(stream_uid, [])
            default_succs = default_succs_by_stream_uid.get(stream_uid, [])
            if default_preds:
                # The stream operation is not ready until all direct default
                # predecessors complete.  Anchoring to the earliest one can
                # incorrectly place an accumulated collective before a later
                # all-gather that feeds one of its remaining contributors.
                anchor_uid = max(
                    default_preds,
                    key=lambda uid: (topo_levels[uid], topo_idx[uid]),
                )
            elif default_succs:
                anchor_uid = min(
                    default_succs,
                    key=lambda uid: (topo_levels[uid], topo_idx[uid]),
                )
            else:
                continue
            associated_by_default.setdefault(anchor_uid, []).append(stream_uid)

        current_uid: str | None = None
        for default_uid in default_uids:
            stream_uids = sorted(
                set(associated_by_default.get(default_uid, [])),
                key=lambda u: (topo_levels[u], topo_idx[u]),
            )
            for next_uid in stream_uids:
                if current_uid is not None and current_uid != next_uid:
                    if reachability.has_path(next_uid, current_uid):
                        raise ValueError(
                            "resolve_total_order_per_stream would create a cycle while ordering "
                            f"stream={stream}: {current_uid} -> {next_uid}"
                        )
                    if not reachability.has_path(current_uid, next_uid):
                        dag.add_edge(
                            TrainingDAGEdge(
                                src_uid=current_uid,
                                dst_uid=next_uid,
                                dep_kind="temporal",
                                tensor_name=None,
                            )
                        )
                        reachability.add_edge(current_uid, next_uid)
                current_uid = next_uid
