from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    _has_path,
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


def _resolve_default_stream_order(dag: TrainingDAG) -> None:
    """Create a total ordering over default-stream COMPUTE nodes.

    Whenever multiple default-stream compute nodes share a topological level,
    chain them with temporal edges in descending order of downstream
    dependencies (more downstream nodes -> earlier in the chain). Downstream
    count is the size of each compute node's transitive successor set, which in
    a well-formed training DAG terminates at UPD.

    The new edges shift topological levels, so after each chain insertion we
    recompute levels and rescan from the earliest level. The pass terminates
    when every default-stream compute node sits at a unique level.
    """
    def _is_default_compute(uid: str) -> bool:
        node = dag.nodes[uid]
        return node.stream == _DEFAULT_STREAM and node.node_kind == "COMPUTE"

    compute_uids = [uid for uid in dag.nodes if _is_default_compute(uid)]
    if len(compute_uids) < 2:
        return

    def _downstream_count(uid: str) -> int:
        seen: set[str] = set()
        stack = list(dag.succs.get(uid, set()))
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            stack.extend(dag.succs.get(v, set()))
        return len(seen)

    while True:
        topo_levels = _topological_levels(dag)
        by_level: dict[int, list[str]] = {}
        for uid in compute_uids:
            by_level.setdefault(topo_levels[uid], []).append(uid)

        conflict_level: int | None = None
        for level in sorted(by_level):
            if len(by_level[level]) > 1:
                conflict_level = level
                break
        if conflict_level is None:
            return

        # Same-level nodes have no path between them, otherwise their levels
        # would differ. Order by downstream count desc; break ties by uid.
        group = by_level[conflict_level]
        ordered = sorted(group, key=lambda u: (-_downstream_count(u), u))
        for src, dst in zip(ordered, ordered[1:]):
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=src,
                    dst_uid=dst,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )


def resolve_total_order_per_stream(dag: TrainingDAG) -> None:
    """Serialize the default stream, then strictly chain non-default streams.

    The default-stream pass runs first so the per-non-default-stream anchors
    below see the post-serialization topological order of default-stream nodes.

    Non-default stream nodes are ordered by the topological order of the
    default-stream nodes they directly depend on or are depended on by.
    """
    _resolve_default_stream_order(dag)

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
        default_anchors_by_stream_uid: dict[str, list[str]] = {}
        for edge in dag.edges:
            src = dag.nodes[edge.src_uid]
            dst = dag.nodes[edge.dst_uid]
            if src.stream == _DEFAULT_STREAM and dst.stream == stream:
                default_anchors_by_stream_uid.setdefault(edge.dst_uid, []).append(edge.src_uid)
            elif src.stream == stream and dst.stream == _DEFAULT_STREAM:
                default_anchors_by_stream_uid.setdefault(edge.src_uid, []).append(edge.dst_uid)

        for stream_uid in {
            uid for uid, node in dag.nodes.items()
            if node.stream == stream
        }:
            default_anchors = default_anchors_by_stream_uid.get(stream_uid, [])
            if not default_anchors:
                continue
            anchor_uid = min(default_anchors, key=lambda uid: (topo_levels[uid], topo_idx[uid]))
            associated_by_default.setdefault(anchor_uid, []).append(stream_uid)

        current_uid: str | None = None
        for default_uid in default_uids:
            stream_uids = sorted(
                set(associated_by_default.get(default_uid, [])),
                key=lambda u: (topo_levels[u], topo_idx[u]),
            )
            for next_uid in stream_uids:
                if current_uid is not None and current_uid != next_uid:
                    if _has_path(dag, next_uid, current_uid):
                        raise ValueError(
                            "resolve_total_order_per_stream would create a cycle while ordering "
                            f"stream={stream}: {current_uid} -> {next_uid}"
                        )
                    if not _has_path(dag, current_uid, next_uid):
                        dag.add_edge(
                            TrainingDAGEdge(
                                src_uid=current_uid,
                                dst_uid=next_uid,
                                dep_kind="temporal",
                                tensor_name=None,
                            )
                        )
                current_uid = next_uid
