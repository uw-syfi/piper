from dataclasses import dataclass, field
from typing import Any

import torch.fx as fx

from .fx import AnnotationSegment

def _collect_triton_constant_args(gm: fx.GraphModule) -> dict[int, Any]:
    from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
    out: dict[int, Any] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        target_str = getattr(node.target, "__name__", "") or str(node.target)
        if "triton_kernel_wrapper" not in target_str:
            continue
        idx = node.kwargs.get("constant_args_idx")
        if idx is not None and idx in kernel_side_table.constant_args:
            out[int(idx)] = kernel_side_table.constant_args[idx]
    return out


@dataclass(frozen=True)
class TrainingDAGEdge:
    src_uid: str
    dst_uid: str
    dep_kind: str  # "data" | "temporal"
    tensor_name: str | None = None


@dataclass
class TrainingDAGNode:
    uid: str
    node_kind: str  # "COMPUTE" | "SEND_COMM" | "RECV_COMM" | "REDUCE_COMM" | "ALL_GATHER_COMM" | "REDUCE_SCATTER_COMM" | "A2A_COMM" | "ORDER_DUMMY"
    compute_subkind: str | None  # "FWD" | "BWD" | "BWD_I" | "BWD_W" when node_kind == "COMPUTE"
    tag: dict[str, int | None]
    device: list[int] | None
    stream: str
    node_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDAG:
    nodes: dict[str, TrainingDAGNode] = field(default_factory=dict)
    edges: list[TrainingDAGEdge] = field(default_factory=list)
    succs: dict[str, set[str]] = field(default_factory=dict)
    preds: dict[str, set[str]] = field(default_factory=dict)
    edge_keys: set[tuple[str, str, str, str | None]] = field(default_factory=set)

    def add_node(self, node: TrainingDAGNode) -> None:
        if node.uid in self.nodes:
            raise ValueError(f"Duplicate node uid: {node.uid}")
        self.nodes[node.uid] = node
        self.succs[node.uid] = set()
        self.preds[node.uid] = set()

    def add_edge(self, edge: TrainingDAGEdge) -> None:
        if edge.src_uid not in self.nodes or edge.dst_uid not in self.nodes:
            raise ValueError(f"Edge references unknown node: {edge}")
        edge_key = (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
        if edge_key in self.edge_keys:
            return
        self.edge_keys.add(edge_key)
        self.edges.append(edge)
        self.succs[edge.src_uid].add(edge.dst_uid)
        self.preds[edge.dst_uid].add(edge.src_uid)



def _flatten_output_nodes(out_arg: Any) -> list[fx.Node]:
    out_nodes: list[fx.Node] = []

    def _walk(v: Any) -> None:
        if isinstance(v, fx.Node):
            out_nodes.append(v)
        elif isinstance(v, (tuple, list)):
            for x in v:
                _walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)

    _walk(out_arg)
    return out_nodes



def _node_io_names(seg_gm: fx.GraphModule) -> tuple[list[str], list[str]]:
    inputs = [n.name for n in seg_gm.graph.nodes if n.op == "placeholder"]
    out_node = next((n for n in seg_gm.graph.nodes if n.op == "output"), None)
    if out_node is None:
        return inputs, []
    outputs = [n.name for n in _flatten_output_nodes(out_node.args[0])]
    return inputs, outputs

def _with_pass_tag(tag: dict[str, Any], pass_value: str | None) -> dict[str, Any]:
    t = dict(tag)
    t["PASS"] = pass_value
    return t

def build_training_dag(
    annotation_segments: list[AnnotationSegment],
) -> TrainingDAG:
    """Build a TrainingDAG where each node is one annotated GraphModule segment."""
    dag = TrainingDAG()

    global_prev_uid: str | None = None

    for segment in annotation_segments:
        uid = f"s{segment.stage_id}.seg{segment.segment_id}"
        tag = _with_pass_tag(segment.tag, "F")
        in_names, out_names = _node_io_names(segment.gm)
        node = TrainingDAGNode(
            uid=uid,
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag=tag,
            device=None,
            stream="default_stream",
            node_meta={
                "bucket_key": uid,
                "stage_id": segment.stage_id,
                "segment_id": segment.segment_id,
                "gm": segment.gm,
                "input_idxs": list(segment.input_idxs),
                "param_idxs": list(segment.param_idxs),
                "graphargs": list(segment.graphargs),
                "input_names": in_names,
                "output_names": out_names,
                "a2a_boundary_after": segment.a2a_boundary_after,
                "triton_constant_args": _collect_triton_constant_args(segment.gm),
            },
        )
        dag.add_node(node)

        # Explicit activation-flow ordering in annotation segment order.
        if global_prev_uid is not None:
            dag.add_edge(TrainingDAGEdge(src_uid=global_prev_uid, dst_uid=uid, dep_kind="data"))
        global_prev_uid = uid

    # Build backward-pass nodes and dependencies by reversing forward data edges.
    fwd_compute_uids = [
        uid for uid, node in dag.nodes.items()
        if node.node_kind == "COMPUTE" and node.compute_subkind == "FWD"
    ]
    fwd_uid_to_bwd_uid: dict[str, str] = {}
    for fwd_uid in fwd_compute_uids:
        fwd_node = dag.nodes[fwd_uid]
        bwd_uid = f"{fwd_uid}.bwd"
        fwd_uid_to_bwd_uid[fwd_uid] = bwd_uid
        dag.add_node(
            TrainingDAGNode(
                uid=bwd_uid,
                node_kind="COMPUTE",
                compute_subkind="BWD",
                tag=_with_pass_tag(fwd_node.tag, "B"),
                device=fwd_node.device,
                stream="default_stream",
                node_meta={
                    "fwd_uid": fwd_uid,
                    "bucket_key": fwd_node.node_meta.get("bucket_key", fwd_uid),
                    "compute_loss": False,
                },
            )
        )

    fwd_edges = [
        e for e in dag.edges
        if e.dep_kind == "data"
        and dag.nodes[e.src_uid].node_kind == "COMPUTE"
        and dag.nodes[e.dst_uid].node_kind == "COMPUTE"
        and dag.nodes[e.src_uid].compute_subkind == "FWD"
        and dag.nodes[e.dst_uid].compute_subkind == "FWD"
    ]

    # Reverse every forward dependency for the backward pass:
    # FWD:  u -> v    ==>    BWD: v' -> u'
    for e in fwd_edges:
        bwd_src = fwd_uid_to_bwd_uid[e.dst_uid]
        bwd_dst = fwd_uid_to_bwd_uid[e.src_uid]
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=bwd_src,
                dst_uid=bwd_dst,
                dep_kind="data",
                tensor_name=e.tensor_name,
            )
        )

    # Bridge from forward phase into backward phase:
    # last_fwd -> first_bwd (where first_bwd is the BWD node paired with last_fwd).
    if fwd_compute_uids:
        def _fwd_sort_key(uid: str) -> tuple[int, int]:
            n = dag.nodes[uid]
            return (
                int(n.node_meta.get("stage_id", -1)),
                int(n.node_meta.get("segment_id", -1)),
            )

        last_fwd_uid = max(fwd_compute_uids, key=_fwd_sort_key)
        first_bwd_uid = fwd_uid_to_bwd_uid[last_fwd_uid]
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=last_fwd_uid,
                dst_uid=first_bwd_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )

        # Mark loss computation point: first BWD node (earliest in BWD execution).
        if first_bwd_uid in dag.nodes:
            dag.nodes[first_bwd_uid].node_meta["compute_loss"] = True

    # Add explicit UPD node with temporal dependency from every BWD compute node.
    # UPD participates in the compute temporal chain and is not scheduled from
    # data edges.
    upd_uid = "upd.0"
    dag.add_node(
        TrainingDAGNode(
            uid=upd_uid,
            node_kind="UPD",
            compute_subkind=None,
            tag={"PASS": None},
            device=None,
            stream="default_stream",
            node_meta={},
        )
    )
    for uid, node in list(dag.nodes.items()):
        if node.node_kind == "COMPUTE" and node.compute_subkind == "BWD":
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=uid,
                    dst_uid=upd_uid,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )

    return dag

def _iter_data_edges(dag: TrainingDAG) -> list[TrainingDAGEdge]:
    return [e for e in dag.edges if e.dep_kind == "data"]


def _remove_edge(dag: TrainingDAG, edge: TrainingDAGEdge) -> None:
    edge_key = (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
    if edge_key not in dag.edge_keys:
        return
    dag.edge_keys.remove(edge_key)
    dag.edges = [
        e for e in dag.edges
        if not (
            e.src_uid == edge.src_uid
            and e.dst_uid == edge.dst_uid
            and e.dep_kind == edge.dep_kind
            and e.tensor_name == edge.tensor_name
        )
    ]
    if edge.src_uid in dag.succs:
        dag.succs[edge.src_uid].discard(edge.dst_uid)
    if edge.dst_uid in dag.preds:
        dag.preds[edge.dst_uid].discard(edge.src_uid)

def _remove_node_and_incident_edges(dag: TrainingDAG, uid: str) -> None:
    incident = [e for e in dag.edges if e.src_uid == uid or e.dst_uid == uid]
    for e in incident:
        _remove_edge(dag, e)
    dag.nodes.pop(uid, None)
    dag.succs.pop(uid, None)
    dag.preds.pop(uid, None)
    for s in dag.succs.values():
        s.discard(uid)
    for p in dag.preds.values():
        p.discard(uid)

def _topological_order(dag: TrainingDAG) -> list[str]:
    in_deg: dict[str, int] = {uid: 0 for uid in dag.nodes}
    out_edges_by_src: dict[str, list[TrainingDAGEdge]] = {uid: [] for uid in dag.nodes}
    for e in dag.edges:
        if e.dst_uid in in_deg:
            in_deg[e.dst_uid] += 1
        if e.src_uid in out_edges_by_src:
            out_edges_by_src[e.src_uid].append(e)
    q = sorted(uid for uid, d in in_deg.items() if d == 0)
    out: list[str] = []
    while q:
        cur_level = q
        q = []
        for u in cur_level:
            out.append(u)
            # Decrement per outgoing edge (not per unique successor), because we
            # can have parallel edges between the same pair (e.g., data + temporal).
            for e in out_edges_by_src.get(u, []):
                v = e.dst_uid
                if v not in in_deg:
                    continue
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    q.append(v)
        q.sort()
    if len(out) != len(dag.nodes):
        remaining = [uid for uid, d in in_deg.items() if d > 0]
        raise ValueError(
            "TrainingDAG topological sort failed: unresolved incoming edges remain; "
            f"possible real cycle or malformed parallel-edge bookkeeping. Remaining nodes: {remaining[:8]}"
        )
    return out


def _topological_levels(dag: TrainingDAG) -> dict[str, int]:
    """Return each node's topological level.

    Source nodes are level 0. Every other node is one level after its latest
    predecessor, so independent successors can share the same level.
    """
    topo = _topological_order(dag)
    levels: dict[str, int] = {}
    for uid in topo:
        pred_levels = [
            levels[pred_uid]
            for pred_uid in dag.preds.get(uid, set())
            if pred_uid in levels
        ]
        levels[uid] = (max(pred_levels) + 1) if pred_levels else 0
    return levels

def _has_path(dag: TrainingDAG, src_uid: str, dst_uid: str) -> bool:
    seen: set[str] = set()
    stack = [src_uid]
    while stack:
        uid = stack.pop()
        if uid == dst_uid:
            return True
        if uid in seen:
            continue
        seen.add(uid)
        stack.extend(dag.succs.get(uid, set()))
    return False
