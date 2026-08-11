from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.fx as fx

from .fx import AnnotationSegment

if TYPE_CHECKING:
    from .profiling import NodeProfile

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
    node_kind: str  # "COMPUTE" | "SEND_COMM" | "RECV_COMM" | "REDUCE_COMM" | "ALL_GATHER_COMM" | "REDUCE_SCATTER_COMM" | "A2A_COMM" | "M_TO_N_COMM" | "N_TO_M_COMM" | "ORDER_DUMMY"
    compute_subkind: str | None  # "FWD" | "BWD" | "BWD_I" | "BWD_W" when node_kind == "COMPUTE"
    tag: dict[str, int | None]
    device: list[int] | None
    stream: str
    node_meta: dict[str, Any] = field(default_factory=dict)
    # Rank-specific named timing datasets. Keep these out of node_meta, which
    # also contains live FX/runtime objects and is not a durable interchange.
    profile_times_by_mode: dict[str, dict[int, NodeProfile]] = field(default_factory=dict)


@dataclass
class TrainingDAG:
    nodes: dict[str, TrainingDAGNode] = field(default_factory=dict)
    edges: list[TrainingDAGEdge] = field(default_factory=list)
    succs: dict[str, set[str]] = field(default_factory=dict)
    preds: dict[str, set[str]] = field(default_factory=dict)
    edge_keys: set[tuple[str, str, str, str | None]] = field(default_factory=set)
    # Edge indexes retain the canonical ``edges`` list as the public/persisted
    # representation while making incident-edge queries proportional to node
    # degree.  ``compare=False`` keeps them as derived storage rather than part
    # of TrainingDAG's semantic identity.
    _out_edges_by_uid: dict[
        str,
        dict[tuple[str, str, str, str | None], TrainingDAGEdge],
    ] = field(default_factory=dict, repr=False, compare=False)
    _in_edges_by_uid: dict[
        str,
        dict[tuple[str, str, str, str | None], TrainingDAGEdge],
    ] = field(default_factory=dict, repr=False, compare=False)
    _edge_pair_counts: dict[tuple[str, str], int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _next_comm_index_by_kind: dict[str, int] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _tag_uids_by_key_value: dict[tuple[str, Any], set[str]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _tag_uids_non_none_by_key: dict[str, set[str]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _node_order: dict[str, int] = field(default_factory=dict, repr=False, compare=False)
    _next_node_order: int = field(default=0, repr=False, compare=False)
    # Gradient-buffer initialization happens before the first scheduled node,
    # so it has no natural TrainingDAGNode to carry its profile.
    prologue_profile_times_by_mode: dict[str, dict[int, NodeProfile]] = field(
        default_factory=dict
    )

    def add_node(self, node: TrainingDAGNode) -> None:
        if node.uid in self.nodes:
            raise ValueError(f"Duplicate node uid: {node.uid}")
        self.nodes[node.uid] = node
        self.succs[node.uid] = set()
        self.preds[node.uid] = set()
        self._out_edges_by_uid[node.uid] = {}
        self._in_edges_by_uid[node.uid] = {}
        self._node_order[node.uid] = self._next_node_order
        self._next_node_order += 1
        self._index_node_tag(node)

    def add_edge(self, edge: TrainingDAGEdge) -> None:
        if edge.src_uid not in self.nodes or edge.dst_uid not in self.nodes:
            raise ValueError(f"Edge references unknown node: {edge}")
        edge_key = (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
        if edge_key in self.edge_keys:
            return
        self.edge_keys.add(edge_key)
        self.edges.append(edge)
        self._out_edges_by_uid.setdefault(edge.src_uid, {})[edge_key] = edge
        self._in_edges_by_uid.setdefault(edge.dst_uid, {})[edge_key] = edge
        pair_key = (edge.src_uid, edge.dst_uid)
        self._edge_pair_counts[pair_key] = self._edge_pair_counts.get(pair_key, 0) + 1
        self.succs[edge.src_uid].add(edge.dst_uid)
        self.preds[edge.dst_uid].add(edge.src_uid)

    def incoming_edges(
        self,
        uid: str,
        *,
        dep_kind: str | None = None,
    ) -> list[TrainingDAGEdge]:
        edges = self._in_edges_by_uid.get(uid, {}).values()
        if dep_kind is None:
            return list(edges)
        return [edge for edge in edges if edge.dep_kind == dep_kind]

    def outgoing_edges(
        self,
        uid: str,
        *,
        dep_kind: str | None = None,
    ) -> list[TrainingDAGEdge]:
        edges = self._out_edges_by_uid.get(uid, {}).values()
        if dep_kind is None:
            return list(edges)
        return [edge for edge in edges if edge.dep_kind == dep_kind]

    def _index_node_tag(self, node: TrainingDAGNode) -> None:
        for key, value in node.tag.items():
            self._tag_uids_by_key_value.setdefault((key, value), set()).add(node.uid)
            if value is not None:
                self._tag_uids_non_none_by_key.setdefault(key, set()).add(node.uid)

    def _unindex_node_tag(self, node: TrainingDAGNode) -> None:
        for key, value in node.tag.items():
            exact = self._tag_uids_by_key_value.get((key, value))
            if exact is not None:
                exact.discard(node.uid)
                if not exact:
                    self._tag_uids_by_key_value.pop((key, value), None)
            if value is not None:
                non_none = self._tag_uids_non_none_by_key.get(key)
                if non_none is not None:
                    non_none.discard(node.uid)
                    if not non_none:
                        self._tag_uids_non_none_by_key.pop(key, None)

    def update_node_tag(self, uid: str, tag: dict[str, Any]) -> None:
        node = self.nodes[uid]
        self._unindex_node_tag(node)
        node.tag = tag
        self._index_node_tag(node)

    def matching_uids(
        self,
        tag_filter: dict[str, Any],
        *,
        wildcard_value: Any,
        absent_value: Any | None = None,
        exclude_node_kinds: set[str] | None = None,
    ) -> list[str]:
        candidates: set[str] | None = None
        for key, value in tag_filter.items():
            if absent_value is not None and value == absent_value:
                present = self._tag_uids_non_none_by_key.get(key, set()) | (
                    self._tag_uids_by_key_value.get((key, None), set())
                )
                matching = set(self.nodes) - present
            elif value == wildcard_value:
                matching = self._tag_uids_non_none_by_key.get(key, set())
            else:
                matching = self._tag_uids_by_key_value.get((key, value), set())
            candidates = set(matching) if candidates is None else candidates & matching
            if not candidates:
                return []
        if candidates is None:
            candidates = set(self.nodes)
        if exclude_node_kinds:
            candidates = {
                uid
                for uid in candidates
                if self.nodes[uid].node_kind not in exclude_node_kinds
            }
        return sorted(candidates, key=self._node_order.__getitem__)

    def _rebuild_tag_index(self) -> None:
        self._tag_uids_by_key_value = {}
        self._tag_uids_non_none_by_key = {}
        self._node_order = {}
        for order, node in enumerate(self.nodes.values()):
            self._node_order[node.uid] = order
            self._index_node_tag(node)
        self._next_node_order = len(self.nodes)

    def _rebuild_edge_indexes(self) -> None:
        self.edge_keys = set()
        self.succs = {uid: set() for uid in self.nodes}
        self.preds = {uid: set() for uid in self.nodes}
        self._out_edges_by_uid = {uid: {} for uid in self.nodes}
        self._in_edges_by_uid = {uid: {} for uid in self.nodes}
        self._edge_pair_counts = {}
        if not hasattr(self, "_next_comm_index_by_kind"):
            self._next_comm_index_by_kind = {}
        deduplicated_edges: list[TrainingDAGEdge] = []
        for edge in self.edges:
            edge_key = (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
            if edge_key in self.edge_keys:
                continue
            if edge.src_uid not in self.nodes or edge.dst_uid not in self.nodes:
                raise ValueError(f"Edge references unknown node: {edge}")
            self.edge_keys.add(edge_key)
            deduplicated_edges.append(edge)
            self._out_edges_by_uid[edge.src_uid][edge_key] = edge
            self._in_edges_by_uid[edge.dst_uid][edge_key] = edge
            pair_key = (edge.src_uid, edge.dst_uid)
            self._edge_pair_counts[pair_key] = self._edge_pair_counts.get(pair_key, 0) + 1
            self.succs[edge.src_uid].add(edge.dst_uid)
            self.preds[edge.dst_uid].add(edge.src_uid)
        self.edges = deduplicated_edges

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Older persistent DAGs predate the derived edge indexes.  Rebuilding
        # here keeps those cache files loadable and also repairs adjacency for
        # parallel data/temporal edges.
        self.__dict__.update(state)
        self._rebuild_edge_indexes()
        self._rebuild_tag_index()



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



def _node_io_names(
    seg_gm: fx.GraphModule,
) -> tuple[list[str], list[str], list[int], list[int]]:
    inputs = [n.name for n in seg_gm.graph.nodes if n.op == "placeholder"]
    out_node = next((n for n in seg_gm.graph.nodes if n.op == "output"), None)
    if out_node is None:
        return inputs, [], [], []
    output_nodes = _flatten_output_nodes(out_node.args[0])
    outputs = [n.name for n in output_nodes]
    tensor_output_indices: list[int] = []
    differentiable_output_indices: list[int] = []
    for index, node in enumerate(output_nodes):
        example = node.meta.get("example_value", node.meta.get("val"))
        if not isinstance(example, torch.Tensor):
            continue
        tensor_output_indices.append(index)
        if bool(getattr(example, "requires_grad", False)):
            differentiable_output_indices.append(index)
    return (
        inputs,
        outputs,
        tensor_output_indices,
        differentiable_output_indices,
    )

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
        (
            in_names,
            out_names,
            tensor_output_indices,
            differentiable_output_indices,
        ) = _node_io_names(segment.gm)
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
                "tensor_output_indices": tensor_output_indices,
                "differentiable_output_indices": differentiable_output_indices,
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


def _edge_key(edge: TrainingDAGEdge) -> tuple[str, str, str, str | None]:
    return (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)


def _remove_edges(dag: TrainingDAG, edges: list[TrainingDAGEdge]) -> None:
    """Remove multiple edges with one canonical-edge-list compaction."""

    remove_keys = {
        _edge_key(edge)
        for edge in edges
        if _edge_key(edge) in dag.edge_keys
    }
    if not remove_keys:
        return
    removed = [edge for edge in dag.edges if _edge_key(edge) in remove_keys]
    dag.edges = [edge for edge in dag.edges if _edge_key(edge) not in remove_keys]
    for edge in removed:
        edge_key = _edge_key(edge)
        dag.edge_keys.remove(edge_key)
        dag._out_edges_by_uid.get(edge.src_uid, {}).pop(edge_key, None)
        dag._in_edges_by_uid.get(edge.dst_uid, {}).pop(edge_key, None)
        pair_key = (edge.src_uid, edge.dst_uid)
        remaining = dag._edge_pair_counts.get(pair_key, 0) - 1
        if remaining > 0:
            dag._edge_pair_counts[pair_key] = remaining
            continue
        dag._edge_pair_counts.pop(pair_key, None)
        if edge.src_uid in dag.succs:
            dag.succs[edge.src_uid].discard(edge.dst_uid)
        if edge.dst_uid in dag.preds:
            dag.preds[edge.dst_uid].discard(edge.src_uid)


def _remove_edge(dag: TrainingDAG, edge: TrainingDAGEdge) -> None:
    _remove_edges(dag, [edge])

def _remove_node_and_incident_edges(dag: TrainingDAG, uid: str) -> None:
    incident = dag.incoming_edges(uid) + dag.outgoing_edges(uid)
    _remove_edges(dag, incident)
    node = dag.nodes.pop(uid, None)
    if node is not None:
        dag._unindex_node_tag(node)
    dag._node_order.pop(uid, None)
    dag.succs.pop(uid, None)
    dag.preds.pop(uid, None)
    dag._out_edges_by_uid.pop(uid, None)
    dag._in_edges_by_uid.pop(uid, None)


def _remove_nodes_and_incident_edges(dag: TrainingDAG, uids: set[str]) -> None:
    """Remove a node set and every incident edge in one bulk operation."""

    existing = uids & dag.nodes.keys()
    incident: list[TrainingDAGEdge] = []
    for uid in existing:
        incident.extend(dag.incoming_edges(uid))
        incident.extend(dag.outgoing_edges(uid))
    _remove_edges(dag, incident)
    for uid in existing:
        node = dag.nodes.pop(uid, None)
        if node is not None:
            dag._unindex_node_tag(node)
        dag._node_order.pop(uid, None)
        dag.succs.pop(uid, None)
        dag.preds.pop(uid, None)
        dag._out_edges_by_uid.pop(uid, None)
        dag._in_edges_by_uid.pop(uid, None)

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


class _ReachabilityIndex:
    """Bitset reachability for a fixed node set with incremental edge adds."""

    def __init__(self, dag: TrainingDAG) -> None:
        topo = _topological_order(dag)
        self._dag = dag
        self._bit_by_uid = {
            uid: 1 << index for index, uid in enumerate(topo)
        }
        self._reachable_by_uid: dict[str, int] = {}
        for uid in reversed(topo):
            reachable = self._bit_by_uid[uid]
            for succ_uid in dag.succs.get(uid, set()):
                reachable |= self._reachable_by_uid[succ_uid]
            self._reachable_by_uid[uid] = reachable

    def has_path(self, src_uid: str, dst_uid: str) -> bool:
        return bool(
            self._reachable_by_uid[src_uid]
            & self._bit_by_uid[dst_uid]
        )

    def descendant_count(self, uid: str) -> int:
        """Return the number of nodes reachable from ``uid``, excluding it."""

        return self._reachable_by_uid[uid].bit_count() - 1

    def add_edge(self, src_uid: str, dst_uid: str) -> None:
        """Update closure after the caller adds ``src_uid -> dst_uid``."""

        new_descendants = self._reachable_by_uid[dst_uid]
        pending = [src_uid]
        while pending:
            uid = pending.pop()
            previous = self._reachable_by_uid[uid]
            updated = previous | new_descendants
            if updated == previous:
                continue
            self._reachable_by_uid[uid] = updated
            pending.extend(self._dag.preds.get(uid, set()))
