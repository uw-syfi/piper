from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch.fx as fx

from .bucket import bucket_stage
from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    TrainingDAGNode,
    _collect_triton_side_table,
    _has_path,
    _iter_data_edges,
    _node_io_names,
    _remove_edge,
    _remove_edges,
    _remove_node_and_incident_edges,
    _remove_nodes_and_incident_edges,
    _topological_order,
    _with_pass_tag,
)
from .state import LOG_LEVEL, create_logger
from .schedule import validate_shard_device_order

logger = create_logger("directives", LOG_LEVEL)
_ANY_TAG_INDEX = "__ANY_TAG_INDEX__"
_ABSENT_TAG_INDEX = "__ABSENT_TAG_INDEX__"
_FORWARD_PASS = "F"
_FUSED_BWD_PASS = "B"
_BWD_INPUT_PASS = "BI"
_BWD_WEIGHT_PASS = "BW"
_DEFAULT_STREAM = "default_stream"
_VALID_ORDER_PASSES = frozenset({
    _FORWARD_PASS,
    _FUSED_BWD_PASS,
    _BWD_INPUT_PASS,
    _BWD_WEIGHT_PASS,
})

def _is_backward_activation_subkind(subkind: str | None) -> bool:
    return subkind in ("BWD", "BWD_I")


def _is_backward_weight_subkind(subkind: str | None) -> bool:
    return subkind in ("BWD", "BWD_W")


def _is_backward_compute_subkind(subkind: str | None) -> bool:
    return subkind in ("BWD", "BWD_I", "BWD_W")

def _match_filter(tag: dict[str, int | None], flt: dict[str, Any]) -> bool:
    for k, v in flt.items():
        if v == _ABSENT_TAG_INDEX:
            if k in tag:
                return False
            continue
        if k not in tag:
            return False
        if v == _ANY_TAG_INDEX:
            # "*" wildcard matches any concrete index for this tag, but not None.
            if tag[k] is None:
                return False
        elif tag[k] != v:
            return False
    return True


def _matching_uids(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    *,
    exclude_node_kinds: set[str] | None = None,
) -> list[str]:
    if not filters:
        return []
    if len(filters) == 1:
        return dag.matching_uids(
            filters[0],
            wildcard_value=_ANY_TAG_INDEX,
            absent_value=_ABSENT_TAG_INDEX,
            exclude_node_kinds=exclude_node_kinds,
        )
    matched: set[str] = set()
    for tag_filter in filters:
        matched.update(
            dag.matching_uids(
                tag_filter,
                wildcard_value=_ANY_TAG_INDEX,
                absent_value=_ABSENT_TAG_INDEX,
                exclude_node_kinds=exclude_node_kinds,
            )
        )
    return sorted(matched, key=dag._node_order.__getitem__)


def _matched_compute_regions(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
) -> set[str]:
    """Return logical model regions selected by a communication directive."""
    regions: set[str] = set()
    for uid in _matching_uids(dag, filters):
        node = dag.nodes[uid]
        if node.node_kind != "COMPUTE":
            continue
        if _is_backward_compute_subkind(node.compute_subkind):
            fwd_uid = node.node_meta.get("fwd_uid")
            if isinstance(fwd_uid, str) and fwd_uid in dag.nodes:
                regions.add(fwd_uid)
                continue
        regions.add(uid)
    return regions


def _validate_replicate_shard_disjoint(
    dag: TrainingDAG,
    directives: list[Any],
) -> None:
    """Reject logical model regions selected by both replicate and shard."""
    selections: dict[str, list[tuple[int, dict[str, Any], set[str]]]] = {
        "replicate": [],
        "shard": [],
    }
    for directive_idx, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") not in selections:
            continue
        op, filters, *_rest = _normalize_filter_devices_directive(raw)
        selections[op].append(
            (
                directive_idx,
                filters[0],
                _matched_compute_regions(dag, filters),
            )
        )

    for replicate_idx, replicate_filter, replicate_regions in selections[
        "replicate"
    ]:
        for shard_idx, shard_filter, shard_regions in selections["shard"]:
            overlap = sorted(replicate_regions & shard_regions)
            if overlap:
                raise ValueError(
                    "replicate and shard directives must select disjoint model "
                    f"regions: replicate directive[{replicate_idx}] filter="
                    f"{replicate_filter} and shard directive[{shard_idx}] "
                    f"filter={shard_filter} both match regions={overlap}"
                )


def _apply_split_backward_stencil(
    dag: TrainingDAG,
    split_base_keys: set[tuple[tuple[str, Any], ...]],
) -> None:
    if not split_base_keys:
        return
    split_filters = [dict(key) for key in split_base_keys]
    for node in list(dag.nodes.values()):
        if (
            node.node_kind != "COMPUTE"
            or node.compute_subkind != "BWD"
            or not any(_match_filter(node.tag, flt) for flt in split_filters)
        ):
            continue
        _split_fused_bwd_node(dag, node.uid)


def _split_fused_bwd_node(dag: TrainingDAG, bwd_uid: str) -> None:
    bwd_i = dag.nodes[bwd_uid]
    if bwd_i.node_kind != "COMPUTE" or bwd_i.compute_subkind != "BWD":
        raise ValueError(f"expected fused BWD compute node, got {bwd_uid}: {bwd_i}")
    bwd_w_uid = f"{bwd_uid}.bw"
    if bwd_w_uid in dag.nodes:
        raise ValueError(f"duplicate BWD_W uid while splitting {bwd_uid}: {bwd_w_uid}")

    bwd_i.compute_subkind = "BWD_I"
    dag.update_node_tag(
        bwd_uid,
        _with_pass_tag(bwd_i.tag, _BWD_INPUT_PASS),
    )

    bwd_w_meta = dict(bwd_i.node_meta)
    bwd_w_meta["compute_loss"] = False
    dag.add_node(
        TrainingDAGNode(
            uid=bwd_w_uid,
            node_kind="COMPUTE",
            compute_subkind="BWD_W",
            tag=_with_pass_tag(bwd_i.tag, _BWD_WEIGHT_PASS),
            device=list(bwd_i.device) if bwd_i.device is not None else None,
            stream=bwd_i.stream,
            node_meta=bwd_w_meta,
        )
    )

    for edge in [
        e for e in dag.outgoing_edges(bwd_uid, dep_kind="temporal")
        if dag.nodes[e.dst_uid].node_kind == "UPD"
    ]:
        _remove_edge(dag, edge)
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=bwd_w_uid,
                dst_uid=edge.dst_uid,
                dep_kind="temporal",
                tensor_name=edge.tensor_name,
            )
        )

    dag.add_edge(
        TrainingDAGEdge(
            src_uid=bwd_uid,
            dst_uid=bwd_w_uid,
            dep_kind="data",
            tensor_name=None,
        )
    )


def _normalize_filter_devices_directive(
    directive: Any
) -> tuple[str, list[dict[str, Any]], list[int], str | None, str | None, str | None, bool, bool, int | None]:
    if isinstance(directive, dict):
        op = directive.get("op")
        if op not in ("place", "replicate", "shard"):
            raise ValueError(f"Unsupported directive op: {op}")
        if "filter" not in directive:
            raise ValueError(f"{op} directive requires current API field 'filter': {directive}")
        if "filters" in directive:
            raise ValueError(f"{op} directive does not accept 'filters': {directive}")
        filter_spec = directive.get("filter")
        devices = directive.get("devices", directive.get("device"))
        stream = directive.get("stream")
        gather_stream = directive.get("gather_stream")
        reduce_stream = directive.get("reduce_stream")
        if op == "replicate":
            if stream is not None:
                raise ValueError(
                    f"replicate directive does not accept 'stream'; use 'gather_stream' and/or 'reduce_stream': {directive}"
                )
        else:
            if gather_stream is not None or reduce_stream is not None:
                raise ValueError(
                    f"{op} directive does not accept 'gather_stream'/'reduce_stream'; use 'stream': {directive}"
                )
        shard_params = bool(directive.get("shard_params", False))
        shard_grads = bool(directive.get("shard_grads", False))
        bucket_size_raw = directive.get("bucket_size", None)
        bucket_size = None if bucket_size_raw is None else int(bucket_size_raw)
        if not isinstance(devices, list) or not devices:
            raise ValueError(f"place directive requires non-empty devices list: {directive}")
        n = _normalize_filter_spec(filter_spec, directive)
        return (
            op,
            [n],
            [int(d) for d in devices],
            (None if stream is None else str(stream)),
            (None if gather_stream is None else str(gather_stream)),
            (None if reduce_stream is None else str(reduce_stream)),
            shard_params,
            shard_grads,
            bucket_size,
        )

    raise ValueError(f"Schedule directives must be JSON objects, got: {type(directive)}")


def _normalize_filter_spec(filter_spec: Any, directive: Any) -> dict[str, Any]:
    """Normalize a single filter spec into {tag_name: value}."""
    def _norm_value(k: str, v: Any) -> Any:
        if v == "-":
            return _ABSENT_TAG_INDEX
        if k == "PASS":
            if not isinstance(v, str):
                raise ValueError(
                    f"'PASS' filter value must be a string in {{'F','B','BI','BW'}}: {directive}"
                )
            if v not in ("F", "B", "BI", "BW"):
                raise ValueError(
                    f"Invalid 'PASS' filter value '{v}'. Allowed: 'F', 'B', 'BI', 'BW': {directive}"
                )
            return v
        if v == "*":
            return _ANY_TAG_INDEX
        if v is None:
            return None
        if isinstance(v, int):
            return int(v)
        if isinstance(v, str):
            # Keep pass-like symbolic tags (e.g., "FWD"/"BWD") as strings.
            try:
                return int(v)
            except ValueError:
                return v
        return v

    if not isinstance(filter_spec, dict):
        raise ValueError(f"filter must be a JSON object: {directive}")
    out: dict[str, Any] = {}
    for k, v in filter_spec.items():
        if not isinstance(k, str):
            raise ValueError(f"Filter keys must be strings: {directive}")
        out[k] = _norm_value(k, v)
    return out


def _iter_filter_specs(spec: Any):
    if isinstance(spec, dict):
        yield spec
        return
    if isinstance(spec, list):
        if all(
            isinstance(item, (list, tuple))
            and len(item) == 2
            and isinstance(item[0], str)
            for item in spec
        ):
            yield spec
            return
        for item in spec:
            yield from _iter_filter_specs(item)


def _schedule_filter_tag_names(directives: Any) -> set[str]:
    if not isinstance(directives, list):
        return set()

    runtime_tags = {"PASS"}
    for directive in directives:
        if isinstance(directive, dict) and directive.get("op") == "split":
            dim_name = directive.get("dim_name")
            if isinstance(dim_name, str) and dim_name:
                runtime_tags.add(dim_name)

    tag_names: set[str] = set()
    for directive in directives:
        if not isinstance(directive, dict):
            continue
        specs = []
        if "filter" in directive:
            specs.append(directive["filter"])
        if isinstance(directive.get("filters"), list):
            specs.append(directive["filters"])
        for spec in _iter_filter_specs(specs):
            if isinstance(spec, dict):
                tag_names.update(
                    k
                    for k, value in spec.items()
                    if k not in runtime_tags and value != "-"
                )
            elif isinstance(spec, list):
                for item in spec:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and item[0] not in runtime_tags
                        and item[1] != "-"
                    ):
                        tag_names.add(item[0])
    return tag_names


def _validate_schedule_tags_exist(
    training_dag: TrainingDAG,
    directives: Any,
) -> None:
    schedule_tags = _schedule_filter_tag_names(directives)
    if not schedule_tags:
        return
    model_tags = {
        tag_name
        for node in training_dag.nodes.values()
        for tag_name in node.tag
        if tag_name != "PASS"
    }
    missing = sorted(schedule_tags - model_tags)
    if missing:
        raise ValueError(
            "Schedule references tag name(s) not found in model annotations: "
            f"{missing}. Model tag names: {sorted(model_tags)}."
        )


def _normalize_split_directive(directive: Any) -> tuple[dict[str, Any], str, int]:
    if not isinstance(directive, dict):
        raise ValueError(f"split directive must be a dict: {directive}")
    if directive.get("op") != "split":
        raise ValueError(f"Unsupported split directive op: {directive}")
    if "filter" not in directive:
        raise ValueError(f"split directive requires 'filter': {directive}")
    dim_name = directive.get("dim_name")
    if not isinstance(dim_name, str) or not dim_name:
        raise ValueError(f"split directive requires non-empty string dim_name: {directive}")
    num_microbatches = int(directive.get("num_microbatches", 0))
    if num_microbatches <= 0:
        raise ValueError(f"split directive requires num_microbatches > 0: {directive}")
    flt = _normalize_filter_spec(directive["filter"], directive)
    return flt, dim_name, num_microbatches


def _parse_order_directive(directive: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(directive, dict):
        raise ValueError(f"order directive must be a dict: {directive}")
    if directive.get("op") != "order":
        raise ValueError(f"Unsupported order directive op: {directive}")
    raw_filters = directive.get("filters")
    if not isinstance(raw_filters, list) or len(raw_filters) < 2:
        raise ValueError(
            f"order directive requires filters list with at least 2 groups: {directive}"
        )

    filter_groups: list[list[dict[str, Any]]] = []
    for group_idx, raw_group in enumerate(raw_filters):
        if not isinstance(raw_group, list) or not raw_group:
            raise ValueError(
                f"order directive filter group[{group_idx}] must be a non-empty list "
                f"of filters: {directive}"
            )

        group = []
        for raw_filter in raw_group:
            if not isinstance(raw_filter, dict):
                raise ValueError(
                    f"order directive filter group[{group_idx}] contains invalid "
                    f"filter object: {raw_filter}"
                )
            flt = _normalize_filter_spec(raw_filter, directive)
            pass_value = flt.get("PASS")
            if pass_value is not None and pass_value not in _VALID_ORDER_PASSES:
                raise ValueError(
                    f"order directive has unsupported pass={pass_value!r}; "
                    f"expected one of {sorted(_VALID_ORDER_PASSES)}"
                )
            group.append(flt)
        filter_groups.append(group)

    return filter_groups


def _filter_key_without(flt: dict[str, Any], ignored_keys: set[str]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted((k, v) for k, v in flt.items() if k not in ignored_keys))


def _validate_split_backward_order_stencil(
    directives: list[Any],
) -> set[tuple[tuple[str, Any], ...]]:
    """Validate BI/BW order entries and return base filter keys to split.

    This is intentionally narrow: it validates only the current split-backward
    stencil until the schedule JSON grows a full schema validation pass.
    """
    split_keys: set[tuple[tuple[str, Any], ...]] = set()
    split_entries: dict[tuple[tuple[str, Any], ...], list[tuple[int, tuple[int, int], str]]] = {}

    for directive_idx, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "order":
            continue
        filter_groups = _parse_order_directive(raw)
        seen_in_row: dict[tuple[tuple[str, Any], ...], dict[str, tuple[int, int]]] = {}
        for group_idx, group in enumerate(filter_groups):
            for filter_idx, flt in enumerate(group):
                filter_pos = (group_idx, filter_idx)
                pass_value = flt.get("PASS")
                if pass_value is None:
                    continue
                if pass_value == _FUSED_BWD_PASS:
                    continue
                if pass_value not in (_BWD_INPUT_PASS, _BWD_WEIGHT_PASS):
                    continue
                key = _filter_key_without(flt, {"PASS"})
                by_pass = seen_in_row.setdefault(key, {})
                if pass_value in by_pass:
                    raise ValueError(
                        f"order directive[{directive_idx}] has duplicate split-backward "
                        f"{pass_value} entry for {dict(key)} at indices "
                        f"{by_pass[pass_value]} and {filter_pos}"
                    )
                by_pass[pass_value] = filter_pos
                split_keys.add(key)
                split_entries.setdefault(key, []).append((directive_idx, filter_pos, pass_value))

    for key, entries in sorted(split_entries.items()):
        bi = [entry for entry in entries if entry[2] == _BWD_INPUT_PASS]
        bw = [entry for entry in entries if entry[2] == _BWD_WEIGHT_PASS]
        if len(bi) != 1 or len(bw) != 1:
            raise ValueError(
                f"split backward order entries for {dict(key)} must contain exactly "
                f"one BI and one BW; got BI={len(bi)} BW={len(bw)}"
            )
        if bi[0][0] != bw[0][0]:
            raise ValueError(
                f"split backward order entries for {dict(key)} must appear in the "
                "same order directive row"
            )
        if bi[0][1] > bw[0][1]:
            raise ValueError(
                f"split backward order entries for {dict(key)} must place BI before BW"
            )

    return split_keys



def _device_to_physical_pp_rank(dag: TrainingDAG) -> dict[tuple[int, ...], int]:
    device_keys = sorted({
        tuple(sorted(node.device))
        for node in dag.nodes.values()
        if node.device is not None
    })
    return {key: idx for idx, key in enumerate(device_keys)}


def _insert_send_recv_comm_nodes(dag: TrainingDAG, comm_stream: str | None = None) -> None:
    data_edges = _iter_data_edges(dag)
    replaced_edges: list[TrainingDAGEdge] = []
    device_to_pp_rank = _device_to_physical_pp_rank(dag)
    send_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "SEND_COMM")
    recv_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "RECV_COMM")
    for edge in data_edges:
        src = dag.nodes[edge.src_uid]
        dst = dag.nodes[edge.dst_uid]
        if src.node_kind != "COMPUTE" or dst.node_kind != "COMPUTE":
            continue
        if src.device is None or dst.device is None:
            raise ValueError(f"cannot insert send/recv for unplaced edge {src.uid} -> {dst.uid}")
        if src.device == dst.device:
            continue
        src_device_key = tuple(sorted(src.device))
        dst_device_key = tuple(sorted(dst.device))
        overlap = sorted(set(src_device_key) & set(dst_device_key))
        if overlap:
            raise ValueError(
                "generic pipeline SEND/RECV requires disjoint actor groups; "
                f"edge {src.uid}->{dst.uid} has source_devices="
                f"{list(src_device_key)}, destination_devices="
                f"{list(dst_device_key)}, overlap={overlap}"
            )
        if len(src_device_key) != len(dst_device_key):
            raise ValueError(
                "generic pipeline SEND/RECV requires equal-width actor "
                f"groups; edge {src.uid}->{dst.uid} has source_devices="
                f"{list(src_device_key)} (width={len(src_device_key)}), "
                f"destination_devices={list(dst_device_key)} "
                f"(width={len(dst_device_key)}). Unequal-width routed "
                "boundaries must be lowered by a shard directive."
            )
        send_uid = f"send.{send_idx}"
        send_idx += 1
        recv_uid = f"recv.{recv_idx}"
        recv_idx += 1
        stream_name = comm_stream if comm_stream is not None else "pp_stream"
        src_pp_rank = device_to_pp_rank[src_device_key]
        dst_pp_rank = device_to_pp_rank[dst_device_key]
        comm_id = f"p2p:{src.uid}->{dst.uid}:{edge.tensor_name or '-'}"
        comm_group_kind = "pp_lo_hi" if src_pp_rank < dst_pp_rank else "pp_hi_lo"
        participants = sorted(set(int(rank) for rank in (*src.device, *dst.device)))
        send_node = TrainingDAGNode(
            uid=send_uid,
            node_kind="SEND_COMM",
            compute_subkind=None,
            tag=dict(src.tag),
            device=(None if src.device is None else list(src.device)),
            stream=stream_name,
            node_meta={
                "from_uid": src.uid,
                "to_uid": dst.uid,
                "peer_pp_rank": dst_pp_rank,
                "bucket_key": src.node_meta.get("bucket_key"),
                "comm_id": comm_id,
                "comm_group_kind": comm_group_kind,
                "participants": participants,
            },
        )
        recv_node = TrainingDAGNode(
            uid=recv_uid,
            node_kind="RECV_COMM",
            compute_subkind=None,
            tag=dict(dst.tag),
            device=(None if dst.device is None else list(dst.device)),
            stream=stream_name,
            node_meta={
                "from_uid": src.uid,
                "to_uid": dst.uid,
                "peer_pp_rank": src_pp_rank,
                "bucket_key": dst.node_meta.get("bucket_key"),
                "comm_id": comm_id,
                "comm_group_kind": comm_group_kind,
                "participants": participants,
            },
        )
        dag.add_node(send_node)
        dag.add_node(recv_node)
        replaced_edges.append(edge)
        dag.add_edge(TrainingDAGEdge(src_uid=src.uid, dst_uid=send_uid, dep_kind="data", tensor_name=edge.tensor_name))
        dag.add_edge(TrainingDAGEdge(src_uid=recv_uid, dst_uid=dst.uid, dep_kind="data", tensor_name=edge.tensor_name))
    _remove_edges(dag, replaced_edges)


def _replicate_update_nodes_by_device(dag: TrainingDAG) -> None:
    """Replicate UPD nodes per upstream device set so split components stay homogeneous."""
    upd_nodes = [n for n in list(dag.nodes.values()) if n.node_kind == "UPD" and ".rep" not in n.uid]
    for upd in upd_nodes:
        in_edges = dag.incoming_edges(upd.uid, dep_kind="temporal")
        by_dev: dict[tuple[int, ...], list[TrainingDAGEdge]] = {}
        for e in in_edges:
            src = dag.nodes[e.src_uid]
            if src.device is None:
                continue
            key = tuple(sorted(src.device))
            by_dev.setdefault(key, []).append(e)
        if len(by_dev) <= 1:
            if len(by_dev) == 1:
                upd.device = list(next(iter(by_dev.keys())))
            continue
        created_uids: list[str] = []
        rewired_edges: list[tuple[TrainingDAGEdge, str]] = []
        for i_upd, (dev_key, dev_edges) in enumerate(sorted(by_dev.items(), key=lambda kv: kv[0])):
            nu = upd.uid if i_upd == 0 else f"{upd.uid}.rep{i_upd}"
            if i_upd == 0:
                upd.device = list(dev_key)
            else:
                dag.add_node(
                    TrainingDAGNode(
                        uid=nu,
                        node_kind="UPD",
                        compute_subkind=None,
                        tag=dict(upd.tag),
                        device=list(dev_key),
                        stream=upd.stream,
                        node_meta=dict(upd.node_meta),
                        profile_times_by_mode={
                            mode: dict(profiles)
                            for mode, profiles in upd.profile_times_by_mode.items()
                        },
                    )
                )
            created_uids.append(nu)
            for e in dev_edges:
                rewired_edges.append((e, nu))
        _remove_edges(dag, [edge for edge, _dst_uid in rewired_edges])
        for edge, dst_uid in rewired_edges:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=edge.src_uid,
                    dst_uid=dst_uid,
                    dep_kind=edge.dep_kind,
                    tensor_name=edge.tensor_name,
                )
            )
        # Duplicate outgoing edges of the original UPD to all replicas.
        out_edges = dag.outgoing_edges(upd.uid, dep_kind="temporal")
        if out_edges:
            for e in out_edges:
                for nu in created_uids[1:]:
                    dag.add_edge(
                        TrainingDAGEdge(
                            src_uid=nu,
                            dst_uid=e.dst_uid,
                            dep_kind=e.dep_kind,
                            tensor_name=e.tensor_name,
                        )
                    )



def _rewire_bwd_successors_through_sync(
    dag: TrainingDAG,
    bwd_uid: str,
    sync_uid: str,
) -> None:
    """Add update ordering through a gradient sync comm node.

    BWD data successors carry activation gradients, including P2P sends to an
    upstream pipeline stage.  Gradient synchronization is for parameter grads,
    so it must not be inserted into those data paths.  UPD dependencies are
    temporal; add sync -> UPD temporal edges so the
    update waits for gradient synchronization as well as the BWD compute.
    """
    upd_temporal_outs = [
        e for e in dag.outgoing_edges(bwd_uid, dep_kind="temporal")
        if dag.nodes[e.dst_uid].node_kind == "UPD"
    ]
    for e in upd_temporal_outs:
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=sync_uid,
                dst_uid=e.dst_uid,
                dep_kind="temporal",
                tensor_name=None,
            )
        )


def _reserve_comm_uid(
    dag: TrainingDAG,
    *,
    node_kind: str,
    prefix: str,
) -> str:
    """Reserve a stable legacy-style communication-node UID."""

    index = dag._next_comm_index_by_kind.get(node_kind)
    if index is None:
        index = sum(
            1 for node in dag.nodes.values() if node.node_kind == node_kind
        )
    dag._next_comm_index_by_kind[node_kind] = index + 1
    return f"{prefix}.{index}"


def _bucket_matched_fwd_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    bucket_size_mb: int,
) -> None:
    if bucket_size_mb <= 0:
        raise ValueError(f"bucket_size must be > 0 MB, got {bucket_size_mb}")
    bucket_size_bytes = int(bucket_size_mb) * 1024 * 1024

    fwd_targets = [
        uid
        for uid in _matching_uids(dag, filters)
        if dag.nodes[uid].node_kind == "COMPUTE"
        and dag.nodes[uid].compute_subkind == "FWD"
    ]

    # Deterministic rewrite order.
    fwd_targets.sort(key=lambda uid: (
        int(dag.nodes[uid].node_meta.get("stage_id", 10**9)),
        int(dag.nodes[uid].node_meta.get("segment_id", 10**9)),
        int(dag.nodes[uid].tag.get("MB", 0) or 0),
    ))

    for fwd_uid in fwd_targets:
        if fwd_uid not in dag.nodes:
            continue
        fwd_node = dag.nodes[fwd_uid]
        base_bucket_key = fwd_node.node_meta.get("bucket_key", fwd_uid)
        bwd_candidates = sorted(
            uid for uid, node in dag.nodes.items()
            if node.node_kind == "COMPUTE"
            and node.compute_subkind == "BWD"
            and node.node_meta.get("fwd_uid") == fwd_uid
        )
        if len(bwd_candidates) != 1:
            raise ValueError(
                f"Expected exactly one fused BWD node for {fwd_uid}, "
                f"found {bwd_candidates}"
            )
        bwd_uid = bwd_candidates[0]
        bwd_node = dag.nodes[bwd_uid]

        gm = fwd_node.node_meta.get("gm")
        graphargs = fwd_node.node_meta.get("graphargs")
        input_idxs = fwd_node.node_meta.get("input_idxs")
        param_idxs = fwd_node.node_meta.get("param_idxs")
        if gm is None or graphargs is None or input_idxs is None or param_idxs is None:
            continue

        def _param_bytes(gargs: list, pidxs: list[int]) -> int:
            total = 0
            for i in pidxs:
                if i < 0 or i >= len(gargs):
                    continue
                t = gargs[i]
                if t is not None and hasattr(t, "numel") and hasattr(t, "element_size"):
                    total += int(t.numel()) * int(t.element_size())
            return total

        def _placeholder_names(module: fx.GraphModule) -> list[str]:
            return [n.name for n in module.graph.nodes if n.op == "placeholder"]

        def _param_names(module: fx.GraphModule, pidxs: list[int]) -> list[str]:
            names = _placeholder_names(module)
            return [
                names[i] if 0 <= i < len(names) else f"<invalid-param-idx:{i}>"
                for i in pidxs
            ]

        pre_bytes = _param_bytes(graphargs, param_idxs)

        try:
            buckets = bucket_stage(
                gm,
                graphargs,
                input_idxs,
                param_idxs,
                bucket_size_bytes=bucket_size_bytes,
                debug_name=fwd_uid,
            )
        except Exception as exc:
            logger.warning(
                "replicate bucketing skipped for node %s (stage=%s seg=%s): %s",
                fwd_uid,
                fwd_node.node_meta.get("stage_id"),
                fwd_node.node_meta.get("segment_id"),
                exc,
            )
            continue
        if len(buckets) <= 1:
            continue
        original_param_names = set(_param_names(gm, param_idxs))
        lowered_param_counts: Counter[str] = Counter()
        for bi, (_bgm, _bin, b_param, b_args) in enumerate(buckets):
            post_bytes = _param_bytes(b_args, b_param)
            bucket_param_names = _param_names(_bgm, b_param)
            lowered_param_counts.update(bucket_param_names)

        missing_param_names = sorted(original_param_names - set(lowered_param_counts))
        unknown_param_names = sorted(set(lowered_param_counts) - original_param_names)
        duplicate_param_names = sorted(
            name for name, count in lowered_param_counts.items() if count != 1
        )

        if missing_param_names or duplicate_param_names or unknown_param_names:
            logger.warning(
                "invalid bucket node=%s missing=%s duplicates=%s unknown=%s",
                fwd_uid,
                missing_param_names[:16],
                duplicate_param_names[:16],
                unknown_param_names[:16],
            )

        # Snapshot old incident edges before rewrite.
        in_fwd = dag.incoming_edges(fwd_uid)
        out_fwd = dag.outgoing_edges(fwd_uid)
        in_bwd = dag.incoming_edges(bwd_uid)
        out_bwd = dag.outgoing_edges(bwd_uid)

        orig_stage = int(fwd_node.node_meta.get("stage_id", -1))
        orig_seg = int(fwd_node.node_meta.get("segment_id", -1))

        fwd_bucket_uids: list[str] = []
        bwd_bucket_uids: list[str] = []
        for bi, (bgm, b_in, b_param, b_args) in enumerate(buckets):
            triton_side_table = _collect_triton_side_table(bgm)
            (
                bucket_input_names,
                bucket_output_names,
                bucket_tensor_output_indices,
                bucket_differentiable_output_indices,
            ) = _node_io_names(bgm)
            f_uid = f"{fwd_uid}.bucket{bi}"
            b_uid = f"{f_uid}.bwd"
            bucket_key = f"{base_bucket_key}.bucket{bi}"
            fwd_bucket_uids.append(f_uid)
            bwd_bucket_uids.append(b_uid)
            dag.add_node(
                TrainingDAGNode(
                    uid=f_uid,
                    node_kind="COMPUTE",
                    compute_subkind="FWD",
                    tag=_with_pass_tag(fwd_node.tag, "F"),
                    device=list(fwd_node.device) if fwd_node.device is not None else None,
                    stream=fwd_node.stream,
                    node_meta={
                        "stage_id": orig_stage,
                        "segment_id": orig_seg * 1000 + bi,
                        "gm": bgm,
                        "input_idxs": list(b_in),
                        "param_idxs": list(b_param),
                        "graphargs": list(b_args),
                        "input_names": bucket_input_names,
                        "output_names": bucket_output_names,
                        "tensor_output_indices": bucket_tensor_output_indices,
                        "differentiable_output_indices": (
                            bucket_differentiable_output_indices
                        ),
                        "a2a_boundary_after": None,
                        "triton_kernels": triton_side_table["kernels"],
                        "triton_constant_args": triton_side_table[
                            "constant_args"
                        ],
                        "bucket_idx": bi,
                        "num_buckets": len(buckets),
                        "bucket_key": bucket_key,
                    },
                )
            )
            dag.add_node(
                TrainingDAGNode(
                    uid=b_uid,
                    node_kind="COMPUTE",
                    compute_subkind="BWD",
                    tag=_with_pass_tag(fwd_node.tag, "B"),
                    device=list(bwd_node.device) if bwd_node.device is not None else None,
                    stream=bwd_node.stream,
                    node_meta={
                        "fwd_uid": f_uid,
                        "bucket_idx": bi,
                        "num_buckets": len(buckets),
                        "bucket_key": bucket_key,
                        "compute_loss": bool(bwd_node.node_meta.get("compute_loss", False)) and bi == (len(buckets) - 1),
                    },
                )
            )

        first_fwd = fwd_bucket_uids[0]
        last_fwd = fwd_bucket_uids[-1]
        # A routed EP boundary belongs to the value produced after the whole
        # original segment. Preserve it on the final bucket; otherwise a later
        # shard directive cannot find the dispatch-count output once this node
        # has been replaced by bucket nodes.
        boundary_after = fwd_node.node_meta.get("a2a_boundary_after")
        if isinstance(boundary_after, dict):
            dag.nodes[last_fwd].node_meta["a2a_boundary_after"] = dict(
                boundary_after
            )
        # Mirror backward bucket chain in reverse order.
        first_bwd = bwd_bucket_uids[-1]
        last_bwd = bwd_bucket_uids[0]

        def _remap(uid: str) -> str:
            if uid == fwd_uid:
                return last_fwd
            if uid == bwd_uid:
                return first_bwd
            return uid

        # Remove originals and their incident edges.
        _remove_node_and_incident_edges(dag, fwd_uid)
        _remove_node_and_incident_edges(dag, bwd_uid)

        # Re-attach incoming/outgoing edges at bucket chain boundaries.
        for e in in_fwd:
            dag.add_edge(TrainingDAGEdge(src_uid=_remap(e.src_uid), dst_uid=first_fwd, dep_kind=e.dep_kind, tensor_name=e.tensor_name))
        for e in out_fwd:
            dst = first_bwd if e.dst_uid == bwd_uid else e.dst_uid
            dag.add_edge(TrainingDAGEdge(src_uid=last_fwd, dst_uid=dst, dep_kind=e.dep_kind, tensor_name=e.tensor_name))
        for e in in_bwd:
            dag.add_edge(TrainingDAGEdge(src_uid=_remap(e.src_uid), dst_uid=first_bwd, dep_kind=e.dep_kind, tensor_name=e.tensor_name))
        for e in out_bwd:
            dag.add_edge(TrainingDAGEdge(src_uid=last_bwd, dst_uid=_remap(e.dst_uid), dep_kind=e.dep_kind, tensor_name=e.tensor_name))

        # Internal chain edges.
        for i in range(len(fwd_bucket_uids) - 1):
            dag.add_edge(TrainingDAGEdge(src_uid=fwd_bucket_uids[i], dst_uid=fwd_bucket_uids[i + 1], dep_kind="data", tensor_name=None))
        for i in range(len(bwd_bucket_uids) - 1, 0, -1):
            dag.add_edge(TrainingDAGEdge(src_uid=bwd_bucket_uids[i], dst_uid=bwd_bucket_uids[i - 1], dep_kind="data", tensor_name=None))


def _insert_reduce_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    """Insert one all-reduce per bucket after all matched weight backwards."""
    expected = sorted(int(d) for d in devices)
    if len(expected) <= 1:
        # A singleton replica group is no-DP. Local autograd still accumulates
        # gradients across microbatches, but there is nothing to synchronize.
        return

    candidates: list[TrainingDAGNode] = []
    for uid in _matching_uids(dag, filters):
        node = dag.nodes[uid]
        if node.node_kind != "COMPUTE" or not _is_backward_weight_subkind(node.compute_subkind):
            continue
        if node.device is None:
            raise ValueError(
                f"replicate requires placed backward weight-gradient nodes, but node {node.uid} has device=None; "
                f"expected devices={devices}"
            )
        got = sorted(int(d) for d in node.device)
        if got != expected:
            raise ValueError(
                f"replicate device mismatch for node {node.uid}: "
                f"expected devices={devices}, node_devices={node.device}"
            )
        candidates.append(node)

    by_bucket: dict[Any, list[TrainingDAGNode]] = {}
    for node in candidates:
        bucket_key = node.node_meta.get("bucket_key")
        # Missing bucket metadata cannot safely alias another node.
        group_key = (
            ("bucket", bucket_key)
            if bucket_key is not None
            else ("uid", node.uid)
        )
        by_bucket.setdefault(group_key, []).append(node)
    reduce_groups = list(by_bucket.values())

    for group in reduce_groups:
        anchor = group[-1]
        common_tag_keys = set(anchor.tag)
        for node in group[:-1]:
            common_tag_keys.intersection_update(node.tag)
        reduce_tag = {
            key: anchor.tag[key]
            for key in sorted(common_tag_keys)
            if all(node.tag[key] == anchor.tag[key] for node in group)
        }
        reduce_uid = _reserve_comm_uid(
            dag,
            node_kind="REDUCE_COMM",
            prefix="reduce",
        )
        bucket_key = anchor.node_meta.get("bucket_key")
        comm_key = (
            bucket_key
            if len(group) > 1 and bucket_key is not None
            else anchor.uid
        )
        reduce_node = TrainingDAGNode(
            uid=reduce_uid,
            node_kind="REDUCE_COMM",
            compute_subkind=None,
            # Drop tags such as MB/PASS that differ among the contributing
            # backwards. This keeps order directives from assigning the
            # accumulated collective to one particular microbatch slot.
            tag=reduce_tag,
            device=list(anchor.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "bwd_uid": anchor.uid,
                "bwd_uids": [node.uid for node in group],
                "bucket_key": bucket_key,
                "comm_id": f"all_reduce:{comm_key}",
                "comm_group_kind": "dp",
                "participants": sorted(int(rank) for rank in anchor.device),
            },
        )
        dag.add_node(reduce_node)
        # Keep one data edge for the existing runtime payload/buffer contract.
        # The remaining dependencies are temporal because the collective reads
        # accumulated parameter grads, not task-output activation buffers.
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=anchor.uid,
                dst_uid=reduce_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )
        for node in group[:-1]:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=node.uid,
                    dst_uid=reduce_uid,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )
        for node in group:
            _rewire_bwd_successors_through_sync(dag, node.uid, reduce_uid)


def _node_has_trainable_params(dag: TrainingDAG, node: TrainingDAGNode) -> bool:
    meta = node.node_meta
    if _is_backward_compute_subkind(node.compute_subkind):
        fwd_uid = meta.get("fwd_uid")
        if isinstance(fwd_uid, str) and fwd_uid in dag.nodes:
            meta = dag.nodes[fwd_uid].node_meta
    graphargs = meta.get("graphargs")
    param_idxs = meta.get("param_idxs")
    if graphargs is None or param_idxs is None:
        return True
    return any(
        0 <= int(i) < len(graphargs)
        and graphargs[int(i)] is not None
        and bool(getattr(graphargs[int(i)], "requires_grad", False))
        for i in param_idxs
    )


def _insert_all_gather_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    expected = sorted(int(d) for d in devices)
    for uid in _matching_uids(dag, filters):
        node = dag.nodes[uid]
        if (
            node.node_kind != "COMPUTE"
            or node.compute_subkind not in ("FWD", "BWD", "BWD_I", "BWD_W")
        ):
            continue
        if not _node_has_trainable_params(dag, node):
            continue
        if node.device is None:
            raise ValueError(
                f"replicate(shard_params=True) requires placed nodes, but node {node.uid} has device=None; "
                f"expected devices={devices}"
            )
        got = sorted(int(d) for d in node.device)
        if got != expected:
            raise ValueError(
                f"replicate device mismatch for node {node.uid}: "
                f"expected devices={devices}, node_devices={node.device}"
            )
        ag_uid = _reserve_comm_uid(
            dag,
            node_kind="ALL_GATHER_COMM",
            prefix="all_gather",
        )
        ag_tag = dict(node.tag)
        ag_node = TrainingDAGNode(
            uid=ag_uid,
            node_kind="ALL_GATHER_COMM",
            compute_subkind=None,
            tag=ag_tag,
            device=list(node.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "compute_uid": node.uid,
                "bucket_key": node.node_meta.get("bucket_key"),
                "comm_id": f"all_gather:{node.uid}",
                "comm_group_kind": "dp",
                "participants": sorted(int(rank) for rank in node.device),
            },
        )
        dag.add_node(ag_node)
        node.node_meta["zero_free_full_params_after"] = True
        if node.compute_subkind == "BWD_W":
            for pred_uid in sorted(dag.preds.get(node.uid, set())):
                pred = dag.nodes[pred_uid]
                if pred.node_kind == "COMPUTE" and pred.compute_subkind == "BWD_I":
                    dag.add_edge(
                        TrainingDAGEdge(
                            src_uid=pred.uid,
                            dst_uid=ag_uid,
                            dep_kind="data",
                            tensor_name=None,
                        )
                    )
        # ALL_GATHER is a pre-dependency of compute: AG -> COMPUTE
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=ag_uid,
                dst_uid=node.uid,
                dep_kind="data",
                tensor_name=None,
            )
        )


def _insert_reduce_scatter_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    """Insert one reduce-scatter per bucket after all matched backwards.

    Full gradients remain live across all microbatches and each backward adds
    into the same flat buffer.  The reduce-scatter therefore has the same
    gradient-accumulation dependency shape as an unsharded all-reduce: one data
    predecessor preserves the runtime payload contract and every other
    contributing backward is a temporal predecessor.

    Parameter lifetime is independent of gradient synchronization.  In
    particular, ``zero_free_full_params_after`` remains on every backward so
    shard_params=True reshares parameters immediately unless the later ZeRO
    lifetime-pruning pass proves that a longer full-parameter lifetime is safe.
    """
    expected = sorted(int(d) for d in devices)
    candidates: list[TrainingDAGNode] = []
    for uid in _matching_uids(dag, filters):
        node = dag.nodes[uid]
        if node.node_kind != "COMPUTE" or not _is_backward_weight_subkind(node.compute_subkind):
            continue
        if not _node_has_trainable_params(dag, node):
            continue
        if node.device is None:
            raise ValueError(
                f"replicate(shard_grads/shard_params) requires placed backward weight-gradient nodes, but node {node.uid} has device=None; "
                f"expected devices={devices}"
            )
        got = sorted(int(d) for d in node.device)
        if got != expected:
            raise ValueError(
                f"replicate device mismatch for node {node.uid}: "
                f"expected devices={devices}, node_devices={node.device}"
            )
        candidates.append(node)

    by_bucket: dict[Any, list[TrainingDAGNode]] = {}
    for node in candidates:
        bucket_key = node.node_meta.get("bucket_key")
        group_key = (
            ("bucket", bucket_key)
            if bucket_key is not None
            else ("uid", node.uid)
        )
        by_bucket.setdefault(group_key, []).append(node)

    for group in by_bucket.values():
        anchor = group[-1]
        common_tag_keys = set(anchor.tag)
        for node in group[:-1]:
            common_tag_keys.intersection_update(node.tag)
        reduce_tag = {
            key: anchor.tag[key]
            for key in sorted(common_tag_keys)
            if all(node.tag[key] == anchor.tag[key] for node in group)
        }
        rs_uid = _reserve_comm_uid(
            dag,
            node_kind="REDUCE_SCATTER_COMM",
            prefix="reduce_scatter",
        )
        bucket_key = anchor.node_meta.get("bucket_key")
        comm_key = (
            bucket_key
            if len(group) > 1 and bucket_key is not None
            else anchor.uid
        )
        rs_node = TrainingDAGNode(
            uid=rs_uid,
            node_kind="REDUCE_SCATTER_COMM",
            compute_subkind=None,
            tag=reduce_tag,
            device=list(anchor.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "bwd_uid": anchor.uid,
                "bwd_uids": [node.uid for node in group],
                "bucket_key": bucket_key,
                "comm_id": f"reduce_scatter:{comm_key}",
                "comm_group_kind": "dp",
                "participants": sorted(int(rank) for rank in anchor.device),
            },
        )
        for node in group:
            node.node_meta["zero_alloc_full_grads_before"] = True
        dag.add_node(rs_node)
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=anchor.uid,
                dst_uid=rs_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )
        for node in group[:-1]:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=node.uid,
                    dst_uid=rs_uid,
                    dep_kind="temporal",
                    tensor_name=None,
                )
            )
        for node in group:
            _rewire_bwd_successors_through_sync(dag, node.uid, rs_uid)


def _routed_boundary_for_edge(
    dag: TrainingDAG,
    src_node: TrainingDAGNode,
    dst_node: TrainingDAGNode,
) -> dict[str, Any]:
    """Resolve marker metadata for a routed forward or backward data edge."""
    # Backward data edges reverse their forward edge.  The tensor position is
    # therefore described by the forward node paired with the backward edge's
    # destination (the original forward producer).
    if (
        _is_backward_activation_subkind(src_node.compute_subkind)
        and _is_backward_activation_subkind(dst_node.compute_subkind)
    ):
        fwd_producer_uid = dst_node.node_meta.get("fwd_uid")
        if not isinstance(fwd_producer_uid, str) or fwd_producer_uid not in dag.nodes:
            raise ValueError(
                "routed communication could not resolve forward producer for "
                f"BWD edge {src_node.uid}->{dst_node.uid}: "
                f"fwd_uid={fwd_producer_uid!r}"
            )
        producer = dag.nodes[fwd_producer_uid]
    else:
        producer = src_node

    boundary_info = producer.node_meta.get("a2a_boundary_after")
    if not isinstance(boundary_info, dict) or boundary_info.get("tensor_idx") is None:
        raise ValueError(
            f"routed communication missing tensor_idx for edge "
            f"{src_node.uid}->{dst_node.uid}; producer={producer.uid} "
            f"boundary_info={boundary_info!r}"
        )
    return boundary_info


def _routed_expert_fwd_uid(compute_node: TrainingDAGNode) -> str:
    if compute_node.compute_subkind == "FWD":
        return compute_node.uid
    fwd_uid = compute_node.node_meta.get("fwd_uid")
    if not isinstance(fwd_uid, str):
        raise ValueError(
            f"Sharded backward node {compute_node.uid} is missing fwd_uid"
        )
    return fwd_uid


_ROUTE_OWNERSHIP_META_KEYS = (
    "num_global_routes",
    "shard_degree",
    "shard_devices",
    "num_local_routes",
    "route_partitions",
)


def _route_ownership_meta(node: TrainingDAGNode) -> dict[str, Any]:
    """Copy schedule-derived ownership metadata without inferring model semantics."""
    present = {
        key: node.node_meta[key]
        for key in _ROUTE_OWNERSHIP_META_KEYS
        if key in node.node_meta
    }
    if present and len(present) != len(_ROUTE_OWNERSHIP_META_KEYS):
        missing = sorted(set(_ROUTE_OWNERSHIP_META_KEYS) - set(present))
        raise ValueError(
            f"routed region {node.uid} has incomplete route ownership metadata; "
            f"missing={missing}"
        )
    return present


def _require_routed_slot(
    value: Any,
    *,
    arity: int,
    label: str,
    routed_uid: str,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= arity
    ):
        raise ValueError(
            f"routed region {routed_uid} has invalid {label}={value!r}; "
            f"expected an integer slot in [0, {arity})"
        )
    return value


def _routed_activation_input_names(node: TrainingDAGNode) -> list[str]:
    """Return routed runtime inputs, excluding lifted parameter placeholders."""
    input_names = node.node_meta.get("input_names", ())
    if not isinstance(input_names, (list, tuple)):
        raise ValueError(
            f"routed region {node.uid} has invalid input_names={input_names!r}"
        )
    input_idxs = node.node_meta.get("input_idxs")
    if input_idxs is None:
        # Compatibility for synthetic and older serialized DAGs, which only
        # recorded runtime inputs in ``input_names``.
        return [str(name) for name in input_names]
    if (
        not isinstance(input_idxs, list)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(input_names)
            for index in input_idxs
        )
        or len(input_idxs) != len(set(input_idxs))
    ):
        raise ValueError(
            f"routed region {node.uid} has invalid runtime input indices "
            f"{input_idxs!r} for {len(input_names)} placeholders"
        )
    return [str(input_names[index]) for index in input_idxs]


def _validate_routed_marker_slots(
    dispatch_owner: TrainingDAGNode,
    routed_node: TrainingDAGNode,
    *,
    shard_devices: tuple[int, ...],
) -> tuple[int, int, int, dict[str, Any]]:
    """Validate routed tensor slots shared by A2A and M/N lowering."""
    dispatch_boundary = dispatch_owner.node_meta.get("a2a_boundary_after")
    combine_boundary = routed_node.node_meta.get("a2a_boundary_after")
    assert isinstance(dispatch_boundary, dict)
    assert isinstance(combine_boundary, dict)

    owner_output_names = dispatch_owner.node_meta.get("output_names")
    routed_output_names = routed_node.node_meta.get("output_names")
    # Serialized/synthetic legacy DAGs may not carry IO names. Marker parsing
    # validates them before such metadata can be lost, so retain a narrow
    # compatibility fallback based on the largest declared marker index.
    owner_arity = (
        len(owner_output_names)
        if isinstance(owner_output_names, (list, tuple))
        else max(
            int(dispatch_boundary["tensor_idx"]),
            int(dispatch_boundary["routing_counts_idx"]),
        )
        + 1
    )
    routed_arity = (
        len(routed_output_names)
        if isinstance(routed_output_names, (list, tuple))
        else int(combine_boundary["tensor_idx"]) + 1
    )
    payload_idx = _require_routed_slot(
        dispatch_boundary.get("tensor_idx"),
        arity=owner_arity,
        label="dispatch payload index",
        routed_uid=routed_node.uid,
    )
    counts_idx = _require_routed_slot(
        dispatch_boundary.get("routing_counts_idx"),
        arity=owner_arity,
        label="dispatch counts index",
        routed_uid=routed_node.uid,
    )
    combine_idx = _require_routed_slot(
        combine_boundary.get("tensor_idx"),
        arity=routed_arity,
        label="combine payload index",
        routed_uid=routed_node.uid,
    )
    if payload_idx == counts_idx:
        raise ValueError(
            f"routed region {routed_node.uid} must use distinct dispatch "
            f"payload/count slots, both resolved to index {payload_idx}"
        )

    owner_tensor_indices = dispatch_owner.node_meta.get("tensor_output_indices")
    if isinstance(owner_tensor_indices, list):
        if (
            payload_idx not in owner_tensor_indices
            or counts_idx not in owner_tensor_indices
        ):
            raise ValueError(
                f"routed region {routed_node.uid} dispatch payload/count "
                "markers must identify tensor outputs; "
                f"tensor_indices={owner_tensor_indices}, "
                f"payload_idx={payload_idx}, counts_idx={counts_idx}"
            )
    routed_tensor_indices = routed_node.node_meta.get("tensor_output_indices")
    if (
        isinstance(routed_tensor_indices, list)
        and combine_idx not in routed_tensor_indices
    ):
        raise ValueError(
            f"routed region {routed_node.uid} combine marker must identify a "
            f"tensor output; tensor_indices={routed_tensor_indices}, "
            f"combine_idx={combine_idx}"
        )

    owner_diff_indices = dispatch_owner.node_meta.get(
        "differentiable_output_indices"
    )
    if isinstance(owner_diff_indices, list):
        if counts_idx in owner_diff_indices:
            raise ValueError(
                f"routed region {routed_node.uid} dispatch count slot "
                f"{counts_idx} must be non-differentiable"
            )
        if payload_idx not in owner_diff_indices:
            raise ValueError(
                f"routed region {routed_node.uid} dispatch payload slot "
                f"{payload_idx} must be differentiable because routed "
                "communication has an unconditional backward lifecycle"
            )
    routed_diff_indices = routed_node.node_meta.get(
        "differentiable_output_indices"
    )
    if (
        isinstance(routed_diff_indices, list)
        and combine_idx not in routed_diff_indices
    ):
        raise ValueError(
            f"routed region {routed_node.uid} combine payload slot "
            f"{combine_idx} must be differentiable because routed "
            "communication has an unconditional backward lifecycle"
        )

    counts_schema = dispatch_boundary.get("routing_counts_schema")
    counts_shape = (
        counts_schema.get("shape") if isinstance(counts_schema, dict) else None
    )
    if not (
        isinstance(counts_shape, (list, tuple))
        and len(counts_shape) == 2
        and all(
            not isinstance(dim, bool) and isinstance(dim, int) and dim >= 0
            for dim in counts_shape
        )
    ):
        raise ValueError(
            f"routed region {routed_node.uid} requires statically known rank-2 "
            "dispatch counts shaped [source_blocks, global_routes] before "
            f"schedule lowering; got routing_counts_shape={counts_shape!r}"
        )
    if counts_shape[0] != 1:
        raise ValueError(
            f"routed region {routed_node.uid} source-side dispatch counts must "
            "contain exactly one source block before communication; got "
            f"routing_counts_shape={tuple(counts_shape)}"
        )

    shard_degree = len(shard_devices)
    if shard_degree <= 0:
        raise ValueError(
            f"routed region {routed_node.uid} requires a nonempty ordered "
            "shard device list"
        )
    if len(set(shard_devices)) != shard_degree:
        raise ValueError(
            f"routed region {routed_node.uid} shard devices must be unique; "
            f"got devices={list(shard_devices)}"
        )
    num_global_routes = counts_shape[1]
    if num_global_routes <= 0:
        raise ValueError(
            f"routed region {routed_node.uid} requires at least one global "
            f"route; got num_global_routes={num_global_routes}"
        )
    if num_global_routes % shard_degree:
        raise ValueError(
            f"routed region {routed_node.uid} has "
            f"num_global_routes={num_global_routes}, which is not divisible by "
            f"shard_degree={shard_degree} for ordered shard devices "
            f"{list(shard_devices)}"
        )

    num_local_routes = num_global_routes // shard_degree
    route_partitions = [
        {
            "device": device,
            "start": shard_index * num_local_routes,
            "stop": (shard_index + 1) * num_local_routes,
        }
        for shard_index, device in enumerate(shard_devices)
    ]
    route_ownership = {
        "num_global_routes": num_global_routes,
        "shard_degree": shard_degree,
        "shard_devices": list(shard_devices),
        "num_local_routes": num_local_routes,
        "route_partitions": route_partitions,
    }
    return payload_idx, counts_idx, combine_idx, route_ownership


def _validate_route_owned_parameters(
    routed_node: TrainingDAGNode,
    route_ownership: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the initial complete-route parameter layout contract.

    A ``shard``-owned routed region materializes every trainable lifted
    parameter locally.  Route cardinality comes only from the marked counts and
    schedule; parameter shapes are checked against that result rather than used
    to infer it.  The initially supported parameter route axis is axis zero.
    """
    graphargs = routed_node.node_meta.get("graphargs")
    param_idxs = routed_node.node_meta.get("param_idxs")
    input_names = routed_node.node_meta.get("input_names")
    if graphargs is None or param_idxs is None:
        # Synthetic DAG fixtures may intentionally omit executable model data.
        return []
    if not isinstance(graphargs, list) or not isinstance(param_idxs, list):
        raise ValueError(
            f"routed region {routed_node.uid} has invalid graphargs/param_idxs "
            "metadata for route-owned parameter validation"
        )
    names = input_names if isinstance(input_names, (list, tuple)) else ()
    num_local_routes = int(route_ownership["num_local_routes"])
    num_global_routes = int(route_ownership["num_global_routes"])
    specs: list[dict[str, Any]] = []
    for raw_idx in param_idxs:
        if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
            raise ValueError(
                f"routed region {routed_node.uid} has invalid parameter index "
                f"{raw_idx!r}"
            )
        if raw_idx < 0 or raw_idx >= len(graphargs):
            raise ValueError(
                f"routed region {routed_node.uid} parameter index {raw_idx} "
                f"is outside graphargs length {len(graphargs)}"
            )
        value = graphargs[raw_idx]
        if value is None or not bool(getattr(value, "requires_grad", False)):
            continue
        shape = tuple(getattr(value, "shape", ()))
        if not shape or type(shape[0]) is not int:
            raise ValueError(
                f"routed region {routed_node.uid} trainable parameter "
                f"index {raw_idx} must have a statically known leading route "
                f"axis, got shape={shape}"
            )
        if int(shape[0]) != num_local_routes:
            name = names[raw_idx] if raw_idx < len(names) else f"arg[{raw_idx}]"
            raise ValueError(
                f"routed region {routed_node.uid} parameter {name!r} must "
                "materialize one complete slice per local route on leading "
                f"axis 0; got shape={shape}, num_local_routes={num_local_routes}"
            )
        global_shape = (num_global_routes, *tuple(int(dim) for dim in shape[1:]))
        specs.append(
            {
                "arg_idx": raw_idx,
                "placeholder_name": (
                    str(names[raw_idx])
                    if raw_idx < len(names)
                    else f"arg[{raw_idx}]"
                ),
                "route_axis": 0,
                "local_shape": tuple(int(dim) for dim in shape),
                "global_shape": global_shape,
            }
        )
    return specs


def _validate_cross_group_routed_interface(
    state_owner: TrainingDAGNode,
    routed_node: TrainingDAGNode,
) -> None:
    """Validate the retained-buffer ABI used only by disjoint M/N routes."""
    boundary = state_owner.node_meta.get("a2a_boundary_after")
    combine_boundary = routed_node.node_meta.get("a2a_boundary_after")
    assert isinstance(boundary, dict)
    assert isinstance(combine_boundary, dict)
    routing_counts_idx = boundary.get("routing_counts_idx")
    state_owner_output_arity = len(state_owner.node_meta.get("output_names", ()))
    routed_input_names = _routed_activation_input_names(routed_node)
    routed_output_names = routed_node.node_meta.get("output_names", ())
    routed_input_arity = len(routed_input_names)
    routed_output_arity = len(routed_output_names)
    if (
        state_owner_output_arity != routed_input_arity
        or routed_input_arity != routed_output_arity + 1
    ):
        raise ValueError(
            f"routed boundary {state_owner.uid}->{routed_node.uid} requires "
            "the dense state-owner output arity to equal the routed region "
            "input arity, with the final counts slot dropped from routed "
            "outputs; got "
            f"state_owner_output_arity={state_owner_output_arity}, "
            f"routed_input_arity={routed_input_arity}, "
            f"routed_output_arity={routed_output_arity}"
        )
    if routing_counts_idx != state_owner_output_arity - 1:
        raise ValueError(
            f"routed boundary {state_owner.uid}->{routed_node.uid} requires "
            "dispatch counts in the final dense state-owner output slot, "
            f"got routing_counts_idx={routing_counts_idx!r}, "
            f"final_index={state_owner_output_arity - 1}"
        )
    payload_idx = boundary.get("tensor_idx")
    combine_idx = combine_boundary.get("tensor_idx")
    if (
        isinstance(payload_idx, bool)
        or not isinstance(payload_idx, int)
        or payload_idx < 0
        or payload_idx >= routed_output_arity
        or combine_idx != payload_idx
    ):
        raise ValueError(
            f"routed boundary {state_owner.uid}->{routed_node.uid} requires "
            "dispatch and combine payloads at the same valid prefix slot, "
            f"got dispatch_idx={payload_idx!r}, combine_idx={combine_idx!r}, "
            f"routed_output_arity={routed_output_arity}"
        )
    non_routed_indices = set(range(routed_output_arity)) - {payload_idx}
    non_identity_indices = [
        index
        for index in sorted(non_routed_indices)
        if routed_input_names[index] != routed_output_names[index]
    ]
    if non_identity_indices:
        raise ValueError(
            f"routed region {routed_node.uid} must identity-pass every "
            "non-routed input through the same output prefix slot; "
            f"mismatched_indices={non_identity_indices}"
        )
    routed_gm = routed_node.node_meta.get("gm")
    if isinstance(routed_gm, fx.GraphModule):
        placeholders = [
            graph_node
            for graph_node in routed_gm.graph.nodes
            if graph_node.op == "placeholder"
        ]
        runtime_input_idxs = routed_node.node_meta.get("input_idxs")
        if runtime_input_idxs is None:
            runtime_input_idxs = list(range(len(routed_input_names)))
        used_non_routed_indices = [
            index
            for index in sorted(non_routed_indices)
            if any(
                user.op != "output"
                for user in placeholders[runtime_input_idxs[index]].users
            )
        ]
        if used_non_routed_indices:
            raise ValueError(
                f"routed region {routed_node.uid} may only compute from its "
                "payload/count inputs; non-routed identity inputs are used "
                f"by compute at indices={used_non_routed_indices}"
            )
    state_owner_diff_indices = state_owner.node_meta.get(
        "differentiable_output_indices"
    )
    routed_diff_indices = routed_node.node_meta.get(
        "differentiable_output_indices"
    )
    if (
        not isinstance(state_owner_diff_indices, list)
        or not isinstance(routed_diff_indices, list)
        or state_owner_diff_indices != routed_diff_indices
        or payload_idx not in state_owner_diff_indices
    ):
        raise ValueError(
            f"routed boundary {state_owner.uid}->{routed_node.uid} requires "
            "dense state-owner differentiable output indices to align with "
            "the routed region backward payload and include the routed slot, "
            f"got state_owner_indices={state_owner_diff_indices!r}, "
            f"routed_indices={routed_diff_indices!r}, "
            f"payload_idx={payload_idx}"
        )

    dispatch_schema = boundary.get("payload_schema")
    combine_schema = combine_boundary.get("payload_schema")
    if isinstance(dispatch_schema, dict) and isinstance(combine_schema, dict):
        dispatch_shape = dispatch_schema.get("shape")
        combine_shape = combine_schema.get("shape")
        if (
            isinstance(dispatch_shape, (list, tuple))
            and isinstance(combine_shape, (list, tuple))
            and len(dispatch_shape) >= 1
            and len(combine_shape) >= 1
            and tuple(dispatch_shape[1:]) != tuple(combine_shape[1:])
        ):
            raise ValueError(
                f"cross-group routed region {routed_node.uid} must preserve "
                "the payload trailing shape because destination ranks allocate "
                "collective buffers from the saved dispatch schema; got "
                f"dispatch={tuple(dispatch_shape)}, "
                f"combine={tuple(combine_shape)}"
            )
        dispatch_dtype = dispatch_schema.get("dtype")
        combine_dtype = combine_schema.get("dtype")
        if (
            dispatch_dtype is not None
            and combine_dtype is not None
            and dispatch_dtype != combine_dtype
        ):
            raise ValueError(
                f"cross-group routed region {routed_node.uid} must preserve "
                "payload dtype because both union-collective endpoint groups "
                f"allocate from one saved schema; got dispatch={dispatch_dtype}, "
                f"combine={combine_dtype}"
            )


def _validate_placed_routed_regions(
    dag: TrainingDAG,
    directives: list[Any],
) -> None:
    """Validate the placement-dependent routed-region grammar.

    Marker construction validates the payload/count slots themselves.  Once
    ``place`` has run, this check validates the larger structural contract
    required by both ragged A2A and cross-ActorGroup M-to-N/N-to-M lowering:

      replicated compute -> one sharded compute -> replicated compute

    The two boundaries must either both stay within one ActorGroup or both
    cross out of the sharded ActorGroup.  A cross-group combine may target a
    later dense ActorGroup, but the original dense state owner and that target
    must have equal widths so the retained buffer can be relayed lane-wise.

    ActorGroup identity is compared as an unordered device set. Route
    ownership maps equal contiguous ranges of the global route axis to the
    shard directive's devices in canonical ascending global-rank order. The
    schedule validator rejects noncanonical shard lists before this pass.
    """
    shard_specs: list[
        tuple[int, list[dict[str, Any]], tuple[int, ...], tuple[int, ...]]
    ] = []
    replicate_specs: list[tuple[list[dict[str, Any]], tuple[int, ...]]] = []
    for directive_idx, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") not in {"replicate", "shard"}:
            continue
        (
            op,
            filters,
            devices,
            _stream,
            _gather_stream,
            _reduce_stream,
            _shard_params,
            _shard_grads,
            _bucket_size,
        ) = _normalize_filter_devices_directive(raw)
        ordered_devices = tuple(int(device) for device in devices)
        device_key = tuple(sorted(ordered_devices))
        if op == "shard":
            shard_specs.append(
                (directive_idx, filters, ordered_devices, device_key)
            )
        else:
            replicate_specs.append((filters, device_key))

    replicated_compute_uids_by_group: dict[tuple[int, ...], set[str]] = {}
    for filters, device_key in replicate_specs:
        matched = replicated_compute_uids_by_group.setdefault(device_key, set())
        matched.update(
            uid
            for uid in _matching_uids(dag, filters)
            if dag.nodes[uid].node_kind == "COMPUTE"
        )

    shard_policy_by_fwd_uid: dict[str, tuple[int, tuple[int, ...]]] = {}
    shard_matched_uids_by_directive: dict[int, set[str]] = {}
    for (
        directive_idx,
        filters,
        ordered_shard_devices,
        expected_device_set,
    ) in shard_specs:
        matched_uids = {
            uid
            for uid in _matching_uids(dag, filters)
            if dag.nodes[uid].node_kind == "COMPUTE"
        }
        shard_matched_uids_by_directive[directive_idx] = matched_uids
        matched_fwd_uids = {
            uid
            for uid in matched_uids
            if dag.nodes[uid].compute_subkind == "FWD"
        }
        if not matched_fwd_uids:
            raise ValueError(
                f"shard directive[{directive_idx}] must match at least one "
                "forward routed region"
            )
        for uid in matched_uids:
            node = dag.nodes[uid]
            actual_devices = (
                None
                if node.device is None
                else tuple(sorted(int(device) for device in node.device))
            )
            if actual_devices != expected_device_set:
                raise ValueError(
                    f"shard directive[{directive_idx}] requires matched node "
                    f"{uid} to be placed on devices={list(ordered_shard_devices)}, "
                    f"got node_devices={node.device}"
                )
        for fwd_uid in matched_fwd_uids:
            previous = shard_policy_by_fwd_uid.get(fwd_uid)
            policy = (directive_idx, ordered_shard_devices)
            if previous is not None:
                raise ValueError(
                    f"routed region {fwd_uid} matches multiple shard "
                    f"directives: directive[{previous[0]}] and "
                    f"directive[{directive_idx}]"
                )
            shard_policy_by_fwd_uid[fwd_uid] = policy

    sharded_fwd_uids = set(shard_policy_by_fwd_uid)

    def _activation_compute_edges(uid: str, *, incoming: bool) -> list[TrainingDAGEdge]:
        edges = (
            dag.incoming_edges(uid, dep_kind="data")
            if incoming
            else dag.outgoing_edges(uid, dep_kind="data")
        )
        result: list[TrainingDAGEdge] = []
        for edge in edges:
            other_uid = edge.src_uid if incoming else edge.dst_uid
            other = dag.nodes[other_uid]
            if other.node_kind != "COMPUTE":
                continue
            if dag.nodes[uid].compute_subkind == "FWD":
                if other.compute_subkind == "FWD":
                    result.append(edge)
            elif _is_backward_activation_subkind(dag.nodes[uid].compute_subkind):
                if _is_backward_activation_subkind(other.compute_subkind):
                    result.append(edge)
        return result

    def _device_key(node: TrainingDAGNode) -> tuple[int, ...]:
        if node.device is None:
            raise ValueError(f"routed-region endpoint {node.uid} is unplaced")
        return tuple(sorted(int(device) for device in node.device))

    def _require_replicated_dense(node: TrainingDAGNode, *, routed_uid: str) -> None:
        devices = _device_key(node)
        # A singleton ActorGroup is trivially replicated.  Wider dense groups
        # need the replicate directive that synchronizes their dense weights.
        if len(devices) > 1 and node.uid not in replicated_compute_uids_by_group.get(
            devices, set()
        ):
            raise ValueError(
                f"routed region {routed_uid} requires adjacent dense region "
                f"{node.uid} on devices={list(devices)} to be covered by a "
                "replicate directive for that exact ActorGroup"
            )

    # Discover routed regions from marker metadata, independently of the
    # schedule's shard filters.  Marker processing has already established
    # that dispatch/combine boundaries bracket one adjacent FX segment; this
    # second check protects that ownership after DAG rewrites such as
    # microbatch expansion and bucketing.
    marker_routed_fwd_uids: set[str] = set()
    explicit_combine_fwd_uids: set[str] = set()
    for node in dag.nodes.values():
        if node.node_kind != "COMPUTE" or node.compute_subkind != "FWD":
            continue
        boundary = node.node_meta.get("a2a_boundary_after")
        if not isinstance(boundary, dict) or boundary.get("tensor_idx") is None:
            continue
        is_dispatch = (
            boundary.get("routed_boundary") == "dispatch"
            or boundary.get("routing_counts_idx") is not None
        )
        if is_dispatch:
            outgoing = _activation_compute_edges(node.uid, incoming=False)
            if len(outgoing) != 1:
                raise ValueError(
                    f"routed dispatch boundary on {node.uid} must feed exactly "
                    "one adjacent forward region; got "
                    f"outgoing={[edge.dst_uid for edge in outgoing]}"
                )
            marker_routed_fwd_uids.add(outgoing[0].dst_uid)
        if boundary.get("routed_boundary") == "combine":
            explicit_combine_fwd_uids.add(node.uid)

    # Old serialized DAGs did not persist ``routed_boundary='combine'`` and
    # represented a combine marker only as ``tensor_idx`` on the immediate
    # dispatch target.  Accept that local legacy encoding without treating
    # every ordinary tensor boundary in such a DAG as an orphan combine.
    combine_marker_fwd_uids = set(explicit_combine_fwd_uids)
    for routed_uid in marker_routed_fwd_uids:
        boundary = dag.nodes[routed_uid].node_meta.get("a2a_boundary_after")
        if (
            isinstance(boundary, dict)
            and boundary.get("tensor_idx") is not None
            and boundary.get("routing_counts_idx") is None
        ):
            combine_marker_fwd_uids.add(routed_uid)

    if marker_routed_fwd_uids != combine_marker_fwd_uids:
        missing_combine = sorted(
            marker_routed_fwd_uids - combine_marker_fwd_uids
        )
        missing_dispatch = sorted(
            combine_marker_fwd_uids - marker_routed_fwd_uids
        )
        raise ValueError(
            "routed marker triplets were not preserved through DAG rewrites; "
            f"missing_combine={missing_combine}, "
            f"missing_dispatch={missing_dispatch}"
        )

    if marker_routed_fwd_uids != sharded_fwd_uids:
        missing_shard = sorted(marker_routed_fwd_uids - sharded_fwd_uids)
        unmarked_shard = sorted(sharded_fwd_uids - marker_routed_fwd_uids)
        raise ValueError(
            "routed marker regions and shard directives must have exact "
            "forward-region ownership; "
            f"marker regions missing a shard directive={missing_shard}, "
            f"shard-matched regions without a marker triplet={unmarked_shard}"
        )

    if not marker_routed_fwd_uids:
        return

    for fwd_uid, (directive_idx, routed_devices) in shard_policy_by_fwd_uid.items():
        fwd_node = dag.nodes[fwd_uid]
        incoming = _activation_compute_edges(fwd_uid, incoming=True)
        outgoing = _activation_compute_edges(fwd_uid, incoming=False)
        if len(incoming) != 1 or len(outgoing) != 1:
            raise ValueError(
                f"routed region {fwd_uid} must have exactly one forward "
                "dispatch edge and one forward combine edge; got "
                f"incoming={[edge.src_uid for edge in incoming]}, "
                f"outgoing={[edge.dst_uid for edge in outgoing]}"
            )
        dispatch_owner = dag.nodes[incoming[0].src_uid]
        combine_consumer = dag.nodes[outgoing[0].dst_uid]
        if (
            dispatch_owner.uid in sharded_fwd_uids
            or combine_consumer.uid in sharded_fwd_uids
        ):
            raise ValueError(
                f"routed region {fwd_uid} must be isolated between dense "
                "regions; adjacent sharded regions are unsupported "
                f"(dispatch_owner={dispatch_owner.uid}, "
                f"combine_consumer={combine_consumer.uid})"
            )

        dispatch_boundary = dispatch_owner.node_meta.get("a2a_boundary_after")
        combine_boundary = fwd_node.node_meta.get("a2a_boundary_after")
        if not (
            isinstance(dispatch_boundary, dict)
            and dispatch_boundary.get("tensor_idx") is not None
            and dispatch_boundary.get("routing_counts_idx") is not None
        ):
            raise ValueError(
                f"routed region {fwd_uid} requires its sole forward "
                f"predecessor {dispatch_owner.uid} to carry one dispatch "
                "payload/count boundary"
            )
        if not (
            isinstance(combine_boundary, dict)
            and combine_boundary.get("tensor_idx") is not None
            and combine_boundary.get("routing_counts_idx") is None
        ):
            raise ValueError(
                f"routed region {fwd_uid} requires one combine payload "
                "boundary on the routed region output"
            )

        _payload_idx, _counts_idx, _combine_idx, route_ownership = (
            _validate_routed_marker_slots(
                dispatch_owner,
                fwd_node,
                shard_devices=routed_devices,
            )
        )
        fwd_node.node_meta.update(route_ownership)
        route_param_specs = _validate_route_owned_parameters(
            fwd_node, route_ownership
        )
        if route_param_specs:
            fwd_node.node_meta["route_sharded_param_specs"] = route_param_specs

        owner_devices = _device_key(dispatch_owner)
        consumer_devices = _device_key(combine_consumer)
        routed_device_set = tuple(sorted(routed_devices))
        _require_replicated_dense(dispatch_owner, routed_uid=fwd_uid)
        _require_replicated_dense(combine_consumer, routed_uid=fwd_uid)

        for role, dense_devices in (
            ("dispatch", owner_devices),
            ("combine", consumer_devices),
        ):
            overlap = set(dense_devices) & set(routed_device_set)
            if dense_devices != routed_device_set and overlap:
                raise ValueError(
                    f"routed region {fwd_uid} has a partially overlapping "
                    f"{role} ActorGroup: routed_devices={list(routed_devices)}, "
                    f"dense_devices={list(dense_devices)}"
                )

        dispatch_crosses_group = owner_devices != routed_device_set
        combine_crosses_group = consumer_devices != routed_device_set
        if dispatch_crosses_group != combine_crosses_group:
            raise ValueError(
                f"routed region {fwd_uid} mixes within-group and cross-group "
                "routing: "
                f"dispatch={list(owner_devices)}->{list(routed_devices)}, "
                f"combine={list(routed_devices)}->{list(consumer_devices)}; "
                "both routed boundaries must use the same communication mode"
            )
        if dispatch_crosses_group:
            _validate_cross_group_routed_interface(dispatch_owner, fwd_node)
        if dispatch_crosses_group and consumer_devices != owner_devices:
            dense_overlap = set(owner_devices) & set(consumer_devices)
            if dense_overlap:
                raise ValueError(
                    f"routed region {fwd_uid} requires dense state-owner and "
                    "consumer ActorGroups to be identical or disjoint; got "
                    f"owner={list(owner_devices)}, "
                    f"consumer={list(consumer_devices)}"
                )
            if len(owner_devices) != len(consumer_devices):
                raise ValueError(
                    f"routed region {fwd_uid} requires equal-width dense "
                    "ActorGroups for its state relay; got "
                    f"owner={list(owner_devices)} (width={len(owner_devices)}), "
                    f"consumer={list(consumer_devices)} "
                    f"(width={len(consumer_devices)})"
                )
            owner_output_names = dispatch_owner.node_meta.get("output_names", ())
            owner_tensor_indices = dispatch_owner.node_meta.get(
                "tensor_output_indices"
            )
            if (
                isinstance(owner_tensor_indices, list)
                and owner_tensor_indices != list(range(len(owner_output_names)))
            ):
                raise ValueError(
                    f"routed region {fwd_uid} relays its retained dense buffer "
                    "between ActorGroups, so every relayed output must be a "
                    f"tensor; output_arity={len(owner_output_names)}, "
                    f"tensor_indices={owner_tensor_indices}"
                )

        matching_bwd = [
            node
            for node in dag.nodes.values()
            if node.node_kind == "COMPUTE"
            and _is_backward_activation_subkind(node.compute_subkind)
            and node.node_meta.get("fwd_uid") == fwd_uid
        ]
        if len(matching_bwd) != 1:
            raise ValueError(
                f"routed region {fwd_uid} must have exactly one backward "
                f"activation region, got {[node.uid for node in matching_bwd]}"
            )
        bwd_node = matching_bwd[0]
        if bwd_node.uid not in shard_matched_uids_by_directive[directive_idx]:
            raise ValueError(
                f"shard directive[{directive_idx}] matches routed forward "
                f"region {fwd_uid} but not its backward activation region "
                f"{bwd_node.uid}; a routed lifecycle must be lowered by one "
                "shard policy"
            )
        if _device_key(bwd_node) != routed_device_set:
            raise ValueError(
                f"routed forward/backward placement mismatch for {fwd_uid}: "
                f"forward={list(routed_devices)}, backward={bwd_node.device}"
            )
        bwd_node.node_meta.update(route_ownership)

        bwd_incoming = _activation_compute_edges(bwd_node.uid, incoming=True)
        bwd_outgoing = _activation_compute_edges(bwd_node.uid, incoming=False)
        if len(bwd_incoming) != 1 or len(bwd_outgoing) != 1:
            raise ValueError(
                f"routed backward region {bwd_node.uid} must have exactly one "
                "reverse-combine edge and one reverse-dispatch edge; got "
                f"incoming={[edge.src_uid for edge in bwd_incoming]}, "
                f"outgoing={[edge.dst_uid for edge in bwd_outgoing]}"
            )
        bwd_in_fwd_uid = dag.nodes[bwd_incoming[0].src_uid].node_meta.get("fwd_uid")
        bwd_out_fwd_uid = dag.nodes[bwd_outgoing[0].dst_uid].node_meta.get("fwd_uid")
        if (
            bwd_in_fwd_uid != combine_consumer.uid
            or bwd_out_fwd_uid != dispatch_owner.uid
        ):
            raise ValueError(
                f"routed lifecycle for {fwd_uid} is not the reverse of its "
                "forward triplet: expected backward "
                f"{combine_consumer.uid}->{fwd_uid}->{dispatch_owner.uid}, "
                f"got {bwd_in_fwd_uid}->{fwd_uid}->{bwd_out_fwd_uid}"
            )


def _insert_cross_group_routed_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    """Lower routed edges between disjoint replicated and sharded groups.

    A collective spanning two actor groups is represented by two local DAG
    nodes with a shared ``comm_id``.  This is the same structural convention
    used by SEND/RECV: the source node remains connected only to source-group
    compute and the destination node only to destination-group compute.  It
    lets actor-group DAGs remain device-homogeneous while both sides retain all
    metadata needed to enter the same variable-size collective.

    Identical device sets are deliberately left untouched for the existing
    ragged A2A lowering.  Partially overlapping sets are invalid because they
    cannot describe two consistent SPMD actor groups.
    """
    expected = sorted(int(d) for d in devices)
    device_to_pp_rank = _device_to_physical_pp_rank(dag)
    matched_uids = [
        uid
        for uid in _matching_uids(dag, filters)
        if dag.nodes[uid].node_kind == "COMPUTE"
    ]
    for uid in matched_uids:
        node = dag.nodes[uid]
        if node.device is None or sorted(int(d) for d in node.device) != expected:
            raise ValueError(
                f"shard requires matched node {uid} to have devices={devices}, "
                f"got node_devices={node.device}"
            )

    # The source dense group owns both the passthrough activation buffer and
    # the dispatch permutation/splits that an eventual combine must reverse.
    # Record that owner per sharded forward node before replacing any edges.
    state_owner_by_expert_fwd_uid: dict[str, TrainingDAGNode] = {}
    for uid in matched_uids:
        node = dag.nodes[uid]
        if node.compute_subkind != "FWD":
            continue
        owners: list[TrainingDAGNode] = []
        for edge in dag.incoming_edges(uid, dep_kind="data"):
            source = dag.nodes[edge.src_uid]
            if source.node_kind != "COMPUTE" or source.compute_subkind != "FWD":
                continue
            boundary = source.node_meta.get("a2a_boundary_after")
            if (
                isinstance(boundary, dict)
                and boundary.get("routing_counts_idx") is not None
            ):
                owners.append(source)
        if len(owners) > 1:
            raise ValueError(
                f"Sharded node {uid} has multiple routed state owners: "
                f"{[owner.uid for owner in owners]}"
            )
        if owners:
            state_owner = owners[0]
            if (
                state_owner.device is not None
                and node.device is not None
                and sorted(int(device) for device in state_owner.device)
                == sorted(int(device) for device in node.device)
            ):
                # The existing within-group ragged A2A path retains its own
                # buffer contract. These checks describe only the disjoint
                # cross-group M-to-N/N-to-M lowering below.
                continue
            # Schedule-level lowering normally ran this before any edges were
            # replaced. Keep the same pure check here for direct/internal
            # callers as a defense in depth.
            _validate_cross_group_routed_interface(state_owner, node)
            state_owner_by_expert_fwd_uid[uid] = state_owner

    edges_to_replace: list[TrainingDAGEdge] = []
    replaced_keys: set[tuple[str, str, str, str | None]] = set()
    incoming_state_source_uid_by_sharded_uid: dict[str, str] = {}

    def _add_paired_comm_nodes(
        *,
        node_kind: str,
        prefix: str,
        source_devices: list[int],
        destination_devices: list[int],
        source_tag: dict[str, Any],
        destination_tag: dict[str, Any],
        common_meta: dict[str, Any],
    ) -> tuple[str, str]:
        source_uid = _reserve_comm_uid(
            dag,
            node_kind=node_kind,
            prefix=prefix,
        )
        destination_uid = _reserve_comm_uid(
            dag,
            node_kind=node_kind,
            prefix=prefix,
        )
        stream = comm_stream if comm_stream is not None else _DEFAULT_STREAM
        dag.add_node(
            TrainingDAGNode(
                uid=source_uid,
                node_kind=node_kind,
                compute_subkind=None,
                tag=dict(source_tag),
                device=list(source_devices),
                stream=stream,
                node_meta={**common_meta, "role": "source"},
            )
        )
        dag.add_node(
            TrainingDAGNode(
                uid=destination_uid,
                node_kind=node_kind,
                compute_subkind=None,
                tag=dict(destination_tag),
                device=list(destination_devices),
                stream=stream,
                node_meta={**common_meta, "role": "destination"},
            )
        )
        return source_uid, destination_uid

    def _add_dense_state_relay(
        *,
        source_devices: list[int],
        destination_devices: list[int],
        source_tag: dict[str, Any],
        destination_tag: dict[str, Any],
        expert_fwd_uid: str,
        state_owner: TrainingDAGNode,
        expert_fwd_node: TrainingDAGNode,
        direction: str,
        edge: TrainingDAGEdge,
        bucket_key: Any,
        source_bucket_key: Any,
        destination_bucket_key: Any,
    ) -> tuple[str, str]:
        if len(source_devices) != len(destination_devices):
            raise ValueError(
                "dense routed-state relay currently requires equal replicated "
                f"degrees, got source={source_devices} and "
                f"destination={destination_devices}"
            )
        if set(source_devices) & set(destination_devices):
            raise ValueError(
                "dense routed-state relay requires disjoint actor groups, got "
                f"source={source_devices}, destination={destination_devices}"
            )
        participants = sorted((*source_devices, *destination_devices))
        payload_arity = len(expert_fwd_node.node_meta.get("output_names", ()))
        forward_tensor_indices = expert_fwd_node.node_meta.get(
            "tensor_output_indices"
        )
        backward_tensor_indices = expert_fwd_node.node_meta.get(
            "differentiable_output_indices"
        )
        if (
            not isinstance(forward_tensor_indices, list)
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in forward_tensor_indices
            )
            or sorted(set(forward_tensor_indices)) != forward_tensor_indices
            or any(index < 0 or index >= payload_arity for index in forward_tensor_indices)
            or len(forward_tensor_indices) != payload_arity
        ):
            raise ValueError(
                f"dense relay for {expert_fwd_uid} cannot prove all "
                f"payload_arity={payload_arity} forward slots are tensors: "
                f"tensor_output_indices={forward_tensor_indices!r}"
            )
        if (
            not isinstance(backward_tensor_indices, list)
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in backward_tensor_indices
            )
            or sorted(set(backward_tensor_indices)) != backward_tensor_indices
            or any(
                index < 0 or index >= payload_arity
                for index in backward_tensor_indices
            )
            or not set(backward_tensor_indices).issubset(forward_tensor_indices)
        ):
            raise ValueError(
                f"dense relay for {expert_fwd_uid} has invalid differentiable "
                f"output indices={backward_tensor_indices!r} for "
                f"payload_arity={payload_arity}"
            )
        relay_comm_id = (
            f"dense_relay:{direction}:{edge.src_uid}->{edge.dst_uid}:"
            f"{edge.tensor_name or '-'}:{expert_fwd_uid}"
        )
        source_key = tuple(source_devices)
        destination_key = tuple(destination_devices)
        source_pp_rank = device_to_pp_rank[source_key]
        destination_pp_rank = device_to_pp_rank[destination_key]
        comm_group_kind = (
            "pp_lo_hi" if source_pp_rank < destination_pp_rank else "pp_hi_lo"
        )
        rank_pairs = [
            [source_rank, destination_rank]
            for source_rank, destination_rank in zip(
                source_devices, destination_devices
            )
        ]
        relay_meta = {
            "dense_relay": True,
            "direction": direction,
            "source_devices": list(source_devices),
            "destination_devices": list(destination_devices),
            "source_degree": len(source_devices),
            "destination_degree": len(destination_devices),
            "participants": participants,
            "expert_fwd_uid": expert_fwd_uid,
            "payload_arity": payload_arity,
            "forward_tensor_indices": list(forward_tensor_indices),
            "backward_tensor_indices": list(backward_tensor_indices),
            "bucket_key": bucket_key,
            "state_owner_bucket_key": state_owner.node_meta.get("bucket_key"),
            "source_bucket_key": source_bucket_key,
            "destination_bucket_key": destination_bucket_key,
            "shape_reference_fwd_uid": state_owner.uid,
            "shape_reference_bucket_key": state_owner.node_meta.get("bucket_key"),
            "comm_id": relay_comm_id,
            "comm_group_kind": comm_group_kind,
            "relay_mode": "ordinal_one_to_one",
            "rank_pairs": rank_pairs,
            "from_uid": edge.src_uid,
            "to_uid": edge.dst_uid,
        }
        send_uid = _reserve_comm_uid(
            dag,
            node_kind="SEND_COMM",
            prefix="send",
        )
        recv_uid = _reserve_comm_uid(
            dag,
            node_kind="RECV_COMM",
            prefix="recv",
        )
        stream = comm_stream if comm_stream is not None else "pp_stream"
        dag.add_node(
            TrainingDAGNode(
                uid=send_uid,
                node_kind="SEND_COMM",
                compute_subkind=None,
                tag=dict(source_tag),
                device=list(source_devices),
                stream=stream,
                node_meta={
                    **relay_meta,
                    "role": "source",
                    "peer_pp_rank": destination_pp_rank,
                    "peer_devices": list(destination_devices),
                },
            )
        )
        dag.add_node(
            TrainingDAGNode(
                uid=recv_uid,
                node_kind="RECV_COMM",
                compute_subkind=None,
                tag=dict(destination_tag),
                device=list(destination_devices),
                stream=stream,
                node_meta={
                    **relay_meta,
                    "role": "destination",
                    "peer_pp_rank": source_pp_rank,
                    "peer_devices": list(source_devices),
                },
            )
        )
        return send_uid, recv_uid

    for uid in matched_uids:
        if uid not in dag.nodes:
            continue
        expert_node = dag.nodes[uid]
        if expert_node.compute_subkind == "BWD_W":
            continue

        if expert_node.compute_subkind == "FWD":
            expected_subkinds = {"FWD"}
        elif _is_backward_activation_subkind(expert_node.compute_subkind):
            expected_subkinds = {"BWD", "BWD_I"}
        else:
            continue

        incident: list[tuple[str, TrainingDAGEdge]] = []
        incident.extend(
            ("incoming", edge)
            for edge in dag.incoming_edges(uid, dep_kind="data")
        )
        incident.extend(
            ("outgoing", edge)
            for edge in dag.outgoing_edges(uid, dep_kind="data")
        )
        for direction, edge in incident:
            edge_key = (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
            if edge_key in replaced_keys:
                continue
            src_node = dag.nodes[edge.src_uid]
            dst_node = dag.nodes[edge.dst_uid]
            if src_node.node_kind != "COMPUTE" or dst_node.node_kind != "COMPUTE":
                continue
            if (
                src_node.compute_subkind not in expected_subkinds
                or dst_node.compute_subkind not in expected_subkinds
            ):
                continue
            if src_node.tag.get("PASS") != dst_node.tag.get("PASS"):
                raise ValueError(
                    "routed communication requires matching compute pass tags; "
                    f"edge {src_node.uid}->{dst_node.uid} has "
                    f"src.PASS={src_node.tag.get('PASS')} "
                    f"dst.PASS={dst_node.tag.get('PASS')}"
                )
            if src_node.device is None or dst_node.device is None:
                raise ValueError(
                    f"cannot lower routed edge {src_node.uid}->{dst_node.uid} "
                    "with an unplaced endpoint"
                )

            src_devices = sorted(int(d) for d in src_node.device)
            dst_devices = sorted(int(d) for d in dst_node.device)
            if src_devices == dst_devices:
                # replicate(N) <-> shard(N): handled as ragged A2A below.
                continue
            overlap = set(src_devices) & set(dst_devices)
            if overlap:
                raise ValueError(
                    f"routed edge {src_node.uid}->{dst_node.uid} has partially "
                    f"overlapping device sets {src_devices} and {dst_devices}; "
                    "actor-group device sets must be identical or disjoint"
                )

            boundary_info = _routed_boundary_for_edge(dag, src_node, dst_node)
            tensor_idx = int(boundary_info["tensor_idx"])
            expert_fwd_uid = _routed_expert_fwd_uid(expert_node)
            expert_fwd_node = dag.nodes.get(expert_fwd_uid)
            if expert_fwd_node is None:
                raise ValueError(
                    f"routed edge {src_node.uid}->{dst_node.uid} references "
                    f"missing expert forward node {expert_fwd_uid}"
                )
            state_owner = state_owner_by_expert_fwd_uid.get(expert_fwd_uid)
            if state_owner is None:
                raise ValueError(
                    f"routed edge {src_node.uid}->{dst_node.uid} cannot resolve "
                    f"the dense state owner for {expert_fwd_uid}"
                )
            if state_owner.device is None:
                raise ValueError(
                    f"routed state owner {state_owner.uid} is unplaced"
                )
            state_owner_devices = sorted(int(d) for d in state_owner.device)

            relay_direction: str | None = None
            effective_src_devices = src_devices
            effective_dst_devices = dst_devices
            source_tag = dict(src_node.tag)
            destination_tag = dict(dst_node.tag)
            if (
                expert_node.compute_subkind == "FWD"
                and direction == "outgoing"
                and dst_devices != state_owner_devices
            ):
                # Combine back to the group that owns the dispatch layout and
                # passthrough buffer, then relay the patched dense buffer to the
                # next dense pipeline group.
                relay_direction = "forward"
                effective_dst_devices = state_owner_devices
                destination_tag = dict(state_owner.tag)
            elif (
                _is_backward_activation_subkind(expert_node.compute_subkind)
                and direction == "incoming"
                and src_devices != state_owner_devices
            ):
                # Reverse the forward relay before applying the gradient of the
                # forward combine on the original state-owning dense group.
                relay_direction = "backward"
                effective_src_devices = state_owner_devices
                source_tag = {
                    **state_owner.tag,
                    "PASS": expert_node.tag.get("PASS"),
                }

            node_kind = (
                "M_TO_N_COMM" if direction == "incoming" else "N_TO_M_COMM"
            )
            prefix = "m_to_n" if direction == "incoming" else "n_to_m"
            comm_id = (
                f"{prefix}:{src_node.uid}->{dst_node.uid}:"
                f"{edge.tensor_name or '-'}:{tensor_idx}"
            )
            participants = sorted((*effective_src_devices, *effective_dst_devices))
            common_meta: dict[str, Any] = {
                "direction": direction,
                "source_devices": effective_src_devices,
                "destination_devices": effective_dst_devices,
                "source_degree": len(effective_src_devices),
                "destination_degree": len(effective_dst_devices),
                "participants": participants,
                "expert_fwd_uid": expert_fwd_uid,
                "expert_input_arity": len(
                    _routed_activation_input_names(expert_fwd_node)
                ),
                "expert_output_arity": len(
                    expert_fwd_node.node_meta.get("output_names", ())
                ),
                "a2a_tensor_idx": tensor_idx,
                "bucket_key": expert_node.node_meta.get(
                    "bucket_key",
                    src_node.node_meta.get("bucket_key"),
                ),
                "comm_id": comm_id,
                "comm_group_kind": "routed_cross_group",
            }
            common_meta.update(_route_ownership_meta(expert_fwd_node))
            if direction == "incoming" and expert_node.compute_subkind == "FWD":
                routing_counts_idx = boundary_info.get("routing_counts_idx")
                if routing_counts_idx is None:
                    raise ValueError(
                        f"Expert node {expert_node.uid} requires a routing-count "
                        "tensor on its incoming M_TO_N boundary"
                    )
                common_meta["routing_counts_idx"] = int(routing_counts_idx)

            source_uid, destination_uid = _add_paired_comm_nodes(
                node_kind=node_kind,
                prefix=prefix,
                source_devices=effective_src_devices,
                destination_devices=effective_dst_devices,
                source_tag=source_tag,
                destination_tag=destination_tag,
                common_meta=common_meta,
            )
            if direction == "incoming":
                incoming_state_source_uid_by_sharded_uid[uid] = source_uid
            else:
                incoming_source_uid = incoming_state_source_uid_by_sharded_uid.get(uid)
                if incoming_source_uid is None:
                    raise ValueError(
                        f"routed outgoing edge {src_node.uid}->{dst_node.uid} "
                        "was lowered before its matching incoming state boundary"
                    )
                # Both endpoints run on the original dense state owner. This
                # expresses the real dispatch-state lifetime, connects the
                # communication-only relay fragment to its actor DAG, and
                # fixes dispatch order without inventing a cross-group edge.
                if dag.nodes[incoming_source_uid].device != dag.nodes[destination_uid].device:
                    raise ValueError(
                        f"routed state endpoints for {expert_fwd_uid} disagree on "
                        f"owner devices: incoming={dag.nodes[incoming_source_uid].device}, "
                        f"outgoing={dag.nodes[destination_uid].device}"
                    )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=incoming_source_uid,
                        dst_uid=destination_uid,
                        dep_kind="temporal",
                        tensor_name=None,
                    )
                )
            if relay_direction == "forward":
                relay_source_uid, relay_destination_uid = _add_dense_state_relay(
                    source_devices=state_owner_devices,
                    destination_devices=dst_devices,
                    source_tag=state_owner.tag,
                    destination_tag=dst_node.tag,
                    expert_fwd_uid=expert_fwd_uid,
                    state_owner=state_owner,
                    expert_fwd_node=expert_fwd_node,
                    direction="forward",
                    edge=edge,
                    bucket_key=common_meta["bucket_key"],
                    source_bucket_key=state_owner.node_meta.get("bucket_key"),
                    destination_bucket_key=dst_node.node_meta.get("bucket_key"),
                )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=src_node.uid,
                        dst_uid=source_uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=destination_uid,
                        dst_uid=relay_source_uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=relay_destination_uid,
                        dst_uid=dst_node.uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
            elif relay_direction == "backward":
                relay_source_uid, relay_destination_uid = _add_dense_state_relay(
                    source_devices=src_devices,
                    destination_devices=state_owner_devices,
                    source_tag=src_node.tag,
                    destination_tag=source_tag,
                    expert_fwd_uid=expert_fwd_uid,
                    state_owner=state_owner,
                    expert_fwd_node=expert_fwd_node,
                    direction="backward",
                    edge=edge,
                    bucket_key=common_meta["bucket_key"],
                    source_bucket_key=src_node.node_meta.get("bucket_key"),
                    destination_bucket_key=state_owner.node_meta.get("bucket_key"),
                )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=src_node.uid,
                        dst_uid=relay_source_uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=relay_destination_uid,
                        dst_uid=source_uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=destination_uid,
                        dst_uid=dst_node.uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
            else:
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=src_node.uid,
                        dst_uid=source_uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=destination_uid,
                        dst_uid=dst_node.uid,
                        dep_kind="data",
                        tensor_name=edge.tensor_name,
                    )
                )
            edges_to_replace.append(edge)
            replaced_keys.add(edge_key)

    _remove_edges(dag, edges_to_replace)


def _insert_shard_a2a_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    # Keep direct/internal callers consistent with the schedule-level lowering.
    # The schedule path invokes this once earlier as well, before generic P2P
    # can erase the routed COMPUTE->COMPUTE boundary; the second call is a
    # no-op because the cross-group edge has already been replaced.
    _insert_cross_group_routed_comm_nodes(
        dag,
        filters,
        devices,
        comm_stream=comm_stream,
    )
    expected = sorted(int(d) for d in devices)

    matched_uids = [
        uid
        for uid in _matching_uids(dag, filters)
        if dag.nodes[uid].node_kind == "COMPUTE"
    ]
    compute_edges_to_delete: list[TrainingDAGEdge] = []
    compute_edge_keys_to_delete: set[
        tuple[str, str, str, str | None]
    ] = set()

    def _a2a_boundary_for_edge(src_node: TrainingDAGNode, dst_node: TrainingDAGNode) -> dict[str, Any]:
        # Backward data edges are reversed from the forward graph:
        #     FWD: B -> A    becomes    BWD: A.bwd -> B.bwd
        # The A2A tensor position is defined by the original forward producer,
        # which is the fwd node paired with the backward edge destination.
        if (
            _is_backward_activation_subkind(src_node.compute_subkind)
            and _is_backward_activation_subkind(dst_node.compute_subkind)
        ):
            fwd_producer_uid = dst_node.node_meta.get("fwd_uid")
            if not isinstance(fwd_producer_uid, str) or fwd_producer_uid not in dag.nodes:
                raise ValueError(
                    f"A2A_COMM could not resolve forward producer for BWD edge "
                    f"{src_node.uid}->{dst_node.uid}: fwd_uid={fwd_producer_uid!r}"
                )
            producer = dag.nodes[fwd_producer_uid]
        else:
            producer = src_node

        binfo = producer.node_meta.get("a2a_boundary_after")
        if not isinstance(binfo, dict) or binfo.get("tensor_idx") is None:
            raise ValueError(
                f"A2A_COMM missing tensor_idx for edge {src_node.uid}->{dst_node.uid}; "
                f"producer={producer.uid} boundary_info={binfo!r}"
            )
        return binfo

    def _expert_fwd_uid(compute_node: TrainingDAGNode) -> str:
        if compute_node.compute_subkind == "FWD":
            return compute_node.uid
        fwd_uid = compute_node.node_meta.get("fwd_uid")
        if not isinstance(fwd_uid, str):
            raise ValueError(
                f"Expert backward node {compute_node.uid} is missing fwd_uid"
            )
        return fwd_uid

    # Routed A2A is the expert-parallel contract. The forward incoming edge
    # supplies token counts; the outgoing and backward nodes reuse the
    # dispatch state keyed by the expert forward node.
    routing_counts_by_fwd_uid: dict[str, int] = {}
    for matched_uid in matched_uids:
        matched_node = dag.nodes[matched_uid]
        if matched_node.compute_subkind != "FWD":
            continue
        routing_count_indices: list[int] = []
        local_incoming_compute_edges = 0
        for edge in list(dag.edges):
            if edge.dep_kind != "data" or edge.dst_uid != matched_uid:
                continue
            src_node = dag.nodes[edge.src_uid]
            if src_node.node_kind != "COMPUTE" or src_node.compute_subkind != "FWD":
                continue
            if src_node.device != matched_node.device:
                continue
            local_incoming_compute_edges += 1
            boundary_info = src_node.node_meta.get("a2a_boundary_after")
            if (
                isinstance(boundary_info, dict)
                and boundary_info.get("routing_counts_idx") is not None
            ):
                routing_count_indices.append(
                    int(boundary_info["routing_counts_idx"])
                )
        if len(routing_count_indices) > 1:
            raise ValueError(
                f"Expert node {matched_uid} has multiple routing-count "
                f"boundaries: {routing_count_indices}"
            )
        if local_incoming_compute_edges and not routing_count_indices:
            raise ValueError(
                f"Expert node {matched_uid} requires a routing-count tensor on "
                "its incoming A2A boundary"
            )
        if routing_count_indices:
            routing_counts_by_fwd_uid[matched_uid] = routing_count_indices[0]

    for uid in matched_uids:
        if uid not in dag.nodes:
            continue
        node = dag.nodes[uid]
        if node.device is None or sorted(int(d) for d in node.device) != expected:
            raise ValueError(
                f"shard requires matched node {uid} to have devices={devices}, got node_devices={node.device}"
            )

        if node.compute_subkind == "BWD_W":
            continue

        def _is_a2a_compute_edge(e: TrainingDAGEdge) -> bool:
            src = dag.nodes[e.src_uid]
            dst = dag.nodes[e.dst_uid]
            if src.node_kind != "COMPUTE" or dst.node_kind != "COMPUTE":
                return False
            if node.compute_subkind == "FWD":
                return src.compute_subkind == "FWD" and dst.compute_subkind == "FWD"
            if _is_backward_activation_subkind(node.compute_subkind):
                return (
                    _is_backward_activation_subkind(src.compute_subkind)
                    and _is_backward_activation_subkind(dst.compute_subkind)
                )
            return False

        incoming = [
            e for e in dag.incoming_edges(uid, dep_kind="data")
            if _is_a2a_compute_edge(e)
            and (e.src_uid, e.dst_uid, e.dep_kind, e.tensor_name)
            not in compute_edge_keys_to_delete
        ]
        outgoing = [
            e for e in dag.outgoing_edges(uid, dep_kind="data")
            if _is_a2a_compute_edge(e)
            and (e.src_uid, e.dst_uid, e.dep_kind, e.tensor_name)
            not in compute_edge_keys_to_delete
        ]

        for e in incoming:
            src = dag.nodes[e.src_uid]
            if src.device is None or sorted(int(d) for d in src.device) != expected:
                raise ValueError(
                    f"shard precondition failed for incoming edge {e.src_uid}->{uid}: "
                    f"upstream compute must be replicated on devices={devices}, got {src.device}"
                )
            if src.tag.get("PASS") != node.tag.get("PASS"):
                raise ValueError(
                    f"A2A_COMM requires matching compute pass tags; "
                    f"incoming edge {e.src_uid}->{uid} has src.PASS={src.tag.get('PASS')} "
                    f"dst.PASS={node.tag.get('PASS')}"
                )
        for e in outgoing:
            dst = dag.nodes[e.dst_uid]
            if dst.device is None or sorted(int(d) for d in dst.device) != expected:
                raise ValueError(
                    f"shard precondition failed for outgoing edge {uid}->{e.dst_uid}: "
                    f"downstream compute must be replicated on devices={devices}, got {dst.device}"
                )
            if dst.tag.get("PASS") != node.tag.get("PASS"):
                raise ValueError(
                    f"A2A_COMM requires matching compute pass tags; "
                    f"outgoing edge {uid}->{e.dst_uid} has src.PASS={node.tag.get('PASS')} "
                    f"dst.PASS={dst.tag.get('PASS')}"
                )

        # Logically remove these edges now so a later matched endpoint does not
        # create a duplicate A2A. Compact the canonical edge list once after all
        # matched nodes have appended their legacy-ordered replacements.
        for edge in incoming + outgoing:
            compute_edges_to_delete.append(edge)
            compute_edge_keys_to_delete.add(
                (edge.src_uid, edge.dst_uid, edge.dep_kind, edge.tensor_name)
            )

        for e in incoming:
            src_node = dag.nodes[e.src_uid]
            binfo = _a2a_boundary_for_edge(src_node, node)
            a2a_tensor_idx = binfo["tensor_idx"]
            expert_fwd_uid = _expert_fwd_uid(node)
            comm_uid = _reserve_comm_uid(
                dag,
                node_kind="A2A_COMM",
                prefix="a2a",
            )
            comm_meta = {
                "direction": "incoming",
                "target_uid": uid,
                "expert_fwd_uid": expert_fwd_uid,
                "a2a_tensor_idx": a2a_tensor_idx,
                "bucket_key": node.node_meta.get("bucket_key", src_node.node_meta.get("bucket_key")),
                "comm_id": f"a2a:{e.src_uid}->{uid}:{a2a_tensor_idx}",
                "comm_group_kind": "ep",
                "participants": sorted(int(rank) for rank in node.device),
            }
            comm_meta.update(_route_ownership_meta(dag.nodes[expert_fwd_uid]))
            if node.compute_subkind == "FWD":
                comm_meta["routing_counts_idx"] = routing_counts_by_fwd_uid[
                    expert_fwd_uid
                ]
            comm_node = TrainingDAGNode(
                uid=comm_uid,
                node_kind="A2A_COMM",
                compute_subkind=None,
                tag=dict(node.tag),
                device=list(node.device) if node.device is not None else None,
                stream=comm_stream if comm_stream is not None else "default_stream",
                node_meta=comm_meta,
            )
            dag.add_node(comm_node)
            dag.add_edge(TrainingDAGEdge(src_uid=e.src_uid, dst_uid=comm_uid, dep_kind="data", tensor_name=e.tensor_name))
            dag.add_edge(TrainingDAGEdge(src_uid=comm_uid, dst_uid=uid, dep_kind="data", tensor_name=e.tensor_name))

        for e in outgoing:
            dst_node = dag.nodes[e.dst_uid]
            binfo = _a2a_boundary_for_edge(node, dst_node)
            a2a_tensor_idx = binfo["tensor_idx"]
            expert_fwd_uid = _expert_fwd_uid(node)
            comm_uid = _reserve_comm_uid(
                dag,
                node_kind="A2A_COMM",
                prefix="a2a",
            )
            comm_node = TrainingDAGNode(
                uid=comm_uid,
                node_kind="A2A_COMM",
                compute_subkind=None,
                tag=dict(node.tag),
                device=list(node.device) if node.device is not None else None,
                stream=comm_stream if comm_stream is not None else "default_stream",
                node_meta={
                    "direction": "outgoing",
                    "source_uid": uid,
                    "expert_fwd_uid": expert_fwd_uid,
                    "a2a_tensor_idx": a2a_tensor_idx,
                    "bucket_key": node.node_meta.get("bucket_key", dst_node.node_meta.get("bucket_key")),
                    "comm_id": f"a2a:{uid}->{e.dst_uid}:{a2a_tensor_idx}",
                    "comm_group_kind": "ep",
                    "participants": sorted(int(rank) for rank in node.device),
                    **_route_ownership_meta(dag.nodes[expert_fwd_uid]),
                },
            )
            dag.add_node(comm_node)
            dag.add_edge(TrainingDAGEdge(src_uid=uid, dst_uid=comm_uid, dep_kind="data", tensor_name=e.tensor_name))
            dag.add_edge(TrainingDAGEdge(src_uid=comm_uid, dst_uid=e.dst_uid, dep_kind="data", tensor_name=e.tensor_name))

    _remove_edges(dag, compute_edges_to_delete)


def _apply_split_directive(
    dag: TrainingDAG,
    flt: dict[str, Any],
    dim_name: str,
    num_microbatches: int,
) -> None:
    matched = set(
        dag.matching_uids(
            flt,
            wildcard_value=_ANY_TAG_INDEX,
            absent_value=_ABSENT_TAG_INDEX,
        )
    )
    if not matched:
        raise ValueError(f"split(filter={flt}) matched zero nodes")

    # UPD nodes are global reduction/update sinks and must not be duplicated
    # by microbatch split; duplicated predecessors should continue to target
    # the same UPD node(s).
    matched = {uid for uid in matched if dag.nodes[uid].node_kind != "UPD"}
    if not matched:
        raise ValueError(f"split(filter={flt}) matched no splittable non-UPD nodes")

    topo = _topological_order(dag)
    idx = {u: i for i, u in enumerate(topo)}
    mpos = sorted(idx[u] for u in matched)
    lo, hi = mpos[0], mpos[-1]
    interleaved = [
        u for u in topo[lo:hi + 1]
        if u not in matched and dag.nodes[u].node_kind != "UPD"
    ]
    if interleaved:
        raise ValueError(
            f"split requires a contiguous sub-DAG; found interleaved non-matching nodes: {interleaved[:8]}"
        )

    incoming_boundary = [e for e in list(dag.edges) if e.dst_uid in matched and e.src_uid not in matched]
    outgoing_boundary = [e for e in list(dag.edges) if e.src_uid in matched and e.dst_uid not in matched]

    sources = {
        u for u in matched
        if not any(pred in matched for pred in dag.preds.get(u, set()))
    }
    sinks = {
        u for u in matched
        if not any(succ in matched for succ in dag.succs.get(u, set()))
    }

    # Ensure outside edges only touch source/sink boundaries.
    for e in incoming_boundary:
        if dag.nodes[e.src_uid].node_kind == "UPD" or dag.nodes[e.dst_uid].node_kind == "UPD":
            continue
        if e.dst_uid not in sources:
            raise ValueError(
                f"split boundary violation: incoming outside edge targets non-source node {e.dst_uid}"
            )
    for e in outgoing_boundary:
        if dag.nodes[e.src_uid].node_kind == "UPD" or dag.nodes[e.dst_uid].node_kind == "UPD":
            continue
        if e.src_uid not in sinks:
            raise ValueError(
                f"split boundary violation: outgoing outside edge starts at non-sink node {e.src_uid}"
            )

    inside_edges = [e for e in list(dag.edges) if e.src_uid in matched and e.dst_uid in matched]

    # Tag originals as microbatch 0.
    for uid in matched:
        node = dag.nodes[uid]
        updated_tag = dict(node.tag)
        updated_tag[dim_name] = 0
        dag.update_node_tag(uid, updated_tag)
        # Split copies of one logical compute bucket must share runtime
        # parameters and gradient storage even when a synthetic/input DAG did
        # not provide the builder's usual bucket_key metadata.
        node.node_meta.setdefault("bucket_key", uid)

    # Create copies for microbatches 1..N-1.
    copy_uid: dict[tuple[str, int], str] = {}
    for mb in range(1, num_microbatches):
        for uid in topo:
            if uid not in matched:
                continue
            nu = f"{uid}.split{dim_name}{mb}"
            copy_uid[(uid, mb)] = nu

    uid_meta_fields = (
        "fwd_uid",
        "expert_fwd_uid",
        "compute_uid",
        "bwd_uid",
        "source_uid",
        "target_uid",
        "src_uid",
        "dst_uid",
        "from_uid",
        "to_uid",
        "shape_reference_fwd_uid",
    )
    for mb in range(1, num_microbatches):
        for uid in topo:
            if uid not in matched:
                continue
            n = dag.nodes[uid]
            nu = copy_uid[(uid, mb)]
            copied_meta = dict(n.node_meta)
            for k in uid_meta_fields:
                v = copied_meta.get(k)
                if isinstance(v, str) and v in matched:
                    copied_meta[k] = copy_uid[(v, mb)]
            if (
                (
                    n.node_kind in {
                        "M_TO_N_COMM",
                        "N_TO_M_COMM",
                    }
                    or (
                        n.node_kind in {"SEND_COMM", "RECV_COMM"}
                        and bool(copied_meta.get("dense_relay"))
                    )
                )
                and isinstance(copied_meta.get("comm_id"), str)
            ):
                copied_meta["comm_id"] = (
                    f"{copied_meta['comm_id']}.split{dim_name}{mb}"
                )
            dag.add_node(
                TrainingDAGNode(
                    uid=nu,
                    node_kind=n.node_kind,
                    compute_subkind=n.compute_subkind,
                    tag={**n.tag, dim_name: mb},
                    device=list(n.device) if n.device is not None else None,
                    stream=n.stream,
                    node_meta=copied_meta,
                    profile_times_by_mode={
                        mode: dict(profiles)
                        for mode, profiles in n.profile_times_by_mode.items()
                    },
                )
            )

    # Duplicate internal sub-DAG edges for each copied microbatch.
    for mb in range(1, num_microbatches):
        for e in inside_edges:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=copy_uid[(e.src_uid, mb)],
                    dst_uid=copy_uid[(e.dst_uid, mb)],
                    dep_kind=e.dep_kind,
                    tensor_name=e.tensor_name,
                )
            )

    # Duplicate incoming edges onto copied sources.
    for mb in range(1, num_microbatches):
        for e in incoming_boundary:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=e.src_uid,
                    dst_uid=copy_uid[(e.dst_uid, mb)],
                    dep_kind=e.dep_kind,
                    tensor_name=e.tensor_name,
                )
            )

    # Duplicate outgoing edges from copied sinks.
    for mb in range(1, num_microbatches):
        for e in outgoing_boundary:
            dag.add_edge(
                TrainingDAGEdge(
                    src_uid=copy_uid[(e.src_uid, mb)],
                    dst_uid=e.dst_uid,
                    dep_kind=e.dep_kind,
                    tensor_name=e.tensor_name,
                )
            )


def _reachable_uids(
    starts: set[str],
    adjacency: dict[str, set[str]],
    *,
    allowed: set[str] | None = None,
) -> set[str]:
    """Return the multi-source transitive closure over ``adjacency``."""

    seen = set(starts)
    stack = list(starts)
    while stack:
        uid = stack.pop()
        for adjacent_uid in adjacency.get(uid, set()):
            if adjacent_uid in seen:
                continue
            if allowed is not None and adjacent_uid not in allowed:
                continue
            seen.add(adjacent_uid)
            stack.append(adjacent_uid)
    return seen


def _match_set_contiguous_subdag(
    dag: TrainingDAG,
    matched: set[str],
    *,
    context: str,
) -> tuple[set[str], set[str], set[str], set[str]]:
    """Validate an order-filter sub-DAG in linear graph-traversal time.

    A non-matching node is interleaved exactly when it is both reachable from
    the matched set and can reach the matched set.  Computing those two
    multi-source closures replaces the legacy per-node forward/backward DFS.
    Compute boundaries use the same technique within the matched induced
    subgraph: seed immediately after/before every compute node, then traverse
    once to find compute nodes with another compute ancestor/descendant.
    """

    if not matched:
        raise ValueError(f"{context}: filter matched zero nodes")

    interleaved = (
        _reachable_uids(matched, dag.succs)
        & _reachable_uids(matched, dag.preds)
    ) - matched
    if interleaved:
        raise ValueError(
            f"{context}: matched nodes are not contiguous; "
            f"interleaved non-matching nodes: {sorted(interleaved)[:8]}"
        )

    sources = {
        uid
        for uid in matched
        if not any(pred in matched for pred in dag.preds.get(uid, set()))
    }
    sinks = {
        uid
        for uid in matched
        if not any(succ in matched for succ in dag.succs.get(uid, set()))
    }
    if not sources or not sinks:
        raise ValueError(f"{context}: failed to identify non-empty source/sink sets")

    compute_nodes = {
        uid for uid in matched if dag.nodes[uid].node_kind == "COMPUTE"
    }
    downstream_seeds = {
        succ
        for uid in compute_nodes
        for succ in dag.succs.get(uid, set())
        if succ in matched
    }
    compute_with_upstream_compute = (
        _reachable_uids(downstream_seeds, dag.succs, allowed=matched)
        & compute_nodes
    )
    upstream_seeds = {
        pred
        for uid in compute_nodes
        for pred in dag.preds.get(uid, set())
        if pred in matched
    }
    compute_with_downstream_compute = (
        _reachable_uids(upstream_seeds, dag.preds, allowed=matched)
        & compute_nodes
    )
    compute_sources = compute_nodes - compute_with_upstream_compute
    compute_sinks = compute_nodes - compute_with_downstream_compute
    if not compute_sources or not compute_sinks:
        raise ValueError(f"{context}: failed to identify non-empty compute source/sink sets")

    return sources, sinks, compute_sources, compute_sinks


@dataclass(frozen=True)
class _OrderSegment:
    source_uids: set[str]
    sink_uids: set[str]


def _single_device_for_nodes(dag: TrainingDAG, uids: set[str]) -> list[int] | None:
    device_keys = {
        tuple(n.device)
        for uid in uids
        for n in [dag.nodes[uid]]
        if n.device is not None
    }
    if len(device_keys) == 1:
        return list(next(iter(device_keys)))
    return None


def _add_order_dummy_node(
    dag: TrainingDAG,
    *,
    uid: str,
    directive_idx: int,
    group_idx: int,
    role: str,
    grouped_uids: set[str],
) -> str:
    dag.add_node(
        TrainingDAGNode(
            uid=uid,
            node_kind="ORDER_DUMMY",
            compute_subkind=None,
            tag={"order": directive_idx, "group": group_idx},
            device=_single_device_for_nodes(dag, grouped_uids),
            stream=_DEFAULT_STREAM,
            node_meta={"role": role},
        )
    )
    return uid


def _device_key_for_order_edge(dag: TrainingDAG, uid: str) -> tuple[int, ...] | None:
    device = dag.nodes[uid].device
    if device is None:
        return None
    return tuple(sorted(int(d) for d in device))


def _validate_order_edge(
    dag: TrainingDAG,
    src_uid: str,
    dst_uid: str,
    *,
    directive_idx: int,
) -> None:
    if _has_path(dag, dst_uid, src_uid):
        raise ValueError(
            f"order directive[{directive_idx}] violates model dataflow: "
            f"adding temporal edge {src_uid} -> {dst_uid} would create a cycle"
        )

    src_device = _device_key_for_order_edge(dag, src_uid)
    dst_device = _device_key_for_order_edge(dag, dst_uid)
    if src_device is None or dst_device is None:
        raise ValueError(
            f"order directive[{directive_idx}] requires placed nodes with a single device set: "
            f"{src_uid} device={src_device}, {dst_uid} device={dst_device}"
        )
    if set(src_device).isdisjoint(dst_device):
        raise ValueError(
            f"order directive[{directive_idx}] crosses device placement: "
            f"{src_uid} device={src_device}, {dst_uid} device={dst_device}"
        )


def _add_temporal_order_edge(
    dag: TrainingDAG,
    src_uid: str,
    dst_uid: str,
    *,
    directive_idx: int,
) -> None:
    _validate_order_edge(dag, src_uid, dst_uid, directive_idx=directive_idx)
    logger.debug("Adding temporal dependency edge for order directive: %s -> %s", src_uid, dst_uid)
    dag.add_edge(
        TrainingDAGEdge(
            src_uid=src_uid,
            dst_uid=dst_uid,
            dep_kind="temporal",
            tensor_name=None,
        )
    )


def _apply_order_directive(
    dag: TrainingDAG,
    filter_groups: list[list[dict[str, Any]]],
    *,
    directive_idx: int,
) -> None:
    segments: list[_OrderSegment] = []

    for group_idx, group in enumerate(filter_groups):
        group_sources: list[set[str]] = []
        group_sinks: list[set[str]] = []
        grouped_uids: set[str] = set()
        for filter_idx, flt in enumerate(group):
            matched = set(
                dag.matching_uids(
                    flt,
                    wildcard_value=_ANY_TAG_INDEX,
                    absent_value=_ABSENT_TAG_INDEX,
                    exclude_node_kinds={"ORDER_DUMMY"},
                )
            )
            srcs, snks, _csrcs, _csnks = _match_set_contiguous_subdag(
                dag,
                matched,
                context=f"order filter[{group_idx}][{filter_idx}]={flt}",
            )
            has_cross_group_boundary = any(
                dag.nodes[uid].node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
                or (
                    dag.nodes[uid].node_kind in {"SEND_COMM", "RECV_COMM"}
                    and bool(dag.nodes[uid].node_meta.get("dense_relay"))
                )
                for uid in matched
            )
            # Cross-group routed regions include their local communication
            # endpoints in the ordered boundary. This keeps both actors on a
            # collective channel in the same order and connects any dense
            # relay fragment through ordinary order edges. Other schedules
            # retain the existing compute-boundary ordering semantics.
            group_sources.append(srcs if has_cross_group_boundary else _csrcs)
            group_sinks.append(snks if has_cross_group_boundary else _csnks)
            grouped_uids.update(matched)

        if len(group) == 1:
            segments.append(_OrderSegment(source_uids=group_sources[0], sink_uids=group_sinks[0]))
            continue

        src_uid = _add_order_dummy_node(
            dag,
            uid=f"order.{directive_idx}.group{group_idx}.source",
            directive_idx=directive_idx,
            group_idx=group_idx,
            role="source",
            grouped_uids=grouped_uids,
        )
        sink_uid = _add_order_dummy_node(
            dag,
            uid=f"order.{directive_idx}.group{group_idx}.sink",
            directive_idx=directive_idx,
            group_idx=group_idx,
            role="sink",
            grouped_uids=grouped_uids,
        )
        for csrcs in group_sources:
            for v in csrcs:
                _add_temporal_order_edge(dag, src_uid, v, directive_idx=directive_idx)
        for csnks in group_sinks:
            for u in csnks:
                _add_temporal_order_edge(dag, u, sink_uid, directive_idx=directive_idx)
        segments.append(_OrderSegment(source_uids={src_uid}, sink_uids={sink_uid}))

    for i in range(len(segments) - 1):
        for u in segments[i].sink_uids:
            for v in segments[i + 1].source_uids:
                _add_temporal_order_edge(dag, u, v, directive_idx=directive_idx)


def apply_order_directives(
    training_dag: TrainingDAG,
    directives: list[Any],
    *,
    skip_unmatched_directives: bool = False,
) -> None:
    """Apply the order portion of a schedule, optionally to one PP-rank DAG."""

    for directive_idx, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "order":
            continue
        filter_groups = _parse_order_directive(raw)
        if skip_unmatched_directives:
            filter_specs = [
                tag_filter
                for group in filter_groups
                for tag_filter in group
            ]
            if not any(_matching_uids(training_dag, [tag_filter]) for tag_filter in filter_specs):
                continue
        logger.debug(
            "Applying directive[%d]: order(filters=%s)",
            directive_idx,
            filter_groups,
        )
        _apply_order_directive(
            training_dag,
            filter_groups,
            directive_idx=directive_idx,
        )


def _apply_communication_directive(
    training_dag: TrainingDAG,
    raw: dict[str, Any],
    *,
    directive_idx: int,
    bucket_already_lowered: bool,
) -> bool:
    (
        op,
        filters,
        devices,
        stream,
        gather_stream,
        reduce_stream,
        shard_params,
        shard_grads,
        bucket_size,
    ) = _normalize_filter_devices_directive(raw)
    matched_nodes = [
        uid
        for uid in _matching_uids(training_dag, filters)
        if training_dag.nodes[uid].node_kind == "COMPUTE"
    ]
    if not matched_nodes:
        return False
    logger.debug(
        "Applying directive[%d]: %s(filters=%s, devices=%s, stream=%s, "
        "gather_stream=%s, reduce_stream=%s, shard_params=%s, "
        "shard_grads=%s, bucket_size=%s)",
        directive_idx,
        op,
        filters,
        devices,
        stream,
        gather_stream,
        reduce_stream,
        shard_params,
        shard_grads,
        bucket_size,
    )
    if op == "replicate":
        if bucket_size is not None and not bucket_already_lowered:
            _bucket_matched_fwd_nodes(training_dag, filters, int(bucket_size))
        if shard_params:
            _insert_all_gather_comm_nodes(
                training_dag,
                filters,
                devices,
                comm_stream=gather_stream,
            )
            _insert_reduce_scatter_comm_nodes(
                training_dag,
                filters,
                devices,
                comm_stream=reduce_stream,
            )
        elif shard_grads:
            _insert_reduce_scatter_comm_nodes(
                training_dag,
                filters,
                devices,
                comm_stream=reduce_stream,
            )
        else:
            _insert_reduce_comm_nodes(
                training_dag,
                filters,
                devices,
                comm_stream=reduce_stream,
            )
    elif op == "shard":
        _insert_shard_a2a_comm_nodes(
            training_dag,
            filters,
            devices,
            comm_stream=stream,
        )
    else:
        raise ValueError(f"Unsupported directive op after normalization: {op}")
    return True


def apply_communication_directives_to_rank_dags(
    rank_dags: list[TrainingDAG],
    directives: list[Any],
) -> None:
    """Lower ActorGroup-local communication after physical DAG splitting."""

    validate_shard_device_order(directives)
    for rank_dag in rank_dags:
        _validate_replicate_shard_disjoint(rank_dag, directives)
    shared_comm_indices: dict[str, int] = {}
    for rank_dag in rank_dags:
        rank_dag._next_comm_index_by_kind = shared_comm_indices

    for directive_idx, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") in ("place", "split", "order"):
            continue
        matched_rank_count = 0
        for rank_dag in rank_dags:
            if _apply_communication_directive(
                rank_dag,
                raw,
                directive_idx=directive_idx,
                # Bucketing is always part of the pre-split structural phase.
                bucket_already_lowered=True,
            ):
                matched_rank_count += 1
        if matched_rank_count == 0:
            raise ValueError(
                f"communication directive[{directive_idx}] matched zero PP-rank DAGs: {raw}"
            )


def apply_schedule_directives(
    training_dag: TrainingDAG,
    directives: list[Any] | None,
    *,
    defer_order: bool = False,
    defer_communication: bool = False,
) -> None:
    if not directives:
        return
    validate_shard_device_order(directives)
    _validate_replicate_shard_disjoint(training_dag, directives)
    split_backward_keys = _validate_split_backward_order_stencil(directives)

    # Buckets partition the base FX graph and are shared by every microbatch.
    # Lower them before split expansion so bucket_stage runs once per base
    # segment and DualPipeV's later B -> BI/BW stencil operates on the copied
    # bucket nodes. Trying to bucket after that stencil incorrectly assumes
    # each copied FWD still has exactly one fused BWD partner.
    bucketed_replicate_directives: set[int] = set()
    for i, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "replicate":
            continue
        (
            _op,
            filters,
            _devices,
            _stream,
            _gather_stream,
            _reduce_stream,
            _shard_params,
            _shard_grads,
            bucket_size,
        ) = _normalize_filter_devices_directive(raw)
        if bucket_size is not None:
            _bucket_matched_fwd_nodes(training_dag, filters, int(bucket_size))
            bucketed_replicate_directives.add(i)

    # Microbatch expansion must run before split-backward lowering so schedules
    # can mix fused B and split BI/BW for the same bucket on different MBs.
    for i, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "split":
            continue
        flt, dim_name, num_microbatches = _normalize_split_directive(raw)
        logger.debug(
            "Applying directive[%d]: split(filter=%s, dim_name=%s, num_microbatches=%d)",
            i, flt, dim_name, num_microbatches
        )
        _apply_split_directive(training_dag, flt, dim_name, num_microbatches)

    _apply_split_backward_stencil(training_dag, split_backward_keys)
    # Recheck filters that become concrete only after MB expansion or
    # backward splitting.
    _validate_replicate_shard_disjoint(training_dag, directives)
    place_streams: set[str] = set()
    placed_devices_by_compute_uid: dict[str, tuple[int, ...]] = {}
    for i, raw in enumerate(directives):
        if isinstance(raw, dict) and raw.get("op") != "place":
            continue
        op, filters, devices, stream, _gather_stream, _reduce_stream, shard_params, shard_grads, bucket_size = _normalize_filter_devices_directive(raw)
        logger.debug(
            "Applying directive[%d]: %s(filters=%s, devices=%s, stream=%s, shard_params=%s, shard_grads=%s, bucket_size=%s)",
            i, op, filters, devices, stream, shard_params, shard_grads, bucket_size
        )
        matched_nodes = 0
        normalized_devices = tuple(sorted(int(device) for device in devices))
        for uid in _matching_uids(training_dag, filters):
            node = training_dag.nodes[uid]
            if node.node_kind not in ("COMPUTE", "UPD"):
                continue
            if node.node_kind == "COMPUTE":
                previous_devices = placed_devices_by_compute_uid.get(uid)
                if (
                    previous_devices is not None
                    and previous_devices != normalized_devices
                ):
                    raise ValueError(
                        f"COMPUTE node {uid} matches conflicting place "
                        f"directives: devices={list(previous_devices)} and "
                        f"devices={list(normalized_devices)}; repeated place "
                        "matches must use the same normalized device set"
                    )
                placed_devices_by_compute_uid[uid] = normalized_devices
            node.device = list(devices)
            matched_nodes += 1
        if matched_nodes == 0:
            raise ValueError(f"place directive[{i}] matched zero nodes: {raw}")
        if stream is not None:
            place_streams.add(stream)

    unplaced_compute_uids = sorted(
        uid
        for uid, node in training_dag.nodes.items()
        if node.node_kind == "COMPUTE" and node.device is None
    )
    if unplaced_compute_uids:
        raise ValueError(
            "place directives must assign every COMPUTE node; unplaced "
            f"nodes={unplaced_compute_uids[:16]}"
        )

    if place_streams:
        if len(place_streams) != 1:
            raise ValueError(f"place directives must agree on stream, got {sorted(place_streams)}")
        place_stream = next(iter(place_streams))
    else:
        place_stream = None
    _replicate_update_nodes_by_device(training_dag)

    # Placement is the first point at which routed marker triplets can be
    # classified as within-group A2A or cross-group M-to-N/N-to-M.  Validate
    # the complete forward/backward lifecycle before either routed lowering or
    # generic P2P insertion destroys the original COMPUTE->COMPUTE boundaries.
    _validate_placed_routed_regions(training_dag, directives)

    # Routed boundaries must be classified while their marker-described
    # COMPUTE->COMPUTE edges are still present.  In particular, a disjoint
    # replicate(M)<->shard(N) edge is one variable-size collective spanning
    # two actor groups, not a set of lane-wise pipeline SEND/RECV operations.
    # Identical device-set edges are left intact here and lowered to the
    # existing ragged A2A by the normal communication-directive pass below (or
    # by the per-rank pass when communication lowering is deferred).
    for i, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "shard":
            continue
        (
            _op,
            filters,
            devices,
            stream,
            _gather_stream,
            _reduce_stream,
            _shard_params,
            _shard_grads,
            _bucket_size,
        ) = _normalize_filter_devices_directive(raw)
        logger.debug(
            "Classifying directive[%d] cross-group routed boundaries "
            "before generic P2P lowering",
            i,
        )
        _insert_cross_group_routed_comm_nodes(
            training_dag,
            filters,
            devices,
            comm_stream=stream,
        )

    _insert_send_recv_comm_nodes(training_dag, comm_stream=place_stream)

    if defer_communication:
        return

    for i, raw in enumerate(directives):
        if isinstance(raw, dict) and raw.get("op") in ("place", "split", "order"):
            continue
        if not _apply_communication_directive(
            training_dag,
            raw,
            directive_idx=i,
            bucket_already_lowered=i in bucketed_replicate_directives,
        ):
            raise ValueError(f"communication directive[{i}] matched zero compute nodes: {raw}")

    if not defer_order:
        apply_order_directives(training_dag, directives)
