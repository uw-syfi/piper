from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch.fx as fx

from .bucket import bucket_stage
from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    TrainingDAGNode,
    _has_path,
    _iter_data_edges,
    _remove_edge,
    _remove_node_and_incident_edges,
    _topological_order,
    _with_pass_tag,
)
from .state import LOG_LEVEL, create_logger

logger = create_logger("directives", LOG_LEVEL)
_ANY_TAG_INDEX = "__ANY_TAG_INDEX__"
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
        if k not in tag:
            return False
        if v == _ANY_TAG_INDEX:
            # "*" wildcard matches any concrete index for this tag, but not None.
            if tag[k] is None:
                return False
        elif tag[k] != v:
            return False
    return True


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
    bwd_i.tag = _with_pass_tag(bwd_i.tag, _BWD_INPUT_PASS)

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
        e for e in list(dag.edges)
        if e.dep_kind == "temporal"
        and e.src_uid == bwd_uid
        and dag.nodes[e.dst_uid].node_kind == "UPD"
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
                tag_names.update(k for k in spec if k not in runtime_tags)
            elif isinstance(spec, list):
                for item in spec:
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) == 2
                        and isinstance(item[0], str)
                        and item[0] not in runtime_tags
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
        send_uid = f"send.{send_idx}"
        send_idx += 1
        recv_uid = f"recv.{recv_idx}"
        recv_idx += 1
        stream_name = comm_stream if comm_stream is not None else "pp_stream"
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
                "peer_pp_rank": device_to_pp_rank[dst_device_key],
                "bucket_key": src.node_meta.get("bucket_key"),
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
                "peer_pp_rank": device_to_pp_rank[src_device_key],
                "bucket_key": dst.node_meta.get("bucket_key"),
            },
        )
        dag.add_node(send_node)
        dag.add_node(recv_node)
        _remove_edge(dag, edge)
        dag.add_edge(TrainingDAGEdge(src_uid=src.uid, dst_uid=send_uid, dep_kind="data", tensor_name=edge.tensor_name))
        dag.add_edge(TrainingDAGEdge(src_uid=recv_uid, dst_uid=dst.uid, dep_kind="data", tensor_name=edge.tensor_name))


def _replicate_update_nodes_by_device(dag: TrainingDAG) -> None:
    """Replicate UPD nodes per upstream device set so split components stay homogeneous."""
    upd_nodes = [n for n in list(dag.nodes.values()) if n.node_kind == "UPD" and ".rep" not in n.uid]
    for upd in upd_nodes:
        in_edges = [e for e in list(dag.edges) if e.dep_kind == "temporal" and e.dst_uid == upd.uid]
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
                    )
                )
            created_uids.append(nu)
            for e in dev_edges:
                _remove_edge(dag, e)
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=e.src_uid,
                        dst_uid=nu,
                        dep_kind=e.dep_kind,
                        tensor_name=e.tensor_name,
                    )
                )
        # Duplicate outgoing edges of the original UPD to all replicas.
        out_edges = [e for e in list(dag.edges) if e.dep_kind == "temporal" and e.src_uid == upd.uid]
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
        e for e in list(dag.edges)
        if e.dep_kind == "temporal"
        and e.src_uid == bwd_uid
        and dag.nodes[e.dst_uid].node_kind == "UPD"
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


def _bucket_matched_fwd_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    bucket_size_mb: int,
) -> None:
    if bucket_size_mb <= 0:
        raise ValueError(f"bucket_size must be > 0 MB, got {bucket_size_mb}")
    bucket_size_bytes = int(bucket_size_mb) * 1024 * 1024

    fwd_targets = [
        uid for uid, node in dag.nodes.items()
        if node.node_kind == "COMPUTE"
        and node.compute_subkind == "FWD"
        and any(_match_filter(node.tag, flt) for flt in filters)
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
        in_fwd = [e for e in dag.edges if e.dst_uid == fwd_uid]
        out_fwd = [e for e in dag.edges if e.src_uid == fwd_uid]
        in_bwd = [e for e in dag.edges if e.dst_uid == bwd_uid]
        out_bwd = [e for e in dag.edges if e.src_uid == bwd_uid]

        orig_stage = int(fwd_node.node_meta.get("stage_id", -1))
        orig_seg = int(fwd_node.node_meta.get("segment_id", -1))

        fwd_bucket_uids: list[str] = []
        bwd_bucket_uids: list[str] = []
        for bi, (bgm, b_in, b_param, b_args) in enumerate(buckets):
            f_uid = f"{fwd_uid}.bucket{bi}"
            b_uid = f"{f_uid}.bwd"
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
                        "input_names": [n.name for n in bgm.graph.nodes if n.op == "placeholder"],
                        "output_names": [],
                        "a2a_boundary_after": None,
                        "bucket_idx": bi,
                        "num_buckets": len(buckets),
                        "bucket_key": f_uid,
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
                        "bucket_key": f_uid,
                        "compute_loss": bool(bwd_node.node_meta.get("compute_loss", False)) and bi == (len(buckets) - 1),
                    },
                )
            )

        first_fwd = fwd_bucket_uids[0]
        last_fwd = fwd_bucket_uids[-1]
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
    reduce_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "REDUCE_COMM")
    expected = sorted(int(d) for d in devices)
    for node in list(dag.nodes.values()):
        if node.node_kind != "COMPUTE" or not _is_backward_weight_subkind(node.compute_subkind):
            continue
        if not any(_match_filter(node.tag, flt) for flt in filters):
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
        reduce_uid = f"reduce.{reduce_idx}"
        reduce_idx += 1
        reduce_node = TrainingDAGNode(
            uid=reduce_uid,
            node_kind="REDUCE_COMM",
            compute_subkind=None,
            tag=dict(node.tag),
            device=list(node.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "bwd_uid": node.uid,
                "bucket_key": node.node_meta.get("bucket_key"),
            },
        )
        dag.add_node(reduce_node)
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=node.uid,
                dst_uid=reduce_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )
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
    ag_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "ALL_GATHER_COMM")
    expected = sorted(int(d) for d in devices)
    for node in list(dag.nodes.values()):
        if (
            node.node_kind != "COMPUTE"
            or node.compute_subkind not in ("FWD", "BWD", "BWD_I", "BWD_W")
        ):
            continue
        if not any(_match_filter(node.tag, flt) for flt in filters):
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
        ag_uid = f"all_gather.{ag_idx}"
        ag_idx += 1
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
    rs_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "REDUCE_SCATTER_COMM")
    expected = sorted(int(d) for d in devices)
    for node in list(dag.nodes.values()):
        if node.node_kind != "COMPUTE" or not _is_backward_weight_subkind(node.compute_subkind):
            continue
        if not any(_match_filter(node.tag, flt) for flt in filters):
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
        rs_uid = f"reduce_scatter.{rs_idx}"
        rs_idx += 1
        rs_node = TrainingDAGNode(
            uid=rs_uid,
            node_kind="REDUCE_SCATTER_COMM",
            compute_subkind=None,
            tag=dict(node.tag),
            device=list(node.device),
            stream=comm_stream if comm_stream is not None else "default_stream",
            node_meta={
                "bwd_uid": node.uid,
                "bucket_key": node.node_meta.get("bucket_key"),
            },
        )
        node.node_meta["zero_alloc_full_grads_before"] = True
        if node.node_meta.pop("zero_free_full_params_after", None) is not None:
            rs_node.node_meta["zero_free_full_params_after"] = True
        dag.add_node(rs_node)
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=node.uid,
                dst_uid=rs_uid,
                dep_kind="data",
                tensor_name=None,
            )
        )
        _rewire_bwd_successors_through_sync(dag, node.uid, rs_uid)


def _insert_shard_a2a_comm_nodes(
    dag: TrainingDAG,
    filters: list[dict[str, Any]],
    devices: list[int],
    comm_stream: str | None = None,
) -> None:
    expected = sorted(int(d) for d in devices)
    a2a_idx = sum(1 for n in dag.nodes.values() if n.node_kind == "A2A_COMM")

    matched_uids = [
        uid for uid, node in dag.nodes.items()
        if node.node_kind == "COMPUTE" and any(_match_filter(node.tag, flt) for flt in filters)
    ]

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

    for uid in matched_uids:
        if uid not in dag.nodes:
            continue
        node = dag.nodes[uid]
        if node.device is None or sorted(int(d) for d in node.device) != expected:
            raise ValueError(
                f"shard requires matched node {uid} to have devices={devices}, got node_devices={node.device}"
            )

        if node.compute_subkind == "FWD":
            node.node_meta["apply_zero"] = False
        elif _is_backward_compute_subkind(node.compute_subkind):
            fwd_uid = node.node_meta.get("fwd_uid")
            if isinstance(fwd_uid, str) and fwd_uid in dag.nodes:
                dag.nodes[fwd_uid].node_meta["apply_zero"] = False

        # Shard replaces replicate-style grad/param sync comms around matched
        # compute nodes; remove existing gather/reduce comm nodes attached to
        # this compute node before inserting A2A edges.
        removable_sync_kinds = {"ALL_GATHER_COMM", "REDUCE_COMM", "REDUCE_SCATTER_COMM"}
        removable_comm_uids: set[str] = set()
        for e in list(dag.edges):
            if e.dep_kind != "data":
                continue
            if e.src_uid == uid and e.dst_uid in dag.nodes:
                other = dag.nodes[e.dst_uid]
                if other.node_kind in removable_sync_kinds:
                    removable_comm_uids.add(other.uid)
            elif e.dst_uid == uid and e.src_uid in dag.nodes:
                other = dag.nodes[e.src_uid]
                if other.node_kind in removable_sync_kinds:
                    removable_comm_uids.add(other.uid)
        for comm_uid in sorted(removable_comm_uids):
            if comm_uid not in dag.nodes:
                continue
            # Bypass removed sync-comm nodes so dataflow remains connected:
            # pred -> COMM -> succ  ==>  pred -> succ
            in_comm = [e for e in list(dag.edges) if e.dep_kind == "data" and e.dst_uid == comm_uid]
            out_comm = [e for e in list(dag.edges) if e.dep_kind == "data" and e.src_uid == comm_uid]
            for ie in in_comm:
                for oe in out_comm:
                    dag.add_edge(
                        TrainingDAGEdge(
                            src_uid=ie.src_uid,
                            dst_uid=oe.dst_uid,
                            dep_kind="data",
                            tensor_name=(oe.tensor_name if oe.tensor_name is not None else ie.tensor_name),
                        )
                    )
            _remove_node_and_incident_edges(dag, comm_uid)
        if removable_comm_uids:
            node.node_meta.pop("zero_alloc_full_grads_before", None)
            node.node_meta.pop("zero_free_full_params_after", None)
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
            e for e in list(dag.edges)
            if e.dep_kind == "data"
            and e.dst_uid == uid
            and _is_a2a_compute_edge(e)
        ]
        outgoing = [
            e for e in list(dag.edges)
            if e.dep_kind == "data"
            and e.src_uid == uid
            and _is_a2a_compute_edge(e)
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

        for e in incoming:
            src_node = dag.nodes[e.src_uid]
            binfo = _a2a_boundary_for_edge(src_node, node)
            a2a_tensor_idx = binfo["tensor_idx"]
            comm_uid = f"a2a.{a2a_idx}"
            a2a_idx += 1
            comm_node = TrainingDAGNode(
                uid=comm_uid,
                node_kind="A2A_COMM",
                compute_subkind=None,
                tag=dict(node.tag),
                device=list(node.device) if node.device is not None else None,
                stream=comm_stream if comm_stream is not None else "default_stream",
                node_meta={
                    "direction": "incoming",
                    "target_uid": uid,
                    "a2a_tensor_idx": a2a_tensor_idx,
                    "bucket_key": node.node_meta.get("bucket_key", src_node.node_meta.get("bucket_key")),
                },
            )
            dag.add_node(comm_node)
            _remove_edge(dag, e)
            dag.add_edge(TrainingDAGEdge(src_uid=e.src_uid, dst_uid=comm_uid, dep_kind="data", tensor_name=e.tensor_name))
            dag.add_edge(TrainingDAGEdge(src_uid=comm_uid, dst_uid=uid, dep_kind="data", tensor_name=e.tensor_name))

        for e in outgoing:
            dst_node = dag.nodes[e.dst_uid]
            binfo = _a2a_boundary_for_edge(node, dst_node)
            a2a_tensor_idx = binfo["tensor_idx"]
            comm_uid = f"a2a.{a2a_idx}"
            a2a_idx += 1
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
                    "a2a_tensor_idx": a2a_tensor_idx,
                    "bucket_key": node.node_meta.get("bucket_key", dst_node.node_meta.get("bucket_key")),
                },
            )
            dag.add_node(comm_node)
            _remove_edge(dag, e)
            dag.add_edge(TrainingDAGEdge(src_uid=uid, dst_uid=comm_uid, dep_kind="data", tensor_name=e.tensor_name))
            dag.add_edge(TrainingDAGEdge(src_uid=comm_uid, dst_uid=e.dst_uid, dep_kind="data", tensor_name=e.tensor_name))


def _apply_split_directive(
    dag: TrainingDAG,
    flt: dict[str, Any],
    dim_name: str,
    num_microbatches: int,
) -> None:
    matched = {uid for uid, n in dag.nodes.items() if _match_filter(n.tag, flt)}
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
        dag.nodes[uid].tag[dim_name] = 0

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
        "compute_uid",
        "bwd_uid",
        "source_uid",
        "target_uid",
        "src_uid",
        "dst_uid",
        "from_uid",
        "to_uid",
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
            dag.add_node(
                TrainingDAGNode(
                    uid=nu,
                    node_kind=n.node_kind,
                    compute_subkind=n.compute_subkind,
                    tag={**n.tag, dim_name: mb},
                    device=list(n.device) if n.device is not None else None,
                    stream=n.stream,
                    node_meta=copied_meta,
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


def _match_set_contiguous_subdag(
    dag: TrainingDAG,
    matched: set[str],
    *,
    context: str,
) -> tuple[set[str], set[str], set[str], set[str]]:
    if not matched:
        raise ValueError(f"{context}: filter matched zero nodes")
    # Contiguity should be dependency-based, not tied to one arbitrary
    # topological ordering (which can interleave independent branches).
    # A non-matching node is considered interleaving iff it is on some path
    # between matched nodes: matched -> ... -> node -> ... -> matched.
    interleaved: list[str] = []
    for u in dag.nodes.keys():
        if u in matched:
            continue
        # Has matched ancestor?
        has_matched_ancestor = False
        seen = {u}
        stack = list(dag.preds.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in matched:
                has_matched_ancestor = True
                break
            stack.extend(dag.preds.get(cur, set()))
        if not has_matched_ancestor:
            continue

        # Has matched descendant?
        has_matched_descendant = False
        seen = {u}
        stack = list(dag.succs.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            if cur in matched:
                has_matched_descendant = True
                break
            stack.extend(dag.succs.get(cur, set()))
        if has_matched_descendant:
            interleaved.append(u)

    if interleaved:
        raise ValueError(
            f"{context}: matched nodes are not contiguous; interleaved non-matching nodes: {interleaved[:8]}"
        )

    sources = {
        u for u in matched
        if not any(pred in matched for pred in dag.preds.get(u, set()))
    }
    sinks = {
        u for u in matched
        if not any(succ in matched for succ in dag.succs.get(u, set()))
    }
    if not sources or not sinks:
        raise ValueError(f"{context}: failed to identify non-empty source/sink sets")
    compute_nodes = {u for u in matched if dag.nodes[u].node_kind == "COMPUTE"}

    def _has_upstream_compute(u: str) -> bool:
        seen = {u}
        stack = list(dag.preds.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in matched:
                continue
            seen.add(cur)
            if dag.nodes[cur].node_kind == "COMPUTE":
                return True
            stack.extend(dag.preds.get(cur, set()))
        return False

    def _has_downstream_compute(u: str) -> bool:
        seen = {u}
        stack = list(dag.succs.get(u, set()))
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in matched:
                continue
            seen.add(cur)
            if dag.nodes[cur].node_kind == "COMPUTE":
                return True
            stack.extend(dag.succs.get(cur, set()))
        return False

    compute_sources = {u for u in compute_nodes if not _has_upstream_compute(u)}
    compute_sinks = {u for u in compute_nodes if not _has_downstream_compute(u)}
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
    if src_device != dst_device:
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
            matched = {
                uid for uid, n in dag.nodes.items()
                if n.node_kind != "ORDER_DUMMY" and _match_filter(n.tag, flt)
            }
            _srcs, _snks, csrcs, csnks = _match_set_contiguous_subdag(
                dag,
                matched,
                context=f"order filter[{group_idx}][{filter_idx}]={flt}",
            )
            group_sources.append(csrcs)
            group_sinks.append(csnks)
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


def apply_schedule_directives(training_dag: TrainingDAG, directives: list[Any] | None) -> None:
    if not directives:
        return
    split_backward_keys = _validate_split_backward_order_stencil(directives)

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
    place_streams: set[str] = set()
    for i, raw in enumerate(directives):
        if isinstance(raw, dict) and raw.get("op") != "place":
            continue
        op, filters, devices, stream, _gather_stream, _reduce_stream, shard_params, shard_grads, bucket_size = _normalize_filter_devices_directive(raw)
        logger.debug(
            "Applying directive[%d]: %s(filters=%s, devices=%s, stream=%s, shard_params=%s, shard_grads=%s, bucket_size=%s)",
            i, op, filters, devices, stream, shard_params, shard_grads, bucket_size
        )
        matched_nodes = 0
        for node in training_dag.nodes.values():
            if node.node_kind not in ("COMPUTE", "UPD"):
                continue
            if any(_match_filter(node.tag, flt) for flt in filters):
                node.device = list(devices)
                matched_nodes += 1
        if matched_nodes == 0:
            raise ValueError(f"place directive[{i}] matched zero nodes: {raw}")
        if stream is not None:
            place_streams.add(stream)

    if place_streams:
        if len(place_streams) != 1:
            raise ValueError(f"place directives must agree on stream, got {sorted(place_streams)}")
        place_stream = next(iter(place_streams))
    else:
        place_stream = None
    _replicate_update_nodes_by_device(training_dag)
    _insert_send_recv_comm_nodes(training_dag, comm_stream=place_stream)

    for i, raw in enumerate(directives):
        if isinstance(raw, dict) and raw.get("op") in ("place", "split", "order"):
            continue
        op, filters, devices, stream, gather_stream, reduce_stream, shard_params, shard_grads, bucket_size = _normalize_filter_devices_directive(raw)
        logger.debug(
            "Applying directive[%d]: %s(filters=%s, devices=%s, stream=%s, gather_stream=%s, reduce_stream=%s, shard_params=%s, shard_grads=%s, bucket_size=%s)",
            i, op, filters, devices, stream, gather_stream, reduce_stream, shard_params, shard_grads, bucket_size
        )
        matched_nodes = [
            uid for uid, node in training_dag.nodes.items()
            if node.node_kind == "COMPUTE"
            and any(_match_filter(node.tag, flt) for flt in filters)
        ]
        if not matched_nodes:
            raise ValueError(f"{op} directive[{i}] matched zero compute nodes: {raw}")
        if op == "replicate":
            if bucket_size is not None:
                _bucket_matched_fwd_nodes(training_dag, filters, int(bucket_size))
            if shard_params:
                _insert_all_gather_comm_nodes(training_dag, filters, devices, comm_stream=gather_stream)
                _insert_reduce_scatter_comm_nodes(training_dag, filters, devices, comm_stream=reduce_stream)
            elif shard_grads:
                _insert_reduce_scatter_comm_nodes(training_dag, filters, devices, comm_stream=reduce_stream)
            else:
                _insert_reduce_comm_nodes(training_dag, filters, devices, comm_stream=reduce_stream)
        elif op == "shard":
            _insert_shard_a2a_comm_nodes(training_dag, filters, devices, comm_stream=stream)
        else:
            raise ValueError(f"Unsupported directive op after normalization: {op}")

    # Final pass for order directives (add temporal dependencies across sub-DAGs).
    for i, raw in enumerate(directives):
        if not isinstance(raw, dict) or raw.get("op") != "order":
            continue
        filter_groups = _parse_order_directive(raw)
        logger.debug("Applying directive[%d]: order(filters=%s)", i, filter_groups)
        _apply_order_directive(training_dag, filter_groups, directive_idx=i)
