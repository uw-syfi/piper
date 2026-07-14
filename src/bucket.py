import operator
from collections import defaultdict

import torch
import torch.fx as fx

from .state import LOG_LEVEL, create_logger

logger = create_logger("bucket", LOG_LEVEL)


# ---------------------------------------------------------------------------
# bucket_stage - stage-level parameter bucketing (placeholder params)
# ---------------------------------------------------------------------------
def bucket_stage(
    stage_gm: fx.GraphModule,
    graphargs: list,
    input_idxs: list[int],
    param_idxs: list[int],
    bucket_size_bytes: int = 25 * 1024 * 1024,
    debug_name: str | None = None,
) -> list[tuple[fx.GraphModule, list[int], list[int], list]]:
    """Split a stage GraphModule into per-parameter-bucket sub-modules.

    This function handles annotation segments where trainable parameters are
    passed as **placeholder inputs** identified by *param_idxs*.

    Args:
        stage_gm: The stage ``fx.GraphModule`` to split.
        graphargs: Flat arg list for ``stage_gm.forward``; entries at
            *param_idxs* are meta tensors, entries at *input_idxs* are None.
        input_idxs: Positions of activation-input placeholders.
        param_idxs: Positions of parameter placeholders.
        bucket_size_bytes: Target bucket size (default 25 MB).
        debug_name: Optional label included in bucket planning logs.

    Returns:
        A list of ``(bucket_gm, bucket_input_idxs, bucket_param_idxs,
        bucket_graphargs)`` tuples in execution order.  Returns a
        single-element list containing the original stage when no split is
        needed.
    """
    nodes = list(stage_gm.graph.nodes)
    node_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}
    ph_nodes = [nd for nd in nodes if nd.op == "placeholder"]

    param_ph_set = {ph_nodes[i] for i in param_idxs if i < len(ph_nodes)}
    param_ph_list = [ph_nodes[i] for i in param_idxs if i < len(ph_nodes)]
    input_ph_set  = {ph_nodes[i] for i in input_idxs  if i < len(ph_nodes)}

    if not param_ph_list:
        return [(stage_gm, list(input_idxs), list(param_idxs), list(graphargs))]

    compute_set: frozenset[int] = frozenset(
        i for i, nd in enumerate(nodes)
        if nd.op not in ("placeholder", "get_attr", "output")
    )

    def _use_range(pnd: fx.Node) -> tuple[int, int]:
        idxs = [node_idx[u] for u in pnd.users if node_idx[u] in compute_set]
        return (min(idxs), max(idxs)) if idxs else (node_idx[pnd], node_idx[pnd])

    def _size(pnd: fx.Node) -> int:
        i = ph_nodes.index(pnd)
        if i < len(graphargs) and graphargs[i] is not None and hasattr(graphargs[i], "numel"):
            return int(graphargs[i].numel() * graphargs[i].element_size())
        ev = pnd.meta.get("example_value")
        if ev is not None and hasattr(ev, "numel"):
            return int(ev.numel() * ev.element_size())
        return 0

    debug_name = debug_name or getattr(stage_gm, "__class__", type(stage_gm)).__name__
    param_ranges = {pn: _use_range(pn) for pn in param_ph_list}
    param_sizes = {pn: _size(pn) for pn in param_ph_list}

    # Greedy bucket assignment
    bucket_id: dict[fx.Node, int] = {}
    cur_b, cur_sz = 0, 0
    for pn in param_ph_list:
        sz = param_sizes[pn]
        if cur_sz + sz > bucket_size_bytes and cur_sz > 0:
            cur_b += 1
            cur_sz = 0
        bucket_id[pn] = cur_b
        cur_sz += sz
    n_init = cur_b + 1

    # Singleton promotion
    def _init_range(b: int) -> tuple[int, int]:
        ms = [pn for pn in param_ph_list if bucket_id[pn] == b]
        if not ms:
            return (0, 0)
        return (min(param_ranges[pn][0] for pn in ms), max(param_ranges[pn][1] for pn in ms))

    init_ranges = [_init_range(b) for b in range(n_init)]

    def _bucket_at(idx: int) -> int:
        for b, (lo, hi) in enumerate(init_ranges):
            if lo <= idx <= hi:
                return b
        return 0

    sn = n_init
    for pn in param_ph_list:
        f, l = param_ranges[pn]
        if _bucket_at(f) != _bucket_at(l):
            bucket_id[pn] = sn
            sn += 1

    # Build bucket list, sort, merge overlapping ranges
    bm: dict[int, list[fx.Node]] = defaultdict(list)
    for pn in param_ph_list:
        bm[bucket_id[pn]].append(pn)

    blist: list[tuple[int, int, list[fx.Node]]] = []
    for members in bm.values():
        f = min(param_ranges[pn][0] for pn in members)
        l = max(param_ranges[pn][1] for pn in members)
        blist.append((f, l, members))
    blist.sort(key=lambda t: t[0])

    merged: list[tuple[int, int, list[fx.Node]]] = []
    for f, l, ms in blist:
        if merged and f <= merged[-1][1]:
            pf, pl, pm = merged[-1]
            merged[-1] = (pf, max(pl, l), pm + ms)
        else:
            merged.append((f, l, ms))

    def _bucket_bytes(members: list[fx.Node]) -> int:
        return sum(param_sizes[pn] for pn in members)

    def _format_members(members: list[fx.Node], *, limit: int = 8) -> str:
        def _one(pn: fx.Node) -> str:
            return f"{pn.name}:{param_sizes[pn]}"

        if len(members) <= limit:
            return "[" + ", ".join(_one(pn) for pn in members) + "]"
        head = ", ".join(_one(pn) for pn in members[:4])
        tail = ", ".join(_one(pn) for pn in members[-2:])
        return f"[{head}, ... ({len(members) - 6} more), {tail}]"

    alias_methods = {
        "view", "_unsafe_view", "reshape", "transpose", "permute", "t",
        "movedim", "moveaxis", "swapdims", "swapaxes",
        "select", "narrow", "slice", "split", "chunk", "unbind",
        "unsqueeze", "squeeze", "flatten", "expand", "diagonal",
        "detach", "alias", "as_strided",
    }
    multi_output_alias_methods = {"split", "chunk", "unbind"}

    alias_function_names = {
        "view", "_unsafe_view", "reshape", "transpose", "permute", "t",
        "movedim", "moveaxis", "swapdims", "swapaxes",
        "select", "narrow", "slice", "split", "chunk", "unbind",
        "unsqueeze", "squeeze", "flatten", "expand", "diagonal",
        "detach", "alias", "as_strided",
    }
    multi_output_alias_function_names = {"split", "chunk", "unbind"}
    hard_forbid_cross_bucket_prefixes = ("bfloat16_",)

    def _target_name(target: object) -> str | None:
        if isinstance(target, str):
            return target
        return getattr(target, "__name__", None)

    def _alias_passthrough_sources(nd: fx.Node) -> set[fx.Node]:
        if nd.op == "call_method":
            if nd.target in alias_methods:
                base = next(iter(nd.all_input_nodes), None)
                return set(alias_sources.get(base, set())) if base is not None else set()
            return set()

        if nd.op != "call_function":
            return set()

        if nd.target == operator.getitem:
            base = nd.args[0] if nd.args else None
            if isinstance(base, fx.Node):
                return set(alias_sources.get(base, set()))
            return set()

        name = _target_name(nd.target)
        if name in alias_function_names:
            base = next(iter(nd.all_input_nodes), None)
            return set(alias_sources.get(base, set())) if base is not None else set()
        return set()

    alias_sources: dict[fx.Node, set[fx.Node]] = {}
    for nd in nodes:
        if nd.op == "placeholder":
            alias_sources[nd] = {nd} if nd in param_ph_set else set()
            continue
        if nd.op in ("get_attr", "output"):
            alias_sources[nd] = set()
            continue
        alias_sources[nd] = _alias_passthrough_sources(nd)

    def _compute_seg_metadata(
        seg_ranges: list[tuple[int, int, list[fx.Node]]]
    ) -> tuple[dict[fx.Node, int], dict[fx.Node, int], list[list[fx.Node]]]:
        """Return (node_seg, node_max_user_seg, seg_cross_in) for segment ranges."""
        cut_after = [seg_ranges[i][1] for i in range(len(seg_ranges) - 1)]

        def _seg_of(idx: int) -> int:
            return sum(1 for c in cut_after if c < idx)

        node_seg: dict[fx.Node, int] = {}
        for nd in nodes:
            if nd.op == "output":
                node_seg[nd] = len(seg_ranges) - 1
            elif nd.op == "placeholder":
                if nd in param_ph_set:
                    user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
                    node_seg[nd] = _seg_of(min(user_idxs)) if user_idxs else 0
                else:
                    node_seg[nd] = 0
            elif nd.op == "get_attr":
                user_idxs = [node_idx[u] for u in nd.users if node_idx[u] in compute_set]
                node_seg[nd] = _seg_of(min(user_idxs)) if user_idxs else 0
            else:
                node_seg[nd] = _seg_of(node_idx[nd])

        node_max_user_seg: dict[fx.Node, int] = {}
        for nd in nodes:
            if nd.op == "output":
                continue
            node_max_user_seg[nd] = (
                max(node_seg[u] for u in nd.users) if nd.users else node_seg[nd]
            )

        seg_cross_in: list[list[fx.Node]] = [[] for _ in range(len(seg_ranges))]
        for nd in nodes:
            if nd.op == "output":
                continue
            s = node_seg[nd]
            mu = node_max_user_seg[nd]
            # Only values that cross a segment boundary must be forwarded.
            if mu <= s:
                continue
            for seg in range(s + 1, mu + 1):
                seg_cross_in[seg].append(nd)

        return node_seg, node_max_user_seg, seg_cross_in

    n_segs = len(merged)
    if n_segs == 1:
        return [(stage_gm, list(input_idxs), list(param_idxs), list(graphargs))]

    while True:
        node_seg, node_max_user_seg, seg_cross_in = _compute_seg_metadata(merged)
        merged_boundary = False
        for seg in range(1, len(merged)):
            alias_crossers = [
                nd for nd in seg_cross_in[seg]
                if alias_sources.get(nd) or any(
                    nd.name.startswith(prefix) for prefix in hard_forbid_cross_bucket_prefixes
                )
            ]
            if not alias_crossers:
                continue
            pf, _pl, pms = merged[seg - 1]
            _cf, cl, cms = merged[seg]
            merged[seg - 1] = (pf, cl, pms + cms)
            del merged[seg]
            merged_boundary = True
            break
        if not merged_boundary:
            break

    _log_bucket_plan("final", merged)

    n_segs = len(merged)
    if n_segs == 1:
        return [(stage_gm, list(input_idxs), list(param_idxs), list(graphargs))]

    # Build sub-graphs
    results: list[tuple[fx.GraphModule, list[int], list[int], list]] = []

    for seg in range(n_segs):
        sub_g = fx.Graph()
        remap: dict[fx.Node, fx.Node] = {}
        new_input_idxs: list[int] = []
        new_param_idxs: list[int] = []
        new_graphargs: list = []
        pos = 0

        def _meta_tensor_for(nd: fx.Node):
            """Return a meta tensor matching nd's shape/dtype, or None if unavailable."""
            ev = nd.meta.get("example_value")
            if ev is None:
                ev = nd.meta.get("val")
            if ev is not None and hasattr(ev, "shape"):
                return torch.empty(ev.shape, dtype=ev.dtype, device="meta", requires_grad=ev.requires_grad)
            return None

        def _add_ph(nd: fx.Node, name: str, is_input: bool) -> None:
            nonlocal pos
            new_ph = sub_g.placeholder(name)
            new_ph.type = nd.type
            remap[nd] = new_ph
            if is_input:
                new_input_idxs.append(pos)
                meta = _meta_tensor_for(nd)
                if meta is None:
                    # Fallback: use the corresponding entry from the incoming
                    # graphargs for freshly-created placeholder nodes that lack
                    # example_value metadata.
                    i_orig = ph_nodes.index(nd) if nd in ph_nodes else -1
                    if 0 <= i_orig < len(graphargs):
                        meta = graphargs[i_orig]
                new_graphargs.append(meta)
            else:
                new_param_idxs.append(pos)
                i_orig = ph_nodes.index(nd) if nd in ph_nodes else -1
                new_graphargs.append(graphargs[i_orig] if 0 <= i_orig < len(graphargs) else None)
            pos += 1

        if seg == 0:
            # Non-param placeholders (activation inputs + other non-param inputs).
            for nd in nodes:
                if nd.op == "placeholder" and nd not in param_ph_set:
                    _add_ph(nd, nd.name, is_input=(nd in input_ph_set))
        else:
            # Cross-segment activation inputs from the previous segment.
            for orig in seg_cross_in[seg]:
                _add_ph(orig, f"_xseg_{orig.name}", is_input=True)

        # Parameter placeholders belonging to this segment.
        for nd in nodes:
            if nd.op == "placeholder" and nd in param_ph_set and node_seg[nd] == seg:
                _add_ph(nd, nd.name, is_input=False)

        # get_attr nodes for this segment.
        for nd in nodes:
            if nd.op == "get_attr" and node_seg[nd] == seg:
                new_ga = sub_g.get_attr(nd.target)
                new_ga.type = nd.type
                remap[nd] = new_ga

        # Compute nodes for this segment (graph order preserved).
        for nd in nodes:
            if nd.op in ("placeholder", "get_attr", "output"):
                continue
            if node_seg[nd] != seg:
                continue
            remap[nd] = sub_g.node_copy(nd, arg_transform=lambda x, r=remap: r[x])

        # Output node.
        if seg == n_segs - 1:
            orig_out = next(nd for nd in nodes if nd.op == "output")
            sub_g.output(fx.map_arg(orig_out.args[0], lambda x: remap[x]))
        else:
            out_nodes = [remap[orig] for orig in seg_cross_in[seg + 1]]
            sub_g.output(tuple(out_nodes) if len(out_nodes) != 1 else out_nodes[0])

        sub_g.lint()
        results.append((fx.GraphModule(stage_gm, sub_g), new_input_idxs, new_param_idxs, new_graphargs))

    return results
