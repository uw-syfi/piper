from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from html import escape
from pathlib import Path

from .dag import TrainingDAG, TrainingDAGNode, _topological_levels
from .ordering import _serial_topological_order
from .state import LOG_LEVEL, create_logger
from .tasks import training_dag_task_type

logger = create_logger("visualization", LOG_LEVEL)

_VALID_PASSES = frozenset({"F", "B", "BI", "BW"})
_PASS_COLOR: dict[str, str] = {
    "F": "#FFE08A",
    "B": "#A8E6A1",
    "BI": "#A9D6FF",
    "BW": "#C6B6FF",
}


@dataclass(frozen=True)
class _ScheduleOp:
    pp: int
    mb: int
    pass_: str


_ScheduleSlot = list[_ScheduleOp]


def _format_tag(tag: dict[str, int | None]) -> str:
    if not tag:
        return "(no-tags)"
    keys = sorted(tag.keys(), key=lambda k: (k != "PP", k != "EP", k))
    return ", ".join(f"{k}={tag[k]}" for k in keys)


def _format_device(device: list[int] | None) -> str:
    return "None" if device is None else str(list(device))


def render_training_dag(training_dag: TrainingDAG, output_path: str = "out/training_dag") -> None:
    """Render TrainingDAG with node labels as tags and data-dependency edges."""
    import graphviz

    dot = graphviz.Digraph("TrainingDAG", comment="Training DAG (tag-labelled)")
    dot.attr(rankdir="LR")
    topo_levels = _topological_levels(training_dag)

    def _format_node_meta_lines(node: TrainingDAGNode) -> str:
        keys = (
            "bucket_key",
            "zero_alloc_full_grads_before",
            "zero_free_full_params_after",
        )
        lines = [
            f"{key}={node.node_meta[key]!r}"
            for key in keys
            if key in node.node_meta
        ]
        return "\\n".join(lines)

    for uid, node in training_dag.nodes.items():
        topo_label = f"topo={topo_levels.get(uid, '?')}"
        meta_label = _format_node_meta_lines(node)
        meta_suffix = f"\\n{meta_label}" if meta_label else ""
        if node.node_kind in (
            "SEND_COMM",
            "RECV_COMM",
            "REDUCE_COMM",
            "ALL_GATHER_COMM",
            "REDUCE_SCATTER_COMM",
            "A2A_COMM",
        ):
            label = (
                f"{topo_label}\\n{node.node_kind}\\n{_format_tag(node.tag)}\\n"
                f"{_format_device(node.device)}\\nstream={node.stream}"
                f"{meta_suffix}"
            )
            shape = "box"
        else:
            subkind = node.compute_subkind or "COMPUTE"
            label = (
                f"{topo_label}\\n{subkind}\\n{_format_tag(node.tag)}\\n{_format_device(node.device)}\\nstream={node.stream}"
                f"{meta_suffix}"
            )
            shape = "ellipse"
        dot.node(uid, label=label, shape=shape)

    nodes_by_level: dict[int, list[str]] = {}
    for uid, level in topo_levels.items():
        nodes_by_level.setdefault(level, []).append(uid)
    for level, uids in sorted(nodes_by_level.items()):
        with dot.subgraph(name=f"topo_{level}") as sub:
            sub.attr(rank="same")
            for uid in sorted(uids):
                sub.node(uid)
    level_reps = [
        sorted(uids)[0]
        for _level, uids in sorted(nodes_by_level.items())
        if uids
    ]
    for src_uid, dst_uid in zip(level_reps, level_reps[1:]):
        dot.edge(src_uid, dst_uid, style="invis", weight="100")

    for edge in training_dag.edges:
        if edge.dep_kind == "data":
            dot.edge(edge.src_uid, edge.dst_uid)
        elif edge.dep_kind == "temporal":
            dot.edge(edge.src_uid, edge.dst_uid, style="dashed", color="blue")

    out = dot.render(output_path, format="png", cleanup=True)
    logger.info("Local DAG IR visualization saved to %s", out)


def log_training_dag_dependencies(training_dag: TrainingDAG) -> None:
    """Log predecessor/successor tags for each TrainingDAG node."""
    def _sort_key(item: tuple[str, TrainingDAGNode]) -> tuple[int, int]:
        _uid, node = item
        if node.node_kind == "COMPUTE":
            return (int(node.node_meta.get("stage_id", 10**9)), int(node.node_meta.get("segment_id", 10**9)))
        return (10**9, 10**9)

    for uid, node in sorted(training_dag.nodes.items(), key=_sort_key):
        pred_tags = [
            _format_tag(training_dag.nodes[p].tag)
            for p in sorted(training_dag.preds.get(uid, []), key=lambda x: x)
        ]
        succ_tags = [
            _format_tag(training_dag.nodes[s].tag)
            for s in sorted(training_dag.succs.get(uid, []), key=lambda x: x)
        ]
        logger.debug(
            "TrainingDAG node=%s kind=%s subkind=%s tag=(%s) device=%s stream=%s preds=[%s] succs=[%s]",
            uid,
            node.node_kind,
            node.compute_subkind,
            _format_tag(node.tag),
            _format_device(node.device),
            node.stream,
            ", ".join(pred_tags),
            ", ".join(succ_tags),
        )


def print_training_dag_order(
    dag: TrainingDAG,
    label: str = "",
    rank: int = 0,
    out_dir: str = "out",
) -> None:
    """Write and log the actor dispatch order for a TrainingDAG."""
    topo_levels = _topological_levels(dag)
    topo = _serial_topological_order(dag, topo_levels)
    data_preds: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    data_succs: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    temporal_preds: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    temporal_succs: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    for e in dag.edges:
        if e.dep_kind == "data":
            data_preds[e.dst_uid].append(e.src_uid)
            data_succs[e.src_uid].append(e.dst_uid)
        else:
            temporal_preds[e.dst_uid].append(e.src_uid)
            temporal_succs[e.src_uid].append(e.dst_uid)

    header = f"--- TrainingDAG execution order{': ' + label if label else ''} ---"
    lines = [header]
    for uid in topo:
        node = dag.nodes[uid]
        task_type = training_dag_task_type(node)
        meta = node.node_meta
        extra = []
        for key in ("bucket_key", "fwd_uid", "peer_pp_rank", "from_uid", "to_uid"):
            if key in meta and meta[key] is not None:
                extra.append(f"{key}={meta[key]}")
        extra_str = ("  " + " ".join(extra)) if extra else ""
        line = (
            f"  topo={topo_levels[uid]:<3d}  "
            f"{task_type.value:<14s}  kind={node.node_kind:<19s}  uid={node.uid:<24s}  "
            f"tag=({_format_tag(node.tag)})  stream={node.stream}{extra_str}  "
            f"data_preds={sorted(data_preds[node.uid])} data_succs={sorted(data_succs[node.uid])}  "
            f"temp_preds={sorted(temporal_preds[node.uid])} temp_succs={sorted(temporal_succs[node.uid])}"
        )
        lines.append(line)

    lines.append("-" * len(header))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"training_dag_order_rank{rank}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Pipeline schedule visualization saved to %s", out_path)


def visualize_order_directives(
    order_directives: list[dict],
    output_path: str | Path = "out/schedule",
    fmt: str = "png",
) -> str:
    """Render a 2-D schedule grid, one row per physical rank."""
    rows = _rows_from_order_directives(order_directives)
    placements = _compute_columns(rows)
    max_col = max(
        (col + _slot_width(rows[rank][slot_idx]) - 1 for (rank, slot_idx), col in placements.items()),
        default=-1,
    )

    table: list[list[_ScheduleSlot | None]] = [
        [None for _ in range(max_col + 1)]
        for _ in rows
    ]
    for key, col in placements.items():
        rank, slot_idx = key
        table[rank][col] = rows[rank][slot_idx]

    output = Path(output_path)
    if output.suffix:
        fmt = output.suffix.lstrip(".")
        output = output.with_suffix("")

    output.parent.mkdir(parents=True, exist_ok=True)
    rendered_suffix = f".{fmt}"
    _cleanup_visualizer_sidecars(output, keep_suffix=rendered_suffix)
    import graphviz

    dot = graphviz.Digraph("PipelineSchedule")
    dot.attr(rankdir="LR")
    dot.node("schedule", label=_schedule_html_table(table), shape="plain")
    try:
        return dot.render(str(output), format=fmt, cleanup=True)
    finally:
        _cleanup_visualizer_sidecars(output, keep_suffix=rendered_suffix)


def _op_width(op: _ScheduleOp) -> int:
    if op.pass_ == "B":
        return 2
    return 1


def _slot_width(slot: _ScheduleSlot) -> int:
    return sum(_op_width(op) for op in slot)


def _cleanup_visualizer_sidecars(
    output_base: Path,
    keep_suffix: str | None = None,
) -> None:
    for suffix in (".dot", ".txt"):
        if suffix == keep_suffix:
            continue
        with suppress(FileNotFoundError):
            output_base.with_suffix(suffix).unlink()


def _rows_from_order_directives(order_directives: list[dict]) -> list[list[_ScheduleSlot]]:
    rows = []
    for directive in order_directives:
        if directive.get("op") != "order":
            raise ValueError(f"expected order directive, got {directive}")
        filters = directive.get("filters")
        if not isinstance(filters, list):
            raise ValueError(f"order directive requires filters list: {directive}")
        row: list[_ScheduleSlot] = []
        for filter_slot in _parse_filter_slots(filters):
            slot: _ScheduleSlot = []
            for flt in filter_slot:
                spec = _filter_to_dict(flt)
                pass_value = spec["PASS"]
                if pass_value not in _VALID_PASSES:
                    raise ValueError(
                        f"unsupported PASS={pass_value!r}; expected one of {sorted(_VALID_PASSES)}"
                    )
                slot.append(_ScheduleOp(pp=int(spec["PP"]), mb=int(spec["MB"]), pass_=pass_value))
            row.append(slot)
        rows.append(row)
    return rows


def _parse_filter_slots(filters: Sequence[object]) -> list[list[dict]]:
    """Group the directive's filter entries into visual schedule slots."""
    slots: list[list[dict]] = []
    for item in filters:
        if not isinstance(item, list) or not item:
            raise ValueError(f"invalid order filter group: {item}")
        if not all(isinstance(flt, dict) for flt in item):
            raise ValueError(f"invalid order filter group: {item}")
        slots.append(list(item))
    return slots


def _filter_to_dict(flt: object) -> dict:
    if isinstance(flt, dict):
        return dict(flt)
    raise ValueError(f"invalid order filter: {flt}")


def _compute_columns(rows: list[list[_ScheduleSlot]]) -> dict[tuple[int, int], int]:
    key_by_op: dict[tuple[int, int, str], tuple[int, int]] = {}
    for rank, row in enumerate(rows):
        for slot_idx, slot in enumerate(row):
            for op in slot:
                op_key = (op.pp, op.mb, op.pass_)
                if op_key in key_by_op:
                    raise ValueError(f"duplicate scheduled op: {op_key}")
                key_by_op[op_key] = (rank, slot_idx)

    preds: dict[tuple[int, int], set[tuple[int, int]]] = {
        (rank, slot_idx): set()
        for rank, row in enumerate(rows)
        for slot_idx, _slot in enumerate(row)
    }
    succs: dict[tuple[int, int], set[tuple[int, int]]] = {
        key: set() for key in preds
    }

    def add_edge(src: tuple[int, int] | None, dst: tuple[int, int] | None) -> None:
        if src is None or dst is None or src == dst:
            return
        preds[dst].add(src)
        succs[src].add(dst)

    for rank, row in enumerate(rows):
        for slot_idx in range(1, len(row)):
            add_edge((rank, slot_idx - 1), (rank, slot_idx))

    for rank, row in enumerate(rows):
        for slot_idx, slot in enumerate(row):
            dst = (rank, slot_idx)
            for op in slot:
                if op.pass_ == "F":
                    add_edge(key_by_op.get((op.pp - 1, op.mb, "F")), dst)
                elif op.pass_ == "B":
                    add_edge(key_by_op.get((op.pp, op.mb, "F")), dst)
                    add_edge(key_by_op.get((op.pp + 1, op.mb, "B")), dst)
                elif op.pass_ == "BI":
                    add_edge(key_by_op.get((op.pp, op.mb, "F")), dst)
                    add_edge(key_by_op.get((op.pp + 1, op.mb, "BI")), dst)
                elif op.pass_ == "BW":
                    add_edge(key_by_op.get((op.pp, op.mb, "BI")), dst)
                else:
                    raise ValueError(f"unsupported pass={op.pass_!r}")

    columns = {key: 0 for key in preds}
    pending = {key: set(value) for key, value in preds.items()}
    ready = [key for key, value in pending.items() if not value]
    visited = 0

    while ready:
        key = ready.pop(0)
        visited += 1
        for succ in sorted(succs[key]):
            rank, slot_idx = key
            columns[succ] = max(columns[succ], columns[key] + _slot_width(rows[rank][slot_idx]))
            pending[succ].discard(key)
            if not pending[succ]:
                ready.append(succ)

    if visited != len(preds):
        blocked = [key for key, value in pending.items() if value]
        raise ValueError(f"schedule dependencies contain a cycle; blocked={blocked[:8]}")
    return columns


def _schedule_html_table(table: list[list[_ScheduleSlot | None]]) -> str:
    max_cols = max((len(row) for row in table), default=0)
    lines = ['<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">']
    lines.append('<TR><TD BGCOLOR="#eeeeee">rank</TD>')
    for col in range(max_cols):
        lines.append(f'<TD BGCOLOR="#eeeeee">t{col}</TD>')
    lines.append("</TR>")
    for rank, row in enumerate(table):
        lines.append(f'<TR><TD BGCOLOR="#eeeeee">r{rank}</TD>')
        col = 0
        while col < len(row):
            slot = row[col]
            if slot is None:
                lines.append('<TD WIDTH="72" HEIGHT="36"></TD>')
                col += 1
            else:
                colspan = _slot_width(slot)
                colspan_attr = f' COLSPAN="{colspan}"' if colspan > 1 else ""
                width = 72 * colspan
                if len(slot) == 1:
                    op = slot[0]
                    color = _PASS_COLOR[op.pass_]
                    label = f"{escape(op.pass_)}<BR/>PP{op.pp} MB{op.mb}"
                    lines.append(
                        f'<TD BGCOLOR="{color}" WIDTH="{width}" HEIGHT="36"{colspan_attr}>{label}</TD>'
                    )
                else:
                    inner = ['<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">']
                    for op in slot:
                        color = _PASS_COLOR[op.pass_]
                        label = f"{escape(op.pass_)}<BR/>PP{op.pp} MB{op.mb}"
                        inner.append(
                            f'<TR><TD BGCOLOR="{color}" WIDTH="{width}" HEIGHT="36">{label}</TD></TR>'
                        )
                    inner.append("</TABLE>")
                    lines.append(
                        f'<TD WIDTH="{width}"{colspan_attr} CELLPADDING="0">{"".join(inner)}</TD>'
                    )
                col += colspan
        lines.append("</TR>")
    lines.append("</TABLE>>")
    return "".join(lines)
