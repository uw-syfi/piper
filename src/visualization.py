# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import os
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from .dag import TrainingDAG, TrainingDAGNode, _topological_levels
from .ordering import _dispatch_order
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
    region: tuple[tuple[str, Any], ...]
    mb: int
    pass_: str
    devices: tuple[int, ...]


_ScheduleSlot = list[_ScheduleOp]


@dataclass(frozen=True)
class _ScheduleRow:
    devices: tuple[int, ...]
    slots: list[_ScheduleSlot]


_RUNTIME_FILTER_KEYS = frozenset({"MB", "PASS"})


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
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}:
            lines.extend(
                (
                    f"routed={node.node_kind.removesuffix('_COMM')}",
                    f"role={node.node_meta.get('role')}",
                    f"direction={node.node_meta.get('direction')}",
                )
            )
        elif node.node_meta.get("dense_state_transfer"):
            lines.extend(
                (
                    "dense_state=direct",
                    f"role={node.node_meta.get('role')}",
                    f"direction={node.node_meta.get('direction')}",
                )
            )
        return "\\n".join(lines)

    for uid, node in training_dag.nodes.items():
        meta_label = _format_node_meta_lines(node)
        meta_suffix = f"\\n{meta_label}" if meta_label else ""
        if node.node_kind in (
            "SEND_COMM",
            "RECV_COMM",
            "REDUCE_COMM",
            "ALL_GATHER_COMM",
            "REDUCE_SCATTER_COMM",
            "A2A_COMM",
            "M_TO_N_COMM",
            "N_TO_M_COMM",
        ):
            label = (
                f"{node.node_kind}\\n{_format_tag(node.tag)}\\n"
                f"{_format_device(node.device)}\\nstream={node.stream}"
                f"{meta_suffix}"
            )
            shape = "box"
        else:
            subkind = node.compute_subkind or "COMPUTE"
            label = (
                f"{subkind}\\n{_format_tag(node.tag)}\\n{_format_device(node.device)}\\nstream={node.stream}"
                f"{meta_suffix}"
            )
            shape = "ellipse"
        dot.node(uid, label=label, shape=shape)

    nodes_by_dispatch_level: dict[int, list[str]] = {}
    for uid, topo_level in topo_levels.items():
        stored_dispatch_level = training_dag.nodes[uid].node_meta.get(
            "dispatch_level"
        )
        display_level = (
            stored_dispatch_level
            if isinstance(stored_dispatch_level, int)
            and not isinstance(stored_dispatch_level, bool)
            else topo_level
        )
        nodes_by_dispatch_level.setdefault(display_level, []).append(uid)
    for level, uids in sorted(nodes_by_dispatch_level.items()):
        with dot.subgraph(name=f"dispatch_{level}") as sub:
            sub.attr(rank="same")
            for uid in sorted(uids):
                sub.node(uid)
    level_reps = [
        sorted(uids)[0]
        for _level, uids in sorted(nodes_by_dispatch_level.items())
        if uids
    ]
    for src_uid, dst_uid in zip(level_reps, level_reps[1:]):
        dot.edge(src_uid, dst_uid, style="invis", weight="100")

    for edge in training_dag.edges:
        if edge.dep_kind == "data":
            dot.edge(edge.src_uid, edge.dst_uid)
        elif edge.dep_kind == "temporal":
            dot.edge(edge.src_uid, edge.dst_uid, style="dashed", color="blue")
        elif edge.dep_kind == "issue":
            dot.edge(
                edge.src_uid,
                edge.dst_uid,
                style="dotted",
                color="darkorange",
                label="issue",
            )

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
    output_name: str | None = None,
) -> None:
    """Write and log the actor dispatch order for a TrainingDAG."""
    topo_levels = _topological_levels(dag)
    topo = _dispatch_order(dag)
    data_preds: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    data_succs: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    temporal_preds: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    temporal_succs: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    issue_preds: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    issue_succs: dict[str, list[str]] = {uid: [] for uid in dag.nodes}
    for e in dag.edges:
        if e.dep_kind == "data":
            data_preds[e.dst_uid].append(e.src_uid)
            data_succs[e.src_uid].append(e.dst_uid)
        elif e.dep_kind == "temporal":
            temporal_preds[e.dst_uid].append(e.src_uid)
            temporal_succs[e.src_uid].append(e.dst_uid)
        elif e.dep_kind == "issue":
            issue_preds[e.dst_uid].append(e.src_uid)
            issue_succs[e.src_uid].append(e.dst_uid)

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
        dispatch_level = node.node_meta.get(
            "dispatch_level", topo_levels[uid]
        )
        line = (
            f"  topo={topo_levels[uid]:<3d} dispatch={dispatch_level:<3d}  "
            f"{task_type.value:<14s}  kind={node.node_kind:<19s}  uid={node.uid:<24s}  "
            f"tag=({_format_tag(node.tag)})  stream={node.stream}{extra_str}  "
            f"data_preds={sorted(data_preds[node.uid])} data_succs={sorted(data_succs[node.uid])}  "
            f"temp_preds={sorted(temporal_preds[node.uid])} temp_succs={sorted(temporal_succs[node.uid])}  "
            f"issue_preds={sorted(issue_preds[node.uid])} issue_succs={sorted(issue_succs[node.uid])}"
        )
        lines.append(line)

    lines.append("-" * len(header))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        output_name or f"training_dag_order_rank{rank}",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Pipeline schedule visualization saved to %s", out_path)


def visualize_order_directives(
    schedule_directives: list[dict],
    output_path: str | Path = "out/schedule",
    fmt: str = "png",
) -> str:
    """Render one timeline row per disjoint ``place.devices`` group."""
    rows = _rows_from_schedule_directives(schedule_directives)
    placements = _compute_columns(rows)
    device_groups = _placed_device_groups(schedule_directives)
    table = _build_device_table(rows, placements, device_groups)

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
    dot.node(
        "schedule",
        label=_schedule_html_table(table, device_groups),
        shape="plain",
    )
    try:
        return dot.render(str(output), format=fmt, cleanup=True)
    finally:
        _cleanup_visualizer_sidecars(output, keep_suffix=rendered_suffix)


def _op_width(op: _ScheduleOp) -> int:
    if op.pass_ == "B":
        return 2
    return 1


def _slot_width(slot: _ScheduleSlot) -> int:
    return max((_op_width(op) for op in slot), default=1)


def _cleanup_visualizer_sidecars(
    output_base: Path,
    keep_suffix: str | None = None,
) -> None:
    for suffix in (".dot", ".txt"):
        if suffix == keep_suffix:
            continue
        with suppress(FileNotFoundError):
            output_base.with_suffix(suffix).unlink()


def _rows_from_schedule_directives(schedule_directives: list[dict]) -> list[_ScheduleRow]:
    place_directives = [
        directive
        for directive in schedule_directives
        if isinstance(directive, dict) and directive.get("op") == "place"
    ]
    if not place_directives:
        raise ValueError("schedule visualization requires at least one place directive")

    rows: list[_ScheduleRow] = []
    for directive in schedule_directives:
        if not isinstance(directive, dict) or directive.get("op") != "order":
            continue
        filters = directive.get("filters")
        if not isinstance(filters, list):
            raise ValueError(f"order directive requires filters list: {directive}")
        row: list[_ScheduleSlot] = []
        row_devices: set[tuple[int, ...]] = set()
        for filter_slot in _parse_filter_slots(filters):
            slot: _ScheduleSlot = []
            for flt in filter_slot:
                spec = _filter_to_dict(flt)
                if "MB" not in spec or "PASS" not in spec:
                    raise ValueError(f"order filter requires MB and PASS: {spec}")
                pass_value = str(spec["PASS"])
                if pass_value not in _VALID_PASSES:
                    raise ValueError(
                        f"unsupported PASS={pass_value!r}; expected one of {sorted(_VALID_PASSES)}"
                    )
                region_spec = {
                    key: value
                    for key, value in spec.items()
                    if key not in _RUNTIME_FILTER_KEYS
                }
                if not region_spec:
                    raise ValueError(f"order filter does not identify a model region: {spec}")
                devices = _resolve_order_filter_devices(region_spec, place_directives)
                row_devices.add(devices)
                slot.append(
                    _ScheduleOp(
                        region=_region_key(region_spec),
                        mb=int(spec["MB"]),
                        pass_=pass_value,
                        devices=devices,
                    )
                )
            row.append(slot)
        devices = tuple(sorted({device for group in row_devices for device in group}))
        rows.append(_ScheduleRow(devices=devices, slots=row))
    if not rows:
        raise ValueError("schedule visualization requires at least one order directive")
    return rows


def _placed_device_groups(schedule_directives: list[dict]) -> list[tuple[int, ...]]:
    """Return disjoint rows while preserving every ``place.devices`` grouping."""
    groups: list[set[int]] = []
    for directive in schedule_directives:
        if not isinstance(directive, dict) or directive.get("op") != "place":
            continue
        raw_devices = directive.get("devices", directive.get("device"))
        if not isinstance(raw_devices, list) or not raw_devices:
            raise ValueError(f"place directive requires non-empty devices: {directive}")
        placed = {int(device) for device in raw_devices}
        overlapping = [group for group in groups if group & placed]
        for group in overlapping:
            placed.update(group)
            groups.remove(group)
        groups.append(placed)
    if not groups:
        raise ValueError("schedule visualization requires at least one placed device")
    return sorted(
        (tuple(sorted(group)) for group in groups),
        key=lambda group: group[0],
    )


def _build_device_table(
    rows: list[_ScheduleRow],
    placements: dict[tuple[int, int], int],
    device_groups: list[tuple[int, ...]],
) -> list[list[_ScheduleSlot]]:
    """Place each chunk once, combining chunks that start in the same timestep."""
    max_col = max(
        (
            col + _slot_width(rows[row_idx].slots[slot_idx])
            for (row_idx, slot_idx), col in placements.items()
        ),
        default=0,
    )
    table: list[list[_ScheduleSlot]] = [
        [[] for _ in range(max_col)]
        for _group in device_groups
    ]
    row_by_device = {
        device: row_idx
        for row_idx, group in enumerate(device_groups)
        for device in group
    }
    for (schedule_row_idx, slot_idx), col in placements.items():
        schedule_row = rows[schedule_row_idx]
        for op in schedule_row.slots[slot_idx]:
            device_row_indices = {row_by_device[device] for device in op.devices}
            if len(device_row_indices) != 1:
                raise ValueError(
                    f"placement {op.devices} spans visual device groups {device_groups}"
                )
            device_row_idx = next(iter(device_row_indices))
            table[device_row_idx][col].append(op)

    for device_row_idx, row in enumerate(table):
        occupied_until = 0
        for col, slot in enumerate(row):
            if not slot:
                continue
            if col < occupied_until:
                raise ValueError(
                    "visualized schedule contains overlapping chunks on "
                    f"GPU group {device_groups[device_row_idx]} at t{col}"
                )
            occupied_until = col + _slot_width(slot)
    return table


def _region_key(region_spec: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(region_spec.items()))


def _selector_contains(container: dict[str, Any], contained: dict[str, Any]) -> bool:
    """Whether every tag selected by ``contained`` also matches ``container``."""
    for key, container_value in container.items():
        if key not in contained:
            return False
        contained_value = contained[key]
        if container_value == "*":
            if contained_value in ("-", None):
                return False
        elif container_value == "-":
            if contained_value != "-":
                return False
        elif contained_value != container_value:
            return False
    return True


def _resolve_order_filter_devices(
    region_spec: dict[str, Any],
    place_directives: list[dict],
) -> tuple[int, ...]:
    """Resolve the effective placement for a symbolic order filter.

    Place directives use ordered override semantics, so the last placement whose
    selector contains the complete order-filter region is the effective one.
    """
    devices: tuple[int, ...] | None = None
    for directive in place_directives:
        place_filter = directive.get("filter")
        if not isinstance(place_filter, dict):
            raise ValueError(f"place directive requires a filter object: {directive}")
        if not _selector_contains(place_filter, region_spec):
            continue
        raw_devices = directive.get("devices", directive.get("device"))
        if not isinstance(raw_devices, list) or not raw_devices:
            raise ValueError(f"place directive requires non-empty devices: {directive}")
        devices = tuple(int(device) for device in raw_devices)
    if devices is None:
        raise ValueError(
            f"order filter region {region_spec} is not covered by a place directive"
        )
    return devices


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


def _compute_columns(rows: list[_ScheduleRow]) -> dict[tuple[int, int], int]:
    key_by_op: dict[tuple[tuple[tuple[str, Any], ...], int, str], tuple[int, int]] = {}
    for row_idx, row in enumerate(rows):
        for slot_idx, slot in enumerate(row.slots):
            for op in slot:
                op_key = (op.region, op.mb, op.pass_)
                if op_key in key_by_op:
                    raise ValueError(f"duplicate scheduled op: {op_key}")
                key_by_op[op_key] = (row_idx, slot_idx)

    preds: dict[tuple[int, int], set[tuple[int, int]]] = {
        (row_idx, slot_idx): set()
        for row_idx, row in enumerate(rows)
        for slot_idx, _slot in enumerate(row.slots)
    }
    succs: dict[tuple[int, int], set[tuple[int, int]]] = {
        key: set() for key in preds
    }

    def add_edge(src: tuple[int, int] | None, dst: tuple[int, int] | None) -> None:
        if src is None or dst is None or src == dst:
            return
        preds[dst].add(src)
        succs[src].add(dst)

    for row_idx, row in enumerate(rows):
        for slot_idx in range(1, len(row.slots)):
            add_edge((row_idx, slot_idx - 1), (row_idx, slot_idx))

    region_order = _infer_region_order(rows)
    region_idx = {region: idx for idx, region in enumerate(region_order)}
    for row_idx, row in enumerate(rows):
        for slot_idx, slot in enumerate(row.slots):
            dst = (row_idx, slot_idx)
            for op in slot:
                idx = region_idx[op.region]
                previous_region = region_order[idx - 1] if idx > 0 else None
                next_region = region_order[idx + 1] if idx + 1 < len(region_order) else None
                if op.pass_ == "F":
                    if previous_region is not None:
                        add_edge(key_by_op.get((previous_region, op.mb, "F")), dst)
                elif op.pass_ == "B":
                    add_edge(key_by_op.get((op.region, op.mb, "F")), dst)
                    if next_region is not None:
                        add_edge(key_by_op.get((next_region, op.mb, "B")), dst)
                elif op.pass_ == "BI":
                    add_edge(key_by_op.get((op.region, op.mb, "F")), dst)
                    if next_region is not None:
                        add_edge(key_by_op.get((next_region, op.mb, "BI")), dst)
                elif op.pass_ == "BW":
                    add_edge(key_by_op.get((op.region, op.mb, "BI")), dst)
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
            row_idx, slot_idx = key
            columns[succ] = max(
                columns[succ],
                columns[key] + _slot_width(rows[row_idx].slots[slot_idx]),
            )
            pending[succ].discard(key)
            if not pending[succ]:
                ready.append(succ)

    if visited != len(preds):
        blocked = [key for key, value in pending.items() if value]
        raise ValueError(f"schedule dependencies contain a cycle; blocked={blocked[:8]}")
    return columns


def _infer_region_order(rows: list[_ScheduleRow]) -> list[tuple[tuple[str, Any], ...]]:
    """Infer logical forward order without assigning meaning to a tag name.

    Numeric annotation regions are ordered by index, with their order-directive
    row breaking ties (e.g. ATTN0, FFN0, ATTN1, FFN1). Regions selected with
    absent-tag constraints are pipeline boundaries.  Their exact numeric
    selector values and first forward occurrence distinguish the leading and
    trailing endpoints even when both endpoints live on the same actor row.
    """
    first_seen: dict[tuple[tuple[str, Any], ...], tuple[int, int]] = {}
    for row_idx, row in enumerate(rows):
        for slot_idx, slot in enumerate(row.slots):
            for op in slot:
                if op.pass_ == "F":
                    first_seen.setdefault(op.region, (row_idx, slot_idx))
    if not first_seen:
        raise ValueError("schedule visualization requires at least one forward operation")

    boundaries: list[tuple[tuple[str, Any], ...]] = []
    interior: list[tuple[tuple[str, Any], ...]] = []
    for region in first_seen:
        has_absent_constraint = any(value == "-" for _key, value in region)
        if has_absent_constraint:
            boundaries.append(region)
        else:
            interior.append(region)

    def indexed_region_key(
        region: tuple[tuple[str, Any], ...],
    ) -> tuple[int, tuple[int, ...], int, int]:
        row_idx, slot_idx = first_seen[region]
        indices = tuple(
            int(value)
            for _key, value in region
            if isinstance(value, int) and not isinstance(value, bool)
        )
        if indices:
            return (0, indices, row_idx, slot_idx)
        return (1, (), row_idx, slot_idx)

    interior.sort(key=indexed_region_key)

    def boundary_key(
        region: tuple[tuple[str, Any], ...],
    ) -> tuple[int, tuple[int, ...], int, int]:
        row_idx, slot_idx = first_seen[region]
        exact_indices = tuple(
            int(value)
            for _key, value in region
            if isinstance(value, int) and not isinstance(value, bool)
        )
        if exact_indices:
            return (0, exact_indices, row_idx, slot_idx)
        return (1, (), row_idx, slot_idx)

    boundaries.sort(key=boundary_key)
    if not boundaries:
        leading = []
        trailing = []
    elif len(boundaries) == 1:
        # A one-ended graph (common in focused schedules) keeps the historical
        # behavior and treats its sole boundary as the input.
        leading = boundaries
        trailing = []
    else:
        leading = boundaries[:1]
        # Any additional boundary selectors are ordered after the annotated
        # regions.  AFD currently has exactly one such output endpoint.
        trailing = boundaries[1:]
    return leading + interior + trailing


def _format_region(region: tuple[tuple[str, Any], ...]) -> str:
    return " ".join(
        f"{key}{value}"
        for key, value in region
        if value != "-"
    )


def _schedule_html_table(
    table: list[list[_ScheduleSlot]],
    device_groups: list[tuple[int, ...]],
) -> str:
    max_cols = max((len(row) for row in table), default=0)
    lines = ['<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">']
    lines.append('<TR><TD BGCOLOR="#eeeeee">GPU</TD>')
    for col in range(max_cols):
        lines.append(f'<TD BGCOLOR="#eeeeee">t{col}</TD>')
    lines.append("</TR>")
    for row_idx, row in enumerate(table):
        devices = ",".join(str(device) for device in device_groups[row_idx])
        lines.append(f'<TR><TD BGCOLOR="#eeeeee">GPU [{devices}]</TD>')
        col = 0
        while col < len(row):
            slot = row[col]
            if not slot:
                lines.append('<TD WIDTH="72" HEIGHT="36"></TD>')
                col += 1
            else:
                colspan = _slot_width(slot)
                colspan_attr = f' COLSPAN="{colspan}"' if colspan > 1 else ""
                width = 72 * colspan
                if len(slot) == 1:
                    op = slot[0]
                    color = _PASS_COLOR[op.pass_]
                    label = (
                        f"{escape(op.pass_)}<BR/>"
                        f"{escape(_format_region(op.region))} MB{op.mb}"
                    )
                    lines.append(
                        f'<TD BGCOLOR="{color}" WIDTH="{width}" HEIGHT="36"{colspan_attr}>{label}</TD>'
                    )
                else:
                    inner = ['<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">']
                    for op in slot:
                        color = _PASS_COLOR[op.pass_]
                        label = (
                            f"{escape(op.pass_)}<BR/>"
                            f"{escape(_format_region(op.region))} MB{op.mb}"
                        )
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
