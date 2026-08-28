#!/usr/bin/env python3
"""Export per-node normal/isolated profile-time ratios to CSV."""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from src.dag import TrainingDAGNode
from src.profiling import (
    ISOLATED_PROFILE_MODE,
    NORMAL_PROFILE_MODE,
    ProfiledTrainingPlan,
)


CSV_FIELDS = (
    "node_uid",
    "median_normal_time_us",
    "median_isolated_time_us",
    "normal_to_isolated_ratio",
)
SUMMARY_CSV_FIELDS = (
    "node_type",
    "node_count",
    "average_normal_to_isolated_ratio",
)


_NODE_KIND_TYPES = {
    "A2A_COMM": "a2a",
    "RECV_COMM": "recv",
    "REDUCE_COMM": "reduce",
    "SEND_COMM": "send",
    "UPD": "update",
}
_COMPUTE_SUBKIND_TYPES = {
    "FWD": "forward",
    "BWD": "backward",
    "BWD_I": "backward_input",
    "BWD_W": "backward_weight",
}


@dataclass(frozen=True)
class NodeProfileRatio:
    node_uid: str
    node_type: str
    median_normal_time_us: float
    median_isolated_time_us: float
    normal_to_isolated_ratio: float

    def to_csv_row(self) -> dict[str, str | float]:
        return {
            "node_uid": self.node_uid,
            "median_normal_time_us": self.median_normal_time_us,
            "median_isolated_time_us": (
                self.median_isolated_time_us
            ),
            "normal_to_isolated_ratio": (
                self.normal_to_isolated_ratio
            ),
        }


@dataclass(frozen=True)
class NodeTypeSummary:
    node_type: str
    node_count: int
    average_normal_to_isolated_ratio: float

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "node_type": self.node_type,
            "node_count": self.node_count,
            "average_normal_to_isolated_ratio": (
                self.average_normal_to_isolated_ratio
            ),
        }


@dataclass(frozen=True)
class NodeProfileRatioExportResult:
    ratio_csv_path: Path
    summary_csv_path: Path
    measured_node_count: int
    skipped_node_count: int
    summaries: tuple[NodeTypeSummary, ...]


def node_type(node: TrainingDAGNode) -> str:
    """Return the stable summary category for a finalized DAG node."""

    if node.node_kind == "COMPUTE":
        return _COMPUTE_SUBKIND_TYPES.get(
            str(node.compute_subkind),
            f"compute_{str(node.compute_subkind).lower()}",
        )
    return _NODE_KIND_TYPES.get(node.node_kind, node.node_kind.lower())


def _samples_for_mode(node: TrainingDAGNode, mode: str) -> tuple[float, ...]:
    """Pool this node's executed raw samples across profiled global ranks."""

    return tuple(
        float(sample)
        for global_rank in sorted(node.profile_times_by_mode.get(mode, {}))
        for profile in (node.profile_times_by_mode[mode][global_rank],)
        if profile.executed
        for sample in profile.samples_us
    )


def build_node_profile_ratios(
    plan: ProfiledTrainingPlan,
) -> tuple[list[NodeProfileRatio], int]:
    """Build descending normal/isolated ratios for measured nodes.

    Nodes with no samples in either mode are non-executing DAG bookkeeping
    nodes and are omitted. Missing samples in only one mode indicate an
    incomplete profile and fail explicitly.
    """

    required_modes = {NORMAL_PROFILE_MODE, ISOLATED_PROFILE_MODE}
    missing_modes = required_modes - set(plan.available_profile_modes)
    if missing_modes:
        raise ValueError(
            "profile artifact is missing required mode(s): "
            + ", ".join(sorted(missing_modes))
        )

    rows: list[NodeProfileRatio] = []
    skipped_nodes = 0
    for dag in plan.per_pp_dags:
        for node in dag.nodes.values():
            normal_samples = _samples_for_mode(node, NORMAL_PROFILE_MODE)
            isolated_samples = _samples_for_mode(
                node, ISOLATED_PROFILE_MODE
            )
            if not normal_samples and not isolated_samples:
                skipped_nodes += 1
                continue
            if not normal_samples or not isolated_samples:
                missing_mode = (
                    NORMAL_PROFILE_MODE
                    if not normal_samples
                    else ISOLATED_PROFILE_MODE
                )
                raise ValueError(
                    f"node {node.uid!r} has no executed samples for {missing_mode!r}"
                )

            normal_median = float(statistics.median(normal_samples))
            isolated_median = float(
                statistics.median(isolated_samples)
            )
            ratio = (
                normal_median / isolated_median
                if isolated_median > 0.0
                else math.inf
            )
            rows.append(
                NodeProfileRatio(
                    node_uid=node.uid,
                    node_type=node_type(node),
                    median_normal_time_us=normal_median,
                    median_isolated_time_us=isolated_median,
                    normal_to_isolated_ratio=ratio,
                )
            )

    rows.sort(key=lambda row: (-row.normal_to_isolated_ratio, row.node_uid))
    return rows, skipped_nodes


def summarize_node_types(
    rows: list[NodeProfileRatio],
) -> list[NodeTypeSummary]:
    """Average per-node ratios within each semantic node type."""

    ratios_by_type: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        ratios_by_type[row.node_type].append(
            row.normal_to_isolated_ratio
        )
    summaries = [
        NodeTypeSummary(
            node_type=type_name,
            node_count=len(ratios),
            average_normal_to_isolated_ratio=statistics.fmean(ratios),
        )
        for type_name, ratios in ratios_by_type.items()
    ]
    summaries.sort(
        key=lambda summary: (
            -summary.average_normal_to_isolated_ratio,
            summary.node_type,
        )
    )
    return summaries


def write_node_profile_ratios(
    output_path: Path, rows: list[NodeProfileRatio]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.to_csv_row() for row in rows)


def write_node_type_summaries(
    output_path: Path, summaries: list[NodeTypeSummary]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=SUMMARY_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary.to_csv_row() for summary in summaries)


def export_node_profile_ratio_csvs(
    profile_path: Path,
    *,
    output_path: Path | None = None,
    summary_output_path: Path | None = None,
) -> NodeProfileRatioExportResult:
    """Export sorted per-node ratios and their node-type summary beside a plan."""

    profile_path = Path(profile_path)
    output_path = output_path or profile_path.with_name(
        "node_profile_time_ratios.csv"
    )
    summary_output_path = summary_output_path or output_path.with_name(
        f"{output_path.stem}_summary.csv"
    )
    plan = ProfiledTrainingPlan.load_json(profile_path)
    rows, skipped_nodes = build_node_profile_ratios(plan)
    summaries = summarize_node_types(rows)
    write_node_profile_ratios(output_path, rows)
    write_node_type_summaries(summary_output_path, summaries)
    return NodeProfileRatioExportResult(
        ratio_csv_path=output_path,
        summary_csv_path=summary_output_path,
        measured_node_count=len(rows),
        skipped_node_count=skipped_nodes,
        summaries=tuple(summaries),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="ProfiledTrainingPlan JSON artifact")
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output CSV path; defaults to node_profile_time_ratios.csv next to "
            "the input artifact"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help=(
            "Node-type summary CSV path; defaults to <output_stem>_summary.csv"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.profile.with_name("node_profile_time_ratios.csv")
    summary_output_path = args.summary_output or output_path.with_name(
        f"{output_path.stem}_summary.csv"
    )
    result = export_node_profile_ratio_csvs(
        args.profile,
        output_path=output_path,
        summary_output_path=summary_output_path,
    )
    print(
        f"Wrote {result.measured_node_count} measured nodes to "
        f"{result.ratio_csv_path} (skipped {result.skipped_node_count} nodes "
        "without executed timing samples)"
    )
    print(
        f"Wrote {len(result.summaries)} node-type summaries to "
        f"{result.summary_csv_path}"
    )
    print("Node type summary (average normal/isolated ratio):")
    for summary in result.summaries:
        print(
            f"  {summary.node_type}: "
            f"{summary.average_normal_to_isolated_ratio:.6f}x "
            f"(nodes={summary.node_count})"
        )


if __name__ == "__main__":
    main()
