#!/usr/bin/env python3
"""Export per-node normal/interference-free profile-time ratios to CSV."""

from __future__ import annotations

import argparse
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
    INTERFERENCE_FREE_PROFILE_MODE,
    NORMAL_PROFILE_MODE,
    ProfiledTrainingPlan,
)


CSV_FIELDS = (
    "node_uid",
    "median_normal_time_us",
    "median_interference_free_time_us",
    "normal_to_interference_free_ratio",
)


@dataclass(frozen=True)
class NodeProfileRatio:
    node_uid: str
    median_normal_time_us: float
    median_interference_free_time_us: float
    normal_to_interference_free_ratio: float

    def to_csv_row(self) -> dict[str, str | float]:
        return {
            "node_uid": self.node_uid,
            "median_normal_time_us": self.median_normal_time_us,
            "median_interference_free_time_us": (
                self.median_interference_free_time_us
            ),
            "normal_to_interference_free_ratio": (
                self.normal_to_interference_free_ratio
            ),
        }


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
    """Build descending normal/interference-free ratios for measured nodes.

    Nodes with no samples in either mode are non-executing DAG bookkeeping
    nodes and are omitted. Missing samples in only one mode indicate an
    incomplete profile and fail explicitly.
    """

    required_modes = {NORMAL_PROFILE_MODE, INTERFERENCE_FREE_PROFILE_MODE}
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
            interference_free_samples = _samples_for_mode(
                node, INTERFERENCE_FREE_PROFILE_MODE
            )
            if not normal_samples and not interference_free_samples:
                skipped_nodes += 1
                continue
            if not normal_samples or not interference_free_samples:
                missing_mode = (
                    NORMAL_PROFILE_MODE
                    if not normal_samples
                    else INTERFERENCE_FREE_PROFILE_MODE
                )
                raise ValueError(
                    f"node {node.uid!r} has no executed samples for {missing_mode!r}"
                )

            normal_median = float(statistics.median(normal_samples))
            interference_free_median = float(
                statistics.median(interference_free_samples)
            )
            ratio = (
                normal_median / interference_free_median
                if interference_free_median > 0.0
                else math.inf
            )
            rows.append(
                NodeProfileRatio(
                    node_uid=node.uid,
                    median_normal_time_us=normal_median,
                    median_interference_free_time_us=interference_free_median,
                    normal_to_interference_free_ratio=ratio,
                )
            )

    rows.sort(key=lambda row: (-row.normal_to_interference_free_ratio, row.node_uid))
    return rows, skipped_nodes


def write_node_profile_ratios(
    output_path: Path, rows: list[NodeProfileRatio]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.to_csv_row() for row in rows)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.profile.with_name("node_profile_time_ratios.csv")
    plan = ProfiledTrainingPlan.load_json(args.profile)
    rows, skipped_nodes = build_node_profile_ratios(plan)
    write_node_profile_ratios(output_path, rows)
    print(
        f"Wrote {len(rows)} measured nodes to {output_path} "
        f"(skipped {skipped_nodes} nodes without executed timing samples)"
    )


if __name__ == "__main__":
    main()
