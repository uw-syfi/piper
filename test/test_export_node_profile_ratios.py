import csv

import pytest

from scripts.export_node_profile_ratios import (
    build_node_profile_ratios,
    write_node_profile_ratios,
)
from src.dag import TrainingDAG, TrainingDAGNode
from src.profiling import NodeProfile, ProfileConfig, ProfiledTrainingPlan


def _profiled_node(
    uid: str,
    *,
    normal_by_rank: dict[int, tuple[float, ...]],
    interference_free_by_rank: dict[int, tuple[float, ...]],
) -> TrainingDAGNode:
    node = TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={},
        device=[0],
        stream="compute",
    )
    node.profile_times_by_mode = {
        "normal": {
            rank: NodeProfile.from_samples(samples)
            for rank, samples in normal_by_rank.items()
        },
        "interference_free": {
            rank: NodeProfile.from_samples(samples)
            for rank, samples in interference_free_by_rank.items()
        },
    }
    return node


def test_build_node_profile_ratios_pools_ranks_and_sorts_descending(tmp_path) -> None:
    dag = TrainingDAG()
    dag.add_node(
        _profiled_node(
            "high_ratio",
            normal_by_rank={0: (10.0, 14.0), 2: (18.0, 22.0)},
            interference_free_by_rank={0: (2.0, 4.0), 2: (6.0, 8.0)},
        )
    )
    dag.add_node(
        _profiled_node(
            "unit_ratio",
            normal_by_rank={0: (2.0, 4.0)},
            interference_free_by_rank={0: (2.0, 4.0)},
        )
    )
    skipped = TrainingDAGNode(
        uid="order",
        node_kind="ORDER",
        compute_subkind=None,
        tag={},
        device=[0],
        stream="compute",
    )
    skipped.profile_times_by_mode = {
        "normal": {0: NodeProfile.skipped("not executed")},
        "interference_free": {0: NodeProfile.skipped("not executed")},
    }
    dag.add_node(skipped)
    plan = ProfiledTrainingPlan.from_dags(
        [dag],
        profile_configs={
            "normal": ProfileConfig(),
            "interference_free": ProfileConfig(),
        },
    )

    rows, skipped_count = build_node_profile_ratios(plan)

    assert [row.node_uid for row in rows] == ["high_ratio", "unit_ratio"]
    assert rows[0].median_normal_time_us == pytest.approx(16.0)
    assert rows[0].median_interference_free_time_us == pytest.approx(5.0)
    assert rows[0].normal_to_interference_free_ratio == pytest.approx(3.2)
    assert skipped_count == 1

    output_path = tmp_path / "ratios.csv"
    write_node_profile_ratios(output_path, rows)
    with output_path.open(newline="") as output_file:
        csv_rows = list(csv.DictReader(output_file))
    assert [row["node_uid"] for row in csv_rows] == ["high_ratio", "unit_ratio"]
