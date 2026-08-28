from pathlib import Path
import sys

import pytest

_EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from examples.test_harness import _add_profile_comparison_metrics, _metrics_rows
from scripts.profile_and_replay.replay_step_times import _comparison_dict
from src.dag import TrainingDAG, TrainingDAGNode
from src.profiling import NodeProfile, ProfiledTrainingPlan
from src.replay import replay_profile_modes


def _comparison():
    node = TrainingDAGNode(
        uid="compute",
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={},
        device=[0],
        stream="compute",
        profile_times_by_mode={
            "normal": {0: NodeProfile.from_samples((10.0, 100.0, 1000.0))},
            "isolated": {
                0: NodeProfile.from_samples((9.0, 50.0, 990.0))
            },
        },
    )
    dag = TrainingDAG()
    dag.add_node(node)
    plan = ProfiledTrainingPlan.from_dags([dag])
    return replay_profile_modes(plan), plan


def test_harness_reports_only_public_replay_metrics() -> None:
    comparison, _plan = _comparison()
    metrics = {
        "program_rank": 0,
        "batch_size": 1,
        "num_microbatches": 1,
        "seq_len": 1,
    }

    _add_profile_comparison_metrics(metrics, comparison)
    row = _metrics_rows([metrics])[0]

    assert metrics["normal_replay_step_time_us"] == pytest.approx(370.0)
    assert metrics["perfect_overlap_replay_step_time_us"] == pytest.approx(
        1049.0 / 3.0
    )
    assert metrics["interference_overhead_pct"] == pytest.approx(
        100.0 * 61.0 / 1110.0
    )
    assert metrics["prediction_error_pct"] is None
    assert row["normal_replay_step_time_estimator"] == "mean"
    assert row["normal_replay_step_time_mean_s"] == pytest.approx(370e-6)
    assert row["normal_replay_step_time_median_s"] == pytest.approx(100e-6)
    assert row["perfect_overlap_replay_step_time_mean_s"] == pytest.approx(
        (1049.0 / 3.0) * 1e-6
    )


def test_offline_payload_contains_exactly_five_public_outputs() -> None:
    comparison, _plan = _comparison()

    payload = _comparison_dict(comparison)

    assert set(payload) == {
        "actual_step_time_us",
        "normal_replay_step_time_us",
        "perfect_overlap_replay_step_time_us",
        "prediction_error_pct",
        "interference_overhead_pct",
    }
    assert payload["actual_step_time_us"] is None
    assert payload["normal_replay_step_time_us"] == pytest.approx(370.0)
    assert payload["perfect_overlap_replay_step_time_us"] == pytest.approx(
        1049.0 / 3.0
    )
