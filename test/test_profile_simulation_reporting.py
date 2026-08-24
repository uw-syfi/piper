from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

_EXAMPLES = str(Path(__file__).resolve().parents[1] / "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from examples.test_harness import _add_profile_comparison_metrics, _metrics_rows
from scripts.profile_and_replay.simulate_step_times import _comparison_dict
from src.dag import TrainingDAG, TrainingDAGNode
from src.profiling import NodeProfile, ProfiledTrainingPlan
from src.simulator import simulate_profile_modes


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
            "interference_free": {
                0: NodeProfile.from_samples((9.0, 50.0, 990.0))
            },
        },
    )
    dag = TrainingDAG()
    dag.add_node(node)
    plan = ProfiledTrainingPlan.from_dags([dag])
    return simulate_profile_modes(plan), plan


def test_harness_reports_mean_endpoints_and_ratio_of_means_savings() -> None:
    comparison, plan = _comparison()
    metrics = {
        "program_rank": 0,
        "batch_size": 1,
        "num_microbatches": 1,
        "seq_len": 1,
    }

    _add_profile_comparison_metrics(metrics, comparison, plan)
    row = _metrics_rows([metrics])[0]

    assert metrics["simulated_normal_step_time_us"] == pytest.approx(370.0)
    assert metrics["simulated_normal_statistics_us"]["median"] == pytest.approx(
        100.0
    )
    assert metrics["dependency_replay_absolute_savings_us"] == pytest.approx(
        61.0 / 3.0
    )
    assert metrics["dependency_replay_interference_reduction_pct"] == pytest.approx(
        100.0 * 61.0 / 1110.0
    )
    assert "dependency_replay_interference_reduction_samples_pct" not in metrics
    assert row["simulated_normal_step_time_estimator"] == "mean"
    assert row["simulated_normal_step_time_mean_s"] == pytest.approx(370e-6)
    assert row["simulated_normal_step_time_median_s"] == pytest.approx(100e-6)
    assert row["simulated_normal_step_time_ci95_low_s"] < 370e-6
    assert row["simulated_normal_step_time_ci95_high_s"] > 370e-6


def test_offline_payload_matches_harness_and_omits_bootstrap_outputs() -> None:
    comparison, _plan = _comparison()

    payload = _comparison_dict(comparison)

    assert payload["statistics_us"]["normal"]["mean"] == pytest.approx(370.0)
    assert payload["statistics_us"]["normal"]["median"] == pytest.approx(100.0)
    assert payload["dependency_replay_absolute_savings_us"] == pytest.approx(
        61.0 / 3.0
    )
    assert payload["dependency_replay_savings_statistics"][
        "percent"
    ] == pytest.approx(100.0 * 61.0 / 1110.0)
    assert "dependency_replay_interference_reduction_samples_pct" not in payload
    assert "dependency_replay_interference_reduction_statistics" not in payload


def test_harness_reports_unavailable_single_sample_interval() -> None:
    statistics = SimpleNamespace(
        estimate=10.0,
        to_dict=lambda: {
            "samples": 1,
            "estimator": "mean",
            "estimate": 10.0,
            "mean": 10.0,
            "median": 10.0,
            "std": None,
            "ci95_low": None,
            "ci95_high": None,
            "error_pct": None,
        },
    )
    savings = SimpleNamespace(
        absolute_mean=2.0,
        percent_standard_error=None,
        percent_ci95_low=None,
        percent_ci95_high=None,
    )
    comparison = SimpleNamespace(
        interference_free_statistics=statistics,
        normal_statistics=statistics,
        interference_free_samples_us=(10.0,),
        normal_samples_us=(10.0,),
        normal_prediction_error=None,
        predicted_interference_reduction_pct=None,
        dependency_replay_interference_reduction_pct=20.0,
        dependency_replay_savings_statistics=savings,
        instrumented_policy_reduction_pct=None,
        profile_event_overhead_pct=None,
        mode_profile_event_overhead_pct=None,
        observed_interference_reduction_pct=None,
        envelope_modeled_interference_reduction_pct=None,
        normal_dependency_replay_excess_pct=None,
        observed_normal_envelope_samples_us=(),
        instrumented_normal_replay_error=None,
        instrumented_interference_free_replay_error=None,
    )
    plan = SimpleNamespace(profile_metadata={})
    metrics = {}

    _add_profile_comparison_metrics(metrics, comparison, plan)

    assert metrics["simulated_normal_statistics_us"]["std"] is None
    assert metrics["simulated_normal_statistics_us"]["ci95_low"] is None
    assert metrics[
        "dependency_replay_interference_reduction_standard_error_pct"
    ] is None
