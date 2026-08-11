from types import SimpleNamespace

import pytest

from src.attribution import attribute_profile_mode_savings
from src.dag import TrainingDAG, TrainingDAGNode
from src.profiling import NodeProfile


def _profile(duration_us: float):
    return SimpleNamespace(estimate_us=duration_us)


def _node(
    uid: str,
    *,
    normal_us: float,
    interference_free_us: float,
    stream: str,
) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={},
        device=[0],
        stream=stream,
        profile_times_by_mode={
            "normal": {0: _profile(normal_us)},
            "interference_free": {0: _profile(interference_free_us)},
        },
    )


def _dag(*nodes: TrainingDAGNode) -> TrainingDAG:
    dag = TrainingDAG()
    for node in nodes:
        dag.add_node(node)
    return dag


def test_sequential_savings_are_fully_exposed_and_additive() -> None:
    result = attribute_profile_mode_savings(
        [
            _dag(
                _node("a", normal_us=10.0, interference_free_us=4.0, stream="s"),
                _node("b", normal_us=8.0, interference_free_us=6.0, stream="s"),
            )
        ],
        permutations=4,
        seed=3,
    )

    by_uid = {item.feature.uids[0]: item for item in result.features}
    assert result.normal_step_time_us == pytest.approx(18.0)
    assert result.interference_free_step_time_us == pytest.approx(10.0)
    assert result.attributed_step_savings_us == pytest.approx(8.0)
    assert result.additivity_residual_us == pytest.approx(0.0)
    assert by_uid["a"].attributed_savings_us == pytest.approx(6.0)
    assert by_uid["b"].attributed_savings_us == pytest.approx(2.0)
    assert by_uid["a"].positive_exposure_probability == pytest.approx(1.0)


def test_parallel_bottleneck_splits_exposed_and_hidden_savings() -> None:
    result = attribute_profile_mode_savings(
        [
            _dag(
                _node("a", normal_us=10.0, interference_free_us=4.0, stream="a"),
                _node("b", normal_us=8.0, interference_free_us=7.0, stream="b"),
            )
        ],
        permutations=2,
        seed=0,
        antithetic=True,
        exposure_threshold_us=0.1,
    )

    by_uid = {item.feature.uids[0]: item for item in result.features}
    assert result.normal_step_time_us == pytest.approx(10.0)
    assert result.interference_free_step_time_us == pytest.approx(7.0)
    # The raw node-duration deltas sum to 7 us, but the overlap-aware
    # counterfactual makespan delta is only 3 us.
    assert sum(
        item.feature.raw_profile_savings_us for item in result.features
    ) == pytest.approx(7.0)
    assert result.attributed_step_savings_us == pytest.approx(3.0)
    assert result.net_exposed_step_savings_us == pytest.approx(3.0)
    assert result.net_exposed_step_savings_pct == pytest.approx(30.0)
    assert result.exposed_savings_validation_residual_us == pytest.approx(0.0)
    assert by_uid["a"].attributed_savings_us == pytest.approx(2.5)
    assert by_uid["b"].attributed_savings_us == pytest.approx(0.5)
    assert by_uid["a"].positive_exposure_probability == pytest.approx(1.0)
    assert by_uid["b"].positive_exposure_probability == pytest.approx(0.5)
    assert by_uid["b"].hidden_probability == pytest.approx(0.5)


def test_signed_exposed_deltas_include_negative_counterfactual_marginals() -> None:
    result = attribute_profile_mode_savings(
        [
            _dag(
                _node("faster", normal_us=10.0, interference_free_us=4.0, stream="s"),
                _node("slower", normal_us=2.0, interference_free_us=5.0, stream="s"),
            )
        ],
        permutations=2,
    )

    assert result.positive_exposed_step_savings_us == pytest.approx(6.0)
    assert result.negative_exposed_step_savings_us == pytest.approx(-3.0)
    assert result.net_exposed_step_savings_us == pytest.approx(3.0)
    assert result.total_step_savings_us == pytest.approx(3.0)
    assert result.exposed_savings_validation_residual_us == pytest.approx(0.0)


def test_attribution_uses_the_reported_median_of_replay_samples() -> None:
    a = _node("a", normal_us=0.0, interference_free_us=0.0, stream="a")
    b = _node("b", normal_us=0.0, interference_free_us=0.0, stream="b")
    a.profile_times_by_mode = {
        "normal": {0: NodeProfile.from_samples((10.0, 1.0))},
        "interference_free": {0: NodeProfile.from_samples((4.0, 1.0))},
    }
    b.profile_times_by_mode = {
        "normal": {0: NodeProfile.from_samples((1.0, 10.0))},
        "interference_free": {0: NodeProfile.from_samples((1.0, 4.0))},
    }

    result = attribute_profile_mode_savings(
        [_dag(a, b)], permutations=2, exposure_threshold_us=0.0
    )

    # Replaying each aligned iteration yields (10, 10) -> (4, 4).  Replaying
    # only the per-node median estimates would incorrectly yield 5.5 -> 2.5.
    assert result.step_time_estimator == "median"
    assert result.sample_iteration_ids == (0, 1)
    assert result.normal_step_time_us == pytest.approx(10.0)
    assert result.interference_free_step_time_us == pytest.approx(4.0)
    assert result.net_exposed_step_savings_us == pytest.approx(6.0)
