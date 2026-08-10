from types import SimpleNamespace

import pytest

from src.attribution import attribute_profile_mode_savings
from src.dag import TrainingDAG, TrainingDAGNode
from src.profile_validation import (
    ProfileAndSimulateValidationError,
    build_and_validate_attributed_trace,
    validate_kineto_profile_results,
    validate_measured_normal_samples,
    validate_simulation_endpoints,
)
from src.profiling import ActorProfileResult, NodeProfile, ProfiledTrainingPlan
from src.piper import simulate_profiled_modes
from src.piper import (
    _compare_kineto_activity_signatures,
    _filter_kineto_profile_iterations,
)
from src.simulator import simulate_profile_modes


def _profile(duration_us: float) -> NodeProfile:
    return NodeProfile.from_samples((duration_us, duration_us + 2.0))


def _plan() -> ProfiledTrainingPlan:
    dag = TrainingDAG()
    dag.add_node(
        TrainingDAGNode(
            uid="compute",
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag={},
            device=[0],
            stream="default_stream",
            profile_times_by_mode={
                "normal": {0: _profile(9.0)},
                "interference_free": {0: _profile(4.0)},
            },
        )
    )
    return ProfiledTrainingPlan.from_dags([dag])


def test_endpoint_and_trace_validations_preserve_the_normal_timeline() -> None:
    plan = _plan()
    comparison = simulate_profile_modes(plan)
    attribution = attribute_profile_mode_savings(
        plan, permutations=2, antithetic=True
    )

    endpoint = validate_simulation_endpoints(comparison)
    trace, rendered = build_and_validate_attributed_trace(
        comparison.normal, attribution
    )

    assert endpoint["passed"] is True
    assert rendered["passed"] is True
    assert rendered["contains_cuda_kernels"] is False
    assert rendered["source_events"] == rendered["rendered_events"] == 1
    assert len([event for event in trace["traceEvents"] if event["ph"] == "X"]) == 1
    assert rendered["net_exposed_step_savings_us"] == pytest.approx(5.0)
    assert rendered["simulator_step_savings_us"] == pytest.approx(5.0)
    assert rendered["exposed_savings_validation_residual_us"] == pytest.approx(0.0)
    assert rendered["trace_checksum_residual_us"] == pytest.approx(0.0)


def test_kineto_validation_rejects_cross_mode_activity_mismatch() -> None:
    profile = NodeProfile.from_samples((10.0,))
    audit = {
        "range_count": 1,
        "launch_count": 1,
        "matched_launch_count": 1,
        "unmatched_launch_count": 0,
        "simulator_duration_source": "correlated_gpu_activity_envelope",
        "simulator_duration_samples_us": [10.0],
        "range_activity_signatures": [
            {
                "activity_time_sum_us": 8.0,
                "activity_union_time_us": 8.0,
                "activity_intra_node_overlap_us": 0.0,
                "activity_envelope_time_us": 10.0,
                "activity_idle_gap_us": 2.0,
            }
        ],
    }
    result = ActorProfileResult(
        graph_fingerprint="unused",
        global_rank=0,
        profile_mode="normal",
        node_profiles={"compute": profile},
        device_metadata={"kineto_attribution_audit": {"compute": audit}},
    )

    with pytest.raises(
        ProfileAndSimulateValidationError,
        match="activity identity differs",
    ):
        validate_kineto_profile_results(
            [result],
            {
                "exact_match": False,
                "mismatched_node_ranges": 1,
            },
        )


def test_kineto_comparison_rejects_same_multiset_in_different_order() -> None:
    def result(mode: str, sequence_hash: str):
        occurrence = {
            "activity_count": 2,
            "signature_sha256": "same-multiset",
            "activity_sequence_sha256": sequence_hash,
            "activities": [
                {"category": "kernel", "name": "a", "count": 1},
                {"category": "kernel", "name": "b", "count": 1},
            ],
            "activity_sequence": [],
        }
        return SimpleNamespace(
            profile_mode=mode,
            global_rank=0,
            device_metadata={
                "kineto_attribution_audit": {
                    "compute": {
                        "range_activity_signatures": [occurrence]
                    }
                }
            },
        )

    summary = _compare_kineto_activity_signatures(
        [result("normal", "ab"), result("interference_free", "ba")]
    )

    assert summary["exact_match"] is False
    assert summary["mismatched_node_ranges"] == 1
    mismatch = summary["mismatches"][0]["ranges"][0]
    assert mismatch["different_activity_entries"] == 0
    assert mismatch["activity_order_match"] is False


def test_kineto_iteration_filter_keeps_paired_samples_and_audits() -> None:
    profile = NodeProfile.from_samples((10.0, 20.0, 30.0))
    signatures = [
        {
            "activity_count": 1,
            "signature_sha256": str(index),
            "activity_sequence_sha256": str(index),
        }
        for index in range(3)
    ]
    result = ActorProfileResult(
        graph_fingerprint="unused",
        global_rank=0,
        profile_mode="normal",
        node_profiles={"compute": profile},
        device_metadata={
            "kineto_attribution_audit": {
                "compute": {
                    "range_count": 3,
                    "range_activity_signatures": signatures,
                    "simulator_duration_samples_us": [10.0, 20.0, 30.0],
                }
            }
        },
    )

    filtered = _filter_kineto_profile_iterations([result], (0, 2))[0]

    assert filtered.node_profiles["compute"].samples_us == (10.0, 30.0)
    assert filtered.node_profiles["compute"].sample_iteration_ids == (0, 1)
    audit = filtered.device_metadata["kineto_attribution_audit"]["compute"]
    assert audit["range_count"] == 2
    assert audit["simulator_duration_samples_us"] == [10.0, 30.0]
    assert audit["range_activity_signatures"] == [signatures[0], signatures[2]]


def test_measured_normal_validation_uses_uninstrumented_truth_and_warns() -> None:
    plan = _plan()
    plan.profile_metadata[
        "instrumented_global_step_samples_us_by_mode"
    ] = {"normal": [8.0, 10.0]}
    comparison = simulate_profile_modes(
        plan, actual_step_samples_us=(8.0, 10.0)
    )

    result = validate_measured_normal_samples(plan, comparison)

    assert result["measured_normal_step_estimate_us"] == pytest.approx(9.0)
    assert result["simulated_normal_step_estimate_us"] == pytest.approx(10.0)
    assert result["median_absolute_step_error_pct"] == pytest.approx(100 / 9)
    instrumentation = result["kineto_cupti_instrumentation_error"]
    assert instrumentation["signed_error_pct"] == pytest.approx(0.0)
    assert "exceeds" in result["warning"]


def test_simulate_profiled_modes_runs_all_automatic_validations() -> None:
    plan = _plan()
    plan.profile_metadata[
        "instrumented_global_step_samples_us_by_mode"
    ] = {"normal": [9.0, 11.0]}

    comparison = simulate_profiled_modes(
        plan, actual_step_samples_s=(9e-6, 11e-6)
    )

    validations = plan.profile_metadata["automatic_validations"]
    assert comparison.normal_samples_us == pytest.approx((9.0, 11.0))
    assert validations["simulation_endpoints"]["passed"] is True
    assert validations["simulation_endpoints"]["additivity_residual_us"] == 0.0
    assert validations["attributed_trace_losslessness"]["passed"] is True
    measured = validations["measured_vs_simulated_normal"]
    assert measured["available"] is True
    assert measured["median_absolute_step_error_pct"] == 0.0
    assert measured["warning"] is None
