from types import SimpleNamespace

import pytest

from src.attribution import attribute_profile_mode_savings
from src.dag import TrainingDAG, TrainingDAGNode
from src.profile_validation import (
    ProfileAndReplayValidationError,
    build_and_validate_attributed_trace,
    validate_kineto_profile_results,
    validate_measured_normal_samples,
    validate_replay_endpoints,
)
from src.profiling import ActorProfileResult, NodeProfile, ProfiledTrainingPlan
from src.piper import replay_profiled_modes
from src.piper import (
    _compare_kineto_activity_signatures,
    _filter_kineto_profile_iterations,
    _strip_kineto_comparison_details,
)
from src.replay import replay_profile_modes


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
                "isolated": {0: _profile(4.0)},
            },
        )
    )
    return ProfiledTrainingPlan.from_dags([dag])


def test_endpoint_and_trace_validations_preserve_the_normal_timeline() -> None:
    plan = _plan()
    comparison = replay_profile_modes(plan)
    attribution = attribute_profile_mode_savings(
        plan, permutations=2, antithetic=True
    )

    endpoint = validate_replay_endpoints(comparison)
    trace, rendered = build_and_validate_attributed_trace(
        comparison.normal, attribution
    )

    assert endpoint["passed"] is True
    assert rendered["passed"] is True
    assert rendered["contains_cuda_kernels"] is False
    assert rendered["source_events"] == rendered["rendered_events"] == 1
    assert len([event for event in trace["traceEvents"] if event["ph"] == "X"]) == 1
    assert rendered["net_exposed_step_savings_us"] == pytest.approx(5.0)
    assert rendered["replay_step_savings_us"] == pytest.approx(5.0)
    assert rendered["exposed_savings_validation_residual_us"] == pytest.approx(0.0)
    assert rendered["trace_checksum_residual_us"] == pytest.approx(0.0)


def test_kineto_validation_rejects_cross_mode_activity_mismatch() -> None:
    profile = NodeProfile.from_samples((10.0,))
    audit = {
        "range_count": 1,
        "launch_count": 1,
        "matched_launch_count": 1,
        "unmatched_launch_count": 0,
        "replay_duration_source": "correlated_gpu_activity_envelope",
        "replay_duration_samples_us": [10.0],
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
        ProfileAndReplayValidationError,
        match="activity identity differs",
    ):
        validate_kineto_profile_results(
            [result],
            {
                "exact_match": False,
                "mismatched_node_ranges": 1,
            },
        )


def test_kineto_validation_accepts_only_classified_launch_loss() -> None:
    occurrence = {
        "activity_count": 0,
        "activity_time_sum_us": 0.0,
        "activity_union_time_us": 0.0,
        "activity_intra_node_overlap_us": 0.0,
        "activity_envelope_time_us": 0.0,
        "activity_idle_gap_us": 0.0,
    }
    result = ActorProfileResult(
        graph_fingerprint="unused",
        global_rank=0,
        profile_mode="normal",
        node_profiles={"compute": NodeProfile.from_samples((0.0,))},
        device_metadata={
            "kineto_attribution_audit": {
                "compute": {
                    "range_count": 1,
                    "launch_count": 1,
                    "matched_launch_count": 0,
                    "unmatched_launch_count": 1,
                    "replay_duration_source": (
                        "correlated_gpu_activity_envelope"
                    ),
                    "replay_duration_samples_us": [0.0],
                    "range_activity_signatures": [occurrence],
                }
            }
        },
    )
    accepted = {
        "exact_match": False,
        "accepted": True,
        "capture_loss_only": True,
        "capture_loss_within_tolerance": True,
        "capture_loss_rate": 0.0001,
        "capture_loss_threshold": 0.0001,
        "expected_gpu_activity_records": 10_000,
        "missing_gpu_activity_records": 1,
        "mismatched_node_ranges": 1,
    }

    report = validate_kineto_profile_results([result], accepted)

    assert report["passed"] is True
    assert report["accepted_capture_loss"] is True
    assert report["capture_loss_rate"] == pytest.approx(0.0001)

    with pytest.raises(
        ProfileAndReplayValidationError, match="launch coverage 0/1"
    ):
        validate_kineto_profile_results(
            [result],
            {"exact_match": True, "accepted": True},
        )


def _collective_activity_result(
    rank: int,
    uid: str,
    activity_count: int,
    *,
    node_kind: str,
    comm_id: str | None = "transaction",
) -> ActorProfileResult:
    duration = 1.0 if activity_count else 0.0
    occurrence = {
        "activity_count": activity_count,
        "activity_time_sum_us": duration,
        "activity_union_time_us": duration,
        "activity_intra_node_overlap_us": 0.0,
        "activity_envelope_time_us": duration,
        "activity_idle_gap_us": 0.0,
    }
    audit = {
        "node_kind": node_kind,
        "comm_id": comm_id,
        "range_count": 1,
        "launch_count": activity_count,
        "matched_launch_count": activity_count,
        "unmatched_launch_count": 0,
        "replay_duration_source": "correlated_gpu_activity_envelope",
        "replay_duration_samples_us": [duration],
        "range_activity_signatures": [occurrence],
    }
    return ActorProfileResult(
        graph_fingerprint="unused",
        global_rank=rank,
        profile_mode="normal",
        node_profiles={uid: NodeProfile.from_samples((duration,))},
        device_metadata={"kineto_attribution_audit": {uid: audit}},
    )


@pytest.mark.parametrize(
    "node_kind", ("A2A_COMM", "M_TO_N_COMM", "N_TO_M_COMM")
)
def test_kineto_validation_accepts_mixed_routed_activity_presence(
    node_kind: str,
) -> None:
    report = validate_kineto_profile_results(
        [
            _collective_activity_result(
                0, "source", 1, node_kind=node_kind
            ),
            _collective_activity_result(
                1, "destination", 0, node_kind=node_kind
            ),
        ],
        {"exact_match": True},
    )

    assert report["passed"] is True


def test_kineto_validation_still_rejects_mixed_nonrouted_activity_presence() -> None:
    with pytest.raises(
        ProfileAndReplayValidationError,
        match="mix zero and nonzero GPU activity",
    ):
        validate_kineto_profile_results(
            [
                _collective_activity_result(
                    0, "reduce.0", 1, node_kind="REDUCE_COMM"
                ),
                _collective_activity_result(
                    1, "reduce.1", 0, node_kind="REDUCE_COMM"
                ),
            ],
            {"exact_match": True},
        )


def test_kineto_validation_requires_a2a_comm_id() -> None:
    with pytest.raises(
        ProfileAndReplayValidationError, match="collective audit has no comm_id"
    ):
        validate_kineto_profile_results(
            [
                _collective_activity_result(
                    0, "a2a", 1, node_kind="A2A_COMM", comm_id=None
                )
            ],
            {"exact_match": True},
        )


def test_kineto_validation_accepts_all_zero_collective_occurrence() -> None:
    occurrence = {
        "activity_count": 0,
        "activity_time_sum_us": 0.0,
        "activity_union_time_us": 0.0,
        "activity_intra_node_overlap_us": 0.0,
        "activity_envelope_time_us": 0.0,
        "activity_idle_gap_us": 0.0,
    }
    results = []
    for rank in (0, 1):
        uid = f"reduce.{rank}"
        results.append(
            ActorProfileResult(
                graph_fingerprint="unused",
                global_rank=rank,
                profile_mode="normal",
                node_profiles={uid: NodeProfile.from_samples((0.0,))},
                device_metadata={
                    "kineto_attribution_audit": {
                        uid: {
                            "node_kind": "REDUCE_COMM",
                            "comm_id": "bucket.0",
                            "range_count": 1,
                            "launch_count": 0,
                            "matched_launch_count": 0,
                            "unmatched_launch_count": 0,
                            "replay_duration_source": (
                                "correlated_gpu_activity_envelope"
                            ),
                            "replay_duration_samples_us": [0.0],
                            "range_activity_signatures": [occurrence],
                        }
                    }
                },
            )
        )

    report = validate_kineto_profile_results(
        results, {"exact_match": True}
    )
    assert report["passed"] is True


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
        [result("normal", "ab"), result("isolated", "ba")]
    )

    assert summary["exact_match"] is False
    assert summary["mismatched_node_ranges"] == 1
    mismatch = summary["mismatches"][0]["ranges"][0]
    assert mismatch["different_activity_entries"] == 0
    assert mismatch["activity_order_match"] is False


def _launch_result(
    mode: str,
    activities: list[dict[str, str]],
    *,
    rank: int = 0,
    operation: str = "kernel",
    mapping_unambiguous: bool = True,
):
    counts = {}
    for activity in activities:
        key = (activity["category"], activity["name"])
        counts[key] = counts.get(key, 0) + 1
    occurrence = {
        "activity_count": len(activities),
        "signature_sha256": f"signature-{mode}-{rank}-{len(activities)}",
        "activity_sequence_sha256": f"sequence-{mode}-{rank}-{len(activities)}",
        "activities": [
            {"category": category, "name": name, "count": count}
            for (category, name), count in sorted(counts.items())
        ],
        "activity_sequence": list(activities),
        "cpu_launch_sequence": [operation],
        "cpu_launch_sequence_sha256": operation,
        "launch_mapping_unambiguous": mapping_unambiguous,
        "unmapped_activities": (
            [] if mapping_unambiguous else [{"reason": "test_ambiguity"}]
        ),
        "cpu_launches": [{
            "launch_index": 0,
            "operation": operation,
            "api_name": "cudaLaunchKernel",
            "mapping_status": (
                "unambiguous" if mapping_unambiguous else "ambiguous"
            ),
            "ambiguity_reasons": (
                [] if mapping_unambiguous else ["test_ambiguity"]
            ),
            "activities": list(activities),
        }],
    }
    return SimpleNamespace(
        profile_mode=mode,
        global_rank=rank,
        device_metadata={
            "kineto_attribution_audit": {
                "compute": {"range_activity_signatures": [occurrence]}
            }
        },
    )


def _launch_sequence_result(
    mode: str,
    operations: list[str],
    activities: list[dict[str, str]],
    *,
    unmapped_activities: list[dict[str, str]] | None = None,
):
    counts = {}
    for activity in activities:
        key = (activity["category"], activity["name"])
        counts[key] = counts.get(key, 0) + 1
    launches = [
        {
            "launch_index": index,
            "operation": operation,
            "api_name": "cudaLaunchKernel",
            "mapping_status": "unambiguous",
            "ambiguity_reasons": [],
            "activities": (
                [activities[index]] if index < len(activities) else []
            ),
        }
        for index, operation in enumerate(operations)
    ]
    occurrence = {
        "activity_count": len(activities),
        "signature_sha256": f"signature-{len(activities)}",
        "activity_sequence_sha256": f"sequence-{len(activities)}",
        "activities": [
            {"category": category, "name": name, "count": count}
            for (category, name), count in sorted(counts.items())
        ],
        "activity_sequence": list(activities),
        "cpu_launch_sequence": list(operations),
        "cpu_launch_sequence_sha256": f"launches-{len(operations)}",
        "launch_mapping_unambiguous": not unmapped_activities,
        "unmapped_activities": list(unmapped_activities or ()),
        "cpu_launches": launches,
    }
    return SimpleNamespace(
        profile_mode=mode,
        global_rank=0,
        device_metadata={
            "kineto_attribution_audit": {
                "compute": {"range_activity_signatures": [occurrence]}
            }
        },
    )


@pytest.mark.parametrize(
    ("expected_count", "accepted"),
    [(9_999, False), (10_000, True), (10_001, True)],
)
def test_kineto_comparison_applies_capture_loss_threshold(
    expected_count: int, accepted: bool
) -> None:
    expected = [{"category": "kernel", "name": "same"}] * expected_count
    summary = _compare_kineto_activity_signatures([
        _launch_result("isolated", expected),
        _launch_result("normal", expected[:-1]),
    ])

    assert summary["exact_match"] is False
    assert summary["capture_loss_only"] is True
    assert summary["expected_gpu_activity_records"] == expected_count
    assert summary["missing_gpu_activity_records"] == 1
    assert summary["capture_loss_rate"] == pytest.approx(1 / expected_count)
    assert summary["accepted"] is accepted
    mismatch = summary["mismatches"][0]["ranges"][0]
    assert mismatch["classification"] == "capture_loss"
    assert mismatch["missing_activities"][0]["missing_mode"] == "normal"


def test_kineto_comparison_aggregates_capture_loss_across_ranks() -> None:
    activity = {"category": "kernel", "name": "same"}
    results = []
    for rank in (0, 1):
        expected = [activity] * 5_000
        results.extend([
            _launch_result("isolated", expected, rank=rank),
            _launch_result(
                "normal",
                expected[:-1] if rank == 0 else expected,
                rank=rank,
            ),
        ])

    summary = _compare_kineto_activity_signatures(results)

    assert summary["expected_gpu_activity_records"] == 10_000
    assert summary["missing_gpu_activity_records"] == 1
    assert summary["capture_loss_rate"] == pytest.approx(0.0001)
    assert summary["accepted"] is True


@pytest.mark.parametrize(
    ("expected_launches", "accepted"),
    [(9_999, False), (10_000, True), (10_001, True)],
)
def test_kineto_comparison_applies_cpu_launch_loss_threshold(
    expected_launches: int, accepted: bool
) -> None:
    activities = [
        {"category": "kernel", "name": f"kernel-{index}"}
        for index in range(expected_launches)
    ]
    expected_operations = ["kernel"] * expected_launches
    captured_operations = expected_operations[:-1]
    summary = _compare_kineto_activity_signatures([
        _launch_sequence_result(
            "isolated", expected_operations, activities
        ),
        _launch_sequence_result(
            "normal",
            captured_operations,
            activities,
            unmapped_activities=[activities[-1]],
        ),
    ])

    assert summary["capture_loss_only"] is True
    assert summary["expected_cpu_launch_records"] == expected_launches
    assert summary["missing_cpu_launch_records"] == 1
    assert summary["cpu_launch_loss_rate"] == pytest.approx(
        1 / expected_launches
    )
    assert summary["missing_gpu_activity_records"] == 0
    assert summary["accepted"] is accepted


def test_kineto_comparison_accepts_combined_cpu_and_gpu_deletion_loss() -> None:
    expected_count = 10_001
    activities = [
        {"category": "kernel", "name": f"kernel-{index}"}
        for index in range(expected_count)
    ]
    operations = ["kernel"] * expected_count
    summary = _compare_kineto_activity_signatures([
        _launch_sequence_result("isolated", operations, activities),
        _launch_sequence_result("normal", operations[:-1], activities[:-1]),
    ])

    assert summary["missing_cpu_launch_records"] == 1
    assert summary["missing_gpu_activity_records"] == 1
    assert summary["cpu_launch_loss_rate"] == pytest.approx(1 / expected_count)
    assert summary["capture_loss_rate"] == pytest.approx(1 / expected_count)
    assert summary["accepted"] is True


def test_kineto_comparison_rejects_cpu_launch_reordering() -> None:
    activities = [
        {"category": "kernel", "name": "a"},
        {"category": "gpu_memcpy", "name": "b"},
    ]
    summary = _compare_kineto_activity_signatures([
        _launch_sequence_result(
            "isolated", ["kernel", "memcpy"], activities
        ),
        _launch_sequence_result(
            "normal", ["memcpy", "kernel"], activities
        ),
    ])

    assert summary["capture_loss_only"] is False
    assert summary["accepted"] is False
    assert "CPU launch sequences differ" in (
        summary["mismatches"][0]["ranges"][0]["reason"]
    )


@pytest.mark.parametrize(
    ("normal_activities", "normal_operation", "mapping_unambiguous", "reason"),
    [
        (
            [{"category": "kernel", "name": "a"}],
            "memcpy",
            True,
            "CPU launch sequences differ",
        ),
        (
            [
                {"category": "kernel", "name": "b"},
                {"category": "kernel", "name": "a"},
            ],
            "kernel",
            True,
            "substituted or reordered",
        ),
        (
            [{"category": "kernel", "name": "a"}],
            "kernel",
            False,
            "mapping is ambiguous",
        ),
    ],
)
def test_kineto_comparison_rejects_non_deletion_differences(
    normal_activities,
    normal_operation,
    mapping_unambiguous,
    reason,
) -> None:
    expected = [
        {"category": "kernel", "name": "a"},
        {"category": "kernel", "name": "b"},
    ] if len(normal_activities) == 2 else normal_activities
    summary = _compare_kineto_activity_signatures([
        _launch_result("isolated", expected),
        _launch_result(
            "normal",
            normal_activities,
            operation=normal_operation,
            mapping_unambiguous=mapping_unambiguous,
        ),
    ])

    assert summary["accepted"] is False
    assert summary["capture_loss_only"] is False
    assert reason in summary["mismatches"][0]["ranges"][0]["reason"]


def test_kineto_comparison_details_are_stripped_only_after_comparison() -> None:
    activity = {"category": "kernel", "name": "same"}
    results = [
        _launch_result("isolated", [activity]),
        _launch_result("normal", []),
    ]
    summary = _compare_kineto_activity_signatures(results)
    assert summary["mismatches"][0]["ranges"][0]["missing_activities"]

    _strip_kineto_comparison_details(results)

    for result in results:
        occurrence = result.device_metadata["kineto_attribution_audit"][
            "compute"
        ]["range_activity_signatures"][0]
        assert "cpu_launches" not in occurrence
        assert "cpu_launch_sequence" not in occurrence
        assert "activities" not in occurrence
        assert "activity_sequence" not in occurrence


def test_kineto_iteration_filter_keeps_paired_samples_and_audits() -> None:
    profile = NodeProfile.from_samples(
        (10.0, 20.0, 30.0),
        start_offset_samples_us=(1.0, 2.0, 3.0),
        p2p_post_offset_samples_us=(0.5, 1.5, 2.5),
    )
    signatures = [
        {
            "activity_count": 1,
            "signature_sha256": str(index),
            "activity_sequence_sha256": str(index),
            "cpu_launch_count": index + 1,
            "matched_cpu_launch_count": index,
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
                    "launch_count": 6,
                    "matched_launch_count": 3,
                    "unmatched_launch_count": 3,
                    "range_activity_signatures": signatures,
                    "replay_duration_samples_us": [10.0, 20.0, 30.0],
                }
            }
        },
    )

    filtered = _filter_kineto_profile_iterations([result], (0, 2))[0]

    assert filtered.node_profiles["compute"].samples_us == (10.0, 30.0)
    assert filtered.node_profiles["compute"].sample_iteration_ids == (0, 1)
    assert filtered.node_profiles["compute"].start_offset_samples_us == (
        1.0,
        3.0,
    )
    assert filtered.node_profiles["compute"].p2p_post_offset_samples_us == (
        0.5,
        2.5,
    )
    audit = filtered.device_metadata["kineto_attribution_audit"]["compute"]
    assert audit["range_count"] == 2
    assert audit["launch_count"] == 4
    assert audit["matched_launch_count"] == 2
    assert audit["unmatched_launch_count"] == 2
    assert audit["launch_coverage_pct"] == pytest.approx(50.0)
    assert audit["replay_duration_samples_us"] == [10.0, 30.0]
    assert audit["range_activity_signatures"] == [signatures[0], signatures[2]]


def test_measured_normal_validation_uses_uninstrumented_truth_and_warns() -> None:
    plan = _plan()
    plan.profile_metadata[
        "instrumented_global_step_samples_us_by_mode"
    ] = {"normal": [8.0, 10.0]}
    comparison = replay_profile_modes(
        plan, actual_step_samples_us=(8.0, 10.0)
    )

    result = validate_measured_normal_samples(plan, comparison)

    assert result["measured_normal_step_estimate_us"] == pytest.approx(9.0)
    assert result["replayed_normal_step_estimate_us"] == pytest.approx(10.0)
    assert result["median_absolute_step_error_pct"] == pytest.approx(100 / 9)
    instrumentation = result["kineto_cupti_instrumentation_error"]
    assert instrumentation["signed_error_pct"] == pytest.approx(0.0)
    assert "exceeds" in result["warning"]


def test_replay_profiled_modes_runs_all_automatic_validations() -> None:
    plan = _plan()
    plan.profile_metadata[
        "instrumented_global_step_samples_us_by_mode"
    ] = {"normal": [9.0, 11.0]}

    comparison = replay_profiled_modes(
        plan, actual_step_samples_s=(9e-6, 11e-6)
    )

    validations = plan.profile_metadata["automatic_validations"]
    assert comparison.normal_samples_us == pytest.approx((9.0, 11.0))
    assert validations["replay_endpoints"]["passed"] is True
    assert validations["replay_endpoints"]["additivity_residual_us"] == 0.0
    assert validations["attributed_trace_losslessness"]["passed"] is True
    measured = validations["measured_vs_replayed_normal"]
    assert measured["available"] is True
    assert measured["median_absolute_step_error_pct"] == 0.0
    assert measured["warning"] is None


def test_replay_profiled_modes_accepts_bounded_capture_loss() -> None:
    plan = _plan()
    plan.profile_metadata["kineto_activity_equivalence"] = {
        "exact_match": False,
        "accepted": True,
    }

    comparison = replay_profiled_modes(plan)

    assert comparison.normal_samples_us == pytest.approx((9.0, 11.0))
