from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.pairwise_overlap_corpus_validation import (
    load_samples,
    render_markdown,
    validate_samples,
    write_report,
)


def _sample(index: int, **updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "case_id": "A01",
        "case_key": "A01:C-DENSE",
        "seq_len": 12_288,
        "sample_index": index,
        "dispatch_mode": "victim_first",
        "victim_index": 0,
        "victim_family": "compute",
        "aggressor_family": "p2p_send",
        "requested_overlap_pct": 50.0,
        "target_feasible_from_isolated": True,
        "valid": True,
        "geometry_error_pct": 1.0,
        "geometry_tolerance_pct": 10.0,
        "baseline_drift_pct": 0.5,
        "victim_isolated_us": 5_000.0,
    }
    row.update(updates)
    return row


def test_raw_json_reports_failures_gaps_and_floors(tmp_path: Path) -> None:
    source = tmp_path / "raw.json"
    source.write_text(json.dumps({
        "schema_version": 2,
        "results": [{
            "assignment": {
                "case_key": "A01:C-DENSE",
                "participants": [{"category": "compute"}],
            },
            "correctness": [{"operation_index": 0, "passed": True}],
            "samples": [
                _sample(0),
                _sample(1, geometry_error_pct=11.0, valid=False),
                _sample(2, baseline_drift_pct=7.0),
            ],
        }],
    }), encoding="utf-8")

    samples, provenance = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288, 18_432],
        expected_cases=["A01:C-DENSE", "A02:C-DENSE"],
        expected_coverage_targets=[50.0, 100.0],
        expected_dispatch_modes=["victim_first", "aggressor_first"],
    )

    assert provenance["input_format"] == "result_json"
    assert report["summary"]["requested_count"] == 3
    assert report["summary"]["feasible_count"] == 3
    assert report["summary"]["runner_valid_count"] == 2
    assert report["summary"]["accepted_valid_count"] == 1
    assert report["summary"]["geometry_failure_count"] == 1
    assert report["summary"]["drift_failure_count"] == 1
    assert report["summary"]["missing_sequence_count"] == 1
    assert report["summary"]["missing_case_count"] == 1
    assert report["summary"]["missing_sweep_point_count"] > 0
    assert not report["summary"]["statistical_floor_status"]["50"]["met"]
    assert not report["summary"]["accepted"]


def test_validator_prefers_counterfactual_error_and_all_operation_drift(
    tmp_path: Path,
) -> None:
    basis = (
        "counterfactual_intervals_from_actual_global_gpu_starts_and_"
        "matched_isolated_pre_post_durations"
    )
    stability = "maximum_absolute_pre_post_drift_across_all_operations"
    source = tmp_path / "counterfactual.json"
    source.write_text(json.dumps({
        "results": [{
            "assignment": {
                "case_key": "A01:C-DENSE",
                "participants": [
                    {"category": "compute"},
                    {"category": "p2p_send"},
                ],
            },
            "correctness": [
                {"operation_index": 0, "passed": True},
                {"operation_index": 1, "passed": True},
            ],
            "samples": [
                _sample(
                    0,
                    geometry_error_pct=80.0,
                    scheduled_geometry_error_pct=1.0,
                    counterfactual_geometry_error_pct=1.0,
                    geometry_validity_error_pct=1.0,
                    geometry_validity_basis=basis,
                    geometry_baseline_drifts_pct={"0": 1.0, "1": -0.5},
                    geometry_baseline_max_abs_drift_pct=1.0,
                    geometry_baseline_stability_basis=stability,
                ),
                _sample(
                    1,
                    geometry_error_pct=0.0,
                    scheduled_geometry_error_pct=0.0,
                    counterfactual_geometry_error_pct=0.0,
                    geometry_validity_error_pct=0.0,
                    geometry_validity_basis=basis,
                    geometry_baseline_drifts_pct={"0": 0.0, "1": 8.0},
                    geometry_baseline_max_abs_drift_pct=8.0,
                    geometry_baseline_stability_basis=stability,
                    valid=False,
                ),
            ],
        }],
    }), encoding="utf-8")

    samples, _ = load_samples(source)
    assert samples[0]["geometry_error_pct"] == 1.0
    assert samples[0]["concurrent_geometry_error_pct"] == 80.0
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    [stratum] = report["strata"]
    assert stratum["accepted_valid_count"] == 1
    assert stratum["geometry_failure_count"] == 1
    assert stratum["geometry_baseline_failure_count"] == 1


def test_validator_rejects_counterfactual_alias_and_drift_map_mismatch(
    tmp_path: Path,
) -> None:
    basis = (
        "counterfactual_intervals_from_actual_global_gpu_starts_and_"
        "matched_isolated_pre_post_durations"
    )
    stability = "maximum_absolute_pre_post_drift_across_all_operations"
    source = tmp_path / "integrity.json"
    source.write_text(json.dumps({
        "results": [{
            "assignment": {
                "case_key": "A01:C-DENSE",
                "participants": [
                    {"category": "compute"},
                    {"category": "p2p_send"},
                ],
            },
            "correctness": [
                {"operation_index": 0, "passed": True},
                {"operation_index": 1, "passed": True},
            ],
            "samples": [
                _sample(
                    0,
                    geometry_validity_error_pct=0.0,
                    scheduled_geometry_error_pct=50.0,
                    counterfactual_geometry_error_pct=0.0,
                    geometry_validity_basis=basis,
                    geometry_baseline_drifts_pct={"0": 0.0, "1": 1.0},
                    geometry_baseline_max_abs_drift_pct=1.0,
                    geometry_baseline_stability_basis=stability,
                ),
                _sample(
                    1,
                    geometry_validity_error_pct=0.0,
                    scheduled_geometry_error_pct=0.0,
                    counterfactual_geometry_error_pct=0.0,
                    geometry_validity_basis=basis,
                    geometry_baseline_drifts_pct={"0": 0.0, "1": 8.0},
                    geometry_baseline_max_abs_drift_pct=1.0,
                    geometry_baseline_stability_basis=stability,
                ),
            ],
        }],
    }), encoding="utf-8")
    samples, _ = load_samples(source)
    assert samples[0]["geometry_integrity_ok"] is False
    assert samples[1]["geometry_baseline_integrity_ok"] is False
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    [stratum] = report["strata"]
    assert stratum["accepted_valid_count"] == 0
    assert stratum["geometry_failure_count"] == 1
    assert stratum["geometry_baseline_failure_count"] == 1


def test_validator_rejects_partial_counterfactual_schema_without_downgrade(
    tmp_path: Path,
) -> None:
    basis = (
        "counterfactual_intervals_from_actual_global_gpu_starts_and_"
        "matched_isolated_pre_post_durations"
    )
    source = tmp_path / "partial_counterfactual.json"
    source.write_text(json.dumps({
        "results": [{
            "assignment": {
                "case_key": "A01:C-DENSE",
                "participants": [{"category": "compute"}],
            },
            "correctness": [{"operation_index": 0, "passed": True}],
            "samples": [
                _sample(0, geometry_validity_basis=basis),
                _sample(
                    1,
                    geometry_validity_error_pct=0.0,
                    scheduled_geometry_error_pct=0.0,
                    geometry_validity_basis=basis,
                ),
            ],
        }],
    }), encoding="utf-8")
    samples, _ = load_samples(source)
    assert all(row["counterfactual_schema_present"] for row in samples)
    assert all(not row["geometry_integrity_ok"] for row in samples)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    [stratum] = report["strata"]
    assert stratum["geometry_failure_count"] == 2
    assert stratum["accepted_valid_count"] == 0


def test_csv_can_meet_both_statistical_floors(tmp_path: Path) -> None:
    source = tmp_path / "samples.csv"
    rows = [_sample(index, correct=True) for index in range(100)]
    # Empty V2 columns from a wide CSV export do not constitute methodology.
    for row in rows:
        row.update({
            "geometry_validity_error_pct": "",
            "scheduled_geometry_error_pct": "",
            "counterfactual_geometry_error_pct": "",
            "geometry_validity_basis": "",
            "geometry_baseline_drifts_pct": "",
            "geometry_baseline_max_abs_drift_pct": "",
            "geometry_baseline_stability_basis": "",
        })
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    samples, provenance = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )

    assert provenance["input_format"] == "samples_csv"
    assert report["summary"]["accepted_valid_count"] == 100
    assert report["summary"]["statistical_floor_status"]["50"]["met"]
    assert report["summary"]["statistical_floor_status"]["100"]["met"]
    assert report["summary"]["recommended_floor_status"]["met"]
    assert report["summary"]["accepted"]


def test_infeasible_stratum_does_not_fail_floor_and_reports_correctness() -> None:
    rows = [
        _sample(index, correct=True) for index in range(100)
    ] + [
        _sample(
            100,
            requested_overlap_pct=100.0,
            target_feasible_from_isolated=False,
            valid=False,
            geometry_error_pct=None,
            correct=False,
        )
    ]
    normalized = [
        {
            "case_id": row["case_id"],
            "case_key": row["case_key"],
            "seq_len": row["seq_len"],
            "sample_index": row["sample_index"],
            "dispatch_mode": row["dispatch_mode"],
            "victim_index": row["victim_index"],
            "victim_family": row["victim_family"],
            "aggressor_family": row["aggressor_family"],
            "requested_overlap_pct": row["requested_overlap_pct"],
            "feasible": row["target_feasible_from_isolated"],
            "valid": row["valid"],
            "geometry_error_pct": row["geometry_error_pct"],
            "geometry_tolerance_pct": row["geometry_tolerance_pct"],
            "baseline_drift_pct": row["baseline_drift_pct"],
            "correct": row["correct"],
            "victim_isolated_us": row["victim_isolated_us"],
        }
        for row in rows
    ]
    report = validate_samples(
        normalized,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0, 100.0],
        expected_dispatch_modes=["victim_first"],
    )

    by_target = {row["requested_overlap_pct"]: row for row in report["strata"]}
    assert by_target[100.0]["infeasible_count"] == 1
    assert by_target[100.0]["statistical_floors"]["100"] is False
    assert report["summary"]["statistical_floor_status"]["100"]["met"]
    assert report["summary"]["correctness_failure_count"] == 1
    assert not report["summary"]["gates"]["correctness_passed"]


def test_report_writes_json_and_markdown(tmp_path: Path) -> None:
    normalized = [{
        "case_id": "A01",
        "case_key": "A01:C-DENSE",
        "seq_len": 12_288,
        "sample_index": index,
        "dispatch_mode": "victim_first",
        "victim_index": 0,
        "victim_family": "compute",
        "aggressor_family": "p2p_send",
        "requested_overlap_pct": 50.0,
        "feasible": True,
        "valid": True,
        "geometry_error_pct": 1.0,
        "geometry_tolerance_pct": 10.0,
        "baseline_drift_pct": 0.5,
        "correct": True,
        "victim_isolated_us": 5_000.0,
    } for index in range(100)]
    report = validate_samples(
        normalized,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    json_path, markdown_path = write_report(report, tmp_path / "acceptance")

    assert json.loads(json_path.read_text())["summary"]["accepted"] is True
    markdown = markdown_path.read_text()
    assert markdown == render_markdown(report)
    assert "Overall acceptance: **PASS**" in markdown


def _adaptive_document(
    *,
    valid_samples: int = 50,
    invalid_samples: int = 2,
    discarded_after_stop: int = 3,
    converged: bool = True,
    all_stopped: bool = True,
) -> dict[str, object]:
    total_attempts = valid_samples + invalid_samples
    issued = total_attempts + discarded_after_stop
    rows = [
        _sample(
            index,
            correct=True,
            baseline_drift_pct=(7.0 if valid_samples <= index < total_attempts else 0.5),
            victim_isolated_us=20_000.0,
        )
        for index in range(issued)
    ]
    cell_key = "A01:C-DENSE:v0:o50:victim_first"
    return {
        "schema_version": 2,
        "results": [{
            "assignment": {
                "case_key": "A01:C-DENSE",
                "participants": [{"category": "compute"}],
            },
            "correctness": [{"operation_index": 0, "passed": True}],
            "samples": rows,
            "adaptive_sampling": {
                "schema_version": 1,
                "summary": {
                    "cell_count": 1,
                    "active_cell_count": 0 if all_stopped else 1,
                    "converged_cell_count": 1 if converged else 0,
                    "stopped_without_convergence_count": 0,
                    "total_attempts": total_attempts,
                    "valid_samples": valid_samples,
                    "invalid_samples": invalid_samples,
                    "discarded_after_stop": discarded_after_stop,
                    "all_stopped": all_stopped,
                },
                "cells": [{
                    "cell": {
                        "case_key": "A01:C-DENSE",
                        "victim_index": 0,
                        "requested_overlap_pct": 50.0,
                        "dispatch_mode": "victim_first",
                    },
                    "cell_key": cell_key,
                    "isolated_duration_ms": 20.0,
                    "total_attempts": total_attempts,
                    "valid_samples": valid_samples,
                    "invalid_samples": invalid_samples,
                    "discarded_after_stop": discarded_after_stop,
                    "minimum_valid_samples": 50,
                    "precision_met": converged,
                    "converged": converged,
                    "explicitly_infeasible": False,
                    "should_continue": not all_stopped,
                    "stop_reason": "ci_precision_met" if all_stopped else None,
                }],
                "rounds": [{
                    "round_index": 0,
                    "ingestion": {
                        "rows_seen": issued,
                        "rows_ingested": total_attempts,
                        "rows_discarded_after_stop": discarded_after_stop,
                    },
                }],
                "actual_attempt_counts": {cell_key: total_attempts},
            },
        }],
    }


def _v2_adaptive_document() -> dict[str, object]:
    document = _adaptive_document()
    result = document["results"][0]  # type: ignore[index]
    result["assignment"]["participants"] = [  # type: ignore[index]
        {"category": "compute"},
        {"category": "p2p_send"},
    ]
    result["correctness"] = [  # type: ignore[index]
        {"operation_index": 0, "passed": True},
        {"operation_index": 1, "passed": True},
    ]
    basis = (
        "counterfactual_intervals_from_actual_global_gpu_starts_and_"
        "matched_isolated_pre_post_durations"
    )
    stability = "maximum_absolute_pre_post_drift_across_all_operations"
    for sample in result["samples"]:  # type: ignore[index]
        sample.update({
            "geometry_validity_error_pct": 0.0,
            "scheduled_geometry_error_pct": 0.0,
            "counterfactual_geometry_error_pct": 0.0,
            "geometry_validity_basis": basis,
            "geometry_baseline_drifts_pct": {"0": 0.5, "1": -1.0},
            "geometry_baseline_max_abs_drift_pct": 1.0,
            "geometry_baseline_stability_basis": stability,
        })
    measurements = [
        {
            "trial_index": 1,
            "confirmation_index": confirmation_index,
            "calibration_sample_index": -100 - confirmation_index,
            "relative_shift_ms": 0.0,
            "delays_ms": {"0": 0.0, "1": 5.0},
            "geometry_validity_basis": basis,
            "scheduled_geometry_error_pct": error,
            "counterfactual_geometry_error_pct": error,
            "paired_baseline_drifts_pct": {"0": 0.5, "1": -drift},
            "geometry_baseline_max_abs_drift_pct": drift,
        }
        for confirmation_index, (error, drift) in enumerate(
            ((-1.0, 1.0), (0.0, 2.0), (1.0, 3.0))
        )
    ]
    candidate = {
        "trial_index": 1,
        "relative_shift_ms": 0.0,
        "delays_ms": {"0": 0.0, "1": 5.0},
        "scheduled_geometry_error_pct": 0.0,
        "counterfactual_geometry_error_pct": 0.0,
        "geometry_baseline_max_abs_drift_pct": 2.0,
        "geometry_validity_basis": basis,
        "geometry_baseline_stability_basis": stability,
        "confirmation_trial_count": 3,
        "confirmation_passing_count": 3,
        "geometry_passing_count": 3,
        "baseline_stable_count": 3,
        "jointly_passing_count": 3,
        "median_centered": True,
        "robustly_converged": True,
        "confirmation_measurements": measurements,
    }
    result["stage_specs"] = [{  # type: ignore[index]
        "victim_index": 0,
        "requested_overlap_pct": 50.0,
        "predicted_max_overlap_pct": 50.0,
        "dispatch_mode": "victim_first",
        "aggressor_indices": [1],
        "isolated_estimates_ms": {"0": 20.0, "1": 10.0},
        "natural_group_offsets_ms": {"0": 0.0, "1": 0.0},
        "aggressor_span_ms": 10.0,
        "duration_feasibility_tolerance_pct_points": 1e-6,
        "duration_feasibility_formula": (
            "min(100, max(offset_i-min_aggressor_offset+"
            "isolated_duration_i)/victim_isolated_duration*100)"
        ),
        "target_feasible_from_duration": True,
        "target_feasible_after_constraints": True,
        "target_feasible_from_isolated": True,
        "target_feasible_after_runtime_calibration": True,
        "geometry_cell_status": "confirmed_feasible",
        "geometry_validity_basis": basis,
        "geometry_calibration_history": [candidate],
        "runtime_geometry_calibration_status": "converged",
        "runtime_relative_shift_ms": 0.0,
        "delays_ms": {"0": 0.0, "1": 5.0},
        "runtime_geometry_selected_trial_index": 1,
        "runtime_calibrated_scheduled_geometry_error_pct": 0.0,
        "runtime_calibrated_counterfactual_geometry_error_pct": 0.0,
        "runtime_geometry_baseline_max_abs_drift_pct": 2.0,
        "runtime_geometry_baseline_stability_basis": stability,
        "runtime_geometry_confirmation_trial_count": 3,
        "runtime_geometry_confirmation_passing_count": 3,
        "runtime_geometry_confirmation_geometry_passing_count": 3,
        "runtime_geometry_confirmation_baseline_stable_count": 3,
        "runtime_geometry_confirmation_jointly_passing_count": 3,
        "runtime_geometry_median_centered": True,
        "runtime_geometry_robustly_converged": True,
    }]
    diagnostics = result["adaptive_sampling"]  # type: ignore[index]
    diagnostics["cells"][0].update({  # type: ignore[index]
        "pre_sampling_status": "ready",
        "infeasibility_proof_basis": None,
    })
    document["configuration"] = {  # type: ignore[index]
        "sampling": {"mode": "adaptive"},
        "scheduling": {"geometry_validity_basis": basis},
    }
    return document


def _v2_proven_infeasible_document() -> dict[str, object]:
    document = _v2_adaptive_document()
    result = document["results"][0]  # type: ignore[index]
    spec = json.loads(json.dumps(result["stage_specs"][0]))  # type: ignore[index]
    spec.update({
        "requested_overlap_pct": 75.0,
        "target_feasible_from_duration": False,
        "target_feasible_after_constraints": False,
        "target_feasible_from_isolated": False,
        "target_feasible_after_runtime_calibration": False,
        "geometry_cell_status": "proven_infeasible",
        "geometry_infeasibility_basis": "duration_limit",
        "runtime_geometry_calibration_status": (
            "isolated_duration_bound_infeasible"
        ),
        "geometry_calibration_history": [],
    })
    result["stage_specs"].append(spec)  # type: ignore[index]
    diagnostics = result["adaptive_sampling"]  # type: ignore[index]
    key = "A01:C-DENSE:v0:o75:victim_first"
    diagnostics["summary"].update({  # type: ignore[index]
        "cell_count": 2,
        "active_cell_count": 0,
        "converged_cell_count": 1,
        "stopped_without_convergence_count": 1,
        "all_stopped": True,
    })
    diagnostics["cells"].append({  # type: ignore[index]
        "cell": {
            "case_key": "A01:C-DENSE",
            "victim_index": 0,
            "requested_overlap_pct": 75.0,
            "dispatch_mode": "victim_first",
        },
        "cell_key": key,
        "isolated_duration_ms": 20.0,
        "total_attempts": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "discarded_after_stop": 0,
        "minimum_valid_samples": 50,
        "precision_met": False,
        "converged": False,
        "explicitly_infeasible": True,
        "pre_sampling_status": "proven_infeasible",
        "infeasibility_proof_basis": "duration_bound",
        "should_continue": False,
        "stop_reason": "geometry_target_proven_infeasible",
    })
    diagnostics["actual_attempt_counts"][key] = 0  # type: ignore[index]
    return document


def test_adaptive_retries_are_reported_but_do_not_globally_fail(tmp_path: Path) -> None:
    source = tmp_path / "adaptive.json"
    source.write_text(json.dumps(_adaptive_document()), encoding="utf-8")

    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )

    assert report["schema_version"] == 3
    assert report["summary"]["sample_count"] == 55
    assert report["summary"]["accepted_valid_count"] == 53
    assert report["summary"]["raw_accepted_valid_count"] == 53
    assert report["summary"]["adaptive_controller_valid_sample_count"] == 50
    assert report["summary"]["adaptive_controller_invalid_sample_count"] == 2
    assert report["summary"]["adaptive_controller_discarded_after_stop_count"] == 3
    assert report["summary"]["estimator_valid_sample_count"] == 50
    assert report["summary"]["drift_failure_count"] == 2
    assert report["summary"]["raw_rejection_rate_pct"] == 2 / 55 * 100.0
    adaptive = report["adaptive_sampling"]
    assert adaptive["ingested_attempt_count"] == 52
    assert adaptive["valid_sample_count"] == 50
    assert adaptive["invalid_sample_count"] == 2
    assert adaptive["discarded_after_stop_count"] == 3
    assert adaptive["feasible_rejection_rate_pct"] == 2 / 52 * 100.0
    assert adaptive["cells"][0]["floor_met"] is True
    assert report["strata"][0]["adaptive_valid_sample_count"] == 50
    assert report["strata"][0]["adaptive_invalid_sample_count"] == 2
    assert report["strata"][0]["adaptive_discarded_after_stop_count"] == 3
    assert report["summary"]["gates"]["rejection_rate_within_bound"] is True
    assert report["summary"]["accepted"] is True
    markdown = render_markdown(report)
    assert "Estimator-valid samples" in markdown
    assert "tail-of-batch" in markdown


def test_v2_stage_calibration_is_independently_reconstructed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v2_adaptive.json"
    source.write_text(json.dumps(_v2_adaptive_document()), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert report["summary"]["accepted"] is True
    assert report["adaptive_sampling"]["cells"][0][
        "stage_calibration_evidence_verified"
    ] is True


def test_v2_duration_bound_exemption_is_independently_reconstructed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "v2_proven_infeasible.json"
    source.write_text(
        json.dumps(_v2_proven_infeasible_document()), encoding="utf-8"
    )
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0, 75.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert report["summary"]["accepted"] is True
    cell = next(
        row for row in report["adaptive_sampling"]["cells"]
        if row["requested_overlap_pct"] == 75.0
    )
    assert cell["infeasibility_proven"] is True
    assert cell["stage_duration_evidence_verified"] is True
    assert cell["stage_recomputed_duration_infeasible"] is True

    mutations = {}
    false_proof = _v2_proven_infeasible_document()
    false_proof["results"][0]["stage_specs"][1][  # type: ignore[index]
        "isolated_estimates_ms"
    ]["0"] = 10.0
    mutations["false proof"] = false_proof
    empty_aggressor = _v2_proven_infeasible_document()
    empty_aggressor["results"][0]["stage_specs"][1][  # type: ignore[index]
        "aggressor_indices"
    ] = []
    mutations["empty aggressor"] = empty_aggressor
    missing_proof = _v2_proven_infeasible_document()
    missing_proof["results"][0]["stage_specs"][1].pop(  # type: ignore[index]
        "isolated_estimates_ms"
    )
    mutations["missing proof"] = missing_proof
    for name, mutated in mutations.items():
        source = tmp_path / f"v2_infeasible_{name.replace(' ', '_')}.json"
        source.write_text(json.dumps(mutated), encoding="utf-8")
        mutated_samples, _ = load_samples(source)
        failed = validate_samples(
            mutated_samples,
            expected_sequences=[12_288],
            expected_cases=["A01:C-DENSE"],
            expected_coverage_targets=[50.0, 75.0],
            expected_dispatch_modes=["victim_first"],
        )
        assert failed["summary"]["accepted"] is False, name
        failed_cell = next(
            row for row in failed["adaptive_sampling"]["cells"]
            if row["requested_overlap_pct"] == 75.0
        )
        assert failed_cell["infeasibility_proven"] is False, name


def test_v2_stage_and_sample_contract_mutations_fail_closed(
    tmp_path: Path,
) -> None:
    mutations = {}

    stripped = _v2_adaptive_document()
    sample = stripped["results"][0]["samples"][0]  # type: ignore[index]
    for field in (
        "geometry_validity_error_pct",
        "scheduled_geometry_error_pct",
        "counterfactual_geometry_error_pct",
        "geometry_validity_basis",
        "geometry_baseline_drifts_pct",
        "geometry_baseline_max_abs_drift_pct",
        "geometry_baseline_stability_basis",
    ):
        sample.pop(field)  # type: ignore[union-attr]
    mutations["stripped sample schema"] = stripped

    missing_counterfactual_alias = _v2_adaptive_document()
    missing_counterfactual_alias["results"][0]["samples"][0].pop(  # type: ignore[index]
        "counterfactual_geometry_error_pct"
    )
    mutations["missing counterfactual alias"] = missing_counterfactual_alias

    sample_missing_operation = _v2_adaptive_document()
    sample_missing_operation["results"][0]["samples"][0].update({  # type: ignore[index]
        "geometry_baseline_drifts_pct": {"0": 0.5},
        "geometry_baseline_max_abs_drift_pct": 0.5,
    })
    mutations["sample drift map missing operation"] = sample_missing_operation

    sample_extra_operation = _v2_adaptive_document()
    sample_extra_operation["results"][0]["samples"][0].update({  # type: ignore[index]
        "geometry_baseline_drifts_pct": {"0": 0.5, "1": -1.0, "2": 0.0},
        "geometry_baseline_max_abs_drift_pct": 1.0,
    })
    mutations["sample drift map extra operation"] = sample_extra_operation

    nested_downgrade = _v2_adaptive_document()
    nested_downgrade["configuration"]["scheduling"] = {  # type: ignore[index]
        "geometry_feedback": {
            "validity_basis": (
                "counterfactual_intervals_from_actual_global_gpu_starts_and_"
                "matched_isolated_pre_post_durations"
            )
        }
    }
    nested_result = nested_downgrade["results"][0]  # type: ignore[index]
    nested_result["stage_specs"][0].pop("geometry_validity_basis")  # type: ignore[index]
    for nested_sample in nested_result["samples"]:  # type: ignore[index]
        for field in (
            "geometry_validity_error_pct",
            "scheduled_geometry_error_pct",
            "counterfactual_geometry_error_pct",
            "geometry_validity_basis",
            "geometry_baseline_drifts_pct",
            "geometry_baseline_max_abs_drift_pct",
            "geometry_baseline_stability_basis",
        ):
            nested_sample.pop(field)  # type: ignore[union-attr]
    mutations["nested preview downgrade"] = nested_downgrade

    missing_participants = _v2_adaptive_document()
    missing_participants["results"][0]["assignment"].pop("participants")  # type: ignore[index]
    mutations["missing assignment participants"] = missing_participants

    empty_participants = _v2_adaptive_document()
    empty_participants["results"][0]["assignment"]["participants"] = []  # type: ignore[index]
    mutations["empty assignment participants"] = empty_participants

    victim_repeated = _v2_adaptive_document()
    victim_repeated["results"][0]["stage_specs"][0][  # type: ignore[index]
        "aggressor_indices"
    ] = [0, 1]
    mutations["victim repeated as aggressor"] = victim_repeated

    duplicate_aggressor = _v2_adaptive_document()
    duplicate_aggressor["results"][0]["stage_specs"][0][  # type: ignore[index]
        "aggressor_indices"
    ] = [1, 1]
    mutations["duplicate aggressor"] = duplicate_aggressor

    empty_aggressors = _v2_adaptive_document()
    empty_aggressors["results"][0]["stage_specs"][0][  # type: ignore[index]
        "aggressor_indices"
    ] = []
    mutations["empty aggressor duration proof"] = empty_aggressors

    false_duration_flag = _v2_adaptive_document()
    false_duration_flag["results"][0]["stage_specs"][0][  # type: ignore[index]
        "target_feasible_from_duration"
    ] = False
    mutations["false duration infeasibility claim"] = false_duration_flag

    missing_duration_proof = _v2_adaptive_document()
    missing_duration_proof["results"][0]["stage_specs"][0].pop(  # type: ignore[index]
        "isolated_estimates_ms"
    )
    mutations["missing duration proof inputs"] = missing_duration_proof

    missing_correctness = _v2_adaptive_document()
    missing_correctness["results"][0]["correctness"] = [  # type: ignore[index]
        {"operation_index": 0, "passed": True}
    ]
    mutations["missing correctness operation"] = missing_correctness

    duplicate_correctness = _v2_adaptive_document()
    duplicate_correctness["results"][0]["correctness"].append(  # type: ignore[index]
        {"operation_index": 0, "passed": True}
    )
    mutations["duplicate correctness identity"] = duplicate_correctness

    extra_correctness = _v2_adaptive_document()
    extra_correctness["results"][0]["correctness"].append(  # type: ignore[index]
        {"operation_index": 2, "passed": True}
    )
    mutations["extra correctness operation"] = extra_correctness

    failed_correctness = _v2_adaptive_document()
    failed_correctness["results"][0]["correctness"][1]["passed"] = False  # type: ignore[index]
    mutations["failed correctness operation"] = failed_correctness

    row_correctness_disagrees = _v2_adaptive_document()
    row_correctness_disagrees["results"][0]["samples"][0]["correct"] = False  # type: ignore[index]
    mutations["row correctness alias disagrees"] = row_correctness_disagrees

    missing_adaptive = _v2_adaptive_document()
    missing_adaptive["results"][0].pop("adaptive_sampling")  # type: ignore[index]
    mutations["missing adaptive diagnostics"] = missing_adaptive

    missing_config_and_stages = _v2_adaptive_document()
    missing_config_and_stages.pop("configuration")
    missing_config_and_stages["results"][0].pop("stage_specs")  # type: ignore[index]
    mutations["sample schema without config or stages"] = (
        missing_config_and_stages
    )

    disabled = _v2_adaptive_document()
    disabled_spec = disabled["results"][0]["stage_specs"][0]  # type: ignore[index]
    disabled_spec["geometry_cell_status"] = "calibration_disabled"  # type: ignore[index]
    mutations["calibration disabled"] = disabled

    not_robust = _v2_adaptive_document()
    not_robust_spec = not_robust["results"][0]["stage_specs"][0]  # type: ignore[index]
    not_robust_spec["runtime_geometry_robustly_converged"] = False  # type: ignore[index]
    mutations["reported non-robust"] = not_robust

    off_center = _v2_adaptive_document()
    off_center_spec = off_center["results"][0]["stage_specs"][0]  # type: ignore[index]
    off_center_spec.update({  # type: ignore[union-attr]
        "runtime_calibrated_scheduled_geometry_error_pct": 3.0,
        "runtime_calibrated_counterfactual_geometry_error_pct": 3.0,
    })
    off_center_candidate = off_center_spec["geometry_calibration_history"][0]  # type: ignore[index]
    off_center_candidate.update({
        "scheduled_geometry_error_pct": 3.0,
        "counterfactual_geometry_error_pct": 3.0,
    })
    for measurement in off_center_candidate["confirmation_measurements"]:
        measurement.update({
            "scheduled_geometry_error_pct": 3.0,
            "counterfactual_geometry_error_pct": 3.0,
        })
    mutations["median exceeds two points"] = off_center

    geometry_miss = _v2_adaptive_document()
    geometry_candidate = geometry_miss["results"][0]["stage_specs"][0][  # type: ignore[index]
        "geometry_calibration_history"
    ][0]
    geometry_candidate["confirmation_measurements"][1].update({  # type: ignore[index]
        "scheduled_geometry_error_pct": 11.0,
        "counterfactual_geometry_error_pct": 11.0,
    })
    geometry_candidate["confirmation_measurements"][2].update({  # type: ignore[index]
        "scheduled_geometry_error_pct": 12.0,
        "counterfactual_geometry_error_pct": 12.0,
    })
    mutations["fewer than two geometry passes"] = geometry_miss

    drift_miss = _v2_adaptive_document()
    drift_candidate = drift_miss["results"][0]["stage_specs"][0][  # type: ignore[index]
        "geometry_calibration_history"
    ][0]
    for index, drift in ((1, 6.0), (2, 7.0)):
        drift_candidate["confirmation_measurements"][index].update({  # type: ignore[index]
            "paired_baseline_drifts_pct": {"0": 0.5, "1": -drift},
            "geometry_baseline_max_abs_drift_pct": drift,
        })
    mutations["fewer than two drift passes"] = drift_miss

    drift_map_corruption = _v2_adaptive_document()
    corrupt_candidate = drift_map_corruption["results"][0]["stage_specs"][0][  # type: ignore[index]
        "geometry_calibration_history"
    ][0]
    corrupt_candidate["confirmation_measurements"][0].update({  # type: ignore[index]
        "paired_baseline_drifts_pct": {"0": 0.5, "1": -9.0},
        "geometry_baseline_max_abs_drift_pct": 1.0,
    })
    mutations["confirmation drift max corruption"] = drift_map_corruption

    confirmation_missing_operation = _v2_adaptive_document()
    missing_candidate = confirmation_missing_operation["results"][0][  # type: ignore[index]
        "stage_specs"
    ][0]["geometry_calibration_history"][0]
    missing_candidate["confirmation_measurements"][0].update({  # type: ignore[index]
        "paired_baseline_drifts_pct": {"0": 0.5},
        "geometry_baseline_max_abs_drift_pct": 0.5,
    })
    mutations["confirmation drift map missing operation"] = (
        confirmation_missing_operation
    )

    confirmation_extra_operation = _v2_adaptive_document()
    extra_candidate = confirmation_extra_operation["results"][0][  # type: ignore[index]
        "stage_specs"
    ][0]["geometry_calibration_history"][0]
    extra_candidate["confirmation_measurements"][0].update({  # type: ignore[index]
        "paired_baseline_drifts_pct": {"0": 0.5, "1": -1.0, "2": 0.0},
        "geometry_baseline_max_abs_drift_pct": 1.0,
    })
    mutations["confirmation drift map extra operation"] = (
        confirmation_extra_operation
    )

    duplicated_confirmation = _v2_adaptive_document()
    duplicate_candidate = duplicated_confirmation["results"][0][  # type: ignore[index]
        "stage_specs"
    ][0]["geometry_calibration_history"][0]
    middle = json.loads(json.dumps(
        duplicate_candidate["confirmation_measurements"][1]
    ))
    duplicate_candidate["confirmation_measurements"] = [
        json.loads(json.dumps(middle)) for _ in range(3)
    ]
    mutations["duplicated confirmation evidence"] = duplicated_confirmation

    changed_final_delays = _v2_adaptive_document()
    changed_final_delays["results"][0]["stage_specs"][0]["delays_ms"] = {  # type: ignore[index]
        "0": 0.0,
        "1": 6.0,
    }
    mutations["final treatment delays differ from selected candidate"] = (
        changed_final_delays
    )

    for name, document in mutations.items():
        source = tmp_path / f"{name.replace(' ', '_')}.json"
        source.write_text(json.dumps(document), encoding="utf-8")
        samples, _ = load_samples(source)
        report = validate_samples(
            samples,
            expected_sequences=[12_288],
            expected_cases=["A01:C-DENSE"],
            expected_coverage_targets=[50.0],
            expected_dispatch_modes=["victim_first"],
        )
        assert report["summary"]["accepted"] is False, name
        assert report["adaptive_sampling"][
            "malformed_or_failed_diagnostic_count"
        ] > 0, name


def test_authoritative_v2_requirement_rejects_total_provenance_strip(
    tmp_path: Path,
) -> None:
    document = _v2_adaptive_document()
    document.pop("configuration")
    result = document["results"][0]  # type: ignore[index]
    result.pop("stage_specs")
    result.pop("adaptive_sampling")
    for sample in result["samples"]:  # type: ignore[index]
        for field in (
            "geometry_validity_error_pct",
            "scheduled_geometry_error_pct",
            "counterfactual_geometry_error_pct",
            "geometry_validity_basis",
            "geometry_baseline_drifts_pct",
            "geometry_baseline_max_abs_drift_pct",
            "geometry_baseline_stability_basis",
        ):
            sample.pop(field)
    source = tmp_path / "total_v2_provenance_strip.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    default_report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    required_report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        require_counterfactual_v2_adaptive=True,
    )
    assert default_report["summary"]["accepted"] is True
    assert required_report["summary"]["accepted"] is False
    assert required_report["summary"]["gates"][
        "counterfactual_v2_adaptive_required"
    ] is False


def test_fixed_sampling_v2_does_not_require_adaptive_diagnostics(
    tmp_path: Path,
) -> None:
    document = _v2_adaptive_document()
    document["configuration"]["sampling"]["mode"] = "fixed"  # type: ignore[index]
    document["results"][0].pop("adaptive_sampling")  # type: ignore[index]
    source = tmp_path / "fixed_v2.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert report["adaptive_sampling"]["present"] is False
    assert report["summary"]["accepted"] is True


def test_v2_selected_trial_index_disambiguates_repeated_candidates(
    tmp_path: Path,
) -> None:
    document = _v2_adaptive_document()
    spec = document["results"][0]["stage_specs"][0]  # type: ignore[index]
    repeated = json.loads(json.dumps(spec["geometry_calibration_history"][0]))  # type: ignore[index]
    repeated["trial_index"] = 0
    repeated["robustly_converged"] = False
    spec["geometry_calibration_history"].insert(0, repeated)  # type: ignore[index]
    source = tmp_path / "repeated_candidate.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert report["summary"]["accepted"] is True
    assert report["adaptive_sampling"]["cells"][0][
        "stage_calibration_evidence_verified"
    ] is True


def test_adaptive_rejection_rate_is_bounded_and_configurable(tmp_path: Path) -> None:
    source = tmp_path / "adaptive_noisy.json"
    source.write_text(json.dumps(_adaptive_document(invalid_samples=6, discarded_after_stop=0)), encoding="utf-8")
    samples, _ = load_samples(source)

    default_report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    relaxed_report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        max_rejection_rate_pct=11.0,
    )

    assert default_report["adaptive_sampling"]["passed"] is True
    assert default_report["summary"]["gates"]["rejection_rate_within_bound"] is False
    assert default_report["summary"]["accepted"] is False
    assert relaxed_report["summary"]["gates"]["rejection_rate_within_bound"] is True
    assert relaxed_report["summary"]["accepted"] is True


def test_adaptive_nonterminal_or_nonconverged_diagnostics_fail(tmp_path: Path) -> None:
    source = tmp_path / "adaptive_incomplete.json"
    source.write_text(
        json.dumps(_adaptive_document(converged=False, all_stopped=False)),
        encoding="utf-8",
    )
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )

    assert report["adaptive_sampling"]["malformed_or_failed_diagnostic_count"] > 0
    assert report["summary"]["gates"]["adaptive_sampling_complete"] is False
    assert report["summary"]["accepted"] is False


def test_new_endpoint_precision_is_independently_verified(tmp_path: Path) -> None:
    document = _adaptive_document()
    cell = document["results"][0]["adaptive_sampling"]["cells"][0]  # type: ignore[index]
    cell.update({  # type: ignore[union-attr]
        "target_ci_half_width_pp": 1.5,
        "median_pp": 0.0,
        "ci95_low_pp": -1.0,
        "ci95_high_pp": 2.0,
        "ci95_half_width_pp": 1.5,
        "max_endpoint_distance_pp": 2.0,
        # These legacy flags would incorrectly accept the asymmetric interval.
        "precision_met": True,
        "converged": True,
    })
    source = tmp_path / "asymmetric_endpoint.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    failed = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert failed["summary"]["accepted"] is False
    assert any(
        "precision_met disagrees" in error
        for error in failed["adaptive_sampling"]["errors"]
    )
    assert failed["adaptive_sampling"]["endpoint_precision_cell_count"] == 1
    assert failed["adaptive_sampling"]["legacy_precision_cell_count"] == 0

    cell.update({  # type: ignore[union-attr]
        "ci95_high_pp": 1.0,
        "ci95_half_width_pp": 1.0,
        "max_endpoint_distance_pp": 1.0,
    })
    source.write_text(json.dumps(document), encoding="utf-8")
    corrected_samples, _ = load_samples(source)
    passed = validate_samples(
        corrected_samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert passed["summary"]["accepted"] is True
    endpoint = passed["adaptive_sampling"]["cells"][0]
    assert endpoint["independently_verified_precision_met"] is True
    assert endpoint["precision_criterion_mode"] == (
        "endpoint_distance_independently_verified"
    )


def test_legacy_adaptive_precision_behavior_is_explicit(tmp_path: Path) -> None:
    source = tmp_path / "legacy_precision.json"
    source.write_text(json.dumps(_adaptive_document()), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert report["summary"]["accepted"] is True
    adaptive = report["adaptive_sampling"]
    assert adaptive["endpoint_precision_cell_count"] == 0
    assert adaptive["legacy_precision_cell_count"] == 1
    assert adaptive["cells"][0]["precision_criterion_mode"] == (
        "legacy_reported_flags_without_endpoint_metric"
    )


def test_terminal_explicitly_infeasible_cell_satisfies_sweep_coverage(
    tmp_path: Path,
) -> None:
    document = _adaptive_document()
    result = document["results"][0]  # type: ignore[index]
    diagnostics = document["results"][0]["adaptive_sampling"]  # type: ignore[index]
    diagnostics["cells"][0]["pre_sampling_status"] = "ready"  # type: ignore[index]
    result["stage_specs"] = [  # type: ignore[index]
        {
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "victim_first",
            "target_feasible_from_isolated": True,
            "geometry_cell_status": "confirmed_feasible",
            "geometry_calibration_history": [{"robustly_converged": True}],
        },
        {
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "aggressor_first",
            "target_feasible_from_duration": False,
            "target_feasible_from_isolated": False,
            "geometry_cell_status": "proven_infeasible",
            "geometry_infeasibility_basis": "duration_limit",
            "geometry_calibration_history": [],
        },
    ]
    infeasible_key = "A01:C-DENSE:v0:o50:aggressor_first"
    diagnostics["cells"].append({  # type: ignore[index,union-attr]
        "cell": {
            "case_key": "A01:C-DENSE",
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "aggressor_first",
        },
        "cell_key": infeasible_key,
        "isolated_duration_ms": 20.0,
        "total_attempts": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "discarded_after_stop": 0,
        "minimum_valid_samples": 50,
        "precision_met": False,
        "converged": False,
        "explicitly_infeasible": True,
        "pre_sampling_status": "proven_infeasible",
        "infeasibility_proof_basis": "duration_bound",
        "should_continue": False,
        "stop_reason": "geometry_target_proven_infeasible",
    })
    diagnostics["summary"].update({  # type: ignore[index,union-attr]
        "cell_count": 2,
        "stopped_without_convergence_count": 1,
    })
    diagnostics["actual_attempt_counts"][infeasible_key] = 0  # type: ignore[index]
    source = tmp_path / "explicitly_infeasible.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first", "aggressor_first"],
    )
    assert report["summary"]["missing_sweep_point_count"] == 0
    assert report["summary"][
        "diagnostic_explicitly_infeasible_sweep_point_count"
    ] == 1
    assert report["summary"]["gates"]["no_sweep_point_gaps"] is True
    assert report["summary"]["accepted"] is True


def test_legacy_infeasibility_exemption_remains_compatible_and_is_labeled(
    tmp_path: Path,
) -> None:
    document = _adaptive_document()
    diagnostics = document["results"][0]["adaptive_sampling"]  # type: ignore[index]
    key = "A01:C-DENSE:v0:o50:aggressor_first"
    diagnostics["cells"].append({  # type: ignore[index,union-attr]
        "cell": {
            "case_key": "A01:C-DENSE",
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "aggressor_first",
        },
        "cell_key": key,
        "isolated_duration_ms": 20.0,
        "total_attempts": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "discarded_after_stop": 0,
        "minimum_valid_samples": 50,
        "precision_met": False,
        "converged": False,
        "explicitly_infeasible": True,
        "should_continue": False,
        "stop_reason": "geometry_target_infeasible",
    })
    diagnostics["summary"].update({  # type: ignore[index,union-attr]
        "cell_count": 2,
        "stopped_without_convergence_count": 1,
    })
    diagnostics["actual_attempt_counts"][key] = 0  # type: ignore[index]
    source = tmp_path / "unproven_legacy_infeasible.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first", "aggressor_first"],
    )

    assert report["adaptive_sampling"]["proven_infeasible_cell_count"] == 0
    assert report["adaptive_sampling"][
        "legacy_infeasibility_exemption_cell_count"
    ] == 1
    assert report["summary"][
        "diagnostic_explicitly_infeasible_sweep_point_count"
    ] == 1
    assert report["summary"]["missing_sweep_point_count"] == 0
    assert report["summary"]["accepted"] is True


def test_calibration_unresolved_is_retained_but_not_sweep_exempt(
    tmp_path: Path,
) -> None:
    document = _adaptive_document()
    result = document["results"][0]  # type: ignore[index]
    diagnostics = result["adaptive_sampling"]  # type: ignore[index]
    diagnostics["cells"][0]["pre_sampling_status"] = "ready"  # type: ignore[index]
    unresolved_key = "A01:C-DENSE:v0:o50:aggressor_first"
    diagnostics["cells"].append({  # type: ignore[index,union-attr]
        "cell": {
            "case_key": "A01:C-DENSE",
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "aggressor_first",
        },
        "cell_key": unresolved_key,
        "isolated_duration_ms": 20.0,
        "total_attempts": 0,
        "valid_samples": 0,
        "invalid_samples": 0,
        "discarded_after_stop": 0,
        "minimum_valid_samples": 50,
        "precision_met": False,
        "converged": False,
        "explicitly_infeasible": False,
        "pre_sampling_status": "calibration_unresolved",
        "should_continue": False,
        "stop_reason": "geometry_calibration_unresolved",
    })
    diagnostics["summary"].update({  # type: ignore[index,union-attr]
        "cell_count": 2,
        "stopped_without_convergence_count": 1,
    })
    diagnostics["actual_attempt_counts"][unresolved_key] = 0  # type: ignore[index]
    result["stage_specs"] = [  # type: ignore[index]
        {
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "victim_first",
            "target_feasible_from_isolated": True,
            "geometry_cell_status": "confirmed_feasible",
            "geometry_calibration_history": [{"robustly_converged": True}],
        },
        {
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "aggressor_first",
            "target_feasible_from_isolated": True,
            "geometry_cell_status": "calibration_unresolved",
            "runtime_geometry_calibration_status": "bounded_trials_exhausted",
            "geometry_calibration_history": [{
                "trial_index": 0,
                "geometry_error_pct": 20.0,
                "paired_baseline_drift_pct": 1.0,
            }],
        },
    ]
    source = tmp_path / "calibration_unresolved.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first", "aggressor_first"],
    )

    assert report["adaptive_sampling"]["calibration_unresolved_cell_count"] == 1
    unresolved = next(
        cell for cell in report["adaptive_sampling"]["cells"]
        if cell["dispatch_mode"] == "aggressor_first"
    )
    assert unresolved["explicitly_infeasible"] is False
    assert unresolved["calibration_unresolved"] is True
    assert unresolved["observed_sample_count"] == 0
    assert report["summary"][
        "diagnostic_explicitly_infeasible_sweep_point_count"
    ] == 0
    assert report["summary"]["missing_sweep_point_count"] == 1
    assert report["summary"]["gates"]["no_sweep_point_gaps"] is False
    assert report["summary"]["gates"]["adaptive_sampling_complete"] is False
    assert report["summary"]["accepted"] is False
    assert any(
        "geometry calibration unresolved" in error
        for error in report["adaptive_sampling"]["errors"]
    )


def test_completed_attempt_semantics_reconcile_with_ingested_counts(tmp_path: Path) -> None:
    document = _adaptive_document()
    diagnostics = document["results"][0]["adaptive_sampling"]  # type: ignore[index]
    cell_key = "A01:C-DENSE:v0:o50:victim_first"
    diagnostics.update({  # type: ignore[union-attr]
        "issued_attempt_counts": {cell_key: 55},
        "completed_attempt_counts": {cell_key: 55},
        "ingested_attempt_counts": {cell_key: 52},
        "discarded_after_stop_counts": {cell_key: 3},
        "actual_attempt_counts": {cell_key: 55},
        "actual_attempt_counts_semantics": (
            "completed treatment attempts; use ingested_attempt_counts "
            "for controller-retained attempts"
        ),
        "attempt_reconciliation": {
            "issued": 55,
            "completed": 55,
            "ingested": 52,
            "discarded_after_stop": 3,
            "passed": True,
        },
    })
    source = tmp_path / "adaptive_accounted.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert report["adaptive_sampling"]["malformed_or_failed_diagnostic_count"] == 0
    assert report["summary"]["accepted"] is True

    diagnostics["ingested_attempt_counts"] = {cell_key: 51}  # type: ignore[index]
    source.write_text(json.dumps(document), encoding="utf-8")
    malformed_samples, _ = load_samples(source)
    malformed_report = validate_samples(
        malformed_samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )
    assert malformed_report["summary"]["gates"]["adaptive_sampling_complete"] is False


def test_unknown_health_fields_remain_strict_failures() -> None:
    normalized = [{
        "case_id": "A01",
        "case_key": "A01:C-DENSE",
        "seq_len": 12_288,
        "sample_index": index,
        "dispatch_mode": "victim_first",
        "victim_index": 0,
        "victim_family": "compute",
        "aggressor_family": "p2p_send",
        "requested_overlap_pct": 50.0,
        "feasible": True,
        "valid": True,
        "geometry_error_pct": None,
        "geometry_tolerance_pct": 10.0,
        "baseline_drift_pct": None,
        "correct": None,
        "victim_isolated_us": 20_000.0,
    } for index in range(50)]
    report = validate_samples(
        normalized,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
    )

    assert report["summary"]["gates"]["geometry_known"] is False
    assert report["summary"]["gates"]["drift_known"] is False
    assert report["summary"]["gates"]["correctness_passed"] is False
    assert report["summary"]["accepted"] is False


def _strict_clock_rows(
    generation: int = 1, label: str = "suite_initial"
) -> list[dict[str, object]]:
    return [
        {
            "global_rank": rank,
            "clock_calibration_best_rtt_ns": 101 + rank,
            "clock_offset_ns": rank - 1,
            "clock_calibration_generation": generation,
            "clock_calibration_timestamp_global_ns": 10_000 + rank,
            "clock_calibration_local_monotonic_ns": 9_999 + rank,
            "clock_calibration_label": label,
            "clock_calibration_half_rtt_uncertainty_ns": (102 + rank) // 2,
        }
        for rank in range(2)
    ]


def _fixed_disposition_document(rows: list[dict[str, object]]) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    implementation_path = "src/pairwise_overlap_corpus_validation.py"
    implementation = {
        implementation_path: hashlib.sha256(
            (repo_root / implementation_path).read_bytes()
        ).hexdigest()
    }
    calibration_rows = _strict_clock_rows()
    calibration_entry = {
        "schema_version": 2,
        "generation": 1,
        "session_index": 0,
        "label": "suite_initial",
        "rows": calibration_rows,
        "rows_sha256": _canonical_test_hash(calibration_rows),
    }
    result = {
        "assignment": {
            "case_key": "A01:C-DENSE",
            "participants": [{"category": "compute"}],
        },
        "correctness": [{"operation_index": 0, "passed": True}],
        "samples": rows,
        "clock_calibration_history": calibration_rows,
    }
    return {
        "schema_version": 2,
        "configuration": {
            "schema_version": 2,
            "sampling": {"mode": "fixed"},
            "scheduling": {
                "gpu_global_time_refresh_trigger_age_ns": 45_000_000_000,
                "gpu_global_time_refresh_headroom_ns": 15_000_000_000,
                "gpu_global_time_refresh_max_age_ns": 60_000_000_000,
                "gpu_global_time_refresh_boundaries": ["fixed_treatment"],
            },
            "topology": {"world_size": 2},
            "implementation_sha256": implementation,
        },
        "configuration_identity": {
            "restart_identity_schema_version": 2,
            "implementation_sha256": implementation,
        },
        "placement": {
            "validation": {
                "world_size": 2,
                "worker_implementation_sha256": implementation,
                "worker_implementation_source_match": True,
            },
            "ranks": [
                {
                    "global_rank": rank,
                    "imported_worker_implementation_sha256": implementation,
                }
                for rank in range(2)
            ],
        },
        "clock_calibration_schema_version": 2,
        "clock_calibration_generation": 1,
        "clock_calibration": calibration_rows,
        "clock_calibration_history": [calibration_entry],
        "restart_provenance": {
            "execution_sessions": [{
                "session_index": 0,
                "clock_calibration_generation": 1,
                "clock_calibration_history": [calibration_entry],
            }],
        },
        "results": [result],
    }


def _strict_static_sample(index: int) -> dict[str, object]:
    return _sample(
        index,
        target_feasible_from_isolated=False,
        target_feasible_from_duration=False,
        target_feasible_after_constraints=False,
        target_feasible_after_runtime_calibration=False,
        runtime_geometry_calibration_status="isolated_duration_bound_infeasible",
        valid=False,
        delay_rebase_evidence={
            "policy": "static_calibrated_delays_v1",
            "dynamic_rebase_applied": False,
            "static_policy_reason": "duration_bound_infeasible",
        },
    )


def _validate_fixed_disposition_document(
    tmp_path: Path,
    name: str,
    rows: list[dict[str, object]],
    *,
    iterations: int,
) -> dict[str, object]:
    source = tmp_path / f"{name}.json"
    source.write_text(
        json.dumps(_fixed_disposition_document(rows)), encoding="utf-8"
    )
    samples, _ = load_samples(source)
    return validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        expected_fixed_iterations=iterations,
    )


def test_fixed_disposition_excludes_legacy_unresolved_from_estimator(
    tmp_path: Path,
) -> None:
    report = _validate_fixed_disposition_document(
        tmp_path,
        "legacy_unresolved",
        [
            _sample(0),
            _sample(
                1,
                valid=False,
                start_anchor_scoring_allowed=False,
                delay_rebase_evidence={
                    "start_anchor_resolution_status": (
                        "corrected_delay_confirmation_unresolved"
                    )
                },
            ),
        ],
        iterations=2,
    )
    summary = report["summary"]
    assert summary["accepted"] is True
    assert summary["sample_count"] == 2
    assert summary["estimator_valid_sample_count"] == 1
    assert summary["runtime_start_geometry_unresolved_count"] == 1
    assert summary["unscored_count"] == 1
    assert summary["scored_feasible_count"] == 1
    assert summary["rejection_count"] == 0
    assert report["fixed_sampling"]["passed"] is True


def test_explicit_disposition_scores_only_confirmed_eligible_rows(
    tmp_path: Path,
) -> None:
    good = _sample(
        0,
        runtime_start_geometry_disposition="confirmed",
        runtime_start_geometry_scoring_eligible=True,
        runtime_start_geometry_treatment_executed=True,
    )
    report = _validate_fixed_disposition_document(
        tmp_path, "explicit_good", [good], iterations=1
    )
    assert report["summary"]["accepted"] is True
    assert report["summary"]["estimator_valid_sample_count"] == 1

    for name, updates in {
        "confirmed_ineligible": {
            "runtime_start_geometry_scoring_eligible": False,
        },
        "unresolved_eligible": {
            "runtime_start_geometry_disposition": (
                "runtime_start_geometry_unresolved"
            ),
            "runtime_start_geometry_scoring_eligible": True,
        },
        "unknown_disposition": {
            "runtime_start_geometry_disposition": "mystery",
        },
    }.items():
        corrupted = dict(good)
        corrupted.update(updates)
        failed = _validate_fixed_disposition_document(
            tmp_path, name, [corrupted], iterations=1
        )
        assert failed["summary"]["accepted"] is False, name
        assert failed["summary"][
            "runtime_start_geometry_contract_failure_count"
        ] == 1, name
        assert failed["summary"]["estimator_valid_sample_count"] == 0, name


def test_static_exemption_requires_exact_duration_bound_proof(
    tmp_path: Path,
) -> None:
    static = _strict_static_sample(0)
    report = _validate_fixed_disposition_document(
        tmp_path, "static_proven", [static], iterations=1
    )
    assert report["summary"]["accepted"] is True
    assert report["summary"]["runtime_start_geometry_static_exempt_count"] == 1
    assert report["summary"]["unscored_count"] == 1

    corruptions = {}
    for field in (
        "target_feasible_from_duration",
        "target_feasible_after_constraints",
        "target_feasible_after_runtime_calibration",
        "runtime_geometry_calibration_status",
    ):
        row = json.loads(json.dumps(static))
        row.pop(field)
        corruptions[f"missing_{field}"] = row
    for evidence_field in (
        "policy",
        "dynamic_rebase_applied",
        "static_policy_reason",
    ):
        row = json.loads(json.dumps(static))
        row["delay_rebase_evidence"].pop(evidence_field)
        corruptions[f"missing_{evidence_field}"] = row
    for name, row in corruptions.items():
        failed = _validate_fixed_disposition_document(
            tmp_path, name, [row], iterations=1
        )
        assert failed["summary"]["accepted"] is False, name
        assert failed["summary"][
            "runtime_start_geometry_contract_failure_count"
        ] == 1, name


def test_fixed_sampling_counts_unresolved_and_static_rows_but_requires_indices(
    tmp_path: Path,
) -> None:
    rows = [
        _sample(0),
        _sample(1, valid=False, start_anchor_scoring_allowed=False),
        _strict_static_sample(2),
    ]
    report = _validate_fixed_disposition_document(
        tmp_path, "fixed_exact", rows, iterations=3
    )
    assert report["fixed_sampling"]["passed"] is True
    assert report["summary"]["sample_count"] == 3
    assert report["summary"]["estimator_valid_sample_count"] == 1

    duplicate = json.loads(json.dumps(rows))
    duplicate[1]["sample_index"] = 0
    failed = _validate_fixed_disposition_document(
        tmp_path, "fixed_duplicate", duplicate, iterations=3
    )
    assert failed["fixed_sampling"]["passed"] is False
    assert failed["summary"]["gates"]["fixed_sampling_complete"] is False
    assert failed["fixed_sampling"]["failed_stratum_count"] == 1


def test_unresolved_only_feasible_stratum_is_a_coverage_hole(
    tmp_path: Path,
) -> None:
    report = _validate_fixed_disposition_document(
        tmp_path,
        "coverage_hole",
        [_sample(0, valid=False, start_anchor_scoring_allowed=False)],
        iterations=1,
    )
    assert report["fixed_sampling"]["passed"] is True
    assert report["summary"]["scored_feasible_count"] == 0
    assert report["summary"]["rejection_count"] == 0
    assert report["summary"]["raw_rejection_rate_pct"] == 0.0
    assert report["summary"]["coverage_hole_count"] == 1
    assert report["summary"]["gates"][
        "otherwise_feasible_strata_covered"
    ] is False
    assert report["summary"]["accepted"] is False


def test_generic_counterfactual_v2_gate_applies_to_fixed_results(
    tmp_path: Path,
) -> None:
    source = tmp_path / "fixed_legacy.json"
    source.write_text(
        json.dumps(_fixed_disposition_document([_sample(0)])),
        encoding="utf-8",
    )
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        expected_fixed_iterations=1,
        require_counterfactual_v2=True,
    )
    assert report["summary"]["gates"]["counterfactual_v2_required"] is False
    assert report["summary"]["accepted"] is False


def _canonical_test_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def _v2_consensus(
    payload: dict[str, object], label: str, decision_ns: int
) -> dict[str, object]:
    digest = _canonical_test_hash(payload)
    records = [
        {
            "global_rank": rank,
            "decision_global_ns": decision_ns + rank,
            "decision_digest_sha256": digest,
        }
        for rank in range(2)
    ]
    return {
        "label": label,
        "world_size": 2,
        "decision_digest_sha256": digest,
        "actor_decision_digests": {str(rank): digest for rank in range(2)},
        "actor_decision_records": records,
        "decision_global_ns_min": decision_ns,
        "decision_global_ns_max": decision_ns + 1,
        "all_ranks_agree": True,
    }


def _v2_adjacent_dynamic_sample(index: int = 0) -> dict[str, object]:
    mode = "coverage_50_v0_victim_first"
    selected = {"0": 0.25}
    final = {"0": 0.20}
    preliminary_decision = {
        "policy": "same_dispatch_k3_preliminary_residual_plus_k3_corrected_confirmation_v2",
        "sample_index": index,
        "treatment_mode": mode,
        "initial_delays_ms": selected,
        "corrected_delays_ms": final,
        "correction_ms": -0.05,
        "constraint_limited": False,
        "preliminary_geometry_errors_pct": [2.0, 1.0, 0.0],
    }
    preliminary_consensus = _v2_consensus(
        preliminary_decision,
        "start_anchor_preliminary_correction",
        200,
    )
    confirmation = {
        "schema_version": 1,
        "policy": "corrected_delay_common_focal_geometry_k3_median_two_of_three_v1",
        "repetitions": 3,
        "minimum_passing_repetitions": 2,
        "unscored": True,
        "measured_counterfactual_overlap_pct_by_repetition": [49.0, 50.0, 51.0],
        "measured_geometry_error_pct_by_repetition": [-1.0, 0.0, 1.0],
        "median_measured_counterfactual_overlap_pct": 50.0,
        "median_measured_geometry_error_pct": 0.0,
        "passing_repetition_count": 3,
        "robustly_within_tolerance": True,
        "constraint_limited": False,
        "resolution_status": "corrected_delay_confirmed",
        "runtime_start_geometry_disposition": "confirmed",
        "corrected_delays_ms": final,
    }
    confirmation_focal = [
        {
            "repetition_index": repetition,
            "operation_index": 0,
            "global_rank": 0,
            "sample_index": index,
            "mode": f"start_anchor_confirmation::{mode}::repeat={repetition}",
            "host_complete_global_ns": 300 + repetition,
        }
        for repetition in range(3)
    ]
    confirmation_completion = [dict(row) for row in confirmation_focal]
    confirmation.update({
        "raw_focal_start_rows_schema_version": 1,
        "raw_focal_start_rows": confirmation_focal,
        "raw_focal_start_rows_sha256": _canonical_test_hash(confirmation_focal),
        "raw_focal_start_row_count": 3,
        "all_endpoint_completion_rows_schema_version": 1,
        "all_endpoint_completion_rows": confirmation_completion,
        "all_endpoint_completion_rows_sha256": _canonical_test_hash(
            confirmation_completion
        ),
        "latest_all_endpoint_complete_global_ns": 302,
    })
    final_payload = {
        "policy": "corrected_delay_common_focal_geometry_k3_median_two_of_three_v1",
        "sample_index": index,
        "treatment_mode": mode,
        "corrected_delays_ms": final,
        "resolution_status": "corrected_delay_confirmed",
        "runtime_start_geometry_disposition": "confirmed",
        "passing_repetition_count": 3,
        "median_measured_geometry_error_pct": 0.0,
    }
    preliminary_focal = [
        {
            "repetition_index": repetition,
            "operation_index": 0,
            "global_rank": 0,
            "sample_index": index,
            "mode": f"start_anchor::{mode}::repeat={repetition}",
            "host_complete_global_ns": 100 + repetition,
        }
        for repetition in range(3)
    ]
    preliminary_completion = [dict(row) for row in preliminary_focal]
    anchor = {
        "schema_version": 1,
        "policy": "same_dispatch_k3_preliminary_residual_plus_k3_corrected_confirmation_v2",
        "repetitions": 3,
        "unscored": True,
        "sample_index": index,
        "treatment_mode": mode,
        "configured_later_group_gap_ms": 0.25,
        "actual_focal_gpu_start_gaps_ms": [0.22, 0.21, 0.20],
        "launch_residuals_ms": [-0.03, -0.04, -0.05],
        "preliminary_decision": preliminary_decision,
        "preliminary_consensus": preliminary_consensus,
        "confirmation": confirmation,
        "post_confirmation_consensus": _v2_consensus(
            final_payload,
            "start_anchor_post_confirmation_decision",
            400,
        ),
        "raw_focal_start_rows_schema_version": 1,
        "raw_focal_start_rows": preliminary_focal,
        "raw_focal_start_rows_sha256": _canonical_test_hash(preliminary_focal),
        "raw_focal_start_row_count": 3,
        "all_endpoint_completion_rows_schema_version": 1,
        "all_endpoint_completion_rows": preliminary_completion,
        "all_endpoint_completion_rows_sha256": _canonical_test_hash(
            preliminary_completion
        ),
        "latest_anchor_all_endpoint_complete_global_ns": 102,
        "confirmation_rows_schema_version": 1,
        "confirmation_rows": confirmation_completion,
        "confirmation_rows_sha256": _canonical_test_hash(
            confirmation_completion
        ),
        "confirmation_row_count": len(confirmation_completion),
        "final_delays_ms": final,
        "resolution_status": "corrected_delay_confirmed",
    }
    calibration = _strict_clock_rows()[0]
    pre_rows = [{
        "operation_index": 0,
        "duration_ms": 1.0,
        "clock_calibration_age_ns": 100,
        **calibration,
    }]
    evidence = {
        "schema_version": 1,
        "policy": "adjacent_pre_runtime_start_rebase_v2",
        "dynamic_rebase_applied": True,
        "static_policy_reason": None,
        "sample_index": index,
        "treatment_mode": mode,
        "pre_durations_ms": {"0": 1.0},
        "pre_rows_schema_version": 1,
        "pre_rows": pre_rows,
        "pre_rows_sha256": _canonical_test_hash(pre_rows),
        "pre_row_count": 1,
        "selected_calibration_delays_ms": selected,
        "final_delays_ms": final,
        "start_anchor_policy": anchor["policy"],
        "start_anchor_evidence": anchor,
        "start_anchor_evidence_sha256": _canonical_test_hash(anchor),
        "runtime_start_geometry_disposition_explicit": True,
        "runtime_start_geometry_disposition": "confirmed",
        "runtime_start_geometry_treatment_executed": True,
        "runtime_start_geometry_scoring_eligible": True,
        "start_anchor_resolution_status": "corrected_delay_confirmed",
        "start_anchor_scoring_allowed": True,
    }
    return _sample(
        index,
        runtime_start_geometry_disposition_explicit=True,
        runtime_start_geometry_disposition="confirmed",
        runtime_start_geometry_treatment_executed=True,
        runtime_start_geometry_scoring_eligible=True,
        start_anchor_resolution_status="corrected_delay_confirmed",
        start_anchor_scoring_allowed=True,
        delay_rebase_evidence=evidence,
        delay_rebase_evidence_sha256=_canonical_test_hash(evidence),
    )


def _rehash_v2_adjacent_sample(row: dict[str, object]) -> None:
    evidence = row["delay_rebase_evidence"]
    anchor = evidence["start_anchor_evidence"]
    evidence["start_anchor_evidence_sha256"] = _canonical_test_hash(anchor)
    row["delay_rebase_evidence_sha256"] = _canonical_test_hash(evidence)


def test_adjacent_pre_v2_reconstructs_distinct_k3_consensus_and_causality(
    tmp_path: Path,
) -> None:
    row = _v2_adjacent_dynamic_sample()
    source = tmp_path / "adjacent_v2.json"
    source.write_text(
        json.dumps(_fixed_disposition_document([row])), encoding="utf-8"
    )
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        expected_fixed_iterations=1,
        require_adjacent_pre_rebased_delays=True,
    )
    assert report["adjacent_pre_rebased_delays"]["passed"] is True
    assert report["summary"]["gates"][
        "adjacent_pre_rebased_delays_required"
    ] is True
    assert report["summary"]["accepted"] is True


def test_adjacent_pre_v2_corruptions_fail_closed(tmp_path: Path) -> None:
    mutations: dict[str, dict[str, object]] = {}
    bad_outer = _v2_adjacent_dynamic_sample()
    bad_outer["delay_rebase_evidence_sha256"] = "0" * 64
    mutations["outer hash"] = bad_outer
    bad_anchor = _v2_adjacent_dynamic_sample()
    bad_anchor["delay_rebase_evidence"]["start_anchor_evidence_sha256"] = "0" * 64
    bad_anchor["delay_rebase_evidence_sha256"] = _canonical_test_hash(
        bad_anchor["delay_rebase_evidence"]
    )
    mutations["anchor hash"] = bad_anchor
    bad_pre = _v2_adjacent_dynamic_sample()
    bad_pre["delay_rebase_evidence"]["pre_rows_sha256"] = "0" * 64
    _rehash_v2_adjacent_sample(bad_pre)
    mutations["pre rows hash"] = bad_pre
    bad_consensus = _v2_adjacent_dynamic_sample()
    bad_consensus["delay_rebase_evidence"]["start_anchor_evidence"][
        "preliminary_consensus"
    ]["decision_digest_sha256"] = "0" * 64
    _rehash_v2_adjacent_sample(bad_consensus)
    mutations["preliminary consensus"] = bad_consensus
    bad_causality = _v2_adjacent_dynamic_sample()
    post = bad_causality["delay_rebase_evidence"]["start_anchor_evidence"][
        "post_confirmation_consensus"
    ]
    post["decision_global_ns_min"] = 300
    _rehash_v2_adjacent_sample(bad_causality)
    mutations["post-confirmation causality"] = bad_causality
    bad_delay = _v2_adjacent_dynamic_sample()
    bad_delay["delay_rebase_evidence"]["final_delays_ms"] = {"0": 9.0}
    _rehash_v2_adjacent_sample(bad_delay)
    mutations["final delays"] = bad_delay

    for name, mutated in mutations.items():
        source = tmp_path / f"adjacent_corrupt_{name.replace(' ', '_')}.json"
        source.write_text(
            json.dumps(_fixed_disposition_document([mutated])), encoding="utf-8"
        )
        samples, _ = load_samples(source)
        report = validate_samples(
            samples,
            expected_sequences=[12_288],
            expected_cases=["A01:C-DENSE"],
            expected_coverage_targets=[50.0],
            expected_dispatch_modes=["victim_first"],
            expected_fixed_iterations=1,
            require_adjacent_pre_rebased_delays=True,
        )
        assert report["adjacent_pre_rebased_delays"]["passed"] is False, name
        assert report["adjacent_pre_rebased_delays"]["failed_row_count"] == 1, name
        assert report["summary"]["accepted"] is False, name


def test_adjacent_pre_static_requires_exact_duration_bound_proof(
    tmp_path: Path,
) -> None:
    row = _strict_static_sample(0)
    row.update({
        "runtime_start_geometry_disposition_explicit": True,
        "runtime_start_geometry_disposition": "not_applicable_static",
        "runtime_start_geometry_treatment_executed": True,
        "runtime_start_geometry_scoring_eligible": False,
    })
    evidence = row["delay_rebase_evidence"]
    evidence.update({
        "schema_version": 1,
        "sample_index": 0,
        "treatment_mode": "coverage_50_v0_victim_first",
        "target_feasible_from_duration": False,
        "target_feasible_after_constraints": False,
        "target_feasible_after_runtime_calibration": False,
        "runtime_geometry_calibration_status": (
            "isolated_duration_bound_infeasible"
        ),
        "runtime_start_geometry_disposition_explicit": True,
        "runtime_start_geometry_disposition": "not_applicable_static",
        "runtime_start_geometry_treatment_executed": True,
        "runtime_start_geometry_scoring_eligible": False,
        "final_delays_ms": {"0": 0.0},
    })
    row["delay_rebase_evidence_sha256"] = _canonical_test_hash(evidence)
    report = _validate_fixed_disposition_document(
        tmp_path, "adjacent_static", [row], iterations=1
    )
    # Re-run with the authoritative adjacent-pre gate enabled.
    source = tmp_path / "adjacent_static.json"
    samples, _ = load_samples(source)
    report = validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        expected_fixed_iterations=1,
        require_adjacent_pre_rebased_delays=True,
    )
    assert report["adjacent_pre_rebased_delays"]["passed"] is True
    assert report["summary"]["accepted"] is True

    corrupted = json.loads(json.dumps(row))
    corrupted["delay_rebase_evidence"].pop("static_policy_reason")
    corrupted["delay_rebase_evidence_sha256"] = _canonical_test_hash(
        corrupted["delay_rebase_evidence"]
    )
    failed_source = tmp_path / "adjacent_static_corrupt.json"
    failed_source.write_text(
        json.dumps(_fixed_disposition_document([corrupted])), encoding="utf-8"
    )
    failed_samples, _ = load_samples(failed_source)
    failed = validate_samples(
        failed_samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        expected_fixed_iterations=1,
        require_adjacent_pre_rebased_delays=True,
    )
    assert failed["adjacent_pre_rebased_delays"]["passed"] is False
    assert failed["summary"]["accepted"] is False



def _validate_strict_provenance_document(
    tmp_path: Path, name: str, document: dict[str, object]
) -> dict[str, object]:
    source = tmp_path / f"{name}.json"
    source.write_text(json.dumps(document), encoding="utf-8")
    samples, _ = load_samples(source)
    return validate_samples(
        samples,
        expected_sequences=[12_288],
        expected_cases=["A01:C-DENSE"],
        expected_coverage_targets=[50.0],
        expected_dispatch_modes=["victim_first"],
        expected_fixed_iterations=1,
        require_adjacent_pre_rebased_delays=True,
    )


def test_strict_final_json_provenance_is_self_contained(tmp_path: Path) -> None:
    document = _fixed_disposition_document([_v2_adjacent_dynamic_sample()])
    report = _validate_strict_provenance_document(
        tmp_path, "strict_provenance_valid", document
    )
    provenance = report["strict_final_json_provenance"]
    assert provenance["passed"] is True
    assert provenance["implementation_files_checked"] == 5
    assert provenance["retained_digest_count_checked"] >= 8
    assert provenance["clock_rows_checked"] == 6
    assert provenance["clock_history_entries_checked"] == 1
    assert provenance["clock_age_rows_checked"] >= 1
    assert report["summary"]["gates"][
        "strict_final_json_provenance_required"
    ] is True


def test_strict_provenance_rejects_local_implementation_mismatch(
    tmp_path: Path,
) -> None:
    document = _fixed_disposition_document([_v2_adjacent_dynamic_sample()])
    document["configuration"]["implementation_sha256"] = {
        "src/pairwise_overlap_corpus_validation.py": "0" * 64
    }
    report = _validate_strict_provenance_document(
        tmp_path, "strict_bad_implementation", document
    )
    assert report["strict_final_json_provenance"]["passed"] is False
    assert any(
        "does not match the local file" in error
        for error in report["strict_final_json_provenance"]["errors"]
    )
    assert report["summary"]["accepted"] is False


def test_strict_provenance_rejects_unrecomputable_confirmation_digest(
    tmp_path: Path,
) -> None:
    row = _v2_adjacent_dynamic_sample()
    anchor = row["delay_rebase_evidence"]["start_anchor_evidence"]
    anchor.pop("confirmation_rows")
    anchor.pop("confirmation_row_count")
    _rehash_v2_adjacent_sample(row)
    document = _fixed_disposition_document([row])
    report = _validate_strict_provenance_document(
        tmp_path, "strict_missing_confirmation_rows", document
    )
    provenance = report["strict_final_json_provenance"]
    assert provenance["passed"] is False
    assert any(
        "confirmation_rows_sha256: retained digest lacks confirmation_rows"
        in error
        for error in provenance["errors"]
    )
    assert report["adjacent_pre_rebased_delays"]["passed"] is False
    assert report["summary"]["accepted"] is False


def test_strict_provenance_rejects_clock_history_corruption(
    tmp_path: Path,
) -> None:
    mutations = {}
    missing_generation = _fixed_disposition_document(
        [_v2_adjacent_dynamic_sample()]
    )
    missing_generation.pop("clock_calibration_generation")
    mutations["missing_generation"] = missing_generation

    stale_digest = _fixed_disposition_document([_v2_adjacent_dynamic_sample()])
    stale_digest["clock_calibration_history"][0]["rows"][0][
        "clock_offset_ns"
    ] += 1
    mutations["stale_rows_digest"] = stale_digest

    session_mismatch = _fixed_disposition_document(
        [_v2_adjacent_dynamic_sample()]
    )
    session_mismatch["restart_provenance"]["execution_sessions"][0][
        "clock_calibration_generation"
    ] = 9
    mutations["session_generation"] = session_mismatch

    missing_result_history = _fixed_disposition_document(
        [_v2_adjacent_dynamic_sample()]
    )
    missing_result_history["results"][0].pop("clock_calibration_history")
    mutations["missing_result_history"] = missing_result_history

    measurement_mismatch_row = _v2_adjacent_dynamic_sample()
    measurement_evidence = measurement_mismatch_row["delay_rebase_evidence"]
    measurement_evidence["pre_rows"][0][
        "clock_calibration_best_rtt_ns"
    ] += 2
    measurement_evidence["pre_rows_sha256"] = _canonical_test_hash(
        measurement_evidence["pre_rows"]
    )
    _rehash_v2_adjacent_sample(measurement_mismatch_row)
    mutations["measurement_clock_reference"] = _fixed_disposition_document(
        [measurement_mismatch_row]
    )

    expired_clock_row = _v2_adjacent_dynamic_sample()
    expired_evidence = expired_clock_row["delay_rebase_evidence"]
    expired_evidence["pre_rows"][0][
        "clock_calibration_age_ns"
    ] = 60_000_000_001
    expired_evidence["pre_rows_sha256"] = _canonical_test_hash(
        expired_evidence["pre_rows"]
    )
    _rehash_v2_adjacent_sample(expired_clock_row)
    mutations["expired_clock_reference"] = _fixed_disposition_document(
        [expired_clock_row]
    )

    for name, document in mutations.items():
        report = _validate_strict_provenance_document(
            tmp_path, f"strict_clock_{name}", document
        )
        assert report["strict_final_json_provenance"]["passed"] is False, name
        assert report["summary"]["gates"][
            "strict_final_json_provenance_required"
        ] is False, name
        assert report["summary"]["accepted"] is False, name


def _merged_strict_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], list[Path]]:
    from scripts.merge_pairwise_overlap_results import merge_documents

    sources: list[tuple[Path, dict[str, object]]] = []
    paths: list[Path] = []
    for index, run_id in enumerate(("source-a", "source-b")):
        document = _fixed_disposition_document(
            [_v2_adjacent_dynamic_sample()]
        )
        document["run_id"] = run_id
        document["environment"] = {
            "all_workers_default_nccl_behavior": True,
            "source_index": index,
        }
        path = tmp_path / f"{run_id}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        sources.append((path, document))
        paths.append(path)
    return merge_documents(sources), paths


def _merged_provenance(document: dict[str, object]) -> dict[str, object]:
    from src.pairwise_overlap_corpus_validation import (
        _validate_strict_document_provenance,
    )

    return _validate_strict_document_provenance(document)


def _json_copy(value: object) -> object:
    return json.loads(json.dumps(value))


def test_strict_provenance_accepts_lossless_merged_wrapper(
    tmp_path: Path,
) -> None:
    merged, _ = _merged_strict_fixture(tmp_path)
    report = _merged_provenance(merged)

    assert report["passed"] is True
    assert report["error_count"] == 0
    assert report["merge_source_count_checked"] == 2
    assert [row["source_run_id"] for row in report["source_reports"]] == [
        "source-a",
        "source-b",
    ]
    assert report["implementation_files_checked"] == 10
    assert report["clock_rows_checked"] == 12
    assert report["clock_history_entries_checked"] == 2


def test_merged_strict_provenance_rejects_source_and_fingerprint_tampering(
    tmp_path: Path,
) -> None:
    merged, paths = _merged_strict_fixture(tmp_path)

    bad_fingerprint = _json_copy(merged)
    bad_fingerprint["source_runs"][0]["document_sha256"] = "0" * 64
    fingerprint_report = _merged_provenance(bad_fingerprint)
    assert fingerprint_report["passed"] is False
    assert any(
        "document_sha256 mismatch" in error
        for error in fingerprint_report["errors"]
    )

    reloaded = json.loads(paths[0].read_text(encoding="utf-8"))
    reloaded["environment"]["post_merge_tamper"] = True
    paths[0].write_text(json.dumps(reloaded), encoding="utf-8")
    source_report = _merged_provenance(merged)
    assert source_report["passed"] is False
    assert any(
        "document_sha256 mismatch" in error
        for error in source_report["errors"]
    )


def test_merged_strict_provenance_rejects_flattened_result_tampering(
    tmp_path: Path,
) -> None:
    merged, _ = _merged_strict_fixture(tmp_path)
    mutations: dict[str, dict[str, object]] = {}

    dropped = _json_copy(merged)
    dropped["results"].pop()
    mutations["dropped"] = dropped

    duplicated = _json_copy(merged)
    duplicated["results"].append(_json_copy(duplicated["results"][-1]))
    mutations["duplicated"] = duplicated

    reordered = _json_copy(merged)
    reordered["results"].reverse()
    mutations["reordered"] = reordered

    payload = _json_copy(merged)
    payload["results"][0]["samples"][0]["victim_isolated_us"] += 1.0
    mutations["payload"] = payload

    source_id = _json_copy(merged)
    source_id["results"][0]["source_run_id"] = "wrong-source"
    mutations["source_id"] = source_id

    for name, mutation in mutations.items():
        report = _merged_provenance(mutation)
        assert report["passed"] is False, name
        assert any(
            "flattened result" in error
            for error in report["errors"]
        ), name


def test_merged_strict_provenance_rejects_id_order_and_metadata_tampering(
    tmp_path: Path,
) -> None:
    merged, _ = _merged_strict_fixture(tmp_path)
    mutations: dict[str, dict[str, object]] = {}

    duplicate_id = _json_copy(merged)
    duplicate_id["source_runs"][1]["source_run_id"] = (
        duplicate_id["source_runs"][0]["source_run_id"]
    )
    mutations["duplicate_id"] = duplicate_id

    wrong_id = _json_copy(merged)
    wrong_id["source_runs"][0]["source_run_id"] = "wrong-source"
    mutations["wrong_id"] = wrong_id

    record_order = _json_copy(merged)
    record_order["source_runs"].reverse()
    mutations["record_order"] = record_order

    source_metadata = _json_copy(merged)
    source_metadata["source_runs"][0]["configuration"]["topology"][
        "world_size"
    ] = 17
    mutations["source_metadata"] = source_metadata

    sequence_lengths = _json_copy(merged)
    sequence_lengths["configuration"]["sequence_lengths"] = [1]
    mutations["sequence_lengths"] = sequence_lengths

    default_nccl = _json_copy(merged)
    default_nccl["environment"][
        "all_source_runs_default_nccl_behavior"
    ] = False
    mutations["default_nccl"] = default_nccl

    placement = _json_copy(merged)
    placement["placement"]["validation"]["world_size"] = 17
    mutations["placement"] = placement

    clock = _json_copy(merged)
    clock["clock_calibration"][0]["clock_offset_ns"] += 1
    mutations["clock"] = clock

    missing_source = _json_copy(merged)
    missing_source["source_runs"][0]["path"] = str(
        tmp_path / "does-not-exist.json"
    )
    mutations["missing_source"] = missing_source

    for name, mutation in mutations.items():
        report = _merged_provenance(mutation)
        assert report["passed"] is False, name
        assert report["errors"], name
