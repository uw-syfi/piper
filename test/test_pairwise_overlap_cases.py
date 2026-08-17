from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from collections import Counter

import pytest


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT if (ROOT / "out").is_dir() else ROOT.parent


def load_script(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_compiler_and_minimal_communicator_plan() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    matrix = DATA_ROOT / (
        "out/20260809_pairwise_overlap_foundation/"
        "EXACT_KERNEL_EXECUTION_MATRIX.json"
    )
    cases = module.load_cases(matrix, ["A04:C-GROUPED", "A20", "M02"])
    assert [row["case_key"] for row in cases] == [
        "A04:C-GROUPED", "A20", "M02"
    ]
    plan = module.communicator_plan(cases)
    assert plan["dp_pp_ranks"] == [1]
    assert plan["ep_pp_ranks"] == []
    assert plan["p2p_groups"] == [(2, 0), (2, 1), (3, 0), (3, 1)]


def test_adaptive_plan_excludes_stopped_cells_and_preserves_attempt_counts() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_adaptive_plan",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    assignment = {"case_key": "A03:C-BW"}
    specs = [
        {
            "mode": "coverage_50_v0_victim_first",
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "victim_first",
        },
        {
            "mode": "coverage_50_v0_aggressor_first",
            "victim_index": 0,
            "requested_overlap_pct": 50.0,
            "dispatch_mode": "aggressor_first",
        },
        {
            "mode": "coverage_90_v0_victim_first",
            "victim_index": 0,
            "requested_overlap_pct": 90.0,
            "dispatch_mode": "victim_first",
        },
    ]
    first = module.adaptive_cell_key(assignment, specs[0])
    second = module.adaptive_cell_key(assignment, specs[1])
    batches = module.adaptive_plan_batches(
        assignment, specs, {first: 2, second: 1}
    )

    assert [[spec["mode"] for spec in batch] for batch in batches] == [
        [
            "coverage_50_v0_victim_first",
            "coverage_50_v0_aggressor_first",
        ],
        ["coverage_50_v0_victim_first"],
    ]
    assert all(
        spec["mode"] != "coverage_90_v0_victim_first"
        for batch in batches for spec in batch
    )


def test_adaptive_controller_row_uses_global_geometry_and_paired_baseline() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_adaptive_reduce",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    assignment = {"case_id": "A03", "case_key": "A03:C-BW"}
    participants = [
        {"focal_dp_rank": 0, "focal_pp_rank": 0},
        {"focal_dp_rank": 0, "focal_pp_rank": 1},
    ]
    spec = {
        "mode": "coverage_50_v0_victim_first",
        "victim_index": 0,
        "aggressor_indices": [1],
        "requested_overlap_pct": 50.0,
        "dispatch_mode": "victim_first",
        "target_feasible_from_isolated": True,
        "target_feasible_after_runtime_calibration": True,
    }
    rows = [
        {
            "mode": "isolated_pre", "operation_index": 0,
            "sample_index": 3, "duration_ms": 10.0,
            "bracket_for_mode": spec["mode"],
        },
        {
            "mode": "isolated_post", "operation_index": 0,
            "sample_index": 3, "duration_ms": 10.0,
            "bracket_for_mode": spec["mode"],
        },
        {
            "mode": "isolated_pre", "operation_index": 1,
            "sample_index": 3, "duration_ms": 10.0,
            "bracket_for_mode": spec["mode"],
        },
        {
            "mode": "isolated_post", "operation_index": 1,
            "sample_index": 3, "duration_ms": 12.0,
            "bracket_for_mode": spec["mode"],
        },
        {
            "mode": "isolated_pre", "operation_index": 0,
            "sample_index": 3, "duration_ms": 99.0,
            "bracket_for_mode": "another_treatment",
        },
        {
            "mode": "isolated_post", "operation_index": 0,
            "sample_index": 3, "duration_ms": 101.0,
            "bracket_for_mode": "another_treatment",
        },
        {
            "mode": spec["mode"], "operation_index": 0, "global_rank": 0,
            "sample_index": 3, "duration_ms": 12.0,
            "gpu_start_global_ns": 100_000_000,
            "gpu_end_global_ns": 110_000_000,
        },
        {
            "mode": spec["mode"], "operation_index": 1, "global_rank": 1,
            "sample_index": 3, "duration_ms": 10.0,
            "gpu_start_global_ns": 105_000_000,
            "gpu_end_global_ns": 115_000_000,
        },
    ]

    for row in rows:
        if row["mode"] == spec["mode"]:
            row.update({
                "runtime_start_geometry_disposition_explicit": True,
                "runtime_start_geometry_disposition": "confirmed",
                "runtime_start_geometry_treatment_executed": True,
                "runtime_start_geometry_scoring_eligible": True,
                "start_anchor_resolution_status": "corrected_delay_confirmed",
                "start_anchor_scoring_allowed": True,
                "delay_rebase_evidence": {"schema_version": 1},
                "delay_rebase_evidence_sha256": "a" * 64,
            })
    [reduced] = module.build_adaptive_controller_rows(
        assignment, participants, rows, [spec], 3, correct=True
    )
    assert reduced["geometry_error_pct"] == 0.0
    assert reduced["scheduled_geometry_error_pct"] == 0.0
    assert reduced["counterfactual_overlap_pct"] == 50.0
    assert reduced["geometry_validity_basis"] == (
        "counterfactual_intervals_from_actual_global_gpu_starts_and_"
        "matched_isolated_pre_post_durations"
    )
    assert reduced["geometry_baseline_drifts_pct"][0] == 0.0
    assert reduced["geometry_baseline_drifts_pct"][1] == pytest.approx(20.0)
    assert reduced["geometry_baseline_max_abs_drift_pct"] == pytest.approx(20.0)
    assert reduced["geometry_baseline_max_abs_drift_role"] == "diagnostic_only"
    assert reduced["victim_baseline_stable"] is True
    assert reduced["pre_post_counterfactual_geometry_robust"] is True
    assert reduced["baseline_drift_pct"] == 0.0
    assert reduced["victim_isolated_us"] == 10_000.0
    assert reduced["victim_slowdown_pct"] == pytest.approx(20.0)
    assert reduced["valid"] is True
    assert reduced["baseline_measurement_method"] == (
        "per_treatment_primed_focal_adjacent_odd_k_median_v1"
    )
    assert reduced["baseline_bracket_for_mode"] == spec["mode"]


def test_matched_baseline_rows_fail_closed_for_wrong_treatment_tag() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_matched_baseline",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    rows = [{
        "mode": "isolated_pre",
        "operation_index": 1,
        "sample_index": 7,
        "duration_ms": 3.0,
        "bracket_for_mode": "coverage_25_v1_victim_first",
    }]
    assert module.matched_baseline_rows(
        rows, "isolated_pre", "coverage_25_v1_victim_first", 1, 7
    ) == rows
    with pytest.raises(ValueError, match="missing matched isolated_pre"):
        module.matched_baseline_rows(
            rows, "isolated_pre", "coverage_75_v1_victim_first", 1, 7
        )


def test_focal_adjacent_bracket_orders_are_symmetric_around_treatment() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_focal_bracket",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    pre, post = module.focal_adjacent_bracket_orders(
        {"victim_index": 0, "order": [2, 0, 1]}, 3
    )
    assert pre == [2, 1, 0]
    assert post == [0, 2, 1]
    assert pre[-1] == post[0] == 0
    with pytest.raises(ValueError, match="every operation once"):
        module.focal_adjacent_bracket_orders(
            {"victim_index": 0, "order": [0, 0, 1]}, 3
        )


def test_coverage_targets_must_have_unique_finite_mode_tags() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_target_identity",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    module.validate_coverage_targets([0.0, 1.0, 12.5, 100.0])
    with pytest.raises(ValueError, match="must be unique"):
        module.validate_coverage_targets([25.0, 25.0])
    with pytest.raises(ValueError, match="finite"):
        module.validate_coverage_targets([float("nan")])
    with pytest.raises(ValueError, match="formatted stage-mode tags"):
        module.validate_coverage_targets([1.0000001, 1.0000002])


def test_adaptive_round_accounting_reconciles_and_annotates_tail() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_adaptive_accounting",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    config = module.AdaptiveBenchmarkControllerConfig(
        sampling=module.AdaptiveSamplingConfig(
            target_ci_half_width_pp=1.5,
            max_valid_samples=50,
            bootstrap_repetitions=100,
            bootstrap_seed=7,
        ),
        max_total_attempt_multiplier=1.0,
        attempt_batch_size=10,
    )
    controller = module.AdaptiveBenchmarkController(config)
    key = module.AdaptiveBenchmarkCellKey(
        "A11", 0, 0.0, "victim_first"
    )
    controller.register_cell(key, 20.0)
    before = {
        row.cell_key: row.to_dict() for row in controller.decisions()
    }
    rows = []
    for sample_index in range(55):
        rows.append({
            "case_key": "A11",
            "sample_index": sample_index,
            "victim_index": 0,
            "requested_overlap_pct": 0.0,
            "dispatch_mode": "victim_first",
            "target_feasible_after_runtime_calibration": True,
            "valid": True,
            "geometry_error_pct": 0.0,
            "geometry_tolerance_pct": 10.0,
            "baseline_drift_pct": 6.0 if sample_index == 0 else 0.0,
            "correct": True,
            "victim_isolated_us": 20_000.0,
            "victim_slowdown_pct": 1.0,
        })
    ingestion = controller.ingest_rows(rows)
    after = {
        row.cell_key: row.to_dict() for row in controller.decisions()
    }

    accounting = module.build_adaptive_round_accounting(
        {key.canonical: 55}, rows, before, after, config
    )

    assert ingestion == {
        "rows_seen": 55,
        "rows_ingested": 50,
        "rows_discarded_after_stop": 5,
    }
    assert accounting["totals"] == {
        "issued": 55,
        "completed": 55,
        "ingested": 50,
        "discarded_after_stop": 5,
        "reconciled": True,
    }
    assert Counter(
        row["disposition"] for row in accounting["sample_dispositions"]
    ) == {
        "ingested_valid": 49,
        "ingested_invalid": 1,
        "discarded_after_stop": 5,
    }

    raw_samples = [dict(row) for row in rows]
    module.annotate_adaptive_aggregate_samples(
        raw_samples,
        {"sample_dispositions": accounting["sample_dispositions"]},
    )
    assert raw_samples[0]["adaptive_controller_disposition"] == (
        "ingested_invalid"
    )
    assert raw_samples[0]["adaptive_controller_invalid_reasons"] == [
        "baseline_drift_out_of_tolerance"
    ]
    assert raw_samples[49]["adaptive_controller_valid_sample"] is True
    assert raw_samples[50]["adaptive_controller_disposition"] == (
        "discarded_after_stop"
    )
    assert raw_samples[50]["adaptive_controller_ingested"] is False
    assert raw_samples[50]["adaptive_controller_valid_sample"] is None

    with pytest.raises(RuntimeError, match="reconciliation failed"):
        module.build_adaptive_round_accounting(
            {key.canonical: 54}, rows, before, after, config
        )


def test_adaptive_cli_flags_are_opt_in(monkeypatch) -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_adaptive_cli",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    monkeypatch.setattr(sys, "argv", [
        "benchmark_pairwise_overlap_cases.py",
        "--output", "/tmp/adaptive.json",
        "--adaptive-sampling",
        "--adaptive-target-ci-half-width-pp", "1.25",
        "--adaptive-max-valid-samples", "250",
        "--adaptive-attempt-multiplier", "2.5",
        "--adaptive-batch-size", "7",
    ])
    args = module.parse_args()
    assert args.adaptive_sampling is True
    assert args.adaptive_target_ci_half_width_pp == 1.25
    assert args.adaptive_max_valid_samples == 250
    assert args.adaptive_attempt_multiplier == 2.5
    assert args.adaptive_batch_size == 7


def test_aggregate_case_accepts_variable_adaptive_attempt_counts() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_adaptive_aggregate",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    participants = [
        {
            "participant_id": "victim", "category": "compute",
            "focal_dp_rank": 0, "focal_pp_rank": 0,
        },
        {
            "participant_id": "aggressor", "category": "ag",
            "focal_dp_rank": 0, "focal_pp_rank": 1,
        },
    ]
    assignment = {
        "case_id": "A03", "case_key": "A03:C-BW",
        "participants": participants,
    }

    def stage_spec(mode: str, dispatch_mode: str) -> dict[str, object]:
        return {
            "mode": mode,
            "victim_index": 0,
            "aggressor_indices": [1],
            "dispatch_mode": dispatch_mode,
            "requested_overlap_pct": 50.0,
            "predicted_max_overlap_pct": 100.0,
            "predicted_scheduled_overlap_pct": 50.0,
            "predicted_constraint_error_pct": 0.0,
            "target_feasible_from_duration": True,
            "target_feasible_after_constraints": True,
            "target_feasible_from_isolated": True,
            "target_feasible_after_runtime_calibration": True,
        }

    specs = [
        stage_spec("coverage_50_v0_victim_first", "victim_first"),
        stage_spec("coverage_50_v0_aggressor_first", "aggressor_first"),
    ]
    unresolved = stage_spec(
        "coverage_75_v0_aggressor_first", "aggressor_first"
    )
    unresolved.update({
        "requested_overlap_pct": 75.0,
        "target_feasible_after_runtime_calibration": False,
        "geometry_cell_status": "calibration_unresolved",
        "runtime_geometry_calibration_status": "bounded_trials_exhausted",
        "geometry_calibration_history": [{
            "trial_index": 0,
            "geometry_error_pct": 22.0,
            "paired_baseline_drift_pct": 1.0,
        }],
    })
    specs.append(unresolved)

    def timing_row(
        mode: str, operation_index: int, sample_index: int,
        start_ms: float, duration_ms: float,
    ) -> dict[str, object]:
        start_ns = int(start_ms * 1e6)
        end_ns = int((start_ms + duration_ms) * 1e6)
        return {
            "mode": mode,
            "operation_index": operation_index,
            "global_rank": operation_index,
            "sample_index": sample_index,
            "duration_ms": duration_ms,
            "gpu_start_global_ns": start_ns,
            "gpu_end_global_ns": end_ns,
            "host_launch_global_ns": start_ns,
            "host_complete_global_ns": end_ns,
            "clock_calibration_generation": 1,
            "clock_calibration_label": "test",
            "clock_calibration_best_rtt_ns": 10,
            "clock_calibration_half_rtt_uncertainty_ns": 5,
        }

    rows = []
    for spec_index, spec in enumerate(specs):
        spec_samples = (0, 1) if spec_index == 0 else (0,)
        for sample_index in spec_samples:
            for phase in ("isolated_pre", "isolated_post"):
                for operation_index in (0, 1):
                    row = timing_row(
                        phase, operation_index, sample_index, 0.0, 10.0
                    )
                    row["bracket_for_mode"] = spec["mode"]
                    rows.append(row)
    for sample_index in (0, 1):
        rows.extend([
            timing_row(specs[0]["mode"], 0, sample_index, 0.0, 10.0),
            timing_row(specs[0]["mode"], 1, sample_index, 5.0, 10.0),
        ])
    rows.extend([
        timing_row(specs[1]["mode"], 0, 0, 5.0, 10.0),
        timing_row(specs[1]["mode"], 1, 0, 0.0, 10.0),
    ])
    for row in rows:
        if str(row["mode"]).startswith("coverage_"):
            row.update({
                "runtime_start_geometry_disposition_explicit": True,
                "runtime_start_geometry_disposition": "confirmed",
                "runtime_start_geometry_treatment_executed": True,
                "runtime_start_geometry_scoring_eligible": True,
                "start_anchor_resolution_status": "corrected_delay_confirmed",
                "start_anchor_scoring_allowed": True,
                "delay_rebase_evidence": {
                    "clock_calibration_generation": 1,
                },
                "delay_rebase_evidence_sha256": "a" * 64,
            })
    diagnostics = {
        "actual_attempt_counts": {
            "A03:C-BW:v0:o50:victim_first": 2,
            "A03:C-BW:v0:o50:aggressor_first": 1,
        }
    }
    rank_results = [{
        "global_rank": 0,
        "rows": rows,
        "stage_specs": specs,
        "correctness": [
            {"operation_index": 0, "passed": True},
            {"operation_index": 1, "passed": True},
        ],
        "compute_correctness_references": [],
        "operation_metadata": [],
        "trace_path": None,
        "peak_memory_bytes": 123,
        "adaptive_sampling": diagnostics,
        "runtime_health": {},
    }]

    result = module.aggregate_case(assignment, rank_results, None)
    assert len(result["samples"]) == 3
    assert [
        (sample["dispatch_mode"], sample["sample_index"])
        for sample in result["samples"]
    ] == [
        ("victim_first", 0),
        ("victim_first", 1),
        ("aggressor_first", 0),
    ]
    assert result["adaptive_sampling"] == diagnostics
    assert all(sample["scheduled_geometry_error_pct"] == 0.0 for sample in result["samples"])
    assert all(sample["geometry_validity_error_pct"] == 0.0 for sample in result["samples"])
    assert all(
        sample["concurrent_overlap_role"] == "reported_outcome_not_validity_gate"
        for sample in result["samples"]
    )
    assert result["stage_specs"] == specs
    retained_unresolved = result["stage_specs"][2]
    assert retained_unresolved["geometry_cell_status"] == "calibration_unresolved"
    assert retained_unresolved["geometry_calibration_history"] == [{
        "trial_index": 0,
        "geometry_error_pct": 22.0,
        "paired_baseline_drift_pct": 1.0,
    }]
    assert not any(
        sample["stage_mode"] == unresolved["mode"]
        for sample in result["samples"]
    )


def test_scaled_ragged_a2a_retains_transpose_contract() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_scale",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    source = DATA_ROOT / (
        "out/20260809_pairwise_overlap_foundation/ragged_a2a_payloads.json"
    )
    profile = json.loads(source.read_text(encoding="utf-8"))["payloads"][0]
    for seq_len in (12_288, 18_432, 49_152):
        scaled = module.scale_split_matrix(profile, seq_len)
        for source_rank in range(8):
            assert scaled[source_rank]["input_numel"] == (
                sum(scaled[source_rank]["input_split_rows"]) * 2_048
            )
            for destination_rank in range(8):
                assert (
                    scaled[source_rank]["input_split_rows"][destination_rank]
                    == scaled[destination_rank]["output_split_rows"][source_rank]
                )


def test_a20_converter_emits_both_directed_victims() -> None:
    module = load_script(
        "convert_a20_pairwise_results",
        "scripts/convert_a20_pairwise_results.py",
    )
    source = DATA_ROOT / "out/20260809_pairwise_overlap_benchmark/a20/results.json"
    document = json.loads(source.read_text(encoding="utf-8"))
    samples, participants = module.convert(document)
    assert len(samples) == 300
    assert len(participants) == 1_200
    assert {row["victim_family"] for row in samples} == {"p2p_send", "p2p_recv"}
    assert all(0.0 <= row["realized_overlap_pct"] <= 100.0 for row in samples)


def test_launch_latency_compensation_hits_gpu_start_geometry() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_compensation",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    result = module.compensate_launch_latencies(
        {0: 0.0, 1: 5.0}, {0: 2.0, 1: 1.0}, [0, 1]
    )
    starts = result["predicted_gpu_starts_ms"]
    assert starts[1] - starts[0] == 5.0
    assert result["order_clamped_indices"] == []
    assert result["host_delays_ms"][0] <= result["host_delays_ms"][1]


def test_launch_latency_compensation_preserves_dispatch_order() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_clamping",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    result = module.compensate_launch_latencies(
        {0: 0.0, 1: 1.0}, {0: 0.0, 1: 5.0}, [0, 1]
    )
    assert result["order_clamped_indices"] == [1]
    assert result["host_delays_ms"][0] <= result["host_delays_ms"][1]


def test_stage_feasibility_accounts_for_order_projection() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_feasibility",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    participants = [
        {"role": "victim"},
        {"role": "aggressor", "natural_group_offset_ms": 0.0},
        {"role": "aggressor", "natural_group_offset_ms": 5.0},
    ]
    specs = module.build_overlap_stage_specs(
        participants,
        {0: 10.0, 1: 5.0, 2: 5.0},
        {0: 0.0, 1: 0.0, 2: 20.0},
        [100.0],
    )
    victim_first = next(
        spec for spec in specs if spec["dispatch_mode"] == "victim_first"
    )
    assert victim_first["target_feasible_from_duration"]
    assert not victim_first["target_feasible_after_constraints"]
    assert not victim_first["target_feasible_from_isolated"]
    assert victim_first["order"] == [0, 1, 2]
    assert victim_first["order_clamped_indices"] == [2]


def test_duration_infeasible_projection_does_not_restore_feasibility() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_cumulative_feasibility",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    participants = [
        {"role": "victim"},
        {"role": "aggressor", "natural_group_offset_ms": 0.0},
    ]
    specs = module.build_overlap_stage_specs(
        participants,
        {0: 10.0, 1: 2.0},
        {0: 0.0, 1: 0.0},
        [25.0],
    )
    assert len(specs) == 2
    assert all(not spec["target_feasible_from_duration"] for spec in specs)
    assert all(spec["dispatch_order_projection_feasible"] for spec in specs)
    assert all(not spec["target_feasible_after_constraints"] for spec in specs)
    assert all(not spec["target_feasible_from_isolated"] for spec in specs)


def test_runtime_geometry_uses_global_focal_intervals() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_runtime_geometry",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    participants = [
        {"focal_dp_rank": 0, "focal_pp_rank": 0},
        {"focal_dp_rank": 0, "focal_pp_rank": 1},
    ]
    rows = [
        {
            "operation_index": 0,
            "global_rank": 0,
            "gpu_start_global_ns": 100_000_000,
            "gpu_end_global_ns": 110_000_000,
        },
        {
            "operation_index": 1,
            "global_rank": 1,
            "gpu_start_global_ns": 105_000_000,
            "gpu_end_global_ns": 115_000_000,
        },
        {
            "operation_index": 1,
            "global_rank": 0,
            "gpu_start_global_ns": 200_000_000,
            "gpu_end_global_ns": 300_000_000,
        },
    ]
    measured = module.measure_runtime_geometry(
        rows,
        participants,
        {
            "victim_index": 0,
            "aggressor_indices": [1],
            "requested_overlap_pct": 50.0,
        },
        isolated_victim_ms=10.0,
    )
    assert measured["union_overlap_ms"] == 5.0
    assert measured["realized_overlap_pct"] == 50.0
    assert measured["geometry_error_pct"] == 0.0


def test_relative_geometry_shift_preserves_dispatch_and_group_offsets() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_relative_shift",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    victim_first = {
        "victim_index": 0,
        "aggressor_indices": [1, 2],
        "dispatch_mode": "victim_first",
        "order": [0, 1, 2],
    }
    base_victim_first = {0: 0.0, 1: 2.0, 2: 5.0}
    assert module.relative_shift_bounds_ms(
        victim_first, base_victim_first, 100.0
    ) == (-2.0, 100.0)
    shifted = module.apply_relative_geometry_shift(
        victim_first, base_victim_first, -2.0
    )
    assert shifted == {0: 0.0, 1: 0.0, 2: 3.0}
    assert shifted[2] - shifted[1] == 3.0

    aggressor_first = {
        "victim_index": 0,
        "aggressor_indices": [1, 2],
        "dispatch_mode": "aggressor_first",
        "order": [1, 2, 0],
    }
    shifted = module.apply_relative_geometry_shift(
        aggressor_first, {0: 3.0, 1: 0.0, 2: 3.0}, 2.0
    )
    assert shifted == {0: 5.0, 1: 0.0, 2: 3.0}


def test_geometry_feedback_uses_overlap_time_for_inflated_victim() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_feedback_inflation",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    history = [{
        "relative_shift_ms": 0.0,
        "union_overlap_ms": 8.0,
        "realized_overlap_pct": 100.0,
        "geometry_error_pct": 100.0,
    }]
    feedback = module.choose_geometry_feedback_shift(
        history,
        requested_overlap_pct=0.0,
        isolated_victim_ms=1.0,
        minimum_shift_ms=0.0,
        maximum_shift_ms=100.0,
    )
    assert feedback["method"] == "overlap_time_feedback"
    assert feedback["next_shift_ms"] == 8.0
    assert not feedback["converged"]

    history.append({
        "relative_shift_ms": 8.0,
        "union_overlap_ms": 0.0,
        "realized_overlap_pct": 0.0,
        "geometry_error_pct": 0.0,
    })
    feedback = module.choose_geometry_feedback_shift(
        history,
        requested_overlap_pct=0.0,
        isolated_victim_ms=1.0,
        minimum_shift_ms=0.0,
        maximum_shift_ms=100.0,
    )
    assert feedback["converged"]


def test_geometry_feedback_bisects_bistable_bracket() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_feedback_bisection",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    history = [
        {
            "relative_shift_ms": 0.0,
            "union_overlap_ms": 1.0,
            "realized_overlap_pct": 100.0,
            "geometry_error_pct": 50.0,
        },
        {
            "relative_shift_ms": 10.0,
            "union_overlap_ms": 0.0,
            "realized_overlap_pct": 0.0,
            "geometry_error_pct": -50.0,
        },
    ]
    feedback = module.choose_geometry_feedback_shift(
        history,
        requested_overlap_pct=50.0,
        isolated_victim_ms=1.0,
        minimum_shift_ms=0.0,
        maximum_shift_ms=100.0,
    )
    assert feedback["method"] == "bracket_bisection"
    assert feedback["next_shift_ms"] == 5.0


def test_geometry_feedback_prefers_paired_overlap_error_ms() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_feedback_paired_baseline",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    feedback = module.choose_geometry_feedback_shift(
        [{
            "relative_shift_ms": 2.0,
            "union_overlap_ms": 99.0,
            "overlap_error_ms": 0.25,
            "realized_overlap_pct": 70.0,
            "geometry_error_pct": 20.0,
        }],
        requested_overlap_pct=50.0,
        # Deliberately inconsistent fallback denominator: direct paired
        # overlap_error_ms must win.
        isolated_victim_ms=1000.0,
        minimum_shift_ms=0.0,
        maximum_shift_ms=100.0,
    )
    assert feedback["method"] == "overlap_time_feedback"
    assert feedback["next_shift_ms"] == 2.25


def test_geometry_confirmation_requires_majority_at_one_fixed_shift() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_geometry_confirmation",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )

    def measurement(
        error: float, shift: float = 2.0, drift: float = 0.0
    ) -> dict[str, float]:
        union = 0.5 + error / 100.0
        return {
            "relative_shift_ms": shift,
            "union_overlap_ms": union,
            "realized_overlap_pct": 50.0 + error,
            "geometry_error_pct": error,
            "paired_baseline_ms": 1.0,
            "paired_baseline_drift_pct": drift,
            "overlap_error_ms": union - 0.5,
        }

    stable = module.summarize_geometry_confirmation_trials(
        [measurement(-3.0), measurement(2.0), measurement(4.0)], 10.0
    )
    assert stable["geometry_error_pct"] == 2.0
    assert stable["confirmation_passing_count"] == 3
    assert stable["robustly_converged"] is True

    # Median signed error alone is zero, but only one of the three trials is
    # actually within tolerance. This must not repeat the pilot's one-lucky-
    # trial acceptance failure.
    unstable = module.summarize_geometry_confirmation_trials(
        [measurement(-20.0), measurement(0.0), measurement(20.0)], 10.0
    )
    assert unstable["geometry_error_pct"] == 0.0
    assert unstable["confirmation_passing_count"] == 1
    assert unstable["robustly_converged"] is False

    unstable_baseline = module.summarize_geometry_confirmation_trials(
        [
            measurement(0.0, drift=-8.0),
            measurement(1.0, drift=0.5),
            measurement(2.0, drift=7.0),
        ],
        10.0,
    )
    assert unstable_baseline["geometry_passing_count"] == 3
    assert unstable_baseline["baseline_stable_count"] == 1
    assert unstable_baseline["jointly_passing_count"] == 1
    assert unstable_baseline["baseline_feedback_eligible"] is False
    assert unstable_baseline["robustly_converged"] is False

    unstable_aggressor_baseline = module.summarize_geometry_confirmation_trials(
        [
            {
                **measurement(error),
                "paired_baseline_drifts_pct": {0: 0.0, 1: 8.0},
                "geometry_baseline_max_abs_drift_pct": 8.0,
            }
            for error in (0.0, 1.0, 2.0)
        ],
        10.0,
    )
    assert unstable_aggressor_baseline["geometry_passing_count"] == 3
    assert unstable_aggressor_baseline["baseline_stable_count"] == 0
    assert unstable_aggressor_baseline["baseline_feedback_eligible"] is False
    assert unstable_aggressor_baseline["robustly_converged"] is False

    off_center = module.summarize_geometry_confirmation_trials(
        [measurement(8.0), measurement(9.0), measurement(10.0)], 10.0
    )
    assert off_center["geometry_passing_count"] == 3
    assert off_center["median_centered"] is False
    assert off_center["robustly_converged"] is False
    feedback = module.choose_geometry_candidate_feedback(
        off_center,
        [],
        requested_overlap_pct=50.0,
        minimum_shift_ms=0.0,
        maximum_shift_ms=100.0,
    )
    assert feedback["feedback_eligible"] is True
    assert feedback["converged"] is False
    assert feedback["next_shift_ms"] > off_center["relative_shift_ms"]

    with pytest.raises(ValueError, match="one fixed relative shift"):
        module.summarize_geometry_confirmation_trials(
            [measurement(0.0, 1.0), measurement(0.0, 2.0)], 10.0
        )


def test_geometry_feedback_repeats_baseline_unstable_candidate() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_feedback_baseline_guard",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    measurements = [
        {
            "relative_shift_ms": 3.0,
            "union_overlap_ms": 0.7,
            "realized_overlap_pct": 70.0,
            "geometry_error_pct": 20.0,
            "paired_baseline_ms": 1.0,
            "paired_baseline_drift_pct": 8.0,
            "overlap_error_ms": 0.2,
        }
        for _ in range(3)
    ]
    candidate = module.summarize_geometry_confirmation_trials(
        measurements, 10.0
    )
    feedback = module.choose_geometry_candidate_feedback(
        candidate,
        [],
        requested_overlap_pct=50.0,
        minimum_shift_ms=0.0,
        maximum_shift_ms=100.0,
    )
    assert feedback["next_shift_ms"] == 3.0
    assert feedback["method"] == "repeat_baseline_unstable_candidate"
    assert feedback["feedback_eligible"] is False


def test_geometry_confirmation_ids_and_treatment_filter_are_unambiguous() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_geometry_identity",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    observed = {
        module.geometry_confirmation_sample_index(spec, candidate, confirmation, 3)
        for spec in range(36)
        for candidate in range(3)
        for confirmation in range(3)
    }
    assert len(observed) == 36 * 3 * 3
    assert max(observed) < 0

    treatment_mode = "coverage_50_v1_victim_first"
    sample_index = module.geometry_confirmation_sample_index(0, 0, 0, 3)
    rows = [
        {
            "mode": "isolated_pre",
            "sample_index": sample_index,
            "bracket_for_mode": treatment_mode,
        },
        {
            "mode": module.geometry_calibration_mode(treatment_mode),
            "sample_index": sample_index,
            "operation_index": 0,
        },
        {
            "mode": "isolated_post",
            "sample_index": sample_index,
            "bracket_for_mode": treatment_mode,
        },
    ]
    assert module.geometry_treatment_rows(
        rows, treatment_mode, sample_index
    ) == [rows[1]]


def test_runtime_geometry_denominator_uses_paired_baseline() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_paired_geometry_denominator",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    participants = [
        {"focal_dp_rank": 0, "focal_pp_rank": 0},
        {"focal_dp_rank": 0, "focal_pp_rank": 1},
    ]
    rows = [
        {
            "operation_index": 0,
            "global_rank": 0,
            "gpu_start_global_ns": 0,
            "gpu_end_global_ns": 10_000_000,
        },
        {
            "operation_index": 1,
            "global_rank": 1,
            "gpu_start_global_ns": 5_000_000,
            "gpu_end_global_ns": 15_000_000,
        },
    ]
    spec = {
        "victim_index": 0,
        "aggressor_indices": [1],
        "requested_overlap_pct": 50.0,
    }
    assert module.measure_runtime_geometry(
        rows, participants, spec, isolated_victim_ms=10.0
    )["realized_overlap_pct"] == 50.0
    assert module.measure_runtime_geometry(
        rows, participants, spec, isolated_victim_ms=20.0
    )["realized_overlap_pct"] == 25.0


def test_counterfactual_geometry_uses_actual_starts_and_isolated_durations() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_counterfactual_geometry",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    participants = [
        {"focal_dp_rank": 0, "focal_pp_rank": 0},
        {"focal_dp_rank": 0, "focal_pp_rank": 0},
    ]
    rows = [
        {
            "operation_index": 0,
            "global_rank": 0,
            "gpu_start_global_ns": 0,
            "gpu_end_global_ns": 20_000_000,
        },
        {
            "operation_index": 1,
            "global_rank": 0,
            "gpu_start_global_ns": 8_000_000,
            "gpu_end_global_ns": 18_000_000,
        },
    ]
    spec = {
        "victim_index": 0,
        "aggressor_indices": [1],
        "requested_overlap_pct": 20.0,
    }
    geometry = module.measure_runtime_geometry(
        rows, participants, spec, {0: 10.0, 1: 2.0}
    )
    # Concurrent ends are outcomes: they imply 100% overlap but do not define
    # the scheduled treatment. Actual starts + isolated durations imply 20%.
    assert geometry["realized_overlap_pct"] == 100.0
    assert geometry["geometry_error_pct"] == 80.0
    assert geometry["scheduled_overlap_pct"] == 20.0
    assert geometry["scheduled_geometry_error_pct"] == 0.0
    assert geometry["counterfactual_overlap_pct"] == 20.0
    assert geometry["counterfactual_geometry_error_pct"] == 0.0
    assert geometry["concurrent_overlap_role"] == (
        "reported_outcome_not_validity_gate"
    )
    assert geometry["geometry_validity_basis"] == (
        "counterfactual_intervals_from_actual_global_gpu_starts_and_"
        "matched_isolated_pre_post_durations"
    )
    [aggressor] = geometry["per_aggressor_overlap"]
    assert aggressor["operation_index"] == 1
    assert aggressor["intersection_ms"] == 10.0
    assert aggressor["victim_coverage_pct"] == 100.0
    assert aggressor["scheduled_intersection_ms"] == 2.0
    assert aggressor["scheduled_victim_coverage_pct"] == 20.0
    assert aggressor["counterfactual_intersection_ms"] == 2.0
    assert aggressor["counterfactual_victim_coverage_pct"] == 20.0


def test_odd_k_baseline_collapse_uses_median_and_latest_completion() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_odd_k",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )

    def stage(repeat: int, duration: float, complete: int) -> dict:
        row = {
            "operation_index": 0,
            "participant_id": "p0",
            "global_rank": 0,
            "sample_index": 7,
            "mode": "isolated_pre",
            "bracket_for_mode": "coverage_50_v0_victim_first",
            "duration_ms": duration,
            "gate_to_start_ms": float(repeat),
            "gate_to_end_ms": float(repeat) + duration,
            "host_launch_global_ns": 100 + repeat,
            "gpu_start_global_ns": 200 + repeat,
            "gpu_end_global_ns": 300 + repeat,
            "host_complete_global_ns": complete,
        }
        return {
            "rows": [row],
            "compute_output_signatures": [],
            "communication_correctness": [],
        }

    collapsed = module.collapse_odd_k_stage_measurements([
        stage(0, 9.0, 1000), stage(1, 5.0, 3000), stage(2, 7.0, 2000)
    ])
    [row] = collapsed["rows"]
    assert row["duration_ms"] == 7.0
    assert row["measurement_representative_repeat_index"] == 2
    assert row["measurement_repetition_count"] == 3
    assert row["measurement_duration_statistics_ms"]["median"] == 7.0
    assert row["host_complete_global_ns"] == 3000
    assert row["host_complete_semantics"] == "measurement_series_latest"
    assert len(row["measurement_repeat_records_sha256"]) == 64


def test_start_anchor_confirmation_requires_median_and_two_of_three() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_anchor_confirmation",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )

    def measurement(error: float) -> dict:
        return {
            "scheduled_geometry_error_pct": error,
            "scheduled_overlap_pct": 50.0 + error,
        }

    confirmed = module.summarize_corrected_start_anchor_confirmation(
        [measurement(-9.0), measurement(2.0), measurement(20.0)],
        constraint_limited=False,
    )
    assert confirmed["passing_repetition_count"] == 2
    assert confirmed["median_measured_geometry_error_pct"] == 2.0
    assert confirmed["robustly_within_tolerance"] is True
    assert confirmed["runtime_start_geometry_disposition"] == "confirmed"

    unresolved = module.summarize_corrected_start_anchor_confirmation(
        [measurement(-20.0), measurement(0.0), measurement(20.0)],
        constraint_limited=False,
    )
    assert unresolved["passing_repetition_count"] == 1
    assert unresolved["median_measured_geometry_error_pct"] == 0.0
    assert unresolved["robustly_within_tolerance"] is False
    assert unresolved["runtime_start_geometry_disposition"] == (
        "runtime_start_geometry_unresolved"
    )


def test_runtime_start_disposition_scores_only_confirmed() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_anchor_disposition",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    for disposition, eligible in (
        ("confirmed", True),
        ("runtime_start_geometry_unresolved", False),
        ("not_applicable_static", False),
    ):
        fields = module.runtime_start_geometry_sample_fields(
            disposition, "status"
        )
        assert fields["runtime_start_geometry_disposition_explicit"] is True
        assert fields["runtime_start_geometry_treatment_executed"] is True
        assert fields["runtime_start_geometry_scoring_eligible"] is eligible
        assert fields["start_anchor_scoring_allowed"] is eligible
    with pytest.raises(ValueError, match="invalid runtime-start disposition"):
        module.runtime_start_geometry_sample_fields("legacy_implicit", None)


def test_start_anchor_executes_k3_preliminary_then_k3_confirmation() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_anchor_execution",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    participants = [
        {"focal_dp_rank": 0, "focal_pp_rank": 0},
        {"focal_dp_rank": 0, "focal_pp_rank": 1},
    ]
    spec = {
        "mode": "coverage_50_v0_victim_first",
        "victim_index": 0,
        "aggressor_indices": [1],
        "requested_overlap_pct": 50.0,
        "dispatch_mode": "victim_first",
        "order": [0, 1],
        "delays_ms": {0: 0.0, 1: 6.0},
        "target_feasible_from_duration": True,
    }
    pre_stages = [
        {"rows": [{
            "mode": "isolated_pre", "operation_index": index,
            "sample_index": 0, "duration_ms": 10.0,
            "bracket_for_mode": spec["mode"],
            "clock_calibration_generation": 1,
            "clock_calibration_label": "test",
            "clock_calibration_best_rtt_ns": 10,
            "clock_calibration_half_rtt_uncertainty_ns": 5,
        }]}
        for index in range(2)
    ]
    calls = []

    def run_stage(active, mode, sample_index, **kwargs):
        calls.append(mode)
        delays = kwargs["delays_ms"]
        rows = []
        for index in active:
            start = int(round(float(delays[index]) * 1_000_000))
            rows.append({
                "operation_index": index,
                "participant_id": f"p{index}",
                "global_rank": index,
                "mode": mode,
                "gpu_start_global_ns": start,
                "gpu_end_global_ns": start + 10_000_000,
                "host_complete_global_ns": start + 10_000_001,
                "clock_calibration_generation": 1,
                "clock_calibration_label": "test",
                "clock_calibration_best_rtt_ns": 10,
                "clock_calibration_half_rtt_uncertainty_ns": 5,
            })
        return {"rows": rows}

    decisions = []
    def consensus(payload, label):
        decisions.append((label, payload))
        return {"label": label, "all_ranks_agree": True}

    delays, metadata = module.execute_runtime_start_anchor(
        spec, 0, participants, 2, 100.0, pre_stages,
        run_stage, lambda rows: rows, consensus,
    )
    assert calls == [
        *[f"start_anchor::{spec['mode']}::repeat={i}" for i in range(3)],
        *[
            f"start_anchor_confirmation::{spec['mode']}::repeat={i}"
            for i in range(3)
        ],
    ]
    assert delays[1] == pytest.approx(5.0)
    assert [label for label, _ in decisions] == [
        "start_anchor_preliminary_correction",
        "start_anchor_post_confirmation_decision",
    ]
    assert metadata["runtime_start_geometry_disposition"] == "confirmed"
    assert metadata["runtime_start_geometry_scoring_eligible"] is True
    assert metadata["runtime_start_geometry_treatment_executed"] is True
    evidence = metadata["delay_rebase_evidence"]["start_anchor_evidence"]
    assert evidence["confirmation"]["passing_repetition_count"] == 3
    assert evidence["post_confirmation_consensus"]["all_ranks_agree"] is True


def test_clock_refresh_and_gpu_timebase_provenance_bounds() -> None:
    module = load_script(
        "benchmark_pairwise_overlap_cases_clock_bounds",
        "scripts/partial_overlap_interference_benchmarks/benchmark_pairwise_overlap_cases.py",
    )
    trigger = module.CLOCK_REFRESH_TRIGGER_AGE_NS
    maximum = module.CLOCK_REFRESH_MAX_AGE_NS
    assert module.clock_refresh_required(trigger - 1) is False
    assert module.clock_refresh_required(trigger) is True
    assert module.clock_refresh_required(0, force=True) is True
    with pytest.raises(ValueError, match="nonnegative"):
        module.clock_refresh_required(-1)

    metadata = module.gpu_timebase_metadata(
        anchor_before_local_ns=maximum - 20,
        anchor_after_local_ns=maximum,
        clock_offset_ns=100,
        calibration_local_monotonic_ns=0,
        gate_elapsed_ms=0.25,
    )
    assert metadata["clock_calibration_age_ns"] == maximum
    assert metadata["gpu_timebase_host_bracket_width_ns"] == 20
    assert metadata["gpu_timebase_host_bracket_before_global_ns"] == maximum + 80
    assert metadata["gpu_timebase_host_bracket_after_global_ns"] == maximum + 100
    assert metadata["gpu_timebase_gate_global_ns"] == maximum + 90 + 250_000
    assert metadata["gpu_timebase_mapping_method"] == (
        "synchronized_gate_host_bracket_midpoint_plus_cuda_elapsed_time"
    )
    with pytest.raises(RuntimeError, match="exceeded"):
        module.gpu_timebase_metadata(
            anchor_before_local_ns=maximum,
            anchor_after_local_ns=maximum + 1,
            clock_offset_ns=0,
            calibration_local_monotonic_ns=0,
            gate_elapsed_ms=0.0,
        )
