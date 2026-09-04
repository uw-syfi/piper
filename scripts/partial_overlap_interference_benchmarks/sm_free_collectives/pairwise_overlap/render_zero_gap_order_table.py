#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Render one zero-intentional-gap launch-order slowdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


LABELS = {
    "compute": "grouped GEMM",
    "a2a": "A2A",
    "ag": "AG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mode-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _zero_mapping(values: Any) -> bool:
    return isinstance(values, dict) and values and all(
        float(value) == 0.0 for value in values.values()
    )


def extract_records(
    document: dict[str, Any], expected_repetitions: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    audit = {
        "cases": 0,
        "treatment_executions": 0,
        "participant_measurements": 0,
        "all_correctness_passed": True,
        "all_stage_spec_delays_zero": True,
        "all_treatment_delays_zero": True,
        "all_orders_complete": True,
    }
    for result in document["results"]:
        audit["cases"] += 1
        assignment = result["assignment"]
        participants = assignment["participants"]
        correctness = result["correctness"]
        if not correctness or any(row.get("passed") is not True for row in correctness):
            audit["all_correctness_passed"] = False
            raise RuntimeError(
                f"{assignment['case_key']} contains failed or missing correctness"
            )
        specs = {str(spec["mode"]): spec for spec in result["stage_specs"]}
        orders = {tuple(int(index) for index in spec["order"]) for spec in specs.values()}
        if orders != {(0, 1), (1, 0)} or any(
            spec.get("experiment_mode")
            != "zero_intentional_gap_launch_order_only"
            for spec in specs.values()
        ):
            audit["all_orders_complete"] = False
            raise RuntimeError(
                f"{assignment['case_key']} does not contain exactly the two zero-gap orders"
            )
        if any(not _zero_mapping(spec.get("delays_ms")) for spec in specs.values()):
            audit["all_stage_spec_delays_zero"] = False
            raise RuntimeError(f"{assignment['case_key']} stage spec has a nonzero delay")

        samples_by_mode: dict[str, list[dict[str, Any]]] = {
            mode: [] for mode in specs
        }
        treatment_rows_by_execution: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in result.get("participants", []):
            row_mode = str(row.get("mode"))
            if row_mode not in specs:
                continue
            execution = (row_mode, int(row["sample_index"]))
            treatment_rows_by_execution.setdefault(execution, []).append(row)
        for sample in result["samples"]:
            mode = str(sample["stage_mode"])
            if mode not in specs:
                raise RuntimeError(f"unexpected treatment mode {mode!r}")
            samples_by_mode[mode].append(sample)
        for mode, samples in samples_by_mode.items():
            if len(samples) < expected_repetitions:
                raise RuntimeError(
                    f"{assignment['case_key']} {mode} has {len(samples)} samples; "
                    f"expected at least {expected_repetitions}"
                )
            order = [int(index) for index in specs[mode]["order"]]
            for sample in samples:
                execution = (mode, int(sample["sample_index"]))
                treatment_rows = treatment_rows_by_execution.get(execution, [])
                if not treatment_rows:
                    raise RuntimeError(
                        f"{assignment['case_key']} {mode} sample "
                        f"{sample['sample_index']} lacks endpoint timing rows"
                    )
                overall_start_global_ns = min(
                    int(row["gpu_start_global_ns"]) for row in treatment_rows
                )
                overall_end_global_ns = max(
                    int(row["gpu_end_global_ns"]) for row in treatment_rows
                )
                overall_makespan_us = (
                    overall_end_global_ns - overall_start_global_ns
                ) / 1e3
                evidence = sample.get("delay_rebase_evidence")
                if (
                    not isinstance(evidence, dict)
                    or evidence.get("policy")
                    != "zero_configured_delay_back_to_back_order_v1"
                    or not _zero_mapping(evidence.get("configured_delays_ms"))
                ):
                    audit["all_treatment_delays_zero"] = False
                    raise RuntimeError(
                        f"{assignment['case_key']} {mode} sample "
                        f"{sample['sample_index']} lacks zero-delay evidence"
                    )
                outcomes = {
                    int(row["operation_index"]): row
                    for row in sample["participant_outcomes"]
                }
                if set(outcomes) != {0, 1}:
                    raise RuntimeError("participant outcomes are incomplete")
                first_start = float(outcomes[order[0]]["start_ms"])
                second_start = float(outcomes[order[1]]["start_ms"])
                later_start_gap_us = (second_start - first_start) * 1000.0
                left = max(
                    float(outcomes[0]["start_ms"]),
                    float(outcomes[1]["start_ms"]),
                )
                right = min(
                    float(outcomes[0]["end_ms"]),
                    float(outcomes[1]["end_ms"]),
                )
                intersection_ms = max(0.0, right - left)
                drifts = sample["geometry_baseline_drifts_pct"]
                execution_baseline_max_abs_drift_pct = max(
                    abs(float(value)) for value in drifts.values()
                )
                for index, outcome in outcomes.items():
                    other = 1 - index
                    records.append({
                        "case_key": assignment["case_key"],
                        "victim_index": index,
                        "victim_family": participants[index]["category"],
                        "other_family": participants[other]["category"],
                        "launch_position": (
                            "first" if order.index(index) == 0 else "second"
                        ),
                        "stage_mode": mode,
                        "sample_index": int(sample["sample_index"]),
                        "slowdown_pct": float(outcome["slowdown_pct"]),
                        "isolated_ms": float(outcome["isolated_ms"]),
                        "concurrent_ms": float(outcome["concurrent_ms"]),
                        "observed_overlap_pct": min(
                            100.0,
                            intersection_ms / float(outcome["isolated_ms"]) * 100.0,
                        ),
                        "later_kernel_actual_start_gap_us": later_start_gap_us,
                        "baseline_drift_pct": float(drifts[str(index)]),
                        "execution_baseline_max_abs_drift_pct": (
                            execution_baseline_max_abs_drift_pct
                        ),
                        "overall_start_global_ns": overall_start_global_ns,
                        "overall_end_global_ns": overall_end_global_ns,
                        "overall_makespan_us": overall_makespan_us,
                    })
                audit["treatment_executions"] += 1
                audit["participant_measurements"] += 2
    return pd.DataFrame.from_records(records), audit


def select_estimator_records(
    records: pd.DataFrame, expected_repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select exact-N stable executions, retaining both participant outcomes."""

    annotated = records.copy()
    annotated["execution_baseline_stable"] = (
        annotated["execution_baseline_max_abs_drift_pct"] <= 5.0
    )
    annotated["analysis_included"] = False
    annotated["analysis_disposition"] = "excluded_surplus_stable"
    annotated.loc[
        ~annotated["execution_baseline_stable"], "analysis_disposition"
    ] = "excluded_baseline_drift"
    keys = ["case_key", "stage_mode"]
    for key, group in annotated.groupby(keys, sort=True):
        execution_sizes = group.groupby("sample_index").size()
        if not execution_sizes.eq(2).all():
            raise RuntimeError(
                f"{key} does not contain exactly two participant outcomes "
                "for every execution"
            )
        makespan_counts = group.groupby("sample_index")[
            "overall_makespan_us"
        ].nunique()
        if not makespan_counts.eq(1).all():
            raise RuntimeError(f"{key} participant rows do not share makespans")
        stable_by_sample = group.groupby("sample_index")[
            "execution_baseline_stable"
        ].all()
        eligible_samples = sorted(
            int(sample_index)
            for sample_index, stable in stable_by_sample.items()
            if bool(stable)
        )
        if len(eligible_samples) < expected_repetitions:
            raise RuntimeError(
                f"{key} has {len(eligible_samples)} jointly baseline-stable "
                "two-collective executions; "
                f"expected {expected_repetitions}"
            )
        selected_samples = set(eligible_samples[:expected_repetitions])
        selected_indices = group.index[group["sample_index"].isin(selected_samples)]
        annotated.loc[selected_indices, "analysis_included"] = True
        annotated.loc[
            selected_indices, "analysis_disposition"
        ] = "included_exact20_jointly_baseline_stable_execution"
    selected = annotated[annotated["analysis_included"]].copy()
    excluded = annotated[~annotated["analysis_included"]].copy()
    return selected.reset_index(drop=True), excluded.reset_index(drop=True)


def aggregate_table(records: pd.DataFrame, expected_repetitions: int) -> pd.DataFrame:
    keys = ["case_key", "victim_index", "victim_family", "other_family"]
    rows = []
    for key, group in records.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["victim"] = LABELS.get(str(row["victim_family"]), row["victim_family"])
        row["other_kernel"] = LABELS.get(
            str(row["other_family"]), row["other_family"]
        )
        for position in ("first", "second"):
            values = group[group["launch_position"] == position]
            if len(values) != expected_repetitions:
                raise RuntimeError(
                    f"{key} {position} has {len(values)} measurements; "
                    f"expected {expected_repetitions}"
                )
            row[f"victim_{position}_repetitions"] = len(values)
            for source, output in (
                ("slowdown_pct", "slowdown"),
                ("overall_makespan_us", "overall_makespan_us"),
                ("observed_overlap_pct", "observed_overlap"),
                ("later_kernel_actual_start_gap_us", "actual_start_gap_us"),
                ("baseline_drift_pct", "baseline_drift"),
            ):
                row[f"victim_{position}_{output}_median"] = values[source].median()
                row[f"victim_{position}_{output}_p25"] = values[source].quantile(0.25)
                row[f"victim_{position}_{output}_p75"] = values[source].quantile(0.75)
            row[f"victim_{position}_baseline_drift_max_abs_pct"] = (
                values["baseline_drift_pct"].abs().max()
            )
        row["second_minus_first_slowdown_median_pct_points"] = (
            row["victim_second_slowdown_median"]
            - row["victim_first_slowdown_median"]
        )
        rows.append(row)
    return pd.DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def markdown_table(table: pd.DataFrame, mode: str) -> str:
    lines = [
        f"# {mode} zero-intentional-gap launch-order results",
        "",
        "Entries report median victim slowdown and the overall two-collective "
        "completion time, each with the 25th--75th percentile interval. Overall "
        "time is latest endpoint completion minus earliest endpoint start for "
        "one paired execution. Both operations were submitted back-to-back with "
        "zero configured delay.",
        "",
        "| Case | Victim | Other kernel | Victim launched first: slowdown | Victim launched first: overall time | Victim launched second: slowdown | Victim launched second: overall time | Second - first slowdown |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in table.to_dict(orient="records"):
        first = (
            f"{row['victim_first_slowdown_median']:.2f}% "
            f"[{row['victim_first_slowdown_p25']:.2f}, "
            f"{row['victim_first_slowdown_p75']:.2f}] (n={row['victim_first_repetitions']})"
        )
        second = (
            f"{row['victim_second_slowdown_median']:.2f}% "
            f"[{row['victim_second_slowdown_p25']:.2f}, "
            f"{row['victim_second_slowdown_p75']:.2f}] (n={row['victim_second_repetitions']})"
        )
        first_makespan = (
            f"{row['victim_first_overall_makespan_us_median'] / 1000.0:.3f} ms "
            f"[{row['victim_first_overall_makespan_us_p25'] / 1000.0:.3f}, "
            f"{row['victim_first_overall_makespan_us_p75'] / 1000.0:.3f}]"
        )
        second_makespan = (
            f"{row['victim_second_overall_makespan_us_median'] / 1000.0:.3f} ms "
            f"[{row['victim_second_overall_makespan_us_p25'] / 1000.0:.3f}, "
            f"{row['victim_second_overall_makespan_us_p75'] / 1000.0:.3f}]"
        )
        lines.append(
            f"| {row['case_key']} | {row['victim']} | {row['other_kernel']} "
            f"| {first} | {first_makespan} | {second} | {second_makespan} | "
            f"{row['second_minus_first_slowdown_median_pct_points']:+.2f} pp |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    config = json.loads(args.mode_config.read_text(encoding="utf-8"))
    if config.get("experiment") != "zero_intentional_gap_launch_order_only":
        raise ValueError("mode config is not a zero-gap order-only experiment")
    if config.get("collective_payload_overrides_input_numel_per_rank"):
        raise ValueError("zero-gap order table requires original payload sizes")
    document = json.loads(args.raw.read_text(encoding="utf-8"))
    scheduling = document["configuration"]["scheduling"]
    if scheduling.get("zero_gap_order_only") is not True:
        raise ValueError("raw result does not declare zero-gap order-only scheduling")
    expected = int(config["required_repetitions_per_launch_order"])
    records, audit = extract_records(document, expected)
    selected, excluded = select_estimator_records(records, expected)
    table = aggregate_table(selected, expected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "zero_gap_order_measurements.csv"
    csv_path = args.output_dir / "zero_gap_order_table.csv"
    markdown_path = args.output_dir / "zero_gap_order_table.md"
    receipt_path = args.output_dir / "zero_gap_order_table_receipt.json"
    combined_records = pd.concat([selected, excluded], ignore_index=True)
    combined_records = combined_records.sort_values(
        ["case_key", "victim_index", "launch_position", "sample_index"]
    )
    combined_records.to_csv(records_path, index=False)
    table.to_csv(csv_path, index=False)
    mode = str(config["collective_backend_selection"]["mode"])
    markdown_path.write_text(markdown_table(table, mode), encoding="utf-8")
    receipt = {
        "schema_version": 2,
        "mode": mode,
        "experiment": config["experiment"],
        "payloads": "original",
        "configured_launch_delay_ms": 0.0,
        "required_repetitions_per_launch_order": expected,
        "directed_table_rows": len(table),
        "selected_participant_measurements": len(selected),
        "selected_paired_treatment_executions": int(len(selected) / 2),
        "excluded_participant_measurements": len(excluded),
        "excluded_baseline_drift_measurements": int(
            (excluded["analysis_disposition"] == "excluded_baseline_drift").sum()
        ),
        "selection_unit": "whole_two_collective_execution",
        "overall_makespan_definition": (
            "latest_treatment_endpoint_gpu_end_global_ns_minus_"
            "earliest_treatment_endpoint_gpu_start_global_ns"
        ),
        "excluded_surplus_stable_measurements": int(
            (excluded["analysis_disposition"] == "excluded_surplus_stable").sum()
        ),
        **audit,
        "measurements_csv": str(records_path.resolve()),
        "table_csv": str(csv_path.resolve()),
        "table_markdown": str(markdown_path.resolve()),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(markdown_path)


if __name__ == "__main__":
    main()
