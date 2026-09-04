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

"""Render the victim-row/launch-order figure for one collective mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
FIGURE_MODULE_ROOT = REPO_ROOT / "benchmark_data/partial_overlap_interference"
sys.path.insert(0, str(FIGURE_MODULE_ROOT))

import pandas as pd

from generate_factor_figures import (
    COMM_COMM_CASE_KEYS,
    COMM_COMPUTE_CASE_KEYS,
    FIGURE1_DISPATCH_COLUMNS,
    FIGURE1_RENDERED_KERNEL_KEYS,
    FIGURE1_RENDERED_VICTIM_ROWS,
    aggregate_directed_cells,
    figure1_rendered_table,
    figure1_source,
    included,
    plot_figure1,
    plot_figure1_by_victim,
)


REQUESTED_REALIZED_AGREEMENT_TOLERANCE_PP = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--mode-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=12_288)
    return parser.parse_args()


def build_figure_source(
    samples: pd.DataFrame,
    sequence_length: int,
    victim_kernel_keys: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    required = {
        "seq_len", "analysis_included", "dispatch_mode", "sample_id",
        "case_id", "case_key", "victim_index", "victim_family",
        "aggressor_family", "requested_overlap_pct", "realized_overlap_pct",
        "victim_slowdown_pct",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"converted sample table lacks columns: {missing}")
    data = samples[
        (samples["seq_len"] == sequence_length)
        & included(samples["analysis_included"])
    ].copy()
    if data.empty:
        raise ValueError("no estimator-valid samples are available for the figure")
    if data["sample_id"].duplicated().any():
        raise ValueError("sample_id is not unique")
    observed_dispatch = set(data["dispatch_mode"].astype(str))
    expected_dispatch = {mode for mode, _ in FIGURE1_DISPATCH_COLUMNS}
    if observed_dispatch - expected_dispatch:
        raise ValueError(
            f"unexpected dispatch modes: {sorted(observed_dispatch - expected_dispatch)}"
        )
    cells = aggregate_directed_cells(data)
    compute = figure1_source(cells, COMM_COMPUTE_CASE_KEYS)
    communication = figure1_source(cells, COMM_COMM_CASE_KEYS)
    result = pd.concat([compute, communication], ignore_index=True)
    result = result.sort_values(
        [
            "victim_kernel_key", "dispatch_mode", "case_id", "case_key",
            "victim_index", "requested_overlap_pct",
        ]
    ).reset_index(drop=True)
    expected_victims = (
        victim_kernel_keys
        if victim_kernel_keys is not None
        else FIGURE1_RENDERED_KERNEL_KEYS
    )
    unexpected_victims = set(expected_victims) - set(FIGURE1_RENDERED_KERNEL_KEYS)
    if unexpected_victims:
        raise ValueError(
            f"unknown configured Figure 1 victims: {sorted(unexpected_victims)}"
        )
    result = result[result["victim_kernel_key"].isin(expected_victims)].copy()
    for victim in expected_victims:
        observed = set(
            result[result["victim_kernel_key"] == victim]["dispatch_mode"].astype(str)
        )
        if observed != expected_dispatch:
            raise ValueError(
                f"victim {victim!r} does not contain both launch orders: "
                f"observed={sorted(observed)}"
            )
    return result


def audit_data_point_repetitions(
    samples: pd.DataFrame,
    config: dict[str, Any],
    sequence_length: int,
) -> pd.DataFrame:
    """Require exactly the configured valid repetitions for every figure cell."""

    required = {
        "seq_len", "case_key", "victim_index", "victim_family",
        "aggressor_family", "dispatch_mode", "requested_overlap_pct",
        "analysis_included",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise ValueError(f"cannot audit data-point repetitions; missing {missing}")
    expected_repetitions = int(
        config["required_valid_repetitions_per_data_point"]
    )
    if expected_repetitions != 20:
        raise ValueError(
            "the original Figure 1 comparison requires exactly 20 valid "
            f"repetitions per data point, got {expected_repetitions}"
        )
    case_keys = [str(value) for value in config["case_keys"]]
    dispatch_modes = [str(value) for value in config["dispatch_modes"]]
    overlap_targets = [float(value) for value in config["coverage_targets_pct"]]
    data = samples[
        (samples["seq_len"] == sequence_length)
        & samples["case_key"].astype(str).isin(case_keys)
    ].copy()
    records: list[dict[str, Any]] = []
    for case_key in case_keys:
        case_rows = data[data["case_key"].astype(str) == case_key]
        for victim_index in (0, 1):
            victim_rows = case_rows[case_rows["victim_index"] == victim_index]
            victim_family = (
                str(victim_rows["victim_family"].iloc[0])
                if not victim_rows.empty else ""
            )
            aggressor_family = (
                str(victim_rows["aggressor_family"].iloc[0])
                if not victim_rows.empty else ""
            )
            for dispatch_mode in dispatch_modes:
                for requested_overlap_pct in overlap_targets:
                    cell = victim_rows[
                        (victim_rows["dispatch_mode"].astype(str) == dispatch_mode)
                        & (
                            victim_rows["requested_overlap_pct"].astype(float)
                            == requested_overlap_pct
                        )
                    ]
                    valid_mask = included(cell["analysis_included"])
                    excluded = cell[~valid_mask]
                    if cell.empty:
                        reasons = {"missing_measurements": expected_repetitions}
                    elif "analysis_disposition" in excluded:
                        reasons = {
                            str(reason): int(count)
                            for reason, count in excluded[
                                "analysis_disposition"
                            ].fillna("missing_analysis_disposition").value_counts().items()
                        }
                    else:
                        reasons = {"analysis_not_included": len(excluded)}
                    issued = int(len(cell))
                    valid = int(valid_mask.sum())
                    records.append({
                        "case_key": case_key,
                        "victim_index": victim_index,
                        "victim_family": victim_family,
                        "aggressor_family": aggressor_family,
                        "dispatch_mode": dispatch_mode,
                        "requested_overlap_pct": requested_overlap_pct,
                        "expected_repetitions": expected_repetitions,
                        "issued_repetitions": issued,
                        "valid_repetitions": valid,
                        "status": (
                            "obtained"
                            if issued == expected_repetitions
                            and valid == expected_repetitions
                            else "unobtainable"
                        ),
                        "failure_reasons_json": json.dumps(
                            reasons, sort_keys=True, separators=(",", ":")
                        ),
                    })
    return pd.DataFrame.from_records(records)


def filter_requested_realized_agreement(
    table: pd.DataFrame,
    tolerance_pp: float = REQUESTED_REALIZED_AGREEMENT_TOLERANCE_PP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split aggregated cells by requested/median-realized overlap agreement."""

    if tolerance_pp < 0:
        raise ValueError("requested/realized agreement tolerance must be nonnegative")
    required = {"requested_overlap_pct", "realized_overlap_median_pct"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(
            "cannot apply requested/realized overlap agreement filter; "
            f"missing {missing}"
        )
    annotated = table.copy()
    annotated["requested_realized_error_pp"] = (
        annotated["realized_overlap_median_pct"].astype(float)
        - annotated["requested_overlap_pct"].astype(float)
    )
    annotated["requested_realized_abs_error_pp"] = annotated[
        "requested_realized_error_pp"
    ].abs()
    annotated["requested_realized_agreement_tolerance_pp"] = float(tolerance_pp)
    agrees = (
        annotated["requested_realized_abs_error_pp"]
        <= float(tolerance_pp) + 1e-9
    )
    annotated["requested_realized_agreement_status"] = agrees.map(
        {True: "included_agrees", False: "excluded_disagrees"}
    )
    return (
        annotated[agrees].reset_index(drop=True),
        annotated[~agrees].reset_index(drop=True),
    )


def main() -> None:
    args = parse_args()
    config = json.loads(args.mode_config.read_text(encoding="utf-8"))
    selection = config.get("collective_backend_selection")
    if not isinstance(selection, dict) or selection.get("mode") not in {
        "nccl", "sm-aware"
    }:
        raise ValueError("mode config lacks a validated collective backend selection")
    samples = pd.read_csv(args.samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_data_point_repetitions(samples, config, args.sequence_length)
    audit_path = args.output_dir / "data_point_repetition_audit.csv"
    failure_path = args.output_dir / "unobtainable_data_points.csv"
    failure_json_path = args.output_dir / "unobtainable_data_points.json"
    receipt_path = args.output_dir / "data_point_repetition_audit.json"
    failures = audit[audit["status"] != "obtained"].copy()
    audit.to_csv(audit_path, index=False)
    failures.to_csv(failure_path, index=False)
    failure_json_path.write_text(
        json.dumps(
            json.loads(failures.to_json(orient="records")), indent=2
        ) + "\n",
        encoding="utf-8",
    )
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "expected_data_points": int(len(audit)),
                "obtained_data_points": int(len(audit) - len(failures)),
                "unobtainable_data_points": int(len(failures)),
                "required_valid_repetitions_per_data_point": int(
                    config["required_valid_repetitions_per_data_point"]
                ),
                "audit_csv": str(audit_path.resolve()),
                "unobtainable_csv": str(failure_path.resolve()),
                "unobtainable_json": str(failure_json_path.resolve()),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    configured_victims = tuple(
        str(value) for value in config.get(
            "rendered_victim_kernel_keys", FIGURE1_RENDERED_KERNEL_KEYS
        )
    )
    cell_keys = [
        "case_key", "victim_index", "dispatch_mode", "requested_overlap_pct"
    ]
    obtained_cells = audit[audit["status"] == "obtained"][cell_keys]
    figure_samples = samples.merge(
        obtained_cells,
        on=cell_keys,
        how="inner",
        validate="many_to_one",
    )
    unfiltered_table = build_figure_source(
        figure_samples, args.sequence_length, configured_victims
    )
    table, disagreement_table = filter_requested_realized_agreement(
        unfiltered_table
    )
    if table.empty:
        raise ValueError(
            "no aggregated points satisfy requested/realized overlap agreement"
        )
    source = args.output_dir / "figure1_overlap_response_by_victim_source.csv"
    png = args.output_dir / "figure1_overlap_response_by_victim.png"
    svg = args.output_dir / "figure1_overlap_response_by_victim.svg"
    disagreement_csv = (
        args.output_dir / "requested_realized_overlap_disagreements.csv"
    )
    disagreement_json = (
        args.output_dir / "requested_realized_overlap_disagreements.json"
    )
    table.to_csv(source, index=False)
    disagreement_table.to_csv(disagreement_csv, index=False)
    disagreement_json.write_text(
        json.dumps(
            json.loads(disagreement_table.to_json(orient="records")), indent=2
        ) + "\n",
        encoding="utf-8",
    )
    victim_rows = tuple(
        row for row in FIGURE1_RENDERED_VICTIM_ROWS
        if row[0] in configured_victims
    )
    plot_figure1_by_victim(
        figure1_rendered_table(table), victim_rows, png, svg
    )
    receipt = {
        "schema_version": 1,
        "collective_backend_selection": selection,
        "sequence_length": args.sequence_length,
        "victim_rows": list(configured_victims),
        "subplot_columns": [mode for mode, _ in FIGURE1_DISPATCH_COLUMNS],
        "exact20_aggregated_points_before_agreement_filter": len(
            unfiltered_table
        ),
        "aggregated_points": len(table),
        "figure_point_filter": (
            "abs(median realized overlap - requested overlap) <= "
            f"{REQUESTED_REALIZED_AGREEMENT_TOLERANCE_PP:g} percentage points"
        ),
        "requested_realized_agreement_tolerance_pp": (
            REQUESTED_REALIZED_AGREEMENT_TOLERANCE_PP
        ),
        "requested_realized_disagreement_points": len(disagreement_table),
        "requested_realized_disagreements_csv": str(
            disagreement_csv.resolve()
        ),
        "requested_realized_disagreements_json": str(
            disagreement_json.resolve()
        ),
        "audit_status": (
            "complete" if failures.empty else "unobtainable_data_points"
        ),
        "unobtainable_data_points": int(len(failures)),
        "unobtainable_data_points_csv": str(failure_path.resolve()),
        "source": str(source.resolve()),
        "png": str(png.resolve()),
        "svg": str(svg.resolve()),
    }
    (args.output_dir / "figure_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(png)
    if not failures.empty:
        raise RuntimeError(
            f"{len(failures)} intended-overlap data points could not provide "
            "exactly 20 estimator-valid repetitions; the available-data "
            f"figure was rendered, and failures are listed in {failure_path.resolve()}"
        )


if __name__ == "__main__":
    main()
