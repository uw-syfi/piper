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

"""Compare normal and SM-aware zero-gap launch-order slowdown tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/zero-gap-order-comparison-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SM_AWARE_DIFFERENT_FAMILIES = frozenset({"a2a", "ag"})
ORDER_COLUMNS = (
    ("victim_first", "Victim launched first"),
    ("victim_second", "Victim launched second"),
)
MODE_STYLE = {
    "normal": {"label": "Normal", "color": "#4c78a8"},
    "sm-aware": {"label": "SM-aware", "color": "#f58518"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-table", type=Path, required=True)
    parser.add_argument("--sm-aware-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sm-aware-different-family",
        action="append",
        dest="different_families",
        help=(
            "Kernel family replaced by the SM-aware mode. May be repeated. "
            "Defaults to a2a and ag."
        ),
    )
    parser.add_argument("--required-repetitions", type=int, default=20)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_table(path: Path, required_repetitions: int) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"zero-gap table is empty: {path}")
    required_columns = {
        "case_key",
        "victim_index",
        "victim_family",
        "other_family",
        "victim",
        "other_kernel",
    }
    for prefix, _ in ORDER_COLUMNS:
        required_columns.update(
            {
                f"{prefix}_repetitions",
                f"{prefix}_slowdown_median",
                f"{prefix}_slowdown_p25",
                f"{prefix}_slowdown_p75",
            }
        )
    missing = sorted(required_columns - set(rows[0]))
    if missing:
        raise ValueError(f"{path} lacks required columns: {missing}")
    identities: set[tuple[str, int]] = set()
    parsed: list[dict[str, Any]] = []
    for raw in rows:
        identity = (raw["case_key"], int(raw["victim_index"]))
        if identity in identities:
            raise ValueError(f"duplicate directed row {identity} in {path}")
        identities.add(identity)
        row: dict[str, Any] = {
            "case_key": raw["case_key"],
            "victim_index": int(raw["victim_index"]),
            "victim_family": raw["victim_family"],
            "other_family": raw["other_family"],
            "victim": raw["victim"],
            "other_kernel": raw["other_kernel"],
        }
        for prefix, _ in ORDER_COLUMNS:
            repetitions = int(raw[f"{prefix}_repetitions"])
            if repetitions != required_repetitions:
                raise ValueError(
                    f"{path}: {identity} {prefix} has n={repetitions}; "
                    f"required n={required_repetitions}"
                )
            row[prefix] = {
                "repetitions": repetitions,
                "median": float(raw[f"{prefix}_slowdown_median"]),
                "p25": float(raw[f"{prefix}_slowdown_p25"]),
                "p75": float(raw[f"{prefix}_slowdown_p75"]),
            }
        parsed.append(row)
    return parsed


def align_rows(
    normal: list[dict[str, Any]], sm_aware: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    sm_by_key = {(row["case_key"], row["victim_index"]): row for row in sm_aware}
    normal_keys = {(row["case_key"], row["victim_index"]) for row in normal}
    if normal_keys != set(sm_by_key):
        raise ValueError(
            "normal and SM-aware tables do not contain the same directed rows"
        )
    result = []
    for normal_row in normal:
        key = (normal_row["case_key"], normal_row["victim_index"])
        sm_row = sm_by_key[key]
        metadata = ("victim_family", "other_family", "victim", "other_kernel")
        differences = {
            field: (normal_row[field], sm_row[field])
            for field in metadata
            if normal_row[field] != sm_row[field]
        }
        if differences:
            raise ValueError(f"row metadata differs for {key}: {differences}")
        result.append((normal_row, sm_row))
    return result


def sm_aware_changes_row(
    row: dict[str, Any], different_families: frozenset[str]
) -> bool:
    return bool(
        {row["victim_family"], row["other_family"]} & different_families
    )


def render(
    aligned: list[tuple[dict[str, Any], dict[str, Any]]],
    different_families: frozenset[str],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        len(aligned),
        2,
        figsize=(11.8, 2.05 * len(aligned) + 1.3),
        squeeze=False,
    )
    plotted_rows: list[dict[str, Any]] = []
    for row_index, (normal_row, sm_row) in enumerate(aligned):
        show_sm_aware = sm_aware_changes_row(normal_row, different_families)
        all_cells = []
        for prefix, _ in ORDER_COLUMNS:
            all_cells.append(normal_row[prefix])
            if show_sm_aware:
                all_cells.append(sm_row[prefix])
        lower = min(0.0, min(cell["p25"] for cell in all_cells))
        upper = max(cell["p75"] for cell in all_cells)
        span = max(upper - lower, 1.0)
        x_min = lower - 0.08 * span
        x_max = upper + 0.22 * span

        for column_index, (prefix, title) in enumerate(ORDER_COLUMNS):
            axis = axes[row_index][column_index]
            series = [("normal", normal_row[prefix])]
            if show_sm_aware:
                series.append(("sm-aware", sm_row[prefix]))
            positions = list(range(len(series)))
            for position, (mode, cell) in zip(positions, series):
                median = cell["median"]
                lower_error = max(0.0, median - cell["p25"])
                upper_error = max(0.0, cell["p75"] - median)
                style = MODE_STYLE[mode]
                axis.barh(
                    position,
                    median,
                    xerr=[[lower_error], [upper_error]],
                    height=0.58,
                    color=style["color"],
                    edgecolor="white",
                    linewidth=0.8,
                    error_kw={"ecolor": "#333333", "capsize": 3, "elinewidth": 1},
                )
                text_x = median + 0.025 * span
                axis.text(
                    text_x,
                    position,
                    f"{median:.2f}%",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                )
                plotted_rows.append(
                    {
                        "case_key": normal_row["case_key"],
                        "victim_index": normal_row["victim_index"],
                        "victim": normal_row["victim"],
                        "other_kernel": normal_row["other_kernel"],
                        "launch_position": prefix,
                        "mode": mode,
                        "repetitions": cell["repetitions"],
                        "slowdown_median_pct": median,
                        "slowdown_p25_pct": cell["p25"],
                        "slowdown_p75_pct": cell["p75"],
                    }
                )
            axis.set_xlim(x_min, x_max)
            axis.set_yticks(
                positions,
                [MODE_STYLE[mode]["label"] for mode, _ in series],
                fontsize=8.5,
            )
            axis.axvline(0.0, color="#555555", linewidth=0.8)
            axis.grid(axis="x", color="#dddddd", linewidth=0.7)
            axis.set_axisbelow(True)
            axis.spines[["top", "right", "left"]].set_visible(False)
            axis.tick_params(axis="y", length=0)
            if row_index == 0:
                axis.set_title(title, fontsize=11, fontweight="bold", pad=9)
            if row_index == len(aligned) - 1:
                axis.set_xlabel("Victim slowdown (%)", fontsize=9.5)
        axes[row_index][0].text(
            -0.29,
            0.5,
            f"{normal_row['case_key']}\n{normal_row['victim']} victim\nvs {normal_row['other_kernel']}",
            transform=axes[row_index][0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        if not show_sm_aware:
            axes[row_index][1].text(
                0.99,
                0.08,
                "same kernels in both modes",
                transform=axes[row_index][1].transAxes,
                ha="right",
                va="bottom",
                color="#666666",
                fontsize=7.5,
                style="italic",
            )

    figure.suptitle(
        "Original-payload zero-gap launch-order comparison",
        fontsize=14,
        fontweight="bold",
        y=0.998,
    )
    figure.text(
        0.5,
        0.982,
        "Bars show median slowdown; whiskers show the 25th–75th percentile (n=20)",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )
    figure.subplots_adjust(left=0.27, right=0.985, top=0.955, bottom=0.045, hspace=0.78, wspace=0.22)

    png = output_dir / "zero_gap_launch_order_backend_comparison.png"
    svg = output_dir / "zero_gap_launch_order_backend_comparison.svg"
    source = output_dir / "zero_gap_launch_order_backend_comparison_source.csv"
    figure.savefig(png, dpi=190, facecolor="white")
    figure.savefig(svg, metadata={"Date": None}, facecolor="white")
    plt.close(figure)
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plotted_rows[0]))
        writer.writeheader()
        writer.writerows(plotted_rows)
    return png, svg, source


def main() -> int:
    args = parse_args()
    if args.required_repetitions <= 0:
        raise ValueError("--required-repetitions must be positive")
    different_families = frozenset(
        args.different_families or DEFAULT_SM_AWARE_DIFFERENT_FAMILIES
    )
    normal = read_table(args.normal_table, args.required_repetitions)
    sm_aware = read_table(args.sm_aware_table, args.required_repetitions)
    aligned = align_rows(normal, sm_aware)
    png, svg, source = render(aligned, different_families, args.output_dir)
    sm_aware_rows = sum(
        sm_aware_changes_row(row, different_families) for row, _ in aligned
    )
    receipt = {
        "schema_version": 1,
        "normal_table": str(args.normal_table.resolve()),
        "normal_table_sha256": sha256(args.normal_table),
        "sm_aware_table": str(args.sm_aware_table.resolve()),
        "sm_aware_table_sha256": sha256(args.sm_aware_table),
        "required_repetitions_per_bar": args.required_repetitions,
        "directed_rows": len(aligned),
        "launch_order_columns": [prefix for prefix, _ in ORDER_COLUMNS],
        "sm_aware_different_families": sorted(different_families),
        "rows_with_two_mode_bars": sm_aware_rows,
        "rows_with_normal_only_bar": len(aligned) - sm_aware_rows,
        "normal_only_rule": (
            "omit the SM-aware bar when neither kernel family changes backend"
        ),
        "png": str(png.resolve()),
        "svg": str(svg.resolve()),
        "source_csv": str(source.resolve()),
    }
    receipt_path = args.output_dir / "zero_gap_launch_order_backend_comparison_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
