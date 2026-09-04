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

"""Render the four-way normal/Zero-CTA A2A-versus-AG launch-order matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONFIGS = (
    ("both_normal", "Normal AG\nNormal A2A", "#4C78A8"),
    ("both_zero_cta", "Zero-CTA AG\nZero-CTA A2A", "#F58518"),
    ("ag_normal_a2a_zero_cta", "Normal AG\nZero-CTA A2A", "#54A24B"),
    ("ag_zero_cta_a2a_normal", "Zero-CTA AG\nNormal A2A", "#E45756"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    for key, label, color in CONFIGS:
        path = args.input_root / key / "table" / "zero_gap_order_table.csv"
        frame = pd.read_csv(path)
        frame = frame[(frame["case_key"] == "A14") & frame["victim_family"].isin(("a2a", "ag"))]
        if len(frame) != 2:
            raise RuntimeError(f"{path}: expected A14 A2A and AG rows, got {len(frame)}")
        frame["backend_configuration"] = key
        frame["backend_label"] = label
        frame["color"] = color
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "a2a_ag_backend_matrix_table.csv", index=False)

    paired_rows = []
    for key, label, _ in CONFIGS:
        configured = table[table["backend_configuration"] == key].set_index(
            "victim_family"
        )
        for launch_order, a2a_position, ag_position in (
            ("a2a_first_ag_second", "first", "second"),
            ("ag_first_a2a_second", "second", "first"),
        ):
            a2a = configured.loc["a2a"]
            ag = configured.loc["ag"]
            a2a_makespan = {
                quantile: float(
                    a2a[f"victim_{a2a_position}_overall_makespan_us_{quantile}"]
                )
                for quantile in ("median", "p25", "p75")
            }
            ag_makespan = {
                quantile: float(
                    ag[f"victim_{ag_position}_overall_makespan_us_{quantile}"]
                )
                for quantile in ("median", "p25", "p75")
            }
            if a2a_makespan != ag_makespan:
                raise RuntimeError(
                    f"{key} {launch_order} participant summaries do not share "
                    "one aggregate runtime distribution"
                )
            paired_rows.append({
                "backend_configuration": key,
                "backend_label": label.replace("\n", " / "),
                "launch_order": launch_order,
                "repetitions": int(
                    a2a[f"victim_{a2a_position}_repetitions"]
                ),
                "a2a_slowdown_median_pct": float(
                    a2a[f"victim_{a2a_position}_slowdown_median"]
                ),
                "a2a_slowdown_p25_pct": float(
                    a2a[f"victim_{a2a_position}_slowdown_p25"]
                ),
                "a2a_slowdown_p75_pct": float(
                    a2a[f"victim_{a2a_position}_slowdown_p75"]
                ),
                "ag_slowdown_median_pct": float(
                    ag[f"victim_{ag_position}_slowdown_median"]
                ),
                "ag_slowdown_p25_pct": float(
                    ag[f"victim_{ag_position}_slowdown_p25"]
                ),
                "ag_slowdown_p75_pct": float(
                    ag[f"victim_{ag_position}_slowdown_p75"]
                ),
                "overall_completion_median_ms": a2a_makespan["median"] / 1000.0,
                "overall_completion_p25_ms": a2a_makespan["p25"] / 1000.0,
                "overall_completion_p75_ms": a2a_makespan["p75"] / 1000.0,
            })
    paired_table = pd.DataFrame.from_records(paired_rows)
    paired_table.to_csv(
        args.output_dir / "a2a_ag_backend_matrix_paired_execution_table.csv",
        index=False,
    )

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 11.4), constrained_layout=True)
    positions = np.arange(len(CONFIGS))
    for row_index, (victim_family, victim_label, other_label) in enumerate((
        ("a2a", "A2A victim", "AG"),
        ("ag", "AG victim", "A2A"),
    )):
        victim_rows = table[table["victim_family"] == victim_family].set_index(
            "backend_configuration"
        )
        for column_index, launch_order in enumerate(("a2a_first", "ag_first")):
            launch_position = (
                "first"
                if launch_order.startswith(victim_family)
                else "second"
            )
            axis = axes[row_index, column_index]
            rows = victim_rows.loc[[key for key, _, _ in CONFIGS]]
            medians = rows[f"victim_{launch_position}_slowdown_median"].to_numpy()
            p25 = rows[f"victim_{launch_position}_slowdown_p25"].to_numpy()
            p75 = rows[f"victim_{launch_position}_slowdown_p75"].to_numpy()
            errors = np.vstack((medians - p25, p75 - medians))
            bars = axis.bar(
                positions,
                medians,
                color=[color for _, _, color in CONFIGS],
                yerr=errors,
                capsize=4,
                width=0.72,
                edgecolor="black",
                linewidth=0.5,
            )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(positions, [label for _, label, _ in CONFIGS], fontsize=8)
            axis.set_ylabel("Victim slowdown (%)")
            axis.set_title(
                f"{victim_label} vs {other_label}: victim launched {launch_position}"
            )
            axis.grid(axis="y", alpha=0.25)
            span = max(1.0, float(max(p75.max(), 0.0) - min(p25.min(), 0.0)))
            for index, (bar, value) in enumerate(zip(bars, medians)):
                label_y = (
                    p75[index] + 0.025 * span
                    if value >= 0
                    else p25[index] - 0.025 * span
                )
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    label_y,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom" if value >= 0 else "top",
                    fontsize=8,
                    fontweight="bold",
                )
    a2a_rows = table[table["victim_family"] == "a2a"].set_index(
        "backend_configuration"
    ).loc[[key for key, _, _ in CONFIGS]]
    ag_rows = table[table["victim_family"] == "ag"].set_index(
        "backend_configuration"
    ).loc[[key for key, _, _ in CONFIGS]]
    for column_index, (launch_order, source_rows, position) in enumerate((
        ("A2A launched first, AG second", a2a_rows, "first"),
        ("AG launched first, A2A second", ag_rows, "first"),
    )):
        axis = axes[2, column_index]
        medians = source_rows[
            f"victim_{position}_overall_makespan_us_median"
        ].to_numpy() / 1000.0
        p25 = source_rows[
            f"victim_{position}_overall_makespan_us_p25"
        ].to_numpy() / 1000.0
        p75 = source_rows[
            f"victim_{position}_overall_makespan_us_p75"
        ].to_numpy() / 1000.0
        counterpart = ag_rows if column_index == 0 else a2a_rows
        counterpart_position = "second"
        for quantile, observed in (("median", medians), ("p25", p25), ("p75", p75)):
            expected = counterpart[
                f"victim_{counterpart_position}_overall_makespan_us_{quantile}"
            ].to_numpy() / 1000.0
            if not np.allclose(observed, expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"{launch_order} participant rows do not share {quantile} makespan"
                )
        errors = np.vstack((medians - p25, p75 - medians))
        bars = axis.bar(
            positions,
            medians,
            color=[color for _, _, color in CONFIGS],
            yerr=errors,
            capsize=4,
            width=0.72,
            edgecolor="black",
            linewidth=0.5,
        )
        axis.set_xticks(positions, [label for _, label, _ in CONFIGS], fontsize=8)
        axis.set_ylabel("Overall completion time (ms)")
        axis.set_title(f"Shared paired execution: {launch_order}")
        axis.grid(axis="y", alpha=0.25)
        span = max(1.0, float(p75.max() - p25.min()))
        for index, (bar, value) in enumerate(zip(bars, medians)):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                p75[index] + 0.025 * span,
                f"{value:.2f} ms",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
    fig.suptitle(
        "A2A–AG zero-gap interference: default-CTA versus Zero-CTA backends\n"
        "Slowdown and paired overall completion time; bars are medians, "
        "error bars are IQR (n=20)",
        fontsize=14,
    )
    fig.savefig(args.output_dir / "a2a_ag_backend_matrix_bars.png", dpi=220)
    fig.savefig(args.output_dir / "a2a_ag_backend_matrix_bars.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
