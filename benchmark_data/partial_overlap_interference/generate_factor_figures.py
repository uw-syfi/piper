#!/usr/bin/env python3
"""Generate timing-only Figures 1--4 from the frozen final corpus.

This script deliberately does not read raw benchmark checkpoints or launch GPU
work.  Its only benchmark input is merged-results/tables/samples.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/piper-factor-figures-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams["svg.hashsalt"] = "piper-factor-separated-figures-v1"


SEQ_LEN = 12288
COMM_FAMILIES = {"a2a", "ag", "rs", "p2p_send", "p2p_recv"}
COMM_COMPUTE_CASE_KEYS = (
    "A01:C-DENSE",
    "A01:C-GROUPED",
    "A02:C-DENSE",
    "A03:C-DENSE",
    "A03:C-GROUPED",
    "A03:C-BW",
    "A04:C-DENSE",
    "A04:C-GROUPED",
    "A05:C-DENSE",
    "A05:C-GROUPED",
)
COMM_COMM_CASE_KEYS = ("A12", "A14", "A15", "A16", "A18", "A20")
CASE_IDS_SHAPE = ("A01", "A03", "A04", "A05")
MULTI_AGGRESSOR_CASES = ("M01", "M02", "M03", "M04", "M05b")
MULTI_AGGRESSOR_LABELS = {
    "M01": ("dense weight-grad GEMM", "P2P receive + AG"),
    "M02": ("CUTLASS grouped GEMM", "P2P send + P2P receive"),
    "M03": ("cuDNN SDPA bprop", "A2A + AG"),
    "M04": ("A2A", "P2P send + P2P receive"),
    "M05b": ("AG", "three-kernel compute train"),
}
MULTI_AGGRESSOR_COLORS = {
    "M01": "#7a5195",
    "M02": "#4c78a8",
    "M03": "#54a24b",
    "M04": "#4c78a8",
    "M05b": "#737373",
}
SHAPES_BY_CASE = {
    "A01": ("C-DENSE", "C-GROUPED"),
    "A03": ("C-DENSE", "C-GROUPED", "C-BW"),
    "A04": ("C-DENSE", "C-GROUPED"),
    "A05": ("C-DENSE", "C-GROUPED"),
}
COMM_LABEL = {
    "a2a": "A2A",
    "ag": "AG",
    "rs": "RS",
    "p2p_send": "P2P send",
    "p2p_recv": "P2P recv",
    "compute": "compute",
}
FIGURE1_RENDERED_VICTIM_ROWS = (
    ("C-GROUPED", "grouped GEMM"),
    ("p2p_send", "P2P send"),
    ("a2a", "A2A"),
    ("ag", "AG"),
    ("rs", "RS"),
)
FIGURE1_RENDERED_KERNEL_KEYS = tuple(row[0] for row in FIGURE1_RENDERED_VICTIM_ROWS)
FIGURE1_DISPATCH_COLUMNS = (
    ("victim_first", "victim launched first"),
    ("aggressor_first", "aggressor launched first"),
)
COMM_COLOR = {
    "p2p_send": "#4c78a8",
    "p2p_recv": "#72b7b2",
    "a2a": "#f58518",
    "ag": "#54a24b",
    "rs": "#e45756",
}
KERNEL_COLOR = {**COMM_COLOR, "compute": "#737373"}
SHAPE_LABEL = {"C-DENSE": "dense", "C-GROUPED": "grouped", "C-BW": "bandwidth"}
SHAPE_STYLE = {"C-DENSE": "-", "C-GROUPED": "--", "C-BW": ":"}
SHAPE_MARKER = {"C-DENSE": "o", "C-GROUPED": "s", "C-BW": "^"}
FIG3_BAND_LOW_PCT = 40.0
FIG3_BAND_HIGH_PCT = 60.0
FIG3_MIN_PAIRED_SAMPLES = 5

SAMPLE_COLUMNS = (
    "sample_id",
    "case_id",
    "case_key",
    "seq_len",
    "dispatch_mode",
    "victim_index",
    "victim_family",
    "aggressor_family",
    "requested_overlap_pct",
    "realized_overlap_pct",
    "victim_slowdown_pct",
    "union_overlap_us",
    "participant_outcomes",
    "analysis_included",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(figure: plt.Figure, png: Path, svg: Path) -> None:
    """Save byte-stable SVG metadata plus the review-resolution PNG."""
    figure.savefig(png, dpi=190)
    figure.savefig(svg, metadata={"Date": None})


def included(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "pass"})


def shape_for(case_key: str) -> str:
    _, separator, variant = str(case_key).partition(":")
    return variant if separator else ""


def case_label(case_key: str) -> str:
    case, separator, variant = str(case_key).partition(":")
    return f"{case} {SHAPE_LABEL.get(variant, variant)}" if separator else case


def comm_family(victim: str, aggressor: str) -> str:
    if victim == "compute":
        return aggressor
    if aggressor == "compute":
        return victim
    return aggressor


def read_frozen_samples(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, usecols=SAMPLE_COLUMNS)
    data = data[(data["seq_len"] == SEQ_LEN) & included(data["analysis_included"])].copy()
    if set(data["dispatch_mode"].unique()) - {"victim_first", "aggressor_first"}:
        raise ValueError("unexpected dispatch mode in frozen estimator-valid corpus")
    if data["sample_id"].duplicated().any():
        raise ValueError("sample_id is not unique")
    return data


def quantile25(values: pd.Series) -> float:
    return float(values.quantile(0.25))


def quantile75(values: pd.Series) -> float:
    return float(values.quantile(0.75))


def aggregate_directed_cells(data: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        data.groupby(
            [
                "case_id",
                "case_key",
                "dispatch_mode",
                "victim_index",
                "victim_family",
                "aggressor_family",
                "requested_overlap_pct",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            realized_overlap_median_pct=("realized_overlap_pct", "median"),
            realized_overlap_p25_pct=("realized_overlap_pct", quantile25),
            realized_overlap_p75_pct=("realized_overlap_pct", quantile75),
            slowdown_median_pct=("victim_slowdown_pct", "median"),
            slowdown_p25_pct=("victim_slowdown_pct", quantile25),
            slowdown_p75_pct=("victim_slowdown_pct", quantile75),
            sample_count=("victim_slowdown_pct", "size"),
        )
        .reset_index()
    )
    grouped["shape"] = grouped["case_key"].map(shape_for)
    grouped["case_label"] = grouped["case_key"].map(case_label)
    grouped["directed_label"] = grouped.apply(
        lambda row: f"{row['case_label']}: {COMM_LABEL.get(row['victim_family'], row['victim_family'])}"
        f" ← {COMM_LABEL.get(row['aggressor_family'], row['aggressor_family'])}",
        axis=1,
    )
    grouped["communication_family"] = grouped.apply(
        lambda row: comm_family(str(row["victim_family"]), str(row["aggressor_family"])), axis=1
    )
    return grouped


def figure1_source(cells: pd.DataFrame, keys: Iterable[str]) -> pd.DataFrame:
    result = cells[cells["case_key"].isin(tuple(keys))].copy()
    result["victim_kernel_key"] = np.where(
        result["victim_family"] == "compute",
        result["shape"],
        result["victim_family"],
    )
    return result.sort_values(
        ["victim_kernel_key", "dispatch_mode", "case_id", "case_key", "victim_index", "requested_overlap_pct"]
    ).reset_index(drop=True)


def add_n_labels(axis: plt.Axes, x: np.ndarray, y: np.ndarray, counts: np.ndarray) -> None:
    for xpos, ypos, count in zip(x, y, counts):
        axis.annotate(
            f"n={int(count)}",
            (xpos, ypos),
            xytext=(2, 5),
            textcoords="offset points",
            fontsize=6.7,
            color="#444444",
        )


def plot_figure1_by_victim(
    table: pd.DataFrame,
    victim_rows: tuple[tuple[str, str], ...],
    png: Path,
    svg: Path,
) -> None:
    """Render one row per victim kernel and one column per dispatch order."""
    present = set(table["victim_kernel_key"].astype(str))
    rows = tuple(row for row in victim_rows if row[0] in present)
    missing = {row[0] for row in victim_rows} - present
    if missing:
        raise ValueError(f"missing requested Figure 1 victim kernels: {sorted(missing)}")
    fig, axes = plt.subplots(
        len(rows),
        len(FIGURE1_DISPATCH_COLUMNS),
        figsize=(22, 4.0 * len(rows) + 2.0),
        sharex=True,
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, (victim_kernel_key, victim_kernel_label) in enumerate(rows):
        for column_index, (dispatch_mode, _) in enumerate(FIGURE1_DISPATCH_COLUMNS):
            axis = axes[row_index, column_index]
            subset = table[
                (table["victim_kernel_key"] == victim_kernel_key)
                & (table["dispatch_mode"] == dispatch_mode)
            ]
            for (_, victim_index), line in subset.groupby(["case_key", "victim_index"], sort=True):
                line = line.sort_values("realized_overlap_median_pct")
                aggressor = str(line["aggressor_family"].iloc[0])
                shape = str(line["shape"].iloc[0])
                x = line["realized_overlap_median_pct"].to_numpy(float)
                y = line["slowdown_median_pct"].to_numpy(float)
                lo = line["slowdown_p25_pct"].to_numpy(float)
                hi = line["slowdown_p75_pct"].to_numpy(float)
                color = KERNEL_COLOR.get(aggressor, "#4c78a8")
                marker = SHAPE_MARKER.get(shape, "o")
                aggressor_label = (
                    "grouped GEMM"
                    if aggressor == "compute" and shape == "C-GROUPED"
                    else COMM_LABEL.get(aggressor, aggressor)
                )
                axis.fill_between(x, lo, hi, color=color, alpha=0.11, linewidth=0)
                axis.plot(
                    x,
                    y,
                    color=color,
                    linestyle=SHAPE_STYLE.get(shape, "-"),
                    marker=marker,
                    linewidth=1.65,
                    markersize=4.5,
                    label=aggressor_label,
                )
            launch_position = "first" if dispatch_mode == "victim_first" else "second"
            axis.set_title(
                f"{victim_kernel_label} launched {launch_position}",
                fontsize=20,
            )
            axis.tick_params(axis="both", which="major", labelsize=15)
            axis.axhline(0, color="#666666", linewidth=0.7)
            axis.grid(True, color="#dddddd", linewidth=0.6)
            if subset.empty:
                axis.text(0.5, 0.5, "No eligible overlap cases", ha="center", va="center", transform=axis.transAxes)
            else:
                axis.legend(fontsize=15, loc="upper left", framealpha=0.9, ncol=2 if len(subset) > 8 else 1)
    fig.supxlabel("realized overlap (%)", fontsize=20)
    fig.supylabel("slowdown (%)", fontsize=20)
    save_figure(fig, png, svg)
    plt.close(fig)


def figure1_rendered_table(table: pd.DataFrame) -> pd.DataFrame:
    """Select retained kernels as both victims and aggressors without changing the source CSV."""
    aggressor_kernel_key = np.where(
        table["aggressor_family"] == "compute",
        table["shape"],
        table["aggressor_family"],
    )
    return table[
        table["victim_kernel_key"].isin(FIGURE1_RENDERED_KERNEL_KEYS)
        & pd.Series(aggressor_kernel_key, index=table.index).isin(FIGURE1_RENDERED_KERNEL_KEYS)
    ].copy()


def plot_figure1(table: pd.DataFrame, png: Path, svg: Path) -> None:
    plot_figure1_by_victim(
        figure1_rendered_table(table),
        FIGURE1_RENDERED_VICTIM_ROWS,
        png,
        svg,
    )


def multi_aggressor_source(cells: pd.DataFrame) -> pd.DataFrame:
    """Select the frozen authoritative multi-aggressor suite."""
    result = cells[cells["case_id"].isin(MULTI_AGGRESSOR_CASES)].copy()
    result["victim_label"] = result["case_id"].map(
        lambda case_id: MULTI_AGGRESSOR_LABELS[str(case_id)][0]
    )
    result["aggressor_group_label"] = result["case_id"].map(
        lambda case_id: MULTI_AGGRESSOR_LABELS[str(case_id)][1]
    )
    return result.sort_values(
        ["case_id", "dispatch_mode", "requested_overlap_pct"]
    ).reset_index(drop=True)


def plot_multi_aggressor_figure(table: pd.DataFrame, png: Path, svg: Path) -> None:
    """Render the multi-aggressor analogue of Figure 1.

    The complete aggressor group is a single line.  For constant-realized-overlap
    sequences (M04 and M05b victim-first), the curve is deliberately drawn as a
    vertical stack in requested-target order.  Thick translucent vertical IQR
    segments and requested-target annotations keep every retained point visible.
    """
    present = set(table["case_id"].astype(str))
    missing = set(MULTI_AGGRESSOR_CASES) - present
    if missing:
        raise ValueError(f"missing requested multi-aggressor cases: {sorted(missing)}")

    fig, axes = plt.subplots(
        len(MULTI_AGGRESSOR_CASES),
        len(FIGURE1_DISPATCH_COLUMNS),
        figsize=(18, 3.15 * len(MULTI_AGGRESSOR_CASES) + 1.6),
        sharex=True,
        sharey="row",
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, case_id in enumerate(MULTI_AGGRESSOR_CASES):
        victim_label, group_label = MULTI_AGGRESSOR_LABELS[case_id]
        color = MULTI_AGGRESSOR_COLORS[case_id]
        for column_index, (dispatch_mode, _) in enumerate(FIGURE1_DISPATCH_COLUMNS):
            axis = axes[row_index, column_index]
            line = table[
                (table["case_id"] == case_id)
                & (table["dispatch_mode"] == dispatch_mode)
            ].copy()
            if line.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No estimator-valid points",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            else:
                # Preserve requested-target order for a truly vertical response;
                # otherwise use the realized-overlap x order used by Figure 1.
                unique_realized = line["realized_overlap_median_pct"].round(9).nunique()
                vertical_stack = unique_realized == 1 and len(line) > 1
                line = line.sort_values(
                    "requested_overlap_pct" if vertical_stack else "realized_overlap_median_pct"
                )
                x = line["realized_overlap_median_pct"].to_numpy(float)
                y = line["slowdown_median_pct"].to_numpy(float)
                lo = line["slowdown_p25_pct"].to_numpy(float)
                hi = line["slowdown_p75_pct"].to_numpy(float)
                if vertical_stack:
                    # fill_between has zero area when x is constant.  These wide,
                    # translucent IQR segments are the vertical-stack equivalent.
                    axis.vlines(x, lo, hi, color=color, alpha=0.16, linewidth=8.0)
                else:
                    axis.fill_between(x, lo, hi, color=color, alpha=0.11, linewidth=0)
                axis.plot(
                    x,
                    y,
                    color=color,
                    linestyle="-",
                    marker="o",
                    linewidth=1.65,
                    markersize=4.5,
                    label=group_label,
                )
                if vertical_stack:
                    target_values = line["requested_overlap_pct"].to_numpy(float)
                    # Alternating offsets keep adjacent targets legible even when
                    # their slowdown medians differ by only a few percentage points.
                    text_offsets = np.where(np.arange(len(y)) % 2 == 0, 7.0, -7.0)
                    for xpos, ypos, target, text_offset in zip(
                        x, y, target_values, text_offsets
                    ):
                        axis.annotate(
                            f"request {target:g}%",
                            (xpos, ypos),
                            xytext=(-8, text_offset),
                            textcoords="offset points",
                            ha="right",
                            va="center",
                            fontsize=8,
                            color="#444444",
                        )
            launch_position = "first" if dispatch_mode == "victim_first" else "second"
            axis.set_title(
                f"{victim_label} slowdown: launched {launch_position}",
                fontsize=13,
            )
            axis.axhline(0, color="#666666", linewidth=0.7)
            axis.grid(True, color="#dddddd", linewidth=0.6)
            if not line.empty:
                axis.legend(fontsize=9, loc="upper left", framealpha=0.9)
    fig.supxlabel("median realized overlap (%)", fontsize=14)
    fig.supylabel("median slowdown (%)", fontsize=14)
    save_figure(fig, png, svg)
    plt.close(fig)


def build_figure2_matches(cells: pd.DataFrame) -> pd.DataFrame:
    selected = cells[cells["case_key"].isin(COMM_COMPUTE_CASE_KEYS)].copy()
    keys = [
        "case_id",
        "case_key",
        "victim_index",
        "victim_family",
        "aggressor_family",
        "communication_family",
        "shape",
        "case_label",
        "directed_label",
        "requested_overlap_pct",
    ]
    victim_first = selected[selected["dispatch_mode"] == "victim_first"].drop(columns="dispatch_mode")
    aggressor_first = selected[selected["dispatch_mode"] == "aggressor_first"].drop(columns="dispatch_mode")
    pairs = victim_first.merge(
        aggressor_first,
        on=keys,
        how="outer",
        suffixes=("_victim_first", "_aggressor_first"),
        indicator=True,
    )
    pairs["both_orders_present"] = pairs["_merge"] == "both"
    pairs["realized_overlap_difference_pp"] = (
        pairs["realized_overlap_median_pct_victim_first"]
        - pairs["realized_overlap_median_pct_aggressor_first"]
    ).abs()
    pairs["overlap_match_tolerance_pp"] = np.select(
        [
            pairs["both_orders_present"] & (pairs["realized_overlap_difference_pp"] <= 5.0),
            pairs["both_orders_present"] & (pairs["realized_overlap_difference_pp"] <= 10.0),
        ],
        [5.0, 10.0],
        default=np.nan,
    )
    pairs["match_status"] = np.select(
        [
            pairs["overlap_match_tolerance_pp"] == 5.0,
            pairs["overlap_match_tolerance_pp"] == 10.0,
            ~pairs["both_orders_present"],
        ],
        ["matched_within_5pp", "matched_within_10pp_fallback", "missing_order"],
        default="unmatched_over_10pp",
    )
    pairs["selected_for_figure"] = False
    direction_keys = ["case_key", "victim_index"]
    for _, group in pairs.groupby(direction_keys, sort=True):
        candidates = group[group["overlap_match_tolerance_pp"].notna()].copy()
        if candidates.empty:
            continue
        candidates["minimum_realized_overlap_pct"] = candidates[
            [
                "realized_overlap_median_pct_victim_first",
                "realized_overlap_median_pct_aggressor_first",
            ]
        ].min(axis=1)
        best = candidates.sort_values(
            ["minimum_realized_overlap_pct", "realized_overlap_difference_pp", "requested_overlap_pct"],
            ascending=[False, True, False],
        ).index[0]
        pairs.loc[best, "selected_for_figure"] = True
    return pairs.drop(columns="_merge").sort_values(
        ["victim_family", "case_id", "case_key", "requested_overlap_pct"]
    ).reset_index(drop=True)


def plot_figure2(matches: pd.DataFrame, png: Path, svg: Path) -> None:
    chosen = matches[matches["selected_for_figure"]].copy()
    panels = (("compute", "compute ← communication"), ("communication", "communication ← compute"))
    fig, axes = plt.subplots(1, 2, figsize=(19, 11), sharex=True, constrained_layout=True)
    for axis, (kind, title) in zip(axes, panels):
        subset = chosen[
            (chosen["victim_family"] == "compute")
            if kind == "compute"
            else (chosen["victim_family"] != "compute")
        ].copy()
        subset = subset.sort_values(["case_id", "case_key"], ascending=[False, False]).reset_index(drop=True)
        positions = np.arange(len(subset))
        for y, row in zip(positions, subset.itertuples(index=False)):
            color = COMM_COLOR[str(row.communication_family)]
            vf = float(row.slowdown_median_pct_victim_first)
            af = float(row.slowdown_median_pct_aggressor_first)
            axis.plot([vf, af], [y, y], color=color, linewidth=2.0, alpha=0.8)
            axis.hlines(
                y - 0.08,
                float(row.slowdown_p25_pct_victim_first),
                float(row.slowdown_p75_pct_victim_first),
                color=color,
                linewidth=5,
                alpha=0.35,
            )
            axis.hlines(
                y + 0.08,
                float(row.slowdown_p25_pct_aggressor_first),
                float(row.slowdown_p75_pct_aggressor_first),
                color=color,
                linewidth=5,
                alpha=0.7,
            )
            axis.scatter(vf, y - 0.08, marker="<", s=58, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)
            axis.scatter(af, y + 0.08, marker=">", s=58, facecolor=color, edgecolor=color, linewidth=1.2, zorder=3)
            axis.annotate(
                f"victim first n={int(row.sample_count_victim_first)}",
                (vf, y - 0.08),
                xytext=(0, -9),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=6.7,
            )
            axis.annotate(
                f"aggressor first n={int(row.sample_count_aggressor_first)}",
                (af, y + 0.08),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.7,
            )
        labels = [
            f"{row.case_label}\nrequest {row.requested_overlap_pct:g}% · realized "
            f"{row.realized_overlap_median_pct_victim_first:.1f}/{row.realized_overlap_median_pct_aggressor_first:.1f}%"
            for row in subset.itertuples(index=False)
        ]
        axis.set_yticks(positions, labels)
        axis.set_title(title)
        axis.axvline(0, color="#666666", linewidth=0.8)
        axis.grid(True, axis="x", color="#dddddd", linewidth=0.7)
    axes[0].set_xlabel("Median victim slowdown (%)")
    axes[1].set_xlabel("Median victim slowdown (%)")
    fig.suptitle(
        "Figure 2 · Launch-order effect within a fixed directed case\n"
        "Highest matched-overlap target per direction; horizontal bars are slowdown IQR",
        fontsize=17,
    )
    save_figure(fig, png, svg)
    plt.close(fig)


def build_figure3_paired_trials(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    selected = data[data["case_key"].isin(COMM_COMPUTE_CASE_KEYS)].copy()
    records: list[dict[str, object]] = []
    intersection_max_error_us = 0.0
    physical_order_mismatches = 0
    malformed = 0
    for row in selected.itertuples(index=False):
        try:
            outcomes = json.loads(row.participant_outcomes)
        except (TypeError, json.JSONDecodeError):
            malformed += 1
            continue
        if len(outcomes) != 2:
            malformed += 1
            continue
        compute = [outcome for outcome in outcomes if outcome.get("family") == "compute"]
        communication = [outcome for outcome in outcomes if outcome.get("family") in COMM_FAMILIES]
        if len(compute) != 1 or len(communication) != 1:
            malformed += 1
            continue
        starts = sorted((float(outcome["start_ms"]), str(outcome["family"])) for outcome in outcomes)
        physical_first_family = starts[0][1]
        physical_order = "compute_first" if physical_first_family == "compute" else "communication_first"
        expected_first = row.aggressor_family if row.dispatch_mode == "aggressor_first" else row.victim_family
        physical_order_mismatches += int(physical_first_family != expected_first)
        intersection_us = 1000.0 * max(
            0.0,
            min(float(outcome["end_ms"]) for outcome in outcomes)
            - max(float(outcome["start_ms"]) for outcome in outcomes),
        )
        intersection_max_error_us = max(
            intersection_max_error_us, abs(intersection_us - float(row.union_overlap_us))
        )
        shorter_concurrent_us = 1000.0 * min(float(outcome["concurrent_ms"]) for outcome in outcomes)
        symmetric_overlap_pct = 100.0 * intersection_us / shorter_concurrent_us
        in_band = FIG3_BAND_LOW_PCT <= symmetric_overlap_pct <= FIG3_BAND_HIGH_PCT
        records.append(
            {
                "sample_id": row.sample_id,
                "case_id": row.case_id,
                "case_key": row.case_key,
                "case_label": case_label(row.case_key),
                "shape": shape_for(row.case_key),
                "communication_family": communication[0]["family"],
                "nominated_victim_family": row.victim_family,
                "dispatch_mode": row.dispatch_mode,
                "physical_order": physical_order,
                "shared_intersection_us": intersection_us,
                "shorter_concurrent_duration_us": shorter_concurrent_us,
                "symmetric_realized_overlap_pct": symmetric_overlap_pct,
                "overlap_band_low_pct": FIG3_BAND_LOW_PCT,
                "overlap_band_high_pct": FIG3_BAND_HIGH_PCT,
                "in_selected_overlap_band": in_band,
                "communication_slowdown_pct": float(communication[0]["slowdown_pct"]),
                "compute_slowdown_pct": float(compute[0]["slowdown_pct"]),
                "compute_minus_communication_slowdown_pp": float(compute[0]["slowdown_pct"])
                - float(communication[0]["slowdown_pct"]),
            }
        )
    return pd.DataFrame.from_records(records), {
        "input_rows": int(len(selected)),
        "parsed_paired_rows": int(len(records)),
        "malformed_rows": int(malformed),
        "physical_order_mismatches": int(physical_order_mismatches),
        "max_intersection_vs_union_error_us": float(intersection_max_error_us),
    }


def build_figure3_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for case_key in COMM_COMPUTE_CASE_KEYS:
        case_trials = trials[trials["case_key"] == case_key]
        for physical_order in ("communication_first", "compute_first"):
            group = case_trials[
                (case_trials["physical_order"] == physical_order)
                & case_trials["in_selected_overlap_band"]
            ]
            row: dict[str, object] = {
                "case_id": str(case_key).split(":", 1)[0],
                "case_key": case_key,
                "case_label": case_label(case_key),
                "shape": shape_for(case_key),
                "communication_family": (
                    str(case_trials["communication_family"].iloc[0]) if len(case_trials) else ""
                ),
                "physical_order": physical_order,
                "overlap_band_low_pct": FIG3_BAND_LOW_PCT,
                "overlap_band_high_pct": FIG3_BAND_HIGH_PCT,
                "paired_sample_count": int(len(group)),
                "minimum_required_paired_samples": FIG3_MIN_PAIRED_SAMPLES,
                "available": bool(len(group) >= FIG3_MIN_PAIRED_SAMPLES),
            }
            if len(group):
                row.update(
                    {
                        "symmetric_realized_overlap_median_pct": float(group["symmetric_realized_overlap_pct"].median()),
                        "communication_slowdown_median_pct": float(group["communication_slowdown_pct"].median()),
                        "communication_slowdown_p25_pct": quantile25(group["communication_slowdown_pct"]),
                        "communication_slowdown_p75_pct": quantile75(group["communication_slowdown_pct"]),
                        "compute_slowdown_median_pct": float(group["compute_slowdown_pct"].median()),
                        "compute_slowdown_p25_pct": quantile25(group["compute_slowdown_pct"]),
                        "compute_slowdown_p75_pct": quantile75(group["compute_slowdown_pct"]),
                        "paired_difference_median_pp": float(
                            group["compute_minus_communication_slowdown_pp"].median()
                        ),
                        "paired_difference_p25_pp": quantile25(
                            group["compute_minus_communication_slowdown_pp"]
                        ),
                        "paired_difference_p75_pp": quantile75(
                            group["compute_minus_communication_slowdown_pp"]
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def plot_figure3(summary: pd.DataFrame, png: Path, svg: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(19, 11), sharex=True, constrained_layout=True)
    for axis, physical_order in zip(axes, ("communication_first", "compute_first")):
        subset = summary[summary["physical_order"] == physical_order].copy()
        subset = subset.sort_values(["case_id", "case_key"], ascending=[False, False]).reset_index(drop=True)
        positions = np.arange(len(subset))
        for y, row in zip(positions, subset.itertuples(index=False)):
            if not bool(row.available):
                axis.annotate(
                    f"unavailable (n={int(row.paired_sample_count)}; need {FIG3_MIN_PAIRED_SAMPLES})",
                    (0, y),
                    color="#777777",
                    fontsize=8,
                    va="center",
                )
                continue
            color = COMM_COLOR[str(row.communication_family)]
            comm = float(row.communication_slowdown_median_pct)
            comp = float(row.compute_slowdown_median_pct)
            axis.plot([comm, comp], [y, y], color=color, linewidth=2.0, alpha=0.8)
            axis.hlines(
                y - 0.08,
                float(row.communication_slowdown_p25_pct),
                float(row.communication_slowdown_p75_pct),
                color=color,
                linewidth=5,
                alpha=0.7,
            )
            axis.hlines(
                y + 0.08,
                float(row.compute_slowdown_p25_pct),
                float(row.compute_slowdown_p75_pct),
                color=color,
                linewidth=5,
                alpha=0.3,
            )
            axis.scatter(comm, y - 0.08, marker="D", s=54, color=color, zorder=3)
            axis.scatter(comp, y + 0.08, marker="o", s=58, facecolor="white", edgecolor=color, linewidth=1.8, zorder=3)
            axis.annotate("comm", (comm, y - 0.08), xytext=(0, -9), textcoords="offset points", ha="center", va="top", fontsize=6.7)
            axis.annotate("compute", (comp, y + 0.08), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=6.7)
            axis.annotate(
                f"paired n={int(row.paired_sample_count)} · overlap {row.symmetric_realized_overlap_median_pct:.1f}%",
                (max(comm, comp), y),
                xytext=(7, 0),
                textcoords="offset points",
                fontsize=6.8,
                va="center",
            )
        axis.set_yticks(positions, subset["case_label"].tolist())
        axis.set_title(physical_order.replace("_", " "))
        axis.axvline(0, color="#666666", linewidth=0.8)
        axis.grid(True, axis="x", color="#dddddd", linewidth=0.7)
        axis.set_xlabel("Median participant slowdown (%)")
    fig.suptitle(
        "Figure 3 · Paired participant response under fixed physical launch order\n"
        f"Same-trial outcomes; actual shared overlap covers {FIG3_BAND_LOW_PCT:g}–{FIG3_BAND_HIGH_PCT:g}% of the shorter concurrent participant; bars are IQR",
        fontsize=17,
    )
    save_figure(fig, png, svg)
    plt.close(fig)


def build_figure4_selection(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = cells[
        cells["case_id"].isin(CASE_IDS_SHAPE)
        & (cells["victim_family"] == "compute")
        & (cells["aggressor_family"].isin(COMM_FAMILIES))
        & (cells["dispatch_mode"] == "aggressor_first")
    ].copy()
    candidates["selected_for_figure"] = False
    candidates["shape_match_tolerance_pp"] = np.nan
    candidates["shape_comparison_overlap_range_pp"] = np.nan
    selection_records: list[dict[str, object]] = []
    for case_id in CASE_IDS_SHAPE:
        group = candidates[candidates["case_id"] == case_id]
        variants = SHAPES_BY_CASE[case_id]
        choices = [group[group["shape"] == variant].to_dict("records") for variant in variants]
        if any(not choice for choice in choices):
            selection_records.append({"case_id": case_id, "available": False, "reason": "missing_shape"})
            continue
        chosen_combo = None
        chosen_tolerance = None
        for tolerance in (5.0, 10.0):
            valid: list[tuple[float, float, float, tuple[dict[str, object], ...]]] = []
            for combo in itertools.product(*choices):
                overlaps = [float(item["realized_overlap_median_pct"]) for item in combo]
                overlap_range = max(overlaps) - min(overlaps)
                if overlap_range <= tolerance:
                    valid.append((min(overlaps), float(np.mean(overlaps)), overlap_range, combo))
            if valid:
                valid.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
                chosen_combo = valid[0]
                chosen_tolerance = tolerance
                break
        if chosen_combo is None or chosen_tolerance is None:
            selection_records.append({"case_id": case_id, "available": False, "reason": "no_match_within_10pp"})
            continue
        minimum_overlap, average_overlap, overlap_range, combo = chosen_combo
        for item in combo:
            mask = (
                (candidates["case_key"] == item["case_key"])
                & (candidates["victim_index"] == item["victim_index"])
                & (candidates["requested_overlap_pct"] == item["requested_overlap_pct"])
            )
            candidates.loc[mask, "selected_for_figure"] = True
            candidates.loc[mask, "shape_match_tolerance_pp"] = chosen_tolerance
            candidates.loc[mask, "shape_comparison_overlap_range_pp"] = overlap_range
        selection_records.append(
            {
                "case_id": case_id,
                "available": True,
                "reason": "matched",
                "shape_match_tolerance_pp": chosen_tolerance,
                "minimum_realized_overlap_pct": minimum_overlap,
                "average_realized_overlap_pct": average_overlap,
                "realized_overlap_range_pp": overlap_range,
                "selected_shapes": len(combo),
            }
        )
    return candidates.sort_values(["case_id", "case_key", "requested_overlap_pct"]), pd.DataFrame(selection_records)


def plot_figure4(candidates: pd.DataFrame, png: Path, svg: Path) -> None:
    chosen = candidates[candidates["selected_for_figure"]]
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharex=True, constrained_layout=True)
    for axis, case_id in zip(axes.flat, CASE_IDS_SHAPE):
        subset = chosen[chosen["case_id"] == case_id].copy()
        order = list(SHAPES_BY_CASE[case_id])
        subset["shape_order"] = subset["shape"].map({shape: index for index, shape in enumerate(order)})
        subset = subset.sort_values("shape_order", ascending=False).reset_index(drop=True)
        positions = np.arange(len(subset))
        for y, row in zip(positions, subset.itertuples(index=False)):
            color = COMM_COLOR[str(row.communication_family)]
            axis.hlines(
                y,
                float(row.slowdown_p25_pct),
                float(row.slowdown_p75_pct),
                color=color,
                linewidth=5,
                alpha=0.45,
            )
            axis.scatter(
                float(row.slowdown_median_pct),
                y,
                marker=SHAPE_MARKER[str(row.shape)],
                s=78,
                color=color,
                zorder=3,
            )
            axis.annotate(
                f"overlap {row.realized_overlap_median_pct:.1f}% · n={int(row.sample_count)}",
                (float(row.slowdown_median_pct), y),
                xytext=(7, 0),
                textcoords="offset points",
                fontsize=8,
                va="center",
            )
        family = COMM_LABEL[str(subset["communication_family"].iloc[0])] if len(subset) else case_id
        tolerance = float(subset["shape_match_tolerance_pp"].iloc[0]) if len(subset) else np.nan
        axis.set_yticks(positions, [SHAPE_LABEL[str(shape)] for shape in subset["shape"]])
        axis.set_title(f"{case_id} · {family} aggressor · match ≤{tolerance:g} pp")
        axis.axvline(0, color="#666666", linewidth=0.8)
        axis.grid(True, axis="x", color="#dddddd", linewidth=0.7)
        axis.set_xlabel("Median compute-victim slowdown (%)")
    minimum = float(chosen["slowdown_p25_pct"].min())
    maximum = float(
        chosen[["slowdown_p75_pct", "slowdown_median_pct"]].max(axis=1).max()
    )
    # Annotations are deliberately outside the point markers; reserve enough
    # room that the high-slowdown A05 labels remain inside the axes.
    span = max(maximum - minimum, 1.0)
    axes.flat[0].set_xlim(min(-5.0, minimum - 0.06 * span), maximum + max(22.0, 0.24 * span))
    fig.suptitle(
        "Figure 4 · Compute kernel-shape susceptibility\n"
        "Communication aggressor first; each panel uses its highest mutually matched realized-overlap cells; bars are IQR",
        fontsize=17,
    )
    save_figure(fig, png, svg)
    plt.close(fig)


def write_readme(
    output: Path,
    fig1: pd.DataFrame,
    fig1_rendered: pd.DataFrame,
    fig1_compute: pd.DataFrame,
    fig1_comm: pd.DataFrame,
    multi: pd.DataFrame,
    fig2: pd.DataFrame,
    fig3: pd.DataFrame,
    fig4: pd.DataFrame,
) -> None:
    fig2_selected = fig2[fig2["selected_for_figure"]].copy()
    fig2_selected["aggressor_minus_victim_first_pp"] = (
        fig2_selected["slowdown_median_pct_aggressor_first"]
        - fig2_selected["slowdown_median_pct_victim_first"]
    )
    fig3_available = fig3[fig3["available"]].copy()
    fig4_selected = fig4[fig4["selected_for_figure"]].copy()
    largest_order = fig2_selected.loc[
        fig2_selected["aggressor_minus_victim_first_pp"].abs().idxmax()
    ]
    largest_shape_case = None
    largest_shape_span = -np.inf
    for case_id, group in fig4_selected.groupby("case_id"):
        span = float(group["slowdown_median_pct"].max() - group["slowdown_median_pct"].min())
        if span > largest_shape_span:
            largest_shape_span = span
            largest_shape_case = case_id
    unavailable = fig3[~fig3["available"]]
    content = f"""# Factor-separated timing figures

These figures implement Section 10.3 of the benchmark plan using only the frozen,
estimator-valid `{SEQ_LEN}`-token corpus. No benchmark was rerun and no physical-resource
proxy is presented as a hardware counter.

## Figure 1: realized-overlap dose response

- `figure1_overlap_response_by_victim.*` renders grouped GEMM, P2P send, A2A, AG, and RS as both the only victims and the only line-series aggressors, with victim-first and aggressor-first columns for ten panels total. The rendered subset contains {len(fig1_rendered)} aggregated points across {fig1_rendered.groupby(['case_key', 'victim_index', 'dispatch_mode']).ngroups} directed dispatch-specific lines.
- `figure1_overlap_response_by_victim_source.csv` remains the complete combined source: {len(fig1)} aggregated communication x compute and communication x communication points across {fig1.groupby(['case_key', 'victim_index', 'dispatch_mode']).ngroups} directed dispatch-specific lines, including the three victim kernels intentionally omitted from rendering.
- At sequence length 12,288, every `C-GROUPED` line runs the same deterministic expert grouped-dW GEMM; cases differ by the overlapping communication kernel, requested overlap, dispatch order, and trial.
- Solid/dashed/dotted gray lines identify dense/grouped/bandwidth compute aggressors, while colored lines identify communication aggressors and columns encode launch order.
- Takeaway: the compact victim-centric layout focuses on grouped-GEMM and the four requested communication victims while retaining the full source data for reproducibility.
- Caveat: the curves are associations because slowdown itself can extend realized overlap; A2A auxiliary and multi-aggressor cases remain intentionally excluded.

## Multi-aggressor companion: realized-overlap dose response

- `multi_aggressor_overlap_response_by_victim.*` uses the same five-row, two-column conventions as Figure 1 for M01, M02, M03, M04, and M05b. Each legend names the complete preserved aggressor group, rather than implying attribution to an individual member.
- Victim titles use the exact execution-matrix identities: M01 dense weight-grad GEMM, M02 CUTLASS grouped GEMM, M03 cuDNN SDPA bprop, M04 A2A, and M05b AG.
- M05a is not plotted because it is absent from the frozen authoritative final corpus; no synthetic or non-final data are substituted.
- The plot contains {len(multi)} aggregated points from {int(multi['sample_count'].sum())} estimator-valid trials across {multi.groupby(['case_id', 'dispatch_mode']).ngroups} case/order lines. Missing requested targets had no estimator-valid retained samples and are not interpolated.
- M04 and M05b victim-first are genuine vertical response stacks at 100% median realized overlap. Requested-target annotations and thick translucent IQR segments expose every point instead of collapsing the sequence into one marker.
- Takeaway: group translation and launch order can change slowdown even at fixed 100% union coverage, so realized union overlap alone does not determine multi-aggressor interference.

## Figure 2: launch-order effect

- Each row chooses the highest-overlap requested target at which victim-first and aggressor-first median realized overlap match within 5 percentage points, falling back to 10 only when necessary.
- Both `compute ← communication` and `communication ← compute` directions are shown and never pooled.
- Largest selected order difference: `{largest_order.directed_label}` at requested overlap {largest_order.requested_overlap_pct:g}%, {largest_order.aggressor_minus_victim_first_pp:+.2f} slowdown percentage points (aggressor-first minus victim-first).
- Caveat: this is a matched aggregate comparison, not a trial-paired estimate.

## Figure 3: direction under fixed physical launch order

- Both participant slowdowns come from the same trial. Physical order is derived from actual global GPU starts and checked against dispatch semantics.
- The comparison band is {FIG3_BAND_LOW_PCT:g}–{FIG3_BAND_HIGH_PCT:g}% actual shared intersection divided by the shorter participant's concurrent duration. At least {FIG3_MIN_PAIRED_SAMPLES} paired samples are required.
- {len(fig3_available)} of {len(fig3)} case/order cells are available; {len(unavailable)} {'is' if len(unavailable) == 1 else 'are'} explicitly marked unavailable rather than interpolated.
- Takeaway: participant susceptibility reverses with physical order in several grouped-kernel cases, which is why a direction plot cannot fix only aggressor-first.

## Figure 4: kernel-shape effect

- Within A01/A03/A04/A05, the plot selects the highest mutually matched realized-overlap cells across the available dense/grouped/bandwidth shapes while holding compute as victim and communication as aggressor-first.
- Largest selected within-family shape span: {largest_shape_case}, {largest_shape_span:.2f} slowdown percentage points.
- Caveat: overlap is matched within each communication family, not across families; A03 requires the documented 10-point fallback.

## Reproduction

```sh
python generate_factor_figures.py
```

The manifest records the frozen input hash, filters, selection rules, validations,
and output hashes. CSV files are the machine-readable source tables behind each
figure. `figure3_paired_trials.csv` preserves the same-trial participant outcomes
used by Figure 3.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--samples",
        type=Path,
        default=default_root / "merged-results" / "tables" / "samples.csv",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    samples = read_frozen_samples(args.samples)
    cells = aggregate_directed_cells(samples)

    fig1_compute = figure1_source(cells, COMM_COMPUTE_CASE_KEYS)
    fig1_comm = figure1_source(cells, COMM_COMM_CASE_KEYS)
    fig1 = (
        pd.concat([fig1_compute, fig1_comm], ignore_index=True)
        .sort_values(
            ["victim_kernel_key", "dispatch_mode", "case_id", "case_key", "victim_index", "requested_overlap_pct"]
        )
        .reset_index(drop=True)
    )
    fig1.to_csv(args.output / "figure1_overlap_response_by_victim_source.csv", index=False)
    fig1_rendered = figure1_rendered_table(fig1)
    plot_figure1(
        fig1,
        args.output / "figure1_overlap_response_by_victim.png",
        args.output / "figure1_overlap_response_by_victim.svg",
    )

    multi = multi_aggressor_source(cells)
    multi.to_csv(
        args.output / "multi_aggressor_overlap_response_by_victim_source.csv",
        index=False,
    )
    plot_multi_aggressor_figure(
        multi,
        args.output / "multi_aggressor_overlap_response_by_victim.png",
        args.output / "multi_aggressor_overlap_response_by_victim.svg",
    )

    fig2 = build_figure2_matches(cells)
    fig2.to_csv(args.output / "figure2_launch_order_matches.csv", index=False)
    plot_figure2(
        fig2,
        args.output / "figure2_launch_order_effect.png",
        args.output / "figure2_launch_order_effect.svg",
    )

    fig3_trials, fig3_parse_validation = build_figure3_paired_trials(samples)
    fig3_trials.to_csv(args.output / "figure3_paired_trials.csv", index=False)
    fig3 = build_figure3_summary(fig3_trials)
    fig3.to_csv(args.output / "figure3_fixed_physical_order_source.csv", index=False)
    plot_figure3(
        fig3,
        args.output / "figure3_fixed_physical_order.png",
        args.output / "figure3_fixed_physical_order.svg",
    )

    fig4, fig4_comparisons = build_figure4_selection(cells)
    fig4.to_csv(args.output / "figure4_kernel_shape_candidates.csv", index=False)
    fig4_comparisons.to_csv(args.output / "figure4_kernel_shape_comparisons.csv", index=False)
    plot_figure4(
        fig4,
        args.output / "figure4_kernel_shape_effect.png",
        args.output / "figure4_kernel_shape_effect.svg",
    )

    expected_fig1_compute_directions = 2 * len(COMM_COMPUTE_CASE_KEYS)
    expected_fig1_comm_directions = 2 * len(COMM_COMM_CASE_KEYS)
    expected_fig1_compute_lines = len(FIGURE1_DISPATCH_COLUMNS) * expected_fig1_compute_directions
    expected_fig1_comm_lines = len(FIGURE1_DISPATCH_COLUMNS) * expected_fig1_comm_directions
    expected_fig1_lines = expected_fig1_compute_lines + expected_fig1_comm_lines
    fig2_selected = fig2[fig2["selected_for_figure"]]
    multi_vertical_stacks = []
    for (case_id, dispatch_mode), group in multi.groupby(
        ["case_id", "dispatch_mode"], sort=True
    ):
        if len(group) > 1 and group["realized_overlap_median_pct"].round(9).nunique() == 1:
            multi_vertical_stacks.append(
                {
                    "case_id": str(case_id),
                    "dispatch_mode": str(dispatch_mode),
                    "realized_overlap_pct": float(group["realized_overlap_median_pct"].iloc[0]),
                    "point_count": int(len(group)),
                }
            )
    multi_expected_cells = pd.MultiIndex.from_product(
        [MULTI_AGGRESSOR_CASES, [mode for mode, _ in FIGURE1_DISPATCH_COLUMNS], [0.0, 5.0, 25.0, 50.0, 75.0, 100.0]],
        names=["case_id", "dispatch_mode", "requested_overlap_pct"],
    )
    multi_present_cells = pd.MultiIndex.from_frame(
        multi[["case_id", "dispatch_mode", "requested_overlap_pct"]]
    )
    multi_missing_cells = [
        {
            "case_id": str(case_id),
            "dispatch_mode": str(dispatch_mode),
            "requested_overlap_pct": float(requested_overlap_pct),
        }
        for case_id, dispatch_mode, requested_overlap_pct in multi_expected_cells.difference(
            multi_present_cells
        )
    ]
    validation = {
        "source_estimator_valid_12k_rows": int(len(samples)),
        "figure1": {
            "combined_directed_dispatch_lines": int(
                fig1.groupby(["case_key", "victim_index", "dispatch_mode"]).ngroups
            ),
            "combined_expected_directed_lines": expected_fig1_lines,
            "combined_points": int(len(fig1)),
            "combined_samples": int(fig1["sample_count"].sum()),
            "source_victim_kernels": sorted(fig1["victim_kernel_key"].unique().tolist()),
            "rendered_victim_kernels": [row[0] for row in FIGURE1_RENDERED_VICTIM_ROWS],
            "rendered_panels": 2 * len(FIGURE1_RENDERED_VICTIM_ROWS),
            "rendered_aggressor_kernels": sorted(
                set(
                    np.where(
                        fig1_rendered["aggressor_family"] == "compute",
                        fig1_rendered["shape"],
                        fig1_rendered["aggressor_family"],
                    )
                )
            ),
            "rendered_directed_dispatch_lines": int(
                fig1_rendered.groupby(["case_key", "victim_index", "dispatch_mode"]).ngroups
            ),
            "rendered_points": int(len(fig1_rendered)),
            "rendered_samples": int(fig1_rendered["sample_count"].sum()),
            "comm_x_compute_directed_dispatch_lines": int(
                fig1_compute.groupby(["case_key", "victim_index", "dispatch_mode"]).ngroups
            ),
            "comm_x_compute_expected_directed_lines": expected_fig1_compute_lines,
            "comm_x_compute_points": int(len(fig1_compute)),
            "comm_x_compute_samples": int(fig1_compute["sample_count"].sum()),
            "comm_x_comm_directed_dispatch_lines": int(
                fig1_comm.groupby(["case_key", "victim_index", "dispatch_mode"]).ngroups
            ),
            "comm_x_comm_expected_directed_lines": expected_fig1_comm_lines,
            "comm_x_comm_points": int(len(fig1_comm)),
            "comm_x_comm_samples": int(fig1_comm["sample_count"].sum()),
            "both_dispatch_modes": bool(
                set(fig1["dispatch_mode"]) == {"victim_first", "aggressor_first"}
            ),
            "compute_victims_partitioned_by_exact_variant": bool(
                set(fig1.loc[fig1["victim_family"] == "compute", "victim_kernel_key"])
                == {"C-DENSE", "C-GROUPED", "C-BW"}
            ),
        },
        "multi_aggressor_companion": {
            "case_ids": list(MULTI_AGGRESSOR_CASES),
            "victim_labels": {
                case_id: MULTI_AGGRESSOR_LABELS[case_id][0]
                for case_id in MULTI_AGGRESSOR_CASES
            },
            "case_count": int(multi["case_id"].nunique()),
            "panels": int(2 * multi["case_id"].nunique()),
            "case_dispatch_lines": int(
                multi.groupby(["case_id", "dispatch_mode"]).ngroups
            ),
            "expected_case_dispatch_lines": int(2 * len(MULTI_AGGRESSOR_CASES)),
            "points": int(len(multi)),
            "samples": int(multi["sample_count"].sum()),
            "both_dispatch_modes_per_case": bool(
                all(
                    set(group["dispatch_mode"]) == {"victim_first", "aggressor_first"}
                    for _, group in multi.groupby("case_id")
                )
            ),
            "missing_estimator_valid_requested_cells": multi_missing_cells,
            "vertical_stack_lines": multi_vertical_stacks,
            "vertical_stacks_explicitly_rendered": True,
        },
        "figure2": {
            "expected_directed_comparisons": expected_fig1_compute_directions,
            "selected_directed_comparisons": int(len(fig2_selected)),
            "selected_within_5pp": int((fig2_selected["overlap_match_tolerance_pp"] == 5).sum()),
            "selected_10pp_fallback": int((fig2_selected["overlap_match_tolerance_pp"] == 10).sum()),
            "unavailable_directed_comparisons": int(expected_fig1_compute_directions - len(fig2_selected)),
            "maximum_selected_overlap_difference_pp": float(fig2_selected["realized_overlap_difference_pp"].max()),
            "selected_10pp_fallbacks": fig2_selected.loc[
                fig2_selected["overlap_match_tolerance_pp"] == 10,
                ["case_key", "victim_family", "aggressor_family", "requested_overlap_pct", "realized_overlap_difference_pp"],
            ].to_dict("records"),
        },
        "figure3": {
            **fig3_parse_validation,
            "overlap_band_low_pct": FIG3_BAND_LOW_PCT,
            "overlap_band_high_pct": FIG3_BAND_HIGH_PCT,
            "minimum_paired_samples": FIG3_MIN_PAIRED_SAMPLES,
            "expected_case_order_cells": int(2 * len(COMM_COMPUTE_CASE_KEYS)),
            "available_case_order_cells": int(fig3["available"].sum()),
            "unavailable_case_order_cells": int((~fig3["available"]).sum()),
            "unavailable": fig3.loc[
                ~fig3["available"], ["case_key", "physical_order", "paired_sample_count"]
            ].to_dict("records"),
        },
        "figure4": {
            "expected_shape_comparisons": len(CASE_IDS_SHAPE),
            "available_shape_comparisons": int(fig4_comparisons["available"].sum()),
            "selected_shape_cells": int(fig4["selected_for_figure"].sum()),
            "within_5pp": int((fig4_comparisons["shape_match_tolerance_pp"] == 5).sum()),
            "10pp_fallback": int((fig4_comparisons["shape_match_tolerance_pp"] == 10).sum()),
            "maximum_overlap_range_pp": float(fig4_comparisons["realized_overlap_range_pp"].max()),
            "10pp_fallbacks": fig4_comparisons.loc[
                fig4_comparisons["shape_match_tolerance_pp"] == 10
            ].to_dict("records"),
        },
    }

    failures = []
    if validation["figure1"]["combined_directed_dispatch_lines"] != expected_fig1_lines:
        failures.append("figure1 combined line coverage")
    if validation["figure1"]["comm_x_compute_directed_dispatch_lines"] != expected_fig1_compute_lines:
        failures.append("figure1 comm_x_compute line coverage")
    if validation["figure1"]["comm_x_comm_directed_dispatch_lines"] != expected_fig1_comm_lines:
        failures.append("figure1 comm_x_comm line coverage")
    if validation["figure1"]["rendered_panels"] != 2 * len(validation["figure1"]["rendered_victim_kernels"]):
        failures.append("figure1 victim/order panel coverage")
    if set(validation["figure1"]["rendered_victim_kernels"]) != {
        "C-GROUPED", "p2p_send", "a2a", "ag", "rs"
    }:
        failures.append("figure1 requested rendered victim selection")
    if set(validation["figure1"]["rendered_aggressor_kernels"]) != {
        "C-GROUPED", "p2p_send", "a2a", "ag", "rs"
    }:
        failures.append("figure1 requested rendered aggressor selection")
    if not validation["figure1"]["both_dispatch_modes"]:
        failures.append("figure1 dispatch coverage")
    if not validation["figure1"]["compute_victims_partitioned_by_exact_variant"]:
        failures.append("figure1 compute victim partition")
    if validation["multi_aggressor_companion"]["case_count"] != len(MULTI_AGGRESSOR_CASES):
        failures.append("multi-aggressor case coverage")
    if validation["multi_aggressor_companion"]["victim_labels"] != {
        "M01": "dense weight-grad GEMM",
        "M02": "CUTLASS grouped GEMM",
        "M03": "cuDNN SDPA bprop",
        "M04": "A2A",
        "M05b": "AG",
    }:
        failures.append("multi-aggressor exact victim labels")
    if validation["multi_aggressor_companion"]["case_dispatch_lines"] != 2 * len(MULTI_AGGRESSOR_CASES):
        failures.append("multi-aggressor case/order line coverage")
    if not validation["multi_aggressor_companion"]["both_dispatch_modes_per_case"]:
        failures.append("multi-aggressor dispatch coverage")
    if {
        (row["case_id"], row["dispatch_mode"])
        for row in validation["multi_aggressor_companion"]["vertical_stack_lines"]
    } != {("M04", "victim_first"), ("M05b", "victim_first")}:
        failures.append("multi-aggressor vertical-stack detection")
    if validation["figure2"]["selected_directed_comparisons"] != expected_fig1_compute_directions:
        failures.append("figure2 directed match coverage")
    if validation["figure2"]["maximum_selected_overlap_difference_pp"] > 10.0:
        failures.append("figure2 overlap match tolerance")
    if validation["figure3"]["malformed_rows"] != 0:
        failures.append("figure3 participant parsing")
    if validation["figure3"]["physical_order_mismatches"] != 0:
        failures.append("figure3 physical order")
    if validation["figure3"]["max_intersection_vs_union_error_us"] > 1e-3:
        failures.append("figure3 overlap reconstruction")
    if validation["figure4"]["available_shape_comparisons"] != len(CASE_IDS_SHAPE):
        failures.append("figure4 shape coverage")
    if validation["figure4"]["maximum_overlap_range_pp"] > 10.0:
        failures.append("figure4 overlap match tolerance")
    validation["status"] = "pass" if not failures else "fail"
    validation["failures"] = failures
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"figure validation failed: {failures}")

    legacy_figure1_paths = (
        "figure1a_comm_x_compute_overlap_response.png",
        "figure1a_comm_x_compute_overlap_response.svg",
        "figure1a_comm_x_compute_source.csv",
        "figure1b_comm_x_comm_overlap_response.png",
        "figure1b_comm_x_comm_overlap_response.svg",
        "figure1b_comm_x_comm_source.csv",
    )
    for name in legacy_figure1_paths:
        (args.output / name).unlink(missing_ok=True)

    write_readme(
        args.output,
        fig1,
        fig1_rendered,
        fig1_compute,
        fig1_comm,
        multi,
        fig2,
        fig3,
        fig4,
    )

    artifacts = sorted(
        path
        for path in args.output.iterdir()
        if path.is_file() and path.name not in {"manifest.json", "SHA256SUMS"}
    )
    manifest = {
        "schema_version": 1,
        "scope": "Section 10.3 timing-only factor-separated Figures 1--4 plus multi-aggressor Figure 1 companion",
        "physical_resource_figures_included": False,
        "gpu_benchmarks_rerun": False,
        "source": {"path": str(args.samples.resolve()), "sha256": sha256(args.samples)},
        "filters": {"seq_len": SEQ_LEN, "analysis_included": True},
        "selection_rules": {
            "figure1": "complete combined source CSV retained; render only grouped GEMM, P2P send, A2A, AG, and RS as both victim rows and aggressor line series; victim-first and aggressor-first columns",
            "multi_aggressor_companion": "M01, M02, M03, M04, and M05b only; exact execution-matrix victim labels; one row per victim, victim-first and group-first columns; each line is the complete preserved aggressor group; constant-realized-overlap responses remain vertical stacks with explicit target annotations",
            "figure2": "same directed case and requested target; prefer median realized-overlap difference <=5pp, fallback <=10pp; choose highest matched overlap",
            "figure3": f"same-trial paired outcomes; actual physical order; symmetric realized overlap in [{FIG3_BAND_LOW_PCT}, {FIG3_BAND_HIGH_PCT}]%; minimum n={FIG3_MIN_PAIRED_SAMPLES}",
            "figure4": "compute victim, communication aggressor, aggressor_first; choose highest all-shape match <=5pp, fallback <=10pp",
        },
        "validation": validation,
        "artifacts": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in artifacts},
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_paths = artifacts + [args.output / "manifest.json"]
    with (args.output / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in checksum_paths:
            handle.write(f"{sha256(path)}  {path.name}\n")


if __name__ == "__main__":
    main()
