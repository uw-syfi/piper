#!/usr/bin/env python3
"""Plot the completed Qwen sharding/schedule benchmark matrix."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "out" / "20260813_qwen_sharding_matrix_rerun" / "summary.csv"
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "charts"

SCHEDULES = ("dualpipev", "interleaved_1f1b", "1f1b", "no_pp")
SCHEDULE_LABELS = {
    "dualpipev": "DualPipeV",
    "1f1b": "1F1B",
    "interleaved_1f1b": "Interleaved 1F1B",
    "no_pp": "No PP",
}
PAIRS = (
    ("none", "off"),
    ("none", "128"),
    ("grad", "off"),
    ("grad", "128"),
    ("param_grad", "off"),
    ("param_grad", "128"),
)
PAIR_LABELS = {
    ("none", "off"): "No sharding / off",
    ("none", "128"): "No sharding / 128 MB",
    ("grad", "off"): "Grad / off",
    ("grad", "128"): "Grad / 128 MB",
    ("param_grad", "off"): "Param + grad / off",
    ("param_grad", "128"): "Param + grad / 128 MB",
}
PAIR_TICK_LABELS = {
    ("none", "off"): "None\noff",
    ("none", "128"): "None\n128 MB",
    ("grad", "off"): "Grad\noff",
    ("grad", "128"): "Grad\n128 MB",
    ("param_grad", "off"): "Param + grad\noff",
    ("param_grad", "128"): "Param + grad\n128 MB",
}


def load_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {
        (row["variant"], row["sharding"], row["bucket_mb"]): row
        for row in rows
    }
    if len(keyed) != 24:
        raise ValueError(f"expected 24 unique rows, found {len(keyed)} in {path}")
    return keyed


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "font.size": 10,
            "legend.fontsize": 9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#d7dce2",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.85,
        }
    )


def finish(fig: plt.Figure, output: Path) -> None:
    fig.savefig(output, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def execution_mode(rows: dict) -> str:
    modes = {str(row.get("use_inductor", "")).lower() for row in rows.values()}
    if modes == {"false"}:
        return "No Inductor"
    if modes == {"true"}:
        return "Inductor"
    return "Mixed execution modes"


def throughput_by_schedule(rows: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    colors = plt.get_cmap("tab10").colors[: len(PAIRS)]
    centers = np.arange(len(SCHEDULES), dtype=float)
    width = 0.125
    offsets = (np.arange(len(PAIRS)) - (len(PAIRS) - 1) / 2) * width
    chart_max = 0.0

    for pair_index, pair in enumerate(PAIRS):
        values = []
        errors = []
        for schedule in SCHEDULES:
            row = rows.get((schedule, pair[0], pair[1]))
            values.append(
                np.nan if row is None else float(row["throughput_tokens_per_s"]) / 1000
            )
            errors.append(
                0.0
                if row is None
                else float(row.get("throughput_sample_std_tokens_per_s", 0.0)) / 1000
            )
        chart_max = max(chart_max, max(value + error for value, error in zip(values, errors)))
        ax.bar(
            centers + offsets[pair_index],
            values,
            width=width * 0.9,
            color=colors[pair_index],
            label=PAIR_LABELS[pair],
            edgecolor="white",
            linewidth=0.6,
            yerr=errors,
            capsize=2.5,
            error_kw={"ecolor": "#222222", "elinewidth": 0.9, "capthick": 0.9},
        )

    ax.set_title(
        f"Qwen3 30B L4 throughput, grouped by schedule — {execution_mode(rows)}\n"
        "Error bars: sample SD of per-step throughput (n=30)"
    )
    ax.set_ylabel("Aggregate throughput (thousand tokens/s)")
    ax.set_xlabel("Schedule")
    ax.set_xticks(centers, [SCHEDULE_LABELS[item] for item in SCHEDULES])
    ax.set_ylim(0, chart_max * 1.12)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title="Sharding / bucket", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.subplots_adjust(bottom=0.32)
    finish(fig, output)


def throughput_by_pair(rows: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.8))
    colors = ("#2563eb", "#ea580c", "#16a34a", "#7c3aed")
    centers = np.arange(len(PAIRS), dtype=float)
    width = 0.19
    offsets = (np.arange(len(SCHEDULES)) - (len(SCHEDULES) - 1) / 2) * width
    chart_max = 0.0

    for schedule_index, schedule in enumerate(SCHEDULES):
        values = []
        errors = []
        for pair in PAIRS:
            row = rows.get((schedule, pair[0], pair[1]))
            values.append(
                np.nan if row is None else float(row["throughput_tokens_per_s"]) / 1000
            )
            errors.append(
                0.0
                if row is None
                else float(row.get("throughput_sample_std_tokens_per_s", 0.0)) / 1000
            )
        chart_max = max(chart_max, max(value + error for value, error in zip(values, errors)))
        ax.bar(
            centers + offsets[schedule_index],
            values,
            width=width * 0.9,
            color=colors[schedule_index],
            label=SCHEDULE_LABELS[schedule],
            edgecolor="white",
            linewidth=0.6,
            yerr=errors,
            capsize=2.5,
            error_kw={"ecolor": "#222222", "elinewidth": 0.9, "capthick": 0.9},
        )

    ax.set_title(
        "Qwen3 30B L4 throughput, grouped by sharding and bucketing"
        f" — {execution_mode(rows)}\n"
        "Error bars: sample SD of per-step throughput (n=30)"
    )
    ax.set_ylabel("Aggregate throughput (thousand tokens/s)")
    ax.set_xlabel("Sharding and bucket configuration")
    ax.set_xticks(centers, [PAIR_TICK_LABELS[item] for item in PAIRS])
    ax.set_ylim(0, chart_max * 1.12)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.legend(title="Schedule", ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.subplots_adjust(bottom=0.34)
    finish(fig, output)


def peak_memory_by_schedule(rows: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    colors = plt.get_cmap("tab10").colors[: len(PAIRS)]
    centers = np.arange(len(SCHEDULES), dtype=float)
    pair_spacing = 0.115
    pair_offsets = (np.arange(len(PAIRS)) - (len(PAIRS) - 1) / 2) * pair_spacing
    pp_offsets = (-0.018, 0.018)
    markers = ("o", "X")

    for schedule_index, schedule in enumerate(SCHEDULES):
        for pair_index, pair in enumerate(PAIRS):
            row = rows.get((schedule, pair[0], pair[1]))
            if row is None:
                continue
            for pp_rank in (0, 1):
                value = row.get(f"peak_memory_pp{pp_rank}_gb", "")
                if not value:
                    continue
                ax.scatter(
                    centers[schedule_index] + pair_offsets[pair_index] + pp_offsets[pp_rank],
                    float(value),
                    s=72,
                    marker=markers[pp_rank],
                    color=colors[pair_index],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )

    ax.set_title("Qwen3 30B L4 peak memory per PP rank, grouped by schedule")
    ax.set_ylabel("Peak allocated memory (GiB)")
    ax.set_xlabel("Schedule")
    ax.set_xticks(centers, [SCHEDULE_LABELS[item] for item in SCHEDULES])
    ax.set_ylim(30, 60)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    pair_handles = [
        Patch(facecolor=colors[index], label=PAIR_LABELS[pair])
        for index, pair in enumerate(PAIRS)
    ]
    pp_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[rank],
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=8,
            label=f"PP{rank}",
        )
        for rank in (0, 1)
    ]
    ax.legend(
        handles=pair_handles + pp_handles,
        title="Color: sharding / bucket    Marker: PP rank",
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
    )
    fig.subplots_adjust(bottom=0.28)
    finish(fig, output)


def peak_memory_by_pair(rows: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    colors = ("#2563eb", "#ea580c", "#16a34a", "#7c3aed")
    centers = np.arange(len(PAIRS), dtype=float)
    schedule_spacing = 0.17
    schedule_offsets = (
        np.arange(len(SCHEDULES)) - (len(SCHEDULES) - 1) / 2
    ) * schedule_spacing
    pp_offsets = (-0.024, 0.024)
    markers = ("o", "X")

    for pair_index, pair in enumerate(PAIRS):
        for schedule_index, schedule in enumerate(SCHEDULES):
            row = rows.get((schedule, pair[0], pair[1]))
            if row is None:
                continue
            for pp_rank in (0, 1):
                value = row.get(f"peak_memory_pp{pp_rank}_gb", "")
                if not value:
                    continue
                ax.scatter(
                    centers[pair_index]
                    + schedule_offsets[schedule_index]
                    + pp_offsets[pp_rank],
                    float(value),
                    s=72,
                    marker=markers[pp_rank],
                    color=colors[schedule_index],
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=3,
                )

    ax.set_title("Qwen3 30B L4 peak memory per PP rank, grouped by sharding and bucketing")
    ax.set_ylabel("Peak allocated memory (GiB)")
    ax.set_xlabel("Sharding and bucket configuration")
    ax.set_xticks(centers, [PAIR_TICK_LABELS[item] for item in PAIRS])
    ax.set_ylim(30, 60)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    schedule_handles = [
        Patch(facecolor=colors[index], label=SCHEDULE_LABELS[schedule])
        for index, schedule in enumerate(SCHEDULES)
    ]
    pp_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[rank],
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=8,
            label=f"PP{rank}",
        )
        for rank in (0, 1)
    ]
    ax.legend(
        handles=schedule_handles + pp_handles,
        title="Color: schedule    Marker: PP rank",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    fig.subplots_adjust(bottom=0.29)
    finish(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    outputs = (
        ("01_throughput_by_schedule.png", throughput_by_schedule),
        ("02_throughput_by_sharding_bucketing.png", throughput_by_pair),
        ("03_peak_memory_by_schedule.png", peak_memory_by_schedule),
        ("04_peak_memory_by_sharding_bucketing.png", peak_memory_by_pair),
    )
    for filename, plotter in outputs:
        destination = args.output_dir / filename
        plotter(rows, destination)
        print(destination.resolve())


if __name__ == "__main__":
    main()
