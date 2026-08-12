#!/usr/bin/env python3
"""Validate and summarize the Figure 1 isolated 3-warmup/10-measurement run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any


LABELS = {
    "ISO-A03-GROUPED-COMPUTE": "Grouped GEMM",
    "ISO-A01-GROUPED-P2P-SEND": "P2P send",
    "ISO-A03-GROUPED-A2A": "A2A",
    "ISO-A04-GROUPED-AG": "AG",
    "ISO-A05-GROUPED-RS": "RS",
}
EXPECTED_ACTIVE_RANKS = {
    "compute": 1,
    "p2p_send": 2,
    "a2a": 8,
    "ag": 8,
    "rs": 8,
}
METRIC_NAMES = (
    "sm_util_pct",
    "dram_bw_util_pct",
    "nvlink_total_gib_per_s",
    "pcie_total_gib_per_s",
    "ib_endpoint_gib_per_s",
)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate percentile of an empty list")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def node_ib_gib_per_s(measured: dict[str, Any]) -> float:
    total_bytes_per_second = 0.0
    for delta in measured["ib_deltas"]:
        elapsed_ns = int(delta["end_timestamp_ns"]) - int(delta["start_timestamp_ns"])
        if elapsed_ns <= 0:
            raise ValueError("non-positive IB counter interval")
        total_bytes_per_second += float(delta["delta_bytes"]) / (elapsed_ns / 1e9)
    return total_bytes_per_second / (1024.0 ** 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.raw_dir / "collection_manifest.json").read_text())
    if manifest["warmups_per_point"] != 3 or manifest["measured_trials_per_point"] != 10:
        raise ValueError("collection manifest is not the requested 3-warmup/10-measurement design")

    point_files = sorted((args.raw_dir / "points").glob("*.json"))
    if len(point_files) != len(LABELS):
        raise ValueError(f"expected {len(LABELS)} point files, found {len(point_files)}")

    trial_rows: list[dict[str, Any]] = []
    validation_points = []
    for point_path in point_files:
        document = json.loads(point_path.read_text())
        point = document["point"]
        signature_id = point["signature_id"]
        if signature_id not in LABELS:
            raise ValueError(f"unexpected signature {signature_id}")
        rank_results = document["rank_results"]
        active = [
            result for result in rank_results
            if any(measured["physical"] is not None for measured in result["measured"])
        ]
        expected_active = EXPECTED_ACTIVE_RANKS[point["family"]]
        if len(active) != expected_active:
            raise ValueError(
                f"{signature_id}: expected {expected_active} active ranks, found {len(active)}"
            )
        for result in rank_results:
            if result["warmup_trial_count"] != 3 or result["measured_trial_count"] != 10:
                raise ValueError(f"{signature_id}: rank trial accounting mismatch")
        active_nodes = sorted(
            {measured["node_ip"] for result in active for measured in result["measured"]}
        )
        if point["family"] == "p2p_send" and len(active_nodes) != 2:
            raise ValueError(f"{signature_id}: P2P endpoints are not cross-node")
        if point["family"] in {"a2a", "ag", "rs"} and len(active_nodes) != 1:
            raise ValueError(f"{signature_id}: collective ranks are not within one node")

        for trial_index in range(1, 11):
            active_measured = [result["measured"][trial_index - 1] for result in active]
            physical = [measured["physical"] for measured in active_measured]
            if any(item is None for item in physical):
                raise ValueError(f"{signature_id}: missing physical metric at trial {trial_index}")
            if any(not item["gpm_interval_valid"] for item in physical):
                raise ValueError(f"{signature_id}: invalid GPM interval at trial {trial_index}")
            if any(float(measured["active_duration_us_sum"]) < 120_000.0 for measured in active_measured):
                raise ValueError(f"{signature_id}: active burst below 120 ms at trial {trial_index}")
            metrics = [item["metrics"] for item in physical]
            ib_by_node = [
                node_ib_gib_per_s(result["measured"][trial_index - 1])
                for result in rank_results
                if result["measured"][trial_index - 1]["ib_deltas"]
            ]
            if len(ib_by_node) != 2:
                raise ValueError(f"{signature_id}: expected two node-level IB readers")
            trial_rows.append({
                "victim": LABELS[signature_id],
                "signature_id": signature_id,
                "family": point["family"],
                "trial": trial_index,
                "active_ranks": len(active),
                "active_nodes": "|".join(active_nodes),
                "sm_util_pct": statistics.median(float(item["sm_util_pct"]) for item in metrics),
                "dram_bw_util_pct": statistics.median(float(item["dram_bw_util_pct"]) for item in metrics),
                "nvlink_total_gib_per_s": statistics.median(
                    (float(item["nvlink_total_rx_mib_per_s"]) + float(item["nvlink_total_tx_mib_per_s"])) / 1024.0
                    for item in metrics
                ),
                "pcie_total_gib_per_s": statistics.median(
                    (float(item["pcie_rx_mib_per_s"]) + float(item["pcie_tx_mib_per_s"])) / 1024.0
                    for item in metrics
                ),
                "ib_endpoint_gib_per_s": statistics.median(ib_by_node),
                "gpm_interval_us_min": min(float(item["gpm_interval_us"]) for item in physical),
                "active_duration_us_min": min(float(item["active_duration_us_sum"]) for item in active_measured),
            })
        validation_points.append({
            "signature_id": signature_id,
            "family": point["family"],
            "active_rank_count": len(active),
            "active_nodes": active_nodes,
            "topology": (
                "cross-node two-rank P2P"
                if point["family"] == "p2p_send"
                else "within-node eight-rank collective"
                if point["family"] in {"a2a", "ag", "rs"}
                else "single GPU"
            ),
        })

    summary_rows = []
    for victim in LABELS.values():
        selected = [row for row in trial_rows if row["victim"] == victim]
        if len(selected) != 10:
            raise ValueError(f"{victim}: expected 10 trial rows, found {len(selected)}")
        row: dict[str, Any] = {
            "victim": victim,
            "signature_id": selected[0]["signature_id"],
            "topology": next(
                item["topology"] for item in validation_points
                if item["signature_id"] == selected[0]["signature_id"]
            ),
            "active_ranks": selected[0]["active_ranks"],
            "measured_trials": len(selected),
        }
        for metric in METRIC_NAMES:
            values = [float(item[metric]) for item in selected]
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_p25"] = percentile(values, 0.25)
            row[f"{metric}_p75"] = percentile(values, 0.75)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        summary_rows.append(row)

    write_csv(args.raw_dir / "resource_metrics_trials.csv", trial_rows)
    write_csv(args.raw_dir / "resource_metrics_summary.csv", summary_rows)
    validation = {
        "status": "pass",
        "warmups_per_point": 3,
        "measured_trials_per_point": 10,
        "point_count": len(summary_rows),
        "active_endpoint_trial_count": sum(
            int(row["active_rank_count"]) * 10 for row in validation_points
        ),
        "all_active_endpoint_trials_gpm_valid": True,
        "all_active_endpoint_trials_at_least_120ms": True,
        "points": validation_points,
        "aggregation": {
            "gpu_metrics": "median across active GPUs within each trial, then trial distribution",
            "ib": "sum of all HCA TX+RX rates per node; median across the two node endpoints within each trial",
            "interval_semantics": "NVML GPM wall-interval average over repeated exact isolated operations; no duty-cycle correction",
        },
    }
    (args.raw_dir / "resource_metrics_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    def interval_cell(row: dict[str, Any], metric: str, digits: int) -> str:
        return (
            f"{row[f'{metric}_median']:.{digits}f} "
            f"[{row[f'{metric}_p25']:.{digits}f}, {row[f'{metric}_p75']:.{digits}f}]"
        )

    report_lines = [
        "# Figure 1 isolated resource metrics",
        "",
        "Each entry is the median across 10 measured trials; brackets contain the",
        "25th and 75th percentiles. Each trial first takes the median across the",
        "operation's active GPU endpoints. The run used exactly 3 warmups.",
        "",
        "| Operation | Topology | SM util (%) | DRAM BW util (%) | NVLink RX+TX (GiB/s) | PCIe RX+TX (GiB/s) | IB TX+RX per endpoint (GiB/s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            "| " + " | ".join([
                str(row["victim"]),
                str(row["topology"]),
                interval_cell(row, "sm_util_pct", 2),
                interval_cell(row, "dram_bw_util_pct", 2),
                interval_cell(row, "nvlink_total_gib_per_s", 2),
                interval_cell(row, "pcie_total_gib_per_s", 3),
                interval_cell(row, "ib_endpoint_gib_per_s", 3),
            ]) + " |"
        )
    report_lines.extend([
        "",
        "## Measurement semantics",
        "",
        "- SM and DRAM are NVML GPM IDs 2 and 10.",
        "- NVLink is the sum of NVML GPM RX and TX rate IDs 60 and 61.",
        "- PCIe is the sum of NVML GPM RX and TX rate IDs 20 and 21.",
        "- IB is derived from `/sys/class/infiniband/*/ports/*/counters/port_{xmit,rcv}_data`; all HCA rates are summed per node, then the two node endpoint rates are median-reduced.",
        "- NVML values are GPU-wide wall-interval averages over repeated exact isolated operations, with no duty-cycle correction.",
        "- P2P is a two-rank cross-node operation. A2A, AG, and RS are eight-rank collectives contained within one node.",
        "",
    ])
    (args.raw_dir / "RESOURCE_METRICS_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(args.raw_dir / "resource_metrics_summary.csv")


if __name__ == "__main__":
    main()
