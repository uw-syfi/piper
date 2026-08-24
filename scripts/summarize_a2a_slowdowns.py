#!/usr/bin/env python3
"""Create a unified, slowdown-sorted table from A2A interference benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


COMPUTE_MODE_LABELS = {
    "concurrent_compute_first": "compute first",
    "concurrent_a2a_first": "A2A first",
}
COLLECTIVE_MODE_LABELS = {
    "concurrent_allreduce_first": "gradient collective first",
    "concurrent_a2a_first": "A2A first",
}

COMPUTE_LABELS = {
    "route_expert": "MoE route + packed expert backward",
    "expert": "Packed expert backward",
    "linear_projection": "Fused QKV+Z projection backward surrogate",
    "gated_q_projection": "Exact gated-Q projection backward",
    "linear_qkv_projection": "Exact Gated-DeltaNet QKV projection backward",
    "linear_z_projection": "Exact Gated-DeltaNet Z projection backward",
    "linear_out_projection": "Exact Gated-DeltaNet output projection backward",
    "route_expert_linear_layer": "MoE + Gated-DeltaNet linear-layer backward",
    "route_expert_full_attention_layer": "MoE + full-attention-layer backward",
}


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def med(values: Iterable[float]) -> float:
    return statistics.median(list(values))


def world_critical_samples(
    rank_samples: list[dict[str, list[dict[str, float]]]],
    concurrent_mode: str,
) -> dict[str, list[float]]:
    sample_count = min(
        min(len(samples["a2a_solo"]), len(samples[concurrent_mode]))
        for samples in rank_samples
    )
    solo: list[float] = []
    concurrent: list[float] = []
    slowdown: list[float] = []
    delay: list[float] = []
    for index in range(sample_count):
        solo_value = max(
            float(samples["a2a_solo"][index]["a2a_ms"])
            for samples in rank_samples
        )
        concurrent_value = max(
            float(samples[concurrent_mode][index]["a2a_ms"])
            for samples in rank_samples
        )
        solo.append(solo_value)
        concurrent.append(concurrent_value)
        slowdown.append(concurrent_value / solo_value)
        delay.append(concurrent_value - solo_value)
    return {
        "solo": solo,
        "concurrent": concurrent,
        "slowdown": slowdown,
        "delay": delay,
    }


def source_quality(relative_path: Path) -> tuple[str, str]:
    text = relative_path.as_posix()
    if "smoke" in text:
        return "excluded", "smoke test"
    if "mnnvl_probe" in text:
        return "excluded", "connectivity probe"
    superseded = {
        "qwen397b_interference/ep_scaling_32k/results.json",
        "qwen397b_interference/logical_ep72_compute_32k/results.json",
        "qwen397b_interference/long_32k_ep16/results.json",
        "qwen397b_interference/standard_4k_ep16/results.json",
    }
    if text in superseded:
        return "excluded", "superseded by stable/stream-reuse run"
    return "authoritative", ""


def summarize_samples(samples: dict[str, list[float]]) -> dict[str, float]:
    return {
        "a2a_solo_ms": med(samples["solo"]),
        "a2a_concurrent_ms": med(samples["concurrent"]),
        "a2a_critical_delay_ms": med(samples["delay"]),
        "a2a_slowdown_ratio": med(samples["slowdown"]),
        "a2a_slowdown_p95": percentile(samples["slowdown"], 0.95),
        "sample_count": len(samples["slowdown"]),
    }


def compute_overlap_label(case: dict[str, Any], compute: dict[str, Any]) -> str:
    profile = case["compute_profile"]
    label = COMPUTE_LABELS.get(profile, profile)
    if "local_experts" in compute:
        label += f" ({compute['local_experts']} local experts"
        if case.get("local_expert_variant"):
            label += f", {case['local_expert_variant']}"
        label += ")"
    compute_repeats = int(case.get("compute_repeats", 1))
    if compute_repeats != 1:
        label += f" × {compute_repeats}"
    return label


def extract_compute(
    path: Path, document: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    quality, exclusion = source_quality(path.relative_to(root))
    rank_case_maps = [
        {case["case_name"]: case for case in result["cases"]}
        for result in document["rank_results"]
    ]
    rows: list[dict[str, Any]] = []
    for aggregate_case in document["aggregate"]:
        case_name = aggregate_case["case_name"]
        case = aggregate_case["case"]
        compute = aggregate_case["compute"]
        rank_samples = [
            rank_cases[case_name]["samples"] for rank_cases in rank_case_maps
        ]
        a2a_repeats = int(case.get("a2a_repeats", 1))
        for mode, order in COMPUTE_MODE_LABELS.items():
            if mode not in rank_samples[0]:
                continue
            raw = world_critical_samples(rank_samples, mode)
            summary = summarize_samples(raw)
            rows.append(
                {
                    "family": "A2A × backward compute",
                    "topology": (
                        f"physical EP{case['physical_ep']} "
                        f"(logical EP{case['logical_ep']}), 1 NVLink domain"
                    ),
                    "overlap": compute_overlap_label(case, compute),
                    "queue_order": order,
                    "device_offset": "none",
                    "local_tokens": int(case["local_tokens"]),
                    "a2a_payload_mib_per_call": float(
                        aggregate_case["payload_mib"]
                    ),
                    "a2a_calls": a2a_repeats,
                    "transport_dtype": case["comm_dtype"],
                    "quality": quality,
                    "exclusion_reason": exclusion,
                    "source": str(path),
                    "_samples": raw,
                    "_condition": (
                        "compute",
                        int(case["physical_ep"]),
                        int(case["logical_ep"]),
                        str(case["compute_profile"]),
                        str(case.get("local_expert_variant", "")),
                        int(case.get("compute_repeats", 1)),
                        int(case.get("a2a_repeats", 1)),
                        int(case["local_tokens"]),
                        float(aggregate_case["payload_mib"]),
                        str(case["comm_dtype"]),
                        order,
                    ),
                    **summary,
                }
            )
    return rows


def collective_buckets_mib(document: dict[str, Any]) -> list[float]:
    config = document["resolved_config"]["allreduce"]
    return [
        float(bucket["bytes"]) / (1024**2) for bucket in config["buckets"]
    ]


def format_buckets(buckets: list[float]) -> str:
    return ",".join(f"{value:g}" for value in buckets)


def extract_collective(
    path: Path, document: dict[str, Any], root: Path
) -> list[dict[str, Any]]:
    quality, exclusion = source_quality(path.relative_to(root))
    args = document["command_args"]
    config = document["resolved_config"]
    rank_samples = [result["samples"] for result in document["rank_results"]]
    buckets = collective_buckets_mib(document)
    primitive = str(args.get("gradient_collective", "allreduce"))
    primitive_label = (
        "Reduce-scatter" if primitive == "reduce_scatter" else "All-reduce"
    )
    a2a_delay = int(args.get("a2a_delay_cycles", 0))
    gradient_delay = int(args.get("gradient_delay_cycles", 0))
    offsets = []
    if a2a_delay:
        offsets.append(f"A2A delayed {a2a_delay / 1_000_000:g}M cycles")
    if gradient_delay:
        offsets.append(
            f"gradient delayed {gradient_delay / 1_000_000:g}M cycles"
        )
    device_offset = "; ".join(offsets) or "none"
    domain_map = config["rank_map"]["node_domain_map"]
    domain_count = len(set(domain_map))
    overlap = f"{primitive_label} [{format_buckets(buckets)}] MiB"
    rows: list[dict[str, Any]] = []
    for mode, order in COLLECTIVE_MODE_LABELS.items():
        if mode not in rank_samples[0]:
            continue
        raw = world_critical_samples(rank_samples, mode)
        summary = summarize_samples(raw)
        rows.append(
            {
                "family": "A2A × gradient collective",
                "topology": (
                    f"EP{args['ep_size']}×DP{args['dp_size']}, "
                    f"{domain_count} NVLink domain"
                    f"{'' if domain_count == 1 else 's'}"
                ),
                "overlap": overlap,
                "queue_order": order,
                "device_offset": device_offset,
                "local_tokens": int(
                    config["training_shape"][
                        "local_tokens_per_ep_source_rank"
                    ]
                ),
                "a2a_payload_mib_per_call": float(
                    config["a2a"]["mib_per_rank"]
                ),
                "a2a_calls": 1,
                "transport_dtype": config["a2a"]["dtype"],
                "quality": quality,
                "exclusion_reason": exclusion,
                "source": str(path),
                "_samples": raw,
                "_condition": (
                    "collective",
                    int(args["ep_size"]),
                    int(args["dp_size"]),
                    domain_count,
                    primitive,
                    tuple(buckets),
                    a2a_delay,
                    gradient_delay,
                    int(
                        config["training_shape"][
                            "local_tokens_per_ep_source_rank"
                        ]
                    ),
                    float(config["a2a"]["mib_per_rank"]),
                    str(config["a2a"]["dtype"]),
                    order,
                ),
                **summary,
            }
        )
    return rows


def extract_all(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        aggregate = document.get("aggregate")
        if (
            path.is_relative_to(root / "qwen397b_interference")
            and isinstance(aggregate, list)
            and "rank_results" in document
        ):
            rows.extend(extract_compute(path, document, root))
        elif (
            (
                path.is_relative_to(root / "qwen397b_a2a_allreduce")
                or path.is_relative_to(root / "qwen397b_a2a_gradient_matrix")
            )
            and isinstance(aggregate, dict)
            and "modes" in aggregate
            and "rank_results" in document
            and "resolved_config" in document
        ):
            rows.extend(extract_collective(path, document, root))
    return rows


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def group_authoritative(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["quality"] == "authoritative":
            grouped[row["_condition"]].append(row)

    summaries: list[dict[str, Any]] = []
    for condition_rows in grouped.values():
        first = condition_rows[0]
        pooled = {
            metric: [
                value
                for row in condition_rows
                for value in row["_samples"][metric]
            ]
            for metric in ("solo", "concurrent", "slowdown", "delay")
        }
        summary = summarize_samples(pooled)
        summaries.append(
            {
                "family": first["family"],
                "topology": first["topology"],
                "overlap": first["overlap"],
                "queue_order": first["queue_order"],
                "device_offset": first["device_offset"],
                "local_tokens": first["local_tokens"],
                "a2a_payload_mib_per_call": first[
                    "a2a_payload_mib_per_call"
                ],
                "a2a_calls": first["a2a_calls"],
                "transport_dtype": first["transport_dtype"],
                **summary,
                "a2a_slowdown_min": min(pooled["slowdown"]),
                "a2a_slowdown_max": max(pooled["slowdown"]),
                "run_count": len(condition_rows),
                "sources": ";".join(
                    sorted({row["source"] for row in condition_rows})
                ),
            }
        )
    summaries.sort(key=lambda row: row["a2a_slowdown_ratio"], reverse=True)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def payload_label(row: dict[str, Any]) -> str:
    payload = f"{row['a2a_payload_mib_per_call']:g}"
    if row["a2a_calls"] != 1:
        payload += f" × {row['a2a_calls']}"
    return payload


def write_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
) -> None:
    excluded = [row for row in all_rows if row["quality"] != "authoritative"]
    lines = [
        "# A2A slowdown across interference microbenchmarks",
        "",
        (
            "Rows are distinct workload/topology/queue-order conditions, sorted "
            "by median A2A slowdown. Repeated authoritative runs are pooled at "
            "the aligned world-critical-sample level."
        ),
        "",
        (
            "`Slowdown = max-rank concurrent A2A / max-rank solo A2A` for each "
            "aligned sample. The table reports the median of those ratios. "
            "Payload is MiB per rank per A2A call."
        ),
        "",
        "| Rank | Family | Topology | Overlapping workload | Queue order | Device offset | Tokens | A2A MiB | Solo ms | Concurrent ms | Delay ms | Slowdown | p95 | Samples | Runs |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(summaries, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    row["family"],
                    row["topology"],
                    row["overlap"],
                    row["queue_order"],
                    row["device_offset"],
                    str(row["local_tokens"]),
                    payload_label(row),
                    f"{row['a2a_solo_ms']:.3f}",
                    f"{row['a2a_concurrent_ms']:.3f}",
                    f"{row['a2a_critical_delay_ms']:.3f}",
                    f"{row['a2a_slowdown_ratio']:.4f}×",
                    f"{row['a2a_slowdown_p95']:.4f}×",
                    str(row["sample_count"]),
                    str(row["run_count"]),
                ]
            )
            + " |"
        )
    reasons: dict[str, int] = defaultdict(int)
    for row in excluded:
        reasons[row["exclusion_reason"]] += 1
    lines.extend(
        [
            "",
            "## Scope",
            "",
            (
                f"The table contains {len(summaries)} distinct authoritative "
                f"conditions derived from "
                f"{sum(row['run_count'] for row in summaries)} run-level "
                f"measurements. The exhaustive CSV also retains "
                f"{len(excluded)} excluded run-level measurements."
            ),
            "",
            "Excluded from the pooled table:",
            "",
        ]
    )
    lines.extend(f"- {reason}: {count}" for reason, count in sorted(reasons.items()))
    lines.extend(
        [
            "",
            (
                "Collective-only all-reduce/reduce microbenchmarks are not "
                "included because they do not measure an A2A overlap."
            ),
            "",
            "Regenerate from the canonical artifacts with:",
            "",
            "```bash",
            "python scripts/summarize_a2a_slowdowns.py",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("out/interference_benchmarks")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/interference_benchmarks/a2a_slowdown_summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = extract_all(args.root)
    rows.sort(key=lambda row: row["a2a_slowdown_ratio"], reverse=True)
    summaries = group_authoritative(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "all_measurements.csv",
        [public_row(row) for row in rows],
    )
    write_csv(args.output_dir / "distinct_conditions.csv", summaries)
    write_markdown(
        args.output_dir / "distinct_conditions.md", summaries, rows
    )
    print(
        f"Wrote {len(rows)} run-level measurements and "
        f"{len(summaries)} distinct authoritative conditions"
    )
    print(
        "Highest slowdown:",
        summaries[0]["a2a_slowdown_ratio"],
        summaries[0]["overlap"],
        summaries[0]["queue_order"],
        summaries[0]["topology"],
    )


if __name__ == "__main__":
    main()
