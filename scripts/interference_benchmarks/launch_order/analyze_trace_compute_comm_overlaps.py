#!/usr/bin/env python3
"""Find significant compute/NCCL kernel overlaps in PyTorch Chrome traces."""

from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Any


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def original_name(event: dict[str, Any]) -> str:
    return str((event.get("args") or {}).get("original_cuda_name", event["name"]))


def is_communication(event: dict[str, Any]) -> bool:
    return "ncclDevKernel" in original_name(event)


def event_record(
    event: dict[str, Any],
    trace: Path,
    cpu_ops_by_external_id: dict[int, set[str]],
) -> dict[str, Any]:
    args = event.get("args") or {}
    external_id = int(args.get("External id", -1))
    rank_match = re.search(r"dprank(\d+)", trace.name)
    return {
        "trace": trace.name,
        "dp_rank": int(rank_match.group(1)) if rank_match else None,
        "name": original_name(event),
        "decorated_name": event["name"],
        "start_us": float(event["ts"]),
        "duration_us": float(event["dur"]),
        "end_us": float(event["ts"]) + float(event["dur"]),
        "device": int(args.get("device", -1)),
        "context": int(args.get("context", -1)),
        "stream": int(args.get("stream", event.get("tid", -1))),
        "grid": [int(value) for value in args.get("grid", [])],
        "block": [int(value) for value in args.get("block", [])],
        "piper_dag_node": args.get("piper_dag_node"),
        "cpu_ops": sorted(cpu_ops_by_external_id.get(external_id, set())),
        "collective_name": args.get("Collective name"),
        "in_msg_nelems": args.get("In msg nelems"),
        "out_msg_nelems": args.get("Out msg nelems"),
        "group_size": args.get("Group size"),
        "dtype": args.get("dtype"),
    }


def case_key(compute: dict[str, Any], comm: dict[str, Any]) -> str:
    value = {
        "compute_name": compute["name"],
        "compute_grid": compute["grid"],
        "compute_block": compute["block"],
        "comm_name": comm["name"],
        "collective_name": comm["collective_name"],
        "in_msg_nelems": comm["in_msg_nelems"],
        "out_msg_nelems": comm["out_msg_nelems"],
        "group_size": comm["group_size"],
        "dtype": comm["dtype"],
        "comm_grid": comm["grid"],
        "comm_block": comm["block"],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def compute_family(name: str) -> str:
    """Collapse shape/ragged-grid variants while retaining kernel semantics."""
    lower = name.lower()
    if "groupproblemshape" in lower:
        if "majore0elsz_0" in lower:
            return "expert_grouped_mm_forward"
        if "majore1elsz_1" in lower:
            return "expert_grouped_mm_backward_input"
        return "expert_grouped_mm_backward_weight"
    if "sdpa" in lower and "bprop" in lower:
        return "sdpa_backward"
    if "sdpa" in lower and "fprop" in lower:
        return "sdpa_forward"
    if "sm80_xmma_gemm" in lower:
        return "router_weighted_output_bmm"
    if "nvjet" in lower and "coopb_ntn" in lower:
        return "vocab_projection_backward_input"
    if "nvjet" in lower and "coopa_nnt" in lower:
        return "vocab_projection_backward_weight"
    if "nvjet" in lower and "coopb_tnn" in lower:
        return "vocab_projection_forward"
    if "softmaxbackward" in lower:
        return "log_softmax_backward"
    if "softmaxforward" in lower:
        return "log_softmax_forward"
    if "fused_mul_silu_silu_backward" in lower:
        return "expert_swiglu_backward"
    if "fused__grouped_mm_mul_silu_transpose" in lower:
        return "expert_swiglu_forward"
    if "index_elementwise_kernel" in lower and "index_put" in lower:
        return "routed_index_put"
    if "vectorized_gather_kernel" in lower:
        return "routed_gather"
    if "catarraybatchedcopy" in lower:
        return "routed_cat"
    if "triton_poi_fused_add" in lower:
        return "compiled_elementwise_add"
    if "vectorized_elementwise_kernel" in lower and "functor_add" in lower:
        return "native_elementwise_add"
    if "direct_copy_kernel" in lower:
        return "attention_gradient_copy"
    return "other"


def collective_family(name: str | None) -> str:
    return {
        "_allgather_base": "all_gather",
        "_reduce_scatter_base": "reduce_scatter",
        "all_to_allv": "all_to_allv",
        "send": "send",
        "recv": "recv",
    }.get(str(name), str(name))


def benchmark_compute_class(family: str) -> str:
    return {
        "expert_grouped_mm_forward": "expert_grouped_gemm",
        "expert_grouped_mm_backward_input": "expert_grouped_gemm",
        "expert_grouped_mm_backward_weight": "expert_grouped_gemm",
        "sdpa_forward": "sdpa_forward",
        "sdpa_backward": "sdpa_backward",
        "vocab_projection_forward": "vocab_projection_gemm",
        "vocab_projection_backward_input": "vocab_projection_gemm",
        "vocab_projection_backward_weight": "vocab_projection_gemm",
        "log_softmax_forward": "log_softmax",
        "log_softmax_backward": "log_softmax",
        "router_weighted_output_bmm": "router_bmm",
        "expert_swiglu_forward": "expert_activation_pointwise",
        "expert_swiglu_backward": "expert_activation_pointwise",
        "routed_cat": "routing_permutation",
        "routed_gather": "routing_permutation",
        "routed_index_put": "routing_permutation",
        "compiled_elementwise_add": "elementwise_copy",
        "native_elementwise_add": "elementwise_copy",
        "attention_gradient_copy": "elementwise_copy",
    }[family]


def reduce_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[
            (
                compute_family(str(case["compute"]["name"])),
                collective_family(case["communication"]["collective_name"]),
            )
        ].append(case)
    reduced = []
    for (compute, collective), members in sorted(grouped.items()):
        representative = max(
            members,
            key=lambda item: (
                int(item["observation_count"]),
                float(item["overlap_us"]["median"]),
            ),
        )
        payloads = {
            (
                item["communication"]["in_msg_nelems"],
                item["communication"]["out_msg_nelems"],
                item["communication"]["group_size"],
                item["communication"]["dtype"],
            )
            for item in members
        }
        compute_variants = {
            (
                item["compute"]["name"],
                tuple(item["compute"]["grid"]),
                tuple(item["compute"]["block"]),
            )
            for item in members
        }
        reduced.append(
            {
                "reduced_case_id": f"reduced_{len(reduced):03d}",
                "compute_family": compute,
                "collective_family": collective,
                "exact_case_count": len(members),
                "observation_count": sum(
                    int(item["observation_count"]) for item in members
                ),
                "compute_variant_count": len(compute_variants),
                "payload_variant_count": len(payloads),
                "representative_case_id": representative["case_id"],
                "representative_compute": representative["compute"],
                "representative_communication": representative["communication"],
                "payload_variants": [
                    {
                        "in_msg_nelems": payload[0],
                        "out_msg_nelems": payload[1],
                        "group_size": payload[2],
                        "dtype": payload[3],
                    }
                    for payload in sorted(payloads, key=str)
                ],
                "exact_case_ids": [item["case_id"] for item in members],
            }
        )
    return reduced


def benchmark_reduce_cases(
    family_cases: list[dict[str, Any]], exact_cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    exact_by_id = {case["case_id"]: case for case in exact_cases}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in family_cases:
        grouped[
            (
                benchmark_compute_class(case["compute_family"]),
                case["collective_family"],
            )
        ].append(case)
    result = []
    for (compute_class, collective), members in sorted(grouped.items()):
        exact_members = [
            exact_by_id[case_id]
            for member in members
            for case_id in member["exact_case_ids"]
        ]
        representative = max(
            exact_members,
            key=lambda item: (
                int(item["observation_count"]),
                float(item["overlap_us"]["median"]),
            ),
        )
        result.append(
            {
                "benchmark_case_id": f"benchmark_{len(result):03d}",
                "compute_class": compute_class,
                "collective_family": collective,
                "observation_count": sum(
                    int(item["observation_count"]) for item in exact_members
                ),
                "family_case_count": len(members),
                "exact_case_count": len(exact_members),
                "representative_case_id": representative["case_id"],
                "representative_compute": representative["compute"],
                "representative_communication": representative["communication"],
                "source_compute_families": sorted(
                    member["compute_family"] for member in members
                ),
            }
        )
    return result


def analyze_trace(
    path: Path, threshold: float, min_kernel_us: float
) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = raw.get("traceEvents", raw) if isinstance(raw, dict) else raw
    cpu_ops_by_external_id: dict[int, set[str]] = defaultdict(set)
    for event in events:
        if event.get("cat") != "cpu_op":
            continue
        external_id = int((event.get("args") or {}).get("External id", -1))
        if external_id >= 0:
            cpu_ops_by_external_id[external_id].add(str(event.get("name")))
    kernels = [
        event_record(event, path, cpu_ops_by_external_id)
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "kernel"
        and float(event.get("dur", 0)) > 0
    ]
    compute_by_device: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    comms: list[dict[str, Any]] = []
    for event in kernels:
        if is_communication(event):
            comms.append(event)
        else:
            compute_by_device[(event["device"], event["context"])].append(event)
    for events_for_device in compute_by_device.values():
        events_for_device.sort(key=lambda event: event["start_us"])

    overlaps: list[dict[str, Any]] = []
    for comm in comms:
        if comm["duration_us"] <= min_kernel_us:
            continue
        candidates = compute_by_device[(comm["device"], comm["context"])]
        starts = [event["start_us"] for event in candidates]
        stop = bisect_left(starts, comm["end_us"])
        for compute in candidates[:stop]:
            if compute["duration_us"] <= min_kernel_us:
                continue
            if compute["end_us"] <= comm["start_us"]:
                continue
            overlap_us = min(compute["end_us"], comm["end_us"]) - max(
                compute["start_us"], comm["start_us"]
            )
            compute_fraction = overlap_us / compute["duration_us"]
            comm_fraction = overlap_us / comm["duration_us"]
            if max(compute_fraction, comm_fraction) <= threshold:
                continue
            if compute["start_us"] < comm["start_us"]:
                order = "compute_first"
            elif comm["start_us"] < compute["start_us"]:
                order = "comm_first"
            else:
                order = "simultaneous"
            overlaps.append(
                {
                    "case_key": case_key(compute, comm),
                    "compute": compute,
                    "comm": comm,
                    "overlap_us": overlap_us,
                    "compute_overlap_fraction": compute_fraction,
                    "comm_overlap_fraction": comm_fraction,
                    "launch_order": order,
                    "launch_delta_us": comm["start_us"] - compute["start_us"],
                }
            )
    return overlaps


def aggregate(
    overlaps: list[dict[str, Any]], threshold: float, min_kernel_us: float
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for overlap in overlaps:
        grouped[overlap["case_key"]].append(overlap)
    cases = []
    for key, observations in grouped.items():
        exemplar = observations[0]
        compute = exemplar["compute"]
        comm = exemplar["comm"]
        cases.append(
            {
                "case_id": f"case_{len(cases):03d}",
                "observation_count": len(observations),
                "rank_count": len({item["compute"]["dp_rank"] for item in observations}),
                "compute": {
                    "name": compute["name"],
                    "grid": compute["grid"],
                    "block": compute["block"],
                    "cpu_ops": sorted(
                        {
                            name
                            for item in observations
                            for name in item["compute"]["cpu_ops"]
                        }
                    ),
                    "duration_us": summary(
                        [item["compute"]["duration_us"] for item in observations]
                    ),
                    "piper_dag_nodes": sorted(
                        {
                            str(item["compute"]["piper_dag_node"])
                            for item in observations
                            if item["compute"]["piper_dag_node"]
                        }
                    ),
                },
                "communication": {
                    "name": comm["name"],
                    "collective_name": comm["collective_name"],
                    "in_msg_nelems": comm["in_msg_nelems"],
                    "out_msg_nelems": comm["out_msg_nelems"],
                    "group_size": comm["group_size"],
                    "dtype": comm["dtype"],
                    "grid": comm["grid"],
                    "block": comm["block"],
                    "duration_us": summary(
                        [item["comm"]["duration_us"] for item in observations]
                    ),
                    "piper_dag_nodes": sorted(
                        {
                            str(item["comm"]["piper_dag_node"])
                            for item in observations
                            if item["comm"]["piper_dag_node"]
                        }
                    ),
                },
                "overlap_us": summary([item["overlap_us"] for item in observations]),
                "compute_overlap_fraction": summary(
                    [item["compute_overlap_fraction"] for item in observations]
                ),
                "comm_overlap_fraction": summary(
                    [item["comm_overlap_fraction"] for item in observations]
                ),
                "launch_orders": dict(
                    Counter(item["launch_order"] for item in observations)
                ),
                "launch_delta_us": summary(
                    [item["launch_delta_us"] for item in observations]
                ),
                "dp_ranks": sorted(
                    {int(item["compute"]["dp_rank"]) for item in observations}
                ),
                "case_key": key,
            }
        )
    cases.sort(
        key=lambda case: (
            -int(case["rank_count"]),
            -int(case["observation_count"]),
            str(case["compute"]["name"]),
        )
    )
    for index, case in enumerate(cases):
        case["case_id"] = f"case_{index:03d}"
    reduced = reduce_cases(cases)
    benchmark_cases = benchmark_reduce_cases(reduced, cases)
    return {
        "threshold_strictly_greater_than": threshold,
        "minimum_kernel_duration_us_strictly_greater_than": min_kernel_us,
        "overlap_observation_count": len(overlaps),
        "unique_case_count": len(cases),
        "cases": cases,
        "reduction": {
            "key": "compute kernel family x collective kind",
            "ignored_differences": [
                "communication payload length",
                "ragged-token compute grid size",
                "rank and Piper segment repetition",
            ],
            "reduced_case_count": len(reduced),
            "cases": reduced,
        },
        "benchmark_reduction": {
            "key": "resource-similar compute class x collective kind",
            "compute_class_count": len(
                {case["compute_class"] for case in benchmark_cases}
            ),
            "communication_kind_count": len(
                {case["collective_family"] for case in benchmark_cases}
            ),
            "benchmark_case_count": len(benchmark_cases),
            "cases": benchmark_cases,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--min-kernel-us",
        type=float,
        default=0.0,
        help="Require both constituent kernel events to be longer than this value.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.trace_dir.glob("pytorch_profile_*.json"))
    if not paths:
        raise FileNotFoundError(f"no profiler traces under {args.trace_dir}")
    overlaps = []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        overlaps.extend(analyze_trace(path, args.threshold, args.min_kernel_us))
    result = aggregate(overlaps, args.threshold, args.min_kernel_us)
    result["trace_count"] = len(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: {result['overlap_observation_count']} observations, "
        f"{result['unique_case_count']} cases"
    )


if __name__ == "__main__":
    main()
