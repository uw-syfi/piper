#!/usr/bin/env python3
"""Replace Figure 1's wall-average NVML SM column with peak footprint estimates.

The estimate is derived from the exact primary kernel launch in the original
12K Qwen trace.  It is intentionally a residency/footprint estimate, not an
instruction-throughput measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CHECKOUT_ROOT))

from scripts.partial_overlap_interference_benchmarks.pairwise_overlap_benchmark import (  # noqa: E402
    MODEL_CONFIG,
    MODEL_NAME,
    TRACE_SEQUENCE_LENGTH,
)


KERNELS = (
    {
        "victim": "Grouped GEMM",
        "signature_id": "ISO-A03-GROUPED-COMPUTE",
        "case_id": "A03",
        "variant": "C-GROUPED",
        "category": "compute",
        "event_id": "d2.p1.i3.e00895",
    },
    {
        "victim": "P2P send",
        "signature_id": "ISO-A01-GROUPED-P2P-SEND",
        "case_id": "A01",
        "variant": "C-GROUPED",
        "category": "p2p_send",
        "event_id": "d6.p1.i3.e00313",
    },
    {
        "victim": "A2A",
        "signature_id": "ISO-A03-GROUPED-A2A",
        "case_id": "A03",
        "variant": "C-GROUPED",
        "category": "a2a",
        "event_id": "d2.p1.i3.e00876",
    },
    {
        "victim": "AG",
        "signature_id": "ISO-A04-GROUPED-AG",
        "case_id": "A04",
        "variant": "C-GROUPED",
        "category": "ag",
        "event_id": "d5.p1.i2.e00850",
    },
    {
        "victim": "RS",
        "signature_id": "ISO-A05-GROUPED-RS",
        "case_id": "A05",
        "variant": "C-GROUPED",
        "category": "rs",
        "event_id": "d4.p0.i2.e01559",
    },
)


LINK_TOPOLOGY_BY_SIGNATURE = {
    "ISO-A03-GROUPED-COMPUTE": "none (single GPU)",
    "ISO-A01-GROUPED-P2P-SEND": (
        "PCIe + InfiniBand GPUDirect RDMA (cross-node)"
    ),
    "ISO-A03-GROUPED-A2A": "NVLink 4/NVSwitch (intra-node, 8 GPU)",
    "ISO-A04-GROUPED-AG": "NVLink 4/NVSwitch (intra-node, 8 GPU)",
    "ISO-A05-GROUPED-RS": "NVLink 4/NVSwitch (intra-node, 8 GPU)",
}


def parse_args() -> argparse.Namespace:
    # The simulator worktree is commonly nested inside the persistent Piper
    # workspace while the collected traces live in the parent's out/ tree.
    # A standalone checkout keeps those artifacts under its own root instead.
    data_root = (
        _CHECKOUT_ROOT
        if (_CHECKOUT_ROOT / "out/interference-benchmarks").is_dir()
        else _CHECKOUT_ROOT.parent
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resource-dir",
        type=Path,
        default=data_root
        / "out/interference-benchmarks/physical-resource-diagnostics/"
        "figure1-isolated-3w10m-20260811",
    )
    parser.add_argument(
        "--trace-dir", type=Path, default=data_root / "out/20260809_132048"
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=data_root
        / "out/interference-benchmarks/methodology/EXACT_KERNEL_EXECUTION_MATRIX.json",
    )
    parser.add_argument("--sm-count", type=int, default=132)
    parser.add_argument("--max-threads-per-sm", type=int, default=2048)
    parser.add_argument("--registers-per-sm", type=int, default=65536)
    parser.add_argument("--shared-memory-per-sm", type=int, default=233472)
    parser.add_argument("--warp-size", type=int, default=32)
    parser.add_argument("--max-blocks-per-sm", type=int, default=32)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def payload_mib(elements: int, dtype: str) -> float:
    bytes_per_element = {"BFloat16": 2, "Float": 4}[dtype]
    return elements * bytes_per_element / (1024.0**2)


def operation_shape(
    item: dict[str, str], participant: dict[str, Any]
) -> tuple[str, str]:
    """Return a plain-text and LaTeX description of the exact operation shape."""

    signature = participant["signature"]
    if item["category"] == "compute":
        groups = MODEL_CONFIG["num_experts"] // 8
        rows_per_group = 2 * MODEL_CONFIG["moe_inter_dim"]
        reduction = (
            8
            * TRACE_SEQUENCE_LENGTH
            * MODEL_CONFIG["top_k"]
            // MODEL_CONFIG["num_experts"]
        )
        dim = MODEL_CONFIG["dim"]
        plain = (
            f"{groups} groups: ({rows_per_group}x{reduction})"
            f"x({reduction}x{dim}) BF16"
        )
        latex = (
            f"{groups} groups: $({rows_per_group}\\times{reduction})"
            f"\\times({reduction}\\times{dim})$ BF16"
        )
        return plain, latex

    dtype = str(signature["dtype"])
    dtype_label = {"BFloat16": "BF16", "Float": "FP32"}[dtype]
    input_mib = payload_mib(int(signature["in_msg_nelems"]), dtype)
    output_mib = payload_mib(int(signature["out_msg_nelems"]), dtype)
    if item["category"] == "p2p_send":
        plain = f"{input_mib:.0f} MiB {dtype_label} send"
        latex = f"{input_mib:.0f} MiB {dtype_label} send"
    else:
        precision = 2 if not input_mib.is_integer() or not output_mib.is_integer() else 0
        plain = (
            f"{input_mib:.{precision}f}->{output_mib:.{precision}f} MiB "
            f"{dtype_label}/rank"
        )
        latex = (
            f"{input_mib:.{precision}f}$\\rightarrow$"
            f"{output_mib:.{precision}f} MiB {dtype_label}/rank"
        )
        if item["category"] == "a2a":
            plain += " (ragged anchor rank)"
            latex += " (ragged anchor rank)"
    return plain, latex


def find_matrix_participant(
    matrix: dict[str, Any], case_id: str, variant: str, category: str
) -> dict[str, Any]:
    matches = [
        participant
        for assignment in matrix["pair_assignments"]
        if assignment["case_id"] == case_id and assignment.get("variant") == variant
        for participant in assignment["participants"]
        if participant["category"] == category
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one matrix participant for {case_id}:{variant}:{category}, "
            f"found {len(matches)}"
        )
    return matches[0]


def trace_kernel(
    trace_dir: Path, event_row: dict[str, str]
) -> tuple[Path, dict[str, Any]]:
    rank = int(event_row["dp_rank"])
    trace_path = trace_dir / f"pytorch_profile_dualpipev_pp2_dp8_mbs4_dprank{rank}.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    correlation = int(event_row["correlation"])
    matches = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "kernel"
        and int((event.get("args") or {}).get("correlation", -1)) == correlation
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one trace kernel for {event_row['event_id']} correlation "
            f"{correlation}, found {len(matches)}"
        )
    return trace_path, matches[0]


def main() -> None:
    args = parse_args()
    resource_dir = args.resource_dir.resolve()
    trace_dir = args.trace_dir.resolve()
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if int(matrix["anchor_sequence_length"]) != 12288:
        raise ValueError("kernel matrix is not the 12K anchor")

    event_rows = {
        row["event_id"]: row
        for row in read_csv(trace_dir / "kernel_overlap_events.csv")
        if row["event_id"] in {item["event_id"] for item in KERNELS}
    }
    if set(event_rows) != {item["event_id"] for item in KERNELS}:
        raise ValueError("not all exact Figure 1 events were found")

    max_warps_per_sm = args.max_threads_per_sm // args.warp_size
    geometry_rows: list[dict[str, Any]] = []
    for item in KERNELS:
        participant = find_matrix_participant(
            matrix, item["case_id"], item["variant"], item["category"]
        )
        shape_plain, shape_latex = operation_shape(item, participant)
        event_row = event_rows[item["event_id"]]
        trace_path, event = trace_kernel(trace_dir, event_row)
        trace_args = event["args"]
        original_name = trace_args.get("original_cuda_name", event["name"])
        signature = participant["signature"]
        if original_name != participant["kernel_name"]:
            raise ValueError(f"{item['victim']}: trace/matrix kernel-name mismatch")
        if list(trace_args["grid"]) != list(signature["grid"]):
            raise ValueError(f"{item['victim']}: trace/matrix grid mismatch")
        if list(trace_args["block"]) != list(signature["block"]):
            raise ValueError(f"{item['victim']}: trace/matrix block mismatch")
        kernel_sha1 = hashlib.sha1(original_name.encode("utf-8")).hexdigest()
        if kernel_sha1 != participant["kernel_sha1"]:
            raise ValueError(f"{item['victim']}: kernel SHA-1 mismatch")

        grid_blocks = product(trace_args["grid"])
        threads_per_block = product(trace_args["block"])
        warps_per_block = math.ceil(threads_per_block / args.warp_size)
        registers_per_thread = int(trace_args["registers per thread"])
        registers_per_block = registers_per_thread * threads_per_block
        shared_memory_per_block = int(trace_args["shared memory"])
        limits = {
            "threads": args.max_threads_per_sm // threads_per_block,
            "registers": args.registers_per_sm // registers_per_block,
            "shared_memory": (
                args.shared_memory_per_sm // shared_memory_per_block
                if shared_memory_per_block
                else args.max_blocks_per_sm
            ),
            "architecture": args.max_blocks_per_sm,
        }
        max_resident_blocks_per_sm = min(limits.values())
        if max_resident_blocks_per_sm < 1:
            raise ValueError(f"{item['victim']}: calculated an impossible occupancy")
        limiting_resources = "+".join(
            key for key, value in limits.items() if value == max_resident_blocks_per_sm
        )
        resident_blocks = min(
            grid_blocks, args.sm_count * max_resident_blocks_per_sm
        )
        # Every selected kernel is one-block-per-SM limited. Therefore, if the
        # initial resident wave fills (the explicit assumption), resident blocks
        # must occupy distinct SMs; no scheduler-spreading heuristic is needed.
        if max_resident_blocks_per_sm != 1:
            raise ValueError(
                f"{item['victim']}: method requires one-block-per-SM kernels"
            )
        estimated_peak_sms = min(args.sm_count, resident_blocks)
        peak_footprint_pct = 100.0 * estimated_peak_sms / args.sm_count
        resident_warp_occupancy_pct = 100.0 * (
            resident_blocks * warps_per_block
        ) / (args.sm_count * max_warps_per_sm)
        geometry_rows.append(
            {
                "victim": item["victim"],
                "signature_id": item["signature_id"],
                "case_key": f"{item['case_id']}:{item['variant']}",
                "event_id": item["event_id"],
                "kernel_alias": participant["kernel_alias"],
                "kernel_sha1": kernel_sha1,
                "kernel_name": original_name,
                "operation_shape_or_payload": shape_plain,
                "operation_shape_or_payload_latex": shape_latex,
                "grid_x": int(trace_args["grid"][0]),
                "grid_y": int(trace_args["grid"][1]),
                "grid_z": int(trace_args["grid"][2]),
                "grid_blocks": grid_blocks,
                "block_x": int(trace_args["block"][0]),
                "block_y": int(trace_args["block"][1]),
                "block_z": int(trace_args["block"][2]),
                "threads_per_block": threads_per_block,
                "warps_per_block": warps_per_block,
                "registers_per_thread": registers_per_thread,
                "registers_per_block": registers_per_block,
                "shared_memory_per_block_bytes": shared_memory_per_block,
                "blocks_per_sm_limit_threads": limits["threads"],
                "blocks_per_sm_limit_registers": limits["registers"],
                "blocks_per_sm_limit_shared_memory": limits["shared_memory"],
                "max_resident_blocks_per_sm_estimate": max_resident_blocks_per_sm,
                "limiting_resources": limiting_resources,
                "gpu_sm_count": args.sm_count,
                "estimated_peak_resident_blocks": resident_blocks,
                "estimated_peak_sms": estimated_peak_sms,
                "estimated_peak_sm_footprint_pct": peak_footprint_pct,
                "estimated_peak_resident_warp_occupancy_pct": resident_warp_occupancy_pct,
                "assumption": "full initial resident wave; one block maximum per SM",
                "trace_path": str(trace_path),
                "trace_correlation": int(event_row["correlation"]),
            }
        )

    geometry_path = resource_dir / "figure1_kernel_launch_geometry.csv"
    write_csv(geometry_path, geometry_rows)

    summary_path = resource_dir / "resource_metrics_summary.csv"
    summary_backup = resource_dir / "resource_metrics_summary_nvml_sm_original.csv"
    if not summary_backup.exists():
        original_rows = read_csv(summary_path)
        if not any(key.startswith("sm_util_pct_") for key in original_rows[0]):
            raise ValueError("cannot archive original summary: NVML SM fields are absent")
        shutil.copyfile(summary_path, summary_backup)
    original_rows = read_csv(summary_backup)
    geometry_by_signature = {row["signature_id"]: row for row in geometry_rows}
    updated_rows: list[dict[str, Any]] = []
    for original in original_rows:
        geometry = geometry_by_signature[original["signature_id"]]
        updated: dict[str, Any] = {}
        for key, value in original.items():
            if key.startswith("sm_util_pct_"):
                continue
            updated[key] = (
                LINK_TOPOLOGY_BY_SIGNATURE[original["signature_id"]]
                if key == "topology"
                else value
            )
            if key == "measured_trials":
                updated.update(
                    {
                        "estimated_peak_sm_footprint_pct": geometry[
                            "estimated_peak_sm_footprint_pct"
                        ],
                        "estimated_peak_sms": geometry["estimated_peak_sms"],
                        "gpu_sm_count": geometry["gpu_sm_count"],
                        "grid_blocks": geometry["grid_blocks"],
                        "threads_per_block": geometry["threads_per_block"],
                        "warps_per_block": geometry["warps_per_block"],
                        "max_resident_blocks_per_sm_estimate": geometry[
                            "max_resident_blocks_per_sm_estimate"
                        ],
                        "limiting_resources": geometry["limiting_resources"],
                        "operation_shape_or_payload": geometry[
                            "operation_shape_or_payload"
                        ],
                    }
                )
        updated_rows.append(updated)
    write_csv(summary_path, updated_rows)

    report_path = resource_dir / "RESOURCE_METRICS_REPORT.md"
    report_backup = resource_dir / "RESOURCE_METRICS_REPORT_NVML_SM_ORIGINAL.md"
    if not report_backup.exists():
        shutil.copyfile(report_path, report_backup)

    def interval(row: dict[str, Any], metric: str, digits: int) -> str:
        return (
            f"{float(row[f'{metric}_median']):.{digits}f} "
            f"[{float(row[f'{metric}_p25']):.{digits}f}, "
            f"{float(row[f'{metric}_p75']):.{digits}f}]"
        )

    lines = [
        "# Figure 1 isolated resource metrics",
        "",
        "Peak SM footprint is a static launch/residency estimate for each exact",
        "primary Figure 1 kernel. Other entries are medians across 10 measured",
        "trials; brackets contain the 25th and 75th percentiles. Each measured",
        "trial first takes the median across active GPU endpoints. The run used",
        "exactly 3 warmups.",
        "",
        "| Operation | Topology | Estimated peak SM footprint (%) | DRAM BW util (%) | NVLink RX+TX (GiB/s) | PCIe RX+TX (GiB/s) | IB TX+RX per endpoint (GiB/s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in updated_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["victim"],
                    row["topology"],
                    f"{float(row['estimated_peak_sm_footprint_pct']):.2f}",
                    interval(row, "dram_bw_util_pct", 2),
                    interval(row, "nvlink_total_gib_per_s", 2),
                    interval(row, "pcie_total_gib_per_s", 3),
                    interval(row, "ib_endpoint_gib_per_s", 3),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Peak-SM-footprint method",
            "",
            "- The exact primary kernels, grids, blocks, registers, and shared-memory values come from the original 12,288-token PyTorch traces indexed by `EXACT_KERNEL_EXECUTION_MATRIX.json`.",
            f"- Hardware constants are H100: {args.sm_count} SMs, {args.max_threads_per_sm} threads/SM, {args.registers_per_sm} registers/SM, {args.shared_memory_per_sm} bytes shared memory/SM, and warp size {args.warp_size}.",
            "- All five kernels are limited to one resident block per SM. Assuming the initial resident wave fills, each resident block must occupy a distinct SM, so `peak footprint = min(grid blocks, 132) / 132`.",
            "- This estimates the maximum spatial SM footprint, not active-cycle utilization, instruction throughput, or the fraction of each SM's execution pipelines consumed.",
            "- Full kernel names and occupancy calculations are in `figure1_kernel_launch_geometry.csv`.",
            "",
            "## Measured-metric semantics",
            "",
            "- DRAM is NVML GPM ID 10.",
            "- NVLink is the sum of NVML GPM RX and TX rate IDs 60 and 61.",
            "- PCIe is the sum of NVML GPM RX and TX rate IDs 20 and 21.",
            "- IB is derived from `/sys/class/infiniband/*/ports/*/counters/port_{xmit,rcv}_data`; all HCA rates are summed per node, then the two node endpoint rates are median-reduced.",
            "- NVML values are GPU-wide wall-interval averages over repeated exact isolated operations, with no duty-cycle correction.",
            "- The removed NVML SM averages remain preserved in `resource_metrics_trials.csv`, `resource_metrics_summary_nvml_sm_original.csv`, and `RESOURCE_METRICS_REPORT_NVML_SM_ORIGINAL.md`.",
            "- P2P is a two-rank cross-node operation. A2A, AG, and RS are eight-rank collectives contained within one node.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    def tex_escape(value: str) -> str:
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        return "".join(replacements.get(character, character) for character in value)

    def tex_interval(row: dict[str, Any], metric: str, digits: int) -> str:
        median = float(row[f"{metric}_median"])
        p25 = float(row[f"{metric}_p25"])
        p75 = float(row[f"{metric}_p75"])
        if median == 0.0:
            if p25 != 0.0 or p75 != 0.0:
                raise ValueError(
                    f"cannot express a nonzero IQR relative to a zero median: "
                    f"{row['signature_id']} {metric}"
                )
            relative_half_iqr_pct = 0.0
        else:
            relative_half_iqr_pct = 100.0 * (p75 - p25) / (2.0 * abs(median))
        return (
            f"{median:.{digits}f} "
            rf"$\pm$ {relative_half_iqr_pct:.2f}\%"
        )

    latex_path = resource_dir / "RESOURCE_METRICS_TABLE.tex"
    latex_lines = [
        r"% Requires \usepackage{booktabs} and \usepackage{graphicx}.",
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lllrrrrr}",
        r"    \toprule",
        r"    Operation & Shape / payload & Topology & Peak SM footprint (\%) & DRAM BW (\%) & NVLink (GiB/s) & PCIe (GiB/s) & IB/endpoint (GiB/s) \\",
        r"    \midrule",
    ]
    for row in updated_rows:
        latex_lines.append(
            "    "
            + " & ".join(
                [
                    tex_escape(row["victim"]),
                    geometry_by_signature[row["signature_id"]][
                        "operation_shape_or_payload_latex"
                    ],
                    tex_escape(row["topology"]),
                    f"{float(row['estimated_peak_sm_footprint_pct']):.2f}",
                    tex_interval(row, "dram_bw_util_pct", 2),
                    tex_interval(row, "nvlink_total_gib_per_s", 2),
                    tex_interval(row, "pcie_total_gib_per_s", 3),
                    tex_interval(row, "ib_endpoint_gib_per_s", 3),
                ]
            )
            + r" \\"
        )
    latex_lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            rf"  \caption{{Isolated resource characteristics of operations modeled after the {MODEL_NAME} trace at sequence length {TRACE_SEQUENCE_LENGTH:,}, batch size {MODEL_CONFIG['batch_size']}, and four microbatches. The traced configuration uses four model layers, BF16, hidden size {MODEL_CONFIG['dim']:,}, expert width {MODEL_CONFIG['moe_inter_dim']:,}, {MODEL_CONFIG['num_experts']} experts with top-{MODEL_CONFIG['top_k']} routing, and DualPipeV over four logical pipeline stages mapped to two physical PP ranks (PP-outer), with DP$=$EP$=8 across two eight-H100 nodes; parameters and gradients are sharded without bucketing. Compute shape gives the {MODEL_CONFIG['num_experts'] // 8} grouped-GEMM problems, while communication shape gives the anchor rank's input$\rightarrow$output tensor payload (A2A is ragged, so other ranks differ). The topology column names the actual data path: cross-node P2P uses PCIe plus InfiniBand GPUDirect RDMA, while the eight-GPU intra-node collectives use NVLink~4 through NVSwitch. Peak SM footprint is a static residency estimate derived from the exact PyTorch launches: all five primary kernels are limited to one resident block per SM by register allocation (and grouped GEMM also by shared memory), so, assuming the initial resident wave fills, it is $100\,\min(G,132)/132$, where $G$ is the grid block count and 132 is the H100 SM count. Other entries use $X\mathbin{{\pm}}Y\%$ over 10 trials after 3 warmups, where $X$ is the median and $Y=100(Q_3-Q_1)/(2|X|)$ is the relative half-interquartile range; an exactly zero median and IQR are reported as $0\mathbin{{\pm}}0\%$. DRAM bandwidth is NVML GPM metric~10; NVLink and PCIe are sums of NVML GPM receive and transmit rates; and IB is the per-node sum of transmit and receive byte-rate deltas from InfiniBand sysfs port counters, median-reduced across the two endpoints.}}",
            r"  \label{tab:figure1-isolated-resource-characteristics}",
            r"\end{table*}",
            "",
        ]
    )
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    validation_path = resource_dir / "resource_metrics_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["peak_sm_footprint"] = {
        "status": "pass",
        "source_sequence_length": 12288,
        "kernel_count": len(geometry_rows),
        "all_exact_event_ids_resolved": True,
        "all_trace_matrix_signatures_match": True,
        "all_kernel_sha1_match": True,
        "all_one_resident_block_per_sm_limited": True,
        "table_sm_column": "estimated_peak_sm_footprint_pct",
        "removed_table_column": "NVML GPM wall-average sm_util_pct",
        "geometry_csv": str(geometry_path),
        "matrix": str(matrix_path),
    }
    validation_path.write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )

    method_path = resource_dir / "PEAK_SM_FOOTPRINT_METHOD.md"
    method_path.write_text(
        "\n".join(
            [
                "# Figure 1 peak SM footprint method",
                "",
                "The table's peak SM column is derived from exact launch geometry,",
                "not NVML. All five selected primary kernels are one-block-per-SM",
                "limited, so a full initial resident wave cannot pack two selected",
                "blocks onto one SM.",
                "",
                "For each kernel:",
                "",
                "```text",
                "resident_blocks = min(grid_blocks, SM_count * max_blocks_per_SM)",
                "peak_SMs = min(SM_count, resident_blocks)  # max_blocks_per_SM == 1",
                "peak_footprint_pct = 100 * peak_SMs / SM_count",
                "```",
                "",
                "This is a peak residency-footprint estimate. It does not claim that",
                "every resident warp issues instructions continuously.",
                "",
                f"Exact per-kernel evidence: `{geometry_path.name}`.",
                f"Exact 12K matrix: `{matrix_path}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(report_path)
    print(latex_path)
    print(summary_path)
    print(geometry_path)


if __name__ == "__main__":
    main()
