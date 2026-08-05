#!/usr/bin/env python3
"""Predict grouped-GEMM/all-reduce overlap from launch order and SM admission.

This standalone benchmark profiles the two operations in isolation, infers
their SM demand from CUDA grid CTA counts, and predicts both launch orders.
The first operation keeps its unrestricted isolated runtime.  The second is
remeasured in isolation with only ``total_sms - first_sm_demand`` SMs.  If no
SMs remain, the prediction is the sum of both unrestricted isolated runtimes.

No proportional SM allocation is used.  Unrestricted concurrent executions
for both launch orders validate the predictions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import benchmark_grouped_gemm_allreduce_sm_partition as bench


OPERATIONS = ("compute", "comm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    bench.add_workload_arguments(parser)
    parser.add_argument(
        "--output",
        default=(
            "out/grouped_gemm_allreduce_sm_partition/"
            "launch_order_predictor/results.json"
        ),
    )
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--managed-launch-order",
        choices=["compute_first", "comm_first"],
        default="compute_first",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def ensure_cuda_compat_first() -> None:
    compat = "/usr/local/cuda-12.8/compat"
    if not Path(compat, "libcuda.so.1").exists():
        return
    current = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    if current and current[0] == compat:
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        [compat, *(path for path in current if path != compat)]
    )
    os.execvpe(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def restriction_spec(
    total_sms: int, second: str, remaining_sms: int
) -> tuple[str, int]:
    """Describe a complementary split that assigns `remaining_sms` to second."""
    if not 0 < remaining_sms < total_sms:
        raise ValueError(f"invalid remaining SM count: {remaining_sms}")
    other = "comm" if second == "compute" else "compute"
    complement = total_sms - remaining_sms
    if remaining_sms <= complement:
        return second, remaining_sms
    return other, complement


def render_report(result: dict[str, Any]) -> str:
    stats = result["isolated"]
    rows = []
    for order_name in ("compute_first", "comm_first"):
        order = result["orders"][order_name]
        first = order["first"]
        second = order["second"]
        second_runtime = (
            f"{order['second_modeled_ms']:.3f} ms"
            if order["remaining_sms"]
            else "serialized"
        )
        rows.append(
            f"| {first} -> {second} | {order['first_isolated_ms']:.3f} ms | "
            f"{order['remaining_sms']} | {second_runtime} | "
            f"{order['predicted_ms']:.3f} ms | {order['actual_median_ms']:.3f} ms | "
            f"{order['actual_p95_ms']:.3f} ms | "
            f"{order['error_percent']:+.1f}% |"
        )
    return "\n".join(
        [
            "# Launch-order interference predictor",
            "",
            f"- Device: {result['device']['name']} ({result['device']['sm_count']} SMs)",
            f"- Samples: {result['warmups']} warmups and {result['iterations']} measurements",
            f"- Compute isolated: {stats['compute']['runtime_ms']:.3f} ms, "
            f"SM demand {stats['compute']['sm_demand']}",
            f"- Communication isolated: {stats['comm']['runtime_ms']:.3f} ms, "
            f"SM demand {stats['comm']['sm_demand']}",
            "",
            "| Launch order | First isolated | SMs left for second | Second modeled | Prediction | Actual median | Actual p95 | Error |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "If the first operation consumes all SMs, prediction is the sum of the two "
            "isolated runtimes. Otherwise prediction is the maximum of the first "
            "isolated runtime and the second operation's restricted-isolated runtime.",
            "",
        ]
    )


def main() -> None:
    ensure_cuda_compat_first()
    args = parse_args()
    bench.validate_args(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="launch-order-predictor-", dir=output.parent))
    traces = run_dir / "traces"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(
        str(index) for index in range(args.world_size)
    )
    environment.setdefault("NCCL_CUMEM_ENABLE", "0")
    environment.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")

    try:
        print("[1/4] Profiling unrestricted isolated operations", flush=True)
        baseline_path = run_dir / "baseline.json"
        bench.run_torchrun_phase(
            args, "baseline", baseline_path, traces, environment
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        total_sms = int(baseline["device"]["sm_count"])
        profiles = {
            operation: bench.parse_profile(
                traces / f"{operation}.json", total_sms, operation
            )
            for operation in OPERATIONS
        }
        isolated = {
            "compute": {
                "runtime_ms": float(
                    baseline["isolated_compute"]["summary_ms"]["median"]
                ),
                "summary_ms": baseline["isolated_compute"]["summary_ms"],
                "sm_demand": int(profiles["compute"]["peak_inferred_sm_demand"]),
            },
            "comm": {
                "runtime_ms": float(
                    baseline["isolated_comm"]["summary_ms"]["median"]
                ),
                "summary_ms": baseline["isolated_comm"]["summary_ms"],
                "sm_demand": int(profiles["comm"]["peak_inferred_sm_demand"]),
            },
        }

        print("[2/4] Measuring second kernels with launch-order SM remainders", flush=True)
        orders: dict[str, dict[str, Any]] = {}
        restricted_results: dict[str, dict[str, Any]] = {}
        for first, second in (("compute", "comm"), ("comm", "compute")):
            order_name = f"{first}_first"
            remaining = max(0, total_sms - int(isolated[first]["sm_demand"]))
            if remaining == 0:
                second_modeled_ms = float(isolated[second]["runtime_ms"])
                predicted_ms = (
                    float(isolated[first]["runtime_ms"])
                    + float(isolated[second]["runtime_ms"])
                )
                restriction = None
                prediction_source = "serialized unrestricted isolated runtimes"
            else:
                partition_side, partition_target = restriction_spec(
                    total_sms, second, remaining
                )
                restricted_path = run_dir / f"restricted_{second}_{remaining}.json"
                bench.run_torchrun_phase(
                    args,
                    f"restricted_{second}",
                    restricted_path,
                    traces,
                    environment,
                    green_partition_target_sms=partition_target,
                    green_partition_side=partition_side,
                )
                restricted = json.loads(
                    restricted_path.read_text(encoding="utf-8")
                )
                actual_restricted_sms = int(restricted["allocated_sm_count"])
                second_modeled_ms = float(restricted["summary_ms"]["median"])
                predicted_ms = max(
                    float(isolated[first]["runtime_ms"]), second_modeled_ms
                )
                restriction = {
                    "requested_remaining_sms": remaining,
                    "actual_restricted_sms": actual_restricted_sms,
                    "partition_side": partition_side,
                    "partition_target_sms": partition_target,
                }
                prediction_source = "max(first isolated, second restricted isolated)"
                restricted_results[order_name] = restricted
            orders[order_name] = {
                "first": first,
                "second": second,
                "first_sm_demand": isolated[first]["sm_demand"],
                "remaining_sms": remaining,
                "first_isolated_ms": isolated[first]["runtime_ms"],
                "second_modeled_ms": second_modeled_ms,
                "predicted_ms": predicted_ms,
                "prediction_source": prediction_source,
                "restriction": restriction,
            }

        print("[3/4] Measuring both unrestricted concurrent launch orders", flush=True)
        actual_results: dict[str, dict[str, Any]] = {}
        for order_name in ("compute_first", "comm_first"):
            args.managed_launch_order = order_name
            actual_path = run_dir / f"actual_{order_name}.json"
            bench.run_torchrun_phase(
                args,
                "unmanaged_ordered",
                actual_path,
                traces,
                environment,
            )
            actual = json.loads(actual_path.read_text(encoding="utf-8"))
            actual_results[order_name] = actual
            actual_ms = float(actual["summary_ms"]["median"])
            orders[order_name].update(
                {
                    "actual_median_ms": actual_ms,
                    "actual_p95_ms": float(actual["summary_ms"]["p95"]),
                    "actual_compute_ms": float(
                        actual["compute_cuda_summary_ms"]["median"]
                    ),
                    "actual_comm_ms": float(
                        actual["comm_cuda_summary_ms"]["median"]
                    ),
                    "error_percent": 100
                    * (actual_ms / float(orders[order_name]["predicted_ms"]) - 1),
                }
            )

        result = {
            "schema_version": 1,
            "methodology": (
                "Launch-order SM admission: first kernel uses its isolated runtime; "
                "second kernel is remeasured on total minus first-kernel SM demand; "
                "zero remainder predicts serialized isolated execution."
            ),
            "workload": bench.workload(args),
            "device": baseline["device"],
            "warmups": args.warmups,
            "iterations": args.iterations,
            "isolated": isolated,
            "profiles": profiles,
            "orders": orders,
            "restricted_results": restricted_results,
            "actual_results": actual_results,
        }
        bench.atomic_write_json(output, result)
        report = output.with_suffix(".md")
        report.write_text(render_report(result), encoding="utf-8")
        print(f"[4/4] Wrote {output} and {report}", flush=True)
    finally:
        if args.keep_artifacts:
            retained = output.parent / f"{output.stem}_artifacts"
            if retained.exists():
                raise RuntimeError(f"refusing to replace existing {retained}")
            shutil.move(str(run_dir), retained)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
