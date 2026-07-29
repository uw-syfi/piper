#!/usr/bin/env python3
"""Run the Qwen397B A2A x gradient-collective matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


def cases() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for size in (4, 8, 16, 32, 64, 96, 128, 256, 512, 1024, 1940):
        result.append(
            {"phase": "single", "name": f"ar_single_{size}", "buckets": str(size)}
        )

    for first, second in (
        (64, 128),
        (128, 64),
        (64, 256),
        (256, 64),
        (128, 256),
        (256, 128),
    ):
        result.append(
            {
                "phase": "two_bucket",
                "name": f"ar_pair_{first}_{second}",
                "buckets": f"{first},{second}",
            }
        )

    for first, second in (
        (1940, 4),
        (4, 1940),
        (1940, 64),
        (64, 1940),
        (1940, 128),
        (128, 1940),
    ):
        result.append(
            {
                "phase": "large",
                "name": f"ar_large_{first}_{second}",
                "buckets": f"{first},{second}",
            }
        )

    for size in (64, 128):
        for depth in (2, 4, 8):
            result.append(
                {
                    "phase": "depth",
                    "name": f"ar_depth_{size}_x{depth}",
                    "buckets": ",".join([str(size)] * depth),
                }
            )

    for cycles in (1_000_000, 2_000_000, 4_000_000, 8_000_000):
        result.extend(
            [
                {
                    "phase": "offset",
                    "name": f"ar_offset_a2a_delay_{cycles}",
                    "buckets": "64,128",
                    "a2a_delay_cycles": cycles,
                },
                {
                    "phase": "offset",
                    "name": f"ar_offset_gradient_delay_{cycles}",
                    "buckets": "64,128",
                    "gradient_delay_cycles": cycles,
                },
            ]
        )

    for name, buckets in (
        ("rs_single_64", "64"),
        ("rs_single_128", "128"),
        ("rs_single_256", "256"),
        ("rs_single_1940", "1940"),
        ("rs_pair_64_128", "64,128"),
        ("rs_pair_128_64", "128,64"),
        ("rs_depth_64_x4", "64,64,64,64"),
        ("rs_large_1940_64", "1940,64"),
        ("rs_large_64_1940", "64,1940"),
    ):
        result.append(
            {
                "phase": "reduce_scatter",
                "name": name,
                "buckets": buckets,
                "collective": "reduce_scatter",
            }
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="10.12.248.91")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--ep-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=4)
    parser.add_argument("--node-domain-map", default="0,1,2,3")
    parser.add_argument(
        "--expected-node-ips",
        default="10.12.248.91,10.12.248.30,10.12.249.214,10.12.249.164",
    )
    parser.add_argument(
        "--phases",
        default="single,two_bucket,large,depth,offset,reduce_scatter",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--output-dir", default="out/qwen397b_a2a_gradient_matrix/ep4_dp4"
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {item for item in args.phases.split(",") if item}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_cases = [case for case in cases() if case["phase"] in selected]
    records: list[dict[str, Any]] = []
    total_started = time.time()

    for index, case in enumerate(selected_cases, start=1):
        output = output_dir / f"{case['name']}.json"
        command = [
            sys.executable,
            "scripts/benchmark_qwen397b_a2a_allreduce.py",
            "--address",
            args.address,
            "--port",
            str(args.port),
            "--world-size",
            "16",
            "--nodes",
            "4",
            "--gpus-per-node",
            "4",
            "--ep-size",
            str(args.ep_size),
            "--dp-size",
            str(args.dp_size),
            "--node-domain-map",
            args.node_domain_map,
            "--expected-node-ips",
            args.expected_node_ips,
            "--microbatch-size",
            "1",
            "--seq-len",
            "32768",
            "--gradient-sequence",
            "custom",
            "--gradient-bucket-mib",
            case["buckets"],
            "--gradient-dtype",
            "bf16",
            "--gradient-collective",
            case.get("collective", "allreduce"),
            "--warmups",
            str(args.warmups),
            "--iterations",
            str(args.iterations),
            "--trials",
            str(args.trials),
            "--skip-serialized",
            "--a2a-delay-cycles",
            str(case.get("a2a_delay_cycles", 0)),
            "--gradient-delay-cycles",
            str(case.get("gradient_delay_cycles", 0)),
            "--output",
            str(output),
        ]
        if output.exists() and not args.rerun:
            print(f"[{index}/{len(selected_cases)}] SKIP {case['name']}", flush=True)
            records.append({**case, "output": str(output), "status": "existing"})
            continue

        print(
            f"[{index}/{len(selected_cases)}] RUN {case['name']} buckets={case['buckets']}",
            flush=True,
        )
        started = time.time()
        environment = os.environ.copy()
        environment.update(
            {
                "RAY_DEDUP_LOGS": "0",
                "NCCL_LAUNCH_ORDER_IMPLICIT": "1",
                "NCCL_DEBUG": "WARN",
            }
        )
        completed = subprocess.run(command, env=environment, check=False)
        elapsed = time.time() - started
        record = {
            **case,
            "output": str(output),
            "command": command,
            "elapsed_s": elapsed,
            "returncode": completed.returncode,
            "status": "completed" if completed.returncode == 0 else "failed",
        }
        records.append(record)
        (output_dir / "matrix_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "args": vars(args),
                    "elapsed_s": time.time() - total_started,
                    "records": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if completed.returncode:
            raise SystemExit(completed.returncode)

    print(
        f"Completed {len(selected_cases)} cases in {time.time() - total_started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
