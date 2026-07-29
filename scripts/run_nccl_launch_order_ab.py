#!/usr/bin/env python3
"""Reproduce the proposal's A2A/P2P x AG launch-order experiments."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--expected-node-ips", required=True)
    parser.add_argument("--expected-hostnames", required=True)
    parser.add_argument("--launch-order", choices=("0", "1"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--workloads",
        default="p2p-sweep,p2p-trace,a2a",
        help="Comma-separated subset of p2p-sweep,p2p-trace,a2a",
    )
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def run(command: list[str], output: Path, env: dict[str, str], rerun: bool) -> None:
    if output.exists() and not rerun:
        print(f"SKIP existing {output}", flush=True)
        return
    print(shlex.join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def main() -> None:
    args = parse_args()
    workloads = {item.strip() for item in args.workloads.split(",") if item.strip()}
    unknown = workloads - {"p2p-sweep", "p2p-trace", "a2a"}
    if unknown:
        raise ValueError(f"unknown workloads: {sorted(unknown)}")

    output_root = args.output_root / f"implicit_{args.launch_order}"
    env = os.environ.copy()
    env.update(
        {
            "NCCL_LAUNCH_ORDER_IMPLICIT": args.launch_order,
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_DEBUG": "WARN",
            "RAY_DEDUP_LOGS": "0",
        }
    )
    python = sys.executable

    p2p_common = [
        python,
        "scripts/benchmark_qwen397b_p2p_allgather.py",
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
        "--dp-size",
        "8",
        "--node-domain-map",
        "0,0,1,1",
        "--expected-node-ips",
        args.expected_node_ips,
        "--expected-hostnames",
        args.expected_hostnames,
        "--microbatch-size",
        "1",
        "--seq-len",
        "32768",
        "--p2p-pair-mode",
        "single",
        "--p2p-pair-index",
        "0",
        "--p2p-directions",
        "both",
        "--p2p-shapes",
        "hidden",
        "--allgather-shapes",
        "",
    ]

    if "p2p-sweep" in workloads:
        output = output_root / "p2p_ag_sweep" / "results.json"
        run(
            p2p_common
            + [
                "--allgather-full-payload-mib",
                "256,512,1024,2048",
                "--allgather-counts",
                "1,2",
                "--warmups",
                "10",
                "--iterations",
                "50",
                "--trials",
                "2",
                "--output",
                str(output),
                "--csv-output",
                str(output.with_name("summary.csv")),
            ],
            output,
            env,
            args.rerun,
        )

    if "p2p-trace" in workloads:
        output = output_root / "p2p_ag_2048_trace" / "results.json"
        run(
            p2p_common
            + [
                "--allgather-full-payload-mib",
                "2048",
                "--allgather-counts",
                "1,2",
                "--warmups",
                "3",
                "--iterations",
                "5",
                "--trials",
                "1",
                "--output",
                str(output),
                "--csv-output",
                str(output.with_name("summary.csv")),
                "--profile-dir",
                str(output.with_name("traces")),
            ],
            output,
            env,
            args.rerun,
        )

    if "a2a" in workloads:
        a2a_common = [
            python,
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
            "8",
            "--dp-size",
            "2",
            "--node-domain-map",
            "0,0,1,1",
            "--expected-node-ips",
            args.expected_node_ips,
            "--microbatch-size",
            "1",
            "--seq-len",
            "32768",
            "--gradient-sequence",
            "custom",
            "--gradient-dtype",
            "bf16",
            "--gradient-collective",
            "allgather",
            "--architecture-profile",
            "production_exact",
            "--expert-layout",
            "fused_gate_up",
            "--layers",
            "60",
            "--warmups",
            "5",
            "--iterations",
            "25",
            "--trials",
            "2",
            "--skip-serialized",
        ]
        for buckets in ("256", "512", "1024", "2048", "512,1024", "1024,512"):
            label = buckets.replace(",", "_")
            output = output_root / f"a2a_ag_{label}" / "results.json"
            command = a2a_common + [
                "--gradient-bucket-mib",
                buckets,
                "--output",
                str(output),
            ]
            if "," in buckets:
                command += ["--profile-dir", str(output.with_name("traces"))]
            run(command, output, env, args.rerun)


if __name__ == "__main__":
    main()
