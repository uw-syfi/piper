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

"""Run the NCCL/Zero-CTA pairwise overlap matrix on one eight-GPU node.

Without ``--execute`` this command performs the runner's CPU-only dry-run and
prints the exact GPU command. With ``--execute`` it runs the pairwise
assignments for grouped-GEMM/A2A, grouped-GEMM/AG, and A2A/AG, converts the
raw result, validates that every data point has exactly 20 estimator-valid
repetitions, and renders the victim-row figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from scripts.partial_overlap_interference_benchmarks.sm_free_collectives.pairwise_overlap.collective_backends import (
    build_selection,
)
from scripts.partial_overlap_interference_benchmarks.sm_free_collectives.pairwise_overlap.selected_sm_aware_backends import (
    ALL_GATHER_SELECTOR,
    BALANCED_A2A_SELECTOR,
)
RUNNER = REPO_ROOT / (
    "scripts/partial_overlap_interference_benchmarks/"
    "benchmark_pairwise_overlap_cases.py"
)
CONVERTER = REPO_ROOT / (
    "benchmark_data/partial_overlap_interference/provenance/frozen_pipeline/"
    "scripts/convert_pairwise_case_results.py"
)
RENDERER = Path(__file__).with_name("render_pairwise_mode_figure.py")
ZERO_GAP_TABLE_RENDERER = Path(__file__).with_name(
    "render_zero_gap_order_table.py"
)
PAIRWISE_MATRIX = Path(__file__).with_name("pairwise_matrix.json")
PAIRWISE_MATRIX_SHA256 = (
    "03e9430d0c5b86f62fad497c4a405dda1b17a9afca527b9e4dd221f1bf9371c8"
)
# The SM-free matrix is intentionally fixed to these three pair families.
PAIRWISE_CASE_KEYS = (
    "A03:C-GROUPED",
    "A04:C-GROUPED",
    "A14",
)
GROUPED_GEMM_CASE_KEYS = PAIRWISE_CASE_KEYS[:2]
EXPECTED_CASE_CATEGORIES = {
    "A03:C-GROUPED": {"a2a", "compute"},
    "A04:C-GROUPED": {"ag", "compute"},
    "A14": {"ag", "a2a"},
}
GROUPED_GEMM_KERNEL_ALIAS = "K-28e33c0c27e1"
GROUPED_GEMM_KERNEL_SHA1 = "28e33c0c27e16aba63e26d313887551f4ad225cb"
GROUPED_GEMM_SIGNATURE = {
    "category": "compute",
    "collective": "",
    "dtype": "",
    "group_size": -1,
    "in_msg_nelems": -1,
    "out_msg_nelems": -1,
    "grid": [132, 1, 1],
    "block": [384, 1, 1],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("nccl", "sm-aware"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--zero-gap-order-only",
        action="store_true",
        help=(
            "Submit each pair back-to-back in both orders with zero configured "
            "delay and emit a slowdown table instead of an overlap figure"
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--address", default="10.65.10.75")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--node-ip", default="10.65.10.75")
    parser.add_argument(
        "--matrix", type=Path,
        default=PAIRWISE_MATRIX,
    )
    parser.add_argument("--seq-len", type=int, default=12_288)
    parser.add_argument(
        "--a2a-input-numel", type=int,
        help="Matched-duration balanced A2A input/output elements per rank",
    )
    parser.add_argument(
        "--ag-input-numel", type=int,
        help="Matched-duration all-gather input elements per rank",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--coverage-targets", default="0,5,25,50,75,100")
    parser.add_argument("--stage-lead-ms", type=float, default=5.0)
    parser.add_argument("--geometry-calibration-trials", type=int, default=8)
    parser.add_argument("--geometry-max-relative-shift-ms", type=float, default=100.0)
    parser.add_argument("--sm-aware-all-gather")
    parser.add_argument("--sm-aware-a2a")
    parser.add_argument("--sm-aware-options", type=Path)
    return parser.parse_args()


def mode_configuration(args: argparse.Namespace) -> dict[str, Any]:
    case_keys = PAIRWISE_CASE_KEYS
    rendered_victims = ["C-GROUPED", "a2a", "ag"]
    selected_defaults = args.mode == "sm-aware"
    if args.zero_gap_order_only:
        if args.iterations < 20:
            raise ValueError(
                "zero-gap order-only mode requires at least 20 issued repetitions"
            )
    elif args.iterations != 20:
        raise ValueError(
            "the prescribed-overlap comparison requires exactly 20 iterations"
        )
    selection = build_selection(
        mode=args.mode,
        all_gather=(
            args.sm_aware_all_gather
            or (ALL_GATHER_SELECTOR if selected_defaults else None)
        ),
        a2a=(
            args.sm_aware_a2a
            or (BALANCED_A2A_SELECTOR if selected_defaults else None)
        ),
        options_path=args.sm_aware_options,
    )
    payload_overrides = {
        category: int(value)
        for category, value in {
            "a2a": args.a2a_input_numel,
            "ag": args.ag_input_numel,
        }.items()
        if value is not None
    }
    for category, value in payload_overrides.items():
        if value <= 0:
            raise ValueError(f"{category} input numel must be positive")
        if category == "a2a" and value % 8:
            raise ValueError(
                f"{category} input numel must be divisible by eight: {value}"
            )
    if args.zero_gap_order_only and payload_overrides:
        raise ValueError(
            "zero-gap order-only comparison requires the original payload sizes"
        )
    return {
        "schema_version": 1,
        "collective_backend_selection": selection.to_dict(),
        "scope": "sm_free_grouped_gemm_ag_a2a_pairwise_matrix",
        "experiment": (
            "zero_intentional_gap_launch_order_only"
            if args.zero_gap_order_only
            else "prescribed_partial_overlap"
        ),
        "case_keys": list(case_keys),
        "rendered_victim_kernel_keys": rendered_victims,
        "required_valid_repetitions_per_data_point": 20,
        "required_repetitions_per_launch_order": 20,
        "issued_repetitions_per_launch_order": int(args.iterations),
        "grouped_gemm_kernel": {
            "case_keys": [
                key for key in GROUPED_GEMM_CASE_KEYS if key in case_keys
            ],
            "kernel_alias": GROUPED_GEMM_KERNEL_ALIAS,
            "kernel_sha1": GROUPED_GEMM_KERNEL_SHA1,
            "signature": GROUPED_GEMM_SIGNATURE,
            "runtime_workload": "expert_grouped_gemm",
        },
        "sequence_length": args.seq_len,
        "collective_payload_overrides_input_numel_per_rank": payload_overrides,
        "payload_matching_basis": (
            "normal_NCCL_isolated_median_runtime_matched_to_fixed_grouped_gemm"
            if payload_overrides else None
        ),
        "balanced_a2a_contract": {
            "input_output_numel_per_rank_at_12288": 201_326_592,
            "peer_numel_at_12288": 25_165_824,
            "dtype": "BFloat16",
            "input_output_mib_per_rank": 384,
            "peer_mib": 48,
            "normal_backend": "torch.distributed all_to_all_single equal split",
            "sm_aware_backend": "NCCL 2.28.3 ncclAllToAll CTAPolicy.Zero",
        },
        "coverage_targets_pct": (
            []
            if args.zero_gap_order_only
            else [
                float(value.strip())
                for value in args.coverage_targets.split(",") if value.strip()
            ]
        ),
        "dispatch_modes": ["victim_first", "aggressor_first"],
        "figure_layout": {
            "rows": "one_per_victim",
            "columns": ["victim_first", "aggressor_first"],
        },
        "zero_gap_order_table_layout": (
            {
                "rows": "one_per_directed_victim_pair",
                "columns": ["victim_launched_first", "victim_launched_second"],
                "configured_launch_delay_ms": 0.0,
            }
            if args.zero_gap_order_only else None
        ),
    }


def runner_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    *,
    dry_run: bool,
) -> list[str]:
    raw_output = args.output_dir / "raw_results.json"
    command = [
        args.python, str(RUNNER),
        "--address", args.address,
        "--port", str(args.port),
        "--node-ips", args.node_ip,
        "--matrix", str(args.matrix),
        "--cases", ",".join(config["case_keys"]),
        "--seq-len", str(args.seq_len),
        "--warmups", str(args.warmups),
        "--iterations", str(args.iterations),
        "--baseline-repetitions", "3",
        "--coverage-targets", args.coverage_targets,
        "--stage-lead-ms", str(args.stage_lead_ms),
        "--geometry-calibration-trials", str(args.geometry_calibration_trials),
        "--geometry-max-relative-shift-ms", str(args.geometry_max_relative_shift_ms),
        "--collective-mode", args.mode,
        "--output", str(raw_output),
        "--checkpoint", str(args.output_dir / "checkpoint.json"),
        "--run-id", _run_id(config),
    ]
    for category, flag in (
        ("a2a", "--a2a-input-numel"),
        ("ag", "--ag-input-numel"),
    ):
        value = config["collective_payload_overrides_input_numel_per_rank"].get(
            category
        )
        if value is not None:
            command.extend([flag, str(value)])
    if args.mode == "sm-aware":
        selected = config["collective_backend_selection"]["backends"]
        command.extend([
            "--sm-aware-all-gather", selected["all_gather"]["selector"],
            "--sm-aware-a2a", selected["balanced_a2a"]["selector"],
        ])
        if args.sm_aware_options is not None:
            command.extend(["--sm-aware-options", str(args.sm_aware_options)])
    if args.resume:
        command.append("--resume")
    if args.zero_gap_order_only:
        command.append("--zero-gap-order-only")
    if dry_run:
        command.append("--dry-run")
    return command


def _run_id(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    mode = config["collective_backend_selection"]["mode"].replace("-", "_")
    return f"pairwise-{mode}-{digest}"


def shell_render(command: list[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_pinned_inputs(args: argparse.Namespace) -> None:
    """Refuse to run a comparison against a changed pairwise matrix."""

    if not args.matrix.is_file():
        raise FileNotFoundError(
            f"missing pairwise matrix: {args.matrix} "
            f"(expected sha256 {PAIRWISE_MATRIX_SHA256})"
        )
    observed = _sha256(args.matrix)
    if observed != PAIRWISE_MATRIX_SHA256:
        raise ValueError(
            f"changed pairwise matrix {args.matrix}: observed {observed}, "
            f"expected {PAIRWISE_MATRIX_SHA256}"
        )
    validate_strict_pairwise_matrix(args.matrix)


def validate_strict_pairwise_matrix(matrix_path: Path) -> None:
    """Pin the three pair families and one exact grouped-GEMM identity."""

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assignments = matrix.get("pair_assignments", [])
    for case_key in PAIRWISE_CASE_KEYS:
        case_id, separator, variant = case_key.partition(":")
        matches = [
            row for row in assignments
            if row.get("case_id") == case_id
            and (not separator or row.get("variant") == variant)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"strict pairwise case {case_key} resolved to {len(matches)} "
                "matrix assignments"
            )
        categories = {str(row["category"]) for row in matches[0]["participants"]}
        expected_categories = EXPECTED_CASE_CATEGORIES[case_key]
        if categories != expected_categories:
            raise ValueError(
                f"strict pairwise case {case_key} has categories "
                f"{sorted(categories)}, expected {sorted(expected_categories)}"
            )
        grouped = [
            row for row in matches[0]["participants"]
            if row.get("category") == "compute"
        ]
        if case_key in GROUPED_GEMM_CASE_KEYS:
            if len(grouped) != 1:
                raise ValueError(
                    f"{case_key} must contain exactly one grouped GEMM"
                )
            identity = grouped[0]
            if (
                identity.get("kernel_alias") != GROUPED_GEMM_KERNEL_ALIAS
                or identity.get("kernel_sha1") != GROUPED_GEMM_KERNEL_SHA1
                or identity.get("signature") != GROUPED_GEMM_SIGNATURE
            ):
                raise ValueError(
                    f"{case_key} grouped-GEMM identity changed: "
                    f"alias={identity.get('kernel_alias')!r}, "
                    f"sha1={identity.get('kernel_sha1')!r}, "
                    f"signature={identity.get('signature')!r}"
                )
        elif grouped:
            raise ValueError(
                f"communication-only strict case {case_key} unexpectedly "
                "contains a compute participant"
            )


def main() -> None:
    args = parse_args()
    validate_pinned_inputs(args)
    config = mode_configuration(args)
    dry_command = runner_command(args, config, dry_run=True)
    result = subprocess.run(
        dry_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    preview = json.loads(result.stdout)
    if preview.get("collective_backend_selection") != config[
        "collective_backend_selection"
    ]:
        raise RuntimeError("runner dry-run changed the collective backend selection")
    execute_command = runner_command(args, config, dry_run=False)
    print("validated mode configuration")
    print(shell_render(execute_command))
    if not args.execute:
        print("no GPU work requested")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "mode_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    subprocess.run(execute_command, cwd=REPO_ROOT, check=True)
    if args.zero_gap_order_only:
        subprocess.run(
            [
                args.python, str(ZERO_GAP_TABLE_RENDERER),
                "--raw", str(args.output_dir / "raw_results.json"),
                "--mode-config", str(config_path),
                "--output-dir", str(args.output_dir / "table"),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        return
    tables = args.output_dir / "tables"
    subprocess.run(
        [args.python, str(CONVERTER), str(args.output_dir / "raw_results.json"), str(tables)],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            args.python, str(RENDERER),
            "--samples", str(tables / "samples.csv"),
            "--mode-config", str(config_path),
            "--output-dir", str(args.output_dir / "figure"),
            "--sequence-length", str(args.seq_len),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
