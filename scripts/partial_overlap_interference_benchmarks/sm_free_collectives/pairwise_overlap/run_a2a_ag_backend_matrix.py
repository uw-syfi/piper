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

"""Run A14 for every normal/Zero-CTA AG and balanced-A2A pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.partial_overlap_interference_benchmarks.sm_free_collectives.pairwise_overlap.collective_backends import build_selection
from scripts.partial_overlap_interference_benchmarks.sm_free_collectives.pairwise_overlap.selected_sm_aware_backends import (
    ALL_GATHER_NORMAL_SELECTOR,
    ALL_GATHER_SELECTOR,
    BALANCED_A2A_NORMAL_SELECTOR,
    BALANCED_A2A_SELECTOR,
)


RUNNER = ROOT / (
    "scripts/partial_overlap_interference_benchmarks/"
    "benchmark_pairwise_overlap_cases.py"
)
TABLE_RENDERER = Path(__file__).with_name("render_zero_gap_order_table.py")
BAR_RENDERER = Path(__file__).with_name("render_a2a_ag_backend_matrix_bars.py")
MATRIX = Path(__file__).with_name("pairwise_matrix.json")

CONFIGURATIONS = (
    ("both_normal", ALL_GATHER_NORMAL_SELECTOR, BALANCED_A2A_NORMAL_SELECTOR),
    ("both_zero_cta", ALL_GATHER_SELECTOR, BALANCED_A2A_SELECTOR),
    ("ag_normal_a2a_zero_cta", ALL_GATHER_NORMAL_SELECTOR, BALANCED_A2A_SELECTOR),
    ("ag_zero_cta_a2a_normal", ALL_GATHER_SELECTOR, BALANCED_A2A_NORMAL_SELECTOR),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--node-ip", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--only",
        choices=tuple(key for key, _, _ in CONFIGURATIONS),
        action="append",
        help="Run only the named configuration; may be repeated",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.iterations < 20:
        raise ValueError(
            "this comparison requires at least 20 issued repetitions per order"
        )

    for key, ag_selector, a2a_selector in CONFIGURATIONS:
        if args.only and key not in args.only:
            continue
        run_dir = args.output_dir / key
        run_dir.mkdir(parents=True, exist_ok=True)
        selection = build_selection(
            mode="sm-aware",
            all_gather=ag_selector,
            a2a=a2a_selector,
        )
        config = {
            "schema_version": 1,
            "experiment": "zero_intentional_gap_launch_order_only",
            "scope": "A14_original_payload_backend_matrix",
            "case_keys": ["A14"],
            "collective_backend_selection": selection.to_dict(),
            "collective_payload_overrides_input_numel_per_rank": {},
            "required_valid_repetitions_per_data_point": 20,
            "required_repetitions_per_launch_order": 20,
            "issued_repetitions_per_launch_order": int(args.iterations),
            "backend_matrix_key": key,
            "configured_launch_delay_ms": 0.0,
        }
        config_path = run_dir / "mode_config.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        digest = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:12]
        command = [
            sys.executable,
            str(RUNNER),
            "--address", args.address,
            "--port", str(args.port),
            "--node-ips", args.node_ip,
            "--matrix", str(MATRIX),
            "--cases", "A14",
            "--warmups", str(args.warmups),
            "--iterations", str(args.iterations),
            "--baseline-repetitions", "3",
            "--zero-gap-order-only",
            "--collective-mode", "sm-aware",
            "--sm-aware-all-gather", ag_selector,
            "--sm-aware-a2a", a2a_selector,
            "--output", str(run_dir / "raw_results.json"),
            "--checkpoint", str(run_dir / "checkpoint.json"),
            "--run-id", f"a2a-ag-backend-matrix-{key}-{digest}",
        ]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run(
            [
                sys.executable,
                str(TABLE_RENDERER),
                "--raw", str(run_dir / "raw_results.json"),
                "--mode-config", str(config_path),
                "--output-dir", str(run_dir / "table"),
            ],
            cwd=ROOT,
            check=True,
        )

    if not args.only:
        subprocess.run(
            [
                sys.executable,
                str(BAR_RENDERER),
                "--input-root", str(args.output_dir),
                "--output-dir", str(args.output_dir / "figure"),
            ],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
