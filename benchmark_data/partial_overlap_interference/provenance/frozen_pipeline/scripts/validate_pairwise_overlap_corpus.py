#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Validate acceptance, health, and sweep coverage of an overlap corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from src.pairwise_overlap_corpus_validation import (
    DEFAULT_COVERAGE_TARGETS,
    DEFAULT_DISPATCH_MODES,
    DEFAULT_MAX_REJECTION_RATE_PCT,
    load_samples,
    validate_samples,
    write_report,
)


def _csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str) -> list[int]:
    return [int(item) for item in _csv_strings(value)]


def _csv_floats(value: str) -> list[float]:
    return [float(item) for item in _csv_strings(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="raw/enriched result JSON or samples CSV")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--expected-sequences",
        type=_csv_ints,
        help="comma-separated sequence lengths; defaults to observed values",
    )
    parser.add_argument(
        "--expected-cases",
        type=_csv_strings,
        help="comma-separated case keys; defaults to observed values",
    )
    parser.add_argument(
        "--expected-coverage-targets",
        type=_csv_floats,
        default=list(DEFAULT_COVERAGE_TARGETS),
    )
    parser.add_argument(
        "--expected-dispatch-modes",
        type=_csv_strings,
        default=list(DEFAULT_DISPATCH_MODES),
    )
    parser.add_argument("--drift-threshold-pct", type=float, default=5.0)
    parser.add_argument("--geometry-tolerance-pct", type=float, default=10.0)
    parser.add_argument(
        "--max-rejection-rate-pct",
        type=float,
        default=DEFAULT_MAX_REJECTION_RATE_PCT,
        help=(
            "Maximum invalid-attempt percentage in each feasible stratum/cell; "
            "geometry and drift rejects remain reported but may be tolerated"
        ),
    )
    parser.add_argument("--statistical-floors", type=_csv_ints, default=[50, 100])
    parser.add_argument(
        "--expected-fixed-iterations",
        type=int,
        help=(
            "Require every nominal stratum to contain each sample index from "
            "0 through N-1 exactly once, including unresolved and infeasible rows"
        ),
    )
    parser.add_argument(
        "--require-counterfactual-v2",
        action="store_true",
        help=(
            "Fail closed unless every JSON result declares the V2 "
            "counterfactual methodology (fixed or adaptive)"
        ),
    )
    parser.add_argument(
        "--require-adjacent-pre-rebased-delays",
        action="store_true",
        help=(
            "Fail closed unless every retained treatment independently passes "
            "the adjacent-pre, distinct K3 consensus, hash, and causal-order "
            "evidence contract (or exact duration-bound static proof)"
        ),
    )
    parser.add_argument(
        "--require-counterfactual-v2-adaptive",
        action="store_true",
        help=(
            "Fail closed unless every JSON result declares the V2 "
            "counterfactual adaptive methodology"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, provenance = load_samples(args.input)
    report = validate_samples(
        samples,
        expected_sequences=args.expected_sequences,
        expected_cases=args.expected_cases,
        expected_coverage_targets=args.expected_coverage_targets,
        expected_dispatch_modes=args.expected_dispatch_modes,
        drift_threshold_pct=args.drift_threshold_pct,
        geometry_tolerance_pct=args.geometry_tolerance_pct,
        max_rejection_rate_pct=args.max_rejection_rate_pct,
        floors=args.statistical_floors,
        expected_fixed_iterations=args.expected_fixed_iterations,
        require_counterfactual_v2=args.require_counterfactual_v2,
        require_adjacent_pre_rebased_delays=(
            args.require_adjacent_pre_rebased_delays
        ),
        require_counterfactual_v2_adaptive=(
            args.require_counterfactual_v2_adaptive
        ),
    )
    report["provenance"] = provenance
    json_path, markdown_path = write_report(report, args.output_prefix)
    print(json.dumps({
        "accepted": report["summary"]["accepted"],
        "json": str(json_path),
        "markdown": str(markdown_path),
        "samples": report["summary"]["sample_count"],
        "strata": report["summary"]["stratum_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
