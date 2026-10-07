# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""Validate the committed Figure 1 data, provenance, and artifact hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_int(row: dict[str, str], name: str) -> int:
    return int(float(row[name]))


def validate_source(path: Path, expected: dict[str, Any]) -> None:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "case_key",
        "dispatch_mode",
        "victim_index",
        "victim_family",
        "aggressor_family",
        "sample_count",
        "shape",
        "victim_kernel_key",
    }
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Figure 1 source is missing columns: {sorted(missing)}")
    directed_lines = {
        (row["case_key"], as_int(row, "victim_index"), row["dispatch_mode"])
        for row in rows
    }
    checks = {
        "combined_points": len(rows),
        "combined_samples": sum(as_int(row, "sample_count") for row in rows),
        "combined_directed_dispatch_lines": len(directed_lines),
    }
    for name, observed in checks.items():
        wanted = int(expected[name])
        if observed != wanted:
            raise ValueError(f"{name}: observed {observed}, expected {wanted}")
    modes = {row["dispatch_mode"] for row in rows}
    if modes != {"victim_first", "aggressor_first"}:
        raise ValueError(f"unexpected dispatch modes: {sorted(modes)}")
    victim_kernels = {row["victim_kernel_key"] for row in rows}
    if victim_kernels != set(expected["source_victim_kernels"]):
        raise ValueError("Figure 1 victim-kernel inventory changed")


def main() -> None:
    bundle = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=bundle)
    args = parser.parse_args()
    root = args.bundle.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["artifact_sha256"].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {relative}: {observed}")
    repository = root.parents[1]
    for relative, expected_hash in manifest["repository_sha256"].items():
        path = repository / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {relative}: {observed}")
    validate_source(
        root / "data/figure1_overlap_response_by_victim_source.csv",
        manifest["figure1_validation"],
    )
    print(
        "PASS: Figure 1 bundle hashes and aggregate validation "
        f"({len(manifest['artifact_sha256'])} bundle artifacts, "
        f"{len(manifest['repository_sha256'])} repository sources)"
    )


if __name__ == "__main__":
    main()
