#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Merge compatible pairwise-overlap benchmark JSON documents.

The merge is deliberately lossless: every input keeps its placement, clock,
configuration, and environment record in ``source_runs`` while every copied
result is annotated with a stable ``source_run_id``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from src.pairwise_overlap_identity import (
    deterministic_document_run_id,
    document_fingerprint,
    explicit_run_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def source_record(
    path: Path,
    document: dict[str, Any],
    source_run_id: str,
) -> dict[str, Any]:
    return {
        "source_run_id": source_run_id,
        "upstream_run_id": explicit_run_id(document),
        "document_sha256": document_fingerprint(document),
        "path": str(path),
        "configuration": document.get("configuration"),
        "placement": document.get("placement"),
        "clock_calibration": document.get("clock_calibration"),
        "environment": document.get("environment"),
        "case_keys": [
            result["assignment"]["case_key"] for result in document["results"]
        ],
    }


def _unique_source_run_ids(
    sources: list[tuple[Path, dict[str, Any]]],
) -> list[str]:
    """Assign stable IDs, disambiguating reused or byte-identical inputs."""

    candidates = [
        deterministic_document_run_id(document) for _, document in sources
    ]
    counts = {candidate: candidates.count(candidate) for candidate in candidates}
    proposed = []
    for (_, document), candidate in zip(sources, candidates):
        if counts[candidate] == 1:
            proposed.append(candidate)
            continue
        fingerprint = document_fingerprint(document)
        proposed.append(f"{candidate}--doc-{fingerprint[:12]}")

    # Two paths can intentionally contain the exact same document. The path
    # hash is needed only in that intrinsically ambiguous case; normal run IDs
    # remain independent of file location.
    resolved = []
    seen: dict[str, int] = {}
    for (path, _), candidate in zip(sources, proposed):
        if proposed.count(candidate) > 1:
            path_hash = hashlib.sha256(
                str(path.resolve()).encode("utf-8")
            ).hexdigest()[:10]
            candidate = f"{candidate}--path-{path_hash}"
        occurrence = seen.get(candidate, 0)
        seen[candidate] = occurrence + 1
        if occurrence:
            candidate = f"{candidate}--copy-{occurrence + 1}"
        resolved.append(candidate)
    if len(set(resolved)) != len(resolved):
        raise AssertionError("failed to assign unique source_run_id values")
    return resolved


def _annotated_result(
    result: dict[str, Any], source_run_id: str
) -> dict[str, Any]:
    copied = copy.deepcopy(result)
    upstream = copied.get("source_run_id")
    if upstream not in (None, "") and str(upstream) != source_run_id:
        copied["upstream_source_run_id"] = upstream
    copied["source_run_id"] = source_run_id
    return copied


def merge_documents(
    sources: list[tuple[Path, dict[str, Any]]],
) -> dict[str, Any]:
    """Merge documents while preserving an unambiguous source-run identity."""

    if not sources:
        raise ValueError("at least one input document is required")
    documents = [document for _, document in sources]
    versions = {document.get("schema_version") for document in documents}
    if len(versions) != 1:
        raise ValueError(f"schema versions differ: {sorted(versions)!r}")
    source_run_ids = _unique_source_run_ids(sources)
    return {
        "schema_version": documents[0]["schema_version"],
        "configuration": {
            "merge_kind": "pairwise_overlap_result_corpus",
            "source_count": len(documents),
            "sequence_lengths": sorted({
                int(sample["seq_len"])
                for document in documents
                for result in document["results"]
                for sample in result["samples"]
            }),
        },
        "placement": documents[0].get("placement"),
        "clock_calibration": documents[0].get("clock_calibration"),
        "source_runs": [
            source_record(path, document, source_run_id)
            for (path, document), source_run_id in zip(sources, source_run_ids)
        ],
        "results": [
            _annotated_result(result, source_run_id)
            for document, source_run_id in zip(documents, source_run_ids)
            for result in document["results"]
        ],
        "environment": {
            "all_source_runs_default_nccl_behavior": all(
                bool(document.get("environment", {}).get(
                    "all_workers_default_nccl_behavior", False
                ))
                for document in documents
            ),
        },
    }


def main() -> None:
    args = parse_args()
    sources = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in args.inputs
    ]
    merged = merge_documents(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "source_runs": len(sources),
        "results": len(merged["results"]),
        "sequence_lengths": merged["configuration"]["sequence_lengths"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
