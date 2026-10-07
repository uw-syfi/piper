# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Stable run and sample identities for pairwise-overlap result corpora."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
from urllib.parse import quote


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def explicit_run_id(record: Mapping[str, Any] | None) -> str | None:
    """Return an explicitly recorded source/run identity, if present."""

    if not record:
        return None
    for key in ("source_run_id", "run_id"):
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    configuration = record.get("configuration")
    if isinstance(configuration, Mapping):
        for key in ("source_run_id", "run_id"):
            value = configuration.get(key)
            if value not in (None, ""):
                return str(value)
    return None


def deterministic_document_run_id(document: Mapping[str, Any]) -> str:
    """Return a content-derived fallback for legacy standalone documents."""

    explicit = explicit_run_id(document)
    if explicit is not None:
        return explicit
    return f"standalone-{_canonical_digest(document)[:20]}"


def result_source_run_id(
    document: Mapping[str, Any], result: Mapping[str, Any]
) -> str:
    """Resolve result identity, preferring merge annotations and run IDs."""

    explicit = explicit_run_id(result)
    if explicit is not None:
        return explicit
    return deterministic_document_run_id(document)


def run_id_for_row(
    document: Mapping[str, Any], result: Mapping[str, Any]
) -> str:
    """Resolve the upstream run ID exposed in tabular output.

    New merged corpora use ``source_run_id``.  Legacy producers occasionally
    use ``run_id`` instead, so the resolved source identity is the compatible
    fallback for both representations.
    """

    for record in (result, document):
        if record:
            value = record.get("run_id")
            if value not in (None, ""):
                return str(value)
            configuration = record.get("configuration")
            if isinstance(configuration, Mapping):
                value = configuration.get("run_id")
                if value not in (None, ""):
                    return str(value)
    return result_source_run_id(document, result)


def sample_identity(
    source_run_id: str,
    case_key: str,
    sample: Mapping[str, Any],
) -> str:
    """Build a collision-resistant, human-readable sample ID."""

    run_token = quote(str(source_run_id), safe="-._~")
    case_token = quote(str(case_key), safe="-._~")
    seq_len = int(sample["seq_len"])
    return (
        f"r{run_token}--{case_token}-s{seq_len}-"
        f"{sample['dispatch_order']}-{int(sample['sample_index']):04d}-"
        f"v{int(sample['victim_index'])}"
    )


def document_fingerprint(document: Mapping[str, Any]) -> str:
    """Return the full canonical digest used to disambiguate repeated IDs."""

    return _canonical_digest(document)


__all__ = [
    "deterministic_document_run_id",
    "document_fingerprint",
    "explicit_run_id",
    "result_source_run_id",
    "run_id_for_row",
    "sample_identity",
]
