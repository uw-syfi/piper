# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Canonical sample inclusion rules for pairwise-overlap estimates.

Adaptive runs preserve rows that the controller rejected, plus a bounded tail
that was already in flight when a cell stopped.  Those rows are useful for
health and rejection diagnostics, but they are not observations ingested by
the sequential estimator.  Every downstream performance estimate must
therefore use the same accounting-aware predicate.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping


ADAPTIVE_DISPOSITION_FIELD = "adaptive_controller_disposition"
ADAPTIVE_INGESTED_FIELD = "adaptive_controller_ingested"
ADAPTIVE_VALID_FIELD = "adaptive_controller_valid_sample"


def _explicit_table_disposition(row: Mapping[str, Any]) -> str | None:
    """Validate and return a disposition propagated into a derived table."""

    raw_disposition = row.get("analysis_disposition")
    included = _optional_bool(row.get("analysis_included"))
    if raw_disposition in (None, "") and included is None:
        return None
    disposition = str(raw_disposition or "").strip()
    if not disposition or included is None:
        raise ValueError("derived analysis accounting is incomplete")
    if included is not disposition.startswith("included_"):
        raise ValueError("derived analysis disposition/boolean disagree")
    return disposition


def _optional_bool(value: Any) -> bool | None:
    if value is None or str(value).strip() == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "pass"}:
        return True
    if normalized in {"0", "false", "no", "n", "fail"}:
        return False
    raise ValueError(f"invalid boolean value {value!r}")


def analysis_sample_disposition(row: Mapping[str, Any]) -> str:
    """Classify a raw or CSV sample for estimates versus diagnostics.

    Non-adaptive/legacy rows retain the historical runner-``valid`` behavior.
    If any adaptive accounting value is populated, however, inclusion requires
    the controller to have ingested the row *and* classified it valid.  Explicit
    disposition values are checked against the redundant boolean fields so
    inconsistent accounting cannot silently enter an estimate.
    """

    explicit = _explicit_table_disposition(row)
    if explicit is not None:
        return explicit

    raw_disposition = row.get(ADAPTIVE_DISPOSITION_FIELD)
    disposition = (
        "" if raw_disposition is None else str(raw_disposition).strip().lower()
    )
    ingested = _optional_bool(row.get(ADAPTIVE_INGESTED_FIELD))
    controller_valid = _optional_bool(row.get(ADAPTIVE_VALID_FIELD))
    runner_valid = _optional_bool(row.get("valid"))
    has_adaptive_accounting = bool(disposition) or ingested is not None or (
        controller_valid is not None
    )

    if not has_adaptive_accounting:
        return (
            "included_legacy_valid"
            if runner_valid is not False
            else "excluded_legacy_invalid"
        )

    if disposition == "discarded_after_stop" or ingested is False:
        return "excluded_adaptive_discarded_after_stop"
    if disposition == "ingested_invalid" or controller_valid is False:
        return "excluded_adaptive_ingested_invalid"
    if disposition not in {"", "ingested_valid"}:
        return "excluded_adaptive_unknown_disposition"
    if ingested is not True or controller_valid is not True:
        return "excluded_adaptive_incomplete_accounting"
    if runner_valid is False:
        return "excluded_adaptive_runner_invalid"
    return "included_adaptive_ingested_valid"


def _declares_fixed_v2(row: Mapping[str, Any]) -> bool:
    """Return whether the raw sample declares the fixed-V2 methodology."""

    return any(
        field in row
        for field in (
            "runtime_start_geometry_disposition",
            "runtime_start_geometry_scoring_eligible",
            "runtime_start_geometry_treatment_executed",
            "geometry_validity_error_pct",
            "scheduled_geometry_error_pct",
            "counterfactual_geometry_error_pct",
        )
    )


def validator_equivalent_sample_disposition(
    row: Mapping[str, Any],
    *,
    result_correct: bool | None,
    expected_operation_indices: Iterable[int | str],
) -> str:
    """Classify a raw sample using the validator's exact scoring predicate.

    Imports are intentionally local: the validator is the normative owner of
    normalization and eligibility, while the validator itself must remain
    independent of downstream analysis code. Legacy rows retain their prior
    accounting-only behavior; authoritative fixed-V2 rows fail closed.
    """

    accounting_disposition = analysis_sample_disposition(row)
    if not accounting_disposition.startswith("included_"):
        return accounting_disposition
    has_adaptive_accounting = any(
        row.get(field) not in (None, "")
        for field in (
            ADAPTIVE_DISPOSITION_FIELD,
            ADAPTIVE_INGESTED_FIELD,
            ADAPTIVE_VALID_FIELD,
        )
    )
    if has_adaptive_accounting or not _declares_fixed_v2(row):
        return accounting_disposition

    from src.pairwise_overlap_corpus_validation import (
        _normalize_sample,
        _sample_is_accepted,
    )

    normalized = _normalize_sample(
        dict(row),
        result_correct,
        expected_operation_indices={
            str(index) for index in expected_operation_indices
        },
    )
    accepted = _sample_is_accepted(
        normalized,
        geometry_tolerance_pct=10.0,
        drift_threshold_pct=5.0,
    )
    return (
        accounting_disposition
        if accepted
        else "excluded_validator_scoring_ineligible"
    )


def is_validator_equivalent_sample(
    row: Mapping[str, Any],
    *,
    result_correct: bool | None,
    expected_operation_indices: Iterable[int | str],
) -> bool:
    """Return fixed-V2 estimator eligibility for one result-owned sample."""

    return validator_equivalent_sample_disposition(
        row,
        result_correct=result_correct,
        expected_operation_indices=expected_operation_indices,
    ).startswith("included_")


def is_analysis_sample(row: Mapping[str, Any]) -> bool:
    """Return whether ``row`` belongs in a performance estimate."""

    return analysis_sample_disposition(row).startswith("included_")


def analysis_inclusion_counts(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Return diagnostic counts without discarding rejected/tail rows."""

    counts = Counter(analysis_sample_disposition(row) for row in rows)
    counts["total_rows"] = sum(counts.values())
    counts["analysis_included_rows"] = sum(
        count for name, count in counts.items() if name.startswith("included_")
    )
    counts["analysis_excluded_rows"] = (
        counts["total_rows"] - counts["analysis_included_rows"]
    )
    return dict(sorted(counts.items()))


__all__ = [
    "ADAPTIVE_DISPOSITION_FIELD",
    "ADAPTIVE_INGESTED_FIELD",
    "ADAPTIVE_VALID_FIELD",
    "analysis_inclusion_counts",
    "analysis_sample_disposition",
    "is_analysis_sample",
    "is_validator_equivalent_sample",
    "validator_equivalent_sample_disposition",
]
