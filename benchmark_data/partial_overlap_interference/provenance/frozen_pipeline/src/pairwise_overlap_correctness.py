"""Shared completeness-aware correctness status for overlap result records."""

from __future__ import annotations

from typing import Any


def _passed(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    return None


def result_correctness_status(result: dict[str, Any]) -> bool | None:
    """Return True, False, or None for a benchmark result's correctness.

    Every participant operation must be represented by at least one passing
    correctness row. Any explicit failure dominates incomplete coverage;
    otherwise incomplete or malformed coverage is reported as ``None``
    (unvalidated), never as a vacuous success.
    """

    participants = result.get("assignment", {}).get("participants", [])
    required = set(range(len(participants)))
    correctness = result.get("correctness") or []
    parsed = [_passed(row.get("passed")) for row in correctness]
    if any(value is False for value in parsed):
        return False
    if any(value is None for value in parsed):
        return None
    covered = {
        int(row["operation_index"])
        for row, value in zip(correctness, parsed)
        if value is True and row.get("operation_index") is not None
    }
    if required and required.issubset(covered):
        return True
    return None
