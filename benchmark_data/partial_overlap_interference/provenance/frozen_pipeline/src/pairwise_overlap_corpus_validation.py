"""Acceptance and sweep-coverage checks for pairwise-overlap corpora.

The validator intentionally depends only on the Python standard library.  It
accepts either the runner's raw/enriched JSON document or the ``samples.csv``
produced by ``convert_pairwise_case_results.py`` and normalizes both to the
same sample-level schema before applying gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

# Authoritative CSV evidence payloads can exceed csv default.
csv.field_size_limit(sys.maxsize)

from src.pairwise_overlap_correctness import result_correctness_status


DEFAULT_COVERAGE_TARGETS = (0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0)
DEFAULT_DISPATCH_MODES = ("victim_first", "aggressor_first")
DEFAULT_MAX_REJECTION_RATE_PCT = 10.0
RUNTIME_START_GEOMETRY_CONFIRMED = "confirmed"
RUNTIME_START_GEOMETRY_UNRESOLVED = "runtime_start_geometry_unresolved"
RUNTIME_START_GEOMETRY_STATIC = "not_applicable_static"
RUNTIME_START_GEOMETRY_DISPOSITIONS = frozenset({
    RUNTIME_START_GEOMETRY_CONFIRMED,
    RUNTIME_START_GEOMETRY_UNRESOLVED,
    RUNTIME_START_GEOMETRY_STATIC,
})
ADAPTIVE_PRECISION_CRITERION = (
    "max(abs(median_pp-ci95_low_pp), abs(ci95_high_pp-median_pp)) "
    "<= target_ci_half_width_pp"
)
COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS = (
    "counterfactual_intervals_from_actual_global_gpu_starts_and_"
    "matched_isolated_pre_post_durations"
)
GEOMETRY_BASELINE_STABILITY_BASIS = (
    "maximum_absolute_pre_post_drift_across_all_operations"
)


class NormalizedSamples(list[dict[str, Any]]):
    """Normalized rows plus optional result-level adaptive diagnostics.

    A list subclass preserves the historical ``load_samples`` API while
    retaining diagnostics for registered adaptive cells that may have emitted
    no treatment row (for example, a calibration-infeasible cell).
    """

    def __init__(
        self,
        rows: Iterable[dict[str, Any]] = (),
        *,
        adaptive_results: Sequence[dict[str, Any]] = (),
        result_metadata: Sequence[dict[str, Any]] = (),
        strict_document_provenance: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(rows)
        self.adaptive_results = list(adaptive_results)
        self.result_metadata = list(result_metadata)
        self.strict_document_provenance = strict_document_provenance


def _bool(value: Any) -> bool | None:
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
    raise ValueError(f"cannot parse boolean value {value!r}")


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _float_mapping(value: Any) -> dict[str, float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict) or not value:
        return None
    parsed: dict[str, float] = {}
    for key, item in value.items():
        number = _float(item)
        if number is None:
            return None
        parsed[str(key)] = number
    return parsed


def _strict_static_duration_bound_proof(row: dict[str, Any]) -> bool:
    """Reconstruct the only row-level exemption from runtime-start scoring."""

    evidence = row.get("delay_rebase_evidence")
    return bool(
        isinstance(evidence, dict)
        and _bool(row.get("target_feasible_from_isolated")) is False
        and _bool(row.get("target_feasible_from_duration")) is False
        and _bool(row.get("target_feasible_after_constraints")) is False
        and _bool(row.get("target_feasible_after_runtime_calibration")) is False
        and row.get("runtime_geometry_calibration_status")
        == "isolated_duration_bound_infeasible"
        and evidence.get("policy") == "static_calibrated_delays_v1"
        and evidence.get("dynamic_rebase_applied") is False
        and evidence.get("static_policy_reason") == "duration_bound_infeasible"
    )


def _runtime_start_geometry_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize explicit V2 dispositions and legacy start-anchor records."""

    strict_static = _strict_static_duration_bound_proof(row)
    evidence = row.get("delay_rebase_evidence")
    static_exemption_claimed = bool(
        row.get("runtime_start_geometry_disposition")
        == RUNTIME_START_GEOMETRY_STATIC
        or row.get("runtime_geometry_calibration_status")
        == "isolated_duration_bound_infeasible"
        or (
            isinstance(evidence, dict)
            and (
                evidence.get("policy") == "static_calibrated_delays_v1"
                or evidence.get("static_policy_reason")
                == "duration_bound_infeasible"
            )
        )
    )
    explicit = any(
        field in row
        for field in (
            "runtime_start_geometry_disposition",
            "runtime_start_geometry_scoring_eligible",
            "runtime_start_geometry_treatment_executed",
        )
    )
    treatment_executed = _bool(
        row.get("runtime_start_geometry_treatment_executed")
    )
    if explicit:
        disposition = str(row.get("runtime_start_geometry_disposition") or "")
        declared_eligible = _bool(
            row.get("runtime_start_geometry_scoring_eligible")
        )
        contract_ok = bool(
            disposition in RUNTIME_START_GEOMETRY_DISPOSITIONS
            and declared_eligible is (disposition == RUNTIME_START_GEOMETRY_CONFIRMED)
            and (treatment_executed is None or treatment_executed is True)
            and (
                disposition != RUNTIME_START_GEOMETRY_STATIC
                or strict_static
            )
        )
        scoring_eligible = bool(
            contract_ok
            and disposition == RUNTIME_START_GEOMETRY_CONFIRMED
            and declared_eligible is True
        )
        source = "explicit"
    else:
        legacy_scoring_allowed = _bool(row.get("start_anchor_scoring_allowed"))
        legacy_resolution_status = (
            evidence.get("start_anchor_resolution_status")
            if isinstance(evidence, dict)
            else None
        )
        reconstructed_unresolved = bool(
            legacy_scoring_allowed is False
            or legacy_resolution_status
            == "corrected_delay_confirmation_unresolved"
        )
        if strict_static:
            disposition = RUNTIME_START_GEOMETRY_STATIC
        elif reconstructed_unresolved:
            disposition = RUNTIME_START_GEOMETRY_UNRESOLVED
        else:
            disposition = RUNTIME_START_GEOMETRY_CONFIRMED
        scoring_eligible = disposition == RUNTIME_START_GEOMETRY_CONFIRMED
        declared_eligible = legacy_scoring_allowed
        contract_ok = not static_exemption_claimed or strict_static
        source = "legacy_reconstructed"
    return {
        "runtime_start_geometry_disposition": disposition,
        "runtime_start_geometry_disposition_source": source,
        "runtime_start_geometry_disposition_explicit": explicit,
        "runtime_start_geometry_treatment_executed": treatment_executed,
        "runtime_start_geometry_declared_scoring_eligible": declared_eligible,
        "runtime_start_geometry_scoring_eligible": scoring_eligible,
        "runtime_start_geometry_contract_ok": contract_ok,
        "runtime_start_geometry_static_exemption_proven": strict_static,
        "runtime_start_geometry_static_exemption_claimed": (
            static_exemption_claimed
        ),
    }


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_string(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_implementation_mapping(
    mapping: Any,
    *,
    context: str,
    errors: list[str],
) -> int:
    """Verify a declared source manifest against this validator checkout."""

    if not isinstance(mapping, dict) or not mapping:
        errors.append(f"{context}: missing non-empty implementation hash mapping")
        return 0
    repo_root = Path(__file__).resolve().parents[1]
    checked = 0
    for relative_name, declared_digest in sorted(mapping.items()):
        if not isinstance(relative_name, str) or not relative_name:
            errors.append(f"{context}: invalid implementation path")
            continue
        candidate = (repo_root / relative_name).resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError:
            errors.append(
                f"{context}.{relative_name}: implementation path escapes repository"
            )
            continue
        if not candidate.is_file():
            errors.append(
                f"{context}.{relative_name}: local implementation file is missing"
            )
            continue
        if not _sha256_string(declared_digest):
            errors.append(
                f"{context}.{relative_name}: invalid implementation SHA-256"
            )
            continue
        checked += 1
        if declared_digest != _file_sha256(candidate):
            errors.append(
                f"{context}.{relative_name}: implementation SHA-256 does not "
                "match the local file"
            )
    return checked


def _verify_retained_digest_pairs(
    value: Any,
    *,
    context: str,
    errors: list[str],
) -> int:
    """Recompute all retained sibling payload digests.

    Row- and record-shaped digests are fail-closed: retaining their digest
    without the corresponding rows is not self-contained provenance. Other
    semantic decision digests are handled by the consensus verifier.
    """

    checked = 0
    if isinstance(value, dict):
        for key, child in value.items():
            child_context = f"{context}.{key}"
            if key.endswith("_sha256"):
                payload_key = key[:-7]
                payload_present = payload_key in value
                rows_or_records = payload_key.endswith(("_rows", "_records"))
                if rows_or_records and not payload_present:
                    errors.append(
                        f"{child_context}: retained digest lacks {payload_key}"
                    )
                elif payload_present:
                    checked += 1
                    if not _sha256_string(child):
                        errors.append(f"{child_context}: invalid SHA-256")
                    else:
                        try:
                            recomputed = _canonical_sha256(value[payload_key])
                        except (TypeError, ValueError):
                            errors.append(
                                f"{child_context}: retained payload is not "
                                "canonically serializable"
                            )
                        else:
                            if child != recomputed:
                                errors.append(
                                    f"{child_context}: digest does not match "
                                    f"retained {payload_key}"
                                )
            checked += _verify_retained_digest_pairs(
                child, context=child_context, errors=errors
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            checked += _verify_retained_digest_pairs(
                child, context=f"{context}[{index}]", errors=errors
            )
    return checked


def _strict_integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _validate_clock_rows(
    rows: Any,
    *,
    context: str,
    world_size: int,
    errors: list[str],
    expected_generation: int | None = None,
    expected_label: str | None = None,
) -> dict[int, dict[str, Any]] | None:
    if not isinstance(rows, list) or len(rows) != world_size:
        errors.append(f"{context}: expected {world_size} calibration rows")
        return None
    by_rank: dict[int, dict[str, Any]] = {}
    required_integer_fields = (
        "clock_calibration_best_rtt_ns",
        "clock_offset_ns",
        "clock_calibration_generation",
        "clock_calibration_timestamp_global_ns",
        "clock_calibration_local_monotonic_ns",
        "clock_calibration_half_rtt_uncertainty_ns",
    )
    for index, row in enumerate(rows):
        row_context = f"{context}[{index}]"
        if not isinstance(row, dict):
            errors.append(f"{row_context}: calibration row must be an object")
            continue
        rank = _strict_integer(row.get("global_rank"))
        if rank is None or rank < 0 or rank >= world_size or rank in by_rank:
            errors.append(f"{row_context}: invalid or duplicate global_rank")
            continue
        by_rank[rank] = row
        parsed: dict[str, int] = {}
        for field in required_integer_fields:
            item = _strict_integer(row.get(field))
            if item is None:
                errors.append(f"{row_context}: missing integer {field}")
            else:
                parsed[field] = item
        for field in (
            "clock_calibration_best_rtt_ns",
            "clock_calibration_generation",
            "clock_calibration_timestamp_global_ns",
            "clock_calibration_local_monotonic_ns",
            "clock_calibration_half_rtt_uncertainty_ns",
        ):
            if field in parsed and parsed[field] < 0:
                errors.append(f"{row_context}: {field} must be non-negative")
        label = row.get("clock_calibration_label")
        if not isinstance(label, str) or not label:
            errors.append(f"{row_context}: missing clock_calibration_label")
        if (
            expected_generation is not None
            and parsed.get("clock_calibration_generation") != expected_generation
        ):
            errors.append(f"{row_context}: calibration generation mismatch")
        if expected_label is not None and label != expected_label:
            errors.append(f"{row_context}: calibration label mismatch")
        rtt = parsed.get("clock_calibration_best_rtt_ns")
        uncertainty = parsed.get("clock_calibration_half_rtt_uncertainty_ns")
        if rtt is not None and uncertainty != (rtt + 1) // 2:
            errors.append(f"{row_context}: half-RTT uncertainty mismatch")
    if set(by_rank) != set(range(world_size)):
        errors.append(f"{context}: calibration rank set is incomplete")
    return by_rank


def _verify_clock_row_references(
    value: Any,
    *,
    context: str,
    history_rows: dict[int, dict[int, dict[str, Any]]],
    max_clock_age_ns: int,
    age_rows_checked: list[int],
    errors: list[str],
) -> int:
    """Tie retained timed rows to a canonical calibration history record."""

    checked = 0
    if isinstance(value, dict):
        is_timed_rank_row = all(
            field in value
            for field in (
                "global_rank",
                "clock_calibration_generation",
                "clock_calibration_best_rtt_ns",
                "clock_calibration_label",
                "clock_calibration_half_rtt_uncertainty_ns",
            )
        )
        if is_timed_rank_row:
            checked += 1
            generation = _strict_integer(value.get("clock_calibration_generation"))
            rank = _strict_integer(value.get("global_rank"))
            canonical = (
                history_rows.get(generation, {}).get(rank)
                if generation is not None and rank is not None
                else None
            )
            if canonical is None:
                errors.append(
                    f"{context}: timed row has no canonical clock history record"
                )
            else:
                for field in (
                    "clock_calibration_best_rtt_ns",
                    "clock_offset_ns",
                    "clock_calibration_generation",
                    "clock_calibration_timestamp_global_ns",
                    "clock_calibration_local_monotonic_ns",
                    "clock_calibration_label",
                    "clock_calibration_half_rtt_uncertainty_ns",
                ):
                    if field in value and value.get(field) != canonical.get(field):
                        errors.append(
                            f"{context}: {field} disagrees with canonical clock "
                            "history"
                        )
                if "clock_calibration_age_ns" in value:
                    age_rows_checked[0] += 1
                    age_ns = _strict_integer(value.get("clock_calibration_age_ns"))
                    if age_ns is None or age_ns < 0 or age_ns > max_clock_age_ns:
                        errors.append(
                            f"{context}: clock_calibration_age_ns exceeds the "
                            "advertised bound"
                        )
        for key, child in value.items():
            checked += _verify_clock_row_references(
                child,
                context=f"{context}.{key}",
                history_rows=history_rows,
                max_clock_age_ns=max_clock_age_ns,
                age_rows_checked=age_rows_checked,
                errors=errors,
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            checked += _verify_clock_row_references(
                child,
                context=f"{context}[{index}]",
                history_rows=history_rows,
                max_clock_age_ns=max_clock_age_ns,
                age_rows_checked=age_rows_checked,
                errors=errors,
            )
    return checked


def _validate_merged_document_provenance(
    document: dict[str, Any],
) -> dict[str, Any]:
    """Validate a lossless merge by reloading and validating every source."""

    # Import the exact merge implementation so validation cannot silently
    # drift from the code that assigns IDs and annotates flattened results.
    from scripts.merge_pairwise_overlap_results import (
        _annotated_result,
        _unique_source_run_ids,
        source_record,
    )
    from src.pairwise_overlap_identity import (
        document_fingerprint,
        explicit_run_id,
    )

    errors: list[str] = []
    counter_names = (
        "implementation_files_checked",
        "retained_digest_count_checked",
        "clock_rows_checked",
        "clock_history_entries_checked",
        "clock_reference_rows_checked",
        "clock_age_rows_checked",
    )
    counters = {name: 0 for name in counter_names}
    source_reports: list[dict[str, Any]] = []

    if document.get("schema_version") != 2:
        errors.append("merged document.schema_version must be 2")
    configuration = document.get("configuration")
    if not isinstance(configuration, dict):
        errors.append("merged document.configuration is missing")
        configuration = {}
    elif configuration.get("merge_kind") != "pairwise_overlap_result_corpus":
        errors.append("merged document merge_kind is invalid")

    source_records = document.get("source_runs")
    merged_results = document.get("results")
    if not isinstance(source_records, list) or not source_records:
        errors.append("merged document.source_runs is missing or empty")
        source_records = []
    if not isinstance(merged_results, list):
        errors.append("merged document.results is missing")
        merged_results = []

    sources: list[tuple[Path, dict[str, Any]]] = []
    source_record_ids: list[str] = []
    for source_index, record in enumerate(source_records):
        context = f"merged document.source_runs[{source_index}]"
        if not isinstance(record, dict):
            errors.append(f"{context}: source record must be an object")
            continue
        source_run_id = record.get("source_run_id")
        if not isinstance(source_run_id, str) or not source_run_id:
            errors.append(f"{context}: source_run_id is missing")
        else:
            source_record_ids.append(source_run_id)
        path_value = record.get("path")
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"{context}: source path is missing")
            continue
        source_path = Path(path_value)
        try:
            source_document = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{context}: source cannot be reloaded: {exc}")
            continue
        if not isinstance(source_document, dict) or not isinstance(
            source_document.get("results"), list
        ):
            errors.append(f"{context}: source is not a final result document")
            continue
        fingerprint = document_fingerprint(source_document)
        if record.get("document_sha256") != fingerprint:
            errors.append(f"{context}: document_sha256 mismatch")
        if record.get("upstream_run_id") != explicit_run_id(source_document):
            errors.append(f"{context}: upstream_run_id mismatch")
        sources.append((source_path, source_document))

    if len(source_record_ids) != len(set(source_record_ids)):
        errors.append("merged document source_run_id values are not unique")

    sources_complete = len(sources) == len(source_records) and bool(sources)
    expected_source_ids: list[str] = []
    if sources_complete:
        expected_source_ids = _unique_source_run_ids(sources)
        if source_record_ids != expected_source_ids:
            errors.append("merged document source_run_id values or order mismatch")

        for source_index, ((source_path, source_document), source_run_id) in enumerate(
            zip(sources, expected_source_ids)
        ):
            context = f"merged document.source_runs[{source_index}]"
            expected_record = source_record(
                source_path, source_document, source_run_id
            )
            if source_records[source_index] != expected_record:
                errors.append(
                    f"{context}: source metadata does not match reloaded source"
                )

            source_report = _validate_strict_document_provenance(source_document)
            source_reports.append({
                "source_index": source_index,
                "source_run_id": source_run_id,
                "path": str(source_path),
                "passed": source_report.get("passed") is True,
                "error_count": source_report.get("error_count", 0),
            })
            for error in source_report.get("errors", []):
                errors.append(f"{context}: {error}")
            for name in counter_names:
                value = source_report.get(name, 0)
                counters[name] += value if isinstance(value, int) else 0

        try:
            sequence_lengths = sorted({
                int(sample["seq_len"])
                for _, source_document in sources
                for result in source_document["results"]
                for sample in result["samples"]
            })
        except (KeyError, TypeError, ValueError):
            sequence_lengths = []
            errors.append("merged source sequence lengths cannot be reconstructed")
        expected_configuration = {
            "merge_kind": "pairwise_overlap_result_corpus",
            "source_count": len(sources),
            "sequence_lengths": sequence_lengths,
        }
        if configuration != expected_configuration:
            errors.append("merged document.configuration mismatch")
        if document.get("placement") != sources[0][1].get("placement"):
            errors.append("merged document.placement mismatch")
        if document.get("clock_calibration") != sources[0][1].get(
            "clock_calibration"
        ):
            errors.append("merged document.clock_calibration mismatch")
        expected_environment = {
            "all_source_runs_default_nccl_behavior": all(
                bool(source_document.get("environment", {}).get(
                    "all_workers_default_nccl_behavior", False
                ))
                for _, source_document in sources
            ),
        }
        if document.get("environment") != expected_environment:
            errors.append("merged document.environment mismatch")

        expected_result_count = sum(
            len(source_document["results"])
            for _, source_document in sources
        )
        if len(merged_results) != expected_result_count:
            errors.append("merged document flattened result count mismatch")
        else:
            merged_index = 0
            mismatch_index: int | None = None
            for (_, source_document), source_run_id in zip(
                sources, expected_source_ids
            ):
                for source_result in source_document["results"]:
                    if merged_results[merged_index] != _annotated_result(
                        source_result, source_run_id
                    ):
                        mismatch_index = merged_index
                        break
                    merged_index += 1
                if mismatch_index is not None:
                    break
            if mismatch_index is not None:
                errors.append(
                    "merged document flattened results differ at index "
                    f"{mismatch_index}"
                )

    return {
        "schema_version": 1,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        **counters,
        "merge_source_count_checked": len(source_reports),
        "source_reports": source_reports,
    }


def _validate_strict_document_provenance(document: Any) -> dict[str, Any]:
    """Validate that a final raw result is independently auditable."""

    errors: list[str] = []
    implementation_files_checked = 0
    clock_rows_checked = 0
    history_entries_checked = 0
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        return {
            "schema_version": 1,
            "passed": False,
            "error_count": 1,
            "errors": ["input is not a final result document"],
            "implementation_files_checked": 0,
            "retained_digest_count_checked": 0,
            "clock_rows_checked": 0,
            "clock_history_entries_checked": 0,
            "clock_reference_rows_checked": 0,
            "clock_age_rows_checked": 0,
        }
    configuration = document.get("configuration")
    if (
        isinstance(configuration, dict)
        and configuration.get("merge_kind")
        == "pairwise_overlap_result_corpus"
    ):
        return _validate_merged_document_provenance(document)
    if document.get("schema_version") != 2:
        errors.append("document.schema_version must be 2 for strict provenance")
    if not isinstance(configuration, dict):
        errors.append("document.configuration is missing")
        configuration = {}
    if configuration.get("schema_version") != 2:
        errors.append("document.configuration.schema_version must be 2")
    scheduling = configuration.get("scheduling")
    if not isinstance(scheduling, dict):
        errors.append("document.configuration.scheduling is missing")
        scheduling = {}
    trigger_age_ns = _strict_integer(
        scheduling.get("gpu_global_time_refresh_trigger_age_ns")
    )
    headroom_ns = _strict_integer(
        scheduling.get("gpu_global_time_refresh_headroom_ns")
    )
    max_clock_age_ns = _strict_integer(
        scheduling.get("gpu_global_time_refresh_max_age_ns")
    )
    if (
        trigger_age_ns is None
        or trigger_age_ns <= 0
        or headroom_ns is None
        or headroom_ns < 0
        or max_clock_age_ns is None
        or max_clock_age_ns != trigger_age_ns + headroom_ns
    ):
        errors.append("document clock refresh age contract is invalid")
        max_clock_age_ns = 0
    refresh_boundaries = scheduling.get("gpu_global_time_refresh_boundaries")
    if not isinstance(refresh_boundaries, list) or not refresh_boundaries:
        errors.append("document clock refresh boundaries are missing")
    implementation = configuration.get("implementation_sha256")
    implementation_files_checked += _validate_implementation_mapping(
        implementation,
        context="document.configuration.implementation_sha256",
        errors=errors,
    )
    configuration_identity = document.get("configuration_identity")
    if not isinstance(configuration_identity, dict):
        errors.append("document.configuration_identity is missing")
        configuration_identity = {}
    if configuration_identity.get("restart_identity_schema_version") != 2:
        errors.append(
            "document.configuration_identity.restart_identity_schema_version "
            "must be 2"
        )
    identity_implementation = configuration_identity.get(
        "implementation_sha256"
    )
    implementation_files_checked += _validate_implementation_mapping(
        identity_implementation,
        context="document.configuration_identity.implementation_sha256",
        errors=errors,
    )
    if identity_implementation != implementation:
        errors.append(
            "document configuration and configuration_identity implementation "
            "mappings disagree"
        )
    topology = configuration.get("topology")
    world_size = (
        _strict_integer(topology.get("world_size"))
        if isinstance(topology, dict)
        else None
    )
    if world_size is None or world_size <= 0:
        errors.append("document.configuration.topology.world_size is invalid")
        world_size = 0

    placement = document.get("placement")
    validation = placement.get("validation") if isinstance(placement, dict) else None
    ranks = placement.get("ranks") if isinstance(placement, dict) else None
    if not isinstance(validation, dict):
        errors.append("document.placement.validation is missing")
        validation = {}
    worker_implementation = validation.get("worker_implementation_sha256")
    implementation_files_checked += _validate_implementation_mapping(
        worker_implementation,
        context="document.placement.validation.worker_implementation_sha256",
        errors=errors,
    )
    if validation.get("world_size") != world_size:
        errors.append("document.placement.validation.world_size mismatch")
    if validation.get("worker_implementation_source_match") is not True:
        errors.append("document placement does not prove worker source agreement")
    if isinstance(implementation, dict) and isinstance(worker_implementation, dict):
        for source_path, digest in worker_implementation.items():
            if implementation.get(source_path) != digest:
                errors.append(
                    "document worker implementation mapping disagrees with "
                    f"configuration for {source_path}"
                )
    if not isinstance(ranks, list) or len(ranks) != world_size:
        errors.append(f"document.placement.ranks must contain {world_size} rows")
        ranks = []
    seen_ranks: set[int] = set()
    for index, rank_row in enumerate(ranks):
        context = f"document.placement.ranks[{index}]"
        if not isinstance(rank_row, dict):
            errors.append(f"{context}: rank metadata must be an object")
            continue
        global_rank = _strict_integer(rank_row.get("global_rank"))
        if global_rank is None or global_rank in seen_ranks:
            errors.append(f"{context}: invalid or duplicate global_rank")
        else:
            seen_ranks.add(global_rank)
        imported = rank_row.get("imported_worker_implementation_sha256")
        implementation_files_checked += _validate_implementation_mapping(
            imported,
            context=f"{context}.imported_worker_implementation_sha256",
            errors=errors,
        )
        if imported != worker_implementation:
            errors.append(f"{context}: imported worker implementation differs")
    if world_size and seen_ranks != set(range(world_size)):
        errors.append("document.placement.ranks has an incomplete global rank set")

    if document.get("clock_calibration_schema_version") != 2:
        errors.append("document.clock_calibration_schema_version must be 2")
    latest_generation = _strict_integer(document.get("clock_calibration_generation"))
    if latest_generation is None or latest_generation <= 0:
        errors.append("document.clock_calibration_generation is invalid")
    history = document.get("clock_calibration_history")
    if not isinstance(history, list) or not history:
        errors.append("document.clock_calibration_history is missing or empty")
        history = []
    history_by_generation: dict[int, dict[str, Any]] = {}
    prior_generation = -1
    for index, entry in enumerate(history):
        context = f"document.clock_calibration_history[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{context}: history entry must be an object")
            continue
        generation = _strict_integer(entry.get("generation"))
        session_index = _strict_integer(entry.get("session_index"))
        label = entry.get("label")
        if entry.get("schema_version") != 2:
            errors.append(f"{context}: schema_version must be 2")
        if generation is None or generation <= 0 or generation <= prior_generation:
            errors.append(f"{context}: generations must be positive and strictly increasing")
            continue
        prior_generation = generation
        if session_index is None or session_index < 0:
            errors.append(f"{context}: session_index is invalid")
        if not isinstance(label, str) or not label:
            errors.append(f"{context}: label is missing")
        rows = entry.get("rows")
        if entry.get("rows_sha256") != _canonical_sha256(rows):
            errors.append(f"{context}: rows_sha256 does not match retained rows")
        by_rank = _validate_clock_rows(
            rows,
            context=f"{context}.rows",
            world_size=world_size,
            errors=errors,
            expected_generation=generation,
            expected_label=label if isinstance(label, str) else None,
        )
        if by_rank is not None:
            clock_rows_checked += len(by_rank)
        history_by_generation[generation] = entry
        history_entries_checked += 1
    latest_rows = document.get("clock_calibration")
    latest_by_rank = _validate_clock_rows(
        latest_rows,
        context="document.clock_calibration",
        world_size=world_size,
        errors=errors,
        expected_generation=latest_generation,
    )
    if latest_by_rank is not None:
        clock_rows_checked += len(latest_by_rank)
    if history and latest_generation is not None:
        latest_entry = history_by_generation.get(latest_generation)
        if latest_entry is None:
            errors.append("latest clock generation is absent from history")
        elif _canonical_sha256(latest_rows) != _canonical_sha256(latest_entry.get("rows")):
            errors.append("latest clock calibration does not match latest history rows")

    restart = document.get("restart_provenance")
    sessions = restart.get("execution_sessions") if isinstance(restart, dict) else None
    if not isinstance(sessions, list) or not sessions:
        errors.append("document.restart_provenance.execution_sessions is missing")
        sessions = []
    for index, session in enumerate(sessions):
        context = f"document.restart_provenance.execution_sessions[{index}]"
        if not isinstance(session, dict):
            errors.append(f"{context}: session must be an object")
            continue
        session_index = _strict_integer(session.get("session_index"))
        session_generation = _strict_integer(
            session.get("clock_calibration_generation")
        )
        session_history = session.get("clock_calibration_history")
        if session_index != index:
            errors.append(f"{context}: session_index mismatch")
        if not isinstance(session_history, list) or not session_history:
            errors.append(f"{context}: clock_calibration_history is missing")
            continue
        session_generations: list[int] = []
        for entry_index, entry in enumerate(session_history):
            entry_context = f"{context}.clock_calibration_history[{entry_index}]"
            if not isinstance(entry, dict):
                errors.append(f"{entry_context}: history entry must be an object")
                continue
            generation = _strict_integer(entry.get("generation"))
            if generation is None:
                errors.append(f"{entry_context}: generation is invalid")
                continue
            session_generations.append(generation)
            canonical_entry = history_by_generation.get(generation)
            if canonical_entry is None or _canonical_sha256(entry) != _canonical_sha256(canonical_entry):
                errors.append(f"{entry_context}: does not match top-level history")
            if entry.get("session_index") != session_index:
                errors.append(f"{entry_context}: session_index mismatch")
        if not session_generations or session_generation != max(session_generations):
            errors.append(f"{context}: latest clock calibration generation mismatch")

    results = document.get("results", [])
    for result_index, result in enumerate(results):
        context = f"document.results[{result_index}].clock_calibration_history"
        flat_history = result.get("clock_calibration_history") if isinstance(result, dict) else None
        if not isinstance(flat_history, list) or not flat_history:
            errors.append(f"{context}: missing flat per-result calibration history")
            continue
        by_generation: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row_index, row in enumerate(flat_history):
            if not isinstance(row, dict):
                errors.append(f"{context}[{row_index}]: row must be an object")
                continue
            generation = _strict_integer(row.get("clock_calibration_generation"))
            if generation is None or generation <= 0:
                errors.append(f"{context}[{row_index}]: generation is invalid")
                continue
            by_generation[generation].append(row)
        for generation, generation_rows in sorted(by_generation.items()):
            by_rank = _validate_clock_rows(
                generation_rows,
                context=f"{context}[generation={generation}]",
                world_size=world_size,
                errors=errors,
                expected_generation=generation,
            )
            if by_rank is not None:
                clock_rows_checked += len(by_rank)
            canonical_entry = history_by_generation.get(generation)
            if canonical_entry is None or _canonical_sha256(generation_rows) != _canonical_sha256(canonical_entry.get("rows")):
                errors.append(
                    f"{context}[generation={generation}]: does not match top-level history"
                )
        if not by_generation:
            errors.append(f"{context}: contains no valid calibration generation")

    canonical_clock_rows = {
        generation: {
            int(row["global_rank"]): row
            for row in entry.get("rows", [])
            if isinstance(row, dict) and _strict_integer(row.get("global_rank")) is not None
        }
        for generation, entry in history_by_generation.items()
    }
    age_rows_checked = [0]
    clock_reference_rows_checked = _verify_clock_row_references(
        document,
        context="document",
        history_rows=canonical_clock_rows,
        max_clock_age_ns=max_clock_age_ns,
        age_rows_checked=age_rows_checked,
        errors=errors,
    )
    retained_digest_count = _verify_retained_digest_pairs(
        document, context="document", errors=errors
    )
    return {
        "schema_version": 1,
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "implementation_files_checked": implementation_files_checked,
        "retained_digest_count_checked": retained_digest_count,
        "clock_rows_checked": clock_rows_checked,
        "clock_history_entries_checked": history_entries_checked,
        "clock_reference_rows_checked": clock_reference_rows_checked,
        "clock_age_rows_checked": age_rows_checked[0],
    }


def _numeric_mapping_equal(left: Any, right: Any) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if {str(key) for key in left} != {str(key) for key in right}:
        return False
    try:
        return all(
            math.isclose(
                float(left[key]),
                float(right.get(key, right.get(str(key)))),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for key in left
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _expected_treatment_mode(row: dict[str, Any]) -> str:
    target = row.get("requested_overlap_pct")
    victim = row.get("victim_index")
    dispatch = row.get("dispatch_mode")
    if target is None or victim is None or not dispatch:
        return ""
    return f"coverage_{float(target):g}_v{int(victim)}_{dispatch}"


def _verify_v2_consensus(
    consensus: Any,
    payload: Any,
    *,
    label: str,
    errors: list[str],
) -> str | None:
    if not isinstance(consensus, dict):
        errors.append(f"missing {label} consensus")
        return None
    if consensus.get("label") != label:
        errors.append(f"{label} consensus label mismatch")
    world_size = _int(consensus.get("world_size"))
    digests = consensus.get("actor_decision_digests")
    reported = consensus.get("decision_digest_sha256")
    expected = _canonical_sha256(payload) if isinstance(payload, dict) else None
    if world_size is None or world_size <= 0:
        errors.append(f"{label} consensus has invalid world_size")
    if not isinstance(digests, dict) or world_size is None or set(digests) != {
        str(rank) for rank in range(world_size)
    }:
        errors.append(f"{label} consensus rank set mismatch")
    elif any(value != reported for value in digests.values()):
        errors.append(f"{label} consensus actors disagree")
    if consensus.get("all_ranks_agree") is not True:
        errors.append(f"{label} consensus does not declare agreement")
    if expected is None or reported != expected:
        errors.append(f"{label} consensus digest does not match payload")
    records = consensus.get("actor_decision_records")
    if not isinstance(records, list) or world_size is None or len(records) != world_size:
        errors.append(f"{label} consensus decision record count mismatch")
    else:
        ranks = {_int(record.get("global_rank")) for record in records}
        if ranks != set(range(world_size)):
            errors.append(f"{label} consensus decision record rank mismatch")
        if any(record.get("decision_digest_sha256") != reported for record in records):
            errors.append(f"{label} consensus decision record digest mismatch")
        decision_times = [_int(record.get("decision_global_ns")) for record in records]
        if any(value is None for value in decision_times):
            errors.append(f"{label} consensus decision timestamp missing")
        else:
            decision_times_int = [int(value) for value in decision_times]
            if consensus.get("decision_global_ns_min") != min(decision_times_int):
                errors.append(f"{label} consensus minimum decision time mismatch")
            if consensus.get("decision_global_ns_max") != max(decision_times_int):
                errors.append(f"{label} consensus maximum decision time mismatch")
    return reported if _sha256_string(reported) else None


def _verify_legacy_actor_consensus(
    container: Any,
    *,
    world_size: int,
    context: str,
    errors: list[str],
) -> str | None:
    if not isinstance(container, dict):
        errors.append(f"missing {context}")
        return None
    digests = container.get("actor_decision_digests")
    records = container.get("actor_decision_records")
    if not isinstance(digests, dict) or set(digests) != {
        str(rank) for rank in range(world_size)
    }:
        errors.append(f"{context} rank digest set mismatch")
        return None
    unique = set(digests.values())
    if len(unique) != 1 or not _sha256_string(next(iter(unique), None)):
        errors.append(f"{context} actors disagree")
        return None
    digest = next(iter(unique))
    if not isinstance(records, list) or len(records) != world_size:
        errors.append(f"{context} record count mismatch")
    else:
        ranks = {_int(record.get("global_rank")) for record in records}
        if ranks != set(range(world_size)):
            errors.append(f"{context} record rank set mismatch")
        if any(record.get("decision_digest_sha256") != digest for record in records):
            errors.append(f"{context} record digest mismatch")
        if container.get("actor_decisions_sha256") != _canonical_sha256(records):
            errors.append(f"{context} records hash mismatch")
    return digest


def _verify_confirmation(
    confirmation: Any,
    row: dict[str, Any],
    evidence: dict[str, Any],
    anchor: dict[str, Any],
    errors: list[str],
) -> tuple[str | None, str | None]:
    if not isinstance(confirmation, dict):
        errors.append("missing corrected-delay K3 confirmation")
        return None, None
    if confirmation.get("schema_version") != 1:
        errors.append("confirmation schema_version mismatch")
    if confirmation.get("policy") != (
        "corrected_delay_common_focal_geometry_k3_median_two_of_three_v1"
    ):
        errors.append("confirmation policy mismatch")
    if confirmation.get("repetitions") != 3:
        errors.append("confirmation is not K3")
    if confirmation.get("minimum_passing_repetitions") != 2:
        errors.append("confirmation minimum passing count is not two")
    if confirmation.get("unscored") is not True:
        errors.append("confirmation is not declared unscored")
    overlaps = confirmation.get(
        "measured_counterfactual_overlap_pct_by_repetition"
    )
    measured_errors = confirmation.get("measured_geometry_error_pct_by_repetition")
    if not isinstance(overlaps, list) or len(overlaps) != 3:
        errors.append("confirmation overlap vector is not K3")
        overlaps = []
    if not isinstance(measured_errors, list) or len(measured_errors) != 3:
        errors.append("confirmation error vector is not K3")
        measured_errors = []
    try:
        overlaps_float = [float(value) for value in overlaps]
        errors_float = [float(value) for value in measured_errors]
    except (TypeError, ValueError, OverflowError):
        overlaps_float = []
        errors_float = []
        errors.append("confirmation vectors contain non-numeric values")
    if any(not math.isfinite(value) for value in overlaps_float + errors_float):
        errors.append("confirmation vectors contain non-finite values")
    target = row.get("requested_overlap_pct")
    if target is not None and len(overlaps_float) == len(errors_float) == 3:
        if any(
            not math.isclose(
                error, overlap - float(target), rel_tol=0.0, abs_tol=1e-8
            )
            for overlap, error in zip(overlaps_float, errors_float)
        ):
            errors.append("confirmation error vector does not match target")
    if len(errors_float) == 3:
        median_error = statistics.median(errors_float)
        median_overlap = statistics.median(overlaps_float)
        passing_count = sum(abs(value) <= 10.0 for value in errors_float)
        robust = abs(median_error) <= 10.0 and passing_count >= 2
        if not math.isclose(
            float(confirmation.get("median_measured_geometry_error_pct", math.inf)),
            median_error,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append("confirmation median error mismatch")
        if not math.isclose(
            float(
                confirmation.get(
                    "median_measured_counterfactual_overlap_pct", math.inf
                )
            ),
            median_overlap,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            errors.append("confirmation median overlap mismatch")
        if confirmation.get("passing_repetition_count") != passing_count:
            errors.append("confirmation passing count mismatch")
        if confirmation.get("robustly_within_tolerance") is not robust:
            errors.append("confirmation robustness mismatch")
    else:
        robust = False
    expected_disposition = (
        RUNTIME_START_GEOMETRY_CONFIRMED
        if robust else RUNTIME_START_GEOMETRY_UNRESOLVED
    )
    reported_disposition = confirmation.get(
        "runtime_start_geometry_disposition", expected_disposition
    )
    if reported_disposition != expected_disposition:
        errors.append("confirmation disposition disagrees with K3 reconstruction")
    if row.get("runtime_start_geometry_disposition") != expected_disposition:
        errors.append("sample disposition disagrees with K3 reconstruction")
    resolution = confirmation.get("resolution_status")
    constraint_limited = confirmation.get(
        "constraint_limited", anchor.get("constraint_limited", False)
    ) is True
    expected_resolution = (
        "constraint_limited_confirmed_within_10_pct_points"
        if robust and constraint_limited
        else "corrected_delay_confirmed"
        if robust
        else "corrected_delay_confirmation_unresolved"
    )
    if resolution != expected_resolution:
        errors.append("confirmation resolution disagrees with K3 reconstruction")
    for owner, value in (
        ("anchor", anchor.get("resolution_status")),
        ("evidence", evidence.get("start_anchor_resolution_status")),
        ("sample", row.get("start_anchor_resolution_status")),
    ):
        if value not in (None, resolution):
            errors.append(f"{owner} start-anchor resolution mismatch")
    if anchor.get("confirmation_robustly_within_tolerance") not in (None, robust):
        errors.append("anchor confirmation robustness alias mismatch")
    if evidence.get("start_anchor_confirmation_robustly_within_tolerance") not in (
        None, robust
    ):
        errors.append("evidence confirmation robustness alias mismatch")
    return expected_disposition, str(resolution) if resolution is not None else None


def _verify_adjacent_pre_rebased_row(row: dict[str, Any]) -> list[str]:
    """Independently reconstruct the retained adjacent-pre/K3 contract."""

    errors: list[str] = []
    evidence = row.get("delay_rebase_evidence")
    if not isinstance(evidence, dict):
        return ["missing delay_rebase_evidence"]
    reported_hash = row.get("delay_rebase_evidence_sha256")
    if reported_hash != _canonical_sha256(evidence):
        errors.append("delay_rebase_evidence hash mismatch")
    if evidence.get("schema_version") != 1:
        errors.append("delay-rebase evidence schema_version mismatch")
    if evidence.get("sample_index") != row.get("sample_index"):
        errors.append("delay-rebase sample identity mismatch")
    expected_mode = _expected_treatment_mode(row)
    if not expected_mode or evidence.get("treatment_mode") != expected_mode:
        errors.append("delay-rebase treatment identity mismatch")
    disposition = row.get("runtime_start_geometry_disposition")
    policy = evidence.get("policy")
    if disposition == RUNTIME_START_GEOMETRY_STATIC:
        if not row.get("runtime_start_geometry_static_exemption_proven"):
            errors.append("static row lacks exact duration-bound proof")
        if not (
            policy == "static_calibrated_delays_v1"
            and evidence.get("dynamic_rebase_applied") is False
            and evidence.get("static_policy_reason") == "duration_bound_infeasible"
        ):
            errors.append("static delay-rebase evidence mismatch")
        if row.get("runtime_start_geometry_disposition_explicit") is True:
            for field, expected in (
                ("target_feasible_from_duration", False),
                ("target_feasible_after_constraints", False),
                ("target_feasible_after_runtime_calibration", False),
                (
                    "runtime_geometry_calibration_status",
                    "isolated_duration_bound_infeasible",
                ),
                ("runtime_start_geometry_disposition_explicit", True),
                ("runtime_start_geometry_disposition", RUNTIME_START_GEOMETRY_STATIC),
                ("runtime_start_geometry_treatment_executed", True),
                ("runtime_start_geometry_scoring_eligible", False),
            ):
                if evidence.get(field) != expected:
                    errors.append(f"static evidence {field} mismatch")
        return errors

    if policy not in {
        "adjacent_pre_later_group_gap_rebase_v1",
        "adjacent_pre_runtime_start_rebase_v2",
    }:
        errors.append("unsupported adjacent-pre dynamic policy")
        return errors
    if evidence.get("dynamic_rebase_applied") is not True:
        errors.append("dynamic adjacent-pre rebase was not applied")
    if evidence.get("static_policy_reason") is not None:
        errors.append("dynamic adjacent-pre row has a static reason")
    selected_delays = evidence.get("selected_calibration_delays_ms")
    final_delays = evidence.get("final_delays_ms")
    if not isinstance(selected_delays, dict) or not selected_delays:
        errors.append("missing selected calibration delays")
    if not isinstance(final_delays, dict) or not final_delays:
        errors.append("missing final delays")
    anchor = evidence.get("start_anchor_evidence")
    if not isinstance(anchor, dict):
        errors.append("missing start-anchor evidence")
        return errors
    if evidence.get("start_anchor_evidence_sha256") != _canonical_sha256(anchor):
        errors.append("start-anchor evidence hash mismatch")
    if anchor.get("schema_version") != 1:
        errors.append("start-anchor schema_version mismatch")
    expected_anchor_policy = (
        "same_dispatch_k3_preliminary_residual_plus_k3_corrected_confirmation_v2"
    )
    if anchor.get("policy") != expected_anchor_policy:
        errors.append("start-anchor policy mismatch")
    if evidence.get("start_anchor_policy") != expected_anchor_policy:
        errors.append("delay-rebase start-anchor policy mismatch")
    if anchor.get("repetitions") != 3 or anchor.get("unscored") is not True:
        errors.append("preliminary start-anchor is not unscored K3")
    if anchor.get("sample_index") != row.get("sample_index"):
        errors.append("start-anchor sample identity mismatch")
    if anchor.get("treatment_mode") != expected_mode:
        errors.append("start-anchor treatment identity mismatch")
    if not _numeric_mapping_equal(anchor.get("final_delays_ms"), final_delays):
        errors.append("start-anchor final delays mismatch")
    confirmation = anchor.get("confirmation")
    if isinstance(confirmation, dict) and confirmation.get(
        "corrected_delays_ms"
    ) is not None and not _numeric_mapping_equal(
        confirmation.get("corrected_delays_ms"), final_delays
    ):
        errors.append("confirmation corrected delays mismatch")
    configured_gap = _float(anchor.get("configured_later_group_gap_ms"))
    actual_gaps = anchor.get("actual_focal_gpu_start_gaps_ms")
    residuals = anchor.get("launch_residuals_ms")
    if (
        configured_gap is None
        or not isinstance(actual_gaps, list)
        or len(actual_gaps) != 3
        or not isinstance(residuals, list)
        or len(residuals) != 3
    ):
        errors.append("start-anchor preliminary gap/residual vectors are not K3")
    else:
        try:
            if any(
                not math.isclose(
                    float(residual),
                    float(gap) - configured_gap,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for gap, residual in zip(actual_gaps, residuals)
            ):
                errors.append("start-anchor launch residuals do not reconstruct")
        except (TypeError, ValueError, OverflowError):
            errors.append("start-anchor gap/residual vectors are non-numeric")
    expected_disposition, resolution = _verify_confirmation(
        confirmation, row, evidence, anchor, errors
    )

    if policy == "adjacent_pre_runtime_start_rebase_v2":
        pre_rows = evidence.get("pre_rows")
        if evidence.get("pre_rows_schema_version") != 1:
            errors.append("V2 adjacent-pre rows schema_version mismatch")
        if not isinstance(pre_rows, list) or not pre_rows:
            errors.append("V2 adjacent-pre rows are missing")
        else:
            if evidence.get("pre_row_count") != len(pre_rows):
                errors.append("V2 adjacent-pre row count mismatch")
            if evidence.get("pre_rows_sha256") != _canonical_sha256(pre_rows):
                errors.append("V2 adjacent-pre rows hash mismatch")
        for owner_name, owner, latest_field in (
            (
                "preliminary",
                anchor,
                "latest_anchor_all_endpoint_complete_global_ns",
            ),
            (
                "confirmation",
                confirmation,
                "latest_all_endpoint_complete_global_ns",
            ),
        ):
            if not isinstance(owner, dict):
                continue
            focal_rows = owner.get("raw_focal_start_rows")
            completion_rows = owner.get("all_endpoint_completion_rows")
            if owner.get("raw_focal_start_rows_schema_version") != 1:
                errors.append(f"V2 {owner_name} focal-row schema mismatch")
            if not isinstance(focal_rows, list) or not focal_rows:
                errors.append(f"V2 {owner_name} focal rows are missing")
            else:
                if owner.get("raw_focal_start_row_count") != len(focal_rows):
                    errors.append(f"V2 {owner_name} focal-row count mismatch")
                if owner.get("raw_focal_start_rows_sha256") != _canonical_sha256(
                    focal_rows
                ):
                    errors.append(f"V2 {owner_name} focal-row hash mismatch")
            if owner.get("all_endpoint_completion_rows_schema_version") != 1:
                errors.append(f"V2 {owner_name} completion-row schema mismatch")
            if not isinstance(completion_rows, list) or not completion_rows:
                errors.append(f"V2 {owner_name} completion rows are missing")
            else:
                if owner.get("all_endpoint_completion_rows_sha256") != (
                    _canonical_sha256(completion_rows)
                ):
                    errors.append(f"V2 {owner_name} completion-row hash mismatch")
                completion_times = [
                    _int(item.get("host_complete_global_ns"))
                    for item in completion_rows
                ]
                if any(value is None for value in completion_times):
                    errors.append(
                        f"V2 {owner_name} completion row lacks a timestamp"
                    )
                elif owner.get(latest_field) != max(
                    int(value) for value in completion_times
                ):
                    errors.append(
                        f"V2 {owner_name} latest completion time mismatch"
                    )
        for field, expected in (
            ("runtime_start_geometry_disposition_explicit", True),
            ("runtime_start_geometry_disposition", expected_disposition),
            ("runtime_start_geometry_treatment_executed", True),
            (
                "runtime_start_geometry_scoring_eligible",
                expected_disposition == RUNTIME_START_GEOMETRY_CONFIRMED,
            ),
            ("start_anchor_resolution_status", resolution),
            (
                "start_anchor_scoring_allowed",
                expected_disposition == RUNTIME_START_GEOMETRY_CONFIRMED,
            ),
        ):
            if evidence.get(field) != expected:
                errors.append(f"V2 evidence {field} mismatch")
        preliminary = anchor.get("preliminary_decision")
        if not isinstance(preliminary, dict):
            errors.append("missing preliminary decision payload")
            preliminary = {}
        if preliminary.get("sample_index") != row.get("sample_index"):
            errors.append("preliminary decision sample identity mismatch")
        if preliminary.get("treatment_mode") != expected_mode:
            errors.append("preliminary decision treatment identity mismatch")
        if not _numeric_mapping_equal(
            preliminary.get("initial_delays_ms"), selected_delays
        ):
            errors.append("preliminary initial delays mismatch")
        if not _numeric_mapping_equal(
            preliminary.get("corrected_delays_ms"), final_delays
        ):
            errors.append("preliminary corrected delays mismatch")
        preliminary_digest = _verify_v2_consensus(
            anchor.get("preliminary_consensus"),
            preliminary,
            label="start_anchor_preliminary_correction",
            errors=errors,
        )
        final_payload = {
            "policy": (
                "corrected_delay_common_focal_geometry_k3_median_two_of_three_v1"
            ),
            "sample_index": row.get("sample_index"),
            "treatment_mode": expected_mode,
            "corrected_delays_ms": final_delays,
            "resolution_status": resolution,
            "runtime_start_geometry_disposition": expected_disposition,
            "passing_repetition_count": (
                confirmation.get("passing_repetition_count")
                if isinstance(confirmation, dict) else None
            ),
            "median_measured_geometry_error_pct": (
                confirmation.get("median_measured_geometry_error_pct")
                if isinstance(confirmation, dict) else None
            ),
        }
        final_digest = _verify_v2_consensus(
            anchor.get("post_confirmation_consensus"),
            final_payload,
            label="start_anchor_post_confirmation_decision",
            errors=errors,
        )
        if preliminary_digest is not None and preliminary_digest == final_digest:
            errors.append("preliminary and post-confirmation decisions are not distinct")
        preliminary_consensus = anchor.get("preliminary_consensus")
        post_consensus = anchor.get("post_confirmation_consensus")
        preliminary_complete = anchor.get(
            "latest_anchor_all_endpoint_complete_global_ns"
        )
        confirmation_complete = (
            confirmation.get("latest_all_endpoint_complete_global_ns")
            if isinstance(confirmation, dict) else None
        )
        if not (
            isinstance(preliminary_consensus, dict)
            and _int(preliminary_consensus.get("decision_global_ns_min")) is not None
            and _int(preliminary_complete) is not None
            and int(preliminary_consensus["decision_global_ns_min"])
            > int(preliminary_complete)
        ):
            errors.append("V2 preliminary decision does not follow K3 completion")
        if not (
            isinstance(post_consensus, dict)
            and _int(post_consensus.get("decision_global_ns_min")) is not None
            and _int(confirmation_complete) is not None
            and int(post_consensus["decision_global_ns_min"])
            > int(confirmation_complete)
        ):
            errors.append("V2 post-confirmation decision is not causally ordered")
        confirmation_rows = anchor.get("confirmation_rows")
        if anchor.get("confirmation_rows_schema_version") != 1:
            errors.append("V2 retained confirmation rows schema mismatch")
        if not isinstance(confirmation_rows, list) or not confirmation_rows:
            errors.append("V2 retained confirmation rows are missing")
        else:
            if anchor.get("confirmation_row_count") != len(confirmation_rows):
                errors.append("V2 retained confirmation row count mismatch")
            if anchor.get("confirmation_rows_sha256") != _canonical_sha256(
                confirmation_rows
            ):
                errors.append("V2 retained confirmation-row hash mismatch")
    else:
        topology = evidence.get("topology")
        world_size = _int(topology.get("world_size")) if isinstance(topology, dict) else None
        if world_size is None or world_size <= 0:
            errors.append("legacy evidence has invalid topology world_size")
            world_size = 0
        preliminary_consensus = evidence.get(
            "start_anchor_preliminary_correction_consensus"
        )
        preliminary_digest = None
        if not isinstance(preliminary_consensus, dict):
            errors.append("missing legacy preliminary correction consensus")
        else:
            payload = preliminary_consensus.get("preliminary_payload")
            preliminary_digest = preliminary_consensus.get(
                "preliminary_payload_sha256"
            )
            if not isinstance(payload, dict) or preliminary_digest != _canonical_sha256(payload):
                errors.append("legacy preliminary payload hash mismatch")
            records = preliminary_consensus.get("records")
            if not isinstance(records, list) or len(records) != world_size:
                errors.append("legacy preliminary consensus record count mismatch")
            else:
                if {_int(record.get("global_rank")) for record in records} != set(
                    range(world_size)
                ):
                    errors.append("legacy preliminary consensus rank set mismatch")
                if any(
                    record.get("preliminary_payload_sha256") != preliminary_digest
                    for record in records
                ):
                    errors.append("legacy preliminary consensus actors disagree")
                if preliminary_consensus.get("records_sha256") != _canonical_sha256(records):
                    errors.append("legacy preliminary consensus records hash mismatch")
            if isinstance(payload, dict) and not _numeric_mapping_equal(
                payload.get("corrected_delays_ms"), final_delays
            ):
                errors.append("legacy preliminary corrected delays mismatch")
        adjacent_digest = _verify_legacy_actor_consensus(
            evidence.get("adjacent_pre_decision_provenance"),
            world_size=world_size,
            context="adjacent-pre decision consensus",
            errors=errors,
        )
        final_digest = _verify_legacy_actor_consensus(
            evidence,
            world_size=world_size,
            context="post-confirmation decision consensus",
            errors=errors,
        )
        retained = {preliminary_digest, adjacent_digest, final_digest}
        if None in retained or len(retained) != 3:
            errors.append("legacy adjacent/preliminary/final decisions are not distinct")
        for owner in (anchor, confirmation):
            if not isinstance(owner, dict):
                continue
            for rows_field, hash_field in (
                ("raw_focal_start_rows", "raw_focal_start_rows_sha256"),
                ("all_endpoint_completion_rows", "all_endpoint_completion_rows_sha256"),
            ):
                retained_rows = owner.get(rows_field)
                if not isinstance(retained_rows, list) or owner.get(hash_field) != _canonical_sha256(retained_rows):
                    errors.append(f"legacy retained {rows_field} hash mismatch")
        pre_anchor = evidence.get("pre_anchor_final_delays_ms")
        applied_delta = _float(evidence.get("applied_additive_rebase_delta_ms"))
        requested_delta = _float(
            evidence.get("requested_additive_rebase_delta_ms")
        )
        later = evidence.get("later_group_operation_indices")
        if (
            not isinstance(pre_anchor, dict)
            or applied_delta is None
            or requested_delta is None
            or not isinstance(later, list)
        ):
            errors.append("legacy adjacent-pre delay inputs are incomplete")
        elif isinstance(selected_delays, dict):
            pre_anchor_delta = (
                applied_delta
                if evidence.get("additive_rebase_delta_clipped") is True
                else requested_delta
            )
            reconstructed = {
                str(index): float(value) + (
                    pre_anchor_delta
                    if int(index) in {int(item) for item in later}
                    else 0.0
                )
                for index, value in selected_delays.items()
            }
            if not _numeric_mapping_equal(pre_anchor, reconstructed):
                errors.append("legacy adjacent-pre rebased delays do not reconstruct")
        timing_fields = (
            "decision_global_ns_min",
            "decision_global_ns_max",
            "decision_clock_uncertainty_ns",
            "latest_pre_complete_global_ns",
            "latest_pre_clock_uncertainty_ns",
            "pre_to_decision_conservative_margin_ns",
            "start_anchor_to_decision_conservative_margin_ns",
            "earliest_treatment_launch_global_ns",
            "earliest_treatment_launch_clock_uncertainty_ns",
            "decision_to_treatment_launch_conservative_margin_ns",
            "earliest_treatment_gpu_start_global_ns",
            "earliest_treatment_gpu_start_uncertainty_ns",
            "decision_to_treatment_gpu_start_conservative_margin_ns",
        )
        if all(_int(evidence.get(field)) is not None for field in timing_fields):
            decision_min = int(evidence["decision_global_ns_min"])
            decision_max = int(evidence["decision_global_ns_max"])
            decision_uncertainty = int(evidence["decision_clock_uncertainty_ns"])
            pre_margin = decision_min - decision_uncertainty - (
                int(evidence["latest_pre_complete_global_ns"])
                + int(evidence["latest_pre_clock_uncertainty_ns"])
            )
            confirmation_complete = (
                confirmation.get("latest_all_endpoint_complete_global_ns")
                if isinstance(confirmation, dict) else None
            )
            confirmation_uncertainty = (
                confirmation.get("latest_clock_uncertainty_ns")
                if isinstance(confirmation, dict) else None
            )
            anchor_margin = None if None in (
                confirmation_complete, confirmation_uncertainty
            ) else decision_min - decision_uncertainty - (
                int(confirmation_complete) + int(confirmation_uncertainty)
            )
            launch_margin = (
                int(evidence["earliest_treatment_launch_global_ns"])
                - int(evidence["earliest_treatment_launch_clock_uncertainty_ns"])
                - (decision_max + decision_uncertainty)
            )
            gpu_margin = (
                int(evidence["earliest_treatment_gpu_start_global_ns"])
                - int(evidence["earliest_treatment_gpu_start_uncertainty_ns"])
                - (decision_max + decision_uncertainty)
            )
            for label, reconstructed, reported in (
                ("pre-to-decision", pre_margin, evidence.get("pre_to_decision_conservative_margin_ns")),
                ("anchor-to-decision", anchor_margin, evidence.get("start_anchor_to_decision_conservative_margin_ns")),
                ("decision-to-launch", launch_margin, evidence.get("decision_to_treatment_launch_conservative_margin_ns")),
                ("decision-to-GPU-start", gpu_margin, evidence.get("decision_to_treatment_gpu_start_conservative_margin_ns")),
            ):
                if reconstructed is None or reconstructed != reported or reconstructed <= 0:
                    errors.append(f"legacy {label} causal margin mismatch")
    return errors


def _normalize_sample(
    row: dict[str, Any],
    correct: bool | None,
    *,
    result_index: int | None = None,
    expected_operation_indices: set[str] | None = None,
) -> dict[str, Any]:
    case_id = str(row.get("case_id") or "")
    case_key = str(row.get("case_key") or case_id)
    dispatch_mode = str(row.get("dispatch_mode") or "")
    if not dispatch_mode:
        dispatch = str(row.get("dispatch_order") or row.get("stage_mode") or "")
        if "aggressor_first" in dispatch:
            dispatch_mode = "aggressor_first"
        elif "victim_first" in dispatch:
            dispatch_mode = "victim_first"
        else:
            dispatch_mode = dispatch
    row_correct = _bool(row.get("correct"))
    counterfactual_schema_fields = (
        "geometry_validity_error_pct",
        "scheduled_geometry_error_pct",
        "counterfactual_geometry_error_pct",
        "geometry_validity_basis",
        "geometry_baseline_drifts_pct",
        "geometry_baseline_max_abs_drift_pct",
        "geometry_baseline_stability_basis",
    )
    counterfactual_error_present = any(
        row.get(field) not in (None, "", {})
        for field in counterfactual_schema_fields
    )
    geometry_validity_error = _float(
        row.get(
            "geometry_validity_error_pct",
            row.get("scheduled_geometry_error_pct"),
        )
        if counterfactual_error_present
        else row.get("geometry_error_pct")
    )
    scheduled_error = _float(row.get("scheduled_geometry_error_pct"))
    counterfactual_error = _float(
        row.get("counterfactual_geometry_error_pct")
    )
    geometry_alias_values = [
        value
        for value in (
            geometry_validity_error,
            scheduled_error,
            counterfactual_error,
        )
        if value is not None
    ]
    geometry_integrity_ok = (
        not counterfactual_error_present
        or (
            geometry_validity_error is not None
            and scheduled_error is not None
            and counterfactual_error is not None
            and all(
                math.isclose(
                    value,
                    geometry_validity_error,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for value in geometry_alias_values
            )
        )
    )
    per_operation_drifts = _float_mapping(
        row.get("geometry_baseline_drifts_pct")
    )
    reported_max_drift = _float(
        row.get("geometry_baseline_max_abs_drift_pct")
    )
    recomputed_max_drift = (
        None
        if per_operation_drifts is None
        else max(abs(value) for value in per_operation_drifts.values())
    )
    geometry_baseline_operation_set_integrity_ok = bool(
        not counterfactual_error_present
        or (
            per_operation_drifts is not None
            and (
                expected_operation_indices is None
                or set(per_operation_drifts) == expected_operation_indices
            )
        )
    )
    geometry_baseline_integrity_ok = (
        not counterfactual_error_present
        or (
            geometry_baseline_operation_set_integrity_ok
            and
            recomputed_max_drift is not None
            and reported_max_drift is not None
            and math.isclose(
                recomputed_max_drift,
                reported_max_drift,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
    )
    runtime_start_geometry = _runtime_start_geometry_fields(row)
    return {
        "case_id": case_id,
        "case_key": case_key,
        "seq_len": _int(row.get("seq_len")),
        "sample_index": _int(row.get("sample_index")),
        "dispatch_mode": dispatch_mode,
        "victim_index": _int(row.get("victim_index")),
        "victim_family": str(row.get("victim_family") or ""),
        "aggressor_family": str(row.get("aggressor_family") or ""),
        "requested_overlap_pct": _float(row.get("requested_overlap_pct")),
        "feasible": _bool(row.get("target_feasible_from_isolated")),
        "valid": _bool(row.get("valid")),
        # Canonical validation error prefers the new pre-treatment geometry.
        # The legacy concurrent field is retained separately for reporting.
        "geometry_error_pct": geometry_validity_error,
        "geometry_validity_error_pct": geometry_validity_error,
        "concurrent_geometry_error_pct": _float(row.get("geometry_error_pct")),
        "scheduled_geometry_error_pct": _float(
            row.get("scheduled_geometry_error_pct")
        ),
        "counterfactual_geometry_error_pct": counterfactual_error,
        "counterfactual_schema_present": counterfactual_error_present,
        "geometry_integrity_ok": geometry_integrity_ok,
        "geometry_validity_basis": (
            str(row.get("geometry_validity_basis") or "")
            if counterfactual_error_present
            else "legacy_concurrent_intervals_fallback"
        ),
        "geometry_baseline_max_abs_drift_pct": _float(
            row.get("geometry_baseline_max_abs_drift_pct")
        ),
        "geometry_baseline_drifts_pct": per_operation_drifts,
        "geometry_baseline_recomputed_max_abs_drift_pct": (
            recomputed_max_drift
        ),
        "geometry_baseline_integrity_ok": (
            geometry_baseline_integrity_ok
        ),
        "geometry_baseline_operation_set_integrity_ok": (
            geometry_baseline_operation_set_integrity_ok
        ),
        "expected_geometry_operation_indices": (
            None
            if expected_operation_indices is None
            else sorted(expected_operation_indices, key=int)
        ),
        "geometry_baseline_stability_basis": str(
            row.get("geometry_baseline_stability_basis") or ""
        ),
        "geometry_tolerance_pct": _float(row.get("geometry_tolerance_pct")),
        "baseline_drift_pct": _float(row.get("baseline_drift_pct")),
        "correct": row_correct if row_correct is not None else correct,
        "row_correct": row_correct,
        "result_correct": correct,
        "victim_isolated_us": _float(row.get("victim_isolated_us")),
        "delay_rebase_evidence": row.get("delay_rebase_evidence"),
        "delay_rebase_evidence_sha256": row.get(
            "delay_rebase_evidence_sha256"
        ),
        "start_anchor_resolution_status": row.get(
            "start_anchor_resolution_status"
        ),
        "start_anchor_scoring_allowed": _bool(
            row.get("start_anchor_scoring_allowed")
        ),
        "_result_index": result_index,
        **runtime_start_geometry,
    }


def load_samples(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and normalize raw/enriched result JSON or a sample CSV."""

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [_normalize_sample(dict(row), None) for row in csv.DictReader(handle)]
        return NormalizedSamples(rows), {
            "input_format": "samples_csv",
            "source": str(path),
        }

    document = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(document, dict) and "results" in document:
        results = document["results"]
    elif isinstance(document, dict) and "assignment" in document:
        results = [document]
    else:
        raise ValueError(
            "JSON input must be a result object or a document containing 'results'"
        )
    rows: list[dict[str, Any]] = []
    adaptive_results: list[dict[str, Any]] = []
    result_metadata: list[dict[str, Any]] = []
    document_scheduling = (
        document.get("configuration", {}).get("scheduling", {})
        if isinstance(document, dict)
        else {}
    )
    document_sampling = (
        document.get("configuration", {}).get("sampling", {})
        if isinstance(document, dict)
        else {}
    )
    document_expects_adaptive = bool(
        isinstance(document_sampling, dict)
        and document_sampling.get("mode") == "adaptive"
    )
    nested_document_geometry = (
        document_scheduling.get("geometry_feedback", {})
        if isinstance(document_scheduling, dict)
        else {}
    )
    document_declares_v2 = bool(
        isinstance(document_scheduling, dict)
        and (
            document_scheduling.get("geometry_validity_basis")
            == COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
            or (
                isinstance(nested_document_geometry, dict)
                and nested_document_geometry.get("validity_basis")
                == COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
            )
        )
    )
    for result_index, result in enumerate(results):
        correct = result_correctness_status(result)
        participants = result.get("assignment", {}).get("participants")
        expected_operation_indices = (
            {str(index) for index in range(len(participants))}
            if isinstance(participants, list) and participants
            else None
        )
        rows.extend(
            _normalize_sample(
                dict(sample),
                correct,
                result_index=result_index,
                expected_operation_indices=expected_operation_indices,
            )
            for sample in result["samples"]
        )
        stage_specs = result.get("stage_specs")
        result_declares_v2 = bool(
            document_declares_v2
            or (
                isinstance(stage_specs, list)
                and any(
                    _stage_declares_counterfactual_geometry(stage_spec)
                    for stage_spec in stage_specs
                )
            )
            or any(
                any(
                    sample.get(field) not in (None, "", {})
                    for field in (
                        "geometry_validity_error_pct",
                        "scheduled_geometry_error_pct",
                        "counterfactual_geometry_error_pct",
                        "geometry_validity_basis",
                        "geometry_baseline_drifts_pct",
                        "geometry_baseline_max_abs_drift_pct",
                        "geometry_baseline_stability_basis",
                    )
                )
                for sample in result.get("samples", [])
                if isinstance(sample, dict)
            )
        )
        result_metadata.append({
            "result_index": result_index,
            "declares_counterfactual_geometry": result_declares_v2,
        })
        if (
            result.get("adaptive_sampling") is not None
            or document_expects_adaptive
        ):
            adaptive_results.append({
                "result_index": result_index,
                "case_key": str(
                    result.get("assignment", {}).get("case_key")
                    or result.get("assignment", {}).get("case_id")
                    or ""
                ),
                "diagnostics": result.get("adaptive_sampling"),
                "stage_specs": stage_specs,
                "expected_operation_indices": expected_operation_indices,
                "correctness": result.get("correctness"),
                "declares_counterfactual_geometry": result_declares_v2,
                "adaptive_diagnostics_required": document_expects_adaptive,
                "scheduling": document_scheduling,
            })
    return NormalizedSamples(
        rows,
        adaptive_results=adaptive_results,
        result_metadata=result_metadata,
        strict_document_provenance=_validate_strict_document_provenance(document),
    ), {
        "input_format": "result_json",
        "source": str(path),
        "schema_version": document.get("schema_version")
        if isinstance(document, dict)
        else None,
    }


def _orientation(row: dict[str, Any]) -> tuple[int | None, str, str]:
    return (row["victim_index"], row["victim_family"], row["aggressor_family"])


def _target_key(value: float) -> str:
    return f"{value:g}"


def _stratum_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["seq_len"],
        row["case_key"],
        *_orientation(row),
        row["dispatch_mode"],
        row["requested_overlap_pct"],
    )


def _geometry_pass(row: dict[str, Any], default_tolerance_pct: float) -> bool | None:
    if row.get("geometry_integrity_ok", True) is not True:
        return False
    basis = row.get("geometry_validity_basis") or (
        "legacy_concurrent_intervals_fallback"
    )
    if basis not in {
        COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS,
        "legacy_concurrent_intervals_fallback",
    }:
        return False
    error = row["geometry_error_pct"]
    if error is None:
        return None
    tolerance = row["geometry_tolerance_pct"]
    if tolerance is None:
        tolerance = default_tolerance_pct
    return abs(error) <= tolerance


def _geometry_baseline_pass(
    row: dict[str, Any], drift_threshold_pct: float
) -> bool | None:
    """Gate all matched durations used to construct scheduled geometry."""

    if row.get("geometry_baseline_integrity_ok", True) is not True:
        return False
    if (row.get("geometry_validity_basis") or (
        "legacy_concurrent_intervals_fallback"
    )) == "legacy_concurrent_intervals_fallback":
        return True
    if row.get("geometry_baseline_stability_basis") != (
        GEOMETRY_BASELINE_STABILITY_BASIS
    ):
        return False
    drift = row.get("geometry_baseline_max_abs_drift_pct")
    if drift is None:
        return None
    return drift <= drift_threshold_pct


def _sample_is_accepted(
    row: dict[str, Any],
    *,
    geometry_tolerance_pct: float,
    drift_threshold_pct: float,
) -> bool:
    """Return whether an evaluable treatment row enters the estimator."""

    return bool(
        row.get("runtime_start_geometry_scoring_eligible") is True
        and row.get("runtime_start_geometry_contract_ok") is True
        and row["feasible"] is True
        and row["valid"] is True
        and _geometry_pass(row, geometry_tolerance_pct) is True
        and _geometry_baseline_pass(row, drift_threshold_pct) is True
        and row["baseline_drift_pct"] is not None
        and abs(row["baseline_drift_pct"]) <= drift_threshold_pct
        and row["correct"] is True
    )


def _recommended_floor(rows: Sequence[dict[str, Any]]) -> int:
    durations_ms = [
        float(row["victim_isolated_us"]) / 1000.0
        for row in rows
        if row["victim_isolated_us"] is not None
    ]
    # The normative plan requires 50 samples for long (>10 ms) operations and
    # 100 for 1--10 ms operations.  Sub-millisecond operations retain the more
    # conservative 100-sample floor because they are no less noise-sensitive.
    return 50 if durations_ms and statistics.median(durations_ms) > 10.0 else 100


def _duration_floor_from_ms(duration_ms: Any) -> int | None:
    duration = _float(duration_ms)
    if duration is None or duration <= 0.0:
        return None
    return 50 if duration > 10.0 else 100


def _stage_declares_counterfactual_geometry(stage_spec: Any) -> bool:
    return bool(
        isinstance(stage_spec, dict)
        and stage_spec.get("geometry_validity_basis")
        == COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
    )


def _verify_v2_stage_duration_evidence(
    stage_spec: dict[str, Any],
    expected_operation_indices: set[str] | None,
) -> tuple[list[str], bool | None]:
    """Recompute the structural duration bound from preserved proof inputs."""

    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)

    try:
        victim_index = str(int(stage_spec["victim_index"]))
        aggressor_indices = [
            str(int(index)) for index in stage_spec["aggressor_indices"]
        ]
    except (KeyError, TypeError, ValueError, OverflowError):
        victim_index = None
        aggressor_indices = []
        fail("missing/invalid stage victim and aggressor operation indices")
    if not aggressor_indices:
        fail("duration bound requires at least one aggressor")
    if victim_index in aggressor_indices:
        fail("stage aggressor indices must exclude the victim")
    if len(aggressor_indices) != len(set(aggressor_indices)):
        fail("stage aggressor indices must be unique")
    operation_indices = (
        None
        if victim_index is None
        else {victim_index, *aggressor_indices}
    )
    if (
        expected_operation_indices is None
        or operation_indices != expected_operation_indices
        or len(aggressor_indices) + 1 != len(expected_operation_indices)
    ):
        fail("stage operation indices do not exactly match assignment participants")

    durations = _float_mapping(stage_spec.get("isolated_estimates_ms"))
    offsets = _float_mapping(stage_spec.get("natural_group_offsets_ms"))
    if durations is None:
        fail("missing/non-finite isolated_estimates_ms duration proof")
    elif expected_operation_indices is None or set(durations) != (
        expected_operation_indices
    ):
        fail("isolated_estimates_ms keys do not match assignment participants")
    elif any(duration <= 0.0 for duration in durations.values()):
        fail("isolated_estimates_ms durations must be positive")
    if offsets is None:
        fail("missing/non-finite natural_group_offsets_ms duration proof")
    elif expected_operation_indices is None or set(offsets) != (
        expected_operation_indices
    ):
        fail("natural_group_offsets_ms keys do not match assignment participants")

    try:
        requested = _float(stage_spec.get("requested_overlap_pct"))
        reported_span = _float(stage_spec.get("aggressor_span_ms"))
        reported_predicted = _float(stage_spec.get("predicted_max_overlap_pct"))
        proof_tolerance = _float(
            stage_spec.get("duration_feasibility_tolerance_pct_points")
        )
        reported_feasible = _bool(
            stage_spec.get("target_feasible_from_duration")
        )
    except (TypeError, ValueError, OverflowError):
        requested = reported_span = reported_predicted = proof_tolerance = None
        reported_feasible = None
    if proof_tolerance is None or not math.isclose(
        proof_tolerance, 1e-6, rel_tol=0.0, abs_tol=1e-15
    ):
        fail("duration feasibility tolerance must be exactly 1e-6 points")
    if stage_spec.get("duration_feasibility_formula") != (
        "min(100, max(offset_i-min_aggressor_offset+"
        "isolated_duration_i)/victim_isolated_duration*100)"
    ):
        fail("duration feasibility formula provenance is missing or invalid")

    recomputed_infeasible: bool | None = None
    if (
        victim_index is not None
        and aggressor_indices
        and durations is not None
        and offsets is not None
        and requested is not None
        and proof_tolerance is not None
        and victim_index in durations
        and all(index in durations and index in offsets for index in aggressor_indices)
    ):
        origin = min(offsets[index] for index in aggressor_indices)
        span = max(
            offsets[index] - origin + durations[index]
            for index in aggressor_indices
        )
        predicted = min(100.0, span / durations[victim_index] * 100.0)
        recomputed_infeasible = requested > predicted + proof_tolerance
        if reported_span is None or not math.isclose(
            reported_span, span, rel_tol=1e-12, abs_tol=1e-9
        ):
            fail("aggressor_span_ms does not match duration proof inputs")
        if reported_predicted is None or not math.isclose(
            reported_predicted, predicted, rel_tol=1e-12, abs_tol=1e-9
        ):
            fail("predicted_max_overlap_pct does not match duration proof inputs")
        if reported_feasible is not (not recomputed_infeasible):
            fail("target_feasible_from_duration disagrees with recomputed bound")
    else:
        fail("duration-bound proof could not be recomputed")
    return failures, recomputed_infeasible


def _verify_confirmed_stage_geometry(
    stage_spec: dict[str, Any],
    expected_operation_indices: set[str] | None,
) -> list[str]:
    """Independently reconstruct the V2 three-trial calibration decision."""

    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(message)

    def finite(mapping: dict[str, Any], field: str) -> float | None:
        try:
            value = _float(mapping.get(field))
        except (TypeError, ValueError, OverflowError):
            value = None
        if value is None:
            fail(f"missing/non-finite {field}")
        return value

    if stage_spec.get("runtime_geometry_robustly_converged") is not True:
        fail("runtime_geometry_robustly_converged is not true")
    if stage_spec.get("runtime_geometry_median_centered") is not True:
        fail("runtime_geometry_median_centered is not true")
    if stage_spec.get("runtime_geometry_calibration_status") != "converged":
        fail("runtime_geometry_calibration_status is not converged")
    if stage_spec.get("runtime_geometry_baseline_stability_basis") != (
        GEOMETRY_BASELINE_STABILITY_BASIS
    ):
        fail("invalid runtime baseline-stability basis")
    calibrated_error = finite(
        stage_spec, "runtime_calibrated_scheduled_geometry_error_pct"
    )
    calibrated_counterfactual_error = finite(
        stage_spec,
        "runtime_calibrated_counterfactual_geometry_error_pct",
    )
    if (
        calibrated_error is not None
        and calibrated_counterfactual_error is not None
        and not math.isclose(
            calibrated_error,
            calibrated_counterfactual_error,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    ):
        fail("calibrated scheduled/counterfactual error aliases disagree")
    if calibrated_error is not None and abs(calibrated_error) > 2.0:
        fail("calibrated scheduled median error exceeds 2 percentage points")

    try:
        selected_shift = _float(stage_spec.get("runtime_relative_shift_ms"))
    except (TypeError, ValueError, OverflowError):
        selected_shift = None
    if selected_shift is None:
        fail("missing/non-finite runtime_relative_shift_ms")

    history = stage_spec.get("geometry_calibration_history")
    if not isinstance(history, list) or not history:
        fail("missing geometry_calibration_history")
        return failures
    try:
        selected_trial_index = _int(
            stage_spec.get("runtime_geometry_selected_trial_index")
        )
    except (TypeError, ValueError, OverflowError):
        selected_trial_index = None
    if selected_trial_index is None:
        fail("missing/invalid runtime_geometry_selected_trial_index")
        return failures
    selected: list[dict[str, Any]] = []
    for candidate in history:
        if not isinstance(candidate, dict):
            continue
        try:
            candidate_shift = _float(candidate.get("relative_shift_ms"))
            candidate_error = _float(
                candidate.get("scheduled_geometry_error_pct")
            )
        except (TypeError, ValueError, OverflowError):
            continue
        try:
            candidate_trial_index = _int(candidate.get("trial_index"))
        except (TypeError, ValueError, OverflowError):
            candidate_trial_index = None
        if (
            candidate_trial_index == selected_trial_index
            and
            candidate_shift is not None
            and candidate_error is not None
            and selected_shift is not None
            and calibrated_error is not None
            and math.isclose(
                candidate_shift, selected_shift, rel_tol=0.0, abs_tol=1e-9
            )
            and math.isclose(
                candidate_error, calibrated_error, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            selected.append(candidate)
    if len(selected) != 1:
        fail("selected runtime calibration trial is not uniquely recoverable")
        return failures

    candidate = selected[0]
    measurements = candidate.get("confirmation_measurements")
    if not isinstance(measurements, list) or len(measurements) != 3:
        fail("selected candidate must contain exactly three confirmations")
        return failures

    candidate_delays = _float_mapping(candidate.get("delays_ms"))
    if candidate_delays is None:
        fail("selected candidate lacks finite delays_ms")
    elif (
        expected_operation_indices is not None
        and set(candidate_delays) != expected_operation_indices
    ):
        fail("selected candidate delays_ms keys do not match assignment participants")
    stage_delays = _float_mapping(stage_spec.get("delays_ms"))
    if stage_delays is None:
        fail("confirmed stage lacks finite final delays_ms")
    elif (
        candidate_delays is None
        or set(stage_delays) != set(candidate_delays)
        or any(
            not math.isclose(
                stage_delays[key],
                candidate_delays[key],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for key in candidate_delays
            if key in stage_delays
        )
    ):
        fail("confirmed stage final delays_ms do not match selected candidate")
    confirmation_indices: list[int] = []
    calibration_sample_indices: list[int] = []

    scheduled_errors: list[float] = []
    all_operation_drifts: list[float] = []
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, dict):
            fail(f"confirmation[{index}] is not an object")
            continue
        try:
            measurement_confirmation_index = _int(
                measurement.get("confirmation_index")
            )
            measurement_trial_index = _int(measurement.get("trial_index"))
            measurement_sample_index = _int(
                measurement.get("calibration_sample_index")
            )
            measurement_shift = _float(measurement.get("relative_shift_ms"))
        except (TypeError, ValueError, OverflowError):
            measurement_confirmation_index = None
            measurement_trial_index = None
            measurement_sample_index = None
            measurement_shift = None
        if measurement_confirmation_index is None:
            fail(f"confirmation[{index}] lacks confirmation_index")
        else:
            confirmation_indices.append(measurement_confirmation_index)
        if measurement_sample_index is None:
            fail(f"confirmation[{index}] lacks calibration_sample_index")
        else:
            calibration_sample_indices.append(measurement_sample_index)
        if measurement_trial_index != selected_trial_index:
            fail(f"confirmation[{index}] trial_index does not match selected trial")
        if (
            measurement_shift is None
            or selected_shift is None
            or not math.isclose(
                measurement_shift, selected_shift, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            fail(f"confirmation[{index}] relative shift does not match selected shift")
        measurement_delays = _float_mapping(measurement.get("delays_ms"))
        if (
            candidate_delays is None
            or measurement_delays is None
            or set(measurement_delays) != set(candidate_delays)
            or any(
                not math.isclose(
                    measurement_delays[key],
                    candidate_delays[key],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for key in candidate_delays
                if key in measurement_delays
            )
        ):
            fail(f"confirmation[{index}] delays_ms do not match selected candidate")
        scheduled = finite(measurement, "scheduled_geometry_error_pct")
        counterfactual = finite(
            measurement, "counterfactual_geometry_error_pct"
        )
        drift_map = _float_mapping(
            measurement.get("paired_baseline_drifts_pct")
        )
        reported_max = finite(
            measurement, "geometry_baseline_max_abs_drift_pct"
        )
        if measurement.get("geometry_validity_basis") != (
            COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
        ):
            fail(f"confirmation[{index}] has invalid geometry validity basis")
        if scheduled is not None and counterfactual is not None and not math.isclose(
            scheduled, counterfactual, rel_tol=0.0, abs_tol=1e-9
        ):
            fail(f"confirmation[{index}] scheduled aliases disagree")
        if drift_map is None:
            fail(f"confirmation[{index}] lacks all-operation drift map")
        elif (
            expected_operation_indices is not None
            and set(drift_map) != expected_operation_indices
        ):
            fail(
                f"confirmation[{index}] drift-map keys do not exactly match "
                "assignment participants"
            )
        elif reported_max is not None:
            recomputed_max = max(abs(value) for value in drift_map.values())
            if not math.isclose(
                reported_max, recomputed_max, rel_tol=0.0, abs_tol=1e-9
            ):
                fail(f"confirmation[{index}] all-operation drift max disagrees")
        if scheduled is not None:
            scheduled_errors.append(scheduled)
        if reported_max is not None:
            all_operation_drifts.append(reported_max)

    if sorted(confirmation_indices) != [0, 1, 2]:
        fail("confirmation_index values must be exactly {0,1,2}")
    if len(calibration_sample_indices) != 3 or len(
        set(calibration_sample_indices)
    ) != 3:
        fail("calibration_sample_index values must identify three distinct runs")

    if len(scheduled_errors) != 3 or len(all_operation_drifts) != 3:
        return failures
    geometry_count = sum(abs(value) <= 10.0 for value in scheduled_errors)
    baseline_count = sum(value <= 5.0 for value in all_operation_drifts)
    joint_count = sum(
        abs(error) <= 10.0 and drift <= 5.0
        for error, drift in zip(scheduled_errors, all_operation_drifts)
    )
    median_error = statistics.median(scheduled_errors)
    median_drift = statistics.median(all_operation_drifts)
    if geometry_count < 2:
        fail("fewer than two of three confirmations meet geometry tolerance")
    if baseline_count < 2:
        fail("fewer than two of three confirmations meet all-operation drift tolerance")
    if joint_count < 2:
        fail("fewer than two of three confirmations jointly pass")
    if abs(median_error) > 2.0:
        fail("confirmation median scheduled error exceeds 2 percentage points")
    if median_drift > 5.0:
        fail("confirmation median all-operation drift exceeds 5 percent")

    expected_values: tuple[tuple[str, Any], ...] = (
        ("runtime_geometry_confirmation_trial_count", 3),
        ("runtime_geometry_confirmation_passing_count", joint_count),
        ("runtime_geometry_confirmation_geometry_passing_count", geometry_count),
        ("runtime_geometry_confirmation_baseline_stable_count", baseline_count),
        ("runtime_geometry_confirmation_jointly_passing_count", joint_count),
    )
    for field, expected in expected_values:
        try:
            observed = _int(stage_spec.get(field))
        except (TypeError, ValueError, OverflowError):
            observed = None
        if observed != expected:
            fail(f"{field} does not match reconstructed confirmations")
    candidate_values: tuple[tuple[str, Any], ...] = (
        ("confirmation_trial_count", 3),
        ("confirmation_passing_count", joint_count),
        ("geometry_passing_count", geometry_count),
        ("baseline_stable_count", baseline_count),
        ("jointly_passing_count", joint_count),
    )
    for field, expected in candidate_values:
        try:
            observed = _int(candidate.get(field))
        except (TypeError, ValueError, OverflowError):
            observed = None
        if observed != expected:
            fail(f"selected candidate {field} does not match confirmations")
    candidate_error = finite(candidate, "scheduled_geometry_error_pct")
    candidate_counterfactual = finite(
        candidate, "counterfactual_geometry_error_pct"
    )
    candidate_drift = finite(candidate, "geometry_baseline_max_abs_drift_pct")
    if candidate_error is not None and not math.isclose(
        candidate_error, median_error, rel_tol=0.0, abs_tol=1e-9
    ):
        fail("selected candidate scheduled error is not confirmation median")
    if candidate_counterfactual is not None and not math.isclose(
        candidate_counterfactual, median_error, rel_tol=0.0, abs_tol=1e-9
    ):
        fail("selected candidate counterfactual error is not confirmation median")
    if candidate_drift is not None and not math.isclose(
        candidate_drift, median_drift, rel_tol=0.0, abs_tol=1e-9
    ):
        fail("selected candidate drift is not confirmation median")
    stage_drift = finite(
        stage_spec, "runtime_geometry_baseline_max_abs_drift_pct"
    )
    if stage_drift is not None and not math.isclose(
        stage_drift, median_drift, rel_tol=0.0, abs_tol=1e-9
    ):
        fail("runtime baseline drift is not confirmation median")
    if candidate.get("median_centered") is not True:
        fail("selected candidate median_centered is not true")
    if candidate.get("robustly_converged") is not True:
        fail("selected candidate robustly_converged is not true")
    if candidate.get("geometry_validity_basis") != (
        COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
    ):
        fail("selected candidate has invalid geometry validity basis")
    if candidate.get("geometry_baseline_stability_basis") != (
        GEOMETRY_BASELINE_STABILITY_BASIS
    ):
        fail("selected candidate has invalid baseline-stability basis")
    return failures


def _adaptive_acceptance(
    rows: Sequence[dict[str, Any]],
    adaptive_results: Sequence[dict[str, Any]],
    *,
    max_rejection_rate_pct: float,
) -> dict[str, Any]:
    """Validate result-level adaptive stopping and accounting diagnostics."""

    errors: list[str] = []
    cells_out: list[dict[str, Any]] = []
    seen_cell_ids: set[tuple[Any, ...]] = set()

    def malformed(message: str) -> None:
        errors.append(message)

    def required_int(mapping: dict[str, Any], field: str, context: str) -> int | None:
        try:
            value = _int(mapping.get(field))
        except (TypeError, ValueError, OverflowError):
            value = None
        if value is None or value < 0:
            malformed(f"{context}: missing/invalid non-negative integer {field}")
            return None
        return value

    for adaptive_result in adaptive_results:
        result_index = adaptive_result.get("result_index")
        context = f"result[{result_index}]"
        expected_operation_indices_raw = adaptive_result.get(
            "expected_operation_indices"
        )
        expected_operation_indices = (
            {str(index) for index in expected_operation_indices_raw}
            if isinstance(expected_operation_indices_raw, (list, set, tuple))
            else None
        )
        diagnostics = adaptive_result.get("diagnostics")
        if not isinstance(diagnostics, dict):
            malformed(f"{context}: adaptive_sampling must be an object")
            continue
        if diagnostics.get("schema_version") != 1:
            malformed(f"{context}: unsupported adaptive schema_version")
        stage_specs = adaptive_result.get("stage_specs")
        stage_spec_by_cell: dict[tuple[int, float, str], dict[str, Any]] = {}
        if stage_specs is not None:
            if not isinstance(stage_specs, list):
                malformed(f"{context}: stage_specs must be an array")
                stage_specs = []
            for spec_index, stage_spec in enumerate(stage_specs):
                if not isinstance(stage_spec, dict):
                    malformed(
                        f"{context}.stage_specs[{spec_index}]: must be an object"
                    )
                    continue
                try:
                    spec_key = (
                        int(stage_spec["victim_index"]),
                        float(stage_spec["requested_overlap_pct"]),
                        str(stage_spec["dispatch_mode"]),
                    )
                except (KeyError, TypeError, ValueError, OverflowError):
                    malformed(
                        f"{context}.stage_specs[{spec_index}]: invalid cell identity"
                    )
                    continue
                if spec_key in stage_spec_by_cell:
                    malformed(
                        f"{context}.stage_specs[{spec_index}]: duplicate cell identity"
                    )
                stage_spec_by_cell[spec_key] = stage_spec
        scheduling = adaptive_result.get("scheduling")
        nested_geometry_feedback = (
            scheduling.get("geometry_feedback", {})
            if isinstance(scheduling, dict)
            else {}
        )
        result_declares_counterfactual_geometry = bool(
            isinstance(scheduling, dict)
            and (
                scheduling.get("geometry_validity_basis")
                == COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
                or (
                    isinstance(nested_geometry_feedback, dict)
                    and nested_geometry_feedback.get("validity_basis")
                    == COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
                )
            )
        ) or any(
            _stage_declares_counterfactual_geometry(stage_spec)
            for stage_spec in stage_spec_by_cell.values()
        ) or adaptive_result.get("declares_counterfactual_geometry") is True
        if result_declares_counterfactual_geometry:
            scheduling_declares_counterfactual_geometry = bool(
                isinstance(scheduling, dict)
                and (
                    scheduling.get("geometry_validity_basis")
                    == COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
                    or (
                        isinstance(nested_geometry_feedback, dict)
                        and nested_geometry_feedback.get("validity_basis")
                        == COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
                    )
                )
            )
            if not scheduling_declares_counterfactual_geometry:
                malformed(
                    f"{context}: V2 result requires configuration scheduling "
                    "counterfactual geometry provenance"
                )
            if expected_operation_indices is None:
                malformed(
                    f"{context}: V2 counterfactual methodology requires a "
                    "non-empty assignment.participants array"
                )
            correctness_rows = adaptive_result.get("correctness")
            correctness_indices: list[str] = []
            correctness_identities: set[tuple[Any, ...] | str] = set()
            correctness_valid = isinstance(correctness_rows, list)
            if correctness_valid:
                for correctness_index, correctness_row in enumerate(
                    correctness_rows
                ):
                    if not isinstance(correctness_row, dict):
                        correctness_valid = False
                        malformed(
                            f"{context}.correctness[{correctness_index}]: "
                            "must be an object"
                        )
                        continue
                    try:
                        operation_index = str(
                            int(correctness_row["operation_index"])
                        )
                    except (KeyError, TypeError, ValueError, OverflowError):
                        correctness_valid = False
                        malformed(
                            f"{context}.correctness[{correctness_index}]: "
                            "invalid operation_index"
                        )
                        continue
                    correctness_indices.append(operation_index)
                    identity_fields = (
                        "operation_index",
                        "global_rank",
                        "mode",
                        "sample_index",
                        "participant_id",
                    )
                    if all(field in correctness_row for field in identity_fields):
                        identity: tuple[Any, ...] | str = tuple(
                            correctness_row[field] for field in identity_fields
                        )
                    else:
                        identity = json.dumps(correctness_row, sort_keys=True)
                    if identity in correctness_identities:
                        correctness_valid = False
                        malformed(
                            f"{context}.correctness[{correctness_index}]: "
                            "duplicate correctness identity"
                        )
                    correctness_identities.add(identity)
                    if correctness_row.get("passed") is not True:
                        correctness_valid = False
                        malformed(
                            f"{context}.correctness[{correctness_index}]: "
                            "passed must be true"
                        )
            else:
                malformed(f"{context}: V2 result requires correctness array")
            if (
                expected_operation_indices is None
                or set(correctness_indices) != expected_operation_indices
            ):
                correctness_valid = False
                malformed(
                    f"{context}: V2 correctness operation indices must exactly "
                    "cover assignment participants without missing or extra ops"
                )
            if correctness_valid is not True:
                malformed(f"{context}: V2 result correctness is not verified")
            if stage_specs is None:
                malformed(
                    f"{context}: V2 counterfactual methodology requires stage_specs"
                )
            for spec_key, stage_spec in stage_spec_by_cell.items():
                if not _stage_declares_counterfactual_geometry(stage_spec):
                    malformed(
                        f"{context}.stage_specs[{spec_key!r}]: V2 result "
                        "requires counterfactual geometry provenance"
                    )
        summary = diagnostics.get("summary")
        cells = diagnostics.get("cells")
        if not isinstance(summary, dict):
            malformed(f"{context}: missing summary object")
            summary = {}
        if not isinstance(cells, list):
            malformed(f"{context}: missing cells array")
            cells = []
        elif not cells:
            malformed(f"{context}: adaptive cells array is empty")

        result_rows = [row for row in rows if row.get("_result_index") == result_index]
        if result_declares_counterfactual_geometry:
            for row_index, row in enumerate(result_rows):
                if (
                    row.get("result_correct") is not True
                    or row.get("row_correct") not in (None, True)
                ):
                    malformed(
                        f"{context}.samples[{row_index}]: row correctness "
                        "alias disagrees with verified V2 result correctness"
                    )
                if (
                    row.get("counterfactual_schema_present") is not True
                    or row.get("geometry_integrity_ok") is not True
                    or row.get("geometry_baseline_integrity_ok") is not True
                    or row.get("geometry_validity_basis")
                    != COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
                    or row.get("geometry_baseline_stability_basis")
                    != GEOMETRY_BASELINE_STABILITY_BASIS
                ):
                    malformed(
                        f"{context}.samples[{row_index}]: V2 result requires "
                        "the complete counterfactual geometry aliases, basis, "
                        "and all-operation drift map"
                    )
        summary_totals = {
            field: required_int(summary, field, f"{context}.summary")
            for field in (
                "cell_count",
                "active_cell_count",
                "converged_cell_count",
                "stopped_without_convergence_count",
                "total_attempts",
                "valid_samples",
                "invalid_samples",
                "discarded_after_stop",
            )
        }
        if summary.get("all_stopped") is not True:
            malformed(f"{context}.summary: all_stopped must be true")
        if summary_totals["cell_count"] != len(cells):
            malformed(f"{context}.summary: cell_count does not match cells array")

        cell_sums = defaultdict(int)
        for cell_index, diagnostic in enumerate(cells):
            cell_context = f"{context}.cells[{cell_index}]"
            if not isinstance(diagnostic, dict):
                malformed(f"{cell_context}: cell diagnostic must be an object")
                continue
            cell = diagnostic.get("cell")
            if not isinstance(cell, dict):
                malformed(f"{cell_context}: missing cell identity object")
                cell = {}
            case_key = str(cell.get("case_key") or adaptive_result.get("case_key") or "")
            victim_index = required_int(cell, "victim_index", f"{cell_context}.cell")
            try:
                target = _float(cell.get("requested_overlap_pct"))
            except (TypeError, ValueError, OverflowError):
                target = None
            dispatch_mode = str(cell.get("dispatch_mode") or "")
            if not case_key or target is None or not dispatch_mode:
                malformed(f"{cell_context}: incomplete cell identity")
            identity = (result_index, case_key, victim_index, target, dispatch_mode)
            if identity in seen_cell_ids:
                malformed(f"{cell_context}: duplicate cell identity {identity!r}")
            seen_cell_ids.add(identity)
            stage_spec = (
                stage_spec_by_cell.get((victim_index, target, dispatch_mode))
                if stage_specs is not None and victim_index is not None
                and target is not None
                else None
            )
            if stage_specs is not None and stage_spec is None:
                malformed(f"{cell_context}: missing matching stage calibration")

            total_attempts = required_int(diagnostic, "total_attempts", cell_context)
            valid_samples = required_int(diagnostic, "valid_samples", cell_context)
            invalid_samples = required_int(diagnostic, "invalid_samples", cell_context)
            discarded = required_int(diagnostic, "discarded_after_stop", cell_context)
            minimum = required_int(diagnostic, "minimum_valid_samples", cell_context)
            expected_floor = _duration_floor_from_ms(diagnostic.get("isolated_duration_ms"))
            if expected_floor is None:
                malformed(f"{cell_context}: missing/invalid isolated_duration_ms")
            explicitly_infeasible = diagnostic.get("explicitly_infeasible")
            pre_sampling_status = diagnostic.get("pre_sampling_status")
            infeasibility_proof_basis = diagnostic.get(
                "infeasibility_proof_basis"
            )
            should_continue = diagnostic.get("should_continue")
            converged = diagnostic.get("converged")
            precision_met = diagnostic.get("precision_met")
            stop_reason = diagnostic.get("stop_reason")
            for field, value in (
                ("explicitly_infeasible", explicitly_infeasible),
                ("should_continue", should_continue),
                ("converged", converged),
                ("precision_met", precision_met),
            ):
                if not isinstance(value, bool):
                    malformed(f"{cell_context}: {field} must be boolean")
            if stage_specs is not None and pre_sampling_status not in {
                "ready", "proven_infeasible", "calibration_unresolved"
            }:
                malformed(f"{cell_context}: invalid pre_sampling_status")
            stage_geometry_status = (
                stage_spec.get("geometry_cell_status")
                if isinstance(stage_spec, dict) else None
            )
            stage_calibration_evidence_verified: bool | None = None
            stage_duration_evidence_verified: bool | None = None
            stage_recomputed_duration_infeasible: bool | None = None
            if stage_specs is not None:
                expected_pre_status = {
                    "confirmed_feasible": "ready",
                    "calibration_disabled": "ready",
                    "proven_infeasible": "proven_infeasible",
                    "calibration_unresolved": "calibration_unresolved",
                }.get(stage_geometry_status)
                if expected_pre_status is None:
                    malformed(f"{cell_context}: invalid stage geometry_cell_status")
                elif pre_sampling_status != expected_pre_status:
                    malformed(
                        f"{cell_context}: controller/stage geometry status mismatch"
                    )
            if result_declares_counterfactual_geometry:
                duration_failures, stage_recomputed_duration_infeasible = (
                    _verify_v2_stage_duration_evidence(
                        stage_spec if isinstance(stage_spec, dict) else {},
                        expected_operation_indices,
                    )
                )
                stage_duration_evidence_verified = not duration_failures
                for failure in duration_failures:
                    malformed(
                        f"{cell_context}: invalid V2 stage duration evidence: "
                        f"{failure}"
                    )
                if stage_geometry_status == "calibration_disabled":
                    stage_calibration_evidence_verified = False
                    malformed(
                        f"{cell_context}: V2 counterfactual methodology cannot "
                        "accept calibration_disabled"
                    )
                elif stage_geometry_status == "confirmed_feasible":
                    calibration_failures = _verify_confirmed_stage_geometry(
                        stage_spec if isinstance(stage_spec, dict) else {},
                        expected_operation_indices,
                    )
                    stage_calibration_evidence_verified = not calibration_failures
                    for failure in calibration_failures:
                        malformed(
                            f"{cell_context}: invalid confirmed-stage "
                            f"calibration evidence: {failure}"
                        )
            calibration_unresolved = (
                pre_sampling_status == "calibration_unresolved"
            )
            expected_proof_basis = (
                "duration_bound"
                if stage_geometry_status == "proven_infeasible"
                else None
            )
            if stage_specs is not None and (
                infeasibility_proof_basis != expected_proof_basis
            ):
                malformed(
                    f"{cell_context}: controller/stage infeasibility proof "
                    "basis mismatch"
                )
            endpoint_metric_present = "max_endpoint_distance_pp" in diagnostic
            independently_verified_precision_met: bool | None = None
            target_ci = median = ci_low = ci_high = reported_half = reported_endpoint = None
            if endpoint_metric_present:
                metric_fields = {
                    "target_ci_half_width_pp": "target_ci",
                    "median_pp": "median",
                    "ci95_low_pp": "ci_low",
                    "ci95_high_pp": "ci_high",
                    "ci95_half_width_pp": "reported_half",
                    "max_endpoint_distance_pp": "reported_endpoint",
                }
                parsed_metrics: dict[str, float | None] = {}
                for field in metric_fields:
                    try:
                        parsed_metrics[field] = _float(diagnostic.get(field))
                    except (TypeError, ValueError, OverflowError):
                        parsed_metrics[field] = None
                target_ci = parsed_metrics["target_ci_half_width_pp"]
                median = parsed_metrics["median_pp"]
                ci_low = parsed_metrics["ci95_low_pp"]
                ci_high = parsed_metrics["ci95_high_pp"]
                reported_half = parsed_metrics["ci95_half_width_pp"]
                reported_endpoint = parsed_metrics["max_endpoint_distance_pp"]
                if explicitly_infeasible is False and not calibration_unresolved:
                    missing_metrics = [
                        field for field, value in parsed_metrics.items()
                        if value is None
                    ]
                    if missing_metrics:
                        malformed(
                            f"{cell_context}: endpoint-precision diagnostics "
                            f"missing/non-finite {missing_metrics}"
                        )
                    elif not (
                        target_ci is not None and target_ci > 0.0
                        and ci_low is not None and median is not None
                        and ci_high is not None and ci_low <= median <= ci_high
                    ):
                        malformed(
                            f"{cell_context}: invalid target or CI endpoint ordering"
                        )
                    else:
                        recomputed_half = (ci_high - ci_low) / 2.0
                        recomputed_endpoint = max(
                            abs(median - ci_low), abs(ci_high - median)
                        )
                        tolerance = 1e-9 * max(
                            1.0, abs(recomputed_half), abs(recomputed_endpoint)
                        )
                        if reported_half is None or not math.isclose(
                            reported_half, recomputed_half,
                            rel_tol=1e-9, abs_tol=tolerance,
                        ):
                            malformed(
                                f"{cell_context}: ci95_half_width_pp does not "
                                "match the reported CI endpoints"
                            )
                        if reported_endpoint is None or not math.isclose(
                            reported_endpoint, recomputed_endpoint,
                            rel_tol=1e-9, abs_tol=tolerance,
                        ):
                            malformed(
                                f"{cell_context}: max_endpoint_distance_pp does "
                                "not match median/CI endpoints"
                            )
                        independently_verified_precision_met = bool(
                            minimum is not None
                            and valid_samples is not None
                            and valid_samples >= minimum
                            and recomputed_endpoint <= target_ci
                        )
                        if precision_met is not independently_verified_precision_met:
                            malformed(
                                f"{cell_context}: precision_met disagrees with "
                                f"{ADAPTIVE_PRECISION_CRITERION}"
                            )
                        if converged is not independently_verified_precision_met:
                            malformed(
                                f"{cell_context}: converged disagrees with "
                                "independently verified endpoint precision"
                            )
            terminal = should_continue is False and isinstance(stop_reason, str) and bool(stop_reason)
            if not terminal:
                malformed(f"{cell_context}: cell is not terminal")
            feasible = explicitly_infeasible is False
            convergence_passed = (
                terminal
                and (
                    explicitly_infeasible is True
                    or (
                        endpoint_metric_present
                        and independently_verified_precision_met is True
                    )
                    or (
                        not endpoint_metric_present
                        and converged is True and precision_met is True
                    )
                )
            )
            floor_passed = (
                explicitly_infeasible is True
                or (
                    expected_floor is not None
                    and minimum == expected_floor
                    and valid_samples is not None
                    and valid_samples >= expected_floor
                )
            )
            if calibration_unresolved:
                if explicitly_infeasible is not False:
                    malformed(
                        f"{cell_context}: unresolved calibration cannot be "
                        "explicitly infeasible"
                    )
                if any(value not in (0, None) for value in (
                    total_attempts, valid_samples, invalid_samples, discarded
                )):
                    malformed(
                        f"{cell_context}: unresolved calibration emitted attempts"
                    )
                if stop_reason != "geometry_calibration_unresolved":
                    malformed(
                        f"{cell_context}: unresolved calibration has invalid stop_reason"
                    )
                malformed(f"{cell_context}: geometry calibration unresolved")
            elif feasible and not convergence_passed:
                malformed(f"{cell_context}: feasible cell did not converge")
            if feasible and not calibration_unresolved and not floor_passed:
                malformed(f"{cell_context}: duration-based floor was not met")
            legacy_infeasibility_exemption = bool(
                explicitly_infeasible is True
                and stage_specs is None
                and stop_reason == "geometry_target_infeasible"
            )
            if explicitly_infeasible is True and stop_reason not in {
                "geometry_target_infeasible",
                "geometry_target_proven_infeasible",
            }:
                malformed(f"{cell_context}: infeasible cell has invalid stop_reason")
            infeasibility_proven = False
            if explicitly_infeasible is True:
                if stage_specs is not None:
                    if pre_sampling_status != "proven_infeasible":
                        malformed(
                            f"{cell_context}: only proven infeasibility is "
                            "exemptible"
                        )
                    if stage_geometry_status != "proven_infeasible":
                        malformed(
                            f"{cell_context}: stage does not prove infeasibility"
                        )
                    if (
                        not isinstance(stage_spec, dict)
                        or stage_spec.get("target_feasible_from_duration") is not False
                    ):
                        malformed(
                            f"{cell_context}: infeasible stage lacks structural "
                            "duration-bound proof"
                        )
                infeasibility_proven = bool(
                    stage_specs is not None
                    and pre_sampling_status == "proven_infeasible"
                    and infeasibility_proof_basis == "duration_bound"
                    and stage_geometry_status == "proven_infeasible"
                    and isinstance(stage_spec, dict)
                    and stage_spec.get("target_feasible_from_duration") is False
                    and (
                        not result_declares_counterfactual_geometry
                        or (
                            stage_duration_evidence_verified is True
                            and stage_recomputed_duration_infeasible is True
                        )
                    )
                    and stop_reason == "geometry_target_proven_infeasible"
                )
            infeasibility_exemption_eligible = bool(
                infeasibility_proven or legacy_infeasibility_exemption
            )

            accounted_attempts = None
            if total_attempts is not None and valid_samples is not None and invalid_samples is not None:
                accounted_attempts = valid_samples + invalid_samples
                if total_attempts != accounted_attempts:
                    malformed(f"{cell_context}: valid + invalid does not equal total_attempts")
            issued = None
            if total_attempts is not None and discarded is not None:
                issued = total_attempts + discarded
            matching_rows = [
                row
                for row in result_rows
                if row["case_key"] == case_key
                and row["victim_index"] == victim_index
                and row["requested_overlap_pct"] == target
                and row["dispatch_mode"] == dispatch_mode
            ]
            if issued is not None and len(matching_rows) != issued:
                malformed(
                    f"{cell_context}: observed row count {len(matching_rows)} "
                    f"does not equal total_attempts + discarded_after_stop ({issued})"
                )
            rejection_rate = (
                None
                if total_attempts in (None, 0) or invalid_samples is None
                else invalid_samples / total_attempts * 100.0
            )
            rejection_rate_passed = (
                explicitly_infeasible is True
                or (
                    rejection_rate is not None
                    and rejection_rate <= max_rejection_rate_pct
                )
            )

            for field, value in (
                ("total_attempts", total_attempts),
                ("valid_samples", valid_samples),
                ("invalid_samples", invalid_samples),
                ("discarded_after_stop", discarded),
            ):
                if value is not None:
                    cell_sums[field] += value
            cells_out.append({
                "result_index": result_index,
                "cell_key": diagnostic.get("cell_key"),
                "case_key": case_key,
                "victim_index": victim_index,
                "requested_overlap_pct": target,
                "dispatch_mode": dispatch_mode,
                "total_attempts": total_attempts,
                "valid_samples": valid_samples,
                "invalid_samples": invalid_samples,
                "discarded_after_stop": discarded,
                "issued_sample_count": issued,
                "observed_sample_count": len(matching_rows),
                "minimum_valid_samples": minimum,
                "expected_duration_floor": expected_floor,
                "explicitly_infeasible": explicitly_infeasible,
                "infeasibility_proven": infeasibility_proven,
                "legacy_infeasibility_exemption": (
                    legacy_infeasibility_exemption
                ),
                "infeasibility_exemption_eligible": (
                    infeasibility_exemption_eligible
                ),
                "pre_sampling_status": pre_sampling_status,
                "infeasibility_proof_basis": infeasibility_proof_basis,
                "stage_geometry_status": stage_geometry_status,
                "stage_calibration_evidence_verified": (
                    stage_calibration_evidence_verified
                ),
                "stage_duration_evidence_verified": (
                    stage_duration_evidence_verified
                ),
                "stage_recomputed_duration_infeasible": (
                    stage_recomputed_duration_infeasible
                ),
                "calibration_unresolved": calibration_unresolved,
                "terminal": terminal,
                "converged": converged,
                "precision_met": precision_met,
                "precision_criterion_mode": (
                    "endpoint_distance_independently_verified"
                    if endpoint_metric_present else
                    "legacy_reported_flags_without_endpoint_metric"
                ),
                "target_ci_half_width_pp": target_ci,
                "median_pp": median,
                "ci95_low_pp": ci_low,
                "ci95_high_pp": ci_high,
                "ci95_half_width_pp": reported_half,
                "max_endpoint_distance_pp": reported_endpoint,
                "independently_verified_precision_met": (
                    independently_verified_precision_met
                ),
                "stop_reason": stop_reason,
                "floor_met": floor_passed,
                "rejection_rate_pct": rejection_rate,
                "rejection_rate_met": rejection_rate_passed,
            })

        for field in (
            "total_attempts",
            "valid_samples",
            "invalid_samples",
            "discarded_after_stop",
        ):
            if summary_totals[field] is not None and summary_totals[field] != cell_sums[field]:
                malformed(f"{context}.summary: {field} does not match cell sum")
        result_cells = [
            row for row in cells_out if row["result_index"] == result_index
        ]
        derived_active = sum(row["terminal"] is not True for row in result_cells)
        derived_converged = sum(row["converged"] is True for row in result_cells)
        derived_stopped_without_convergence = sum(
            row["terminal"] is True and row["converged"] is not True
            for row in result_cells
        )
        for field, derived in (
            ("active_cell_count", derived_active),
            ("converged_cell_count", derived_converged),
            ("stopped_without_convergence_count", derived_stopped_without_convergence),
        ):
            if summary_totals[field] is not None and summary_totals[field] != derived:
                malformed(f"{context}.summary: {field} does not match cells")
        issued_total = cell_sums["total_attempts"] + cell_sums["discarded_after_stop"]
        if issued_total != len(result_rows):
            malformed(
                f"{context}: observed row count {len(result_rows)} does not "
                f"reconcile with ingested + discarded ({issued_total})"
            )
        actual_counts = diagnostics.get("actual_attempt_counts")
        if not isinstance(actual_counts, dict):
            malformed(f"{context}: missing actual_attempt_counts object")
        else:
            ingested_expected = {
                str(row["cell_key"]): row["total_attempts"] for row in cells_out
                if row["result_index"] == result_index
            }
            completed_expected = {
                str(row["cell_key"]): int(row["total_attempts"] or 0)
                + int(row["discarded_after_stop"] or 0)
                for row in cells_out
                if row["result_index"] == result_index
            }
            semantics = diagnostics.get("actual_attempt_counts_semantics")
            if semantics is None:
                # Legacy schema: actual_attempt_counts meant controller-ingested.
                if actual_counts != ingested_expected:
                    malformed(f"{context}: legacy actual_attempt_counts do not match cells")
            elif semantics == (
                "completed treatment attempts; use ingested_attempt_counts "
                "for controller-retained attempts"
            ):
                if actual_counts != completed_expected:
                    malformed(f"{context}: completed actual_attempt_counts do not match cells")
                ingested_counts = diagnostics.get("ingested_attempt_counts")
                if ingested_counts != ingested_expected:
                    malformed(f"{context}: ingested_attempt_counts do not match cells")
                for field, expected in (
                    ("issued_attempt_counts", completed_expected),
                    ("completed_attempt_counts", completed_expected),
                    (
                        "discarded_after_stop_counts",
                        {
                            str(row["cell_key"]): row["discarded_after_stop"]
                            for row in cells_out
                            if row["result_index"] == result_index
                        },
                    ),
                ):
                    if diagnostics.get(field) != expected:
                        malformed(f"{context}: {field} do not match cells")
                reconciliation = diagnostics.get("attempt_reconciliation")
                expected_reconciliation = {
                    "issued": sum(completed_expected.values()),
                    "completed": sum(completed_expected.values()),
                    "ingested": sum(int(value or 0) for value in ingested_expected.values()),
                    "discarded_after_stop": sum(
                        int(row["discarded_after_stop"] or 0)
                        for row in cells_out
                        if row["result_index"] == result_index
                    ),
                    "passed": True,
                }
                if reconciliation != expected_reconciliation:
                    malformed(f"{context}: attempt_reconciliation does not match cells")
            else:
                malformed(f"{context}: unknown actual_attempt_counts_semantics")
        rounds = diagnostics.get("rounds")
        if not isinstance(rounds, list):
            malformed(f"{context}: missing rounds array")
        else:
            round_seen = round_ingested = round_discarded = 0
            for round_index, round_row in enumerate(rounds):
                ingestion = round_row.get("ingestion") if isinstance(round_row, dict) else None
                if not isinstance(ingestion, dict):
                    malformed(f"{context}.rounds[{round_index}]: missing ingestion")
                    continue
                values = []
                for field in ("rows_seen", "rows_ingested", "rows_discarded_after_stop"):
                    values.append(required_int(ingestion, field, f"{context}.rounds[{round_index}].ingestion"))
                if all(value is not None for value in values):
                    round_seen += int(values[0])
                    round_ingested += int(values[1])
                    round_discarded += int(values[2])
            if round_seen != len(result_rows):
                malformed(f"{context}: round rows_seen does not match observed rows")
            if round_ingested != cell_sums["total_attempts"]:
                malformed(f"{context}: round rows_ingested does not match cell attempts")
            if round_discarded != cell_sums["discarded_after_stop"]:
                malformed(f"{context}: round discarded rows do not match cell sum")

    feasible_cells = [row for row in cells_out if row["explicitly_infeasible"] is False]
    total_attempts = sum(int(row["total_attempts"] or 0) for row in feasible_cells)
    invalid_attempts = sum(int(row["invalid_samples"] or 0) for row in feasible_cells)
    return {
        "present": bool(adaptive_results),
        "result_count": len(adaptive_results),
        "cell_count": len(cells_out),
        "feasible_cell_count": len(feasible_cells),
        "explicitly_infeasible_cell_count": sum(
            row["explicitly_infeasible"] is True for row in cells_out
        ),
        "proven_infeasible_cell_count": sum(
            row["infeasibility_proven"] is True for row in cells_out
        ),
        "legacy_infeasibility_exemption_cell_count": sum(
            row["legacy_infeasibility_exemption"] is True for row in cells_out
        ),
        "calibration_unresolved_cell_count": sum(
            row["calibration_unresolved"] is True for row in cells_out
        ),
        "terminal_cell_count": sum(row["terminal"] is True for row in cells_out),
        "converged_feasible_cell_count": sum(
            row["converged"] is True and row["precision_met"] is True
            for row in feasible_cells
        ),
        "floor_met_feasible_cell_count": sum(row["floor_met"] is True for row in feasible_cells),
        "rejection_rate_met_feasible_cell_count": sum(
            row["rejection_rate_met"] is True for row in feasible_cells
        ),
        "precision_criterion": ADAPTIVE_PRECISION_CRITERION,
        "endpoint_precision_cell_count": sum(
            row["precision_criterion_mode"]
            == "endpoint_distance_independently_verified"
            for row in cells_out
        ),
        "legacy_precision_cell_count": sum(
            row["precision_criterion_mode"]
            == "legacy_reported_flags_without_endpoint_metric"
            for row in cells_out
        ),
        "ingested_attempt_count": sum(int(row["total_attempts"] or 0) for row in cells_out),
        "valid_sample_count": sum(int(row["valid_samples"] or 0) for row in cells_out),
        "invalid_sample_count": sum(int(row["invalid_samples"] or 0) for row in cells_out),
        "discarded_after_stop_count": sum(int(row["discarded_after_stop"] or 0) for row in cells_out),
        "issued_sample_count": sum(int(row["issued_sample_count"] or 0) for row in cells_out),
        "feasible_rejection_rate_pct": (
            0.0 if total_attempts == 0 else invalid_attempts / total_attempts * 100.0
        ),
        "rejection_rates_passed": all(
            row["rejection_rate_met"] is True for row in feasible_cells
        ),
        "malformed_or_failed_diagnostic_count": len(errors),
        "errors": errors,
        "cells": cells_out,
        "passed": bool(adaptive_results) and not errors,
    }


def validate_samples(
    samples: Sequence[dict[str, Any]],
    *,
    expected_sequences: Sequence[int] | None = None,
    expected_cases: Sequence[str] | None = None,
    expected_coverage_targets: Sequence[float] = DEFAULT_COVERAGE_TARGETS,
    expected_dispatch_modes: Sequence[str] = DEFAULT_DISPATCH_MODES,
    drift_threshold_pct: float = 5.0,
    geometry_tolerance_pct: float = 10.0,
    max_rejection_rate_pct: float = DEFAULT_MAX_REJECTION_RATE_PCT,
    floors: Sequence[int] = (50, 100),
    require_counterfactual_v2_adaptive: bool = False,
    require_counterfactual_v2: bool = False,
    expected_fixed_iterations: int | None = None,
    require_adjacent_pre_rebased_delays: bool = False,
) -> dict[str, Any]:
    """Validate normalized samples and return a JSON-serializable report.

    A sample counts toward a statistical floor only when it is naturally
    feasible, the runner marked it valid, its measured geometry and baseline
    drift pass the configured thresholds, and correctness is explicitly true.
    Missing fields therefore cannot silently satisfy an acceptance gate.
    """

    if not math.isfinite(max_rejection_rate_pct) or not 0.0 <= max_rejection_rate_pct <= 100.0:
        raise ValueError("max_rejection_rate_pct must be between 0 and 100")
    if expected_fixed_iterations is not None and expected_fixed_iterations <= 0:
        raise ValueError("expected_fixed_iterations must be a positive integer")
    adaptive_results = list(getattr(samples, "adaptive_results", ()))
    result_metadata = list(getattr(samples, "result_metadata", ()))
    strict_document_provenance = getattr(
        samples, "strict_document_provenance", None
    )
    if strict_document_provenance is None:
        strict_document_provenance = {
            "schema_version": 1,
            "passed": False,
            "error_count": 1,
            "errors": [
                "strict provenance requires samples loaded from final raw JSON"
            ],
            "implementation_files_checked": 0,
            "retained_digest_count_checked": 0,
            "clock_rows_checked": 0,
            "clock_history_entries_checked": 0,
            "clock_reference_rows_checked": 0,
            "clock_age_rows_checked": 0,
        }
    counterfactual_v2_adaptive_present = bool(adaptive_results) and all(
        result.get("declares_counterfactual_geometry") is True
        for result in adaptive_results
    )
    counterfactual_v2_present = bool(result_metadata) and all(
        result.get("declares_counterfactual_geometry") is True
        for result in result_metadata
    )
    rows = [dict(row) for row in samples]
    # Preserve the public API for callers that pass already-normalized rows
    # instead of using ``load_samples``.  Such legacy rows predate explicit
    # start-anchor dispositions and are confirmed/scored unless the loader
    # reconstructed stronger evidence.
    for row in rows:
        row.setdefault(
            "runtime_start_geometry_disposition",
            RUNTIME_START_GEOMETRY_CONFIRMED,
        )
        row.setdefault(
            "runtime_start_geometry_disposition_source",
            "legacy_normalized_input",
        )
        row.setdefault("runtime_start_geometry_disposition_explicit", False)
        row.setdefault("runtime_start_geometry_scoring_eligible", True)
        row.setdefault("runtime_start_geometry_contract_ok", True)
        row.setdefault(
            "runtime_start_geometry_static_exemption_proven", False
        )
        row.setdefault(
            "runtime_start_geometry_static_exemption_claimed", False
        )
    adjacent_pre_rows = []
    for row_index, row in enumerate(rows):
        evidence_errors = _verify_adjacent_pre_rebased_row(row)
        row["adjacent_pre_rebased_delays_integrity_ok"] = not evidence_errors
        adjacent_pre_rows.append({
            "row_index": row_index,
            "case_key": row.get("case_key"),
            "seq_len": row.get("seq_len"),
            "victim_index": row.get("victim_index"),
            "dispatch_mode": row.get("dispatch_mode"),
            "requested_overlap_pct": row.get("requested_overlap_pct"),
            "sample_index": row.get("sample_index"),
            "disposition": row.get("runtime_start_geometry_disposition"),
            "policy": (
                row.get("delay_rebase_evidence", {}).get("policy")
                if isinstance(row.get("delay_rebase_evidence"), dict)
                else None
            ),
            "passed": not evidence_errors,
            "errors": evidence_errors,
        })
    if not rows:
        raise ValueError("corpus contains no samples")
    requested_sequences = sorted(
        set(expected_sequences)
        if expected_sequences is not None
        else {row["seq_len"] for row in rows if row["seq_len"] is not None}
    )
    requested_cases = sorted(
        set(expected_cases)
        if expected_cases is not None
        else {row["case_key"] for row in rows if row["case_key"]}
    )
    coverage_targets = sorted({float(value) for value in expected_coverage_targets})
    dispatch_modes = list(dict.fromkeys(str(value) for value in expected_dispatch_modes))

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_stratum_key(row)].append(row)

    strata = []
    totals = defaultdict(int)
    for key, subset in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        (
            seq_len,
            case_key,
            victim_index,
            victim_family,
            aggressor_family,
            dispatch_mode,
            target,
        ) = key
        requested_count = len(subset)
        feasible_count = sum(row["feasible"] is True for row in subset)
        infeasible_count = sum(row["feasible"] is False for row in subset)
        feasible_unknown_count = sum(row["feasible"] is None for row in subset)
        runner_valid_count = sum(row["valid"] is True for row in subset)
        geometry_failures = sum(
            row["feasible"] is True
            and (
                row["valid"] is False
                or _geometry_pass(row, geometry_tolerance_pct) is False
            )
            for row in subset
        )
        geometry_unknown = sum(
            row["feasible"] is True
            and (
                row["valid"] is None
                or _geometry_pass(row, geometry_tolerance_pct) is None
            )
            for row in subset
        )
        geometry_baseline_failures = sum(
            row["feasible"] is True
            and _geometry_baseline_pass(row, drift_threshold_pct) is False
            for row in subset
        )
        geometry_baseline_unknown = sum(
            row["feasible"] is True
            and _geometry_baseline_pass(row, drift_threshold_pct) is None
            for row in subset
        )
        drift_failures = sum(
            row["baseline_drift_pct"] is not None
            and abs(row["baseline_drift_pct"]) > drift_threshold_pct
            for row in subset
        )
        drift_unknown = sum(row["baseline_drift_pct"] is None for row in subset)
        correctness_failures = sum(row["correct"] is False for row in subset)
        correctness_unknown = sum(row["correct"] is None for row in subset)
        disposition_contract_failure_count = sum(
            row["runtime_start_geometry_contract_ok"] is not True
            for row in subset
        )
        unresolved_count = sum(
            row["runtime_start_geometry_disposition"]
            == RUNTIME_START_GEOMETRY_UNRESOLVED
            for row in subset
        )
        static_exempt_count = sum(
            row["runtime_start_geometry_disposition"]
            == RUNTIME_START_GEOMETRY_STATIC
            and row["runtime_start_geometry_static_exemption_proven"] is True
            for row in subset
        )
        unscored_count = sum(
            row["runtime_start_geometry_scoring_eligible"] is not True
            for row in subset
        )
        scored_count = requested_count - unscored_count
        scored_feasible_count = sum(
            row["runtime_start_geometry_scoring_eligible"] is True
            and row["runtime_start_geometry_contract_ok"] is True
            and row["feasible"] is True
            for row in subset
        )
        accepted_count = sum(
            _sample_is_accepted(
                row,
                geometry_tolerance_pct=geometry_tolerance_pct,
                drift_threshold_pct=drift_threshold_pct,
            )
            for row in subset
        )
        # Every scored, feasible row is evaluable and must be in exactly one
        # side of this partition.  Unresolved and proven-static rows are not in
        # either the estimator or its rejection denominator.
        rejection_count = scored_feasible_count - accepted_count
        rejection_rate_pct = (
            0.0
            if scored_feasible_count == 0
            else rejection_count / scored_feasible_count * 100.0
        )
        coverage_hole = bool(
            feasible_count > 0 and accepted_count == 0
        )
        sample_indices = [row["sample_index"] for row in subset]
        expected_indices = (
            set(range(expected_fixed_iterations))
            if expected_fixed_iterations is not None
            else set()
        )
        observed_indices = {
            index for index in sample_indices if index is not None
        }
        fixed_missing_indices = sorted(expected_indices - observed_indices)
        fixed_unexpected_indices = sorted(observed_indices - expected_indices)
        fixed_duplicate_indices = sorted({
            index for index in observed_indices
            if sample_indices.count(index) != 1
        })
        fixed_unknown_index_count = sum(index is None for index in sample_indices)
        fixed_sampling_met = bool(
            expected_fixed_iterations is None
            or (
                requested_count == expected_fixed_iterations
                and fixed_unknown_index_count == 0
                and not fixed_missing_indices
                and not fixed_unexpected_indices
                and not fixed_duplicate_indices
            )
        )
        recommended_floor = _recommended_floor(subset)
        record = {
            "seq_len": seq_len,
            "case_key": case_key,
            "victim_index": victim_index,
            "victim_family": victim_family,
            "aggressor_family": aggressor_family,
            "dispatch_mode": dispatch_mode,
            "requested_overlap_pct": target,
            "requested_count": requested_count,
            "feasible_count": feasible_count,
            "infeasible_count": infeasible_count,
            "feasibility_unknown_count": feasible_unknown_count,
            "runner_valid_count": runner_valid_count,
            "accepted_valid_count": accepted_count,
            "scored_count": scored_count,
            "scored_feasible_count": scored_feasible_count,
            "rejection_denominator_count": scored_feasible_count,
            "unscored_count": unscored_count,
            "runtime_start_geometry_unresolved_count": unresolved_count,
            "runtime_start_geometry_static_exempt_count": static_exempt_count,
            "runtime_start_geometry_contract_failure_count": (
                disposition_contract_failure_count
            ),
            "coverage_hole": coverage_hole,
            "scored_partition_reconciled": (
                accepted_count + rejection_count == scored_feasible_count
            ),
            "geometry_failure_count": geometry_failures,
            "geometry_unknown_count": geometry_unknown,
            "geometry_baseline_failure_count": geometry_baseline_failures,
            "geometry_baseline_unknown_count": geometry_baseline_unknown,
            "drift_failure_count": drift_failures,
            "drift_unknown_count": drift_unknown,
            "correctness_failure_count": correctness_failures,
            "correctness_unknown_count": correctness_unknown,
            "rejection_count": rejection_count,
            "rejection_rate_pct": rejection_rate_pct,
            "rejection_rate_met": (
                scored_feasible_count == 0
                or rejection_rate_pct <= max_rejection_rate_pct
            ),
            "fixed_sampling_met": fixed_sampling_met,
            "fixed_missing_sample_indices": fixed_missing_indices,
            "fixed_unexpected_sample_indices": fixed_unexpected_indices,
            "fixed_duplicate_sample_indices": fixed_duplicate_indices,
            "fixed_unknown_sample_index_count": fixed_unknown_index_count,
            "recommended_floor": recommended_floor,
            "recommended_floor_met": accepted_count >= recommended_floor,
            "statistical_floors": {
                str(floor): accepted_count >= floor for floor in floors
            },
        }
        strata.append(record)
        for field in (
            "requested_count",
            "feasible_count",
            "infeasible_count",
            "feasibility_unknown_count",
            "runner_valid_count",
            "accepted_valid_count",
            "scored_count",
            "scored_feasible_count",
            "rejection_denominator_count",
            "unscored_count",
            "runtime_start_geometry_unresolved_count",
            "runtime_start_geometry_static_exempt_count",
            "runtime_start_geometry_contract_failure_count",
            "geometry_failure_count",
            "geometry_unknown_count",
            "geometry_baseline_failure_count",
            "geometry_baseline_unknown_count",
            "drift_failure_count",
            "drift_unknown_count",
            "correctness_failure_count",
            "correctness_unknown_count",
            "rejection_count",
        ):
            totals[field] += int(record[field])
        totals["coverage_hole_count"] += int(record["coverage_hole"])
        totals["scored_partition_failure_count"] += int(
            not record["scored_partition_reconciled"]
        )

    adaptive = _adaptive_acceptance(
        rows,
        adaptive_results,
        max_rejection_rate_pct=max_rejection_rate_pct,
    ) if adaptive_results else {
        "present": False,
        "result_count": 0,
        "cell_count": 0,
        "feasible_cell_count": 0,
        "explicitly_infeasible_cell_count": 0,
        "proven_infeasible_cell_count": 0,
        "legacy_infeasibility_exemption_cell_count": 0,
        "calibration_unresolved_cell_count": 0,
        "terminal_cell_count": 0,
        "converged_feasible_cell_count": 0,
        "floor_met_feasible_cell_count": 0,
        "rejection_rate_met_feasible_cell_count": 0,
        "precision_criterion": ADAPTIVE_PRECISION_CRITERION,
        "endpoint_precision_cell_count": 0,
        "legacy_precision_cell_count": 0,
        "ingested_attempt_count": 0,
        "valid_sample_count": 0,
        "invalid_sample_count": 0,
        "discarded_after_stop_count": 0,
        "issued_sample_count": 0,
        "feasible_rejection_rate_pct": 0.0,
        "rejection_rates_passed": True,
        "malformed_or_failed_diagnostic_count": 0,
        "errors": [],
        "cells": [],
        "passed": True,
    }
    if adaptive["present"]:
        for record in strata:
            matching_cells = [
                cell
                for cell in adaptive["cells"]
                if cell["case_key"] == record["case_key"]
                and cell["victim_index"] == record["victim_index"]
                and cell["requested_overlap_pct"] == record["requested_overlap_pct"]
                and cell["dispatch_mode"] == record["dispatch_mode"]
            ]
            record["adaptive_cell_count"] = len(matching_cells)
            if matching_cells:
                diagnostic_floors = {
                    cell["expected_duration_floor"] for cell in matching_cells
                }
                record["recommended_floor"] = (
                    next(iter(diagnostic_floors))
                    if len(diagnostic_floors) == 1
                    else None
                )
                record["recommended_floor_met"] = all(
                    cell["floor_met"] is True for cell in matching_cells
                )
                record["adaptive_valid_sample_count"] = sum(
                    int(cell["valid_samples"] or 0) for cell in matching_cells
                )
                record["adaptive_invalid_sample_count"] = sum(
                    int(cell["invalid_samples"] or 0) for cell in matching_cells
                )
                record["adaptive_discarded_after_stop_count"] = sum(
                    int(cell["discarded_after_stop"] or 0) for cell in matching_cells
                )
                record["adaptive_ingested_attempt_count"] = (
                    record["adaptive_valid_sample_count"]
                    + record["adaptive_invalid_sample_count"]
                )
            else:
                record["recommended_floor_met"] = False
                record["adaptive_valid_sample_count"] = 0
                record["adaptive_invalid_sample_count"] = 0
                record["adaptive_discarded_after_stop_count"] = 0
                record["adaptive_ingested_attempt_count"] = 0

    orientations_by_case: dict[str, set[tuple[int | None, str, str]]] = defaultdict(set)
    observed_case_sequences = set()
    observed_points = set()
    for row in rows:
        orientations_by_case[row["case_key"]].add(_orientation(row))
        observed_case_sequences.add((row["case_key"], row["seq_len"]))
        observed_points.add(
            (
                row["case_key"],
                row["seq_len"],
                _orientation(row),
                row["dispatch_mode"],
                row["requested_overlap_pct"],
            )
        )

    # A controller may prune a geometrically impossible cell before issuing a
    # treatment attempt. Such a terminal, reconciled diagnostic is positive
    # evidence that the requested sweep point was evaluated, not a coverage
    # gap. Keep the compact diagnostic identity separate because cell
    # diagnostics intentionally identify orientations by victim index rather
    # than repeating participant-family labels.
    diagnostic_infeasible_points = set()
    for cell in adaptive["cells"]:
        case_key = cell["case_key"]
        victim_index = cell["victim_index"]
        known_orientations = {
            orientation for orientation in orientations_by_case.get(case_key, set())
            if orientation[0] == victim_index
        }
        if not known_orientations:
            orientations_by_case[case_key].add((victim_index, "", ""))
        result_sequences = {
            row["seq_len"] for row in rows
            if row.get("_result_index") == cell["result_index"]
            and row["seq_len"] is not None
        }
        if not result_sequences and len(requested_sequences) == 1:
            result_sequences = {requested_sequences[0]}
        if (
            cell["infeasibility_exemption_eligible"] is True
            and cell["terminal"] is True
        ):
            for seq_len in result_sequences:
                diagnostic_infeasible_points.add((
                    case_key,
                    seq_len,
                    victim_index,
                    cell["dispatch_mode"],
                    cell["requested_overlap_pct"],
                ))

    missing_sequences = [
        seq_len
        for seq_len in requested_sequences
        if not any(row["seq_len"] == seq_len for row in rows)
    ]
    missing_cases = [
        case_key
        for case_key in requested_cases
        if not any(row["case_key"] == case_key for row in rows)
    ]
    case_sequence_gaps = [
        {"case_key": case_key, "seq_len": seq_len}
        for case_key in requested_cases
        for seq_len in requested_sequences
        if (case_key, seq_len) not in observed_case_sequences
    ]
    missing_points = []
    for case_key in requested_cases:
        for seq_len in requested_sequences:
            for orientation in sorted(
                orientations_by_case.get(case_key, set()), key=lambda item: tuple(str(x) for x in item)
            ):
                for dispatch_mode in dispatch_modes:
                    for target in coverage_targets:
                        point = (case_key, seq_len, orientation, dispatch_mode, target)
                        diagnostic_point = (
                            case_key, seq_len, orientation[0], dispatch_mode, target
                        )
                        if (
                            point not in observed_points
                            and diagnostic_point not in diagnostic_infeasible_points
                        ):
                            missing_points.append({
                                "case_key": case_key,
                                "seq_len": seq_len,
                                "victim_index": orientation[0],
                                "victim_family": orientation[1],
                                "aggressor_family": orientation[2],
                                "dispatch_mode": dispatch_mode,
                                "requested_overlap_pct": target,
                            })

    floor_status = {
        str(floor): {
            "met": all(
                record["accepted_valid_count"] >= floor
                for record in strata
                if record["feasible_count"] > 0
            ),
            "failing_feasible_strata": sum(
                record["feasible_count"] > 0
                and record["accepted_valid_count"] < floor
                for record in strata
            ),
        }
        for floor in floors
    }
    recommended_status = {
        "met": bool(
            expected_fixed_iterations is not None
            or all(
                record["recommended_floor_met"]
                for record in strata
                if record["feasible_count"] > 0
            )
        ),
        "failing_feasible_strata": sum(
            expected_fixed_iterations is None
            and
            record["feasible_count"] > 0
            and not record["recommended_floor_met"]
            for record in strata
        ),
    }
    raw_rejection_rate_met = all(
        record["rejection_rate_met"]
        for record in strata
        if record["feasible_count"] > 0
    )
    fixed_sampling = {
        "present": expected_fixed_iterations is not None,
        "expected_iterations_per_stratum": expected_fixed_iterations,
        "expected_sample_indices": (
            list(range(expected_fixed_iterations))
            if expected_fixed_iterations is not None
            else None
        ),
        "stratum_count": len(strata),
        "exact_index_stratum_count": sum(
            record["fixed_sampling_met"] is True for record in strata
        ),
        "failed_stratum_count": sum(
            record["fixed_sampling_met"] is not True for record in strata
        ) if expected_fixed_iterations is not None else 0,
        "failed_strata": [
            {
                "seq_len": record["seq_len"],
                "case_key": record["case_key"],
                "victim_index": record["victim_index"],
                "dispatch_mode": record["dispatch_mode"],
                "requested_overlap_pct": record["requested_overlap_pct"],
                "observed_count": record["requested_count"],
                "missing_sample_indices": record["fixed_missing_sample_indices"],
                "unexpected_sample_indices": record["fixed_unexpected_sample_indices"],
                "duplicate_sample_indices": record["fixed_duplicate_sample_indices"],
                "unknown_sample_index_count": record[
                    "fixed_unknown_sample_index_count"
                ],
            }
            for record in strata
            if expected_fixed_iterations is not None
            and record["fixed_sampling_met"] is not True
        ],
        "includes_unscored_and_infeasible_rows": True,
        "passed": bool(
            expected_fixed_iterations is None
            or all(record["fixed_sampling_met"] is True for record in strata)
        ),
    }
    adjacent_pre_rebased_delays = {
        "required": bool(require_adjacent_pre_rebased_delays),
        "row_count": len(adjacent_pre_rows),
        "verified_row_count": sum(row["passed"] for row in adjacent_pre_rows),
        "failed_row_count": sum(not row["passed"] for row in adjacent_pre_rows),
        "dynamic_row_count": sum(
            row["disposition"] != RUNTIME_START_GEOMETRY_STATIC
            for row in adjacent_pre_rows
        ),
        "strict_static_exempt_row_count": sum(
            row["disposition"] == RUNTIME_START_GEOMETRY_STATIC
            and row["passed"]
            for row in adjacent_pre_rows
        ),
        "failed_rows": [row for row in adjacent_pre_rows if not row["passed"]],
        "passed": all(row["passed"] for row in adjacent_pre_rows),
    }
    gates = {
        "no_sequence_or_case_gaps": not (
            missing_sequences or missing_cases or case_sequence_gaps
        ),
        "no_sweep_point_gaps": not missing_points,
        "feasibility_known": totals["feasibility_unknown_count"] == 0,
        "geometry_known": totals["geometry_unknown_count"] == 0,
        "geometry_baselines_known": (
            totals["geometry_baseline_unknown_count"] == 0
        ),
        "drift_known": totals["drift_unknown_count"] == 0,
        "correctness_passed": totals["correctness_failure_count"] == 0
        and totals["correctness_unknown_count"] == 0,
        "rejection_rate_within_bound": (
            adaptive["rejection_rates_passed"]
            if adaptive["present"]
            else raw_rejection_rate_met
        ),
        "adaptive_sampling_complete": adaptive["passed"],
        "fixed_sampling_complete": fixed_sampling["passed"],
        "runtime_start_geometry_contract_valid": (
            totals["runtime_start_geometry_contract_failure_count"] == 0
        ),
        "scored_feasible_partition_reconciled": (
            totals["scored_partition_failure_count"] == 0
            and totals["accepted_valid_count"] + totals["rejection_count"]
            == totals["scored_feasible_count"]
        ),
        "otherwise_feasible_strata_covered": (
            totals["coverage_hole_count"] == 0
        ),
        "counterfactual_v2_required": bool(
            not require_counterfactual_v2 or counterfactual_v2_present
        ),
        "adjacent_pre_rebased_delays_required": bool(
            not require_adjacent_pre_rebased_delays
            or adjacent_pre_rebased_delays["passed"]
        ),
        "strict_final_json_provenance_required": bool(
            not require_adjacent_pre_rebased_delays
            or strict_document_provenance["passed"]
        ),
        "counterfactual_v2_adaptive_required": bool(
            not require_counterfactual_v2_adaptive
            or counterfactual_v2_adaptive_present
        ),
        "recommended_statistical_floor_met": recommended_status["met"],
    }
    return {
        "schema_version": 3,
        "configuration": {
            "expected_sequences": requested_sequences,
            "expected_cases": requested_cases,
            "expected_coverage_targets": coverage_targets,
            "expected_dispatch_modes": dispatch_modes,
            "drift_threshold_pct": drift_threshold_pct,
            "default_geometry_tolerance_pct": geometry_tolerance_pct,
            "max_rejection_rate_pct": max_rejection_rate_pct,
            "statistical_floors": list(floors),
            "require_counterfactual_v2_adaptive": bool(
                require_counterfactual_v2_adaptive
            ),
            "require_counterfactual_v2": bool(require_counterfactual_v2),
            "expected_fixed_iterations": expected_fixed_iterations,
            "require_adjacent_pre_rebased_delays": bool(
                require_adjacent_pre_rebased_delays
            ),
            "accepted_valid_definition": (
                "feasible AND runner-valid AND scheduled counterfactual "
                "geometry within tolerance AND every geometry baseline within "
                "drift threshold AND victim baseline drift within threshold "
                "AND correctness true; legacy rows explicitly fall back to "
                "concurrent geometry"
            ),
            "geometry_validity_basis": (
                COUNTERFACTUAL_GEOMETRY_VALIDITY_BASIS
            ),
            "geometry_baseline_stability_basis": (
                GEOMETRY_BASELINE_STABILITY_BASIS
            ),
            "adaptive_floor_definition": (
                "controller valid_samples meets the duration-based 50/100 "
                "floor; discarded tail-of-batch rows never count toward it"
            ),
        },
        "summary": {
            "sample_count": len(rows),
            "stratum_count": len(strata),
            **dict(totals),
            # ``accepted_valid_count`` is retained for report-schema
            # compatibility. It is a raw retained-row count and is not the
            # estimator sample count for adaptive runs.
            "raw_accepted_valid_count": totals["accepted_valid_count"],
            "adaptive_controller_valid_sample_count": (
                adaptive["valid_sample_count"] if adaptive["present"] else None
            ),
            "adaptive_controller_invalid_sample_count": (
                adaptive["invalid_sample_count"] if adaptive["present"] else None
            ),
            "adaptive_controller_discarded_after_stop_count": (
                adaptive["discarded_after_stop_count"] if adaptive["present"] else None
            ),
            "estimator_valid_sample_count": (
                adaptive["valid_sample_count"]
                if adaptive["present"]
                else totals["accepted_valid_count"]
            ),
            "missing_sequence_count": len(missing_sequences),
            "missing_case_count": len(missing_cases),
            "case_sequence_gap_count": len(case_sequence_gaps),
            "missing_sweep_point_count": len(missing_points),
            "diagnostic_explicitly_infeasible_sweep_point_count": len(
                diagnostic_infeasible_points
            ),
            "statistical_floor_status": floor_status,
            "recommended_floor_status": recommended_status,
            "raw_rejection_rate_pct": (
                0.0
                if totals["scored_feasible_count"] == 0
                else totals["rejection_count"]
                / totals["scored_feasible_count"]
                * 100.0
            ),
            "gates": gates,
            "accepted": all(gates.values()),
        },
        "gaps": {
            "missing_sequences": missing_sequences,
            "missing_cases": missing_cases,
            "case_sequence_gaps": case_sequence_gaps,
            "missing_sweep_points": missing_points,
        },
        "strata": strata,
        "adaptive_sampling": adaptive,
        "fixed_sampling": fixed_sampling,
        "adjacent_pre_rebased_delays": adjacent_pre_rebased_delays,
        "strict_final_json_provenance": strict_document_provenance,
    }


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable acceptance report from ``validate_samples``."""

    summary = report["summary"]
    gates = summary["gates"]
    lines = [
        "# Pairwise-overlap corpus acceptance report",
        "",
        f"Overall acceptance: **{'PASS' if summary['accepted'] else 'FAIL'}**",
        "",
        "## Corpus summary",
        "",
        *_markdown_table(
            (
                "Raw retained rows",
                "Strata",
                "Feasible rows",
                "Runner-valid rows",
                "Raw accepted rows",
                "Estimator-valid samples",
                "Controller invalid",
                "Discarded tail",
            ),
            [(
                summary["sample_count"],
                summary["stratum_count"],
                summary["feasible_count"],
                summary["runner_valid_count"],
                summary["raw_accepted_valid_count"],
                summary["estimator_valid_sample_count"],
                (
                    summary["adaptive_controller_invalid_sample_count"]
                    if summary["adaptive_controller_invalid_sample_count"] is not None
                    else "n/a"
                ),
                (
                    summary["adaptive_controller_discarded_after_stop_count"]
                    if summary["adaptive_controller_discarded_after_stop_count"] is not None
                    else "n/a"
                ),
            )],
        ),
        "",
        (
            "For adaptive runs, **Estimator-valid samples** is authoritative. "
            "Raw accepted rows include otherwise-valid tail-of-batch measurements "
            "that completed after a cell stopped and are not used by the estimator."
            if report["adaptive_sampling"]["present"]
            else "Raw accepted rows are the estimator sample count for this fixed-sampling run."
        ),
        "",
        "## Runtime-start scoring disposition",
        "",
        *_markdown_table(
            (
                "Scored",
                "Scored feasible",
                "Unscored",
                "Unresolved",
                "Proven static",
                "Contract failures",
                "Coverage holes",
            ),
            [(
                summary["scored_count"],
                summary["scored_feasible_count"],
                summary["unscored_count"],
                summary["runtime_start_geometry_unresolved_count"],
                summary["runtime_start_geometry_static_exempt_count"],
                summary["runtime_start_geometry_contract_failure_count"],
                summary["coverage_hole_count"],
            )],
        ),
        "",
        "## Acceptance gates",
        "",
        *_markdown_table(
            ("Gate", "Status"),
            ((name, "PASS" if passed else "FAIL") for name, passed in gates.items()),
        ),
        "",
        "## Health failures",
        "",
        *_markdown_table(
            (
                "Geometry rejects",
                "Geometry unknown",
                "Geometry-baseline rejects",
                "Geometry-baseline unknown",
                "Drift rejects",
                "Drift unknown",
                "Correctness failures",
                "Correctness unknown",
                "Raw rejection rate",
            ),
            [(
                summary["geometry_failure_count"],
                summary["geometry_unknown_count"],
                summary["geometry_baseline_failure_count"],
                summary["geometry_baseline_unknown_count"],
                summary["drift_failure_count"],
                summary["drift_unknown_count"],
                summary["correctness_failure_count"],
                summary["correctness_unknown_count"],
                f"{summary['raw_rejection_rate_pct']:.3f}%",
            )],
        ),
        "",
        "## Missing coverage",
        "",
        *_markdown_table(
            ("Missing sequences", "Missing cases", "Case/sequence gaps", "Sweep-point gaps"),
            [(
                summary["missing_sequence_count"],
                summary["missing_case_count"],
                summary["case_sequence_gap_count"],
                summary["missing_sweep_point_count"],
            )],
        ),
        "",
        "## Informational uniform floors",
        "",
        (
            "These fixed 50/100 checks are reported for comparison only. "
            "Acceptance uses each stratum's duration-based floor shown below."
        ),
        "",
        *_markdown_table(
            ("Floor", "Status", "Failing feasible strata"),
            (
                (floor, "PASS" if value["met"] else "FAIL", value["failing_feasible_strata"])
                for floor, value in summary["statistical_floor_status"].items()
            ),
        ),
        "",
        "## Adaptive sampling",
        "",
        (
            "Endpoint precision criterion: `"
            f"{report['adaptive_sampling']['precision_criterion']}`. "
            f"Independently verified endpoint-metric cells: "
            f"{report['adaptive_sampling']['endpoint_precision_cell_count']}; "
            f"legacy reported-flag cells: "
            f"{report['adaptive_sampling']['legacy_precision_cell_count']}."
        ),
        "",
        *_markdown_table(
            (
                "Present",
                "Cells",
                "Ingested",
                "Valid",
                "Invalid",
                "Discarded tail",
                "Rejection rate",
                "Diagnostic failures",
            ),
            [(
                "yes" if report["adaptive_sampling"]["present"] else "no",
                report["adaptive_sampling"]["cell_count"],
                report["adaptive_sampling"]["ingested_attempt_count"],
                report["adaptive_sampling"]["valid_sample_count"],
                report["adaptive_sampling"]["invalid_sample_count"],
                report["adaptive_sampling"]["discarded_after_stop_count"],
                f"{report['adaptive_sampling']['feasible_rejection_rate_pct']:.3f}%",
                report["adaptive_sampling"]["malformed_or_failed_diagnostic_count"],
            )],
        ),
        "",
        "## Per-stratum results",
        "",
        *_markdown_table(
            (
                "Sequence",
                "Case",
                "Victim",
                "Aggressor",
                "Dispatch",
                "Target %",
                "Requested",
                "Feasible",
                "Runner valid",
                "Raw accepted",
                "Controller valid",
                "Controller invalid",
                "Discarded tail",
                "Geometry fail",
                "Geometry-baseline fail",
                "Drift fail",
                "Correctness fail",
                "Rejected",
                "Reject rate",
                "Floor",
                "Floor met",
            ),
            (
                (
                    row["seq_len"],
                    row["case_key"],
                    row["victim_family"],
                    row["aggressor_family"],
                    row["dispatch_mode"],
                    _target_key(row["requested_overlap_pct"]),
                    row["requested_count"],
                    row["feasible_count"],
                    row["runner_valid_count"],
                    row["accepted_valid_count"],
                    row.get("adaptive_valid_sample_count", "n/a"),
                    row.get("adaptive_invalid_sample_count", "n/a"),
                    row.get("adaptive_discarded_after_stop_count", "n/a"),
                    row["geometry_failure_count"],
                    row["geometry_baseline_failure_count"],
                    row["drift_failure_count"],
                    row["correctness_failure_count"],
                    row["rejection_count"],
                    f"{row['rejection_rate_pct']:.3f}%",
                    row["recommended_floor"],
                    "yes" if row["recommended_floor_met"] else "no",
                )
                for row in report["strata"]
            ),
        ),
        "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    """Write sibling JSON and Markdown reports and return their paths."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    markdown_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, markdown_path
