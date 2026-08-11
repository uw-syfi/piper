"""Mandatory validation for profile-and-simulate artifacts and replays."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .profiling import NORMAL_PROFILE_MODE, estimate_profile_time


class ProfileAndSimulateValidationError(ValueError):
    """Raised when a mandatory profile-and-simulate invariant fails."""


def _close(left: float, right: float, *, tolerance_us: float) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=tolerance_us
    )


def validate_kineto_profile_results(
    results: Sequence[Any],
    activity_equivalence: Mapping[str, Any],
    *,
    tolerance_us: float = 1e-3,
    raise_on_error: bool = True,
) -> dict[str, Any]:
    """Validate activity identity, attribution coverage, and ingested timing."""

    errors: list[str] = []
    checked_nodes = 0
    checked_occurrences = 0
    checked_launches = 0
    if not bool(activity_equivalence.get("exact_match", False)):
        errors.append(
            "normal/IF GPU activity identity differs for "
            f"{activity_equivalence.get('mismatched_node_ranges', 0)} "
            "node-range occurrences"
        )

    for result in results:
        audit_by_uid = result.device_metadata.get(
            "kineto_attribution_audit", {}
        )
        for uid, profile in result.node_profiles.items():
            if not bool(getattr(profile, "executed", True)):
                continue
            checked_nodes += 1
            audit = audit_by_uid.get(uid)
            if audit is None:
                errors.append(
                    f"mode={result.profile_mode} rank={result.global_rank} "
                    f"uid={uid}: missing Kineto attribution audit"
                )
                continue
            unmatched = int(audit.get("unmatched_launch_count", 0))
            launch_count = int(audit.get("launch_count", 0))
            matched = int(audit.get("matched_launch_count", 0))
            checked_launches += launch_count
            if unmatched or matched != launch_count:
                errors.append(
                    f"mode={result.profile_mode} rank={result.global_rank} "
                    f"uid={uid}: launch coverage {matched}/{launch_count}, "
                    f"unmatched={unmatched}"
                )
            occurrences = list(audit.get("range_activity_signatures", ()))
            range_count = int(audit.get("range_count", len(occurrences)))
            duration_samples = tuple(
                float(value)
                for value in audit.get("simulator_duration_samples_us", ())
            )
            checked_occurrences += len(occurrences)
            if len(occurrences) != range_count:
                errors.append(
                    f"mode={result.profile_mode} rank={result.global_rank} "
                    f"uid={uid}: range_count={range_count}, "
                    f"occurrences={len(occurrences)}"
                )
            if len(duration_samples) != range_count:
                errors.append(
                    f"mode={result.profile_mode} rank={result.global_rank} "
                    f"uid={uid}: duration samples={len(duration_samples)}, "
                    f"ranges={range_count}"
                )
            source = audit.get("simulator_duration_source")
            for occurrence_index, occurrence in enumerate(occurrences):
                activity_sum = float(
                    occurrence.get("activity_time_sum_us", 0.0)
                )
                activity_union = float(
                    occurrence.get("activity_union_time_us", 0.0)
                )
                overlap = float(
                    occurrence.get("activity_intra_node_overlap_us", 0.0)
                )
                envelope = float(
                    occurrence.get("activity_envelope_time_us", 0.0)
                )
                idle_gap = float(
                    occurrence.get("activity_idle_gap_us", 0.0)
                )
                if not _close(
                    activity_union,
                    activity_sum - overlap,
                    tolerance_us=tolerance_us,
                ):
                    errors.append(
                        f"mode={result.profile_mode} rank={result.global_rank} "
                        f"uid={uid} occurrence={occurrence_index}: "
                        "activity union identity failed"
                    )
                if not _close(
                    envelope,
                    activity_union + idle_gap,
                    tolerance_us=tolerance_us,
                ):
                    errors.append(
                        f"mode={result.profile_mode} rank={result.global_rank} "
                        f"uid={uid} occurrence={occurrence_index}: "
                        "activity envelope identity failed"
                    )
                if (
                    source == "correlated_gpu_activity_envelope"
                    and occurrence_index < len(duration_samples)
                    and not _close(
                        duration_samples[occurrence_index],
                        envelope,
                        tolerance_us=tolerance_us,
                    )
                ):
                    errors.append(
                        f"mode={result.profile_mode} rank={result.global_rank} "
                        f"uid={uid} occurrence={occurrence_index}: stored "
                        "simulator duration does not equal activity envelope"
                    )
            profile_samples = tuple(
                float(value) for value in getattr(profile, "samples_us", ())
            )
            if len(profile_samples) != len(duration_samples) or any(
                not _close(left, right, tolerance_us=tolerance_us)
                for left, right in zip(profile_samples, duration_samples)
            ):
                errors.append(
                    f"mode={result.profile_mode} rank={result.global_rank} "
                    f"uid={uid}: NodeProfile samples differ from attributed "
                    "simulator durations"
                )
            if profile_samples and not _close(
                float(profile.estimate_us),
                estimate_profile_time(
                    profile_samples,
                    str(getattr(profile, "estimator", "median")),
                ),
                tolerance_us=1e-6,
            ):
                errors.append(
                    f"mode={result.profile_mode} rank={result.global_rank} "
                    f"uid={uid}: estimate does not match its configured "
                    "sample estimator"
                )

    report = {
        "name": "kineto_activity_and_timing",
        "passed": not errors,
        "actors": len(results),
        "checked_nodes": checked_nodes,
        "checked_occurrences": checked_occurrences,
        "checked_launches": checked_launches,
        "exact_activity_match": bool(
            activity_equivalence.get("exact_match", False)
        ),
        "mismatched_node_ranges": int(
            activity_equivalence.get("mismatched_node_ranges", 0)
        ),
        "errors": errors,
    }
    if errors and raise_on_error:
        raise ProfileAndSimulateValidationError(
            "Kineto profile validation failed: " + "; ".join(errors[:8])
        )
    return report


def _event_map(result: Any) -> dict[tuple[int, str], Any]:
    mapped: dict[tuple[int, str], Any] = {}
    for event in result.timeline:
        key = (int(event.operation.global_rank), str(event.operation.uid))
        if key in mapped:
            raise ProfileAndSimulateValidationError(
                f"simulation timeline contains duplicate event {key!r}"
            )
        mapped[key] = event
    return mapped


def validate_simulation_endpoints(comparison: Any) -> dict[str, Any]:
    """Validate that normal and IF replay the same logical execution graph."""

    normal = _event_map(comparison.normal)
    interference_free = _event_map(comparison.interference_free)
    errors: list[str] = []
    if set(normal) != set(interference_free):
        errors.append(
            "normal and IF timeline event sets differ: "
            f"normal_only={sorted(set(normal) - set(interference_free))[:8]}, "
            f"if_only={sorted(set(interference_free) - set(normal))[:8]}"
        )
    for key in sorted(set(normal) & set(interference_free)):
        normal_event = normal[key]
        if_event = interference_free[key]
        for field in ("node_kind", "stream", "communication_id"):
            if getattr(normal_event, field) != getattr(if_event, field):
                errors.append(
                    f"event {key!r} differs in {field}: "
                    f"normal={getattr(normal_event, field)!r}, "
                    f"IF={getattr(if_event, field)!r}"
                )
        for mode, event in (
            ("normal", normal_event),
            ("interference_free", if_event),
        ):
            if not _close(
                event.end_us - event.start_us,
                event.effective_duration_us,
                tolerance_us=1e-6,
            ):
                errors.append(
                    f"{mode} event {key!r}: interval duration differs from "
                    "effective duration"
                )
    report = {
        "name": "simulation_endpoints",
        "passed": not errors,
        "normal_events": len(normal),
        "interference_free_events": len(interference_free),
        "same_event_set": set(normal) == set(interference_free),
        "errors": errors,
    }
    if errors:
        raise ProfileAndSimulateValidationError(
            "simulation endpoint validation failed: "
            + "; ".join(errors[:8])
        )
    return report


def _timeline_dict(result: Any) -> list[dict[str, Any]]:
    return [
        {
            "global_rank": event.operation.global_rank,
            "uid": event.operation.uid,
            "node_kind": event.node_kind,
            "stream": event.stream,
            "start_us": event.start_us,
            "end_us": event.end_us,
            "profile_duration_us": event.profile_duration_us,
            "effective_duration_us": event.effective_duration_us,
            "communication_id": event.communication_id,
        }
        for event in result.timeline
    ]


def build_and_validate_attributed_trace(
    normal_result: Any,
    attribution: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the attributed trace and prove it preserves every input event."""

    from scripts.visualize_simulation_savings import (
        build_attributed_simulation_trace,
    )

    source = _timeline_dict(normal_result)
    attribution_payload = attribution.to_dict()
    trace = build_attributed_simulation_trace(source, attribution_payload)
    rendered = [
        event for event in trace.get("traceEvents", ())
        if event.get("ph") == "X"
    ]
    errors: list[str] = []
    if len(source) != len(rendered):
        errors.append(
            f"source events={len(source)}, rendered events={len(rendered)}"
        )
    for index, (original, event) in enumerate(zip(source, rendered)):
        args = event.get("args", {})
        expected = (
            int(original["global_rank"]),
            str(original["uid"]),
            str(original["node_kind"]),
            str(original["stream"]),
            original.get("communication_id"),
        )
        actual = (
            int(args.get("global_rank", -1)),
            str(args.get("uid", "")),
            str(args.get("node_kind", "")),
            str(args.get("stream", "")),
            args.get("communication_id"),
        )
        if actual != expected:
            errors.append(
                f"rendered event {index} identity differs: "
                f"expected={expected!r}, actual={actual!r}"
            )
        if not _close(event.get("ts", -1.0), original["start_us"], tolerance_us=1e-9):
            errors.append(f"rendered event {index} start changed")
        if not _close(
            event.get("dur", -1.0),
            original["end_us"] - original["start_us"],
            tolerance_us=1e-9,
        ):
            errors.append(f"rendered event {index} duration changed")
    summary = trace.get("piperSavingsAttribution", {})
    if summary.get("unmatched_feature_ids"):
        errors.append(
            "attribution features did not match the normal timeline: "
            f"{summary['unmatched_feature_ids'][:8]!r}"
        )
    if int(summary.get("matched_feature_count", -1)) != len(
        attribution.features
    ):
        errors.append(
            "not every attribution feature matched a normal timeline event"
        )
    if not bool(summary.get("exposed_savings_validation_passed", False)):
        errors.append(
            "unique-feature exposed-savings checksum does not equal the "
            "simulator normal-to-IF step delta: residual="
            f"{summary.get('exposed_savings_validation_residual_us')!r} us"
        )
    trace_checksum = sum(
        float(event.get("args", {}).get("trace_checksum_share_us", 0.0))
        for event in rendered
    )
    net_exposed = float(summary.get("net_exposed_step_savings_us", 0.0))
    checksum_tolerance = float(
        summary.get("exposed_savings_validation_tolerance_us", 1e-6)
    )
    trace_checksum_residual = trace_checksum - net_exposed
    if not _close(
        trace_checksum,
        net_exposed,
        tolerance_us=checksum_tolerance,
    ):
        errors.append(
            "per-event trace checksum shares double-count or omit feature "
            f"savings: residual={trace_checksum_residual!r} us"
        )
    report = {
        "name": "attributed_trace_losslessness",
        "passed": not errors,
        "contains_cuda_kernels": False,
        "source_events": len(source),
        "rendered_events": len(rendered),
        "matched_features": int(summary.get("matched_feature_count", 0)),
        "total_features": len(attribution.features),
        "positive_exposed_step_savings_us": summary.get(
            "positive_exposed_step_savings_us"
        ),
        "negative_exposed_step_savings_us": summary.get(
            "negative_exposed_step_savings_us"
        ),
        "net_exposed_step_savings_us": net_exposed,
        "net_exposed_step_savings_pct": summary.get(
            "net_exposed_step_savings_pct"
        ),
        "simulator_step_savings_us": summary.get(
            "simulator_step_savings_us"
        ),
        "exposed_savings_validation_residual_us": summary.get(
            "exposed_savings_validation_residual_us"
        ),
        "trace_checksum_sum_us": trace_checksum,
        "trace_checksum_residual_us": trace_checksum_residual,
        "errors": errors,
    }
    if errors:
        raise ProfileAndSimulateValidationError(
            "attributed trace validation failed: " + "; ".join(errors[:8])
        )
    return trace, report


def validate_measured_normal_samples(
    plan: Any,
    comparison: Any,
    *,
    warning_threshold_pct: float = 2.0,
) -> dict[str, Any]:
    """Validate simulated normal against ordinary uninstrumented measurement.

    The synchronized Kineto step remains useful as a separate replay-fidelity
    and instrumentation-overhead diagnostic, but it is deliberately not the
    measured truth used by validation 4.
    """

    instrumented = tuple(
        float(value)
        for value in plan.profile_metadata.get(
            "instrumented_global_step_samples_us_by_mode", {}
        ).get(NORMAL_PROFILE_MODE, ())
    )
    actual_statistics = getattr(comparison, "actual_statistics", None)
    normal_statistics = getattr(comparison, "normal_statistics", None)
    instrumented_statistics = getattr(
        comparison, "instrumented_normal_statistics", None
    )
    if actual_statistics is None or normal_statistics is None:
        return {
            "name": "measured_vs_simulated_normal",
            "passed": True,
            "available": False,
            "warning": "no ordinary measured normal step samples",
            "warning_threshold_pct": warning_threshold_pct,
        }

    measured_estimate = float(actual_statistics.estimate)
    simulated_estimate = float(normal_statistics.estimate)
    signed_error = (
        (simulated_estimate - measured_estimate) / measured_estimate * 100.0
    )
    absolute_error = abs(signed_error)
    profile_replay = None
    instrumentation = None
    if instrumented_statistics is not None:
        instrumented_estimate = float(instrumented_statistics.estimate)
        profile_replay_signed = (
            (simulated_estimate - instrumented_estimate)
            / instrumented_estimate
            * 100.0
        )
        instrumentation_signed = (
            (instrumented_estimate - measured_estimate)
            / measured_estimate
            * 100.0
        )
        profile_replay = {
            "instrumented_normal_step_estimate_us": instrumented_estimate,
            "simulated_normal_step_estimate_us": simulated_estimate,
            "signed_error_pct": profile_replay_signed,
            "absolute_error_pct": abs(profile_replay_signed),
            "instrumented_samples": int(instrumented_statistics.samples),
        }
        instrumentation = {
            "measured_normal_step_estimate_us": measured_estimate,
            "instrumented_normal_step_estimate_us": instrumented_estimate,
            "signed_error_pct": instrumentation_signed,
            "absolute_error_pct": abs(instrumentation_signed),
            "measured_samples": int(actual_statistics.samples),
            "instrumented_samples": int(instrumented_statistics.samples),
        }
    return {
        "name": "measured_vs_simulated_normal",
        "passed": True,
        "available": True,
        "measured_samples": int(actual_statistics.samples),
        "simulated_samples": int(normal_statistics.samples),
        "measured_normal_step_estimate_us": measured_estimate,
        "simulated_normal_step_estimate_us": simulated_estimate,
        "signed_median_step_error_pct": signed_error,
        "median_absolute_step_error_pct": absolute_error,
        "instrumented_profile_replay_error": profile_replay,
        "kineto_cupti_instrumentation_error": instrumentation,
        "warning_threshold_pct": warning_threshold_pct,
        "warning": (
            None
            if absolute_error <= warning_threshold_pct
            else (
                "median absolute measured-vs-simulated normal step error "
                f"{absolute_error:.3f}% exceeds {warning_threshold_pct:.3f}%"
            )
        ),
    }


__all__ = [
    "ProfileAndSimulateValidationError",
    "build_and_validate_attributed_trace",
    "validate_kineto_profile_results",
    "validate_measured_normal_samples",
    "validate_simulation_endpoints",
]
