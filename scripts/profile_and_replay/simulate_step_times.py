"""Replay normal and interference-free Piper step-time profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from src.profiling import ProfiledTrainingPlan
from src.simulator import simulate_profile_modes, simulate_training_step


def _result_dict(result) -> dict:
    return {
        "profile_mode": result.profile_mode,
        "step_time_us": result.step_time_us,
        "step_time_s": result.step_time_s,
        "per_rank_makespan_us": {
            str(rank): duration
            for rank, duration in sorted(result.per_rank_makespan_us.items())
        },
        "warnings": list(result.warnings),
        "critical_path": [
            {
                "operation_id": entry.operation_id,
                "members": [
                    {"global_rank": member.global_rank, "uid": member.uid}
                    for member in entry.members
                ],
                "start_us": entry.start_us,
                "end_us": entry.end_us,
                "duration_us": entry.effective_duration_us,
            }
            for entry in result.critical_path
        ],
        "timeline": [
            {
                "global_rank": event.operation.global_rank,
                "uid": event.operation.uid,
                "node_kind": event.node_kind,
                "stream": event.stream,
                "start_us": event.start_us,
                "end_us": event.end_us,
                "profile_duration_us": event.profile_duration_us,
                "profile_start_offset_us": event.profile_start_offset_us,
                "effective_duration_us": event.effective_duration_us,
                "communication_id": event.communication_id,
            }
            for event in result.timeline
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay normal and interference-free Piper step-time profiles"
    )
    parser.add_argument("profile", type=Path, help="ProfiledTrainingPlan JSON artifact")
    parser.add_argument("--output", type=Path, help="Optional result JSON path")
    parser.add_argument(
        "--mode",
        choices=("auto", "interference_free", "normal", "both"),
        default="auto",
        help="Profile dataset to replay; auto uses both when both are present.",
    )
    parser.add_argument(
        "--actual-step-time-s",
        action="append",
        type=float,
        default=[],
        help="Optional measured step sample in seconds; repeat for calibration.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Warn instead of failing for surplus/incomplete optional metadata",
    )
    args = parser.parse_args()

    plan = ProfiledTrainingPlan.load_json(args.profile)
    mode = args.mode
    if mode == "auto":
        mode = (
            "both"
            if {"interference_free", "normal"}
            <= set(plan.available_profile_modes)
            else "interference_free"
        )
    if mode == "both":
        comparison = simulate_profile_modes(
            plan,
            actual_step_samples_us=tuple(
                sample * 1_000_000.0 for sample in args.actual_step_time_s
            )
            or None,
            strict=not args.no_strict,
        )
        payload = {
            "interference_free": _result_dict(comparison.interference_free),
            "normal": _result_dict(comparison.normal),
            "statistics_us": {
                "interference_free": comparison.interference_free_statistics.to_dict(),
                "normal": comparison.normal_statistics.to_dict(),
                "actual": (
                    None
                    if comparison.actual_statistics is None
                    else comparison.actual_statistics.to_dict()
                ),
                "instrumented_normal_profile": (
                    None
                    if comparison.instrumented_normal_statistics is None
                    else comparison.instrumented_normal_statistics.to_dict()
                ),
            },
            "normal_prediction_error_pct": (
                None
                if comparison.normal_prediction_error is None
                else comparison.normal_prediction_error.signed_pct
            ),
            "normal_absolute_error_pct": (
                None
                if comparison.normal_prediction_error is None
                else comparison.normal_prediction_error.absolute_pct
            ),
            "predicted_interference_reduction_pct": (
                comparison.predicted_interference_reduction_pct
            ),
            "profile_event_overhead_pct": comparison.profile_event_overhead_pct,
        }
    else:
        payload = _result_dict(
            simulate_training_step(
                plan, profile_mode=mode, strict=not args.no_strict
            )
        )

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
