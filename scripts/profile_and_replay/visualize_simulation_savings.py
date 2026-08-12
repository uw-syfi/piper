#!/usr/bin/env python3
"""Render simulator savings attribution as a Perfetto/Chrome trace.

The input timeline is the *normal simulator replay*, rather than a hardware
Kineto timeline.  This is important: the attribution explains a change in the
simulator's makespan, so the highlighted intervals must use the same schedule
and timestamps that the counterfactual replays used.

``NORMAL_INPUT`` may be either a ``ProfiledTrainingPlan`` JSON artifact (the
normal replay is generated automatically) or JSON emitted by
``scripts/profile_and_replay/simulate_step_times.py``.  The latter may contain a top-level
``timeline`` or a ``normal.timeline`` comparison result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from src.profiling import NORMAL_PROFILE_MODE, ProfiledTrainingPlan
from src.simulator import SimulationResult, simulate_training_step


@dataclass(frozen=True)
class _Feature:
    feature_id: str
    pp_indices: frozenset[int]
    uids: frozenset[str]
    ranks: frozenset[int]
    node_kinds: frozenset[str]
    attributed_savings_us: float
    raw_profile_savings_us: float
    exposure_probability: float
    hidden_probability: float
    negative_probability: float
    share_of_step_savings_pct: float | None

    @property
    def communication_identity(self) -> tuple[str, str] | None:
        if not self.feature_id.startswith("comm:"):
            return None
        _prefix, family, identity = self.feature_id.split(":", 2)
        return family, identity


def _finite_float(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return result


def _parse_features(attribution: Mapping[str, Any]) -> tuple[_Feature, ...]:
    features: list[_Feature] = []
    for row in attribution.get("features", ()):
        descriptor = row.get("feature", row)
        feature_id = str(descriptor["feature_id"])
        share = row.get("share_of_step_savings_pct")
        features.append(
            _Feature(
                feature_id=feature_id,
                pp_indices=frozenset(
                    int(value) for value in descriptor.get("pp_indices", ())
                ),
                uids=frozenset(str(value) for value in descriptor.get("uids", ())),
                ranks=frozenset(int(value) for value in descriptor.get("ranks", ())),
                node_kinds=frozenset(
                    str(value) for value in descriptor.get("node_kinds", ())
                ),
                attributed_savings_us=_finite_float(
                    row.get("attributed_savings_us", 0.0),
                    field=f"{feature_id}.attributed_savings_us",
                ),
                raw_profile_savings_us=_finite_float(
                    descriptor.get("raw_profile_savings_us", 0.0),
                    field=f"{feature_id}.raw_profile_savings_us",
                ),
                exposure_probability=_finite_float(
                    row.get("positive_exposure_probability", 0.0),
                    field=f"{feature_id}.positive_exposure_probability",
                ),
                hidden_probability=_finite_float(
                    row.get("hidden_probability", 0.0),
                    field=f"{feature_id}.hidden_probability",
                ),
                negative_probability=_finite_float(
                    row.get("negative_probability", 0.0),
                    field=f"{feature_id}.negative_probability",
                ),
                share_of_step_savings_pct=(
                    None
                    if share is None
                    else _finite_float(
                        share, field=f"{feature_id}.share_of_step_savings_pct"
                    )
                ),
            )
        )
    if not features:
        raise ValueError("attribution contains no features")
    feature_ids = [feature.feature_id for feature in features]
    duplicates = sorted(
        feature_id
        for feature_id, count in Counter(feature_ids).items()
        if count > 1
    )
    if duplicates:
        raise ValueError(f"attribution contains duplicate feature IDs: {duplicates!r}")
    return tuple(features)


def _matches_communication(event: Mapping[str, Any], feature: _Feature) -> bool:
    parsed = feature.communication_identity
    communication_id = event.get("communication_id")
    if parsed is None or communication_id is None:
        return False
    family, identity = parsed
    observed = str(communication_id)
    if family == "p2p":
        # The feature is atomic across all DP replicas, while the simulator
        # appends the DP lane to each physical P2P rendezvous ID.
        return observed == identity or observed.startswith(f"{identity}:")
    # Collective simulator IDs are ``NODE_KIND:comm_id``.  Accept a bare
    # identity too so older exported timelines remain renderable.
    return observed == identity or observed.lower() == f"{family}:{identity}".lower()


def _match_feature(
    event: Mapping[str, Any], features: Sequence[_Feature]
) -> _Feature | None:
    rank = int(event["global_rank"])
    uid = str(event["uid"])
    kind = str(event.get("node_kind", ""))

    communication_matches = [
        feature
        for feature in features
        if _matches_communication(event, feature)
        and (not feature.ranks or rank in feature.ranks)
        and (not feature.node_kinds or kind in feature.node_kinds)
    ]
    if len(communication_matches) == 1:
        return communication_matches[0]
    if len(communication_matches) > 1:
        exact = [feature for feature in communication_matches if uid in feature.uids]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(
            f"event rank={rank} uid={uid!r} ambiguously matches communication "
            f"features {[item.feature_id for item in communication_matches]!r}"
        )

    node_matches = [
        feature
        for feature in features
        if feature.communication_identity is None
        and uid in feature.uids
        and (not feature.ranks or rank in feature.ranks)
        and (not feature.node_kinds or kind in feature.node_kinds)
    ]
    if len(node_matches) > 1:
        raise ValueError(
            f"event rank={rank} uid={uid!r} ambiguously matches node features "
            f"{[item.feature_id for item in node_matches]!r}"
        )
    return node_matches[0] if node_matches else None


def _status(feature: _Feature, threshold_us: float) -> str:
    if feature.attributed_savings_us > threshold_us:
        return "exposed"
    if feature.attributed_savings_us < -threshold_us:
        return "negative"
    return "hidden"


def _color(feature: _Feature | None, threshold_us: float, largest_us: float) -> str:
    if feature is None:
        return "thread_state_unknown"
    status = _status(feature, threshold_us)
    if status == "negative":
        return "good"
    if status == "hidden":
        return "thread_state_sleeping"
    relative = feature.attributed_savings_us / max(largest_us, threshold_us, 1e-12)
    if relative >= 0.50:
        return "terrible"
    if relative >= 0.10:
        return "bad"
    return "cq_build_abandoned"


def _timeline_dict(result: SimulationResult) -> list[dict[str, Any]]:
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


def build_attributed_simulation_trace(
    timeline: Sequence[Mapping[str, Any]],
    attribution: Mapping[str, Any],
    *,
    exposure_threshold_us: float | None = None,
) -> dict[str, Any]:
    """Build a Chrome trace with feature-level savings on normal events."""

    features = _parse_features(attribution)
    threshold_us = (
        _finite_float(
            attribution.get("exposure_threshold_us", 1.0),
            field="exposure_threshold_us",
        )
        if exposure_threshold_us is None
        else _finite_float(exposure_threshold_us, field="exposure_threshold_us")
    )
    if threshold_us < 0.0:
        raise ValueError("exposure_threshold_us must be non-negative")
    largest_us = max(
        (feature.attributed_savings_us for feature in features), default=0.0
    )

    matches: list[_Feature | None] = [
        _match_feature(event, features) for event in timeline
    ]
    feature_event_counts = Counter(
        feature.feature_id for feature in matches if feature is not None
    )
    matched_feature_ids = set(feature_event_counts)
    streams = sorted(
        {
            (int(event["global_rank"]), str(event.get("stream", "unknown")))
            for event in timeline
        }
    )
    track_id = {key: index + 1 for index, key in enumerate(streams)}
    ranks = sorted({rank for rank, _stream in streams})

    trace_events: list[dict[str, Any]] = []
    for rank in ranks:
        trace_events.extend(
            [
                {
                    "name": "process_name",
                    "ph": "M",
                    "pid": rank,
                    "tid": 0,
                    "args": {"name": f"simulated GPU rank {rank}"},
                },
                {
                    "name": "process_sort_index",
                    "ph": "M",
                    "pid": rank,
                    "tid": 0,
                    "args": {"sort_index": rank},
                },
            ]
        )
    for rank, stream in streams:
        tid = track_id[(rank, stream)]
        trace_events.extend(
            [
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": rank,
                    "tid": tid,
                    "args": {"name": stream},
                },
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": rank,
                    "tid": tid,
                    "args": {"sort_index": tid},
                },
            ]
        )

    status_counts: Counter[str] = Counter()
    for event, feature in zip(timeline, matches):
        rank = int(event["global_rank"])
        stream = str(event.get("stream", "unknown"))
        start_us = _finite_float(event["start_us"], field="event.start_us")
        end_us = _finite_float(event["end_us"], field="event.end_us")
        if end_us < start_us:
            raise ValueError(
                f"event {event.get('uid')!r} ends before it starts: "
                f"{start_us} > {end_us}"
            )
        status = "unattributed" if feature is None else _status(feature, threshold_us)
        status_counts[status] += 1
        args: dict[str, Any] = {
            "global_rank": rank,
            "uid": str(event["uid"]),
            "node_kind": str(event.get("node_kind", "")),
            "stream": stream,
            "normal_start_us": start_us,
            "normal_end_us": end_us,
            "normal_duration_us": end_us - start_us,
            "profile_duration_us": event.get("profile_duration_us"),
            "effective_duration_us": event.get("effective_duration_us"),
            "communication_id": event.get("communication_id"),
            "savings_status": status,
        }
        name = str(event["uid"])
        if feature is not None:
            args.update(
                {
                    "attribution_feature_id": feature.feature_id,
                    "attributed_savings_us": feature.attributed_savings_us,
                    # ``attributed_savings_us`` is a logical-feature value and
                    # is intentionally repeated on every physical event that
                    # represents the feature.  This equal share is the value
                    # to sum across trace events when checking the step delta.
                    "trace_checksum_share_us": (
                        feature.attributed_savings_us
                        / feature_event_counts[feature.feature_id]
                    ),
                    "raw_profile_savings_us": feature.raw_profile_savings_us,
                    "exposure_probability": feature.exposure_probability,
                    "hidden_probability": feature.hidden_probability,
                    "negative_probability": feature.negative_probability,
                    "share_of_step_savings_pct": (
                        feature.share_of_step_savings_pct
                    ),
                    "feature_event_count": feature_event_counts[feature.feature_id],
                    "atomic_communication_feature": (
                        feature.communication_identity is not None
                    ),
                }
            )
            if status == "exposed":
                name = f"★ {name} (+{feature.attributed_savings_us:.1f} us)"
            elif status == "negative":
                name = f"▼ {name} ({feature.attributed_savings_us:.1f} us)"
        trace_events.append(
            {
                "name": name,
                "cat": f"piper.savings.{status}",
                "ph": "X",
                "ts": start_us,
                "dur": end_us - start_us,
                "pid": rank,
                "tid": track_id[(rank, stream)],
                "cname": _color(feature, threshold_us, largest_us),
                "args": args,
            }
        )

    total_features = len(features)
    unmatched_feature_ids = sorted(
        feature.feature_id
        for feature in features
        if feature.feature_id not in matched_feature_ids
    )
    positive_exposed_step_savings_us = sum(
        max(0.0, feature.attributed_savings_us) for feature in features
    )
    negative_exposed_step_savings_us = sum(
        min(0.0, feature.attributed_savings_us) for feature in features
    )
    net_exposed_step_savings_us = (
        positive_exposed_step_savings_us + negative_exposed_step_savings_us
    )
    simulator_step_savings_us = _finite_float(
        attribution.get(
            "total_step_savings_us",
            _finite_float(
                attribution.get("normal_step_time_us", 0.0),
                field="normal_step_time_us",
            )
            - _finite_float(
                attribution.get("interference_free_step_time_us", 0.0),
                field="interference_free_step_time_us",
            ),
        ),
        field="total_step_savings_us",
    )
    exposed_savings_validation_residual_us = (
        net_exposed_step_savings_us - simulator_step_savings_us
    )
    checksum_tolerance_us = max(
        1e-6,
        1e-9
        * max(
            abs(net_exposed_step_savings_us),
            abs(simulator_step_savings_us),
            1.0,
        ),
    )
    summary = {
        "normal_step_time_us": attribution.get("normal_step_time_us"),
        "interference_free_step_time_us": attribution.get(
            "interference_free_step_time_us"
        ),
        "total_step_savings_us": attribution.get("total_step_savings_us"),
        "attributed_step_savings_us": attribution.get(
            "attributed_step_savings_us"
        ),
        "positive_exposed_step_savings_us": (
            positive_exposed_step_savings_us
        ),
        "negative_exposed_step_savings_us": (
            negative_exposed_step_savings_us
        ),
        "net_exposed_step_savings_us": net_exposed_step_savings_us,
        "net_exposed_step_savings_pct": (
            None
            if not attribution.get("normal_step_time_us")
            else 100.0
            * net_exposed_step_savings_us
            / float(attribution["normal_step_time_us"])
        ),
        "simulator_step_savings_us": simulator_step_savings_us,
        "step_time_estimator": attribution.get("step_time_estimator", "point"),
        "sample_iteration_ids": attribution.get("sample_iteration_ids", []),
        "exposed_savings_validation_residual_us": (
            exposed_savings_validation_residual_us
        ),
        "exposed_savings_validation_tolerance_us": checksum_tolerance_us,
        "exposed_savings_validation_passed": (
            abs(exposed_savings_validation_residual_us)
            <= checksum_tolerance_us
        ),
        "exposure_threshold_us": threshold_us,
        "feature_count": total_features,
        "matched_feature_count": len(matched_feature_ids),
        "unmatched_feature_ids": unmatched_feature_ids,
        "timeline_event_count": len(timeline),
        "event_status_counts": dict(sorted(status_counts.items())),
        "color_legend": {
            "terrible": "exposed, >=50% of largest feature contribution",
            "bad": "exposed, >=10% of largest feature contribution",
            "cq_build_abandoned": "exposed, <10% of largest contribution",
            "thread_state_sleeping": "hidden at the selected threshold",
            "good": "negative contribution",
            "thread_state_unknown": "no matching attribution feature",
        },
    }
    return {
        "schemaVersion": 1,
        "displayTimeUnit": "ms",
        "traceEvents": trace_events,
        "piperSavingsAttribution": summary,
    }


def _load_normal_timeline(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("timeline"), list):
        return list(payload["timeline"])
    normal = payload.get("normal") if isinstance(payload, Mapping) else None
    if isinstance(normal, Mapping) and isinstance(normal.get("timeline"), list):
        return list(normal["timeline"])

    # It is a ProfiledTrainingPlan rather than an already exported replay.
    plan = ProfiledTrainingPlan.load_json(path)
    return _timeline_dict(
        simulate_training_step(plan, profile_mode=NORMAL_PROFILE_MODE)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "normal_input",
        type=Path,
        help="ProfiledTrainingPlan JSON or an exported normal simulation JSON",
    )
    parser.add_argument("attribution", type=Path, help="Savings attribution JSON")
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--exposure-threshold-us",
        type=float,
        help="Override the attribution artifact's exposed/hidden threshold",
    )
    args = parser.parse_args()

    attribution = json.loads(args.attribution.read_text(encoding="utf-8"))
    trace = build_attributed_simulation_trace(
        _load_normal_timeline(args.normal_input),
        attribution,
        exposure_threshold_us=args.exposure_threshold_us,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(trace, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    summary = trace["piperSavingsAttribution"]
    print(
        f"wrote {args.output}: {summary['timeline_event_count']} events, "
        f"{summary['matched_feature_count']}/{summary['feature_count']} features "
        "matched"
    )
    if summary["unmatched_feature_ids"]:
        print(
            "unmatched features: "
            + ", ".join(summary["unmatched_feature_ids"]),
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
