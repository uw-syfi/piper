from scripts.profile_and_replay.visualize_simulation_savings import (
    build_attributed_simulation_trace,
)


def _feature(
    feature_id: str,
    *,
    uids: list[str],
    ranks: list[int],
    savings_us: float,
    raw_us: float,
    probability: float,
) -> dict:
    return {
        "feature": {
            "feature_id": feature_id,
            "pp_indices": [0, 1],
            "uids": uids,
            "node_kinds": ["SEND_COMM", "RECV_COMM"],
            "streams": ["send_stream", "recv_stream"],
            "ranks": ranks,
            "binding_count": len(ranks),
            "raw_profile_savings_us": raw_us,
        },
        "attributed_savings_us": savings_us,
        "share_of_step_savings_pct": 60.0,
        "positive_exposure_probability": probability,
        "hidden_probability": 1.0 - probability,
        "negative_probability": 0.0,
    }


def test_trace_highlights_every_endpoint_of_atomic_communication_feature() -> None:
    attribution = {
        "normal_step_time_us": 100.0,
        "interference_free_step_time_us": 93.8,
        "total_step_savings_us": 6.2,
        "attributed_step_savings_us": 6.2,
        "exposure_threshold_us": 1.0,
        "features": [
            _feature(
                "comm:p2p:activation-0->activation-1",
                uids=["send", "recv"],
                ranks=[0, 1],
                savings_us=6.0,
                raw_us=12.0,
                probability=0.75,
            ),
            {
                "feature": {
                    "feature_id": "node:pp0:compute",
                    "pp_indices": [0],
                    "uids": ["compute"],
                    "node_kinds": ["COMPUTE"],
                    "streams": ["default_stream"],
                    "ranks": [0],
                    "binding_count": 1,
                    "raw_profile_savings_us": 5.0,
                },
                "attributed_savings_us": 0.2,
                "share_of_step_savings_pct": 2.0,
                "positive_exposure_probability": 0.25,
                "hidden_probability": 0.75,
                "negative_probability": 0.0,
            },
        ],
    }
    timeline = [
        {
            "global_rank": 0,
            "uid": "send",
            "node_kind": "SEND_COMM",
            "stream": "send_stream",
            "start_us": 10.0,
            "end_us": 15.0,
            "profile_duration_us": 5.0,
            "effective_duration_us": 5.0,
            "communication_id": "activation-0->activation-1:0",
        },
        {
            "global_rank": 1,
            "uid": "recv",
            "node_kind": "RECV_COMM",
            "stream": "recv_stream",
            "start_us": 10.0,
            "end_us": 15.0,
            "profile_duration_us": 5.0,
            "effective_duration_us": 5.0,
            "communication_id": "activation-0->activation-1:0",
        },
        {
            "global_rank": 0,
            "uid": "compute",
            "node_kind": "COMPUTE",
            "stream": "default_stream",
            "start_us": 15.0,
            "end_us": 25.0,
            "profile_duration_us": 10.0,
            "effective_duration_us": 10.0,
            "communication_id": None,
        },
    ]

    trace = build_attributed_simulation_trace(timeline, attribution)
    complete = [event for event in trace["traceEvents"] if event["ph"] == "X"]

    assert complete[0]["name"].startswith("★ ")
    assert complete[1]["name"].startswith("★ ")
    assert complete[0]["cname"] == "terrible"
    assert complete[0]["args"]["atomic_communication_feature"] is True
    assert complete[0]["args"]["feature_event_count"] == 2
    assert complete[0]["args"]["attributed_savings_us"] == 6.0
    assert complete[0]["args"]["trace_checksum_share_us"] == 3.0
    assert complete[1]["args"]["trace_checksum_share_us"] == 3.0
    assert complete[0]["args"]["raw_profile_savings_us"] == 12.0
    assert complete[0]["args"]["exposure_probability"] == 0.75
    assert complete[0]["args"]["share_of_step_savings_pct"] == 60.0
    assert complete[0]["args"]["savings_status"] == "exposed"
    assert complete[2]["args"]["savings_status"] == "hidden"
    assert not complete[2]["name"].startswith("★ ")
    assert trace["piperSavingsAttribution"]["matched_feature_count"] == 2
    assert trace["piperSavingsAttribution"]["unmatched_feature_ids"] == []
    summary = trace["piperSavingsAttribution"]
    assert summary["net_exposed_step_savings_us"] == 6.2
    assert summary["simulator_step_savings_us"] == 6.2
    assert summary["exposed_savings_validation_residual_us"] == 0.0
    assert summary["exposed_savings_validation_passed"] is True
    assert sum(
        event["args"].get("trace_checksum_share_us", 0.0)
        for event in complete
    ) == 6.2
