from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.profiling import NodeProfile, ProfileConfig, ProfiledTrainingPlan
from src.replay import (
    ReplayValidationError,
    observed_profile_step_time_samples,
    replay_profile_modes,
    replay_training_step,
    replay_training_step_samples,
)


def _profile(duration_us: float):
    return SimpleNamespace(estimate_us=duration_us)


def _node(
    uid: str,
    duration_by_rank: dict[int, float],
    *,
    stream: str = "default_stream",
    kind: str = "COMPUTE",
    device: list[int] | None = None,
    meta: dict | None = None,
) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind=kind,
        compute_subkind="FWD" if kind == "COMPUTE" else None,
        tag={},
        device=list(device if device is not None else duration_by_rank),
        stream=stream,
        node_meta=dict(meta or {}),
        profile_times_by_mode={
            "isolated": {
                rank: _profile(duration)
                for rank, duration in duration_by_rank.items()
            }
        },
    )


def _dag(*nodes: TrainingDAGNode, edges=()) -> TrainingDAG:
    dag = TrainingDAG()
    for node in nodes:
        dag.add_node(node)
    for src, dst, kind in edges:
        dag.add_edge(TrainingDAGEdge(src, dst, kind))
    return dag


def _aligned_plan(*dags: TrainingDAG, mode: str = "isolated"):
    for dag in dags:
        for node in dag.nodes.values():
            profiles = node.profile_times_by_mode.get(mode, {})
            aligned_profiles = {}
            for rank, profile in profiles.items():
                if (
                    isinstance(profile, NodeProfile)
                    and profile.start_offset_samples_us
                ):
                    aligned = profile
                else:
                    aligned = NodeProfile.from_samples(
                        (float(profile.estimate_us),),
                        start_offset_samples_us=(0.0,),
                    )
                if (
                    node.node_kind in {"SEND_COMM", "RECV_COMM"}
                    and not aligned.p2p_post_offset_samples_us
                ):
                    aligned = replace(
                        aligned,
                        p2p_post_offset_samples_us=(0.0,) * len(
                            aligned.samples_us
                        ),
                        p2p_post_offset_estimate_us=0.0,
                    )
                aligned_profiles[rank] = aligned
            node.profile_times_by_mode[mode] = aligned_profiles
    return SimpleNamespace(
        per_pp_dags=list(dags),
        topology={},
        profile_metadata={
            "timeline_alignment": {
                "modes": {
                    mode: {
                        "samples": 1,
                        "max_pairwise_uncertainty_us": 0.0,
                    }
                }
            }
        },
    )


def test_single_stream_fifo_sums_profile_times() -> None:
    dag = _dag(
        _node("a", {0: 3.0}),
        _node("b", {0: 4.0}),
        edges=(("a", "b", "temporal"),),
    )

    result = replay_training_step([dag])

    assert result.step_time_us == pytest.approx(7.0)
    assert result.event(0, "a").start_us == pytest.approx(0.0)
    assert result.event(0, "b").start_us == pytest.approx(3.0)


def test_independent_streams_overlap_without_device_capacity_clock() -> None:
    dag = _dag(
        _node("a", {0: 8.0}, stream="compute"),
        _node("b", {0: 5.0}, stream="communication"),
    )

    result = replay_training_step([dag])

    assert result.step_time_us == pytest.approx(8.0)
    assert result.event(0, "a").start_us == pytest.approx(0.0)
    assert result.event(0, "b").start_us == pytest.approx(0.0)


def test_issue_edge_does_not_constrain_replay_timing() -> None:
    producer = _node("producer", {0: 10.0}, stream="compute")
    first = _node(
        "first",
        {0: 8.0},
        stream="ep_stream",
        kind="A2A_COMM",
        meta={"comm_id": "first", "participants": [0]},
    )
    second = _node(
        "second",
        {0: 5.0},
        stream="dp_stream",
        kind="REDUCE_COMM",
        meta={"comm_id": "second", "participants": [0]},
    )
    tail = _node("tail", {0: 20.0}, stream="tail")
    dag = _dag(
        producer,
        first,
        second,
        tail,
        edges=(
            ("producer", "first", "data"),
            ("first", "second", "issue"),
            ("second", "tail", "data"),
        ),
    )

    result = replay_training_step(_aligned_plan(dag))

    assert result.step_time_us == pytest.approx(25.0)
    assert result.event(0, "first").start_us == pytest.approx(10.0)
    assert result.event(0, "second").start_us == pytest.approx(0.0)
    assert result.event(0, "tail").start_us == pytest.approx(5.0)


def test_named_profile_mode_selects_a_second_duration_dataset() -> None:
    node = _node("a", {0: 3.0})
    node.profile_times_by_mode["normal"] = {0: _profile(7.0)}
    dag = _dag(node)

    isolated = replay_training_step([dag], profile_mode="isolated")
    normal = replay_training_step([dag], profile_mode="normal")

    assert isolated.step_time_us == pytest.approx(3.0)
    assert normal.step_time_us == pytest.approx(7.0)
    assert isolated.profile_mode == "isolated"
    assert normal.profile_mode == "normal"


def test_raw_profile_iterations_produce_paired_replay_samples() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode["isolated"] = {
        0: NodeProfile.from_samples(
            (3.0, 5.0, 4.0), sample_iteration_ids=(10, 11, 12)
        )
    }
    dag = _dag(node)

    samples = replay_training_step_samples([dag], profile_mode="isolated")

    assert samples == pytest.approx((3.0, 5.0, 4.0))


def test_replay_uses_earliest_legal_start_and_retains_observed_offsets() -> None:
    profile_mode = "normal"
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode[profile_mode] = {
        0: NodeProfile.from_samples(
            (2.0, 4.0),
            sample_iteration_ids=(10, 11),
            start_offset_samples_us=(10.0, 20.0),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }
    dag = _dag(node)

    estimate = replay_training_step([dag], profile_mode=profile_mode)
    samples = replay_training_step_samples([dag], profile_mode=profile_mode)

    assert estimate.event(0, "a").start_us == pytest.approx(0.0)
    assert estimate.event(0, "a").profile_start_offset_us == pytest.approx(0.0)
    assert estimate.event(0, "a").observed_profile_start_offset_us == pytest.approx(15.0)
    assert estimate.step_time_us == pytest.approx(3.0)
    assert samples == pytest.approx((2.0, 4.0))


def test_profile_mode_comparison_reports_uncertainty_and_prediction_error() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode = {
        "isolated": {0: NodeProfile.from_samples((3.0, 4.0, 5.0))},
        "normal": {0: NodeProfile.from_samples((7.0, 8.0, 9.0))},
    }
    comparison = replay_profile_modes(
        [_dag(node)], actual_step_samples_us=(7.5, 8.0, 8.5)
    )

    assert comparison.isolated.step_time_us == pytest.approx(4.0)
    assert comparison.normal.step_time_us == pytest.approx(8.0)
    assert comparison.normal_replay_error.signed_pct == pytest.approx(0.0)
    assert comparison.prediction_error_pct == pytest.approx(50.0)
    assert comparison.isolated_statistics.samples == 3
    assert comparison.normal_statistics.samples == 3


def test_profile_mode_comparison_uses_mean_and_paired_isolated_savings() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode = {
        "isolated": {
            0: NodeProfile.from_samples((9.0, 50.0, 990.0))
        },
        "normal": {0: NodeProfile.from_samples((10.0, 100.0, 1000.0))},
    }

    comparison = replay_profile_modes([_dag(node)])

    assert comparison.normal_statistics.estimate == pytest.approx(370.0)
    assert comparison.normal_statistics.median == pytest.approx(100.0)
    assert comparison.isolated_statistics.estimate == pytest.approx(
        1049.0 / 3.0
    )
    assert comparison.isolated_statistics.median == pytest.approx(50.0)
    assert comparison.perfect_overlap_absolute_savings_us == pytest.approx(
        61.0 / 3.0
    )
    assert (
        comparison.interference_overhead_pct
        == pytest.approx(100.0 * 61.0 / 1110.0)
    )
    assert (
        comparison.perfect_overlap_savings_statistics.percent_ci95_low
        < comparison.interference_overhead_pct
        < comparison.perfect_overlap_savings_statistics.percent_ci95_high
    )


def test_profile_mode_comparison_rejects_unaligned_iteration_ids() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode = {
        "isolated": {
            0: NodeProfile.from_samples(
                (9.0, 19.0), sample_iteration_ids=(0, 1)
            )
        },
        "normal": {
            0: NodeProfile.from_samples(
                (10.0, 20.0), sample_iteration_ids=(1, 2)
            )
        },
    }

    with pytest.raises(ReplayValidationError, match="must align"):
        replay_profile_modes([_dag(node)])


def test_observed_profile_envelope_does_not_cross_compare_rank_origins() -> None:
    node = _node("a", {0: 1.0, 1: 1.0})
    node.profile_times_by_mode = {
        "isolated": {
            0: NodeProfile.from_samples((8.0, 8.0)),
            1: NodeProfile.from_samples((8.0, 8.0)),
        },
        "normal": {
            0: NodeProfile.from_samples(
                (10.0, 10.0), start_offset_samples_us=(0.0, 20.0)
            ),
            1: NodeProfile.from_samples(
                (5.0, 5.0), start_offset_samples_us=(50.0, 5.0)
            ),
        },
    }
    plan = ProfiledTrainingPlan.from_dags([_dag(node)])

    samples = observed_profile_step_time_samples(plan, profile_mode="normal")
    comparison = replay_profile_modes(plan)

    assert samples == pytest.approx((55.0, 30.0))
    assert comparison.observed_normal_envelope_samples_us == pytest.approx(
        (55.0, 30.0)
    )


def test_profile_comparison_calibrates_distribution_estimate_not_point_timeline() -> None:
    left = _node("left", {0: 1.0}, stream="left")
    right = _node("right", {0: 1.0}, stream="right")
    left.profile_times_by_mode = {
        "isolated": {0: NodeProfile.from_samples((50.0, 50.0, 50.0))},
        "normal": {0: NodeProfile.from_samples((1.0, 100.0, 1.0))},
    }
    right.profile_times_by_mode = {
        "isolated": {0: NodeProfile.from_samples((50.0, 50.0, 50.0))},
        "normal": {0: NodeProfile.from_samples((100.0, 1.0, 1.0))},
    }

    comparison = replay_profile_modes(
        [_dag(left, right)], actual_step_samples_us=(100.0, 100.0, 100.0)
    )

    # Mean-per-node point replay is 34 us, while replaying each paired raw
    # iteration before reduction yields (100, 100, 1), whose mean is 67 us.
    # The nonlinear max must be applied before the sample mean.
    assert comparison.normal.step_time_us == pytest.approx(1.0)
    assert comparison.normal_statistics.estimate == pytest.approx(67.0)
    assert comparison.normal_replay_error.signed_pct == pytest.approx(-33.0)
    assert comparison.prediction_error_pct == pytest.approx(50.0)


def test_cross_stream_fork_join_uses_explicit_dependencies() -> None:
    dag = _dag(
        _node("fork", {0: 1.0}, stream="launch"),
        _node("left", {0: 3.0}, stream="left"),
        _node("right", {0: 4.0}, stream="right"),
        _node("join", {0: 2.0}, stream="launch"),
        edges=(
            ("fork", "left", "data"),
            ("fork", "right", "temporal"),
            ("left", "join", "data"),
            ("right", "join", "temporal"),
        ),
    )

    result = replay_training_step([dag])

    assert result.step_time_us == pytest.approx(7.0)
    assert result.event(0, "left").start_us == pytest.approx(1.0)
    assert result.event(0, "right").start_us == pytest.approx(1.0)
    assert result.event(0, "join").start_us == pytest.approx(5.0)
    assert [entry.members for entry in result.critical_path] == [
        (result.event(0, uid).operation,) for uid in ("fork", "right", "join")
    ]


def test_p2p_transactions_rendezvous_but_do_not_serialize_each_other() -> None:
    send1 = _node(
        "send.1",
        {0: 5.0},
        stream="pp_stream",
        kind="SEND_COMM",
        meta={
            "comm_id": "first",
            "peer_pp_rank": 1,
            "comm_group_kind": "pp_lo_hi",
        },
    )
    recv2 = _node(
        "recv.2",
        {0: 6.0},
        stream="pp_stream",
        kind="RECV_COMM",
        meta={
            "comm_id": "second",
            "peer_pp_rank": 1,
            "comm_group_kind": "pp_hi_lo",
        },
    )
    recv1 = _node(
        "recv.1",
        {1: 4.0},
        stream="pp_stream",
        kind="RECV_COMM",
        meta={
            "comm_id": "first",
            "peer_pp_rank": 0,
            "comm_group_kind": "pp_lo_hi",
        },
    )
    send2 = _node(
        "send.2",
        {1: 7.0},
        stream="pp_stream",
        kind="SEND_COMM",
        meta={
            "comm_id": "second",
            "peer_pp_rank": 0,
            "comm_group_kind": "pp_hi_lo",
        },
    )

    result = replay_training_step([_dag(send1, recv2), _dag(recv1, send2)])

    assert result.step_time_us == pytest.approx(7.0)
    assert result.event(0, "send.1").start_us == pytest.approx(0.0)
    assert result.event(1, "send.2").start_us == pytest.approx(0.0)
    assert result.event(1, "recv.1").end_us == pytest.approx(4.0)
    assert result.event(0, "send.1").end_us == pytest.approx(5.0)
    assert result.event(0, "recv.2").end_us == pytest.approx(6.0)
    assert result.event(1, "send.2").end_us == pytest.approx(7.0)


def test_preposted_dense_state_send_breaks_gpu_dependency_cycle() -> None:
    send = _node(
        "send",
        {0: 1.0},
        stream="dense_state_stream",
        kind="SEND_COMM",
        meta={
            "comm_id": "state-transfer",
            "peer_pp_rank": 1,
            "comm_group_kind": "dense_state_lo_hi",
            "dense_state_transfer": True,
        },
    )
    collective_0 = _node(
        "collective",
        {0: 2.0},
        stream="ep_stream",
        kind="ALL_REDUCE_COMM",
        meta={"comm_id": "routed", "participants": [0, 1]},
    )
    collective_1 = _node(
        "collective",
        {1: 2.0},
        stream="ep_stream",
        kind="ALL_REDUCE_COMM",
        meta={"comm_id": "routed", "participants": [0, 1]},
    )
    recv = _node(
        "recv",
        {1: 3.0},
        stream="dense_state_stream",
        kind="RECV_COMM",
        meta={
            "comm_id": "state-transfer",
            "peer_pp_rank": 0,
            "comm_group_kind": "dense_state_lo_hi",
            "dense_state_transfer": True,
        },
    )

    result = replay_training_step(
        _aligned_plan(
            _dag(
                send,
                collective_0,
                edges=(("send", "collective", "temporal"),),
            ),
            _dag(
                collective_1,
                recv,
                edges=(("collective", "recv", "temporal"),),
            ),
        )
    )

    assert result.event(0, "send").end_us == pytest.approx(1.0)
    assert result.event(1, "recv").start_us == pytest.approx(3.0)
    assert result.event(1, "recv").end_us == pytest.approx(6.0)


def test_p2p_profile_uses_host_post_match_and_independent_completions() -> None:
    profile_mode = "normal"
    send = _node(
        "send",
        {0: 1.0},
        kind="SEND_COMM",
        meta={"comm_id": "delayed-transfer", "peer_pp_rank": 1},
    )
    recv = _node(
        "recv",
        {1: 1.0},
        kind="RECV_COMM",
        meta={"comm_id": "delayed-transfer", "peer_pp_rank": 0},
    )
    send.profile_times_by_mode[profile_mode] = {
        0: NodeProfile.from_samples(
            (5.0,),
            start_offset_samples_us=(0.0,),
            p2p_post_offset_samples_us=(0.0,),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }
    recv.profile_times_by_mode[profile_mode] = {
        1: NodeProfile.from_samples(
            (2.0,),
            start_offset_samples_us=(10.0,),
            p2p_post_offset_samples_us=(1.0,),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }

    result = replay_training_step(
        _aligned_plan(_dag(send), _dag(recv), mode=profile_mode),
        profile_mode=profile_mode,
    )

    assert result.event(0, "send").start_us == pytest.approx(0.0)
    assert result.event(1, "recv").start_us == pytest.approx(1.0)
    assert result.event(0, "send").profile_duration_us == pytest.approx(5.0)
    assert result.event(0, "send").effective_duration_us == pytest.approx(5.0)
    assert result.event(1, "recv").effective_duration_us == pytest.approx(2.0)
    assert result.event(0, "send").end_us == pytest.approx(5.0)
    assert result.event(1, "recv").end_us == pytest.approx(3.0)
    assert result.step_time_us == pytest.approx(5.0)


def test_both_replays_use_normal_p2p_post_schedule() -> None:
    send = _node(
        "send",
        {0: 1.0},
        kind="SEND_COMM",
        meta={"comm_id": "transfer", "peer_pp_rank": 1},
    )
    recv = _node(
        "recv",
        {1: 1.0},
        kind="RECV_COMM",
        meta={"comm_id": "transfer", "peer_pp_rank": 0},
    )
    send.profile_times_by_mode = {
        "normal": {
            0: NodeProfile.from_samples(
                (4.0,),
                start_offset_samples_us=(8.0,),
                p2p_post_offset_samples_us=(5.0,),
            )
        },
        "isolated": {
            0: NodeProfile.from_samples(
                (2.0,),
                start_offset_samples_us=(120.0,),
                p2p_post_offset_samples_us=(100.0,),
            )
        },
    }
    recv.profile_times_by_mode = {
        "normal": {
            1: NodeProfile.from_samples(
                (3.0,),
                start_offset_samples_us=(10.0,),
                p2p_post_offset_samples_us=(8.0,),
            )
        },
        "isolated": {
            1: NodeProfile.from_samples(
                (2.0,),
                start_offset_samples_us=(130.0,),
                p2p_post_offset_samples_us=(120.0,),
            )
        },
    }
    plan = SimpleNamespace(
        per_pp_dags=[_dag(send), _dag(recv)],
        topology={},
        profile_metadata={
            "timeline_alignment": {
                "modes": {
                    "normal": {"max_pairwise_uncertainty_us": 0.0},
                    "isolated": {"max_pairwise_uncertainty_us": 0.0},
                }
            }
        },
    )

    normal = replay_training_step(plan, profile_mode="normal")
    isolated = replay_training_step(plan, profile_mode="isolated")

    assert normal.event(0, "send").start_us == pytest.approx(5.0)
    assert isolated.event(0, "send").start_us == pytest.approx(5.0)
    assert normal.event(1, "recv").start_us == pytest.approx(8.0)
    assert isolated.event(1, "recv").start_us == pytest.approx(8.0)
    assert normal.event(0, "send").end_us == pytest.approx(9.0)
    assert isolated.event(0, "send").end_us == pytest.approx(7.0)
    assert normal.event(1, "recv").end_us == pytest.approx(11.0)
    assert isolated.event(1, "recv").end_us == pytest.approx(10.0)


def test_p2p_receive_completion_waits_for_send_source_readiness() -> None:
    producer = _node("producer", {0: 10.0})
    send = _node(
        "send",
        {0: 1.0},
        kind="SEND_COMM",
        meta={"comm_id": "transfer", "peer_pp_rank": 1},
    )
    recv = _node(
        "recv",
        {1: 1.0},
        kind="RECV_COMM",
        meta={"comm_id": "transfer", "peer_pp_rank": 0},
    )
    producer.profile_times_by_mode["normal"] = {
        0: NodeProfile.from_samples(
            (10.0,), start_offset_samples_us=(0.0,)
        )
    }
    send.profile_times_by_mode["normal"] = {
        0: NodeProfile.from_samples(
            (2.0,),
            start_offset_samples_us=(10.0,),
            p2p_post_offset_samples_us=(0.0,),
        )
    }
    recv.profile_times_by_mode["normal"] = {
        1: NodeProfile.from_samples(
            (12.0,),
            start_offset_samples_us=(0.0,),
            p2p_post_offset_samples_us=(0.0,),
        )
    }

    result = replay_training_step(
        _aligned_plan(
            _dag(producer, send, edges=(("producer", "send", "data"),)),
            _dag(recv),
            mode="normal",
        ),
        profile_mode="normal",
    )

    assert result.event(0, "send").start_us == pytest.approx(10.0)
    assert result.event(1, "recv").start_us == pytest.approx(10.0)
    assert result.event(0, "send").end_us == pytest.approx(12.0)
    assert result.event(1, "recv").end_us == pytest.approx(12.0)


def test_p2p_send_may_complete_before_receive_is_posted() -> None:
    send = _node(
        "send",
        {0: 1.0},
        kind="SEND_COMM",
        meta={"comm_id": "inconsistent-transfer", "peer_pp_rank": 1},
    )
    recv = _node(
        "recv",
        {1: 1.0},
        kind="RECV_COMM",
        meta={"comm_id": "inconsistent-transfer", "peer_pp_rank": 0},
    )
    send.profile_times_by_mode["normal"] = {
        0: NodeProfile.from_samples(
            (5.0,),
            start_offset_samples_us=(0.0,),
            p2p_post_offset_samples_us=(0.0,),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }
    recv.profile_times_by_mode["normal"] = {
        1: NodeProfile.from_samples(
            (2.0,),
            start_offset_samples_us=(10.0,),
            p2p_post_offset_samples_us=(10.0,),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }

    result = replay_training_step(
        _aligned_plan(_dag(send), _dag(recv), mode="normal"),
        profile_mode="normal",
    )

    assert result.event(0, "send").end_us == pytest.approx(5.0)
    assert result.event(1, "recv").start_us == pytest.approx(10.0)
    assert result.event(1, "recv").end_us == pytest.approx(12.0)


def test_p2p_rejects_receive_ending_before_send_source_is_ready() -> None:
    send = _node(
        "send",
        {0: 1.0},
        kind="SEND_COMM",
        meta={"comm_id": "inconsistent-transfer", "peer_pp_rank": 1},
    )
    recv = _node(
        "recv",
        {1: 1.0},
        kind="RECV_COMM",
        meta={"comm_id": "inconsistent-transfer", "peer_pp_rank": 0},
    )
    send.profile_times_by_mode["normal"] = {
        0: NodeProfile.from_samples(
            (1.0,),
            start_offset_samples_us=(10.0,),
            p2p_post_offset_samples_us=(0.0,),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }
    recv.profile_times_by_mode["normal"] = {
        1: NodeProfile.from_samples(
            (2.0,),
            start_offset_samples_us=(0.0,),
            p2p_post_offset_samples_us=(0.0,),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }

    with pytest.raises(
        ReplayValidationError, match="inconsistent aligned endpoint"
    ):
        replay_training_step(
            _aligned_plan(_dag(send), _dag(recv), mode="normal"),
            profile_mode="normal",
        )


def test_preposted_opposite_p2p_transfers_do_not_form_gpu_cycle() -> None:
    send_b = _node(
        "send.b",
        {0: 7.0},
        stream="pp_stream",
        kind="SEND_COMM",
        meta={
            "comm_id": "b",
            "peer_pp_rank": 1,
            "comm_group_kind": "pp_lo_hi",
        },
    )
    recv_a = _node(
        "recv.a",
        {0: 4.0},
        stream="pp_stream",
        kind="RECV_COMM",
        meta={
            "comm_id": "a",
            "peer_pp_rank": 1,
            "comm_group_kind": "pp_hi_lo",
        },
    )
    send_a = _node(
        "send.a",
        {1: 5.0},
        stream="pp_stream",
        kind="SEND_COMM",
        meta={
            "comm_id": "a",
            "peer_pp_rank": 0,
            "comm_group_kind": "pp_hi_lo",
        },
    )
    recv_b = _node(
        "recv.b",
        {1: 6.0},
        stream="pp_stream",
        kind="RECV_COMM",
        meta={
            "comm_id": "b",
            "peer_pp_rank": 0,
            "comm_group_kind": "pp_lo_hi",
        },
    )

    result = replay_training_step(
        [
            _dag(send_b, recv_a, edges=(("send.b", "recv.a", "temporal"),)),
            _dag(send_a, recv_b, edges=(("send.a", "recv.b", "temporal"),)),
        ]
    )

    assert result.event(0, "send.b").end_us == pytest.approx(7.0)
    assert result.event(1, "send.a").end_us == pytest.approx(5.0)
    assert result.event(0, "recv.a").end_us == pytest.approx(11.0)
    assert result.event(1, "recv.b").end_us == pytest.approx(11.0)


def test_same_direction_p2p_transactions_preserve_directional_fifo() -> None:
    send1 = _node(
        "send.1",
        {0: 5.0},
        kind="SEND_COMM",
        meta={
            "comm_id": "first",
            "peer_pp_rank": 1,
            "comm_group_kind": "pp_lo_hi",
        },
    )
    send2 = _node(
        "send.2",
        {0: 7.0},
        kind="SEND_COMM",
        meta={
            "comm_id": "second",
            "peer_pp_rank": 1,
            "comm_group_kind": "pp_lo_hi",
        },
    )
    recv1 = _node(
        "recv.1",
        {1: 4.0},
        kind="RECV_COMM",
        meta={
            "comm_id": "first",
            "peer_pp_rank": 0,
            "comm_group_kind": "pp_lo_hi",
        },
    )
    recv2 = _node(
        "recv.2",
        {1: 6.0},
        kind="RECV_COMM",
        meta={
            "comm_id": "second",
            "peer_pp_rank": 0,
            "comm_group_kind": "pp_lo_hi",
        },
    )

    result = replay_training_step(
        [
            _dag(send1, send2, edges=(("send.1", "send.2", "temporal"),)),
            _dag(recv1, recv2, edges=(("recv.1", "recv.2", "temporal"),)),
        ]
    )

    assert result.event(0, "send.1").start_us == pytest.approx(0.0)
    assert result.event(0, "send.2").start_us == pytest.approx(5.0)
    assert result.event(1, "recv.2").start_us == pytest.approx(5.0)
    assert result.event(0, "send.2").end_us == pytest.approx(12.0)
    assert result.event(1, "recv.2").end_us == pytest.approx(11.0)
    assert result.step_time_us == pytest.approx(12.0)


def test_p2p_group_participants_are_validated_across_actor_group_local_ranks() -> None:
    participants = [0, 1, 2, 3]
    send = _node(
        "send",
        {0: 5.0, 2: 6.0},
        kind="SEND_COMM",
        device=[0, 2],
        meta={
            "comm_id": "pipeline-transfer",
            "peer_pp_rank": 1,
            "comm_group_kind": "pp_lo_hi",
            "participants": participants,
        },
    )
    recv = _node(
        "recv",
        {1: 4.0, 3: 7.0},
        kind="RECV_COMM",
        device=[1, 3],
        meta={
            "comm_id": "pipeline-transfer",
            "peer_pp_rank": 0,
            "comm_group_kind": "pp_lo_hi",
            "participants": participants,
        },
    )

    result = replay_training_step([_dag(send), _dag(recv)])

    assert result.event(0, "send").duration_us == pytest.approx(5.0)
    assert result.event(1, "recv").duration_us == pytest.approx(4.0)
    assert result.event(2, "send").duration_us == pytest.approx(6.0)
    assert result.event(3, "recv").duration_us == pytest.approx(7.0)
    assert result.step_time_us == pytest.approx(7.0)


def test_p2p_rejects_participants_outside_endpoint_device_groups() -> None:
    send = _node(
        "send",
        {0: 2.0},
        kind="SEND_COMM",
        device=[0],
        meta={
            "comm_id": "bad-participants",
            "peer_pp_rank": 1,
            "participants": [0, 1, 9],
        },
    )
    recv = _node(
        "recv",
        {1: 2.0},
        kind="RECV_COMM",
        device=[1],
        meta={
            "comm_id": "bad-participants",
            "peer_pp_rank": 0,
            "participants": [0, 1, 9],
        },
    )

    with pytest.raises(ReplayValidationError, match="endpoint device-group ranks"):
        replay_training_step([_dag(send), _dag(recv)])


def test_unaligned_collective_profile_is_rejected() -> None:
    producer = _node(
        "backward.0",
        {0: 8.0, 1: 2.0},
        stream="compute",
        device=[0, 1],
    )
    collective = _node(
        "reduce.0",
        {0: 3.0, 1: 10.0},
        stream="dp_stream",
        kind="REDUCE_COMM",
        device=[0, 1],
        meta={"comm_id": "grad.bucket", "participants": [0, 1]},
    )

    with pytest.raises(
        ReplayValidationError, match="no cross-actor timeline alignment"
    ):
        replay_training_step(
            [
                _dag(
                    producer,
                    collective,
                    edges=(("backward.0", "reduce.0", "data"),),
                )
            ]
        )


def test_aligned_collective_reuses_observed_post_rendezvous_tails() -> None:
    producer = _node(
        "producer",
        {0: 8.0, 1: 2.0},
        stream="compute",
        device=[0, 1],
    )
    producer.profile_times_by_mode["isolated"] = {
        0: NodeProfile.from_samples(
            (8.0,), start_offset_samples_us=(0.0,)
        ),
        1: NodeProfile.from_samples(
            (2.0,), start_offset_samples_us=(0.0,)
        ),
    }
    collective = _node(
        "reduce",
        {0: 1.0, 1: 1.0},
        stream="dp_stream",
        kind="REDUCE_COMM",
        device=[0, 1],
        meta={"comm_id": "aligned.bucket", "participants": [0, 1]},
    )
    collective.profile_times_by_mode["isolated"] = {
        # Observed rendezvous is at 8 us: rank 0 completes 2 us later and
        # rank 1 completes 3 us later, at observed time 11 us.
        0: NodeProfile.from_samples(
            (10.0,), start_offset_samples_us=(0.0,)
        ),
        1: NodeProfile.from_samples(
            (3.0,), start_offset_samples_us=(8.0,)
        ),
    }
    plan = SimpleNamespace(
        per_pp_dags=[
            _dag(
                producer,
                collective,
                edges=(("producer", "reduce", "data"),),
            )
        ],
        topology={
            "pp_degree": 1,
            "actor_groups": [{"id": 0, "devices": [0, 1]}],
        },
        profile_metadata={
            "timeline_alignment": {
                "modes": {"isolated": {"samples": 1}}
            }
        },
    )

    result = replay_training_step(plan, profile_mode="isolated")

    assert result.event(0, "reduce").start_us == pytest.approx(8.0)
    assert result.event(1, "reduce").start_us == pytest.approx(8.0)
    assert result.event(0, "reduce").effective_duration_us == pytest.approx(2.0)
    assert result.event(1, "reduce").effective_duration_us == pytest.approx(3.0)
    assert result.step_time_us == pytest.approx(11.0)


def test_aligned_all_zero_collective_has_zero_service() -> None:
    collective = _node(
        "reduce",
        {0: 0.0, 1: 0.0},
        stream="dp_stream",
        kind="REDUCE_COMM",
        device=[0, 1],
        meta={"comm_id": "zero.bucket", "participants": [0, 1]},
    )
    plan = _aligned_plan(_dag(collective))
    collective.profile_times_by_mode["isolated"] = {
        0: NodeProfile.from_samples(
            (0.0,), start_offset_samples_us=(2.0,)
        ),
        1: NodeProfile.from_samples(
            (0.0,), start_offset_samples_us=(20.0,)
        ),
    }

    result = replay_training_step(plan)

    assert result.step_time_us == pytest.approx(0.0)
    assert result.event(0, "reduce").effective_duration_us == pytest.approx(0.0)
    assert result.event(1, "reduce").effective_duration_us == pytest.approx(0.0)


def test_aligned_collective_rejects_mixed_zero_activity_endpoints() -> None:
    collective = _node(
        "reduce",
        {0: 1.0, 1: 0.0},
        stream="dp_stream",
        kind="REDUCE_COMM",
        device=[0, 1],
        meta={"comm_id": "mixed.bucket", "participants": [0, 1]},
    )
    plan = _aligned_plan(_dag(collective))

    with pytest.raises(
        ReplayValidationError,
        match="mixes zero-activity and nonzero-activity",
    ):
        replay_training_step(plan)


def test_aligned_collective_clamps_only_within_declared_uncertainty() -> None:
    collective = _node(
        "reduce",
        {0: 9.5, 1: 1.0},
        stream="dp_stream",
        kind="ALL_REDUCE_COMM",
        device=[0, 1],
        meta={"comm_id": "uncertain.bucket", "participants": [0, 1]},
    )
    plan = _aligned_plan(_dag(collective))
    plan.profile_metadata["timeline_alignment"]["modes"][
        "isolated"
    ]["max_pairwise_uncertainty_us"] = 1.0
    collective.profile_times_by_mode["isolated"] = {
        0: NodeProfile.from_samples(
            (9.5,), start_offset_samples_us=(0.0,)
        ),
        1: NodeProfile.from_samples(
            (1.0,), start_offset_samples_us=(10.0,)
        ),
    }

    result = replay_training_step(plan)
    assert result.event(0, "reduce").effective_duration_us == pytest.approx(0.0)
    assert result.event(1, "reduce").effective_duration_us == pytest.approx(1.0)

    collective.profile_times_by_mode["isolated"][0] = (
        NodeProfile.from_samples(
            (8.0,), start_offset_samples_us=(0.0,)
        )
    )
    with pytest.raises(
        ReplayValidationError, match="inconsistent aligned endpoint"
    ):
        replay_training_step(plan)


def test_collective_estimate_is_derived_from_paired_service_samples() -> None:
    collective = _node(
        "reduce",
        {0: 0.1, 1: 0.1},
        stream="dp_stream",
        kind="REDUCE_SCATTER_COMM",
        device=[0, 1],
        meta={"comm_id": "paired.bucket", "participants": [0, 1]},
    )
    plan = _aligned_plan(_dag(collective))
    plan.profile_metadata["timeline_alignment"]["modes"][
        "isolated"
    ]["samples"] = 3
    collective.profile_times_by_mode["isolated"] = {
        # Every paired occurrence completes 0.1 us after rendezvous, but the
        # independent medians are start=0 and duration=0.1, ending before the
        # other endpoint's median start of 0.9 us.
        0: NodeProfile.from_samples(
            (0.1, 1.0, 0.1),
            start_offset_samples_us=(0.0, 0.0, 1.0),
        ),
        1: NodeProfile.from_samples(
            (0.1, 0.1, 0.1),
            start_offset_samples_us=(0.0, 0.9, 1.0),
        ),
    }

    result = replay_training_step(plan)

    assert result.event(0, "reduce").effective_duration_us == pytest.approx(0.1)
    assert result.event(1, "reduce").effective_duration_us == pytest.approx(0.1)


def test_ragged_a2a_uses_one_all_participant_rendezvous() -> None:
    producer = _node(
        "producer",
        {0: 10.0, 1: 0.0},
        stream="compute",
        device=[0, 1],
    )
    a2a = _node(
        "a2a",
        {0: 2.0, 1: 3.0},
        stream="ep_stream",
        kind="A2A_COMM",
        device=[0, 1],
        meta={"comm_id": "ragged", "participants": [0, 1]},
    )

    result = replay_training_step(
        _aligned_plan(
            _dag(producer, a2a, edges=(("producer", "a2a", "data"),))
        )
    )

    assert result.event(0, "a2a").start_us == pytest.approx(10.0)
    assert result.event(1, "a2a").start_us == pytest.approx(10.0)
    assert result.event(0, "a2a").end_us == pytest.approx(12.0)
    assert result.event(1, "a2a").end_us == pytest.approx(13.0)
    assert result.step_time_us == pytest.approx(13.0)


def test_distinct_routed_comm_ids_do_not_share_a_rendezvous() -> None:
    producer = _node(
        "producer",
        {0: 10.0, 1: 0.0},
        stream="compute",
        device=[0, 1],
    )
    delayed = _node(
        "delayed",
        {0: 2.0, 1: 2.0},
        stream="ep.delayed",
        kind="A2A_COMM",
        device=[0, 1],
        meta={"comm_id": "delayed", "participants": [0, 1]},
    )
    independent = _node(
        "independent",
        {0: 3.0, 1: 3.0},
        stream="ep.independent",
        kind="A2A_COMM",
        device=[0, 1],
        meta={"comm_id": "independent", "participants": [0, 1]},
    )

    result = replay_training_step(
        _aligned_plan(
            _dag(
                producer,
                delayed,
                independent,
                edges=(("producer", "delayed", "data"),),
            )
        )
    )

    assert result.event(0, "delayed").start_us == pytest.approx(10.0)
    assert result.event(1, "delayed").start_us == pytest.approx(10.0)
    assert result.event(0, "independent").start_us == pytest.approx(0.0)
    assert result.event(1, "independent").start_us == pytest.approx(0.0)


def test_cross_group_routed_collective_rendezvous_supports_unequal_groups() -> None:
    participants = [0, 1, 2, 3, 4, 5]
    source = _node(
        "m_to_n.source",
        {0: 2.0, 2: 3.0, 4: 4.0, 5: 5.0},
        stream="ep_stream",
        kind="M_TO_N_COMM",
        device=[0, 2, 4, 5],
        meta={
            "comm_id": "dispatch.mb0",
            "participants": participants,
            "role": "source",
            "source_devices": [0, 2, 4, 5],
            "destination_devices": [1, 3],
        },
    )
    destination = _node(
        "m_to_n.destination",
        {1: 6.0, 3: 7.0},
        stream="ep_stream",
        kind="M_TO_N_COMM",
        device=[1, 3],
        meta={
            "comm_id": "dispatch.mb0",
            "participants": participants,
            "role": "destination",
            "source_devices": [0, 2, 4, 5],
            "destination_devices": [1, 3],
        },
    )

    result = replay_training_step(
        _aligned_plan(_dag(source), _dag(destination))
    )

    assert result.step_time_us == pytest.approx(7.0)
    assert {
        result.event(rank, "m_to_n.source").communication_id
        for rank in (0, 2, 4, 5)
    } == {"M_TO_N_COMM:dispatch.mb0"}
    assert {
        result.event(rank, "m_to_n.destination").communication_id
        for rank in (1, 3)
    } == {"M_TO_N_COMM:dispatch.mb0"}


def test_cross_group_routed_collective_requires_matching_comm_id() -> None:
    participants = [0, 1, 2]
    source = _node(
        "source",
        {0: 2.0, 2: 2.0},
        kind="N_TO_M_COMM",
        device=[0, 2],
        meta={"comm_id": "combine.source", "participants": participants},
    )
    destination = _node(
        "destination",
        {1: 2.0},
        kind="N_TO_M_COMM",
        device=[1],
        meta={"comm_id": "combine.destination", "participants": participants},
    )

    with pytest.raises(ReplayValidationError, match="has ranks.*expected"):
        replay_training_step(
            _aligned_plan(_dag(source), _dag(destination))
        )


def test_routed_endpoints_must_declare_the_same_participants() -> None:
    first = _node(
        "first",
        {0: 1.0},
        kind="M_TO_N_COMM",
        device=[0],
        meta={"comm_id": "bad.participants", "participants": [0, 1]},
    )
    second = _node(
        "second",
        {1: 1.0},
        kind="M_TO_N_COMM",
        device=[1],
        meta={"comm_id": "bad.participants", "participants": [0, 1, 2]},
    )

    with pytest.raises(
        ReplayValidationError, match="endpoints disagree on participants"
    ):
        replay_training_step(_aligned_plan(_dag(first), _dag(second)))


def test_inconsistent_routed_transaction_order_is_rejected() -> None:
    participants = [0, 1]
    dispatch_source = _node(
        "dispatch.source",
        {0: 3.0},
        stream="ep_stream",
        kind="M_TO_N_COMM",
        device=[0],
        meta={
            "comm_id": "dispatch", "participants": participants,
            "role": "source", "source_devices": [0],
            "destination_devices": [1],
        },
    )
    combine_destination = _node(
        "combine.destination",
        {0: 4.0},
        stream="ep_stream",
        kind="N_TO_M_COMM",
        device=[0],
        meta={
            "comm_id": "combine", "participants": participants,
            "role": "destination", "source_devices": [1],
            "destination_devices": [0],
        },
    )
    combine_source = _node(
        "combine.source",
        {1: 5.0},
        stream="ep_stream",
        kind="N_TO_M_COMM",
        device=[1],
        meta={
            "comm_id": "combine", "participants": participants,
            "role": "source", "source_devices": [1],
            "destination_devices": [0],
        },
    )
    dispatch_destination = _node(
        "dispatch.destination",
        {1: 6.0},
        stream="ep_stream",
        kind="M_TO_N_COMM",
        device=[1],
        meta={
            "comm_id": "dispatch", "participants": participants,
            "role": "destination", "source_devices": [0],
            "destination_devices": [1],
        },
    )

    with pytest.raises(
        ReplayValidationError, match="communication replay graph contains a cycle"
    ):
        replay_training_step(
            _aligned_plan(
                _dag(
                    dispatch_source,
                    combine_destination,
                    edges=(("dispatch.source", "combine.destination", "temporal"),),
                ),
                _dag(
                    combine_source,
                    dispatch_destination,
                    edges=(("combine.source", "dispatch.destination", "temporal"),),
                ),
            )
        )


@pytest.mark.parametrize("kind", ("M_TO_N_COMM", "N_TO_M_COMM"))
def test_routed_collective_all_participants_share_one_rendezvous(
    kind: str,
) -> None:
    participants = [0, 1, 2]
    common = {
        "comm_id": f"directional.{kind}",
        "participants": participants,
        "source_devices": [0],
        "destination_devices": [1, 2],
    }
    source_producer = _node("source.producer", {0: 5.0})
    source = _node(
        "source", {0: 2.0}, kind=kind, device=[0],
        meta={**common, "role": "source"},
    )
    first_producer = _node("first.producer", {1: 2.0})
    first_destination = _node(
        "first.destination", {1: 3.0}, kind=kind, device=[1],
        meta={**common, "role": "destination"},
    )
    second_producer = _node("second.producer", {2: 10.0})
    second_destination = _node(
        "second.destination", {2: 4.0}, kind=kind, device=[2],
        meta={**common, "role": "destination"},
    )
    plan = _aligned_plan(
        _dag(
            source_producer, source,
            edges=(("source.producer", "source", "data"),),
        ),
        _dag(
            first_producer, first_destination,
            edges=(("first.producer", "first.destination", "data"),),
        ),
        _dag(
            second_producer, second_destination,
            edges=(("second.producer", "second.destination", "data"),),
        ),
    )
    source.profile_times_by_mode["isolated"][0] = (
        NodeProfile.from_samples((23.0,), start_offset_samples_us=(0.0,))
    )
    first_destination.profile_times_by_mode["isolated"][1] = (
        NodeProfile.from_samples((14.0,), start_offset_samples_us=(10.0,))
    )
    second_destination.profile_times_by_mode["isolated"][2] = (
        NodeProfile.from_samples((5.0,), start_offset_samples_us=(20.0,))
    )

    result = replay_training_step(plan)

    for rank, uid in (
        (0, "source"),
        (1, "first.destination"),
        (2, "second.destination"),
    ):
        assert result.event(rank, uid).start_us == pytest.approx(10.0)
    assert result.event(0, "source").end_us == pytest.approx(13.0)
    assert result.event(1, "first.destination").end_us == pytest.approx(14.0)
    assert result.event(2, "second.destination").end_us == pytest.approx(15.0)


def test_routed_replay_requires_participants_but_not_directional_roles() -> None:
    participants = [0, 1]
    first = _node(
        "first", {0: 1.0}, kind="M_TO_N_COMM", device=[0],
        meta={"comm_id": "bad.roles", "participants": participants},
    )
    second = _node(
        "second", {1: 1.0}, kind="M_TO_N_COMM", device=[1],
        meta={"comm_id": "bad.roles", "participants": participants},
    )

    result = replay_training_step(_aligned_plan(_dag(first), _dag(second)))
    assert result.event(0, "first").start_us == pytest.approx(0.0)
    assert result.event(1, "second").start_us == pytest.approx(0.0)

    first.node_meta.pop("participants")
    second.node_meta.pop("participants")
    with pytest.raises(
        ReplayValidationError, match="has no declared participants"
    ):
        replay_training_step(_aligned_plan(_dag(first), _dag(second)))


def test_routed_collective_accepts_zero_activity_endpoint() -> None:
    routed = _node(
        "routed",
        {0: 4.0, 1: 0.0},
        stream="ep_stream",
        kind="A2A_COMM",
        device=[0, 1],
        meta={"comm_id": "zero.routed", "participants": [0, 1]},
    )
    plan = _aligned_plan(_dag(routed))
    routed.profile_times_by_mode["isolated"] = {
        0: NodeProfile.from_samples(
            (9.0,), start_offset_samples_us=(0.0,)
        ),
        1: NodeProfile.from_samples(
            (0.0,), start_offset_samples_us=(5.0,)
        ),
    }

    result = replay_training_step(plan)

    assert result.event(0, "routed").effective_duration_us == pytest.approx(4.0)
    assert result.event(1, "routed").effective_duration_us == pytest.approx(0.0)
    assert result.event(0, "routed").end_us == pytest.approx(4.0)
    assert result.event(1, "routed").end_us == pytest.approx(0.0)


def test_missing_rank_profile_is_rejected() -> None:
    node = _node("a", {0: 2.0}, device=[0, 1])
    dag = _dag(node)

    with pytest.raises(ReplayValidationError, match="missing profile.*rank 1"):
        replay_training_step([dag])


def test_schedule_topology_rejects_device_sets_that_disagree_with_actor_groups() -> None:
    pp0 = _dag(_node("pp0", {1: 1.0, 3: 1.0}, device=[1, 3]))
    pp1 = _dag(_node("pp1", {0: 1.0, 2: 1.0}, device=[0, 2]))
    plan = SimpleNamespace(
        per_pp_dags=[pp0, pp1],
        topology={
            "pp_degree": 2,
            "actor_groups": [
                {"id": 0, "devices": [0, 2]},
                {"id": 1, "devices": [1, 3]},
            ],
        },
    )

    with pytest.raises(ReplayValidationError, match="explicit ActorGroups"):
        replay_training_step(plan)


def test_missing_p2p_peer_is_rejected() -> None:
    send = _node(
        "send.0",
        {0: 2.0},
        kind="SEND_COMM",
        meta={"comm_id": "orphan", "peer_pp_rank": 1},
    )

    with pytest.raises(ReplayValidationError, match="exactly one SEND and one RECV"):
        replay_training_step([_dag(send)])


def test_rank_local_prologue_gates_every_stream() -> None:
    dag = _dag(
        _node("a", {0: 6.0}, stream="left"),
        _node("b", {0: 2.0}, stream="right"),
    )
    dag.prologue_profile_times_by_mode = {
        "isolated": {0: _profile(4.0)}
    }

    result = replay_training_step([dag])

    assert result.step_time_us == pytest.approx(10.0)
    assert result.event(0, "__step_prologue__").end_us == pytest.approx(4.0)
    assert result.event(0, "a").start_us == pytest.approx(4.0)
    assert result.event(0, "b").start_us == pytest.approx(4.0)


def test_profiled_plan_json_round_trip_is_direct_replay_input(tmp_path) -> None:
    node = TrainingDAGNode(
        uid="forward",
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={"PP": 0},
        device=[0],
        stream="compute",
        profile_times_by_mode={
            "isolated": {
                0: NodeProfile.from_samples((9.0, 11.0))
            }
        },
    )
    dag = _dag(node)
    dag.prologue_profile_times_by_mode = {
        "isolated": {0: NodeProfile.from_samples((2.0,))}
    }
    plan = ProfiledTrainingPlan.from_dags(
        [dag],
        profile_configs={
            "isolated": ProfileConfig(warmup_iters=0, repeat_iters=2)
        },
        topology={
            "pp_degree": 1,
            "actor_groups": [{"id": 0, "devices": [0]}],
        },
    )
    artifact = tmp_path / "profile.json"
    plan.save_json(artifact)

    loaded = ProfiledTrainingPlan.load_json(artifact)
    result = replay_training_step(loaded)

    assert result.step_time_us == pytest.approx(12.0)
    assert result.event(0, "forward").profile_duration_us == pytest.approx(10.0)
