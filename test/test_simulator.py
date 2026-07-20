from types import SimpleNamespace

import pytest

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.profiling import NodeProfile, ProfileConfig, ProfiledTrainingPlan
from src.simulator import (
    SimulationValidationError,
    simulate_profile_modes,
    simulate_training_step,
    simulate_training_step_samples,
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
            "interference_free": {
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


def test_single_stream_fifo_sums_profile_times() -> None:
    dag = _dag(_node("a", {0: 3.0}), _node("b", {0: 4.0}))

    result = simulate_training_step([dag])

    assert result.step_time_us == pytest.approx(7.0)
    assert result.event(0, "a").start_us == pytest.approx(0.0)
    assert result.event(0, "b").start_us == pytest.approx(3.0)


def test_independent_streams_overlap_without_device_capacity_clock() -> None:
    dag = _dag(
        _node("a", {0: 8.0}, stream="compute"),
        _node("b", {0: 5.0}, stream="communication"),
    )

    result = simulate_training_step([dag])

    assert result.step_time_us == pytest.approx(8.0)
    assert result.event(0, "a").start_us == pytest.approx(0.0)
    assert result.event(0, "b").start_us == pytest.approx(0.0)


def test_named_profile_mode_selects_a_second_duration_dataset() -> None:
    node = _node("a", {0: 3.0})
    node.profile_times_by_mode["normal"] = {0: _profile(7.0)}
    dag = _dag(node)

    interference_free = simulate_training_step([dag], profile_mode="interference_free")
    normal = simulate_training_step([dag], profile_mode="normal")

    assert interference_free.step_time_us == pytest.approx(3.0)
    assert normal.step_time_us == pytest.approx(7.0)
    assert interference_free.profile_mode == "interference_free"
    assert normal.profile_mode == "normal"


def test_raw_profile_iterations_produce_paired_simulation_samples() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode["interference_free"] = {
        0: NodeProfile.from_samples(
            (3.0, 5.0, 4.0), sample_iteration_ids=(10, 11, 12)
        )
    }
    dag = _dag(node)

    samples = simulate_training_step_samples([dag], profile_mode="interference_free")

    assert samples == pytest.approx((3.0, 5.0, 4.0))


def test_normal_replay_preserves_iteration_aligned_start_offsets() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode["normal"] = {
        0: NodeProfile.from_samples(
            (2.0, 4.0),
            sample_iteration_ids=(10, 11),
            start_offset_samples_us=(10.0, 20.0),
            isolation_scope="logical_streams_with_directional_p2p",
        )
    }
    dag = _dag(node)

    estimate = simulate_training_step([dag], profile_mode="normal")
    samples = simulate_training_step_samples([dag], profile_mode="normal")

    assert estimate.event(0, "a").start_us == pytest.approx(15.0)
    assert estimate.event(0, "a").profile_start_offset_us == pytest.approx(15.0)
    assert estimate.step_time_us == pytest.approx(18.0)
    assert samples == pytest.approx((12.0, 24.0))


def test_interference_free_replay_does_not_apply_start_offsets() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode["interference_free"] = {
        0: NodeProfile.from_samples(
            (2.0, 4.0),
            start_offset_samples_us=(10.0, 20.0),
        )
    }

    result = simulate_training_step([_dag(node)], profile_mode="interference_free")

    assert result.event(0, "a").start_us == pytest.approx(0.0)
    assert result.step_time_us == pytest.approx(3.0)


def test_profile_mode_comparison_reports_uncertainty_and_prediction_error() -> None:
    node = _node("a", {0: 1.0})
    node.profile_times_by_mode = {
        "interference_free": {0: NodeProfile.from_samples((3.0, 4.0, 5.0))},
        "normal": {0: NodeProfile.from_samples((7.0, 8.0, 9.0))},
    }
    comparison = simulate_profile_modes(
        [_dag(node)], actual_step_samples_us=(7.5, 8.0, 8.5)
    )

    assert comparison.interference_free.step_time_us == pytest.approx(4.0)
    assert comparison.normal.step_time_us == pytest.approx(8.0)
    assert comparison.normal_prediction_error.signed_pct == pytest.approx(0.0)
    assert comparison.predicted_interference_reduction_pct == pytest.approx(50.0)
    assert comparison.interference_free_statistics.samples == 3
    assert comparison.normal_statistics.samples == 3


def test_profile_comparison_calibrates_distribution_estimate_not_point_timeline() -> None:
    left = _node("left", {0: 1.0}, stream="left")
    right = _node("right", {0: 1.0}, stream="right")
    left.profile_times_by_mode = {
        "interference_free": {0: NodeProfile.from_samples((50.0, 50.0, 50.0))},
        "normal": {0: NodeProfile.from_samples((1.0, 100.0, 1.0))},
    }
    right.profile_times_by_mode = {
        "interference_free": {0: NodeProfile.from_samples((50.0, 50.0, 50.0))},
        "normal": {0: NodeProfile.from_samples((100.0, 1.0, 1.0))},
    }

    comparison = simulate_profile_modes(
        [_dag(left, right)], actual_step_samples_us=(100.0, 100.0, 100.0)
    )

    # Median-per-node point replay is 1 us, while paired raw iterations have a
    # 100 us median after the nonlinear max across streams.  Calibration and
    # reduction must use the latter distribution estimate.
    assert comparison.normal.step_time_us == pytest.approx(1.0)
    assert comparison.normal_statistics.estimate == pytest.approx(100.0)
    assert comparison.normal_prediction_error.absolute_pct == pytest.approx(0.0)
    assert comparison.predicted_interference_reduction_pct == pytest.approx(50.0)


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

    result = simulate_training_step([dag])

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

    result = simulate_training_step([_dag(send1, recv2), _dag(recv1, send2)])

    assert result.step_time_us == pytest.approx(6.0)
    assert result.event(0, "send.1").start_us == pytest.approx(0.0)
    assert result.event(1, "send.2").start_us == pytest.approx(0.0)
    assert result.event(1, "recv.1").end_us == pytest.approx(4.0)
    assert result.event(0, "recv.2").end_us == pytest.approx(6.0)


def test_opposite_p2p_temporal_order_does_not_create_rendezvous_cycle() -> None:
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

    result = simulate_training_step(
        [
            _dag(send_b, recv_a, edges=(("send.b", "recv.a", "temporal"),)),
            _dag(send_a, recv_b, edges=(("send.a", "recv.b", "temporal"),)),
        ]
    )

    assert result.event(0, "send.b").start_us == pytest.approx(0.0)
    assert result.event(1, "send.a").start_us == pytest.approx(0.0)
    assert result.step_time_us == pytest.approx(6.0)


def test_p2p_transactions_do_not_gain_synthetic_directional_fifo() -> None:
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

    result = simulate_training_step([_dag(send1, send2), _dag(recv1, recv2)])

    assert result.event(0, "send.1").start_us == pytest.approx(0.0)
    assert result.event(0, "send.2").start_us == pytest.approx(0.0)
    assert result.step_time_us == pytest.approx(6.0)


def test_p2p_group_participants_are_validated_across_dp_lanes() -> None:
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

    result = simulate_training_step([_dag(send), _dag(recv)])

    assert result.event(0, "send").duration_us == pytest.approx(4.0)
    assert result.event(1, "recv").duration_us == pytest.approx(4.0)
    assert result.event(2, "send").duration_us == pytest.approx(6.0)
    assert result.event(3, "recv").duration_us == pytest.approx(6.0)
    assert result.step_time_us == pytest.approx(6.0)


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

    with pytest.raises(SimulationValidationError, match="endpoint device-group ranks"):
        simulate_training_step([_dag(send), _dag(recv)])


def test_collective_uses_latest_arrival_and_minimum_service_proxy() -> None:
    producer = _node(
        "backward.0",
        {0: 8.0, 1: 2.0},
        stream="compute",
        device=[0, 1],
    )
    collective = _node(
        "reduce.0",
        {0: 3.0, 1: 5.0},
        stream="dp_stream",
        kind="REDUCE_COMM",
        device=[0, 1],
        meta={"comm_id": "grad.bucket", "participants": [0, 1]},
    )

    result = simulate_training_step(
        [_dag(producer, collective, edges=(("backward.0", "reduce.0", "data"),))]
    )

    # Rank 0 arrives last at 8 us.  Its 3 us endpoint timing is the optimistic
    # post-rendezvous service proxy; rank 1's 5 us raw interval includes 2 us
    # of arrival skew that the graph has already modeled.
    assert result.step_time_us == pytest.approx(11.0)
    assert result.event(0, "reduce.0").start_us == pytest.approx(8.0)
    assert result.event(1, "reduce.0").start_us == pytest.approx(8.0)
    assert result.event(0, "reduce.0").profile_duration_us == pytest.approx(3.0)
    assert result.event(1, "reduce.0").profile_duration_us == pytest.approx(5.0)
    assert result.event(0, "reduce.0").effective_duration_us == pytest.approx(3.0)
    assert result.event(1, "reduce.0").effective_duration_us == pytest.approx(3.0)
    assert result.event(0, "reduce.0").duration_us == pytest.approx(3.0)
    assert result.event(1, "reduce.0").duration_us == pytest.approx(3.0)


def test_missing_rank_profile_is_rejected() -> None:
    node = _node("a", {0: 2.0}, device=[0, 1])
    dag = _dag(node)

    with pytest.raises(SimulationValidationError, match="missing profile.*rank 1"):
        simulate_training_step([dag])


def test_schedule_topology_rejects_device_sets_that_disagree_with_rank_math() -> None:
    pp0 = _dag(_node("pp0", {1: 1.0, 3: 1.0}, device=[1, 3]))
    pp1 = _dag(_node("pp1", {0: 1.0, 2: 1.0}, device=[0, 2]))
    plan = SimpleNamespace(
        per_pp_dags=[pp0, pp1],
        topology={"pp_degree": 2, "dp_degree": 2},
    )

    with pytest.raises(SimulationValidationError, match="global-rank arithmetic"):
        simulate_training_step(plan)


def test_missing_p2p_peer_is_rejected() -> None:
    send = _node(
        "send.0",
        {0: 2.0},
        kind="SEND_COMM",
        meta={"comm_id": "orphan", "peer_pp_rank": 1},
    )

    with pytest.raises(SimulationValidationError, match="exactly one SEND and one RECV"):
        simulate_training_step([_dag(send)])


def test_rank_local_prologue_gates_every_stream() -> None:
    dag = _dag(
        _node("a", {0: 6.0}, stream="left"),
        _node("b", {0: 2.0}, stream="right"),
    )
    dag.prologue_profile_times_by_mode = {
        "interference_free": {0: _profile(4.0)}
    }

    result = simulate_training_step([dag])

    assert result.step_time_us == pytest.approx(10.0)
    assert result.event(0, "__step_prologue__").end_us == pytest.approx(4.0)
    assert result.event(0, "a").start_us == pytest.approx(4.0)
    assert result.event(0, "b").start_us == pytest.approx(4.0)


def test_profiled_plan_json_round_trip_is_direct_simulator_input(tmp_path) -> None:
    node = TrainingDAGNode(
        uid="forward",
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={"PP": 0},
        device=[0],
        stream="compute",
        profile_times_by_mode={
            "interference_free": {
                0: NodeProfile.from_samples((9.0, 11.0))
            }
        },
    )
    dag = _dag(node)
    dag.prologue_profile_times_by_mode = {
        "interference_free": {0: NodeProfile.from_samples((2.0,))}
    }
    plan = ProfiledTrainingPlan.from_dags(
        [dag],
        profile_configs={
            "interference_free": ProfileConfig(warmup_iters=0, repeat_iters=2)
        },
        topology={"pp_degree": 1, "dp_degree": 1},
    )
    artifact = tmp_path / "profile.json"
    plan.save_json(artifact)

    loaded = ProfiledTrainingPlan.load_json(artifact)
    result = simulate_training_step(loaded)

    assert result.step_time_us == pytest.approx(12.0)
    assert result.event(0, "forward").profile_duration_us == pytest.approx(10.0)
