from types import SimpleNamespace

from src.runtime import RuntimeState


def _runtime(global_pp_rank: int = 1) -> RuntimeState:
    runtime = RuntimeState(
        pp_rank=global_pp_rank,
        dp_rank=0,
        dp_degree=1,
        pp_degree=4,
        world_size=4,
    )
    runtime.directional_p2p_streams = {
        "pp_lo_hi": object(),
        "pp_hi_lo": object(),
    }
    runtime.streams = {
        "default_stream": object(),
        "compute_stream": object(),
    }
    return runtime


def test_directional_streams_match_p2p_process_groups() -> None:
    runtime = _runtime(global_pp_rank=1)

    send_up = SimpleNamespace(
        uid="send-up", node_kind="SEND_COMM", peer_pp_rank=2
    )
    recv_from_up = SimpleNamespace(
        uid="recv-up", node_kind="RECV_COMM", peer_pp_rank=2
    )
    send_down = SimpleNamespace(
        uid="send-down", node_kind="SEND_COMM", peer_pp_rank=0
    )
    recv_from_down = SimpleNamespace(
        uid="recv-down", node_kind="RECV_COMM", peer_pp_rank=0
    )

    assert (
        runtime.directional_p2p_stream_for_task(send_up)
        is runtime.directional_p2p_streams["pp_lo_hi"]
    )
    assert (
        runtime.directional_p2p_stream_for_task(recv_from_up)
        is runtime.directional_p2p_streams["pp_hi_lo"]
    )
    assert (
        runtime.directional_p2p_stream_for_task(send_down)
        is runtime.directional_p2p_streams["pp_hi_lo"]
    )
    assert (
        runtime.directional_p2p_stream_for_task(recv_from_down)
        is runtime.directional_p2p_streams["pp_lo_hi"]
    )


def test_directional_p2p_stream_rejects_non_p2p_task() -> None:
    runtime = _runtime()
    task = SimpleNamespace(
        uid="compute", node_kind="COMPUTE", peer_pp_rank=2
    )

    try:
        runtime.directional_p2p_stream_for_task(task)
    except ValueError as exc:
        assert "SEND_COMM or RECV_COMM" in str(exc)
    else:
        raise AssertionError("expected non-P2P profile stream lookup to fail")


def test_execution_stream_policy_uses_directional_p2p_streams() -> None:
    runtime = _runtime(global_pp_rank=1)
    p2p = SimpleNamespace(
        uid="send-up", node_kind="SEND_COMM", peer_pp_rank=2
    )
    compute = SimpleNamespace(uid="compute", node_kind="COMPUTE")
    logical_stream = runtime.streams["compute_stream"]

    assert (
        runtime.execution_stream_for_task(p2p, logical_stream)
        is runtime.directional_p2p_streams["pp_lo_hi"]
    )
    assert (
        runtime.execution_stream_for_task(compute, logical_stream)
        is logical_stream
    )


def test_interference_free_profile_preserves_non_p2p_logical_stream() -> None:
    runtime = _runtime()
    task = SimpleNamespace(uid="compute", node_kind="COMPUTE")
    logical_stream = runtime.streams["compute_stream"]

    assert (
        runtime.profile_stream_for_task(
            task, logical_stream, profile_mode="interference_free"
        )
        is logical_stream
    )


def test_normal_profile_preserves_non_p2p_logical_stream() -> None:
    runtime = _runtime()
    task = SimpleNamespace(uid="compute", node_kind="COMPUTE")
    logical_stream = runtime.streams["compute_stream"]

    assert (
        runtime.profile_stream_for_task(
            task, logical_stream, profile_mode="normal"
        )
        is logical_stream
    )


def test_both_profile_modes_keep_directional_p2p_streams() -> None:
    runtime = _runtime(global_pp_rank=1)
    task = SimpleNamespace(
        uid="send-up", node_kind="SEND_COMM", peer_pp_rank=2
    )
    logical_stream = runtime.streams["compute_stream"]

    for profile_mode in ("interference_free", "normal"):
        assert (
            runtime.profile_stream_for_task(
                task, logical_stream, profile_mode=profile_mode
            )
            is runtime.directional_p2p_streams["pp_lo_hi"]
        )


def test_profile_stream_policy_rejects_unknown_mode() -> None:
    runtime = _runtime()
    task = SimpleNamespace(uid="compute", node_kind="COMPUTE")

    try:
        runtime.profile_stream_for_task(
            task, runtime.streams["compute_stream"], profile_mode="typo"
        )
    except ValueError as exc:
        assert "unsupported profile mode" in str(exc)
    else:
        raise AssertionError("expected an unknown profile mode to fail")
