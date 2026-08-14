import pytest
import torch

from src.compile import (
    _actor_group_dags_by_id,
    _input_actor_group_id,
    _partition_actor_group_batch,
)
from src.dag import TrainingDAG, TrainingDAGNode
from src.runtime import RuntimeState
from src.simulator import _build_rank_layout
from src.state import piper_metadata


def _local_dag(
    uid: str,
    devices: list[int],
    *,
    stage_id: int = 0,
    segment_id: int = 0,
) -> TrainingDAG:
    dag = TrainingDAG()
    dag.add_node(
        TrainingDAGNode(
            uid=uid,
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag={"PASS": "F"},
            device=devices,
            stream="default_stream",
            node_meta={"stage_id": stage_id, "segment_id": segment_id},
        )
    )
    return dag


def _schedule_info() -> dict:
    return {
        "actor_groups": [
            {"id": 0, "devices": [0, 2, 4, 5], "size": 4},
            {"id": 1, "devices": [1, 3], "size": 2},
        ],
        "loss_actor_group_ids": [0],
    }


def test_local_dags_match_actor_groups_by_devices_not_list_position() -> None:
    ffn = _local_dag("ffn", [1, 3])
    attn = _local_dag("attn", [0, 2, 4, 5])

    result = _actor_group_dags_by_id([ffn, attn], _schedule_info())

    assert result == {0: attn, 1: ffn}


def test_input_is_loaded_on_group_with_first_traced_forward_segment() -> None:
    dag_by_group = {
        0: _local_dag("later", [0, 2, 4, 5], stage_id=4),
        1: _local_dag("input", [1, 3], stage_id=0),
    }

    assert _input_actor_group_id(dag_by_group) == 1


def test_disconnected_components_for_one_actor_group_are_rejected() -> None:
    with pytest.raises(ValueError, match="order edges"):
        _actor_group_dags_by_id(
            [
                _local_dag("attn0", [0, 2, 4, 5]),
                _local_dag("attn1", [0, 2, 4, 5]),
                _local_dag("ffn", [1, 3]),
            ],
            _schedule_info(),
        )


def test_runtime_exposes_exact_actor_group_and_pairwise_union_groups() -> None:
    runtime = RuntimeState(
        global_rank=4,
        pp_rank=0,
        pp_degree=2,
        world_size=6,
        actor_group_id=0,
        actor_group_local_rank=2,
        actor_group_devices=(0, 2, 4, 5),
        actor_groups=((0, 2, 4, 5), (1, 3)),
    )

    assert runtime.global_rank == 4
    assert runtime.actor_group_size == 4
    assert not runtime.is_actor_group_representative
    assert runtime.required_process_group_participants() == (
        (0, 2, 4, 5),
        (1, 3),
    )
    assert runtime.required_routed_process_group_participants() == (
        (0, 1, 2, 3, 4, 5),
    )
    assert runtime.process_group_creation_specs() == (
        ((0, 2, 4, 5), None),
        ((1, 3), None),
        ((0, 1, 2, 3, 4, 5), "routed_layout"),
        ((0, 1, 2, 3, 4, 5), "m_to_n"),
        ((0, 1, 2, 3, 4, 5), "n_to_m"),
    )

    participants = (0, 1, 2, 3, 4, 5)
    runtime.process_groups_by_participants[(participants, "routed_layout")] = "layout"
    runtime.process_groups_by_participants[(participants, "m_to_n")] = "dispatch"
    runtime.process_groups_by_participants[(participants, "n_to_m")] = "combine"
    runtime.process_groups_by_participants[(0, 2, 4, 5)] = "within-attn"
    assert runtime.process_group_for_participants(
        participants, channel="routed_layout"
    ) == "layout"
    assert runtime.process_group_for_participants(
        [5, 3, 1, 4, 2, 0], channel="m_to_n"
    ) == "dispatch"
    assert runtime.process_group_for_participants(
        participants, channel="n_to_m"
    ) == "combine"
    assert runtime.process_group_for_participants([5, 0, 4, 2]) == "within-attn"
    with pytest.raises(ValueError, match="collective channel"):
        runtime.process_group_for_participants(participants, channel="dispatch")
    with pytest.raises(KeyError, match="No process group"):
        runtime.process_group_for_participants([0, 4])
    with pytest.raises(ValueError, match="equal-width ActorGroups"):
        runtime.pipeline_peer_global_rank(1)

    representative = RuntimeState(
        global_rank=0,
        pp_rank=0,
        pp_degree=2,
        world_size=6,
        actor_group_id=0,
        actor_group_local_rank=0,
        actor_group_devices=(0, 2, 4, 5),
        actor_groups=((0, 2, 4, 5), (1, 3)),
    )
    assert representative.is_actor_group_representative


def test_equal_width_actor_groups_define_aligned_direct_p2p_peers() -> None:
    groups = (
        (0, 4, 8, 10),
        (1, 5),
        (2, 6, 9, 11),
        (3, 7),
    )
    runtime = RuntimeState(
        global_rank=4,
        pp_rank=0,
        pp_degree=4,
        world_size=12,
        actor_group_id=0,
        actor_group_local_rank=1,
        actor_group_devices=groups[0],
        actor_groups=groups,
    )

    assert runtime.pipeline_peer_global_rank(2) == 6
    expected_pair_specs = (
        ((0, 2), "pp_lo_hi"),
        ((0, 2), "pp_hi_lo"),
        ((4, 6), "pp_lo_hi"),
        ((4, 6), "pp_hi_lo"),
        ((8, 9), "pp_lo_hi"),
        ((8, 9), "pp_hi_lo"),
        ((10, 11), "pp_lo_hi"),
        ((10, 11), "pp_hi_lo"),
    )
    all_specs = runtime.required_p2p_process_group_specs()
    assert all(spec in all_specs for spec in expected_pair_specs)

    runtime.process_groups_by_participants[((4, 6), "pp_lo_hi")] = "forward"
    runtime.process_groups_by_participants[((4, 6), "pp_hi_lo")] = "reverse"
    send = type(
        "Task",
        (),
        {"uid": "send", "node_kind": "SEND_COMM", "peer_pp_rank": 2},
    )()
    recv = type(
        "Task",
        (),
        {"uid": "recv", "node_kind": "RECV_COMM", "peer_pp_rank": 2},
    )()
    assert runtime.p2p_process_group_for_task(send) == "forward"
    assert runtime.p2p_process_group_for_task(recv) == "reverse"


def test_simulator_accepts_unequal_schedule_actor_groups() -> None:
    layout = _build_rank_layout(
        [
            _local_dag("attn", [0, 2, 4, 5]),
            _local_dag("ffn", [1, 3]),
        ],
        _schedule_info() | {"pp_degree": 2, "world_size": 6},
        pp_degree=None,
        profile_mode="interference_free",
    )

    assert layout.ranks_by_pp == ((0, 2, 4, 5), (1, 3))
    assert layout.lane_by_rank == {0: 0, 2: 1, 4: 2, 5: 3, 1: 0, 3: 1}


class _FakeRemoteMethod:
    def __init__(self, result):
        self.result = result

    def remote(self, *args, **kwargs):
        del args, kwargs
        return self.result


class _FakeActor:
    def __init__(self, result):
        self.run_dag = _FakeRemoteMethod(result)


class _RecordingRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return None


class _LoadActor:
    def __init__(self):
        self.load_training_dag = _RecordingRemoteMethod()


def test_actor_group_batch_partition_gives_each_attn_rank_distinct_tokens() -> None:
    tokens = torch.arange(8 * 3).view(8, 3)

    shards = _partition_actor_group_batch(
        [tokens],
        [0, 2, 4, 5],
        value_name="input",
    )

    assert [shard[0].tolist() for shard in shards] == [
        tokens[0:2].tolist(),
        tokens[2:4].tolist(),
        tokens[4:6].tolist(),
        tokens[6:8].tolist(),
    ]
    assert torch.equal(torch.cat([shard[0] for shard in shards]), tokens)


@pytest.mark.parametrize("batch_size", [3, 6])
def test_actor_group_batch_partition_rejects_too_small_or_uneven_microbatch(
    batch_size: int,
) -> None:
    with pytest.raises(ValueError, match="at least and divisible"):
        _partition_actor_group_batch(
            [torch.arange(batch_size)],
            [0, 2, 4, 5],
            value_name="input",
        )


def test_piper_exec_averages_distinct_loss_shards_within_actor_group(monkeypatch) -> None:
    import src.piper as piper_module

    actors = {
        0: _FakeActor({"losses": [1.0, 5.0]}),
        2: _FakeActor({"losses": [2.0, 6.0]}),
        4: _FakeActor({"losses": [3.0, 7.0]}),
        5: _FakeActor({"losses": [4.0, 8.0]}),
        1: _FakeActor({"losses": []}),
        3: _FakeActor({"losses": []}),
    }
    monkeypatch.setattr(piper_metadata, "actors", actors)
    monkeypatch.setattr(
        piper_metadata,
        "schedule_info",
        _schedule_info(),
    )
    monkeypatch.setattr(piper_module.ray, "get", lambda refs: refs)

    losses = piper_module.piper_exec_dag(loss_fn=None)

    assert losses == [2.5, 6.5]


def test_piper_exec_uses_only_declared_loss_actor_group(monkeypatch) -> None:
    import src.piper as piper_module

    actors = {
        0: _FakeActor({"losses": []}),
        2: _FakeActor({"losses": []}),
        1: _FakeActor({"losses": [3.0]}),
        3: _FakeActor({"losses": [5.0]}),
    }
    monkeypatch.setattr(piper_metadata, "actors", actors)
    monkeypatch.setattr(
        piper_metadata,
        "schedule_info",
        {
            "actor_groups": [
                {"id": 0, "devices": [0, 2]},
                {"id": 1, "devices": [1, 3]},
            ],
            "loss_actor_group_ids": [1],
        },
    )
    monkeypatch.setattr(piper_module.ray, "get", lambda refs: refs)

    assert piper_module.piper_exec_dag(loss_fn=None) == [4.0]


def test_compile_loads_identical_dag_on_every_actor_group_member(monkeypatch) -> None:
    import src.compile as compile_module

    actors = {rank: _LoadActor() for rank in range(6)}
    monkeypatch.setattr(piper_metadata, "actors", actors)
    monkeypatch.setattr(compile_module.ray, "get", lambda refs: refs)
    attn = _local_dag("attn", [0, 2, 4, 5])
    ffn = _local_dag("ffn", [1, 3])
    info = _schedule_info()

    compile_module._load_training_dags_on_actors([ffn, attn], info)

    for rank in (0, 2, 4, 5):
        assert actors[rank].load_training_dag.calls == [((attn,), {})]
    for rank in (1, 3):
        assert actors[rank].load_training_dag.calls == [((ffn,), {})]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"actor_groups": ()}, "actor_groups must not be empty"),
        ({"actor_group_devices": (0, 4)}, "do not match"),
        ({"actor_group_local_rank": 0}, "global_rank=4 does not match"),
    ],
)
def test_runtime_rejects_inconsistent_actor_group_metadata(
    overrides: dict, message: str
) -> None:
    kwargs = {
        "global_rank": 4,
        "pp_rank": 0,
        "pp_degree": 2,
        "world_size": 6,
        "actor_group_id": 0,
        "actor_group_local_rank": 2,
        "actor_group_devices": (0, 2, 4, 5),
        "actor_groups": ((0, 2, 4, 5), (1, 3)),
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        RuntimeState(**kwargs)
