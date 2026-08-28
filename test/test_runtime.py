"""Actor-group topology and model-state runtime tests."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import src.actor as actor_module
import src.runtime as runtime_module

from src.compile import (
    _actor_group_dags_by_id,
    _input_actor_group_id,
    _partition_actor_group_batch,
)
from src.actor import PiperActor
from src.dag import TrainingDAG, TrainingDAGNode
from src.model_state import (
    dynamo_buffer_name_to_state_dict_key,
    dynamo_parameter_name_to_state_dict_key,
    initialize_model_parameters_,
    initialize_parameter_,
    initialize_route_parameter_,
)
from src.runtime import BucketState, ParamStorage, RuntimeState, StageStore
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
        ((0, 2), "dense_state_lo_hi"),
        ((0, 2), "dense_state_hi_lo"),
        ((4, 6), "dense_state_lo_hi"),
        ((4, 6), "dense_state_hi_lo"),
        ((8, 9), "dense_state_lo_hi"),
        ((8, 9), "dense_state_hi_lo"),
        ((10, 11), "dense_state_lo_hi"),
        ((10, 11), "dense_state_hi_lo"),
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


def _two_rank_runtime(**overrides) -> RuntimeState:
    values = {
        "global_rank": 0,
        "pp_rank": 0,
        "pp_degree": 1,
        "world_size": 2,
        "device": "cpu",
        "actor_group_id": 0,
        "actor_group_local_rank": 0,
        "actor_group_devices": (0, 1),
        "actor_groups": ((0, 1),),
    }
    values.update(overrides)
    return RuntimeState(**values)





def test_named_initialization_is_independent_of_partition_and_order() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 3, bias=False),
        torch.nn.Linear(3, 2, bias=False),
    )
    initialize_model_parameters_(model, model_seed=1234)

    independently_initialized = {}
    for name, parameter in reversed(list(model.named_parameters())):
        value = torch.empty_like(parameter)
        initialize_parameter_(value, name, model_seed=1234)
        independently_initialized[name] = value

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, independently_initialized[name])




def test_route_initialization_matches_contiguous_global_slices() -> None:
    global_parameter = torch.empty(8, 3, 2)
    initialize_route_parameter_(
        global_parameter,
        "layers.0.moe.experts.w1",
        model_seed=123,
    )

    first = torch.empty(4, 3, 2)
    second = torch.empty(4, 3, 2)
    initialize_route_parameter_(
        first,
        "layers.0.moe.experts.w1",
        global_route_start=0,
        model_seed=123,
    )
    initialize_route_parameter_(
        second,
        "layers.0.moe.experts.w1",
        global_route_start=4,
        model_seed=123,
    )

    assert torch.equal(torch.cat((first, second), dim=0), global_parameter)




def test_zero2_rebuilds_full_parameters_after_shard_update(monkeypatch) -> None:
    flat_params = torch.tensor([1.0, 2.0, 3.0, 4.0])
    shard_param = flat_params[:2]
    shard_param.requires_grad_(True)
    shard_optimizer = Mock()
    bucket = BucketState(
        flat_params=flat_params,
        shard_param=shard_param,
        shard_optimizer=shard_optimizer,
        reduce_scatter_grads=torch.tensor([0.25, 0.5]),
    )
    stages = StageStore(
        buckets={"bucket": bucket},
        grad_sharded_ubids={"bucket"},
        param_sharded_ubids=set(),
    )
    sync_group = object()
    validate_metadata = Mock()
    storage = ParamStorage(
        runtime=SimpleNamespace(
            actor_group_size=2,
            actor_group_sync_group=sync_group,
            validate_actor_group_metadata=validate_metadata,
        ),
        stages=stages,
        logger=Mock(),
    )
    stream = Mock()
    gathered_values = torch.tensor([10.0, 20.0, 30.0, 40.0])

    monkeypatch.setattr(
        runtime_module.torch.cuda, "stream", lambda _stream: nullcontext()
    )

    def all_gather_into_tensor(output, input_tensor, *, group):
        assert input_tensor is shard_param
        assert group is sync_group
        output.copy_(gathered_values)

    monkeypatch.setattr(
        runtime_module.dist,
        "all_gather_into_tensor",
        all_gather_into_tensor,
    )

    storage.step_zero_shard_optimizers(stream, {})

    shard_optimizer.step.assert_called_once_with()
    assert bucket.shard_param.grad is None
    assert torch.equal(bucket.flat_params, gathered_values)
    validate_metadata.assert_called_once_with(
        context="rebuild ZeRO-2 parameters for ZeRO bucket 'bucket'"
    )




def test_zero_collective_errors_when_actor_group_communicator_is_missing(
    monkeypatch,
) -> None:
    flat_params = torch.ones(4)
    shard_param = flat_params[:2]
    bucket = BucketState(
        flat_params=flat_params,
        shard_param=shard_param,
        shard_optimizer=Mock(),
        reduce_scatter_grads=torch.ones(2),
    )
    stages = StageStore(
        buckets={"bucket": bucket},
        grad_sharded_ubids={"bucket"},
    )
    storage = ParamStorage(
        runtime=_two_rank_runtime(),
        stages=stages,
        logger=Mock(),
    )
    monkeypatch.setattr(
        runtime_module.torch.cuda, "stream", lambda _stream: nullcontext()
    )

    with pytest.raises(RuntimeError, match="communicator was not initialized"):
        storage.step_zero_shard_optimizers(Mock(), {})




def test_route_owned_export_uses_canonical_collective_order(monkeypatch) -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    local_parameter = torch.nn.Parameter(
        torch.tensor([[10.0], [20.0]])
    )
    bucket = BucketState(
        forward_args=[local_parameter],
        trainable_param_idxs=[0],
        param_idxs=[0],
        param_names=["l_self_parameters_experts_"],
        route_sharded_param_specs={
            0: {
                "arg_idx": 0,
                "route_axis": 0,
                "local_shape": (2, 1),
                "global_shape": (4, 1),
                "global_route_start": 0,
                "shard_devices": (0, 1),
            }
        },
    )
    actor.stages = StageStore(buckets={"bucket": bucket})
    stream = Mock()
    actor.runtime = SimpleNamespace(
        actor_group_size=2,
        actor_group_sync_group="sync",
        actor_group_devices=(0, 1),
        is_actor_group_representative=True,
        default_stream=lambda: stream,
    )
    actor.params = SimpleNamespace(drain_pending_frees=Mock())

    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(
        actor_module.torch.cuda, "stream", lambda _stream: nullcontext()
    )

    def fake_all_gather(output, input_tensor, *, group):
        assert input_tensor is not None
        assert group == "sync"
        output.copy_(torch.tensor([[10.0], [20.0], [30.0], [40.0]]))

    monkeypatch.setattr(
        actor_module.dist,
        "all_gather_into_tensor",
        fake_all_gather,
    )

    exported = actor.export_model_parameter_state()
    assert torch.equal(
        exported["experts"],
        torch.tensor([[10.0], [20.0], [30.0], [40.0]]),
    )
    stream.synchronize.assert_called_once_with()





