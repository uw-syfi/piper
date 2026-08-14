from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import src.actor as actor_module
import src.runtime as runtime_module
from src.actor import PiperActor
from src.model_state import (
    dynamo_buffer_name_to_state_dict_key,
    dynamo_parameter_name_to_state_dict_key,
    initialize_model_parameters_,
    initialize_parameter_,
    initialize_route_parameter_,
)
from src.runtime import BucketState, ParamStorage, RuntimeState, StageStore


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


def test_runtime_rejects_missing_actor_group_metadata() -> None:
    with pytest.raises(ValueError, match="actor_groups must not be empty"):
        RuntimeState(
            global_rank=0,
            pp_rank=0,
            pp_degree=1,
            world_size=1,
        )


def test_runtime_rejects_inconsistent_actor_group_local_rank() -> None:
    with pytest.raises(ValueError, match="global_rank=0 does not match"):
        _two_rank_runtime(actor_group_local_rank=1)


def test_dynamo_parameter_name_to_state_dict_key() -> None:
    assert (
        dynamo_parameter_name_to_state_dict_key(
            "l_self_modules_tok_embeddings_parameters_weight_"
        )
        == "tok_embeddings.weight"
    )
    assert (
        dynamo_parameter_name_to_state_dict_key(
            "l_self_modules_layers_modules_3_modules_moe_modules_experts_parameters_w1_"
        )
        == "layers.3.moe.experts.w1"
    )
    assert (
        dynamo_parameter_name_to_state_dict_key("l_self_parameters_scale_")
        == "scale"
    )
    assert dynamo_parameter_name_to_state_dict_key("l_tokens_") is None


def test_dynamo_buffer_name_to_state_dict_key() -> None:
    assert (
        dynamo_buffer_name_to_state_dict_key("l_self_buffers_rope_cache_")
        == "rope_cache"
    )
    assert (
        dynamo_buffer_name_to_state_dict_key(
            "l_self_modules_layer_modules_norm_buffers_scale_"
        )
        == "layer.norm.scale"
    )
    assert dynamo_buffer_name_to_state_dict_key("l_tokens_") is None


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


def test_named_initialization_changes_with_model_seed() -> None:
    first = torch.empty(64)
    second = torch.empty(64)
    initialize_parameter_(first, "layer.weight", model_seed=1)
    initialize_parameter_(second, "layer.weight", model_seed=2)
    assert not torch.equal(first, second)


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


def test_model_route_initialization_requires_explicit_parameter_names() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="not model parameters"):
        initialize_model_parameters_(
            model,
            route_parameter_axes={"missing.weight": 0},
        )


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


def test_zero_gradient_storage_uses_actor_group_size(monkeypatch) -> None:
    param = torch.nn.Parameter(torch.ones(3))
    bucket = BucketState(
        param_shard_info=(0, 3, 5),
        param_view_specs=[(param, 0, 3, (3,))],
    )
    stages = StageStore(
        buckets={"bucket": bucket},
        grad_sharded_ubids={"bucket"},
        zero_managed_ubids={"bucket"},
    )
    storage = ParamStorage(
        runtime=_two_rank_runtime(),
        stages=stages,
        logger=Mock(),
    )
    monkeypatch.setattr(
        runtime_module.torch.cuda, "stream", lambda _stream: nullcontext()
    )

    storage.alloc_full_grads("bucket", object())

    assert bucket.flat_grads.shape == (6,)
    assert bucket.reduce_scatter_grads.shape == (3,)


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


def test_nonrepresentative_participates_in_sharded_parameter_export(
    monkeypatch,
) -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    shard_param = torch.tensor([3.0, 4.0])
    bucket = BucketState(
        shard_param=shard_param,
        param_shard_info=(2, 2, 4),
        trainable_param_idxs=[0],
        param_idxs=[0],
        param_names=["l_self_parameters_weight_"],
        param_view_specs=[(torch.nn.Parameter(torch.ones(4)), 0, 4, (4,))],
    )
    actor.stages = StageStore(
        buckets={"bucket": bucket},
        param_sharded_ubids={"bucket"},
    )
    stream = Mock()
    actor.runtime = SimpleNamespace(
        actor_group_size=2,
        actor_group_sync_group="sync",
        is_actor_group_representative=False,
        default_stream=lambda: stream,
    )
    actor.params = SimpleNamespace(drain_pending_frees=Mock())
    collective_calls = []

    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(
        actor_module.torch.cuda, "stream", lambda _stream: nullcontext()
    )

    def fake_all_gather(output, input_tensor, *, group):
        collective_calls.append((input_tensor, group))
        output.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    monkeypatch.setattr(
        actor_module.dist,
        "all_gather_into_tensor",
        fake_all_gather,
    )

    assert actor.export_model_parameter_state() == {}
    assert collective_calls == [(shard_param, "sync")]
    stream.synchronize.assert_called_once_with()


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
