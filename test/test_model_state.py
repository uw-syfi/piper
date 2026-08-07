from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import torch

import src.runtime as runtime_module
from src.model_state import (
    dynamo_buffer_name_to_state_dict_key,
    dynamo_parameter_name_to_state_dict_key,
    initialize_model_parameters_,
    initialize_parameter_,
)
from src.runtime import BucketState, ParamStorage, StageStore


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
    dp_group = object()
    storage = ParamStorage(
        runtime=SimpleNamespace(dp_degree=2, dp_group=dp_group),
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
        assert group is dp_group
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
