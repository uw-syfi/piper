from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock, call

import torch

import src.executors as executors
from src.executors import CommunicationExecutor, ComputeExecutor, DagExecutor
from src.runtime import BucketState, StageStore
from src.tasks import TaskType


def test_backward_input_events_waits_for_recv_all_gather_and_a2a() -> None:
    stream = Mock()
    recv_evt = object()
    all_gather_evt = object()
    a2a_evt = object()

    recv_pred = SimpleNamespace(uid="recv", task_type=TaskType.RECV)
    all_gather_pred = SimpleNamespace(uid="all_gather", task_type=TaskType.ALL_GATHER)
    a2a_pred = SimpleNamespace(uid="a2a", task_type=TaskType.BWD_A2A)
    node = SimpleNamespace(data_preds=[recv_pred, all_gather_pred, a2a_pred])

    events = SimpleNamespace(
        recv={recv_pred.uid: recv_evt},
        all_gather={all_gather_pred.uid: all_gather_evt},
        a2a={a2a_pred.uid: a2a_evt},
    )
    executor = DagExecutor(
        runtime=SimpleNamespace(stream_for_task=lambda _node: stream),
        stages=None,
        buffers=None,
        events=events,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )

    actual_recv_pred, actual_a2a_pred = executor._wait_for_backward_input_events(
        node, stream
    )

    assert actual_recv_pred is recv_pred
    assert actual_a2a_pred is a2a_pred
    assert stream.wait_event.call_args_list == [
        call(recv_evt),
        call(all_gather_evt),
        call(a2a_evt),
    ]
    assert events.recv == {}
    assert events.a2a == {}


def test_split_backward_input_consumes_local_downstream_grads() -> None:
    local_pred = SimpleNamespace(uid="local_bwd", task_type=TaskType.BWD_I)
    node = SimpleNamespace(data_preds=[local_pred])
    output_a = torch.randn(2, 3, requires_grad=True)
    output_b = torch.randn(2, 3, requires_grad=True)
    detached_a = output_a.detach().requires_grad_(True)
    nondifferentiable = torch.ones(2, dtype=torch.int64)
    detached_b = output_b.detach().requires_grad_(True)
    grad_a = torch.randn_like(output_a)
    grad_b = torch.randn_like(output_b)
    buffers = SimpleNamespace(
        task={
            local_pred.uid: {
                "inp_grads": [grad_a, None, grad_b],
            }
        },
        release=Mock(),
    )
    executor = DagExecutor(
        runtime=None,
        stages=None,
        buffers=buffers,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )

    outputs, output_grads = executor._consume_local_backward_output_grads(
        node,
        {"detached_outs": [detached_a, nondifferentiable, detached_b]},
        [output_a, output_b],
    )

    assert outputs[0] is output_a
    assert outputs[1] is output_b
    assert output_grads is not None
    assert output_grads[0] is grad_a
    assert output_grads[1] is grad_b
    buffers.release.assert_called_once_with(local_pred.uid)


def test_update_steps_sharded_and_regular_optimizers(monkeypatch) -> None:
    stream = Mock()
    all_reduce_evt = object()
    reduce_scatter_evt = object()
    backward_evt = object()
    regular_optimizer = Mock()
    regular_bucket = SimpleNamespace(optimizer=regular_optimizer)
    sharded_bucket = SimpleNamespace(optimizer=None)

    params = Mock()
    params.has_zero_shard_optimizers.return_value = True
    stages = SimpleNamespace(
        buckets={"regular": regular_bucket, "sharded": sharded_bucket}
    )
    events = SimpleNamespace(
        all_reduce={"all_reduce": all_reduce_evt},
        reduce_scatter={"reduce_scatter": reduce_scatter_evt},
        backward={"regular": backward_evt},
    )
    executor = DagExecutor(
        runtime=None,
        stages=stages,
        buffers=None,
        events=events,
        params=params,
        communication=None,
        compute=None,
        logger=None,
    )
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())
    synchronize = Mock()
    monkeypatch.setattr(executors.torch.cuda, "synchronize", synchronize)

    executor._update(stream, [])

    params.drain_pending_frees.assert_called_once_with()
    assert stream.wait_event.call_args_list == [
        call(all_reduce_evt),
        call(backward_evt),
    ]
    params.step_zero_shard_optimizers.assert_called_once_with(
        stream, events.reduce_scatter
    )
    regular_optimizer.step.assert_called_once_with()
    synchronize.assert_called_once_with()


def test_grad_with_param_layout_honors_declared_grad_dtype() -> None:
    weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    weight.grad_dtype = torch.float32

    grad = ComputeExecutor.grad_with_param_layout(
        weight,
        torch.ones_like(weight),
    )

    assert grad.dtype == torch.float32
    weight.grad = grad


def test_reduce_scatter_temporary_uses_compute_cache_and_records_dp_stream(
    monkeypatch,
) -> None:
    flat_grads = torch.arange(8, dtype=torch.float32)
    reduced = Mock()
    bucket = BucketState(
        flat_grads=flat_grads,
        reduce_scatter_grads=reduced,
        param_shard_info=(0, 4, 8),
        trainable_param_idxs=[0],
    )
    stages = StageStore(
        buckets={"bucket": bucket},
        grad_sharded_ubids={"bucket"},
    )
    dp_group = object()
    allocation_stream = object()
    default_stream = Mock(return_value=allocation_stream)
    executor = CommunicationExecutor(
        runtime=SimpleNamespace(
            dp_group=dp_group,
            default_stream=default_stream,
        ),
        stages=stages,
        logger=Mock(),
    )
    dp_stream = object()
    scratch = Mock()
    stream_context = Mock(side_effect=lambda _stream: nullcontext())

    monkeypatch.setattr(executors.torch.cuda, "stream", stream_context)
    monkeypatch.setattr(executors.torch, "empty_like", Mock(return_value=scratch))

    def reduce_scatter_tensor(output, input_tensor, *, group):
        assert output is scratch
        assert input_tensor is flat_grads
        assert group is dp_group

    monkeypatch.setattr(executors.dist, "reduce_scatter_tensor", reduce_scatter_tensor)

    expected_bytes = flat_grads.numel() * flat_grads.element_size()
    assert executor.reduce_scatter("bucket", dp_stream) == expected_bytes
    default_stream.assert_called_once_with()
    assert stream_context.call_args_list == [
        call(allocation_stream),
        call(dp_stream),
    ]
    scratch.record_stream.assert_called_once_with(dp_stream)
    reduced.add_.assert_called_once_with(scratch)


def test_a2a_boundary_reuses_dispatch_state_for_combine(monkeypatch) -> None:
    def fake_permute(tensor, counts, ep_degree, num_local_experts):
        assert ep_degree == 1
        assert num_local_experts == counts.numel()
        return (
            tensor.shape,
            tensor + 1,
            torch.arange(tensor.shape[0]),
            counts + 2,
        )

    monkeypatch.setattr(executors, "_permute", fake_permute)
    monkeypatch.setattr(
        executors,
        "_unpermute",
        lambda tensor, _shape, _indices: tensor + 3,
    )
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    executor = DagExecutor(
        runtime=SimpleNamespace(ep_group="ep", global_rank=3),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=Mock(),
        compute=None,
        logger=None,
    )
    incoming = SimpleNamespace(
        uid="incoming",
        task_type=TaskType.FWD_A2A,
        node_meta={
            "direction": "incoming",
            "expert_fwd_uid": "expert",
            "participants": [3],
        },
    )
    outgoing = SimpleNamespace(
        uid="outgoing",
        task_type=TaskType.FWD_A2A,
        node_meta={
            "direction": "outgoing",
            "expert_fwd_uid": "expert",
            "participants": [3],
        },
    )
    tensor = torch.ones(4, 2)
    counts = torch.tensor([1, 1, 1, 1])

    dispatched, local_counts = executor._all_to_all_ep_boundary(
        incoming,
        tensor,
        stream=object(),
        routing_counts=counts,
    )
    combined = executor._all_to_all_ep_boundary(
        outgoing,
        dispatched,
        stream=object(),
    )

    assert torch.equal(dispatched, tensor + 1)
    assert torch.equal(local_counts, counts + 2)
    assert torch.equal(combined, tensor + 4)
    state = executor.expert_a2a_states["expert"]
    assert (state.ep_rank, state.ep_degree) == (0, 1)
