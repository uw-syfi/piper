from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock, call

import torch

import src.executors as executors
from src.executors import ComputeExecutor, DagExecutor
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
