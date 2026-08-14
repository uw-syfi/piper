from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock, call

import pytest
import torch

import src.executors as executors
from src.executors import CommunicationExecutor, ComputeExecutor, DagExecutor
from src.tasks import TaskType


def _route_meta(shard_devices, num_global_routes):
    width = num_global_routes // len(shard_devices)
    return {
        "num_global_routes": num_global_routes,
        "shard_degree": len(shard_devices),
        "num_local_routes": width,
        "shard_devices": list(shard_devices),
        "route_partitions": [
            {
                "device": device,
                "start": index * width,
                "stop": (index + 1) * width,
            }
            for index, device in enumerate(shard_devices)
        ],
    }


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


def test_actor_group_p2p_uses_the_task_specific_pair_communicator(
    monkeypatch,
) -> None:
    task = SimpleNamespace(uid="send", node_kind="SEND_COMM")
    runtime = SimpleNamespace(
        actor_groups=((0, 2), (1, 3)),
        global_rank=0,
        pipeline_peer_global_rank=lambda _peer_group: 1,
        p2p_process_group_for_task=Mock(return_value="pair-group"),
    )
    communication = CommunicationExecutor(
        runtime=runtime,
        stages=None,
        logger=None,
    )
    tensor = torch.randn(2, 3)
    send = Mock()
    monkeypatch.setattr(executors.dist, "send", send)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    communication.send(tensor, 1, object(), task=task)

    runtime.p2p_process_group_for_task.assert_called_once_with(task)
    send.assert_called_once_with(tensor, dst=1, group="pair-group")


def test_replicated_gradient_all_reduce_averages_over_actor_group(
    monkeypatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([3.0, 5.0])
    bucket = SimpleNamespace(
        trainable_param_idxs=[0],
        forward_args=[parameter],
    )
    stages = SimpleNamespace(
        get_bucket=lambda _ubid: bucket,
        bucket=lambda _ubid: bucket,
    )
    runtime = SimpleNamespace(
        actor_group_size=2,
        actor_group_sync_group="replicas",
    )
    communication = CommunicationExecutor(runtime, stages, logger=None)

    def fake_all_reduce(tensor, *, group):
        assert group == "replicas"
        tensor.mul_(2)

    monkeypatch.setattr(executors.dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    transferred = communication.all_reduce_grads("bucket", object())

    assert torch.equal(parameter.grad, torch.tensor([3.0, 5.0]))
    assert transferred == parameter.grad.numel() * parameter.grad.element_size()


def test_replicated_gradient_reduce_scatter_averages_before_accumulation(
    monkeypatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    bucket = SimpleNamespace(
        trainable_param_idxs=[0],
        forward_args=[parameter],
        param_shard_info=(0, 2, 4),
        flat_grads=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        reduce_scatter_grads=torch.tensor([10.0, 20.0]),
    )
    stages = SimpleNamespace(
        get_bucket=lambda _ubid: bucket,
        bucket=lambda _ubid: bucket,
        grad_sharded_ubids={"bucket"},
    )
    runtime = SimpleNamespace(
        actor_group_size=2,
        actor_group_sync_group="replicas",
    )
    communication = CommunicationExecutor(runtime, stages, logger=None)

    def fake_reduce_scatter(output, _input, *, group):
        assert group == "replicas"
        output.copy_(torch.tensor([6.0, 10.0]))

    monkeypatch.setattr(
        executors.dist, "reduce_scatter_tensor", fake_reduce_scatter
    )
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    transferred = communication.reduce_scatter("bucket", object())

    assert torch.equal(bucket.reduce_scatter_grads, torch.tensor([13.0, 25.0]))
    assert transferred == bucket.flat_grads.numel() * bucket.flat_grads.element_size()


def test_grad_with_param_layout_honors_declared_grad_dtype() -> None:
    weight = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    weight.grad_dtype = torch.float32

    grad = ComputeExecutor.grad_with_param_layout(
        weight,
        torch.ones_like(weight),
    )

    assert grad.dtype == torch.float32
    weight.grad = grad


def test_a2a_boundary_reuses_dispatch_state_for_combine(monkeypatch) -> None:
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    executor = DagExecutor(
        runtime=SimpleNamespace(actor_group_routed_group="ep", global_rank=3),
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
            **_route_meta([3], 4),
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
    counts = torch.tensor([[1, 1, 1, 1]])

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

    assert dispatched is tensor
    assert torch.equal(local_counts, counts)
    assert combined is dispatched
    state = executor.expert_a2a_states["expert"]
    assert state.participants == (3,)
    assert state.shard_devices == (3,)


def test_a2a_boundary_rejects_noncanonical_serialized_shard_order(
    monkeypatch,
) -> None:
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())
    executor = DagExecutor(
        runtime=SimpleNamespace(actor_group_routed_group="ep", global_rank=0),
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
            "participants": [0, 1],
            **_route_meta([1, 0], 4),
        },
    )

    with pytest.raises(ValueError, match="ascending global actor-rank order"):
        executor._all_to_all_ep_boundary(
            incoming,
            torch.ones(4, 2),
            stream=object(),
            routing_counts=torch.ones((1, 4), dtype=torch.int64),
        )


def test_a2a_dispatch_preserves_source_major_counts_and_opaque_rank_blocks(
    monkeypatch,
) -> None:
    state = executors._ExpertA2AState(
        participants=(0, 1),
        shard_devices=(0, 1),
        num_global_routes=4,
        route_partitions=((0, 2), (2, 4)),
    )
    counts = torch.tensor([[1, 2, 3, 4]])
    routed = torch.randn(10, 3)
    received = torch.cat((torch.zeros(11, 3), torch.ones(15, 3)))
    count_collective_inputs = []
    payload_collective_inputs = []

    def fake_count_a2a(tensor, *_args, **_kwargs):
        count_collective_inputs.append(tensor.clone())
        return torch.tensor([5, 6, 7, 8])

    monkeypatch.setattr(executors, "all_to_all_single", fake_count_a2a)

    def fake_payload_a2a(tensor, *_args, **_kwargs):
        payload_collective_inputs.append(tensor.clone())
        return received

    monkeypatch.setattr(
        executors,
        "all_to_all_single_autograd",
        fake_payload_a2a,
    )
    monkeypatch.setattr(
        executors.torch.ops._c10d_functional,
        "wait_tensor",
        lambda tensor: tensor,
    )
    executor = DagExecutor(
        runtime=SimpleNamespace(actor_group_routed_group="ep"),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )

    dispatched, local_counts = executor._dispatch_expert_a2a(
        state, routed, counts
    )

    assert torch.equal(count_collective_inputs[0], counts.flatten())
    assert torch.equal(payload_collective_inputs[0], routed)
    assert dispatched is received
    assert torch.equal(local_counts, torch.tensor([[5, 6], [7, 8]]))
    assert state.input_splits == [3, 7]
    assert state.output_splits == [11, 15]


def test_a2a_combine_and_backward_use_canonical_rank_block_order(
    monkeypatch,
) -> None:
    state = executors._ExpertA2AState(
        participants=(0, 1),
        shard_devices=(0, 1),
        num_global_routes=4,
        route_partitions=((0, 2), (2, 4)),
        input_splits=[3, 7],
        output_splits=[11, 15],
    )
    executor = DagExecutor(
        runtime=SimpleNamespace(actor_group_routed_group="ep"),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )
    calls = []

    def fake_a2a(tensor, output_splits, input_splits, _group):
        calls.append((tensor.clone(), output_splits, input_splits))
        if output_splits == [3, 7]:
            return torch.cat((torch.zeros(3, 2), torch.ones(7, 2)))
        return torch.cat((torch.zeros(11, 2), torch.ones(15, 2)))

    monkeypatch.setattr(executors, "all_to_all_single_autograd", fake_a2a)
    model_source_major = torch.cat(
        (torch.full((11, 2), 0.0), torch.full((15, 2), 1.0))
    )
    combined = executor._combine_expert_a2a(state, model_source_major)
    assert torch.equal(
        calls[0][0],
        model_source_major,
    )
    assert torch.equal(combined, torch.cat((torch.zeros(3, 2), torch.ones(7, 2))))

    original_route_major_grad = torch.cat(
        (torch.full((3, 2), 0.0), torch.full((7, 2), 1.0))
    )
    dispatched_grad = executor._dispatch_expert_a2a_gradient(
        state, original_route_major_grad
    )
    assert torch.equal(
        calls[1][0],
        original_route_major_grad,
    )
    assert torch.equal(
        dispatched_grad,
        torch.cat((torch.zeros(11, 2), torch.ones(15, 2))),
    )


def test_a2a_dispatch_accepts_empty_payload_and_zero_route_counts() -> None:
    state = executors._ExpertA2AState(
        participants=(0,),
        shard_devices=(0,),
        num_global_routes=3,
        route_partitions=((0, 3),),
    )
    payload = torch.empty(0, 5)
    counts = torch.zeros((1, 3), dtype=torch.int64)
    executor = DagExecutor(
        runtime=SimpleNamespace(actor_group_routed_group="ep"),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )

    received, received_counts = executor._dispatch_expert_a2a(
        state, payload, counts
    )

    assert received.shape == (0, 5)
    assert torch.equal(received_counts, counts)
    assert state.input_splits == [0]
    assert state.output_splits == [0]


def test_a2a_dispatch_state_isolated_by_routed_microbatch_uid(monkeypatch) -> None:
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())
    executor = DagExecutor(
        runtime=SimpleNamespace(actor_group_routed_group="ep", global_rank=0),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )

    for uid, counts in (
        ("expert-mb0", torch.tensor([[1, 0]])),
        ("expert-mb1", torch.tensor([[2, 1]])),
    ):
        node = SimpleNamespace(
            uid=f"dispatch-{uid}",
            task_type=TaskType.FWD_A2A,
            node_meta={
                "direction": "incoming",
                "expert_fwd_uid": uid,
                "participants": [0],
                **_route_meta([0], 2),
            },
        )
        executor._all_to_all_ep_boundary(
            node,
            torch.zeros(int(counts.sum()), 3),
            stream=object(),
            routing_counts=counts,
        )

    first = executor.expert_a2a_states["expert-mb0"]
    second = executor.expert_a2a_states["expert-mb1"]
    assert first is not second
    assert first.input_splits == [1]
    assert second.input_splits == [3]


@pytest.mark.parametrize(
    ("payload", "counts", "error", "message"),
    [
        (
            torch.randn(2, 2, 3),
            torch.tensor([[1, 1]]),
            ValueError,
            r"shape \[tokens, hidden\]",
        ),
        (
            torch.randn(2, 3),
            torch.tensor([[1.0, 1.0]]),
            TypeError,
            "integer dtype",
        ),
        (
            torch.randn(2, 3),
            torch.tensor([[3, -1]]),
            ValueError,
            "negative counts",
        ),
        (
            torch.randn(2, 3),
            torch.tensor([[1, 2]]),
            ValueError,
            "counts sum",
        ),
    ],
)
def test_a2a_dispatch_rejects_invalid_routed_layout(
    payload, counts, error, message
) -> None:
    executor = DagExecutor(
        runtime=SimpleNamespace(actor_group_routed_group="ep"),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )

    with pytest.raises(error, match=message):
        executor._dispatch_expert_a2a(
            executors._ExpertA2AState(
                participants=(0,),
                shard_devices=(0,),
                num_global_routes=2,
                route_partitions=((0, 2),),
            ),
            payload,
            counts,
        )


def _routed_state(**overrides):
    values = {
        "source_devices": (0, 2),
        "destination_devices": (1,),
        "shard_devices": (1,),
        "num_global_routes": 4,
        "route_partitions": ((0, 4),),
        "participants": (0, 1, 2),
        "layout_group": "layout",
        "m_to_n_group": "m-to-n",
        "n_to_m_group": "n-to-m",
        "input_splits": [0, 4, 0],
        "output_splits": [0, 0, 0],
        "payload_shape": (4, 3),
        "payload_dtype": torch.float32,
    }
    values.update(overrides)
    return executors._RoutedCrossGroupState(**values)


def test_dense_n_to_m_destination_needs_splits_but_not_expert_permutation(
    monkeypatch,
) -> None:
    state = _routed_state()
    runtime = SimpleNamespace(device="cpu", global_rank=0)
    executor = DagExecutor(
        runtime=runtime,
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    node = SimpleNamespace(
        uid="combine-destination",
        task_type=TaskType.FWD_N_TO_M,
        node_meta={
            "direction": "outgoing",
            "role": "destination",
            "expert_fwd_uid": "expert",
            "a2a_tensor_idx": 3,
        },
    )
    received = torch.randn(4, 3)
    calls = []

    def fake_a2a(tensor, output_splits, input_splits, group):
        calls.append((tensor, output_splits, input_splits, group))
        return received

    monkeypatch.setattr(executors, "all_to_all_single_autograd", fake_a2a)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    result = executor._routed_cross_group_boundary(node, None, object())

    assert result is received
    assert calls[0][0].shape == (0, 3)
    assert calls[0][1:] == ([0, 4, 0], [0, 0, 0], "n-to-m")


def test_cross_group_n_to_m_uses_canonical_rank_block_order(monkeypatch) -> None:
    state = _routed_state(
        source_devices=(0, 2),
        destination_devices=(1, 3),
        shard_devices=(1, 3),
        num_global_routes=4,
        route_partitions=((0, 2), (2, 4)),
        participants=(0, 1, 2, 3),
        input_splits=[0, 3, 0, 7],
        output_splits=[5, 0, 6, 0],
        payload_shape=(11, 2),
        payload_dtype=torch.float32,
    )
    executor = DagExecutor(
        runtime=SimpleNamespace(device="cpu", global_rank=2),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    node = SimpleNamespace(
        uid="combine-source",
        task_type=TaskType.FWD_N_TO_M,
        node_meta={
            "direction": "outgoing",
            "role": "source",
            "expert_fwd_uid": "expert",
            "a2a_tensor_idx": 0,
        },
    )
    source_major = torch.cat((torch.zeros(5, 2), torch.full((6, 2), 2.0)))
    communicator_output = torch.cat((torch.ones(3, 2), torch.full((7, 2), 3.0)))
    calls = []

    def fake_a2a(tensor, output_splits, input_splits, group):
        calls.append((tensor.clone(), output_splits, input_splits, group))
        return communicator_output

    monkeypatch.setattr(executors, "all_to_all_single_autograd", fake_a2a)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    result = executor._routed_cross_group_boundary(node, source_major, object())

    assert result is None
    assert torch.equal(calls[0][0], source_major)
    assert calls[0][1:] == ([0, 3, 0, 7], [5, 0, 6, 0], "n-to-m")

    destination_node = SimpleNamespace(
        uid="combine-destination",
        task_type=TaskType.FWD_N_TO_M,
        node_meta={
            "direction": "outgoing",
            "role": "destination",
            "expert_fwd_uid": "expert",
            "a2a_tensor_idx": 0,
        },
    )
    executor.runtime.global_rank = 1
    result = executor._routed_cross_group_boundary(
        destination_node, None, object()
    )
    assert result is communicator_output


def test_m_to_n_destination_derives_source_major_splits_and_local_counts(
    monkeypatch,
) -> None:
    state = executors._RoutedCrossGroupState(
        source_devices=(0, 2),
        destination_devices=(1, 3),
        shard_devices=(1, 3),
        num_global_routes=4,
        route_partitions=((0, 2), (2, 4)),
        participants=(0, 1, 2, 3),
        layout_group="layout",
        m_to_n_group="m-to-n",
        n_to_m_group="n-to-m",
    )
    runtime = SimpleNamespace(device="cpu", global_rank=1)
    stages = SimpleNamespace(
        bucket=lambda _key: SimpleNamespace(
            forward_input_meta=[
                ((10, 3), torch.float32, True),
                ((1, 4), torch.int64, False),
            ]
        )
    )
    executor = DagExecutor(
        runtime=runtime,
        stages=stages,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    node = SimpleNamespace(
        uid="dispatch-destination",
        task_type=TaskType.FWD_M_TO_N,
        node_meta={
            "direction": "incoming",
            "role": "destination",
            "expert_fwd_uid": "expert",
            "bucket_key": "expert",
            "a2a_tensor_idx": 0,
            "routing_counts_idx": 1,
        },
    )
    gathered_counts = torch.tensor(
        [
            [1, 2, 3, 4],
            [0, 0, 0, 0],
            [5, 6, 7, 8],
            [0, 0, 0, 0],
        ]
    )
    received = torch.cat((torch.zeros(3, 3), torch.full((11, 3), 2.0)))
    calls = []

    layout_calls = []

    def fake_all_gather(output, _counts, *, group, async_op):
        output.copy_(gathered_counts)
        layout_calls.append((group, async_op))
        return SimpleNamespace(wait=lambda: None)

    monkeypatch.setattr(executors.dist, "all_gather_into_tensor", fake_all_gather)

    def fake_a2a(tensor, output_splits, input_splits, group):
        calls.append((tensor, output_splits, input_splits, group))
        return received

    monkeypatch.setattr(executors, "all_to_all_single_autograd", fake_a2a)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    routed, local_counts = executor._routed_cross_group_boundary(
        node, None, object()
    )

    assert routed is received
    assert torch.equal(local_counts, torch.tensor([[1, 2], [5, 6]]))
    assert state.input_splits == [0, 0, 0, 0]
    assert state.output_splits == [3, 0, 11, 0]
    assert layout_calls == [("layout", True)]
    assert calls[0][0].shape == (0, 3)
    assert calls[0][1:] == ([3, 0, 11, 0], [0, 0, 0, 0], "m-to-n")


def test_m_to_n_source_retains_layout_work_without_waiting(monkeypatch) -> None:
    state = executors._RoutedCrossGroupState(
        source_devices=(0, 2),
        destination_devices=(1, 3),
        shard_devices=(1, 3),
        num_global_routes=4,
        route_partitions=((0, 2), (2, 4)),
        participants=(0, 1, 2, 3),
        layout_group="layout",
        m_to_n_group="m-to-n",
        n_to_m_group="n-to-m",
    )
    executor = DagExecutor(
        runtime=SimpleNamespace(device="cpu", global_rank=0),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    node = SimpleNamespace(
        uid="dispatch-source",
        node_meta={"role": "source", "routing_counts_idx": 1},
    )
    local_counts = torch.tensor([[4, 1, 2, 3]])
    waited = []
    work = SimpleNamespace(wait=lambda: waited.append(True))

    def fake_all_gather(output, _counts, *, group, async_op):
        assert group == "layout"
        assert async_op
        output.zero_()
        return work

    monkeypatch.setattr(executors.dist, "all_gather_into_tensor", fake_all_gather)

    returned_counts, grouped_counts = (
        executor._initialize_routed_cross_group_layout(
            node,
            state,
            local_counts,
        )
    )

    assert returned_counts is local_counts
    assert grouped_counts is None
    assert state.input_splits == [0, 5, 0, 5]
    assert state.layout_work is work
    assert state.layout_output is not None
    assert waited == []


def test_m_to_n_source_uses_canonical_shard_and_collective_order(
    monkeypatch,
) -> None:
    state = executors._RoutedCrossGroupState(
        source_devices=(0,),
        destination_devices=(1, 3),
        shard_devices=(1, 3),
        num_global_routes=4,
        route_partitions=((0, 2), (2, 4)),
        participants=(0, 1, 3),
        layout_group="layout",
        m_to_n_group="m-to-n",
        n_to_m_group="n-to-m",
    )
    executor = DagExecutor(
        runtime=SimpleNamespace(device="cpu", global_rank=0),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    node = SimpleNamespace(
        uid="dispatch-source",
        task_type=TaskType.FWD_M_TO_N,
        node_meta={
            "direction": "incoming",
            "role": "source",
            "expert_fwd_uid": "expert",
            "a2a_tensor_idx": 0,
            "routing_counts_idx": 1,
        },
    )
    counts = torch.tensor([[1, 2, 3, 4]])
    payload = torch.arange(30, dtype=torch.float32).view(10, 3)
    calls = []

    def fake_all_gather(output, _counts, *, group, async_op):
        output.zero_()
        return SimpleNamespace(wait=lambda: None)

    def fake_a2a(tensor, output_splits, input_splits, group):
        calls.append((tensor.clone(), output_splits, input_splits, group))
        return tensor.new_empty((0, tensor.shape[1]))

    monkeypatch.setattr(executors.dist, "all_gather_into_tensor", fake_all_gather)
    monkeypatch.setattr(executors, "all_to_all_single_autograd", fake_a2a)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    result = executor._routed_cross_group_boundary(
        node, payload, object(), routing_counts=counts
    )

    assert result is None
    assert state.input_splits == [0, 3, 7]
    assert torch.equal(calls[0][0], payload)
    assert calls[0][1:] == ([0, 0, 0], [0, 3, 7], "m-to-n")


def test_cross_group_forward_and_backward_rejoin_dense_passthroughs(
    monkeypatch,
) -> None:
    dense_forward = [
        torch.randn(2, 3, requires_grad=True),
        torch.randn(2, 2),
        torch.ones(2, dtype=torch.int64),
        torch.randn(4, 3, requires_grad=True),
        torch.ones(8, dtype=torch.int64),
    ]
    dense_backward = {
        "inp_grads": [
            torch.randn(2, 3),
            torch.randn(2, 2),
            None,
            torch.randn(4, 3),
        ]
    }
    state = _routed_state(
        retained_forward_buffer={"detached_outs": dense_forward},
        retained_backward_buffer=dense_backward,
    )
    buffers = SimpleNamespace(task={}, release=Mock())
    events = SimpleNamespace(a2a={})
    executor = DagExecutor(
        runtime=SimpleNamespace(),
        stages=None,
        buffers=buffers,
        events=events,
        params=None,
        communication=None,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    common_meta = {
        "direction": "outgoing",
        "role": "destination",
        "expert_fwd_uid": "expert",
        "expert_input_arity": 5,
        "expert_output_arity": 4,
        "a2a_tensor_idx": 3,
    }
    fwd_node = SimpleNamespace(
        uid="fwd-combine",
        task_type=TaskType.FWD_N_TO_M,
        node_meta=common_meta,
    )
    bwd_node = SimpleNamespace(
        uid="bwd-combine",
        task_type=TaskType.BWD_N_TO_M,
        node_meta=common_meta,
    )
    combined = torch.randn(4, 3)
    combined_grad = torch.randn(4, 3)
    results = iter((combined, combined_grad))
    monkeypatch.setattr(
        executor,
        "_routed_cross_group_boundary",
        lambda *_args, **_kwargs: next(results),
    )

    class FakeEvent:
        def record(self, _stream):
            pass

    monkeypatch.setattr(executors.torch.cuda, "Event", FakeEvent)

    executor._run_forward_routed_cross_group(fwd_node, object())
    fwd_inputs = buffers.task[fwd_node.uid]["detached_outs"]
    assert len(fwd_inputs) == 4
    assert all(
        actual is expected
        for actual, expected in zip(fwd_inputs[:3], dense_forward[:3])
    )
    assert fwd_inputs[3] is combined
    assert all(
        actual is expected
        for actual, expected in zip(
            buffers.task[fwd_node.uid]["send_output"], fwd_inputs
        )
    )
    assert state.retained_forward_buffer is None

    executor._run_backward_routed_cross_group(bwd_node, object())
    bwd_inputs = buffers.task[bwd_node.uid]["inp_grads"]
    assert len(bwd_inputs) == 5
    assert all(
        actual is expected
        for actual, expected in zip(bwd_inputs[:3], dense_backward["inp_grads"][:3])
    )
    assert bwd_inputs[3] is combined_grad
    assert bwd_inputs[4] is None
    assert state.retained_backward_buffer is None


def test_backward_dense_relay_reconstructs_full_gradient_payload() -> None:
    grad_a = torch.randn(2, 3)
    grad_b = torch.randn(4, 3)
    shape_meta = [(grad_a.shape, grad_a.dtype), (grad_b.shape, grad_b.dtype)]
    buffers = SimpleNamespace(
        task={("shape_ref", "dense-bucket", "dense-fwd"): shape_meta}
    )
    communication = Mock()
    communication.recv_bwd.return_value = [grad_a, grad_b]
    executor = DagExecutor(
        runtime=None,
        stages=None,
        buffers=buffers,
        events=None,
        params=None,
        communication=communication,
        compute=None,
        logger=None,
    )
    stream = object()
    node = SimpleNamespace(
        uid="backward-relay-recv",
        peer_pp_rank=0,
        node_meta={
            "dense_relay": True,
            "direction": "backward",
            "payload_arity": 4,
            "backward_tensor_indices": [0, 3],
            "shape_reference_bucket_key": "dense-bucket",
            "shape_reference_fwd_uid": "dense-fwd",
        },
    )

    result = executor._recv_backward_dense_relay(node, stream)

    assert result["inp_grads"][0] is grad_a
    assert result["inp_grads"][1:3] == [None, None]
    assert result["inp_grads"][3] is grad_b
    communication.recv_bwd.assert_called_once_with(
        shape_meta,
        0,
        stream=stream,
        task=node,
    )


def test_backward_m_to_n_consumes_dense_relay_receive(monkeypatch) -> None:
    routed_grad = torch.randn(4, 3)
    relay_recv = SimpleNamespace(uid="relay-recv", task_type=TaskType.RECV)
    buffers = SimpleNamespace(
        task={relay_recv.uid: {"inp_grads": [None, routed_grad]}},
        release=Mock(),
    )
    events = SimpleNamespace(a2a={})
    state = _routed_state()
    executor = DagExecutor(
        runtime=None,
        stages=None,
        buffers=buffers,
        events=events,
        params=None,
        communication=None,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    node = SimpleNamespace(
        uid="backward-dispatch-source",
        task_type=TaskType.BWD_M_TO_N,
        data_preds=[relay_recv],
        node_meta={
            "direction": "incoming",
            "role": "source",
            "expert_fwd_uid": "expert",
            "a2a_tensor_idx": 1,
        },
    )
    boundary = Mock(return_value=None)
    monkeypatch.setattr(executor, "_routed_cross_group_boundary", boundary)

    class FakeEvent:
        def record(self, _stream):
            pass

    monkeypatch.setattr(executors.torch.cuda, "Event", FakeEvent)
    stream = object()

    executor._run_backward_routed_cross_group(node, stream)

    boundary.assert_called_once_with(node, routed_grad, stream)
    buffers.release.assert_called_once_with(relay_recv.uid)
    assert state.retained_backward_buffer is not None
    assert state.retained_backward_buffer["inp_grads"][1] is routed_grad
