from types import SimpleNamespace
from contextlib import contextmanager, nullcontext
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
    work = Mock()
    isend = Mock(return_value=work)
    monkeypatch.setattr(executors.dist, "isend", isend)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    launch = communication.send(tensor, 1, object(), task=task)

    runtime.p2p_process_group_for_task.assert_called_once_with(task)
    isend.assert_called_once_with(tensor, dst=1, group="pair-group")
    work.wait.assert_called_once_with()
    assert launch.result is None
    assert launch.works == (work,)
    assert launch.tensors == (tensor,)




def test_pending_p2p_lifetime_is_reaped_only_after_event_completion() -> None:
    communication = CommunicationExecutor(
        SimpleNamespace(),
        stages=None,
        logger=None,
    )
    work = Mock()
    tensor = Mock()
    launch = executors._P2PLaunch(None, (work,), (tensor,))
    event = Mock()
    event.query.return_value = False

    communication.retain_p2p_launch(launch, event)
    communication.reap_completed_p2p()

    assert communication.pending_p2p == [(event, launch)]
    work.wait.assert_not_called()

    event.query.return_value = True
    communication.reap_completed_p2p()

    assert communication.pending_p2p == []
    work.wait.assert_not_called()






def test_replicated_gradient_reduce_scatter_averages_before_accumulation(
    monkeypatch,
) -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    reduced = Mock()
    bucket = SimpleNamespace(
        trainable_param_idxs=[0],
        forward_args=[parameter],
        param_shard_info=(0, 2, 4),
        flat_grads=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        reduce_scatter_grads=reduced,
    )
    stages = SimpleNamespace(
        get_bucket=lambda _ubid: bucket,
        bucket=lambda _ubid: bucket,
        grad_sharded_ubids={"bucket"},
    )
    allocation_stream = object()
    default_stream = Mock(return_value=allocation_stream)
    runtime = SimpleNamespace(
        actor_group_size=2,
        actor_group_sync_group="replicas",
        default_stream=default_stream,
    )
    communication = CommunicationExecutor(runtime, stages, logger=None)

    scratch = Mock()

    def fake_reduce_scatter(output, _input, *, group):
        assert group == "replicas"
        assert output is scratch

    monkeypatch.setattr(
        executors.dist, "reduce_scatter_tensor", fake_reduce_scatter
    )
    stream_context = Mock(side_effect=lambda _stream: nullcontext())
    monkeypatch.setattr(executors.torch.cuda, "stream", stream_context)
    monkeypatch.setattr(executors.torch, "empty_like", Mock(return_value=scratch))

    collective_stream = object()
    transferred = communication.reduce_scatter("bucket", collective_stream)

    default_stream.assert_called_once_with()
    assert stream_context.call_args_list == [
        call(allocation_stream),
        call(collective_stream),
    ]
    scratch.record_stream.assert_called_once_with(collective_stream)
    scratch.div_.assert_called_once_with(2)
    reduced.add_.assert_called_once_with(scratch)
    assert transferred == bucket.flat_grads.numel() * bucket.flat_grads.element_size()




def test_split_backward_uses_captured_groups_directly() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 4, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(4, 5))
    output = x @ weight
    output_grad = torch.randn_like(output)
    expected_x_grad, expected_weight_grad = torch.autograd.grad(
        output,
        (x, weight),
        output_grad,
    )

    x = x.detach().requires_grad_(True)
    weight = torch.nn.Parameter(weight.detach())
    outputs = [x @ weight]
    compute = ComputeExecutor(runtime=None, stages=None, logger=None)

    _, param_groups = compute.bucket_backward_input(
        outputs,
        [output_grad],
        [x],
        [weight],
    )

    assert outputs[0].grad_fn is None
    assert torch.allclose(x.grad, expected_x_grad)
    assert weight.grad is None

    compute.bucket_backward_weight([weight], param_groups)

    assert torch.allclose(weight.grad, expected_weight_grad)






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

    def fake_count_a2a(output, tensor, *_args, **_kwargs):
        count_collective_inputs.append(tensor.clone())
        output.copy_(torch.tensor([5, 6, 7, 8]))

    monkeypatch.setattr(executors.dist, "all_to_all_single", fake_count_a2a)

    def fake_payload_a2a(tensor, *_args, **_kwargs):
        payload_collective_inputs.append(tensor.clone())
        return received

    monkeypatch.setattr(
        executors,
        "_stream_ordered_all_to_all",
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
        "dispatch_devices": (0, 2),
        "routed_devices": (1,),
        "combine_devices": (0, 2),
        "shard_devices": (1,),
        "num_global_routes": 4,
        "route_partitions": ((0, 4),),
        "dispatch_participants": (0, 1, 2),
        "combine_participants": (0, 1, 2),
        "layout_group": "layout",
        "dispatch_m_to_n_group": "dispatch-m-to-n",
        "dispatch_n_to_m_group": "dispatch-n-to-m",
        "combine_m_to_n_group": "combine-m-to-n",
        "combine_n_to_m_group": "combine-n-to-m",
        "dispatch_input_splits": [0, 4, 0],
        "routed_dispatch_output_splits": [0, 0, 0],
        "routed_combine_input_splits": [0, 0, 0],
        "combine_output_splits": [0, 4, 0],
        "combine_counts": torch.tensor([[1, 1, 1, 1]]),
        "payload_shape": (4, 3),
        "payload_dtype": torch.float32,
        "routing_counts_dtype": torch.int64,
    }
    values.update(overrides)
    return executors._RoutedCrossGroupState(**values)


def _routed_node_meta(**overrides):
    values = {
        "expert_fwd_uid": "expert",
        "dispatch_devices": [0, 2],
        "routed_devices": [1],
        "combine_devices": [0, 2],
        **_route_meta([1], 4),
    }
    values.update(overrides)
    return values






def test_m_to_n_destination_derives_source_major_splits_and_local_counts(
    monkeypatch,
) -> None:
    state = executors._RoutedCrossGroupState(
        dispatch_devices=(0, 2),
        routed_devices=(1, 3),
        combine_devices=(0, 2),
        shard_devices=(1, 3),
        num_global_routes=4,
        route_partitions=((0, 2), (2, 4)),
        dispatch_participants=(0, 1, 2, 3),
        combine_participants=(0, 1, 2, 3),
        layout_group="layout",
        dispatch_m_to_n_group="dispatch-m-to-n",
        dispatch_n_to_m_group="dispatch-n-to-m",
        combine_m_to_n_group="combine-m-to-n",
        combine_n_to_m_group="combine-n-to-m",
        payload_dtype=torch.float32,
        routing_counts_dtype=torch.int64,
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
        node_meta=_routed_node_meta(
            direction="incoming",
            role="destination",
            bucket_key="expert",
            a2a_tensor_idx=0,
            routing_counts_idx=1,
            routed_devices=[1, 3],
            **_route_meta([1, 3], 4),
        ),
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

    def fake_all_gather(output, _counts, *, group, **kwargs):
        if tuple(output.shape) == tuple(gathered_counts.shape):
            output.copy_(gathered_counts)
        else:
            output.copy_(
                torch.tensor([[10, 3], [0, 0], [20, 3], [0, 0]])
            )
        layout_calls.append((group, kwargs))

    monkeypatch.setattr(executors.dist, "all_gather_into_tensor", fake_all_gather)

    def fake_a2a(tensor, output_splits, input_splits, group):
        calls.append((tensor, output_splits, input_splits, group))
        return received

    monkeypatch.setattr(executors, "_stream_ordered_all_to_all", fake_a2a)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    routed, local_counts = executor._routed_cross_group_boundary(
        node, None, object()
    )

    assert routed is received
    assert torch.equal(local_counts, torch.tensor([[1, 2], [5, 6]]))
    assert state.dispatch_input_splits == [0, 0, 0, 0]
    assert state.routed_dispatch_output_splits == [3, 0, 11, 0]
    assert layout_calls == [("layout", {}), ("layout", {})]
    assert calls[0][0].shape == (0, 3)
    assert calls[0][1:] == (
        [3, 0, 11, 0],
        [0, 0, 0, 0],
        "dispatch-m-to-n",
    )






def _dense_state_meta(direction: str) -> dict:
    return _routed_node_meta(
        dense_state_transfer=True,
        direction=direction,
        routing_counts_idx=1,
        routing_counts_schema={
            "rank": 2,
            "shape": [1, 4],
            "dtype": "torch.int64",
            "requires_grad": False,
        },
        model_tensor_indices=[0, 2],
        destination_input_indices=[0, 1],
        model_value_schemas=[
            {
                "rank": 2,
                "shape": [None, 3],
                "dtype": "torch.float32",
                "requires_grad": True,
            },
            {
                "rank": 1,
                "shape": [None],
                "dtype": "torch.int64",
                "requires_grad": False,
            },
        ],
        differentiable_model_indices=[0],
        dispatch_payload_idx=3,
        dispatch_output_arity=4,
    )


def test_direct_dense_state_forward_envelope_transfers_counts_and_bypasses() -> None:
    bypass = torch.randn(2, 3, requires_grad=True)
    counts = torch.tensor([[1, 1, 1, 1]])
    indices = torch.tensor([2, 0])
    payload = torch.randn(4, 3, requires_grad=True)
    producer = SimpleNamespace(uid="dispatch", task_type=TaskType.FWD)
    buffers = SimpleNamespace(
        task={producer.uid: {"detached_outs": [bypass, counts, indices, payload]}},
        release=Mock(),
    )
    communication = Mock()
    launch = executors._P2PLaunch(None, (), ())
    communication.send_sized_tensors.return_value = launch
    state = _routed_state()
    executor = DagExecutor(
        runtime=None,
        stages=None,
        buffers=buffers,
        events=None,
        params=None,
        communication=communication,
        compute=None,
        logger=None,
        routed_cross_group_states={"expert": state},
    )
    send = SimpleNamespace(
        uid="dense-state-send",
        task_type=TaskType.SEND,
        peer_pp_rank=1,
        data_preds=[producer],
        node_meta=_dense_state_meta("forward"),
    )

    assert executor._send_dense_state(send, object()) is launch

    sent, peer, stream = communication.send_sized_tensors.call_args.args[:3]
    assert sent[0] is counts
    assert torch.equal(sent[1], torch.tensor([4, 3]))
    assert sent[2:] == [bypass, indices]
    assert peer == 1
    assert stream is not None
    assert communication.send_sized_tensors.call_args.kwargs == {"task": send}
    assert state.dispatch_counts is counts
    buffers.release.assert_called_once_with(producer.uid)

    communication.recv_sized_tensors.reset_mock()
    recv_transport_launch = executors._P2PLaunch(
        [counts, torch.tensor([4, 3]), bypass, indices],
        (Mock(),),
        (counts, bypass, indices),
    )
    communication.recv_sized_tensors.return_value = recv_transport_launch
    recv = SimpleNamespace(
        uid="dense-state-recv",
        task_type=TaskType.RECV,
        peer_pp_rank=0,
        node_meta=_dense_state_meta("forward"),
    )
    recv_launch = executor._recv_dense_state(recv, object())
    result = recv_launch.result

    assert state.combine_counts is counts
    assert recv_launch.works == recv_transport_launch.works
    assert recv_launch.tensors == recv_transport_launch.tensors
    assert result["routing_counts"] is counts
    assert result["runtime_inputs"] == {0: bypass, 1: indices}






@pytest.mark.parametrize(
    ("task_type", "output_splits", "input_splits", "group"),
    [
        (
            TaskType.BWD_M_TO_N,
            [0, 0, 0],
            [0, 4, 0],
            "combine-m-to-n",
        ),
        (
            TaskType.BWD_N_TO_M,
            [0, 4, 0],
            [0, 0, 0],
            "dispatch-n-to-m",
        ),
    ],
)
def test_cross_group_backward_reverses_the_matching_forward_boundary(
    monkeypatch, task_type, output_splits, input_splits, group
) -> None:
    state = _routed_state()
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
        uid="backward-source",
        task_type=task_type,
        node_meta=_routed_node_meta(
            direction="incoming",
            role="source",
            a2a_tensor_idx=0,
        ),
    )
    tensor = torch.randn(4, 3)
    calls = []

    def fake_a2a(value, recv_splits, send_splits, process_group):
        calls.append((value, recv_splits, send_splits, process_group))
        return value.new_empty((0, 3))

    monkeypatch.setattr(executors, "_stream_ordered_all_to_all", fake_a2a)
    monkeypatch.setattr(executors.torch.cuda, "stream", lambda _stream: nullcontext())

    assert executor._routed_cross_group_boundary(node, tensor, object()) is None
    assert calls[0][0] is tensor
    assert calls[0][1:] == (output_splits, input_splits, group)
