# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
import torch.distributed as dist
from torch.autograd.graph import GradientEdge, Node, get_gradient_edge
from torch.nn import Parameter


P2P_POST_PROFILER_MARKER = "piper_p2p_post"


def _route_partition_metadata(
    meta: dict[str, Any],
    destination_devices: tuple[int, ...],
) -> tuple[int, tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Validate schedule-owned equal contiguous route partitions."""
    try:
        num_global_routes = int(meta["num_global_routes"])
        shard_degree = int(meta["shard_degree"])
        num_local_routes = int(meta["num_local_routes"])
        shard_devices = tuple(int(rank) for rank in meta["shard_devices"])
        raw_partitions = meta["route_partitions"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "routed communication is missing valid schedule route metadata "
            "(num_global_routes, shard_degree, num_local_routes, "
            "shard_devices, route_partitions)"
        ) from error
    if num_global_routes < 1:
        raise ValueError(f"num_global_routes must be positive, got {num_global_routes}")
    if shard_degree < 1:
        raise ValueError(f"shard_degree must be positive, got {shard_degree}")
    if shard_degree != len(shard_devices):
        raise ValueError(
            f"shard_degree={shard_degree} does not match "
            f"shard_devices={shard_devices}"
        )
    if len(shard_devices) != len(set(shard_devices)):
        raise ValueError(f"shard_devices contains duplicates: {shard_devices}")
    if shard_devices != tuple(sorted(shard_devices)):
        raise ValueError(
            "shard_devices must be in ascending global actor-rank order, got "
            f"{shard_devices}"
        )
    if set(shard_devices) != set(destination_devices):
        raise ValueError(
            "shard_devices must identify exactly the routed destinations: "
            f"shard_devices={shard_devices}, destinations={destination_devices}"
        )
    if num_global_routes % shard_degree:
        raise ValueError(
            f"num_global_routes={num_global_routes} must be divisible by "
            f"shard_degree={shard_degree}"
        )
    if not isinstance(raw_partitions, list) or len(raw_partitions) != shard_degree:
        raise ValueError(
            f"route_partitions must contain {shard_degree} entries, got "
            f"{raw_partitions!r}"
        )
    expected_width = num_global_routes // shard_degree
    if num_local_routes != expected_width:
        raise ValueError(
            f"num_local_routes={num_local_routes} does not match the equal "
            f"contiguous partition width={expected_width}"
        )
    partitions: list[tuple[int, int]] = []
    for index, (device, raw) in enumerate(zip(shard_devices, raw_partitions)):
        if not isinstance(raw, dict):
            raise ValueError(f"route partition {index} is not a mapping: {raw!r}")
        try:
            partition_device = int(raw["device"])
            start = int(raw["start"])
            stop = int(raw["stop"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid route partition {index}: {raw!r}") from error
        expected_start = index * expected_width
        expected_stop = expected_start + expected_width
        if (
            partition_device != device
            or start != expected_start
            or stop != expected_stop
        ):
            raise ValueError(
                "route_partitions must assign equal contiguous ranges in "
                "shard_devices order; got "
                f"partition={raw!r}, expected device={device}, "
                f"start={expected_start}, stop={expected_stop}"
            )
        partitions.append((start, stop))
    return num_global_routes, shard_devices, tuple(partitions)


def _validate_routed_payload_schema(
    tensor: torch.Tensor,
    payload_shape: tuple[int, ...] | None,
    payload_dtype: torch.dtype | None,
    *,
    context: str,
) -> None:
    if tensor.ndim != 2:
        raise ValueError(
            f"{context} payload must have shape [rows, hidden], got "
            f"shape={tuple(tensor.shape)}"
        )
    if payload_shape is not None and tuple(tensor.shape[1:]) != payload_shape[1:]:
        raise ValueError(
            f"{context} payload trailing shape={tuple(tensor.shape[1:])} does "
            f"not match dispatch trailing shape={payload_shape[1:]}"
        )
    if payload_dtype is not None and tensor.dtype != payload_dtype:
        raise TypeError(
            f"{context} payload dtype={tensor.dtype} does not match dispatch "
            f"dtype={payload_dtype}"
        )


def _torch_dtype_from_name(name: Any) -> torch.dtype:
    if not isinstance(name, str):
        raise TypeError(f"tensor dtype metadata must be a string, got {name!r}")
    attribute = name.removeprefix("torch.")
    dtype = getattr(torch, attribute, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unknown torch dtype metadata {name!r}")
    return dtype


def _stream_ordered_all_to_all(
    tensor: torch.Tensor,
    output_splits: list[int],
    input_splits: list[int],
    process_group: Any,
) -> torch.Tensor:
    """Launch a payload all-to-all whose completion is visible on this stream."""
    output = torch.empty(
        (sum(output_splits), *tensor.shape[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_to_all_single(
        output,
        tensor,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=process_group,
    )
    return output


from .backward import construct_reverse_graph, get_param_groups, _get_grad_fn_or_grad_acc
from .execution_plan import (
    PlannedOperation,
    build_execution_plan,
)
from .runtime import BufferStore, EventStore, ParamStorage, RuntimeState, StageStore
from .tasks import TaskType


class TimingRecorder(Protocol):
    """Callbacks used to measure payloads without duplicating dispatch logic.

    Node dependency waits happen before ``before_node``.  A recorder can
    synthesize zero duration for ``ORDER_DUMMY`` by making its node callbacks
    no-ops and emitting an explicit zero-valued profile in ``finish``.
    """

    def before_prologue(self, stream: torch.cuda.Stream) -> None: ...

    def after_prologue(self, stream: torch.cuda.Stream) -> None: ...

    def before_node(self, node: Any, stream: torch.cuda.Stream) -> None: ...

    def after_node(self, node: Any, stream: torch.cuda.Stream) -> None: ...

    def finish(self) -> None: ...


StreamOverride = Callable[[Any, torch.cuda.Stream], torch.cuda.Stream]


@dataclass(frozen=True)
class PreparedDagExecution:
    """Invariant state needed to execute one loaded TrainingDAG.

    Building the dependency plan and UID lookup tables is proportional to DAG
    size.  Keep that setup work out of the per-iteration dispatch loop.  A
    remapped execution shares the static plan and node order while retaining a
    separate physical-stream table for its execution/profile policy.
    """

    dag: Any
    sorted_dag_nodes: tuple[Any, ...]
    execution_plan: tuple[PlannedOperation, ...]
    planned_by_uid: dict[str, PlannedOperation]
    physical_stream_by_uid: dict[str, torch.cuda.Stream]
    initial_wait_streams: tuple[torch.cuda.Stream, ...]
    buffer_refcount_template: dict[Any, int]


@dataclass
class _ExpertA2AState:
    participants: tuple[int, ...]
    shard_devices: tuple[int, ...]
    num_global_routes: int
    route_partitions: tuple[tuple[int, int], ...]
    input_splits: list[int] | None = None
    output_splits: list[int] | None = None
    payload_shape: tuple[int, ...] | None = None
    payload_dtype: torch.dtype | None = None

    @property
    def ep_degree(self) -> int:
        return len(self.shard_devices)


@dataclass
class _RoutedCrossGroupState:
    """Actor-local transport state for one D -> R -> C routed lifecycle."""

    dispatch_devices: tuple[int, ...]
    routed_devices: tuple[int, ...]
    combine_devices: tuple[int, ...]
    shard_devices: tuple[int, ...]
    num_global_routes: int
    route_partitions: tuple[tuple[int, int], ...]
    dispatch_participants: tuple[int, ...]
    combine_participants: tuple[int, ...]
    layout_group: Any = None
    dispatch_m_to_n_group: Any = None
    dispatch_n_to_m_group: Any = None
    combine_m_to_n_group: Any = None
    combine_n_to_m_group: Any = None
    dispatch_input_splits: list[int] | None = None
    routed_dispatch_output_splits: list[int] | None = None
    routed_combine_input_splits: list[int] | None = None
    combine_output_splits: list[int] | None = None
    dispatch_counts: torch.Tensor | None = None
    routed_counts: torch.Tensor | None = None
    combine_counts: torch.Tensor | None = None
    payload_shape: tuple[int, ...] | None = None
    payload_dtype: torch.dtype | None = None
    routing_counts_dtype: torch.dtype | None = None


@dataclass
class _P2PLaunch:
    """State that must outlive a host-nonblocking P2P launch."""

    result: Any
    works: tuple[Any, ...]
    tensors: tuple[torch.Tensor, ...]


@dataclass
class CommunicationExecutor:
    """Actor-local communication operations used by the DAG dispatcher."""

    runtime: RuntimeState
    stages: StageStore
    logger: Any
    pending_p2p: list[tuple[Any, _P2PLaunch]] = field(
        default_factory=list,
        init=False,
    )

    def __post_init__(self) -> None:
        blocking_wait = os.environ.get("TORCH_NCCL_BLOCKING_WAIT", "")
        if blocking_wait.strip().lower() in {"1", "true", "yes", "on"}:
            raise RuntimeError(
                "Piper's nonblocking P2P dispatcher is incompatible with "
                "TORCH_NCCL_BLOCKING_WAIT=1"
            )

    def retain_p2p_launch(
        self,
        launch: _P2PLaunch,
        completion_event: Any,
    ) -> None:
        self.reap_completed_p2p()
        self.pending_p2p.append((completion_event, launch))

    def reap_completed_p2p(self) -> None:
        self.pending_p2p = [
            (event, launch)
            for event, launch in self.pending_p2p
            if not event.query()
        ]

    @staticmethod
    def _record_p2p_tensor_stream(
        tensor: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> None:
        if tensor.is_cuda:
            tensor.record_stream(stream)

    def _record_p2p_post(self) -> None:
        """Mark entry to the first P2P call made by one DAG node.

        Entering ``isend``/``irecv`` makes the compound node eligible to match
        its peer before that API necessarily returns.  The node's aggregate
        CUDA envelope covers any additional P2P calls it makes.
        """

        if not getattr(self.runtime, "pytorch_profiler_enabled", False):
            return
        with torch.profiler.record_function(P2P_POST_PROFILER_MARKER):
            pass

    def send(
        self,
        send_data: Any,
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any | None = None,
    ) -> _P2PLaunch:
        global_dst_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        with torch.cuda.stream(stream):
            tensors = send_data if isinstance(send_data, (list, tuple)) else [send_data]
            if task is None:
                raise ValueError("ActorGroup P2P send requires its DAG task")
            pp_group = self.runtime.p2p_process_group_for_task(task)
            works = []
            if not tensors:
                self._record_p2p_post()
            for tensor_index, tensor in enumerate(tensors):
                self._record_p2p_tensor_stream(tensor, stream)
                if tensor_index == 0:
                    self._record_p2p_post()
                work = dist.isend(
                    tensor,
                    dst=global_dst_rank,
                    group=pp_group,
                )
                work.wait()
                works.append(work)
        return _P2PLaunch(None, tuple(works), tuple(tensors))

    def recv_fwd(
        self,
        recv_ubid: Any,
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any | None = None,
    ) -> _P2PLaunch:
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        buf = [
            torch.empty(
                shape,
                dtype=dtype,
                requires_grad=requires_grad,
                device=self.runtime.device,
            )
            for shape, dtype, requires_grad in self.stages.bucket(recv_ubid).forward_input_meta
        ]
        with torch.cuda.stream(stream):
            if task is None:
                raise ValueError("ActorGroup P2P receive requires its DAG task")
            pp_group = self.runtime.p2p_process_group_for_task(task)
            works = []
            if not buf:
                self._record_p2p_post()
            for tensor_index, tensor in enumerate(buf):
                self._record_p2p_tensor_stream(tensor, stream)
                if tensor_index == 0:
                    self._record_p2p_post()
                work = dist.irecv(
                    tensor,
                    src=global_src_rank,
                    group=pp_group,
                )
                work.wait()
                works.append(work)
        return _P2PLaunch(buf, tuple(works), tuple(buf))

    def recv_bwd(
        self,
        shape_meta: list,
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any | None = None,
    ) -> _P2PLaunch:
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        buf = [
            torch.empty(shape, dtype=dtype, device=self.runtime.device)
            for shape, dtype in shape_meta
        ]
        with torch.cuda.stream(stream):
            if task is None:
                raise ValueError("ActorGroup P2P receive requires its DAG task")
            pp_group = self.runtime.p2p_process_group_for_task(task)
            works = []
            if not buf:
                self._record_p2p_post()
            for tensor_index, tensor in enumerate(buf):
                self._record_p2p_tensor_stream(tensor, stream)
                if tensor_index == 0:
                    self._record_p2p_post()
                work = dist.irecv(
                    tensor,
                    src=global_src_rank,
                    group=pp_group,
                )
                work.wait()
                works.append(work)
        return _P2PLaunch(buf, tuple(works), tuple(buf))

    def send_sized_tensors(
        self,
        tensors: list[torch.Tensor],
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any,
    ) -> _P2PLaunch:
        """Send one combined shape header followed by batched payload P2P."""
        global_dst_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)
        pp_group = self.runtime.p2p_process_group_for_task(task)
        if not tensors:
            self._record_p2p_post()
            return _P2PLaunch(None, (), ())

        payloads = []
        flat_shape: list[int] = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"dense-state payload must contain tensors, got "
                    f"{type(tensor).__name__}"
                )
            flat_shape.extend(int(dim) for dim in tensor.shape)

        works = []
        retained_tensors = []
        shape_values = None
        shape_header = None
        p2p_post_recorded = False
        if flat_shape:
            header_is_cuda = torch.device(self.runtime.device).type == "cuda"
            shape_values = torch.tensor(
                flat_shape,
                dtype=torch.int64,
                device="cpu",
                pin_memory=header_is_cuda,
            )
            shape_header = torch.empty(
                (len(flat_shape),),
                dtype=torch.int64,
                device=self.runtime.device,
            )
        with torch.cuda.stream(stream):
            for tensor in tensors:
                payload = tensor.contiguous()
                self._record_p2p_tensor_stream(payload, stream)
                payloads.append(payload)

            if flat_shape:
                assert shape_values is not None and shape_header is not None
                shape_header.copy_(
                    shape_values,
                    non_blocking=shape_header.is_cuda,
                )
                self._record_p2p_tensor_stream(shape_header, stream)
                self._record_p2p_post()
                p2p_post_recorded = True
                shape_work = dist.isend(
                    shape_header,
                    dst=global_dst_rank,
                    group=pp_group,
                )
                shape_work.wait()
                works.append(shape_work)
                retained_tensors.extend((shape_values, shape_header))

            payload_ops = [
                dist.P2POp(
                    dist.isend,
                    payload,
                    global_dst_rank,
                    pp_group,
                )
                for payload in payloads
            ]
            if not p2p_post_recorded:
                self._record_p2p_post()
            payload_works = dist.batch_isend_irecv(payload_ops)
            for work in payload_works:
                work.wait()
            works.extend(payload_works)
            retained_tensors.extend(payloads)
        return _P2PLaunch(
            None,
            tuple(works),
            tuple(retained_tensors),
        )

    def recv_sized_tensors(
        self,
        schemas: list[dict[str, Any]],
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any,
    ) -> _P2PLaunch:
        """Receive one combined shape header followed by batched payload P2P."""
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)
        pp_group = self.runtime.p2p_process_group_for_task(task)
        if not schemas:
            self._record_p2p_post()
            return _P2PLaunch([], (), ())

        schema_ranks = []
        for schema in schemas:
            if not isinstance(schema, dict):
                raise TypeError(
                    f"dense-state tensor schema must be a mapping, got "
                    f"{schema!r}"
                )
            raw_shape = schema.get("shape")
            rank = schema.get("rank")
            if rank is None and isinstance(raw_shape, (list, tuple)):
                rank = len(raw_shape)
            if (
                isinstance(rank, bool)
                or not isinstance(rank, int)
                or rank < 0
            ):
                raise ValueError(
                    f"dense-state schema has invalid rank={rank!r}"
                )
            schema_ranks.append(rank)

        result: list[torch.Tensor] = []
        works = []
        retained_tensors = []
        p2p_post_recorded = False
        with torch.cuda.stream(stream):
            shape_values: list[int] = []
            shape_header = None
            if sum(schema_ranks):
                shape_header = torch.empty(
                    (sum(schema_ranks),),
                    dtype=torch.int64,
                    device=self.runtime.device,
                )
                self._record_p2p_tensor_stream(shape_header, stream)
                self._record_p2p_post()
                p2p_post_recorded = True
                shape_work = dist.irecv(
                    shape_header,
                    src=global_src_rank,
                    group=pp_group,
                )
                shape_work.wait()
                shape_values = [int(dim) for dim in shape_header.tolist()]
                works.append(shape_work)
                retained_tensors.append(shape_header)

            shape_offset = 0
            for schema, rank in zip(schemas, schema_ranks):
                raw_shape = schema.get("shape")
                shape = tuple(shape_values[shape_offset : shape_offset + rank])
                shape_offset += rank
                if any(dim < 0 for dim in shape):
                    raise ValueError(
                        f"received negative dense-state shape={shape}"
                    )
                if isinstance(raw_shape, (list, tuple)):
                    if len(raw_shape) != rank or any(
                        isinstance(expected, int)
                        and not isinstance(expected, bool)
                        and expected != actual
                        for expected, actual in zip(raw_shape, shape)
                    ):
                        raise ValueError(
                            f"received dense-state shape={shape} does not "
                            f"match captured schema={tuple(raw_shape)}"
                        )
                tensor = torch.empty(
                    shape,
                    dtype=_torch_dtype_from_name(schema.get("dtype")),
                    device=self.runtime.device,
                )
                self._record_p2p_tensor_stream(tensor, stream)
                result.append(tensor)

            payload_ops = [
                dist.P2POp(
                    dist.irecv,
                    tensor,
                    global_src_rank,
                    pp_group,
                )
                for tensor in result
            ]
            if not p2p_post_recorded:
                self._record_p2p_post()
            payload_works = dist.batch_isend_irecv(payload_ops)
            for work in payload_works:
                work.wait()
            for schema, tensor in zip(schemas, result):
                if bool(schema.get("requires_grad")):
                    tensor.requires_grad_(True)
            works.extend(payload_works)
            retained_tensors.extend(result)
        return _P2PLaunch(
            result,
            tuple(works),
            tuple(retained_tensors),
        )

    def all_reduce_grads(self, ubid: Any, stream: torch.cuda.Stream) -> int:
        assert ubid is not None, "all_reduce_grads requires a non-None ubid"
        if not self.has_trainable_params_for_collective(ubid, "all_reduce_grads"):
            return 0
        bucket = self.stages.bucket(ubid)
        grad_tensors = []
        for idx in bucket.trainable_param_idxs:
            param = bucket.forward_args[idx]
            assert param is not None, (
                f"all_reduce_grads: ubid={ubid} idx={idx} param is None"
            )
            assert param.grad is not None, (
                f"all_reduce_grads: ubid={ubid} idx={idx} param.grad is None"
            )
            grad_tensors.append(param.grad)

        total_bytes = sum(grad.numel() * grad.element_size() for grad in grad_tensors)
        replica_degree = int(self.runtime.actor_group_size)
        if replica_degree <= 0:
            raise ValueError(
                f"all_reduce_grads requires a positive replica group width, "
                f"got {replica_degree}"
            )
        sync_group = self.runtime.actor_group_sync_group
        if replica_degree > 1 and sync_group is None:
            raise RuntimeError(
                "all_reduce_grads requires an initialized ActorGroup "
                "synchronization communicator"
            )
        with torch.cuda.stream(stream):
            for grad in grad_tensors:
                dist.all_reduce(
                    grad,
                    group=sync_group,
                )
                if replica_degree > 1:
                    grad.div_(replica_degree)
        return total_bytes

    def reduce_scatter(self, ubid: Any, stream: torch.cuda.Stream) -> int:
        assert ubid is not None, "reduce_scatter requires a non-None ubid"
        if not self.has_trainable_params_for_collective(ubid, "reduce_scatter"):
            return 0
        assert ubid in self.stages.grad_sharded_ubids, (
            f"reduce_scatter: ubid={ubid} is not in grad_sharded_ubids="
            f"{self.stages.grad_sharded_ubids}"
        )
        bucket = self.stages.bucket(ubid)
        assert bucket.param_shard_info is not None, (
            f"reduce_scatter: missing param_shard_info for ubid={ubid}"
        )
        flat_grads = bucket.flat_grads
        rs_out = bucket.reduce_scatter_grads
        assert flat_grads is not None, (
            f"reduce_scatter: missing flat_grads buffer for ubid={ubid}"
        )
        assert rs_out is not None, (
            f"reduce_scatter: missing reduce_scatter_grads buffer for ubid={ubid}"
        )
        replica_degree = int(self.runtime.actor_group_size)
        if replica_degree <= 0:
            raise ValueError(
                f"reduce_scatter requires a positive replica group width, "
                f"got {replica_degree}"
            )
        sync_group = self.runtime.actor_group_sync_group
        if replica_degree > 1 and sync_group is None:
            raise RuntimeError(
                "reduce_scatter requires an initialized ActorGroup "
                "synchronization communicator"
            )
        # Allocate from the compute stream's caching-allocator pool, then mark
        # the scratch tensor's use on the collective stream.  This avoids
        # competing large stream-local caches near device capacity.
        allocation_stream = self.runtime.default_stream()
        with torch.cuda.stream(allocation_stream):
            tmp = torch.empty_like(rs_out)
        with torch.cuda.stream(stream):
            total_bytes = flat_grads.numel() * flat_grads.element_size()
            tmp.record_stream(stream)
            dist.reduce_scatter_tensor(
                tmp,
                flat_grads,
                group=sync_group,
            )
            if replica_degree > 1:
                tmp.div_(replica_degree)
            rs_out.add_(tmp)
            return total_bytes

    def has_trainable_params_for_collective(self, ubid: Any, op_name: str) -> bool:
        bucket = self.stages.get_bucket(ubid)
        if bucket is not None and bucket.trainable_param_idxs:
            return True
        self.logger.debug(
            "%s: skipping collective for ubid=%s because it has no trainable param indices",
            op_name,
            ubid,
        )
        return False


@dataclass
class ComputeExecutor:
    """Actor-local bucket forward/backward execution."""

    runtime: RuntimeState
    stages: StageStore
    logger: Any

    def log_compute_loss_inputs(
        self,
        labels: Any,
        node: Any,
        fwd_key: Any,
        fwd_out: dict,
    ) -> None:
        def _summarize_value(value: Any) -> str:
            if isinstance(value, torch.Tensor):
                return (
                    f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
                    f"requires_grad={value.requires_grad}, device={value.device})"
                )
            if isinstance(value, (list, tuple)):
                return "[" + ", ".join(_summarize_value(v) for v in value) + "]"
            if value is None:
                return "None"
            return type(value).__name__

        self.logger.debug(
            "compute_loss inputs: rank=%s node_uid=%s node_type=%s tag=%s "
            "fwd_key=%s labels=%s out_with_grad=%s pre_detach_outs=%s "
            "detached_outs=%s send_output=%s",
            self.runtime.global_rank,
            getattr(node, "uid", None),
            getattr(getattr(node, "task_type", None), "value", None),
            getattr(node, "tag", None),
            fwd_key,
            _summarize_value(labels),
            _summarize_value(fwd_out.get("out_with_grad")),
            _summarize_value(fwd_out.get("pre_detach_outs")),
            _summarize_value(fwd_out.get("detached_outs")),
            _summarize_value(fwd_out.get("send_output")),
        )

    def forward(self, ubid: Any, input_tensors: Any, compute_stream: torch.cuda.Stream) -> dict:
        bucket = self.stages.bucket(ubid)
        fwd_fn = bucket.forward_fn
        fwd_args = bucket.forward_args
        input_idxs = bucket.input_idxs

        if not isinstance(input_tensors, (list, tuple)):
            input_tensors = [input_tensors]

        for i, tensor in zip(input_idxs, input_tensors):
            if isinstance(tensor, (list, tuple)):
                tensor = tensor[0]
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor = tensor.detach().requires_grad_(True)
            fwd_args[i] = tensor

        fwd_inputs = [fwd_args[i] for i in input_idxs]
        inp_with_grad = [
            t
            for t in fwd_inputs
            if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        with torch.cuda.stream(compute_stream):
            output = fwd_fn(fwd_args)

        for i in input_idxs:
            fwd_args[i] = None

        out_list = list(output) if isinstance(output, tuple) else [output]
        possibly_detached = [
            t.detach().requires_grad_(True)
            if isinstance(t, torch.Tensor) and t.requires_grad
            else t
            for t in out_list
        ]
        out_with_grad = [
            t for t in out_list if isinstance(t, torch.Tensor) and t.requires_grad
        ]

        return {
            "pre_detach_outs": out_list,
            "detached_outs": possibly_detached,
            "out_with_grad": out_with_grad,
            "send_output": output,
            "inp_with_grad": inp_with_grad,
            "fwd_inputs": fwd_inputs,
        }

    def backward(
        self,
        ubid: Any,
        mb_idx: int,
        outputs_or_loss: list,
        upstream_grads: Any,
        pre_detach_outs: Any,
        detached_outs: Any,
        inp_with_grad: Any,
        out_with_grad: Any,
        compute_stream: torch.cuda.Stream,
    ) -> dict | None:
        if pre_detach_outs is None:
            if upstream_grads is not None:
                with torch.cuda.stream(compute_stream):
                    torch.autograd.backward(outputs_or_loss, upstream_grads)
            else:
                with torch.cuda.stream(compute_stream):
                    outputs_or_loss[0].backward()
        else:
            bwd_pairs = [
                (p, d.grad)
                for p, d in zip(pre_detach_outs, detached_outs)
                if (isinstance(d, torch.Tensor) and d.requires_grad and d.grad is not None)
            ]
            assert bwd_pairs, (
                f"BWD ubid={ubid} mb={mb_idx}: detached boundary has no tensors "
                "with a materialized grad"
            )
            if bwd_pairs:
                outputs_bwd = [p for p, _g in bwd_pairs]
                grads_bwd = [g for _p, g in bwd_pairs]
                with torch.cuda.stream(compute_stream):
                    torch.autograd.backward(outputs_bwd, grads_bwd)

        if inp_with_grad:
            output_grads = [t.grad for t in inp_with_grad if t.grad is not None]
            if output_grads:
                return {"send_output": output_grads}
        return None

    @staticmethod
    def grad_with_param_layout(weight: Parameter, grad: torch.Tensor) -> torch.Tensor:
        grad_dtype = (
            weight.grad_dtype
            if getattr(weight, "grad_dtype", None) is not None
            else weight.dtype
        )
        if (
            grad.dtype == grad_dtype
            and grad.device == weight.device
            and grad.layout == weight.layout
            and tuple(grad.stride()) == tuple(weight.stride())
        ):
            return grad
        if weight.layout == torch.strided and grad.layout != torch.strided:
            grad = grad.to_dense()
        out = torch.empty_strided(
            tuple(weight.shape),
            tuple(weight.stride()),
            dtype=grad_dtype,
            device=weight.device,
        )
        out.copy_(grad)
        return out

    def bucket_backward_input(
        self,
        stage_outputs_or_loss: list,
        output_grads: Any,
        input_values: list,
        weights: Any,
    ):
        weights = list(weights)

        if output_grads is None:
            output_grads = [torch.ones_like(o) for o in stage_outputs_or_loss]

        input_values = [inp for inp in input_values if inp.requires_grad]
        if not input_values:
            if any(weight.requires_grad for weight in weights):
                raise RuntimeError(
                    "split backward invariant failed: DAG lowering emitted BWD_I/BWD_W "
                    "for a bucket with trainable weights but no differentiable runtime inputs"
                )
            return (), []

        stage_output_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, stage_outputs_or_loss)))
        stage_input_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, input_values)))
        weight_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, weights)))

        # Python/new-style autograd nodes (including Inductor's compiled
        # backward Function) do not themselves keep their C++ graph alive.
        # get_gradient_edge() creates the ownership token required to reuse
        # captured interior edges after the stage outputs are detached.
        ownership_tokens = tuple(
            edge.ownership_token
            for output in stage_outputs_or_loss
            if isinstance(output, torch.Tensor) and output.requires_grad
            for edge in (get_gradient_edge(output),)
            if edge.ownership_token is not None
        )

        reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
        param_groups = get_param_groups(stage_input_grad_fns, weight_grad_fns, reverse_edges_dict)
        if ownership_tokens:
            for param_group in param_groups:
                param_group["ownership_tokens"] = ownership_tokens
        handles = []
        for param_group in param_groups:
            for i, intermediate in enumerate(param_group["intermediates"]):
                def get_hook(pg, idx):
                    def hook(grad_inputs):
                        if pg.get("grads") is None:
                            pg["grads"] = [None] * len(pg["intermediates"])
                        pg["grads"][idx] = tuple(grad_inputs)
                    return hook
                handles.append(intermediate.register_prehook(get_hook(param_group, i)))

        dinputs = torch.autograd.grad(
            stage_outputs_or_loss,
            inputs=input_values,
            grad_outputs=output_grads,
            retain_graph=True,
            allow_unused=True,
        )
        for inp, dinput in zip(input_values, dinputs):
            if inp.grad is None:
                inp.grad = dinput
            else:
                inp.grad += dinput

        for handle in handles:
            handle.remove()

        active_grad_fns = set(reverse_edges_dict) | set(stage_output_grad_fns)
        uncaptured_groups = []
        for param_group in param_groups:
            if not any(param in active_grad_fns for param in param_group["params"]):
                continue
            captured_grads = param_group.get("grads", [])
            if not param_group["intermediates"] or not any(
                grad is not None
                for grads_tuple in captured_grads
                if grads_tuple is not None
                for grad in grads_tuple
            ):
                uncaptured_groups.append(param_group)
        if uncaptured_groups:
            raise RuntimeError(
                "split backward invariant failed: one or more trainable parameter "
                "groups produced no captured boundary gradients"
            )

        for i, output in enumerate(stage_outputs_or_loss):
            if isinstance(output, torch.Tensor):
                stage_outputs_or_loss[i] = output.detach()

        return dinputs, param_groups

    def bucket_backward_weight(
        self,
        weights: Any,
        param_groups: list,
        ubid: Any | None = None,
        mb_idx: int | None = None,
    ) -> None:
        if not param_groups:
            return

        grad_acc_to_weight: dict[Node, Parameter] = {}

        for weight in weights:
            grad_acc = _get_grad_fn_or_grad_acc(weight)
            grad_acc_to_weight[grad_acc] = weight

        for param_group in param_groups:
            # Keep these tokens referenced until the captured-edge backward
            # below has completed.  Their values are otherwise unused.
            ownership_tokens = param_group.pop("ownership_tokens", ())
            valid_edges: list[GradientEdge] = []
            valid_grad_outputs: list[torch.Tensor] = []

            for grads_tuple, intermediate in zip(
                param_group.get("grads", []), param_group["intermediates"]
            ):
                if grads_tuple is None:
                    continue
                for i, grad in enumerate(grads_tuple):
                    if grad is not None:
                        valid_edges.append(GradientEdge(intermediate, i))
                        valid_grad_outputs.append(grad)

            del param_group["intermediates"]

            if not valid_edges:
                continue

            weight_edges = tuple(GradientEdge(w, 0) for w in param_group["params"])
            if not weight_edges:
                continue

            try:
                dweights = torch.autograd.grad(
                    valid_edges,
                    weight_edges,
                    grad_outputs=valid_grad_outputs,
                    retain_graph=False,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "split backward invariant failed while computing captured "
                    f"weight gradients for bucket={ubid!r} microbatch={mb_idx!r}"
                ) from exc

            del param_group["grads"]

            for grad_acc, dw in zip(param_group["params"], dweights):
                if dw is None or grad_acc not in grad_acc_to_weight:
                    continue
                weight = grad_acc_to_weight[grad_acc]
                dw = self.grad_with_param_layout(weight, dw)
                if weight.grad is None:
                    weight.grad = dw
                else:
                    if (
                        weight.grad.dtype != weight.dtype
                        or weight.grad.device != weight.device
                        or weight.grad.layout != weight.layout
                        or tuple(weight.grad.stride()) != tuple(weight.stride())
                    ):
                        weight.grad = self.grad_with_param_layout(weight, weight.grad)
                    weight.grad += dw


@dataclass
class DagExecutor:
    """Execute a loaded TrainingDAG in sorted order."""

    runtime: RuntimeState
    stages: StageStore
    buffers: BufferStore
    events: EventStore
    params: ParamStorage
    communication: CommunicationExecutor
    compute: ComputeExecutor
    logger: Any
    expert_a2a_states: dict[str, _ExpertA2AState] = field(
        default_factory=dict
    )
    routed_cross_group_states: dict[str, _RoutedCrossGroupState] = field(
        default_factory=dict
    )

    @staticmethod
    def _node_meta(node: Any) -> dict:
        return getattr(node, "node_meta", {}) or {}

    def _node_bucket_key(self, node: Any) -> Any | None:
        return self._node_meta(node).get("bucket_key")

    def _sync_payload_ubid(self, node: Any) -> Any | None:
        bk = self._node_bucket_key(node)
        if bk is not None:
            return bk
        if node.data_preds:
            pred_bk = self._node_bucket_key(node.data_preds[0])
            if pred_bk is not None:
                return pred_bk
        return None

    def _sync_payload_ubids(self, node: Any) -> list[Any]:
        sync_ubids = self._node_meta(node).get("sync_ubids")
        if sync_ubids:
            return list(sync_ubids)
        ubid = self._sync_payload_ubid(node)
        return [ubid] if ubid is not None else []

    def _rf_enter(self, label: str):
        if not self.runtime.pytorch_profiler_enabled:
            return None
        mt = torch.autograd.set_multithreading_enabled(False)
        mt.__enter__()
        rf = torch.profiler.record_function(label)
        rf.__enter__()
        return (rf, mt)

    @staticmethod
    def _rf_exit(rf) -> None:
        if rf is not None:
            rf_ctx, mt = rf
            rf_ctx.__exit__(None, None, None)
            mt.__exit__(None, None, None)

    @staticmethod
    def _node_tag_str(node: Any) -> str:
        tag = getattr(node, "tag", None)
        if not isinstance(tag, dict) or not tag:
            return "{}"
        items = sorted(tag.items(), key=lambda kv: kv[0])
        return "{" + ",".join(f"{k}={v}" for k, v in items) + "}"

    def _wait_for_all_gather(
        self,
        compute_node: Any,
        compute_stream: torch.cuda.Stream | None = None,
    ) -> None:
        if compute_stream is None:
            compute_stream = self.runtime.stream_for_task(compute_node)
        for pred in compute_node.data_preds:
            if pred.task_type == TaskType.ALL_GATHER:
                ag_evt = self.events.all_gather.get(pred.uid)
                if ag_evt is not None:
                    compute_stream.wait_event(ag_evt)

    def _wait_for_backward_input_events(
        self,
        node: Any,
        compute_stream: torch.cuda.Stream,
        *,
        wait: bool = True,
    ) -> tuple[Any | None, Any | None]:
        """Consume communication inputs shared by BWD and BWD_I.

        ``run`` passes ``wait=False`` because the shared execution plan has
        already enqueued these waits before profiling starts.  The default is
        retained for direct callers and older execution paths.
        """
        recv_pred = next(
            (p for p in node.data_preds if p.task_type == TaskType.RECV), None
        )
        if recv_pred is not None and recv_pred.uid in self.events.recv:
            recv_event = self.events.recv.pop(recv_pred.uid)
            if wait:
                compute_stream.wait_event(recv_event)

        if wait:
            self._wait_for_all_gather(node, compute_stream)

        a2a_pred = next(
            (
                p
                for p in node.data_preds
                if p.task_type
                in (
                    TaskType.BWD_A2A,
                    TaskType.BWD_M_TO_N,
                    TaskType.BWD_N_TO_M,
                )
            ),
            None,
        )
        if a2a_pred is not None and a2a_pred.uid in self.events.a2a:
            a2a_event = self.events.a2a.pop(a2a_pred.uid)
            if wait:
                compute_stream.wait_event(a2a_event)

        return recv_pred, a2a_pred

    def _consume_local_backward_output_grads(
        self,
        node: Any,
        fwd_out: dict,
        stage_outputs: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Consume gradients from a downstream compute node on this actor."""
        local_bwd_pred = next(
            (
                pred
                for pred in node.data_preds
                if pred.task_type in (TaskType.BWD, TaskType.BWD_I)
            ),
            None,
        )
        if local_bwd_pred is None:
            return stage_outputs, None

        local_buf = self.buffers.task[local_bwd_pred.uid]
        detached_outs = fwd_out["detached_outs"]
        output_grads_full = [
            local_buf["inp_grads"][i]
            for i, tensor in enumerate(detached_outs)
            if (
                tensor is not None
                and isinstance(tensor, torch.Tensor)
                and tensor.requires_grad
            )
        ]
        pairs = [
            (output, grad)
            for output, grad in zip(stage_outputs, output_grads_full)
            if grad is not None
        ]
        self.buffers.release(local_bwd_pred.uid)
        return (
            [output for output, _grad in pairs],
            [grad for _output, grad in pairs],
        )

    def _forward_inputs_for_node(
        self,
        node: Any,
        ubid: Any,
        external_inputs: Any,
    ) -> Any:
        """Assemble a compute ABI from all explicit FX producer edges."""
        data_preds = [
            pred
            for pred in node.data_preds
            if pred.task_type
            in {
                TaskType.FWD,
                TaskType.FWD_A2A,
                TaskType.FWD_M_TO_N,
                TaskType.FWD_N_TO_M,
                TaskType.RECV,
            }
        ]
        if not data_preds:
            return external_inputs

        input_arity = len(self.stages.bucket(ubid).input_idxs)
        slots: list[Any] = [None] * input_arity
        filled: set[int] = set()
        used_structured = False
        fallback: Any | None = None
        mappings_by_source = self._node_meta(node).get(
            "runtime_inputs_by_source", {}
        )
        for pred in data_preds:
            buffer = self.buffers.task[pred.uid]
            runtime_inputs = (
                buffer.get("runtime_inputs")
                if isinstance(buffer, dict)
                else None
            )
            if isinstance(runtime_inputs, dict):
                used_structured = True
                for index, value in runtime_inputs.items():
                    index = int(index)
                    if index in filled:
                        raise RuntimeError(
                            f"forward node {node.uid} receives input slot "
                            f"{index} from multiple predecessors"
                        )
                    slots[index] = value
                    filled.add(index)
            elif pred.task_type == TaskType.FWD and isinstance(
                mappings_by_source, dict
            ) and isinstance(mappings_by_source.get(pred.uid), list):
                used_structured = True
                detached_outs = buffer["detached_outs"]
                for mapping in mappings_by_source[pred.uid]:
                    source_idx = int(mapping["source_output_idx"])
                    destination_idx = int(mapping["destination_input_idx"])
                    if destination_idx in filled:
                        raise RuntimeError(
                            f"forward node {node.uid} receives input slot "
                            f"{destination_idx} from multiple predecessors"
                        )
                    slots[destination_idx] = detached_outs[source_idx]
                    filled.add(destination_idx)
            elif isinstance(buffer, dict) and "detached_outs" in buffer:
                fallback = buffer["detached_outs"]
            else:
                fallback = buffer
            self.buffers.release(pred.uid)

        if not used_structured:
            if fallback is None:
                raise RuntimeError(
                    f"forward node {node.uid} has no activation payload"
                )
            return fallback
        missing = sorted(set(range(input_arity)) - filled)
        if missing:
            raise RuntimeError(
                f"forward node {node.uid} is missing runtime input slots "
                f"{missing}; filled={sorted(filled)}"
            )
        return slots

    def _backward_outputs_and_grads(
        self,
        node: Any,
        fwd_out: dict[str, Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Merge gradients from every routed, bypass, or ordinary consumer."""
        detached_outs = list(fwd_out["detached_outs"])
        full_grads: list[torch.Tensor | None] = [None] * len(detached_outs)
        grad_mappings = self._node_meta(node).get(
            "runtime_output_grads_by_source", {}
        )
        consumed = False

        def _accumulate(index: int, grad: Any) -> None:
            nonlocal consumed
            if grad is None:
                return
            if not isinstance(grad, torch.Tensor):
                raise TypeError(
                    f"backward node {node.uid} received non-tensor gradient "
                    f"for output slot {index}: {type(grad).__name__}"
                )
            if index < 0 or index >= len(full_grads):
                raise ValueError(
                    f"backward node {node.uid} gradient slot {index} is "
                    f"outside output arity {len(full_grads)}"
                )
            full_grads[index] = (
                grad
                if full_grads[index] is None
                else full_grads[index] + grad
            )
            consumed = True

        relevant_types = {
            TaskType.BWD,
            TaskType.BWD_I,
            TaskType.BWD_A2A,
            TaskType.BWD_M_TO_N,
            TaskType.BWD_N_TO_M,
            TaskType.RECV,
        }
        differentiable_slots = [
            index
            for index, tensor in enumerate(detached_outs)
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad
        ]
        for pred in node.data_preds:
            if pred.task_type not in relevant_types:
                continue
            buffer = self.buffers.task[pred.uid]
            if isinstance(buffer, dict) and "inp_grads" in buffer:
                inp_grads = list(buffer["inp_grads"])
                mapping = (
                    grad_mappings.get(pred.uid)
                    if isinstance(grad_mappings, dict)
                    else None
                )
                if isinstance(mapping, list):
                    for item in mapping:
                        _accumulate(
                            int(item["source_output_idx"]),
                            inp_grads[int(item["destination_input_idx"])],
                        )
                else:
                    for index, grad in enumerate(inp_grads):
                        _accumulate(index, grad)
            else:
                raw_grads = (
                    list(buffer)
                    if isinstance(buffer, (list, tuple))
                    else [buffer]
                )
                for index, grad in zip(differentiable_slots, raw_grads):
                    _accumulate(index, grad)
            self.buffers.release(pred.uid)

        outputs = [
            output
            for output in fwd_out["pre_detach_outs"]
            if isinstance(output, torch.Tensor) and output.requires_grad
        ]
        if not consumed:
            return outputs, None
        grads = [full_grads[index] for index in differentiable_slots]
        pairs = [
            (output, grad)
            for output, grad in zip(outputs, grads)
            if grad is not None
        ]
        if not pairs:
            raise RuntimeError(
                f"backward node {node.uid} received no materialized gradients"
            )
        return (
            [output for output, _grad in pairs],
            [grad for _output, grad in pairs],
        )

    def _all_to_all_ep_boundary(
        self,
        node: Any,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream,
        routing_counts: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Keep boundary layout kernels on the same stream as the collective.
        # This is especially important in profiling mode, where the timing
        # events delimit this physical stream rather than CUDA's ambient one.
        with torch.cuda.stream(stream):
            node_meta = self._node_meta(node)
            direction = node_meta.get("direction")
            if direction == "outgoing":
                tensor = tensor.contiguous()
            elif direction != "incoming":
                raise ValueError(
                    f"A2A node uid={getattr(node, 'uid', '<unknown>')} has invalid "
                    f"direction={direction!r}; expected 'incoming' or 'outgoing'"
                )

            expert_fwd_uid = node_meta.get("expert_fwd_uid")
            if not isinstance(expert_fwd_uid, str):
                raise ValueError(
                    f"A2A node {getattr(node, 'uid', '<unknown>')} is missing "
                    "expert_fwd_uid"
                )
            state = self.expert_a2a_states.get(expert_fwd_uid)
            if state is None:
                if routing_counts is None:
                    raise RuntimeError(
                        f"A2A node {getattr(node, 'uid', '<unknown>')} ran before "
                        f"forward dispatch for expert {expert_fwd_uid}"
                    )
                participants = node_meta.get("participants")
                if not isinstance(participants, list) or not participants:
                    raise ValueError(
                        f"A2A node {node.uid} has invalid participants="
                        f"{participants!r}"
                    )
                participant_order = tuple(sorted(int(rank) for rank in participants))
                try:
                    participant_order.index(self.runtime.global_rank)
                except ValueError as error:
                    raise ValueError(
                        f"Actor rank {self.runtime.global_rank} is not in A2A "
                        f"participants={participants}"
                    ) from error
                (
                    num_global_routes,
                    shard_devices,
                    route_partitions,
                ) = _route_partition_metadata(node_meta, participant_order)
                state = _ExpertA2AState(
                    participants=participant_order,
                    shard_devices=shard_devices,
                    num_global_routes=num_global_routes,
                    route_partitions=route_partitions,
                )
                self.expert_a2a_states[expert_fwd_uid] = state

            task_type = getattr(node, "task_type", None)
            if task_type == TaskType.FWD_A2A:
                if direction == "incoming":
                    if routing_counts is None:
                        raise ValueError(
                            f"Forward dispatch node {node.uid} is missing "
                            "route_counts"
                        )
                    return self._dispatch_expert_a2a(
                        state,
                        tensor.contiguous(),
                        routing_counts,
                    )
                return self._combine_expert_a2a(state, tensor)

            if task_type == TaskType.BWD_A2A:
                if direction == "incoming":
                    return self._dispatch_expert_a2a_gradient(
                        state,
                        tensor.contiguous(),
                    )
                return self._combine_expert_a2a(state, tensor)

            raise ValueError(
                f"A2A node {getattr(node, 'uid', '<unknown>')} has unsupported "
                f"task_type={task_type}"
            )

    @staticmethod
    def _require_expert_a2a_state(state: _ExpertA2AState) -> None:
        if state.input_splits is None or state.output_splits is None:
            raise RuntimeError(
                "A2A combine/gradient dispatch ran before forward dispatch"
            )

    def _dispatch_expert_a2a(
        self,
        state: _ExpertA2AState,
        routed_input: torch.Tensor,
        route_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if routed_input.ndim != 2:
            raise ValueError(
                "routed expert payload must have shape [tokens, hidden], got "
                f"shape={tuple(routed_input.shape)}"
            )
        if route_counts.ndim != 2 or int(route_counts.shape[0]) != 1:
            raise ValueError(
                "route_counts must have shape [1, num_global_routes] before "
                f"dispatch, got shape={tuple(route_counts.shape)}"
            )
        if torch.is_floating_point(route_counts) or torch.is_complex(route_counts):
            raise TypeError(
                "route_counts must use an integer dtype, got "
                f"{route_counts.dtype}"
            )
        if bool(torch.any(route_counts < 0).item()):
            raise ValueError("route_counts cannot contain negative counts")
        if int(route_counts.shape[1]) != state.num_global_routes:
            raise ValueError(
                f"route_counts has {route_counts.shape[1]} routes, expected "
                f"num_global_routes={state.num_global_routes}"
            )
        flat_counts = route_counts.reshape(-1)
        routed_tokens = int(flat_counts.sum().item())
        if int(routed_input.shape[0]) != routed_tokens:
            raise ValueError(
                f"routed expert payload has {routed_input.shape[0]} rows, but "
                f"route_counts sum to {routed_tokens}"
            )
        num_local_routes = state.num_global_routes // state.ep_degree
        # Schedule validation makes ownership order identical to process-group
        # rank order, so model route-major blocks are already collective-ready.
        counts_for_collective = flat_counts
        collective_input = routed_input.contiguous()
        state.payload_shape = tuple(routed_input.shape)
        state.payload_dtype = routed_input.dtype
        if state.ep_degree > 1:
            with torch.no_grad():
                # Fixed-size metadata exchange. The result is source-major:
                # [source rank][local expert].
                grouped_counts = torch.empty_like(counts_for_collective)
                dist.all_to_all_single(
                    grouped_counts,
                    counts_for_collective,
                    group=self.runtime.actor_group_routed_group,
                )
                input_splits = (
                    counts_for_collective.view(state.ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=True)
                )
                output_splits = (
                    grouped_counts.view(state.ep_degree, -1)
                    .sum(dim=1)
                    .to(torch.device("cpu"), non_blocking=False)
                )
                state.input_splits = input_splits.tolist()
                state.output_splits = output_splits.tolist()

            collective_output = _stream_ordered_all_to_all(
                collective_input,
                state.output_splits,
                state.input_splits,
                self.runtime.actor_group_routed_group,
            )
        else:
            grouped_counts = counts_for_collective
            collective_output = collective_input
            state.input_splits = [routed_tokens]
            state.output_splits = [routed_tokens]

        source_major_payload = collective_output
        source_major_counts = grouped_counts.view(
            state.ep_degree, num_local_routes
        )
        return source_major_payload, source_major_counts

    def _dispatch_expert_a2a_gradient(
        self,
        state: _ExpertA2AState,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._require_expert_a2a_state(state)
        _validate_routed_payload_schema(
            tensor,
            state.payload_shape,
            state.payload_dtype,
            context="A2A backward dispatch",
        )
        if state.ep_degree > 1:
            tensor = _stream_ordered_all_to_all(
                tensor.contiguous(),
                state.output_splits,
                state.input_splits,
                self.runtime.actor_group_routed_group,
            )

        return tensor

    def _combine_expert_a2a(
        self,
        state: _ExpertA2AState,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._require_expert_a2a_state(state)
        _validate_routed_payload_schema(
            tensor,
            state.payload_shape,
            state.payload_dtype,
            context="A2A combine",
        )
        if state.ep_degree > 1:
            tensor = _stream_ordered_all_to_all(
                tensor.contiguous(),
                state.input_splits,
                state.output_splits,
                self.runtime.actor_group_routed_group,
            )
        return tensor

    def _routed_cross_group_state(
        self,
        node: Any,
    ) -> _RoutedCrossGroupState:
        """Find or initialize actor-local state for one D -> R -> C route."""
        meta = self._node_meta(node)
        expert_fwd_uid = meta.get("expert_fwd_uid")
        if not isinstance(expert_fwd_uid, str):
            raise ValueError(
                f"Routed node {getattr(node, 'uid', '<unknown>')} is missing "
                "expert_fwd_uid"
            )

        try:
            dispatch_devices = tuple(int(d) for d in meta["dispatch_devices"])
            routed_devices = tuple(int(d) for d in meta["routed_devices"])
            combine_devices = tuple(int(d) for d in meta["combine_devices"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Routed lifecycle {expert_fwd_uid} is missing D/R/C devices"
            ) from error
        for name, devices in (
            ("dispatch", dispatch_devices),
            ("routed", routed_devices),
            ("combine", combine_devices),
        ):
            if not devices or devices != tuple(sorted(set(devices))):
                raise ValueError(
                    f"{name}_devices must be nonempty, unique, and ascending; "
                    f"got {devices}"
                )
        if set(dispatch_devices) & set(routed_devices) or set(
            routed_devices
        ) & set(combine_devices):
            raise ValueError(
                f"Routed lifecycle {expert_fwd_uid} requires D/R and R/C to "
                "be disjoint"
            )
        if dispatch_devices != combine_devices:
            if set(dispatch_devices) & set(combine_devices):
                raise ValueError(
                    f"Routed lifecycle {expert_fwd_uid} requires D/C to be "
                    "identical or disjoint"
                )
            if len(dispatch_devices) != len(combine_devices):
                raise ValueError(
                    f"Routed lifecycle {expert_fwd_uid} requires equal D/C "
                    "widths"
                )
        (
            num_global_routes,
            shard_devices,
            route_partitions,
        ) = _route_partition_metadata(meta, routed_devices)
        state = self.routed_cross_group_states.get(expert_fwd_uid)
        if state is not None:
            expected = (
                dispatch_devices,
                routed_devices,
                combine_devices,
                shard_devices,
                num_global_routes,
                route_partitions,
            )
            actual = (
                state.dispatch_devices,
                state.routed_devices,
                state.combine_devices,
                state.shard_devices,
                state.num_global_routes,
                state.route_partitions,
            )
            if actual != expected:
                raise ValueError(
                    f"Routed lifecycle {expert_fwd_uid} metadata changed: "
                    f"state={actual}, node={expected}"
                )
            return state

        dispatch_participants = tuple(
            sorted((*dispatch_devices, *routed_devices))
        )
        combine_participants = tuple(
            sorted((*routed_devices, *combine_devices))
        )
        global_rank = int(self.runtime.global_rank)

        def _group(
            participants: tuple[int, ...], channel: str
        ) -> Any | None:
            if global_rank not in participants:
                return None
            return self.runtime.process_group_for_participants(
                participants, channel=channel
            )

        payload_shape: tuple[int, ...] | None = None
        payload_dtype: torch.dtype | None = None
        routing_counts_dtype: torch.dtype | None = None
        payload_schema = meta.get("payload_schema")
        if isinstance(payload_schema, dict):
            raw_shape = payload_schema.get("shape")
            if isinstance(raw_shape, (list, tuple)) and all(
                isinstance(dim, int) and not isinstance(dim, bool)
                for dim in raw_shape
            ):
                payload_shape = tuple(int(dim) for dim in raw_shape)
            if payload_schema.get("dtype") is not None:
                payload_dtype = _torch_dtype_from_name(
                    payload_schema["dtype"]
                )
        routing_counts_schema = meta.get("routing_counts_schema")
        if isinstance(routing_counts_schema, dict):
            if routing_counts_schema.get("dtype") is not None:
                routing_counts_dtype = _torch_dtype_from_name(
                    routing_counts_schema["dtype"]
                )
        state = _RoutedCrossGroupState(
            dispatch_devices=dispatch_devices,
            routed_devices=routed_devices,
            combine_devices=combine_devices,
            shard_devices=shard_devices,
            num_global_routes=num_global_routes,
            route_partitions=route_partitions,
            dispatch_participants=dispatch_participants,
            combine_participants=combine_participants,
            layout_group=_group(dispatch_participants, "routed_layout"),
            dispatch_m_to_n_group=_group(
                dispatch_participants, "m_to_n"
            ),
            dispatch_n_to_m_group=_group(
                dispatch_participants, "n_to_m"
            ),
            combine_m_to_n_group=_group(combine_participants, "m_to_n"),
            combine_n_to_m_group=_group(combine_participants, "n_to_m"),
            payload_shape=payload_shape,
            payload_dtype=payload_dtype,
            routing_counts_dtype=routing_counts_dtype,
        )
        self.routed_cross_group_states[expert_fwd_uid] = state
        return state

    def _empty_routed_payload(
        self,
        node: Any,
        state: _RoutedCrossGroupState,
        _tensor_idx: int,
    ) -> torch.Tensor:
        """Allocate a zero-token payload on either side of the boundary."""
        if state.payload_shape is not None and state.payload_dtype is not None:
            return torch.empty(
                (0, *state.payload_shape[1:]),
                dtype=state.payload_dtype,
                device=self.runtime.device,
            )
        raise RuntimeError(
            f"Routed destination {node.uid} has no runtime payload schema; "
            "forward dispatch must exchange the payload shape and disjoint "
            "dense lanes must receive it in Piper's direct state envelope"
        )

    @staticmethod
    def _require_dispatch_layout(state: _RoutedCrossGroupState) -> None:
        if (
            state.dispatch_input_splits is None
            or state.routed_dispatch_output_splits is None
        ):
            raise RuntimeError(
                "dispatch-side routed communication ran before forward M-to-N "
                "established its split layout"
            )

    @staticmethod
    def _require_combine_layout(state: _RoutedCrossGroupState) -> None:
        if (
            state.routed_combine_input_splits is None
            or state.combine_output_splits is None
        ):
            raise RuntimeError(
                "combine-side routed communication ran before its routed or "
                "dense lane received route counts"
            )

    def _initialize_dispatch_layout(
        self,
        node: Any,
        state: _RoutedCrossGroupState,
        routing_counts: torch.Tensor | None,
        payload: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exchange route counts/shape and derive union-A2A splits."""
        meta = self._node_meta(node)
        role = meta.get("role")
        counts_idx = int(meta["routing_counts_idx"])
        if role == "source":
            if routing_counts is None:
                raise ValueError(
                    f"Forward M-to-N source {node.uid} is missing routing counts"
                )
            if payload is None or payload.ndim != 2:
                raise ValueError(
                    f"Forward M-to-N source {node.uid} requires a rank-2 "
                    "payload before layout exchange"
                )
            local_counts = routing_counts
            local_payload_shape = torch.tensor(
                tuple(payload.shape),
                dtype=torch.int64,
                device=payload.device,
            )
            state.payload_shape = tuple(int(dim) for dim in payload.shape)
            state.payload_dtype = payload.dtype
        elif role == "destination":
            if state.routing_counts_dtype is None:
                raise RuntimeError(
                    f"Forward M-to-N destination {node.uid} lacks the "
                    "serialized routing-count dtype"
                )
            local_counts = torch.zeros(
                (1, state.num_global_routes),
                dtype=state.routing_counts_dtype,
                device=self.runtime.device,
            )
            local_payload_shape = torch.zeros(
                (2,), dtype=torch.int64, device=self.runtime.device
            )
        else:
            raise ValueError(
                f"Routed node {node.uid} has invalid role={role!r}"
            )

        if local_counts.ndim != 2 or int(local_counts.shape[0]) != 1:
            raise ValueError(
                "route_counts must have shape [1, num_global_routes] before "
                f"dispatch, got shape={tuple(local_counts.shape)}"
            )
        if torch.is_floating_point(local_counts) or torch.is_complex(local_counts):
            raise TypeError(
                "route_counts must use an integer dtype, got "
                f"{local_counts.dtype}"
            )
        if bool(torch.any(local_counts < 0).item()):
            raise ValueError("route_counts cannot contain negative counts")
        if int(local_counts.shape[1]) != state.num_global_routes:
            raise ValueError(
                f"route_counts has {local_counts.shape[1]} routes, expected "
                f"num_global_routes={state.num_global_routes}"
            )
        flat_counts = local_counts.reshape(-1)

        with torch.no_grad():
            participant_index = {
                rank: index
                for index, rank in enumerate(state.dispatch_participants)
            }
            input_splits = [0] * len(state.dispatch_participants)
            output_splits = [0] * len(state.dispatch_participants)
            global_rank = self.runtime.global_rank
            if global_rank in state.dispatch_devices:
                # Derive source-local send sizes before enqueuing the layout
                # collective.  In particular, the scalar reads below must not
                # accidentally synchronize behind that collective on a backend
                # that launches NCCL work on the current stream.
                for rank, (start, stop) in zip(
                    state.shard_devices, state.route_partitions
                ):
                    input_splits[participant_index[rank]] = int(
                        flat_counts[start:stop].sum().item()
                    )
                state.dispatch_counts = local_counts
            elif global_rank not in state.routed_devices:
                raise ValueError(
                    f"Actor rank {global_rank} is not a participant in routed "
                    f"boundary {state.dispatch_participants}"
                )

            gathered_counts = torch.empty(
                (len(state.dispatch_participants), state.num_global_routes),
                dtype=local_counts.dtype,
                device=local_counts.device,
            )
            dist.all_gather_into_tensor(
                gathered_counts,
                local_counts.contiguous(),
                group=state.layout_group,
            )
            gathered_shapes = torch.empty(
                (len(state.dispatch_participants), 2),
                dtype=torch.int64,
                device=local_payload_shape.device,
            )
            dist.all_gather_into_tensor(
                gathered_shapes,
                local_payload_shape,
                group=state.layout_group,
            )
            grouped_counts: torch.Tensor | None = None
            if global_rank in state.dispatch_devices:
                # Sources need only their local row. Both metadata collectives
                # remain ordered before the payload on this node's stream.
                pass
            else:
                # Destinations need every source row before they can derive
                # receive splits and compact source-major local-route counts.
                source_counts = torch.stack(
                    [
                        gathered_counts[participant_index[rank]]
                        for rank in state.dispatch_devices
                    ]
                )
                source_shapes = torch.stack(
                    [
                        gathered_shapes[participant_index[rank]]
                        for rank in state.dispatch_devices
                    ]
                )
                if bool(torch.any(source_shapes <= 0).item()):
                    raise ValueError(
                        f"Forward M-to-N destination {node.uid} received "
                        f"invalid source payload shapes={source_shapes.tolist()}"
                    )
                hidden_widths = source_shapes[:, 1]
                if bool(torch.any(hidden_widths != hidden_widths[0]).item()):
                    raise ValueError(
                        f"Forward M-to-N destination {node.uid} requires one "
                        f"payload hidden width, got shapes={source_shapes.tolist()}"
                    )
                state.payload_shape = (
                    int(source_shapes[0, 0].item()),
                    int(hidden_widths[0].item()),
                )
                destination_index = state.shard_devices.index(global_rank)
                start, stop = state.route_partitions[destination_index]
                grouped_counts = source_counts[:, start:stop].contiguous()
                for source_index, rank in enumerate(state.dispatch_devices):
                    output_splits[participant_index[rank]] = int(
                        source_counts[source_index, start:stop].sum().item()
                    )
                state.routed_counts = grouped_counts
            state.dispatch_input_splits = input_splits
            state.routed_dispatch_output_splits = output_splits

        return local_counts, grouped_counts

    def _initialize_combine_layout(
        self,
        state: _RoutedCrossGroupState,
    ) -> None:
        """Derive R-to-C splits from counts held by the local actor role."""
        participant_index = {
            rank: index
            for index, rank in enumerate(state.combine_participants)
        }
        input_splits = [0] * len(state.combine_participants)
        output_splits = [0] * len(state.combine_participants)
        global_rank = int(self.runtime.global_rank)
        if global_rank in state.routed_devices:
            counts = state.routed_counts
            if counts is None:
                raise RuntimeError(
                    "routed combine source has not received dispatch counts"
                )
            if tuple(counts.shape) != (
                len(state.dispatch_devices),
                state.num_global_routes // len(state.routed_devices),
            ):
                raise ValueError(
                    f"routed local counts have invalid shape={tuple(counts.shape)}"
                )
            for source_index, combine_rank in enumerate(
                state.combine_devices
            ):
                input_splits[participant_index[combine_rank]] = int(
                    counts[source_index].sum().item()
                )
        elif global_rank in state.combine_devices:
            counts = state.combine_counts
            if counts is None:
                if state.dispatch_devices == state.combine_devices:
                    counts = state.dispatch_counts
                if counts is None:
                    raise RuntimeError(
                        "combine destination has not received its paired "
                        "dispatch counts"
                    )
            if tuple(counts.shape) != (1, state.num_global_routes):
                raise ValueError(
                    f"combine counts have invalid shape={tuple(counts.shape)}"
                )
            flat_counts = counts.reshape(-1)
            for routed_rank, (start, stop) in zip(
                state.routed_devices, state.route_partitions
            ):
                output_splits[participant_index[routed_rank]] = int(
                    flat_counts[start:stop].sum().item()
                )
        else:
            raise ValueError(
                f"Actor rank {global_rank} is not in combine participants "
                f"{state.combine_participants}"
            )
        state.routed_combine_input_splits = input_splits
        state.combine_output_splits = output_splits

    def _routed_cross_group_boundary(
        self,
        node: Any,
        tensor: torch.Tensor | None,
        stream: torch.cuda.Stream,
        *,
        routing_counts: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Execute a disjoint-group routed transfer over the union group.

        Every rank in both ActorGroups participates in one variable-split
        all-to-all.  For M-to-N only dense ranks send and expert ranks receive;
        N-to-M swaps those split vectors.  The operation is therefore not an
        expert-parallel all-to-all even when the two group sizes are equal.
        """
        with torch.cuda.stream(stream):
            meta = self._node_meta(node)
            direction = meta.get("direction")
            role = meta.get("role")
            task_type = getattr(node, "task_type", None)
            if direction not in {"incoming", "outgoing"}:
                raise ValueError(
                    f"Routed node {node.uid} has invalid direction={direction!r}"
                )
            if role not in {"source", "destination"}:
                raise ValueError(
                    f"Routed node {node.uid} has invalid role={role!r}"
                )

            state = self._routed_cross_group_state(node)
            tensor_idx = int(meta["a2a_tensor_idx"])
            grouped_counts: torch.Tensor | None = None
            if task_type == TaskType.FWD_M_TO_N:
                local_counts, grouped_counts = (
                    self._initialize_dispatch_layout(
                        node, state, routing_counts, tensor
                    )
                )
                output_splits = state.routed_dispatch_output_splits
                input_splits = state.dispatch_input_splits
                process_group = state.dispatch_m_to_n_group
            elif task_type == TaskType.FWD_N_TO_M:
                self._initialize_combine_layout(state)
                output_splits = state.combine_output_splits
                input_splits = state.routed_combine_input_splits
                process_group = state.combine_n_to_m_group
            elif task_type == TaskType.BWD_M_TO_N:
                self._require_combine_layout(state)
                output_splits = state.routed_combine_input_splits
                input_splits = state.combine_output_splits
                process_group = state.combine_m_to_n_group
            elif task_type == TaskType.BWD_N_TO_M:
                self._require_dispatch_layout(state)
                output_splits = state.dispatch_input_splits
                input_splits = state.routed_dispatch_output_splits
                process_group = state.dispatch_n_to_m_group
            else:
                raise ValueError(
                    f"unsupported routed task type {task_type!r}"
                )
            assert output_splits is not None and input_splits is not None
            if process_group is None:
                raise RuntimeError(
                    f"Routed node {node.uid} has no local process group"
                )

            if role == "source":
                if tensor is None:
                    raise ValueError(f"Routed source node {node.uid} has no tensor")
                if task_type == TaskType.FWD_M_TO_N:
                    routed_tokens = int(local_counts.sum().item())
                    if tensor.ndim < 1 or int(tensor.shape[0]) != routed_tokens:
                        raise ValueError(
                            f"Forward M-to-N source {node.uid} has payload "
                            f"shape={tuple(tensor.shape)} but routing counts "
                            f"sum to {routed_tokens} tokens"
                        )
                _validate_routed_payload_schema(
                    tensor,
                    state.payload_shape,
                    state.payload_dtype,
                    context=f"Routed source {node.uid}",
                )
                collective_input = tensor.contiguous()
            else:
                collective_input = self._empty_routed_payload(
                    node, state, tensor_idx
                )
            collective_output = _stream_ordered_all_to_all(
                collective_input,
                output_splits,
                input_splits,
                process_group,
            )
            if role == "source":
                # Source ranks receive zero rows. The stream completion event,
                # rather than an escaped lazy tensor, represents completion.
                return None
            if task_type == TaskType.FWD_M_TO_N:
                assert grouped_counts is not None
                state.routed_counts = grouped_counts
                return collective_output, grouped_counts
            return collective_output

    def _run_forward_routed_cross_group(
        self,
        node: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        meta = self._node_meta(node)
        role = meta["role"]
        direction = meta["direction"]
        task_type = getattr(node, "task_type", None)
        tensor_idx = (
            int(meta["dispatch_payload_idx"])
            if task_type == TaskType.FWD_M_TO_N
            else int(meta["combine_payload_idx"])
        )
        state = self._routed_cross_group_state(node)

        if task_type == TaskType.FWD_N_TO_M and role == "destination":
            layout_pred = next(
                (
                    pred
                    for pred in node.data_preds
                    if pred.task_type == TaskType.RECV
                    and self._node_meta(pred).get("dense_state_transfer")
                ),
                None,
            )
            if layout_pred is not None:
                if state.combine_counts is None:
                    raise RuntimeError(
                        f"Forward N-to-M destination {node.uid} ran before "
                        "direct routing layout was received"
                    )
                self.buffers.release(layout_pred.uid)

        tensor: torch.Tensor | None = None
        routing_counts: torch.Tensor | None = None
        if role == "source":
            fwd_pred = next(
                pred for pred in node.data_preds if pred.task_type == TaskType.FWD
            )
            fwd_buf = dict(self.buffers.task[fwd_pred.uid])
            self.buffers.release(fwd_pred.uid)
            detached_outs = list(fwd_buf["detached_outs"])
            tensor = detached_outs[tensor_idx]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"Routed source node {node.uid} payload index {tensor_idx} "
                    f"is {type(tensor).__name__}, not a Tensor"
                )
            if direction == "incoming":
                counts_idx = int(meta["routing_counts_idx"])
                routing_counts = detached_outs[counts_idx]

        result = self._routed_cross_group_boundary(
            node,
            tensor,
            stream,
            routing_counts=routing_counts,
        )

        if role == "destination":
            if direction == "incoming":
                if not isinstance(result, tuple):
                    raise RuntimeError(
                        f"Forward M-to-N destination {node.uid} did not return "
                        "a routed tensor and local counts"
                    )
                routed_tensor, local_counts = result
                inputs = [None] * int(meta["expert_input_arity"])
                dispatch_input_idx = int(meta["routed_dispatch_input_idx"])
                counts_input_idx = int(meta["routed_counts_input_idx"])
                inputs[dispatch_input_idx] = routed_tensor.requires_grad_(True)
                inputs[counts_input_idx] = local_counts
                self.buffers.task[node.uid] = {
                    "detached_outs": inputs,
                    "runtime_inputs": {
                        dispatch_input_idx: inputs[dispatch_input_idx],
                        counts_input_idx: local_counts,
                    },
                }
            else:
                if not isinstance(result, torch.Tensor):
                    raise RuntimeError(
                        f"Forward N-to-M destination {node.uid} did not return "
                        "a routed tensor"
                    )
                destination_idx = int(meta["combine_destination_input_idx"])
                self.buffers.task[node.uid] = {
                    "runtime_inputs": {
                        destination_idx: result.requires_grad_(True)
                    }
                }
        elif result is not None:
            raise RuntimeError(
                f"Routed source node {node.uid} unexpectedly received a tensor"
            )

        event = torch.cuda.Event()
        event.record(stream)
        self.events.a2a[node.uid] = event

    def _run_backward_routed_cross_group(
        self,
        node: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        meta = self._node_meta(node)
        role = meta["role"]
        direction = meta["direction"]
        task_type = getattr(node, "task_type", None)
        tensor_idx = (
            int(meta["combine_destination_input_idx"])
            if task_type == TaskType.BWD_M_TO_N
            else int(meta["routed_dispatch_input_idx"])
        )
        state = self._routed_cross_group_state(node)

        tensor: torch.Tensor | None = None
        if role == "source":
            bwd_pred = next(
                pred
                for pred in node.data_preds
                if pred.task_type in (TaskType.BWD, TaskType.BWD_I, TaskType.RECV)
            )
            bwd_buf = dict(self.buffers.task[bwd_pred.uid])
            self.buffers.release(bwd_pred.uid)
            inp_grads = list(bwd_buf["inp_grads"])
            tensor = inp_grads[tensor_idx]
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(
                    f"Routed backward source {node.uid} has no tensor gradient "
                    f"at index {tensor_idx}"
                )
        result = self._routed_cross_group_boundary(node, tensor, stream)

        if role == "destination":
            if not isinstance(result, torch.Tensor):
                raise RuntimeError(
                    f"Routed backward destination {node.uid} did not receive "
                    "a tensor gradient"
                )
            if direction == "incoming":
                inp_grads = [None] * int(meta["expert_output_arity"])
                inp_grads[int(meta["combine_payload_idx"])] = result
                self.buffers.task[node.uid] = {"inp_grads": inp_grads}
            else:
                output_arity = int(meta["dispatch_output_arity"])
                inp_grads = [None] * output_arity
                inp_grads[int(meta["dispatch_payload_idx"])] = result
                self.buffers.task[node.uid] = {"inp_grads": inp_grads}
        elif result is not None:
            raise RuntimeError(
                f"Routed backward source node {node.uid} unexpectedly received "
                "a tensor"
            )

        event = torch.cuda.Event()
        event.record(stream)
        self.events.a2a[node.uid] = event

    @staticmethod
    def _dense_state_model_entries(
        meta: dict[str, Any],
        *,
        differentiable_only: bool,
    ) -> list[tuple[int, int, dict[str, Any]]]:
        source_indices = meta.get("model_tensor_indices")
        destination_indices = meta.get("destination_input_indices")
        schemas = meta.get("model_value_schemas")
        differentiable = set(meta.get("differentiable_model_indices", ()))
        if not (
            isinstance(source_indices, list)
            and isinstance(destination_indices, list)
            and isinstance(schemas, list)
            and len(source_indices) == len(destination_indices) == len(schemas)
        ):
            raise ValueError("invalid dense-state model slot metadata")
        entries = [
            (int(source_idx), int(destination_idx), schema)
            for source_idx, destination_idx, schema in zip(
                source_indices, destination_indices, schemas
            )
            if not differentiable_only or source_idx in differentiable
        ]
        if any(not isinstance(schema, dict) for _src, _dst, schema in entries):
            raise ValueError("dense-state model schemas must be mappings")
        return entries

    def _send_dense_state(
        self,
        node: Any,
        stream: torch.cuda.Stream,
    ) -> _P2PLaunch:
        """Send model bypass values plus Piper-private counts, or gradients."""
        meta = self._node_meta(node)
        direction = meta.get("direction")
        state = self._routed_cross_group_state(node)
        if direction == "forward":
            pred = next(
                predecessor
                for predecessor in node.data_preds
                if predecessor.task_type == TaskType.FWD
            )
            buffer = self.buffers.task[pred.uid]
            detached_outs = list(buffer["detached_outs"])
            counts = detached_outs[int(meta["routing_counts_idx"])]
            if not isinstance(counts, torch.Tensor):
                raise TypeError(
                    f"dense-state SEND {node.uid} route counts are not a tensor"
                )
            payload = detached_outs[int(meta["dispatch_payload_idx"])]
            if not isinstance(payload, torch.Tensor) or payload.ndim != 2:
                raise TypeError(
                    f"dense-state SEND {node.uid} dispatch payload must be a "
                    "rank-2 tensor"
                )
            payload_shape = torch.tensor(
                tuple(payload.shape),
                dtype=torch.int64,
                device=payload.device,
            )
            if payload_shape.is_cuda:
                payload_shape_ready = torch.cuda.Event()
                payload_shape_ready.record(
                    torch.cuda.current_stream(payload_shape.device)
                )
                stream.wait_event(payload_shape_ready)
            tensors = [counts, payload_shape]
            for source_idx, _destination_idx, _schema in (
                self._dense_state_model_entries(
                    meta, differentiable_only=False
                )
            ):
                tensor = detached_outs[source_idx]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(
                        f"dense-state SEND {node.uid} output slot {source_idx} "
                        "is not a tensor"
                    )
                tensors.append(tensor)
            state.dispatch_counts = counts
            state.payload_shape = tuple(int(dim) for dim in payload.shape)
            state.payload_dtype = payload.dtype
        elif direction == "backward":
            pred = next(
                predecessor
                for predecessor in node.data_preds
                if predecessor.task_type in (TaskType.BWD, TaskType.BWD_I)
            )
            buffer = self.buffers.task[pred.uid]
            inp_grads = list(buffer["inp_grads"])
            tensors = []
            for _source_idx, destination_idx, _schema in (
                self._dense_state_model_entries(
                    meta, differentiable_only=True
                )
            ):
                tensor = inp_grads[destination_idx]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        f"dense-state SEND {node.uid} has no gradient for "
                        f"destination input slot {destination_idx}"
                    )
                tensors.append(tensor)
        else:
            raise ValueError(
                f"dense-state SEND {node.uid} has invalid direction={direction!r}"
            )
        launch = self.communication.send_sized_tensors(
            tensors,
            node.peer_pp_rank,
            stream,
            task=node,
        )
        self.buffers.release(pred.uid)
        return launch

    def _recv_dense_state(
        self,
        node: Any,
        stream: torch.cuda.Stream,
    ) -> _P2PLaunch:
        """Receive a direct dense-state envelope and map it to FX slots."""
        meta = self._node_meta(node)
        direction = meta.get("direction")
        state = self._routed_cross_group_state(node)
        entries = self._dense_state_model_entries(
            meta, differentiable_only=direction == "backward"
        )
        schemas = [entry[2] for entry in entries]
        if direction == "forward":
            counts_schema = meta.get("routing_counts_schema")
            if not isinstance(counts_schema, dict):
                raise ValueError(
                    f"dense-state RECV {node.uid} lacks routing-count schema"
                )
            schemas = [
                counts_schema,
                {
                    "rank": 1,
                    "shape": [2],
                    "dtype": "torch.int64",
                    "requires_grad": False,
                },
                *schemas,
            ]
        elif direction != "backward":
            raise ValueError(
                f"dense-state RECV {node.uid} has invalid direction={direction!r}"
            )
        launch = self.communication.recv_sized_tensors(
            schemas,
            node.peer_pp_rank,
            stream,
            task=node,
        )
        tensors = launch.result
        if direction == "forward":
            counts, payload_shape, *model_tensors = tensors
            if payload_shape.is_cuda:
                payload_shape_ready = torch.cuda.Event()
                payload_shape_ready.record(stream)
                torch.cuda.current_stream(payload_shape.device).wait_event(
                    payload_shape_ready
                )
            if tuple(counts.shape) != (1, state.num_global_routes):
                raise ValueError(
                    f"dense-state RECV {node.uid} got route counts shape="
                    f"{tuple(counts.shape)}"
                )
            if tuple(payload_shape.shape) != (2,) or bool(
                torch.any(payload_shape <= 0).item()
            ):
                raise ValueError(
                    f"dense-state RECV {node.uid} got invalid routed payload "
                    f"shape descriptor={payload_shape.tolist()}"
                )
            state.combine_counts = counts
            state.payload_shape = tuple(int(dim) for dim in payload_shape.tolist())
            result = {
                "routing_counts": counts,
                "runtime_inputs": {
                    destination_idx: tensor
                    for (_source_idx, destination_idx, _schema), tensor in zip(
                        entries, model_tensors
                    )
                },
            }
            return _P2PLaunch(result, launch.works, launch.tensors)

        inp_grads: list[Any | None] = [None] * int(
            meta["dispatch_output_arity"]
        )
        for (source_idx, _destination_idx, _schema), tensor in zip(
            entries, tensors
        ):
            inp_grads[source_idx] = tensor
        return _P2PLaunch(
            {"inp_grads": inp_grads},
            launch.works,
            launch.tensors,
        )

    def prepare(
        self,
        dag: Any,
        sorted_dag_nodes: list[Any] | tuple[Any, ...],
        *,
        stream_override: StreamOverride | None = None,
    ) -> PreparedDagExecution:
        """Build immutable-per-DAG execution state once during actor setup."""
        assert dag is not None, "cannot prepare a missing TrainingDAG"
        assert sorted_dag_nodes is not None, "cannot prepare a missing node order"

        nodes = tuple(sorted_dag_nodes)
        execution_plan = tuple(
            build_execution_plan(dag, [node.uid for node in nodes])
        )
        planned_by_uid = {
            operation.uid: operation for operation in execution_plan
        }
        physical_stream_by_uid = self._physical_streams(
            nodes,
            stream_override=stream_override,
        )
        return PreparedDagExecution(
            dag=dag,
            sorted_dag_nodes=nodes,
            execution_plan=execution_plan,
            planned_by_uid=planned_by_uid,
            physical_stream_by_uid=physical_stream_by_uid,
            initial_wait_streams=self._initial_wait_streams(
                physical_stream_by_uid
            ),
            buffer_refcount_template=self._buffer_refcount_template(dag),
        )

    def remap_streams(
        self,
        prepared: PreparedDagExecution,
        *,
        stream_override: StreamOverride | None = None,
    ) -> PreparedDagExecution:
        """Prepare another stream policy without rebuilding the static plan."""
        physical_stream_by_uid = self._physical_streams(
            prepared.sorted_dag_nodes,
            stream_override=stream_override,
        )
        return PreparedDagExecution(
            dag=prepared.dag,
            sorted_dag_nodes=prepared.sorted_dag_nodes,
            execution_plan=prepared.execution_plan,
            planned_by_uid=prepared.planned_by_uid,
            physical_stream_by_uid=physical_stream_by_uid,
            initial_wait_streams=self._initial_wait_streams(
                physical_stream_by_uid
            ),
            buffer_refcount_template=prepared.buffer_refcount_template,
        )

    def _physical_streams(
        self,
        sorted_dag_nodes: tuple[Any, ...],
        *,
        stream_override: StreamOverride | None,
    ) -> dict[str, torch.cuda.Stream]:
        physical_stream_by_uid: dict[str, torch.cuda.Stream] = {}
        for node in sorted_dag_nodes:
            logical_stream = self.runtime.stream_for_task(node)
            physical_stream_by_uid[node.uid] = (
                logical_stream
                if stream_override is None
                else stream_override(node, logical_stream)
            )
        return physical_stream_by_uid

    def _initial_wait_streams(
        self,
        physical_stream_by_uid: dict[str, torch.cuda.Stream],
    ) -> tuple[torch.cuda.Stream, ...]:
        """Deduplicate invariant prologue wait targets by object identity."""
        default_stream = self.runtime.default_stream()
        result: list[torch.cuda.Stream] = []
        for stream in (
            *self.runtime.streams.values(),
            *physical_stream_by_uid.values(),
        ):
            if stream is default_stream:
                continue
            if any(stream is seen for seen in result):
                continue
            result.append(stream)
        return tuple(result)

    @staticmethod
    def _buffer_refcount_template(dag: Any) -> dict[Any, int]:
        result: dict[Any, int] = {}
        for edge in dag.edges:
            if edge.dep_kind == "data":
                result[edge.src_uid] = result.get(edge.src_uid, 0) + 1
        return result

    def run(
        self,
        prepared: PreparedDagExecution,
        inputs: Any,
        labels: Any,
        loss_buffer: list,
        loss_fn=None,
        *,
        timing_recorder: TimingRecorder | None = None,
        serialize_nodes: bool = False,
    ) -> None:
        """Run one iteration using state prepared outside the hot loop.

        When ``serialize_nodes`` is true, each node waits for a CUDA event
        recorded after the preceding node. The wait and handoff-event record
        occur outside the node's profiling range, so CUPTI timing contains
        only the node's own work while retaining its intended logical stream.
        """
        assert prepared is not None, (
            "load_training_dag() must prepare execution before run_dag()"
        )
        dag = prepared.dag
        sorted_dag_nodes = prepared.sorted_dag_nodes
        planned_by_uid = prepared.planned_by_uid
        physical_stream_by_uid = prepared.physical_stream_by_uid

        self.params.drain_pending_frees()
        self.communication.reap_completed_p2p()
        debug_enabled = self.logger.isEnabledFor(logging.DEBUG)

        self.buffers.reset()
        self.events.reset()
        self.params.clear_param_grads()
        default_stream = self.runtime.default_stream()
        if timing_recorder is not None:
            timing_recorder.before_prologue(default_stream)
        self.params.zero_grad_buffers(default_stream)
        if timing_recorder is not None:
            timing_recorder.after_prologue(default_stream)
        zero_evt = torch.cuda.Event()
        zero_evt.record(default_stream)

        # An override may return streams not registered in RuntimeState.streams
        # (for example dedicated directional P2P streams). Gate each distinct
        # physical stream on the prologue too.
        for stream in prepared.initial_wait_streams:
            stream.wait_event(zero_evt)
        completion_events: dict[str, torch.cuda.Event] = {}
        completion_streams: dict[str, torch.cuda.Stream] = {}
        serialization_event: torch.cuda.Event | None = None
        serialization_stream: torch.cuda.Stream | None = None
        update_result = None

        self.buffers.init_refcounts(prepared.buffer_refcount_template)

        for node in sorted_dag_nodes:
            p2p_launch: _P2PLaunch | None = None
            task_type = node.task_type
            batch = node.batches[0]
            mb_idx = batch.mb_idx
            ubid = self._node_bucket_key(node)
            node_stream = physical_stream_by_uid[node.uid]
            node_tag = self._node_tag_str(node)

            # Preserve each node's intended physical stream while forcing the
            # isolated profiling policy to execute one node at a
            # time.  This wait is intentionally before the profiling range.
            if (
                serialize_nodes
                and serialization_event is not None
                and serialization_stream is not node_stream
            ):
                node_stream.wait_event(serialization_event)

            operation = planned_by_uid[node.uid]
            for predecessor_uid in operation.completion_predecessors:
                predecessor_event = completion_events.get(predecessor_uid)
                if predecessor_event is None:
                    raise RuntimeError(
                        f"planned predecessor {predecessor_uid!r} for node "
                        f"{node.uid!r} has no completion event"
                    )
                predecessor_stream = completion_streams[predecessor_uid]
                if predecessor_stream is not node_stream:
                    node_stream.wait_event(predecessor_event)

            if debug_enabled:
                self.logger.debug(
                    f"run_dag dispatch: {task_type.value} "
                    f"tag={node_tag}"
                )

            task_label = f"{task_type.value}:{node_tag}:uid{node.uid}"
            self.runtime.nvtx_push(task_label)
            rf = self._rf_enter(task_label)
            if timing_recorder is not None:
                timing_recorder.before_node(node, node_stream)

            match task_type:
                case TaskType.SEND:
                    if self._node_meta(node).get("dense_state_transfer"):
                        p2p_launch = self._send_dense_state(node, node_stream)
                    else:
                        compute_node = node.data_preds[0]
                        send_data = self.buffers.task[compute_node.uid]["send_output"]
                        p2p_launch = self.communication.send(
                            send_data,
                            node.peer_pp_rank,
                            stream=node_stream,
                            task=node,
                        )
                        send_buf = self.buffers.task.get(compute_node.uid)
                        if isinstance(send_buf, dict):
                            send_buf["send_output"] = None
                        self.buffers.release(compute_node.uid)

                case TaskType.RECV:
                    node_meta = self._node_meta(node)
                    if node_meta.get("dense_state_transfer"):
                        p2p_launch = self._recv_dense_state(
                            node, node_stream
                        )
                        self.buffers.task[node.uid] = p2p_launch.result
                    else:
                        receive_kind = node_meta.get("receive_kind")
                        if receive_kind == "forward":
                            recv_ubid = self._node_bucket_key(node)
                            p2p_launch = self.communication.recv_fwd(
                                recv_ubid,
                                node.peer_pp_rank,
                                stream=node_stream,
                                task=node,
                            )
                            self.buffers.task[node.uid] = p2p_launch.result
                        elif receive_kind == "backward":
                            fwd_uid = node_meta.get("shape_reference_fwd_uid")
                            if not isinstance(fwd_uid, str) or not fwd_uid:
                                raise RuntimeError(
                                    f"backward RECV {node.uid} lacks an explicit "
                                    "shape_reference_fwd_uid"
                                )
                            fwd_key = (self._node_bucket_key(node), fwd_uid)
                            shape_meta = self.buffers.task[("shape_ref",) + fwd_key]
                            p2p_launch = self.communication.recv_bwd(
                                shape_meta,
                                node.peer_pp_rank,
                                stream=node_stream,
                                task=node,
                            )
                            self.buffers.task[node.uid] = p2p_launch.result
                        else:
                            raise RuntimeError(
                                f"pipeline RECV {node.uid} has invalid "
                                f"receive_kind={receive_kind!r}"
                            )
                    recv_evt = torch.cuda.Event()
                    recv_evt.record(node_stream)
                    self.events.recv[node.uid] = recv_evt

                case TaskType.FWD_A2A:
                    fwd_pred = next(p for p in node.data_preds if p.task_type == TaskType.FWD)
                    node_meta = self._node_meta(node)
                    tensor_idx = node_meta["a2a_tensor_idx"]
                    fwd_buf = dict(self.buffers.task[fwd_pred.uid])
                    self.buffers.release(fwd_pred.uid)
                    detached_outs = list(fwd_buf["detached_outs"])
                    direction = node_meta["direction"]
                    counts_idx = (
                        node_meta["routing_counts_idx"]
                        if direction == "incoming"
                        else None
                    )
                    routing_counts = (
                        detached_outs[counts_idx] if counts_idx is not None else None
                    )
                    a2a_result = self._all_to_all_ep_boundary(
                        node,
                        detached_outs[tensor_idx],
                        node_stream,
                        routing_counts=routing_counts,
                    )
                    if direction == "incoming":
                        if not isinstance(a2a_result, tuple):
                            raise RuntimeError(
                                f"Forward dispatch node {node.uid} did not return "
                                "local routing counts"
                            )
                        a2a_tensor, local_counts = a2a_result
                        assert counts_idx is not None
                        detached_outs[counts_idx] = local_counts
                    else:
                        if isinstance(a2a_result, tuple):
                            raise RuntimeError(
                                f"Forward combine node {node.uid} unexpectedly "
                                "returned routing counts"
                            )
                        a2a_tensor = a2a_result
                    detached_outs[tensor_idx] = a2a_tensor.requires_grad_(True)
                    fwd_buf["detached_outs"] = detached_outs
                    runtime_mapping = node_meta.get("runtime_value_mapping")
                    if isinstance(runtime_mapping, list):
                        fwd_buf["runtime_inputs"] = {
                            int(item["destination_input_idx"]): detached_outs[
                                int(item["source_output_idx"])
                            ]
                            for item in runtime_mapping
                        }
                    self.buffers.task[node.uid] = fwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(node_stream)
                    self.events.a2a[node.uid] = a2a_evt

                case TaskType.BWD_A2A:
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    node_meta = self._node_meta(node)
                    tensor_idx = int(node_meta["a2a_tensor_idx"])
                    bwd_buf = dict(self.buffers.task[bwd_pred.uid])
                    self.buffers.release(bwd_pred.uid)
                    inp_grads = list(bwd_buf["inp_grads"])
                    runtime_mapping = node_meta.get("runtime_value_mapping")
                    routed_mapping = None
                    if isinstance(runtime_mapping, list):
                        routed_mapping = next(
                            (
                                item
                                for item in runtime_mapping
                                if int(item["source_output_idx"]) == tensor_idx
                            ),
                            None,
                        )
                    source_grad_idx = (
                        int(routed_mapping["destination_input_idx"])
                        if routed_mapping is not None
                        else tensor_idx
                    )
                    grad_a2a_out = inp_grads[source_grad_idx]
                    assert grad_a2a_out is not None, (
                        f"BWD_A2A tag={node_tag}: grad at consumer input "
                        f"index={source_grad_idx} is None"
                    )
                    routed_grad = self._all_to_all_ep_boundary(
                        node, grad_a2a_out, node_stream
                    )
                    if routed_mapping is not None:
                        producer_grads = [None] * int(
                            node_meta["producer_output_arity"]
                        )
                        producer_grads[tensor_idx] = routed_grad
                        bwd_buf["inp_grads"] = producer_grads
                    else:
                        inp_grads[tensor_idx] = routed_grad
                        bwd_buf["inp_grads"] = inp_grads
                    self.buffers.task[node.uid] = bwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(node_stream)
                    self.events.a2a[node.uid] = a2a_evt

                case TaskType.FWD_M_TO_N | TaskType.FWD_N_TO_M:
                    self._run_forward_routed_cross_group(node, node_stream)

                case TaskType.BWD_M_TO_N | TaskType.BWD_N_TO_M:
                    self._run_backward_routed_cross_group(node, node_stream)

                case TaskType.ALL_REDUCE:
                    bwd_node = node.data_preds[0]
                    ar_ubids = self._sync_payload_ubids(node)
                    assert ar_ubids, (
                        f"ALL_REDUCE node uid={node.uid} has no sync_payload_ubids"
                    )
                    for ar_ubid in ar_ubids:
                        self.communication.all_reduce_grads(ar_ubid, stream=node_stream)
                    ar_evt = torch.cuda.Event()
                    ar_evt.record(node_stream)
                    self.events.all_reduce[node.uid] = ar_evt
                    self.buffers.release(bwd_node.uid)

                case TaskType.REDUCE_SCATTER:
                    bwd_node = node.data_preds[0]
                    rs_ubid = self._node_bucket_key(node)
                    assert rs_ubid is not None, (
                        f"REDUCE_SCATTER node uid={node.uid} has no bucket_key"
                    )
                    rs_bytes = self.communication.reduce_scatter(rs_ubid, stream=node_stream)
                    rs_evt = torch.cuda.Event()
                    rs_evt.record(node_stream)
                    self.events.reduce_scatter[node.uid] = rs_evt
                    if rs_bytes:
                        self.params.defer_free_full_grads(rs_ubid, rs_evt)
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        assert False and "Param free should happen after a compute node"
                    self.buffers.release(bwd_node.uid)

                case TaskType.ALL_GATHER:
                    ag_ubid = self._node_bucket_key(node)
                    assert ag_ubid is not None, (
                        f"ALL_GATHER node uid={node.uid} has no bucket_key"
                    )
                    self.params.all_gather_full_params(ag_ubid, stream=node_stream)
                    ag_evt = torch.cuda.Event()
                    ag_evt.record(node_stream)
                    self.events.all_gather[node.uid] = ag_evt

                case TaskType.FWD:
                    for pred in node.data_preds:
                        if pred.task_type == TaskType.RECV:
                            self.events.recv.pop(pred.uid, None)
                        elif pred.task_type in {
                            TaskType.FWD_A2A,
                            TaskType.FWD_M_TO_N,
                            TaskType.FWD_N_TO_M,
                        }:
                            self.events.a2a.pop(pred.uid, None)

                    input_tensors = self._forward_inputs_for_node(
                        node, ubid, inputs
                    )

                    fwd_out = self.compute.forward(ubid, input_tensors, node_stream)
                    self.buffers.task[node.uid] = fwd_out
                    fwd_key = (node.node_meta.get("bucket_key"), node.uid)
                    self.buffers.task[("shape_ref",) + fwd_key] = [
                        (t.shape, t.dtype) for t in fwd_out["out_with_grad"]
                    ]
                    self.buffers.task[fwd_key] = fwd_out
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD:
                    self._wait_for_backward_input_events(
                        node, node_stream, wait=False
                    )

                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.buffers.task[fwd_key]

                    if self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.compute.log_compute_loss_inputs(labels, node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
                            loss = loss_fn(fwd_out["out_with_grad"][0], labels)
                            outputs_or_loss = [loss]
                            loss_buffer.append(loss.detach())
                        upstream_grads = None
                    else:
                        outputs_or_loss, upstream_grads = (
                            self._backward_outputs_and_grads(node, fwd_out)
                        )
                    pre_detach_outs = None
                    detached_outs = None

                    inp_with_grad = fwd_out.get("inp_with_grad")
                    if self._node_meta(node).get("zero_alloc_full_grads_before"):
                        self.params.alloc_full_grads(ubid, node_stream)

                    bwd_out = self.compute.backward(
                        ubid, mb_idx, outputs_or_loss, upstream_grads,
                        pre_detach_outs, detached_outs, inp_with_grad,
                        fwd_out.get("out_with_grad"),
                        node_stream,
                    )
                    buf = bwd_out if bwd_out is not None else {}
                    fwd_inputs_full = fwd_out.get("fwd_inputs")
                    buf["inp_grads"] = (
                        [t.grad if (t is not None and t.requires_grad) else None
                         for t in fwd_inputs_full]
                        if fwd_inputs_full is not None
                        else [t.grad for t in (inp_with_grad or [])]
                    )
                    self.params.accumulate_zero_param_grads_to_flat(ubid, node_stream)
                    self.buffers.task[node.uid] = buf
                    fwd_out.clear()
                    del self.buffers.task[fwd_key]
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    self.events.backward[ubid] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD_I:
                    self._wait_for_backward_input_events(
                        node, node_stream, wait=False
                    )

                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.buffers.task[fwd_key]

                    if self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.compute.log_compute_loss_inputs(labels, node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
                            loss = loss_fn(fwd_out["out_with_grad"][0], labels)
                            stage_outputs_or_loss = [loss]
                            loss_buffer.append(loss.detach())
                        output_grads = None
                    else:
                        stage_outputs_or_loss, output_grads = (
                            self._backward_outputs_and_grads(
                                node, fwd_out
                            )
                        )

                    input_values = fwd_out.get("inp_with_grad") or []
                    weights = self.stages.bucket(ubid).weights()

                    with torch.cuda.stream(node_stream):
                        dinputs, param_groups = self.compute.bucket_backward_input(
                            stage_outputs_or_loss, output_grads, input_values, iter(weights)
                        )
                    fwd_inputs_full = fwd_out.get("fwd_inputs")
                    if fwd_inputs_full is not None:
                        inp_grads_full = [
                            t.grad if (t is not None and t.requires_grad) else None
                            for t in fwd_inputs_full
                        ]
                    else:
                        inp_grads_full = list(dinputs)
                    self.buffers.task[node.uid] = {
                        "inp_grads": inp_grads_full,
                        "param_groups": param_groups,
                    }
                    if dinputs:
                        self.buffers.task[node.uid]["send_output"] = list(dinputs)
                    fwd_out.clear()
                    del stage_outputs_or_loss, fwd_out
                    del self.buffers.task[fwd_key]
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    self.events.backward[ubid] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD_W:
                    if self._node_meta(node).get("zero_alloc_full_grads_before"):
                        self.params.alloc_full_grads(ubid, node_stream)
                    bwdi_node = next(p for p in node.data_preds if p.task_type == TaskType.BWD_I)
                    bwdi_buf = self.buffers.task[bwdi_node.uid]
                    param_groups = bwdi_buf["param_groups"]
                    weights = self.stages.bucket(ubid).weights()
                    with torch.cuda.stream(node_stream):
                        self.compute.bucket_backward_weight(
                            iter(weights), param_groups, ubid=ubid, mb_idx=mb_idx
                        )
                    self.params.accumulate_zero_param_grads_to_flat(ubid, node_stream)
                    self.buffers.task[node.uid] = {}
                    self.buffers.release(bwdi_node.uid)
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    self.events.backward[ubid] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.UPD:
                    update_result = self._update(
                        node_stream,
                        loss_buffer,
                        wait_for_dependencies=False,
                    )

                case TaskType.ORDER_DUMMY:
                    pass

            if timing_recorder is not None:
                timing_recorder.after_node(node, node_stream)
            completion_event = torch.cuda.Event()
            completion_event.record(node_stream)
            completion_events[node.uid] = completion_event
            completion_streams[node.uid] = node_stream
            if p2p_launch is not None:
                self.communication.retain_p2p_launch(
                    p2p_launch,
                    completion_event,
                )
            self._rf_exit(rf)
            self.runtime.nvtx_pop()

            # Keep this profiling-only handoff outside the node range.  The
            # following node waits on it before opening its own range.
            if serialize_nodes:
                serialization_event = torch.cuda.Event()
                serialization_event.record(node_stream)
                serialization_stream = node_stream

        if timing_recorder is not None:
            # CUDA event elapsed times are valid only after their work has
            # completed.  Synchronize once per iteration, never per node: a
            # per-node sync can deadlock rendezvous communication.
            torch.cuda.synchronize()
            timing_recorder.finish()
        return update_result

    def _update(
        self,
        stream: torch.cuda.Stream,
        loss_buffer: list,
        *,
        wait_for_dependencies: bool = True,
    ):
        self.params.drain_pending_frees()
        has_zero_shard_optimizers = self.params.has_zero_shard_optimizers()

        if wait_for_dependencies:
            for ar_evt in self.events.all_reduce.values():
                stream.wait_event(ar_evt)

        if has_zero_shard_optimizers:
            self.params.step_zero_shard_optimizers(
                stream,
                self.events.reduce_scatter if wait_for_dependencies else {},
            )

        for ubid, bucket in self.stages.buckets.items():
            if bucket.optimizer is None:
                continue
            bwd_evt = self.events.backward.get(ubid)
            if wait_for_dependencies and bwd_evt is not None:
                stream.wait_event(bwd_evt)

            with torch.cuda.stream(stream):
                bucket.optimizer.step()

        losses = list(loss_buffer)
        loss_buffer.clear()

        if has_zero_shard_optimizers:
            return losses

        return {
            "losses": losses,
        }
