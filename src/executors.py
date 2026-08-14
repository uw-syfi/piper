from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter


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


from .backward import construct_reverse_graph, get_param_groups, _get_grad_fn_or_grad_acc
from .execution_plan import (
    PlannedOperation,
    build_execution_plan,
    is_relaxable_p2p_ordering_dependency,
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
    """Per-microbatch layout state for one cross-ActorGroup expert boundary."""

    source_devices: tuple[int, ...]
    destination_devices: tuple[int, ...]
    shard_devices: tuple[int, ...]
    num_global_routes: int
    route_partitions: tuple[tuple[int, int], ...]
    participants: tuple[int, ...]
    layout_group: Any
    m_to_n_group: Any
    n_to_m_group: Any
    layout_work: Any = None
    layout_output: torch.Tensor | None = None
    input_splits: list[int] | None = None
    output_splits: list[int] | None = None
    payload_shape: tuple[int, ...] | None = None
    payload_dtype: torch.dtype | None = None
    retained_forward_buffer: dict[str, Any] | None = None
    retained_backward_buffer: dict[str, Any] | None = None


@dataclass
class CommunicationExecutor:
    """Actor-local communication operations used by the DAG dispatcher."""

    runtime: RuntimeState
    stages: StageStore
    logger: Any

    def send(
        self,
        send_data: Any,
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any | None = None,
    ) -> None:
        global_dst_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        with torch.cuda.stream(stream):
            tensors = send_data if isinstance(send_data, (list, tuple)) else [send_data]
            if task is None:
                raise ValueError("ActorGroup P2P send requires its DAG task")
            pp_group = self.runtime.p2p_process_group_for_task(task)
            for tensor in tensors:
                dist.send(tensor, dst=global_dst_rank, group=pp_group)

    def recv_fwd(
        self,
        recv_ubid: Any,
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any | None = None,
    ) -> list:
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
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def recv_bwd(
        self,
        shape_meta: list,
        peer_pp_rank: int,
        stream: torch.cuda.Stream,
        *,
        task: Any | None = None,
    ) -> list:
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        buf = [
            torch.empty(shape, dtype=dtype, device=self.runtime.device)
            for shape, dtype in shape_meta
        ]
        with torch.cuda.stream(stream):
            if task is None:
                raise ValueError("ActorGroup P2P receive requires its DAG task")
            pp_group = self.runtime.p2p_process_group_for_task(task)
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

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
        with torch.cuda.stream(stream):
            total_bytes = flat_grads.numel() * flat_grads.element_size()
            tmp = torch.empty_like(rs_out)
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

    def fused_backward(self, stage_outputs_or_loss: list, output_grads: Any, weights: list) -> None:
        wts = [w for w in weights if w.requires_grad]
        if not wts:
            return
        dweights = torch.autograd.grad(
            stage_outputs_or_loss,
            inputs=wts,
            grad_outputs=output_grads,
            retain_graph=False,
            allow_unused=True,
        )
        for w, dw in zip(wts, dweights):
            if dw is None:
                continue
            dw = self.grad_with_param_layout(w, dw)
            if w.grad is None:
                w.grad = dw
            else:
                grad_dtype = (
                    w.grad_dtype
                    if getattr(w, "grad_dtype", None) is not None
                    else w.dtype
                )
                if (
                    w.grad.dtype != grad_dtype
                    or w.grad.device != w.device
                    or w.grad.layout != w.layout
                    or tuple(w.grad.stride()) != tuple(w.stride())
                ):
                    w.grad = self.grad_with_param_layout(w, w.grad)
                w.grad += dw

    def backward_weight_from_outputs(
        self,
        stage_outputs_or_loss: list,
        output_grads: Any,
        weights: Any,
    ) -> None:
        self.fused_backward(stage_outputs_or_loss, output_grads, list(weights))

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
            self.fused_backward(stage_outputs_or_loss, output_grads, weights)
            for i, t in enumerate(stage_outputs_or_loss):
                if isinstance(t, torch.Tensor):
                    stage_outputs_or_loss[i] = t.detach()
            return (), [], None

        stage_output_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, stage_outputs_or_loss)))
        stage_input_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, input_values)))
        weight_grad_fns = list(filter(None, map(_get_grad_fn_or_grad_acc, weights)))

        reverse_edges_dict = construct_reverse_graph(stage_output_grad_fns)
        param_groups = get_param_groups(stage_input_grad_fns, weight_grad_fns, reverse_edges_dict)
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

        output_backward_ctx = {
            "stage_outputs_or_loss": list(stage_outputs_or_loss),
            "output_grads": output_grads,
        }

        for handle in handles:
            handle.remove()

        return dinputs, param_groups, output_backward_ctx

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

            dweights = torch.autograd.grad(
                valid_edges,
                weight_edges,
                grad_outputs=valid_grad_outputs,
                retain_graph=False,
            )

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
                grouped_counts = all_to_all_single(
                    counts_for_collective,
                    None,
                    None,
                    group=self.runtime.actor_group_routed_group,
                )
                grouped_counts = torch.ops._c10d_functional.wait_tensor(
                    grouped_counts
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

            collective_output = all_to_all_single_autograd(
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
            tensor = all_to_all_single_autograd(
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
            tensor = all_to_all_single_autograd(
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
        """Find or initialize the shared dispatch/combine layout state."""
        meta = self._node_meta(node)
        expert_fwd_uid = meta.get("expert_fwd_uid")
        if not isinstance(expert_fwd_uid, str):
            raise ValueError(
                f"Routed node {getattr(node, 'uid', '<unknown>')} is missing "
                "expert_fwd_uid"
            )

        state = self.routed_cross_group_states.get(expert_fwd_uid)
        if state is not None:
            return state

        if meta.get("direction") != "incoming" or getattr(
            node, "task_type", None
        ) != TaskType.FWD_M_TO_N:
            raise RuntimeError(
                f"Routed node {getattr(node, 'uid', '<unknown>')} ran before "
                f"forward M-to-N dispatch for expert {expert_fwd_uid}"
            )

        source_devices = tuple(int(d) for d in meta["source_devices"])
        destination_devices = tuple(int(d) for d in meta["destination_devices"])
        if not source_devices or not destination_devices:
            raise ValueError(
                f"Routed node {node.uid} requires non-empty source and "
                "destination device sets"
            )
        if set(source_devices) & set(destination_devices):
            raise ValueError(
                f"Routed node {node.uid} requires disjoint groups, got "
                f"source={source_devices}, destination={destination_devices}"
            )
        if source_devices != tuple(sorted(source_devices)):
            raise ValueError(
                "source_devices must be in ascending global actor-rank order, "
                f"got {source_devices}"
            )
        if destination_devices != tuple(sorted(destination_devices)):
            raise ValueError(
                "destination_devices must be in ascending global actor-rank "
                f"order, got {destination_devices}"
            )
        participants = tuple(sorted((*source_devices, *destination_devices)))
        (
            num_global_routes,
            shard_devices,
            route_partitions,
        ) = _route_partition_metadata(meta, destination_devices)
        state = _RoutedCrossGroupState(
            source_devices=source_devices,
            destination_devices=destination_devices,
            shard_devices=shard_devices,
            num_global_routes=num_global_routes,
            route_partitions=route_partitions,
            participants=participants,
            layout_group=self.runtime.process_group_for_participants(
                participants, channel="routed_layout"
            ),
            m_to_n_group=self.runtime.process_group_for_participants(
                participants, channel="m_to_n"
            ),
            n_to_m_group=self.runtime.process_group_for_participants(
                participants, channel="n_to_m"
            ),
        )
        self.routed_cross_group_states[expert_fwd_uid] = state
        return state

    def _empty_routed_input(
        self,
        node: Any,
        input_idx: int,
        *,
        leading_dim: int | None = None,
    ) -> torch.Tensor:
        """Allocate an actor-local placeholder from the expert input schema."""
        bucket_key = self._node_meta(node).get("bucket_key")
        if bucket_key is None:
            raise ValueError(f"Routed node {node.uid} is missing bucket_key")
        input_meta = self.stages.bucket(bucket_key).forward_input_meta
        if input_idx < 0 or input_idx >= len(input_meta):
            raise ValueError(
                f"Routed node {node.uid} input index {input_idx} is outside "
                f"the expert input schema of length {len(input_meta)}"
            )
        shape, dtype, _requires_grad = input_meta[input_idx]
        shape = tuple(shape)
        if leading_dim is not None:
            if not shape:
                raise ValueError(
                    f"Routed input index {input_idx} for node {node.uid} is scalar"
                )
            shape = (leading_dim, *shape[1:])
        return torch.empty(shape, dtype=dtype, device=self.runtime.device)

    def _empty_routed_payload(
        self,
        node: Any,
        state: _RoutedCrossGroupState,
        tensor_idx: int,
    ) -> torch.Tensor:
        """Allocate a zero-token payload on either side of the boundary."""
        if state.payload_shape is not None and state.payload_dtype is not None:
            return torch.empty(
                (0, *state.payload_shape[1:]),
                dtype=state.payload_dtype,
                device=self.runtime.device,
            )
        # The expert-side actor owns the destination bucket and can recover
        # the schema locally before it has seen a payload.
        return self._empty_routed_input(node, tensor_idx, leading_dim=0)

    @staticmethod
    def _require_routed_cross_group_layout(
        state: _RoutedCrossGroupState,
    ) -> None:
        if (
            state.input_splits is None
            or state.output_splits is None
        ):
            raise RuntimeError(
                "M-to-N/N-to-M communication ran before forward dispatch "
                "established its split layout"
            )
    def _initialize_routed_cross_group_layout(
        self,
        node: Any,
        state: _RoutedCrossGroupState,
        routing_counts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Exchange routing counts and derive this rank's union-A2A splits."""
        meta = self._node_meta(node)
        role = meta.get("role")
        counts_idx = int(meta["routing_counts_idx"])
        if role == "source":
            if routing_counts is None:
                raise ValueError(
                    f"Forward M-to-N source {node.uid} is missing routing counts"
                )
            local_counts = routing_counts
        elif role == "destination":
            local_counts = self._empty_routed_input(node, counts_idx)
            local_counts.zero_()
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
                rank: index for index, rank in enumerate(state.participants)
            }
            input_splits = [0] * len(state.participants)
            output_splits = [0] * len(state.participants)
            global_rank = self.runtime.global_rank
            if global_rank in state.source_devices:
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
            elif global_rank not in state.destination_devices:
                raise ValueError(
                    f"Actor rank {global_rank} is not a participant in routed "
                    f"boundary {state.participants}"
                )

            gathered_counts = torch.empty(
                (len(state.participants), state.num_global_routes),
                dtype=local_counts.dtype,
                device=local_counts.device,
            )
            layout_work = dist.all_gather_into_tensor(
                gathered_counts,
                local_counts.contiguous(),
                group=state.layout_group,
                async_op=True,
            )
            grouped_counts: torch.Tensor | None = None
            if global_rank in state.source_devices:
                # Sources need only their local row to derive send splits. Keep
                # the asynchronous gather alive without waiting so a later
                # microbatch can be enqueued before the reverse routed transfer.
                state.layout_work = layout_work
                state.layout_output = gathered_counts
            else:
                # Destinations need every source row before they can derive
                # receive splits and compact source-major local-route counts.
                layout_work.wait()
                source_counts = torch.stack(
                    [
                        gathered_counts[participant_index[rank]]
                        for rank in state.source_devices
                    ]
                )
                destination_index = state.shard_devices.index(global_rank)
                start, stop = state.route_partitions[destination_index]
                grouped_counts = source_counts[:, start:stop].contiguous()
                for source_index, rank in enumerate(state.source_devices):
                    output_splits[participant_index[rank]] = int(
                        source_counts[source_index, start:stop].sum().item()
                    )
            state.input_splits = input_splits
            state.output_splits = output_splits

        return local_counts, grouped_counts

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
            if task_type == TaskType.FWD_M_TO_N:
                local_counts, grouped_counts = (
                    self._initialize_routed_cross_group_layout(
                        node, state, routing_counts
                    )
                )
            else:
                self._require_routed_cross_group_layout(state)
                grouped_counts = None

            if direction == "incoming":
                if role == "source":
                    if tensor is None:
                        raise ValueError(f"Routed source node {node.uid} has no tensor")
                    if task_type == TaskType.FWD_M_TO_N:
                        _validate_routed_payload_schema(
                            tensor,
                            None,
                            None,
                            context=f"Forward M-to-N source {node.uid}",
                        )
                        routed_tokens = int(local_counts.sum().item())
                        if tensor.ndim < 1 or int(tensor.shape[0]) != routed_tokens:
                            raise ValueError(
                                f"Forward M-to-N source {node.uid} has payload "
                                f"shape={tuple(tensor.shape)} but routing counts "
                                f"sum to {routed_tokens} tokens"
                            )
                        state.payload_shape = tuple(tensor.shape)
                        state.payload_dtype = tensor.dtype
                    else:
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
                collective_output = all_to_all_single_autograd(
                    collective_input,
                    state.output_splits,
                    state.input_splits,
                    state.m_to_n_group,
                )
                if role == "source":
                    return None
                if task_type == TaskType.FWD_M_TO_N:
                    assert grouped_counts is not None
                    state.payload_shape = tuple(collective_output.shape)
                    state.payload_dtype = collective_output.dtype
                    return collective_output, grouped_counts

                # Backward M-to-N carries the gradient of the forward combine.
                return collective_output

            # N-to-M is exactly the reverse of the saved dispatch layout.
            if role == "source":
                if tensor is None:
                    raise ValueError(f"Routed source node {node.uid} has no tensor")
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
            collective_output = all_to_all_single_autograd(
                collective_input,
                state.input_splits,
                state.output_splits,
                state.n_to_m_group,
            )
            if role != "destination":
                return None
            return collective_output

    def _run_forward_routed_cross_group(
        self,
        node: Any,
        stream: torch.cuda.Stream,
    ) -> None:
        meta = self._node_meta(node)
        role = meta["role"]
        direction = meta["direction"]
        tensor_idx = int(meta["a2a_tensor_idx"])
        state = self._routed_cross_group_state(node)

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
                state.retained_forward_buffer = fwd_buf

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
                inputs[tensor_idx] = routed_tensor.requires_grad_(True)
                inputs[int(meta["routing_counts_idx"])] = local_counts
                self.buffers.task[node.uid] = {"detached_outs": inputs}
            else:
                if not isinstance(result, torch.Tensor):
                    raise RuntimeError(
                        f"Forward N-to-M destination {node.uid} did not return "
                        "a routed tensor"
                    )
                retained = state.retained_forward_buffer
                if retained is None:
                    raise RuntimeError(
                        f"Forward N-to-M destination {node.uid} has no retained "
                        "dense-side buffer"
                    )
                output_arity = int(meta["expert_output_arity"])
                detached_outs = list(retained["detached_outs"][:output_arity])
                if len(detached_outs) != output_arity:
                    raise RuntimeError(
                        f"Forward N-to-M destination {node.uid} expected "
                        f"{output_arity} retained outputs, got {len(detached_outs)}"
                    )
                detached_outs[tensor_idx] = result.requires_grad_(True)
                retained["detached_outs"] = detached_outs
                # A cross-Attention-stage combine is followed by an ordinary
                # SEND/RECV relay.  The routed state owner has reconstructed
                # the complete dense boundary here, so expose that patched
                # boundary through the same buffer field used by generic P2P.
                retained["send_output"] = tuple(detached_outs)
                self.buffers.task[node.uid] = retained
                state.retained_forward_buffer = None
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
        tensor_idx = int(meta["a2a_tensor_idx"])
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
            if direction == "incoming":
                state.retained_backward_buffer = bwd_buf

        result = self._routed_cross_group_boundary(node, tensor, stream)

        if role == "destination":
            if not isinstance(result, torch.Tensor):
                raise RuntimeError(
                    f"Routed backward destination {node.uid} did not receive "
                    "a tensor gradient"
                )
            if direction == "incoming":
                inp_grads = [None] * int(meta["expert_output_arity"])
                inp_grads[tensor_idx] = result
                self.buffers.task[node.uid] = {"inp_grads": inp_grads}
            else:
                retained = state.retained_backward_buffer
                if retained is None:
                    raise RuntimeError(
                        f"Backward N-to-M destination {node.uid} has no retained "
                        "dense-side gradient buffer"
                    )
                input_arity = int(meta["expert_input_arity"])
                inp_grads = list(retained["inp_grads"][:input_arity])
                if len(inp_grads) < input_arity:
                    inp_grads.extend([None] * (input_arity - len(inp_grads)))
                inp_grads[tensor_idx] = result
                retained["inp_grads"] = inp_grads
                self.buffers.task[node.uid] = retained
                state.retained_backward_buffer = None
        elif result is not None:
            raise RuntimeError(
                f"Routed backward source node {node.uid} unexpectedly received "
                "a tensor"
            )

        event = torch.cuda.Event()
        event.record(stream)
        self.events.a2a[node.uid] = event

    def _recv_backward_dense_relay(
        self,
        node: Any,
        stream: torch.cuda.Stream,
    ) -> dict[str, list[Any | None]]:
        """Receive relay gradients and restore the dense boundary arity."""
        meta = self._node_meta(node)
        payload_arity = meta.get("payload_arity")
        tensor_indices = meta.get("backward_tensor_indices")
        shape_bucket_key = meta.get("shape_reference_bucket_key")
        shape_fwd_uid = meta.get("shape_reference_fwd_uid")
        if (
            isinstance(payload_arity, bool)
            or not isinstance(payload_arity, int)
            or payload_arity < 0
            or not isinstance(tensor_indices, list)
            or not isinstance(shape_fwd_uid, str)
            or shape_bucket_key is None
        ):
            raise ValueError(
                f"Backward dense relay {node.uid} has invalid "
                "payload/shape-reference metadata"
            )
        shape_key = ("shape_ref", shape_bucket_key, shape_fwd_uid)
        shape_meta = self.buffers.task.get(shape_key)
        if shape_meta is None:
            raise RuntimeError(
                f"Backward dense relay {node.uid} is missing shape reference "
                f"{shape_key!r}"
            )
        if len(shape_meta) != len(tensor_indices):
            raise RuntimeError(
                f"Backward dense relay {node.uid} expected "
                f"{len(tensor_indices)} gradient tensors from "
                f"indices={tensor_indices}, but shape reference contains "
                f"{len(shape_meta)} entries"
            )
        recv_tensors = self.communication.recv_bwd(
            shape_meta,
            node.peer_pp_rank,
            stream=stream,
            task=node,
        )
        inp_grads: list[Any | None] = [None] * payload_arity
        for tensor_index, tensor in zip(tensor_indices, recv_tensors):
            if (
                isinstance(tensor_index, bool)
                or not isinstance(tensor_index, int)
                or tensor_index < 0
                or tensor_index >= payload_arity
            ):
                raise ValueError(
                    f"Backward dense relay {node.uid} has invalid gradient "
                    f"tensor index={tensor_index!r} for "
                    f"payload_arity={payload_arity}"
                )
            inp_grads[tensor_index] = tensor
        return {"inp_grads": inp_grads}

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
    ) -> None:
        """Run one iteration using state prepared outside the hot loop."""
        assert prepared is not None, (
            "load_training_dag() must prepare execution before run_dag()"
        )
        dag = prepared.dag
        sorted_dag_nodes = prepared.sorted_dag_nodes
        planned_by_uid = prepared.planned_by_uid
        physical_stream_by_uid = prepared.physical_stream_by_uid

        self.params.drain_pending_frees()
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
        update_result = None

        self.buffers.init_refcounts(prepared.buffer_refcount_template)

        for node in sorted_dag_nodes:
            task_type = node.task_type
            batch = node.batches[0]
            mb_idx = batch.mb_idx
            ubid = self._node_bucket_key(node)
            node_stream = physical_stream_by_uid[node.uid]
            node_tag = self._node_tag_str(node)

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
                    dependency_reason = operation.reason_for(predecessor_uid)
                    if is_relaxable_p2p_ordering_dependency(
                        dag.nodes[predecessor_uid],
                        node,
                        dependency_reason,
                    ):
                        continue
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
                    compute_node = node.data_preds[0]
                    send_data = self.buffers.task[compute_node.uid]["send_output"]
                    self.communication.send(
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
                    compute_node = node.data_succs[0]
                    node_meta = self._node_meta(node)
                    is_backward_dense_relay = (
                        bool(node_meta.get("dense_relay"))
                        and node_meta.get("direction") == "backward"
                    )
                    if is_backward_dense_relay:
                        self.buffers.task[node.uid] = (
                            self._recv_backward_dense_relay(node, node_stream)
                        )
                    elif compute_node.task_type == TaskType.FWD:
                        recv_ubid = self._node_bucket_key(compute_node)
                        recv_tensors = self.communication.recv_fwd(
                            recv_ubid,
                            node.peer_pp_rank,
                            stream=node_stream,
                            task=node,
                        )
                        self.buffers.task[node.uid] = recv_tensors
                    else:
                        fwd_uid = compute_node.node_meta.get("fwd_uid")
                        fwd_key = (compute_node.node_meta.get("bucket_key"), fwd_uid)
                        shape_meta = self.buffers.task[("shape_ref",) + fwd_key]
                        recv_tensors = self.communication.recv_bwd(
                            shape_meta,
                            node.peer_pp_rank,
                            stream=node_stream,
                            task=node,
                        )
                        self.buffers.task[node.uid] = recv_tensors
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
                    self.buffers.task[node.uid] = fwd_buf
                    a2a_evt = torch.cuda.Event()
                    a2a_evt.record(node_stream)
                    self.events.a2a[node.uid] = a2a_evt

                case TaskType.BWD_A2A:
                    bwd_pred = next(
                        p for p in node.data_preds
                        if p.task_type in (TaskType.BWD, TaskType.BWD_I)
                    )
                    tensor_idx = self._node_meta(node)["a2a_tensor_idx"]
                    bwd_buf = dict(self.buffers.task[bwd_pred.uid])
                    self.buffers.release(bwd_pred.uid)
                    inp_grads = list(bwd_buf["inp_grads"])
                    grad_a2a_out = inp_grads[tensor_idx]
                    assert grad_a2a_out is not None, (
                        f"BWD_A2A tag={node_tag}: grad at a2a_tensor_idx={tensor_idx} is None"
                    )
                    inp_grads[tensor_idx] = self._all_to_all_ep_boundary(
                        node, grad_a2a_out, node_stream
                    )
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
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.events.recv:
                        self.events.recv.pop(recv_pred.uid)

                    a2a_pred = next(
                        (
                            p
                            for p in node.data_preds
                            if p.task_type
                            in (
                                TaskType.FWD_A2A,
                                TaskType.FWD_M_TO_N,
                                TaskType.FWD_N_TO_M,
                            )
                        ),
                        None,
                    )
                    if a2a_pred is not None and a2a_pred.uid in self.events.a2a:
                        self.events.a2a.pop(a2a_pred.uid)

                    fwd_data_pred = next(
                        (p for p in node.data_preds
                         if p.task_type in (
                             TaskType.FWD,
                             TaskType.FWD_A2A,
                             TaskType.FWD_M_TO_N,
                             TaskType.FWD_N_TO_M,
                         )), None
                    )
                    if fwd_data_pred is not None:
                        input_tensors = self.buffers.task[fwd_data_pred.uid]["detached_outs"]
                        self.buffers.release(fwd_data_pred.uid)
                    elif recv_pred is not None:
                        input_tensors = self.buffers.task[recv_pred.uid]
                        self.buffers.release(recv_pred.uid)
                    else:
                        input_tensors = inputs

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
                    recv_pred, a2a_pred = self._wait_for_backward_input_events(
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
                    elif recv_pred is not None:
                        upstream_grads = self.buffers.task[recv_pred.uid]
                        self.buffers.release(recv_pred.uid)
                        if not isinstance(upstream_grads, (list, tuple)):
                            upstream_grads = [upstream_grads]
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = list(upstream_grads)
                    else:
                        outputs_or_loss = fwd_out["out_with_grad"]
                        upstream_grads = None

                    if a2a_pred is not None:
                        a2a_buf = self.buffers.task[a2a_pred.uid]
                        pre_detach_outs = fwd_out["pre_detach_outs"]
                        detached_outs = fwd_out["detached_outs"]
                        for d, g in zip(detached_outs, a2a_buf["inp_grads"]):
                            if (
                                d is not None
                                and isinstance(d, torch.Tensor)
                                and d.requires_grad
                                and g is not None
                            ):
                                d.grad = g
                        self.buffers.release(a2a_pred.uid)
                    else:
                        bwd_pred = next(
                            (p for p in node.data_preds
                             if p.task_type in (TaskType.BWD, TaskType.BWD_I)), None
                        )
                        if bwd_pred is not None and recv_pred is None:
                            prev_buf = self.buffers.task[bwd_pred.uid]
                            pre_detach_outs = fwd_out.get("pre_detach_outs")
                            detached_outs = fwd_out.get("detached_outs")
                            for d, g in zip(detached_outs, prev_buf["inp_grads"]):
                                if (
                                    d is not None
                                    and isinstance(d, torch.Tensor)
                                    and d.requires_grad
                                    and g is not None
                                ):
                                    d.grad = g
                            self.buffers.release(bwd_pred.uid)
                        else:
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
                    recv_pred, bwd_a2a_pred = self._wait_for_backward_input_events(
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
                    elif recv_pred is not None:
                        upstream_raw = self.buffers.task[recv_pred.uid]
                        self.buffers.release(recv_pred.uid)
                        if not isinstance(upstream_raw, (list, tuple)):
                            upstream_raw = [upstream_raw]
                        stage_outputs_or_loss = fwd_out["out_with_grad"]
                        output_grads = list(upstream_raw)
                    else:
                        stage_outputs_or_loss = fwd_out["out_with_grad"]
                        output_grads = None

                    if (
                        bwd_a2a_pred is not None
                        and not self._node_meta(node).get("compute_loss", False)
                        and recv_pred is None
                    ):
                        a2a_buf = self.buffers.task[bwd_a2a_pred.uid]
                        detached_outs = fwd_out["detached_outs"]
                        output_grads_full = [
                            a2a_buf["inp_grads"][i]
                            for i, t in enumerate(detached_outs)
                            if t is not None and getattr(t, "requires_grad", False)
                        ]
                        pairs = [
                            (o, g) for o, g in zip(stage_outputs_or_loss, output_grads_full)
                            if g is not None
                        ]
                        stage_outputs_or_loss = [p[0] for p in pairs]
                        output_grads = [p[1] for p in pairs]
                        self.buffers.release(bwd_a2a_pred.uid)
                    elif (
                        not self._node_meta(node).get("compute_loss", False)
                        and recv_pred is None
                    ):
                        (
                            stage_outputs_or_loss,
                            local_output_grads,
                        ) = self._consume_local_backward_output_grads(
                            node,
                            fwd_out,
                            stage_outputs_or_loss,
                        )
                        if local_output_grads is not None:
                            output_grads = local_output_grads

                    input_values = fwd_out.get("inp_with_grad") or []
                    weights = self.stages.bucket(ubid).weights()

                    with torch.cuda.stream(node_stream):
                        dinputs, param_groups, output_backward_ctx = self.compute.bucket_backward_input(
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
                    if output_backward_ctx is not None:
                        self.buffers.task[node.uid]["output_backward_ctx"] = output_backward_ctx
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
                        output_backward_ctx = bwdi_buf.get("output_backward_ctx")
                        if output_backward_ctx is not None:
                            self.compute.backward_weight_from_outputs(
                                output_backward_ctx["stage_outputs_or_loss"],
                                output_backward_ctx["output_grads"],
                                iter(weights),
                            )
                        else:
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
                        synchronize=timing_recorder is None,
                    )

                case TaskType.ORDER_DUMMY:
                    pass

            if timing_recorder is not None:
                timing_recorder.after_node(node, node_stream)
            completion_event = torch.cuda.Event()
            completion_event.record(node_stream)
            completion_events[node.uid] = completion_event
            completion_streams[node.uid] = node_stream
            self._rf_exit(rf)
            self.runtime.nvtx_pop()

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
        synchronize: bool = True,
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

        if synchronize:
            torch.cuda.synchronize()

        if has_zero_shard_optimizers:
            return losses

        return {
            "losses": losses,
        }
