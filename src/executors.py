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
from torch.autograd.graph import GradientEdge, Node, get_gradient_edge
from torch.nn import Parameter

try:
    from torchtitan.models.moe.utils import _permute, _unpermute
except ImportError:  # Newer TorchTitan source layout.
    from torchtitan.models.common.moe.utils import _permute, _unpermute

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
    ep_rank: int
    ep_degree: int
    input_splits: list[int] | None = None
    output_splits: list[int] | None = None
    input_shape: torch.Size | None = None
    permuted_indices: torch.Tensor | None = None


@dataclass
class CommunicationExecutor:
    """Actor-local communication operations used by the DAG dispatcher."""

    runtime: RuntimeState
    stages: StageStore
    logger: Any

    def send(self, send_data: Any, peer_pp_rank: int, stream: torch.cuda.Stream) -> None:
        global_dst_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        with torch.cuda.stream(stream):
            tensors = send_data if isinstance(send_data, (list, tuple)) else [send_data]
            use_lo_hi = global_dst_rank > self.runtime.global_rank
            pp_group = self.runtime.pp_lo_hi if use_lo_hi else self.runtime.pp_hi_lo
            for tensor in tensors:
                dist.send(tensor, dst=global_dst_rank, group=pp_group)

    def recv_fwd(self, recv_ubid: Any, peer_pp_rank: int, stream: torch.cuda.Stream) -> list:
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
            use_hi_lo = global_src_rank > self.runtime.global_rank
            pp_group = self.runtime.pp_hi_lo if use_hi_lo else self.runtime.pp_lo_hi
            for tensor in buf:
                dist.recv(tensor, src=global_src_rank, group=pp_group)
        return buf

    def recv_bwd(self, shape_meta: list, peer_pp_rank: int, stream: torch.cuda.Stream) -> list:
        global_src_rank = self.runtime.pipeline_peer_global_rank(peer_pp_rank)

        buf = [
            torch.empty(shape, dtype=dtype, device=self.runtime.device)
            for shape, dtype in shape_meta
        ]
        with torch.cuda.stream(stream):
            use_hi_lo = global_src_rank > self.runtime.global_rank
            pp_group = self.runtime.pp_hi_lo if use_hi_lo else self.runtime.pp_lo_hi
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
        with torch.cuda.stream(stream):
            for grad in grad_tensors:
                dist.all_reduce(grad, group=self.runtime.dp_group)
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
        with torch.cuda.stream(stream):
            total_bytes = flat_grads.numel() * flat_grads.element_size()
            tmp = torch.empty_like(rs_out)
            dist.reduce_scatter_tensor(tmp, flat_grads, group=self.runtime.dp_group)
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
            if tensor.requires_grad:
                tensor = tensor.detach().requires_grad_(True)
            fwd_args[i] = tensor

        fwd_inputs = [fwd_args[i] for i in input_idxs]
        inp_with_grad = [t for t in fwd_inputs if t is not None and t.requires_grad]

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
            (p for p in node.data_preds if p.task_type == TaskType.BWD_A2A), None
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
                ep_degree = len(participants)
                try:
                    ep_rank = participants.index(self.runtime.global_rank)
                except ValueError as error:
                    raise ValueError(
                        f"Actor rank {self.runtime.global_rank} is not in A2A "
                        f"participants={participants}"
                    ) from error
                state = _ExpertA2AState(
                    ep_rank=ep_rank,
                    ep_degree=ep_degree,
                )
                self.expert_a2a_states[expert_fwd_uid] = state

            task_type = getattr(node, "task_type", None)
            if task_type == TaskType.FWD_A2A:
                if direction == "incoming":
                    if routing_counts is None:
                        raise ValueError(
                            f"Forward dispatch node {node.uid} is missing "
                            "num_tokens_per_expert"
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
        if (
            state.input_splits is None
            or state.output_splits is None
            or state.input_shape is None
            or state.permuted_indices is None
        ):
            raise RuntimeError(
                "A2A combine/gradient dispatch ran before forward dispatch"
            )

    def _dispatch_expert_a2a(
        self,
        state: _ExpertA2AState,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if num_tokens_per_expert.ndim != 1:
            raise ValueError(
                "num_tokens_per_expert must be one-dimensional, got "
                f"shape={tuple(num_tokens_per_expert.shape)}"
            )
        if num_tokens_per_expert.shape[0] % state.ep_degree:
            raise ValueError(
                f"num_experts={num_tokens_per_expert.shape[0]} must be divisible "
                f"by ep_degree={state.ep_degree}"
            )

        num_local_experts = (
            num_tokens_per_expert.shape[0] // state.ep_degree
        )
        if state.ep_degree > 1:
            with torch.no_grad():
                # Fixed-size metadata exchange. The result is source-major:
                # [source rank][local expert].
                grouped_counts = all_to_all_single(
                    num_tokens_per_expert,
                    None,
                    None,
                    group=self.runtime.ep_group,
                )
                grouped_counts = torch.ops._c10d_functional.wait_tensor(
                    grouped_counts
                )
                input_splits = (
                    num_tokens_per_expert.view(state.ep_degree, -1)
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

            routed_input = all_to_all_single_autograd(
                routed_input,
                state.output_splits,
                state.input_splits,
                self.runtime.ep_group,
            )
        else:
            grouped_counts = num_tokens_per_expert
            state.input_splits = []
            state.output_splits = []

        (
            state.input_shape,
            routed_input,
            state.permuted_indices,
            local_counts,
        ) = _permute(
            routed_input,
            grouped_counts,
            state.ep_degree,
            num_local_experts,
        )

        # Piper's expert weights retain their global first dimension. Zero
        # counts select only the local rank's contiguous expert range.
        global_counts = torch.zeros_like(num_tokens_per_expert)
        local_start = state.ep_rank * num_local_experts
        global_counts[local_start : local_start + num_local_experts] = (
            local_counts
        )
        return routed_input, global_counts

    def _dispatch_expert_a2a_gradient(
        self,
        state: _ExpertA2AState,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._require_expert_a2a_state(state)
        if state.ep_degree > 1:
            tensor = all_to_all_single_autograd(
                tensor,
                state.output_splits,
                state.input_splits,
                self.runtime.ep_group,
            )

        # Reapply the forward source-major -> expert-major permutation. The
        # sentinel row supplies every padding index generated by TorchTitan.
        tensor = torch.vstack((tensor, tensor.new_zeros((tensor.shape[-1],))))
        return tensor[state.permuted_indices, :]

    def _combine_expert_a2a(
        self,
        state: _ExpertA2AState,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        self._require_expert_a2a_state(state)
        tensor = _unpermute(
            tensor,
            state.input_shape,
            state.permuted_indices,
        )
        if state.ep_degree == 1:
            return tensor
        return all_to_all_single_autograd(
            tensor,
            state.input_splits,
            state.output_splits,
            self.runtime.ep_group,
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
                    self.communication.send(send_data, node.peer_pp_rank, stream=node_stream)
                    send_buf = self.buffers.task.get(compute_node.uid)
                    if isinstance(send_buf, dict):
                        send_buf["send_output"] = None
                    self.buffers.release(compute_node.uid)

                case TaskType.RECV:
                    compute_node = node.data_succs[0]
                    if compute_node.task_type == TaskType.FWD:
                        recv_ubid = self._node_bucket_key(compute_node)
                        recv_tensors = self.communication.recv_fwd(
                            recv_ubid, node.peer_pp_rank, stream=node_stream
                        )
                    else:
                        fwd_uid = compute_node.node_meta.get("fwd_uid")
                        fwd_key = (compute_node.node_meta.get("bucket_key"), fwd_uid)
                        shape_meta = self.buffers.task[("shape_ref",) + fwd_key]
                        recv_tensors = self.communication.recv_bwd(
                            shape_meta, node.peer_pp_rank, stream=node_stream
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
                        (p for p in node.data_preds if p.task_type == TaskType.FWD_A2A), None
                    )
                    if a2a_pred is not None and a2a_pred.uid in self.events.a2a:
                        self.events.a2a.pop(a2a_pred.uid)

                    fwd_data_pred = next(
                        (p for p in node.data_preds
                         if p.task_type in (TaskType.FWD, TaskType.FWD_A2A)), None
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
