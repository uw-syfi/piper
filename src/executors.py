from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter

from .backward import construct_reverse_graph, get_param_groups, _get_grad_fn_or_grad_acc
from .runtime import BufferStore, EventStore, ParamStorage, RuntimeState, StageStore
from .tasks import TaskType


def _maybe_inject_fault(pass_name: str, iter_count: int, dp_rank: int) -> None:
    """Inject a debug fault when PIPER_FAULT matches the given execution point.

    pass_name: name of the executing pass (e.g. "bwd").
    iter_count: 0-based run_dag iteration counter (warmup iterations count).
    dp_rank: this actor's DP rank.

    PIPER_FAULT formats (never set in production runs):
      "bwd:<iter>:<dp_rank>"            -> raise RuntimeError (loud mode)
      "bwd:<iter>:<dp_rank>:sleep:<s>"  -> sleep s seconds    (stuck mode)
    """
    spec = os.environ.get("PIPER_FAULT")
    if not spec:
        return
    parts = spec.split(":")
    if parts[:3] != [pass_name, str(iter_count), str(dp_rank)]:
        return
    # One BWD node per annotated segment — latch to fire once per iteration.
    key = (spec, iter_count)
    if key in _FIRED_FAULTS:
        return
    _FIRED_FAULTS.add(key)
    print(f"PIPER_FAULT firing: {spec}", flush=True)
    if len(parts) >= 5 and parts[3] == "sleep":
        time.sleep(float(parts[4]))
        return
    raise RuntimeError(f"injected fault ({spec})")


_FIRED_FAULTS: set = set()


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

    def all_to_all(self, input_tensor: torch.Tensor, stream: torch.cuda.Stream) -> torch.Tensor:
        output_buf = torch.empty_like(input_tensor, device=self.runtime.device)
        with torch.cuda.stream(stream):
            dist.all_to_all_single(output_buf, input_tensor, group=self.runtime.ep_group)
        return output_buf

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
        if (
            grad.dtype == weight.dtype
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
            dtype=weight.dtype,
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
                if (
                    w.grad.dtype != w.dtype
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
    # 0-based iteration counter, set by PiperActor.run_dag each call.
    # Debug-only: consumed by _maybe_inject_fault for E2E fault injection.
    _iter_count: int = 0
    # Commit marker: optimizer update fully applied (recovery redoes +1).
    _last_committed: int = -1
    # Set (before the comm abort) by PiperActor.abort_comms: an aborted
    # collective releases kernels with garbage, so a fenced step must not commit.
    _fenced: bool = False
    # Standby mode only; off by default to keep the baseline hot path unchanged.
    _cpu_sync_allreduce: bool = False

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

    def _wait_for_all_gather(self, compute_node: Any) -> None:
        compute_stream = self.runtime.stream_for_task(compute_node)
        for pred in compute_node.data_preds:
            if pred.task_type == TaskType.ALL_GATHER:
                ag_evt = self.events.all_gather.get(pred.uid)
                if ag_evt is not None:
                    compute_stream.wait_event(ag_evt)

    def _all_to_all_ep_boundary(
        self,
        node: Any,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        direction = self._node_meta(node).get("direction")
        if direction == "outgoing":
            tensor = tensor.contiguous()
        elif direction != "incoming":
            raise ValueError(
                f"A2A node uid={getattr(node, 'uid', '<unknown>')} has invalid "
                f"direction={direction!r}; expected 'incoming' or 'outgoing'"
            )

        tensor = self.communication.all_to_all(tensor, stream=stream)
        if direction == "incoming":
            tensor = tensor.contiguous()
        return tensor

    def run(
        self,
        dag: Any,
        sorted_dag_nodes: list[Any],
        inputs: Any,
        labels: Any,
        loss_buffer: list,
        loss_fn=None,
    ) -> None:
        """Run one iteration of the loaded TrainingDAG."""
        assert dag is not None, "load_training_dag() must be called before run_dag()"
        assert sorted_dag_nodes is not None, "load_training_dag() must initialize sorted node order"
        if self._fenced:
            raise RuntimeError(
                "fenced by coordinator: communicators aborted, iteration refused"
            )

        self.params.drain_pending_frees()
        debug_enabled = self.logger.isEnabledFor(logging.DEBUG)

        self.buffers.reset()
        self.events.reset()
        self.params.clear_param_grads()
        default_stream = self.runtime.default_stream()
        self.params.zero_grad_buffers(default_stream)
        zero_evt = torch.cuda.Event()
        zero_evt.record(default_stream)
        for stream in self.runtime.streams.values():
            if stream is not default_stream:
                stream.wait_event(zero_evt)
        comp_events: dict[Any, torch.cuda.Event] = {}

        self.buffers.init_refcounts(dag)
        last_comp_event_by_stream: dict[str, torch.cuda.Event] = {}

        for node in sorted_dag_nodes:
            task_type = node.task_type
            batch = node.batches[0]
            mb_idx = batch.mb_idx
            ubid = self._node_bucket_key(node)
            node_stream = self.runtime.stream_for_task(node)
            node_stream_id = self.runtime.stream_id(node)
            node_tag = self._node_tag_str(node)

            if debug_enabled:
                self.logger.debug(
                    f"run_dag dispatch: {task_type.value} "
                    f"tag={node_tag}"
                )

            task_label = f"{task_type.value}:{node_tag}:uid{node.uid}"
            self.runtime.nvtx_push(task_label)
            rf = self._rf_enter(task_label)

            match task_type:
                case TaskType.SEND:
                    compute_node = node.data_preds[0]
                    node_stream.wait_event(comp_events[compute_node.uid])
                    send_data = self.buffers.task[compute_node.uid]["send_output"]
                    self.communication.send(send_data, node.peer_pp_rank, stream=node_stream)
                    send_buf = self.buffers.task.get(compute_node.uid)
                    if isinstance(send_buf, dict):
                        send_buf["send_output"] = None
                    self.buffers.release(compute_node.uid)

                case TaskType.RECV:
                    compute_node = node.data_succs[0]
                    comp_evt = last_comp_event_by_stream.get(self.runtime.stream_id(compute_node))
                    if comp_evt is not None:
                        node_stream.wait_event(comp_evt)
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
                    node_stream.wait_event(comp_events[fwd_pred.uid])
                    tensor_idx = self._node_meta(node)["a2a_tensor_idx"]
                    fwd_buf = dict(self.buffers.task[fwd_pred.uid])
                    self.buffers.release(fwd_pred.uid)
                    detached_outs = list(fwd_buf["detached_outs"])
                    detached_outs[tensor_idx] = self._all_to_all_ep_boundary(
                        node, detached_outs[tensor_idx], node_stream
                    ).requires_grad_(True)
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
                    node_stream.wait_event(comp_events[bwd_pred.uid])
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
                    node_stream.wait_event(comp_events[bwd_node.uid])
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
                    node_stream.wait_event(comp_events[bwd_node.uid])
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
                        node_stream.wait_event(self.events.recv.pop(recv_pred.uid))

                    a2a_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.FWD_A2A), None
                    )
                    if a2a_pred is not None and a2a_pred.uid in self.events.a2a:
                        node_stream.wait_event(self.events.a2a.pop(a2a_pred.uid))

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

                    self._wait_for_all_gather(node)

                    fwd_out = self.compute.forward(ubid, input_tensors, node_stream)
                    self.buffers.task[node.uid] = fwd_out
                    fwd_key = (node.node_meta.get("bucket_key"), node.uid)
                    self.buffers.task[("shape_ref",) + fwd_key] = [
                        (t.shape, t.dtype) for t in fwd_out["out_with_grad"]
                    ]
                    self.buffers.task[fwd_key] = fwd_out
                    evt = torch.cuda.Event()
                    evt.record(node_stream)
                    comp_events[node.uid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD:
                    _maybe_inject_fault("bwd", self._iter_count, self.runtime.dp_rank)
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.events.recv:
                        node_stream.wait_event(self.events.recv.pop(recv_pred.uid))
                    self._wait_for_all_gather(node)

                    a2a_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.BWD_A2A), None
                    )
                    if a2a_pred is not None and a2a_pred.uid in self.events.a2a:
                        node_stream.wait_event(self.events.a2a.pop(a2a_pred.uid))

                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.buffers.task[fwd_key]

                    if self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.compute.log_compute_loss_inputs(labels, node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
                            outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], labels)]
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
                    comp_events[node.uid] = evt
                    self.events.backward[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD_I:
                    recv_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.RECV), None
                    )
                    if recv_pred is not None and recv_pred.uid in self.events.recv:
                        node_stream.wait_event(self.events.recv.pop(recv_pred.uid))
                    self._wait_for_all_gather(node)

                    fwd_uid = node.node_meta.get("fwd_uid")
                    fwd_key = (node.node_meta.get("bucket_key"), fwd_uid)
                    fwd_out = self.buffers.task[fwd_key]

                    if self._node_meta(node).get("compute_loss", False):
                        assert loss_fn is not None
                        if self.logger.isEnabledFor(logging.DEBUG):
                            self.compute.log_compute_loss_inputs(labels, node, fwd_key, fwd_out)
                        with torch.cuda.stream(node_stream):
                            stage_outputs_or_loss = [loss_fn(fwd_out["out_with_grad"][0], labels)]
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

                    bwd_a2a_pred = next(
                        (p for p in node.data_preds if p.task_type == TaskType.BWD_A2A), None
                    )
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
                    comp_events[node.uid] = evt
                    self.events.backward[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.BWD_W:
                    self._wait_for_all_gather(node)
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
                    comp_events[node.uid] = evt
                    self.events.backward[ubid] = evt
                    last_comp_event_by_stream[node_stream_id] = evt
                    if self._node_meta(node).get("zero_free_full_params_after"):
                        self.params.defer_free_full_params(ubid, evt)

                case TaskType.UPD:
                    self._update(node_stream, loss_buffer)

                case TaskType.ORDER_DUMMY:
                    pass

            self._rf_exit(rf)
            self.runtime.nvtx_pop()

    def _update(self, stream: torch.cuda.Stream, loss_buffer: list):
        self.params.drain_pending_frees()
        if self.params.has_zero_shard_optimizers():
            if self._fenced:
                raise RuntimeError(
                    "fenced during collective; ZeRO optimizer step refused"
                )
            self.params.step_zero_shard_optimizers(stream, self.events.reduce_scatter)
            losses = loss_buffer
            loss_buffer.clear()
            torch.cuda.synchronize()
            self._last_committed = self._iter_count
            return losses

        if self._cpu_sync_allreduce:
            # An aborted collective fires its events with garbage gradients;
            # the outcome must be known (and unfenced) before stepping.
            for ar_evt in self.events.all_reduce.values():
                ar_evt.synchronize()
            if self._fenced:
                raise RuntimeError(
                    "fenced during gradient all-reduce; optimizer step refused "
                    f"(last_committed={self._last_committed})"
                )
        else:
            for ar_evt in self.events.all_reduce.values():
                stream.wait_event(ar_evt)

        for ubid, bucket in self.stages.buckets.items():
            if bucket.optimizer is None:
                continue
            bwd_evt = self.events.backward.get(ubid)
            if bwd_evt is not None:
                stream.wait_event(bwd_evt)

            with torch.cuda.stream(stream):
                bucket.optimizer.step()

        losses = loss_buffer
        loss_buffer.clear()

        torch.cuda.synchronize()
        # Only after synchronize returns are the weight updates provably applied.
        self._last_committed = self._iter_count

        return {
            "losses": losses,
        }
