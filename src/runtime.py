from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from concurrent.futures import Future, ThreadPoolExecutor


@dataclass
class BufferStore:
    """Per-iteration task outputs and their consumer refcounts."""

    task: dict[Any, Any] = field(default_factory=dict)
    refcounts: dict[Any, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.task.clear()
        self.refcounts.clear()

    def init_refcounts(self, dag: Any) -> None:
        nodes_iter = dag.nodes.values() if isinstance(dag.nodes, dict) else dag.nodes
        self.refcounts = {
            node.uid: len(node.data_succs)
            for node in nodes_iter
            if node.data_succs
        }

    def release(self, uid: Any) -> None:
        remaining = self.refcounts.get(uid)
        if remaining is None:
            self.task.pop(uid, None)
            return

        remaining -= 1
        if remaining <= 0:
            self.refcounts.pop(uid, None)
            self.task.pop(uid, None)
        else:
            self.refcounts[uid] = remaining


@dataclass
class EventStore:
    """CUDA events produced by non-compute DAG tasks during one iteration."""

    recv: dict[Any, Any] = field(default_factory=dict)
    a2a: dict[Any, Any] = field(default_factory=dict)
    all_reduce: dict[Any, Any] = field(default_factory=dict)
    reduce_scatter: dict[Any, Any] = field(default_factory=dict)
    all_gather: dict[Any, Any] = field(default_factory=dict)
    backward: dict[Any, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.recv.clear()
        self.a2a.clear()
        self.all_reduce.clear()
        self.reduce_scatter.clear()
        self.all_gather.clear()
        self.backward.clear()


@dataclass
class BucketState:
    """Loaded runtime state for one globally unique compute bucket."""

    forward_fn: Any = None
    forward_args: list[Any] = field(default_factory=list)
    forward_input_meta: list[Any] = field(default_factory=list)
    input_idxs: list[int] = field(default_factory=list)
    param_idxs: list[int] = field(default_factory=list)
    param_names: list[str] = field(default_factory=list)
    optimizer: Any = None
    trainable_param_idxs: list[int] = field(default_factory=list)
    activation_checkpoint_subgraph_count: int = 1

    flat_params: Any = None
    flat_grads: Any = None
    shard_param: Any = None
    shard_optimizer: Any = None
    reduce_scatter_grads: Any = None
    param_shard_info: tuple[int, int, int] | None = None
    param_view_specs: list[Any] = field(default_factory=list)
    full_params_fresh: bool = False

    def weights(self) -> list[Any]:
        return [
            self.forward_args[idx]
            for idx in self.param_idxs
            if self.forward_args[idx] is not None
        ]

    def trainable_params(self) -> list[Any]:
        return [
            self.forward_args[idx]
            for idx in self.trainable_param_idxs
            if self.forward_args[idx] is not None
        ]


@dataclass
class StageStore:
    """Loaded stage and bucket state owned by one actor."""

    graph_modules: dict[Any, Any] = field(default_factory=dict)
    buckets: dict[Any, BucketState] = field(default_factory=dict)

    param_sharded_ubids: set[Any] = field(default_factory=set)
    grad_sharded_ubids: set[Any] = field(default_factory=set)
    zero_managed_ubids: set[Any] = field(default_factory=set)

    def clear_loaded_modules(self) -> None:
        self.graph_modules.clear()
        self.buckets.clear()

    def ensure_bucket(self, ubid: Any) -> BucketState:
        return self.buckets.setdefault(ubid, BucketState())

    def bucket(self, ubid: Any) -> BucketState:
        try:
            return self.buckets[ubid]
        except KeyError as exc:
            raise KeyError(f"Unknown bucket_key {ubid!r}") from exc

    def get_bucket(self, ubid: Any) -> BucketState | None:
        return self.buckets.get(ubid)


@dataclass
class RuntimeState:
    """Actor-local distributed runtime environment."""

    pp_rank: int
    dp_rank: int
    dp_degree: int
    pp_degree: int
    world_size: int
    no_nvtx: bool = False
    device: str = "cuda"
    dp_group: Any = None
    ep_group: Any = None
    pp_lo_hi: Any = None
    pp_hi_lo: Any = None
    streams: dict[str, torch.cuda.Stream] = field(default_factory=dict)
    pytorch_profiler_enabled: bool = False
    torch_profiler: Any = None

    @property
    def global_rank(self) -> int:
        return self.pp_rank + self.dp_rank * self.pp_degree

    def pipeline_peer_global_rank(self, pp_rank: int) -> int:
        return pp_rank + self.dp_rank * self.pp_degree

    def stream_id(self, node_or_stream: Any) -> str:
        if isinstance(node_or_stream, str):
            return node_or_stream
        return str(getattr(node_or_stream, "stream", "default_stream"))

    def initialize_streams_for_training_dag(self, training_dag: Any) -> None:
        stream_ids = {
            self.stream_id(n)
            for n in training_dag.nodes.values()
            if getattr(n, "stream", None) is not None
        }
        stream_ids.add("default_stream")
        self.streams = {
            stream_id: torch.cuda.Stream(device=self.device)
            for stream_id in sorted(stream_ids)
        }

        # Force cuBLAS context initialization on every logical stream used by
        # this DAG so the first backward pass does not hit lazy CUDA warnings.
        for stream in self.streams.values():
            with torch.cuda.stream(stream):
                w = torch.zeros(4, 4, device=self.device)
                torch.mm(w, w)

    def stream_for_id(self, stream_id: str) -> torch.cuda.Stream:
        assert stream_id in self.streams, (
            f"TrainingDAG referenced stream={stream_id!r}, but load_training_dag "
            f"initialized only {sorted(self.streams)}"
        )
        return self.streams[stream_id]

    def stream_for_task(self, task: Any) -> torch.cuda.Stream:
        return self.stream_for_id(self.stream_id(task))

    def default_stream(self) -> torch.cuda.Stream:
        return self.stream_for_id("default_stream")

    def nvtx_push(self, label: str) -> None:
        if not self.no_nvtx:
            torch.cuda.nvtx.range_push(label)

    def nvtx_pop(self) -> None:
        if not self.no_nvtx:
            torch.cuda.nvtx.range_pop()


@dataclass
class ParamStorage:
    """ZeRO parameter and gradient storage owned by one actor."""

    runtime: RuntimeState
    stages: StageStore
    logger: Any
    grad_buffer_dtype: torch.dtype = torch.float32
    cleanup_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1)
    )
    pending_param_frees: dict[Any, Future] = field(default_factory=dict)
    pending_grad_frees: dict[Any, Future] = field(default_factory=dict)

    def clear_param_grads(self) -> None:
        for bucket in self.stages.buckets.values():
            for idx in bucket.trainable_param_idxs:
                param = bucket.forward_args[idx]
                if param is not None:
                    param.grad = None

    def zero_grad_buffers(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            for bucket in self.stages.buckets.values():
                if bucket.flat_grads is not None:
                    bucket.flat_grads.zero_()
                if bucket.reduce_scatter_grads is not None:
                    bucket.reduce_scatter_grads.zero_()

    def wait_pending_free(
        self,
        pending: dict[Any, Future],
        ubid: Any | None,
    ) -> None:
        if ubid is None:
            return
        fut = pending.pop(ubid, None)
        if fut is not None:
            fut.result()

    def drain_pending_frees(self) -> None:
        for pending in (self.pending_param_frees, self.pending_grad_frees):
            futures = list(pending.values())
            pending.clear()
            for fut in futures:
                fut.result()

    def accumulate_zero_param_grads_to_flat(
        self,
        ubid: Any | None,
        stream: torch.cuda.Stream,
    ) -> None:
        if ubid is None or ubid not in self.stages.zero_managed_ubids or self.runtime.dp_degree <= 1:
            return
        if ubid in self.stages.grad_sharded_ubids:
            self.wait_pending_free(self.pending_grad_frees, ubid)
        bucket = self.stages.bucket(ubid)
        specs = bucket.param_view_specs
        if not specs:
            return
        with torch.cuda.stream(stream):
            flat_grads = bucket.flat_grads
            if flat_grads is None:
                shard_info = bucket.param_shard_info
                if shard_info is None:
                    return
                _shard_start, shard_size, _orig_numel = shard_info
                flat_grads = torch.zeros(
                    shard_size * self.runtime.dp_degree,
                    dtype=self.grad_buffer_dtype,
                    device=self.runtime.device,
                )
                bucket.flat_grads = flat_grads
            for param, offset, numel, _shape in specs:
                grad = param.grad
                if grad is None:
                    continue
                flat_grads[offset:offset + numel].add_(
                    grad.detach().reshape(-1).to(flat_grads.dtype)
                )
                param.grad = None

    def defer_free_full_params(self, ubid: Any | None, evt: torch.cuda.Event) -> None:
        if ubid is None or ubid not in self.stages.param_sharded_ubids:
            return
        self.wait_pending_free(self.pending_param_frees, ubid)
        self.pending_param_frees[ubid] = self.cleanup_executor.submit(
            self._wait_then_free_full_params,
            ubid,
            evt,
        )

    def defer_free_full_grads(self, ubid: Any | None, evt: torch.cuda.Event) -> None:
        if ubid is None or ubid not in self.stages.grad_sharded_ubids:
            return
        self.wait_pending_free(self.pending_grad_frees, ubid)
        self.pending_grad_frees[ubid] = self.cleanup_executor.submit(
            self._wait_then_free_full_grads,
            ubid,
            evt,
        )

    def _wait_then_free_full_params(self, ubid: Any, evt: torch.cuda.Event) -> None:
        evt.synchronize()
        self.free_full_params(ubid)

    def _wait_then_free_full_grads(self, ubid: Any, evt: torch.cuda.Event) -> None:
        evt.synchronize()
        self.free_full_grads(ubid)

    def alloc_full_params(self, ubid: Any) -> None:
        assert ubid is not None, "alloc_full_params requires a non-None ubid"
        assert ubid in self.stages.param_sharded_ubids, (
            f"alloc_full_params: ubid={ubid} is not in param_sharded_ubids="
            f"{self.stages.param_sharded_ubids}"
        )
        bucket = self.stages.bucket(ubid)
        self.wait_pending_free(self.pending_param_frees, ubid)
        assert bucket.param_shard_info is not None, (
            f"alloc_full_params: missing param_shard_info for ubid={ubid}"
        )
        full = bucket.flat_params
        assert full is not None, (
            f"alloc_full_params: missing flat_params buffer for ubid={ubid}"
        )
        specs = bucket.param_view_specs
        assert specs, f"alloc_full_params: missing param_view_specs for ubid={ubid}"
        storage = full.untyped_storage()
        required_bytes = full.numel() * full.element_size()
        storage.resize_(required_bytes)
        self.logger.debug(
            f"[alloc_full_params] rank={self.runtime.global_rank} ubid={ubid}: "
            f"numel={full.numel()} required_bytes={required_bytes} "
            f"storage_size={storage.size()} fresh={bucket.full_params_fresh}"
        )
        for param, offset, numel, shape in specs:
            param.data = full[offset:offset + numel].view(shape)
            param.requires_grad_(True)
        zero_storage = []
        for i, (param, offset, numel, shape) in enumerate(specs):
            p_storage = param.untyped_storage()
            if p_storage.size() == 0:
                name = bucket.param_names[i] if i < len(bucket.param_names) else f"param{i}"
                zero_storage.append(
                    f"{name}: offset={offset} numel={numel} "
                    f"shape={shape} stride={tuple(param.stride())}"
                )
        assert not zero_storage, (
            f"[alloc_full_params_zero_storage] rank={self.runtime.global_rank} "
            f"ubid={ubid}: " + " | ".join(zero_storage)
        )

    def free_full_params(self, ubid: Any) -> None:
        assert ubid is not None, "free_full_params requires a non-None ubid"
        assert ubid in self.stages.param_sharded_ubids, (
            f"free_full_params: ubid={ubid} is not in param_sharded_ubids="
            f"{self.stages.param_sharded_ubids}"
        )
        bucket = self.stages.bucket(ubid)
        full = bucket.flat_params
        assert full is not None, (
            f"free_full_params: missing flat_params buffer for ubid={ubid}"
        )
        storage = full.untyped_storage()
        self.logger.debug(
            f"[free_full_params] rank={self.runtime.global_rank} ubid={ubid}: "
            f"storage_size_before={storage.size()} fresh_before={bucket.full_params_fresh}"
        )
        storage.resize_(0)
        bucket.full_params_fresh = False

    def alloc_full_grads(self, ubid: Any, stream: torch.cuda.Stream) -> None:
        assert ubid is not None, "alloc_full_grads requires a non-None ubid"
        assert ubid in self.stages.grad_sharded_ubids, (
            f"alloc_full_grads: ubid={ubid} is not in grad_sharded_ubids="
            f"{self.stages.grad_sharded_ubids}"
        )
        bucket = self.stages.bucket(ubid)
        self.wait_pending_free(self.pending_grad_frees, ubid)
        specs = bucket.param_view_specs
        assert specs, f"alloc_full_grads: missing param_view_specs for ubid={ubid}"
        shard_info = bucket.param_shard_info
        assert shard_info is not None, (
            f"alloc_full_grads: missing param_shard_info for ubid={ubid}"
        )
        shard_size = shard_info[1]
        with torch.cuda.stream(stream):
            if bucket.flat_grads is None:
                bucket.flat_grads = torch.zeros(
                    shard_size * self.runtime.dp_degree,
                    dtype=self.grad_buffer_dtype,
                    device=self.runtime.device,
                )
            if bucket.reduce_scatter_grads is None:
                bucket.reduce_scatter_grads = torch.zeros(
                    shard_size,
                    dtype=self.grad_buffer_dtype,
                    device=self.runtime.device,
                )

    def free_full_grads(self, ubid: Any) -> None:
        assert ubid is not None, "free_full_grads requires a non-None ubid"
        assert ubid in self.stages.grad_sharded_ubids, (
            f"free_full_grads: ubid={ubid} is not in grad_sharded_ubids="
            f"{self.stages.grad_sharded_ubids}"
        )
        bucket = self.stages.bucket(ubid)
        specs = bucket.param_view_specs
        assert specs, f"free_full_grads: missing param_view_specs for ubid={ubid}"
        for param, *_ in specs:
            param.grad = None
        bucket.flat_grads = None

    def all_gather_full_params(self, ubid: Any, stream: torch.cuda.Stream) -> int:
        assert ubid is not None, "all_gather_full_params requires a non-None ubid"
        if not self._has_trainable_params_for_collective(ubid, "all_gather_full_params"):
            return 0
        assert ubid in self.stages.param_sharded_ubids, (
            f"all_gather_full_params: ubid={ubid} is not in param_sharded_ubids="
            f"{self.stages.param_sharded_ubids}"
        )
        bucket = self.stages.bucket(ubid)
        self.alloc_full_params(ubid)
        flat_params = bucket.flat_params
        shard_in = bucket.shard_param
        assert flat_params is not None, (
            f"all_gather_full_params: missing flat_params buffer for ubid={ubid}"
        )
        assert shard_in is not None, (
            f"all_gather_full_params: missing shard_param buffer for ubid={ubid}"
        )
        assert not bucket.full_params_fresh, (
            f"all_gather_full_params: ubid={ubid} dispatched but full params are already fresh; "
            "DAG is constructing a redundant ALL_GATHER"
        )
        with torch.cuda.stream(stream):
            self.logger.debug(
                f"[all_gather_begin] rank={self.runtime.global_rank} ubid={ubid}: "
                f"flat_numel={flat_params.numel()} flat_storage={flat_params.untyped_storage().size()} "
                f"shard_numel={shard_in.numel()} shard_storage={shard_in.untyped_storage().size()}"
            )
            dist.all_gather_into_tensor(flat_params, shard_in, group=self.runtime.dp_group)
            bucket.full_params_fresh = True
            self.logger.debug(
                f"[all_gather_end] rank={self.runtime.global_rank} ubid={ubid}: "
                f"flat_storage={flat_params.untyped_storage().size()}"
            )
            return flat_params.numel() * flat_params.element_size()

    def has_zero_shard_optimizers(self) -> bool:
        return any(bucket.shard_optimizer is not None for bucket in self.stages.buckets.values())

    def step_zero_shard_optimizers(
        self,
        stream: torch.cuda.Stream,
        reduce_scatter_events: dict[Any, torch.cuda.Event],
    ) -> None:
        for evt in reduce_scatter_events.values():
            stream.wait_event(evt)

        for bucket in self.stages.buckets.values():
            shard_optim = bucket.shard_optimizer
            if shard_optim is None:
                continue
            shard_param = bucket.shard_param
            rs_grad = bucket.reduce_scatter_grads
            if rs_grad is None:
                continue
            with torch.cuda.stream(stream):
                shard_param.grad = rs_grad.to(shard_param.dtype)
                shard_optim.step()
            shard_param.grad = None

        for ubid in self.stages.param_sharded_ubids:
            bucket = self.stages.bucket(ubid)
            for param, *_ in bucket.param_view_specs:
                param.grad = None
            full = bucket.flat_params
            if full is not None:
                storage = full.untyped_storage()
                if storage.size() != 0:
                    storage.resize_(0)
            bucket.full_params_fresh = False

    def _has_trainable_params_for_collective(self, ubid: Any, op_name: str) -> bool:
        bucket = self.stages.get_bucket(ubid)
        if bucket is not None and bucket.trainable_param_idxs:
            return True
        self.logger.warning(
            "%s: skipping collective for ubid=%s because it has no trainable param indices",
            op_name,
            ubid,
        )
        return False
