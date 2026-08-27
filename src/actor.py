import ray
import threading
import torch
import os
import re
from typing import Any
import gc
from torch.autograd.graph import set_warn_on_accumulate_grad_stream_mismatch
import torch.distributed as dist
from collections import defaultdict

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .state import (
    create_logger,
    LOG_LEVEL,
)
from .fx import _deserialize_graphmodule, _serialize_graphmodule
from .executors import CommunicationExecutor, ComputeExecutor, DagExecutor
from .ordering import _serial_topological_order
from .runtime import BufferStore, EventStore, ParamStorage, RuntimeState, StageStore
from .tasks import training_dag_task_type as _training_dag_task_type

CLEANUP_MEMORY = False

logger = create_logger("actor", LOG_LEVEL)

def _disable_functorch_donated_buffers() -> None:
    import importlib

    config = importlib.import_module("torch._functorch.config")
    config.donated_buffer = False


def _get_rank(pp_rank, dp_rank, pp_degree):
    return pp_rank + dp_rank * pp_degree


def _create_actors(
    num_actors,
    optim_class,
    profile=False,
    no_nvtx: bool = False,
    pg=None,
    temp_dir: str = None,
    use_inductor: bool = False,
    pp_outer: bool = False,
):
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    world_size = int(os.environ["PIPER_WORLD_SIZE"])
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])
    pp_degree = int(os.environ["PIPER_PP_DEGREE"])

    from .state import piper_metadata

    for pp_rank in range(num_actors):
        global_rank = _get_rank(pp_rank, dp_rank, pp_degree)
        nsight_env = {"nsight": {
            "t": "cuda,cudnn,cublas,nvtx",
            "sample": "process-tree",
            "backtrace": "dwarf",
            "cudabacktrace": "sync:0,memory:0",
            "python-backtrace": "cuda",
            "stop-on-exit": "true",
        }} if profile else {}
        nccl_env = {
            "env_vars": {
                # "NCCL_SOCKET_IFNAME": "ens32",
                # "GLOO_SOCKET_IFNAME": "ens32",
                # "NCCL_P2P_DISABLE": "1",
                # "NCCL_DEBUG": "INFO",
                **({"TMPDIR": temp_dir} if (profile and temp_dir) else {}),
                # Actors don't inherit the driver's os.environ.
                "PIPER_NUM_STANDBY": os.environ.get("PIPER_NUM_STANDBY", "0"),
                # Watchdog auto-handling reacts to our intentional abort by
                # aborting every comm (see _abort_process_group docs).
                **({"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"}
                   if int(os.environ.get("PIPER_NUM_STANDBY", "0")) > 0 else {}),
            }
        }
        # When pp_outer=True, one bundle corresponds to one pipeline stage and
        # holds all DP replicas for that stage (placement group shape is
        # [{"GPU": dp}] * pp). Otherwise one bundle is one DP replica holding
        # all PP ranks (shape is [{"GPU": pp}] * dp).
        bundle_index = pp_rank if pp_outer else dp_rank
        actor = PiperActor.options(
            num_gpus=0.6,
            # Threaded actor:
            # - Survivor: abort_comms must run while run_dag is blocked.
            # - Standby: load_state must run while wait_state_loaded is blocked.
            max_concurrency=2,
            enable_tensor_transport=True,
            runtime_env={**nsight_env, **nccl_env},
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
            ),
        ).remote(
            pp_rank,
            optim_class,
            world_size,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
            no_nvtx=no_nvtx,
            use_inductor=use_inductor,
        )
        piper_metadata.actors[pp_rank] = actor


@ray.remote
class PiperActor:
    def __init__(
        self,
        pp_rank,
        optim_class,
        world_size,
        dp_rank=0,
        dp_degree=1,
        pp_degree=1,
        no_nvtx: bool = False,
        use_inductor: bool = False,
    ):
        self.logger = create_logger("actor", LOG_LEVEL)

        # BWD_I uses torch.autograd.grad (not .backward), so AccumulateGrad nodes are
        # traversed for stream-sync bookkeeping but never accumulate to p.grad.
        # Suppress the spurious stream-mismatch warning.
        set_warn_on_accumulate_grad_stream_mismatch(False)

        self.optim_class = optim_class
        self.use_inductor = bool(use_inductor)
        if self.use_inductor:
            _disable_functorch_donated_buffers()
        self.runtime = RuntimeState(
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            pp_degree=pp_degree,
            world_size=world_size,
            no_nvtx=no_nvtx,
        )

        self.logger.debug(
            f"Initializing Ray actor {self.runtime.global_rank} GPU {os.environ['CUDA_VISIBLE_DEVICES']}"
        )

        self.inputs = None
        self.labels = None

        self.stages = StageStore()
        # accumuate loss for each microbatch
        self.loss = []

        # DAG execution state
        self.dag = None
        self.sorted_dag_nodes = None

        self.buffers = BufferStore()
        self.events = EventStore()
        self.communication = CommunicationExecutor(self.runtime, self.stages, self.logger)
        self.compute = ComputeExecutor(self.runtime, self.stages, self.logger)
        self.params = ParamStorage(self.runtime, self.stages, self.logger)
        self.dag_executor = DagExecutor(
            self.runtime,
            self.stages,
            self.buffers,
            self.events,
            self.params,
            self.communication,
            self.compute,
            self.logger,
        )
        # Standby mode: UPD must CPU-sync all-reduces and honor the fence flag
        # (an aborted collective otherwise commits garbage gradients).
        self.dag_executor._cpu_sync_allreduce = (
            int(os.environ.get("PIPER_NUM_STANDBY", "0")) > 0
        )
        # Set when abort_comms has fully finished; join_standby_group must not
        # build a new comm while an abort is still in flight in this process.
        self._abort_done = threading.Event()
        self._state_loaded = threading.Event()
        self._resume_iter = None
        self._promotion_tensors = None

        # Non-trainable constant tensor attributes (e.g. freqs_cis, mask) pushed
        # from the coordinator before compilation so _load_stage can fill them in
        # instead of zero-initializing.  Keyed by bare attribute name (e.g. "freqs_cis").
        self.model_const_attrs: dict = {}

    def get_and_reset_peak_memory_stats(self) -> tuple:
        """Return (global_rank, max_memory_allocated_bytes) and reset peak stats."""
        max_alloc = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        return self.runtime.global_rank, max_alloc

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()

    def _nvtx_push(self, label: str) -> None:
        self.runtime.nvtx_push(label)

    def _nvtx_pop(self) -> None:
        self.runtime.nvtx_pop()

    def start_pytorch_profiler(self) -> None:
        """Begin a torch.profiler session spanning the next run_dag iterations.

        Each node's execution in DagExecutor is wrapped in a record_function
        labelled the same as its NVTX range, so the resulting trace identifies
        per-node work.
        """
        self.runtime.torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        )
        self.runtime.torch_profiler.__enter__()
        self.runtime.pytorch_profiler_enabled = True
        self.logger.debug(f"Actor {self.runtime.global_rank}: PyTorch profiler started")

    def stop_pytorch_profiler(self, profile_dir: str) -> str:
        """End the profiler session and export a chrome trace.

        The file is named ``dp{dp_rank}_pp{pp_rank}.json`` inside *profile_dir*
        so the harness can group same-dp-rank actors. Returns the written path.
        """
        self.runtime.pytorch_profiler_enabled = False
        self.runtime.torch_profiler.__exit__(None, None, None)
        os.makedirs(profile_dir, exist_ok=True)
        filepath = os.path.join(
            profile_dir, f"dp{self.runtime.dp_rank}_pp{self.runtime.pp_rank}.json"
        )
        self.runtime.torch_profiler.export_chrome_trace(filepath)
        self.runtime.torch_profiler = None
        self.logger.info(
            f"Actor {self.runtime.global_rank}: PyTorch profiler trace written to {filepath}"
        )
        return filepath

    def load_input(self, inputs):
        self.inputs = [inp.to(self.runtime.device) for inp in inputs]
        self.logger.debug(f"Actor {self.runtime.global_rank} loaded inputs {len(self.inputs)}")

    def load_labels(self, labels):
        self.labels = labels.to(self.runtime.device)
        self.logger.debug(f"Actor {self.runtime.global_rank} loaded labels {self.labels.shape}")

    def load_const_attrs(self, const_attrs: dict) -> None:
        """Store non-trainable constant tensor attributes (e.g. freqs_cis, mask).

        *const_attrs* maps bare attribute name → CPU tensor.  Values are moved to
        the actor's device so ``_load_stage`` can copy them directly.
        """
        self.model_const_attrs = {k: v.to(self.runtime.device) for k, v in const_attrs.items()}

    def get_node_ip_and_free_port(self):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", 0))
            port = s.getsockname()[1]
        return ray.util.get_node_ip_address(), port

    def _join_process_groups(self, master_addr, master_port):

        self.logger.debug(f"Actor {self.runtime.global_rank} using GPU {os.environ['CUDA_VISIBLE_DEVICES']}, master addr {master_addr}:{master_port}")

        init_method = f"tcp://{master_addr}:{master_port}"

        self.runtime.device = f"cuda:{self.runtime.global_rank % torch.cuda.device_count()}"
        torch.cuda.set_device(self.runtime.device)

        if self.runtime.pp_degree > 1 or self.runtime.dp_degree > 1:
            dist.init_process_group(
                "nccl",
                init_method=init_method,
                rank=self.runtime.global_rank,
                world_size=self.runtime.world_size,
            )

            if self.runtime.dp_degree > 1:
                self._join_dp_process_group()
            if self.runtime.pp_degree > 1:
                self._join_pp_process_group()

            self.logger.debug(f"Actor {self.runtime.global_rank} joined process groups")

    def _join_dp_process_group(self):
        # Standby ranks execute every collective new_group call (required by
        # default-mode new_group) but are members of no dp/ep group.
        n_trainers = self.runtime.dp_degree * self.runtime.pp_degree
        num_dp_groups = n_trainers // self.runtime.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [
                (dp_group_id + num_dp_groups * i) for i in range(self.runtime.dp_degree)
            ]
            # Two separate NCCL communicators over the same ranks: one for allreduce,
            # one for all2all.  Sharing a communicator causes both op types to run on
            # the same internal NCCL proxy stream, which prevents true overlap.
            process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            ep_process_group = dist.new_group(ranks=group_ranks, backend="nccl")
            if self.runtime.global_rank in group_ranks:
                self.runtime.dp_group = process_group
                self.runtime.ep_group = ep_process_group

    def _join_pp_process_group(self):
        n_trainers = self.runtime.dp_degree * self.runtime.pp_degree
        num_pp_groups = n_trainers // self.runtime.pp_degree

        for pp_group_id in range(num_pp_groups):
            group_ranks = [
                (pp_group_id * self.runtime.pp_degree + i) for i in range(self.runtime.pp_degree)
            ]
            lo_hi_group = dist.new_group(ranks=group_ranks, backend="nccl")
            hi_lo_group = dist.new_group(ranks=group_ranks, backend="nccl")

            if self.runtime.global_rank in group_ranks:
                self.runtime.pp_lo_hi = lo_hi_group
                self.runtime.pp_hi_lo = hi_lo_group

    def abort_comms(self):
        """Abort this actor's dp/ep NCCL communicators (fencing during promotion)."""
        from torch.distributed.distributed_c10d import _abort_process_group

        # The fence flag must be visible BEFORE the abort releases any kernel,
        # so the executor refuses the poisoned iteration's optimizer step.
        self.dag_executor._fenced = True
        aborted = []
        if self.runtime.dp_group is not None:
            _abort_process_group(self.runtime.dp_group)
            aborted.append("dp_group")
        if self.runtime.ep_group is not None:
            _abort_process_group(self.runtime.ep_group)
            aborted.append("ep_group")
        self.logger.info(f"abort_comms: aborted {aborted}")
        self._abort_done.set()
        return aborted

    def join_standby_group(self, new_ranks):
        """Create the survivor+standby dp/ep groups, make them the active
        training groups, and lift the fence.

        new_ranks: sorted global ranks of the members; only members call this.
        """
        new_ranks = list(new_ranks)
        if self.dag_executor._fenced:
            # Building a fresh NCCL comm while an abort is in flight in this
            # process deadlocks; wait for abort_comms to finish.
            if not self._abort_done.wait(timeout=120):
                raise RuntimeError("abort_comms did not finish within 120s")
            torch.cuda.synchronize()
        self.logger.info(f"join_standby_group: creating groups ranks={new_ranks}")
        new_dp = dist.new_group(
            ranks=new_ranks, backend="nccl", use_local_synchronization=True
        )
        new_ep = dist.new_group(
            ranks=new_ranks, backend="nccl", use_local_synchronization=True
        )
        # Communicator construction is lazy: the first collective of the
        # resumed training initializes it.
        self.runtime.dp_group = new_dp
        self.runtime.ep_group = new_ep
        self.dag_executor._fenced = False
        self.logger.info(f"join_standby_group: joined ranks={new_ranks}")
        return new_ranks

    def get_last_committed(self):
        """Last iteration whose optimizer update fully committed (-1 if none)."""
        return getattr(self.dag_executor, "_last_committed", -1)

    def _promotion_state_tensors(self):
        """Flat list of param + Adam state tensors transferred at promotion.

        Survivor: the tensors to send. Standby: the buffers to receive
        into; its missing Adam state is created as zeros.
        """
        if self._promotion_tensors is not None:
            return self._promotion_tensors
        if self.stages.zero_managed_ubids:
            raise NotImplementedError(
                "standby weight transfer supports pure DP only"
            )
        tensors = []
        for _, bucket in self.stages.buckets.items():
            opt = bucket.optimizer
            if opt is None:
                continue
            for p in bucket.trainable_params():
                st = opt.state.get(p)
                if st is None:
                    # Fused Adam creates state lazily on first step; the
                    # standby never steps, so materialize matching zeros.
                    st = {
                        "step": torch.zeros(
                            (), dtype=torch.float32, device=p.device
                        ),
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    }
                    opt.state[p] = st
                tensors.extend(
                    [p.detach(), st["exp_avg"], st["exp_avg_sq"], st["step"]]
                ) # Flat because set_target_for_ref matches by position.
        self._promotion_tensors = tensors
        return tensors

    def prepare_standby_state(self):
        """Materialize optimizer state and pre-register NIXL buffers while parked."""
        tensors = self._promotion_state_tensors()
        try:
            for t in tensors:
                ray.experimental.register_nixl_memory(t)
        except Exception:
            self.logger.exception(
                "register_nixl_memory failed; registration will happen at transfer time"
            )
        return len(tensors)

    def make_state_ref(self):
        """Publish params + optimizer state + resume iteration via NIXL RDT.

        Returns a ray.put-created ObjectRef (borrowable, see ray#59644).
        """
        state = {
            "next_iter": self.dag_executor._last_committed + 1,
            "tensors": self._promotion_state_tensors(),
        }
        return ray.put(state, _tensor_transport="nixl")

    def load_state(self, refs):
        """Receive the survivor's state directly into this actor's buffers.

        refs: single-element list holding the state ObjectRef.
        """
        ref, = refs
        targets = self._promotion_state_tensors()
        ray.experimental.set_target_for_ref(ref, targets)
        state = ray.get(ref)
        next_iter = int(state["next_iter"])
        self._iter_counter = next_iter
        self.dag_executor._last_committed = next_iter - 1
        self._resume_iter = next_iter
        self._state_loaded.set()
        return next_iter

    def wait_state_loaded(self):
        """Block until load_state has landed; returns the iteration to resume at."""
        if not self._state_loaded.wait(timeout=600):
            raise RuntimeError("standby state was not loaded within 600s")
        return self._resume_iter

    def _derive_dag_bucket_modes(self, training_dag: Any) -> None:
        self.stages.param_sharded_ubids = set()
        self.stages.grad_sharded_ubids = set()
        for node in training_dag.nodes.values():
            meta = getattr(node, "node_meta", {}) or {}
            ubid = meta.get("bucket_key")
            if ubid is None:
                continue
            if node.node_kind == "ALL_GATHER_COMM":
                self.stages.param_sharded_ubids.add(ubid)
            elif node.node_kind == "REDUCE_SCATTER_COMM":
                self.stages.grad_sharded_ubids.add(ubid)
        self.stages.zero_managed_ubids = self.stages.param_sharded_ubids | self.stages.grad_sharded_ubids

    def _relocate_meta_devices(self, gm) -> None:
        """Rewrite baked ``torch.device('meta')`` literals to the actor device.

        Meta-device compilation captures whatever device was current at trace
        time (inside ``type_as`` / ``.to(device=x.device)`` / device-aware
        factory ops like ``torch.zeros(..., device=...)``). That literal
        round-trips through serialization as ``meta`` and would otherwise make
        those ops emit meta tensors at runtime, mismatching the CUDA inputs.
        Lifted params/buffers/inputs are unaffected (placed on device directly),
        so this only touches device literals embedded in node args/kwargs.
        """
        import torch.fx as fx

        device = self.runtime.device
        replaced = 0

        def _fix(value):
            nonlocal replaced
            if isinstance(value, torch.device):
                if value.type == "meta":
                    replaced += 1
                    return torch.device(device)
                return value
            if isinstance(value, tuple):
                return tuple(_fix(v) for v in value)
            if isinstance(value, list):
                return [_fix(v) for v in value]
            if isinstance(value, dict):
                return {k: _fix(v) for k, v in value.items()}
            return value

        for module in gm.modules():
            if not isinstance(module, fx.GraphModule):
                continue
            before = replaced
            for node in module.graph.nodes:
                node.args = _fix(node.args)
                node.kwargs = _fix(node.kwargs)
            if replaced != before:
                module.recompile()

    def _load_stage(
        self,
        stage_id: int,
        modules_data: list,
        a2a_boundaries: dict = None,
        use_activation_checkpointing: bool = False,
    ) -> None:
        """Load a (possibly bucketed) stage.

        *modules_data* is a list of dicts, one per module/bucket, each with keys:
        ``gm_data``, ``graphargs``, ``input_idxs``, ``param_idxs``, ``bucket_key``.

        A non-bucketed stage is represented as a single-element list.
        All per-bucket data structures are keyed by ``bucket_key`` so run_dag
        can look up bucket data without knowing which stage owns a bucket.
        """
        self.logger.debug(
            f"Loading stage {stage_id} ({len(modules_data)} module(s)) on actor {self.runtime.global_rank}"
        )

        g = torch.Generator(device=self.runtime.device)
        g.manual_seed(1000 * self.runtime.global_rank + stage_id)

        first_gm = None

        for b_idx, bd in enumerate(modules_data):
            ubid: Any = bd["bucket_key"]
            bucket = self.stages.ensure_bucket(ubid)
            ac_num_subgraphs = int(bd.get("ac_num_subgraphs", 1))
            ac_requested_subgraphs = int(bd.get("ac_requested_subgraphs", ac_num_subgraphs))
            gms = [_deserialize_graphmodule(gm_data) for gm_data in bd["gm_data_list"]] if "gm_data_list" in bd else [_deserialize_graphmodule(bd["gm_data"])]
            for gm in gms:
                self._relocate_meta_devices(gm)
            if self.use_inductor:
                compiled_gms = []
                for subgraph_idx, gm in enumerate(gms):
                    compiled_gm = torch.compile(gm)
                    compiled_gms.append(compiled_gm)
                    self.logger.debug(
                        f"[load_stage_compile] rank={self.runtime.global_rank} stage={stage_id} "
                        f"ubid={ubid} subgraph={subgraph_idx} compiled=True"
                    )
                gms = compiled_gms

            if b_idx == 0:
                first_gm = gms[0]

            forward_args = list(bd["graphargs"])
            b_input_idxs = list(bd["input_idxs"])
            b_param_idxs = list(bd["param_idxs"])
            apply_zero = bool(bd.get("apply_zero", True))
            shared_placeholder_names = list(
                bd.get("shared_placeholder_names", bd.get("placeholder_names", []))
            )
            # Extract FX placeholder names for each param index.
            bucket.param_names = [
                shared_placeholder_names[i] if i < len(shared_placeholder_names) else f"ubid{ubid}_p{i}"
                for i in b_param_idxs
            ]

            # Save input tensor metadata for pre-allocating FWD recv buffers.
            # Stored as a list of (shape, dtype, requires_grad) in input-slot order.
            recv_meta = []
            for i in b_input_idxs:
                meta = forward_args[i]
                if meta is not None:
                    recv_meta.append((tuple(meta.shape), meta.dtype,
                                      bool(getattr(meta, "requires_grad", False))))
                forward_args[i] = None  # clear slot; run_dag will fill it at execution time
            bucket.forward_input_meta = recv_meta

            bucket.input_idxs = b_input_idxs

            # Realize parameter tensors.
            realized = [None] * len(forward_args)
            for i, arg in enumerate(forward_args):
                if arg is None:
                    continue
                t = torch.empty(arg.shape, dtype=arg.dtype, device=self.runtime.device)
                if arg.requires_grad:
                    t.requires_grad_(True)
                    torch.nn.init.normal_(t, mean=0.0, std=0.02, generator=g)
                else:
                    # Non-trainable slot: try to fill from const attrs (freqs_cis, mask, …)
                    # before falling back to zeros.  Dynamo names direct model attrs as
                    # "l_self_<attr_name>", so strip that prefix to get the bare name.
                    ph_name = shared_placeholder_names[i] if i < len(shared_placeholder_names) else ""
                    attr_name = re.sub(r'^l_self_', '', ph_name)
                    const_val = self.model_const_attrs.get(attr_name)
                    if (
                        const_val is not None
                        and tuple(const_val.shape) == tuple(arg.shape)
                        and const_val.dtype == arg.dtype
                    ):
                        t.copy_(const_val)
                    else:
                        t.zero_()
                realized[i] = t

            shared_name_to_idx = {
                name: i for i, name in enumerate(shared_placeholder_names)
            }
            subgraph_specs = []
            for gm in gms:
                sub_placeholder_names = [
                    n.name for n in gm.graph.nodes if n.op == "placeholder"
                ]
                dynamic_names = [
                    name for name in sub_placeholder_names
                    if name not in shared_name_to_idx
                ]
                subgraph_specs.append((gm.forward, sub_placeholder_names, dynamic_names))

            def _bucket_forward_runner(
                shared_args,
                _specs=subgraph_specs,
                _name_to_idx=shared_name_to_idx,
                _use_ac=use_activation_checkpointing,
                _stage_id=stage_id,
                _ubid=ubid,
                _shared_placeholder_names=tuple(shared_placeholder_names),
            ):
                out = None
                for subgraph_idx, (forward_impl, placeholder_names, dynamic_names) in enumerate(_specs):
                    if len(dynamic_names) == 0:
                        dynamic_values = []
                    elif len(dynamic_names) == 1:
                        dyn_arg = out
                        if isinstance(dyn_arg, (tuple, list)) and len(dyn_arg) == 1:
                            dyn_arg = dyn_arg[0]
                        dynamic_values = [dyn_arg]
                    else:
                        if not isinstance(out, (tuple, list)):
                            raise RuntimeError(
                                f"Bucket forward expected {len(dynamic_names)} dynamic inputs "
                                f"but previous subgraph produced {type(out).__name__}"
                            )
                        dynamic_values = list(out)
                        if len(dynamic_values) != len(dynamic_names):
                            raise RuntimeError(
                                f"Bucket forward expected {len(dynamic_names)} dynamic inputs "
                                f"but previous subgraph produced {len(dynamic_values)} values"
                            )
                    dynamic_name_to_value = {
                        name: value for name, value in zip(dynamic_names, dynamic_values)
                    }
                    call_args = []
                    for name in placeholder_names:
                        if name in dynamic_name_to_value:
                            call_args.append(dynamic_name_to_value[name])
                        else:
                            call_args.append(shared_args[_name_to_idx[name]])
                    if _use_ac:
                        out = torch.utils.checkpoint.checkpoint(
                            forward_impl,
                            *call_args,
                            use_reentrant=False,
                        )
                    else:
                        out = forward_impl(*call_args)
                return out

            bucket.forward_fn = _bucket_forward_runner
            bucket.forward_args = realized
            bucket.param_idxs = b_param_idxs
            bucket.activation_checkpoint_subgraph_count = ac_num_subgraphs
            trainable_idxs = [
                i for i in b_param_idxs
                if realized[i] is not None and realized[i].requires_grad
            ]
            bucket.trainable_param_idxs = trainable_idxs

            zero_managed = (
                self.runtime.dp_degree > 1
                and apply_zero
                and bool(trainable_idxs)
                and ubid in self.stages.zero_managed_ubids
            )
            params_sharded = ubid in self.stages.param_sharded_ubids
            grads_sharded = ubid in self.stages.grad_sharded_ubids

            if zero_managed:
                trainable = [realized[i] for i in trainable_idxs]
                flat_params = torch.cat([p.detach().view(-1) for p in trainable]).contiguous()
                flat_params.requires_grad_(False)
                orig_numel = flat_params.numel()
                dp = self.runtime.dp_degree
                shard_size = (orig_numel + dp - 1) // dp
                padded_numel = shard_size * dp
                if padded_numel > orig_numel:
                    padded = flat_params.new_zeros(padded_numel)
                    padded[:orig_numel].copy_(flat_params)
                    flat_params = padded

                offset = 0
                view_specs = []
                for idx, p in zip(trainable_idxs, trainable):
                    numel = p.numel()
                    realized[idx] = realized[idx].detach()
                    realized[idx].data = flat_params[offset:offset + numel].view(p.shape)
                    realized[idx].requires_grad_(True)
                    realized[idx].grad_dtype = self.params.grad_buffer_dtype
                    view_specs.append((realized[idx], offset, numel, tuple(p.shape)))
                    offset += numel

                shard_start = self.runtime.dp_rank * shard_size
                if params_sharded:
                    shard_param = flat_params[shard_start:shard_start + shard_size].detach().clone()
                else:
                    shard_param = flat_params[shard_start:shard_start + shard_size]
                shard_param.requires_grad_(True)
                # Keep ZeRO comm/storage buffers in fp32, but let the optimizer consume
                # grads in the shard-param dtype to match Adam's bf16 foreach path.
                shard_param.grad_dtype = None

                bucket.flat_params = flat_params
                bucket.flat_grads = (
                    None
                    if grads_sharded
                    else torch.zeros(padded_numel, dtype=self.params.grad_buffer_dtype, device=self.runtime.device)
                )
                bucket.shard_param = shard_param
                bucket.shard_optimizer = self.optim_class([shard_param])
                bucket.reduce_scatter_grads = (
                    torch.zeros(shard_size, dtype=self.params.grad_buffer_dtype, device=self.runtime.device)
                    if grads_sharded
                    else None
                )
                bucket.param_shard_info = (shard_start, shard_size, orig_numel)
                bucket.param_view_specs = view_specs
                bucket.full_params_fresh = False

                if params_sharded:
                    storage = flat_params.untyped_storage()
                    if storage.size() != 0:
                        storage.resize_(0)

                optim = None
            else:
                # Keep trainable params as separate tensors for the non-ZeRO path.
                bucket.param_view_specs = []
                bucket.flat_params = None
                bucket.flat_grads = None
                bucket.shard_param = None
                bucket.shard_optimizer = None
                bucket.reduce_scatter_grads = None
                bucket.param_shard_info = None
                bucket.full_params_fresh = False
                trainable_for_optim = [realized[i] for i in trainable_idxs]
                optim = self.optim_class(trainable_for_optim, fused=True) if trainable_for_optim else None
            bucket.optimizer = optim

        # Keep first GraphModule for compatibility with external inspection tools.
        self.stages.graph_modules[stage_id] = first_gm

    def load_training_dag(self, training_dag: Any) -> None:
        """Load per-PP-rank TrainingDAG compute nodes using existing _load_stage logic.

        This is an adapter for the new TrainingDAG representation where each
        COMPUTE/FWD node with ``node_meta['gm']`` is equivalent to one bucket/module.
        The runtime executes TrainingDAG nodes via ``run_dag``; this method
        only populates actor-side bucket/module state.
        """
        if training_dag is None:
            raise ValueError("load_training_dag requires non-None training_dag")
        if not hasattr(training_dag, "nodes"):
            raise TypeError("load_training_dag expected object with 'nodes' field")

        self._derive_dag_bucket_modes(training_dag)
        self.runtime.initialize_streams_for_training_dag(training_dag)

        # Clear any prior module state before loading a new training DAG.
        self.stages.clear_loaded_modules()

        stage_to_modules: dict[int, list[dict]] = defaultdict(list)

        # Deterministic order by stage/segment then uid.
        compute_nodes = [
            n for n in training_dag.nodes.values()
            if getattr(n, "node_kind", None) == "COMPUTE"
            and getattr(n, "compute_subkind", None) == "FWD"
        ]
        compute_nodes.sort(
            key=lambda n: (
                int(getattr(n, "node_meta", {}).get("stage_id", 10**9)),
                int(getattr(n, "node_meta", {}).get("segment_id", 10**9)),
                str(getattr(n, "uid", "")),
            )
        )

        seen_bucket_keys: set[Any] = set()
        skipped_dupe_fwd_nodes = 0
        for node in compute_nodes:
            meta = getattr(node, "node_meta", {}) or {}
            bucket_key = meta.get("bucket_key", getattr(node, "uid", None))
            if bucket_key in seen_bucket_keys:
                skipped_dupe_fwd_nodes += 1
                continue
            seen_bucket_keys.add(bucket_key)
            gm = meta.get("gm")
            gm_data = meta.get("gm_data")
            stage_id = meta.get("stage_id")
            input_idxs = meta.get("input_idxs")
            param_idxs = meta.get("param_idxs")
            graphargs = meta.get("graphargs")
            input_names = meta.get("input_names", [])
            output_names = meta.get("output_names", [])

            if gm_data is None and gm is None:
                raise ValueError(
                    f"load_training_dag: compute node {getattr(node, 'uid', '<unknown>')} "
                    f"is missing required metadata field 'gm_data' (or fallback 'gm')"
                )
            if stage_id is None or input_idxs is None or param_idxs is None or graphargs is None:
                raise ValueError(
                    f"load_training_dag: compute node {getattr(node, 'uid', '<unknown>')} "
                    f"is missing required metadata fields"
                )

            module_data = {
                "gm_data": gm_data if gm_data is not None else _serialize_graphmodule(gm),
                "graphargs": list(graphargs),
                "input_idxs": list(input_idxs),
                "param_idxs": list(param_idxs),
                "placeholder_names": list(input_names),
                "output_names": list(output_names),
                "shared_placeholder_names": list(input_names),
                "bucket_key": bucket_key,
                "ac_num_subgraphs": 1,
                "ac_requested_subgraphs": 1,
                "apply_zero": bool(meta.get("apply_zero", True)),
                "training_dag_uid": getattr(node, "uid", None),
                "triton_constant_args": dict(meta.get("triton_constant_args", {})),
            }
            stage_to_modules[int(stage_id)].append(module_data)

        assert stage_to_modules, (
            "load_training_dag: no FWD compute nodes with gm metadata found"
        )

        for stage_id in sorted(stage_to_modules.keys()):
            self._load_stage(
                stage_id=stage_id,
                modules_data=stage_to_modules[stage_id],
                a2a_boundaries={},
                use_activation_checkpointing=False,
            )

        # Build runtime adjacency on TrainingDAG nodes so run_dag can execute directly.
        for n in training_dag.nodes.values():
            n.data_preds = []
            n.data_succs = []
            n.temporal_preds = []
            n.temporal_succs = []
            if n.node_kind == "SEND_COMM":
                n.peer_pp_rank = n.node_meta.get("peer_pp_rank")
            elif n.node_kind == "RECV_COMM":
                n.peer_pp_rank = n.node_meta.get("peer_pp_rank")
            else:
                n.peer_pp_rank = None
        for e in training_dag.edges:
            if e.src_uid not in training_dag.nodes or e.dst_uid not in training_dag.nodes:
                continue
            src = training_dag.nodes[e.src_uid]
            dst = training_dag.nodes[e.dst_uid]
            if e.dep_kind == "temporal":
                src.temporal_succs.append(dst)
                dst.temporal_preds.append(src)
            else:
                src.data_succs.append(dst)
                dst.data_preds.append(src)

        for n in training_dag.nodes.values():
            n.task_type = _training_dag_task_type(n)
            mb = n.tag.get("MB", 0)
            st = n.tag.get("PP", 0)
            n.batches = [type("RuntimeBatch", (), {"stage_id": st, "mb_idx": mb})()]

        self.dag = training_dag
        self.sorted_dag_nodes = [
            training_dag.nodes[uid] for uid in _serial_topological_order(training_dag)
        ]

    def run_dag(self, loss_fn=None):
        # Mark the entire iteration boundary for the NVTX timeline.
        iter_idx = getattr(self, "_iter_counter", 0)
        self._iter_counter = iter_idx + 1
        # Debug-only: expose the iteration counter for E2E fault injection.
        self.dag_executor._iter_count = iter_idx
        self._nvtx_push(f"iter_{iter_idx}_rank_{self.runtime.global_rank}")
        self.dag_executor.run(
            self.dag,
            self.sorted_dag_nodes,
            self.inputs,
            self.labels,
            self.loss,
            loss_fn=loss_fn,
        )
        self._nvtx_pop()
