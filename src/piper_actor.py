import ray
import torch
import logging
import os
import time
import gc
from torch._guards import CompileId
from torch.nn import Parameter
from collections import defaultdict
from .piper_utils import deserialize_graphmodule, piper_metadata, create_logger
import torch.distributed as dist
from typing import Callable

CLEANUP_MEMORY = False

@ray.remote
class PiperActor:
    def __init__(
        self,
        actor_id: int,
        world_size: int,
        dp_rank: int = 0,
        dp_degree: int = 1,
        pp_degree: int = 1,
        optim_fn: Callable | None = None,
        overlap_grad_sync: bool = True,
    ):
        self.logger = create_logger("piper_actor", "INFO")
        
        start = time.perf_counter()

        self.actor_id = actor_id
        self.optim_fn = optim_fn
        
        # Data parallel attributes
        self.dp_rank = dp_rank
        self.dp_degree = dp_degree
        self.world_size = world_size
        self.dp_group = None
        self.pp_degree = pp_degree

        if pp_degree == 1:
            self.global_rank = dp_rank
        elif dp_degree == 1:
            self.global_rank = actor_id
        else:
            self.global_rank = dp_rank * dp_degree + actor_id

        self.logger.info(f"Initializing Ray actor {actor_id} global rank {self.global_rank} with PID: {os.getpid()}")

        self.input = None
        self.truth = None
        self.fwd_objs = {}
        self.bwd_objs = {}

        # ordered list of frame ids for ordering the fx.Graphs on this actor
        self.frame_ids = []
        # map stage id -> compiled fx.Graph function
        self.compiled_fns = dict()
        # map stage id -> original GraphModule (for hook registration)
        self.graph_modules = dict()
        # map stage id -> model parameters used by the fx.Graph with holes (None values) for input tensors
        self.parameters = dict()
        # map stage id -> indices of the input tensors (as opposed to model parameters) used by the fx.Graph
        self.input_idxs = dict()
        # map stage id -> optimizer for the fx.Graph
        self.optims = dict()
        # map stage -> mb_idx -> previous activation (if this stage is not first)
        self.prev_activation = defaultdict(dict)
        # map stage id -> mb_idx -> current activation
        self.activation = defaultdict(dict)
        # accumuate loss for each microbatch
        self.loss = []
        # map stage_id -> list of (param_group, optimizer) for overlapped sync+step
        self.overlap_grad_sync = overlap_grad_sync
        if overlap_grad_sync:
            self.param_group_optims: dict[int, list] = {}

        # Timing infrastructure
        self.tracing = False  # Toggle for timing and memory tracing
        self.trace_data = {'update': {'total': [], 'peak_memory_delta': [], 'peak_memory': []}}

        end = time.perf_counter()
        self.logger.debug(f"__init__ took {(end-start)*1000:.2f}ms")

        self.logger.debug(f"Initialized actor {self.actor_id} for global rank {self.global_rank}")

    def id(self):
        return self.actor_id

    def send_input(self, tensor):
        self.input = tensor.to('cuda')
        return "done"
    
    def send_truth(self, tensor):
        self.truth = tensor.to('cuda')
        return "done"

    def reset_peak_memory(self):
        torch.cuda.reset_peak_memory_stats()
        return "done"

    def get_peak_memory(self):
        return torch.cuda.max_memory_allocated() / (1024**3)

    @ray.method
    def join_process_groups(self):
        master_addr = os.environ.get('PIPER_MASTER_ADDR', "127.0.0.1")
        master_port = os.environ.get('PIPER_MASTER_PORT', "10000")
        init_method = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group("nccl", init_method=init_method, rank=self.global_rank, world_size=self.world_size)
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} has GPU {os.environ['CUDA_VISIBLE_DEVICES']}, joined the global process group")

        if self.dp_degree > 1:
            self.join_dp_process_group()
    
    def join_dp_process_group(self):
        # Every process needs to participate in every subgroup creation
        num_dp_groups = self.world_size // self.dp_degree
        for dp_group_id in range(num_dp_groups):
            group_ranks = [(dp_group_id + num_dp_groups * i) for i in range(self.dp_degree)]
            process_group = dist.new_group(ranks=group_ranks, backend='nccl')
            if self.global_rank % num_dp_groups == dp_group_id:
                self.dp_group = process_group
                self.logger.info(f"Global rank {self.global_rank} joined its dp group {dp_group_id} along with ranks {group_ranks}")

    def compile_graph(self, stage_id, gm_data, compiler_fn, graphargs, input_idxs):
        self.logger.debug(f"Compiling graph on actor {self.actor_id} for stage id: {stage_id} with inputs: {len(graphargs)}")
        start = time.perf_counter()

        # set up tracing data structure
        self.trace_data[stage_id] = {
            'forward': {
                'forward': [],
                'total': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
            'backward': {
                'backward': [],
                'total': [],
                'peak_memory_delta': [],
                'peak_memory': []
            },
        }

        # compile the graph with the given graphargs
        gm = deserialize_graphmodule(gm_data)
        
        # Store GraphModule reference
        self.graph_modules[stage_id] = gm
        
        compiled_fn = compiler_fn(gm, graphargs)
        assert callable(compiled_fn), "compiler_fn did not return callable"
        self.compiled_fns[stage_id] = compiled_fn

        # discard the graphargs that correspond to input tensors
        for i in input_idxs:
            graphargs[i] = None
        self.input_idxs[stage_id] = input_idxs

        # save the graphargs that correspond to model parameters
        self.parameters[stage_id] = graphargs

        # initialize the optimizer for this stage
        params = [param for param in self.parameters[stage_id] if param is not None]
        if params:
            self.optims[stage_id] = self.optim_fn(params)

            if self.overlap_grad_sync:
                # Create per-group optimizers for overlapped sync+step in DP mode
                GROUP_SIZE = 64  # Tunable: trade-off between overlap and optimizer overhead
                param_groups = [
                    params[i:i + GROUP_SIZE]
                    for i in range(0, len(params), GROUP_SIZE)
                ]
                self.param_group_optims[stage_id] = [
                    (group, self.optim_fn(group)) for group in param_groups
                ]

        del gm_data

        end = time.perf_counter()
        self.logger.debug(f"compile_graph took {(end-start)*1000:.2f}ms")
        return "Finished compiling"

    @ray.method(tensor_transport="nccl")
    def forward(self, stage_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            forward_start_event = torch.cuda.Event(enable_timing=True)
            forward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()
        
        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        def unwrap_arg(arg):
            if isinstance(arg, list) or isinstance(arg, tuple):
                assert len(arg) == 1
                return arg[0]
            else:
                return arg
        args = list(map(unwrap_arg, args))

        def detach_arg(arg):
            return arg.detach().clone()
        args = list(map(detach_arg, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
            return x[0] if isinstance(x, list) or isinstance(x, tuple) else x
        args = list(map(unwrap, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as previous activation
        if stage_id != 0:
            if not args[0].requires_grad:
                args[0].requires_grad_()
            self.prev_activation[stage_id][mb_idx] = args[0]

        # Record start event for forward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            forward_start_memory = torch.cuda.memory_allocated()
            forward_start_event.record()

        # Call compiled function
        out = self.compiled_fns[stage_id](*self.parameters[stage_id])

        # Record end event and calculate forward timing
        if self.tracing:
            forward_end_event.record()
            torch.cuda.synchronize()
            
            # Calculate total forward time
            forward_time = forward_start_event.elapsed_time(forward_end_event)
            
            forward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - forward_start_memory) / (1024**3)
            forward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        # save first output as activation BEFORE clearing input tensors
        # This ensures the activation doesn't share memory with parameters
        # that might be modified when processing other stages on the same actor
        activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.activation[stage_id][mb_idx] = activation_tensor

        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = forward_start_event.elapsed_time(end_event)
            # Store in trace_data (all microbatches stored sequentially)
            self.trace_data[stage_id]['forward']['forward'].append(forward_time)
            self.trace_data[stage_id]['forward']['total'].append(forward_time)
            self.trace_data[stage_id]['forward']['peak_memory_delta'].append(forward_peak_memory_delta_gb)
            self.trace_data[stage_id]['forward']['peak_memory'].append(forward_peak_memory_gb)

            if isinstance(out, list) or isinstance(out, tuple):
                self.fwd_objs[stage_id] = [torch.ones_like(t) for t in out]
            else:
                self.fwd_objs[stage_id] = torch.ones_like(out)
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out

    def forward_no_nccl(self, stage_id: int, mb_idx: int, *args):
        self.logger.debug(f"Calling cpu forward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        def pre_loaded_input(param):
            if param is None:
                return self.input
            else:
                return param
        args = list(map(pre_loaded_input, args))

        # Ray object refs resolve to a single element list
        def unwrap(x):
            if isinstance(x, list) or isinstance(x, tuple):
                assert len(x) == 1
                return x[0]
            else:
                return x
        args = list(map(unwrap, args))

        # place input tensors in the correct indices
        for i, arg in zip(self.input_idxs[stage_id], args):
            self.parameters[stage_id][i] = arg

        # save first input as previous activation
        if stage_id != 0:
            assert args[0].requires_grad
            self.prev_activation[stage_id][mb_idx] = args[0]

        out = self.compiled_fns[stage_id](*self.parameters[stage_id])
        
        # save first output as activation BEFORE clearing input tensors
        # This ensures the activation doesn't share memory with parameters
        # that might be modified when processing other stages on the same actor
        activation_tensor = out[0] if isinstance(out, (list, tuple)) else out
        self.activation[stage_id][mb_idx] = activation_tensor
        
        # clear the input tensors
        for i in self.input_idxs[stage_id]:
            self.parameters[stage_id][i] = None
        del args
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return out

    @ray.method(tensor_transport="nccl")
    def backward(self, stage_id: int, mb_idx: int, inp, loss_fn=None):
        self.logger.debug(f"Calling backward {stage_id} mb {mb_idx} on actor {self.actor_id} global rank {self.global_rank}")

        if self.tracing:
            beginning_event = torch.cuda.Event(enable_timing=True)
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            beginning_event.record()

        # get the activation for the current stage
        assert mb_idx in self.activation[stage_id], f"mb_idx {mb_idx} not in activation[stage_id {stage_id}]"
        activation = self.activation[stage_id][mb_idx]
        
        # Record start event for backward timing
        if self.tracing:
            torch.cuda.reset_peak_memory_stats()
            backward_start_memory = torch.cuda.memory_allocated()
            backward_start_event.record()
        
        # compute loss in the last stage. use the saved activation rather
        # than inp because the saved activation remembers the computation graph
        if loss_fn is not None:
            if self.truth is not None:
                labels = self.truth
            else:
                labels = inp[0]
            loss = loss_fn(activation, labels)
            loss.backward()
            self.loss.append(loss.item())
            # Clean up loss tensor after extracting item
            del loss
        # if not the last stage, backprop on the stored activation given 
        # the input gradient from the subsequent stage
        else:
            assert inp is not None
            assert activation.shape == inp.shape
            activation.backward(gradient=inp)

        # Record end event and calculate backward timing
        if self.tracing:
            backward_end_event.record()
            torch.cuda.synchronize()
            
            # Calculate total backward time
            backward_time = backward_start_event.elapsed_time(backward_end_event)
            
            backward_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - backward_start_memory) / (1024**3)
            backward_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

        del self.activation[stage_id][mb_idx]
        del activation

        # propagate the gradient backwards if not the first stage
        if stage_id != 0:
            ret = [self.prev_activation[stage_id][mb_idx]]
        else:
            ret = ["done"]
        
        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = beginning_event.elapsed_time(end_event)
            # Store in trace_data (all microbatches stored sequentially)
            self.trace_data[stage_id]['backward']['backward'].append(backward_time)
            self.trace_data[stage_id]['backward']['total'].append(total_time)
            self.trace_data[stage_id]['backward']['peak_memory_delta'].append(backward_peak_memory_delta_gb)
            self.trace_data[stage_id]['backward']['peak_memory'].append(backward_peak_memory_gb)
            if stage_id != 0:
                self.bwd_objs[stage_id] = torch.ones_like(ret[0])
        
        if CLEANUP_MEMORY:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return ret + ret

    def synchronize_gradients(self):
        """Synchronize gradients across all DP ranks for this stage using all-reduce."""
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} synchronizing gradients")

        # Iterate over all stages on this actor and synchronize their parameters
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=self.dp_group)

    def _overlapped_sync_and_step(self):
        """Overlap gradient sync with optimizer step using per-group optimizers."""
        self.logger.debug(f"Actor {self.actor_id} global rank {self.global_rank} running overlapped sync+step")
        comm_stream = torch.cuda.Stream()

        for stage_id in self.parameters:
            if stage_id not in self.param_group_optims:
                continue

            for param_group, optim in self.param_group_optims[stage_id]:
                handles = []
                with torch.cuda.stream(comm_stream):
                    for param in param_group:
                        if param.grad is not None:
                            h = dist.all_reduce(
                                param.grad,
                                op=dist.ReduceOp.AVG,
                                group=self.dp_group,
                                async_op=True
                            )
                            handles.append(h)

                for h in handles:
                    h.wait()
                optim.step()
                optim.zero_grad()

    def update(self, *done_mbs):
        self.logger.debug(f"Calling update on actor {self.actor_id} global rank {self.global_rank}")
        
        if self.tracing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            torch.cuda.reset_peak_memory_stats()
            update_start_memory = torch.cuda.memory_allocated()

            start_event.record()
        
        # step the optimizer for each stage
        assert self.optim_fn
        if self.dp_degree > 1:
            if self.overlap_grad_sync:
                # Use overlapped sync+step for DP mode
                self._overlapped_sync_and_step()
            else:
                # Standard all-reduce then step
                self.synchronize_gradients()
                for _, optim in self.optims.items():
                    optim.step()
                    optim.zero_grad()
        else:
            # Non-DP mode: just step optimizers
            for _, optim in self.optims.items():
                optim.step()
                optim.zero_grad()
        losses = self.loss
        self.loss.clear()

        if self.tracing:
            end_event.record()
            torch.cuda.synchronize()
            total_time = start_event.elapsed_time(end_event)
            update_peak_memory_delta_gb = (torch.cuda.max_memory_allocated() - update_start_memory) / (1024**3)
            update_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)

            self.trace_data['update']['total'].append(total_time)
            self.trace_data['update']['peak_memory_delta'].append(update_peak_memory_delta_gb)
            self.trace_data['update']['peak_memory'].append(update_peak_memory_gb)

        return losses

    def verify_weights(self, msg="first parameter"):
        for stage_id, parameters in self.parameters.items():
            for param in parameters:
                if param is not None and param.grad is not None:
                    if len(param.shape) == 2:
                        self.logger.info(f"Actor {self.actor_id} global rank {self.global_rank} stage {stage_id} {msg}: {param.shape, param[0][0], param.grad[0][0]}")
                    elif len(param.shape) == 1:
                        self.logger.info(f"Actor {self.actor_id} global rank {self.global_rank} stage {stage_id} {msg}: {param.shape, param[0], param.grad[0]}")
                    else:
                        assert False, f"Unsupported parameter shape: {param.shape}"
                    return "done"

    def clear_trace_data(self) -> None:
        """
        Clear all collected timing data.
        """
        for stage_id in self.trace_data:
            self.trace_data[stage_id] = {
                'forward': {
                    'forward': [],
                    'total': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
                'backward': {
                    'backward': [],
                    'total': [],
                    'peak_memory_delta': [],
                    'peak_memory': []
                },
            }
        self.trace_data['update'] = {
            'total': [],
            'peak_memory_delta': [],
            'peak_memory': []
        }
    
    def get_trace_data(self) -> dict:
        return self.trace_data

    def set_tracing(self, enabled: bool) -> None:
        """
        Enable or disable timing and memory tracing.
        
        Args:
            enabled (bool): True to enable tracing, False to disable.
        """
        self.tracing = enabled
        self.logger.info(f"Actor {self.actor_id}: Tracing {'enabled' if enabled else 'disabled'}")

    def start_mem_tracing(self) -> None:
        torch.cuda.memory._record_memory_history()
        return "done"

    def stop_mem_tracing(self) -> None:
        torch.cuda.memory._dump_snapshot(f"actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        self.logger.info(f"Saved memory snapshot to actor{self.actor_id}_memory_snapshot_mb4_gpipe.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return "done"
