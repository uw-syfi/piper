#!/usr/bin/env python3
"""Replay significant compute/communication trace classes on eight H100s.

The input is produced by ``analyze_trace_compute_comm_overlaps.py`` with both
``--threshold 0.75`` and ``--min-kernel-us 1000``.  Shape-only variants are
collapsed into nine resource-similar compute classes while all five recorded
communication kinds remain distinct.  For every observed class/collective
combination this standalone program:

1. measures and profiles both operations in isolation;
2. predicts compute-first as serialized execution when compute consumes all
   SMs;
3. predicts communication-first by remeasuring compute in a CUDA Green Context
   with the communication kernel's profiled CTA demand removed; and
4. measures both unrestricted concurrent launch orders.

The physical replay uses the eight GPUs available on one node.  The source
trace used EP/DP group size 16, so all-gather and reduce-scatter preserve the
recorded per-rank input size but necessarily have an eight-rank output shard
ratio.  P2P and all-to-allv preserve their representative per-rank payloads.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import benchmark_grouped_gemm_allreduce_sm_partition as common
import analyze_trace_compute_comm_overlaps as trace_analysis


COMPUTE_CLASSES = (
    "expert_grouped_gemm",
    "sdpa_forward",
    "sdpa_backward",
    "vocab_projection_gemm",
    "log_softmax",
    "router_bmm",
    "expert_activation_pointwise",
    "routing_permutation",
    "elementwise_copy",
)
COMM_KINDS = ("all_to_allv", "all_gather", "reduce_scatter", "send", "recv")

MODEL = {
    "name": "Qwen3-30B-A3B-empirical1to1-L4",
    "sequence_length": 12_288,
    "hidden_dim": 2_048,
    "attention_heads": 32,
    "kv_heads": 4,
    "head_dim": 128,
    "expert_width": 1_344,
    "experts": 128,
    "top_k": 8,
    "logical_ep": 16,
    "vocab_size": 151_936,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("out/20260804_104256/compute_comm_overlap_gt1ms.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("out/20260804_104256/launch_order_benchmarks/results.json"),
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode", choices=["baseline", "restricted"], help=argparse.SUPPRESS
    )
    parser.add_argument("--phase-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--trace-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--comm-demand", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_cuda_compat_first() -> None:
    compat = "/usr/local/cuda-12.8/compat"
    if not Path(compat, "libcuda.so.1").exists():
        return
    paths = [item for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item]
    if paths and paths[0] == compat:
        return
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        [compat, *(item for item in paths if item != compat)]
    )
    os.execvpe(
        sys.executable,
        [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        environment,
    )


def load_analysis(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    reduction = result.get("benchmark_reduction") or {}
    if int(reduction.get("benchmark_case_count", 0)) != 42:
        raise RuntimeError(
            f"expected the 42-case benchmark reduction in {path}; rerun the analyzer"
        )
    return result


def representative_communications(analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    weights: Counter[tuple[Any, ...]] = Counter()
    exemplars: dict[tuple[Any, ...], dict[str, Any]] = {}
    for case in analysis["cases"]:
        comm = case["communication"]
        kind = trace_analysis.collective_family(comm["collective_name"])
        key = (
            kind,
            comm["collective_name"],
            comm["in_msg_nelems"],
            comm["out_msg_nelems"],
            comm["group_size"],
            comm["dtype"],
            tuple(comm["grid"]),
            tuple(comm["block"]),
        )
        weights[key] += int(case["observation_count"])
        exemplars[key] = comm
    selected = {}
    for kind in COMM_KINDS:
        candidates = [(count, key) for key, count in weights.items() if key[0] == kind]
        if not candidates:
            continue
        _, key = max(candidates)
        selected[kind] = exemplars[key]
    return selected


def benchmark_pairs(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    return list(analysis["benchmark_reduction"]["cases"])


def make_offsets(torch: Any, groups: int, rows_per_group: int) -> Any:
    rows = groups * rows_per_group
    return torch.arange(
        rows_per_group,
        rows + 1,
        rows_per_group,
        dtype=torch.int32,
        device="cuda",
    )


def make_compute(torch: Any, compute_class: str) -> tuple[Callable[[], Any], list[Any], dict[str, Any]]:
    seq = MODEL["sequence_length"]
    dim = MODEL["hidden_dim"]
    width = MODEL["expert_width"]
    groups = MODEL["experts"] // MODEL["logical_ep"]
    tokens_per_expert = (
        MODEL["logical_ep"] * seq * MODEL["top_k"] // MODEL["experts"]
    )
    dtype = torch.bfloat16
    owned: list[Any] = []

    def empty(shape: tuple[int, ...], *, tensor_dtype: Any = dtype, requires_grad: bool = False) -> Any:
        value = torch.empty(shape, dtype=tensor_dtype, device="cuda", requires_grad=requires_grad)
        owned.append(value)
        return value

    if compute_class == "expert_grouped_gemm":
        # Representative expert dW grouped GEMM. Each expert multiplies
        # [2*width, tokens] by [tokens, dim].
        m = 2 * width
        lhs = empty((groups * m + 1, tokens_per_expert))
        rhs = empty((groups, tokens_per_expert, dim))
        offsets = make_offsets(torch, groups, m)
        owned.append(offsets)

        def launch() -> Any:
            return torch._grouped_mm(lhs, rhs, offs=offsets)

        meta = {
            "operation": "expert grouped dW GEMM",
            "groups": groups,
            "per_group": [[m, tokens_per_expert], [tokens_per_expert, dim]],
        }
    elif compute_class in {"sdpa_forward", "sdpa_backward"}:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        import torch.nn.functional as functional

        q = empty(
            (1, MODEL["attention_heads"], seq, MODEL["head_dim"]),
        )
        # TorchTitan expands GQA K/V heads before the cuDNN SDPA call.
        k = empty(
            (1, MODEL["attention_heads"], seq, MODEL["head_dim"]),
        )
        v = empty(
            (1, MODEL["attention_heads"], seq, MODEL["head_dim"]),
        )

        def attention() -> Any:
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True
                )

        if compute_class == "sdpa_forward":
            launch = attention
        else:
            # Call the ATen cuDNN operators directly so repeated isolated
            # backward launches remain valid in a Green Context. Autograd's
            # engine changes CUDA context state after the first Green launch.
            saved = torch.ops.aten._cudnn_attention_forward.default(
                q,
                k,
                v,
                None,
                None,
                None,
                seq,
                seq,
                True,
                0.0,
                True,
                False,
                scale=None,
            )
            output, logsumexp = saved[0], saved[1]
            gradient = empty(tuple(output.shape))
            owned.extend(value for value in saved if isinstance(value, torch.Tensor))

            def launch() -> Any:
                return torch.ops.aten._cudnn_attention_backward.default(
                    gradient,
                    q,
                    k,
                    v,
                    output,
                    logsumexp,
                    saved[6],
                    saved[7],
                    None,
                    saved[2],
                    saved[3],
                    saved[4],
                    saved[5],
                    0.0,
                    True,
                    scale=None,
                )

        meta = {
            "operation": compute_class,
            "qkv_shape": [1, MODEL["attention_heads"], seq, MODEL["head_dim"]],
            "causal": True,
        }
    elif compute_class == "vocab_projection_gemm":
        # Most-observed projection family member is the weight-gradient GEMM.
        output_gradient = empty((seq, MODEL["vocab_size"]))
        hidden = empty((seq, dim))

        def launch() -> Any:
            return torch.mm(output_gradient.transpose(0, 1), hidden)

        meta = {
            "operation": "vocabulary projection weight-gradient GEMM",
            "lhs": [MODEL["vocab_size"], seq],
            "rhs": [seq, dim],
        }
    elif compute_class == "log_softmax":
        output = empty((seq, MODEL["vocab_size"]))
        gradient = empty((seq, MODEL["vocab_size"]))

        def launch() -> Any:
            return torch.ops.aten._log_softmax_backward_data.default(
                gradient, output, -1, torch.bfloat16
            )

        meta = {
            "operation": "vocabulary log-softmax backward",
            "shape": [seq, MODEL["vocab_size"]],
        }
    elif compute_class == "router_bmm":
        scores = empty((seq, 1, MODEL["top_k"]), tensor_dtype=torch.float32)
        routed = empty((seq, MODEL["top_k"], dim), tensor_dtype=torch.float32)

        def launch() -> Any:
            return torch.bmm(scores, routed)

        meta = {
            "operation": "router weighted-output BMM",
            "lhs": [seq, 1, MODEL["top_k"]],
            "rhs": [seq, MODEL["top_k"], dim],
        }
    elif compute_class == "expert_activation_pointwise":
        # The dominant trace variant has 516,264 CTAs at XBLOCK=256.
        numel = 516_264 * 256
        gate = empty((numel,))
        up = empty((numel,))

        def launch() -> Any:
            # Keep this as native eager kernels. TorchInductor's Python wrapper
            # restores the primary CUDA context before launching a cached
            # Triton kernel, which defeats Green-context restriction.
            return torch.nn.functional.silu(gate) * up

        meta = {"operation": "native SwiGLU pointwise", "numel": numel}
    elif compute_class == "routing_permutation":
        # Reproduce the dominant index_put routing kernel's 786,688-CTA grid.
        rows = 196_672
        output = empty((rows, dim))
        source = empty((rows, dim))
        index = torch.arange(rows - 1, -1, -1, dtype=torch.long, device="cuda")
        owned.append(index)

        def launch() -> Any:
            output[index] = source
            return output

        meta = {
            "operation": "routed BF16 index_put permutation",
            "shape": [rows, dim],
        }
    elif compute_class == "elementwise_copy":
        # Reproduce the trace's native float-add kernel: 303,872 blocks,
        # 128 threads/block, four vectorized elements/thread.
        numel = 303_872 * 128 * 4
        lhs = empty((numel,), tensor_dtype=torch.float32)
        rhs = empty((numel,), tensor_dtype=torch.float32)

        def launch() -> Any:
            return torch.add(lhs, rhs)

        meta = {"operation": "native FP32 elementwise add", "numel": numel}
    else:
        raise ValueError(compute_class)
    return launch, owned, meta


def torch_dtype(torch: Any, dtype_name: str) -> Any:
    return {
        "BFloat16": torch.bfloat16,
        "Float": torch.float32,
        "Long": torch.int64,
    }[dtype_name]


def make_communication(
    torch: Any,
    dist: Any,
    rank: int,
    world_size: int,
    kind: str,
    spec: dict[str, Any],
) -> tuple[Callable[[], Any], list[Any], dict[str, Any]]:
    dtype = torch_dtype(torch, str(spec["dtype"]))
    in_numel = int(spec["in_msg_nelems"])
    out_numel = int(spec["out_msg_nelems"])
    owned: list[Any] = []

    if kind in {"send", "recv"}:
        numel = in_numel
        tensor = torch.empty((numel,), dtype=dtype, device="cuda")
        owned.append(tensor)
        rank_zero_sends = kind == "send"
        sends = (rank % 2 == 0) == rank_zero_sends
        peer = rank + 1 if rank % 2 == 0 else rank - 1

        def launch() -> None:
            work = dist.isend(tensor, peer) if sends else dist.irecv(tensor, peer)
            work.wait()

        physical = {"input_numel": numel, "output_numel": numel}
    elif kind == "all_to_allv":
        # Alternate the representative role and its inverse so total sends and
        # receives balance while rank 0 exactly matches the traced role.
        representative_role = rank % 2 == 0
        local_in = in_numel if representative_role else out_numel
        local_out = out_numel if representative_role else in_numel
        input_tensor = torch.empty((local_in,), dtype=dtype, device="cuda")
        output_tensor = torch.empty((local_out,), dtype=dtype, device="cuda")
        owned.extend([input_tensor, output_tensor])
        peer = rank + 1 if rank % 2 == 0 else rank - 1
        input_splits = [0] * world_size
        output_splits = [0] * world_size
        input_splits[peer] = local_in
        output_splits[peer] = local_out

        def launch() -> None:
            work = dist.all_to_all_single(
                output_tensor,
                input_tensor,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                async_op=True,
            )
            work.wait()

        physical = {"input_numel": local_in, "output_numel": local_out}
    elif kind == "all_gather":
        input_tensor = torch.empty((in_numel,), dtype=dtype, device="cuda")
        output_tensor = torch.empty(
            (world_size * in_numel,), dtype=dtype, device="cuda"
        )
        owned.extend([input_tensor, output_tensor])

        def launch() -> None:
            work = dist.all_gather_into_tensor(
                output_tensor, input_tensor, async_op=True
            )
            work.wait()

        physical = {
            "input_numel": in_numel,
            "output_numel": world_size * in_numel,
        }
    elif kind == "reduce_scatter":
        if in_numel % world_size:
            raise ValueError(f"reduce-scatter input {in_numel} not divisible by {world_size}")
        input_tensor = torch.empty((in_numel,), dtype=dtype, device="cuda")
        output_tensor = torch.empty(
            (in_numel // world_size,), dtype=dtype, device="cuda"
        )
        owned.extend([input_tensor, output_tensor])

        def launch() -> None:
            work = dist.reduce_scatter_tensor(
                output_tensor, input_tensor, async_op=True
            )
            work.wait()

        physical = {
            "input_numel": in_numel,
            "output_numel": in_numel // world_size,
        }
    else:
        raise ValueError(kind)

    metadata = {
        "kind": kind,
        "trace": {
            "collective_name": spec["collective_name"],
            "input_numel": in_numel,
            "output_numel": out_numel,
            "group_size": int(spec["group_size"]),
            "dtype": spec["dtype"],
            "grid": spec["grid"],
            "block": spec["block"],
        },
        "physical_replay": {
            "group_size": world_size if kind not in {"send", "recv"} else 2,
            **physical,
        },
    }
    return launch, owned, metadata


def cleanup(torch: Any, owned: list[Any]) -> None:
    owned.clear()
    gc.collect()
    torch.cuda.empty_cache()


def prepare_operation(
    torch: Any,
    launch: Callable[[], Any],
    before_launch: Callable[[], None] | None = None,
) -> None:
    prepare = getattr(launch, "prepare", None)
    if prepare is None:
        return
    if before_launch is not None:
        before_launch()
    prepare()
    if before_launch is not None:
        before_launch()
    torch.cuda.synchronize()


def time_isolated(
    torch: Any,
    dist: Any,
    control: Any,
    rank: int,
    launch: Callable[[], Any],
    warmups: int,
    iterations: int,
    before_launch: Callable[[], None] | None = None,
) -> tuple[list[float], list[float]]:
    for _ in range(warmups):
        prepare_operation(torch, launch, before_launch)
        dist.barrier(group=control)
        if before_launch is not None:
            before_launch()
        result = launch()
        if before_launch is not None:
            before_launch()
        torch.cuda.synchronize()
        del result
    local = []
    world = []
    for _ in range(iterations):
        prepare_operation(torch, launch, before_launch)
        dist.barrier(group=control)
        if before_launch is not None:
            before_launch()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = launch()
        if before_launch is not None:
            before_launch()
        end.record()
        end.synchronize()
        value = float(start.elapsed_time(end))
        local.append(value)
        gathered = common.gather_sample(control, rank, {"ms": value})
        if gathered is not None:
            world.append(max(float(item["ms"]) for item in gathered))
        del result
    return local, world


def overlap_sample(
    torch: Any,
    compute: Callable[[], Any],
    communication: Callable[[], Any],
    compute_stream: Any,
    comm_stream: Any,
    order: str,
) -> dict[str, Any]:
    def enqueue_compute() -> tuple[Any, Any, Any, int]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_ns = time.perf_counter_ns()
        with torch.cuda.stream(compute_stream):
            start.record()
            result = compute()
            end.record()
        return start, end, result, host_ns

    def enqueue_comm() -> tuple[Any, Any, Any, int]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_ns = time.perf_counter_ns()
        with torch.cuda.stream(comm_stream):
            start.record()
            result = communication()
            end.record()
        return start, end, result, host_ns

    if order == "compute_first":
        cs, ce, compute_result, compute_host = enqueue_compute()
        ns, ne, comm_result, comm_host = enqueue_comm()
    else:
        ns, ne, comm_result, comm_host = enqueue_comm()
        cs, ce, compute_result, compute_host = enqueue_compute()
    ce.synchronize()
    ne.synchronize()
    host_end = time.perf_counter_ns()
    result = {
        "compute_ms": float(cs.elapsed_time(ce)),
        "comm_ms": float(ns.elapsed_time(ne)),
        "compute_minus_comm_start_ms": (compute_host - comm_host) / 1e6,
        "earliest_host_start_ns": min(compute_host, comm_host),
        "latest_host_end_ns": host_end,
    }
    del compute_result, comm_result
    return result


def time_overlap(
    torch: Any,
    dist: Any,
    control: Any,
    rank: int,
    compute: Callable[[], Any],
    communication: Callable[[], Any],
    order: str,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()
    for _ in range(warmups):
        prepare_operation(torch, compute)
        dist.barrier(group=control)
        overlap_sample(torch, compute, communication, compute_stream, comm_stream, order)
    local = []
    world = []
    for _ in range(iterations):
        prepare_operation(torch, compute)
        dist.barrier(group=control)
        sample = overlap_sample(
            torch, compute, communication, compute_stream, comm_stream, order
        )
        local.append(sample)
        gathered = common.gather_sample(control, rank, sample)
        if gathered is not None:
            earliest = min(int(item["earliest_host_start_ns"]) for item in gathered)
            latest = max(int(item["latest_host_end_ns"]) for item in gathered)
            world.append(
                {
                    "completion_ms": (latest - earliest) / 1e6,
                    "max_compute_ms": max(float(item["compute_ms"]) for item in gathered),
                    "max_comm_ms": max(float(item["comm_ms"]) for item in gathered),
                    "start_offset_ms": [
                        float(item["compute_minus_comm_start_ms"]) for item in gathered
                    ],
                }
            )
    if rank != 0:
        return {"local_samples": local}
    return {
        "world_samples": world,
        "summary_ms": common.summarize([item["completion_ms"] for item in world]),
        "compute_summary_ms": common.summarize([item["max_compute_ms"] for item in world]),
        "comm_summary_ms": common.summarize([item["max_comm_ms"] for item in world]),
    }


def profile_operation(
    torch: Any,
    dist: Any,
    control: Any,
    rank: int,
    launch: Callable[[], Any],
    output: Path,
) -> None:
    prepare_operation(torch, launch)
    dist.barrier(group=control)
    if rank == 0:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            result = launch()
            torch.cuda.synchronize()
            del result
        output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(output))
    else:
        result = launch()
        torch.cuda.synchronize()
        del result
    dist.barrier(group=control)


def initialize_worker(torch: Any, need_nccl: bool = True) -> tuple[Any, int, int]:
    return common.initialize_distributed(torch, need_nccl=need_nccl)


def run_baseline_worker(args: argparse.Namespace, torch: Any) -> None:
    import torch.distributed as dist

    control, rank, world_size = initialize_worker(torch)
    analysis = load_analysis(args.analysis)
    comm_specs = representative_communications(analysis)
    pairs = benchmark_pairs(analysis)
    compute_results: dict[str, Any] = {}
    comm_results: dict[str, Any] = {}
    actual: dict[str, Any] = {}
    metadata: dict[str, Any] = {"compute": {}, "communication": {}}

    try:
        # Initialize and profile the five representative communications once.
        for kind in COMM_KINDS:
            launch, owned, meta = make_communication(
                torch, dist, rank, world_size, kind, comm_specs[kind]
            )
            local, world = time_isolated(
                torch,
                dist,
                control,
                rank,
                launch,
                args.warmups,
                args.iterations,
            )
            profile_operation(
                torch, dist, control, rank, launch, args.trace_dir / f"comm_{kind}.json"
            )
            if rank == 0:
                comm_results[kind] = {
                    "world_completion_ms": world,
                    "summary_ms": common.summarize(world),
                }
                metadata["communication"][kind] = meta
            cleanup(torch, owned)

        pairs_by_compute: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in pairs:
            pairs_by_compute[pair["compute_class"]].append(pair)

        for compute_class in COMPUTE_CLASSES:
            launch_compute, compute_owned, compute_meta = make_compute(
                torch, compute_class
            )
            # Synchronize any setup work (compilation graph construction or
            # saved SDPA forward) before timing.
            torch.cuda.synchronize()
            local, world = time_isolated(
                torch,
                dist,
                control,
                rank,
                launch_compute,
                args.warmups,
                args.iterations,
            )
            profile_operation(
                torch,
                dist,
                control,
                rank,
                launch_compute,
                args.trace_dir / f"compute_{compute_class}.json",
            )
            if rank == 0:
                compute_results[compute_class] = {
                    "world_completion_ms": world,
                    "summary_ms": common.summarize(world),
                }
                metadata["compute"][compute_class] = compute_meta

            for pair in pairs_by_compute[compute_class]:
                kind = pair["collective_family"]
                launch_comm, comm_owned, _ = make_communication(
                    torch, dist, rank, world_size, kind, comm_specs[kind]
                )
                for order in ("compute_first", "comm_first"):
                    measured = time_overlap(
                        torch,
                        dist,
                        control,
                        rank,
                        launch_compute,
                        launch_comm,
                        order,
                        args.warmups,
                        args.iterations,
                    )
                    if rank == 0:
                        actual.setdefault(pair["benchmark_case_id"], {})[order] = measured
                cleanup(torch, comm_owned)
            cleanup(torch, compute_owned)

        properties = torch.cuda.get_device_properties(0)
        if rank == 0:
            common.atomic_write_json(
                args.phase_output,
                {
                    "device": {
                        "name": properties.name,
                        "sm_count": properties.multi_processor_count,
                        "compute_capability": list(torch.cuda.get_device_capability(0)),
                    },
                    "software": {
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "nccl": torch.cuda.nccl.version(),
                    },
                    "compute": compute_results,
                    "communication": comm_results,
                    "actual": actual,
                    "metadata": metadata,
                },
            )
    finally:
        dist.destroy_process_group()


def run_restricted_worker(args: argparse.Namespace, torch: Any) -> None:
    import torch.distributed as dist

    if not args.comm_demand or args.comm_demand <= 0:
        raise ValueError("restricted worker needs a positive --comm-demand")
    # The smaller communication partition is requested directly; Hopper may
    # round it to the Green Context allocation granularity.
    green = common.GreenContextPair(args.comm_demand, "comm", device=0)
    green.set_compute()
    control, rank, _ = common.initialize_distributed(
        torch, need_nccl=False, set_cuda_device=False
    )
    green.set_compute()
    results: dict[str, Any] = {}
    try:
        for compute_class in COMPUTE_CLASSES:
            green.set_compute()
            launch, owned, _ = make_compute(torch, compute_class)
            torch.cuda.synchronize()
            local, world = time_isolated(
                torch,
                dist,
                control,
                rank,
                launch,
                args.warmups,
                args.iterations,
                before_launch=green.set_compute,
            )
            if rank == 0:
                results[compute_class] = {
                    "world_completion_ms": world,
                    "summary_ms": common.summarize(world),
                }
            cleanup(torch, owned)
        if rank == 0:
            common.atomic_write_json(
                args.phase_output,
                {
                    "requested_comm_sm_demand": args.comm_demand,
                    "actual_compute_sms": green.compute_sms,
                    "actual_comm_sms": green.comm_sms,
                    "device_sms": green.total_sms,
                    "compute": results,
                },
            )
    finally:
        dist.destroy_process_group()


def worker_main(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    common.set_worker_device_environment(local_rank, None)
    import torch

    if args.worker_mode == "baseline":
        run_baseline_worker(args, torch)
    elif args.worker_mode == "restricted":
        run_restricted_worker(args, torch)
    else:
        raise ValueError(args.worker_mode)


def run_phase(
    args: argparse.Namespace,
    mode: str,
    output: Path,
    trace_dir: Path,
    environment: dict[str, str],
    *,
    comm_demand: int | None = None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.world_size}",
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--worker-mode",
        mode,
        "--analysis",
        str(args.analysis.resolve()),
        "--phase-output",
        str(output),
        "--trace-dir",
        str(trace_dir),
        "--world-size",
        str(args.world_size),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
    ]
    if comm_demand is not None:
        command.extend(["--comm-demand", str(comm_demand)])
    completed = subprocess.run(command, env=environment, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"{mode} phase failed ({completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout[-12000:]}\n"
            f"--- stderr ---\n{completed.stderr[-24000:]}"
        )
    if not output.is_file():
        raise RuntimeError(f"phase {mode} did not create {output}")


def parse_profiles(
    trace_dir: Path, total_sms: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    compute = {
        name: common.parse_profile(
            trace_dir / f"compute_{name}.json", total_sms, "compute"
        )
        for name in COMPUTE_CLASSES
    }
    comm = {
        name: common.parse_profile(trace_dir / f"comm_{name}.json", total_sms, "comm")
        for name in COMM_KINDS
    }
    return compute, comm


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Trace-derived launch-order interference predictor",
        "",
        f"- Source: `{result['source_trace_dir']}`",
        f"- Selection: both kernels >1 ms in trace and overlap fraction >75%",
        f"- Reduction: {result['case_count']} cases, "
        f"{len(result['compute_profiles'])} compute classes, "
        f"{len(result['communication_profiles'])} communication kinds",
        f"- Replay: {result['world_size']} × {result['device']['name']}",
        "",
        "| Compute class | Communication | Compute isolated | Comm isolated | "
        "Compute-first predicted / actual / error | Comm-first predicted / actual / error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        cf = case["orders"]["compute_first"]
        nf = case["orders"]["comm_first"]
        lines.append(
            f"| {case['compute_class']} | {case['collective_family']} | "
            f"{case['compute_isolated_ms']:.3f} ms | {case['comm_isolated_ms']:.3f} ms | "
            f"{cf['predicted_ms']:.3f} / {cf['actual_ms']:.3f} / {cf['error_percent']:+.1f}% | "
            f"{nf['predicted_ms']:.3f} / {nf['actual_ms']:.3f} / {nf['error_percent']:+.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Aggregate fidelity",
            "",
            "| Launch order | Median absolute error | Mean absolute percentage error | Max absolute percentage error |",
            "|---|---:|---:|---:|",
        ]
    )
    for order, stats in result["fidelity"].items():
        lines.append(
            f"| {order} | {stats['median_absolute_error_ms']:.3f} ms | "
            f"{stats['mape_percent']:.1f}% | {stats['max_ape_percent']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The source collectives used group size 16 across nodes. The local replay "
            "uses eight GPUs: P2P/all-to-allv preserve representative per-rank payloads; "
            "all-gather/reduce-scatter preserve the recorded per-rank input and use the "
            "physical eight-rank output ratio. Both trace and physical metadata are in JSON.",
            "",
        ]
    )
    return "\n".join(lines)


def fidelity(cases: list[dict[str, Any]], order: str) -> dict[str, float]:
    absolute = [
        abs(case["orders"][order]["actual_ms"] - case["orders"][order]["predicted_ms"])
        for case in cases
    ]
    ape = [abs(case["orders"][order]["error_percent"]) for case in cases]
    return {
        "median_absolute_error_ms": float(common.statistics.median(absolute)),
        "mean_absolute_error_ms": float(common.statistics.mean(absolute)),
        "mape_percent": float(common.statistics.mean(ape)),
        "median_ape_percent": float(common.statistics.median(ape)),
        "max_ape_percent": max(ape),
    }


def controller_main(args: argparse.Namespace) -> None:
    if args.world_size != 8:
        raise ValueError("this trace replay is defined for the eight local GPUs")
    if args.warmups < 0 or args.iterations <= 0:
        raise ValueError("warmups must be nonnegative and iterations positive")
    analysis = load_analysis(args.analysis)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="trace-launch-order-", dir=output.parent))
    trace_dir = run_dir / "profiles"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.world_size))
    environment.setdefault("NCCL_CUMEM_ENABLE", "0")
    environment.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")

    try:
        print("[1/3] Isolated profiles and unrestricted overlap cases", flush=True)
        baseline_path = run_dir / "baseline.json"
        run_phase(args, "baseline", baseline_path, trace_dir, environment)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        total_sms = int(baseline["device"]["sm_count"])
        compute_profiles, comm_profiles = parse_profiles(trace_dir, total_sms)
        demands = sorted(
            {int(profile["peak_inferred_sm_demand"]) for profile in comm_profiles.values()}
        )

        print(f"[2/3] Restricted compute profiles for comm demands {demands}", flush=True)
        restricted: dict[int, dict[str, Any]] = {}
        for demand in demands:
            path = run_dir / f"restricted_comm_demand_{demand}.json"
            run_phase(
                args,
                "restricted",
                path,
                trace_dir,
                environment,
                comm_demand=demand,
            )
            restricted[demand] = json.loads(path.read_text(encoding="utf-8"))

        result_cases = []
        exact_by_id = {case["case_id"]: case for case in analysis["cases"]}
        for pair in benchmark_pairs(analysis):
            compute_class = pair["compute_class"]
            kind = pair["collective_family"]
            compute_iso = float(
                baseline["compute"][compute_class]["summary_ms"]["median"]
            )
            comm_iso = float(
                baseline["communication"][kind]["summary_ms"]["median"]
            )
            compute_demand = int(
                compute_profiles[compute_class]["peak_inferred_sm_demand"]
            )
            comm_demand = int(comm_profiles[kind]["peak_inferred_sm_demand"])
            if compute_demand >= total_sms:
                compute_first_prediction = compute_iso + comm_iso
                compute_first_source = "serialized isolated runtimes"
            else:
                raise RuntimeError(
                    f"{compute_class} used {compute_demand}/{total_sms} SMs; "
                    "restricted-first communication path is not implemented"
                )
            restricted_record = restricted[comm_demand]
            restricted_compute = float(
                restricted_record["compute"][compute_class]["summary_ms"]["median"]
            )
            comm_first_prediction = max(comm_iso, restricted_compute)
            actual_pair = baseline["actual"][pair["benchmark_case_id"]]
            representative = exact_by_id[pair["representative_case_id"]]
            orders = {}
            for order, prediction, source in (
                ("compute_first", compute_first_prediction, compute_first_source),
                (
                    "comm_first",
                    comm_first_prediction,
                    "max(comm isolated, compute restricted isolated)",
                ),
            ):
                actual_ms = float(actual_pair[order]["summary_ms"]["median"])
                orders[order] = {
                    "predicted_ms": prediction,
                    "actual_ms": actual_ms,
                    "actual_p95_ms": float(actual_pair[order]["summary_ms"]["p95"]),
                    "error_ms": actual_ms - prediction,
                    "error_percent": 100 * (actual_ms / prediction - 1),
                    "source": source,
                }
            result_cases.append(
                {
                    "benchmark_case_id": pair["benchmark_case_id"],
                    "compute_class": compute_class,
                    "collective_family": kind,
                    "trace_observation_count": pair["observation_count"],
                    "source_compute_families": pair["source_compute_families"],
                    "representative_trace_case": representative,
                    "compute_isolated_ms": compute_iso,
                    "comm_isolated_ms": comm_iso,
                    "compute_sm_demand": compute_demand,
                    "comm_sm_demand": comm_demand,
                    "desired_compute_sms_when_comm_first": total_sms - comm_demand,
                    "actual_green_compute_sms": restricted_record["actual_compute_sms"],
                    "actual_green_comm_sms": restricted_record["actual_comm_sms"],
                    "restricted_compute_ms": restricted_compute,
                    "orders": orders,
                }
            )

        result = {
            "schema_version": 1,
            "source_trace_dir": str(args.analysis.parent),
            "selection": {
                "overlap_fraction_strictly_greater_than": 0.75,
                "both_kernel_durations_us_strictly_greater_than": 1000,
                "raw_exact_case_count": analysis["unique_case_count"],
                "benchmark_case_count": len(result_cases),
            },
            "case_count": len(result_cases),
            "world_size": args.world_size,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "model": MODEL,
            "device": baseline["device"],
            "software": baseline["software"],
            "compute_profiles": compute_profiles,
            "communication_profiles": comm_profiles,
            "workload_metadata": baseline["metadata"],
            "restricted_profiles": restricted,
            "cases": result_cases,
        }
        result["fidelity"] = {
            order: fidelity(result_cases, order)
            for order in ("compute_first", "comm_first")
        }
        common.atomic_write_json(output, result)
        output.with_suffix(".md").write_text(render_report(result), encoding="utf-8")
        print(f"[3/3] Wrote {output} and {output.with_suffix('.md')}", flush=True)
    finally:
        if args.keep_artifacts:
            retained = output.parent / "artifacts"
            if retained.exists():
                raise RuntimeError(f"refusing to replace {retained}")
            shutil.move(str(run_dir), retained)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


def main() -> None:
    ensure_cuda_compat_first()
    args = parse_args()
    if args.internal_worker:
        worker_main(args)
    else:
        controller_main(args)


if __name__ == "__main__":
    main()
