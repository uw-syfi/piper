#!/usr/bin/env python3
"""Predict and measure grouped-GEMM x all-reduce interference on one GPU node.

This is a standalone microbenchmark; it does not import or invoke Piper.  The
default workload uses Qwen3.5-397B-A17B shapes at EP=8:

* compute: one BF16 grouped GEMM for the fused gate/up expert projection,
  with 64 local experts and a balanced 5,120 routed tokens per expert;
* communication: one BF16 all-reduce of the corresponding 1-GiB local
  fused-gate/up gradient shard over eight GPUs.

The benchmark implements the following model:

1. Measure both operations in isolation and profile their CUDA grid sizes.
2. Infer SM demand as ``min(device SMs, grid CTAs)`` for every kernel and use
   the maximum demand in each operation.
3. If the demands fit, predict perfect overlap.  Otherwise, divide the
   available SMs in proportion to the two isolated demands.
4. Re-measure each operation under its calculated SM allocation and predict
   the managed overlap as the maximum of those two restricted runtimes.
5. Measure real overlap both without SM management and with the calculated
   allocation.

The restricted and managed experiments use two disjoint CUDA Green Contexts
created from a single SM-resource split on every GPU.  Hopper rounds the
smaller partition to its supported allocation granularity; the actual SM
counts returned by the driver are used for the restricted measurements.

The public invocation is self-orchestrating:

    python scripts/benchmark_grouped_gemm_allreduce_sm_partition.py \
        --output out/sm_partition/results.json

The output JSON and adjacent Markdown report contain the raw samples, derived
allocation, profile kernel grids, prediction, and both overlap measurements.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable


QWEN_DIM = 4096
QWEN_EXPERT_WIDTH = 1024
QWEN_NUM_EXPERTS = 512
QWEN_TOP_K = 10
DEFAULT_LOCAL_TOKENS = 32768
DEFAULT_LOGICAL_EP = 8


class _CuDevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint)]


class _CuDevResourcePayload(ctypes.Union):
    _fields_ = [
        ("sm", _CuDevSmResource),
        ("oversize", ctypes.c_ubyte * 48),
    ]


class _CuDevResource(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("internal_padding", ctypes.c_ubyte * 92),
        ("payload", _CuDevResourcePayload),
    ]


class GreenContextPair:
    """Two complementary Green Contexts from one disjoint SM-resource split."""

    SM_RESOURCE = 1
    DEFAULT_STREAM = 1

    def __init__(
        self, partition_target_sms: int, partition_side: str, device: int = 0
    ) -> None:
        if partition_side not in {"compute", "comm"}:
            raise ValueError(f"invalid Green Context partition side: {partition_side}")
        self.partition_side = partition_side
        self._driver = ctypes.CDLL("libcuda.so.1")
        self._configure_driver_api()
        self._check(self._driver.cuInit(0), "cuInit")
        cu_device = ctypes.c_int()
        self._check(
            self._driver.cuDeviceGet(ctypes.byref(cu_device), device), "cuDeviceGet"
        )
        complete = _CuDevResource()
        partition = _CuDevResource()
        remaining = _CuDevResource()
        self._check(
            self._driver.cuDeviceGetDevResource(
                cu_device, ctypes.byref(complete), self.SM_RESOURCE
            ),
            "cuDeviceGetDevResource",
        )
        group_count = ctypes.c_uint(1)
        self._check(
            self._driver.cuDevSmResourceSplitByCount(
                ctypes.byref(partition),
                ctypes.byref(group_count),
                ctypes.byref(complete),
                ctypes.byref(remaining),
                0,
                partition_target_sms,
            ),
            "cuDevSmResourceSplitByCount",
        )
        if group_count.value != 1:
            raise RuntimeError(f"Green Context split returned {group_count.value} groups")
        self.total_sms = int(complete.payload.sm.smCount)
        partition_sms = int(partition.payload.sm.smCount)
        remaining_sms = int(remaining.payload.sm.smCount)
        if partition_sms <= 0 or remaining_sms <= 0:
            raise RuntimeError(
                f"invalid Green Context split: {partition_sms}+{remaining_sms}"
            )

        partition_desc = ctypes.c_void_p()
        remaining_desc = ctypes.c_void_p()
        self._check(
            self._driver.cuDevResourceGenerateDesc(
                ctypes.byref(partition_desc), ctypes.byref(partition), 1
            ),
            "cuDevResourceGenerateDesc(partition)",
        )
        self._check(
            self._driver.cuDevResourceGenerateDesc(
                ctypes.byref(remaining_desc), ctypes.byref(remaining), 1
            ),
            "cuDevResourceGenerateDesc(remaining)",
        )
        partition_green = ctypes.c_void_p()
        remaining_green = ctypes.c_void_p()
        self._check(
            self._driver.cuGreenCtxCreate(
                ctypes.byref(partition_green),
                partition_desc,
                cu_device,
                self.DEFAULT_STREAM,
            ),
            "cuGreenCtxCreate(partition)",
        )
        self._check(
            self._driver.cuGreenCtxCreate(
                ctypes.byref(remaining_green),
                remaining_desc,
                cu_device,
                self.DEFAULT_STREAM,
            ),
            "cuGreenCtxCreate(remaining)",
        )
        partition_context = ctypes.c_void_p()
        remaining_context = ctypes.c_void_p()
        self._check(
            self._driver.cuCtxFromGreenCtx(
                ctypes.byref(partition_context), partition_green
            ),
            "cuCtxFromGreenCtx(partition)",
        )
        self._check(
            self._driver.cuCtxFromGreenCtx(
                ctypes.byref(remaining_context), remaining_green
            ),
            "cuCtxFromGreenCtx(remaining)",
        )
        if partition_side == "compute":
            self.compute_sms = partition_sms
            self.comm_sms = remaining_sms
            self._compute_green = partition_green
            self._comm_green = remaining_green
            self._compute_context = partition_context
            self._comm_context = remaining_context
        else:
            self.compute_sms = remaining_sms
            self.comm_sms = partition_sms
            self._compute_green = remaining_green
            self._comm_green = partition_green
            self._compute_context = remaining_context
            self._comm_context = partition_context
        self._closed = False

    def _configure_driver_api(self) -> None:
        resource_pointer = ctypes.POINTER(_CuDevResource)
        self._driver.cuInit.argtypes = [ctypes.c_uint]
        self._driver.cuInit.restype = ctypes.c_int
        self._driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        self._driver.cuDeviceGet.restype = ctypes.c_int
        self._driver.cuDeviceGetDevResource.argtypes = [
            ctypes.c_int,
            resource_pointer,
            ctypes.c_int,
        ]
        self._driver.cuDeviceGetDevResource.restype = ctypes.c_int
        self._driver.cuDevSmResourceSplitByCount.argtypes = [
            resource_pointer,
            ctypes.POINTER(ctypes.c_uint),
            resource_pointer,
            resource_pointer,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self._driver.cuDevSmResourceSplitByCount.restype = ctypes.c_int
        self._driver.cuDevResourceGenerateDesc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            resource_pointer,
            ctypes.c_uint,
        ]
        self._driver.cuDevResourceGenerateDesc.restype = ctypes.c_int
        self._driver.cuGreenCtxCreate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        self._driver.cuGreenCtxCreate.restype = ctypes.c_int
        self._driver.cuCtxFromGreenCtx.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
        ]
        self._driver.cuCtxFromGreenCtx.restype = ctypes.c_int
        self._driver.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self._driver.cuCtxSetCurrent.restype = ctypes.c_int
        self._driver.cuGreenCtxDestroy.argtypes = [ctypes.c_void_p]
        self._driver.cuGreenCtxDestroy.restype = ctypes.c_int
        self._driver.cuGetErrorName.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        self._driver.cuGetErrorString.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        name = ctypes.c_char_p()
        message = ctypes.c_char_p()
        self._driver.cuGetErrorName(result, ctypes.byref(name))
        self._driver.cuGetErrorString(result, ctypes.byref(message))
        raise RuntimeError(
            f"{operation} failed: "
            f"{name.value.decode() if name.value else result}: "
            f"{message.value.decode() if message.value else 'unknown error'}"
        )

    def set_compute(self) -> None:
        self._check(
            self._driver.cuCtxSetCurrent(self._compute_context),
            "cuCtxSetCurrent(compute)",
        )

    def set_comm(self) -> None:
        self._check(
            self._driver.cuCtxSetCurrent(self._comm_context),
            "cuCtxSetCurrent(comm)",
        )

    def close(self) -> None:
        if self._closed:
            return
        self._check(self._driver.cuCtxSetCurrent(None), "cuCtxSetCurrent(NULL)")
        self._check(
            self._driver.cuGreenCtxDestroy(self._compute_green),
            "cuGreenCtxDestroy(compute)",
        )
        self._check(
            self._driver.cuGreenCtxDestroy(self._comm_green),
            "cuGreenCtxDestroy(comm)",
        )
        self._closed = True


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "p95": percentile(values, 0.95),
        "max": max(values),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def workload(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_experts % args.logical_ep_size:
        raise ValueError("num-experts must be divisible by logical-ep-size")
    routed_per_expert = (
        args.logical_ep_size * args.local_tokens * args.top_k / args.num_experts
    )
    if not routed_per_expert.is_integer():
        raise ValueError("Qwen shape produces a non-integral tokens-per-expert")
    local_experts = args.num_experts // args.logical_ep_size
    tokens_per_expert = int(routed_per_expert)
    routed_rows = local_experts * tokens_per_expert
    gemm_n = 2 * args.expert_width
    allreduce_numel = (
        args.allreduce_numel
        if args.allreduce_numel is not None
        else local_experts * args.dim * gemm_n
    )
    element_size = 2
    memory = {
        "grouped_gemm_input_bytes": (routed_rows + 1) * args.dim * element_size,
        "grouped_gemm_weight_bytes": (
            local_experts * args.dim * gemm_n * element_size
        ),
        "grouped_gemm_output_bytes": (
            (routed_rows + 1) * gemm_n * element_size
        ),
        "allreduce_tensor_bytes": allreduce_numel * element_size,
    }
    return {
        "model": "Qwen3.5-397B-A17B",
        "source": (
            "Shapes match examples/models/qwen3.py and the Qwen-397B "
            "standalone collective/backward microbenchmarks."
        ),
        "dtype": "torch.bfloat16",
        "world_size": args.world_size,
        "local_tokens": args.local_tokens,
        "logical_ep_size": args.logical_ep_size,
        "dim": args.dim,
        "expert_width": args.expert_width,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "local_experts": local_experts,
        "tokens_per_expert": tokens_per_expert,
        "routed_rows": routed_rows,
        "grouped_gemm": {
            "groups": local_experts,
            "per_group_m": tokens_per_expert,
            "k": args.dim,
            "n": gemm_n,
            "flops": 2 * routed_rows * args.dim * gemm_n,
            "projection": "fused_gate_up",
        },
        "allreduce": {
            "numel": allreduce_numel,
            "bytes": allreduce_numel * element_size,
            "mib": allreduce_numel * element_size / 2**20,
            "parameter": "one local fused_gate_up expert-gradient shard",
        },
        "per_gpu_memory": memory
        | {"total_bytes": sum(memory.values()), "total_gib": sum(memory.values()) / 2**30},
    }


def add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--local-tokens", type=int, default=DEFAULT_LOCAL_TOKENS)
    parser.add_argument("--logical-ep-size", type=int, default=DEFAULT_LOGICAL_EP)
    parser.add_argument("--dim", type=int, default=QWEN_DIM)
    parser.add_argument("--expert-width", type=int, default=QWEN_EXPERT_WIDTH)
    parser.add_argument("--num-experts", type=int, default=QWEN_NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=QWEN_TOP_K)
    parser.add_argument(
        "--allreduce-numel",
        type=int,
        help="Override the derived local fused-gate/up gradient element count",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--gate-sleep-cycles",
        type=int,
        default=20_000_000,
        help="Device-side delay used to enqueue both unmanaged streams before release",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workload_arguments(parser)
    parser.add_argument(
        "--output", default="out/grouped_gemm_allreduce_sm_partition/results.json"
    )
    parser.add_argument("--keep-traces", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--managed-launch-order",
        choices=["compute_first", "comm_first"],
        default="compute_first",
        help="Host enqueue order for the Green Context managed overlap phase",
    )
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-compute-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode",
        choices=[
            "baseline",
            "restricted_compute",
            "restricted_comm",
            "managed",
            "unmanaged_ordered",
        ],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--phase-output", help=argparse.SUPPRESS)
    parser.add_argument("--trace-dir", help=argparse.SUPPRESS)
    parser.add_argument("--mps-percent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--compute-mps-percent", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--comm-mps-percent", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--green-partition-target-sms", type=int, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--green-partition-side",
        choices=["compute", "comm"],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "world-size": args.world_size,
        "local-tokens": args.local_tokens,
        "logical-ep-size": args.logical_ep_size,
        "dim": args.dim,
        "expert-width": args.expert_width,
        "num-experts": args.num_experts,
        "top-k": args.top_k,
        "iterations": args.iterations,
        "gate-sleep-cycles": args.gate_sleep_cycles,
    }
    if args.allreduce_numel is not None:
        positive["allreduce-numel"] = args.allreduce_numel
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.warmups < 0:
        raise ValueError("warmups may not be negative")
    if args.world_size > 8:
        raise ValueError("this local-node benchmark supports at most 8 GPUs")
    workload(args)


def set_worker_device_environment(local_rank: int, mps_percent: int | None) -> None:
    # This must run before importing torch.  Every torchrun rank and managed
    # compute child then sees its assigned physical GPU as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    if mps_percent is not None:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_percent)
        os.environ["CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING"] = "1"


def initialize_distributed(
    torch: Any, need_nccl: bool, *, set_cuda_device: bool = True
) -> tuple[Any, int, int]:
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if need_nccl:
        if set_cuda_device:
            torch.cuda.set_device(0)
        dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
        control_group = dist.new_group(backend="gloo")
    else:
        dist.init_process_group("gloo")
        control_group = dist.group.WORLD
        if set_cuda_device:
            torch.cuda.set_device(0)
    return control_group, rank, world_size


def green_pair_from_args(args: argparse.Namespace) -> GreenContextPair:
    if args.green_partition_target_sms is None or args.green_partition_side is None:
        raise ValueError("Green Context worker is missing its partition specification")
    return GreenContextPair(
        args.green_partition_target_sms, args.green_partition_side, device=0
    )


def make_grouped_gemm(torch: Any, shape: dict[str, Any]) -> tuple[Callable[[], Any], list[Any]]:
    groups = int(shape["local_experts"])
    rows = int(shape["routed_rows"])
    dim = int(shape["dim"])
    gemm_n = int(shape["grouped_gemm"]["n"])
    tokens_per_expert = int(shape["tokens_per_expert"])
    # _grouped_mm requires offs[-1] < input rows, hence one unused padding row.
    inputs = torch.empty((rows + 1, dim), dtype=torch.bfloat16, device="cuda")
    weights = torch.empty((groups, dim, gemm_n), dtype=torch.bfloat16, device="cuda")
    offsets = torch.arange(
        tokens_per_expert,
        rows + 1,
        tokens_per_expert,
        dtype=torch.int32,
        device="cuda",
    )
    if offsets.numel() != groups or int(offsets[-1].item()) != rows:
        raise RuntimeError("invalid grouped-GEMM offsets")

    def launch() -> Any:
        return torch._grouped_mm(inputs, weights, offs=offsets)

    return launch, [inputs, weights, offsets]


def gather_sample(control_group: Any, rank: int, sample: dict[str, Any]) -> list[dict[str, Any]] | None:
    import torch.distributed as dist

    gathered: list[Any] | None = [None] * dist.get_world_size(group=control_group) if rank == 0 else None
    dist.gather_object(sample, gathered, dst=0, group=control_group)
    return gathered


def time_isolated(
    torch: Any,
    control_group: Any,
    rank: int,
    launch: Callable[[], Any],
    warmups: int,
    iterations: int,
) -> tuple[list[float], list[float]]:
    import torch.distributed as dist

    for _ in range(warmups):
        dist.barrier(group=control_group)
        result = launch()
        torch.cuda.synchronize()
        del result
    local_ms: list[float] = []
    world_ms: list[float] = []
    for _ in range(iterations):
        dist.barrier(group=control_group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = launch()
        end.record()
        end.synchronize()
        duration = start.elapsed_time(end)
        local_ms.append(duration)
        gathered = gather_sample(control_group, rank, {"duration_ms": duration})
        if gathered is not None:
            world_ms.append(max(float(item["duration_ms"]) for item in gathered))
        del result
    return local_ms, world_ms


def profile_operation(
    torch: Any,
    control_group: Any,
    rank: int,
    launch: Callable[[], Any],
    trace_path: Path,
) -> None:
    import torch.distributed as dist

    dist.barrier(group=control_group)
    if rank == 0:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            result = launch()
            torch.cuda.synchronize()
            del result
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
    else:
        result = launch()
        torch.cuda.synchronize()
        del result
    dist.barrier(group=control_group)


def unmanaged_sample(
    torch: Any,
    grouped_gemm: Callable[[], Any],
    comm_tensor: Any,
    gate_sleep_cycles: int,
    streams: tuple[Any, Any, Any],
) -> dict[str, float]:
    import torch.distributed as dist

    gate_stream, compute_stream, comm_issue_stream = streams
    gate = torch.cuda.Event(enable_timing=True)
    compute_start = torch.cuda.Event(enable_timing=True)
    compute_end = torch.cuda.Event(enable_timing=True)
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)
    overall_end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(gate_stream):
        torch.cuda._sleep(gate_sleep_cycles)
        gate.record()
    with torch.cuda.stream(compute_stream):
        compute_stream.wait_event(gate)
        compute_start.record()
        result = grouped_gemm()
        compute_end.record()
    with torch.cuda.stream(comm_issue_stream):
        comm_issue_stream.wait_event(gate)
        comm_start.record()
        work = dist.all_reduce(comm_tensor, async_op=True)
        work.wait()
        comm_end.record()
    join_stream = torch.cuda.current_stream()
    join_stream.wait_event(compute_end)
    join_stream.wait_event(comm_end)
    overall_end.record(join_stream)
    overall_end.synchronize()
    sample = {
        "overall_ms": gate.elapsed_time(overall_end),
        "compute_ms": compute_start.elapsed_time(compute_end),
        "comm_ms": comm_start.elapsed_time(comm_end),
    }
    del result
    return sample


def run_baseline_worker(args: argparse.Namespace, torch: Any) -> None:
    import torch.distributed as dist

    control, rank, _ = initialize_distributed(torch, need_nccl=True)
    shape = workload(args)
    grouped_gemm, owned = make_grouped_gemm(torch, shape)
    comm_tensor = torch.zeros(
        int(shape["allreduce"]["numel"]), dtype=torch.bfloat16, device="cuda"
    )

    def allreduce() -> None:
        work = dist.all_reduce(comm_tensor, async_op=True)
        work.wait()
        return None

    try:
        compute_local, compute_world = time_isolated(
            torch, control, rank, grouped_gemm, args.warmups, args.iterations
        )
        comm_local, comm_world = time_isolated(
            torch, control, rank, allreduce, args.warmups, args.iterations
        )
        trace_dir = Path(args.trace_dir)
        profile_operation(torch, control, rank, grouped_gemm, trace_dir / "compute.json")
        profile_operation(torch, control, rank, allreduce, trace_dir / "comm.json")

        streams = (torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream())
        for _ in range(args.warmups):
            dist.barrier(group=control)
            unmanaged_sample(
                torch, grouped_gemm, comm_tensor, args.gate_sleep_cycles, streams
            )
        unmanaged_local: list[dict[str, float]] = []
        unmanaged_world: list[float] = []
        for _ in range(args.iterations):
            dist.barrier(group=control)
            sample = unmanaged_sample(
                torch, grouped_gemm, comm_tensor, args.gate_sleep_cycles, streams
            )
            unmanaged_local.append(sample)
            gathered = gather_sample(control, rank, sample)
            if gathered is not None:
                unmanaged_world.append(
                    max(float(item["overall_ms"]) for item in gathered)
                )

        properties = torch.cuda.get_device_properties(0)
        rank_result = {
            "rank": rank,
            "device_name": properties.name,
            "visible_sm_count": properties.multi_processor_count,
            "compute_isolated_ms": compute_local,
            "comm_isolated_ms": comm_local,
            "unmanaged_samples": unmanaged_local,
        }
        ranks = gather_sample(control, rank, rank_result)
        if rank == 0:
            result = {
                "mode": "baseline",
                "workload": shape,
                "software": {
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "nccl": torch.cuda.nccl.version(),
                },
                "device": {
                    "name": properties.name,
                    "sm_count": properties.multi_processor_count,
                    "compute_capability": list(torch.cuda.get_device_capability(0)),
                    "total_memory": properties.total_memory,
                },
                "isolated_compute": {
                    "world_completion_ms": compute_world,
                    "summary_ms": summarize(compute_world),
                },
                "isolated_comm": {
                    "world_completion_ms": comm_world,
                    "summary_ms": summarize(comm_world),
                },
                "unmanaged_overlap": {
                    "world_completion_ms": unmanaged_world,
                    "summary_ms": summarize(unmanaged_world),
                },
                "rank_results": ranks,
            }
            atomic_write_json(Path(args.phase_output), result)
    finally:
        del owned, comm_tensor
        dist.destroy_process_group()


def run_restricted_worker(args: argparse.Namespace, torch: Any, compute: bool) -> None:
    import torch.distributed as dist

    green: GreenContextPair | None = None
    if args.green_partition_target_sms is not None:
        green = green_pair_from_args(args)
        (green.set_compute if compute else green.set_comm)()
        control, rank, _ = initialize_distributed(
            torch, need_nccl=not compute, set_cuda_device=False
        )
        # Process-group initialization may query the runtime; restore the
        # intended Green Context before creating workload allocations.
        (green.set_compute if compute else green.set_comm)()
    else:
        control, rank, _ = initialize_distributed(torch, need_nccl=not compute)
    shape = workload(args)
    owned: list[Any] = []
    if compute:
        launch, owned = make_grouped_gemm(torch, shape)
    else:
        comm_tensor = torch.zeros(
            int(shape["allreduce"]["numel"]), dtype=torch.bfloat16, device="cuda"
        )
        owned.append(comm_tensor)

        def launch() -> None:
            work = dist.all_reduce(comm_tensor, async_op=True)
            work.wait()
            return None

    try:
        local_ms, world_ms = time_isolated(
            torch, control, rank, launch, args.warmups, args.iterations
        )
        profile_operation(
            torch,
            control,
            rank,
            launch,
            Path(args.trace_dir)
            / ("restricted_compute.json" if compute else "restricted_comm.json"),
        )
        properties = torch.cuda.get_device_properties(0)
        allocated_sms = (
            (green.compute_sms if compute else green.comm_sms)
            if green is not None
            else properties.multi_processor_count
        )
        rank_result = {
            "rank": rank,
            "allocated_sm_count": allocated_sms,
            "device_sm_count": properties.multi_processor_count,
            "samples_ms": local_ms,
        }
        ranks = gather_sample(control, rank, rank_result)
        if rank == 0:
            result = {
                "mode": "restricted_compute" if compute else "restricted_comm",
                "management": (
                    "CUDA Green Context" if green is not None else "full device"
                ),
                "green_partition_side": (
                    green.partition_side if green is not None else None
                ),
                "green_partition_target_sms": args.green_partition_target_sms,
                "allocated_sm_count": allocated_sms,
                "visible_sm_count": allocated_sms,
                "device_sm_count": properties.multi_processor_count,
                "world_completion_ms": world_ms,
                "summary_ms": summarize(world_ms),
                "rank_results": ranks,
            }
            atomic_write_json(Path(args.phase_output), result)
    finally:
        del owned
        dist.destroy_process_group()


def child_send(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def run_compute_child(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["BENCHMARK_PHYSICAL_GPU"])
    set_worker_device_environment(local_rank, args.compute_mps_percent)
    import torch

    torch.cuda.set_device(0)
    shape = workload(args)
    grouped_gemm, owned = make_grouped_gemm(torch, shape)
    # Initialize libraries and the caching allocator before reporting ready.
    result = grouped_gemm()
    torch.cuda.synchronize()
    del result
    properties = torch.cuda.get_device_properties(0)
    child_send(
        {
            "status": "ready",
            "physical_gpu": local_rank,
            "visible_sm_count": properties.multi_processor_count,
        }
    )
    try:
        for line in sys.stdin:
            command = json.loads(line)
            if command["command"] == "quit":
                child_send({"status": "stopped"})
                return
            if command["command"] != "launch":
                raise ValueError(f"unknown child command: {command}")
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start_ns = time.perf_counter_ns()
            start.record()
            result = grouped_gemm()
            end.record()
            end.synchronize()
            host_end_ns = time.perf_counter_ns()
            child_send(
                {
                    "status": "complete",
                    "cuda_ms": start.elapsed_time(end),
                    "host_start_ns": host_start_ns,
                    "host_end_ns": host_end_ns,
                }
            )
            del result
    finally:
        del owned


def child_command(child: subprocess.Popen[str], command: str) -> dict[str, Any]:
    assert child.stdin is not None and child.stdout is not None
    child.stdin.write(json.dumps({"command": command}) + "\n")
    child.stdin.flush()
    line = child.stdout.readline()
    if not line:
        raise RuntimeError(f"compute child exited unexpectedly with {child.returncode}")
    return json.loads(line)


def run_managed_worker(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    set_worker_device_environment(local_rank, None)
    import torch
    import torch.distributed as dist

    green = green_pair_from_args(args)
    green.set_comm()
    control, rank, _ = initialize_distributed(
        torch, need_nccl=True, set_cuda_device=False
    )
    shape = workload(args)
    green.set_comm()
    comm_tensor = torch.zeros(
        int(shape["allreduce"]["numel"]), dtype=torch.bfloat16, device="cuda"
    )
    green.set_compute()
    grouped_gemm, compute_owned = make_grouped_gemm(torch, shape)

    # Initialize the grouped-GEMM libraries and NCCL communicator in their
    # respective contexts before collecting concurrent samples.
    result = grouped_gemm()
    torch.cuda.synchronize()
    del result
    green.set_comm()
    work = dist.all_reduce(comm_tensor, async_op=True)
    work.wait()
    torch.cuda.synchronize()

    def one_sample() -> dict[str, Any]:
        def enqueue_compute() -> tuple[Any, Any, Any, int]:
            green.set_compute()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start_ns = time.perf_counter_ns()
            start.record()
            result = grouped_gemm()
            end.record()
            return start, end, result, host_start_ns

        def enqueue_comm() -> tuple[Any, Any, int]:
            green.set_comm()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start_ns = time.perf_counter_ns()
            start.record()
            work = dist.all_reduce(comm_tensor, async_op=True)
            work.wait()
            end.record()
            return start, end, host_start_ns

        if args.managed_launch_order == "compute_first":
            compute_start, compute_end, compute_result, compute_host_start_ns = (
                enqueue_compute()
            )
            comm_start, comm_end, comm_host_start_ns = enqueue_comm()
        else:
            comm_start, comm_end, comm_host_start_ns = enqueue_comm()
            compute_start, compute_end, compute_result, compute_host_start_ns = (
                enqueue_compute()
            )
        green.set_comm()
        comm_end.synchronize()
        comm_cuda_ms = comm_start.elapsed_time(comm_end)
        green.set_compute()
        compute_end.synchronize()
        compute_cuda_ms = compute_start.elapsed_time(compute_end)
        host_end_ns = time.perf_counter_ns()
        del compute_result
        starts = [comm_host_start_ns, compute_host_start_ns]
        return {
            "order": args.managed_launch_order,
            "overall_host_ms": (host_end_ns - min(starts)) / 1e6,
            "launch_skew_ms": abs(starts[0] - starts[1]) / 1e6,
            "compute_minus_comm_start_ms": (starts[1] - starts[0]) / 1e6,
            "comm_cuda_ms": comm_cuda_ms,
            "compute_cuda_ms": compute_cuda_ms,
            "earliest_host_start_ns": min(starts),
            "latest_host_end_ns": host_end_ns,
        }

    def collect_samples() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        for _ in range(args.warmups):
            dist.barrier(group=control)
            one_sample()
        local: list[dict[str, Any]] = []
        world: list[dict[str, Any]] = []
        for _ in range(args.iterations):
            dist.barrier(group=control)
            sample = one_sample()
            local.append(sample)
            gathered = gather_sample(control, rank, sample)
            if gathered is not None:
                earliest = min(int(item["earliest_host_start_ns"]) for item in gathered)
                latest = max(int(item["latest_host_end_ns"]) for item in gathered)
                world.append(
                    {
                        "completion_ms": (latest - earliest) / 1e6,
                        "max_launch_skew_ms": max(
                            float(item["launch_skew_ms"]) for item in gathered
                        ),
                        "max_compute_cuda_ms": max(
                            float(item["compute_cuda_ms"]) for item in gathered
                        ),
                        "max_comm_cuda_ms": max(
                            float(item["comm_cuda_ms"]) for item in gathered
                        ),
                        "rank_compute_minus_comm_start_ms": [
                            float(item["compute_minus_comm_start_ms"])
                            for item in gathered
                        ],
                    }
                )
        return local, world

    def summarize_mode(world: list[dict[str, Any]]) -> dict[str, Any]:
        completions = [float(sample["completion_ms"]) for sample in world]
        compute = [float(sample["max_compute_cuda_ms"]) for sample in world]
        comm = [float(sample["max_comm_cuda_ms"]) for sample in world]
        offsets = [
            float(offset)
            for sample in world
            for offset in sample["rank_compute_minus_comm_start_ms"]
        ]
        return {
            "world_samples": world,
            "world_completion_ms": completions,
            "summary_ms": summarize(completions),
            "compute_cuda_summary_ms": summarize(compute),
            "comm_cuda_summary_ms": summarize(comm),
            "compute_minus_comm_start_summary_ms": summarize(offsets),
        }

    try:
        local_samples, world_samples = collect_samples()
        green.set_compute()
        properties = torch.cuda.get_device_properties(0)
        rank_result = {
            "rank": rank,
            "compute_allocated_sm_count": green.compute_sms,
            "comm_allocated_sm_count": green.comm_sms,
            "device_sm_count": properties.multi_processor_count,
            "samples": local_samples,
        }
        ranks = gather_sample(control, rank, rank_result)
        if rank == 0:
            mode_summary = summarize_mode(world_samples)
            result = {
                "mode": "managed",
                "management": "disjoint CUDA Green Contexts",
                "green_partition_side": green.partition_side,
                "green_partition_target_sms": args.green_partition_target_sms,
                "launch_order": args.managed_launch_order,
                "compute_visible_sm_count": green.compute_sms,
                "comm_visible_sm_count": green.comm_sms,
                "device_sm_count": properties.multi_processor_count,
                **mode_summary,
                "rank_results": ranks,
                "timing_note": (
                    "The two disjoint Green Contexts cannot share CUDA timing events. "
                    "Completion uses the node-wide monotonic host clock around both "
                    "synchronized operations; launch skew and per-context CUDA event "
                    "durations are retained for every sample."
                ),
            }
            atomic_write_json(Path(args.phase_output), result)
    finally:
        green.set_compute()
        torch.cuda.synchronize()
        del compute_owned
        green.set_comm()
        torch.cuda.synchronize()
        del comm_tensor
        dist.destroy_process_group()


def run_unmanaged_ordered_worker(args: argparse.Namespace, torch: Any) -> None:
    """Overlap both operations in one unrestricted primary CUDA context."""
    import torch.distributed as dist

    control, rank, _ = initialize_distributed(torch, need_nccl=True)
    shape = workload(args)
    grouped_gemm, compute_owned = make_grouped_gemm(torch, shape)
    comm_tensor = torch.zeros(
        int(shape["allreduce"]["numel"]), dtype=torch.bfloat16, device="cuda"
    )
    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    # Initialize both library paths before timing overlap.
    result = grouped_gemm()
    torch.cuda.synchronize()
    del result
    work = dist.all_reduce(comm_tensor, async_op=True)
    work.wait()
    torch.cuda.synchronize()

    def one_sample() -> dict[str, Any]:
        def enqueue_compute() -> tuple[Any, Any, Any, int]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start_ns = time.perf_counter_ns()
            with torch.cuda.stream(compute_stream):
                start.record()
                result = grouped_gemm()
                end.record()
            return start, end, result, host_start_ns

        def enqueue_comm() -> tuple[Any, Any, int]:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start_ns = time.perf_counter_ns()
            with torch.cuda.stream(comm_stream):
                start.record()
                work = dist.all_reduce(comm_tensor, async_op=True)
                work.wait()
                end.record()
            return start, end, host_start_ns

        if args.managed_launch_order == "compute_first":
            compute_start, compute_end, compute_result, compute_host_start_ns = (
                enqueue_compute()
            )
            comm_start, comm_end, comm_host_start_ns = enqueue_comm()
        else:
            comm_start, comm_end, comm_host_start_ns = enqueue_comm()
            compute_start, compute_end, compute_result, compute_host_start_ns = (
                enqueue_compute()
            )
        compute_end.synchronize()
        comm_end.synchronize()
        host_end_ns = time.perf_counter_ns()
        sample = {
            "order": args.managed_launch_order,
            "overall_host_ms": (
                host_end_ns - min(compute_host_start_ns, comm_host_start_ns)
            )
            / 1e6,
            "launch_skew_ms": abs(compute_host_start_ns - comm_host_start_ns)
            / 1e6,
            "compute_minus_comm_start_ms": (
                compute_host_start_ns - comm_host_start_ns
            )
            / 1e6,
            "compute_cuda_ms": compute_start.elapsed_time(compute_end),
            "comm_cuda_ms": comm_start.elapsed_time(comm_end),
            "earliest_host_start_ns": min(
                compute_host_start_ns, comm_host_start_ns
            ),
            "latest_host_end_ns": host_end_ns,
        }
        del compute_result
        return sample

    def summarize_world(world: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "world_samples": world,
            "world_completion_ms": [
                float(sample["completion_ms"]) for sample in world
            ],
            "summary_ms": summarize(
                [float(sample["completion_ms"]) for sample in world]
            ),
            "compute_cuda_summary_ms": summarize(
                [float(sample["max_compute_cuda_ms"]) for sample in world]
            ),
            "comm_cuda_summary_ms": summarize(
                [float(sample["max_comm_cuda_ms"]) for sample in world]
            ),
            "compute_minus_comm_start_summary_ms": summarize(
                [
                    float(offset)
                    for sample in world
                    for offset in sample["rank_compute_minus_comm_start_ms"]
                ]
            ),
        }

    try:
        for _ in range(args.warmups):
            dist.barrier(group=control)
            one_sample()
        local_samples: list[dict[str, Any]] = []
        world_samples: list[dict[str, Any]] = []
        for _ in range(args.iterations):
            dist.barrier(group=control)
            sample = one_sample()
            local_samples.append(sample)
            gathered = gather_sample(control, rank, sample)
            if gathered is not None:
                earliest = min(
                    int(item["earliest_host_start_ns"]) for item in gathered
                )
                latest = max(int(item["latest_host_end_ns"]) for item in gathered)
                world_samples.append(
                    {
                        "completion_ms": (latest - earliest) / 1e6,
                        "max_launch_skew_ms": max(
                            float(item["launch_skew_ms"]) for item in gathered
                        ),
                        "max_compute_cuda_ms": max(
                            float(item["compute_cuda_ms"]) for item in gathered
                        ),
                        "max_comm_cuda_ms": max(
                            float(item["comm_cuda_ms"]) for item in gathered
                        ),
                        "rank_compute_minus_comm_start_ms": [
                            float(item["compute_minus_comm_start_ms"])
                            for item in gathered
                        ],
                    }
                )
        properties = torch.cuda.get_device_properties(0)
        ranks = gather_sample(
            control,
            rank,
            {"rank": rank, "samples": local_samples},
        )
        if rank == 0:
            result = {
                "mode": "unmanaged_ordered",
                "management": "none",
                "launch_order": args.managed_launch_order,
                "compute_visible_sm_count": properties.multi_processor_count,
                "comm_visible_sm_count": properties.multi_processor_count,
                **summarize_world(world_samples),
                "rank_results": ranks,
                "timing_note": (
                    "Compute and NCCL use separate streams in one unrestricted primary "
                    "CUDA context. Overall completion uses the node-wide host clock."
                ),
            }
            atomic_write_json(Path(args.phase_output), result)
    finally:
        del compute_owned, comm_tensor
        dist.destroy_process_group()


def run_internal(args: argparse.Namespace) -> None:
    if args.internal_compute_child:
        run_compute_child(args)
        return
    if not args.internal_worker or not args.worker_mode:
        raise ValueError("invalid internal invocation")
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.worker_mode == "managed":
        run_managed_worker(args)
        return
    set_worker_device_environment(local_rank, args.mps_percent)
    import torch

    if args.worker_mode == "baseline":
        run_baseline_worker(args, torch)
    elif args.worker_mode == "unmanaged_ordered":
        run_unmanaged_ordered_worker(args, torch)
    elif args.worker_mode == "restricted_compute":
        run_restricted_worker(args, torch, compute=True)
    elif args.worker_mode == "restricted_comm":
        run_restricted_worker(args, torch, compute=False)
    else:
        raise ValueError(args.worker_mode)


def forwarded_workload_args(args: argparse.Namespace) -> list[str]:
    values = [
        "--world-size",
        str(args.world_size),
        "--local-tokens",
        str(args.local_tokens),
        "--logical-ep-size",
        str(args.logical_ep_size),
        "--dim",
        str(args.dim),
        "--expert-width",
        str(args.expert_width),
        "--num-experts",
        str(args.num_experts),
        "--top-k",
        str(args.top_k),
        "--warmups",
        str(args.warmups),
        "--iterations",
        str(args.iterations),
        "--gate-sleep-cycles",
        str(args.gate_sleep_cycles),
        "--managed-launch-order",
        args.managed_launch_order,
    ]
    if args.allreduce_numel is not None:
        values.extend(["--allreduce-numel", str(args.allreduce_numel)])
    return values


def run_torchrun_phase(
    args: argparse.Namespace,
    mode: str,
    output: Path,
    trace_dir: Path,
    environment: dict[str, str],
    *,
    mps_percent: int | None = None,
    compute_mps_percent: int | None = None,
    comm_mps_percent: int | None = None,
    green_partition_target_sms: int | None = None,
    green_partition_side: str | None = None,
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
        "--phase-output",
        str(output),
        "--trace-dir",
        str(trace_dir),
        *forwarded_workload_args(args),
    ]
    if mps_percent is not None:
        command.extend(["--mps-percent", str(mps_percent)])
    if compute_mps_percent is not None:
        command.extend(["--compute-mps-percent", str(compute_mps_percent)])
    if comm_mps_percent is not None:
        command.extend(["--comm-mps-percent", str(comm_mps_percent)])
    if green_partition_target_sms is not None:
        command.extend(
            ["--green-partition-target-sms", str(green_partition_target_sms)]
        )
    if green_partition_side is not None:
        command.extend(["--green-partition-side", green_partition_side])
    completed = subprocess.run(command, env=environment, text=True, capture_output=True)
    if completed.returncode:
        message = [
            f"phase {mode!r} failed with exit code {completed.returncode}",
            "--- stdout (tail) ---",
            completed.stdout[-8000:],
            "--- stderr (tail) ---",
            completed.stderr[-16000:],
        ]
        raise RuntimeError("\n".join(message))
    if not output.is_file():
        raise RuntimeError(f"phase {mode!r} did not produce {output}")


def parse_profile(path: Path, total_sms: int, target: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    events = raw.get("traceEvents", raw) if isinstance(raw, dict) else raw
    kernels: list[dict[str, Any]] = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        name = str(event.get("name", ""))
        if target == "comm" and "nccl" not in name.lower():
            continue
        grid = (event.get("args") or {}).get("grid")
        block = (event.get("args") or {}).get("block")
        if not isinstance(grid, list) or len(grid) != 3:
            continue
        ctas = math.prod(int(value) for value in grid)
        kernels.append(
            {
                "name": name,
                "duration_us": float(event.get("dur", 0.0)),
                "grid": [int(value) for value in grid],
                "block": [int(value) for value in block] if isinstance(block, list) else None,
                "cta_count": ctas,
                "inferred_sm_demand": min(total_sms, ctas),
            }
        )
    if not kernels:
        raise RuntimeError(f"no {target} CUDA kernel grids found in {path}")
    dominant = max(kernels, key=lambda kernel: kernel["duration_us"])
    return {
        "kernel_count": len(kernels),
        "dominant_kernel": dominant,
        "kernels": kernels,
        "peak_inferred_sm_demand": max(
            int(kernel["inferred_sm_demand"]) for kernel in kernels
        ),
        "estimator": (
            "peak over operation kernels of min(device_sm_count, grid CTA count); "
            "this is a CTA-admission demand estimate, not a sampled hardware counter"
        ),
    }


def allocate_sms(total: int, compute_demand: int, comm_demand: int) -> dict[str, Any]:
    demand_sum = compute_demand + comm_demand
    if demand_sum <= total:
        compute_sms = compute_demand
        comm_sms = comm_demand
        overloaded = False
    else:
        exact_compute = total * compute_demand / demand_sum
        compute_sms = round(exact_compute)
        compute_sms = min(total - 1, max(1, compute_sms))
        comm_sms = total - compute_sms
        overloaded = True
    # MPS accepts integral percentages. Keep the two overloaded shares at 100%;
    # the runtime-visible SM counts are recorded and used to interpret results.
    compute_percent = max(1, round(100 * compute_sms / total))
    comm_percent = max(1, round(100 * comm_sms / total))
    if overloaded:
        comm_percent = 100 - compute_percent
    return {
        "overloaded": overloaded,
        "available_sms": total,
        "compute_demand_sms": compute_demand,
        "comm_demand_sms": comm_demand,
        "demand_sum_sms": demand_sum,
        "compute_allocated_sms": compute_sms,
        "comm_allocated_sms": comm_sms,
        "compute_share": compute_demand / demand_sum,
        "comm_share": comm_demand / demand_sum,
        "compute_mps_percent": compute_percent,
        "comm_mps_percent": comm_percent,
    }


def calibrate_green_allocation(allocation: dict[str, Any]) -> dict[str, Any]:
    """Resolve the proportional target to a driver-supported disjoint split."""
    compute_target = int(allocation["compute_allocated_sms"])
    comm_target = int(allocation["comm_allocated_sms"])
    if compute_target <= comm_target:
        partition_side = "compute"
        partition_target = compute_target
    else:
        partition_side = "comm"
        partition_target = comm_target
    pair = GreenContextPair(partition_target, partition_side, device=0)
    try:
        if pair.total_sms != int(allocation["available_sms"]):
            raise RuntimeError(
                f"Green Context driver reports {pair.total_sms} SMs, expected "
                f"{allocation['available_sms']}"
            )
        return {
            "partition_side": partition_side,
            "partition_target_sms": partition_target,
            "compute_actual_sms": pair.compute_sms,
            "comm_actual_sms": pair.comm_sms,
            "total_sms": pair.total_sms,
            "rounded": (
                pair.compute_sms != compute_target or pair.comm_sms != comm_target
            ),
        }
    finally:
        pair.close()


def green_context_probe(python: str, environment: dict[str, str]) -> dict[str, Any]:
    code = (
        "import json,torch\n"
        "result={'api_present':hasattr(torch.cuda,'GreenContext')}\n"
        "try:\n"
        " torch.cuda.set_device(0)\n"
        " ctx=torch.cuda.GreenContext.create(8,0)\n"
        " result.update({'available':True,'error':None})\n"
        " del ctx\n"
        "except Exception as error:\n"
        " result.update({'available':False,'error':str(error)})\n"
        "print(json.dumps(result))\n"
    )
    completed = subprocess.run(
        [python, "-c", code], env=environment, text=True, capture_output=True
    )
    try:
        result = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result = {
            "api_present": None,
            "available": False,
            "error": (completed.stderr or completed.stdout).strip(),
        }
    result["returncode"] = completed.returncode
    return result


def mps_processes() -> list[str]:
    completed = subprocess.run(
        ["pgrep", "-af", "[n]vidia-cuda-mps-(control|server)"],
        text=True,
        capture_output=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def start_mps(base: Path, visible_devices: str) -> dict[str, str]:
    existing = mps_processes()
    if existing:
        raise RuntimeError(
            "refusing to alter an existing CUDA MPS daemon: " + "; ".join(existing)
        )
    # CUDA_MPS_PIPE_DIRECTORY contains Unix-domain sockets and must stay below
    # the platform's short sun_path limit. The Lustre workspace path is already
    # longer than that, so use an owned, uniquely named /tmp directory.
    del base
    mps_root = Path(tempfile.mkdtemp(prefix="sm-partition-mps-", dir="/tmp"))
    pipe_dir = mps_root / "pipe"
    log_dir = mps_root / "log"
    pipe_dir.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "CUDA_MPS_PIPE_DIRECTORY": str(pipe_dir),
            "CUDA_MPS_LOG_DIRECTORY": str(log_dir),
            "SM_PARTITION_MPS_TEMP_ROOT": str(mps_root),
        }
    )
    deadline = time.monotonic() + 15
    attempts: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        completed = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            env=environment,
            text=True,
            capture_output=True,
        )
        attempts.append(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        if mps_processes():
            return environment
        # CUDA worker teardown is asynchronous. A daemon launched immediately
        # after torchrun can transiently fail with an empty diagnostic.
        time.sleep(0.5)
    logs: dict[str, str] = {}
    for path in log_dir.glob("*"):
        if path.is_file():
            logs[path.name] = path.read_text(encoding="utf-8", errors="replace")[-8000:]
    shutil.rmtree(mps_root, ignore_errors=True)
    raise RuntimeError(
        "CUDA MPS daemon did not become ready: "
        + json.dumps({"attempts": attempts, "logs": logs})
    )


def stop_mps(environment: dict[str, str]) -> None:
    completed = subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        env=environment,
        text=True,
        capture_output=True,
        timeout=15,
    )
    if completed.returncode:
        raise RuntimeError(
            "failed to stop CUDA MPS: " + (completed.stderr or completed.stdout)
        )
    mps_root = environment.get("SM_PARTITION_MPS_TEMP_ROOT")
    if mps_root:
        shutil.rmtree(mps_root, ignore_errors=True)


def probe_mps_visible_sms(environment: dict[str, str], percent: int) -> int:
    probe_environment = environment.copy()
    probe_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(percent),
            "CUDA_MPS_ENABLE_PER_CTX_DEVICE_MULTIPROCESSOR_PARTITIONING": "1",
        }
    )
    code = (
        "import torch; torch.cuda.set_device(0); "
        "print(torch.cuda.get_device_properties(0).multi_processor_count)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=probe_environment,
        text=True,
        capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"MPS {percent}% SM probe failed: "
            + (completed.stderr or completed.stdout)
        )
    try:
        return int(completed.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as error:
        raise RuntimeError(
            f"could not parse MPS {percent}% SM probe: {completed.stdout!r}"
        ) from error


def calibrate_mps_percent(
    environment: dict[str, str], target_sms: int, total_sms: int
) -> dict[str, Any]:
    """Find the integral MPS percentage whose visible SM count is closest."""
    percent = min(100, max(1, round(100 * target_sms / total_sms)))
    observations: dict[int, int] = {}
    for _ in range(12):
        if percent not in observations:
            observations[percent] = probe_mps_visible_sms(environment, percent)
        observed = observations[percent]
        if observed == target_sms:
            break
        next_percent = percent + (1 if observed < target_sms else -1)
        if not 1 <= next_percent <= 100 or next_percent in observations:
            break
        percent = next_percent
    selected_percent, selected_sms = min(
        observations.items(),
        key=lambda item: (abs(item[1] - target_sms), item[1] > target_sms, item[0]),
    )
    return {
        "target_sms": target_sms,
        "selected_percent": selected_percent,
        "visible_sms": selected_sms,
        "exact": selected_sms == target_sms,
        "observations": [
            {"percent": candidate, "visible_sms": visible}
            for candidate, visible in sorted(observations.items())
        ],
    }


def render_report(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    prediction = result["prediction"]
    allocation = result["allocation"]
    managed = result["managed_overlap"]
    compute_profile = result["profiles"]["compute"]
    comm_profile = result["profiles"]["comm"]
    shape = result["workload"]

    def ms(value: Any) -> str:
        return f"{float(value):.3f} ms"

    def measured(summary: dict[str, Any]) -> str:
        return f"{ms(summary['median'])} median; {ms(summary['p95'])} p95"

    lines = [
        "# Grouped GEMM × all-reduce SM-partition experiment",
        "",
        "## Workload",
        "",
        f"- GPU group: {shape['world_size']} × {baseline['device']['name']} "
        f"({baseline['device']['sm_count']} SMs/GPU)",
        f"- Grouped GEMM: {shape['local_experts']} groups of "
        f"[{shape['tokens_per_expert']}, {shape['dim']}] × "
        f"[{shape['dim']}, {shape['grouped_gemm']['n']}] BF16",
        f"- All-reduce: {shape['allreduce']['mib']:.0f} MiB BF16 over "
        f"{shape['world_size']} ranks",
        "",
        "## Isolated profile and allocation",
        "",
        "| Operation | Dominant grid CTAs | Inferred peak SM demand | Isolated | Proportional target / Green actual | Restricted isolated |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Compute | {compute_profile['dominant_kernel']['cta_count']} | "
        f"{compute_profile['peak_inferred_sm_demand']} | "
        f"{ms(baseline['isolated_compute']['summary_ms']['median'])} | "
        f"{allocation['compute_allocated_sms']} / "
        f"{allocation['compute_green_visible_sms']} SMs | "
        f"{ms(result['restricted_compute']['summary_ms']['median'])} |",
        f"| Communication | {comm_profile['dominant_kernel']['cta_count']} | "
        f"{comm_profile['peak_inferred_sm_demand']} | "
        f"{ms(baseline['isolated_comm']['summary_ms']['median'])} | "
        f"{allocation['comm_allocated_sms']} / "
        f"{allocation['comm_green_visible_sms']} SMs | "
        f"{ms(result['restricted_comm']['summary_ms']['median'])} |",
        "",
        "## Prediction and validation",
        "",
        "| Result | Runtime | Error vs prediction |",
        "|---|---:|---:|",
        f"| Predicted managed overlap | {ms(prediction['overlap_ms'])} | — |",
        f"| Measured overlap, unmanaged | "
        f"{measured(baseline['unmanaged_overlap']['summary_ms'])} | "
        f"{prediction['unmanaged_error_percent']:+.1f}% |",
        f"| Measured overlap, Green Context managed | "
        f"{measured(managed['summary_ms'])} | "
        f"{prediction['managed_error_percent']:+.1f}% |",
        "",
        f"Green Contexts allocated {result['restricted_compute']['visible_sm_count']} "
        f"SMs to compute and {result['restricted_comm']['visible_sm_count']} SMs "
        "to communication. Managed completion is host-clock timed because "
        "the two CUDA contexts cannot share timing events; per-sample launch skew and "
        "per-kernel CUDA durations are retained in the JSON.",
        "",
        f"Managed samples ranged from {ms(managed['summary_ms']['min'])} to "
        f"{ms(managed['summary_ms']['max'])} "
        f"(CV {100 * float(managed['summary_ms']['cv']):.1f}%). Both contexts were "
        "created from the same driver-level split, so their SM resources are disjoint.",
        "",
        "The SM-demand estimate is CTA-grid based, not an Nsight hardware-counter "
        "measurement. It is exactly the deliberately simple model under test.",
    ]
    return "\n".join(lines) + "\n"


def controller_main(args: argparse.Namespace) -> None:
    shape = workload(args)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    visible_devices = ",".join(str(index) for index in range(args.world_size))
    base_environment = os.environ.copy()
    base_environment["CUDA_VISIBLE_DEVICES"] = visible_devices
    base_environment.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    base_environment.setdefault("NCCL_CUMEM_ENABLE", "0")

    run_dir = Path(tempfile.mkdtemp(prefix="sm-partition-", dir=output.parent))
    traces = run_dir / "traces"
    baseline_path = run_dir / "baseline.json"
    restricted_compute_path = run_dir / "restricted_compute.json"
    restricted_comm_path = run_dir / "restricted_comm.json"
    managed_path = run_dir / "managed.json"
    try:
        print("[1/5] Measuring isolated kernels and unmanaged overlap", flush=True)
        run_torchrun_phase(
            args, "baseline", baseline_path, traces, base_environment
        )
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        total_sms = int(baseline["device"]["sm_count"])
        compute_profile = parse_profile(traces / "compute.json", total_sms, "compute")
        comm_profile = parse_profile(traces / "comm.json", total_sms, "comm")
        allocation = allocate_sms(
            total_sms,
            int(compute_profile["peak_inferred_sm_demand"]),
            int(comm_profile["peak_inferred_sm_demand"]),
        )
        green_calibration = calibrate_green_allocation(allocation)
        allocation.update(
            {
                "compute_green_visible_sms": green_calibration[
                    "compute_actual_sms"
                ],
                "comm_green_visible_sms": green_calibration["comm_actual_sms"],
                "green_calibration": green_calibration,
            }
        )
        print(
            "[2/5] Derived allocation: "
            f"proportional compute/comm="
            f"{allocation['compute_allocated_sms']}/"
            f"{allocation['comm_allocated_sms']} SMs; "
            f"Green Context actual="
            f"{allocation['compute_green_visible_sms']}/"
            f"{allocation['comm_green_visible_sms']} SMs",
            flush=True,
        )
        green_probe = green_context_probe(sys.executable, base_environment)

        partition_target = int(green_calibration["partition_target_sms"])
        partition_side = str(green_calibration["partition_side"])
        print(
            "[3/5] Measuring kernels in disjoint CUDA Green Context partitions",
            flush=True,
        )
        run_torchrun_phase(
            args,
            "restricted_compute",
            restricted_compute_path,
            traces,
            base_environment,
            green_partition_target_sms=partition_target,
            green_partition_side=partition_side,
        )
        run_torchrun_phase(
            args,
            "restricted_comm",
            restricted_comm_path,
            traces,
            base_environment,
            green_partition_target_sms=partition_target,
            green_partition_side=partition_side,
        )
        print(
            "[4/5] Measuring concurrent execution in disjoint Green Contexts",
            flush=True,
        )
        run_torchrun_phase(
            args,
            "managed",
            managed_path,
            traces,
            base_environment,
            green_partition_target_sms=partition_target,
            green_partition_side=partition_side,
        )

        restricted_compute = json.loads(
            restricted_compute_path.read_text(encoding="utf-8")
        )
        restricted_comm = json.loads(restricted_comm_path.read_text(encoding="utf-8"))
        managed = json.loads(managed_path.read_text(encoding="utf-8"))
        restricted_compute_profile = parse_profile(
            traces / "restricted_compute.json",
            int(restricted_compute["visible_sm_count"]),
            "compute",
        )
        restricted_comm_profile = parse_profile(
            traces / "restricted_comm.json",
            int(restricted_comm["visible_sm_count"]),
            "comm",
        )
        if allocation["overloaded"]:
            predicted_ms = max(
                float(restricted_compute["summary_ms"]["median"]),
                float(restricted_comm["summary_ms"]["median"]),
            )
            prediction_source = "maximum of restricted isolated medians"
        else:
            predicted_ms = max(
                float(baseline["isolated_compute"]["summary_ms"]["median"]),
                float(baseline["isolated_comm"]["summary_ms"]["median"]),
            )
            prediction_source = "perfect overlap of unrestricted isolated medians"
        unmanaged_ms = float(baseline["unmanaged_overlap"]["summary_ms"]["median"])
        managed_ms = float(managed["summary_ms"]["median"])
        result = {
            "schema_version": 1,
            "workload": shape,
            "methodology": {
                "sm_demand": compute_profile["estimator"],
                "allocation": (
                    "If demand sum exceeds device SMs, allocate in proportion to "
                    "isolated CTA-derived SM demand; otherwise assume perfect overlap."
                ),
                "management": (
                    "Two disjoint CUDA Green Contexts generated from one SM-resource "
                    "split per GPU"
                ),
                "green_context_probe": green_probe,
            },
            "profiles": {
                "compute": compute_profile,
                "comm": comm_profile,
                "restricted_compute": restricted_compute_profile,
                "restricted_comm": restricted_comm_profile,
            },
            "allocation": allocation,
            "baseline": baseline,
            "restricted_compute": restricted_compute,
            "restricted_comm": restricted_comm,
            "managed_overlap": managed,
            "prediction": {
                "overlap_ms": predicted_ms,
                "source": prediction_source,
                "unmanaged_measured_ms": unmanaged_ms,
                "managed_measured_ms": managed_ms,
                "unmanaged_error_percent": 100 * (unmanaged_ms / predicted_ms - 1),
                "managed_error_percent": 100 * (managed_ms / predicted_ms - 1),
            },
        }
        atomic_write_json(output, result)
        report_path = output.with_suffix(".md")
        report_path.write_text(render_report(result), encoding="utf-8")
        print(f"[5/5] Wrote {output} and {report_path}", flush=True)
    finally:
        if args.keep_traces:
            retained = output.parent / (output.stem + "_artifacts")
            if retained.exists():
                raise RuntimeError(f"refusing to replace existing artifact directory {retained}")
            shutil.move(str(run_dir), retained)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.internal_worker or args.internal_compute_child:
        run_internal(args)
    elif args.dry_run:
        print(json.dumps(workload(args), indent=2))
    else:
        compat = "/usr/local/cuda-12.8/compat"
        if Path(compat, "libcuda.so.1").is_file() or Path(
            compat, "libcuda.so.1"
        ).is_symlink():
            paths = [
                path
                for path in os.environ.get("LD_LIBRARY_PATH", "").split(":")
                if path and path != compat
            ]
            if os.environ.get("LD_LIBRARY_PATH", "").split(":")[0] != compat:
                environment = os.environ.copy()
                environment["LD_LIBRARY_PATH"] = ":".join([compat, *paths])
                os.execvpe(
                    sys.executable,
                    [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                    environment,
                )
        controller_main(args)


if __name__ == "__main__":
    main()
