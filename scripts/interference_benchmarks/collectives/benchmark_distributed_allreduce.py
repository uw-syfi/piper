#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Benchmark one NCCL all-reduce rank on each of several Ray nodes.

The actors are placed in a STRICT_SPREAD placement group, so every rank runs
on a distinct node.  CUDA events measure only the GPU-stream duration of each
collective; a synchronization after the end event makes every reported sample
complete before the next iteration starts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import statistics
import time
from typing import Any

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_cpus=1, num_gpus=1)
class AllReduceRank:
    def __init__(self, rank: int, physical_device_index: int | None = None) -> None:
        if physical_device_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_device_index)
        import torch

        self.rank = rank
        self.physical_device_index = physical_device_index
        self.torch = torch
        self.node_id = ray.get_runtime_context().get_node_id()
        self.node_ip = ray.util.get_node_ip_address()
        self.visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self.ray_gpu_ids = [str(gpu_id) for gpu_id in ray.get_gpu_ids()]
        torch.cuda.set_device(0)

    def metadata(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": socket.gethostname(),
            "ray_gpu_ids": self.ray_gpu_ids,
            "cuda_visible_devices": self.visible_devices,
            "cuda_device_name": self.torch.cuda.get_device_name(0),
            "physical_device_index_requested": self.physical_device_index,
        }

    def find_rendezvous(self) -> tuple[str, int]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.node_ip, 0))
            return self.node_ip, sock.getsockname()[1]

    def run(
        self,
        world_size: int,
        master_addr: str,
        master_port: int,
        numels: list[int],
        warmups: int,
        iterations: int,
        trace_output_dir: str | None,
        trace_iterations: int,
    ) -> dict[str, Any]:
        import torch.distributed as dist

        torch = self.torch
        init_method = f"tcp://{master_addr}:{master_port}"
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            rank=self.rank,
            world_size=world_size,
            device_id=torch.device("cuda", 0),
        )
        try:
            tensors = [
                torch.zeros(numel, dtype=torch.bfloat16, device="cuda")
                for numel in numels
            ]
            torch.cuda.synchronize()
            dist.barrier(device_ids=[0])

            for _ in range(warmups):
                for tensor in tensors:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            dist.barrier(device_ids=[0])

            cuda_times_ms: list[float] = []
            wall_times_ms: list[float] = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_start = time.perf_counter()
                start.record()
                for tensor in tensors:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                end.record()
                end.synchronize()
                wall_times_ms.append((time.perf_counter() - wall_start) * 1e3)
                cuda_times_ms.append(start.elapsed_time(end))

            dist.barrier(device_ids=[0])
            trace_path = None
            if trace_output_dir and trace_iterations:
                from torch.profiler import ProfilerActivity, profile, record_function

                trace_dir = Path(trace_output_dir)
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / (
                    f"isolated_reduce_bucket_rank{self.rank}_"
                    f"{socket.gethostname()}_gpu{self.physical_device_index}.json"
                )
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as profiler:
                    for trace_iteration in range(trace_iterations):
                        with record_function(
                            f"isolated_reduce_bucket_iter_{trace_iteration}"
                        ):
                            for tensor_index, tensor in enumerate(tensors):
                                with record_function(
                                    f"all_reduce_{tensor_index}_numel_{tensor.numel()}"
                                ):
                                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                        torch.cuda.synchronize()
                profiler.export_chrome_trace(str(trace_path))

            dist.barrier(device_ids=[0])
            payload_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
            median_ms = statistics.median(cuda_times_ms)
            algorithm_bandwidth_gbps = payload_bytes / (median_ms / 1e3) / 1e9
            bus_bandwidth_gbps = (
                algorithm_bandwidth_gbps * 2 * (world_size - 1) / world_size
            )
            return {
                **self.metadata(),
                "numels": numels,
                "dtype": str(tensors[0].dtype),
                "payload_bytes": payload_bytes,
                "warmups": warmups,
                "iterations": iterations,
                "trace_path": str(trace_path) if trace_path else None,
                "trace_iterations": trace_iterations,
                "cuda_times_ms": cuda_times_ms,
                "wall_times_ms": wall_times_ms,
                "cuda_median_ms": median_ms,
                "cuda_mean_ms": statistics.mean(cuda_times_ms),
                "cuda_min_ms": min(cuda_times_ms),
                "cuda_max_ms": max(cuda_times_ms),
                "algorithm_bandwidth_GBps": algorithm_bandwidth_gbps,
                "bus_bandwidth_GBps": bus_bandwidth_gbps,
            }
        finally:
            dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", required=True)
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--numel", type=int, default=622_329_856)
    parser.add_argument(
        "--numels",
        help="Comma-separated collective payload sequence; overrides --numel",
    )
    parser.add_argument(
        "--physical-device-index",
        type=int,
        choices=range(4),
        help="Use this physical GPU index on every node instead of Ray's assigned GPU",
    )
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--trace-output-dir")
    parser.add_argument("--trace-iterations", type=int, default=0)
    parser.add_argument("--output", help="Optional path for the JSON result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    numels = (
        [int(value) for value in args.numels.split(",")]
        if args.numels
        else [args.numel]
    )
    if not numels or any(numel <= 0 for numel in numels):
        raise SystemExit("Every collective payload must contain at least one element")
    if args.trace_iterations < 0:
        raise SystemExit("--trace-iterations must be non-negative")
    forwarded_env = {
        name: os.environ[name]
        for name in (
            "NCCL_CUMEM_ENABLE",
            "NCCL_ALGO",
            "NCCL_PROTO",
            "NCCL_DEBUG",
            "NCCL_DEBUG_SUBSYS",
            "NCCL_SOCKET_IFNAME",
        )
        if os.environ.get(name)
    }
    ray.init(
        address=f"{args.address}:{args.port}",
        namespace="allreduce_microbenchmark",
        runtime_env={"env_vars": forwarded_env} if forwarded_env else None,
    )
    pg = placement_group(
        [{"CPU": 1, "GPU": 1} for _ in range(args.world_size)],
        strategy="STRICT_SPREAD",
    )
    actors: list[Any] = []
    try:
        ray.get(pg.ready(), timeout=600)
        actors = [
            AllReduceRank.options(
                num_gpus=0 if args.physical_device_index is not None else 1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=rank,
                )
            ).remote(rank, args.physical_device_index)
            for rank in range(args.world_size)
        ]
        placements = ray.get([actor.metadata.remote() for actor in actors])
        node_ids = {placement["node_id"] for placement in placements}
        if len(node_ids) != args.world_size:
            raise RuntimeError(
                f"Expected {args.world_size} distinct nodes, got {len(node_ids)}: "
                f"{placements}"
            )
        master_addr, master_port = ray.get(actors[0].find_rendezvous.remote())
        futures = [
            actor.run.remote(
                args.world_size,
                master_addr,
                master_port,
                numels,
                args.warmups,
                args.iterations,
                str(Path(args.trace_output_dir).resolve())
                if args.trace_output_dir
                else None,
                args.trace_iterations,
            )
            for actor in actors
        ]
        results = ray.get(futures)
        rank_medians = [result["cuda_median_ms"] for result in results]
        output = {
            "configuration": {
                **vars(args),
                "resolved_numels": numels,
                "dtype": "torch.bfloat16",
                "payload_bytes": sum(numels) * 2,
                "forwarded_env": forwarded_env,
            },
            "placements": placements,
            "rank_results": results,
            "aggregate": {
                "median_of_rank_medians_ms": statistics.median(rank_medians),
                "mean_of_rank_medians_ms": statistics.mean(rank_medians),
                "min_rank_median_ms": min(rank_medians),
                "max_rank_median_ms": max(rank_medians),
            },
        }
        rendered = json.dumps(output, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
