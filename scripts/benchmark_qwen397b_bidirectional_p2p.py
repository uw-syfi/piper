#!/usr/bin/env python3
"""Benchmark concurrent bidirectional Qwen3.5-397B pipeline P2P traffic.

The benchmark places exactly two one-GPU Ray actors on explicitly selected
nodes in different NVLink domains.  It compares an A -> B transfer and a
B -> A transfer in isolation with the two transfers running concurrently.
The concurrent directions use distinct ProcessGroupNCCL communicators and
CUDA streams, which makes them separate GPU communication kernels instead of
one grouped send/recv kernel.

The production-shaped default is the pipeline boundary in
``examples.models.qwen3.PiperQwen3Model``: a BF16 residual hidden state with
shape ``[microbatch=1, sequence=32768, dim=4096]``.  The corresponding forward
activation and backward gradient each contain 134,217,728 elements (256 MiB).
Sequence-length and payload-scale sweeps are supported for sensitivity tests.

``--dry-run`` resolves all payloads without importing Ray or touching CUDA.
Real runs require two explicitly pinned node IPs and two distinct NVLink-domain
labels.  The script writes detailed JSON samples and a flat CSV summary.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, is_dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from typing import Any


MODEL_NAME = "Qwen3.5-397B-A17B"
MODEL_SYMBOL = "_QWEN35_397B"
DTYPE_NAME = "bf16"
DTYPE_BYTES = 2
RANK_A = 0
RANK_B = 1


@dataclass(frozen=True)
class Payload:
    name: str
    microbatch_size: int
    seq_len: int
    dim: int
    scale: float | None
    numel: int
    dtype: str = DTYPE_NAME

    @property
    def bytes(self) -> int:
        return self.numel * DTYPE_BYTES

    @property
    def mib(self) -> float:
        return self.bytes / 2**20

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.microbatch_size, self.seq_len, self.dim)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {}
    mean = statistics.mean(values)
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": mean,
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "cv": statistics.pstdev(values) / mean if mean else 0.0,
    }


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_qwen_config() -> dict[str, Any]:
    repo_root = str(Path(__file__).resolve().parents[1])
    if sys.path[0] != repo_root:
        sys.path.insert(0, repo_root)
    from examples.models.qwen3 import _QWEN35_397B

    if is_dataclass(_QWEN35_397B):
        config = asdict(_QWEN35_397B)
    else:
        config = dict(vars(_QWEN35_397B))
    required = {"dim", "n_layers", "vocab_size", "num_experts", "top_k"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{MODEL_SYMBOL} is missing required fields: {missing}")
    return config


def build_payloads(
    args: argparse.Namespace, config: dict[str, Any]
) -> list[Payload]:
    payloads: list[Payload] = []
    dim = int(config["dim"])
    for seq_len in parse_csv_ints(args.seq_lens):
        architectural_numel = args.microbatch_size * seq_len * dim
        for scale in parse_csv_floats(args.payload_scales):
            numel = int(round(architectural_numel * scale))
            if numel <= 0:
                raise ValueError("resolved payload numel must be positive")
            scale_name = str(scale).replace(".", "p")
            payloads.append(
                Payload(
                    name=f"hidden_mb{args.microbatch_size}_seq{seq_len}_x{scale_name}",
                    microbatch_size=args.microbatch_size,
                    seq_len=seq_len,
                    dim=dim,
                    scale=scale,
                    numel=numel,
                )
            )
    for index, mib in enumerate(parse_csv_floats(args.custom_payload_mib or "")):
        numel = int(round(mib * 2**20 / DTYPE_BYTES))
        if numel <= 0:
            raise ValueError("custom payload sizes must be positive")
        payloads.append(
            Payload(
                name=f"custom_{index}_{str(mib).replace('.', 'p')}mib",
                microbatch_size=args.microbatch_size,
                seq_len=-1,
                dim=dim,
                scale=None,
                numel=numel,
            )
        )
    if not payloads:
        raise ValueError("at least one payload must be configured")
    names = [payload.name for payload in payloads]
    if len(names) != len(set(names)):
        raise ValueError(f"payload names are not unique: {names}")
    return payloads


def benchmark_modes() -> list[str]:
    return [
        "solo_ab",
        "solo_ba",
        "concurrent_ab_first",
        "concurrent_ba_first",
    ]


def balanced_mode_order(modes: list[str], sample_index: int) -> list[str]:
    """Rotate balanced blocks so every mode occupies every list position."""

    if not modes:
        return []
    square = sample_index // len(modes)
    base = modes if square % 2 == 0 else list(reversed(modes))
    shift = sample_index % len(modes)
    return base[shift:] + base[:shift]


def build_preview(
    args: argparse.Namespace,
    config: dict[str, Any],
    payloads: list[Payload],
) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "model_symbol": MODEL_SYMBOL,
        "qwen_config": config,
        "pipeline_boundary": {
            "source": "PiperQwen3Model stage annotation",
            "tensor": "residual hidden state (forward activation or backward gradient)",
            "architectural_shape": [
                args.microbatch_size,
                "sequence_length",
                config["dim"],
            ],
            "dtype": DTYPE_NAME,
            "note": (
                "All annotated transformer-stage boundaries have the same hidden "
                "dimension. Scales other than 1.0 and custom MiB values are "
                "sensitivity/stress variants, not additional model boundaries."
            ),
        },
        "payloads": [
            asdict(payload)
            | {
                "shape": list(payload.shape),
                "bytes": payload.bytes,
                "mib": payload.mib,
            }
            for payload in payloads
        ],
        "placement": {
            "world_size": 2,
            "rank_a": RANK_A,
            "rank_b": RANK_B,
            "expected_node_ips": parse_csv_strings(args.expected_node_ips or ""),
            "expected_nvlink_domains": parse_csv_strings(
                args.expected_nvlink_domains or ""
            ),
            "requirements": (
                "one Ray actor and one GPU per selected node; the two selected "
                "nodes must have distinct NVLink-domain labels"
            ),
        },
        "communicators": {
            "ab": "dedicated two-rank ProcessGroupNCCL communicator",
            "ba": "second dedicated two-rank ProcessGroupNCCL communicator",
            "control": "Gloo barrier group",
            "reason": (
                "Distinct NCCL communicators and CUDA streams produce independently "
                "queueable P2P kernels for the two directions."
            ),
        },
        "measurement": {
            "warmups_per_mode_per_payload": args.warmups,
            "iterations_per_trial": args.iterations,
            "trials": args.trials,
            "samples_per_mode_per_payload": args.iterations * args.trials,
            "modes": benchmark_modes(),
            "gate_sleep_cycles": args.gate_sleep_cycles,
            "ab_delay_cycles": args.ab_delay_cycles,
            "ba_delay_cycles": args.ba_delay_cycles,
            "timing": (
                "CUDA events on each direction stream; a direction completes at "
                "the slower of its sender and receiver endpoints"
            ),
        },
    }


class BidirectionalP2PRank:
    def __init__(self, rank: int, node_slot: int) -> None:
        import ray
        import torch

        self.rank = rank
        self.node_slot = node_slot
        self.ray = ray
        self.torch = torch
        torch.cuda.set_device(0)
        self.node_id = ray.get_runtime_context().get_node_id()
        self.node_ip = ray.util.get_node_ip_address()
        self.visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    def rendezvous(self) -> tuple[str, int]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self.node_ip, 0))
            return self.node_ip, sock.getsockname()[1]

    def metadata(self) -> dict[str, Any]:
        torch = self.torch
        properties = torch.cuda.get_device_properties(0)
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "rank": self.rank,
            "node_slot": self.node_slot,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": socket.gethostname(),
            "ray_gpu_ids": [str(value) for value in self.ray.get_gpu_ids()],
            "cuda_visible_devices": self.visible_devices,
            "cuda_name": properties.name,
            "cuda_total_memory": properties.total_memory,
            "cuda_sm_count": properties.multi_processor_count,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "nccl_version": torch.cuda.nccl.version(),
            "nvidia_smi": smi.stdout.strip(),
        }

    def run(
        self,
        master_addr: str,
        master_port: int,
        payload_dicts: list[dict[str, Any]],
        warmups: int,
        iterations: int,
        trials: int,
        gate_sleep_cycles: int,
        ab_delay_cycles: int,
        ba_delay_cycles: int,
    ) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        payloads = [Payload(**item) for item in payload_dicts]
        torch.cuda.set_device(0)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=self.rank,
            world_size=2,
            device_id=torch.device("cuda", 0),
            timeout=timedelta(minutes=60),
        )
        control_group = None
        try:
            ranks = [RANK_A, RANK_B]
            control_group = dist.new_group(ranks=ranks, backend="gloo")
            # All ranks create the two same-membership NCCL groups in the same
            # order. They are intentionally different communicators.
            ab_group = dist.new_group(ranks=ranks, backend="nccl")
            ba_group = dist.new_group(ranks=ranks, backend="nccl")

            ab_stream = torch.cuda.Stream(priority=0)
            ba_stream = torch.cuda.Stream(priority=0)
            control_stream = torch.cuda.Stream(priority=-1)
            max_numel = max(payload.numel for payload in payloads)
            send_value = 101.0 if self.rank == RANK_A else 202.0
            send_buffer = torch.full(
                (max_numel,), send_value, dtype=torch.bfloat16, device="cuda"
            )
            recv_buffer = torch.full(
                (max_numel,), -1.0, dtype=torch.bfloat16, device="cuda"
            )

            def enqueue_direction(
                direction: str, tensor: Any
            ) -> list[Any]:
                if direction == "ab":
                    group = ab_group
                    source = RANK_A
                    destination = RANK_B
                elif direction == "ba":
                    group = ba_group
                    source = RANK_B
                    destination = RANK_A
                else:
                    raise ValueError(direction)
                if self.rank == source:
                    op = dist.P2POp(
                        dist.isend,
                        tensor,
                        destination,
                        group=group,
                    )
                else:
                    op = dist.P2POp(
                        dist.irecv,
                        tensor,
                        source,
                        group=group,
                    )
                # batch_isend_irecv wraps even the one local operation in
                # ncclGroupStart/End. Both peers enter the same communicator
                # operation, avoiding asymmetric first-call deadlocks.
                return dist.batch_isend_irecv([op])

            def reset_receive() -> None:
                recv_buffer.fill_(-1.0)
                torch.cuda.synchronize()

            # Initialize both communicator connection paths before timing.
            init_numel = min(max_numel, 1024)
            init_tensor = (
                send_buffer[:init_numel]
                if self.rank == RANK_A
                else recv_buffer[:init_numel]
            )
            with torch.cuda.stream(ab_stream):
                init_ab = enqueue_direction("ab", init_tensor)
                for work in init_ab:
                    work.wait()
            torch.cuda.synchronize()
            reset_receive()
            init_tensor = (
                send_buffer[:init_numel]
                if self.rank == RANK_B
                else recv_buffer[:init_numel]
            )
            with torch.cuda.stream(ba_stream):
                init_ba = enqueue_direction("ba", init_tensor)
                for work in init_ba:
                    work.wait()
            torch.cuda.synchronize()
            reset_receive()
            dist.barrier(group=control_group)

            def run_once(
                payload: Payload,
                mode: str,
                trial: int,
                iteration: int,
            ) -> dict[str, float | int]:
                has_ab = mode != "solo_ba"
                has_ba = mode != "solo_ab"
                if self.rank == RANK_A:
                    ab_tensor = send_buffer[: payload.numel]
                    ba_tensor = recv_buffer[: payload.numel]
                else:
                    ab_tensor = recv_buffer[: payload.numel]
                    ba_tensor = send_buffer[: payload.numel]

                gate = torch.cuda.Event(enable_timing=True)
                ab_start = torch.cuda.Event(enable_timing=True)
                ab_end = torch.cuda.Event(enable_timing=True)
                ba_start = torch.cuda.Event(enable_timing=True)
                ba_end = torch.cuda.Event(enable_timing=True)
                ab_work: list[Any] = []
                ba_work: list[Any] = []
                dist.barrier(group=control_group)

                # Queue a future device event first. Both host threads have time
                # to enqueue both communicators before either GPU releases work.
                with torch.cuda.stream(control_stream):
                    torch.cuda._sleep(gate_sleep_cycles)
                    gate.record()

                def launch_ab() -> None:
                    nonlocal ab_work
                    ab_stream.wait_event(gate)
                    with torch.cuda.stream(ab_stream):
                        if ab_delay_cycles:
                            torch.cuda._sleep(ab_delay_cycles)
                        ab_start.record()
                        ab_work = enqueue_direction("ab", ab_tensor)

                def finish_ab() -> None:
                    if not ab_work:
                        raise RuntimeError("AB completion requested before enqueue")
                    with torch.cuda.stream(ab_stream):
                        for work in ab_work:
                            work.wait()
                        ab_end.record()

                def launch_ba() -> None:
                    nonlocal ba_work
                    ba_stream.wait_event(gate)
                    with torch.cuda.stream(ba_stream):
                        if ba_delay_cycles:
                            torch.cuda._sleep(ba_delay_cycles)
                        ba_start.record()
                        ba_work = enqueue_direction("ba", ba_tensor)

                def finish_ba() -> None:
                    if not ba_work:
                        raise RuntimeError("BA completion requested before enqueue")
                    with torch.cuda.stream(ba_stream):
                        for work in ba_work:
                            work.wait()
                        ba_end.record()

                if mode == "solo_ab":
                    launch_ab()
                    finish_ab()
                elif mode == "solo_ba":
                    launch_ba()
                    finish_ba()
                elif mode == "concurrent_ab_first":
                    # Both communicators are enqueued before either completion
                    # wait. Every rank uses the same communicator launch order.
                    launch_ab()
                    launch_ba()
                    finish_ab()
                    finish_ba()
                elif mode == "concurrent_ba_first":
                    launch_ba()
                    launch_ab()
                    finish_ba()
                    finish_ab()
                else:
                    raise ValueError(mode)

                if has_ab:
                    ab_end.synchronize()
                if has_ba:
                    ba_end.synchronize()
                dist.barrier(group=control_group)

                sample: dict[str, float | int] = {
                    "trial": trial,
                    "iteration": iteration,
                }
                starts: list[float] = []
                ends: list[float] = []
                if has_ab:
                    sample["ab_ms"] = ab_start.elapsed_time(ab_end)
                    sample["ab_gate_to_start_ms"] = gate.elapsed_time(ab_start)
                    sample["ab_gate_to_end_ms"] = gate.elapsed_time(ab_end)
                    starts.append(float(sample["ab_gate_to_start_ms"]))
                    ends.append(float(sample["ab_gate_to_end_ms"]))
                if has_ba:
                    sample["ba_ms"] = ba_start.elapsed_time(ba_end)
                    sample["ba_gate_to_start_ms"] = gate.elapsed_time(ba_start)
                    sample["ba_gate_to_end_ms"] = gate.elapsed_time(ba_end)
                    starts.append(float(sample["ba_gate_to_start_ms"]))
                    ends.append(float(sample["ba_gate_to_end_ms"]))
                sample["pair_ms"] = max(ends) - min(starts)
                if len(starts) == 2:
                    sample["start_skew_ms"] = abs(starts[0] - starts[1])
                return sample

            case_results: list[dict[str, Any]] = []
            modes = benchmark_modes()
            for payload_index, payload in enumerate(payloads):
                for warmup in range(warmups):
                    order = balanced_mode_order(
                        modes, payload_index * max(warmups, 1) + warmup
                    )
                    for mode in order:
                        run_once(payload, mode, -1, warmup)

                samples: dict[str, list[dict[str, float | int]]] = {
                    mode: [] for mode in modes
                }
                for trial in range(trials):
                    for iteration in range(iterations):
                        order = balanced_mode_order(
                            modes,
                            payload_index * iterations * trials
                            + trial * iterations
                            + iteration,
                        )
                        for mode in order:
                            samples[mode].append(
                                run_once(payload, mode, trial, iteration)
                            )

                # Validate both directions outside timed samples.
                reset_receive()
                run_once(payload, "solo_ab", -2, 0)
                ab_observed = (
                    float(recv_buffer[0].item())
                    if self.rank == RANK_B
                    else None
                )
                if self.rank == RANK_B and ab_observed != 101.0:
                    raise RuntimeError(
                        f"AB validation failed on rank B: {ab_observed} != 101"
                    )
                reset_receive()
                run_once(payload, "solo_ba", -2, 1)
                ba_observed = (
                    float(recv_buffer[0].item())
                    if self.rank == RANK_A
                    else None
                )
                if self.rank == RANK_A and ba_observed != 202.0:
                    raise RuntimeError(
                        f"BA validation failed on rank A: {ba_observed} != 202"
                    )
                dist.barrier(group=control_group)

                summaries = {
                    mode: {
                        metric: summarize(
                            [float(sample[metric]) for sample in mode_samples]
                        )
                        for metric in mode_samples[0]
                        if metric not in {"trial", "iteration"}
                    }
                    for mode, mode_samples in samples.items()
                }
                case_results.append(
                    {
                        "payload": asdict(payload)
                        | {
                            "shape": list(payload.shape),
                            "bytes": payload.bytes,
                            "mib": payload.mib,
                        },
                        "samples": samples,
                        "summaries": summaries,
                        "validation": {
                            "ab_observed_on_receiver": ab_observed,
                            "ab_expected_on_receiver": (
                                101.0 if self.rank == RANK_B else None
                            ),
                            "ba_observed_on_receiver": ba_observed,
                            "ba_expected_on_receiver": (
                                202.0 if self.rank == RANK_A else None
                            ),
                        },
                    }
                )
            dist.barrier(group=control_group)
            return {
                "metadata": self.metadata(),
                "case_results": case_results,
            }
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def aggregate_case(
    rank_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    modes = benchmark_modes()
    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        mode_samples = rank_cases[0]["samples"][mode]
        metrics = [
            metric
            for metric in mode_samples[0]
            if metric not in {"trial", "iteration"}
        ]
        mode_summaries[mode] = {}
        for metric in metrics:
            rank_medians = [
                statistics.median(
                    float(sample[metric]) for sample in case["samples"][mode]
                )
                for case in rank_cases
            ]
            all_samples = [
                float(sample[metric])
                for case in rank_cases
                for sample in case["samples"][mode]
            ]
            mode_summaries[mode][metric] = {
                "rank_medians": rank_medians,
                "median_rank_median": statistics.median(rank_medians),
                "max_rank_median": max(rank_medians),
                "all_samples": summarize(all_samples),
            }

    sample_count = len(rank_cases[0]["samples"]["solo_ab"])
    derived: dict[str, Any] = {}
    for concurrent in ("concurrent_ab_first", "concurrent_ba_first"):
        critical_samples: list[dict[str, float | int]] = []
        for sample_index in range(sample_count):
            ab_solo = max(
                float(case["samples"]["solo_ab"][sample_index]["ab_ms"])
                for case in rank_cases
            )
            ba_solo = max(
                float(case["samples"]["solo_ba"][sample_index]["ba_ms"])
                for case in rank_cases
            )
            ab_concurrent = max(
                float(case["samples"][concurrent][sample_index]["ab_ms"])
                for case in rank_cases
            )
            ba_concurrent = max(
                float(case["samples"][concurrent][sample_index]["ba_ms"])
                for case in rank_cases
            )
            pair_ms = max(
                float(case["samples"][concurrent][sample_index]["pair_ms"])
                for case in rank_cases
            )
            start_skew_ms = max(
                float(case["samples"][concurrent][sample_index]["start_skew_ms"])
                for case in rank_cases
            )
            ideal_pair = max(ab_solo, ba_solo)
            serialized_pair = ab_solo + ba_solo
            critical_samples.append(
                {
                    "trial": rank_cases[0]["samples"][concurrent][sample_index][
                        "trial"
                    ],
                    "iteration": rank_cases[0]["samples"][concurrent][sample_index][
                        "iteration"
                    ],
                    "ab_solo_ms": ab_solo,
                    "ba_solo_ms": ba_solo,
                    "ab_concurrent_ms": ab_concurrent,
                    "ba_concurrent_ms": ba_concurrent,
                    "ab_slowdown": ab_concurrent / ab_solo,
                    "ba_slowdown": ba_concurrent / ba_solo,
                    "ideal_isolated_pair_ms": ideal_pair,
                    "normal_pair_ms": pair_ms,
                    "pair_overhead_vs_isolated": pair_ms / ideal_pair,
                    "removable_fraction_of_normal": 1.0 - ideal_pair / pair_ms,
                    "speedup_vs_serialized": serialized_pair / pair_ms,
                    "overlap_efficiency": (
                        (serialized_pair - pair_ms) / min(ab_solo, ba_solo)
                    ),
                    "start_skew_ms": start_skew_ms,
                }
            )
        derived[concurrent] = {
            "world_critical_samples": critical_samples,
            "world_critical": {
                metric: summarize(
                    [float(sample[metric]) for sample in critical_samples]
                )
                for metric in critical_samples[0]
                if metric not in {"trial", "iteration"}
            },
        }
    return {"modes": mode_summaries, "derived": derived}


def aggregate(
    rank_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_count = len(rank_results[0]["case_results"])
    if any(len(result["case_results"]) != case_count for result in rank_results):
        raise ValueError("rank results contain different payload counts")
    aggregated: list[dict[str, Any]] = []
    for case_index in range(case_count):
        rank_cases = [
            result["case_results"][case_index] for result in rank_results
        ]
        payload = rank_cases[0]["payload"]
        if any(case["payload"] != payload for case in rank_cases[1:]):
            raise ValueError("payload order differs across rank results")
        aggregated.append(
            {
                "payload": payload,
                **aggregate_case(rank_cases),
            }
        )
    return aggregated


def csv_rows(aggregate_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in aggregate_cases:
        payload = case["payload"]
        for mode, derived in case["derived"].items():
            metrics = derived["world_critical"]
            rows.append(
                {
                    "model": MODEL_NAME,
                    "payload_name": payload["name"],
                    "microbatch_size": payload["microbatch_size"],
                    "seq_len": payload["seq_len"],
                    "dim": payload["dim"],
                    "scale": payload["scale"],
                    "dtype": payload["dtype"],
                    "numel": payload["numel"],
                    "payload_bytes": payload["bytes"],
                    "payload_mib": payload["mib"],
                    "mode": mode,
                    "queue_first": (
                        "A_to_B"
                        if mode == "concurrent_ab_first"
                        else "B_to_A"
                    ),
                    "samples": metrics["ab_slowdown"]["count"],
                    "ab_solo_ms_median": metrics["ab_solo_ms"]["median"],
                    "ab_concurrent_ms_median": metrics["ab_concurrent_ms"][
                        "median"
                    ],
                    "ab_slowdown_median": metrics["ab_slowdown"]["median"],
                    "ba_solo_ms_median": metrics["ba_solo_ms"]["median"],
                    "ba_concurrent_ms_median": metrics["ba_concurrent_ms"][
                        "median"
                    ],
                    "ba_slowdown_median": metrics["ba_slowdown"]["median"],
                    "ideal_pair_ms_median": metrics[
                        "ideal_isolated_pair_ms"
                    ]["median"],
                    "normal_pair_ms_median": metrics["normal_pair_ms"]["median"],
                    "pair_overhead_median": metrics[
                        "pair_overhead_vs_isolated"
                    ]["median"],
                    "removable_fraction_median": metrics[
                        "removable_fraction_of_normal"
                    ]["median"],
                    "speedup_vs_serialized_median": metrics[
                        "speedup_vs_serialized"
                    ]["median"],
                    "overlap_efficiency_median": metrics["overlap_efficiency"][
                        "median"
                    ],
                    "start_skew_ms_median": metrics["start_skew_ms"]["median"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure two cross-domain P2P directions alone and concurrently "
            "using Qwen3.5-397B pipeline-boundary payloads."
        )
    )
    parser.add_argument("--address", default="10.12.249.162")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument(
        "--expected-node-ips",
        help=(
            "Exactly two comma-separated Ray node IPs, one from each selected "
            "NVLink domain; required for a real run"
        ),
    )
    parser.add_argument(
        "--expected-nvlink-domains",
        help=(
            "Exactly two distinct comma-separated domain labels corresponding "
            "to --expected-node-ips; required for a real run"
        ),
    )
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument(
        "--seq-lens",
        default="32768",
        help="Comma-separated sequence lengths for hidden-state payloads",
    )
    parser.add_argument(
        "--payload-scales",
        default="1",
        help=(
            "Comma-separated scales of the architectural hidden-state payload; "
            "1 is production-shaped and other values are stress variants"
        ),
    )
    parser.add_argument(
        "--custom-payload-mib",
        help="Optional comma-separated custom BF16 payload sizes in MiB",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--gate-sleep-cycles",
        type=int,
        default=20_000_000,
        help="Device-side delay before releasing both direction streams",
    )
    parser.add_argument(
        "--ab-delay-cycles",
        type=int,
        default=0,
        help="Optional device delay after the common gate and before A -> B",
    )
    parser.add_argument(
        "--ba-delay-cycles",
        type=int,
        default=0,
        help="Optional device delay after the common gate and before B -> A",
    )
    parser.add_argument(
        "--dry-run",
        "--print-config",
        action="store_true",
        dest="dry_run",
        help="Resolve configuration without Ray, NCCL, CUDA, or output writes",
    )
    parser.add_argument(
        "--output",
        default=(
            "out/interference_benchmarks/qwen397b_bidirectional_p2p/results.json"
        ),
    )
    parser.add_argument(
        "--csv-output",
        help="CSV path; defaults to summary.csv next to --output",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "port": args.port,
        "microbatch-size": args.microbatch_size,
        "iterations": args.iterations,
        "trials": args.trials,
        "gate-sleep-cycles": args.gate_sleep_cycles,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.warmups < 0:
        raise ValueError("warmups may not be negative")
    if args.ab_delay_cycles < 0 or args.ba_delay_cycles < 0:
        raise ValueError("direction delay cycles may not be negative")
    seq_lens = parse_csv_ints(args.seq_lens)
    scales = parse_csv_floats(args.payload_scales)
    if not seq_lens or any(value <= 0 for value in seq_lens):
        raise ValueError("seq-lens must contain positive integers")
    if not scales or any(value <= 0 for value in scales):
        raise ValueError("payload-scales must contain positive numbers")
    if args.custom_payload_mib:
        custom = parse_csv_floats(args.custom_payload_mib)
        if not custom or any(value <= 0 for value in custom):
            raise ValueError("custom-payload-mib must contain positive numbers")


def validate_real_placement_args(
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    ips = parse_csv_strings(args.expected_node_ips or "")
    domains = parse_csv_strings(args.expected_nvlink_domains or "")
    if len(ips) != 2 or len(set(ips)) != 2:
        raise ValueError(
            "a real run requires exactly two distinct --expected-node-ips"
        )
    if len(domains) != 2 or len(set(domains)) != 2:
        raise ValueError(
            "a real run requires exactly two distinct "
            "--expected-nvlink-domains"
        )
    return ips, domains


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_qwen_config()
    payloads = build_payloads(args, config)
    preview = build_preview(args, config, payloads)
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return

    expected_ips, expected_domains = validate_real_placement_args(args)
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    forwarded_env = {
        name: os.environ[name]
        for name in (
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "NCCL_ALGO",
            "NCCL_CUMEM_ENABLE",
            "NCCL_DEBUG",
            "NCCL_DEBUG_FILE",
            "NCCL_DEBUG_SUBSYS",
            "NCCL_IB_DISABLE",
            "NCCL_LAUNCH_ORDER_IMPLICIT",
            "NCCL_MAX_CTAS",
            "NCCL_MIN_CTAS",
            "NCCL_MNNVL_ENABLE",
            "NCCL_PROTO",
            "NCCL_SOCKET_IFNAME",
        )
        if os.environ.get(name)
    }
    # Required by NCCL 2.26+ for safe concurrent launch across communicators.
    forwarded_env.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    ray.init(
        address=f"{args.address}:{args.port}",
        namespace="qwen397b_bidirectional_p2p_interference",
        runtime_env={"env_vars": forwarded_env},
    )
    bundles = [
        {"CPU": 1, "GPU": 1, f"node:{node_ip}": 0.001}
        for node_ip in expected_ips
    ]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    actor_type = ray.remote(num_cpus=1, num_gpus=1)(BidirectionalP2PRank)
    actors: list[Any] = []
    try:
        ray.get(pg.ready(), timeout=600)
        for rank in range(2):
            actors.append(
                actor_type.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=rank,
                    )
                ).remote(rank, rank)
            )
        placements = ray.get([actor.metadata.remote() for actor in actors])
        if [item["node_ip"] for item in placements] != expected_ips:
            raise RuntimeError(
                "Ray placement does not match requested node IP order: "
                f"{[item['node_ip'] for item in placements]} != {expected_ips}"
            )
        node_counts = Counter(item["node_id"] for item in placements)
        if sorted(node_counts.values()) != [1, 1]:
            raise RuntimeError(
                f"Expected one actor on each of two nodes: {node_counts}"
            )
        gpu_assignments = {
            (item["node_id"], tuple(item["ray_gpu_ids"])) for item in placements
        }
        if len(gpu_assignments) != 2:
            raise RuntimeError(
                f"GPU assignment is not unique across actors: {placements}"
            )

        master_addr, master_port = ray.get(actors[RANK_A].rendezvous.remote())
        started = time.time()
        rank_results = ray.get(
            [
                actor.run.remote(
                    master_addr,
                    master_port,
                    [asdict(payload) for payload in payloads],
                    args.warmups,
                    args.iterations,
                    args.trials,
                    args.gate_sleep_cycles,
                    args.ab_delay_cycles,
                    args.ba_delay_cycles,
                )
                for actor in actors
            ]
        )
        finished = time.time()
        aggregate_cases = aggregate(rank_results)
        output = {
            "schema_version": 1,
            "command_args": vars(args),
            "resolved_config": preview,
            "environment": {
                "forwarded_env": forwarded_env,
                "started_unix_s": started,
                "finished_unix_s": finished,
                "elapsed_s": finished - started,
            },
            "placement": {
                "expected_node_ips": expected_ips,
                "expected_nvlink_domains": expected_domains,
                "observed": placements,
            },
            "rank_results": rank_results,
            "aggregate": aggregate_cases,
        }
        output_path = Path(args.output)
        csv_path = (
            Path(args.csv_output)
            if args.csv_output
            else output_path.with_name("summary.csv")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        rows = csv_rows(aggregate_cases)
        write_csv(csv_path, rows)
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        print(f"Wrote {output_path}")
        print(f"Wrote {csv_path}")
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
