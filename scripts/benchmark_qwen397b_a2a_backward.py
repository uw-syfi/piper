#!/usr/bin/env python3
"""Measure Qwen3.5-397B backward-compute x collective interference.

The benchmark launches 16 one-GPU Ray actors as four ranks on each of four
GB200 compute trays.  Every actor joins a NCCL communication group and a
separate Gloo control group.  The communication operation may be an EP
all-to-all or an FSDP-style all-gather.  After a host-side readiness barrier, a
control stream queues a short delay followed by a future start event; the work
streams wait for that event so compute and communication are both queued before
their device release.

The synthetic compute kernels use the public Qwen3.5-397B tensor dimensions:
D=4096, routed expert width=1024, 512 experts, top-k=10.  They intentionally do
not instantiate the full training model. The expert backward instead uses
production-style packed gate/up and down batched matmuls, compiled through
TorchInductor. Attention-projection profiles include the production hybrid
backbone's exact Gated-DeltaNet and gated-full-attention projection shapes. A
legacy ``linear_projection`` profile keeps the fused QKV+Z stress surrogate
used by the first experiment set and labels it as a surrogate in its metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
from typing import Any, Callable

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


QWEN_DIM = 4096
QWEN_EXPERT_WIDTH = 1024
QWEN_NUM_EXPERTS = 512
QWEN_TOP_K = 10


@dataclass(frozen=True)
class Case:
    local_tokens: int
    logical_ep: int
    physical_ep: int
    compute_profile: str
    collective: str
    comm_dtype: str
    payload_scale: float
    allgather_full_mib: float
    local_expert_variant: str
    compute_repeats: int
    comm_repeats: int

    @property
    def routed_rows(self) -> int:
        return int(round(self.local_tokens * QWEN_TOP_K * self.payload_scale))

    @property
    def payload_bytes(self) -> int:
        if self.collective == "allgather":
            return int(round(self.allgather_full_mib * 2**20))
        element_size = 2 if self.comm_dtype == "bf16" else 1
        return self.routed_rows * QWEN_DIM * element_size

    @property
    def name(self) -> str:
        scale = str(self.payload_scale).replace(".", "p")
        if self.collective == "allgather":
            communication = (
                f"allgather{self.allgather_full_mib:g}mib_{self.comm_dtype}"
            )
        else:
            communication = f"a2a_{self.comm_dtype}_x{scale}"
        return (
            f"tokens{self.local_tokens}_lep{self.logical_ep}_pep{self.physical_ep}_"
            f"{self.compute_profile}_{communication}_"
            f"{self.local_expert_variant}_c{self.compute_repeats}_"
            f"m{self.comm_repeats}"
        )


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


def summarize(values: list[float]) -> dict[str, float]:
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


@ray.remote(num_cpus=1, num_gpus=1)
class QwenInterferenceRank:
    def __init__(self, rank: int, node_slot: int, local_slot: int) -> None:
        import torch

        self.rank = rank
        self.node_slot = node_slot
        self.local_slot = local_slot
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
            "local_slot": self.local_slot,
            "node_id": self.node_id,
            "node_ip": self.node_ip,
            "hostname": socket.gethostname(),
            "ray_gpu_ids": [str(value) for value in ray.get_gpu_ids()],
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
        world_size: int,
        master_addr: str,
        master_port: int,
        case_dicts: list[dict[str, Any]],
        warmups: int,
        iterations: int,
        compile_backend: str,
        gate_sleep_cycles: int,
        profile_dir: str | None,
    ) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(0)
        os.environ.setdefault(
            "TORCHINDUCTOR_CACHE_DIR", f"/tmp/qwen397b_inductor_rank{self.rank}"
        )
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=self.rank,
            world_size=world_size,
            device_id=torch.device("cuda", 0),
            timeout=timedelta(minutes=30),
        )
        control_group = None
        ep_groups: dict[int, Any] = {}
        try:
            all_ranks = list(range(world_size))
            control_group = dist.new_group(ranks=all_ranks, backend="gloo")
            for group_size in sorted({case["physical_ep"] for case in case_dicts}):
                if world_size % group_size:
                    raise ValueError(
                        f"physical EP {group_size} must divide world size {world_size}"
                    )
                for first in range(0, world_size, group_size):
                    ranks = list(range(first, first + group_size))
                    group = dist.new_group(ranks=ranks, backend="nccl")
                    if self.rank in ranks:
                        ep_groups[group_size] = group

            # Production executors keep a stable stream set for the process
            # lifetime. Reusing it also prevents case order from changing the
            # CUDA hardware-work-queue/connection assignment.
            streams = (
                torch.cuda.Stream(priority=0),
                torch.cuda.Stream(priority=0),
                torch.cuda.Stream(priority=-1),
            )
            results: list[dict[str, Any]] = []
            for case_dict in case_dicts:
                case = Case(**case_dict)
                result = self._run_case(
                    case,
                    ep_groups[case.physical_ep],
                    control_group,
                    warmups,
                    iterations,
                    compile_backend,
                    gate_sleep_cycles,
                    streams,
                    profile_dir,
                )
                results.append(result)
                dist.barrier(group=control_group)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return {"metadata": self.metadata(), "cases": results}
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def _local_expert_shape(self, case: Case) -> tuple[int, int]:
        routed_per_expert = (
            case.logical_ep * case.local_tokens * QWEN_TOP_K / QWEN_NUM_EXPERTS
        )
        if not routed_per_expert.is_integer():
            raise ValueError(f"non-integral balanced tokens/expert: {routed_per_expert}")
        tokens_per_expert = int(routed_per_expert)
        if QWEN_NUM_EXPERTS % case.logical_ep == 0:
            return QWEN_NUM_EXPERTS // case.logical_ep, tokens_per_expert
        low = QWEN_NUM_EXPERTS // case.logical_ep
        if case.local_expert_variant == "typical":
            return low, tokens_per_expert
        if case.local_expert_variant == "tail":
            return low + 1, tokens_per_expert
        raise ValueError(
            "Non-divisible logical EP requires local_expert_variant=typical or tail"
        )

    def _make_compute(
        self, case: Case, compile_backend: str
    ) -> tuple[Callable[[], Any], dict[str, Any], list[Any]]:
        torch = self.torch
        dtype = torch.bfloat16
        owned: list[Any] = []

        composite_profiles = {
            # Backward traverses MoE before the token mixer. These profiles are
            # measured lower bounds on a full production layer: they include
            # the exact major projection GEMMs but intentionally exclude the
            # Gated-DeltaNet/attention core and small kernels.
            "route_expert_linear_layer": (
                "route_expert",
                "linear_out_projection",
                "linear_z_projection",
                "linear_qkv_projection",
            ),
            "route_expert_full_attention_layer": (
                "route_expert",
                "linear_out_projection",
                "gated_q_projection",
            ),
        }
        if case.compute_profile in composite_profiles:
            launches: list[Callable[[], Any]] = []
            components: list[dict[str, Any]] = []
            for profile in composite_profiles[case.compute_profile]:
                launch, metadata, component_owned = self._make_compute(
                    replace(case, compute_profile=profile), compile_backend
                )
                launches.append(launch)
                components.append(metadata)
                owned.extend(component_owned)

            def launch_composite():
                return tuple(launch() for launch in launches)

            return (
                launch_composite,
                {
                    "kernel_model": case.compute_profile,
                    "shape_fidelity": "major_projection_lower_bound",
                    "components": components,
                    "backward_flops": sum(
                        int(component.get("backward_flops", 0))
                        for component in components
                    ),
                },
                owned,
            )

        def make(shape: tuple[int, ...], value: float = 0.01):
            tensor = torch.full(shape, value, dtype=dtype, device="cuda")
            owned.append(tensor)
            return tensor

        backend = None if compile_backend == "none" else compile_backend

        if case.compute_profile in {"expert", "route_expert"}:
            local_experts, tokens_per_expert = self._local_expert_shape(case)
            rows = local_experts * tokens_per_expert
            routed = make((rows, QWEN_DIM))
            d_out = make((local_experts, tokens_per_expert, QWEN_DIM))
            gate_up = make(
                (local_experts, tokens_per_expert, 2 * QWEN_EXPERT_WIDTH)
            )
            hidden = make((local_experts, tokens_per_expert, QWEN_EXPERT_WIDTH))
            w_gate_up = make(
                (local_experts, 2 * QWEN_EXPERT_WIDTH, QWEN_DIM)
            )
            w_down = make((local_experts, QWEN_DIM, QWEN_EXPERT_WIDTH))
            if case.compute_profile == "route_expert":
                # Deterministic non-identity permutation.  It represents the
                # routed-gradient gather preceding the first expert dW GEMM in
                # the earlier Piper trace.
                route_index = torch.arange(rows - 1, -1, -1, device="cuda")
                owned.append(route_index)
            else:
                route_index = None

            def expert_backward(
                routed_input,
                output_grad,
                saved_gate_up,
                saved_hidden,
                gate_up_weight,
                down_weight,
                index,
            ):
                if index is not None:
                    expert_input = routed_input.index_select(0, index).reshape(
                        local_experts, tokens_per_expert, QWEN_DIM
                    )
                else:
                    expert_input = routed_input.reshape(
                        local_experts, tokens_per_expert, QWEN_DIM
                    )
                d_hidden = torch.bmm(output_grad, down_weight)
                d_w_down = torch.bmm(
                    output_grad.transpose(1, 2), saved_hidden
                )
                gate, up = saved_gate_up.chunk(2, dim=-1)
                sigmoid = torch.sigmoid(gate)
                silu = gate * sigmoid
                d_gate = d_hidden * up * sigmoid * (1 + gate * (1 - sigmoid))
                d_up = d_hidden * silu
                d_gate_up = torch.cat((d_gate, d_up), dim=-1)
                d_input = torch.bmm(d_gate_up, gate_up_weight)
                d_w_gate_up = torch.bmm(
                    d_gate_up.transpose(1, 2), expert_input
                )
                return d_input, d_w_gate_up, d_w_down

            compiled = torch.compile(
                expert_backward,
                backend=backend,
                fullgraph=True,
                dynamic=False,
            ) if backend else expert_backward

            def launch():
                return compiled(
                    routed,
                    d_out,
                    gate_up,
                    hidden,
                    w_gate_up,
                    w_down,
                    route_index,
                )

            metadata = {
                "local_experts": local_experts,
                "tokens_per_expert": tokens_per_expert,
                "compute_routed_rows": rows,
                "backward_flops": 12 * rows * QWEN_DIM * QWEN_EXPERT_WIDTH,
                "kernel_model": "packed_gate_up_and_down_backward",
                "includes_route_gather": route_index is not None,
            }
            return launch, metadata, owned

        # (input width, output width, fidelity note). Qwen3.5-397B uses
        # Gated-DeltaNet in three out of every four layers. Its QKV and Z
        # parameters are separate in the reference model even though their
        # output widths sum to 20,480. Keep the original concatenated profile
        # for comparison while providing exact production parameter shapes.
        projection_shapes = {
            "linear_projection": (
                4_096,
                20_480,
                "fused_qkv_plus_z_stress_surrogate",
            ),
            "linear_qkv_projection": (4_096, 12_288, "exact_gated_deltanet_qkv"),
            "linear_z_projection": (4_096, 8_192, "exact_gated_deltanet_z"),
            "linear_out_projection": (8_192, 4_096, "exact_gated_deltanet_out"),
            "gated_q_projection": (4_096, 16_384, "exact_full_attention_q_gate"),
            "full_o_projection": (8_192, 4_096, "exact_full_attention_out"),
        }
        if case.compute_profile not in projection_shapes:
            raise ValueError(f"unknown compute profile {case.compute_profile!r}")
        input_width, output_width, fidelity = projection_shapes[case.compute_profile]
        x = make((case.local_tokens, input_width))
        d_y = make((case.local_tokens, output_width))
        weight = make((output_width, input_width))

        def projection_backward(input_tensor, output_grad, projection_weight):
            d_input = torch.mm(output_grad, projection_weight)
            d_weight = torch.mm(output_grad.transpose(0, 1), input_tensor)
            return d_input, d_weight

        compiled = torch.compile(
            projection_backward,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        ) if backend else projection_backward

        def launch():
            return compiled(x, d_y, weight)

        metadata = {
            "projection_input_width": input_width,
            "projection_output_width": output_width,
            # Kept for compatibility with the earlier result schema.
            "projection_width": output_width,
            "backward_flops": (
                4 * case.local_tokens * input_width * output_width
            ),
            "kernel_model": f"{case.compute_profile}_dgrad_wgrad",
            "shape_fidelity": fidelity,
        }
        return launch, metadata, owned

    def _run_case(
        self,
        case: Case,
        ep_group: Any,
        control_group: Any,
        warmups: int,
        iterations: int,
        compile_backend: str,
        gate_sleep_cycles: int,
        streams: tuple[Any, Any, Any],
        profile_dir: str | None,
    ) -> dict[str, Any]:
        torch = self.torch
        import torch.distributed as dist

        comm_dtype = torch.bfloat16 if case.comm_dtype == "bf16" else torch.uint8
        element_size = 2 if case.comm_dtype == "bf16" else 1
        if case.payload_bytes % element_size:
            raise ValueError(
                f"Payload {case.payload_bytes} bytes is not aligned to {case.comm_dtype}"
            )
        full_numel = case.payload_bytes // element_size
        if full_numel % case.physical_ep:
            raise ValueError(
                f"{case.collective} numel {full_numel} is not divisible by "
                f"group size {case.physical_ep}"
            )
        if case.collective == "a2a":
            comm_input = torch.full(
                (full_numel,),
                self.rank % 127,
                dtype=comm_dtype,
                device="cuda",
            )
            comm_output = torch.empty_like(comm_input)
            validation_chunk_numel = full_numel // case.physical_ep

            def launch_comm() -> None:
                dist.all_to_all_single(comm_output, comm_input, group=ep_group)

        elif case.collective == "allgather":
            shard_numel = full_numel // case.physical_ep
            comm_input = torch.full(
                (shard_numel,),
                self.rank % 127,
                dtype=comm_dtype,
                device="cuda",
            )
            comm_output = torch.empty(
                (full_numel,), dtype=comm_dtype, device="cuda"
            )
            validation_chunk_numel = shard_numel

            def launch_comm() -> None:
                dist.all_gather_into_tensor(
                    comm_output, comm_input, group=ep_group
                )

        else:
            raise ValueError(f"Unknown collective {case.collective!r}")

        compute, compute_meta, compute_owned = self._make_compute(
            case, compile_backend
        )
        compute_stream, comm_stream, control_stream = streams
        collective_label = case.collective
        comm_solo_mode = f"{collective_label}_solo"
        concurrent_comm_first = f"concurrent_{collective_label}_first"
        serialized_compute_comm = f"serialized_compute_{collective_label}"
        serialized_comm_compute = f"serialized_{collective_label}_compute"
        comm_metric = f"{collective_label}_ms"

        # Compile and initialize communicators before timed work.
        with torch.cuda.stream(compute_stream):
            compile_start = time.perf_counter()
            compile_output = compute()
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - compile_start
        del compile_output
        with torch.cuda.stream(comm_stream):
            launch_comm()
        torch.cuda.synchronize()
        dist.barrier(group=control_group)

        modes = [
            "compute_solo",
            comm_solo_mode,
            "concurrent_compute_first",
            concurrent_comm_first,
            serialized_compute_comm,
            serialized_comm_compute,
        ]

        def run_once(mode: str) -> dict[str, float]:
            gate = torch.cuda.Event(enable_timing=True)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            comm_start = torch.cuda.Event(enable_timing=True)
            comm_end = torch.cuda.Event(enable_timing=True)
            compute_results: list[Any] = []

            dist.barrier(group=control_group)
            # A wait on an event that has never been recorded is a no-op. Put
            # a short delay and the gate record onto the control stream first,
            # then enqueue both work streams behind that known future record.
            # The delay ends before either measured operation begins, so its
            # one-CTA sleep kernel cannot contend with the measured work.
            with torch.cuda.stream(control_stream):
                torch.cuda._sleep(gate_sleep_cycles)
                gate.record()
            if mode == "compute_solo":
                compute_stream.wait_event(gate)
                with torch.cuda.stream(compute_stream):
                    compute_start.record()
                    for _ in range(case.compute_repeats):
                        compute_results.append(compute())
                    compute_end.record()
            elif mode == comm_solo_mode:
                comm_stream.wait_event(gate)
                with torch.cuda.stream(comm_stream):
                    comm_start.record()
                    for _ in range(case.comm_repeats):
                        launch_comm()
                    comm_end.record()
            elif mode.startswith("concurrent"):
                compute_stream.wait_event(gate)
                comm_stream.wait_event(gate)

                def enqueue_compute():
                    with torch.cuda.stream(compute_stream):
                        compute_start.record()
                        for _ in range(case.compute_repeats):
                            compute_results.append(compute())
                        compute_end.record()

                def enqueue_comm():
                    with torch.cuda.stream(comm_stream):
                        comm_start.record()
                        for _ in range(case.comm_repeats):
                            launch_comm()
                        comm_end.record()

                if mode == "concurrent_compute_first":
                    enqueue_compute()
                    enqueue_comm()
                else:
                    enqueue_comm()
                    enqueue_compute()
            elif mode == serialized_compute_comm:
                compute_stream.wait_event(gate)
                with torch.cuda.stream(compute_stream):
                    compute_start.record()
                    for _ in range(case.compute_repeats):
                        compute_results.append(compute())
                    compute_end.record()
                comm_stream.wait_event(compute_end)
                with torch.cuda.stream(comm_stream):
                    comm_start.record()
                    for _ in range(case.comm_repeats):
                        launch_comm()
                    comm_end.record()
            elif mode == serialized_comm_compute:
                comm_stream.wait_event(gate)
                with torch.cuda.stream(comm_stream):
                    comm_start.record()
                    for _ in range(case.comm_repeats):
                        launch_comm()
                    comm_end.record()
                compute_stream.wait_event(comm_end)
                with torch.cuda.stream(compute_stream):
                    compute_start.record()
                    for _ in range(case.compute_repeats):
                        compute_results.append(compute())
                    compute_end.record()
            else:
                raise ValueError(mode)

            if mode != comm_solo_mode:
                compute_end.synchronize()
            if mode != "compute_solo":
                comm_end.synchronize()
            dist.barrier(group=control_group)

            sample: dict[str, float] = {}
            starts: list[float] = []
            ends: list[float] = []
            if mode != comm_solo_mode:
                sample["compute_ms"] = compute_start.elapsed_time(compute_end)
                sample["compute_gate_to_start_ms"] = gate.elapsed_time(compute_start)
                sample["compute_gate_to_end_ms"] = gate.elapsed_time(compute_end)
                starts.append(sample["compute_gate_to_start_ms"])
                ends.append(sample["compute_gate_to_end_ms"])
            if mode != "compute_solo":
                sample[comm_metric] = comm_start.elapsed_time(comm_end)
                sample[f"{collective_label}_gate_to_start_ms"] = gate.elapsed_time(
                    comm_start
                )
                sample[f"{collective_label}_gate_to_end_ms"] = gate.elapsed_time(
                    comm_end
                )
                starts.append(sample[f"{collective_label}_gate_to_start_ms"])
                ends.append(sample[f"{collective_label}_gate_to_end_ms"])
            sample["pair_ms"] = max(ends) - min(starts)
            if len(starts) == 2:
                sample["start_skew_ms"] = abs(starts[0] - starts[1])
            # Keep results live until their kernels complete; synchronization
            # above makes deletion and allocator reuse safe.
            compute_results.clear()
            return sample

        for _ in range(warmups):
            for mode in modes:
                run_once(mode)

        samples: dict[str, list[dict[str, float]]] = {mode: [] for mode in modes}
        for iteration in range(iterations):
            order = modes if iteration % 2 == 0 else list(reversed(modes))
            for mode in order:
                samples[mode].append(run_once(mode))

        trace_path = None
        if profile_dir and self.rank == 0:
            from torch.profiler import ProfilerActivity, profile, record_function

            target = Path(profile_dir)
            target.mkdir(parents=True, exist_ok=True)
            trace_path = target / f"{case.name}_rank0.json"
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                acc_events=True,
            ) as profiler:
                with record_function(f"qwen397b_{case.name}_concurrent"):
                    run_once(concurrent_comm_first)
            profiler.export_chrome_trace(str(trace_path))
        elif profile_dir:
            # Rank zero profiles, but every peer must execute its collective.
            run_once(concurrent_comm_first)

        # Validate one received chunk without copying the multi-GiB tensor.
        source = (self.rank + 1) % case.physical_ep
        group_first = self.rank // case.physical_ep * case.physical_ep
        expected_source_rank = group_first + source
        observed = int(comm_output[source * validation_chunk_numel].item())
        expected = expected_source_rank % 127
        if observed != expected:
            raise RuntimeError(
                f"rank {self.rank} {case.collective} validation failed: "
                f"{observed} != {expected}"
            )

        flattened = {
            mode: {
                metric: summarize([sample[metric] for sample in mode_samples])
                for metric in mode_samples[0]
            }
            for mode, mode_samples in samples.items()
        }
        return {
            "case": asdict(case),
            "case_name": case.name,
            "collective": case.collective,
            "payload_bytes": case.payload_bytes,
            "payload_mib": case.payload_bytes / 2**20,
            "input_shard_bytes": comm_input.numel() * element_size,
            "input_shard_mib": comm_input.numel() * element_size / 2**20,
            "compute": compute_meta,
            "compile_s": compile_s,
            "trace_path": str(trace_path) if trace_path else None,
            "samples": samples,
            "summaries": flattened,
            "validation": {"observed": observed, "expected": expected},
        }


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_csv_strings(value: str) -> list[str]:
    return [item for item in value.split(",") if item]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="10.12.250.179")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument(
        "--local-tokens",
        default="32768",
        help="Comma-separated tokens per EP source rank",
    )
    parser.add_argument("--logical-ep", default="16")
    parser.add_argument("--physical-ep", default="16")
    parser.add_argument(
        "--compute-profiles",
        default="route_expert,expert,linear_projection,gated_q_projection",
    )
    parser.add_argument(
        "--collective",
        default="a2a",
        choices=["a2a", "allgather"],
        help="Communication operation overlapped with backward compute",
    )
    parser.add_argument("--comm-dtypes", default="bf16")
    parser.add_argument("--payload-scales", default="1")
    parser.add_argument(
        "--allgather-full-payload-mib",
        default="256,512,1024,2048",
        help="Full gathered payload per AG, before division across ranks",
    )
    parser.add_argument(
        "--local-expert-variants",
        default="typical",
        help="typical or tail; tail matters when experts are not divisible by EP",
    )
    parser.add_argument(
        "--compute-repeats",
        type=int,
        default=1,
        help="Back-to-back compute launches inside each measured sequence",
    )
    parser.add_argument(
        "--comm-repeats",
        "--a2a-repeats",
        dest="comm_repeats",
        type=int,
        default=1,
        help="Back-to-back collectives inside each measured communication sequence",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--compile-backend", default="inductor", choices=["inductor", "none"])
    parser.add_argument(
        "--gate-sleep-cycles",
        type=int,
        default=20_000_000,
        help="Future-event delay used to enqueue both streams before release",
    )
    parser.add_argument(
        "--expected-nvlink-domain",
        help="Require every selected hostname to share this prefix before -T",
    )
    parser.add_argument("--profile-dir")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def aggregate_cases(rank_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    case_names = [case["case_name"] for case in rank_results[0]["cases"]]
    aggregated: list[dict[str, Any]] = []
    for case_index, case_name in enumerate(case_names):
        rank_cases = [result["cases"][case_index] for result in rank_results]
        if any(case["case_name"] != case_name for case in rank_cases):
            raise RuntimeError("Rank case ordering differs")
        modes = rank_cases[0]["samples"]
        mode_summaries: dict[str, dict[str, Any]] = {}
        for mode in modes:
            metrics = rank_cases[0]["samples"][mode][0]
            mode_summaries[mode] = {}
            for metric in metrics:
                rank_medians = [
                    statistics.median(
                        sample[metric] for sample in case["samples"][mode]
                    )
                    for case in rank_cases
                ]
                all_samples = [
                    sample[metric]
                    for case in rank_cases
                    for sample in case["samples"][mode]
                ]
                mode_summaries[mode][metric] = {
                    "rank_medians": rank_medians,
                    "median_rank_median": statistics.median(rank_medians),
                    "max_rank_median": max(rank_medians),
                    "all_samples": summarize(all_samples),
                }

        collective = rank_cases[0]["collective"]
        comm_solo_mode = f"{collective}_solo"
        concurrent_comm_first = f"concurrent_{collective}_first"
        comm_metric = f"{collective}_ms"
        slowdown_metric = f"{collective}_slowdown"
        compute_solo = mode_summaries["compute_solo"]["compute_ms"][
            "rank_medians"
        ]
        comm_solo = mode_summaries[comm_solo_mode][comm_metric]["rank_medians"]
        derived: dict[str, Any] = {}
        for concurrent in ("concurrent_compute_first", concurrent_comm_first):
            compute_concurrent = mode_summaries[concurrent]["compute_ms"][
                "rank_medians"
            ]
            comm_concurrent = mode_summaries[concurrent][comm_metric][
                "rank_medians"
            ]
            pair_concurrent = mode_summaries[concurrent]["pair_ms"]["rank_medians"]
            rank_metrics = []
            for rank in range(len(rank_cases)):
                ideal_pair = max(compute_solo[rank], comm_solo[rank])
                rank_metrics.append(
                    {
                        "rank": rank,
                        "compute_slowdown": compute_concurrent[rank]
                        / compute_solo[rank],
                        slowdown_metric: comm_concurrent[rank] / comm_solo[rank],
                        "pair_overhead_vs_interference_free": pair_concurrent[rank]
                        / ideal_pair,
                        "overlap_efficiency": (
                            compute_solo[rank]
                            + comm_solo[rank]
                            - pair_concurrent[rank]
                        )
                        / min(compute_solo[rank], comm_solo[rank]),
                    }
                )
            derived[concurrent] = {
                "per_rank": rank_metrics,
                "compute_slowdown": summarize(
                    [row["compute_slowdown"] for row in rank_metrics]
                ),
                slowdown_metric: summarize(
                    [row[slowdown_metric] for row in rank_metrics]
                ),
                "pair_overhead_vs_interference_free": summarize(
                    [
                        row["pair_overhead_vs_interference_free"]
                        for row in rank_metrics
                    ]
                ),
                "overlap_efficiency": summarize(
                    [row["overlap_efficiency"] for row in rank_metrics]
                ),
            }
        aggregated.append(
            {
                "case_name": case_name,
                "case": rank_cases[0]["case"],
                "collective": collective,
                "payload_bytes": rank_cases[0]["payload_bytes"],
                "payload_mib": rank_cases[0]["payload_mib"],
                "input_shard_bytes": rank_cases[0]["input_shard_bytes"],
                "input_shard_mib": rank_cases[0]["input_shard_mib"],
                "compute": rank_cases[0]["compute"],
                "compile_s": summarize([case["compile_s"] for case in rank_cases]),
                "modes": mode_summaries,
                "derived": derived,
            }
        )
    return aggregated


def main() -> None:
    args = parse_args()
    if args.world_size != args.nodes * args.gpus_per_node:
        raise SystemExit("world size must equal nodes * gpus per node")
    if args.compute_repeats <= 0 or args.comm_repeats <= 0:
        raise SystemExit("compute-repeats and comm-repeats must be positive")
    local_tokens = parse_csv_ints(args.local_tokens)
    logical_eps = parse_csv_ints(args.logical_ep)
    physical_eps = parse_csv_ints(args.physical_ep)
    compute_profiles = parse_csv_strings(args.compute_profiles)
    comm_dtypes = parse_csv_strings(args.comm_dtypes)
    payload_scales = parse_csv_floats(args.payload_scales)
    allgather_full_mibs = parse_csv_floats(args.allgather_full_payload_mib)
    variants = parse_csv_strings(args.local_expert_variants)
    communication_values = (
        allgather_full_mibs if args.collective == "allgather" else payload_scales
    )
    if any(value <= 0 for value in communication_values):
        raise SystemExit("Communication payload sizes and scales must be positive")
    cases = [
        Case(
            local_tokens=tokens,
            logical_ep=logical_ep,
            physical_ep=physical_ep,
            compute_profile=profile,
            collective=args.collective,
            comm_dtype=dtype,
            payload_scale=value if args.collective == "a2a" else 1.0,
            allgather_full_mib=(
                value if args.collective == "allgather" else 0.0
            ),
            local_expert_variant=variant,
            compute_repeats=args.compute_repeats,
            comm_repeats=args.comm_repeats,
        )
        for tokens in local_tokens
        for logical_ep in logical_eps
        for physical_ep in physical_eps
        for profile in compute_profiles
        for dtype in comm_dtypes
        for value in communication_values
        for variant in variants
        if QWEN_NUM_EXPERTS % logical_ep == 0
        or variant in {"typical", "tail"}
    ]
    if not cases:
        raise SystemExit("No cases selected")

    forwarded_env = {
        name: os.environ[name]
        for name in (
            "CUDA_DEVICE_MAX_CONNECTIONS",
            "NCCL_ALGO",
            "NCCL_CUMEM_ENABLE",
            "NCCL_DEBUG",
            "NCCL_DEBUG_FILE",
            "NCCL_DEBUG_SUBSYS",
            "NCCL_MAX_CTAS",
            "NCCL_MIN_CTAS",
            "NCCL_MNNVL_ENABLE",
            "NCCL_PROTO",
        )
        if os.environ.get(name)
    }
    ray.init(
        address=f"{args.address}:{args.port}",
        namespace="qwen397b_collective_backward_interference",
        runtime_env={"env_vars": forwarded_env} if forwarded_env else None,
    )
    pg = placement_group(
        [
            {"CPU": args.gpus_per_node, "GPU": args.gpus_per_node}
            for _ in range(args.nodes)
        ],
        strategy="STRICT_SPREAD",
    )
    actors: list[Any] = []
    try:
        ray.get(pg.ready(), timeout=600)
        for node_slot in range(args.nodes):
            for local_slot in range(args.gpus_per_node):
                rank = node_slot * args.gpus_per_node + local_slot
                actors.append(
                    QwenInterferenceRank.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=node_slot,
                        )
                    ).remote(rank, node_slot, local_slot)
                )
        placements = ray.get([actor.metadata.remote() for actor in actors])
        node_counts = Counter(item["node_id"] for item in placements)
        if sorted(node_counts.values()) != [args.gpus_per_node] * args.nodes:
            raise RuntimeError(f"Expected four ranks/node: {node_counts}, {placements}")
        gpu_pairs = {
            (item["node_id"], tuple(item["ray_gpu_ids"])) for item in placements
        }
        if len(gpu_pairs) != args.world_size:
            raise RuntimeError(f"GPU assignment is not unique: {placements}")
        nvlink_domains = {
            item["hostname"].split("-T", 1)[0] for item in placements
        }
        if args.expected_nvlink_domain and nvlink_domains != {
            args.expected_nvlink_domain
        }:
            raise RuntimeError(
                f"Expected NVLink domain {args.expected_nvlink_domain}, "
                f"observed {sorted(nvlink_domains)} in {placements}"
            )

        master_addr, master_port = ray.get(actors[0].rendezvous.remote())
        futures = [
            actor.run.remote(
                args.world_size,
                master_addr,
                master_port,
                [asdict(case) for case in cases],
                args.warmups,
                args.iterations,
                args.compile_backend,
                args.gate_sleep_cycles,
                str(Path(args.profile_dir).resolve()) if args.profile_dir else None,
            )
            for actor in actors
        ]
        started = time.time()
        rank_results = ray.get(futures)
        finished = time.time()
        output = {
            "schema_version": 1,
            "command_args": vars(args),
            "qwen_config": {
                "model": "Qwen3.5-397B-A17B",
                "dim": QWEN_DIM,
                "expert_width": QWEN_EXPERT_WIDTH,
                "num_experts": QWEN_NUM_EXPERTS,
                "top_k": QWEN_TOP_K,
            },
            "environment": {
                "forwarded_env": forwarded_env,
                "started_unix_s": started,
                "finished_unix_s": finished,
                "elapsed_s": finished - started,
            },
            "placements": placements,
            "rank_results": rank_results,
            "aggregate": aggregate_cases(rank_results),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        for case in output["aggregate"]:
            collective = case["collective"]
            comm_solo_mode = f"{collective}_solo"
            concurrent_comm_first = f"concurrent_{collective}_first"
            comm_metric = f"{collective}_ms"
            slowdown_metric = f"{collective}_slowdown"
            concurrent = case["derived"][concurrent_comm_first]
            print(
                json.dumps(
                    {
                        "case": case["case_name"],
                        "collective": collective,
                        "payload_mib": case["payload_mib"],
                        "input_shard_mib": case["input_shard_mib"],
                        "compute_solo_ms": case["modes"]["compute_solo"]
                        ["compute_ms"]["median_rank_median"],
                        "comm_solo_ms": case["modes"][comm_solo_mode][comm_metric]
                        ["median_rank_median"],
                        "compute_slowdown_median": concurrent[
                            "compute_slowdown"
                        ]["median"],
                        "comm_slowdown_median": concurrent[slowdown_metric][
                            "median"
                        ],
                        "pair_overhead_median": concurrent[
                            "pair_overhead_vs_interference_free"
                        ]["median"],
                        "overlap_efficiency_median": concurrent[
                            "overlap_efficiency"
                        ]["median"],
                    },
                    sort_keys=True,
                )
            )
        print(f"Wrote {output_path}")
    finally:
        for actor in actors:
            ray.kill(actor, no_restart=True)
        remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
