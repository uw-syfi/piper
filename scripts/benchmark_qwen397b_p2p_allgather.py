#!/usr/bin/env python3
"""Benchmark Qwen3.5-397B pipeline P2P x FSDP all-gather interference.

The intended topology is PP=2 x DP=8 on four 4-GPU GB200 nodes split across
two NVLink domains.  Each DP group occupies one 8-GPU domain, and corresponding
DP ranks define eight possible cross-domain PP pairs:

    domain 0 / PP stage 0: ranks  0..7
    domain 1 / PP stage 1: ranks  8..15
    possible PP pairs: (0, 8), (1, 9), ..., (7, 15)

By default, only pair index 0 (ranks 0 and 8) launches P2P while all 16 ranks
continue to participate in their within-domain all-gather and global control
groups.  ``--p2p-pair-mode all`` activates all eight possible pairs.  Ray
placement reserves one GPU per actor and pins four actors to each explicitly
named node, preventing benchmark matrices from oversubscribing a GPU.

The production Qwen3.5-397B-A17B shape is loaded from
``examples.models.qwen3._QWEN35_397B``.  The model's annotated pipeline
boundary is the residual hidden state, so microbatch=1 and sequence=32768
produce a BF16 P2P tensor of ``1 * 32768 * 4096`` elements (256 MiB).  Named
all-gather payloads are full parameter sizes derived from the same model
configuration.  For an 8-rank all-gather, each rank contributes one eighth of
the named full payload.

CUDA events on the operation's launch stream measure P2P duration, each
individual all-gather, the all-gather sequence, and the local pair makespan.
The JSON output preserves rank-local samples.  Its aggregate and companion CSV
use aligned samples and distributed critical durations (the maximum rank) when
calculating median slowdowns.

``--dry-run`` resolves all model shapes, cases, rank maps, and allocation
estimates without importing Ray, touching NCCL, reserving GPUs, or writing
output files.
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


@dataclass(frozen=True)
class Payload:
    label: str
    numel: int
    source: str

    @property
    def bytes(self) -> int:
        return self.numel * DTYPE_BYTES

    @property
    def mib(self) -> float:
        return self.bytes / 2**20


@dataclass(frozen=True)
class Topology:
    world_size: int
    nodes: int
    gpus_per_node: int
    node_domain_map: tuple[int, ...]
    domain_ids: tuple[int, int]
    allgather_groups: tuple[tuple[int, ...], tuple[int, ...]]
    p2p_pairs: tuple[tuple[int, int], ...]

    def domain_for_rank(self, rank: int) -> int:
        return self.node_domain_map[rank // self.gpus_per_node]

    def allgather_group_for(self, rank: int) -> tuple[int, ...]:
        return next(group for group in self.allgather_groups if rank in group)

    def p2p_pair_for(self, rank: int) -> tuple[int, int] | None:
        return next((pair for pair in self.p2p_pairs if rank in pair), None)


@dataclass(frozen=True)
class Case:
    p2p_label: str
    p2p_numel: int
    p2p_direction: str
    allgather_label: str
    allgather_full_numel: int
    allgather_count: int

    @property
    def name(self) -> str:
        return (
            f"p2p_{self.p2p_label}_{self.p2p_direction}_"
            f"ag_{self.allgather_label}_x{self.allgather_count}"
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
    required = {
        "dim",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "moe_inter_dim",
        "num_experts",
        "num_shared_experts",
        "vocab_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{MODEL_SYMBOL} is missing required fields: {missing}")
    return config


def model_p2p_payloads(
    config: dict[str, Any], microbatch_size: int, seq_len: int
) -> dict[str, Payload]:
    # PiperQwen3Model only annotates PP boundaries around complete transformer
    # stages.  Consequently the transmitted activation and its backward
    # gradient have the same residual-stream shape [microbatch, seq, dim].
    hidden_numel = microbatch_size * seq_len * int(config["dim"])
    return {
        "hidden": Payload(
            "hidden",
            hidden_numel,
            (
                "PiperQwen3Model PP boundary [microbatch, sequence, dim]; "
                "also the backward gradient shape"
            ),
        )
    }


def model_allgather_payloads(config: dict[str, Any]) -> dict[str, Payload]:
    """Return full BF16 parameter sizes before DP sharding.

    The exact production Qwen3.5 full-attention shape uses a gated Q projection
    with twice the normal Q output width.  The local model proxy preserves the
    model/MoE tensor dimensions but does not implement the hybrid backbone, so
    labels explicitly identify production-derived projections.
    """

    dim = int(config["dim"])
    head_dim = int(config["head_dim"])
    q_width = int(config["n_heads"]) * head_dim
    kv_width = int(config["n_kv_heads"]) * head_dim
    expert_width = int(config["moe_inter_dim"])
    experts = int(config["num_experts"])
    shared_experts = int(config["num_shared_experts"])
    vocab = int(config["vocab_size"])
    key_width = 16 * 128
    value_width = 64 * 128
    conv_width = 2 * key_width + value_width

    common_nonrouted = (
        shared_experts * 2 * dim * expert_width
        + shared_experts * dim * expert_width
        + dim * experts
        + dim  # shared-expert gate
        + 2 * dim  # input and post-attention RMSNorm
    )
    full_attention = (
        q_width * dim
        + 2 * dim * kv_width
        + dim * 2 * q_width
        + 2 * head_dim  # Q/K norm
    )
    linear_attention = (
        dim * value_width  # out projection
        + 128  # gated RMSNorm
        + 64  # A_log
        + 64  # dt_bias
        + 64 * dim  # in_proj_b
        + 64 * dim  # in_proj_a
        + value_width * dim  # in_proj_z
        + conv_width * 4  # depthwise conv1d
        + conv_width * dim  # in_proj_qkv
    )
    routed_down_all = experts * dim * expert_width
    routed_gate_up_all = experts * 2 * dim * expert_width

    shapes = {
        "norm": (
            dim,
            "one residual RMSNorm vector",
        ),
        "attention_k_or_v": (
            dim * kv_width,
            "one K or V projection [dim, n_kv_heads * head_dim]",
        ),
        "router": (
            dim * experts,
            "MoE router [dim, num_experts]",
        ),
        "shared_expert_down": (
            shared_experts * dim * expert_width,
            "all shared-expert down projections",
        ),
        "shared_expert_gate_up": (
            shared_experts * 2 * dim * expert_width,
            "all shared-expert fused gate/up projections",
        ),
        "routed_expert_down_one": (
            dim * expert_width,
            "one routed expert down projection",
        ),
        "routed_expert_gate_up_one": (
            2 * dim * expert_width,
            "one routed expert fused gate/up projection",
        ),
        "attention_o": (
            q_width * dim,
            "full-attention O projection [q_width, dim]",
        ),
        "attention_gated_q": (
            dim * 2 * q_width,
            "production gated-Q projection [dim, 2 * q_width]",
        ),
        "full_attention_nonexpert_layer": (
            common_nonrouted + full_attention,
            (
                "all non-expert parameters in one production full-attention "
                "MoE layer"
            ),
        ),
        "linear_attention_nonexpert_layer": (
            common_nonrouted + linear_attention,
            (
                "all non-expert parameters in one production Gated-DeltaNet "
                "MoE layer"
            ),
        ),
        "embedding": (
            vocab * dim,
            "token embedding or untied output projection",
        ),
        "routed_expert_down_all": (
            routed_down_all,
            "all routed-expert down projections in one layer",
        ),
        "routed_expert_gate_up_all": (
            routed_gate_up_all,
            "all routed-expert fused gate/up projections in one layer",
        ),
        "full_attention_whole_layer": (
            common_nonrouted
            + full_attention
            + routed_down_all
            + routed_gate_up_all,
            (
                "conceptual full-parameter all-gather for one production "
                "full-attention MoE layer"
            ),
        ),
        "linear_attention_whole_layer": (
            common_nonrouted
            + linear_attention
            + routed_down_all
            + routed_gate_up_all,
            (
                "conceptual full-parameter all-gather for one production "
                "Gated-DeltaNet MoE layer"
            ),
        ),
    }
    return {
        label: Payload(label, numel, source)
        for label, (numel, source) in shapes.items()
    }


def build_topology(args: argparse.Namespace) -> Topology:
    domains = tuple(parse_csv_ints(args.node_domain_map))
    if len(domains) != args.nodes:
        raise ValueError("node-domain-map must contain one ID per node slot")
    unique_domains = tuple(sorted(set(domains)))
    if len(unique_domains) != 2:
        raise ValueError(
            f"PP2 requires exactly two NVLink domains, got {unique_domains}"
        )
    if args.world_size != args.nodes * args.gpus_per_node:
        raise ValueError("world-size must equal nodes * gpus-per-node")

    groups: list[tuple[int, ...]] = []
    for domain in unique_domains:
        ranks = tuple(
            rank
            for rank in range(args.world_size)
            if domains[rank // args.gpus_per_node] == domain
        )
        groups.append(ranks)
    if any(len(group) != args.dp_size for group in groups):
        raise ValueError(
            f"Each domain must contain DP={args.dp_size} ranks, got "
            f"{[len(group) for group in groups]}"
        )
    possible_pairs = tuple(zip(groups[0], groups[1]))
    if any(
        domains[left // args.gpus_per_node]
        == domains[right // args.gpus_per_node]
        for left, right in possible_pairs
    ):
        raise ValueError("Every P2P pair must cross NVLink domains")
    if args.p2p_pair_mode == "all":
        active_pairs = possible_pairs
    else:
        if not 0 <= args.p2p_pair_index < len(possible_pairs):
            raise ValueError(
                f"p2p-pair-index must be in [0, {len(possible_pairs) - 1}]"
            )
        active_pairs = (possible_pairs[args.p2p_pair_index],)
    return Topology(
        world_size=args.world_size,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        node_domain_map=domains,
        domain_ids=(unique_domains[0], unique_domains[1]),
        allgather_groups=(groups[0], groups[1]),
        p2p_pairs=active_pairs,
    )


def resolve_payloads(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[list[Payload], list[Payload], dict[str, Payload], dict[str, Payload]]:
    known_p2p = model_p2p_payloads(
        config, args.microbatch_size, args.seq_len
    )
    known_ag = model_allgather_payloads(config)
    requested_p2p = parse_csv_strings(args.p2p_shapes)
    requested_ag = parse_csv_strings(args.allgather_shapes)
    unknown_p2p = sorted(set(requested_p2p) - set(known_p2p))
    unknown_ag = sorted(set(requested_ag) - set(known_ag))
    if unknown_p2p:
        raise ValueError(
            f"Unknown P2P shapes {unknown_p2p}; choices={sorted(known_p2p)}"
        )
    if unknown_ag:
        raise ValueError(
            f"Unknown all-gather shapes {unknown_ag}; choices={sorted(known_ag)}"
        )
    p2p = [known_p2p[label] for label in requested_p2p]
    ag = [known_ag[label] for label in requested_ag]

    for index, mib in enumerate(parse_csv_floats(args.p2p_payload_mib)):
        if mib <= 0:
            raise ValueError("custom P2P MiB values must be positive")
        numel = int(mib * 2**20) // DTYPE_BYTES
        p2p.append(
            Payload(f"custom_{mib:g}mib", numel, "explicit CLI P2P payload")
        )
    for index, mib in enumerate(parse_csv_floats(args.allgather_full_payload_mib)):
        if mib <= 0:
            raise ValueError("custom all-gather MiB values must be positive")
        numel = int(mib * 2**20) // DTYPE_BYTES
        ag.append(
            Payload(
                f"custom_{mib:g}mib",
                numel,
                "explicit CLI full all-gather payload before DP sharding",
            )
        )
    if not p2p:
        raise ValueError("At least one P2P payload must be selected")
    if not ag:
        raise ValueError("At least one all-gather payload must be selected")
    return p2p, ag, known_p2p, known_ag


def build_cases(
    args: argparse.Namespace,
    topology: Topology,
    p2p_payloads: list[Payload],
    allgather_payloads: list[Payload],
) -> list[Case]:
    counts = parse_csv_ints(args.allgather_counts)
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("allgather-counts must be positive")
    directions = (
        ["forward", "backward"]
        if args.p2p_directions == "both"
        else [args.p2p_directions]
    )
    cases = [
        Case(
            p2p_label=p2p.label,
            p2p_numel=p2p.numel,
            p2p_direction=direction,
            allgather_label=ag.label,
            allgather_full_numel=ag.numel,
            allgather_count=count,
        )
        for p2p in p2p_payloads
        for direction in directions
        for ag in allgather_payloads
        for count in counts
    ]
    group_size = len(topology.allgather_groups[0])
    invalid = [
        case.name
        for case in cases
        if case.allgather_full_numel % group_size
    ]
    if invalid:
        raise ValueError(
            "Full all-gather numel must divide DP size; invalid cases: "
            + ", ".join(invalid)
        )
    return cases


def benchmark_modes() -> list[str]:
    return [
        "p2p_solo",
        "allgather_solo",
        "concurrent_p2p_first",
        "concurrent_allgather_first",
        "concurrent_one_allgather_then_p2p",
    ]


def balanced_mode_order(modes: list[str], sample_index: int) -> list[str]:
    """Rotate a Williams-style block so each mode occupies every position.

    A simple forward/reverse alternation leaves solo and concurrent modes tied
    to different positions inside each measurement block.  Long communication
    cases showed measurable position-dependent baselines, so use cyclic Latin
    square blocks and reverse every other square.
    """

    if not modes:
        return []
    square = sample_index // len(modes)
    base = modes if square % 2 == 0 else list(reversed(modes))
    shift = sample_index % len(modes)
    return base[shift:] + base[:shift]


def build_preview(
    args: argparse.Namespace,
    config: dict[str, Any],
    topology: Topology,
    cases: list[Case],
    known_p2p: dict[str, Payload],
    known_ag: dict[str, Payload],
) -> dict[str, Any]:
    dp_size = len(topology.allgather_groups[0])
    case_rows = []
    for case in cases:
        p2p_bytes = case.p2p_numel * DTYPE_BYTES
        ag_full_bytes = case.allgather_full_numel * DTYPE_BYTES
        ag_shard_bytes = ag_full_bytes // dp_size
        case_rows.append(
            {
                **asdict(case),
                "p2p_bytes_per_rank": p2p_bytes,
                "p2p_mib_per_rank": p2p_bytes / 2**20,
                "allgather_full_bytes_per_kernel": ag_full_bytes,
                "allgather_full_mib_per_kernel": ag_full_bytes / 2**20,
                "allgather_input_shard_bytes_per_rank": ag_shard_bytes,
                "allgather_input_shard_mib_per_rank": ag_shard_bytes / 2**20,
                "allgather_ring_traffic_mib_per_rank_per_kernel": (
                    ag_full_bytes * (dp_size - 1) / dp_size / 2**20
                ),
                "allgather_sequence_ring_traffic_mib_per_rank": (
                    ag_full_bytes
                    * (dp_size - 1)
                    / dp_size
                    * case.allgather_count
                    / 2**20
                ),
                "estimated_live_tensor_bytes_per_rank": (
                    p2p_bytes + ag_full_bytes + ag_shard_bytes
                ),
                "estimated_live_tensor_gib_per_rank": (
                    p2p_bytes + ag_full_bytes + ag_shard_bytes
                )
                / 2**30,
            }
        )
    return {
        "model": MODEL_NAME,
        "model_symbol": MODEL_SYMBOL,
        "qwen_config": config,
        "training_shape": {
            "microbatch_size": args.microbatch_size,
            "sequence_length": args.seq_len,
            "pipeline_degree": 2,
            "data_parallel_degree": args.dp_size,
        },
        "topology": asdict(topology),
        "p2p_selection": {
            "mode": args.p2p_pair_mode,
            "pair_index": (
                args.p2p_pair_index
                if args.p2p_pair_mode == "single"
                else None
            ),
            "active_pairs": topology.p2p_pairs,
        },
        "known_p2p_payloads": {
            label: asdict(payload)
            | {"bytes": payload.bytes, "mib": payload.mib}
            for label, payload in known_p2p.items()
        },
        "known_allgather_full_payloads": {
            label: asdict(payload)
            | {
                "full_bytes": payload.bytes,
                "full_mib": payload.mib,
                "input_shard_mib_at_requested_dp": payload.mib / args.dp_size,
            }
            for label, payload in known_ag.items()
        },
        "cases": case_rows,
        "measurement": {
            "warmups": args.warmups,
            "iterations_per_trial": args.iterations,
            "trials": args.trials,
            "samples_per_mode": args.iterations * args.trials,
            "modes": benchmark_modes(),
            "gate_sleep_cycles": args.gate_sleep_cycles,
        },
        "queue_order_semantics": {
            "concurrent_p2p_first": "P2P is enqueued before every all-gather",
            "concurrent_allgather_first": (
                "Every sequential all-gather is enqueued before P2P"
            ),
            "concurrent_one_allgather_then_p2p": (
                "One all-gather is enqueued, then P2P, then remaining all-gathers"
            ),
        },
        "allocation_note": (
            "Sequential all-gathers deliberately reuse one input and output "
            "buffer. NCCL communicator ordering prevents concurrent writes, "
            "and the final output is validated."
        ),
    }


class P2PAllGatherRank:
    def __init__(self, rank: int, node_slot: int, local_slot: int) -> None:
        import ray
        import torch

        self.rank = rank
        self.node_slot = node_slot
        self.local_slot = local_slot
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
            "local_slot": self.local_slot,
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
            "nccl_launch_order_implicit": os.environ.get(
                "NCCL_LAUNCH_ORDER_IMPLICIT"
            ),
            "nvidia_smi": smi.stdout.strip(),
        }

    def run(
        self,
        world_size: int,
        master_addr: str,
        master_port: int,
        topology_dict: dict[str, Any],
        case_dicts: list[dict[str, Any]],
        warmups: int,
        iterations: int,
        trials: int,
        gate_sleep_cycles: int,
        profile_dir: str | None,
    ) -> list[dict[str, Any]]:
        import torch
        import torch.distributed as dist

        topology = Topology(
            world_size=topology_dict["world_size"],
            nodes=topology_dict["nodes"],
            gpus_per_node=topology_dict["gpus_per_node"],
            node_domain_map=tuple(topology_dict["node_domain_map"]),
            domain_ids=tuple(topology_dict["domain_ids"]),
            allgather_groups=tuple(
                tuple(group) for group in topology_dict["allgather_groups"]
            ),
            p2p_pairs=tuple(
                tuple(pair) for pair in topology_dict["p2p_pairs"]
            ),
        )
        cases = [Case(**case_dict) for case_dict in case_dicts]
        torch.cuda.set_device(0)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=self.rank,
            world_size=world_size,
            device_id=torch.device("cuda", 0),
            timeout=timedelta(minutes=60),
        )
        try:
            all_ranks = list(range(world_size))
            control_group = dist.new_group(ranks=all_ranks, backend="gloo")
            p2p_group = None
            allgather_group = None
            # All ranks create every communicator in identical global order.
            for pair in topology.p2p_pairs:
                handle = dist.new_group(ranks=list(pair), backend="nccl")
                if self.rank in pair:
                    p2p_group = handle
            for group in topology.allgather_groups:
                handle = dist.new_group(ranks=list(group), backend="nccl")
                if self.rank in group:
                    allgather_group = handle
            if allgather_group is None:
                raise RuntimeError(
                    f"rank {self.rank} lacks an all-gather benchmark group"
                )

            pair = topology.p2p_pair_for(self.rank)
            p2p_active = pair is not None
            ag_ranks = topology.allgather_group_for(self.rank)
            p2p_stream = torch.cuda.Stream(priority=0)
            ag_stream = torch.cuda.Stream(priority=0)
            control_stream = torch.cuda.Stream(priority=-1)
            modes = benchmark_modes()
            rank_results: list[dict[str, Any]] = []

            for case in cases:
                ag_shard_numel = case.allgather_full_numel // len(ag_ranks)
                sender: int | None = None
                receiver: int | None = None
                partner: int | None = None
                is_sender = False
                p2p_buffer = None
                if p2p_active:
                    assert pair is not None
                    if case.p2p_direction == "forward":
                        sender, receiver = pair
                    elif case.p2p_direction == "backward":
                        receiver, sender = pair
                    else:
                        raise ValueError(case.p2p_direction)
                    partner = receiver if self.rank == sender else sender
                    is_sender = self.rank == sender
                    p2p_buffer = torch.full(
                        (case.p2p_numel,),
                        self.rank + 1 if is_sender else -1,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                ag_input = torch.full(
                    (ag_shard_numel,),
                    self.rank + 1,
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                ag_output = torch.empty(
                    (case.allgather_full_numel,),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                torch.cuda.synchronize()
                dist.barrier(group=control_group)

                def run_once(
                    mode: str, trial: int, iteration: int
                ) -> dict[str, Any]:
                    has_p2p = p2p_active and mode != "allgather_solo"
                    has_ag = mode != "p2p_solo"
                    gate = torch.cuda.Event(enable_timing=True)
                    p2p_start = torch.cuda.Event(enable_timing=True)
                    p2p_end = torch.cuda.Event(enable_timing=True)
                    ag_sequence_start = torch.cuda.Event(enable_timing=True)
                    ag_sequence_end = torch.cuda.Event(enable_timing=True)
                    ag_starts = [
                        torch.cuda.Event(enable_timing=True)
                        for _ in range(case.allgather_count)
                    ]
                    ag_ends = [
                        torch.cuda.Event(enable_timing=True)
                        for _ in range(case.allgather_count)
                    ]
                    p2p_work: list[Any] = []
                    ag_work: list[Any] = []
                    ag_launched = 0

                    dist.barrier(group=control_group)
                    # A future device-side event allows every host rank to
                    # enqueue both communicators before either can execute.
                    with torch.cuda.stream(control_stream):
                        torch.cuda._sleep(gate_sleep_cycles)
                        gate.record()

                    def launch_p2p() -> None:
                        nonlocal p2p_work
                        if not p2p_active:
                            return
                        assert p2p_buffer is not None
                        assert partner is not None
                        assert p2p_group is not None
                        p2p_stream.wait_event(gate)
                        with torch.cuda.stream(p2p_stream):
                            p2p_start.record()
                            if is_sender:
                                op = dist.P2POp(
                                    dist.isend,
                                    p2p_buffer,
                                    partner,
                                    group=p2p_group,
                                )
                            else:
                                op = dist.P2POp(
                                    dist.irecv,
                                    p2p_buffer,
                                    partner,
                                    group=p2p_group,
                                )
                            p2p_work = dist.batch_isend_irecv([op])
                            # For NCCL, wait inserts a dependency into the
                            # current CUDA stream. The following event therefore
                            # marks device completion, not host enqueue.
                            for work in p2p_work:
                                work.wait()
                            p2p_end.record()

                    def launch_next_allgather() -> None:
                        nonlocal ag_launched
                        index = ag_launched
                        if index >= case.allgather_count:
                            raise RuntimeError("all all-gathers already launched")
                        ag_stream.wait_event(gate)
                        with torch.cuda.stream(ag_stream):
                            if index == 0:
                                ag_sequence_start.record()
                            ag_starts[index].record()
                            work = dist.all_gather_into_tensor(
                                ag_output,
                                ag_input,
                                group=allgather_group,
                                async_op=True,
                            )
                            # This stream dependency serializes reuse of the
                            # output buffer and makes the per-kernel event span
                            # exclude queueing behind earlier all-gathers.
                            work.wait()
                            ag_ends[index].record()
                            if index == case.allgather_count - 1:
                                ag_sequence_end.record()
                        ag_work.append(work)
                        ag_launched += 1

                    if mode == "p2p_solo":
                        if p2p_active:
                            launch_p2p()
                    elif mode == "allgather_solo":
                        while ag_launched < case.allgather_count:
                            launch_next_allgather()
                    elif mode == "concurrent_p2p_first":
                        if p2p_active:
                            launch_p2p()
                        while ag_launched < case.allgather_count:
                            launch_next_allgather()
                    elif mode == "concurrent_allgather_first":
                        while ag_launched < case.allgather_count:
                            launch_next_allgather()
                        if p2p_active:
                            launch_p2p()
                    elif mode == "concurrent_one_allgather_then_p2p":
                        launch_next_allgather()
                        if p2p_active:
                            launch_p2p()
                        while ag_launched < case.allgather_count:
                            launch_next_allgather()
                    else:
                        raise ValueError(mode)

                    if has_p2p:
                        p2p_end.synchronize()
                    if has_ag:
                        ag_sequence_end.synchronize()
                    # Retain Work references until device completion.
                    del p2p_work, ag_work
                    dist.barrier(group=control_group)

                    starts: list[float] = []
                    ends: list[float] = []
                    sample: dict[str, Any] = {
                        "trial": trial,
                        "iteration": iteration,
                    }
                    if has_p2p:
                        sample["p2p_ms"] = p2p_start.elapsed_time(p2p_end)
                        sample["p2p_gate_to_start_ms"] = gate.elapsed_time(
                            p2p_start
                        )
                        sample["p2p_gate_to_end_ms"] = gate.elapsed_time(p2p_end)
                        starts.append(sample["p2p_gate_to_start_ms"])
                        ends.append(sample["p2p_gate_to_end_ms"])
                    if has_ag:
                        sample["allgather_ms"] = ag_sequence_start.elapsed_time(
                            ag_sequence_end
                        )
                        sample["allgather_gate_to_start_ms"] = gate.elapsed_time(
                            ag_sequence_start
                        )
                        sample["allgather_gate_to_end_ms"] = gate.elapsed_time(
                            ag_sequence_end
                        )
                        sample["allgather_kernel_ms"] = [
                            start.elapsed_time(end)
                            for start, end in zip(ag_starts, ag_ends)
                        ]
                        starts.append(sample["allgather_gate_to_start_ms"])
                        ends.append(sample["allgather_gate_to_end_ms"])
                    sample["pair_ms"] = (
                        max(ends) - min(starts) if starts else 0.0
                    )
                    if len(starts) == 2:
                        sample["start_skew_ms"] = abs(starts[0] - starts[1])
                    return sample

                # Initialize every communicator outside measured samples.
                run_once("p2p_solo", -2, -2)
                run_once("allgather_solo", -2, -1)
                for warmup in range(warmups):
                    order = balanced_mode_order(modes, warmup)
                    for mode in order:
                        run_once(mode, -1, warmup)

                samples: dict[str, list[dict[str, Any]]] = {
                    mode: [] for mode in modes
                }
                for trial in range(trials):
                    for iteration in range(iterations):
                        order = balanced_mode_order(
                            modes, trial * iterations + iteration
                        )
                        for mode in order:
                            samples[mode].append(
                                run_once(mode, trial, iteration)
                            )

                trace_path = None
                if profile_dir:
                    trace_dir = Path(profile_dir)
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    safe_case_name = "".join(
                        character
                        if character.isalnum() or character in "._-"
                        else "_"
                        for character in case.name
                    )
                    trace_path = trace_dir / (
                        f"{safe_case_name}_rank{self.rank}_"
                        f"{socket.gethostname()}.json"
                    )
                    activities = [
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                    with torch.profiler.profile(
                        activities=activities,
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    ) as profiler:
                        for profile_iteration, mode in enumerate(modes):
                            with torch.profiler.record_function(
                                f"benchmark_mode::{mode}"
                            ):
                                run_once(mode, -4, profile_iteration)
                        torch.cuda.synchronize()
                    profiler.export_chrome_trace(str(trace_path))

                # Explicitly validate both operations after measurements.
                if p2p_active and not is_sender:
                    assert p2p_buffer is not None
                    p2p_buffer.fill_(-1)
                run_once("p2p_solo", -3, -2)
                observed_p2p = (
                    float(p2p_buffer[0].item())
                    if p2p_buffer is not None
                    else None
                )
                expected_p2p = (
                    float(sender + 1) if sender is not None else None
                )
                if (
                    p2p_active
                    and not is_sender
                    and observed_p2p != expected_p2p
                ):
                    raise RuntimeError(
                        f"rank {self.rank} P2P validation failed: "
                        f"{observed_p2p} != {expected_p2p}"
                    )
                run_once("allgather_solo", -3, -1)
                observed_ag = [
                    float(ag_output[index * ag_shard_numel].item())
                    for index in range(len(ag_ranks))
                ]
                expected_ag = [float(rank + 1) for rank in ag_ranks]
                if observed_ag != expected_ag:
                    raise RuntimeError(
                        f"rank {self.rank} all-gather validation failed: "
                        f"{observed_ag} != {expected_ag}"
                    )

                summaries: dict[str, dict[str, Any]] = {}
                for mode, mode_samples in samples.items():
                    scalar_metrics = [
                        key
                        for key, value in mode_samples[0].items()
                        if key not in {"trial", "iteration"}
                        and isinstance(value, (int, float))
                    ]
                    summaries[mode] = {
                        metric: summarize(
                            [float(sample[metric]) for sample in mode_samples]
                        )
                        for metric in scalar_metrics
                    }
                    if "allgather_kernel_ms" in mode_samples[0]:
                        summaries[mode]["allgather_kernel_ms_by_index"] = [
                            summarize(
                                [
                                    sample["allgather_kernel_ms"][index]
                                    for sample in mode_samples
                                ]
                            )
                            for index in range(case.allgather_count)
                        ]

                rank_results.append(
                    {
                        "case": asdict(case),
                        "metadata": self.metadata(),
                        "p2p_pair": pair,
                        "p2p_active": p2p_active,
                        "p2p_role": (
                            "sender"
                            if p2p_active and is_sender
                            else "receiver"
                            if p2p_active
                            else "inactive"
                        ),
                        "p2p_partner": partner,
                        "allgather_ranks": ag_ranks,
                        "trace_path": str(trace_path) if trace_path else None,
                        "samples": samples,
                        "summaries": summaries,
                        "validation": {
                            "p2p_observed": observed_p2p,
                            "p2p_expected": expected_p2p,
                            "p2p_checked_on_receiver": (
                                p2p_active and not is_sender
                            ),
                            "allgather_observed": observed_ag,
                            "allgather_expected": expected_ag,
                        },
                    }
                )
                del p2p_buffer, ag_input, ag_output
                torch.cuda.empty_cache()
                dist.barrier(group=control_group)

            dist.barrier(group=control_group)
            return rank_results
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def aggregate_case(
    case: Case,
    rank_results: list[dict[str, Any]],
    topology: Topology,
) -> dict[str, Any]:
    modes = benchmark_modes()
    sample_count = len(rank_results[0]["samples"]["p2p_solo"])
    if any(
        len(result["samples"][mode]) != sample_count
        for result in rank_results
        for mode in modes
    ):
        raise ValueError("rank sample counts are not aligned")
    active_rank_results = [
        result for result in rank_results if result["p2p_active"]
    ]
    if len(active_rank_results) != 2 * len(topology.p2p_pairs):
        raise ValueError(
            "Each active P2P pair must have exactly two endpoint results"
        )

    mode_summaries: dict[str, Any] = {}
    for mode in modes:
        scalar_metrics = sorted(
            {
                key
                for result in rank_results
                for key, value in result["samples"][mode][0].items()
                if key not in {"trial", "iteration"}
                and isinstance(value, (int, float))
            }
        )
        mode_summaries[mode] = {}
        for metric in scalar_metrics:
            metric_results = [
                result
                for result in rank_results
                if metric in result["samples"][mode][0]
            ]
            rank_medians = [
                statistics.median(
                    float(sample[metric])
                    for sample in result["samples"][mode]
                )
                for result in metric_results
            ]
            mode_summaries[mode][metric] = {
                "rank_medians": rank_medians,
                "median_rank_median": statistics.median(rank_medians),
                "max_rank_median": max(rank_medians),
                "all_rank_samples": summarize(
                    [
                        float(sample[metric])
                        for result in metric_results
                        for sample in result["samples"][mode]
                    ]
                ),
            }
        if "allgather_kernel_ms" in rank_results[0]["samples"][mode][0]:
            mode_summaries[mode]["allgather_kernel_ms_by_index"] = []
            for kernel_index in range(case.allgather_count):
                critical_samples = [
                    max(
                        result["samples"][mode][sample_index][
                            "allgather_kernel_ms"
                        ][kernel_index]
                        for result in rank_results
                    )
                    for sample_index in range(sample_count)
                ]
                mode_summaries[mode]["allgather_kernel_ms_by_index"].append(
                    summarize(critical_samples)
                )

    concurrent_modes = [
        "concurrent_p2p_first",
        "concurrent_allgather_first",
        "concurrent_one_allgather_then_p2p",
    ]
    derived: dict[str, Any] = {}
    for mode in concurrent_modes:
        critical_samples: list[dict[str, float]] = []
        pair_samples: dict[str, list[dict[str, float]]] = {
            f"{left}-{right}": [] for left, right in topology.p2p_pairs
        }
        domain_samples: dict[str, list[dict[str, float]]] = {
            str(domain): [] for domain in topology.domain_ids
        }
        for sample_index in range(sample_count):
            p2p_solo = max(
                result["samples"]["p2p_solo"][sample_index]["p2p_ms"]
                for result in active_rank_results
            )
            ag_solo = max(
                result["samples"]["allgather_solo"][sample_index][
                    "allgather_ms"
                ]
                for result in rank_results
            )
            p2p_concurrent = max(
                result["samples"][mode][sample_index]["p2p_ms"]
                for result in active_rank_results
            )
            ag_concurrent = max(
                result["samples"][mode][sample_index]["allgather_ms"]
                for result in rank_results
            )
            pair_makespan = max(
                result["samples"][mode][sample_index]["pair_ms"]
                for result in rank_results
            )
            ideal_pair = max(p2p_solo, ag_solo)
            critical_samples.append(
                {
                    "p2p_solo_ms": p2p_solo,
                    "p2p_concurrent_ms": p2p_concurrent,
                    "p2p_slowdown": p2p_concurrent / p2p_solo,
                    "allgather_solo_ms": ag_solo,
                    "allgather_concurrent_ms": ag_concurrent,
                    "allgather_slowdown": ag_concurrent / ag_solo,
                    "normal_pair_ms": pair_makespan,
                    "ideal_isolated_pair_ms": ideal_pair,
                    "pair_overhead_vs_isolated": (
                        pair_makespan / ideal_pair
                    ),
                    "removable_fraction_of_normal": (
                        1.0 - ideal_pair / pair_makespan
                    ),
                }
            )
            for left, right in topology.p2p_pairs:
                result_by_rank = {
                    result["metadata"]["rank"]: result
                    for result in rank_results
                }
                solo = max(
                    result_by_rank[rank]["samples"]["p2p_solo"][
                        sample_index
                    ]["p2p_ms"]
                    for rank in (left, right)
                )
                concurrent = max(
                    result_by_rank[rank]["samples"][mode][sample_index][
                        "p2p_ms"
                    ]
                    for rank in (left, right)
                )
                pair_samples[f"{left}-{right}"].append(
                    {
                        "solo_ms": solo,
                        "concurrent_ms": concurrent,
                        "slowdown": concurrent / solo,
                    }
                )
            for domain, ranks in zip(
                topology.domain_ids, topology.allgather_groups
            ):
                result_by_rank = {
                    result["metadata"]["rank"]: result
                    for result in rank_results
                }
                solo = max(
                    result_by_rank[rank]["samples"]["allgather_solo"][
                        sample_index
                    ]["allgather_ms"]
                    for rank in ranks
                )
                concurrent = max(
                    result_by_rank[rank]["samples"][mode][sample_index][
                        "allgather_ms"
                    ]
                    for rank in ranks
                )
                domain_samples[str(domain)].append(
                    {
                        "solo_ms": solo,
                        "concurrent_ms": concurrent,
                        "slowdown": concurrent / solo,
                    }
                )

        derived[mode] = {
            "world_critical_samples": critical_samples,
            "world_critical": {
                metric: summarize(
                    [sample[metric] for sample in critical_samples]
                )
                for metric in critical_samples[0]
            },
            "p2p_pairs": {
                pair: {
                    metric: summarize([sample[metric] for sample in samples])
                    for metric in samples[0]
                }
                for pair, samples in pair_samples.items()
            },
            "allgather_domains": {
                domain: {
                    metric: summarize([sample[metric] for sample in samples])
                    for metric in samples[0]
                }
                for domain, samples in domain_samples.items()
            },
        }
    return {
        "case": asdict(case),
        "modes": mode_summaries,
        "derived": derived,
    }


def csv_rows(
    aggregates: list[dict[str, Any]], dp_size: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    order_names = {
        "concurrent_p2p_first": "p2p_first",
        "concurrent_allgather_first": "allgather_sequence_first",
        "concurrent_one_allgather_then_p2p": "one_allgather_then_p2p",
    }
    for aggregate in aggregates:
        case = Case(**aggregate["case"])
        for mode, queue_order in order_names.items():
            critical = aggregate["derived"][mode]["world_critical"]
            rows.append(
                {
                    "case": case.name,
                    "queue_order": queue_order,
                    "p2p_direction": case.p2p_direction,
                    "p2p_payload": case.p2p_label,
                    "p2p_mib": case.p2p_numel * DTYPE_BYTES / 2**20,
                    "allgather_payload": case.allgather_label,
                    "allgather_full_mib_per_kernel": (
                        case.allgather_full_numel * DTYPE_BYTES / 2**20
                    ),
                    "allgather_shard_mib_per_rank": (
                        case.allgather_full_numel
                        * DTYPE_BYTES
                        / dp_size
                        / 2**20
                    ),
                    "allgather_kernel_count": case.allgather_count,
                    "sample_count": critical["p2p_slowdown"]["count"],
                    "p2p_solo_median_ms": critical["p2p_solo_ms"]["median"],
                    "p2p_concurrent_median_ms": critical[
                        "p2p_concurrent_ms"
                    ]["median"],
                    "p2p_slowdown_median": critical["p2p_slowdown"]["median"],
                    "allgather_solo_median_ms": critical[
                        "allgather_solo_ms"
                    ]["median"],
                    "allgather_concurrent_median_ms": critical[
                        "allgather_concurrent_ms"
                    ]["median"],
                    "allgather_slowdown_median": critical[
                        "allgather_slowdown"
                    ]["median"],
                    "normal_pair_median_ms": critical["normal_pair_ms"][
                        "median"
                    ],
                    "ideal_pair_median_ms": critical[
                        "ideal_isolated_pair_ms"
                    ]["median"],
                    "pair_overhead_median": critical[
                        "pair_overhead_vs_isolated"
                    ]["median"],
                    "removable_fraction_median": critical[
                        "removable_fraction_of_normal"
                    ]["median"],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qwen3.5-397B PP2 cross-domain P2P x within-domain DP8 "
            "all-gather interference benchmark"
        )
    )
    parser.add_argument("--address", default="10.12.249.162")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=8)
    parser.add_argument(
        "--node-domain-map",
        default="0,0,1,1",
        help="NVLink domain ID for each ordered Ray node slot",
    )
    parser.add_argument(
        "--expected-node-ips",
        help=(
            "Required for real runs: ordered comma-separated IPs used to pin "
            "each node slot"
        ),
    )
    parser.add_argument(
        "--expected-hostnames",
        default=(
            "nvl72d052-T01,nvl72d052-T17,"
            "nvl72d056-T14,nvl72d056-T15"
        ),
        help=(
            "Optional ordered hostname assertion. Order must correspond to "
            "--expected-node-ips and --node-domain-map"
        ),
    )
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument(
        "--p2p-shapes",
        default="hidden",
        help="Named model-derived P2P payloads (currently hidden)",
    )
    parser.add_argument(
        "--p2p-payload-mib",
        default="",
        help="Additional custom BF16 P2P payload sizes in MiB",
    )
    parser.add_argument(
        "--p2p-directions",
        choices=["forward", "backward", "both"],
        default="both",
        help="Domain 0 -> 1, domain 1 -> 0, or both as separate cases",
    )
    parser.add_argument(
        "--p2p-pair-mode",
        choices=["single", "all"],
        default="single",
        help=(
            "Activate one cross-domain P2P pair (default) or every possible "
            "pair. All ranks participate in all-gather in either mode."
        ),
    )
    parser.add_argument(
        "--p2p-pair-index",
        type=int,
        default=0,
        help=(
            "Possible P2P pair index used in single mode; index 0 is ranks "
            "0 and 8 for the default topology"
        ),
    )
    parser.add_argument(
        "--allgather-shapes",
        default=(
            "attention_k_or_v,shared_expert_gate_up,attention_o,"
            "attention_gated_q,embedding"
        ),
        help=(
            "Comma-separated named full parameter payloads. Use --dry-run "
            "to list every available Qwen-derived shape."
        ),
    )
    parser.add_argument(
        "--allgather-full-payload-mib",
        default="",
        help=(
            "Additional custom full all-gather payload sizes in MiB before "
            "DP sharding"
        ),
    )
    parser.add_argument(
        "--allgather-counts",
        default="1,2,4",
        help="Repeated sequential all-gather kernel counts",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--gate-sleep-cycles",
        type=int,
        default=20_000_000,
        help="Device-side delay allowing both communicators to be enqueued",
    )
    parser.add_argument(
        "--dry-run",
        "--print-config",
        action="store_true",
        dest="dry_run",
        help="Resolve and print cases without Ray, NCCL, CUDA, or output writes",
    )
    parser.add_argument(
        "--output",
        default=(
            "out/interference_benchmarks/qwen397b_p2p_allgather/"
            "results.json"
        ),
    )
    parser.add_argument(
        "--csv-output",
        help="Companion CSV path; defaults to <output stem>.csv",
    )
    parser.add_argument(
        "--profile-dir",
        help=(
            "Optional directory for one CPU+CUDA PyTorch trace per rank and "
            "case. Every benchmark mode is traced once after timed samples."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "world-size": args.world_size,
        "nodes": args.nodes,
        "gpus-per-node": args.gpus_per_node,
        "dp-size": args.dp_size,
        "microbatch-size": args.microbatch_size,
        "seq-len": args.seq_len,
        "iterations": args.iterations,
        "trials": args.trials,
        "gate-sleep-cycles": args.gate_sleep_cycles,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.warmups < 0:
        raise ValueError("warmups may not be negative")
    if args.p2p_pair_index < 0:
        raise ValueError("p2p-pair-index may not be negative")
    if args.expected_node_ips:
        ips = parse_csv_strings(args.expected_node_ips)
        if len(ips) != args.nodes:
            raise ValueError("expected-node-ips must contain one IP per node")
    if args.expected_hostnames:
        hostnames = parse_csv_strings(args.expected_hostnames)
        if len(hostnames) != args.nodes:
            raise ValueError("expected-hostnames must contain one name per node")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_qwen_config()
    topology = build_topology(args)
    p2p_payloads, ag_payloads, known_p2p, known_ag = resolve_payloads(
        args, config
    )
    cases = build_cases(args, topology, p2p_payloads, ag_payloads)
    preview = build_preview(
        args, config, topology, cases, known_p2p, known_ag
    )
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    if not args.expected_node_ips:
        raise ValueError(
            "A real run requires --expected-node-ips to pin and validate all "
            "four nodes; this prevents silent topology mismatch"
        )

    # Delay Ray imports until after dry-run exits.
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
            "NCCL_IB_HCA",
            "NCCL_LAUNCH_ORDER_IMPLICIT",
            "NCCL_MAX_CTAS",
            "NCCL_MIN_CTAS",
            "NCCL_MNNVL_ENABLE",
            "NCCL_NET",
            "NCCL_PROTO",
            "NCCL_SOCKET_IFNAME",
        )
        if os.environ.get(name)
    }
    forwarded_env.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    ray.init(
        address=f"{args.address}:{args.port}",
        namespace="qwen397b_p2p_allgather_interference",
        runtime_env={"env_vars": forwarded_env},
    )
    expected_ips = parse_csv_strings(args.expected_node_ips)
    expected_hostnames = parse_csv_strings(args.expected_hostnames)
    bundles = [
        {
            "CPU": args.gpus_per_node,
            "GPU": args.gpus_per_node,
            f"node:{expected_ips[node_slot]}": 0.001,
        }
        for node_slot in range(args.nodes)
    ]
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    actor_type = ray.remote(num_cpus=1, num_gpus=1)(P2PAllGatherRank)
    actors: list[Any] = []
    try:
        ray.get(pg.ready(), timeout=600)
        for node_slot in range(args.nodes):
            for local_slot in range(args.gpus_per_node):
                rank = node_slot * args.gpus_per_node + local_slot
                actors.append(
                    actor_type.options(
                        scheduling_strategy=PlacementGroupSchedulingStrategy(
                            placement_group=pg,
                            placement_group_bundle_index=node_slot,
                        )
                    ).remote(rank, node_slot, local_slot)
                )
        placements = ray.get([actor.metadata.remote() for actor in actors])
        node_counts = Counter(item["node_id"] for item in placements)
        if sorted(node_counts.values()) != [args.gpus_per_node] * args.nodes:
            raise RuntimeError(
                f"Expected {args.gpus_per_node} actors/node: {node_counts}"
            )
        gpu_assignments = {
            (item["node_id"], tuple(item["ray_gpu_ids"]))
            for item in placements
        }
        if len(gpu_assignments) != args.world_size:
            raise RuntimeError(
                f"Ray assigned a GPU more than once: {placements}"
            )
        for node_slot in range(args.nodes):
            slot_placements = [
                item
                for item in placements
                if item["node_slot"] == node_slot
            ]
            observed_ips = {item["node_ip"] for item in slot_placements}
            observed_hosts = {item["hostname"] for item in slot_placements}
            if observed_ips != {expected_ips[node_slot]}:
                raise RuntimeError(
                    f"node slot {node_slot} IP mismatch: "
                    f"{observed_ips} != {expected_ips[node_slot]}"
                )
            if expected_hostnames and observed_hosts != {
                expected_hostnames[node_slot]
            }:
                raise RuntimeError(
                    f"node slot {node_slot} hostname mismatch: "
                    f"{observed_hosts} != {expected_hostnames[node_slot]}"
                )

        master_addr, master_port = ray.get(actors[0].rendezvous.remote())
        started = time.time()
        # Every actor receives every case and runs them in the same order.
        per_actor_results = ray.get(
            [
                actor.run.remote(
                    args.world_size,
                    master_addr,
                    master_port,
                    asdict(topology),
                    [asdict(case) for case in cases],
                    args.warmups,
                    args.iterations,
                    args.trials,
                    args.gate_sleep_cycles,
                    (
                        str(Path(args.profile_dir).resolve())
                        if args.profile_dir
                        else None
                    ),
                )
                for actor in actors
            ]
        )
        finished = time.time()
        case_outputs = []
        for case_index, case in enumerate(cases):
            rank_results = [
                actor_results[case_index]
                for actor_results in per_actor_results
            ]
            case_outputs.append(
                {
                    "case": asdict(case),
                    "rank_results": rank_results,
                    "aggregate": aggregate_case(
                        case, rank_results, topology
                    ),
                }
            )
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
            "placements": placements,
            "cases": case_outputs,
        }
        output_path = Path(args.output)
        csv_path = (
            Path(args.csv_output)
            if args.csv_output
            else output_path.with_suffix(".csv")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        rows = csv_rows(
            [case_output["aggregate"] for case_output in case_outputs],
            args.dp_size,
        )
        write_csv(csv_path, rows)
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        print(f"Wrote {output_path}")
        print(f"Wrote {csv_path}")
    finally:
        # Ray owns these non-detached actors. Removing their placement group
        # and disconnecting the driver releases them; explicitly ray.kill-ing
        # all 16 concurrently triggers a Ray 2.44 arm64 task-event teardown
        # crash after otherwise successful runs.
        remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
