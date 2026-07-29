#!/usr/bin/env python3
"""Measure Qwen3.5-397B EP all-to-all x DP all-reduce interference.

The harness targets a 16-GPU placement in which every expert-parallel (EP)
group is contained by one NVLink domain and every data-parallel (DP) group
crosses NVLink domains. It deliberately keeps
the EP and DP communicators, CUDA streams, and buffers distinct.

The default workload uses the Qwen3.5-397B-A17B preset from
``examples.models.qwen3._QWEN35_397B`` and the long-context workload point used
by the compute-interference benchmark: microbatch=1 and sequence length=32768
(32,768 local tokens per EP source rank).  The equal-split BF16 A2A payload is
therefore
``local_tokens * top_k * dim`` elements.  DP gradient buckets are synthesized
from that same preset's attention, router, shared-expert, routed-expert,
normalization, embedding, and output-projection parameter shapes.

No communicator or CUDA work is performed with ``--dry-run`` (or its alias
``--print-config``); the resolved rank map, payload, gradient bucket sequence,
and estimated allocation are printed as JSON instead.
"""

from __future__ import annotations

import argparse
from collections import Counter
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


@dataclass(frozen=True)
class RankMap:
    name: str
    world_size: int
    ep_groups: tuple[tuple[int, ...], ...]
    dp_groups: tuple[tuple[int, ...], ...]
    node_domain_map: tuple[int, ...] | None

    def ep_group_for(self, rank: int) -> tuple[int, ...]:
        return next(group for group in self.ep_groups if rank in group)

    def dp_group_for(self, rank: int) -> tuple[int, ...]:
        return next(group for group in self.dp_groups if rank in group)


@dataclass(frozen=True)
class Bucket:
    index: int
    numel: int
    dtype: str
    pieces: tuple[tuple[str, int], ...]

    @property
    def element_size(self) -> int:
        return 2 if self.dtype == "bf16" else 4

    @property
    def bytes(self) -> int:
        return self.numel * self.element_size


@dataclass(frozen=True)
class Case:
    microbatch_size: int
    seq_len: int
    payload_scale: float
    gradient_sequence: str
    gradient_dtype: str
    bucket_cap_mib: float
    gradient_collective: str
    a2a_delay_cycles: int
    gradient_delay_cycles: int

    @property
    def local_tokens(self) -> int:
        return self.microbatch_size * self.seq_len

    @property
    def name(self) -> str:
        scale = str(self.payload_scale).replace(".", "p")
        cap = str(self.bucket_cap_mib).replace(".", "p")
        return (
            f"mb{self.microbatch_size}_seq{self.seq_len}_a2ax{scale}_"
            f"{self.gradient_sequence}_{self.gradient_dtype}_"
            f"{self.gradient_collective}_cap{cap}mib_"
            f"adelay{self.a2a_delay_cycles}_gdelay{self.gradient_delay_cycles}"
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


def benchmark_modes(skip_serialized: bool) -> list[str]:
    modes = ["a2a_solo", "allreduce_solo"]
    if not skip_serialized:
        modes.extend(
            ["serialized_a2a_allreduce", "serialized_allreduce_a2a"]
        )
    modes.extend(["concurrent_a2a_first", "concurrent_allreduce_first"])
    return modes


def load_qwen_config() -> dict[str, Any]:
    # Import the actual registry entry rather than duplicating architecture
    # constants here. This also makes dry-run output expose local preset edits.
    # Some cluster images inject a stale /workspace/piper into PYTHONPATH, so
    # make the repository containing this script authoritative.
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
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "head_dim",
        "hidden_dim",
        "moe_inter_dim",
        "num_experts",
        "num_shared_experts",
        "top_k",
        "vocab_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{MODEL_SYMBOL} is missing required fields: {missing}")
    if not config["moe_inter_dim"] or not config["num_experts"]:
        raise ValueError(f"{MODEL_SYMBOL} is not an MoE configuration")
    return config


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def automatic_rank_map(
    world_size: int,
    ep_size: int,
    dp_size: int | None,
    node_domain_map: tuple[int, ...] | None,
) -> RankMap:
    if ep_size <= 0 or world_size % ep_size:
        raise ValueError("EP size must be positive and divide world size")
    inferred_dp = world_size // ep_size
    if dp_size is not None and dp_size != inferred_dp:
        raise ValueError(
            f"EP={ep_size}, DP={dp_size} does not factor world={world_size}"
        )
    ep_groups = tuple(
        tuple(range(first, first + ep_size))
        for first in range(0, world_size, ep_size)
    )
    dp_groups = tuple(
        tuple(ep_index + replica * ep_size for replica in range(inferred_dp))
        for ep_index in range(ep_size)
    )
    return RankMap(
        name=f"contiguous_ep{ep_size}_dp{inferred_dp}",
        world_size=world_size,
        ep_groups=ep_groups,
        dp_groups=dp_groups,
        node_domain_map=node_domain_map,
    )


def load_rank_map(args: argparse.Namespace) -> RankMap:
    domains = (
        tuple(parse_csv_ints(args.node_domain_map))
        if args.node_domain_map
        else None
    )
    if args.rank_map_file:
        raw = json.loads(Path(args.rank_map_file).read_text(encoding="utf-8"))
        file_domains = raw.get("node_domain_map")
        if domains is not None and file_domains is not None:
            raise ValueError(
                "Specify node_domain_map in the rank-map file or CLI, not both"
            )
        return RankMap(
            name=raw.get("name", Path(args.rank_map_file).stem),
            world_size=int(raw.get("world_size", args.world_size)),
            ep_groups=tuple(tuple(map(int, group)) for group in raw["ep_groups"]),
            dp_groups=tuple(tuple(map(int, group)) for group in raw["dp_groups"]),
            node_domain_map=(
                tuple(map(int, file_domains)) if file_domains is not None else domains
            ),
        )
    return automatic_rank_map(args.world_size, args.ep_size, args.dp_size, domains)


def validate_rank_map(
    rank_map: RankMap,
    nodes: int,
    gpus_per_node: int,
    allow_intradomain_dp: bool,
) -> list[str]:
    if rank_map.world_size != nodes * gpus_per_node:
        raise ValueError("rank-map world size must equal nodes * gpus-per-node")
    ranks = set(range(rank_map.world_size))
    for label, groups in (("EP", rank_map.ep_groups), ("DP", rank_map.dp_groups)):
        flattened = [rank for group in groups for rank in group]
        if set(flattened) != ranks or len(flattened) != rank_map.world_size:
            raise ValueError(f"{label} groups must partition every rank exactly once")
        if not groups or any(not group for group in groups):
            raise ValueError(f"{label} groups may not be empty")
        if any(tuple(sorted(group)) != group for group in groups):
            raise ValueError(
                f"{label} group ranks must be ascending so collective chunk order "
                "is unambiguous"
            )
    for ep_group in rank_map.ep_groups:
        for dp_group in rank_map.dp_groups:
            if len(set(ep_group) & set(dp_group)) != 1:
                raise ValueError(
                    "Every EP/DP group pair must intersect in exactly one rank"
                )
    warnings: list[str] = []
    if rank_map.node_domain_map is None:
        warnings.append(
            "No node-domain map supplied; EP locality and DP crossing are unchecked"
        )
        return warnings
    if len(rank_map.node_domain_map) != nodes:
        raise ValueError("node-domain-map must contain one domain ID per Ray node")

    def domain(rank: int) -> int:
        node_slot = rank // gpus_per_node
        return rank_map.node_domain_map[node_slot]

    for group in rank_map.ep_groups:
        group_domains = {domain(rank) for rank in group}
        if len(group_domains) != 1:
            raise ValueError(f"EP group {group} crosses domains {group_domains}")
    for group in rank_map.dp_groups:
        group_domains = {domain(rank) for rank in group}
        if len(group) > 1 and len(group_domains) == 1:
            message = f"DP group {group} is intra-domain ({group_domains})"
            if allow_intradomain_dp:
                warnings.append(message + "; accepted as a negative control")
            else:
                raise ValueError(message + "; use --allow-intradomain-dp to permit")
    return warnings


def gradient_components(
    config: dict[str, Any],
    ep_size: int,
    layers: int,
    include_io: bool,
    architecture_profile: str,
    expert_layout: str,
    layer_pattern: tuple[str, ...],
) -> list[tuple[str, int]]:
    """Return a backward-order proxy of per-rank DP-reduced parameters.

    Routed experts are EP-sharded; other listed parameters are replicated.
    This is a shape-derived replay, not a claim about the precise bucket order
    of a particular FSDP/DDP implementation.  A captured production bucket
    list can be supplied with ``--gradient-bucket-mib``.
    """

    dim = int(config["dim"])
    q_width = int(config["n_heads"]) * int(config["head_dim"])
    kv_width = int(config["n_kv_heads"]) * int(config["head_dim"])
    expert_width = int(config["moe_inter_dim"])
    num_experts = int(config["num_experts"])
    shared_experts = int(config["num_shared_experts"])
    base = num_experts // ep_size
    extra = num_experts % ep_size
    # The largest local shard is conservative when expert count is not evenly
    # divisible. Metadata records this choice in the dry-run report.
    local_experts = base + int(extra > 0)
    if expert_layout == "fused_gate_up":
        # Backward reaches the down projection before gate/up. Keep this order
        # when replaying exact parameter boundaries.
        routed_experts = [
            ("routed_expert.down", local_experts * dim * expert_width),
            ("routed_expert.fused_gate_up", local_experts * 2 * dim * expert_width),
        ]
    elif expert_layout == "split_w1_w3":
        routed_experts = [
            ("routed_expert.w2_down", local_experts * dim * expert_width),
            ("routed_expert.w1", local_experts * dim * expert_width),
            ("routed_expert.w3", local_experts * dim * expert_width),
        ]
    else:
        raise ValueError(expert_layout)

    common_nonrouted = [
        ("shared_expert.gate_up", shared_experts * 2 * dim * expert_width),
        ("shared_expert.down", shared_experts * dim * expert_width),
        ("router", dim * num_experts),
        ("shared_expert_gate", dim),
        ("input_layernorm", dim),
        ("post_attention_layernorm", dim),
    ]
    if architecture_profile == "production_exact":
        # Full-attention Qwen3.5 uses a gated Q projection: [16384, 4096],
        # K/V [512, 4096], and O [4096, 8192]. Together with the common
        # nonrouted tensors above this is 228.024414 MiB of BF16 gradients.
        full_attention = [
            ("full_attention.o_proj", q_width * dim),
            ("full_attention.v_proj", dim * kv_width),
            ("full_attention.k_proj", dim * kv_width),
            ("full_attention.gated_q_proj", dim * 2 * q_width),
            ("full_attention.q_norm", int(config["head_dim"])),
            ("full_attention.k_norm", int(config["head_dim"])),
        ]
        # Preserve the exact Gated-DeltaNet parameter boundaries. They matter
        # for a bucket/collective replay even though QKV and Z have a combined
        # output width of 20,480.
        key_width = 16 * 128
        value_width = 64 * 128
        conv_width = 2 * key_width + value_width
        linear_attention = [
            ("linear_attention.out_proj", dim * value_width),
            ("linear_attention.rmsnorm_gated", 128),
            ("linear_attention.A_log", 64),
            ("linear_attention.dt_bias", 64),
            ("linear_attention.in_proj_b", 64 * dim),
            ("linear_attention.in_proj_a", 64 * dim),
            ("linear_attention.in_proj_z", value_width * dim),
            ("linear_attention.conv1d", conv_width * 4),
            ("linear_attention.in_proj_qkv", conv_width * dim),
        ]
    elif architecture_profile == "repo_proxy":
        # The local TorchTitan proxy has a conventional non-gated Q projection
        # and does not reproduce the production L,L,L,F forward layer pattern.
        full_attention = [
            ("attention.o_proj", q_width * dim),
            ("attention.v_proj", dim * kv_width),
            ("attention.k_proj", dim * kv_width),
            ("attention.q_proj", dim * q_width),
        ]
        linear_attention = full_attention
    else:
        raise ValueError(architecture_profile)

    components: list[tuple[str, int]] = []
    for layer in range(layers - 1, -1, -1):
        kind = layer_pattern[layer % len(layer_pattern)]
        layer_specific = full_attention if kind == "F" else linear_attention
        per_layer = routed_experts + common_nonrouted + layer_specific
        components.extend(
            (f"layers.{layer}.{name}", numel) for name, numel in per_layer if numel
        )
    if include_io:
        vocab = int(config["vocab_size"])
        components.append(("output_projection", vocab * dim))
        if not bool(config.get("enable_weight_tying", False)):
            components.append(("token_embedding", vocab * dim))
    return components


def pack_components(
    components: list[tuple[str, int]], cap_numel: int, dtype: str
) -> list[Bucket]:
    if cap_numel <= 0:
        raise ValueError("gradient bucket cap must contain at least one element")
    buckets: list[Bucket] = []
    pieces: list[tuple[str, int]] = []
    used = 0
    for name, component_numel in components:
        remaining = component_numel
        piece_index = 0
        while remaining:
            take = min(remaining, cap_numel - used)
            suffix = f"[{piece_index}]" if component_numel > cap_numel else ""
            pieces.append((name + suffix, take))
            used += take
            remaining -= take
            piece_index += 1
            if used == cap_numel:
                buckets.append(Bucket(len(buckets), used, dtype, tuple(pieces)))
                pieces = []
                used = 0
    if used:
        buckets.append(Bucket(len(buckets), used, dtype, tuple(pieces)))
    return buckets


def derive_buckets(
    args: argparse.Namespace,
    config: dict[str, Any],
    rank_map: RankMap,
) -> tuple[list[Bucket], dict[str, Any]]:
    dtype_bytes = 2 if args.gradient_dtype == "bf16" else 4
    ep_size = len(rank_map.ep_groups[0])
    if any(len(group) != ep_size for group in rank_map.ep_groups):
        raise ValueError("all EP groups must have the same size")
    layers = args.layers if args.layers is not None else int(config["n_layers"])
    pattern = tuple(
        item.strip().upper() for item in args.layer_pattern.split(",") if item.strip()
    )
    if not pattern or any(item not in {"F", "L"} for item in pattern):
        raise ValueError("layer-pattern must be a comma-separated sequence of F and L")
    all_components = gradient_components(
        config,
        ep_size,
        layers,
        args.include_io,
        args.architecture_profile,
        args.expert_layout,
        pattern,
    )
    if args.gradient_sequence == "layer":
        component_prefix = f"layers.{layers - 1}."
        components = [
            (name, numel)
            for name, numel in all_components
            if name.startswith(component_prefix)
        ]
    elif args.gradient_sequence == "expert":
        component_prefix = f"layers.{layers - 1}.routed_expert."
        components = [
            (name, numel)
            for name, numel in all_components
            if name.startswith(component_prefix)
        ]
    elif args.gradient_sequence == "model":
        components = all_components
    elif args.gradient_sequence == "individual":
        components = all_components
    elif args.gradient_sequence == "custom":
        components = []
    else:
        raise ValueError(args.gradient_sequence)

    if args.gradient_bucket_mib:
        custom_sizes = parse_csv_floats(args.gradient_bucket_mib)
        if any(size <= 0 for size in custom_sizes):
            raise ValueError("custom gradient bucket MiB values must be positive")
        buckets = [
            Bucket(
                index,
                int(size * 2**20) // dtype_bytes,
                args.gradient_dtype,
                ((f"custom_bucket_{index}", int(size * 2**20) // dtype_bytes),),
            )
            for index, size in enumerate(custom_sizes)
        ]
        source = "custom_mib"
    else:
        if args.gradient_sequence == "custom":
            raise ValueError("custom sequence requires --gradient-bucket-mib")
        if args.gradient_sequence == "expert":
            # Preserve production fused parameter boundaries: at EP=p the two
            # BF16 collectives are exactly 8 GiB/p (gate_up) and 4 GiB/p (down).
            buckets = [
                Bucket(index, numel, args.gradient_dtype, ((name, numel),))
                for index, (name, numel) in enumerate(components)
            ]
        else:
            cap_numel = int(args.bucket_cap_mib * 2**20) // dtype_bytes
            packed = pack_components(components, cap_numel, args.gradient_dtype)
            buckets = [max(packed, key=lambda bucket: bucket.numel)] if (
                args.gradient_sequence == "individual"
            ) else packed
        if buckets and args.gradient_sequence == "individual":
            only = buckets[0]
            buckets = [Bucket(0, only.numel, only.dtype, only.pieces)]
        source = "qwen_config_components"
    if not buckets:
        raise ValueError("gradient bucket sequence is empty")
    total_component_numel = sum(numel for _, numel in components)
    metadata = {
        "source": source,
        "sequence": args.gradient_sequence,
        "layers": layers,
        "include_io": args.include_io,
        "architecture_profile": args.architecture_profile,
        "expert_layout": args.expert_layout,
        "layer_pattern": pattern,
        "production_depth_note": (
            "Default is the 60-layer production L,L,L,F forward pattern "
            "(F,L,L,L in backward traversal). The local "
            f"{MODEL_SYMBOL} registry entry currently reports "
            f"n_layers={config['n_layers']}."
        ),
        "bucket_cap_mib": args.bucket_cap_mib,
        "component_numel": total_component_numel,
        "component_bytes": total_component_numel * dtype_bytes,
        "bucket_count": len(buckets),
        "sequence_numel": sum(bucket.numel for bucket in buckets),
        "sequence_bytes": sum(bucket.bytes for bucket in buckets),
        "buckets": [asdict(bucket) | {"bytes": bucket.bytes} for bucket in buckets],
        "caveat": (
            "Production-exact parameter totals, but shape-derived bucket order. "
            "MTP is excluded. Use --gradient-bucket-mib with a captured production "
            "sequence when exact DDP/FSDP bucketization is available."
        ),
    }
    return buckets, metadata


def build_preview(
    args: argparse.Namespace,
    config: dict[str, Any],
    rank_map: RankMap,
    map_warnings: list[str],
    buckets: list[Bucket],
    bucket_metadata: dict[str, Any],
) -> dict[str, Any]:
    local_tokens = args.microbatch_size * args.seq_len
    routed_rows = int(round(local_tokens * int(config["top_k"]) * args.payload_scale))
    a2a_numel = routed_rows * int(config["dim"])
    ep_size = len(rank_map.ep_groups[0])
    if a2a_numel % ep_size:
        raise ValueError(f"A2A numel {a2a_numel} is not divisible by EP={ep_size}")
    a2a_bytes = a2a_numel * 2
    gradient_bytes = sum(bucket.bytes for bucket in buckets)
    dp_size = len(rank_map.dp_groups[0])
    if args.gradient_collective == "allgather":
        gradient_buffer_bytes = gradient_bytes // dp_size
        gradient_output_bytes = gradient_bytes
    elif args.gradient_collective == "reduce_scatter":
        gradient_buffer_bytes = gradient_bytes
        gradient_output_bytes = gradient_bytes // dp_size
    else:
        gradient_buffer_bytes = gradient_bytes
        gradient_output_bytes = 0
    return {
        "model": MODEL_NAME,
        "model_symbol": MODEL_SYMBOL,
        "qwen_config": config,
        "training_shape": {
            "microbatch_size_per_rank": args.microbatch_size,
            "sequence_length": args.seq_len,
            "local_tokens_per_ep_source_rank": local_tokens,
            "note": "global batch depends on DP replicas and gradient accumulation",
        },
        "rank_map": asdict(rank_map),
        "rank_map_warnings": map_warnings,
        "a2a": {
            "dtype": "bf16",
            "payload_scale": args.payload_scale,
            "routed_rows": routed_rows,
            "numel_per_rank": a2a_numel,
            "bytes_per_rank": a2a_bytes,
            "mib_per_rank": a2a_bytes / 2**20,
            "equal_split_numel_per_peer": a2a_numel // ep_size,
        },
        "allreduce": bucket_metadata,
        "gradient_communication": bucket_metadata,
        "payload_dependency_note": (
            "A2A bytes scale with microbatch and sequence length. DP/FSDP "
            "collective bytes are parameter-bucket sizes and are independent "
            "of batch/sequence length."
        ),
        "collective_semantics": {
            "requested_primitive": args.gradient_collective,
            "expert_sequence": (
                "Expert parameter/gradient buckets communicate over the "
                "cross-domain DP/EDP group at fixed EP index."
            ),
            "dense_sequence_caveat": (
                "Layer/model sequences are controlled traffic replays. Dense "
                "parameters can require aggregation across a larger dense-data "
                "mesh than the fixed-EP-index DP groups used here."
            ),
            "production_caveat": (
                "FSDP all-gather materializes parameter buckets, while "
                "reduce-scatter communicates gradients. The selected primitive "
                "must be interpreted at its corresponding training phase."
            ),
        },
        "estimated_live_tensor_allocation": {
            "a2a_input_output_bytes_per_rank": 2 * a2a_bytes,
            "gradient_buffers_bytes_per_rank": gradient_buffer_bytes,
            "gradient_output_bytes_per_rank": gradient_output_bytes,
            "total_bytes_per_rank": (
                2 * a2a_bytes
                + gradient_buffer_bytes
                + gradient_output_bytes
            ),
            "total_gib_per_rank": (
                2 * a2a_bytes + gradient_bytes + gradient_output_bytes
            )
            / 2**30,
        },
        "measurement": {
            "warmups": args.warmups,
            "iterations_per_trial": args.iterations,
            "trials": args.trials,
            "samples_per_mode": args.iterations * args.trials,
            "a2a_delay_cycles": args.a2a_delay_cycles,
            "gradient_delay_cycles": args.gradient_delay_cycles,
            "modes": benchmark_modes(args.skip_serialized),
        },
    }


class A2AAllReduceRank:
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
        rank_map_dict: dict[str, Any],
        case_dict: dict[str, Any],
        bucket_dicts: list[dict[str, Any]],
        model_config: dict[str, Any],
        warmups: int,
        iterations: int,
        trials: int,
        gate_sleep_cycles: int,
        modes: list[str],
        profile_dir: str | None,
    ) -> dict[str, Any]:
        import torch
        import torch.distributed as dist

        rank_map = RankMap(
            name=rank_map_dict["name"],
            world_size=rank_map_dict["world_size"],
            ep_groups=tuple(tuple(group) for group in rank_map_dict["ep_groups"]),
            dp_groups=tuple(tuple(group) for group in rank_map_dict["dp_groups"]),
            node_domain_map=(
                tuple(rank_map_dict["node_domain_map"])
                if rank_map_dict["node_domain_map"] is not None
                else None
            ),
        )
        case = Case(**case_dict)
        buckets = [
            Bucket(
                index=item["index"],
                numel=item["numel"],
                dtype=item["dtype"],
                pieces=tuple(tuple(piece) for piece in item["pieces"]),
            )
            for item in bucket_dicts
        ]
        torch.cuda.set_device(0)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://{master_addr}:{master_port}",
            rank=self.rank,
            world_size=world_size,
            device_id=torch.device("cuda", 0),
            timeout=timedelta(minutes=60),
        )
        control_group = None
        try:
            all_ranks = list(range(world_size))
            control_group = dist.new_group(ranks=all_ranks, backend="gloo")
            ep_group = None
            dp_group = None
            # Every rank creates every group in exactly this order. EP and DP
            # are deliberately separate NCCL communicators even if a custom
            # map happens to give them identical memberships.
            for ranks in rank_map.ep_groups:
                handle = dist.new_group(ranks=list(ranks), backend="nccl")
                if self.rank in ranks:
                    ep_group = handle
            for ranks in rank_map.dp_groups:
                handle = dist.new_group(ranks=list(ranks), backend="nccl")
                if self.rank in ranks:
                    dp_group = handle
            if ep_group is None or dp_group is None:
                raise RuntimeError(f"rank {self.rank} lacks an EP or DP group")
            ep_ranks = rank_map.ep_group_for(self.rank)
            dp_ranks = rank_map.dp_group_for(self.rank)

            # Stable streams are created once and reused for every mode/trial.
            ep_stream = torch.cuda.Stream(priority=0)
            dp_stream = torch.cuda.Stream(priority=0)
            control_stream = torch.cuda.Stream(priority=-1)

            routed_rows = int(
                round(
                    case.local_tokens
                    * int(model_config["top_k"])
                    * case.payload_scale
                )
            )
            a2a_numel = routed_rows * int(model_config["dim"])
            if a2a_numel % len(ep_ranks):
                raise ValueError("A2A numel must divide evenly over the EP group")
            a2a_input = torch.full(
                (a2a_numel,), self.rank + 1, dtype=torch.bfloat16, device="cuda"
            )
            a2a_output = torch.empty_like(a2a_input)
            gradient_torch_dtype = (
                torch.bfloat16 if case.gradient_dtype == "bf16" else torch.float32
            )
            if case.gradient_collective == "allgather":
                if any(bucket.numel % len(dp_ranks) for bucket in buckets):
                    raise ValueError(
                        "all-gather full bucket numel must divide the DP group size"
                    )
                gradient_buffers = [
                    torch.full(
                        (bucket.numel // len(dp_ranks),),
                        self.rank + 1,
                        dtype=gradient_torch_dtype,
                        device="cuda",
                    )
                    for bucket in buckets
                ]
                gradient_outputs = [
                    torch.empty(
                        (bucket.numel,),
                        dtype=gradient_torch_dtype,
                        device="cuda",
                    )
                    for bucket in buckets
                ]
            else:
                gradient_buffers = [
                    torch.full(
                        (bucket.numel,),
                        self.rank + 1,
                        dtype=gradient_torch_dtype,
                        device="cuda",
                    )
                    for bucket in buckets
                ]
            if case.gradient_collective == "reduce_scatter":
                if any(bucket.numel % len(dp_ranks) for bucket in buckets):
                    raise ValueError(
                        "reduce-scatter bucket numel must divide the DP group size"
                    )
                gradient_outputs = [
                    torch.empty(
                        (bucket.numel // len(dp_ranks),),
                        dtype=gradient_torch_dtype,
                        device="cuda",
                    )
                    for bucket in buckets
                ]
            elif case.gradient_collective != "allgather":
                gradient_outputs = []

            def enqueue_a2a() -> Any:
                # async_op avoids a host-side wait between communicator calls;
                # the stream completion event remains the timing authority.
                return dist.all_to_all_single(
                    a2a_output, a2a_input, group=ep_group, async_op=True
                )

            def enqueue_gradient(index: int) -> Any:
                if case.gradient_collective == "allreduce":
                    return dist.all_reduce(
                        gradient_buffers[index], group=dp_group, async_op=True
                    )
                if case.gradient_collective == "reduce_scatter":
                    return dist.reduce_scatter_tensor(
                        gradient_outputs[index],
                        gradient_buffers[index],
                        group=dp_group,
                        async_op=True,
                    )
                if case.gradient_collective == "allgather":
                    return dist.all_gather_into_tensor(
                        gradient_outputs[index],
                        gradient_buffers[index],
                        group=dp_group,
                        async_op=True,
                    )
                raise ValueError(case.gradient_collective)

            def enqueue_gradient_sequence() -> list[Any]:
                return [enqueue_gradient(index) for index in range(len(buckets))]

            def reset_gradients() -> None:
                with torch.cuda.stream(dp_stream):
                    for buffer in gradient_buffers:
                        buffer.fill_(self.rank + 1)
                dp_stream.synchronize()

            # Pay communicator initialization and connection setup before any
            # timed sample. One AR is sufficient to initialize the communicator;
            # mode warmups exercise every configured bucket size.
            with torch.cuda.stream(ep_stream):
                init_a2a_work = enqueue_a2a()
            with torch.cuda.stream(dp_stream):
                init_ar_work = enqueue_gradient(0)
            torch.cuda.synchronize()
            del init_a2a_work, init_ar_work
            reset_gradients()
            dist.barrier(group=control_group)

            def run_once(mode: str, trial: int, iteration: int) -> dict[str, float]:
                has_a2a = mode != "allreduce_solo"
                has_ar = mode != "a2a_solo"
                if has_ar:
                    reset_gradients()

                gate = torch.cuda.Event(enable_timing=True)
                a2a_start = torch.cuda.Event(enable_timing=True)
                a2a_end = torch.cuda.Event(enable_timing=True)
                ar_start = torch.cuda.Event(enable_timing=True)
                ar_end = torch.cuda.Event(enable_timing=True)
                a2a_work: Any | None = None
                ar_work: list[Any] = []
                dist.barrier(group=control_group)

                # Queue a delayed future record, then queue both work streams
                # behind it. The delay gives every host rank time to enqueue
                # both communicators before the device releases either one.
                with torch.cuda.stream(control_stream):
                    torch.cuda._sleep(gate_sleep_cycles)
                    gate.record()

                def launch_a2a() -> None:
                    nonlocal a2a_work
                    ep_stream.wait_event(gate)
                    with torch.cuda.stream(ep_stream):
                        if case.a2a_delay_cycles:
                            torch.cuda._sleep(case.a2a_delay_cycles)
                        a2a_start.record()
                        a2a_work = enqueue_a2a()

                def finish_a2a() -> None:
                    if a2a_work is None:
                        raise RuntimeError("A2A completion requested before enqueue")
                    # ProcessGroupNCCL runs on internal streams. Work.wait()
                    # inserts their completion dependency into the current
                    # stream; only an event recorded after it times the actual
                    # collective rather than host-side enqueue.
                    with torch.cuda.stream(ep_stream):
                        a2a_work.wait()
                        a2a_end.record()

                def launch_ar() -> None:
                    nonlocal ar_work
                    dp_stream.wait_event(gate)
                    with torch.cuda.stream(dp_stream):
                        if case.gradient_delay_cycles:
                            torch.cuda._sleep(case.gradient_delay_cycles)
                        ar_start.record()
                        ar_work = enqueue_gradient_sequence()

                def finish_ar() -> None:
                    if not ar_work:
                        raise RuntimeError("AR completion requested before enqueue")
                    with torch.cuda.stream(dp_stream):
                        for work in ar_work:
                            work.wait()
                        ar_end.record()

                if mode == "a2a_solo":
                    launch_a2a()
                    finish_a2a()
                elif mode == "allreduce_solo":
                    launch_ar()
                    finish_ar()
                elif mode == "serialized_a2a_allreduce":
                    launch_a2a()
                    finish_a2a()
                    dp_stream.wait_event(a2a_end)
                    launch_ar()
                    finish_ar()
                elif mode == "serialized_allreduce_a2a":
                    launch_ar()
                    finish_ar()
                    ep_stream.wait_event(ar_end)
                    launch_a2a()
                    finish_a2a()
                elif mode == "concurrent_a2a_first":
                    # Enqueue both communicator calls before either wait. This
                    # preserves overlap even if a backend makes Work.wait()
                    # host-blocking rather than only stream-blocking.
                    launch_a2a()
                    launch_ar()
                    finish_a2a()
                    finish_ar()
                elif mode == "concurrent_allreduce_first":
                    launch_ar()
                    launch_a2a()
                    finish_ar()
                    finish_a2a()
                else:
                    raise ValueError(mode)

                if has_a2a:
                    a2a_end.synchronize()
                if has_ar:
                    ar_end.synchronize()
                dist.barrier(group=control_group)

                starts: list[float] = []
                ends: list[float] = []
                sample: dict[str, float] = {
                    "trial": float(trial),
                    "iteration": float(iteration),
                }
                if has_a2a:
                    sample["a2a_ms"] = a2a_start.elapsed_time(a2a_end)
                    sample["a2a_gate_to_start_ms"] = gate.elapsed_time(a2a_start)
                    sample["a2a_gate_to_end_ms"] = gate.elapsed_time(a2a_end)
                    starts.append(sample["a2a_gate_to_start_ms"])
                    ends.append(sample["a2a_gate_to_end_ms"])
                if has_ar:
                    sample["allreduce_ms"] = ar_start.elapsed_time(ar_end)
                    sample["allreduce_gate_to_start_ms"] = gate.elapsed_time(ar_start)
                    sample["allreduce_gate_to_end_ms"] = gate.elapsed_time(ar_end)
                    starts.append(sample["allreduce_gate_to_start_ms"])
                    ends.append(sample["allreduce_gate_to_end_ms"])
                sample["pair_ms"] = max(ends) - min(starts)
                if len(starts) == 2:
                    sample["start_skew_ms"] = abs(starts[0] - starts[1])
                return sample

            for warmup in range(warmups):
                order = modes if warmup % 2 == 0 else list(reversed(modes))
                for mode in order:
                    run_once(mode, -1, warmup)

            samples: dict[str, list[dict[str, float]]] = {
                mode: [] for mode in modes
            }
            for trial in range(trials):
                for iteration in range(iterations):
                    order = (
                        modes
                        if (trial * iterations + iteration) % 2 == 0
                        else list(reversed(modes))
                    )
                    for mode in order:
                        samples[mode].append(run_once(mode, trial, iteration))

            trace_path = None
            if profile_dir:
                trace_dir = Path(profile_dir)
                trace_dir.mkdir(parents=True, exist_ok=True)
                safe_case_name = "".join(
                    character if character.isalnum() or character in "._-" else "_"
                    for character in case.name
                )
                trace_path = trace_dir / (
                    f"{safe_case_name}_rank{self.rank}_{socket.gethostname()}.json"
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

            # Validate one value from every A2A source chunk and a bounded
            # sample of the DP bucket sequence after an explicit known run.
            reset_gradients()
            with torch.cuda.stream(ep_stream):
                validation_a2a_work = enqueue_a2a()
            with torch.cuda.stream(dp_stream):
                validation_ar_work = enqueue_gradient_sequence()
            torch.cuda.synchronize()
            del validation_a2a_work, validation_ar_work
            ep_chunk = a2a_numel // len(ep_ranks)
            a2a_observed = [
                float(a2a_output[index * ep_chunk].item())
                for index in range(len(ep_ranks))
            ]
            a2a_expected = [float(rank + 1) for rank in ep_ranks]
            if a2a_observed != a2a_expected:
                raise RuntimeError(
                    f"rank {self.rank} A2A validation failed: "
                    f"{a2a_observed} != {a2a_expected}"
                )
            check_indices = sorted({0, len(gradient_buffers) // 2, len(gradient_buffers) - 1})
            if case.gradient_collective == "allgather":
                expected_gradient: Any = [float(rank + 1) for rank in dp_ranks]
                ar_observed = {}
                for index in check_indices:
                    shard_numel = buckets[index].numel // len(dp_ranks)
                    ar_observed[str(index)] = [
                        float(gradient_outputs[index][offset * shard_numel].item())
                        for offset in range(len(dp_ranks))
                    ]
                if any(
                    value != expected_gradient for value in ar_observed.values()
                ):
                    raise RuntimeError(
                        f"rank {self.rank} AG validation failed: "
                        f"{ar_observed} != {expected_gradient}"
                    )
            else:
                expected_gradient = float(sum(rank + 1 for rank in dp_ranks))
                validation_buffers = (
                    gradient_outputs
                    if case.gradient_collective == "reduce_scatter"
                    else gradient_buffers
                )
                ar_observed = {
                    str(index): float(validation_buffers[index][0].item())
                    for index in check_indices
                }
                if any(value != expected_gradient for value in ar_observed.values()):
                    raise RuntimeError(
                        f"rank {self.rank} gradient validation failed: "
                        f"{ar_observed} != {expected_gradient}"
                    )

            summaries = {
                mode: {
                    metric: summarize([sample[metric] for sample in mode_samples])
                    for metric in mode_samples[0]
                    if metric not in {"trial", "iteration"}
                }
                for mode, mode_samples in samples.items()
            }
            result = {
                "metadata": self.metadata(),
                "ep_ranks": ep_ranks,
                "dp_ranks": dp_ranks,
                "case_name": case.name,
                "trace_path": str(trace_path) if trace_path else None,
                "samples": samples,
                "summaries": summaries,
                "validation": {
                    "a2a_observed": a2a_observed,
                    "a2a_expected": a2a_expected,
                    "allreduce_observed": ar_observed,
                    "allreduce_expected": expected_gradient,
                },
            }
            # Keep the rendezvous store alive until every rank has finished
            # result construction. Without this, rank 0 can tear down the
            # store while peer heartbeat monitors are still checking it.
            dist.barrier(group=control_group)
            return result
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


def aggregate(rank_results: list[dict[str, Any]]) -> dict[str, Any]:
    modes = rank_results[0]["samples"]
    mode_summaries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        metrics = [
            metric
            for metric in rank_results[0]["samples"][mode][0]
            if metric not in {"trial", "iteration"}
        ]
        mode_summaries[mode] = {}
        for metric in metrics:
            rank_medians = [
                statistics.median(sample[metric] for sample in result["samples"][mode])
                for result in rank_results
            ]
            all_samples = [
                sample[metric]
                for result in rank_results
                for sample in result["samples"][mode]
            ]
            mode_summaries[mode][metric] = {
                "rank_medians": rank_medians,
                "median_rank_median": statistics.median(rank_medians),
                "max_rank_median": max(rank_medians),
                "all_samples": summarize(all_samples),
            }

    a2a_solo = mode_summaries["a2a_solo"]["a2a_ms"]["rank_medians"]
    ar_solo = mode_summaries["allreduce_solo"]["allreduce_ms"]["rank_medians"]
    derived: dict[str, Any] = {}
    for concurrent in ("concurrent_a2a_first", "concurrent_allreduce_first"):
        a2a_concurrent = mode_summaries[concurrent]["a2a_ms"]["rank_medians"]
        ar_concurrent = mode_summaries[concurrent]["allreduce_ms"]["rank_medians"]
        pair_concurrent = mode_summaries[concurrent]["pair_ms"]["rank_medians"]
        per_rank = []
        for rank in range(len(rank_results)):
            ideal_pair = max(a2a_solo[rank], ar_solo[rank])
            serialized_pair = a2a_solo[rank] + ar_solo[rank]
            per_rank.append(
                {
                    "rank": rank,
                    "a2a_slowdown": a2a_concurrent[rank] / a2a_solo[rank],
                    "a2a_critical_delay_ms": (
                        a2a_concurrent[rank] - a2a_solo[rank]
                    ),
                    "allreduce_slowdown": ar_concurrent[rank] / ar_solo[rank],
                    "pair_overhead_vs_interference_free": (
                        pair_concurrent[rank] / ideal_pair
                    ),
                    "speedup_vs_solo_sum": serialized_pair / pair_concurrent[rank],
                    "overlap_efficiency": (
                        serialized_pair - pair_concurrent[rank]
                    )
                    / min(a2a_solo[rank], ar_solo[rank]),
                }
            )
        derived[concurrent] = {
            "per_rank": per_rank,
            **{
                metric: summarize([row[metric] for row in per_rank])
                for metric in (
                    "a2a_slowdown",
                    "a2a_critical_delay_ms",
                    "allreduce_slowdown",
                    "pair_overhead_vs_interference_free",
                    "speedup_vs_solo_sum",
                    "overlap_efficiency",
                )
            },
        }

        # A distributed phase completes at its slowest rank. Preserve aligned
        # trial/iteration samples and take the rank maximum before computing
        # ratios; a median of rank-local ratios is diagnostic but is not the
        # collective critical path.
        sample_count = len(rank_results[0]["samples"][concurrent])
        critical_samples: list[dict[str, float]] = []
        for sample_index in range(sample_count):
            a2a = max(
                result["samples"]["a2a_solo"][sample_index]["a2a_ms"]
                for result in rank_results
            )
            allreduce = max(
                result["samples"]["allreduce_solo"][sample_index]["allreduce_ms"]
                for result in rank_results
            )
            normal = max(
                result["samples"][concurrent][sample_index]["pair_ms"]
                for result in rank_results
            )
            contended_a2a = max(
                result["samples"][concurrent][sample_index]["a2a_ms"]
                for result in rank_results
            )
            contended_allreduce = max(
                result["samples"][concurrent][sample_index]["allreduce_ms"]
                for result in rank_results
            )
            ideal = max(a2a, allreduce)
            serialized = a2a + allreduce
            critical_samples.append(
                {
                    "a2a_solo_ms": a2a,
                    "allreduce_solo_ms": allreduce,
                    "ideal_interference_free_pair_ms": ideal,
                    "normal_pair_ms": normal,
                    "a2a_slowdown": contended_a2a / a2a,
                    "a2a_critical_delay_ms": contended_a2a - a2a,
                    "allreduce_slowdown": contended_allreduce / allreduce,
                    "interference_removal_speedup_ceiling": normal / ideal,
                    "removable_fraction_of_normal": 1.0 - ideal / normal,
                    "ideal_overlap_gain_vs_serialized": serialized / ideal,
                    "achieved_gain_vs_serialized": serialized / normal,
                }
            )
        derived[concurrent]["world_critical_samples"] = critical_samples
        derived[concurrent]["world_critical"] = {
            metric: summarize([sample[metric] for sample in critical_samples])
            for metric in critical_samples[0]
        }
    return {"modes": mode_summaries, "derived": derived}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="10.12.248.176")
    parser.add_argument("--port", type=int, default=6379)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=4)
    parser.add_argument("--gpus-per-node", type=int, default=4)
    parser.add_argument(
        "--ep-size",
        type=int,
        default=4,
        help="Automatic contiguous EP group size; ignored with --rank-map-file",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        help="Optional assertion; inferred as world-size / ep-size",
    )
    parser.add_argument(
        "--rank-map-file",
        help="JSON containing ep_groups, dp_groups, and optional node_domain_map",
    )
    parser.add_argument(
        "--node-domain-map",
        help="Comma-separated NVLink domain ID per node slot, e.g. 0,1,2,3",
    )
    parser.add_argument(
        "--expected-node-ips",
        help=(
            "Comma-separated Ray node IP for each node slot. Required for real "
            "runs when an EP group spans multiple nodes, so the supplied domain "
            "labels cannot silently attach to a different STRICT_SPREAD order."
        ),
    )
    parser.add_argument("--allow-intradomain-dp", action="store_true")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=32768)
    parser.add_argument("--payload-scale", type=float, default=1.0)
    parser.add_argument(
        "--gradient-sequence",
        choices=["individual", "expert", "layer", "model", "custom"],
        default="expert",
        help=(
            "expert preserves fused routed-expert parameter boundaries; layer/model "
            "pack shape-derived tensors to --bucket-cap-mib"
        ),
    )
    parser.add_argument(
        "--gradient-dtype", choices=["bf16", "fp32"], default="bf16"
    )
    parser.add_argument(
        "--gradient-collective",
        choices=["allreduce", "reduce_scatter", "allgather"],
        default="allreduce",
        help=(
            "Gradient/FSDP collective primitive. For all-gather, bucket sizes "
            "denote the full gathered tensor; metric names retain allreduce "
            "for schema compatibility."
        ),
    )
    parser.add_argument("--bucket-cap-mib", type=float, default=128.0)
    parser.add_argument(
        "--gradient-bucket-mib",
        help="Exact comma-separated bucket sequence; overrides shape packing",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=60,
        help=(
            "Production Qwen3.5-397B depth; override deliberately when replaying "
            "the locally reduced registry preset"
        ),
    )
    parser.add_argument(
        "--layer-pattern",
        default="L,L,L,F",
        help=(
            "Forward layer-index pattern, repeated to --layers. Production "
            "Qwen3.5 traverses it as F,L,L,L during backward."
        ),
    )
    parser.add_argument(
        "--architecture-profile",
        choices=["production_exact", "repo_proxy"],
        default="production_exact",
        help="Exact production projection totals or the simplified local model proxy",
    )
    parser.add_argument(
        "--expert-layout",
        choices=["fused_gate_up", "split_w1_w3"],
        default="fused_gate_up",
        help="Production fused [2048,4096] gate_up or proxy split W1/W3 tensors",
    )
    parser.add_argument(
        "--include-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include embedding/output gradients in model sequence",
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--skip-serialized",
        action="store_true",
        help="Omit serialized sanity modes for large experiment matrices",
    )
    parser.add_argument(
        "--gate-sleep-cycles",
        type=int,
        default=20_000_000,
        help="Device-side delay before the future start event is recorded",
    )
    parser.add_argument(
        "--a2a-delay-cycles",
        type=int,
        default=0,
        help="Additional device cycles between the common gate and A2A enqueue",
    )
    parser.add_argument(
        "--gradient-delay-cycles",
        type=int,
        default=0,
        help="Additional device cycles between the common gate and gradient collective enqueue",
    )
    parser.add_argument(
        "--dry-run",
        "--print-config",
        action="store_true",
        dest="dry_run",
        help="Resolve and print configuration without Ray, NCCL, CUDA, or output writes",
    )
    parser.add_argument(
        "--output",
        default="out/qwen397b_a2a_allreduce/results.json",
    )
    parser.add_argument(
        "--profile-dir",
        help=(
            "Optional directory for one CPU+CUDA PyTorch trace per rank. "
            "Every benchmark mode is traced once after timed samples."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "world-size": args.world_size,
        "nodes": args.nodes,
        "gpus-per-node": args.gpus_per_node,
        "microbatch-size": args.microbatch_size,
        "seq-len": args.seq_len,
        "payload-scale": args.payload_scale,
        "bucket-cap-mib": args.bucket_cap_mib,
        "iterations": args.iterations,
        "trials": args.trials,
        "layers": args.layers,
    }
    invalid = {name: value for name, value in positive.items() if value <= 0}
    if invalid:
        raise ValueError(f"arguments must be positive: {invalid}")
    if args.warmups < 0:
        raise ValueError("warmups may not be negative")
    if args.gate_sleep_cycles <= 0:
        raise ValueError("gate-sleep-cycles must be positive")
    if args.a2a_delay_cycles < 0 or args.gradient_delay_cycles < 0:
        raise ValueError("launch-delay cycles may not be negative")
    if args.expected_node_ips:
        expected = [
            item.strip() for item in args.expected_node_ips.split(",") if item.strip()
        ]
        if len(expected) != args.nodes:
            raise ValueError("expected-node-ips must contain one IP per node slot")


def main() -> None:
    args = parse_args()
    validate_args(args)
    config = load_qwen_config()
    rank_map = load_rank_map(args)
    map_warnings = validate_rank_map(
        rank_map, args.nodes, args.gpus_per_node, args.allow_intradomain_dp
    )
    buckets, bucket_metadata = derive_buckets(args, config, rank_map)
    if args.gradient_collective in {"reduce_scatter", "allgather"}:
        dp_group_size = len(rank_map.dp_groups[0])
        if any(bucket.numel % dp_group_size for bucket in buckets):
            raise ValueError(
                f"{args.gradient_collective} bucket numel must divide the DP group size"
            )
    preview = build_preview(
        args, config, rank_map, map_warnings, buckets, bucket_metadata
    )
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return
    if rank_map.node_domain_map is None:
        raise ValueError(
            "A real run requires --node-domain-map (or node_domain_map in the "
            "rank-map file) to enforce EP-local/DP-cross-domain intent"
        )
    if any(len(group) > args.gpus_per_node for group in rank_map.ep_groups):
        if not args.expected_node_ips:
            raise ValueError(
                "Multi-node EP groups require --expected-node-ips so Ray bundle "
                "order is checked against the domain map"
            )

    # Import Ray only after dry-run has exited, keeping config inspection free
    # of cluster initialization and distributed side effects.
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
            "NCCL_LAUNCH_ORDER_IMPLICIT",
            "NCCL_MAX_CTAS",
            "NCCL_MIN_CTAS",
            "NCCL_MNNVL_ENABLE",
            "NCCL_PROTO",
        )
        if os.environ.get(name)
    }
    # Concurrent communicators need a deterministic global host launch order.
    # NCCL 2.26+ uses this option to keep ordering safe while permitting overlap.
    forwarded_env.setdefault("NCCL_LAUNCH_ORDER_IMPLICIT", "1")
    ray.init(
        address=f"{args.address}:{args.port}",
        namespace="qwen397b_a2a_allreduce_interference",
        runtime_env={"env_vars": forwarded_env},
    )
    expected_ips = (
        [item.strip() for item in args.expected_node_ips.split(",")]
        if args.expected_node_ips
        else None
    )
    bundles = []
    for node_slot in range(args.nodes):
        bundle = {"CPU": args.gpus_per_node, "GPU": args.gpus_per_node}
        if expected_ips:
            # Pin every logical node slot to its intended physical node. Ray
            # exposes node:<ip> as a placement resource on each worker.
            bundle[f"node:{expected_ips[node_slot]}"] = 0.001
        bundles.append(bundle)
    pg = placement_group(bundles, strategy="STRICT_SPREAD")
    actor_type = ray.remote(num_cpus=1, num_gpus=1)(A2AAllReduceRank)
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
                f"Expected {args.gpus_per_node} ranks/node: {node_counts}"
            )
        gpu_pairs = {
            (item["node_id"], tuple(item["ray_gpu_ids"])) for item in placements
        }
        if len(gpu_pairs) != args.world_size:
            raise RuntimeError(f"GPU assignment is not unique: {placements}")
        if args.expected_node_ips:
            observed_ips: list[str] = []
            for node_slot in range(args.nodes):
                slot_ips = {
                    item["node_ip"]
                    for item in placements
                    if item["node_slot"] == node_slot
                }
                if len(slot_ips) != 1:
                    raise RuntimeError(
                        f"node slot {node_slot} resolved to IPs {slot_ips}"
                    )
                observed_ips.append(next(iter(slot_ips)))
            if observed_ips != expected_ips:
                raise RuntimeError(
                    "Ray bundle order does not match --expected-node-ips: "
                    f"observed={observed_ips}, expected={expected_ips}"
                )

        master_addr, master_port = ray.get(actors[0].rendezvous.remote())
        case = Case(
            microbatch_size=args.microbatch_size,
            seq_len=args.seq_len,
            payload_scale=args.payload_scale,
            gradient_sequence=args.gradient_sequence,
            gradient_dtype=args.gradient_dtype,
            bucket_cap_mib=args.bucket_cap_mib,
            gradient_collective=args.gradient_collective,
            a2a_delay_cycles=args.a2a_delay_cycles,
            gradient_delay_cycles=args.gradient_delay_cycles,
        )
        started = time.time()
        rank_results = ray.get(
            [
                actor.run.remote(
                    args.world_size,
                    master_addr,
                    master_port,
                    asdict(rank_map),
                    asdict(case),
                    [asdict(bucket) for bucket in buckets],
                    config,
                    args.warmups,
                    args.iterations,
                    args.trials,
                    args.gate_sleep_cycles,
                    benchmark_modes(args.skip_serialized),
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
            "rank_results": rank_results,
            "aggregate": aggregate(rank_results),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        for mode, derived in output["aggregate"]["derived"].items():
            critical = derived["world_critical"]
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "gradient_collective": args.gradient_collective,
                        "a2a_mib": preview["a2a"]["mib_per_rank"],
                        "allreduce_sequence_mib": (
                            bucket_metadata["sequence_bytes"] / 2**20
                        ),
                        "a2a_slowdown_median": critical["a2a_slowdown"]["median"],
                        "a2a_critical_delay_ms_median": critical[
                            "a2a_critical_delay_ms"
                        ]["median"],
                        "allreduce_slowdown_median": critical[
                            "allreduce_slowdown"
                        ]["median"],
                        "pair_overhead_median": critical[
                            "interference_removal_speedup_ceiling"
                        ]["median"],
                        "removable_fraction_median": critical[
                            "removable_fraction_of_normal"
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
