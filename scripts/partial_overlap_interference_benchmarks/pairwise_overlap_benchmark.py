"""Trace-grounded definitions and safety checks for the overlap benchmark.

This module deliberately has no torch, NCCL, or Ray imports. A fresh worker
can therefore sanitize its environment before importing any library capable of
creating a communicator.
"""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, MutableMapping, Sequence


MODEL_NAME = "Qwen3-30B-A3B-empirical1to1-L4"
TRACE_SEQUENCE_LENGTH = 12_288
DEFAULT_SEQUENCE_LENGTHS = (12_288, 18_432, 24_576, 36_864, 49_152)
DEFAULT_COVERAGE_TARGETS = (0, 1, 5, 10, 25, 50, 75, 90, 100)
DEFAULT_TRACE_DIR = Path("out/20260809_132048")
NCCL_DEBUG_VARIABLE_PREFIX = "NCCL_DEBUG"

MODEL_CONFIG = {
    "dim": 2_048,
    "n_layers": 4,
    "n_heads": 32,
    "n_kv_heads": 4,
    "head_dim": 128,
    "hidden_dim": 6_144,
    "moe_inter_dim": 1_344,
    "num_experts": 128,
    "top_k": 8,
    "vocab_size": 151_936,
    "dtype": "BFloat16",
    "batch_size": 1,
    "microbatches": 4,
}


@dataclass(frozen=True)
class OperationAnchor:
    operation_id: str
    family: str
    kernel_match: str
    stream_role: str
    dtype: str | None = None
    group_size: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class PairCase:
    case_id: str
    left_family: str
    right_family: str
    evidence_pairs: tuple[str, ...]
    trace_pair_count: int
    payload_pair_count: int
    requires_bf16_a2a_payload: bool = False
    notes: str = ""


@dataclass(frozen=True)
class MultiCase:
    case_id: str
    victim_family: str
    aggressor_families: tuple[str, ...]
    evidence_event: str
    trace_shape_count: int
    notes: str


@dataclass(frozen=True)
class PPOuterTopology:
    """PP=2 outer, DP=EP=8 inner, on two eight-GPU nodes."""

    node_ips: tuple[str, str]
    gpus_per_node: int = 8
    physical_pp: int = 2
    dp: int = 8
    ep: int = 8

    @property
    def world_size(self) -> int:
        return self.physical_pp * self.dp

    def global_rank(self, dp_rank: int, physical_pp_rank: int) -> int:
        if not 0 <= dp_rank < self.dp:
            raise ValueError(f"invalid DP rank {dp_rank}")
        if not 0 <= physical_pp_rank < self.physical_pp:
            raise ValueError(f"invalid physical PP rank {physical_pp_rank}")
        return dp_rank * self.physical_pp + physical_pp_rank

    def physical_pp_rank(self, global_rank: int) -> int:
        self._check_rank(global_rank)
        return global_rank % self.physical_pp

    def dp_rank(self, global_rank: int) -> int:
        self._check_rank(global_rank)
        return global_rank // self.physical_pp

    def node_ip_for_rank(self, global_rank: int) -> str:
        return self.node_ips[self.physical_pp_rank(global_rank)]

    @property
    def dp_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        groups = tuple(
            tuple(self.global_rank(dp_rank, pp_rank) for dp_rank in range(self.dp))
            for pp_rank in range(self.physical_pp)
        )
        return (groups[0], groups[1])

    @property
    def ep_groups(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # The traced schedule has DP8 and EP8 over the same physical actors but
        # uses distinct communicators and streams.
        return self.dp_groups

    @property
    def pp_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                self.global_rank(dp_rank, 0),
                self.global_rank(dp_rank, 1),
            )
            for dp_rank in range(self.dp)
        )

    def rank_record(self, global_rank: int) -> dict[str, Any]:
        pp_rank = self.physical_pp_rank(global_rank)
        logical_pp = (0, 3) if pp_rank == 0 else (1, 2)
        return {
            "global_rank": global_rank,
            "physical_pp_rank": pp_rank,
            "logical_pp_stages": logical_pp,
            "dp_rank": self.dp_rank(global_rank),
            "ep_rank": self.dp_rank(global_rank),
            "node_ip": self.node_ip_for_rank(global_rank),
            "node_local_slot": self.dp_rank(global_rank),
        }

    def validate(self) -> None:
        if len(self.node_ips) != 2 or len(set(self.node_ips)) != 2:
            raise ValueError("PP-outer topology requires two distinct node IPs")
        if self.gpus_per_node != 8 or self.dp != 8 or self.ep != 8:
            raise ValueError("trace topology is fixed at two nodes and DP=EP=8")
        expected = set(range(self.world_size))
        for label, groups in (("DP", self.dp_groups), ("EP", self.ep_groups)):
            flattened = [rank for group in groups for rank in group]
            if set(flattened) != expected or len(flattened) != self.world_size:
                raise ValueError(f"{label} groups do not partition the world")
            for group in groups:
                nodes = {self.node_ip_for_rank(rank) for rank in group}
                if len(nodes) != 1:
                    raise ValueError(f"{label} group unexpectedly crosses nodes")
        for first, second in self.pp_pairs:
            if self.node_ip_for_rank(first) == self.node_ip_for_rank(second):
                raise ValueError("PP pair must cross nodes")

    def _check_rank(self, global_rank: int) -> None:
        if not 0 <= global_rank < self.world_size:
            raise ValueError(f"invalid global rank {global_rank}")


OPERATION_ANCHORS = {
    "C-DENSE": OperationAnchor(
        "C-DENSE", "dense_gemm",
        "nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NTT", "compute",
        notes="Trace-real dense BF16 GEMM.",
    ),
    "C-GROUPED": OperationAnchor(
        "C-GROUPED", "grouped_expert_gemm",
        "GemmUniversal<GroupProblemShape", "compute",
        notes=(
            "SM90 CUTLASS BF16 grouped expert GEMM; kernel-name SHA1 prefix "
            "28e33c0c27e1."
        ),
    ),
    "C-BW": OperationAnchor(
        "C-BW", "expert_pointwise",
        "triton_poi_fused_mul_silu_silu_backward_0", "compute",
        notes="Bandwidth-oriented expert activation backward.",
    ),
    "X-INDEX-PUT": OperationAnchor(
        "X-INDEX-PUT", "a2a_aux", "index_elementwise_kernel<128, 4", "ep_aux",
        notes="Routing index-put inside a tagged A2A node.",
    ),
    "A2A-DATA": OperationAnchor(
        "A2A-DATA", "a2a", "ncclDevKernel_SendRecv", "ep",
        dtype="BFloat16", group_size=8,
        notes="Tagged ragged dispatch/combine payload, never Long metadata.",
    ),
    "AG": OperationAnchor(
        "AG", "all_gather", "ncclDevKernel_AllGather_RING_LL", "dp",
        dtype="BFloat16", group_size=8,
    ),
    "RS": OperationAnchor(
        "RS", "reduce_scatter",
        "ncclDevKernel_ReduceScatter_Sum_f32_RING_LL", "dp",
        dtype="Float", group_size=8,
    ),
    "P2P-SEND": OperationAnchor(
        "P2P-SEND", "p2p_send", "ncclDevKernel_SendRecv", "pp_send",
        dtype="BFloat16", group_size=2,
        notes="25,165,824 elements (48 MiB), cross-node.",
    ),
    "P2P-RECV": OperationAnchor(
        "P2P-RECV", "p2p_recv", "ncclDevKernel_SendRecv", "pp_recv",
        dtype="BFloat16", group_size=2,
        notes="25,165,824 elements (48 MiB), cross-node.",
    ),
}


PAIR_CASES = (
    PairCase("A01", "compute", "p2p_send", (
        "d2.p1.i3.e00314--d2.p1.i3.e00437",
        "d6.p1.i3.e00313--d6.p1.i3.e00403",
        "d7.p1.i3.e00313--d7.p1.i3.e00404",
    ), 5_536, 5_536),
    PairCase("A02", "compute", "p2p_recv", (
        "d6.p0.i2.e01356--d6.p0.i2.e01394",
        "d6.p0.i2.e01356--d6.p0.i2.e01367",
        "d5.p0.i3.e01357--d5.p0.i3.e01447",
    ), 2_062, 2_062),
    PairCase("A03", "compute", "a2a", (
        "d3.p1.i3.e00594--d3.p1.i3.e00604",
        "d2.p1.i3.e00876--d2.p1.i3.e00895",
        "d6.p1.i3.e00744--d6.p1.i3.e00745",
    ), 2_520, 2_520, True),
    PairCase("A04", "compute", "ag", (
        "d5.p0.i3.e00881--d5.p0.i3.e00894",
        "d5.p1.i2.e00850--d5.p1.i2.e00904",
        "d2.p1.i3.e00853--d2.p1.i3.e00896",
    ), 10_758, 10_758),
    PairCase("A05", "compute", "rs", (
        "d3.p1.i2.e01434--d3.p1.i2.e01445",
        "d4.p0.i2.e01559--d4.p0.i2.e01560",
        "d4.p0.i2.e01559--d4.p0.i2.e01561",
    ), 1_212, 1_212),
    PairCase("A06", "a2a_aux", "a2a", (
        "d0.p1.i2.e00583--d0.p1.i2.e00591",
    ), 4_102, 4_102, True),
    PairCase("A07", "a2a_aux", "p2p_send", (
        "d2.p1.i3.e00314--d2.p1.i3.e00384",
    ), 2_138, 2_138),
    PairCase("A08", "a2a_aux", "p2p_recv", (
        "d5.p0.i3.e01357--d5.p0.i3.e01381",
    ), 915, 915),
    PairCase("A09", "a2a_aux", "compute", (
        "d6.p1.i3.e00924--d6.p1.i3.e00925",
    ), 1_417, 1_417),
    PairCase("A10", "a2a_aux", "ag", (
        "d6.p0.i2.e00563--d6.p0.i2.e00579",
    ), 3_362, 3_362),
    PairCase("A11", "a2a_aux", "rs", (
        "d2.p0.i3.e01283--d2.p0.i3.e01284",
    ), 96, 96),
    PairCase("A12", "a2a", "p2p_send", (
        "d6.p0.i3.e00238--d6.p0.i3.e00273",
    ), 291, 214, True),
    PairCase("A13", "a2a", "p2p_recv", (
        "d3.p1.i3.e00348--d3.p1.i3.e00385",
    ), 144, 112, True),
    PairCase("A14", "a2a", "ag", (
        "d2.p1.i2.e00068--d2.p1.i2.e00076",
    ), 630, 494, True),
    PairCase("A15", "a2a", "rs", (
        "d0.p0.i2.e01254--d0.p0.i2.e01255",
    ), 32, 32, True),
    PairCase("A16", "ag", "p2p_send", (
        "d3.p0.i3.e00152--d3.p0.i3.e00160",
    ), 401, 401),
    PairCase("A17", "ag", "p2p_recv", (
        "d2.p1.i3.e00306--d2.p1.i3.e00313",
    ), 358, 358),
    PairCase("A18", "rs", "p2p_send", (
        "d3.p0.i2.e01347--d3.p0.i2.e01349",
    ), 16, 16, notes="Rare trace-real edge case."),
    PairCase("A19", "rs", "p2p_recv", (
        "d4.p0.i3.e01347--d4.p0.i3.e01349",
    ), 16, 16, notes="Rare trace-real edge case."),
    PairCase("A20", "p2p_send", "p2p_recv", (
        "d2.p1.i2.e01191--d2.p1.i2.e01193",
    ), 184, 184, notes="Two distinct cross-node transfers and peer endpoints."),
)


MULTI_CASES = (
    MultiCase("M01", "dense_gemm", ("p2p_recv", "ag"),
              "d3.p0.i2.e00463", 296,
              "64/64 dense instances are fully covered."),
    MultiCase("M02", "grouped_expert_gemm", ("p2p_send", "p2p_recv"),
              "d3.p1.i3.e00377", 208,
              "Grouped GEMM, correcting the draft dense-GEMM label."),
    MultiCase("M03", "sdpa_backward", ("ag", "a2a"),
              "d0.p1.i3.e00954", 955,
              "7.411 ms flash-attention backward victim."),
    MultiCase("M04", "a2a", ("p2p_send", "p2p_recv"),
              "d3.p1.i3.e00385", 16,
              "Count includes BF16 payload shapes only."),
    MultiCase("M05a", "ag", ("grouped_expert_gemm", "elementwise_add"),
              "d0.p1.i3.e00564", 28,
              "One compute stream, maximum simultaneous width one."),
    MultiCase("M05b", "ag", (
        "grouped_expert_gemm", "grouped_gemm_prepare", "grouped_expert_gemm",
    ), "d6.p0.i3.e01142", 52,
              "One compute stream, maximum simultaneous width one."),
)


def nccl_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {
        key: source[key] for key in sorted(source) if key.startswith("NCCL_")
    }


def nccl_debug_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return observability-only NCCL debug variables allowed by the suite."""

    return {
        key: value
        for key, value in nccl_environment(environ).items()
        if key == NCCL_DEBUG_VARIABLE_PREFIX
        or key.startswith(f"{NCCL_DEBUG_VARIABLE_PREFIX}_")
    }


def nccl_behavior_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return NCCL variables that can alter runtime behavior or selection."""

    debug = nccl_debug_environment(environ)
    return {
        key: value
        for key, value in nccl_environment(environ).items()
        if key not in debug
    }


def sanitize_nccl_environment(
    environ: MutableMapping[str, str] | None = None,
    *,
    phase: str,
) -> dict[str, Any]:
    """Remove NCCL behavior overrides while retaining debug observability."""

    target = os.environ if environ is None else environ
    inherited = nccl_environment(target)
    inherited_debug = nccl_debug_environment(target)
    inherited_behavior = nccl_behavior_environment(target)
    for key in inherited_behavior:
        target.pop(key, None)
    effective_behavior = nccl_behavior_environment(target)
    retained_debug = nccl_debug_environment(target)
    if effective_behavior:
        raise RuntimeError(
            f"failed to remove NCCL behavior overrides: {effective_behavior}"
        )
    return {
        "phase": phase,
        "pid": os.getpid(),
        "inherited_nccl_environment": inherited,
        "inherited_overrides": inherited_behavior,
        "inherited_debug_variables": inherited_debug,
        "removed_names": sorted(inherited_behavior),
        "retained_debug_variables": retained_debug,
        "effective_nccl_environment": nccl_environment(target),
        "effective_overrides": effective_behavior,
        "uses_nccl_library_behavior_defaults": True,
        "launch_order_implicit_enabled": False,
        "launch_order_basis": (
            "NCCL_LAUNCH_ORDER_IMPLICIT absent; NCCL library default applies"
        ),
    }


def assert_default_nccl_environment(
    environ: Mapping[str, str] | None = None,
) -> None:
    overrides = nccl_behavior_environment(environ)
    if overrides:
        raise RuntimeError(
            "communicator creation requires no NCCL behavior overrides, "
            f"found {overrides}"
        )


def validate_case_catalog() -> None:
    pair_ids = [case.case_id for case in PAIR_CASES]
    if pair_ids != [f"A{index:02d}" for index in range(1, 21)]:
        raise ValueError(f"pair catalog is incomplete or unordered: {pair_ids}")
    multi_ids = [case.case_id for case in MULTI_CASES]
    if multi_ids != ["M01", "M02", "M03", "M04", "M05a", "M05b"]:
        raise ValueError(f"multi catalog is incomplete: {multi_ids}")
    evidence = [
        pair_id for case in PAIR_CASES for pair_id in case.evidence_pairs
    ]
    duplicates = sorted(
        pair_id for pair_id, count in Counter(evidence).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate evidence pair IDs: {duplicates}")


def build_suite_manifest(
    topology: PPOuterTopology,
    *,
    sequence_lengths: Sequence[int] = DEFAULT_SEQUENCE_LENGTHS,
    coverage_targets: Sequence[int] = DEFAULT_COVERAGE_TARGETS,
) -> dict[str, Any]:
    validate_case_catalog()
    topology.validate()
    if not sequence_lengths or any(value <= 0 for value in sequence_lengths):
        raise ValueError("sequence lengths must be positive")
    if not coverage_targets or any(
        value < 0 or value > 100 for value in coverage_targets
    ):
        raise ValueError("coverage targets must be in [0, 100]")
    return {
        "schema_version": 1,
        "model": MODEL_NAME,
        "model_config": MODEL_CONFIG,
        "trace_sequence_length": TRACE_SEQUENCE_LENGTH,
        "sequence_lengths": list(sequence_lengths),
        "coverage_targets_pct": list(coverage_targets),
        "dispatch_modes": ["victim_first", "aggressor_first"],
        "pair_orientations": "both",
        "nccl_environment_policy": {
            "mode": "library_behavior_defaults_with_debug_observability",
            "allowed_overrides": [],
            "allowed_debug_variables": ["NCCL_DEBUG", "NCCL_DEBUG_*"],
            "launch_order_implicit": "disabled_by_default",
            "worker_requirement": (
                "fresh process; sanitize before torch import and communicator "
                "creation"
            ),
        },
        "topology": {
            **asdict(topology),
            "world_size": topology.world_size,
            "rank_order": "global_rank = dp_rank * physical_pp + physical_pp_rank",
            "ranks": [
                topology.rank_record(rank) for rank in range(topology.world_size)
            ],
            "dp_groups": topology.dp_groups,
            "ep_groups": topology.ep_groups,
            "pp_pairs": topology.pp_pairs,
        },
        "operation_anchors": {
            key: asdict(value) for key, value in OPERATION_ANCHORS.items()
        },
        "pair_cases": [asdict(case) for case in PAIR_CASES],
        "multi_cases": [asdict(case) for case in MULTI_CASES],
        "visualization_policy": {
            "required_for_every_analysis": True,
            "source_table_required": True,
            "sample_counts_required": True,
            "uncertainty_required_where_applicable": True,
        },
    }


def validate_trace_anchors(trace_dir: Path = DEFAULT_TRACE_DIR) -> dict[str, Any]:
    """Validate the catalog against the trace-derived evidence files."""

    pair_path = trace_dir / "kernel_overlap_pairs.csv"
    cases_path = trace_dir / "kernel_overlap_cases.jsonl"
    schedule_path = trace_dir / "dualpipev_pp2_mbs4.json"
    for path in (pair_path, cases_path, schedule_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    csv.field_size_limit(sys.maxsize)
    expected_pair_ids = {
        pair_id: case
        for case in PAIR_CASES
        for pair_id in case.evidence_pairs
    }
    pair_rows: dict[str, list[dict[str, str]]] = {
        pair_id: [] for pair_id in expected_pair_ids
    }
    with pair_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pair_id = row["pair_id"]
            if pair_id in pair_rows and row["scope"] == "steady":
                pair_rows[pair_id].append(row)

    errors: list[str] = []
    payload_checks = 0
    for pair_id, rows in pair_rows.items():
        if len(rows) != 2:
            errors.append(f"{pair_id}: expected two directed rows, got {len(rows)}")
            continue
        case = expected_pair_ids[pair_id]
        if case.requires_bf16_a2a_payload:
            a2a_sides = []
            for row in rows:
                for side in ("origin", "partner"):
                    if row[f"{side}_category"] == "a2a":
                        a2a_sides.append(json.loads(row[f"{side}_signature"]))
            if not a2a_sides:
                errors.append(f"{pair_id}: expected an A2A participant")
            for signature in a2a_sides:
                payload_checks += 1
                if not (
                    signature.get("collective") == "all_to_allv"
                    and signature.get("dtype") == "BFloat16"
                    and int(signature.get("group_size", -1)) == 8
                    and int(signature.get("in_msg_nelems", -1)) > 1_000_000
                    and int(signature.get("out_msg_nelems", -1)) > 1_000_000
                ):
                    errors.append(
                        f"{pair_id}: selected A2A is not a tagged payload: "
                        f"{signature}"
                    )

    expected_events = {case.evidence_event for case in MULTI_CASES}
    found_events: set[str] = set()
    with cases_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if (
                record["event_id"] in expected_events
                and record["scope"] == "steady"
            ):
                found_events.add(record["event_id"])
    for event_id in sorted(expected_events - found_events):
        errors.append(f"missing steady multi-aggressor event {event_id}")

    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if "bucket_size" in json.dumps(schedule):
        errors.append("generated schedule unexpectedly enables bucketing")
    replicas = [item for item in schedule if item.get("op") == "replicate"]
    if len(replicas) != 4:
        errors.append(f"expected four replicate directives, got {len(replicas)}")
    for item in replicas:
        if item.get("shard_grads") is not True:
            errors.append(f"replicate directive does not shard grads: {item}")
        if item.get("shard_params") is not True:
            errors.append(f"replicate directive does not shard params: {item}")
        if item.get("gather_stream") != item.get("reduce_stream"):
            errors.append(f"AG and RS are not on the same stream: {item}")

    if errors:
        raise ValueError("trace anchor validation failed:\n- " + "\n- ".join(errors))
    return {
        "trace_dir": str(trace_dir),
        "pair_anchor_count": len(pair_rows),
        "directed_pair_row_count": sum(len(rows) for rows in pair_rows.values()),
        "bf16_a2a_payload_signature_checks": payload_checks,
        "multi_anchor_count": len(found_events),
        "schedule": {
            "shard_grads": True,
            "shard_params": True,
            "bucketing": False,
            "all_gather_reduce_scatter_same_stream": True,
        },
        "passed": True,
    }


def validate_observed_placement(
    topology: PPOuterTopology,
    placements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    topology.validate()
    by_rank = {int(item["global_rank"]): item for item in placements}
    expected_ranks = set(range(topology.world_size))
    if set(by_rank) != expected_ranks:
        raise ValueError(
            f"placement ranks mismatch: expected {expected_ranks}, got {set(by_rank)}"
        )
    gpu_keys: set[tuple[str, tuple[str, ...]]] = set()
    node_counts: Counter[str] = Counter()
    for rank, item in by_rank.items():
        observed_ip = str(item["node_ip"])
        expected_ip = topology.node_ip_for_rank(rank)
        if observed_ip != expected_ip:
            raise ValueError(
                f"rank {rank} expected node {expected_ip}, observed {observed_ip}"
            )
        gpu_ids = tuple(str(value) for value in item["ray_gpu_ids"])
        key = (observed_ip, gpu_ids)
        if key in gpu_keys:
            raise ValueError(f"GPU assigned more than once: {key}")
        gpu_keys.add(key)
        node_counts[observed_ip] += 1
        if item.get("effective_nccl_overrides"):
            raise ValueError(
                f"rank {rank} retained NCCL overrides: "
                f"{item['effective_nccl_overrides']}"
            )
    if node_counts != Counter({ip: 8 for ip in topology.node_ips}):
        raise ValueError(f"expected eight actors per node, got {node_counts}")
    return {
        "passed": True,
        "world_size": topology.world_size,
        "node_actor_counts": dict(sorted(node_counts.items())),
        "unique_gpu_assignments": len(gpu_keys),
        "pp_pairs_cross_node": all(
            by_rank[first]["node_ip"] != by_rank[second]["node_ip"]
            for first, second in topology.pp_pairs
        ),
        "dp_ep_groups_within_node": all(
            len({by_rank[rank]["node_ip"] for rank in group}) == 1
            for group in topology.dp_groups
        ),
        "all_workers_use_default_nccl_behavior_environment": True,
    }
