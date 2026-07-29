from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Callable

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .schedule import load_schedule_info
from .state import LOG_LEVEL, create_logger


_NODE_PIN_RESOURCE_FRACTION = 0.001
_GB200_DOMAIN_RE = re.compile(r"^(nvl72d\d+)-T", re.IGNORECASE)


@dataclass(frozen=True)
class PiperPlacementBundle:
    """One node-sized fragment of an innermost parallel group."""

    node_id: str
    node_address: str
    hostname: str
    nvlink_domain: str | None
    outer_rank: int
    actor_global_ranks: tuple[int, ...]
    driver_dp_ranks: tuple[int, ...]
    resources: dict[str, float]


@dataclass(frozen=True)
class PiperPlacementPlan:
    """Logical-to-Ray bundle mapping shared by drivers and GPU actors."""

    pp_degree: int
    dp_degree: int
    pp_outer: bool
    bundles: tuple[PiperPlacementBundle, ...]
    actor_bundle_indices: tuple[int, ...]
    driver_bundle_indices: tuple[int, ...]

    def actor_bundle_index(self, pp_rank: int, dp_rank: int) -> int:
        global_rank = int(pp_rank) + int(dp_rank) * self.pp_degree
        return self.actor_bundle_indices[global_rank]

    def driver_bundle_index(self, dp_rank: int) -> int:
        return self.driver_bundle_indices[int(dp_rank)]

    def describe(self) -> str:
        outer_name = "PP" if self.pp_outer else "DP/EP"
        inner_name = "DP/EP" if self.pp_outer else "PP"
        lines = [
            f"{outer_name} outer, {inner_name} inner; "
            f"PP={self.pp_degree}, DP/EP={self.dp_degree}"
        ]
        for index, bundle in enumerate(self.bundles):
            domain = (
                f", NVLink domain={bundle.nvlink_domain}"
                if bundle.nvlink_domain is not None
                else ""
            )
            lines.append(
                f"bundle {index}: outer_rank={bundle.outer_rank}, "
                f"host={bundle.hostname}{domain}, "
                f"actors={list(bundle.actor_global_ranks)}, "
                f"DP drivers={list(bundle.driver_dp_ranks)}"
            )
        return "\n".join(lines)

    def validate_bundle_nodes(self, placement_group_info: dict) -> None:
        actual = placement_group_info.get("bundles_to_node_id", {})
        mismatches = []
        for index, bundle in enumerate(self.bundles):
            actual_node_id = actual.get(index, actual.get(str(index)))
            if actual_node_id != bundle.node_id:
                mismatches.append(
                    f"bundle {index}: planned {bundle.hostname} "
                    f"({bundle.node_id}), Ray placed on {actual_node_id!r}"
                )
        if mismatches:
            raise RuntimeError(
                "Ray placement group did not honor Piper's node plan:\n"
                + "\n".join(mismatches)
            )


@dataclass(frozen=True)
class PiperPlacementGroup:
    """A Ray placement group together with Piper's bundle-index plan."""

    placement_group: Any
    plan: PiperPlacementPlan


@dataclass
class _PlacementNode:
    node_id: str
    address: str
    hostname: str
    gpu_capacity: int
    cpu_capacity: int
    node_resource: str
    nvlink_domain: str | None
    remaining_gpus: int

    @property
    def used_gpus(self) -> int:
        return self.gpu_capacity - self.remaining_gpus


def _placement_nodes(ray_nodes: list[dict]) -> list[_PlacementNode]:
    nodes: list[_PlacementNode] = []
    for raw_node in ray_nodes:
        alive = bool(raw_node.get("Alive", raw_node.get("alive", False)))
        resources = dict(raw_node.get("Resources", {}) or {})
        gpu_value = float(resources.get("GPU", 0.0))
        if not alive or gpu_value <= 0:
            continue
        gpu_capacity = int(gpu_value)
        if gpu_capacity != gpu_value:
            raise ValueError(
                "Piper placement requires an integral GPU count per Ray node, "
                f"got {gpu_value} on {raw_node.get('NodeManagerHostname')}"
            )

        address = str(raw_node.get("NodeManagerAddress", ""))
        hostname = str(
            raw_node.get("NodeManagerHostname")
            or raw_node.get("NodeName")
            or address
        )
        node_resource = f"node:{address}"
        if not address or float(resources.get(node_resource, 0.0)) <= 0:
            raise ValueError(
                f"Ray node {hostname!r} does not expose the expected "
                f"{node_resource!r} placement resource"
            )

        domain = None
        if float(resources.get("accelerator_type:GB200", 0.0)) > 0:
            match = _GB200_DOMAIN_RE.match(hostname)
            if match is not None:
                domain = match.group(1).lower()

        nodes.append(
            _PlacementNode(
                node_id=str(raw_node.get("NodeID", "")),
                address=address,
                hostname=hostname,
                gpu_capacity=gpu_capacity,
                cpu_capacity=int(float(resources.get("CPU", 0.0))),
                node_resource=node_resource,
                nvlink_domain=domain,
                remaining_gpus=gpu_capacity,
            )
        )

    if not nodes:
        raise ValueError("Piper placement found no live Ray nodes with GPUs")
    return sorted(nodes, key=lambda node: (node.hostname, node.address))


def _take_actor_slots(
    candidates: list[_PlacementNode],
    actor_count: int,
) -> list[tuple[_PlacementNode, int]]:
    """Consume actor slots, filling as few candidate nodes as possible."""

    remaining = actor_count
    chunks: list[tuple[_PlacementNode, int]] = []
    while remaining:
        available = [node for node in candidates if node.remaining_gpus > 0]
        if not available:
            raise ValueError(
                f"Unable to allocate {actor_count} actor slots from the selected nodes"
            )
        node = min(
            available,
            key=lambda item: (
                item.used_gpus != 0,
                -item.remaining_gpus,
                item.hostname,
                item.address,
            ),
        )
        take = min(remaining, node.remaining_gpus)
        node.remaining_gpus -= take
        chunks.append((node, take))
        remaining -= take
    return chunks


def _allocate_inner_group(
    nodes: list[_PlacementNode],
    inner_degree: int,
    *,
    use_gb200_domains: bool,
) -> list[tuple[_PlacementNode, int]]:
    """Prefer one node, then one GB200 NVLink domain, then generic packing."""

    single_node_candidates = [
        node for node in nodes if node.remaining_gpus >= inner_degree
    ]
    if single_node_candidates:
        node = min(
            single_node_candidates,
            key=lambda item: (
                item.used_gpus != 0,
                item.used_gpus,
                item.hostname,
                item.address,
            ),
        )
        return _take_actor_slots([node], inner_degree)

    if use_gb200_domains:
        by_domain: dict[str, list[_PlacementNode]] = {}
        for node in nodes:
            assert node.nvlink_domain is not None
            by_domain.setdefault(node.nvlink_domain, []).append(node)
        candidate_domains = [
            (domain, domain_nodes)
            for domain, domain_nodes in by_domain.items()
            if sum(node.remaining_gpus for node in domain_nodes) >= inner_degree
        ]
        if not candidate_domains:
            remaining = {
                domain: sum(node.remaining_gpus for node in domain_nodes)
                for domain, domain_nodes in sorted(by_domain.items())
            }
            raise ValueError(
                f"Cannot keep an innermost group of {inner_degree} actors inside "
                f"one GB200 NVLink domain; remaining domain capacities: {remaining}"
            )
        _, domain_nodes = min(
            candidate_domains,
            key=lambda item: (
                any(node.used_gpus for node in item[1]),
                sum(node.used_gpus for node in item[1]),
                item[0],
            ),
        )
        return _take_actor_slots(domain_nodes, inner_degree)

    if sum(node.remaining_gpus for node in nodes) < inner_degree:
        raise ValueError(
            f"Not enough remaining GPU capacity for an innermost group of "
            f"{inner_degree} actors"
        )
    return _take_actor_slots(nodes, inner_degree)


def _build_piper_placement_plan(
    *,
    pp_degree: int,
    dp_degree: int,
    pp_outer: bool,
    ray_nodes: list[dict],
) -> PiperPlacementPlan:
    pp_degree = int(pp_degree)
    dp_degree = int(dp_degree)
    if pp_degree <= 0 or dp_degree <= 0:
        raise ValueError(
            f"Piper parallel degrees must be positive, got PP={pp_degree}, DP={dp_degree}"
        )

    nodes = _placement_nodes(ray_nodes)
    world_size = pp_degree * dp_degree
    total_gpus = sum(node.gpu_capacity for node in nodes)
    if total_gpus < world_size:
        raise ValueError(
            f"Piper requires {world_size} GPUs for PP={pp_degree}, DP={dp_degree}, "
            f"but the Ray cluster exposes only {total_gpus}"
        )

    # Domain-aware cross-node grouping is intentionally specific to homogeneous
    # GB200 allocations whose hostnames expose the nvl72dNNN-TNN convention.
    use_gb200_domains = all(node.nvlink_domain is not None for node in nodes)
    outer_degree = pp_degree if pp_outer else dp_degree
    inner_degree = dp_degree if pp_outer else pp_degree

    mutable_bundles: list[dict[str, Any]] = []
    actor_bundle_indices = [-1] * world_size
    for outer_rank in range(outer_degree):
        chunks = _allocate_inner_group(
            nodes,
            inner_degree,
            use_gb200_domains=use_gb200_domains,
        )
        inner_rank = 0
        for node, actor_count in chunks:
            bundle_index = len(mutable_bundles)
            actor_global_ranks = []
            for _ in range(actor_count):
                if pp_outer:
                    pp_rank, dp_rank = outer_rank, inner_rank
                else:
                    pp_rank, dp_rank = inner_rank, outer_rank
                global_rank = pp_rank + dp_rank * pp_degree
                actor_bundle_indices[global_rank] = bundle_index
                actor_global_ranks.append(global_rank)
                inner_rank += 1
            mutable_bundles.append(
                {
                    "node": node,
                    "outer_rank": outer_rank,
                    "actor_global_ranks": tuple(actor_global_ranks),
                    "driver_dp_ranks": [],
                }
            )
        assert inner_rank == inner_degree

    if any(index < 0 for index in actor_bundle_indices):
        raise AssertionError(
            f"Internal placement error: incomplete actor map {actor_bundle_indices}"
        )

    driver_bundle_indices = []
    for dp_rank in range(dp_degree):
        if pp_outer:
            # Spread DP drivers over the PP groups while keeping each driver on
            # the node that hosts one of its own PP actors.
            driver_pp_rank = dp_rank % pp_degree
            global_rank = driver_pp_rank + dp_rank * pp_degree
        else:
            # The DP driver owns this whole outer DP group.
            global_rank = dp_rank * pp_degree
        bundle_index = actor_bundle_indices[global_rank]
        driver_bundle_indices.append(bundle_index)
        mutable_bundles[bundle_index]["driver_dp_ranks"].append(dp_rank)

    cpu_by_node: dict[str, int] = {}
    bundles: list[PiperPlacementBundle] = []
    for raw_bundle in mutable_bundles:
        node = raw_bundle["node"]
        actor_ranks = tuple(raw_bundle["actor_global_ranks"])
        driver_ranks = tuple(raw_bundle["driver_dp_ranks"])
        cpu_count = len(actor_ranks) + len(driver_ranks)
        cpu_by_node[node.address] = cpu_by_node.get(node.address, 0) + cpu_count
        resources = {
            "CPU": float(cpu_count),
            "GPU": float(len(actor_ranks)),
            node.node_resource: _NODE_PIN_RESOURCE_FRACTION,
        }
        bundles.append(
            PiperPlacementBundle(
                node_id=node.node_id,
                node_address=node.address,
                hostname=node.hostname,
                nvlink_domain=node.nvlink_domain,
                outer_rank=int(raw_bundle["outer_rank"]),
                actor_global_ranks=actor_ranks,
                driver_dp_ranks=driver_ranks,
                resources=resources,
            )
        )

    nodes_by_address = {node.address: node for node in nodes}
    for address, cpu_count in cpu_by_node.items():
        cpu_capacity = nodes_by_address[address].cpu_capacity
        if cpu_count > cpu_capacity:
            raise ValueError(
                f"Piper placement requires {cpu_count} CPUs on "
                f"{nodes_by_address[address].hostname}, but Ray exposes "
                f"only {cpu_capacity}"
            )

    return PiperPlacementPlan(
        pp_degree=pp_degree,
        dp_degree=dp_degree,
        pp_outer=bool(pp_outer),
        bundles=tuple(bundles),
        actor_bundle_indices=tuple(actor_bundle_indices),
        driver_bundle_indices=tuple(driver_bundle_indices),
    )


# Coordinator needs GPUs when using profiling to infer stage boundaries.
# Use manual stage annotations for more stable placement.
@ray.remote(num_gpus=0.1)
def run_dp_rank(
    dp_rank,
    dp_degree,
    pp_degree,
    world_size,
    training_func: Callable,
    *args,
    **kwargs,
):
    logger = create_logger("coordinator", LOG_LEVEL)
    logger.debug(f"Running DP rank {dp_rank+1} of {dp_degree}")

    os.environ["PIPER_DP_RANK"] = str(dp_rank)
    os.environ["PIPER_DP_DEGREE"] = str(dp_degree)
    os.environ["PIPER_PP_DEGREE"] = str(pp_degree)
    os.environ["PIPER_WORLD_SIZE"] = str(world_size)
    os.environ["TORCH_LOGS"] = "+graph_breaks"
    return training_func(*args, **kwargs)


@ray.remote
class PiperProgramCoordinator:
    """Central actor that coordinates all DP replicas of one pipeline."""

    def __init__(
        self,
        pp_outer: bool = False,
        schedule_directives_file: str | None = None,
    ):
        if schedule_directives_file is None:
            raise ValueError("PiperProgramCoordinator requires schedule_directives_file")
        info = load_schedule_info(schedule_directives_file)
        self.dp_degree = info["dp_degree"]
        self.pp_degree = info["pp_degree"]
        self.world_size = self.dp_degree * self.pp_degree
        self.pp_outer = pp_outer

    def run_program(self, training_func: Callable, pg, *args, **kwargs):
        from .compile import _COMPILED_DATA_ACTOR, _RANK0_ADDR_ACTOR

        logger = create_logger("coordinator", LOG_LEVEL)
        try:
            ray.kill(ray.get_actor(_RANK0_ADDR_ACTOR))
        except ValueError:
            logger.debug("No stale Ray actor named %s to kill", _RANK0_ADDR_ACTOR)
        except Exception:
            logger.exception("Failed to kill stale Ray actor named %s", _RANK0_ADDR_ACTOR)
            raise
        # Kill any stale compiled-data store from a previous run so that
        # dp_rank>0 workers cannot read outdated (e.g. wrong-model) stage data.
        try:
            ray.kill(ray.get_actor(_COMPILED_DATA_ACTOR))
        except ValueError:
            logger.debug("No stale Ray actor named %s to kill", _COMPILED_DATA_ACTOR)
        except Exception:
            logger.exception("Failed to kill stale Ray actor named %s", _COMPILED_DATA_ACTOR)
            raise

        ray_pg = getattr(pg, "placement_group", pg)
        plan = getattr(pg, "plan", None)
        return ray.get(
            [
                run_dp_rank.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=ray_pg,
                        placement_group_bundle_index=(
                            plan.driver_bundle_index(dp_rank)
                            if plan is not None
                            else (
                                dp_rank % self.pp_degree
                                if self.pp_outer
                                else dp_rank
                            )
                        ),
                    )
                ).remote(
                    dp_rank,
                    self.dp_degree,
                    self.pp_degree,
                    self.world_size,
                    training_func,
                    *args,
                    **kwargs,
                )
                for dp_rank in range(self.dp_degree)
            ]
        )


def create_piper_placement_group(
    schedule_directives_file: str,
    pp_outer: bool = False,
) -> PiperPlacementGroup:
    info = load_schedule_info(schedule_directives_file)
    plan = _build_piper_placement_plan(
        pp_degree=info["pp_degree"],
        dp_degree=info["dp_degree"],
        pp_outer=pp_outer,
        ray_nodes=ray.nodes(),
    )
    ray_pg = placement_group(
        [dict(bundle.resources) for bundle in plan.bundles],
        strategy="PACK",
    )
    return PiperPlacementGroup(placement_group=ray_pg, plan=plan)
