# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import json
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
    program_driver_ids: tuple[int, ...]
    resources: dict[str, float]


@dataclass(frozen=True)
class PiperPlacementPlan:
    """Logical-to-Ray bundle mapping shared by drivers and GPU actors."""

    pp_degree: int
    bundles: tuple[PiperPlacementBundle, ...]
    actor_bundle_indices: tuple[int, ...]
    driver_bundle_indices: tuple[int, ...]
    actor_groups: tuple[tuple[int, ...], ...] = ()

    def actor_global_rank_bundle_index(self, global_rank: int) -> int:
        return self.actor_bundle_indices[int(global_rank)]

    def driver_bundle_index(self, program_driver_id: int = 0) -> int:
        return self.driver_bundle_indices[int(program_driver_id)]

    def describe(self) -> str:
        lines = [
            f"ActorGroup placement; groups={len(self.actor_groups)}, "
            f"world_size={len(self.actor_bundle_indices)}"
        ]
        for group_id, members in enumerate(self.actor_groups):
            lines.append(f"actor_group {group_id}: ranks={list(members)}")
        for index, bundle in enumerate(self.bundles):
            domain = (
                f", NVLink domain={bundle.nvlink_domain}"
                if bundle.nvlink_domain is not None
                else ""
            )
            lines.append(
                f"bundle {index}: actor_group={bundle.outer_rank}, "
                f"host={bundle.hostname}{domain}, "
                f"actors={list(bundle.actor_global_ranks)}, "
                f"program drivers={list(bundle.program_driver_ids)}"
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


def _build_actor_group_placement_plan(
    *,
    actor_groups: list[list[int]] | tuple[tuple[int, ...], ...],
    ray_nodes: list[dict],
) -> PiperPlacementPlan:
    """Place schedule-defined ActorGroups on Ray nodes.

    Every schedule device ordinal is one global distributed rank.  An
    ActorGroup is allocated as one innermost SPMD group, including the same
    GB200 NVLink-domain constraint used for large groups.
    A single program driver creates and controls all actors; Piper does not
    interpret ActorGroup width as data parallelism.
    """
    normalized_groups = tuple(
        tuple(int(rank) for rank in group) for group in actor_groups
    )
    if not normalized_groups or any(not group for group in normalized_groups):
        raise ValueError("actor_groups must contain non-empty rank lists")
    flat_ranks = [rank for group in normalized_groups for rank in group]
    if sorted(flat_ranks) != list(range(len(flat_ranks))):
        raise ValueError(
            "actor_groups must partition contiguous global ranks starting at 0; "
            f"got {sorted(flat_ranks)}"
        )

    nodes = _placement_nodes(ray_nodes)
    world_size = len(flat_ranks)
    total_gpus = sum(node.gpu_capacity for node in nodes)
    if total_gpus < world_size:
        raise ValueError(
            f"Piper requires {world_size} GPUs for schedule ActorGroups, "
            f"but the Ray cluster exposes only {total_gpus}"
        )
    use_gb200_domains = all(node.nvlink_domain is not None for node in nodes)

    mutable_bundles: list[dict[str, Any]] = []
    actor_bundle_indices = [-1] * world_size
    for group_id, members in enumerate(normalized_groups):
        chunks = _allocate_inner_group(
            nodes,
            len(members),
            use_gb200_domains=use_gb200_domains,
        )
        member_offset = 0
        for node, actor_count in chunks:
            bundle_index = len(mutable_bundles)
            actor_ranks = members[member_offset : member_offset + actor_count]
            member_offset += actor_count
            for global_rank in actor_ranks:
                actor_bundle_indices[global_rank] = bundle_index
            mutable_bundles.append(
                {
                    "node": node,
                    "outer_rank": group_id,
                    "actor_global_ranks": actor_ranks,
                    "program_driver_ids": [],
                }
            )
        assert member_offset == len(members)

    if any(index < 0 for index in actor_bundle_indices):
        raise AssertionError(
            f"Internal placement error: incomplete actor map {actor_bundle_indices}"
        )

    # The rank-0 actor and sole program driver share a bundle. This preserves
    # the existing compile master's locality.
    driver_bundle_indices = [actor_bundle_indices[0]]
    mutable_bundles[driver_bundle_indices[0]]["program_driver_ids"].append(0)

    cpu_by_node: dict[str, int] = {}
    bundles: list[PiperPlacementBundle] = []
    for raw_bundle in mutable_bundles:
        node = raw_bundle["node"]
        actor_ranks = tuple(raw_bundle["actor_global_ranks"])
        driver_ids = tuple(raw_bundle["program_driver_ids"])
        cpu_count = len(actor_ranks) + len(driver_ids)
        cpu_by_node[node.address] = cpu_by_node.get(node.address, 0) + cpu_count
        bundles.append(
            PiperPlacementBundle(
                node_id=node.node_id,
                node_address=node.address,
                hostname=node.hostname,
                nvlink_domain=node.nvlink_domain,
                outer_rank=int(raw_bundle["outer_rank"]),
                actor_global_ranks=actor_ranks,
                program_driver_ids=driver_ids,
                resources={
                    "CPU": float(cpu_count),
                    "GPU": float(len(actor_ranks)),
                    node.node_resource: _NODE_PIN_RESOURCE_FRACTION,
                },
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
        pp_degree=len(normalized_groups),
        bundles=tuple(bundles),
        actor_bundle_indices=tuple(actor_bundle_indices),
        driver_bundle_indices=tuple(driver_bundle_indices),
        actor_groups=normalized_groups,
    )


@ray.remote(num_gpus=0.1)
def run_actor_group_program(
    actor_groups: list[list[int]],
    world_size: int,
    training_func: Callable,
    *args,
    **kwargs,
):
    """Run one program driver that owns every schedule-defined ActorGroup."""
    # Ray workers may be reused. Ensure retired launch metadata cannot leak in
    # from an earlier task and influence execution.
    os.environ.pop("PIPER_DP_RANK", None)
    os.environ.pop("PIPER_DP_DEGREE", None)
    os.environ["PIPER_PP_DEGREE"] = str(len(actor_groups))
    os.environ["PIPER_WORLD_SIZE"] = str(world_size)
    os.environ["PIPER_ACTOR_GROUPS"] = json.dumps(actor_groups)
    os.environ["TORCH_LOGS"] = "+graph_breaks"
    return training_func(*args, **kwargs)


@ray.remote
class PiperProgramCoordinator:
    """Central actor that runs one program over all schedule ActorGroups."""

    def __init__(
        self,
        pp_outer: bool = False,
        schedule_directives_file: str | None = None,
    ):
        if schedule_directives_file is None:
            raise ValueError("PiperProgramCoordinator requires schedule_directives_file")
        info = load_schedule_info(schedule_directives_file)
        self.pp_degree = info["pp_degree"]
        self.actor_groups = [
            list(group["devices"]) for group in info["actor_groups"]
        ]
        self.world_size = int(info["world_size"])
        # Retained as a source-compatible constructor option. Placement is
        # determined solely by explicit ActorGroup memberships.
        self.pp_outer = pp_outer

    def run_program(self, training_func: Callable, pg, *args, **kwargs):
        ray_pg = getattr(pg, "placement_group", pg)
        plan = getattr(pg, "plan", None)
        bundle_index = plan.driver_bundle_index() if plan is not None else 0
        return [ray.get(
            run_actor_group_program.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=ray_pg,
                    placement_group_bundle_index=bundle_index,
                )
            ).remote(
                self.actor_groups,
                self.world_size,
                training_func,
                *args,
                **kwargs,
            )
        )]


def create_piper_placement_group(
    schedule_directives_file: str,
    pp_outer: bool = False,
    node_addresses: tuple[str, ...] | list[str] | None = None,
) -> PiperPlacementGroup:
    # ``pp_outer`` remains accepted for API compatibility. Explicit place
    # device lists now fully define topology and placement grouping.
    del pp_outer
    info = load_schedule_info(schedule_directives_file)
    ray_nodes = ray.nodes()
    if node_addresses is not None:
        requested = {str(address) for address in node_addresses}
        ray_nodes = [
            node
            for node in ray_nodes
            if str(node.get("NodeManagerAddress", "")) in requested
        ]
        found = {
            str(node.get("NodeManagerAddress", "")) for node in ray_nodes
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(
                "Requested Piper placement nodes are not live Ray nodes: "
                f"{missing}"
            )
    plan = _build_actor_group_placement_plan(
        actor_groups=[group["devices"] for group in info["actor_groups"]],
        ray_nodes=ray_nodes,
    )
    ray_pg = placement_group(
        [dict(bundle.resources) for bundle in plan.bundles],
        strategy="PACK",
    )
    return PiperPlacementGroup(placement_group=ray_pg, plan=plan)
