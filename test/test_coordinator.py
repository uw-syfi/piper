"""Coordinator placement and topology tests."""

import json
import os

import pytest

from src.coordinator import (
    PiperProgramCoordinator,
    _build_actor_group_placement_plan,
    create_piper_placement_group,
    run_actor_group_program,
)


def _ray_node(
    hostname: str,
    address: str,
    *,
    gpus: int = 4,
    cpus: int = 64,
    gb200: bool = True,
) -> dict:
    resources = {
        "CPU": float(cpus),
        "GPU": float(gpus),
        f"node:{address}": 1.0,
    }
    if gb200:
        resources["accelerator_type:GB200"] = 1.0
    return {
        "NodeID": f"id-{hostname}",
        "Alive": True,
        "NodeManagerAddress": address,
        "NodeManagerHostname": hostname,
        "Resources": resources,
    }


def _target_gb200_nodes() -> list[dict]:
    return [
        _ray_node("nvl72d052-T02", "10.12.250.179"),
        _ray_node("nvl72d052-T03", "10.12.251.163"),
        _ray_node("nvl72d057-T02", "10.12.248.129"),
        _ray_node("nvl72d057-T05", "10.12.251.201"),
    ]


def test_equal_width_replicated_stages_keep_each_group_in_one_gb200_domain():
    plan = _build_actor_group_placement_plan(
        actor_groups=[tuple(range(0, 16, 2)), tuple(range(1, 16, 2))],
        ray_nodes=_target_gb200_nodes(),
    )

    assert [bundle.hostname for bundle in plan.bundles] == [
        "nvl72d052-T02",
        "nvl72d052-T03",
        "nvl72d057-T02",
        "nvl72d057-T05",
    ]
    assert [bundle.nvlink_domain for bundle in plan.bundles] == [
        "nvl72d052",
        "nvl72d052",
        "nvl72d057",
        "nvl72d057",
    ]
    assert [bundle.actor_global_ranks for bundle in plan.bundles] == [
        (0, 2, 4, 6),
        (8, 10, 12, 14),
        (1, 3, 5, 7),
        (9, 11, 13, 15),
    ]
    assert [bundle.program_driver_ids for bundle in plan.bundles] == [
        (0,),
        (),
        (),
        (),
    ]
    assert plan.actor_bundle_indices == (
        0,
        2,
        0,
        2,
        0,
        2,
        0,
        2,
        1,
        3,
        1,
        3,
        1,
        3,
        1,
        3,
    )
    assert plan.driver_bundle_indices == (0,)
    assert all(bundle.resources["GPU"] == 4.0 for bundle in plan.bundles)
    assert [bundle.resources["CPU"] for bundle in plan.bundles] == [
        5.0,
        4.0,
        4.0,
        4.0,
    ]
    for bundle in plan.bundles:
        assert bundle.resources[f"node:{bundle.node_address}"] == 0.001


def test_non_gb200_cross_node_group_does_not_apply_domain_reasoning():
    nodes = [
        _ray_node("worker-a", "10.0.0.1", gb200=False),
        _ray_node("worker-b", "10.0.0.2", gb200=False),
    ]
    plan = _build_actor_group_placement_plan(
        actor_groups=[tuple(range(6))],
        ray_nodes=nodes,
    )

    assert [bundle.hostname for bundle in plan.bundles] == [
        "worker-a",
        "worker-b",
    ]
    assert [len(bundle.actor_global_ranks) for bundle in plan.bundles] == [4, 2]
    assert all(bundle.nvlink_domain is None for bundle in plan.bundles)


def test_gb200_group_fails_instead_of_crossing_nvlink_domains():
    nodes = [
        _ray_node("nvl72d001-T01", "10.0.0.1"),
        _ray_node("nvl72d001-T02", "10.0.0.2"),
        _ray_node("nvl72d002-T01", "10.0.0.3"),
    ]

    with pytest.raises(ValueError, match="one GB200 NVLink domain"):
        _build_actor_group_placement_plan(
            actor_groups=[tuple(range(12))],
            ray_nodes=nodes,
        )


def test_placement_group_node_validation_checks_every_bundle():
    plan = _build_actor_group_placement_plan(
        actor_groups=[tuple(range(0, 16, 2)), tuple(range(1, 16, 2))],
        ray_nodes=_target_gb200_nodes(),
    )
    actual = {
        "bundles_to_node_id": {
            index: bundle.node_id for index, bundle in enumerate(plan.bundles)
        }
    }
    plan.validate_bundle_nodes(actual)

    actual["bundles_to_node_id"][2] = "wrong-node"
    with pytest.raises(RuntimeError, match="bundle 2"):
        plan.validate_bundle_nodes(actual)


def test_unequal_actor_groups_are_placed_from_exact_memberships():
    plan = _build_actor_group_placement_plan(
        actor_groups=[(0, 2, 4, 5), (1, 3)],
        ray_nodes=_target_gb200_nodes(),
    )

    assert plan.pp_degree == 2
    assert plan.actor_groups == ((0, 2, 4, 5), (1, 3))
    assert [bundle.actor_global_ranks for bundle in plan.bundles] == [
        (0, 2, 4, 5),
        (1, 3),
    ]
    assert plan.actor_bundle_indices == (0, 1, 0, 1, 0, 0)
    assert plan.driver_bundle_indices == (0,)
    assert plan.actor_global_rank_bundle_index(3) == 1




def test_program_coordinator_derives_only_actor_group_topology(tmp_path):
    schedule_path = tmp_path / "replicated_pipeline.json"
    schedule_path.write_text(
        json.dumps(
            [
                {"op": "place", "filter": {"PP": 0}, "devices": [0, 2]},
                {"op": "place", "filter": {"PP": 1}, "devices": [1, 3]},
            ]
        ),
        encoding="utf-8",
    )
    coordinator_class = PiperProgramCoordinator.__ray_metadata__.modified_class

    coordinator = coordinator_class(schedule_directives_file=str(schedule_path))

    assert coordinator.actor_groups == [[0, 2], [1, 3]]
    assert coordinator.world_size == 4
    assert not hasattr(coordinator, "dp_degree")


def test_program_driver_removes_retired_dp_environment(monkeypatch):
    monkeypatch.setenv("PIPER_DP_RANK", "7")
    monkeypatch.setenv("PIPER_DP_DEGREE", "8")

    observed = run_actor_group_program._function(
        [[0, 2], [1, 3]],
        4,
        lambda: dict(os.environ),
    )

    assert "PIPER_DP_RANK" not in observed
    assert "PIPER_DP_DEGREE" not in observed
    assert json.loads(observed["PIPER_ACTOR_GROUPS"]) == [[0, 2], [1, 3]]
    assert observed["PIPER_PP_DEGREE"] == "2"
    assert observed["PIPER_WORLD_SIZE"] == "4"


