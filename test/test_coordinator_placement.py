import pytest

from src.coordinator import _build_piper_placement_plan


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


def test_pp_outer_dp8_keeps_each_inner_group_in_one_gb200_domain():
    plan = _build_piper_placement_plan(
        pp_degree=2,
        dp_degree=8,
        pp_outer=True,
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
    assert [bundle.driver_dp_ranks for bundle in plan.bundles] == [
        (0, 2),
        (4, 6),
        (1, 3),
        (5, 7),
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
    assert plan.driver_bundle_indices == (0, 2, 0, 2, 1, 3, 1, 3)
    assert all(bundle.resources["GPU"] == 4.0 for bundle in plan.bundles)
    assert all(bundle.resources["CPU"] == 6.0 for bundle in plan.bundles)
    for bundle in plan.bundles:
        assert bundle.resources[f"node:{bundle.node_address}"] == 0.001


def test_inner_groups_that_fit_are_kept_on_individual_nodes():
    plan = _build_piper_placement_plan(
        pp_degree=2,
        dp_degree=4,
        pp_outer=False,
        ray_nodes=_target_gb200_nodes(),
    )

    assert len(plan.bundles) == 4
    assert len({bundle.hostname for bundle in plan.bundles}) == 4
    assert [bundle.actor_global_ranks for bundle in plan.bundles] == [
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    ]
    assert plan.driver_bundle_indices == (0, 1, 2, 3)


def test_non_gb200_cross_node_group_does_not_apply_domain_reasoning():
    nodes = [
        _ray_node("worker-a", "10.0.0.1", gb200=False),
        _ray_node("worker-b", "10.0.0.2", gb200=False),
    ]
    plan = _build_piper_placement_plan(
        pp_degree=1,
        dp_degree=6,
        pp_outer=True,
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
        _build_piper_placement_plan(
            pp_degree=1,
            dp_degree=12,
            pp_outer=True,
            ray_nodes=nodes,
        )


def test_placement_group_node_validation_checks_every_bundle():
    plan = _build_piper_placement_plan(
        pp_degree=2,
        dp_degree=8,
        pp_outer=True,
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
