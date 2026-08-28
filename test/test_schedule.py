"""Schedule parsing, topology derivation, and schedule-builder tests."""

import json
from pathlib import Path

import pytest

from examples.build_schedule import (
    build_afpipe_baseline_schedule,
    infer_afpipe_num_layers,
)
from src.schedule import load_schedule_directives, load_schedule_info


_BASE_SCHEDULES = Path(__file__).parents[1] / "examples" / "base-schedules"


def _load_base_schedule(name: str) -> list[dict]:
    with (_BASE_SCHEDULES / name).open(encoding="utf-8") as schedule_file:
        return json.load(schedule_file)


def _flat_order_row(directive: dict) -> list[dict]:
    assert directive["op"] == "order"
    assert all(len(slot) == 1 for slot in directive["filters"])
    return [slot[0] for slot in directive["filters"]]




def test_load_schedule_info_derives_equal_width_actor_groups(tmp_path) -> None:
    schedule_path = tmp_path / "qwen_example.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"PP": 0}, "devices": [0, 2]},
            {"op": "place", "filter": {"PP": 1}, "devices": [1, 3]},
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 4},
        ]),
        encoding="utf-8",
    )

    info = load_schedule_info(str(schedule_path))

    assert info == {
        "name": "qwen_example",
        "path": str(schedule_path),
        "num_stages": 2,
        "actor_groups": [
            {"id": 0, "devices": [0, 2], "size": 2},
            {"id": 1, "devices": [1, 3], "size": 2},
        ],
        "actor_group_count": 2,
        "actor_group_sizes": [2, 2],
        "global_ranks": [0, 1, 2, 3],
        "global_rank_to_actor_group": {
            0: {"actor_group_id": 0, "local_rank": 0},
            1: {"actor_group_id": 1, "local_rank": 0},
            2: {"actor_group_id": 0, "local_rank": 1},
            3: {"actor_group_id": 1, "local_rank": 1},
        },
        "world_size": 4,
        "pp_degree": 2,
        "num_microbatches": 4,
    }




def test_load_schedule_info_derives_unequal_actor_groups_from_all_places(tmp_path) -> None:
    schedule_path = tmp_path / "afd_placement_pp4.json"
    schedule_path.write_text(
        json.dumps([
            {
                "op": "place",
                "filter": {"PP": 0, "ATTN": "-", "FFN": "-"},
                "devices": [0, 4, 8, 10],
            },
            {"op": "place", "filter": {"ATTN": 0}, "devices": [10, 8, 4, 0]},
            {"op": "place", "filter": {"FFN": 0}, "devices": [1, 5]},
            {"op": "place", "filter": {"ATTN": 1}, "devices": [2, 6, 9, 11]},
            {"op": "place", "filter": {"FFN": 1}, "devices": [3, 7]},
            {
                "op": "place",
                "filter": {"PP": 3, "ATTN": "-", "FFN": "-"},
                "devices": [0, 4, 8, 10],
            },
        ]),
        encoding="utf-8",
    )

    info = load_schedule_info(str(schedule_path))

    assert info["num_stages"] == 4
    assert info["actor_groups"] == [
        {"id": 0, "devices": [0, 4, 8, 10], "size": 4},
        {"id": 1, "devices": [1, 5], "size": 2},
        {"id": 2, "devices": [2, 6, 9, 11], "size": 4},
        {"id": 3, "devices": [3, 7], "size": 2},
    ]
    assert info["actor_group_sizes"] == [4, 2, 4, 2]
    assert info["world_size"] == 12
    assert info["pp_degree"] == 4
    assert "topology_kind" not in info
    assert "dp_degree" not in info
    assert info["global_rank_to_actor_group"][9] == {
        "actor_group_id": 2,
        "local_rank": 2,
    }


def test_load_schedule_info_rejects_missing_global_actor_rank(tmp_path) -> None:
    schedule_path = tmp_path / "missing_rank.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"X": 0}, "devices": [0, 2]},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous global actor ranks"):
        load_schedule_info(str(schedule_path))


def test_load_schedule_directives_rejects_placeholder_microbatch_count(tmp_path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps([
            {
                "op": "split",
                "filter": {},
                "dim_name": "MB",
                "num_microbatches": "__MBS__",
            }
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="__MBS__"):
        load_schedule_directives(str(schedule_path))




def test_load_schedule_directives_rejects_nested_replicate_and_shard_filters(
    tmp_path,
) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
            {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1]},
            {
                "op": "shard",
                "filter": {"PP": 0, "EP": "*"},
                "devices": [0, 1],
            },
        ]),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="replicate and shard directives must select disjoint model regions",
    ):
        load_schedule_directives(str(schedule_path))






@pytest.mark.parametrize(
    ("op", "devices", "message"),
    [
        ("replicate", [], "non-empty devices"),
        ("shard", [False], "integer actor ranks"),
    ],
)
def test_load_schedule_directives_validates_all_communication_device_lists(
    tmp_path,
    op,
    devices,
    message,
) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps(
            [
                {"op": "place", "filter": {"X": "*"}, "devices": [0]},
                {"op": op, "filter": {"X": "*"}, "devices": devices},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_schedule_directives(str(schedule_path))


def test_load_schedule_directives_rejects_unplaced_communication_group(
    tmp_path,
) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps(
            [
                {"op": "place", "filter": {"X": "*"}, "devices": [0]},
                {"op": "place", "filter": {"Y": "*"}, "devices": [1]},
                {"op": "shard", "filter": {"Y": "*"}, "devices": [2]},
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly match a place ActorGroup"):
        load_schedule_directives(str(schedule_path))


@pytest.mark.parametrize(
    ("base_name", "expected_groups"),
    (("afd_placement_pp2.json", 2), ("afd_placement_pp4.json", 4)),
)
def test_afpipe_builds_one_complete_order_row_per_actor_group(
    base_name: str,
    expected_groups: int,
) -> None:
    placement = _load_base_schedule(base_name)
    n_layers = infer_afpipe_num_layers(placement, fallback=4)

    rows = list(
        map(
            _flat_order_row,
            build_afpipe_baseline_schedule(
                placement,
                n_mbs=2,
                n_layers=n_layers,
                expected_actor_groups=expected_groups,
            ),
        )
    )

    assert n_layers == 4
    assert len(rows) == expected_groups
    flattened = [flt for row in rows for flt in row]
    assert {flt["PASS"] for flt in flattened} == {"F", "B"}
    assert {flt["MB"] for flt in flattened} == {0, 1}
    assert any(flt.get("PP") == 0 for flt in flattened)
    assert any(flt.get("PP") == expected_groups - 1 for flt in flattened)
    assert {
        flt["FFN"]
        for flt in flattened
        if isinstance(flt.get("FFN"), int)
    } == set(range(4))


def test_afpipe_rejects_incorrect_actor_group_count() -> None:
    with pytest.raises(ValueError, match="actor-group count"):
        build_afpipe_baseline_schedule(
            _load_base_schedule("afd_placement_pp2.json"),
            n_mbs=2,
            n_layers=4,
            expected_actor_groups=4,
        )
