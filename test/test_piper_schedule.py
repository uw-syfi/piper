import json

import pytest

from src.schedule import load_schedule_directives, load_schedule_info


def test_load_schedule_directives_accepts_current_filter_shape(tmp_path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
            {
                "op": "order",
                "filters": [
                    [{"PP": 0, "MB": 0, "PASS": "F"}],
                    [{"PP": 0, "MB": 0, "PASS": "B"}],
                ],
            },
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
        ]),
        encoding="utf-8",
    )

    directives = load_schedule_directives(str(schedule_path))

    assert directives[0]["filter"] == {"PP": 0}
    assert directives[1]["filters"] == [
        [{"PP": 0, "MB": 0, "PASS": "F"}],
        [{"PP": 0, "MB": 0, "PASS": "B"}],
    ]


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


def test_load_schedule_info_allows_non_pp_placements_without_split(tmp_path) -> None:
    schedule_path = tmp_path / "afd_placement.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"PP": 0}, "devices": [0]},
            {"op": "place", "filter": {"PP": 1}, "devices": [1]},
            {"op": "place", "filter": {"ATTN": "*"}, "devices": [0]},
            {"op": "place", "filter": {"FFN": "*"}, "devices": [1]},
        ]),
        encoding="utf-8",
    )

    info = load_schedule_info(str(schedule_path))

    assert info == {
        "name": "afd_placement",
        "path": str(schedule_path),
        "num_stages": 2,
        "actor_groups": [
            {"id": 0, "devices": [0], "size": 1},
            {"id": 1, "devices": [1], "size": 1},
        ],
        "actor_group_count": 2,
        "actor_group_sizes": [1, 1],
        "global_ranks": [0, 1],
        "global_rank_to_actor_group": {
            0: {"actor_group_id": 0, "local_rank": 0},
            1: {"actor_group_id": 1, "local_rank": 0},
        },
        "world_size": 2,
        "pp_degree": 2,
        "num_microbatches": 1,
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


def test_load_schedule_directives_accepts_consistent_device_sets(tmp_path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"X": "*"}, "devices": [0, 2]},
            {"op": "replicate", "filter": {"X": "*"}, "devices": [2, 0]},
            {"op": "place", "filter": {"Y": "*"}, "devices": [1, 3]},
            {"op": "shard", "filter": {"Y": "*"}, "devices": [1, 3]},
        ]),
        encoding="utf-8",
    )

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


def test_load_schedule_directives_rejects_unsorted_shard_devices(tmp_path) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"X": "*"}, "devices": [1, 3]},
            {"op": "shard", "filter": {"X": "*"}, "devices": [3, 1]},
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ascending global actor-rank order"):
        load_schedule_directives(str(schedule_path))


@pytest.mark.parametrize("conflicting_devices", ([0], [0, 4]))
def test_load_schedule_directives_rejects_overlapping_device_sets(
    tmp_path,
    conflicting_devices,
) -> None:
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"X": "*"}, "devices": [0, 2]},
            {
                "op": "shard",
                "filter": {"Y": "*"},
                "devices": conflicting_devices,
            },
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="identical or disjoint"):
        load_schedule_directives(str(schedule_path))


@pytest.mark.parametrize(
    ("op", "devices", "message"),
    [
        ("replicate", [], "non-empty devices"),
        ("replicate", [0, 0], "must not contain duplicates"),
        ("shard", [-1], "must be non-negative"),
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
