import json
from pathlib import Path

import pytest

from examples.build_schedule import (
    build_afpipe_baseline_schedule,
    infer_afpipe_num_layers,
)


_BASE_SCHEDULES = Path(__file__).parents[1] / "examples" / "base-schedules"


def _load(name: str) -> list[dict]:
    with (_BASE_SCHEDULES / name).open(encoding="utf-8") as schedule_file:
        return json.load(schedule_file)


def _flat_row(directive: dict) -> list[dict]:
    assert directive["op"] == "order"
    assert all(len(slot) == 1 for slot in directive["filters"])
    return [slot[0] for slot in directive["filters"]]


def test_afpipe_pp2_builds_two_actor_local_rows_with_dense_endpoints() -> None:
    placement = _load("afd_placement_pp2.json")
    n_layers = infer_afpipe_num_layers(placement, fallback=4)

    orders = build_afpipe_baseline_schedule(
        placement,
        n_mbs=2,
        n_layers=n_layers,
        expected_actor_groups=2,
    )

    assert n_layers == 4
    assert len(orders) == 2
    dense_row, expert_row = map(_flat_row, orders)

    assert dense_row[:4] == [
        {"PP": 0, "ATTN": "-", "FFN": "-", "MB": 0, "PASS": "F"},
        {"ATTN": 0, "MB": 0, "PASS": "F"},
        {"PP": 0, "ATTN": "-", "FFN": "-", "MB": 1, "PASS": "F"},
        {"ATTN": 0, "MB": 1, "PASS": "F"},
    ]
    assert dense_row[10:14] == [
        {"ATTN": 4, "MB": 0, "PASS": "F"},
        {"PP": 1, "ATTN": "-", "FFN": "-", "MB": 0, "PASS": "F"},
        {"ATTN": 4, "MB": 1, "PASS": "F"},
        {"PP": 1, "ATTN": "-", "FFN": "-", "MB": 1, "PASS": "F"},
    ]
    # Backward reverses the co-located terminal and input chunks.
    assert dense_row[14:18] == [
        {"PP": 1, "ATTN": "-", "FFN": "-", "MB": 0, "PASS": "B"},
        {"ATTN": 4, "MB": 0, "PASS": "B"},
        {"PP": 1, "ATTN": "-", "FFN": "-", "MB": 1, "PASS": "B"},
        {"ATTN": 4, "MB": 1, "PASS": "B"},
    ]
    assert dense_row[-4:] == [
        {"ATTN": 0, "MB": 0, "PASS": "B"},
        {"PP": 0, "ATTN": "-", "FFN": "-", "MB": 0, "PASS": "B"},
        {"ATTN": 0, "MB": 1, "PASS": "B"},
        {"PP": 0, "ATTN": "-", "FFN": "-", "MB": 1, "PASS": "B"},
    ]

    assert expert_row == [
        {"FFN": layer, "MB": mb, "PASS": pass_}
        for pass_, layers in (("F", range(4)), ("B", reversed(range(4))))
        for layer in layers
        for mb in range(2)
    ]


def test_afpipe_pp4_infers_layers_and_builds_four_rows() -> None:
    placement = _load("afd_placement_pp4.json")
    assert infer_afpipe_num_layers(placement, fallback=99) == 4

    orders = build_afpipe_baseline_schedule(
        placement,
        n_mbs=2,
        n_layers=4,
        expected_actor_groups=4,
    )
    rows = list(map(_flat_row, orders))

    assert len(rows) == 4
    assert {flt.get("ATTN") for flt in rows[0]} == {"-", 0, 2, 4}
    assert {flt["FFN"] for flt in rows[1]} == {0, 2}
    assert {flt["ATTN"] for flt in rows[2]} == {1, 3}
    assert {flt["FFN"] for flt in rows[3]} == {1, 3}
    assert any(flt.get("PP") == 0 for flt in rows[0])
    assert any(flt.get("PP") == 3 for flt in rows[0])


def test_afpipe_rejects_incorrect_actor_group_count() -> None:
    with pytest.raises(ValueError, match="actor-group count"):
        build_afpipe_baseline_schedule(
            _load("afd_placement_pp2.json"),
            n_mbs=2,
            n_layers=4,
            expected_actor_groups=4,
        )

