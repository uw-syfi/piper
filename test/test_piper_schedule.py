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


def test_load_schedule_info_derives_parallel_degrees(tmp_path) -> None:
    schedule_path = tmp_path / "qwen_example.json"
    schedule_path.write_text(
        json.dumps([
            {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
            {"op": "place", "filter": {"PP": 1}, "devices": [2, 3]},
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 4},
        ]),
        encoding="utf-8",
    )

    info = load_schedule_info(str(schedule_path))

    assert info == {
        "name": "qwen_example",
        "path": str(schedule_path),
        "num_stages": 2,
        "pp_degree": 2,
        "dp_degree": 2,
        "num_microbatches": 4,
    }


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
