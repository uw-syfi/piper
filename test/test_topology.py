import json

import pytest

from src.topology import Topology


def test_rank_layout_is_pp_inner() -> None:
    topo = Topology(dp=2, pp=3)
    assert topo.world_size == 6
    assert [topo.rank(dp, pp) for dp, pp in topo.slots()] == list(range(6))
    assert topo.slot(4) == (1, 1)
    assert topo.slot(6) is None
    assert topo.pp_groups() == [[0, 1, 2], [3, 4, 5]]
    assert topo.dp_groups() == [[0, 3], [1, 4], [2, 5]]


def test_standby_ranks_follow_trainers() -> None:
    topo = Topology(dp=2, pp=1, num_standby=1)
    assert topo.num_trainers == 2
    assert topo.world_size == 3
    assert topo.standby_ranks == [2]
    assert topo.groups() == {"dp": [[0, 1]], "pp": [[0], [1]]}


def test_from_schedule_reads_degrees(tmp_path) -> None:
    schedule = [
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 2]},
        {"op": "place", "filter": {"PP": 1}, "devices": [1, 3]},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 4},
    ]
    path = tmp_path / "pp2_dp2.json"
    path.write_text(json.dumps(schedule))
    topo = Topology.from_schedule(str(path))
    assert (topo.dp, topo.pp) == (2, 2)


def test_rejects_degenerate_degrees() -> None:
    with pytest.raises(ValueError):
        Topology(dp=0)
