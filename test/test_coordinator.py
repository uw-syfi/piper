import json
import os
import time

import pytest
import ray

from src.coordinator import PiperProgramCoordinator, create_piper_placement_group


HUNG_RANK_SLEEP = 120


def _crash_one_rank(*args, **kwargs):
    """Driver stand-in: dp_rank 1 fails after 1 s; every other rank hangs,
    like a peer stuck in the collective the failed rank abandoned."""
    if int(os.environ["PIPER_DP_RANK"]) == 1:
        time.sleep(1)
        raise RuntimeError("injected driver failure")
    time.sleep(HUNG_RANK_SLEEP)
    return "unreachable"


@pytest.fixture
def ray_cluster():
    # Fake GPU resources: run_dp_rank and the placement group ask for GPUs,
    # but the driver stand-in never touches one.
    ray.init(num_cpus=4, num_gpus=2, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def test_job_fails_when_one_dp_rank_task_fails(ray_cluster, tmp_path) -> None:
    schedule = tmp_path / "dp2.json"
    schedule.write_text(
        json.dumps([
            {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
        ]),
        encoding="utf-8",
    )
    pg = create_piper_placement_group(str(schedule))
    ray.get(pg.ready(), timeout=60)
    coordinator = PiperProgramCoordinator.remote(schedule_directives_file=str(schedule))

    t0 = time.perf_counter()
    # Ray re-raises a task's error as an instance of the original class,
    # so the failing rank's RuntimeError surfaces through run_program as one.
    with pytest.raises(RuntimeError, match="injected driver failure"):
        ray.get(
            coordinator.run_program.remote(_crash_one_rank, pg),
            timeout=HUNG_RANK_SLEEP,
        )
    # Failed without waiting for the hung rank to finish.
    assert time.perf_counter() - t0 < HUNG_RANK_SLEEP
