import sys
from pathlib import Path


_EXAMPLES_DIR = str(Path(__file__).resolve().parents[1] / "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from test_harness import _metrics_rows  # noqa: E402


def test_actor_group_metrics_count_per_rank_work_and_keep_rank_peak_labels() -> None:
    rows = _metrics_rows([
        {
            "program_rank": 0,
            "actor_groups": [
                {"id": 0, "devices": [0, 2, 4, 5]},
                {"id": 1, "devices": [1, 3]},
            ],
            "world_size": 6,
            "input_actor_group_size": 4,
            "batch_size": 2,
            "num_microbatches": 2,
            "seq_len": 8,
            "iter_times_s": [2.0],
            "peak_memory_by_rank": {0: 1024**3, 3: 2 * 1024**3},
        }
    ])

    assert len(rows) == 1
    row = rows[0]
    assert row["program_rank"] == 0
    assert row["actor_group_count"] == 2
    assert row["input_actor_group_size"] == 4
    assert row["global_tokens_per_step"] == 128
    assert row["throughput_tokens_per_s"] == 64.0
    assert row["peak_memory_rank0_gb"] == 1.0
    assert row["peak_memory_rank3_gb"] == 2.0
    assert not any(key.startswith("peak_memory_pp") for key in row)
