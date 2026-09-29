"""Every DP replica starts from the same weights."""
import pytest
import torch
import torch.nn as nn

from src.piper import annotate
from test.cpu_e2e import ray_cpu, run_program, train  # noqa: F401


class _TwoLayer(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        with annotate("PP"):
            return self.b(torch.relu(self.a(x)))


def _schedule(replicate_options: dict) -> list[dict]:
    return [
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
        {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1], **replicate_options},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    ]


@pytest.mark.parametrize("replicate_options", [{}, {"shard_grads": True}], ids=["replicate", "zero2"])
def test_dp_replicas_start_from_the_same_weights(ray_cpu, tmp_path, replicate_options) -> None:
    batches = [((torch.randn(2, 8),), torch.randn(2, 8)) for _ in range(2)]
    results = run_program(
        tmp_path, _schedule(replicate_options), train,
        _TwoLayer, str(tmp_path / "schedule.json"), batches, 0,
    )
    (rank0,), (rank1,) = (r["initial"].values() for r in results)
    assert rank0.keys() == rank1.keys()
    for name in rank0:
        assert torch.equal(rank0[name], rank1[name]), name
