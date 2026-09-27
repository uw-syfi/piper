"""A Piper step matches one single-process step on the whole global batch."""
import pytest
import torch
import torch.nn as nn

from examples.build_schedule import build_1f1b_schedule
from src.piper import annotate
from test.cpu_e2e import merged, ray_cpu, reference_sgd_step, run_program, train  # noqa: F401

DIM = 8


class _TwoStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(DIM, DIM, bias=False)
        self.b = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        with annotate("PP"):
            h = torch.tanh(self.a(x))
        with annotate("PP"):
            return self.b(h)


def _schedule(pp: int, dp: int, num_microbatches: int = 1, order=build_1f1b_schedule, **replicate) -> list[dict]:
    """Place the two stages on *pp* ranks, each replicated on *dp* ranks."""
    directives: list[dict] = []
    for stage in range(2):
        pp_rank = stage * pp // 2
        devices = [pp_rank + dp_rank * pp for dp_rank in range(dp)]
        directives.append({"op": "place", "filter": {"PP": stage}, "devices": devices})
        if dp > 1:
            directives.append({"op": "replicate", "filter": {"PP": stage}, "devices": devices, **replicate})
    directives.append({"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": num_microbatches})
    if pp > 1:
        directives += order(pp, num_microbatches)
    return directives


def _check_one_step(tmp_path, schedule: list[dict], dp: int, local_batch: int) -> None:
    torch.manual_seed(0)
    batches = [
        ((torch.randn(local_batch, DIM),), torch.randn(local_batch, DIM)) for _ in range(dp)
    ]
    results = run_program(
        tmp_path, schedule, train, _TwoStage, str(tmp_path / "schedule.json"), batches, 1,
    )
    expected = reference_sgd_step(_TwoStage(), merged(results[0]["initial"]), batches)
    for rank, result in enumerate(results):
        got = merged(result["final"])
        assert got.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(got[name], expected[name], msg=f"dp rank {rank}: {name}")


def test_dp2_replicate(ray_cpu, tmp_path) -> None:
    _check_one_step(tmp_path, _schedule(pp=1, dp=2), dp=2, local_batch=4)
