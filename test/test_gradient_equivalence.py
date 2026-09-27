"""A Piper step matches one single-process step on the whole global batch."""
import pytest
import torch
import torch.nn as nn

from examples.build_schedule import build_1f1b_schedule
from src.piper import annotate
from test.cpu_e2e import load_initial, merged, ray_cpu, reference_sgd_step, run_program, train  # noqa: F401

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


def _gpipe(n_ranks: int, n_mbs: int, split_backward: bool = False) -> list[dict]:
    """Order every forward before every backward, on each rank.

    On CPU, gloo's send blocks until the peer receives, so 1F1B and ZeroBubble,
    whose sends cross between ranks, deadlock there. This order never crosses.
    """
    backward = ["BI", "BW"] if split_backward else ["B"]
    return [
        {
            "op": "order",
            "filters": [[{"PP": rank, "MB": mb, "PASS": "F"}] for mb in range(n_mbs)]
            + [[{"PP": rank, "MB": mb, "PASS": p}] for mb in range(n_mbs) for p in backward],
        }
        for rank in range(n_ranks)
    ]


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


def _check_one_step(tmp_path, schedule: list[dict], dp: int, local_batch: int, model_class=_TwoStage, dim=DIM) -> None:
    torch.manual_seed(0)
    batches = [
        ((torch.randn(local_batch, dim),), torch.randn(local_batch, dim)) for _ in range(dp)
    ]
    results = run_program(
        tmp_path, schedule, train, model_class, str(tmp_path / "schedule.json"), batches, 1,
    )
    expected = reference_sgd_step(model_class(), merged(results[0]["initial"]), batches)
    for rank, result in enumerate(results):
        got = merged(result["final"])
        assert got.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(got[name], expected[name], msg=f"dp rank {rank}: {name}")
    return results


def test_dp2_replicate(ray_cpu, tmp_path) -> None:
    _check_one_step(tmp_path, _schedule(pp=1, dp=2), dp=2, local_batch=4)


def test_dp2_two_microbatches(ray_cpu, tmp_path) -> None:
    _check_one_step(tmp_path, _schedule(pp=1, dp=2, num_microbatches=2), dp=2, local_batch=4)


@pytest.mark.parametrize("split_backward", [False, True], ids=["fused_backward", "split_backward"])
def test_pp2_two_microbatches(ray_cpu, tmp_path, split_backward) -> None:
    order = lambda n_ranks, n_mbs: _gpipe(n_ranks, n_mbs, split_backward)
    (result,) = _check_one_step(
        tmp_path, _schedule(pp=2, dp=1, num_microbatches=2, order=order), dp=1, local_batch=4
    )
    assert len(result["losses"][0]) == 2



class _FourLayers(nn.Module):
    """One stage of four 1 MiB fp32 layers, so bucket_size=1 makes four buckets."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(512, 512, bias=False) for _ in range(4))

    def forward(self, x):
        with annotate("PP"):
            for layer in self.layers:
                x = torch.tanh(layer(x))
            return x


def test_dp2_buckets_two_microbatches(ray_cpu, tmp_path) -> None:
    schedule = [
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
        {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1], "bucket_size": 1},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
    ]
    _check_one_step(tmp_path, schedule, dp=2, local_batch=4, model_class=_FourLayers, dim=512)


def test_each_microbatch_gets_its_own_slice(ray_cpu, tmp_path) -> None:
    torch.manual_seed(0)
    x, y = torch.randn(4, DIM), torch.randn(4, DIM)
    (result,) = run_program(
        tmp_path, _schedule(pp=1, dp=1, num_microbatches=2), train,
        _TwoStage, str(tmp_path / "schedule.json"), [((x,), y)], 1,
    )
    model = load_initial(_TwoStage(), merged(result["initial"]))
    with torch.no_grad():
        expected = [nn.functional.mse_loss(model(xs), ys).item() for xs, ys in zip(x.chunk(2), y.chunk(2))]
    assert result["losses"][0] == pytest.approx(expected)
