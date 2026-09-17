"""Sharding the MLP must not change what it computes.

    torchrun --nproc_per_node=2 test/test_tp_equivalence.py

Run this before trusting a Piper TP run. It validates the *math and the
placement of the collectives* with no Ray, no DAG and no scheduling in the way,
so a wrong answer here is a wrong answer about tensor parallelism rather than
about the runtime.

Comparing loss curves cannot answer the question. Piper initializes parameters
with `manual_seed(1000 * global_rank + stage_id)`, so two
configurations never start from the same weights. The weights here are therefore
built once from a single global seed and *sliced* per rank, so global column j of
`up` holds the same values at every TP degree. Any difference in the output is
then the sharding, which is the thing under test.

`_F` and `_G` below are Megatron's conjugate pair, and they are the executable
statement of what Piper's two TP_COMM nodes are supposed to do:

  * `_G` at the region exit  -- forward all-reduce, backward identity;
  * `_F` at the region entry -- forward identity, backward all-reduce.
"""
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

# examples/ is not a package on sys.path outside the harness, but the weight
# construction has to be the *same* one the in-Piper run uses -- otherwise
# "both start from the same weights" is a coincidence between two hand-copied
# constructions rather than a shared fact.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from models.tp_mlp import global_weights, shard_weights  # noqa: E402

DIM = 64
HIDDEN = 256
BATCH = 8
SEED = 1234
# fp32: the point is to catch a misplaced or missing collective, and bf16's ~3
# decimal digits would hide a real error inside its own noise. Piper itself runs
# bf16; that is a separate measurement, not this test's job.
DTYPE = torch.float32
RTOL, ATOL = 1e-5, 1e-5


class _G(torch.autograd.Function):
    """Region exit: the output is a partial sum. Backward is identity."""

    @staticmethod
    def forward(ctx, x):
        out = x.clone()
        dist.all_reduce(out)
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad


class _F(torch.autograd.Function):
    """Region entry: forward is identity, the input gradient is a partial sum."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad):
        out = grad.clone()
        dist.all_reduce(out)
        return out


def _global_weights(device):
    """The same weights examples/test_tp_mlp.py uses, on this device."""
    return {
        k.removesuffix(".weight"): v.to(device)
        for k, v in global_weights(DIM, HIDDEN, SEED, DTYPE).items()
    }


def _reference(weights, x):
    """The unsharded computation, identical on every rank."""
    z = F.linear(x, weights["pre"])
    h = F.linear(F.gelu(F.linear(z, weights["up"])), weights["down"])
    return F.linear(h, weights["post"])


def _sharded(weights, x, rank, tp_degree):
    # Same slicing rule as the in-Piper run: shard_weights keys on ".weight".
    local_w = shard_weights(
        {f"{k}.weight": v for k, v in weights.items()}, rank, tp_degree
    )
    up_local = local_w["up.weight"]
    down_local = local_w["down.weight"]

    z = F.linear(x, weights["pre"])
    z = _F.apply(z)
    h = F.linear(F.gelu(F.linear(z, up_local)), down_local)
    h = _G.apply(h)
    return F.linear(h, weights["post"])


def _run() -> None:
    rank = int(os.environ["RANK"])
    tp_degree = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    weights = _global_weights(device)
    torch.manual_seed(SEED)
    x0 = torch.randn(BATCH, DIM, dtype=DTYPE, device=device)

    outs = {}
    for name, fn in (
        ("reference", lambda w, x: _reference(w, x)),
        ("sharded", lambda w, x: _sharded(w, x, rank, tp_degree)),
    ):
        x = x0.clone().requires_grad_(True)
        y = fn(weights, x)
        y.pow(2).sum().backward()
        outs[name] = (y.detach().clone(), x.grad.detach().clone())

    y_ref, gx_ref = outs["reference"]
    y_tp, gx_tp = outs["sharded"]
    y_err = (y_ref - y_tp).abs().max().item()
    gx_err = (gx_ref - gx_tp).abs().max().item()

    assert torch.allclose(y_ref, y_tp, rtol=RTOL, atol=ATOL), (
        f"rank {rank}: TP={tp_degree} output differs from unsharded by {y_err:.3e}. "
        f"A forward all-reduce is missing or misplaced."
    )
    assert torch.allclose(gx_ref, gx_tp, rtol=RTOL, atol=ATOL), (
        f"rank {rank}: TP={tp_degree} input gradient differs from unsharded by "
        f"{gx_err:.3e}. This is the check that constrains the backward all-reduce: "
        f"without it each rank keeps only its own partial sum."
    )

    # A guard against the test proving nothing: dropping either collective must
    # actually break the comparison at this scale.
    x = x0.clone().requires_grad_(True)
    local_w = shard_weights(
        {f"{k}.weight": v for k, v in weights.items()}, rank, tp_degree
    )
    z = F.linear(x, weights["pre"])
    h = F.linear(
        F.gelu(F.linear(z, local_w["up.weight"])), local_w["down.weight"]
    )
    y_nocomm = F.linear(h, weights["post"])
    y_nocomm.pow(2).sum().backward()
    assert not torch.allclose(y_ref, y_nocomm, rtol=RTOL, atol=ATOL), (
        "removing both collectives did not change the result, so this test would "
        "pass even with no communication at all"
    )

    if rank == 0:
        print(
            f"\nPASS  TP={tp_degree} matches the unsharded MLP "
            f"(output {y_err:.2e}, input grad {gx_err:.2e}); "
            f"dropping the collectives does change it"
        )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.gpu
def test_tp_matches_unsharded_mlp() -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        pytest.skip("run under torchrun --nproc_per_node=2")
    _run()


if __name__ == "__main__":
    _run()
