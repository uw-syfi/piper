"""Ring attention must compute dense attention.

    torchrun --nproc_per_node=2 test/test_cp_equivalence.py

The context-parallel counterpart of test_tp_equivalence.py: it validates the
*math and the placement of the ring exchange* with no Ray, no DAG and no
scheduling in the way. Each rank holds one sequence chunk of Q, K and V; K/V
travel the ring; every rank ends with attention for its own Q rows over the full
sequence. The reference gathers the chunks and runs dense attention.

`_Rotate` is the executable statement of what a RING_COMM pair does: forward
hands the chunk to the next rank, backward hands the chunk's gradient back to
the previous one. Two controls guard against the test proving nothing:

  * no rotation at all       -> the output must differ (each rank sees one chunk);
  * forward-only rotation    -> the output matches but dK/dV must differ. This is
                                the control that constrains the *backward* ring,
                                which a loss curve would never expose.
"""
import math
import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))
from models.ring_attn import ring_step  # noqa: E402

B, H, S_LOCAL, D = 2, 2, 16, 8
SEED = 0
DTYPE = torch.float32
RTOL, ATOL = 1e-5, 1e-5


def _exchange(tensors, shift):
    n, me = dist.get_world_size(), dist.get_rank()
    dst, src = (me + shift) % n, (me - shift) % n
    sends = [t.detach().contiguous() for t in tensors]
    recvs = [torch.empty_like(s) for s in sends]
    ops = [dist.P2POp(dist.isend, s, dst) for s in sends]
    ops += [dist.P2POp(dist.irecv, r, src) for r in recvs]
    for w in dist.batch_isend_irecv(ops):
        w.wait()
    return recvs


class _Rotate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, v):
        return tuple(_exchange([k, v], +1))

    @staticmethod
    def backward(ctx, gk, gv):
        return tuple(_exchange([gk, gv], -1))


class _RotateFwdOnly(torch.autograd.Function):
    """Control: the chunk moves, its gradient does not come back."""
    @staticmethod
    def forward(ctx, k, v):
        return tuple(_exchange([k, v], +1))

    @staticmethod
    def backward(ctx, gk, gv):
        return gk, gv


def _ring_attention(q, k, v, rotate):
    n = dist.get_world_size()
    scale = 1.0 / math.sqrt(D)
    o = torch.zeros_like(q)
    m = torch.full(q.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)
    l = torch.zeros(q.shape[:-1], dtype=q.dtype, device=q.device)
    for step in range(n):
        o, m, l = ring_step(q, k, v, o, m, l, scale)
        if rotate is not None and step < n - 1:
            k, v = rotate.apply(k, v)
    return o / l.unsqueeze(-1)


def _gather(t):
    n = dist.get_world_size()
    parts = [torch.empty_like(t) for _ in range(n)]
    dist.all_gather(parts, t.detach().contiguous())
    return torch.cat(parts, dim=-2)


def _dense_reference(q, k, v):
    """Every rank runs the whole computation; grads for its chunk are the slice."""
    rank = dist.get_rank()
    Q, K, V = (_gather(t).requires_grad_(True) for t in (q, k, v))
    out = F.scaled_dot_product_attention(Q, K, V)
    out.pow(2).sum().backward()
    sl = slice(rank * S_LOCAL, (rank + 1) * S_LOCAL)
    return out[..., sl, :].detach(), Q.grad[..., sl, :], K.grad[..., sl, :], V.grad[..., sl, :]


def _run_arm(q0, k0, v0, rotate):
    q, k, v = (t.clone().requires_grad_(True) for t in (q0, k0, v0))
    out = _ring_attention(q, k, v, rotate)
    out.pow(2).sum().backward()
    return out.detach(), q.grad, k.grad, v.grad


def _run() -> None:
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    n = dist.get_world_size()

    # Each rank's chunk is a different slice of one global seed, so the gathered
    # tensors are the same on every rank and equal to what a 1-rank run would see.
    g = torch.Generator().manual_seed(SEED)
    full = [torch.randn(B, H, S_LOCAL * n, D, generator=g, dtype=DTYPE) for _ in range(3)]
    sl = slice(rank * S_LOCAL, (rank + 1) * S_LOCAL)
    q0, k0, v0 = (t[..., sl, :].contiguous().to(device) for t in full)

    ref = _dense_reference(q0, k0, v0)
    got = _run_arm(q0, k0, v0, _Rotate)
    for name, r, g_ in zip(("output", "dQ", "dK", "dV"), ref, got):
        err = (r - g_).abs().max().item()
        assert torch.allclose(r, g_, rtol=RTOL, atol=ATOL), (
            f"rank {rank}: CP={n} {name} differs from dense attention by {err:.3e}"
        )

    # Controls. Both must actually break, or the assertions above are vacuous.
    no_rot = _run_arm(q0, k0, v0, None)
    assert not torch.allclose(ref[0], no_rot[0], rtol=RTOL, atol=ATOL), (
        "dropping the rotation did not change the output; the test has no power"
    )
    fwd_only = _run_arm(q0, k0, v0, _RotateFwdOnly)
    assert torch.allclose(ref[0], fwd_only[0], rtol=RTOL, atol=ATOL)
    assert not torch.allclose(ref[2], fwd_only[2], rtol=RTOL, atol=ATOL), (
        "dropping the backward rotation did not change dK; the backward ring is "
        "unconstrained by this test"
    )
    if rank == 0:
        errs = {n_: (r - g_).abs().max().item() for n_, r, g_ in zip(("out", "dQ", "dK", "dV"), ref, got)}
        print(f"CP={n} ring == dense: {errs}  | controls broke as required")
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.gpu
def test_ring_attention_matches_dense_attention_on_two_gpus() -> None:
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", __file__],
        check=True,
    )


if __name__ == "__main__":
    _run()
