"""Ring-attention-shaped model for context parallelism.

Written in Piper's style: the ring exchange is *not* in the model code, exactly
as EP's all-to-all is not. Each ring step is one annotated region, and K/V are
passed through the regions unchanged; a directive is meant to replace each
pass-through hop with a neighbour exchange. The single-process trace is
therefore not dense attention -- n steps over the same chunk -- in the same way
that Piper's EP trace is not a real MoE without its all-to-all. The math is
gated separately in test/test_ring_attn_math.py.

Form B (`RingAttnRebound`) rebinds K/V with an explicit op inside each region.
It exists to show what a model must look like if pass-through values do not
survive segmentation; it is the fallback, not the intent.
"""
import math

import torch
import torch.nn as nn

from src.piper import annotate


def ring_step(q, k, v, o, m, l, scale):
    """One online-softmax accumulation against a K/V chunk.

    Returns updated (o, m, l) in unnormalized form; divide o by l at the end.
    """
    s = torch.matmul(q, k.transpose(-1, -2)) * scale
    m_new = torch.maximum(m, s.amax(dim=-1))
    alpha = torch.exp(m - m_new)
    p = torch.exp(s - m_new.unsqueeze(-1))
    l = l * alpha + p.sum(dim=-1)
    o = o * alpha.unsqueeze(-1) + torch.matmul(p, v)
    return o, m_new, l


class RingAttn(nn.Module):
    """n ring steps as annotated CP regions; K/V forwarded unchanged."""

    def __init__(self, dim: int, heads: int, n_steps: int):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.n_steps = dim, heads, n_steps
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def _split_heads(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x, k, v):
        # Piper requires every compute node annotated; PP is the outer stage
        # region (as in tp_mlp), CP the per-step region nested inside it.
        with annotate("PP"):
            q = self._split_heads(self.q_proj(x))
            k = self._split_heads(k)
            v = self._split_heads(v)
            o = torch.zeros_like(q)
            m = torch.full(q.shape[:-1], float("-inf"), dtype=q.dtype, device=q.device)
            l = torch.zeros(q.shape[:-1], dtype=q.dtype, device=q.device)
            for _ in range(self.n_steps):
                with annotate("CP"):
                    o, m, l = self._step(q, k, v, o, m, l)
                    k, v = self._rebind(k, v)
            o = o / l.unsqueeze(-1)
            b, h, s, d = o.shape
            return self.o_proj(o.transpose(1, 2).reshape(b, s, h * d))

    def _step(self, q, k, v, o, m, l):
        return ring_step(q, k, v, o, m, l, self.scale)

    def _rebind(self, k, v):
        return k, v


class RingAttnRebound(RingAttn):
    """Same, but K/V are rebound by an op inside each region (fallback form)."""

    def _rebind(self, k, v):
        return k.clone(), v.clone()


def global_weights(dim: int, seed: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Identical projection weights on every rank and at every CP degree.

    CP shards the sequence, not the parameters, so unlike tp_mlp there is no
    per-rank slice: the same dict is pushed to every rank. Built from one global
    seed so a CP=2 run and a dense CP=1 run start from the same model.
    """
    g = torch.Generator().manual_seed(seed)
    return {
        "q_proj.weight": (torch.randn(dim, dim, generator=g) * 0.02).to(dtype),
        "o_proj.weight": (torch.randn(dim, dim, generator=g) * 0.02).to(dtype),
    }
