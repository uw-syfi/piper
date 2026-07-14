"""Batched-matmul experts patch for Qwen3 MoE layers.

This patch is needed to avoid Triton kernels used by TorchTitan's Qwen3 model.
"""

import torch
import torch.nn.functional as F
from torch import nn


def _run_bmm_experts(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if x.is_meta:
        return torch.empty_like(x)

    dim = x.shape[-1]
    num_experts = w1.shape[0]
    counts = num_tokens_per_expert.to(device=x.device, dtype=torch.long)
    # Keep capacity shape-derived so torch.compile never has to materialize a
    # routed-token count as a Python scalar.
    capacity = x.shape[0]
    starts = torch.cumsum(counts, dim=0) - counts
    positions = torch.arange(capacity, device=x.device, dtype=torch.long)
    zero = x.new_zeros((capacity, dim))

    packed_rows = []
    for expert_idx in range(num_experts):
        valid = positions < counts[expert_idx]
        source_idx = starts[expert_idx] + positions
        safe_idx = torch.where(valid, source_idx, positions.new_zeros(()))
        packed_rows.append(torch.where(valid[:, None], x[safe_idx], zero))
    packed = torch.stack(packed_rows, dim=0)

    h = F.silu(torch.bmm(packed, w1.transpose(-2, -1)))
    h = h * torch.bmm(packed, w3.transpose(-2, -1))
    expert_out = torch.bmm(h, w2.transpose(-2, -1))

    out = x.new_zeros(x.shape)
    for expert_idx in range(num_experts):
        valid = positions < counts[expert_idx]
        dest_idx = starts[expert_idx] + positions
        safe_idx = torch.where(valid, dest_idx, positions.new_zeros(()))
        src = torch.where(valid[:, None], expert_out[expert_idx], zero)
        out = out.scatter_add(0, safe_idx[:, None].expand(-1, dim), src)
    return out


_BMM_EXPERTS_LIBS = []


def _register_bmm_experts_op() -> None:
    if (
        hasattr(torch.ops, "piper_artifact")
        and hasattr(torch.ops.piper_artifact, "bmm_experts")
    ):
        return

    def_lib = torch.library.Library("piper_artifact", "DEF")
    def_lib.define(
        "bmm_experts(Tensor w1, Tensor w2, Tensor w3, Tensor x, Tensor num_tokens_per_expert) -> Tensor"
    )
    impl_lib = torch.library.Library("piper_artifact", "IMPL")
    impl_lib.impl("bmm_experts", _run_bmm_experts, "CPU")
    impl_lib.impl("bmm_experts", _run_bmm_experts, "CUDA")
    impl_lib.impl("bmm_experts", _run_bmm_experts, "Autograd")
    impl_lib.impl("bmm_experts", _run_bmm_experts, "Meta")
    _BMM_EXPERTS_LIBS.extend([def_lib, impl_lib])


_register_bmm_experts_op()


def bmm_experts(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.piper_artifact.bmm_experts(
        w1, w2, w3, x, num_tokens_per_expert
    )


class BmmExperts(nn.Module):
    """Triton-free expert GEMM with GroupedExperts-compatible routing.

    Mirrors torchtitan ``GroupedExperts`` parameter shapes/names so the param
    registry is unchanged, but runs the per-expert SwiGLU via batched matmul
    instead of ``torch._grouped_mm``. The input must be sorted by expert, and
    ``num_tokens_per_expert`` gives each expert's dynamic token count.
    """

    def __init__(self, dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

    def forward(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        counts = num_tokens_per_expert.to(device=x.device, dtype=torch.long)
        return bmm_experts(self.w1, self.w2, self.w3, x, counts)
