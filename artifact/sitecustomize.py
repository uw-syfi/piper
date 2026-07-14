"""Runtime hooks for artifact-only TorchTitan experiments."""

import os


if os.environ.get("TORCHTITAN_USE_BMM_EXPERTS") == "1":
    try:
        import torch
        import torch.nn.functional as F
        from torch.distributed.tensor import DTensor

        from torchtitan.models.common.moe import moe as moe_mod

        def _run_experts_bmm(
            w1: torch.Tensor,
            w2: torch.Tensor,
            w3: torch.Tensor,
            x: torch.Tensor,
            num_tokens_per_expert: torch.Tensor,
        ) -> torch.Tensor:
            dim = x.shape[-1]
            num_experts = w1.shape[0]
            counts = num_tokens_per_expert.to(device=x.device, dtype=torch.long)
            capacity = int(counts.max().item()) if counts.numel() else 0
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

        def _bmm_experts_forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
            if isinstance(self.w1, DTensor):
                w1 = self.w1.to_local()
                w2 = self.w2.to_local()
                w3 = self.w3.to_local()
            else:
                w1 = self.w1
                w2 = self.w2
                w3 = self.w3
            return _run_experts_bmm(w1, w2, w3, x, num_tokens_per_expert)

        moe_mod.GroupedExperts.forward = _bmm_experts_forward
        print("[artifact] TorchTitan GroupedExperts.forward patched to BMM experts", flush=True)
    except Exception as exc:
        raise RuntimeError("Failed to install TorchTitan BMM experts artifact hook") from exc
