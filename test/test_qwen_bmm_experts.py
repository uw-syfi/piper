import torch
import torch.nn.functional as F

from examples.models.qwen3_bmm_experts_patch import _run_bmm_experts


def test_bmm_experts_routes_unsorted_packed_rows_by_expert_id() -> None:
    generator = torch.Generator().manual_seed(17)
    num_experts = 3
    dim = 4
    hidden_dim = 5
    rows = 7
    w1 = torch.randn(num_experts, hidden_dim, dim, generator=generator)
    w2 = torch.randn(num_experts, dim, hidden_dim, generator=generator)
    w3 = torch.randn(num_experts, hidden_dim, dim, generator=generator)
    x = torch.randn(rows, dim, generator=generator)
    expert_ids = torch.tensor([2, 0, 1, 2, 1, 0, 2])
    packed = torch.cat(
        [
            x,
            torch.remainder(expert_ids, 256)[:, None].to(x.dtype),
            torch.div(expert_ids, 256, rounding_mode="floor")[:, None].to(x.dtype),
        ],
        dim=-1,
    )

    actual = _run_bmm_experts(w1, w2, w3, packed)
    expected_rows = []
    for row, expert_id in zip(x, expert_ids):
        expert = int(expert_id)
        hidden = F.silu(w1[expert] @ row) * (w3[expert] @ row)
        expected_rows.append(w2[expert] @ hidden)
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected)
