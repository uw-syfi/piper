"""Run TorchTitan's unmodified Qwen3 MoE path under fullgraph compilation.

This is intentionally separate from Piper's compiler. It establishes whether
TorchTitan's default GroupedExperts and complete MoE implementations introduce
TorchDynamo graph breaks before Piper adds expert-parallel communication.
"""

import json
import traceback

import torch

from torchtitan.models.moe.moe import GroupedExperts, MoE, MoEArgs


DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


def dynamo_counters() -> dict[str, dict[str, int]]:
    counters = torch._dynamo.utils.counters
    return {
        category: {str(key): int(value) for key, value in values.items()}
        for category, values in counters.items()
        if category in {"graph_break", "frames", "stats", "inductor"}
    }


def run_grouped_experts() -> dict:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.manual_seed(1234)

    experts = GroupedExperts(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        use_grouped_mm=True,
    ).to(device=DEVICE, dtype=DTYPE)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)

    compiled = torch.compile(experts, backend="inductor", fullgraph=True)
    patterns = (
        torch.tensor([1, 5, 2, 8], device=DEVICE, dtype=torch.int64),
        torch.tensor([7, 0, 6, 3], device=DEVICE, dtype=torch.int64),
        torch.tensor([0, 0, 16, 0], device=DEVICE, dtype=torch.int64),
    )
    results = []
    for counts in patterns:
        x = torch.randn(
            (int(counts.sum().item()), 32),
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        output = compiled(x, counts)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        results.append(
            {
                "counts": counts.cpu().tolist(),
                "output_shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "input_grad_finite": bool(torch.isfinite(x.grad).all().item()),
            }
        )

    return {
        "name": "GroupedExperts",
        "results": results,
        "counters": dynamo_counters(),
    }


def run_moe() -> dict:
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    torch.manual_seed(5678)

    args = MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        top_k=2,
        use_grouped_mm=True,
        num_expert_groups=None,
        num_limited_groups=None,
        score_func="softmax",
        route_norm=False,
        route_scale=1.0,
        gate_bias=False,
        score_before_experts=False,
        load_balance_coeff=None,
        _debug_force_load_balance=False,
    )
    moe = MoE(args, dim=32, hidden_dim=64).to(device=DEVICE, dtype=DTYPE)
    for parameter in moe.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)

    compiled = torch.compile(moe, backend="inductor", fullgraph=True)
    results = []
    for seed in (10, 20, 30):
        torch.manual_seed(seed)
        x = torch.randn(
            (2, 8, 32),
            device=DEVICE,
            dtype=DTYPE,
            requires_grad=True,
        )
        output = compiled(x)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
        results.append(
            {
                "seed": seed,
                "output_shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
                "input_grad_finite": bool(torch.isfinite(x.grad).all().item()),
            }
        )

    return {
        "name": "MoE",
        "results": results,
        "counters": dynamo_counters(),
    }


def main() -> None:
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    outcomes = []
    for probe in (run_grouped_experts, run_moe):
        try:
            outcomes.append({"ok": True, **probe()})
        except Exception as error:
            outcomes.append(
                {
                    "ok": False,
                    "name": probe.__name__,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "counters": dynamo_counters(),
                }
            )
    print(json.dumps(outcomes, indent=2, sort_keys=True), flush=True)
    if not all(outcome["ok"] for outcome in outcomes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
