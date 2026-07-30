"""Probe TorchTitan's actual ragged ExpertParallel wrapper under fullgraph.

Run with:

    torchrun --standalone --nproc-per-node=2 \
        scripts/test_torchtitan_expert_parallel_fullgraph.py

Unlike ``test_torchtitan_grouped_experts_fullgraph.py``, this includes the
count exchange, uneven all-to-all dispatch, expert permutation, grouped GEMM,
inverse permutation, and uneven all-to-all combine.
"""

import json
import os
import traceback

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.models.moe.moe import GroupedExperts


def dynamo_counters() -> dict[str, dict[str, int]]:
    counters = torch._dynamo.utils.counters
    return {
        category: {str(key): int(value) for key, value in values.items()}
        for category, values in counters.items()
        if category in {"graph_break", "frames", "stats", "inductor"}
    }


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This probe expects two ranks, got {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234)

    mesh = init_device_mesh(
        "cuda",
        (world_size,),
        mesh_dim_names=("ep",),
    )
    experts = GroupedExperts(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        use_grouped_mm=True,
    ).to(device=device, dtype=torch.bfloat16)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    experts = parallelize_module(experts, mesh, ExpertParallel())

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(experts, backend="inductor", fullgraph=True)

    count_patterns = (
        (
            (1, 5, 2, 8),
            (7, 1, 6, 2),
        ),
        (
            (0, 8, 8, 0),
            (4, 4, 0, 8),
        ),
    )
    results = []
    try:
        for iteration, per_rank_counts in enumerate(count_patterns):
            counts = torch.tensor(
                per_rank_counts[rank],
                device=device,
                dtype=torch.int64,
            )
            x = torch.randn(
                (int(counts.sum().item()), 32),
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            output = compiled(x, counts)
            output.float().square().mean().backward()
            torch.cuda.synchronize()
            results.append(
                {
                    "iteration": iteration,
                    "counts": counts.cpu().tolist(),
                    "input_shape": list(x.shape),
                    "output_shape": list(output.shape),
                    "finite": bool(torch.isfinite(output).all().item()),
                    "input_grad_finite": bool(torch.isfinite(x.grad).all().item()),
                }
            )

        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(
            {
                "rank": rank,
                "ok": True,
                "results": results,
                "counters": dynamo_counters(),
            },
            gathered,
            dst=0,
        )
        if rank == 0:
            print(json.dumps(gathered, indent=2, sort_keys=True), flush=True)
    except Exception as error:
        failure = {
            "rank": rank,
            "ok": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "counters": dynamo_counters(),
        }
        print(json.dumps(failure, indent=2, sort_keys=True), flush=True)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
