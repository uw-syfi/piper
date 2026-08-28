"""Run a two-rank correctness probe for Piper's ragged EP boundary."""

import json
import os
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist

from src.executors import DagExecutor
from src.tasks import TaskType
from torchtitan.models.moe.moe import GroupedExperts, _run_experts_grouped_mm


def dynamo_counters() -> dict[str, dict[str, int]]:
    counters = torch._dynamo.utils.counters
    return {
        category: {str(key): int(value) for key, value in values.items()}
        for category, values in counters.items()
        if category in {"graph_break", "frames", "stats", "inductor"}
    }


def make_a2a_node(uid, task_type, direction, expert_uid, participants):
    return SimpleNamespace(
        uid=uid,
        task_type=task_type,
        node_meta={
            "direction": direction,
            "expert_fwd_uid": expert_uid,
            "participants": participants,
        },
    )


def make_executor(ep_group, global_rank):
    return DagExecutor(
        runtime=SimpleNamespace(
            ep_group=ep_group,
            global_rank=global_rank,
        ),
        stages=None,
        buffers=None,
        events=None,
        params=None,
        communication=None,
        compute=None,
        logger=None,
    )


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This probe expects two ranks, got {world_size}")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    executor = make_executor(dist.group.WORLD, rank)
    participants = list(range(world_size))
    dispatch_node = make_a2a_node(
        "dispatch",
        TaskType.FWD_A2A,
        "incoming",
        "expert",
        participants,
    )
    combine_node = make_a2a_node(
        "combine",
        TaskType.FWD_A2A,
        "outgoing",
        "expert",
        participants,
    )
    gradient_node = make_a2a_node(
        "gradient_dispatch",
        TaskType.BWD_A2A,
        "incoming",
        "expert",
        participants,
    )
    torch.manual_seed(1234)
    experts = GroupedExperts(
        dim=32,
        hidden_dim=64,
        num_experts=4,
        use_grouped_mm=True,
    ).to(device=device, dtype=torch.bfloat16)
    for parameter in experts.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
    def externally_dispatched_experts(x, counts):
        return _run_experts_grouped_mm(
            experts.w1,
            experts.w2,
            experts.w3,
            x,
            counts,
        )

    compiled_experts = torch.compile(
        externally_dispatched_experts,
        backend="inductor",
        fullgraph=True,
        dynamic=True,
    )

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
            row_ids = torch.arange(
                counts.sum(),
                device=device,
                dtype=torch.float32,
            )
            x = (
                row_ids[:, None]
                + rank * 1000
                + torch.arange(32, device=device, dtype=torch.float32)[None, :] / 32
            ).to(torch.bfloat16).requires_grad_(True)
            reference_output = experts(x, counts)

            dispatched, local_counts = executor._all_to_all_ep_boundary(
                dispatch_node,
                x,
                torch.cuda.current_stream(),
                routing_counts=counts,
            )
            expert_output = compiled_experts(dispatched, local_counts)
            combined_expert_output = executor._all_to_all_ep_boundary(
                combine_node,
                expert_output,
                torch.cuda.current_stream(),
            )
            assert combined_expert_output.shape == x.shape
            assert bool(torch.isfinite(combined_expert_output).all().item())
            torch.testing.assert_close(
                combined_expert_output,
                reference_output,
                atol=2e-3,
                rtol=2e-2,
            )

            combined = executor._all_to_all_ep_boundary(
                combine_node,
                dispatched * 2,
                torch.cuda.current_stream(),
            )
            torch.testing.assert_close(combined, x * 2)

            source_grad = x + 100
            dispatched_grad = executor._all_to_all_ep_boundary(
                gradient_node,
                source_grad,
                torch.cuda.current_stream(),
            )
            combined_grad = executor._all_to_all_ep_boundary(
                combine_node,
                dispatched_grad,
                torch.cuda.current_stream(),
            )
            torch.testing.assert_close(combined_grad, source_grad)

            num_local_experts = counts.numel() // world_size
            local_start = rank * num_local_experts
            nonzero_slice = local_counts[
                local_start : local_start + num_local_experts
            ]
            # TorchTitan's pinned permutation reserves a statically bounded
            # tail in addition to aligning every local expert group. Grouped
            # GEMM consumes rows only through the final count offset.
            assert int(local_counts.sum().item()) <= dispatched.shape[0]
            assert int(nonzero_slice.sum().item()) == int(
                local_counts.sum().item()
            )
            assert int(local_counts[:local_start].sum().item()) == 0
            assert (
                int(
                    local_counts[
                        local_start + num_local_experts :
                    ].sum().item()
                )
                == 0
            )

            results.append(
                {
                    "iteration": iteration,
                    "counts": counts.cpu().tolist(),
                    "dispatched_shape": list(dispatched.shape),
                    "local_counts": local_counts.cpu().tolist(),
                }
            )

        degree_one = make_executor(None, rank)
        degree_one_dispatch = make_a2a_node(
            "degree_one_dispatch",
            TaskType.FWD_A2A,
            "incoming",
            "degree_one_expert",
            [rank],
        )
        degree_one_combine = make_a2a_node(
            "degree_one_combine",
            TaskType.FWD_A2A,
            "outgoing",
            "degree_one_expert",
            [rank],
        )
        counts = torch.tensor(
            count_patterns[0][rank],
            device=device,
            dtype=torch.int64,
        )
        x = torch.randn(
            (int(counts.sum().item()), 32),
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        reference_output = experts(x, counts)
        dispatched, aligned_counts = degree_one._all_to_all_ep_boundary(
            degree_one_dispatch,
            x,
            torch.cuda.current_stream(),
            routing_counts=counts,
        )
        expert_output = compiled_experts(dispatched, aligned_counts)
        combined = degree_one._all_to_all_ep_boundary(
            degree_one_combine,
            expert_output,
            torch.cuda.current_stream(),
        )
        torch.testing.assert_close(
            combined,
            reference_output,
            atol=2e-3,
            rtol=2e-2,
        )
        results.append(
            {
                "degree_one": True,
                "counts": counts.cpu().tolist(),
                "dispatched_shape": list(dispatched.shape),
                "local_counts": aligned_counts.cpu().tolist(),
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
        print(
            json.dumps(
                {
                    "rank": rank,
                    "ok": False,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "counters": dynamo_counters(),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
