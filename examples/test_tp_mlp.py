"""End-to-end tensor-parallel MLP on two GPUs, driven by the TrainingDAG backend.

    python examples/test_harness.py \
      --test-file examples/test_tp_mlp.py \
      --base-schedule examples/base-schedules/tp2.json \
      --schedule custom

`--schedule custom` passes the base schedule through untouched, so the harness
needs no TP-specific changes: tp2.json already carries its own place, shard_tensor
and split directives.

Defaults differ from the LLaMA/Qwen examples on purpose. This example exists to
check that TP computes the right thing, not to measure throughput, so it runs in
fp32 (bf16's ~3 decimal digits would swamp the comparison) and with Inductor off
(less to go wrong between the IR and the number).
"""
import argparse
import json
import os
import time

import ray
import torch

from src.compile import piper_setup
from src.piper import piper_exec_dag
from src.state import LOG_LEVEL, create_logger, piper_metadata

from models.tp_mlp import TPMlp, global_weights, shard_weights

logger = create_logger("test_tp_mlp", LOG_LEVEL)

_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _raw_metrics(args, iter_times, losses, peak_memory_stats):
    info = dict(getattr(piper_metadata, "schedule_info", {}) or {})
    return {
        "dp_rank": int(os.environ["PIPER_DP_RANK"]),
        "model": f"tp_mlp_d{args.dim}_h{args.hidden}_tp{args.tp}",
        "schedule": info.get(
            "name", os.path.splitext(os.path.basename(args.schedule_directives_file))[0]
        ),
        "schedule_directives_file": args.schedule_directives_file,
        "pp": info.get("pp_degree"),
        "dp": info.get("dp_degree"),
        "batch_size": args.batch_size,
        "num_microbatches": int(info.get("num_microbatches", 1)),
        "seq_len": args.dim,
        "seed": args.seed,
        "init": args.init,
        "tp": args.tp,
        "iter_times_s": [float(t) for t in iter_times],
        "losses": [float(l) for l in losses],
        "peak_memory_by_rank": {
            int(rank): int(max_alloc) for rank, max_alloc in peak_memory_stats
        },
    }


def main(args, pg):
    # The TP degree is the size of the place group, which Piper reports as
    # dp_degree: its world is pp x dp and has no third axis.
    info = piper_metadata.schedule_info or {}
    dtype = _DTYPES[args.dtype]

    loss_fn = lambda output, labels: (output.float() - labels.float()).pow(2).mean()

    # With --init fixed, every rank and every TP degree starts from the same
    # global weights, sliced for this rank. That is what makes a TP=2 run
    # comparable to the TP=1 baseline; Piper's default per-rank seeding
    # (manual_seed(1000 * global_rank + stage_id)) makes any such comparison
    # meaningless.
    param_overrides = None
    if args.init == "fixed":
        tp_rank = int(os.environ["PIPER_DP_RANK"])
        param_overrides = shard_weights(
            global_weights(args.dim, args.hidden, args.seed, dtype),
            tp_rank,
            args.tp,
        )

    # Seed the data. Piper seeds parameters per rank already
    # (manual_seed(1000 * global_rank + stage_id)), so with a fixed data seed a
    # rerun of the same schedule is bit-reproducible and two schedules are
    # comparable. Without this the inputs differ per run and any loss comparison
    # between schedules measures the inputs, not the schedule.
    torch.manual_seed(args.seed)
    x = torch.randn(args.batch_size, args.dim, dtype=dtype)
    y = torch.randn(args.batch_size, args.dim, dtype=dtype)

    piper_setup(
        TPMlp,
        model_args=(args.dim, args.hidden, args.tp),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        model_dtype=dtype,
        pg=pg,
        temp_dir=args.temp_dir,
        param_overrides=param_overrides,
        visualize_dag=args.viz,
        use_inductor=args.use_inductor,
        pp_outer=args.pp_outer,
        schedule_directives_file=args.schedule_directives_file,
    )

    actors = piper_metadata.actors
    logger.info(f"Running {args.warmup} warmup iterations")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn, log_stats=True)

    ray.get([actor.reset_peak_memory.remote() for actor in actors.values()])
    ray.get([actor.reset_step_timestamps.remote() for actor in actors.values()])
    logger.info(f"Running {args.iters} timed iterations")
    iter_times, losses = [], []
    for _ in range(args.iters):
        start = time.perf_counter()
        step_losses = piper_exec_dag(loss_fn, log_stats=True)
        iter_times.append(time.perf_counter() - start)
        losses.extend(step_losses or [])

    peak_memory_stats = ray.get(
        [actor.get_and_reset_peak_memory_stats.remote() for actor in actors.values()]
    )
    step_timestamps = ray.get(
        [actor.get_step_timestamps.remote() for actor in actors.values()]
    )
    metrics = _raw_metrics(args, iter_times, losses, peak_memory_stats)
    # Wall-clock, so samples from other driver processes line up: TP peers live
    # in different processes (one per dp_rank), so skew can only be seen by
    # joining these across the per-dp_rank artifacts.
    metrics["step_timestamps"] = {
        str(rank): [list(t) for t in ts] for rank, ts in step_timestamps
    }
    # The harness CSV drops per-iteration losses, and Ray can truncate driver
    # logs at exit, so write our own artifact next to the generated schedule.
    artifact = os.path.join(
        getattr(piper_metadata, "artifact_dir", "out"),
        f"tp_metrics_dp{metrics['dp_rank']}.json",
    )
    with open(artifact, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(
        "dp_rank=%s pp=%s dp=%s losses=%s -> %s",
        metrics["dp_rank"], metrics["pp"], metrics["dp"],
        [round(l, 6) for l in metrics["losses"]], artifact,
    )

    if args.pytorch_profiler:
        profile_dir = getattr(args, "profile_dir", "") or os.path.join(
            "out", "pytorch_profiles"
        )
        logger.info(f"Running {args.pytorch_profiler_iters} PyTorch-profiled iterations")
        ray.get([actor.start_pytorch_profiler.remote() for actor in actors.values()])
        for _ in range(args.pytorch_profiler_iters):
            piper_exec_dag(loss_fn)
        ray.get([
            actor.stop_pytorch_profiler.remote(profile_dir) for actor in actors.values()
        ])

    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Tensor-parallel MLP on the TrainingDAG backend")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument(
        "--tp", type=int, default=2,
        help="TP degree the model is authored for; must equal the place-group size.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="fp32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--init", choices=("random", "fixed"), default="fixed",
        help="fixed: identical global weights sliced per rank, so TP degrees "
             "are comparable. random: Piper's per-rank seeded noise.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--viz", action="store_true", default=False)
    parser.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    parser.add_argument("--use-inductor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pp-outer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--schedule-directives-file",
        type=str,
        default="examples/base-schedules/tp2.json",
    )
    parser.add_argument("--pytorch-profiler", action="store_true", default=False)
    parser.add_argument("--pytorch-profiler-iters", type=int, default=3)
    return parser.parse_args(argv)
