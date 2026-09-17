"""Ring-attention context parallelism on two GPUs, driven by the TrainingDAG backend.

    python examples/test_harness.py \
      --test-file examples/test_ring_attn.py \
      --base-schedule examples/base-schedules/cp2_ring_dp.json \
      --schedule custom --steps 2

CP shards the *sequence*, not the parameters: rank r feeds its own chunk of x, k,
v and the labels, the projection weights are identical on every rank
(global_weights) and kept in sync by `replicate`, and the model's ring step count
must equal the place-group size. With `--steps 1` on one device the same model
is dense attention over the full sequence -- the baseline
experiments/check_cp_equivalence.py compares against.

fp32 and Inductor off for the same reason as test_tp_mlp.py: this checks that CP
computes the right thing, not how fast.
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

from models.ring_attn import RingAttn, global_weights

logger = create_logger("test_ring_attn", LOG_LEVEL)
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _raw_metrics(args, iter_times, losses, peak_memory_stats):
    info = dict(getattr(piper_metadata, "schedule_info", {}) or {})
    return {
        "dp_rank": int(os.environ["PIPER_DP_RANK"]),
        "model": f"ring_attn_d{args.dim}_h{args.heads}_s{args.seq}_cp{args.steps}",
        "schedule": info.get(
            "name", os.path.splitext(os.path.basename(args.schedule_directives_file))[0]
        ),
        "schedule_directives_file": args.schedule_directives_file,
        "pp": info.get("pp_degree"),
        "dp": info.get("dp_degree"),
        "cp": args.steps,
        "batch_size": args.batch_size,
        "num_microbatches": int(info.get("num_microbatches", 1)),
        "seq_len": args.seq,
        "seq_local": args.seq // args.steps,
        "seed": args.seed,
        "iter_times_s": [float(t) for t in iter_times],
        "losses": [float(l) for l in losses],
        "peak_memory_by_rank": {
            int(rank): int(max_alloc) for rank, max_alloc in peak_memory_stats
        },
    }


def main(args, pg):
    dtype = _DTYPES[args.dtype]
    loss_fn = lambda output, labels: (output.float() - labels.float()).pow(2).mean()

    # The CP rank is the position in the place group, which Piper reports as
    # dp_rank (its world is pp x dp with no third axis).
    cp_rank = int(os.environ["PIPER_DP_RANK"])
    assert args.seq % args.steps == 0, "sequence must split evenly across ring steps"
    s_local = args.seq // args.steps

    # Every rank draws the same full-sequence tensors from one seed and keeps its
    # chunk, so a CP=2 run and the dense CP=1 run see identical data and labels.
    g = torch.Generator().manual_seed(args.seed)
    full = [
        torch.randn(args.batch_size, args.seq, args.dim, generator=g, dtype=dtype)
        for _ in range(4)
    ]
    sl = slice(cp_rank * s_local, (cp_rank + 1) * s_local)
    x, k, v, y = (t[:, sl].contiguous() for t in full)

    piper_setup(
        RingAttn,
        model_args=(args.dim, args.heads, args.steps),
        optim_fn=torch.optim.Adam,
        example_inputs=[x, k, v],
        example_outputs=y,
        model_dtype=dtype,
        pg=pg,
        temp_dir=args.temp_dir,
        param_overrides=global_weights(args.dim, args.seed, dtype),
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
    metrics = _raw_metrics(args, iter_times, losses, peak_memory_stats)
    out_dir = os.path.dirname(os.path.abspath(args.schedule_directives_file))
    path = os.path.join(out_dir, f"cp_metrics_dp{metrics['dp_rank']}.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
    logger.info(f"wrote {path}: losses={metrics['losses']}")
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Ring-attention CP on the TrainingDAG backend")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seq", type=int, default=64, help="Full sequence length.")
    parser.add_argument(
        "--steps", type=int, default=2,
        help="Ring steps = CP degree; must equal the place-group size.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="fp32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--viz", action="store_true", default=False)
    parser.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    parser.add_argument("--use-inductor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pp-outer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--schedule-directives-file", type=str,
        default="examples/base-schedules/cp2_ring_dp.json",
    )
    return parser.parse_args(argv)
