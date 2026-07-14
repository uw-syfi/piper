"""End-to-end LLaMA test for the JSON-driven TrainingDAG backend."""
import argparse
import os
import time

import ray
import torch

from src.piper import piper_exec_dag
from src.compile import piper_setup
from src.state import (
    LOG_LEVEL,
    create_logger,
    piper_metadata,
)

from models.llama3 import (
    LLAMA_1B,
    LLAMA_3B,
    LLAMA_8B,
    LLAMA_70B,
    LLAMA_DEBUG,
    Transformer,
    precompute_freqs_cis,
)

logger = create_logger("test_llama", LOG_LEVEL)


def _model_config(name: str):
    match name:
        case "debug":
            return LLAMA_DEBUG
        case "1b":
            return LLAMA_1B
        case "3b":
            return LLAMA_3B
        case "8b":
            return LLAMA_8B
        case "70b":
            return LLAMA_70B
    raise ValueError(f"Unknown LLaMA model: {name}")


def _raw_metrics(args, iter_times, peak_memory_stats):
    """Assemble raw per-dp-rank measurements + run config for the harness.

    No derived statistics are computed here; the harness summarizes the metrics
    and writes the CSV. ``peak_memory_by_rank`` maps global rank -> peak bytes.
    """
    info = dict(getattr(piper_metadata, "schedule_info", {}) or {})
    return {
        "dp_rank": int(os.environ["PIPER_DP_RANK"]),
        "model": args.model,
        "schedule": info.get(
            "name", os.path.splitext(os.path.basename(args.schedule_directives_file))[0]
        ),
        "schedule_directives_file": args.schedule_directives_file,
        "pp": info.get("pp_degree"),
        "dp": info.get("dp_degree"),
        "batch_size": args.batch_size,
        "num_microbatches": int(info.get("num_microbatches", 1)),
        "seq_len": args.seq_len,
        "iter_times_s": [float(t) for t in iter_times],
        "peak_memory_by_rank": {
            int(rank): int(max_alloc) for rank, max_alloc in peak_memory_stats
        },
    }


def main(args, pg):
    llama_config = _model_config(args.model)

    loss_mod = torch.nn.CrossEntropyLoss()
    loss_fn = lambda output, labels: loss_mod(output.view(-1, output.size(-1)), labels.view(-1))

    x = torch.randint(0, llama_config.vocab_size, (args.batch_size, args.seq_len))
    y = torch.randint(0, llama_config.vocab_size, (args.batch_size, args.seq_len))

    freqs_cis = precompute_freqs_cis(
        llama_config.dim // llama_config.n_heads,
        args.seq_len,
        llama_config.rope_theta,
    )
    causal = torch.full((args.seq_len, args.seq_len), float("-inf"))
    causal = torch.triu(causal, diagonal=1)
    mask = torch.hstack([torch.zeros((args.seq_len, 0)), causal])

    piper_setup(
        Transformer,
        model_args=(llama_config, args.seq_len),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        activation_checkpointing=args.activation_checkpointing,
        num_checkpoints=args.num_checkpoints,
        model_dtype=torch.bfloat16,
        pg=pg,
        nsight=args.nsight,
        temp_dir=args.temp_dir,
        visualize_dag=args.viz,
        const_attrs={"freqs_cis": freqs_cis, "mask": mask},
        use_inductor=args.use_inductor,
        pp_outer=args.pp_outer,
        schedule_directives_file=args.schedule_directives_file,
    )

    actors = piper_metadata.actors
    logger.info(f"Running {args.warmup} warmup iterations")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn, log_stats=True)
        time.sleep(1)

    ray.get([actor.reset_peak_memory.remote() for actor in actors.values()])
    logger.info(f"Running {args.iters} timed iterations")
    iter_times = []
    for _ in range(args.iters):
        start = time.perf_counter()
        piper_exec_dag(loss_fn, log_stats=True)
        end = time.perf_counter()
        iter_times.append(end - start)
        time.sleep(1)

    peak_memory_stats = ray.get(
        [actor.get_and_reset_peak_memory_stats.remote() for actor in actors.values()]
    )

    metrics = _raw_metrics(args, iter_times, peak_memory_stats)

    if args.pytorch_profiler:
        profile_dir = getattr(args, "profile_dir", "") or os.path.join(
            "out", "pytorch_profiles"
        )
        logger.info(f"Running {args.pytorch_profiler_iters} PyTorch-profiled iterations")
        ray.get([actor.start_pytorch_profiler.remote() for actor in actors.values()])
        for _ in range(args.pytorch_profiler_iters):
            piper_exec_dag(loss_fn)
            time.sleep(1)
        ray.get([
            actor.stop_pytorch_profiler.remote(profile_dir)
            for actor in actors.values()
        ])

    if args.nsight:
        logger.info("Stopping Piper actors so Nsight Systems reports are flushed")
        try:
            ray.get([actor.__ray_terminate__.remote() for actor in actors.values()])
        except ray.exceptions.ActorDiedError as exc:
            logger.info(f"Piper actors stopped for Nsight flush: {exc}")
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run LLaMA with JSON-driven TrainingDAG execution")
    parser.add_argument("--model", choices=["debug", "1b", "3b", "8b", "70b"], default="debug")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--activation-checkpointing", action="store_true", default=False)
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=1,
        help="Number of sequential activation-checkpoint regions per Piper bucket",
    )
    parser.add_argument("--nsight", action="store_true", default=False)
    parser.add_argument("--viz", action="store_true", default=False)
    parser.add_argument("--temp-dir", default="/tmp/piper/ray_tmp")
    parser.add_argument("--use-inductor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pp-outer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--schedule-directives-file",
        type=str,
        default="examples/base-schedules/pp2.json",
        help="JSON file containing TrainingDAG schedule directives",
    )
    parser.add_argument(
        "--pytorch-profiler",
        action="store_true",
        default=False,
        help="Run extra iterations under torch.profiler on every actor and write "
             "per-actor chrome traces (combined per dp-rank by test_harness).",
    )
    parser.add_argument(
        "--pytorch-profiler-iters",
        type=int,
        default=3,
        help="Number of iterations to run under the PyTorch profiler.",
    )
    return parser.parse_args(argv)
