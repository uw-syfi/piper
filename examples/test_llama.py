"""End-to-end LLaMA test for the JSON-driven TrainingDAG backend."""
import argparse
import os
import time

import ray
import torch

from src.piper import (
    piper_exec_dag,
    piper_measure_dags,
    piper_profile_dags,
    simulate_profiled_modes,
)
from src.profiling import ProfileConfig
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

    logger.info("Running %d timed iterations", args.iters)
    timing = piper_measure_dags(
        loss_fn,
        iterations=args.iters,
        iteration_sleep=1.0,
        log_stats=True,
        collect_peak_memory=True,
    )
    iter_times = list(timing.step_samples_s)
    peak_memory_samples_by_rank = {
        rank: list(samples)
        for rank, samples in timing.peak_memory_samples_by_rank.items()
    }

    peak_memory_stats = [
        (rank, max(samples))
        for rank, samples in sorted(peak_memory_samples_by_rank.items())
    ]

    metrics = _raw_metrics(args, iter_times, peak_memory_stats)
    metrics["peak_memory_samples_by_rank"] = peak_memory_samples_by_rank

    if args.profile_and_simulate:
        profile_config = ProfileConfig(
            warmup_iters=args.profile_warmup_iters,
            repeat_iters=args.profile_iters,
        )
        plan = piper_profile_dags(
            loss_fn,
            profile_config,
            profile_modes=("interference_free", "normal"),
            output_path=args.step_time_profile_output or None,
        )
        comparison = simulate_profiled_modes(
            plan, actual_step_samples_s=iter_times
        )
        _add_profile_comparison_metrics(metrics, comparison, plan)

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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
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
    parser.add_argument(
        "--profile-and-simulate",
        dest="profile_and_simulate",
        action="store_true",
        default=False,
        help="Profile and simulate both normal and interference-free step times.",
    )
    parser.add_argument("--profile-warmup-iters", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=30)
    parser.add_argument("--step-time-profile-output", default="")
    return parser.parse_args(argv)


def _add_profile_comparison_metrics(metrics, comparison, plan) -> None:
    metrics["interference_free_step_time_us"] = (
        comparison.interference_free_statistics.estimate
    )
    metrics["interference_free_step_time_s"] = (
        comparison.interference_free_statistics.estimate / 1_000_000.0
    )
    metrics["simulated_interference_free_step_time_samples_s"] = [
        sample / 1_000_000.0
        for sample in comparison.interference_free_samples_us
    ]
    metrics["simulated_normal_step_time_us"] = comparison.normal_statistics.estimate
    metrics["simulated_normal_step_time_s"] = (
        comparison.normal_statistics.estimate / 1_000_000.0
    )
    metrics["simulated_normal_step_time_samples_s"] = [
        sample / 1_000_000.0 for sample in comparison.normal_samples_us
    ]
    if comparison.normal_prediction_error is not None:
        metrics["simulated_normal_prediction_error_pct"] = (
            comparison.normal_prediction_error.signed_pct
        )
        metrics["simulated_normal_absolute_error_pct"] = (
            comparison.normal_prediction_error.absolute_pct
        )
    metrics["predicted_interference_reduction_pct"] = (
        comparison.predicted_interference_reduction_pct
    )
    metrics["profile_event_overhead_pct"] = comparison.profile_event_overhead_pct
    instrumented = plan.profile_metadata.get(
        "instrumented_global_step_samples_us_by_mode", {}
    ).get("normal", ())
    metrics["instrumented_normal_profile_step_time_samples_s"] = [
        float(sample) / 1_000_000.0 for sample in instrumented
    ]
