"""End-to-end Qwen test for the JSON-driven TrainingDAG backend."""
import ray
import torch
import argparse
import time
import os

from src.compile import piper_setup
from src.piper import (
    piper_exec_dag,
    piper_measure_dags,
    piper_measure_profile_modes,
    piper_profile_dags,
    simulate_profiled_modes,
)
from src.profiling import ProfileConfig
from src.schedule import load_schedule_directives
from src.state import piper_metadata, create_logger, LOG_LEVEL

from models.qwen3 import PiperQwen3Model, QWEN_MODEL_NAMES, create_qwen3_config
from torchtitan.models.qwen3.model.model import precompute_rope_cache

logger = create_logger("test_qwen", LOG_LEVEL)


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
    batch_size = args.batch_size

    config = create_qwen3_config(args.model)
    num_stages = int(
        getattr(args, "num_stages", 0)
        or _derive_num_stages(args.schedule_directives_file)
    )

    x = torch.randint(0, config.vocab_size, (batch_size, args.seq_len))
    y = torch.randint(0, config.vocab_size, (batch_size, args.seq_len))

    _ce = torch.nn.CrossEntropyLoss()
    loss_fn = lambda output, labels: _ce(output.view(-1, output.size(-1)), labels.view(-1))

    rope_cache = precompute_rope_cache(
        config.head_dim,
        config.max_seq_len,
        config.rope_theta,
    )
    piper_setup(
        PiperQwen3Model,
        model_args=(config, num_stages),
        optim_fn=torch.optim.Adam,
        example_inputs=[x],
        example_outputs=y,
        activation_checkpointing=args.activation_checkpointing,
        model_dtype=torch.bfloat16,
        pg=pg,
        nsight=args.nsight,
        temp_dir=args.temp_dir,
        visualize_dag=args.viz,
        const_attrs={"rope_cache": rope_cache},
        use_inductor=args.use_inductor,
        pp_outer=args.pp_outer,
        schedule_directives_file=args.schedule_directives_file,
    )

    del x, y

    actors = piper_metadata.actors

    logger.info(f"Running {args.warmup} warmup iterations")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn)
        if args.iteration_sleep > 0:
            time.sleep(args.iteration_sleep)

    logger.info("Running %d timed iterations", args.iters)
    timing = piper_measure_dags(
        loss_fn,
        iterations=args.iters,
        iteration_sleep=args.iteration_sleep,
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

    if args.measure_profile_modes_only:
        mode_samples = piper_measure_profile_modes(
            loss_fn,
            warmup_iters=args.profile_warmup_iters,
            repeat_iters=args.profile_iters,
            block_iters=args.profile_mode_block_iters,
            profile_modes=("interference_free", "normal"),
        )
        metrics["uninstrumented_normal_step_time_samples_s"] = [
            sample / 1_000_000.0 for sample in mode_samples["normal"]
        ]
        metrics["uninstrumented_interference_free_step_time_samples_s"] = [
            sample / 1_000_000.0
            for sample in mode_samples["interference_free"]
        ]

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
            if args.iteration_sleep > 0:
                time.sleep(args.iteration_sleep)
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


def _derive_num_stages(schedule_directives_file: str) -> int:
    schedule_directives = load_schedule_directives(schedule_directives_file)
    num_stages = sum(
        1
        for directive in schedule_directives
        if isinstance(directive, dict) and directive.get("op") == "place"
    )
    if num_stages <= 0:
        raise ValueError(
            f"schedule directives file must contain at least one place directive: "
            f"{schedule_directives_file}"
        )
    return num_stages


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Test JSON-driven Qwen TrainingDAG execution"
    )
    parser.add_argument(
        "--model",
        choices=QWEN_MODEL_NAMES,
        default="9M",
        help="Model configuration (default: 9M)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--iteration-sleep", type=float, default=0.0)
    parser.add_argument('--activation-checkpointing', action='store_true', default=False)
    parser.add_argument("--nsight", action="store_true", default=False,
                        help="Whether to use Nsight Systems for tracing")
    parser.add_argument("--viz", action="store_true", default=False,
                        help="Save schedule and per-rank DAG visualizations")
    parser.add_argument("--temp-dir", default="/tmp/piper/ray_tmp",
                        help="Ray temp directory (default: /tmp/piper/ray_tmp)")
    parser.add_argument(
        "--use-inductor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether actors torch.compile stage GraphModules in _load_stage (default: true)",
    )
    parser.add_argument(
        "--pp-outer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use PP as the outer placement dim. Each PP rank's DP/EP group is "
            "kept on one node when possible, or within one GB200 NVLink domain "
            "when it spans nodes. Default: false (DP/EP outer, PP inner)."
        ),
    )
    parser.add_argument(
        "--schedule-directives-file",
        type=str,
        default="examples/base-schedules/pp2.json",
        help="JSON file containing schedule directives for the piper backend",
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
        default=1,
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
    parser.add_argument(
        "--measure-profile-modes-only",
        action="store_true",
        help=(
            "Measure normal and interference-free stream policies without "
            "per-node CUDA-event instrumentation."
        ),
    )
    parser.add_argument("--profile-mode-block-iters", type=int, default=1)
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
    metrics["dependency_replay_interference_reduction_pct"] = (
        comparison.dependency_replay_interference_reduction_pct
    )
    metrics["instrumented_policy_reduction_pct"] = (
        comparison.instrumented_policy_reduction_pct
    )
    metrics["profile_event_overhead_pct"] = comparison.profile_event_overhead_pct
    metrics["mode_profile_event_overhead_pct"] = (
        comparison.mode_profile_event_overhead_pct
    )
    metrics["observed_interference_reduction_pct"] = (
        comparison.observed_interference_reduction_pct
    )
    metrics["envelope_modeled_interference_reduction_pct"] = (
        comparison.envelope_modeled_interference_reduction_pct
    )
    metrics["normal_dependency_replay_excess_pct"] = (
        comparison.normal_dependency_replay_excess_pct
    )
    metrics["observed_normal_envelope_step_time_samples_s"] = [
        sample / 1_000_000.0
        for sample in comparison.observed_normal_envelope_samples_us
    ]
    if comparison.instrumented_normal_replay_error is not None:
        metrics["instrumented_normal_replay_error_pct"] = (
            comparison.instrumented_normal_replay_error.signed_pct
        )
    if comparison.instrumented_interference_free_replay_error is not None:
        metrics["instrumented_interference_free_replay_error_pct"] = (
            comparison.instrumented_interference_free_replay_error.signed_pct
        )
    instrumented = plan.profile_metadata.get(
        "instrumented_global_step_samples_us_by_mode", {}
    ).get("normal", ())
    metrics["instrumented_normal_profile_step_time_samples_s"] = [
        float(sample) / 1_000_000.0 for sample in instrumented
    ]
    uninstrumented = plan.profile_metadata.get(
        "uninstrumented_global_step_samples_us_by_mode", {}
    )
    metrics["uninstrumented_normal_step_time_samples_s"] = [
        float(sample) / 1_000_000.0
        for sample in uninstrumented.get("normal", ())
    ]
    metrics["uninstrumented_interference_free_step_time_samples_s"] = [
        float(sample) / 1_000_000.0
        for sample in uninstrumented.get("interference_free", ())
    ]
