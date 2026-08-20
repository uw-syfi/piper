"""End-to-end Qwen test for the JSON-driven TrainingDAG backend."""
import hashlib

import ray
import torch
import argparse
import time
import os

from src.compile import piper_setup
from src.model_state import DEFAULT_MODEL_SEED
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

from models.qwen3 import (
    AFDQwen3Model,
    PiperQwen3Model,
    QWEN_MODEL_NAMES,
    create_qwen3_config,
    qwen_expert_shard_degree,
)
from torchtitan.models.qwen3.model.model import precompute_rope_cache

try:
    from .test_harness import _add_profile_comparison_metrics
except ImportError:
    from test_harness import _add_profile_comparison_metrics

logger = create_logger("test_qwen", LOG_LEVEL)


def _raw_metrics(args, iter_times, peak_memory_stats, input_data_sha256):
    """Assemble the single program driver's measurements for the harness.

    No derived statistics are computed here; the harness summarizes the metrics
    and writes the CSV. ``peak_memory_by_rank`` maps actor rank -> peak bytes.
    """
    info = dict(getattr(piper_metadata, "schedule_info", {}) or {})
    return {
        "program_rank": 0,
        "model": args.model,
        "schedule": info.get(
            "name", os.path.splitext(os.path.basename(args.schedule_directives_file))[0]
        ),
        "schedule_directives_file": args.schedule_directives_file,
        "pp": info.get("pp_degree"),
        "actor_groups": info.get("actor_groups"),
        "world_size": info.get("world_size"),
        "input_actor_group_id": info.get("input_actor_group_id"),
        "input_actor_group_size": info.get("input_actor_group_size", 1),
        "batch_size": args.batch_size,
        "num_microbatches": int(info.get("num_microbatches", 1)),
        "seq_len": args.seq_len,
        "model_seed": args.model_seed,
        "input_seed": args.input_seed,
        "input_data_sha256": input_data_sha256,
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

    input_generator = torch.Generator(device="cpu")
    input_generator.manual_seed(args.input_seed)
    x = torch.randint(
        0,
        config.vocab_size,
        (batch_size, args.seq_len),
        generator=input_generator,
    )
    y = torch.randint(
        0,
        config.vocab_size,
        (batch_size, args.seq_len),
        generator=input_generator,
    )
    input_data_sha256 = hashlib.sha256(
        x.contiguous().numpy().tobytes() + y.contiguous().numpy().tobytes()
    ).hexdigest()

    _ce = torch.nn.CrossEntropyLoss()
    loss_fn = lambda output, labels: _ce(output.view(-1, output.size(-1)), labels.view(-1))

    rope_cache = precompute_rope_cache(
        config.head_dim,
        config.max_seq_len,
        config.rope_theta,
    )
    model_cls = _model_class_for_schedule(args.schedule_directives_file)
    shard_degree = qwen_expert_shard_degree(
        load_schedule_directives(args.schedule_directives_file)
    )
    logger.info("Using Qwen wrapper %s", model_cls.__name__)
    piper_setup(
        model_cls,
        model_args=(config, num_stages, shard_degree),
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
        dynamic_input_dims={0: (1,)},
        model_seed=args.model_seed,
    )

    del x, y

    actors = piper_metadata.actors

    logger.info(f"Running {args.warmup} warmup iterations")
    for _ in range(args.warmup):
        piper_exec_dag(loss_fn)
        if args.iteration_sleep > 0:
            time.sleep(args.iteration_sleep)

    if args.iters < 0:
        raise ValueError("--iters must be non-negative")
    if args.iters == 0 and not args.pytorch_profiler:
        raise ValueError("--iters 0 is only valid for a PyTorch-profiler-only run")
    if args.iters:
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
    else:
        logger.info("Skipping normal timed iterations for profiler-only run")
        iter_times = []
        peak_memory_samples_by_rank = {}

    peak_memory_stats = [
        (rank, max(samples))
        for rank, samples in sorted(peak_memory_samples_by_rank.items())
    ]

    metrics = _raw_metrics(
        args,
        iter_times,
        peak_memory_stats,
        input_data_sha256,
    )
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
        if args.pytorch_profiler_iters < 1:
            raise ValueError("--pytorch-profiler-iters must be at least 1")
        profile_dir = getattr(args, "profile_dir", "") or os.path.join(
            "out", "pytorch_profiles"
        )
        logger.info(
            "Running %d PyTorch-profiler warmup iteration(s) and %d active iteration(s)",
            args.pytorch_profiler_warmup_iters,
            args.pytorch_profiler_iters,
        )
        ray.get([
            actor.start_pytorch_profiler.remote(
                warmup_iters=args.pytorch_profiler_warmup_iters,
                active_iters=args.pytorch_profiler_iters,
            )
            for actor in actors.values()
        ])
        for _ in range(
            args.pytorch_profiler_warmup_iters + args.pytorch_profiler_iters
        ):
            piper_exec_dag(loss_fn)
            if args.iteration_sleep > 0:
                time.sleep(args.iteration_sleep)
        ray.get([
            actor.stop_pytorch_profiler.remote(profile_dir)
            for actor in actors.values()
        ])
        if args.iteration_sleep > 0:
            time.sleep(args.iteration_sleep)

    if args.nsight:
        logger.info("Stopping Piper actors so Nsight Systems reports are flushed")
        try:
            ray.get([actor.__ray_terminate__.remote() for actor in actors.values()])
        except ray.exceptions.ActorDiedError as exc:
            logger.info(f"Piper actors stopped for Nsight flush: {exc}")
    return metrics


def _derive_num_stages(schedule_directives_file: str) -> int:
    schedule_directives = load_schedule_directives(schedule_directives_file)
    pp_indices = {
        int(directive["filter"]["PP"])
        for directive in schedule_directives
        if isinstance(directive, dict)
        and directive.get("op") == "place"
        and isinstance(directive.get("filter"), dict)
        and isinstance(directive["filter"].get("PP"), int)
    }
    if not pp_indices:
        raise ValueError(
            "schedule directives file must contain a place directive with an "
            f"exact PP filter: "
            f"{schedule_directives_file}"
        )
    if min(pp_indices) != 0:
        raise ValueError(
            f"schedule PP indices must start at 0, got {sorted(pp_indices)}"
        )
    return max(pp_indices) + 1


def _model_class_for_schedule(schedule_directives_file: str):
    """Select the AFD wrapper when placement addresses ATTN/FFN regions."""
    directives = load_schedule_directives(schedule_directives_file)
    uses_afd_regions = any(
        directive.get("op") == "place"
        and isinstance(directive.get("filter"), dict)
        and (
            "ATTN" in directive["filter"]
            or "FFN" in directive["filter"]
        )
        for directive in directives
    )
    return AFDQwen3Model if uses_afd_regions else PiperQwen3Model


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
    parser.add_argument("--model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument("--input-seed", type=int, default=DEFAULT_MODEL_SEED + 1)
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
            "Compatibility option retained for existing commands. Explicit "
            "place device sets now determine ActorGroup placement."
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
             "one chrome trace per global actor rank.",
    )
    parser.add_argument(
        "--pytorch-profiler-warmup-iters",
        type=int,
        default=2,
        help="Number of profiler-instrumented warmup iterations discarded before capture.",
    )
    parser.add_argument(
        "--pytorch-profiler-iters",
        type=int,
        default=1,
        help="Number of active iterations retained in the PyTorch trace.",
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
