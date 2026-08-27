"""End-to-end Qwen test for the JSON-driven TrainingDAG backend."""
import ray
import torch
import argparse
import time
import os

from src.compile import piper_setup
from src.piper import piper_exec_dag, PiperResume
from src.schedule import load_schedule_directives
from src.state import piper_metadata, create_logger, LOG_LEVEL

from models.qwen3 import PiperQwen3Model, create_qwen3_config
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


def _run_standby(dp_rank, args, loss_fn):
    """Park until promoted or shut down; on promotion, receive the survivor's
    state and train the remaining iterations as the failed rank's replacement.

    dp_rank: this standby's dp_rank (>= dp_degree).
    args: parsed harness arguments.
    loss_fn: loss used by piper_exec_dag.
    """
    coordinator = piper_metadata.coordinator
    actor = piper_metadata.actors[0] # pp_degree == 1
    ray.get(actor.prepare_standby_state.remote())
    logger.info(f"standby dp_rank {dp_rank}: initialized and parked; "
                "waiting for promotion or shutdown")
    cmd = ray.get(coordinator.wait_for_cmd.remote())
    if cmd.get("op") == "promote":
        ray.get(
            actor.join_standby_group.remote(cmd["new_ranks"]),
            timeout=180,
        )
        logger.info(f"standby dp_rank {dp_rank}: joined NCCL group "
                    f"{cmd['new_ranks']}")
        next_iter = ray.get(actor.wait_state_loaded.remote(), timeout=600)
        total = args.warmup + args.iters
        logger.info(f"standby dp_rank {dp_rank}: state loaded; running "
                    f"iterations {next_iter}..{total - 1} as replacement")
        for _ in range(total - next_iter):
            piper_exec_dag(loss_fn)
        logger.info(f"standby dp_rank {dp_rank}: replacement training finished")
    else:
        logger.info(f"standby dp_rank {dp_rank}: shutdown received; exiting")
    return None


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

    # Standby ranks never train: park until promoted or shut down.
    dp_rank = int(os.environ["PIPER_DP_RANK"])
    dp_degree = int(os.environ["PIPER_DP_DEGREE"])
    if dp_rank >= dp_degree:
        return _run_standby(dp_rank, args, loss_fn)

    # No step_timeout during warmup; afterward 5x the last warmup step
    # (~steady step time), floored at 5s.
    logger.info(f"Running {args.warmup} warmup iterations")
    last_warmup_time = None
    step_timeout = None
    iter_times = []
    total = args.warmup + args.iters
    it = 0
    while it < total:
        try:
            if it < args.warmup:
                t0 = time.perf_counter()
                piper_exec_dag(loss_fn)
                last_warmup_time = time.perf_counter() - t0
                if args.iteration_sleep > 0:
                    time.sleep(args.iteration_sleep)
                it += 1
                if it == args.warmup:
                    step_timeout = (
                        max(5.0, 5 * last_warmup_time)
                        if last_warmup_time is not None
                        else None
                    )
                    logger.info(f"Running {args.iters} timed iterations")
                    ray.get([
                        actor.reset_peak_memory.remote()
                        for actor in actors.values()
                    ])
            else:
                start = time.perf_counter()
                piper_exec_dag(loss_fn, log_stats=True, step_timeout=step_timeout)
                end = time.perf_counter()
                iter_times.append(end - start)
                if args.iteration_sleep > 0:
                    time.sleep(args.iteration_sleep)
                it += 1
        except PiperResume as exc:
            # Promotion recovery replaced the failed peer; redo the
            # interrupted iteration in lockstep with the promoted standby.
            logger.info(f"resuming after promotion at iteration "
                        f"{exc.next_iter} (was at {it})")
            it = exc.next_iter

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
            piper_exec_dag(loss_fn, step_timeout=step_timeout)
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
    parser.add_argument('--model', choices=['9M', '1B', '9B', '48B', '30B-A3B', '30-A3B-half', '72B'], default='9M',
                        help='Model configuration: 9M, 1B, 9B, 48B, 30B-A3B, 30-A3B-half, or 72B (default: 9M)')
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=3)
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
            "Use PP as the outer placement dim (one pipeline stage per node, "
            "all DP replicas for that stage colocated). Makes per-stage EP/DP "
            "collectives intra-node at the cost of inter-node PP P2P. "
            "Default: false (one DP replica per node, PP inner)."
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
        default=3,
        help="Number of iterations to run under the PyTorch profiler.",
    )
    return parser.parse_args(argv)
