#!/usr/bin/env python3
"""Benchmark Piper control-plane scaling with real Qwen3.5-397B DAGs.

The model is traced on ``meta`` and lowered through Piper's actual DualPipeV
schedule path.  The matrix varies PP/DP/EP degrees, ZeRO-3 bucket sizes, and
whether DP/EP collectives share a stream.  FX graphs and model/tensor state are
then stripped before Piper's real ``DagExecutor.run`` is dry-dispatched on
CPU-only Ray actors.  The suite never reserves or initializes a GPU.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import platform
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence


# DAG generation traces on ``meta`` and timing uses host-only stand-ins.  Hide
# accelerators before importing Piper/Torch so Dynamo does not initialize a
# CUDA stream context merely to snapshot compiler state.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.control_plane_benchmark import (  # noqa: E402
    ControlPlaneBenchmarkActor,
    MAX_BENCHMARK_ACTORS,
    linear_fit,
    summarize_ns,
    validate_actor_count,
)
from src.qwen_control_plane_workloads import (  # noqa: E402
    COLLECTIVE_STREAM_POLICIES,
    DEFAULT_QWEN_MODEL,
    QwenControlPlaneWorkload,
    QwenDualPipeWorkloadSpec,
    QwenTrainingDAGFactory,
    load_qwen_control_plane_workload,
    save_qwen_control_plane_workload,
)


DEFAULT_PARALLEL_CONFIGS = (
    "1x8x8,1x16x16,1x64x64,"
    "4x8x8,4x16x16,4x64x64,"
    "16x8x8,16x16x16,16x64x64"
)


def _comma_separated_ints(raw: str, *, name: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{name} must be a comma-separated list of integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} cannot be empty")
    return values


def _parallel_configs(raw: str) -> list[tuple[int, int, int]]:
    configs: list[tuple[int, int, int]] = []
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        fields = entry.lower().replace(":", "x").split("x")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError(
                "parallel-configs entries must have the form PPxDPxEP"
            )
        try:
            config = tuple(int(field) for field in fields)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "parallel-configs entries must contain integers"
            ) from exc
        configs.append(config)  # type: ignore[arg-type]
    if not configs:
        raise argparse.ArgumentTypeError("parallel-configs cannot be empty")
    return configs


def _bucket_sizes(raw: str) -> list[int | None]:
    buckets: list[int | None] = []
    for field in (part.strip().lower() for part in raw.split(",")):
        if not field:
            continue
        if field in {"none", "off", "unbucketed"}:
            buckets.append(None)
        else:
            try:
                buckets.append(int(field))
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    "bucket-sizes must contain 'none' or integer MiB values"
                ) from exc
    if not buckets:
        raise argparse.ArgumentTypeError("bucket-sizes cannot be empty")
    return buckets


def _stream_policies(raw: str) -> list[str]:
    policies = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not policies:
        raise argparse.ArgumentTypeError("stream-policies cannot be empty")
    unknown = sorted(set(policies) - set(COLLECTIVE_STREAM_POLICIES))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown stream policies {unknown}; choose from "
            f"{COLLECTIVE_STREAM_POLICIES}"
        )
    return policies


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("all", "generate", "dag-scaling", "actor-scaling"),
        default="all",
        help="Generate structural data, time the workload matrix, scale actors, or all.",
    )
    parser.add_argument("--model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32768)
    parser.add_argument(
        "--parallel-configs",
        default=DEFAULT_PARALLEL_CONFIGS,
        help="Comma-separated PPxDPxEP configurations.",
    )
    parser.add_argument(
        "--bucket-sizes",
        default="none,256,64,16",
        help="Comma-separated ZeRO bucket sizes in MiB, including 'none'.",
    )
    parser.add_argument(
        "--stream-policies",
        default="separate,shared",
        help=(
            "First policy is used for the full matrix; remaining policies are "
            "only used for the configured stream-comparison case."
        ),
    )
    parser.add_argument(
        "--stream-comparison-config",
        default="16x64x64",
        help="PPxDPxEP configuration used for the alternate-stream comparison.",
    )
    parser.add_argument(
        "--stream-comparison-bucket",
        default="16",
        help="Bucket MiB or 'none' used for the alternate-stream comparison.",
    )
    parser.add_argument(
        "--actor-scale-config",
        default="16x64x64",
        help="PPxDPxEP schedule used for the actor-count sweep.",
    )
    parser.add_argument(
        "--actor-scale-bucket",
        default="16",
        help="Bucket MiB or 'none' used for the actor-count sweep.",
    )
    parser.add_argument(
        "--actor-scale-stream-policy",
        choices=COLLECTIVE_STREAM_POLICIES,
        default="separate",
    )
    parser.add_argument(
        "--actor-counts",
        default="1,2,4,8,16",
        help="Concurrent CPU-only Ray actor counts.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--issue-gap-iterations", type=int, default=10)
    parser.add_argument("--cpus-per-actor", type=float, default=1.0)
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray address such as 'auto'; omitted starts a local CPU-only runtime.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/control_plane_scalability"),
    )
    parser.add_argument(
        "--dag-cache-dir",
        type=Path,
        default=Path("out/control_plane_scalability/dag_cache"),
        help="Persistent cache of stripped physical-PP-rank-0 TrainingDAGs.",
    )
    parser.add_argument(
        "--rebuild-dag-cache",
        action="store_true",
        help="Rebuild cached DAGs instead of reusing validated persistent files.",
    )
    args = parser.parse_args(argv)
    args.parallel_configs = _parallel_configs(args.parallel_configs)
    args.bucket_sizes = _bucket_sizes(args.bucket_sizes)
    args.stream_policies = _stream_policies(args.stream_policies)
    stream_comparison_configs = _parallel_configs(args.stream_comparison_config)
    if len(stream_comparison_configs) != 1:
        raise ValueError(
            "stream-comparison-config must specify exactly one PPxDPxEP tuple"
        )
    args.stream_comparison_config = stream_comparison_configs[0]
    stream_comparison_buckets = _bucket_sizes(args.stream_comparison_bucket)
    if len(stream_comparison_buckets) != 1:
        raise ValueError("stream-comparison-bucket must specify exactly one value")
    args.stream_comparison_bucket = stream_comparison_buckets[0]
    actor_scale_configs = _parallel_configs(args.actor_scale_config)
    if len(actor_scale_configs) != 1:
        raise ValueError("actor-scale-config must specify exactly one PPxDPxEP tuple")
    args.actor_scale_config = actor_scale_configs[0]
    actor_scale_buckets = _bucket_sizes(args.actor_scale_bucket)
    if len(actor_scale_buckets) != 1:
        raise ValueError("actor-scale-bucket must specify exactly one value")
    args.actor_scale_bucket = actor_scale_buckets[0]
    args.actor_counts = _comma_separated_ints(
        args.actor_counts, name="actor-counts"
    )
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.sequence_length <= 0:
        raise ValueError("batch-size and sequence-length must be positive")
    for count in args.actor_counts:
        validate_actor_count(count)
    if args.iterations <= 0 or args.issue_gap_iterations <= 0:
        raise ValueError("iterations and issue-gap-iterations must be positive")
    if args.warmups < 0:
        raise ValueError("warmups cannot be negative")
    if args.cpus_per_actor <= 0:
        raise ValueError("cpus-per-actor must be positive")
    for bucket in args.bucket_sizes:
        if bucket is not None and bucket <= 0:
            raise ValueError("bucket sizes must be positive or none")
    for pp, dp, ep in [
        *args.parallel_configs,
        args.stream_comparison_config,
        args.actor_scale_config,
    ]:
        QwenDualPipeWorkloadSpec(
            pp_degree=pp,
            dp_degree=dp,
            ep_degree=ep,
            bucket_size_mb=None,
            collective_stream_policy="shared",
            model_name=args.model,
        ).validate()


def workload_specs(args: argparse.Namespace) -> list[QwenDualPipeWorkloadSpec]:
    baseline_policy = args.stream_policies[0]
    specs = [
        QwenDualPipeWorkloadSpec(
            pp_degree=pp,
            dp_degree=dp,
            ep_degree=ep,
            bucket_size_mb=bucket,
            collective_stream_policy=baseline_policy,
            model_name=args.model,
        )
        for pp, dp, ep in args.parallel_configs
        for bucket in args.bucket_sizes
    ]
    pp, dp, ep = args.stream_comparison_config
    for policy in args.stream_policies[1:]:
        comparison = QwenDualPipeWorkloadSpec(
            pp_degree=pp,
            dp_degree=dp,
            ep_degree=ep,
            bucket_size_mb=args.stream_comparison_bucket,
            collective_stream_policy=policy,
            model_name=args.model,
        )
        if comparison not in specs:
            specs.append(comparison)
    return specs


def actor_scale_spec(args: argparse.Namespace) -> QwenDualPipeWorkloadSpec:
    pp, dp, ep = args.actor_scale_config
    return QwenDualPipeWorkloadSpec(
        pp_degree=pp,
        dp_degree=dp,
        ep_degree=ep,
        bucket_size_mb=args.actor_scale_bucket,
        collective_stream_policy=args.actor_scale_stream_policy,
        model_name=args.model,
    )


def _create_workers(
    ray: Any,
    remote_type: Any,
    count: int,
    cpus_per_actor: float,
) -> list[Any]:
    validate_actor_count(count)
    return [
        remote_type.options(
            num_cpus=cpus_per_actor,
            num_gpus=0,
            scheduling_strategy="SPREAD",
        ).remote(rank)
        for rank in range(count)
    ]


def _kill_workers(ray: Any, workers: Iterable[Any]) -> None:
    for worker in workers:
        ray.kill(worker, no_restart=True)


def _configure_workers(
    ray: Any,
    workers: Sequence[Any],
    workload: QwenControlPlaneWorkload,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    pp_dag_count = len(workload.per_actor_dags)
    dag_indices = [index % pp_dag_count for index in range(len(workers))]
    refs = []
    for worker, dag_index in zip(workers, dag_indices):
        compact_metadata = {
            "case_name": workload.spec.case_name,
            "world_size": workload.spec.world_size,
            "pp_dag_index": dag_index,
            "spec": asdict(workload.spec),
        }
        refs.append(
            worker.configure.remote(
                workload.per_actor_dags[dag_index],
                compact_metadata,
            )
        )
    metadata = ray.get(refs)
    return metadata, [entry["node_count"] for entry in metadata], dag_indices


def _measure_steps(
    ray: Any,
    workers: Sequence[Any],
    node_counts: Sequence[int],
    *,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    ray.get([worker.warmup.remote(warmups) for worker in workers])
    driver_step_ns: list[int] = []
    actor_critical_ns: list[int] = []
    actor_mean_ns: list[float] = []
    actor_iteration_per_node_ns: list[float] = []
    driver_iteration_per_node_ns: list[float] = []
    all_actor_samples_ns: list[int] = []
    all_actor_gc_collections: list[list[int]] = []
    orchestration_ns: list[int] = []
    max_actor_nodes = max(node_counts)

    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        actor_results = ray.get(
            [worker.run_once_profiled.remote() for worker in workers]
        )
        actor_samples = [entry["elapsed_ns"] for entry in actor_results]
        elapsed_ns = time.perf_counter_ns() - start_ns
        critical_ns = max(actor_samples)
        driver_step_ns.append(elapsed_ns)
        actor_critical_ns.append(critical_ns)
        actor_mean_ns.append(sum(actor_samples) / len(actor_samples))
        actor_iteration_per_node_ns.append(
            max(
                sample / node_count
                for sample, node_count in zip(actor_samples, node_counts)
            )
        )
        driver_iteration_per_node_ns.append(elapsed_ns / max_actor_nodes)
        all_actor_samples_ns.extend(actor_samples)
        all_actor_gc_collections.extend(
            entry["gc_collections"] for entry in actor_results
        )
        orchestration_ns.append(max(0, elapsed_ns - critical_ns))

    samples_with_gc_ns = [
        sample
        for sample, collections in zip(
            all_actor_samples_ns, all_actor_gc_collections
        )
        if any(collections)
    ]
    samples_without_gc_ns = [
        sample
        for sample, collections in zip(
            all_actor_samples_ns, all_actor_gc_collections
        )
        if not any(collections)
    ]
    return {
        "driver_iteration_us": summarize_ns(driver_step_ns),
        "executor_iteration_critical_us": summarize_ns(actor_critical_ns),
        "executor_iteration_mean_across_actors_us": summarize_ns(actor_mean_ns),
        "executor_iteration_all_samples_us": summarize_ns(all_actor_samples_ns),
        "ray_round_trip_overhead_us": summarize_ns(orchestration_ns),
        "executor_iteration_per_node_us": summarize_ns(
            actor_iteration_per_node_ns
        ),
        "driver_iteration_per_node_us": summarize_ns(
            driver_iteration_per_node_ns
        ),
        "executor_iteration_samples_us": [
            sample / 1000.0 for sample in all_actor_samples_ns
        ],
        "gc": {
            "collections_by_generation": [
                sum(collections[generation] for collections in all_actor_gc_collections)
                for generation in range(3)
            ],
            "iterations_with_collection": len(samples_with_gc_ns),
            "iterations_with_generation_2_collection": sum(
                bool(collections[2])
                for collections in all_actor_gc_collections
            ),
            "fraction_with_collection": (
                len(samples_with_gc_ns) / len(all_actor_samples_ns)
                if all_actor_samples_ns
                else 0.0
            ),
            "collections_per_iteration": all_actor_gc_collections,
            "executor_iteration_with_gc_us": summarize_ns(samples_with_gc_ns),
            "executor_iteration_without_gc_us": summarize_ns(samples_without_gc_ns),
        },
    }


def _measure_issue_gaps(
    ray: Any,
    workers: Sequence[Any],
    *,
    issue_gap_iterations: int,
) -> dict[str, Any]:
    issue_by_actor = ray.get(
        [
            worker.issue_gap_summary_us.remote(issue_gap_iterations)
            for worker in workers
        ]
    )
    return {
        "steady_node_issue_gap_us": {
            "actor_p50_max": max(entry["median"] for entry in issue_by_actor),
            "actor_p95_max": max(entry["p95"] for entry in issue_by_actor),
            "actor_max": max(entry["max"] for entry in issue_by_actor),
            "per_actor": issue_by_actor,
        },
    }


def _placement_summary(metadata: Sequence[dict[str, Any]]) -> dict[str, Any]:
    placements = [entry["worker"] for entry in metadata]
    return {
        "ray_node_count": len({entry["ray_node_id"] for entry in placements}),
        "workers": placements,
    }


def benchmark_workload(
    ray: Any,
    workers: Sequence[Any],
    workload: QwenControlPlaneWorkload,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata, node_counts, dag_indices = _configure_workers(
        ray, workers, workload
    )
    measurements = _measure_steps(
        ray,
        workers,
        node_counts,
        warmups=args.warmups,
        iterations=args.iterations,
    )
    issue_gaps = _measure_issue_gaps(
        ray,
        workers,
        issue_gap_iterations=args.issue_gap_iterations,
    )
    row = {
        "case_name": workload.spec.case_name,
        "spec": asdict(workload.spec),
        "declared_world_size": workload.spec.world_size,
        "benchmarked_actor_count": len(workers),
        "selected_physical_pp_rank": 0,
        "selected_logical_pp_stages": [0, 2 * workload.spec.pp_degree - 1],
        "benchmarked_pp_dag_indices": dag_indices,
        "selected_dag_nodes": node_counts[0],
        "selected_dag_edges": len(workload.per_actor_dags[0].edges),
        "selected_dag_pickle_bytes": len(
            pickle.dumps(workload.per_actor_dags[0], protocol=pickle.HIGHEST_PROTOCOL)
        ),
        "all_pp_dag_node_counts": [
            len(dag.nodes) for dag in workload.per_actor_dags
        ],
        "all_pp_dag_edge_counts": [
            len(dag.edges) for dag in workload.per_actor_dags
        ],
        "worker_setup_us": summarize_ns(
            [entry["setup_us"] * 1000.0 for entry in metadata]
        ),
        "execution_prepare_us": summarize_ns(
            [entry["execution_prepare_us"] * 1000.0 for entry in metadata]
        ),
        "placement": _placement_summary(metadata),
        **measurements,
        **issue_gaps,
    }
    if any(dag_index != 0 for dag_index in dag_indices):
        raise RuntimeError(
            f"DAG-scaling benchmark must select only PP rank 0, got {dag_indices}"
        )
    return row


def run_dag_scaling(
    ray: Any,
    remote_type: Any,
    specs: Sequence[QwenDualPipeWorkloadSpec],
    build_workload: Callable[[QwenDualPipeWorkloadSpec], QwenControlPlaneWorkload],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workers = _create_workers(ray, remote_type, 1, args.cpus_per_actor)
    rows: list[dict[str, Any]] = []
    workloads: list[dict[str, Any]] = []
    try:
        for index, spec in enumerate(specs, start=1):
            print(
                f"[matrix {index}/{len(specs)}] {spec.case_name}",
                flush=True,
            )
            workload = build_workload(spec)
            workloads.append(workload.metadata)
            rows.append(
                benchmark_workload(
                    ray,
                    workers,
                    workload,
                    args,
                )
            )
    finally:
        _kill_workers(ray, workers)
    return rows, workloads


def run_actor_scaling(
    ray: Any,
    remote_type: Any,
    workload: QwenControlPlaneWorkload,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, actor_count in enumerate(args.actor_counts, start=1):
        print(
            f"[actors {index}/{len(args.actor_counts)}] actors={actor_count} "
            f"case={workload.spec.case_name}",
            flush=True,
        )
        workers = _create_workers(
            ray, remote_type, actor_count, args.cpus_per_actor
        )
        try:
            row = benchmark_workload(ray, workers, workload, args)
            row["actor_count"] = actor_count
            rows.append(row)
        finally:
            _kill_workers(ray, workers)

    if rows:
        baseline = next(
            (row for row in rows if row["actor_count"] == 1), rows[0]
        )
        baseline_actor = baseline["executor_iteration_critical_us"]["median"]
        baseline_driver = baseline["driver_iteration_us"]["median"]
        for row in rows:
            row["slowdown_vs_baseline"] = {
                "actor_critical": row["executor_iteration_critical_us"]["median"]
                / baseline_actor,
                "driver_step": row["driver_iteration_us"]["median"]
                / baseline_driver,
            }
    return rows


def dag_scaling_fits(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fits: list[dict[str, Any]] = []
    for policy in COLLECTIVE_STREAM_POLICIES:
        group = [
            row
            for row in rows
            if row["spec"]["collective_stream_policy"] == policy
        ]
        unique_x = {row["selected_dag_nodes"] for row in group}
        if len(unique_x) < 2:
            continue
        fits.append(
            {
                "collective_stream_policy": policy,
                "actor_loop": linear_fit(
                    [
                        (
                            row["selected_dag_nodes"],
                            row["executor_iteration_critical_us"]["median"],
                        )
                        for row in group
                    ]
                ),
            }
        )
    return fits


def _host_environment() -> dict[str, Any]:
    import ray
    import torch

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ray": ray.__version__,
        "logical_cpu_count": os.cpu_count(),
        "benchmark_uses_cuda": False,
    }


def _ray_environment(ray: Any) -> dict[str, Any]:
    nodes = []
    for node in ray.nodes():
        if node.get("Alive"):
            resources = node.get("Resources", {})
            nodes.append(
                {
                    "node_manager_address": node.get("NodeManagerAddress"),
                    "cpu_resources": resources.get("CPU", 0),
                    "gpu_resources_reserved_by_benchmark": 0,
                }
            )
    return {"ray_nodes": nodes}


def _git_metadata() -> dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=REPO_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    return {
        "branch": git("branch", "--show-current"),
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _flatten_row(row: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in row.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_row(value, full_key))
        elif isinstance(value, (list, tuple)):
            flattened[full_key] = json.dumps(value, sort_keys=True)
        else:
            flattened[full_key] = value
    return flattened


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    flattened = [_flatten_row(row) for row in rows]
    fields = sorted({key for row in flattened for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def _bucket_label(bucket: int | None) -> str:
    return "none" if bucket is None else str(bucket)


def _format_report(result: dict[str, Any]) -> str:
    lines = [
        "# Piper Qwen3.5-397B control-plane scalability",
        "",
        "The model is traced on `meta` and lowered with Piper's real DualPipeV, "
        "PP P2P, ZeRO-3, EP A2A, bucketing, and stream-ordering passes. FX/model "
        "payloads are stripped before the real actor DAG loop is dry-dispatched "
        "with host-only CUDA stream/event stand-ins.",
        "",
        "The 397B preset uses all 60 layers from Qwen's published checkpoint "
        "configuration; see the recorded model configuration in `results.json`.",
        "",
    ]
    workloads = result.get("workloads", [])
    if workloads:
        lines.extend(
            [
                "## Generated workload DAGs",
                "",
                "| PP | DP | EP | Bucket MiB | Streams | World | MBs | "
                "PP0 nodes | PP0 edges | Build (s) | Trace (s) | Lowering (s) | "
                "Serialize (s) |",
                "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for workload in workloads:
            spec = workload["spec"]
            dag = workload["dag_summary"]
            lines.append(
                f"| {spec['pp_degree']} | {spec['dp_degree']} | "
                f"{spec['ep_degree']} | {_bucket_label(spec['bucket_size_mb'])} | "
                f"{spec['collective_stream_policy']} | {workload['world_size']} | "
                f"{workload['microbatch_count']} | "
                f"{dag['pp_rank_0_nodes']} | {dag['pp_rank_0_edges']} | "
                f"{workload['dag_build_time_s']:.3f} | "
                f"{workload['trace_time_s']:.3f} | "
                f"{workload['lowering_time_s']:.3f} | "
                f"{workload['dag_cache']['serialization_time_s']:.3f} |"
            )
        lines.append("")

    dag_rows = result.get("dag_scaling", [])
    if dag_rows:
        lines.extend(
            [
                "## DAG execution-loop scaling",
                "",
                "| PP/DP/EP | Bucket | Streams | PP rank | Nodes | Edges | "
                "Iteration mean ± stdev (us) | Median (us) | p95 (us) | "
                "CV | Gen2 GC iters | Prepare once (us) | us/node |",
                "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in dag_rows:
            spec = row["spec"]
            gen2_gc_iterations = row["gc"].get(
                "iterations_with_generation_2_collection",
                sum(
                    bool(collections[2])
                    for collections in row["gc"]["collections_per_iteration"]
                ),
            )
            lines.append(
                f"| {spec['pp_degree']}/{spec['dp_degree']}/{spec['ep_degree']} | "
                f"{_bucket_label(spec['bucket_size_mb'])} | "
                f"{spec['collective_stream_policy']} | "
                f"{row['selected_physical_pp_rank']} | "
                f"{row['selected_dag_nodes']} | {row['selected_dag_edges']} | "
                f"{row['executor_iteration_critical_us']['mean']:.3f} ± "
                f"{row['executor_iteration_critical_us']['stdev']:.3f} | "
                f"{row['executor_iteration_critical_us']['median']:.3f} | "
                f"{row['executor_iteration_critical_us']['p95']:.3f} | "
                f"{row['executor_iteration_critical_us']['coefficient_of_variation']:.3f} | "
                f"{gen2_gc_iterations} | "
                f"{row['execution_prepare_us']['median']:.3f} | "
                f"{row['executor_iteration_per_node_us']['median']:.4f} |"
            )
        lines.extend(["", "Linear fits across the realistic matrix:", ""])
        for fit in result.get("dag_scaling_fits", []):
            loop = fit["actor_loop"]
            lines.append(
                f"- {fit['collective_stream_policy']}: actor loop slope "
                f"{loop['slope_us_per_node']:.4f} us/node "
                f"(R²={loop['r_squared']:.4f})."
            )
        lines.append("")

    actor_rows = result.get("actor_scaling", [])
    if actor_rows:
        lines.extend(
            [
                "## Actor-count scaling",
                "",
                "| Actors | Ray nodes | Actor critical median (us) | "
                "Driver median (us) | Ray fan-out/fan-in (us) | "
                "Driver slowdown | End-to-end us/node |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in actor_rows:
            lines.append(
                f"| {row['actor_count']} | {row['placement']['ray_node_count']} | "
                f"{row['executor_iteration_critical_us']['median']:.3f} | "
                f"{row['driver_iteration_us']['median']:.3f} | "
                f"{row['ray_round_trip_overhead_us']['median']:.3f} | "
                f"{row['slowdown_vs_baseline']['driver_step']:.3f}x | "
                f"{row['driver_iteration_per_node_us']['median']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Executor iteration time is the CPU overhead of one real "
            "`DagExecutor.run` traversal of a prepared physical-PP-rank-0 DAG; "
            "plan and lookup-table construction are excluded.",
            "- Prepare-once time covers dependency-plan construction plus the "
            "UID-to-operation and UID-to-physical-stream lookup tables built "
            "during actor setup.",
            "- Completion-event timestamps provide the steady intra-loop issue gap; "
            "total us/node additionally amortizes per-iteration setup and teardown.",
            "- Raw iteration samples and per-generation Python GC counters are "
            "retained. Generation-2 collections explain the long-tail iterations "
            "and are included in the reported mean and variance.",
            "- Host stand-ins retain Python event/stream calls but omit CUDA-driver "
            "cost, so measured executor overhead is a lower bound for a CUDA run.",
            "- EP=1 means replicated MoE; EP=DP uses Piper's existing DP/EP co-group. "
            "The current schedule IR does not express an independent Cartesian EP group.",
            "- DAG scaling times exactly one cached physical PP-rank-0 DAG per case "
            "with one CPU-only Ray actor and reserves zero GPUs.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "workloads.csv", result.get("workloads", []))
    _write_csv(output_dir / "dag_scaling.csv", result.get("dag_scaling", []))
    _write_csv(
        output_dir / "actor_scaling.csv", result.get("actor_scaling", [])
    )
    (output_dir / "report.md").write_text(
        _format_report(result), encoding="utf-8"
    )


def _init_ray(args: argparse.Namespace) -> Any:
    import ray

    kwargs: dict[str, Any] = {
        "namespace": "piper_control_plane_scalability",
        "log_to_driver": False,
    }
    if args.ray_address:
        kwargs["address"] = args.ray_address
    else:
        kwargs["include_dashboard"] = False
    ray.init(**kwargs)
    return ray


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cache: dict[QwenDualPipeWorkloadSpec, QwenControlPlaneWorkload] = {}
    factory = (
        QwenTrainingDAGFactory(
            model_name=args.model,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
        )
        if args.mode in {"all", "generate"}
        else None
    )

    def generate(spec: QwenDualPipeWorkloadSpec) -> QwenControlPlaneWorkload:
        if factory is None:
            raise RuntimeError("DAG generation is disabled in timing-only modes")
        if spec not in cache:
            if not args.rebuild_dag_cache:
                try:
                    cache[spec] = load_qwen_control_plane_workload(
                        spec,
                        args.dag_cache_dir,
                    )
                except FileNotFoundError:
                    pass
            if spec not in cache:
                cache[spec] = factory.build(spec)
                save_qwen_control_plane_workload(cache[spec], args.dag_cache_dir)
        return cache[spec]

    def load(spec: QwenDualPipeWorkloadSpec) -> QwenControlPlaneWorkload:
        if spec not in cache:
            cache[spec] = load_qwen_control_plane_workload(
                spec,
                args.dag_cache_dir,
            )
        return cache[spec]

    result: dict[str, Any] = {
        "schema_version": 4,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": _git_metadata(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "methodology": {
            "model": args.model,
            "schedule": "DualPipeV",
            "trace_device": "meta",
            "executor": "src.executors.DagExecutor.run",
            "execution_plan_prepared_once_per_loaded_dag": True,
            "stream_lookup_prepared_once_per_execution_policy": True,
            "model_payload_stripped_before_ray": True,
            "runtime_dispatch_task_type": "ORDER_DUMMY",
            "structural_node_kinds_preserved": True,
            "cuda": False,
            "gpu_resources_per_actor": 0,
            "max_actor_count": MAX_BENCHMARK_ACTORS,
            "selected_physical_pp_rank": 0,
            "logical_stages_per_physical_rank": 2,
            "timing_modes_require_persisted_dag": True,
        },
        "environment": _host_environment(),
        "workloads": [],
        "dag_scaling": [],
        "actor_scaling": [],
    }

    matrix_specs = workload_specs(args)
    ray = None
    try:
        if args.mode in {"all", "generate"}:
            for index, spec in enumerate(matrix_specs, start=1):
                print(
                    f"[generate {index}/{len(matrix_specs)}] {spec.case_name}",
                    flush=True,
                )
                result["workloads"].append(generate(spec).metadata)
                write_results(args.output_dir.resolve(), result)
        if args.mode != "generate":
            ray = _init_ray(args)
            max_requested = (
                max(args.actor_counts)
                if args.mode in {"all", "actor-scaling"}
                else 1
            )
            required_cpus = max_requested * args.cpus_per_actor
            cluster_cpus = float(ray.cluster_resources().get("CPU", 0.0))
            if required_cpus > cluster_cpus:
                raise RuntimeError(
                    f"largest case requests {required_cpus:g} Ray CPUs, cluster "
                    f"exposes {cluster_cpus:g}; reduce actor counts or cpus-per-actor"
                )
            result["environment"].update(_ray_environment(ray))
            remote_type = ray.remote(ControlPlaneBenchmarkActor)
            if args.mode in {"all", "dag-scaling"}:
                rows, workloads = run_dag_scaling(
                    ray,
                    remote_type,
                    matrix_specs,
                    load,
                    args,
                )
                result["dag_scaling"] = rows
                result["workloads"] = workloads
                result["dag_scaling_fits"] = dag_scaling_fits(rows)
            if args.mode in {"all", "actor-scaling"}:
                scale_workload = load(actor_scale_spec(args))
                if not any(
                    entry["case_name"] == scale_workload.spec.case_name
                    for entry in result["workloads"]
                ):
                    result["workloads"].append(scale_workload.metadata)
                result["actor_scaling"] = run_actor_scaling(
                    ray,
                    remote_type,
                    scale_workload,
                    args,
                )
        write_results(args.output_dir.resolve(), result)
    finally:
        if ray is not None:
            ray.shutdown()

    print(f"Wrote {args.output_dir.resolve() / 'results.json'}", flush=True)
    print(f"Wrote {args.output_dir.resolve() / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
