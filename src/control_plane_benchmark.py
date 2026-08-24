"""CPU-only microbenchmark support for Piper's actor DAG execution loop.

The benchmark intentionally drives :meth:`src.executors.DagExecutor.run`
instead of maintaining a second, simplified copy of the dispatch loop.  It
accepts a model-free ``TrainingDAG`` produced by the normal workload-lowering
path, hydrates the actor-local runtime fields, and dry-dispatches every node as
``ORDER_DUMMY``.  The DAG's real operation kinds, tags, metadata, dependencies,
and stream assignments remain available to execution-plan construction.  Host
stream and event objects replace CUDA at the edge of the loop.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import logging
import math
import os
import socket
import statistics
import time
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import torch

from .dag import TrainingDAG
from .executors import DagExecutor
from .ordering import _dispatch_order
from .runtime import BufferStore, EventStore
from .tasks import TaskType


MAX_BENCHMARK_ACTORS = 16


def validate_actor_count(actor_count: int) -> None:
    if actor_count <= 0:
        raise ValueError("actor_count must be positive")
    if actor_count > MAX_BENCHMARK_ACTORS:
        raise ValueError(
            f"actor_count={actor_count} exceeds the benchmark limit of "
            f"{MAX_BENCHMARK_ACTORS}"
        )


class _HostStream:
    """Minimal CUDA-stream stand-in used only by the CPU benchmark."""

    __slots__ = ("name", "record_count", "wait_count")

    def __init__(self, name: str) -> None:
        self.name = name
        self.record_count = 0
        self.wait_count = 0

    def wait_event(self, _event: Any) -> None:
        self.wait_count += 1


class _HostEvent:
    """Minimal CUDA-event stand-in for total-latency measurements."""

    __slots__ = ("recorded_stream",)

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.recorded_stream: _HostStream | None = None

    def record(self, stream: _HostStream) -> None:
        self.recorded_stream = stream
        stream.record_count += 1


def _recording_event_type(timestamps_ns: list[int]) -> type[_HostEvent]:
    class _RecordingHostEvent(_HostEvent):
        def record(self, stream: _HostStream) -> None:
            super().record(stream)
            timestamps_ns.append(time.perf_counter_ns())

    return _RecordingHostEvent


@contextmanager
def _replace_cuda_events(event_type: type[_HostEvent]) -> Iterator[None]:
    original_event = torch.cuda.Event
    torch.cuda.Event = event_type
    try:
        yield
    finally:
        torch.cuda.Event = original_event


class _NoOpParameterStorage:
    def drain_pending_frees(self) -> None:
        pass

    def clear_param_grads(self) -> None:
        pass

    def zero_grad_buffers(self, _stream: _HostStream) -> None:
        pass


class _HostRuntime:
    def __init__(self, stream_names: Sequence[str]) -> None:
        names = {"default_stream", *stream_names}
        self.streams = {name: _HostStream(name) for name in sorted(names)}
        self.pytorch_profiler_enabled = False

    def default_stream(self) -> _HostStream:
        return self.streams["default_stream"]

    def stream_for_task(self, node: Any) -> _HostStream:
        return self.streams[str(node.stream or "default_stream")]

    @staticmethod
    def execution_stream_for_task(
        _node: Any,
        logical_stream: _HostStream,
    ) -> _HostStream:
        return logical_stream

    @staticmethod
    def nvtx_push(_label: str) -> None:
        pass

    @staticmethod
    def nvtx_pop() -> None:
        pass


class ControlPlaneHarness:
    """One actor's model-free DAG and exact Piper dispatcher invocation."""

    def __init__(
        self,
        training_dag: TrainingDAG,
        *,
        actor_rank: int = 0,
        workload_metadata: dict[str, Any] | None = None,
    ) -> None:
        setup_start_ns = time.perf_counter_ns()
        self.dag = training_dag
        self.actor_rank = actor_rank
        self.workload_metadata = dict(workload_metadata or {})
        self._hydrate_for_dry_dispatch()
        self.sorted_nodes = [
            self.dag.nodes[uid]
            for uid in _dispatch_order(self.dag)
        ]
        stream_names = {
            str(node.stream or "default_stream")
            for node in self.dag.nodes.values()
        }
        self.runtime = _HostRuntime(sorted(stream_names))
        self.logger = logging.getLogger(
            f"piper.control_plane_benchmark.actor_{actor_rank}"
        )
        self.logger.setLevel(logging.CRITICAL + 1)
        self.executor = DagExecutor(
            runtime=self.runtime,
            stages=SimpleNamespace(buckets={}),
            buffers=BufferStore(),
            events=EventStore(),
            params=_NoOpParameterStorage(),
            communication=SimpleNamespace(reap_completed_p2p=lambda: None),
            compute=SimpleNamespace(),
            logger=self.logger,
        )
        prepare_start_ns = time.perf_counter_ns()
        self.prepared_execution = self.executor.prepare(
            self.dag,
            self.sorted_nodes,
            stream_override=self.runtime.execution_stream_for_task,
        )
        self.execution_prepare_ns = time.perf_counter_ns() - prepare_start_ns
        self.setup_ns = time.perf_counter_ns() - setup_start_ns
        self._iteration = 0

    def _hydrate_for_dry_dispatch(self) -> None:
        """Mirror actor DAG hydration while making every operation a no-op."""

        if not isinstance(self.dag, TrainingDAG):
            raise TypeError("ControlPlaneHarness requires a TrainingDAG")
        if not self.dag.nodes:
            raise ValueError("ControlPlaneHarness requires a non-empty TrainingDAG")

        for node in self.dag.nodes.values():
            meta = node.node_meta or {}
            if meta.get("gm") is not None or meta.get("gm_data") is not None:
                raise ValueError(
                    f"node {node.uid!r} contains model/FX graph state; "
                    "the control-plane benchmark requires a model-free TrainingDAG"
                )
            node.data_preds = []
            node.data_succs = []
            node.temporal_preds = []
            node.temporal_succs = []
            if node.node_kind in {"SEND_COMM", "RECV_COMM"}:
                node.peer_pp_rank = meta.get("peer_pp_rank")
            else:
                node.peer_pp_rank = None

            # DagExecutor still uses node_kind/compute_subkind/stream/node_meta
            # to build its real dependency plan.  Only task dispatch is replaced
            # by the no-op case so no model, collective, or CUDA work is invoked.
            node.task_type = TaskType.ORDER_DUMMY
            node.batches = [
                SimpleNamespace(
                    stage_id=node.tag.get("PP", self.actor_rank),
                    mb_idx=node.tag.get("MB", 0),
                )
            ]

        for edge in self.dag.edges:
            try:
                src = self.dag.nodes[edge.src_uid]
                dst = self.dag.nodes[edge.dst_uid]
            except KeyError as exc:
                raise ValueError(f"edge references unknown node: {edge}") from exc
            if edge.dep_kind == "data":
                src.data_succs.append(dst)
                dst.data_preds.append(src)
            elif edge.dep_kind == "temporal":
                src.temporal_succs.append(dst)
                dst.temporal_preds.append(src)
            elif edge.dep_kind == "issue":
                # Static dispatch order satisfies issue-only edges.  The
                # control-plane executor must not treat them as completion
                # dependencies.
                continue
            else:
                raise ValueError(
                    f"unsupported dependency kind {edge.dep_kind!r} on "
                    f"{edge.src_uid} -> {edge.dst_uid}"
                )

    def _run(self) -> None:
        iteration = self._iteration
        self._iteration += 1
        self.runtime.nvtx_push(f"iter_{iteration}_rank_{self.actor_rank}")
        try:
            self.executor.run(
                self.prepared_execution,
                inputs=None,
                labels=None,
                loss_buffer=[],
            )
        finally:
            self.runtime.nvtx_pop()

    def run_once_ns(self) -> int:
        """Return actor-internal latency for one exact dispatcher iteration."""

        with _replace_cuda_events(_HostEvent):
            start_ns = time.perf_counter_ns()
            self._run()
            return time.perf_counter_ns() - start_ns

    def warmup(self, iterations: int) -> None:
        if iterations < 0:
            raise ValueError("warmup iterations cannot be negative")
        for _ in range(iterations):
            self.run_once_ns()

    def issue_gap_summary_us(self, iterations: int = 3) -> dict[str, float]:
        """Measure host issue gaps between consecutive DAG nodes.

        Timestamps are captured by the fake completion-event ``record`` call,
        which is the last common operation in Piper's per-node dispatch path.
        The prologue event is discarded.  The timer instrumentation is kept out
        of the primary total-latency measurements.
        """

        if iterations <= 0:
            raise ValueError("issue-gap iterations must be positive")
        gaps_ns: list[int] = []
        for _ in range(iterations):
            timestamps_ns: list[int] = []
            with _replace_cuda_events(_recording_event_type(timestamps_ns)):
                self._run()
            expected = len(self.dag.nodes) + 1  # prologue plus one per node
            if len(timestamps_ns) != expected:
                raise RuntimeError(
                    f"expected {expected} event records, saw {len(timestamps_ns)}"
                )
            node_timestamps = timestamps_ns[1:]
            gaps_ns.extend(
                later - earlier
                for earlier, later in zip(node_timestamps, node_timestamps[1:])
            )
        return summarize_ns(gaps_ns)

    def metadata(self) -> dict[str, Any]:
        return {
            "actor_rank": self.actor_rank,
            "workload_metadata": dict(self.workload_metadata),
            "node_count": len(self.dag.nodes),
            "edge_count": len(self.dag.edges),
            "setup_us": self.setup_ns / 1000.0,
            "execution_prepare_us": self.execution_prepare_ns / 1000.0,
            "contains_model_state": False,
            "contains_fx_graphs": False,
            "uses_cuda": False,
        }


class ControlPlaneBenchmarkActor:
    """Ray-compatible wrapper; the script applies ``ray.remote`` dynamically."""

    def __init__(self, actor_rank: int) -> None:
        self.actor_rank = actor_rank
        self.harness: ControlPlaneHarness | None = None

    def configure(
        self,
        training_dag: TrainingDAG,
        workload_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.harness = ControlPlaneHarness(
            training_dag,
            actor_rank=self.actor_rank,
            workload_metadata=workload_metadata,
        )
        return {
            **self.harness.metadata(),
            "worker": self.worker_info(),
        }

    def worker_info(self) -> dict[str, Any]:
        import ray

        return {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "ray_node_id": ray.get_runtime_context().get_node_id(),
        }

    def _require_harness(self) -> ControlPlaneHarness:
        if self.harness is None:
            raise RuntimeError("configure() must be called before running a case")
        return self.harness

    def warmup(self, iterations: int) -> None:
        self._require_harness().warmup(iterations)

    def run_once_ns(self) -> int:
        return self._require_harness().run_once_ns()

    def run_once_profiled(self) -> dict[str, Any]:
        before = [entry["collections"] for entry in gc.get_stats()]
        elapsed_ns = self._require_harness().run_once_ns()
        after = [entry["collections"] for entry in gc.get_stats()]
        return {
            "elapsed_ns": elapsed_ns,
            "gc_collections": [
                later - earlier for earlier, later in zip(before, after)
            ],
        }

    def issue_gap_summary_us(self, iterations: int) -> dict[str, float]:
        return self._require_harness().issue_gap_summary_us(iterations)


def percentile(values: Sequence[int | float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_ns(samples_ns: Sequence[int | float]) -> dict[str, float]:
    """Return common latency statistics converted from nanoseconds to us."""

    if not samples_ns:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "stdev": 0.0,
            "variance": 0.0,
            "standard_error": 0.0,
            "coefficient_of_variation": 0.0,
            "mean_ci95_low": 0.0,
            "mean_ci95_high": 0.0,
        }
    values = [float(value) / 1000.0 for value in samples_ns]
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    standard_error = stdev / math.sqrt(len(values))
    ci_half_width = 1.96 * standard_error
    return {
        "count": len(values),
        "min": min(values),
        "mean": mean,
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "stdev": stdev,
        "variance": variance,
        "standard_error": standard_error,
        "coefficient_of_variation": stdev / mean if mean else 0.0,
        "mean_ci95_low": mean - ci_half_width,
        "mean_ci95_high": mean + ci_half_width,
    }


def linear_fit(points: Sequence[tuple[int, float]]) -> dict[str, float]:
    """Ordinary least-squares fit of latency_us = intercept + slope * nodes."""

    if len(points) < 2:
        raise ValueError("linear_fit requires at least two points")
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0.0:
        raise ValueError("linear_fit x values must not all be equal")
    slope = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, ys)
    ) / denominator
    intercept = y_mean - slope * x_mean
    predicted = [intercept + slope * x for x in xs]
    total_variance = sum((y - y_mean) ** 2 for y in ys)
    residual_variance = sum(
        (y - y_hat) ** 2 for y, y_hat in zip(ys, predicted)
    )
    r_squared = (
        1.0 - residual_variance / total_variance
        if total_variance > 0.0
        else 1.0
    )
    return {
        "intercept_us": intercept,
        "slope_us_per_node": slope,
        "r_squared": r_squared,
    }


__all__ = [
    "ControlPlaneBenchmarkActor",
    "ControlPlaneHarness",
    "MAX_BENCHMARK_ACTORS",
    "linear_fit",
    "percentile",
    "summarize_ns",
    "validate_actor_count",
]
