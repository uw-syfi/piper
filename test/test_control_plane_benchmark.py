from __future__ import annotations

import pytest
import torch

from src.control_plane_benchmark import (
    ControlPlaneHarness,
    MAX_BENCHMARK_ACTORS,
    linear_fit,
    summarize_ns,
    validate_actor_count,
)
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.tasks import TaskType


def _tiny_model_free_dag() -> TrainingDAG:
    """Small hand-authored fixture; benchmark DAG generation lives elsewhere."""

    dag = TrainingDAG()
    nodes = [
        TrainingDAGNode(
            uid="fwd",
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag={"PASS": "F", "MB": 2, "PP": 4},
            device=[4],
            stream="default_stream",
            node_meta={"bucket_key": "layer.0"},
        ),
        TrainingDAGNode(
            uid="ep_a2a",
            node_kind="A2A_COMM",
            compute_subkind=None,
            tag={"PASS": "F", "MB": 2, "PP": 4},
            device=[4],
            stream="ep_stream",
            node_meta={"comm_group_kind": "ep"},
        ),
        TrainingDAGNode(
            uid="bwd",
            node_kind="COMPUTE",
            compute_subkind="BWD",
            tag={"PASS": "B", "MB": 2, "PP": 4},
            device=[4],
            stream="default_stream",
            node_meta={"bucket_key": "layer.0", "fwd_uid": "fwd"},
        ),
        TrainingDAGNode(
            uid="dp_reduce",
            node_kind="REDUCE_COMM",
            compute_subkind=None,
            tag={"PASS": "B", "MB": 2, "PP": 4},
            device=[4],
            stream="dp_stream",
            node_meta={"bucket_key": "layer.0"},
        ),
    ]
    for node in nodes:
        dag.add_node(node)
    for src, dst in zip(nodes, nodes[1:]):
        dag.add_edge(
            TrainingDAGEdge(
                src_uid=src.uid,
                dst_uid=dst.uid,
                dep_kind="data",
            )
        )
    return dag


def test_harness_runs_real_executor_without_cuda_and_restores_event_type() -> None:
    original_event_type = torch.cuda.Event
    dag = _tiny_model_free_dag()
    structural_fields = {
        uid: (
            node.node_kind,
            node.compute_subkind,
            dict(node.tag),
            node.stream,
            dict(node.node_meta),
        )
        for uid, node in dag.nodes.items()
    }
    harness = ControlPlaneHarness(
        dag,
        actor_rank=4,
        workload_metadata={"model": "fixture"},
    )

    elapsed_ns = harness.run_once_ns()
    gaps = harness.issue_gap_summary_us(iterations=2)

    assert elapsed_ns > 0
    assert gaps["count"] == 2 * (len(dag.nodes) - 1)
    assert gaps["median"] > 0.0
    assert torch.cuda.Event is original_event_type
    assert sum(stream.wait_count for stream in harness.runtime.streams.values()) > 0
    assert all(node.task_type is TaskType.ORDER_DUMMY for node in dag.nodes.values())
    assert [pred.uid for pred in dag.nodes["bwd"].data_preds] == ["ep_a2a"]
    assert dag.nodes["fwd"].batches[0].stage_id == 4
    assert dag.nodes["fwd"].batches[0].mb_idx == 2
    assert harness.metadata()["workload_metadata"] == {"model": "fixture"}
    assert {
        uid: (
            node.node_kind,
            node.compute_subkind,
            dict(node.tag),
            node.stream,
            dict(node.node_meta),
        )
        for uid, node in dag.nodes.items()
    } == structural_fields


def test_statistics_and_linear_fit() -> None:
    summary = summarize_ns([1000, 2000, 3000, 4000])
    fit = linear_fit([(1, 7.0), (2, 9.0), (4, 13.0)])

    assert summary["median"] == 2.5
    assert summary["p95"] == pytest.approx(3.85)
    assert summary["variance"] == pytest.approx(5.0 / 3.0)
    assert summary["standard_error"] == pytest.approx(
        summary["stdev"] / 2.0
    )
    assert summary["mean_ci95_low"] < summary["mean"]
    assert summary["mean_ci95_high"] > summary["mean"]
    assert fit["intercept_us"] == pytest.approx(5.0)
    assert fit["slope_us_per_node"] == pytest.approx(2.0)
    assert fit["r_squared"] == pytest.approx(1.0)


def test_actor_count_is_capped_at_two_h100_nodes() -> None:
    validate_actor_count(MAX_BENCHMARK_ACTORS)

    try:
        validate_actor_count(MAX_BENCHMARK_ACTORS + 1)
    except ValueError as exc:
        assert "exceeds the benchmark limit" in str(exc)
    else:
        raise AssertionError("actor count above 16 should be rejected")
