from __future__ import annotations

import pytest
import torch

from src.dag import TrainingDAG, TrainingDAGNode
from src.qwen_control_plane_workloads import (
    QwenDualPipeWorkloadSpec,
    QwenTrainingDAGFactory,
    build_qwen_dualpipe_schedule,
    load_qwen_control_plane_workload,
    save_qwen_control_plane_workload,
    strip_training_dag_payload,
)


def _directives(schedule, op: str):
    return [directive for directive in schedule if directive["op"] == op]


def test_schedule_uses_v_placement_zero3_buckets_and_shared_collectives() -> None:
    spec = QwenDualPipeWorkloadSpec(
        pp_degree=2,
        dp_degree=4,
        ep_degree=4,
        bucket_size_mb=64,
        collective_stream_policy="shared",
    )

    schedule = build_qwen_dualpipe_schedule(spec)
    places = _directives(schedule, "place")
    replicas = _directives(schedule, "replicate")
    shards = _directives(schedule, "shard")

    assert [directive["devices"] for directive in places] == [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [1, 3, 5, 7],
        [0, 2, 4, 6],
    ]
    assert len(replicas) == 4
    assert all(directive["shard_params"] for directive in replicas)
    assert all(directive["shard_grads"] for directive in replicas)
    assert all(directive["bucket_size"] == 64 for directive in replicas)
    assert all(
        directive["gather_stream"] == "collective_stream"
        and directive["reduce_stream"] == "collective_stream"
        for directive in replicas
    )
    assert len(shards) == 4
    assert all(directive["stream"] == "collective_stream" for directive in shards)
    assert _directives(schedule, "split")[0]["num_microbatches"] == 4
    assert len(_directives(schedule, "order")) == 2


def test_replicated_ep_omits_a2a_and_separates_dp_stream() -> None:
    spec = QwenDualPipeWorkloadSpec(
        pp_degree=2,
        dp_degree=4,
        ep_degree=1,
        bucket_size_mb=None,
        collective_stream_policy="separate",
    )

    schedule = build_qwen_dualpipe_schedule(spec)

    assert not _directives(schedule, "shard")
    assert all(
        directive["gather_stream"] == "dp_stream"
        and directive["reduce_stream"] == "dp_stream"
        for directive in _directives(schedule, "replicate")
    )


def test_schedule_rejects_independent_ep_subgroup_not_expressible_by_piper() -> None:
    spec = QwenDualPipeWorkloadSpec(
        pp_degree=2,
        dp_degree=8,
        ep_degree=4,
        bucket_size_mb=None,
        collective_stream_policy="shared",
    )

    with pytest.raises(ValueError, match="ep_degree=1 or ep_degree=dp_degree"):
        spec.validate()


def test_payload_stripping_retains_only_control_metadata() -> None:
    dag = TrainingDAG()
    node = TrainingDAGNode(
        uid="compute",
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={"PP": 0, "PASS": "F"},
        device=[0, 2],
        stream="default_stream",
        node_meta={
            "gm": object(),
            "graphargs": [torch.empty(4, device="meta")],
            "bucket_key": "bucket.0",
            "participants": [0, 2],
        },
    )
    dag.add_node(node)

    strip_training_dag_payload(dag)

    assert node.node_meta == {
        "bucket_key": "bucket.0",
        "participants": [0, 2],
    }


def test_real_397b_meta_trace_bucketing_increases_finalized_dag_size(
    tmp_path,
) -> None:
    factory = QwenTrainingDAGFactory(sequence_length=64)
    base = dict(
        pp_degree=1,
        dp_degree=2,
        ep_degree=2,
        collective_stream_policy="separate",
    )

    unbucketed = factory.build(
        QwenDualPipeWorkloadSpec(bucket_size_mb=None, **base)
    )
    retargeted = factory.build(
        QwenDualPipeWorkloadSpec(
            pp_degree=1,
            dp_degree=4,
            ep_degree=4,
            bucket_size_mb=None,
            collective_stream_policy="separate",
        )
    )
    assert retargeted.metadata["structural_template_cache_hit"] is True
    assert retargeted.metadata["structural_template_source_case"] == (
        unbucketed.spec.case_name
    )
    assert set(retargeted.per_actor_dags[0].nodes) == set(
        unbucketed.per_actor_dags[0].nodes
    )
    assert set(retargeted.per_actor_dags[0].edges) == set(
        unbucketed.per_actor_dags[0].edges
    )
    assert all(
        len(node.device) == 4
        for node in retargeted.per_actor_dags[0].nodes.values()
        if node.device is not None
    )
    assert all(
        node.node_meta.get("participants") == [0, 1, 2, 3]
        for node in retargeted.per_actor_dags[0].nodes.values()
        if "participants" in node.node_meta
    )
    bucketed = factory.build(
        QwenDualPipeWorkloadSpec(bucket_size_mb=16, **base)
    )

    unbucketed_summary = unbucketed.metadata["dag_summary"]
    bucketed_summary = bucketed.metadata["dag_summary"]
    assert bucketed_summary["nodes_per_actor_max"] > unbucketed_summary[
        "nodes_per_actor_max"
    ]
    assert bucketed_summary["node_kinds"]["ALL_GATHER_COMM"] > (
        unbucketed_summary["node_kinds"]["ALL_GATHER_COMM"]
    )
    assert bucketed_summary["node_kinds"]["REDUCE_SCATTER_COMM"] > (
        unbucketed_summary["node_kinds"]["REDUCE_SCATTER_COMM"]
    )
    assert bucketed_summary["node_kinds"]["A2A_COMM"] == (
        unbucketed_summary["node_kinds"]["A2A_COMM"]
    )
    for dag in bucketed.per_actor_dags:
        assert all("gm" not in node.node_meta for node in dag.nodes.values())
        assert all("graphargs" not in node.node_meta for node in dag.nodes.values())

    cache_info = save_qwen_control_plane_workload(bucketed, tmp_path)
    restored = load_qwen_control_plane_workload(bucketed.spec, tmp_path)
    assert cache_info["physical_pp_rank"] == 0
    assert cache_info["node_count"] == len(bucketed.per_actor_dags[0].nodes)
    assert len(restored.per_actor_dags[0].nodes) == cache_info["node_count"]
    assert restored.metadata["dag_cache"]["pickle_sha256"] == (
        cache_info["pickle_sha256"]
    )
