import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.build_schedule import build_afpipe_baseline_schedule
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.directives import (
    _apply_split_directive,
    _insert_cross_group_routed_comm_nodes,
    _insert_shard_a2a_comm_nodes,
    _validate_route_owned_parameters,
    apply_schedule_directives,
)
from src.ordering import (
    _resolve_default_stream_order,
    _serial_topological_order,
    execution_stream_domain,
    resolve_total_order_per_stream,
)
from src.piper import (
    _order_communication_transactions,
    _split_global_training_dag_by_pp_rank,
    _validate_dense_state_sequences,
    _validate_global_p2p_transaction_graph,
    _validate_p2p_sequences,
    _validate_routed_collective_sequences,
)
from src.simulator import simulate_training_step
from src.tasks import TaskType, training_dag_task_type


def _compute(
    uid: str,
    *,
    region: str,
    index: int,
    compute_subkind: str,
    devices: list[int],
    fwd_uid: str | None = None,
) -> TrainingDAGNode:
    node_meta = {}
    if fwd_uid is not None:
        node_meta["fwd_uid"] = fwd_uid
        node_meta["bucket_key"] = fwd_uid
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind=compute_subkind,
        tag={region: index, "PASS": "F" if compute_subkind == "FWD" else "B"},
        device=list(devices),
        stream="default_stream",
        node_meta=node_meta,
    )


def _cross_group_routed_dag(
    dense_devices: list[int],
    expert_devices: list[int],
) -> TrainingDAG:
    dag = TrainingDAG()
    pre = _compute(
        "pre.fwd",
        region="DENSE",
        index=0,
        compute_subkind="FWD",
        devices=dense_devices,
    )
    expert = _compute(
        "expert.fwd",
        region="ROUTED",
        index=0,
        compute_subkind="FWD",
        devices=expert_devices,
    )
    expert.node_meta["input_names"] = ["routed_tokens", "routing_counts"]
    expert.node_meta["output_names"] = ["expert_output"]
    expert.node_meta["tensor_output_indices"] = [0]
    expert.node_meta["differentiable_output_indices"] = [0]
    post = _compute(
        "post.fwd",
        region="DENSE",
        index=1,
        compute_subkind="FWD",
        devices=dense_devices,
    )
    pre.node_meta["a2a_boundary_after"] = {
        "tensor_idx": 0,
        "routing_counts_idx": 1,
        "routing_counts_schema": {
            "shape": (1, 8),
            "dtype": "torch.int64",
            "requires_grad": False,
        },
    }
    pre.node_meta["output_names"] = ["dense_output", "routing_counts"]
    pre.node_meta["tensor_output_indices"] = [0, 1]
    pre.node_meta["differentiable_output_indices"] = [0]
    expert.node_meta["a2a_boundary_after"] = {"tensor_idx": 0}
    post.node_meta["input_names"] = ["expert_output"]
    pre.node_meta["runtime_outputs_by_destination"] = {
        "expert.fwd": [
            {
                "value_id": "dense_output",
                "source_output_idx": 0,
                "destination_input_idx": 0,
                "is_tensor": True,
            },
            {
                "value_id": "routing_counts",
                "source_output_idx": 1,
                "destination_input_idx": 1,
                "is_tensor": True,
            },
        ]
    }
    expert.node_meta["runtime_outputs_by_destination"] = {
        "post.fwd": [
            {
                "value_id": "expert_output",
                "source_output_idx": 0,
                "destination_input_idx": 0,
                "is_tensor": True,
            }
        ]
    }

    post_bwd = _compute(
        "post.bwd",
        region="DENSE",
        index=1,
        compute_subkind="BWD",
        devices=dense_devices,
        fwd_uid="post.fwd",
    )
    expert_bwd = _compute(
        "expert.bwd",
        region="ROUTED",
        index=0,
        compute_subkind="BWD",
        devices=expert_devices,
        fwd_uid="expert.fwd",
    )
    pre_bwd = _compute(
        "pre.bwd",
        region="DENSE",
        index=0,
        compute_subkind="BWD",
        devices=dense_devices,
        fwd_uid="pre.fwd",
    )
    for node in (pre, expert, post, post_bwd, expert_bwd, pre_bwd):
        dag.add_node(node)
    dag.add_edge(TrainingDAGEdge("pre.fwd", "expert.fwd", "data", "tokens"))
    dag.add_edge(TrainingDAGEdge("expert.fwd", "post.fwd", "data", "tokens"))
    dag.add_edge(TrainingDAGEdge("post.bwd", "expert.bwd", "data", "tokens"))
    dag.add_edge(TrainingDAGEdge("expert.bwd", "pre.bwd", "data", "tokens"))
    return dag


@pytest.mark.parametrize(
    ("dense_devices", "expert_devices"),
    [
        ([0, 2, 4, 5], [1, 3]),
        # Equal degrees are still cross-group communication when participants
        # are disjoint; this must not collapse to an EP all-to-all.
        ([0, 2], [1, 3]),
    ],
)
def test_cross_group_routed_edges_lower_to_paired_m_to_n_nodes(
    dense_devices: list[int],
    expert_devices: list[int],
) -> None:
    dag = _cross_group_routed_dag(dense_devices, expert_devices)

    _insert_shard_a2a_comm_nodes(
        dag,
        [{"ROUTED": 0}],
        expert_devices,
        comm_stream="routed_stream",
    )

    routed = [
        node
        for node in dag.nodes.values()
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
    ]
    assert len(routed) == 8
    assert not any(node.node_kind == "A2A_COMM" for node in dag.nodes.values())
    assert not any(
        dag.nodes[edge.src_uid].node_kind == "COMPUTE"
        and dag.nodes[edge.dst_uid].node_kind == "COMPUTE"
        for edge in dag.edges
    )

    by_comm_id: dict[str, list[TrainingDAGNode]] = {}
    for node in routed:
        by_comm_id.setdefault(node.node_meta["comm_id"], []).append(node)
        assert node.stream == "routed_stream"
        assert node.node_meta["comm_group_kind"] == "routed_cross_group"
        assert node.node_meta["expert_fwd_uid"] == "expert.fwd"
        assert node.node_meta["expert_input_arity"] == 2
        assert node.node_meta["expert_output_arity"] == 1
        assert node.node_meta["participants"] == sorted(
            node.node_meta["source_devices"]
            + node.node_meta["destination_devices"]
        )
    assert len(by_comm_id) == 4
    for pair in by_comm_id.values():
        assert {node.node_meta["role"] for node in pair} == {
            "source",
            "destination",
        }
        source = next(node for node in pair if node.node_meta["role"] == "source")
        destination = next(
            node for node in pair if node.node_meta["role"] == "destination"
        )
        assert source.device == source.node_meta["source_devices"]
        assert destination.device == destination.node_meta["destination_devices"]

    dispatch = [
        node
        for node in routed
        if node.node_meta["direction"] == "incoming"
        and node.tag["PASS"] == "F"
    ]
    assert len(dispatch) == 2
    assert {node.node_meta["routing_counts_idx"] for node in dispatch} == {1}
    assert {node.node_kind for node in dispatch} == {"M_TO_N_COMM"}

    assert {
        training_dag_task_type(node)
        for node in routed
    } == {
        TaskType.FWD_M_TO_N,
        TaskType.BWD_M_TO_N,
        TaskType.FWD_N_TO_M,
        TaskType.BWD_N_TO_M,
    }


def test_cross_group_routed_arity_excludes_lifted_parameter_placeholders() -> None:
    dag = _cross_group_routed_dag([0, 2, 4, 5], [1, 3])
    expert = dag.nodes["expert.fwd"]
    expert.node_meta["input_names"] = [
        "routed_tokens",
        "routing_counts",
        "expert_weight_0",
        "expert_weight_1",
        "expert_weight_2",
    ]
    expert.node_meta["input_idxs"] = [0, 1]
    expert.node_meta["param_idxs"] = [2, 3, 4]

    _insert_shard_a2a_comm_nodes(
        dag,
        [{"ROUTED": 0}],
        [1, 3],
        comm_stream="routed_stream",
    )

    routed = [
        node
        for node in dag.nodes.values()
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
    ]
    assert routed
    assert {node.node_meta["expert_input_arity"] for node in routed} == {2}


def test_route_owned_parameters_use_schedule_derived_local_route_width() -> None:
    node = _compute(
        "expert.fwd",
        region="ROUTED",
        index=0,
        compute_subkind="FWD",
        devices=[3, 1],
    )
    node.node_meta.update(
        graphargs=[
            torch.empty(3, 4),
            torch.empty(1, 8, dtype=torch.int64),
            torch.nn.Parameter(torch.empty(4, 2, 3)),
        ],
        input_idxs=[0, 1],
        param_idxs=[2],
        input_names=["payload", "counts", "expert_weight"],
    )
    ownership = {
        "num_global_routes": 8,
        "num_local_routes": 4,
    }

    assert _validate_route_owned_parameters(node, ownership) == [
        {
            "arg_idx": 2,
            "placeholder_name": "expert_weight",
            "route_axis": 0,
            "local_shape": (4, 2, 3),
            "global_shape": (8, 2, 3),
        }
    ]

    node.node_meta["graphargs"][2] = torch.nn.Parameter(torch.empty(8, 2, 3))
    with pytest.raises(ValueError, match="num_local_routes=4"):
        _validate_route_owned_parameters(node, ownership)


def test_replicate_and_shard_cannot_select_the_same_model_region() -> None:
    dag = _cross_group_routed_dag([0, 1], [0, 1])
    dag.add_node(
        TrainingDAGNode(
            uid="upd.0",
            node_kind="UPD",
            compute_subkind=None,
            tag={"PASS": None},
            device=[0, 1],
            stream="default_stream",
            node_meta={},
        )
    )
    for uid in ("pre.bwd", "expert.bwd", "post.bwd"):
        dag.add_edge(TrainingDAGEdge(uid, "upd.0", "temporal"))
    expert = dag.nodes["expert.fwd"]
    expert.node_meta.update(
        graphargs=[
            torch.empty(3, 4),
            torch.empty(1, 8, dtype=torch.int64),
            torch.nn.Parameter(torch.empty(4, 2, 3)),
        ],
        input_idxs=[0, 1],
        param_idxs=[2],
        input_names=["routed_tokens", "routing_counts", "expert_weight"],
    )
    directives = [
        {"op": "place", "filter": {}, "devices": [0, 1]},
        {"op": "replicate", "filter": {}, "devices": [0, 1]},
        {
            "op": "shard",
            "filter": {"ROUTED": 0},
            "devices": [0, 1],
        },
    ]

    with pytest.raises(
        ValueError,
        match="replicate and shard directives must select disjoint model regions",
    ):
        apply_schedule_directives(dag, directives)


def test_place_can_select_regions_owned_by_replicate_or_shard() -> None:
    dag = _cross_group_routed_dag([0, 1], [0, 1])
    dag.add_node(
        TrainingDAGNode(
            uid="upd.0",
            node_kind="UPD",
            compute_subkind=None,
            tag={"PASS": None},
            device=None,
            stream="default_stream",
            node_meta={},
        )
    )
    for uid in ("pre.bwd", "expert.bwd", "post.bwd"):
        dag.add_edge(TrainingDAGEdge(uid, "upd.0", "temporal"))
    for node in dag.nodes.values():
        node.device = None
    expert = dag.nodes["expert.fwd"]
    expert.node_meta.update(
        graphargs=[
            torch.empty(3, 4),
            torch.empty(1, 8, dtype=torch.int64),
            torch.nn.Parameter(torch.empty(4, 2, 3)),
        ],
        input_idxs=[0, 1],
        param_idxs=[2],
        input_names=["routed_tokens", "routing_counts", "expert_weight"],
    )

    apply_schedule_directives(
        dag,
        [
            {"op": "place", "filter": {}, "devices": [0, 1]},
            {
                "op": "replicate",
                "filter": {"ROUTED": "-"},
                "devices": [0, 1],
            },
            {
                "op": "shard",
                "filter": {"ROUTED": 0},
                "devices": [0, 1],
            },
        ],
    )

    assert expert.node_meta["route_sharded_param_specs"]


def test_schedule_classifies_cross_group_routing_before_generic_p2p() -> None:
    dense_devices = [0, 2, 4, 5]
    expert_devices = [1, 3]
    dag = _cross_group_routed_dag(dense_devices, expert_devices)
    for node in dag.nodes.values():
        node.device = None

    apply_schedule_directives(
        dag,
        [
            {
                "op": "place",
                "filter": {"DENSE": "*"},
                "devices": dense_devices,
            },
            {
                "op": "place",
                "filter": {"ROUTED": "*"},
                "devices": expert_devices,
            },
            {
                "op": "replicate",
                "filter": {"DENSE": "*"},
                "devices": dense_devices,
            },
            {
                "op": "shard",
                "filter": {"ROUTED": "*"},
                "devices": expert_devices,
            },
        ],
    )

    assert not any(
        node.node_kind in {"SEND_COMM", "RECV_COMM", "A2A_COMM"}
        for node in dag.nodes.values()
    )
    assert sum(node.node_kind == "M_TO_N_COMM" for node in dag.nodes.values()) == 4
    assert sum(node.node_kind == "N_TO_M_COMM" for node in dag.nodes.values()) == 4


def _placed_routed_schedule(
    *,
    pre_devices: list[int],
    routed_devices: list[int],
    post_devices: list[int],
    shard_filter: dict[str, object] | None = None,
    replicate_dense: bool = True,
) -> list[dict[str, object]]:
    schedule: list[dict[str, object]] = [
        {
            "op": "place",
            "filter": {"DENSE": 0},
            "devices": pre_devices,
        },
        {
            "op": "place",
            "filter": {"ROUTED": 0},
            "devices": routed_devices,
        },
        {
            "op": "place",
            "filter": {"DENSE": 1},
            "devices": post_devices,
        },
    ]
    if replicate_dense:
        for dense_index, dense_devices in ((0, pre_devices), (1, post_devices)):
            schedule.append(
                {
                    "op": "replicate",
                    "filter": {"DENSE": dense_index},
                    "devices": dense_devices,
                }
            )
    schedule.append(
        {
            "op": "shard",
            "filter": shard_filter or {"ROUTED": 0},
            "devices": routed_devices,
        }
    )
    return schedule


def test_schedule_rejects_hybrid_within_and_cross_group_routed_triplet() -> None:
    dag = _cross_group_routed_dag([0, 1], [0, 1])
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="mixes within-group and cross-group"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 1],
                routed_devices=[0, 1],
                post_devices=[2, 3],
            ),
        )


def test_schedule_rejects_routed_region_fanout_before_comm_insertion() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    extra_post = _compute(
        "post2.fwd",
        region="DENSE",
        index=1,
        compute_subkind="FWD",
        devices=[0, 2],
    )
    extra_post_bwd = _compute(
        "post2.bwd",
        region="DENSE",
        index=1,
        compute_subkind="BWD",
        devices=[0, 2],
        fwd_uid="post2.fwd",
    )
    dag.add_node(extra_post)
    dag.add_node(extra_post_bwd)
    dag.add_edge(
        TrainingDAGEdge("expert.fwd", "post2.fwd", "data", "other_tokens")
    )
    dag.add_edge(
        TrainingDAGEdge("post2.bwd", "expert.bwd", "data", "other_tokens")
    )
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="exactly one forward dispatch edge"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[0, 2],
            ),
        )


def test_schedule_rejects_partial_routed_forward_backward_lifecycle() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="but not its backward activation region"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[0, 2],
                shard_filter={"ROUTED": 0, "PASS": "F"},
            ),
        )


def test_schedule_rejects_marker_region_without_shard_policy() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    for node in dag.nodes.values():
        node.device = None
    schedule = _placed_routed_schedule(
        pre_devices=[0, 2],
        routed_devices=[1, 3],
        post_devices=[0, 2],
    )
    schedule = [directive for directive in schedule if directive["op"] != "shard"]

    with pytest.raises(ValueError, match="missing a shard directive"):
        apply_schedule_directives(dag, schedule)


def test_schedule_rejects_shard_policy_for_unmarked_region() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    dag.nodes["pre.fwd"].node_meta.pop("a2a_boundary_after")
    dag.nodes["expert.fwd"].node_meta.pop("a2a_boundary_after")
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="without a marker triplet"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[0, 2],
            ),
        )


def test_schedule_accepts_explicit_singleton_shard_policy() -> None:
    dag = _cross_group_routed_dag([0], [0])
    for node in dag.nodes.values():
        node.device = None

    apply_schedule_directives(
        dag,
        _placed_routed_schedule(
            pre_devices=[0],
            routed_devices=[0],
            post_devices=[0],
        ),
    )

    assert sum(node.node_kind == "A2A_COMM" for node in dag.nodes.values()) == 4


def _use_flexible_same_group_routed_layout(dag: TrainingDAG) -> None:
    pre = dag.nodes["pre.fwd"]
    routed = dag.nodes["expert.fwd"]
    pre.node_meta["a2a_boundary_after"].update(
        tensor_idx=1,
        routing_counts_idx=0,
    )
    pre.node_meta["output_names"] = ["routing_counts", "routed_tokens"]
    pre.node_meta["tensor_output_indices"] = [0, 1]
    pre.node_meta["differentiable_output_indices"] = [1]
    routed.node_meta["input_names"] = ["routing_counts", "routed_tokens"]
    routed.node_meta["output_names"] = ["expert_output"]
    routed.node_meta["a2a_boundary_after"]["tensor_idx"] = 0
    pre.node_meta["runtime_outputs_by_destination"]["expert.fwd"] = [
        {
            "value_id": "routing_counts",
            "source_output_idx": 0,
            "destination_input_idx": 0,
            "is_tensor": True,
        },
        {
            "value_id": "routed_tokens",
            "source_output_idx": 1,
            "destination_input_idx": 1,
            "is_tensor": True,
        },
    ]


def test_same_group_a2a_allows_flexible_routed_buffer_layout() -> None:
    dag = _cross_group_routed_dag([0, 1], [0, 1])
    _use_flexible_same_group_routed_layout(dag)
    dag.nodes["pre.fwd"].node_meta["a2a_boundary_after"]["payload_schema"] = {
        "shape": (8, 4),
        "dtype": "torch.bfloat16",
        "requires_grad": True,
    }
    dag.nodes["expert.fwd"].node_meta["a2a_boundary_after"]["payload_schema"] = {
        "shape": (8, 4),
        "dtype": "torch.float32",
        "requires_grad": True,
    }
    for node in dag.nodes.values():
        node.device = None

    apply_schedule_directives(
        dag,
        _placed_routed_schedule(
            pre_devices=[0, 1],
            routed_devices=[0, 1],
            post_devices=[0, 1],
        ),
    )

    assert sum(node.node_kind == "A2A_COMM" for node in dag.nodes.values()) == 4


@pytest.mark.parametrize(
    ("node_uid", "indices", "message"),
    [
        ("pre.fwd", [], "dispatch payload slot"),
        ("expert.fwd", [], "combine payload slot"),
    ],
)
def test_same_group_a2a_requires_differentiable_routed_payloads(
    node_uid: str,
    indices: list[int],
    message: str,
) -> None:
    dag = _cross_group_routed_dag([0, 1], [0, 1])
    dag.nodes[node_uid].node_meta["differentiable_output_indices"] = indices
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match=message):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 1],
                routed_devices=[0, 1],
                post_devices=[0, 1],
            ),
        )


def test_cross_group_route_allows_marker_defined_payload_count_slots() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    _use_flexible_same_group_routed_layout(dag)
    for node in dag.nodes.values():
        node.device = None

    apply_schedule_directives(
        dag,
        _placed_routed_schedule(
            pre_devices=[0, 2],
            routed_devices=[1, 3],
            post_devices=[0, 2],
        ),
    )

    routed = [
        node
        for node in dag.nodes.values()
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
    ]
    assert len(routed) == 8
    incoming = next(
        node
        for node in routed
        if node.node_kind == "M_TO_N_COMM"
        and node.tag["PASS"] == "F"
        and node.node_meta["role"] == "source"
    )
    assert incoming.node_meta["a2a_tensor_idx"] == 1
    assert incoming.node_meta["routing_counts_idx"] == 0


def test_cross_group_route_rejects_payload_dtype_change_early() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    dag.nodes["pre.fwd"].node_meta["a2a_boundary_after"]["payload_schema"] = {
        "shape": (8, 4),
        "dtype": "torch.bfloat16",
        "requires_grad": True,
    }
    dag.nodes["expert.fwd"].node_meta["a2a_boundary_after"]["payload_schema"] = {
        "shape": (8, 4),
        "dtype": "torch.float32",
        "requires_grad": True,
    }
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="must preserve payload dtype"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[0, 2],
            ),
        )


def test_cross_group_route_rejects_payload_shape_change_early() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    dag.nodes["pre.fwd"].node_meta["a2a_boundary_after"]["payload_schema"] = {
        "shape": (8, 4),
        "dtype": "torch.bfloat16",
        "requires_grad": True,
    }
    dag.nodes["expert.fwd"].node_meta["a2a_boundary_after"]["payload_schema"] = {
        "shape": (8, 8),
        "dtype": "torch.bfloat16",
        "requires_grad": True,
    }
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="must preserve the payload trailing shape"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[0, 2],
            ),
        )


def test_schedule_rejects_static_route_count_width_mismatch() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3, 5])
    dag.nodes["pre.fwd"].node_meta["a2a_boundary_after"][
        "routing_counts_schema"
    ] = {
        "shape": (1, 8),
        "dtype": "torch.int64",
        "requires_grad": False,
    }
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="not divisible"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3, 5],
                post_devices=[0, 2],
            ),
        )


def test_schedule_rejects_legacy_rank1_route_counts_schema() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    dag.nodes["pre.fwd"].node_meta["a2a_boundary_after"][
        "routing_counts_schema"
    ]["shape"] = (8,)
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="statically known rank-2"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[0, 2],
            ),
        )


def test_shard_devices_define_contiguous_ownership_in_canonical_rank_order() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3, 5])
    dag.nodes["pre.fwd"].node_meta["a2a_boundary_after"][
        "routing_counts_schema"
    ]["shape"] = (1, 12)
    for node in dag.nodes.values():
        node.device = None

    schedule = _placed_routed_schedule(
        pre_devices=[0, 2],
        routed_devices=[1, 3, 5],
        post_devices=[0, 2],
    )
    apply_schedule_directives(dag, schedule)

    expected = {
        "num_global_routes": 12,
        "shard_degree": 3,
        "shard_devices": [1, 3, 5],
        "num_local_routes": 4,
        "route_partitions": [
            {"device": 1, "start": 0, "stop": 4},
            {"device": 3, "start": 4, "stop": 8},
            {"device": 5, "start": 8, "stop": 12},
        ],
    }
    routed_nodes = [
        dag.nodes["expert.fwd"],
        dag.nodes["expert.bwd"],
        *[
            node
            for node in dag.nodes.values()
            if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
        ],
    ]
    assert len(routed_nodes) == 10
    for node in routed_nodes:
        assert {key: node.node_meta[key] for key in expected} == expected


def test_apply_schedule_rejects_unsorted_shard_devices() -> None:
    dag = _cross_group_routed_dag([0, 4], [0, 4])
    dag.nodes["pre.fwd"].node_meta["a2a_boundary_after"][
        "routing_counts_schema"
    ]["shape"] = (1, 6)
    for node in dag.nodes.values():
        node.device = None

    schedule = _placed_routed_schedule(
        pre_devices=[0, 4],
        routed_devices=[0, 4],
        post_devices=[0, 4],
    )
    schedule[-1]["devices"] = [4, 0]
    with pytest.raises(ValueError, match="ascending global actor-rank order"):
        apply_schedule_directives(dag, schedule)


def test_schedule_requires_wide_dense_neighbors_to_be_replicated() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="covered by a replicate directive"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[0, 2],
                replicate_dense=False,
            ),
        )


def test_schedule_rejects_unequal_dense_state_relay_widths_early() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    for node in dag.nodes.values():
        node.device = None

    with pytest.raises(ValueError, match="equal-width dense ActorGroups"):
        apply_schedule_directives(
            dag,
            _placed_routed_schedule(
                pre_devices=[0, 2],
                routed_devices=[1, 3],
                post_devices=[4, 5, 6],
            ),
        )


def test_cross_group_routing_rejects_partially_overlapping_actor_groups() -> None:
    dag = _cross_group_routed_dag([0, 2, 4], [1, 2])
    with pytest.raises(ValueError, match="partially overlapping device sets"):
        _insert_cross_group_routed_comm_nodes(
            dag,
            [{"ROUTED": 0}],
            [1, 2],
        )


def test_split_remaps_routed_expert_state_and_transaction_ids() -> None:
    dag = _cross_group_routed_dag([0, 2], [1, 3])
    _insert_shard_a2a_comm_nodes(dag, [{"ROUTED": 0}], [1, 3])

    _apply_split_directive(dag, {}, "MB", 2)

    mb0 = [
        node
        for node in dag.nodes.values()
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
        and node.tag["MB"] == 0
    ]
    mb1 = [
        node
        for node in dag.nodes.values()
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
        and node.tag["MB"] == 1
    ]
    assert len(mb0) == len(mb1) == 8
    assert {node.node_meta["expert_fwd_uid"] for node in mb0} == {"expert.fwd"}
    assert {node.node_meta["expert_fwd_uid"] for node in mb1} == {
        "expert.fwd.splitMB1"
    }
    assert not (
        {node.node_meta["comm_id"] for node in mb0}
        & {node.node_meta["comm_id"] for node in mb1}
    )
    assert all(node.node_meta["comm_id"].endswith(".splitMB1") for node in mb1)


def _routed_actor_node(
    uid: str,
    *,
    node_kind: str,
    role: str,
    comm_id: str,
    local_devices: list[int],
    source_devices: list[int],
    destination_devices: list[int],
) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind=node_kind,
        compute_subkind=None,
        tag={"PASS": "F"},
        device=list(local_devices),
        stream="routed_stream",
        node_meta={
            "role": role,
            "direction": (
                "incoming" if node_kind == "M_TO_N_COMM" else "outgoing"
            ),
            "source_devices": list(source_devices),
            "destination_devices": list(destination_devices),
            "source_degree": len(source_devices),
            "destination_degree": len(destination_devices),
            "participants": sorted(source_devices + destination_devices),
            "expert_fwd_uid": f"expert.{comm_id}",
            "expert_input_arity": 2,
            "expert_output_arity": 1,
            "a2a_tensor_idx": 0,
            "routing_counts_idx": (
                1 if node_kind == "M_TO_N_COMM" else None
            ),
            "bucket_key": f"expert.{comm_id}",
            "comm_group_kind": "routed_cross_group",
            "comm_id": comm_id,
        },
    )


def _routed_actor_dag(
    local_devices: list[int],
    specs: list[tuple[str, str, str, str, list[int], list[int]]],
    order: list[str],
) -> TrainingDAG:
    dag = TrainingDAG()
    for uid, node_kind, role, comm_id, source_devices, destination_devices in specs:
        dag.add_node(
            _routed_actor_node(
                uid,
                node_kind=node_kind,
                role=role,
                comm_id=comm_id,
                local_devices=local_devices,
                source_devices=source_devices,
                destination_devices=destination_devices,
            )
        )
    for src_uid, dst_uid in zip(order, order[1:]):
        dag.add_edge(TrainingDAGEdge(src_uid, dst_uid, "temporal"))
    return dag


def test_shared_routed_stream_rejects_cross_channel_issue_cycle() -> None:
    dense = [0, 2]
    expert = [1, 3]
    dense_dag = _routed_actor_dag(
        dense,
        [
            ("m0.src", "M_TO_N_COMM", "source", "m0", dense, expert),
            ("m1.src", "M_TO_N_COMM", "source", "m1", dense, expert),
            ("n1.dst", "N_TO_M_COMM", "destination", "n1", expert, dense),
            ("n0.dst", "N_TO_M_COMM", "destination", "n0", expert, dense),
        ],
        ["m0.src", "m1.src", "n1.dst", "n0.dst"],
    )
    expert_dag = _routed_actor_dag(
        expert,
        [
            ("n1.src", "N_TO_M_COMM", "source", "n1", expert, dense),
            ("n0.src", "N_TO_M_COMM", "source", "n0", expert, dense),
            ("m0.dst", "M_TO_N_COMM", "destination", "m0", dense, expert),
            ("m1.dst", "M_TO_N_COMM", "destination", "m1", dense, expert),
        ],
        ["n1.src", "n0.src", "m0.dst", "m1.dst"],
    )

    # Matching per-channel subsequences are insufficient when both channels
    # share one CUDA stream: the opposite local orders form a wait cycle.
    _validate_routed_collective_sequences([dense_dag, expert_dag])
    with pytest.raises(
        ValueError, match="global communication transaction graph is cyclic"
    ):
        _order_communication_transactions([dense_dag, expert_dag])


def test_shared_routed_stream_gets_one_mixed_collective_issue_order() -> None:
    dense = [0, 2]
    expert = [1, 3]
    dense_dag = _routed_actor_dag(
        dense,
        [
            ("m.src", "M_TO_N_COMM", "source", "m", dense, expert),
            ("n.dst", "N_TO_M_COMM", "destination", "n", expert, dense),
        ],
        [],
    )
    a2a = TrainingDAGNode(
        uid="a2a",
        node_kind="A2A_COMM",
        compute_subkind=None,
        tag={"PASS": "F"},
        device=list(dense),
        stream="routed_stream",
        node_meta={
            "comm_id": "a",
            "comm_group_kind": "ep",
            "participants": list(dense),
        },
    )
    dense_dag.add_node(a2a)
    expert_dag = _routed_actor_dag(
        expert,
        [
            ("m.dst", "M_TO_N_COMM", "destination", "m", dense, expert),
            ("n.src", "N_TO_M_COMM", "source", "n", expert, dense),
        ],
        [],
    )

    _order_communication_transactions([dense_dag, expert_dag])

    dense_temporal = {
        (edge.src_uid, edge.dst_uid)
        for edge in dense_dag.edges
        if edge.dep_kind == "temporal"
    }
    expert_temporal = {
        (edge.src_uid, edge.dst_uid)
        for edge in expert_dag.edges
        if edge.dep_kind == "temporal"
    }
    assert dense_temporal == {("a2a", "m.src"), ("m.src", "n.dst")}
    assert expert_temporal == {("m.dst", "n.src")}


def test_communication_ordering_rejects_endpoint_stream_mismatch() -> None:
    dense = [0, 2]
    expert = [1, 3]
    dense_dag = _routed_actor_dag(
        dense,
        [("src", "M_TO_N_COMM", "source", "m", dense, expert)],
        [],
    )
    expert_dag = _routed_actor_dag(
        expert,
        [("dst", "M_TO_N_COMM", "destination", "m", dense, expert)],
        [],
    )
    expert_dag.nodes["dst"].stream = "other_routed_stream"

    with pytest.raises(ValueError, match="endpoints disagree"):
        _order_communication_transactions([dense_dag, expert_dag])


def test_communication_ordering_requires_directional_p2p_metadata() -> None:
    dag = TrainingDAG()
    dag.add_node(
        TrainingDAGNode(
            uid="send",
            node_kind="SEND_COMM",
            compute_subkind=None,
            tag={},
            device=[0],
            stream="pp_stream",
            node_meta={"comm_id": "p2p", "peer_pp_rank": 1},
        )
    )

    with pytest.raises(ValueError, match="requires a directional"):
        _order_communication_transactions([dag])


def _p2p_compute_node(uid: str, device: int) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={"PASS": "F"},
        device=[device],
        stream="default_stream",
    )


def _add_p2p_transaction(
    source_dag: TrainingDAG,
    destination_dag: TrainingDAG,
    *,
    comm_id: str,
    source_compute_uid: str,
    destination_compute_uid: str,
    source_rank: int,
    destination_rank: int,
) -> tuple[str, str]:
    direction = "lo_hi" if source_rank < destination_rank else "hi_lo"
    common_meta = {
        "comm_id": comm_id,
        "comm_group_kind": f"pp_{direction}",
        "participants": sorted([source_rank, destination_rank]),
        "source_compute_uid": source_compute_uid,
        "destination_compute_uid": destination_compute_uid,
    }
    send_uid = f"send.{comm_id}"
    recv_uid = f"recv.{comm_id}"
    source_dag.add_node(
        TrainingDAGNode(
            uid=send_uid,
            node_kind="SEND_COMM",
            compute_subkind=None,
            tag={"PASS": "F"},
            device=[source_rank],
            stream="pp_stream",
            node_meta={**common_meta, "peer_pp_rank": destination_rank},
        )
    )
    destination_dag.add_node(
        TrainingDAGNode(
            uid=recv_uid,
            node_kind="RECV_COMM",
            compute_subkind=None,
            tag={"PASS": "F"},
            device=[destination_rank],
            stream="pp_stream",
            node_meta={**common_meta, "peer_pp_rank": source_rank},
        )
    )
    source_dag.add_edge(
        TrainingDAGEdge(source_compute_uid, send_uid, "data")
    )
    destination_dag.add_edge(
        TrainingDAGEdge(recv_uid, destination_compute_uid, "data")
    )
    return send_uid, recv_uid


def test_p2p_order_prefers_earlier_consumer_over_comm_id() -> None:
    destination_dag = TrainingDAG()
    source_dag = TrainingDAG()
    for dag, device, prefix in (
        (destination_dag, 0, "destination"),
        (source_dag, 1, "source"),
    ):
        dag.add_node(_p2p_compute_node(f"{prefix}.early", device))
        dag.add_node(_p2p_compute_node(f"{prefix}.late", device))
        dag.add_edge(
            TrainingDAGEdge(
                f"{prefix}.early",
                f"{prefix}.late",
                "temporal",
            )
        )
    early_send, early_recv = _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="z.early",
        source_compute_uid="source.early",
        destination_compute_uid="destination.early",
        source_rank=1,
        destination_rank=0,
    )
    late_send, late_recv = _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="a.late",
        source_compute_uid="source.late",
        destination_compute_uid="destination.late",
        source_rank=1,
        destination_rank=0,
    )

    _order_communication_transactions([destination_dag, source_dag])
    for dag in (destination_dag, source_dag):
        resolve_total_order_per_stream(dag)
    _validate_global_p2p_transaction_graph(
        [destination_dag, source_dag]
    )
    _validate_p2p_sequences([destination_dag, source_dag])

    assert any(
        edge.src_uid == early_send and edge.dst_uid == late_send
        for edge in source_dag.edges
    )
    assert any(
        edge.src_uid == early_recv and edge.dst_uid == late_recv
        for edge in destination_dag.edges
    )
    assert [
        node.node_meta["comm_id"]
        for node in (
            destination_dag.nodes[uid]
            for uid in _serial_topological_order(destination_dag)
        )
        if node.node_kind == "RECV_COMM"
        and execution_stream_domain(node) == "pp_hi_lo"
    ] == ["z.early", "a.late"]

    for dag in (destination_dag, source_dag):
        for node in dag.nodes.values():
            node.profile_times_by_mode = {
                "interference_free": {
                    node.device[0]: SimpleNamespace(estimate_us=1.0)
                }
            }
    assert simulate_training_step(
        [destination_dag, source_dag]
    ).step_time_us > 0


def test_p2p_order_keeps_opposite_directions_independent() -> None:
    dag_0 = TrainingDAG()
    dag_1 = TrainingDAG()
    for uid in ("compute.0.source", "compute.0.destination"):
        dag_0.add_node(_p2p_compute_node(uid, 0))
    for uid in ("compute.1.source", "compute.1.destination"):
        dag_1.add_node(_p2p_compute_node(uid, 1))
    one_to_zero_send, one_to_zero_recv = _add_p2p_transaction(
        dag_1,
        dag_0,
        comm_id="one_to_zero",
        source_compute_uid="compute.1.source",
        destination_compute_uid="compute.0.destination",
        source_rank=1,
        destination_rank=0,
    )
    zero_to_one_send, zero_to_one_recv = _add_p2p_transaction(
        dag_0,
        dag_1,
        comm_id="zero_to_one",
        source_compute_uid="compute.0.source",
        destination_compute_uid="compute.1.destination",
        source_rank=0,
        destination_rank=1,
    )

    _order_communication_transactions([dag_0, dag_1])

    edges = {
        (edge.src_uid, edge.dst_uid)
        for dag in (dag_0, dag_1)
        for edge in dag.edges
    }
    assert (one_to_zero_send, zero_to_one_recv) not in edges
    assert (zero_to_one_send, one_to_zero_recv) not in edges


def test_p2p_order_rejects_cyclic_directional_transaction_graph() -> None:
    dag_0, dag_1, dag_2 = TrainingDAG(), TrainingDAG(), TrainingDAG()
    for dag, device, uids in (
        (dag_0, 0, ("a.source", "c.source")),
        (dag_1, 1, ("a.destination", "b.source")),
        (dag_2, 2, ("b.destination", "c.destination")),
    ):
        for uid in uids:
            dag.add_node(_p2p_compute_node(uid, device))
    send_a, recv_a = _add_p2p_transaction(
        dag_0,
        dag_1,
        comm_id="a",
        source_compute_uid="a.source",
        destination_compute_uid="a.destination",
        source_rank=0,
        destination_rank=1,
    )
    send_b, recv_b = _add_p2p_transaction(
        dag_1,
        dag_2,
        comm_id="b",
        source_compute_uid="b.source",
        destination_compute_uid="b.destination",
        source_rank=1,
        destination_rank=2,
    )
    send_c, recv_c = _add_p2p_transaction(
        dag_0,
        dag_2,
        comm_id="c",
        source_compute_uid="c.source",
        destination_compute_uid="c.destination",
        source_rank=0,
        destination_rank=2,
    )
    dag_0.add_edge(TrainingDAGEdge(send_c, send_a, "temporal"))
    dag_1.add_edge(TrainingDAGEdge(recv_a, send_b, "temporal"))
    dag_2.add_edge(TrainingDAGEdge(recv_b, recv_c, "temporal"))

    with pytest.raises(
        ValueError,
        match="global communication transaction graph is cyclic.*a.*b.*c",
    ):
        _order_communication_transactions([dag_0, dag_1, dag_2])


def test_p2p_order_rejects_missing_compute_anchor() -> None:
    destination_dag = TrainingDAG()
    source_dag = TrainingDAG()
    destination_dag.add_node(_p2p_compute_node("destination", 0))
    source_dag.add_node(_p2p_compute_node("source", 1))
    send_uid, _recv_uid = _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="transfer",
        source_compute_uid="source",
        destination_compute_uid="destination",
        source_rank=1,
        destination_rank=0,
    )
    del source_dag.nodes[send_uid].node_meta["source_compute_uid"]

    with pytest.raises(ValueError, match="source_compute_uid"):
        _order_communication_transactions([destination_dag, source_dag])


def test_p2p_validation_rejects_participant_mismatch() -> None:
    destination_dag = TrainingDAG()
    source_dag = TrainingDAG()
    destination_dag.add_node(_p2p_compute_node("destination", 0))
    source_dag.add_node(_p2p_compute_node("source", 1))
    send_uid, _recv_uid = _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="transfer",
        source_compute_uid="source",
        destination_compute_uid="destination",
        source_rank=1,
        destination_rank=0,
    )
    source_dag.nodes[send_uid].node_meta["participants"] = [0, 1, 2]

    with pytest.raises(ValueError, match="participants"):
        _validate_global_p2p_transaction_graph(
            [destination_dag, source_dag]
        )


def test_p2p_validation_ignores_non_p2p_collective_metadata() -> None:
    destination_dag = TrainingDAG()
    source_dag = TrainingDAG()
    destination_dag.add_node(_p2p_compute_node("destination", 0))
    source_dag.add_node(_p2p_compute_node("source", 1))
    _add_p2p_transaction(
        source_dag,
        destination_dag,
        comm_id="shared",
        source_compute_uid="source",
        destination_compute_uid="destination",
        source_rank=1,
        destination_rank=0,
    )
    source_dag.add_node(
        TrainingDAGNode(
            uid="unrelated.collective",
            node_kind="ALL_GATHER_COMM",
            compute_subkind=None,
            tag={},
            device=[1],
            stream="collective_stream",
            node_meta={"comm_id": "shared", "participants": "malformed"},
        )
    )

    _validate_global_p2p_transaction_graph(
        [destination_dag, source_dag]
    )


def test_communication_ordering_rejects_multi_actor_pair_cycle() -> None:
    group_a, group_b, group_c = [0], [1], [2]
    dag_a = _routed_actor_dag(
        group_a,
        [
            ("ab.src", "M_TO_N_COMM", "source", "ab", group_a, group_b),
            ("ac.src", "M_TO_N_COMM", "source", "ac", group_a, group_c),
        ],
        ["ac.src", "ab.src"],
    )
    dag_b = _routed_actor_dag(
        group_b,
        [
            ("ab.dst", "M_TO_N_COMM", "destination", "ab", group_a, group_b),
            ("bc.src", "M_TO_N_COMM", "source", "bc", group_b, group_c),
        ],
        ["ab.dst", "bc.src"],
    )
    dag_c = _routed_actor_dag(
        group_c,
        [
            ("bc.dst", "M_TO_N_COMM", "destination", "bc", group_b, group_c),
            ("ac.dst", "M_TO_N_COMM", "destination", "ac", group_a, group_c),
        ],
        ["bc.dst", "ac.dst"],
    )

    with pytest.raises(
        ValueError, match="global communication transaction graph is cyclic"
    ):
        _order_communication_transactions([dag_a, dag_b, dag_c])


def test_routed_collective_sequence_validation_rejects_mismatch() -> None:
    dense = [0, 2]
    expert = [1, 3]
    dense_specs = [
        ("m0.src", "M_TO_N_COMM", "source", "m0", dense, expert),
        ("m1.src", "M_TO_N_COMM", "source", "m1", dense, expert),
    ]
    expert_specs = [
        ("m1.dst", "M_TO_N_COMM", "destination", "m1", dense, expert),
        ("m0.dst", "M_TO_N_COMM", "destination", "m0", dense, expert),
    ]
    dense_dag = _routed_actor_dag(
        dense,
        dense_specs,
        ["m0.src", "m1.src"],
    )
    expert_dag = _routed_actor_dag(
        expert,
        expert_specs,
        ["m1.dst", "m0.dst"],
    )

    with pytest.raises(ValueError, match="call-sequence mismatch"):
        _validate_routed_collective_sequences([dense_dag, expert_dag])


def test_routed_collective_validation_rejects_endpoint_metadata_mismatch() -> None:
    dense = [0, 2]
    expert = [1, 3]
    dense_dag = _routed_actor_dag(
        dense,
        [("src", "M_TO_N_COMM", "source", "m0", dense, expert)],
        ["src"],
    )
    expert_dag = _routed_actor_dag(
        expert,
        [("dst", "M_TO_N_COMM", "destination", "m0", dense, expert)],
        ["dst"],
    )
    expert_dag.nodes["dst"].node_meta["a2a_tensor_idx"] = 1
    for node in (dense_dag.nodes["src"], expert_dag.nodes["dst"]):
        node.node_meta["expert_input_arity"] = 3
        node.node_meta["routing_counts_idx"] = 2

    with pytest.raises(ValueError, match="endpoints disagree on a2a_tensor_idx"):
        _validate_routed_collective_sequences([dense_dag, expert_dag])


def test_routed_collective_validation_rejects_state_key_reused_across_mbs() -> None:
    dense = [0, 2]
    expert = [1, 3]
    dense_dag = _routed_actor_dag(
        dense,
        [
            ("src0", "M_TO_N_COMM", "source", "m0", dense, expert),
            ("src1", "M_TO_N_COMM", "source", "m1", dense, expert),
        ],
        ["src0", "src1"],
    )
    expert_dag = _routed_actor_dag(
        expert,
        [
            ("dst0", "M_TO_N_COMM", "destination", "m0", dense, expert),
            ("dst1", "M_TO_N_COMM", "destination", "m1", dense, expert),
        ],
        ["dst0", "dst1"],
    )
    for dag in (dense_dag, expert_dag):
        for uid, node in dag.nodes.items():
            node.tag["MB"] = 0 if uid.endswith("0") else 1
            node.node_meta["expert_fwd_uid"] = "expert.shared"

    with pytest.raises(ValueError, match="reused across microbatches"):
        _validate_routed_collective_sequences([dense_dag, expert_dag])


def _synthetic_afd_training_dag(
    n_layers: int,
    *,
    n_pp_stages: int = 2,
) -> TrainingDAG:
    dag = TrainingDAG()
    fwd_nodes: list[TrainingDAGNode] = []

    def add_fwd(uid: str, tag: dict[str, int]) -> TrainingDAGNode:
        node = TrainingDAGNode(
            uid=uid,
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag={**tag, "PASS": "F"},
            device=None,
            stream="default_stream",
            node_meta={
                "bucket_key": uid,
                "input_names": ["tokens", "routing_counts"],
                "output_names": ["tokens", "routing_counts"],
                "tensor_output_indices": [0, 1],
                "differentiable_output_indices": [0],
            },
        )
        dag.add_node(node)
        fwd_nodes.append(node)
        return node

    add_fwd("input", {"PP": 0})
    for layer in range(n_layers):
        pp = layer * n_pp_stages // n_layers
        dense = add_fwd(f"dense{layer}", {"PP": pp, "ATTN": layer})
        dense.node_meta["a2a_boundary_after"] = {
            "tensor_idx": 0,
            "routing_counts_idx": 1,
            "routing_counts_schema": {
                "shape": (1, 8),
                "dtype": "torch.int64",
                "requires_grad": False,
            },
        }
        routed = add_fwd(f"routed{layer}", {"PP": pp, "FFN": layer})
        routed.node_meta["output_names"] = ["expert_output"]
        routed.node_meta["tensor_output_indices"] = [0]
        routed.node_meta["differentiable_output_indices"] = [0]
        routed.node_meta["a2a_boundary_after"] = {"tensor_idx": 0}
    add_fwd(
        "dense_tail",
        {"PP": n_pp_stages - 1, "ATTN": n_layers},
    )
    add_fwd("output", {"PP": n_pp_stages - 1})

    for src, dst in zip(fwd_nodes, fwd_nodes[1:]):
        dag.add_edge(TrainingDAGEdge(src.uid, dst.uid, "data", "tokens"))

    bwd_nodes: dict[str, TrainingDAGNode] = {}
    for fwd in fwd_nodes:
        bwd = TrainingDAGNode(
            uid=f"{fwd.uid}.bwd",
            node_kind="COMPUTE",
            compute_subkind="BWD",
            tag={**fwd.tag, "PASS": "B"},
            device=None,
            stream="default_stream",
            node_meta={"fwd_uid": fwd.uid, "bucket_key": fwd.uid},
        )
        dag.add_node(bwd)
        bwd_nodes[fwd.uid] = bwd
    for src, dst in zip(fwd_nodes, fwd_nodes[1:]):
        dag.add_edge(
            TrainingDAGEdge(
                bwd_nodes[dst.uid].uid,
                bwd_nodes[src.uid].uid,
                "data",
                "tokens",
            )
        )
    # Preserve the ordinary forward-to-backward dependency without adding an
    # optimizer sink, which is irrelevant to routed communication lowering.
    dag.add_edge(
        TrainingDAGEdge(
            fwd_nodes[-1].uid,
            bwd_nodes[fwd_nodes[-1].uid].uid,
            "data",
            "tokens",
        )
    )
    return dag


def test_full_afd_directive_path_builds_connected_sequence_safe_actor_dags() -> None:
    placement_path = (
        Path(__file__).parents[1]
        / "examples"
        / "base-schedules"
        / "afd_placement_pp2.json"
    )
    base = json.loads(placement_path.read_text())
    places = [directive for directive in base if directive["op"] == "place"]
    order = build_afpipe_baseline_schedule(
        places,
        n_mbs=2,
        n_layers=2,
        expected_actor_groups=2,
    )
    schedule = [
        *base,
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
        *order,
    ]
    dag = _synthetic_afd_training_dag(n_layers=2)

    apply_schedule_directives(dag, schedule)
    actor_dags = _split_global_training_dag_by_pp_rank(dag)
    assert len(actor_dags) == 2
    assert {
        tuple(sorted(node.device))
        for actor_dag in actor_dags
        for node in actor_dag.nodes.values()
        if node.device is not None
    } == {(0, 2, 4, 5), (1, 3)}
    assert not any(
        node.node_kind in {"SEND_COMM", "RECV_COMM", "A2A_COMM", "SCALE_GRADS"}
        for actor_dag in actor_dags
        for node in actor_dag.nodes.values()
    )
    for actor_dag in actor_dags:
        _resolve_default_stream_order(actor_dag)
    _order_communication_transactions(actor_dags)
    for actor_dag in actor_dags:
        resolve_total_order_per_stream(actor_dag)
    _validate_routed_collective_sequences(actor_dags)
    _validate_dense_state_sequences(actor_dags)
    _validate_p2p_sequences(actor_dags)

    routed = [
        node
        for actor_dag in actor_dags
        for node in actor_dag.nodes.values()
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
    ]
    assert len(routed) == 32
    by_comm_id: dict[str, list[TrainingDAGNode]] = {}
    for node in routed:
        by_comm_id.setdefault(node.node_meta["comm_id"], []).append(node)
    assert len(by_comm_id) == 16
    assert all(
        {node.node_meta["role"] for node in pair} == {"source", "destination"}
        for pair in by_comm_id.values()
    )


def test_pp4_routes_combine_and_state_to_the_actual_dense_consumer() -> None:
    placement_path = (
        Path(__file__).parents[1]
        / "examples"
        / "base-schedules"
        / "afd_placement_pp4.json"
    )
    base = json.loads(placement_path.read_text())
    places = [directive for directive in base if directive["op"] == "place"]
    schedule = [
        *base,
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
        *build_afpipe_baseline_schedule(
            places,
            n_mbs=1,
            n_layers=4,
            expected_actor_groups=4,
        ),
    ]
    dag = _synthetic_afd_training_dag(n_layers=4, n_pp_stages=4)

    apply_schedule_directives(dag, schedule)
    actor_dags = _split_global_training_dag_by_pp_rank(dag)
    assert len(actor_dags) == 4
    for actor_dag in actor_dags:
        _resolve_default_stream_order(actor_dag)
    _order_communication_transactions(actor_dags)
    for actor_dag in actor_dags:
        resolve_total_order_per_stream(actor_dag)
    _validate_routed_collective_sequences(actor_dags)
    _validate_dense_state_sequences(actor_dags)
    _validate_p2p_sequences(actor_dags)

    routed = [
        node
        for actor_dag in actor_dags
        for node in actor_dag.nodes.values()
        if node.node_kind in {"M_TO_N_COMM", "N_TO_M_COMM"}
    ]
    state_transfers = [
        node
        for actor_dag in actor_dags
        for node in actor_dag.nodes.values()
        if node.node_kind in {"SEND_COMM", "RECV_COMM"}
        and node.node_meta.get("dense_state_transfer")
    ]
    assert len(routed) == 32
    assert len(state_transfers) == 16
    assert {
        (node.node_meta["direction"], node.tag["PASS"])
        for node in state_transfers
    } == {("forward", "F"), ("backward", "B")}
    assert {node.node_kind for node in state_transfers} == {
        "SEND_COMM",
        "RECV_COMM",
    }
    assert all(node.stream == "dense_state_stream" for node in state_transfers)
    assert {
        node.node_meta["comm_group_kind"] for node in state_transfers
    } == {"dense_state_lo_hi", "dense_state_hi_lo"}
    assert all(
        node.node_meta["transfer_mode"] == "ordinal_one_to_one"
        for node in state_transfers
    )
    for actor_dag in actor_dags:
        for node in actor_dag.nodes.values():
            if not node.node_meta.get("dense_state_transfer"):
                continue
            anchor_field = (
                "source_compute_uid"
                if node.node_kind == "SEND_COMM"
                else "destination_compute_uid"
            )
            anchor_uid = node.node_meta[anchor_field]
            assert actor_dag.nodes[anchor_uid].node_kind == "COMPUTE"
            if node.node_meta["direction"] == "backward":
                assert actor_dag.nodes[anchor_uid].compute_subkind in {
                    "BWD",
                    "BWD_I",
                }
    assert all(node.node_meta["dense_state_arity"] == 0 for node in state_transfers)
    assert all(node.node_meta["model_tensor_indices"] == [] for node in state_transfers)
    assert all(
        node.node_meta["differentiable_model_indices"] == []
        for node in state_transfers
    )
    assert all(
        node.node_meta["carries_routing_layout"] is True
        for node in state_transfers
    )
    assert all(
        node.node_meta["rank_pairs"]
        == [
            [source_rank, destination_rank]
            for source_rank, destination_rank in zip(
                node.node_meta["source_devices"],
                node.node_meta["destination_devices"],
            )
        ]
        for node in state_transfers
    )

    for layer in range(4):
        expert_fwd_uid = f"routed{layer}"
        dispatch_source = next(
            node
            for node in routed
            if node.node_kind == "M_TO_N_COMM"
            and node.tag["PASS"] == "F"
            and node.node_meta["role"] == "source"
            and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
        )
        combine_destination = next(
            node
            for node in routed
            if node.node_kind == "N_TO_M_COMM"
            and node.tag["PASS"] == "F"
            and node.node_meta["role"] == "destination"
            and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
        )
        forward_state_source = next(
            node
            for node in state_transfers
            if node.node_kind == "SEND_COMM"
            and node.tag["PASS"] == "F"
            and node.node_meta["role"] == "source"
            and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
        )
        forward_state_destination = next(
            node
            for node in state_transfers
            if node.node_kind == "RECV_COMM"
            and node.tag["PASS"] == "F"
            and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
        )
        assert combine_destination.device != dispatch_source.device
        assert forward_state_source.device == dispatch_source.device
        assert forward_state_destination.device == combine_destination.device

        backward_dispatch_source = next(
            node
            for node in routed
            if node.node_kind == "M_TO_N_COMM"
            and node.tag["PASS"] == "B"
            and node.node_meta["role"] == "source"
            and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
        )
        backward_state_destination = next(
            node
            for node in state_transfers
            if node.node_kind == "RECV_COMM"
            and node.tag["PASS"] == "B"
            and node.node_meta["expert_fwd_uid"] == expert_fwd_uid
        )
        assert backward_state_destination.device == dispatch_source.device
        assert backward_dispatch_source.device == combine_destination.device
