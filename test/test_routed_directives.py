# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Same-group and cross-group routed directive tests."""

import pytest
import torch

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.directives import (
    _apply_split_directive,
    _insert_cross_group_routed_comm_nodes,
    _insert_shard_a2a_comm_nodes,
    _validate_route_owned_parameters,
    apply_schedule_directives,
)
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


