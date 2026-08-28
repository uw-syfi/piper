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

"""Qwen model capture and routed-expert layout tests."""

import torch
import torch.nn as nn

from examples.models.qwen3 import AnnotatedMoE, create_qwen3_config
from src import directives
from src.dag import build_training_dag
from src.fx import (
    _deserialize_graphmodule,
    _serialize_graphmodule,
    split_gm_by_annotations,
)
from src.piper import _reset_annotation_state, annotate


class _AnnotatedQwenMoE(nn.Module):
    def __init__(self, *, num_local_experts: int | None = None) -> None:
        super().__init__()
        config = create_qwen3_config("9M")
        self.moe = AnnotatedMoE(
            config.moe_args,
            dim=config.dim,
            hidden_dim=config.moe_inter_dim,
            num_local_experts=num_local_experts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            return self.moe(x)


def test_model_side_regrouping_is_stable_and_exactly_reversible() -> None:
    counts = torch.tensor(
        [[2, 0, 1], [1, 2, 0]],
        dtype=torch.int64,
    )
    source_major = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    expert_major, local_counts, transport_order = (
        AnnotatedMoE._to_expert_local_order(source_major, counts)
    )

    assert local_counts.tolist() == [3, 2, 1]
    assert expert_major.flatten().tolist() == [0, 1, 3, 4, 5, 2]
    restored = (expert_major + 100).index_select(0, transport_order)
    assert restored.flatten().tolist() == [100, 101, 102, 103, 104, 105]


def test_model_side_regrouping_uses_runtime_destination_count_shape() -> None:
    """Capture source counts, then execute with a different routed shape."""
    captured = {}

    class _Regroup(nn.Module):
        def forward(self, payload, counts):
            return AnnotatedMoE._to_expert_local_order(payload, counts)

    def backend(gm, example_inputs):
        del example_inputs
        captured["gm"] = gm
        return gm.forward

    source_payload = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    source_counts = torch.tensor([[1, 1, 1, 1]], dtype=torch.int64)
    torch._dynamo.reset()
    torch.compile(_Regroup(), backend=backend, fullgraph=True)(
        source_payload,
        source_counts,
    )

    # Two source ranks and two local routes. Source 0 supplied route 1 rows;
    # source 1 supplied route 0 rows, so expert-major order swaps the halves.
    destination_counts = torch.tensor([[0, 2], [2, 0]], dtype=torch.int64)
    restored_gm = _deserialize_graphmodule(
        _serialize_graphmodule(captured["gm"])
    )
    expert_major, local_counts, transport_order = restored_gm(
        source_payload,
        destination_counts,
    )
    assert expert_major.flatten().tolist() == [2, 3, 0, 1]
    assert local_counts.tolist() == [2, 2]
    assert transport_order.tolist() == [2, 3, 0, 1]


def test_qwen_captures_local_experts_with_dynamic_routed_rows(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(torch.utils._triton, "has_triton", lambda: True)

    def backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    torch._dynamo.reset()
    _reset_annotation_state()
    with torch.device("meta"):
        model = _AnnotatedQwenMoE(num_local_experts=4).to(torch.bfloat16)
        x = torch.empty((1, 8, 256), dtype=torch.bfloat16, device="meta")
        torch._dynamo.mark_dynamic(x, 1)
        torch.compile(model, backend=backend, fullgraph=True)(x)

    _, segments = split_gm_by_annotations(captured["gm"])
    expert = segments[1]
    incoming = segments[0].a2a_boundary_after
    assert incoming is not None
    assert tuple(
        expert.graphargs[incoming["routing_counts_idx"]].shape
    ) == (1, 8)
    assert {
        int(expert.graphargs[index].shape[0]) for index in expert.param_idxs
    } == {4}

    size_nodes = [
        node
        for node in expert.gm.graph.nodes
        if node.op == "call_method" and node.target == "size"
    ]
    assert size_nodes
    assert any(
        isinstance(node.args[0], torch.fx.Node)
        and "routed_input" in node.args[0].name
        for node in size_nodes
    )

    ep2_dag = build_training_dag(segments)
    for node in ep2_dag.nodes.values():
        if node.node_kind == "COMPUTE":
            node.device = [0, 1]
    directives._insert_shard_a2a_comm_nodes(
        ep2_dag,
        [{"PP": 0, "EP": 0}],
        [0, 1],
    )
    ep2_expert_nodes = list(
        ep2_dag.nodes["s0.seg1"].node_meta["gm"].graph.nodes
    )
    assert sum(
        node.op == "call_function" and node.target is torch._grouped_mm
        for node in ep2_expert_nodes
    ) == 3
    assert any(
        "triton_kernel_wrapper_mutation" in str(node.target)
        for node in ep2_expert_nodes
    )
    assert all(
        "ragged" not in node.node_meta
        for node in ep2_dag.nodes.values()
        if node.node_kind == "A2A_COMM"
    )
    assert sum(
        node.node_meta.get("routing_counts_idx") is not None
        for node in ep2_dag.nodes.values()
        if node.node_kind == "A2A_COMM"
    ) == 1


