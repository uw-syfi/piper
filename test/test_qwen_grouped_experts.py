import torch
import torch.nn as nn

from examples.models.qwen3 import AnnotatedMoE, create_qwen3_config
from src import directives
from src.dag import build_training_dag
from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state, annotate


class _AnnotatedQwenMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = create_qwen3_config("9M")
        self.moe = AnnotatedMoE(
            config.moe_args,
            dim=config.dim,
            hidden_dim=config.moe_inter_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            return self.moe(x)


def test_qwen_uses_torchtitan_grouped_experts_and_exposes_routing_counts() -> None:
    captured = {}

    def backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    torch._dynamo.reset()
    _reset_annotation_state()
    with torch.device("meta"):
        model = _AnnotatedQwenMoE().to(torch.bfloat16)
        x = torch.empty((1, 8, 256), dtype=torch.bfloat16, device="meta")
        output = torch.compile(model, backend=backend, fullgraph=True)(x)

    assert tuple(output.shape) == (1, 8, 256)
    assert model.moe.experts.__class__.__name__ == "GroupedExperts"

    _, segments = split_gm_by_annotations(captured["gm"])
    assert [segment.tag for segment in segments] == [
        {"PP": 0},
        {"PP": 0, "EP": 0},
        {"PP": 0},
    ]
    incoming = segments[0].a2a_boundary_after
    assert incoming is not None
    assert incoming["routing_counts_idx"] != incoming["tensor_idx"]
    expert_graphargs = segments[1].graphargs
    routed_tokens = expert_graphargs[incoming["tensor_idx"]]
    routing_counts = expert_graphargs[incoming["routing_counts_idx"]]
    assert routed_tokens.ndim == 2
    assert routing_counts.ndim == 1

    expert_nodes = list(segments[1].gm.graph.nodes)
    assert sum(
        node.op == "call_function" and node.target is torch._grouped_mm
        for node in expert_nodes
    ) == 3
    assert not any(
        "triton_kernel_wrapper_mutation" in str(node.target)
        for node in expert_nodes
    )
    assert not any(
        node.op == "call_method" and node.target == "new_empty"
        for node in expert_nodes
    )

    ep1_dag = build_training_dag(segments)
    for node in ep1_dag.nodes.values():
        if node.node_kind == "COMPUTE":
            node.device = [0]
    directives._insert_shard_a2a_comm_nodes(
        ep1_dag,
        [{"PP": 0, "EP": 0}],
        [0],
    )
    ep1_expert_nodes = list(
        ep1_dag.nodes["s0.seg1"].node_meta["gm"].graph.nodes
    )
    assert not any(
        node.op == "call_method" and node.target == "new_empty"
        for node in ep1_expert_nodes
    )
    assert all(
        "ragged" not in node.node_meta
        for node in ep1_dag.nodes.values()
        if node.node_kind == "A2A_COMM"
    )
    assert sum(
        node.node_meta.get("routing_counts_idx") is not None
        for node in ep1_dag.nodes.values()
        if node.node_kind == "A2A_COMM"
    ) == 1

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
    assert not any(
        node.op == "call_method" and node.target == "new_empty"
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
