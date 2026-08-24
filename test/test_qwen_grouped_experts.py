import torch
import torch.nn as nn

from examples.models.qwen3 import AFDQwen3Model, AnnotatedMoE, create_qwen3_config
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


class _ToyExperts(nn.Module):
    def forward(
        self,
        routed_input: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        scale = num_tokens_per_expert.sum().to(routed_input.dtype) * 0 + 1
        return routed_input * scale


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


def test_qwen_preserves_torchtitan_grouped_expert_graph_and_exposes_counts(
    monkeypatch,
) -> None:
    captured = {}

    # The CPU PyTorch wheel reports no active Triton-capable device, which
    # prevents Dynamo from classifying a Triton JITFunction even for a pure
    # meta-tensor capture. The compiler package is installed and this test
    # never executes the kernel, so enable only Dynamo's capture-time path.
    monkeypatch.setattr(torch.utils._triton, "has_triton", lambda: True)

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
    assert routing_counts.ndim == 2
    assert tuple(routing_counts.shape) == (1, 8)

    expert_nodes = list(segments[1].gm.graph.nodes)
    assert sum(
        node.op == "call_function" and node.target is torch._grouped_mm
        for node in expert_nodes
    ) == 3
    assert any(
        "triton_kernel_wrapper_mutation" in str(node.target)
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
    assert sum(
        node.op == "call_function" and node.target is torch._grouped_mm
        for node in ep1_expert_nodes
    ) == 3
    assert any(
        "triton_kernel_wrapper_mutation" in str(node.target)
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


def test_afd_qwen_assigns_only_routed_experts_to_ffn_regions() -> None:
    captured = {}

    def backend(gm, example_inputs):
        captured["gm"] = gm
        return gm.forward

    torch._dynamo.reset()
    _reset_annotation_state()
    with torch.device("meta"):
        config = create_qwen3_config("9M")
        model = AFDQwen3Model(config, num_stages=2).to(torch.bfloat16)
        for layer in model.layers.values():
            layer.moe.experts = _ToyExperts()
        tokens = torch.zeros((1, 8), dtype=torch.int64, device="meta")
        output = torch.compile(model, backend=backend, fullgraph=True)(tokens)

    assert tuple(output.shape) == (1, 8, config.vocab_size)
    _, segments = split_gm_by_annotations(captured["gm"])
    assert [segment.tag for segment in segments] == [
        {"PP": 0},
        {"PP": 0, "ATTN": 0},
        {"PP": 0, "FFN": 0},
        {"PP": 0, "ATTN": 1},
        {"PP": 0, "ATTN": 1},
        {"PP": 0, "FFN": 1},
        {"PP": 0, "ATTN": 2},
        {"PP": 1, "ATTN": 2},
        {"PP": 1, "FFN": 2},
        {"PP": 1, "ATTN": 3},
        {"PP": 1, "ATTN": 3},
        {"PP": 1, "FFN": 3},
        {"PP": 1, "ATTN": 4},
        {"PP": 1},
    ]
    assert [
        (segment.a2a_boundary_after or {}).get("routed_boundary")
        for segment in segments
    ] == [
        None,
        "dispatch",
        "combine",
        None,
        "dispatch",
        "combine",
        None,
        "dispatch",
        "combine",
        None,
        "dispatch",
        "combine",
        None,
        None,
    ]
