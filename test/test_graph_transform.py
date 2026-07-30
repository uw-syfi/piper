import pytest
import torch
import torch.fx as fx
import torch.nn as nn

from src.piper import (
    _reset_annotation_state,
    annotate,
    mark_combine_payload,
    mark_dispatch_counts,
    mark_dispatch_payload,
)
from src.fx import (
    EP_COMBINE_PAYLOAD,
    EP_DISPATCH_COUNTS,
    EP_DISPATCH_PAYLOAD,
    PIPER_EP_BOUNDARY_META_KEY,
    split_gm_by_annotations,
)


class _AnnotatedNet(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)
        self.fc3 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with annotate("PP"):
            x = torch.relu(self.fc1(x))
            counts = torch.ones((1,), dtype=torch.int64, device=x.device)
            x = mark_dispatch_payload(x)
            counts = mark_dispatch_counts(counts)
            with annotate("EP"):
                x = torch.relu(self.fc2(x))
                scale = counts.sum().to(x.dtype) * 0 + 1
                x = mark_combine_payload(x * scale)
        with annotate("PP"):
            return self.fc3(x)


def _capture_gm(model: nn.Module, *inputs: torch.Tensor):
    captured: dict = {}

    def _backend(gm, example_inputs):
        captured["gm"] = gm
        captured["inputs"] = list(example_inputs)
        return gm.forward

    _reset_annotation_state()
    compiled = torch.compile(model, backend=_backend, fullgraph=True)
    with torch.no_grad():
        ref = compiled(*inputs).detach().clone()

    return captured["gm"], captured["inputs"], ref


def _placeholder_values(gm: fx.GraphModule, example_inputs: list) -> dict[str, object]:
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    return {node.name: value for node, value in zip(placeholders, example_inputs)}


def _run_segments(segments, placeholder_values: dict[str, object]) -> torch.Tensor:
    prev_output = None

    for segment in segments:
        ph_nodes = [node for node in segment.gm.graph.nodes if node.op == "placeholder"]
        args = []
        activation_inputs = (
            ()
            if prev_output is None
            else prev_output
            if isinstance(prev_output, tuple)
            else (prev_output,)
        )
        activation_iter = iter(activation_inputs)

        for node in ph_nodes:
            if node.name in placeholder_values:
                args.append(placeholder_values[node.name])
            else:
                args.append(next(activation_iter))

        with torch.no_grad():
            prev_output = segment.gm(*args)

    if isinstance(prev_output, (tuple, list)) and len(prev_output) == 1:
        prev_output = prev_output[0]
    return prev_output


def test_split_gm_by_annotations_handles_nested_annotation_segments() -> None:
    torch.manual_seed(0)
    model = _AnnotatedNet()
    x = torch.randn(2, 8)

    gm, example_inputs, ref = _capture_gm(model, x)
    _, segments = split_gm_by_annotations(gm)

    assert [segment.tag for segment in segments] == [
        {"PP": 0},
        {"PP": 0, "EP": 0},
        {"PP": 1},
    ]
    assert [segment.stage_id for segment in segments] == [0, 0, 1]
    assert [segment.a2a_boundary_after["from_tag"] for segment in segments[:-1]] == [
        {"PP": 0},
        {"PP": 0, "EP": 0},
    ]
    assert [segment.a2a_boundary_after["to_tag"] for segment in segments[:-1]] == [
        {"PP": 0, "EP": 0},
        {"PP": 1},
    ]

    result = _run_segments(segments, _placeholder_values(gm, example_inputs))

    assert torch.allclose(ref, result, atol=1e-6)


def test_split_gm_by_annotations_ignores_non_piper_custom_metadata() -> None:
    gm = fx.symbolic_trace(nn.Sequential(nn.ReLU()))
    relu = next(node for node in gm.graph.nodes if node.op == "call_module")
    relu.meta["custom"] = {"name": "PP", "index": 0}

    _, segments = split_gm_by_annotations(gm)

    assert segments == []


def test_split_gm_by_annotations_rejects_mixed_annotated_and_unannotated_compute() -> None:
    torch.manual_seed(0)
    gm, _, _ = _capture_gm(_AnnotatedNet(), torch.randn(2, 8))
    first_compute = next(
        node
        for node in gm.graph.nodes
        if node.op not in ("placeholder", "get_attr", "output")
    )
    first_compute.meta.pop("custom", None)

    with pytest.raises(ValueError, match="without Piper annotations"):
        split_gm_by_annotations(gm)


class _AnnotatedExpertNet(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        with annotate("PP"):
            x = x + 1
            x = mark_dispatch_payload(x)
            num_tokens_per_expert = mark_dispatch_counts(
                num_tokens_per_expert
            )
            with annotate("EP"):
                scale = num_tokens_per_expert.sum().to(x.dtype) * 0 + 1
                x = x * scale
                x = mark_combine_payload(x)
        with annotate("PP"):
            return x - 1


def test_split_gm_uses_explicit_ep_tensor_boundaries() -> None:
    x = torch.randn(8, 4)
    counts = torch.tensor([1, 3, 2, 2], dtype=torch.int64)
    gm, example_inputs, ref = _capture_gm(_AnnotatedExpertNet(), x, counts)

    _, segments = split_gm_by_annotations(gm)

    boundary = segments[0].a2a_boundary_after
    assert boundary is not None
    assert boundary["routing_counts_idx"] != boundary["tensor_idx"]

    first_output = segments[0].gm(*example_inputs)
    assert isinstance(first_output, tuple)
    routed_counts = first_output[boundary["routing_counts_idx"]]
    assert routed_counts.dtype == torch.int64
    assert tuple(routed_counts.shape) == (4,)

    result = _run_segments(segments, _placeholder_values(gm, example_inputs))
    assert torch.allclose(ref, result, atol=1e-6)


@pytest.mark.parametrize(
    ("role", "api_name"),
    [
        (EP_DISPATCH_PAYLOAD, "mark_dispatch_payload"),
        (EP_DISPATCH_COUNTS, "mark_dispatch_counts"),
        (EP_COMBINE_PAYLOAD, "mark_combine_payload"),
    ],
)
def test_split_gm_requires_each_explicit_ep_tensor_boundary(
    role: str,
    api_name: str,
) -> None:
    gm, _, _ = _capture_gm(
        _AnnotatedExpertNet(),
        torch.randn(8, 4),
        torch.tensor([1, 3, 2, 2], dtype=torch.int64),
    )
    marker = next(
        node
        for node in gm.graph.nodes
        if (node.meta.get("custom") or {}).get(PIPER_EP_BOUNDARY_META_KEY)
        == role
    )
    marker.meta["custom"].pop(PIPER_EP_BOUNDARY_META_KEY)

    with pytest.raises(ValueError, match=api_name):
        split_gm_by_annotations(gm)


class _DuplicateDispatchPayloadNet(_AnnotatedExpertNet):
    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        with annotate("PP"):
            x = x + 1
            first_payload = mark_dispatch_payload(x)
            second_payload = mark_dispatch_payload(x)
            num_tokens_per_expert = mark_dispatch_counts(
                num_tokens_per_expert
            )
            with annotate("EP"):
                scale = num_tokens_per_expert.sum().to(x.dtype) * 0 + 1
                x = (first_payload + second_payload) * scale
                x = mark_combine_payload(x)
        with annotate("PP"):
            return x - 1


def test_split_gm_rejects_duplicate_ep_tensor_boundary_role() -> None:
    gm, _, _ = _capture_gm(
        _DuplicateDispatchPayloadNet(),
        torch.randn(8, 4),
        torch.tensor([1, 3, 2, 2], dtype=torch.int64),
    )

    with pytest.raises(
        ValueError,
        match=r"exactly one mark_dispatch_payload\(\.\.\.\) tensor, found 2",
    ):
        split_gm_by_annotations(gm)
