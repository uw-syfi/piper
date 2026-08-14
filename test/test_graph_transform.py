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
            counts = torch.full(
                (1, 1), x.shape[0], dtype=torch.int64, device=x.device
            )
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

    torch._dynamo.reset()
    _reset_annotation_state()
    training_inputs = tuple(
        value.detach().clone().requires_grad_(True)
        if isinstance(value, torch.Tensor)
        and (torch.is_floating_point(value) or torch.is_complex(value))
        else value
        for value in inputs
    )
    compiled = torch.compile(model, backend=_backend, fullgraph=True)
    ref = compiled(*training_inputs).detach().clone()

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
    assert segments[0].input_idxs == [0]
    runtime_placeholder = next(
        node
        for node in gm.graph.nodes
        if node.op == "placeholder" and not node.name.startswith("l_self_")
    )
    assert segments[0].placeholders[0].name == runtime_placeholder.name
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


class _BatchedMatmulExpertNet(nn.Module):
    """A routed expert implementation unrelated to TorchTitan/grouped MM."""

    def __init__(self, num_experts: int = 3, hidden: int = 4) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.randn(num_experts, hidden, hidden)
        )

    def forward(self, payload: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        with annotate("DENSE_USER_TAG"):
            payload = mark_dispatch_payload(payload)
            counts = mark_dispatch_counts(counts)
            with annotate("ARBITRARY_ROUTED_TAG"):
                local_counts = counts.sum(dim=0)
                row_experts = torch.repeat_interleave(
                    torch.arange(
                        self.num_experts,
                        device=payload.device,
                    ),
                    local_counts,
                    output_size=payload.shape[0],
                )
                row_weights = self.weight.index_select(0, row_experts)
                output = torch.bmm(
                    payload.unsqueeze(1),
                    row_weights,
                ).squeeze(1)
                output = mark_combine_payload(output)
        with annotate("DENSE_USER_TAG"):
            return output + 0


def _unwrap_single_output(value):
    while isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return value


def _run_segments_with_grad(segments, placeholder_values):
    prev_output = None
    for segment in segments:
        ph_nodes = [
            node for node in segment.gm.graph.nodes if node.op == "placeholder"
        ]
        activation_inputs = (
            ()
            if prev_output is None
            else prev_output
            if isinstance(prev_output, tuple)
            else (prev_output,)
        )
        activation_iter = iter(activation_inputs)
        args = [
            placeholder_values[node.name]
            if node.name in placeholder_values
            else next(activation_iter)
            for node in ph_nodes
        ]
        prev_output = segment.gm(*args)
    return _unwrap_single_output(prev_output)


def test_structurally_different_bmm_experts_preserve_forward_and_backward() -> None:
    """Splitting is operator-opaque for a non-TorchTitan expert graph."""
    gm, example_inputs, _ = _capture_gm(
        _BatchedMatmulExpertNet(),
        torch.randn(3, 4),
        torch.tensor([[2, 0, 1]], dtype=torch.int64),
    )
    _, segments = split_gm_by_annotations(gm)
    assert [segment.tag for segment in segments] == [
        {"DENSE_USER_TAG": 0},
        {"DENSE_USER_TAG": 0, "ARBITRARY_ROUTED_TAG": 0},
        {"DENSE_USER_TAG": 1},
    ]
    routed_nodes = list(segments[1].gm.graph.nodes)
    assert any(
        node.op == "call_function" and node.target is torch.bmm
        for node in routed_nodes
    )
    assert all(node.target is not torch._grouped_mm for node in routed_nodes)

    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]

    def clone_inputs():
        result = []
        for value in example_inputs:
            cloned = value.detach().clone()
            if torch.is_floating_point(cloned) or torch.is_complex(cloned):
                cloned.requires_grad_(True)
            result.append(cloned)
        return result

    reference_inputs = clone_inputs()
    split_inputs = clone_inputs()
    reference = _unwrap_single_output(gm(*reference_inputs))
    actual = _run_segments_with_grad(
        segments,
        {
            node.name: value
            for node, value in zip(placeholders, split_inputs)
        },
    )
    assert torch.equal(actual, reference)

    reference.sum().backward()
    actual.sum().backward()
    for reference_input, split_input in zip(reference_inputs, split_inputs):
        if reference_input.grad is None:
            assert split_input.grad is None
        else:
            assert torch.equal(split_input.grad, reference_input.grad)


def test_split_gm_uses_explicit_ep_tensor_boundaries() -> None:
    x = torch.randn(8, 4)
    counts = torch.tensor([[1, 3, 2, 2]], dtype=torch.int64)
    gm, example_inputs, ref = _capture_gm(_AnnotatedExpertNet(), x, counts)

    _, segments = split_gm_by_annotations(gm)

    boundary = segments[0].a2a_boundary_after
    assert boundary is not None
    assert boundary["routing_counts_idx"] != boundary["tensor_idx"]
    assert boundary["payload_schema"] == {
        "shape": (8, 4),
        "dtype": "torch.float32",
        "requires_grad": True,
    }
    assert boundary["routing_counts_schema"] == {
        "shape": (1, 4),
        "dtype": "torch.int64",
        "requires_grad": False,
    }
    combine_boundary = segments[1].a2a_boundary_after
    assert combine_boundary is not None
    assert combine_boundary["payload_schema"] == boundary["payload_schema"]

    first_output = segments[0].gm(*example_inputs)
    assert isinstance(first_output, tuple)
    routed_counts = first_output[boundary["routing_counts_idx"]]
    assert routed_counts.dtype == torch.int64
    assert tuple(routed_counts.shape) == (1, 4)

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
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
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
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )

    with pytest.raises(
        ValueError,
        match=r"exactly one mark_dispatch_payload\(\.\.\.\) tensor, found 2",
    ):
        split_gm_by_annotations(gm)


class _GenericTagRoutedNet(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        with annotate("DENSE"):
            x = mark_dispatch_payload(x + 1)
            counts = mark_dispatch_counts(counts)
            with annotate("ANY_USER_ROUTE_TAG"):
                scale = counts.sum().to(x.dtype) * 0 + 1
                x = mark_combine_payload(x * scale)
        with annotate("DENSE"):
            return x - 1


def test_routed_marker_validation_is_independent_of_annotation_tag_names() -> None:
    gm, _, _ = _capture_gm(
        _GenericTagRoutedNet(),
        torch.randn(8, 4),
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )

    _, segments = split_gm_by_annotations(gm)

    assert [segment.tag for segment in segments] == [
        {"DENSE": 0},
        {"DENSE": 0, "ANY_USER_ROUTE_TAG": 0},
        {"DENSE": 1},
    ]


def test_split_gm_rejects_non_matrix_routed_payload() -> None:
    gm, _, _ = _capture_gm(
        _AnnotatedExpertNet(),
        torch.randn(2, 4, 8),
        torch.tensor([[2, 2]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match=r"2-D \[tokens, hidden\] payload"):
        split_gm_by_annotations(gm)


@pytest.mark.parametrize(
    "counts",
    [
        torch.ones((4,), dtype=torch.int64),
        torch.ones((4,), dtype=torch.float32),
        torch.ones((1, 4), dtype=torch.float32),
        torch.ones((1, 4), dtype=torch.bool),
        torch.ones((1, 1, 4), dtype=torch.int64),
    ],
)
def test_split_gm_rejects_invalid_routed_count_schema(counts: torch.Tensor) -> None:
    gm, _, _ = _capture_gm(
        _AnnotatedExpertNet(),
        torch.randn(8, 4),
        counts,
    )

    with pytest.raises(ValueError, match="count tensor"):
        split_gm_by_annotations(gm)


class _ShapeChangingRoutedNet(nn.Module):
    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        with annotate("DENSE"):
            x = mark_dispatch_payload(x)
            counts = mark_dispatch_counts(counts)
            with annotate("ROUTED"):
                scale = counts.sum().to(x.dtype) * 0 + 1
                x = mark_combine_payload(x[:, :-1] * scale)
        with annotate("DENSE"):
            return x + 1


def test_split_gm_rejects_shape_changing_routed_region() -> None:
    gm, _, _ = _capture_gm(
        _ShapeChangingRoutedNet(),
        torch.randn(8, 4),
        torch.tensor([[2, 2, 2, 2]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="preserve the concrete payload shape"):
        split_gm_by_annotations(gm)


class _DtypeChangingRoutedNet(nn.Module):
    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        with annotate("DENSE"):
            x = mark_dispatch_payload(x)
            counts = mark_dispatch_counts(counts)
            with annotate("ROUTED"):
                scale = counts.sum().to(x.dtype) * 0 + 1
                x = mark_combine_payload((x * scale).to(torch.float64))
        with annotate("DENSE"):
            return x + 1


def test_split_gm_rejects_routed_payload_dtype_change() -> None:
    gm, _, _ = _capture_gm(
        _DtypeChangingRoutedNet(),
        torch.randn(8, 4),
        torch.tensor([[2, 2, 2, 2]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="preserve payload dtype"):
        split_gm_by_annotations(gm)


class _PayloadIgnoringRoutedNet(nn.Module):
    def forward(
        self,
        payload: torch.Tensor,
        other: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        with annotate("DENSE"):
            payload = mark_dispatch_payload(payload)
            counts = mark_dispatch_counts(counts)
            with annotate("ROUTED"):
                payload_passthrough = payload + 0
                scale = counts.sum().to(other.dtype) * 0 + 1
                combined = mark_combine_payload(other * scale)
        with annotate("DENSE"):
            return combined + payload_passthrough * 0


def test_split_gm_requires_combine_to_depend_on_dispatch_payload() -> None:
    gm, _, _ = _capture_gm(
        _PayloadIgnoringRoutedNet(),
        torch.randn(8, 4),
        torch.randn(8, 4),
        torch.tensor([[2, 2, 2, 2]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="must compute its combine payload from"):
        split_gm_by_annotations(gm)


class _MultiSegmentRoutedNet(nn.Module):
    def forward(self, x: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        with annotate("DENSE"):
            x = mark_dispatch_payload(x)
            counts = mark_dispatch_counts(counts)
        with annotate("ROUTED_PART_A"):
            x = x * (counts.sum().to(x.dtype) * 0 + 1)
        with annotate("ROUTED_PART_B"):
            x = mark_combine_payload(x + 1)
        with annotate("DENSE"):
            return x - 1


def test_split_gm_rejects_routed_region_spanning_annotation_segments() -> None:
    gm, _, _ = _capture_gm(
        _MultiSegmentRoutedNet(),
        torch.randn(8, 4),
        torch.tensor([[2, 2, 2, 2]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="Every routed region must have adjacent"):
        split_gm_by_annotations(gm)


def test_split_gm_rejects_marker_that_does_not_cross_a_boundary() -> None:
    gm, _, _ = _capture_gm(
        _AnnotatedExpertNet(),
        torch.randn(8, 4),
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )
    dispatch = next(
        node
        for node in gm.graph.nodes
        if (node.meta.get("custom") or {}).get(PIPER_EP_BOUNDARY_META_KEY)
        == EP_DISPATCH_PAYLOAD
    )
    assert isinstance(dispatch.args[0], fx.Node)
    dispatch.replace_all_uses_with(dispatch.args[0])
    gm.graph.lint()
    gm.recompile()

    with pytest.raises(ValueError, match="mark_dispatch_payload"):
        split_gm_by_annotations(gm)


class _ExtraRoutedInputNet(nn.Module):
    def forward(
        self,
        payload: torch.Tensor,
        auxiliary: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        with annotate("DENSE"):
            payload = mark_dispatch_payload(payload + 1)
            auxiliary = auxiliary + 2
            counts = mark_dispatch_counts(counts)
            with annotate("ROUTED"):
                scale = counts.sum().to(payload.dtype) * 0 + 1
                combined = mark_combine_payload(
                    payload * scale + auxiliary * 0
                )
        with annotate("DENSE"):
            return combined - 1


def test_split_gm_rejects_unmarked_dynamic_input_to_routed_region() -> None:
    gm, _, _ = _capture_gm(
        _ExtraRoutedInputNet(),
        torch.randn(8, 4),
        torch.randn(8, 4),
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="exactly.*dynamic activation inputs"):
        split_gm_by_annotations(gm)


class _ExtraRoutedOutputNet(nn.Module):
    def forward(self, payload: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        with annotate("DENSE"):
            payload = mark_dispatch_payload(payload)
            counts = mark_dispatch_counts(counts)
            with annotate("ROUTED"):
                scale = counts.sum().to(payload.dtype) * 0 + 1
                routed = payload * scale
                combined = mark_combine_payload(routed)
                extra = routed + 1
        with annotate("DENSE"):
            return combined + extra * 0


def test_split_gm_rejects_unmarked_dynamic_output_from_routed_region() -> None:
    gm, _, _ = _capture_gm(
        _ExtraRoutedOutputNet(),
        torch.randn(8, 4),
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="exactly.*dynamic activation output"):
        split_gm_by_annotations(gm)


class _RoutedRegionWithBypassNet(nn.Module):
    def forward(
        self,
        payload: torch.Tensor,
        bypass: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        with annotate("DENSE"):
            payload = mark_dispatch_payload(payload)
            bypass = bypass + 1
            counts = mark_dispatch_counts(counts)
            with annotate("ROUTED"):
                scale = counts.sum().to(payload.dtype) * 0 + 1
                combined = mark_combine_payload(payload * scale)
        with annotate("DENSE"):
            return combined + bypass


def test_split_gm_allows_dynamic_value_to_bypass_routed_region() -> None:
    gm, example_inputs, ref = _capture_gm(
        _RoutedRegionWithBypassNet(),
        torch.randn(8, 4),
        torch.randn(8, 4),
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )

    _, segments = split_gm_by_annotations(gm)
    result = _run_segments(segments, _placeholder_values(gm, example_inputs))

    assert len(segments) == 3
    assert torch.allclose(result, ref, atol=1e-6)


def test_split_gm_rejects_concrete_negative_route_counts_metadata() -> None:
    gm, _, _ = _capture_gm(
        _AnnotatedExpertNet(),
        torch.randn(8, 4),
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )
    counts_marker = next(
        node
        for node in gm.graph.nodes
        if (node.meta.get("custom") or {}).get(PIPER_EP_BOUNDARY_META_KEY)
        == EP_DISPATCH_COUNTS
    )
    counts_marker.meta["example_value"] = torch.tensor(
        [[1, -1, 4, 4]], dtype=torch.int64
    )

    with pytest.raises(ValueError, match="nonnegative route counts"):
        split_gm_by_annotations(gm)


def test_split_gm_rejects_concrete_payload_count_sum_mismatch() -> None:
    gm, _, _ = _capture_gm(
        _AnnotatedExpertNet(),
        torch.randn(8, 4),
        torch.tensor([[1, 3, 2, 2]], dtype=torch.int64),
    )
    counts_marker = next(
        node
        for node in gm.graph.nodes
        if (node.meta.get("custom") or {}).get(PIPER_EP_BOUNDARY_META_KEY)
        == EP_DISPATCH_COUNTS
    )
    counts_marker.meta["example_value"] = torch.tensor(
        [[1, 2, 2, 2]], dtype=torch.int64
    )

    with pytest.raises(ValueError, match="rows to equal the sum of route counts"):
        split_gm_by_annotations(gm)
