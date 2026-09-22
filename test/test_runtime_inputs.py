"""The model input must reach the graph as an input, not as a zero-filled param.

Dynamo attaches `meta["grapharg"]` to every placeholder while a backend is
running and clears it afterwards, so this has to be asserted *inside* the
backend call. Checking it after `torch.compile` returns would pass either way.
"""
import pytest
import torch
import torch.nn as nn

from src.fx import _placeholder_is_runtime_input, split_gm_by_annotations
from src.piper import _reset_annotation_state, annotate


class _Net(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, x):
        with annotate("PP"):
            return self.fc(x) * self.scale


def _compile_and_capture(model, x) -> dict:
    seen: dict = {}

    def _backend(gm, example_inputs):
        seen["placeholders"] = {
            n.name: (
                _placeholder_is_runtime_input(n),
                "grapharg" in n.meta,
            )
            for n in gm.graph.nodes
            if n.op == "placeholder"
        }
        _, seen["segments"] = split_gm_by_annotations(gm)
        return gm.forward

    _reset_annotation_state()
    torch._dynamo.reset()
    torch.compile(model, backend=_backend, fullgraph=True)(x)
    return seen


def test_model_input_is_a_runtime_input_inside_the_backend() -> None:
    seen = _compile_and_capture(_Net(), torch.randn(2, 8))
    phs = seen["placeholders"]

    x_name = next(n for n in phs if "self" not in n)
    is_runtime, has_grapharg = phs[x_name]

    # The precondition this test exists for: the meta key is present here, so a
    # check on it would reject the input.
    assert has_grapharg, (
        "meta['grapharg'] is absent during the backend call on this torch "
        "version; the misclassification this test guards can no longer occur "
        "and the guard should be revisited"
    )
    assert is_runtime, f"{x_name} must be a runtime input, not a grapharg"

    # Parameters and lifted module attributes must stay graphargs.
    for name, (runtime, _) in phs.items():
        if "self" in name:
            assert not runtime, f"{name} is a lifted attribute and must not be a runtime input"


def test_first_segment_receives_the_model_input() -> None:
    """seg 0 with input_idxs=[] is the signature of the zero-input bug.

    _load_stage zero-fills every slot that is neither a trainable parameter nor
    a known const attr, and the FWD executor arm only substitutes real inputs for
    indices in input_idxs. An empty input_idxs therefore trains on zeros with no
    error anywhere.
    """
    seen = _compile_and_capture(_Net(), torch.randn(2, 8))
    first = seen["segments"][0]

    assert first.input_idxs, "segment 0 has no runtime inputs; it would train on zeros"
    assert len(first.input_idxs) == 1
    idx = first.input_idxs[0]
    assert tuple(first.graphargs[idx].shape) == (2, 8)
    assert not isinstance(first.graphargs[idx], torch.nn.Parameter)
