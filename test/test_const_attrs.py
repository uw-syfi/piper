"""Non-trainable buffers and attributes load from const_attrs by module path."""
import pytest
import torch
import torch.nn as nn

from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state, annotate
from test.cpu_e2e import ray_cpu, run_program, train  # noqa: F401


class _Buffered(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.register_buffer("scale", torch.ones(dim))
        self.inner = nn.Module()
        self.inner.register_buffer("shift", torch.zeros(dim))
        self.freqs = torch.ones(dim)

    def forward(self, x):
        with annotate("PP"):
            return self.fc(x) * self.scale + self.inner.shift + self.freqs


CONSTS = {
    "scale": torch.linspace(0.5, 1.5, 8),
    "inner.shift": torch.linspace(-1.0, 1.0, 8),
    "freqs": torch.full((8,), 1.0 / 3.0),
}

SCHEDULE = [
    {"op": "place", "filter": {"PP": 0}, "devices": [0]},
    {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
]


def test_segment_records_slot_module_paths() -> None:
    seen: dict = {}

    def _backend(gm, example_inputs):
        _, segments = split_gm_by_annotations(gm, ["x"])
        seen["fqns"] = segments[0].slot_fqns
        return gm.forward

    _reset_annotation_state()
    torch._dynamo.reset()
    torch.compile(_Buffered(), backend=_backend, fullgraph=True)(torch.randn(2, 8))
    assert sorted(seen["fqns"].values()) == ["fc.weight", "freqs", "inner.shift", "scale"]


def _slot_values(result: dict) -> dict:
    (params,) = result["initial"].values()
    return {name: value for name, value in params.items() if "fc" not in name}


class _ListAttribute(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.tables = [torch.ones(dim)]

    def forward(self, x):
        with annotate("PP"):
            return self.fc(x) * self.tables[0]


def test_slot_without_a_module_path_is_refused() -> None:
    seen: dict = {}

    def _backend(gm, example_inputs):
        try:
            split_gm_by_annotations(gm, ["x"])
        except ValueError as err:
            seen["error"] = err
        return gm.forward

    _reset_annotation_state()
    torch._dynamo.reset()
    torch.compile(_ListAttribute(), backend=_backend, fullgraph=True)(torch.randn(2, 8))
    assert "cannot find the module path" in str(seen.get("error"))


def test_const_attrs_fill_nested_buffers_in_model_dtype(ray_cpu, tmp_path) -> None:
    batch = [(torch.randn(2, 8),), torch.randn(2, 8)]
    (result,) = run_program(
        tmp_path, SCHEDULE, train, _Buffered, str(tmp_path / "schedule.json"), [batch], 0,
        model_dtype=torch.bfloat16, const_attrs=CONSTS,
    )
    values = _slot_values(result)
    assert torch.equal(values["l_self_buffers_scale_"], CONSTS["scale"].to(torch.bfloat16))
    assert torch.equal(values["l_self_modules_inner_buffers_shift_"], CONSTS["inner.shift"].to(torch.bfloat16))
    # model.to() does not cast a plain attribute, so its slot keeps fp32.
    assert torch.equal(values["l_self_freqs"], CONSTS["freqs"])


def test_strict_const_attrs_refuses_a_missing_constant(ray_cpu, tmp_path) -> None:
    batch = [(torch.randn(2, 8),), torch.randn(2, 8)]
    missing = {k: v for k, v in CONSTS.items() if k != "inner.shift"}
    with pytest.raises(Exception, match=r"inner\.shift"):
        run_program(
            tmp_path, SCHEDULE, train, _Buffered, str(tmp_path / "schedule.json"), [batch], 0,
            const_attrs=missing, strict_const_attrs=True,
        )
