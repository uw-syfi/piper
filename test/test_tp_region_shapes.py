"""Two combinations shard_tensor had never been exercised against."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.directives as directives
from src.dag import build_training_dag
from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state, annotate

DEV = [0, 1]


def _trace(model, x):
    got = {}

    def backend(gm, _inputs):
        got["segs"] = split_gm_by_annotations(gm)[1]
        return gm.forward

    _reset_annotation_state()
    torch._dynamo.reset()
    torch.compile(model, backend=backend, fullgraph=True)(x)
    return build_training_dag(got["segs"])


class _SingleOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Linear(16, 16, bias=False)
        self.up = nn.Linear(16, 32, bias=False)
        self.down = nn.Linear(32, 16, bias=False)
        self.post = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        with annotate("PP"):
            x = self.pre(x)
            with annotate("TP"):
                x = self.down(F.gelu(self.up(x)))
            return self.post(x)


class _MultiOut(_SingleOut):
    def __init__(self):
        super().__init__()
        self.aux = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        with annotate("PP"):
            x = self.pre(x)
            with annotate("TP"):
                h = self.down(F.gelu(self.up(x)))  # partial sum, needs the collective
                g = self.aux(x)                    # replicated, must not be reduced
            return self.post(h + g)


def test_multi_output_tp_region_is_rejected() -> None:
    """One boundary carries one tensor index, so only one output gets reduced.

    Which one depends on _select_boundary_tensor_idx's score, so the failure is
    silent and data-dependent: either the replicated tensor is summed across ranks
    or the partial sum is never reduced.
    """
    with torch.device("meta"):
        model = _MultiOut().to(torch.float32)
    dag = _trace(model, torch.empty(2, 16, device="meta"))

    with pytest.raises(ValueError, match="tensors .* across its boundary"):
        directives.apply_schedule_directives(dag, [
            {"op": "place", "filter": {"PP": 0}, "devices": DEV},
            {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEV},
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
        ])


def test_tp_composes_with_split_backward() -> None:
    """Zero-bubble style schedules split BWD into BWD_I and BWD_W.

    That happens *before* shard_tensor runs, so the TP pass sees the split nodes.
    The backward collective belongs on BWD_I, which carries the activation
    gradient; BWD_W produces only weight gradients and needs none.
    """
    with torch.device("meta"):
        model = _SingleOut().to(torch.float32)
    dag = _trace(model, torch.empty(2, 16, device="meta"))

    directives.apply_schedule_directives(dag, [
        {"op": "place", "filter": {"PP": 0}, "devices": DEV},
        {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEV},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
        {"op": "order", "filters": [
            [{"MB": 0, "PASS": "F"}], [{"MB": 1, "PASS": "F"}],
            [{"MB": 0, "PASS": "BI"}], [{"MB": 0, "PASS": "BW"}],
            [{"MB": 1, "PASS": "BI"}], [{"MB": 1, "PASS": "BW"}]]},
    ])

    subkind = {u: n.compute_subkind for u, n in dag.nodes.items()
               if n.node_kind == "COMPUTE"}
    assert "BWD_I" in subkind.values() and "BWD_W" in subkind.values()

    tp = [(n.tag.get("PASS"), n.node_meta["source_uid"])
          for n in dag.nodes.values() if n.node_kind == "TP_COMM"]
    assert len(tp) == 4, tp
    assert sorted(p for p, _ in tp) == ["BI", "BI", "F", "F"]
    for p, src in tp:
        assert subkind[src] == ("FWD" if p == "F" else "BWD_I"), (p, src)
