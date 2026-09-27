"""With microbatches, each gradient bucket is all-reduced once per step."""
import torch
import torch.nn as nn

from examples.build_schedule import build_1f1b_schedule, build_zerobubble_schedule
from src.dag import _has_path
from src.piper import _reset_annotation_state, annotate, piper
from src.state import piper_metadata


class _TwoStage(nn.Module):
    def __init__(self, dim: int = 8):
        super().__init__()
        self.a = nn.Linear(dim, dim, bias=False)
        self.b = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        with annotate("PP"):
            h = self.a(x)
        with annotate("PP"):
            return self.b(h)


def _per_rank_dags(directives: list[dict]) -> list:
    piper_metadata.schedule_directives = directives
    piper_metadata.forward_arg_names = ["x"]
    _reset_annotation_state()
    torch._dynamo.reset()
    with torch.device("meta"):
        model = _TwoStage()
    torch.compile(model, backend=piper, fullgraph=True)(torch.randn(2, 8, device="meta"))
    return piper_metadata.per_pp_training_dags


def _check_one_reduce_after_every_backward(dag) -> None:
    reduces = [n for n in dag.nodes.values() if n.node_kind == "REDUCE_COMM"]
    buckets = {n.node_meta["bucket_key"] for n in reduces}
    assert len(reduces) == len(buckets), [n.uid for n in reduces]
    for reduce in reduces:
        weight_grads = [
            n.uid for n in dag.nodes.values()
            if n.node_kind == "COMPUTE"
            and n.compute_subkind in ("BWD", "BWD_W")
            and n.node_meta.get("bucket_key") == reduce.node_meta["bucket_key"]
        ]
        assert len(weight_grads) == 2
        assert all(_has_path(dag, uid, reduce.uid) for uid in weight_grads)


def test_one_reduce_per_bucket_at_one_pipeline_rank() -> None:
    (dag,) = _per_rank_dags([
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 1]},
        {"op": "place", "filter": {"PP": 1}, "devices": [0, 1]},
        {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1], "reduce_stream": "dp_stream"},
        {"op": "replicate", "filter": {"PP": 1}, "devices": [0, 1], "reduce_stream": "dp_stream"},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
    ])
    _check_one_reduce_after_every_backward(dag)


def test_one_reduce_per_bucket_with_split_backward() -> None:
    dags = _per_rank_dags([
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 2]},
        {"op": "place", "filter": {"PP": 1}, "devices": [1, 3]},
        {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 2], "reduce_stream": "dp_stream"},
        {"op": "replicate", "filter": {"PP": 1}, "devices": [1, 3], "reduce_stream": "dp_stream"},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
    ] + build_zerobubble_schedule(2, 2))
    for dag in dags:
        _check_one_reduce_after_every_backward(dag)


def test_one_reduce_per_bucket_across_pipeline_ranks() -> None:
    dags = _per_rank_dags([
        {"op": "place", "filter": {"PP": 0}, "devices": [0, 2]},
        {"op": "place", "filter": {"PP": 1}, "devices": [1, 3]},
        {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 2], "reduce_stream": "dp_stream"},
        {"op": "replicate", "filter": {"PP": 1}, "devices": [1, 3], "reduce_stream": "dp_stream"},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 2},
    ] + build_1f1b_schedule(2, 2))
    for dag in dags:
        _check_one_reduce_after_every_backward(dag)
