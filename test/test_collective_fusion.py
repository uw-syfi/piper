"""fuse_collectives merges TP collectives, unless `order` says otherwise."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.directives as directives
from src.dag import build_training_dag, _topological_order
from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state, annotate

DEV = [0, 1]
MB = 4


class _Net(nn.Module):
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


def _dag():
    got = {}

    def backend(gm, _inputs):
        got["segs"] = split_gm_by_annotations(gm)[1]
        return gm.forward

    _reset_annotation_state()
    torch._dynamo.reset()
    with torch.device("meta"):
        model = _Net().to(torch.float32)
    torch.compile(model, backend=backend, fullgraph=True)(torch.empty(2, 16, device="meta"))
    return build_training_dag(got["segs"])


_BASE = [
    {"op": "place", "filter": {"PP": 0}, "devices": DEV},
    {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEV, "stream": "tp_stream"},
    {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": MB},
]
_FUSE = {"op": "fuse_collectives", "filter": {"TP": "*"}}

# 1F1B: interleave microbatch i's backward with i+2's forward.
_1F1B = {"op": "order", "filters":
         [[{"MB": 0, "PASS": "F"}], [{"MB": 1, "PASS": "F"}]]
         + [[{"MB": i, "PASS": "F"}, {"MB": i - 2, "PASS": "B"}] for i in (2, 3)]
         + [[{"MB": i, "PASS": "B"}] for i in (2, 3)]}


def _groups(dag):
    out = {}
    for n in dag.nodes.values():
        gid = n.node_meta.get("fusion_group")
        if gid:
            out.setdefault(gid, []).append(n.uid)
    return out


def test_collectives_fuse_without_an_order_directive() -> None:
    """Four microbatches give one forward group and one backward group."""
    dag = _dag()
    directives.apply_schedule_directives(dag, _BASE + [_FUSE])

    groups = _groups(dag)
    assert len(groups) == 2, groups
    for members in groups.values():
        assert len(members) == MB, members
        passes = {dag.nodes[u].tag["PASS"] for u in members}
        assert len(passes) == 1, passes

    leaders = {dag.nodes[m[0]].node_meta["fusion_leader"] for m in groups.values()}
    assert len(leaders) == 2
    _topological_order(dag)  # fusion edges must not create a cycle


def test_order_wins_over_fusion() -> None:
    """1F1B deliberately interleaves microbatches; fusion must not undo that.

    Fusing forces every member's producer to finish before any member runs, which
    contradicts running microbatch 0's backward before microbatch 2's forward.
    The group is left unfused rather than reordered.
    """
    dag = _dag()
    directives.apply_schedule_directives(dag, _BASE + [_1F1B, _FUSE])

    assert _groups(dag) == {}, "1F1B should have blocked every fusion group"
    _topological_order(dag)


def test_fusion_records_what_the_runtime_needs() -> None:
    dag = _dag()
    directives.apply_schedule_directives(dag, _BASE + [_FUSE])

    for members in _groups(dag).values():
        metas = [dag.nodes[u].node_meta for u in members]
        assert {m["fusion_size"] for m in metas} == {MB}
        assert sorted(m["fusion_index"] for m in metas) == list(range(MB))
        # every member agrees on the member list and the leader
        assert len({tuple(m["fusion_members"]) for m in metas}) == 1
        assert len({m["fusion_leader"] for m in metas}) == 1
