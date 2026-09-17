"""Segment boundaries record every crossing tensor and whether the segment
produced it or forwarded it."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.ring_attn import RingAttn  # noqa: E402
from models.tp_mlp import TPMlp  # noqa: E402

from src.fx import split_gm_by_annotations
from src.piper import _reset_annotation_state


def _segments(model, *inputs):
    box = {}

    def backend(gm, example_inputs):
        box["gm"] = gm
        return gm.forward

    _reset_annotation_state()
    torch._dynamo.reset()
    with torch.no_grad():
        torch.compile(model, backend=backend, fullgraph=True)(*inputs)
    return split_gm_by_annotations(box["gm"])[1]


def test_ring_regions_forward_kv_and_produce_accumulators() -> None:
    torch.manual_seed(0)
    x, k, v = (torch.randn(2, 8, 32) for _ in range(3))
    segs = _segments(RingAttn(32, 2, 4), x, k, v)
    cp = [s for s in segs if "CP" in s.tag]
    assert len(cp) == 4

    for s in cp[:-1]:
        outs = s.a2a_boundary_after["outputs"]
        by_src = {o["src_name"]: o for o in outs}
        assert by_src["k"]["forwarded"] and by_src["v"]["forwarded"] and by_src["q"]["forwarded"]
        produced = [o["src_name"] for o in outs if not o["forwarded"]]
        assert len(produced) == 3, produced        # o, m, l accumulators
        # idx matches position in the output tuple, and tensor_idx names one of them
        assert [o["idx"] for o in outs] == list(range(len(outs)))
        assert s.a2a_boundary_after["tensor_idx"] in range(len(outs))

    # The last ring step emits only accumulators; nothing is forwarded.
    assert all(not o["forwarded"] for o in cp[-1].a2a_boundary_after["outputs"])

    # The prologue produced everything it hands over.
    assert all(not o["forwarded"] for o in segs[0].a2a_boundary_after["outputs"])


def test_tp_region_boundary_is_single_produced_tensor() -> None:
    """Existing TP path: one partial sum, produced, so shard_tensor's
    single-index contract keeps holding."""
    with torch.device("meta"):
        m = TPMlp(64, 128, 1).to(torch.float32)
    segs = _segments(m, torch.empty(4, 64, device="meta"))
    tp = [s for s in segs if "TP" in s.tag and s.a2a_boundary_after is not None]
    assert tp
    for s in tp:
        outs = s.a2a_boundary_after["outputs"]
        assert len(outs) == 1 and not outs[0]["forwarded"], outs
        assert s.a2a_boundary_after["tensor_idx"] == 0
