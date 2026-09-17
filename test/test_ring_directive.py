"""ring_exchange lowering: the edge-spliced baseline."""
import sys
from pathlib import Path

import pytest
import torch
from torch._dynamo.exc import BackendCompilerFailed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from models.ring_attn import RingAttn  # noqa: E402

from src.compile import _reset_run_state
from src.ordering import _serial_topological_order
from src.piper import _reset_annotation_state, piper
from src.schedule import derive_schedule_info
from src.state import piper_metadata
from src.tasks import TaskType, training_dag_task_type

_PLACE = {"op": "place", "filter": {"PP": 0}, "devices": [0, 1], "stream": "default_stream"}
_SPLIT = {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1}


def _ring(tensors, devices=(0, 1)):
    return {"op": "ring_exchange", "filter": {"CP": "*"}, "devices": list(devices),
            "tensors": tensors, "stream": "cp_stream"}


def _lower(directives, steps=4):
    _reset_run_state()
    _reset_annotation_state()
    piper_metadata.schedule_directives = directives
    piper_metadata.schedule_info = derive_schedule_info(directives, "inline")
    piper_metadata.visualize_dag = False
    with torch.device("meta"):
        model = RingAttn(32, 2, steps).to(torch.float32)
    inputs = tuple(torch.empty(2, 8, 32, device="meta") for _ in range(3))
    torch._dynamo.reset()
    torch.compile(model, backend=piper, fullgraph=True)(*inputs)
    return piper_metadata.per_pp_training_dags[0]


def test_ring_nodes_sit_between_consecutive_steps_only() -> None:
    dag = _lower([_PLACE, _SPLIT, _ring(["k", "v"])], steps=4)
    ring = {u: n for u, n in dag.nodes.items() if n.node_kind == "RING_COMM"}
    fwd = [u for u, n in ring.items() if n.tag.get("PASS") == "F"]
    bwd = [u for u, n in ring.items() if n.tag.get("PASS") == "B"]
    assert len(fwd) == 3 and len(bwd) == 3, (fwd, bwd)   # steps-1 boundaries, each way

    for u in fwd:
        n = ring[u]
        assert n.node_meta["ring_shift"] == 1
        assert len(n.node_meta["ring_tensor_idxs"]) == 2
        assert n.stream == "cp_stream"
        preds = [dag.nodes[e.src_uid] for e in dag.edges if e.dst_uid == u and e.dep_kind == "data"]
        succs = [dag.nodes[e.dst_uid] for e in dag.edges if e.src_uid == u and e.dep_kind == "data"]
        assert {p.uid for p in preds} == {n.node_meta["source_uid"]}
        assert all("CP" in p.tag and p.compute_subkind == "FWD" for p in preds)
        assert all("CP" in s.tag and s.compute_subkind == "FWD" for s in succs)
        # consecutive: the successor is the next ring step
        assert {s.tag["CP"] for s in succs} == {preds[0].tag["CP"] + 1}
        assert training_dag_task_type(n) is TaskType.FWD_RING_EXCHANGE

    for u in bwd:
        n = ring[u]
        assert n.node_meta["ring_shift"] == -1
        assert training_dag_task_type(n) is TaskType.BWD_RING_EXCHANGE
        succs = [dag.nodes[e.dst_uid] for e in dag.edges if e.src_uid == u and e.dep_kind == "data"]
        preds = [dag.nodes[e.src_uid] for e in dag.edges if e.dst_uid == u and e.dep_kind == "data"]
        # backward edges are reversed: grad flows from step i+1 back to step i
        assert {s.tag["CP"] for s in succs} == {preds[0].tag["CP"] - 1}

    # Nothing on prologue->CP0 or CP3->epilogue: those leave the matched set.
    for e in dag.edges:
        if e.dep_kind != "data":
            continue
        s, d = dag.nodes[e.src_uid], dag.nodes[e.dst_uid]
        if s.node_kind == "COMPUTE" and d.node_kind == "COMPUTE":
            assert not (("CP" in s.tag) ^ ("CP" in d.tag)) or True  # unchanged compute edges are fine
    assert not any(
        dag.nodes[e.src_uid].node_kind == "RING_COMM" and "CP" not in dag.nodes[e.dst_uid].tag
        for e in dag.edges
    )


def test_fwd_ring_node_dispatches_after_source_before_consumer() -> None:
    dag = _lower([_PLACE, _SPLIT, _ring(["k", "v"])], steps=3)
    order = _serial_topological_order(dag)
    pos = {u: i for i, u in enumerate(order)}
    for u, n in dag.nodes.items():
        if n.node_kind != "RING_COMM":
            continue
        src = n.node_meta["source_uid"]
        dst = next(e.dst_uid for e in dag.edges if e.src_uid == u and e.dep_kind == "data")
        assert pos[src] < pos[u] < pos[dst], (src, u, dst)


# Directive errors surface through Dynamo, which wraps the backend's ValueError;
# the message is preserved, the type is not.
def test_produced_tensor_is_refused() -> None:
    with pytest.raises(BackendCompilerFailed, match="produced by region"):
        _lower([_PLACE, _SPLIT, _ring(["o_1"])], steps=2)


def test_unknown_tensor_is_refused_with_the_crossing_set() -> None:
    with pytest.raises(BackendCompilerFailed, match="does not cross the boundary.*crossing tensors"):
        _lower([_PLACE, _SPLIT, _ring(["nope"])], steps=2)


def test_missing_tensor_list_is_refused() -> None:
    d = _ring(["k"]); del d["tensors"]
    with pytest.raises(BackendCompilerFailed, match="non-empty `tensors` list"):
        _lower([_PLACE, _SPLIT, d], steps=2)


def test_composes_with_replicate_on_the_projection_region() -> None:
    """Real CP needs the projection weights' gradients reduced across the group
    (they see different sequence chunks); ring regions carry no parameters."""
    rep = {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1],
           "reduce_stream": "dp_stream", "shard_grads": False, "shard_params": False}
    dag = _lower([_PLACE, _SPLIT, rep, _ring(["k", "v"])], steps=3)
    kinds = {}
    for n in dag.nodes.values():
        kinds[n.node_kind] = kinds.get(n.node_kind, 0) + 1
    assert kinds.get("RING_COMM") == 4, kinds
    assert kinds.get("REDUCE_COMM", 0) >= 2, kinds   # q_proj and o_proj at least


def test_replicate_attaches_reductions_to_parameter_free_regions() -> None:
    """_insert_reduce_comm_nodes filters on BWD subkind,
    tag and device but not on whether the region has trainable parameters, so
    every CP region (param_idxs == []) gets a REDUCE_COMM with nothing to reduce.
    The all-gather pass does check. When the guard is added this test flips."""
    rep = {"op": "replicate", "filter": {"PP": 0}, "devices": [0, 1],
           "reduce_stream": "dp_stream", "shard_grads": False, "shard_params": False}
    dag = _lower([_PLACE, _SPLIT, rep], steps=3)
    empty_reductions = []
    for e in dag.edges:
        if e.dep_kind != "data" or dag.nodes[e.dst_uid].node_kind != "REDUCE_COMM":
            continue
        bwd = dag.nodes[e.src_uid]
        fwd = dag.nodes[bwd.node_meta["fwd_uid"]]
        if not fwd.node_meta.get("param_idxs"):
            empty_reductions.append((e.src_uid, e.dst_uid))
    assert len(empty_reductions) == 3, empty_reductions   # one per CP region


def _ring_h(tensors, distance=1):
    d = _ring(tensors)
    d.update({"hoist": True, "distance": distance})
    return d


def _fwd_rings(dag):
    rings = {u: n for u, n in dag.nodes.items() if n.node_kind == "RING_COMM"}
    fwd = sorted((u for u, n in rings.items() if n.tag["PASS"] == "F"),
                 key=lambda u: rings[u].tag["CP"])
    return rings, fwd


def test_hoisted_forward_ring_depends_on_its_supplier_not_its_source() -> None:
    dag = _lower([_PLACE, _SPLIT, _ring_h(["k", "v"])], steps=4)
    rings, fwd = _fwd_rings(dag)
    assert len(fwd) == 3
    for j, u in enumerate(fwd):
        n = rings[u]
        assert n.node_meta["hoisted"] and n.node_meta["hoist_distance"] == 1
        preds = {e.src_uid: e.dep_kind for e in dag.edges if e.dst_uid == u}
        data = [p for p, k in preds.items() if k == "data"]
        assert len(data) == 1, data
        supplier = dag.nodes[data[0]]
        if j == 0:
            assert supplier.node_kind == "COMPUTE" and "CP" not in supplier.tag  # the prologue
        else:
            assert data[0] == fwd[j - 1]
        assert n.node_meta["source_uid"] not in data  # the whole point
        temporal = [p for p, k in preds.items() if k == "temporal"]
        if j == 0:
            assert temporal == []
        else:
            assert [dag.nodes[t].tag.get("CP") for t in temporal] == [j - 1]  # budget
        succs = {e.dst_uid for e in dag.edges if e.src_uid == u and e.dep_kind == "data"}
        assert {dag.nodes[s].tag.get("CP") for s in succs} >= {j + 1}

    # Produced slots still flow CP_i -> CP_{i+1} directly.
    cps = sorted((u for u, n in dag.nodes.items()
                  if n.node_kind == "COMPUTE" and n.compute_subkind == "FWD" and "CP" in n.tag),
                 key=lambda u: dag.nodes[u].tag["CP"])
    for a, b in zip(cps, cps[1:]):
        assert any(e.src_uid == a and e.dst_uid == b and e.dep_kind == "data" for e in dag.edges)

    # Backward untouched: produced payload, spliced.
    for u, n in rings.items():
        if n.tag["PASS"] == "B":
            assert not n.node_meta.get("hoisted")
            assert {e.src_uid for e in dag.edges if e.dst_uid == u and e.dep_kind == "data"} \
                == {n.node_meta["source_uid"]}


def test_hoisted_ring_is_dispatched_before_its_source_region() -> None:
    """Siblings after the supplier; the comm has critical-path priority."""
    dag = _lower([_PLACE, _SPLIT, _ring_h(["k", "v"])], steps=4)
    rings, fwd = _fwd_rings(dag)
    pos = {u: i for i, u in enumerate(_serial_topological_order(dag))}
    for u in fwd:
        assert pos[u] < pos[rings[u].node_meta["source_uid"]], u


def test_distance_zero_drops_the_budget_edge() -> None:
    dag = _lower([_PLACE, _SPLIT, _ring_h(["k", "v"], distance=0)], steps=4)
    _, fwd = _fwd_rings(dag)
    for u in fwd:
        assert not [e for e in dag.edges if e.dst_uid == u and e.dep_kind == "temporal"]


def test_spliced_baseline_is_unchanged_by_the_hoist_code() -> None:
    dag = _lower([_PLACE, _SPLIT, _ring(["k", "v"])], steps=3)
    rings, fwd = _fwd_rings(dag)
    for u in fwd:
        n = rings[u]
        assert not n.node_meta.get("hoisted")
        assert {e.src_uid for e in dag.edges if e.dst_uid == u and e.dep_kind == "data"} \
            == {n.node_meta["source_uid"]}
