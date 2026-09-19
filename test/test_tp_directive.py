"""CPU tests for the `shard_tensor` directive and the TP_COMM node kind.

These assert the IR transformation only. Nothing here needs a GPU, which is the
point: the shape of the rewrite is the cheapest thing to get wrong and the most
expensive to debug through Ray.
"""
import json

import pytest

import src.directives as directives
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode, _topological_order
from src.piper import _split_global_training_dag_by_pp_rank
from src.schedule import load_schedule_directives
from src.tasks import TaskType, training_dag_task_type

DEVICES = [0, 1]


def _fwd(uid: str, tag: dict, *, boundary: bool) -> TrainingDAGNode:
    """A forward segment node. `boundary` mirrors fx.py's a2a_boundary_after."""
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="FWD",
        tag={**tag, "PASS": "F"},
        device=list(DEVICES),
        stream="default_stream",
        node_meta={
            "bucket_key": uid,
            "a2a_boundary_after": (
                {"tensor_idx": 0, "reshape_input": None, "reshape_output": None}
                if boundary else None
            ),
        },
    )


def _bwd(uid: str, fwd_uid: str, tag: dict) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind="BWD",
        tag={**tag, "PASS": "B"},
        device=list(DEVICES),
        stream="default_stream",
        node_meta={"fwd_uid": fwd_uid, "bucket_key": fwd_uid},
    )


def _chain(tags: list[dict]) -> TrainingDAG:
    """Build a pre -> ... -> post DAG shaped like build_training_dag's output.

    One FWD node per tag, forward data edges in order, a mirrored BWD chain, the
    last_fwd -> first_bwd bridge, and a UPD sink on temporal edges.
    """
    dag = TrainingDAG()
    uids = [f"s{i}" for i in range(len(tags))]
    for i, (uid, tag) in enumerate(zip(uids, tags)):
        dag.add_node(_fwd(uid, tag, boundary=i < len(tags) - 1))
    for a, b in zip(uids, uids[1:]):
        dag.add_edge(TrainingDAGEdge(a, b, "data"))

    for uid, tag in zip(uids, tags):
        dag.add_node(_bwd(f"{uid}.bwd", uid, tag))
    # Reverse every forward edge: FWD u -> v becomes BWD v' -> u'.
    for a, b in zip(uids, uids[1:]):
        dag.add_edge(TrainingDAGEdge(f"{b}.bwd", f"{a}.bwd", "data"))
    dag.add_edge(TrainingDAGEdge(uids[-1], f"{uids[-1]}.bwd", "data"))

    dag.add_node(TrainingDAGNode(
        uid="upd.0", node_kind="UPD", compute_subkind=None, tag={"PASS": None},
        device=list(DEVICES), stream="default_stream", node_meta={}))
    for uid in uids:
        dag.add_edge(TrainingDAGEdge(f"{uid}.bwd", "upd.0", "temporal"))
    return dag


def _tp_nodes(dag: TrainingDAG) -> list[TrainingDAGNode]:
    return sorted(
        (n for n in dag.nodes.values() if n.node_kind == "TP_COMM"),
        key=lambda n: n.uid,
    )


def _data_edge(dag: TrainingDAG, src: str, dst: str) -> bool:
    return any(
        e.src_uid == src and e.dst_uid == dst and e.dep_kind == "data"
        for e in dag.edges
    )


# --------------------------------------------------------------------- schema

def test_schedule_loader_accepts_shard_tensor(tmp_path) -> None:
    path = tmp_path / "tp2.json"
    path.write_text(json.dumps([
        {"op": "place", "filter": {"PP": 0}, "devices": DEVICES},
        {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEVICES,
         "stream": "tp_stream"},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    ]), encoding="utf-8")

    directives_list = load_schedule_directives(str(path))

    assert directives_list[1]["op"] == "shard_tensor"


def test_shard_tensor_rejects_gather_and_reduce_stream() -> None:
    with pytest.raises(ValueError, match="does not accept"):
        directives._normalize_filter_devices_directive({
            "op": "shard_tensor", "filter": {"TP": 0}, "devices": DEVICES,
            "reduce_stream": "dp_stream",
        })


# ------------------------------------------------------------- the TP rewrite

def test_shard_tensor_inserts_one_forward_and_one_backward_all_reduce() -> None:
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])

    directives._insert_tp_all_reduce_comm_nodes(
        dag, [{"TP": 0}], DEVICES, comm_stream="tp_stream")

    tp = _tp_nodes(dag)
    assert len(tp) == 2, [n.uid for n in tp]

    by_pass = {n.tag["PASS"]: n for n in tp}
    assert set(by_pass) == {"F", "B"}

    fwd_comm, bwd_comm = by_pass["F"], by_pass["B"]
    for n in tp:
        assert n.node_meta["direction"] == "outgoing"
        assert n.node_meta["tp_tensor_idx"] == 0
        assert n.stream == "tp_stream"
        assert n.device == DEVICES

    # g: forward output of the region is a partial sum -> reduce on the way out.
    assert fwd_comm.node_meta["source_uid"] == "s1"
    assert _data_edge(dag, "s1", fwd_comm.uid)
    assert _data_edge(dag, fwd_comm.uid, "s2")
    assert not _data_edge(dag, "s1", "s2")

    # f: gradient w.r.t. the region's input is a partial sum, and backward edges
    # are reversed, so it is also an outgoing edge.
    assert bwd_comm.node_meta["source_uid"] == "s1.bwd"
    assert _data_edge(dag, "s1.bwd", bwd_comm.uid)
    assert _data_edge(dag, bwd_comm.uid, "s0.bwd")
    assert not _data_edge(dag, "s1.bwd", "s0.bwd")

    # The edge *into* the region carries a replicated tensor: no collective.
    assert _data_edge(dag, "s2.bwd", "s1.bwd")

    # Still a well-formed single-device-group DAG.
    _topological_order(dag)
    assert len(_split_global_training_dag_by_pp_rank(dag)) == 1


def test_shard_tensor_task_types_map_by_pass() -> None:
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])
    directives._insert_tp_all_reduce_comm_nodes(dag, [{"TP": 0}], DEVICES)

    got = {n.tag["PASS"]: training_dag_task_type(n) for n in _tp_nodes(dag)}

    assert got == {
        "F": TaskType.FWD_TP_ALL_REDUCE,
        "B": TaskType.BWD_TP_ALL_REDUCE,
    }


def test_shard_tensor_does_not_reduce_edges_internal_to_the_region() -> None:
    """A column-parallel output feeding a row-parallel input stays sharded."""
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0, "TP": 1}, {"PP": 0}])

    wildcard = directives._normalize_filter_spec({"TP": "*"}, {})
    directives._insert_tp_all_reduce_comm_nodes(dag, [wildcard], DEVICES)

    tp = _tp_nodes(dag)
    assert len(tp) == 2, [n.uid for n in tp]
    sources = {n.node_meta["source_uid"] for n in tp}
    # Reduce after the last region segment, and before the first one in backward.
    assert sources == {"s2", "s1.bwd"}
    # The internal boundary is untouched in both directions.
    assert _data_edge(dag, "s1", "s2")
    assert _data_edge(dag, "s2.bwd", "s1.bwd")


def test_shard_tensor_inserts_nothing_without_a_downstream_consumer() -> None:
    """A region with no upstream segment has no input gradient to all-reduce.

    This is why the minimal TP example keeps a replicated segment on each side:
    otherwise the backward half of the conjugate pair is never exercised.
    """
    dag = _chain([{"PP": 0, "TP": 0}, {"PP": 0}])

    directives._insert_tp_all_reduce_comm_nodes(dag, [{"TP": 0}], DEVICES)

    tp = _tp_nodes(dag)
    assert [n.tag["PASS"] for n in tp] == ["F"]


# ------------------------------------------------------------- rejected cases

def test_shard_tensor_rejects_device_mismatch() -> None:
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])
    dag.nodes["s1"].device = [0, 2]

    with pytest.raises(ValueError, match="requires matched node"):
        directives._insert_tp_all_reduce_comm_nodes(dag, [{"TP": 0}], DEVICES)


def test_shard_tensor_rejects_composition_with_replicate() -> None:
    """TP weight gradients are shard-local; reducing them would be wrong.

    `shard` silently deletes the DP sync comms it finds. TP refuses instead, so the
    composition limitation surfaces rather than being resolved behind the user's back.
    """
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])
    dag.nodes["s1.bwd"].node_meta["graphargs"] = None
    directives._insert_reduce_comm_nodes(dag, [{"TP": 0}], DEVICES)
    assert any(n.node_kind == "REDUCE_COMM" for n in dag.nodes.values())

    with pytest.raises(ValueError, match="cannot compose with replicate"):
        directives._insert_tp_all_reduce_comm_nodes(dag, [{"TP": 0}], DEVICES)


# ----------- regression lock on the boundary resolver shared with the EP pass

def test_boundary_resolver_is_shared_by_the_ep_and_tp_passes() -> None:
    """_boundary_info_for_edge was lifted out of _insert_shard_a2a_comm_nodes.

    Exercise it through both callers, including the backward branch that resolves
    the producer via the edge *destination's* fwd_uid, so the extraction cannot
    regress EP silently.
    """
    ep = _chain([{"PP": 0}, {"PP": 0, "EP": 0}, {"PP": 0}])
    directives._insert_shard_a2a_comm_nodes(ep, [{"EP": 0}], DEVICES)
    a2a = [n for n in ep.nodes.values() if n.node_kind == "A2A_COMM"]
    # EP reduces on both sides of the region, in both passes: 4 nodes.
    assert len(a2a) == 4, [n.uid for n in a2a]
    assert {n.node_meta["direction"] for n in a2a} == {"incoming", "outgoing"}
    assert all(n.node_meta["a2a_tensor_idx"] == 0 for n in a2a)

    tp = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])
    directives._insert_tp_all_reduce_comm_nodes(tp, [{"TP": 0}], DEVICES)
    assert all(n.node_meta["tp_tensor_idx"] == 0 for n in _tp_nodes(tp))


def test_boundary_resolver_reports_the_calling_pass_on_failure() -> None:
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])
    dag.nodes["s1"].node_meta["a2a_boundary_after"] = None

    with pytest.raises(ValueError, match="TP_COMM missing tensor_idx"):
        directives._insert_tp_all_reduce_comm_nodes(dag, [{"TP": 0}], DEVICES)

# ------------------------------------------------- the directive, end to end

def test_shard_tensor_dispatches_through_apply_schedule_directives() -> None:
    """The whole path: JSON directive -> normalization -> place -> TP rewrite.

    Exercises the dispatch wiring, which testing the pass directly does not.
    """
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])
    for node in dag.nodes.values():
        node.device = None

    directives.apply_schedule_directives(dag, [
        {"op": "place", "filter": {"PP": 0}, "devices": DEVICES, "stream": "pp_stream"},
        {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEVICES,
         "stream": "tp_stream"},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    ])

    tp = _tp_nodes(dag)
    assert {n.tag["PASS"] for n in tp} == {"F", "B"}
    assert all(n.stream == "tp_stream" for n in tp)
    # No DP collective anywhere: TP weight gradients stay shard-local.
    assert not any(
        n.node_kind in {"REDUCE_COMM", "ALL_GATHER_COMM", "REDUCE_SCATTER_COMM"}
        for n in dag.nodes.values()
    )
    # And no pipeline P2P, since everything sits on one device group.
    assert not any(
        n.node_kind in {"SEND_COMM", "RECV_COMM"} for n in dag.nodes.values()
    )
    assert len(_split_global_training_dag_by_pp_rank(dag)) == 1


def test_shard_tensor_survives_microbatch_split() -> None:
    """split runs before shard_tensor, so each microbatch gets its own pair."""
    dag = _chain([{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])
    for node in dag.nodes.values():
        node.device = None

    directives.apply_schedule_directives(dag, [
        {"op": "place", "filter": {"PP": 0}, "devices": DEVICES},
        {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEVICES,
         "stream": "tp_stream"},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 3},
    ])

    tp = _tp_nodes(dag)
    assert len(tp) == 6, [n.uid for n in tp]
    assert sorted(n.tag["MB"] for n in tp) == [0, 0, 1, 1, 2, 2]
