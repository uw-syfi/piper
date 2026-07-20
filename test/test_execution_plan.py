from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.executors as executors
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.execution_plan import (
    build_execution_plan,
    is_relaxable_p2p_ordering_dependency,
)
from src.executors import DagExecutor
from src.tasks import TaskType


def _node(
    uid: str,
    *,
    stream: str = "default_stream",
    kind: str = "COMPUTE",
    subkind: str | None = "FWD",
) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind=kind,
        compute_subkind=subkind,
        tag={},
        device=[0],
        stream=stream,
    )


def _dag(*nodes: TrainingDAGNode) -> TrainingDAG:
    dag = TrainingDAG()
    for node in nodes:
        dag.add_node(node)
    return dag


def _by_uid(dag: TrainingDAG, order: list[str] | None = None):
    return {op.uid: op for op in build_execution_plan(dag, order)}


def test_plan_materializes_stream_fifo_and_cross_stream_edges() -> None:
    dag = _dag(
        _node("a"),
        _node("b", stream="comm"),
        _node("c"),
        _node("d", stream="comm"),
    )
    dag.add_edge(TrainingDAGEdge("a", "b", "data"))
    dag.add_edge(TrainingDAGEdge("b", "c", "temporal"))
    dag.add_edge(TrainingDAGEdge("a", "c", "data"))  # same-stream, FIFO implies it
    dag.add_edge(TrainingDAGEdge("c", "d", "data"))

    plan = _by_uid(dag, ["a", "b", "c", "d"])

    assert plan["a"].completion_predecessors == ()
    assert plan["b"].completion_predecessors == ("a",)
    assert plan["b"].reason_for("a") == "cross_stream_data"
    assert plan["c"].completion_predecessors == ("a", "b")
    assert plan["c"].reason_for("a") == "stream_fifo+same_stream_data"
    assert plan["c"].reason_for("b") == "cross_stream_temporal"
    assert plan["d"].completion_predecessors == ("b", "c")
    assert plan["d"].reason_for("b") == "stream_fifo"
    assert plan["d"].reason_for("c") == "cross_stream_data"


def test_recv_waits_for_latest_compute_on_consumer_stream() -> None:
    dag = _dag(
        _node("prior_compute"),
        _node("recv", stream="pp", kind="RECV_COMM", subkind=None),
        _node("consumer"),
    )
    dag.add_edge(TrainingDAGEdge("recv", "consumer", "data"))

    plan = _by_uid(dag, ["prior_compute", "recv", "consumer"])

    assert plan["recv"].completion_predecessors == ("prior_compute",)
    assert (
        plan["recv"].reason_for("prior_compute")
        == "recv_consumer_stream_guard"
    )
    assert plan["consumer"].completion_predecessors == (
        "prior_compute",
        "recv",
    )


def test_update_waits_for_backward_and_gradient_synchronization() -> None:
    dag = _dag(
        _node("bwd", subkind="BWD"),
        _node("all_reduce", stream="dp", kind="REDUCE_COMM", subkind=None),
        _node(
            "reduce_scatter",
            stream="rs",
            kind="REDUCE_SCATTER_COMM",
            subkind=None,
        ),
        _node("upd", kind="UPD", subkind=None),
    )

    plan = _by_uid(dag, ["bwd", "all_reduce", "reduce_scatter", "upd"])

    assert plan["upd"].completion_predecessors == (
        "bwd",
        "all_reduce",
        "reduce_scatter",
    )
    assert plan["upd"].reason_for("bwd") == (
        "stream_fifo+update_backward"
    )
    assert plan["upd"].reason_for("all_reduce") == "update_gradient_sync"
    assert plan["upd"].reason_for("reduce_scatter") == "update_gradient_sync"


def test_plan_rejects_dispatch_order_that_violates_an_edge() -> None:
    dag = _dag(_node("a"), _node("b"))
    dag.add_edge(TrainingDAGEdge("a", "b", "data"))

    with pytest.raises(ValueError, match="violates DAG dependency"):
        build_execution_plan(dag, ["b", "a"])


def test_directional_p2p_relaxes_ordering_only_but_not_data_dependencies() -> None:
    lo_hi = _node("lo_hi", stream="pp", kind="SEND_COMM", subkind=None)
    hi_lo = _node("hi_lo", stream="pp", kind="RECV_COMM", subkind=None)
    lo_hi.node_meta["comm_group_kind"] = "pp_lo_hi"
    hi_lo.node_meta["comm_group_kind"] = "pp_hi_lo"
    dag = _dag(lo_hi, hi_lo)

    fifo_plan = _by_uid(dag, ["lo_hi", "hi_lo"])
    fifo_reason = fifo_plan["hi_lo"].reason_for("lo_hi")
    assert fifo_reason == "stream_fifo"
    assert is_relaxable_p2p_ordering_dependency(
        lo_hi, hi_lo, fifo_reason
    )

    dag.add_edge(TrainingDAGEdge("lo_hi", "hi_lo", "temporal"))
    explicit_plan = _by_uid(dag, ["lo_hi", "hi_lo"])
    explicit_reason = explicit_plan["hi_lo"].reason_for("lo_hi")
    assert explicit_reason == "stream_fifo+same_stream_temporal"
    assert is_relaxable_p2p_ordering_dependency(
        lo_hi, hi_lo, explicit_reason
    )

    dag.edges.clear()
    dag.add_edge(TrainingDAGEdge("lo_hi", "hi_lo", "data"))
    data_plan = _by_uid(dag, ["lo_hi", "hi_lo"])
    data_reason = data_plan["hi_lo"].reason_for("lo_hi")
    assert data_reason == "stream_fifo+same_stream_data"
    assert not is_relaxable_p2p_ordering_dependency(
        lo_hi, hi_lo, data_reason
    )


def test_plan_rejects_missing_or_duplicate_dispatch_nodes() -> None:
    dag = _dag(_node("a"), _node("b"))

    with pytest.raises(ValueError, match="does not match DAG nodes"):
        build_execution_plan(dag, ["a"])
    with pytest.raises(ValueError, match="duplicate node uids"):
        build_execution_plan(dag, ["a", "a"])


def test_executor_applies_plan_waits_before_timing_callbacks(monkeypatch) -> None:
    trace: list[tuple] = []

    class FakeStream:
        def __init__(self, name: str) -> None:
            self.name = name

        def wait_event(self, event) -> None:
            trace.append(("wait", self.name, event.index))

    class FakeEvent:
        next_index = 0

        def __init__(self) -> None:
            self.index = FakeEvent.next_index
            FakeEvent.next_index += 1

        def record(self, stream: FakeStream) -> None:
            trace.append(("record", self.index, stream.name))

    default_stream = FakeStream("logical-default")
    comm_stream = FakeStream("logical-comm")
    profile_compute = FakeStream("profile-compute")
    profile_comm = FakeStream("profile-comm")
    runtime = SimpleNamespace(
        streams={"default_stream": default_stream, "comm": comm_stream},
        default_stream=lambda: default_stream,
        stream_for_task=lambda node: (
            default_stream if node.stream == "default_stream" else comm_stream
        ),
        stream_id=lambda node: node.stream,
        pytorch_profiler_enabled=False,
        nvtx_push=lambda _label: None,
        nvtx_pop=lambda: None,
    )

    first = _node("first", kind="ORDER_DUMMY", subkind=None)
    second = _node(
        "second", stream="comm", kind="ORDER_DUMMY", subkind=None
    )
    for node in (first, second):
        node.task_type = TaskType.ORDER_DUMMY
        node.batches = [SimpleNamespace(mb_idx=0)]
    dag = _dag(first, second)
    dag.add_edge(TrainingDAGEdge("first", "second", "temporal"))

    class Recorder:
        def before_prologue(self, stream) -> None:
            trace.append(("before_prologue", stream.name))

        def after_prologue(self, stream) -> None:
            trace.append(("after_prologue", stream.name))

        def before_node(self, node, stream) -> None:
            trace.append(("before_node", node.uid, stream.name))

        def after_node(self, node, stream) -> None:
            trace.append(("after_node", node.uid, stream.name))

        def finish(self) -> None:
            trace.append(("finish",))

    monkeypatch.setattr(executors.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        executors.torch.cuda,
        "synchronize",
        lambda: trace.append(("synchronize",)),
    )

    params = Mock()
    buffers = Mock()
    events = Mock()
    logger = Mock()
    logger.isEnabledFor.return_value = False
    executor = DagExecutor(
        runtime=runtime,
        stages=Mock(),
        buffers=buffers,
        events=events,
        params=params,
        communication=Mock(),
        compute=Mock(),
        logger=logger,
    )

    override_calls: list[str] = []

    def stream_override(node, _logical_stream):
        override_calls.append(node.uid)
        return profile_compute if node.uid == "first" else profile_comm

    executor.run(
        dag,
        [first, second],
        inputs=None,
        labels=None,
        loss_buffer=[],
        stream_override=stream_override,
        timing_recorder=Recorder(),
    )

    assert override_calls == ["first", "second"]
    first_completion = next(
        entry[1]
        for entry in trace
        if len(entry) == 3
        and entry[0] == "record"
        and entry[2] == "profile-compute"
    )
    dependency_wait_index = trace.index(
        ("wait", "profile-comm", first_completion)
    )
    second_timing_index = trace.index(
        ("before_node", "second", "profile-comm")
    )
    assert dependency_wait_index < second_timing_index
    assert trace[-2:] == [("synchronize",), ("finish",)]
