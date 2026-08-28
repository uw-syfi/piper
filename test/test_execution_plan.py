# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import src.executors as executors
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
from src.execution_plan import (
    build_execution_plan,
)
from src.executors import DagExecutor
from src.ordering import assign_dispatch_levels
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


def test_plan_materializes_only_finalized_dag_edges() -> None:
    dag = _dag(
        _node("a"),
        _node("b", stream="comm"),
        _node("c"),
        _node("d", stream="comm"),
    )
    dag.add_edge(TrainingDAGEdge("a", "b", "data"))
    dag.add_edge(TrainingDAGEdge("b", "c", "temporal"))
    dag.add_edge(TrainingDAGEdge("a", "c", "data"))
    dag.add_edge(TrainingDAGEdge("c", "d", "data"))

    plan = _by_uid(dag, ["a", "b", "c", "d"])

    assert plan["a"].completion_predecessors == ()
    assert plan["b"].completion_predecessors == ("a",)
    assert plan["b"].reason_for("a") == "cross_stream_data"
    assert plan["c"].completion_predecessors == ("a", "b")
    assert plan["c"].reason_for("a") == "same_stream_data"
    assert plan["c"].reason_for("b") == "cross_stream_temporal"
    assert plan["d"].completion_predecessors == ("c",)
    assert plan["d"].reason_for("c") == "cross_stream_data"


def test_plan_does_not_invent_recv_consumer_guard() -> None:
    dag = _dag(
        _node("prior_compute"),
        _node("recv", stream="pp", kind="RECV_COMM", subkind=None),
        _node("consumer"),
    )
    dag.add_edge(TrainingDAGEdge("recv", "consumer", "data"))

    plan = _by_uid(dag, ["prior_compute", "recv", "consumer"])

    assert plan["recv"].completion_predecessors == ()
    assert plan["consumer"].completion_predecessors == ("recv",)




def test_dispatch_priority_never_overrides_an_encoded_dependency() -> None:
    dag = _dag(
        _node("recv", kind="RECV_COMM", subkind=None),
        _node("send", kind="SEND_COMM", subkind=None),
    )
    dag.add_edge(TrainingDAGEdge("recv", "send", "temporal"))

    assert [op.uid for op in build_execution_plan(dag)] == ["recv", "send"]






def test_plan_rejects_persisted_dispatch_level_edge_violation() -> None:
    dag = _dag(_node("source"), _node("destination"))
    dag.add_edge(TrainingDAGEdge("source", "destination", "data"))
    dag.nodes["source"].node_meta["dispatch_level"] = 1
    dag.nodes["destination"].node_meta["dispatch_level"] = 1

    with pytest.raises(ValueError, match="dispatch levels violate DAG edge"):
        build_execution_plan(dag)


def test_update_uses_explicit_backward_and_gradient_sync_edges() -> None:
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

    for predecessor in ("bwd", "all_reduce", "reduce_scatter"):
        dag.add_edge(TrainingDAGEdge(predecessor, "upd", "temporal"))

    plan = _by_uid(dag, ["bwd", "all_reduce", "reduce_scatter", "upd"])

    assert plan["upd"].completion_predecessors == (
        "bwd",
        "all_reduce",
        "reduce_scatter",
    )
    assert plan["upd"].reason_for("bwd") == "same_stream_temporal"
    assert plan["upd"].reason_for("all_reduce") == "cross_stream_temporal"
    assert plan["upd"].reason_for("reduce_scatter") == "cross_stream_temporal"






def test_directional_p2p_plan_does_not_invent_fifo_edges() -> None:
    lo_hi_1 = _node("lo_hi_1", stream="pp", kind="SEND_COMM", subkind=None)
    hi_lo = _node("hi_lo", stream="pp", kind="RECV_COMM", subkind=None)
    lo_hi_2 = _node("lo_hi_2", stream="pp", kind="SEND_COMM", subkind=None)
    lo_hi_1.node_meta["comm_group_kind"] = "pp_lo_hi"
    hi_lo.node_meta["comm_group_kind"] = "pp_hi_lo"
    lo_hi_2.node_meta["comm_group_kind"] = "pp_lo_hi"
    dag = _dag(lo_hi_1, hi_lo, lo_hi_2)

    plan = _by_uid(dag, ["lo_hi_1", "hi_lo", "lo_hi_2"])

    assert plan["hi_lo"].completion_predecessors == ()
    assert plan["lo_hi_2"].completion_predecessors == ()




def test_routed_collectives_use_declared_stream_and_explicit_edges() -> None:
    dispatch0 = _node(
        "dispatch0", stream="ep", kind="M_TO_N_COMM", subkind=None
    )
    combine = _node(
        "combine", stream="ep", kind="N_TO_M_COMM", subkind=None
    )
    dispatch1 = _node(
        "dispatch1", stream="ep", kind="M_TO_N_COMM", subkind=None
    )
    dag = _dag(dispatch0, combine, dispatch1)

    plan = _by_uid(dag, ["dispatch0", "combine", "dispatch1"])

    assert plan["dispatch0"].logical_stream == "ep"
    assert plan["combine"].logical_stream == "ep"
    assert plan["combine"].completion_predecessors == ()
    assert plan["dispatch1"].completion_predecessors == ()

    dag.add_edge(TrainingDAGEdge("dispatch0", "combine", "temporal"))
    temporal_plan = _by_uid(dag, ["dispatch0", "combine", "dispatch1"])
    reason = temporal_plan["combine"].reason_for("dispatch0")
    assert reason == "same_stream_temporal"

    dag.edges.clear()
    dag._rebuild_edge_indexes()
    dag.add_edge(TrainingDAGEdge("dispatch0", "combine", "data"))
    data_plan = _by_uid(dag, ["dispatch0", "combine", "dispatch1"])
    data_reason = data_plan["combine"].reason_for("dispatch0")
    assert data_reason == "same_stream_data"




def test_executor_uses_fifo_without_event_wait_for_same_physical_stream(
    monkeypatch,
) -> None:
    waits = []

    class FakeStream:
        def wait_event(self, event) -> None:
            waits.append(event)

    class FakeEvent:
        def record(self, _stream) -> None:
            pass

    stream = FakeStream()
    runtime = SimpleNamespace(
        streams={"default_stream": stream},
        default_stream=lambda: stream,
        stream_for_task=lambda _node: stream,
        stream_id=lambda node: node.stream,
        nvtx_push=lambda _label: None,
        nvtx_pop=lambda: None,
    )
    first = _node("first", kind="ORDER_DUMMY", subkind=None)
    second = _node("second", kind="ORDER_DUMMY", subkind=None)
    for node in (first, second):
        node.task_type = TaskType.ORDER_DUMMY
        node.batches = [SimpleNamespace(mb_idx=0)]
    dag = _dag(first, second)
    dag.add_edge(TrainingDAGEdge("first", "second", "temporal"))

    monkeypatch.setattr(executors.torch.cuda, "Event", FakeEvent)
    logger = Mock()
    logger.isEnabledFor.return_value = False
    executor = DagExecutor(
        runtime=runtime,
        stages=Mock(),
        buffers=Mock(),
        events=Mock(),
        params=Mock(),
        communication=Mock(),
        compute=Mock(),
        logger=logger,
    )
    executor._rf_enter = lambda _label: None
    executor._rf_exit = lambda _handle: None
    prepared = executor.prepare(dag, [first, second])

    executor.run(prepared, None, None, [], serialize_nodes=True)

    assert waits == []
