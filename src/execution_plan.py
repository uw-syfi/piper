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

"""Lower a :class:`TrainingDAG` into actor-local stream dependencies.

The CUDA executor and dependency replay both consume this
representation.  Keeping the lowering CUDA-free makes the dependency model
testable without a GPU and, more importantly, prevents replay from
quietly drifting away from the ordering implemented by the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .dag import TrainingDAG
from .ordering import _dispatch_order, execution_stream_domain


@dataclass(frozen=True)
class PlannedOperation:
    """One dispatched operation and the completions that gate its start.

    ``completion_predecessors`` and ``dependency_reasons`` are parallel
    tuples.  ``issue_predecessors`` constrain only the host dispatch order and
    never cause a CUDA event wait.
    """

    uid: str
    dispatch_index: int
    logical_stream: str
    completion_predecessors: tuple[str, ...]
    dependency_reasons: tuple[str, ...]
    issue_predecessors: tuple[str, ...]

    def reason_for(self, predecessor_uid: str) -> str:
        """Return the dependency reason for ``predecessor_uid``."""
        try:
            index = self.completion_predecessors.index(predecessor_uid)
        except ValueError as exc:
            raise KeyError(predecessor_uid) from exc
        return self.dependency_reasons[index]


def _validated_dispatch_order(
    dag: TrainingDAG,
    dispatch_order: Iterable[str] | None,
) -> list[str]:
    order = (
        list(_dispatch_order(dag))
        if dispatch_order is None
        else list(dispatch_order)
    )
    expected = set(dag.nodes)
    actual = set(order)
    if len(order) != len(actual):
        raise ValueError("execution-plan dispatch order contains duplicate node uids")
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "execution-plan dispatch order does not match DAG nodes: "
            f"missing={missing}, unknown={unknown}"
        )

    dispatch_index = {uid: index for index, uid in enumerate(order)}
    for edge in dag.edges:
        if edge.dep_kind not in {"data", "temporal", "issue"}:
            raise ValueError(
                f"unsupported dependency kind {edge.dep_kind!r} on "
                f"{edge.src_uid} -> {edge.dst_uid}"
            )
        if dispatch_index[edge.src_uid] >= dispatch_index[edge.dst_uid]:
            raise ValueError(
                "execution-plan dispatch order violates DAG dependency "
                f"{edge.src_uid} -> {edge.dst_uid} ({edge.dep_kind})"
            )
    return order


def build_execution_plan(
    dag: TrainingDAG,
    dispatch_order: Iterable[str] | None = None,
) -> list[PlannedOperation]:
    """Materialize the local CUDA completion dependencies for ``dag``.

    The finalized DAG is the sole source of inter-node dependencies. Same-
    stream edges remain visible for diagnostics and replay, while CUDA FIFO
    satisfies them without an additional event wait.
    """
    order = _validated_dispatch_order(dag, dispatch_order)
    dispatch_index = {uid: index for index, uid in enumerate(order)}

    incoming_edges: dict[str, list[tuple[str, str]]] = {
        uid: [] for uid in order
    }
    incoming_issue_edges: dict[str, list[str]] = {
        uid: [] for uid in order
    }
    for edge in dag.edges:
        if edge.dep_kind == "issue":
            incoming_issue_edges[edge.dst_uid].append(edge.src_uid)
            continue
        src = dag.nodes[edge.src_uid]
        dst = dag.nodes[edge.dst_uid]
        locality = (
            "cross_stream"
            if execution_stream_domain(src) != execution_stream_domain(dst)
            else "same_stream"
        )
        incoming_edges[edge.dst_uid].append(
            (edge.src_uid, f"{locality}_{edge.dep_kind}")
        )
    for uid in incoming_edges:
        incoming_edges[uid].sort(key=lambda item: dispatch_index[item[0]])
        incoming_issue_edges[uid].sort(key=dispatch_index.__getitem__)

    result: list[PlannedOperation] = []

    for index, uid in enumerate(order):
        node = dag.nodes[uid]
        logical_stream = execution_stream_domain(node)
        reasons_by_predecessor: dict[str, list[str]] = {}

        def add_dependency(predecessor_uid: str | None, reason: str) -> None:
            if predecessor_uid is None or predecessor_uid == uid:
                return
            reasons = reasons_by_predecessor.setdefault(predecessor_uid, [])
            if reason not in reasons:
                reasons.append(reason)

        for predecessor_uid, edge_reason in incoming_edges[uid]:
            add_dependency(predecessor_uid, edge_reason)

        ordered_predecessors = sorted(
            reasons_by_predecessor,
            key=lambda predecessor_uid: dispatch_index[predecessor_uid],
        )
        result.append(
            PlannedOperation(
                uid=uid,
                dispatch_index=index,
                logical_stream=logical_stream,
                completion_predecessors=tuple(ordered_predecessors),
                dependency_reasons=tuple(
                    "+".join(reasons_by_predecessor[predecessor_uid])
                    for predecessor_uid in ordered_predecessors
                ),
                issue_predecessors=tuple(incoming_issue_edges[uid]),
            )
        )

    return result


__all__ = [
    "PlannedOperation",
    "build_execution_plan",
]
