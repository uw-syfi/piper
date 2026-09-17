"""The executor's happens-before contract, stated instead of implied.

`DagExecutor.run` dispatches on exact `task_type` and finds its predecessors the
same way. A comm kind that is dispatched but missing from the consumer lookups is
therefore skipped in silence, and the consumer falls through to whatever its
`else` branch does -- reading the wrong inputs with no error. That happened
during this project: `TP_COMM` had a dispatch arm before the four predecessor
lookups knew about it.

These tests make the contract explicit. Adding a comm kind now fails here until
it is classified and, if it feeds compute, wired into the lookups.
"""
import inspect
import re

import src.executors as ex
from src.executors import DagExecutor
from src.tasks import TaskType

# Task types that run model arithmetic rather than communication.
_COMPUTE = {
    TaskType.FWD, TaskType.BWD, TaskType.BWD_I, TaskType.BWD_W,
    TaskType.UPD, TaskType.FWD_BWD,
}

# Comm kinds whose output a COMPUTE node consumes. Each must be awaited by the
# consumer arms, or the consumer reads a tensor that is still being written.
_FEEDS_COMPUTE = {
    TaskType.RECV,
    TaskType.ALL_GATHER,
    TaskType.FWD_A2A,
    TaskType.BWD_A2A,
    TaskType.FWD_TP_ALL_REDUCE,
    TaskType.BWD_TP_ALL_REDUCE,
    TaskType.FWD_RING_EXCHANGE,
    TaskType.BWD_RING_EXCHANGE,
}

# Comm kinds that feed only UPD or nothing, so no COMPUTE arm needs to wait.
_FEEDS_NO_COMPUTE = {
    TaskType.SEND,
    TaskType.ALL_REDUCE,
    TaskType.REDUCE_SCATTER,
    TaskType.ORDER_DUMMY,
}


def test_every_task_type_is_classified() -> None:
    """A new task type must be classified here, not silently defaulted."""
    comm = set(TaskType) - _COMPUTE

    assert comm == _FEEDS_COMPUTE | _FEEDS_NO_COMPUTE, (
        f"unclassified: {sorted(t.name for t in comm - _FEEDS_COMPUTE - _FEEDS_NO_COMPUTE)}; "
        f"stale: {sorted(t.name for t in (_FEEDS_COMPUTE | _FEEDS_NO_COMPUTE) - comm)}"
    )


def test_kinds_that_feed_compute_are_awaited_by_consumers() -> None:
    """Being dispatched is not enough; consumers must look for it too."""
    awaited = (
        {TaskType.RECV, TaskType.ALL_GATHER}
        | set(ex._FWD_BOUNDARY_COMM_TASKS)
        | set(ex._BWD_BOUNDARY_COMM_TASKS)
    )

    missing = _FEEDS_COMPUTE - awaited
    assert not missing, (
        f"{sorted(t.name for t in missing)} produce tensors a COMPUTE node reads, "
        f"but no consumer arm waits on their event. The consumer will take its "
        f"fallback branch and read the wrong inputs, silently."
    )


def test_every_task_type_has_a_dispatch_arm() -> None:
    """A node kind with no arm is executed as a no-op."""
    src = inspect.getsource(DagExecutor.run)
    armed = set(re.findall(r"case TaskType\.(\w+)", src))

    # FWD_BWD is declared in TaskType but never produced by training_dag_task_type.
    expected = {t.name for t in TaskType} - {TaskType.FWD_BWD.name}
    assert expected <= armed, f"no dispatch arm for {sorted(expected - armed)}"
