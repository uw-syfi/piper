"""A single-device TP group has nothing to all-reduce."""
import pytest

import src.directives as directives
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode


def _region_dag(device):
    dag = TrainingDAG()
    tags = [{"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}]
    uids = ["s0", "s1", "s2"]
    for i, (u, t) in enumerate(zip(uids, tags)):
        dag.add_node(TrainingDAGNode(
            uid=u, node_kind="COMPUTE", compute_subkind="FWD",
            tag={**t, "PASS": "F"}, device=list(device), stream="default_stream",
            node_meta={"bucket_key": u,
                       "a2a_boundary_after": {"tensor_idx": 0} if i < 2 else None}))
    for a, b in zip(uids, uids[1:]):
        dag.add_edge(TrainingDAGEdge(a, b, "data"))
    return dag


def test_single_device_tp_group_is_rejected() -> None:
    """Otherwise it fails at runtime with a torch.distributed error.

    A one-device group leaves dp_degree and pp_degree at 1, so
    _join_process_groups never initializes anything and ep_group stays None; the
    all-reduce then runs on an uninitialized default group, deep in the executor,
    with a message that says nothing about the schedule.
    """
    dag = _region_dag([0])

    with pytest.raises(ValueError, match="at least two distinct devices"):
        directives._insert_tp_all_reduce_comm_nodes(dag, [{"TP": 0}], [0])


def test_repeated_device_is_rejected_too() -> None:
    """[0, 0] is two entries but one device."""
    dag = _region_dag([0, 0])

    with pytest.raises(ValueError, match="at least two distinct devices"):
        directives._insert_tp_all_reduce_comm_nodes(dag, [{"TP": 0}], [0, 0])
