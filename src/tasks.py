from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dag import TrainingDAGNode


class TaskType(Enum):
    FWD = "forward"
    BWD = "backward"
    UPD = "update"
    BWD_I = "backward_input"
    BWD_W = "backward_weight"
    FWD_BWD = "forward_backward"
    SEND = "send"
    RECV = "recv"
    ALL_REDUCE = "all_reduce"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_GATHER = "all_gather"
    FWD_A2A = "forward_a2a"
    BWD_A2A = "backward_a2a"
    ORDER_DUMMY = "order_dummy"


def training_dag_task_type(node: "TrainingDAGNode") -> TaskType:
    if node.node_kind == "COMPUTE":
        if node.compute_subkind == "FWD":
            return TaskType.FWD
        if node.compute_subkind == "BWD":
            return TaskType.BWD
        if node.compute_subkind == "BWD_I":
            return TaskType.BWD_I
        if node.compute_subkind == "BWD_W":
            return TaskType.BWD_W
    if node.node_kind == "UPD":
        return TaskType.UPD
    if node.node_kind == "SEND_COMM":
        return TaskType.SEND
    if node.node_kind == "RECV_COMM":
        return TaskType.RECV
    if node.node_kind == "ALL_GATHER_COMM":
        return TaskType.ALL_GATHER
    if node.node_kind == "REDUCE_SCATTER_COMM":
        return TaskType.REDUCE_SCATTER
    if node.node_kind == "REDUCE_COMM":
        return TaskType.ALL_REDUCE
    if node.node_kind == "A2A_COMM":
        return TaskType.FWD_A2A if node.tag.get("PASS") == "F" else TaskType.BWD_A2A
    if node.node_kind == "ORDER_DUMMY":
        return TaskType.ORDER_DUMMY
    return TaskType.FWD
