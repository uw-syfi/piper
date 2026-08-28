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
    FWD_M_TO_N = "forward_m_to_n"
    BWD_M_TO_N = "backward_m_to_n"
    FWD_N_TO_M = "forward_n_to_m"
    BWD_N_TO_M = "backward_n_to_m"
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
    if node.node_kind == "M_TO_N_COMM":
        return (
            TaskType.FWD_M_TO_N
            if node.tag.get("PASS") == "F"
            else TaskType.BWD_M_TO_N
        )
    if node.node_kind == "N_TO_M_COMM":
        return (
            TaskType.FWD_N_TO_M
            if node.tag.get("PASS") == "F"
            else TaskType.BWD_N_TO_M
        )
    if node.node_kind == "ORDER_DUMMY":
        return TaskType.ORDER_DUMMY
    return TaskType.FWD
