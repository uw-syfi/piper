# ===========================================================================================
# This file contains helper functions to implement the split backward pass.
#
# These helper functions were copied from the PyTorch pipelining backward implementation.
#
#   PyTorch Team,
#   "torch.distributed.pipelining._backward",
#   https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/_backward.py
#
# There are some minor implementation differences, but these functions are nearly identical.
#
# ===========================================================================================


from __future__ import annotations

import collections
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import torch
from torch.autograd.graph import GradientEdge, Node
from torch.nn import Parameter


def _get_grad_fn_or_grad_acc(t: torch.Tensor) -> Optional[Node]:
    """
    Return the autograd Node for a tensor:
      - non-leaf tensors: t.grad_fn
      - leaf tensors requiring grad (Parameters): AccumulateGrad node (created lazily)
    """
    if not t.requires_grad:
        return None
    if t.grad_fn is not None:
        return t.grad_fn

    # If t is a leaf tensor, we create a view to generate its AccumulateGrad node
    viewed = t.view_as(t)
    grad_fn = viewed.grad_fn
    if grad_fn is None:
        raise RuntimeError(
            "Tried to get grad accumulator but got None. "
            "Are you in a no-grad context?"
        )
    return grad_fn.next_functions[0][0]


def reverse_closure(
    roots: List[Node],
    target_nodes: Set[Node],
    reverse_edges: Dict[Node, List[Node]],
) -> Tuple[Set[Node], Set[Node]]:
    """
    From roots, follow reverse_edges to collect reachable nodes, but stop traversal
    when hitting any node in target_nodes. Returns the closure (set of reachable nodes) and
    the set of target nodes that were able to be visited.
    """
    closure: Set[Node] = set()
    visited_targets: Set[Node] = set()
    q: collections.deque[Node] = collections.deque()

    for node in roots:
        if node is None or node in closure:
            continue
        closure.add(node)
        q.append(node)

    while q:
        currNode = q.popleft()
        for node in reverse_edges.get(currNode, []):
            if node is None or node in closure:
                continue
            if node in target_nodes:
                visited_targets.add(node)
                continue
            closure.add(node)
            q.append(node)

    return closure, visited_targets


def construct_reverse_graph(roots: List[Node]) -> Dict[Node, List[Node]]:
    """
    Builds a reverse graph. In a backward graph, for each node X, 
    reverse_edges[X] is the list of nodes that X sends gradients to.
    """
    q: collections.deque[Node] = collections.deque()
    seen: Set[Node] = set()
    reverse_edges: Dict[Node, List[Node]] = collections.defaultdict(list)

    for node in roots:
        if node is not None and node not in seen:
            seen.add(node)
            q.append(node)

    while q:
        node = q.popleft()
        for fn, _ in node.next_functions:
            if fn is None:
                continue

            # fn -> node in reverse graph
            reverse_edges[fn].append(node)
            if fn not in seen:
                seen.add(fn)
                q.append(fn)

    return reverse_edges


def get_param_groups(
    inputs: List[Node],
    params: List[Node],
    reverse_edges: Dict[Node, List[Node]],
) -> List[Dict[str, Any]]:
    """
    Returns a list of parameter groups. Parameters are in the same group if they share the same 
    intermediate boundary nodes where the 'param-side' graph intersects the input closure.
    """
    inputs_closure, _ = reverse_closure(inputs, set(), reverse_edges)

    intermediate_to_param_group: Dict[Node, Dict[str, Any]] = {}
    solo_groups: List[Dict[str, Any]] = []

    for p in params:
        if p is None:
            continue

        _, intersected_nodes = reverse_closure([p], inputs_closure, reverse_edges)

        if not intersected_nodes:
            # if the parameter doesn't intersect inputs (unused or disconnected), then it 
            # we place it in its own group
            solo_groups.append({"params": {p}, "intermediates": []})
            continue

        # Merge into an existing group if any intersected intermediate node already has one
        group: Optional[Dict[str, Any]] = None
        for inter_node in intersected_nodes:
            if inter_node in intermediate_to_param_group:
                group = intermediate_to_param_group[inter_node]
                break

        if group is None:
            group = {"params": set(), "intermediates": []}

        group["params"].add(p)

        # remap all intersected intermediates to this group
        for inter_node in intersected_nodes:
            intermediate_to_param_group[inter_node] = group

    # add unique groups to param_groups
    param_groups: List[Dict[str, Any]] = []
    seen_ids: Set[int] = set()
    for params in intermediate_to_param_group.values():
        if id(params) in seen_ids:
            continue
        seen_ids.add(id(params))
        param_groups.append(params)

    for params in param_groups:
        inters = [k for k, v in intermediate_to_param_group.items() if v is params]
        # sort nodes to maintain deterministic order when iterating over them in backward_weight
        inters.sort(key=id)
        params["intermediates"] = inters

    param_groups.extend(solo_groups)

    return param_groups
