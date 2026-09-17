"""Two boundary-comm directives must not claim the same region."""
import pytest

import src.directives as directives
from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode

DEV = [0, 1]


def _dag(tags):
    dag = TrainingDAG()
    uids = [f"s{i}" for i in range(len(tags))]
    for i, (u, t) in enumerate(zip(uids, tags)):
        dag.add_node(TrainingDAGNode(
            uid=u, node_kind="COMPUTE", compute_subkind="FWD",
            tag={**t, "PASS": "F"}, device=list(DEV), stream="default_stream",
            node_meta={"bucket_key": u,
                       "a2a_boundary_after": ({"tensor_idx": 0}
                                              if i < len(tags) - 1 else None)}))
    for a, b in zip(uids, uids[1:]):
        dag.add_edge(TrainingDAGEdge(a, b, "data"))
    return dag


def test_ep_and_tp_on_the_same_region_is_rejected() -> None:
    """Without this, order decides the semantics and one order drops TP entirely.

    shard then shard_tensor gave 4 A2A_COMM and zero TP_COMM: the EP pass
    rewrites the region's edges to point at its own comm nodes, so the TP pass
    sees no compute-to-compute boundary left and inserts nothing. Wrong
    arithmetic, no diagnostic.
    """
    dag = _dag([{"PP": 0}, {"PP": 0, "EP": 0, "TP": 0}, {"PP": 0}])

    with pytest.raises(ValueError, match="both match compute node"):
        directives.apply_schedule_directives(dag, [
            {"op": "place", "filter": {"PP": 0}, "devices": DEV},
            {"op": "shard", "filter": {"EP": "*"}, "devices": DEV},
            {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEV},
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
        ])


def test_the_reverse_order_is_rejected_too() -> None:
    """The check runs before either pass, so it cannot itself be order-dependent."""
    dag = _dag([{"PP": 0}, {"PP": 0, "EP": 0, "TP": 0}, {"PP": 0}])

    with pytest.raises(ValueError, match="both match compute node"):
        directives.apply_schedule_directives(dag, [
            {"op": "place", "filter": {"PP": 0}, "devices": DEV},
            {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEV},
            {"op": "shard", "filter": {"EP": "*"}, "devices": DEV},
            {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
        ])


def test_disjoint_regions_still_compose() -> None:
    """EP on one region and TP on another is fine and must keep working."""
    dag = _dag([{"PP": 0, "EP": 0}, {"PP": 0}, {"PP": 0, "TP": 0}, {"PP": 0}])

    directives.apply_schedule_directives(dag, [
        {"op": "place", "filter": {"PP": 0}, "devices": DEV},
        {"op": "shard", "filter": {"EP": "*"}, "devices": DEV},
        {"op": "shard_tensor", "filter": {"TP": "*"}, "devices": DEV},
        {"op": "split", "filter": {}, "dim_name": "MB", "num_microbatches": 1},
    ])

    kinds = {n.node_kind for n in dag.nodes.values()}
    assert "A2A_COMM" in kinds and "TP_COMM" in kinds
