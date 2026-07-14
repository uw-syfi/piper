from typing import Any

from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    _has_path,
    _remove_node_and_incident_edges,
    _topological_levels,
)
from .ordering import _serial_topological_order
from .state import LOG_LEVEL, create_logger

logger = create_logger("zero", LOG_LEVEL)


def _prune_zero_lifetime_metadata(dag: TrainingDAG) -> list[list[str]]:
    """Prune redundant ZeRO param/grad lifetimes across same-bucket compute chains.

    Returns the union of gradient and parameter chains used for pruning, one
    list per chain ordered by topo. The caller passes these to
    ``_add_inter_chain_temporal_edges`` *after*
    ``resolve_total_order_per_stream`` runs, so the inter-chain check sees the
    post-stream-serialization topo levels for the gather_stream and
    reduce_stream nodes.
    """
    topo_levels = _topological_levels(dag)
    topo = _serial_topological_order(dag, topo_levels)

    def _is_compute(uid: str) -> bool:
        node = dag.nodes[uid]
        return (
            node.node_kind == "COMPUTE"
            and node.compute_subkind in ("FWD", "BWD", "BWD_I", "BWD_W")
        )

    def _chain_bucket(uid: str) -> Any:
        return dag.nodes[uid].node_meta.get("bucket_key")

    def _compute_uids_with(topo_order: list[str], predicate) -> list[str]:
        return [
            uid for uid in topo_order
            if uid in dag.nodes and _is_compute(uid) and predicate(uid)
        ]

    def _has_direct_compute_dependency(src_uid: str, dst_uid: str) -> bool:
        return any(
            e.src_uid == src_uid
            and e.dst_uid == dst_uid
            and e.dep_kind in ("data", "temporal")
            and e.src_uid in dag.nodes
            and e.dst_uid in dag.nodes
            and _is_compute(e.src_uid)
            and _is_compute(e.dst_uid)
            for e in dag.edges
        )

    def _build_chains(candidates: list[str], topo_order: list[str]) -> list[list[str]]:
        chains: list[list[str]] = []
        by_bucket: dict[Any, list[str]] = {}
        for uid in candidates:
            by_bucket.setdefault(_chain_bucket(uid), []).append(uid)
        topo_idx = {uid: i for i, uid in enumerate(topo_order)}
        for bucket, uids in sorted(by_bucket.items(), key=lambda item: str(item[0])):
            uid_set = set(uids)
            links: dict[str, set[str]] = {uid: set() for uid in uids}
            for e in dag.edges:
                if (
                    e.dep_kind in ("data", "temporal")
                    and e.src_uid in uid_set
                    and e.dst_uid in uid_set
                    and _is_compute(e.src_uid)
                    and _is_compute(e.dst_uid)
                ):
                    links[e.src_uid].add(e.dst_uid)
                    links[e.dst_uid].add(e.src_uid)

            seen: set[str] = set()
            for uid in uids:
                if uid in seen:
                    continue
                component: list[str] = []
                stack = [uid]
                seen.add(uid)
                while stack:
                    cur = stack.pop()
                    component.append(cur)
                    for nxt in sorted(links[cur], key=lambda u: topo_idx[u]):
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
                chains.append(sorted(component, key=lambda u: topo_idx[u]))
        return chains

    def _data_preds(uid: str) -> list[TrainingDAGEdge]:
        return [
            e for e in list(dag.edges)
            if e.dep_kind == "data" and e.dst_uid == uid
        ]

    def _data_succs(uid: str) -> list[TrainingDAGEdge]:
        return [
            e for e in list(dag.edges)
            if e.dep_kind == "data" and e.src_uid == uid
        ]

    def _all_gather_preds(uid: str) -> list[str]:
        return sorted(
            e.src_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.dst_uid == uid
            and e.src_uid in dag.nodes
            and dag.nodes[e.src_uid].node_kind == "ALL_GATHER_COMM"
        )

    def _reduce_scatter_succs(uid: str) -> list[str]:
        return sorted(
            e.dst_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.src_uid == uid
            and e.dst_uid in dag.nodes
            and dag.nodes[e.dst_uid].node_kind == "REDUCE_SCATTER_COMM"
        )

    def _remove_all_gather(ag_uid: str, compute_uid: str, bucket: Any) -> None:
        del compute_uid, bucket
        if ag_uid not in dag.nodes:
            return
        in_ag = _data_preds(ag_uid)
        out_ag = _data_succs(ag_uid)
        for ie in in_ag:
            for oe in out_ag:
                dag.add_edge(
                    TrainingDAGEdge(
                        src_uid=ie.src_uid,
                        dst_uid=oe.dst_uid,
                        dep_kind="data",
                        tensor_name=(oe.tensor_name if oe.tensor_name is not None else ie.tensor_name),
                    )
                )
        _remove_node_and_incident_edges(dag, ag_uid)

    def _remove_reduce_scatter(rs_uid: str, compute_uid: str, bucket: Any) -> None:
        del compute_uid, bucket
        if rs_uid not in dag.nodes:
            return
        _remove_node_and_incident_edges(dag, rs_uid)

    has_grad_sync = any(
        n.node_kind in ("REDUCE_COMM", "REDUCE_SCATTER_COMM")
        for n in dag.nodes.values()
    )
    has_all_gathers = any(n.node_kind == "ALL_GATHER_COMM" for n in dag.nodes.values())

    grad_chains: list[list[str]] = []
    param_chains: list[list[str]] = []

    # Gradient lifetimes: keep one full-grad allocation and one reduce-scatter
    # per directly-connected same-bucket gradient chain.
    if has_grad_sync:
        grad_candidates = _compute_uids_with(
            topo,
            lambda uid: bool(dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before"))
        )
        if grad_candidates:
            grad_chains = _build_chains(grad_candidates, topo)
            covered = {uid for chain in grad_chains for uid in chain}
            assert covered == set(grad_candidates), (
                "zero_lifetime_prune: gradient mode did not assign every grad-allocation "
                f"compute node to a chain; missing={sorted(set(grad_candidates) - covered)}"
            )

            for chain in grad_chains:
                bucket = _chain_bucket(chain[0])
                assert all(_chain_bucket(uid) == bucket for uid in chain), (
                    f"zero_lifetime_prune: gradient chain has mixed buckets: "
                    f"{[(uid, _chain_bucket(uid)) for uid in chain]}"
                )
                logger.debug(
                    "pruned zero lifetime: bucket=%s nodes=%s keep_alloc_for=%s keep_reduce_scatter_for=%s",
                    bucket,
                    chain,
                    chain[0],
                    chain[-1],
                )

                for uid in chain:
                    rs_uids = _reduce_scatter_succs(uid)
                    assert rs_uids, (
                        f"zero_lifetime_prune: gradient chain node {uid} has "
                        "zero_alloc_full_grads_before but no outgoing reduce-scatter"
                    )

                for uid in chain[1:]:
                    dag.nodes[uid].node_meta.pop("zero_alloc_full_grads_before", None)
                for uid in chain[:-1]:
                    for rs_uid in _reduce_scatter_succs(uid):
                        _remove_reduce_scatter(rs_uid, uid, bucket)

                assert dag.nodes[chain[0]].node_meta.get("zero_alloc_full_grads_before"), (
                    f"zero_lifetime_prune: gradient chain root {chain[0]} lost grad allocation"
                )
                for prev_uid, uid in zip(chain, chain[1:]):
                    assert _has_direct_compute_dependency(prev_uid, uid), (
                        f"zero_lifetime_prune: gradient chain is not directly linked: "
                        f"{prev_uid} does not have a direct compute dependency to {uid}; chain={chain}"
                    )
                    assert not dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before"), (
                        f"zero_lifetime_prune: non-root gradient chain node {uid} still allocates grads"
                    )
                live_rs = [
                    (uid, rs_uid)
                    for uid in chain
                    for rs_uid in _reduce_scatter_succs(uid)
                ]
                assert live_rs, (
                    f"zero_lifetime_prune: gradient chain has no live reduce-scatter; chain={chain}"
                )
                assert all(uid == chain[-1] for uid, _rs_uid in live_rs), (
                    f"zero_lifetime_prune: only chain tail {chain[-1]} should keep reduce-scatter, "
                    f"but live reduce-scatters are {live_rs}"
                )

    # Parameter lifetimes: keep one all-gather and one full-param free per
    # directly-connected same-bucket parameter chain. This pass intentionally
    # does not change grad allocation metadata or remove reduce-scatters.
    if has_all_gathers:
        topo = _serial_topological_order(dag, _topological_levels(dag))
        ag_by_compute = {uid: _all_gather_preds(uid) for uid in topo if uid in dag.nodes and _is_compute(uid)}
        ag_by_compute = {uid: ags for uid, ags in ag_by_compute.items() if ags}

        candidates = _compute_uids_with(topo, lambda uid: uid in ag_by_compute)
        param_chains = _build_chains(candidates, topo)

        covered = {uid for chain in param_chains for uid in chain}
        assert covered == set(candidates), (
            "zero_lifetime_prune: parameter mode did not assign every all-gathered "
            f"compute node to a chain; missing={sorted(set(candidates) - covered)}"
        )

        for chain in param_chains:
            bucket = _chain_bucket(chain[0])
            assert all(_chain_bucket(uid) == bucket for uid in chain), (
                f"zero_lifetime_prune: parameter chain has mixed buckets: "
                f"{[(uid, _chain_bucket(uid)) for uid in chain]}"
            )

            for uid in chain[1:]:
                for ag_uid in ag_by_compute[uid]:
                    _remove_all_gather(ag_uid, uid, bucket)

            tail = chain[-1]
            for uid in chain:
                dag.nodes[uid].node_meta.pop("zero_free_full_params_after", None)
                for rs_uid in _reduce_scatter_succs(uid):
                    dag.nodes[rs_uid].node_meta.pop("zero_free_full_params_after", None)
            dag.nodes[tail].node_meta["zero_free_full_params_after"] = True

            remaining_ag = {uid: _all_gather_preds(uid) for uid in chain}
            assert remaining_ag[chain[0]], (
                f"zero_lifetime_prune: parameter chain root {chain[0]} lost all-gather"
            )
            for prev_uid, uid in zip(chain, chain[1:]):
                assert _has_direct_compute_dependency(prev_uid, uid), (
                    f"zero_lifetime_prune: parameter chain is not directly linked: "
                    f"{prev_uid} does not have a direct compute dependency to {uid}; chain={chain}"
                )
                assert not remaining_ag[uid], (
                    f"zero_lifetime_prune: non-root parameter chain node {uid} still "
                    f"has all-gather predecessors {remaining_ag[uid]}; chain={chain}"
                )
            assert dag.nodes[tail].node_meta.get("zero_free_full_params_after"), (
                f"zero_lifetime_prune: parameter chain tail {tail} should free params"
            )
            for uid in chain[:-1]:
                assert not dag.nodes[uid].node_meta.get("zero_free_full_params_after"), (
                    f"zero_lifetime_prune: non-tail parameter chain node {uid} still frees params"
                )
            stale_rs_param_frees = [
                rs_uid
                for uid in chain
                for rs_uid in _reduce_scatter_succs(uid)
                if dag.nodes[rs_uid].node_meta.get("zero_free_full_params_after")
            ]
            assert not stale_rs_param_frees, (
                f"zero_lifetime_prune: reduce-scatters still own param free metadata: "
                f"{stale_rs_param_frees}; chain={chain}"
            )

    return grad_chains + param_chains


def _add_inter_chain_temporal_edges(
    dag: TrainingDAG,
    chains: list[list[str]],
) -> None:
    """Force later same-bucket ZeRO chains to wait for earlier bucket frees.

    Parameter chains: prev_tail compute frees full params, next root's
    all-gather predecessor materializes them. Edge: prev_tail -> ag_uid.

    Gradient chains: prev_tail's reduce-scatter successor frees full grads,
    next root compute allocates them. Edge: rs_uid -> next_root.
    """
    if not chains:
        return

    topo_levels = _topological_levels(dag)

    def _chain_bucket(uid: str) -> Any:
        return dag.nodes[uid].node_meta.get("bucket_key")

    def _all_gather_preds(uid: str) -> list[str]:
        return sorted(
            e.src_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.dst_uid == uid
            and e.src_uid in dag.nodes
            and dag.nodes[e.src_uid].node_kind == "ALL_GATHER_COMM"
        )

    def _reduce_scatter_succs(uid: str) -> list[str]:
        return sorted(
            e.dst_uid
            for e in dag.edges
            if e.dep_kind == "data"
            and e.src_uid == uid
            and e.dst_uid in dag.nodes
            and dag.nodes[e.dst_uid].node_kind == "REDUCE_SCATTER_COMM"
        )

    def _maybe_add_edge(src_uid: str, dst_uid: str) -> None:
        if topo_levels.get(src_uid, 0) < topo_levels.get(dst_uid, 0):
            return
        if _has_path(dag, src_uid, dst_uid):
            return
        if _has_path(dag, dst_uid, src_uid):
            return
        dag.add_edge(TrainingDAGEdge(
            src_uid=src_uid,
            dst_uid=dst_uid,
            dep_kind="temporal",
            tensor_name=None,
        ))

    # Group chains by type and bucket, then add type-specific temporal edges
    # between consecutive same-bucket chains.
    param_by_bucket: dict[Any, list[list[str]]] = {}
    grad_by_bucket: dict[Any, list[list[str]]] = {}
    for chain in chains:
        bucket = _chain_bucket(chain[0])
        if _all_gather_preds(chain[0]):
            param_by_bucket.setdefault(bucket, []).append(chain)
        if dag.nodes[chain[0]].node_meta.get("zero_alloc_full_grads_before"):
            grad_by_bucket.setdefault(bucket, []).append(chain)

    for bucket_chains in param_by_bucket.values():
        bucket_chains.sort(key=lambda c: topo_levels.get(c[0], 0))
        for prev_chain, next_chain in zip(bucket_chains, bucket_chains[1:]):
            for ag_uid in _all_gather_preds(next_chain[0]):
                _maybe_add_edge(prev_chain[-1], ag_uid)

    for bucket_chains in grad_by_bucket.values():
        bucket_chains.sort(key=lambda c: topo_levels.get(c[0], 0))
        for prev_chain, next_chain in zip(bucket_chains, bucket_chains[1:]):
            for rs_uid in _reduce_scatter_succs(prev_chain[-1]):
                _maybe_add_edge(rs_uid, next_chain[0])
