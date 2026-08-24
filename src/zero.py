from dataclasses import dataclass
from typing import Any

from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    _ReachabilityIndex,
    _remove_edges,
    _remove_nodes_and_incident_edges,
    _topological_levels,
)
from .ordering import _serial_topological_order
from .state import LOG_LEVEL, create_logger

logger = create_logger("zero", LOG_LEVEL)


@dataclass(frozen=True)
class _ZeroLifetimeChains:
    gradient: list[list[str]]
    parameter: list[list[str]]


def _prune_zero_lifetime_metadata(dag: TrainingDAG) -> _ZeroLifetimeChains:
    """Prune redundant ZeRO param/grad lifetimes across same-bucket compute chains.

    Returns explicitly typed gradient and parameter chains used for pruning,
    each chain ordered by topo. Default compute order is finalized before this
    pass; the caller then adds inter-chain lifetime edges before materializing
    gather/reduce stream FIFO.
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
            e.dst_uid == dst_uid
            and e.dep_kind in ("data", "temporal")
            and e.src_uid in dag.nodes
            and e.dst_uid in dag.nodes
            and _is_compute(e.src_uid)
            and _is_compute(e.dst_uid)
            for e in dag.outgoing_edges(src_uid)
        )

    def _build_chains(
        candidates: list[str],
        topo_order: list[str],
        *,
        partition_key=None,
    ) -> list[list[str]]:
        chains: list[list[str]] = []
        by_bucket: dict[Any, list[str]] = {}
        for uid in candidates:
            key = _chain_bucket(uid)
            if partition_key is not None:
                key = (key, partition_key(uid))
            by_bucket.setdefault(key, []).append(uid)
        topo_idx = {uid: i for i, uid in enumerate(topo_order)}
        for bucket, uids in sorted(by_bucket.items(), key=lambda item: str(item[0])):
            uid_set = set(uids)
            links: dict[str, set[str]] = {uid: set() for uid in uids}
            for src_uid in uids:
                for e in dag.outgoing_edges(src_uid):
                    if (
                        e.dep_kind in ("data", "temporal")
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
                ordered_component = sorted(component, key=lambda u: topo_idx[u])
                chain = [ordered_component[0]]
                for next_uid in ordered_component[1:]:
                    if _has_direct_compute_dependency(chain[-1], next_uid):
                        chain.append(next_uid)
                    else:
                        chains.append(chain)
                        chain = [next_uid]
                chains.append(chain)
        return chains

    def _data_preds(uid: str) -> list[TrainingDAGEdge]:
        return dag.incoming_edges(uid, dep_kind="data")

    def _data_succs(uid: str) -> list[TrainingDAGEdge]:
        return dag.outgoing_edges(uid, dep_kind="data")

    def _all_gather_preds(uid: str) -> list[str]:
        return sorted(
            e.src_uid
            for e in dag.incoming_edges(uid, dep_kind="data")
            if e.src_uid in dag.nodes
            and dag.nodes[e.src_uid].node_kind == "ALL_GATHER_COMM"
        )

    def _reduce_scatter_succs(uid: str) -> list[str]:
        return sorted(
            e.dst_uid
            for e in dag.outgoing_edges(uid, dep_kind="data")
            if e.dst_uid in dag.nodes
            and dag.nodes[e.dst_uid].node_kind == "REDUCE_SCATTER_COMM"
        )

    def _prepare_remove_all_gather(
        ag_uid: str,
        compute_uid: str,
        bucket: Any,
    ) -> str | None:
        del compute_uid, bucket
        if ag_uid not in dag.nodes:
            return None
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
        return ag_uid

    has_grad_sync = any(
        n.node_kind in ("REDUCE_COMM", "REDUCE_SCATTER_COMM")
        for n in dag.nodes.values()
    )
    has_all_gathers = any(n.node_kind == "ALL_GATHER_COMM" for n in dag.nodes.values())

    grad_chains: list[list[str]] = []
    param_chains: list[list[str]] = []

    # Gradient lifetimes always span every backward microbatch contributing to
    # a bucket. Replicate lowering records those contributors on the bucket's
    # single reduce-scatter. Order directives have been applied by this point,
    # so normalize the contributor list to topological order, keep the full-grad
    # allocation on the first backward, and anchor the collective's one data
    # edge on the final backward. All other contributor edges are ordering-only.
    if has_grad_sync:
        topo_idx = {uid: index for index, uid in enumerate(topo)}
        reduce_scatters = sorted(
            (
                node
                for node in dag.nodes.values()
                if node.node_kind == "REDUCE_SCATTER_COMM"
            ),
            key=lambda node: topo_idx[node.uid],
        )
        covered_grad_uids: set[str] = set()
        contribution_edges_to_remove: list[TrainingDAGEdge] = []
        normalized_contribution_edges: list[TrainingDAGEdge] = []
        for rs_node in reduce_scatters:
            raw_contributors = rs_node.node_meta.get("bwd_uids")
            if raw_contributors is None:
                raw_contributors = [rs_node.node_meta.get("bwd_uid")]
            contributor_uids = [
                str(uid)
                for uid in raw_contributors
                if isinstance(uid, str)
                and uid in dag.nodes
                and _is_compute(uid)
            ]
            assert contributor_uids, (
                f"zero_lifetime_prune: reduce-scatter {rs_node.uid} has no "
                f"live backward contributors: {raw_contributors}"
            )
            assert len(contributor_uids) == len(set(contributor_uids)), (
                f"zero_lifetime_prune: reduce-scatter {rs_node.uid} has duplicate "
                f"contributors: {contributor_uids}"
            )
            chain = sorted(contributor_uids, key=lambda uid: topo_idx[uid])
            bucket = rs_node.node_meta.get("bucket_key")
            assert all(_chain_bucket(uid) == bucket for uid in chain), (
                f"zero_lifetime_prune: gradient chain has mixed buckets: "
                f"{[(uid, _chain_bucket(uid)) for uid in chain]}, rs_bucket={bucket}"
            )
            assert not covered_grad_uids.intersection(chain), (
                f"zero_lifetime_prune: backward contributors belong to multiple "
                f"reduce-scatters: {sorted(covered_grad_uids.intersection(chain))}"
            )
            covered_grad_uids.update(chain)

            for uid in chain:
                assert dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before"), (
                    f"zero_lifetime_prune: gradient contributor {uid} is missing "
                    "zero_alloc_full_grads_before"
                )
            for uid in chain[1:]:
                dag.nodes[uid].node_meta.pop("zero_alloc_full_grads_before", None)

            contributor_set = set(chain)
            contribution_edges = [
                edge
                for edge in dag.incoming_edges(rs_node.uid)
                if edge.src_uid in contributor_set
                and edge.dep_kind in ("data", "temporal")
            ]
            assert {edge.src_uid for edge in contribution_edges} == contributor_set, (
                f"zero_lifetime_prune: reduce-scatter {rs_node.uid} dependencies "
                f"do not cover contributors {chain}: {contribution_edges}"
            )
            contribution_edges_to_remove.extend(contribution_edges)
            for uid in chain:
                normalized_contribution_edges.append(
                    TrainingDAGEdge(
                        src_uid=uid,
                        dst_uid=rs_node.uid,
                        dep_kind=("data" if uid == chain[-1] else "temporal"),
                        tensor_name=None,
                    )
                )

            rs_node.node_meta["bwd_uid"] = chain[-1]
            rs_node.node_meta["bwd_uids"] = list(chain)
            grad_chains.append(chain)
            logger.debug(
                "normalized accumulated gradients: bucket=%s nodes=%s "
                "alloc_before=%s reduce_scatter_after=%s",
                bucket,
                chain,
                chain[0],
                chain[-1],
            )

        _remove_edges(dag, contribution_edges_to_remove)
        for edge in normalized_contribution_edges:
            dag.add_edge(edge)

        marked_grad_uids = set(
            _compute_uids_with(
                topo,
                lambda uid: bool(
                    dag.nodes[uid].node_meta.get("zero_alloc_full_grads_before")
                ),
            )
        )
        expected_roots = {chain[0] for chain in grad_chains}
        assert marked_grad_uids == expected_roots, (
            "zero_lifetime_prune: gradient allocation roots do not match "
            f"reduce-scatter chains; marked={sorted(marked_grad_uids)}, "
            f"expected={sorted(expected_roots)}"
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

        all_gathers_to_remove: set[str] = set()
        for chain in param_chains:
            bucket = _chain_bucket(chain[0])
            assert all(_chain_bucket(uid) == bucket for uid in chain), (
                f"zero_lifetime_prune: parameter chain has mixed buckets: "
                f"{[(uid, _chain_bucket(uid)) for uid in chain]}"
            )

            for uid in chain[1:]:
                for ag_uid in ag_by_compute[uid]:
                    prepared_uid = _prepare_remove_all_gather(ag_uid, uid, bucket)
                    if prepared_uid is not None:
                        all_gathers_to_remove.add(prepared_uid)

        _remove_nodes_and_incident_edges(dag, all_gathers_to_remove)

        for chain in param_chains:
            bucket = _chain_bucket(chain[0])
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

    return _ZeroLifetimeChains(
        gradient=grad_chains,
        parameter=param_chains,
    )


def _add_inter_chain_temporal_edges(
    dag: TrainingDAG,
    chains: _ZeroLifetimeChains,
) -> None:
    """Order disconnected same-bucket ZeRO lifetimes without overlap.

    Parameter chains: prev_tail compute frees full params, next root's
    all-gather predecessor materializes them. Edge: prev_tail -> ag_uid.

    Gradient chains: prev_tail's reduce-scatter successor frees full grads,
    next root compute allocates them. Edge: rs_uid -> next_root.
    """
    if not chains.gradient and not chains.parameter:
        return

    topo_levels = _topological_levels(dag)
    topo_order = _serial_topological_order(dag, topo_levels)
    topo_index = {uid: index for index, uid in enumerate(topo_order)}
    reachability = _ReachabilityIndex(dag)

    def _chain_bucket(uid: str) -> Any:
        return dag.nodes[uid].node_meta.get("bucket_key")

    def _all_gather_preds(uid: str) -> list[str]:
        return sorted(
            e.src_uid
            for e in dag.incoming_edges(uid, dep_kind="data")
            if e.src_uid in dag.nodes
            and dag.nodes[e.src_uid].node_kind == "ALL_GATHER_COMM"
        )

    def _reduce_scatter_succs(uid: str) -> list[str]:
        return sorted(
            e.dst_uid
            for e in dag.outgoing_edges(uid, dep_kind="data")
            if e.dst_uid in dag.nodes
            and dag.nodes[e.dst_uid].node_kind == "REDUCE_SCATTER_COMM"
        )

    def _order_intervals(
        bucket_chains: list[list[str]],
        *,
        bucket: Any,
        lifetime_kind: str,
        starts,
        ends,
    ) -> list[list[str]]:
        """Choose a stable interval order compatible with DAG reachability."""
        count = len(bucket_chains)
        if count < 2:
            return bucket_chains

        must_precede: dict[int, set[int]] = {
            index: set() for index in range(count)
        }
        indegree = [0] * count
        for before in range(count):
            for after in range(count):
                if before == after:
                    continue
                # ``before -> after`` would add every end(before) ->
                # start(after). If an after-start already reaches a
                # before-end, that direction is cyclic, so after must precede
                # before instead.
                if not any(
                    reachability.has_path(after_start, before_end)
                    for after_start in starts(bucket_chains[after])
                    for before_end in ends(bucket_chains[before])
                ):
                    continue
                if before not in must_precede[after]:
                    must_precede[after].add(before)
                    indegree[before] += 1

        def _preference(index: int) -> tuple[int, int]:
            root = bucket_chains[index][0]
            return (
                topo_levels.get(root, 0),
                topo_index.get(root, 0),
            )

        ready = sorted(
            (index for index, degree in enumerate(indegree) if degree == 0),
            key=_preference,
        )
        ordered_indices: list[int] = []
        while ready:
            index = ready.pop(0)
            ordered_indices.append(index)
            for successor in sorted(
                must_precede[index], key=_preference
            ):
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)
                    ready.sort(key=_preference)

        if len(ordered_indices) != count:
            cyclic = [
                {
                    "chain": bucket_chains[index],
                    "starts": starts(bucket_chains[index]),
                    "ends": ends(bucket_chains[index]),
                }
                for index, degree in enumerate(indegree)
                if degree > 0
            ]
            raise ValueError(
                "zero_lifetime_order: cyclic "
                f"{lifetime_kind} lifetime order for bucket={bucket!r}: "
                f"{cyclic}"
            )
        return [bucket_chains[index] for index in ordered_indices]

    def _maybe_add_edge(
        src_uid: str,
        dst_uid: str,
        *,
        bucket: Any,
        lifetime_kind: str,
    ) -> None:
        if reachability.has_path(src_uid, dst_uid):
            return
        if reachability.has_path(dst_uid, src_uid):
            raise ValueError(
                "zero_lifetime_order: contradictory "
                f"{lifetime_kind} lifetime for bucket={bucket!r}: required "
                f"{src_uid}->{dst_uid}, but the DAG already orders "
                f"{dst_uid}->{src_uid}"
            )
        dag.add_edge(TrainingDAGEdge(
            src_uid=src_uid,
            dst_uid=dst_uid,
            dep_kind="temporal",
            tensor_name=None,
        ))
        reachability.add_edge(src_uid, dst_uid)

    # Group chains by type and bucket, then add type-specific temporal edges
    # between consecutive same-bucket chains.
    param_by_bucket: dict[Any, list[list[str]]] = {}
    grad_by_bucket: dict[Any, list[list[str]]] = {}
    for chain in chains.parameter:
        bucket = _chain_bucket(chain[0])
        param_by_bucket.setdefault(bucket, []).append(chain)
    for chain in chains.gradient:
        bucket = _chain_bucket(chain[0])
        grad_by_bucket.setdefault(bucket, []).append(chain)

    for bucket, bucket_chains in param_by_bucket.items():
        bucket_chains = _order_intervals(
            bucket_chains,
            bucket=bucket,
            lifetime_kind="parameter",
            starts=lambda chain: _all_gather_preds(chain[0]),
            ends=lambda chain: [chain[-1]],
        )
        for prev_chain, next_chain in zip(bucket_chains, bucket_chains[1:]):
            for ag_uid in _all_gather_preds(next_chain[0]):
                _maybe_add_edge(
                    prev_chain[-1],
                    ag_uid,
                    bucket=bucket,
                    lifetime_kind="parameter",
                )

    for bucket, bucket_chains in grad_by_bucket.items():
        bucket_chains = _order_intervals(
            bucket_chains,
            bucket=bucket,
            lifetime_kind="gradient",
            starts=lambda chain: [chain[0]],
            ends=lambda chain: _reduce_scatter_succs(chain[-1]),
        )
        for prev_chain, next_chain in zip(bucket_chains, bucket_chains[1:]):
            for rs_uid in _reduce_scatter_succs(prev_chain[-1]):
                _maybe_add_edge(
                    rs_uid,
                    next_chain[0],
                    bucket=bucket,
                    lifetime_kind="gradient",
                )
