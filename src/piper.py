import time
import json
from pathlib import Path
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping, Sequence

import ray
import torch
import torch.fx as fx
from torch._dynamo.backends.registry import register_backend

from .fx import (
    EP_COMBINE_PAYLOAD,
    EP_DISPATCH_COUNTS,
    EP_DISPATCH_PAYLOAD,
    PIPER_ANNOTATIONS_META_KEY,
    PIPER_EP_BOUNDARY_META_KEY,
    split_gm_by_annotations,
)
from .dag import (
    TrainingDAG,
    TrainingDAGEdge,
    TrainingDAGNode,
    build_training_dag,
)
from .directives import (
    _apply_order_directive,
    _apply_split_backward_stencil,
    _apply_split_directive,
    _bucket_matched_fwd_nodes,
    _parse_order_directive,
    _validate_schedule_tags_exist,
    _validate_split_backward_order_stencil,
    apply_schedule_directives,
)
from .ordering import (
    _serial_topological_order,
    resolve_total_order_per_stream,
)
from .profile_statistics import summarize_timing_samples
from .profiling import (
    NORMAL_PROFILE_MODE,
    DEFAULT_NORMAL_SCOPE,
    INTERFERENCE_FREE_PROFILE_MODE,
    ProfileConfig,
    ProfiledTrainingPlan,
    estimate_profile_time,
    merge_actor_profiles,
    normalize_profile_mode,
)
from .state import LOG_LEVEL, create_logger, piper_metadata
from .visualization import (
    log_training_dag_dependencies,
    print_training_dag_order,
    render_training_dag,
)
_MAX_KINETO_CAPTURE_LOSS_RATE = 1e-4
_MAX_KINETO_CPU_LAUNCH_LOSS_RATE = 1e-4
from .zero import _add_inter_chain_temporal_edges, _prune_zero_lifetime_metadata

logger = create_logger("piper_backend", LOG_LEVEL)
_ANNOTATION_STACK: list[dict[str, int]] = []
_ANNOTATION_COUNTS: dict[str, int] = {}
_ANNOTATION_UID = 0


@dataclass(frozen=True)
class TimingResult:
    """A fixed-size set of hardware step and peak-memory samples."""

    step_samples_s: tuple[float, ...]
    peak_memory_samples_by_rank: dict[int, tuple[int, ...]]


def _reset_annotation_state() -> None:
    global _ANNOTATION_UID
    _ANNOTATION_STACK.clear()
    _ANNOTATION_COUNTS.clear()
    _ANNOTATION_UID = 0


@contextmanager
def annotate(name: str, *, index: int | None = None) -> Iterator[dict[str, int]]:
    """Annotate traced model code with a Piper schedule tag.

    Piper normally assigns the integer index for each tag name automatically
    in the order annotation scopes are entered during tracing.  ``index`` may
    be supplied when two distinct Python scopes are fragments of the same
    logical schedule region.  Each scope still receives a unique ``uid`` so
    FX splitting can preserve its lexical boundaries.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"piper.annotate requires a non-empty string tag name, got {name!r}")
    if index is not None and (
        isinstance(index, bool) or not isinstance(index, int) or index < 0
    ):
        raise ValueError(
            "piper.annotate index must be a non-negative int or None, "
            f"got {index!r}"
        )

    global _ANNOTATION_UID
    next_index = _ANNOTATION_COUNTS.get(name, 0)
    if index is None:
        index = next_index
    _ANNOTATION_COUNTS[name] = max(next_index, index + 1)
    uid = _ANNOTATION_UID
    _ANNOTATION_UID += 1

    annotation = {"name": name, "index": int(index), "uid": int(uid)}
    _ANNOTATION_STACK.append(annotation)
    fx_metadata_stack = tuple(dict(item) for item in _ANNOTATION_STACK)
    try:
        with fx.traceback.annotate({
            PIPER_ANNOTATIONS_META_KEY: fx_metadata_stack,
            "name": name,
            "index": int(index),
        }):
            yield annotation
    finally:
        popped = _ANNOTATION_STACK.pop()
        if popped is not annotation:
            raise RuntimeError("piper.annotate stack corrupted during tracing")


def _mark_ep_boundary(tensor: torch.Tensor, role: str) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "Piper routed boundary markers require a torch.Tensor, "
            f"got {type(tensor)!r}"
        )
    with fx.traceback.annotate({PIPER_EP_BOUNDARY_META_KEY: role}):
        return torch.ops.aten.alias.default(tensor)


def mark_dispatch_payload(tensor: torch.Tensor) -> torch.Tensor:
    """Mark the sole dynamic payload entering a routed model region.

    ``tensor`` must be a rank-2 ``[rows, hidden]`` tensor whose rows are
    packed as consecutive route blocks in the order documented by the
    accompanying :func:`mark_dispatch_counts` tensor. Its row count must equal
    the sum of those counts. Piper transports these rows opaquely; model code
    inside the routed region owns any expert-local regrouping or
    kernel-specific padding.

    Tensor values produced before the routed region and consumed afterward may
    bypass it in the FX graph. Piper records their actual producer/consumer
    slots and carries them on a separate dense-to-dense path; they must not be
    consumed or produced by the routed model region.

    The Piper user is responsible for packing each row into the semantically
    correct route block. Piper can validate the payload shape and total row
    count, but cannot infer a row's intended route from its tensor values.
    """
    return _mark_ep_boundary(tensor, EP_DISPATCH_PAYLOAD)


def mark_dispatch_counts(tensor: torch.Tensor) -> torch.Tensor:
    """Mark the sole routing descriptor entering a routed model region.

    ``tensor`` must be a nonnegative, non-differentiable integer tensor shaped
    ``[source_blocks, local_routes]``. Entry ``[source, route]`` gives the
    number of consecutive dispatch-payload rows for that source block and
    route. The route-axis order is the model's logical route order; after
    sharding, Piper assigns equal contiguous route ranges to shard devices in
    ascending global actor-rank order. Unsorted ``shard.devices`` lists are
    rejected rather than silently changing ownership.

    Piper uses these counts only to derive and validate transport metadata. It
    does not interpret routes as experts, regroup rows into expert-local order,
    or apply expert-kernel alignment. Both this descriptor and the marked
    dispatch payload must contribute to the marked combine payload. The Piper
    user is responsible for ensuring that every count describes the actual
    consecutive row block packed into the dispatch payload; Piper cannot prove
    that semantic correspondence from tensor values.
    """
    return _mark_ep_boundary(tensor, EP_DISPATCH_COUNTS)


def mark_combine_payload(tensor: torch.Tensor) -> torch.Tensor:
    """Mark the sole dynamic payload leaving a routed model region.

    ``tensor`` must contain exactly one result row for every dispatch row and
    must restore the source-major, local-route row order received by the
    routed region. Its dtype and trailing shape must match the dispatch
    payload. Piper reverses the transport from saved split metadata and cannot
    recover an ordering changed by model-side expert computation. This is the
    only dynamic activation produced by the routed region; tensor values that
    bypass it retain their original FX producer/consumer identity instead.

    The Piper user is responsible for restoring the semantic identity of every
    row to the exact source-major, local-route order received by the region.
    Piper can validate shapes, dtypes, dependencies, and row totals, but cannot
    prove semantic row identity from tensor values. Models should test this
    contract with identifiable rows.
    """
    return _mark_ep_boundary(tensor, EP_COMBINE_PAYLOAD)


def _split_global_training_dag_by_pp_rank(
    training_dag: TrainingDAG,
    *,
    merge_same_device_components: bool = False,
) -> list[TrainingDAG]:
    """Split the global DAG into per-device-set disconnected DAGs."""
    # SEND/RECV pairs intentionally have no edge between them, so cross-rank
    # placement dependencies separate into disconnected local components here.
    undirected: dict[str, set[str]] = {uid: set() for uid in training_dag.nodes}
    for e in training_dag.edges:
        if e.src_uid in undirected and e.dst_uid in undirected:
            undirected[e.src_uid].add(e.dst_uid)
            undirected[e.dst_uid].add(e.src_uid)

    components: list[set[str]] = []
    seen: set[str] = set()
    for uid in training_dag.nodes:
        if uid in seen:
            continue
        comp: set[str] = set()
        stack = [uid]
        seen.add(uid)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for nxt in undirected.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(comp)

    # Validate each component is device-homogeneous and component device-sets are distinct.
    comp_device_keys: list[tuple[int, ...]] = []
    for ci, comp in enumerate(components):
        device_keys = {
            tuple(sorted(node.device)) for uid in comp for node in [training_dag.nodes[uid]] if node.device is not None
        }
        if not device_keys:
            raise ValueError(f"component[{ci}] has no device assignment after P2P split")
        if len(device_keys) != 1:
            raise ValueError(
                f"component[{ci}] is not device-homogeneous; device sets present: {sorted(device_keys)}"
            )
        comp_device_keys.append(next(iter(device_keys)))
    if (
        len(set(comp_device_keys)) != len(comp_device_keys)
        and merge_same_device_components
    ):
        merged_by_device: dict[tuple[int, ...], set[str]] = {}
        for comp, device_key in zip(components, comp_device_keys):
            merged_by_device.setdefault(device_key, set()).update(comp)
        components = list(merged_by_device.values())
        comp_device_keys = list(merged_by_device)
    elif len(set(comp_device_keys)) != len(comp_device_keys):
        raise ValueError(
            f"expected distinct device sets across split components, got {comp_device_keys}"
        )

    # Materialize each component as a standalone TrainingDAG.
    subdags: list[TrainingDAG] = []
    for comp in components:
        sub = TrainingDAG()
        sub.prologue_profile_times_by_mode = {
            mode: dict(profiles)
            for mode, profiles in training_dag.prologue_profile_times_by_mode.items()
        }
        for uid in training_dag.nodes:
            if uid not in comp:
                continue
            n = training_dag.nodes[uid]
            sub.add_node(
                TrainingDAGNode(
                    uid=n.uid,
                    node_kind=n.node_kind,
                    compute_subkind=n.compute_subkind,
                    tag=dict(n.tag),
                    device=(None if n.device is None else list(n.device)),
                    stream=n.stream,
                    node_meta=dict(n.node_meta),
                    profile_times_by_mode={
                        mode: dict(profiles)
                        for mode, profiles in n.profile_times_by_mode.items()
                    },
                )
            )
        for e in training_dag.edges:
            if e.src_uid in comp and e.dst_uid in comp:
                sub.add_edge(
                    TrainingDAGEdge(
                        src_uid=e.src_uid,
                        dst_uid=e.dst_uid,
                        dep_kind=e.dep_kind,
                        tensor_name=e.tensor_name,
                    )
                )
        subdags.append(sub)

    def _dag_device_key(d: TrainingDAG) -> tuple[int, ...]:
        keys = {tuple(sorted(n.device)) for n in d.nodes.values() if n.device is not None}
        if len(keys) != 1:
            raise ValueError(f"sub-DAG should have exactly one device key, got {keys}")
        return next(iter(keys))

    subdags.sort(key=_dag_device_key)
    return subdags


_ROUTED_CROSS_GROUP_KINDS = frozenset({"M_TO_N_COMM", "N_TO_M_COMM"})


def _validate_routed_collective_sequences(
    actor_group_dags: Sequence[TrainingDAG],
) -> None:
    """Reject actor-local routed-collective orders that would mismatch NCCL.

    Cross-group communication is represented by one local node on each side
    of a collective.  Nodes with the same actor-group pair, physical
    direction, and routed node kind share one communicator channel.  NCCL
    requires both sides to enter collectives on that channel in exactly the
    same order, so compare the actor execution orders after all split/order and
    per-stream ordering transforms have run.

    M-to-N and N-to-M intentionally use distinct channels.  Their sequences
    are therefore validated independently and may interleave differently.
    """

    def dag_device_key(dag: TrainingDAG) -> tuple[int, ...]:
        keys = {
            tuple(sorted(int(device) for device in node.device))
            for node in dag.nodes.values()
            if node.device is not None
        }
        if len(keys) != 1:
            raise ValueError(
                "routed collective validation requires one device set per "
                f"actor DAG, got {sorted(keys)}"
            )
        return next(iter(keys))

    # channel -> role -> (local actor-group devices, ordered comm ids)
    channel_sequences: dict[
        tuple[
            tuple[tuple[int, ...], tuple[int, ...]],
            str,
            str,
        ],
        dict[str, tuple[tuple[int, ...], list[str]]],
    ] = {}
    transactions: dict[
        str,
        list[
            tuple[
                tuple[
                    tuple[tuple[int, ...], tuple[int, ...]],
                    str,
                    str,
                ],
                str,
                str,
                TrainingDAGNode,
            ]
        ],
    ] = {}
    microbatches_by_expert_fwd_uid: dict[str, set[int]] = {}

    for dag in actor_group_dags:
        local_devices = dag_device_key(dag)
        ordered_uids = _serial_topological_order(dag)
        local_by_channel: dict[
            tuple[
                tuple[tuple[int, ...], tuple[int, ...]],
                str,
                str,
            ],
            tuple[str, list[str]],
        ] = {}
        for uid in ordered_uids:
            node = dag.nodes[uid]
            if node.node_kind not in _ROUTED_CROSS_GROUP_KINDS:
                continue
            meta = node.node_meta
            role = meta.get("role")
            if role not in {"source", "destination"}:
                raise ValueError(
                    f"routed collective node {uid} has invalid role={role!r}"
                )
            direction = meta.get("direction")
            expected_direction = (
                "incoming" if node.node_kind == "M_TO_N_COMM" else "outgoing"
            )
            if direction != expected_direction:
                raise ValueError(
                    f"routed collective node {uid} kind={node.node_kind} must "
                    f"have direction={expected_direction!r}, got {direction!r}"
                )
            try:
                source_devices = tuple(
                    sorted(int(device) for device in meta["source_devices"])
                )
                destination_devices = tuple(
                    sorted(int(device) for device in meta["destination_devices"])
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"routed collective node {uid} has invalid source/destination "
                    "device metadata"
                ) from error
            if not source_devices or not destination_devices:
                raise ValueError(
                    f"routed collective node {uid} requires non-empty source "
                    "and destination device sets"
                )
            if set(source_devices) & set(destination_devices):
                raise ValueError(
                    f"routed collective node {uid} requires disjoint groups, got "
                    f"source={source_devices}, destination={destination_devices}"
                )
            expected_local_devices = (
                source_devices if role == "source" else destination_devices
            )
            if local_devices != expected_local_devices:
                raise ValueError(
                    f"routed collective node {uid} role={role} is in actor DAG "
                    f"devices={local_devices}, expected {expected_local_devices}"
                )
            participants = tuple(
                sorted(int(device) for device in meta.get("participants", ()))
            )
            expected_participants = tuple(
                sorted((*source_devices, *destination_devices))
            )
            if participants != expected_participants:
                raise ValueError(
                    f"routed collective node {uid} has participants={participants}, "
                    f"expected {expected_participants}"
                )
            comm_id = meta.get("comm_id")
            if not isinstance(comm_id, str) or not comm_id:
                raise ValueError(
                    f"routed collective node {uid} has invalid comm_id={comm_id!r}"
                )
            expert_fwd_uid = meta.get("expert_fwd_uid")
            if not isinstance(expert_fwd_uid, str) or not expert_fwd_uid:
                raise ValueError(
                    f"routed collective node {uid} has invalid "
                    f"expert_fwd_uid={expert_fwd_uid!r}"
                )
            for arity_name in ("expert_input_arity", "expert_output_arity"):
                arity = meta.get(arity_name)
                if (
                    isinstance(arity, bool)
                    or not isinstance(arity, int)
                    or arity < 0
                ):
                    raise ValueError(
                        f"routed collective node {uid} has invalid "
                        f"{arity_name}={arity!r}"
                    )
            tensor_idx = meta.get("a2a_tensor_idx")
            if (
                isinstance(tensor_idx, bool)
                or not isinstance(tensor_idx, int)
                or tensor_idx < 0
            ):
                raise ValueError(
                    f"routed collective node {uid} has invalid "
                    f"a2a_tensor_idx={tensor_idx!r}"
                )
            is_forward = node.tag.get("PASS") == "F"
            uses_expert_inputs = (
                node.node_kind == "M_TO_N_COMM" and is_forward
            ) or (
                node.node_kind == "N_TO_M_COMM" and not is_forward
            )
            tensor_arity_name = (
                "expert_input_arity"
                if uses_expert_inputs
                else "expert_output_arity"
            )
            tensor_arity = meta[tensor_arity_name]
            if tensor_idx >= tensor_arity:
                raise ValueError(
                    f"routed collective node {uid} has a2a_tensor_idx="
                    f"{tensor_idx}, outside {tensor_arity_name}={tensor_arity}"
                )
            if meta.get("bucket_key") is None:
                raise ValueError(
                    f"routed collective node {uid} is missing bucket_key"
                )
            if meta.get("comm_group_kind") != "routed_cross_group":
                raise ValueError(
                    f"routed collective node {uid} has invalid comm_group_kind="
                    f"{meta.get('comm_group_kind')!r}"
                )
            if meta.get("source_degree") != len(source_devices):
                raise ValueError(
                    f"routed collective node {uid} has source_degree="
                    f"{meta.get('source_degree')!r}, expected {len(source_devices)}"
                )
            if meta.get("destination_degree") != len(destination_devices):
                raise ValueError(
                    f"routed collective node {uid} has destination_degree="
                    f"{meta.get('destination_degree')!r}, expected "
                    f"{len(destination_devices)}"
                )
            if node.node_kind == "M_TO_N_COMM" and node.tag.get("PASS") == "F":
                counts_idx = meta.get("routing_counts_idx")
                if (
                    isinstance(counts_idx, bool)
                    or not isinstance(counts_idx, int)
                    or counts_idx < 0
                ):
                    raise ValueError(
                        f"forward M_TO_N node {uid} has invalid "
                        f"routing_counts_idx={counts_idx!r}"
                    )
                if counts_idx >= meta["expert_input_arity"]:
                    raise ValueError(
                        f"forward M_TO_N node {uid} has routing_counts_idx="
                        f"{counts_idx}, outside expert_input_arity="
                        f"{meta['expert_input_arity']}"
                    )
                if counts_idx == tensor_idx:
                    raise ValueError(
                        f"forward M_TO_N node {uid} must use distinct routed "
                        f"payload/count indices, both are {tensor_idx}"
                    )
            microbatch = node.tag.get("MB")
            if isinstance(microbatch, int) and not isinstance(microbatch, bool):
                microbatches_by_expert_fwd_uid.setdefault(
                    expert_fwd_uid, set()
                ).add(microbatch)

            actor_pair = tuple(sorted((source_devices, destination_devices)))
            channel = (actor_pair, node.node_kind, direction)
            transactions.setdefault(comm_id, []).append(
                (channel, role, uid, node)
            )
            previous = local_by_channel.get(channel)
            if previous is None:
                local_by_channel[channel] = (role, [comm_id])
            else:
                previous_role, sequence = previous
                if previous_role != role:
                    raise ValueError(
                        f"actor DAG devices={local_devices} appears as both roles "
                        f"on routed channel {channel}"
                    )
                sequence.append(comm_id)

        for channel, (role, sequence) in local_by_channel.items():
            roles = channel_sequences.setdefault(channel, {})
            if role in roles:
                previous_devices, _previous_sequence = roles[role]
                raise ValueError(
                    f"routed channel {channel} has multiple {role} actor DAGs: "
                    f"{previous_devices} and {local_devices}"
                )
            roles[role] = (local_devices, sequence)

    for expert_fwd_uid, microbatches in microbatches_by_expert_fwd_uid.items():
        if len(microbatches) > 1:
            raise ValueError(
                f"routed expert state key {expert_fwd_uid!r} is reused across "
                f"microbatches={sorted(microbatches)}"
            )

    shared_meta_fields = (
        "source_devices",
        "destination_devices",
        "source_degree",
        "destination_degree",
        "participants",
        "expert_fwd_uid",
        "expert_input_arity",
        "expert_output_arity",
        "dispatch_devices",
        "routed_devices",
        "combine_devices",
        "dispatch_payload_idx",
        "dispatch_output_arity",
        "combine_payload_idx",
        "routed_dispatch_input_idx",
        "routed_counts_input_idx",
        "combine_destination_input_idx",
        "payload_schema",
        "a2a_tensor_idx",
        "routing_counts_idx",
        "bucket_key",
        "comm_group_kind",
        "direction",
    )
    for comm_id, endpoints in transactions.items():
        if len(endpoints) != 2 or {endpoint[1] for endpoint in endpoints} != {
            "source",
            "destination",
        }:
            raise ValueError(
                f"routed transaction {comm_id!r} requires exactly one source "
                f"and destination endpoint, got "
                f"{[(role, uid) for _channel, role, uid, _node in endpoints]}"
            )
        source = next(endpoint for endpoint in endpoints if endpoint[1] == "source")
        destination = next(
            endpoint for endpoint in endpoints if endpoint[1] == "destination"
        )
        if source[0] != destination[0]:
            raise ValueError(
                f"routed transaction {comm_id!r} endpoints disagree on channel: "
                f"source={source[0]}, destination={destination[0]}"
            )
        source_node = source[3]
        destination_node = destination[3]
        for field in shared_meta_fields:
            if source_node.node_meta.get(field) != destination_node.node_meta.get(field):
                raise ValueError(
                    f"routed transaction {comm_id!r} endpoints disagree on "
                    f"{field}: source={source_node.node_meta.get(field)!r}, "
                    f"destination={destination_node.node_meta.get(field)!r}"
                )
        for tag_name in ("PASS", "MB"):
            if source_node.tag.get(tag_name) != destination_node.tag.get(tag_name):
                raise ValueError(
                    f"routed transaction {comm_id!r} endpoints disagree on "
                    f"{tag_name}: source={source_node.tag.get(tag_name)!r}, "
                    f"destination={destination_node.tag.get(tag_name)!r}"
                )

    for channel, roles in channel_sequences.items():
        if set(roles) != {"source", "destination"}:
            raise ValueError(
                f"routed channel {channel} requires source and destination actor "
                f"DAGs, got roles={sorted(roles)}"
            )
        source_devices, source_sequence = roles["source"]
        destination_devices, destination_sequence = roles["destination"]
        if source_sequence != destination_sequence:
            mismatch_index = next(
                (
                    index
                    for index, (source_id, destination_id) in enumerate(
                        zip(source_sequence, destination_sequence)
                    )
                    if source_id != destination_id
                ),
                min(len(source_sequence), len(destination_sequence)),
            )
            raise ValueError(
                "routed collective call-sequence mismatch on shared channel "
                f"{channel}: source actor {source_devices} sequence="
                f"{source_sequence}, destination actor {destination_devices} "
                f"sequence={destination_sequence}; first mismatch at index "
                f"{mismatch_index}"
            )


def _validate_p2p_sequences(
    actor_group_dags: Sequence[TrainingDAG],
) -> None:
    """Validate transactions and call order for every SEND/RECV channel."""

    actor_devices: list[tuple[int, ...]] = []
    for dag in actor_group_dags:
        keys = {
            tuple(sorted(int(device) for device in node.device))
            for node in dag.nodes.values()
            if node.device is not None
        }
        if len(keys) != 1:
            raise ValueError(
                "P2P validation requires one device set per actor DAG, got "
                f"{sorted(keys)}"
            )
        actor_devices.append(next(iter(keys)))
    ordered_actor_devices = sorted(actor_devices)
    if len(set(ordered_actor_devices)) != len(ordered_actor_devices):
        raise ValueError("P2P validation requires distinct actor-group device sets")
    device_to_pp_rank = {
        devices: rank for rank, devices in enumerate(ordered_actor_devices)
    }

    transactions: dict[str, list[tuple[str, str, TrainingDAGNode]]] = {}
    sequences: dict[
        tuple[tuple[tuple[int, ...], tuple[int, ...]], str],
        dict[str, tuple[tuple[int, ...], list[str]]],
    ] = {}
    for dag, local_devices in zip(actor_group_dags, actor_devices):
        local_rank = device_to_pp_rank[local_devices]
        local_sequences: dict[
            tuple[tuple[tuple[int, ...], tuple[int, ...]], str],
            tuple[str, list[str]],
        ] = {}
        for uid in _serial_topological_order(dag):
            node = dag.nodes[uid]
            if node.node_kind not in {"SEND_COMM", "RECV_COMM"}:
                continue
            meta = node.node_meta
            peer_rank = meta.get("peer_pp_rank")
            if (
                isinstance(peer_rank, bool)
                or not isinstance(peer_rank, int)
                or peer_rank < 0
                or peer_rank >= len(ordered_actor_devices)
                or peer_rank == local_rank
            ):
                raise ValueError(
                    f"P2P node {uid} has invalid peer_pp_rank={peer_rank!r}"
                )
            peer_devices = ordered_actor_devices[peer_rank]
            if set(local_devices) & set(peer_devices):
                raise ValueError(
                    f"P2P node {uid} requires disjoint actor groups, got "
                    f"local={local_devices}, peer={peer_devices}"
                )
            if len(local_devices) != len(peer_devices):
                raise ValueError(
                    f"P2P node {uid} requires equal-width actor groups, got "
                    f"local={local_devices}, peer={peer_devices}"
                )

            role = "source" if node.node_kind == "SEND_COMM" else "destination"
            source_rank = local_rank if role == "source" else peer_rank
            destination_rank = peer_rank if role == "source" else local_rank
            group_prefix = (
                "dense_state"
                if bool(meta.get("dense_state_transfer"))
                else "pp"
            )
            expected_group_kind = (
                f"{group_prefix}_lo_hi"
                if source_rank < destination_rank
                else f"{group_prefix}_hi_lo"
            )
            group_kind = meta.get("comm_group_kind")
            if group_kind != expected_group_kind:
                raise ValueError(
                    f"P2P node {uid} has comm_group_kind={group_kind!r}, "
                    f"expected {expected_group_kind!r}"
                )
            participants = tuple(
                sorted(int(device) for device in meta.get("participants", ()))
            )
            expected_participants = tuple(
                sorted((*local_devices, *peer_devices))
            )
            if participants != expected_participants:
                raise ValueError(
                    f"P2P node {uid} has participants={participants}, expected "
                    f"{expected_participants}"
                )
            comm_id = meta.get("comm_id")
            if not isinstance(comm_id, str) or not comm_id:
                raise ValueError(
                    f"P2P node {uid} has invalid comm_id={comm_id!r}"
                )

            actor_pair = tuple(sorted((local_devices, peer_devices)))
            channel = (actor_pair, group_kind)
            transactions.setdefault(comm_id, []).append((role, uid, node))
            previous = local_sequences.get(channel)
            if previous is None:
                local_sequences[channel] = (role, [comm_id])
            else:
                previous_role, sequence = previous
                if previous_role != role:
                    raise ValueError(
                        f"actor DAG devices={local_devices} appears as both "
                        f"roles on P2P channel {channel}"
                    )
                sequence.append(comm_id)

        for channel, (role, sequence) in local_sequences.items():
            roles = sequences.setdefault(channel, {})
            if role in roles:
                previous_devices, _previous_sequence = roles[role]
                raise ValueError(
                    f"P2P channel {channel} has multiple {role} actor DAGs: "
                    f"{previous_devices} and {local_devices}"
                )
            roles[role] = (local_devices, sequence)

    for comm_id, endpoints in transactions.items():
        if len(endpoints) != 2 or {endpoint[0] for endpoint in endpoints} != {
            "source",
            "destination",
        }:
            raise ValueError(
                f"P2P transaction {comm_id!r} requires exactly one SEND and "
                f"one RECV endpoint, got "
                f"{[(role, uid) for role, uid, _node in endpoints]}"
            )
        source = next(endpoint[2] for endpoint in endpoints if endpoint[0] == "source")
        destination = next(
            endpoint[2] for endpoint in endpoints if endpoint[0] == "destination"
        )
        for field in ("comm_group_kind", "participants"):
            if source.node_meta.get(field) != destination.node_meta.get(field):
                raise ValueError(
                    f"P2P transaction {comm_id!r} endpoints disagree on "
                    f"{field}"
                )
        if source.tag.get("MB") != destination.tag.get("MB"):
            raise ValueError(
                f"P2P transaction {comm_id!r} endpoints disagree on MB"
            )
        source_pass = source.tag.get("PASS")
        destination_pass = destination.tag.get("PASS")
        # A backward activation may cross from a fused B region into a split BI
        # region (or vice versa). PASS describes local compute decomposition,
        # not the identity of the P2P transaction. Reject only a true
        # forward/backward direction mismatch.
        backward_passes = {"B", "BI", "BW"}
        same_direction = (
            source_pass == destination_pass
            or source_pass in backward_passes
            and destination_pass in backward_passes
        )
        if not same_direction:
            raise ValueError(
                f"P2P transaction {comm_id!r} endpoints disagree on PASS "
                f"direction: source={source_pass!r}, "
                f"destination={destination_pass!r}"
            )

    for channel, roles in sequences.items():
        if set(roles) != {"source", "destination"}:
            raise ValueError(
                f"P2P channel {channel} requires source and destination actor "
                f"DAGs, got roles={sorted(roles)}"
            )
        source_devices, source_sequence = roles["source"]
        destination_devices, destination_sequence = roles["destination"]
        if source_sequence != destination_sequence:
            mismatch_index = next(
                (
                    index
                    for index, (source_id, destination_id) in enumerate(
                        zip(source_sequence, destination_sequence)
                    )
                    if source_id != destination_id
                ),
                min(len(source_sequence), len(destination_sequence)),
            )
            raise ValueError(
                "P2P call-sequence mismatch on shared channel "
                f"{channel}: source actor {source_devices} sequence="
                f"{source_sequence}, destination actor {destination_devices} "
                f"sequence={destination_sequence}; first mismatch at index "
                f"{mismatch_index}"
            )


def _validate_dense_state_sequences(
    actor_group_dags: Sequence[TrainingDAG],
) -> None:
    """Validate paired direct dense-state P2P endpoints and call order."""

    def dag_device_key(dag: TrainingDAG) -> tuple[int, ...]:
        keys = {
            tuple(sorted(int(device) for device in node.device))
            for node in dag.nodes.values()
            if node.device is not None
        }
        if len(keys) != 1:
            raise ValueError(
                "dense-state validation requires one device set per actor "
                f"DAG, got {sorted(keys)}"
            )
        return next(iter(keys))

    actor_devices = [dag_device_key(dag) for dag in actor_group_dags]
    device_to_pp_rank = {
        devices: rank for rank, devices in enumerate(sorted(actor_devices))
    }
    if len(device_to_pp_rank) != len(actor_devices):
        raise ValueError(
            "dense-state validation requires distinct actor-group device sets"
        )

    channel_sequences: dict[
        tuple[tuple[int, ...], tuple[int, ...], str],
        dict[str, tuple[tuple[int, ...], list[str]]],
    ] = {}
    transactions: dict[
        str,
        list[
            tuple[
                tuple[tuple[int, ...], tuple[int, ...], str],
                str,
                str,
                TrainingDAGNode,
            ]
        ],
    ] = {}

    for dag, local_devices in zip(actor_group_dags, actor_devices):
        local_by_channel: dict[
            tuple[tuple[int, ...], tuple[int, ...], str],
            tuple[str, list[str]],
        ] = {}
        for uid in _serial_topological_order(dag):
            node = dag.nodes[uid]
            meta = node.node_meta
            if (
                node.node_kind not in {"SEND_COMM", "RECV_COMM"}
                or not bool(meta.get("dense_state_transfer"))
            ):
                continue

            role = meta.get("role")
            expected_role = (
                "source" if node.node_kind == "SEND_COMM" else "destination"
            )
            if role != expected_role:
                raise ValueError(
                    f"dense-state node {uid} kind={node.node_kind} must have "
                    f"role={expected_role!r}, got {role!r}"
                )
            try:
                source_devices = tuple(
                    sorted(int(device) for device in meta["source_devices"])
                )
                destination_devices = tuple(
                    sorted(int(device) for device in meta["destination_devices"])
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"dense-state node {uid} has invalid source/destination "
                    "device metadata"
                ) from error
            if not source_devices or not destination_devices:
                raise ValueError(
                    f"dense-state node {uid} requires non-empty device sets"
                )
            if set(source_devices) & set(destination_devices):
                raise ValueError(
                    f"dense-state node {uid} requires disjoint actor groups, "
                    f"got source={source_devices}, destination="
                    f"{destination_devices}"
                )
            if len(source_devices) != len(destination_devices):
                raise ValueError(
                    f"dense-state node {uid} currently requires equal actor "
                    f"degrees, got {len(source_devices)} and "
                    f"{len(destination_devices)}"
                )
            expected_local = (
                source_devices if role == "source" else destination_devices
            )
            if local_devices != expected_local:
                raise ValueError(
                    f"dense-state node {uid} role={role} is in actor DAG "
                    f"devices={local_devices}, expected {expected_local}"
                )

            participants = tuple(
                sorted(int(device) for device in meta.get("participants", ()))
            )
            expected_participants = tuple(
                sorted((*source_devices, *destination_devices))
            )
            if participants != expected_participants:
                raise ValueError(
                    f"dense-state node {uid} has participants={participants}, "
                    f"expected {expected_participants}"
                )
            if meta.get("source_degree") != len(source_devices):
                raise ValueError(
                    f"dense-state node {uid} has invalid source_degree="
                    f"{meta.get('source_degree')!r}"
                )
            if meta.get("destination_degree") != len(destination_devices):
                raise ValueError(
                    f"dense-state node {uid} has invalid destination_degree="
                    f"{meta.get('destination_degree')!r}"
                )

            direction = meta.get("direction")
            if direction not in {"forward", "backward"}:
                raise ValueError(
                    f"dense-state node {uid} has invalid direction="
                    f"{direction!r}"
                )
            node_pass = node.tag.get("PASS")
            if (direction == "forward" and node_pass != "F") or (
                direction == "backward" and node_pass not in {"B", "BI"}
            ):
                raise ValueError(
                    f"dense-state node {uid} direction={direction!r} is "
                    f"incompatible with PASS={node_pass!r}"
                )

            dense_state_arity = meta.get("dense_state_arity")
            if (
                isinstance(dense_state_arity, bool)
                or not isinstance(dense_state_arity, int)
                or dense_state_arity < 0
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid dense_state_arity="
                    f"{dense_state_arity!r}"
                )
            dispatch_payload_idx = meta.get("dispatch_payload_idx")
            combine_payload_idx = meta.get("combine_payload_idx")
            dispatch_output_arity = meta.get("dispatch_output_arity")
            if (
                isinstance(dispatch_payload_idx, bool)
                or not isinstance(dispatch_payload_idx, int)
                or dispatch_payload_idx < 0
                or isinstance(combine_payload_idx, bool)
                or not isinstance(combine_payload_idx, int)
                or combine_payload_idx < 0
                or isinstance(meta.get("routing_counts_idx"), bool)
                or not isinstance(meta.get("routing_counts_idx"), int)
                or meta["routing_counts_idx"] < 0
                or meta["routing_counts_idx"] == dispatch_payload_idx
                or isinstance(dispatch_output_arity, bool)
                or not isinstance(dispatch_output_arity, int)
                or dispatch_output_arity <= max(
                    dispatch_payload_idx, meta["routing_counts_idx"]
                )
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid dispatch/combine "
                    f"indices={dispatch_payload_idx!r}/{combine_payload_idx!r} "
                    f"and counts={meta.get('routing_counts_idx')!r}"
                )
            model_indices = meta.get("model_tensor_indices")
            if (
                not isinstance(model_indices, list)
                or len(model_indices) != dense_state_arity
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    for index in model_indices
                )
                or len(set(model_indices)) != len(model_indices)
                or dispatch_payload_idx in model_indices
                or meta["routing_counts_idx"] in model_indices
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid model tensor slots="
                    f"{model_indices!r}"
                )
            destination_indices = meta.get("destination_input_indices")
            if (
                not isinstance(destination_indices, list)
                or len(destination_indices) != dense_state_arity
                or any(
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    for index in destination_indices
                )
                or len(set(destination_indices)) != len(destination_indices)
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid destination input "
                    f"slots={destination_indices!r}"
                )
            differentiable_indices = meta.get("differentiable_model_indices")
            if (
                not isinstance(differentiable_indices, list)
                or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in differentiable_indices
                )
                or sorted(set(differentiable_indices))
                != differentiable_indices
                or not set(differentiable_indices).issubset(model_indices)
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid differentiable model "
                    f"indices={differentiable_indices!r}"
                )
            model_value_names = meta.get("model_value_names")
            model_value_schemas = meta.get("model_value_schemas")
            if (
                not isinstance(model_value_names, list)
                or len(model_value_names) != len(model_indices)
                or not all(isinstance(name, str) for name in model_value_names)
                or not isinstance(model_value_schemas, list)
                or len(model_value_schemas) != len(model_indices)
                or not all(
                    isinstance(schema, dict) for schema in model_value_schemas
                )
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid model value "
                    f"descriptors names={model_value_names!r}, "
                    f"schemas={model_value_schemas!r}"
                )
            for field in (
                "routed_dispatch_input_idx",
                "routed_counts_input_idx",
                "combine_destination_input_idx",
            ):
                value = meta.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(
                        f"dense-state node {uid} has invalid {field}={value!r}"
                    )
            if (
                meta["routed_dispatch_input_idx"]
                == meta["routed_counts_input_idx"]
            ):
                raise ValueError(
                    f"dense-state node {uid} maps dispatch payload and counts "
                    "to the same routed input slot"
                )
            if meta.get("carries_routing_layout") is not True:
                raise ValueError(
                    f"dense-state node {uid} must carry Piper's private "
                    "routing layout"
                )
            routing_counts_schema = meta.get("routing_counts_schema")
            if (
                not isinstance(routing_counts_schema, dict)
                or not isinstance(routing_counts_schema.get("dtype"), str)
                or routing_counts_schema.get("shape") is None
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid routing-count schema="
                    f"{routing_counts_schema!r}"
                )

            rank_pairs = meta.get("rank_pairs")
            expected_rank_pairs = [
                [source_rank, destination_rank]
                for source_rank, destination_rank in zip(
                    source_devices, destination_devices
                )
            ]
            if meta.get("transfer_mode") != "ordinal_one_to_one" or (
                rank_pairs != expected_rank_pairs
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid ordinal rank mapping="
                    f"{rank_pairs!r}, expected {expected_rank_pairs!r}"
                )

            source_pp_rank = device_to_pp_rank.get(source_devices)
            destination_pp_rank = device_to_pp_rank.get(destination_devices)
            if source_pp_rank is None or destination_pp_rank is None:
                raise ValueError(
                    f"dense-state node {uid} references an unknown actor group"
                )
            expected_group_kind = (
                "dense_state_lo_hi"
                if source_pp_rank < destination_pp_rank
                else "dense_state_hi_lo"
            )
            if meta.get("comm_group_kind") != expected_group_kind:
                raise ValueError(
                    f"dense-state node {uid} has comm_group_kind="
                    f"{meta.get('comm_group_kind')!r}, expected "
                    f"{expected_group_kind!r}"
                )
            expected_peer_pp_rank = (
                destination_pp_rank if role == "source" else source_pp_rank
            )
            expected_peer_devices = (
                destination_devices if role == "source" else source_devices
            )
            if meta.get("peer_pp_rank") != expected_peer_pp_rank:
                raise ValueError(
                    f"dense-state node {uid} has peer_pp_rank="
                    f"{meta.get('peer_pp_rank')!r}, expected "
                    f"{expected_peer_pp_rank}"
                )
            if tuple(meta.get("peer_devices", ())) != expected_peer_devices:
                raise ValueError(
                    f"dense-state node {uid} has peer_devices="
                    f"{meta.get('peer_devices')!r}, expected "
                    f"{expected_peer_devices}"
                )

            for field in (
                "bucket_key",
                "state_owner_bucket_key",
                "combine_consumer_bucket_key",
            ):
                if meta.get(field) is None:
                    raise ValueError(
                        f"dense-state node {uid} is missing {field}"
                    )
            state_owner_fwd_uid = meta.get("state_owner_fwd_uid")
            combine_consumer_fwd_uid = meta.get("combine_consumer_fwd_uid")
            if not isinstance(state_owner_fwd_uid, str) or not isinstance(
                combine_consumer_fwd_uid, str
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid lifecycle endpoints "
                    f"{state_owner_fwd_uid!r}->{combine_consumer_fwd_uid!r}"
                )
            expected_from_uid = (
                state_owner_fwd_uid
                if direction == "forward"
                else combine_consumer_fwd_uid
            )
            expected_to_uid = (
                combine_consumer_fwd_uid
                if direction == "forward"
                else state_owner_fwd_uid
            )
            if (
                meta.get("from_uid") != expected_from_uid
                or meta.get("to_uid") != expected_to_uid
            ):
                raise ValueError(
                    f"dense-state node {uid} has invalid logical endpoints "
                    f"{meta.get('from_uid')!r}->{meta.get('to_uid')!r}; "
                    f"expected {expected_from_uid!r}->{expected_to_uid!r}"
                )

            comm_id = meta.get("comm_id")
            if not isinstance(comm_id, str) or not comm_id:
                raise ValueError(
                    f"dense-state node {uid} has invalid comm_id={comm_id!r}"
                )
            channel = (source_devices, destination_devices, direction)
            transactions.setdefault(comm_id, []).append(
                (channel, role, uid, node)
            )
            previous = local_by_channel.get(channel)
            if previous is None:
                local_by_channel[channel] = (role, [comm_id])
            else:
                previous_role, sequence = previous
                if previous_role != role:
                    raise ValueError(
                        f"actor DAG devices={local_devices} appears as both "
                        f"roles on dense-state channel {channel}"
                    )
                sequence.append(comm_id)

        for channel, (role, sequence) in local_by_channel.items():
            roles = channel_sequences.setdefault(channel, {})
            if role in roles:
                previous_devices, _previous_sequence = roles[role]
                raise ValueError(
                    f"dense-state channel {channel} has multiple {role} actor "
                    f"DAGs: {previous_devices} and {local_devices}"
                )
            roles[role] = (local_devices, sequence)

    shared_meta_fields = (
        "direction",
        "source_devices",
        "destination_devices",
        "source_degree",
        "destination_degree",
        "participants",
        "expert_fwd_uid",
        "dispatch_devices",
        "routed_devices",
        "combine_devices",
        "dense_state_arity",
        "dispatch_output_arity",
        "model_tensor_indices",
        "destination_input_indices",
        "differentiable_model_indices",
        "model_value_names",
        "model_value_schemas",
        "dispatch_payload_idx",
        "routing_counts_idx",
        "combine_payload_idx",
        "routed_dispatch_input_idx",
        "routed_counts_input_idx",
        "combine_destination_input_idx",
        "routing_counts_schema",
        "payload_schema",
        "carries_routing_layout",
        "bucket_key",
        "state_owner_bucket_key",
        "combine_consumer_bucket_key",
        "state_owner_fwd_uid",
        "combine_consumer_fwd_uid",
        "comm_group_kind",
        "transfer_mode",
        "rank_pairs",
        "from_uid",
        "to_uid",
    )
    dense_state_lifecycles: dict[
        tuple[str, object],
        dict[str, TrainingDAGNode],
    ] = {}
    for comm_id, endpoints in transactions.items():
        if len(endpoints) != 2 or {endpoint[1] for endpoint in endpoints} != {
            "source",
            "destination",
        }:
            raise ValueError(
                f"dense-state transaction {comm_id!r} requires exactly one "
                "SEND source and RECV destination endpoint"
            )
        source = next(endpoint for endpoint in endpoints if endpoint[1] == "source")
        destination = next(
            endpoint for endpoint in endpoints if endpoint[1] == "destination"
        )
        if source[0] != destination[0]:
            raise ValueError(
                f"dense-state transaction {comm_id!r} endpoints disagree on "
                f"channel: source={source[0]}, destination={destination[0]}"
            )
        source_node = source[3]
        destination_node = destination[3]
        for field in shared_meta_fields:
            if source_node.node_meta.get(field) != destination_node.node_meta.get(field):
                raise ValueError(
                    f"dense-state transaction {comm_id!r} endpoints disagree "
                    f"on {field}: source={source_node.node_meta.get(field)!r}, "
                    f"destination={destination_node.node_meta.get(field)!r}"
                )
        for tag_name in ("PASS", "MB"):
            if source_node.tag.get(tag_name) != destination_node.tag.get(tag_name):
                raise ValueError(
                    f"dense-state transaction {comm_id!r} endpoints disagree "
                    f"on {tag_name}"
                )
        expert_fwd_uid = source_node.node_meta.get("expert_fwd_uid")
        if not isinstance(expert_fwd_uid, str) or not expert_fwd_uid:
            raise ValueError(
                f"dense-state transaction {comm_id!r} has invalid "
                f"expert_fwd_uid={expert_fwd_uid!r}"
            )
        direction = source_node.node_meta["direction"]
        lifecycle_key = (expert_fwd_uid, source_node.tag.get("MB"))
        directions = dense_state_lifecycles.setdefault(lifecycle_key, {})
        if direction in directions:
            raise ValueError(
                f"dense-state lifecycle {lifecycle_key} has multiple "
                f"{direction} transactions"
            )
        directions[direction] = source_node

    for lifecycle_key, directions in dense_state_lifecycles.items():
        if set(directions) != {"forward", "backward"}:
            raise ValueError(
                f"dense-state lifecycle {lifecycle_key} requires paired "
                f"forward/backward transactions, got {sorted(directions)}"
            )
        forward = directions["forward"].node_meta
        backward = directions["backward"].node_meta
        if (
            forward.get("source_devices") != backward.get("destination_devices")
            or forward.get("destination_devices") != backward.get("source_devices")
        ):
            raise ValueError(
                f"dense-state lifecycle {lifecycle_key} backward transaction "
                "must reverse the forward actor-group pair"
            )
        for field in (
            "dense_state_arity",
            "model_tensor_indices",
            "destination_input_indices",
            "differentiable_model_indices",
            "model_value_names",
            "model_value_schemas",
            "dispatch_payload_idx",
            "routing_counts_idx",
            "combine_payload_idx",
            "routed_dispatch_input_idx",
            "routed_counts_input_idx",
            "combine_destination_input_idx",
            "routing_counts_schema",
            "payload_schema",
            "dispatch_output_arity",
            "dispatch_devices",
            "routed_devices",
            "combine_devices",
            "carries_routing_layout",
            "state_owner_bucket_key",
            "combine_consumer_bucket_key",
            "state_owner_fwd_uid",
            "combine_consumer_fwd_uid",
        ):
            if forward.get(field) != backward.get(field):
                raise ValueError(
                    f"dense-state lifecycle {lifecycle_key} forward/backward "
                    f"transactions disagree on {field}"
                )

    for channel, roles in channel_sequences.items():
        if set(roles) != {"source", "destination"}:
            raise ValueError(
                f"dense-state channel {channel} requires source and destination "
                f"actor DAGs, got roles={sorted(roles)}"
            )
        source_devices, source_sequence = roles["source"]
        destination_devices, destination_sequence = roles["destination"]
        if source_sequence != destination_sequence:
            mismatch_index = next(
                (
                    index
                    for index, (source_id, destination_id) in enumerate(
                        zip(source_sequence, destination_sequence)
                    )
                    if source_id != destination_id
                ),
                min(len(source_sequence), len(destination_sequence)),
            )
            raise ValueError(
                "dense-state call-sequence mismatch on shared channel "
                f"{channel}: source actor {source_devices} sequence="
                f"{source_sequence}, destination actor {destination_devices} "
                f"sequence={destination_sequence}; first mismatch at index "
                f"{mismatch_index}"
            )


@register_backend
def piper(gm, example_inputs, **kwargs):
    """TrainingDAG backend: split by Piper annotations and lower schedule directives."""
    del example_inputs, kwargs

    schedule_info = getattr(piper_metadata, "schedule_info", {}) or {}
    schedule_directives = getattr(piper_metadata, "schedule_directives", None)
    _top_level_gm, annotation_segments = split_gm_by_annotations(gm)

    if not annotation_segments:
        raise ValueError(
            "No Piper annotations found in the traced graph. Wrap model compute "
            "with src.piper.annotate(...) before compiling with Piper."
        )

    # Build and store the new directed DAG representation for later scheduling transforms.
    training_dag = build_training_dag(annotation_segments)
    _validate_schedule_tags_exist(training_dag, schedule_directives)
    apply_schedule_directives(
        training_dag,
        schedule_directives,
    )
    piper_metadata.training_dag = training_dag
    artifact_dir = getattr(piper_metadata, "artifact_dir", "out")
    if getattr(piper_metadata, "visualize_dag", False):
        print_training_dag_order(
            training_dag,
            label="global_training_dag_order",
            out_dir=artifact_dir,
            output_name="global_training_dag_order",
        )
    per_pp_training_dags = _split_global_training_dag_by_pp_rank(training_dag)
    for i, subdag in enumerate(per_pp_training_dags):
        zero_chains = _prune_zero_lifetime_metadata(subdag)
        resolve_total_order_per_stream(subdag)
        _add_inter_chain_temporal_edges(subdag, zero_chains)
        if getattr(piper_metadata, "visualize_dag", False):
            log_training_dag_dependencies(subdag)
            print_training_dag_order(subdag, label=f"pp{i}", rank=i, out_dir=artifact_dir)
            # render_training_dag(subdag, output_path=f"{artifact_dir}/training_dag_pp{i}")
    _validate_routed_collective_sequences(per_pp_training_dags)
    _validate_dense_state_sequences(per_pp_training_dags)
    _validate_p2p_sequences(per_pp_training_dags)
    piper_metadata.per_pp_training_dags = per_pp_training_dags

    logger.debug(
        "built TrainingDAG with %d nodes and %d edges, split into %d local PP-rank DAG(s)",
        len(training_dag.nodes),
        len(training_dag.edges),
        len(per_pp_training_dags),
    )

    def callback(*args, _gm=gm):
        logger.debug(
            "piper compiled callback invoked directly; running local graph execution"
        )
        return _gm(*args)

    return callback


def piper_exec_dag(
    loss_fn, log_stats: bool = False, *, return_losses: bool = True
) -> list:
    """Execute one training step using the loaded per-rank TrainingDAG."""
    actors = piper_metadata.actors
    run_refs = [
        actor.run_dag.remote(loss_fn=loss_fn, return_losses=return_losses)
        for actor in actors.values()
    ]
    t0 = time.perf_counter()
    results = ray.get(run_refs)
    step_time = time.perf_counter() - t0

    if log_stats:
        _log_step_stats(step_time, log_stats, actors)

    if not return_losses:
        return []

    schedule_info = dict(getattr(piper_metadata, "schedule_info", {}) or {})
    actor_ranks = list(actors)
    result_by_rank = {
        int(actor_rank): result
        for actor_rank, result in zip(actor_ranks, results)
    }
    actor_groups = {
        int(group["id"]): group
        for group in schedule_info.get("actor_groups", [])
    }
    loss_group_ids = schedule_info.get("loss_actor_group_ids")
    if not actor_groups or not isinstance(loss_group_ids, list) or not loss_group_ids:
        raise RuntimeError(
            "schedule metadata must identify ActorGroups that compute loss"
        )

    losses = []
    for group_id in loss_group_ids:
        group = actor_groups.get(int(group_id))
        if group is None:
            raise RuntimeError(
                f"loss ActorGroup id {group_id} is missing from schedule metadata"
            )
        losses_by_rank = []
        for actor_rank in group["devices"]:
            result = result_by_rank[int(actor_rank)]
            rank_losses = result.get("losses", []) if isinstance(result, dict) else result
            losses_by_rank.append(list(rank_losses or []))
        sample_counts = {len(rank_losses) for rank_losses in losses_by_rank}
        if len(sample_counts) != 1:
            raise RuntimeError(
                f"loss ActorGroup {group_id} ranks returned inconsistent loss "
                f"sample counts: {sorted(sample_counts)}"
            )
        # Each rank computes the mean loss for its equally sized, distinct
        # microbatch shard. Their arithmetic mean is the global-microbatch loss.
        for samples in zip(*losses_by_rank):
            losses.append(sum(float(sample) for sample in samples) / len(samples))
    return losses


def piper_measure_dags(
    loss_fn,
    *,
    iterations: int = 30,
    iteration_sleep: float = 0.0,
    log_stats: bool = True,
    collect_peak_memory: bool = True,
) -> TimingResult:
    """Measure exactly ``iterations`` normal training steps.

    Every actor receives the same iteration count from the single program
    driver, preserving collective operation alignment.
    """

    iterations = int(iterations)
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    actors = piper_metadata.actors
    if not actors:
        raise RuntimeError("piper_setup() must create actors before timing")

    step_samples_s: list[float] = []
    peak_samples: dict[int, list[int]] = {}
    for _ in range(iterations):
        if collect_peak_memory:
            ray.get([
                actor.reset_peak_memory.remote()
                for actor in actors.values()
            ])
        started = time.perf_counter()
        piper_exec_dag(
            loss_fn,
            log_stats=log_stats,
            return_losses=False,
        )
        step_samples_s.append(time.perf_counter() - started)
        if collect_peak_memory:
            for rank, peak_bytes in ray.get([
                actor.get_and_reset_peak_memory_stats.remote()
                for actor in actors.values()
            ]):
                peak_samples.setdefault(int(rank), []).append(int(peak_bytes))
        if iteration_sleep > 0.0:
            time.sleep(iteration_sleep)

    summary = summarize_timing_samples(step_samples_s, estimator="median")
    logger.info(
        "hardware timing samples=%d, local 95%% CI error=%.3f%%",
        len(step_samples_s),
        summary.error_pct,
    )
    return TimingResult(
        step_samples_s=tuple(step_samples_s),
        peak_memory_samples_by_rank={
            rank: tuple(samples) for rank, samples in sorted(peak_samples.items())
        },
    )


def _compare_kineto_activity_signatures(results: Sequence[Any]) -> dict[str, Any]:
    """Compare exact per-node-range GPU activity sequences across IF/normal."""

    by_mode_rank = {
        (normalize_profile_mode(result.profile_mode), int(result.global_rank)): result
        for result in results
    }
    if_ranks = {
        rank for mode, rank in by_mode_rank
        if mode == INTERFERENCE_FREE_PROFILE_MODE
    }
    normal_ranks = {
        rank for mode, rank in by_mode_rank
        if mode == NORMAL_PROFILE_MODE
    }
    common_ranks = sorted(if_ranks & normal_ranks)
    summary: dict[str, Any] = {
        "comparison": (
            "exact ordered (GPU activity category, Kineto name) sequence and "
            "multiplicity per node-range occurrence"
        ),
        "actors_compared": len(common_ranks),
        "nodes_compared": 0,
        "node_ranges_compared": 0,
        "gpu_activities_compared": 0,
        "expected_gpu_activity_records": 0,
        "missing_gpu_activity_records": 0,
        "capture_loss_rate": 0.0,
        "capture_loss_threshold": _MAX_KINETO_CAPTURE_LOSS_RATE,
        "expected_cpu_launch_records": 0,
        "missing_cpu_launch_records": 0,
        "cpu_launch_loss_rate": 0.0,
        "cpu_launch_loss_threshold": _MAX_KINETO_CPU_LAUNCH_LOSS_RATE,
        "launch_evidence_complete": True,
        "capture_loss_only": False,
        "capture_loss_within_tolerance": False,
        "mismatched_nodes": 0,
        "mismatched_node_ranges": 0,
        "unclassified_node_ranges": 0,
        "missing_mode_ranks": {
            INTERFERENCE_FREE_PROFILE_MODE: sorted(normal_ranks - if_ranks),
            NORMAL_PROFILE_MODE: sorted(if_ranks - normal_ranks),
        },
        "mismatches": [],
    }

    def activity_counts(occurrence: Mapping[str, Any]) -> dict[tuple[str, str], int]:
        return {
            (str(item["category"]), str(item["name"])): int(item["count"])
            for item in occurrence.get("activities", ())
        }

    def missing_subsequence_entries(
        captured: Sequence[Any],
        expected: Sequence[Any],
    ) -> list[tuple[int, Any]] | None:
        """Return expected entries deleted from captured, or None if reordered."""

        captured_index = 0
        missing = []
        for expected_index, expected_entry in enumerate(expected):
            if (
                captured_index < len(captured)
                and captured[captured_index] == expected_entry
            ):
                captured_index += 1
            else:
                missing.append((expected_index, expected_entry))
        return missing if captured_index == len(captured) else None

    for rank in common_ranks:
        mode_audits = {}
        for mode in (INTERFERENCE_FREE_PROFILE_MODE, NORMAL_PROFILE_MODE):
            result = by_mode_rank[(mode, rank)]
            mode_audits[mode] = result.device_metadata.get(
                "kineto_attribution_audit", {}
            )
        node_uids = sorted(set(mode_audits[INTERFERENCE_FREE_PROFILE_MODE]) | set(
            mode_audits[NORMAL_PROFILE_MODE]
        ))
        for uid in node_uids:
            if_audit = mode_audits[INTERFERENCE_FREE_PROFILE_MODE].get(uid)
            normal_audit = mode_audits[NORMAL_PROFILE_MODE].get(uid)
            summary["nodes_compared"] += 1
            if if_audit is None or normal_audit is None:
                summary["mismatched_nodes"] += 1
                summary["mismatches"].append({
                    "global_rank": rank,
                    "uid": uid,
                    "reason": "node audit missing from one profile mode",
                    "if_present": if_audit is not None,
                    "normal_present": normal_audit is not None,
                })
                continue
            if_ranges = list(if_audit.get("range_activity_signatures", ()))
            normal_ranges = list(normal_audit.get("range_activity_signatures", ()))
            range_mismatches = []
            for occurrence in range(max(len(if_ranges), len(normal_ranges))):
                if_range = if_ranges[occurrence] if occurrence < len(if_ranges) else None
                normal_range = (
                    normal_ranges[occurrence]
                    if occurrence < len(normal_ranges)
                    else None
                )
                if if_range is not None and normal_range is not None:
                    summary["node_ranges_compared"] += 1
                range_detail: dict[str, Any] = {
                    "occurrence": occurrence,
                    "classification": "unclassified",
                    "missing_activities": [],
                }
                if if_range is None or normal_range is None:
                    range_detail["reason"] = "node range missing from one mode"
                else:
                    if_launch_sequence = if_range.get("cpu_launch_sequence")
                    normal_launch_sequence = normal_range.get(
                        "cpu_launch_sequence"
                    )
                    if_launches = if_range.get("cpu_launches")
                    normal_launches = normal_range.get("cpu_launches")
                    has_launch_evidence = all(
                        isinstance(value, list)
                        for value in (
                            if_launch_sequence,
                            normal_launch_sequence,
                            if_launches,
                            normal_launches,
                        )
                    )
                    exact_gpu_match = (
                        if_range.get("signature_sha256")
                        == normal_range.get("signature_sha256")
                        and if_range.get("activity_sequence_sha256")
                        == normal_range.get("activity_sequence_sha256")
                    )
                    if has_launch_evidence:
                        summary["expected_cpu_launch_records"] += max(
                            len(if_launch_sequence), len(normal_launch_sequence)
                        )
                    if not has_launch_evidence:
                        summary["launch_evidence_complete"] = False
                        if exact_gpu_match:
                            expected_count = int(
                                if_range.get("activity_count", 0)
                            )
                            summary["expected_gpu_activity_records"] += (
                                expected_count
                            )
                            summary["gpu_activities_compared"] += expected_count
                            continue
                        range_detail["reason"] = (
                            "per-occurrence CPU launch evidence is missing"
                        )
                    elif if_launch_sequence != normal_launch_sequence:
                        if (
                            len(if_launch_sequence) != len(if_launches)
                            or len(normal_launch_sequence) != len(normal_launches)
                        ):
                            range_detail.update({
                                "reason": (
                                    "canonical CPU launch records do not match "
                                    "their normalized sequences"
                                ),
                                "if_cpu_launch_count": len(if_launches),
                                "normal_cpu_launch_count": len(normal_launches),
                            })
                            if_missing_launches = None
                            normal_missing_launches = None
                        else:
                            if_missing_launches = missing_subsequence_entries(
                                if_launch_sequence, normal_launch_sequence
                            )
                            normal_missing_launches = missing_subsequence_entries(
                                normal_launch_sequence, if_launch_sequence
                            )
                        if if_missing_launches is not None:
                            expected_launches = normal_launches
                            missing_mode = INTERFERENCE_FREE_PROFILE_MODE
                            missing_launches = if_missing_launches
                            nondeficient_range = normal_range
                        elif normal_missing_launches is not None:
                            expected_launches = if_launches
                            missing_mode = NORMAL_PROFILE_MODE
                            missing_launches = normal_missing_launches
                            nondeficient_range = if_range
                        else:
                            expected_launches = []
                            missing_mode = None
                            missing_launches = []
                            nondeficient_range = None
                        missing_cpu_launch_details = [
                                {
                                    "launch_index": launch_index,
                                    "missing_mode": missing_mode,
                                    "operation": str(operation),
                                    "api_name": str(
                                        expected_launches[launch_index].get(
                                            "api_name", ""
                                        )
                                    ),
                                }
                                for launch_index, operation in missing_launches
                            ]
                        if missing_mode is None:
                            range_detail.update({
                                "classification": "execution_divergence",
                                "reason": "normalized CPU launch sequences differ",
                                "if_cpu_launch_sequence": if_launch_sequence,
                                "normal_cpu_launch_sequence": normal_launch_sequence,
                                "missing_cpu_launches": [],
                            })
                        elif any(
                            launch.get("mapping_status") != "unambiguous"
                            for launch in (*if_launches, *normal_launches)
                        ) or bool(
                            nondeficient_range.get("unmapped_activities", ())
                        ):
                            range_detail.update({
                                "reason": (
                                    "CPU launch deletion has unrelated ambiguous "
                                    "launch/activity mapping"
                                ),
                                "missing_cpu_launches": missing_cpu_launch_details,
                                "if_unmapped_activities": if_range.get(
                                    "unmapped_activities", []
                                ),
                                "normal_unmapped_activities": normal_range.get(
                                    "unmapped_activities", []
                                ),
                            })
                        else:
                            if_activity_sequence = if_range.get(
                                "activity_sequence"
                            )
                            normal_activity_sequence = normal_range.get(
                                "activity_sequence"
                            )
                            if not isinstance(if_activity_sequence, list) or not isinstance(
                                normal_activity_sequence, list
                            ):
                                summary["launch_evidence_complete"] = False
                                range_detail.update({
                                    "reason": (
                                        "per-occurrence GPU activity sequence "
                                        "evidence is missing"
                                    ),
                                    "missing_cpu_launches": missing_cpu_launch_details,
                                })
                            else:
                                if_missing_activities = missing_subsequence_entries(
                                    if_activity_sequence,
                                    normal_activity_sequence,
                                )
                                normal_missing_activities = missing_subsequence_entries(
                                    normal_activity_sequence,
                                    if_activity_sequence,
                                )
                                if if_missing_activities is not None:
                                    expected_activities = normal_activity_sequence
                                    gpu_missing_mode = (
                                        INTERFERENCE_FREE_PROFILE_MODE
                                    )
                                    missing_activity_entries = if_missing_activities
                                elif normal_missing_activities is not None:
                                    expected_activities = if_activity_sequence
                                    gpu_missing_mode = NORMAL_PROFILE_MODE
                                    missing_activity_entries = normal_missing_activities
                                else:
                                    expected_activities = []
                                    gpu_missing_mode = None
                                    missing_activity_entries = []
                                if gpu_missing_mode is None:
                                    range_detail.update({
                                        "classification": "execution_divergence",
                                        "reason": (
                                            "GPU activities were substituted or "
                                            "reordered alongside CPU launch loss"
                                        ),
                                        "missing_cpu_launches": (
                                            missing_cpu_launch_details
                                        ),
                                    })
                                else:
                                    missing_activities = [
                                        {
                                            "launch_index": None,
                                            "missing_mode": gpu_missing_mode,
                                            "operation": "",
                                            "if_api_name": "",
                                            "normal_api_name": "",
                                            "expected_activity_index": (
                                                activity_index
                                            ),
                                            "category": str(activity["category"]),
                                            "name": str(activity["name"]),
                                        }
                                        for activity_index, activity in (
                                            missing_activity_entries
                                        )
                                    ]
                                    summary["missing_cpu_launch_records"] += len(
                                        missing_launches
                                    )
                                    summary["expected_gpu_activity_records"] += len(
                                        expected_activities
                                    )
                                    summary["gpu_activities_compared"] += len(
                                        expected_activities
                                    )
                                    summary["missing_gpu_activity_records"] += len(
                                        missing_activities
                                    )
                                    range_detail.update({
                                        "classification": "capture_loss",
                                        "reason": (
                                            "deletion-only CPU launch loss"
                                            + (
                                                " and GPU activity loss"
                                                if missing_activities
                                                else ""
                                            )
                                        ),
                                        "missing_cpu_launches": (
                                            missing_cpu_launch_details
                                        ),
                                        "missing_activities": missing_activities,
                                    })
                    elif len(if_launches) != len(normal_launches):
                        range_detail.update({
                            "classification": "execution_divergence",
                            "reason": "canonical CPU launch counts differ",
                            "if_cpu_launch_count": len(if_launches),
                            "normal_cpu_launch_count": len(normal_launches),
                        })
                    elif (
                        not bool(if_range.get("launch_mapping_unambiguous", False))
                        or not bool(
                            normal_range.get("launch_mapping_unambiguous", False)
                        )
                    ):
                        range_detail.update({
                            "reason": "launch/activity mapping is ambiguous",
                            "if_unmapped_activities": if_range.get(
                                "unmapped_activities", []
                            ),
                            "normal_unmapped_activities": normal_range.get(
                                "unmapped_activities", []
                            ),
                        })
                    else:
                        missing_activities = []
                        expected_count = 0
                        launch_divergence = None
                        for launch_index, (if_launch, normal_launch) in enumerate(
                            zip(if_launches, normal_launches)
                        ):
                            if (
                                if_launch.get("operation")
                                != normal_launch.get("operation")
                            ):
                                launch_divergence = {
                                    "classification": "execution_divergence",
                                    "reason": (
                                        f"launch {launch_index} operation differs"
                                    ),
                                }
                                break
                            if_activities = list(if_launch.get("activities", ()))
                            normal_activities = list(
                                normal_launch.get("activities", ())
                            )
                            if not if_activities and not normal_activities:
                                launch_divergence = {
                                    "classification": "unclassified",
                                    "reason": (
                                        f"launch {launch_index} has no captured "
                                        "GPU activity in either mode"
                                    ),
                                }
                                break
                            if if_activities == normal_activities:
                                expected_count += len(if_activities)
                                continue
                            if_missing = missing_subsequence_entries(
                                if_activities, normal_activities
                            )
                            normal_missing = missing_subsequence_entries(
                                normal_activities, if_activities
                            )
                            if if_missing is not None:
                                expected = normal_activities
                                missing_mode = INTERFERENCE_FREE_PROFILE_MODE
                                missing_entries = if_missing
                            elif normal_missing is not None:
                                expected = if_activities
                                missing_mode = NORMAL_PROFILE_MODE
                                missing_entries = normal_missing
                            else:
                                launch_divergence = {
                                    "classification": "execution_divergence",
                                    "reason": (
                                        f"launch {launch_index} GPU activities "
                                        "were substituted or reordered"
                                    ),
                                }
                                break
                            expected_count += len(expected)
                            missing_activities.extend({
                                "launch_index": launch_index,
                                "missing_mode": missing_mode,
                                "operation": str(if_launch.get("operation", "")),
                                "if_api_name": str(if_launch.get("api_name", "")),
                                "normal_api_name": str(
                                    normal_launch.get("api_name", "")
                                ),
                                "expected_activity_index": activity_index,
                                "category": str(activity["category"]),
                                "name": str(activity["name"]),
                            } for activity_index, activity in missing_entries)

                        summary["expected_gpu_activity_records"] += expected_count
                        summary["gpu_activities_compared"] += expected_count
                        if launch_divergence is not None:
                            range_detail.update(launch_divergence)
                        elif missing_activities:
                            range_detail.update({
                                "classification": "capture_loss",
                                "reason": "deletion-only GPU activity loss",
                                "missing_activities": missing_activities,
                            })
                            summary["missing_gpu_activity_records"] += len(
                                missing_activities
                            )
                        else:
                            continue
                summary["mismatched_node_ranges"] += 1
                if range_detail["classification"] != "capture_loss":
                    summary["unclassified_node_ranges"] += 1
                if_counts = {} if if_range is None else activity_counts(if_range)
                normal_counts = (
                    {} if normal_range is None else activity_counts(normal_range)
                )
                deltas = []
                for category, name in sorted(set(if_counts) | set(normal_counts)):
                    if_count = if_counts.get((category, name), 0)
                    normal_count = normal_counts.get((category, name), 0)
                    if if_count != normal_count:
                        deltas.append({
                            "category": category,
                            "name": name,
                            "if_count": if_count,
                            "normal_count": normal_count,
                        })
                range_mismatches.append({
                    **range_detail,
                    "if_activity_count": (
                        None if if_range is None
                        else int(if_range.get("activity_count", 0))
                    ),
                    "normal_activity_count": (
                        None if normal_range is None
                        else int(normal_range.get("activity_count", 0))
                    ),
                    "if_signature_sha256": (
                        None if if_range is None
                        else if_range.get("signature_sha256")
                    ),
                    "normal_signature_sha256": (
                        None if normal_range is None
                        else normal_range.get("signature_sha256")
                    ),
                    "if_activity_sequence_sha256": (
                        None if if_range is None
                        else if_range.get("activity_sequence_sha256")
                    ),
                    "normal_activity_sequence_sha256": (
                        None if normal_range is None
                        else normal_range.get("activity_sequence_sha256")
                    ),
                    "activity_order_match": (
                        False
                        if if_range is None or normal_range is None
                        else if_range.get("activity_sequence_sha256")
                        == normal_range.get("activity_sequence_sha256")
                    ),
                    "different_activity_entries": len(deltas),
                    "activity_deltas": deltas[:25],
                })
            if range_mismatches:
                summary["mismatched_nodes"] += 1
                summary["mismatches"].append({
                    "global_rank": rank,
                    "uid": uid,
                    "if_range_count": len(if_ranges),
                    "normal_range_count": len(normal_ranges),
                    "ranges": range_mismatches,
                })

    summary["exact_match"] = (
        not any(summary["missing_mode_ranks"].values())
        and summary["mismatched_nodes"] == 0
    )
    expected = int(summary["expected_gpu_activity_records"])
    missing = int(summary["missing_gpu_activity_records"])
    summary["capture_loss_rate"] = 0.0 if expected == 0 else missing / expected
    expected_cpu = int(summary["expected_cpu_launch_records"])
    missing_cpu = int(summary["missing_cpu_launch_records"])
    summary["cpu_launch_loss_rate"] = (
        0.0 if expected_cpu == 0 else missing_cpu / expected_cpu
    )
    summary["capture_loss_only"] = (
        not summary["exact_match"]
        and not any(summary["missing_mode_ranks"].values())
        and summary["mismatched_node_ranges"] > 0
        and summary["unclassified_node_ranges"] == 0
        and (missing > 0 or missing_cpu > 0)
        and bool(summary["launch_evidence_complete"])
    )
    summary["capture_loss_within_tolerance"] = (
        bool(summary["capture_loss_only"])
        and float(summary["capture_loss_rate"])
        <= _MAX_KINETO_CAPTURE_LOSS_RATE
        and float(summary["cpu_launch_loss_rate"])
        <= _MAX_KINETO_CPU_LAUNCH_LOSS_RATE
    )
    summary["accepted"] = bool(
        summary["exact_match"] or summary["capture_loss_within_tolerance"]
    )
    return summary


def _strip_kineto_comparison_details(results: Sequence[Any]) -> None:
    """Remove transient full launch/activity sequences after comparison."""

    for result in results:
        for audit in result.device_metadata.get(
            "kineto_attribution_audit", {}
        ).values():
            for occurrence in audit.get("range_activity_signatures", ()):
                occurrence.pop("activities", None)
                occurrence.pop("activity_sequence", None)
                occurrence.pop("cpu_launch_sequence", None)
                occurrence.pop("cpu_launches", None)
                occurrence.pop("unmapped_activities", None)


def _filter_kineto_profile_iterations(
    results: Sequence[Any], iteration_indices: Sequence[int]
) -> list[Any]:
    """Retain the same fully paired Kineto iterations in every actor shard."""

    selected = tuple(int(index) for index in iteration_indices)
    filtered_results = []
    for result in results:
        filtered_profiles = {}
        for uid, profile in result.node_profiles.items():
            if not profile.executed:
                filtered_profiles[uid] = profile
                continue
            samples = tuple(profile.samples_us[index] for index in selected)
            offsets = (
                tuple(profile.start_offset_samples_us[index] for index in selected)
                if profile.start_offset_samples_us
                else ()
            )
            filtered_profiles[uid] = replace(
                profile,
                samples_us=samples,
                sample_iteration_ids=tuple(range(len(samples))),
                estimate_us=estimate_profile_time(samples, profile.estimator),
                start_offset_samples_us=offsets,
                start_offset_estimate_us=(
                    estimate_profile_time(offsets, profile.estimator)
                    if offsets
                    else None
                ),
            )

        metadata = dict(result.device_metadata)
        filtered_audits = {}
        for uid, raw_audit in metadata.get(
            "kineto_attribution_audit", {}
        ).items():
            audit = dict(raw_audit)
            signatures = list(audit.get("range_activity_signatures", ()))
            durations = list(audit.get("simulator_duration_samples_us", ()))
            audit["range_activity_signatures"] = [
                signatures[index] for index in selected
            ]
            audit["simulator_duration_samples_us"] = [
                durations[index] for index in selected
            ]
            audit["range_count"] = len(selected)
            selected_signatures = audit["range_activity_signatures"]
            if all(
                "cpu_launch_count" in occurrence
                and "matched_cpu_launch_count" in occurrence
                for occurrence in selected_signatures
            ):
                launch_count = sum(
                    int(occurrence["cpu_launch_count"])
                    for occurrence in selected_signatures
                )
                matched_launch_count = sum(
                    int(occurrence["matched_cpu_launch_count"])
                    for occurrence in selected_signatures
                )
                audit["launch_count"] = launch_count
                audit["matched_launch_count"] = matched_launch_count
                audit["unmatched_launch_count"] = (
                    launch_count - matched_launch_count
                )
                audit["launch_coverage_pct"] = (
                    100.0
                    if not launch_count
                    else 100.0 * matched_launch_count / launch_count
                )
            filtered_audits[uid] = audit
        metadata["kineto_attribution_audit"] = filtered_audits
        allocator_stats = list(metadata.get("allocator_iteration_stats", ()))
        if allocator_stats and max(selected, default=-1) < len(allocator_stats):
            metadata["allocator_iteration_stats"] = [
                allocator_stats[index] for index in selected
            ]
        filtered_results.append(
            replace(
                result,
                node_profiles=filtered_profiles,
                device_metadata=metadata,
                timeline_origin_samples_us=tuple(
                    result.timeline_origin_samples_us[index]
                    for index in selected
                ) if result.timeline_origin_samples_us else (),
                timeline_origin_uncertainty_samples_us=tuple(
                    result.timeline_origin_uncertainty_samples_us[index]
                    for index in selected
                ) if result.timeline_origin_uncertainty_samples_us else (),
            )
        )
    return filtered_results


def _select_actor_clock_calibration(
    samples: Sequence[tuple[int, int, int]],
) -> tuple[int, int]:
    """Return actor-to-driver offset and uncertainty from midpoint samples."""

    if not samples:
        raise ValueError("actor clock calibration requires at least one sample")
    candidates = []
    for driver_before_ns, actor_ns, driver_after_ns in samples:
        before = int(driver_before_ns)
        actor_time = int(actor_ns)
        after = int(driver_after_ns)
        if after < before:
            raise ValueError("actor clock calibration observed a negative round trip")
        round_trip_ns = after - before
        driver_midpoint_ns = (before + after) // 2
        candidates.append(
            (
                round_trip_ns,
                driver_midpoint_ns - actor_time,
                (round_trip_ns + 1) // 2,
            )
        )
    _round_trip_ns, offset_ns, uncertainty_ns = min(
        candidates, key=lambda item: (item[0], item[2], item[1])
    )
    return offset_ns, uncertainty_ns


def _calibrate_profile_actor_clocks(
    actors: Mapping[int, Any], *, samples: int = 9
) -> dict[int, tuple[int, int]]:
    """Calibrate arbitrary Ray actors to this program driver's clock."""

    if int(samples) <= 0:
        raise ValueError("actor clock calibration sample count must be positive")
    result: dict[int, tuple[int, int]] = {}
    for raw_rank, actor in sorted(actors.items()):
        observations = []
        for _ in range(int(samples)):
            driver_before_ns = time.monotonic_ns()
            actor_ns = int(ray.get(actor.profile_clock_sample.remote()))
            driver_after_ns = time.monotonic_ns()
            observations.append(
                (driver_before_ns, actor_ns, driver_after_ns)
            )
        result[int(raw_rank)] = _select_actor_clock_calibration(observations)
    return result


def piper_profile_dags(
    loss_fn,
    config: ProfileConfig | None = None,
    *,
    profile_modes: Sequence[str] | None = None,
    profile_configs: Mapping[str, ProfileConfig] | None = None,
    output_path: str | None = None,
    measure_uninstrumented_modes: bool = False,
    trace_output_dir: str | None = None,
    trace_ranks: Sequence[int] | None = None,
) -> ProfiledTrainingPlan:
    """Profile Piper nodes with Kineto/CUPTI and merge a replay plan.

    Piper intentionally does *not* insert CUDA timing events around individual
    nodes here: those events alter the concurrent-stream schedule being
    measured.  Kineto correlates GPU activity to the existing node
    ``record_function`` ranges.  The simulator subsequently derives every
    execution start from its DAG dependencies and stream FIFO ordering.
    Cross-actor-aligned activity offsets are collected only to decompose a
    collective's measured endpoint interval into rendezvous wait and service;
    they are not replay release constraints.
    """
    base_config = config or ProfileConfig()
    requested_modes = tuple(
        normalize_profile_mode(mode)
        for mode in (
            profile_modes
            if profile_modes is not None
            else tuple(profile_configs or (INTERFERENCE_FREE_PROFILE_MODE,))
        )
    )
    if not requested_modes or len(set(requested_modes)) != len(requested_modes):
        raise ValueError(f"invalid requested profile modes: {requested_modes!r}")
    configs: dict[str, ProfileConfig] = {}
    for mode in requested_modes:
        selected = (profile_configs or {}).get(mode, base_config)
        if mode == NORMAL_PROFILE_MODE and selected.isolation_scope != DEFAULT_NORMAL_SCOPE:
            selected = replace(selected, isolation_scope=DEFAULT_NORMAL_SCOPE)
        configs[mode] = selected

    actors = piper_metadata.actors
    if not actors or not piper_metadata.per_pp_training_dags:
        raise RuntimeError("piper_setup() must finalize actors and TrainingDAGs before profiling")
    receive_spec_shards = ray.get([
        actor.p2p_receive_specs.remote() for actor in actors.values()
    ])
    forward_specs_by_comm_id: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for shard in receive_spec_shards:
        for comm_id, specs in shard.items():
            forward_specs_by_comm_id.setdefault(comm_id, specs)
    forward_specs_by_edge: dict[tuple[str, str], list[tuple[tuple[int, ...], str]]] = {}
    forward_uid_by_backward_uid: dict[str, str] = {}
    for dag in piper_metadata.per_pp_training_dags:
        for node in dag.nodes.values():
            if node.node_kind == "COMPUTE" and node.compute_subkind in {"BWD", "BWD_I", "BWD_W"}:
                fwd_uid = node.node_meta.get("fwd_uid")
                if fwd_uid is not None:
                    forward_uid_by_backward_uid[node.uid] = str(fwd_uid)
            comm_id = str(node.node_meta.get("comm_id", ""))
            if node.node_kind == "RECV_COMM" and node.tag.get("PASS") == "F" and comm_id in forward_specs_by_comm_id:
                forward_specs_by_edge[(str(node.node_meta["from_uid"]), str(node.node_meta["to_uid"]))] = forward_specs_by_comm_id[comm_id]
    p2p_specs_by_comm_id: dict[str, list[tuple[tuple[int, ...], str]]] = {}
    for dag in piper_metadata.per_pp_training_dags:
        for node in dag.nodes.values():
            if node.node_kind not in {"SEND_COMM", "RECV_COMM"}:
                continue
            comm_id = str(node.node_meta["comm_id"])
            if node.tag.get("PASS") == "F":
                specs = forward_specs_by_comm_id.get(comm_id)
            else:
                source = forward_uid_by_backward_uid.get(str(node.node_meta["from_uid"]))
                target = forward_uid_by_backward_uid.get(str(node.node_meta["to_uid"]))
                specs = (
                    None if source is None or target is None
                    else forward_specs_by_edge.get((target, source))
                )
            if specs is None:
                raise RuntimeError(f"missing receiver payload metadata for P2P {comm_id}")
            p2p_specs_by_comm_id[comm_id] = specs
    # Bucketing can leave large unused cached blocks after the measured
    # training steps.  Release them before allocating the standalone P2P
    # payloads so the microprofile does not fail from allocator fragmentation.
    # This happens outside every measured interval and does not affect either
    # policy's recorded node times.
    ray.get([
        actor.clear_profile_allocator_cache.remote()
        for actor in actors.values()
    ])
    local_p2p_baseline_shards = ray.get([
        actor.profile_p2p_transfer_baselines.remote(p2p_specs_by_comm_id)
        for actor in actors.values()
    ])
    local_p2p_baselines = {
        comm_id: max(shard[comm_id] for shard in local_p2p_baseline_shards if comm_id in shard)
        for comm_id in p2p_specs_by_comm_id
    }

    local_results = []
    local_instrumented_step_samples_us: dict[str, tuple[float, ...]] = {}
    requested_repeat_counts = {
        int(configs[mode].repeat_iters) for mode in requested_modes
    }
    # CUPTI can very occasionally omit one activity in an otherwise complete
    # node range.  Capture a few spare whole steps so those corrupt paired
    # samples can be rejected without weakening exact activity validation.
    kineto_spare_iters = (
        6
        if set(requested_modes)
        == {INTERFERENCE_FREE_PROFILE_MODE, NORMAL_PROFILE_MODE}
        and len(requested_repeat_counts) == 1
        else 0
    )
    # Every policy must profile the same training state.  Restoring only at a
    # policy boundary is insufficient for data-dependent MoE kernels: small
    # numerical differences between the normal and IF schedules accumulate
    # across optimizer steps, alter expert token counts, and make grouped GEMM
    # dispatch different CUTLASS kernels.  Capture once, then restore before
    # every warmup/profile occurrence outside its timed interval.  This makes
    # each occurrence a repeat measurement of the same logical CUDA work.
    ray.get([
        actor.capture_profile_training_state.remote()
        for actor in actors.values()
    ])
    for mode in requested_modes:
        mode_config = configs[mode]
        clock_calibrations = _calibrate_profile_actor_clocks(actors)
        ray.get([
            actor.restore_profile_training_state.remote()
            for actor in actors.values()
        ])
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])
        ray.get([
            actors[rank].start_cupti_node_profiler.remote(
                warmup_iters=mode_config.warmup_iters,
                active_iters=mode_config.repeat_iters + kineto_spare_iters,
                coordinator_clock_offset_ns=clock_calibrations[rank][0],
                clock_sync_uncertainty_ns=clock_calibrations[rank][1],
            )
            for rank in sorted(actors)
        ])
        mode_step_samples_us: list[float] = []
        for iteration in range(
            mode_config.warmup_iters
            + mode_config.repeat_iters
            + kineto_spare_iters
        ):
            ray.get([
                actor.restore_profile_training_state.remote()
                for actor in actors.values()
            ])
            active = iteration >= mode_config.warmup_iters
            if active:
                ray.get([
                    actor.synchronize_device.remote()
                    for actor in actors.values()
                ])
                started = time.perf_counter()
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    synchronize_before=not active,
                    profile_mode=mode,
                    advance_profiler=False,
                    # Allocator snapshots are diagnostics, not part of the
                    # represented training step.  In particular,
                    # memory_snapshot() can dominate short-kernel schedules.
                    capture_allocator_stats=False,
                )
                for actor in actors.values()
            ])
            if active:
                ray.get([
                    actor.synchronize_device.remote()
                    for actor in actors.values()
                ])
                mode_step_samples_us.append(
                    (time.perf_counter() - started) * 1_000_000.0
                )
            # Keep the profiler state transition and final Kineto flush outside
            # the same-ID synchronized step interval used by validation 4.
            ray.get([
                actor.step_pytorch_profiler.remote()
                for actor in actors.values()
            ])
        local_instrumented_step_samples_us[mode] = tuple(
            mode_step_samples_us
        )
        local_results.extend(ray.get([
            actor.stop_cupti_profile_result.remote(
                estimator=mode_config.estimator,
                profile_mode=mode,
                p2p_baselines_by_comm_id=local_p2p_baselines,
                trace_output_dir=trace_output_dir,
                trace_ranks=(
                    None
                    if trace_ranks is None
                    else tuple(int(rank) for rank in trace_ranks)
                ),
            )
            for actor in actors.values()
        ]))

    # Profiling is observational: leave the actors at the state at which the
    # paired capture began rather than after the final policy's optimizer steps.
    ray.get([
        actor.restore_profile_training_state.remote()
        for actor in actors.values()
    ])

    # Keep policy-level wall-clock measurements separate from node timing.
    # They are diagnostics only; node durations above remain CUPTI-derived.
    local_uninstrumented_step_samples_us: dict[str, tuple[float, ...]] = {}
    if measure_uninstrumented_modes:
        local_uninstrumented_step_samples_us = piper_measure_profile_modes(
            loss_fn,
            warmup_iters=max(configs[mode].warmup_iters for mode in requested_modes),
            repeat_iters=max(configs[mode].repeat_iters for mode in requested_modes),
            profile_modes=requested_modes,
        )

    all_results = local_results
    compare_both_modes = set(requested_modes) == {
        INTERFERENCE_FREE_PROFILE_MODE,
        NORMAL_PROFILE_MODE,
    }
    raw_activity_equivalence = (
        _compare_kineto_activity_signatures(all_results)
        if compare_both_modes
        else {
            "comparison": "not_applicable_single_profile_mode",
            "exact_match": True,
            "accepted": True,
            "capture_loss_only": False,
            "capture_loss_within_tolerance": False,
            "capture_loss_rate": 0.0,
            "capture_loss_threshold": _MAX_KINETO_CAPTURE_LOSS_RATE,
            "expected_cpu_launch_records": 0,
            "missing_cpu_launch_records": 0,
            "cpu_launch_loss_rate": 0.0,
            "cpu_launch_loss_threshold": (
                _MAX_KINETO_CPU_LAUNCH_LOSS_RATE
            ),
            "expected_gpu_activity_records": 0,
            "missing_gpu_activity_records": 0,
            "actors_compared": 0,
            "nodes_compared": 0,
            "node_ranges_compared": 0,
            "gpu_activities_compared": 0,
            "mismatched_nodes": 0,
            "mismatched_node_ranges": 0,
            "missing_mode_ranks": {
                INTERFERENCE_FREE_PROFILE_MODE: [],
                NORMAL_PROFILE_MODE: [],
            },
            "mismatches": [],
        }
    )
    nonexact_iteration_ids = sorted({
        int(item["occurrence"])
        for mismatch in raw_activity_equivalence.get("mismatches", ())
        for item in mismatch.get("ranges", ())
        if item.get("occurrence") is not None
    })
    unfilterable_mismatches = [
        mismatch
        for mismatch in raw_activity_equivalence.get("mismatches", ())
        if not mismatch.get("ranges")
    ]
    requested_iterations = (
        next(iter(requested_repeat_counts))
        if len(requested_repeat_counts) == 1
        else 0
    )
    captured_iterations = min(
        (
            len(profile.samples_us)
            for result in all_results
            for profile in result.node_profiles.values()
            if profile.executed
        ),
        default=0,
    )
    retained_iteration_ids = [
        index
        for index in range(captured_iterations)
        if index not in set(nonexact_iteration_ids)
    ][:requested_iterations]
    can_select_exact = (
        kineto_spare_iters > 0
        and not any(raw_activity_equivalence["missing_mode_ranks"].values())
        and not unfilterable_mismatches
        and len(retained_iteration_ids) == requested_iterations
    )
    if can_select_exact:
        selection_policy = "globally_exact_spare_filter"
        all_results = _filter_kineto_profile_iterations(
            all_results, retained_iteration_ids
        )
        activity_equivalence = _compare_kineto_activity_signatures(all_results)
    else:
        selection_policy = (
            "deterministic_requested_prefix"
            if requested_iterations > 0
            else "all_captured_iterations"
        )
        retained_iteration_ids = list(range(
            min(captured_iterations, requested_iterations)
            if requested_iterations > 0
            else captured_iterations
        ))
        if len(retained_iteration_ids) != captured_iterations:
            all_results = _filter_kineto_profile_iterations(
                all_results, retained_iteration_ids
            )
            activity_equivalence = _compare_kineto_activity_signatures(
                all_results
            )
        else:
            activity_equivalence = raw_activity_equivalence
        if compare_both_modes:
            raw_capture_accepted = bool(
                raw_activity_equivalence.get("exact_match", False)
                or raw_activity_equivalence.get(
                    "capture_loss_within_tolerance", False
                )
            )
            retained_capture_accepted = bool(
                activity_equivalence.get("exact_match", False)
                or activity_equivalence.get(
                    "capture_loss_within_tolerance", False
                )
            )
            activity_equivalence["raw_capture_accepted"] = (
                raw_capture_accepted
            )
            activity_equivalence["accepted"] = bool(
                raw_capture_accepted and retained_capture_accepted
            )
    _strip_kineto_comparison_details(all_results)
    from .profile_validation import validate_kineto_profile_results

    kineto_validation = validate_kineto_profile_results(
        all_results,
        activity_equivalence,
        raise_on_error=False,
    )
    for dag in piper_metadata.per_pp_training_dags:
        dag.prologue_profile_times_by_mode.clear()
        for node in dag.nodes.values():
            node.profile_times_by_mode.clear()
    plan = merge_actor_profiles(
        piper_metadata.per_pp_training_dags,
        all_results,
        profile_configs=configs,
        topology=dict(getattr(piper_metadata, "schedule_info", {}) or {}),
    )
    plan.profile_metadata["timing_source"] = "kineto_cupti"
    plan.profile_metadata["kineto_activity_equivalence"] = activity_equivalence
    plan.profile_metadata["kineto_raw_activity_equivalence"] = (
        raw_activity_equivalence
    )
    plan.profile_metadata["kineto_sample_rejection"] = {
        "captured_iterations": captured_iterations,
        "requested_iterations": requested_iterations,
        "spare_iterations": kineto_spare_iters,
        "selection_policy": selection_policy,
        "nonexact_iteration_ids": nonexact_iteration_ids,
        "rejected_iteration_ids": [
            index for index in range(captured_iterations)
            if index not in set(retained_iteration_ids)
        ],
        "retained_iteration_ids": retained_iteration_ids,
        "unused_spare_iteration_ids": [
            index
            for index in range(captured_iterations)
            if index not in set(retained_iteration_ids)
        ],
        "raw_mismatched_node_ranges": int(
            raw_activity_equivalence.get("mismatched_node_ranges", 0)
        ),
        "retained_exact_activity_match": bool(
            activity_equivalence.get("exact_match", False)
        ),
        "retained_capture_loss_rate": float(
            activity_equivalence.get("capture_loss_rate", 0.0)
        ),
        "retained_capture_loss_accepted": bool(
            activity_equivalence.get("accepted", False)
            and not activity_equivalence.get("exact_match", False)
        ),
    }
    plan.profile_metadata["automatic_validations"] = {
        "kineto_activity_and_timing": kineto_validation,
    }
    plan.profile_metadata[
        "instrumented_global_step_samples_us_by_mode"
    ] = {
        mode: [
            float(local_instrumented_step_samples_us[mode][index])
            for index in retained_iteration_ids
        ]
        for mode in requested_modes
    }
    if measure_uninstrumented_modes:
        plan.profile_metadata["uninstrumented_global_step_samples_us_by_mode"] = {
            mode: list(local_uninstrumented_step_samples_us.get(mode, ()))
            for mode in requested_modes
        }
    if output_path is not None:
        plan.profile_metadata["profile_output_path"] = str(
            Path(output_path).resolve()
        )
    piper_metadata.profiled_training_plan = plan
    if output_path is not None:
        plan.save_json(output_path)
    if not kineto_validation["passed"]:
        from .profile_validation import ProfileAndSimulateValidationError

        raise ProfileAndSimulateValidationError(
            "Kineto profile validation failed: "
            + "; ".join(kineto_validation["errors"][:8])
        )
    return plan


def piper_measure_profile_modes(
    loss_fn,
    *,
    warmup_iters: int,
    repeat_iters: int,
    block_iters: int = 1,
    profile_modes: Sequence[str] = (
        INTERFERENCE_FREE_PROFILE_MODE,
        NORMAL_PROFILE_MODE,
    ),
) -> dict[str, tuple[float, ...]]:
    """Measure stream policies without per-node CUDA-event instrumentation.

    Mode order alternates so temporal/cache drift is balanced.  Each returned
    sample is program-driver wall time in microseconds and therefore covers all
    actors participating in the step.
    """

    if int(warmup_iters) < 0:
        raise ValueError("warmup_iters must be non-negative")
    if int(repeat_iters) <= 0:
        raise ValueError("repeat_iters must be positive")
    if int(block_iters) <= 0:
        raise ValueError("block_iters must be positive")
    modes = tuple(normalize_profile_mode(mode) for mode in profile_modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError(f"invalid profile modes: {modes!r}")
    actors = piper_metadata.actors
    if not actors:
        raise RuntimeError("piper_setup() must create actors before mode timing")

    samples: dict[str, list[float]] = {mode: [] for mode in modes}
    ray.get([
        actor.capture_profile_training_state.remote()
        for actor in actors.values()
    ])
    for mode in modes:
        ray.get([
            actor.restore_profile_training_state.remote()
            for actor in actors.values()
        ])
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])
        for _ in range(int(warmup_iters)):
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    synchronize_before=False,
                    synchronize=False,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])
        for _ in range(int(repeat_iters)):
            ray.get([
                actor.synchronize_device.remote() for actor in actors.values()
            ])
            started = time.perf_counter()
            for _ in range(int(block_iters)):
                ray.get([
                    actor.profile_run_dag.remote(
                        loss_fn=loss_fn,
                        synchronize_before=False,
                        synchronize=False,
                        profile_mode=mode,
                    )
                    for actor in actors.values()
                ])
            ray.get([
                actor.synchronize_device.remote() for actor in actors.values()
            ])
            samples[mode].append(
                (time.perf_counter() - started)
                * 1_000_000.0
                / int(block_iters)
            )
    ray.get([
        actor.restore_profile_training_state.remote()
        for actor in actors.values()
    ])
    return {mode: tuple(values) for mode, values in samples.items()}


def simulate_profiled_dags(
    plan: ProfiledTrainingPlan | None = None,
    *,
    profile_mode: str = INTERFERENCE_FREE_PROFILE_MODE,
    **simulation_kwargs,
):
    """Run the stream simulator using one named duration dataset."""
    if plan is None:
        plan = getattr(piper_metadata, "profiled_training_plan", None)
    if plan is None:
        raise RuntimeError("piper_profile_dags() must run before simulation")
    from .simulator import simulate_training_step

    return simulate_training_step(
        plan, profile_mode=profile_mode, **simulation_kwargs
    )


def simulate_profiled_modes(
    plan: ProfiledTrainingPlan | None = None,
    *,
    actual_step_samples_s: Sequence[float] | None = None,
    **simulation_kwargs,
):
    """Replay interference-free/normal timings and validate normal replay."""

    if plan is None:
        plan = getattr(piper_metadata, "profiled_training_plan", None)
    if plan is None:
        raise RuntimeError("piper_profile_dags() must run before simulation")
    from .profile_validation import ProfileAndSimulateValidationError

    kineto_validation = plan.profile_metadata.get(
        "automatic_validations", {}
    ).get("kineto_activity_and_timing")
    activity_equivalence = plan.profile_metadata.get(
        "kineto_activity_equivalence"
    )
    if kineto_validation is not None and not kineto_validation.get(
        "passed", False
    ):
        raise ProfileAndSimulateValidationError(
            "saved Kineto activity/timing validation did not pass"
        )
    if activity_equivalence is not None and not activity_equivalence.get(
        "accepted", activity_equivalence.get("exact_match", False)
    ):
        raise ProfileAndSimulateValidationError(
            "saved normal/IF Kineto activity-equivalence validation did not pass"
        )
    from .simulator import simulate_profile_modes

    actual_us = (
        None
        if actual_step_samples_s is None
        else tuple(float(sample) * 1_000_000.0 for sample in actual_step_samples_s)
    )
    comparison = simulate_profile_modes(
        plan,
        actual_step_samples_us=actual_us,
        **simulation_kwargs,
    )
    from .profile_validation import (
        build_and_validate_attributed_trace,
        validate_measured_normal_samples,
        validate_simulation_endpoints,
    )

    endpoint_validation = validate_simulation_endpoints(comparison)
    if endpoint_validation is not None:
        from .attribution import attribute_profile_mode_savings

        # One deterministic order is sufficient for the mandatory structural
        # validation because it must telescope exactly to the replay endpoint.
        # The report consumes only the order-invariant net checksum; higher-
        # sample Shapley attribution remains available through the offline CLI.
        attribution = attribute_profile_mode_savings(
            plan,
            permutations=1,
            seed=0,
            antithetic=False,
            strict=bool(simulation_kwargs.get("strict", True)),
        )
        endpoint_validation.update({
            "attribution_features": len(attribution.features),
            "attribution_permutations": attribution.permutations,
            "attribution_step_time_estimator": (
                attribution.step_time_estimator
            ),
            "attribution_sample_iteration_ids": list(
                attribution.sample_iteration_ids
            ),
            "normal_step_time_us": attribution.normal_step_time_us,
            "interference_free_step_time_us": (
                attribution.interference_free_step_time_us
            ),
            "attributed_step_savings_us": (
                attribution.attributed_step_savings_us
            ),
            "additivity_residual_us": attribution.additivity_residual_us,
            "positive_exposed_step_savings_us": (
                attribution.positive_exposed_step_savings_us
            ),
            "negative_exposed_step_savings_us": (
                attribution.negative_exposed_step_savings_us
            ),
            "net_exposed_step_savings_us": (
                attribution.net_exposed_step_savings_us
            ),
            "net_exposed_step_savings_pct": (
                attribution.net_exposed_step_savings_pct
            ),
            "exposed_savings_validation_residual_us": (
                attribution.exposed_savings_validation_residual_us
            ),
        })
        trace, trace_validation = build_and_validate_attributed_trace(
            comparison.normal,
            attribution,
        )
        measured_validation = validate_measured_normal_samples(
            plan,
            comparison,
            warning_threshold_pct=2.0,
        )
        automatic = plan.profile_metadata.setdefault(
            "automatic_validations", {}
        )
        automatic["simulation_endpoints"] = endpoint_validation
        automatic["attributed_trace_losslessness"] = trace_validation
        automatic["measured_vs_simulated_normal"] = measured_validation

        warning = measured_validation.get("warning")
        if warning:
            logger.warning("profile-and-simulate validation: %s", warning)
            warnings.warn(
                f"profile-and-simulate validation: {warning}",
                RuntimeWarning,
                stacklevel=2,
            )

        output_value = plan.profile_metadata.get("profile_output_path")
        if output_value:
            profile_path = Path(str(output_value))
            output_dir = profile_path.parent
            attribution_path = output_dir / "savings_attribution.json"
            trace_path = output_dir / "savings_attribution_trace.json"
            validation_path = output_dir / "profile_and_simulate_validation.json"
            attribution_path.write_text(
                json.dumps(attribution.to_dict(), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            trace_path.write_text(
                json.dumps(trace, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            validation_path.write_text(
                json.dumps(automatic, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            plan.save_json(profile_path)
    return comparison


def _log_step_stats(step_time: float, log_memory: bool, actors: dict) -> None:
    stats = [f"step_time={step_time:.3f}s"]
    tokens = getattr(piper_metadata, "tokens_per_step", None)
    if tokens is not None:
        stats.append(f"throughput={tokens / step_time:.1f} tok/s")
    logger.info("  ".join(stats))
