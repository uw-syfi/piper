import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator, Mapping, Sequence

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
    merge_actor_profiles,
    normalize_profile_mode,
)
from .state import LOG_LEVEL, create_logger, piper_metadata
from .visualization import (
    log_training_dag_dependencies,
    print_training_dag_order,
    render_training_dag,
)
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


def piper_exec_dag(loss_fn, log_stats: bool = False) -> list:
    """Execute one training step using the loaded per-rank TrainingDAG."""
    actors = piper_metadata.actors
    run_refs = [
        actor.run_dag.remote(loss_fn=loss_fn)
        for actor in actors.values()
    ]
    t0 = time.perf_counter()
    results = ray.get(run_refs)
    step_time = time.perf_counter() - t0

    if log_stats:
        _log_step_stats(step_time, log_stats, actors)

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
        piper_exec_dag(loss_fn, log_stats=log_stats)
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


def piper_profile_dags(
    loss_fn,
    config: ProfileConfig | None = None,
    *,
    profile_modes: Sequence[str] | None = None,
    profile_configs: Mapping[str, ProfileConfig] | None = None,
    output_path: str | None = None,
    measure_uninstrumented_modes: bool = False,
) -> ProfiledTrainingPlan:
    """Collect and merge one or more timing modes across every actor.

    The program driver launches all actors for every configured iteration and
    merges their rank-local profile results into one durable plan.
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
    if not requested_modes:
        raise ValueError("piper_profile_dags requires at least one profile mode")
    if len(set(requested_modes)) != len(requested_modes):
        raise ValueError(f"duplicate profile modes requested: {requested_modes!r}")
    configs: dict[str, ProfileConfig] = {}
    for mode in requested_modes:
        selected = (profile_configs or {}).get(mode, base_config)
        if mode == NORMAL_PROFILE_MODE and (
            selected.isolation_scope != DEFAULT_NORMAL_SCOPE
        ):
            selected = replace(selected, isolation_scope=DEFAULT_NORMAL_SCOPE)
        configs[mode] = selected

    actors = piper_metadata.actors
    if not actors:
        raise RuntimeError("piper_setup() must create actors before profiling")
    if not piper_metadata.per_pp_training_dags:
        raise RuntimeError("piper_setup() must finalize TrainingDAGs before profiling")

    store = piper_metadata.compiled_data_store
    if store is None:
        from .compile import _COMPILED_DATA_ACTOR

        store = ray.get_actor(_COMPILED_DATA_ACTOR)
        piper_metadata.compiled_data_store = store

    actor_results = []
    step_samples_us_by_mode: dict[str, list[float]] = {}
    for mode in requested_modes:
        mode_config = configs[mode]
        # Give every actor a mapping to the same coordinator-owned monotonic
        # clock before collecting CUDA timeline origins for this mode.
        ray.get([
            actor.calibrate_profile_clock.remote(store)
            for actor in actors.values()
        ])
        ray.get([
            actor.reset_profile_samples.remote(profile_mode=mode)
            for actor in actors.values()
        ])
        for _ in range(mode_config.warmup_iters):
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=False,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])

        # Memory-tight workloads can leave large, fragmented cached regions
        # after warmup.  Release only unused blocks before starting the timed
        # series; this call is outside every measured interval.
        ray.get([
            actor.clear_profile_allocator_cache.remote()
            for actor in actors.values()
        ])

        step_samples_us: list[float] = []
        for _ in range(mode_config.repeat_iters):
            started = time.perf_counter()
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=True,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
            step_samples_us.append((time.perf_counter() - started) * 1_000_000.0)

        step_samples_us_by_mode[mode] = step_samples_us
        actor_results.extend(
            ray.get([
                actor.get_profile_result.remote(
                    estimator=mode_config.estimator,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
        )

    # CUDA events around every node perturb launch timing and can materially
    # change overlap.  When requested, measure the two stream policies again
    # without per-node events.  Alternate their order on every sample so
    # allocator/cache drift is balanced instead of being attributed to one
    # mode.  Actor-side synchronization makes each driver wall sample cover
    # the work just launched rather than the preceding asynchronous step.
    uninstrumented_step_samples_us_by_mode: dict[str, list[float]] = {}
    if measure_uninstrumented_modes:
        uninstrumented_step_samples_us_by_mode = {
            mode: [] for mode in requested_modes
        }
        max_warmups = max(configs[mode].warmup_iters for mode in requested_modes)
        for warmup_index in range(max_warmups):
            order = (
                requested_modes
                if warmup_index % 2 == 0
                else tuple(reversed(requested_modes))
            )
            for mode in order:
                if warmup_index >= configs[mode].warmup_iters:
                    continue
                ray.get([
                    actor.profile_run_dag.remote(
                        loss_fn=loss_fn,
                        record=False,
                        synchronize=True,
                        profile_mode=mode,
                    )
                    for actor in actors.values()
                ])

        max_repeats = max(configs[mode].repeat_iters for mode in requested_modes)
        for sample_index in range(max_repeats):
            order = (
                requested_modes
                if sample_index % 2 == 0
                else tuple(reversed(requested_modes))
            )
            for mode in order:
                if sample_index >= configs[mode].repeat_iters:
                    continue
                started = time.perf_counter()
                ray.get([
                    actor.profile_run_dag.remote(
                        loss_fn=loss_fn,
                        record=False,
                        synchronize=True,
                        profile_mode=mode,
                    )
                    for actor in actors.values()
                ])
                uninstrumented_step_samples_us_by_mode[mode].append(
                    (time.perf_counter() - started) * 1_000_000.0
                )

    # A later profiling call replaces, rather than mixes with, old samples.
    for dag in piper_metadata.per_pp_training_dags:
        dag.prologue_profile_times_by_mode.clear()
        for node in dag.nodes.values():
            node.profile_times_by_mode.clear()
    plan = merge_actor_profiles(
        piper_metadata.per_pp_training_dags,
        actor_results,
        profile_configs=configs,
        topology=dict(getattr(piper_metadata, "schedule_info", {}) or {}),
    )
    plan.profile_metadata["instrumented_global_step_samples_us_by_mode"] = {
        mode: list(step_samples_us_by_mode[mode])
        for mode in requested_modes
    }
    if measure_uninstrumented_modes:
        plan.profile_metadata[
            "uninstrumented_global_step_samples_us_by_mode"
        ] = {
            mode: list(uninstrumented_step_samples_us_by_mode[mode])
            for mode in requested_modes
        }
    piper_metadata.profiled_training_plan = plan
    if output_path is not None:
        plan.save_json(output_path)
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

    for mode in modes:
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])
        for _ in range(int(warmup_iters)):
            ray.get([
                actor.profile_run_dag.remote(
                    loss_fn=loss_fn,
                    record=False,
                    synchronize_before=False,
                    synchronize=False,
                    profile_mode=mode,
                )
                for actor in actors.values()
            ])
        ray.get([actor.synchronize_device.remote() for actor in actors.values()])

    samples: dict[str, list[float]] = {mode: [] for mode in modes}
    for iteration in range(int(repeat_iters)):
        order = modes if iteration % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            ray.get([
                actor.synchronize_device.remote() for actor in actors.values()
            ])
            started = time.perf_counter()
            for _ in range(int(block_iters)):
                ray.get([
                    actor.profile_run_dag.remote(
                        loss_fn=loss_fn,
                        record=False,
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
    from .simulator import simulate_profile_modes

    actual_us = (
        None
        if actual_step_samples_s is None
        else tuple(float(sample) * 1_000_000.0 for sample in actual_step_samples_s)
    )
    return simulate_profile_modes(
        plan,
        actual_step_samples_us=actual_us,
        **simulation_kwargs,
    )


def _log_step_stats(step_time: float, log_memory: bool, actors: dict) -> None:
    stats = [f"step_time={step_time:.3f}s"]
    tokens = getattr(piper_metadata, "tokens_per_step", None)
    if tokens is not None:
        stats.append(f"throughput={tokens / step_time:.1f} tok/s")
    logger.info("  ".join(stats))
