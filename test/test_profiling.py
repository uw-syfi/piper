import json

import pytest

from src.dag import TrainingDAG, TrainingDAGEdge, TrainingDAGNode
import src.directives as directives
from src.profiling import (
    ActorProfileResult,
    NORMAL_PROFILE_MODE,
    INTERFERENCE_FREE_PROFILE_MODE,
    NodeProfile,
    ProfileConfig,
    ProfiledTrainingPlan,
    TensorProfileSpec,
    finalized_dag_fingerprint,
    get_node_profile_times,
    get_prologue_profile_times,
    merge_actor_profiles,
    normalize_profile_mode,
)


def _node(
    uid: str,
    *,
    device: list[int] | None = None,
    stream: str = "default_stream",
    subkind: str = "FWD",
) -> TrainingDAGNode:
    return TrainingDAGNode(
        uid=uid,
        node_kind="COMPUTE",
        compute_subkind=subkind,
        tag={"PASS": "F" if subkind == "FWD" else "B"},
        device=[0] if device is None else device,
        stream=stream,
        node_meta={"bucket_key": uid},
    )


def _two_node_dag(device: list[int] | None = None) -> TrainingDAG:
    dag = TrainingDAG()
    dag.add_node(_node("a", device=device))
    dag.add_node(_node("b", device=device, stream="other"))
    dag.add_edge(TrainingDAGEdge("a", "b", "data", "activation"))
    return dag


def test_node_profile_estimator_and_json_round_trip() -> None:
    spec = TensorProfileSpec(shape=(2, 4), dtype="torch.bfloat16", bytes=16)
    profile = NodeProfile.from_samples((3.0, 1.0, 2.0), tensor_specs=(spec,))

    assert profile.estimate_us == 2.0
    assert NodeProfile.from_dict(profile.to_dict()) == profile
    assert NodeProfile.skipped("zero-byte collective").executed is False


def test_node_profile_preserves_explicit_iteration_alignment() -> None:
    profile = NodeProfile.from_samples(
        (3.0, 1.0, 2.0), sample_iteration_ids=(20, 30, 40)
    )

    assert profile.sample_iteration_ids == (20, 30, 40)
    assert NodeProfile.from_dict(profile.to_dict()) == profile
    with pytest.raises(ValueError, match="one entry per raw sample"):
        NodeProfile.from_samples((1.0, 2.0), sample_iteration_ids=(7,))


def test_node_profile_preserves_iteration_aligned_start_offsets() -> None:
    profile = NodeProfile.from_samples(
        (3.0, 1.0, 2.0),
        sample_iteration_ids=(20, 30, 40),
        start_offset_samples_us=(100.0, 120.0, 110.0),
    )

    assert profile.start_offset_samples_us == (100.0, 120.0, 110.0)
    assert profile.start_offset_estimate_us == 110.0
    assert NodeProfile.from_dict(profile.to_dict()) == profile
    with pytest.raises(ValueError, match="one entry per raw sample"):
        NodeProfile.from_samples(
            (1.0, 2.0), start_offset_samples_us=(10.0,)
        )


def test_node_profile_loads_duration_only_artifact_without_start_offsets() -> None:
    encoded = NodeProfile.from_samples((2.0, 3.0)).to_dict()
    encoded.pop("start_offset_samples_us")
    encoded.pop("start_offset_estimate_us")

    restored = NodeProfile.from_dict(encoded)

    assert restored.start_offset_samples_us == ()
    assert restored.start_offset_estimate_us is None


def test_profile_config_uses_fixed_iteration_count() -> None:
    config = ProfileConfig()

    assert config.warmup_iters == 5
    assert config.repeat_iters == 30


def test_actor_profile_result_has_no_legacy_dp_metadata() -> None:
    result = ActorProfileResult(
        graph_fingerprint="fingerprint",
        global_rank=7,
        pp_rank=2,
        node_profiles={},
    )

    encoded = result.to_dict()
    assert "dp_rank" not in encoded

    # Older transient payloads may contain extra metadata.  It must not be
    # restored into the ActorGroup-native result type.
    encoded["dp_rank"] = 3
    restored = ActorProfileResult.from_dict(encoded)
    assert restored == result
    assert not hasattr(restored, "dp_rank")


def test_finalized_fingerprint_is_order_and_profile_invariant() -> None:
    dag = _two_node_dag()
    expected = finalized_dag_fingerprint(dag)

    reordered = TrainingDAG()
    reordered.add_node(_node("b", stream="other"))
    reordered.add_node(_node("a"))
    reordered.add_edge(TrainingDAGEdge("a", "b", "data", "activation"))
    reordered.nodes["a"].profile_times_by_mode["interference_free"] = {
        0: NodeProfile.from_samples((5.0,))
    }

    assert finalized_dag_fingerprint(reordered) == expected
    reordered.nodes["a"].stream = "changed"
    assert finalized_dag_fingerprint(reordered) != expected


def test_merge_actor_profiles_attaches_rank_keyed_measurements() -> None:
    dag = _two_node_dag(device=[0, 2])
    fingerprint = finalized_dag_fingerprint(dag)
    rank0 = ActorProfileResult(
        graph_fingerprint=fingerprint,
        global_rank=0,
        pp_rank=0,
        node_profiles={
            "a": NodeProfile.from_samples((10.0, 12.0)),
            "b": NodeProfile.from_samples((20.0, 22.0)),
        },
        prologue_profile=NodeProfile.from_samples((1.0,)),
    )
    rank2 = ActorProfileResult(
        graph_fingerprint=fingerprint,
        global_rank=2,
        pp_rank=0,
        node_profiles={
            "a": NodeProfile.from_samples((11.0,)),
            "b": NodeProfile.from_samples((21.0,)),
        },
    )

    plan = merge_actor_profiles([dag], [rank2, rank0], topology={"world_size": 4})

    assert plan.per_pp_dags[0] is dag
    profiles = get_node_profile_times(dag.nodes["a"], "interference_free")
    assert profiles[0].estimate_us == 11.0
    assert profiles[2].estimate_us == 11.0
    assert get_prologue_profile_times(dag, "interference_free")[0].estimate_us == 1.0
    assert plan.topology == {"world_size": 4}


def test_merge_actor_profiles_keeps_interference_free_and_normal_datasets() -> None:
    dag = _two_node_dag(device=[0])
    fingerprint = finalized_dag_fingerprint(dag)

    def _result(mode: str, duration: float) -> ActorProfileResult:
        return ActorProfileResult(
            graph_fingerprint=fingerprint,
            global_rank=0,
            pp_rank=0,
            profile_mode=mode,
            node_profiles={
                uid: NodeProfile.from_samples(
                    (duration, duration + 1), sample_iteration_ids=(10, 11)
                )
                for uid in dag.nodes
            },
            prologue_profile=NodeProfile.from_samples(
                (duration,), sample_iteration_ids=(10,)
            ),
        )

    plan = merge_actor_profiles(
        [dag],
        [
            _result(INTERFERENCE_FREE_PROFILE_MODE, 10.0),
            _result(NORMAL_PROFILE_MODE, 20.0),
        ],
        profile_configs={
            INTERFERENCE_FREE_PROFILE_MODE: ProfileConfig(repeat_iters=2),
            NORMAL_PROFILE_MODE: ProfileConfig(
                repeat_iters=2,
                isolation_scope="logical_streams_with_directional_p2p",
            ),
        },
    )

    assert plan.available_profile_modes == ("interference_free", "normal")
    assert get_node_profile_times(dag.nodes["a"], "interference_free")[0].estimate_us == 10.5
    assert get_node_profile_times(dag.nodes["a"], "normal")[0].estimate_us == 20.5
    assert get_prologue_profile_times(dag, "normal")[0].estimate_us == 20.0
    assert plan.config_for_mode("normal").repeat_iters == 2


def test_merge_actor_profiles_aligns_cuda_offsets_across_actor_clocks() -> None:
    dag = _two_node_dag(device=[0, 2])
    fingerprint = finalized_dag_fingerprint(dag)

    def _result(rank: int, origin_us: float) -> ActorProfileResult:
        return ActorProfileResult(
            graph_fingerprint=fingerprint,
            global_rank=rank,
            pp_rank=0,
            profile_mode=NORMAL_PROFILE_MODE,
            node_profiles={
                "a": NodeProfile.from_samples(
                    (3.0,), start_offset_samples_us=(5.0,)
                ),
                "b": NodeProfile.from_samples(
                    (4.0,), start_offset_samples_us=(9.0,)
                ),
            },
            timeline_origin_samples_us=(origin_us,),
            timeline_origin_uncertainty_samples_us=(2.0,),
        )

    plan = merge_actor_profiles(
        [dag], [_result(0, 1000.0), _result(2, 1012.0)]
    )

    profiles = get_node_profile_times(dag.nodes["a"], NORMAL_PROFILE_MODE)
    assert profiles[0].start_offset_samples_us == pytest.approx((5.0,))
    assert profiles[2].start_offset_samples_us == pytest.approx((17.0,))
    alignment = plan.profile_metadata["timeline_alignment"]
    assert (
        alignment["method"]
        == "driver_actor_midpoint_plus_kineto_iteration_anchor"
    )
    assert alignment["modes"][NORMAL_PROFILE_MODE]["max_origin_uncertainty_us"] == 2.0
    assert alignment["modes"][NORMAL_PROFILE_MODE]["max_pairwise_uncertainty_us"] == 4.0
    restored = ProfiledTrainingPlan.from_dict(plan.to_dict())
    restored_profiles = get_node_profile_times(
        restored.per_pp_dags[0].nodes["a"], NORMAL_PROFILE_MODE
    )
    assert restored_profiles[2].start_offset_samples_us == pytest.approx((17.0,))
    assert restored.profile_metadata["timeline_alignment"] == alignment


def test_merge_actor_profiles_rejects_misaligned_iteration_ids() -> None:
    dag = _two_node_dag(device=[0, 2])
    fingerprint = finalized_dag_fingerprint(dag)

    def result(rank: int, sample_id: int) -> ActorProfileResult:
        return ActorProfileResult(
            graph_fingerprint=fingerprint,
            global_rank=rank,
            pp_rank=0,
            profile_mode=NORMAL_PROFILE_MODE,
            node_profiles={
                uid: NodeProfile.from_samples(
                    (1.0,),
                    sample_iteration_ids=(sample_id,),
                    start_offset_samples_us=(0.0,),
                )
                for uid in dag.nodes
            },
            timeline_origin_samples_us=(1000.0,),
            timeline_origin_uncertainty_samples_us=(1.0,),
        )

    with pytest.raises(ValueError, match="mismatched actor sample iteration IDs"):
        merge_actor_profiles([dag], [result(0, 3), result(2, 4)])


def test_merge_actor_profiles_fails_closed_on_missing_node_or_fingerprint() -> None:
    dag = _two_node_dag()
    with pytest.raises(ValueError, match="node coverage mismatch"):
        merge_actor_profiles(
            [dag],
            [
                ActorProfileResult(
                    graph_fingerprint=finalized_dag_fingerprint(dag),
                    global_rank=0,
                    node_profiles={"a": NodeProfile.from_samples((1.0,))},
                )
            ],
        )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        merge_actor_profiles(
            [dag],
            [
                ActorProfileResult(
                    graph_fingerprint="not-the-finalized-graph",
                    global_rank=0,
                    node_profiles={
                        uid: NodeProfile.from_samples((1.0,)) for uid in dag.nodes
                    },
                )
            ],
        )


def test_merge_actor_profiles_requires_every_actor_group_rank() -> None:
    dag = _two_node_dag(device=[0, 2])
    fingerprint = finalized_dag_fingerprint(dag)
    result = ActorProfileResult(
        graph_fingerprint=fingerprint,
        global_rank=0,
        node_profiles={
            uid: NodeProfile.from_samples((1.0,)) for uid in dag.nodes
        },
    )

    with pytest.raises(ValueError, match=r"missing=\[2\]"):
        merge_actor_profiles(
            [dag],
            [result],
            topology={
                "actor_groups": [
                    {"id": 0, "devices": [0, 2], "size": 2},
                ]
            },
        )


def test_profiled_plan_slim_round_trip_drops_live_metadata() -> None:
    dag = _two_node_dag()
    dag.nodes["a"].node_meta["gm"] = object()
    dag.nodes["a"].node_meta["safe"] = {"shape": [2, 3]}
    for node in dag.nodes.values():
        node.profile_times_by_mode["interference_free"] = {
            0: NodeProfile.from_samples((2.5,))
        }
    plan = ProfiledTrainingPlan.from_dags(
        [dag],
        profile_configs={
            "interference_free": ProfileConfig(warmup_iters=1, repeat_iters=2)
        },
    )

    encoded = json.loads(json.dumps(plan.to_dict()))
    restored = ProfiledTrainingPlan.from_dict(encoded)

    assert restored.graph_fingerprint == plan.graph_fingerprint
    assert "gm" not in restored.per_pp_dags[0].nodes["a"].node_meta
    assert restored.per_pp_dags[0].nodes["a"].node_meta["safe"] == {"shape": [2, 3]}
    assert get_node_profile_times(
        restored.per_pp_dags[0].nodes["a"], "interference_free"
    )[0].estimate_us == 2.5


@pytest.mark.parametrize("mode", ("isolated", "concurrent"))
def test_removed_profile_mode_aliases_are_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="unsupported profile mode"):
        normalize_profile_mode(mode)


def test_p2p_endpoints_share_stable_identity_and_directional_group() -> None:
    dag = TrainingDAG()
    dag.add_node(_node("stage0", device=[0, 2]))
    dag.add_node(_node("stage1", device=[1, 3]))
    dag.add_edge(TrainingDAGEdge("stage0", "stage1", "data", "x"))

    directives._insert_send_recv_comm_nodes(dag)

    send = next(node for node in dag.nodes.values() if node.node_kind == "SEND_COMM")
    recv = next(node for node in dag.nodes.values() if node.node_kind == "RECV_COMM")
    assert send.node_meta["comm_id"] == recv.node_meta["comm_id"]
    assert send.node_meta["comm_group_kind"] == "pp_lo_hi"
    assert recv.node_meta["comm_group_kind"] == "pp_lo_hi"
    assert send.node_meta["participants"] == [0, 1, 2, 3]


def test_collective_metadata_uses_node_devices() -> None:
    dag = TrainingDAG()
    backward = _node("backward", device=[0, 2], subkind="BWD")
    backward.tag.update({"PP": 0, "EP": 0})
    dag.add_node(backward)

    directives._insert_reduce_comm_nodes(dag, [{"PP": 0, "EP": 0}], [0, 2])

    collective = next(
        node for node in dag.nodes.values() if node.node_kind == "REDUCE_COMM"
    )
    assert collective.node_meta["comm_id"] == "all_reduce:backward"
    assert collective.node_meta["comm_group_kind"] == "dp"
    assert collective.node_meta["participants"] == [0, 2]
