from collections import defaultdict
from types import SimpleNamespace

import pytest

import src.actor as actor_module
from src.actor import PiperActor, _CudaEventTimingRecorder
from src.dag import TrainingDAG, TrainingDAGNode


class _FakeStream:
    def __init__(self, timestamp_ms: float = 0.0) -> None:
        self.timestamp_ms = timestamp_ms


class _FakeEvent:
    def __init__(self) -> None:
        self.timestamp_ms = 0.0

    def record(self, stream: _FakeStream) -> None:
        self.timestamp_ms = stream.timestamp_ms

    def elapsed_time(self, end: "_FakeEvent") -> float:
        return end.timestamp_ms - self.timestamp_ms


def test_cuda_event_recorder_reuses_events_for_cross_stream_start_offsets(
    monkeypatch,
) -> None:
    recorder = _CudaEventTimingRecorder()
    monkeypatch.setattr(recorder, "_event", lambda: _FakeEvent())
    default = _FakeStream(10.0)
    compute = _FakeStream(12.5)
    communication = _FakeStream(13.25)

    recorder.before_prologue(default)
    default.timestamp_ms = 10.5
    recorder.after_prologue(default)
    recorder.before_node(SimpleNamespace(uid="compute"), compute)
    compute.timestamp_ms = 14.0
    recorder.after_node(SimpleNamespace(uid="compute"), compute)
    recorder.before_node(SimpleNamespace(uid="comm"), communication)
    communication.timestamp_ms = 13.75
    recorder.after_node(SimpleNamespace(uid="comm"), communication)
    recorder.finish()

    assert recorder.prologue_sample_us == pytest.approx(500.0)
    assert recorder.node_samples_us == pytest.approx(
        {"compute": 1500.0, "comm": 500.0}
    )
    assert recorder.node_start_offsets_us == pytest.approx(
        {"compute": 2500.0, "comm": 3250.0}
    )


def test_actor_only_retains_start_offsets_for_normal_profiles(monkeypatch) -> None:
    class _Recorder:
        node_samples_us = {"node": 7.0}
        node_start_offsets_us = {"node": 19.0}
        prologue_sample_us = None

    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor.runtime = SimpleNamespace(pytorch_profiler_enabled=False, device="cuda:0")
    actor.dag = SimpleNamespace(nodes={"node": object()})
    actor._profile_node_samples = {}
    actor._profile_node_start_offsets = {}
    actor._profile_prologue_samples = {}
    actor._execute_dag = lambda **_kwargs: None
    monkeypatch.setattr(actor_module, "_CudaEventTimingRecorder", _Recorder)
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda *_args: None)

    actor.profile_run_dag(profile_mode="interference_free")
    actor.profile_run_dag(profile_mode="normal")

    assert actor._profile_node_samples == {
        "interference_free": defaultdict(list, {"node": [7.0]}),
        "normal": defaultdict(list, {"node": [7.0]}),
    }
    assert "interference_free" not in actor._profile_node_start_offsets
    assert actor._profile_node_start_offsets["normal"] == {
        "node": [19.0]
    }


def test_actor_normal_run_uses_directional_execution_stream_policy() -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    directional_stream = object()
    actor.runtime = SimpleNamespace(
        execution_stream_for_task=lambda _node, _logical: directional_stream
    )
    captured = {}
    expected_result = object()

    def execute_dag(**kwargs):
        captured.update(kwargs)
        return expected_result

    actor._execute_dag = execute_dag
    loss_fn = object()

    assert actor.run_dag(loss_fn=loss_fn) is expected_result
    assert captured["loss_fn"] is loss_fn
    assert (
        captured["stream_override"](object(), object())
        is directional_stream
    )


def test_actor_result_keeps_normal_duration_and_offset_samples_paired(
    monkeypatch,
) -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    dag = TrainingDAG()
    dag.add_node(
        TrainingDAGNode(
            uid="node",
            node_kind="COMPUTE",
            compute_subkind="FWD",
            tag={"PASS": "F"},
            device=[0],
            stream="compute_stream",
        )
    )
    actor.dag = dag
    actor.runtime = SimpleNamespace(
        device="cuda:0", global_rank=0, dp_rank=0, pp_rank=0
    )
    actor._profile_node_samples = {"normal": {"node": [7.0, 8.0]}}
    actor._profile_node_start_offsets = {
        "normal": {"node": [19.0, 23.0]}
    }
    actor._profile_prologue_samples = {}
    monkeypatch.setattr(
        actor_module.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            name="fake", major=9, minor=0, total_memory=1024
        ),
    )

    result = actor.get_profile_result(profile_mode="normal")
    profile = result.node_profiles["node"]

    assert profile.samples_us == (7.0, 8.0)
    assert profile.sample_iteration_ids == (0, 1)
    assert profile.start_offset_samples_us == (19.0, 23.0)
    assert profile.start_offset_estimate_us == 21.0
