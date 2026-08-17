from collections import defaultdict
from types import SimpleNamespace

import pytest

import src.actor as actor_module
from src.actor import (
    PiperActor,
    _CudaEventTimingRecorder,
    _kineto_correlated_activity_envelope_samples_us,
)
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

    def synchronize(self) -> None:
        pass


def test_kineto_simulator_duration_uses_activity_envelope() -> None:
    measured = {
        # This is intentionally much shorter than the complete correlated
        # activity, reproducing key_averages() on an outer Piper range.
        "device_time_mean_us": 3.0,
        "range_count": 2,
        "unmatched_launch_count": 0,
        "range_activity_signatures": [
            {
                "activity_time_sum_us": 11.0,
                "activity_envelope_time_us": 17.0,
            },
            {
                "activity_time_sum_us": 13.0,
                "activity_envelope_time_us": 19.0,
            },
        ],
    }

    assert _kineto_correlated_activity_envelope_samples_us(
        measured, node_uid="backward"
    ) == (17.0, 19.0)


def test_kineto_simulator_duration_rejects_incomplete_attribution() -> None:
    with pytest.raises(RuntimeError, match="left 1 CUDA launches unmatched"):
        _kineto_correlated_activity_envelope_samples_us(
            {
                "range_count": 1,
                "unmatched_launch_count": 1,
                "range_activity_signatures": [
                    {
                        "activity_time_sum_us": 11.0,
                        "activity_envelope_time_us": 17.0,
                    }
                ],
            },
            node_uid="all_gather",
        )


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


def test_actor_retains_start_offsets_for_both_profile_modes(monkeypatch) -> None:
    class _Recorder:
        node_samples_us = {"node": 7.0}
        node_start_offsets_us = {"node": 19.0}
        prologue_sample_us = None

        def __init__(self, **_kwargs) -> None:
            pass

    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor.runtime = SimpleNamespace(pytorch_profiler_enabled=False, device="cuda:0")
    actor.dag = SimpleNamespace(nodes={"node": object()})
    actor._profile_node_samples = {}
    actor._profile_node_start_offsets = {}
    actor._profile_prologue_samples = {}
    actor.profile_prepared_dag_executions = {
        "interference_free": object(),
        "normal": object(),
    }
    execution_calls = []
    actor._execute_dag = lambda **kwargs: execution_calls.append(kwargs)
    monkeypatch.setattr(actor_module, "_CudaEventTimingRecorder", _Recorder)
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda *_args: None)

    actor.profile_run_dag(profile_mode="interference_free")
    actor.profile_run_dag(profile_mode="normal")

    assert actor._profile_node_samples == {
        "interference_free": defaultdict(list, {"node": [7.0]}),
        "normal": defaultdict(list, {"node": [7.0]}),
    }
    assert actor._profile_node_start_offsets["interference_free"] == {
        "node": [19.0]
    }
    assert actor._profile_node_start_offsets["normal"] == {
        "node": [19.0]
    }
    assert execution_calls[0]["serialize_nodes"] is True
    assert execution_calls[1]["serialize_nodes"] is False


def test_profile_run_dag_can_defer_profiler_advance(monkeypatch) -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    steps = []
    actor.runtime = SimpleNamespace(
        pytorch_profiler_enabled=True,
        torch_profiler=SimpleNamespace(step=lambda: steps.append("step")),
        device="cuda:0",
    )
    actor._execute_dag = lambda **_kwargs: None
    actor.profile_prepared_dag_executions = {"interference_free": None}
    actor._cupti_allocator_iteration_stats = []
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda *_args: None)

    actor.profile_run_dag(
        record=False,
        advance_profiler=False,
        capture_allocator_stats=False,
    )
    assert steps == []

    actor.step_pytorch_profiler()
    assert steps == ["step"]


def test_actor_normal_run_uses_execution_prepared_during_setup() -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    captured = {}
    expected_result = object()

    def execute_dag(**kwargs):
        captured.update(kwargs)
        return expected_result

    actor._execute_dag = execute_dag
    actor.runtime = SimpleNamespace(pytorch_profiler_enabled=False)
    loss_fn = object()

    assert actor.run_dag(loss_fn=loss_fn) is expected_result
    assert captured["loss_fn"] is loss_fn
    assert "stream_override" not in captured


def test_pytorch_profiler_uses_warmup_then_active_and_steps_per_dag(
    monkeypatch,
) -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    profiler_steps = []
    schedule_kwargs = {}

    class _FakeProfiler:
        def __enter__(self):
            return self

        def step(self):
            profiler_steps.append("step")

    def fake_schedule(**kwargs):
        schedule_kwargs.update(kwargs)
        return "warmup_then_active"

    def fake_profile(**kwargs):
        assert kwargs["schedule"] == "warmup_then_active"
        return _FakeProfiler()

    monkeypatch.setattr(actor_module.torch.profiler, "schedule", fake_schedule)
    monkeypatch.setattr(actor_module.torch.profiler, "profile", fake_profile)
    actor.runtime = SimpleNamespace(
        global_rank=0,
        pytorch_profiler_enabled=False,
        torch_profiler=None,
    )
    actor.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    expected_result = object()
    actor._execute_dag = lambda **_kwargs: expected_result

    actor.start_pytorch_profiler(warmup_iters=2, active_iters=1)
    assert schedule_kwargs == {"wait": 0, "warmup": 2, "active": 1, "repeat": 1}
    assert actor.run_dag() is expected_result
    assert actor.run_dag() is expected_result
    assert actor.run_dag() is expected_result
    assert profiler_steps == ["step", "step", "step"]


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
