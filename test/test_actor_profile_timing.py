import json
from types import SimpleNamespace

import pytest

import src.actor as actor_module
from src.actor import (
    PiperActor,
    _kineto_node_attribution_audit,
    _kineto_correlated_activity_envelope_samples_us,
)
from src.piper import _select_actor_clock_calibration


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


def test_kineto_simulator_duration_defers_incomplete_launch_validation() -> None:
    assert _kineto_correlated_activity_envelope_samples_us(
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
    ) == (17.0,)


def test_kineto_audit_aligns_activity_and_zero_activity_to_iteration_anchors(
    tmp_path,
) -> None:
    first_anchor = "piper_profile_timeline_anchor:normal:2:rank0"
    second_anchor = "piper_profile_timeline_anchor:normal:3:rank0"
    events = [
        {
            "ph": "X", "cat": "user_annotation", "name": first_anchor,
            "pid": 1, "tid": 2, "ts": 100.0, "dur": 50.0,
        },
        {
            "ph": "X", "cat": "user_annotation", "name": "COMPUTE:uidnode",
            "pid": 1, "tid": 2, "ts": 105.0, "dur": 30.0,
        },
        {
            "ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
            "pid": 1, "tid": 2, "ts": 110.0, "dur": 1.0,
            "args": {"External id": 7, "correlation": 99},
        },
        {
            "ph": "X", "cat": "cuda_driver", "name": "cuLaunchKernel",
            "pid": 1, "tid": 2, "ts": 110.2, "dur": 0.5,
            "args": {"External id": 7, "correlation": 99},
        },
        {
            "ph": "X", "cat": "kernel", "name": "kernel",
            "pid": 9, "tid": 10, "ts": 120.0, "dur": 5.0,
            "args": {"External id": 7, "correlation": 99},
        },
        {
            "ph": "X", "cat": "user_annotation", "name": second_anchor,
            "pid": 1, "tid": 2, "ts": 200.0, "dur": 50.0,
        },
        {
            "ph": "X", "cat": "user_annotation", "name": "COMPUTE:uidnode",
            "pid": 1, "tid": 2, "ts": 206.0, "dur": 20.0,
        },
    ]

    class _Profiler:
        def export_chrome_trace(self, path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"traceEvents": events}, handle)

    audit, timeline = _kineto_node_attribution_audit(
        _Profiler(),
        trace_path=str(tmp_path / "trace.json"),
        timeline_anchor_origins={
            first_anchor: (1000.0, 2.0),
            second_anchor: (1100.0, 3.0),
        },
        expected_timeline_anchors=2,
    )

    occurrences = audit["COMPUTE:uidnode"]["range_activity_signatures"]
    assert occurrences[0]["activity_start_offset_us"] == pytest.approx(20.0)
    assert occurrences[0]["activity_timestamp_source"] == "correlated_gpu_activity"
    assert occurrences[0]["cpu_launch_sequence"] == ["kernel"]
    assert occurrences[0]["cpu_launch_count"] == 1
    assert occurrences[0]["matched_cpu_launch_count"] == 1
    assert occurrences[0]["launch_mapping_unambiguous"] is True
    assert occurrences[0]["cpu_launches"][0]["api_name"] == "cudaLaunchKernel"
    assert occurrences[0]["cpu_launches"][0]["activities"] == [
        {"category": "kernel", "name": "kernel"}
    ]
    assert occurrences[1]["activity_start_offset_us"] == pytest.approx(6.0)
    assert occurrences[1]["activity_timestamp_source"] == "cpu_range_zero_activity"
    assert timeline["timeline_origin_samples_us"] == (1000.0, 1100.0)
    assert timeline["timeline_origin_uncertainty_samples_us"] == (2.0, 3.0)


def test_kineto_audit_records_missing_activity_per_occurrence(tmp_path) -> None:
    anchors = [
        "piper_profile_timeline_anchor:normal:0:rank0",
        "piper_profile_timeline_anchor:normal:1:rank0",
    ]
    events = []
    for occurrence, anchor in enumerate(anchors):
        origin = 100.0 + occurrence * 100.0
        external_id = 10 + occurrence
        correlation = 20 + occurrence
        events.extend([
            {
                "ph": "X", "cat": "user_annotation", "name": anchor,
                "pid": 1, "tid": 2, "ts": origin, "dur": 50.0,
            },
            {
                "ph": "X", "cat": "user_annotation", "name": "COMPUTE:uidnode",
                "pid": 1, "tid": 2, "ts": origin + 2.0, "dur": 30.0,
            },
            {
                "ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
                "pid": 1, "tid": 2, "ts": origin + 5.0, "dur": 1.0,
                "args": {
                    "External id": external_id,
                    "correlation": correlation,
                },
            },
        ])
        if occurrence == 0:
            events.extend([
                {
                    "ph": "X", "cat": "kernel", "name": "kernel.a",
                    "pid": 9, "tid": 10, "ts": origin + 10.0, "dur": 2.0,
                    "args": {
                        "External id": external_id,
                        "correlation": correlation,
                    },
                },
                {
                    "ph": "X", "cat": "kernel", "name": "kernel.b",
                    "pid": 9, "tid": 10, "ts": origin + 13.0, "dur": 2.0,
                    "args": {
                        "External id": external_id,
                        "correlation": correlation,
                    },
                },
            ])

    class _Profiler:
        def export_chrome_trace(self, path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"traceEvents": events}, handle)

    audit, _timeline = _kineto_node_attribution_audit(
        _Profiler(),
        trace_path=str(tmp_path / "per-occurrence.json"),
        timeline_anchor_origins={anchor: (0.0, 0.0) for anchor in anchors},
        expected_timeline_anchors=2,
    )

    node_audit = audit["COMPUTE:uidnode"]
    assert node_audit["launch_count"] == 2
    assert node_audit["matched_launch_count"] == 1
    assert node_audit["unmatched_launch_count"] == 1
    first, second = node_audit["range_activity_signatures"]
    assert first["matched_cpu_launch_count"] == 1
    assert first["cpu_launches"][0]["activities"] == [
        {"category": "kernel", "name": "kernel.a"},
        {"category": "kernel", "name": "kernel.b"},
    ]
    assert second["matched_cpu_launch_count"] == 0
    assert second["cpu_launches"][0]["activities"] == []


def test_actor_clock_calibration_selects_minimum_round_trip() -> None:
    assert _select_actor_clock_calibration(
        [(100, 80, 120), (200, 177, 206)]
    ) == (26, 3)


def test_kineto_audit_rejects_missing_duplicate_and_misassigned_anchors(
    tmp_path,
) -> None:
    anchor = "piper_profile_timeline_anchor:normal:2:rank0"

    class _Profiler:
        def __init__(self, events):
            self.events = events

        def export_chrome_trace(self, path):
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"traceEvents": self.events}, handle)

    node = {
        "ph": "X", "cat": "user_annotation", "name": "COMPUTE:uidnode",
        "pid": 1, "tid": 2, "ts": 105.0, "dur": 10.0,
    }
    with pytest.raises(RuntimeError, match="contains no profiling timeline anchors"):
        _kineto_node_attribution_audit(
            _Profiler([node]),
            trace_path=str(tmp_path / "missing.json"),
            timeline_anchor_origins={anchor: (1000.0, 1.0)},
            expected_timeline_anchors=1,
        )

    anchor_event = {
        "ph": "X", "cat": "user_annotation", "name": anchor,
        "pid": 1, "tid": 2, "ts": 100.0, "dur": 20.0,
    }
    with pytest.raises(RuntimeError, match="duplicate Kineto timeline anchor"):
        _kineto_node_attribution_audit(
            _Profiler([anchor_event, dict(anchor_event)]),
            trace_path=str(tmp_path / "duplicate.json"),
            timeline_anchor_origins={anchor: (1000.0, 1.0)},
            expected_timeline_anchors=1,
        )

    outside = dict(node, ts=125.0)
    with pytest.raises(RuntimeError, match="exactly one timeline anchor"):
        _kineto_node_attribution_audit(
            _Profiler([anchor_event, outside]),
            trace_path=str(tmp_path / "outside.json"),
            timeline_anchor_origins={anchor: (1000.0, 1.0)},
            expected_timeline_anchors=1,
        )


def test_profile_run_dag_records_one_host_only_iteration_anchor(monkeypatch) -> None:
    actor_class = PiperActor.__ray_metadata__.modified_class
    actor = object.__new__(actor_class)
    actor.runtime = SimpleNamespace(
        pytorch_profiler_enabled=True,
        torch_profiler=SimpleNamespace(step=lambda: None),
        device="cuda:0",
        global_rank=3,
    )
    actor.profile_prepared_dag_executions = {"normal": None}
    actor._cupti_timeline_anchor_origins = {}
    actor._cupti_timeline_anchor_counter = 0
    actor._cupti_coordinator_clock_offset_ns = 400
    actor._cupti_clock_sync_uncertainty_ns = 50
    actor._execute_dag = lambda **_kwargs: None
    entered = []
    exited = []

    class _Anchor:
        def __init__(self, label):
            self.label = label

        def __enter__(self):
            entered.append(self.label)

        def __exit__(self, *_args):
            exited.append(self.label)

    monotonic_values = iter((1000, 1200))
    monkeypatch.setattr(actor_module.time, "monotonic_ns", lambda: next(monotonic_values))
    monkeypatch.setattr(actor_module.torch.profiler, "record_function", _Anchor)
    monkeypatch.setattr(actor_module.torch.cuda, "synchronize", lambda _device: None)

    actor.profile_run_dag(
        profile_mode="normal",
        capture_allocator_stats=False,
        advance_profiler=False,
    )

    label = "piper_profile_timeline_anchor:normal:0:rank3"
    assert entered == exited == [label]
    assert actor._cupti_timeline_anchor_origins[label] == pytest.approx((1.5, 0.15))


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
    synchronizations = []
    monkeypatch.setattr(
        actor_module.torch.cuda,
        "synchronize",
        lambda device: synchronizations.append(device),
    )

    actor.profile_run_dag(
        advance_profiler=False,
        capture_allocator_stats=False,
    )
    assert steps == []
    assert synchronizations == ["cuda:0"]

    actor.step_pytorch_profiler()
    assert steps == ["step"]
    assert synchronizations == ["cuda:0", "cuda:0"]


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


@pytest.mark.parametrize(
    ("profiler_kwargs", "expected_schedule"),
    [
        ({}, {"wait": 0, "warmup": 2, "active": 1, "repeat": 1}),
        (
            {"warmup_iters": 1, "active_iters": 3},
            {"wait": 0, "warmup": 1, "active": 3, "repeat": 1},
        ),
    ],
)
def test_pytorch_profiler_uses_warmup_then_active_and_steps_per_dag(
    monkeypatch, profiler_kwargs, expected_schedule
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
        device="cuda:0",
    )
    actor.logger = SimpleNamespace(debug=lambda *_args, **_kwargs: None)
    expected_result = object()
    actor._execute_dag = lambda **_kwargs: expected_result
    profiler_boundaries = []
    monkeypatch.setattr(
        actor_module.torch.cuda,
        "synchronize",
        lambda device: profiler_boundaries.append(device),
    )

    actor.start_pytorch_profiler(**profiler_kwargs)
    assert schedule_kwargs == expected_schedule
    iterations = expected_schedule["warmup"] + expected_schedule["active"]
    for _ in range(iterations):
        assert actor.run_dag() is expected_result
    assert profiler_steps == ["step"] * iterations
    assert profiler_boundaries == ["cuda:0"] * iterations
