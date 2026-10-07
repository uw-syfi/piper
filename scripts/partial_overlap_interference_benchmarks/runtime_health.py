# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Optional NVML runtime-health sampling for GPU benchmarks.

The benchmark suite must remain usable in environments where ``pynvml`` is
not installed.  This module therefore imports it lazily, and keeps collection
behind a small backend protocol that is straightforward to fake in unit tests.

Every requested metric is represented in :class:`DeviceHealthSample`, even
when the driver does not expose it.  ``metric_status`` distinguishes a real
``None`` from an unsupported query or a query that failed unexpectedly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import importlib
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
import math
import statistics
import time
from typing import Any, Iterable, Literal, Mapping, Protocol, Sequence


MetricStatus = Literal["ok", "unsupported", "error"]

SCALAR_METRICS = (
    "gpu_uuid",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "power_w",
    "temperature_c",
    "gpu_utilization_pct",
    "throttle_reason_mask",
    "total_memory_bytes",
    "used_memory_bytes",
    "free_memory_bytes",
    "ecc_volatile_corrected",
    "ecc_volatile_uncorrected",
    "ecc_aggregate_corrected",
    "ecc_aggregate_uncorrected",
    "pcie_replay_counter",
)

NUMERIC_METRICS = tuple(metric for metric in SCALAR_METRICS if metric != "gpu_uuid")


class UnsupportedMetric(RuntimeError):
    """Raised by a backend when a metric is unavailable on this device."""


class RuntimeHealthUnavailable(RuntimeError):
    """Raised when the optional runtime-health provider cannot be initialized."""


class RuntimeHealthBackend(Protocol):
    """Minimal backend contract used by :class:`RuntimeHealthCollector`."""

    def open(self) -> None:
        """Initialize the provider."""

    def close(self) -> None:
        """Release provider resources."""

    def begin_snapshot(self) -> None:
        """Start one logically consistent round of metric queries."""

    def device_indices(self) -> Sequence[int]:
        """Return provider-local device indices."""

    def read_metric(self, device_index: int, metric: str) -> Any:
        """Read one metric, raising :class:`UnsupportedMetric` if unavailable."""


@dataclass(frozen=True)
class DeviceHealthSample:
    """One best-effort health observation for one GPU."""

    timestamp_ns: int
    timestamp_utc: str
    gpu_index: int
    gpu_uuid: str | None = None
    sm_clock_mhz: int | None = None
    memory_clock_mhz: int | None = None
    power_w: float | None = None
    temperature_c: int | None = None
    gpu_utilization_pct: float | None = None
    throttle_reason_mask: int | None = None
    throttle_reasons: tuple[str, ...] | None = None
    total_memory_bytes: int | None = None
    used_memory_bytes: int | None = None
    free_memory_bytes: int | None = None
    ecc_volatile_corrected: int | None = None
    ecc_volatile_uncorrected: int | None = None
    ecc_aggregate_corrected: int | None = None
    ecc_aggregate_uncorrected: int | None = None
    pcie_replay_counter: int | None = None
    metric_status: Mapping[str, MetricStatus] = field(default_factory=dict)
    metric_errors: Mapping[str, str] = field(default_factory=dict)

    @property
    def unsupported_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, status in self.metric_status.items() if status == "unsupported")
        )

    @property
    def error_fields(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, status in self.metric_status.items() if status == "error")
        )

    @property
    def memory_utilization_pct(self) -> float | None:
        if self.used_memory_bytes is None or not self.total_memory_bytes:
            return None
        return 100.0 * self.used_memory_bytes / self.total_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["throttle_reasons"] = (
            list(self.throttle_reasons) if self.throttle_reasons is not None else None
        )
        result["metric_status"] = dict(self.metric_status)
        result["metric_errors"] = dict(self.metric_errors)
        result["unsupported_fields"] = list(self.unsupported_fields)
        result["error_fields"] = list(self.error_fields)
        result["memory_utilization_pct"] = self.memory_utilization_pct
        return result


@dataclass(frozen=True)
class HealthSnapshot:
    """A timestamped collection of all visible devices."""

    timestamp_ns: int
    timestamp_utc: str
    devices: tuple[DeviceHealthSample, ...]
    provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "timestamp_utc": self.timestamp_utc,
            "devices": [device.to_dict() for device in self.devices],
            "provider": self.provider,
        }


@dataclass(frozen=True)
class MetricAggregate:
    """Status counts and distribution summary for one numeric metric."""

    sample_count: int
    supported_count: int
    unsupported_count: int
    error_count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    p95: float | None
    last: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceHealthAggregate:
    """Health distributions for repeated observations of one GPU."""

    gpu_index: int
    gpu_uuid: str | None
    sample_count: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    metrics: Mapping[str, MetricAggregate]
    throttle_reason_counts: Mapping[str, int]
    throttle_reason_mask_or: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_index": self.gpu_index,
            "gpu_uuid": self.gpu_uuid,
            "sample_count": self.sample_count,
            "first_timestamp_ns": self.first_timestamp_ns,
            "last_timestamp_ns": self.last_timestamp_ns,
            "metrics": {
                name: aggregate.to_dict() for name, aggregate in self.metrics.items()
            },
            "throttle_reason_counts": dict(self.throttle_reason_counts),
            "throttle_reason_mask_or": self.throttle_reason_mask_or,
        }


def _utc_timestamp(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1e9, timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class RuntimeHealthCollector:
    """Collect health snapshots with an injectable backend.

    Reusing a collector as a context manager avoids repeatedly initializing
    NVML when a benchmark samples before and after each trial.
    """

    def __init__(
        self,
        backend: RuntimeHealthBackend | None = None,
        *,
        strict: bool = False,
    ) -> None:
        self._backend = backend if backend is not None else AutoRuntimeHealthBackend()
        self._strict = strict
        self._opened = False

    def open(self) -> "RuntimeHealthCollector":
        if not self._opened:
            self._backend.open()
            self._opened = True
        return self

    def close(self) -> None:
        if self._opened:
            self._backend.close()
            self._opened = False

    def __enter__(self) -> "RuntimeHealthCollector":
        return self.open()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def collect(
        self,
        *,
        timestamp_ns: int | None = None,
        device_indices: Sequence[int] | None = None,
    ) -> HealthSnapshot:
        if not self._opened:
            raise RuntimeError("RuntimeHealthCollector must be opened before collection")
        observed_ns = time.time_ns() if timestamp_ns is None else int(timestamp_ns)
        observed_utc = _utc_timestamp(observed_ns)
        self._backend.begin_snapshot()
        available_indices = tuple(int(index) for index in self._backend.device_indices())
        selected_indices = (
            available_indices
            if device_indices is None
            else tuple(int(index) for index in device_indices)
        )
        unknown = sorted(set(selected_indices) - set(available_indices))
        if unknown:
            raise ValueError(
                f"requested GPU indices {unknown} are not visible; "
                f"available indices are {list(available_indices)}"
            )
        devices = tuple(
            self._collect_device(index, observed_ns, observed_utc)
            for index in selected_indices
        )
        return HealthSnapshot(
            observed_ns,
            observed_utc,
            devices,
            provider=str(getattr(self._backend, "provider_name", type(self._backend).__name__)),
        )

    def _collect_device(
        self,
        device_index: int,
        timestamp_ns: int,
        timestamp_utc: str,
    ) -> DeviceHealthSample:
        values: dict[str, Any] = {}
        statuses: dict[str, MetricStatus] = {}
        errors: dict[str, str] = {}
        for metric in SCALAR_METRICS:
            try:
                values[metric] = self._backend.read_metric(device_index, metric)
                statuses[metric] = "ok"
            except UnsupportedMetric as error:
                values[metric] = None
                statuses[metric] = "unsupported"
                if str(error):
                    errors[metric] = str(error)
            except Exception as error:
                if self._strict:
                    raise
                values[metric] = None
                statuses[metric] = "error"
                errors[metric] = f"{type(error).__name__}: {error}"

        throttle_reasons: tuple[str, ...] | None = None
        try:
            throttle_reasons = tuple(
                self._backend.read_metric(device_index, "throttle_reasons")
            )
            statuses["throttle_reasons"] = "ok"
        except UnsupportedMetric as error:
            statuses["throttle_reasons"] = "unsupported"
            if str(error):
                errors["throttle_reasons"] = str(error)
        except Exception as error:
            if self._strict:
                raise
            statuses["throttle_reasons"] = "error"
            errors["throttle_reasons"] = f"{type(error).__name__}: {error}"

        return DeviceHealthSample(
            timestamp_ns=timestamp_ns,
            timestamp_utc=timestamp_utc,
            gpu_index=int(device_index),
            throttle_reasons=throttle_reasons,
            metric_status=statuses,
            metric_errors=errors,
            **values,
        )


def collect_runtime_health(
    backend: RuntimeHealthBackend | None = None,
    *,
    timestamp_ns: int | None = None,
    device_indices: Sequence[int] | None = None,
    strict: bool = False,
) -> HealthSnapshot:
    """Collect one snapshot while owning the backend lifecycle."""

    with RuntimeHealthCollector(backend, strict=strict) as collector:
        return collector.collect(
            timestamp_ns=timestamp_ns,
            device_indices=device_indices,
        )


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def aggregate_metric(
    samples: Sequence[DeviceHealthSample], metric: str
) -> MetricAggregate:
    """Aggregate one numeric metric while retaining availability counts."""

    if metric not in NUMERIC_METRICS:
        raise ValueError(f"not an aggregatable numeric metric: {metric}")
    values = [
        float(value)
        for sample in samples
        if sample.metric_status.get(metric) == "ok"
        and (value := getattr(sample, metric)) is not None
    ]
    statuses = Counter(sample.metric_status.get(metric, "error") for sample in samples)
    return MetricAggregate(
        sample_count=len(samples),
        supported_count=len(values),
        unsupported_count=statuses["unsupported"],
        error_count=statuses["error"] + statuses[None],
        minimum=min(values) if values else None,
        maximum=max(values) if values else None,
        mean=statistics.fmean(values) if values else None,
        median=statistics.median(values) if values else None,
        p95=_percentile(values, 0.95) if values else None,
        last=values[-1] if values else None,
    )


def aggregate_device_samples(
    samples: Iterable[DeviceHealthSample],
) -> DeviceHealthAggregate:
    """Aggregate repeated samples for exactly one physical/logical GPU."""

    materialized = sorted(samples, key=lambda sample: sample.timestamp_ns)
    if not materialized:
        raise ValueError("at least one device-health sample is required")
    indices = {sample.gpu_index for sample in materialized}
    uuids = {sample.gpu_uuid for sample in materialized if sample.gpu_uuid is not None}
    if len(indices) != 1 or len(uuids) > 1:
        raise ValueError("device-health samples must describe exactly one GPU")
    reason_counts: Counter[str] = Counter()
    masks = []
    for sample in materialized:
        reason_counts.update(sample.throttle_reasons or ())
        if sample.throttle_reason_mask is not None:
            masks.append(sample.throttle_reason_mask)
    combined_mask = None
    if masks:
        combined_mask = 0
        for mask in masks:
            combined_mask |= mask
    return DeviceHealthAggregate(
        gpu_index=materialized[0].gpu_index,
        gpu_uuid=next(iter(uuids), None),
        sample_count=len(materialized),
        first_timestamp_ns=materialized[0].timestamp_ns,
        last_timestamp_ns=materialized[-1].timestamp_ns,
        metrics={
            metric: aggregate_metric(materialized, metric) for metric in NUMERIC_METRICS
        },
        throttle_reason_counts=dict(sorted(reason_counts.items())),
        throttle_reason_mask_or=combined_mask,
    )


def aggregate_snapshots(
    snapshots: Iterable[HealthSnapshot],
) -> dict[str, DeviceHealthAggregate]:
    """Group snapshots by UUID (falling back to local index) and aggregate."""

    grouped: defaultdict[str, list[DeviceHealthSample]] = defaultdict(list)
    for snapshot in snapshots:
        for sample in snapshot.devices:
            key = sample.gpu_uuid or f"index:{sample.gpu_index}"
            grouped[key].append(sample)
    return {
        key: aggregate_device_samples(samples) for key, samples in sorted(grouped.items())
    }


class NvmlBackend:
    """Lazy ``pynvml`` adapter implementing :class:`RuntimeHealthBackend`."""

    provider_name = "pynvml"

    _THROTTLE_REASON_CONSTANTS = (
        ("nvmlClocksThrottleReasonGpuIdle", "gpu_idle"),
        ("nvmlClocksThrottleReasonApplicationsClocksSetting", "applications_clocks"),
        ("nvmlClocksThrottleReasonSwPowerCap", "software_power_cap"),
        ("nvmlClocksThrottleReasonHwSlowdown", "hardware_slowdown"),
        ("nvmlClocksThrottleReasonSyncBoost", "sync_boost"),
        ("nvmlClocksThrottleReasonSwThermalSlowdown", "software_thermal_slowdown"),
        ("nvmlClocksThrottleReasonHwThermalSlowdown", "hardware_thermal_slowdown"),
        ("nvmlClocksThrottleReasonHwPowerBrakeSlowdown", "hardware_power_brake"),
        ("nvmlClocksThrottleReasonDisplayClockSetting", "display_clock_setting"),
    )

    def __init__(self, module: Any | None = None) -> None:
        self._module = module
        self._opened = False
        self._handles: dict[int, Any] = {}
        self._memory_cache: dict[int, Any] = {}
        self._throttle_cache: dict[int, int] = {}

    def open(self) -> None:
        if self._opened:
            return
        if self._module is None:
            try:
                self._module = importlib.import_module("pynvml")
            except ImportError as error:
                raise RuntimeHealthUnavailable(
                    "runtime health requires the optional pynvml package"
                ) from error
        try:
            self._module.nvmlInit()
        except Exception as error:
            raise RuntimeHealthUnavailable(f"NVML initialization failed: {error}") from error
        self._opened = True

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self._module.nvmlShutdown()
        finally:
            self._opened = False
            self._handles.clear()
            self._memory_cache.clear()
            self._throttle_cache.clear()

    def begin_snapshot(self) -> None:
        self._memory_cache.clear()
        self._throttle_cache.clear()

    def device_indices(self) -> Sequence[int]:
        self._require_open()
        return tuple(range(int(self._module.nvmlDeviceGetCount())))

    def _require_open(self) -> None:
        if not self._opened:
            raise RuntimeError("NVML backend is not initialized")

    def _handle(self, device_index: int) -> Any:
        self._require_open()
        if device_index not in self._handles:
            self._handles[device_index] = self._module.nvmlDeviceGetHandleByIndex(
                device_index
            )
        return self._handles[device_index]

    def _not_supported_classes(self) -> tuple[type[BaseException], ...]:
        classes = []
        for name in ("NVMLError_NotSupported", "NVMLError_FunctionNotFound"):
            candidate = getattr(self._module, name, None)
            if isinstance(candidate, type) and issubclass(candidate, BaseException):
                classes.append(candidate)
        return tuple(classes)

    def _query(self, function_name: str, *args: Any) -> Any:
        function = getattr(self._module, function_name, None)
        if function is None:
            raise UnsupportedMetric(f"pynvml has no {function_name}")
        try:
            return function(*args)
        except Exception as error:
            unsupported = self._not_supported_classes()
            if unsupported and isinstance(error, unsupported):
                raise UnsupportedMetric(str(error) or function_name) from error
            raise

    def _memory(self, device_index: int) -> Any:
        if device_index not in self._memory_cache:
            self._memory_cache[device_index] = self._query(
                "nvmlDeviceGetMemoryInfo", self._handle(device_index)
            )
        return self._memory_cache[device_index]

    def _throttle_mask(self, device_index: int) -> int:
        if device_index not in self._throttle_cache:
            self._throttle_cache[device_index] = int(
                self._query(
                    "nvmlDeviceGetCurrentClocksThrottleReasons",
                    self._handle(device_index),
                )
            )
        return self._throttle_cache[device_index]

    def _decode_throttle_reasons(self, mask: int) -> tuple[str, ...]:
        if mask == 0:
            return ()
        reasons = []
        known_mask = 0
        for constant_name, label in self._THROTTLE_REASON_CONSTANTS:
            bit = int(getattr(self._module, constant_name, 0))
            known_mask |= bit
            if bit and mask & bit:
                reasons.append(label)
        unknown = mask & ~known_mask
        if unknown:
            reasons.append(f"unknown_0x{unknown:x}")
        return tuple(reasons)

    def _total_ecc_errors(
        self, device_index: int, error_kind: str, counter_scope: str
    ) -> int:
        kind_constant = {
            "corrected": "NVML_MEMORY_ERROR_TYPE_CORRECTED",
            "uncorrected": "NVML_MEMORY_ERROR_TYPE_UNCORRECTED",
        }[error_kind]
        scope_constant = {
            "volatile": "NVML_VOLATILE_ECC",
            "aggregate": "NVML_AGGREGATE_ECC",
        }[counter_scope]
        try:
            kind = getattr(self._module, kind_constant)
            scope = getattr(self._module, scope_constant)
        except AttributeError as error:
            raise UnsupportedMetric(f"pynvml lacks ECC constant: {error}") from error
        return int(
            self._query(
                "nvmlDeviceGetTotalEccErrors",
                self._handle(device_index),
                kind,
                scope,
            )
        )

    def read_metric(self, device_index: int, metric: str) -> Any:
        handle = self._handle(device_index)
        module = self._module
        if metric == "gpu_uuid":
            value = self._query("nvmlDeviceGetUUID", handle)
            return value.decode() if isinstance(value, bytes) else str(value)
        if metric == "sm_clock_mhz":
            return int(
                self._query("nvmlDeviceGetClockInfo", handle, module.NVML_CLOCK_SM)
            )
        if metric == "memory_clock_mhz":
            return int(
                self._query("nvmlDeviceGetClockInfo", handle, module.NVML_CLOCK_MEM)
            )
        if metric == "power_w":
            return float(self._query("nvmlDeviceGetPowerUsage", handle)) / 1000.0
        if metric == "temperature_c":
            return int(
                self._query(
                    "nvmlDeviceGetTemperature", handle, module.NVML_TEMPERATURE_GPU
                )
            )
        if metric == "gpu_utilization_pct":
            return float(
                self._query("nvmlDeviceGetUtilizationRates", handle).gpu
            )
        if metric == "throttle_reason_mask":
            return self._throttle_mask(device_index)
        if metric == "throttle_reasons":
            return self._decode_throttle_reasons(self._throttle_mask(device_index))
        if metric == "total_memory_bytes":
            return int(self._memory(device_index).total)
        if metric == "used_memory_bytes":
            return int(self._memory(device_index).used)
        if metric == "free_memory_bytes":
            return int(self._memory(device_index).free)
        if metric.startswith("ecc_"):
            _, scope, kind = metric.split("_", 2)
            return self._total_ecc_errors(device_index, kind, scope)
        if metric == "pcie_replay_counter":
            return int(self._query("nvmlDeviceGetPcieReplayCounter", handle))
        raise UnsupportedMetric(f"unknown runtime-health metric {metric!r}")


class NvidiaSmiBackend:
    """Package-free health backend using one cached XML query per snapshot."""

    provider_name = "nvidia-smi"
    _THROTTLE_REASON_BITS = (
        ("gpu_idle", "gpu_idle", 0x1),
        ("applications_clocks_setting", "applications_clocks", 0x2),
        ("sw_power_cap", "software_power_cap", 0x4),
        ("hw_slowdown", "hardware_slowdown", 0x8),
        ("sync_boost", "sync_boost", 0x10),
        ("sw_thermal_slowdown", "software_thermal_slowdown", 0x20),
        ("hw_thermal_slowdown", "hardware_thermal_slowdown", 0x40),
        ("hw_power_brake_slowdown", "hardware_power_brake", 0x80),
        ("display_clock", "display_clock_setting", 0x100),
    )

    def __init__(
        self,
        executable: str | None = None,
        *,
        timeout_s: float = 5.0,
        runner: Any | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("nvidia-smi timeout must be positive")
        self._executable = executable
        self._timeout_s = float(timeout_s)
        self._runner = runner if runner is not None else subprocess.run
        self._opened = False
        self._metrics: dict[int, dict[str, Any]] = {}

    def open(self) -> None:
        if self._opened:
            return
        if self._executable is None:
            self._executable = shutil.which("nvidia-smi")
        if not self._executable:
            raise RuntimeHealthUnavailable("nvidia-smi executable was not found")
        self._opened = True

    def close(self) -> None:
        self._opened = False
        self._metrics.clear()

    def _run_xml_query(self) -> str:
        if not self._opened or not self._executable:
            raise RuntimeError("nvidia-smi backend is not initialized")
        environment = dict(os.environ)
        environment["LC_ALL"] = "C"
        try:
            result = self._runner(
                [self._executable, "-q", "-x"],
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeHealthUnavailable(
                f"nvidia-smi timed out after {self._timeout_s:g}s"
            ) from error
        except (FileNotFoundError, OSError) as error:
            raise RuntimeHealthUnavailable(
                f"could not execute nvidia-smi: {error}"
            ) from error
        if int(result.returncode) != 0:
            detail = (result.stderr or result.stdout or "").strip()
            if len(detail) > 300:
                detail = detail[:300] + "..."
            raise RuntimeHealthUnavailable(
                f"nvidia-smi exited with status {result.returncode}: {detail}"
            )
        if not result.stdout.strip():
            raise RuntimeHealthUnavailable("nvidia-smi returned empty XML output")
        return result.stdout

    @staticmethod
    def _text(gpu: Any, paths: Sequence[str], metric: str) -> str:
        for path in paths:
            value = gpu.findtext(path)
            if value is None:
                continue
            stripped = value.strip()
            if stripped and stripped.upper() not in {"N/A", "NA", "[N/A]"}:
                return stripped
        raise UnsupportedMetric(f"nvidia-smi XML does not expose {metric}")

    @staticmethod
    def _number(text: str, metric: str) -> float:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
        if match is None:
            raise UnsupportedMetric(
                f"nvidia-smi returned an unparseable {metric}: {text!r}"
            )
        value = float(match.group(0))
        if not math.isfinite(value):
            raise UnsupportedMetric(
                f"nvidia-smi returned non-finite {metric}: {text!r}"
            )
        return value

    @classmethod
    def _numeric(
        cls, gpu: Any, paths: Sequence[str], metric: str, *, integer: bool = False
    ) -> int | float:
        value = cls._number(cls._text(gpu, paths, metric), metric)
        return int(value) if integer else value

    @classmethod
    def _memory_bytes(cls, gpu: Any, paths: Sequence[str], metric: str) -> int:
        text = cls._text(gpu, paths, metric)
        value = cls._number(text, metric)
        unit_match = re.search(r"\b([KMGT]?i?B)\b", text, re.IGNORECASE)
        if unit_match is None:
            raise UnsupportedMetric(
                f"nvidia-smi returned {metric} without a memory unit: {text!r}"
            )
        unit = unit_match.group(1).upper()
        multipliers = {
            "B": 1,
            "KB": 1_000,
            "KIB": 1 << 10,
            "MB": 1_000_000,
            "MIB": 1 << 20,
            "GB": 1_000_000_000,
            "GIB": 1 << 30,
            "TB": 1_000_000_000_000,
            "TIB": 1 << 40,
        }
        return int(round(value * multipliers[unit]))

    @classmethod
    def _throttle(cls, gpu: Any) -> tuple[int, tuple[str, ...]]:
        container = gpu.find("clocks_event_reasons")
        if container is None:
            container = gpu.find("clocks_throttle_reasons")
        if container is None:
            raise UnsupportedMetric(
                "nvidia-smi XML does not expose clock throttle reasons"
            )
        active_tags = {
            child.tag.rsplit("}", 1)[-1].lower()
            for child in container
            if (child.text or "").strip().lower() in {
                "active", "yes", "true", "enabled"
            }
        }
        known_tags = {
            child.tag.rsplit("}", 1)[-1].lower()
            for child in container
            if (child.text or "").strip().upper() not in {"", "N/A", "NA", "[N/A]"}
        }
        if not known_tags:
            raise UnsupportedMetric(
                "nvidia-smi returned no usable clock throttle reasons"
            )
        mask = 0
        reasons = []
        matched_active = set()
        for fragment, label, bit in cls._THROTTLE_REASON_BITS:
            matches = {tag for tag in active_tags if fragment in tag}
            if matches:
                mask |= bit
                reasons.append(label)
                matched_active.update(matches)
        reasons.extend(
            f"unknown_{tag}" for tag in sorted(active_tags - matched_active)
        )
        return mask, tuple(reasons)

    @classmethod
    def _ecc_count(
        cls, gpu: Any, scope: str, *, corrected: bool
    ) -> int:
        kind_paths = (
            ("single_bit/total", "corrected/total")
            if corrected
            else ("double_bit/total", "uncorrected/total")
        )
        base = f"ecc_errors/{scope}"
        for suffix in kind_paths:
            text = gpu.findtext(f"{base}/{suffix}")
            if text and text.strip().upper() not in {"N/A", "NA", "[N/A]"}:
                return int(cls._number(text, f"ECC {scope}"))
        scope_node = gpu.find(base)
        if scope_node is None:
            raise UnsupportedMetric(
                f"nvidia-smi XML does not expose {scope} ECC counters"
            )
        values = []
        for leaf in scope_node.iter():
            if list(leaf):
                continue
            tag = leaf.tag.rsplit("}", 1)[-1].lower()
            text = (leaf.text or "").strip()
            if not text or text.upper() in {"N/A", "NA", "[N/A]"}:
                continue
            is_uncorrected = "uncorrect" in tag or "double_bit" in tag
            is_corrected = (
                ("correct" in tag and "uncorrect" not in tag)
                or "single_bit" in tag
            )
            if (corrected and is_corrected) or (
                not corrected and is_uncorrected
            ):
                values.append(int(cls._number(text, f"ECC {scope} {tag}")))
        if not values:
            raise UnsupportedMetric(
                f"nvidia-smi XML has no usable {scope} ECC counter"
            )
        return sum(values)

    @classmethod
    def _pcie_replay_counter(cls, gpu: Any) -> int:
        for path in ("pci/replay_counter", "pci/pcie_replay_counter"):
            text = gpu.findtext(path)
            if text and text.strip().upper() not in {"N/A", "NA", "[N/A]"}:
                return int(cls._number(text, "PCIe replay counter"))
        for element in gpu.iter():
            tag = element.tag.rsplit("}", 1)[-1].lower()
            if "replay" not in tag or "count" not in tag:
                continue
            text = (element.text or "").strip()
            if text and text.upper() not in {"N/A", "NA", "[N/A]"}:
                return int(cls._number(text, "PCIe replay counter"))
        raise UnsupportedMetric(
            "nvidia-smi XML does not expose a PCIe replay counter"
        )

    @classmethod
    def _parse_gpu(cls, gpu: Any, index: int) -> dict[str, Any]:
        def query(function: Any) -> Any:
            try:
                return function()
            except UnsupportedMetric as error:
                return error

        throttle = query(lambda: cls._throttle(gpu))
        return {
            "gpu_uuid": query(
                lambda: cls._text(gpu, ("uuid",), "GPU UUID")
            ),
            "sm_clock_mhz": query(
                lambda: cls._numeric(
                    gpu, ("clocks/sm_clock",), "SM clock", integer=True
                )
            ),
            "memory_clock_mhz": query(
                lambda: cls._numeric(
                    gpu, ("clocks/mem_clock",), "memory clock", integer=True
                )
            ),
            "power_w": query(
                lambda: cls._numeric(
                    gpu,
                    (
                        "gpu_power_readings/power_draw",
                        "power_readings/power_draw",
                    ),
                    "power draw",
                )
            ),
            "temperature_c": query(
                lambda: cls._numeric(
                    gpu,
                    ("temperature/gpu_temp",),
                    "GPU temperature",
                    integer=True,
                )
            ),
            "gpu_utilization_pct": query(
                lambda: cls._numeric(
                    gpu, ("utilization/gpu_util",), "GPU utilization"
                )
            ),
            "throttle_reason_mask": (
                throttle[0] if isinstance(throttle, tuple) else throttle
            ),
            "throttle_reasons": (
                throttle[1] if isinstance(throttle, tuple) else throttle
            ),
            "total_memory_bytes": query(
                lambda: cls._memory_bytes(
                    gpu, ("fb_memory_usage/total",), "total memory"
                )
            ),
            "used_memory_bytes": query(
                lambda: cls._memory_bytes(
                    gpu, ("fb_memory_usage/used",), "used memory"
                )
            ),
            "free_memory_bytes": query(
                lambda: cls._memory_bytes(
                    gpu, ("fb_memory_usage/free",), "free memory"
                )
            ),
            "ecc_volatile_corrected": query(
                lambda: cls._ecc_count(gpu, "volatile", corrected=True)
            ),
            "ecc_volatile_uncorrected": query(
                lambda: cls._ecc_count(gpu, "volatile", corrected=False)
            ),
            "ecc_aggregate_corrected": query(
                lambda: cls._ecc_count(gpu, "aggregate", corrected=True)
            ),
            "ecc_aggregate_uncorrected": query(
                lambda: cls._ecc_count(gpu, "aggregate", corrected=False)
            ),
            "pcie_replay_counter": query(
                lambda: cls._pcie_replay_counter(gpu)
            ),
        }

    def begin_snapshot(self) -> None:
        xml = self._run_xml_query()
        try:
            root = ElementTree.fromstring(xml)
        except ElementTree.ParseError as error:
            raise RuntimeHealthUnavailable(
                f"nvidia-smi returned malformed XML: {error}"
            ) from error
        gpu_elements = root.findall(".//gpu")
        if not gpu_elements:
            raise RuntimeHealthUnavailable("nvidia-smi XML contains no GPUs")
        self._metrics = {
            index: self._parse_gpu(gpu, index)
            for index, gpu in enumerate(gpu_elements)
        }

    def device_indices(self) -> Sequence[int]:
        if not self._opened:
            raise RuntimeError("nvidia-smi backend is not initialized")
        if not self._metrics:
            raise RuntimeError("begin_snapshot must run before device_indices")
        return tuple(sorted(self._metrics))

    def read_metric(self, device_index: int, metric: str) -> Any:
        if device_index not in self._metrics:
            raise ValueError(f"nvidia-smi has no GPU index {device_index}")
        if metric not in self._metrics[device_index]:
            raise UnsupportedMetric(
                f"unknown nvidia-smi runtime-health metric {metric!r}"
            )
        value = self._metrics[device_index][metric]
        if isinstance(value, UnsupportedMetric):
            raise UnsupportedMetric(str(value))
        return value


class AutoRuntimeHealthBackend:
    """Select pynvml when available, otherwise package-free nvidia-smi."""

    def __init__(
        self,
        backends: Sequence[RuntimeHealthBackend] | None = None,
    ) -> None:
        self._backends = tuple(
            backends
            if backends is not None
            else (NvmlBackend(), NvidiaSmiBackend())
        )
        self._selected: RuntimeHealthBackend | None = None
        self._initialization_errors: list[str] = []

    @property
    def provider_name(self) -> str:
        if self._selected is None:
            return "uninitialized"
        return str(
            getattr(
                self._selected,
                "provider_name",
                type(self._selected).__name__,
            )
        )

    def open(self) -> None:
        if self._selected is not None:
            return
        self._initialization_errors.clear()
        for backend in self._backends:
            try:
                backend.open()
                self._selected = backend
                return
            except RuntimeHealthUnavailable as error:
                self._initialization_errors.append(
                    f"{type(backend).__name__}: {error}"
                )
                try:
                    backend.close()
                except Exception:
                    pass
        detail = "; ".join(self._initialization_errors)
        raise RuntimeHealthUnavailable(
            f"no runtime-health backend is available: {detail}"
        )

    def close(self) -> None:
        if self._selected is not None:
            self._selected.close()
            self._selected = None

    def _backend(self) -> RuntimeHealthBackend:
        if self._selected is None:
            raise RuntimeError("runtime-health backend is not initialized")
        return self._selected

    def begin_snapshot(self) -> None:
        self._backend().begin_snapshot()

    def device_indices(self) -> Sequence[int]:
        return self._backend().device_indices()

    def read_metric(self, device_index: int, metric: str) -> Any:
        return self._backend().read_metric(device_index, metric)
