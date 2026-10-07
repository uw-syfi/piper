# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Hopper physical-resource measurements for overlap diagnostics.

The supported cluster does not grant CUPTI/Nsight Compute performance-counter
permission.  Hopper NVML GPM is therefore the natural-concurrency backend for
SM, occupancy, Tensor Core, DRAM, PCIe, and aggregate NVLink interval metrics.
Exact cumulative NVLink and host InfiniBand counters supplement the rates.

NVML documents a greater-than-100 ms interval between GPM samples.  Callers
must build fixed-count repeated-operation trials long enough to satisfy that
constraint.  Values returned here are wall-interval averages and are never
duty-cycle corrected.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Iterable


GPM_MINIMUM_INTERVAL_US = 100_000
GPM_METRICS: dict[int, tuple[str, str]] = {
    2: ("sm_util_pct", "%"),
    3: ("sm_occupancy_pct", "%"),
    5: ("any_tensor_util_pct", "%"),
    7: ("hmma_tensor_util_pct", "%"),
    10: ("dram_bw_util_pct", "%"),
    20: ("pcie_tx_mib_per_s", "MiB/s"),
    21: ("pcie_rx_mib_per_s", "MiB/s"),
    60: ("nvlink_total_rx_mib_per_s", "MiB/s"),
    61: ("nvlink_total_tx_mib_per_s", "MiB/s"),
}

NVLINK_FIELD_IDS = {
    138: ("nvlink_data_tx", "KiB"),
    139: ("nvlink_data_rx", "KiB"),
}
NVML_SUCCESS = 0
NVML_GPM_METRIC_MAX = 98
NVML_GPM_METRICS_GET_VERSION = 1
NVML_GPM_SUPPORT_VERSION = 1


class PhysicalMetricError(RuntimeError):
    """Raised when a required physical measurement cannot be completed."""


class _NvmlValue(ctypes.Union):
    _fields_ = [
        ("dVal", ctypes.c_double),
        ("siVal", ctypes.c_int),
        ("uiVal", ctypes.c_uint),
        ("ulVal", ctypes.c_ulong),
        ("ullVal", ctypes.c_ulonglong),
        ("sllVal", ctypes.c_longlong),
        ("usVal", ctypes.c_ushort),
    ]


class _NvmlFieldValue(ctypes.Structure):
    _fields_ = [
        ("fieldId", ctypes.c_uint),
        ("scopeId", ctypes.c_uint),
        ("timestamp", ctypes.c_longlong),
        ("latencyUsec", ctypes.c_longlong),
        ("valueType", ctypes.c_int),
        ("nvmlReturn", ctypes.c_int),
        ("value", _NvmlValue),
    ]


class _NvmlGpmMetricInfo(ctypes.Structure):
    _fields_ = [
        ("shortName", ctypes.c_char_p),
        ("longName", ctypes.c_char_p),
        ("unit", ctypes.c_char_p),
    ]


class _NvmlGpmMetric(ctypes.Structure):
    _fields_ = [
        ("metricId", ctypes.c_uint),
        ("nvmlReturn", ctypes.c_int),
        ("value", ctypes.c_double),
        ("metricInfo", _NvmlGpmMetricInfo),
    ]


class _NvmlGpmMetricsGet(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("numMetrics", ctypes.c_uint),
        ("sample1", ctypes.c_void_p),
        ("sample2", ctypes.c_void_p),
        ("metrics", _NvmlGpmMetric * NVML_GPM_METRIC_MAX),
    ]


class _NvmlGpmSupport(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("isSupportedDevice", ctypes.c_uint),
    ]


_VALUE_READERS = {
    0: lambda value: value.dVal,
    1: lambda value: value.uiVal,
    2: lambda value: value.ulVal,
    3: lambda value: value.ullVal,
    4: lambda value: value.sllVal,
    5: lambda value: value.siVal,
    6: lambda value: value.usVal,
}


def _decode(value: bytes | None) -> str | None:
    return value.decode("utf-8") if value else None


def unsigned_counter_delta(
    before: int, after: int, *, bits: int = 64
) -> tuple[int, int]:
    """Return a wrap-aware unsigned counter delta and wrap count."""

    if bits <= 0:
        raise ValueError("counter width must be positive")
    modulus = 1 << bits
    if not 0 <= before < modulus or not 0 <= after < modulus:
        raise ValueError("counter values are outside the declared width")
    if after >= before:
        return after - before, 0
    return modulus - before + after, 1


def read_ib_port_data_counters() -> list[dict[str, Any]]:
    """Read host-wide InfiniBand port data counters.

    The standard ``port_{xmit,rcv}_data`` unit is a 32-bit word, so raw values
    are converted to bytes here.  The counter is host-wide, not process-local.
    """

    rows = []
    base = Path("/sys/class/infiniband")
    for hca in sorted(base.glob("*")):
        for port in sorted((hca / "ports").glob("*")):
            for direction, filename in (
                ("tx", "port_xmit_data"),
                ("rx", "port_rcv_data"),
            ):
                path = port / "counters" / filename
                if not path.exists():
                    continue
                raw_words = int(path.read_text(encoding="utf-8").strip())
                rows.append({
                    "hca": hca.name,
                    "port": int(port.name),
                    "direction": direction,
                    "counter_path": str(path),
                    "raw_4byte_words": raw_words,
                    "bytes": raw_words * 4,
                    "timestamp_ns": time.monotonic_ns(),
                })
    return rows


def ib_counter_deltas(
    before: Iterable[dict[str, Any]],
    after: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join two host IB snapshots and return wrap-aware byte deltas."""

    keys = ("hca", "port", "direction")
    before_by_key = {tuple(row[key] for key in keys): row for row in before}
    after_by_key = {tuple(row[key] for key in keys): row for row in after}
    rows = []
    for key in sorted(before_by_key.keys() & after_by_key.keys()):
        first = before_by_key[key]
        second = after_by_key[key]
        raw_delta, wraps = unsigned_counter_delta(
            int(first["raw_4byte_words"]),
            int(second["raw_4byte_words"]),
        )
        rows.append({
            **dict(zip(keys, key)),
            "counter_path": first["counter_path"],
            "start_timestamp_ns": int(first["timestamp_ns"]),
            "end_timestamp_ns": int(second["timestamp_ns"]),
            "delta_bytes": raw_delta * 4,
            "counter_wraps": wraps,
        })
    return rows


@dataclass
class GpmMeasurementStart:
    sample1: ctypes.c_void_p
    sample2: ctypes.c_void_p
    monotonic_ns: int
    nvlink_fields: list[dict[str, Any]]


class NvmlGpmCollector:
    """Bracket one GPU's natural-execution interval with Hopper GPM."""

    def __init__(self, gpu_uuid: str) -> None:
        self.gpu_uuid = (
            gpu_uuid if gpu_uuid.startswith("GPU-") else f"GPU-{gpu_uuid}"
        )
        self._library: Any = None
        self._handle = ctypes.c_void_p()

    def open(self) -> "NvmlGpmCollector":
        if self._library is not None:
            return self
        library = ctypes.CDLL("libnvidia-ml.so.1")
        library.nvmlInit_v2.restype = ctypes.c_int
        library.nvmlShutdown.restype = ctypes.c_int
        library.nvmlDeviceGetHandleByUUID.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        library.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
        library.nvmlDeviceGetFieldValues.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_NvmlFieldValue),
        ]
        library.nvmlDeviceGetFieldValues.restype = ctypes.c_int
        library.nvmlGpmQueryDeviceSupport.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_NvmlGpmSupport),
        ]
        library.nvmlGpmQueryDeviceSupport.restype = ctypes.c_int
        library.nvmlGpmSampleAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        library.nvmlGpmSampleAlloc.restype = ctypes.c_int
        library.nvmlGpmSampleGet.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        library.nvmlGpmSampleGet.restype = ctypes.c_int
        library.nvmlGpmMetricsGet.argtypes = [
            ctypes.POINTER(_NvmlGpmMetricsGet)
        ]
        library.nvmlGpmMetricsGet.restype = ctypes.c_int
        library.nvmlGpmSampleFree.argtypes = [ctypes.c_void_p]
        library.nvmlGpmSampleFree.restype = ctypes.c_int
        init_status = library.nvmlInit_v2()
        if init_status != NVML_SUCCESS:
            raise PhysicalMetricError(f"nvmlInit_v2 failed: {init_status}")
        handle_status = library.nvmlDeviceGetHandleByUUID(
            self.gpu_uuid.encode("ascii"), ctypes.byref(self._handle)
        )
        if handle_status != NVML_SUCCESS:
            library.nvmlShutdown()
            raise PhysicalMetricError(
                f"nvmlDeviceGetHandleByUUID failed: {handle_status}"
            )
        support = _NvmlGpmSupport()
        support.version = NVML_GPM_SUPPORT_VERSION
        support_status = library.nvmlGpmQueryDeviceSupport(
            self._handle, ctypes.byref(support)
        )
        if support_status != NVML_SUCCESS or not support.isSupportedDevice:
            library.nvmlShutdown()
            raise PhysicalMetricError(
                "NVML GPM is unavailable: "
                f"status={support_status}, supported={support.isSupportedDevice}"
            )
        self._library = library
        return self

    def close(self) -> None:
        if self._library is None:
            return
        status = self._library.nvmlShutdown()
        self._library = None
        if status != NVML_SUCCESS:
            raise PhysicalMetricError(f"nvmlShutdown failed: {status}")

    def __enter__(self) -> "NvmlGpmCollector":
        return self.open()

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def _require_open(self) -> Any:
        if self._library is None:
            raise PhysicalMetricError("NVML GPM collector is not open")
        return self._library

    def _nvlink_fields(self) -> list[dict[str, Any]]:
        library = self._require_open()
        requests = [
            (field_id, link)
            for field_id in NVLINK_FIELD_IDS
            for link in range(18)
        ]
        values = (_NvmlFieldValue * len(requests))()
        for index, (field_id, link) in enumerate(requests):
            values[index].fieldId = field_id
            values[index].scopeId = link
        status = library.nvmlDeviceGetFieldValues(
            self._handle, len(values), values
        )
        if status != NVML_SUCCESS:
            raise PhysicalMetricError(
                f"nvmlDeviceGetFieldValues failed: {status}"
            )
        rows = []
        for value in values:
            name, unit = NVLINK_FIELD_IDS[int(value.fieldId)]
            metric_value = None
            if value.nvmlReturn == NVML_SUCCESS:
                reader = _VALUE_READERS.get(int(value.valueType))
                if reader is None:
                    raise PhysicalMetricError(
                        f"unsupported NVML value type {value.valueType}"
                    )
                metric_value = int(reader(value.value))
            rows.append({
                "field_id": int(value.fieldId),
                "name": name,
                "unit": unit,
                "link": int(value.scopeId),
                "timestamp_us": int(value.timestamp),
                "latency_us": int(value.latencyUsec),
                "status": int(value.nvmlReturn),
                "value": metric_value,
            })
        return rows

    def start(self) -> GpmMeasurementStart:
        library = self._require_open()
        sample1 = ctypes.c_void_p()
        sample2 = ctypes.c_void_p()
        for sample in (sample1, sample2):
            status = library.nvmlGpmSampleAlloc(ctypes.byref(sample))
            if status != NVML_SUCCESS:
                if sample1.value:
                    library.nvmlGpmSampleFree(sample1)
                raise PhysicalMetricError(
                    f"nvmlGpmSampleAlloc failed: {status}"
                )
        nvlink = self._nvlink_fields()
        sample1_attempts = 0
        status = 10
        while status == 10 and sample1_attempts < 100:
            sample1_attempts += 1
            status = library.nvmlGpmSampleGet(self._handle, sample1)
            if status == 10:
                time.sleep(0.005)
        if status != NVML_SUCCESS:
            library.nvmlGpmSampleFree(sample1)
            library.nvmlGpmSampleFree(sample2)
            raise PhysicalMetricError(
                "first nvmlGpmSampleGet failed: "
                f"status={status}, attempts={sample1_attempts}"
            )
        return GpmMeasurementStart(
            sample1=sample1,
            sample2=sample2,
            monotonic_ns=time.monotonic_ns(),
            nvlink_fields=nvlink,
        )

    def finish(self, start: GpmMeasurementStart) -> dict[str, Any]:
        library = self._require_open()
        try:
            # NVML_ERROR_TIMEOUT (10) is transient under simultaneous 16-GPU
            # sampling. Retrying the counter read does not execute another
            # benchmark trial; retain the retry count in the result.
            sample2_attempts = 0
            status = 10
            while status == 10 and sample2_attempts < 100:
                sample2_attempts += 1
                status = library.nvmlGpmSampleGet(self._handle, start.sample2)
                if status == 10:
                    time.sleep(0.005)
            if status != NVML_SUCCESS:
                raise PhysicalMetricError(
                    "second nvmlGpmSampleGet failed: "
                    f"status={status}, attempts={sample2_attempts}"
                )
            end_monotonic_ns = time.monotonic_ns()
            end_fields = self._nvlink_fields()
            request = _NvmlGpmMetricsGet()
            request.version = NVML_GPM_METRICS_GET_VERSION
            request.numMetrics = len(GPM_METRICS)
            request.sample1 = start.sample1
            request.sample2 = start.sample2
            for index, metric_id in enumerate(GPM_METRICS):
                request.metrics[index].metricId = metric_id
            metrics_status = library.nvmlGpmMetricsGet(ctypes.byref(request))
            if metrics_status != NVML_SUCCESS:
                raise PhysicalMetricError(
                    f"nvmlGpmMetricsGet failed: {metrics_status}"
                )
            metrics = {}
            metric_details = []
            for item in request.metrics[:request.numMetrics]:
                metric_id = int(item.metricId)
                name, expected_unit = GPM_METRICS[metric_id]
                value = (
                    float(item.value)
                    if item.nvmlReturn == NVML_SUCCESS else None
                )
                metrics[name] = value
                metric_details.append({
                    "metric_id": metric_id,
                    "name": name,
                    "expected_unit": expected_unit,
                    "nvml_status": int(item.nvmlReturn),
                    "value": value,
                    "short_name": _decode(item.metricInfo.shortName),
                    "long_name": _decode(item.metricInfo.longName),
                    "unit": _decode(item.metricInfo.unit),
                })
            before = {
                (row["field_id"], row["link"]): row
                for row in start.nvlink_fields
            }
            after = {
                (row["field_id"], row["link"]): row for row in end_fields
            }
            nvlink_deltas = []
            for key in sorted(before.keys() & after.keys()):
                first = before[key]
                second = after[key]
                if first["value"] is None or second["value"] is None:
                    nvlink_deltas.append({
                        **second,
                        "delta_kib": None,
                        "delta_bytes": None,
                        "counter_wraps": 0,
                        "measurement_status": "unavailable",
                    })
                    continue
                delta_kib, wraps = unsigned_counter_delta(
                    int(first["value"]), int(second["value"])
                )
                nvlink_deltas.append({
                    **second,
                    "start_timestamp_us": first["timestamp_us"],
                    "delta_kib": delta_kib,
                    "delta_bytes": delta_kib * 1024,
                    "counter_wraps": wraps,
                    "measurement_status": "ok",
                })
            interval_us = (end_monotonic_ns - start.monotonic_ns) / 1000.0
            return {
                "gpu_uuid": self.gpu_uuid,
                "start_monotonic_ns": start.monotonic_ns,
                "end_monotonic_ns": end_monotonic_ns,
                "gpm_interval_us": interval_us,
                "gpm_minimum_interval_us": GPM_MINIMUM_INTERVAL_US,
                "gpm_interval_valid": interval_us > GPM_MINIMUM_INTERVAL_US,
                "gpm_sample2_attempts": sample2_attempts,
                "interval_average_semantics": (
                    "wall_interval_no_duty_correction"
                ),
                "metrics": metrics,
                "metric_details": metric_details,
                "nvlink_deltas": nvlink_deltas,
            }
        finally:
            for sample in (start.sample1, start.sample2):
                if sample.value:
                    library.nvmlGpmSampleFree(sample)


def physical_metric_metadata() -> dict[str, Any]:
    """Return the exact backend-to-standardized-field mapping."""

    return {
        "schema_version": 1,
        "backend": "nvml_gpm_plus_nvml_fields_plus_ib_sysfs",
        "gpm_minimum_interval_us_exclusive": GPM_MINIMUM_INTERVAL_US,
        "interval_average_semantics": "wall_interval_no_duty_correction",
        "gpm_metrics": {
            str(metric_id): {"field": field, "unit": unit}
            for metric_id, (field, unit) in GPM_METRICS.items()
        },
        "nvlink_cumulative_fields": {
            str(field_id): {"name": name, "unit": unit}
            for field_id, (name, unit) in NVLINK_FIELD_IDS.items()
        },
        "ib_sysfs": {
            "paths": (
                "/sys/class/infiniband/*/ports/*/counters/"
                "port_{xmit,rcv}_data"
            ),
            "raw_unit": "4-byte_words",
            "standardized_unit": "bytes",
            "scope": "host_wide",
        },
        "unavailable_metrics": {
            "l2_throughput_gbps": "ERR_NVGPUCTRPERM",
            "l2_pct_peak": "ERR_NVGPUCTRPERM",
            "l2_hit_rate_pct": "ERR_NVGPUCTRPERM",
            "dram_read_gbps": "not_exposed_by_nvml_gpm",
            "dram_write_gbps": "not_exposed_by_nvml_gpm",
            "active_warps": "not_exposed_as_absolute_count_by_nvml_gpm",
        },
    }
